# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""The engine-facing protocol of the DeepSeek-V4.1 serving layer: what the batcher (``batch.py``), the session store
(``sessions.py`` / ``sessdisk.py``) and the app need from the forward, as typed interfaces. The forward (M1:
``forward.py`` / ``engine.py``) implements ``Forward``; the serving layer is tested against a fake forward that keeps
the same rules (``tests/serving/fake_forward.py``).

Positions and rows:

- a slot's **position** ``pos`` = the tokens whose state is written (pool rows, rings, carries, lookback, taps);
- a prompt of n tokens is **prefilled** over [0, n - 1) in pieces (``Piece``); its last token is the first verify
  window's pending row, so every logit the server samples comes from ``window`` (prefill rows == decode rows: one
  sampling path, one grammar path, M1 gate 3);
- pieces start on the 16-token grid (``GRID``); a piece ends on the grid or at n - 1; one piece ends exactly at the
  prompt's replay point S = ``sessions.snapshot_point(n)`` (GLM 0540) so the slot can be snapshotted there;
- ``finish_prompt`` runs once a prompt's pieces are done, before its first window: in ``replay`` mode (CED,
  ``TF_DSV41_PREFILL=replay``) the decoder over the prompt's last rows (``tail``); in ``full`` mode a no-op;
- ``window`` writes each window's rows at [start, start + len(tokens)), every row computed as the serial step at
  its position (row-invariant kernels); ``commit`` keeps the pending row + ``accepted`` drafts, so pos = start +
  accepted + 1. Rejected rows' writes lie past pos and are overwritten before anything reads them.

Pages: the batcher owns the slot's page table (``pool.Slot``: GLM 0290's ``SlotPages`` with shared pages). Before any
call that writes positions < end it has mapped them (``Slot.ensure``); the forward reads ``pool.view(slot, family)``
(a GLM ``Paged`` per family) or the device page table ``Slot.table`` + ``Pool.phys``.

Ranks (TP=2): rank 0 plans every round and shares it (``Link``); both ranks make the same calls with the same
arguments in the same order. ``window`` returns the candidates merged over both ranks' vocabulary halves (one
exchange, GLM ``decode.sample_rows``), so both ranks see the same ``Candidates``; only rank 0 samples.

Snapshots: ``snapshot`` returns the slot's bounded state (``state.Bounded``) at its position; the pool pages below
the position are the store's business (``sessions.Store`` keeps them mapped, ``sessdisk`` parks them). ``restore``
loads a bounded state into a slot whose pages the store has mapped already.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

GRID = 16                    # snapshot / piece alignment in tokens (ratio 2 x candidate block 8)
ROW_BYTES, VB, SB = 584, 576, 8          # csa2.rows' FP8 KV record: 448 e4m3 + 64 bf16 RoPE dims, 8 scale bytes
                                         # (sizes only: the host modules need no Triton; tests check they agree)
REPLAY = 128                 # CED: the decoder's rows over a prompt's end (replay mode)
MODES = ("full", "replay")


@dataclass(frozen=True)
class Piece:
    """Prompt tokens ``ids`` of ``slot`` at positions [start, start + len(ids)) (prefill, no logits)."""

    slot: int
    start: int
    ids: tuple[int, ...]

    @property
    def end(self) -> int:
        return self.start + len(self.ids)


@dataclass(frozen=True)
class Rows:
    """One slot's verify window: ``tokens[0]`` the pending token at ``start``, then the drafts (a chain)."""

    slot: int
    start: int
    tokens: tuple[int, ...]


@dataclass
class Candidates:
    """The top ``count`` columns of each verify row, merged over the ranks: ``ids`` [R, C] int64, ``values`` [R, C]
    float32 logits, rows in window order (window 0's rows first). Masked columns (structured output) never appear
    unless a row allows fewer than C tokens; then the rest are -inf (sampling skips them)."""

    ids: np.ndarray
    values: np.ndarray

    def rows(self, a: int, b: int) -> Candidates:
        return Candidates(self.ids[a:b], self.values[a:b])


@dataclass
class Bounded:
    """A slot's bounded state at ``pos`` (``state.py``): the SWA rings' last rows, the compressor carries when pos is
    odd, the Engram lookback, the DSpark taps. ``arrays`` are host numpy arrays (the forward copies them; ~10 MB),
    ``meta`` small ints / strings (the CED mode it was taken in, ring layout)."""

    pos: int
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.arrays.values()))


@runtime_checkable
class Forward(Protocol):
    """What the serving layer calls on each rank (M1 implements it in ``forward.py``)."""

    vocab: int                    # the full vocabulary (129,280)

    def reset(self, slot: int) -> None:
        """A new request in ``slot``: position 0, rings / carries / lookback / taps cleared (pages: the batcher)."""

    def prefill(self, pieces: Sequence[Piece], *, mode: str) -> None:
        """One forward over several slots' pieces (GLM 0560: the experts' pass shared; rows independent). In
        ``replay`` mode only the encoder layers (0-19, + layer 20's compressor projection)."""

    def finish_prompt(self, slot: int, end: int, tail: Sequence[int], *, mode: str) -> None:
        """The prompt's prefilled part ends at ``end`` (= n - 1); ``tail`` = its last ids (at most REPLAY - 1, ending
        at ``end``). Replay mode: rebuild the decoder rings from ``tail``. Full mode: nothing."""

    def window(self, windows: Sequence[Rows], *, count: int,
               masks: Sequence[np.ndarray | None] | None = None) -> Candidates:
        """One verify forward over every window (rows concatenated); ``masks[i]``: xgrammar bits [len(rows_i),
        ceil(vocab / 32)] int32 for window i (None: unconstrained), applied as -inf before the top ``count``."""

    def commit(self, slot: int, accepted: int) -> None:
        """Keep the last window's pending row + ``accepted`` drafts of ``slot`` (carry, lookback, taps, pos)."""

    def snapshot(self, slot: int) -> Bounded:
        """The slot's bounded state at its position (any position; carries included when it is odd)."""

    def restore(self, slot: int, snap: Bounded) -> None:
        """Load ``snap`` into ``slot`` (its pages below snap.pos are mapped and hold the entry's rows)."""


class Bind(Protocol):
    """Optional: the executor hands the forward its slots' page tables once at load (``pool.Slot``: ``table`` is the
    device int32 page table, ``pages`` the host list); the forward keeps them for every call."""

    def bind(self, slots: Sequence[Any]) -> None: ...


class Drafts(Protocol):
    """Optional (M2): DSpark proposals, run on both ranks inside the round (``plan.WindowSpec.depth`` > 0).
    ``draft(slots, pendings, starts, depths)`` -> each slot's drafted tokens after its pending token (at most its
    depth; the taps of the slot's last window + the pending token, keyed draft noise at T > 0). Both ranks must get
    the same tokens (they come from merged candidates)."""

    def draft(self, slots: Sequence[int], pendings: Sequence[int], starts: Sequence[int],
              depths: Sequence[int]) -> list[list[int]]: ...


class Prefetch(Protocol):
    """Optional: Engram reads for the next forward, issued before it (``engram.Reader``)."""

    def prefetch(self, pieces: Sequence[Piece], windows: Sequence[Rows]) -> None: ...


class Link(Protocol):
    """Rank agreement for round plans: rank 0 ``send``s each plan, rank 1 ``recv``s it (GLM ``_share`` of an int
    list: a length gather then a value gather on the communicator); ``wake`` / ``wait``: the idle doorbell."""

    def send(self, plan: list[int]) -> None: ...

    def recv(self) -> list[int]: ...


@dataclass
class LocalLink:
    """One rank (tests, single-node bring-up): plans go nowhere."""

    def send(self, plan: list[int]) -> None:
        return None

    def recv(self) -> list[int]:
        raise RuntimeError("a one-rank link has nothing to receive")


def pack_bytes(data: bytes) -> list[int]:
    """3 bytes an int (non-negative int32 words, GLM ``grammar._pack_bytes``)."""

    data = data + b"\0" * (-len(data) % 3)
    return [data[i] | data[i + 1] << 8 | data[i + 2] << 16 for i in range(0, len(data), 3)]


def unpack_bytes(words: Sequence[int], n: int) -> bytes:
    out = bytearray()
    for v in words:
        v = int(v)
        out += bytes((v & 255, v >> 8 & 255, v >> 16 & 255))
    return bytes(out[:n])


def accepted(window_tokens: Sequence[int], chosen: Sequence[int]) -> int:
    """Drafts kept from a chain window: ``window_tokens`` = [pending, d1, d2, ...], ``chosen`` = the target's keyed
    choice after each row. Keep d_i while d_i == chosen[i - 1]; the round emits accepted + 1 tokens (drafted ==
    serial for any drafter: ``drafting.accepted``)."""

    n = 0
    for d, c in zip(window_tokens[1:], chosen):
        if d != c:
            break
        n += 1
    return n


def check_forward(fwd: Any) -> None:
    """Raise TypeError when ``fwd`` lacks a ``Forward`` method (load-time check of M1's object)."""

    missing = [m for m in ("reset", "prefill", "finish_prompt", "window", "commit", "snapshot", "restore")
               if not callable(getattr(fwd, m, None))]
    if missing or not isinstance(getattr(fwd, "vocab", None), int):
        raise TypeError(f"not a V4.1 Forward: missing {', '.join(missing) or 'vocab'}")
