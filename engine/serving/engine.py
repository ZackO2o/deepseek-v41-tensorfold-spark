"""The engine contract of the DeepSeek-V4.1 CUDA family (interface stub; the forward lands in milestone M1).

It is the GLM Spark engine's contract (``glm5_next/spark/engine.py: GlmEngine``), so the server, the batcher, the
session store and the ops tooling of our GLM stack take it unchanged:

- ``generate(prompt, max_tokens, sampling, on_tokens, draft=True) -> dict`` on rank 0, ``follow()`` on rank 1;
  per-request extras (stop_eos, grammar, knobs, background, vision, policy) arrive through ``self.request``, a
  ``threading.local`` the app fills (GLM's ``GlmApp.run``);
- ``eos``, ``limit`` (the most tokens a request may hold: min(--context, pool)), ``batch`` (the Batcher when
  ``--parallel`` > 1), ``store`` (the session store), ``vision`` (rank 0's encoder or None);
- rank agreement: rank 0 shares an int header and the prompt (``_share``: a length gather then a value gather on
  the communicator), rank 1 runs the same forwards; with a batcher, a round plan (``batchplan.encode_plan``);
- sampling: ``tensorfold.engine.exact_sampling`` (keyed: seed, absolute position, token id), each rank taking the
  top candidates of its half of the vocabulary and one ``fast_gather`` merging them (GLM ``decode.sample_rows``).

What is V4.1's (``Forward`` below, built from engine/kernels and engine/reference's layer math):

- a slot's state is ``state.SlotState`` + pool pages, not KDA state;
- a round is ``Forward.window``: Engram reads already issued (``engram.Reader``), then 40 layers of
  mHC -> attention (CSA2 by role, ``topology``) -> mHC -> MoE (router + grouped mul1 experts + shared), the head;
- prompts run ``Forward.prefill`` pieces (full decoder, or CED replay: ``TF_DSV41_PREFILL``);
- DSpark (``drafting.DSpark``) proposes from the taps of layers 37-39.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

EOS = (1,)                               # config.json eos_token_id; the template's end-of-turn ids join at load


class Forward(Protocol):
    """What the batcher calls; every method runs on both ranks with the same arguments (the plan says which)."""

    def prefill(self, slot: int, ids: Sequence[int], start: int, *, mode: str) -> None:
        """Prompt tokens [start, start + len(ids)) of ``slot``: ``mode`` "full" (all 40 layers) or "replay"
        (encoder layers 0-19 here; the decoder runs over the prompt's last 128 tokens in ``finish_prompt``)."""

    def finish_prompt(self, slot: int, n: int, *, mode: str) -> Any:
        """The prompt's end: in replay mode the decoder over tokens [max(0, n - 128), n) with its SWA windows
        starting at that segment; returns the last row's logits (rank-local vocabulary half)."""

    def window(self, slots: Sequence[int], rows: Sequence[Sequence[int]]) -> Any:
        """One verify round: each slot's pending token + drafts at its next positions, all 40 layers, every row
        independent (row invariance); returns logits [sum rows, vocab / 2] and fills the DSpark taps."""

    def commit(self, slot: int, accepted: int) -> None:
        """Keep ``accepted`` rows of the slot's last window: positions, the compressor carry (the last accepted
        row's projection), the Engram lookback, the taps; rejected rows' pool / ring writes are overwritten later."""

    def snapshot(self, slot: int) -> Any:
        """``sessions.Snapshot`` of the slot at its current (grid-aligned) position."""

    def restore(self, slot: int, snap: Any) -> None:
        """Load a snapshot into the slot (pages already mapped by the session store)."""


@dataclass
class Request:
    """``engine.request``'s fields (set by the app per request thread)."""

    stop_eos: bool = True
    grammar: Any = None
    knobs: dict | None = None
    background: bool = False
    vision: Any = None
    policy: Any = None


class Dsv41Engine:
    """Interface stub: the constructor's arguments and attributes are final; the bodies arrive with M1 / M3."""

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, context: int = 0,
                 serial_only: bool = False, draft_depth: int | None = None, comm=None) -> None:
        self.model_dir = Path(model_dir)
        self.rank = rank
        self.serial_only = serial_only
        self.draft_depth = draft_depth
        self.eos = EOS
        self.request = threading.local()
        self.batch = None
        self.store = None
        self.vision = None
        self.limit = context or 300_000
        self._master, self._port, self._comm = master, port, comm

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool],
                 draft: bool = True) -> dict[str, Any]:
        raise NotImplementedError("M1: serial decode; M2: DSpark rounds; M3: through the batcher")

    def follow(self) -> None:
        raise NotImplementedError("M1")
