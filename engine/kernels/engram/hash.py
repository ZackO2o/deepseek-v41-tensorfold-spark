"""Engram addresses on the host (outside every CUDA graph) and the prefetch that issues a round's reads at drafter end.

The hash (vLLM ``deepseek_v4_1/common/engram.py``, Apache-2.0, math only; ``engine/reference/engram.py: NgramHasher``
is the oracle, bit for bit): with x' the compressed id of a token (``token_map``; DEAD for an image token),

    rolling_s = rolling_{s-1} XOR (value_s * mult[layer, s])     s = 0 .. 3, value_s = x'[t - s] or the pad id once
                                                                  slot s or a newer one is before the start or DEAD
    row[layer, (n - 2) * 8 + h] = rolling_{n-1} % prime[layer, n, h] + offset[layer, n, h]     n = 2 .. 4, h = 0 .. 7

int64 arithmetic with wrap-around products and a non-negative remainder (numpy's ``%`` is torch's). A token's rows
depend on its own id and the 3 before it, so a slot keeps a lookback of 3 compressed ids (DEAD kept as such), and a
round's rows (the pending token's and every draft's) are known the moment the drafter has proposed.

``Tables`` holds what the prepared folder stores (token map, multipliers, primes, offsets, pad id); ``Prefetch``
turns (lookback, window tokens) into the rank's 12 rows a token a layer, submits them to the NVMe reader
(``engine/serving/engram.py: Reader``) at once, and later lands them in a pinned staging block that one H2D copy
moves to the device buffer the Engram kernels (``engram.kernels``) read. Rows of rejected drafts cost a read only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEAD = -1
LOOKBACK = 3
HEAD_COLS = 24          # 3 orders x 8 heads a layer
RECORD = 264            # 256 e4m3 + 8 UE8M0 scale bytes


@dataclass
class Tables:
    token_map: np.ndarray       # [vocab] int64 -> compressed id
    mult: np.ndarray            # [layers, max_ngram] int64 (odd)
    primes: np.ndarray          # [layers, 24] int64
    offsets: np.ndarray         # [layers, 24] int64
    pad_id: int                 # compressed id of the pad token
    layer_ids: tuple[int, ...] = (1, 14)
    n_heads: int = 8
    max_ngram: int = 4

    @classmethod
    def from_config(cls, cfg, token_map) -> "Tables":
        from engine.reference.engram import EngramLayout, hash_multipliers

        lay = EngramLayout(cfg)
        tm = np.asarray(token_map, dtype=np.int64)
        mult = hash_multipliers(lay.layer_ids, cfg.engram_max_ngram_size, cfg.engram_compressed_vocab_size)
        return cls(tm, mult.numpy().astype(np.int64), lay.primes.numpy(), lay.offsets.numpy(),
                   int(tm[cfg.engram_pad_token_id]), tuple(lay.layer_ids), cfg.engram_n_heads,
                   cfg.engram_max_ngram_size)

    def save(self, path: str | Path) -> None:
        np.savez(path, token_map=self.token_map, mult=self.mult, primes=self.primes, offsets=self.offsets,
                 meta=np.array([self.pad_id, self.n_heads, self.max_ngram, *self.layer_ids], dtype=np.int64))

    @classmethod
    def load(cls, path: str | Path) -> "Tables":
        z = np.load(path)
        meta = z["meta"].tolist()
        return cls(z["token_map"], z["mult"], z["primes"], z["offsets"], meta[0], tuple(meta[3:]), meta[1], meta[2])

    @property
    def cols(self) -> int:
        return (self.max_ngram - 1) * self.n_heads

    def shard(self, li: int, rank: int, world: int) -> tuple[int, int, int, int]:
        """(row lo, row hi, first column, columns) of a TP rank: complete heads, rank-major (vLLM's sharding)."""

        part = (self.cols + world - 1) // world
        c0 = rank * part
        c1 = min(c0 + part, self.cols)
        sizes = self.primes[li]
        return int(sizes[:c0].sum()), int(sizes[:c1].sum()), c0, c1 - c0

    def compress(self, ids, dead=None) -> np.ndarray:
        """Token ids -> compressed ids, DEAD where ``dead`` (image tokens)."""

        x = self.token_map[np.asarray(ids, dtype=np.int64)]
        if dead is not None:
            x = np.where(np.asarray(dead, dtype=bool), DEAD, x)
        return x

    def rows(self, ids, dead=None, lookback=None) -> np.ndarray:
        """[T] token ids (``dead`` [T] image tokens) after ``lookback`` (up to 3 compressed ids, oldest first,
        DEAD allowed; fewer = the sequence starts there) -> [T, layers, 24] int64 table rows."""

        cur = self.compress(ids, dead)
        lb = np.zeros(0, dtype=np.int64) if lookback is None else np.asarray(lookback, dtype=np.int64)
        src = np.concatenate([lb, cur])
        t, nb = cur.size, lb.size
        nl = len(self.layer_ids)
        out = np.empty((t, nl, self.cols), dtype=np.int64)
        idx = np.arange(t) + nb
        with np.errstate(over="ignore"):
            for li in range(nl):
                blocked = np.zeros(t, dtype=bool)
                rolling = np.zeros(t, dtype=np.int64)
                for s in range(self.max_ngram):
                    j = idx - s
                    before = j < 0
                    v = np.where(before, self.pad_id, src[np.maximum(j, 0)])
                    blocked = blocked | before | (v == DEAD)
                    value = np.where(blocked, self.pad_id, v)
                    rolling = rolling ^ (value * self.mult[li, s])
                    if s > 0:
                        cols = slice((s - 1) * self.n_heads, s * self.n_heads)
                        out[:, li, cols] = rolling[:, None] % self.primes[li, cols][None, :] + \
                            self.offsets[li, cols][None, :]
        return out


def next_lookback(lookback, tables: Tables, ids, dead=None) -> np.ndarray:
    """A slot's lookback after committing ``ids``: the last 3 compressed ids (DEAD kept)."""

    lb = np.zeros(0, dtype=np.int64) if lookback is None else np.asarray(lookback, dtype=np.int64)
    return np.concatenate([lb, tables.compress(ids, dead)])[-LOOKBACK:].copy()


@dataclass
class RoundTicket:
    n: int                                  # window rows (tokens)
    tickets: list                           # one reader ticket a layer
    landed: list = field(default_factory=list)


class Prefetch:
    """The rank's Engram reads for decode rounds and prefill chunks.

    ``issue(lookback, tokens, dead)`` hashes the window (``[pending, d1, ..]`` at drafter end; a chunk when it is
    cut) and submits the rank's 12 rows a token a layer to the reader at once (O_DIRECT, deduplicated, coalesced);
    ``land(ticket, layer_index)`` waits for one layer's rows and writes them into the pinned staging block
    [layers, max_rows, 12, 264] uint8; ``upload(layer_index, dst, n)`` is the one H2D copy a layer (non-blocking
    on the current stream), into the fixed device buffer the graphs read. Layer 1's rows are landed before the
    forward reaches layer 1; layer 14's have half a forward of slack (land them after layer 1 is queued)."""

    def __init__(self, tables: Tables, reader, rank: int, world: int, max_rows: int, *, pin: bool = True) -> None:
        import torch

        self.t, self.reader, self.rank, self.world = tables, reader, rank, world
        self.max_rows = max_rows
        self.sh = [tables.shard(li, rank, world) for li in range(len(tables.layer_ids))]
        ncol = self.sh[0][3]
        self.staging = torch.empty((len(tables.layer_ids), max_rows, ncol, RECORD), dtype=torch.uint8,
                                   pin_memory=pin and torch.cuda.is_available())

    def issue(self, lookback, tokens, dead=None) -> RoundTicket:
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.size > self.max_rows:
            raise ValueError(f"Engram prefetch: {tokens.size} rows > {self.max_rows}")
        rows = self.t.rows(tokens, dead, lookback)
        tickets = []
        for li, (_, _, c0, nc) in enumerate(self.sh):
            mine = rows[:, li, c0:c0 + nc].reshape(-1)
            tickets.append(self.reader.submit(self.t.layer_ids[li], mine))
        return RoundTicket(int(tokens.size), tickets, [False] * len(tickets))

    def land(self, rt: RoundTicket, li: int) -> None:
        if rt.landed[li]:
            return
        got = self.reader.result(rt.tickets[li])                                 # [n * 12, 264] uint8
        import torch

        self.staging[li, :rt.n].copy_(torch.from_numpy(got).view(rt.n, -1, RECORD))
        rt.landed[li] = True

    def upload(self, li: int, dst, n: int) -> None:
        dst[:n].copy_(self.staging[li, :n], non_blocking=True)
