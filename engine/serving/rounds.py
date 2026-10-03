# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""The round executor of the V4.1 batcher: one ``plan.Plan`` applied to the forward, the pool, the session store and
the slots' grammars, identically on both ranks (rank 0's ``batch.Batcher`` plans and samples; rank 1 only executes).

Everything here is deterministic given the plan and the rank's previous state, so both ranks' pools, stores and
grammar matchers stay identical; the only cross-rank data is the plan (and, inside the forward, its exchanges).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .plan import Admit, Plan
from .pool import Pool
from .protocol import REPLAY, Candidates, Piece, Rows


@dataclass
class Result:
    rows: list[Rows] = field(default_factory=list)          # the windows run, in plan order (after grammar cuts)
    cand: Candidates | None = None
    errors: dict[int, BaseException] = field(default_factory=dict)    # slot -> its request's failure
    restored: dict[int, int] = field(default_factory=dict)            # slot -> tokens resumed
    skipped: set[int] = field(default_factory=set)    # slots whose restore failed: nothing more for them this round
    piece_s: float = 0.0                                # the prefill forward's seconds (fairness)
    window_s: float = 0.0                               # the verify forward's seconds

    def out(self, slot: int) -> bool:
        return slot in self.errors or slot in self.skipped


class Executor:
    """One rank's slots: page tables, token histories (what each slot's state holds), grammar states."""

    def __init__(self, fwd, pool: Pool, n_slots: int, capacity: int, *, store=None, grammars=None, tag: int = 0,
                 mode: str = "full", rank: int = 0, session_min: int = 64) -> None:
        self.fwd = fwd
        self.pool = pool
        self.store = store
        self.grammars = grammars
        self.tag = int(tag)
        self.mode = mode
        self.rank = rank
        self.session_min = int(session_min)
        self.capacity = int(capacity)
        self.slots = [pool.new_slot(capacity) for _ in range(n_slots)]
        if callable(getattr(fwd, "bind", None)):
            fwd.bind(self.slots)                # the forward reads the slots' page tables (``Forward.bind``)
        self.hist: list[list[int]] = [[] for _ in range(n_slots)]
        self.prompts: list[list[int] | None] = [None] * n_slots
        self.cons: list[Any] = [None] * n_slots
        self.last: list[Rows | None] = [None] * n_slots
        self.rounds = 0

    # -- steps ---------------------------------------------------------------------------------------------------
    def run(self, plan: Plan) -> Result:
        res = Result()
        self.rounds += 1
        for slot, acc, bonus in plan.commits:
            self._commit(slot, int(acc), int(bonus))
        for slot, save, cancelled in plan.finishes:
            self._finish(slot, bool(save))
        for key in plan.spills:
            self.store.evict(key)
        for a in plan.admits:
            self._admit(a, res)
        if plan.pieces:
            pieces = []
            for slot, start, end in plan.pieces:
                if res.out(slot):
                    continue
                self.slots[slot].ensure(end)
                ids = tuple(self.prompts[slot][start:end])
                pieces.append(Piece(slot, start, ids))
                self.hist[slot].extend(ids)
            if pieces:
                t0 = time.perf_counter()
                self.fwd.prefill(pieces, mode=plan.mode)
                res.piece_s = time.perf_counter() - t0
        for slot in plan.saves:
            if not res.out(slot):
                self._save(slot, "prompt")
        for slot in plan.finals:
            if res.out(slot):
                continue
            p = self.prompts[slot]
            n = len(p)
            self.fwd.finish_prompt(slot, n - 1, tuple(p[max(0, n - REPLAY):n - 1]), mode=plan.mode)
        if plan.windows:
            t0 = time.perf_counter()
            self._windows(plan, res)
            res.window_s = time.perf_counter() - t0
        return res

    def _commit(self, slot: int, acc: int, bonus: int) -> None:
        rows = self.last[slot]
        if rows is None:
            raise RuntimeError(f"commit for slot {slot} without a window")
        self.fwd.commit(slot, acc)
        kept = list(rows.tokens[:acc + 1])
        self.hist[slot].extend(kept)
        con = self.cons[slot]
        if con is not None:
            con.advance(kept[1:] + [bonus])
        self.last[slot] = None

    def _finish(self, slot: int, save: bool) -> None:
        if save:
            self._save(slot, "turn")
        self.slots[slot].release()
        self.hist[slot] = []
        self.prompts[slot] = None
        self.cons[slot] = None
        self.last[slot] = None

    def _save(self, slot: int, kind: str):
        if self.store is None or len(self.hist[slot]) < self.session_min:
            return None
        snap = self.fwd.snapshot(slot)
        if snap.pos != len(self.hist[slot]):
            raise RuntimeError(f"slot {slot}: snapshot at {snap.pos}, history {len(self.hist[slot])}")
        return self.store.save(self.slots[slot], snap, self.hist[slot], self.tag, kind)

    def _admit(self, a: Admit, res: Result) -> None:
        sl = self.slots[a.slot]
        if sl.pages:
            raise RuntimeError(f"slot {a.slot} admitted while it maps pages")
        sl.reserve(a.quota)
        self.fwd.reset(a.slot)
        self.prompts[a.slot] = list(a.prompt)
        self.hist[a.slot] = []
        self.last[a.slot] = None
        cached = 0
        if a.tier:
            try:
                snap = self.store.restore(a.tier, a.key, sl)
                if snap.pos != a.cached:
                    raise ValueError(f"entry holds {snap.pos} tokens, the plan says {a.cached}")
                self.fwd.restore(a.slot, snap)
                cached = a.cached
            except (ValueError, KeyError):      # a damaged NVMe entry: prefill from the start instead
                sl.truncate(0)
                self.fwd.reset(a.slot)
                cached = 0
                res.skipped.add(a.slot)
        self.hist[a.slot] = list(a.prompt[:cached])
        res.restored[a.slot] = cached
        self.cons[a.slot] = None
        if a.grammar:
            from tensorfold.families.glm5_next.spark import grammar as gm

            body, _ = gm.split(a.grammar)
            try:
                bound = self.grammars.follow(body)
                self.cons[a.slot] = self.grammars.constraint(bound)
            except Exception as exc:            # noqa: BLE001  (the request fails, not the server)
                res.errors[a.slot] = exc

    def _windows(self, plan: Plan, res: Result) -> None:
        specs = [w for w in plan.windows if not res.out(w.slot)]
        drafts: dict[int, list[int]] = {}
        deep = [w for w in specs if w.depth > 0]
        if deep and callable(getattr(self.fwd, "draft", None)):
            for w in deep:                      # the window's rows (pending + depth) may be written by the drafter
                self.slots[w.slot].ensure(w.start + w.depth + 1)
            got = self.fwd.draft([w.slot for w in deep], [w.pending for w in deep], [w.start for w in deep],
                                 [w.depth for w in deep])
            drafts = {w.slot: [int(t) for t in d][:w.depth] for w, d in zip(deep, got)}
        rows, masks, any_mask = [], [], False
        for w in specs:
            tokens = [int(w.pending)] + (drafts.get(w.slot) or [int(t) for t in w.drafts])
            con = self.cons[w.slot]
            bits = None
            if con is not None:
                try:
                    gwin = con.fill(con.cut(tokens))
                    tokens, bits = gwin.tokens, self._full_bits(gwin, len(gwin.tokens), con.words)
                except Exception as exc:        # noqa: BLE001  (GrammarError: that request ends)
                    res.errors[w.slot] = exc
                    continue
            any_mask = any_mask or bits is not None
            self.slots[w.slot].ensure(w.start + len(tokens))
            r = Rows(w.slot, w.start, tuple(tokens))
            self.last[w.slot] = r
            rows.append(r)
            masks.append(bits)
        res.rows = rows
        if rows:
            res.cand = self.fwd.window(rows, count=plan.count, masks=masks if any_mask else None)

    @staticmethod
    def _full_bits(gwin, n: int, words: int) -> np.ndarray | None:
        """The window's mask for every row ([n, words]; unconstrained rows (thinking) all ones), or None."""

        if gwin.bits is None or not gwin.rows:
            return None
        bits = np.full((n, words), -1, dtype=np.int32)
        bits[np.asarray(gwin.rows)] = gwin.bits
        return bits

    # -- views ---------------------------------------------------------------------------------------------------
    def describe(self) -> str:
        busy = sum(1 for p in self.prompts if p is not None)
        return f"{busy} of {len(self.slots)} slots busy; pool {self.pool.describe()}"


def take_rows(cand: Candidates, rows: Sequence[Rows]) -> list[Candidates]:
    """Split a window's candidates per slot window."""

    out, off = [], 0
    for r in rows:
        out.append(cand.rows(off, off + len(r.tokens)))
        off += len(r.tokens)
    return out
