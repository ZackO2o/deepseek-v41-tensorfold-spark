# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""The V4.1 Batcher (``--parallel`` slots on one engine): GLM's batcher protocol (``glm5_next/spark/batch.py``: a job
queue, rank 0 plans each round and shares it, both ranks execute, cancellation by plan, tokens back through each
job's queue) driving the V4.1 ``Forward`` (``protocol.py``).

A round (rank 0, ``step``): plan (``_plan``), share (``Link.send``), execute (``rounds.Executor``), sample and emit
(``_sample``). Rank 1 runs ``follow``: receive, execute. The plan carries everything rank 1 needs (``plan.py``).

- **Admission**: foreground before background, FIFO; a request needs a free slot, its pool pages (prompt +
  max_tokens + slack, GLM ``kvpool.need_tokens``; a RAM session hit's full pages are shared, not new), and the
  memory floor (``memory.Floor``, GLM 0550's ``memsafe``). When the pool is short, the coldest session entries are
  parked to NVMe (or dropped) first (``Store.plan_spill``, GLM ``batchplan.pool_spills``). A prompt resumes from the
  longest stored prefix (RAM or NVMe, ``sessions.Store.find``).
- **Prefill**: pieces on the 16-token grid, ``TF_DSV41_PREFILL_ROWS`` rows a round shared by the prefilling slots
  (GLM 0560 multi-slot prefill: one ``Forward.prefill`` over every slot's piece); one piece ends at the replay point
  (a prompt snapshot there, GLM 0540); decoding slots run every round beside the pieces.
- **Decode**: one ``Forward.window`` over every decoding slot (batched == alone: row-invariant kernels); DSpark
  drafts of ``TF_DSV41_DRAFT_DEPTH`` when the forward drafts (``Forward.draft``) and the request allows;
  tokens chosen by ``exact_sampling.choose_rows`` keyed by (seed, position, id), kept up to the first draft that
  differs (``drafting.accepted``): drafted == serial, batched == alone.
- **Structured output**: the request's grammar (GLM 0610 ``Grammars`` / ``Constraint``; DSML tools through
  ``structured.py``) cuts each window and masks its rows on both ranks.
- **Cancellation** (GLM 0600): ``on_tokens`` returning True, or ``on_tokens.cancelled()`` turning true while the
  request waits, prefills or decodes, ends it at the next plan (no snapshot of a cancelled turn).
"""

from __future__ import annotations

import collections
import itertools
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.families.glm5_next.spark.batchplan import Fairness

from .plan import Admit, Plan, WindowSpec, piece_end
from .pool import PAGE, Pool
from .protocol import GRID, LocalLink, accepted, check_forward
from .rounds import Executor, take_rows
from .sessions import snapshot_point

_ids = itertools.count(1)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@dataclass
class Job:
    prompt: list[int]
    max_tokens: int
    sampling: Any = None                       # exact_sampling.Sampling, None = greedy
    stop_eos: bool = True
    draft: bool = True
    grammar: Any = None                        # GLM grammar.Bound
    background: bool = False
    out: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)
    cancel: bool = False
    id: int = field(default_factory=lambda: next(_ids))
    submitted: float = field(default_factory=time.monotonic)
    stats: dict = field(default_factory=dict)


@dataclass
class Seq:
    job: Job
    slot: int
    done: int                                  # prompt tokens in the slot's state (prefill progress)
    save_at: int | None                        # the replay point still to snapshot at
    pos: int = 0                               # decode: the pending token's position
    pending: int = 0
    decoding: bool = False
    out: list[int] = field(default_factory=list)
    t0: float = field(default_factory=time.monotonic)

    @property
    def stepper(self):
        """GLM health's view: not None while decoding (``/health``'s decoding / prefilling counts)."""

        return self if self.decoding else None


class Batcher:
    def __init__(self, fwd, pool: Pool, *, n_slots: int = 4, capacity: int, eos: Sequence[int] = (1,), store=None,
                 link=None, rank: int = 0, grammars=None, floor=None, mode: str = "full", tag: int = 0,
                 prefill_rows: int | None = None, depth: int | None = None, session_min: int = 64,
                 poll_s: float = 0.05, start: bool = True) -> None:
        check_forward(fwd)
        self.fwd = fwd
        self.n = int(n_slots)
        self.capacity = int(capacity)
        self.eos = frozenset(int(t) for t in eos)
        self.store, self.pool, self.kvp = store, pool, pool
        self.link = link or LocalLink()
        self.rank = rank
        self.floor = floor
        self.mode, self.tag = mode, int(tag)
        self.prefill_rows = prefill_rows or _env_int("TF_DSV41_PREFILL_ROWS", 2048)
        can_draft = callable(getattr(fwd, "draft", None))
        self.depth = (_env_int("TF_DSV41_DRAFT_DEPTH", 3) if depth is None else int(depth)) if can_draft else 0
        self.poll_s = poll_s
        # GLM 0120 / 0200: prefill pieces take at most this share of the time while others decode; prompts with at
        # most ``short`` tokens left go at once
        self.fair = Fairness(float(os.environ.get("TF_DSV41_PREFILL_SHARE", "") or 0.5))
        self.short = _env_int("TF_DSV41_BATCH_SHORT", 512)
        self.ex = Executor(fwd, pool, self.n, capacity, store=store, grammars=grammars, tag=tag, mode=mode,
                           rank=rank, session_min=session_min)
        self.queue: collections.deque[Job] = collections.deque()
        self.cv = threading.Condition()
        self.seqs: list[Seq | None] = [None] * self.n
        self.commits: list[tuple[int, int, int]] = []
        self.finishes: list[tuple[int, bool, bool]] = []
        self.rounds = 0
        self.stopping = False
        self.in_round = False
        self.counts = collections.Counter()
        self.thread = None
        if rank == 0 and start:
            self.thread = threading.Thread(target=self._serve, name="dsv41-batch", daemon=True)
            self.thread.start()

    # -- the request side (HTTP threads, rank 0) -------------------------------------------------------------------
    def submit(self, job: Job) -> Job:
        n = len(job.prompt)
        if n < 1:
            raise ValueError("an empty prompt")
        if n + max(job.max_tokens, 1) > self.capacity:
            raise ValueError(f"prompt {n} + max_tokens {job.max_tokens} is past the slot capacity {self.capacity}")
        with self.cv:
            self.queue.append(job)
            self.cv.notify_all()
        return job

    def cancel(self, job: Job) -> None:
        with self.cv:
            job.cancel = True
            self.cv.notify_all()

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool],
                 draft: bool = True, *, grammar=None, stop_eos: bool = True, background: bool = False) -> dict:
        job = self.submit(Job(list(prompt), int(max_tokens), sampling, stop_eos, draft, grammar, background))
        return self.collect(job, on_tokens)

    def generate_request(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True, *,
                         request=None, host=None) -> dict:
        """The body of the engine's ``generate`` (M1's ``Dsv41Engine``): the app's per-thread fields
        (``engine.request``: stop_eos, grammar as (spec, compiled), background) applied; ``host``:
        ``structured.Host``."""

        grammar = host.bind(getattr(request, "grammar", None), prompt) if host is not None else None
        return self.generate(prompt, max_tokens, sampling, on_tokens, draft, grammar=grammar,
                             stop_eos=getattr(request, "stop_eos", True),
                             background=bool(getattr(request, "background", False)))

    def collect(self, job: Job, on_tokens: Callable[[list[int]], bool]) -> dict:
        """Tokens to ``on_tokens`` until the job ends; its stats. Raises the job's failure (ValueError: the client's)."""

        gone = getattr(on_tokens, "cancelled", None)
        while True:
            try:
                item = job.out.get(timeout=self.poll_s)
            except queue.Empty:
                if gone is not None and not job.cancel and gone():
                    self.cancel(job)
                continue
            if item is None:
                return dict(job.stats)
            if isinstance(item, BaseException):
                raise item
            try:
                stop = on_tokens(item)
            except BaseException:
                self.cancel(job)
                raise
            if stop and not job.cancel:
                self.cancel(job)

    def stop(self) -> None:
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=10)

    # -- the round loop (rank 0) -----------------------------------------------------------------------------------
    def _serve(self) -> None:
        while not self.stopping:
            self.step(block=True)

    def busy(self) -> bool:
        return bool(self.in_round or self.queue or self.commits or self.finishes
                    or any(s is not None for s in self.seqs))

    def step(self, block: bool = False) -> bool:
        """One round; False when there was nothing to do."""

        with self.cv:
            while block and not self.stopping and not self._work():
                self.cv.wait(timeout=0.5)
            plan = self._plan()
            if plan.empty():
                return False
            self.in_round = True
        try:
            self.link.send(plan.encode())
            res = self.ex.run(plan)
            self._sample(plan, res)
            self.fair.after(res.piece_s, res.window_s,
                            any(s is not None and not s.decoding for s in self.seqs))
        except Exception as exc:                # noqa: BLE001  every running request fails; the loop goes on
            traceback.print_exc()
            self._fail_all(exc)
        finally:
            self.in_round = False
        self.rounds += 1
        return True

    def _work(self) -> bool:
        if self.commits or self.finishes:
            return True
        if any(s is not None for s in self.seqs):
            return True
        return bool(self.queue) and any(s is None for s in self.seqs)

    def _plan(self) -> Plan:
        plan = Plan(round=self.rounds, mode=self.mode)
        plan.commits, self.commits = self.commits, []
        plan.finishes, self.finishes = self.finishes, []
        for s in self.seqs:                     # cancelled while running: no turn snapshot
            if s is not None and s.job.cancel:
                self._end(s, plan, cancelled=True)
        for job in [j for j in self.queue if j.cancel]:
            self.queue.remove(job)
            job.stats.update(finish="cancelled", cached=0)
            job.out.put(None)
        self._admit(plan)
        self._pieces(plan)
        self._decode(plan)
        return plan

    def _admit(self, plan: Plan) -> None:
        free = [i for i, s in enumerate(self.seqs) if s is None]
        if not free or not self.queue:
            return
        order = sorted(self.queue, key=lambda j: (j.background, j.submitted))
        new_pages = 0
        used: list[str] = []
        for job in order:
            if not free:
                break
            if self.floor is not None and not self.floor.check(0, queued=len(self.queue)):
                self.counts["mem_waits"] += 1
                break
            n = len(job.prompt)
            tier, key, cached = "", "", 0
            hit = self.store.find(job.prompt, self.tag) if self.store is not None else None
            if hit is not None:
                tier, key = hit
                cached = self.store.length(tier, key)
            need = self.pool.need_pages(n, job.max_tokens, self.capacity)
            shared = cached // PAGE if tier == "ram" else 0
            want = need - shared + new_pages
            if need > self.pool.npages:
                self.queue.remove(job)
                job.out.put(ValueError(f"the request needs {need} KV pool pages, the pool has {self.pool.npages}"))
                continue
            spills: list[str] | None = []
            if self.pool.available() < want:
                spills = self.store.plan_spill(want, keep=used + [key]) if self.store is not None else None
                if spills is None or plan.spills:
                    self.counts["pool_waits"] += 1
                    break                       # running requests free pages as they end
            plan.spills += spills
            used.append(key)
            new_pages = want
            slot = free.pop(0)
            self.queue.remove(job)
            from tensorfold.families.glm5_next.spark import grammar as gm

            plan.admits.append(Admit(slot, list(job.prompt), job.max_tokens, tier, key, cached, need,
                                     gm.pack(job.grammar) if job.grammar is not None else None, job.id))
            s_at = snapshot_point(n)
            save_at = s_at if self.store is not None and s_at > cached and s_at >= self.ex.session_min else None
            self.seqs[slot] = Seq(job, slot, cached, save_at)
            job.stats.update(prompt=n, cached=cached, tier=tier or "none", slot=slot,
                             wait_s=round(time.monotonic() - job.submitted, 4))

    def _pieces(self, plan: Plan) -> None:
        """This round's prefill pieces: prompts with at most ``short`` tokens left always; the others, fewest tokens
        left first, when the fair share allows (GLM ``batchplan.Fairness``), as many as the rows budget holds
        (GLM 0560: several slots' pieces in one forward)."""

        live = [s for s in self.seqs if s is not None and not s.job.cancel]
        waiting = [s for s in live if not s.decoding and s.done < len(s.job.prompt) - 1]
        allow = self.fair.allow(any(s.decoding for s in live))
        budget = self.prefill_rows
        for s in sorted(waiting, key=lambda s: (len(s.job.prompt) - 1 - s.done, s.job.submitted)):
            left = len(s.job.prompt) - 1 - s.done
            if budget <= 0 or (left > self.short and not allow):
                continue
            target = s.save_at if s.save_at is not None and s.done < s.save_at else len(s.job.prompt) - 1
            end = piece_end(s.done, target, budget, GRID)
            plan.pieces.append((s.slot, s.done, end))
            budget -= end - s.done
            s.done = end
            if s.save_at is not None and end == s.save_at:
                plan.saves.append(s.slot)
                s.save_at = None
        for s in live:
            last = len(s.job.prompt) - 1
            if not s.decoding and s.done == last:
                plan.finals.append(s.slot)
                s.decoding, s.pos, s.pending = True, last, int(s.job.prompt[-1])

    def _decode(self, plan: Plan) -> None:
        count = 1
        for s in self.seqs:
            if s is None or not s.decoding or s.job.cancel:
                continue
            left = s.job.max_tokens - len(s.out)
            depth = min(self.depth, max(0, left - 1)) if s.job.draft else 0
            plan.windows.append(WindowSpec(s.slot, s.pos, s.pending, depth))
            sp = s.job.sampling
            if sp is None:
                k = 1 + MARGIN
            else:
                k = self.fwd.vocab if not sp.top_k else int(sp.top_k) + MARGIN
            count = max(count, min(k, self.fwd.vocab))
        plan.count = count

    # -- after the forward (rank 0) --------------------------------------------------------------------------------
    def _sample(self, plan: Plan, res) -> None:
        for slot in res.skipped:
            s = self.seqs[slot]
            if s is not None:                   # a damaged NVMe entry: prefill from the start
                s.done, s.pos = 0, 0
                s.decoding = False
                s.save_at = snapshot_point(len(s.job.prompt)) if self.store is not None else None
                s.job.stats.update(cached=0, tier="none")
        for slot, exc in res.errors.items():
            s = self.seqs[slot]
            if s is not None:
                s.job.out.put(exc)
                self._end(s, None, cancelled=True)
        if res.cand is None:
            return
        for r, c in zip(res.rows, take_rows(res.cand, res.rows)):
            s = self.seqs[r.slot]
            if s is None:
                continue
            positions = [r.start + 1 + i for i in range(len(r.tokens))]
            chosen = self._choose(c, positions, s.job.sampling)
            acc = accepted(r.tokens, chosen)
            out = chosen[:acc + 1]
            done = False
            if s.job.stop_eos:
                for i, t in enumerate(out):
                    if t in self.eos:
                        out, done = out[:i + 1], True
                        break
            left = s.job.max_tokens - len(s.out)
            if len(out) >= left:
                out, done = out[:left], True
            self.commits.append((r.slot, len(out) - 1, out[-1]))
            s.pos, s.pending = r.start + len(out), out[-1]
            if not s.out:
                s.job.stats["ttft_s"] = round(time.monotonic() - s.job.submitted, 4)
            s.out += out
            self.counts["rows"] += len(r.tokens)
            self.counts["drafts_kept"] += len(out) - 1
            s.job.out.put(list(out))
            if done:
                self._end(s, None, cancelled=False)

    def _choose(self, c, positions: list[int], sampling) -> list[int]:
        values, ids = c.values.astype(np.float32), c.ids.astype(np.int64)
        if sampling is None:                    # greedy: the largest logit, ties to the lower id
            out = []
            for r in range(ids.shape[0]):
                order = np.lexsort((ids[r], -values[r]))
                out.append(int(ids[r][order[0]]))
            return out
        return choose_rows(values, ids, positions, sampling)

    def _end(self, s: Seq, plan: Plan | None, *, cancelled: bool) -> None:
        save = self.store is not None and not cancelled
        if plan is not None:
            plan.finishes.append((s.slot, save, cancelled))
        else:
            self.finishes.append((s.slot, save, cancelled))
        self.seqs[s.slot] = None
        dt = time.monotonic() - s.t0
        s.job.stats.update(finish="cancelled" if cancelled else "done", completion=len(s.out),
                           decode_s=round(dt, 4), rounds=self.rounds)
        s.job.out.put(None)

    def _fail_all(self, exc: BaseException) -> None:
        for s in list(self.seqs):
            if s is not None:
                s.job.out.put(RuntimeError(f"{type(exc).__name__}: {exc}"))
                self._end(s, None, cancelled=True)
        self.commits = []

    # -- rank 1 ----------------------------------------------------------------------------------------------------
    def follow(self) -> None:
        """Rank 1: execute rank 0's plans until a stop plan."""

        while True:
            plan = Plan.decode(self.link.recv())
            if plan.stop:
                return
            self.ex.run(plan)
            self.rounds += 1

    def send_stop(self) -> None:
        self.link.send(Plan(stop=True).encode())

    def describe(self) -> str:
        s = f"{self.n} slot(s) x {self.capacity} tokens; {self.ex.describe()}"
        if self.store is not None:
            s += f"; sessions: {self.store.describe()}"
        return s
