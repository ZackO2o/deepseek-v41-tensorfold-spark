# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""The per-node memory budget of DeepSeek-V4.1-Flash on 2 Sparks (TP=2), as code: docs/ENGINE-PLAN.md's tables are
``python -m engine.serving.memory``'s output (a copy of the engine's ``tensorfold.families.deepseek_v41.cuda.memory``),
and the engine's admission uses the same terms.

Inputs, per rank, GiB:

- **weights**: ``mia29`` from the kit's ``scripts/weight_budget.py`` on the 2.9 bpw pack (measured from the headers:
  DSV41-BASELINE.md section 1.3); ``exp30`` = the same pack with every routed expert at 3.0 bits (layers 18-22
  too: +3.63, from the trellis bytes); ``cool30`` = coolbho3k's layout (EXL3 mul1 3.0 for routed and DSpark experts,
  every other matrix in the source FP8 / BF16 format): an estimate from parameter counts, +-1 GiB;
- **pool**: ``pool.pool_bytes`` (FP8 rows of the 4 kv sources + their index keys) for the active streams;
- **rings**: ``state.slot_bytes`` a slot;
- **workspace**: CUDA graphs, prefill chunk buffers (sub-blocked: experts and attention in 512-row blocks, the
  indexer's selection scratch capped), DSpark, sampler: sized, not grown (the 0550 rule);
- **runtime**: CUDA context, NCCL + RoCE buffers, the Python process.

``available`` is MemAvailable after the start's page-cache drop with the OS, docker and desktop running: 112 GiB is
the kit's measured preflight point (weights + 12 GiB of its 111.5 GiB check, both nodes); worker has ~1 GiB more.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from .pool import pool_bytes
from .state import slot_bytes
from .topology import Topology

GiB = float(1 << 30)
AVAILABLE = 112.0

PACKS = {
    # name: (per-rank GiB on rank 0 (vision on rank 0), what it is)
    "mia29": (99.48, "Mia / dealignai 2.9 bpw EXL3 mul1 (measured: weight_budget.py)"),
    "exp30": (99.48 + 3.63, "the same pack with routed experts at 3.0 bits on every layer (computed)"),
    "cool30": (104.6, "coolbho3k-style: 3.0 bpw experts only, the rest source FP8 / BF16 (estimated +-1)"),
}
VISION = 0.90                  # replicated in the kit; ours on rank 0 only


@dataclass(frozen=True)
class Workspace:
    graphs: float = 1.2        # decode / verify graphs for widths 1-16 x slot mixes, DSpark graphs
    prefill: float = 1.3       # a 2,048-row chunk: mHC streams, q / attention out, expert scratch (512-row blocks),
                               # attention partials (512-row blocks), Engram staging
    select: float = 0.25       # the indexer's scores + keys, blocks of rows (GLM 0065's SELECT_MB rule)
    drafting: float = 0.35     # DSpark pass buffers, sampler, logits (vocab half x 16 rows), lookup index
    slack: float = 0.9         # allocator fragmentation and the largest transient (measured on GLM: 0.6-1.0)

    @property
    def total(self) -> float:
        return self.graphs + self.prefill + self.select + self.drafting + self.slack


RUNTIME = 1.0


@dataclass(frozen=True)
class Budget:
    pack: str
    rank: int
    streams: int
    context: int
    index_kv: str
    weights: float
    pool: float
    rings: float
    workspace: float
    runtime: float
    available: float

    @property
    def used(self) -> float:
        return self.weights + self.pool + self.rings + self.workspace + self.runtime

    @property
    def floor(self) -> float:
        return self.available - self.used

    def row(self) -> str:
        return (f"| {self.pack} | r{self.rank} | {self.streams} x {self.context // 1000}K | {self.index_kv} | "
                f"{self.weights:.2f} | {self.pool:.2f} | {self.rings:.2f} | {self.workspace:.2f} | {self.runtime:.2f} | "
                f"{self.used:.2f} | **{self.floor:.2f}** |")


def budget(topo: Topology, pack: str = "mia29", *, rank: int = 0, streams: int = 4, context: int = 300_000,
           index_kv: str = "bf16", workspace: Workspace | None = None, available: float = AVAILABLE) -> Budget:
    workspace = workspace or Workspace()
    w = PACKS[pack][0] - (VISION if rank == 1 else 0.0)
    return Budget(pack, rank, streams, context, index_kv, w, pool_bytes(topo, streams, context, index_kv) / GiB,
                  streams * slot_bytes(topo).total / GiB, workspace.total, RUNTIME, available)


def max_streams(topo: Topology, pack: str, context: int, floor: float, **kw) -> int:
    """The most active streams at ``context`` that keep ``floor`` GiB on the tighter rank (rank 0)."""

    s = 0
    while budget(topo, pack, streams=s + 1, context=context, **kw).floor >= floor:
        s += 1
        if s > 64:
            break
    return s


def table(topo: Topology) -> str:
    rows = ["| pack | rank | active | index keys | weights | pool | rings | workspace | runtime | used | floor |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for pack in PACKS:
        for streams, ctx in ((4, 300_000), (2, 300_000), (1, 300_000)):
            for ikv in ("bf16", "fp8"):
                if pack != "mia29" and ikv == "fp8":
                    continue
                rows.append(budget(topo, pack, streams=streams, context=ctx, index_kv=ikv).row())
    rows.append(budget(topo, "mia29", rank=1).row())
    return "\n".join(rows)


# -- the measured budget: G2 / G3 anchors (ENGINE-PLAN section 5, 2026-10-02) ------------------------------------------
@dataclass(frozen=True)
class Anchors:
    """GiB a rank, from the 0.5 s samplers and the boot lines of G2 / G3 (TP=2, the 2.9 bpw pack, prepared folders).

    - ``start``: MemAvailable at the process start after a cache drop (G3: head 116.7-117.2, worker 115.1-115.3);
    - ``runtime``: memory outside the caching allocator once the forward is built = start - MemAvailable(built) -
      allocated (CUDA context, NCCL / RoCE buffers, kernel modules, the process). G3 with the drafter: head 5.7,
      worker 5.4 (G2 without it: 4.7 / 4.9). The plan had 1.0;
    - ``serving``: what G3's eager 4 x 16K bench used beyond the expert scratches, pool and rings (activations, the
      CSA2 attention scratch, the session RAM tier, host growth): built -> bench minimum 4.7 GiB on both ranks, less
      2.68 GiB of per-layer expert scratches (40 x 68.7 MiB at 128 rows), the 4 x 16K pool and the rings;
    - ``weights`` 95.4 allocated on both ranks (text model; no vision tower is loaded); ``drafter`` 3.6 (99.0 -
      95.4)."""

    start: float
    runtime: float
    serving: float = 1.8
    weights: float = 95.4
    drafter: float = 3.6


ANCHORS = {0: Anchors(117.0, 5.7), 1: Anchors(115.1, 5.4)}
GRAPHS = 0.4          # graph pool: one shared pool, boot captures one slot at widths 1-6 (TF_DSV41_GRAPH_WARM=decode),
                      # and no capture starts under TF_DSV41_GRAPH_FLOOR_GIB (5) MemAvailable (e2a2333). Estimate; G3's
                      # 3.8 GiB dip was per-key pools + per-layer scratches regrown inside captures (both fixed)
GRAPHS_WORST = 1.2    # the plan's budget, the table's pessimistic column
ENGRAM_ROW = 600      # host bytes a cached Engram row: the 264-byte record, its numpy object and the LRU entry
CHUNK_ACT = 0.3       # a 2,048-row prefill chunk's activations beyond G3's 128-row windows (estimate)
SELECT = 0.25         # the indexer's selection blocks at a 300K position (GLM 0065's 256 MiB cap)


def engram_cache_gib() -> float:
    """The Engram row prefetch's LRU (beff78c): TF_DSV41_ENGRAM_CACHE rows a layer (65,536) x 2 layers, ~600 host bytes
    a row, + the 16 reader threads' stacks and the RoCE shards (<= 1 MiB each): ~0.1 GiB."""

    rows = int(os.environ.get("TF_DSV41_ENGRAM_CACHE", "65536") or 0)
    return (2 * rows * ENGRAM_ROW + 16 * (1 << 20) + 16 * (1 << 20)) / GiB


def expert_scratch_gib(drafter: bool = True) -> float:
    """The shared expert scratches (``moe.scratch_for``): the target's block (1,024 rows) + DSpark's (64)."""

    try:
        from . import experts as X
        from . import moe

        b = X.scratch_bytes(moe.expert_block(X.MODEL.slots), X.MODEL)
        if drafter:
            b += X.scratch_bytes(moe.expert_block(X.DSPARK.slots, draft=True), X.DSPARK)
        return b / GiB
    except ImportError:                                 # the staging copy (engine/serving): the default blocks
        return 0.57 if drafter else 0.55


@dataclass(frozen=True)
class Measured:
    rank: int
    streams: int
    context: int
    index_kv: str
    terms: dict

    @property
    def floor(self) -> float:
        return self.terms["start"] - sum(v for k, v in self.terms.items() if k != "start")

    def row(self) -> str:
        t = self.terms
        cells = " | ".join(f"{t[k]:.2f}" for k in ("weights", "drafter", "runtime", "experts", "serving", "pool",
                                                    "graphs", "chunk", "select", "sessions"))
        return (f"| r{self.rank} | {self.streams} x {self.context // 1000}K | {self.index_kv} | {t['start']:.1f} | "
                f"{cells} | **{self.floor:.2f}** |")


def measured(topo: Topology, *, rank: int = 1, streams: int = 4, context: int = 300_000, index_kv: str = "bf16",
             drafter: bool = True, graphs: float = GRAPHS, pack: str = "mia29", session_ram_gib: float = 0.25,
             anchors: dict | None = None) -> Measured:
    """The worst phase (a 300K prefill while the others decode) projected from the anchors."""

    a = (anchors or ANCHORS)[rank]
    w = a.weights + (PACKS[pack][0] - PACKS["mia29"][0])
    terms = {"start": a.start, "weights": w, "drafter": a.drafter if drafter else 0.0,
             "runtime": a.runtime, "experts": expert_scratch_gib(drafter), "serving": a.serving,
             "pool": (pool_bytes(topo, streams, context, index_kv) + streams * slot_bytes(topo).total) / GiB,
             "graphs": graphs, "chunk": CHUNK_ACT, "select": SELECT * min(1.0, context / 300_000),
             "sessions": session_ram_gib + engram_cache_gib()}
    return Measured(rank, streams, context, index_kv, terms)


def fit_context(topo: Topology, floor: float, *, rank: int = 1, streams: int = 4, step: int = 4096, **kw) -> int:
    """The longest context a stream (a multiple of ``step``) whose projected floor on ``rank`` stays >= ``floor``."""

    ctx = 0
    while measured(topo, rank=rank, streams=streams, context=ctx + step, **kw).floor >= floor and ctx < 2_000_000:
        ctx += step
    return ctx


def measured_table(topo: Topology) -> str:
    rows = ["| rank | active | index keys | start | weights | drafter | runtime | experts | serving | pool + rings | "
            "graphs | 2K chunk | select | sessions + Engram cache | **floor** |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for rank in (1, 0):
        for ctx in (300_000, 256_000, 196_608):
            for ikv in ("bf16", "fp8"):
                rows.append(measured(topo, rank=rank, context=ctx, index_kv=ikv).row())
    fits = []
    for ikv in ("bf16", "fp8"):
        for g in (GRAPHS_WORST, GRAPHS):
            for f in (5.0, 4.0):
                fits.append(f"4 streams, worker >= {f} GiB, {ikv} keys, graphs {g} GiB: "
                            f"{fit_context(topo, f, index_kv=ikv, graphs=g) // 1000}K tokens a stream")
    return "\n".join(rows + [""] + fits)


# -- the floor at run time (GLM 0550's memsafe accounting) -------------------------------------------------------------
FLOOR_ENV, HARD_ENV = "TF_DSV41_FLOOR_GIB", "TF_DSV41_FLOOR_HARD_GIB"
FREE_ENV = "TF_DSV41_ADMIT_FREE_FLOOR_GIB"


def free_floor_gib() -> float:
    """TF_DSV41_ADMIT_FREE_FLOOR_GIB (default 0): least *immediately* free memory (MemFree + the allocator's unused
    cache) an admission needs on top of the floor. GLM's 1 GiB rule is for a stack that allocates a chunk's buffers at
    admission; here the pool pages and the workspace exist already, so reclaimable page cache counts in full. With
    1 GiB, G3's cold boot waited 253 s at MemFree 0.84-0.99 GiB beside 6.7 GiB of clean page cache that nothing
    reclaimed (the kernel frees it under pressure only, and a waiting batcher makes none)."""

    v = float(os.environ.get(FREE_ENV, "0") or 0)
    if v < 0:
        raise ValueError(f"{FREE_ENV}={v}: expected >= 0")
    return v


def allocator_cached() -> int:
    """The caching allocator's reserved-but-unused bytes on this rank's GPU (0 without CUDA)."""

    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return max(0, int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated()))
    except Exception:                                   # noqa: BLE001 - accounting must never stop serving
        return 0


def floor_settings() -> tuple[float, float]:
    """(target, hard) floor in GiB: TF_DSV41_FLOOR_GIB (5) and TF_DSV41_FLOOR_HARD_GIB (4), 0 < hard <= target."""

    target = float(os.environ.get(FLOOR_ENV, "5") or 5)
    hard = float(os.environ.get(HARD_ENV, "4") or 4)
    if not 0 < hard <= target:
        raise ValueError(f"{HARD_ENV}={hard} / {FLOOR_ENV}={target}: expected 0 < hard <= target")
    return target, hard


@dataclass
class Floor:
    """MemAvailable kept above a floor by accounting (ENGINE-PLAN section 5): every buffer is sized at load, so at run
    time only the session store's RAM tier (bounded state, ~7 MB an entry) and stray growth move memory. ``check``
    before an admission: ``usable`` (GLM ``memsafe.view``: MemFree + allocator cache + the page cache the kernel can
    hand over: clean, unmapped file pages) less ``need`` must stay at or above the hard floor (4 GiB), else nothing
    new starts and admission waits (logged, GLM ``AdmitLog``); no immediately-free minimum by default
    (``free_floor_gib``: page cache is reclaimed when an allocation needs it, never while the batcher waits); the
    target (5 GiB) is what the load-time budget plans for (``load_check``). On a GB10 the GPU allocates from the
    same memory, so MemFree is the CUDA free figure (unified memory)."""

    target_gib: float = 5.0
    hard_gib: float = 4.0
    meminfo: Callable[[], dict] | None = None        # tests inject /proc/meminfo
    cached: Callable[[], int] | None = None          # the caching allocator's unused bytes (torch.cuda)
    free_gib: float = 0.0                            # TF_DSV41_ADMIT_FREE_FLOOR_GIB: immediately free on top
    log: object = None
    stats: dict = field(default_factory=lambda: {"checks": 0, "waits": 0, "low": 0})

    @classmethod
    def from_env(cls, **kw) -> Floor:
        target, hard = floor_settings()
        kw.setdefault("cached", allocator_cached)
        kw.setdefault("free_gib", free_floor_gib())
        return cls(target, hard, **kw)

    def view(self):
        from tensorfold.families.glm5_next.spark import memsafe

        mi = self.meminfo() if self.meminfo is not None else memsafe.read_meminfo()
        cached = self.cached() if self.cached is not None else 0
        return memsafe.view(mi.get("MemFree", 0), cached, "available", mi)

    def check(self, need: int = 0, *, queued: int = 0) -> bool:
        """Whether something new may start: ``usable`` less ``need`` stays at or above the hard floor and GLM's
        immediately-free floor holds (``memsafe.admit_ok``). Under the target it starts anyway (the pool and the
        workspace are allocated already: waiting would free nothing) and says so once an episode."""

        from tensorfold.families.glm5_next.spark import memsafe

        self.stats["checks"] += 1
        v = self.view()
        ok = memsafe.admit_ok(v, 0, int(need) + int(self.hard_gib * GiB), floor=int(self.free_gib * GiB))
        if self.log is None:
            self.log = memsafe.AdmitLog()
        if not ok:
            self.stats["waits"] += 1
            self.log.waiting(v, 0, int(need) + int(self.hard_gib * GiB), queued)
        else:
            self.log.admitted()
            if v.usable - int(need) < self.target_gib * GiB:
                self.stats["under_target"] = self.stats.get("under_target", 0) + 1
        return ok

    def low(self) -> bool:
        """Below the hard floor now (status and /health; admission already waits there)."""

        low = self.view().usable < self.hard_gib * GiB
        self.stats["low"] += int(low)
        return low


# -- what a rank holds, and the host memory a boot leaves behind ------------------------------------------------------
def _status_kib(key: str) -> int:
    try:
        for line in open("/proc/self/status"):
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def snapshot() -> dict:
    """GiB: MemAvailable / MemFree / Cached / Mapped / AnonPages / Shmem (/proc/meminfo), this process's VmRSS, the
    allocator's allocated / reserved, the shared expert scratches. G2 left ~3 GiB of the worker's drop at warm-up and
    ~3 GiB more over the gate run outside the allocator: these terms attribute it (G4)."""

    from tensorfold.families.glm5_next.spark import memsafe

    mi = memsafe.read_meminfo()
    out = {k: round(mi.get(k, 0) / GiB, 2) for k in ("MemAvailable", "MemFree", "Cached", "Mapped", "AnonPages",
                                                      "Shmem")}
    out["VmRSS"] = round(_status_kib("VmRSS") / 2 ** 20, 2)
    try:
        import torch

        if torch.cuda.is_available():
            out["allocated"] = round(torch.cuda.memory_allocated() / GiB, 2)
            out["reserved"] = round(torch.cuda.memory_reserved() / GiB, 2)
        from . import moe

        out["expert_scratch"] = round(moe.scratch_bytes_total() / GiB, 2)
    except Exception:                                   # noqa: BLE001 - a report
        pass
    return out


def host_trim() -> float:
    """After the boot (and a folder build): Python garbage, glibc's free arenas (``malloc_trim``: the Triton / extension
    compiles and the loaders' transient buffers) and torch's cached pinned blocks back to the system; on a GB10 that
    memory is the GPU's. -> GiB of VmRSS given back."""

    import ctypes
    import gc

    before = _status_kib("VmRSS")
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    try:
        import torch

        torch._C._host_emptyCache()
    except (ImportError, AttributeError, RuntimeError):
        pass
    return max(0, before - _status_kib("VmRSS")) / 2 ** 20


def load_check(topo: Topology, pack: str = "mia29", *, rank: int = 0, streams: int = 4, context: int = 300_000,
               index_kv: str = "bf16", session_ram_gib: float = 0.5, available: float = AVAILABLE) -> Budget:
    """The load-time budget with the session store's RAM tier: raises when the floor would fall under the hard floor
    (start fewer streams, a shorter context, or a smaller pack)."""

    target, hard = floor_settings()
    b = budget(topo, pack, rank=rank, streams=streams, context=context, index_kv=index_kv,
               available=available - session_ram_gib)
    if b.floor < hard:
        raise ValueError(f"{streams} x {context} on rank {rank} leaves {b.floor:.2f} GiB, under the {hard} GiB hard "
                         f"floor ({HARD_ENV}); fit: {max_streams(topo, pack, context, hard)} stream(s)")
    if b.floor < target:
        print(f"[tensorfold] memory: floor {b.floor:.2f} GiB is under the {target} GiB target ({FLOOR_ENV})",
              flush=True)
    return b


if __name__ == "__main__":
    import sys

    from .topology import load

    topo = load(sys.argv[1])
    print(table(topo))
    print()
    print(measured_table(topo))
