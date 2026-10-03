"""DeepSeek-V4.1-Flash's routed experts (and DSpark's) on TensorFold 0.6.0's EXL3 module, TP=2.

What this checkpoint has (``dsv41-uncensored-2.9bpw`` headers, docs/ENGINE-PLAN.md section 3):

- ``layers.N.ffn.experts.E.{w1,w3,w2}``: 384 experts, gate / up [5,120 -> 2,304], down [2,304 -> 5,120]; mul1
  codebook, 3 bits (trellis [320, 144, 48]) and 2 bits on layers 18-22 ([320, 144, 32]).
- ``layers.N.ffn.shared_experts.{w1,w3,w2}``: one shared expert of the same shape at 5 / 4 bits.
- ``mtp.N.ffn.experts.E.*``: DSpark's 3 blocks x 128 experts (top-3) at 4 bits ([320, 144, 64]) + a shared expert.
- Router: ``gate.weight`` fp16 [384, 5,120], ``gate.bias`` fp16 [384]; sqrt-softplus scores, top-6 by score + bias,
  renormalized, x 1.5 (``engine.kernels.router``). SwiGLU clamped at 10 (gate from above, up both ways), fp32.

TP=2 (a rank holds half of every expert's intermediate width, 1,152 = 9 Hadamard blocks of 128):

- gate / up: trellis columns (N tiles) [rank * 72, +72), ``svh`` columns likewise, ``suh`` whole;
- down: trellis rows (K tiles) [rank * 72, +72), ``suh`` rows likewise, ``svh`` whole;
- every rank computes a partial [R, 5,120] fp32 (its half of the intermediate width), the ranks exchange them
  (``tensorfold.cuda.comm.fast_gather``) and add them in rank order. EXL3's Hadamard rotations act on blocks of 128
  along K (input) and N (output), so a split on a 128 boundary keeps every block whole: a rank's matrix is exactly the
  slice of the full one (``tests/kernels/test_exl3_experts.py`` checks it with upstream's numpy decoder).

The shared expert rides in the routed launch as expert index E (one more table entry, its own width; every row picks
it with weight 1.0, after the routed slots). That keeps one grouped launch a projection in decode instead of three
dense calls; its arithmetic is the routed experts' (fp32 SwiGLU, the fixed-order combine).

Upstream's ``routed`` sequence is mirrored here only to put ``loads.grouped`` (patch 0580's load path) in front of
the two ``ext.grouped`` launches; with the knob off the calls are upstream's, argument for argument (tested).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import loads, prefill

DIMS = 5120
INTER = 2304
EXPERTS = 384
TOPK = 6
DSPARK_EXPERTS = 128
DSPARK_TOPK = 3
SWIGLU_LIMIT = 10.0
ROUTED_SCALE = 1.5
HAD = 128
CODEBOOK = "mul1"

GATE, UP, DOWN = "w1", "w3", "w2"


@dataclass(frozen=True)
class Geometry:
    """One MoE block's shape on one rank."""

    dims: int = DIMS
    inter: int = INTER
    experts: int = EXPERTS
    topk: int = TOPK
    shared: int = 1
    world: int = 2

    def __post_init__(self) -> None:
        if self.inter % (self.world * HAD) or self.dims % HAD:
            raise ValueError(f"intermediate {self.inter} must split into {self.world} whole Hadamard blocks of {HAD}")

    @property
    def width(self) -> int:
        """The intermediate width a rank holds."""

        return self.inter // self.world

    @property
    def slots(self) -> int:
        """Expert picks a row: the routed top-k, then the shared expert(s)."""

        return self.topk + self.shared

    @property
    def table(self) -> int:
        """Entries of the expert table: routed experts, then the shared one(s) (index ``experts``)."""

        return self.experts + self.shared


MODEL = Geometry()
DSPARK = Geometry(experts=DSPARK_EXPERTS, topk=DSPARK_TOPK)


def k2_of(trellis_shape: Sequence[int]) -> int:
    """Half-bits a value of a trellis [K/16, N/16, 16 * bits] (3 bits -> 6)."""

    w = int(trellis_shape[-1])
    if w % 8 or not 2 <= w // 8 <= 16:
        raise ValueError(f"trellis last dim {w}: not an EXL3 width")
    return w // 8


# -- tensor parallel slices ----------------------------------------------------------------------------------------
def split_triple(trellis, suh, svh, kind: str, rank: int, world: int):
    """One matrix's (trellis, suh, svh) for ``rank``: ``kind`` "col" (gate / up: split N) or "row" (down: split K).

    Works on torch tensors and numpy arrays alike (slices + a contiguous copy)."""

    kt, nt = int(trellis.shape[0]), int(trellis.shape[1])
    if kind == "col":
        if (nt * 16) % (world * HAD):
            raise ValueError(f"N={nt * 16} does not split into {world} whole blocks of {HAD}")
        per = nt // world
        t = trellis[:, rank * per:(rank + 1) * per]
        return _contig(t), suh, _contig(svh[rank * per * 16:(rank + 1) * per * 16])
    if kind == "row":
        if (kt * 16) % (world * HAD):
            raise ValueError(f"K={kt * 16} does not split into {world} whole blocks of {HAD}")
        per = kt // world
        t = trellis[rank * per:(rank + 1) * per]
        return _contig(t), _contig(suh[rank * per * 16:(rank + 1) * per * 16]), svh
    raise ValueError(f"kind must be 'col' or 'row', not {kind!r}")


def _contig(t):
    if hasattr(t, "contiguous"):
        return t.contiguous()
    import numpy as np

    return np.ascontiguousarray(t)


def split_expert(mats: Mapping[str, tuple], rank: int, world: int) -> dict[str, tuple]:
    """{w1, w3, w2: (trellis, suh, svh)} of one expert -> the same for ``rank``."""

    return {GATE: split_triple(*mats[GATE], "col", rank, world), UP: split_triple(*mats[UP], "col", rank, world),
            DOWN: split_triple(*mats[DOWN], "row", rank, world)}


# -- a layer's plan from headers alone ----------------------------------------------------------------------------
@dataclass
class LayerPlan:
    """Widths and bytes of one MoE block on one rank (from trellis shapes; no weight data)."""

    geom: Geometry
    k2: dict[str, list[int]]           # w1 / w3 / w2 -> half-bits per table entry (routed, then shared)

    @property
    def k2_gu(self) -> tuple[int, int]:
        v = self.k2[GATE] + self.k2[UP]
        return min(v), max(v)

    @property
    def k2_d(self) -> tuple[int, int]:
        return min(self.k2[DOWN]), max(self.k2[DOWN])

    def entry_bytes(self, e: int) -> int:
        """Trellis bytes of table entry ``e`` on this rank (gate + up + down)."""

        g = self.geom
        per = g.dims * g.width // 256          # 16x16 tiles a matrix on this rank, 16 * K2 bytes a tile
        return per * 16 * (self.k2[GATE][e] + self.k2[UP][e] + self.k2[DOWN][e])

    def rank_bytes(self) -> int:
        return sum(self.entry_bytes(e) for e in range(self.geom.table))

    def ld_instance(self) -> dict[str, tuple[int, int] | None]:
        """The x3ld instance each launch uses (None: upstream's kernel only)."""

        return {"gu": loads.k2_range(*self.k2_gu), "dn": loads.k2_range(*self.k2_d)}


def plan_layer(shapes: Mapping[str, Sequence[int]], geom: Geometry = MODEL, prefix: str = "") -> LayerPlan:
    """A block's plan from its trellis shapes: ``shapes[f"{prefix}experts.{e}.{w}.trellis"]`` and
    ``shapes[f"{prefix}shared_experts.{w}.trellis"]`` (full, unsplit shapes as stored)."""

    k2: dict[str, list[int]] = {GATE: [], UP: [], DOWN: []}
    names = [f"{prefix}experts.{e}." for e in range(geom.experts)]
    names += [f"{prefix}shared_experts."] * geom.shared
    for n in names:
        for w in (GATE, UP, DOWN):
            shp = tuple(shapes[f"{n}{w}.trellis"])
            want = (geom.dims // 16, geom.inter // 16) if w != DOWN else (geom.inter // 16, geom.dims // 16)
            if shp[:2] != want:
                raise ValueError(f"{n}{w}.trellis {list(shp)}: expected [{want[0]}, {want[1]}, *]")
            k2[w].append(k2_of(shp))
    return LayerPlan(geom, k2)


# -- tile settings -------------------------------------------------------------------------------------------------
def configs(geom: Geometry = MODEL) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Upstream's tile settings (n tiles, warps, K splits, tiles in flight) for gate/up and down at this shape: a
    function of the shape alone, so a row's reduction never depends on its window."""

    from tensorfold.cuda.exl3.experts import default_config

    return default_config(geom.dims, geom.width, True), default_config(geom.width, geom.dims, False)


def configs_host(geom: Geometry = MODEL) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """``configs`` without importing torch (upstream's candidate rule, verbatim; tests compare the two)."""

    def pick(K: int, N: int, gateup: bool):
        cands = [(8, 4, 4, 1) if gateup else (8, 4, 1, 1), (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
        for nt, w, sk, pf in cands:
            if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
                return nt, w, sk, pf
        raise ValueError(f"no tile setting divides K={K}, N={N}")

    return pick(geom.dims, geom.width, True), pick(geom.width, geom.dims, False)


def scratch_bytes(rows: int, geom: Geometry = MODEL) -> int:
    """Bytes of upstream's ``Scratch`` for windows of ``rows`` rows (P = rows x slots picks)."""

    (_, _, sk_gu, _), (_, _, sk_d, _) = configs_host(geom)
    P = rows * geom.slots
    D, I = geom.dims, geom.width
    maxu = min(P, geom.table)
    return (2 * P * D * 2 + P * I * 2 + max(2 * sk_gu * I, sk_d * D) * P * 4 + P * D * 4 + maxu * 4 + 4
            + maxu * rows * 4)


def prefill_block_rows(budget_bytes: int, geom: Geometry = MODEL, cap: int = 2048) -> int:
    """The largest power-of-two row block (<= cap) whose expert scratch fits ``budget_bytes``: prompt chunks run
    their expert pass in blocks of this many rows (rows are independent, so the bits do not depend on it)."""

    r = cap
    while r > 16 and scratch_bytes(r, geom) > budget_bytes:
        r //= 2
    return r


# -- on the GPU -----------------------------------------------------------------------------------------------------
@dataclass
class Experts:
    """A rank's MoE block: upstream's ``Exl3RoutedExperts`` (routed + shared entries) and its plan."""

    ex: Any                       # tensorfold.cuda.exl3.experts.Exl3RoutedExperts
    geom: Geometry
    keep: list = field(default_factory=list, repr=False)


def prepare_layer(routed: Sequence[Mapping[str, tuple]], shared: Sequence[Mapping[str, tuple]], rank: int,
                  geom: Geometry = MODEL, device="cuda") -> Experts:
    """A block from per-expert {w1, w3, w2: (trellis, suh, svh)} (full, as stored): split for ``rank``, moved to the
    device, the shared expert(s) appended as table entries ``geom.experts``..."""

    import torch

    from tensorfold.cuda.exl3 import experts as x3

    if len(routed) != geom.experts or len(shared) != geom.shared:
        raise ValueError(f"{len(routed)} routed + {len(shared)} shared experts, expected {geom.experts} + "
                         f"{geom.shared}")
    gate, up, down = [], [], []
    for mats in [*routed, *shared]:
        part = split_expert(mats, rank, geom.world)
        for w, dst in ((GATE, gate), (UP, up), (DOWN, down)):
            t, su, sv = (torch.as_tensor(a).to(device).contiguous() for a in part[w])
            if t.data_ptr() % 16:
                raise ValueError("trellis must be 16-byte aligned (the load path reads 16-byte words)")
            dst.append((t, su, sv))
    return Experts(x3.prepare(gate, up, down, CODEBOOK, device=device), geom)


def scratch(layer: Experts, rows: int, device="cuda"):
    """Upstream's ``Scratch`` for windows of up to ``rows`` rows with this block's slots and tile settings."""

    from tensorfold.cuda.exl3 import experts as x3

    gu, dn = configs(layer.geom)
    return x3.Scratch(layer.ex, rows, layer.geom.slots, gu, dn, device=device)


def routed(x, pick, wts, layer: Experts, s, out, R: int, *, limit: float = SWIGLU_LIMIT, group: bool = True):
    """Upstream's ``experts.routed`` (fp32 SwiGLU, ``act_mode`` ACT_F32), with ``prefill.grouped`` (windows of
    many rows) then ``loads.grouped`` tried first for the two grouped launches (all three: the same Z). pick [R, slots] int32 (routed ids, then ``geom.experts`` for the shared expert),
    wts [R, slots] fp32 (routed weights already x 1.5, the shared slot 1.0). Returns this rank's partial."""

    import torch

    from tensorfold.cuda.exl3 import experts as x3

    ex = layer.ex
    ext = x3._ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    if R > s.rows:
        raise ValueError(f"{R} rows but the scratch holds {s.rows}")
    ids, members = s.window(R)
    if group:
        ext.group(pick, ids, s.count, members, R, slots, E)
    ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E)
    nt, w, sk, pf = s.cfg_gu
    if not (prefill.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2,
                            D, I, P, sk, slots, ex.cb, w, ex.k2_gu[0], ex.k2_gu[1], rows=R)
            or loads.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2,
                             D, I, P, sk, slots, ex.cb, w, ex.k2_gu[0], ex.k2_gu[1])):
        ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                    P, sk, slots, ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1])
    ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, sk, slots, E, float(limit),
                        x3.ACT_F32)
    nt, w, sk, pf = s.cfg_d
    if not (prefill.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members,
                            s.z, 1, I, D, P, sk, slots, ex.cb, w, ex.k2_d[0], ex.k2_d[1], rows=R)
            or loads.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members,
                             s.z, 1, I, D, P, sk, slots, ex.cb, w, ex.k2_d[0], ex.k2_d[1])):
        ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                    D, P, sk, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1])
    if wts is None:
        ext.down_epilogue(s.z, pick, ex.svh_d, s.y, R, P, D, sk, slots, E)
        return s.y[:P]
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    ext.down_combine(s.z, pick, ex.svh_d, s.y, wts, out, R, P, D, sk, slots, E)
    return out


def with_shared(pick, wts, geom: Geometry = MODEL):
    """Routed picks [R, topk] / weights -> [R, slots] with the shared entries appended (weight 1.0); the router
    kernel writes this layout directly, this is the torch form for tests and the reference."""

    import torch

    R = pick.shape[0]
    sp = torch.arange(geom.experts, geom.table, dtype=pick.dtype, device=pick.device).expand(R, geom.shared)
    sw = torch.ones((R, geom.shared), dtype=wts.dtype, device=wts.device)
    return torch.cat([pick, sp], 1).contiguous(), torch.cat([wts, sw], 1).contiguous()


def decode_bytes(plan: LayerPlan, distinct: float) -> float:
    """Expected trellis bytes a verify window reads in one block on one rank: ``distinct`` routed experts (mean
    entry bytes) plus every shared entry."""

    g = plan.geom
    routed = sum(plan.entry_bytes(e) for e in range(g.experts)) / g.experts
    shared = sum(plan.entry_bytes(e) for e in range(g.experts, g.table))
    return distinct * routed + shared


def distinct_experts(rows: int, experts: int = EXPERTS, topk: int = TOPK, overlap: float = 1.0) -> float:
    """Distinct experts a window of ``rows`` rows touches if picks were independent and uniform (``overlap`` < 1
    shrinks it toward measured locality): E (1 - (1 - k/E)^rows) x overlap, at least k."""

    u = experts * (1.0 - (1.0 - topk / experts) ** rows)
    return max(float(topk), u * overlap) if rows else 0.0


def gbps(nbytes: float, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9 if ms > 0 else math.inf
