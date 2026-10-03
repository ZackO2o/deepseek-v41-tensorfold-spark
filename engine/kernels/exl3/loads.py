"""The expert load path for DeepSeek-V4.1-Flash's routed experts (``x3ld.cu``): our GLM patch 0580 over TensorFold
0.6.0's grouped EXL3 expert kernel, with upstream's bits.

Knobs (load-time, each rank reads its own; ranks may differ without changing a bit):

``TF_DSV41_EXPERT_LOADS=1``      the two grouped launches of a routed layer (gate/up, down) run ``x3ld``'s
                                 ``ld_kernel`` instead of upstream's ``grouped_kernel`` (default 0 until the GPU gate).
``TF_DSV41_EXPERT_LOADS_CFG``    "nt,pd" for both, or "nt,pd/nt,pd" (gate/up, down); default "8,1" (GLM's adopted
                                 setting, W19). nt: column tiles a program (8 = upstream's), pd: k steps in flight a
                                 warp (1, 2). Every setting places work only.
``TF_DSV41_EXPERT_LOADS_PDL=1``  launch as a programmatic dependent (sm_90+): the prologue overlaps the previous
                                 launch's tail (rot_in / gateup_epilogue, plain launches).

Exactness: the grid, each warp's K range, the mma chain per accumulator (k ascending from +0.0, upstream's
``decode_tile`` B fragments, ``load_pair``'s A values), the warp sum order and the Z rows are upstream's; only when the
bytes arrive changes. ``tests/kernels/test_exl3_loads_emulator.py`` checks the lane mapping and the ring schedule for
every width and setting, ``tests/kernels/test_exl3_compile.py`` the sm_121 build; the GPU bitwise gate is
``Z(ld) == Z(grouped_kernel)`` (docs/ENGINE-PLAN.md, window G1).
"""

from __future__ import annotations

import os
from functools import lru_cache

ENV = "TF_DSV41_EXPERT_LOADS"
ENV_CFG = "TF_DSV41_EXPERT_LOADS_CFG"
ENV_PDL = "TF_DSV41_EXPERT_LOADS_PDL"
CFGS = ((8, 1), (8, 2), (4, 2))
PROBE_CFGS = ((8, 1), (8, 2))
DEFAULT = (8, 1)
WARPS = 4
# (lo, hi) half-bit ranges the extension instantiates (upstream's TF_RANGES minus 2..16: no 5.5+ bit experts here)
RANGES = ((8, 8), (2, 10))
OFF = ("", "0", "false", "no", "off")


def _cfg(text: str) -> tuple[int, int]:
    try:
        nt, pd = (int(p) for p in text.split(","))
    except ValueError:
        nt = pd = -1
    if (nt, pd) not in CFGS:
        raise ValueError(f"{ENV_CFG}={os.environ.get(ENV_CFG)!r}: each half must be one of "
                         f"{', '.join(f'{a},{b}' for a, b in CFGS)}")
    return nt, pd


def parse(env: dict | None = None) -> dict:
    """The knobs, validated: a bad value refuses to start rather than running something else."""

    env = os.environ if env is None else env
    raw = env.get(ENV_CFG, "").strip()
    if raw:
        halves = raw.split("/")
        if len(halves) > 2:
            raise ValueError(f"{ENV_CFG}={raw!r}: expected nt,pd or nt,pd/nt,pd")
        gu = _cfg(halves[0])
        dn = _cfg(halves[1]) if len(halves) == 2 else gu
    else:
        gu = dn = DEFAULT
    return {"on": env.get(ENV, "0").strip().lower() not in OFF, "gu": gu, "dn": dn,
            "pdl": env.get(ENV_PDL, "0").strip().lower() not in OFF}


CFG = parse()


def fits(K: int, N: int, sk: int, warps: int, cfg: tuple[int, int]) -> bool:
    """Whether (nt, pd) runs a K x N matrix in upstream's work items (warps a program, K splits)."""

    nt, pd = cfg
    if warps != WARPS or K % (16 * sk * warps) or N % (16 * nt):
        return False
    per_warp = K // (16 * sk * warps)
    return per_warp >= pd and per_warp % pd == 0


def k2_range(lo: int, hi: int) -> tuple[int, int] | None:
    """The instance covering a layer's half-bit widths [lo, hi], or None (upstream's kernel runs the layer)."""

    for a, b in RANGES:
        if a <= lo and hi <= b:
            return a, b
    return None


@lru_cache(maxsize=1)
def _ext():
    """Built for this GPU only (``tensorfold.cuda.build``), with upstream's exl3 directory on the include path."""

    from pathlib import Path

    from tensorfold.cuda.build import load

    from ..tf import exl3_dir

    here = Path(__file__).parent
    return load(name="tf_dsv41_x3ld_v1", sources=[str(here / "x3ld.cpp"), str(here / "x3ld.cu")],
                extra_include_paths=[str(exl3_dir())], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


@lru_cache(maxsize=None)
def _pdl_ok(device: int) -> bool:
    import torch

    return torch.cuda.get_device_capability(device)[0] >= 9


def grouped(x0, x1, tp0, tp1, k2_0, k2_1, ids, count, members, z, mats: int, K: int, N: int, P: int, sk: int,
            slots: int, cb: int, warps: int, lo: int, hi: int, *, probe: int = 0, cfg: tuple[int, int] | None = None,
            pdl: bool | None = None) -> bool:
    """Upstream's ``ext.grouped(x0, x1, tp0, tp1, k2_0, k2_1, ids, count, members, z, mats, K, N, P, sk, slots, cb,
    nt, warps, pf, lo, hi)`` through ``ld_kernel``, the same Z: (sk, warps) are upstream's setting (they fix the K
    ranges), nt / pd ours. False: not taken (off, a shape, width or codebook it does not cover); the caller then runs
    upstream's kernel."""

    if cfg is None:
        if not CFG["on"]:
            return False
        cfg = CFG["dn" if mats == 1 else "gu"]
    rng = k2_range(lo, hi)
    if rng is None or cb != 2 or not fits(K, N, sk, warps, cfg):
        return False
    if probe and (probe != 3 or cfg not in PROBE_CFGS):
        raise ValueError(f"no probe {probe} at {cfg}")
    pdl = CFG["pdl"] if pdl is None else bool(pdl)
    pdl = pdl and _pdl_ok(x0.device.index or 0)
    _ext().grouped(x0, x1, tp0, tp1, k2_0, k2_1, ids, count, members, z, mats, K, N, P, sk, slots, cb, cfg[0], cfg[1],
                   probe, rng[0], rng[1], pdl)
    return True
