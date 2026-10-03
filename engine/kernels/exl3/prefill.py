"""Prefill routed experts for DeepSeek-V4.1-Flash (``x3pf.cu``): upstream's grouped EXL3 expert GEMM with each decoded
trellis tile applied to MTL member tiles (64 members at the default (nt 4, mtl 4)) instead of one, upstream's bits.

Knobs (load-time, per rank; ranks may differ without changing a bit):

``TF_DSV41_EXPERT_PREFILL=1``      a routed layer's two grouped launches run ``pf_kernel`` when the window has at least
                                   ``TF_DSV41_EXPERT_PREFILL_ROWS`` rows (default 64: below it the members an expert
                                   rarely fill a second tile and the decode path's loads win)
``TF_DSV41_EXPERT_PREFILL_CFG``    "nt,mtl" (4,4 or 8,2) for both, or "nt,mtl/nt,mtl" (gate/up, down)

Exactness: the K ranges (upstream's SK and 4 warps), every element's mma chain (k ascending from +0.0, upstream's
``decode_tile`` B fragments and ``load_pair`` A values), the warp-order sum and the Z addresses are upstream's; only
which program holds a member tile and how many tiles share a decoded B fragment change. So prefill rows get the
decode path's bits (the exact prefill tag). ``tests/kernels/x3pf_emu.py`` + ``test_exl3_prefill.py`` check the
schedule and the sm_121 build; the GPU bitwise gate is ``Z(pf) == Z(grouped_kernel)`` (window G1).
"""

from __future__ import annotations

import os
from functools import lru_cache

ENV = "TF_DSV41_EXPERT_PREFILL"
ENV_CFG = "TF_DSV41_EXPERT_PREFILL_CFG"
ENV_ROWS = "TF_DSV41_EXPERT_PREFILL_ROWS"
CFGS = ((4, 4), (8, 2))
DEFAULT = (4, 4)
WARPS = 4
RANGES = ((8, 8), (2, 10))
OFF = ("", "0", "false", "no", "off")


def _cfg(text: str) -> tuple[int, int]:
    try:
        nt, mtl = (int(p) for p in text.split(","))
    except ValueError:
        nt = mtl = -1
    if (nt, mtl) not in CFGS:
        raise ValueError(f"{ENV_CFG}={os.environ.get(ENV_CFG)!r}: each half must be one of "
                         f"{', '.join(f'{a},{b}' for a, b in CFGS)}")
    return nt, mtl


def parse(env: dict | None = None) -> dict:
    env = os.environ if env is None else env
    raw = env.get(ENV_CFG, "").strip()
    if raw:
        halves = raw.split("/")
        if len(halves) > 2:
            raise ValueError(f"{ENV_CFG}={raw!r}: expected nt,mtl or nt,mtl/nt,mtl")
        gu = _cfg(halves[0])
        dn = _cfg(halves[1]) if len(halves) == 2 else gu
    else:
        gu = dn = DEFAULT
    rows = int(env.get(ENV_ROWS, "64") or 64)
    if rows < 1:
        raise ValueError(f"{ENV_ROWS} must be >= 1")
    return {"on": env.get(ENV, "0").strip().lower() not in OFF, "gu": gu, "dn": dn, "rows": rows}


CFG = parse()


def fits(K: int, N: int, sk: int, warps: int, cfg: tuple[int, int]) -> bool:
    nt, _ = cfg
    return warps == WARPS and K % (16 * sk * warps) == 0 and N % (16 * nt) == 0


def k2_range(lo: int, hi: int) -> tuple[int, int] | None:
    for a, b in RANGES:
        if a <= lo and hi <= b:
            return a, b
    return None


@lru_cache(maxsize=1)
def _ext():
    from pathlib import Path

    from tensorfold.cuda.build import load

    from ..tf import exl3_dir

    here = Path(__file__).parent
    return load(name="tf_dsv41_x3pf_v1", sources=[str(here / "x3pf.cpp"), str(here / "x3pf.cu")],
                extra_include_paths=[str(exl3_dir())], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def grouped(x0, x1, tp0, tp1, k2_0, k2_1, ids, count, members, z, mats: int, K: int, N: int, P: int, sk: int,
            slots: int, cb: int, warps: int, lo: int, hi: int, *, rows: int, cfg: tuple[int, int] | None = None) -> bool:
    """Upstream's ``ext.grouped`` through ``pf_kernel`` (the same Z) for a window of ``rows`` rows; False: not taken
    (off, too few rows, a shape, width or codebook it does not cover): the caller runs the decode path's kernel."""

    if cfg is None:
        if not CFG["on"] or rows < CFG["rows"]:
            return False
        cfg = CFG["dn" if mats == 1 else "gu"]
    rng = k2_range(lo, hi)
    if rng is None or cb != 2 or not fits(K, N, sk, warps, cfg):
        return False
    _ext().grouped(x0, x1, tp0, tp1, k2_0, k2_1, ids, count, members, z, mats, K, N, P, sk, slots, cb, cfg[0], cfg[1],
                   rng[0], rng[1])
    return True
