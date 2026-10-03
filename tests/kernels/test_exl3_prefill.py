"""x3pf.cu (prefill routed experts: each decoded trellis tile applied to MTL member tiles), no GPU:

- schedule emulator (``x3pf_emu``): every Z element pf_kernel stores is upstream grouped_kernel's -- the same address,
  A row, mma chain (k tiles ascending a warp, column tile, n8 half) and warp-order sum -- and it stores exactly
  upstream's set, for both settings (nt 4 / mtl 4, nt 8 / mtl 2), both matrices, K splits 1 / 4, member counts that
  fill 0..MTL tiles of a group, dead experts past ucount and padded member lists;
- the knobs and the dispatch rules (fall back to the decode path for few rows, unsupported widths or codebooks);
- nvcc sm_121: every instance builds with 0 spills, at most 255 registers (1 CTA of 4 warps), its shared memory, the
  f16 mma and mul1's dp4a in the PTX, no atomics, the helpers upstream's (included, never redefined).

    NVCC=<cuda 13.x>/bin/nvcc python -m pytest -q -s tests/kernels/test_exl3_prefill.py
"""

from __future__ import annotations

import random
import re
import subprocess
from pathlib import Path

import pytest
import x3pf_emu as EMU
from conftest import nvcc, torch_include

from engine.kernels.exl3 import prefill

HERE = Path(__file__).resolve().parents[2] / "engine" / "kernels" / "exl3"


def _members(nexp: int, counts: list[int], R: int, slots: int, seed: int) -> list[list[int]]:
    """Distinct-expert member lists (codes row << 5 | slot, rows ascending, -1 padded to R), like ext.group: each
    (row, slot) pair belongs to one expert, an expert takes a row at most once."""

    g = random.Random(seed)
    free = {r: list(range(slots)) for r in range(R)}
    out = []
    for c in counts[:nexp]:
        rows = sorted(g.sample([r for r in range(R) if free[r]], c))
        codes = []
        for r in rows:
            sl = free[r].pop(g.randrange(len(free[r])))
            codes.append((r << 5) | sl)
        out.append(codes + [-1] * (R - c))
    return out


@pytest.mark.parametrize("nt,mtl", prefill.CFGS)
@pytest.mark.parametrize("mats,K,N,SK", [(2, 512, 128, 4), (1, 256, 256, 1)])
def test_schedule_equals_upstream(nt, mtl, mats, K, N, SK):
    R, slots = 200, 7
    P = R * slots
    counts = [0, 1, 15, 16, 17, 64, 65, 127, 128, 129, 200, 33]                # <= R, sum <= R * slots
    members = _members(len(counts), counts, R, slots, nt * 10 + mtl)
    for ucount in (len(counts), len(counts) - 3):
        up = EMU.trace_grouped(members, ucount, K, N, P, SK, slots, mats, nt)
        ours = EMU.trace_pf(members, ucount, K, N, P, SK, slots, mats, nt, mtl)
        assert ours == up
        assert len(up) == sum(min(c, R) for c in counts[:ucount]) * N * mats * SK


def test_knobs_and_dispatch():
    assert prefill.parse({})["on"] is False and prefill.parse({})["gu"] == prefill.DEFAULT
    c = prefill.parse({prefill.ENV: "1", prefill.ENV_CFG: "8,2/4,4", prefill.ENV_ROWS: "128"})
    assert c["on"] and c["gu"] == (8, 2) and c["dn"] == (4, 4) and c["rows"] == 128
    with pytest.raises(ValueError):
        prefill.parse({prefill.ENV_CFG: "16,1"})
    assert prefill.k2_range(4, 6) == (2, 10) and prefill.k2_range(8, 8) == (8, 8) and prefill.k2_range(2, 12) is None
    assert prefill.fits(5120, 1152, 4, 4, (4, 4)) and prefill.fits(1152, 5120, 1, 4, (8, 2))
    assert not prefill.fits(5120, 1152, 4, 8, (4, 4))
    # not taken: off by default / too few rows (no extension needed for these answers)
    assert prefill.grouped(*([None] * 10), 2, 5120, 1152, 1, 4, 7, 2, 4, 6, 6, rows=4096) is False
    old = prefill.CFG
    try:
        prefill.CFG = dict(old, on=True)
        assert prefill.grouped(*([None] * 10), 2, 5120, 1152, 1, 4, 7, 2, 4, 6, 6, rows=8) is False
        assert prefill.grouped(*([None] * 10), 2, 5120, 1152, 1, 4, 7, 1, 4, 6, 6, rows=4096) is False   # mcg
        assert prefill.grouped(*([None] * 10), 2, 5120, 1152, 1, 4, 7, 2, 4, 11, 12, rows=4096) is False
    finally:
        prefill.CFG = old


NVCC = nvcc()
TORCH_INC = torch_include()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if NVCC is None or TORCH_INC is None:
        pytest.skip("needs nvcc (NVCC=...) and torch headers")
    import test_exl3_compile as C

    tmp = tmp_path_factory.mktemp("x3pf")
    info = C._ptxas(HERE / "x3pf.cu", tmp)
    r = subprocess.run([NVCC, *C._flags(), "-ptx", "-o", str(tmp / "k.ptx"), str(HERE / "x3pf.cu")],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    return info, (tmp / "k.ptx").read_text()


def test_every_instance_builds_without_spills(built):
    info, _ = built
    pfs = {k: v for k, v in info.items() if "pf_kernel" in k}
    assert len(pfs) == 4, sorted(pfs)                        # 2 settings x 2 width ranges
    for name, v in pfs.items():
        cb, nt, mtl, lo, hi = (int(x) for x in re.findall(r"Li(\d+)E", name.split("Ev")[0]))
        assert v["spill"] == 0 and v["regs"] <= 255, (name, v)
        assert v["smem"] <= 4 * 16 * nt * 16 * 4 + mtl * 16 * 4 + 256, (name, v)
        print(f"pf_kernel nt={nt} mtl={mtl} k2=[{lo},{hi}]: {v['regs']} regs, {v['smem']} B smem")


def test_ptx(built):
    _, ptx = built
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx
    assert "dp4a" in ptx
    assert "atom." not in ptx and "red.global" not in ptx


def test_helpers_are_upstreams():
    src = (HERE / "x3pf.cu").read_text()
    assert '#include "experts_grouped.cuh"' in src
    body = src.replace("using tf_exl3x::", "")
    for name in ("cb_pair", "decode_tile", "mma16816", "load_pair", "load_words", "LaneMap", "Fmt"):
        assert not re.search(rf"(__device__|struct)[^;{{]*\b{name}\b\s*[({{<]", body), name
