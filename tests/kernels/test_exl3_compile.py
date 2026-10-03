"""x3ld.cu for sm_121 with nvcc (no GPU): every instance builds with 0 spills, within its launch bounds, with the
shared memory of upstream's grouped_kernel at the same NT (so the same CTAs an SM); its PTX has the 16-byte
non-coherent no-allocate loads, the f16 mma, griddepcontrol.wait and no atomics; the decode / mma helpers are
upstream's (included, never redefined).

    NVCC=<cuda 13.x>/bin/nvcc python -m pytest -q -s tests/kernels/test_exl3_compile.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from conftest import STUBS, nvcc, torch_include

from engine.kernels.tf import exl3_dir

HERE = Path(__file__).resolve().parents[2] / "engine" / "kernels" / "exl3"
NVCC = nvcc()
TORCH_INC = torch_include()
pytestmark = pytest.mark.skipif(NVCC is None or TORCH_INC is None, reason="needs nvcc (NVCC=...) and torch headers")


def _flags():
    cuda_inc = Path(NVCC).parents[1] / "include"
    return ["-arch=sm_121", "-O3", "-std=c++20", "-allow-unsupported-compiler", f"-I{STUBS}", f"-I{cuda_inc}",
            f"-I{TORCH_INC}", f"-I{TORCH_INC / 'torch/csrc/api/include'}", f"-I{exl3_dir()}",
            "-diag-suppress", "20013,20015"]


def _ptxas(src: Path, tmp: Path) -> dict[str, dict]:
    r = subprocess.run([NVCC, *_flags(), "-cubin", "-Xptxas", "-v", "-o", str(tmp / "k.cubin"), str(src)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    out, cur = {}, None
    for ln in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", ln)
        if m:
            cur = m.group(1)
            out[cur] = {}
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
        if m:
            out[cur]["spill"] = int(m.group(1)) + int(m.group(2))
        m = re.search(r"Used (\d+) registers.*?(\d+) bytes smem", ln)
        if m:
            out[cur]["regs"], out[cur]["smem"] = int(m.group(1)), int(m.group(2))
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("x3ld")
    ours = _ptxas(HERE / "x3ld.cu", tmp)
    up = _ptxas(exl3_dir() / "experts_cb2.cu", tmp_path_factory.mktemp("cb2"))
    r = subprocess.run([NVCC, *_flags(), "-ptx", "-o", str(tmp / "k.ptx"), str(HERE / "x3ld.cu")],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    return ours, up, (tmp / "k.ptx").read_text()


def _args(name: str) -> tuple[int, ...]:
    """ld_kernel<CB, NT, PD, LO, HI, PROBE> / grouped_kernel<CB, NT, W, PF, LO, HI> template integers."""

    return tuple(int(v) for v in re.findall(r"Li(\d+)E", name.split("Ev")[0]))


def test_every_instance_builds_without_spills(built):
    ours, _, _ = built
    lds = {k: v for k, v in ours.items() if "ld_kernel" in k}
    assert len(lds) == 10, sorted(lds)                       # 3 settings x 2 ranges + 2 probes x 2 ranges
    for name, info in lds.items():
        cb, nt, pd, lo, hi, probe = _args(name)
        assert info["spill"] == 0, name
        ctas = 3 if nt == 8 else 4
        assert info["regs"] * 128 * ctas <= 65536, (name, info)
        print(f"ld_kernel nt={nt} pd={pd} k2=[{lo},{hi}] probe={probe}: {info['regs']} regs, {info['smem']} B smem")


def test_shared_memory_equals_upstream(built):
    ours, up, _ = built
    up_smem = {}
    for name, info in up.items():
        if "grouped_kernel" in name:
            cb, nt, w, pf, lo, hi = _args(name)
            up_smem.setdefault(nt, set()).add(info["smem"])
    for name, info in ours.items():
        if "ld_kernel" in name:
            nt = _args(name)[1]
            assert {info["smem"]} == up_smem[nt], (name, info["smem"], up_smem[nt])


def test_ptx(built):
    _, _, ptx = built
    assert "ld.global.nc.L1::no_allocate.v4.u32" in ptx
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx
    assert "griddepcontrol.wait" in ptx
    assert "atom." not in ptx and "red.global" not in ptx
    assert "dp4a" in ptx                                     # mul1's decode (upstream's cb_pair<2>)


def test_helpers_are_upstreams():
    src = (HERE / "x3ld.cu").read_text()
    assert '#include "experts_grouped.cuh"' in src
    for name in ("cb_pair", "decode_tile", "mma16816", "struct LaneMap", "struct Fmt"):
        assert not re.search(rf"(__device__|struct)[^;{{]*\b{name.split()[-1]}\b\s*[({{<]", src.replace(
            "using tf_exl3x::", "")), name
