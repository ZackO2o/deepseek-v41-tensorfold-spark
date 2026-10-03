"""x3gm.cu (TF_DSV41_FAST_EXPERTS=gm, the fast prefill routed experts in the TensorFold tree) for sm_121 with nvcc, no
GPU: every gm_kernel instance (gate/up: 2 / 3 / 4-bit x one or two inputs x cfg 0 / 1; down: 3 widths x cfg 0 / 1)
builds with 0 spills, its registers fit the CTAs an SM its launch bounds promise, its shared memory (dynamic, from
the kernel's Cfg, + static) fits GB10's 99 KB a block; the PTX has the f16 mma, ldmatrix, cp.async and mul1's dp4a,
and the only atomic is the ticket; the decode / fwht helpers are upstream's (included, never redefined).

    NVCC=<cuda 13.x>/bin/nvcc python -m pytest -q -s tests/kernels/test_x3gm_compile.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from conftest import STUBS, TF, nvcc, torch_include

from engine.kernels.tf import exl3_dir

NVCC = nvcc()
TORCH_INC = torch_include()
pytestmark = pytest.mark.skipif(NVCC is None or TORCH_INC is None or TF is None,
                                reason="needs nvcc (NVCC=...), torch headers and the TensorFold tree (TF_SRC)")
SMEM_BLOCK = 99 * 1024
NB = 8


def _src() -> Path:
    return Path(TF) / "tensorfold" / "families" / "deepseek_v41" / "cuda" / "x3gm.cu"


def _flags():
    cuda_inc = Path(NVCC).parents[1] / "include"
    return ["-arch=sm_121", "-O3", "-std=c++20", "-allow-unsupported-compiler", f"-I{STUBS}", f"-I{cuda_inc}",
            f"-I{TORCH_INC}", f"-I{TORCH_INC / 'torch/csrc/api/include'}", f"-I{exl3_dir()}",
            "-diag-suppress", "20013,20015"]


def cfg(mats, xm, k2, mtl, ng, ks, nsa, ngr) -> dict:
    """x3gm.cu's Cfg: threads, dynamic shared memory and MINB (the CTAs an SM __launch_bounds__ caps registers for)."""

    bm, wpm = ng * 8, NB // mtl
    threads = mats * wpm * 32
    stage = xm * bm * ks * 16 * 2 + mats * ks * NB * 4 * k2 * 4
    smem = max(nsa * stage, mats * ngr * 8 * 132 * 4)
    return {"threads": threads, "smem": smem, "minb": 2 if 2 * (smem + 1536) <= 102400 else 1}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("x3gm")
    r = subprocess.run([NVCC, *_flags(), "-cubin", "-Xptxas", "-v", "-o", str(tmp / "k.cubin"), str(_src())],
                       capture_output=True, text=True, timeout=900, check=False)
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
        elif re.search(r"Used (\d+) registers", ln):
            out[cur]["regs"], out[cur]["smem"] = int(re.search(r"Used (\d+) registers", ln).group(1)), 0
    p = subprocess.run([NVCC, *_flags(), "-ptx", "-o", str(tmp / "k.ptx"), str(_src())], capture_output=True,
                       text=True, timeout=900, check=False)
    assert p.returncode == 0, p.stderr[-4000:]
    return out, (tmp / "k.ptx").read_text()


def test_every_instance_builds_without_spills(built):
    info, _ = built
    gm = {k: v for k, v in info.items() if "gm_kernel" in k}
    assert len(gm) == 18, sorted(gm)                       # gate/up 3 widths x 2 inputs x 2 cfgs + down 3 x 2
    for name, v in gm.items():
        args = [int(a) for a in re.findall(r"Li(\d+)E", name)]
        c = cfg(*args)
        assert v["spill"] == 0, (name, v)
        assert v["regs"] * c["threads"] * c["minb"] <= 65536, (name, v, c)
        assert c["smem"] + v["smem"] <= SMEM_BLOCK, (name, v, c)
        print(f"gm_kernel {args}: {v['regs']} regs, {c['threads']} threads x {c['minb']} CTAs an SM, "
              f"{c['smem']} + {v['smem']} B smem")
    for name, v in info.items():
        assert v["spill"] == 0, (name, v)


def test_default_configs_run_two_ctas_an_sm():
    """auto: gate/up cfg 0 with one input, 1 with two; down cfg 0 (x3gm.py) -- 2 CTAs an SM at 3 bits."""

    assert cfg(2, 1, 6, 2, 8, 4, 3, 4)["minb"] == 2
    assert cfg(2, 2, 6, 2, 8, 2, 4, 4)["minb"] == 2
    assert cfg(1, 1, 6, 2, 8, 4, 4, 8)["minb"] == 2


def test_ptx(built):
    _, ptx = built
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx
    assert "ldmatrix.sync.aligned.m8n8.x4.shared.b16" in ptx
    assert "cp.async.cg.shared.global" in ptx
    assert "dp4a" in ptx                                     # mul1's decode (upstream's cb_pair<2>)
    atoms = re.findall(r"\b(atom\.[\w.]+|red\.[\w.]+)", ptx)
    assert set(atoms) <= {"atom.global.add.u32", "atom.global.add.s32"}, set(atoms)   # the ticket only


def test_helpers_are_upstreams():
    src = _src().read_text()
    assert '#include "experts_grouped.cuh"' in src and '#include "decode.cuh"' in src
    for name in ("cb_pair", "LaneMap", "Fmt", "fwht128", "mma16816"):
        assert not re.search(rf"__device__[^;{{(]*\b{name}\s*\(", src), name           # a function definition
        assert not re.search(rf"struct\s+{name}\b", src), name
