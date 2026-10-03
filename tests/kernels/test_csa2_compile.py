"""Every CSA2 / router Triton kernel compiles for sm_121a (GB10) without a GPU (Triton's CPU wheel + its bundled
ptxas): 0 spills, no atomics, tensor-core bf16 mma where the arithmetic says so (attention, indexer scores), the
FMA path (no mma) for the router's ieee fp32 dots, and the paged variants compile beside the contiguous ones.
``csa2_ptx.py`` prints the PTX hashes for before / after identity checks."""

from __future__ import annotations

import os

import pytest

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("compile tests run without TRITON_INTERPRET (their own process)", allow_module_level=True)
pytest.importorskip("triton")

import csa2_ptx  # noqa: E402

MMA_BF16 = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"


@pytest.fixture(scope="module")
def built():
    from engine.kernels.csa2 import compress

    if type(compress._kv_store).__name__ == "InterpretedFunction":
        pytest.skip("the kernels were imported for the interpreter: run the compile tests in their own process")
    out = {}
    for name, (fn, sig, cst, warps) in csa2_ptx.cases().items():
        ptx = csa2_ptx.comp(fn, sig, cst, warps)
        out[name] = (ptx, csa2_ptx.ptxas_info(ptx))
    return out


def test_all_compile_without_spills(built):
    assert len(built) >= 21
    for name, (_, info) in built.items():
        assert info["spill"] == 0, (name, info)
        print(f"{name:28s} {info['regs']:4d} regs")


def test_no_atomics(built):
    for name, (ptx, _) in built.items():
        assert "atom." not in ptx and "red.global" not in ptx, name


def test_tensor_cores_where_meant(built):
    for name in ("attn_chunks", "attn_chunks_paged", "scores", "scores_gather"):
        assert MMA_BF16 in built[name][0], name
    for name in ("router_s1", "router_s0"):
        assert "mma.sync" not in built[name][0] and "fma.rn.f32" in built[name][0], name


def test_paging_is_pruned_when_off(built):
    """PSH = 0 compiles no page-table load: the unpaged kernel has fewer global loads than its paged twin."""

    for base in ("kv_store_r1", "attn_chunks", "scores"):
        n0 = built[base][0].count("ld.global")
        n1 = built[base + "_paged"][0].count("ld.global")
        assert n0 < n1, (base, n0, n1)
