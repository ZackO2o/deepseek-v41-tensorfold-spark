"""The mHC / Engram / DSpark / streaming top-k kernels compile for sm_121a (GB10) without a GPU (Triton's CPU wheel +
its bundled ptxas), with the launch options their wrappers use: 0 spills, no atomics, the FMA chains of the ieee dots
as ``fma.rn.f32`` and no tensor-core mma where the arithmetic is fp32, no FMA contraction where a wrapper launches
with ``enable_fp_fusion=False`` outside the dots, no ``.ftz`` in the bit-exact paths. ``blockers_ptx.py`` prints the
PTX hashes for before / after identity checks."""

from __future__ import annotations

import os
import re

import pytest

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("compile tests run without TRITON_INTERPRET (their own process)", allow_module_level=True)
pytest.importorskip("triton")

import blockers_ptx  # noqa: E402
import csa2_ptx  # noqa: E402


@pytest.fixture(scope="module")
def built():
    from engine.kernels.mhc import kernels as MK

    if type(MK._site).__name__ == "InterpretedFunction":
        pytest.skip("the kernels were imported for the interpreter: run the compile tests in their own process")
    out = {}
    for name, (fn, sig, cst, warps, fusion) in blockers_ptx.cases().items():
        ptx = blockers_ptx.comp(fn, sig, cst, warps, fusion)
        out[name] = (ptx, csa2_ptx.ptxas_info(ptx))
    return out


def test_all_compile_without_spills(built):
    assert len(built) >= 18
    for name, (_, info) in built.items():
        assert info["spill"] == 0, (name, info)
        print(f"{name:28s} {info['regs']:4d} regs")


def test_no_atomics(built):
    for name, (ptx, _) in built.items():
        assert "atom." not in ptx and "red.global" not in ptx, name


def test_fp32_chains_are_fma_not_mma(built):
    for name in ("mhc_boundary", "mhc_site", "mhc_site_entry"):
        ptx = built[name][0]
        assert "mma.sync" not in ptx and "fma.rn.f32" in ptx, name
    # the sums against ones: LLVM folds fma(p, 1, acc) into add.rn (the same single rounding)
    ptx = built["engram_fuse"][0]
    assert "mma.sync" not in ptx and "add.rn.f32" in ptx


def test_exact_paths_have_no_ftz(built):
    for name in ("mhc_boundary", "mhc_post_only", "engram_dequant", "engram_fuse"):
        ptx = built[name][0]
        assert not re.search(r"(mul|add|fma|cvt)[a-z0-9.]*\.ftz", ptx), name


def test_no_contraction_outside_the_dots(built):
    """post_only has no dot: with fusion off its streams' a * b + c are separate mul / add (no fma at all)."""

    ptx = built["mhc_post_only"][0]
    assert "fma.rn.f32" not in ptx and "mul.rn.f32" in ptx or "mul.f32" in ptx
    assert "fma" not in built["engram_dequant"][0]


def test_streaming_scores_on_tensor_cores(built):
    """The streaming top-k scores its tiles with the indexer's bf16 mma (``index.score_tile``), in every mode."""

    for name in ("stream_select", "stream_reindex", "stream_blocks", "stream_select_paged"):
        assert "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in built[name][0], name


def test_dspark_chain_bias_on_tensor_cores(built):
    assert "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in built["dspark_chain"][0]
