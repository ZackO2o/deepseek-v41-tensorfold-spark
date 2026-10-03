"""The kernels' definitions against engine/reference (the torch oracle built in parallel): skipped until the
reference module is there."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
ops = pytest.importorskip("engine.reference.ops")

from engine.kernels.csa2 import ref  # noqa: E402


def test_kv_row_is_the_references_fp8_ds_mla():
    """csa2's 584-byte row dequantizes to exactly the reference's fp8_ds_mla round trip (the kit's SM12x record)."""

    g = torch.Generator().manual_seed(0)
    x = torch.randn((256, 512), generator=g) * 3
    x[1, :64] = 0.0
    x[2, :64] = 448.0
    x[3] *= 1e-6
    x[4] *= 3e4
    want = ops.fp8_ds_mla_roundtrip(x.clone())
    got = ref.dequantize_rows(*ref.quantize_rows(x)).float()
    assert torch.equal(got, want.float())
