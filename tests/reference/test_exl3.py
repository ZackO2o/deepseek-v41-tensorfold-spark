"""The torch EXL3 decoder against the vendored numpy oracle (TensorFold v0.6.0, itself bit-exact with ExLlamaV3's
``reconstruct``), for every codebook and width; the layer algebra; the checkpoint helpers."""

import numpy as np
import pytest
import torch

from engine.reference import exl3 as E
from engine.reference.ops import Exl3Linear, Numerics

WIDTHS = [(cb, b) for cb in E.CODEBOOKS for b in E.BITS if float(b).is_integer() or cb == "mul1"]


@pytest.mark.parametrize("codebook,bits", WIDTHS)
def test_unpack_matches_oracle_bit_for_bit(codebook, bits):
    w = E.quantize_random(256, 384, bits, codebook, seed=int(10 * bits) + len(codebook))
    got = E.unpack(w.trellis, bits, codebook).numpy()
    want = E.np_unpack(w.trellis.numpy(), bits, codebook)
    assert (got.view(np.int16) == want.view(np.int16)).all()


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_dequantize_matches_oracle(bits):
    w = E.quantize_random(128, 256, bits, "mul1", seed=bits)
    got = w.dequantize(torch.float64).numpy()
    want = E.np_dequantize(w.trellis.numpy(), w.suh.numpy(), w.svh.numpy(), bits, "mul1")
    assert np.abs(got - want).max() < 1e-12


def test_forward_order_equals_dequantized_matmul():
    w = E.quantize_random(256, 128, 3, "mul1", seed=3)
    x = np.random.default_rng(0).standard_normal((5, 256))
    y_kernel_order = E.np_forward(x, w.trellis.numpy(), w.suh.numpy(), w.svh.numpy(), 3, "mul1")
    y = x @ w.dequantize(torch.float64).numpy()
    assert np.allclose(y, y_kernel_order, atol=1e-10)


def test_exl3_linear_is_x_at_w_with_fp16_input():
    w = E.quantize_random(128, 256, 4, "mul1", seed=4)
    x = torch.randn(3, 128) * 3
    lin = Exl3Linear(w, Numerics.kit())
    ref = x.to(torch.float16).double() @ w.dequantize(torch.float64)
    assert torch.allclose(lin(x).double(), ref, atol=1e-4, rtol=1e-4)
    exact = Exl3Linear(w, Numerics.exact())
    assert torch.allclose(exact(x).double(), x.double() @ w.dequantize(torch.float64), atol=1e-4, rtol=1e-4)


def test_bits_from_trellis_shape_and_markers():
    assert E.bits_of((320, 144, 48)) == 3 and E.bits_of((320, 144, 32)) == 2
    assert E.bits_of((320, 80, 96)) == 6 and E.bits_of((10, 10, 40)) == 2.5
    assert E.codebook_from_marker(-2082680531) == "mul1"         # the kit's int32 marker
    assert E.codebook_from_marker(0xCBAC1FED) == "mcg"
    assert E.codebook_from_marker(None) == "3inst"
    with pytest.raises(ValueError):
        E.codebook_from_marker(1234)
    with pytest.raises(ValueError):
        E.Exl3Weight(torch.zeros(1, 1, 40, dtype=torch.int16), torch.zeros(16), torch.zeros(16), "mcg")


def test_mul1_codebook_is_unit_gaussian_like():
    t = E.np_codebook("mul1").astype(np.float64)
    assert np.isfinite(t).all() and 0.5 < t.std() < 2.0 and abs(t.mean()) < 0.1
