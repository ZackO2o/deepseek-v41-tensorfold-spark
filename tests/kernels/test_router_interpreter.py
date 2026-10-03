"""The router kernel (engine/kernels/router.py) in Triton's CPU interpreter: experts and weights against the float64
routing, ties to the lower expert id, the shared slot, DSpark's 128 / top-3, and a row alone == the row in a window
(bit for bit)."""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from engine.kernels import router  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(router._route).__name__ == "InterpretedFunction"
pytestmark = [pytest.mark.skipif(not INTERP, reason="Triton's CPU interpreter (TRITON_INTERPRET=1)"),
              pytest.mark.usefixtures("gpu_like")]


def _inputs(R: int, E: int, seed: int, D: int = router.DIMS):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((R, D), generator=g).to(torch.bfloat16)
    w = (torch.randn((E, D), generator=g) * 0.02).to(torch.float16).contiguous()
    b = (torch.randn(E, generator=g) * 0.1).to(torch.float16)
    return x, w, b


@pytest.mark.parametrize("E,topk", [(384, 6), (128, 3)])
def test_against_float64(E, topk):
    x, w, b = _inputs(19, E, E)
    pick, wts = router.route(x, w, b, topk=topk)
    rp, rw = router.reference(x, w, b, topk)
    assert torch.equal(pick[:, :topk], rp)
    torch.testing.assert_close(wts[:, :topk].double(), rw, rtol=1e-5, atol=1e-6)
    assert bool((pick[:, topk] == E).all()) and bool((wts[:, topk] == 1.0).all())
    torch.testing.assert_close(wts[:, :topk].sum(1).double(), torch.full((19,), 1.5, dtype=torch.float64))


def test_ties_go_to_the_lower_id():
    x, w, b = _inputs(3, 384, 1)
    w[10] = w[200]                      # identical rows: identical scores
    w[11] = w[300]
    b[10] = b[200] = b[11] = b[300] = 5.0   # and the largest biased scores
    pick, _ = router.route(x, w, b, shared=False)
    for r in range(3):
        row = pick[r].tolist()
        assert row.index(10) < row.index(200) and row.index(11) < row.index(300)


def test_row_invariance():
    x, w, b = _inputs(21, 384, 2)
    pick, wts = router.route(x, w, b)
    for r in (0, 15, 16, 20):
        p1, w1 = router.route(x[r:r + 1].contiguous(), w, b)
        assert torch.equal(p1[0], pick[r]) and torch.equal(w1[0].view(torch.int32), wts[r].view(torch.int32))
