"""Single-Pass mHC kernels (engine/kernels/mhc) in Triton's CPU interpreter with the ``gpu_like`` model:

- the boundary pass (post, collapse with the carried pre-mix, taps, the next site's partials) == the torch emulation
  ``mhc.ref.boundary`` bit for bit, for world 1 / 2, collapse with pre-mix / stream 0 / none;
- the normed input == ``ref.finish_norm`` bit for bit; pre / post / comb within 2e-6 of the torch fp32 formulas and
  of float64 (``ref.math``), and the whole site against engine/reference/hc.py;
- row invariance: a row alone == the same row in a window (every output, every coefficient, bit for bit);
- ``final`` (hidden + model norm) and ``post_only``; in place (xout is x) == out of place.

Run: TRITON_INTERPRET=1 python -m pytest -q tests/kernels/test_mhc_interpreter.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from engine.kernels import mhc  # noqa: E402
from engine.kernels.mhc import kernels as K  # noqa: E402
from engine.kernels.mhc import ref  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(K._site).__name__ == "InterpretedFunction"
pytestmark = [pytest.mark.skipif(not INTERP, reason="Triton's CPU interpreter (TRITON_INTERPRET=1)"),
              pytest.mark.usefixtures("gpu_like")]


def _bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def _case(R: int, d: int, world: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((R, 4 * d), generator=g).to(torch.bfloat16)
    gathered = torch.randn((world, R, d), generator=g) * 0.7
    comb = torch.rand((R, 4, 4), generator=g)
    comb = (comb / comb.sum(1, keepdim=True)).reshape(R, 16).contiguous()
    prev = mhc.Coefs(R, "cpu", pre=torch.rand((R, 4), generator=g) + 0.1, post=torch.rand((R, 4), generator=g) * 2,
                     comb=comb)
    hc = mhc.Hc(torch.randn((24, 4 * d), generator=g) * 0.01, torch.randn(24, generator=g) * 0.5,
                torch.rand(3, generator=g) + 0.5)
    nw = (torch.rand(d, generator=g) + 0.5).to(torch.bfloat16)
    return x, gathered, prev, hc, nw


def _boundary(x, gathered, prev, hc, nw, *, tap=True):
    R, d = gathered.shape[1], gathered.shape[2]
    xout = torch.zeros_like(x)
    out = torch.zeros((R, d), dtype=torch.bfloat16)
    co = mhc.Coefs(R, "cpu")
    sc = mhc.Scratch(R, "cpu", d)
    tp = torch.zeros((R, 3 * d), dtype=torch.bfloat16)[:, d:2 * d] if tap else None
    mhc.boundary(x, xout, gathered, prev, hc, nw, out, co, sc, tap=tp)
    return xout, out, co, sc, tp


@pytest.mark.parametrize("world", [1, 2])
def test_boundary_bits_against_emulation(world):
    d = 5120
    x, gathered, prev, hc, nw = _case(3, d, world, world)
    xout, out, co, sc, tp = _boundary(x, gathered, prev, hc, nw)
    want = ref.boundary(x, gathered, prev.post, prev.comb, prev.pre, 2, hc.fn, K.NB)
    assert torch.equal(_bits(xout), _bits(want["x"]))
    assert torch.equal(_bits(sc.c[:3]), _bits(want["c"]))
    assert torch.equal(_bits(tp), _bits(want["tap"]))
    got = sc.part[:3]
    assert torch.equal(_bits(got[..., :25]), _bits(want["part"][..., :25]))
    assert torch.equal(_bits(got[:, 0, :, 25]), _bits(want["part"][:, 0, :, 25]))
    assert torch.equal(_bits(out), _bits(ref.finish_norm(want["part"], want["c"], nw, mhc.EPS)))
    pre, post, comb = ref.coefficients(want["part"], hc.base, hc.scale, d, mhc.EPS, mhc.HC_EPS, mhc.POST_ALPHA,
                                       mhc.ITERS)
    torch.testing.assert_close(co.pre, pre, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(co.post, post, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(co.comb.view(-1, 4, 4), comb, rtol=1e-5, atol=2e-6)
    m = ref.math(x, gathered, prev.post, prev.comb, prev.pre, hc.fn, hc.base, hc.scale, nw)
    torch.testing.assert_close(xout.double(), m["x"], rtol=2 ** -7, atol=1e-30)      # within a bf16 ulp
    torch.testing.assert_close(co.pre.double(), m["pre"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(co.post.double(), m["post"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(co.comb.double().view(-1, 4, 4), m["comb"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out.double(), m["input"], rtol=8e-3, atol=1e-2)


def test_site_entry_and_engine_reference():
    """``site`` with no pre-mix (stream 0) and with one, against engine/reference/hc.py's hc_pre."""

    from engine.reference.hc import HcParams, hc_pre

    d = 1280
    for pre_in in (None, "carry"):
        x, _, prev, hc, nw = _case(5, d, 1, 7)
        p_in = prev.pre if pre_in else None
        out = torch.zeros((5, d), dtype=torch.bfloat16)
        co = mhc.Coefs(5, "cpu")
        sc = mhc.Scratch(5, "cpu", d)
        mhc.site(x, hc, nw, out, co, sc, pre_in=p_in)
        want = ref.boundary(x, None, None, None, p_in, 2 if pre_in else 1, hc.fn, K.NB)
        assert torch.equal(_bits(sc.c[:5]), _bits(want["c"]))
        assert torch.equal(_bits(out), _bits(ref.finish_norm(want["part"], want["c"], nw, mhc.EPS)))
        post, comb, xin, pre = hc_pre(x.float().view(5, 4, d), HcParams(hc.fn, hc.base, hc.scale), p_in, nw,
                                      mhc.EPS, mhc.EPS, mhc.HC_EPS, mhc.POST_ALPHA, mhc.ITERS)
        torch.testing.assert_close(co.pre, pre, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(co.post, post, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(co.comb.view(-1, 4, 4), comb, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(out.float(), xin, rtol=8e-3, atol=1e-2)


def test_row_invariance():
    d = 1280
    R = 19
    x, gathered, prev, hc, nw = _case(R, d, 2, 11)
    xout, out, co, sc, tp = _boundary(x, gathered, prev, hc, nw)
    for r in (0, 7, 16, 18):
        sl = slice(r, r + 1)
        p1 = mhc.Coefs(1, "cpu", pre=prev.pre[sl].clone(), post=prev.post[sl].clone(), comb=prev.comb[sl].clone())
        x1, o1, c1, s1, t1 = _boundary(x[sl].clone(), gathered[:, sl].clone(), p1, hc, nw)
        for a, b in ((x1, xout[sl]), (o1, out[sl]), (t1, tp[sl]), (c1.pre, co.pre[sl]), (c1.post, co.post[sl]),
                     (c1.comb, co.comb[sl]), (s1.c[:1], sc.c[sl])):
            assert torch.equal(_bits(a), _bits(b)), r


def test_final_post_only_and_in_place():
    d = 1280
    R = 4
    x, gathered, prev, hc, nw = _case(R, d, 2, 13)
    # final: hidden = collapse with the carried pre-mix, then the model norm
    xo = torch.zeros_like(x)
    out = torch.zeros((R, d), dtype=torch.bfloat16)
    sc = mhc.Scratch(R, "cpu", d)
    mhc.final(x, xo, gathered, prev, nw, out, sc)
    want = ref.boundary(x, gathered, prev.post, prev.comb, prev.pre, 2, None, K.NB)
    assert torch.equal(_bits(xo), _bits(want["x"])) and torch.equal(_bits(sc.c[:R]), _bits(want["c"]))
    assert torch.equal(_bits(out), _bits(ref.finish_norm(want["part"], want["c"], nw, mhc.EPS)))
    # post only (before an Engram layer), in place
    xi = x.clone()
    tp = torch.zeros((R, d), dtype=torch.bfloat16)
    mhc.post_only(xi, xi, gathered, prev, tap=tp)
    assert torch.equal(_bits(xi), _bits(want["x"])) and torch.equal(_bits(tp), _bits(want["tap"]))
    # the fused boundary in place == out of place
    xb = x.clone()
    out2 = torch.zeros((R, d), dtype=torch.bfloat16)
    co2 = mhc.Coefs(R, "cpu")
    mhc.boundary(xb, xb, gathered, prev, hc, nw, out2, co2, mhc.Scratch(R, "cpu", d))
    xa, oa, ca, _, _ = _boundary(x, gathered, prev, hc, nw, tap=False)
    assert torch.equal(_bits(xb), _bits(xa)) and torch.equal(_bits(out2), _bits(oa))
    assert torch.equal(_bits(co2.comb), _bits(ca.comb))
