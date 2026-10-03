"""RoPE flavours and the mHC pieces."""

import math

import torch

from engine.reference.config import Config
from engine.reference.hc import HcParams, hc_collapse, hc_mixes, hc_post, hc_pre
from engine.reference.ops import bf16
from engine.reference.rope import rope_for, yarn_inv_freq


def test_yarn_matches_deepseek_formula():
    # independent re-derivation of DeepSeek's YaRN frequency blend (theta 160000, x16 over 65536)
    dim, base, factor, orig = 64, 160000.0, 16.0, 65536
    inv = yarn_inv_freq(dim, base, factor, orig, 32, 1)
    freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)

    def corr(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))

    low, high = max(math.floor(corr(32)), 0), min(math.ceil(corr(1)), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    want = freq / factor * ramp + freq * (1 - ramp)
    assert torch.allclose(inv.double(), want, rtol=1e-6)
    assert inv[0] == 1.0 and inv[-1] < freq[-1]      # fast dims kept, slow dims interpolated


def test_layer_flavours():
    cfg = Config()
    swa, comp = rope_for(cfg, 0), rope_for(cfg, 2)
    assert torch.allclose(swa.inv_freq, 1.0 / 10000.0 ** (torch.arange(0, 64, 2).float() / 64))
    assert comp is rope_for(cfg, 1)                   # ratio 1 and 2 share the compressed YaRN rope
    assert not torch.allclose(swa.inv_freq, comp.inv_freq)


def test_rope_rotates_last_dims_gptj_and_inverts():
    r = rope_for(Config(), 2)
    x = torch.randn(7, 3, 512)
    pos = torch.arange(100, 107)
    y = r.apply(x, pos)
    assert torch.equal(y[..., :448], x[..., :448])
    assert torch.allclose(r.apply(y, pos, inverse=True), x, atol=1e-5)
    # GPT-J pairs: (448, 449) rotate together by inv_freq[0] * pos
    c, s = math.cos(100.0), math.sin(100.0)
    assert torch.allclose(y[0, 0, 448], x[0, 0, 448] * c - x[0, 0, 449] * s, atol=1e-5)
    # relative-position property of the rotated part
    q, k = torch.randn(1, 64), torch.randn(1, 64)
    q = torch.cat([torch.zeros(1, 448), q], -1)
    k = torch.cat([torch.zeros(1, 448), k], -1)
    d1 = (r.apply(q, torch.tensor([50])) * r.apply(k, torch.tensor([20]))).sum()
    d2 = (r.apply(q, torch.tensor([530])) * r.apply(k, torch.tensor([500]))).sum()
    assert torch.allclose(d1, d2, atol=1e-3)


def _params(hc=4, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = hc * (hc + 2)
    return HcParams(torch.randn(m, hc * d, generator=g) * 0.1, torch.randn(m, generator=g),
                    torch.rand(3, generator=g) + 0.5)


def test_comb_is_doubly_stochastic():
    p = _params()
    s = bf16(torch.randn(5, 4, 16))
    pre, post, comb = hc_mixes(s, p, 1e-20, 1e-6, 2.0, 20)
    assert torch.allclose(comb.sum(-2), torch.ones(5, 4), atol=1e-5)
    assert torch.allclose(comb.sum(-1), torch.ones(5, 4), atol=2e-2)
    assert (pre > 0).all() and (pre < 1 + 1e-5).all() and (post > 0).all() and (post < 2).all()


def test_first_layer_broadcast_equals_expanded_streams():
    # vLLM feeds layer 0 the embedding with fn summed over the hc copies; same mixes as the expanded streams
    p = _params()
    e = bf16(torch.randn(3, 16))
    streams = e.unsqueeze(1).expand(-1, 4, -1)
    _, post, comb = hc_mixes(streams, p, 1e-20, 1e-6, 2.0, 20)
    fn_b = p.fn.view(-1, 4, 16).sum(1)
    mixes = (e @ fn_b.t()) * torch.rsqrt(e.square().mean(-1, keepdim=True) + 1e-20)
    post_b = torch.sigmoid(mixes[:, 4:8] * p.scale[1] + p.base[4:8]) * 2.0
    assert torch.allclose(post, post_b, atol=1e-5)


def test_pre_uses_carried_mix_and_post_formula():
    p = _params()
    s = bf16(torch.randn(2, 4, 16))
    carried = torch.rand(2, 4)
    post, comb, x, pre = hc_pre(s, p, carried, torch.ones(16), 1e-20, 1e-20, 1e-6, 2.0, 20)
    collapsed = bf16((carried[..., None] * s).sum(1))
    want = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-20)
    assert torch.allclose(x, bf16(want))
    _, _, x0, _ = hc_pre(s, p, None, torch.ones(16), 1e-20, 1e-20, 1e-6, 2.0, 20)
    assert torch.allclose(hc_collapse(s, None), s[:, 0])
    out = torch.randn(2, 16)
    new = hc_post(out, s, post, comb)
    for t in range(2):
        for j in range(4):
            want_j = sum(comb[t, i, j] * s[t, i] for i in range(4)) + post[t, j] * out[t]
            assert torch.allclose(new[t, j], bf16(want_j), atol=1e-2)
