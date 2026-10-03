"""Engram hashing / layout / gate, and the MoE router and experts."""

import torch

from engine.reference.config import Config
from engine.reference.engram import Engram, EngramLayout, EngramWeights, NgramHasher, TensorRows, hash_multipliers
from engine.reference.moe import ExpertWeights, MoE, MoEWeights, route
from engine.reference.ops import DenseLinear, bf16, fp8_e4m3_dequant, swiglu_clamp


def test_real_layout_reproduces_the_checkpoint_table_sizes():
    cfg = Config()
    lay = EngramLayout(cfg)
    assert lay.primes.shape == (2, 24)
    assert [lay.total_rows(i) for i in range(2)] == list(cfg.engram_num_embeddings)   # 384,006,168 / 384,016,682
    flat = lay.primes.flatten().tolist()
    assert len(set(flat)) == 48 and min(flat) > cfg.engram_vocab_size - 1
    assert lay.offsets[0, 1] == lay.primes[0, 0]
    lo, hi, h0, nh = lay.head_shard(0, 1, 2)
    assert (h0, nh) == (12, 12) and hi == lay.total_rows(0) and lo == int(lay.primes[0, :12].sum())


def test_multipliers_are_odd_and_overflow_free():
    m = hash_multipliers((1, 14), 4, 99092)
    assert (m % 2 == 1).all() and (m > 0).all()
    assert int(m.max()) * 99092 < 2 ** 63


def _hasher(cfg):
    return NgramHasher(cfg, [i % cfg.engram_compressed_vocab_size for i in range(cfg.vocab_size)])


def test_hash_blocks_at_the_start_and_on_dead_tokens(cfg):
    h = _hasher(cfg)
    ids = torch.tensor([5, 6, 7, 8, 9])
    out = h(ids)
    assert out.shape == (5, 1, cfg.n_hash_cols)
    # position 0: every n-gram reaches before the start -> padded; must differ from position 1's 2-gram
    nh = cfg.engram_n_heads
    rows = out[:, 0]
    assert (rows < h.layout.offsets[0][None, :] + h.layout.primes[0][None, :]).all()
    assert (rows >= h.layout.offsets[0][None, :]).all()
    # a token sequence's hash depends only on the last max_ngram ids
    other = h(torch.tensor([1, 2, 7, 8, 9]))
    assert torch.equal(other[4, 0], out[4, 0])                          # (7, 8, 9) in both
    assert torch.equal(other[3, 0, :nh], out[3, 0, :nh])                # 2-gram (7, 8) in both
    assert not torch.equal(other[3, 0, nh:], out[3, 0, nh:])            # 3-gram (6|2, 7, 8) differs
    # a dead (image) token blocks itself and every longer n-gram through it
    dead = torch.tensor([False, False, True, False, False])
    d = h(ids, dead)
    assert torch.equal(d[4, 0, :nh], out[4, 0, :nh])                # 2-gram (8, 9) does not reach it
    assert not torch.equal(d[4, 0, nh:], out[4, 0, nh:])            # 3-gram (7, 8, 9) does


def test_hash_lookback_equals_full_sequence(cfg):
    h = _hasher(cfg)
    ids = torch.randint(0, cfg.vocab_size, (12,))
    full = h(ids)
    tail = h(ids[7:], lookback=ids[5:7])
    assert torch.equal(full[7:], tail)


def test_fp8_rows_and_gate(cfg):
    raw = torch.randn(4, 64).to(torch.float8_e4m3fn)
    sc = torch.tensor([[127, 128], [126, 127], [130, 120], [127, 127]], dtype=torch.uint8)
    rows = fp8_e4m3_dequant(raw.view(torch.uint8), sc)
    assert torch.allclose(rows[0, :32], raw[0, :32].float()) and torch.allclose(rows[0, 32:], 2 * raw[0, 32:].float())
    d, hc = cfg.hidden_size, cfg.hc_mult
    g = torch.Generator().manual_seed(0)
    lay = EngramLayout(cfg)
    w = EngramWeights(DenseLinear(bf16(torch.randn((hc + 1) * d, cfg.n_hash_cols * cfg.engram_head_dim, generator=g)
                                       * 0.05)),
                      torch.randn(hc, d, generator=g), torch.randn(hc, d, generator=g),
                      TensorRows(torch.randn(lay.total_rows(0), cfg.engram_head_dim, generator=g)))
    e = Engram(cfg, 1, w)
    s = bf16(torch.randn(5, hc, d))
    hashes = _hasher(cfg)(torch.randint(0, cfg.vocab_size, (5,)))
    out = e(s, hashes)
    rows = bf16(w.table.rows(hashes[:, 0].reshape(-1))).view(5, -1)
    kv = bf16(w.wkv(rows))
    t, j = 2, 1
    x, key, val = s[t, j], kv[t, j * d:(j + 1) * d], kv[t, hc * d:]
    qk = bf16(w.q_weight[j]) * bf16(w.k_weight[j])
    dot = (x * qk * key).sum() / torch.sqrt(x.square().mean()) / torch.sqrt(key.square().mean()) / d ** 0.5
    gate = torch.sigmoid(torch.sign(dot) * torch.sqrt(dot.abs().clamp(min=1e-6)))
    assert torch.allclose(out[t, j], bf16(x + gate * val), atol=2e-2, rtol=2e-2)
    off = e(s, hashes, keep=torch.zeros(5, dtype=torch.bool))
    assert torch.equal(off, s)


def test_router_sqrtsoftplus_noaux_tc():
    x = torch.randn(4, 8)
    gate = torch.randn(6, 8)
    bias = torch.tensor([0, 0, 0, 0, 0, 100.0])          # forces expert 5 in, without weighting it more
    w, ids = route(x, gate, bias, 2, 1.5)
    assert (ids == 5).any(-1).all()
    s = torch.sqrt(torch.nn.functional.softplus(x @ gate.t()))
    assert torch.allclose(w.sum(-1), torch.full((4,), 1.5))
    assert torch.allclose(w, 1.5 * s.gather(1, ids) / s.gather(1, ids).sum(-1, keepdim=True))


def test_moe_equals_dense_loop(cfg):
    g = torch.Generator().manual_seed(1)
    d, inter = cfg.hidden_size, cfg.moe_intermediate_size

    def lin(i, o):
        return DenseLinear(bf16(torch.randn(o, i, generator=g) * 0.1))

    experts = [ExpertWeights(lin(d, inter), lin(inter, d), lin(d, inter)) for _ in range(cfg.n_routed_experts)]
    shared = ExpertWeights(lin(d, inter), lin(inter, d), lin(d, inter))
    mw = MoEWeights(bf16(torch.randn(cfg.n_routed_experts, d, generator=g)), torch.zeros(cfg.n_routed_experts),
                    lambda e: experts[e], shared)
    m = MoE(cfg, 3, mw)
    x = bf16(torch.randn(5, d))
    out = m(x)
    w, ids = route(x, mw.gate, mw.bias, cfg.num_experts_per_tok, cfg.routed_scaling_factor)
    want = torch.zeros(5, d)
    for t in range(5):
        for k in range(cfg.num_experts_per_tok):
            e = experts[int(ids[t, k])]
            want[t] += w[t, k] * e.w2(swiglu_clamp(e.w1(x[t:t + 1]), e.w3(x[t:t + 1]), 10.0))[0]
    sh = shared.w2(bf16(swiglu_clamp(bf16(shared.w1(x)), bf16(shared.w3(x)), 10.0)))
    assert torch.allclose(out, bf16(bf16(want) + bf16(sh)), atol=3e-2, rtol=2e-2)


def test_swiglu_clamp_matches_vllm():
    gate = torch.tensor([20.0, -20.0, 1.0])
    up = torch.tensor([20.0, -20.0, 2.0])
    y = swiglu_clamp(gate, up, 10.0)
    assert torch.allclose(y, torch.nn.functional.silu(torch.tensor([10.0, -20.0, 1.0])) * torch.tensor([10.0, -10.0, 2.0]))
