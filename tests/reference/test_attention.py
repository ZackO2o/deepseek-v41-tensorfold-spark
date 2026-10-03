"""CSA2: topology, the sink softmax, the compressor, the indexer selection and the candidate filter."""

import math

import torch

from engine.reference.attention import (Attention, CsaState, attend, dense_attn_weights, select_candidate_blocks,
                                        window_mask)
from engine.reference.config import Config
from engine.reference.ops import Numerics, bf16


def test_v41_topology_matches_the_baseline_study():
    cfg = Config()
    modes = [cfg.attention_mode(L) for L in range(43)]
    assert modes[:2] == ["swa", "swa"] and modes[40:] == ["swa"] * 3
    assert [L for L, m in enumerate(modes) if m == "full"] == [2, 8, 14, 20]
    assert [L for L, m in enumerate(modes) if m == "reindex"] == [24, 28, 32, 36]
    assert cfg.kv_source(7) == 2 and cfg.kv_source(13) == 8 and cfg.kv_source(19) == 14
    assert cfg.kv_source(39) == 20 and cfg.index_source(39) == 36 and cfg.index_source(23) == 20
    assert all(cfg.compress_ratio(L) == 2 for L in range(2, 20))
    assert all(cfg.compress_ratio(L) == 1 for L in range(20, 40))
    assert [L for L in range(40) if cfg.uses_candidates(L)] == [24, 28, 32, 36]
    assert cfg.experts_of(5) == (384, 6) and cfg.experts_of(41) == (128, 3)
    assert cfg.aux_layer_ids == (37, 38, 39)


def test_attend_is_sink_softmax_with_v_equal_k():
    q = torch.randn(3, 2, 8)
    k = torch.randn(5, 8)
    vis = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 1], [0, 0, 1, 1, 1]], dtype=torch.bool)
    sink = torch.tensor([0.3, -1.0])
    out = attend(q, k, vis, sink, chunk=2)
    for t in range(3):
        for h in range(2):
            s = [(q[t, h] @ k[j]) / math.sqrt(8) for j in range(5) if vis[t, j]]
            ks = [k[j] for j in range(5) if vis[t, j]]
            e = torch.exp(torch.stack(s))
            den = e.sum() + torch.exp(sink[h])
            want = sum(e[i] / den * ks[i] for i in range(len(ks)))
            assert torch.allclose(out[t, h], want, atol=1e-5)


def test_window_mask():
    m = window_mask(torch.arange(6), torch.arange(6), 3)
    assert m[5].tolist() == [False, False, False, True, True, True] and m[0].tolist()[0]


def _attn(cfg, layer, num=Numerics.exact()):
    return Attention(cfg, layer, dense_attn_weights(cfg, layer, torch.Generator().manual_seed(layer)), num)


def test_ratio2_compressor_is_a_softmax_gated_pair(cfg):
    a = _attn(cfg, 2)
    x = bf16(torch.randn(7, cfg.hidden_size))
    latent, gpos = a.compress(x, torch.arange(7))
    assert latent.shape == (3, cfg.head_dim) and gpos.tolist() == [0, 2, 4]       # the open group (6) waits
    c = a.w.compressor
    kv, sc = bf16(c.wkv(x)), bf16(c.wgate(x))
    g = 1
    w0 = torch.exp(sc[2]) / (torch.exp(sc[2]) + torch.exp(sc[3]))
    pooled = kv[2] * w0 + kv[3] * (1 - w0)
    want = pooled * torch.rsqrt(pooled.square().mean() + cfg.rms_norm_eps) * c.norm
    assert torch.allclose(latent[g], bf16(want), atol=1e-2, rtol=1e-2)


def test_selection_takes_all_when_few_and_topk_otherwise(cfg):
    a = _attn(cfg, 4)                       # ratio 1, topk 3, candidate source
    state = CsaState()
    t = 6
    logits = torch.randn(t, t)
    sel = a.select(logits, torch.arange(t), state)
    for row in range(t):
        n_valid = row + 1
        got = sorted(int(i) for i in sel[row] if i >= 0)
        if n_valid <= cfg.index_topk:
            assert got == list(range(n_valid))
        else:
            assert got == sorted(logits[row, :n_valid].topk(cfg.index_topk).indices.tolist())
    assert state.candidates is not None and state.candidates.shape == (t, 3)


def test_candidate_blocks_pin_the_newest_and_mask_later_layers(cfg):
    logits = torch.tensor([[5.0, 4.0, -9.0, -9.0, 1.0, 0.0, 0.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)
    keep = select_candidate_blocks(logits, valid, block=2, topk_blocks=2)
    assert keep.tolist() == [[True, False, False, True]]          # best block 0, newest block 3 pinned
    a6 = _attn(cfg, 6)                      # reindex layer, uses the layer-4 candidates
    state = CsaState(candidates=torch.tensor([[True, False, False, True]]).expand(7, -1).clone())
    scores = torch.zeros(7, 7)
    scores[:, 2:4] = 100.0                  # the best positions sit in a masked block
    sel = a6.select(scores, torch.arange(7), state)
    assert not any(int(i) in (2, 3) for i in sel[6] if i >= 0)


def test_layers_reuse_their_sources_cache(cfg):
    x = bf16(torch.randn(9, cfg.hidden_size))
    pos = torch.arange(9)
    state = CsaState()
    _attn(cfg, 2)(x, pos, state)
    assert set(state.ckv) == {2} and state.ckv[2].shape == (4, cfg.head_dim)
    topk_2 = state.topk.clone()
    _attn(cfg, 3)(x, pos, state)            # reuse: same cache, same top-k
    assert torch.equal(state.topk, topk_2) and set(state.ckv) == {2}
    _attn(cfg, 4)(x, pos, state)            # ratio-1 source: own cache and new top-k
    assert state.ckv[4].shape == (9, cfg.head_dim) and state.ikey[4].shape == (9, cfg.index_head_dim)


def test_kit_numerics_only_perturb(cfg):
    x = bf16(torch.randn(9, cfg.hidden_size))
    a = _attn(cfg, 2, Numerics.exact())
    b = Attention(cfg, 2, a.w, Numerics.kit())
    ya = a(x, torch.arange(9), CsaState())
    yb = b(x, torch.arange(9), CsaState())
    assert (ya - yb).norm() / ya.norm() < 0.05
