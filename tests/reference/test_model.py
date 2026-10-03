"""Whole-model properties on the tiny config: causality (a prefix's logits do not depend on later tokens), layer
streaming, the DSpark drafter."""

import pytest
import torch

from engine.reference import synthetic as S
from engine.reference.engram import NgramHasher
from engine.reference.model import DSpark, Model
from engine.reference.ops import Numerics


def _model(cfg, num):
    w = S.model_weights(cfg)
    return Model(cfg, w, num, NgramHasher(cfg, [i % cfg.engram_compressed_vocab_size for i in range(cfg.vocab_size)]))


@pytest.mark.parametrize("num", [Numerics.exact(), Numerics.kit()], ids=["exact", "kit"])
def test_causal_prefix_invariance(cfg, num):
    m = _model(cfg, num)
    ids = torch.randint(0, cfg.vocab_size, (21,), generator=torch.Generator().manual_seed(3))
    full = m.forward(ids)
    # long enough that the indexer leaves its "select everything" path on the ratio-1 layers (21 > topk 3)
    for cut in (5, 12, 17):
        part = m.forward(ids[:cut])
        assert torch.allclose(part.logits, full.logits[:cut], atol=1e-5), cut
    assert torch.isfinite(full.logits).all()
    assert set(full.aux) == set(cfg.aux_layer_ids)


def test_future_tokens_do_not_leak_but_past_ones_matter(cfg):
    m = _model(cfg, Numerics.exact())
    a = torch.randint(0, cfg.vocab_size, (16,), generator=torch.Generator().manual_seed(4))
    b = a.clone()
    b[2] = (b[2] + 1) % cfg.vocab_size
    la, lb = m.forward(a).logits, m.forward(b).logits
    assert torch.equal(la[:2], lb[:2])
    assert not torch.allclose(la[15], lb[15])       # reaches through the compressed / selected rows


def test_routing_is_recorded_and_streams_shaped(cfg):
    m = _model(cfg, Numerics.exact())
    out = m.forward(torch.arange(10), keep_layer_out=(0, 7))
    assert out.streams.shape == (10, cfg.hc_mult, cfg.hidden_size)
    assert out.routing[3].shape == (10, cfg.num_experts_per_tok)
    assert set(out.layer_out) == {0, 7}


def test_dspark_block_is_noncausal_and_reads_a_window(cfg):
    m = _model(cfg, Numerics.exact())
    ids = torch.randint(0, cfg.vocab_size, (14,), generator=torch.Generator().manual_seed(5))
    out = m.forward(ids)
    ds = DSpark(cfg, S.dspark_weights(cfg), m.w.embed, m.w.head, Numerics.exact())
    mx = ds.main_x(out.aux)
    assert mx.shape == (14, cfg.hidden_size)
    d1 = ds.draft(mx[:10], int(ids[10]))
    assert d1.tokens.shape == (cfg.dspark_block_size,)
    assert d1.confidence is not None and ((d1.confidence > 0) & (d1.confidence < 1)).all()
    # deterministic
    assert torch.equal(ds.draft(mx[:10], int(ids[10])).tokens, d1.tokens)
    # only the last sliding_window context rows matter
    mx2 = mx.clone()
    mx2[: 10 - cfg.sliding_window] = 0
    assert torch.allclose(ds.draft(mx2[:10], int(ids[10])).base_logits, d1.base_logits)
    mx3 = mx.clone()
    mx3[9] += 1
    assert not torch.allclose(ds.draft(mx3[:10], int(ids[10])).base_logits, d1.base_logits)
    # the Markov bias: logits = base + w2 @ w1[previous token]
    w = ds.w
    want = d1.base_logits[1] + w.markov_w1[int(d1.tokens[0])] @ w.markov_w2.t()
    assert torch.allclose(d1.logits[1], want, atol=1e-4)


def test_layer_major_batch_equals_one_by_one(cfg):
    m = _model(cfg, Numerics.kit())
    g = torch.Generator().manual_seed(8)
    seqs = [torch.randint(0, cfg.vocab_size, (n,), generator=g) for n in (7, 15, 4)]
    many = m.forward_many(seqs)
    for s, o in zip(seqs, many):
        one = m.forward(s)
        assert torch.allclose(o.logits, one.logits, atol=1e-5)
        assert all(torch.equal(o.routing[L], one.routing[L]) for L in range(cfg.num_hidden_layers))


def test_kit_logits_are_bf16_exact_ones_are_not(cfg):
    """The kit's head returns bf16 logits (G2: its top-1 / top-2 margins are multiples of 1/16, its argmax takes the
    lower id on exact ties); kit numerics round the reference's logits the same way, exact numerics do not."""

    ids = torch.randint(0, cfg.vocab_size, (9,), generator=torch.Generator().manual_seed(5))
    kit = _model(cfg, Numerics.kit()).forward(ids).logits
    assert torch.equal(kit, kit.to(torch.bfloat16).float())
    ex = _model(cfg, Numerics.exact()).forward(ids).logits
    assert not torch.equal(ex, ex.to(torch.bfloat16).to(ex.dtype))
