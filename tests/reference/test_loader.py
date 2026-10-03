"""The checkpoint loader on a synthetic EXL3 checkpoint in the V4.1 namespace (every dimension a multiple of 128):
the model the loader builds from the files must equal the model built directly from the same weights."""

import json

import pytest
import torch
from safetensors.torch import save_file

from engine.reference.attention import AttnWeights, CompressorWeights, IndexerWeights
from engine.reference.config import tiny_config
from engine.reference.engram import EngramLayout, EngramWeights, NgramHasher, TensorRows
from engine.reference.exl3 import MARKERS, quantize_random
from engine.reference.hc import HcParams
from engine.reference.loader import CheckpointLoader, SafetensorsDir
from engine.reference.model import DSpark, DSparkWeights, LayerWeights, Model, ModelWeights
from engine.reference.moe import ExpertWeights, MoEWeights
from engine.reference.ops import DenseLinear, Exl3Linear, Numerics, bf16

NUM = Numerics.exact()


def cfg128():
    return tiny_config(hidden_size=128, head_dim=128, qk_rope_head_dim=64, q_lora_rank=128, num_attention_heads=2,
                       o_groups=2, o_lora_rank=128, index_n_heads=2, index_head_dim=128, moe_intermediate_size=128,
                       vocab_size=256, engram_head_dim=32, dspark_noise_token_id=255)


class Builder:
    def __init__(self, cfg, seed=0):
        self.cfg, self.t, self.n, self.g = cfg, {}, 0, torch.Generator().manual_seed(seed)

    def exl3(self, name, k, n, bits=4):
        self.n += 1
        w = quantize_random(k, n, bits, "mul1", seed=self.n, scale=0.5 / k ** 0.5)
        self.t[f"{name}.trellis"], self.t[f"{name}.suh"], self.t[f"{name}.svh"] = w.trellis, w.suh, w.svh
        self.t[f"{name}.mul1"] = torch.tensor(MARKERS["mul1"] - 2 ** 32, dtype=torch.int32)
        return Exl3Linear(w, NUM)

    def native(self, name, shape, dtype=torch.bfloat16, scale=1.0, offset=0.0):
        v = (torch.randn(*shape, generator=self.g) * scale + offset).to(dtype)
        self.t[name] = v
        return v.float()

    def hc(self, p, which):
        c = self.cfg
        m = c.hc_mult * (c.hc_mult + 2)
        return HcParams(self.native(f"{p}.hc_{which}_fn", (m, c.hc_mult * c.hidden_size), torch.float32, 0.02),
                        self.native(f"{p}.hc_{which}_base", (m,), torch.float32, 0.5),
                        self.native(f"{p}.hc_{which}_scale", (3,), torch.float32, 0.1, 1.0))

    def layer(self, L, table):
        c = self.cfg
        p = f"layers.{L}" if L < c.num_hidden_layers else f"mtp.{L - c.num_hidden_layers}"
        d, hd = c.hidden_size, c.head_dim
        a = f"{p}.attn"
        comp = ix = None
        if c.is_kv_source(L):
            comp = CompressorWeights(self.exl3(f"{a}.compressor.wkv", d, hd),
                                     self.exl3(f"{a}.compressor.wgate", d, hd) if c.compress_ratio(L) > 1 else None,
                                     self.native(f"{a}.compressor.norm.weight", (hd,), scale=0.1, offset=1))
        if c.is_index_source(L):
            own = c.is_kv_source(L)
            wq = self.exl3(f"{a}.indexer.wq_b", c.q_lora_rank, c.index_n_heads * c.index_head_dim)
            wp = DenseLinear(bf16(self.native(f"{a}.indexer.weights_proj.weight", (c.index_n_heads, d),
                                              torch.float16, 0.1)))
            wk = self.exl3(f"{a}.indexer.wk", hd, c.index_head_dim, 8) if own else None
            kn = self.native(f"{a}.indexer.k_norm.weight", (c.index_head_dim,), scale=0.1, offset=1) if own else None
            ix = IndexerWeights(wq, wp, wk, kn)
        attn = AttnWeights(
            wq_a=self.exl3(f"{a}.wq_a", d, c.q_lora_rank, 6), wkv=self.exl3(f"{a}.wkv", d, hd, 6),
            wq_b=self.exl3(f"{a}.wq_b", c.q_lora_rank, c.num_attention_heads * hd, 5),
            wo_a=[self.exl3(f"{a}.wo_a.slice.{g}", c.num_attention_heads * hd // c.o_groups, c.o_lora_rank, 5)
                  for g in range(c.o_groups)],
            wo_b=self.exl3(f"{a}.wo_b", c.o_groups * c.o_lora_rank, d, 5),
            q_norm=self.native(f"{a}.q_norm.weight", (c.q_lora_rank,), scale=0.1, offset=1),
            kv_norm=self.native(f"{a}.kv_norm.weight", (hd,), scale=0.1, offset=1),
            attn_sink=self.native(f"{a}.attn_sink", (c.num_attention_heads,), torch.float32),
            compressor=comp, indexer=ix)
        n_exp, _ = c.experts_of(L)
        f = f"{p}.ffn"
        experts = [ExpertWeights(self.exl3(f"{f}.experts.{e}.w1", d, c.moe_intermediate_size, 3),
                                 self.exl3(f"{f}.experts.{e}.w2", c.moe_intermediate_size, d, 3),
                                 self.exl3(f"{f}.experts.{e}.w3", d, c.moe_intermediate_size, 3))
                   for e in range(n_exp)]
        shared = ExpertWeights(self.exl3(f"{f}.shared_experts.w1", d, c.moe_intermediate_size, 5),
                               self.exl3(f"{f}.shared_experts.w2", c.moe_intermediate_size, d, 5),
                               self.exl3(f"{f}.shared_experts.w3", d, c.moe_intermediate_size, 5))
        moe = MoEWeights(bf16(self.native(f"{f}.gate.weight", (n_exp, d), torch.float16, 0.1)),
                         self.native(f"{f}.gate.bias", (n_exp,), torch.float16, 0.01), lambda e: experts[e], shared)
        eng = None
        if L in c.engram_layer_ids:
            e = f"{p}.engram"
            eng = EngramWeights(self.exl3(f"{e}.wkv", c.n_hash_cols * c.engram_head_dim, (c.hc_mult + 1) * d, 5),
                                self.native(f"{e}.q_weight", (c.hc_mult, d), torch.float32),
                                self.native(f"{e}.k_weight", (c.hc_mult, d), torch.float32), table)
        return LayerWeights(attn_norm=self.native(f"{p}.attn_norm.weight", (d,), scale=0.1, offset=1),
                            ffn_norm=self.native(f"{p}.ffn_norm.weight", (d,), scale=0.1, offset=1),
                            hc_attn=self.hc(p, "attn"), hc_ffn=self.hc(p, "ffn"), attn=attn, moe=moe, engram=eng)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    cfg = cfg128()
    root = tmp_path_factory.mktemp("ckpt")
    eroot = tmp_path_factory.mktemp("engram")
    b = Builder(cfg)
    rows = EngramLayout(cfg).total_rows(0)
    raw = (torch.randn(rows, cfg.engram_head_dim, generator=b.g) * 2).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 126, (rows, cfg.engram_head_dim // 32), generator=b.g, dtype=torch.uint8)
    save_file({"layers.1.engram.embed.weight": raw,
               "layers.1.engram.embed.scale": scale.view(torch.float8_e8m0fnu)}, str(eroot / "model-e.safetensors"))
    table = TensorRows(raw, scale)
    embed = b.native("embed.weight", (cfg.vocab_size, cfg.hidden_size))
    layers = {L: b.layer(L, table) for L in range(cfg.num_hidden_layers)}
    norm = b.native("norm.weight", (cfg.hidden_size,), scale=0.1, offset=1)
    head = b.exl3("head", cfg.hidden_size, cfg.vocab_size, 6)
    blocks = [b.layer(cfg.num_hidden_layers + i, table) for i in range(cfg.num_nextn_predict_layers)]
    main_proj = b.exl3("mtp.0.main_proj", 3 * cfg.hidden_size, cfg.hidden_size, 4)
    main_norm = b.native("mtp.0.main_norm.weight", (cfg.hidden_size,), scale=0.1, offset=1)
    last = f"mtp.{cfg.num_nextn_predict_layers - 1}"
    dnorm = b.native(f"{last}.norm.weight", (cfg.hidden_size,), scale=0.1, offset=1)
    mw1 = b.native(f"{last}.markov_head.embed.weight", (cfg.vocab_size, cfg.dspark_markov_rank))
    mw2 = bf16(b.native(f"{last}.markov_head.head.weight", (cfg.vocab_size, cfg.dspark_markov_rank), torch.float16, .1))
    conf = b.native(f"{last}.confidence_head.proj.weight", (1, cfg.hidden_size + cfg.dspark_markov_rank),
                    torch.float16, 0.05)
    names = sorted(b.t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    wmap = {}
    for f, ns in shards.items():
        save_file({n: b.t[n].contiguous() for n in ns}, str(root / f))
        wmap.update({n: f for n in ns})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}))
    direct = ModelWeights(embed=lambda ids: embed[ids], layer=lambda L: layers[L], norm=norm, head=head)
    dspark = DSparkWeights(blocks, main_proj, main_norm, dnorm, mw1, mw2, conf)
    return cfg, root, eroot, direct, dspark


def test_loader_model_equals_direct_model(checkpoint):
    cfg, root, eroot, direct, _ = checkpoint
    hasher = NgramHasher(cfg, [i % cfg.engram_compressed_vocab_size for i in range(cfg.vocab_size)])
    ld = CheckpointLoader(cfg, SafetensorsDir(root), NUM, engram_src=SafetensorsDir(eroot))
    ids = torch.randint(0, cfg.vocab_size, (13,), generator=torch.Generator().manual_seed(9))
    got = Model(cfg, ld.model_weights(), NUM, hasher).forward(ids, release=True)
    want = Model(cfg, direct, NUM, hasher).forward(ids)
    assert torch.isfinite(want.logits).all()
    assert torch.allclose(got.logits, want.logits, atol=1e-4, rtol=1e-4)
    for L in range(cfg.num_hidden_layers):
        assert torch.equal(got.routing[L], want.routing[L])


def test_loader_dspark_equals_direct(checkpoint):
    cfg, root, eroot, direct, dspark = checkpoint
    ld = CheckpointLoader(cfg, SafetensorsDir(root), NUM, engram_src=SafetensorsDir(eroot))
    main_x = bf16(torch.randn(9, cfg.hidden_size))
    a = DSpark(cfg, ld.dspark(), ld.embed, ld.head(), NUM).draft(main_x, 7)
    b = DSpark(cfg, dspark, direct.embed, direct.head, NUM).draft(main_x, 7)
    assert torch.allclose(a.logits, b.logits, atol=1e-4, rtol=1e-4) and torch.equal(a.tokens, b.tokens)
    assert torch.allclose(a.confidence, b.confidence, atol=1e-5)


def test_bits_come_from_the_trellis(checkpoint):
    cfg, root, *_ = checkpoint
    ld = CheckpointLoader(cfg, SafetensorsDir(root), NUM)
    assert ld.exl3_weight("layers.0.attn.wq_a").bits == 6
    assert ld.exl3_weight("layers.0.ffn.experts.0.w1").bits == 3
    assert ld.exl3_weight("layers.2.attn.indexer.wk").bits == 8
    assert ld.exl3_weight("head").codebook == "mul1"
