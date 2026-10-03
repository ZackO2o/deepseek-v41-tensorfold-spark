"""Random weights for any ``Config`` (the unit tests use ``tiny_config()``): dense linears, or EXL3-packed ones with
random trellises when every dimension is a multiple of 128."""

from __future__ import annotations

import torch

from .attention import dense_attn_weights
from .config import Config
from .engram import EngramLayout, EngramWeights, TensorRows
from .hc import HcParams
from .model import DSparkWeights, LayerWeights, ModelWeights
from .moe import ExpertWeights, MoEWeights
from .ops import DenseLinear, bf16


def _lin(gen: torch.Generator, i: int, o: int, std: float) -> DenseLinear:
    return DenseLinear(bf16(torch.randn(o, i, generator=gen) * std))


def _hc(cfg: Config, gen: torch.Generator) -> HcParams:
    m = cfg.hc_mult * (cfg.hc_mult + 2)
    return HcParams(fn=torch.randn(m, cfg.hc_mult * cfg.hidden_size, generator=gen) * 0.02,
                    base=torch.randn(m, generator=gen) * 0.5, scale=torch.rand(3, generator=gen) + 0.5)


def layer_weights(cfg: Config, layer: int, seed: int = 0, std: float = 0.05,
                  table: TensorRows | None = None) -> LayerWeights:
    gen = torch.Generator().manual_seed(1000 + 17 * layer + seed)
    d = cfg.hidden_size
    n_exp, _ = cfg.experts_of(layer)
    inter = cfg.moe_intermediate_size
    experts = [ExpertWeights(_lin(gen, d, inter, std), _lin(gen, inter, d, std), _lin(gen, d, inter, std))
               for _ in range(n_exp)]
    shared = ExpertWeights(_lin(gen, d, inter, std), _lin(gen, inter, d, std), _lin(gen, d, inter, std))
    moe = MoEWeights(gate=bf16(torch.randn(n_exp, d, generator=gen) * 0.1),
                     bias=torch.randn(n_exp, generator=gen) * 0.01, experts=lambda e: experts[e], shared=shared)
    eng = None
    if layer in cfg.engram_layer_ids:
        li = cfg.engram_layer_ids.index(layer)
        rows = EngramLayout(cfg).total_rows(li)
        tab = table or TensorRows(bf16(torch.randn(rows, cfg.engram_head_dim, generator=gen)))
        eng = EngramWeights(wkv=_lin(gen, cfg.n_hash_cols * cfg.engram_head_dim, (cfg.hc_mult + 1) * d, std),
                            q_weight=torch.randn(cfg.hc_mult, d, generator=gen),
                            k_weight=torch.randn(cfg.hc_mult, d, generator=gen), table=tab)
    return LayerWeights(attn_norm=bf16(1 + 0.1 * torch.randn(d, generator=gen)),
                        ffn_norm=bf16(1 + 0.1 * torch.randn(d, generator=gen)),
                        hc_attn=_hc(cfg, gen), hc_ffn=_hc(cfg, gen), attn=dense_attn_weights(cfg, layer, gen, std),
                        moe=moe, engram=eng)


def model_weights(cfg: Config, seed: int = 0) -> ModelWeights:
    gen = torch.Generator().manual_seed(seed)
    embed = bf16(torch.randn(cfg.vocab_size, cfg.hidden_size, generator=gen))
    layers = {L: layer_weights(cfg, L, seed) for L in range(cfg.num_hidden_layers)}
    return ModelWeights(embed=lambda ids: embed[ids], layer=lambda L: layers[L],
                        norm=bf16(1 + 0.1 * torch.randn(cfg.hidden_size, generator=gen)),
                        head=_lin(gen, cfg.hidden_size, cfg.vocab_size, 0.05))


def dspark_weights(cfg: Config, seed: int = 0) -> DSparkWeights:
    gen = torch.Generator().manual_seed(seed + 77)
    d, r = cfg.hidden_size, cfg.dspark_markov_rank
    blocks = [layer_weights(cfg, cfg.num_hidden_layers + i, seed) for i in range(cfg.num_nextn_predict_layers)]
    return DSparkWeights(blocks=blocks, main_proj=_lin(gen, len(cfg.aux_layer_ids) * d, d, 0.05),
                         main_norm=bf16(1 + 0.1 * torch.randn(d, generator=gen)),
                         norm=bf16(1 + 0.1 * torch.randn(d, generator=gen)),
                         markov_w1=bf16(torch.randn(cfg.vocab_size, r, generator=gen)),
                         markov_w2=bf16(torch.randn(cfg.vocab_size, r, generator=gen) * 0.1),
                         confidence=torch.randn(1, d + r, generator=gen) * 0.05)
