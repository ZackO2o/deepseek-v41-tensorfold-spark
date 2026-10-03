"""DeepSeek-V4.1-Flash text config (checkpoint ``config.json`` -> ``text_config``) and the per-layer CSA2 topology.

Every derived rule here follows vLLM's ``vllm/models/deepseek_v4_1/attention.py`` (Apache-2.0):

- ``compress_ratios[L]``: 0 = sliding window only, 1 = full-length compressed cache, 2 = ratio-2 compressed.
  The list runs past ``num_hidden_layers`` to describe the DSpark blocks (all 0).
- KV source of layer L (ratio > 0): the latest ``kv_source_layer_ids`` entry <= L. Only sources own a compressor
  (and the indexer K); the others read the source's compressed cache ("Reuse").
- Index source of layer L (ratio > 0): the latest ``index_source_layer_ids`` entry <= L. Only index sources run an
  indexer; the others reuse the top-k it published ("Reuse"). Index sources that are not KV sources ("Reindex") score
  the KV source's indexer keys with their own query.
- The indexer at ``candidate_source_layer_id`` also publishes candidate blocks; later index sources mask to them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RopeConfig:
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    factor: float = 16.0
    original_max_position_embeddings: int = 65536
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    max_position_embeddings: int = 1048576
    yarn: bool = True            # rope_scaling.rope_type != "default"


@dataclass(frozen=True)
class Config:
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    rms_norm_eps: float = 1e-20
    sliding_window: int = 128
    # MoE
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2304
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    # CSA2
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)
    kv_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    hc_post_alpha: float = 2.0
    # Engram
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092
    engram_gate_clamp: float = 1e-6
    # DSpark (checkpoint mtp.*)
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple[int, ...] = (37, 38, 39)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    rope: RopeConfig = field(default_factory=RopeConfig)
    bos_token_id: int = 0
    eos_token_id: int = 1

    # -- the CSA2 topology ----------------------------------------------------------------------------------------

    def compress_ratio(self, layer: int) -> int:
        return int(self.compress_ratios[layer]) if layer < len(self.compress_ratios) else 0

    def is_backbone(self, layer: int) -> bool:
        return layer < self.num_hidden_layers

    def kv_source(self, layer: int) -> int | None:
        if self.compress_ratio(layer) == 0:
            return None
        return max(s for s in self.kv_source_layer_ids if s <= layer)

    def index_source(self, layer: int) -> int | None:
        if self.compress_ratio(layer) == 0:
            return None
        return max(s for s in self.index_source_layer_ids if s <= layer)

    def is_kv_source(self, layer: int) -> bool:
        return self.is_backbone(layer) and layer in self.kv_source_layer_ids

    def is_index_source(self, layer: int) -> bool:
        return self.is_backbone(layer) and layer in self.index_source_layer_ids

    def is_candidate_source(self, layer: int) -> bool:
        return layer == self.candidate_source_layer_id and self.candidate_topk_blocks > 0

    def uses_candidates(self, layer: int) -> bool:
        return (self.is_index_source(layer) and self.candidate_topk_blocks > 0
                and 0 <= self.candidate_source_layer_id < layer)

    def attention_mode(self, layer: int) -> str:
        """``swa`` / ``full`` (KV source + indexer) / ``reindex`` / ``reuse``, as DSV41-BASELINE.md names them."""

        if self.compress_ratio(layer) == 0:
            return "swa"
        if self.is_kv_source(layer):
            return "full"
        if self.is_index_source(layer):
            return "reindex"
        return "reuse"

    def experts_of(self, layer: int) -> tuple[int, int]:
        """(routed experts, experts a token) of a backbone layer or a DSpark block."""

        if self.is_backbone(layer):
            return self.n_routed_experts, self.num_experts_per_tok
        return (self.dspark_n_routed_experts or self.n_routed_experts,
                self.dspark_num_experts_per_tok or self.num_experts_per_tok)

    @property
    def n_hash_cols(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_heads

    @property
    def aux_layer_ids(self) -> tuple[int, ...]:
        """Backbone stream taps for DSpark: the entry stream of each target layer (vLLM eagle3_utils, v4.1 rule)."""

        return tuple(self.dspark_target_layer_ids)

    # -- construction ---------------------------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "Config":
        text = cfg.get("text_config") or cfg
        rs = text.get("rope_scaling") or text.get("rope_parameters") or {}
        rope = RopeConfig(
            rope_theta=float(text.get("rope_theta", 10000.0)),
            compress_rope_theta=float(text.get("compress_rope_theta", 160000.0)),
            factor=float(rs.get("factor", 1.0)),
            original_max_position_embeddings=int(rs.get("original_max_position_embeddings",
                                                        text.get("max_position_embeddings", 1048576))),
            beta_fast=float(rs.get("beta_fast", 32)),
            beta_slow=float(rs.get("beta_slow", 1)),
            max_position_embeddings=int(text.get("max_position_embeddings", 1048576)),
            yarn=str(rs.get("rope_type", rs.get("type", "default"))) != "default",
        )
        names = {f for f in cls.__dataclass_fields__ if f != "rope"}
        kw: dict[str, Any] = {}
        for k in names:
            if k in text:
                v = text[k]
                kw[k] = tuple(v) if isinstance(v, list) else v
        if "bos_token_id" in cfg:
            kw["bos_token_id"] = cfg["bos_token_id"]
        if "eos_token_id" in cfg:
            kw["eos_token_id"] = cfg["eos_token_id"]
        return cls(rope=rope, **kw)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def with_(self, **kw: Any) -> "Config":
        return replace(self, **kw)


def tiny_config(**overrides: Any) -> Config:
    """A small config with every V4.1 mechanism present: SWA layers, a ratio-2 KV source and its consumers, a
    ratio-1 KV source that is also the candidate source, a Reindex layer, Engram on layer 1, and two DSpark blocks."""

    base = Config(
        vocab_size=97, hidden_size=64, num_hidden_layers=8, num_attention_heads=4, head_dim=32, qk_rope_head_dim=8,
        q_lora_rank=32, o_lora_rank=16, o_groups=2, rms_norm_eps=1e-6, sliding_window=4,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2, moe_intermediate_size=24,
        compress_ratios=(0, 0, 2, 2, 1, 1, 1, 1, 0, 0),
        kv_source_layer_ids=(2, 4), index_source_layer_ids=(2, 4, 6),
        index_n_heads=2, index_head_dim=16, index_topk=3,
        candidate_source_layer_id=4, candidate_topk_blocks=2, candidate_block_size=2,
        engram_layer_ids=(1,), engram_num_embeddings=(0,), engram_max_ngram_size=3, engram_vocab_size=31,
        engram_n_heads=2, engram_head_dim=8, engram_pad_token_id=2, engram_compressed_vocab_size=50,
        num_nextn_predict_layers=2, dspark_block_size=3, dspark_noise_token_id=96, dspark_target_layer_ids=(5, 6, 7),
        dspark_markov_rank=8, dspark_n_routed_experts=4, dspark_num_experts_per_tok=2,
        rope=RopeConfig(rope_theta=10000.0, compress_rope_theta=160000.0, factor=4.0,
                        original_max_position_embeddings=64, max_position_embeddings=256),
    )
    from .engram import EngramLayout   # the table size follows from the primes
    layout = EngramLayout.from_config(base)
    base = base.with_(engram_num_embeddings=tuple(layout.total_rows(i) for i in range(len(base.engram_layer_ids))))
    return base.with_(**overrides) if overrides else base
