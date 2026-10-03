"""DeepSeek-V4.1-Flash's per-layer roles from ``config.json`` alone: which caches a layer owns, which it reads, and
which CSA2 kernels it runs. The engine, the KV pool layout, the session snapshot and the memory budget all derive
from this one table (no torch).

Roles (vLLM ``deepseek_v41/attention.py`` semantics, Apache-2.0; the checkpoint's config):

- ``swa``      compress ratio 0 (layers 0-1, the DSpark blocks): the 128-token window only;
- ``full``     a kv source (2, 8, 14, 20): compressor + compressed KV + index keys, scans every visible position;
- ``reindex``  an index source that is not a kv source (24, 28, 32, 36): its own index query over layer 20's keys,
               only on layer 20's candidate blocks;
- ``reuse``    every other compressed layer: the latest kv source's KV and the latest index source's selection.

Layer 20 (``candidate_source_layer_id``) also picks the 2,048 candidate blocks of 8. Encoder = layers 0-19 (ratio
2 after layer 1), decoder = 20-39 (ratio 1): CED's split, whose decoder global KV is layer 20's projection of H_19.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Layer:
    index: int
    ratio: int                       # 0 (window only), 1, 2
    role: str                        # swa | full | reindex | reuse
    kv_source: int | None            # whose compressed KV it reads
    index_source: int | None         # whose selection it attends
    candidates: str                  # "" | "write" (layer 20) | "read" (reindex layers past 20)
    engram: bool
    encoder: bool                    # CED: layers before the decoder's first layer
    dspark: bool = False


@dataclass(frozen=True)
class Topology:
    layers: tuple[Layer, ...]
    dspark: tuple[Layer, ...]
    window: int
    index_topk: int
    candidate_blocks: int
    candidate_block_size: int
    decoder_start: int
    taps: tuple[int, ...]            # DSpark reads the streams after these layers
    hidden: int
    heads: int
    head_dim: int
    vocab: int
    engram_layers: tuple[int, ...]
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def by_role(self, role: str) -> list[int]:
        return [ly.index for ly in self.layers if ly.role == role]

    @property
    def kv_sources(self) -> list[int]:
        return self.by_role("full")

    def compressed_rows_per_token(self) -> dict[int, float]:
        """{kv source: compressed rows a token}: 1 / ratio."""

        return {ly.index: 1.0 / ly.ratio for ly in self.layers if ly.role == "full"}


def text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("text_config", config)


def build(config: Mapping[str, Any]) -> Topology:
    """The topology of a ``config.json`` (top level or its ``text_config``)."""

    t = text_config(config)
    n = int(t["num_hidden_layers"])
    ratios = [int(r) for r in t["compress_ratios"]]
    kv_src = sorted(int(x) for x in t["kv_source_layer_ids"])
    idx_src = sorted(int(x) for x in t["index_source_layer_ids"])
    cand = int(t.get("candidate_source_layer_id", -1))
    engram = tuple(int(x) for x in t.get("engram_layer_ids", ()))
    if not set(kv_src) <= set(idx_src):
        raise ValueError("every kv source must be an index source (it owns the index keys)")
    decoder = min((i for i in range(n) if ratios[i] == 1), default=n)
    layers = []
    for i in range(n):
        r = ratios[i] if i < len(ratios) else 0
        if r == 0:
            layers.append(Layer(i, 0, "swa", None, None, "", i in engram, i < decoder))
            continue
        if r not in (1, 2):
            raise ValueError(f"layer {i}: compress ratio {r} (V4.1 has 0, 1, 2)")
        ks = max((s for s in kv_src if s <= i), default=None)
        js = max((s for s in idx_src if s <= i), default=None)
        if ks is None or js is None:
            raise ValueError(f"layer {i}: compressed but no kv / index source at or below it")
        role = "full" if i in kv_src else ("reindex" if i in idx_src else "reuse")
        c = "write" if i == cand else ("read" if role == "reindex" and 0 <= cand < i else "")
        if role == "reuse" and ratios[ks] != r:
            raise ValueError(f"layer {i}: ratio {r} reuses layer {ks}'s ratio-{ratios[ks]} KV")
        layers.append(Layer(i, r, role, ks, js, c, i in engram, i < decoder))
    nd = int(t.get("num_nextn_predict_layers", 0))
    dspark = tuple(Layer(n + j, 0, "swa", None, None, "", False, False, True) for j in range(nd))
    return Topology(tuple(layers), dspark, int(t["sliding_window"]), int(t["index_topk"]),
                    int(t.get("candidate_topk_blocks", 0)), int(t.get("candidate_block_size", 0)), decoder,
                    tuple(int(x) for x in t.get("dspark_target_layer_ids", ())), int(t["hidden_size"]),
                    int(t["num_attention_heads"]), int(t["head_dim"]), int(t["vocab_size"]), engram, dict(t))


def load(model_dir: str | Path) -> Topology:
    return build(json.loads((Path(model_dir) / "config.json").read_text()))
