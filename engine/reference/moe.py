"""The MoE FFN: sqrt-softplus router with the noaux_tc selection bias, normalized top-k x 1.5, routed SwiGLU experts
clamped at 10, plus one shared expert. 384 experts top-6 on the backbone, 128 top-3 on the DSpark blocks.

From vLLM ``deepseek_v4/nvidia/model.py`` (``DeepseekV4MoE``, ``DeepseekV4MLP``), the fused_topk_bias router
(``vllm_topk_softplus_sqrt``) and the kit's EXL3 expert loop (``/opt/dsv41/exl3.py: apply_exl3_python_loop``):

- ``logits = x @ gate.T`` in fp32 (bf16 gate weight);
- ``s = sqrt(softplus(logits))``; the chosen experts are the top-k of ``s + bias``; weights are their ``s``,
  divided by their sum, times 1.5;
- routed expert e: ``g = x @ W1_e``, ``u = x @ W3_e`` (fp16 input, fp32 out),
  ``down = (silu(min(g, 10)) * clamp(u, -10, 10)) @ W2_e`` (fp16 input, fp32 out), summed in fp32 with the weights
  (fp16 in kit numerics: the kit's fused / grouped EXL3 expert kernels under moe_x take them as fp16);
- shared expert: same SwiGLU with bf16 activations between the linears (vLLM's ``DeepseekV4MLP``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .config import Config
from .ops import F32, Linear, Numerics, bf16, linear_out, swiglu_clamp


@dataclass
class ExpertWeights:
    w1: Linear          # gate: hidden -> inter
    w2: Linear          # down: inter -> hidden
    w3: Linear          # up:   hidden -> inter


@dataclass
class MoEWeights:
    gate: torch.Tensor                                  # [E, hidden] (bf16 values)
    bias: torch.Tensor                                  # [E] fp32 (e_score_correction_bias)
    experts: Callable[[int], ExpertWeights]             # lazy: expert id -> weights
    shared: ExpertWeights | None
    prefetch: Callable[[list[int]], None] | None = None     # batch-fetch the routed experts (remote sources)
    done: Callable[[list[int]], None] | None = None         # forget them after the layer


def route(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, top_k: int, scaling: float,
          renormalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """(weights [T, k] fp32, expert ids [T, k])."""

    logits = x.to(F32) @ gate.to(F32).t()
    scores = torch.sqrt(F.softplus(logits))
    ids = (scores + bias.to(F32)[None, :]).topk(top_k, dim=-1).indices
    w = scores.gather(1, ids)
    if renormalize:
        w = w / w.sum(-1, keepdim=True)
    return w * scaling, ids


class MoE:
    def __init__(self, cfg: Config, layer: int, w: MoEWeights, numerics: Numerics | None = None):
        self.cfg, self.layer, self.w = cfg, layer, w
        self.num = numerics or Numerics()
        self.n_experts, self.top_k = cfg.experts_of(layer)
        self.last_ids: torch.Tensor | None = None

    def routed(self, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        limit = self.cfg.swiglu_limit
        out = torch.zeros((x.shape[0], x.shape[1]), dtype=F32, device=x.device)
        unique = torch.unique(ids).tolist()
        if self.w.prefetch is not None:
            self.w.prefetch(unique)
        for e in unique:
            tok, slot = (ids == e).nonzero(as_tuple=True)
            ew = self.w.experts(int(e))
            h = x.index_select(0, tok)
            act = swiglu_clamp(ew.w1(h), ew.w3(h), limit)
            down = ew.w2(act)
            out.index_add_(0, tok, down * weights[tok, slot].unsqueeze(-1))
        if self.w.done is not None:
            self.w.done(unique)
        return out

    def shared(self, x: torch.Tensor) -> torch.Tensor:
        s = self.w.shared
        assert s is not None
        act = bf16(swiglu_clamp(linear_out(s.w1, x), linear_out(s.w3, x), self.cfg.swiglu_limit))
        return linear_out(s.w2, act)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        weights, ids = route(x, self.w.gate, self.w.bias, self.top_k, self.cfg.routed_scaling_factor,
                             self.cfg.norm_topk_prob)
        self.last_ids = ids
        if self.num.routed_fp16_weights:
            weights = weights.to(torch.float16).to(F32)
        out = bf16(self.routed(x, weights, ids))
        if self.w.shared is not None:
            out = bf16(out + self.shared(x))
        return out
