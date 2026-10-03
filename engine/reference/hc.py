"""Manifold-constrained hyper-connections (mHC), Single-Pass ("delayed pre-mix") form, as vLLM's V4.1 path runs it.

The residual is ``hc_mult`` (4) bf16 streams ``[T, 4, D]``. Each sublayer (attention, FFN) has
``fn [24, 4D]`` (fp32), ``base [24]`` and ``scale [3]``. From the *current* streams it computes:

- ``mixes = (streams.flat @ fn.T) * rsqrt(mean(streams.flat^2) + rms_eps)``;
- ``pre  = sigmoid(mixes[:4] * scale[0] + base[:4]) + hc_eps``          (published for the NEXT sublayer);
- ``post = sigmoid(mixes[4:8] * scale[1] + base[4:8]) * 2``;
- ``comb = Sinkhorn(softmax(mixes[8:].view(4, 4) * scale[2] + base[8:]))``, 20 iterations.

The sublayer input is collapsed with the pre-mix carried in from the PREVIOUS sublayer (identity = stream 0 at model
entry), rounded to bf16, then RMSNorm'd with the sublayer's norm weight. Afterwards
``streams'[j] = sum_i comb[i, j] * streams[i] + post[j] * out``. The final hidden state is the streams collapsed
with the last FFN's pre-mix (V4.1 has no learned ``hc_head``). Sources: ``kernels/mhc/torch.py`` (reference ops),
``tilelang_kernels.py`` (the bf16 boundary before the delayed RMSNorm), ``deepseek_v4_1/nvidia/model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ops import F32, bf16, rms_norm


@dataclass
class HcParams:
    fn: torch.Tensor        # [hc * (hc + 2), hc * D] fp32
    base: torch.Tensor      # [hc * (hc + 2)] fp32
    scale: torch.Tensor     # [3] fp32


def hc_mixes(streams: torch.Tensor, p: HcParams, rms_eps: float, hc_eps: float, post_alpha: float,
             iters: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pre [T, hc], post [T, hc], comb [T, hc, hc]) from the streams [T, hc, D]."""

    t, hc, _ = streams.shape
    x = streams.reshape(t, -1).to(F32)
    mixes = (x @ p.fn.to(F32).t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    scale, base = p.scale.to(F32), p.base.to(F32)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_eps
    post = torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc]) * post_alpha
    comb = mixes[:, 2 * hc:].reshape(t, hc, hc) * scale[2] + base[2 * hc:].reshape(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    return pre, post, comb


def hc_collapse(streams: torch.Tensor, pre_mix: torch.Tensor | None) -> torch.Tensor:
    """sum_i pre[i] * streams[i] (fp32), rounded to bf16; stream 0 when no pre-mix is carried (model entry)."""

    if pre_mix is None:
        return bf16(streams[:, 0])
    return bf16((pre_mix.to(F32).unsqueeze(-1) * streams.to(F32)).sum(dim=1))


def hc_pre(streams: torch.Tensor, p: HcParams, pre_mix_in: torch.Tensor | None, norm_weight: torch.Tensor,
           rms_eps: float, norm_eps: float, hc_eps: float, post_alpha: float, iters: int):
    """Returns (post, comb, sublayer input [T, D] bf16 values, pre-mix for the next sublayer)."""

    pre, post, comb = hc_mixes(streams, p, rms_eps, hc_eps, post_alpha, iters)
    x = hc_collapse(streams, pre_mix_in)
    return post, comb, bf16(rms_norm(x, norm_weight, norm_eps)), pre


def hc_post(out: torch.Tensor, streams: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """streams'[j] = sum_i comb[i, j] * streams[i] + post[j] * out, rounded to bf16."""

    mixed = torch.einsum("tij,tid->tjd", comb.to(F32), streams.to(F32))
    return bf16(mixed + post.to(F32).unsqueeze(-1) * out.to(F32).unsqueeze(1))
