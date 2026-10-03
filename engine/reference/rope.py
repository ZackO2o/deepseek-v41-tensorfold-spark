"""DeepSeek-V4.1 RoPE: GPT-J interleaved pairs on the LAST ``rope_dim`` dims, two flavours per layer.

From vLLM ``deepseek_v4_1/common/rope.py`` + ``rotary_embedding/deepseek_scaling_rope.py``:

- compressed layers (ratio > 0): theta = ``compress_rope_theta`` (160,000) with DeepSeek YaRN (factor 16 over an
  original 65,536 positions, beta_fast 32, beta_slow 1, mscale disabled -> 1);
- sliding-window-only layers (ratio 0: layers 0-1 and the DSpark blocks): theta = ``rope_theta`` (10,000), plain.

The cos/sin cache is fp32 (``t * inv_freq`` in fp32, as vLLM builds it); rotations run in fp32 on bf16 inputs.
``inverse`` negates sin (the attention output's de-rotation).
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from .config import Config

F32 = torch.float32


def _correction_dim(num_rotations: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_inv_freq(rope_dim: int, base: float, factor: float, original_max: int, beta_fast: float,
                  beta_slow: float) -> torch.Tensor:
    pos_freqs = base ** (torch.arange(0, rope_dim, 2, dtype=F32) / rope_dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)
    low = math.floor(_correction_dim(beta_fast, rope_dim, base, original_max))
    high = math.ceil(_correction_dim(beta_slow, rope_dim, base, original_max))
    low, high = max(low, 0), min(high, rope_dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(rope_dim // 2, dtype=F32) - low) / (high - low)).clamp(0, 1)
    mask = 1 - ramp                                    # extrapolation_factor 1
    return interpolation * (1 - mask) + extrapolation * mask


class Rope:
    """cos/sin for one layer flavour, computed per position on demand (no 1M-row cache on the CPU)."""

    def __init__(self, inv_freq: torch.Tensor, rope_dim: int):
        self.inv_freq = inv_freq.to(F32)
        self.rope_dim = rope_dim

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = positions.to(F32)
        freqs = t[:, None] * self.inv_freq.to(t.device)[None, :]       # fp32, like the einsum in vLLM
        return torch.cos(freqs), torch.sin(freqs)          # [T, rope_dim / 2]

    def apply(self, x: torch.Tensor, positions: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Rotate the last ``rope_dim`` dims of ``x`` [T, ..., D] (GPT-J pairs), fp32 out."""

        x = x.to(F32)
        cos, sin = self.cos_sin(positions.reshape(-1))
        if inverse:
            sin = -sin
        extra = x.dim() - 2
        shape = (cos.shape[0],) + (1,) * extra + (cos.shape[1],)
        cos, sin = cos.reshape(shape), sin.reshape(shape)
        rot = x[..., -self.rope_dim:]
        even, odd = rot[..., 0::2], rot[..., 1::2]
        out = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
        return torch.cat((x[..., :-self.rope_dim], out), dim=-1)


@lru_cache(maxsize=8)
def _rope(rope_dim: int, base: float, factor: float, original_max: int, beta_fast: float, beta_slow: float) -> Rope:
    return Rope(yarn_inv_freq(rope_dim, base, factor, original_max, beta_fast, beta_slow), rope_dim)


def rope_for(cfg: Config, compress_ratio: int) -> Rope:
    r = cfg.rope
    if compress_ratio > 0 and r.yarn:
        return _rope(cfg.qk_rope_head_dim, r.compress_rope_theta, r.factor, r.original_max_position_embeddings,
                     r.beta_fast, r.beta_slow)
    base = r.compress_rope_theta if compress_ratio > 0 else r.rope_theta
    return _rope(cfg.qk_rope_head_dim, base, 1.0, r.max_position_embeddings, r.beta_fast, r.beta_slow)
