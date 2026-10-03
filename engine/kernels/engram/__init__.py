"""Engram (layers 1 and 14) for DeepSeek-V4.1-Flash on sm_121: host hash + prefetch, row dequant, gated fusion.

Interface:

    hash.Tables.from_config(cfg, token_map) / .load(npz) / .save(npz)
        token map, multipliers, primes, offsets (the prepared folder stores them)
    Tables.rows(ids, dead=None, lookback=None) -> [T, 2, 24] int64     host hash, outside the graphs
    hash.next_lookback(lookback, tables, ids, dead) -> the slot's last 3 compressed ids after a commit
    hash.Prefetch(tables, reader, rank, world, max_rows)
        .issue(lookback, [pending, d1, ...]) at drafter end (prefill: when a chunk is cut); .land(ticket, li) waits
        for a layer's rows into pinned staging [2, max_rows, 12, 264]; .upload(li, device_buf, n) one H2D copy

    dequant(raw uint8 [R, H, 264], out bf16 [R, 24 x 256], col0=0)      a rank's H = 12 heads at column col0 (TP:
                                                                          gather the raw records first: 3 KB a row)
    fuse(x bf16 [R, 4 D] (updated in place), kv bf16 [R, 5 D], qk fp32 [4, D], keep fp32 [R] or None, gate=None)
        kv = the Engram wkv output (EXL3 linear, 6,144 -> 25,600, bf16 out); qk = qk_weights(q_weight, k_weight)

Order in a forward: ``mhc.post_only`` (the FFN boundary of layer L - 1), ``fuse``, then ``mhc.site`` with the carried
pre-mix (the attention site of layer L).
"""

from __future__ import annotations

import math

import torch
import triton

from . import kernels as K

EPS = 1e-20
CLAMP = 1e-6
LAUNCH = {"enable_fp_fusion": False}


def qk_weights(q_weight: torch.Tensor, k_weight: torch.Tensor) -> torch.Tensor:
    """bf16(q) * bf16(k) in fp32 [4, D] (exact: two 8-bit mantissas), computed once at load."""

    return (q_weight.to(torch.bfloat16).float() * k_weight.to(torch.bfloat16).float()).contiguous()


def dequant(raw: torch.Tensor, out: torch.Tensor, col0: int = 0) -> torch.Tensor:
    R, H, rb = raw.shape
    if raw.dtype != torch.uint8 or rb != K.RECORD or raw.stride(2) != 1 or raw.stride(1) != K.RECORD:
        raise ValueError("engram.dequant: raw uint8 [R, H, 264] with contiguous records")
    if out.dtype != torch.bfloat16 or out.stride(1) != 1 or out.shape[1] < (col0 + H) * K.HD:
        raise ValueError("engram.dequant: out bf16 [R, >= (col0 + H) 256]")
    if R:
        K._dequant[(R,)](raw, raw.stride(0), out, out.stride(0), col0, H=H, HP=triton.next_power_of_2(H),
                         num_warps=4, **LAUNCH)
    return out


def fuse(x: torch.Tensor, kv: torch.Tensor, qk: torch.Tensor, keep: torch.Tensor | None = None,
         gate: torch.Tensor | None = None, *, eps: float = EPS, clamp: float = CLAMP) -> torch.Tensor:
    R = x.shape[0]
    d = qk.shape[1]
    if x.dtype != torch.bfloat16 or x.shape[1] != 4 * d or x.stride(1) != 1 or d % K.BK:
        raise ValueError("engram.fuse: x bf16 [R, 4 D] unit-stride, D a multiple of 64")
    if kv.dtype != torch.bfloat16 or kv.shape[1] < 5 * d or kv.stride(1) != 1 or qk.dtype != torch.float32:
        raise ValueError("engram.fuse: kv bf16 [R, 5 D], qk fp32 [4, D]")
    if keep is not None and keep.dtype != torch.float32:
        keep = keep.to(torch.float32)
    if R:
        K._fuse[(R,)](x, x.stride(0), kv, kv.stride(0), qk, keep if keep is not None else qk,
                      gate if gate is not None else qk, eps, clamp, float(math.sqrt(d)), D=d, BK=K.BK,
                      HAS_KEEP=keep is not None, HAS_GATE=gate is not None, num_warps=K.FUSE_WARPS, **LAUNCH)
    return x


__all__ = ["dequant", "fuse", "qk_weights"]
