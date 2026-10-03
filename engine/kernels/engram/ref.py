"""Torch references of the Engram kernels (CPU): ``dequant`` is the reference's formula (bit for bit by
construction); ``fuse_stats`` / ``fuse_apply`` emulate ``_fuse``'s arithmetic (FMA chains through mhc.ref.chain) so the
row sums and the update compare bit for bit; the gate itself is compared within tolerance (hardware exp)."""

from __future__ import annotations

import math

import numpy as np
import torch

from ..mhc.ref import chain

F32 = torch.float32


def dequant(raw: torch.Tensor) -> torch.Tensor:
    """raw uint8 [R, H, 264] -> bf16 [R, H x 256] (engine/reference/ops.py: fp8_e4m3_dequant, then bf16)."""

    from engine.reference.ops import fp8_e4m3_dequant

    R, H, _ = raw.shape
    v = raw[..., :256].contiguous()
    s = raw[..., 256:].contiguous()
    return fp8_e4m3_dequant(v, s, 32).to(torch.bfloat16).reshape(R, H * 256)


def fuse_sums(x: torch.Tensor, kv: torch.Tensor, qk: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row and stream: (sum (x qk) key, sum x^2, sum key^2) as the kernel's FMA chains: [R, 4] each."""

    R = x.shape[0]
    d = qk.shape[1]
    sp = np.zeros((R, 4), np.float32)
    sx = np.zeros((R, 4), np.float32)
    sk = np.zeros((R, 4), np.float32)
    ones = np.ones((d, 1), np.float32)
    for r in range(R):
        xs = x[r].float().view(4, d)
        ks = kv[r, :4 * d].float().view(4, d)
        p = ((xs * qk) * ks).numpy()
        sp[r] = chain(p, ones)[:, 0]
        sx[r] = chain((xs * xs).numpy(), ones)[:, 0]
        sk[r] = chain((ks * ks).numpy(), ones)[:, 0]
    return sp, sx, sk


def gates(x: torch.Tensor, kv: torch.Tensor, qk: torch.Tensor, keep=None, eps: float = 1e-20,
          clamp: float = 1e-6) -> torch.Tensor:
    d = qk.shape[1]
    sp, sx, sk = (torch.from_numpy(a) for a in fuse_sums(x, kv, qk))
    rx = 1.0 / torch.sqrt(sx / float(d) + eps)
    rk = 1.0 / torch.sqrt(sk / float(d) + eps)
    dot = ((sp * rx) * rk) / torch.tensor(math.sqrt(d), dtype=F32)
    g = torch.sqrt(dot.abs().clamp(min=clamp))
    gate = torch.sigmoid(torch.where(dot < 0, -g, g))
    if keep is not None:
        gate = gate * keep.float()[:, None]
    return gate


def apply(x: torch.Tensor, kv: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """bf16(x_h + gate_h * value), the kernel's update given its gates (bit for bit)."""

    R = x.shape[0]
    d = kv.shape[1] // 5
    xs = x.float().view(R, 4, d)
    v = kv[:, 4 * d:5 * d].float()
    return (xs + gate[:, :, None] * v[:, None, :]).to(torch.bfloat16).reshape(R, 4 * d)
