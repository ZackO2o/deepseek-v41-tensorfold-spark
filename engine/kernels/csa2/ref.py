"""Torch references for the CSA2 kernels (CPU, float64 where the kernels' arithmetic is not the point).

``quantize_rows`` / ``dequantize_rows`` are the row format's exact definition (byte for byte); the others are the
math, for tolerance checks against the kernels and as the oracle of engine/reference's layer tests.
"""

from __future__ import annotations

import math

import torch

from . import rows as RW


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [n, 512] (fp32 / bf16, RoPE applied) -> (values uint8 [n, 576], scales uint8 [n, 8])."""

    x = x.float()
    n = x.shape[0]
    nope = x[:, :RW.NOPE].reshape(n, RW.NT, RW.TILE)
    amax = nope.abs().amax(2).clamp_min(RW.AMAX_FLOOR)
    bits = amax.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 135 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    inv = ((127 - e) << 23).view(torch.float32)
    q = (nope * inv[:, :, None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(n, RW.NOPE)
    rope = x[:, RW.NOPE:].to(torch.bfloat16).view(torch.uint8).reshape(n, 2 * RW.ROPE)
    values = torch.cat([q, rope], 1).contiguous()
    scales = torch.zeros((n, RW.SB), dtype=torch.uint8)
    scales[:, :RW.NT] = (e + 127).to(torch.uint8)
    return values, scales


def dequantize_rows(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """(values, scales) -> bf16 [n, 512], exactly (e4m3 x 2^(s - 127) is a bf16)."""

    n = values.shape[0]
    q = values[:, :RW.NOPE].contiguous().view(torch.float8_e4m3fn).float().reshape(n, RW.NT, RW.TILE)
    s = ((scales[:, :RW.NT].to(torch.int32)) << 23).view(torch.float32)   # 2^(s - 127)
    nope = (q * s[:, :, None]).reshape(n, RW.NOPE)
    rope = values[:, RW.NOPE:].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope, rope], 1).to(torch.bfloat16)


def rope(x: torch.Tensor, cs: torch.Tensor, pos: torch.Tensor, dims: int = 512, inverse: bool = False) -> torch.Tensor:
    """GPT-J RoPE on the last 64 of ``dims`` (float64 math): x [n, dims], cs [max_pos, 64] (cos 32, sin 32)."""

    x = x.double().clone()
    c = cs[pos.long(), :32].double()
    s = cs[pos.long(), 32:].double()
    if inverse:
        s = -s
    lo = dims - 64
    e, o = x[:, lo::2].clone(), x[:, lo + 1::2].clone()
    x[:, lo::2] = e * c - o * s
    x[:, lo + 1::2] = o * c + e * s
    return x


def pool_norm(buf: torch.Tensor, w: torch.Tensor, P: int, n: int, ratio: int, eps: float = 1e-20) -> dict[int, torch.Tensor]:
    """{window row r: latent fp64 [512]} for rows where a group closes (``compress.pool_norm``'s buffer layout)."""

    out = {}
    for r in range(n):
        q = P + r
        if ratio == 2:
            if (q + 1) % 2:
                continue
            a, b = buf[r].double(), buf[r + 1].double()
            sc = torch.stack([a[512:], b[512:]])
            kv = torch.stack([a[:512], b[:512]])
            pooled = (kv * torch.softmax(sc, 0)).sum(0)
        else:
            pooled = buf[r].double()
        y = pooled / torch.sqrt((pooled * pooled).mean() + eps) * w.double()
        out[r] = y
    return out


def index_scores(qi: torch.Tensor, w: torch.Tensor, ik: torch.Tensor, positions: list[torch.Tensor]) -> list[torch.Tensor]:
    """Float64 s_j = sum_h (w_h / sqrt(32)) relu(qi_h . k_j / sqrt(128)) for each row's key positions."""

    out = []
    for r, js in enumerate(positions):
        k = ik[js.long()].double()
        dots = qi[r].double() @ k.T / math.sqrt(128)
        out.append((w[r].double()[:, None] / math.sqrt(32) * dots.clamp_min(0)).sum(0))
    return out


def select_stable(s: torch.Tensor, positions: torch.Tensor, count: int) -> list[int]:
    """The ``count`` best (score desc, then lower position) of one row's (s, positions), ascending."""

    order = sorted(range(len(s)), key=lambda i: (-float(s[i]), int(positions[i])))
    return sorted(int(positions[i]) for i in order[:count])


def attention(q: torch.Tensor, comp: torch.Tensor | None, tokens: list[list[int]], swa: dict[int, torch.Tensor],
              lo: list[int], hi: list[int], pos: list[int], sink: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
    """Float64 reference: q [R, H, 512] (bf16 values), comp bf16 rows [n, 512], swa {position: bf16 row} ->
    [R, H, 512] float64 (inverse RoPE applied)."""

    R, H, _ = q.shape
    out = torch.empty((R, H, 512), dtype=torch.float64)
    for r in range(R):
        keys = [comp[t].double() for t in tokens[r]] if comp is not None else []
        a = max(lo[r], hi[r] - 127, 0)
        keys += [swa[p].double() for p in range(a, hi[r] + 1)]
        k = torch.stack(keys)                                     # [n, 512] (K = V)
        s = q[r].double() @ k.T / math.sqrt(512)                  # [H, n]
        m = torch.maximum(s.max(1).values, sink.double())
        p = torch.exp(s - m[:, None])
        den = p.sum(1) + torch.exp(sink.double() - m)
        o = (p @ k) / den[:, None]
        out[r] = rope(o, cs, torch.full((H,), pos[r]), inverse=True)
    return out
