"""Engram's device side on sm_121: the row dequant and the gated fusion into the 4 residual streams.

Math (vLLM ``deepseek_v4_1/common/engram.py``, Apache-2.0, math only; ``engine/reference/engram.py: Engram``):

    e     = bf16(e4m3 x 2^(ue8m0 - 127) every 32 values)               [R, 24 x 256], the rows of a token
    kv    = bf16(wkv(e))                                               [R, 5 D] (EXL3 linear, outside this module)
    dot_h = sum_d (x_h q_h k_h)[d] key_h[d] * rsqrt(ms(x_h) + eps) * rsqrt(ms(key_h) + eps) / sqrt(D)
    gate  = sigmoid(sign(dot_h) sqrt(max(|dot_h|, 1e-6))) * keep       (keep 0 on image tokens)
    x_h  <- bf16(x_h + gate_h * value)                                 value = kv[:, 4 D:]

Arithmetic (``enable_fp_fusion=False``; ``ref.py`` emulates it):

- ``_dequant``: the e4m3 byte decoded from its bits (normal: sign | (e + 120) << 23 | m << 20; subnormal m x 2^-9),
  times the scale built from its byte (``byte << 23``: 2^(byte - 127), 0 for byte 0, as the reference), one IEEE
  product, one round to bf16: bit for bit the reference's ``bf16(fp8_e4m3_dequant(...))``.
- ``_fuse``: one program a row. ``qk = bf16(q) * bf16(k)`` (exact in fp32, precomputed at load); per stream the
  three sums (x qk key with x qk exact and one rounding of the product, x^2 and key^2 exact) are FMA chains against
  ones in column order (``tl.dot`` ieee: no reduction tree, the same for any row); 1 / sqrt with ``sqrt_rn`` /
  ``div_rn``; the gate's sigmoid uses the hardware exp (tolerance, not bits); then the update elementwise.
"""

from __future__ import annotations

import triton
import triton.language as tl

HD = 256           # values a row
SCALES = 8         # UE8M0 bytes a row (one a 32 values)
RECORD = 264
BK = 64            # columns a step of the fusion (no effect on bits)
FUSE_WARPS = 4


@triton.jit
def e4m3_bits(b):
    """uint8-valued int32 b -> the e4m3 value as fp32, exactly (NaN for 0x7F / 0xFF)."""

    s = (b >> 7) & 1
    e = (b >> 3) & 15
    m = b & 7
    normal = (s << 31) | ((e + 120) << 23) | (m << 20)
    sub = m.to(tl.float32) * 0.001953125
    v = tl.where(e > 0, normal.to(tl.float32, bitcast=True), tl.where(s > 0, -sub, sub))
    nan = (e == 15) & (m == 7)
    return tl.where(nan, ((s << 31) | 0x7FC00000).to(tl.float32, bitcast=True), v)


@triton.jit
def _dequant(RAW, raw_stride, OUT, o_stride, col0, H: tl.constexpr, HP: tl.constexpr):
    """Program r: RAW[r] = H records of 264 bytes (256 e4m3, 8 scale bytes) -> OUT[r, (col0 + h) 256 + i] bf16."""

    r = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, HP)
    i = tl.arange(0, 256)
    ok = h < H
    b = tl.load(RAW + r * raw_stride + h[:, None] * 264 + i[None, :], mask=ok[:, None], other=0).to(tl.int32)
    sc = tl.load(RAW + r * raw_stride + h[:, None] * 264 + 256 + i[None, :] // 32, mask=ok[:, None],
                 other=0).to(tl.int32)
    v = e4m3_bits(b) * (sc << 23).to(tl.float32, bitcast=True)
    tl.store(OUT + r * o_stride + (col0 + h[:, None]) * 256 + i[None, :], v.to(tl.bfloat16), mask=ok[:, None])


@triton.jit
def _fuse(X, x_stride, KV, kv_stride, QK, KEEP, GATE, eps, clamp, sqrt_d, D: tl.constexpr, BK: tl.constexpr,
          HAS_KEEP: tl.constexpr, HAS_GATE: tl.constexpr):
    """Program r: the 4 gates of row r from X [R, 4 D] (bf16), KV [R, 5 D] (bf16: keys of streams 0-3, the value)
    and QK [4, D] fp32; X updated in place. KEEP [R] fp32 (1 / 0), GATE [R, 4] fp32 (optional output)."""

    r = tl.program_id(0).to(tl.int64)
    hh = tl.arange(0, 16)
    hm = hh < 4
    kk = tl.arange(0, BK)
    ones = tl.full((BK, 16), 1.0, tl.float32)
    sp = tl.zeros((16, 16), tl.float32)
    sx = tl.zeros((16, 16), tl.float32)
    sk = tl.zeros((16, 16), tl.float32)
    for t in range(D // BK):
        d = t * BK + kk
        x = tl.load(X + r * x_stride + hh[:, None] * D + d[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
        k = tl.load(KV + r * kv_stride + hh[:, None] * D + d[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
        q = tl.load(QK + hh[:, None] * D + d[None, :], mask=hm[:, None], other=0.0)
        p = (x * q) * k
        sp = tl.dot(p, ones, sp, input_precision="ieee")
        sx = tl.dot(x * x, ones, sx, input_precision="ieee")
        sk = tl.dot(k * k, ones, sk, input_precision="ieee")
    c0 = tl.arange(0, 16)[None, :] == 0
    dp = tl.sum(tl.where(c0, sp, 0.0), 1)
    rx = tl.div_rn(1.0, tl.sqrt_rn(tl.div_rn(tl.sum(tl.where(c0, sx, 0.0), 1), 1.0 * D) + eps))
    rk = tl.div_rn(1.0, tl.sqrt_rn(tl.div_rn(tl.sum(tl.where(c0, sk, 0.0), 1), 1.0 * D) + eps))
    dot = tl.div_rn((dp * rx) * rk, sqrt_d)
    g = tl.sqrt_rn(tl.maximum(tl.abs(dot), clamp))
    gate = 1.0 / (1.0 + tl.exp(-tl.where(dot < 0, -g, g)))
    if HAS_KEEP:
        gate = gate * tl.load(KEEP + r)
    gate = tl.where(hm, gate, 0.0)
    if HAS_GATE:
        tl.store(GATE + r * 4 + hh, gate, mask=hm)
    for t in range(D // BK):
        d = t * BK + kk
        x = tl.load(X + r * x_stride + hh[:, None] * D + d[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
        v = tl.load(KV + r * kv_stride + 4 * D + d).to(tl.float32)
        tl.store(X + r * x_stride + hh[:, None] * D + d[None, :], (x + gate[:, None] * v[None, :]).to(tl.bfloat16),
                 mask=hm[:, None])
