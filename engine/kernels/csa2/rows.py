"""The FP8 KV row of DeepSeek-V4.1-Flash's attention (compressed main KV and the 128-token SWA rings).

One token's (or compressed group's) 512-wide K = V row, RoPE already applied to its last 64 dims:

    values [576] uint8: bytes 0..447 the 448 NoPE dims as e4m3, bytes 448..575 the 64 RoPE dims as bf16
    scales [8]   uint8: one UE8M0 exponent (2^(s - 127)) a 64-dim NoPE tile, byte 7 zero

584 bytes a row: the record the kit's vLLM writes on SM12x (``fp8_ds_mla``, the "V4" layout), so M1's agreement
with the kit is not diluted by a different KV rounding; values and scales live in two tensors (both paged by one
page table) so value rows stay 16-byte aligned. Quantization of a NoPE tile, all in exact arithmetic:

    amax = max(max |x|, 1e-4)
    e    = the smallest integer with amax <= 448 * 2^e     (= ceil(log2(amax / 448)), from the float's bits)
    q    = e4m3(x * 2^-e), rounded to nearest even        (|x * 2^-e| <= 448: never saturates)

A stored value dequantizes to e4m3 x 2^e exactly, and that is exactly a bf16 (3 mantissa bits, e >= -22), so the
attention kernels' bf16 tiles hold the dequantized rows bit for bit and every bf16-cache guarantee carries over
(the GLM 0220 argument). A row is quantized by itself: no row's bytes depend on another row or on the window.
"""

from __future__ import annotations

import triton
import triton.language as tl

D = 512
NOPE = 448
ROPE = 64
TILE = 64
NT = NOPE // TILE          # 7 scale bytes
VB = NOPE + 2 * ROPE       # 576 value bytes
SB = 8                     # scale bytes
ROW_BYTES = VB + SB        # 584
AMAX_FLOOR = 1e-4
E4M3_MAX = 448.0


@triton.jit
def e4m3_rne(y):
    """fp32 values with |y| <= 448 rounded to the e4m3 grid, nearest even, in fp32 (normal range: 3 mantissa bits;
    below 2^-6 the fixed 2^-9 step via + / - 1.5 x 2^14), then converted: the conversion only relabels exact values
    whatever rounding the compiler's cvt would do (GLM patch 0220's rule)."""

    yb = y.to(tl.int32, bitcast=True)
    near = ((yb + 0x7FFFF + ((yb >> 20) & 1)) & -1048576).to(tl.float32, bitcast=True)
    a = tl.abs(y)
    small = (a + 24576.0) - 24576.0
    small = (small.to(tl.int32, bitcast=True) | (yb & -2147483648)).to(tl.float32, bitcast=True)
    return tl.where(a < 0.015625, small, near).to(tl.float8e4nv)


@triton.jit
def tile_exponent(amax):
    """The smallest e with amax <= 448 * 2^e (448 = 1.75 x 2^8), from amax's bits (amax >= 1e-4, normal)."""

    bits = amax.to(tl.int32, bitcast=True)
    return ((bits >> 23) & 0xFF) - 135 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)


@triton.jit
def pow2(e):
    """2^e as fp32 for -126 <= e <= 127, exactly."""

    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def store_row(V, S, row, x, d):
    """Row ``row`` (int64) of a (values, scales) cache from x [512] fp32 (RoPE applied): 7 NoPE tiles quantized on
    their own, the RoPE dims rounded to bf16. ``d`` = tl.arange(0, 512)."""

    xn = tl.reshape(tl.where(d < 448, x, 0.0), (8, 64))
    amax = tl.maximum(tl.max(tl.abs(xn), 1), 1e-4)                   # [8]; tile 7 (the RoPE dims) unused
    e = tile_exponent(amax)
    q = e4m3_rne(xn * tl.reshape(pow2(-e), (8, 1)))
    q = tl.reshape(q.to(tl.uint8, bitcast=True), (512,))
    tl.store(V + row * 576 + d, q, mask=d < 448)
    rope = (V + row * 576 + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope + tl.maximum(d - 448, 0), x.to(tl.bfloat16), mask=d >= 448)
    j = tl.arange(0, 8)
    tl.store(S + row * 8 + j, tl.where(j < 7, e + 127, 0).to(tl.uint8))


@triton.jit
def load_rows(V, S, rows, ok, d):
    """Rows ``rows`` [n] (int64, masked by ``ok``) dequantized to bf16 [n, 512]: e4m3 x 2^(s - 127) for the NoPE
    dims, the stored bf16 for the RoPE dims; masked rows read 0."""

    nope = d[None, :] < 448
    v = tl.load(V + rows[:, None] * 576 + d[None, :], mask=ok[:, None] & nope, other=0)
    s = tl.load(S + rows[:, None] * 8 + d[None, :] // 64, mask=ok[:, None] & nope, other=127).to(tl.int32)
    xn = v.to(tl.float8e4nv, bitcast=True).to(tl.float32) * pow2(s - 127)
    rope = (V + rows[:, None] * 576 + 448).to(tl.pointer_type(tl.bfloat16))
    xr = tl.load(rope + tl.maximum(d[None, :] - 448, 0), mask=ok[:, None] & ~nope, other=0.0).to(tl.float32)
    return tl.where(nope, xn, xr).to(tl.bfloat16)


@triton.jit
def prow(rows, PT, PSH: tl.constexpr, ok):
    """Logical rows -> physical rows through a slot's page table (GLM patch 0290's ``_prow``; PSH = log2 rows a
    page, 0 = contiguous: the branch is not compiled). ``ok`` None: a scalar row, always mapped."""

    if PSH > 0:
        if ok is None:
            pg = tl.load(PT + (rows >> PSH)).to(tl.int64)
        else:
            pg = tl.load(PT + (rows >> PSH), mask=ok, other=0).to(tl.int64)
        rows = (pg << PSH) + (rows & ((1 << PSH) - 1))
    return rows


@triton.jit
def rope_pairs(x, c, s, INVERSE: tl.constexpr):
    """GPT-J RoPE on the last 64 dims of x [512] fp32 (pairs 224..255 of the interleaved (even, odd) pairs); c / s
    [256] fp32 hold (cos, sin) for those pairs and (1, 0) elsewhere. INVERSE rotates by -theta."""

    even, odd = tl.split(tl.reshape(x, (256, 2)))
    if INVERSE:
        ne = even * c + odd * s
        no = odd * c - even * s
    else:
        ne = even * c - odd * s
        no = odd * c + even * s
    return tl.reshape(tl.join(ne, no), (512,))


@triton.jit
def cos_sin(CS, pos, cs_stride, PAIRS: tl.constexpr, ROPE_PAIRS: tl.constexpr):
    """(cos, sin) [PAIRS] fp32 at position ``pos`` from a [max_pos, 2 * ROPE_PAIRS] table (cos half, then sin half;
    the layout vLLM's ``cos_sin_cache`` uses): the last ROPE_PAIRS pairs rotate, the others get (1, 0)."""

    p = tl.arange(0, PAIRS) - (PAIRS - ROPE_PAIRS)
    rot = p >= 0
    q = tl.maximum(p, 0)
    base = CS + pos.to(tl.int64) * cs_stride
    c = tl.load(base + q, mask=rot, other=1.0).to(tl.float32)
    s = tl.load(base + ROPE_PAIRS + q, mask=rot, other=0.0).to(tl.float32)
    return c, s
