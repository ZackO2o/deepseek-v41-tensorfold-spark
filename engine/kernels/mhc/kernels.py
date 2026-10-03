"""Single-Pass mHC for DeepSeek-V4.1-Flash on sm_121: one pass over the 4 residual streams a sublayer boundary.

Math (vLLM ``model_executor/kernels/mhc/torch.py: mhc_pre_delayed_torch / mhc_post_torch``, Apache-2.0, cited for the
math only; ``engine/reference/hc.py``): a boundary after sublayer X (its coefficients ``post_X``, ``comb_X``, ``pre_X``
were computed before X ran) does

    X'_j = bf16(post_X[j] * branch + (((comb_X[0, j] X_0 + comb_X[1, j] X_1) + comb_X[2, j] X_2) + comb_X[3, j] X_3))
    c    = bf16(((pre_X[0] X'_0 + pre_X[1] X'_1) + pre_X[2] X'_2) + pre_X[3] X'_3)      the NEXT sublayer's input
    mix  = (X'.flat . fn_{X+1}) * rsqrt(mean(X'.flat^2) + eps)                          the next site's 24 mixes

and the next site's input is ``bf16((c * rsqrt(mean(c^2) + eps)) * norm_w)``. The pre-mix used to collapse is the
*previous* site's (V4.1's delayed / Single-Pass form, TR 2.4.1), so the collapse needs nothing from this pass's dots:
the streams are read once and written once a boundary (GLM ran hc_post, then hc_pre's partials over the written
streams, then the finish: 0320 / 0520 / 0190). ``branch = bf16(sum of the ranks' fp32 partials, rank 0 first)``.

Arithmetic (everything here is fixed by the shapes, never by the row count; kernels launch with
``enable_fp_fusion=False`` so every written ``a * b + c`` is a rounded product and a rounded sum, as in the torch
emulation ``ref.py``):

- ``_site``: program (16-row tile, column block b of CB = D / NB columns of all four streams). Per stream j and block
  b, the 24 mixes are an FMA chain over the block's columns in ascending order from 0 (``tl.dot`` ieee on fp32: one
  ``fma.rn.f32`` an element a column), and the stream's square sum is the FMA chain of the exact squares (a bf16
  squared is exact in fp32) against ones; the collapsed row's square sum likewise. No reduction tree anywhere: a row's
  partials are the same in any tile, for any BM / BK.
- ``_finish``: one program a row; the 4 x NB partials summed in (stream, block) order, then pre / post / comb
  (sigmoid, softmax, 20 Sinkhorn iterations; ``tl.exp`` is the hardware approximation: these coefficients are within
  tolerance of torch, not bitwise), and the normed input with IEEE ``sqrt_rn`` / ``div_rn``.

No atomics; rows never mix. Batched == alone, verify == serial: bit for bit.
"""

from __future__ import annotations

import triton
import triton.language as tl

S = 4             # streams
NMIX = 24         # 2 S + S^2
BM = 16           # rows a program (tl.dot's minimum M; no effect on bits)
BK = 16           # columns a step (no effect on bits: the FMA chains run in column order whatever the step)
NB = 40           # column blocks of every stream (5,120 / 40 = 128 columns: 8 steps). Part of the arithmetic
WARPS = 8         # with BK 16: 152 registers, no spills (BK 32 or 4 warps spill)
FIN_BLOCK = 1024  # columns a step of the finish's norm (elementwise: no effect on bits)


@triton.jit
def _chain(v, w, acc):
    """acc[m, n] = fma(v[m, k], w[k, n], acc) for k ascending: the ieee fp32 dot's FMA chain."""

    return tl.dot(v, w, acc, input_precision="ieee")


@triton.jit
def _mix_step(v, FN, col, mm, ones, acc, sq, WIDE: tl.constexpr):
    """One stream tile v [BM, BK] (fp32 holding bf16 values): acc += v fn^T (24 mixes in 32 columns), sq += v^2 . 1."""

    w = tl.load(FN + mm[None, :] * WIDE + col[:, None], mask=mm[None, :] < 24, other=0.0)      # [BK, 32]
    acc = _chain(v, w, acc)
    sq = _chain(v * v, ones, sq)
    return acc, sq


@triton.jit
def _site(X, x_stride, XOUT, G, g_rank, POST, COMB, PRE, FN, PART, C, TAP, tap_stride, R,
          D: tl.constexpr, NB: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, WORLD: tl.constexpr,
          POST_ON: tl.constexpr, COLLAPSE: tl.constexpr, MIX: tl.constexpr, TAP_ON: tl.constexpr):
    """Program (row tile, column block b).

    POST_ON: the new streams from X (rows ``x_stride`` apart), the gathered partials G [WORLD, R, D] (``g_rank``
    apart) and POST [R, 4] / COMB [R, 16] (comb[i, j] at i * 4 + j), stored to XOUT [R, 4 D]; else the streams are X.
    COLLAPSE 1: c = stream 0 (model / DSpark entry); 2: c = sum_i PRE[i] X'_i; stored bf16 to C [R, D].
    MIX: the next site's partials PART [R, 4, NB, 32]: [0:24] mixes, [24] the stream's square sum, and at stream 0
    [25] the collapsed row's square sum. TAP_ON: TAP[r] = bf16((((X'_0 + X'_1) + X'_2) + X'_3) * 0.25) (DSpark taps).
    """

    pid = tl.program_id(0)
    b = tl.program_id(1)
    WIDE: tl.constexpr = 4 * D
    CB: tl.constexpr = D // NB
    rows = pid * BM + tl.arange(0, BM)
    ok = rows < R
    rr = tl.where(ok, rows, 0).to(tl.int64)
    kk = tl.arange(0, BK)
    mm = tl.arange(0, 32)
    ones = tl.full((BK, 16), 1.0, tl.float32)
    xin = rr[:, None] * x_stride
    xo = rr[:, None] * WIDE
    if POST_ON:
        cb = COMB + rr * 16
        c00 = tl.load(cb + 0, mask=ok, other=0.0)
        c01 = tl.load(cb + 1, mask=ok, other=0.0)
        c02 = tl.load(cb + 2, mask=ok, other=0.0)
        c03 = tl.load(cb + 3, mask=ok, other=0.0)
        c10 = tl.load(cb + 4, mask=ok, other=0.0)
        c11 = tl.load(cb + 5, mask=ok, other=0.0)
        c12 = tl.load(cb + 6, mask=ok, other=0.0)
        c13 = tl.load(cb + 7, mask=ok, other=0.0)
        c20 = tl.load(cb + 8, mask=ok, other=0.0)
        c21 = tl.load(cb + 9, mask=ok, other=0.0)
        c22 = tl.load(cb + 10, mask=ok, other=0.0)
        c23 = tl.load(cb + 11, mask=ok, other=0.0)
        c30 = tl.load(cb + 12, mask=ok, other=0.0)
        c31 = tl.load(cb + 13, mask=ok, other=0.0)
        c32 = tl.load(cb + 14, mask=ok, other=0.0)
        c33 = tl.load(cb + 15, mask=ok, other=0.0)
        p0 = tl.load(POST + rr * 4 + 0, mask=ok, other=0.0)
        p1 = tl.load(POST + rr * 4 + 1, mask=ok, other=0.0)
        p2 = tl.load(POST + rr * 4 + 2, mask=ok, other=0.0)
        p3 = tl.load(POST + rr * 4 + 3, mask=ok, other=0.0)
    if COLLAPSE == 2:
        q0 = tl.load(PRE + rr * 4 + 0, mask=ok, other=0.0)
        q1 = tl.load(PRE + rr * 4 + 1, mask=ok, other=0.0)
        q2 = tl.load(PRE + rr * 4 + 2, mask=ok, other=0.0)
        q3 = tl.load(PRE + rr * 4 + 3, mask=ok, other=0.0)
    a0 = tl.zeros((BM, 32), tl.float32)
    a1 = tl.zeros((BM, 32), tl.float32)
    a2 = tl.zeros((BM, 32), tl.float32)
    a3 = tl.zeros((BM, 32), tl.float32)
    s0 = tl.zeros((BM, 16), tl.float32)
    s1 = tl.zeros((BM, 16), tl.float32)
    s2 = tl.zeros((BM, 16), tl.float32)
    s3 = tl.zeros((BM, 16), tl.float32)
    sc = tl.zeros((BM, 16), tl.float32)
    for t in range(CB // BK):
        d = b * CB + t * BK + kk
        m2 = ok[:, None]
        x0 = tl.load(X + xin + d[None, :], mask=m2, other=0.0).to(tl.float32)
        x1 = tl.load(X + xin + D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        x2 = tl.load(X + xin + 2 * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        x3 = tl.load(X + xin + 3 * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        if POST_ON:
            g = tl.load(G + rr[:, None] * D + d[None, :], mask=m2, other=0.0)
            for w in tl.static_range(1, WORLD):
                g = g + tl.load(G + w * g_rank + rr[:, None] * D + d[None, :], mask=m2, other=0.0)
            br = g.to(tl.bfloat16).to(tl.float32)
            v0 = (p0[:, None] * br + (((c00[:, None] * x0 + c10[:, None] * x1) + c20[:, None] * x2)
                                      + c30[:, None] * x3)).to(tl.bfloat16)
            v1 = (p1[:, None] * br + (((c01[:, None] * x0 + c11[:, None] * x1) + c21[:, None] * x2)
                                      + c31[:, None] * x3)).to(tl.bfloat16)
            v2 = (p2[:, None] * br + (((c02[:, None] * x0 + c12[:, None] * x1) + c22[:, None] * x2)
                                      + c32[:, None] * x3)).to(tl.bfloat16)
            v3 = (p3[:, None] * br + (((c03[:, None] * x0 + c13[:, None] * x1) + c23[:, None] * x2)
                                      + c33[:, None] * x3)).to(tl.bfloat16)
            tl.store(XOUT + xo + d[None, :], v0, mask=m2)
            tl.store(XOUT + xo + D + d[None, :], v1, mask=m2)
            tl.store(XOUT + xo + 2 * D + d[None, :], v2, mask=m2)
            tl.store(XOUT + xo + 3 * D + d[None, :], v3, mask=m2)
            x0 = v0.to(tl.float32)
            x1 = v1.to(tl.float32)
            x2 = v2.to(tl.float32)
            x3 = v3.to(tl.float32)
        if COLLAPSE > 0:
            if COLLAPSE == 2:
                cv = (((q0[:, None] * x0 + q1[:, None] * x1) + q2[:, None] * x2) + q3[:, None] * x3).to(tl.bfloat16)
            else:
                cv = x0.to(tl.bfloat16)
            tl.store(C + rr[:, None] * D + d[None, :], cv, mask=m2)
            cf = cv.to(tl.float32)
            sc = _chain(cf * cf, ones, sc)
        if TAP_ON:
            tp = ((((x0 + x1) + x2) + x3) * 0.25).to(tl.bfloat16)
            tl.store(TAP + rr[:, None] * tap_stride + d[None, :], tp, mask=m2)
        if MIX:
            a0, s0 = _mix_step(x0, FN, d, mm, ones, a0, s0, WIDE)
            a1, s1 = _mix_step(x1, FN, D + d, mm, ones, a1, s1, WIDE)
            a2, s2 = _mix_step(x2, FN, 2 * D + d, mm, ones, a2, s2, WIDE)
            a3, s3 = _mix_step(x3, FN, 3 * D + d, mm, ones, a3, s3, WIDE)
    col0 = tl.arange(0, 16)[None, :] == 0
    base = PART + (rr[:, None] * 4 * NB + b) * 32 + mm[None, :]
    if MIX:
        first = tl.where(mm[None, :] == 24, tl.sum(tl.where(col0, s0, 0.0), 1)[:, None], a0)
        first = tl.where(mm[None, :] == 25, tl.sum(tl.where(col0, sc, 0.0), 1)[:, None], first)
        tl.store(base, first, mask=ok[:, None] & (mm[None, :] < 26))
        tl.store(base + NB * 32, tl.where(mm[None, :] == 24, tl.sum(tl.where(col0, s1, 0.0), 1)[:, None], a1),
                 mask=ok[:, None] & (mm[None, :] < 25))
        tl.store(base + 2 * NB * 32, tl.where(mm[None, :] == 24, tl.sum(tl.where(col0, s2, 0.0), 1)[:, None], a2),
                 mask=ok[:, None] & (mm[None, :] < 25))
        tl.store(base + 3 * NB * 32, tl.where(mm[None, :] == 24, tl.sum(tl.where(col0, s3, 0.0), 1)[:, None], a3),
                 mask=ok[:, None] & (mm[None, :] < 25))
    elif COLLAPSE > 0:
        tl.store(base + tl.zeros((BM, 32), tl.int32), tl.sum(tl.where(col0, sc, 0.0), 1)[:, None] +
                 tl.zeros((BM, 32), tl.float32), mask=ok[:, None] & (mm[None, :] == 25))


@triton.jit
def _finish(PART, BASE, SCALE, PRE, POST, COMB, C, NW, OUT, eps, hc_eps, post_alpha,
            D: tl.constexpr, NB: tl.constexpr, ITERS: tl.constexpr, COEF: tl.constexpr, BLOCK: tl.constexpr):
    """Row r: the partials summed in (stream, block) order; COEF: pre / post / comb of the next site (PRE, POST
    [R, 4], COMB [R, 16]); then OUT = bf16((c * (1 / sqrt(ssc / D + eps))) * NW) from the collapsed row C."""

    r = tl.program_id(0).to(tl.int64)
    m = tl.arange(0, 32)
    pr = PART + r * (4 * NB * 32)
    ssc = 0.0
    for bb in range(NB):
        ssc += tl.load(pr + bb * 32 + 25)
    if COEF:
        mix = tl.zeros((32,), tl.float32)
        ss = 0.0
        for j in range(4 * NB):
            mix += tl.load(pr + j * 32 + m)
            ss += tl.load(pr + j * 32 + 24)
        rinv = tl.div_rn(1.0, tl.sqrt_rn(tl.div_rn(ss, 4.0 * D) + eps))
        mix = tl.where(m < 24, mix, 0.0) * rinv
        s_pre = tl.load(SCALE + 0)
        s_post = tl.load(SCALE + 1)
        s_comb = tl.load(SCALE + 2)
        sv = tl.arange(0, 4)
        base = tl.load(BASE + m, mask=m < 24, other=0.0)
        pre_l = tl.sum(tl.where(m[None, :] == sv[:, None], (mix * s_pre + base)[None, :], 0.0), axis=1)
        post_l = tl.sum(tl.where(m[None, :] == (sv[:, None] + 4), (mix * s_post + base)[None, :], 0.0), axis=1)
        pre = 1.0 / (1.0 + tl.exp(-pre_l)) + hc_eps
        post = (1.0 / (1.0 + tl.exp(-post_l))) * post_alpha
        ii = tl.arange(0, 4)[:, None]
        jj = tl.arange(0, 4)[None, :]
        flat = 8 + ii * 4 + jj
        cl = tl.sum(tl.where(m[None, None, :] == flat[:, :, None], (mix * s_comb + base)[None, None, :], 0.0), axis=2)
        ce = tl.exp(cl - tl.max(cl, axis=1)[:, None])
        comb = ce / tl.sum(ce, axis=1)[:, None] + hc_eps
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
        for _ in range(ITERS - 1):
            comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)
            comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
        tl.store(PRE + r * 4 + sv, pre)
        tl.store(POST + r * 4 + sv, post)
        tl.store(COMB + r * 16 + ii * 4 + jj, comb)
    rn = tl.div_rn(1.0, tl.sqrt_rn(tl.div_rn(ssc, 1.0 * D) + eps))
    for t in range(D // BLOCK):
        d = t * BLOCK + tl.arange(0, BLOCK)
        c = tl.load(C + r * D + d).to(tl.float32)
        w = tl.load(NW + d).to(tl.float32)
        tl.store(OUT + r * D + d, ((c * rn) * w).to(tl.bfloat16))
