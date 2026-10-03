"""DSpark's sequential head on sm_121: the Markov chain over the draft block with keyed draft noise, and the
confidence head, in one program a slot (inside the drafting graph).

Math (vLLM ``v1/worker/gpu/spec_decode/dspark/speculator.py: _sample_sequential_topk`` and
``model_executor/models/qwen3_dspark.py: DSparkMarkovHead / DSparkConfidenceHead``, Apache-2.0, math only;
``engine/reference/model.py: DSpark.draft``): for draft positions i = 0 .. N - 1 (row i predicts the token after
position P + i; prev_0 = the anchor):

    me_i      = markov_w1[prev_i]                                       (bf16 [256])
    z_i[c]    = base_i[c] + markov_w2[c] . me_i                         over the row's candidates c only
    token_i   = keyed choice over z_i (below);   prev_{i+1} = token_i
    conf_i    = sigmoid(w_conf . [h_i ; me_i])                          (h_i: the block's pre-norm head hidden)

Candidates: the base logits' best K a rank (``candidates``), gathered (WORLD x K entries): the bias on candidates only
(vLLM's ``dspark_draft_topk``; ARCH-LEVERAGE 5.2) instead of 5 sequential 66 MB GEMVs.

Keyed draft noise (ENGINE-PLAN 7, GLM DFlash2's rule): token_i follows ``exact_sampling.choose`` on the candidates
with the request's seed at the absolute position of the drafted token (POS0 + i): order by (value desc, id asc),
top_k, temperature, top_p, min_p, then argmax of z / T - log(-log(u(seed, position, id))). When the draft's
distribution equals the target's, the draft IS the target's keyed sample, so acceptance at T > 0 tracks greedy's.
Greedy (T = 0): the first of the order. The float64 log / exp here are libdevice's, not glibc's (the host's), and the
candidate set is the drafter's: drafts only change speed, never a reply (verification is exact, ``verify.py``).
"""

from __future__ import annotations

import triton
import triton.language as tl

RANK = 256
KP = 128          # candidate slots a position (WORLD x K <= 128)
CH = 16           # candidates a compare chunk
RB = 32           # Markov rank columns a step of the bias
WARPS = 4         # 8 warps cap ptxas at 128 registers and spill with the confidence head


@triton.jit
def _mix(x):
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    return x ^ (x >> 31)


@triton.jit
def keyed_uniform(seed, pos, ids):
    """``exact_sampling.uniform``: splitmix64 of (seed, position, id) -> (0, 1) float64."""

    x = _mix(seed.to(tl.int64).to(tl.uint64, bitcast=True) + 0x9E3779B97F4A7C15)
    x = _mix(x ^ (pos.to(tl.int64).to(tl.uint64, bitcast=True) * 0xD1B54A32D192ED03))
    x = _mix(x ^ ids.to(tl.int64).to(tl.uint64, bitcast=True))
    return (x >> 11).to(tl.float64) * 1.1102230246251565e-16 + 5.551115123125783e-17


@triton.jit
def _chain(CAND, CVAL, c_stride, n_cand, W1, W2, HID, CW, ANCHOR, IPAR, FPAR, DRAFT, CONF, SCR, D: tl.constexpr,
           N: tl.constexpr, KP: tl.constexpr, CH: tl.constexpr, R: tl.constexpr, RB: tl.constexpr, HB: tl.constexpr,
           HAS_CONF: tl.constexpr):
    """Program s (a slot). CAND int32 / CVAL fp32 [S, N, c_stride] the candidates (n_cand used, id -1 = none; CVAL is
    overwritten with the biased draft logits z), IPAR int64 [S, 4] (seed, POS0, top_k (0 = all), sampled 0 / 1),
    FPAR fp32 [S, 3] (temperature, top_p, min_p), HID bf16 [S, N, D], CW fp32 [D + R], SCR fp64 [S, 2, KP] (ranks
    and weights, the program's own) -> DRAFT int32 [S, N], CONF fp32 [S, N]."""

    s = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, KP)
    ok = j < n_cand
    col0 = tl.arange(0, 16)[None, :] == 0
    seed = tl.load(IPAR + s * 4 + 0)
    pos0 = tl.load(IPAR + s * 4 + 1)
    topk = tl.load(IPAR + s * 4 + 2)
    sampled = tl.load(IPAR + s * 4 + 3)
    temp = tl.maximum(tl.load(FPAR + s * 3 + 0).to(tl.float64), 1e-6)
    top_p = tl.load(FPAR + s * 3 + 1).to(tl.float64)
    min_p = tl.load(FPAR + s * 3 + 2).to(tl.float64)
    kk = tl.where(topk > 0, topk, n_cand)
    prev = tl.load(ANCHOR + s).to(tl.int64)
    scr = SCR + s * (2 * KP)
    for i in range(N):
        row = (s * N + i) * c_stride
        ids = tl.load(CAND + row + j, mask=ok, other=-1)
        live = ok & (ids >= 0)
        base = tl.load(CVAL + row + j, mask=live, other=float("-inf"))
        # the Markov bias of every candidate: markov_w2[ids] . markov_w1[prev] (bf16 products, fp32 sums)
        acc = tl.zeros((KP, 16), tl.float32)
        for r0 in range(0, R, RB):
            cols = r0 + tl.arange(0, RB)
            w2 = tl.load(W2 + tl.where(live, ids, 0).to(tl.int64)[:, None] * R + cols[None, :],
                         mask=live[:, None] & (cols[None, :] < R), other=0.0)                    # [KP, RB] bf16
            me = tl.load(W1 + prev * R + cols, mask=cols < R, other=0.0)
            acc = tl.dot(w2, tl.where(col0, me[:, None], 0.0).to(tl.bfloat16), acc)
        z = tl.where(live, base + tl.sum(tl.where(col0, acc, 0.0), 1), float("-inf"))
        tl.store(CVAL + row + j, z, mask=ok)
        tl.debug_barrier()
        # the order (z desc, id asc) as ranks
        rank = tl.zeros((KP,), tl.int32)
        for c0 in range(0, KP, CH):
            jc = c0 + tl.arange(0, CH)
            okc = jc < n_cand
            idc = tl.load(CAND + row + jc, mask=okc, other=-1)
            zc = tl.load(CVAL + row + jc, mask=okc, other=float("-inf"))
            before = (zc[None, :] > z[:, None]) | ((zc[None, :] == z[:, None]) & (idc[None, :] < ids[:, None]))
            rank += tl.sum((before & (idc >= 0)[None, :]).to(tl.int32), 1)
        rank = tl.where(live, rank, KP + 1)
        tok = tl.sum(tl.where(rank == 0, ids, 0))
        if sampled != 0:
            keep = rank < kk
            sc = z.to(tl.float64) / temp
            top = tl.max(tl.where(keep, sc, float("-inf")))
            mkeep = sc >= tl.where(min_p > 0.0, top + tl.log(tl.maximum(min_p, 1e-300)), float("-inf"))
            if (top_p > 0.0) & (top_p < 1.0):            # over the top_k set, before min_p (``choose``'s order)
                e = tl.where(keep, tl.exp(sc - top), 0.0)
                tot = tl.sum(e)
                tl.store(scr + j, rank.to(tl.float64))
                tl.store(scr + KP + j, e)
                tl.debug_barrier()
                cx = tl.zeros((KP,), tl.float64)          # the mass strictly before each candidate in the order
                for c0 in range(0, KP, CH):
                    jc = c0 + tl.arange(0, CH)
                    rc = tl.load(scr + jc)
                    ec = tl.load(scr + KP + jc)
                    cx += tl.sum(tl.where(rc[None, :] < rank.to(tl.float64)[:, None], ec[None, :], 0.0), 1)
                tl.debug_barrier()
                keep = keep & ((rank == 0) | (cx / tot < top_p))
            keep = keep & mkeep
            u = keyed_uniform(seed, pos0 + i, ids)
            score = tl.where(keep, sc - tl.log(-tl.log(u)), float("-inf"))
            best = tl.max(score)
            rmin = tl.min(tl.where(score == best, rank, KP + 1))
            tok = tl.sum(tl.where(rank == rmin, ids, 0))
        tl.store(DRAFT + s * N + i, tok)
        if HAS_CONF:
            cacc = 0.0
            for r0 in range(0, R, RB):
                cols = r0 + tl.arange(0, RB)
                cacc += tl.sum(tl.load(W1 + prev * R + cols, mask=cols < R, other=0.0).to(tl.float32) *
                               tl.load(CW + D + cols, mask=cols < R, other=0.0))
            for t in range(D // HB):
                d = t * HB + tl.arange(0, HB)
                cacc += tl.sum(tl.load(HID + (s * N + i) * D + d).to(tl.float32) * tl.load(CW + d))
            tl.store(CONF + s * N + i, 1.0 / (1.0 + tl.exp(-cacc)))
        prev = tok.to(tl.int64)
