"""Lane-level numpy transliterations of one warp's k loop: upstream's ``warp_tiles`` (experts_grouped.cuh) and our
``ld_tiles`` (engine/kernels/exl3/x3ld.cu), each yielding the operands of every decode + mma in issue order.

The two kernels share everything after the operands (``decode_tile`` / ``mma16816`` from the same header, the same
accumulators, the same warp reduction), so equal operand streams -- per k step, per column tile, per lane: the
lane's trellis words and its four A-fragment words -- mean equal Z. ``bugs`` injects the faults the tests must catch.
"""

from __future__ import annotations

import numpy as np

SENTINEL = np.uint32(0xDEADBEEF)


def fmt(k2: int) -> tuple[int, int]:
    """(TW, LW): words a tile, words a lane a tile (upstream's Fmt<K2>)."""

    tw = 4 * k2
    return tw, (tw + 31) // 32


def upstream_stream(T, NTILES, kt0, nkt, nt0, X, K, r0, r1, k2, NT):
    """warp_tiles<CB, K2, NT, PF>: for it, i -> (words [32, LW], a [32, 4]) (PF only moves loads in time)."""

    tw, lw = fmt(k2)
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    out = []
    for it in range(nkt):
        k = (kt0 + it) * 16
        a = _a_frag(X, K, r0, r1, g, t, k)
        for i in range(NT):
            base = ((kt0 + it) * NTILES + nt0 + i) * tw
            w = np.zeros((32, lw), dtype=np.uint32)
            for l in range(lw):
                ok = (l * 32 + lane < tw) if tw % 32 else np.ones(32, bool)
                w[:, l] = np.where(ok, T[np.minimum(base + l * 32 + lane, T.size - 1)], 0)
            out.append((it, i, w, a))
    return out


def _a_frag(X, K, r0, r1, g, t, k):
    """load_pair's four values for every lane: rows r0 / r1 (per lane, -1 = dead) at k + 2t and k + 8 + 2t."""

    rr0, rr1 = r0[g], r1[g]

    def pair(rows, kk):
        ok = rows >= 0
        idx = np.where(ok, rows, 0) * K + kk + 2 * t
        lo, hi = X[idx], X[idx + 1]
        return np.where(ok, (lo.astype(np.uint32) | (hi.astype(np.uint32) << 16)), 0).astype(np.uint32)

    return np.stack([pair(rr0, k), pair(rr1, k), pair(rr0, k + 8), pair(rr1, k + 8)], 1)


def ld_stream(T, NTILES, kt0, nkt, nt0, X, K, r0, r1, k2, NT, PD, bugs=frozenset(), loads=None):
    """ld_tiles<CB, K2, NT, PD, 0>, statement by statement for all 32 lanes at once. ``loads`` collects every
    (word offset, step) a v4 load reads, for the alignment / bounds checks."""

    tw, lw = fmt(k2)
    words = NT * tw
    nv = (words + 127) // 128
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    kstride = NTILES * tw
    tp = (kt0 * NTILES + nt0) * tw
    stage = np.full(max(words, 16 * NT * 16), SENTINEL, dtype=np.uint32)      # the warp's slice of red, junk
    ring = np.full((PD, nv, 32, 4), SENTINEL, dtype=np.uint32)
    a_ring = np.zeros((PD, 32, 4), dtype=np.uint32)

    def issue_w(d, s):
        src = tp + s * kstride
        for v in range(nv):
            off = 128 * v + 4 * lane
            ok = off < words
            if "no_guard" in bugs:
                ok = np.ones(32, bool)
            for q in range(4):
                idx = src + off + q
                ring[d, v, :, q] = np.where(ok, T[np.minimum(idx, T.size - 1)], ring[d, v, :, q])
            if loads is not None:
                for ln in range(32):
                    if ok[ln]:
                        loads.append((int(src + off[ln]), s))

    def issue_a(d, s):
        k = (kt0 + s) * 16
        if "a_wrong_step" in bugs:
            k = (kt0 + max(s - 1, 0)) * 16
        a_ring[d] = _a_frag(X, K, r0, r1, g, t, k)

    for d in range(PD):
        issue_w(d, d)
    for d in range(PD):
        issue_a(d, d)
    out = []
    for kb in range(0, nkt, PD):
        for d in range(PD):
            s = kb + d
            more = s + PD < nkt
            if "refill_before_store" in bugs and more:
                issue_w(d, s + PD)
            for v in range(nv):
                off = 128 * v + 4 * lane
                ok = off < words
                for q in range(4):
                    stage[np.where(ok, off + q, stage.size - 1)] = np.where(ok, ring[d, v, :, q],
                                                                            stage[stage.size - 1])
            if more and "refill_before_store" not in bugs:
                issue_w(d, s + PD)
            w = np.zeros((NT, 32, lw), dtype=np.uint32)
            for i in range(NT):
                for l in range(lw):
                    ok = (l * 32 + lane < tw) if tw % 32 else np.ones(32, bool)
                    idx = i * tw + l * 32 + lane
                    if "lane_major" in bugs:
                        idx = lane * NT + i + l * 32 * NT
                    w[i, :, l] = np.where(ok, stage[np.minimum(idx, stage.size - 1)], 0)
            a = a_ring[d].copy()
            if more:
                issue_a(d, s + PD)
            order = range(NT - 1, -1, -1) if "tiles_reversed" in bugs else range(NT)
            for i in order:
                out.append((s, i, w[i], a))
    return out


def same(a, b) -> bool:
    if len(a) != len(b):
        return False
    for (s1, i1, w1, a1), (s2, i2, w2, a2) in zip(a, b):
        if s1 != s2 or i1 != i2 or not np.array_equal(w1, w2) or not np.array_equal(a1, a2):
            return False
    return True
