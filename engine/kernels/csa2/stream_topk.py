"""Streaming score + top-k for CSA2's indexer at long context (prefill at 300K): the scores never reach memory.

``index.select`` materialises a row block's scores [rows, keys] (1.2 MB a row at 300K) and their int64 keys; here a
program (row r, key split c) scores its split tile by tile with ``index.score_tile`` (the same tile shape, warps
and arithmetic as ``index._scores``: the same bits), turns each score into ``index``'s unique int64 key (score
descending, then the lower position) and keeps a running candidate set in a bounded scratch (DeepSelect-style
threshold filtering):

- a key enters the split's buffer [CAP = 2 K] only if it beats the running threshold (the K-th best key of the
  buffer at the last compaction; KEY_NONE at first): appended in tile order (``tl.cumsum`` offsets);
- when a tile could overflow the buffer, the buffer is compacted: the K-th largest key is found by a 64-step radix
  search over the order-preserving unsigned image of the keys (exact: keys are unique), the K best are kept, and the
  threshold becomes the K-th;
- at the end the split's K best (or fewer) are left in buffer slots [0, K), KEY_NONE after.

Then ``merge`` takes the row's top K over its splits' candidates (``torch.topk`` on unique keys) and returns the
positions ascending, -1 padded (``index.top_positions``). Keys are unique, so the selected set is the top K of the
row's keys whatever the splits, tiles, insertion order or compaction points: the result is ``index.select``'s
whenever the row has at least K visible keys, and every visible key (never an invisible one) otherwise.

Modes: ``positions`` (Full layers 2 / 8 / 14 / 20, top-512 of the visible compressed positions), ``gather`` (Reindex
layers: the candidates' positions, as ``index.reindex``), ``blocks`` (layer 20's candidates: block maxima over 8
positions, the newest block pinned to +inf, top-2,048 blocks, as ``index.candidate_blocks``).

Scratch: rows x splits x CAP int64. ``plan(rows, keys, budget)`` picks the row block that fits ``budget`` (0.25 GiB
by default: ENGINE-PLAN 5's selection scratch): at 300K keys and SPLIT 16,384 that is 19 splits x 8 KB = 152 KB a row
(top-512) or 5 splits x 32 KB = 160 KB (top-2,048 of 37,500 blocks: 65,536 positions a split), i.e. >= 1,600
rows a launch.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from . import index
from .index import BP, HD, NH, KEY_NONE, SCORE_WARPS, score_tile

SPLIT = 16384          # keys a program (positions, gathered keys)
SPLIT_BLOCKS = 65536   # positions a program in blocks mode (8,192 blocks: top-2,048 of them)
BUDGET = 256 << 20     # bytes of selection scratch (ENGINE-PLAN 5)


@triton.jit
def _kth(BUF, n, K: tl.constexpr, CAP: tl.constexpr):
    """The K-th largest of BUF[0:n] (unique int64 keys, n > K) as an order-preserving uint64."""

    i = tl.arange(0, CAP)
    v = tl.load(BUF + i, mask=i < n, other=-9223372036854775808)
    u = v.to(tl.uint64, bitcast=True) ^ 0x8000000000000000
    ans = tl.zeros((), tl.uint64)
    top = tl.full((), 0x8000000000000000, tl.uint64)
    for t in range(64):
        cand = ans | (top >> t)
        cnt = tl.sum(((u >= cand) & (i < n)).to(tl.int32))
        ans = tl.where(cnt >= K, cand, ans)
    return ans


@triton.jit
def _compact(BUF, n, K: tl.constexpr, CAP: tl.constexpr):
    """Keep the K best of BUF[0:n] in BUF[0:K]; returns (K, the K-th key as int64)."""

    thr_u = _kth(BUF, n, K, CAP)
    i = tl.arange(0, CAP)
    v = tl.load(BUF + i, mask=i < n, other=-9223372036854775808)
    u = v.to(tl.uint64, bitcast=True) ^ 0x8000000000000000
    keep = (u >= thr_u) & (i < n)
    dst = tl.cumsum(keep.to(tl.int32), 0) - 1
    tl.debug_barrier()
    tl.store(BUF + dst, v, mask=keep)
    tl.debug_barrier()
    return (thr_u ^ 0x8000000000000000).to(tl.int64, bitcast=True)


@triton.jit
def _offer(BUF, key, live, n, thr, K: tl.constexpr, CAP: tl.constexpr):
    """Append the tile's keys above the threshold (compacting first when they might not fit)."""

    m = live & (key > thr)
    cnt = tl.sum(m.to(tl.int32))
    if n + cnt > CAP:
        thr = _compact(BUF, n, K, CAP)
        n = K
        m = live & (key > thr)
        cnt = tl.sum(m.to(tl.int32))
    dst = n + tl.cumsum(m.to(tl.int32), 0) - 1
    tl.store(BUF + dst, key, mask=m)
    tl.debug_barrier()
    return n + cnt, thr


@triton.jit
def _stream(QI, W, w_stride, IK, POS, KEYS, k_stride, NK, BUF, nsplit, RATIO: tl.constexpr, H: tl.constexpr,
            D: tl.constexpr, BP: tl.constexpr, WS: tl.constexpr, SCALE: tl.constexpr, MODE: tl.constexpr,
            SPLIT: tl.constexpr, K: tl.constexpr, CAP: tl.constexpr, BS: tl.constexpr, PT=None, PSH: tl.constexpr = 0):
    """Program (row r, split c). MODE 0: positions [c SPLIT, ..) < n(q) = (q + 1) // RATIO; 1: gathered keys
    KEYS[r, i] (i < NK, -1 = none); 2: blocks of BS positions (block b's key from its visible members' max, the
    newest block +inf). BUF[(r nsplit + c) CAP ..]: the split's K best keys in [0, K), KEY_NONE after."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    q = tl.load(POS) + r
    nvis = (q + 1) // RATIO
    buf = BUF + (r.to(tl.int64) * nsplit + c) * CAP
    n = 0
    thr = tl.full((), -9223372036854775808, tl.int64)
    lo = c * SPLIT
    if MODE == 1:
        hi = tl.minimum(lo + SPLIT, NK)
    else:
        hi = tl.minimum(lo + SPLIT, nvis)
    for t0 in range(lo, hi, BP):
        i = t0 + tl.arange(0, BP)
        if MODE == 1:
            j = tl.load(KEYS + r.to(tl.int64) * k_stride + i, mask=i < hi, other=-1)
            ok = (j >= 0) & (j < nvis) & (i < hi)
        else:
            j = i
            ok = i < hi
        s = score_tile(QI, W, w_stride, IK, r, j, ok, H, D, WS, SCALE, PT, PSH)
        if MODE == 2:
            NBT: tl.constexpr = BP // BS
            m = tl.max(tl.reshape(s, (NBT, BS)), 1)
            b = t0 // BS + tl.arange(0, NBT)
            m = tl.where(b == (nvis - 1) // BS, float("inf"), m)
            live = b * BS < nvis
            n, thr = _offer(buf, index.score_key(m, b), live, n, thr, K, CAP)
        else:
            n, thr = _offer(buf, index.score_key(s, j), ok, n, thr, K, CAP)
    if n > K:
        thr = _compact(buf, n, K, CAP)
        n = K
    kk = tl.arange(0, K)
    tl.store(buf + kk, tl.full((K,), -9223372036854775808, tl.int64), mask=kk >= n)


def splits(keys: int, mode: str = "positions") -> int:
    """Programs a row: ``keys`` = the positions scanned (gather: the candidates' count)."""

    return max(1, triton.cdiv(keys, SPLIT_BLOCKS if mode == "blocks" else SPLIT))


def scratch_bytes(rows: int, keys: int, k: int = index.TOPK, mode: str = "positions") -> int:
    return rows * splits(keys, mode) * 2 * k * 8


def plan(rows: int, keys: int, k: int = index.TOPK, mode: str = "positions", budget: int = BUDGET) -> int:
    """Rows a launch so the scratch fits ``budget``."""

    per = scratch_bytes(1, keys, k, mode)
    return max(1, min(rows, budget // per))


def _launch(qi, w, ik, pos, R, nkeys, ratio, mode, k, keys=None, buf=None, page_table=None, page_shift=0):
    if qi.shape[1:] != (NH, HD) or not qi.is_contiguous() or w.stride(1) != 1:
        raise ValueError("stream_topk: qi [R, 32, 128] contiguous, w unit-stride heads")
    if k & (k - 1):
        raise ValueError("stream_topk: k a power of two (512, 2,048)")
    m = {"positions": 0, "gather": 1, "blocks": 2}[mode]
    ns = splits(nkeys, mode)
    cap = 2 * k
    if buf is None:
        buf = torch.empty((R, ns, cap), dtype=torch.int64, device=qi.device)
    elif buf.numel() < R * ns * cap:
        raise ValueError("stream_topk: scratch too small (see scratch_bytes / plan)")
    kt = keys if keys is not None else buf
    pg = {"PT": page_table, "PSH": page_shift} if page_table is not None else {}
    _stream[(R, ns)](qi, w, w.stride(0), ik, pos, kt, kt.stride(0) if keys is not None else 0, nkeys, buf, ns,
                     RATIO=ratio, H=NH, D=HD, BP=BP, WS=1.0 / math.sqrt(NH), SCALE=1.0 / math.sqrt(HD), MODE=m,
                     SPLIT=SPLIT_BLOCKS if m == 2 else SPLIT, K=k, CAP=cap, BS=index.BLOCK, num_warps=SCORE_WARPS, **pg)
    return buf.view(-1)[:R * ns * cap].view(R, ns, cap)


def merge(buf: torch.Tensor, k: int) -> torch.Tensor:
    """The rows' top ``k`` over their splits' candidates -> positions (or blocks) ascending, -1 padded."""

    R = buf.shape[0]
    return index.top_positions(buf[:, :, :k].reshape(R, -1), k)


def select(qi, w, ik, pos: torch.Tensor, R: int, nvis_max: int, ratio: int, *, buf=None, page_table=None,
           page_shift: int = 0) -> torch.Tensor:
    """Full mode: ``index.select`` without the materialised scores: [R, 512] positions ascending, -1 padded."""

    b = _launch(qi, w, ik, pos, R, max(nvis_max, 1), ratio, "positions", index.TOPK, buf=buf, page_table=page_table,
                page_shift=page_shift)
    return merge(b, index.TOPK)


def reindex(qi, w, ik, pos: torch.Tensor, R: int, cand: torch.Tensor, ratio: int = 1, *, buf=None,
            page_table=None, page_shift: int = 0) -> torch.Tensor:
    """Reindex mode over the candidate blocks' positions (``index.reindex``'s result)."""

    keys = index.candidate_keys(cand)
    b = _launch(qi, w, ik, pos, R, keys.shape[1], ratio, "gather", index.TOPK, keys=keys, buf=buf,
                page_table=page_table, page_shift=page_shift)
    return merge(b, index.TOPK)


def candidate_blocks(qi, w, ik, pos: torch.Tensor, R: int, nvis_max: int, ratio: int = 1, *, buf=None,
                     page_table=None, page_shift: int = 0) -> torch.Tensor:
    """Layer 20's candidate blocks straight from the keys (``index.candidate_blocks`` of the dense scores)."""

    b = _launch(qi, w, ik, pos, R, max(nvis_max, 1), ratio, "blocks", index.TOPK_BLOCKS, buf=buf,
                page_table=page_table, page_shift=page_shift)
    return merge(b, index.TOPK_BLOCKS)


__all__ = ["BUDGET", "SPLIT", "candidate_blocks", "merge", "plan", "reindex", "scratch_bytes", "select", "splits"]
