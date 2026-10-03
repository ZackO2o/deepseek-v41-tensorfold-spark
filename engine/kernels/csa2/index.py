"""CSA2's lightning indexer for DeepSeek-V4.1-Flash: scores, the top-512 selection, and the hierarchical candidate
blocks of the decoder (adapted from our GLM DSA indexer, ``glm5_next/spark/sparse.py``: the same score formula and
the same tie rule).

A query row at position q (index query qi = wq_b(q_lora) RoPE'd: 32 heads x 128; head weights w = weights_proj(x),
32) scores the visible compressed positions j < n(q) = (q + 1) // ratio of its index source:

    s_j = sum_h (w_h / sqrt(32)) * relu(qi_h . k_j / sqrt(128))       (bf16 inputs, fp32 tensor-core dots and sums)

- Full mode (index sources 2, 8, 14, 20): every visible j. Rows with n(q) <= 512 keep everything (no scoring).
- Candidate source (layer 20): blocks of 8 positions score max_j s_j over their visible members, the block holding
  the newest position is pinned to +inf, and the 2,048 best blocks (16,384 positions) become the row's candidates.
- Reindex (24, 28, 32, 36): the layer's own qi / w over the shared keys of layer 20, scored only on the row's
  candidates (``keys`` = the candidates' positions in ascending order): the same per-key arithmetic as a full scan
  with every other position at -inf, so the selection is the one the masked full scan gives.
- Reuse layers attend to the latest index source's selection.

Selection: the 512 best by (score descending, then the lower position): every score maps to a unique int64 key
(``_keys``: the score's bits in two's-complement order, then 0x7FFFFFFF - j), so ``torch.topk`` over the keys has no
ties and the set does not depend on the sort's implementation; the result is sorted ascending. A row's scores,
keys and selection depend only on that row (one program row a query row, tiles of fixed shape), so a window row
gets the serial step's selection: drafted == serial, batched == alone.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .rows import prow

NH = 32
HD = 128
TOPK = 512
BLOCK = 8
TOPK_BLOCKS = 2048
BP = 64                       # keys a scoring program tile
SCORE_WARPS = 8               # part of the arithmetic (the head sum's reduction tree): fixed, never tuned per call
KEY_NONE = -(2 ** 63)         # below every real key (padding)


@triton.jit
def _scores(QI, W, w_stride, IK, OUT, POS, KEYS, k_stride, NK, o_stride, RATIO: tl.constexpr, H: tl.constexpr,
            D: tl.constexpr, BP: tl.constexpr, WS: tl.constexpr, SCALE: tl.constexpr, GATHER: tl.constexpr,
            PT=None, PSH: tl.constexpr = 0):
    """Program (row r, key tile): OUT[r, i] = s at key i of the row (i < NK; -inf past the row's visible keys).
    GATHER: key i is position KEYS[r, i] (-1 = none); else key i is position i."""

    r = tl.program_id(0)
    pb = tl.program_id(1)
    q = tl.load(POS) + r
    nvis = (q + 1) // RATIO
    i = pb * BP + tl.arange(0, BP)
    if GATHER:
        j = tl.load(KEYS + r.to(tl.int64) * k_stride + i, mask=i < NK, other=-1)
        ok = (j >= 0) & (j < nvis) & (i < NK)
    else:
        j = i
        ok = (j < nvis) & (i < NK)
    s = score_tile(QI, W, w_stride, IK, r, j, ok, H, D, WS, SCALE, PT, PSH)
    tl.store(OUT + r.to(tl.int64) * o_stride + i, s, mask=i < NK)


@triton.jit
def score_tile(QI, W, w_stride, IK, r, j, ok, H: tl.constexpr, D: tl.constexpr, WS: tl.constexpr,
               SCALE: tl.constexpr, PT, PSH: tl.constexpr):
    """Row r's scores at key positions j (a tile; -inf where not ok): the indexer's arithmetic, shared by ``_scores``
    and ``stream_topk`` (same tile shape and warps: the same bits)."""

    hh = tl.arange(0, H)
    d = tl.arange(0, D)
    qv = tl.load(QI + (r.to(tl.int64) * H + hh[:, None]) * D + d[None, :]).to(tl.bfloat16)          # [H, D]
    w = tl.load(W + r.to(tl.int64) * w_stride + hh).to(tl.float32) * WS
    jr = prow(tl.where(ok, j, 0).to(tl.int64), PT, PSH, ok)
    k = tl.load(IK + jr[:, None] * D + d[None, :], mask=ok[:, None], other=0.0).to(tl.bfloat16)       # [BP, D]
    dots = tl.dot(qv, tl.trans(k))                                                                     # [H, BP] fp32
    s = tl.sum(w[:, None] * tl.maximum(dots * SCALE, 0.0), axis=0)
    return tl.where(ok, s, float("-inf"))


@triton.jit
def score_key(s, p):
    """``_keys``' unique int64 key of score s at position p: (score descending, then the lower position); -0.0 ranks
    as +0.0, NaN above +inf."""

    b = s.to(tl.int32, bitcast=True)
    b = tl.where(s == 0.0, 0, b)
    b = tl.where(s != s, 0x7FC00000, b)
    b = tl.where(b < 0, b ^ 0x7FFFFFFF, b)
    return (b.to(tl.int64) << 32) + (0x7FFFFFFF - p).to(tl.int64)


@triton.jit
def _keys(S, s_stride, K, k_stride, POSN, p_stride, NK, BLOCK: tl.constexpr, GATHER: tl.constexpr):
    """Program (r, tile): a unique int64 key per score ordered as (score descending, then the lower position):
    -0.0 ranks as +0.0, NaN above +inf; a missing key (GATHER, position -1) is KEY_NONE."""

    r = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = i < NK
    s = tl.load(S + r * s_stride + i, mask=ok, other=float("-inf"))
    if GATHER:
        p = tl.load(POSN + r * p_stride + i, mask=ok, other=-1)
    else:
        p = i
    b = s.to(tl.int32, bitcast=True)
    b = tl.where(s == 0.0, 0, b)
    b = tl.where(s != s, 0x7FC00000, b)
    b = tl.where(b < 0, b ^ 0x7FFFFFFF, b)
    key = (b.to(tl.int64) << 32) + (0x7FFFFFFF - p).to(tl.int64)
    key = tl.where(p >= 0, key, tl.full((BLOCK,), -9223372036854775808, tl.int64))
    tl.store(K + r * k_stride + i, key, mask=ok)


@triton.jit
def _block_keys(S, s_stride, K, k_stride, POS, NB, RATIO: tl.constexpr, BS: tl.constexpr, TB: tl.constexpr):
    """Program (r, block tile): block b's key from max over its visible scores (the newest block: +inf), ordered as
    (score descending, then the lower block); blocks past the row's visible ones: KEY_NONE."""

    r = tl.program_id(0)
    b = tl.program_id(1) * TB + tl.arange(0, TB)
    q = tl.load(POS) + r
    nvis = (q + 1) // RATIO
    j = b[:, None] * BS + tl.arange(0, BS)[None, :]
    v = tl.load(S + r.to(tl.int64) * s_stride + j, mask=j < nvis, other=float("-inf"))
    m = tl.max(v, 1)
    live = b * BS < nvis
    m = tl.where(b == (nvis - 1) // BS, float("inf"), m)
    bits = m.to(tl.int32, bitcast=True)
    bits = tl.where(m == 0.0, 0, bits)
    bits = tl.where(m != m, 0x7FC00000, bits)
    bits = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    key = (bits.to(tl.int64) << 32) + (0x7FFFFFFF - b).to(tl.int64)
    key = tl.where(live, key, tl.full((TB,), -9223372036854775808, tl.int64))
    tl.store(K + r.to(tl.int64) * k_stride + b, key, mask=b < NB)


def _paging(page_table, page_shift):
    return dict(PT=page_table, PSH=page_shift) if page_table is not None else {}


def scores(qi: torch.Tensor, w: torch.Tensor, ik, pos: torch.Tensor, R: int, nkeys: int, ratio: int, *,
           keys: torch.Tensor | None = None, out: torch.Tensor | None = None, page_table=None,
           page_shift: int = 0) -> torch.Tensor:
    """qi bf16 [R, 32, 128], w fp32 [R, 32], ik bf16 [*, 128] -> fp32 [R, nkeys]: dense (key i = position i) or
    over ``keys`` int32 [R, nkeys] positions (-1 padded)."""

    if qi.shape[1:] != (NH, HD) or not qi.is_contiguous() or w.stride(1) != 1:
        raise ValueError("scores: qi [R, 32, 128] contiguous, w unit-stride heads")
    if out is None:
        out = torch.empty((R, nkeys), dtype=torch.float32, device=qi.device)
    gather = keys is not None
    kt = keys if gather else out
    _scores[(R, triton.cdiv(nkeys, BP))](qi, w, w.stride(0), ik, out, pos, kt, kt.stride(0), nkeys, out.stride(0),
                                         RATIO=ratio, H=NH, D=HD, BP=BP, WS=1.0 / math.sqrt(NH),
                                         SCALE=1.0 / math.sqrt(HD), GATHER=gather, num_warps=SCORE_WARPS,
                                         **_paging(page_table, page_shift))
    return out


def sort_keys(s: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
    """Scores [R, n] (and their positions for gathered keys) -> unique int64 keys [R, n]."""

    R, n = s.shape
    k = torch.empty((R, n), dtype=torch.int64, device=s.device)
    p = positions if positions is not None else s
    _keys[(R, triton.cdiv(n, 1024))](s, s.stride(0), k, k.stride(0), p, p.stride(0), n, BLOCK=1024,
                                     GATHER=positions is not None, num_warps=4)
    return k


def top_positions(k: torch.Tensor, count: int) -> torch.Tensor:
    """The ``count`` largest keys' positions, ascending, -1 padded (missing keys): int32 [R, count]."""

    R, n = k.shape
    top = torch.topk(k, min(count, n), dim=1, sorted=False).values
    pos = torch.where(top == KEY_NONE, torch.full_like(top, -1), 0x7FFFFFFF - (top & 0xFFFFFFFF))
    pos = torch.where(pos < 0, torch.full_like(pos, 2 ** 31 - 1), pos)
    pos = torch.sort(pos, dim=1).values
    pos = torch.where(pos == 2 ** 31 - 1, torch.full_like(pos, -1), pos)
    if pos.shape[1] < count:
        pos = torch.cat([pos, torch.full((R, count - pos.shape[1]), -1, dtype=pos.dtype, device=pos.device)], 1)
    return pos.to(torch.int32).contiguous()


def select(qi, w, ik, pos: torch.Tensor, R: int, nvis_max: int, ratio: int, *, page_table=None,
           page_shift: int = 0) -> torch.Tensor:
    """Full mode: every row's 512 selected positions (ascending, -1 padded; rows with <= 512 visible keep all)."""

    s = scores(qi, w, ik, pos, R, max(nvis_max, 1), ratio, page_table=page_table, page_shift=page_shift)
    return top_positions(sort_keys(s), TOPK)


def candidate_blocks(s: torch.Tensor, pos: torch.Tensor, R: int, ratio: int = 1) -> torch.Tensor:
    """Layer 20's candidates from its full scores s [R, n]: the 2,048 best blocks of 8 (newest pinned), ascending
    block ids, -1 padded."""

    n = s.shape[1]
    nb = triton.cdiv(n, BLOCK)
    k = torch.empty((R, nb), dtype=torch.int64, device=s.device)
    _block_keys[(R, triton.cdiv(nb, 256))](s, s.stride(0), k, k.stride(0), pos, nb, RATIO=ratio, BS=BLOCK, TB=256,
                                          num_warps=4)
    return top_positions(k, TOPK_BLOCKS)


def candidate_keys(blocks: torch.Tensor) -> torch.Tensor:
    """Ascending candidate blocks [R, 2,048] -> their positions [R, 16,384] ascending (-1 for missing blocks)."""

    j = blocks.to(torch.int64)[:, :, None] * BLOCK + torch.arange(BLOCK, device=blocks.device)
    j = torch.where(blocks[:, :, None] >= 0, j, torch.full_like(j, -1))
    return j.reshape(blocks.shape[0], -1).to(torch.int32).contiguous()


def reindex(qi, w, ik, pos: torch.Tensor, R: int, cand: torch.Tensor, ratio: int = 1, *, page_table=None,
            page_shift: int = 0) -> torch.Tensor:
    """Reindex mode: the row's own scores over its candidates' positions -> its 512 selected positions."""

    keys = candidate_keys(cand)
    s = scores(qi, w, ik, pos, R, keys.shape[1], ratio, keys=keys, page_table=page_table, page_shift=page_shift)
    return top_positions(sort_keys(s, keys), TOPK)
