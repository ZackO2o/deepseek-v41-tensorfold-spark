"""CSA2 sparse + sliding-window attention with sinks for DeepSeek-V4.1-Flash on sm_121 (adapted from our GLM latent
kernels ``glm5_next/spark/latent.py``: ``_lsparse_chunks`` / ``_lsparse_merge``).

The math matches GLM's absorbed-MLA latent attention without the absorb / expand: 64 query heads (32 a rank) attend
one 512-wide K = V row per key. A query row r at position q (RoPE'd q bf16 [32, 512]) attends

- its index source's selected compressed rows (<= 512, ``index.select`` / ``reindex``; none on layers 0-1 and the
  DSpark blocks) from the kv-source layer's compressed cache, and
- its own layer's SWA rows at positions [lo_r, q] (lo_r = max(q - 127, the bounded-replay segment start)) from the
  slot's ring,

with a per-head sink logit (a key with logit ``sink_h`` and value 0: it only adds exp(sink - m) to the softmax's
denominator), scale 512^-0.5, and the output's last 64 dims rotated back (inverse GPT-J RoPE at q). The output is
bf16 [R, 32, 512] for the grouped ``wo_a``.

Arithmetic, fixed by the shapes and never by the row count or the window:

- keys come in chunks of CH = 128 list entries: chunks 0-3 are the compressed list's entries [128 c, 128 c + 128)
  (in list order, ascending positions), chunk 4 the SWA window (ascending positions), 32 keys a tile;
- a chunk's partial is the GLM tile step: bf16 q x bf16 rows on tensor cores (fp32 sums), the online softmax in fp32,
  bf16 probabilities x rows; the FP8 rows are dequantized exactly (``rows.load_rows``);
- the partials merge in chunk order (compressed 0..3, then SWA), then the sink, then 1 / l, the inverse RoPE, one
  bf16 rounding.

Each (row, head) is one row of a 16-head tile whatever its tile-mates, and its chunks are its own lists, so a window
row gets the serial step's bits (verify == serial, drafted == serial, batched == alone); prefill rows run the same
kernels. No atomics.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import rows as RW

HEADS = 32           # local heads a rank (64 / TP 2)
D = 512
CH = 128             # list entries a chunk
NCOMP = 4            # compressed chunks (index_topk 512 / CH)
KT = 32              # keys a tile
BMQ = 16             # heads a tensor-core tile
WINDOW = 128
WARPS = 8


@triton.jit
def _tile(q, kv, m, l, o, valid, SCALE: tl.constexpr):
    """GLM's ``_ltile``: one tile of rows (keys = values) for BMQ queries, the online softmax step."""

    s = tl.dot(q, tl.trans(kv)).to(tl.float32) * SCALE
    s = tl.where(valid, s, float("-inf"))
    tm = tl.max(s, 1)
    active = tm != float("-inf")
    nm = tl.where(active, tl.maximum(m, tm), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - nm)), 1.0)
    p = tl.where(valid & active[:, None], tl.exp(s - nm[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    l = l * alpha + tl.sum(p, 1)
    return nm, l, o


@triton.jit
def _chunks(Q, CV, CSC, TOK, t_stride, CNT, SV, SSC, LO, HI, POS, PO, PM, PL, R, H: tl.constexpr, CH: tl.constexpr,
            NCOMP: tl.constexpr, KT: tl.constexpr, BMQ: tl.constexpr, SCALE: tl.constexpr, RING: tl.constexpr,
            WINDOW: tl.constexpr, HAS_HI: tl.constexpr, PT=None, PSH: tl.constexpr = 0):
    """Program (row r, head tile, chunk c): c < NCOMP: the row's compressed entries [c CH, (c + 1) CH);
    c == NCOMP: its SWA window, positions [max(LO[r], hi - 127), hi] with hi = HI[r] (HAS_HI: DSpark's block rows
    see their whole block) or the row's own position. Partials (o, m, l) at [c][r][h]."""

    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    h = hb * BMQ + tl.arange(0, BMQ)
    d = tl.arange(0, 512)
    m = tl.full((BMQ,), float("-inf"), tl.float32)
    l = tl.zeros((BMQ,), tl.float32)
    o = tl.zeros((BMQ, 512), tl.float32)
    q = tl.load(Q + (r.to(tl.int64) * H + h[:, None]) * 512 + d[None, :]).to(tl.bfloat16)
    if c < NCOMP:
        n = tl.load(CNT + r)
        if c * CH < n:
            for t in range(CH // KT):
                idx = c * CH + t * KT + tl.arange(0, KT)
                ok = idx < n
                tok = tl.load(TOK + r.to(tl.int64) * t_stride + idx, mask=ok, other=0).to(tl.int64)
                ok = ok & (tok >= 0)
                rows = RW.prow(tl.where(ok, tok, 0), PT, PSH, ok)
                kv = RW.load_rows(CV, CSC, rows, ok, d)
                m, l, o = _tile(q, kv, m, l, o, ok[None, :], SCALE)
    else:
        if HAS_HI:
            hi = tl.load(HI + r)
        else:
            hi = tl.load(POS) + r
        lo = tl.maximum(tl.load(LO + r), hi - (WINDOW - 1))
        for t in range(WINDOW // KT):
            p = hi - (WINDOW - 1) + t * KT + tl.arange(0, KT)
            ok = (p >= lo) & (p >= 0)
            rows = (tl.where(ok, p, 0) % RING).to(tl.int64)
            kv = RW.load_rows(SV, SSC, rows, ok, d)
            m, l, o = _tile(q, kv, m, l, o, ok[None, :], SCALE)
    base = (c * R + r) * H + h
    tl.store(PO + base[:, None].to(tl.int64) * 512 + d[None, :], o)
    tl.store(PM + base, m)
    tl.store(PL + base, l)


@triton.jit
def _merge(PO, PM, PL, SINK, CS, cs_stride, POS, OUT, R, H: tl.constexpr, NCH: tl.constexpr):
    """Program (r, h): the chunks merged in order, then the sink, 1 / l, the inverse RoPE at q, bf16."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 512)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((512,), tl.float32)
    for c in range(NCH):
        base = (c * R + r) * H + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        co = tl.load(PO + base.to(tl.int64) * 512 + d)
        active = cl > 0.0
        nm = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - nm)), 1.0)
        b = tl.where(active, tl.exp(cm - nm), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = nm
    sink = tl.load(SINK + h).to(tl.float32)
    top = tl.maximum(m, sink)
    a = tl.where(m == float("-inf"), 0.0, tl.exp(m - top))
    e = tl.where(sink == float("-inf"), 0.0, tl.exp(sink - top))
    x = (o * a) / (l * a + e)
    c, s = RW.cos_sin(CS, tl.load(POS) + r, cs_stride, 256, 32)
    x = RW.rope_pairs(x, c, s, True)
    tl.store(OUT + (r.to(tl.int64) * H + h) * 512 + d, x.to(tl.bfloat16))


class Scratch:
    """Chunk partials for windows of up to ``rows`` rows (fp32: 5 chunks x rows x 32 heads x 512)."""

    def __init__(self, rows: int, device, heads: int = HEADS) -> None:
        n = (NCOMP + 1) * rows * heads
        self.rows, self.heads = rows, heads
        self.po = torch.empty((n * D,), dtype=torch.float32, device=device)
        self.pm = torch.empty((n,), dtype=torch.float32, device=device)
        self.pl = torch.empty((n,), dtype=torch.float32, device=device)

    def nbytes(self) -> int:
        return sum(t.numel() * 4 for t in (self.po, self.pm, self.pl))


def scratch_bytes(rows: int, heads: int = HEADS) -> int:
    return (NCOMP + 1) * rows * heads * (D + 2) * 4


def attention(q: torch.Tensor, comp, tokens: torch.Tensor | None, counts: torch.Tensor | None, swa, lo: torch.Tensor,
              pos: torch.Tensor, sink: torch.Tensor, cs: torch.Tensor, out: torch.Tensor, scratch: Scratch, *,
              ring: int, hi: torch.Tensor | None = None, page_table=None, page_shift: int = 0) -> torch.Tensor:
    """q bf16 [R, 32, 512] (RoPE'd); comp = (values, scales) of the compressed cache or None (SWA-only layers);
    tokens int32 [R, <= 512] ascending compressed rows (-1 pad), counts int32 [R]; swa = (values, scales) of the
    slot's ring (``ring`` rows); lo int32 [R] window starts (bounded replay; <= q - 127 means the full window);
    pos int32 [1] the window's first position; hi int32 [R] the window ends (DSpark's non-causal block; default
    each row's position); sink fp32 [32]; cs the RoPE table [max_pos, 64] -> out bf16 [R, 32, 512].

    Prefill chunks use the same kernels with a staging ring of next_pow2(R + 128) rows per layer (the slot ring's
    last 127 rows copied in, the chunk's rows written by ``compress.kv_store``, the last 128 copied back after):
    the same rows at other addresses, so the same bits as decode."""

    R, H, _ = q.shape
    if R > scratch.rows or H != scratch.heads or not q.is_contiguous() or q.dtype != torch.bfloat16:
        raise ValueError("attention: q bf16 [R, 32, 512] contiguous, within the scratch")
    if ring <= 0 or ring & (ring - 1) or ring < WINDOW:
        raise ValueError("attention: the SWA ring is a power of two of at least 128 rows")
    if comp is None:
        cv, csc = swa
        tokens = torch.full((R, 1), -1, dtype=torch.int32, device=q.device)
        counts = torch.zeros((R,), dtype=torch.int32, device=q.device)
        pg = {}
    else:
        cv, csc = comp
        if tokens.shape[1] > NCOMP * CH:
            raise ValueError(f"attention: at most {NCOMP * CH} compressed rows a query row")
        pg = dict(PT=page_table, PSH=page_shift) if page_table is not None else {}
    n = (NCOMP + 1) * R * H
    po, pm, pl = scratch.po[:n * D], scratch.pm[:n], scratch.pl[:n]
    if ring < R + WINDOW - 1:
        raise ValueError(f"attention: a ring of {ring} rows cannot hold {R} rows and their windows")
    _chunks[(R, H // BMQ, NCOMP + 1)](q, cv, csc, tokens, tokens.stride(0), counts, swa[0], swa[1], lo,
                                      hi if hi is not None else lo, pos, po, pm, pl, R, H=H, CH=CH, NCOMP=NCOMP,
                                      KT=KT, BMQ=BMQ, SCALE=D ** -0.5, RING=ring, WINDOW=WINDOW,
                                      HAS_HI=hi is not None, num_warps=WARPS, **pg)
    _merge[(R, H)](po, pm, pl, sink, cs, cs.stride(0), pos, out, R, H=H, NCH=NCOMP + 1, num_warps=4)
    return out
