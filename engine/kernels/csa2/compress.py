"""CSA2's row writers: the kv-source compressors (ratio 2 pooling / ratio 1), the compressed-KV and SWA row stores,
and the indexer key store.

Math (vLLM ``deepseek_v41/compressor.py`` + ``fused_compress_quant_cache.py`` + ``indexer_k_store.py``, Apache-2.0;
our arithmetic, our kernels):

- ratio 2 (layers 2, 8, 14): a group closes at odd position q and pools rows q - 1, q of the fp32 projection
  [kv | score] (1,024 wide): pooled_d = sum_i kv_i,d softmax_i(score_i,d) (a softmax over the 2 rows per dim);
  ratio 1 (layer 20): pooled = kv (512 wide). Then latent = bf16(RMSNorm(pooled) * w) (eps 1e-20, mean over 512).
- the compressed row: RoPE (GPT-J pairs, last 64 dims) at the group's first position (q // r * r, the
  compress-RoPE table: YaRN x16, theta 160,000) of the bf16 latent, stored as ``rows`` records;
- the SWA row of every layer: the same store from kv_norm(wkv x) at the token's own position, into the slot's ring;
- the indexer key (kv-source layers): k = bf16(RMSNorm(wk(latent)) * w_k), RoPE on its last 64 of 128 dims at the
  group's first position, rounded to bf16 and stored bf16 (128 wide: the keys decide the selection, so they stay
  bf16 as GLM's index keys do).

Window handling: the caller passes the window's projection rows with ONE leading carry row (the previous step's last
row, position P - 1), so the pair of a group closing at q sits at buffer rows (q - P, q - P + 1). After a round the
carry is the projection row of the last accepted position (row-invariant, so the window's own row). Snapshots sit on
even positions (a 16-token grid), where no group is open: no carry is ever stored.

Every program handles one row: nothing depends on other rows of the window (row invariance by construction).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import rows as R

EPS = 1e-20


@triton.jit
def _pool_norm(BUF, b_stride, W, LAT, POS, n, EPS: tl.constexpr, RATIO: tl.constexpr):
    """Program r (window row at position P + r): if a group closes there, LAT[r] = bf16(RMSNorm(pooled) * W).
    BUF rows: [carry, row 0, row 1, ...] (ratio 2; fp32 [*, 1024]) or [row 0, ...] (ratio 1; fp32 [*, 512])."""

    r = tl.program_id(0)
    q = tl.load(POS) + r
    d = tl.arange(0, 512)
    if RATIO == 2:
        if (q + 1) % 2 != 0:
            return
        a = BUF + r.to(tl.int64) * b_stride                 # position q - 1 (row r - 1 of the window, or the carry)
        b = a + b_stride                                     # position q
        ka = tl.load(a + d)
        kb = tl.load(b + d)
        sa = tl.load(a + 512 + d)
        sb = tl.load(b + 512 + d)
        peak = tl.maximum(sa, sb)
        wa = tl.exp(sa - peak)
        wb = tl.exp(sb - peak)
        pooled = (ka * wa + kb * wb) / (wa + wb)
    else:
        pooled = tl.load(BUF + r.to(tl.int64) * b_stride + d)
    var = tl.sum(pooled * pooled, 0) / 512.0
    y = pooled * tl.rsqrt(var + EPS) * tl.load(W + d).to(tl.float32)
    tl.store(LAT + r.to(tl.int64) * 512 + d, y.to(tl.bfloat16))


def pool_norm(buf: torch.Tensor, weight: torch.Tensor, pos: torch.Tensor, n: int, ratio: int,
              out: torch.Tensor) -> torch.Tensor:
    """buf fp32 [n + 1, 1024] (ratio 2, carry first) or [n, 512] (ratio 1); out bf16 [n, 512] (rows where no group
    closes are left as they are). ``pos`` int32 [1] on the device: the window's first position."""

    if ratio not in (1, 2) or buf.dtype != torch.float32 or buf.stride(1) != 1:
        raise ValueError("pool_norm: fp32 rows, ratio 1 or 2")
    if buf.shape[1] != 512 * ratio or buf.shape[0] < n + (ratio == 2):
        raise ValueError(f"pool_norm: buf {tuple(buf.shape)} for {n} rows at ratio {ratio}")
    _pool_norm[(n,)](buf, buf.stride(0), weight, out, pos, n, EPS=EPS, RATIO=ratio, num_warps=4)
    return out


@triton.jit
def _kv_store(LAT, l_stride, CS, cs_stride, V, S, POS, RATIO: tl.constexpr, RING: tl.constexpr, PT=None,
              PSH: tl.constexpr = 0):
    """Program r: the row of window row r (position q = P + r). RATIO 0: an SWA row, RoPE at q, ring slot
    q % RING. RATIO 1 / 2: a compressed row (only where a group closes), RoPE at the group's first position,
    compressed row q // RATIO (paged through PT / PSH)."""

    r = tl.program_id(0)
    q = tl.load(POS) + r
    if RATIO == 2:
        if (q + 1) % 2 != 0:
            return
    d = tl.arange(0, 512)
    x = tl.load(LAT + r.to(tl.int64) * l_stride + d).to(tl.float32)
    if RATIO == 0:
        at = q
        row = (q % RING).to(tl.int64)
    else:
        at = q // RATIO * RATIO
        row = R.prow((q // RATIO).to(tl.int64), PT, PSH, None)
    c, s = R.cos_sin(CS, at, cs_stride, 256, 32)
    x = R.rope_pairs(x, c, s, False)
    R.store_row(V, S, row, x, d)


def kv_store(lat: torch.Tensor, cs: torch.Tensor, values, scales, pos: torch.Tensor, n: int, ratio: int,
             ring: int = 0, page_table=None, page_shift: int = 0) -> None:
    """Window rows lat bf16 [n, 512] -> FP8 records: ratio 0 into an SWA ring of ``ring`` rows (a power of two),
    ratio 1 / 2 into the compressed cache (``page_table`` / ``page_shift``: the slot's pages, None = contiguous)."""

    if ratio == 0 and (ring <= 0 or ring & (ring - 1)):
        raise ValueError("kv_store: an SWA ring needs a power-of-two size")
    if values.shape[-1] != R.VB or scales.shape[-1] != R.SB:
        raise ValueError("kv_store: (values [*, 576], scales [*, 8]) uint8")
    pg = dict(PT=page_table, PSH=page_shift) if page_table is not None else {}
    _kv_store[(n,)](lat, lat.stride(0), cs, cs.stride(0), values, scales, pos, RATIO=ratio, RING=max(ring, 1),
                    num_warps=4, **pg)


@triton.jit
def _index_k(KP, k_stride, W, CS, cs_stride, IK, POS, EPS: tl.constexpr, RATIO: tl.constexpr, PT=None,
             PSH: tl.constexpr = 0):
    """Program r: where a group closes at q = P + r, IK[q // RATIO] = bf16(rope(bf16(RMSNorm(KP[r]) * W)))."""

    r = tl.program_id(0)
    q = tl.load(POS) + r
    if RATIO == 2:
        if (q + 1) % 2 != 0:
            return
    d = tl.arange(0, 128)
    k = tl.load(KP + r.to(tl.int64) * k_stride + d).to(tl.float32)
    var = tl.sum(k * k, 0) / 128.0
    k = (k * tl.rsqrt(var + EPS) * tl.load(W + d).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    c, s = R.cos_sin(CS, q // RATIO * RATIO, cs_stride, 64, 32)
    even, odd = tl.split(tl.reshape(k, (64, 2)))
    ne = (even * c - odd * s).to(tl.bfloat16)
    no = (odd * c + even * s).to(tl.bfloat16)
    row = R.prow((q // RATIO).to(tl.int64), PT, PSH, None)
    tl.store(IK + row * 128 + d, tl.reshape(tl.join(ne, no), (128,)))


def index_k(k_pre: torch.Tensor, weight: torch.Tensor, cs: torch.Tensor, ik: torch.Tensor, pos: torch.Tensor, n: int,
            ratio: int, page_table=None, page_shift: int = 0) -> None:
    """k_pre bf16 [n, 128] (wk(latent) of the window's rows) -> bf16 index keys [*, 128] at compressed rows."""

    if k_pre.shape[1] != 128 or ik.shape[-1] != 128 or ik.dtype != torch.bfloat16:
        raise ValueError("index_k: 128-wide bf16 keys")
    pg = dict(PT=page_table, PSH=page_shift) if page_table is not None else {}
    _index_k[(n,)](k_pre, k_pre.stride(0), weight, cs, cs.stride(0), ik, pos, EPS=EPS, RATIO=ratio, num_warps=1,
                   **pg)
