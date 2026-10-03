"""CSA2 kernels (engine/kernels/csa2) in Triton's CPU interpreter, no GPU, with the ``gpu_like`` dot / cast model:

- the FP8 row: the kernel's bytes == ``ref.quantize_rows`` (NoPE e4m3 + UE8M0 scales, RoPE bf16), and the kernel's
  dequantized tile == ``ref.dequantize_rows`` (exact bf16), over hard rows (zero tiles, 448 boundaries, tiny / huge,
  signed zeros);
- compressors (ratio 1 / 2 pooling + RMSNorm), the compressed / SWA row stores and the index-key store against the
  float64 math, and paged == contiguous;
- the indexer: scores against float64, gathered scores == dense scores at the same positions (bit for bit), the
  top-512 == the stable sort (ties to the lower position, signed zeros), candidate blocks (block max, newest pinned,
  2,048 best), reindex over candidates == the masked dense selection;
- attention with sinks over compressed + SWA rows against float64, SWA-only, bounded replay, DSpark's non-causal
  block, paged == contiguous;
- row invariance everywhere: a row alone == the same row inside windows of other sizes and tile-mates, bit for bit.

Run: TRITON_INTERPRET=1 python -m pytest -q tests/kernels/test_csa2_interpreter.py (~1-3 min).
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from engine.kernels.csa2 import attn, compress, index, ref  # noqa: E402
from engine.kernels.csa2 import rows as RW  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(compress._kv_store).__name__ == "InterpretedFunction"
pytestmark = [pytest.mark.skipif(not INTERP, reason="Triton's CPU interpreter (TRITON_INTERPRET=1)"),
              pytest.mark.usefixtures("gpu_like")]


def _bits(t):
    return t.reshape(-1).contiguous().view(torch.uint8)


def _rope_table(n: int, seed: int = 0, theta: float = 160000.0) -> torch.Tensor:
    """[n, 64] fp32: cos for 32 pair frequencies, then sin (a plain table; YaRN only changes the values)."""

    inv = theta ** (-torch.arange(0, 64, 2, dtype=torch.float64) / 64)
    ang = torch.arange(n, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], 1).float().contiguous()


def _pages(n_logical: int, seed: int, page: int):
    """A scrambled page table for ``n_logical`` pages over n_logical + 3 physical pages."""

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_logical + 3, generator=g)[:n_logical].to(torch.int32)
    return order.contiguous(), page.bit_length() - 1


# -- the FP8 row -----------------------------------------------------------------------------------------------------
@triton.jit
def _roundtrip(X, V, S, OUT, n):
    r = tl.program_id(0)
    d = tl.arange(0, 512)
    RW.store_row(V, S, r.to(tl.int64), tl.load(X + r * 512 + d), d)


@triton.jit
def _load(V, S, OUT, n, KT: tl.constexpr):
    d = tl.arange(0, 512)
    i = tl.arange(0, KT)
    ok = i < n
    y = RW.load_rows(V, S, i.to(tl.int64), ok, d)
    tl.store(OUT + i[:, None] * 512 + d[None, :], y, mask=ok[:, None])


def _hard_rows():
    g = torch.Generator().manual_seed(3)
    x = torch.randn((16, 512), generator=g) * 2
    x[1, :64] = 0.0                                    # a zero tile: the 1e-4 floor
    x[2, :64] = 448.0                                  # exactly 448: e = 0
    x[2, 64:128] = 448.0 * 1.0000001                   # just past it: e = 1
    x[3] *= 1e-6                                       # tiny
    x[4] *= 3e4                                        # huge
    x[5, ::2] = -0.0
    x[6, 100] = 224.0                                  # a tile at 1.75 x 2^7
    x[7] = torch.where(x[7].abs() < 0.5, x[7] * 1e-5, x[7])   # mixed magnitudes in one tile (subnormal e4m3)
    return x.contiguous()


def test_row_bytes_and_dequant_match_reference():
    x = _hard_rows()
    n = x.shape[0]
    v = torch.zeros((n, RW.VB), dtype=torch.uint8)
    s = torch.zeros((n, RW.SB), dtype=torch.uint8)
    _roundtrip[(n,)](x, v, s, None, n)
    rv, rs = ref.quantize_rows(x)
    assert torch.equal(v, rv) and torch.equal(s, rs)
    out = torch.zeros((n, 512), dtype=torch.bfloat16)
    _load[(1,)](v, s, out, n, KT=16)
    assert torch.equal(_bits(out), _bits(ref.dequantize_rows(rv, rs)))
    # the dequantized NoPE values are within half an e4m3 ulp of the input: <= 16 x 2^e (ulp 32 at 256..448)
    deq = ref.dequantize_rows(rv, rs).float()[:, :448]
    err = (deq - x[:, :448]).abs()
    step = ((rs[:, :7].to(torch.int32)) << 23).view(torch.float32).repeat_interleave(64, 1)
    assert bool((err <= 16 * step).all())


# -- compressors and stores --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ratio", [1, 2])
def test_pool_norm(ratio):
    g = torch.Generator().manual_seed(ratio)
    n, P = 7, 13                                       # odd first position: the carry closes the first group
    buf = (torch.randn((n + (ratio == 2), 512 * ratio), generator=g) * 3).contiguous()
    w = (torch.rand(512, generator=g) + 0.5).to(torch.bfloat16)
    pos = torch.tensor([P], dtype=torch.int32)
    out = torch.zeros((n, 512), dtype=torch.bfloat16)
    compress.pool_norm(buf, w, pos, n, ratio, out)
    want = ref.pool_norm(buf, w, P, n, ratio)
    assert set(want) == ({r for r in range(n) if (P + r) % 2 == 1} if ratio == 2 else set(range(n)))
    for r, y in want.items():
        torch.testing.assert_close(out[r].double(), y, rtol=1e-2, atol=1e-2)
    # row invariance: each row from its own two-row (or one-row) buffer at its own position
    for r in want:
        one = torch.zeros((1, 512), dtype=torch.bfloat16)
        sub = buf[r:r + (2 if ratio == 2 else 1)].contiguous()
        compress.pool_norm(sub, w, torch.tensor([P + r], dtype=torch.int32), 1, ratio, one)
        assert torch.equal(_bits(one[0]), _bits(out[r]))


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_kv_store(ratio):
    g = torch.Generator().manual_seed(10 + ratio)
    n, P, ring = 9, 300, 256
    lat = (torch.randn((n, 512), generator=g)).to(torch.bfloat16)
    cs = _rope_table(1024)
    pos = torch.tensor([P], dtype=torch.int32)
    rows = ring if ratio == 0 else 1024
    v = torch.zeros((rows, RW.VB), dtype=torch.uint8)
    s = torch.zeros((rows, RW.SB), dtype=torch.uint8)
    compress.kv_store(lat, cs, v, s, pos, n, ratio, ring=ring)
    for r in range(n):
        q = P + r
        if ratio == 2 and (q + 1) % 2:
            continue
        at = q if ratio == 0 else q // ratio * ratio
        row = q % ring if ratio == 0 else q // ratio
        x = ref.rope(lat[r:r + 1], cs, torch.tensor([at]))
        rv, rs = ref.quantize_rows(x.float())
        assert torch.equal(v[row, :448], rv[0, :448]) and torch.equal(s[row], rs[0])
        got = v[row, 448:].contiguous().view(torch.bfloat16).float()
        torch.testing.assert_close(got, x[0, 448:].float(), rtol=1e-2, atol=1e-2)
    if ratio:
        # paged == contiguous, and nothing else written
        page = 64
        pt, psh = _pages(1024 // page, seed=ratio, page=page)
        pv = torch.full(((1024 // page + 3) * page, RW.VB), 0x7F, dtype=torch.uint8)
        ps = torch.full(((1024 // page + 3) * page, RW.SB), 0x7F, dtype=torch.uint8)
        compress.kv_store(lat, cs, pv, ps, pos, n, ratio, page_table=pt, page_shift=psh)
        for j in range(P // ratio, (P + n) // ratio):
            phys = int(pt[j // page]) * page + j % page
            assert torch.equal(pv[phys], v[j]) and torch.equal(ps[phys], s[j])


def test_index_k():
    g = torch.Generator().manual_seed(4)
    n, P = 6, 41
    kp = torch.randn((n, 128), generator=g).to(torch.bfloat16)
    w = (torch.rand(128, generator=g) + 0.5).to(torch.bfloat16)
    cs = _rope_table(256)
    ik = torch.zeros((128, 128), dtype=torch.bfloat16)
    compress.index_k(kp, w, cs, ik, torch.tensor([P], dtype=torch.int32), n, 2)
    for r in range(n):
        q = P + r
        if (q + 1) % 2:
            continue
        k = kp[r].double()
        k = (k / torch.sqrt((k * k).mean() + 1e-20) * w.double()).to(torch.bfloat16)
        want = ref.rope(k[None], cs, torch.tensor([q - 1]), dims=128)[0]
        torch.testing.assert_close(ik[q // 2].double(), want, rtol=1e-2, atol=1e-2)


# -- the indexer -------------------------------------------------------------------------------------------------------
def _index_inputs(R: int, nkeys: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    qi = torch.randn((R, 32, 128), generator=g).to(torch.bfloat16).contiguous()
    w = torch.randn((R, 32), generator=g).contiguous()
    ik = torch.randn((nkeys, 128), generator=g).to(torch.bfloat16).contiguous()
    return qi, w, ik


def test_scores_against_float64_and_gather():
    R, P, ratio = 3, 1400, 2
    nv = (P + R) // ratio
    qi, w, ik = _index_inputs(R, nv + 8, 1)
    pos = torch.tensor([P], dtype=torch.int32)
    s = index.scores(qi, w, ik, pos, R, nv, ratio)
    for r in range(R):
        n = (P + r + 1) // ratio
        want = ref.index_scores(qi[r:r + 1], w[r:r + 1], ik, [torch.arange(n)])[0]
        torch.testing.assert_close(s[r, :n].double(), want, rtol=1e-4, atol=1e-4)
        assert bool(torch.isinf(s[r, n:]).all())
    # gathered keys (any order, gaps, -1) give the dense kernel's bits at the same positions
    g = torch.Generator().manual_seed(9)
    keys = torch.randint(-1, nv, (R, 300), generator=g, dtype=torch.int32)
    sg = index.scores(qi, w, ik, pos, R, 300, ratio, keys=keys)
    for r in range(R):
        n = (P + r + 1) // ratio
        for i, j in enumerate(keys[r].tolist()):
            if 0 <= j < n:
                assert _bits(sg[r, i]).tolist() == _bits(s[r, j]).tolist()
            else:
                assert sg[r, i] == float("-inf")


def test_scores_row_invariance_and_paging():
    R, P, ratio = 5, 999, 1
    nv = P + R
    qi, w, ik = _index_inputs(R, nv, 2)
    pos = torch.tensor([P], dtype=torch.int32)
    s = index.scores(qi, w, ik, pos, R, nv, ratio)
    for r in (0, 3, 4):
        one = index.scores(qi[r:r + 1].contiguous(), w[r:r + 1].contiguous(), ik, torch.tensor([P + r], dtype=torch.int32),
                           1, nv, ratio)
        assert torch.equal(_bits(one[0]), _bits(s[r]))
    page = 256
    pt, psh = _pages(triton.cdiv(nv, page), seed=5, page=page)
    phys = torch.full(((pt.numel() + 3) * page, 128), float("nan"), dtype=torch.bfloat16)
    for j in range(nv):
        phys[int(pt[j // page]) * page + j % page] = ik[j]
    sp = index.scores(qi, w, phys, pos, R, nv, ratio, page_table=pt, page_shift=psh)
    assert torch.equal(_bits(sp), _bits(s))


def test_selection_is_the_stable_sort():
    R, P, ratio = 2, 2600, 2
    nv = (P + R) // ratio
    qi, w, ik = _index_inputs(R, nv, 3)
    pos = torch.tensor([P], dtype=torch.int32)
    s = index.scores(qi, w, ik, pos, R, nv, ratio)
    s[0, 10:40] = s[0, 5]                               # ties
    s[1, 7], s[1, 8] = 0.0, -0.0                        # signed zeros compare equal
    s[1, 100:700] = 0.0                                 # a large tied block straddling the cut
    sel = index.top_positions(index.sort_keys(s), index.TOPK)
    for r in range(R):
        n = (P + r + 1) // ratio
        want = ref.select_stable(s[r, :n], torch.arange(n), index.TOPK)
        assert sel[r].tolist() == want
    # short rows keep everything (-1 padded)
    few = index.top_positions(index.sort_keys(s[:, :100]), index.TOPK)
    assert few[0, :100].tolist() == list(range(100)) and bool((few[0, 100:] == -1).all())


def test_candidates_and_reindex(monkeypatch):
    monkeypatch.setattr(index, "TOPK_BLOCKS", 24)       # 24 blocks of 8 = 192 candidates (the real: 2,048 x 8)
    monkeypatch.setattr(index, "TOPK", 64)
    R, P = 3, 700
    nv = P + R
    qi, w, ik = _index_inputs(R, nv, 4)
    pos = torch.tensor([P], dtype=torch.int32)
    s = index.scores(qi, w, ik, pos, R, nv, 1)
    s[1, 16:24] = s[1, 0:8]                             # two blocks with equal maxima
    blocks = index.candidate_blocks(s, pos, R)
    for r in range(R):
        n = P + r + 1
        nb = math.ceil(n / 8)
        bmax = [max(float(x) for x in s[r, 8 * b:min(8 * b + 8, n)]) for b in range(nb)]
        bmax[nb - 1] = math.inf
        order = sorted(range(nb), key=lambda b: (-bmax[b], b))[:24]
        assert blocks[r].tolist() == sorted(order)
    # reindex over the candidates == the dense scan with every non-candidate at -inf
    qi2, w2, _ = _index_inputs(R, 1, 5)
    sel = index.reindex(qi2, w2, ik, pos, R, blocks)
    dense = index.scores(qi2, w2, ik, pos, R, nv, 1)
    keys = index.candidate_keys(blocks)
    for r in range(R):
        keep = set(j for j in keys[r].tolist() if j >= 0)
        masked = torch.tensor([float(dense[r, j]) if j in keep else -math.inf for j in range(nv)])
        want = ref.select_stable(masked[:P + r + 1], torch.arange(P + r + 1), 64)
        want = [j for j in want if j in keep] + [-1] * (64 - len([j for j in want if j in keep]))
        assert sel[r].tolist() == want


# -- attention ---------------------------------------------------------------------------------------------------------
def _attn_case(R: int, P: int, ncomp: int, seed: int, ring: int = 256):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn((R, 32, 512), generator=g) * 0.5).to(torch.bfloat16).contiguous()
    comp_x = torch.randn((ncomp, 512), generator=g) * 2
    cv, csc = ref.quantize_rows(comp_x)
    swa_pos = range(max(0, P - 140), P + R)
    sv = torch.zeros((ring, RW.VB), dtype=torch.uint8)
    ssc = torch.zeros((ring, RW.SB), dtype=torch.uint8)
    swa = {}
    for p in swa_pos:
        v, s = ref.quantize_rows(torch.randn((1, 512), generator=g) * 2)
        sv[p % ring], ssc[p % ring] = v[0], s[0]
        swa[p] = ref.dequantize_rows(v, s)[0]
    sink = torch.randn(32, generator=g) * 2
    cs = _rope_table(P + R + 8)
    return q, (cv, csc), (sv, ssc), swa, sink, cs


def _lists(R: int, ncomp: int, seed: int, sizes=None):
    g = torch.Generator().manual_seed(seed)
    tok = torch.full((R, 512), -1, dtype=torch.int32)
    cnt = torch.zeros(R, dtype=torch.int32)
    lists = []
    for r in range(R):
        k = sizes[r] if sizes else int(torch.randint(1, 512, (1,), generator=g))
        sel = sorted(torch.randperm(ncomp, generator=g)[:k].tolist())
        tok[r, :len(sel)] = torch.tensor(sel, dtype=torch.int32)
        cnt[r] = len(sel)
        lists.append(sel)
    return tok, cnt, lists


def _run_attn(q, comp, tok, cnt, swa_t, lo, P, sink, cs, *, hi=None, ring=256, pg=None):
    R = q.shape[0]
    out = torch.zeros((R, 32, 512), dtype=torch.bfloat16)
    sc = attn.Scratch(max(R, 1), "cpu")
    attn.attention(q, comp, tok, cnt, swa_t, lo, torch.tensor([P], dtype=torch.int32), sink, cs, out, sc, ring=ring,
                   hi=hi, **(pg or {}))
    return out


def test_attention_against_float64():
    R, P, ncomp = 4, 900, 700
    q, comp, swa_t, swa, sink, cs = _attn_case(R, P, ncomp, 1)
    tok, cnt, lists = _lists(R, ncomp, 2, sizes=[512, 300, 1, 129])
    lo = torch.zeros(R, dtype=torch.int32)
    out = _run_attn(q, comp, tok, cnt, swa_t, lo, P, sink, cs)
    comp_rows = ref.dequantize_rows(*comp)
    want = ref.attention(q, comp_rows, lists, swa, [0] * R, [P + r for r in range(R)], [P + r for r in range(R)],
                         sink, cs)
    torch.testing.assert_close(out.double(), want, rtol=2e-2, atol=5e-3)


def test_attention_swa_only_replay_and_dspark():
    R, P = 5, 60
    q, _, swa_t, swa, sink, cs = _attn_case(R, P, 1, 3)
    lo = torch.tensor([0, 0, 40, 40, 61], dtype=torch.int32)      # bounded replay: SWA truncated to a segment
    out = _run_attn(q, None, None, None, swa_t, lo, P, sink, cs)
    want = ref.attention(q, None, [[]] * R, swa, lo.tolist(), [P + r for r in range(R)], [P + r for r in range(R)],
                         sink, cs)
    torch.testing.assert_close(out.double(), want, rtol=2e-2, atol=5e-3)
    hi = torch.full((R,), P + R - 1, dtype=torch.int32)           # DSpark: every block row sees the whole block
    outd = _run_attn(q, None, None, None, swa_t, torch.zeros(R, dtype=torch.int32), P, sink, cs, hi=hi)
    wantd = ref.attention(q, None, [[]] * R, swa, [0] * R, hi.tolist(), [P + r for r in range(R)], sink, cs)
    torch.testing.assert_close(outd.double(), wantd, rtol=2e-2, atol=5e-3)


def test_attention_row_invariance_and_paging():
    R, P, ncomp = 6, 1000, 900
    q, comp, swa_t, swa, sink, cs = _attn_case(R, P, ncomp, 5)
    tok, cnt, lists = _lists(R, ncomp, 6)
    lo = torch.zeros(R, dtype=torch.int32)
    out = _run_attn(q, comp, tok, cnt, swa_t, lo, P, sink, cs)
    for r in (0, 2, 5):                                          # alone, at its own position
        one = _run_attn(q[r:r + 1].contiguous(), comp, tok[r:r + 1].contiguous(), cnt[r:r + 1].contiguous(), swa_t,
                        lo[r:r + 1].contiguous(), P + r, sink, cs)
        assert torch.equal(_bits(one[0]), _bits(out[r]))
    # rows 1..3 as a 3-row window (other tile-mates, other window size)
    mid = _run_attn(q[1:4].contiguous(), comp, tok[1:4].contiguous(), cnt[1:4].contiguous(), swa_t,
                    lo[1:4].contiguous(), P + 1, sink, cs)
    assert torch.equal(_bits(mid), _bits(out[1:4]))
    # the compressed cache paged through a scrambled table: same bits
    page = 64
    pt, psh = _pages(triton.cdiv(ncomp, page), seed=7, page=page)
    nphys = (pt.numel() + 3) * page
    pv = torch.full((nphys, RW.VB), 0x7F, dtype=torch.uint8)
    ps = torch.full((nphys, RW.SB), 0xFF, dtype=torch.uint8)
    for j in range(ncomp):
        pv[int(pt[j // page]) * page + j % page] = comp[0][j]
        ps[int(pt[j // page]) * page + j % page] = comp[1][j]
    outp = _run_attn(q, (pv, ps), tok, cnt, swa_t, lo, P, sink, cs, pg=dict(page_table=pt, page_shift=psh))
    assert torch.equal(_bits(outp), _bits(out))
    # a -inf sink is no sink: the plain softmax
    none = _run_attn(q[:1].contiguous(), comp, tok[:1].contiguous(), cnt[:1].contiguous(), swa_t, lo[:1].contiguous(),
                     P, torch.full((32,), float("-inf")), cs)
    want = ref.attention(q[:1], ref.dequantize_rows(*comp), lists[:1], swa, [0], [P], [P],
                         torch.full((32,), -1e30, dtype=torch.float64), cs)
    torch.testing.assert_close(none.double(), want, rtol=2e-2, atol=5e-3)


def test_prefill_ring_gives_decode_bits():
    """A prefill chunk through a larger staging ring (the same rows at other addresses) == the decode ring."""

    R, P = 3, 500
    q, comp, swa_t, swa, sink, cs = _attn_case(R, P, 600, 8)
    tok, cnt, _ = _lists(R, 600, 9)
    lo = torch.zeros(R, dtype=torch.int32)
    out = _run_attn(q, comp, tok, cnt, swa_t, lo, P, sink, cs, ring=256)
    big_v = torch.zeros((1024, RW.VB), dtype=torch.uint8)
    big_s = torch.zeros((1024, RW.SB), dtype=torch.uint8)
    for p in swa:
        big_v[p % 1024], big_s[p % 1024] = swa_t[0][p % 256], swa_t[1][p % 256]
    out2 = _run_attn(q, comp, tok, cnt, (big_v, big_s), lo, P, sink, cs, ring=1024)
    assert torch.equal(_bits(out2), _bits(out))
