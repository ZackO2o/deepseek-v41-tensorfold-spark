"""The streaming score + top-k (engine/kernels/csa2/stream_topk.py) in Triton's CPU interpreter (``gpu_like``):

- Full mode == ``index.select`` (materialised scores, then the keys' top-512) position for position, for rows with
  at least K visible keys, with tied scores (duplicate keys, a large block of zero scores straddling the cut), many
  splits and compactions (small SPLIT / K) and the real constants; short rows get exactly their visible positions;
- Reindex == ``index.reindex``; candidate blocks == ``index.candidate_blocks`` of the dense scores (newest pinned);
- paged == contiguous; a row alone == in a window; the 300K plan fits the 0.25 GiB scratch.

Run: TRITON_INTERPRET=1 python -m pytest -q tests/kernels/test_stream_topk_interpreter.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from engine.kernels.csa2 import index, ref  # noqa: E402
from engine.kernels.csa2 import stream_topk as ST  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(ST._stream).__name__ == "InterpretedFunction"
pytestmark = [pytest.mark.skipif(not INTERP, reason="Triton's CPU interpreter (TRITON_INTERPRET=1)"),
              pytest.mark.usefixtures("gpu_like")]


def _inputs(R: int, nkeys: int, seed: int, ties: bool = True):
    g = torch.Generator().manual_seed(seed)
    qi = torch.randn((R, 32, 128), generator=g).to(torch.bfloat16).contiguous()
    w = torch.randn((R, 32), generator=g).contiguous()
    ik = torch.randn((nkeys, 128), generator=g)
    if ties:
        ik[100:140] = ik[7]                              # 40 positions with one score each row
        ik[300:900] = 0.0                                # a block of zero scores
        ik[nkeys - 20:nkeys - 10] = ik[nkeys - 21]
    return qi, w, ik.to(torch.bfloat16).contiguous()


def _dense(qi, w, ik, P, R, nv, ratio):
    pos = torch.tensor([P], dtype=torch.int32)
    s = index.scores(qi, w, ik, pos, R, nv, ratio)
    return pos, s, index.top_positions(index.sort_keys(s), index.TOPK)


@pytest.mark.parametrize("ratio,split,k", [(2, 256, 64), (1, 512, 512), (1, 16384, 512)])
def test_select_equals_materialised(monkeypatch, ratio, split, k):
    monkeypatch.setattr(ST, "SPLIT", split)
    monkeypatch.setattr(index, "TOPK", k)
    R, P = 4, 2600 if ratio == 2 else 1900
    nv = (P + R) // ratio
    qi, w, ik = _inputs(R, nv, ratio + split)
    pos, s, want = _dense(qi, w, ik, P, R, nv, ratio)
    got = ST.select(qi, w, ik, pos, R, nv, ratio)
    assert torch.equal(got, want)
    for r in range(R):
        n = (P + r + 1) // ratio
        assert got[r].tolist() == ref.select_stable(s[r, :n], torch.arange(n), k)


def test_short_rows_keep_exactly_their_visible_positions(monkeypatch):
    monkeypatch.setattr(ST, "SPLIT", 256)
    R = 3                                                # 3 rows of a ratio-1 source ...
    qi, w, ik = _inputs(R + 600, 700, 3, ties=False)
    qi, w = qi[R:R + R].contiguous(), w[R:R + R].contiguous()
    pos = torch.tensor([400], dtype=torch.int32)        # ... at 400..402: 401..403 visible keys < 512
    got = ST.select(qi, w, ik, pos, R, 403, 1)
    for r in range(R):
        n = 401 + r
        assert got[r, :n].tolist() == list(range(n)) and bool((got[r, n:] == -1).all())


def test_reindex_and_candidate_blocks(monkeypatch):
    monkeypatch.setattr(index, "TOPK_BLOCKS", 32)
    monkeypatch.setattr(index, "TOPK", 64)
    monkeypatch.setattr(ST, "SPLIT", 128)
    monkeypatch.setattr(ST, "SPLIT_BLOCKS", 256)
    R, P = 3, 1300
    nv = P + R
    qi, w, ik = _inputs(R, nv, 4)
    pos, s, _ = _dense(qi, w, ik, P, R, nv, 1)
    want_b = index.candidate_blocks(s, pos, R)
    got_b = ST.candidate_blocks(qi, w, ik, pos, R, nv, 1)
    assert torch.equal(got_b, want_b)
    qi2, w2, _ = _inputs(R, 1, 5, ties=False)
    want = index.reindex(qi2, w2, ik, pos, R, want_b)
    got = ST.reindex(qi2, w2, ik, pos, R, want_b)
    assert torch.equal(got, want)


def test_paging_and_row_invariance(monkeypatch):
    monkeypatch.setattr(ST, "SPLIT", 512)
    R, P, ratio = 5, 1700, 1
    nv = P + R
    qi, w, ik = _inputs(R, nv, 6)
    pos = torch.tensor([P], dtype=torch.int32)
    got = ST.select(qi, w, ik, pos, R, nv, ratio)
    page = 256
    npg = triton.cdiv(nv, page)
    g = torch.Generator().manual_seed(7)
    pt = torch.randperm(npg + 3, generator=g)[:npg].to(torch.int32).contiguous()
    phys = torch.full(((npg + 3) * page, 128), float("nan"), dtype=torch.bfloat16)
    for j in range(nv):
        phys[int(pt[j // page]) * page + j % page] = ik[j]
    assert torch.equal(ST.select(qi, w, phys, pos, R, nv, ratio, page_table=pt, page_shift=8), got)
    for r in (0, 4):
        one = ST.select(qi[r:r + 1].contiguous(), w[r:r + 1].contiguous(), ik, torch.tensor([P + r], dtype=torch.int32),
                        1, P + r + 1, ratio)
        assert torch.equal(one[0], got[r])


def test_scratch_plan_at_300k():
    for mode, k, keys in (("positions", 512, 300_000), ("positions", 512, 150_000), ("blocks", 2048, 300_000)):
        rows = ST.plan(2048, keys, k, mode)
        assert ST.scratch_bytes(rows, keys, k, mode) <= ST.BUDGET and rows >= 1024, (mode, rows)
