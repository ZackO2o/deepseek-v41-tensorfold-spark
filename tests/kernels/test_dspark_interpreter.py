"""DSpark's kernels and glue (engine/kernels/dspark) in Triton's CPU interpreter:

- the chain == ``dspark.ref.chain`` (the Markov bias on candidates + ``exact_sampling.choose``) token for token,
  greedy and sampled (top_k / top_p / min_p), with the confidence head; a slot alone == in a batch (bit for bit);
- on the tiny config, the chain over the whole vocabulary == engine/reference/model.py's DSpark.draft (tokens and
  confidence) from the reference's own base logits and head hidden;
- keyed draft noise: draft logits equal to the target's give the target's keyed choices (full acceptance at T > 0);
- ``candidates`` == the stable (value desc, id asc) top-k with ties; ``attention_meta`` + ``csa2.attn`` == float64
  attention over vLLM's non-causal DSpark window (the 128 context rows + the whole block, with sinks).

Run: TRITON_INTERPRET=1 python -m pytest -q tests/kernels/test_dspark_interpreter.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from engine.kernels import dspark  # noqa: E402
from engine.kernels.dspark import kernels as K  # noqa: E402
from engine.kernels.dspark import ref  # noqa: E402
from engine.kernels.tf import tensorfold_src  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(K._chain).__name__ == "InterpretedFunction"
pytestmark = [pytest.mark.skipif(not INTERP, reason="Triton's CPU interpreter (TRITON_INTERPRET=1)"),
              pytest.mark.usefixtures("gpu_like")]


def _bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def _chain_case(S: int, N: int, C: int, V: int, D: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    cand = torch.stack([torch.stack([torch.randperm(V, generator=g)[:C] for _ in range(N)]) for _ in range(S)])
    cand = cand.to(torch.int32).contiguous()
    cand[0, 1, C - 3:] = -1                                              # padding
    cval = (torch.randn((S, N, C), generator=g) * 3).contiguous()
    w1 = torch.randn((V, 256), generator=g).to(torch.bfloat16)
    w2 = (torch.randn((V, 256), generator=g) * 0.1).to(torch.bfloat16)
    hid = torch.randn((S, N, D), generator=g).to(torch.bfloat16)
    cw = torch.randn(D + 256, generator=g) * 0.02
    anchor = torch.randint(0, V, (S,), generator=g, dtype=torch.int32)
    return cand, cval, w1, w2, hid, cw, anchor


def _params(S: int, samplings, p0: int = 1000):
    tensorfold_src()
    ip, fp = zip(*[dspark.sampling_params(s, p0 + 17 * i) for i, s in enumerate(samplings)])
    return torch.tensor(ip, dtype=torch.int64), torch.tensor(fp, dtype=torch.float32)


def _samplings():
    tensorfold_src()
    from tensorfold.engine.exact_sampling import Sampling

    return [None, Sampling(seed=5, temperature=0.9, top_k=20, top_p=0.95),
            Sampling(seed=6, temperature=1.2, top_k=0, top_p=0.8, min_p=0.02),
            Sampling(seed=7, temperature=0.7, top_k=8, top_p=1.0)]


def _run(cand, cval, w1, w2, hid, cw, anchor, ip, fp):
    S, N, _ = cand.shape
    draft = torch.zeros((S, N), dtype=torch.int32)
    conf = torch.zeros((S, N))
    dspark.chain(cand, cval.clone(), w1, w2, anchor, ip, fp, draft, hid=hid, cw=cw, conf=conf)
    return draft, conf


def test_chain_against_reference():
    S, N, C, V, D = 4, 5, 128, 3000, 1024
    case = _chain_case(S, N, C, V, D, 1)
    ip, fp = _params(S, _samplings())
    draft, conf = _run(*case, ip, fp)
    cand, cval, w1, w2, hid, cw, anchor = case
    want, wconf = ref.chain(cand, cval, w1, w2, anchor, ip, fp, hid, cw)
    assert draft.long().tolist() == want.tolist()
    torch.testing.assert_close(conf, wconf, rtol=1e-5, atol=1e-6)
    # a slot alone == the slot in the batch
    for s in (0, 3):
        sl = slice(s, s + 1)
        d1, c1 = _run(*(t[sl].contiguous() if t.dim() and t.shape[0] == S else t for t in case), ip[sl], fp[sl])
        assert torch.equal(d1[0], draft[s]) and torch.equal(_bits(c1[0]), _bits(conf[s]))


def test_chain_equals_reference_dspark_on_tiny_config():
    from engine.reference import synthetic as SY
    from engine.reference.config import tiny_config
    from engine.reference.model import DSpark, Model
    from engine.reference.ops import Numerics

    cfg = tiny_config(dspark_markov_rank=8)
    m = Model(cfg, SY.model_weights(cfg), Numerics.exact())
    w = SY.dspark_weights(cfg)
    ds = DSpark(cfg, w, m.w.embed, m.w.head, Numerics.exact())
    out = m.forward(torch.tensor([3, 17, 22, 5, 40, 8, 11]), with_logits=False)
    mx = ds.main_x(out.aux)
    d = ds.draft(mx, anchor=8)
    n = d.base_logits.shape[0]
    cand = torch.arange(cfg.vocab_size, dtype=torch.int32).repeat(1, n, 1).contiguous()
    cval = d.base_logits.float()[None].contiguous()
    hid = d.head_hidden.to(torch.bfloat16)[None].contiguous()
    draft, conf = _run(cand, cval, w.markov_w1.to(torch.bfloat16), w.markov_w2.to(torch.bfloat16), hid,
                       w.confidence[0].contiguous(),
                       torch.tensor([8], dtype=torch.int32), *_params(1, [None]))
    assert draft[0].long().tolist() == d.tokens.tolist()
    torch.testing.assert_close(conf[0], d.confidence.float(), rtol=1e-5, atol=1e-6)


def test_keyed_noise_matches_the_target():
    """Candidates = a 96-token vocabulary whose draft logits equal the target's (the Markov dot of small integers
    is exact in any order): the draft is the target's keyed choice at every position."""

    tensorfold_src()
    from tensorfold.engine.exact_sampling import choose_rows

    S, N, V = 3, 5, 96
    g = torch.Generator().manual_seed(4)
    base = torch.randn((S, N, V), generator=g) * 2
    w1 = torch.randint(-2, 3, (V, 256), generator=g).to(torch.bfloat16)
    w2 = (torch.randint(-2, 3, (V, 256), generator=g) * 0.0625).to(torch.bfloat16)
    anchor = torch.tensor([5, 50, 90], dtype=torch.int32)
    sams = _samplings()[1:]
    ip, fp = _params(S, sams)
    cand = torch.arange(V, dtype=torch.int32).repeat(S, N, 1).contiguous()
    draft = torch.zeros((S, N), dtype=torch.int32)
    dspark.chain(cand, base.clone(), w1, w2, anchor, ip, fp, draft)
    ids = np.arange(V, dtype=np.int64)[None]
    for s in range(S):
        prev = int(anchor[s])
        for i in range(N):
            target = (base[s, i] + w2.float() @ w1[prev].float()).numpy()[None]       # the target's row
            want = choose_rows(target, ids, [int(ip[s, 1]) + i], sams[s])[0]
            assert int(draft[s, i]) == want, (s, i)
            prev = want


def test_candidates_stable_topk():
    g = torch.Generator().manual_seed(2)
    x = torch.randn((3, 500), generator=g)
    x[0, 10:20] = 7.0                                    # ties
    x[1, 3], x[1, 4] = 0.0, -0.0
    ids, vals = dspark.candidates(x, 16, id0=1000)
    for r in range(3):
        order = sorted(range(500), key=lambda j: (-float(x[r, j]), j))[:16]
        assert (ids[r] - 1000).tolist() == order
        assert torch.equal(vals[r], x[r, order])


def test_attention_meta_is_vllms_noncausal_window():
    from engine.kernels.csa2 import attn, ref as CR

    P, n, ring = 300, 5, 256
    g = torch.Generator().manual_seed(8)
    q = (torch.randn((n, 32, 512), generator=g) * 0.5).to(torch.bfloat16).contiguous()
    sv = torch.zeros((ring, 576), dtype=torch.uint8)
    ssc = torch.zeros((ring, 8), dtype=torch.uint8)
    rows = {}
    for p in range(P - 140, P + n):
        v, s = CR.quantize_rows(torch.randn((1, 512), generator=g) * 2)
        sv[p % ring], ssc[p % ring] = v[0], s[0]
        rows[p] = CR.dequantize_rows(v, s)[0]
    sink = torch.randn(32, generator=g)
    inv = 10000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float64) / 64)
    ang = torch.arange(P + n + 8, dtype=torch.float64)[:, None] * inv[None, :]
    cs = torch.cat([ang.cos(), ang.sin()], 1).float().contiguous()
    tok, cnt, lo, hi = dspark.attention_meta(P, n, ring)
    out = torch.zeros((n, 32, 512), dtype=torch.bfloat16)
    attn.attention(q, (sv, ssc), tok, cnt, (sv, ssc), lo, torch.tensor([P], dtype=torch.int32), sink, cs, out,
                   attn.Scratch(n, "cpu"), ring=ring, hi=hi)
    keys = torch.stack([rows[p].double() for p in range(P - 128, P + n)])          # 128 context + the block
    for r in range(n):
        s = q[r].double() @ keys.T / math.sqrt(512)
        m = torch.maximum(s.max(1).values, sink.double())
        e = torch.exp(s - m[:, None])
        o = (e @ keys) / (e.sum(1) + torch.exp(sink.double() - m))[:, None]
        want = CR.rope(o, cs, torch.full((32,), P + r), inverse=True)
        torch.testing.assert_close(out[r].double(), want, rtol=2e-2, atol=2e-2)
    # the plain hi window (128 rows ending at the block's end) would miss the n oldest context rows
    assert int(cnt[0]) == 128 and int(lo[0]) == P and int(hi[0]) == P + n - 1
