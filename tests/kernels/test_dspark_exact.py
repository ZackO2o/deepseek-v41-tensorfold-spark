"""Drafted == serial (engine/kernels/dspark/verify.py), on the host with TensorFold's keyed sampler:

a toy target whose logits row is a function of its own context only (what the row-invariant kernels guarantee for
the real one) decodes serially, then speculatively with drafts from very different drafters (the serial future, an
always-wrong one, random ones at random depths, a perturbed copy of the target with the keyed noise): every reply is
the serial reply, token for token, at T = 0 and T > 0 (top_k / top_p / min_p), past 2,048 positions; and the keyed
draft noise makes a drafter whose distribution equals the target's accept everything at T > 0."""

from __future__ import annotations

import numpy as np
import pytest

from engine.kernels.dspark import verify
from engine.kernels.tf import tensorfold_src

if tensorfold_src() is None:
    pytest.skip("TensorFold 0.6.0 not found (TF_SRC)", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling, choose_rows  # noqa: E402

V = 64


class Target:
    def __init__(self, seed: int = 0):
        g = np.random.default_rng(seed)
        self.b = g.normal(size=(97, V)).astype(np.float32)
        self.e = g.normal(size=(V, V)).astype(np.float32) * 1.5
        self.f = g.normal(size=(V, V)).astype(np.float32) * 0.7

    def row(self, ctx: list[int]) -> np.ndarray:
        """Logits for the token after ``ctx`` (its own context only: row invariance)."""

        p = len(ctx)
        t1 = ctx[-1]
        t2 = ctx[-2] if p > 1 else 0
        return (self.b[p % 97] + self.e[t1] + self.f[t2]).astype(np.float32)


def _choose(rows: np.ndarray, positions, s: Sampling | None) -> list[int]:
    ids = np.broadcast_to(np.arange(V, dtype=np.int64), rows.shape).copy()
    if s is None:
        return [int(ids[r, np.lexsort((ids[r], -rows[r]))[0]]) for r in range(rows.shape[0])]
    return choose_rows(rows, ids, list(positions), s)


def serial(t: Target, prompt: list[int], n: int, s) -> list[int]:
    ctx = list(prompt)
    for _ in range(n):
        ctx.append(_choose(t.row(ctx)[None], [len(ctx)], s)[0])
    return ctx[len(prompt):]


def speculative(t: Target, prompt: list[int], n: int, s, drafter) -> tuple[list[int], int, int]:
    ctx = list(prompt)
    rounds = acc_total = 0
    while len(ctx) - len(prompt) < n:
        p = len(ctx) - 1                                 # the pending token's position
        drafts = drafter(ctx, p)
        window = [ctx[-1], *drafts]
        rows = np.stack([t.row(ctx[:-1] + window[:i + 1]) for i in range(len(window))])
        chosen = _choose(rows, verify.positions(p, len(drafts)), s)
        a, emit = verify.accept(window, chosen)
        ctx.extend(emit)
        rounds += 1
        acc_total += a
    return ctx[len(prompt):len(prompt) + n], rounds, acc_total


SAMPLINGS = [None, Sampling(seed=7, temperature=0.8, top_k=20, top_p=0.95),
             Sampling(seed=11, temperature=1.3, top_k=0, top_p=0.9, min_p=0.05),
             Sampling(seed=3, temperature=0.6, top_k=5, top_p=1.0)]


@pytest.mark.parametrize("si", range(len(SAMPLINGS)))
def test_any_drafter_gives_the_serial_reply(si):
    s = SAMPLINGS[si]
    t = Target(1)
    prompt = [1, 5, 9]
    n = 300
    want = serial(t, prompt, n, s)
    g = np.random.default_rng(si)
    future = prompt + want

    drafters = {
        "oracle": lambda ctx, p: future[p + 1:p + 6],
        "wrong": lambda ctx, p: [(future[p + 1 + i] + 1) % V if p + 1 + i < len(future) else 0 for i in range(4)],
        "random": lambda ctx, p: g.integers(0, V, size=int(g.integers(0, 8))).tolist(),
        "mixed": lambda ctx, p: (future[p + 1:p + 3] + g.integers(0, V, size=3).tolist()),
    }
    for name, dr in drafters.items():
        got, rounds, _ = speculative(t, prompt, n, s, dr)
        assert got == want, name
        if name == "oracle":
            assert rounds < n / 4                       # 5 drafts kept a round (+1)


def test_long_context_and_keyed_draft_noise():
    """Past 2,048 positions; a drafter sampling the target's own distribution with the target's keyed noise at the
    drafted positions is accepted every time, and the reply is still the serial one."""

    s = Sampling(seed=21, temperature=1.0, top_k=30, top_p=0.97)
    t = Target(2)
    prompt = [3, 4]
    n = 2100
    want = serial(t, prompt, n, s)

    def keyed(ctx, p):
        c = list(ctx)
        out = []
        for i in range(4):
            tok = _choose(t.row(c)[None], [p + 1 + i], s)[0]      # the target's distribution, the target's noise
            out.append(tok)
            c.append(tok)
        return out

    got, rounds, acc = speculative(t, prompt, n, s, keyed)
    assert got == want
    assert acc == 4 * rounds                            # every draft accepted

    def unkeyed(ctx, p):                                # the same distribution, other noise: drafts get rejected
        c = list(ctx)
        out = []
        for i in range(4):
            tok = _choose(t.row(c)[None], [p + 1 + i], Sampling(seed=999, temperature=1.0, top_k=30, top_p=0.97))[0]
            out.append(tok)
            c.append(tok)
        return out

    got2, rounds2, acc2 = speculative(t, prompt, 400, s, unkeyed)
    assert got2 == want[:400] and acc2 < 4 * rounds2
