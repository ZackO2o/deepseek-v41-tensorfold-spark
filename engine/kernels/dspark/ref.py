"""Torch / numpy references of DSpark's sequential head (CPU): the Markov chain over candidates with the keyed choice
of ``tensorfold.engine.exact_sampling.choose`` (the host's sampler, float64), and the confidence head. The kernel is
compared with it token for token (its bias sums and float64 log / exp may differ in the last bits: a token can only
differ at a near-tie, which the tests' data avoid), the confidence within tolerance."""

from __future__ import annotations

import numpy as np
import torch


def _choose(z: np.ndarray, ids: np.ndarray, position: int, ipar, fpar) -> int:
    seed, _, top_k, sampled = (int(v) for v in ipar)
    if not sampled:
        order = np.lexsort((ids, -z))
        return int(ids[order[0]])
    from engine.kernels.tf import tensorfold_src

    tensorfold_src()
    from tensorfold.engine.exact_sampling import Sampling, choose

    s = Sampling(seed=seed, temperature=float(fpar[0]), top_k=top_k, top_p=float(fpar[1]), min_p=float(fpar[2]))
    return choose(z.astype(np.float32), ids.astype(np.int64), position, s)


def chain(cand: torch.Tensor, cval: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, anchor: torch.Tensor,
          ipar: torch.Tensor, fpar: torch.Tensor, hid: torch.Tensor | None = None, cw: torch.Tensor | None = None):
    """(drafts int64 [S, N], confidence fp32 [S, N] or None)."""

    S, N, _ = cand.shape
    drafts = torch.zeros((S, N), dtype=torch.int64)
    conf = torch.zeros((S, N), dtype=torch.float32) if hid is not None else None
    for s in range(S):
        prev = int(anchor[s])
        for i in range(N):
            ids = cand[s, i].to(torch.int64)
            live = ids >= 0
            ids = ids[live]
            me = w1[prev].float()
            z = cval[s, i][live].float() + w2[ids].float() @ me
            tok = _choose(z.numpy(), ids.numpy(), int(ipar[s, 1]) + i, ipar[s].tolist(), fpar[s].tolist())
            drafts[s, i] = tok
            if conf is not None:
                feats = torch.cat([hid[s, i].float(), me])
                conf[s, i] = torch.sigmoid(feats @ cw.float())
            prev = tok
    return drafts, conf


def markov_greedy(base: torch.Tensor, hidden: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                  conf_w: torch.Tensor | None, anchor: int):
    """engine/reference/model.py's DSpark Markov loop over the full vocabulary (greedy), for one slot."""

    prev, toks, embeds = anchor, [], []
    for i in range(base.shape[0]):
        me = w1[prev].float()
        li = base[i].float() + me @ w2.float().t()
        tok = int(li.argmax())
        toks.append(tok)
        embeds.append(me)
        prev = tok
    conf = None
    if conf_w is not None:
        feats = torch.cat([hidden.float(), torch.stack(embeds)], dim=-1)
        conf = torch.sigmoid(feats @ conf_w.float().t()).squeeze(-1)
    return torch.tensor(toks), conf
