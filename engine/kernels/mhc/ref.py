"""Torch references of the mHC kernels (CPU).

- ``boundary`` / ``finish_norm``: the kernels' arithmetic emulated operation for operand (fp32, one rounding an
  operation; the FMA chains of ``tl.dot`` ieee through ``fma32``): the streams, collapse, taps, partials and normed
  input are compared bit for bit;
- ``coefficients``: pre / post / comb from the partials in torch fp32 (the transcendental functions differ from the
  GPU's approximations: tolerance);
- ``math``: float64 of the whole site (engine/reference/hc.py's formulas), for tolerance checks.
"""

from __future__ import annotations

import numpy as np
import torch

F32 = torch.float32


def fma32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """fp32 fma(a, b, c) rounded once (nearest even): the exact product in float64, the float64 sum rounded to odd,
    then to fp32 (no double rounding)."""

    p = a.astype(np.float64) * b.astype(np.float64)
    c = np.broadcast_to(c.astype(np.float64), p.shape)
    s = p + c
    bb = s - p
    err = (p - (s - bb)) + (c - bb)
    even = (s.view(np.int64) & 1) == 0
    fix = (err != 0) & even & np.isfinite(s)
    if fix.any():
        s = np.where(fix, np.nextafter(s, np.where(err > 0, np.inf, -np.inf)), s)
    return s.astype(np.float32)


def chain(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """acc[m, n] = fma(v[m, k], w[k, n], acc) for k ascending from 0: v [M, K], w [K, N] fp32."""

    acc = np.zeros((v.shape[0], w.shape[1]), dtype=np.float32)
    for k in range(v.shape[1]):
        acc = fma32(v[:, k:k + 1], w[k:k + 1, :], acc)
    return acc


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(F32)


def boundary(x: torch.Tensor, gathered: torch.Tensor | None, post: torch.Tensor | None, comb: torch.Tensor | None,
             pre: torch.Tensor | None, collapse: int, fn: torch.Tensor | None, nb: int) -> dict:
    """x [R, 4 D] bf16 -> {'x': new streams bf16, 'c': collapsed bf16, 'tap': bf16, 'part': [R, 4, nb, 32] fp32 with
    [.., 24] square sums and [:, 0, :, 25] the collapsed square sums}."""

    R = x.shape[0]
    d = x.shape[1] // 4
    xs = [x[:, j * d:(j + 1) * d].to(F32) for j in range(4)]
    out = {}
    if gathered is not None:
        g = gathered[0].to(F32)
        for w in range(1, gathered.shape[0]):
            g = g + gathered[w].to(F32)
        br = _bf(g)
        cm = comb.view(R, 4, 4)
        ns = []
        for j in range(4):
            mixed = ((cm[:, 0, j:j + 1] * xs[0] + cm[:, 1, j:j + 1] * xs[1]) + cm[:, 2, j:j + 1] * xs[2]) + \
                cm[:, 3, j:j + 1] * xs[3]
            ns.append(_bf(post[:, j:j + 1] * br + mixed))
        xs = ns
        out["x"] = torch.cat(xs, 1).to(torch.bfloat16)
    if collapse == 2:
        c = _bf(((pre[:, 0:1] * xs[0] + pre[:, 1:2] * xs[1]) + pre[:, 2:3] * xs[2]) + pre[:, 3:4] * xs[3])
    elif collapse == 1:
        c = xs[0].clone()
    else:
        c = None
    if c is not None:
        out["c"] = c.to(torch.bfloat16)
    out["tap"] = _bf((((xs[0] + xs[1]) + xs[2]) + xs[3]) * 0.25).to(torch.bfloat16)
    cb = d // nb
    part = np.zeros((R, 4, nb, 32), dtype=np.float32)
    ones = np.ones((cb, 1), dtype=np.float32)
    for b in range(nb):
        cols = slice(b * cb, (b + 1) * cb)
        if fn is not None:
            for j in range(4):
                v = xs[j][:, cols].numpy()
                w = fn[:, j * d + b * cb: j * d + (b + 1) * cb].to(F32).numpy().T
                part[:, j, b, :24] = chain(v, w)
                part[:, j, b, 24] = chain(v * v, ones)[:, 0]
        if c is not None:
            cv = c[:, cols].numpy()
            part[:, 0, b, 25] = chain(cv * cv, ones)[:, 0]
    out["part"] = torch.from_numpy(part)
    return out


def mix_sums(part: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mixes [R, 24], stream square sum [R], collapsed square sum [R]) summed in the finish's order."""

    R, _, nb, _ = part.shape
    p = part.reshape(R, 4 * nb, 32)
    mix = torch.zeros((R, 32), dtype=F32)
    ss = torch.zeros((R,), dtype=F32)
    for j in range(4 * nb):
        mix = mix + p[:, j]
        ss = ss + p[:, j, 24]
    ssc = torch.zeros((R,), dtype=F32)
    for b in range(nb):
        ssc = ssc + part[:, 0, b, 25]
    return mix[:, :24], ss, ssc


def finish_norm(part: torch.Tensor, c: torch.Tensor, norm_w: torch.Tensor, eps: float) -> torch.Tensor:
    """The normed sublayer input, bit for bit: bf16((c * (1 / sqrt(ssc / D + eps))) * w)."""

    d = c.shape[1]
    _, _, ssc = mix_sums(part)
    rn = torch.tensor(1.0, dtype=F32) / torch.sqrt(ssc / torch.tensor(float(d), dtype=F32) +
                                                   torch.tensor(eps, dtype=F32))
    return ((c.to(F32) * rn[:, None]) * norm_w.to(F32)).to(torch.bfloat16)


def coefficients(part: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, d: int, eps: float, hc_eps: float,
                 post_alpha: float, iters: int):
    """pre, post [R, 4], comb [R, 4, 4] from the partials (torch fp32; tolerance against the kernel)."""

    mix, ss, _ = mix_sums(part)
    R = mix.shape[0]
    rinv = 1.0 / torch.sqrt(ss / (4.0 * d) + eps)
    mix = mix * rinv[:, None]
    pre = torch.sigmoid(mix[:, :4] * scale[0] + base[:4]) + hc_eps
    post = torch.sigmoid(mix[:, 4:8] * scale[1] + base[4:8]) * post_alpha
    cl = mix[:, 8:].view(R, 4, 4) * scale[2] + base[8:].view(1, 4, 4)
    comb = torch.softmax(cl, -1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


def math(x: torch.Tensor, gathered: torch.Tensor | None, post: torch.Tensor | None, comb: torch.Tensor | None,
         pre: torch.Tensor | None, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, norm_w: torch.Tensor,
         eps: float = 1e-20, hc_eps: float = 1e-6, post_alpha: float = 2.0, iters: int = 20) -> dict:
    """Float64 site (bf16 storage boundaries kept): streams, the next site's pre / post / comb, the normed input."""

    R = x.shape[0]
    d = x.shape[1] // 4
    s = x.double().view(R, 4, d)
    if gathered is not None:
        br = gathered.double().sum(0).to(torch.bfloat16).double()
        s = torch.einsum("rij,rid->rjd", comb.double().view(R, 4, 4), s) + post.double()[:, :, None] * br[:, None]
        s = s.to(torch.bfloat16).double()
    flat = s.reshape(R, -1)
    mix = (flat @ fn.double().T) / torch.sqrt((flat * flat).mean(-1, keepdim=True) + eps)
    sc, bs = scale.double(), base.double()
    npre = torch.sigmoid(mix[:, :4] * sc[0] + bs[:4]) + hc_eps
    npost = torch.sigmoid(mix[:, 4:8] * sc[1] + bs[4:8]) * post_alpha
    cm = torch.softmax(mix[:, 8:].view(R, 4, 4) * sc[2] + bs[8:].view(1, 4, 4), -1) + hc_eps
    cm = cm / (cm.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        cm = cm / (cm.sum(-1, keepdim=True) + hc_eps)
        cm = cm / (cm.sum(-2, keepdim=True) + hc_eps)
    c = s[:, 0] if pre is None else (pre.double()[:, :, None] * s).sum(1)
    c = c.to(torch.bfloat16).double()
    xin = c / torch.sqrt((c * c).mean(-1, keepdim=True) + eps) * norm_w.double()
    return {"x": s.reshape(R, -1), "pre": npre, "post": npost, "comb": cm, "c": c, "input": xin}
