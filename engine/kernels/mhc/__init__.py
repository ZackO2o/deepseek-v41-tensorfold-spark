"""Single-Pass mHC (V4.1's delayed pre-mix) on sm_121: the boundary kernels between sublayers. The math and the
arithmetic are in ``kernels.py``; ``ref.py`` is the torch emulation (bit for bit) and the float64 math.

Interface (all tensors CUDA, rows contiguous; R rows; D = 5,120; ``world`` = TP ranks of the gathered partials):

    Coefs(rows, device)                  pre / post fp32 [R, 4], comb fp32 [R, 16] (comb[i, j] at i * 4 + j): one site's
                                         coefficients, made by the site's finish, consumed by the boundary after it
    Scratch(rows, device)                partials [R, 4, NB, 32] fp32 (655 KB at R = 32) + the collapsed row [R, D] bf16

    site(x, hc, norm_w, out, coefs, scratch, pre_in=None)
        no post (model / DSpark entry, after Engram): the site's coefficients from the streams x [R, 4 D] bf16, the
        sublayer input out [R, D] bf16 = norm(collapse(x, pre_in)); pre_in None = stream 0
    boundary(x, xout, gathered, prev, hc, norm_w, out, coefs, scratch, tap=None)
        the single pass after a sublayer: xout = post(x, branch, prev.post, prev.comb) (x and xout may alias),
        out = norm(collapse(xout, prev.pre)), coefs = the next site's; ``tap`` [R, D] bf16 view (row stride any):
        the streams' mean (DSpark taps at the entries of layers 37-39)
    post_only(x, xout, gathered, prev, tap=None)       the post alone (before an Engram layer: Engram then ``site``)
    final(x, xout, gathered, prev, norm_w, out, scratch)
        the last post, then hidden = collapse(xout, prev.pre) (in ``scratch.c``, bf16, pre-norm: the DSpark
        confidence head reads it) and out = norm(hidden) with the model's final norm

    hc = Hc(fn fp32 [24, 4 D] contiguous, base fp32 [24], scale fp32 [3]); gathered fp32 [world, R, D] (rows
    contiguous, any rank stride): the sublayer's per-rank partial outputs, summed rank 0 first then rounded to bf16.

Per block: ``boundary`` x 2 (attention, FFN) and 2 finishes inside them: 4 launches a block (GLM's unfused: 6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton

from . import kernels as K

DIMS = 5120
EPS = 1e-20
HC_EPS = 1e-6
POST_ALPHA = 2.0
ITERS = 20
LAUNCH = {"enable_fp_fusion": False}     # every a * b + c is two roundings (the torch emulation's arithmetic)


@dataclass
class Hc:
    fn: torch.Tensor          # [24, 4 D] fp32
    base: torch.Tensor        # [24] fp32
    scale: torch.Tensor       # [3] fp32


class Coefs:
    def __init__(self, rows: int, device, *, pre=None, post=None, comb=None) -> None:
        z = lambda n: torch.zeros((rows, n), dtype=torch.float32, device=device)      # noqa: E731
        self.pre = z(4) if pre is None else pre
        self.post = z(4) if post is None else post
        self.comb = z(16) if comb is None else comb


class Scratch:
    def __init__(self, rows: int, device, dims: int = DIMS) -> None:
        self.rows, self.dims = rows, dims
        self.part = torch.zeros((rows, 4, K.NB, 32), dtype=torch.float32, device=device)
        self.c = torch.empty((rows, dims), dtype=torch.bfloat16, device=device)

    def nbytes(self) -> int:
        return self.part.numel() * 4 + self.c.numel() * 2


def scratch_bytes(rows: int, dims: int = DIMS) -> int:
    return rows * (4 * K.NB * 32 * 4 + dims * 2)


def _check(x: torch.Tensor, d: int, rows: int, scratch: Scratch | None) -> None:
    if x.dtype != torch.bfloat16 or x.shape[1] != 4 * d or x.stride(1) != 1 or d % K.NB or (d // K.NB) % K.BK:
        raise ValueError("mhc: streams bf16 [R, 4 D] unit-stride, D a multiple of NB * BK")
    if scratch is not None and (rows > scratch.rows or scratch.dims != d):
        raise ValueError("mhc: the scratch is too small")


def _launch(x, xout, gathered, prev: Coefs | None, pre_in, hc: Hc | None, scratch: Scratch | None, tap, rows: int,
            d: int, post_on: bool, collapse: int, mix: bool):
    if rows == 0:
        return
    world = gathered.shape[0] if post_on else 1
    if post_on and (gathered.dtype != torch.float32 or gathered.stride(2) != 1 or gathered.stride(1) != d):
        raise ValueError("mhc: gathered fp32 [world, R, D] with contiguous rows")
    if mix and (hc.fn.dtype != torch.float32 or not hc.fn.is_contiguous() or hc.fn.shape != (24, 4 * d)):
        raise ValueError("mhc: fn fp32 [24, 4 D] contiguous")
    dummy = x
    part = scratch.part if scratch is not None else dummy
    c = scratch.c if scratch is not None else dummy
    K._site[(triton.cdiv(rows, K.BM), K.NB)](
        x, x.stride(0), xout if xout is not None else dummy, gathered if post_on else dummy,
        gathered.stride(0) if post_on else 0, prev.post if post_on else dummy, prev.comb if post_on else dummy,
        pre_in if collapse == 2 else dummy, hc.fn if mix else dummy, part, c,
        tap if tap is not None else dummy, tap.stride(0) if tap is not None else 0, rows,
        D=d, NB=K.NB, BM=K.BM, BK=K.BK, WORLD=world, POST_ON=post_on, COLLAPSE=collapse, MIX=mix,
        TAP_ON=tap is not None, num_warps=K.WARPS, **LAUNCH)


def _finish(rows: int, d: int, scratch: Scratch, hc: Hc | None, coefs: Coefs | None, norm_w, out, coef: bool):
    if rows == 0:
        return
    dummy = scratch.part
    K._finish[(rows,)](scratch.part, hc.base if coef else dummy, hc.scale if coef else dummy,
                       coefs.pre if coef else dummy, coefs.post if coef else dummy, coefs.comb if coef else dummy,
                       scratch.c, norm_w, out, EPS, HC_EPS, POST_ALPHA, D=d, NB=K.NB, ITERS=ITERS, COEF=coef,
                       BLOCK=math.gcd(K.FIN_BLOCK, d), num_warps=4, **LAUNCH)


def site(x: torch.Tensor, hc: Hc, norm_w: torch.Tensor, out: torch.Tensor, coefs: Coefs, scratch: Scratch,
         pre_in: torch.Tensor | None = None) -> None:
    rows, d = x.shape[0], norm_w.shape[0]
    _check(x, d, rows, scratch)
    _launch(x, None, None, None, pre_in, hc, scratch, None, rows, d, False, 2 if pre_in is not None else 1, True)
    _finish(rows, d, scratch, hc, coefs, norm_w, out, True)


def boundary(x: torch.Tensor, xout: torch.Tensor, gathered: torch.Tensor, prev: Coefs, hc: Hc, norm_w: torch.Tensor,
             out: torch.Tensor, coefs: Coefs, scratch: Scratch, tap: torch.Tensor | None = None) -> None:
    rows, d = gathered.shape[1], norm_w.shape[0]
    _check(x, d, rows, scratch)
    if coefs is prev:
        raise ValueError("mhc.boundary: the next site's coefficients need their own buffers")
    _launch(x, xout, gathered, prev, prev.pre, hc, scratch, tap, rows, d, True, 2, True)
    _finish(rows, d, scratch, hc, coefs, norm_w, out, True)


def post_only(x: torch.Tensor, xout: torch.Tensor, gathered: torch.Tensor, prev: Coefs,
              tap: torch.Tensor | None = None) -> None:
    rows, d = gathered.shape[1], gathered.shape[2]
    _check(x, d, rows, None)
    _launch(x, xout, gathered, prev, None, None, None, tap, rows, d, True, 0, False)


def final(x: torch.Tensor, xout: torch.Tensor, gathered: torch.Tensor, prev: Coefs, norm_w: torch.Tensor,
          out: torch.Tensor, scratch: Scratch, tap: torch.Tensor | None = None) -> None:
    rows, d = gathered.shape[1], norm_w.shape[0]
    _check(x, d, rows, scratch)
    _launch(x, xout, gathered, prev, prev.pre, None, scratch, tap, rows, d, True, 2, False)
    _finish(rows, d, scratch, None, None, norm_w, out, False)


__all__ = ["Coefs", "Hc", "Scratch", "boundary", "final", "post_only", "scratch_bytes", "site"]
