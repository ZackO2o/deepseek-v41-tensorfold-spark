"""DeepSeek-V4.1-Flash's MoE router on sm_121: logits, sqrt-softplus scores, top-6 by score + bias (``noaux_tc``),
renormalized x 1.5, and the shared expert's slot, in one kernel (vLLM ``deepseek_v4`` routing semantics: scoring
``sqrtsoftplus``, ``norm_topk_prob``, ``routed_scaling_factor`` 1.5; the bias only chooses, it never weighs).

    logit_e = sum_k x_k W_e,k                      fp32 FMA chain over k ascending (``tl.dot`` ieee), x bf16, W fp16
    s_e     = sqrt(softplus(logit_e))              softplus(z) = z for z > 20 (torch's threshold), else log(1 + e^z)
    pick    = the 6 largest s_e + bias_e           ties to the lower expert id
    w_i     = 1.5 * s_pick_i / (s_pick_0 + ... + s_pick_5)    (the sum in pick order)
    slots   = [pick_0 .. pick_5, E]  weights [w_0 .. w_5, 1.0]   (E = the shared expert's table entry)

Every row's logits are its own FMA chains (a tile's other rows never enter them), and the selection reads only the
row: a window row gets the serial step's experts and weights. DSpark's blocks use the same kernel at E = 128, top-3.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

DIMS = 5120
EXPERTS = 384
TOPK = 6
SCALE = 1.5
BR = 16           # rows a program (tl.dot's minimum M)
BE = 512          # experts padded to a power of two
BK = 32           # k a step
WARPS = 8


@triton.jit
def _route(X, x_stride, WG, BIAS, PICK, WTS, R, D: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
           SLOTS: tl.constexpr, SCALE: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(0)
    rows = pb * 16 + tl.arange(0, 16)
    rok = rows < R
    e = tl.arange(0, BE)
    eok = e < E
    kk = tl.arange(0, BK)
    acc = tl.zeros((16, BE), tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + rows[:, None].to(tl.int64) * x_stride + (k0 + kk)[None, :], mask=rok[:, None],
                    other=0.0).to(tl.float32)
        w = tl.load(WG + e[None, :].to(tl.int64) * D + (k0 + kk)[:, None], mask=eok[None, :], other=0.0).to(
            tl.float32)
        acc = tl.dot(x, w, acc, input_precision="ieee")
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(tl.minimum(acc, 20.0))))
    s = tl.sqrt(sp)
    choice = tl.where(eok[None, :], s + tl.load(BIAS + e, mask=eok, other=0.0).to(tl.float32)[None, :],
                      float("-inf"))
    total = tl.zeros((16,), tl.float32)
    for i in tl.static_range(K):
        best = tl.max(choice, 1)
        idx = tl.min(tl.where(choice == best[:, None], e[None, :], BE), 1)          # ties: the lower id
        hit = e[None, :] == idx[:, None]
        sv = tl.sum(tl.where(hit, s, 0.0), 1)
        tl.store(PICK + rows * SLOTS + i, idx, mask=rok)
        tl.store(WTS + rows * SLOTS + i, sv, mask=rok)
        total = total + sv
        choice = tl.where(hit, float("-inf"), choice)
    for i in tl.static_range(K):
        sv = tl.load(WTS + rows * SLOTS + i, mask=rok, other=0.0)
        tl.store(WTS + rows * SLOTS + i, SCALE * (sv / total), mask=rok)
    if SLOTS > K:
        tl.store(PICK + rows * SLOTS + K, tl.full((16,), E, tl.int32), mask=rok)
        tl.store(WTS + rows * SLOTS + K, tl.full((16,), 1.0, tl.float32), mask=rok)


def route(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, topk: int = TOPK, shared: bool = True,
          scale: float = SCALE, pick: torch.Tensor | None = None, wts: torch.Tensor | None = None):
    """x bf16 [R, 5120], weight fp16 [E, 5120], bias fp16 [E] -> (pick int32 [R, topk + shared], wts fp32)."""

    R, D = x.shape
    E = weight.shape[0]
    slots = topk + int(shared)
    if weight.shape[1] != D or not weight.is_contiguous() or x.stride(1) != 1 or E > BE:
        raise ValueError("route: weight [E <= 512, D] contiguous, x unit-stride")
    if pick is None:
        pick = torch.empty((R, slots), dtype=torch.int32, device=x.device)
    if wts is None:
        wts = torch.empty((R, slots), dtype=torch.float32, device=x.device)
    _route[(triton.cdiv(R, 16),)](x, x.stride(0), weight, bias, pick, wts, R, D=D, E=E, K=topk, SLOTS=slots,
                                  SCALE=scale, BE=BE, BK=BK, num_warps=WARPS)
    return pick, wts


def reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, topk: int = TOPK, scale: float = SCALE):
    """Float64 routing: (pick [R, topk] by (score + bias desc, id asc), weights [R, topk])."""

    logit = x.double() @ weight.double().T
    s = torch.nn.functional.softplus(logit).sqrt()
    choice = s + bias.double()
    picks, ws = [], []
    for r in range(x.shape[0]):
        order = sorted(range(weight.shape[0]), key=lambda e: (-float(choice[r, e]), e))[:topk]
        sv = s[r, order]
        picks.append(order)
        ws.append((scale * sv / sv.sum()).tolist())
    return torch.tensor(picks, dtype=torch.int32), torch.tensor(ws, dtype=torch.float64)
