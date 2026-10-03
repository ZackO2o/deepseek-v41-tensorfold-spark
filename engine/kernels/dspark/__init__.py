"""DSpark (checkpoint ``mtp.0-2``) for DeepSeek-V4.1-Flash on sm_121: the glue of the 3 draft blocks, the Markov
head, the confidence head and keyed draft noise. Verification is exact whatever this module drafts (``verify.py``).

A drafting pass for a slot whose pending token sits at position P (vLLM ``nvidia/dspark.py``, ``spec_decode/dspark``;
engine/reference/model.py: DSpark):

1. taps: ``mhc.boundary(..., tap=taps[:, i * D:(i + 1) * D])`` at the boundaries entering layers 37, 38, 39 (the
   streams' mean); ``main_x = bf16(rms_norm(main_proj(taps), main_norm))`` (EXL3 linear + ``rms_norm``) for the
   committed rows of the verify;
2. context rows: each block b stores ``rope(kv_norm(wkv_b(main_x)))`` of the committed positions into its own ring
   (``csa2.compress.kv_store`` ratio 0, ``ring`` rows): the ring then holds positions P - 128 .. P - 1;
3. the block input: ``block_ids`` = [anchor, noise x (N - 1)] at positions P .. P + N - 1, embedded and copied to 4
   streams; the 3 blocks run like backbone blocks (``mhc``, attention, the 128-expert top-3 MoE), their attention
   through ``csa2.attn.attention(q, comp=ring, tokens, counts, swa=ring, lo, pos, ..., hi=hi)`` with
   ``attention_meta``: every draft row sees the 128 context rows (as the compressed list, ring slots in position
   order) plus all N block rows (the SWA chunk with lo = P, hi = P + N - 1): vLLM's non-causal DSpark window
   (``max(P - 128, 0) .. P + N - 1``, 128 + N keys; the plain ``hi`` window would see only 128 - N context rows);
4. the head: ``mhc.final`` (hidden = collapse with the last pre-mix, in scratch.c; normed), ``head`` (vocabulary
   halves), ``candidates`` (a rank's best K by (value desc, id asc)), gathered across ranks, then ``chain``.

Interface:

    block_ids(anchor int32 [S], noise, n) -> ids int32 [S, n]; positions(P int32 [S], n) -> int32 [S, n]
    attention_meta(P: int, n: int, ring: int, device) -> (tokens int32 [n, 128], counts int32 [n], lo int32 [n],
        hi int32 [n])                         the csa2.attn arguments of one slot's draft rows (ring slots)
    candidates(logits fp32 [M, V_rank], k, id0) -> (ids int32 [M, k], values fp32 [M, k])
    chain(cand int32 [S, N, C], cval fp32 [S, N, C] (overwritten with the draft logits z),
          w1 bf16 [V, 256], w2 bf16 [V, 256], anchor int32 [S],
          ipar int64 [S, 4] (seed, POS0 = P + 1, top_k, sampled), fpar fp32 [S, 3] (T, top_p, min_p),
          hid bf16 [S, N, D] | None, cw fp32 [D + 256] | None, draft int32 [S, N], conf fp32 [S, N] | None)
    sampling_params(sampling, P) -> (ipar row, fpar row)       from ``exact_sampling.Sampling`` (None = greedy)
"""

from __future__ import annotations

import math

import torch

from . import kernels as K

BLOCK = 5
NOISE = 128799
WINDOW = 128


def block_ids(anchor: torch.Tensor, noise: int = NOISE, n: int = BLOCK) -> torch.Tensor:
    ids = torch.full((anchor.shape[0], n), noise, dtype=torch.int32, device=anchor.device)
    ids[:, 0] = anchor
    return ids


def positions(p: torch.Tensor, n: int = BLOCK) -> torch.Tensor:
    return (p.to(torch.int32)[:, None] + torch.arange(n, dtype=torch.int32, device=p.device)[None, :]).contiguous()


def attention_meta(p: int, n: int, ring: int, device="cpu", window: int = WINDOW):
    """One slot's draft rows at P .. P + n - 1: the context positions max(P - window, 0) .. P - 1 as the
    "compressed" list (ring slots, ascending positions), the block itself as the SWA window [P, P + n - 1]."""

    if ring < window + n or ring & (ring - 1):
        raise ValueError("dspark: the ring is a power of two holding the window and the block")
    a = max(p - window, 0)
    ctx = torch.arange(a, p, dtype=torch.int64) % ring
    tokens = torch.full((n, window), -1, dtype=torch.int32)
    tokens[:, :ctx.numel()] = ctx.to(torch.int32)
    counts = torch.full((n,), ctx.numel(), dtype=torch.int32)
    lo = torch.full((n,), p, dtype=torch.int32)
    hi = torch.full((n,), p + n - 1, dtype=torch.int32)
    return tokens.to(device), counts.to(device), lo.to(device), hi.to(device)


def candidates(logits: torch.Tensor, k: int, id0: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A rank's best ``k`` of each row by (value desc, id asc): unique keys (``csa2.index.sort_keys``), so the set
    does not depend on the top-k implementation."""

    from ..csa2 import index

    keys = index.sort_keys(logits.float().contiguous())
    top = torch.topk(keys, k, dim=1).values
    col = (0x7FFFFFFF - (top & 0xFFFFFFFF)).to(torch.int64)
    vals = torch.gather(logits.float(), 1, col)
    return (col + id0).to(torch.int32), vals


def sampling_params(sampling, p: int) -> tuple[list[int], list[float]]:
    """(seed, POS0, top_k, sampled), (T, top_p, min_p) for a slot whose pending token is at position ``p``: the
    first draft is the token at p + 1, keyed at that position like the target's choice for it."""

    if sampling is None or float(sampling.temperature) <= 0.0:
        return [0, p + 1, 0, 0], [1.0, 1.0, 0.0]
    return ([int(sampling.seed) & ((1 << 63) - 1), p + 1, int(sampling.top_k), 1],
            [float(sampling.temperature), float(sampling.top_p), float(sampling.min_p)])


def chain(cand: torch.Tensor, cval: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, anchor: torch.Tensor,
          ipar: torch.Tensor, fpar: torch.Tensor, draft: torch.Tensor, *, hid: torch.Tensor | None = None,
          cw: torch.Tensor | None = None, conf: torch.Tensor | None = None,
          scr: torch.Tensor | None = None) -> torch.Tensor:
    """``cval`` is overwritten with the biased draft logits; ``scr`` fp64 [S, 2, 128] (allocated once for graphs)."""

    S, N, C = cand.shape
    rank = w1.shape[1]
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("dspark.chain: the Markov head's tables are bf16")
    if C > K.KP or not cand.is_contiguous() or not cval.is_contiguous() or rank & (rank - 1) or w2.shape[1] != rank:
        raise ValueError("dspark.chain: at most 128 contiguous candidates a position, a power-of-two Markov rank")
    has_conf = conf is not None
    d = hid.shape[2] if has_conf else 1024
    if has_conf and (cw.shape[0] != d + rank or not hid.is_contiguous() or d % 64):
        raise ValueError("dspark.chain: hid [S, N, D] contiguous, cw [D + rank]")
    if scr is None:
        scr = torch.empty((S, 2, K.KP), dtype=torch.float64, device=cand.device)
    if S:
        K._chain[(S,)](cand, cval, cand.stride(1), C, w1, w2, hid if has_conf else cval, cw if has_conf else cval,
                       anchor, ipar, fpar, draft, conf if has_conf else cval, scr, D=d, N=N, KP=K.KP, CH=K.CH,
                       R=rank, RB=max(16, min(K.RB, rank)), HB=math.gcd(d, 1024), HAS_CONF=has_conf, num_warps=K.WARPS)
    return draft


__all__ = ["attention_meta", "block_ids", "candidates", "chain", "positions", "sampling_params"]
