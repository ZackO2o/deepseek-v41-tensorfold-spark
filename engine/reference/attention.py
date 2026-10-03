"""CSA2 attention of DeepSeek-V4.1-Flash: one 512-wide K=V latent head, 64 query heads, attention sinks, a 128-token
sliding window on every layer plus (ratio > 0) top-k selected rows of a compressed KV cache shared by a KV-source
group, an FP8 lightning indexer, and the two-level candidate filter.

Semantics follow vLLM ``deepseek_v4_1/attention.py``, ``compressor.py``, ``common/ops/*`` and
``kernels/attention/dsa/candidate_blocks.py`` (Apache-2.0). The reference runs one sequence from position 0 with all
of its tokens at once (a full prefill); a chunked prefill or a decode step of vLLM produces the same per-row values.

Per layer L, with x = the hc-collapsed, attn_norm'd input [T, 5120]:

1. ``qr = q_norm(wq_a x)`` [T, 1280], ``kv = kv_norm(wkv x)`` [T, 512] (bf16 each);
2. ``q = rope(wq_b qr)`` [T, 64, 512]: RoPE on the last 64 dims, no per-head norm (V4.1: vLLM's fused q-norm /
   RoPE / KV insert runs with apply_q_norm=False since qr is normed before wq_b; V4.0 normed each head; G2);
3. SWA key/value row of each token = ``rope(kv)`` at its position (fp8_ds_mla in the kit's cache);
4. KV source only: the compressor pools ``ratio`` tokens (ratio 2: per-dim softmax gate over the pair, no APE;
   ratio 1: the token) of ``wkv_c x`` with gate ``wgate_c x``, RMSNorms the result (the "latent"), and publishes
   ``rope(latent)`` at the group's first position as a compressed row; its indexer publishes
   ``rope(k_norm(wk latent))`` [128] as the group's index key;
5. index source only: ``score[t, s] = sum_h w[t, h] * relu(qi[t, h] . k[s])`` with ``qi = rope(wq_b_i qr)``
   [T, 32, 128], ``w = weights_proj(x) / sqrt(128) / sqrt(32)``, over the closed groups s < (pos + 1) // ratio; the
   candidate source keeps the 2,048 best blocks of 8 positions (block score = max, newest block pinned); later index
   sources mask to them; top-512 (all of them when fewer are valid);
6. softmax over (window rows U selected compressed rows) with a per-head sink logit, scale 1 / sqrt(512), V = K;
7. inverse RoPE on the output's last 64 dims, ``wo_a`` per group of 8 heads (4,096 -> 1,024), ``wo_b`` (8,192 ->
   5,120).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .config import Config
from .ops import (F32, DenseLinear, Linear, Numerics, bf16, fp8_ds_mla_roundtrip, fp8_token_roundtrip, linear_out,
                  rms_norm)
from .rope import Rope, rope_for


@dataclass
class CompressorWeights:
    wkv: Linear
    wgate: Linear | None          # ratio 1 has no gate
    norm: torch.Tensor


@dataclass
class IndexerWeights:
    wq_b: Linear                  # 1280 -> 32 * 128
    weights_proj: Linear          # 5120 -> 32 (native, bf16 in vLLM)
    wk: Linear | None = None      # KV sources only: 512 -> 128 (EXL3 K8)
    k_norm: torch.Tensor | None = None


@dataclass
class AttnWeights:
    wq_a: Linear
    wkv: Linear
    wq_b: Linear
    wo_a: list[Linear]            # one per o_group
    wo_b: Linear
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    attn_sink: torch.Tensor       # [heads] fp32
    compressor: CompressorWeights | None = None
    indexer: IndexerWeights | None = None


@dataclass
class CsaState:
    """What CSA2 layers publish for the layers above them within one forward."""

    ckv: dict[int, torch.Tensor] = field(default_factory=dict)          # kv source -> [C, 512] compressed rows
    ikey: dict[int, torch.Tensor] = field(default_factory=dict)         # kv source -> [C, 128] index keys
    topk: torch.Tensor | None = None                                    # [T, topk] local compressed ids, -1 pad
    candidates: torch.Tensor | None = None                              # [T, n_blocks] bool
    taps: dict[str, torch.Tensor] = field(default_factory=dict)         # debugging: named intermediates


def attend(q: torch.Tensor, keys: torch.Tensor, visible: torch.Tensor, sink: torch.Tensor,
           chunk: int = 64) -> torch.Tensor:
    """Sink softmax attention, V = K. q [T, H, D], keys [S, D], visible [T, S] bool, sink [H] -> [T, H, D] fp32."""

    t_total, h, d = q.shape
    scale = 1.0 / math.sqrt(d)
    out = torch.empty((t_total, h, d), dtype=F32, device=q.device)
    keys = keys.to(F32)
    for a in range(0, t_total, chunk):
        b = min(t_total, a + chunk)
        cols = visible[a:b].any(0).nonzero().squeeze(-1)
        k = keys.index_select(0, cols)
        s = torch.einsum("thd,sd->ths", q[a:b].to(F32), k) * scale
        s = s.masked_fill(~visible[a:b].index_select(1, cols)[:, None, :], float("-inf"))
        m = torch.maximum(s.amax(-1), sink.to(F32)[None, :])
        e = torch.exp(s - m[..., None])
        den = e.sum(-1) + torch.exp(sink.to(F32)[None, :] - m)
        out[a:b] = torch.einsum("ths,sd->thd", e / den[..., None], k)
    return out


def window_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, window: int) -> torch.Tensor:
    """[Tq, Tk]: key position in (q - window, q]."""

    d = q_pos[:, None] - k_pos[None, :]
    return (d >= 0) & (d < window)


def select_candidate_blocks(logits: torch.Tensor, valid: torch.Tensor, block: int, topk_blocks: int) -> torch.Tensor:
    """Candidate blocks [T, nblocks] bool: the best ``topk_blocks`` blocks by max score, the newest block pinned."""

    t, width = logits.shape
    nblocks = (width + block - 1) // block
    pad = nblocks * block - width
    lv = logits.masked_fill(~valid, float("-inf"))
    if pad:
        lv = torch.cat([lv, lv.new_full((t, pad), float("-inf"))], 1)
    scores = lv.view(t, nblocks, block).amax(-1)
    n_valid = valid.sum(-1)
    newest = ((n_valid - 1).clamp(min=0) // block)
    rows = torch.arange(t, device=logits.device)
    has = n_valid > 0
    scores[rows[has], newest[has]] = float("inf")
    k = min(topk_blocks, nblocks)
    values, indices = stable_topk(scores, k)
    keep = torch.zeros((t, nblocks), dtype=torch.bool, device=logits.device)
    keep.scatter_(1, indices, values > float("-inf"))
    return keep


def stable_topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k along the last dim with a fixed tie order (lower index first), so a row's selection never depends on
    how wide the batch is. vLLM leaves ties to its kernels' order; exact ties need equal fp32 scores (e.g. all
    heads' ReLU at 0), which real prompts essentially never produce."""

    values, indices = torch.sort(x, dim=-1, descending=True, stable=True)
    return values[..., :k], indices[..., :k]


class Attention:
    def __init__(self, cfg: Config, layer: int, w: AttnWeights, numerics: Numerics = Numerics()):
        self.cfg, self.layer, self.w, self.num = cfg, layer, w, numerics
        self.ratio = cfg.compress_ratio(layer)
        self.mode = cfg.attention_mode(layer)
        self.kv_source = cfg.kv_source(layer)
        self.rope: Rope = rope_for(cfg, self.ratio)
        self.eps = cfg.rms_norm_eps
        if self.mode == "full":
            assert w.compressor is not None and w.indexer is not None and w.indexer.wk is not None
        if self.mode == "reindex":
            assert w.indexer is not None

    # -- pieces shared with the DSpark blocks -------------------------------------------------------------------

    def project(self, x: torch.Tensor, positions: torch.Tensor):
        """(q [T, H, D], rope'd kv rows [T, D], qr [T, q_lora]) for the attention input x [T, hidden]."""

        cfg, w = self.cfg, self.w
        qr = bf16(rms_norm(linear_out(w.wq_a, x), w.q_norm, self.eps))
        kv = bf16(rms_norm(linear_out(w.wkv, x), w.kv_norm, self.eps))
        q = linear_out(w.wq_b, qr).view(x.shape[0], cfg.num_attention_heads, cfg.head_dim)
        q = bf16(self.rope.apply(q, positions))          # no per-head RMS in V4.1 (apply_q_norm=False)
        k = self.kv_rows(kv, positions)
        return q, k, qr

    def kv_rows(self, kv_normed: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        k = bf16(self.rope.apply(kv_normed, positions))
        return fp8_ds_mla_roundtrip(k, self.cfg.qk_rope_head_dim) if self.num.kv_fp8_ds_mla else k

    def output(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w
        o = bf16(self.rope.apply(bf16(o), positions, inverse=True))
        t = o.shape[0]
        g = cfg.o_groups
        og = o.reshape(t, g, (cfg.num_attention_heads // g) * cfg.head_dim)
        z = torch.cat([linear_out(w.wo_a[i], og[:, i]) for i in range(g)], dim=-1)
        return linear_out(w.wo_b, z)

    # -- the compressor and the indexer -----------------------------------------------------------------------

    def compress(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(latent [C, 512] bf16 values, group first positions [C]) for the closed groups of the sequence."""

        c = self.w.compressor
        assert c is not None
        r = self.ratio
        kv = linear_out(c.wkv, x)
        t = x.shape[0]
        n = t // r
        if r == 1:
            pooled = kv
        else:
            assert c.wgate is not None
            score = linear_out(c.wgate, x)
            kv2 = kv[: n * r].view(n, r, -1)
            sc2 = score[: n * r].view(n, r, -1)
            pooled = (kv2 * torch.softmax(sc2, dim=1)).sum(1)
        latent = bf16(rms_norm(pooled, c.norm, self.eps))
        return latent, positions[: n * r].view(n, r)[:, 0]

    def index_keys(self, latent: torch.Tensor, group_pos: torch.Tensor) -> torch.Tensor:
        ix = self.w.indexer
        assert ix is not None and ix.wk is not None and ix.k_norm is not None
        k = bf16(rms_norm(linear_out(ix.wk, latent), ix.k_norm, self.eps))
        k = bf16(self.rope.apply(k, group_pos))
        return fp8_token_roundtrip(k) if self.num.indexer_fp8 else k

    def index_scores(self, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor,
                     keys: torch.Tensor) -> torch.Tensor:
        cfg, ix = self.cfg, self.w.indexer
        assert ix is not None
        t = x.shape[0]
        q = linear_out(ix.wq_b, qr).view(t, cfg.index_n_heads, cfg.index_head_dim)
        q = bf16(self.rope.apply(q, positions))
        if self.num.indexer_fp8:
            q = fp8_token_roundtrip(q)
        w = linear_out(ix.weights_proj, x) * (cfg.index_head_dim ** -0.5) * (cfg.index_n_heads ** -0.5)
        dots = torch.relu(torch.einsum("thd,sd->ths", q, keys.to(F32)))
        return torch.einsum("ths,th->ts", dots, w)

    def select(self, logits: torch.Tensor, positions: torch.Tensor, state: CsaState) -> torch.Tensor:
        cfg = self.cfg
        t, width = logits.shape
        n_valid = (positions + 1) // self.ratio
        valid = torch.arange(width, device=logits.device)[None, :] < n_valid[:, None]
        short = (int(positions.max()) + 1) // self.ratio <= cfg.index_topk     # vLLM's whole-batch short path
        if not short and cfg.is_candidate_source(self.layer):
            state.candidates = select_candidate_blocks(logits, valid, cfg.candidate_block_size,
                                                       cfg.candidate_topk_blocks)
        lv = logits.masked_fill(~valid, float("-inf"))
        if not short and cfg.uses_candidates(self.layer) and state.candidates is not None:
            blk = torch.arange(width, device=logits.device) // cfg.candidate_block_size
            keep = state.candidates[:, blk[blk < state.candidates.shape[1]]]
            lv = lv.masked_fill(~keep, float("-inf"))
        k = min(cfg.index_topk, width)
        sel = torch.full((t, cfg.index_topk), -1, dtype=torch.long, device=logits.device)
        if k > 0:
            values, indices = stable_topk(lv, k)
            sel[:, :k] = torch.where(values > float("-inf"), indices, -1)
        return sel

    # -- the layer --------------------------------------------------------------------------------------------

    def __call__(self, x: torch.Tensor, positions: torch.Tensor, state: CsaState) -> torch.Tensor:
        cfg = self.cfg
        q, k_swa, qr = self.project(x, positions)
        visible = window_mask(positions, positions, cfg.sliding_window)
        keys = k_swa
        if self.ratio > 0:
            if self.mode == "full":
                latent, gpos = self.compress(x, positions)
                ckv = bf16(self.rope.apply(latent, gpos))
                state.ckv[self.layer] = fp8_ds_mla_roundtrip(ckv, cfg.qk_rope_head_dim) if self.num.kv_fp8_ds_mla \
                    else ckv
                state.ikey[self.layer] = self.index_keys(latent, gpos)
            if self.mode in ("full", "reindex"):
                logits = self.index_scores(x, qr, positions, state.ikey[self.kv_source])
                state.topk = self.select(logits, positions, state)
            ckv = state.ckv[self.kv_source]
            assert state.topk is not None
            sel = torch.zeros((x.shape[0], ckv.shape[0] + 1), dtype=torch.bool, device=x.device)
            sel.scatter_(1, torch.where(state.topk >= 0, state.topk, ckv.shape[0]), True)
            visible = torch.cat([visible, sel[:, :-1]], dim=1)
            keys = torch.cat([k_swa, ckv], dim=0)
        o = attend(q, keys, visible, self.w.attn_sink)
        return self.output(o, positions)


def dense_attn_weights(cfg: Config, layer: int, gen: torch.Generator, std: float = 0.05) -> AttnWeights:
    """Random dense weights of the right shapes (tests)."""

    def lin(i: int, o: int) -> DenseLinear:
        return DenseLinear(bf16(torch.randn(o, i, generator=gen) * std))

    def norm(n: int) -> torch.Tensor:
        return bf16(1.0 + 0.1 * torch.randn(n, generator=gen))

    h, d = cfg.num_attention_heads, cfg.head_dim
    ratio = cfg.compress_ratio(layer)
    comp = ix = None
    if cfg.is_kv_source(layer):
        comp = CompressorWeights(lin(cfg.hidden_size, d), lin(cfg.hidden_size, d) if ratio > 1 else None, norm(d))
    if cfg.is_index_source(layer):
        own = cfg.is_kv_source(layer)
        ix = IndexerWeights(lin(cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim),
                            lin(cfg.hidden_size, cfg.index_n_heads),
                            lin(d, cfg.index_head_dim) if own else None, norm(cfg.index_head_dim) if own else None)
    return AttnWeights(
        wq_a=lin(cfg.hidden_size, cfg.q_lora_rank), wkv=lin(cfg.hidden_size, d), wq_b=lin(cfg.q_lora_rank, h * d),
        wo_a=[lin(h * d // cfg.o_groups, cfg.o_lora_rank) for _ in range(cfg.o_groups)],
        wo_b=lin(cfg.o_groups * cfg.o_lora_rank, cfg.hidden_size), q_norm=norm(cfg.q_lora_rank), kv_norm=norm(d),
        attn_sink=torch.randn(h, generator=gen), compressor=comp, indexer=ix)
