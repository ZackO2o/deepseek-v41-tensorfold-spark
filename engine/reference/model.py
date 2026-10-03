"""DeepSeek-V4.1-Flash text model: embedding -> 40 hyper-connected blocks (CED: encoder 0-19, decoder 20-39) ->
hc collapse -> norm -> head, plus the DSpark drafter (checkpoint ``mtp.*``).

Block L (vLLM ``deepseek_v4_1/nvidia/model.py``):

    if L in engram layers:  streams = engram(streams)          # layers 1 and 14
    post, comb, x, pre_a = hc_pre(streams, hc_attn, pre_in = previous block's FFN pre-mix (None at L = 0), attn_norm)
    streams = hc_post(attention(x), streams, post, comb)
    post, comb, x, pre_f = hc_pre(streams, hc_ffn, pre_in = pre_a, ffn_norm)
    streams = hc_post(moe(x), streams, post, comb)

CED in vLLM (the M1 oracle) is the plain layer loop over every token ("full" mode): the decoder's global KV is layer
20's compressor over its own input (the stream leaving the encoder). DeepSeek's decoder bounded replay (decoder over
the prompt's last 128 tokens only) is an engine-level prefill policy and is not part of this reference.

The reference works on one sequence starting at position 0, all tokens at once, in fp32 holding bf16 values at the
storage boundaries (see ``ops.py``). Activations of a 2k-token prompt are ~100 MB, so a whole-model forward streams
one layer's weights at a time (``Model.forward(..., release=True)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .attention import Attention, AttnWeights, CsaState, attend
from .config import Config
from .engram import Engram, EngramWeights, NgramHasher
from .hc import HcParams, hc_collapse, hc_post, hc_pre
from .moe import MoE, MoEWeights
from .ops import F32, Linear, Numerics, bf16, linear_out, rms_norm


@dataclass
class LayerWeights:
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    hc_attn: HcParams
    hc_ffn: HcParams
    attn: AttnWeights
    moe: MoEWeights
    engram: EngramWeights | None = None
    release: Callable[[], None] | None = None        # frees dequantized / fetched tensors after use


@dataclass
class ModelWeights:
    embed: Callable[[torch.Tensor], torch.Tensor]    # ids [T] -> [T, D] bf16 values
    layer: Callable[[int], LayerWeights]
    norm: torch.Tensor | None = None
    head: Linear | None = None


@dataclass
class ForwardOut:
    logits: torch.Tensor | None                      # [T, V] fp32
    hidden: torch.Tensor                             # [T, D] final hc-collapsed hidden (pre-norm)
    streams: torch.Tensor                            # [T, hc, D] after the last block run
    pre_mix: torch.Tensor                            # [T, hc] the last FFN pre-mix
    aux: dict[int, torch.Tensor] = field(default_factory=dict)          # tap layer -> [T, D] stream mean
    routing: dict[int, torch.Tensor] = field(default_factory=dict)      # layer -> [T, k] expert ids
    layer_out: dict[int, torch.Tensor] = field(default_factory=dict)    # layer -> [T, hc, D] (if requested)


class Block:
    def __init__(self, cfg: Config, layer: int, w: LayerWeights, numerics: Numerics):
        self.cfg, self.layer, self.w = cfg, layer, w
        self.attn = Attention(cfg, layer, w.attn, numerics)
        self.moe = MoE(cfg, layer, w.moe, numerics)
        self.engram = Engram(cfg, layer, w.engram) if w.engram is not None else None

    def _pre(self, streams, p: HcParams, pre_in, norm):
        c = self.cfg
        return hc_pre(streams, p, pre_in, norm, c.rms_norm_eps, c.rms_norm_eps, c.hc_eps, c.hc_post_alpha,
                      c.hc_sinkhorn_iters)

    def __call__(self, streams: torch.Tensor, pre_in: torch.Tensor | None, positions: torch.Tensor,
                 state: CsaState, hashes: torch.Tensor | None = None, keep: torch.Tensor | None = None,
                 attention: Callable[[torch.Tensor], torch.Tensor] | None = None):
        if self.engram is not None and hashes is not None:
            streams = self.engram(streams, hashes, keep)
        post, comb, x, pre_a = self._pre(streams, self.w.hc_attn, pre_in, self.w.attn_norm)
        a = attention(x) if attention is not None else self.attn(x, positions, state)
        streams = hc_post(a, streams, post, comb)
        post, comb, x, pre_f = self._pre(streams, self.w.hc_ffn, pre_a, self.w.ffn_norm)
        streams = hc_post(self.moe(x), streams, post, comb)
        return streams, pre_f


class Model:
    def __init__(self, cfg: Config, weights: ModelWeights, numerics: Numerics = Numerics(),
                 hasher: NgramHasher | None = None):
        self.cfg, self.w, self.num, self.hasher = cfg, weights, numerics, hasher

    def forward(self, ids: torch.Tensor, n_layers: int | None = None, release: bool = False,
                keep_layer_out: tuple[int, ...] = (), with_logits: bool = True,
                image_mask: torch.Tensor | None = None) -> ForwardOut:
        return self.forward_many([ids], n_layers, release, keep_layer_out, with_logits,
                                 None if image_mask is None else [image_mask])[0]

    def forward_many(self, seqs: list[torch.Tensor], n_layers: int | None = None, release: bool = False,
                     keep_layer_out: tuple[int, ...] = (), with_logits: bool = True,
                     image_masks: list[torch.Tensor] | None = None,
                     progress: Callable[[int, float], None] | None = None) -> list[ForwardOut]:
        """Several independent sequences, layer-major: each layer's weights are fetched and dequantized once, the
        MoE runs on all sequences' tokens at once (it is per token), attention and Engram per sequence."""

        import time

        cfg = self.cfg
        n = cfg.num_hidden_layers if n_layers is None else n_layers
        runs = []
        for i, ids in enumerate(seqs):
            ids = ids.to(torch.int64).reshape(-1).cpu()
            mask = None if image_masks is None else image_masks[i]
            h = bf16(self.w.embed(ids))
            positions = torch.arange(ids.numel(), device=h.device)
            hashes = keep = None
            if self.hasher is not None and any(L < n for L in cfg.engram_layer_ids):
                hashes = self.hasher(ids, None if mask is None else mask.cpu())
                keep = None if mask is None else ~mask.to(h.device)
            runs.append({"pos": positions, "streams": h.unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous(),
                         "pre": None, "state": CsaState(), "hashes": hashes, "keep": keep,
                         "out": ForwardOut(None, h, h, torch.zeros(0))})
        for L in range(n):
            t0 = time.time()
            lw = self.w.layer(L)
            blk = Block(cfg, L, lw, self.num)
            xs, posts = [], []
            for r in runs:
                st = r["streams"]
                if blk.engram is not None and r["hashes"] is not None:
                    st = blk.engram(st, r["hashes"], r["keep"])
                post, comb, x, pre_a = blk._pre(st, lw.hc_attn, r["pre"], lw.attn_norm)
                st = hc_post(blk.attn(x, r["pos"], r["state"]), st, post, comb)
                post, comb, x, pre_f = blk._pre(st, lw.hc_ffn, pre_a, lw.ffn_norm)
                r["streams"], r["pre"] = st, pre_f
                xs.append(x)
                posts.append((post, comb))
            y = blk.moe(torch.cat(xs, 0))
            ids_all = blk.moe.last_ids
            a = 0
            for r, x, (post, comb) in zip(runs, xs, posts):
                b = a + x.shape[0]
                r["streams"] = hc_post(y[a:b], r["streams"], post, comb)
                out = r["out"]
                out.routing[L] = ids_all[a:b]
                if L in keep_layer_out:
                    out.layer_out[L] = r["streams"].clone()
                if L + 1 in cfg.aux_layer_ids:
                    out.aux[L + 1] = bf16(r["streams"].mean(dim=1))
                a = b
            if release and lw.release is not None:
                lw.release()
            if progress is not None:
                progress(L, time.time() - t0)
        outs = []
        for r in runs:
            out = r["out"]
            hidden = hc_collapse(r["streams"], r["pre"])
            out.hidden, out.streams, out.pre_mix = hidden, r["streams"], r["pre"]
            if with_logits and self.w.norm is not None and self.w.head is not None:
                out.logits = self.w.head(bf16(rms_norm(hidden, self.w.norm, cfg.rms_norm_eps)))
                if self.num.logits_bf16:
                    out.logits = bf16(out.logits)
            outs.append(out)
        return outs


# ---------------------------------------------------------------------------------------------------------------
# DSpark (vLLM deepseek_v4_1/nvidia/dspark.py + v1/worker/gpu/spec_decode/{dflash,dspark}/speculator.py)
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class DSparkWeights:
    blocks: list[LayerWeights]
    main_proj: Linear                    # [3 * D] -> D
    main_norm: torch.Tensor
    norm: torch.Tensor
    markov_w1: torch.Tensor              # [V, r] (``markov_head.embed``)
    markov_w2: torch.Tensor              # [V, r] (``markov_head.head``)
    confidence: torch.Tensor | None      # [1, D + r]


@dataclass
class DraftOut:
    tokens: torch.Tensor                 # [n] greedy draft tokens
    base_logits: torch.Tensor            # [n, V] before the Markov bias
    logits: torch.Tensor                 # [n, V] with the Markov bias of the sampled prefix
    confidence: torch.Tensor | None      # [n] acceptance probability per position
    head_hidden: torch.Tensor            # [n, D]


class DSpark:
    """One drafting pass: anchor + (n - 1) noise tokens at positions P .. P + n - 1, every position predicting the
    next token. Each block attends non-causally over the whole draft block plus the last ``sliding_window`` context
    positions, whose K/V rows each block computes from ``main_x`` (the target's taps) with its own ``wkv``."""

    def __init__(self, cfg: Config, w: DSparkWeights, embed: Callable[[torch.Tensor], torch.Tensor], head: Linear,
                 numerics: Numerics = Numerics()):
        self.cfg, self.w, self.embed, self.head, self.num = cfg, w, embed, head, numerics
        self.blocks = [Block(cfg, cfg.num_hidden_layers + i, bw, numerics) for i, bw in enumerate(w.blocks)]

    def main_x(self, aux: dict[int, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """main_norm(main_proj(cat(taps in dspark_target_layer_ids order))) [T, D]."""

        if isinstance(aux, dict):
            aux = torch.cat([aux[L] for L in self.cfg.aux_layer_ids], dim=-1)
        return bf16(rms_norm(linear_out(self.w.main_proj, aux), self.w.main_norm, self.cfg.rms_norm_eps))

    def draft(self, main_x: torch.Tensor, anchor: int, n: int | None = None) -> DraftOut:
        cfg = self.cfg
        n = cfg.dspark_block_size if n is None else n
        p = main_x.shape[0]                                        # anchor position = context length
        dev = main_x.device
        lo = max(0, p - cfg.sliding_window)
        ctx_pos = torch.arange(lo, p, device=dev)
        q_pos = torch.arange(p, p + n, device=dev)
        ids = torch.tensor([anchor] + [cfg.dspark_noise_token_id] * (n - 1))
        h = bf16(self.embed(ids))
        streams = h.unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        pre = None
        state = CsaState()
        for blk in self.blocks:
            attn = blk.attn
            ctx_kv = bf16(rms_norm(linear_out(attn.w.wkv, main_x[lo:p]), attn.w.kv_norm, cfg.rms_norm_eps))
            ctx_k = attn.kv_rows(ctx_kv, ctx_pos)

            def attention(x: torch.Tensor, attn=attn, ctx_k=ctx_k) -> torch.Tensor:
                q, k_q, _ = attn.project(x, q_pos)
                keys = torch.cat([ctx_k, k_q], dim=0)
                visible = torch.ones((n, keys.shape[0]), dtype=torch.bool, device=keys.device)
                return attn.output(attend(q, keys, visible, attn.w.attn_sink), q_pos)

            streams, pre = blk(streams, pre, q_pos, state, attention=attention)
        hidden = hc_collapse(streams, pre)
        base = self.head(bf16(rms_norm(hidden, self.w.norm, cfg.rms_norm_eps)))
        prev = anchor
        toks, logits, embeds = [], [], []
        for i in range(n):
            me = bf16(self.w.markov_w1[prev].to(F32))
            li = base[i] + me @ self.w.markov_w2.to(F32).t()
            tok = int(li.argmax())
            toks.append(tok)
            logits.append(li)
            embeds.append(me)
            prev = tok
        confidence = None
        if self.w.confidence is not None:
            feats = torch.cat([hidden, torch.stack(embeds)], dim=-1)
            confidence = torch.sigmoid(feats @ self.w.confidence.to(F32).t()).squeeze(-1)
        return DraftOut(torch.tensor(toks), base, torch.stack(logits), confidence, hidden)
