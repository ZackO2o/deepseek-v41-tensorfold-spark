"""Engram: n-gram hash lookups gated into the hyper-connection streams (layers 1 and 14 of V4.1-Flash).

Port of vLLM ``deepseek_v4_1/common/engram.py`` (Apache-2.0), which ports DeepSeek's ``inference/engram.py``:

- token ids map to a compressed vocabulary (99,092 ids) through a tokenizer normalizer (NFKC, NFD, strip accents,
  lowercase, collapse whitespace);
- at each position p and n-gram order n = 2..4, ``rolling = XOR_{s < n} id'[p - s] * mult[layer, s]`` (int64; a
  position before the sequence start or an image token blocks the rest of the window: it and every older slot
  contribute the pad id instead), and each of the 8 heads of that order maps it to ``rolling % prime + offset``: a
  row of the layer's table. Primes are the next unused primes above ``engram_vocab_size - 1``, drawn in (layer,
  order, head) order; offsets are their running sums, so the 24 heads' ranges are disjoint;
- rows are 256 FP8 e4m3 values with a UE8M0 scale every 32 (never quantized further), dequantized to bf16;
- ``kv = wkv(rows.flat)`` [T, 5 * 5120]: one key a stream plus a shared value;
- each stream h is gated: ``dot = sum(x_h * q_h * k_h * key_h) * rsqrt(ms(x_h)) * rsqrt(ms(key_h)) / sqrt(5120)``,
  ``gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))``, ``x_h += gate * value`` (gate 0 on image tokens).

The injection happens on the full stream tensor between the previous layer's FFN post and this layer's attention
pre (``deepseek_v4_1/nvidia/model.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from .config import Config
from .ops import F32, Linear, bf16, fp8_e4m3_dequant, linear_out, rms_norm

DEAD_ID = -1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


class EngramLayout:
    """Prime bucket sizes and offsets of every (layer, order, head)."""

    def __init__(self, cfg: Config):
        self.layer_ids = tuple(cfg.engram_layer_ids)
        self.max_ngram = cfg.engram_max_ngram_size
        self.n_heads = cfg.engram_n_heads
        seen: set[int] = set()
        primes = []
        for _ in self.layer_ids:
            per_layer = []
            for _ in range(self.max_ngram - 1):
                cur = cfg.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    cur = find_next_prime(cur, seen)
                    seen.add(cur)
                    per_layer.append(cur)
            primes.append(per_layer)
        self.primes = torch.tensor(primes, dtype=torch.int64)                       # [layers, cols]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *p[:-1]]) for p in primes]), dtype=torch.int64)

    @classmethod
    def from_config(cls, cfg: Config) -> "EngramLayout":
        return cls(cfg)

    def total_rows(self, layer_index: int) -> int:
        return int(self.primes[layer_index].sum())

    def head_shard(self, layer_index: int, rank: int, tp: int) -> tuple[int, int, int, int]:
        """(row start, row end, first head, heads) a TP rank owns (vLLM shards complete heads, TP-major)."""

        sizes = self.primes[layer_index].tolist()
        part = (len(sizes) + tp - 1) // tp
        h0 = rank * part
        h1 = min(h0 + part, len(sizes))
        return int(sum(sizes[:h0])), int(sum(sizes[:h1])), h0, h1 - h0


def hash_multipliers(layer_ids: tuple[int, ...], max_ngram: int, compressed_vocab: int) -> torch.Tensor:
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab) // 2)
    rows = []
    for layer_id in layer_ids:
        g = np.random.default_rng(10007 * layer_id)
        v = g.integers(low=0, high=bound, size=(max_ngram,), dtype=np.int64)
        rows.append(torch.tensor(v * 2 + 1))
    return torch.stack(rows)


def build_compressed_token_map(tokenizer_json: str) -> tuple[list[int], int]:
    """Token id -> compressed id, from ``tokenizer.json`` with the ``tokenizers`` package (vLLM's normalizer)."""

    from tokenizers import Regex, Tokenizer, normalizers

    tok = Tokenizer.from_file(str(tokenizer_json))
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * n
    for i in range(n):
        text = tok.decode([i], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(i)
        else:
            normalized = norm.normalize_str(text)
            key = normalized if normalized else text
        j = key_to_new.get(key)
        if j is None:
            j = len(key_to_new)
            key_to_new[key] = j
        lookup[i] = j
    return lookup, len(key_to_new)


class NgramHasher:
    """[T] token ids of one sequence (from position 0) -> [T, layers, cols] int64 table rows."""

    def __init__(self, cfg: Config, token_map: list[int] | torch.Tensor):
        self.cfg = cfg
        self.layout = EngramLayout(cfg)
        self.token_map = torch.as_tensor(token_map, dtype=torch.int64)
        self.multipliers = hash_multipliers(self.layout.layer_ids, cfg.engram_max_ngram_size,
                                            cfg.engram_compressed_vocab_size)
        self.pad_id = int(self.token_map[cfg.engram_pad_token_id])

    def __call__(self, ids: torch.Tensor, dead: torch.Tensor | None = None,
                 lookback: torch.Tensor | None = None) -> torch.Tensor:
        """``dead`` [T]: image tokens. ``lookback``: ids just before ``ids`` (oldest first), for a chunk that does
        not start the sequence; omitted = the chunk starts at position 0."""

        ids = ids.to(torch.int64)
        dead = torch.zeros_like(ids, dtype=torch.bool) if dead is None else dead.to(torch.bool)
        nb = 0 if lookback is None else lookback.numel()
        all_ids = ids if lookback is None else torch.cat([lookback.to(torch.int64), ids])
        all_dead = dead if lookback is None else torch.cat([torch.zeros(nb, dtype=torch.bool), dead])
        src = torch.where(all_dead, DEAD_ID, self.token_map[all_ids])
        t = ids.numel()
        nl = len(self.layout.layer_ids)
        mx = self.cfg.engram_max_ngram_size
        nh = self.cfg.engram_n_heads
        out = torch.empty((t, nl, (mx - 1) * nh), dtype=torch.int64)
        idx = torch.arange(t) + nb
        for li in range(nl):
            blocked = torch.zeros(t, dtype=torch.bool)
            rolling = torch.zeros(t, dtype=torch.int64)
            for s in range(mx):
                j = idx - s
                before = j < 0
                v = torch.where(before, torch.full_like(j, self.pad_id), src[j.clamp(min=0)])
                blocked = blocked | before | (v == DEAD_ID)
                value = torch.where(blocked, torch.full_like(v, self.pad_id), v)
                rolling = rolling ^ (value * self.multipliers[li, s])
                if s > 0:
                    cols = slice((s - 1) * nh, s * nh)
                    out[:, li, cols] = rolling[:, None] % self.layout.primes[li, cols][None, :] + \
                        self.layout.offsets[li, cols][None, :]
        return out


class RowSource(Protocol):
    def rows(self, index: torch.Tensor) -> torch.Tensor:
        """[N] int64 row ids -> [N, head_dim] fp32 (dequantized FP8 rows)."""
        ...


class TensorRows:
    """A table held in memory: fp32 rows, or raw FP8 bytes + UE8M0 scales."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor | None = None, block: int = 32):
        self.weight, self.scale, self.block = weight, scale, block

    def rows(self, index: torch.Tensor) -> torch.Tensor:
        w = self.weight[index]
        if self.scale is None:
            return w.to(F32)
        return fp8_e4m3_dequant(w, self.scale[index], self.block)


@dataclass
class EngramWeights:
    wkv: Linear                       # [24 * 256] -> [5 * 5120] (EXL3)
    q_weight: torch.Tensor            # [4, 5120] (bf16 parameter in vLLM)
    k_weight: torch.Tensor            # [4, 5120]
    table: RowSource


class Engram:
    def __init__(self, cfg: Config, layer: int, w: EngramWeights):
        self.cfg, self.layer, self.w = cfg, layer, w
        self.hash_index = cfg.engram_layer_ids.index(layer)

    def __call__(self, streams: torch.Tensor, hashes: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """streams [T, hc, D]; hashes [T, layers, cols] (this layer's column block is used); keep [T] (False =
        image token, gate 0)."""

        cfg = self.cfg
        t, hc, d = streams.shape
        ids = hashes[:, self.hash_index]                                 # [T, cols]
        rows = bf16(self.w.table.rows(ids.reshape(-1).cpu())).view(t, -1).to(streams.device)
        kv = linear_out(self.w.wkv, rows)                                # [T, (hc + 1) * D]
        x = streams.to(F32)
        key = kv[:, : hc * d].view(t, hc, d)
        value = kv[:, hc * d:]
        qk = bf16(self.w.q_weight.to(F32)) * bf16(self.w.k_weight.to(F32))
        dot = (x * qk[None] * key).sum(-1)
        dot = dot * torch.rsqrt(x.square().mean(-1) + cfg.rms_norm_eps) * \
            torch.rsqrt(key.square().mean(-1) + cfg.rms_norm_eps) / (d ** 0.5)
        g = torch.sqrt(dot.abs().clamp(min=cfg.engram_gate_clamp))
        gate = torch.sigmoid(torch.where(dot < 0, -g, g))
        if keep is not None:
            gate = gate * keep.to(device=gate.device, dtype=F32)[:, None]
        return bf16(x + gate.unsqueeze(-1) * value.unsqueeze(1))


__all__ = ["EngramLayout", "NgramHasher", "Engram", "EngramWeights", "TensorRows", "RowSource",
           "build_compressed_token_map", "hash_multipliers", "rms_norm"]
