"""Small numeric building blocks shared by the reference modules.

Precision rules (vLLM's DeepSeek-V4.1 path on GB10, which the M1 oracle runs):

- activations between modules are bf16; norms, mixes, softmaxes and routing are computed in fp32;
- every EXL3 linear casts its input to fp16 (ExLlamaV3 kernels take fp16), accumulates in fp32 and the caller casts
  the output to the activation dtype (``Numerics.exl3_fp16_input``);
- the KV cache is ``fp8_ds_mla``: per token, the 448 NoPE dims as FP8 e4m3 with a UE8M0 scale every 64 dims and the
  64 RoPE dims as bf16 (``Numerics.kv_fp8_ds_mla``);
- the head's logits are bf16 (``Numerics.logits_bf16``: the EXL3 head casts its fp32 result to the activation dtype;
  the kit's top-1 / top-2 margins are all multiples of 1/16, and on exact ties its argmax takes the lower id);
- the routed expert weights are fp16 in the kit's EXL3 expert kernels (``Numerics.routed_fp16_weights``);
- the indexer runs on FP8 keys (one power-of-two scale a token) and FP8 queries (one power-of-two scale a token and
  head) on SM12x (``Numerics.indexer_fp8``).

``Numerics.exact()`` turns all five emulations off (plain bf16 / fp32), which is what the unit tests compare the
pieces against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import torch
import torch.nn.functional as F

from .exl3 import Exl3Weight

BF16 = torch.bfloat16
F32 = torch.float32
FP8_MAX = 448.0


@dataclass(frozen=True)
class Numerics:
    exl3_fp16_input: bool = True
    kv_fp8_ds_mla: bool = True
    indexer_fp8: bool = True
    logits_bf16: bool = True          # the kit's EXL3 head returns bf16 logits; its argmax takes the lower id on ties
    routed_fp16_weights: bool = True  # the kit's EXL3 expert kernels take the routed weights (x 1.5) as fp16
    act_dtype: torch.dtype = BF16

    @classmethod
    def kit(cls) -> "Numerics":
        return cls()

    @classmethod
    def exact(cls) -> "Numerics":
        return cls(exl3_fp16_input=False, kv_fp8_ds_mla=False, indexer_fp8=False, logits_bf16=False,
                   routed_fp16_weights=False)


def bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 and come back to fp32 (a storage boundary)."""

    return x.to(BF16).to(F32)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """fp32 RMSNorm (weight optional), result in fp32; the caller rounds."""

    x = x.to(F32)
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y if weight is None else y * weight.to(F32)


def swiglu_clamp(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """vLLM's ``SiluAndMulWithClamp``: silu(min(gate, limit)) * clamp(up, -limit, limit)."""

    g = gate.to(F32).clamp(max=limit)
    return F.silu(g) * up.to(F32).clamp(min=-limit, max=limit)


# -- quantization emulation ---------------------------------------------------------------------------------------

def _fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(F32)


def pow2_scale(amax: torch.Tensor, qmax: float = FP8_MAX, floor: float = 1e-4) -> torch.Tensor:
    """2^ceil(log2(max(amax, floor) / qmax)): the UE8M0 / power-of-two scales vLLM uses."""

    return torch.exp2(torch.ceil(torch.log2(amax.clamp(min=floor) / qmax)))


def fp8_ds_mla_roundtrip(row: torch.Tensor, rope_dim: int = 64, tile: int = 64) -> torch.Tensor:
    """A [.., D] KV row (fp32 holding bf16 values) through the fp8_ds_mla cache: NoPE dims FP8 with a UE8M0 scale per
    ``tile``, RoPE dims bf16 (``rope_quant_insert`` / the fused SWA insert)."""

    nope = row.shape[-1] - rope_dim
    tile = tile if nope % tile == 0 else nope          # (tiny test configs: one tile)
    x = row.to(F32)
    head = x[..., :nope].reshape(*x.shape[:-1], nope // tile, tile)
    s = pow2_scale(head.abs().amax(-1, keepdim=True))
    head = (_fp8(head / s) * s).reshape(*x.shape[:-1], nope)
    return torch.cat([head, bf16(x[..., nope:])], -1)


def fp8_token_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Per-row FP8 with one power-of-two scale over the last dim (indexer K, and indexer Q per head)."""

    x = bf16(x)
    s = pow2_scale(x.abs().amax(-1, keepdim=True))
    return _fp8(x / s) * s


def fp8_e4m3_dequant(raw: torch.Tensor, scale_e8m0: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP8 rows with UE8M0 scales every ``block`` values (Engram rows, the original FP8 linears) -> fp32."""

    v = raw.view(torch.float8_e4m3fn).to(F32) if raw.dtype != torch.float8_e4m3fn else raw.to(F32)
    e = scale_e8m0.view(torch.uint8).to(torch.int32) if scale_e8m0.dtype != torch.uint8 else scale_e8m0.to(torch.int32)
    s = (e << 23).view(F32)              # the byte is the fp32 exponent field (as the lookup kernel decodes it)
    return (v.reshape(*v.shape[:-1], v.shape[-1] // block, block) * s[..., None]).reshape(v.shape)


# -- linears ------------------------------------------------------------------------------------------------------

class Linear(Protocol):
    in_features: int
    out_features: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class DenseLinear:
    """torch-layout weight [out, in]; fp32 math on the given values (no input cast)."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.out_features, self.in_features = weight.shape

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(F32) @ self.weight.to(F32).t()

    def release(self) -> None:
        pass


class Exl3Linear:
    """An EXL3 matrix applied as ``y = x @ W`` (W [K, N]); dequantized on first use and cached until ``release``."""

    def __init__(self, weight: Exl3Weight | Callable[[], Exl3Weight], numerics: Numerics = Numerics(),
                 k: int | None = None, n: int | None = None, name: str = ""):
        self._packed = weight if isinstance(weight, Exl3Weight) else None
        self._fetch = None if isinstance(weight, Exl3Weight) else weight
        self.numerics = numerics
        self.name = name
        self._w: torch.Tensor | None = None
        if self._packed is not None:
            k, n = self._packed.k, self._packed.n
        self.in_features, self.out_features = int(k or 0), int(n or 0)

    def packed(self) -> Exl3Weight:
        if self._packed is None:
            assert self._fetch is not None
            self._packed = self._fetch()
            self.in_features, self.out_features = self._packed.k, self._packed.n
        return self._packed

    def weight(self) -> torch.Tensor:
        if self._w is None:
            self._w = self.packed().dequantize(F32)
        return self._w

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight()
        x = x.to(torch.float16).to(F32) if self.numerics.exl3_fp16_input else x.to(F32)
        return x.to(w.device) @ w

    def release(self, packed: bool = False) -> None:
        self._w = None
        if packed and self._fetch is not None:
            self._packed = None


def linear_out(lin: Linear, x: torch.Tensor, dtype: torch.dtype = BF16) -> torch.Tensor:
    """Apply and round the output to the activation dtype (what vLLM's linear wrappers return)."""

    return lin(x).to(dtype).to(F32)


def scaled_dot(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
