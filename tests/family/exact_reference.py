"""engine/reference with every storage rounding removed: float64 math, no bf16 / fp16 / FP8 boundaries (the
reference's real-number function), to compare the family's forward under ``numerics.exact()`` at ~1e-10."""

from __future__ import annotations

from contextlib import contextmanager

import torch

F64 = torch.float64


@contextmanager
def exact_reference():
    import engine.reference.attention as A
    import engine.reference.engram as E
    import engine.reference.hc as H
    import engine.reference.loader as LD
    import engine.reference.model as M
    import engine.reference.moe as MO
    import engine.reference.ops as O
    import engine.reference.rope as R

    mods = (O, A, H, MO, E, M, LD)
    names = ("F32", "bf16", "fp8_e4m3_dequant", "linear_out")
    saved = [(m, k, getattr(m, k)) for m in mods for k in names if hasattr(m, k)]
    saved.append((R.Rope, "apply", R.Rope.apply))
    saved.append((LD.CheckpointLoader, "engram", LD.CheckpointLoader.engram))
    engram = LD.CheckpointLoader.engram

    def engram_weights(self, layer, reg):           # q / k weights are bf16 parameters (vLLM): rounded once
        w = engram(self, layer, reg)
        if w is not None:
            w.q_weight = w.q_weight.to(torch.bfloat16).to(F64)
            w.k_weight = w.k_weight.to(torch.bfloat16).to(F64)
        return w
    dq = O.fp8_e4m3_dequant

    def dequant(raw, scale, block=32):          # its bit trick needs fp32; the values are exact either way
        O.F32 = torch.float32
        try:
            return dq(raw, scale, block).to(F64)
        finally:
            O.F32 = F64

    def apply(self, x, positions, inverse=False):    # fp32 cos / sin (the table's values), float64 rotation
        x = x.to(F64)
        cos, sin = self.cos_sin(positions.reshape(-1))
        cos, sin = cos.to(F64), (-sin if inverse else sin).to(F64)
        shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[1],)
        cos, sin = cos.reshape(shape), sin.reshape(shape)
        rot = x[..., -self.rope_dim:]
        even, odd = rot[..., 0::2], rot[..., 1::2]
        out = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
        return torch.cat((x[..., :-self.rope_dim], out), dim=-1)

    try:
        for m in mods:
            if hasattr(m, "F32"):
                m.F32 = F64
            if hasattr(m, "bf16") and m is not LD:      # the loader's bf16 is a parameter's dtype: kept
                m.bf16 = lambda x: x.to(F64)
            if hasattr(m, "fp8_e4m3_dequant"):
                m.fp8_e4m3_dequant = dequant
            if hasattr(m, "linear_out"):
                m.linear_out = lambda lin, x, dtype=None: lin(x).to(F64)
        R.Rope.apply = apply
        LD.CheckpointLoader.engram = engram_weights
        yield
    finally:
        for m, k, v in saved:
            setattr(m, k, v)
