"""EXL3 (ExLlamaV3 trellis quantization) decoding for the reference, every codebook and width, mul1 included.

Two decoders that must agree bit for bit on ``W_q``:

- ``np_*``: vendored from TensorFold v0.6.0 ``src/tensorfold/cuda/exl3/format.py`` (Apache-2.0, Copyright 2026
  TensorFold contributors; the EXL3 format is ExLlamaV3's, MIT, Copyright (c) 2025 Turboderp). Upstream tests it
  bit for bit against ExLlamaV3's ``reconstruct`` for all 27 codebook x width pairs. Kept as the oracle.
- ``unpack`` / ``dequantize`` / ``Exl3Weight``: a torch port of the same arithmetic (CPU or CUDA), used by the model.

Layout: a layer's ``trellis`` is int16 ``[K/16, N/16, 16 * bits]`` (K = inputs, N = outputs, so the bits per weight
come from the trellis' last dim, never from ``int(avg_bits)``); ``suh`` fp16 ``[K]``, ``svh`` fp16 ``[N]``; a scalar
int32 marker ``mul1`` (0x83DCD12D) or ``mcg`` (0xCBAC1FED) names the codebook (none = ``3inst``).
``W = diag(suh) @ H_K @ W_q @ H_N @ diag(svh)`` with H the 128-block Sylvester Hadamard / sqrt(128), and the layer
computes ``y = x @ W`` (not ``x @ W.T``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

CODEBOOKS = ("3inst", "mcg", "mul1")
MARKERS = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
INST3_MUL, INST3_ADD = 89226354, 64248484
MCG_MUL, MUL1_MUL = 0xCBAC1FED, 0x83DCD12D
MASK, FLIP = 0x8FFF8FFF, 0x3B603B60
MUL1_SCALE, MUL1_BIAS = 0x1EEE, 0xC931
HAD = 128
BITS = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)


# ---------------------------------------------------------------------------------------------------------------
# numpy oracle (vendored from TensorFold v0.6.0 cuda/exl3/format.py, Apache-2.0)
# ---------------------------------------------------------------------------------------------------------------

def _fp16(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.uint16).view(np.float16).astype(np.float64)


@lru_cache(maxsize=3)
def np_codebook(name: str) -> np.ndarray:
    """The fp16 value of every 16-bit state, [65536], for codebook ``name``."""

    s = np.arange(65536, dtype=np.uint64)
    if name == "mul1":
        x = (s * MUL1_MUL) & 0xFFFFFFFF
        h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
        scale, bias = _fp16(np.array(MUL1_SCALE)), _fp16(np.array(MUL1_BIAS))
        return (h.astype(np.float64) * scale + bias).astype(np.float16)
    if name == "mcg":
        x = (s * MCG_MUL) & 0xFFFFFFFF
    elif name == "3inst":
        x = (s * INST3_MUL + INST3_ADD) & 0xFFFFFFFF
    else:
        raise ValueError(f"unknown EXL3 codebook {name!r} (known: {', '.join(CODEBOOKS)})")
    x = (x & MASK) ^ FLIP
    return (_fp16(x & 0xFFFF) + _fp16(x >> 16)).astype(np.float16)


def check_bits(bits: float) -> float:
    b = float(bits)
    if b not in BITS:
        raise ValueError(f"EXL3 bits must be one of {', '.join(str(v) for v in BITS)}, got {bits}")
    return b


def stream_ends(bits: float) -> np.ndarray:
    """E(p) for p = 0..255: where value p's 16-bit window ends in the tile's bitstream (exclusive)."""

    b = check_bits(bits)
    p1 = np.arange(1, 257, dtype=np.int64)
    if b.is_integer():
        return p1 * int(b)
    k2 = int(2 * b)
    return (p1 * k2 - (p1 % 2)) // 2


@lru_cache(maxsize=1)
def tile_positions() -> tuple[np.ndarray, np.ndarray]:
    """(row, column) in its 16x16 tile of each of a tile's 256 values, in stream order."""

    p = np.arange(256)
    lane, j = p // 8, p % 8
    rows = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    cols = lane // 4 + 8 * (j >> 2)
    return rows, cols


def tile_words(bits: float) -> int:
    return int(16 * check_bits(bits))


def bits_of(trellis_shape) -> float:
    """A trellis' bits per weight from its shape (last dim / 16)."""

    last = int(tuple(trellis_shape)[-1])
    b = last / 16
    b = int(b) if float(b).is_integer() else b
    return check_bits(b) if last % 8 == 0 else check_bits(-1)


def np_states(t: np.ndarray, bits: float) -> np.ndarray:
    nw16 = tile_words(bits)
    if t.dtype != np.int16 or t.shape[-1] != nw16:
        raise ValueError(f"trellis must be int16 [..., {nw16}] for {bits} bits, got {t.dtype} {t.shape}")
    w = t.view(np.uint16).astype(np.uint64)
    words = w[..., 0::2] | (w[..., 1::2] << 16)
    nw = nw16 // 2
    ring = 32 * nw
    first = stream_ends(bits) - 16 + ring
    i0, off = (first // 32) % nw, first % 32
    i1 = (i0 + 1) % nw
    pair = (words[..., i0] << 32) | words[..., i1]
    return ((pair >> (48 - off).astype(np.uint64)) & 0xFFFF).astype(np.uint32)


def np_unpack(trellis: np.ndarray, bits: float, codebook_name: str, chunk: int = 16) -> np.ndarray:
    """W_q [K, N] fp16 in the rotated domain (ExLlamaV3's ``reconstruct`` before the rotations)."""

    t = np.ascontiguousarray(trellis)
    kt, nt = t.shape[0], t.shape[1]
    table = np_codebook(codebook_name)
    rows, cols = tile_positions()
    w = np.empty((kt, 16, nt, 16), dtype=np.float16)
    for k0 in range(0, kt, chunk):
        vals = table[np_states(t[k0:k0 + chunk], bits).astype(np.int64)]
        w[k0:k0 + chunk][:, rows, :, cols] = vals.transpose(2, 0, 1)
    return w.reshape(kt * 16, nt * 16)


@lru_cache(maxsize=4)
def np_hadamard(n: int = HAD) -> np.ndarray:
    i = np.arange(n)
    parity = np.array([bin(v).count("1") & 1 for v in range(n)])
    return np.where(parity[(i[:, None] & i[None, :])] == 1, -1.0, 1.0)


def np_rotate(x: np.ndarray, axis: int) -> np.ndarray:
    h = np_hadamard() / np.sqrt(HAD)
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    shape = x.shape
    x = (x.reshape(*shape[:-1], shape[-1] // HAD, HAD) @ h).reshape(shape)
    return np.moveaxis(x, -1, axis)


def np_dequantize(trellis, suh, svh, bits: float, codebook_name: str) -> np.ndarray:
    """W [K, N] float64: diag(suh) @ H_K @ W_q @ H_N @ diag(svh)."""

    wq = np_unpack(np.asarray(trellis), bits, codebook_name).astype(np.float64)
    w = np_rotate(wq, 0) * np.asarray(suh).astype(np.float64)[:, None]
    return np_rotate(w, 1) * np.asarray(svh).astype(np.float64)[None, :]


def np_forward(x, trellis, suh, svh, bits: float, codebook_name: str) -> np.ndarray:
    """y = x @ W in float64, in the kernels' order: rotate the input, W_q, rotate the output."""

    xh = np_rotate(np.asarray(x, dtype=np.float64) * np.asarray(suh).astype(np.float64), -1)
    y = np_rotate(xh @ np_unpack(np.asarray(trellis), bits, codebook_name).astype(np.float64), -1)
    return y * np.asarray(svh).astype(np.float64)


# ---------------------------------------------------------------------------------------------------------------
# torch port
# ---------------------------------------------------------------------------------------------------------------

_TABLES: dict[tuple[str, str], torch.Tensor] = {}


def codebook(name: str, device: torch.device | str = "cpu") -> torch.Tensor:
    key = (name, str(device))
    if key not in _TABLES:
        _TABLES[key] = torch.from_numpy(np_codebook(name).copy()).to(device)
    return _TABLES[key]


@lru_cache(maxsize=16)
def _window_plan(bits: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nw = tile_words(bits) // 2
    first = stream_ends(bits) - 16 + 32 * nw
    i0, off = (first // 32) % nw, first % 32
    return i0, (i0 + 1) % nw, off


@lru_cache(maxsize=1)
def _tile_inverse() -> np.ndarray:
    """For each row-major tile cell r * 16 + c, the stream position p holding it."""

    rows, cols = tile_positions()
    inv = np.empty(256, dtype=np.int64)
    inv[rows * 16 + cols] = np.arange(256)
    return inv


def states(trellis: torch.Tensor, bits: float) -> torch.Tensor:
    """The 16-bit state of every value, int64 [..., 256] in stream order (no uint64 needed)."""

    nw16 = tile_words(bits)
    if trellis.dtype != torch.int16 or trellis.shape[-1] != nw16:
        raise ValueError(f"trellis must be int16 [..., {nw16}] for {bits} bits, got {trellis.dtype} "
                         f"{tuple(trellis.shape)}")
    w = trellis.to(torch.int64) & 0xFFFF
    words = w[..., 0::2] | (w[..., 1::2] << 16)                       # [..., nw] uint32 values in int64
    i0, i1, off = _window_plan(bits)
    dev = trellis.device
    i0t, i1t = torch.from_numpy(i0).to(dev), torch.from_numpy(i1).to(dev)
    offt = torch.from_numpy(off).to(dev)
    a = words.index_select(-1, i0t)                                     # [..., 256]
    b = words.index_select(-1, i1t)
    # 16 bits starting `off` bits below the top of the 64-bit pair (a << 32 | b)
    lo_shift = (16 - offt).clamp(min=0)
    from_a = (a >> lo_shift) & 0xFFFF                                   # off <= 16: all bits in a
    hi_shift = (offt - 16).clamp(min=0)
    split = ((a << hi_shift) | (b >> (48 - offt).clamp(max=32))) & 0xFFFF
    return torch.where(offt <= 16, from_a, split)


def unpack(trellis: torch.Tensor, bits: float, codebook_name: str, chunk: int = 64) -> torch.Tensor:
    """W_q [K, N] fp16 from trellis int16 [K/16, N/16, 16 * bits]."""

    if trellis.ndim != 3:
        raise ValueError(f"trellis must be [K/16, N/16, 16 * bits], got {tuple(trellis.shape)}")
    kt, nt = trellis.shape[0], trellis.shape[1]
    table = codebook(codebook_name, trellis.device)
    inv = torch.from_numpy(_tile_inverse()).to(trellis.device)
    out = torch.empty((kt, 16, nt, 16), dtype=torch.float16, device=trellis.device)
    for k0 in range(0, kt, chunk):
        vals = table[states(trellis[k0:k0 + chunk], bits)]               # [c, nt, 256] stream order
        tiles = vals.index_select(-1, inv).view(-1, nt, 16, 16)          # [c, nt, r, c]
        out[k0:k0 + chunk] = tiles.permute(0, 2, 1, 3)
    return out.view(kt * 16, nt * 16)


_HAD: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def hadamard(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _HAD:
        _HAD[key] = torch.from_numpy(np_hadamard() / np.sqrt(HAD)).to(device=device, dtype=dtype)
    return _HAD[key]


def rotate(x: torch.Tensor, axis: int) -> torch.Tensor:
    """H / sqrt(128) on every block of 128 along ``axis``."""

    h = hadamard(x.device, x.dtype)
    x = x.movedim(axis, -1)
    shape = x.shape
    if shape[-1] % HAD:
        raise ValueError(f"the rotated dimension must be a multiple of {HAD}, got {shape[-1]}")
    y = (x.reshape(*shape[:-1], shape[-1] // HAD, HAD) @ h).reshape(shape)
    return y.movedim(-1, axis)


def dequantize(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, bits: float, codebook_name: str,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """W [K, N] (``dtype``, fp32 by default): diag(suh) @ H_K @ W_q @ H_N @ diag(svh)."""

    wq = unpack(trellis, bits, codebook_name).to(dtype)
    w = rotate(wq, 0) * suh.to(dtype)[:, None]
    return rotate(w, 1) * svh.to(dtype)[None, :]


@dataclass
class Exl3Weight:
    """One packed EXL3 matrix (K inputs, N outputs)."""

    trellis: torch.Tensor
    suh: torch.Tensor
    svh: torch.Tensor
    codebook: str = "mul1"

    def __post_init__(self) -> None:
        self.bits = bits_of(self.trellis.shape)
        if not float(self.bits).is_integer() and self.codebook != "mul1":
            raise ValueError(f"{self.bits}-bit tiles need the mul1 codebook, got {self.codebook}")

    @property
    def k(self) -> int:
        return 16 * self.trellis.shape[0]

    @property
    def n(self) -> int:
        return 16 * self.trellis.shape[1]

    @property
    def nbytes(self) -> int:
        return (self.trellis.numel() * 2 + self.suh.numel() * 2 + self.svh.numel() * 2)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return dequantize(self.trellis, self.suh, self.svh, self.bits, self.codebook, dtype)

    def to(self, device: torch.device | str) -> "Exl3Weight":
        return Exl3Weight(self.trellis.to(device), self.suh.to(device), self.svh.to(device), self.codebook)


def codebook_from_marker(value: int | None) -> str:
    if value is None:
        return "3inst"
    v = int(value) & 0xFFFFFFFF
    for name, m in MARKERS.items():
        if v == m:
            return name
    raise ValueError(f"unknown EXL3 codebook marker {v:#x}")


def quantize_random(k: int, n: int, bits: float, codebook_name: str = "mul1", seed: int = 0,
                    scale: float = 0.02) -> Exl3Weight:
    """A random packed matrix (random trellis words, random scales): for tests of the layout, not of quality."""

    g = torch.Generator().manual_seed(seed)
    tw = tile_words(bits)
    trellis = torch.randint(-2 ** 15, 2 ** 15, (k // 16, n // 16, tw), generator=g, dtype=torch.int32).to(torch.int16)
    suh = ((torch.rand(k, generator=g) + 0.5) * torch.sign(torch.randn(k, generator=g)) * scale).to(torch.float16)
    svh = ((torch.rand(n, generator=g) + 0.5) * torch.sign(torch.randn(n, generator=g))).to(torch.float16)
    return Exl3Weight(trellis, suh, svh, codebook_name)
