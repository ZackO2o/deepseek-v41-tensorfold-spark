"""CPU tests of engine/kernels: paths, toolchain discovery, and Triton-interpreter fixtures that model the GPU.

Run from the repo root (TensorFold 0.6.0 from ``TF_SRC`` or the default work trees, see engine/kernels/tf.py):

    TF_SRC=<engine checkout>/src python -m pytest -q tests/kernels -k "not interpreter"     # host, emulator, compile
    TRITON_INTERPRET=1 python -m pytest -q tests/kernels -k interpreter                           # Triton's interpreter

(Triton decides interpreter or compiler when a kernel module is imported, so the two sets run in two processes.)

A venv with torch (CPU) + triton 3.8 is enough; the compile tests also want nvcc (``NVCC=...``, CUDA 13.x with
sm_121) and torch's headers, and skip without them.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.kernels.tf import tensorfold_src  # noqa: E402

TF = tensorfold_src()
STUBS = Path(__file__).parent / "stubs"


def nvcc() -> str | None:
    cands = [os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"]
    for venv in sorted(Path.home().glob(".cache/*/lib/python3*/site-packages/nvidia/cu13/bin/nvcc")):
        cands.append(str(venv))
    for venv in sorted(Path.home().glob(".cache/*/*/lib/python3*/site-packages/nvidia/cu13/bin/nvcc")):
        cands.append(str(venv))
    for c in cands:
        if c and Path(c).is_file():
            return c
    return None


def torch_include() -> Path | None:
    try:
        import torch
    except ImportError:
        return None
    inc = Path(torch.__file__).parent / "include"
    return inc if (inc / "ATen" / "ATen.h").is_file() else None


@pytest.fixture(scope="session")
def tf_src() -> Path:
    if TF is None:
        pytest.skip("TensorFold 0.6.0 not found (TF_SRC)")
    return TF


# -- Triton's CPU interpreter, made to compute like the GPU where the tests depend on it ----------------------------
def fma32(a, b, c):
    """fp32 fma(a, b, c) rounded once (nearest even), elementwise: the product is exact in float64, the float64 sum
    is rounded to odd (TwoSum error term), so the final rounding to fp32 is correct."""

    import numpy as np

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


@pytest.fixture
def gpu_like(monkeypatch):
    """TRITON_INTERPRET kernels with the GPU's semantics where Triton 3.8's interpreter differs:

    - ``tl.dot`` of bf16 / fp16 operands: the interpreter multiplies the raw uint16 storage; here the operands are
      widened exactly and each output element is the float64 sum of exact products rounded once to fp32, plus the
      accumulator (a fixed function of the element's own row and column: tile-mates never change it, like mma);
    - ``tl.dot`` of fp32 operands (``input_precision="ieee"``): the GPU's FMA chain, acc = fma(a_k, b_k, acc) for k
      ascending from the accumulator, in exactly emulated fp32 (GLM's MLA-EXPAND interpreter model); numpy's BLAS
      would make an element depend on alignment;
    - fp32 -> bf16 rounds to nearest even (``cvt.rn.bf16.f32``); the interpreter truncates.
    """

    import numpy as np
    import torch
    import triton.language as tl
    from triton.runtime import interpreter as itp

    def widen(h):
        if h.dtype.scalar == tl.bfloat16:
            return (h.data.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        if h.dtype.scalar == tl.float16:
            return h.data.astype(np.float64) if h.data.dtype == np.float16 else h.data.view(np.float16).astype(
                np.float64)
        return h.data.astype(np.float64)

    orig_dot = itp.InterpreterBuilder.create_dot

    def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
        if a.dtype.scalar in (tl.bfloat16, tl.float16) and b.dtype.scalar in (tl.bfloat16, tl.float16):
            prod = np.matmul(widen(a), widen(b)).astype(np.float32)
            return itp.TensorHandle((prod + d.data.astype(np.float32)).astype(np.float32), d.dtype.scalar)
        if a.data.dtype == np.float32 and b.data.dtype == np.float32 and a.data.ndim == 2:
            acc = d.data.astype(np.float32)
            for k in range(a.data.shape[1]):
                acc = fma32(a.data[:, k:k + 1], b.data[k:k + 1, :], acc)
            return itp.TensorHandle(acc, d.dtype.scalar)
        return orig_dot(self, a, b, d, input_precision, max_num_imprecise_acc)

    orig_cast = itp.InterpreterBuilder.cast_impl

    def cast_impl(self, src, dst_type):
        if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
            data = torch.from_numpy(np.ascontiguousarray(src.data)).to(torch.bfloat16).view(torch.int16).numpy()
            return itp.TensorHandle(data.view(np.uint16), dst_type.scalar)
        return orig_cast(self, src, dst_type)

    monkeypatch.setattr(itp.InterpreterBuilder, "create_dot", create_dot)
    monkeypatch.setattr(itp.InterpreterBuilder, "cast_impl", cast_impl)
    monkeypatch.setattr(itp.InterpreterBuilder, "create_fp_trunc", cast_impl)
    yield
