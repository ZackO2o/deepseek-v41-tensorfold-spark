"""Engram kernels (engine/kernels/engram) in Triton's CPU interpreter with the ``gpu_like`` model:

- dequant == the reference's ``bf16(fp8_e4m3_dequant(...))`` bit for bit over every e4m3 byte and hard scales
  (0, subnormal results, 2^127), a rank's 12 heads at their column offset;
- fusion: the update == ``bf16(x + gate * value)`` bit for bit given the kernel's gates, the gates within 1e-6 of the
  torch formula on the kernel's FMA-chain sums; image tokens (keep 0) leave the row's bits unchanged;
- row invariance (a row alone == the row in a window, bit for bit);
- the whole module (host hash -> records -> dequant -> wkv -> fusion) against engine/reference/engram.py's Engram.

Run: TRITON_INTERPRET=1 python -m pytest -q tests/kernels/test_engram_interpreter.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from engine.kernels import engram  # noqa: E402
from engine.kernels.engram import kernels as K  # noqa: E402
from engine.kernels.engram import ref  # noqa: E402

INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(K._fuse).__name__ == "InterpretedFunction"
pytestmark = [pytest.mark.skipif(not INTERP, reason="Triton's CPU interpreter (TRITON_INTERPRET=1)"),
              pytest.mark.usefixtures("gpu_like")]

NAN_BYTES = (0x7F, 0xFF)


def _bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def _records(R: int, H: int, seed: int, hard: bool = False) -> torch.Tensor:
    g = np.random.default_rng(seed)
    v = g.integers(0, 256, size=(R, H, 256), dtype=np.uint8)
    v[np.isin(v, NAN_BYTES)] = 0x38
    s = g.integers(118, 136, size=(R, H, 8), dtype=np.uint8)
    if hard:
        v[0, 0] = np.array([b for b in range(256) if b not in NAN_BYTES] + [0, 0], dtype=np.uint8)
        s[0, 1] = [0, 1, 2, 127, 200, 250, 254, 3]       # zero scale, fp32 subnormal products, huge
        s[0, 2] = 1
        v[0, 2, :8] = [1, 2, 7, 8, 0x80, 0x81, 0x87, 0x88]  # e4m3 subnormals and signed zeros
    return torch.from_numpy(np.concatenate([v, s], axis=2)).contiguous()


def test_dequant_bits():
    raw = _records(5, 12, 1, hard=True)
    out = torch.zeros((5, 24 * 256), dtype=torch.bfloat16)
    engram.dequant(raw, out, col0=12)
    want = ref.dequant(raw)
    assert torch.equal(_bits(out[:, 12 * 256:]), _bits(want))
    assert bool((out[:, :12 * 256] == 0).all())
    nan = raw.clone()
    nan[1, 3, 5] = 0x7F
    engram.dequant(nan, out, col0=12)
    assert bool(torch.isnan(out[1, (12 + 3) * 256 + 5]))


def _fuse_case(R: int, d: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((R, 4 * d), generator=g).to(torch.bfloat16)
    kv = (torch.randn((R, 5 * d), generator=g) * 0.5).to(torch.bfloat16)
    kv[0, :d] = x[0, :d]                                 # a large positive dot on (row 0, stream 0)
    kv[1, d:2 * d] = -x[1, d:2 * d]                      # a negative one
    q = torch.rand((4, d), generator=g) + 0.5
    k = torch.rand((4, d), generator=g) + 0.5
    return x, kv, engram.qk_weights(q, k)


def test_fuse_against_emulation():
    d = 5120
    x, kv, qk = _fuse_case(3, d, 2)
    keep = torch.tensor([1.0, 1.0, 0.0])
    xo = x.clone()
    gate = torch.zeros((3, 4))
    engram.fuse(xo, kv, qk, keep, gate)
    torch.testing.assert_close(gate, ref.gates(x, kv, qk, keep), rtol=1e-6, atol=1e-6)
    assert torch.equal(_bits(xo), _bits(ref.apply(x, kv, gate)))
    assert torch.equal(_bits(xo[2]), _bits(x[2]))         # an image token: unchanged
    assert float(gate[0, 0]) > 0.5 and float(gate[1, 1]) < 0.5


def test_fuse_row_invariance():
    d = 1024
    x, kv, qk = _fuse_case(7, d, 3)
    xo = x.clone()
    engram.fuse(xo, kv, qk)
    for r in (0, 4, 6):
        one = x[r:r + 1].clone()
        engram.fuse(one, kv[r:r + 1].clone(), qk)
        assert torch.equal(_bits(one), _bits(xo[r:r + 1])), r


def test_module_against_reference_engram():
    from engine.reference.config import Config
    from engine.reference.engram import Engram, EngramWeights
    from engine.reference.ops import DenseLinear, fp8_e4m3_dequant
    from engine.kernels.engram import hash as H

    d = 1024
    cfg = Config()
    tm = np.random.default_rng(4).integers(0, cfg.engram_compressed_vocab_size, size=cfg.vocab_size)
    t = H.Tables.from_config(cfg, tm)
    bank = _records(1, 4099, 5)[0]                      # 4,099 records stand in for the 384M-row table

    class Rows:
        def rows(self, index):
            r = bank[index % 4099]
            return fp8_e4m3_dequant(r[:, :256].contiguous(), r[:, 256:].contiguous(), 32)

    g = torch.Generator().manual_seed(6)
    w = torch.randn((5 * d, 24 * 256), generator=g) * 0.02
    q = torch.rand((4, d), generator=g) + 0.5
    k = torch.rand((4, d), generator=g) + 0.5
    mod = Engram(cfg, 1, EngramWeights(DenseLinear(w), q, k, Rows()))
    ids = torch.randint(0, cfg.vocab_size, (6,), generator=g)
    dead = torch.tensor([False, False, True, False, False, False])
    x = (torch.randn((6, 4, d), generator=g)).to(torch.bfloat16).float()
    hashes = torch.from_numpy(t.rows(ids.numpy(), dead.numpy()))
    want = mod(x, hashes, ~dead)
    # ours: records of the rows -> dequant -> wkv (the EXL3 linear's place) -> fuse
    raw = bank[hashes[:, 0].reshape(-1) % 4099].view(6, 24, 264).contiguous()
    e = torch.zeros((6, 24 * 256), dtype=torch.bfloat16)
    engram.dequant(raw, e)
    kv = (e.float() @ w.T).to(torch.bfloat16)
    xs = x.to(torch.bfloat16).reshape(6, 4 * d).contiguous()
    engram.fuse(xs, kv, engram.qk_weights(q, k), (~dead).float())
    got = xs.float().view(6, 4, d)
    assert torch.equal(got[2], x[2])
    torch.testing.assert_close(got, want, rtol=2 ** -7, atol=1e-6)
