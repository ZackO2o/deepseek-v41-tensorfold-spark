"""DeepSeek-V4.1-Flash experts on TensorFold 0.6.0's EXL3 module (engine/kernels/exl3/experts.py), no GPU:

- the TP=2 split is exact: a rank's dequantized matrix is the slice of the full one (upstream's numpy decoder,
  ``tensorfold.cuda.exl3.format``), for mul1 at every width this checkpoint uses, and the two ranks' expert outputs
  add up to the full expert's (float64, SwiGLU clamped at 10);
- tile settings: upstream's rule at this shape is GLM's (8, 4, 4, 1) / (8, 4, 1, 1); the host copy agrees;
- plans from the real header shapes: widths, x3ld instances, bytes a rank (6.6 MB a 3-bit expert);
- ``routed`` issues upstream's ``routed`` launches argument for argument with the load path off, and only swaps the
  two grouped launches with it on;
- scratch sizing equals upstream's ``Scratch`` allocation.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.kernels.exl3 import experts as dx
from engine.kernels.exl3 import loads

fmt = pytest.importorskip("tensorfold.cuda.exl3.format")


def _matrix(rng, k: int, n: int, k2: int):
    t = rng.integers(-2 ** 15, 2 ** 15, size=(k // 16, n // 16, 8 * k2), dtype=np.int64).astype(np.int16)
    suh = (rng.standard_normal(k) * 0.05).astype(np.float16)
    svh = (rng.standard_normal(n) * 0.05).astype(np.float16)
    return t, suh, svh


@pytest.mark.parametrize("k2", [4, 6, 8, 10])
def test_split_is_the_slice(k2):
    rng = np.random.default_rng(k2)
    D, I = 256, 512                                         # 2 x 2 Hadamard blocks a rank
    gate = _matrix(rng, D, I, k2)
    down = _matrix(rng, I, D, k2)
    full_g = fmt.dequantize(*gate, k2 / 2, "mul1")
    full_d = fmt.dequantize(*down, k2 / 2, "mul1")
    w = I // 2
    for rank in (0, 1):
        g = dx.split_triple(*gate, "col", rank, 2)
        d = dx.split_triple(*down, "row", rank, 2)
        assert g[0].flags["C_CONTIGUOUS"] and d[0].flags["C_CONTIGUOUS"]
        assert np.array_equal(fmt.dequantize(*g, k2 / 2, "mul1"), full_g[:, rank * w:(rank + 1) * w])
        assert np.array_equal(fmt.dequantize(*d, k2 / 2, "mul1"), full_d[rank * w:(rank + 1) * w])


def _expert64(x, mats, k2s, limit=dx.SWIGLU_LIMIT):
    g = fmt.forward(x, *mats["w1"], k2s[0] / 2, "mul1")
    u = fmt.forward(x, *mats["w3"], k2s[1] / 2, "mul1")
    g = np.minimum(g, limit)
    u = np.clip(u, -limit, limit)
    a = g / (1.0 + np.exp(-g)) * u
    return fmt.forward(a, *mats["w2"], k2s[2] / 2, "mul1")


def test_ranks_add_up_to_the_expert():
    rng = np.random.default_rng(5)
    D, I = 256, 512
    k2s = (6, 6, 6)
    mats = {"w1": _matrix(rng, D, I, 6), "w3": _matrix(rng, D, I, 6), "w2": _matrix(rng, I, D, 6)}
    x = rng.standard_normal((3, D)) * 4
    full = _expert64(x, mats, k2s)
    parts = [_expert64(x, dx.split_expert(mats, r, 2), k2s) for r in (0, 1)]
    np.testing.assert_allclose(parts[0] + parts[1], full, rtol=1e-12, atol=1e-12)


def test_split_refuses_partial_blocks():
    rng = np.random.default_rng(0)
    t, su, sv = _matrix(rng, 256, 384, 6)                   # 384 / 2 = 192: not a whole number of 128-blocks
    with pytest.raises(ValueError):
        dx.split_triple(t, su, sv, "col", 0, 2)


# -- tile settings and plans ------------------------------------------------------------------------------------------
def test_tile_settings_are_upstreams():
    pytest.importorskip("torch")
    for geom in (dx.MODEL, dx.DSPARK):
        assert dx.configs(geom) == dx.configs_host(geom) == ((8, 4, 4, 1), (8, 4, 1, 1))


def _shapes(k2_routed: int, k2_shared: int, geom=dx.MODEL, prefix="layers.3.ffn."):
    s = {}
    for e in range(geom.experts):
        for w in ("w1", "w3"):
            s[f"{prefix}experts.{e}.{w}.trellis"] = (320, 144, 8 * k2_routed)
        s[f"{prefix}experts.{e}.w2.trellis"] = (144, 320, 8 * k2_routed)
    for w in ("w1", "w3"):
        s[f"{prefix}shared_experts.{w}.trellis"] = (320, 144, 8 * k2_shared)
    s[f"{prefix}shared_experts.w2.trellis"] = (144, 320, 8 * k2_shared)
    return s


def test_plans_from_header_shapes():
    # layer 3: routed 3-bit (trellis [320, 144, 48] in the checkpoint), shared 5-bit ([.., .., 80])
    p = dx.plan_layer(_shapes(6, 10), prefix="layers.3.ffn.")
    assert p.k2_gu == (6, 10) and p.k2_d == (6, 10)
    assert p.ld_instance() == {"gu": (2, 10), "dn": (2, 10)}
    assert p.entry_bytes(0) == 6_635_520                     # 3 x 5,120 x 1,152 x 3 bits / 8
    assert p.entry_bytes(dx.EXPERTS) == 11_059_200           # the shared expert's half at 5 bits
    assert abs(p.rank_bytes() / 2 ** 30 - 2.383) < 0.01      # GiB a rank, one layer
    # layers 18-22: routed 2-bit, shared 4-bit
    q = dx.plan_layer(_shapes(4, 8, prefix="layers.20.ffn."), prefix="layers.20.ffn.")
    assert q.k2_gu == (4, 8) and q.ld_instance()["gu"] == (2, 10)
    # DSpark: 128 experts top-3 at 4 bits
    m = dx.plan_layer(_shapes(8, 8, dx.DSPARK, "mtp.0.ffn."), dx.DSPARK, prefix="mtp.0.ffn.")
    assert m.ld_instance() == {"gu": (8, 8), "dn": (8, 8)}
    assert dx.DSPARK.slots == 4 and dx.MODEL.slots == 7 and dx.MODEL.table == 385


def test_plan_refuses_wrong_shapes():
    s = _shapes(6, 10)
    s["layers.3.ffn.experts.7.w2.trellis"] = (320, 144, 48)
    with pytest.raises(ValueError):
        dx.plan_layer(s, prefix="layers.3.ffn.")


def test_decode_bytes_model():
    p = dx.plan_layer(_shapes(6, 10), prefix="layers.3.ffn.")
    one = dx.decode_bytes(p, dx.distinct_experts(1))
    assert abs(one - (6 * 6_635_520 + 11_059_200)) < 1
    assert dx.distinct_experts(6) > dx.distinct_experts(4) > 6


# -- the launch sequence ------------------------------------------------------------------------------------------------
class _Rec:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getattr__(self, op):
        def call(*args):
            self.log.append((self.name, op, args))
        return call


def _fake_layer(torch, geom, k2_gu=(6, 10), k2_d=(6, 10)):
    from tensorfold.cuda.exl3.experts import Exl3RoutedExperts

    E, D, I = geom.table, geom.dims, geom.width
    z = lambda *s, dt=torch.float16: torch.zeros(s, dtype=dt)  # noqa: E731
    ex = Exl3RoutedExperts(z(E, dt=torch.int64), z(E, dt=torch.int64), z(E, dt=torch.int64), z(E, dt=torch.int32),
                           z(E, dt=torch.int32), z(E, dt=torch.int32), z(E, D), z(E, D), z(E, I), z(E, I), z(E, I),
                           z(E, D), E, D, I, 2, k2_gu, k2_d, z(E, dt=torch.int64))
    return dx.Experts(ex, geom)


def _run(monkeypatch, torch, on: bool, R=3):
    from tensorfold.cuda.exl3 import experts as x3

    log: list = []
    monkeypatch.setattr(x3, "_ext", lambda: _Rec("up", log))
    monkeypatch.setattr(loads, "_ext", lambda: _Rec("ld", log))
    monkeypatch.setitem(loads.CFG, "on", on)
    monkeypatch.setitem(loads.CFG, "pdl", False)
    layer = _fake_layer(torch, dx.MODEL)
    s = x3.Scratch(layer.ex, 8, dx.MODEL.slots, *dx.configs(dx.MODEL), device="cpu")
    x = torch.zeros((R, dx.DIMS), dtype=torch.bfloat16)
    pick = torch.zeros((R, dx.MODEL.slots), dtype=torch.int32)
    wts = torch.zeros((R, dx.MODEL.slots), dtype=torch.float32)
    out = torch.zeros((R, dx.DIMS), dtype=torch.float32)
    dx.routed(x, pick, wts, layer, s, out, R)
    log_up: list = []
    monkeypatch.setattr(x3, "_ext", lambda: _Rec("up", log_up))
    x3.routed(x, pick, wts, layer.ex, s, out, R, limit=dx.SWIGLU_LIMIT, act_mode=x3.ACT_F32)
    return log, log_up


def _same_args(a, b):
    if len(a) != len(b):
        return False
    for u, v in zip(a, b):
        if hasattr(u, "data_ptr"):
            if not (hasattr(v, "data_ptr") and u.data_ptr() == v.data_ptr() and u.shape == v.shape):
                return False
        elif u != v:
            return False
    return True


def test_routed_is_upstreams_with_loads_off(monkeypatch):
    torch = pytest.importorskip("torch")
    ours, up = _run(monkeypatch, torch, on=False)
    assert [op for _, op, _ in ours] == [op for _, op, _ in up] == \
        ["group", "rot_in", "grouped", "gateup_epilogue", "grouped", "down_combine"]
    for (_, _, a), (_, _, b) in zip(ours, up):
        assert _same_args(a, b)


def test_routed_swaps_only_the_grouped_launches(monkeypatch):
    torch = pytest.importorskip("torch")
    ours, up = _run(monkeypatch, torch, on=True)
    assert [(n, op) for n, op, _ in ours] == [("up", "group"), ("up", "rot_in"), ("ld", "grouped"),
                                              ("up", "gateup_epilogue"), ("ld", "grouped"), ("up", "down_combine")]
    for (n, op, a), (_, _, b) in zip(ours, up):
        if n == "up":
            assert _same_args(a, b)
            continue
        # upstream: (..., slots, cb, nt, w, pf, lo, hi); ours: (..., slots, cb, nt', pd, probe, lo', hi', pdl)
        assert _same_args(a[:17], b[:17])                    # tensors, mats, K, N, P, SK, slots, cb
        assert a[17:20] == (8, 1, 0) and a[20:22] == (2, 10) and a[22] is False


def test_scratch_bytes_match_upstream():
    torch = pytest.importorskip("torch")
    from tensorfold.cuda.exl3 import experts as x3

    layer = _fake_layer(torch, dx.MODEL)
    for rows in (1, 8, 64):
        s = x3.Scratch(layer.ex, rows, dx.MODEL.slots, *dx.configs(dx.MODEL), device="cpu")
        have = sum(t.numel() * t.element_size() for t in (s.xg, s.xu, s.xd, s.z, s.y, s.ids, s.count, s.members_buf))
        assert dx.scratch_bytes(rows) == have
    assert dx.prefill_block_rows(300 << 20) == 512           # a 2,048-row chunk runs its experts in 512-row blocks


def test_with_shared():
    torch = pytest.importorskip("torch")
    pick = torch.tensor([[5, 1, 9, 2, 0, 7]], dtype=torch.int32)
    wts = torch.full((1, 6), 0.25)
    p, w = dx.with_shared(pick, wts)
    assert p.tolist() == [[5, 1, 9, 2, 0, 7, 384]] and w[0, 6].item() == 1.0
