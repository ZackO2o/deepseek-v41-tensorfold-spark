"""x3ld's load path gives upstream's grouped_kernel operands, bit for bit (no GPU): every width K2 = 2..16, every
setting (nt 4 / 8, pd 1 / 2), the real K ranges of this model's experts, live and dead member rows; every v4 load
16-byte aligned and inside its k step; injected faults caught (negative controls)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.kernels.exl3 import loads

from x3ld_emu import ld_stream, same, upstream_stream

K2S = list(range(2, 17))


def _case(k2: int, NT: int, nkt: int, seed: int, dead: int = 3):
    rng = np.random.default_rng(seed)
    tw = 4 * k2
    NTILES = 3 * NT
    KT = nkt + 5
    T = rng.integers(0, 2 ** 32, size=KT * NTILES * tw, dtype=np.uint64).astype(np.uint32)
    K = KT * 16
    X = rng.integers(0, 2 ** 16, size=40 * K + 2, dtype=np.uint64).astype(np.uint16)
    rows = rng.permutation(40)[:16].astype(np.int64)
    rows[16 - dead:] = -1                                   # dead member rows (the tile's tail)
    return T, NTILES, 2, nkt, NT, X, K, rows[:8], rows[8:]


@pytest.mark.parametrize("k2", K2S)
@pytest.mark.parametrize("nt,pd", [(8, 1), (8, 2), (4, 2)])
def test_operands_equal_upstream(k2, nt, pd):
    for nkt in sorted({pd, 2 * pd, 6, 18, 20} - {x for x in (6, 18, 20) if x % pd}):
        T, NTILES, kt0, nkt_, NT, X, K, r0, r1 = _case(k2, nt, nkt, seed=k2 * 100 + nkt)
        want = upstream_stream(T, NTILES, kt0, nkt_, NT, X, K, r0, r1, k2, NT)
        got = ld_stream(T, NTILES, kt0, nkt_, NT, X, K, r0, r1, k2, NT, pd)
        assert same(want, got), (k2, nt, pd, nkt)


@pytest.mark.parametrize("k2", K2S)
@pytest.mark.parametrize("nt,pd", [(8, 1), (8, 2), (4, 2)])
def test_loads_aligned_and_in_step(k2, nt, pd):
    nkt = 4
    T, NTILES, kt0, _, NT, X, K, r0, r1 = _case(k2, nt, nkt, seed=7)
    got: list = []
    ld_stream(T, NTILES, kt0, nkt, NT, X, K, r0, r1, k2, NT, pd, loads=got)
    tw = 4 * k2
    words = NT * tw
    seen = {}
    for off, s in got:
        base = ((kt0 + s) * NTILES + NT) * tw
        assert off % 4 == 0, "16-byte alignment"
        assert base <= off and off + 4 <= base + words, "inside the step"
        seen.setdefault(s, set()).update(range(off, off + 4))
    for s in range(nkt):                                    # every word of every step read exactly as needed
        base = ((kt0 + s) * NTILES + NT) * tw
        assert seen[s] == set(range(base, base + words))


@pytest.mark.parametrize("bug", ["lane_major", "refill_before_store", "a_wrong_step", "tiles_reversed"])
def test_negative_controls(bug):
    caught = 0
    for k2, nt, pd in ((6, 8, 1), (4, 8, 2), (10, 4, 2), (8, 8, 2)):
        T, NTILES, kt0, nkt, NT, X, K, r0, r1 = _case(k2, nt, 6, seed=1)
        want = upstream_stream(T, NTILES, kt0, nkt, NT, X, K, r0, r1, k2, NT)
        got = ld_stream(T, NTILES, kt0, nkt, NT, X, K, r0, r1, k2, NT, pd, bugs={bug})
        caught += not same(want, got)
    assert caught >= 3, bug


def test_unguarded_tail_load_is_caught():
    """K2 = 6, NT = 8: a step is 192 words, so lanes 16-31 of the second v4 round must not load."""

    T, NTILES, kt0, nkt, NT, X, K, r0, r1 = _case(6, 8, 2, seed=3)
    got: list = []
    ld_stream(T, NTILES, kt0, nkt, NT, X, K, r0, r1, 6, NT, 1, bugs={"no_guard"}, loads=got)
    words = NT * 24
    out = [off for off, s in got if off + 4 > ((kt0 + s) * NTILES + NT) * 24 + words]
    assert out, "the bounds check must see the unguarded loads"


# -- the real shapes -------------------------------------------------------------------------------------------------
def test_real_k_ranges():
    """Gate/up 5,120 -> 1,152 (4 K splits, 4 warps: 20 k steps a warp), down 1,152 -> 5,120 (1 split: 18): every
    setting fits gate/up; pd 2 fits down (18 % 2 == 0)."""

    for cfg in loads.CFGS:
        assert loads.fits(5120, 1152, 4, 4, cfg)
        assert loads.fits(1152, 5120, 1, 4, cfg)
    assert not loads.fits(1152, 5120, 1, 4, (8, 4)) if (8, 4) in loads.CFGS else True
    assert not loads.fits(1152, 5120, 1, 8, (8, 1)), "warps other than upstream's 4 change the K ranges"


def test_k2_ranges():
    assert loads.k2_range(6, 10) == (2, 10)          # routed 3-bit + shared 5-bit
    assert loads.k2_range(4, 8) == (2, 10)           # layers 18-22 (2-bit) + shared 4-bit
    assert loads.k2_range(8, 8) == (8, 8)            # DSpark, 4-bit throughout
    assert loads.k2_range(6, 12) is None             # 6-bit: upstream's kernel (2..16 instance) runs it


def test_knobs():
    assert loads.parse({}) == {"on": False, "gu": (8, 1), "dn": (8, 1), "pdl": False}
    assert loads.parse({"TF_DSV41_EXPERT_LOADS": "1", "TF_DSV41_EXPERT_LOADS_CFG": "8,2/4,2",
                        "TF_DSV41_EXPERT_LOADS_PDL": "1"}) == {"on": True, "gu": (8, 2), "dn": (4, 2), "pdl": True}
    for bad in ("8", "8,3", "16,1", "8,1/8,1/8,1", "x,y"):
        with pytest.raises(ValueError):
            loads.parse({"TF_DSV41_EXPERT_LOADS_CFG": bad})
