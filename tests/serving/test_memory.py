"""Memory accounting at 4 x 300K: the pool's real allocation equals the plan's formula, four 300K requests reserve
the whole pool and a fifth waits, the floor holds on both ranks with the session store's RAM tier, and the runtime
floor check (GLM 0550 memsafe) admits, waits and logs as it should."""

from __future__ import annotations

import io

import pytest

from engine.serving import memory, state
from engine.serving import pool as pool_mod
from engine.serving.pool import PAGE, Pool
from engine.serving.sessions import Store

GiB = 1 << 30
CTX = 300_000


def test_pool_allocation_is_the_formula(topo):
    tokens = pool_mod.pool_tokens(4, CTX)
    p = Pool(topo, tokens, device="meta")
    assert p.nbytes() == pool_mod.pool_bytes(topo, 4, CTX) + p.page_bytes()       # + GLM's null page
    assert 2.3 < p.nbytes() / GiB < 2.4
    fp8 = Pool(topo, tokens, device="meta", index_kv="fp8")
    assert fp8.nbytes() < p.nbytes()


def test_four_streams_of_300k_reserve_the_pool(topo):
    p = Pool(topo, pool_mod.pool_tokens(4, CTX), device="meta")
    st = Store(p)
    need = p.need_pages(CTX - 2_000, 2_000, CTX + 64)
    assert need == -(-(CTX + pool_mod.SLACK) // PAGE)
    slots = [p.new_slot(CTX + 64) for _ in range(4)]
    for s in slots:
        assert p.available() >= need
        s.reserve(need)
    assert p.available() == 0                       # a fifth 300K stream must wait (or park a session)
    assert st.plan_spill(need) is None
    slots[0].release()
    assert p.available() == need


def test_floor_both_ranks_with_sessions(topo):
    for rank in (0, 1):
        b = memory.load_check(topo, "mia29", rank=rank, streams=4, context=CTX, session_ram_gib=0.25)
        assert b.floor >= 4.0, b
    r0 = memory.budget(topo, "mia29", rank=0)
    assert 5.0 <= r0.floor <= 6.0                   # the plan's 5.13 GiB on the tighter rank
    assert r0.rings * GiB == pytest.approx(4 * state.slot_bytes(topo).total)
    with pytest.raises(ValueError, match="hard"):
        memory.load_check(topo, "cool30", streams=4, context=CTX)


def _mi(free_gib: float, avail_gib: float) -> dict:
    return {"MemTotal": 121 * GiB, "MemFree": int(free_gib * GiB), "MemAvailable": int(avail_gib * GiB),
            "Dirty": 0, "Writeback": 0, "Mapped": 0}


def test_runtime_floor(monkeypatch):
    monkeypatch.setenv("GLM53_TF_ADMIT_CACHE_KEEP_GB", "0")
    reading = {"mi": _mi(1.5, 6.0)}                 # unified memory: little MemFree, page cache to reclaim
    out = io.StringIO()
    from tensorfold.families.glm5_next.spark import memsafe

    f = memory.Floor(5.0, 4.0, meminfo=lambda: reading["mi"], log=memsafe.AdmitLog(out=out), free_gib=1.0)
    assert f.check(0)                               # 6.0 usable >= the 4 GiB hard floor
    assert not f.check(3 * GiB, queued=3)           # 6.0 - 3 < 4: waits, logged
    assert "admission waits for memory (3 queued)" in out.getvalue()
    reading["mi"] = _mi(0.5, 4.5)
    assert not f.check(0) and not f.low()           # 0.5 GiB free right now is under the 1 GiB immediate floor asked
    reading["mi"] = _mi(3.0, 3.5)
    assert f.low() and not f.check(0)
    reading["mi"] = _mi(8.0, 9.0)
    assert f.check(1 * GiB) and "admission resumed" in out.getvalue()
    reading["mi"] = _mi(2.0, 4.5)
    assert f.check(0) and f.stats["under_target"] == 1        # under the 5 GiB target: starts, counted
    assert f.stats["waits"] == 3


def test_cold_boot_page_cache_does_not_stall_admission(monkeypatch):
    """G3's cold boot: MemFree 0.87 GiB beside 6.7 GiB of clean page cache (MemAvailable 9.68, Mapped 2.11). GLM's
    1 GiB immediately-free rule waited 253 s (nothing reclaims page cache while the batcher waits); the default now
    counts the reclaimable cache in full and admits, while a real shortage still waits."""

    for k in ("GLM53_TF_ADMIT_CACHE_KEEP_GB", "TF_DSV41_ADMIT_FREE_FLOOR_GIB", "GLM53_TF_ADMIT_MEM"):
        monkeypatch.delenv(k, raising=False)
    g3 = {"MemTotal": 121 * GiB, "MemFree": int(0.87 * GiB), "MemAvailable": int(9.68 * GiB), "Dirty": 0,
          "Writeback": 0, "Mapped": int(2.11 * GiB)}
    out = io.StringIO()
    from tensorfold.families.glm5_next.spark import memsafe

    f = memory.Floor.from_env(meminfo=lambda: g3, log=memsafe.AdmitLog(out=out), cached=lambda: 0)
    assert f.free_gib == 0.0
    assert f.check(0) and f.stats["waits"] == 0
    old = memory.Floor(5.0, 4.0, meminfo=lambda: g3, log=memsafe.AdmitLog(out=io.StringIO()), free_gib=1.0)
    assert not old.check(0)                                   # the G3 stall, reproduced with the old rule
    short = dict(g3, MemAvailable=int(5.5 * GiB))             # credit 5.5 - 0.87 - 2.11 = 2.5: usable 3.4 < 4
    f2 = memory.Floor.from_env(meminfo=lambda: short, log=memsafe.AdmitLog(out=out), cached=lambda: 0)
    assert not f2.check(0)
    monkeypatch.setenv("TF_DSV41_ADMIT_FREE_FLOOR_GIB", "1")
    assert memory.Floor.from_env(meminfo=lambda: g3, cached=lambda: 0).free_gib == 1.0
    monkeypatch.setenv("TF_DSV41_ADMIT_FREE_FLOOR_GIB", "-1")
    with pytest.raises(ValueError):
        memory.free_floor_gib()


def test_allocator_cache_counts_as_free(monkeypatch):
    monkeypatch.delenv("GLM53_TF_ADMIT_CACHE_KEEP_GB", raising=False)
    mi = {"MemTotal": 121 * GiB, "MemFree": int(0.5 * GiB), "MemAvailable": int(0.5 * GiB), "Dirty": 0,
          "Writeback": 0, "Mapped": 0}
    assert not memory.Floor(5.0, 4.0, meminfo=lambda: mi, cached=lambda: 0).check(0)
    assert memory.Floor(5.0, 4.0, meminfo=lambda: mi, cached=lambda: 4 * GiB).check(0)
    assert memory.allocator_cached() >= 0


def test_floor_settings(monkeypatch):
    monkeypatch.setenv("TF_DSV41_FLOOR_GIB", "6")
    monkeypatch.setenv("TF_DSV41_FLOOR_HARD_GIB", "4.5")
    assert memory.floor_settings() == (6.0, 4.5)
    monkeypatch.setenv("TF_DSV41_FLOOR_HARD_GIB", "7")
    with pytest.raises(ValueError):
        memory.floor_settings()


def test_row_record_sizes_match_the_kernels():
    from engine.kernels.csa2 import rows
    from engine.serving import protocol

    assert (protocol.ROW_BYTES, protocol.VB, protocol.SB) == (rows.ROW_BYTES, rows.VB, rows.SB)


def test_boot_snapshot_and_host_trim():
    snap = memory.snapshot()
    for k in ("MemAvailable", "MemFree", "Cached", "Mapped", "AnonPages", "VmRSS"):
        assert k in snap and snap[k] >= 0
    assert memory.host_trim() >= 0.0


def test_measured_projection_4x300k(topo):
    """ENGINE-PLAN section 5.0: the worker is the tighter rank; with the shared graph pool (0.4) and FP8 index keys
    4 x 300K sits at the 5 GiB target there (5.0, no margin); with bf16 keys or the plan's 1.2 GiB graph budget it
    holds only the 4 GiB stop."""

    fp8 = memory.measured(topo, rank=1, context=CTX, index_kv="fp8")
    bf16 = memory.measured(topo, rank=1, context=CTX, index_kv="bf16")
    worst = memory.measured(topo, rank=1, context=CTX, index_kv="fp8", graphs=memory.GRAPHS_WORST)
    head = memory.measured(topo, rank=0, context=CTX, index_kv="fp8")
    assert fp8.floor == pytest.approx(5.0, abs=0.05) and head.floor > 6.0
    assert 4.0 <= bf16.floor < 4.9 and worst.floor >= 4.0
    assert fp8.floor - bf16.floor == pytest.approx(0.35, abs=0.02)
    assert fp8.terms["experts"] < 0.6 and 0.05 < memory.engram_cache_gib() < 0.15
    fit = memory.fit_context(topo, 5.0, index_kv="fp8", graphs=memory.GRAPHS_WORST)
    assert 150_000 < fit < CTX and memory.fit_context(topo, 5.0, index_kv="fp8") >= 290_000
    assert "**" in memory.measured_table(topo)
