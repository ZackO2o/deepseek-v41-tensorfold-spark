"""engine/serving on the CPU: the layer topology from the checkpoint's config, the pool / slot / snapshot bytes, the
memory budget the plan quotes, the Engram reader (O_DIRECT and buffered paths against the file), the family hook,
and the small exact-drafting helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from engine.serving import app, drafting, engram, family, memory, pool, sessions, state, topology

# config.json's text_config of dsv41-uncensored-2.9bpw (head, read 2026-10-01), the fields the topology reads
TEXT = {
    "model_type": "deepseek_v41_text", "vocab_size": 129280, "hidden_size": 5120, "num_hidden_layers": 40,
    "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64, "sliding_window": 128,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "kv_source_layer_ids": [2, 8, 14, 20], "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512, "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048, "candidate_block_size": 8, "engram_layer_ids": [1, 14],
    "num_nextn_predict_layers": 3, "dspark_target_layer_ids": [37, 38, 39],
}
CONFIG = {"model_type": "deepseek_v41", "quantization_config": {"quant_method": "exl3", "codebook": "mul1"},
          "text_config": TEXT}
GiB = 1 << 30


@pytest.fixture(scope="module")
def topo():
    return topology.build(CONFIG)


def test_roles(topo):
    roles = {ly.index: ly.role for ly in topo.layers}
    assert [i for i, r in roles.items() if r == "swa"] == [0, 1]
    assert topo.by_role("full") == [2, 8, 14, 20]
    assert topo.by_role("reindex") == [24, 28, 32, 36]
    assert len(topo.by_role("reuse")) == 40 - 2 - 4 - 4
    ly = topo.layers
    assert ly[5].kv_source == 2 and ly[5].index_source == 2 and ly[5].ratio == 2
    assert ly[19].kv_source == 14 and ly[21].kv_source == 20 and ly[21].index_source == 20
    assert ly[27].kv_source == 20 and ly[27].index_source == 24 and ly[24].candidates == "read"
    assert ly[20].candidates == "write" and ly[14].engram and ly[1].engram
    assert topo.decoder_start == 20 and all(x.encoder for x in ly[:20]) and not any(x.encoder for x in ly[20:])
    assert [d.index for d in topo.dspark] == [40, 41, 42] and topo.taps == (37, 38, 39)


def test_topology_refuses_inconsistent_configs():
    bad = json.loads(json.dumps(TEXT))
    bad["compress_ratios"][21] = 2                      # a ratio-2 layer reusing layer 20's ratio-1 KV
    with pytest.raises(ValueError):
        topology.build(bad)
    bad = json.loads(json.dumps(TEXT))
    bad["compress_ratios"][5] = 4                       # V4's ratio 4: not a V4.1 layout
    with pytest.raises(ValueError):
        topology.build(bad)


def test_pool_bytes(topo):
    assert pool.bytes_per_token(topo) == pytest.approx(3 * (584 + 256) / 2 + 584 + 256)      # 2,100 B a token
    assert pool.bytes_per_token(topo, "fp8") == pytest.approx(3 * (584 + 132) / 2 + 584 + 132)
    gib = pool.pool_bytes(topo, 4, 300_000) / GiB
    assert 2.3 < gib < 2.4
    assert all(f.page_rows() * f.ratio == pool.PAGE for f in pool.families(topo))
    assert pool.PAGE % pool.GRID == 0


def test_slot_and_snapshot_bytes(topo):
    sb = state.slot_bytes(topo)
    assert sb.swa == 43 * 256 * 584 and sb.carry == 3 * 4096
    assert sb.total < 10 << 20
    full = sessions.snapshot_bytes(topo, 300_000) / GiB
    rep = sessions.snapshot_bytes(topo, 300_000, "replay") / GiB
    assert 0.58 < rep < full < 0.6
    assert sessions.snapshot_point(1000) == 992 and sessions.snapshot_point(1001) == 992
    assert sessions.snapshot_point(1009) == 1008
    codes = {sessions.Tag(p, g, c, k).code() for p, g in (("exact", 0), ("fast", 2048)) for c in sessions.MODES
             for k in ("fp8", "fp4")}
    assert len(codes) == 8


def test_memory_budget(topo):
    b = memory.budget(topo, "mia29")
    assert 4.0 <= b.floor <= 6.0, b                      # the user's floor band at 4 x 300K on the tighter rank
    assert memory.budget(topo, "mia29", rank=1).floor == pytest.approx(b.floor + memory.VISION)
    assert memory.budget(topo, "cool30").floor < 1.0     # coolbho3k-style 3.0: does not fit 4 x 300K at a 4 GiB floor
    assert memory.max_streams(topo, "mia29", 300_000, 4.0) >= 4
    assert memory.max_streams(topo, "cool30", 300_000, 4.0) == 0
    assert memory.max_streams(topo, "exp30", 300_000, 4.0) <= 1
    assert "| mia29 | r0 | 4 x 300K |" in memory.table(topo)


# -- the Engram reader ---------------------------------------------------------------------------------------------
def _shard(dirpath: Path, layer: int, lo: int, n: int, seed: int):
    rows = np.random.default_rng(seed).integers(0, 256, size=(n, engram.ROW_BYTES), dtype=np.uint8)
    path = dirpath / f"engram-l{layer}-r0of2.bin"
    engram.write_shard(path, layer, lo, rows, total=2 * n)
    return path, rows


def _dirs(tmp_path):
    """tmp_path (tmpfs on this box: buffered fallback) and a dir on the home filesystem (O_DIRECT works there)."""

    home = Path.home() / ".cache" / "dsv41-engram-test" / f"{os.getpid()}"
    home.mkdir(parents=True, exist_ok=True)
    yield tmp_path
    yield home
    for p in home.iterdir():
        p.unlink()
    home.rmdir()


def test_engram_reader_paths(tmp_path):
    seen_direct = set()
    for d in _dirs(tmp_path):
        p1, r1 = _shard(d, 1, 1000, 5000, 1)
        p14, r14 = _shard(d, 14, 0, 3000, 2)
        rd = engram.open_rank(d, 0, layers=(1, 14), workers=4, cache_rows=64)
        seen_direct.update(f.direct for f in rd.files.values())
        rng = np.random.default_rng(3)
        for _ in range(5):
            want = rng.integers(1000, 6000, size=150)
            want[:10] = want[10]                         # duplicates
            want[-1] = 5999                              # the file's last row (a sector past the end)
            t = rd.submit(1, want)
            got = rd.result(t)
            assert np.array_equal(got, r1[want - 1000])
            w14 = rng.integers(0, 3000, size=40)
            assert np.array_equal(rd.result(rd.submit(14, w14)), r14[w14])
        assert rd.stats["hits"] > 0                      # the LRU served repeats
        with pytest.raises(ValueError):
            rd.submit(1, np.array([999]))                # not this rank's rows
        rd.close()
    assert True in seen_direct or os.uname().sysname != "Linux"


def test_extents_coalesce_and_cover():
    rows = np.array([0, 1, 2, 10, 11, 5000], dtype=np.int64)
    ext = engram.extents(rows, 0, engram.ROW_BYTES, gap=4096)
    assert len(ext) == 2
    for off, length, rs in ext:
        assert off % engram.SECTOR == 0 and length % engram.SECTOR == 0
        for r in rs.tolist():
            a = engram.HEADER + r * engram.ROW_BYTES
            assert off <= a and a + engram.ROW_BYTES <= off + length
    assert len(engram.extents(rows, 0, engram.ROW_BYTES, gap=0)) == 3


def test_header_checks(tmp_path):
    p, _ = _shard(tmp_path, 1, 0, 10, 0)
    with open(p, "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError):
        engram.read_header(p)


# -- the family hook and small helpers --------------------------------------------------------------------------------
def test_family_check(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    assert family.check(tmp_path) is None
    other = dict(CONFIG, model_type="glm5_next", text_config=dict(TEXT, model_type="glm"))
    (tmp_path / "config.json").write_text(json.dumps(other))
    assert "not deepseek_v41" in family.check(tmp_path)
    with pytest.raises(ValueError):
        family.cuda_engine(tmp_path, tp=1)


def test_accepted_and_effort():
    assert drafting.accepted([7, 1, 2, 3], [1, 2, 9]) == 2
    assert drafting.accepted([7, 1, 2], [5, 2]) == 0
    assert app.effort("low") == 50 and app.effort("max") == 100 and app.effort("37") == 37
    with pytest.raises(ValueError):
        app.effort(0)
