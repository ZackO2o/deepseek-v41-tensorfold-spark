"""Checks against the real checkpoint.

- Always: the shipped ``config.json`` parses into the defaults the reference assumes.
- When the headers are cached (``.cache/ckpt/<pack>/headers.json``, written by any remote run, e.g.
  ``tests/reference/real_layers.py``): every tensor the loader needs exists with the shape it expects, every EXL3
  group is mul1 with the width the kit's ``exl3_k_map.json`` predicts, the Engram tables have the rows the prime
  layout gives.
- With ``DSV41_REAL=1`` (reads tensors over ssh from head, read-only): an EXL3 matrix against its original FP8
  weights, and one real block on a few tokens.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from engine.reference.config import Config
from engine.reference.engram import EngramLayout
from engine.reference.loader import parse_header

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
CACHE = ROOT / ".cache" / "ckpt"
PACK = CACHE / "dsv41-uncensored-2.9bpw" / "headers.json"
ENGRAM = CACHE / "dsv41-engram-src" / "headers.json"


def test_shipped_config_is_the_default_config():
    got = Config.from_file(DATA / "config.json")
    assert got == Config(), [f for f in Config.__dataclass_fields__ if getattr(got, f) != getattr(Config(), f)]


@pytest.fixture(scope="module")
def headers():
    if not PACK.exists():
        pytest.skip("no cached checkpoint headers (run tests/reference/real_layers.py once)")
    m = json.loads(PACK.read_text())
    out = {}
    for f, v in m["headers"].items():
        out.update(parse_header(v["header"], f, v["header_len"]))
    return out


def _groups(h):
    g = defaultdict(dict)
    for n, i in h.items():
        p, _, part = n.rpartition(".")
        if part in ("trellis", "suh", "svh", "mul1", "mcg"):
            g[p][part] = i
    return g


def _kn(h, p):
    s = h[f"{p}.trellis"].shape
    return 16 * s[0], 16 * s[1], s[2] / 16


def test_every_exl3_group_is_well_formed_mul1(headers):
    groups = _groups(headers)
    assert len(groups) == 47900            # DSV41-BASELINE.md: 47,900 packed matrices
    for p, parts in groups.items():
        assert set(parts) == {"trellis", "suh", "svh", "mul1"}, p
        k, n = 16 * parts["trellis"].shape[0], 16 * parts["trellis"].shape[1]
        assert parts["suh"].shape == (k,) and parts["svh"].shape == (n,) and parts["mul1"].shape == ()


def test_shapes_and_topology(headers):
    cfg = Config()
    h = headers
    d, hd = cfg.hidden_size, cfg.head_dim
    for L in range(cfg.num_hidden_layers + cfg.num_nextn_predict_layers):
        p = f"layers.{L}" if L < cfg.num_hidden_layers else f"mtp.{L - cfg.num_hidden_layers}"
        a = f"{p}.attn"
        assert _kn(h, f"{a}.wq_a")[:2] == (d, cfg.q_lora_rank)
        assert _kn(h, f"{a}.wkv")[:2] == (d, hd)
        assert _kn(h, f"{a}.wq_b")[:2] == (cfg.q_lora_rank, cfg.num_attention_heads * hd)
        for g in range(cfg.o_groups):
            assert _kn(h, f"{a}.wo_a.slice.{g}")[:2] == (cfg.num_attention_heads * hd // cfg.o_groups,
                                                          cfg.o_lora_rank)
        assert _kn(h, f"{a}.wo_b")[:2] == (cfg.o_groups * cfg.o_lora_rank, d)
        assert h[f"{a}.attn_sink"].shape == (cfg.num_attention_heads,)
        assert h[f"{p}.hc_attn_fn"].shape == (24, 4 * d)
        assert (f"{a}.compressor.wkv.trellis" in h) == cfg.is_kv_source(L), L
        assert (f"{a}.compressor.wgate.trellis" in h) == (cfg.is_kv_source(L) and cfg.compress_ratio(L) == 2), L
        assert (f"{a}.indexer.wq_b.trellis" in h) == cfg.is_index_source(L), L
        assert (f"{a}.indexer.wk.trellis" in h) == cfg.is_kv_source(L), L
        if cfg.is_index_source(L):
            assert h[f"{a}.indexer.weights_proj.weight"].shape == (cfg.index_n_heads, d)
            assert _kn(h, f"{a}.indexer.wq_b")[:2] == (cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim)
        if cfg.is_kv_source(L):
            assert _kn(h, f"{a}.indexer.wk")[:2] == (hd, cfg.index_head_dim)
        assert (f"{p}.engram.wkv.trellis" in h) == (L in cfg.engram_layer_ids)
        n_exp, _ = cfg.experts_of(L)
        assert h[f"{p}.ffn.gate.weight"].shape == (n_exp, d)
        assert f"{p}.ffn.experts.{n_exp - 1}.w1.trellis" in h and f"{p}.ffn.experts.{n_exp}.w1.trellis" not in h
        assert _kn(h, f"{p}.ffn.experts.0.w1")[:2] == (d, cfg.moe_intermediate_size)
        assert _kn(h, f"{p}.ffn.experts.0.w2")[:2] == (cfg.moe_intermediate_size, d)
    assert _kn(h, "layers.1.engram.wkv")[:2] == (cfg.n_hash_cols * cfg.engram_head_dim, (cfg.hc_mult + 1) * d)
    assert _kn(h, "head")[:2] == (d, cfg.vocab_size)
    assert _kn(h, "mtp.0.main_proj")[:2] == (3 * d, d)
    assert h["mtp.2.confidence_head.proj.weight"].shape == (1, d + cfg.dspark_markov_rank)
    assert h["layers.1.engram.q_weight"].shape == (cfg.hc_mult, d)


def test_widths_follow_the_kit_k_map(headers):
    km = json.loads((DATA / "exl3_k_map.json").read_text())
    h = headers
    bits = {p: _kn(h, p)[2] for p in _groups(h)}
    for p, b in bits.items():
        m = re.match(r"layers\.(\d+)\.(.*)", p)
        if p.startswith("mtp."):
            assert b == km["mtp_bits"], p
        elif p == "head":
            assert b == km["head_bits"]
        elif ".ffn.experts." in p:
            assert b == km["routed"][m.group(1)], p
        elif ".shared_experts." in p and m.group(1) != "29":
            assert b == km["shared"][m.group(1)], p
        elif ".engram." in p:
            assert b == km["engram"][m.group(1)], p
        elif ".indexer.wk" in p:
            assert b == km["indexer_wk_bits"], p
    # the K map's "attn_default 5" is not the whole story: layer 0's wq_a / wkv are 6-bit
    assert bits["layers.0.attn.wq_a"] == 6 and bits["layers.0.attn.wkv"] == 6 and bits["layers.1.attn.wq_a"] == 5
    assert {bits[f"layers.29.ffn.shared_experts.w{j}"] for j in (1, 2, 3)} == {4, 5}


def test_engram_tables_match_the_prime_layout():
    if not ENGRAM.exists():
        pytest.skip("no cached Engram source headers")
    m = json.loads(ENGRAM.read_text())
    h = {}
    for f, v in m["headers"].items():
        h.update(parse_header(v["header"], f, v["header_len"]))
    cfg = Config()
    lay = EngramLayout(cfg)
    for i, L in enumerate(cfg.engram_layer_ids):
        w, s = h[f"layers.{L}.engram.embed.weight"], h[f"layers.{L}.engram.embed.scale"]
        assert w.shape == (lay.total_rows(i), cfg.engram_head_dim) and w.dtype == "F8_E4M3"
        assert s.shape == (lay.total_rows(i), cfg.engram_head_dim // 32) and s.dtype == "F8_E8M0"


real = pytest.mark.skipif(os.environ.get("DSV41_REAL") != "1", reason="set DSV41_REAL=1 (ssh to head, read-only)")


@real
def test_exl3_dequant_tracks_the_original_fp8_weights():
    import importlib.util
    spec = importlib.util.spec_from_file_location("real_layers", Path(__file__).parent / "real_layers.py")
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    from engine.reference.loader import CheckpointLoader, RemoteSafetensorsDir
    src = RemoteSafetensorsDir(rl.HOST, rl.MODEL, CACHE / "dsv41-uncensored-2.9bpw")
    eng = RemoteSafetensorsDir(rl.HOST, rl.ENGRAM, CACHE / "dsv41-engram-src")
    r = rl.check_exl3_against_fp8(CheckpointLoader(Config(), src), eng)
    assert r["bits"] == 5 and 0.02 < r["rel_err"] < 0.06 and r["row_cos_min"] > 0.99, r


@real
def test_one_real_block_runs():
    from engine.reference.attention import CsaState
    from engine.reference.loader import CheckpointLoader, RemoteSafetensorsDir
    from engine.reference.model import Block
    from engine.reference.ops import Numerics, bf16
    cfg = Config()
    src = RemoteSafetensorsDir("<head-ssh>", "~/models/dsv41-uncensored-2.9bpw",
                               CACHE / "dsv41-uncensored-2.9bpw")
    ld = CheckpointLoader(cfg, src, Numerics.kit())
    ids = torch.tensor([0, 671, 6102, 294])
    h = bf16(ld.embed(ids))
    streams = h.unsqueeze(1).expand(-1, 4, -1).contiguous()
    lw = ld.layer(0)
    out, pre = Block(cfg, 0, lw, Numerics.kit())(streams, None, torch.arange(4), CsaState())
    assert torch.isfinite(out).all() and pre.shape == (4, 4)
    assert 0.01 < float(out.pow(2).mean().sqrt()) < 100
