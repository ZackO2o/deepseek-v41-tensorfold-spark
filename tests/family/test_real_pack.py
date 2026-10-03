"""The family's loader, config and Engram host side on the real 2.9 bpw pack, offline: the cached safetensors
headers (``.cache/ckpt/dsv41-uncensored-2.9bpw/headers.json``, read from head read-only), its config.json and
tokenizer.json. The loader runs on meta tensors (no weight data): every tensor it needs exists with the shape and
width it expects, both ranks' trees have the TP shapes of docs/ARCHITECTURE.md section 7, and their bytes match the
memory budget. Skipped where the cache is absent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PACK = Path(__file__).resolve().parents[2] / ".cache" / "ckpt" / "dsv41-uncensored-2.9bpw"
pytestmark = pytest.mark.skipif(not (PACK / "headers.json").is_file(), reason="the real pack's cached headers")


class MetaShards:
    """``weights.Shards`` over the cached headers: meta tensors of the right dtype and shape; the EXL3 codebook
    markers (a scalar each) carry the mul1 value the pack stores."""

    def __init__(self, root: Path) -> None:
        from tensorfold.families.deepseek_v41.cuda import weights as W

        self.W = W
        m = json.loads((root / "headers.json").read_text())
        self.info_ = {n: (e["dtype"], tuple(e["shape"])) for f, v in m["headers"].items()
                      for n, e in v["header"].items() if n != "__metadata__"}
        self.root = root

    def __contains__(self, name: str) -> bool:
        return name in self.info_

    def info(self, name: str):
        dt, shape = self.info_[name]
        return self.W.Info("", dt, shape, 0, 0)

    def tensor(self, name: str) -> torch.Tensor:
        dt, shape = self.info_[name]
        if name.endswith(".mul1"):
            return torch.tensor(0x83DCD12D - 2 ** 32, dtype=torch.int32)
        return torch.empty(shape, dtype=self.W.DTYPES[dt], device="meta")

    def rows(self, name: str, lo: int, hi: int) -> torch.Tensor:
        dt, shape = self.info_[name]
        return torch.empty((hi - lo, *shape[1:]), dtype=self.W.DTYPES[dt], device="meta")


@pytest.fixture(scope="module")
def cfg():
    from tensorfold.families.deepseek_v41.cuda.config import Config

    return Config.from_file(PACK / "files" / "config.json")


def test_family_accepts_the_pack(cfg, tmp_path):
    from tensorfold.families import deepseek_v41 as family

    (tmp_path / "config.json").write_text((PACK / "files" / "config.json").read_text())
    assert family.check(tmp_path) is None
    assert (cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers, cfg.head_dim) == (129280, 5120, 40, 512)
    assert cfg.compress_ratio(20) == 1 and cfg.rope.yarn and cfg.rope.factor == 16


def _bytes(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_bytes(v) for v in x)
    if hasattr(x, "__dataclass_fields__"):
        return sum(_bytes(v) for v in vars(x).values())
    return 0


@pytest.mark.parametrize("rank", [0, 1])
def test_loader_on_the_real_headers(cfg, rank):
    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.families.deepseek_v41.cuda import weights as W

    tree = W.Loader(cfg, MetaShards(PACK), rank, 2, "meta").load()
    assert tree.embed.shape == (64640, 5120) and tree.head.n == 64640 and tree.vocab_lo == rank * 64640
    bits = lambda x: fmt.bits_of(tuple(x.trellis.shape[-3:]))  # noqa: E731
    for L, ly in enumerate(tree.layers):
        a, m = ly.attn, ly.moe
        assert (a.wq_a.k, a.wq_a.n, a.wkv.n) == (5120, 1280, 512)
        assert bits(a.wq_a) == (6 if L == 0 else 5) and bits(a.wkv) == (6 if L == 0 else 5)
        assert a.wq_b.n == 16384 and a.wo_b.k == 4096 and len(a.wo_a) == 4 and a.sink.shape == (32,)
        assert m.w1.trellis.shape == (384, 320, 72, 8 * (4 if 18 <= L <= 22 else 6))
        assert m.w2.trellis.shape[:3] == (384, 72, 320) and m.gate.dtype == torch.bfloat16
        assert m.shared[0][0].n == 1152 and m.shared[0][1].k == 1152
        mode = cfg.attention_mode(L)
        assert (a.comp_wkv is not None) == (mode == "full") and (a.ix_wk is not None) == (mode == "full")
        assert (a.comp_wgate is not None) == (mode == "full" and L != 20)
        assert (a.ix_wq_b is not None) == (mode in ("full", "reindex"))
        if mode == "full":
            assert bits(a.ix_wk) == 8 and a.ix_wk.n == 128 and a.ix_wp.shape == (32, 5120)
        assert (ly.engram is not None) == (L in (1, 14))
    assert bits(tree.layers[1].engram.wkv) == 5 and bits(tree.layers[14].engram.wkv) == 4
    assert tree.layers[1].engram.wkv.n == 25600 and tree.layers[1].engram.wkv.k == 6144
    gib = _bytes(tree) / 2 ** 30
    # docs/ARCHITECTURE.md section 7: 98.0 GiB a rank without vision, of which DSpark 3.47 (not loaded in M1):
    # 94.5, + the replicated Engram wkv half (0.08) and the token map: 94.8
    assert 94.0 < gib < 95.5, gib


def test_engram_layout_and_token_map(cfg):
    from tensorfold.families.deepseek_v41.cuda import engram_host as EH

    lay = EH.EngramLayout(cfg)
    assert [lay.total_rows(i) for i in range(2)] == list(cfg.engram_num_embeddings)
    lo, hi, h0, nh = lay.head_shard(0, 1, 2)
    assert (h0, nh) == (12, 12) and hi == lay.total_rows(0)
    lookup, n = EH.token_map(PACK / "files" / "tokenizer.json")
    assert n == cfg.engram_compressed_vocab_size == 99092 and len(lookup) >= cfg.vocab_size - 64
