"""The family's forward (TensorFold branch dsv41-060, the CPU path: torch twins of the kernels) against
engine/reference on the same synthetic EXL3 checkpoint, read by each side's own loader:

- exact numerics (no bf16 / fp16 / FP8 roundings on either side, float64 sums): per layer kind (SWA, SWA + Engram,
  ratio-2 kv source with compressor and indexer, its reuse layer, ratio-1 kv source with the candidate blocks, its
  reuse layer, reindex over the candidates, reuse of the reindex selection) and the whole model's logits agree to
  1e-12, routing identical; one rank == TP=2 == the reference; whole prompt == windows of 8 == one row at a time;
- the kit's numerics (bf16 activations, fp16 linear inputs, FP8 KV rows; the engine's bf16 index keys): the logits
  of most rows agree to bf16 rounding noise and the argmax agrees (a rounding can flip a near-tie of a top-k).
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from exact_reference import exact_reference  # noqa: E402

KINDS = ["swa", "swa+engram", "full r2 (compressor, indexer)", "reuse r2", "full r1 + candidates", "reuse r1",
         "reindex", "reuse of reindex"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import dsv41_fakes as FK
    from safetensors.torch import save_file

    from tensorfold.families.deepseek_v41.cuda.config import Config

    root = FK.write_checkpoint(tmp_path_factory.mktemp("ck"))
    raw = json.loads((root / "config.json").read_text())
    cfg = Config.from_dict(raw)
    table = FK.engram_table(cfg)
    eroot = tmp_path_factory.mktemp("eng")
    save_file({k: v for L, (r, s) in table.items() for k, v in
               ((f"layers.{L}.engram.embed.weight", r), (f"layers.{L}.engram.embed.scale",
                                                        s.view(torch.float8_e8m0fnu)))}, str(eroot / "e.safetensors"))
    ids = torch.randint(0, cfg.vocab_size, (29,), generator=torch.Generator().manual_seed(3))
    return root, eroot, raw, cfg, table, FK.token_map(cfg), ids


def reference(env, numerics, n_layers=None):
    from engine.reference.config import Config as RConfig
    from engine.reference.engram import NgramHasher
    from engine.reference.loader import CheckpointLoader, SafetensorsDir
    from engine.reference.model import Model

    root, eroot, raw, _, _, tmap, ids = env
    rcfg = RConfig.from_dict(raw)
    ld = CheckpointLoader(rcfg, SafetensorsDir(root), numerics, engram_src=SafetensorsDir(eroot))
    return Model(rcfg, ld.model_weights(), numerics, NgramHasher(rcfg, tmap)).forward(
        ids, n_layers=n_layers, keep_layer_out=tuple(range(8)), with_logits=n_layers is None)


def ours(env, windows, layers=None, rank=0, world=1, comm=None):
    from tensorfold.families.deepseek_v41.cuda import engram_host as EH
    from tensorfold.families.deepseek_v41.cuda import forward as FW
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.csa2.twin import Twin

    root, _, _, cfg, table, tmap, ids = env
    tree = W.Loader(cfg, W.Shards(root), rank, world).load(layers, token_map=tmap)
    rows = {L: EH.MemoryRows(r, s) for L, (r, s) in table.items()}
    fw = FW.Forward(cfg, tree, csa2=Twin(cfg.qk_rope_head_dim, cfg.sliding_window), comm=comm, limit=64,
                    chunk=max(windows), engram_rows=rows, hasher=EH.NgramHasher(cfg, tmap))
    fw.trace = {}
    logits, traces, a = [], [], 0
    for n in windows:
        logits.append(fw.step(ids[a:a + n].tolist()))
        traces.append(dict(fw.trace))
        a += n
    streams = {L: torch.cat([t[L] for t in traces]) for L in traces[0]}
    return torch.cat(logits), streams, fw


def rel(a, b) -> float:
    return float((a.double() - b.double()).abs().max() / max(1.0, float(b.double().abs().max())))


@pytest.mark.parametrize("k", range(1, 9), ids=KINDS)
def test_each_layer_kind_exact(env, k):
    from engine.reference.ops import Numerics
    from tensorfold.families.deepseek_v41.cuda import numerics as N

    with exact_reference():
        ref = reference(env, Numerics.exact(), n_layers=k)
    with N.exact():
        _, streams, fw = ours(env, [29], layers=k)
    assert rel(streams[k - 1], ref.layer_out[k - 1]) < 1e-12, KINDS[k - 1]
    got = fw.blocks[k - 1].moe.last_pick[:, :-1].sort(-1).values
    assert torch.equal(got.long(), ref.routing[k - 1].sort(-1).values.long())


@pytest.mark.parametrize("windows", [[29], [8, 8, 8, 5], [1] * 29], ids=["prompt", "windows of 8", "one row"])
def test_whole_model_exact(env, windows):
    from engine.reference.ops import Numerics
    from tensorfold.families.deepseek_v41.cuda import numerics as N

    with exact_reference():
        ref = reference(env, Numerics.exact())
    with N.exact():
        logits, streams, _ = ours(env, windows)
    assert rel(logits, ref.logits) < 1e-12
    for L in range(8):
        assert rel(streams[L], ref.layer_out[L]) < 1e-12, L


def test_tp2_exact(env):
    import dsv41_fakes as FK
    from engine.reference.ops import Numerics
    from tensorfold.families.deepseek_v41.cuda import numerics as N

    with exact_reference():
        ref = reference(env, Numerics.exact())
    pair = FK.ThreadPair()
    comms = pair.comms()
    with N.exact():
        halves = pair.run([lambda r=r: ours(env, [8, 8, 8, 5], rank=r, world=2, comm=comms[r])[0] for r in range(2)])
    assert rel(torch.cat(halves, 1), ref.logits) < 1e-12


def test_kit_numerics(env):
    from engine.reference.ops import Numerics

    ref = reference(env, Numerics(exl3_fp16_input=True, kv_fp8_ds_mla=True, indexer_fp8=False))
    logits, _, _ = ours(env, [8, 8, 8, 5])
    assert (logits.argmax(-1) == ref.logits.argmax(-1)).float().mean() >= 0.95
    # bf16 / FP8 rounding noise through 8 layers; a near-tie in a top-k (index selection, routing) can flip for a
    # row and move it further, so the bound is on most rows, not all (0.85 since G2 removed the per-head q norm: the
    # un-normalised queries sharpen the attention, 26 of 29 rows under 3%; the exact-numerics tests stay at 1e-12)
    row = (logits.double() - ref.logits.double()).abs().amax(-1) / ref.logits.double().abs().amax(-1)
    assert float(row.median()) < 0.01 and float((row < 0.03).double().mean()) >= 0.85
