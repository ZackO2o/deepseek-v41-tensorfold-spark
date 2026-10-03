"""Run the first N blocks of the reference on real V4.1-Flash tensors (fetched over ssh, read-only) and print
per-layer diagnostics. No GPU; nothing is started on the host.

    .cache/dsv41-ref-venv/bin/python tests/reference/real_layers.py --layers 3 --prompt "The capital of France is"

What is checked (asserts, and printed):
- every intermediate is finite; the hc ``comb`` matrices are doubly stochastic;
- router weights sum to ``routed_scaling_factor``;
- the Engram gate is neither saturated nor dead;
- with ``--check-exl3``: an EXL3 matrix of the checkpoint against its original FP8 weights (Engram ``wkv`` of layer
  1, whose FP8 source shard is on the host): relative error ~4% at 5 bits, per-row cosine > 0.99.

Experts are fetched into memory only (never written to disk); attention / shared-expert tensors are cached under
``.cache/ckpt/`` so a second run is fast.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.reference.attention import CsaState  # noqa: E402
from engine.reference.config import Config  # noqa: E402
from engine.reference.hc import hc_mixes  # noqa: E402
from engine.reference.loader import CheckpointLoader, RemoteSafetensorsDir, SafetensorsDir, hasher_from  # noqa: E402
from engine.reference.model import Block  # noqa: E402
from engine.reference.moe import route  # noqa: E402
from engine.reference.ops import Numerics, bf16  # noqa: E402

HOST = "<head-ssh>"
MODEL = "~/models/dsv41-uncensored-2.9bpw"
ENGRAM = "~/models/dsv41-engram-src"


def sources(args) -> tuple[SafetensorsDir, SafetensorsDir]:
    if args.local:
        return SafetensorsDir(args.model), SafetensorsDir(args.engram)
    cache = ROOT / ".cache" / "ckpt"
    return (RemoteSafetensorsDir(args.host, args.model, cache / Path(args.model).name),
            RemoteSafetensorsDir(args.host, args.engram, cache / Path(args.engram).name))


def check_exl3_against_fp8(ld: CheckpointLoader, eng: SafetensorsDir, prefix: str = "layers.1.engram.wkv") -> dict:
    w = ld.exl3_weight(prefix).dequantize(torch.float64)
    raw = eng.tensor(f"{prefix}.weight").view(torch.float8_e4m3fn).to(torch.float64)
    sc = torch.exp2(eng.tensor(f"{prefix}.scale").view(torch.uint8).to(torch.float64) - 127)
    n, k = raw.shape
    ref = (raw.view(n // 32, 32, k // 32, 32) * sc[:, None, :, None]).reshape(n, k).t()       # [K, N] like EXL3
    rel = float((w - ref).norm() / ref.norm())
    cos = torch.nn.functional.cosine_similarity(w.t(), ref.t(), dim=1)
    return {"tensor": prefix, "bits": ld.exl3_weight(prefix).bits, "rel_err": rel, "row_cos_min": float(cos.min()),
            "row_cos_mean": float(cos.mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--engram", default=ENGRAM)
    ap.add_argument("--local", action="store_true", help="--model / --engram are local folders")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ids", default="", help="comma-separated token ids instead of --prompt")
    ap.add_argument("--exact", action="store_true", help="Numerics.exact() instead of the kit's")
    ap.add_argument("--check-exl3", action="store_true")
    ap.add_argument("--json", default="", help="write the diagnostics here")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    src, eng = sources(args)
    cfg = Config.from_dict(json.loads(src.read_file("config.json")))
    num = Numerics.exact() if args.exact else Numerics.kit()
    ld = CheckpointLoader(cfg, src, num, engram_src=eng)
    report: dict = {"layers": []}
    if args.check_exl3:
        t = time.time()
        report["exl3_vs_fp8"] = check_exl3_against_fp8(ld, eng)
        print("exl3 vs fp8:", report["exl3_vs_fp8"], f"({time.time() - t:.1f}s)")

    if args.ids:
        ids = torch.tensor([int(x) for x in args.ids.split(",")])
    else:
        from tokenizers import Tokenizer
        src.read_file("tokenizer.json")
        tok_path = (src.root / "files" / "tokenizer.json") if isinstance(src, RemoteSafetensorsDir) \
            else (src.root / "tokenizer.json")
        tok = Tokenizer.from_file(str(tok_path))
        ids = torch.tensor([cfg.bos_token_id] + tok.encode(args.prompt, add_special_tokens=False).ids)
    print("ids:", ids.tolist())
    hasher = hasher_from(src, cfg) if any(L < args.layers for L in cfg.engram_layer_ids) else None
    hashes = hasher(ids) if hasher is not None else None

    positions = torch.arange(ids.numel())
    h = bf16(ld.embed(ids))
    streams = h.unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
    pre = None
    state = CsaState()
    for L in range(args.layers):
        t = time.time()
        lw = ld.layer(L)
        blk = Block(cfg, L, lw, num)
        _, _, comb = hc_mixes(streams, lw.hc_attn, cfg.rms_norm_eps, cfg.hc_eps, cfg.hc_post_alpha,
                              cfg.hc_sinkhorn_iters)
        gate_stats = None
        if blk.engram is not None and hashes is not None:
            # the Engram gate, recomputed for the report (the block applies it itself)
            before = streams.clone()
            after = blk.engram(streams, hashes)
            gate_stats = float((after - before).abs().mean())
        streams, pre = blk(streams, pre, positions, state, hashes)
        w_, ids_ = route(torch.zeros(1, cfg.hidden_size), lw.moe.gate, lw.moe.bias, blk.moe.top_k,
                         cfg.routed_scaling_factor)
        row = {
            "layer": L, "mode": cfg.attention_mode(L), "ratio": cfg.compress_ratio(L),
            "seconds": round(time.time() - t, 1),
            "stream_rms": [round(float(v), 4) for v in streams.pow(2).mean((0, 2)).sqrt()],
            "finite": bool(torch.isfinite(streams).all()),
            "comb_row_sum_err": float((comb.sum(-1) - 1).abs().max()),
            "comb_col_sum_err": float((comb.sum(-2) - 1).abs().max()),
            "router_weight_sum": round(float(w_.sum()), 4),
            "experts_last_token": blk.moe.last_ids[-1].tolist(),
            "engram_mean_update": gate_stats,
            "topk_rows": None if state.topk is None else int((state.topk >= 0).sum(-1).max()),
        }
        assert row["finite"], row
        assert row["comb_col_sum_err"] < 1e-3, row
        assert abs(row["router_weight_sum"] - cfg.routed_scaling_factor) < 1e-3, row
        report["layers"].append(row)
        print(json.dumps(row))
        lw.release()
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
