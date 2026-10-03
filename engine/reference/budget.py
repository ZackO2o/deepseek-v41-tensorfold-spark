"""Bytes of the V4.1-Flash EXL3 pack per rank under the TP=2 plan of docs/ARCHITECTURE.md, from the safetensors
headers alone (``.cache/ckpt/<pack>/headers.json``, written by any remote reference run).

    python -m engine.reference.budget [--headers PATH] [--rows 1,4,6,8] [--context 65536]

Split rules (fraction of a tensor a rank holds and reads):

- replicated (1): wq_a, wkv (vLLM's fused ``disable_tp`` projection), the compressor and the indexer (their
  outputs feed the single-head KV / index caches every rank holds), the router gate, hc_*, norms, sinks, Engram
  q/k, DSpark ``main_proj`` and Markov / confidence heads;
- split in two (1/2): wq_b (heads 32 | 32), wo_a (groups 4 | 4), wo_b (K 4,096 | 4,096), routed and shared experts
  (intermediate 1,152 | 1,152), Engram wkv (N 12,800 | 12,800), head and embedding (vocab 64,640 | 64,640).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADERS = ROOT / ".cache" / "ckpt" / "dsv41-uncensored-2.9bpw" / "headers.json"

SPLIT = [
    (r"\.attn\.(wq_a|wkv)\.", 1.0, "attn q_a / kv_a (replicated)"),
    (r"\.attn\.wq_b\.", 0.5, "attn wq_b (heads split)"),
    (r"\.attn\.wo_a\.", 0.5, "attn wo_a (groups split)"),
    (r"\.attn\.wo_b\.", 0.5, "attn wo_b (K split)"),
    (r"\.attn\.compressor\.", 1.0, "compressor (replicated)"),
    (r"\.attn\.indexer\.", 1.0, "indexer (replicated)"),
    (r"\.attn\.(attn_sink|q_norm|kv_norm)", 1.0, "attn norms / sinks"),
    (r"\.ffn\.experts\.", 0.5, "routed experts (inter split)"),
    (r"\.ffn\.shared_experts\.", 0.5, "shared expert (inter split)"),
    (r"\.ffn\.gate\.", 1.0, "router gate + bias"),
    (r"\.hc_", 1.0, "hc fn / base / scale (fp32)"),
    (r"_norm\.weight$", 1.0, "block norms"),
    (r"\.engram\.wkv\.", 0.5, "Engram wkv (N split)"),
    (r"\.engram\.", 1.0, "Engram q / k"),
    (r"^mtp\.\d+\.main_proj\.", 1.0, "DSpark main_proj"),
    (r"^mtp\.\d+\.(markov_head|confidence_head|norm|main_norm)", 1.0, "DSpark heads"),
    (r"^head\.", 0.5, "lm head (vocab split)"),
    (r"^embed\.", 0.5, "embedding (vocab split)"),
    (r"^norm\.", 1.0, "final norm"),
    (r"^(vision|aligner|image_)", 1.0, "vision (rank 0 only in practice)"),
]


def load(path: Path = HEADERS) -> dict[str, int]:
    m = json.loads(Path(path).read_text())
    out = {}
    for v in m["headers"].values():
        for n, e in v["header"].items():
            if n != "__metadata__":
                out[n] = e["data_offsets"][1] - e["data_offsets"][0]
    return out


def classify(name: str) -> tuple[str, float]:
    for pat, frac, label in SPLIT:
        if re.search(pat, name):
            return label, frac
    return "other", 1.0


def per_rank(sizes: dict[str, int]) -> dict[str, dict]:
    """Resident bytes a rank per class, split by backbone / DSpark."""

    out: dict[str, dict] = defaultdict(lambda: {"backbone": 0.0, "dspark": 0.0, "top": 0.0})
    for n, b in sizes.items():
        label, frac = classify(n)
        where = "dspark" if n.startswith("mtp.") else "backbone" if n.startswith("layers.") else "top"
        out[label][where] += b * frac
    return out


def decode_bytes(sizes: dict[str, int], rows: int, distinct: dict[int, float], context: int,
                 n_layers: int = 40) -> dict[str, float]:
    """Bytes a rank reads for one forward over ``rows`` verify rows (no drafting), at ``context`` tokens."""

    expert = defaultdict(float)
    nonexp = 0.0
    for n, b in sizes.items():
        if n.startswith(("mtp.", "vision", "aligner", "image_")) or n.startswith("embed."):
            continue
        label, frac = classify(n)
        m = re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.", n)
        if m:
            expert[int(m.group(1))] += b * frac / 384.0           # mean bytes of one expert (half) in the layer
            continue
        nonexp += b * frac
    u = distinct[rows]
    exp_bytes = sum(v * u for v in expert.values())
    # KV reads of R rows of one sequence: the SWA window (shared, + R - 1 newer rows) on 40 layers, each row's own
    # top-512 compressed rows on the 36 compressed layers, and the indexer keys, scanned once for all rows: the
    # three ratio-2 sources (context / 2 keys), layer 20 (context keys) and the four Reindex layers (their
    # 16,384-key candidate pool)
    row = 584                                                     # fp8_ds_mla row: 448 + 128 + 8
    ikey = 132                                                    # FP8 index key: 128 + 4
    kv = n_layers * (128 + rows - 1) * row + rows * 36 * min(512, context) * row
    idx = (3 * context // 2 + context) * ikey + 4 * min(16384, context) * ikey
    return {"non_expert": nonexp, "experts": exp_bytes, "kv": kv, "indexer": idx,
            "total": nonexp + exp_bytes + kv + idx, "expert_layer_mean_half": sum(expert.values()) / n_layers}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headers", default=str(HEADERS))
    ap.add_argument("--rows", default="1,4,6,8")
    ap.add_argument("--context", type=int, default=65536)
    args = ap.parse_args()
    sizes = load(Path(args.headers))
    res = per_rank(sizes)
    tot = defaultdict(float)
    print(f"{'class':38s} {'backbone':>10s} {'dspark':>10s} {'top':>10s}  (GiB a rank)")
    for label, v in sorted(res.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{label:38s} {v['backbone'] / 2**30:10.3f} {v['dspark'] / 2**30:10.3f} {v['top'] / 2**30:10.3f}")
        for k in v:
            tot[k] += v[k]
    print(f"{'total':38s} {tot['backbone'] / 2**30:10.3f} {tot['dspark'] / 2**30:10.3f} {tot['top'] / 2**30:10.3f}"
          f"  = {sum(tot.values()) / 2**30:.2f} GiB")
    distinct = {1: 6, 2: 11, 3: 15, 4: 18.5, 5: 22, 6: 25, 8: 30, 16: 48}     # DSV41-BASELINE.md assumption
    print(f"\nbytes a rank for one forward at context {args.context} (U(R) distinct experts a layer assumed):")
    for r in [int(x) for x in args.rows.split(",")]:
        d = decode_bytes(sizes, r, distinct, args.context)
        print(f"R={r}: non-expert {d['non_expert'] / 1e9:.3f} GB, experts {d['experts'] / 1e9:.3f} GB "
              f"(U={distinct[r]}, {d['expert_layer_mean_half'] / 1e6:.2f} MB an expert half), "
              f"KV {d['kv'] / 1e6:.1f} MB, indexer {d['indexer'] / 1e6:.1f} MB, total {d['total'] / 1e9:.3f} GB "
              f"= {d['total'] / 230e9 * 1e3:.1f} ms at 230 GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
