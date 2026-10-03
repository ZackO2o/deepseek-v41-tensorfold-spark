#!/usr/bin/env python3
"""Per-window kernel classes of a graphs-on decode trace (nsys export --type sqlite, --cuda-graph-trace=node), split
by the window's row count (G9). A window is one graph replay: a maximal run of kernels with the same graphId. Rows =
``_finish_k``'s gridX (one program a row). A graph without the indexer's kernels (``_scores`` / ``_keys``) and with
fewer than 20 ``_finish_k`` launches is a DSpark pass (3 drafter blocks).

Per row count: windows, span ms (first kernel start -> last kernel end), kernel ms by class, GPU idle inside the
window (span - the union of kernel intervals), exchanges split into transfer (the per-row-count p10 of a gather) and
wait (the rest). Between windows: idle from one window's end to the next one's start, and the eager kernels there.
Dense EXL3 shapes are listed by (kernel, grid) with us a window.

    nsys_window.py REPORT.sqlite [--json OUT.json] [--min-kernels 50]
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics

CLASSES = (                                     # (class, name substrings): first match wins
    ("exchange", ("tfroce::gather_kernel", "ncclDevKernel")),
    ("router", ("dsv41_rg::", "_logits", "_select", "_fused_router")),
    ("experts", ("x3ld::", "gateup_epilogue", "down_combine", "rot_in_kernel<__nv_bfloat16>", "expert")),
    ("dense x3dn", ("x3dn", "dsv41_dn::")),
    ("dense seg (x / q / o groups)", ("seg_linear_kernel", "seg_rot_in")),
    ("dense linear (wo_b, head, wk, drafter)", ("linear_kernel", "rot_in_kernel")),
    ("mhc site", ("_site",)),
    ("mhc finish", ("_finish_k",)),
    ("norms", ("_rms", "_pool_norm", "MeanOps<double")),
    ("attention", ("_chunks", "_merge", "_kv_store", "_rope", "_fuse", "attn")),
    ("indexer", ("_scores", "_keys", "_index_k", "_plain", "_block_keys", "_cand")),
    ("torch topk / sort", ("gatherTopK", "radixSort", "sbtopk", "bitonic", "_dtopk")),     # G9 glue: dtopk here too
    ("engram", ("engram", "_gate_wait", "gate_kernel")),
    ("copies / elementwise", ("copy_kernel", "elementwise_kernel", "reduce_kernel", "OpaqueType", "CatArray",
                              "FillFunctor", "index", "where")),
)


def cls(name: str) -> str:
    for c, keys in CLASSES:
        if any(k in name for k in keys):
            return c
    return "other"


def union(iv: list) -> float:
    tot, end = 0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            tot += e - s
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def analyse(path: str, min_kernels: int = 50) -> dict:
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = list(db.execute("SELECT start, end, demangledName, graphId, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL "
                           "ORDER BY start"))
    wins, cur, curg, eager = [], [], None, []
    for st, en, nid, gid, gx, gy, gz in rows:
        name = names.get(nid, "?")
        if gid is None:
            if cur:
                wins.append(cur)
                cur, curg = [], None
            eager.append((st, en, name))
            continue
        if gid != curg and cur:
            wins.append(cur)
            cur = []
        curg = gid
        cur.append((st, en, name, (gx, gy, gz)))
    if cur:
        wins.append(cur)
    wins = [w for w in wins if len(w) >= min_kernels]
    groups = collections.defaultdict(list)
    for w in wins:
        fin = [k for k in w if "_finish_k" in k[2]]
        idx = any("_scores" in k[2] or "_keys" in k[2] for k in w)
        if len(fin) < 20:                       # the DSpark pass: 3 drafter blocks (target windows: ~83 finishes)
            key = f"draft pass ({fin[0][3][0] if fin else '?'} rows)"
        elif fin and idx:
            key = f"target {fin[0][3][0]} row(s)"
        else:
            key = "target ? rows"
        groups[key].append(w)
    out: dict = {"windows": len(wins), "groups": {}}
    for key, ws in sorted(groups.items()):
        n = len(ws)
        ms = collections.Counter()
        shapes = collections.Counter()
        spans, idles, gathers = [], [], []
        for w in ws:
            spans.append((w[-1][1] - w[0][0]) / 1e6)
            idles.append((w[-1][1] - w[0][0] - union([(k[0], k[1]) for k in w])) / 1e6)
            for st, en, name, grid in w:
                c = cls(name)
                ms[c] += (en - st) / 1e6
                if c.startswith("dense"):
                    shapes[(name.split("(")[0].split("<")[0][-34:] + "<" + name.split("<")[1][:14] if "<" in name
                            else name.split("(")[0][-40:], grid)] += (en - st) / 1e3
                if c == "exchange":
                    gathers.append((en - st) / 1e3)
        gathers.sort()
        floor = gathers[len(gathers) // 10] if gathers else 0.0
        wait = sum(max(0.0, g - floor) for g in gathers) / 1e3 / n
        g = {"windows": n, "span_ms": round(statistics.mean(spans), 3), "idle_in_window_ms": round(statistics.mean(idles), 3),
             "kernel_ms": round(sum(ms.values()) / n, 3),
             "classes_ms": {c: round(v / n, 3) for c, v in ms.most_common()},
             "exchange": {"count": round(len(gathers) / n, 1), "floor_us": round(floor, 1),
                          "transfer_ms": round(len(gathers) * floor / 1e3 / n, 3), "wait_ms": round(wait, 3)},
             "dense_shapes_us": {f"{k[0]} grid {k[1]}": round(v / n, 1) for k, v in shapes.most_common(14)}}
        out["groups"][key] = g
    gaps = [(wins[i + 1][0][0] - wins[i][-1][1]) / 1e6 for i in range(len(wins) - 1)]
    out["between_windows"] = {"gaps": len(gaps), "median_ms": round(statistics.median(gaps), 3) if gaps else None,
                              "mean_ms": round(statistics.mean(gaps), 3) if gaps else None}
    ek = collections.Counter()
    for st, en, name in eager:
        ek[name.split("(")[0][-50:]] += (en - st) / 1e6
    out["eager_kernels_ms_total"] = {k: round(v, 2) for k, v in ek.most_common(10)}
    return out


def report(r: dict) -> str:
    L = [f"{r['windows']} windows; between windows: median {r['between_windows']['median_ms']} ms, mean "
         f"{r['between_windows']['mean_ms']} ms over {r['between_windows']['gaps']} gaps"]
    for key, g in r["groups"].items():
        x = g["exchange"]
        L.append(f"== {key}: {g['windows']} windows, span {g['span_ms']} ms, kernels {g['kernel_ms']} ms, idle in window "
                 f"{g['idle_in_window_ms']} ms; exchanges {x['count']} a window: transfer {x['transfer_ms']} ms "
                 f"(floor {x['floor_us']} us) + wait {x['wait_ms']} ms")
        for c, v in g["classes_ms"].items():
            L.append(f"   {c:42s} {v:8.3f} ms")
        for s, v in g["dense_shapes_us"].items():
            L.append(f"     dense {s:60s} {v:9.1f} us")
    L.append("eager kernels (total ms): " + "; ".join(f"{k} {v}" for k, v in r["eager_kernels_ms_total"].items()))
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--json")
    ap.add_argument("--min-kernels", type=int, default=50)
    a = ap.parse_args()
    r = analyse(a.sqlite, a.min_kernels)
    print(report(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
