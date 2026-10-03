#!/usr/bin/env python3
"""Where a prefill's GPU time and GPU idle time go, from an nsys report exported to SQLite (``nsys export --type
sqlite``): the busy / idle split of the capture, the idle time inside each NVTX range name (``share``,
``engram.wait``, ``prefill`` ...), the largest gaps with the ranges around them, and the kernel time of the prefill
lever families (attention, mHC, exchange glue, RoPE, router, experts, GEMMs). G7-prefill2.sh's ``nsys`` step; the G5
analysis (2026-10-02) with it: share 8.5 s with the GPU 99.8% busy inside it; idle 2.1 s, of which 1.87 s is layer
1's Engram wait at each segment's start.

    nsys_gaps.py REPORT.sqlite [--json OUT.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3

FAMILIES = {                                    # kernel short name prefix -> lever family
    "attention": ("_chunks", "_merge", "_fused"),
    "indexer": ("_scores", "_keys", "_block_keys", "computeBlockDigitCounts", "gatherTopK", "radixSort",
                "computeBlockwise", "_stream", "_kth", "_select_k"),
    "mhc": ("_site", "_finish_k"),
    "rope": ("_rope",),
    "router": ("_logits", "_select", "_route"),
    "routed experts": ("ld_kernel", "group_kernel", "down_combine_kernel", "gateup_epilogue_kernel", "gm_", "x3gm",
                       "combine"),
    "prefill GEMMs": ("_gemm", "unpack_kernel", "rot_in_kernel"),
    "nccl": ("ncclDevKernel",),
    "elementwise / copies": ("vectorized_elementwise_kernel", "unrolled_elementwise_kernel", "elementwise_kernel",
                             "CatArrayBatchedCopy", "reduce_kernel"),
}


def family(name: str) -> str:
    for fam, prefixes in FAMILIES.items():
        if any(name.startswith(p) or p in name for p in prefixes):
            return fam
    return "other"


def merged(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a: int, b: int, ivs) -> int:
    return sum(max(0, min(b, e) - max(a, s)) for s, e in ivs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    iv = []
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in tables:
            iv += db.execute(f"select start, end from {t}").fetchall()
    if not iv:
        print("no GPU activity")
        return
    busy = merged(iv)
    t0, t1 = busy[0][0], busy[-1][1]
    gaps = [(busy[i][1], busy[i + 1][0]) for i in range(len(busy) - 1) if busy[i + 1][0] > busy[i][1]]
    nv = db.execute("select n.start, n.end, coalesce(n.text, s.value) from NVTX_EVENTS n left join StringIds s "
                    "on n.textId = s.id where n.end is not null").fetchall() if "NVTX_EVENTS" in tables else []
    span, on = (t1 - t0) / 1e9, sum(e - s for s, e in busy) / 1e9
    rep = {"span_s": round(span, 3), "busy_s": round(on, 3), "idle_s": round(span - on, 3), "idle_in": {},
           "families_s": {}, "gaps": []}
    print(f"GPU span {span:.2f} s, busy {on:.2f} s ({100 * on / span:.1f}%), idle {span - on:.2f} s")
    by_name = collections.defaultdict(list)
    for s, e, t in nv:
        by_name[t].append((s, e))
    print("NVTX range: total s, GPU idle inside s")
    for name, rs in sorted(by_name.items(), key=lambda kv: -sum(e - s for s, e in kv[1]))[:12]:
        tot = sum(e - s for s, e in rs) / 1e9
        idle = sum(overlap(s, e, gaps) for s, e in merged(rs)) / 1e9
        rep["idle_in"][name] = {"total_s": round(tot, 3), "idle_s": round(idle, 3), "n": len(rs)}
        print(f"  {name:32s} {tot:8.3f} {idle:8.3f}  (n {len(rs)})")
    print(f"largest gaps (ms at s, inside):")
    for s, e in sorted(gaps, key=lambda g: g[0] - g[1])[:a.top]:
        inside = sorted({t for rs, t in ((r, n) for n, r in by_name.items()) for x, y in rs if x <= s and y >= e})
        rep["gaps"].append({"ms": round((e - s) / 1e6, 2), "at_s": round((s - t0) / 1e9, 3), "in": inside})
        print(f"  {(e - s) / 1e6:7.1f} ms at {(s - t0) / 1e9:7.3f} s  {inside}")
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        fam = collections.Counter()
        for name, dur in db.execute("select s.value, k.end - k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s "
                                    "on k.shortName = s.id"):
            fam[family(name)] += dur
        tot = sum(fam.values())
        print("kernel time by family (s, % of kernel time)")
        for k, v in fam.most_common():
            rep["families_s"][k] = round(v / 1e9, 3)
            print(f"  {k:24s} {v / 1e9:8.3f} {100 * v / tot:5.1f}%")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
