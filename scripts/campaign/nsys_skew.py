#!/usr/bin/env python3
"""Rank skew of a TP = 2 decode run: two nsys reports (rank 0 and rank 1, each exported to SQLite) of the same 1-stream
graphs-on run, traced with ``--cuda-graph-trace=node`` (G7.sh skew). Answers "rank 0 waits ~2.3 ms a window in its
exchanges: which rank is late, where, and why".

Windows are graph replays (nsys_decode.py's rule: a maximal run of graph-node kernels). Inside a window, the exchanges
(RoCE ``gather_kernel`` / NCCL all-gathers) cut it into segments; segment i is the work a rank does between exchange
i - 1 and exchange i. The two ranks run the same windows and exchanges in the same order, so segment i of window w on
rank 0 and rank 1 is the same piece of the model. Clocks of the two nodes are not compared; only durations are.

Per segment index (averaged over the windows that have the modal exchange count):
- busy: the summed kernel time inside the segment on each rank (the GPU work, no idle);
- the exchange's duration on each rank (wait for the peer + transport): the later rank's exchange is short, the
  earlier rank's long;
and per kernel name, the time a window on each rank (the kernels that are slower on one rank, or only on one rank).

    nsys_skew.py R0.sqlite R1.sqlite [--json OUT.json] [--min-kernels 200] [--top 25]
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics

XCHG = ("tfroce::gather_kernel", "ncclDevKernel_AllGather", "ncclDevKernel_AllReduce")


def windows(path: str, min_kernels: int) -> list:
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = db.execute("SELECT start, end, demangledName, graphId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    out, cur = [], []
    for start, end, nid, gid in rows:
        if gid is None:
            if len(cur) >= min_kernels:
                out.append(cur)
            cur = []
            continue
        cur.append((start, end, names.get(nid, "?")))
    if len(cur) >= min_kernels:
        out.append(cur)
    return out


def short(name: str) -> str:
    n = name.split("(")[0]
    return n[-90:]


def segments(w: list) -> tuple:
    """-> (busy ns a segment, exchange ns a segment, wall ns a segment); the tail after the last exchange is a segment."""
    busy, xch, wall = [], [], []
    b, seg0 = 0, w[0][0]
    for start, end, name in w:
        if any(k in name for k in XCHG):
            busy.append(b)
            xch.append(end - start)
            wall.append(start - seg0)
            b, seg0 = 0, end
        else:
            b += end - start
    busy.append(b)
    xch.append(0)
    wall.append(w[-1][1] - seg0)
    return busy, xch, wall


def analyse(p0: str, p1: str, min_kernels: int = 200, top: int = 25) -> dict:
    ws = [windows(p0, min_kernels), windows(p1, min_kernels)]
    segs = [[segments(w) for w in r] for r in ws]
    nx = [collections.Counter(len(s[0]) for s in r) for r in segs]
    mode = (nx[0] + nx[1]).most_common(1)[0][0] if nx[0] and nx[1] else 0
    keep = [[s for s in r if len(s[0]) == mode] for r in segs]
    n = min(len(keep[0]), len(keep[1]))
    keep = [k[len(k) - n:] for k in keep]           # the tail: rank 1 is traced from boot (its follower loop never
    ws = [w[len(w) - len(ws[0]):] if i else w for i, w in enumerate(ws)]   # calls cudaProfilerStart): its last
    # windows are rank 0's captured ones
    out: dict = {"windows": [len(ws[0]), len(ws[1])], "segments_a_window": mode, "aligned_windows": n}
    if not n:
        return out
    per = []
    for i in range(mode):
        b0 = statistics.mean(keep[0][w][0][i] for w in range(n)) / 1e3
        b1 = statistics.mean(keep[1][w][0][i] for w in range(n)) / 1e3
        x0 = statistics.mean(keep[0][w][1][i] for w in range(n)) / 1e3
        x1 = statistics.mean(keep[1][w][1][i] for w in range(n)) / 1e3
        per.append({"i": i, "busy_r0_us": round(b0, 2), "busy_r1_us": round(b1, 2), "d_busy_us": round(b1 - b0, 2),
                    "xchg_r0_us": round(x0, 2), "xchg_r1_us": round(x1, 2)})
    out["segments"] = per
    tot = lambda k: round(sum(s[k] for s in per) / 1e3, 3)
    out["window_ms"] = {"busy_r0": tot("busy_r0_us"), "busy_r1": tot("busy_r1_us"), "xchg_r0": tot("xchg_r0_us"),
                        "xchg_r1": tot("xchg_r1_us")}
    out["segments_r1_slower"] = sum(1 for s in per if s["d_busy_us"] > 0)
    # per kernel name: ms a window on each rank (all windows, not only aligned)
    kn = []
    for r in ws:
        c = collections.Counter()
        cnt = collections.Counter()
        for w in r:
            for start, end, name in w:
                c[short(name)] += (end - start) / 1e6
                cnt[short(name)] += 1
        kn.append(({k: v / max(len(r), 1) for k, v in c.items()}, {k: v / max(len(r), 1) for k, v in cnt.items()}))
    names = set(kn[0][0]) | set(kn[1][0])
    rows = []
    for k in names:
        a, b = kn[0][0].get(k, 0.0), kn[1][0].get(k, 0.0)
        rows.append({"kernel": k, "ms_r0": round(a, 4), "ms_r1": round(b, 4), "d_ms": round(b - a, 4),
                     "n_r0": round(kn[0][1].get(k, 0), 1), "n_r1": round(kn[1][1].get(k, 0), 1),
                     "ratio": round(b / a, 3) if a > 0 else None})
    rows.sort(key=lambda r: -abs(r["d_ms"]))
    out["kernels"] = rows[:top]
    # the uniform slow-down (clocks): median ratio of the kernels both ranks run alike (same launches a window)
    same = [r["ratio"] for r in rows if r["ratio"] and r["n_r0"] == r["n_r1"] and r["ms_r0"] > 0.05]
    out["median_ratio_same_kernels"] = round(statistics.median(same), 4) if same else None
    return out


def report(r: dict) -> str:
    if not r.get("aligned_windows"):
        return f"no aligned windows: {r}"
    w = r["window_ms"]
    lines = [f"windows r0 / r1 {r['windows']}, {r['segments_a_window']} segments a window, {r['aligned_windows']} aligned",
             f"a window: busy r0 {w['busy_r0']} ms, r1 {w['busy_r1']} ms (r1 - r0 {w['busy_r1'] - w['busy_r0']:+.3f}); "
             f"exchanges r0 {w['xchg_r0']} ms, r1 {w['xchg_r1']} ms",
             f"segments where r1 is busier: {r['segments_r1_slower']} of {r['segments_a_window']}; median r1 / r0 "
             f"ratio of the kernels both run alike: {r['median_ratio_same_kernels']}",
             "largest busy differences (segment: r0 / r1 us, exchange r0 / r1 us):"]
    for s in sorted(r["segments"], key=lambda s: -abs(s["d_busy_us"]))[:12]:
        lines.append(f"  seg {s['i']:3d}: {s['busy_r0_us']:8.1f} / {s['busy_r1_us']:8.1f} ({s['d_busy_us']:+7.1f}); "
                     f"xchg {s['xchg_r0_us']:6.1f} / {s['xchg_r1_us']:6.1f}")
    lines.append("kernels by |r1 - r0| ms a window (ms r0 / r1, launches r0 / r1, ratio):")
    for k in r["kernels"]:
        lines.append(f"  {k['d_ms']:+8.4f}  {k['ms_r0']:8.4f} / {k['ms_r1']:8.4f}  {k['n_r0']:6.1f} / {k['n_r1']:6.1f}  "
                     f"{k['ratio']}  {k['kernel']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("r0")
    ap.add_argument("r1")
    ap.add_argument("--json")
    ap.add_argument("--min-kernels", type=int, default=200)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    r = analyse(a.r0, a.r1, a.min_kernels, a.top)
    print(report(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
