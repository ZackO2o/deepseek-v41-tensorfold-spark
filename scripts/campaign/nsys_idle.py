#!/usr/bin/env python3
"""GPU idle per decode round from an nsys report exported to SQLite (the lead's G7 method,
docs/DECODE-ROOFLINE.md section 6): GPU busy = the union of CUPTI kernel, memcpy and memset intervals; a round = an
NVTX ``execute`` span (``TF_DSV41_PHASES=nvtx``); idle = the span minus busy. Each other NVTX phase gets its wall
and the idle inside it, per round (nested phases count in each ancestor: ``engram.wait`` inside ``graph.stage``).

Stall kernels (``--stall``, default the Engram gate's ``gate_copy_kernel``) are busy for CUPTI but may be the GPU
spinning on a flag: their time a round is reported apart, and ``idle+stall`` is the round's GPU time not spent on
model work. Trace with ``--trace=cuda,nvtx,osrt --cuda-graph-trace=node`` (graph kernels as nodes).

    nsys_idle.py REPORT.sqlite [--span execute] [--stall gate_copy_kernel] [--skip 3] [--json OUT.json]

``--between plan`` (G8 ``host``): a round is instead the time from one ``plan`` range's start to the next one's on the
same thread, so GPU work the host overlaps with sampling / planning / sharing (the speculative DSpark pass) and
idle outside ``execute`` both count.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import sqlite3


def _tables(db) -> set:
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def busy_intervals(db, stall: tuple) -> tuple[list, list]:
    """(merged busy intervals [(start, end)], stall-kernel intervals) in ns."""

    names = dict(db.execute("SELECT id, value FROM StringIds"))
    have = _tables(db)
    iv, st = [], []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in have:
        for s, e, nid in db.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            iv.append((s, e))
            if stall and any(k in names.get(nid, "") for k in stall):
                st.append((s, e))
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in have:
            iv += list(db.execute(f"SELECT start, end FROM {t}"))
    iv.sort()
    out: list = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(a, b) for a, b in out], sorted(st)


def covered(merged: list, starts: list, a: int, b: int) -> int:
    """ns of [a, b) covered by the merged intervals (``starts`` their start times)."""

    if b <= a:
        return 0
    i = max(0, bisect.bisect_right(starts, a) - 1)
    tot = 0
    while i < len(merged) and merged[i][0] < b:
        s, e = merged[i]
        tot += max(0, min(e, b) - max(s, a))
        i += 1
    return tot


def nvtx_ranges(db) -> list:
    """[(name, start, end, thread)] of the push / pop and start / end ranges."""

    names = dict(db.execute("SELECT id, value FROM StringIds"))
    cols = {r[1] for r in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    tid = "textId" if "textId" in cols else "NULL"
    rows = db.execute(f"SELECT text, {tid}, start, end, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL")
    return [(t if t is not None else names.get(i, "?"), s, e, g) for t, i, s, e, g in rows]


def between(rng, name: str) -> list:
    """[(start, end, thread)]: from each ``name`` range's start to the next one's on the same thread."""

    by = collections.defaultdict(list)
    for n, s, _, g in rng:
        if n == name:
            by[g].append(s)
    out = []
    for g, v in by.items():
        v.sort()
        out += [(a, b, g) for a, b in zip(v, v[1:])]
    return sorted(out)


def analyse(path: str, span: str = "execute", stall: tuple = ("gate_copy_kernel",), skip: int = 3,
            between_name: str = "") -> dict:
    db = sqlite3.connect(path)
    merged, stalls = busy_intervals(db, stall)
    starts = [s for s, _ in merged]
    sstarts = [s for s, _ in stalls]
    rng = nvtx_ranges(db)
    if between_name:
        span = f"{between_name}->{between_name}"
        rounds = between(rng, between_name)[skip:]
    else:
        rounds = sorted((s, e, g) for n, s, e, g in rng if n == span)[skip:]
    if not rounds:
        return {"rounds": 0}
    by_thread = collections.defaultdict(list)
    for n, s, e, g in rng:
        if n != span:
            by_thread[g].append((s, e, n))
    for v in by_thread.values():
        v.sort()
    first = {g: [s for s, _, _ in v] for g, v in by_thread.items()}
    wall = collections.Counter()
    idle = collections.Counter()
    count = collections.Counter()
    tot = {"span": 0, "idle": 0, "stall": 0}
    for a, b, g in rounds:
        tot["span"] += b - a
        tot["idle"] += (b - a) - covered(merged, starts, a, b)
        tot["stall"] += covered(stalls, sstarts, a, b)
        mine = by_thread.get(g, [])
        for s, e, n in mine[bisect.bisect_left(first.get(g, []), a):]:
            if s >= b:
                break
            if e > b:
                continue
            wall[n] += e - s
            idle[n] += (e - s) - covered(merged, starts, s, e)
            count[n] += 1
    k = len(rounds)
    ms = lambda v: round(v / 1e6 / k, 3)        # noqa: E731
    return {"rounds": k, "span": span, "round_ms": ms(tot["span"]), "idle_ms": ms(tot["idle"]),
            "stall_ms": ms(tot["stall"]), "idle_pct": round(100 * tot["idle"] / max(tot["span"], 1), 1),
            "phases": {n: {"wall_ms": ms(wall[n]), "idle_ms": ms(idle[n]), "per_round": round(count[n] / k, 2)}
                       for n, _ in wall.most_common()}}


def report(r: dict) -> str:
    if not r.get("rounds"):
        return "no rounds (NVTX spans) found: run with TF_DSV41_PHASES=nvtx and --trace including nvtx"
    lines = [f"{r['rounds']} rounds ({r['span']}): {r['round_ms']:.2f} ms a round, GPU idle {r['idle_ms']:.2f} ms "
             f"({r['idle_pct']}%), stall kernels {r['stall_ms']:.2f} ms, idle+stall "
             f"{r['idle_ms'] + r['stall_ms']:.2f} ms"]
    for n, v in r["phases"].items():
        lines.append(f"  {n:14s} wall {v['wall_ms']:7.3f}  idle {v['idle_ms']:7.3f} ms a round ({v['per_round']} a round)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--span", default="execute")
    ap.add_argument("--stall", default="gate_copy_kernel", help="comma-separated kernel name parts ('' = none)")
    ap.add_argument("--skip", type=int, default=3, help="first rounds dropped (warm-up)")
    ap.add_argument("--between", default="", help="rounds from one NVTX range's start to the next (e.g. plan)")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = analyse(a.sqlite, a.span, tuple(s for s in a.stall.split(",") if s), a.skip, a.between)
    print(report(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
