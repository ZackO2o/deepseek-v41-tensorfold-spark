#!/usr/bin/env python3
"""The per-kernel boundary budget of a decode window (G10, PDL / fusion): from a graphs-on nsys trace exported to SQLite
(``--cuda-graph-trace=node``), for the target windows of a given row count (nsys_window.py's split: one graph replay,
rows = ``_finish_k``'s grid):

- launches a window, and the sum of the gaps between consecutive kernels (start of k+1 - end of k, > 0) and of the
  overlaps (< 0: an early PDL start);
- the **floor**: the p10 duration of a 1-CTA trivial kernel (torch Fill / copy). Every launch pays at least this
  (dispatch, the one CTA's ramp, drain and the end-of-grid flush), whatever its work; ``launches x (floor + gap)`` is
  the fixed cost of the window's kernel count, the part fusion removes and PDL hides in part;
- **SM coverage**: per kernel, CTAs a SM by registers / threads / shared memory (the trace's GB10 limits), the SMs the
  grid covers (min(48, CTAs)) and its waves; ms below full coverage = sum of duration x (1 - SMs covered / 48), and
  the ms of multi-wave kernels' last partial wave;
- **ramp + tail** of the weight streamers: ``duration = a + b x work`` fitted per family over the window's instances
  where the work is known from the grid (x3ld: gridX = the experts of the launch; gate/up = 2 x down; dense shapes by
  the G9 MB a rank table), ``a`` = the fixed us a launch (prologue + ramp + tail), against the floor;
- per class: launches, ms, launches <= 3 us (trivial: fusion candidates), and the PDL-able boundary count (a
  launch whose predecessor is one of our kernels: x3ld / x3seg / exl3 linear / Triton / tfroce);
- **the G10 levers, estimated on this trace** (``--pdl-stream``, ``--pdl-glue``, ``--hot``, ``--l2pf-mb``):
  - PDL: every launch TF_DSV41_PDL=1 makes a programmatic dependent (x3seg / upstream EXL3 linears via ``single`` /
    x3ld: a weight streamer, its first tiles before the wait; the Triton glue and the rot_in launches) saves
    ``pdl-stream`` / ``pdl-glue`` us (low, high), half that after a torch kernel (which never triggers early);
  - L2PF: a dense launch of known bytes (DENSE_MB) whose DRAM-idle window before it (since the last weight streamer
    ended) is W us gets min(MB, l2pf-mb, 0.8 x W x 0.23 MB/us) from L2: saving that x (1 / cold - 1 / hot) GB/s;
  - fusion: the trivial (<= 3 us) torch launches, each its duration + the median gap (an upper bound).

    nsys_ramp.py REPORT.sqlite [--rows 1] [--json OUT.json] [--top 25]
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nsys_window import cls  # noqa: E402  (the G9 classes)

OURS = ("x3ld::", "dsv41_x3seg::", "linear_kernel", "rot_in_kernel", "dsv41_rg::", "dsv41_dn::", "gateup_epilogue",
        "down_combine", "group_kernel", "tfroce::")
TRIVIAL_US = 3.0
STREAMERS = ("x3ld::ld_kernel", "seg_linear_kernel", "linear_kernel<", "dsv41_rg::gemv_kernel", "dsv41_dn::",
             "x3dn")
PDL_STREAM = ("x3ld::ld_kernel", "seg_linear_kernel", "linear_kernel<")       # what TF_DSV41_PDL=1 covers
PDL_GLUE_PREFIX = ("dsv41_x3seg::seg_rot_in_kernel", "<unnamed>::rot_in_kernel(const void")
PDL_TRITON = {"_rms", "_rms2", "_rope", "_chunks", "_merge", "_pool_norm", "_kv_store", "_index_k", "_fuse", "_plain"}
DRAM_MB_US = 0.23                       # ~230 GB/s a long stream on GB10
# G9 section 2: dense shapes (kernel template, grid) -> MB a rank (13.1 / 10.5 / 13.1 / 5.7 / 16.4)
DENSE_MB = {("linear_kernel<(int)10, (int)2, (int)4>", (40, 8, 1)): 13.1,
            ("seg_linear_kernel<(int)10, (int)2, (int)4, (int)3>", (256, 1, 1)): 10.5,
            ("linear_kernel<(int)10, (int)2, (int)8>", (128, 1, 1)): 13.1,
            ("seg_linear_kernel<(int)10, (int)2, (int)4, (int)3>", (112, 1, 1)): 5.7,
            ("seg_linear_kernel<(int)12, (int)2, (int)4, (int)3>", (112, 1, 1)): 5.7,
            ("seg_linear_kernel<(int)10, (int)2, (int)8, (int)1>", (192, 1, 1)): 16.4}


def short(name: str) -> str:
    base = name.split("(")[0] if not name.startswith("void ") else name[5:].split("(")[0]
    return base.split("::")[-1][:60] if "<" not in base else base[:70]


def gpu(db) -> dict:
    try:
        r = db.execute("SELECT smCount, maxRegistersPerSm, maxShmemPerSm, maxWarpsPerSm, maxBlocksPerSm "
                       "FROM TARGET_INFO_GPU").fetchone()
        return dict(zip(("sms", "regs", "smem", "warps", "blocks"), r))
    except sqlite3.Error:
        return {"sms": 48, "regs": 65536, "smem": 102400, "warps": 48, "blocks": 24}


def ctas_per_sm(g: dict, threads: int, regs: int, smem: int) -> int:
    warps = max(1, (threads + 31) // 32)
    by_w = g["warps"] // warps
    by_r = g["regs"] // max(1, warps * ((max(regs, 1) * 32 + 255) // 256 * 256)) if regs else g["blocks"]
    by_s = g["smem"] // (smem + 1024) if smem else g["blocks"]    # + the 1 KB a CTA the driver reserves
    return max(1, min(g["blocks"], by_w, by_r, by_s))


def windows(db, names: dict, rows: int, min_kernels: int = 50) -> list:
    q = ("SELECT start, end, demangledName, graphId, gridX, gridY, gridZ, blockX * blockY * blockZ, registersPerThread,"
         " staticSharedMemory + dynamicSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    wins, cur, curg = [], [], None
    for r in db.execute(q):
        if r[3] is None:
            if cur:
                wins.append(cur)
            cur, curg = [], None
            continue
        if r[3] != curg and cur:
            wins.append(cur)
            cur = []
        curg = r[3]
        cur.append((r[0], r[1], names.get(r[2], "?"), (r[4], r[5], r[6]), r[7], r[8], r[9]))
    if cur:
        wins.append(cur)
    out = []
    for w in wins:
        if len(w) < min_kernels:
            continue
        fin = [k for k in w if "_finish_k" in k[2]]
        if len(fin) >= 20 and any("_scores" in k[2] for k in w) and fin[0][3][0] == rows:
            out.append(w)
    return out


def fit(xs: list, ys: list) -> tuple[float, float] | None:
    if len(set(xs)) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def pdl_kind(name: str) -> str | None:
    if any(k in name for k in PDL_STREAM):
        return "stream"
    if name in PDL_TRITON or any(name.startswith(k) or k in name for k in PDL_GLUE_PREFIX):
        return "glue"
    return None


def analyse(path: str, rows: int = 1, top: int = 25, pdl_stream=(1.0, 2.0), pdl_glue=(0.25, 0.45), hot=(300.0, 400.0),
            l2pf_mb: float = 8.0) -> dict:
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    g = gpu(db)
    ws = windows(db, names, rows)
    if not ws:
        return {"rows": rows, "windows": 0}
    n = len(ws)
    trivial = sorted(k[1] - k[0] for w in ws for k in w
                     if k[3] == (1, 1, 1) and ("FillFunctor" in k[2] or "copy_kernel" in k[2]))
    floor = trivial[len(trivial) // 10] / 1e3 if trivial else 1.0
    agg = collections.defaultdict(lambda: collections.Counter())
    per_kernel = collections.defaultdict(lambda: [0, 0.0])
    tot = collections.Counter()
    xld, dense = [], collections.defaultdict(list)
    lev = collections.Counter()
    for w in ws:
        last_stream_end = w[0][0]
        for i, (st, en, name, grid, *_r) in enumerate(w):
            d = (en - st) / 1e3
            kind = pdl_kind(name)
            if kind and i:
                f = 0.5 if "at::native" in w[i - 1][2] or "cub" in w[i - 1][2] else 1.0
                lo, hi = pdl_stream if kind == "stream" else pdl_glue
                lev["pdl_lo_us"] += f * lo
                lev["pdl_hi_us"] += f * hi
                lev["pdl_" + kind] += 1
            if "at::native" in name and d <= TRIVIAL_US:
                lev["fuse_n"] += 1
                lev["fuse_us"] += d
            for (tmpl, gr), mb in DENSE_MB.items():
                if tmpl in name and gr == grid:
                    win = max(0.0, (st - last_stream_end) / 1e3)
                    pf = min(mb, l2pf_mb, 0.8 * win * DRAM_MB_US)
                    cold = mb / d * 1e3                                 # GB/s in the trace
                    for tag, h in zip(("lo", "hi"), hot):
                        if h > cold:
                            lev["l2pf_" + tag + "_us"] += pf * (1 / cold - 1 / h) * 1e3
                    lev["l2pf_mb"] += pf
                    lev["l2pf_win_us"] += win
                    lev["l2pf_n"] += 1
                    break
            if any(k in name for k in STREAMERS):
                last_stream_end = en
        tot["span_us"] += (w[-1][1] - w[0][0]) / 1e3
        tot["launches"] += len(w)
        prev = None
        for i, (st, en, name, grid, thr, regs, smem) in enumerate(w):
            d = (en - st) / 1e3
            c = cls(name)
            a = agg[c]
            a["n"] += 1
            a["all_us"] += d
            if i:
                gap = (st - w[i - 1][1]) / 1e3
                tot["gap_us"] += max(gap, 0.0)
                tot["overlap_us"] += min(gap, 0.0)
                tot["overlaps"] += gap < 0
            if d <= TRIVIAL_US:
                a["trivial_n"] += 1
                a["trivial_us"] += d
            ctas = grid[0] * grid[1] * grid[2]
            cps = ctas_per_sm(g, thr, regs, smem)
            covered = min(g["sms"], ctas)
            a["below_cover_us"] += d * (1 - covered / g["sms"])
            slots = g["sms"] * cps
            waves = -(-ctas // slots)
            if waves > 1:                    # the last wave's share of the time, at its fill
                fill = (ctas - (waves - 1) * slots) / slots
                a["tail_wave_us"] += d / waves * (1 - fill)
            ours = any(o in name for o in OURS) or not any(t in name for t in ("at::native", "at_cuda", "cub", "nccl"))
            if prev is not None and ours:
                a["pdl_edges"] += 1          # a boundary where this launch could be a programmatic dependent
            prev = name
            key = (short(name), grid)
            per_kernel[key][0] += 1
            per_kernel[key][1] += d
            if "x3ld::ld_kernel" in name:
                gate_up = grid[2] > 1 or grid[1] < 40     # gate/up: (experts, 9, 8); down: (experts, 40, 1)
                xld.append((grid[0] * (2 if gate_up else 1), d))
            for (tmpl, gr), mb in DENSE_MB.items():
                if tmpl in name and gr == grid:
                    dense[(tmpl.split("<")[0], gr)].append((mb, d))
    r = {"rows": rows, "windows": n, "gpu": g, "floor_us": round(floor, 2),
         "launches": round(tot["launches"] / n, 1), "span_ms": round(tot["span_us"] / n / 1e3, 3),
         "gap_ms": round(tot["gap_us"] / n / 1e3, 3), "overlap_ms": round(tot["overlap_us"] / n / 1e3, 3),
         "overlaps": round(tot["overlaps"] / n, 1)}
    r["fixed_ms"] = round(r["launches"] * floor / 1e3 + r["gap_ms"], 3)
    med_gap = r["gap_ms"] * 1e3 / max(1.0, r["launches"] - 1)
    r["levers"] = {"pdl_ms": [round(lev["pdl_lo_us"] / n / 1e3, 3), round(lev["pdl_hi_us"] / n / 1e3, 3)],
                   "pdl_launches": {"stream": round(lev["pdl_stream"] / n, 1), "glue": round(lev["pdl_glue"] / n, 1)},
                   "l2pf_ms": [round(lev["l2pf_lo_us"] / n / 1e3, 3), round(lev["l2pf_hi_us"] / n / 1e3, 3)],
                   "l2pf_launches": round(lev["l2pf_n"] / n, 1), "l2pf_mb": round(lev["l2pf_mb"] / n, 1),
                   "l2pf_window_us_mean": round(lev["l2pf_win_us"] / max(1, lev["l2pf_n"]), 1),
                   "fuse_trivial_torch": round(lev["fuse_n"] / n, 1),
                   "fuse_ms": round((lev["fuse_us"] + lev["fuse_n"] * med_gap) / n / 1e3, 3),
                   "params": {"pdl_stream_us": list(pdl_stream), "pdl_glue_us": list(pdl_glue), "hot_gbps": list(hot),
                              "l2pf_mb": l2pf_mb}}
    r["classes"] = {c: {k: round(v / n / (1e3 if k.endswith("_us") else 1), 3) for k, v in a.items()}
                    for c, a in sorted(agg.items(), key=lambda kv: -kv[1]["all_us"])}
    for v in r["classes"].values():
        for k in [k for k in v if k.endswith("_us")]:
            v[k[:-3] + "_ms"] = v.pop(k)
    f = fit([x for x, _ in xld], [y for _, y in xld]) if xld else None
    r["x3ld_fit"] = {"a_us": round(f[0], 2), "us_per_down_unit": round(f[1], 2), "launches": round(len(xld) / n, 1),
                     "fixed_ms": round(f[0] * len(xld) / n / 1e3, 3)} if f else None
    pts = [p for v in dense.values() for p in v]
    f = fit([p[0] for p in pts], [p[1] for p in pts]) if pts else None
    r["dense_fit"] = {"a_us": round(f[0], 2), "gbps": round(1e3 / f[1], 1) if f[1] > 0 else None,
                      "launches": round(len(pts) / n, 1), "fixed_ms": round(f[0] * len(pts) / n / 1e3, 3),
                      "shapes": {f"{k[0]} {k[1]}": round(statistics.mean(d for _, d in v), 1)
                                 for k, v in dense.items()}} if f else None
    r["top"] = [{"kernel": k[0], "grid": list(k[1]), "per_window": round(c / n, 1), "us_each": round(t / c, 2),
                 "ms_window": round(t / n / 1e3, 3)}
                for k, (c, t) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:top]]
    return r


def report(r: dict) -> str:
    if not r.get("windows"):
        return f"== {r['rows']}-row windows: none"
    L = [f"== {r['rows']}-row windows: {r['windows']}, span {r['span_ms']} ms, {r['launches']} launches a window",
         f"   gaps (start k+1 - end k > 0): {r['gap_ms']} ms; early starts {r['overlaps']} ({r['overlap_ms']} ms)",
         f"   floor (p10 of a 1-CTA Fill / copy): {r['floor_us']} us -> launches x floor + gaps = {r['fixed_ms']} ms"]
    if r.get("x3ld_fit"):
        x = r["x3ld_fit"]
        L.append(f"   x3ld ld_kernel fit: {x['a_us']} us a launch + {x['us_per_down_unit']} us a down-expert unit "
                 f"({x['launches']} launches: {x['fixed_ms']} ms fixed a window)")
    if r.get("dense_fit"):
        x = r["dense_fit"]
        L.append(f"   dense fit (G9 MB a rank): {x['a_us']} us a launch + bytes at {x['gbps']} GB/s "
                 f"({x['launches']} launches: {x['fixed_ms']} ms fixed a window)")
    v = r["levers"]
    L.append(f"   G10 levers (estimates, ms a window): PDL {v['pdl_ms'][0]}-{v['pdl_ms'][1]} "
             f"({v['pdl_launches']['stream']} streamers + {v['pdl_launches']['glue']} glue launches); L2PF "
             f"{v['l2pf_ms'][0]}-{v['l2pf_ms'][1]} ({v['l2pf_launches']} dense launches, {v['l2pf_mb']} MB from L2, "
             f"idle window mean {v['l2pf_window_us_mean']} us); fusing the {v['fuse_trivial_torch']} trivial torch "
             f"launches <= {v['fuse_ms']}")
    L.append(f"   {'class':40s} {'n':>6s} {'ms':>7s} {'<=3us n':>8s} {'ms':>6s} {'<48 SMs ms':>10s} {'tail wave':>9s} "
             f"{'PDL edges':>9s}")
    for c, v in r["classes"].items():
        L.append(f"   {c:40s} {v.get('n', 0):6.0f} {v.get('all_ms', 0):7.3f} {v.get('trivial_n', 0):8.0f} "
                 f"{v.get('trivial_ms', 0):6.3f} {v.get('below_cover_ms', 0):10.3f} {v.get('tail_wave_ms', 0):9.3f} "
                 f"{v.get('pdl_edges', 0):9.0f}")
    L.append("   most launched (per window, us each, ms a window):")
    for t in r["top"]:
        L.append(f"     {t['per_window']:6.1f} x {t['us_each']:8.2f} us = {t['ms_window']:6.3f} ms  {t['kernel']} "
                 f"{tuple(t['grid'])}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--rows", default="1,2", help="row counts, comma separated")
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--pdl-stream", default="1.0,2.0", help="us saved a PDL weight-streamer launch (low,high)")
    ap.add_argument("--pdl-glue", default="0.25,0.45", help="us saved a PDL glue launch (low,high; W11: 0.31)")
    ap.add_argument("--hot", default="300,400", help="L2-hot GB/s of a dense launch (low,high; G8: 288-443)")
    ap.add_argument("--l2pf-mb", type=float, default=8.0)
    a = ap.parse_args()
    pair = lambda t: tuple(float(x) for x in t.split(","))              # noqa: E731
    res = [analyse(a.sqlite, int(x), a.top, pair(a.pdl_stream), pair(a.pdl_glue), pair(a.hot), a.l2pf_mb)
           for x in a.rows.split(",")]
    print("\n".join(report(r) for r in res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
