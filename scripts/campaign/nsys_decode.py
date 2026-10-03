#!/usr/bin/env python3
"""Where a decode verify window's exchange and glue time goes, from an nsys report exported to SQLite (``nsys export
--type sqlite``) of a graphs-on decode run traced with ``--cuda-graph-trace=node``. G7-decode.sh's ``xchg`` section
(levers 3 and 4 of docs/DECODE-ROOFLINE.md).

A window is one graph replay: a maximal run of graph-node kernels (``graphId`` set) of at least ``--min-kernels``
kernels; the eager kernels between replays (the DSpark pass, samplers) are not counted. Per window, averaged:

- exchanges: the RoCE gathers (``tfroce::gather_kernel``) and NCCL all-gathers: count, ms, p10 / p50 / p90 us, and
  the part above the p10 floor (``wait``: mostly the peer finishing later, i.e. rank skew, which no transport change
  removes; GLM's PREFETCH-COMM.md section 3.2);
- the partial glue: the fp32 -> bf16 casts and bf16 -> fp32 copies around the exchanges;
- mHC: ``_site`` and ``_finish_k`` (count, ms, us each);
- RMSNorm: the fused ``_rms`` kernel, or torch's float64 set (``MeanOps<double>``, pow, rsqrt, double muls / adds);
- other elementwise / copy / cat kernels, and every kernel's total.

G4's nsys1 (rank 0, 137 windows, before G7): gathers 81 a window, 2.96 ms (p10 8.4, p50 15.1, p90 75 us; capped at
15 us they would be 1.0 ms), _site 2.34 ms, _finish_k 0.84 ms, the float64 RMSNorm sets ~1.3 ms.

    nsys_decode.py REPORT.sqlite [--json OUT.json] [--min-kernels 200]
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3

GROUPS = (                                       # (group, substrings of the demangled name), first match wins
    ("gather", ("tfroce::gather_kernel", "ncclDevKernel_AllGather")),
    ("mhc site", ("_site",)),
    ("mhc finish", ("_finish_k",)),
    ("rms fused", ("_rms",)),
    ("rms float64", ("MeanOps<double", "pow_tensor_scalar_kernel_impl<double", "rsqrt_kernel_cuda",
                     "MulFunctor<double", "CUDAFunctorOnSelf_add<double")),
    ("cast to bf16", ("bfloat16_copy_kernel_cuda",)),
    ("copies", ("direct_copy_kernel_cuda", "CatArrayBatchedCopy")),
    ("other elementwise", ("elementwise_kernel", "reduce_kernel", "FillFunctor", "where_kernel")),
)


def group(name: str) -> str:
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


def _pct(v: list, q: float) -> float:
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0.0


def analyse(path: str, min_kernels: int = 200) -> dict:
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = db.execute("SELECT start, end, demangledName, graphId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    windows, cur = [], []
    for start, end, nid, gid in rows:
        if gid is None:
            if len(cur) >= min_kernels:
                windows.append(cur)
            cur = []
            continue
        cur.append((start, end, names.get(nid, "?")))
    if len(cur) >= min_kernels:
        windows.append(cur)
    if not windows:
        return {"windows": 0}
    ms = collections.Counter()
    count = collections.Counter()
    gathers = []
    span = 0.0
    for w in windows:
        span += (w[-1][1] - w[0][0]) / 1e6
        for start, end, name in w:
            g = group(name)
            ms[g] += (end - start) / 1e6
            ms["all kernels"] += (end - start) / 1e6
            count[g] += 1
            if g == "gather":
                gathers.append((end - start) / 1e3)
    n = len(windows)
    gathers.sort()
    floor = _pct(gathers, 0.10)
    out = {
        "windows": n,
        "span_ms": round(span / n, 3),
        "ms": {g: round(v / n, 3) for g, v in ms.most_common()},
        "launches": {g: round(v / n, 1) for g, v in count.most_common()},
        "gather_us": {"p10": round(floor, 1), "p50": round(_pct(gathers, 0.5), 1), "p90": round(_pct(gathers, 0.9), 1),
                      "p99": round(_pct(gathers, 0.99), 1)},
        "gather_wait_ms": round(sum(max(0.0, d - floor) for d in gathers) / 1e3 / n, 3),
    }
    return out


def report(r: dict) -> str:
    if not r.get("windows"):
        return "no graph windows found (decode with graphs on, traced with --cuda-graph-trace=node?)"
    g = r["gather_us"]
    lines = [f"{r['windows']} windows, kernels {r['ms'].get('all kernels', 0):.2f} ms a window (span {r['span_ms']:.2f} ms)",
             f"  exchanges: {r['launches'].get('gather', 0):.0f} a window, {r['ms'].get('gather', 0):.3f} ms; us p10 "
             f"{g['p10']} p50 {g['p50']} p90 {g['p90']} p99 {g['p99']}; above the p10 floor (wait / skew) "
             f"{r['gather_wait_ms']:.3f} ms"]
    for k in ("mhc site", "mhc finish", "rms fused", "rms float64", "cast to bf16", "copies", "other elementwise"):
        if k in r["ms"]:
            n = r["launches"][k]
            lines.append(f"  {k:18s} {r['ms'][k]:.3f} ms, {n:.0f} launches ({1000 * r['ms'][k] / max(n, 1):.1f} us each)")
    glue = sum(r["ms"].get(k, 0) for k in ("mhc site", "mhc finish", "rms fused", "rms float64", "cast to bf16",
                                           "copies", "other elementwise"))
    lines.append(f"  glue (mHC + norms + casts + copies + elementwise) {glue:.3f} ms a window")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--json")
    ap.add_argument("--min-kernels", type=int, default=200)
    a = ap.parse_args()
    r = analyse(a.sqlite, a.min_kernels)
    print(report(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
