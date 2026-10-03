#!/usr/bin/env python3
"""G4-perf.sh's summary: the speed table, C1 / C2 / C4, the exactness gates, the phase attribution, memory minima, from
$OUT/m2-<step>.json (m2bench), the samplers' mem-r0.log / mem-r1.log and the gate report. Prints plain text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TARGET = {"code": 52.0, "prose": 38.0}
KIT = {"code": 36.4, "prose": 25.1, "structured": 38.0, "C1": 32.2, "C2": 46.7, "C4": 37.6}


def load(out: Path, tag: str) -> dict | None:
    f = out / f"m2-{tag}.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def tok(r: dict | None, name: str, t: str = "t0"):
    try:
        return r["workloads"][name][t]["tok_s"]
    except (KeyError, TypeError):
        return None


def speed_table(runs: dict) -> list[str]:
    fast, eager, nccl, nuc0 = (runs.get(k) for k in ("fast", "eager", "nccl", "nucleus0"))
    names = list((fast or eager or {}).get("workloads", {}))
    lines = ["", "single stream (tok/s; T = 0 / T = 0.7)",
             "| workload | graphs on | graphs off | NCCL | nucleus off (T=0.7) | tok/round | target | kit |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        d = (fast or eager)["workloads"][n]["t0"].get("drafting", {})
        lines.append(f"| {n} | {tok(fast, n)} / {tok(fast, n, 't07')} | {tok(eager, n)} / {tok(eager, n, 't07')} | "
                     f"{tok(nccl, n)} / {tok(nccl, n, 't07')} | {tok(nuc0, n, 't07')} | "
                     f"{d.get('tokens_per_round')} | {TARGET.get(n, '')} | {KIT.get(n, '')} |")
    lines += ["", "concurrent (aggregate tok/s from the first submit; decode tok/s first to last token)",
              "| C | graphs on | graphs off | NCCL | kit |", "| --- | ---: | ---: | ---: | ---: |"]
    for c in ("C1", "C2", "C4"):
        cell = []
        for r in (fast, eager, nccl):
            v = (r or {}).get("concurrent", {}).get(c) or ((r or {}).get("c4") if c == "C4" else None)
            cell.append(f"{v['aggregate_tok_s']} ({v['decode_aggregate_tok_s']})" if v else "-")
        lines.append(f"| {c} | {' | '.join(cell)} | {KIT[c]} |")
    return lines


def gates(runs: dict) -> list[str]:
    lines = ["", "exactness"]
    for tag, r in runs.items():
        if r and "workloads" in r:
            lines.append(f"  {tag}: drafted == serial every workload (T = 0 / 0.7): {r.get('exact_all')}")
    fast, eager, nuc0 = runs.get("fast"), runs.get("eager"), runs.get("nucleus0")
    if fast and eager:
        same = all(fast["workloads"][n][t]["reply"] == eager["workloads"][n][t]["reply"]
                   for n in fast["workloads"] for t in ("t0", "t07") if n in eager["workloads"])
        lines.append(f"  graphs on == off replies: {same}")
    if fast and nuc0:
        same = all(fast["workloads"][n]["t07"]["reply"] == nuc0["workloads"][n]["t07"]["reply"]
                   for n in nuc0["workloads"] if n in fast["workloads"])
        lines.append(f"  nucleus candidates == full vocabulary (T = 0.7 replies): {same}")
    for tag in ("fast", "eager"):
        r = runs.get(tag)
        if r:
            lines.append(f"  {tag} state: {r.get('m2_after', '')[:400]}")
            lines.append(f"  {tag} comm: {r.get('comm')}")
    return lines


def phases(runs: dict) -> list[str]:
    lines = ["", "phases (ms a round; 'phases' step = TF_DSV41_PHASES=sync: GPU + host per phase)"]
    for tag in ("phases", "fast", "eager"):
        r = runs.get(tag)
        if not r:
            continue
        items = [(n, w["t0"].get("phases")) for n, w in r.get("workloads", {}).items()]
        items += [(c, v.get("phases")) for c, v in r.get("concurrent", {}).items()]
        for name, ph in items:
            if not ph or "round" not in ph:
                continue
            n = max(ph["round"]["n"], 1)
            keys = [k for k in ph if isinstance(ph[k], dict)]
            cells = ", ".join(f"{k} {ph[k]['ms'] / n:.2f}" for k in sorted(keys, key=lambda k: -ph[k]["ms"]))
            lines.append(f"  [{tag}] {name} ({ph['mode']}, {n} rounds): {cells}")
    for tag in ("nsys1", "nsys4"):
        r = runs.get(tag)
        if r and r.get("profile"):
            lines.append(f"  [{tag}] {json.dumps(r['profile'].get('runs'))[:300]}")
    return lines


def memory(out: Path) -> list[str]:
    lines = ["", "memory (0.5 s samplers over the whole G4-perf run)"]
    for node, f in (("head", "mem-r0.log"), ("worker", "mem-r1.log")):
        try:
            rows = [ln.split() for ln in (out / f).read_text().splitlines() if len(ln.split()) == 3]
            free = min(float(r[1]) for r in rows)
            avail = min(float(r[2]) for r in rows)
            flag = "" if avail >= 4.0 else "  << UNDER THE 4 GiB HARD FLOOR"
            lines.append(f"  {node}: MemFree min {free:.2f}, MemAvailable min {avail:.2f} GiB{flag}")
        except (OSError, ValueError):
            lines.append(f"  {node}: no samples")
    return lines


def gate(out: Path) -> list[str]:
    try:
        d = json.loads((out / "g4p-gate-report.json").read_text())
    except (OSError, ValueError):
        return []
    return ["", f"M1 gate: top-1 {d.get('top1_agreement')} (G2 0.9963), first copy {d.get('top1_first_copy')}, "
                f"serial decode {d.get('decode_tok_s_median')} tok/s (G2 19.39)"]


def main() -> int:
    out = Path(sys.argv[1])
    runs = {t: load(out, t) for t in ("fast", "eager", "nucleus0", "nccl", "phases", "nsys1", "nsys4")}
    runs = {k: v for k, v in runs.items() if v is not None}
    lines = speed_table(runs) + gates(runs) + phases(runs) + memory(out) + gate(out)
    fast = runs.get("fast", {})
    c4 = fast.get("concurrent", {}).get("C4", {}).get("aggregate_tok_s")
    lines += ["", f"PASS code >= 52: {(tok(fast, 'code') or 0) >= 52}; prose >= 38: {(tok(fast, 'prose') or 0) >= 38}; "
                  f"C4 > 37.6: {(c4 or 0) > 37.6} (C4 {c4}, TARGETS 120)"]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
