#!/usr/bin/env python3
"""G5-prefill.sh's summary: cold prefill tok/s per config and size against the kit and the targets, the phase split
of the fastest config, the fast tag's gate top-1, from $OUT/m2-pf-<cfg>.json (m2bench --prefill) and
$OUT/g5-gate-<cfg>.json. Prints plain text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

KIT = {8192: 1073, 32768: 1075, 65536: 1060, 131072: 1031, 262144: 983}       # BASELINE-RESULTS (2,048-token chunks)
TARGET_FULL, TARGET_REPLAY = 1500, 2200
ORDER = ["base", "exact", "exact4k", "fast", "tc", "tc4k", "tc8k", "tccfg-128", "tccfg-418", "stream", "replay",
         "replay-exact", "long", "long-replay"]


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def main(out: Path) -> None:
    runs = {}
    for f in sorted(out.glob("m2-pf-*.json")):
        r = load(f)
        if r and r.get("prefill"):
            runs[f.stem[len("m2-pf-"):]] = {row["tokens"]: row for row in r["prefill"]}
    sizes = sorted({n for r in runs.values() for n in r})
    if not runs:
        print("no prefill reports")
        return
    print("cold prefill tok/s (TTFT s), one slot")
    print("| config | " + " | ".join(f"{n // 1024}K" for n in sizes) + " |")
    print("| --- | " + " | ".join("---:" for _ in sizes) + " |")
    names = [k for k in ORDER if k in runs] + sorted(k for k in runs if k not in ORDER)
    for k in names:
        cells = []
        for n in sizes:
            row = runs[k].get(n)
            cells.append(f"{row['tok_s']:.0f} ({row['ttft_s']:.1f})" if row else "")
        print(f"| {k} | " + " | ".join(cells) + " |")
    print("| kit (vLLM) | " + " | ".join(str(KIT.get(n, "")) for n in sizes) + " |")
    print(f"| target | full {TARGET_FULL} / replay {TARGET_REPLAY} |" + " |" * (len(sizes) - 1))
    best = max(((k, n, row["tok_s"]) for k, r in runs.items() for n, row in r.items() if not k.startswith("long")),
               key=lambda t: t[2])
    print(f"\nfastest: {best[0]} at {best[1] // 1024}K: {best[2]:.0f} tok/s ({best[2] / KIT.get(best[1], 1050):.2f}x kit)")
    for k in ("exact", "fast", "tc", "replay"):
        r = runs.get(k, {}).get(32768)
        b = runs.get("base", {}).get(32768) or {"tok_s": 210.0}
        if r:
            print(f"  {k} at 32K: {r['tok_s']:.0f} tok/s, {r['tok_s'] / b['tok_s']:.1f}x the 128-row path")
    ph = runs.get(best[0], {}).get(best[1], {}).get("phases")
    if ph:
        print(f"\nphases of {best[0]} at {best[1] // 1024}K (host-timed; TF_DSV41_PHASES=1):")
        items = ph.items() if isinstance(ph, dict) else enumerate(ph)
        for name, v in list(items)[:16]:
            print(f"  {name}: {v}")
    for g in sorted(out.glob("g5-gate-*.json")):
        d = load(g)
        if d:
            print(f"\ngate {g.stem[len('g5-gate-'):]}: top-1 {d.get('top1_agreement')}, first copy "
                  f"{d.get('top1_first_copy')} (G2 exact: 0.9963 / 0.9535)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
