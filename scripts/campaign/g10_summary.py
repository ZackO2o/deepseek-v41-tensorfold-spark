#!/usr/bin/env python3
"""G10's tables from a window directory ($OUT): the boot calibration's verify ms by rows (r0 logs), m2bench's tok/s,
exactness and replies against ``off``, nsys_ramp.py's gap / lever table a config, and the pick.

    g10_summary.py speed OUT CFGS      table: verify 1 / 2 / 4 / 16 rows, code / prose tok/s, C1 / C2, exact, replies
    g10_summary.py window OUT CFGS     table: span / launches / gaps / early starts a 1- and 2-row window (nsys)
    g10_summary.py pick OUT CFGS       the adoption line (gates in G10-pdl.sh's header) -> "adopt: <cfg>" or "adopt: off"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CAL = re.compile(r"dsv41 calibration: .*verify ([0-9/]+) rows ([0-9., ]+) ms")


def calib(out: Path, tag: str) -> dict:
    for f in (out / f"{tag}-r0.log",):
        try:
            for line in f.read_text(errors="replace").splitlines():
                m = CAL.search(line)
                if m:
                    rows = [int(r) for r in m.group(1).split("/")]
                    ms = [float(v) for v in m.group(2).replace(" ", "").split(",") if v]
                    return dict(zip(rows, ms))
        except OSError:
            pass
    return {}


def report(out: Path, tag: str) -> dict | None:
    try:
        return json.loads((out / f"m2-{tag}.json").read_text())
    except (OSError, ValueError):
        return None


def replies(r: dict | None) -> dict:
    if not r:
        return {}
    return {(w, t): v[t].get("reply") for w, v in r.get("workloads", {}).items() for t in ("t0", "t07") if t in v}


def speed_rows(out: Path, cfgs: list[str]) -> list[dict]:
    base = replies(report(out, "g10s-off"))
    rows = []
    for c in cfgs:
        r = report(out, f"g10s-{c}")
        cal = calib(out, f"g10s-{c}")
        row = {"cfg": c, "verify": {k: cal.get(k) for k in (1, 2, 4, 16)}, "ok": r is not None}
        if r:
            w = r.get("workloads", {})
            row["code"] = w.get("code", {}).get("t0", {}).get("tok_s")
            row["prose"] = w.get("prose", {}).get("t0", {}).get("tok_s")
            cc = r.get("concurrent", {})
            row["c1"] = cc.get("C1", {}).get("decode_aggregate_tok_s")
            row["c2"] = cc.get("C2", {}).get("decode_aggregate_tok_s")
            row["exact"] = r.get("exact_all")
            mine = replies(r)
            row["replies_eq_off"] = (mine == base) if base and c != "off" else None
        rows.append(row)
    return rows


def fmt(v, nd=1):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def speed(out: Path, cfgs: list[str]) -> str:
    L = ["| cfg | verify 1 / 2 / 4 / 16 rows ms | code tok/s | prose tok/s | C1 / C2 decode | exact | replies == off |",
         "| --- | --- | ---: | ---: | --- | --- | --- |"]
    for r in speed_rows(out, cfgs):
        v = r["verify"]
        L.append(f"| {r['cfg']} | {' / '.join(fmt(v[k]) for k in (1, 2, 4, 16))} | {fmt(r.get('code'))} | "
                 f"{fmt(r.get('prose'))} | {fmt(r.get('c1'))} / {fmt(r.get('c2'))} | {r.get('exact')} | "
                 f"{r.get('replies_eq_off')} |")
    return "\n".join(L)


def window(out: Path, cfgs: list[str]) -> str:
    L = ["| cfg | rows | windows | span ms | launches | gaps ms | early starts (ms) | PDL est | L2PF est | calib verify-1 |",
         "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |"]
    for c in cfgs:
        try:
            res = json.loads((out / f"ramp-g10n-{c}.json").read_text())
        except (OSError, ValueError):
            L.append(f"| {c} | no nsys_ramp report | | | | | | | | |")
            continue
        cal = calib(out, f"g10n-{c}")
        for r in res:
            if not r.get("windows"):
                continue
            v = r["levers"]
            L.append(f"| {c} | {r['rows']} | {r['windows']} | {r['span_ms']} | {r['launches']} | {r['gap_ms']} | "
                     f"{r['overlaps']} ({r['overlap_ms']}) | {v['pdl_ms'][0]}-{v['pdl_ms'][1]} | "
                     f"{v['l2pf_ms'][0]}-{v['l2pf_ms'][1]} | {fmt(cal.get(1))} |")
    return "\n".join(L)


def pick(out: Path, cfgs: list[str]) -> str:
    rows = {r["cfg"]: r for r in speed_rows(out, cfgs)}
    off = rows.get("off")
    if not off or not off.get("ok"):
        return "adopt: off (no off report)"
    best, lines = None, []
    for c, r in rows.items():
        if c == "off" or not r.get("ok"):
            continue
        why = []
        if not r.get("exact") or r.get("replies_eq_off") is False:
            why.append("bits")
        v1, o1 = r["verify"].get(1), off["verify"].get(1)
        if v1 is None or o1 is None or v1 > o1 - 0.25:
            why.append(f"verify-1 {fmt(v1)} vs off {fmt(o1)} (need <= off - 0.25)")
        for k in ("code", "prose"):
            if (r.get(k) or 0) < 1.01 * (off.get(k) or 0):
                why.append(f"{k} {fmt(r.get(k))} vs off {fmt(off.get(k))} (need >= +1%)")
        if r.get("c2") is not None and off.get("c2") is not None and r["c2"] < 0.99 * off["c2"]:
            why.append(f"C2 {fmt(r['c2'])} vs off {fmt(off['c2'])} (need >= -1%)")
        lines.append(f"{c}: {'PASS' if not why else 'fail: ' + '; '.join(why)}")
        if not why and (best is None or (r.get("prose") or 0) > (rows[best].get("prose") or 0)):
            best = c
    return "\n".join(lines + [f"adopt: {best or 'off'}"])


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    step, out, cfgs = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split()
    print({"speed": speed, "window": window, "pick": pick}[step](out, cfgs))


if __name__ == "__main__":
    main()
