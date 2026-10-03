#!/usr/bin/env python3
"""G6-ship.sh's summary: one PASS / FAIL / MISSING line per ship gate against its threshold, then the speed table and
the A/B pairs. Reads $OUT (argv[1]): g6-cgate-report.json, compare-mmlu*.json, needle-*.json, chains-*.json,
tooleval/**/teb.json, structured.json, stress.json, soak.json, m2-g6speed.json, m2-g6ab-*.json, steps.tsv and the
0.5 s samplers mem-r0.log / mem-r1.log (memory minima per step window). Prints plain text; exit 1 if a gate FAILs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOP1_GATE, TOP1_AIM = 0.96, 0.98      # docs/TARGETS.md (1d83947, decision): top-1 vs the kit >= 96%, aim 98%
G4_TOP1 = 0.9962                     # G4 cgate (prod settings): reference only
KIT_MMLU = 0.875                     # the kit's MMLU-200, docs/BASELINE-RESULTS.md
FLOOR = 5.0                          # MemAvailable target (GiB), both nodes
HARD = 4.0
G4 = {"code": 59.0, "prose": 30.5, "structured": 88.2, "tweet": 93.0, "edit": 90.5, "long": 54.4,
      "C1": 59.8, "C2": 45.5, "C4": 69.8}            # G4 new defaults (every prefill change on), graphs on, RoCE
TARGET = {"code": 52.0, "prose": 38.0, "C4": 120.0}
KIT = {"code": "41.9-45", "prose": 32.5, "structured": "38-50", "C1": 32.2, "C2": 46.7, "C4": 37.6}
SPEED_TOL = 0.97                     # a regression gate: >= 97% of G4 on code, prose, C4


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def windows(out: Path) -> dict[str, tuple[float, float]]:
    w, begin = {}, {}
    try:
        for line in (out / "steps.tsv").read_text().splitlines():
            f = line.split()
            if len(f) >= 3 and f[1] == "begin":
                begin[f[0]] = float(f[2])
            elif len(f) >= 3 and f[1] == "end" and f[0] in begin:
                w[f[0]] = (begin[f[0]], float(f[2]))
    except OSError:
        pass
    return w


def mem_min(out: Path, node: str, span: tuple[float, float] | None) -> float | None:
    try:
        rows = [ln.split() for ln in (out / f"mem-{node}.log").read_text().splitlines()]
    except OSError:
        return None
    vals = [float(r[2]) for r in rows if len(r) == 3 and (span is None or span[0] <= float(r[0]) <= span[1])]
    return min(vals) if vals else None


class Gates:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.fail = 0

    def add(self, name: str, ok: bool | None, detail: str) -> None:
        tag = "MISSING" if ok is None else ("PASS" if ok else "FAIL")
        self.fail += ok is False
        self.lines.append(f"| {name} | {tag} | {detail} |")


def tok(r, name, t="t0"):
    try:
        return round(r["workloads"][name][t]["tok_s"], 1)
    except (KeyError, TypeError):
        return None


def conc(r, c):
    try:
        v = r.get("concurrent", {}).get(c) or (r.get("c4") if c == "C4" else None)
        return round(v["aggregate_tok_s"], 1)
    except (AttributeError, KeyError, TypeError):
        return None


def main() -> int:
    out = Path(sys.argv[1])
    g, win = Gates(), windows(out)
    g.lines += ["| gate | result | detail |", "| --- | --- | --- |"]

    d = load(out / "g6-cgate-report.json")
    g.add("replay: teacher-forced top-1 vs kit oracle >= 96% (aim 98%)", None if d is None else d["top1_agreement"] >= TOP1_GATE,
          "-" if d is None else f"top-1 {100 * d['top1_agreement']:.2f}% ({'meets' if d['top1_agreement'] >= TOP1_AIM else 'under'} "
                                f"the 98% aim; G4 {100 * G4_TOP1:.2f}% for reference), first copy "
                                f"{100 * d.get('top1_first_copy', 0):.2f}%")
    for f in sorted(out.glob("compare-mmlu*.json")):
        c = load(f)
        shots = f.stem.replace("compare-mmlu", "")
        name = f"replay: MMLU{'-200' if shots == '0' else f' +{shots}-shot preamble'} replay within 1 pt of full" + \
            (" and kit 87.5%" if shots == "0" else "")
        g.add(name, None if c is None else bool(c["ok"]),
              "-" if c is None else f"replay {100 * c['replay']:.1f}%, full {100 * c['full']:.1f}%, delta "
                                    f"{c['delta_points']:+.1f}, same answer {c['same']}/{c['n']}")
    if not list(out.glob("compare-mmlu*.json")):
        g.add("replay: MMLU replay vs full vs kit", None, "no compare-mmlu*.json")
    for m in ("replay", "full"):
        n = load(out / f"needle-{m}.json")
        g.add(f"replay: needles ({m})", None if n is None else bool(n["ok"]),
              "-" if n is None else ", ".join(f"{x['size'] // 1024}K {'found' if x['found'] else 'MISSED'}"
                                              for x in n["needles"]))
    for md in ("off", "high"):
        c = load(out / f"chains-{md}.json")
        g.add(f"tooleval: multi-step chains ({md}) >= 10/12", None if c is None else bool(c["pass"]),
              "-" if c is None else f"score {c['scores']} of {c['max_points']}, malformed calls {c['malformed']}")
    tebs = sorted(out.glob("tooleval/**/teb.json"))
    if tebs:
        t = load(tebs[-1]) or {}
        score = t.get("score", t.get("total_score", t.get("summary", {}).get("score") if isinstance(t.get("summary"), dict) else None))
        g.add("tooleval: tool-eval-bench C >= GLM prod (8/8, score 100)", None if score is None else float(score) >= 100,
              f"score {score} ({tebs[-1].relative_to(out)})")
    else:
        g.add("tooleval: tool-eval-bench C >= GLM prod (8/8)", None, "not run here (no install); run it from the workstation")
    s = load(out / "structured.json")
    g.add("structured: schemas + tool choice suites all pass", None if s is None else bool(s["pass"]),
          "-" if s is None else "; ".join(f"{k} {'ok' if v['pass'] else str(len(v['fails'])) + ' fails'}"
                                           for k, v in s.items() if isinstance(v, dict)))
    st = load(out / "stress.json")
    span = win.get("stress")
    h, w = mem_min(out, "r0", span), mem_min(out, "r1", span)
    g.add("stress: decode streams reach 2,048 tokens (ignore_eos)", None if st is None else bool(st.get("decode_full")),
          "-" if st is None else ", ".join(f"{x.get('tokens')} tok ({x.get('during_prefill_tok_s')} tok/s during the long "
                                           f"prefill)" for x in st["streams"][1:]) + f"; long first token {st.get('long_first_token_s')} s")
    g.add(f"stress: MemAvailable >= {FLOOR} GiB both nodes", None if h is None or w is None or span is None else min(h, w) >= FLOOR,
          f"head {h}, worker {w} GiB (hard stop {HARD})")
    so = load(out / "soak.json")
    span = win.get("soak")
    h, w = mem_min(out, "r0", span), mem_min(out, "r1", span)
    g.add("soak: 0 errors, inflight back to 0, 17*23, no fatal", None if so is None else bool(so["pass"]),
          "-" if so is None else f"{so['requests']} requests, {so['errors']} errors, {so['cancelled']} cancelled, "
                                 f"{so['check_fails']} content checks failed, drained {so['drained']} in {so['drain_s']} s, "
                                 f"17*23 -> {so['answer_17x23']!r}")
    g.add(f"soak: MemAvailable >= {HARD} GiB (target {FLOOR})", None if h is None or w is None or span is None else min(h, w) >= HARD,
          f"head {h}, worker {w} GiB")
    sp = load(out / "m2-g6speed.json")
    vals = {n: tok(sp, n) for n in ("code", "prose")} | {"C4": conc(sp, "C4")} if sp else {}
    ok = None if not sp else all((vals.get(k) or 0) >= SPEED_TOL * G4[k] for k in ("code", "prose", "C4")) and bool(sp.get("exact_all", True))
    g.add(f"speed: code / prose / C4 >= {int(SPEED_TOL * 100)}% of G4, drafted == serial", ok,
          "-" if not sp else f"code {vals['code']}, prose {vals['prose']}, C4 {vals['C4']} (G4 59.0 / 30.5 / 69.8); exact {sp.get('exact_all')}")

    lines = ["G6 ship gates", ""] + g.lines
    if sp:
        lines += ["", "speed (tok/s, graphs on, RoCE; T = 0 / T = 0.7)", "| workload | G6 | G4 | target | kit |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for n in ("code", "prose", "structured", "tweet", "edit", "long"):
            lines.append(f"| {n} | {tok(sp, n)} / {tok(sp, n, 't07')} | {G4[n]} | {TARGET.get(n, '')} | {KIT.get(n, '')} |")
        for c in ("C1", "C2", "C4"):
            lines.append(f"| {c} | {conc(sp, c)} | {G4[c]} | {TARGET.get(c, '')} | {KIT.get(c, '')} |")
    abs_ = sorted(out.glob("m2-g6ab-*.json"))
    if abs_:
        lines += ["", "A/B pairs (code / prose single stream T = 0, C4 aggregate; exact = drafted == serial)",
                  "| pair | code | prose | C4 | exact |", "| --- | ---: | ---: | ---: | --- |"]
        for f in abs_:
            r = load(f) or {}
            lines.append(f"| {f.stem.replace('m2-g6ab-', '')} | {tok(r, 'code')} | {tok(r, 'prose')} | {conc(r, 'C4')} | "
                         f"{r.get('exact_all')} |")
    lines += ["", "memory minima per step (MemAvailable GiB, head / worker)"]
    for name, span in win.items():
        lines.append(f"  {name}: {mem_min(out, 'r0', span)} / {mem_min(out, 'r1', span)}")
    lines += ["", f"verdict: {'SHIP' if g.fail == 0 and not any('MISSING' in x for x in g.lines) else 'NOT YET'} "
                  f"({g.fail} FAIL, {sum('MISSING' in x for x in g.lines)} MISSING)"]
    print("\n".join(lines))
    return 1 if g.fail else 0


if __name__ == "__main__":
    sys.exit(main())
