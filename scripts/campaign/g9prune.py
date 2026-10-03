#!/usr/bin/env python3
"""G9 section prune's report helpers (G9.sh):

    g9prune.py experts WINDOW.json          experts ms / span ms a window group (nsys_window.py's JSON)
    g9prune.py pick OUT "cfg cfg ..."       a line a setting (gate top-1, C1 / C2 / C4 tok/s, speed vs off, exact_all,
                                            MMLU-200 0-shot), then "mmlu: <best two>" and "adopt: <cfg | none>"

Adoption: the fastest setting (geometric mean of its decode tok/s ratios vs off over code C1, prose C1, C2, C4) with
top-1 >= 0.97, exact_all and MMLU-200 0-shot >= 0.865.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys

TOP1, MMLU = 0.97, 0.865


def experts(path: str) -> str:
    try:
        r = json.load(open(path))
    except (OSError, ValueError) as e:
        return f"no window report ({e})"

    def key(k):
        m = re.search(r"(\d+)", k)
        return (k.startswith("draft"), int(m.group(1)) if m else 0)

    return "; ".join(f"{k} x{g['windows']}: experts {g['classes_ms'].get('experts', 0):.2f} / span {g['span_ms']:.1f} ms"
                     for k, g in sorted(r.get("groups", {}).items(), key=lambda kv: key(kv[0])))


def _load(out: str, name: str):
    try:
        return json.load(open(os.path.join(out, name)))
    except (OSError, ValueError):
        return None


def rates(r) -> dict:
    if r is None:
        return {}
    v = {f"{n} C1": (w.get("t0") or {}).get("tok_s") for n, w in r.get("workloads", {}).items()}
    v.update({k: (x or {}).get("decode_aggregate_tok_s") for k, x in r.get("concurrent", {}).items()
              if k in ("C2", "C4")})
    return {k: x for k, x in v.items() if x}


def pick(out: str, cfgs: list[str]) -> list[str]:
    base = rates(_load(out, "m2-g9p-off.json"))
    lines, cand = [], []
    for c in cfgs:
        g, m, mm = _load(out, f"g5-gate-g9prune-{c}.json"), _load(out, f"m2-g9p-{c}.json"), \
            _load(out, f"mmlu0-prune-{c}.json")
        t1 = g.get("top1_agreement") if g else None
        fc = g.get("top1_first_copy") if g else None
        rt = rates(m)
        ratio = [rt[k] / base[k] for k in base if k in rt and base[k]]
        score = math.exp(sum(math.log(x) for x in ratio) / len(ratio)) if ratio else None
        acc = mm.get("accuracy") if mm else None
        exact = m.get("exact_all") if m else None
        r4 = (lambda x: None if x is None else round(x, 4))                                  # noqa: E731
        lines.append(f"{c:6s} top1 {r4(t1)} (first copy {r4(fc)}); " + ", ".join(f"{k} {v}" for k, v in rt.items())
                     + f"; speed x{r4(score)}; exact_all {exact}; mmlu0 {acc}")
        if c != "off" and t1 is not None and t1 >= TOP1 and score and exact:
            cand.append((score, c, acc))
    cand.sort(reverse=True)
    lines.append("mmlu: " + " ".join(c for _, c, _ in cand[:2]))
    ok = [c for _, c, a in cand if a is not None and a >= MMLU]
    lines.append(f"adopt: {ok[0] if ok else 'none'}")
    return lines


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "experts":
        print(experts(sys.argv[2]))
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "pick":
        print("\n".join(pick(sys.argv[2], sys.argv[3].split())))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
