#!/usr/bin/env python3
"""G10 section budget's quality and report helpers (G10-budget.sh): expert-budgeted verification
(TF_DSV41_VERIFY_BUDGET, engine cuda/budget.py), a LOSSY speed mode. Standard library only (runs on the head).

  g10budget.py greedy --base URL --model M --label L --out G.json [--tokens 256]
      32 fixed prompts (code / prose / structured / reasoning), greedy, thinking off, ignore_eos, return_token_ids:
      every reply's token ids.
  g10budget.py agree --ref EXACT.json --run B.json [--out A.json]
      per prompt: identical tokens (position by position, over the exact reply's length) and the first divergence;
      overall % identical, prompts fully identical, median / min first divergence.
  g10budget.py mmlugen --base URL --model M --data mmlu200.jsonl --label L --out R.json [--max-tokens 320]
      MMLU-200 0-shot, GENERATION-based: "think step by step briefly, end with 'Answer: X'", greedy, thinking off.
      (shipq.py's MMLU reads the reply's FIRST letter: under the budget the first generated token is always exact,
      so it cannot see this mode; here every answer comes after many approximately verified tokens.)
  g10budget.py pick OUT "exact 8 4 2 0"
      the table (speed C1 code / prose / structured, C2, C4, tokens and rows a round; identical %, first divergence;
      MMLU-gen; chains) and "adopt: <B | none>": the fastest B with MMLU-gen >= 0.865 AND >= 95% identical greedy
      tokens (speed: the geometric mean of decode tok/s vs exact over code C1, prose C1, structured C1, C2, C4).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

MMLU_MIN, IDENT_MIN = 0.865, 0.95
LETTERS = "ABCD"
PROMPTS = [
    # code
    "Write a Python function that merges overlapping intervals and explain its complexity.",
    "Implement a thread-safe LRU cache in Go with Get and Put methods.",
    "Write a Bash script that finds the 10 largest files under a directory and prints their sizes.",
    "Write a SQL query that returns the second highest salary per department, with an explanation.",
    "Implement binary search in Rust over a sorted slice of i64 and write two unit tests.",
    "Write a TypeScript React component for a debounced search box.",
    "Explain and implement Dijkstra's algorithm in C++ using a priority queue.",
    "Write a Python script that parses an Apache access log and counts requests per status code.",
    # prose
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain to a ten-year-old why the sky is blue.",
    "Write a persuasive essay on why cities should invest in public libraries.",
    "Describe the history of the printing press and its impact on Europe.",
    "Write a letter to a friend describing a trip to the mountains in autumn.",
    "Summarize the causes of the First World War in a few paragraphs.",
    "Write a product description for a handmade ceramic coffee mug.",
    "What are the pros and cons of remote work? Discuss in detail.",
    # structured
    "Return a JSON object describing three fictional employees with name, age, role and skills arrays.",
    "Produce a Markdown table comparing five programming languages by typing, speed and main use.",
    "List the planets of the solar system as YAML with their order, type and number of moons.",
    "Write a CSV with ten rows of fictional sales data: date, region, product, units, revenue.",
    "Give a numbered checklist for deploying a web application to production.",
    "Write an OpenAPI 3 YAML snippet for a REST endpoint that creates a user.",
    "Return a JSON array of the first 15 prime numbers with their index.",
    "Write an XML document describing a library with three books.",
    # reasoning / math
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip? Show the steps.",
    "Solve 3x + 7 = 2x - 5 and check the answer.",
    "If a rectangle has perimeter 36 and area 80, what are its sides? Explain.",
    "Prove that the sum of two even numbers is even.",
    "How many ways can 5 people sit around a round table? Explain the reasoning.",
    "Explain the Monty Hall problem and why switching is better.",
    "Compute 17 * 23 and 289 / 17 step by step.",
    "What is the derivative of x^3 * sin(x)? Show the product rule.",
]


def post(base: str, path: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def greedy(a) -> int:
    out, t0 = [], time.time()
    for i, p in enumerate(PROMPTS):
        body = {"model": a.model, "messages": [{"role": "user", "content": p}], "temperature": 0,
                "max_tokens": a.tokens, "ignore_eos": True, "return_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = post(a.base, "/v1/chat/completions", body)
            ids = (r.get("tensorfold") or {}).get("token_ids") or []
            text = r["choices"][0]["message"].get("content") or ""
        except Exception as exc:                    # noqa: BLE001 - recorded
            ids, text = [], f"ERROR {type(exc).__name__}: {exc}"[:200]
        out.append({"i": i, "ids": [int(t) for t in ids], "text": text[:2000]})
        print(f"[g10budget] greedy {a.label} {i}: {len(ids)} tokens", flush=True)
    Path(a.out).write_text(json.dumps({"label": a.label, "tokens": a.tokens, "items": out,
                                       "wall_s": round(time.time() - t0, 1)}, indent=1))
    return 0


def agreement(ref: dict, run: dict) -> dict:
    by = {x["i"]: x["ids"] for x in run["items"]}
    same = total = full = 0
    firsts = []
    for x in ref["items"]:
        a, b = x["ids"], by.get(x["i"], [])
        n = len(a)
        if not n:
            continue
        eq = sum(1 for k in range(n) if k < len(b) and a[k] == b[k])
        first = next((k for k in range(n) if k >= len(b) or a[k] != b[k]), n)
        same, total = same + eq, total + n
        full += first == n
        firsts.append(first)
    firsts.sort()
    return {"identical": same / total if total else None, "prompts": len(firsts), "fully_identical": full,
            "first_div_median": firsts[len(firsts) // 2] if firsts else None,
            "first_div_min": firsts[0] if firsts else None}


def agree(a) -> int:
    r = agreement(json.loads(Path(a.ref).read_text()), json.loads(Path(a.run).read_text()))
    print(f"[g10budget] agree {a.run}: identical {r['identical']}, fully identical {r['fully_identical']}/"
          f"{r['prompts']}, first divergence median {r['first_div_median']} min {r['first_div_min']}")
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
    return 0


def question(q: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    return (f"The following is a multiple choice question about {q['subject'].replace('_', ' ')}.\n\n"
            f"{q['question']}\n{opts}\n\nThink step by step briefly, then end your reply with a line of the form "
            f"'Answer: X' where X is the letter of the correct option.")


def mmlugen(a) -> int:
    rows = [json.loads(line) for line in Path(a.data).read_text().splitlines() if line.strip()]
    t0, correct, items = time.time(), 0, []
    for i, q in enumerate(rows):
        body = {"model": a.model, "messages": [{"role": "user", "content": question(q)}], "temperature": 0,
                "max_tokens": a.max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = post(a.base, "/v1/chat/completions", body)
            reply = r["choices"][0]["message"].get("content") or ""
            m = re.findall(r"Answer:\s*\**\s*\(?([ABCD])\b", reply)
            got = LETTERS.index(m[-1]) if m else -1
            n = int(r.get("usage", {}).get("completion_tokens") or 0)
        except Exception as exc:                    # noqa: BLE001
            reply, got, n = f"ERROR {type(exc).__name__}: {exc}"[:200], -2, 0
        correct += got == q["answer"]
        items.append({"i": i, "answer": q["answer"], "got": got, "completion_tokens": n, "reply": reply[-120:]})
    acc = correct / max(len(rows), 1)
    rec = {"label": a.label, "accuracy": acc, "correct": correct, "n": len(rows),
           "unparsed": sum(x["got"] == -1 for x in items), "errors": sum(x["got"] == -2 for x in items),
           "completion_tokens_median": sorted(x["completion_tokens"] for x in items)[len(items) // 2] if items else 0,
           "wall_s": round(time.time() - t0, 1), "items": items}
    print(f"[g10budget] mmlugen {a.label}: {acc:.3f} ({correct}/{len(rows)}), unparsed {rec['unparsed']}, errors "
          f"{rec['errors']}, median {rec['completion_tokens_median']} tokens", flush=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


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


def rounds(r) -> str:
    if r is None:
        return ""
    out = []
    for n, w in r.get("workloads", {}).items():
        d = (w.get("t0") or {}).get("drafting") or {}
        win = d.get("windows") or 0
        rows = f"{d['rows'] / win:.2f}" if d.get("rows") and win else "?"
        out.append(f"{n} {d.get('tokens_per_round') or d.get('dspark_tokens_per_round')} tok / {rows} rows")
    return ", ".join(out)


def pick(out: str, cfgs: list[str]) -> list[str]:
    base = rates(_load(out, "m2-g10b-exact.json"))
    lines, ok = [], []
    for c in cfgs:
        m = _load(out, f"m2-g10b-{c}.json")
        ag = _load(out, f"agree-{c}.json") if c != "exact" else {"identical": 1.0, "first_div_median": None}
        mm = _load(out, f"mmlugen-{c}.json")
        ch = _load(out, f"chains-{c}.json")
        rt = rates(m)
        ratio = [rt[k] / base[k] for k in base if k in rt and base[k]]
        score = math.exp(sum(math.log(x) for x in ratio) / len(ratio)) if ratio else None
        ident = (ag or {}).get("identical")
        acc = (mm or {}).get("accuracy")
        r4 = (lambda x: None if x is None else round(x, 4))                                  # noqa: E731
        lines.append(f"B {c:5s} " + ", ".join(f"{k} {v}" for k, v in rt.items()) + f"; speed x{r4(score)}; "
                     f"{rounds(m)}; identical {r4(ident)} (first div median {(ag or {}).get('first_div_median')}, "
                     f"fully {(ag or {}).get('fully_identical')}); mmlu-gen {r4(acc)}; chains {(ch or {}).get('scores')}")
        if c != "exact" and score and ident is not None and acc is not None and ident >= IDENT_MIN and acc >= MMLU_MIN:
            ok.append((score, c))
    ok.sort(reverse=True)
    lines.append(f"adopt: {ok[0][1] if ok and ok[0][0] > 1.0 else 'none'} (fastest B with mmlu-gen >= {MMLU_MIN} and "
                 f">= {100 * IDENT_MIN:.0f}% identical greedy tokens, faster than exact)")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("greedy", "mmlugen"):
        p = sub.add_parser(name)
        p.add_argument("--base", default="http://127.0.0.1:8001")
        p.add_argument("--model", default="DeepSeek-V4.1-Flash-TF")
        p.add_argument("--label", default="")
        p.add_argument("--out", required=True)
    sub.choices["greedy"].add_argument("--tokens", type=int, default=256)
    sub.choices["mmlugen"].add_argument("--data", required=True)
    sub.choices["mmlugen"].add_argument("--max-tokens", type=int, default=320)
    g = sub.add_parser("agree")
    g.add_argument("--ref", required=True)
    g.add_argument("--run", required=True)
    g.add_argument("--out", default="")
    k = sub.add_parser("pick")
    k.add_argument("dir")
    k.add_argument("cfgs")
    a = ap.parse_args(argv)
    if a.cmd == "pick":
        print("\n".join(pick(a.dir, a.cfgs.split())))
        return 0
    return {"greedy": greedy, "agree": agree, "mmlugen": mmlugen}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
