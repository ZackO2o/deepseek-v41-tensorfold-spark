"""M1 gate 2 (greedy replies == the kit's on >= 6 of 8), measured so that a failure says why.

G2's capture ran the kit with DSpark on (k = 3): vLLM's batched verify is not bit-exact with its own serial greedy,
and it kept only the kit's top-1 logprob, so a divergence could not be told from a kit near-tie. This tool:

1. ``capture`` (kit up with ``SPEC_METHOD=none``; stdlib only): each oracle prompt's first copy (BOS + one period,
   as ``oracle_prompt_logprobs.py greedy`` and our gate send it), ``--decode`` greedy tokens with EOS ignored, and
   the kit's top-``--k`` logprobs at every step (``return_tokens_as_token_ids``)::

       python3 tests/reference/gate2.py capture --capture results/BASELINE-20261001/oracle-kit.json \\
           --base http://127.0.0.1:8888 --spec none --out results/G4/kit-greedy-nospec.json

2. ``oracle``: the capture as an oracle file (ids = prompt + the kit's reply, the kit's top-k at every reply
   position, none on the prompt), so ``gate --oracle`` teacher-forces our engine along the kit's own trajectory: top-1
   agreement over the reply positions, EOS / BOS mid-reply included, with the kit's margins at our misses.

3. ``score``: the capture against our gate report(s): per prompt the first divergence, the kit's top-1 / top-2
   margin there and our token's rank in the kit's top-k, and a verdict: ``identical``, ``tie`` (the kit's margin <=
   ``--tie`` and our token is its #2: bf16 noise decides) or ``real``. With ``--forced REPORT`` (the gate on the
   ``oracle`` file) also the teacher-forced agreement on reply positions. Gate 2 passes on ``--need`` identical
   replies; the report adds how many are identical or end at a kit tie.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def _post(base: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def period(ids: list[int]) -> int:
    """The first copy's length after BOS (the oracle repeats BOS + text; as the family's gate.py)."""

    n = len(ids)
    for k in range(1, n - 1):
        if all(ids[1 + i] == ids[1 + i + k] for i in range(n - 1 - k)):
            return k
    return n - 1


def _tid(key: str) -> int:
    return int(key.split(":", 1)[1]) if key.startswith("token_id:") else int(key)


def parse_choice(choice: dict) -> tuple[list[int], list[list[list[float]]]]:
    """(reply ids, per step [[id, logprob], ...] best first) from a /v1/completions choice with token-id logprobs."""

    lp = choice["logprobs"]
    reply = [_tid(t) for t in lp["tokens"]]
    tops = []
    for step in lp.get("top_logprobs") or [{}] * len(reply):
        tops.append(sorted(([_tid(k), float(v)] for k, v in (step or {}).items()), key=lambda e: (-e[1], e[0])))
    return reply, tops


def capture(a) -> int:
    rec = json.loads(Path(a.capture).read_text())
    model = a.model or _get(a.base, "/v1/models")["data"][0]["id"]
    out = {"meta": {"base": a.base, "model": model, "decode": a.decode, "k": a.k, "spec": a.spec,
                    "capture": a.capture, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, "prompts": []}
    for i, p in enumerate(rec["prompts"][: a.max_prompts]):
        ids = p["ids"][: 1 + period(p["ids"])]
        t = time.time()
        r = _post(a.base, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": a.decode, "temperature": 0,
                                              "ignore_eos": True, "logprobs": a.k, "return_tokens_as_token_ids": True,
                                              "skip_special_tokens": False})
        reply, tops = parse_choice(r["choices"][0])
        out["prompts"].append({"prompt": ids, "reply": reply, "top": tops})
        print(f"prompt {i}: {len(ids)} prompt tokens, {len(reply)} reply tokens, {time.time() - t:.1f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}")
    return 0


def to_oracle(cap: dict) -> dict:
    """The capture as ``oracle-kit.json``: positions[p] = the kit's top-k for ids[p] given ids[:p]."""

    prompts = []
    for p in cap["prompts"]:
        ids = list(p["prompt"]) + list(p["reply"])
        pos: list = [None] * len(p["prompt"])
        for tok, top in zip(p["reply"], p["top"]):
            pos.append({"token": tok, "top": top, "token_logprob": next((lp for t, lp in top if t == tok), None)})
        prompts.append({"ids": ids, "positions": pos})
    return {"meta": dict(cap.get("meta", {}), kind="kit greedy as oracle (gate2.py oracle)"), "prompts": prompts}


def oracle(a) -> int:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(to_oracle(json.loads(Path(a.capture).read_text()))))
    print(f"wrote {a.out}")
    return 0


def first_divergence(x: list[int], y: list[int]) -> int:
    for i, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return i
    return min(len(x), len(y))


def verdict(kit: dict, ours: list[int], tie: float = 0.125) -> dict:
    reply, tops = kit["reply"], kit["top"]
    n = min(len(reply), len(ours))
    d = first_divergence(reply, ours)
    row = {"first_divergence": d, "identical": d >= n}
    if d >= n:
        row["verdict"] = "identical"
        return row
    top = tops[d] if d < len(tops) else []
    ranks = [t for t, _ in top]
    row["kit"], row["ours"] = reply[d], ours[d]
    row["kit_margin"] = round(top[0][1] - top[1][1], 4) if len(top) > 1 else None
    row["ours_kit_rank"] = ranks.index(ours[d]) + 1 if ours[d] in ranks else None
    row["ours_kit_gap"] = round(top[0][1] - top[ranks.index(ours[d])][1], 4) if ours[d] in ranks else None
    is_tie = row["ours_kit_rank"] == 2 and row["kit_margin"] is not None and row["kit_margin"] <= tie
    row["verdict"] = "tie" if is_tie else "real"
    return row


def forced(report: dict, cap: dict) -> list[dict]:
    """Teacher-forced agreement on reply positions from our gate's report on the ``oracle`` file."""

    rows = []
    for p, k in zip(report["prompts"], cap["prompts"]):
        P, n = len(k["prompt"]), len(k["reply"])
        miss = [m for m in p.get("misses", []) if m[0] >= P]
        ties = [m for m in miss if m[3] is not None and m[3] <= 0.125]
        rows.append({"reply_positions": n, "misses": len(miss), "top1": round(1 - len(miss) / max(n, 1), 4),
                     "misses_not_tie": len(miss) - len(ties), "first_miss": (miss[0][0] - P) if miss else None})
    return rows


def score(a) -> int:
    cap = json.loads(Path(a.capture).read_text())
    rep = json.loads(Path(a.ours).read_text())
    ours = [p["greedy_reply"] for p in rep["prompts"]]
    rows = [verdict(k, o, a.tie) for k, o in zip(cap["prompts"], ours)]
    out = {"capture": a.capture, "ours": a.ours, "rows": rows,
           "identical": sum(r["identical"] for r in rows),
           "identical_or_tie": sum(r["verdict"] in ("identical", "tie") for r in rows)}
    if a.forced:
        fr = forced(json.loads(Path(a.forced).read_text()), cap)
        out["forced"] = fr
        tot = sum(r["reply_positions"] for r in fr)
        out["forced_top1"] = round(1 - sum(r["misses"] for r in fr) / max(tot, 1), 4)
        out["forced_top1_without_ties"] = round(1 - sum(r["misses_not_tie"] for r in fr) / max(tot, 1), 4)
    out["pass"] = out["identical"] >= a.need
    for i, r in enumerate(rows):
        extra = "" if r["identical"] else (f" kit {r['kit']} ours {r['ours']}, kit margin {r['kit_margin']}, "
                                           f"ours = kit #{r['ours_kit_rank']} (gap {r['ours_kit_gap']})")
        f = f"; forced top-1 {out['forced'][i]['top1']}" if a.forced else ""
        print(f"prompt {i}: {r['verdict']} at {r['first_divergence']}{extra}{f}")
    print(f"gate 2: {out['identical']} of {len(rows)} identical (need {a.need}): {'PASS' if out['pass'] else 'FAIL'}; "
          f"identical or a kit tie: {out['identical_or_tie']}"
          + (f"; forced top-1 {out['forced_top1']} ({out['forced_top1_without_ties']} without ties)" if a.forced else ""))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--capture", required=True, help="the oracle capture (its prompts)")
    c.add_argument("--base", default="http://127.0.0.1:8888")
    c.add_argument("--model", default="")
    c.add_argument("--decode", type=int, default=256)
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--max-prompts", type=int, default=8)
    c.add_argument("--spec", default="none", help="the kit's SPEC_METHOD for the record (set it on the kit)")
    c.add_argument("--out", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--capture", required=True)
    o.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--capture", required=True)
    s.add_argument("--ours", required=True, help="our gate report (greedy_reply per prompt)")
    s.add_argument("--forced", default="", help="our gate report on the oracle file (teacher-forced)")
    s.add_argument("--tie", type=float, default=0.125)
    s.add_argument("--need", type=int, default=6)
    s.add_argument("--out", default="")
    a = ap.parse_args(argv)
    return {"capture": capture, "oracle": oracle, "score": score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
