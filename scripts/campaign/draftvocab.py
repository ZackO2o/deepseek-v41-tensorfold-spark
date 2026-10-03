#!/usr/bin/env python3
"""DeepSeek-V4.1 draft vocabulary (TF_DSV41_DRAFT_HEAD=trim): the ranking file the trimmed DSpark head reads, and its
held-out coverage. Offline: no GPU, no torch (the ``tokenizers`` package only). GLM's study is glm53 repo
bench/draftvocab.py + docs/DRAFT-VOCAB.md; this is the same method on V4.1's tokenizer and halves.

Corpus: the local opencode database. Replies of DeepSeek's own Flash model (providerID ``deepseek``) are the target
text, tokenized as V4.1 emits them (reasoning, ``</think>``, text, DSML tool calls, ``<｜end▁of▁sentence｜>``); every
other text (user turns, tool outputs, other models' replies) is the prior. Classes: ``prose`` (reasoning, text
outside code fences), ``code`` (fenced code; ``content`` / ``oldString`` / ``newString`` of write / edit calls),
``tool`` (the rest of a call).

Ranking: frequency in DeepSeek replies (training split), then in the prior, then id. Per rank: rank r holds ids
r V / 2 .. (r + 1) V / 2 - 1 (V = 129,280); the trimmed head of a rank lists the first M ids of its half in the
ranking, then the half's lowest unlisted ids (``draft_head.per_rank``, the same rule). Held out by session hash.

Reported per M (rows a rank): coverage by class, and ``run_j``: the share of reply positions whose next j tokens are
all listed (a draft chain of j kept tokens needs all j listed: the acceptance ceiling of a trimmed pass).

    python3 scripts/draftvocab.py --tokenizer .cache/dsv41-tok/tokenizer.json \\
        --write-list <tf>/src/tensorfold/families/deepseek_v41/cuda/draft_vocab.txt --json results/draftvocab.json

The list holds token ids only, but its tail reflects which rare subwords appeared in these private transcripts:
regenerate it from public text before publishing an image.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

VOCAB = 129280
HALF = VOCAB // 2
SIZES = (8192, 12288, 16384, 24576, 32768)
CODE_ARGS = {"content", "oldString", "newString", "new_string", "old_string", "patch"}
DSML = "｜DSML｜"
EOS = "<｜end▁of▁sentence｜>"


def text_pieces(text: str) -> list[tuple[str, str]]:
    out, code = [], False
    for part in re.split(r"(```)", text):
        if part == "```":
            out.append((part, "code"))
            code = not code
        elif part:
            out.append((part, "code" if code else "prose"))
    return out


def tool_pieces(name: str, args: dict) -> list[tuple[str, str]]:
    """A call as V4.1 emits it (``encoding.py``: DSML invoke / parameter blocks)."""

    out = [(f'<{DSML}invoke name="{name}">\n', "tool")]
    for k, v in args.items():
        out.append((f'<{DSML}parameter name="{k}" string="{"true" if isinstance(v, str) else "false"}">', "tool"))
        out.append((v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), "code" if k in CODE_ARGS else "tool"))
        out.append((f"</{DSML}parameter>\n", "tool"))
    out.append((f"</{DSML}invoke>\n", "tool"))
    return out


def load(db: str, tok) -> tuple[list[dict], list[tuple[str, str]]]:
    """(DeepSeek replies {session, ids, classes}, prior texts (session, text))."""

    con = sqlite3.connect(f"file:{os.path.expanduser(db)}?mode=ro", uri=True)
    parts = collections.defaultdict(list)
    for mid, data in con.execute("select message_id, data from part order by message_id, id"):
        parts[mid].append(json.loads(data))
    replies, prior = [], []
    for mid, sid, data in con.execute("select id, session_id, data from message order by time_created, id"):
        m = json.loads(data)
        ps = parts.get(mid, [])
        if m.get("role") == "user":
            prior.append((sid, "".join(p.get("text", "") for p in ps if p.get("type") == "text")))
            continue
        if m.get("role") != "assistant":
            continue
        pieces = [("".join(p.get("text", "") for p in ps if p.get("type") == "reasoning"), "prose"), ("</think>", "prose")]
        calls = [p for p in ps if p.get("type") == "tool"]
        for p in ps:
            if p.get("type") == "text" and p.get("text"):
                pieces += text_pieces("\n" + p["text"])
        if calls:
            pieces.append((f"\n\n<{DSML}calls>\n", "tool"))
            for p in calls:
                pieces += tool_pieces(p.get("tool", ""), (p.get("state") or {}).get("input") or {})
            pieces.append((f"</{DSML}calls>", "tool"))
        pieces.append((EOS, pieces[-1][1]))
        if m.get("providerID") == "deepseek":
            text, owner, at = "", [], 0
            for s, c in pieces:
                if s:
                    owner.append((at, at + len(s), c))
                    text += s
                    at += len(s)
            enc = tok.encode(text, add_special_tokens=False)
            classes, j = [], 0
            for a, _ in enc.offsets:
                while j + 1 < len(owner) and a >= owner[j][1]:
                    j += 1
                classes.append(owner[j][2])
            replies.append({"session": sid, "ids": list(enc.ids), "classes": classes})
        else:
            prior.append((sid, "".join(s for s, _ in pieces)))
        for p in calls:
            o = (p.get("state") or {}).get("output")
            if isinstance(o, str) and o:
                prior.append((sid, o[:200000]))
    return replies, prior


def per_rank(order: list[int], m: int) -> set[int]:
    """``draft_head.per_rank`` for both halves: the first ``m`` ids of each half in ``order``, then its lowest
    unlisted ids."""

    out: set[int] = set()
    for lo, hi in ((0, HALF), (HALF, VOCAB)):
        mine = [i for i in order if lo <= i < hi][:m]
        taken, j = set(mine), lo
        while len(mine) < min(m, hi - lo):
            if j not in taken:
                mine.append(j)
            j += 1
        out |= set(mine)
    return out


def measure(replies: list[dict], keep: set[int], runs=(1, 2, 3, 4)) -> dict:
    hit, tot = collections.Counter(), collections.Counter()
    run_hit = collections.Counter()
    run_tot = 0
    for r in replies:
        ok = [i in keep for i in r["ids"]]
        for o, c in zip(ok, r["classes"]):
            tot[c] += 1
            hit[c] += o
        for p in range(len(ok) - max(runs)):
            run_tot += 1
            for j in runs:
                run_hit[j] += all(ok[p:p + j])
    out = {c: round(hit[c] / tot[c], 4) for c in sorted(tot)}
    out["all"] = round(sum(hit.values()) / max(1, sum(tot.values())), 4)
    out.update({f"run_{j}": round(run_hit[j] / max(1, run_tot), 4) for j in runs})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default="~/.local/share/opencode/opencode.db")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--write-list", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--holdout", type=float, default=0.2, help="share of sessions held out (by hash)")
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    replies, prior_texts = load(a.db, tok)
    held = lambda sid: int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) / 2 ** 32 < a.holdout
    prior = collections.Counter()
    for e in tok.encode_batch([t for _, t in prior_texts if t], add_special_tokens=False):
        prior.update(e.ids)
    train, test = collections.Counter(), [r for r in replies if held(r["session"])]
    full = collections.Counter()
    for r in replies:
        full.update(r["ids"])
        if not held(r["session"]):
            train.update(r["ids"])
    rank = lambda ds: sorted((i for i in set(ds) | set(prior) if i < VOCAB), key=lambda i: (-ds[i], -prior[i], i))
    order = rank(train)
    n_tok = sum(len(r["ids"]) for r in replies)
    rep = {"replies": len(replies), "reply_tokens": n_tok, "held_out_tokens": sum(len(r["ids"]) for r in test),
           "prior_tokens": sum(prior.values()), "distinct_reply_ids": len(full), "sizes": {}}
    print(f"{len(replies)} DeepSeek replies ({n_tok} tokens, {len(full)} distinct ids), prior {rep['prior_tokens']} "
          f"tokens; held out {rep['held_out_tokens']} tokens")
    print("rows a rank | " + " | ".join(("prose", "code", "tool", "all", "run_2", "run_4")))
    for m in SIZES:
        cov = measure(test, per_rank(order, m))
        rep["sizes"][m] = cov
        print(f"{m:>11} | " + " | ".join(f"{cov.get(k, 0):.4f}" for k in ("prose", "code", "tool", "all", "run_2",
                                                                           "run_4")))
    if a.write_list:
        final = rank(full)                       # the shipped list ranks on every reply (no hold-out)
        Path(a.write_list).write_text(
            "# DeepSeek-V4.1 draft vocabulary ranking (scripts/draftvocab.py in the dsv41 repo): token ids, most "
            "frequent first\n# (DeepSeek Flash replies, then the prior text, then id). Ids only; regenerate from "
            "public text before publishing.\n" + "\n".join(" ".join(map(str, final[i:i + 16]))
                                                          for i in range(0, len(final), 16)) + "\n")
        print(f"wrote {len(final)} ids to {a.write_list}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
