"""M1 oracle: the reference's next-token argmax against the MiaAI kit's ``prompt_logprobs`` (pass: top-1 >= 99%).

Two steps, so the kit and the reference never need the GPU at the same time:

1. ``capture`` (kit up; stdlib only, runs anywhere that reaches the API):

       python3 tests/reference/oracle_prompt_logprobs.py capture --base http://127.0.0.1:8888 \\
           --tokenizer .cache/ckpt/dsv41-uncensored-2.9bpw/files/tokenizer.json \\
           --tokens 2048 --out results/reference/oracle-kit.json

   Sends each prompt as token ids (BOS + text, truncated to ``--tokens``) to ``/v1/completions`` with
   ``max_tokens=1, temperature=0, prompt_logprobs=K`` and stores, per position, the kit's top-K ids / logprobs and
   the logprob and rank of the actual token. Prompts: ``--prompts FILE`` (JSONL, ``{"text": ...}`` or
   ``{"ids": [...]}``) or the 8 built-in ones (prose, code, math, chat-formatted, multilingual, structured data,
   a long instruction, repetition), each repeated / extended to the requested length.

2. ``compare`` (kit down; needs torch; the full model streams one layer at a time, so run it where the
   checkpoint is local, e.g. on head in a container with torch + CUDA):

       python tests/reference/oracle_prompt_logprobs.py compare --capture results/reference/oracle-kit.json \\
           --model /models/dsv41-uncensored-2.9bpw --engram /models/dsv41-engram-src --device cuda \\
           --token-map results/reference/engram-token-map.json --report results/reference/oracle-report.json

   ``--remote HOST`` reads the checkpoint over ssh instead (fine for ``--layers`` debugging runs, not for a full
   pass: that would move the whole checkpoint). Exit status 0 when the top-1 agreement reaches ``--threshold``.

``token-map`` writes the Engram compressed-vocabulary map (needs the ``tokenizers`` package) for hosts without it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BUILTIN_PROMPTS = [
    "The history of the printing press begins in East Asia, where woodblock printing was used for centuries before "
    "movable type appeared. In Europe, Johannes Gutenberg's press of around 1440 combined",
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n    i = j = 0\n    out = []\n"
    "    while i < len(a) and j < len(b):\n",
    "Problem: A train leaves at 9:40 and travels 210 km at an average speed of 84 km/h. When does it arrive?\n"
    "Solution: The travel time is 210 / 84 = 2.5 hours, so",
    "<｜User｜>Explain, step by step, why the sky looks blue during the day and red at sunset.<｜Assistant｜>",
    "Le château de Versailles fut d'abord un pavillon de chasse construit en 1623 par Louis XIII. 后来，路易十四"
    "将其扩建为欧洲最宏伟的宫殿之一。Im Jahr 1789",
    '{"id": 1042, "name": "Ada Lovelace", "born": 1815, "fields": ["mathematics", "computing"], "notes": "',
    "Write a detailed, well-organized report on the causes of the 2008 financial crisis. Cover mortgage lending, "
    "securitization, leverage, rating agencies and the policy response, and end with lessons learned.\n\n1.",
    "one two three four five six seven eight nine ten one two three four five six seven eight nine ten one two",
]


# -- capture ----------------------------------------------------------------------------------------------------

def _post(base: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read())


def _encode(tokenizer_json: str | None, base: str, model: str, text: str) -> list[int]:
    if tokenizer_json:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tokenizer_json).encode(text, add_special_tokens=False).ids
    r = _post(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})
    return r["tokens"]


def load_prompts(args) -> list[list[int]]:
    items: list[dict] = []
    if args.prompts:
        for line in Path(args.prompts).read_text().splitlines():
            if line.strip():
                items.append(json.loads(line))
    else:
        items = [{"text": t} for t in BUILTIN_PROMPTS]
    out = []
    for it in items[: args.max_prompts]:
        if "ids" in it:
            ids = list(it["ids"])
        else:
            ids = _encode(args.tokenizer, args.base, args.model, it["text"])
            base_ids = list(ids)
            while len(ids) + 1 < args.tokens and base_ids:      # extend short prompts by repetition
                ids += base_ids
            ids = [args.bos] + ids
        out.append(ids[: args.tokens])
    return out


def capture(args) -> int:
    if not args.model:
        args.model = _get(args.base, "/v1/models")["data"][0]["id"]
    prompts = load_prompts(args)
    rec = {"meta": {"base": args.base, "model": args.model, "k": args.k, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tokens": args.tokens}, "prompts": []}
    for i, ids in enumerate(prompts):
        t = time.time()
        r = _post(args.base, "/v1/completions", {"model": args.model, "prompt": ids, "max_tokens": 1,
                                                 "temperature": 0, "prompt_logprobs": args.k, "logprobs": args.k})
        pl = r["choices"][0].get("prompt_logprobs")
        if pl is None:
            raise SystemExit("the server returned no prompt_logprobs (vLLM: pass prompt_logprobs in the body)")
        positions = []
        for pos, entry in enumerate(pl):
            if entry is None:
                positions.append(None)
                continue
            top = sorted(((int(tid), v["logprob"], v.get("rank", 0)) for tid, v in entry.items()), key=lambda x: x[2])
            actual = entry.get(str(ids[pos]))
            positions.append({"token": ids[pos], "top": [[tid, lp] for tid, lp, rk in top if rk and rk <= args.k],
                              "token_logprob": None if actual is None else actual["logprob"],
                              "token_rank": None if actual is None else actual.get("rank")})
        rec["prompts"].append({"ids": ids, "positions": positions})
        print(f"prompt {i}: {len(ids)} tokens, {time.time() - t:.1f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec))
    print(f"wrote {args.out}")
    return 0


# -- greedy (M1 gate 2) -------------------------------------------------------------------------------------------

def period(ids: list[int]) -> int:
    """The first copy's length after BOS (as the family's gate.py): the capture repeats BOS + text to length."""

    n = len(ids)
    for k in range(1, n - 1):
        if all(ids[1 + i] == ids[1 + i + k] for i in range(n - 1 - k)):
            return k
    return n - 1


def greedy(args) -> int:
    """The kit's greedy reply (``--decode`` tokens, EOS ignored) after each oracle prompt's first copy, as token ids
    (``return_tokens_as_token_ids``): what gate 2 compares our greedy replies with."""

    rec = json.loads(Path(args.capture).read_text())
    if not args.model:
        args.model = _get(args.base, "/v1/models")["data"][0]["id"]
    out = {"meta": {"base": args.base, "model": args.model, "decode": args.decode, "capture": args.capture,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}, "prompts": [], "replies": []}
    for i, p in enumerate(rec["prompts"][: args.max_prompts]):
        ids = p["ids"][: 1 + period(p["ids"])]
        t = time.time()
        r = _post(args.base, "/v1/completions", {"model": args.model, "prompt": ids, "max_tokens": args.decode,
                                                 "temperature": 0, "ignore_eos": True, "logprobs": 1,
                                                 "return_tokens_as_token_ids": True, "skip_special_tokens": False})
        toks = r["choices"][0]["logprobs"]["tokens"]
        reply = [int(x.split(":", 1)[1]) for x in toks]
        out["prompts"].append({"prompt_tokens": len(ids), "text": r["choices"][0]["text"]})
        out["replies"].append(reply)
        print(f"prompt {i}: {len(ids)} prompt tokens, {len(reply)} reply tokens, {time.time() - t:.1f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")
    return 0


# -- compare ----------------------------------------------------------------------------------------------------

def score(capture_rec: dict, logits_by_prompt: list) -> dict:
    """Top-1 agreement of argmax(logits[i - 1]) with the kit's rank-1 token at position i, plus diagnostics."""

    import torch

    agree = total = top5 = 0
    lp_err = []
    per_prompt = []
    for p, logits in zip(capture_rec["prompts"], logits_by_prompt):
        logp = torch.log_softmax(logits.float(), dim=-1)
        a = n = 0
        for i, e in enumerate(p["positions"]):
            if e is None or i == 0 or not e["top"]:
                continue
            kit_top1 = e["top"][0][0]
            ours = logp[i - 1]
            n += 1
            if int(ours.argmax()) == kit_top1:
                a += 1
            if kit_top1 in ours.topk(5).indices.tolist():
                top5 += 1
            lp_err.append(abs(float(ours[kit_top1]) - e["top"][0][1]))
        agree += a
        total += n
        per_prompt.append({"tokens": len(p["ids"]), "positions": n, "top1": a / max(n, 1)})
    lp_err.sort()
    return {"top1_agreement": agree / max(total, 1), "kit_top1_in_our_top5": top5 / max(total, 1),
            "positions": total, "median_abs_logprob_err": lp_err[len(lp_err) // 2] if lp_err else None,
            "p99_abs_logprob_err": lp_err[int(0.99 * (len(lp_err) - 1))] if lp_err else None,
            "per_prompt": per_prompt}


def compare(args) -> int:
    import torch
    sys.path.insert(0, str(ROOT))
    from engine.reference.engram import NgramHasher
    from engine.reference.loader import CheckpointLoader, RemoteSafetensorsDir, SafetensorsDir, config_from, \
        hasher_from
    from engine.reference.model import Model
    from engine.reference.ops import Numerics

    rec = json.loads(Path(args.capture).read_text())
    if args.remote:
        cache = ROOT / ".cache" / "ckpt"
        src = RemoteSafetensorsDir(args.remote, args.model, cache / Path(args.model).name)
        eng = RemoteSafetensorsDir(args.remote, args.engram, cache / Path(args.engram).name)
    else:
        src, eng = SafetensorsDir(args.model), SafetensorsDir(args.engram)
    cfg = config_from(src)
    num = Numerics.exact() if args.exact else Numerics.kit()
    ld = CheckpointLoader(cfg, src, num, engram_src=eng, device=args.device)
    if args.token_map:
        hasher = NgramHasher(cfg, json.loads(Path(args.token_map).read_text()))
    else:
        hasher = hasher_from(src, cfg)
    model = Model(cfg, ld.model_weights(with_head=args.layers is None), num, hasher)
    seqs = [torch.tensor(p["ids"]) for p in rec["prompts"][: args.max_prompts]]
    t0 = time.time()

    def progress(layer: int, secs: float) -> None:
        print(f"layer {layer}: {secs:.1f}s (total {time.time() - t0:.0f}s)", flush=True)

    with torch.inference_mode():
        outs = model.forward_many(seqs, n_layers=args.layers, release=True, progress=progress)
    if args.layers is not None:
        print("partial run (--layers): no logits to score")
        return 0
    rep = score({"prompts": rec["prompts"][: args.max_prompts]}, [o.logits.cpu() for o in outs])
    rep["meta"] = {"capture": args.capture, "numerics": "exact" if args.exact else "kit", "device": args.device,
                   "seconds": round(time.time() - t0, 1), "threshold": args.threshold}
    print(json.dumps({k: v for k, v in rep.items() if k != "per_prompt"}, indent=1))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=1))
    return 0 if rep["top1_agreement"] >= args.threshold else 1


def token_map(args) -> int:
    sys.path.insert(0, str(ROOT))
    from engine.reference.config import Config
    from engine.reference.engram import build_compressed_token_map
    lookup, n = build_compressed_token_map(args.tokenizer)
    cfg = Config.from_file(args.config) if args.config else Config()
    if n != cfg.engram_compressed_vocab_size:
        raise SystemExit(f"compressed vocab {n} != {cfg.engram_compressed_vocab_size}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(lookup))
    print(f"wrote {args.out} ({len(lookup)} ids -> {n})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--base", default="http://127.0.0.1:8888")
    c.add_argument("--model", default="")
    c.add_argument("--tokenizer", default="", help="tokenizer.json (else the server's /tokenize)")
    c.add_argument("--prompts", default="")
    c.add_argument("--max-prompts", type=int, default=8)
    c.add_argument("--tokens", type=int, default=2048)
    c.add_argument("--bos", type=int, default=0)
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--out", default="results/reference/oracle-kit.json")
    gr = sub.add_parser("greedy")
    gr.add_argument("--base", default="http://127.0.0.1:8888")
    gr.add_argument("--model", default="")
    gr.add_argument("--capture", required=True, help="the oracle capture (its prompts)")
    gr.add_argument("--decode", type=int, default=256)
    gr.add_argument("--max-prompts", type=int, default=8)
    gr.add_argument("--out", default="results/reference/kit-greedy.json")
    m = sub.add_parser("compare")
    m.add_argument("--capture", required=True)
    m.add_argument("--model", default="~/models/dsv41-uncensored-2.9bpw")
    m.add_argument("--engram", default="~/models/dsv41-engram-src")
    m.add_argument("--remote", default="", help="ssh host holding --model / --engram (read-only)")
    m.add_argument("--device", default="cpu")
    m.add_argument("--token-map", default="")
    m.add_argument("--layers", type=int, default=None, help="debug: run only the first N blocks")
    m.add_argument("--max-prompts", type=int, default=8)
    m.add_argument("--exact", action="store_true", help="no kit quantization emulation")
    m.add_argument("--threshold", type=float, default=0.99)
    m.add_argument("--report", default="")
    t = sub.add_parser("token-map")
    t.add_argument("--tokenizer", required=True)
    t.add_argument("--config", default="")
    t.add_argument("--out", default="results/reference/engram-token-map.json")
    args = ap.parse_args()
    return {"capture": capture, "greedy": greedy, "compare": compare, "token-map": token_map}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
