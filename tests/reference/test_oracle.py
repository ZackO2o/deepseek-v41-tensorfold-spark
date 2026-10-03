"""The oracle script end to end against a fake vLLM server whose prompt_logprobs come from the tiny reference model:
capture parses vLLM's format, and scoring the same model's logits gives 100% top-1."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import torch

from engine.reference import synthetic as S
from engine.reference.engram import NgramHasher
from engine.reference.model import Model
from engine.reference.ops import Numerics

spec = importlib.util.spec_from_file_location("oracle", Path(__file__).parent / "oracle_prompt_logprobs.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_capture_and_score(cfg, tmp_path):
    model = Model(cfg, S.model_weights(cfg), Numerics.exact(),
                  NgramHasher(cfg, [i % cfg.engram_compressed_vocab_size for i in range(cfg.vocab_size)]))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = body["prompt"]
            lp = torch.log_softmax(model.forward(torch.tensor(ids)).logits, -1)
            k = body["prompt_logprobs"]
            out = [None]
            for i in range(1, len(ids)):
                row = lp[i - 1]
                order = row.argsort(descending=True)
                d = {str(int(t)): {"logprob": float(row[t]), "rank": r + 1, "decoded_token": "x"}
                     for r, t in enumerate(order[:k])}
                rank = int((row > row[ids[i]]).sum()) + 1
                d[str(ids[i])] = {"logprob": float(row[ids[i]]), "rank": rank, "decoded_token": "x"}
                out.append(d)
            resp = {"choices": [{"text": "", "prompt_logprobs": out}]}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        prompts = tmp_path / "p.jsonl"
        g = torch.Generator().manual_seed(2)
        prompts.write_text("\n".join(json.dumps({"ids": torch.randint(0, cfg.vocab_size, (n,), generator=g).tolist()})
                                     for n in (9, 17)))
        out = tmp_path / "cap.json"
        args = SimpleNamespace(base=f"http://127.0.0.1:{srv.server_port}", model="m", tokenizer="", prompts=str(prompts),
                               max_prompts=8, tokens=64, bos=0, k=3, out=str(out))
        assert oracle.capture(args) == 0
    finally:
        srv.shutdown()
    rec = json.loads(out.read_text())
    assert len(rec["prompts"]) == 2 and rec["prompts"][0]["positions"][0] is None
    assert all(len(p["top"]) == 3 for p in rec["prompts"][1]["positions"][1:])
    logits = [model.forward(torch.tensor(p["ids"])).logits for p in rec["prompts"]]
    rep = oracle.score(rec, logits)
    assert rep["top1_agreement"] == 1.0 and rep["positions"] == 8 + 16
    assert rep["median_abs_logprob_err"] < 1e-5
    other = Model(cfg, S.model_weights(cfg, seed=5), Numerics.exact())
    worse = oracle.score(rec, [other.forward(torch.tensor(p["ids"])).logits for p in rec["prompts"]])
    assert worse["top1_agreement"] < 0.9
