"""Reference (engine/reference, kit numerics) on the first N tokens of oracle prompts: per-layer streams + top-1."""
import json, os, sys, time
import torch
sys.path.insert(0, "/repo")
from engine.reference.loader import CheckpointLoader, SafetensorsDir, config_from, hasher_from
from engine.reference.model import Model
from engine.reference.ops import Numerics

N = int(sys.argv[1]); prompts = [int(x) for x in sys.argv[2].split(",")]
rec = json.load(open("/repo/results/BASELINE-20261001/oracle-kit.json"))
src, eng = SafetensorsDir("/model"), SafetensorsDir("/engram-src")
cfg = config_from(src)
ABL = os.environ.get("ABL", "")
num = Numerics.exact() if ABL == "exact" else Numerics.kit()
ld = CheckpointLoader(cfg, src, num, engram_src=eng, device="cuda")
model = Model(cfg, ld.model_weights(with_head=True), num, None if ABL == "noengram" else hasher_from(src, cfg))
print("ablation", ABL or "none", flush=True)
seqs = [torch.tensor(rec["prompts"][p]["ids"][:N]) for p in prompts]
t0 = time.time()
with torch.inference_mode():
    outs = model.forward_many(seqs, release=True, keep_layer_out=tuple(range(cfg.num_hidden_layers)),
                              progress=lambda L, s: print(f"layer {L} {s:.1f}s", flush=True))
res = {}
for p, o, ids in zip(prompts, outs, seqs):
    top1 = o.logits.argmax(-1).tolist()
    kit = [e["top"][0][0] if e and e.get("top") else -1 for e in rec["prompts"][p]["positions"][:N]]
    agree = [int(top1[i] == kit[i + 1]) for i in range(N - 1)]
    print(f"prompt {p}: ref vs kit top-1 {sum(agree)}/{len(agree)}; misses at {[i+1 for i,a in enumerate(agree) if not a][:40]}")
    res[p] = {"top1": top1, "agree": agree}
    torch.save({L: t.cpu() for L, t in o.layer_out.items()} | {"logits": o.logits.cpu()}, f"/out/ref-p{p}-n{N}{ABL}.pt")
json.dump(res, open(f"/out/ref-n{N}{ABL}.json", "w"))
print("done", time.time() - t0)
