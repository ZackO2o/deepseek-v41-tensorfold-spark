import sys, os
sys.path[:0] = ["/dsv41-tf/tests", "/dsv41-tf/tests/cuda"]
os.environ["TF_DSV41_SEAMS"] = "torch"
import numpy as np, torch, tempfile, pathlib
import dsv41_dspark_fakes as DF
from tensorfold.families.deepseek_v41.cuda import graphs as G
from tensorfold.families.deepseek_v41.cuda.forward import Seg
from tensorfold.families.deepseek_v41.cuda.protocol import Piece
ck = DF.ckpt(pathlib.Path(tempfile.mkdtemp()))
LONG = 4096
PROMPTS = [[int(t) for t in np.random.default_rng(20 + i).integers(3, 250, size=2060 + 31 * i)] for i in range(4)]
SETTINGS = {"on": True, "budget_s": 120.0, "max_graphs": 64, "bucket": 2048, "slots": 4}
MODE = sys.argv[1]
g, e = DF.forward(ck, slots=4, limit=LONG, dspark=True, device="cuda", chunk=128), DF.forward(ck, slots=4, limit=LONG, dspark=True, device="cuda", chunk=128)
bg = DF.batcher(ck, g, n=4, capacity=LONG, pool_tokens=5 * LONG)
be = DF.batcher(ck, e, n=4, capacity=LONG, pool_tokens=5 * LONG)
if MODE != "nograph":
    g.graphs = G.CudaGraphs(g, settings=SETTINGS)
    n = g.graphs.warmup(4, lambda s, end: bg.ex.slots[s].ensure(end))
    print("warmup graphs", n, flush=True)
order = ((g, bg), (e, be)) if MODE != "eagerfirst" else ((e, be), (g, bg))
for f, b in order:
    for s in range(4):
        b.ex.slots[s].ensure(LONG)
        f.reset(s); f.prefill([Piece(s, 0, tuple(PROMPTS[s][:2030 + 3 * s]))], mode="full")
rng = np.random.default_rng(5)
segs = [Seg(0, g.slots[0].pos, tuple(int(t) for t in rng.integers(3, 250, size=8)))]
e.trace = {}
le = e.run(segs)
lg = (g.graphs.run(segs) if g.graphs else g.run(segs)).clone()
print(MODE, "eager nan", bool(torch.isnan(le).any()), "graph nan", bool(torch.isnan(lg).any()), "equal", torch.equal(lg, le))
bad = [L for L, t in sorted(e.trace.items()) if torch.isnan(t).any()]
print("first eager layer with NaN streams:", bad[:3])
# also: prefill result of e itself (logits of last prefill?) check slot states
