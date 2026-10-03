import sys, os
sys.path[:0] = ["/dsv41-tf/tests", "/dsv41-tf/tests/cuda"]
os.environ["TF_DSV41_SEAMS"] = "torch"
import numpy as np, torch
import dsv41_dspark_fakes as DF
from tensorfold.families.deepseek_v41.cuda.forward import Seg
from tensorfold.families.deepseek_v41.cuda.protocol import Piece
import tempfile, pathlib
ck = DF.ckpt(pathlib.Path(tempfile.mkdtemp()))
LONG = 4096
P = [int(t) for t in np.random.default_rng(20).integers(3, 250, size=2100)]
def make(dspark):
    e = DF.forward(ck, slots=1, limit=LONG, dspark=dspark, device="cuda", chunk=128)
    be = DF.batcher(ck, e, n=1, capacity=LONG, pool_tokens=2 * LONG)
    be.ex.slots[0].ensure(LONG)
    return e, be
for n0 in (100, 300, 2030):
    for dspark in (False,):
        a, ba = make(dspark); b, bb = make(dspark)
        for f in (a, b):
            f.reset(0); f.prefill([Piece(0, 0, tuple(P[:n0]))], mode="full")
        ids = tuple(P[n0:n0 + 8])
        la = a.run([Seg(0, a.slots[0].pos, ids)]).clone()
        rows = []
        for i, t in enumerate(ids):
            rows.append(b.run([Seg(0, b.slots[0].pos, (t,))]).clone()); b.keep(0, 0)
        lb = torch.cat(rows)
        d = (la - lb).abs().amax(-1)
        print(f"n0={n0} dspark={dspark}: window of 8 == 8 single rows: {torch.equal(la, lb)}; per-row max|d| {[f'{x:.2e}' for x in d.tolist()]}", flush=True)
