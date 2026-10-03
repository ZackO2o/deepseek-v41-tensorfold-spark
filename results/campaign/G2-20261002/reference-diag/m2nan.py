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
PROMPTS = [[int(t) for t in np.random.default_rng(20 + i).integers(3, 250, size=2060 + 31 * i)] for i in range(4)]
for dspark in (False, True):
    for nslots in (1, 4):
        e = DF.forward(ck, slots=nslots, limit=LONG, dspark=dspark, device="cuda", chunk=128)
        be = DF.batcher(ck, e, n=nslots, capacity=LONG, pool_tokens=5 * LONG)
        for s in range(nslots):
            be.ex.slots[s].ensure(LONG)
            e.reset(s)
            e.prefill([Piece(s, 0, tuple(PROMPTS[s][:2030 + 3 * s]))], mode="full")
        rng = np.random.default_rng(5)
        segs = [Seg(0, e.slots[0].pos, tuple(int(t) for t in rng.integers(3, 250, size=8)))]
        le = e.run(segs)
        print(f"dspark={dspark} slots={nslots}: nan {bool(torch.isnan(le).any())}", flush=True)
        if torch.isnan(le).any() and e.trace is None:
            e.trace = {}
            e.keep(0, 0) if False else None
