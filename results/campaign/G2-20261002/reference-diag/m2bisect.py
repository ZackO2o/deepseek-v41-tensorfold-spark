import sys, os
sys.path[:0] = ["/dsv41-tf/tests", "/dsv41-tf/tests/cuda"]
os.environ["TF_DSV41_SEAMS"] = "torch"
import numpy as np, torch, tempfile, pathlib
import dsv41_dspark_fakes as DF
from tensorfold.families.deepseek_v41.cuda.batch import Job
ck = DF.ckpt(pathlib.Path(tempfile.mkdtemp()))
LONG = 4096
for plen in (300, 2060):
    P = [int(t) for t in np.random.default_rng(20).integers(3, 250, size=plen)]
    def go(dspark, lookup, depth=None):
        fw = DF.forward(ck, slots=1, limit=LONG, dspark=dspark, device="cuda", chunk=128)
        b = DF.batcher(ck, fw, n=1, capacity=LONG, pool_tokens=2 * LONG, depth=depth, lookup=lookup)
        j = Job(P, 48, None)
        out = DF.run(b, [j])[0]
        return out, j.stats.get("drafting")
    ser, _ = go(False, False, depth=0)
    for name, args in (("dspark, no lookup", (True, False)), ("no dspark, lookup", (False, True)), ("dspark+lookup", (True, True))):
        out, st = go(*args)
        div = next((i for i, (x, y) in enumerate(zip(out, ser)) if x != y), None)
        print(f"plen={plen} {name}: == serial {out == ser}, first divergence {div}; {st}", flush=True)
