import sys, os
sys.path[:0] = ["/dsv41-tf/tests", "/dsv41-tf/tests/cuda"]
os.environ["TF_DSV41_SEAMS"] = "torch"
import numpy as np, torch, tempfile, pathlib
import dsv41_dspark_fakes as DF
from tensorfold.families.deepseek_v41.cuda.batch import Job
from tensorfold.engine.exact_sampling import Sampling
ck = DF.ckpt(pathlib.Path(tempfile.mkdtemp()))
LONG = 4096
SAMPLINGS = [None, Sampling(seed=7, temperature=0.8, top_k=20, top_p=0.95), None,
             Sampling(seed=11, temperature=1.1, top_k=0, top_p=0.9, min_p=0.02)]
PROMPTS = [[int(t) for t in np.random.default_rng(20 + i).integers(3, 250, size=2060 + 31 * i)] for i in range(4)]
N = 48
def serial(jobs_idx):
    b = DF.batcher(ck, DF.forward(ck, slots=1, limit=LONG, dspark=False, device="cuda", chunk=128), capacity=LONG, pool_tokens=2 * LONG, depth=0)
    return DF.run(b, [Job(PROMPTS[i], N, SAMPLINGS[i]) for i in jobs_idx])
os.environ.pop("TF_DSV41_SEAMS"); sk = serial([0]); os.environ["TF_DSV41_SEAMS"] = "torch"; print("serial with seam kernels job0[:16]", sk[0][:16], flush=True)
s4 = serial([0, 1, 2, 3]); s0 = serial([0]); s1 = serial([1])
print("serial 4 jobs job0[:16]", s4[0][:16]); print("serial job0 alone    ", s0[0][:16])
print("job0 same:", s4[0] == s0[0], " job1 same:", s4[1] == s1[0])
fw = DF.forward(ck, slots=1, limit=LONG, dspark=True, device="cuda", chunk=128)
b = DF.batcher(ck, fw, n=1, capacity=LONG, pool_tokens=2 * LONG, lookup=True)
d4 = DF.run(b, [Job(p, N, s) for p, s in zip(PROMPTS, SAMPLINGS)])
print("drafted 4 jobs job0[:16]", d4[0][:16])
print("drafted == serial per job:", [a == c for a, c in zip(d4, s4)])
