#!/usr/bin/env python3
"""Build the CUDA extensions a DeepSeek-V4.1 rank loads, before any weights are (G6: a boot-time build beside 100 GB
of weights took the worker to 1.26 GiB MemAvailable, the guard killed it, and the stale lock hung the next build).

Run inside the image with the cache volume mounted (`scripts/serve.sh prebuild` does, on both nodes; `scripts/serve.sh
build` runs it after shipping the image). Each module's `_ext()` compiles into TORCH_EXTENSIONS_DIR, or returns the
cached build. One line a module ("built" / "FAIL"); the exit status is the number of failures (0 = every one built).

The G13 rewrites (mhc_cuda, csa2.attn_cuda with the indexer top-k, moe_fused, dense3) are built whether or not their
lever is on, so turning one on later does not compile beside the weights.
"""
import importlib
import sys
import time

MODS = [
    "tensorfold.cuda.exl3.experts", "tensorfold.cuda.exl3.linear", "tensorfold.families.glm5_next.spark.roce",
    "tensorfold.families.deepseek_v41.cuda.x3gm", "tensorfold.families.deepseek_v41.cuda.expert_loads",
    "tensorfold.families.deepseek_v41.cuda.expert_prefill", "tensorfold.families.deepseek_v41.cuda.fused_proj",
    "tensorfold.families.deepseek_v41.cuda.router_gemv", "tensorfold.families.deepseek_v41.cuda.engram_gate",
    "tensorfold.families.deepseek_v41.cuda.dense", "tensorfold.families.deepseek_v41.cuda.l2pf",
    # G13 (engine 767ad9f): TF_DSV41_MHC_CUDA, TF_DSV41_ATTN_CUDA, TF_DSV41_MOE_FUSED, TF_DSV41_DENSE_V3
    "tensorfold.families.deepseek_v41.cuda.mhc_cuda", "tensorfold.families.deepseek_v41.cuda.csa2.attn_cuda",
    "tensorfold.families.deepseek_v41.cuda.moe_fused", "tensorfold.families.deepseek_v41.cuda.dense3",
]

failed = 0
for m in MODS:
    t = time.time()
    try:
        importlib.import_module(m)._ext()
        print("built", m, round(time.time() - t, 1), flush=True)
    except Exception as e:  # noqa: BLE001 - report every module, then fail
        failed += 1
        print("FAIL", m, repr(e)[:200], flush=True)
print(f"prebuild: {len(MODS) - failed} / {len(MODS)} built", flush=True)
sys.exit(min(failed, 100))
