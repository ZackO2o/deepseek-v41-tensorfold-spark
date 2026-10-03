#!/usr/bin/env python3
"""Build the CUDA extensions a DeepSeek-V4.1 rank loads, before any weights are (G6: a boot-time build beside 100 GB
of weights took the worker to 1.26 GiB MemAvailable, the guard killed it, and the stale lock hung the next build)."""
import importlib, time
mods = ["tensorfold.cuda.exl3.experts", "tensorfold.cuda.exl3.linear", "tensorfold.families.glm5_next.spark.roce",
        "tensorfold.families.deepseek_v41.cuda.x3gm", "tensorfold.families.deepseek_v41.cuda.expert_loads",
        "tensorfold.families.deepseek_v41.cuda.expert_prefill", "tensorfold.families.deepseek_v41.cuda.fused_proj",
        "tensorfold.families.deepseek_v41.cuda.router_gemv", "tensorfold.families.deepseek_v41.cuda.engram_gate",
        "tensorfold.families.deepseek_v41.cuda.dense"]
for m in mods:
    t = time.time()
    try:
        importlib.import_module(m)._ext(); print("built", m, round(time.time() - t, 1), flush=True)
    except Exception as e:
        print("FAIL", m, repr(e)[:200], flush=True)
