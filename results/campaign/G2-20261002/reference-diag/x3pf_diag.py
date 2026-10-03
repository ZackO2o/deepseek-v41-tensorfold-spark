import sys
from tensorfold.families.deepseek_v41.cuda import kbench
for rows in [(1024,), (768,), (2048,)]:
    try:
        kbench.bench_prefill("/model", 3, 0, 3, rows=rows)
    except Exception as e:
        print("rows", rows, "->", type(e).__name__, str(e)[:200], flush=True)
        break
