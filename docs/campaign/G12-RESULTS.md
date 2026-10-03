# G12: the three open bugs (2026-10-03, 10:33-12:59)

One held campaign window (`campaign.sh open` at 10:33 with `DEADMAN_MIN=240`: lease + refresher, deadman at 14:33,
GLM's and DeepSeek's watchdog timers stopped, DeepSeek prod stopped). Prod had 2 requests in flight; the 3 min wait
ran out and prod was stopped anyway. Test servers ran on :8001 from prod's recipe (`scripts/serve.sh` +
`config/prod.env`). Caches were dropped before every start, every step ran under `timeout`, and the extensions were
prebuilt after every restage (10 / 10 both nodes, 6 times). Long steps ran in tmux on the head.
Raw files: `results/G12-20261003/`. Branch `dsv41-060`: staged at `f1bc947`, then at the commits below.

## Verdict

| bug | root cause | fix (dsv41-060) | evidence after |
| --- | --- | --- | --- |
| 3: fast-prefill segmentation dependence | the **RoPE table row** (`cs`). The forward builds its tables at `limit + pchunk` positions, so their length follows the segment size. On the Sparks' aarch64 CPU, one torch `cos` / `sin` call gives other bits at hundreds of positions for lengths 4,224 / 4,608 / 6,144 than for 8,192 (vector body vs scalar tail, ATen's thread chunks) | `f0010be`: `rope.Table.build` computes fixed blocks of 8,192 angles (one thread, the same call shape every block), so an entry depends only on its position | the 2 red tests (6 cases) pass; the new CPU test fails on the old build |
| 1: rank RssAnon grows 1.8 -> 6-7 GiB in a 299K prefill | **c10's CPU allocator is mimalloc** (v2.3.2, embedded in NVIDIA's torch 2.13 `libc10.so`). `prefetch.bulk_rows` makes a segment's fp32 Engram rows (~25 MB, two layers a segment) on a prefetch thread; the round thread frees them. mimalloc only MADV_FREEs a huge block freed by another thread and keeps its segment until the owning thread collects. Live CPU tensors stayed at ~83 MiB, tracemalloc at ~130 MiB and glibc's arenas at ~0.33 GiB, while mimalloc's 1 GiB arenas held 0 -> 1.9 GiB (1.2 GiB LazyFree) | `4b9081e`: the rows go into NumPy-owned memory (`engram_host.dequant_host`, the same ops, bit for bit). `a6f5792`: those buffers are reused (`HostBuffers`, returned by `weakref.finalize` when the last user is gone), because fresh 25 MB mmap / munmaps made the prefill 13-29% slower | rank RssAnon growth **0.67 GiB** (was 4-5.6); flat at ~2.05 GiB from 18K rows to the end |
| 2: the 19-minute "stall" | **not an NCCL hang**: transparent-huge-page faults under memory pressure. mimalloc madvises its arenas for THP (`allow_thp` 1) and NumPy madvises every buffer of 4 MiB or more. With THP `madvise` / defrag `madvise` and ~103 GiB held by the weights and the KV pool, each first touch was a synchronous **direct compaction** (page migration + SMMU TLB invalidations: GB10 shares the page tables with the GPU) | `10c3ba5`: NumPy's madvise off at engine start. **`MIMALLOC_ALLOW_THP=0`** in `config/prod.env`, passed by `scripts/serve.sh` (it now forwards `MIMALLOC_*`). mimalloc reads it at process start, so it cannot live in the engine | 0 THP-flagged mappings on both ranks; 4 stress runs with it had no slow prefill and no stall, one with nvidia-smi polling every second |

**Prod: pinned to the fixed commit `a6f5792`** (`config/prod.env` TF_COMMIT + `MIMALLOC_ALLOW_THP=0`, on the workstation
and the head, `serve.sh stage` done). The pin criteria were met: memory stays bounded, there were no stalls, the
long prefill was as fast as G10's or faster, and every test passed. Close and verification: see the last section.

## Bug 3: segmentation (`segdiag`, run on the worker's GPU beside the head's test queue)

- **Tables** (`segdiag-tables.txt`, container CPU: torch 2.13, 20 threads). Against the 8,192-position table, these
  positions differ: length 4,224: 307 / 480; 4,608: 299 / 517; 6,144: 324 / 701 (table 0 / table 1). Lengths 4,352,
  5,120 and 302,112 differ in none.
- **First differing step** (`segdiag-cmp*.txt`, layer 3, rows 1259 / 1277 / 1329 / 1423 / 2186): the same in fresh
  processes and in one process. Fresh vs same-process at 2,048 is identical. The first step that differs is always
  **`cs`**, the table row, whenever the two tables' lengths differ (5,120 vs 4,224 / 4,608 / 6,144). After it come
  `q_rope` and the attention output. `x`, wq_a, the norms and wq_b are equal. Sizes whose tables agree (5,120 vs
  4,352) give equal logits. Rows 1277 / 1329 at 512 differ already at `x`: the earlier layers' tables.
- **Fix `f0010be`** + `tests/test_dsv41_rope_tables.py` (lengths 1-9,001 agree with 8,192; 1 vs 20 threads agree;
  values within fp32 of the angles; a single-block table is the old call). On the worker the test fails on the old build
  and passes on the new. GPU: `test_fast_prefill_row_independent_and_close` + `test_fast_prefill_with_g7_knobs`:
  **3 failed** at f1bc947, **6 passed** at f0010be and at a6f5792.
- **Numerics.** At prod's length (302,048) the new table differs from the old one in 0.08% (plain) / 0.15% (YaRN) of
  entries, by at most 1 fp32 ulp. The code digest changes too, so session entries written by the old commit on NVMe
  are not resumed.

## Bugs 1 and 2: stress runs

All runs: one 299K prefill + three 64K prompts x 2,048 tokens, prod's knobs, `TF_DSV41_MEMPROBE=16`,
`TF_DSV41_STALL_S=120`, `_EXIT_S=900`, 1 s samplers. "growth" is RssAnon max minus first.

| run (commit, env) | 299K TTFT s | r0 RssAnon growth GiB | r1 RssAnon growth GiB | MemAvailable min head / worker GiB | stall lines | decode 2,048 x3 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| G10 (38f6500 / 5ace28b, 8 runs) | 175-217 | | ~4-6 | | the 19-min hang once (7d940f0 + nvidia-smi) | yes |
| stress-fix #1 (f0010be) | **~150 tok/s; aborted at 11:00** | | 1.81 -> 4.12 at 84K rows | | none (rounds ~12 s, never 120 s) | |
| stress-notrim (f0010be, `HOST_TRIM_ROWS=0`) | 180.8 | 5.24 | 4.08 | 6.00 / 3.26 | 0 | yes |
| stress-fix #2 (10c3ba5: NumPy THP off) | **527** (slow stretches) | 2.72 | 4.98 | 4.75 / **3.00** | 0 | yes |
| stress-mi (10c3ba5 + `MIMALLOC_ALLOW_THP=0`) | 182.5 | 5.63 | 4.72 | 3.96 / 3.34 | 0 | yes |
| stress-mipurge (+ `PURGE_DELAY=0`, `ABANDONED_RECLAIM_ON_FREE=1`) | 185.2 | 5.27 | 4.95 | 5.11 / 3.69 | 0 | yes |
| stress-bulk (4b9081e + THP=0) | 205.5 | 1.00 | 0.93 | 3.78 / 3.15 | 0 | yes |
| stress-bulk-smi (same + nvidia-smi 1 s, both nodes) | 234.3 | 1.11 | 0.97 | 4.89 / 3.66 | 0 | yes |
| **stress-pool (a6f5792 + THP=0)** | **174.8** | **0.69** | **0.67** | 4.31 / 3.60 | **0** | **yes** |

- **The slow prefill is bug 2's signature.** In stress-fix #1, both GPUs sat at 96% utilization and 17 W (G10's
  hang: 92-96% at 17-19 W). Load was 37, CPU 45-51% system time, PSI memory "full" 64% on the head and 71% on the
  worker, compact_stall +33/s. Every busy rank-0 thread was in `__do_huge_pmd_anonymous_page ->
  __alloc_pages_direct_compact -> compact_zone -> migrate_pages -> try_to_migrate_one -> arm_smmu_tlb_inv_range_asid`
  (`stress-fix-r1-memprobe.txt`, `stress-fix-*-thp.txt`). Rounds took ~12 s, so the 120 s stall watchdog never
  fired. G10's 19 minutes for 63 tokens is this at its worst, not a wedged collective.
- **Why NumPy's switch was not enough.** With NumPy's madvise off (10c3ba5), 7-8 mappings were still flagged `hg`:
  1 GiB MAP_NORESERVE regions with 32 MiB segment headers and fp32 Engram values inside. These are mimalloc's
  arenas (libc10 embeds mimalloc; `MIMALLOC_VERBOSE=1` lists `allow_thp: 1`). The worker kept PSI "full" at 23-70%
  and direct compaction in PMD faults, and the 299K prefill alternated between 1,600 rows/s and ~300 rows/s.
  `MIMALLOC_ALLOW_THP=0` left 0 `hg` mappings on both ranks, and the prefill was back at 182 s.
- **Where the growth was** (`trace2-*`, 7d40776's probe fields, 128K prompt): rank-1 RssAnon 1.80 -> 4.58 GiB.
  Live CPU tensors were 33-83 MiB, tracemalloc 98-136 MiB, glibc arenas 0.07 -> 0.34, [heap] 0.92 -> 1.00, other
  0.81 -> 0.90, and **mimalloc arenas 0 -> 1.92 GiB** (released only now and then: 1.92 -> 0.20). LazyFree
  (MADV_FREE) was 1.22 GiB of 3.7 on each rank. mimalloc's purge options changed nothing (stress-mipurge), which
  pointed at the cross-thread frees of huge blocks. f1bc947's malloc_trim cannot reach mimalloc, which is why it
  did nothing.
- **Trims (the A/B).** At f0010be, `TF_DSV41_HOST_TRIM_ROWS=0` made no difference to the growth (r1 4.08 vs
  4.1-5.0 with trims). The trims stay as they are (they only reach glibc; harmless).
- **Stall (bug 2) reproduction.** With nvidia-smi polling every second on both nodes (stress-bulk-smi), it did not
  reproduce: no `[dsv41 stall]` line in any run, and every decode stream reached 2,048. No NCCL / RoCE dump to
  report. f1bc947's deadlines and watchdog stay as a safety net (prod uses the default 300 s report, no exit).
- **MemAvailable floor (not fixed; not bug 1).** The worker's minimum stays 3.0-3.7 GiB in every run today. The
  worker is already at 4.5-7.8 GiB when the server is ready, and the minimum comes in the first 20K rows (cuda
  reserved +1.3 GiB as the first segments run). It is the boot budget (4 x 300K pool) plus the device-side part G10
  saw, not host growth. With the fixes, the floor stops eroding during the prefill: worker 3.7-5.0 GiB from 80K rows
  on in stress-pool, against 3.3-4.9 before. The 3 GiB guard never fired. G4's 5 GiB line still does not hold.

## Tests

| suite | f1bc947 | a6f5792 |
| --- | --- | --- |
| new + touched (stall, long-prefill memory, Engram native, serving batch, M2 decode, perf paths, Engram gate GPU) | 64 passed | **68 passed** |
| fast tag red tests (bug 3) | 3 failed, 3 passed | **6 passed** |
| CPU: forward, prefill, prefill fast, slots, replay, RoPE tables / fused / fp8 keys, prefill2, long-prefill memory (worker) | | **77 passed** (after a20be14's order fix) |
| CPU: long-prefill memory, Engram native / gate / interp / pcache | | 37 passed |

## Commits (dsv41-060)

| commit | what |
| --- | --- |
| `f0010be` | fix(rope): bug 3, tables built in fixed blocks + `tests/test_dsv41_rope_tables.py` |
| `10c3ba5` | fix(hostmem): NumPy's MADV_HUGEPAGE off at engine start (`TF_DSV41_NUMPY_HUGEPAGE=1` restores it) + test |
| `7d40776` | diag(hostmem): `TF_DSV41_MEMPROBE_TENSORS=K` (live CPU tensors) and `_SMAPS=1` (anonymous RSS by mapping) + test |
| `4b9081e` | fix(prefetch): bug 1, bulk Engram rows in NumPy-owned memory (`engram_host.dequant_host`, bit for bit) + test |
| `a6f5792` | perf(engram_host): reused row buffers (`HostBuffers`, `TF_DSV41_HOST_BUFFERS`, 6) + test. **Prod pin** |
| `a20be14` | test: the two G12 tests made independent of the suite's order (sources = a6f5792) |

Development repo: `scripts/serve.sh` forwards `MIMALLOC_*`. `config/prod.env` gets TF_COMMIT `a6f5792` and
`MIMALLOC_ALLOW_THP=0`. `G12-bugs.sh` gets `step stress1 TAG "ENV" [smi]`. `results/G12-20261003/` holds
`segdiag-worker.sh`, `red-worker.sh` and `psi.sh` (worker-side runners and the PSI / compaction logger).

## Run notes

- **Abort of the first stress step.** Killing its tmux session at 11:00 did not stop the script. Its stress-notrim
  (f0010be) ran to the end and is in the table. Its stress-smi died 5 s in, when 10c3ba5 was restaged under it
  (invalid; not in the table). Container logs of stress-fix #1 were lost; its memprobe lines were copied by hand
  into `stress-fix-r1-memprobe.txt`. The second step was stopped cleanly after its stress-fix #2.
- **Lost lines.** `tf_dsv41_l2pf_v1` still builds at the first server start after a restage (not in prebuild_ext.py).

## Open

- The worker's MemAvailable floor (3.0-3.7 GiB in this window's runs) is boot budget plus the device side, not
  growth. Next: budget the KV pool to the 5 GiB target, or cut the first segments' +1.3 GiB of cuda reserved.
- mimalloc's other cross-thread frees (smaller blocks) are bounded now (RssAnon flat), but every new long-lived
  CPU allocation made on one thread and freed on another should use NumPy memory or a pool.

## Close and verification (prod = **a6f5792**, pinned)

- **12:57.** `campaign.sh close` -> `prod-switch.sh restore` (marker dsv41). It started DeepSeek prod from
  `config/prod.env` (TF_COMMIT a6f5792, `MIMALLOC_ALLOW_THP=0`) and verified it at 12:59:04: models, 17*23 = 391,
  canary ok, https ok. It enabled the dsv41 automation, wrote the marker and released the lease. rc=0; prod was
  down 146 min.
- **Checked after.** Both ranks mount the staged `a6f5792` sources, carry `MIMALLOC_ALLOW_THP=0`, and have 0
  THP-flagged mappings.
  - :8000 and https list DeepSeek-V4.1-Flash-TF, deepseek-v4.1-flash and GLM-5.3-Flash-EXL3.
  - 17*23 = 391, also through the GLM alias. A tool call returns `get_weather {"city":"Hanoi"}`.
  - 4 concurrent streams finish (TTFT 0.28-0.79 s, 5.2-7.0 s each).
- **Automation.** Only dsv41's is enabled: dsv41-tf-watchdog.timer, dsv41-watchdog-rearm.timer and
  dsv41-boot-start. glm53-tf-watchdog, glm53-watchdog-rearm and glm53-boot-start are disabled.
- **Leftovers: none.** No lease, deadman, refresher, sampler, PSI logger or tmux session. Only dsv41-tf-r0 /
  -r1 run (other containers on the nodes are old exited ones, not ours, untouched). The worker's scratch copies
  (`g12-tf-x`, `g12-segdiag`) were removed.
- **Rollback.** `config/prod.env` TF_COMMIT back to `38f65008e323b8b1c19c44fff50897f6f90a7611` (still staged next
  to it), then `scripts/serve.sh stop && scripts/serve.sh start` on the head.
  `MIMALLOC_ALLOW_THP=0` is harmless for 38f6500 too.
