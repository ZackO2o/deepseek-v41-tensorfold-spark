# G7: every G7 lever in one window, ship gates, DeepSeek to production (2026-10-02, 13:33-)

One held campaign window, opened right after a storm power outage rebooted both Sparks (~13:30) and killed G6
mid-run. User direction: every latest optimization in a single run; GLM downtime does not matter; DeepSeek becomes
production at the end.

- **Start state.** A stale G6 lease (13:14) and G6 pid files, all dead after the reboot. GLM's watchdog and re-arm
  timers were armed, and GLM's boot-start had run and stood down because the lease was fresh. Nothing served :8000,
  and `the prod-stack marker` was unset.
- **Cleanup.** GLM's `glm53-tf-watchdog.timer` and `glm53-watchdog-rearm.timer` were stopped first. The stale pid
  files were moved to `g6-stale/`, the exited `dsv41-tf-g6-*` containers removed and the stale lease deleted.
- **Window.** `campaign.sh open` at 13:33:15 (lease + refresher, deadman at +480 min, 21:33). GLM never came up
  (`serve.sh stop` found nothing). The deadman was extended at 16:02 to 23:33 (`window.log`).
- **Branch.** `dsv41-060`, staged at `defc9ba`, then restaged for my own fixes: `24392ea` (a stale router test) and
  `e87e813` (prefill attention). Every bench and every ship gate ran at `e87e813`.
- **Harness.** Test servers on :8001, caches dropped before every start (MemFree >= 104 GiB), every step under
  `timeout`, 0.5 s samplers on both nodes, the 3 GiB guard.
- **Raw files.** `results/G7-20261002/` (`ship/`: the ship gates). The driver is the new `scripts/windows/G7.sh`
  (`tests`, `ab`, `decode`, `dgate`, `skew`, `pf`), and `scripts/windows/nsys_skew.py` is new.

**Bottom line.**

- **Decode.** The gemv router is the one new decode lever that pays: code 69.1, prose 37.4, structured 102.2 tok/s,
  C1 70.0, C4 67.0 (1.15-2.7x the kit; 1.2x G4 single stream).
- **Prefill.** Levers 2 (all4) bring cold prefill to 1,833-2,068 tok/s at 8K-128K (1.7-1.95x the kit, 1.4-1.5x G6).
- **Rank skew.** G4's skew is gone; what is left is per-segment jitter.
- **Ship gates.** Replay quality passed (MMLU 88.5 / 88.0 / kit 87.5, every needle). The rest of the ship gates and
  the prod switch were **stopped at 16:53 by direction** (performance first: G8 next, prod after G8).
- **Pinned for prod.** `e87e813` + the G7 knobs, in `config/prod.env` (dsv41 `1366c66`; image pinned, prod dirs
  staged, static preflight ok).

## 1. GPU tests (prebuild, then two queues at once: head and worker)

`G6.sh prebuild` built all 8 extensions on both nodes. The worker's cache volume had no pytest in `/cache/pylib`
(its first queue died in 1 s), so it was copied from the head's volume.

| queue | first run | after fixes |
| --- | --- | --- |
| router (gemv GPU + fused / split GPU + emulator + sm_121 build) | 67 passed | |
| fused_proj (random + real weights, both plans) | 37 passed, 3 skipped (offline nvcc) | |
| decode glue + forward GPU + CPU (xchg, forward, slots) | 45 passed, **2 failed** | `test_router_split_equals_one_program[target/dspark]`: **test bug**. Since `8f24eac` the default mode is `fused` (another split-K order by design), so the test compared fused with the one-program router (picks equal, weights 1-8 ulp apart). Fixed to name `mode="split"` (`24392ea`); passes |
| draft + M2 GPU (graph replay == eager, drafted == serial; every head, 1 / 4 slots) + CPU | 57 passed | |
| interpreter suites (router, draft head, glue, mHC, prefill2) | 63 passed | |
| prefill2 GPU + CPU | 38 passed, **4 failed** | (a) `test_fused_attention_close_row_invariant_and_time[32-8 / 32-16]`: **real bug**. `TF_DSV41_PREFILL_ATTN_BMQ=32` at 2 stages needs 114,816 B of shared memory, and sm_121 has 101,376 (`OutOfResources`). Fixed: 1 stage at BMQ 32 (`e87e813`); passes, 3.73 ms vs chunks 10.6 ms at 2,048 rows. (b) `test_fast_prefill_with_g7_knobs[knobs2]` raised on the synthetic checkpoint (its head count is not a multiple of 32). Fixed: such a head count keeps the chunk kernels (`e87e813`). It then fails like (c). (c) `[knobs0]` (+ knobs2): see x3gm |
| x3gm GPU + fast tag e2e + emulator | 46 passed, **1 failed** | `test_fast_prefill_row_independent_and_close[gm]`: **real, pre-existing, not fixed** (below) |

**The fast tag's segmentation dependence (open).** The test prefills a 2,301-token synthetic prompt with 1,024-row
segments and checks that 96-row segments (and 256 with a cut at 700) give the same bits. Grouped and tc pass; gm
fails. Diagnosis (`segdiag-*.txt`):

- **It is not x3gm and not new.** x3gm alone is row-independent at every shape tried (random subsets, 1-1,024 rows,
  5 shapes x 2 widths). Grouped experts fail too at 512-row segments, already at `5f74397` (G4/G5) and at every
  commit since. The test only samples 96 and 256-cut-700, where grouped happens to pass.
- **Size.** 1-3 of 2,300 rows differ, by 1 bf16 ulp (max 0.0156 on O(2) values). The first diverging layer is 2 or 3.
  It is deterministic: 1,024 vs 1,024 is equal three times.
- **The fused attention (adopted below) removes the layer-2 part.** Its rows 1259 / 1277 / 1329 match.
- **The rest is in layer 3's q path.** Layer 3's attention input is bit-identical and its key lists, compressed rows,
  scales and SWA rows are equal. Yet q itself (row 2186) differs between the 1,024- and the 2,048-row runs, although
  that row's own segment ([2048, 2300)) is the same in both. Something on the q path reads state left by earlier
  segments.
- **Not fixed in this window.** The decode structural properties (drafted == serial, batched == alone, graph replay
  == eager, row invariance of every decode lever) all pass. What this touches is the fast tag's "a prompt segmented
  differently gives the same bits", which session resume relies on, in ~0.1% of rows at 1 ulp.

## 2. Decode levers A/B

Base: G6's defaults with the new bit-exact levers on. That is the branch defaults at `e87e813`: fused router for
decode, fused projections at upstream's plan, bf16 exchanges + mHC split / unroll + the fp64 norm launch, draft
graphs, head full and draft skip. Every run is m2bench, 1 stream, code + prose, 2 reps, RoCE, T = 0 (T = 0.7 alongside),
`exact True` in every run. "verify 1 / 6 / 16" is the boot calibration's window ms (the 1-row window is the first).
Gates are teacher-forced top-1 vs the kit's oracle on the **exact** kernels. On the fast kernels the prompt windows
take the prefill router and the prefill GEMMs, so a decode lever would not even run: G7-decode.sh's `cfg_env fast`
gates were blind to the router and plan levers (`G7.sh dgate` uses exact).

| config | code tok/s (tok / round) | prose tok/s (tok / round) | verify 1 / 6 / 16 ms | draft ms | gate top-1 / first copy | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| base (router fused) | 66.3 (3.88) | 35.3 (1.67) | 33.1 / 55.2 / 92.2 | 3.42 | 0.9963 / 0.9601 | |
| **router gemv** | **68.9** (3.73) | **37.7** (1.65) | **29.2** / 52.0 / 89.9 | 3.35 | 0.9958 / 0.9601 | **adopted**: +3.9% / +6.8%, -3.9 ms a 1-row window |
| router split | 65.2 | 35.0 | 32.7 / 55.0 / 92.9 | 3.58 | (G6: 0.9961) | slower than both |
| proj plan long | 65.3 | 35.5 | 33.4 / 54.3 / 90.9 | 3.38 | 0.9958 / 0.9568 | **not adopted**: no 1-row gain (-0.9 to -1.3 ms only at 6-16 rows), tok/s within noise; MMLU not needed |
| xchg old (all off) | 64.9 | **31.9** (1.52) | 33.9 / 56.5 / 94.0 | 3.54 | | the new levers stay: -0.8 ms a 1-row window, prose +11% |
| draft head q4 | 64.9 (3.77) | 35.6 (1.66) | 32.8 / 54.9 / 89.5 | **3.15** | (exact) | **not adopted**: -0.27 ms a pass, but code acceptance -0.11 tokens a round (-2% tok/s) |
| draft head trim | 66.0 (3.88) | 35.5 (1.68) | 33.3 / 54.5 / 89.5 | 3.39 | (exact) | **not adopted**: draft pass unchanged (3.39 vs 3.42), tok/s flat |
| draft skip off | 66.0 | 35.3 | 32.7 / 54.3 / 94.5 | 3.39 | (exact) | flat: these workloads never trigger skip; stays on (default) |

Router details:

- routerbench (`routerbench-gemv.json`), graphed us a layer, gemv vs split: R = 1: **24.6 vs 112.6 (4.58x)**;
  R = 4: 25.5 vs 112.7; R = 16: 36.9 vs 113.0; R = 64: 74.0 vs 196.5; R = 2,048 (eager): 1,338 vs 5,589 us (4.18x).
  That saves ~3.9 ms in a 44-call window, matching the calibration's 33.1 -> 29.2 ms.
- Agreement and invariance: 2,560 real rows agree in order 100% with split and float64. Invariance: 17,544 row checks,
  0 mismatches; 4,100-row chunks == 16-row windows. Graphs: replays == eager, counters 0. Every config gives the same
  bits.
- The gate run's router check (gemv vs split on all 672,560 real rows): order 99.9985%, set 99.9996%. The 10 differing
  rows are all float64 near-ties (margins 4.5e-8 to 5.3e-7), 5 a side.

**Adopted set** (`TF_DSV41_ROUTER=gemv` + the defaults; `m2-ab-adopted.json`, code / prose / structured, streams
1 / 2 / 4): verify 1 / 2 / 4 / 6 / 8 / 16 rows **29.5** / 36.3 / 44.2 / 52.1 / 60.1 / 90.2 ms, draft 3.32 ms,
`exact_all True`, gate 0.9958.

| | code | prose | structured | C1 | C2 | C4 (aggregate / decode aggregate) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **G7 adopted** | **69.1** (T 0.7: 67.2) | **37.4** (37.7) | **102.2** (107.5) | **70.0** | **48.9** | **67.0** / 76.4 |
| kit (BASELINE) | 41.9-45 | 32.5 | 38-50 | 32.2 | 46.7 | 37.6 |
| G7 / kit | 1.54-1.65x | 1.15x | 2.0-2.7x | 2.17x | 1.05x | 1.78x |
| G4 | 57.8 | 31.3 | 86.0 | 60.5 | 46.1 | 70.7 |
| G7 / G4 | 1.20x | 1.19x | 1.19x | 1.16x | 1.06x | 0.95x |

C4 is 5% under G4's reading and C2 only 1.05x the kit. Per stream, C4 was 27.8 / 19.9 / 32.1 / 39.9 under mixed
sampling (one run, 2 reps). Not investigated in this window (open).

## 3. Rank skew (nsys on both ranks, 1-stream decode, gemv router; `skew.txt`, `decode-skew-r*.txt`)

Rank 1's follower loop never calls `cudaProfilerStart`, so the first try captured nothing on rank 1 and
`nsys_skew.py` had no input. Rank 1 is now traced for its whole run, and the analyser aligns the last 147 windows of
both ranks (147 / 147 aligned, 82 segments a window).

| | rank 0 | rank 1 |
| --- | ---: | ---: |
| kernel time a window (busy, exchanges excluded) | 38.52 ms | 38.52 ms |
| exchanges a window (81 gathers) | 2.37 ms (1.63 without rank 1 traced) | 2.14 ms |
| wait above the p10 floor | 0.99 ms (rank 0 alone traced) | 1.39 ms |
| median per-kernel time ratio r1 / r0 (kernels both run alike) | | 1.001 |
| routed experts `x3ld` | 16.96 ms | 16.98 ms |
| launches | identical counts for every kernel | |

- **G4's rank skew is gone.** Neither rank is systematically slower: busy time a window is equal to 0.001 ms, every
  kernel runs at the same speed (ratio 1.001), and rank 1 has no work of its own (identical launch counts).
- **Clocks and thermals: ruled out.** gpuwatch during G4's decode: head 2,241 / worker 2,229 MHz, 68 C both, 33 W
  both. In this run both idle at the same clocks, with no throttle reasons.
- **The expert split: ruled out on average.** x3ld is equal.
- **What is left is per-segment jitter, not skew.** In 42 of 82 segments rank 1 is busier, in 40 rank 0, by 10-33
  us (largest: segments 70 / 36 / 72, rank 0 +25-33 us; 64 / 66 / 8, rank 1 +15-23 us). Each rank waits for the
  other in alternate segments, about 1 ms a window each.
- **Why it shrank.** Rank 0's exchange wait went from ~2.3 ms (G4) to 0.99 ms. The likely reason is that G7's bf16
  exchanges and the glue / router changes removed per-layer work that ran unevenly. The router alone was 4.75 ms a
  window of Triton fp32 dots.
- **Nothing simple left to fix.** What remains needs per-layer load balance (data-dependent expert sets a token).

## 4. Prefill (G7-prefill2 `knobs`, then all4; cold prompt, one slot, replay + fast tag + gm)

Knobs at 8K / 32K, each alone on base (gm experts + gemv decode router; base 1,343 / 1,421):

| knob | 8K | 32K | x base at 32K |
| --- | ---: | ---: | ---: |
| ahead (`TF_DSV41_PREFETCH_AHEAD=1`) | 1,408 | 1,538 | 1.082 |
| gather (`PREFILL_GATHER=bf16`) | 1,390 | 1,447 | 1.018 |
| rope (`ROPE_INPLACE=1`) | 1,370 | 1,434 | 1.009 |
| mhc32 / **mhc64** / mhc32w16 / mhc64w16 | 1,348 / **1,378** / 1,352 / 1,348 | 1,424 / **1,448** / 1,416 / 1,411 | 1.002 / **1.019** / 0.997 / 0.993 |
| attn (fused, BMQ 16) / **attn32** / attn32w16 | 1,414 / **1,470** / 1,431 | 1,497 / **1,567** / 1,518 | 1.053 / **1.103** / 1.068 |
| stream4k (streaming top-k from 4K keys) | 1,341 | 1,391 | 0.979 (not adopted) |
| prefill router gemv (`TF_DSV41_PREFILL_ROUTER=gemv`) | 1,428 | 1,508 | 1.061 |

**all4** = ahead + gather + rope + mhc64 + attn32 + the gemv prefill router, adopted (`config/prod.env`):

| config | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: |
| **G7 all4, one process 8K -> 128K** (`m2-pf-g7x-s-all4.json`) | **1,833** | **2,043** | **2,068** | **1,953** |
| G7 all4, fresh process a size | | | 2,068 | 2,008 |
| G7 base (gm + gemv decode router), one process | 1,343 | 1,421 | 1,407 (fresh) | 1,379 (fresh) |
| G6 all on + gm | 1,314 | 1,390 | 1,397 | 1,365 |
| kit (vLLM) | 1,073 | 1,075 | 1,060 | 1,031 |
| full, no replay: G7 full4 / base full | 969 / 756 | 1,004 / 772 | 1,003 / 764 | 876 / 739 |

- **Against G6 and the kit.** all4 is 1.39-1.48x G6 and **1.71-1.95x the kit**. TTFT at 128K: 67 s (G6 96 s). Full
  prefill (no replay) is now about the kit's speed (969-1,004 vs 1,031-1,075).
- **Gate (fast kernels; the fused attention changes the fast tag's bits: tag + 1).** all4 top-1 0.9958, first copy
  0.9601. Fast base with gemv: 0.9967 / 0.9601. Both are over 96%.
- **One all4 run was anomalous.** The first one-process all4 run (phase 4, 15:13-15:17) gave 1,830 / 1,897 / 1,516 /
  854. Its rounds were 2.3 s at 128K against 1.0 s. Clocks were normal (2,226-2,234 MHz), but both GPUs drew ~36 W
  instead of ~52 W, so they were stalled, not busy. Every knob alone in one process stays flat to 128K. Two later
  all4 runs (fresh at 64K and 128K, and the full one-process run above) did not reproduce it. Listed under open items.
- **Memory.** Worst MemAvailable in any prefill run: worker 5.66 GiB (all4 128K), head 7.6 GiB.

## 5. Ship gates (production recipe on :8001 at `e87e813` + G7 knobs; `ship/`)

| gate | result |
| --- | --- |
| teacher-forced top-1 vs the kit, prod knobs (cgate) | **0.9958**, first copy 0.9601 (>= 96%; G4 0.9962) |
| MMLU-200 0-shot, replay / full / kit | **88.5% / 88.0% / 87.5%**, same answer 199 / 200: PASS |
| MMLU 20-shot preamble (~2.1K-token prompts), replay / full | 81.1% / 81.1%, same answer 178 / 180: PASS |
| needles, replay 32K / 128K / 299K | found (19.9 s / 74.0 s / **195 s**) |
| needles, full 32K / 128K | found (34.7 s / 137.3 s) |
| structured (17:05, `0933b77` = e87e813 + G8 Engram gate on, prod knobs) | **PASS**: json_schema 12 cases, tools 10 cases (drafted == serial == batched, valid, no markup) |
| tooleval: chains.py thinking off / high | **11 / 12** and **11 / 12** (pass line 10) |
| tool-eval-bench category C (multi-step; workstation over a tunnel to :8001) | **8 / 8, score 100** (GLM prod: 8 / 8) |
| stress 4 x 300K, soak | below (run after 17:10) |

## Memory floors (MemAvailable minimum, GiB; 0.5 s samplers)

| run | head | worker |
| --- | ---: | ---: |
| decode A/B benches (1 stream) | >= 10.3 | >= 7.3 |
| adopted speed table (C1 / C2 / C4) | 10.6 | 8.9 |
| nsys on both ranks | 7.95 | 5.69 |
| prefill knobs and all4 (to 128K, one process) | 7.6 | 5.66 |
| ship replay quality (servers with 4 x 300K) | see `ship/mem-r*.log` | |

Never under the 3 GiB guard. The one guard kill (14:02) was the aborted first A/B: extensions built at boot after a
restage, beside 100 GB of weights. Since then every restage is followed by `G6.sh prebuild`.

## Fixes made (dsv41-060 unless stated)

| commit | what |
| --- | --- |
| `24392ea` | test: split == one-program router names `mode="split"` (the default is fused since `8f24eac`) |
| `e87e813` | prefill attention: BMQ 32 on one pipeline stage (2 stages exceed sm_121's shared memory); a head count BMQ does not divide keeps the chunk kernels |
| dsv41 | `G7.sh` (two-node test queues, decode A/B, exact-kernel gates, both-rank nsys, prefill configs), `nsys_skew.py`, prod.env G7 knobs, TF_COMMIT |

## Open

- **Fast-prefill segmentation dependence** (section 1): ~1 ulp in 1-3 of 2,300 rows. It is in layer 3's q path and
  pre-existing. Tests `test_fast_prefill_row_independent_and_close[gm]` and `test_fast_prefill_with_g7_knobs[knobs0,
  knobs2]` stay red until it is found.
- **C4 / C2 below G4 / the kit's ratio.** C4 is 67.0 vs G4's 70.7, and C2 is 1.05x the kit: eager multi-slot rounds
  (DECODE-ROOFLINE section 6; G8 `multi`).
- **The one slow all4 prefill run** (36 W instead of 52 W at normal clocks; did not reproduce).
- **Not adopted, for later.** Draft heads q4 / trim: q4 costs acceptance, and trim does not shorten the pass here.
  The long projection plan: no 1-row gain.
- **Ship gates not run:** structured, tooleval, stress, soak. They run before the prod switch.
- **Harness.** A RoCE timeout leaves `/cache/roce-failed`, and the next runs silently use NCCL (ab-base ran on NCCL
  once). Windows should check and clear it at each start.
