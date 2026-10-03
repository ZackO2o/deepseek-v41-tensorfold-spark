# G4: decode performance, correctness, prefill, memory on both Sparks (2026-10-02, 06:26-10:42 +07)

One held campaign window (`campaign.sh open` 06:26:22, both nodes rebooted 06:26-06:29, worker first), GLM prod
down for the whole window, G5 (prefill adoption, [G5-RESULTS.md](G5-RESULTS.md)) run in the same window at the
lead's request. Branch `dsv41-060`: G4 ran at `0ff0fc3` (+ `8a6f3a6` test fix, `4b6d155` comm fix restaged
mid-window), the post-fix and new-default runs at `dfc5684` / `5f74397`. Raw files: `results/G4-20261002/`
(`postfix/`: correctness re-run with the router fix; `newdefaults/`: decode, stress, cold boot, NCCL with every
prefill change on). Every bench used prod's FP8 index keys (`DSV_ENV=TF_DSV41_INDEX_KV=fp8`), RoCE unless stated,
`TF_DSV41_EXPERT_LOADS=1`, 0.5 s samplers on both nodes, the 3 GiB guard, `timeout` on every step.

## Bottom line

- **Correctness gates.** Top-1 vs the kit's oracle 99.62% (G2 99.63%), first copy 96.35% (G2 95.35%); row
  invariance, 2,048-row chunk bitwise (both ranks, layers 3 and 20), drafted == serial on every workload at T = 0 and
  0.7, graphs on == off, nucleus == full vocabulary: all hold. **G4 found and fixed a real bug**: the router's
  softplus underflowed to 0 for sink-like rows, 0 / 0 = NaN, and after a mid-sequence EOS + BOS every later token was
  NaN (G2's "prompt-5 BOS loop"). Forced top-1 along the kit's own greedy replies went 86.3% -> **98.2%** (98.8%
  without the kit's ties). Gate 2 (identical 256-token replies) still fails, 0-1 of 8, but every first divergence is
  at a kit near-tie (our token is the kit's #2, margin 0-0.875, mostly <= 0.25).
- **Decode speed (graphs on, RoCE).** code **57.8** tok/s (target 52, kit 41.9-45), prose **31.3** (target 38, kit
  32.5), structured **86.0** (kit 38-50), C1 / C2 / C4 **60.5 / 46.1 / 70.7** (kit 32.2 / 46.7 / 37.6). Round-1 code
  target met; prose is 18% short of target and 4% under the kit.
- **Prefill (old 128-row path) was 211-233 tok/s, flat from 8K to 128K**, against the kit's ~1,050: 128-row windows
  re-read every routed expert each window (58% of GPU time) and the Engram rows were read synchronously (19-23% of the
  wall time). The expert-prefill knob (x3pf) made it slower (215 vs 233). With the G5 changes adopted the same
  measurement gives **931 / 965 / 965 / 949 tok/s at 8K / 32K / 64K / 128K** (0.90x the kit; G5-RESULTS.md).
- **Step time.** A verify window costs 47 ms at 1 row and +5.1 ms per extra row (graphs on); of that, **+3.4 ms is
  routed-expert weight streaming** (each new row brings ~6 new experts per layer), +1 ms GPU idle gaps, +0.3 comm,
  +0.3 EXL3 linears. The fixed ~43 ms: EXL3 linears 12.6, routed experts 9.8, router 4.8, mHC 3.0, CSA2 3.0, torch
  elementwise 2.8, comm 2.1-2.7, a cuBLAS float64 GEMM 2.2 (fixed in G4), gaps ~4.
- **Memory.** Worst MemAvailable over the decode benches: head 7.1, worker 6.4 GiB (nsys runs excepted). The
  **4 x 300K stress with every prefill change on: worker 4.38 GiB minimum (head 5.46)**, a transient in the first
  ~90 s after four prompts were admitted together; the steady state over the rest of the 299K prefill was worker
  5.3-5.8, head 7.0-7.4. Above the 4 GiB hard stop, under the 5 GiB target. Cold boot: first token 1.2 s with MemFree
  < 1 GiB on both nodes, no admission wait (G3: 253 s).

## 1. Correctness gates

| Gate | Result | |
| --- | --- | --- |
| GPU tests (perf + M2 + CPU suites, 62) | 61 passed; 1 failed: `test_graphs_survive_prefill_commit_and_restore` asserted >= 8 replays of 10 windows, but slot 1's keys are captured on first use (logits equal, nothing dropped). Test fixed (`8a6f3a6`), passes on the GPU | PASS after the test fix |
| G5 GPU tests (prefill paths, 51) | 51 passed (fast grouped / tc vs exact logits: rel max 0.0098 / 0.0074, argmax agree 1.00) | PASS |
| cgate: top-1 vs the kit oracle (prod settings: kit rounding on, FP8 keys) | **99.62%** whole, **96.35%** first copy (G2: 99.63 / 95.35); kit top-1 in our top-5 99.97%; median abs logprob error 4.9e-6 | PASS |
| cgate A/B `TF_DSV41_KIT_ROUNDING=0` | 99.67% / 96.01% | no clear winner (+0.05 whole, -0.34 first copy) |
| After the router fix (`postfix/`), exact prefill | 99.62% / 95.35% (rounding on); 99.66% / 96.68% (off) | unchanged |
| Fast prefill tag (G5 gate: grouped / tc) | 99.61% / 97.34%; 99.63% / 96.35% | no quality loss on teacher-forced top-1 |
| chunk: 2,048-row window through GpuMoE == 128-row windows | bitwise on both ranks, layers 3 and 20 (936 / 912 ms a 2,048-row pass) | PASS |
| drafted == serial (T = 0 and 0.7), graphs on == off, nucleus == full vocabulary | True in every run (fast, eager, nucleus0, phases, nsys1/4, new defaults) | PASS |
| Serial decode (gate 4, >= 23) | 20.4 tok/s in M1's eager one-slot gate path; **27.1 tok/s** serial through the serving path with graphs (fast run's `serial_tok_s`) | PASS on the serving path |

### Gate 2: the kit with SPEC_METHOD=none, rescored

The kit really ran without drafting: `spec=none`, `speculative_config=None` in its engine log. (G4-correct.sh's
check flagged "still drafts" from the word `dspark` in an lm_head module path: a false alarm.) 8 x 256 greedy tokens
with top-5 logprobs each step: `kit-greedy-nospec.json`.

| | first run (`0ff0fc3`) | after the router fix (`23991b1`) |
| --- | --- | --- |
| forced top-1 (teacher-forced along the kit's replies) | 86.28% (prompt 5: **2.3%**, NaN after reply token 4) | **98.24%** (98.83% without kit ties) |
| identical replies (kit rounding on / off) | 0 / 0 of 8 | 0 / 1 of 8 |
| identical or ending at a kit tie | 2 / 3 | 2 / 4 |

After the fix, every first divergence is our token = the kit's #2 with a kit margin of 0, 0.125, 0.25 (x3), 0.375,
0.625 or 0.875. The kit's logprobs are bf16 (multiples of 1/16), so these are kit near-ties; no prompt diverges where
the kit is confident. Gate 2 as written (6 of 8 identical) is not reachable without bit-matching vLLM's kernels.

### The bug: router softplus underflow -> NaN after a mid-sequence BOS

Prompt 5's kit reply contains EOS (1) then BOS (0) at reply tokens 3-4 (`ignore_eos`). `gate --trace 60:5` saved
every block's mHC streams: the token after the mid BOS grew sink-like (stream max 3e5 by block 36, max / rms ~43
against the real sinks' ~21), and at block 37 all four streams went NaN; attention then spread it to every later
row and greedy picked id 0 forever. Root cause (found offline by a subagent from the trace and the kit's source):
`router.py` computed softplus as `log(1 + exp(z))` in fp32; below z = -16.6, `1 + e^z` rounds to 1, s = 0 for every
expert, and the renormalised weights are 0 / 0. torch / vLLM's softplus (log1p) stays positive to z ~ -103. Fix
`23991b1`: unchanged arithmetic on [-10, 20] (the same bits for ordinary rows), log1p's series below -10;
interpreter tests show NaN before / finite after. Confirmed on the GPU by the forced rerun (prompt 5: 2.3% ->
99.6%).

## 2. Decode speed (graphs on vs off, RoCE vs NCCL)

`m2bench`, 4 slots x 16K context, REPS 2 (median), mixed sampling for C-runs (every other stream T = 0.7). Kit and
targets as given for round 1.

| workload | graphs on, T=0 / 0.7 | graphs off, T=0 / 0.7 | tok/round | serial (graphs) | kit | target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | **57.8** / 55.9 | 54.0 / 51.4 | 3.92 | 27.1 | 41.9-45 | 52 |
| prose | **31.3** / 30.0 | 27.8 / 26.9 | 1.77 | 26.9 | 32.5 | 38 |
| structured | **86.0** / 92.4 | 79.1 / 84.0 | 5.91 | 27.1 | 38-50 | |
| tweet | 91.1 / 71.0 | 84.2 / 64.1 | 6.55 | 27.2 | | |
| edit | 93.2 / 93.5 | 88.3 / 87.0 | 7.21 | 27.4 | | |
| long (3K ctx) | 53.6 / 49.7 | 51.0 / 44.9 | 3.72 | 27.1 | | |

| concurrent (aggregate tok/s; decode tok/s) | graphs on | graphs off | kit |
| --- | ---: | ---: | ---: |
| C1 | **60.5** (63.3) | 55.1 (57.6) | 32.2 |
| C2 | **46.1** (47.5) | 44.3 (45.7) | 46.7 |
| C4 | **70.7** (73.2) | 67.8 (70.1) | 37.6 |

- Graphs: +7% code, +13% prose, +4% C4. Replies identical on and off.
- C2 < C1 is the workload mix, not a regression: C2 pairs code (T = 0) with prose at T = 0.7 (23.7 tok/s), and the
  aggregate is total tokens over the slower stream's wall clock.
- Nucleus candidates (`TF_DSV41_NUCLEUS=0` A/B): T = 0.7 code 55.9 vs **17.4** with the full vocabulary, prose 30.0
  vs 12.1; T = 0.7 replies identical. The 512-candidate top-p path is worth 3x at T > 0.
- **NCCL vs RoCE**: the first NCCL run crashed at the first T > 0 window (`KeyError: torch.float64`: the nucleus row
  statistics are float64 and the NCCL backend's dtype map had none). Fixed in `4b6d155` and rerun with the new
  defaults (`newdefaults/m2-nccl.json`, REPS 1, graphs on): **NCCL code 52.9 / prose 27.6 / C4 59.8 vs RoCE 59.0 /
  30.5 / 69.8**. RoCE is worth +11% single stream and +17% at C4.
- **After G4's fixes** (router softplus, Triton `weights_proj`), exact prefill (`postfix/m2-fast.json`): code
  **59.9**, prose 30.4, structured 87.9, C1 / C2 / C4 62.1 / 45.7 / 69.2. Rounds are 4-5% shorter (the 2.2 ms fp64
  GEMM is gone). Prose acceptance moved 1.77 -> 1.63 tok/round because the bits of the index weights changed, so the
  prose number is a wash.
- **With every prefill change on** (the new defaults, REPS 2, `newdefaults/m2-fast.json`): code **59.0**, prose
  **30.5**, structured 88.2, tweet 93.0, edit 90.5, long 54.4, C1 / C2 / C4 59.8 / 45.5 / **69.8**. All exact. No
  decode regression from adopting the prefill changes. Prose acceptance after a fast / replayed prompt is 1.66 tok/round.

## 3. Step-time attribution (nsys + phase timers)

nsys of 1-stream decode (code then prose, 160 tokens each, graphs on, `--cuda-graph-trace=node`), kernels attributed
to each verify window's NVTX range by correlation id, rows read from the `_kv_store` grid (`nsys-nsys1.sqlite`):

| rows | windows | window ms (GPU span) | GPU busy | routed experts | EXL3 linears | router | comm | mHC | CSA2 | fp64 GEMM | torch elem. |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 46 | 48.5 | 44.6 | 13.4 | 12.6 | 4.8 | 2.7 | 2.9 | 2.9 | 2.2 | 2.8 |
| 3 | 47 | 55.9 | 50.6 | 17.5 | 12.8 | 4.9 | 4.1 | 3.0 | 2.9 | 2.2 | 2.9 |
| 4 | 9 | 59.2 | 53.1 | 20.7 | 13.1 | 4.9 | 2.9 | 3.0 | 3.0 | 2.2 | 3.0 |
| 5 | 12 | 62.9 | 56.8 | 23.6 | 13.5 | 4.9 | 3.3 | 3.1 | 3.0 | 2.2 | 3.0 |
| 6 | 22 | 69.0 | 61.2 | 27.0 | 13.8 | 4.9 | 3.7 | 3.1 | 3.2 | 2.2 | 3.0 |

**Per extra verify row: +5.1 ms** (G3's eager ~9 ms). Of it: routed experts +3.4 ms (66%: a new row brings ~6 new
experts in each of 40 layers, and each is weight streaming), GPU idle gaps +1.0, comm +0.26, EXL3 linears +0.3, the
rest flat. Boot calibration agrees: verify 1 / 2 / 4 / 6 / 8 / 16 rows = 35.5 / 43.5 / 51.6 / 59.7 / 67.8 / 96.7 ms.

Rounds (`TF_DSV41_PHASES=sync`): code 75.1 ms a round at 4.8 rows a window = graph replay 55.8 + stage 6.3 + draft
7.3 + Engram wait 6.4 + commit 1.0; prose 61.4 ms at 2.7 rows = replay 46.7 + draft 7.3 + stage 3.7 + Engram 3.7.
Unsynced (`PHASES=1`), the round is 64.4 (code) / 54.2 (prose) ms, and 85-90% of it waits in `candidates` for the GPU:
**decode is GPU-bound**.

C4 (nsys4): windows of 16-22 rows cost 134-181 ms (experts 69-90 ms, 52%; comm 10-14 ms; GPU gaps 14-19 ms).

Fixed costs that are not physics:

- **cuBLAS float64 GEMM, 2.2 ms a window**: the indexer's `weights_proj` (`linear.Plain`, 32 x 5,120) ran as a 5-block
  split-K DGEMM, 275 us a call, 8 calls a window, and its algorithm changes with the row count (0.5 ms at 16+ rows):
  not row-invariant by construction. **Fixed** (`af38029`): one Triton launch with a K-only summation order, 0.011 ms
  vs 0.287 ms at 2 rows on the GPU, fp32 values equal to cuBLAS's.
- **Router, 4.8 ms a window, flat**: `_logits` takes 108 us a layer at 1-16 rows (24 programs of serial 5,120-long
  FMA chains on 48 SMs). A split-K form would be ~4x faster but changes the bits; left for a decision.
- **Engram wait 3.7-6.4 ms a round** (sync mode) and **graph stage 3.7-6.3 ms**, host side.
- **GPU idle gaps ~4 ms a window at 2 rows, +1 ms a row**.

## 4. Prefill before G5 (the 128-row path)

`m2bench --prefill` (new in `0ff0fc3`): one slot, a fresh random-id prompt per size (no shared prefix), one token,
after a 2,048-token warm-up. TTFT from the batcher.

| `TF_DSV41_EXPERT_PREFILL` | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: |
| 1 (x3pf) | 215 (38.1 s) | 215 (152.8 s) | 216 (302.9 s) | 211 (621.1 s) |
| 0 (prod) | **232** (35.4 s) | **233** (140.7 s) | **233** (281.5 s) | **228** (574.5 s) |
| kit | 1,073 | 1,075 | 1,060 | 1,031 |

The estimate of ~210 tok/s was right. The rate is flat to 128K, so the cost is per token, not attention. x3pf
loses at 128-row windows (prod keeps it off). A 300K prefill on this path takes ~22 min.

**Where the time goes** (nsys of one 32K prefill, 161.8 s under nsys; `nsys-nsyspre_*.csv`):

- GPU busy 104 s (64%), of which:
  - routed experts 60.1 s (58%): the decode expert kernel (`x3ld ld_kernel`, 2.6 ms a call, 2 a layer a
    128-row window). Each 128-row window touches ~333 of 384 experts, so the whole expert set is streamed once per
    128 rows.
  - EXL3 linears 14.6 s (14%)
  - CSA2 attention / indexer 8.8 s (8%)
  - NCCL all-gathers 7.0 s (7%): prefill partials over 1 MiB go to NCCL, not RoCE
  - router 3.9 s, mHC 3.8 s
- **Engram row reads 30.6 s (19%)**, host-blocked with the GPU idle (synchronous O_DIRECT reads, ~1.06 ms a token)
- host / launch / sync gaps ~27 s (17%)

The G5 changes address the first two: 2,048-row segments read the experts 16x less often, and the native reader
plus bulk prefetch make the Engram wait 3.2 s at 32K.

## 5. Memory floors

| run | head MemAvailable min | worker MemAvailable min |
| --- | ---: | ---: |
| decode benches (fast / eager / nucleus0 / nccl / phases, 4 x 16K) | 10.1-11.1 | 8.6-9.9 |
| nsys runs (nsys adds host memory) | 4.49 (nsyspre), 7.08 (nsys1) | 8.6-8.9 |
| prefill 128K, one slot, old path | 7.11 | **5.40** |
| cgate (4K, one slot) | 14.4 | 7.5 |
| prefill with every G5 change on, 128K | 8.35 | 6.36 |
| **stress 4 x 300K** (pool for 4 x 300K, one 299K + three 64K prompts, every prefill change on) | **5.46** | **4.38** |
| cold boot (4 x 32K, page cache filled to MemFree < 1 GiB) | 11.4 (MemFree 0.8) | 8.8 (MemFree 0.9) |

Stress timeline (worker, 30 s minima): 5.51, **4.38**, 4.58, 4.74, 4.93 GiB for the first 2.5 min after boot (the
four prompts admitted together), then 5.3-5.8 to the end of the 299K prefill. The dip under 5 GiB is the
admission / first-round transient; ENGINE-PLAN 5.0's fallback (CONTEXT 196608) applies if it must stay over 5 GiB.

Stress details: serve booted in 37 s with the 4 x 300K pool (allocated 101.8 GiB a rank). The 299K prompt's first
token came at 667 s, sharing prefill rounds with three 64K prompts (TTFT 75 / 166 / 265 s), ~742 tok/s aggregate
prefill (495K tokens). The three "decode" streams stopped after ~140 streamed chunks instead of 2,048 tokens: our
server does not honour `ignore_eos`, so they ended at EOS. Decode under a concurrent long prefill was therefore only
briefly exercised. The run was not a 4-stream steady-state decode.

## 6. Fixes made (dsv41-060 unless stated)

| commit | what |
| --- | --- |
| `0ff0fc3` | m2bench `--prefill SIZES` and `--nsys prefill` (cold prefill tok/s; nsys of one prefill) |
| `8a6f3a6` | graph test: replays counted per (slot, width) key met before (boot warms slot 0 only) |
| `4b6d155` | NCCL all-gathers of float64 (ncclFloat64 = 8): the NCCL backend crashed at T > 0 |
| `23991b1` | **router softplus keeps s > 0 below z = -16.6**: the NaN after a mid-sequence BOS; forced top-1 86.3 -> 98.2% |
| `af38029` | indexer `weights_proj` in one Triton float64 launch (`TF_DSV41_PLAIN_TRITON`, default on): -2.2 ms a decode window, row-invariant |
| `dfc5684` | every prefill change on by default (direction): fast tag, replay, streaming top-k (2,048-row segments and the native Engram reader were already); test suites pin the old values |
| `5f74397` | fast prefill keeps grouped experts by default: x3tc was 3-5x slower on the real weights (below) |
| dsv41 `3537e9c`, `aed2fbc` | G4-perf.sh prefill / nsyspre steps; prod.env with every prefill change on (`FAST_EXPERTS=grouped` after the measurement) |

## 7. Still broken / open

1. **Gate 2** (6 of 8 identical replies) fails at 0-1 of 8; every divergence is a kit near-tie (section 1).
2. **Prose 31.3 tok/s** is under the round-1 target (38) and the kit (32.5). Prose drafts accept 1.77 tok/round; at
   ~54 ms a round, 38 tok/s needs ~46 ms rounds or ~2.1 tok/round. Levers in the step: router 4.8 ms, idle gaps ~4 ms,
   graph stage 3.7 ms, Engram wait, draft pass 4.8-7.3 ms.
3. **x3tc** (`TF_DSV41_FAST_EXPERTS=tc`) is correct but 3-5x slower than grouped on the real weights (G5): left off.
4. **Stress** dips to 4.38 GiB on the worker in the first ~90 s after four long prompts are admitted together
   (target 5 GiB). The server ignores `ignore_eos`, so the stress client's decode streams end at EOS.
5. Replay (CED) quality was not measured against the kit (the gate's teacher-forced path does not use replay;
   MMLU-200 did not fit the window). Its speed share is 2x (G5).
6. The plain_fp64 interpreter test fails when run in one process after `test_dsv41_m2_decode.py` (Triton interpreter
   state; it passes alone and on the GPU).

## Harness record

- **GLM prod down 255 min 35 s** (06:26:22 -> 10:41:57). Restored by `campaign.sh close` on config/prod.env
  (b13-060 + R8): 4 request slots, /v1/models on :8000 and <your-https-endpoint>, 17*23 = 391, canary
  rc 0. Watchdog timer active, lease deleted. No refresher, deadman, sampler or runner process and no `dsv41-tf-*`
  container left on either node.

- Lease created 06:26:22, refresher re-armed 08:04 (8 h) and 08:21 (10 h); watchdog timer stopped for the window.
- Deadman armed at +6 h (12:26:22). **Extended at 08:21 to 15:30** (recorded in `window.log`) when the lead
  added G5 to the window.
- Reboots: worker 06:26-06:27 (76 s), head 06:28.
- Restaged mid-window, only our own fixes: `comm.py` (07:38), then the whole branch at `dfc5684` for G5 (09:03, after
  the G4 steps), then `prefill_mm.py` (`5f74397`, 09:37).
- Test servers on :8001 only. Caches were dropped before every start (memory gate, MemFree >= 104 GiB). Only
  `dsv41-tf-*` containers were removed; the kit was stopped by its own `start.sh stop`.
