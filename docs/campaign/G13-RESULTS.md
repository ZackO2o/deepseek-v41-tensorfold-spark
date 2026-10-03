# G13: the five rewrites, the combo, and the strict receipt (2026-10-03, 15:04-17:37)

One held campaign window (`campaign.sh open` at 15:04 with `DEADMAN_MIN=240`: lease + refresher, deadman at 19:04,
watchdog timer stopped, GLM's and DeepSeek's prod stopped with 0 requests in flight). Test runs used prod's knobs
(`config/prod.env`) plus each section's lever. Caches were dropped before every start, every step ran under
`timeout`, and the extensions were prebuilt after every restage (15 / 15 on both nodes). Raw files:
`results/G13-20261003/` (`driver.log` = the step log, `window.log` = everything, `*-window.txt`, `*-verdict.txt`,
`summary-g13-*.txt`, `combo-*`, `m2-*.json`, `up-*`). Branch `dsv41-060`: `59b065f` (the five rewrites, every lever
default 0), then `b7b1201` (four test assumptions fixed on hardware, no engine change), then `767ad9f` (the branches
debug-graph test fix, no engine change).

## Verdict

"1-row" is the 1-row verify window from the boot calibration, per rank. "exact" is drafted == serial at T = 0 and
T > 0 in m2bench, and replies == off where the section checked it. Expected ranges are REWRITE-PLAN section 5's.

| rewrite (lever) | expected 1-row | exact | measured on its own (on - off) | adopted |
| --- | --- | --- | --- | --- |
| d1: hide the window graph's submission (no lever) | -0.9 to -1.2 a round | | `graph.replay` host wall **0.02 ms a round** without a profiler. The 1.27-1.35 ms in REWRITE-PLAN 1.2 was measured under nsys | **closed**: nothing to hide |
| e1: mHC boundary in one CUDA launch (`MHC_CUDA`) | -1.2 to -2.0 | yes; gate 0.9960 == off | **1-row -1.3 ms**, 2-row -0.5, 4-row 0.0, 16-row +1.0 (prose +3.0%). Kernel: 32.9 -> 12.7 us at R=1 (2.6x), 35.1 -> 55.5 us at R=16 | **yes** |
| b1: MoE chain shortened (`MOE_FUSED`) | -0.4 to -0.9 | yes (bit for bit R 1..16, real layers); gate 0.9961 == off | **1-row +0.7 ms** (27.4 vs 26.7), 2-row 0.0, 16-row -0.3 | no |
| g + e4: indexer / compressor on a side stream (`BRANCHES`) | -0.5 to -1.0 | yes (on == off over 24 windows) | two runs each: 1-row on 26.2 / **31.5** vs off 26.6 / 26.6 (mean **+2.25 ms**), 2-row +0.4, 16-row -0.55 | no |
| e2 + e3: attention core + indexer top-k in CUDA (`ATTN_CUDA`) | -0.6 to -1.0 | yes; gate 0.9960 == off | both parts **1-row -0.7 ms**, 2-row -0.3; top-k alone -0.5; attention core alone 0.0. Core kernel slower than Triton (23.1 -> 29.1 us at R=1, the <= 12 us line failed); top-k 3-7x faster at n >= 16K | section said "not yet"; **yes in the combo** |
| f2: dense EXL3 over a 16-byte-coalesced repack (`DENSE_V3`) | -0.7 to -1.0 | yes (bitwise == x3seg, 335 matrices) | **1-row -0.2 ms**, 2 / 4 / 16-row **-1.2 ms** (prose +3.7%). MINB 4,2 (`on42`) worse at 16 rows (+1.2) | section said no (1-row line); **yes in the combo** |

**Prod: restored on `767ad9f`** with `TF_DSV41_MHC_CUDA=1`, `TF_DSV41_ATTN_CUDA=1` and `TF_DSV41_DENSE_V3=1`
(`config/prod.env`). **(verified 17:37: 767ad9f with MHC_CUDA, ATTN_CUDA and DENSE_V3 on both ranks; 17*23 -> 391; watchdog on; restore took about 64 s)**

## The combo against off

Window runs (`combo-window.txt`, one boot each; "combo" ran `ATTN_CUDA_PARTS=topk`, minus-ATTN_CUDA_PARTS ran both
attention parts, which is prod's set):

| config | 1 row | 2 rows | 4 rows | 16 rows | prose tok/s | exact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| off / off2 | 26.8 / 27.0 | 32.3 / 32.2 | 39.7 / 39.8 | 73.4 / 72.0 | 40.4 / 40.4 | yes |
| combo / combo2 (MHC + ATTN top-k + DENSE_V3) | 24.8 / 23.7 | 29.7 / 29.7 | 37.9 / 38.0 | 72.7 / 73.3 | 43.5 / 43.2 | yes |
| **minus-ATTN_CUDA_PARTS (prod: MHC + ATTN both parts + DENSE_V3)** | **23.4** | **29.3** | **37.5** | **71.9** | 43.4 | yes |
| minus-DENSE_V3 | 24.9 | 30.8 | 39.1 | 73.8 | 42.0 | yes |
| plus-MHC_CUDA_ROWS-8 | 23.8 | 29.8 | 38.0 | 72.8 | 43.6 | yes |

- **1-row window 26.8 -> about 23.4-24.8 ms** (-2.0 to -3.4 ms), 2-row 32.3 -> 29.3-29.7. Prod's set measured the
  fastest window. The two combo runs differ by 1.1 ms at 1 row, so the gap between "top-k only" and "both parts" is
  inside run-to-run spread. Both parts are in prod because they were never slower.
- **DENSE_V3 pays in the combo** (minus-DENSE_V3: 1-row +0.1 to +1.2, 2-row +1.1 ms) though it missed its own 1-row
  line. **ATTN_CUDA** too: its section failed only the attention core's microbench line, and the window says
  otherwise.
- `MHC_CUDA_ROWS=8` changes nothing measurable. It stays at its default.
- The plan's ranks 1-5 expected ~23.2 ms. With MOE_FUSED and BRANCHES out and d1 closed, 23.4 is at that line.

Speed (`combo-speed.txt`, m2bench, prod's set against off, one boot each; C1 / C2 / C4 are decode aggregates with
mixed sampling):

| config | code T0 | code T0.7 | prose T0 | prose T0.7 | struct T0 | struct T0.7 | C1 | C2 | C4 | exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| off | 81.74 | 71.94 | 41.28 | 42.22 | 116.79 | 116.61 | 82.46 | 67.64 | 93.95 | yes |
| **prod (combo)** | **82.83** | 75.02 | **44.25** | 45.30 | 117.84 | 117.59 | 85.05 | **70.27** | **96.61** | yes |
| vs off | **+1.3%** | +4.3% | **+7.2%** | +7.3% | +0.9% | +0.8% | +3.1% | **+3.9%** | **+2.8%** | |

- **Quality (prod's set):** gate top-1 **0.9961 == off's 0.9961** (first copy 0.9502 both). MMLU-200 0-shot
  **0.875** (175 / 200, 0 errors). Tool chains 11 / 12 (pass line 10). One tool call `get_weather {"city":"Hanoi"}`.
- **Why code gains only 1.3%.** Code verifies about 4.6 rows a round, where the savings are smaller (4 rows -1.8 ms of
  ~40, 16 rows ~0). Its tokens a round also dropped 3.879 -> 3.802 in this pair of runs: the reply is the same, but
  the depth policy reads the cheaper calibrated windows and drafts differently. Prose, at ~1.6 tokens a round, lives
  on the 1-row window and gets the +7.2%. The plan's +14% prose / +11% code assumed d1's ~1 ms a round on top, and d1
  turned out to have nothing to take.

## Why MOE_FUSED and BRANCHES were not adopted

- **MOE_FUSED (b1).** Everything was exact: route == router -> prune -> kit -> group -> rot_in, bit for bit for
  R 1..16 on synthetic and real layers 1 / 20 / 39, with 21 kernels and no spills. In isolation the fused chain saves
  9-46 us a layer at R=1, which is up to ~1 ms a window over 40 layers. **In the window graph it measured +0.7 ms at 1
  row and nothing at 2 rows.** The nsys step (DRAM-idle us a layer, target <= 35) produced no numbers, so where the
  isolated gain goes is not measured. The likely place is that the old chain's small kernels already overlapped
  other work in the graph. It fails its own window line (<= -0.4 ms), and `TF_DSV41_MOE_FUSED` stays 0. Next step:
  the nsys run, before more work on b2.
- **BRANCHES (g + e4).** Exact (on == off over 24 windows; 18 fan-out nodes, 3,075 graph nodes against 3,089). But
  the 1-row window was **unstable with it on**: 26.2 ms in one boot and **31.5 ms** in the other (spread 5.3 ms,
  off's 0.0). The mean is +2.25 ms. Even the good boot (-0.4) missed the -0.5 target, and 2 rows were +0.4 ms. Only
  16 rows gained (-0.55). A side stream in the graph adds a fork / join per layer, which costs more than the overlap
  returns on a 1-row window. `TF_DSV41_BRANCHES` stays 0.

## Strict mode (the upstream issue's receipt)

Prod's set (MHC_CUDA, ATTN_CUDA, DENSE_V3) with every precision-trading knob off: `TF_DSV41_EXPERT_TOPP=0`,
`EXPERT_RENORM=orig`, `MHC_FN=fp32`, `KIT_ROUNDING=0`, `LOGITS=fp32`, `INDEX_KV=bf16`, `PREFILL=full`. The fast
prefill GEMMs and the fused prefill attention stay as prod has them. Same build (`767ad9f`), same session.
Kit = MiaAI-Lab's vLLM kit on the same pair and weights, from the earlier receipts (`m2bench`'s `kit` block); the
kit's cells came over HTTP, with prompts that are not identical.

| | prod (G13 combo) | strict | strict vs prod | kit | strict / kit |
| --- | ---: | ---: | ---: | ---: | ---: |
| code, T0 (tok/s) | 82.83 | **76.88** | -7.2% | 42-45 | 1.71-1.83x |
| code, T0.7 | 75.02 | 74.94 | -0.1% | | |
| prose, T0 | 44.25 | **44.21** | -0.1% | 32.5 | 1.36x |
| prose, T0.7 | 45.30 | 42.18 | -6.9% | | |
| structured, T0 | 117.84 | **111.92** | -5.0% | 38-50 | 2.24-2.95x |
| structured, T0.7 | 117.59 | 109.59 | -6.8% | | |
| C1 (decode, mixed) | 85.05 | **78.96** | -7.2% | 32.2 | 2.45x |
| C2 aggregate | 70.27 | **64.00** | -8.9% | 46.7 | 1.37x |
| C4 aggregate | 96.61 | **89.53** | -7.3% | 37.6 | 2.38x |
| 1-row / 16-row window (calibration, ms) | 23.5 / 72.1 | 24.7 / 81.3 | +1.2 / +9.2 ms | | |
| cold prefill 8K / 32K / 64K / 128K (tok/s, `full`) | (prod runs replay) | **915 / 965 / 959 / 923** | | 1,073 / 1,075 / 1,060 / 1,031 | **0.85 / 0.90 / 0.90 / 0.89x** |
| gate top-1 (first copy) | 0.9961 (0.9502) | **0.9963** (0.9601) | | | |
| exact (drafted == serial) | yes | yes | | | |

- **Decode costs 5-9% against prod** (code -7%, structured -5%, C1 -7%, C2 -9%, C4 -7%), a little better than 2.3's
  8-12% estimate. **Prose at T0 is level** only because the strict reply differs from prod's on this prompt and
  drafts better (1.607 tokens a round against 1.567). At T0.7 it is -6.9%, in line with the rest.
- The strict window is 1.2 ms slower at 1 row and 9.2 ms at 16 rows. The 16-row cost is mostly the unpruned experts:
  more distinct experts a window.
- **Strict prefill is below the kit: 0.85-0.90x** (in-engine, `full`). It is faster than the earlier full-prefill run
  at 128K (923 against 876) and slower at 8K-64K (915-965 against 969-1,004).
- The gate is 0.9963, G2's exact value. Strict quality (MMLU, chains) was not run.

## Upstream's tools (`up-*`, strict server on :8001, `767ad9f`, prod's config + the strict words)

- **`bench_concurrent`** (4 streams, 256 tokens, 3 reps, `--alone --serial`): **0 failed. Every stream == alone
  (12 / 12 in every cell) and == serial**, at T = 1.0 and T = 0.

  | prompt, T | aggregate tok/s | steady | per stream | TTFT max s |
  | --- | ---: | ---: | ---: | ---: |
  | code, 1.0 | 105.7 | 112.0 | 27.8 | 0.55 |
  | chat, 1.0 | 95.6 | 101.1 | 25.6 | 0.58 |
  | code, 0.0 | **161.5** | 166.9 | 42.2 | 0.53 |
  | chat, 0.0 | 130.3 | 135.5 | 33.7 | 0.60 |

- **`bench_openai`** (1 stream, 5 reps, medians): fibonacci-raw **56.0** tok/s at T = 1.0 and **56.1** at T = 0;
  gpu-chat-no-think **50.0** at T = 1.0 and **45.7** at T = 0. TTFT 0.24-0.26 s. These are HTTP decode rates on
  upstream's prompts, not comparable to m2bench's in-engine cells.
- **`prefill_cold`** (HTTP, cold, medians of 3): 8K **9.00 s (910 tok/s)**, 16K 17.8 s (920), 32K **38.4 s (853)**,
  64K **76.5 s (857)**; 1K 2.05 s (1 rep).
  - Two anomalies, not explained yet. At 2K, two of the three reps took ~7 s (6.89 / 7.15) against 2.56 s for the
    third, so the median is 297 tok/s. The first 16K rep took 29.0 s against ~17.6.
  - At 32K-64K, HTTP prefill is ~11% below the in-engine figure (853-857 against 959-965).

## Tests

| section | sources | GPU | CPU / compile | interpreter / engine |
| --- | --- | --- | --- | --- |
| mhc | 59b065f | 73 passed | 20 passed | 19 passed |
| attn | 59b065f | 117 passed | 48 passed | 33 passed |
| moe | 59b065f | 2 failed (`route_equals_production_sequence[128-3-p85k / p85]`) | 1 failed (`router_gemv_compile` spills) | |
| moe | **b7b1201** | **79 passed**, 1 skipped | **155 passed** | |
| dense3 | 59b065f | 1 failed (`graph_replay_and_single_calls`) | 74 passed | |
| dense3 | **b7b1201** | **24 passed** | 74 passed, 6 skipped | |
| branches | 59b065f / b7b1201 | 1 failed (`the_window_graph_forks`, the DOT dump) | 63 passed | 16 passed (BRANCHES=1) |
| branches | **767ad9f** | **6 passed** | 63 passed | 16 passed |

Every failure was a test assumption, with no engine change: `b7b1201` (four tests) and `767ad9f` (the debug graph
class kept `keep_graph`). The failing sections were skipped by the driver and rerun after the restage. Their first
logs are in `first-tests/`.

## Run notes

- The sections ran in three passes: mhc / attn on `59b065f`, moe / dense3 on `b7b1201`, then branches / combo /
  strict on `767ad9f`. Every pass rebuilt the extensions.
- Gate baselines moved between passes: off was 0.9960 (first copy 0.9668) on `59b065f` and 0.9961 (0.9502) on
  `b7b1201` / `767ad9f`. On == off held in every pair. The verdict scripts' "differs from 0.9963" notes are this
  shift, not the levers.
- The combo window step ran `ATTN_CUDA_PARTS=topk`. Speed, gate, quality and strict ran prod's set, with both parts.
- Memory: MemAvailable min was >= 8.7 / 6.9 GiB (head / worker) in every decode run (the worker's 6.9 in the launch
  phases run, >= 8.0 in all the others), and 7.9 / 6.1 GiB in the strict 128K prefill.
- The nsys steps for mhc and moe produced no numbers ("None" in the verdicts). Nothing hangs on them except MOE_FUSED's
  diagnosis.

## Open

- MOE_FUSED: run its nsys step to see where the isolated 9-46 us a layer goes in the graph, before starting b2.
- BRANCHES: the 31.5 ms boot. Find out whether a fork / join per layer is the cost, or a bad boot.
- The ATTN_CUDA attention core is slower than Triton in isolation and only helps paired with top-k. Revisit the
  kernel or ship top-k alone after a repeat window.
- `prefill_cold`'s 2K reps (~7 s) and the 11% gap between HTTP and in-engine prefill at 32K-64K.
- Strict quality (MMLU-200, chains) for the upstream receipt.

## Close and verification (prod = **767ad9f** + MHC_CUDA / ATTN_CUDA / DENSE_V3)

- **17:36.** Manual restore (`prod-switch.sh restore`, marker dsv41), 12 min after the strict tools finished. It
  started DeepSeek prod from `config/prod.env` and verified it at 17:37:22: models, 17*23 = 391, canary ok, https ok.
  It enabled dsv41-boot-start, dsv41-tf-watchdog.timer and dsv41-watchdog-rearm.timer, and wrote the marker. Prod was
  down ~153 min (15:04-17:37).
- **Verified 17:37:** 767ad9f with MHC_CUDA, ATTN_CUDA and DENSE_V3 on both ranks; 17*23 -> 391; watchdog on; the
  restore took about 64 s.
- **Rollback.** `config/prod.env`: TF_COMMIT back to `a6f5792e053c1d281d6e12de975a3bbf3a535ca1` (still staged), and
  drop the three `TF_DSV41_*` lever lines (or set them to 0, which is `767ad9f`'s default and runs the G12 code). Then
  `scripts/serve.sh stop && scripts/serve.sh start` on the head.
