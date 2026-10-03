# G10 (overnight 2026-10-02 -> 03, inside the G9 window): the combined adopted set, verify budget, PDL / L2PF, draft acceptance

## MORNING SUMMARY (final, 2026-10-03 06:10)

**Prod at 08:00 runs G8 (`5ace28b`).** The deadman restores DeepSeek prod from config/prod.env (marker dsv41). Not
the best config, because its stress gate (worker MemAvailable >= 5 GiB) failed and the rule was "pin only if stress
passes". **But G8 fails that same stress just as far** (4.07 GiB vs the best's 3.6-4.4 over six runs), and no commit
since G4 has held 5 GiB there. Every other ship gate passed on the best config.

**Recommendation: switch to the best config**, about +10% across the board with the same memory behaviour. Its prod
directory is already staged on both nodes. Run on head:

    cd ~/src/deepseek-v41-tensorfold-spark
    sed -i 's/^TF_COMMIT=.*/TF_COMMIT=38f65008e323b8b1c19c44fff50897f6f90a7611/' config/prod.env
    CONFIG=config/prod.env scripts/serve.sh restart

(and the same TF_COMMIT edit in the workstation's repo).

**Best exact config** = G8 prod + G9 glue (on + fn16) + prune p85k (decode) + joint off + dense off + G10 L2
prefetch 12 MiB. Staged `38f6500`; the knobs are in config/prod.env.

| tok/s (T0 / T0.7; C = decode aggregate) | code | prose | structured | C1 | C2 | C4 | 1-row window |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| **best exact (G10)** | **79.0-81.5 / 72** | **40.9-41.2 / 42** | **116.6 / 117.5** | **82-83** | **67.1** | **93.5** | **26.9 ms** |
| G8 (the prod the deadman restores) | 72.7 / 69.2 | 39.0 / 38.8 | 109.1 / 107.9 | 73.6 | 61.7 | 83.9 | 29.5 ms |
| kit (Mia vLLM) | 41.9-45 | 32.5 | 38-50 | 32.2 | 46.7 | 37.6 | |
| best / kit | 1.8-1.9x | 1.26x | 2.3-3.1x | 2.6x | 1.44x | 2.5x | |

**Ship gates on the best config:**

| gate | result |
| --- | --- |
| top-1 vs the kit | 0.9963 |
| MMLU-200 | 87.5% (kit 87.5) |
| structured | 22 / 22 |
| tool chains (thinking off / high) | 11 / 12, 11 / 12 |
| tool-eval-bench C | 8 / 8 |
| 30-min soak | PASS: 529 requests, 0 errors |
| **stress 4 x 300K** | **FAIL**: worker MemAvailable 4.01 GiB against the 5 GiB line, for 58 s of the 299K prefill. Not L2PF (4.35), not pruning (3.94); G7's commit dipped to 4.42 for 11 s |

**Still under 2x the kit on prose (1.26x) and 2 streams (1.44x).**

- **DRAFT_K 256 / 1024:** no gain (prose +0.6%).
- **Drafter self-distillation (G11, docs/G11-RESULTS.md):** delta A gives prose **+5.5% (43.3 tok/s)** but code
  -4.4%. Delta B is +17.6% tokens a round offline, but only +3.1% prose in the engine and -5.7% code. Not adopted.
  This is the one lever that moved prose acceptance; it needs balanced data and the training port's fidelity fixed.

**Lossy verify budget (report only, NOT adopted; do not use):**

| B | code | prose | C2 | C4 | greedy identical to exact | MMLU-gen (exact 86.0%) | tool chains (exact 11 / 12) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| exact | 78.4 | 39.2 | 66.1 | 93.3 | 100% | 86.0% | 11 / 12 |
| 4 | 89.3 | 41.1 | 65.2 | 100.7 | 12% | 84.5% | 4 / 12 |
| 2 | 95.0 | 44.6 | 66.6 | 107.6 | server failed its canary | - | - |
| 0 | 184.6 (degenerate loops) | 55.0 | 83.9 | 131.6 | 5% | 79.5% | 1 / 12 |

**Verdicts tonight:**

| lever | verdict |
| --- | --- |
| joint | FAIL |
| dense v2 | slower |
| glue | adopted, -0.8 / -1.2 ms |
| prune p85k | adopted, x1.05, MMLU 88.5 |
| L2PF | adopted, +2.5-3.5% |
| PDL | no gain |
| budget | not viable |
| draft trees | s2 0.17, skipped |
| DRAFT_K 256 / 1024 | no gain |
| drafter delta A / B (G11) | prose +5.5% / +3.1%, code -4.4% / -5.7%: not adopted |
| ship gates, best config | all pass except stress memory (4.01 GiB; G8 4.07: same) |
| stress dip cause | rank-1 host anon growth during a 299K prefill; not sessions / malloc arenas / L2PF / prune / prefetch-ahead; open |
| balanced delta A (G11b) | prose +4.1%, code -2.9%: not adopted |

**Fixed:**

- dsv41-060 `2905959`: the PDL import shadowed `P` in expert_loads; every decode MoE call raised.
- `38f6500`: test bug.
- G9 / G10 harness additions.

**State (04:25).** The G9 window is held: lease, refresher, GLM and dsv41 timers stopped, **deadman 08:00**.

- **What 08:00 brings up.** DeepSeek prod (marker dsv41) from config/prod.env at `TF_COMMIT=5ace28b`: G8's validated
  commit, its prod directory staged on both nodes, its CUDA extensions verified cached at 04:21. The newer knobs in
  prod.env are unknown to that commit and ignored.
- **To serve the best config instead:**
  1. accept or fix the stress memory dip;
  2. set `TF_COMMIT=38f65008e323b8b1c19c44fff50897f6f90a7611`;
  3. run `serve.sh stage`, then `serve.sh restart`.

- **Window.** Window G9 (opened 19:30) stays held. Deadman extended to 08:00 (`results/G9-20261002/window.log`).
  DeepSeek prod is down, and the user decides in the morning.
- **Branch.** `dsv41-060` staged at `38f6500`: G9's sections + the PDL import fix (`2905959`) + the expert_block test
  fix.
- **Results.** Combined run: `results/G9-20261002/` (`m2-ab-combined*.json`, `g5-gate-combined.json`,
  `mmlu0-combined.json`). G10 sections: `results/G10-20261002/`.

## 1. The combined adopted set (exact)

`results/G9-20261002/combined-knobs.txt`:

- **G8 prod:** gemv router, Engram gate, row graphs, TCP plan link, speculative DSpark, all4 prefill.
- **G9:** joint depth 0, dense 0, glue on + fn16, prune p85k (decode).

| | code T0 / T0.7 | prose T0 / T0.7 | structured T0 / T0.7 | C1 | C2 | C4 (decode aggregate) |
| --- | --- | --- | --- | ---: | ---: | --- |
| **combined (mixed)** | **80.2** / 70.6 | **39.9** / 40.9 | **114.5** / 115.1 | **76.0** | **63.0** | 79.1 (**92.8**) |
| combined (C runs greedy) | 78.5 / 69.0 | | | 75.8 | 60.8 | 85.3 (92.2) |
| G8 (all on) | 72.7 / 69.2 | 39.0 / 38.8 | 109.1 / 107.9 | 69.1 | 58.9 | 77.5 (83.9) |
| kit | 41.9-45 | 32.5 | 38-50 | 32.2 | 46.7 | 37.6 |
| combined / kit | 1.78-1.91x | 1.23x | 2.3-3.0x | 2.36x | 1.35x | 2.10x (agg) |

- **Window.** verify 1 / 2 / 4 / 6 / 8 / 16 rows **27.8** / 33.5 / 40.9 / 48.3 / 55.8 / 73.4 ms (G8 29.5 / 35.8 / 43.9
  / 52.0 / 60.2 / 77.8); draft 3.50 ms. `exact_all True` (drafted == serial).
- **Quality.** Top-1 vs the kit (exact kernels) **0.9963**, first copy 0.9502. MMLU-200 0-shot **87.5%** (kit 87.5;
  p85k alone 88.5). Tool chains (thinking off) **11 / 12** (pass line 10). The regression queue (decode glue +
  forward GPU suites at `38f6500`): 47 passed.

## 2. Verify budget (`TF_DSV41_VERIFY_BUDGET=B`, LOSSY; on the combined set): **not viable** (report only, not adopted)

- **Tests.** 101 passed, 38 skipped (the interpreter-only suites; the GPU suite ran).
- **Speed and quality.** m2bench at C1 and C2 / C4 decode aggregate. The budget runs use `--no-exact`. Quality comes
  from a test server a setting: greedy 32 x 256 tokens vs exact, MMLU-200 generation-based, tool chains with thinking
  off.

| B | code | prose | structured | C2 | C4 | greedy tokens identical to exact | MMLU-gen | tool chains |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| exact | 78.4 | 39.2 | 112.2 | 66.1 | 93.3 | 100% | 86.0% | **11 / 12** |
| 4 | 89.3 | 41.1 | 132.4 | 65.2 | 100.7 | **12.4%** (first divergence median token 13; 0 / 32 fully identical) | 84.5% | **4 / 12** |
| 2 | 95.0 | 44.6 | 138.9 | 66.6 | 107.6 | server **failed its canary** (wrong answer) | - | - |
| 0 | 184.6 (7.4 tokens a round from 2.4 drafted: degenerate repetition) | 55.0 | 148.6 | 83.9 | 131.6 | 5.5% (median divergence at token 5) | 79.5% | **1 / 12** |

- **Calibration** (budgeted verify, its own table). B 4: 1 / 4 / 16 rows 28.7 / 36.4 / 44.3 ms; B 0: 28.6 / 32.3 /
  40.1. The windows do get much cheaper.
- **The cost is quality.** Approximate verification changes most tokens after the first few: 88-95% of greedy tokens
  differ. Tool calling collapses (4 / 12 at B 4, 1 / 12 at B 0), and B 2 cannot pass the server's own 17*23 canary.
- **Speed is not a real win.** B 0's code 2.4x is the drafter and lookup accepting degenerate loops, not speed.
- **pick: adopt none.** The pass line is MMLU-gen >= 0.865 and >= 95% identical tokens; no B is close.
  **Recommendation: drop this mode.** Even as an opt-in it breaks tool calls.

## 3. PDL and cross-op L2 prefetch (`TF_DSV41_PDL`, `TF_DSV41_L2PF`; bit for bit; on the combined set)

- **Tests.** CPU 72 passed (55 interpreter-only skipped). GPU with the knobs off: 11 passed. The bitwise GPU suites
  again with `PDL=1 L2PF=1`: **63 passed** (graphs == eager, drafted == serial, multistream).
- **l2probe** (GB10: L2 24 MiB, persisting max 18 MiB). 12 MB: cold 52.0 us (242 GB/s), hot 13.2, L2-prefetched
  18.2 (690 GB/s).
- **Speed.** Calibration with a fresh dir; code / prose T0 / T0.7, C1 / C2 decode aggregate; exact True in every
  run.

| config | verify 1 ms | code | prose | C1 | C2 |
| --- | ---: | --- | --- | ---: | ---: |
| off | 28.0 | 79.6 / 70.3 | 39.8 / 40.9 | 80.7 | 65.8 |
| pdl | 27.8 | 79.6 | 39.7 | | 64.2 |
| l2pf (default 8 MiB a site) | 27.2 | 80.9 / 71.5 | 40.8 / 42.0 | 82.2 | 67.1 |
| **l2pf, 12 MiB a site** | 27.4 | **81.6** / 71.8 | **41.2** / 42.3 | **82.5** | **67.4** |
| both | 27.3 | 81.8 / 72.0 | 41.0 / 42.6 | 82.2 | 67.7 |

- **Bisect.** No PDL part passes on its own (seg, single, experts, triton: verify-1 within +-0.1 ms). No single
  L2PF site passes either (x, o, f alone: 28.3-29.2 ms): the gain needs all three sites. 4 MiB a site fails; 12
  passes.
- **pick: adopt `TF_DSV41_L2PF=1 TF_DSV41_L2PF_MB=12`, `TF_DSV41_PDL=0`** (in config/prod.env). Code +2.5%, prose
  +3.5%, C2 +2.4%. That is the expected -0.4 to -0.6 ms at the low end of the model (GLM 0460 saw a third of its
  model too); PDL's -0.35 to -0.68 ms did not show.

## 4. Draft acceptance (G10-draft: tests, capture, analysis; no training)

- **Tests.** CPU 47 passed; GPU 19 passed (PC tree replies == serial with row + pass graphs, the log's hidden rows).
- **Capture.** 184 prompts at C4 on the combined set, `TF_DSV41_DRAFT_LOG` on for the capture only: **34,753
  DSpark passes**.
- **Analysis** (`draftsim.py`):

| draft position | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| conditional acceptance | **0.506** | 0.415 | 0.359 | 0.349 | 0.385 |
| recall@1 / @2 / @4 (teacher forced) | 0.506 / 0.592 / 0.642 | 0.340 / 0.403 / 0.439 | 0.220 / 0.262 / 0.291 | 0.154 / 0.183 / 0.203 | 0.109 / 0.131 / 0.147 |
| **s2** = P(serial is the draft's 2nd \| not its 1st) | **0.174** | 0.095 | 0.055 | 0.035 | 0.025 |

- **s2 = 0.174 at position 1, under the 0.35 line**, so `speed` was **not run** (per instruction). The treesim
  replays agree: chain 39.13 tok/s, tree d1 w2 B3 39.18, PC trees d2 / d3 / w3 39.17-39.18 (+0.1%). Siblings win
  415-461 times of 34,753 passes.
- **Where the miss is.** At rejections the target's own 2nd choice is the draft only 24.6% of the time; the margin
  median is 2.5 logits (10.6% under 0.5). The serial token is outside the drafter's top-4 in 28% of first positions.
  So width does not help: the drafter is wrong, not near.
- **Other levers in the log.** The Markov scale is best at 1.0 (today's), an n-gram mix adds +0.3% tokens a round,
  and per-position temperature calibration moves ECE 5.5 -> 5.6% at position 1 (no gain). The remaining lever for
  prose acceptance is the drafter's weights (self-distillation on this log: `train`, gated on the user's
  go-ahead).


## 5. Ship gates on the best exact config (`38f6500` + config/prod.env's adopted knobs, test server on :8001; `ship/`)

| gate | result |
| --- | --- |
| structured (json_schema x thinking off / on, tool calls) | **PASS**: 12 + 10 cases |
| tool chains, thinking off / high | **11 / 12**, **11 / 12** (pass line 10) |
| tool-eval-bench C (workstation over a tunnel) | **8 / 8, score 100** |
| soak, 30 min | **PASS**: 529 requests, 0 errors, 77 cancels, drained, 17*23 = 391; MemAvailable min head 5.11 / worker 4.07 GiB (line 4) |
| stress 4 x 300K (one 299K prefill + 3 x 64K decoding 2,048 tokens, ignore_eos) | every stream complete (2,048 tokens), long first token 217 s; **worker MemAvailable min 4.01 GiB: FAIL** against the 5 GiB gate (58 s under 5 during the long prefill; G7: 4.42 for ~11 s) |
| stress, L2PF off | worker min 4.35 GiB, 67 s under 5: not the cause |
| stress, pruning off | worker min 3.94 GiB, 58 s under 5: not the cause |

**Verdict.** Stress fails its 5 GiB line, and the dip comes from the commit's long prefill under concurrent decode,
not from a knob. It stays above the 3 GiB guard and near the 4 GiB admission floor. Per the rule, config/prod.env
**stays on `TF_COMMIT=5ace28b`**: the 08:00 deadman restores G8. Everything else passed. To ship the best config
anyway: accept a ~4.0 GiB floor for ~1 min at a 299K prefill, or find the prefill-side allocation (open).

## 6. Drafter candidates (`TF_DSV41_DRAFT_K`, exact; best config, 1 stream, 2 reps)

| K a rank | code T0 / T0.7 | prose T0 / T0.7 | prose DSpark tokens a round | draft ms | verify 1 / 2 rows |
| --- | --- | --- | ---: | ---: | --- |
| 64 (today) | 81.25 / 72.42 | 41.05 / 42.29 | 1.576 | 3.68 | 26.8 / 32.9 |
| 256 | 81.60 / 71.47 | 41.28 / 41.91 | 1.589 | 3.60 | 27.1 / 32.5 |
| 1024 | 81.47 / 72.23 | 41.09 / 42.09 | 1.596 | 3.63 | 26.9 / 31.8 |

`exact_all True` everywhere. Prose gains are +0.6% / +0.1% (line +2%): **not adopted**. More candidates add ~1% tokens
a round. The missing tokens are not won back by width (G10 section 4: the drafter is wrong, not near). G11 runs
with K = 64.

## 7. The stress memory dip (early 2026-10-03): blocked, and **G8 has it too**

Same stress each run: one 299K prefill + three 64K prompts decoding 2,048 tokens. The 64K prompts queue behind the
long one (long prompts are admitted one at a time), so the dip happens during the 299K prefill alone. Worker
MemAvailable minimum:

| run | commit / change | worker min GiB |
| --- | --- | ---: |
| ship (00:47) | 38f6500, best knobs | 4.01 |
| L2PF off | 38f6500 | 4.35 |
| pruning off | 38f6500 | 3.94 |
| diag2 (1 s worker sampler) | 38f6500 | 3.61 |
| sessions off | 38f6500 | 4.35 |
| prefetch-ahead off | 38f6500 | 4.17 |
| MALLOC_ARENA_MAX=2 (serve.sh now passes `MALLOC_*`) | 38f6500 | 4.01 |
| **G8 prod (5ace28b), prod.env's knobs** | 5ace28b | **4.07** |

- **What grows (worker sampler, `ship-diag2/wsamp-diag2.log`).** The rank-1 process's host anonymous memory climbs
  from 0.87 GB at boot to 2.7-4.7 GB at the minimum, and later to ~7 GB, without coming back. It is not glibc arena
  fragmentation (arena max 2: the same), not the session tier, not the Engram row cache (capped at 17 MB), and not
  prefetch-ahead. The early minimum also has a device-side part: MemAvailable is lowest before RSS peaks.
- **Not found in the time box.** Next step: a `tracemalloc` / `torch.cuda.memory_snapshot` hook in the rank-1 loop.
- **G8 has the same dip.** The G8 commit the 08:00 deadman restores dips just as far (4.07 GiB), so the best config
  is no worse on memory. The 5 GiB line has never held for any commit since G4 (G4 4.38 transient; G7 4.42). It stays
  above the 4 GiB hard floor (nothing new admitted below it) and the 3 GiB guard.
- **A hang seen once.** The first diagnostic stress hung at stage `7d940f0` (G11's commits) with a sampler running
  nvidia-smi every second: 4 requests in flight for 19 min, 63 tokens, both GPUs at 92-96% utilization at 17-19 W
  (spin-waiting), no error. It did not recur on 38f6500 without nvidia-smi (6 later stress runs). Cause unknown:
  either G11's commits or nvidia-smi polling; open.

**Prod switched (2026-10-03 09:00 +07).** With the user's approval ("4 GB memory is fine"), production runs 38f6500,
the best exact config. The 08:00 deadman restored dsv41. At 08:59, `campaign.sh close` (G9 campaign dir) ran
`prod-switch.sh restore` (rc=0): sources 38f65008e323, 4 slots, canary ok. Verified: :8000 and https list
DeepSeek-V4.1-Flash-TF, deepseek-v4.1-flash and the GLM-5.3-Flash-EXL3 alias. 17*23 = 391, also via the GLM alias.
A tool call returns `get_weather {"city":"Hanoi"}`. A strict json_schema reply is valid. 4 concurrent streams
finish (TTFT 0.4-0.8 s). Only the dsv41 automation is enabled. There is no lease, deadman, refresher or sampler,
and only dsv41-tf-r0 / -r1 run on the two nodes.
