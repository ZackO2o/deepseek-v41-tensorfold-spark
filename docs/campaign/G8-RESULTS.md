# G8: host-side decode fixes (DECODE-ROOFLINE section 6), inside the G7 window (2026-10-02, 16:54-)

The window opened for G7 at 13:33 stays held: lease + refresher, deadman 23:33, GLM's timers stopped, no prod on
:8000 (direction: performance first, prod after G8). Branch `dsv41-060` staged at `0933b77`. Base: G7's adopted
set (`TF_DSV41_ROUTER=gemv` on HEAD's defaults). Raw files: `results/G8-20261002/`.

## 1. Engram gate (`TF_DSV41_ENGRAM_GATE=1`, 0933b77): the GPU, not the host, waits for decode's Engram rows

- **Tests.** Prebuild: 9 / 9 extensions on both nodes (+ `engram_gate`). The gate's GPU and CPU suites plus
  engram_native: **27 passed**.
- **A/B.** One off / on A/B, m2bench code at C1 / C2 / C4 (1 rep, mixed sampling); ms a round from the phases
  (`engram-phases.txt`):

| C | tok/s decode aggregate off -> on | round ms off -> on | engram.wait ms | graph.stage ms | forward ms |
| --- | --- | --- | --- | --- | --- |
| C1 | 71.9 -> **72.5** (+0.8%) | 54.6 -> 54.3 | 0.97 -> 0.16 | 0.90 -> 0.07 | 0.94 -> 0.12 |
| C2 | 52.3 -> **54.1** (+3.5%) | 66.3 -> 64.2 | 8.19 -> 0.19 | 1.55 -> 0.04 | 24.7 -> 19.4 |
| C4 | 79.0 -> **81.0** (+2.6%) | 88.3 -> 85.5 | 13.12 -> 0.06 | 0.27 -> 0.04 | 34.6 -> 31.0 |

- **Exactness.** `exact_all True` in both runs (drafted == serial). The host waits are gone (C4: 13.1 -> 0.06 ms a
  round).
- **Smaller than the wait it removed.** The round shrinks by 2-3 ms, not 8-13. Most of the old wait overlapped GPU
  work already queued, and the remaining multi-slot cost is the eager forward (19-31 ms a round at C2 / C4: section
  `multi`).
- **Verdict: adopted** (exact and faster at every concurrency). The G8 adopted set is now
  `TF_DSV41_ROUTER=gemv TF_DSV41_ENGRAM_GATE=1`. nsys was skipped (results not odd; direction).

## Notes for the next sections

- **Gate on exact kernels.** G8.sh's `dense gate` uses `cfg_env fast`. On the fast kernels the teacher-forced windows
  take the prefill GEMMs and the prefill router, so a decode-only lever such as dense decode linears is not exercised
  (G7 found this for the router and projection plan). Gate decode levers with `G7.sh dgate` (exact kernels).
- **Samplers.** G8.sh does not start the 0.5 s samplers. They now write to `results/G8-20261002/` (started 17:02).

## 2. Dense (`TF_DSV41_DENSE`, f47cb31) and multi (`TF_DSV41_GRAPH_MODE=rows`, acc165b), batched

Staged once at `acc165b`. Prebuild: 10 / 10 on both nodes (`dense` added to `prebuild_ext.py`). Tests ran at once:
dense on the head GPU, multi on the worker GPU, multi's interpreter suites on the head CPU.

| tests | result |
| --- | --- |
| dense (x3dn GPU + emulator + sm_121 build + the fused_proj suites) | **65 passed**, 5 skipped. x3dn within 3.3e-7 of float64 at every real shape, R 1..64 |
| multi interpreter (row-mode kernels, forward and DSpark pass in row mode == per segment) | **22 passed** |
| multi GPU (multistream + M2 + draft GPU + CPU suites) | 85 passed, **1 failed: test bug**. `test_row_graph_replay_equals_eager`: every replay == eager assertion passed (logits, proj, taps, positions, carries over 24 mixes). Only the one-graph-a-key bound failed (17 vs 9), because it counted the graphs held from bind (context bucket 0), which the loop never uses. The sibling test subtracts them. Fixed in `948a558` |

Gates (teacher-forced top-1 vs the kit, **exact** kernels, on gemv + Engram gate):

| TF_DSV41_DENSE | top-1 | first copy |
| --- | ---: | ---: |
| 0 | 0.9958 | 0.9601 |
| attn | 0.9968 | 0.9668 |
| all | 0.9964 | 0.9701 |

One combined A/B. Base = the G8 Engram base: gemv + Engram gate, `GRAPH_MODE=mix` (G7's graphs), `DENSE=0`. m2bench
code / prose / structured, 2 reps, C1 / C2 / C4 with mixed sampling; tok/s at T = 0 / T = 0.7, concurrent aggregate
(decode aggregate). `exact True` in every run.

| config | code | prose | structured | C1 | C2 | C4 | verify 1 / 16 rows ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base (mix, dense off) | 71.7 / 67.1 | 38.4 / 38.6 | 108.4 / 106.8 | 68.4 (72.8) | 43.1 (50.0) | 68.4 (75.9) | 29.5 / 81.5 |
| **rows** | **72.6** / 67.8 | 38.3 / 38.6 | 108.3 / 106.5 | 68.8 (73.4) | **57.7** (60.4) | **76.3** (82.7) | 29.5 / 79.3 |
| rows + dense attn | 69.7 / 64.0 | 37.0 / 36.6 | 103.6 / 102.4 | 66.4 (70.6) | 54.9 (57.3) | 74.5 (80.4) | 30.9 / 84.6 |
| rows + dense all | 67.6 / 66.2 | 35.6 / 33.6 | 101.6 / 99.9 | 64.8 (68.8) | 51.2 (53.4) | 68.7 (73.8) | 31.2 / 84.5 |

- **Row graphs: adopted.** C2 +34%, C4 +12%; single stream unchanged. It is the branch default at `acc165b`.
- **Dense: not adopted.** Slower on every cell: the 1-row window is +1.4 ms (attn) and +1.7 ms (all), code -4% / -7%.
  Its gate is fine (top-1 0.9968 / 0.9964) but it does not pay on GB10 against x3seg. The dense tests' own
  micro-timings agreed (x3seg 21-31 us vs x3dn 24-45 us on `x4`, R 1-16). `TF_DSV41_DENSE=0`.
- **Memory.** The ab-g8* memory lines are empty: the G7 ship steps had stopped the samplers. The 3 GiB guard was live
  throughout (no kill), and the samplers were restarted at 18:02.

## 3. Host tail (`c9658a2`) and the combined run (staged `5ace28b`; tree verification present, off)

- **Tests.** Host GPU + CPU suites (device candidates == lexsort, device choice == choose_rows, speculation:
  drafted == serial at 1 / 4 slots, two-rank TCP plan link) + regressions: **92 passed**. The multistream test after
  its fix (`948a558`): 5 passed.
- **Combined A/B.** Same stage, all with gemv + Engram gate + dense off; code / prose / structured at T = 0 / 0.7,
  C1 / C2 / C4 mixed (decode aggregate). `exact_all True` everywhere.

| config | code | prose | structured | C1 | C2 | C4 | verify 1 / 2 / 4 / 16 rows ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G8 Engram base (mix graphs) | 71.8 / 68.3 | 38.5 / 38.5 | 107.9 / 106.7 | 68.5 (72.9) | 52.9 (55.2) | 76.1 (82.3) | 29.6 / 35.6 / 44.1 / 81.6 |
| + row graphs | 71.5 / 68.6 | 38.0 / 38.6 | 108.5 / 107.8 | 68.3 (73.0) | 58.9 (61.7) | 78.3 (84.9) | 29.6 / 35.8 / 43.9 / 80.5 |
| **+ TCP plan link + speculative DSpark (all on)** | **72.7 / 69.2** | **39.0 / 38.8** | **109.1 / 107.9** | **69.1** (73.6) | **58.9** (61.7) | **77.5** (83.9) | 29.5 / 35.8 / 43.9 / 77.8 |

- **Speculative DSpark.** 899 passes launched, **892 used (99.2%)**, 7 dropped. Pass line 60%.
- **Second verify row.** It costs ~6.3 ms (29.5 -> 35.8), against ~4 ms for each later row (43.9 at 4 rows): a prose
  lever.
- **Gate.** All on, exact kernels: top-1 **0.9958**, first copy 0.9601.
- **Adopted.** Engram gate, row graphs, TCP plan link, speculative DSpark pass. Dense off (slower). Tree not run yet.

Against the kit and earlier windows (G8 all on):

| | code | prose | structured | C1 | C2 | C4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **G8** | **72.7** | **39.0** | **109.1** | **69.1** | **58.9** | **77.5** |
| G7 adopted | 69.1 | 37.4 | 102.2 | 70.0 | 48.9 | 67.0 |
| G4 | 57.8 | 31.3 | 86.0 | 60.5 | 46.1 | 70.7 |
| kit | 41.9-45 | 32.5 | 38-50 | 32.2 | 46.7 | 37.6 |
| G8 / kit | 1.62-1.74x | 1.20x | 2.2-2.9x | 2.15x | 1.26x | 2.06x |

## 4. Production switch (18:51-18:52)

- **Pinned.** `TF_COMMIT=5ace28b` + G7 / G8 knobs (`config/prod.env`, dsv41 `f058cbe`), staged to
  `~/src/dsv41-prod/5ace28b...` on both nodes, image `dsv41-tensorfold:b13-060-base` (pinned, equal content keys),
  static preflight ok.
- **Switch.** `prod-switch.sh install-units`, `echo dsv41 > the prod-stack marker`, `campaign.sh close`. restore_prod ->
  `prod-switch.sh restore`: DeepSeek started (memory gate, 4 slots, canary), verified, its automation enabled. The
  window closed at 18:52:17 (lease, refresher and deadman gone). GLM had been down since 13:33 (319 min).
- **Checks after the switch:**
  - `<your-https-endpoint>/v1/models` lists DeepSeek-V4.1-Flash-TF, deepseek-v4.1-flash and
    GLM-5.3-Flash-EXL3;
  - model "GLM-5.3-Flash-EXL3" is answered (17*23 -> 391);
  - a tool call (`get_weather {"city": "Hanoi"}`, finish `tool_calls`) works;
  - a strict json_schema request returns valid JSON;
  - 4 concurrent streams completed (~14 s each, 0 errors);
  - `prod-switch.sh status`: only the dsv41 watchdog / re-arm / boot-start are enabled; GLM's three are disabled.
- **Reboot survival not yet tested.** dsv41-boot-start is enabled. A test reboot takes prod down ~5-8 min, so it was
  left for the user's go-ahead.

## Ship gates at the pinned config

- **Run on e87e813 + G7 knobs** (`G7-RESULTS.md` section 5): replay quality (top-1 0.9958, MMLU 88.5 / 88.0 / kit
  87.5, needles to 299K).
- **Run on 0933b77 + Engram gate:** structured (22 cases), chains 11 / 12 x 2, tool-eval-bench C 8 / 8, stress.
- **Stress.** Every decode stream reached 2,048 tokens. Worker MemAvailable min **4.42 GiB** (gate 5): an ~11 s
  transient at the 299K prefill start; also 4.43 at CONTEXT 262144, so not the KV pool. CONTEXT stays 300000. Above
  the 4 GiB hard floor.
- **Not run:** the soak (skipped for the switch). The G8 decode levers (row graphs, plan link, spec draft) changed
  only decode scheduling. Every structural test and every exact run passed, and the post-switch checks covered tools,
  json_schema and 4 streams.

## Open after G8

- The 5 GiB stress transient (worker 4.42).
- The soak and a reboot-survival check on the prod config.
- Tree verification (`G8.sh prose`): needs a window (prod holds the GPUs).
- The fast-prefill segmentation dependence (G7).
- Dense x3dn slower than x3seg on GB10.
- `/health` reports `drafted_total` / `accepted_total` 0 under DSpark (the counters are not wired for dsv41).
