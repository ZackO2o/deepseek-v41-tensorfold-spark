# G6: router, x3gm routed-expert prefill, speed table, ship gates, prod switch (2026-10-02, 11:15- +07)

One held campaign window (`campaign.sh open` 11:15:24; worker rebooted 11:15-11:17, then head 11:17), GLM prod
down for the whole window (direction: GLM downtime does not matter). Branch `dsv41-060`, staged several times
(each restage logged in `window.log`): `5d4b963` (window start) -> `8eef66e` (router tooling, x3gm test fix) ->
`d9a95e6` -> `3ac044c` (gm default) -> `c751bc1` (the `--alias` fix; the ship section and the speed table ran here).
Raw files: `results/G6-20261002/` (`ship/`: the ship section; `ship-try1/`, `ship-try2/`: the two aborted starts).
Every run: FP8 index keys, RoCE (1024 KiB), 0.5 s samplers on both nodes, the 3 GiB guard, `timeout` on every step,
test servers on :8001, caches dropped before every start (MemFree >= 104 GiB).

@@BOTTOM@@

## 1. Router (b0aa301, the fused split-K router) on the real weights

Tools added: `router_check.py` (`TF_DSV41_ROUTER_CHECK=report.json`: every eager fused call also runs `split` and
counts agreement, float64 adjudication of every differing row), `routerbench.py` (all 43 real routers: 40 target
layers E = 384 top-6, 3 DSpark blocks E = 128 top-3), `tests/cuda/test_dsv41_router_gpu.py`, `G6.sh router`.

| check | result |
| --- | --- |
| picks fused vs split, real rows (the M1 gate run, 8 prompts x 4K, every router call: 10,280 calls) | **99.9984% identical order, 99.9994% identical set** of 672,560 rows; the 11 differing rows are all float64 near-ties (margin 8e-8 .. 7e-7); fused == float64 on 7, split on 4 |
| picks, random rows (88,064 over 43 routers) | 99.9989% order, 100% set; 1 differing row (fused == float64) |
| max logit difference fused vs split | 2.1e-4 (real), 8e-5 (random); weights 1.6e-5 |
| row invariance R = 1..16, tile offsets 0 / 7 / N - R, 43 routers | 17,544 row checks, **0 mismatches**, picks + weights bit for bit |
| chunks (4,100 rows = 3 launches) == 16-row windows | bit for bit |
| CUDA graphs (R = 1 / 3 / 16, one layer and the 43-router sweep; capture-first) | replays twice == eager bits, **counters back to 0 after every replay**, new inputs == eager: no stale partials |
| GPU tests (router, x3gm, fast tag end to end) | 20 passed |
| M1 top-1 vs the kit oracle (fast kernels, grouped experts) | fused **99.58% / 96.35%** first copy; split 99.61% / 97.34% (= G5 exactly) |
| time a layer (routerbench, graphed 43-router sweep; nsys agrees: `_fused` median 107 us vs `_logits` 100 + `_select` 11.5) | R = 1-16: **fused 102-104 us vs split 112-115 us (1.09x, ~0.4 ms a verify window)**; R = 64: 362 vs 193 us (0.53x); R = 2,048: 10.9 vs 5.6 ms (0.51x) |

**Verdict.** The fused router is correct by construction on the real weights: no stale partials, counters clean,
invariant, graph-safe. It is not the 4x the cost model expected: ~25 us was predicted, 102 us measured (1.09x split),
so it saves ~0.4 ms of the 4.8 ms router share of a verify window, not 3.7 ms. At prefill widths it is 2x slower.
Its top-1 moved 99.61 -> 99.58% (first copy 97.34 -> 96.35%): near-tie flips. Under the original plan's rule that
made `split` the default (`d9a95e6`). After the user's decision (TARGETS.md: top-1 >= 96% vs the kit, judge on
speed), **`8f24eac`: `fused` for decode / verify windows and exact prefill (they must share bits), `split` for fast
prefill runs** (`TF_DSV41_PREFILL_ROUTER`, their own session tag). The G7 `gemv` router (another track's) is the
real lever here; G6's prefill nsys puts the router at 9% of a 32K prefill (`_logits` 4.3 ms a 2,048-row call).

## 2. x3gm, the CUDA routed-expert prefill (`TF_DSV41_FAST_EXPERTS=gm`)

### Tests

- First run (`5d4b963`): 6 of 46 failed. Neither was an x3gm fault:
  - 3 x `test_routed_vs_float64_and_grouped`: gm vs float64 3e-4 (pass), gm vs grouped rel ~1. Diagnosed on the
    GPU: the test's skewed picks could name expert 2 twice in a row; upstream's grouping keeps at most R members an
    expert, so the extra members of the last rows were dropped from the grouped reference (the tail ~20% of rows
    wrong; all-rows-same-6-experts at 512 rows is exact). A real router never repeats an expert in a row. Test fixed
    (`8eef66e`): gm vs grouped 4.9e-5.
  - 3 x `test_fast_prefill_row_independent_and_close`: the fused router refused D = 128 (the synthetic checkpoint).
    `ac52383`: fused falls back to split where 512 does not divide D.
- After the fixes: **20 / 20** (router + x3gm + fast tag e2e with grouped / tc / gm), then **38 / 38** at `3ac044c`
  (+ fused_proj GPU, prefill_fast CPU suite), **25 / 25** draft + decode glue GPU suites at `c751bc1`.

### kbench (gmbench, rank 0's half, real router picks on random rows)

| layer | rows | floor ms | grouped ms | tc ms | **gm ms** | gm x floor | gm TFLOP/s | gm / grouped | rel vs grouped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 (3-bit) | 512 | 12.0 | 17.2 | 129.8 | **12.0** | 1.00 | 9.1 | 0.70 | 7e-5 |
| 3 | 1,024 | 13.2 | 29.8 | 160.2 | **13.9** | 1.05 | 15.7 | 0.47 | 6e-5 |
| 3 | 2,048 | 15.5 | 60.0 | 244.3 | **18.1** | 1.16 | 24.0 | **0.30** | 9e-5 |
| 3 | 4,096 | 20.2 | 120.0 | 484.3 | **35.3** | 1.74 | 24.6 | 0.29 | 8e-5 |
| 20 (2-bit) | 512 | 8.4 | 15.3 | 159.7 | **11.0** | 1.30 | 9.9 | 0.72 | 6e-5 |
| 20 | 1,024 | 9.6 | 26.2 | 182.9 | **12.9** | 1.35 | 16.8 | 0.49 | 3e-5 |
| 20 | 2,048 | 11.9 | 52.7 | 261.1 | **16.1** | 1.35 | 26.9 | **0.31** | 3e-5 |
| 20 | 4,096 | 16.6 | 105.3 | 516.0 | **31.9** | 1.92 | 27.3 | 0.30 | 3e-5 |

Rows independent, repeats and all 8 configurations (gu / dn / ticket) bit for bit; configurations within 31.9-41.6 ms
at 4,096 (no reason to change the default). The cost model's 18-23 ms at 2,048 rows was right (18.1 / 16.1).

### Cold prefill (one slot, fresh random prompt, after a 2K warm-up; tok/s, TTFT s)

| config | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: |
| G5 all on (grouped) | 931 (8.8) | 965 (34.0) | 965 (67.9) | 949 (138.2) |
| **G6 all on + gm (new default)** | **1,314 (6.2)** | **1,390 (23.6)** | **1,397 (46.9)** | **1,365 (96.1)** |
| all on + gm, 4,096-row segments | 1,314 (6.2) | 1,417 (23.1) | | |
| full (no replay), grouped | 480 (17.1) | 486 (67.4) | | |
| full (no replay), gm | 698 (11.7) | 716 (45.8) | | |
| G6 all on, grouped (re-measured, see below) | 930 (8.8) | 329 (99.6) | 226 (290.6) | 438 (299.3) |
| kit (vLLM, BASELINE) | 1,073 | 1,075 | 1,060 | 1,031 |

**gm: 1.41-1.44x G5, 1.22-1.32x the kit, flat to 128K.** Full prefill 1.47x (486 -> 716). 4,096-row segments add 2%
at 32K but cost 0.9 GiB of worker floor (5.40 vs 6.30 GiB): kept at 2,048. A 300K prompt now takes ~3.7 min.

The grouped all-on re-run in this window was anomalous (fine at 8K, then 4-5 s a segment at 32K-128K, against G5's
965 and this window's grouped *full* run at G5's exact numbers): not understood; grouped is no longer the default.
Listed under open items.

### Gate and nsys

- M1 top-1 vs the kit, fast kernels + gm: **99.59% / 95.35%** first copy (grouped 99.61% / 97.34%). Over the 96% line.
- nsys of one cold 32K prefill with gm (24.2 s under nsys, GPU kernels 21.0 s = 87%): routed experts (x3gm kernels
  4.3 s + rotation 0.4 + combine 0.4) **5.2 s (25%)**, down from 17.0 s (52%) in G5. Next items: CSA2 prefill
  attention `_chunks` + `_merge` 3.6 s (17%), prefill GEMMs `_gemm` 2.4 s (11%), mHC `_site` 2.1 s (10%), router
  `_logits` 1.9 s (9%), NCCL 1.1 s, copies 1.2 s; Engram wait 3.1 s of wall (13%; G7's prefetch-ahead).

**Verdict: adopted** (`3ac044c`, prod.env `TF_DSV41_FAST_EXPERTS=gm`): every pass line of `G6.sh` held (tests,
kbench <= 0.5x grouped and <= 1.6x floor at 2,048, allon-gm >= 1,200 at 32K, full-gm >= 560, 128K within 15% of
32K, gate, memory >= 4 GiB). The fast tag becomes 48 (grouped-era fast snapshots miss once).

@@SPEED@@

@@SHIP@@

## Memory floors (MemAvailable minimum, GiB; 0.5 s samplers)

@@MEMORY@@

## Fixes made (dsv41-060 unless stated)

| commit | what |
| --- | --- |
| `ac52383` | fused router falls back to split where 512 does not divide D (synthetic GPU checkpoints raised) |
| `8c1c348`, `d9a95e6` | `router_check.py` (fused vs split on a run's real rows), `routerbench.py`, GPU router tests |
| `8eef66e` | x3gm GPU test: skewed picks keep distinct experts a row (the grouped reference dropped duplicate members) |
| `d9a95e6` -> `8f24eac` | router default: split (original rule) -> fused for decode rows, split for fast prefill (user's speed rule) |
| `3ac044c` | **x3gm routed experts by default** (1.44x prefill) |
| `c751bc1` | **`--alias` accepted by the DeepSeek app and listed in /v1/models** (the prod recipe could not start: the GLM server refuses `--alias`) |
| dsv41 `f8041f5` | `G6.sh router`, `G6.sh prebuild` (+ `prebuild_ext.py`), serve.sh compares images by content key (the nodes' docker stores give one image different ids), prod.env gm |

@@OPEN@@

@@HARNESS@@
