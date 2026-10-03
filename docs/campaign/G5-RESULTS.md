# G5: prefill adoption on both Sparks (2026-10-02, 09:03-10:40 +07, inside G4's held window)

The user asked for every prefill (PP) change to become the baseline. G5 ran inside G4's window after G4's
kitgreedy / forced steps, with sources restaged at `dfc5684` (the offline prefill commits `e499fef..c81a8e9` plus
G4's fixes and the default flip) and `prefill_mm.py` from `5f74397`. G4's 128-row stress / cold boot were skipped
as directed, and the stress and cold boot ran here with the new defaults (G4-RESULTS.md section 5). Raw files:
`results/G5-20261002/` (`m2-pf-*.json`, `pf-*-r0.log`, `tests-prefill.log`, `nsys-nsys-allon*`, `g5-gate-*.json`,
`summary-prefill.txt`, `runner.log`).

## Adoption gate: tests

`G5-prefill.sh step tests`: **51 passed** on the GPU. These are the big segments vs 128-row ones, streaming top-k,
fast-tag row independence with grouped / tc experts, x3tc decode / routed / PTX vs plain, prefill GEMM column blocks,
CED replay, the native Engram reader, plus the CPU suites. Fast vs exact logits: rel max 0.0098 (grouped) / 0.0074
(tc), argmax agreement 1.00. Nothing crashed and nothing was wrong, so every piece is adopted.

## Defaults now (branch + dsv41 `config/prod.env`), each with its off switch

| knob | default | off |
| --- | --- | --- |
| `TF_DSV41_PREFILL` | **replay** (CED bounded replay) | `full` |
| `TF_DSV41_PREFILL_CHUNK` | 2048 | `128` (the G2-G4 path) |
| `TF_DSV41_PREFILL_KERNELS` | **fast** | `exact` |
| `TF_DSV41_FAST_EXPERTS` | **grouped** (tc measured 3-5x slower, below) | `tc` to try x3tc |
| `TF_DSV41_ENGRAM_NATIVE` | 1 | `0` |
| `TF_DSV41_INDEX_STREAM_MIN` | **16384** | `0` |
| `TF_DSV41_PLAIN_TRITON` (G4, decode) | 1 | `0` |

The x3tc decision deviates from "turn everything on": tc is correct (tests, gate), but on the real weights it made
prefill 3-5x slower (it was on in the first all-on run: 286 / 183 / 189 tok/s at 8K / 32K / 64K). Its knob stays.

## Cold prefill (tok/s, TTFT in s; one slot, fresh random prompt, after a 2K warm-up)

| config | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: |
| G4 old path (128-row exact, x3pf off) | 232 (35.4) | 233 (140.7) | 233 (281.5) | 228 (574.5) |
| exact: 2,048-row segments, exact kernels, full prefill | 414 (19.8) | 419 (78.2) | | |
| fast: + fast tag, grouped experts (full prefill) | 478 (17.1) | 483 (67.8) | | |
| **all on (new defaults: + replay + streaming top-k)** | **931 (8.8)** | **965 (34.0)** | **965 (67.9)** | **949 (138.2)** |
| all on, 4,096-row segments | 925 (8.9) | 965 (34.0) | | |
| all on, streaming top-k off | | | 948 (69.2) | 928 (141.2) |
| all on with tc experts (the first all-on run) | 286 (28.7) | 183 (179.5) | 189 (346.5) | |
| no replay, tc experts | 174 (47.0) | 177 (185.1) | | |
| kit (vLLM, BASELINE) | 1,073 | 1,075 | 1,060 | 1,031 |
| target | full 1,500 / replay 2,200 | | | |

Each piece's share at 32K:

- 2,048-row segments: 1.8x (233 -> 419).
- Fast tag: 1.15x (-> 483).
- Replay: 2.0x (-> 965).
- Streaming top-k: +2% at 64K / 128K.
- 4K segments: nothing.

The all-on config reaches **0.90x the kit**, and stays flat to 128K. A 300K prompt now takes about 5.3 min, against
~22 min before.

## Quality (G5 gate: teacher-forced top-1 vs the kit's oracle, 8 prompts)

| prefill kernels | top-1 whole | first copy |
| --- | ---: | ---: |
| exact (G4 cgate) | 99.62% | 96.35% |
| fast, grouped experts | **99.61%** | **97.34%** |
| fast, tc experts | 99.63% | 96.35% |

The fast tag shows no measurable quality loss. **Replay was not measured against the kit**: the gate's
teacher-forced windows do not go through the serving path's replay. MMLU-200 did not fit the window. Decode
acceptance after a replayed / fast prompt is a little lower: code 3.80 vs 3.92 tok/round, prose 1.66 vs 1.77
(G4-RESULTS.md section 2). That is the only quality signal on replay so far, and it should be measured next.

## Where the time goes now (nsys of one 32K prefill, all on: 35.2 s, 930 tok/s under nsys)

GPU kernels sum to ~32.9 s, so the GPU is ~93% busy and prefill is GPU-bound. The `share` phase (8.5 s) is rank 0
waiting on its own queued GPU work at the plan exchange, not idle time.

- routed experts 17.0 s (52%): still the decode kernel `x3ld ld_kernel` (9.4 ms a call at 2,048 rows), with group /
  gate-up / down epilogues. The kit uses GEMM-style expert kernels here. **This is the next lever**: x3tc was meant to
  be it, but it is slower.
- CSA2 attention / indexer 4.1 s (12%)
- fast prefill GEMMs (`_gemm`) 2.3 s (7%)
- mHC `_site` 2.3 s (7%)
- router `_logits` 1.9 s (6%; 4.3 ms a call at 2,048 rows)
- NCCL 0.8 s
- `weights_proj` (`_plain`, G4's Triton fp64 kernel) 0.26 s. At 2,048 rows it takes 3.6 ms a call, worse than cuBLAS
  at that width. Its share is under 1%; a prefill-width variant is a small follow-up.

The Engram wait fell from 30.6 s (19%) to 3.2-3.7 s at 32K (native reader + bulk prefetch).

## Memory (0.5 s samplers)

The single-slot prefill runs bottomed out at head 7.7-9.3 / worker 6.0-7.7 GiB MemAvailable (128K all on: 8.35 /
6.36). The 4 x 300K stress with every change on: worker 4.38 GiB (transient), steady 5.3-5.8; head 5.46
(G4-RESULTS.md section 5).
