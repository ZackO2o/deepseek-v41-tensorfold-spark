# G3: M2 decode with DSpark on both Sparks (2026-10-02, 03:50-05:09 +07)

The third window of the campaign, run straight after G2 in the same held window (lead's request; see
"Why G3 ran" below). M2 is the other track's work: commits `b0a6a79`, `adf1db6`, `d9d57ad` on `dsv41-060` and
`scripts/windows/G3-m2.sh` (`87d6643`). The sources were staged at `fc915ab`. Raw files are in
[`results/G3-20261002/`](../../results/campaign/G3-20261002/): `m2-eager.json`, `m2-kit.json`, the logs and `summary.txt`.

**Bottom line.**

- With CUDA graphs off, M2 is **exact on the real weights**: drafted == serial on every workload, at T = 0 and
  T = 0.7.
- DSpark acceptance is **above the kit's**: 3.10 tokens a round with kit-like drafting (static k = 3; the kit
  measured 2.27), and 3.48 with M2's cost-derived depth, confidence head and lookup.
- **Single-stream speed:** code 41.9 tok/s is the kit's level (the kit measured 41.9 at T = 0), against the 52
  target. Prose 22.4 is below the kit (25.1 anchor, 19.2-32.5 measured). The 4-stream total of 16.0 is far below
  the kit's C4 of 37.6.
- **CUDA graphs are broken on the real model.** The first replays crash with an illegal memory access, or produce
  garbage draft ids that crash the Engram hash. The synthetic graph-replay test fails too.

## Why G3 ran although M1 gate 4 failed

The lead asked for G3 in the same window if the M1 gates passed. The correctness gates passed: top-1 99.63%
and row invariance. Gate 4 (serial decode 23 tok/s) failed in M1's eager one-slot path. M2's serving path, with
Engram prefetch and graphs, is the planned fix for exactly that gap, so G3 is where it gets measured. The user's
standing rule is results over uptime. The deadman was extended by 1 h to 09:42 (allowed up to +3.5 h).

## Fixes made during G3 (on `dsv41-060` and in this repo)

| Commit | What |
| --- | --- |
| `fc915ab` | The router's logits scratch is allocated each call. The shared, grown buffer of `ccffe64` could be reallocated after a graph captured its address |
| `fc915ab` | `tests/cuda/test_dsv41_m2_gpu.py`: the module-scoped `serial` fixture ran before the function-scoped `TF_DSV41_SEAMS=torch`. Serial replies came from the mHC / Engram kernels and drafted replies from the torch seams, so they differed at token 12 of job 0. That was 3 of the 4 GPU failures (drafted == serial [1] and [4], graphs on == off) |
| `6c4a69a` | The drafter's attention had the same per-head q norm as the target (G2's bug) |
| `42783cb` | `fastboot.prune` removed the other kind's folder: the drafter's folder pruned the weights' folder and the next boot pruned the drafter's. **Every M2 boot rebuilt 101.8 GB from the checkpoint (301 s).** A CPU test covers it. The fix landed after the benches, so the boots below are about 320-337 s; with it, they should be about 30 s plus warm-up |
| dsv41 `b2d2379` (`G3-m2.sh`) | `bench`: `local tag=$1 ... wout=.../$tag` in one `local` statement is unbound under `set -u` and stopped the first kit step |

## GPU tests (`tests.log`, then the reruns)

- First run: 4 failed, 24 passed.
- After the fixture fix (`tests-graphsfix.log`): **4 of 5 M2 GPU tests pass**. These are drafted == serial at 1
  and 4 slots (past 2,048 tokens, T = 0 / T > 0, lookup mixed in) and graphs on == off replies.
- **`test_graph_replay_equals_eager` still fails**: at step 0 (slot 0, 8 rows after a ~2K prefill), the *eager*
  forward's logits are NaN when a second forward has captured graphs in the same process.
- Separate scripts, in `results/G2-20261002/reference-diag/m2*.py`, isolate it:
  - Paged row invariance on the GPU holds: an 8-row window == 8 single rows at 100 / 300 / 2,030 tokens.
  - Each forward alone has no NaN.
  - Drafted == serial holds alone, with and without lookup.
  - Graph-vs-eager equality depends on the order of the prefills: equal when g is prefilled first, unequal when
    e is. That points at shared state between graph-captured and eager runs, not at the kernels.

## Benches on the real weights (m2bench, both ranks; 4 slots x 16,448 tokens; TF_DSV41_EXPERT_LOADS=1)

DSpark tokens a round = 1 + accepted drafts a verify. "serial" is the same workload with drafting off, through the
same serving path (eager).

| Workload | kit-like: static k = 3, no lookup, graphs off | M2 default: cost depth (cap 5) + confidence + lookup, graphs off | kit (BASELINE) |
| --- | --- | --- | --- |
| code, T = 0 | 33.7 tok/s, 3.25 tok/round (serial 18.8) | **41.9 tok/s, 4.04** (serial 19.1) | 41.9 (T = 0), 40.6 (T = 1); target 52 |
| prose, T = 0 | 20.0, 1.80 (serial 19.1) | **22.4, 1.68** (serial 19.0) | 19.2 short chat / 32.5 essay; target 38 |
| tweet (primes / code / JSON) | 42.1, 3.86 | **55.5, 5.66** (lookup 40 / 51 kept) | 47-50; ~3.08 tok/round |
| edit (copy-heavy) | 41.9, 3.98 | **61.6, 7.01** (lookup 259 / 264) | 55.1-56.2; ~3.84 tok/round |
| long context | 24.7, 2.34 | 26.7, 2.34 | |
| MMLU (1-letter answers) | 3.0 tok/round | 3.0 tok/round | 1.46 |
| **DSpark mixed tok/round** | **3.10** (target >= 3.1 at k = 3: met) | **3.48** | **2.27** |
| T = 0.7 (code / prose / edit) | 13.9 / 7.6 / 17.1 tok/s | 15.4 / 10.3 / 20.2 tok/s | |
| **4 streams, aggregate** | 14.1 tok/s | **16.0 tok/s** (5.6 / 4.3 / 6.1 / 6.9 a stream) | **37.6** (C4) |
| exact (drafted == serial, T = 0 and 0.7, every workload) | **True** | **True** | |

The verify costs measured at boot, with graphs off, were 53.9 / 70.6 / 88.8 / 107 / 125 / 191 ms at 1 / 2 / 4 / 6 / 8
/ 16 rows, plus a 5.5 ms draft pass. A verify row costs 9-18 ms beyond the first. That is why 4 streams total only
16 tok/s: 4 slots x (1 + k) rows, eager. The kit's 4-stream rounds are much cheaper. In the kit-mode boot, before
the graph crash, graphs gave a 1-row verify of **44.3 ms** (22.6 tok/s serial), against 53.9 ms eager.

**Fast mode (graphs on) could not be measured.** In the first kit run (graphs on by default), the drafter returned
token id 4,140,992 (vocabulary 129,280) and the Engram hasher raised `IndexError` in `prefetch.issue`. In the fast
run, the first window after boot hit `CUDA error: an illegal memory access` in `slots.candidates`. Both only
happen with graphs. The prime suspect is a graph capturing device addresses that the serving layer later
reallocates or reuses. Candidates:

- page tables grown by `ensure()` after capture;
- the shared graph pool: replays of other keys overwrite the `pending` proj / taps buffers that `keep` keeps as
  views for the drafter. A clone of them after each replay was tried; it did not fix the synthetic test and was
  reverted.

This is the M2 owner's to fix next. G3 has its logs.

## Memory (0.5 s samplers, both nodes)

| Phase | head MemAvailable min | worker MemAvailable min |
| --- | ---: | ---: |
| First kit run, graphs on: warm-up + capture, then the crash (04:03-04:14) | **3.78 GiB** | **2.31 GiB** |
| eager bench, graphs off (04:15-04:40) | 7.74 | 6.09 |
| kit-like bench, graphs off (04:48-05:09) | 7.90 | 6.48 |

- With the drafter (+3.6 GiB a rank: 99.0 GiB allocated after boot) and the 4 x 16K pool, the graphs-off floor is
  6.1 GiB on the worker.
- **With graphs, the worker fell to 2.3 GiB, under the 4 GiB hard stop.** Graph pools need their own budget
  before graphs can be on in production.
- After a cold boot, the serving layer's admission waits for **MemFree** (the GLM rule). The checkpoint read and
  folder write leave about 7 GiB of page cache, so admission stalled 253 s until the caches were dropped by hand.
  It should drop its own page cache after writing a folder, as GLM's 0550 does.

## Compared with the targets

| | Measured (best mode) | Kit | Target (round 1) |
| --- | ---: | ---: | ---: |
| Code c1 | 41.9 | 41.9 | 52 |
| Prose c1 | 22.4 | 19.2 / 32.5 | 38 |
| C4 aggregate | 16.0 | 37.6 | 120 (M4) |
| DSpark tok/round, k = 3 | 3.10 | 2.27 | >= 3.1 |

The acceptance gate is met. Round-1 speed is not, and the 4-stream path is the largest gap. That gap is verify
cost per row, eager, before graphs.
