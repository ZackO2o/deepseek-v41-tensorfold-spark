# G1: the DeepSeek-V4.1 kernels on the GPU (2026-10-02, 02:46-03:04)

The first GPU window of the family on branch `dsv41-060` of TensorFold. It ran as the first half of one campaign
with G2 (and then G3), with GLM prod down from 02:42:59 for the whole campaign (`scripts/windows/campaign.sh`).
Raw files are in [`results/G1-20261002/`](../../results/campaign/G1-20261002/), and the campaign's samplers and log are in
[`results/campaign-20261002/`](../../results/campaign/campaign-20261002/).

**Bottom line.**

- Every kernel is bit-exact on sm_121, once 4 bugs that only the GPU shows were fixed (2 shared-memory overflows, 1
  signed-zero bug, 1 `python -m` packaging bug). Fixes are on the branch, each with a CPU test or a GPU re-check.
- `x3ld` passes the 0580 gate: bitwise == upstream at every width and setting, and probe 3 is 255.5 GB/s against
  the 220 bar. In the real layers it is +5-15% over upstream's grouped kernel at R = 2-16.
- `x3pf` is bitwise == upstream at 16-2,048 rows but no faster at the forward's 128-row chunks. It stays off.
- The prepared folders are written: 101.8 GB a rank, in about 80 s a node.

## Harness

| | |
| --- | --- |
| Nodes | Both rebooted after GLM prod stopped: worker first (back in 79 s), then head (back in ~50 s). The lease was kept fresh, and boot-start stood down. GPU clocks 2,223 MHz under load |
| Lease / watchdog / deadman | Lease `the window lease file` with a refresher every 4 min. `glm53-tf-watchdog.timer` stopped (and stopped again after the head reboot). Deadman restoring GLM prod at +6 h (08:42), re-armed after the reboot, extended to 09:42 before G3 |
| Samplers | 0.5 s MemFree / MemAvailable on both nodes, from 02:46 (campaign dir `mem-r{0,1}.log`) |
| Sources | `dsv41-060` staged as `git archive` on both nodes. The final G1 state is `eeb59f0`. The ranks run GLM prod's image `glm53-tensorfold:b13-060` with the branch on `PYTHONPATH` |
| Image gap | The prod image has no pytest. It is installed with `pip --target` into the `dsv41-tf-cache` volume (`/cache/pylib` on `PYTHONPATH`; its `packaging` copy is removed so the image's own is used) |

## Bugs the GPU found (fixed on `dsv41-060`)

| Commit | Kernel | What sm_121 did | Fix | Check |
| --- | --- | --- | --- | --- |
| `bbfb71d` | router `_route` | `OutOfResources`: 132 KiB of shared memory against the 99 KiB limit. Triton's default 3 stages of the [32, 512] fp32 weight tile | `num_stages=1`. Stages only overlap loads, so the FMA chains are unchanged | GPU router tests (float64 picks, row invariance) |
| `bbfb71d` | `fastboot` under `python -m` | `ModuleNotFoundError: __main__.weights`: the key named `__main__`'s modules | `PKG` from `__spec__.parent` | CPU test: `-m ... check` names the engine's folder (fails before the fix) |
| `36bf477` | CSA2 attention `_chunks` | `OutOfResources`: 112 KiB of shared memory | `num_stages=2` (prefetch only) | GPU: rows alone == in windows; the forward == the CPU twin; row invariance |
| `36bf477` | Engram `e4m3_bits` | byte 0x80 (e4m3 -0) came out +0.0: 72 of 15,360 values. Triton lowers `-x` to `0 - x` | The sign is set as a bit (`\| s << 31`) | GPU bitwise test now passes; Engram / CSA2 interpreter suites 22 passed |
| `eeb59f0` | `kbench --prefill` | upstream's `group_kernel` failed to launch at 1,024 rows inside a 2,048-row scratch (invalid argument). At 2,048 rows x 7 picks its one block needs 56 KiB of dynamic shared memory (48 KiB limit) | A scratch for each width; the real-layer bench stops at 1,024 rows. The GPU unit test groups 2,048-row windows in torch (checked == upstream's grouping at 1,024) | |

Upstream limit to remember: `tensorfold/cuda/exl3/experts.cu` `group_kernel` cannot launch above 1,755 rows at 7
slots. This does not matter for M1, whose prefill windows are 128 rows, but a 2,048-row prefill chunk would need
`cudaFuncSetAttribute` (or our own grouping).

## Tests on sm_121 (final: `tests-final.log`)

`tests/cuda/test_dsv41_gpu.py` + `tests/cuda/test_dsv41_blockers_gpu.py` + the family's CPU suites (`test_dsv41_forward`,
`_weights`, `_fastboot`, `_engine`, `_slots`): **67 passed** (152 s). The first run had 7 failed and 18 passed; the
fixes above account for all of them.

| Check | Result |
| --- | --- |
| x3ld == upstream bit for bit, random mul1 experts at the real geometry (3-bit + 5-bit shared, 2 + 4, 4-bit), R = 1-16, all 3 settings | pass |
| Router: picks == float64, weights within 1e-5, a row alone == in a window | pass (also the split router, see G2) |
| CSA2: rows' FP8 records == the reference, attention rows alone == in windows | pass |
| The whole forward on the GPU (synthetic checkpoint at the kernels' shapes) == the CPU twin, rows invariant across windows 45 / 8 / 1 | pass |
| Prepared folder read to the GPU == the tree | pass |
| **mHC** boundary (D = 5,120, world 1 / 2): streams, collapse, taps, partials, normed input == the torch emulation bit for bit, coefficients within 1e-5, a row alone == in a window of 19 | pass |
| **Engram** dequant bit for bit (every e4m3 byte, zero / subnormal-product / huge scales, no flush to zero); fusion update bit for bit, gates within 1e-6; row invariance | pass after `36bf477` |
| **stream_topk** == materialised `index.select` / `reindex` / `candidate_blocks` at 3K / 20K / 64K keys | pass. Single calls with 8 rows: 0.34 / 0.37 / 0.47 ms materialised vs 3.2 / 1.4 / 0.58 ms streaming (the first two include compiles). The streaming path is for >= 300K, where materialised scores do not fit |
| **DSpark chain** drafts vs the host `exact_sampling` over 64 seeds x 5 positions | **0 of 320 differ**; slot alone == in a batch |
| **x3pf** Z == upstream's grouped kernel bit for bit, widths 2 / 3 / 4-bit, 16 / 256 / 1,024 / 2,048 rows, both settings | pass |

## kbench: x3ld vs upstream on real mul1 experts (rank 0's half, router picks on random activations)

GB/s of trellis bytes read: the distinct routed experts plus the shared entry. Probe 3 is the gate/up launch through
x3ld's load path alone (no decode, no mma). The 0580 gate is >= 220 GB/s.

| Layer | R | experts | upstream | x3ld 8,1 | x3ld 8,2 | x3ld 4,2 | probe 3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 (3-bit, k2 6-10) | 1 | 6 | 198.9 | 217.1 | 208.5 | 214.7 | 243.2 |
| 3 | 2 | 12 | 210.0 | 220.2 | 220.2 | 220.9 | 236.9 |
| 3 | 4 | 24 | 211.6 | 231.9 | 239.2 | 233.9 | 245.4 |
| 3 | 8 | 44 | 220.5 | 239.1 | 243.3 | 242.3 | 255.5 |
| 3 | 16 | 81 | 219.0 | 242.3 | **245.0** | 242.8 | 252.2 |
| 20 (2-bit, k2 4-8) | 1 | 6 | 187.1 | 180.0 | 184.0 | 202.5 | 514.6 (*) |
| 20 | 2 | 12 | 193.0 | 202.9 | 201.8 | 202.1 | 238.5 |
| 20 | 4 | 23 | 197.1 | 213.0 | 215.1 | 227.2 | 233.3 |
| 20 | 8 | 44 | 201.5 | 220.2 | 231.7 | 232.1 | 251.0 |
| 20 | 16 | 82 | 205.0 | 227.2 | **235.3** | 234.2 | 246.2 |

All 42 x3ld cells are bitwise == upstream. **Probe 3 best: 255.5 GB/s (layer 3) and 251.0 (layer 20, R >= 2): gate
PASS.** (*) Layer 20 at R = 1 reads 7 small 2-bit entries, about 13 MB, which stay in L2 across the timed
repeats, so that cell is not a DRAM figure. x3ld is +4-12% over upstream at R >= 2 (8,2 is the best setting at
R >= 4). The worst cell is layer 20 at R = 1 with 8,1, at -4%. G2 / G3 run with `TF_DSV41_EXPERT_LOADS=1`
(default setting 8,1).

## x3pf (prefill experts) vs upstream on the real layers (ms for one `routed` call, rank 0's half)

| Layer | rows | upstream | x3pf 4,4 | x3pf 8,2 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 16 | 2.57 | 3.96 (0.65x) | 2.76 (0.93x) |
| 3 | 128 | 8.84 | 12.06 (0.73x) | 9.17 (0.96x) |
| 3 | 256 | 12.75 | 15.83 (0.81x) | 12.75 (1.00x) |
| 3 | 512 | 21.90 | 21.26 (1.03x) | 20.35 (1.08x) |
| 3 | 768 | 31.92 | 30.23 (1.06x) | 29.51 (1.08x) |
| 3 | 1,024 | 39.20 | 43.70 (0.90x) | 38.04 (1.03x) |
| 20 | 16 | 1.81 | 3.74 (0.49x) | 2.20 (0.83x) |
| 20 | 128 | 7.27 | 12.82 (0.57x) | 8.38 (0.87x) |
| 20 | 512 | 19.14 | 22.80 (0.84x) | 18.85 (1.02x) |
| 20 | 1,024 | 36.95 | 41.33 (0.89x) | 37.25 (0.99x) |

Every cell is bitwise. x3pf wins only at 512-768 rows (+2-8%). It loses at the forward's 128-row windows.
`TF_DSV41_EXPERT_PREFILL` stays 0. In the unit test's random uniform picks over 64 experts (about 190 members an
expert at 2,048 rows), it was 1.3-1.9x faster. The real router gives about 32 members an expert at 1,024 rows, too
few to reuse a decoded tile well. A wider prefill chunk with a fixed grouping limit is where x3pf could pay off.

## Prepared folders

| Rank | Node | Size | Build from the checkpoint | Write | Folder |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | worker | 101.8 GB | 215 s | 80.6 s | `~/dsv41-prepared/model/9529c28bc20acb64/rank1` |
| 0 | head | 101.8 GB | ~220 s | 80.1 s | `~/dsv41-prepared/model/3c52fb1b8d6c90e6/rank0` |

Free disk before: head 1,142 GiB, worker 2,442 GiB (needed about 95 GiB a node). G2 read them at 8.7-9.2 GB/s
(O_DIRECT, 8 threads): 11-12 s for the weights.

## Memory during G1 (0.5 s samplers)

MemAvailable minimum: head 16.6 GiB, worker 15.3 GiB. Those minima came from the prepare runs, which hold the
whole tree on the GPU.
