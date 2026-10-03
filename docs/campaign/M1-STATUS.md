# M1 status: the DeepSeek-V4.1-Flash family on TensorFold (2026-10-02)

**G7 + G8 and G4 + G5 (2026-10-02) summarised in the next sections.** **Updated after the G1 + G2 (+ G3) campaign of 2026-10-02 (02:43-05:18 +07, GLM prod down 155 min):** see "M1 gates on the GPU" below,
[G1-RESULTS.md](G1-RESULTS.md), [G2-RESULTS.md](G2-RESULTS.md) and [G3-RESULTS.md](G3-RESULTS.md). The text from
"What is implemented" on is the offline state before the windows, apart from the blockers list.

## G9-G11 overnight (2026-10-02 19:30 -> 10-03 04:25; [G9](G9-RESULTS.md), [G10](G10-RESULTS.md) morning summary, [G11](G11-RESULTS.md))

Best exact config (`38f6500`): code 79-81.5, prose 41, structured 117, C1 / C2 / C4 82 / 67 / 93.5 (kit 41.9-45 /
32.5 / 38-50 / 32.2 / 46.7 / 37.6); 1-row window 26.9 ms. Top-1 0.9963, MMLU 87.5, tools 11 / 12, TEB C 8 / 8, soak
pass. Stress memory 4.01 GiB (< 5): prod.env stays on `5ace28b` (G8) for the 08:00 restore. Not adopted: joint,
dense v2, PDL, the lossy verify budget, draft trees, DRAFT_K, the G11 drafter deltas (prose +5.5% but code -4.4%).

## G7 + G8 on the GPU (2026-10-02, 13:33- +07, after the storm outage; [G7-RESULTS.md](G7-RESULTS.md), [G8-RESULTS.md](G8-RESULTS.md))

| | result |
| --- | --- |
| Tests | every G7 suite on two GPU queues at once. 3 fixes: a stale router test (`24392ea`), BMQ 32 shared memory and the synthetic head count (`e87e813`). Open: the fast tag's ~1-ulp segmentation dependence (layer 3's q path, pre-existing) |
| Decode levers | **gemv router adopted** (24.6 vs 112.6 us a layer; 1-row window 33.1 -> 29.5 ms). Long projection plan, q4 / trim draft heads: not adopted. bf16 exchanges / mHC / norm: kept (prose +11%). Engram gate (G8): exact, C2 / C4 +3.5 / +2.6%, adopted |
| Speed (G7 adopted) | code 69.1, prose 37.4, structured 102.2, C1 / C2 / C4 70.0 / 48.9 / 67.0 (kit 41.9-45 / 32.5 / 38-50 / 32.2 / 46.7 / 37.6) |
| Prefill | levers 2 all4 + gemv prefill router: **1,833 / 2,043 / 2,068 / 1,953 tok/s** at 8K-128K (kit ~1,050; G6 1,314-1,397); 299K needle in 195 s |
| Rank skew | gone: busy time a window equal on both ranks (38.52 ms), kernels 1.001x; what is left is per-segment jitter (~1 ms a window of exchange waits) |
| Quality | top-1 0.9958 (exact kernels, gemv) / cgate 0.9958 with prod knobs; MMLU-200 replay 88.5 / full 88.0 / kit 87.5 |
| G8 (all on: Engram gate, row graphs, TCP plan link, speculative DSpark) | code 72.7, prose 39.0, structured 109.1, C1 / C2 / C4 69.1 / 58.9 / 77.5; dense x3dn slower (off); gate 0.9958 |
| Prod | **DeepSeek serves :8000 since 18:52** on `5ace28b` + G7 / G8 knobs (dsv41 `f058cbe`); GLM automation disabled, dsv41's enabled |

## G4 + G5 on the GPU (2026-10-02, 06:26-10:42 +07, GLM prod down 256 min; [G4-RESULTS.md](G4-RESULTS.md), [G5-RESULTS.md](G5-RESULTS.md))

| | G4 / G5 result |
| --- | --- |
| Gate 1 top-1 vs the kit | 99.62% whole, 96.35% first copy (exact prefill); fast prefill tag 99.61% / 97.34% |
| Gate 2 (kit recaptured with SPEC_METHOD=none) | 0-1 of 8 identical, every first divergence at a kit near-tie; forced top-1 **98.2%** after the router fix (86.3% before: NaN after a mid-sequence BOS, the "prompt-5 BOS loop") |
| Gate 3 / chunk | row invariance and 2,048-row chunk bitwise on both ranks |
| Gate 4 serial decode | 27.1 tok/s through the serving path with graphs (20.4 in M1's eager gate path) |
| Decode, graphs on, RoCE | code 57.8 (59.9 with G4's fixes), prose 31.3, structured 86.0, C1 / C2 / C4 60.5 / 46.1 / 70.7 (kit 41.9-45 / 32.5 / 38-50 / 32.2 / 46.7 / 37.6; targets code 52, prose 38) |
| Prefill | 128-row path 228-233 tok/s; **all G5 changes on: 931 / 965 / 965 / 949 tok/s at 8K / 32K / 64K / 128K** (kit ~1,050) |
| Memory | 4 x 300K stress with every prefill change on: worker 4.38 GiB transient, 5.3-5.8 steady; head 5.46. Cold boot: first token 1.2 s, no admission wait |
| Fixes | router softplus underflow (NaN), Triton fp64 `weights_proj` (-2.2 ms a window), NCCL float64, graph test, m2bench prefill mode; prefill changes on by default (x3tc off: 3-5x slower) |

## M1 gates on the GPU (G2, branch `dsv41-060` @ `fc915ab`)

| # | Gate | Result | |
| --- | --- | --- | --- |
| 1 | Teacher-forced top-1 >= 99% vs the kit's oracle | **99.63%** whole (16,376 positions); first copies 95.35% (287 / 301), 99.31% without the kit's own near-ties (top-2 within 0.125) | PASS (whole) |
| 2 | Greedy 256-token replies == the kit's on >= 6 of 8 | **1 of 8** identical; first divergences at 1-55 tokens (kit captured at the campaign's end, DSpark on); a mid-reply BOS loop on prompt 5 to check | **FAIL** |
| 3 | Row invariance on the real weights (128-row / 8-row / 1-row windows, both ranks) | bit for bit | PASS |
| 4 | Serial decode >= 23 tok/s | **19.4 tok/s** (eager one-slot path; 11.4 before the router split). M2's serving path: 19.1-19.3 serial eager, verify-1 44.3 ms with graphs (~22.6 tok/s) | **FAIL** |
| 5 | Boot from the prepared folder <= 90 s, <= 60 s | **18 s** (21 s from `docker run`) with the extension cache warm; 77-80 s on the first start after a restage (extension rebuild) | PASS |

Fixed on the GPU (`dsv41-060`): the per-head q norm V4.1 does not have (`9fd3764`, the reference `b778b1f`, the
drafter `6c4a69a`: top-1 92.3% -> 99.6%), the router as one program (`ccffe64`: 1.0 -> 0.1 ms a layer, same
bits), shared-memory overflows of the router and CSA2 attention on sm_121 (`bbfb71d`, `36bf477`), Engram's
e4m3 -0 (`36bf477`), `fastboot` under `python -m` (`bbfb71d`), router scratch made graph-safe (`fc915ab`).

Memory (0.5 s samplers): MemAvailable minimum head 6.24 / worker 5.23 GiB at 4K context, one slot; weights
95.4 GiB a rank; workspace ~3.0 GiB. **4 x 300K at 95 GiB a rank does not fit the 5 GiB floor** (worker ~2.9 GiB
projected, before M2's drafter and graphs). FP8 index keys: top-1 99.646% vs bf16 99.634% (no measurable cost).

 `dsv41-060` of the work tree `<engine checkout>` (from `glm-spark-stack-060`
`fbb904a`); family `src/tensorfold/families/deepseek_v41/`. Window scripts in this repo: `scripts/windows/`.

## Offline after G2 / G3: correctness and memory (2026-10-02, for G4)

TF `dsv41-060` bc57213..f2af03d, this repo 98bade4..d686696. G4's steps: `scripts/windows/G4-correct.sh`.

- **Gate 2 root causes found (numerics, not structure).** (1) The kit's logits are bf16 (its EXL3 head) and its greedy
  argmax takes the lower id on ties: every kit top-1 / top-2 margin is a multiple of 1/16, and at all 8 exact ties
  among the first-copy misses the kit's token was the lower id. We now round the head to bf16 and break ties by id
  (`pick.py`). (2) The kit's TP roundings: bf16 rank partials before its bf16 all-reduce, fp16 routed weights in its
  expert kernels (`TF_DSV41_KIT_ROUNDING`, default on). (3) The G2 kit capture drafted (k = 3) and kept only top-1
  logprobs: `tests/reference/gate2.py` recaptures with SPEC_METHOD=none and top-5, teacher-forces us along the kit's
  replies, and labels each divergence identical / tie / real.
- **Checked equal to vLLM, no change:** Engram n-gram hashing with BOS / EOS mid-sequence, chunk and prompt / reply
  lookback, token map (`test_dsv41_engram_hash_vllm.py`); SWA window (128 incl. the row); compressed-row visibility
  ((p + 1) // ratio) and their RoPE position (the group's first); YaRN (160K theta, factor 16 over 65,536, beta 32 / 1,
  mscale 1, GPT-J pairs on the last 64 dims) and plain RoPE on SWA / DSpark layers; the sink; no per-head q norm in
  the target, the drafter or the reference. The prompt-5 BOS loop is not the hash; G4's forced run shows whether it is
  a near-tie.
- **Admission stall fixed:** the 1 GiB immediately-free rule held admission at MemFree 0.84 GiB beside 6.7 GiB of
  clean page cache; now reclaimable cache counts (`TF_DSV41_ADMIT_FREE_FLOOR_GIB`, 0), the allocator cache is wired,
  and serve.sh drops caches before the canary too.
- **2,048-row chunks:** experts run in 1,024-row blocks (upstream's group kernel takes 1,750 at 7 picks).
- **Memory:** one shared expert scratch instead of 40 per-layer ones (2.68 -> 0.56 GiB; the old growth also broke
  graph captures' addresses), FP8 index keys (-0.35 GiB), host trim + memory snapshot at boot. Projection (ENGINE-PLAN
  5.0, with the perf track's shared graph pool and Engram cache): worker 5.00 GiB at 4 x 300K with FP8 keys (at the
  target, no margin), 4.29 if graphs cost the old 1.2 GiB; fallback 4 x 192K.

## Prefill (offline, 2026-10-02, for G5)

TF `dsv41-060` e499fef..a7a02a8; window script `scripts/windows/G5-prefill.sh` (+ `g5prefill_summary.py`). No GPU was
used: every number below is an estimate from G1 / G2 measurements until G5.

**Cost model of the old path** (one rank; the ranks run in parallel). G2: a 2,048-token prompt took 9.3 s =
4.5 ms a row (~220 tok/s). The prompt ran as 128-row forward runs (`forward.CHUNK`), so every 2,048 rows streamed all
40 layers 16 times and every kernel ran at verify-window sizes:

| Item (a prompt row, 40 layers) | 128-row runs | Why |
| --- | ---: | --- |
| routed + shared experts (upstream grouped, `x3ld`) | **~2.8 ms** (61%) | G1: 69 us a row a layer at 128 rows (38 at 1,024): the trellis is decoded again every 16 member rows |
| Engram reads (Python reader, 24 records a row a rank) | ~0.4-0.5 ms | ~20 us a record (0.52 s for 24.5K random rows on the workstation), on the critical path |
| host: ~2,500-3,000 launches + syncs a run | ~0.3 ms | 30-45 ms a 128-row run |
| dense EXL3 linears (`Exl3Linear`, 128-row blocks) | ~0.25 ms | 68M weights a layer a rank decoded every 16 rows (+ Engram `wkv` 157M on 2 layers) |
| attention, indexer, compressors, mHC, router, norms | ~0.4-0.6 ms | per-row work (sparse 512 + SWA 128; indexer scores grow with the position) |
| partial exchanges (fp32, 2 a layer) | ~0.1 ms | 2 x 20 KB a row a layer over RoCE |

**Changes** (each behind a knob; defaults keep prefill rows == decode rows):

| Commit | What | Knob | Exactness |
| --- | --- | --- | --- |
| `e499fef` | prefill segments of 2,048 rows (4,096 / 8,192): one staging SWA ring (next_pow2(C + 127) rows, same bytes), attention in scratch-sized row blocks, materialised indexer / candidates / reindex in row blocks under 256 MiB, optional streaming top-k, device-made window positions, pinned id copies | `TF_DSV41_PREFILL_CHUNK` (2048), `TF_DSV41_PREFILL_ROWS` (= chunk), `TF_DSV41_INDEX_BUDGET_MIB` (256), `TF_DSV41_INDEX_STREAM_MIN` (0 = off), `TF_DSV41_PREFILL_ATTN_ROWS` | exact tag: prefill rows == decode rows (tests: 36 / 8 / 6-row segments == 4-row == one row at a time, bit for bit; interpreter: staging + blocks + streaming == one window) |
| `c600c03` | the **fast prefill tag**: dense linears through upstream `exl3.prefill.matmul` (W_q decoded once a call, 128-column blocks under 64 MiB), the shared expert as three such GEMMs, routed experts on upstream's grouped kernel | `TF_DSV41_PREFILL_KERNELS=fast`, `TF_DSV41_PREFILL_WS_MIB` | GLM 0080 / 0085: row-independent kernels, so a fast prompt's state depends on its tokens only; fast rows != decode rows, so its own session tag (`Tag.grid` 16 / 32), prompt snapshots only (no turn snapshots), both ranks checked at boot |
| `238b5d0` | the **native Engram reader** (C threads + `pread`, O_DIRECT, GIL released; vectorized extents), bulk prefetch of a segment's rows, pinned H2D | `TF_DSV41_ENGRAM_THREADS` (32), `TF_DSV41_ENGRAM_NATIVE_ROWS` (1024), `TF_DSV41_NATIVE_DIR` | the file's bytes (== the Python reader); workstation: 0.09-0.14 s vs 0.52 s for 24.5K random rows |
| `9d1ac19` | **CED replay**: encoder pass (layers 0-19 + layer 20's compressor) over every prompt row, a stash of the rows entering the decoder (last 127 positions, in prompt snapshots), decoder replay over the tail with windows from R0 | `TF_DSV41_PREFILL=replay` | approximate by design, its own tag; n <= 128 == full bit for bit; never reads decoder rows before R0; resumed == fresh; drafted == serial |
| `fcc4668` | **x3tc**: Triton routed experts for fast runs: one program a (expert, 32 members, 128 columns), K whole, the mul1 trellis decoded in registers once for 32 members (dp4a by inline PTX), fused Hadamard / SwiGLU / rotation epilogues; buffers alias the expert scratch | `TF_DSV41_FAST_EXPERTS=tc`, `TF_DSV41_TC_CFG` (2,2,8), `TF_DSV41_TC_ROWS` (2048) | row-independent (tag 32); decode == `format.unpack` bit for bit; == float64 at 3e-4; sm_121a: 0 spills, <= 77 KiB shared |
| `927b1de` | prefill exchanges as bf16 (the kit-rounded partials are bf16 values): half the bytes | `TF_DSV41_PREFILL_EXCHANGE` (bf16) | same bits (TP=2 test) |
| `a7a02a8` | memory: the segment term scales with the chunk (0.39 / 0.71 / 1.36 GiB at 2K / 4K / 8K rows, fast) | | 8,192-row segments do not fit the 5 GiB floor at 4 x 300K |

**Expected ladder** (ms a row; ranges because only the expert kernel's rate at 128 / 1,024 rows was measured):

| Path | ms a row | tok/s | 300K prompt |
| --- | ---: | ---: | ---: |
| 128-row runs (G2) | 4.5 | ~220 | ~23 min |
| 2,048-row exact segments + native Engram + bf16 exchanges | 2.1-2.6 | **~380-480** | ~11-13 min |
| + fast tag (prefill GEMMs, dense shared expert) | 1.9-2.3 | ~430-530 | ~10-12 min |
| + x3tc routed experts (15-30 TFLOP/s at 32 members) | 0.9-1.4 | **~700-1,100** | ~5-7 min |
| + CED replay (half the layers a prompt row) | 0.5-0.8 | **~1,300-2,000** | ~2.5-4 min |

The experts dominate at every step until x3tc; past it the per-row attention / indexer / mHC work (~0.4-0.6 ms a
row, not touched here) is the next item, and G5's nsys of a 32K prefill splits it. 1,500 full-decoder and 2,200 replay
need x3tc at the top of its range plus that next item.

**Tests** (CPU): `test_dsv41_prefill.py` (8), `test_dsv41_prefill_interp.py` (2, interpreter), `test_dsv41_prefill_fast.py`
(8), `test_dsv41_engram_native.py` (6), `test_dsv41_replay.py` (11), `test_dsv41_x3tc_interp.py` (7, interpreter),
`test_dsv41_x3tc_compile.py` (1, sm_121a ptxas); the serving tests' replay turn-2 expectations updated (prompt
snapshots only). GPU (G5): `tests/cuda/test_dsv41_prefill_gpu.py`.

**G5 commands** (head, inside a held campaign window):

```bash
scripts/windows/G5-prefill.sh stage                       # workstation
scripts/windows/G5-prefill.sh check
WINDOW_HELD=1 scripts/windows/G5-prefill.sh step tests    # GPU + CPU suites of the prefill paths
WINDOW_HELD=1 scripts/windows/G5-prefill.sh step exact    # then fast, tc, tc4k, stream, replay, gate, nsys
WINDOW_HELD=1 scripts/windows/G5-prefill.sh step all      # all of them + summary-prefill.txt (~3.5-4 h)
```

## Prefill: routed experts (offline, 2026-10-02, for G6)

TF `dsv41-060` f4a162a..5d4b963 (`cuda/x3gm.{cu,cpp,py}`, `cuda/gmbench.py`); window script `scripts/windows/G6.sh
prefill-experts`. No GPU was used: the numbers below are a cost model until G6.

**Why x3tc was slow** (G5: 3-5x slower than grouped; ~230 ms a 2,048-row layer segment = 1.9 TFLOP/s and 11 GB/s of
trellis, so neither compute- nor bandwidth-bound): it was bound by its decode path's codegen and latency.

- Triton gives no fragment-level control: the decoded tile's 7-D reshape / permute and its conversion to the dot
  operand layout go through shared memory every k step, for both matrices. In CUDA, upstream's lane decode already is
  the mma fragment.
- Each value gathers its two trellis words with its own int64 address: two scalar loads a value, 16x upstream's one
  16-byte load a lane a tile. The loads are irregular, so the pipeliner does not make cp.async stages of them, and every
  k step waits on L2 / DRAM latency.
- One 8-warp CTA an SM hides little of that latency, and 32-row member blocks decode each weight 1.5-2x.
- sm_121 has no wgmma or TMA, so Triton has nothing to lean on: mma.sync is the only tensor-core path.

**Cost model** (one rank, 3-bit layer, 2,048 rows x top-6 of 384 = 12,288 pairs, ~32 members an expert):

| | per layer segment |
| --- | ---: |
| DRAM floor: trellis 2.55 GB once + X / Xd / Y traffic ~0.8 GB, at 235 GB/s | ~14 ms |
| MMA: 435 GFLOP at 110 TFLOP/s (`mma.sync` f16) | 4 ms |
| decode once: 6.8 G weights x ~6 instructions | ~3 ms (overlaps) |
| today: upstream grouped (1,024-row blocks, decode every 16 members, split-K Z + epilogue kernels) | ~45 ms |
| (a) decode to a bf16 scratch, then a grouped bf16 GEMM: 4 bytes of scratch traffic a weight vs 0.375 read | ~115 ms |
| **(b) x3gm**: decode into the mma fragment from a cp.async ring, 64-member passes | **~18-23 ms** expected |

(a) pays only above ~170 rows an expert (8K+ segments, which the memory floor does not allow at 4 x 300K). GLM 0080's
v1 (dequant once into shared memory) was shared-memory-bound. (b) is the GLM spark engine's `fat` kernel (measured
1.25-1.36x its DRAM floor at 2,048 rows) at this family's geometry.

**x3gm** (`TF_DSV41_FAST_EXPERTS=gm`, fast tag 48; default stays grouped until G6):

- Persistent CTAs claim items (expert pass of <= 64 members, one 128-column Hadamard block) by ticket.
- Per stage of a 3-4 deep cp.async ring: the members' rows (gathered by pair index, XOR-swizzled for ldmatrix) and
  the item's trellis words as stored.
- Each warp decodes 2 column tiles from shared memory straight into the m16n8k16 A fragment (W^T: upstream's lane
  layout is the A fragment), then runs it against up to 8 n8 member groups.
- Full K in one mma chain (no split-K, no Z). Epilogues (fwht128, ACT_F32 SwiGLU, the down input's rotation; svh_d) in
  the CTA. Y fp32 a pair, then upstream's `combine` in slot order.
- One 2,048-row block a segment. Xg / Xu, Xd and Y alias the shared expert scratch, so nothing new is allocated.
- mul1 at 2 / 3 / 4 bits; anything else falls back to the grouped kernels.
- Gate and up share one rotated input when their sign vectors are equal (checked once a layer).
- sm_121: 18 instances, 0 spills, 2 CTAs an SM at the default configurations.

**Expected:** -23 to -26 ms a layer segment. At 32K with replay (~340 layer segments) that is -8 to -9 s of 34 s:
**~965 -> ~1,250-1,330 tok/s** (1.2x the kit). Full prefill: ~483 -> ~600. The 2,200 replay target then needs the next
items: CSA2 attention / indexer (12%), the fast GEMMs, mHC and the router (~20% together), and the 8.5 s `share` wait.

**Tests:** `tests/test_dsv41_x3gm_emu.py` (24, CPU). The lane-level emulator `tests/dsv41_x3gm_emu.py` (every index of
the kernel; stale shared memory between items) shows:

- decode == `format.unpack` bit for bit;
- vs float64: 3e-4;
- every output stored once;
- row independence, any item order / CTA count, and every configuration bit for bit;
- the plan, knobs, tag and scratch aliasing.

The mutations it catches include fragment register order and ldmatrix order. Also `tests/kernels/test_x3gm_compile.py`
here (nvcc sm_121: 0 spills, registers x CTAs, shared memory, PTX, helpers upstream's). GPU (G6):
`tests/cuda/test_dsv41_x3gm_gpu.py`, plus gm in the fast-tag end-to-end test.

**G6 commands** (head, inside a held campaign window):

```bash
scripts/windows/G6.sh stage                                   # workstation
scripts/windows/G6.sh check
WINDOW_HELD=1 scripts/windows/G6.sh prefill-experts tests     # then kbench, perf, gate, nsys (cfg optional)
WINDOW_HELD=1 scripts/windows/G6.sh prefill-experts all       # all + summary-g6-prefill.txt (~2.5 h)
```

**Risks:**

- Untimed. GLM's in-engine transfer of isolated kernel wins has been a half or less.
- The real router's skew: popular experts take several passes, so their weights are decoded several times (from L2).
- The two-input gate / up path (if the sign vectors differ) runs at 45 KB a stage set.
- Y in fp32 is ~0.5 GB of the floor (bf16 Y is a follow-up).
- The first fast segment builds the extension (~1-2 min, cached in /cache/torch_extensions).
- A changed tag value misses grouped-era fast snapshots once.

## Prefill levers 2 (offline, 2026-10-02, for G7)

TF `dsv41-060` (G7 commits after 9e54180); window script `scripts/windows/G7-prefill2.sh` (+ `nsys_gaps.py`). No GPU
was used: the numbers are a cost model from G5's nsys of a 32K prefill (exported to SQLite and read kernel by kernel
against the NVTX ranges) until G7. Every lever is a knob, default off.

**The 8.5 s `share` wait is not a cost.** It is rank 0's plan exchange (`engine._share`: an all-gather of the plan on
the compute stream, then `.tolist()`), which waits for the previous segment's queued GPU work: the GPU is 99.8% busy
inside it (17 ms idle of 8.5 s). The GPU's real idle time in that run is 2.1 s of 35.2 s, and 1.87 s of it is layer
1's Engram wait at the start of every segment (~115 ms of a ~220 ms bulk read issued at round start, after the plan
exchange drained the GPU). Making the exchange asynchronous would save ~5 ms a segment; reading the Engram rows a
round early removes the bubble. Rank skew is not visible (both ranks pair up in every exchange), and the host never
gathers logits for prefill rows (`run(logits=False)`; replay skips the head).

| Knob (G7) | What | Exact? | 32K replay, after x3gm (~25.5 s) |
| --- | --- | --- | ---: |
| `TF_DSV41_PREFETCH_AHEAD=1` | a prefill round also issues the next segment's Engram bulk reads (`rounds` -> `prefetch.next_pieces`); the next round keeps a matching read in flight | same bytes | -1.7 s (**+7%**) |
| `TF_DSV41_PREFILL_GATHER=bf16` | the prefill partials go to the mHC boundary as bf16 (one rounding, no fp32 round trip before the exchange, no fp32 copy after it); `_site` widens as it loads | bit for bit | -0.75 to -1.0 s (+3-4%) |
| `TF_DSV41_ROPE_INPLACE=1` | the query / index-query RoPE in place on the bf16 projection (the widened fp32 copy and the NoPE copy go) | bit for bit | -0.55 s (+2%) |
| `TF_DSV41_MHC_BM=32/64` (`_WARPS`) | mHC row tile for windows >= 256 rows (`_site` is 3.3-3.9 ms at 2,048 rows, ~3x its DRAM floor, latency-bound 16 x 16 tiles) | bit for bit (FMA chains per row) | 0 to -1.1 s (unknown) |
| `TF_DSV41_PREFILL_ATTN=fused` (`_BMQ=32`) | `csa2/attn_pf.py`: one program a (row, head tile) over the compressed list then the SWA window, one softmax, no partials / merge / scratch, the whole segment in one launch (today: 16 launches of 128 rows, 42 MB of partials a launch written and read back) | fast tag + 1 (other bits; row-invariant) | -1.6 to -2.3 s (+7-9%) |

All five: ~25.5 s -> ~19-21 s at 32K, **~1,550-1,700 tok/s** (pass line 1,450 at GLM's half-transfer rule); full
prefill (no replay) ~600 -> ~700-780. Not done here: bf16 Y in x3gm's epilogue (-0.25 GB a layer segment, ~-0.35 s at
32K, +1.4%: x3gm.cu is under G6 now); streams between segments (the GPU is 94-99% busy and x3gm's persistent CTAs own
every SM, so overlap only has the Engram bubble to win, which the prefetch takes); the indexer (0.6 s, already on
tensor cores: `[32 heads x 128] x [128 x 64 keys]`). Streaming top-k is used from 16K visible keys (64K+ prompts; +2%
in G5); G7 measures it from 4K (`stream4k`).

**Replay check:** the encoder pass stops at layer 20's site + compressor (`run(upto=20)`: no head, no logits, no
decoder layer), the decoder runs only over the last 127 rows (`replay.finish`), Engram layers 1 / 14 are both encoder
layers (no wasted reads). Host work a segment is vectorized (the n-gram hash, one bulk read a layer, id tuples of
2,048 ints); nothing runs per row in Python on the critical path.

**Tests** (CPU): `tests/test_dsv41_prefill2.py` (6: ahead reads kept / exact / wrong guess a miss, the next pieces and
their lookback; bf16 gathered TP=2 == fp32 bit for bit, 36- and 8-row segments), `tests/test_dsv41_prefill2_interp.py`
(14: mHC bf16 G and BM 32 / 64 / 4 warps bitwise, RoPE in place bitwise, fused attention vs float64 / vs chunks / SWA +
replay / row invariance / paging / staging ring, knobs + tag, the fast-tag forward 20-row == 8-row segments). GPU (G7):
`tests/cuda/test_dsv41_prefill2_gpu.py` (bitwise at 2,048 x 5,120 + ms of each kernel, the fast tag end to end).

**G7 commands** (head, inside a held campaign window, after G6):

```bash
scripts/windows/G7-prefill2.sh stage                          # workstation
scripts/windows/G7-prefill2.sh check
WINDOW_HELD=1 scripts/windows/G7-prefill2.sh step tests       # bitwise + kernel ms; then knobs (8K / 32K a knob)
WINDOW_HELD=1 MHC_BEST=32 ATTN_BMQ_BEST=32 scripts/windows/G7-prefill2.sh step all4    # 8K-128K, everything on
WINDOW_HELD=1 scripts/windows/G7-prefill2.sh step all         # tests knobs exact all4 full4 nsys gate (~2.5 h)
```

**Risks:** untimed (GLM's in-engine transfer of isolated wins has been half or less); the mHC tile may spill at
BM 64 (the knob sweep shows it); the fused attention at BMQ 32 holds a 32 x 512 fp32 accumulator (~64 registers a
thread at 8 warps: spills would show as a slower `attn32`; BMQ 16 is the fallback); it changes the fast tag (49 with
gm), so fast prompt snapshots miss once, and its quality needs the gate; any of these source changes alters
`engine.code_digest` (NVMe session entries miss once); the ahead read holds one more segment's Engram rows on the host
(2 layers x 25 MB).

## What is implemented (M1 core)

| Piece | File(s) | State |
| --- | --- | --- |
| Family | `__init__.py` | MODEL_TYPES, `check` (config.json only), `cuda_engine` (TP=2), knobs `TF_DSV41_*`, `CUDA_KV_DTYPES = ("fp8",)` (the CLI's `--kv-dtype` now takes `fp8`), `CUDA_APP` / `CUDA_SERVE` when the serving layer's app exists |
| Ported kernels | `cuda/csa2/{rows,compress,index,attn,ref}.py`, `cuda/router.py`, `cuda/experts.py`, `cuda/expert_loads.py` + `x3ld.{cu,cpp}`, `cuda/topology.py` | from `engine/kernels`, arithmetic unchanged; added: top-k sizes as parameters, row strides (so the kernels read the KV pool's 584-byte rows) |
| Config, RoPE, Engram host | `cuda/config.py`, `cuda/rope.py`, `cuda/engram_host.py` | from `engine/reference`; Engram rows in memory or from the kit's packed NVMe shards (O_DIRECT) |
| Loader | `cuda/weights.py` | the EXL3 pack as stored, width from each trellis shape, TP=2 splits on 128 boundaries, router / weights_proj as bf16 params, `gate.bias_vl` from `TF_DSV41_BIAS_VL` |
| Prepared folders | `cuda/fastboot.py` | GLM 0140's design: digest-keyed per-rank folders, O_DIRECT readers into pinned buffers, chunk checksums, `python -m ...fastboot prepare / check` |
| Forward | `cuda/forward.py`, `cuda/blocks.py`, `cuda/linear.py`, `cuda/moe.py`, `cuda/csa2/{backend,twin,stores,fmt}.py`, `cuda/seams.py`, `cuda/numerics.py` | multi-segment runs (several slots' rows at once), 40 layers by CSA2 role, Engram sync, full prefill; positions move only on commit; GPU path = Triton CSA2 + router, upstream Exl3Linear (row-invariant) and grouped experts (+ 0580 `x3ld`, + `x3pf` prefill experts behind knobs); CPU path = float64 torch twins |
| Serving protocol | `cuda/slots.py` | `protocol.Forward` + `bind` for the serving layer: pool pages, prefill pieces, windows -> merged `Candidates` (grammar masks), commit with rollback, snapshot / restore |
| Engine | `cuda/engine.py` | GLM Spark contract; default: `stack.build` (pool, sessions + NVMe tier with both ranks' entries intersected at boot, batcher), `generate` -> `batch.generate_request`, `follow` -> `batch.follow`; `serving=False`: the one-slot serial path (M1 gates) |
| M1 harness | `cuda/gate.py`, `cuda/kbench.py` | gates vs the kit's oracle (whole + first copy), invariance, decode, boot; expert bench (bitwise, GB/s, probe 3) |

## Tests (CPU, no GPU)

- TF tree `tests/test_dsv41_*.py`: forward (TP=2 on threads == one rank in exact numerics, ranks bit-identical, row
  invariance bit for bit across window shapes, Engram from shards), engine (+ serving engine on two ranks, gate
  harness), slots (batched == alone == serial, commit rollback, resumed == fresh, masks, two ranks), weights, fastboot,
  interpreter: CSA2 / router kernels (incl. packed pool rows and short-next-to-long rows), the forward's Triton path.
- This repo `tests/family/` (17): the family vs `engine/reference` per layer kind and the whole model at 1e-12 in
  exact numerics (prompt, windows of 8, one row, TP=2), kit numerics argmax agreement; the loader on the real pack's
  headers (meta tensors), config, Engram layout, token map.
- GPU-only (collected only with CUDA): `tests/cuda/test_dsv41_gpu.py` (ours), `tests/cuda/test_dsv41_blockers_gpu.py`
  (the kernel track's).

## What still falls back to torch

- On CUDA the forward uses the kernel track's `cuda/mhc.py` and `cuda/engram.py` (fusion) through `seams` at D =
  5,120; the torch fallbacks remain the reference (`TF_DSV41_SEAMS=torch`) and run for other widths (tests).
- Torch on the GPU (row-invariant float64 / fp32 elementwise): the RMSNorms, the q / index-query RoPE, the indexer's
  `weights_proj` (32 outputs), Engram dequant (host side, from the shards) and hashing (host).
- Not wired yet: DSpark drafting (M2; `cuda/dspark.py` exists; its attention must use `dspark.attention_meta`, not a
  plain `hi` window), the Engram prefetch (reads are synchronous), CUDA graphs. (CED replay prefill: since
  `9d1ac19`, see Prefill below.)

## G1 / G2 commands (run 2026-10-02)

```bash
# workstation: export the branch to both Sparks
scripts/windows/G1-kernels.sh stage
# head, in tmux: one campaign for several windows (GLM prod down for the whole campaign, restored at the end)
scripts/windows/campaign.sh open            # lease + refresher, watchdog off, deadman +6 h, GLM down
scripts/windows/campaign.sh reboot-worker   # worker first; then reboot head itself and:
scripts/windows/campaign.sh arm             # refresher + deadman again, watchdog off again
scripts/windows/campaign.sh samplers start  # 0.5 s MemFree / MemAvailable on both nodes
WINDOW_HELD=1 scripts/windows/G1-kernels.sh run      # or: step tests | kbench L | x3pf L | prepare R
WINDOW_HELD=1 TF_DSV41_EXPERT_LOADS=1 scripts/windows/G2-m1.sh run
TF_DSV41_EXPERT_LOADS=1 scripts/windows/G2-m1.sh pair TAG MIN [gate args]   # one more gate run
scripts/windows/campaign.sh close           # GLM prod verified, samplers killed, lease deleted, watchdog on
```

## Blockers / open items (after G1 / G2)

1. **Gate 4 (serial decode 23 tok/s) fails at 19.4.** The GPU is busy ~39 ms of a ~52 ms step; the rest is host
   time: Engram inputs 5.75 ms a step before any GPU work, `choose`'s D2H sync, ~2,500 launches. M2's
   `DecodeForward` (Engram prefetch + CUDA graphs) is the path that removes it, but its graphs are broken on the
   real weights (G3-RESULTS.md).
2. **Memory**: 4 x 300K does not fit at 95 GiB a rank with the 5 GiB floor (G2-RESULTS.md, Memory).
3. Gate 2 fails (1 of 8 replies identical). Next: capture the kit with `SPEC_METHOD=none`, and check how a
   mid-sequence BOS is hashed by Engram (prompt 5: we loop on token 0, the kit does not).
4. Upstream's `group_kernel` cannot group more than 1,755 rows x 7 picks (48 KiB dynamic shared memory): a
   2,048-row prefill chunk needs a fix there or our own grouping. x3pf only wins at 512-768 rows on real picks.
5. The serving path's admission waits on MemFree after a cold boot (page cache from the checkpoint / folder
   write): 253 s in G3 until the caches were dropped by hand.
6. Prefill is slow (128-row windows: ~210 tok/s at 2K): offline work below (Prefill), measured in G5.
