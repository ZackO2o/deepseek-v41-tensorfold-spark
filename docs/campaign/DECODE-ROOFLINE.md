# Decode roofline: where a V4.1 verify window goes, and the DeepSeek-specific levers (2026-10-02)

Source: G4 `nsys1` (1-stream decode, graphs on, RoCE; `results/G4-20261002/nsys-nsys1_cuda_gpu_kern_sum.csv`), about
136 verify windows of 1-6 rows (mean ~2.6), plus G4 calibration (verify 1/2/4/6/8/16 rows = 35.0 / 43.2 / 51.6 /
60.0 / 68.4 / 95.9 ms) and G6 `routerbench` / `nsys-router`. Shapes are from the real `config.json`.

## 1. The floor of a 1-row window, per rank (TP = 2)

| Weights read for one token | Params (per rank) | Bits | MB | ms @ 220 GB/s |
| --- | ---: | ---: | ---: | ---: |
| Routed + shared experts: 7 x 3 x 5,120 x 1,152, x 40 layers | 4.95 G | ~2.6 | 1,610 | 7.3 |
| Attention dense (wq_a, wq_b 1,280 -> 32 x 512, wkv 512, wo_a 4 groups 4,096 -> 1,024, wo_b 4,096 -> 5,120, indexer, compressors) x 40 | ~2.8 G | ~5 | ~1,760 | 8.0 |
| Router gate 384 x 5,120 bf16, replicated, x 43 | 85 M | 16 | 170 | 0.8 |
| Head (64,640 x 5,120, K6) | 331 M | 6 | 248 | 1.1 |
| **Total** | | | **~3,790** | **~17** |

So a 1-row verify window has a floor of about **17 ms** per rank, and we measure **35 ms**: about 2x off. Every
extra row adds ~6 new experts a layer (~3.4 ms measured, near its own floor), which is inherent to the MoE.

## 2. Where the time goes now (per verify window, G4 nsys1)

| Kernel class | ms / window | vs floor | Why it is slow |
| --- | ---: | --- | --- |
| x3ld routed + shared experts | 19.4 | near floor for the ~2.6-row union | inherent: more rows, more distinct experts |
| EXL3 dense `linear_kernel` (~300 launches, 28 us each) | **15.4** | ~2x | many small matrices, each one a separate launch that cannot fill 48 SMs (2 MB at 28 us = 73 GB/s) |
| Router `_logits` + `_select` | **4.75** | ~6x | Triton fp32 `tl.dot` ieee with 16-row tiles: ~100 us for 3.9 MB (36 GB/s). The G6 fused variant kept the same inner loop, hence only 1.09x |
| RoCE gathers (93 a window, 45 us each) | **4.25** | ~3x | decode partials travel as **fp32** [n, 5,120] although after `kit_partial` they are exactly bf16 values; per-call `torch.empty` and copies |
| mHC `_site` + `_finish_k` + fp64 RMSNorm (`MeanOps<double>`, rsqrt, copies) | ~4.5 | ~3x | separate tiny kernels: norm is not fused into the mHC site that produces its input |
| cutlass fp64 GEMM (`d884gemm`) | 2.2 | | already replaced in G4 by one Triton kernel |
| Other small torch ops (~200 copies / elementwise a window) | ~2 | | launch-bound |
| Head (target) | 1.0 | at floor | |

## 3. DeepSeek-specific levers (what GLM did not have)

1. **Many small projections off the same input.** V4.1's attention reads the normalized input with wq_a (1,280),
   wkv (512), the compressor's wkv and wgate, and the indexer's weights_proj; the q latent then feeds wq_b (32 heads x
   512 a rank) and the indexer's q. GLM had a few big matrices; V4.1 has ~7.5 small ones a layer. **Horizontal
   fusion**: one launch over the concatenated rows of every projection that shares an input (one EXL3 GEMV over a
   stacked weight, outputs split by offset), and the same for the q-latent consumers. Fewer, bigger launches run at
   bandwidth. Expected: dense 15.4 -> ~9 ms.
2. **The router is a plain bf16 GEMV.** 384 x 5,120 is 3.9 MB; a warp-per-expert CUDA GEMV with 16-byte loads and a
   fixed warp-tree reduction runs at ~200 GB/s (~20 us), with the top-6 selection fused by the last block. The
   reduction order is fixed and per row, so row invariance holds; top-1 vs the kit is the gate. **And the shared
   expert does not depend on routing**: it can run while the router is computed. Expected: 4.75 -> ~1 ms.
3. **Single-head MQA (head_dim 512) and grouped wo_a (o_groups 8).** The KV latent is one 512-wide row: attention
   math is tiny, so every attention microsecond is projections and launches. wo_a is grouped (4 groups a rank), so
   the only attention exchange is wo_b's partial: exchange it as **bf16** (bit-exact, since `kit_partial` already
   rounds to bf16), into persistent buffers, and fuse the rank-order add into the next mHC site. Expected: RoCE
   4.25 -> ~1.5 ms.
4. **Single-Pass mHC already computes the next sublayer's input.** Fold the RMSNorm (and the fp64 mean) into the mHC
   site kernel that writes the collapsed row, and the residual add of the gathered partials into the same pass.
   Expected: ~4.5 -> ~1.5 ms.
5. **DSpark drafts never change the output.** Verification is exact, so the drafter's head and expert numerics are
   free to choose: a vocabulary-trimmed or 4-bit drafter head (our GLM drafter trim), and a graph-captured draft pass
   (launch-bound today: 4.7 ms eager vs a ~3.5 ms floor, 7.9 ms under nsys). Expected: draft 4.7 -> ~2.5 ms a round.
6. **Engram is a pure function of the last 4 token ids.** It is already prefetched; keep it off the window entirely.

## 4. Target

1-row window 35 -> **~20-22 ms**; draft pass 4.7 -> ~2.5 ms. With G4's acceptance:

| Workload | tokens / round | rows | round now | round target | tok/s now | tok/s target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Code | 3.9 | ~4 | ~66 ms | ~35 ms | 58 | **~110** |
| Prose | 1.77 | ~1.6 | ~57 ms | ~27 ms | 31 | **~65** |

That is roughly 2.5x Mia's kit on code and 2x on prose, before any multi-stream gains.

## 5. Lever 1 in detail: the dense EXL3 projections (fused by shared input, `TF_DSV41_FUSED_PROJ`)

Owner: the dense-projection track. Code: dsv41-060 `cuda/fused_proj.py`, `cuda/x3seg.cu` / `.cpp`, `cuda/projbench.py`,
the wiring in `cuda/blocks.py` (`Block.fused`, `attention`). GPU step: `scripts/windows/G7-decode.sh proj ...`.

### 5.1 Inventory: every dense EXL3 matmul of a layer, one rank (TP = 2), decode

Shapes and widths from the real pack's headers (`.cache/ckpt/dsv41-uncensored-2.9bpw/headers.json`; every group is
`mul1`). The plan (SK K splits, WK warps a program) is upstream's `plan(K, N)`; every plan walks only 8-10 k steps a
warp.

| projection | K -> N (rank) | bits (K2) | plan SK, WK | programs | MB | input | layers | launches / window |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: |
| `wq_a` | 5,120 -> 1,280 | 5 (L0: 6) | 8, 4 | 80 | 4.10 | x (normed) | all 40 | 40 |
| `wkv` | 5,120 -> 512 | 5 (L0: 6) | 8, 4 | 32 | 1.64 | x | all 40 | 40 |
| compressor `wkv` | 5,120 -> 512 | 5 | 8, 4 | 32 | 1.64 | x | kv sources 2, 8, 14, 20 | 4 |
| compressor `wgate` | 5,120 -> 512 | 5 | 8, 4 | 32 | 1.64 | x | ratio-2 kv sources 2, 8, 14 | 3 |
| `wq_b` (32 heads) | 1,280 -> 16,384 | 5 | 1, 8 | 128 | 13.11 | qr (q latent, normed) | all 40 | 40 |
| indexer `wq_b` | 1,280 -> 4,096 | 5 | 2, 4 | 64 | 3.28 | qr | index sources 2, 8, 14, 20, 24, 28, 32, 36 | 8 |
| `wo_a` (4 groups) | 4,096 -> 1,024 each | 5 | 8, 4 | 64 each | 2.62 each | its 4,096 columns of o | all 40 | 160 |
| `wo_b` (rows of this rank) | 4,096 -> 5,120 | 5 | 8, 4 | 320 | 13.11 | wo_a's output | all 40 | 40 |
| indexer `wk` | 512 -> 128 | 8 | 1, 4 | 1 | 0.07 | the compressed latent | kv sources, a segment | ~4 |

Not EXL3: the indexer `weights_proj` (32 x 5,120 bf16, `linear.Plain`, one Triton fp64 launch) and the router. Outside
attention: Engram `wkv` (layers 1 and 14, 6,144 -> 25,600), the head (K6), the drafter's 3 layers (4-bit attention:
nsys's `linear_kernel<8, 2, 4>`, ~24 launches a window).

Shared inputs: x feeds `wq_a`, `wkv` and the compressor's two (and `weights_proj`, not EXL3); qr feeds `wq_b` and the
indexer `wq_b`; the 4 `wo_a` groups read 4 column slices of the same o. `wo_b` reads `wo_a`'s output, so it stays alone.
Every group has one width per layer (5 bits; layer 0's `wq_a` / `wkv` both 6).

This matches G4 nsys1 exactly: `linear_kernel<10, 2, 4>` 40,727 launches / ~139 windows = 293 a window (39 + 39 + 7 + 8
+ 160 + 40), `<10, 2, 8>` 41 (`wq_b`), `<12, 2, 4>` 2 (layer 0), `<16, 2, 4>` 4 (indexer `wk`), and 370 `rot_in`
launches a window (one a projection). Attention's EXL3 time: 11.6 ms of linear + 0.6 ms of rot_in a window.

**Correction to section 3.** The projections average 4.1 MB (not ~2), and a two-parameter fit of the nsys class means
(4.11 MB at 27.96 us, 13.11 MB at 84.6 us) gives t = 2.05 us + bytes / 159 GB/s, which also reproduces the median
(19.4 us measured, `wo_a`'s 18.6 modelled). So a launch is not mostly fixed cost: every plan's programs walk only
8-10 k steps a warp (split-K chosen to fill the SMs), and a short walk's prologue / reduction / split-K epilogue
caps the kernel at ~159 GB/s. The same kernel at 40 steps a warp (the head) runs at 239 GB/s. Horizontal fusion removes
the fixed cost and overlaps the small launches' latency, but it does not lengthen the walks. Expect ~1-2.5 ms from it,
not 15.4 -> 9 ms.

### 5.2 Design: one launch per shared input, from a segment table (no restacked weights)

`x3seg.cu`'s `seg_linear_kernel` walks a table of up to 8 segments, passed as a kernel parameter (graph-capture safe,
nothing to copy). Each segment holds its trellis words, `svh`, K, N, upstream's own (SK, WK), its rotated input, its
output pointer / column / row stride, and its Z and counter slices. Program p belongs to the last segment with p0 <= p
and is upstream's program `(nb, split) = ((p - p0) % NB, (p - p0) // NB)` of that segment. Its statements are
`linear_kernel`'s, transcribed: the same k tile ranges, mma chain, warp-order sums, Z partials, last arriver's sum in
split order, Hadamard epilogue and output rounding. A 4-warp segment inside an 8-warp launch (indexer `wq_b` with
`wq_b`) leaves warps 4-7 idle; they share only barriers and the epilogue's work split, never an addition. `seg_rot_in`
is upstream's `rot_in` for up to 8 (input, `suh`) pairs in one launch, with row-strided inputs (`wo_a`'s slices of o).

Why not restack the weights in the prepared folder? Stacking along output rows would be a concatenation of
128-column strips, but it cannot keep the bits. (1) Each projection has its own input rotation `suh`, so a stacked
matrix has no single `rot_in`. (2) `plan(K, N)` depends on N: wq_a + wkv stacked (N = 1,792) happens to keep (8, 4),
but other stacks would change SK, and so the split-K summation order. The table keeps every projection's own words and
plan, and needs no new weights, no prepared-folder change and no extra memory beyond the counters (a few KB).

Groups a layer (`fused_proj.Groups`): **x** = wq_a + wkv [+ compressor wkv + wgate, written as one `[kv | score]`
buffer, which is the old `torch.cat`], **qr** = wq_b + indexer wq_b (index sources), **o** = wo_a's 4 groups into one
`[n, 4,096]` buffer (the old 4 `.contiguous()` copies + `cat` are gone). `wo_b` keeps upstream's call. Launches a window:
**335 -> 160** linear (+ ~4 indexer `wk`), rot_in the same, and ~200 small copies / cats gone. `TF_DSV41_FUSED_PROJ=0`
restores the per-projection calls. Fast prefill runs (`prefill_mm.active()`) and the CPU twin always use them.
`TF_DSV41_FUSED_PROJ_MINB` (3 or 2) selects the 4-warp kernels' register budget (sm_121, K2 10: 3 gives 168
registers and upstream's occupancy with 112 B of spills; upstream itself spills 60 B; 2 gives 218-252 registers and
no spills). That only changes speed, never a value.

### 5.3 Numerics: bit for bit with the unfused path

- `tests/test_dsv41_fused_proj_emu.py` + `tests/dsv41_x3seg_emu.py`: a numpy program-level emulator of upstream's
  `linear_kernel` / `rot_in` and of the segment kernels. The segment launches run over `fused_proj.layout`'s own tables
  in shuffled program order, with shared flat scratch (xh, Z, counters, outputs), so a wrong offset shows up as a
  collision. Every element equals upstream's bit for bit, is written once, and all counters end at zero. Cases: 1-128
  rows, split-K and one-split, 8- and 4-warp plans in one launch, two widths, several segments a buffer, and the real
  layer-1 `wq_a` + `wkv` and `wo_a` from the cached pack tensors. Mutating any offset in `layout` fails the tests.
- `tests/test_dsv41_fused_proj.py`: `blocks.attention` with stand-in groups gives the forward's logits bit for bit
  across every layer kind and windows of 1-16 rows. The switch and the fast-prefill bypass are covered.
- `tests/test_dsv41_fused_proj_compile.py`: every dispatched instance builds for sm_121 with upstream's flags (`-O3
  --expt-relaxed-constexpr`, the same contraction choices). Verified offline with the pip CUDA 13.4 nvcc (`TF_NVCC`).
- GPU (`tests/cuda/test_dsv41_fused_proj_gpu.py`, `G7-decode.sh proj tests`): fused == upstream on random trellises at
  the real shapes (R = 1..16, 64, 128, 130; bf16 / fp32; graph replays; counters zero), and on the real weights for
  every layer at R = 1..16 (every dense output of `blocks.attention`).

### 5.4 Expected effect (cost model, per verify window, rank 0)

Per launch t = c + bytes / bw (c = 2.05 us, bw = 159 GB/s, fitted above). For a fused launch, the pessimistic case
keeps bandwidth additive (only c and the gaps go). The optimistic case overlaps the members' latency chains:
t = c + max(the slowest member's walk, total bytes / 239 GB/s), 239 GB/s being the head's measured rate.

| | launches | linear ms | rot_in ms | copies / cats | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| unfused (G4 nsys) | 335 + 370 rot_in + ~203 copies | 11.6 | 0.60 | ~0.4 | ~12.6 |
| fused, pessimistic | 160 + ~165 rot_in | 11.3 | 0.26 | 0 | ~11.6 (-1.0) |
| fused, optimistic | 160 + ~165 rot_in | 9.8 | 0.26 | 0 | ~10.1 (-2.5) |

The 1-row window goes from 35 to ~32.5-34 ms. Each extra row adds the same ~1-2.5 ms saving to every window,
because launches do not grow with rows. G7's pass line is >= 0.8 ms on the calibration's verify-1 time.

### 5.5 Next, and risks

- **1b: longer walks, implemented as `TF_DSV41_PROJ_PLAN` (default `upstream` until G7's gate).** The quality gate
  is now top-1 vs the kit >= 96% (aim 98%) plus MMLU within 1 point, so a different reduction order is allowed. See
  5.6.
- **The drafter's attention** (3 layers, 4-bit, ~24 launches a window, ~0.8 ms) can use the same groups: drafter
  blocks are `blocks.Block`s, so `blk.fused` is already built. `drafter._attention` could call `blk.fused.x(...)` /
  `oproj(...)`; that is the drafter owner's file.
- `calib.KNOBS` should list `TF_DSV41_FUSED_PROJ` (and `_MINB`), so a cached calibration is not reused across the
  switch. calib.py has another track's uncommitted edits, so it is left to them. G7 runs with `TF_DSV41_CALIB=real`.
- Risks: (1) bit identity rests on transcribed statements compiled with the same flags. The emulator proves the
  indexing; only the GPU test proves the codegen (FMA contraction in `finish` / `rot_in` is the same source
  expression). (2) MINB 3 spills 112 B at K2 10 (upstream: 60 B); if the GPU shows a slowdown, MINB 2 is the
  fallback. (3) Graph capture allocates xh / Z per call from the graph pool, as upstream does; the counters are per
  group, so one group must not run on two streams at once (same as upstream's per-layer counters). (4) The gain may be
  at the pessimistic end if each launch is bandwidth-bound rather than latency-bound; `proj bench` measures it a layer.

### 5.6 The numerics change allowed by the new gate: `TF_DSV41_PROJ_PLAN=long` (lever 1b)

The user's decision (2026-10-02): the quality gate is top-1 vs the kit's oracle >= 96% (aim 98%; the kit itself is
~89% vs the reference) and MMLU within 1 point. Bit identity with the old path is no longer required. Row invariance (a
row alone == in a window), batched == alone and drafted == serial must still hold.

The faster design under those rules keeps the fused launches and lengthens the walks. Upstream's `plan(K, N)` splits K
until a lone launch fills the SMs (8-10 k steps a warp). A fused launch is already full with fewer splits, and the
same kernel at 40 steps a warp runs at 239 GB/s (the head) against ~159 at 8-10.

| group | upstream (SK, WK) -> steps | long (SK, WK) -> steps | programs (long) |
| --- | --- | --- | ---: |
| x: wq_a + wkv [+ compressor] | (8, 4) -> 10 | (4, 4) -> 20 | 56 / 88 |
| q: wq_b + indexer wq_b | (1, 8) -> 10 / (2, 4) -> 10 | (1, 4) -> 20 | 128 / 160 |
| o: wo_a x 4 | (8, 4) -> 8 | (2, 4) -> 32 | 64 |
| wo_b | (8, 4) -> 8 | (2, 4) -> 32 | 80 |

What changes: only the split-K summation order of each output element. That is fp32 partial sums of 1-4 splits
instead of 8, and 4-warp sums over longer chains. The emulator test shows the difference is at fp32 rounding (<= 1e-5
of the max value). The GPU test prints the real-weight drift and asserts < 2e-2.

What holds:
- The plan is the shape's alone (never the row count). It is set on the `Exl3Linear` objects at Block build
  (`fused_proj.apply_plan`), so fused and unfused calls, CED's compressor pass, the drafter's blocks and exact
  prefill windows all sum alike.
- Row invariance under any plan is tested in the emulator (a row alone == in a 17-row launch, synthetic and the real
  layer 1) and on the GPU (rows of a 16-row window == the same rows in 1..16-row windows, every real layer).
- Disk sessions key on every `TF_DSV41_*` knob, so entries made under one plan are never resumed under another.

Expected on top of fusion: dense ~7.5-9 ms instead of ~9.8-11.3, another 1.5-3 ms a window. With fusion, 1-row
window 35 -> ~30-32 ms. Uncertainty: 56-88 programs of 4 warps on 48 SMs put only ~5-7 warps on an SM. The head
reached 239 GB/s with 8 resident warps an SM. If `proj bench` shows the long plan below upstream's for a group, that
group's SK goes back up (`TF_DSV41_PROJ_PLAN=x:8,q:1,o:2,wob:2` style specs, no code change). G7: `proj bench`
(drift, bitwise, row invariance, ms per plan), `proj window` (off / on / long), `proj gate` and `gate-up`, and `proj
quality` (MMLU through G6-ship.sh). Adoption is `plan_spec`'s default plus config/prod.env.

## 5. Update after the exchange track's trace work (2026-10-02)

- The in-graph decode gathers are 81 a window, 2.96 ms. **About 2.3 ms of that is waiting on the other rank**, not
  transport: MoE gathers median ~20 us against attention ~9 us, consistent with rank 1 running its MoE ~11 us a layer
  slower. That skew becomes the largest exchange lever. G7 must trace rank 1 too (nsys on both ranks, or GLM's
  rocedump tick in the round loop) and find why: worker clocks or thermals, an uneven expert split, or rank-1-only
  work (its vocabulary half, the drafter).
- RMSNorm after mHC was already fused. The fp64 norm sets are q_norm and kv_norm: now one launch each (ca31bc3).

## 6. What G7 shows we missed: host blocking and eager multi-slot rounds (lead, 2026-10-02 evening)

Source: G7 `m2-ab-adopted.json` phases (wall, `TF_DSV41_PHASES=1`) and the rank-0 nsys of 1-stream code
(`nsys-skew-r0.sqlite`: GPU busy intervals vs NVTX phases per round).

**1 stream (code, 69 tok/s):** a round is 56.4 ms under nsys, with the **GPU idle for 8.1 ms of it (14%)**:
`graph.stage` 4.8 ms wall / 4.1 ms idle, which is `engram.wait` (4.8 ms wall / 4.0 idle: the drafted rows' Engram
reads are issued after the draft pass and the host blocks on them before replaying the window), and ~1.4 ms more in
`forward`. Commit / share / prefetch add ~1.4 ms of idle.

**2 and 4 streams are slower than 1 for three reasons, all host-side:**

| phase (ms a round) | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| round | 53.8 | 68.3 | 95.5 |
| window (wall) | 46.3 | 54.2 | 72.4 |
| forward (host launch) | **0.41** | **23.0** | **34.2** |
| graph replays / rounds | 101 / 101 | **122 / 229** | **128 / 229** |
| draft pass | 3.6 | **10.2** | **13.1** |
| engram.wait (x2 a round) | 0.23 | **3.3** | **6.5** |

1. **Multi-slot windows run eager.** Only single-slot widths are captured as CUDA graphs (the perf track assumed mixed
   rounds were GPU-bound), so every mixed round launches ~thousands of kernels from Python: 23-34 ms of host time a
   round.
2. **The DSpark pass is only graphed for slot 0's keys**: multi-slot draft passes run eager (10-13 ms vs 3.6).
3. **Engram reads block the host** before the window: 2 waits a round, 3.3-6.5 ms each with more slots.

**Fixes (G8):**
- Slot-agnostic CUDA graphs keyed by the padded total row count. Per-row slot / position / page metadata lives in
  device tables written by one async copy, so one graph serves any slot mix. The same for the DSpark pass.
- Engram off the host's critical path: issue the drafted rows' reads from inside the draft pass's completion (or
  speculatively for the top draft candidates), use a native high-queue-depth reader (io_uring), and make the *GPU*
  wait on an event right before layer 1 / layer 14 instead of the host waiting before the replay (layer 14's reads
  then overlap 13 layers).
- Expected: C1 round ~56 -> ~47 ms (code ~80+ tok/s); C4 round ~95 -> ~60 ms (C4 ~100+ tok/s aggregate).

## 7. The dense EXL3 linears at bandwidth: x3dn (`TF_DSV41_DENSE`, the dense track, 2026-10-02)

Source: G7 `nsys-skew-r0.sqlite` (rank 0, 1-stream code), decode period only (96 windows of ~4.7 rows, after the
prompt prefills; dynamic shared memory gives each launch's row count). Code: dsv41-060 `cuda/dense.py`,
`cuda/x3dn.cu` / `.cpp`, `cuda/densebench.py` (`projbench --gbps`); G8: `scripts/windows/G8.sh dense ...`.

| class (per rank, 5 bits) | launch grid | MB | us | GB/s | ms / window | floor @ 240 GB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| wo_b `linear_kernel<10,2,4>` (SK 8) | 40 x 8 = 320 programs, 3 an SM | 13.1 | 68.5 | 191 | 2.75 | 54.6 us |
| o: wo_a x 4 `seg_linear<10,2,4,3>` (SK 8) | 256, 3 an SM | 10.5 | 59.6 | 176 | 2.39 | 43.7 |
| wq_b `linear_kernel<10,2,8>` (SK 1) | 128, **1 an SM** (220 regs x 256) | 13.1 | 73.9 | 177 | 2.37 | 54.6 |
| x: wq_a + wkv `seg_linear<10,2,4,3>` | 112, 3 an SM | 5.7 | 36.5 | 157 | 1.28 | 23.9 |
| q + indexer wq_b `seg_linear<10,2,8,1>` | 192, 1 an SM | 16.4 | 97.5 | 168 | 0.78 | 68.3 |
| head `<12,2,8>` (x 2 a window: the draft pass uses the full head) | 505 | 248 | 1,057 | 235 | 2.11 | |
| Engram wkv `<10,2,8>` / `<8,2,8>`, drafter `<8,2,4>` etc. | | | | 109-215 | ~1.6 | |

**Why the attention's matrices run at 155-190 GB/s.** Not decode cost (~400 instructions a warp a k step, ~20%
of issue at bandwidth), not launch count alone (fusion took ~1-2.5 ms). A per-SM byte model fits every class
within 2-13% (x 35.7 vs 36.5 us, o 54 vs 60, wo_b 62 vs 69, wq_b 66 vs 74): t = (bytes of the busiest SM) / (240 / 48 GB/s) + ~5 us. One program is one wave slot, so the time
is set by the SM holding the most programs: x has 112 programs on 48 SMs (16 SMs run 3, 32 run 2: 1.29x the
balanced time), wo_b 320 = 2.22 waves of 144 (a third wave of 32 lone programs, latency-bound), o 256 = 1.78 waves,
wq_b 128 programs at 1 an SM (2.67 waves; 8 warps an SM). On top: every program ends in a fence + atomic + Z round
trip, trellis words come as 24 4-byte loads a lane a k step (each word read by ~2.4 lanes), and every k step waits
on an unprefetched input fragment (an L2 round trip). TF_DSV41_PROJ_PLAN=long only lengthened the walks (~1% in
G7): the imbalance stayed.

**The `unpack_kernel` calls (5.5 a window in the G7 table) are not decode.** All 938 of them run in two bursts at
0-0.4 s and 3.3-3.7 s of the trace: the two prompts' fast prefill (`prefill_mm` -> upstream `exl3.prefill.matmul`,
which decodes W_q once a call into a workspace for its GEMM, by design). Zero run inside decode windows.

**x3dn.** Persistent CTAs (SMs x occupancy = 144 at K2 10, all resident) take K atoms from a counter; the x3ld load
path (16-byte `cp.async.cg` into a 3-deep shared-memory ring a warp, input fragments `cp.async` 4 B, one group a k
step), with each warp's load cursor running ahead across atoms so reductions never drain the pipe; a strip's last
atom sums the partials; PDL: the weights of a CTA's first atom are requested before `griddepcontrol.wait`, behind a
PDL `rot_in` that triggers at once. Every SM streams the same bytes to within one atom (~20 KB). sm_121: 141-150
registers, no spills / stack, 3 CTAs an SM at K2 10 (29.7 KB of shared memory a CTA, max carveout).

**Bits** (a new summation order, gated like the long plan): a strip's K/16 k steps in S = max(1, K/16 / 16) atoms,
each atom's 4 warp quarters summed in warp order, the S atom partials in atom order, upstream's epilogue. The atoms
depend on K alone, so a matrix has the same bits in any group (x2 or x4: CED's compressor pass), fused or not, at
any row count, CTA count, ring depth or schedule: row invariance, batched == alone, drafted == serial hold. The
switch is on the `Exl3Linear` objects (`dense.mark`, from `fused_proj.build`), so every path that computes an
attention matrix uses it. `TF_DSV41_DENSE=0` (default until G8) | `attn` | `all` (+ head, Engram, drafter
`main_proj`, indexer wk); `_PD` 2-4 (3), `_PDL` (1), `_CTAS` (auto): speed only.

**Expected** (t = 3 us + bytes / 240 GB/s, per window): x 36.5 -> ~27 us, o 59.6 -> ~47, wq_b 73.9 -> ~58, q+ix
97.5 -> ~71, wo_b 68.5 -> ~58; the attention's dense 9.9 -> ~7.7 ms (-2.2 ms a window at any row count: launches
do not grow with rows); `all`: the drafter's 4-bit matrices ~1.2 -> ~0.8 ms, head / Engram ~-0.1 ms. Dense total
13.7 -> ~11.2 ms. The remaining gap to "~8-9 ms" is not kernel bandwidth: the window reads the 248 MB head twice
(target + draft pass, 2.1 ms); a trimmed / 4-bit draft head (`TF_DSV41_DRAFT_HEAD`, the drafter's lever) takes ~1 ms.

**Tests.** `tests/test_dsv41_dense_emu.py` + `tests/dsv41_x3dn_emu.py`: a CTA-level numpy emulator over dense.py's
own tables (dynamic item counter, sequence publishing, every warp's cursor / ring slot checked against what its mma
chain consumes, shared flat scratch): written-once outputs, counters / ctl back to zero, bits equal across CTA
counts 1-400, ring depths and interleavings, rows alone == in a window (1-32 rows), a matrix's bits in any group,
within 2e-5 of float64 and of upstream (incl. the real layer-1 wq_a + wkv and wo_a). `tests/test_dsv41_dense_compile.py`:
sm_121 build of every instance (regs <= 168, no spills) and the SASS (LDGSTS.E.BYPASS.128, DEPBAR, 16 HMMA a step,
no LDL / STL). `tests/cuda/test_dsv41_dense_gpu.py`: G8.

**Risks.** (1) Only the GPU proves the cp.async / PDL codegen and the occupancy (if the driver's carveout leaves 2
CTAs an SM, G drops to 96: still balanced, fewer bytes in flight; `_PD 4` is the lever). (2) PDL inside CUDA graphs
on GB10 is new to this family (x3ld's PDL stayed opt-in); `_PDL=0` is the fallback, bit for bit. (3) Per-launch Z
of passes x atoms x rows x 512 B (q group at 32 rows: 6.5 MB) from the graph pool. (4) A group must not run on two
streams at once (its counters), as upstream's per-layer counters.

### 7.1 v1 lost in the engine; v2 (dense track, 2026-10-02 night)

G9 (G9-RESULTS section 2) measured v1 9-24% SLOWER than upstream on every shape in the engine (attention dense
9.32 -> 11.06 ms a 1-row window), and no schedule knob (PD, PDL, CTA count) moved it. The section-7 design numbers
came from an L2-hot test (upstream itself at 288-443 GB/s there), and the per-SM imbalance model behind the
persistent schedule did not hold: v1 balanced the SMs and still lost to its 16-20 partials a strip, each with a
fence, two atomics and a last-arriver epilogue.

v2 keeps what should matter at cold DRAM and drops the rest: upstream's grid shape (one program a strip x K split,
no persistent loop, no work counter), the x3ld load path (16-byte cp.async ring, PD 3: ~15 KB in flight a CTA), one
reduction a program (the 4 warps in shared memory; with SK > 1 one partial + one arrival atomic), and few splits a
shape (`dense.SPLITS`: wq_b 1, wo_b 2, wo_a 2, indexer wq_b 2, wq_a / wkv 4; upstream 1 / 8 / 8 / 2 / 8;
`TF_DSV41_DENSE_SPLIT` overrides it for a sweep). The premise: with ~10 GB/s in flight a CTA, ~24 busy CTAs carry
DRAM, so fewer and larger programs lose nothing to uneven counts an SM and save the per-split round trips. It is
still unmeasured on a GPU. G9.sh `dense` benches COLD (densebench.cold_us: a 256 MB write between calls inside the
graph, minus the writes alone) and sweeps the split tables before the window / gate / speed steps.
