# Rewrite study: DeepSeek-V4.1-Flash decode on 2x DGX Spark (2026-10-03)

> Research and analysis only: no GPU was used and no engine code was changed. Inputs:
> - this repo: [DECODE-ROOFLINE](DECODE-ROOFLINE.md), [G9](G9-RESULTS.md), [G10](G10-RESULTS.md), [G12](G12-RESULTS.md),
>   [M1-STATUS](M1-STATUS.md), `results/G9-20261002/` (`window-p1-r0.json`, `glue-nsys.txt`, `idle-p1-r0.txt`,
>   `nsys-g9dense-off_cuda_gpu_kern_sum.csv`);
> - the engine, `dsv41-060` @ `a20be14` (`src/tensorfold/families/deepseek_v41/cuda/`);
> - the GLM repo's earlier attempts at the same ideas: DECODE-KERNELS (0440), HC-FUSED (0520), NATIVE-ROUND, GPU-ROUND
>   (0450), THEORY-2 and DECODE-PLAN;
> - antirez/ds4 at `0aaea5a` (code, issue #773, PRs #266 / #621 / #763 / #804 / #993, commits e50104f6 / 15f42aaf /
>   b42682f1);
> - published megakernel work (section 3).
>
> **Every ms below is per rank, for one verify window of 1 row, unless a line says otherwise.** "Expected" already
> applies the transfer discount this project has measured. GLM: in-engine gains have been half or less of isolated
> wins. Here: glue came in at a third of its model, L2PF at the low end of its range, and dense x3dn was negative.

## 0. Bottom line

- **A 1-row window is 26.9 ms against a ~16.8 ms floor (62.5%).** The ~10 ms gap breaks down as:

  | part of the gap | ms |
  | --- | ---: |
  | glue above its own bytes | ~5.7 |
  | expert-class small kernels and ramp / tail | ~2.1 |
  | dense EXL3 below 232 GB/s | ~1.3 |
  | exchange wait above transfer | ~0.7 |
  | idle inside the graph | 0.3 |

  Outside the window, each round loses another **~2.3 ms of GPU idle**. **1.27 ms of that is the driver submitting
  the ~1,616-node window graph** (`cudaGraphLaunch`, measured as `graph.replay`: 1.35 ms wall, 1.27 ms GPU idle). It
  is not Python.
- **ds4 is not a megakernel and is not at 85-90% (section 2).**
  - It is conventional C with one launch per op: eager, the NULL stream, ~1,500-1,900 launches a token, and partial
    graphs ("decode islands") that bought +1-2%.
  - Recomputed against its own 233 GB/s probe, it reaches 72% (issue #773), 77% (main) and ~87% (open PR #804).
  - Its gains came from bytes and load geometry: aligned 16-byte weight layout (+9-27% a GEMV), more warps in flight
    (+25%), 4-bit attention projections (+14-18%), and scratch reuse (+7.5%). They did not come from fusion. Its
    launch-removing fusions were flat.
- **Reconciling "fusion was neutral on GB10" with our gap.** Launch count alone is worth almost nothing on the device
  (0.3 ms of in-graph idle for 1,616 launches). Our glue is 26% of a window, against a few percent of ds4's 50 ms
  single-GPU token, because TP=2 halves the weight bytes a GPU reads but not the per-layer latency chains (mHC, the
  replicated router, norms, top-k, 2 exchanges a layer).
  - A rewrite pays here only if it shortens a **DRAM-idle dependent chain on the critical path** or **raises a
    streamer's GB/s**.
  - The one place where launch count itself costs time is the host: graph submission scales with node count.
- **What is realistic.** Mid-size kernels, no megakernel:
  - the exact-quality set (ranks 1-5 in section 5) takes the 1-row window to **~23 ms** and the round's host idle
    down by ~1 ms: **prose 41 -> ~47 tok/s, code 80 -> ~89**;
  - adding 4-bit dense attention (f1, quality-gated), b2 and f2 gives **~20-21 ms**, ds4-class efficiency (~80% of
    floor): **prose ~50-52, code ~93-95**, at today's acceptance.
  - A whole-window megakernel (c) adds at most another ~1-2 ms on top of that, for 500-900 h, at very high risk. It
    is the last step of the path, not the first.
- **Top 3 to start now** (section 6):
  1. **(d1) Hide the window graph's submission**: split / pre-enqueued graphs. -0.9 to -1.2 ms a round, ~12-24 h.
  2. **(e1) The mHC boundary as one hand-CUDA kernel** (post + site + finish + norm, 1-16 rows). -1.2 to -2.0 ms a
     window, ~40-70 h.
  3. **(b1) The MoE chain**: router -> select -> rot_in in one kernel, and the shared expert concurrent with the
     router. Then (b2) gate/up -> act -> down with per-expert flags. -0.8 to -1.8 ms, ~60 h for b1 and +100-140 h
     for b2.
- **A C++ / Rust host loop (d2) is not worth doing for one stream.**
  - Python's share of a round's idle is ~0.9-1.1 ms (prefetch 0.34, draft bookkeeping 0.31, candidates readback
    0.27). Much of that is waiting on the GPU rather than interpreting.
  - GLM's C++ round measured +0.33% at 1 stream (NATIVE-ROUND, W24).
  - Revisit it for C2 / C4, where host idle is ~1.8 ms a round (G9 section 3).

## 1. The measured budget this plan works from

### 1.1 The 1-row window today (26.9 ms, `a6f5792` = `38f6500`'s knobs)

There is no nsys of the prod config, so the class split is built from G9's p1 trace with the "on" column of
`glue-nsys.txt` (30.24 ms under nsys). Two adopted levers are then applied:

- **prune p85k**: -0.9 ms calibrated (G10 section 1: 28.7 -> 27.8), all of it experts;
- **L2PF 12 MiB**: -0.6 ms (G10 section 3), split 0.4 dense and 0.2 experts / router.

The result is scaled by 26.9 / 28.8 for nsys overhead (G9: 31.0 under nsys vs 29.5 calibrated).

| class | ms today | its bytes, per rank | floor @ 232 GB/s | gap |
| --- | ---: | --- | ---: | ---: |
| routed + shared experts (2 `x3ld` + group, rot_in, gate/up epilogue, down_combine, per layer) | **7.9** | ~1.35 GB (p85k: ~5.9 of 7 experts a layer) | 5.8 | 2.1 |
| dense EXL3 (x / q / o groups, wq_b, wo_b, head, Engram wkv) | **10.5** | ~2.14 GB | 9.2 | 1.3 |
| mHC site + finish (80 + 80 launches) | **3.0** | 80 MB (`fn` bf16) | 0.35 | 2.65 |
| router (`rg::gemv`, 40 + 3, 32 us each) | **1.2** | 170 MB (replicated) | 0.73 | 0.47 |
| attention core (`_chunks`, `_merge`, `_kv_store`, `_rope`, Engram `_fuse`) | **1.2** | ~15 MB of KV | 0.1 | 1.1 |
| torch top-k / sort, copies, norms, indexer | **1.5** | small | 0.05 | 1.45 |
| exchanges (81: 40 x 2 + 1) | **1.3** | 81 x 7.5 us transfer floor | 0.61 | 0.7 |
| idle inside the graph (1,616 boundaries) | **0.3** | | 0 | 0.3 |
| **total** | **26.9** | ~3.8 GB | **~16.8** | **~10.1** |

Glue (mHC + router + attention core + the small rest) is **6.9 ms**, matching the "~5-7 ms" in the brief.

The expert and dense rates support the user's numbers:

- **Experts.** `x3ld` itself streams at ~240 GB/s. The class runs at ~170, because of four small kernels a layer
  (group 7.5 + rot_in 5.3 + epilogue 7.8 + down_combine 5.6 = **26 us a layer**, 1.05 ms a window, at near-zero
  DRAM use) and two ramp / tail pairs.
- **Dense.** Shapes run at 157-201 GB/s (G9 section 2); the head runs at 235.

### 1.2 Outside the window, a round (G9 `idle-p1-r0.txt`: prose, 1 stream, 43.3 ms round under nsys)

| GPU-idle phase | ms a round | what it is |
| --- | ---: | --- |
| `graph.replay` | **1.27** | `torch.cuda.CUDAGraph.replay()` = `cudaGraphLaunch` of the window graph: driver time (1.35 ms wall / ~1,616 nodes: ~0.8 us a node, inferred) |
| `prefetch` | 0.34 | Engram reads issued at drafter end (Python + native reader) |
| `draft` | 0.31 | depth choice, bookkeeping around the graphed DSpark pass |
| `candidates` | 0.27 | readback + Python candidate handling |
| other | ~0.2 | commit, staging |
| **total** | **~2.4** | 5.5% of the round. Code: 2.21 (graph launch 1.30) |

### 1.3 How ms become tok/s (C1, today's acceptance)

Prose is 41 tok/s at 1.58 tokens a round, so a 38.5 ms round. Code is 80 tok/s at ~3.9 tokens a round, so ~48.7 ms.
Every window lever below is row-independent: launches and glue do not grow with rows. So a saving of D ms a window
is D ms a round.

| D (ms a round) | 1 | 2 | 3 | 5 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose tok/s (41.0) | 42.1 (+2.7%) | 43.3 (+5.5%) | 44.5 (+8.5%) | 47.2 (+15%) | 50.3 (+22%) |
| code tok/s (80) | 81.7 (+2.1%) | 83.4 (+4.3%) | 85.2 (+6.6%) | 89.2 (+11%) | 93.5 (+17%) |

Prose is drafter-bound (1.6 tokens a round), but a cheaper window still helps it *more* than code: its round is
mostly the fixed 1-2-row window.

## 2. antirez/ds4: what it really is, and what transfers

Read at `0aaea5a`: `ds4.c` (85k lines, host), `ds4_cuda.cu` (34k), and `cuda/mmq/` (vendored llama.cpp MMQ / MMVQ
plus the GB10 "aligned" kernels). Built with `-O3 --use_fast_math -arch sm_121a`. Note that it runs **V4-Flash**, not
V4.1: 43 layers, d 4,096, 256 experts top-6, FFN 2,048. That is 9.3 GB a token on one GPU.

### 2.1 Structure

| question | ds4 | us |
| --- | --- | --- |
| host loop | single-threaded C, **eager on the NULL stream**; ~10 ms of host encode a token, running ahead of ~45 ms of GPU | Python, row graphs (slot-agnostic), ~1,616 nodes a window |
| graphs | "decode islands" (`ds4_cuda.cu:927-1150`): per-layer pre-attention and post-attention pieces, captured on second sighting; the position-dependent middle stays eager. **+1-2%** (e50104f6: "launch overhead is not the dominant serial-decode cost"). PR #804 turns them **off** on integrated Blackwell (+0.5%) | whole-window graphs; needed, since Python eager is ~15 us a launch |
| launches a token | ~35-45 a layer, ~1,500-1,900 a token (estimated from the code paths) | ~40 a layer, 1,616 a window |
| persistent / megakernel | **No.** No `grid.sync` in production. Cooperative "M2" fusions (`proto_m2_hc.cu`: 96 blocks, 4 `grid.sync`; router; compressor) exist **only as prototypes in `cuda/mmq/test/` and did not ship** | No (x3dn's persistent dense was slower) |
| MoE, batch 1 | one launch for all 6 experts' gate+up (IQ2_XXS, **warp per output row**, no split-K, gate and up interleaved so the Q8_1 activation loads once, SwiGLU + clamp + router weight in the epilogue), one Q2_K down launch, one sum. Verify: dedup of distinct experts. Shared expert separate (Q8_0); its down fused with the HC expand | 2 x3ld launches (gate/up, down) + 4 small kernels + router, union of distinct experts |
| attention | MQA latent 512, 64 heads, F32 KV storage with FP8-rounded values, 16-head x 16-row smem tile score-split kernel | same model family; FP8 KV records, 16-head tiles (Triton) |
| formats | routed gate/up IQ2_XXS 2.06 bpw, down Q2_K 2.625, **attention projections, shared expert and head Q8_0 (8.5 bpw)**, router / HC / compressor F16 | EXL3 mul1 trellis: experts ~2.6 bpw, dense attention **5 bits**, head 6 |
| GEMV style | warp per row, Q8_1 activations, `dp4a`, shuffles, no smem for weights, no cp.async / TMA / L2 policy on the decode path | EXL3: mma.m16n8k16 on decoded tiles, split-K + Z partials (dense), cp.async / 16-byte ring (x3ld) |
| TP=2 over RoCE | 21.9 t/s, **only ~+14%** over one Spark | 37 tok/s-equivalent for 1-row windows (26.9 ms), on a larger model (V4.1) |

### 2.2 The bandwidth claim, recomputed

- **Issue #773's method is wrong on two counts.** It divides "10-13 GB" by the 45 ms `execute` time using 61 layers.
  Flash has 43 (61 is Pro), and the real wall time is ~55.8 ms (17.9 t/s).
- **With 9.34 GB a token (PR #804's 8.7 GiB) against ds4's own 233 GB/s probe:**

  | run | t/s | GB/s | of the probe |
  | --- | ---: | ---: | ---: |
  | #773 | 17.9 | 167 | 72% |
  | main, 2K context | 19.25 | 180 | 77% |
  | PR #804 (open) | 21.8 | 204 | ~87% |

- **On the same scale, our 1-row window is 62.5% (whole window) and the kernels are close to ds4's.** Our streamers
  run near ds4's per-GEMV rates: x3ld ~240, head 235, attention dense 157-201 (ds4: 214-243 after repack, 150-195
  before). The difference is the fixed per-layer work that one GPU with 2.5x the bytes a layer amortises and two
  GPUs do not.
- **ds4-equivalent efficiency (~77-80%) would put our 1-row window at ~21-22 ms.** That is this plan's realistic
  target.

### 2.3 What ds4 measured as neutral, and why it fits our data

| ds4 note | why flat | our analogue |
| --- | --- | --- |
| "M5 triple fusion (qkv norm + KV RoPE + FP8 + raw store): bit-identical, -61 launches/token, flat" (#773) | the host runs ahead of an eager queue, so each removed boundary was ~1 us of a 50 ms token (~0.1%) | G9: 0.3 ms of in-graph idle for 1,616 boundaries; G10 PDL: no gain |
| "~300 launches moved into graph replay, flat"; graphs off on GB10 (#804, +0.5%) | launch latency was already hidden; replay adds submission cost | **the opposite for us:** Python cannot launch eagerly fast enough, and graph submission costs 1.27 ms a round of GPU idle, so we pay the submission and gain nothing back on the device |
| "smem staging for the ordered-chunk pair matvec: -3.6%, reverted" | an extra smem round trip on a load-bound GEMV | x3dn v1 / v2: persistent ring + partials lost 9-24% |
| "head-fused score-split finalize slower on GB10: parallelism-bound" | fusing a reduction serialised it onto fewer SMs | GLM 0130's last-program epilogues lost; GLM 0520 fused hc: 0.58x at R = 1, slower at R >= 6 |

**Rule taken into this plan.** Launch count does not count. Only these do:

- DRAM-idle critical-path latency removed;
- streamer GB/s recovered;
- host submission time (node count x ~0.8 us).

The fusion results that did pay here all fit this rule:

| change | measured | what it actually removed |
| --- | --- | --- |
| gemv router (G7) | 4.75 -> 1.26 ms | a 36 GB/s Triton fp32 dot replaced by a 115-200 GB/s CUDA GEMV |
| glue on (G9) | -0.8 ms | |
| L2PF (G10) | -0.6 ms | DRAM-idle windows filled with the next weights |

### 2.4 ds4 techniques, by transfer to TP=2 EXL3

| ds4 technique | its gain | transfer to us | candidate |
| --- | --- | --- | --- |
| **fewer dense bytes**: AProjQ4 (Q8 -> Q4 attention projections, PR #621) | +14-18% decode | Our dense attention is 5-bit EXL3: 1.74 GB a rank. At 4 bits that is 1.39 GB, **-0.35 GB = -1.5 to -1.7 ms** at today's dense rates. A quality lever, not a kernel lever; needs requantization (section 4, f1) | **f1** |
| **aligned SoA repack for 16-byte loads** (`ds4_cuda.cu:4531`) | +9-27% a GEMV (Q8 2048x4096 172 -> 218) | Upstream `linear_kernel` loads trellis words as 24 4-byte loads a lane a k step, each word read by ~2.4 lanes (DECODE-ROOFLINE 7): the same disease. x3ld's 16-byte path fixed it for experts (~240). x3dn put it on dense but lost on its reduction structure, not its loads | **f2** |
| **more warps in flight** (rows per block 8 -> 16: +25%; 512-thread Q8 blocks on GB10) | +25% (older base) | wq_b runs 1 CTA an SM (220 registers x 256 threads, 8 warps); the q group too. Little's law: 232 GB/s x ~1 us of loaded LPDDR5X latency needs ~4.8 KB in flight an SM. Fold into f2 | f2 |
| **one launch for all routed experts, gate+up+SwiGLU+router weight in the epilogue, warp per row, no split-K** | in their baseline | We split gate/up and down into two launches plus four small kernels. EXL3's output Hadamard (128-column blocks) needs a CTA-level epilogue, so it cannot be done a warp a row. Our version is a ticketed epilogue (b2) | **b** |
| persistent scratch, no per-call allocation (b42682f1) | +7.5% | already true: static graph buffers | none |
| weights in `cudaMalloc`, not the mmap (15f42aaf: 13.9 -> 16.1 t/s) | +16% | check only: fastboot reads into pinned buffers then device tensors. A 5-minute check of `weights.py` / fastboot in the next window | check |
| dedup of distinct experts across verify rows | 1.5-2x at overlap | already: x3ld runs the union of distinct experts | none |
| decode islands (per-layer graph pieces) | +1-2% | supports **d1** (split graphs) for a different reason: we need the GPU to start before the whole graph is submitted | d1 |
| `dp4a` with Q8_1-quantised activations | in baseline | none: EXL3 needs mma on decoded fp16 tiles; activations stay fp16 | none |
| TP=2 over RoCE | +14% only | Our exchanges are already 1.3 ms a window (81 x 7.5 us + wait); ds4 has nothing to teach here | none |

## 3. Megakernel / persistent designs, and what transfers to GB10 + host-staged RoCE

| work | design | reported gain and where it came from | transfer to 48 SMs, unified memory, host-staged TP=2 |
| --- | --- | --- | --- |
| Hazy "No Bubbles", Llama-1B ([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)) | per-SM instruction interpreter, 7 fused op types, 13 x 16 KiB smem pages handed op to op so the next op's weight loads start under the previous op, global counters | H100 78% of BW, ~2.5x vLLM / 1.5x SGLang on a **<1 ms** step. B200 time split: 250 us of 600 activation store / load + sync, 40 us warp sync: **~40% of a megakernel step is still sync and activation traffic** | Page pool needs ~213 KB smem; sm_121 has ~101 KB a SM (G1's overflows), so 5-6 pages. Gain scales with how short the step is: our step is 27 ms |
| Hazy TP Llama-70B ([blog](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)) | comms fused into instructions, NVLink storer warps | +22% throughput at batch 1K-8K; work queue 14%, interleaving 6.4%, pipelining 6.1%; **at batch 1K the work queue is neutral, interleaving <= 0.1%** | No low-batch latency numbers; NVLink stores do not exist for us |
| Mirage MPK ([arXiv 2512.22219](https://arxiv.org/abs/2512.22219)) | compiler -> SM-level task graph, 4 scheduler SMs, event queues; NVSHMEM signal waits for TP | 1.0-1.7x vs SGLang / vLLM on dense models; MoE layer 1.07-1.18x; ~0.8 us a launch under graphs on B200; comm overlap 1.1x | 4 scheduler SMs = 8% of 48. No EXL3 / CSA2 / mHC support. NVSHMEM's IB path needs GPUDirect; ours is a CPU proxy |
| Cohere MoE megakernel ([blog](https://cohere.com/blog/megakernels), Sep 2026) | 1 CTA an SM, 12 warps (controller, TMA producer, storer, 8 consumers), static wave order + drain queues, prefetches router / QKV weights under the previous O-proj tail | batch 1: 62% vs vLLM's 39% of BW (1.58x decode, 1.25-1.41x end to end), single H100; schedule order alone up to 19% | TMA and producer-warp designs need sm_90 features; sm_121 has no TMA tensor maps or wgmma. Single GPU only |
| Event Tensor / ETC ([arXiv 2604.13327](https://arxiv.org/abs/2604.13327)) | event tensors with wait counts encode tile dependencies, including data-dependent MoE ones | batch 1: 1.48x vLLM, 1.20x SGLang (Qwen3-30B-A3B); **TP=4: 0.99-1.06x**; MoE layer: **static schedule 1.03x, dynamic 0.95x at 1 token** | static order beats dynamic at decode: no work queues at 1-16 rows. Communication drowns the gain under TP |
| MonoMoE ([arXiv 2609.04244](https://arxiv.org/abs/2609.04244)) | weight-major persistent MoE: route + top-k + up + act + down + reduce, readiness flags, prefetch depth 2-4 | batch 1: **boundary gaps 0.24 ms vs bandwidth recovery 1.14 ms** (gaps < 20% of the win); MoE 1.17-1.54x; **DeepSeek-V3.1 TP=8: -1.2% to +3.7%** | Our x3ld already streams at ~240; their baseline was 10-21% of H200 BW. Weight-major helps only C4-sized unions. The flag chaining is the part to copy (b2) |
| ForgeMegakernel ([arXiv 2609.12379](https://arxiv.org/abs/2609.12379)) | dense megakernel | a graph-replayed launch still costs **4.63 us of device ramp / drain** (H100 fit); small RMSNorm / RoPE / KV-store kernels are 17-21% of a step; 1.21x SGLang | That ramp / drain is already inside our measured kernel times (the glue rows of 1.1), not extra |
| Perseus ([arXiv 2605.00686](https://arxiv.org/abs/2605.00686)), UCCL-EP | CPU-proxy GPU-initiated comm: batch signals per destination, NIC-side fences, 16-byte FIFO commands | proxy path matches GPU-direct once fences are batched | Our `tfroce` already works this way (pinned slots, doorbell, a busy-spinning proxy, system-scope flag polling, one flag per HCA stripe), so it drops into a persistent kernel as-is |

What transfers:

1. **The exchange is already kernel-native.** `tfroce::gather_kernel` stages the shard into pinned host memory, rings
   a doorbell for the C++ proxy (RDMA write + sequence flag) and polls the peer's flag with system-scope loads. A
   persistent kernel can do the same at its exchange points, with no GPUDirect needed. The host-staged path costs
   the same ~7.5 us either way. What a megakernel adds is only the ability to **keep streaming the next weights into
   L2 while it polls**: ~1.7 MB of DRAM time per exchange at 232 GB/s, bounded by L2 (24 MiB, 18 persisting).
2. **Static schedules, no scheduler SMs, no dynamic queues at 1-16 rows** (ETC: dynamic 0.95x at 1 token).
3. **Expect transfer to shrink under TP** (ETC TP=4 0.99-1.06x, MonoMoE TP=8 -1.2..+3.7%). Our 2 exchanges a layer
   are irreducible: the MoE input depends on attention's all-gathered output, and replicating attention would double
   its 1.74 GB.
4. **The L2 bound on cross-op prefetch is real and already measured.** L2PF has 3 sites a layer with ~30 MB of
   DRAM-idle opportunity. It modelled -0.4 to -0.6 ms (its own conservative model), but a cold -> L2-hot upper bound
   (2.86 us saved a MB) gives ~3.4 ms. It realised 0.6 ms. EXL3 kernels L2-hot run at 288-443 GB/s, not 690, so
   prefetch saves ~1.45 us a MB, not 2.86. A megakernel's "prefetch the next op's weights" lever is therefore worth
   **~0.5-1.5 ms more** at best, not the ~4 ms an L2 / DRAM ratio suggests.

## 4. The candidates

Effort is hours of work to an offline-complete change: CUDA / Triton source, numpy lane-level emulator, sm_121 compile
+ SASS checks (`TF_NVCC`, the pip CUDA 13.4 nvcc), CPU tests. "GPU" is the window time for bitwise GPU tests, cold
microbench, nsys and the A/B gate.

Numerics rule (G8 / DECODE-ROOFLINE 5.6):

- reduction orders may change (top-1 vs the kit >= 96%, MMLU within 1 point);
- row invariance, batched == alone and drafted == serial must hold;
- the exact prefill tag's rows must equal decode rows. So any decode kernel that the exact 2,048-row prefill segments
  also run must keep the same per-element arithmetic, or the exact tag moves with it.

### (a) Per-layer persistent "attention block" kernel

- **Scope.** From the mHC site's normed row to wo_b's partial in the RoCE send slot, in one launch a layer:
  - x group (wq_a + wkv [+ compressor]);
  - rms2 (q / kv norms);
  - q group (wq_b [+ indexer wq_b]);
  - RoPE, KV store;
  - on 8 layers: compressor pool + indexer scores + top-k;
  - attention chunks + merge + inverse RoPE;
  - o group (wo_a x 4), wo_b.
- **Dependencies.** qr needs the whole x group (grid-wide). Head h's attention needs only its 512 q columns (flag a
  head). wo_a group g needs its 8 heads (flag a group). wo_b needs all of wo_a (grid-wide).
- **Today, per layer:** ~20 launches.
  - dense 233 us at ~187 GB/s for 42.5 MB (floor 183 us);
  - attention glue ~45 us (attention core, norms, the q / kv side of copies);
  - top-k ~74 us on the 8 index layers.
- **What it removes:**
  - 4-5 dense ramp / tail pairs a layer (the fitted ~5 us fixed + SM imbalance of DECODE-ROOFLINE 7: ~45-50 us a
    layer is the dense gap);
  - the glue launches' dependent round trips (rms2 into the x group's last-arriver, RoPE + KV store into the q
    group's epilogue);
  - wo_a / wo_b streaming into smem / registers on idle SMs while the attention core runs on few CTAs.
- **ms.** Upper bound: dense gap 1.3 + attention glue ~1.2 of its 2.5 = **-2.5 ms**. Expected (GLM / x3dn transfer):
  **-0.8 to -1.6 ms**.
- **Effort ~150-250 h.** Dense EXL3 decode as resident virtual programs with upstream's (or x3seg's) per-program
  inner loop, phase flags, 1-16 rows, every CSA2 role (SWA-only layers 0-1, full / reindex / candidate source),
  CED's compressor pass.
- **Risk: high.**
  - x3dn v1 / v2 lost 9-24% doing the dense half of this.
  - Registers: x3seg is 141-220 a thread, so one resident CTA design must hold the attention core's fragment state
    beside the dense loop's: occupancy drops.
  - The exact prefill tag runs these kernels too (keep per-element arithmetic, or move prefill with it).
- **Offline:** the emulator for phase / flag ordering and per-element bits; the sm_121 compile.
  **GPU:** everything about speed.
- **Verdict.** Do it after (e1) and (b), and only as an assembly of their proven pieces.

### (b) Per-layer fused MoE block (router -> experts -> combine + shared) with cross-op prefetch

- **Today, per layer:** 7 launches.

  | kernel | us |
  | --- | ---: |
  | `rg::gemv` (3.9 MB: 17 us of it is DRAM) | 32 |
  | group | 7.5 |
  | rot_in | 5.3 |
  | x3ld gate/up | ~115 |
  | gateup_epilogue | 7.8 |
  | x3ld down | ~57 |
  | down_combine | 5.6 |

  The DRAM-idle chain is ~41 us a layer (router minus its stream, plus the four small kernels), plus ~10-16 us of
  ramp / tail at the gate/up -> down boundary and the router -> experts boundary.
- **Stage b1** (exact: the same arithmetic, so same bits):
  - The router kernel's last arriver, which already runs the top-k, also builds the group table and runs rot_in
    (5,120 values: the CTA has x in smem). That removes 2 launches and ~10 us a layer.
  - The shared expert gets its own x3ld launch on a forked graph branch that starts with the router. It does not
    depend on routing (DECODE-ROOFLINE lever 2, never built). Its ~5.7 MB (~25 us) streams under the router's
    latency, and the routed launch shrinks by one expert.
  - The combine adds shared + routed in slot order as today.
  - Expected **-0.4 to -0.9 ms**.
- **Stage b2:**
  - one launch for gate/up + SwiGLU + down's input rotation + down + combine;
  - per-expert readiness flags (the last gate/up CTA of expert e runs its 128-column Hadamard / SwiGLU epilogue and
    publishes e);
  - down items of e wait on e's flag while other experts' gate/up items stream;
  - the combine is done by the last down CTA in slot order (deterministic).
  - This differs from GLM 0130's failed "last-program epilogue": there the epilogue ran after the stream had
    drained. Here other experts' items are still streaming under it.
  - Expected another **-0.4 to -0.9 ms**.
- **ms.** Upper bound 40 x ~50 us = **-2.0 ms**. Expected **-0.8 to -1.8 ms** for b1 + b2.
- **Effort.** b1 ~50-70 h. b2 ~100-140 h: the x3ld inner loop is kept verbatim; the new parts are the item tables,
  flags and epilogue placement.
- **Risk.**
  - b1: low-medium. Graph fork / join is already used by L2PF; the router's ticket pattern exists.
  - b2: medium-high. GLM E1 (persistent ring, 160 vs 200 GB/s) and 0130 both lost. The gate: x3ld's in-situ GB/s
    must not drop (>= 235 at R = 1).
- **Offline:** emulators (the router's and x3ld's already exist), compile. **GPU:** cold microbench a layer, then
  the window.

### (c) Whole-window persistent megakernel with host-staged exchange points

- **What it adds over (a) + (b) + (e):**
  - the 0.3 ms in-graph idle;
  - the 1.27 ms a round of graph submission (one launch);
  - exchanges folded into producer / consumer phases (the gather kernel's own staging pass, ~2-3 us x 81 = ~0.2 ms);
  - prefetch of layer L+1's first weights while polling (~0.5-1.5 ms at best, section 3 item 4).
- **What it cannot remove:**
  - transfer (0.61 ms);
  - jitter wait (~0.7 ms, E|a - b| / 2 a barrier: GLM THEORY-2 1.4);
  - the per-layer dependent chain (2 exchanges + 2 mHC boundaries + router select + norms: ~40 x ~25-30 us = 1.0-1.2
    ms of irreducible latency even fused).
- **ms.** Upper bound vs today: ~10.1 - 1.3 (exchange) - 1.1 (chain) - ~0.8 (dense / expert residual) = **~-7 ms**
  a window, -1.3 a round of host. Expected **-3.5 to -5.5 ms a window, -1.2 a round**. Of that, **only ~1-2 ms
  beyond what (a) + (b) + (d1) + (e) already take.**
- **Effort ~500-900 h.**
  - Every layer kind (CSA2 roles, Engram layers 1 / 14 with the GPU-side flag wait, 8 index layers, compressor
    ratios) and every row count 1-16.
  - The DSpark pass (3 layers + head) as a second program.
  - A second implementation of every decode kernel to maintain beside prefill's.
  - GLM sized the same thing at 25+ days for +10-18% (PARADIGMS #12).
- **Risk: very high.**
  - Register pressure of one resident CTA serving EXL3 decode + attention + mHC.
  - ~101 KB smem a SM (Hazy's page pool needs 213 KB).
  - Debuggability: a hang is a wedged GPU on both ranks.
  - Literature transfer under TP is 0.99-1.06x (section 3).
- **Offline:** an interpreter-level schedule simulator + per-task emulators. **GPU:** everything about speed.
- **Verdict.** Not now. Build it as the assembly of (e1) -> (b) -> (a) tasks once those show >= 50% transfer.

### (d) Host round loop

**d1. Hide the window graph's submission (no language change).**

- `graph.replay` is 1.27-1.30 ms of GPU idle a round. That matches ~0.8 us a node x 1,616 nodes (MPK reports ~0.8 us a launch under graphs on
  B200).
- Options, in order of effort:
  1. **Split the window graph** into 4-8 pieces by layer range (ds4's "islands", for a different reason), replayed
     back to back. The GPU starts after the first piece's ~0.2 ms, and the rest submit under it. Expected **-0.9 to
     -1.1 ms a round**.
  2. **Pre-enqueue** the next window's graph during the DSpark pass, behind a device-side wait on a pinned flag (the
     Engram gate's flag kernel already does this inside graphs). The host writes the staged rows / tables, then flips
     the flag. Expected **-1.1 to -1.27 ms**. Needs the round's row count known before the pass ends, which it is
     not today: depth is chosen after the pass. So use the padded-rows bucket of the previous round and fall back to
     option 1 on a miss.
  3. Device-side graph launch (`cudaGraphInstantiateFlagDeviceLaunch`) from the DSpark pass's tail. Removes the
     host from the critical path entirely, but needs GPU-resident depth / acceptance (GLM 0450's "resident" mode,
     which measured +0.5-1.2%). Not now.
- Also: every fusion in (b) / (e) cuts node count, and so submission time, proportionally (~0.8 us a node).
- **Effort:** option 1 ~12-24 h (`rowgraphs` captures per layer range; static buffers already shared); option 2 +20 h.
  **Risk:** low; scheduling only, same kernels, same bits.
- **Offline:** CPU tests of capture keys / replay order. **GPU:** one nsys.

**d2. The round loop in C++ (nanobind) or Rust.**

- Addressable: Python's ~0.9-1.1 ms a round (prefetch issue 0.34, draft bookkeeping 0.31, candidates 0.27, commit /
  staging ~0.2). Some of it is the readback sync and the Engram reader waiting on the GPU, not interpretation.
- **Expected -0.2 to -0.6 ms a round** at C1 (+0.5-1.5%).
- GLM evidence:
  - NATIVE-ROUND moved prep / draw / accept into C++ (post-forward host time 231 -> 49 us greedy, 687 -> 78 us
    sampled, measured) and got **+0.33% at 1 stream, +0.76-1.27% at 4** (W24): not adopted.
  - GPU-ROUND (0450, device sampling) got +0.5-1.2%.
- Rust has no advantage over C++ here: the work is ATen calls, numpy-exact draws and the RoCE / Engram extensions,
  which are already C++.
- **Effort ~60-120 h** (rounds / decode / spec / nucleus / pick / depth: ~2,200 lines, with numpy-exact sampling
  ported, as GLM did). **Risk:** low-medium (exactness of the draw; both ranks in lockstep).
- **Verdict:** only for C2 / C4 (host idle ~1.8 ms a round there: commit 0.58, draft 0.58, prefetch 0.45, nucleus
  0.23), and only after d1.

### (e) Hand-CUDA rewrites of the glue

| item | today (1 row) | floor | CUDA design | expected | effort | risk |
| --- | --- | --- | --- | --- | --- | --- |
| **e1 mHC boundary**: post + site + finish (+ the next sublayer's RMSNorm) | site ~26 us + finish ~11 us, x 80 = **3.0 ms** | `fn` bf16 1 MB = 4.5 us a boundary + a 20-iteration Sinkhorn on 4x4 in one warp (~2-3 us) | one launch a boundary: every CTA loads its column block of the 4 streams + both ranks' partials (consumer side: straight from the RoCE recv slot) and its `fn` rows with 16-byte loads; per-row FMA chains in a fixed order (no `tl.dot`, so no 16-row padding at R = 1); partials to a ticketed last CTA that runs finish + the norm and writes the collapsed row | **-1.2 to -2.0 ms** (target ~12-15 us a boundary; GLM 0520's CUDA hc boundary measured **0.58x at R = 1**, which here is 37 -> ~21 us = -1.3 ms) | 40-70 h | low-medium: GLM 0520 lost at R >= 6. Keep the Triton path above R = 4 (a row-count switch is allowed: row invariance holds per row) |
| **e2 attention core**: RoPE + KV store + `_chunks` + `_merge` + inverse RoPE | 1.2 ms (`_chunks` 21 us, `_merge` 2.2, `_kv_store` 2.9, `_rope` 1.6, Engram `_fuse` 2 x ~100 us) | ~0.1 ms of KV bytes | split-KV flash-decode, 16-head tile x chunk CTAs (as today), merge by the last CTA of a (row, head tile) in chunk order; RoPE / KV store in the prologue of the q group's last arriver (or of (a)). Engram `_fuse` (G9's `engram_dec` already restructured): leave | -0.3 to -0.5 ms | 40-60 h | low-medium: chunk partial bits must stay row-invariant |
| **e3 indexer top-k** (8 index layers) | torch `gatherTopK` + `radixSortKVInPlace` ~0.59 ms (G9 dtopk already in), `_scores` / `_plain` 0.17 | tiny | one kernel: scores into smem / L2, a two-pass radix select of the top-512, write sorted ids; fused with the `weights_proj` (`_plain`) GEMV | -0.3 to -0.5 ms | 30-50 h | low: the selection must keep ties-to-lower-id |
| **e4 copies / elementwise** | ~0.59 ms (~200 torch copies / casts / cats) | 0 | fold into producers: outputs written at final offsets and dtypes | -0.3 to -0.5 ms | 10-20 h (mostly Python) | low |
| e5 router | 32 us vs 17 floor, x 43 = 1.37 ms | 0.73 | part of b1 (selection + group + rot_in fused, shared expert concurrent) | (in b) | | |

- **(e) total:** upper bound ~-4.5 ms, **expected -2.1 to -3.5 ms**.
- **Offline:** everything but timing (the repo's emulator + compile pattern). **GPU:** bitwise suites, cold
  microbench in a graph after a 50 MB predecessor (GLM 0520's harness), window A/B.

### (f) Adopting ds4's techniques

| item | ms | effort | risk | offline / GPU |
| --- | --- | --- | --- | --- |
| **f1 4-bit dense attention** (ds4 AProjQ4: +14-18%): wq_b, wo_b, wo_a, wq_a, wkv from 5 to 4 bits (layer 0 stays 6) | 1.74 -> 1.39 GB: **-1.5 to -1.7 ms** at today's 204 GB/s; also -1.4 GB of weights a node | 20-40 h of tooling + GPU quantization hours | **quality**: gated like prune (top-1 >= 96%, MMLU within 1, tool chains >= 10 / 12). Needs the source (BF16 / FP8) checkpoint and the EXL3 quantizer; the pack is the kit's `2.9bpw` | quantization and the gate are GPU; the code change is none (widths are read from the trellis shape) |
| **f2 dense EXL3 GEMV, ds4-style load geometry**: no split-K at R <= 4, 16-byte `ld.global.nc.v4` straight to registers (no smem ring, no partials), enough warps an SM (2+ CTAs of 8 warps; registers capped), a strip a warp group | dense 10.5 -> ~9.5-9.8: **-0.7 to -1.0 ms** | 40-60 h | **medium-high**: x3dn v1 / v2 both lost (their partials, atomics and rings were the cost; this design has none of them). GLM's plain-read probe reached 235 GB/s at 25% occupancy with 4 independent 16-byte loads a thread: that is the mechanism | offline emulator + SASS (LDG.E.128, no LDL / STL); GPU cold bench vs upstream a shape, then the window |
| f3 weights in device memory, not mapped | 0 or large | 1 h | none | GPU check |

### (g) Graph-level concurrency (not a rewrite; listed because it is the cheap half of several)

- **Fork / join branches in the captured graph** (L2PF already forks a side stream inside graphs):
  - shared expert || router (this is b1's second half);
  - indexer branch (`weights_proj`, `_scores`, top-k) || the main q path (wq_b -> RoPE);
  - compressor || attention core on kv-source layers.
- The streamers do not add bandwidth to each other. What is gained is the latency-bound branch hiding under a
  streamer.
- **Expected -0.4 to -1.0 ms.** **Effort** 16-30 h. **Risk:** low (same kernels, same bits). Watch the shared
  counters: one group must not run on two streams at once.

## 5. Ranking

Ordered by expected ms a round per hour, adjusted for risk and dependencies: f1 waits on the source checkpoint, and
b2 builds on b1. Each "expected" range is the discounted one; the cumulative column assumes everything above it is
in (mid of each range).

| rank | candidate | expected ms a round | effort h | ms / 10 h | risk | cumulative 1-row window (26.9 today) |
| ---: | --- | --- | ---: | ---: | --- | ---: |
| 1 | **d1** split / pre-enqueued window graph | -0.9 to -1.2 (round, not window) | 12-24 | ~0.6 | low | 26.9 (round -1.0) |
| 2 | **e1** mHC boundary in CUDA | -1.2 to -2.0 | 40-70 | ~0.3 | low-med | ~25.3 |
| 3 | **b1** router -> select -> rot_in fused; shared expert concurrent | -0.4 to -0.9 | 50-70 | ~0.1 | low-med | ~24.7 |
| 4 | g (indexer / compressor branches) + e4 (copies) | -0.5 to -1.0 | 26-50 | ~0.2 | low | ~24.0 |
| 5 | e3 indexer top-k + e2 attention core | -0.6 to -1.0 | 70-110 | ~0.09 | low-med | ~23.2 |
| 6 | **f1** 4-bit dense attention (quality-gated) | -1.5 to -1.7 | 20-40 + GPU quantization | ~0.5 | quality | ~21.6 |
| 7 | b2 gate/up -> down with flags | -0.4 to -0.9 | 100-140 | ~0.05 | med-high | ~21.0 |
| 8 | f2 dense GEMV load geometry | -0.7 to -1.0 | 40-60 | ~0.17 | med-high | ~20.2 |
| 9 | d2 C++ round loop | -0.2 to -0.6 (C1); ~-1 at C2 / C4 | 60-120 | ~0.04 | low-med | |
| 10 | a attention block | -0.8 to -1.6, much of it overlapping e2 / f2 / g | 150-250 | ~0.05 | high | |
| 11 | c megakernel | ~-1 to -2 beyond all of the above | 500-900 | ~0.02 | very high | |

- **Without f1** (exact quality, ranks 1-5): window ~23.2 ms + round -1.0, so D ~4.7 ms. That is prose ~47 (+14%),
  code ~89 (+11%).
- **With f1, b2 and f2:** window ~20-21 ms, D ~6.9-7.9 ms. That is prose ~50-52, code ~93-95, at the ds4-class ~80%
  of floor.

Rank 6 (f1) sits low only because it needs the source checkpoint and a quality gate. By ms per hour it is the second
best lever on the list.

## 6. The top 3 to start now, with a first milestone each

### 1. d1: the window graph starts before it is fully submitted

- **Change.** `rowgraphs.RowGraphs` captures a window as N layer-range pieces (the embedding + layers 0-4, then
  5-9, ...; the head + candidates last) on the same static buffers and pool, replayed back to back.
  `TF_DSV41_GRAPH_PIECES` (1 = today). Engram layers 1 / 14 keep their GPU-side flag waits.
- **Milestone 1 (offline, ~1 day):**
  - CPU tests: piece capture keys, replay order, and that the pieces' kernel sequence equals the one-graph sequence
    on the interpreter path;
  - the boot warm-up budget (15 s) still holds with N x graphs (memory: the same pool, so no new buffers).
- **Milestone 2 (GPU, 30 min):**
  - nsys p1 with `PIECES` 1 / 4 / 8: `graph.replay` GPU idle 1.27 -> **<= 0.35 ms** a round;
  - the window's in-graph idle does not grow by more than 0.1 ms;
  - prose 1-stream +2% or better;
  - replies equal (same kernels).
- **Then:** pre-enqueue (option 2) if pieces leave > 0.3 ms.

### 2. e1: the mHC boundary as one CUDA kernel for 1-4 rows

- **Change.** `mhc_cuda.cu`: post (both ranks' partials in rank order) + site (the 4 streams' mixes against `fn`
  bf16) + finish (Sinkhorn) + the next sublayer's RMSNorm. One launch a boundary, a fixed reduction tree a row
  (row-invariant), the Triton `_site_dec` / `_finish_k` path kept above 4 rows. `TF_DSV41_MHC_CUDA`.
- **Milestone 1 (offline, ~3-4 days):**
  - kernel + numpy lane-level emulator;
  - drift vs the Triton path <= fp32 rounding x chain length (asserted), row invariance (1 alone == in 2 / 3 /
    4-row windows);
  - sm_121 compile with registers / spills / SASS checks.
- **Milestone 2 (GPU):**
  - cold in-graph microbench after a 50 MB predecessor: **<= 15 us a boundary at R = 1 (today ~37)**, <= today's at
    R = 2-4;
  - then the window: 1-row verify **<= 25.7 ms (-1.2)**;
  - gate: top-1 >= 0.97, MMLU within 1.

### 3. b1: the router chain and the shared expert off the critical path

- **Change.**
  - `router_gemv.cu`'s last arriver also writes the group table (`group_kernel`'s output) and the rotated input
    (`rot_in`'s), so 2 launches go.
  - `experts.py` splits the shared expert into its own x3ld launch on a forked branch that starts with the router
    and joins before `down_combine`.
  - Both under `TF_DSV41_MOE_CHAIN`.
- **Milestone 1 (offline, ~3 days):**
  - emulator: the fused router's group table and rotated input equal today's bit for bit (same arithmetic, so this is
    exact);
  - the shared expert alone + the routed union == today's single launch's Z for every row (the per-expert
    arithmetic is independent of the table entry);
  - graph fork / join test with the per-group counters.
- **Milestone 2 (GPU):**
  - nsys: the MoE chain's DRAM-idle time a layer **58 -> <= 35 us**;
  - 1-row verify -0.4 ms or better;
  - replies equal.
- **Then:** b2 (flags + fused epilogues), gated on x3ld's in-situ GB/s not dropping.

**In parallel and cheap:** f3 (the device-memory check) and f1's prerequisite: find out whether the source checkpoint
and the EXL3 quantizer are available for 4-bit attention. f1 is the second-best ms per hour on the list and needs no
kernel work.

## 7. Triton vs hand CUDA for our small kernels on sm_121

Where Triton's codegen loses, from this repo's and GLM's measurements:

1. **`tl.dot` needs M >= 16.**
   - A 1-row decode pads to 16 rows, and the exact paths used ieee fp32 dots.
   - The old router: 36 GB/s in Triton vs 115-200 GB/s as a CUDA warp-per-expert GEMV; **4.75 -> 1.26 ms
     measured**, the clearest Triton -> CUDA win here.
   - mHC `_site` phase B still runs `tl.dot` over 16-row tiles for 1-row windows. e1 removes it.
2. **No fragment-level control.**
   - EXL3 trellis decode cannot be expressed as an mma operand.
   - x3tc (Triton experts) went through shared memory every k step and was 3-5x slower than the CUDA grouped kernel
     (M1-STATUS, prefill experts).
3. **Grid-level patterns.**
   - Ticketed last-arriver epilogues, readiness flags and multi-phase reductions are awkward. GLM's Triton fused hc
     boundary was bitwise but **7-8x slower** (`num_stages = 1`, a multi-phase grid reduction) and overflowed smem
     (131 KB > 101 KB).
   - The CUDA version of the same boundary measured 0.58x of the 3-kernel time at R = 1.
4. **sm_121 has no wgmma and no TMA tensor maps.**
   - Triton's Hopper / Blackwell pipelines do not apply. What is left is `mma.sync` + `cp.async`, and Triton's
     pipeliner does not stage irregular gathers (trellis words, KV rows by page table).
   - CUDA can issue 16-byte `ld.global.nc` / `cp.async.cg` rings by hand (x3ld: ~240 GB/s).
5. **~101 KB of smem an SM.** Configs written for 228 KB parts overflowed: the router's and CSA2's smem overflows
   fixed in G1.
6. **Codegen drift across Triton versions.** GLM found Triton 3.8 leaving 16-36 epilogue products unfused, which
   changes bits a row bucket. Under the relaxed gate this is a row-invariance risk, not a quality one: pin Triton or
   own the kernel.

Where Triton is fine, and CUDA gains only through fusion:

- elementwise / norms (`_rms` 5 us, `_rope` 1.6 us), where a launch costs the same in either language;
- the attention `_chunks` tile (16 heads = M 16 on tensor cores, 21 us);
- the indexer `_scores`.

These move into CUDA only as parts of e1 / e2 / e3, not as one-for-one ports.

| kernel | today (ms a 1-row window) | Triton's limit | CUDA rewrite | expected |
| --- | ---: | --- | --- | --- |
| mHC `_site_dec` + `_finish_k` | 3.0 | 16-row `tl.dot` padding; 2 launches + a global round trip between them | e1 | -1.2 to -2.0 |
| attention `_chunks` / `_merge` / `_rope` / `_kv_store` | ~0.8 (+ Engram `_fuse` 0.2-0.4) | fine per kernel; 4 launches with dependent round trips | e2 (fused split-KV) | -0.3 to -0.5 |
| indexer top-k (torch) + `_scores` / `_plain` | 0.76 | torch's generic top-k / sort | e3 | -0.3 to -0.5 |
| norms (`_rms`, `_pool_norm`) | 0.27 | launch-bound | fold into e1 / producers | (in e1 / e4) |
| router (already CUDA) | 1.2 | | b1 | (in b) |

## 8. What would change this plan

- **A prod-config nsys (1 stream prose + code, both ranks)** would replace section 1.1's reconstruction. That is a
  30-minute window step (`G9.sh` profiling with prod knobs). If the expert class is not ~7.9 ms (prune's real
  effect), b moves up or down.
- **If d1's pieces do not remove the 1.27 ms**, the cost is not submission (for example a sync inside `replay()`).
  Then it is pre-enqueue or device launch.
- **If e1 transfers at >= 70%**, the same template (one CUDA kernel for a whole latency-bound boundary, consumer-side
  RoCE read) goes to e2 / e3 first, before any persistent dense work.
- **If f1 passes the quality gate**, it is worth more than b2, f2 and a together, at a fraction of their effort.
