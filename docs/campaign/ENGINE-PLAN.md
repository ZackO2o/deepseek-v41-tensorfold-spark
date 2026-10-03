# DeepSeek-V4.1-Flash as a TensorFold CUDA family on our 0.6.0 branch: the engine plan (2026-10-01)

> Offline work: no GPU was used; the Sparks were read only (config.json, safetensors headers, the kit's packed Engram
> files and NVMe block sizes on head). Inputs: glm53 repo docs DSV41-BASELINE, NEXT-DEEPSEEK-V41-FLASH, PORT-050 /
> PORT-060, UPSTREAM-CONTRIB-PLAN, ROOFLINE, DECODE-KERNELS-2, MLA-EXPAND, BOOT; our branch `glm-spark-stack-060`
> (`<engine checkout>`); upstream TensorFold 0.6.0 (`c464617`, `<upstream 0.6.0 checkout>`); vLLM's
> `models/deepseek_v41` (Apache-2.0, the math reference); this repo's [TARGETS](TARGETS.md),
> [ARCH-LEVERAGE](ARCH-LEVERAGE.md) and [research/LANDSCAPE](LANDSCAPE.md). Scope: V4.1-Flash only;
> every comparison is against V4.1 setups (the Mia 2.9 bpw kit on head, tonyd2wild's 4-Spark recipe, others in
> LANDSCAPE).

## 0. Bottom line

- **The family**: `tensorfold/families/deepseek_v41/` on our branch `glm-spark-stack-060`, a second CUDA family
  beside the GLM Spark engine, built from **upstream 0.6.0's shared modules** (EXL3 experts / linear / prefill GEMM,
  the communicator, exact sampling, build, server pieces) and **our GLM Spark engine's serving stack through its
  interfaces** (batcher plans, KV pool, session store + NVMe tier, RoCE, memory safety, drafting policy, server).
  New code: the V4.1 forward (CSA2, Engram, CED, Single-Pass mHC, DSpark, ViT). This repo stages it:
  `engine/kernels` (kernels), `engine/serving` (the family's seams), `engine/reference` (the torch oracle, in progress
  by the parallel track).
- **Implemented here** (section 10; 187 CPU tests, 0 GPU): the EXL3 mul1 expert layer for this model's shapes on
  upstream's module (TP=2 split proven exact), **our 0580 load path generalised to upstream's grouped kernel** (every
  width, same bits, sm_121 build with 0 spills), the **CSA2 kernels** (FP8 row format = the kit's SM12x record,
  compressors, row stores, indexer with candidates and reindex, sparse + SWA attention with sinks, and the
  **streaming score + top-k** for 300K prefill), **Single-Pass mHC**, **Engram** (host hash + prefetch, dequant,
  fusion), **DSpark's head** (Markov + confidence chain with keyed draft noise, block glue, the drafted == serial
  argument and its test), the **router** (row-invariant, interpreter-tested, 39 sm_121 compiles with 0 spills), and
  the serving seams (topology, pool layout, slot state, sessions, the Engram NVMe reader, the memory budget, the
  family / engine / drafter contracts).
- **Memory at 2.9 bpw, 4 active streams x 300K, FP8 pool: floor 5.1 GiB on rank 0, 6.0 on rank 1** (section 5),
  inside the user's 4-6 GiB band; 3.0 bpw experts (Mia-style or coolbho3k-style) do not fit that floor at 4 x 300K.
- **Round 1 = fast boot + blanket wins** (section 1): restart **~45-60 s** from prepared folders (kit: ~6.5 min),
  then code **~52-67 tok/s** and prose **~38-44 tok/s** single stream at round 1's efficiency (65% of the floor):
  **1.4-1.8x the independent kit numbers (code 36.4 / prose 25.1)**; the 2x targets of TARGETS.md need round 2's
  efficiency work (75% of the floor) and depth policy.
- **Milestones**: M1 exact TP=2 forward (top-1 >= 99% vs the kit's `prompt_logprobs`), M2 DSpark decode (drafted ==
  serial), M3 serving (batcher, pool, sessions / NVMe park, 4 x 300K), M4 RigMark parity+ (section 11). **~38-52 agent
  days, 7-9 GPU windows** (section 12).
- **Blockers** (section 13): GPU bitwise runs of the new kernels (G1), the native Engram reader for prefill, fast
  mul1 prefill expert GEMMs, every window needs GLM production down. (Written since: the streaming top-k, mHC,
  Engram and DSpark kernels.)

## 1. Round 1: iteration speed first (decision, 2026-10-01)

Round 1 delivers what makes every later window faster, then the wins that lift every workload. Workload-specific
work (CED replay as the default, prose drafter work, tool-calling polish, vision) comes in round 2+.

### 1.1 Fast boot (built in M1, so the first bring-up window already uses it)

Our GLM 0140 design (docs/BOOT.md in the glm53 repo; GLM restarts in 25-40 s), carried over:

| Phase (a rank) | How | Estimate |
| --- | --- | ---: |
| launcher | both ranks started at once, readiness polled every 1 s, `docker rm` in parallel | ~2 s |
| container + imports | torch / triton warm, extensions and Triton kernels from the persisted cache volume (`tensorfold.cuda.build` cache, `/cache/nv/ComputeCache`); **none rebuilt** after the first start | ~3 s |
| NCCL + RoCE | rendezvous, the collective RoCE probe (0230 / 0350), fallback marker check | ~3-5 s |
| page-cache drop + slot check | `drop_caches` on both nodes, MemFree / MemAvailable gate (the 0550 preflight), the floor check of section 5 | ~1-2 s |
| **weights** | the **prepared per-rank folder**: split, EXL3 words in our layouts, DSpark, router / norms / mHC / sinks, Engram `wkv` + token map + primes + hash multipliers, as stored bytes; 8 readers x 64 MiB O_DIRECT chunks into pinned buffers, async H2D; no page cache, no conversion; ~106 GB a rank at the GLM rate (82 GB in 11.8 s) | **~15-17 s** |
| Engram shards | header + a sampled row checksum of the packed node-local files (`engram-l{1,14}-r{rank}of2.bin`, already packed on both nodes by the kit's `./start.sh pack`); no NFS, never page-cached | <1 s |
| caches, rings, pool | the KV pool (section 5) and slot rings allocated once | ~1-2 s |
| CUDA graphs | verify widths 1-6 and 8 / 12 / 16 for 1 slot and the 4-slot mixes, the DSpark pass, the 128-row CED replay shape; a **15 s warm-up budget**, the rarer shapes captured on first use | ~12-15 s |
| calibration | cached per (image, knobs, engine shape, GPUs, clock caps): round costs C(R) for the depth policy | ~1-2 s |
| session store + NVMe tier attach | index reconcile (0250) | ~1-3 s |
| **total** | | **~45-60 s** (target <= 60 s, TARGETS: <= 2 min) |

- First start after an image / pack / layout change: the old path (read the node-local checkpoint, split, write the
  folder): ~2.5-3.5 min one time, or `scripts/prepare.sh` ahead of the window.
- The kit: ~6 min 30 s (179 s weights + 25 s DSpark from EXT4 with rsync, init, graphs, warm-up). **~6x faster
  restarts**: each GPU window gains ~5 min per restart, and A/B windows restart 4-8 times.

### 1.2 The blanket wins (in M1-M3, on from their first window)

| Win | Where it acts | Source | Expected |
| --- | --- | --- | --- |
| Grouped mul1 experts (upstream `cuda/exl3`) + **our 0580 load path** (`x3ld`) | 91 GiB of routed experts, every round | upstream 180-225 GB/s on GB10; 0580: in-situ routed decode -4.3% on GLM (W19) | expert time at 215-225 GB/s |
| **RoCE one-shot all-gather** (0230 / 0350) for the ~90 exchanges a round (2 a layer + Engram + head + DSpark) | every round | NCCL 24-29 us vs 11-27 us at 16-128 KiB; GLM W9 +7% decode | ~1 ms a round |
| CUDA graphs per verify width and slot mix | every round | the kit and ours | host overhead out of the round |
| **Exact DSpark** + cost-derived depth (0071) with the confidence head + **suffix lookup drafts** (0020) | every decode | user's session 3.11 tokens a round at static k=3; coolbho3k lookup 47.6 vs ~31 on edits | +10-25% vs static k=3 (workload-dependent) |
| Engram reads off the critical path (section 6) | every round, every prompt | tonyd2wild 2.8 ms a step unhidden | -2.8 ms a round (~5%) |
| FP8 pool + NVMe session tier, 4 x 300K active (sections 5, 8) | long contexts, multi-turn | GLM 0290 / 0250 / 0540 | replay TTFT <= 0.25 s, park / resume <= 1 s |
| Memory floor 4-6 GiB by accounting, not margin (section 5) | everything | GLM 0550 + watchdog | 4 x 300K fits at 2.9 bpw |
| SwiGLU clamp fused (upstream's gate/up epilogue already does it) | experts | tonyd2wild +3% | included |

### 1.3 Round 1 speed vs the Mia kit, with the reasoning

The round model is TARGETS 3.1's (bytes a verify per rank on the 2.9 bpw pack; measured distinct experts U(R)),
evaluated at **round 1's efficiency, 65% of the 230 GB/s floor** (between the kit's ~59% and our GLM's ~70%: the
round-overhead work of round 2 is not in yet), plus the DSpark pass (~5.6 ms) and ~1 ms of host and Engram slack:

| Workload | Verify rows R | Floor | Round (65%) | Tokens a round | Round 1 tok/s | Kit, independent (helge) | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Code, k=5 | 6 | 39.2 ms | ~67 ms | 3.5-4.5 | **~52-67** | 36.4 | **1.4-1.8x** |
| Prose, k=2 | 3 | 29.3 ms | ~52 ms | 2.0-2.3 | **~38-44** | 25.1 | **1.5-1.8x** |
| Structured, k=5 + lookup | 6 | 39.2 ms | ~67 ms | 5.0-5.5 | ~75-82 | (kit README 40, pre-moe_x) | ~2x |

Why faster than the kit on the same bytes, item by item:

- **Efficiency** 59% -> 65%: 0580 loads on the experts, RoCE instead of NCCL all-reduce, graphs for every width,
  Engram reads hidden; the kit's measured 60 ms at R=4 is 59% of its floor.
- **Depth**: the kit runs a static k=3; ours picks k a round from the confidence head and measured costs (code k=5,
  prose k=2), and lookup drafts extend copy-heavy rounds.
- **Acceptance at temperature**: helge's 36.4 / 25.1 sit well below the kit README's 56-57 / 32-66, and sfxnz
  attributes the kit's low acceptance to its 4-bit drafter (LANDSCAPE); the protocol details of helge's run are not
  published, so part of the gap may be content. What our engine changes regardless: rejection sampling is replaced
  by keyed sampling that shares the target's noise with the drafter at every position (GLM DFlash2's rule), which
  keeps acceptance at T > 0 close to greedy's. W0 measures the kit's acceptance per phase on our pair.
- **Exactness costs nothing here**: row-invariant kernels are the fast kernels (section 10).

TARGETS' 2x (code 72, prose 45) needs round 2: efficiency toward 75% (fused epilogues, L2 prefetch 0460, the dense
size switch, the attention gather once per CSA2 group from ARCH-LEVERAGE section 4) and the depth policy tuned on our
measured acceptance. Prose at 45 is at the edge (2.0-2.3 tokens a round); the stretch needs a better drafter
(on-policy DSpark re-fit, MIT) or fewer bytes.

## 2. The family on the branch

```
src/tensorfold/families/deepseek_v41/
  __init__.py          MODEL_TYPES = ("deepseek_v41",), check, cuda_engine, CUDA_APP / CUDA_SERVE (lazy), CUDA_KV_DTYPES
                       (staged: engine/serving/family.py)
  cuda/
    engine.py          Dsv41Engine: GlmEngine's contract (generate / follow / eos / limit / request / batch / store)
    forward.py         the 40-layer round and prefill pieces; Forward protocol (engine/serving/engine.py)
    topology.py        layer roles from config.json (engine/serving/topology.py)
    weights.py         loader of the EXL3 pack: per-tensor K from the trellis shape, TP=2 splits, the prepared folder
    fastboot.py        GLM 0140's prepared-folder reader / writer, keyed by the weight code's digest
    experts.py         routed + shared experts on tensorfold.cuda.exl3 (engine/kernels/exl3/experts.py)
    x3ld.cu/.cpp       the 0580 load path over upstream's grouped kernel (engine/kernels/exl3/)
    router.py          sqrt-softplus top-6 (engine/kernels/router.py)
    csa2/              rows, compress, index, attn (engine/kernels/csa2/)
    mhc.py             Single-Pass mHC (GLM 0320 / 0520 kernels, coefficient timing shifted a block)
    engram.py          hash (host, outside graphs), NVMe reader (engine/serving/engram.py; native reader later),
                       dequant + gate + fusion kernels
    dspark.py          the 3 DSpark blocks, Markov head, confidence head, keyed draft noise
    state.py, pool.py, sessions.py      slot state, pool families, snapshots (engine/serving/)
    batch.py           the Batcher on GLM's batchplan / Job / Seq protocol, V4.1 compute_multi
    app.py             the V4.1 app on our server: template (DeepSeek encoding, MIT), DSML tools, effort, images
    vision.py          DeepSeek-ViT tower on rank 0 + GLM 0500's host side
```

- Knobs follow upstream's style (`TF_DSV41_*`, CLI flags `--context`, `--kv-dtype fp8`, `--parallel`); the GLM
  deployment's `GLM53_TF_*` stay GLM's. The family needs no change in upstream modules beyond PR 1 (the
  communicator interface and `CUDA_SERVE`, already on the branch).
- Licences: TensorFold 0.6.0 is Apache-2.0; our GLM engine code is MIT (Jay Leaton); vLLM's model code is the
  Apache-2.0 *math* reference (no code copied: our kernels and their docstrings cite it); DeepSeek's encoding is MIT;
  **nothing from the kit's AGPL overlay** (only file-format facts of its packed Engram shards, section 6).

## 3. What is reused, through which interface, and what is new

| Component | Status | Interface / source | V4.1 specifics |
| --- | --- | --- | --- |
| EXL3 grouped routed experts (mul1, mixed widths) | **reuse upstream** | `tensorfold.cuda.exl3.experts` (`prepare`, `Scratch`, `routed`, `group`) | shapes 5,120 <-> 1,152 a rank; shared expert as table entry 384; `ACT_F32`, limit 10 |
| 0580 expert load path | **reuse ours, generalised** | `engine/kernels/exl3/x3ld.cu` over upstream's `experts_grouped.cuh` | every width K2 2-16, nt / pd settings, PDL, probe 3 |
| EXL3 dense linear (attention K5, shared, head K6, indexer, compressors, Engram wkv) | **reuse upstream** | `tensorfold.cuda.exl3.linear.Exl3Linear` (row-invariant, 1-128 rows, `out_dtype` fp32 for the compressors) and `prefill.matmul` (fast prompt GEMM, own tag) | grouped `wo_a` = 4 of the 8 `wo_a.slice.N` a rank |
| Communicator, RoCE | **reuse ours** | `tensorfold.cuda.comm` (`fast_gather`, `swap`, `check`), `roce.select` (0230 / 0350) | ~90 exchanges a round, <= 40 KB a row |
| Exact sampling | **reuse upstream** | `tensorfold.engine.exact_sampling` (keyed: seed, position, token) | vocabulary 129,280, half a rank |
| Batcher protocol | **reuse ours** | `batchplan` (header / plan codecs, pickers, fairness, spills), `Job` / `Seq`, the plan / follow protocol, cancellation, the idle doorbell (PORT-060 3.1) | V4.1 `compute_multi`, per-slot rings |
| Multi-slot prefill | **reuse ours** | `mpf` (0560): grouped experts pass across slots' pieces | pieces on the 16-token grid |
| KV pool | **reuse ours** | `kvpool` (0290: `PoolBook`, `SlotPages`, `Paged`, `_prow`) | families of section 5 (`engine/serving/pool.py`) |
| Session store, NVMe tier, prefix share, replay point | **reuse ours** | `sessions` (`SessionIndex`, `Plan`, `SessionStore`), `sessdisk` (O_DIRECT tier, compat ident), 0310, 0540 | snapshot contents and tags (`engine/serving/sessions.py`) |
| Memory safety | **reuse ours** | `memsafe` (`view`, `admit_ok`, `AdmitLog`, `Trimmer`), 0550 sized scratch | floor 4-6 GiB (section 5) |
| Drafting policy | **reuse ours** | `depth` (cost-derived depth), `lookup` (suffix drafts), `deep` (verify width <= 16), calibration | DSpark's confidence head as the acceptance input |
| Server | **reuse ours** | `glm5_next/spark/server.py` via `CUDA_SERVE` (0150-0620: health, metrics, request log, context errors, cancellation, structured output, tool fixes) | V4.1 app: template, DSML parser, effort |
| Vision host side | **reuse ours** | 0500's `vision_prep` (fetch, digest, virtual ids, `expand`) and `vision` (`Table`, `active`, `exchange`) | the DeepSeek-ViT tower is new |
| Fast boot, ops | **reuse ours** | 0140 `fastboot`, calibration cache, watchdog, canary, gpuwatch, `serve.sh`, lease, restore-prod | new probes and alert thresholds |
| CSA2 (compressors, indexer, candidates, sparse + SWA attention, sinks) | **new, adapted from our latent / DSA kernels** | `engine/kernels/csa2` | section 10 |
| Router | **new** | `engine/kernels/router.py` | sqrt-softplus, bias selects only, x 1.5 |
| Single-Pass mHC | **adapt** (GLM 0320 / 0520) | | coefficients from the previous block |
| Engram | **new** | hash (reference), reader (`engine/serving/engram.py`), dequant + gate kernels | section 6 |
| CED | **new** | prefill modes, `Forward.finish_prompt` | section 8 |
| DSpark | **new** (drafter), **reused** policy | `engine/serving/drafting.py` | section 7 |
| Not used | | KDA, DFlash2, MLA absorb / expand (0390), GLM q4 non-expert layouts | |

## 4. TP=2 sharding

| Tensor (checkpoint name) | Split | A rank holds | Exchange |
| --- | --- | --- | --- |
| `attn.wq_a` (5,120 -> 1,280), `attn.wkv` (-> 512), `compressor.wkv/wgate`, `indexer.wq_b`, `indexer.wk`, `indexer.weights_proj`, norms, sinks, `hc_*` | replicated | all | none (each rank computes the latent, the SWA row, the compressed row, the index keys and the selection: the single KV head cannot be split by heads) |
| `attn.wq_b` (1,280 -> 64 x 512) | heads (columns) | 32 heads | none |
| attention core | heads | 32 of 64 query heads over the whole rows | none |
| `attn.wo_a.slice.0-7` (grouped: 8 heads -> 1,024 each) | groups | slices 0-3 / 4-7 | none |
| `attn.wo_b` (8,192 -> 5,120) | rows (input) | 4,096 rows | partial [R, 5,120] fp32: `fast_gather` + rank-order sum |
| routed experts `w1` / `w3` (5,120 -> 2,304) | columns | 1,152 (9 Hadamard blocks) | none |
| routed experts `w2` (2,304 -> 5,120) | rows | 1,152 | partial [R, 5,120] fp32, summed with the shared expert's in the same exchange |
| shared expert | as routed | 1,152 | (same exchange) |
| `ffn.gate` + bias (router) | replicated | all | none (both ranks route identically) |
| Engram tables (24 hash heads) | heads | 12 heads = a contiguous row range of each table | the 12 rows' 256-dim values a token (all-gather, 6 KB a token a module), or row-parallel `wkv` + a partial-sum exchange (ARCH-LEVERAGE 3.3) |
| `engram.wkv` (6,144 -> 25,600) | replicated (or rows, with the option above) | | |
| `head` (5,120 -> 129,280) | vocabulary (columns) | 64,640 | top-k candidates a row (sampling) |
| `embed` | vocabulary rows | half | one gather of the prompt's rows |
| DSpark blocks | as the main layers (heads, groups, expert halves) | | 2 a block |
| vision tower | rank 0 only | 0.9 GiB | rows shared with rank 1 by `vision.exchange` |

- Every split boundary is a multiple of 128 (EXL3's Hadamard block): a rank's matrix is exactly the slice of the
  full one, proven on synthetic mul1 matrices with upstream's numpy decoder (`tests/kernels/test_exl3_experts.py`).
- The two partials a layer are added in rank order (rank 0 + rank 1), so both ranks hold identical residuals.

## 5. Memory per node: 2.9 bpw, 4 x 300K active, FP8 pool, a 4-6 GiB floor

### 5.0 Measured budget (2026-10-02, after G2 / G3; supersedes the plan table below)

`python -m tensorfold.families.deepseek_v41.cuda.memory <model_dir>` (copy: `engine/serving/memory.py`) prints the
plan table and this projection, `memory.measured` / `fit_context` in code. GiB a rank, the worst phase (a 300K prefill
while 3 streams decode), drafter on, graphs as built in e2a2333 (one shared pool):

| rank | active | index keys | start | weights | drafter | runtime | experts | serving | pool + rings | graphs | 2K chunk | select | sessions + Engram cache | **floor** |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 4 x 300K | bf16 | 115.1 | 95.40 | 3.60 | 5.40 | 0.56 | 1.80 | 2.39 | 0.40 | 0.30 | 0.25 | 0.35 | **4.65** |
| r1 | 4 x 300K | fp8 | 115.1 | 95.40 | 3.60 | 5.40 | 0.56 | 1.80 | 2.04 | 0.40 | 0.30 | 0.25 | 0.35 | **5.00** |
| r1 | 4 x 256K | fp8 | 115.1 | 95.40 | 3.60 | 5.40 | 0.56 | 1.80 | 1.75 | 0.40 | 0.30 | 0.21 | 0.35 | **5.33** |
| r0 | 4 x 300K | fp8 | 117.0 | 95.40 | 3.60 | 5.70 | 0.56 | 1.80 | 2.04 | 0.40 | 0.30 | 0.25 | 0.35 | **6.60** |

Where the terms come from:

- **start**: MemAvailable at the process start after a cache drop (G3 boot lines: head 116.7-117.2, worker
  115.1-115.3). Not the plan's 112: the worker starts ~2 GiB under the head and is the tighter rank.
- **weights** 95.4 allocated on both ranks (text model; no vision tower, no Engram table is loaded: nothing unused
  left to drop). **drafter** 3.6 (99.0 - 95.4: DSpark's 3 x 128 experts at 4 bits are 3.2 of it).
- **runtime** = start - MemAvailable(forward built) - allocated: head 5.7, worker 5.4 (G2 without the drafter: 4.7 /
  4.9). The plan had 1.0. CUDA context + NCCL / RoCE (~2.4 at the NCCL step), kernel modules, the process. The boot
  line now prints `memory.snapshot()` and `memory.host_trim()` (gc, `malloc_trim`, torch's pinned cache) runs after
  the boot (G2's compiling warm-up took 6.4 GiB on the worker outside the allocator). G4 attributes this term.
- **experts** 0.56: one expert scratch per geometry shared by all layers (1,024-row blocks; DSpark 64). **Before,
  every layer grew its own: 40 x 68.7 MiB = 2.68 GiB at 128-row windows** (G2's ~3.0 GiB "workspace"), 10.7 GiB at
  512-row windows, 43 GiB at prod.env's 2,048 prefill rows; grown after a graph capture it also freed memory the
  graphs still wrote to (G3's crashes; e2a2333 now also fingerprints the buffers).
- **serving** 1.8: the rest of G3's eager 4 x 16K bench (built -> minimum 4.7 GiB on both ranks, less the expert
  scratches, pool and rings): activations, the CSA2 attention scratch, the session tier, host growth.
- **pool + rings**: `pool.pool_bytes` + 4 SWA ring sets; **FP8 index keys** (`TF_DSV41_INDEX_KV=fp8`, the kit's
  per-token key, 132 B a row; 8f48303) save 0.35 at 4 x 300K with no top-1 loss in G2 (99.646 vs 99.634%).
- **graphs** 0.4 (estimate): one shared pool, boot captures one slot at widths 1-6, and no capture starts under 5 GiB
  MemAvailable (`TF_DSV41_GRAPH_FLOOR_GIB`), so captures cannot push a node under the floor; 1.2 is the pessimistic
  column. **chunk** 0.3 and **select** 0.25 are estimates. **sessions + Engram cache**: the RAM tier's 256 MiB + the
  row prefetch's LRU (2 x 65,536 rows, ~600 B each), 16 reader threads, RoCE shards (~0.1).

**Result.** 4 x 300K active with FP8 index keys: worker **5.00 GiB** (at the 5 GiB target, no margin), head 6.60;
with bf16 keys 4.65 / 6.25 (the 4 GiB stop holds). If graphs cost the old 1.2 GiB budget, the worker is 4.29 (FP8).
The 5 GiB target on both ranks (`fit_context`):

| index keys | graphs 0.4 GiB | graphs 1.2 GiB |
| --- | ---: | ---: |
| bf16 | 4 x 258K | 4 x 163K |
| fp8 | **4 x 299K** | 4 x 192K |

So prod runs **FP8 index keys and CONTEXT 300000** (the hard stop has ~1 GiB to spare; admission holds new work under
4 GiB). If G4's stress shows the worker under 5 GiB, the fallback is CONTEXT 196608 (4 x 192K meets 5 GiB even with
1.2 GiB of graphs); smaller levers: `TF_DSV41_EXPERT_BLOCK=512` (+0.27; prefill pays 2x the expert weight passes),
`TF_DSV41_SESSION_RAM_MIB=64` (+0.19), `TF_DSV41_ENGRAM_CACHE` (rows a layer). Idle sessions park on NVMe and hold no
pool pages; they do not change the active-stream budget.

### 5.1 The plan (2026-10-01)

From `engine/serving/memory.py` (`python -m engine.serving.memory <model_dir>`), GiB per rank; `available` = 112 GiB,
the kit's measured preflight point after a cache drop with the OS, docker and desktop up:

| pack | rank | active streams | index keys | weights | pool | rings | workspace | runtime | used | **floor** |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mia29 | r0 | 4 x 300K | bf16 | 99.48 | 2.35 | 0.04 | 4.00 | 1.00 | 106.87 | **5.13** |
| mia29 | r0 | 4 x 300K | fp8 | 99.48 | 2.00 | 0.04 | 4.00 | 1.00 | 106.52 | **5.48** |
| mia29 | r1 | 4 x 300K | bf16 | 98.58 | 2.35 | 0.04 | 4.00 | 1.00 | 105.97 | **6.03** |
| mia29 | r0 | 2 x 300K | bf16 | 99.48 | 1.17 | 0.02 | 4.00 | 1.00 | 105.67 | 6.33 |
| exp30 (routed experts 3.0 everywhere, EXL3 non-experts) | r0 | 4 x 300K | bf16 | 103.11 | 2.35 | 0.04 | 4.00 | 1.00 | 110.50 | **1.50** |
| exp30 | r0 | 1 x 300K | bf16 | 103.11 | 0.59 | 0.01 | 4.00 | 1.00 | 108.71 | 3.29 |
| cool30 (coolbho3k-style: 3.0 experts only, the rest source FP8 / BF16; +-1 GiB) | r0 | 4 x 300K | bf16 | 104.60 | 2.35 | 0.04 | 4.00 | 1.00 | 111.99 | **0.01** |
| cool30 | r0 | 1 x 300K | bf16 | 104.60 | 0.59 | 0.01 | 4.00 | 1.00 | 110.20 | 1.80 |

The terms:

- **Pool**: the four kv sources' FP8 rows (584 B: 1/2 a token on layers 2, 8, 14, 1 on layer 20) and their index
  keys (bf16, 256 B) = **2,100 B a token**, whole rows on each rank (one KV head). 4 x 300K (+64 slack, whole
  256-token pages) = 2.35 GiB; 2.00 with FP8 index keys (a later knob: the keys decide the selection, so bf16 first,
  as on GLM). FP4 main KV (the model's QAT format, 288 B) would bring the pool to ~1.2 GiB later.
- **Rings**: 43 SWA rings of 256 rows a slot (6.4 MB) + DSpark taps (3.9 MB) + carries.
- **Workspace** 4.0 = graphs 1.2 + a 2,048-row prefill chunk 1.3 (experts and attention in 512-row blocks: rows
  are independent, so the bits do not change; a whole-chunk expert scratch would be 1.15 GB alone) + the indexer's
  selection scratch 0.25 (rows in blocks so scores + keys of a block fit: ~75 rows at 300K) + drafting 0.35 +
  allocator slack 0.9. Sized, never grown (0550's rule); the first GPU window measures it.
- **Runtime** 1.0: CUDA context, NCCL + RoCE buffers, the process.

**The floor we target: 5 GiB worst node, worst phase** (a 300K prefill while 3 streams decode at 300K and a fourth
session parks or resumes), with 4 GiB as the hard stop. Why not GLM's 8 GiB, and why it is safe:

- At an 8 GiB floor the 2.9 bpw pack holds **no** 300K stream next to a 4 GiB workspace (99.5 + 4 + 1 + 8 = 112.5).
  The lower floor is what makes 4 x 300K possible at all; other V4.1 setups run at 0.9-4 GiB.
- The floor is held by **accounting instead of margin**: every buffer is sized at load (no growth during prefill:
  the kit's 2.1 GiB floor came from scratch growing in a 601K prefill), admission charges a request its pages
  before it starts (`kvpool.need_tokens`), and when the pool or the floor would be breached the batcher parks the
  coldest idle session to NVMe (0290's `pool_spills` + the 0250 tier) instead of dipping. Plus: the page-cache
  drop at start, the Engram reads O_DIRECT (no page cache growth, ever), the watchdog / OOM handling and 0550's
  allocator trims before each prefill piece.
- **What it buys**: 4 active streams at 300K (the kit: one 785K pool shared by 4 sequences), a 2,048-row prefill
  chunk, and the 2.9 bpw pack. With 1 GiB more (the 5.48 row) the index keys go FP8 or the Engram row cache gets
  ~0.5 GiB.

**Which checkpoint fits**: the 2.9 bpw pack (Mia / dealignai) at 4 x 300K with 5.1 GiB to spare. A 3.0 bpw-experts
pack fits only 1 stream (exp30: 3.3 GiB at 1 x 300K, under the floor) and coolbho3k's layout not even that; 3.0 is
a quality-for-concurrency trade the user would make knowingly (coolbho3k KLD 0.041 at 3.0; no KLD published for the
2.9 pack), not this build's default.

## 6. Engram rows from NVMe

The tables (layers 1 and 14; 384M rows of 264 B a layer) never enter RAM. `engine/serving/engram.py` is the CPU
reference implementation; the production reader is its native twin.

- **Files**: the kit's packed per-rank shards, already on both nodes' NVMe (`~/dsv41-engram/engram-l{1,14}-r{0,1}of2.bin`,
  ~47 GiB a layer a rank). Format facts (read from the files, no code taken): a 4,096-byte header of 6 little-endian
  u64 (magic `DSV41EN1`, layer, lo, hi, the layer's total rows, row bytes 264), then rows lo .. hi - 1. A rank owns
  12 of 24 hash heads = one contiguous row range.
- **Reads**: O_DIRECT, sector-aligned (512-B logical blocks on both Sparks' Samsung NVMe, checked): a row costs one
  or two sectors (512-1,024 B), not a 4 KiB page; rows of a request are deduplicated, sorted and coalesced into one
  read when the hole between them is <= 4 KiB. A thread pool (32 workers) issues them; results land in pinned staging
  and go to the GPU in one copy a layer.
- **When** (the prefetch): addresses are pure functions of token ids (the last 4, after tokenizer compression), so:
  - the pending token's rows are issued at the end of the previous verify, under the DSpark pass (~5.6 ms);
  - the drafted tokens' rows when DSpark has proposed, under host bookkeeping, layer 0 and layer 1's attention
    preparation; layer 14's rows have half a forward of slack;
  - a prefill chunk's rows are issued when the chunk is cut, under the previous chunk's forward; tokens served from
    the prefix cache need no rows at all.
  This removes the 2.8 ms a step tonyd2wild measured for synchronous local reads (~5% of a round).
- **Volume**: decode ~150-300 reads a round a rank (R x 12 x 2); prefill ~49K reads a 2,048-token chunk before
  deduplication. The Python reader is enough for decode; **prefill needs the native reader** (C++ thread pool or
  io_uring, ~32 queue depth, into pinned memory; ~0.1-0.2 s a chunk at ~500K IOPS, hidden under a ~1 s chunk).
- **Row cache** (optional): an LRU of hot rows (Zipfian: the top 100M rows cover 92.7%, bot-lab-21) sized only from
  memory above the 5 GiB floor; off by default. Exact by construction (the same bytes).
- **Hashing** runs on the host, outside the CUDA graphs (the graphs read a staging buffer), from the reference's
  tokenizer map, primes and multipliers precomputed into the prepared folder. Image tokens break n-grams (pad) and
  mask the gate.
- **Exactness**: a row's bytes are the file's whatever the path (direct, buffered, cached, prefetched or not):
  replies never depend on I/O timing. The tests compare every path against the file.
- **Implemented** (`engine/kernels/engram/hash.py`): `Tables` (the prepared folder's token map, multipliers,
  primes, offsets) hashes on the host == the reference `NgramHasher` bit for bit (chunks with the carried 3-id
  lookback, image tokens kept as DEAD, == the whole sequence); `Prefetch.issue(lookback, [pending, d1, ..])` at
  drafter end submits the rank's 12 rows a token a layer to the reader, `land` / `upload` move a layer's rows into
  the fixed device buffer before layer 1 (layer 14's after layer 1 is queued).

## 7. DSpark with exact keyed verification

- **The drafter**: `mtp.0-2` of the checkpoint (3 blocks, SWA 128, 128 experts top-3 + a shared expert, 4 bits),
  taps = the mHC outputs of layers 37-39 (mean over the 4 streams), one pass drafts a block of 5 positions (noise
  token 128,799), the rank-256 Markov head chains them, the confidence head estimates per-position acceptance.
  Its experts run through the same grouped kernel (E = 128, slots 3 + 1 shared, `x3ld` instance K2 8..8); its
  attention through `csa2.attn` with no compressed rows and the per-row window end `hi` (the block's rows see each
  other: DSpark's non-causal block).
- **Verification** (drafted == serial):
  - each verify row is the serial step at its position: every kernel is row-invariant (section 10), so a row's
    logits do not depend on the window's other rows;
  - its token is `exact_sampling.choose_rows` with the request's seed at that absolute position (each rank's half
    of the vocabulary, one candidate exchange);
  - a draft is kept up to the first position where it differs from that keyed choice
    (`engine/serving/drafting.py: accepted`); the round emits accepted + 1 tokens.
  So any drafter, depth or acceptance gives the serial reply. vLLM / the kit use rejection sampling: their T > 0
  replies depend on the drafts and the batch. **Ours is the first exact DSpark serving.**
- **Draft noise**: DSpark samples its T > 0 draft tokens with the target's keyed noise at the same positions (GLM
  DFlash2's rule): when draft and target distributions agree, so do their samples; acceptance at temperature stays
  near greedy's. Drafts never change a reply, so this is pure speed.
- **Depth**: the confidence head's q_i and the calibrated verify costs C(R) feed our cost-derived depth
  (`depth.best_k`); suffix lookup (0020) extends copy-heavy rounds past 5 positions up to 16 rows (0380). The
  confidence head is DeepSeek's own scheduler input, so this is DSpark as designed, with measured costs.
- **Implemented** (`engine/kernels/dspark/`): `chain` (one program a slot inside the drafting graph: the Markov
  bias on the gathered candidates on tensor cores, the (value desc, id asc) order, `exact_sampling.choose`'s keyed
  rule at the drafted positions, the confidence head), `candidates`, `attention_meta`, `verify.accept` and the
  argument in `verify.py`. The non-causal window is vLLM's: 128 context rows **plus** the N block rows (133 keys);
  `csa2.attn`'s plain `hi` window (128 rows ending at the block's end) would drop N context rows, so the draft rows
  pass the context as the compressed list and the block as the SWA chunk (`lo = P`, `hi = P + N - 1`).

## 8. CED: decoder bounded replay as a knob (`TF_DSV41_PREFILL=full|replay`)

- **full** (round 1 default; the M1 oracle mode): all 40 layers over every prompt token, numerically comparable with
  the kit's vLLM (which never uses CED).
- **replay** (the design's prefill, ~2x prefill: TARGETS 2,200 vs 1,500 tok/s): the encoder (layers 0-19, plus layer
  20's compressor projection of H_19 = the decoder's whole global KV and index keys) over every prompt token; the
  decoder (20-39) only over the last min(n, 128) prompt tokens, its SWA windows truncated to that segment
  (`csa2.attn`'s per-row window start `lo`); reply tokens always run all 40 layers.
- **Exactness rules** (ARCH-LEVERAGE section 2 states the same result):
  1. it is approximate *by design* against `full` (the model was post-trained for it); each mode has its own tag
     (`engine/serving/sessions.py: Tag`), and snapshots of one mode never resume the other;
  2. **always replay**: every prompt (first turn, follow-up, resumed or fresh) rebuilds the decoder state from its own
     last 128 tokens; a previous turn's decoder rings are never carried into a prompt. Then the decode start state is
     a function of the token sequence alone: **resumed == fresh**, **batched == alone** (the replay rows use the same
     row-invariant kernels), and drafted == serial (decode rows are full 40-layer rows);
  3. encoder state is exact: snapshots keep the 20 encoder rings; DeepSeek's *encoder* bounded replay (approximate:
     depends on the hit position) is not used, except as a separate opt-in for the case where only global KV
     survived;
  4. every prefill replays, whatever its size (vLLM replays only in eager steps over 1,024 tokens, so its output
     depends on the step size).
- Quality gate before `replay` becomes the default (round 2): MMLU-200 and the tool-call gate within 1 point of
  `full`, needles at 64K / 128K / 300K.

## 9. Vision: our 0500 with a DeepSeek-ViT tower

- Reused: the host side (fetch with our GIF / NAT64 additions, canvas digest, virtual token ids = content hashes so
  sessions and prefix shares key images for free, `expand`), the rank-0 tower + `vision.exchange` of rows to rank 1,
  the digest-keyed row cache, `vision.active` around the embed.
- New: the tower (32 layers x 1,024, 16 heads, MLP 2,816, patch 14, 2D RoPE, 3 x 3 pixel-unshuffle, <= 1,024 tokens
  an image, `min_pixels` 295,936), the MLP aligner, `image_start` / `image_newline` / `image_end` rows, V4.1's processor
  constants, Engram masking for image tokens, and `gate.bias_vl` loaded from the source shards (the EXL3 packs dropped
  it: ARCH-LEVERAGE section 0 item 9).
- In-image bidirectional attention (the kit clamps it off on SM12x): `csa2.attn`'s per-row window end makes an
  in-image row see its image's later rows; the tower's own attention is a plain dense kernel. Round 2+, after a
  quality comparison.
- 0.9 GiB on rank 0 only (section 5's rank 1 has it free).

## 10. Kernels: what is implemented, and the exactness argument

All under `engine/kernels/`, tests under `tests/kernels/` (run instructions in `tests/kernels/conftest.py`):

| Kernel | What | Exactness argument | Offline evidence |
| --- | --- | --- | --- |
| `exl3/experts.py` | the V4.1 MoE block on upstream's EXL3 module: TP split, shared expert as entry 384, plans from header shapes (a 3-bit expert = 6,635,520 B a rank; 2.38 GiB a layer a rank), scratch sizing, prefill row blocks, upstream's `routed` sequence with the load-path hook | upstream's kernels: fixed K splits a shape, rows independent, no atomics (its recipe's row-invariance tests on MiMo mul1) | split == slice with upstream's numpy decoder (mul1, K2 4/6/8/10); two rank partials == the full expert (float64); our `routed` issues upstream's launches argument for argument (off) and swaps only the two grouped launches (on); scratch bytes == upstream's allocation |
| `exl3/x3ld.cu` | our 0580 load path over upstream's `grouped_kernel` for any width (2-16 half-bits), mul1, nt 8 / 4, pd 1 / 2, PDL, probe 3 | the grid, K ranges (upstream's 4 warps x K splits), the mma chain per accumulator on upstream's `decode_tile` / `load_pair` operands, the warp-order sum and the Z rows are upstream's (helpers included from its header, never copied); only when bytes arrive changes | lane-level emulator: every lane's trellis words and A fragments, every step, == upstream's for K2 2-16 x 3 settings x 5 step counts (98 tests), every v4 load 16-byte aligned and inside its step, 4 injected faults caught; nvcc 13.4 sm_121: 10 instances, 0 spills, 64-168 registers within 3 CTAs an SM, upstream's shared memory, PTX `ld.global.nc.L1::no_allocate.v4`, `mma.sync ... f16`, `griddepcontrol.wait`, no atomics |
| `csa2/rows.py` | the 584-B FP8 row (448 e4m3 + 7 UE8M0 scales of 64 dims, 64 RoPE dims bf16 = the kit's SM12x record), store / dequant / paging / RoPE helpers | a row quantized by itself, scale from the float's bits (exact `ceil(log2(amax / 448))`), values pre-rounded to the e4m3 grid (compiler-proof, GLM 0220's rule); dequantized rows are exact bf16 | kernel bytes == the torch reference on hard rows (zero tiles, the 448 boundary, tiny / huge, signed zeros, subnormal e4m3); dequant == reference bit for bit; == engine/reference's `fp8_ds_mla_roundtrip` bit for bit (`test_reference_agreement.py`) |
| `csa2/compress.py` | ratio-2 gated pooling / ratio 1 + RMSNorm, compressed / SWA row stores (RoPE at the group start / the token), index-key store | one program a row; the carry-row buffer layout makes a window's rows independent of the window | vs float64; a row alone == in its window; paged == contiguous with junk-filled unmapped pages |
| `csa2/index.py` | GLM's DSA score formula over compressed positions; dense or gathered (layer 20's candidates); unique int64 keys (score desc, lower position); top-512; candidate blocks (block max, newest pinned, 2,048 best); reindex | a row's scores are its own tile rows (fixed tile shape, fixed 8 warps: the head sum's tree is part of the arithmetic); the keys make the selection independent of the sort | vs float64; gathered == dense bit for bit at the same positions; selection == the stable sort with ties and signed zeros; candidates == the block rule; reindex == the masked dense selection |
| `csa2/attn.py` | sparse (<= 512 selected compressed rows) + SWA (128) attention, 32 local heads x 512, sinks, inverse RoPE, bf16 out; bounded replay (`lo`), DSpark's block (`hi`), paging | GLM's latent tile arithmetic (tensor-core bf16 dots, fp32 online softmax) over 5 chunks fixed by list index (4 x 128 compressed, then the window), merged in order, then the sink | vs float64 (with sinks, SWA-only, replay, DSpark); a row alone == in windows of other sizes / tile-mates; paged == contiguous; a larger prefill ring == the decode ring |
| `router.py` | logits (fp32 FMA chains), sqrt-softplus, top-6 by score + bias (ties to the lower id), renormalised x 1.5, the shared slot | ieee fp32 dot = one FMA chain an element (MLA-EXPAND's analysis); selection reads only the row | vs float64; ties; a row alone == in a window, bit for bit (with an exact FMA-chain model of the dot) |
| `csa2/stream_topk.py` | the indexer's score + top-k without materialised scores (300K prefill): programs (row, key split) score with `index.score_tile`, keep a bounded buffer (threshold + radix-select compaction), a merge over splits; Full, Reindex and candidate-block modes, paging | the tile is `index._scores`' own code (all 21 CSA2 PTX hashes unchanged by the factoring); keys are unique, so the set is the top K whatever the splits / order | == `index.select` / `reindex` / `candidate_blocks` position for position (ties, zero blocks straddling the cut, many compactions, real K); short rows keep exactly their visible keys; paged == contiguous; row alone == in a window; scratch <= 0.25 GiB at 300K for >= 1,024 rows a launch |
| `mhc/` | Single-Pass mHC: one pass a boundary (post with the gathered partials, collapse with the carried pre-mix, the next site's 24 mixes and square sums, DSpark taps) + a per-row finish (Sinkhorn, normed input); `site`, `boundary`, `post_only`, `final` | the mixes are FMA chains per (stream, 128-column block) in column order (ieee dot), summed in a fixed order; no reduction trees; `enable_fp_fusion=False` | streams, collapse, taps, partials, normed input == the torch emulation bit for bit (D 5,120, world 1 / 2); pre / post / comb within 1e-5 of engine/reference/hc.py and float64; row alone == in a window; in place == out of place |
| `engram/` | host hash + prefetch (`hash.py`), record dequant, gated fusion into the 4 streams | dequant: exact bit decode, one IEEE product, one bf16 rounding; fusion sums as FMA chains against ones | dequant == the reference bit for bit (every e4m3 byte, zero / subnormal / huge scales); update == bf16(x + gate v) bit for bit, gates within 1e-6; image tokens unchanged; row alone == in a window; the whole module == engine/reference/engram.py |
| `dspark/` | the chain (Markov bias + keyed choice + confidence), candidates, the block's attention metadata, `verify.accept` | drafts never enter a reply (`verify.py`); the chain is keyed like the target | tokens == `exact_sampling.choose` over the biased candidates (greedy, top_k / top_p / min_p); == the reference DSpark's drafts and confidence (tiny config); a draft from the target's distribution == the target's keyed choice; serial == speculative with any drafter, T = 0 / T > 0, past 2,048 tokens; the attention metadata == float64 over 128 context + the block |

- **Interpreter model** (`gpu_like` fixture): Triton 3.8's interpreter multiplies bf16 storage as integers and
  truncates fp32 -> bf16; the tests model the GPU (exact operand widening, exact fp32 FMA chains for ieee dots,
  nearest-even casts). The GPU bitwise runs (window G1) are still the proof for tensor-core dots and FMA contraction.
- **sm_121a compiles** (`tests/kernels/csa2_ptx.py`, the GLM `kvpool_ptx.py` pattern, Triton's bundled ptxas):
  21 kernels, 0 spills, no atomics, `mma.sync ... bf16` in attention and scores, the FMA path (no mma) in the router,
  paging compiled away when off. `attn_chunks` uses 245-254 registers (1 CTA of 8 warps an SM, like GLM's latent
  kernels); a KT = 16 tile is the occupancy knob if the bench asks for it (same bits). `tests/kernels/blockers_ptx.py`:
  18 more (mHC 8, Engram 2, DSpark 2, streaming top-k 6), 0 spills, no atomics, fp32 chains as `fma.rn.f32` (no
  mma), no `.ftz` in the exact paths, mma in the streaming scores and the Markov bias; mHC boundary 152 registers
  (8 warps), stream blocks mode 255 (8 warps, 0 spills).
- **Window G1 checks for these** (no GPU was used): (1) mHC / Engram / stream bits on the GPU == the interpreter's
  (the FMA-chain claim of Triton's ieee dot, `enable_fp_fusion=False` honoured); (2) `stream_topk.select` ==
  `index.select` on GPU scores at 4K-300K keys (the shared tile gives the same bits in both kernels) and its time
  against the materialised path; (3) the chain's drafts against the host `exact_sampling` over many seeds (libdevice
  log / exp vs glibc: only near-ties may differ); (4) Engram dequant with fp32 subnormal products (no flush to zero).
- **Not yet**: the native Engram reader, the attention gather once per CSA2 group (ARCH-LEVERAGE 4); `x3pf` is the
  first prefill expert step (decode once per 64 members), a fat2-class pipelined GEMM the next if G1 asks for it.

**Status 2026-10-02 (M1 core on the branch `dsv41-060`, offline):** the kernels above are ported into
`families/deepseek_v41/cuda/` (with row strides for the pool's 584-byte rows), and the family has its loader,
prepared folders, the multi-slot TP=2 forward behind the serving layer's protocol, the engine and the M1 gate harness;
the forward equals `engine/reference` to 1e-12 in exact numerics over every layer kind. Details, test counts and the
G1 / G2 commands: [M1-STATUS.md](M1-STATUS.md).

## 11. Milestones

Each milestone ends with gates, never with tok/s alone. Round 1 (section 1) is M1-M3 with the fast boot and the
blanket wins built in from their first window; round 2 is M4 + the efficiency work + `replay` as default.

### M1: the checkpoint runs exact on our engine (TP=2)

- Offline first: the reference (engine/reference) per layer kind (SWA-only, Full r2, Reuse r2, Full r1 +
  candidates, Reindex, Reuse r1) + Engram + mHC + MoE, checked against vLLM's math on synthetic weights; the loader
  (per-tensor K from trellis shapes, TP splits, the prepared-folder writer); the forward on synthetic checkpoints
  (upstream `tests/dsv4_fakes.py` style) == the reference.
- On the GPU: `full` prefill + serial decode, FP8 KV, Engram reads synchronous, `x3ld` on.
- **Gates**:
  1. teacher-forced top-1 of our logits == the kit's `prompt_logprobs` argmax at **>= 99% of positions**, 8 prompts x
     2K tokens (the kit's logprobs collected once in the baseline window and stored, so M1 needs no kit window of its
     own);
  2. 256-token greedy replies match the kit's to the first divergence on >= 6 of 8 prompts;
  3. row invariance at the forward level: a row alone == the same row in an 8-row window; prefill rows == decode
     rows (exact prefill tag);
  4. serial decode >= the kit's `SPEC_METHOD=none` 23 tok/s (the floor says ~30 at 65%);
  5. boot from the prepared folder <= 90 s, then <= 60 s.

### M2: decode with DSpark

- DSpark blocks, Markov + confidence heads, keyed draft noise; cost-derived depth with the confidence head; suffix
  lookup; graphs per verify width.
- **Gates**: drafted == serial (exact suite, past 2,048 tokens of context, T = 0 and T > 0); DSpark tokens a round
  >= 3.1 at k=3 on the user's session mix; round 1 speeds of section 1.3 (code >= 52, prose >= 38) on RigMark x3.

### M3: serving

- The Batcher (V4.1 `compute_multi`, multi-slot prefill), the pool families, the session store + NVMe tier with V4.1
  snapshots, park / resume, the idle doorbell, the server (template, DSML tools, effort, structured output,
  `/tokenize`, logprobs later), Engram prefetch, memory accounting.
- **Gates**: batched == alone at 4 slots; resumed == fresh (RAM and NVMe, after a restart); 4 x 300K stress with
  0.5 s memory samplers on both nodes: MemAvailable >= 4 GiB in every phase (target 5); park / resume of a 300K
  session <= 1 s; replay TTFT 64K <= 0.25 s; MMLU-200 and the tool-call gate within 1 point of the kit.

### M4: RigMark parity and beyond

- RigMark x3 on our build against the baseline receipt; the efficiency work toward 75% of the floor; `replay` as the
  default after its quality gate; vision; FP8 / FP4 index keys or main KV if the floor wants room.
- **Gates**: TARGETS.md section 2 (code 72 / prose 45 / structured 90 / C4 120 / prefill 2,200 replay); exact suite
  10/10 in both CED modes; the noise rule (>= 6 runs for claims under 10%).

## 12. Effort and GPU windows

| Phase | Work | Agent days | GPU windows |
| --- | --- | ---: | ---: |
| 0 (now) | kernels here, the reference (parallel track), the loader + prepared folders, the baseline window's script with the logprobs capture | 6-8 | **W0**: the kit baseline (DSV41-BASELINE section 3, + `prompt_logprobs` capture), ~3 h |
| M1 | forward (CSA2 by role, mHC, MoE, Engram sync, head), TP plumbing, fast boot | 12-16 | **G1** kernels: GPU bitwise of x3ld / CSA2 / router + `kbench` of x3ld vs upstream on real mul1 experts at R = 1-16 (probe-3 gate >= 220 GB/s), ~2 h, one node; **G2** M1 bring-up + gates, ~3-4 h |
| M2 | DSpark, depth, lookup, graphs | 5-7 | **G3** ~3 h |
| M3 | batcher, pool, sessions / NVMe, Engram prefetch + native reader, server, memory accounting | 10-14 | **G4** ~4 h (4 x 300K stress, park / resume, restarts) |
| M4 | RigMark, efficiency, replay default, vision | 5-7 | **G5** RigMark x3 ~3 h; **G6** efficiency A/B ~3 h |
| | spare / localisation | | 1-2 |
| **Total** | | **~38-52 days** | **7-9 windows** (~20-26 h of GLM production down) |

- One agent owns the hardware per window; everything under `timeout`; lease + watchdog + deadman as the GLM windows
  (RIGMARK.md / W18 harness); GLM production restored at the end of every window (`restore.sh`).
- The fast boot pays for itself from G2 on: ~5 min saved per restart, 4-8 restarts a window.

## 13. Blockers and risks

1. **The reference** (engine/reference, parallel track) is M1's offline oracle; the forward cannot be checked on
   synthetic weights without it. Open details it settles: the per-head query normalisation before RoPE (vLLM's fused
   kernel takes it with `apply_q_norm` off), the sink formula (FlashMLA's convention assumed: a zero-value key with
   logit `sink_h`), the RoPE tables (YaRN x16 on compressed layers, `compress_rope_theta` 160,000), the Engram hash
   and fusion, Single-Pass mHC's coefficient timing, the routing weight's place (before `w2` in DeepSeek's code;
   after the down projection here: equal in exact arithmetic, a rounding difference only).
2. **GPU bitwise proof** of the new kernels is pending (G1): the interpreter models tensor-core dots and casts, not
   ptxas (FMA contraction in the RoPE pairs, mma accumulation order); row invariance holds structurally, the bits
   against the CPU references do not have to.
3. **Prefill indexer at 300K**: ~~`csa2.index.select` materialises a row block's scores~~ **written**:
   `csa2.stream_topk` (bounded scratch, >= 1,024 rows a launch within 0.25 GiB at 300K, the same selection as
   `index.select`); G1 measures it on the GPU. Contract note for the forward: `index.select` lists invisible
   positions (score -inf) after the visible ones in rows with fewer than 512 visible keys, so the attention's
   `counts` must be min(visible, 512) there; `stream_topk` pads such rows with -1 instead. Indexer FLOPs at 300K (bf16 tensor cores at ~80
   TFLOP/s): ~12 s of a ~200 s `full` prefill at 1,500 tok/s (layers 2 / 8 / 14 + 20), ~7 s of a ~136 s `replay`
   prefill at 2,200 (layer 20 scans only for the 128 replay rows).
4. **Engram reader throughput** for prefill: the Python reader is fine for decode; prefill needs the native reader
   (M3).
5. **Kernels**: Single-Pass mHC, Engram dequant / gate / fusion (+ host hash and prefetch), DSpark heads **written**
   (section 10; GPU bits in G1). mul1 prefill expert GEMMs: `exl3/x3pf.cu` (`TF_DSV41_EXPERT_PREFILL=1`) applies
   each decoded trellis tile to 4 member tiles (64 members) instead of 1, upstream's Z bit for bit (schedule
   emulator vs grouped_kernel; sm_121: 4 instances, 0 spills, 198-240 registers); G1 measures it and checks the bits.
6. **Workspace** 4.0 GiB is an estimate; G2 measures it (the floor moves 1:1 with it). Measured: section 5.0 (the
   per-layer expert scratches were 2.68 of G2's ~3.0; shared now: 0.56).
7. **Acceptance** is the largest speed uncertainty (TARGETS 3.1); W0 measures it on the kit with our pack.
8. **Downtime**: every window takes GLM production down; the two models cannot be resident together.
9. **`gate.bias_vl`** is missing from both EXL3 packs (image routing): load it from the source shards (66 KB).

**Update 2026-10-02:** item 1 is done (the forward == the reference); item 5's mHC, Engram and DSpark kernels are
landed by the kernel track (mHC / Engram wired through `seams`, DSpark for M2). Open for M1: the GPU windows G1 / G2,
a kit capture of greedy replies for gate 2, and the bf16-vs-FP8 index keys' top-1 cost (M1-STATUS.md).

## 14. Files

| Path | What |
| --- | --- |
| `engine/kernels/tf.py` | where TensorFold 0.6.0 comes from (`TF_SRC`, the work trees) |
| `engine/kernels/exl3/experts.py`, `loads.py`, `x3ld.cu`, `x3ld.cpp` | the MoE block on upstream's EXL3 module; the 0580 load path |
| `engine/kernels/csa2/rows.py`, `compress.py`, `index.py`, `attn.py`, `ref.py` | CSA2 kernels and their torch reference |
| `engine/kernels/router.py` | the router |
| `engine/kernels/csa2/stream_topk.py` | the streaming score + top-k (300K prefill indexer) |
| `engine/kernels/mhc/`, `engram/`, `dspark/` | Single-Pass mHC; Engram hash / prefetch / dequant / fusion; DSpark's chain, glue and `verify.py` (interfaces in each `__init__` docstring) |
| `engine/serving/topology.py`, `pool.py`, `state.py`, `sessions.py`, `memory.py`, `engram.py` | the family's seams that are fully determined (tested) |
| `engine/serving/family.py`, `engine.py`, `batcher.py`, `drafting.py`, `app.py` | the contracts of `families/deepseek_v41` (stubs) |
| `tests/kernels/` | 187 CPU tests: `test_exl3_*` (emulator, split, sequence, compile), `test_csa2_*` (interpreter, compile), `test_router_interpreter.py`, `test_serving.py`, `test_{mhc,engram,dspark,stream_topk}_interpreter.py`, `test_engram_hash.py`, `test_dspark_exact.py`, `test_blockers_compile.py`; `csa2_ptx.py` / `blockers_ptx.py` (PTX hashes before / after a change); `stubs/` (CUDA library headers for the nvcc compile test) |
