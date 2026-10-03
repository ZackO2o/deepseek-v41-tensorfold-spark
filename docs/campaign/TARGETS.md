# Targets for DeepSeek-V4.1-Flash on TensorFold, 2x DGX Spark (2026-10-01)

What our build must reach on the same two Sparks and the same checkpoint the user runs today
(`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, EXL3 mul1, Engram rows on local NVMe). The evidence is in
[`research/LANDSCAPE.md`](LANDSCAPE.md). The byte model is DSV41-BASELINE.md section 4.2 (glm53 repo),
with one input updated from measured data (below).

**The goal, stated by the user: more than 2x the Mia kit on 2 Sparks**, with TensorFold 0.6.0 and our stack.

## 1. What "the Mia kit" is, in numbers

The kit has three sets of numbers, and they differ by up to 2x. A "2x" claim has to name which one it beats.

| Source | Code c1 | Prose c1 | Structured c1 | C2 / C4 aggregate | Cold prefill 8-64K | Notes |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| (a) **Independent measurement**: helge, forum 383242, Mia 2.9 | **36.4** | **25.1** | | | | dealignai card ~27; say3 26; sfxnz 16.4 (stock k=3) |
| (b) **Our own run** (uncensored pack, moe_x, 3 h session) | ~44-48 single stream (mixed content, thinking on) | | | x2 ~78; **x4 53-63** | **1,064 @39K** | DSpark 3.11 tokens a round at k=3 (70.2% a draft) |
| (c) Kit README, moe_x, author's harness | 56-57 | easy 63-66 / hard 32 | count-to-200 40 (pre-moe_x) | x2 **97** | 970-1,055 | 4-token verify 60 ms |

**Anchor: (a) for single-stream cells and (b) for concurrency and prefill**, until our baseline window
(DSV41-BASELINE.md section 3) measures the kit with RigMark x3 on our pair. Then every target below is re-based on that
receipt, which will be the first V4.1 RigMark receipt anywhere. (c) is the kit's best case on its own prompts. Where
2x of (c) is beyond the hardware, this file says so (section 3).

## 2. Targets

Same checkpoint, TP=2, 4 request slots, FP8 KV, RigMark protocol (`reasoning_effort` low) unless a cell says
otherwise. "Exact" keeps the glm53 meaning: drafted == serial, batched == alone, resumed == fresh, byte for byte. CED
replay is a separate mode with its own exactness contract (DSV41-BASELINE section 2.3).

| Metric | Kit anchor | Best known on 2 Sparks (V4.1) | **Target** | Stretch | vs anchor |
| --- | ---: | --- | ---: | ---: | --- |
| Decode, code c1 (RigMark code) | 36.4 (a); 56-57 (c) | 37.4 coolbho3k; 47.6 with prompt lookup on edits | **72** | 80 | 2.0x (a), 1.3x (c) |
| Decode, prose c1 (RigMark prose) | 25.1 (a); 32 hard / 63-66 easy (c) | 42.6 real-use, 50.4 `ignore_eos` (sfxnz, 2.0 bpw) | **45** | 50 | 1.8x (a), 2.0x at stretch |
| Decode, structured c1 (RigMark structured) | 40 count-to-200 (c, pre-moe_x) | 84.2 (sfxnz, 2.0 bpw) | **90** | 100 | 2.25x (c) |
| C4 aggregate (RigMark short code, 256-token cap) | 53-63 (b) | C6 62.1 (coolbho3k); C2 156 structured (sfxnz) | **120** | 140 | 2.0x (b) |
| Cold prefill 32K / 64K, CED replay mode | ~1,050 (b, c) | ~1.0-1.2k (Mia, coolbho3k) | **2,200** | 2,600 | 2.1x |
| Cold prefill 32K / 64K, full-decoder mode (vLLM-equivalent) | ~1,050 | same | **1,500** | 1,800 | 1.4x |
| 300K-token cold prompt, end to end (CED) | 810 tok/s at 601K (742 s TTFT) | 676 tok/s at 1M (25.6 min, coolbho3k) | **≤ 2.5 min** (≥ 2.0k avg) | ≤ 2 min | ~2.3x the kit (~850 tok/s at that length) |
| Replay TTFT, 64K resident (RigMark immediate replay) | 0.46 s identical resend (b) | APC in RAM only | **≤ 0.25 s** | 0.18 s | ≥ 1.8x |
| Resume a 64K session after eviction or restart (NVMe tier) | none (the kit's offload tier failed at init) | none on 2 Sparks (4x: 92K in 0.86 s, ZackO2o) | **≤ 1.0 s** | 0.5 s | new capability |
| Context: active streams / parked sessions | 600K a request, 785,676-token pool, 4 seqs; no session tier | 1,048,576 a request / 3.3M pool (coolbho3k, NVFP4 KV in display memory) | **300K a stream on 4 active streams (1.2M FP8 pool); idle sessions park on the NVMe tier and resume in ≤ 1 s** | parked sessions limited only by NVMe (a 300K session ~0.5 GB) | 4 x 300K resident vs the kit's 785K shared |
| Decode at depth | 19-24 at 100-601K (c) | 32.5 at 1M (coolbho3k) | **≥ 85% of the short-context rate at 300K** | | |
| Memory floor, MemAvailable, worst node, worst phase (300K prefill + 3 streams at 300K decoding + a session park / resume) | 2.1 GiB at a 601K prefill | 0.89 GiB lowest (coolbho3k) | **4-6 GiB**, held by the memory-safety work (sized scratch, admission, trims) | 6 GiB | 2-3x the kit's floor |
| Boot to serving (prepared weights) | ~6 min | | **≤ 2 min (round 1 priority)** | 1 min | 3x |
| DSpark tokens a round, same content as (b) | 3.11 at k=3 | | **≥ 3.1 at k=3** (then adaptive depth) | | no loss |
| Exactness | none (rejection sampling; kernel path changes with row count) | none anywhere | **drafted == serial, batched == alone, resumed == fresh** in each CED mode | | new |
| Quality, full-decoder mode | | | **top-1 ≥ 99%** vs the kit's `prompt_logprobs` (M1); MMLU-200 within 1 point of the kit; tool-call gate ≥ the kit | TEB 2.6.1 hard ≥ 88 (helge: 89) | parity |
| Quality, CED replay mode | | | **MMLU-200 and tool-call gate within 1 point of full mode**; needles at 64K / 128K / 300K | | |

### 2.1 Round 1 priority (decision, 2026-10-01)

Iteration speed first:

1. **Boot ≤ 2 min** from prepared weight folders. Every GPU window and every A/B pays the boot, and the kit's is ~6 min.
2. **Blanket speed wins** that lift every workload at once, ahead of workload-specific tuning. These are:
   - round efficiency toward 75% of the floor (grouped mul1 experts, RoCE all-gather, L2 prefetch, fused SwiGLU clamp,
     Engram reads off the critical path);
   - CED replay prefill.

Per-workload depth tuning comes later: adaptive verify length, prose drafter work, and suffix lookup for structured
output.

## 3. Why these numbers

### 3.1 Decode: the round model

Bytes a verify forward, per rank, on this pack (DSV41-BASELINE section 4.2):

- non-expert 3.41 GB (attention at K5 is 2.11 of it);
- 6.38 MB an expert read;
- a DSpark pass ~0.9 GB (3 blocks of 128 experts at 4 bits, plus the head).

One input is updated with measured data. vcruz305 counted **16.1 unique experts for 24 routing slots at 4 rows, and 44
for 96 slots at 16 rows**, on real k=3 traffic. That is below the baseline's assumed U(4) = 18.5. The table uses
U(1 / 3 / 4 / 6 / 12) = 6 / 13 / 16 / 22 / 37.

| Verify rows R | Bytes / rank | Floor at 230 GB/s | Round at 75% + DSpark pass + 2 ms | Use |
| ---: | ---: | ---: | ---: | --- |
| 1 | 4.94 GB | 21.5 ms | ~31 ms (no draft) | serial: ~32 tok/s |
| 3 (k=2) | 6.73 GB | 29.3 ms | **~47 ms** | prose |
| 4 (k=3) | 7.52 GB | 32.7 ms | ~51 ms | mixed |
| 6 (k=5) | 9.02 GB | 39.2 ms | **~60 ms** | code, structured |
| 12 (4 streams x k=2) | 12.85 GB | 55.9 ms | **~85 ms** (one batched DSpark pass) | C4 |

- **75% of the floor** is above our GLM rounds (~70%) and the kit (~59%: 60 ms for R=4). It assumes upstream's
  grouped mul1 kernel at 205-225 GB/s and the round-overhead work from glm53 (E1 / E2, RoCE, L2 prefetch).
- The DSpark pass (~5.6 ms) is a fixed cost a round. A trimmed draft vocabulary or head would cut part of it.

Tokens a round come from section 4 of LANDSCAPE. They are the largest uncertainty, and the baseline window measures
them on our pack.

| Workload | Accepted drafts (others) | Tokens a round assumed | Round | Projection | Target |
| --- | --- | ---: | ---: | ---: | ---: |
| Code, k=5 | 3.2 (christopherowen, thinking off), 4.0-4.3 (tonyd2wild), 1.95 (thinking on) | 3.5-4.5 | 60 ms | 58-75 | **72** |
| Prose, k=2 | 1.0-1.7 at k=5; per-position 0.63 / 0.33 / 0.15 (yangqinhuan) | 2.0-2.3 | 47 ms | 43-49 | **45** |
| Structured, k=5 + suffix lookup | JSON 2.9-3.3, tables 4.0, count 4.8-5.0 | 5.0-5.5 | 60 ms | 83-92 | **90** |
| C4 short code, k=2 per stream | | 10-12 a batch | 85 ms | 118-141 | **120** |

**Prose is the hard one.** Acceptance on prose is ~1-2 drafts at any depth, so depth buys nothing. The only levers are
the round time (bytes and efficiency) and a better drafter:

- the source-precision DSpark test (sfxnz 1.33 vs 2.77; but the user's session already shows 3.11 tokens a round);
- a BF16 embedding for the drafter;
- an on-policy DSpark re-fit to the abliterated target (MIT, allowed).

2x of the independent prose number (50) needs 2.35 tokens a round at 47 ms. That is the stretch, not the target.

### 3.2 Where 2x is physically out of reach

Against the kit README's own best cells (anchor c), 2x is not available on 2 Sparks at 2.9 bpw. At 100% of the
bandwidth floor plus a 3.9 ms DSpark pass:

| Cell | (c) | 2x (c) | Ceiling at 100% of the floor |
| --- | ---: | ---: | ---: |
| Code, R=6, 4.5 tokens a round | 56-57 | ~113 | 4.5 / 43.1 ms = **~104** |
| Easy prose, R=3, 2.3 tokens a round | 63-66 | ~129 | 2.3 / 33.2 ms = **~69** |

Only fewer bytes per token beat that ceiling:

- a smaller pack (sfxnz's 2.0 bpw Viterbi pack has NLL 0.139 and 0.69x our expert bytes);
- a faster attention read (attention is 2.11 GB of K5 every round);
- attention on a discrete GPU (ds41rt: 2 Sparks + a 5090-class card, code 122.5).

Each one is a quality or hardware decision for the user, not part of this build. The README's "easy prose 63-66"
implies ~3.9 tokens a round at its 60 ms verify: its prompts draft like structured text. RigMark prose will not.

### 3.3 Prefill

The same efficiency as our GLM prefill (18.4 GFLOP a token a rank at 1,620 tok/s = ~30 TFLOP/s effective):

- **full decoder**, ~17 GFLOP a token a rank -> ~1,750 tok/s. The Engram reads, the four Full-mode indexer scans and
  the compressors are new costs, so the target is 1,500;
- **CED replay**, ~9 GFLOP a token a rank (encoder over all tokens, decoder over 128) -> ~3,300 at the same
  efficiency. Indexer and Engram costs do not halve, so the target is 2,200 and the stretch 2,600.

Others' evidence that CED is worth ~2x on GB10:

- YoungAi on one Spark: 1,055 with replay vs 539 with `--decoder-full`;
- SGLang: 1.37-1.56x on datacenter parts.

Cross-check against the fastest GB10 stacks:

- christopherowen gets 3.8k on 3 Sparks (no CED, native FP4 / FP8 tensor math), ~1.27k a node;
- knapcio gets 4.8k on 4 (with CED), ~1.2k a node.

Our arithmetic is bf16 x EXL3 at FP32 accumulate, which ran ~0.55x kindling's FP4 / FP8 path on GLM. On 2 Sparks that
predicts ~1.4k full / ~2.6k CED, which brackets the targets.

### 3.4 Replay, sessions, context, memory

- **Replay.** Our GLM stack gives 0.22-0.27 s at 64K (0540 + the session store). Here the snapshot adds 40 SWA rings
  (~3.4 MB), compressor carries, the Engram lookback and the DSpark window. With CED, a resumed prompt also replays the
  decoder over its last 128 tokens (tens of ms).
  - Global KV is ~1.65 KB a token in FP8, about a quarter of GLM's 7.4 KB. A 64K session is ~105 MB on NVMe, so a
    restore under 1 s is I/O-trivial.
- **Context (decision, 2026-10-01).** 300K a stream on 4 active streams, not 1M on every slot.
  - 4 x 300K = 1.2M tokens of FP8 global KV, ~1.85 GiB a rank (the whole row stays on each rank: one KV head).
  - Idle or parked sessions leave the pool for the NVMe session tier: global KV pages plus the SWA rings, compressor
    carries, Engram lookback and DSpark window, ~0.5 GB for a 300K session. Resume ≤ 1 s keeps resumed == fresh.
  - The kit offers 600K a request, but its 785K pool is shared: two long requests cannot both be resident.
  - FP4 main KV (the model's QAT format, half the bytes) stays an option for longer contexts later, exact if one format
    is used throughout. coolbho3k's display-carve-out pool is an option, not a target.
- **Memory.** The plan (DSV41-BASELINE section 4.1), per rank:

  | Item | GiB |
  | --- | ---: |
  | weights | 99.5 |
  | 4 x 300K pool (FP8) | ~1.85 |
  | rings, 4 slots | <0.1 |
  | workspace and graphs | ≤5 |
  | CUDA and comm | ~1 |
  | **total** | **~107.5** |

  Against ~112 GiB available after a cache drop (the kit's preflight needs 111.5), that leaves ~4.5-5 GiB.
  - The kit's floor (2.1 GiB) comes from a long prefill growing scratch.
  - Holding 4-6 GiB in the worst phase therefore needs the memory-safety work: prefill and indexer scratch sized for a
    300K prompt, 0550-style admission that parks a session to NVMe rather than dipping below the floor, and allocator
    trims. Workspace and graphs must fit 5 GiB, not 6.
  - No RAM session store (NVMe tier only). An Engram RAM row cache only from whatever sits above 6 GiB.
- **Boot.** Prepared weight folders at ~9 GB/s O_DIRECT read 99.5 GiB in ~12-15 s. The kit spends 179 + 25 s on
  loading alone.

### 3.5 What is not targeted, and why

- **Concurrency beyond 4.** The 4-slot memory plan is the user's production shape. Throughput at C8+ needs memory we
  do not have at 2.9 bpw.
- **Vision quality beyond parity.** An exact in-image bidirectional kernel would be the faithful path (the kit clamps
  it off). It is optional work, and there is no speed target for it.
- **Lower-bpw packs.** The targets hold the pack fixed, so the comparison with the kit is engine-only. A 2.0-2.5 bpw
  pack (sfxnz Viterbi, a usage re-split like jeet0733's) is a separate, measured quality decision: it is the only route
  past the section 3.2 ceilings.

## 4. How the targets are checked

1. **Baseline window** (DSV41-BASELINE section 3): RigMark x3 on the kit, glmbench cells, multiturn, MMLU-200, the
   256K long prefill with memory minima, and DSpark acceptance a phase. Every "anchor" above is replaced with these
   numbers, and the targets keep their ratios.
2. **The same harness on our build:** RigMark x3 with a fresh `cache_salt` per cold pair (protocol 1.2, rigmark PR #2,
   so "cold" is verified), glmbench, multiturn including park-to-NVMe and restart-and-resume cells, and 4 x 300K streams with 0.5 s memory
   samplers on both nodes.
3. **Gates before any speed number counts:**
   - exact suite 10/10;
   - drafted == serial past 2,048 tokens of context;
   - batched == alone at 4 slots;
   - resumed == fresh;
   - M1 top-1 ≥ 99% against the kit;
   - MMLU-200 and the tool-call gate within 1 point.
4. **Noise.** Per rigmark issue #1, two runs detect only ~9% differences. Claims of a gain under ~10% need 6+ runs a
   setup; the 2x claims need 3.

## Quality gate (decision, 2026-10-02)

Mia's kit itself agrees with the reference at only ~89%, so bit-closeness to the kit is not the bar. A speed change
whose numerics differ is accepted when **teacher-forced top-1 vs the kit oracle stays >= 96% (aim >= 98%)** and
MMLU-200 stays within 1 point of the kit's 87.5%. The engine's own structural properties are unchanged and still
required: batched == alone, drafted == serial, row invariance (they are what make replies reproducible).
