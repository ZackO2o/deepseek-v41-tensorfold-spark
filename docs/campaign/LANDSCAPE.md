# DeepSeek-V4.1-Flash: what others run and measure (2026-10-01)

Desk research only. No GPU was used, and the Sparks were touched only for read-only file inspection over ssh
(head: the kit, the model cards and the tech report). Nothing was pushed.

**Scope.** This file covers DeepSeek-V4.1-Flash only: the 2026-09-10 model with CED, CSA2, Engram, DSpark and vision.
DeepSeek-V4-Flash, V4-Flash-0731, the V4 preview and V4-Flash-Vision-Exp are an older, different architecture
(CSA/HCA, 256 experts of 4,096 x 2,048, no Engram, no CED). Their numbers are not comparable and are left out.

**How to read the numbers.**

- Almost every number here is **single-source**: one author, one fleet, their own harness, no independent rerun.
  Where two sources disagree, both are given.
- "Decode" means tok/s for one stream after the first token, unless a cell says "aggregate" (all streams' tokens over
  wall time).
- Cells are not comparable across repos without care, because harnesses differ:
  - prompt sets (counting and JSON draft far better than prose);
  - thinking on or off;
  - `ignore_eos` padding;
  - SSE chunk counting versus `usage` counting (Mia warns the first under-reports by ~3.5x);
  - repeated-token prefill fillers, which hit a hot Engram row and inflate prefill by 16-42% (MiaAI #21).
- **No V4.1 RigMark receipt exists anywhere.** RigMark (`alexellis/rigmark`, `c5a0db0` plus open PR #2) has V4-0731,
  GLM and Qwen receipts only. Our planned baseline window (DSV41-BASELINE.md section 3) would produce the first one.

Sources for this file:

- our read of the MiaAI kit, both 2.9 bpw cards and the tech report on head, and our own run notes
  (private notes);
- repos and cards read directly: vcruz305's recipe (main, last push 2026-09-25), tonyd2wild, sfxnz;
- six parallel web sweeps (2 nodes, 3 nodes, 4+ nodes and hybrids, 1 node and llama.cpp, non-Spark hardware and
  quants, forums / HF / X). Reddit was unreachable, and X only through search snippets.

## 0. Top findings

1. **The best 2-Spark V4.1 numbers are low, and there is room to win.**
   - Best single-stream, honest cells: sfxnz 2.0 bpw (prose c1 50.4 with `ignore_eos`, 42.6 real-use prose, structured
     84.2) and coolbho3k 3 bpw (~31 serial).
   - Our own Mia-kit session: 44-48 tok/s single stream, ~78 at x2.
   - The independent forum measurement of the Mia kit (helge): **code 36.4 / prose 25.1**.
   - jeet0733's 81 / 111 (C1 / C2) is one card line, unreproduced.
2. **Nobody serves V4.1 exactly.** vcruz305 documented that DSpark verify and plain decode take different EXL3
   kernels (int8 GEMV at <= 2 rows, GEMM above), so greedy output changes with speculation, with clear-preference flips.
   - Kits report non-deterministic greedy output (nero-, MiaAI "fused MoE finalize off").
   - Our drafted == serial / batched == alone / resumed == fresh contract is unclaimed ground.
3. **CED decoder bounded replay is the prefill lever, and only a few use it.**
   - knapcio (4x), 0xBakeer and sayyidfareed (1x), and YoungAi (1x: 1,055 vs 539 tok/s with `--decoder-full`) use it.
   - SGLang reports 1.37-1.56x prefill from it on datacenter GPUs.
   - The 2-Spark kits (Mia, coolbho3k, sfxnz) run all 40 layers over every prompt token.
4. **DSpark acceptance is the decode ceiling, and it is extremely content-dependent.**
   - Accepted drafts a step at k=5 (tonyd2wild): prose 1.0, narrative 0.9, summary 1.2, reasoning 3.0, math / JSON 3.3,
     tables 4.0, code 4.0-4.3, counting 4.8-5.0.
   - Prose at per-position 0.63 / 0.33 / 0.15 / 0.06 / 0.02 (8x RTX PRO 6000, native weights).
   - Prose cannot be bought with depth: a 2x-on-prose target needs a faster round, not more drafts.
5. **Drafter and embedding precision matter on low-bpw packs (conflicting evidence, measure first).**
   - sfxnz: Mia's 2.9 bpw pack quantizes DSpark to 4-bit EXL3 (`mtp_bits 4`), and measured acceptance 1.33 vs 2.77 at
     source precision.
   - Our own 3-hour session on the same layout measured 70.2% a draft (3.11 tokens a round at k=3), and coolbho3k
     saw only 53.2% vs 54.2% for a 3 bpw drafter.
   - JigSawPT: the target's `token_embd` at Q3_K vs BF16 moves acceptance 41.1 / 54.8% -> 45.5 / 60.1%.
6. **Quality at low bpw.**
   - ~3.0 bpw EXL3 experts sit at KL ~0.04-0.05 to the FP4 release (coolbho3k 0.041, diffbot 0.054).
   - 2.0-2.9 bpw is 0.06-0.44, depending on the recipe: Viterbi re-encode, calibration and usage re-split all matter.
   - MMLU drops ~3-5 points at 2.9 bpw (82.15 vs 86.96, cross-card). Abliteration costs another 2.95 (0.58 outside
     ethics).
   - **No agentic benchmark has been run on any V4.1 quant**, except helge's TEB 89/100 on the Mia kit.
7. **The 2-Spark architecture levers others found:**
   - pinned-host RDMA collectives for small messages (coolbho3k fastcomm +4%; our 0230 / 0350 is the same idea);
   - node-local Engram rows (tonyd2wild: 2.8 ms vs 5.9-7.8 over NFS);
   - Engram row prefetch off the critical path, and moving the Engram hash out of the graph;
   - GPU clock-latch hygiene (tonyd2wild: 41 -> 77 tok/s after clearing two latched GPUs);
   - display-carve-out memory for KV or embeddings (coolbho3k +1.8 GiB a GPU; christopherowen puts head and embedding
     there).
8. **Engines.**
   - vLLM supports V4.1 upstream from **v0.30.0** (2026-09-22), including B12X attention for SM12x and an SM12x FP8
     swizzle for GB10.
   - SGLang has `dev-dsv41` images and forks (Mia, knapcio).
   - **No TensorRT-LLM (issue #19481 open), no NIM for Spark, no upstream ExLlamaV3 / TabbyAPI, and no upstream
     llama.cpp** (PR #28696 open).
   - **No TensorFold V4.1 family** (0.6.0 recipe: "DeepSeek-V4 has no CUDA engine yet"; its MLX DeepSeek family is V4
     only).
   - mlx-lm merged V4.1 on 2026-09-29 without Engram.

## 1. Two Sparks (the bracket we compete in)

| Setup | Engine | Quant (routed experts / rest) | Drafter | Decode c1 (tok/s) | Concurrency | Prefill (tok/s) | Context / KV | Memory | Quality | Method notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **MiaAI kit** [repo](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks), kit README | vLLM `0.1.dev20904` + ExLlamaV3 1.4.5, TP2; moe_x kernel (PR #6) | EXL3 mul1 2.9 bpw avg (experts K3, K2 on L18-22; attn K5; head K6; indexer wk K8) | DSpark k=3, 4-bit | README pre-moe_x 31.6 (400-token prose, T=0, thinking off); with moe_x: easy prose 63-66, code 56-57, hard prose 32; `SPEC_METHOD=none` 23 | x2 42.5 (pre-moe_x) / **97** (moe_x); x4 42.8; spec-off x4 53.7 | 970-1,055 (8-64K), 872 (256K), 810 (601K, TTFT 742 s) | 600K req, 785,676-token pool (2.5 GiB), fp8_ds_mla, 64-token blocks | 99.48 GiB weights/rank; ~4 GiB free idle, **2.1 GiB floor at 601K** | WikiText-2 PPL 6.30 | Author's the kit dashboard cells; 4-token verify 60 ms with moe_x |
| Mia kit, cooperative MoE overlay | + `cooperative_moe.so` (not bit-exact) | same | k=3 | poetry 29.3, code 43.0, C1 T=0 40.2 | C2 61.1 | 32K uncached 1,138 | 600K | | | 3-seed medians, before moe_x; 54-case gate |
| **Mia kit, measured by others** | same | same, uncensored | k=3 | **helge** (forum 383242 #11): **code 36.4, prose 25.1**; say3 (382725 #102): 26; dealignai card: ~27 (45% acceptance, length ~2); sfxnz: stock k=3 prose 16.35, spec-off 23.6 | | | helge: KV 785,676 at 600K; boot 6 min 30 s with rsync | | helge: **TEB 2.6.1 hard 89/100** (T=0, effort max), 86 at T=1; TEB 2.7.1 86-94 by effort; `index_topk` 512 / 1024 / 2048 -> 88 / 89 / 89 | sfxnz attributes the low acceptance to the 4-bit drafter |
| **Our own run** of the Mia kit (head / worker, uncensored, 2026-09-20; private notes, DSV41-BASELINE.md) | as Mia + moe_x `2,12,8`, 4 seqs, vision on | dealignai UNCENSORED 2.9 | k=3 | **~44-48**; engine log 30-42 in thinking / image / 65K sessions | x2 ~78; x4 53-63 | 1,064 @39K, 1,331 @156K | 600K, 785,676 pool | 7.41 GiB after graphs | | **DSpark 70.2% a draft, 3.11 tokens a round** over 3 h (44,004 drafted). Identical resend 63 s -> 0.46 s (99.83% prefix hit). Not a RigMark run |
| **sfxnz** [repo](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark), round 36 (2026-09-27) | vLLM 0909 + vllm-exl3 + 10 bit-exact decode kernels (`canonical-e14`), TP2 | **EXL3 MCG K=2, 2.0 bpw**, Viterbi re-encode + scale refit; head MXFP8 | DSpark k=3, source precision | prose 50.41 (`ignore_eos`, ~60% post-EOS); **real-use L.A.I.L prose 42.58** (512 tokens, T=0.2); greedy 39.6 (acc 2.61); structured **84.20** | prose C2 81.0, structured C2 **156.3** | 838 @8K, 811 @32K (novel text) | 8 GiB KV pool, fp8 | 21-23 GiB free after 32K prefill | NLL 0.139 (-42%), GSM8K-100 97, MMLU 4x57 206, tool-call args 88/88 | 36 A/B rounds with noise bands; 9 runs x 2 boots; golden-flip gate. MUL1 lost to MCG (23.5 vs 28.0) |
| **coolbho3k** (emihuang) [repo](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark), forum 383583 | vLLM `e47aa780` + ExLlamaV3 `6ff3a17e` (Mia lineage), TP2 / DCP2 | EXL3 mul1 3.0 experts only; rest source format, BF16 vision | DSpark k=3 at 3 bpw + prompt lookup | pooled serial ~31 (prose 26.5, Python 37.4, explanation 36.7); card 28-46; prompt lookup on edits 47.6 | **C6 62.1** (T=0), 54.1 (T=1) | 32K 1,054-1,136; 8K 1,221; 1M at 676 (25.6 min) | **1,048,576 per request, 3.31M aggregate tokens** (NVFP4 main and indexer KV, FP8 SWA) in 1,792 MiB display memory | 2.2-3.8 GiB after tests, 0.887 GiB lowest | **KLD 0.0414, top-1 93.6%**, PPL 3.817 vs 3.798; GSM8K 63/64; ARC 123/128 | 12 serial requests, group medians. Rejected: k=4 / 5, adaptive EMA depth, verify cap. 1M decode 32.5 tok/s |
| jeet0733 C4h36 [card](https://huggingface.co/jeet0733/DeepSeek-V4.1-Flash-EXL3-C4h36) | Mia kit | Mia 2.9 bits re-split by usage | k=5 + "adaptive branch" | **81.0** (from 49.5) | **C2 111** (from 57) | | | | WikiText-2 PPL +2.46% vs FP8 (Mia base +4.05%) | One card line, method not stated. **Unverified** |
| Libertai [repo](https://github.com/Libertai/dsv41-flash-vllm-2x-spark) | vLLM nightly, eager | REAP-256E (384 -> 256 experts), native MXFP4 / FP8 | k=5 greedy | counting 49.4, code 30.4, chat 27.0, prose 19.2; no spec 13.7 | | 700-2,200 | 32K (64K OOM) | 1.5-2.0 GiB | text PPL +14.1% | 600-token cells |
| Reederey87 [repo](https://github.com/Reederey87/deepseek-v41-flash-reap-2x-dgx-spark) | vLLM 0909, eager | REAP-256E | k=5 greedy | 25.7 (acc length 3.12; per-position 0.80 / 0.58 / 0.37 / 0.23 / 0.14); no spec 13.2 | c2 21.6 (no spec) | | 64K, 841,973 KV | | needle 64K | admits an earlier ~6% SSE timing bug |
| ivanusto [repo](https://github.com/ivanusto/dsv41-flash-vllm030-2x-gb10) | **stock vLLM v0.30.0** + 9 files | REAP-256E | k=5 | 9-10 median | c8 31.5 | TTFT 7.8K 11.7 s | 32K | | | load ~69 min |
| ultradaoto / drowzeys | Mia kit | Mia 2.9 + K5 `wo_b` splice (L10-35) | k=3 | 26.1 hook vs 27.6 stock | | | 131K | | refusals 10 -> 0 of 133 | n=3, endpoint timing |
| Snail3D | sfxnz vision patches | sfxnz 2.0 | DSpark | ~21 prose | | | | 64 GB swap needed | | |
| fireworm71 (forum 384035) | antirez ds4 GGUF, TP2 | Q2 | DSpark / ngram | ~20 (from memory) | | ~300 pp | | | | Recalled, not measured |

**Best 2-Spark numbers to beat** (V4.1 only; "credible" = documented harness):

| Workload | Best credible on 2 Sparks | Who | Claimed (unverified) |
| --- | --- | --- | --- |
| Prose c1 | **42.6** real-use prose (50.4 with `ignore_eos`) | sfxnz 2.0 bpw | Mia moe_x "easy prose 63-66" (own harness); jeet0733 81 |
| Code c1 | **37.4** (coolbho3k Python), 36.4 (helge on the Mia kit), 47.6 with prompt lookup on edits | coolbho3k, helge | Mia moe_x 56-57 |
| Structured c1 | **84.2** | sfxnz | |
| C2 aggregate | 156.3 structured, 81.0 prose | sfxnz | jeet0733 111 |
| C4-C6 aggregate | C6 **62.1** | coolbho3k | our own x4 53-63 |
| Cold prefill 8-64K | **~1.0-1.2k** (Mia 1,055 @32K, coolbho3k 1,221 @8K, the user's 1,064 @39K / 1,331 @156K) | Mia, coolbho3k | Libertai "700-2,200" on REAP |
| Replay | APC in RAM only: identical resend 0.46 s (user, Mia kit) | | |
| Context | **1,048,576 per request, 3.3M-token aggregate pool, C6** | coolbho3k | |
| Memory floor | Mia 2.1 GiB at 601K; coolbho3k 0.887 GiB lowest | | |
| Quality | coolbho3k KLD 0.041; helge TEB 89 on Mia 2.9 | | |

## 2. Three Sparks (context: 1.5x our compute and bandwidth)

| Setup | Engine | Quant | Drafter | Decode c1 | Aggregate | Prefill | Context / KV | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **christopherowen** [repo](https://github.com/christopherowen/spark3-vllm-ds41f), forum 384438 | LIL vLLM `karmic-kraken-beta` + 26 patches, B12X kernels, L2 prefetch, SP prefill, switchless dual CX-7 | native FP4 / FP8 (head and embedding in display carve-out) | DSpark 5, adaptive verify, dead-row skip, NVFP4 drafter head | thinking off: **prose 59.1, code 80.6, JSON 77.0**; thinking on: prose 50.3, code 59.7. Step 42.4 ms (prose) / 46.9 ms (code) | C2 / C4 / C8 code (off) 118.2 / 165.1 / 233.0 | **real text ~3.8-3.9k up to 200K** (filler 4.9k) | 262K, 1,348,708 KV in 2.2 GiB/rank | Welch 95% CIs; overlapping samples discarded; 32K replay 0.245 s warm; min MemAvailable 6.4-7.5 GiB; boot ~150 s. Accepted drafts a step at C1: prose 1.22, code 1.95, prose (off) 1.73, code (off) 3.21, JSON 2.92 |
| MiaAI TP3 [repo](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | SGLang `dev-dsv41` + overlay | native MXFP4 / FP8 | k=5, verify cap conf 0.1 | the kit dashboard prose 51.0; own bench prose 34.2, code 62.2 | C4 85.4 | ~1.2-2.0k | 256K configured, ~32K usable at C4; 1,670.75 B/token/rank | acceptance length prose 2.04, code 3.98 |
| tonyd2wild EXL3-TP3 | vLLM + cuda-exl3, virtual heads 64 -> 72 | Pollard EXL3 3.51 | k=5 | 51.5 per-stream (8 categories); code 70.1, prose 29.5 | C6 152.9 | ~1.12-1.20k | 300K, 678,950 pool | |
| masquerator-coder | Mia TP3 SGLang | native | k=5 | prose 33.6, code 79.3 | C4 prose 76.5 | 1.6-2.2k | 262K | production P50 TTFT 0.98 s, P99 114 s over 2,403 requests |
| jakejharris jspark3 | vLLM (tonyd2wild lineage) | Pollard 3.5 | k=4 | prose 33-34, code 67-76 | C6 146 | 64K cold TTFT 42.6 s | 300K | repeat TTFT 0.47-0.71 s |
| benkatzir | SGLang, Mia | Pollard 3.5, **routing cut to top-3** | k=5 | | C6 208 | | 524K | 500K x 6: 0/24 exact lookups (quality failure) |

## 3. Four or more Sparks, and hybrids (upper reference)

| Setup | Nodes / engine | Quant | Decode c1 | Aggregate | Prefill | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **knapcio** [repo](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4) v2.3 | 4x, SGLang `dsv4.1` + RoCEnante, switched, b12x MoE, EP1 | native (MXFP4); exact FP8 twins of `wo_a` and the draft head; NVFP4 rejected ("no bandwidth saved on GB10") | **prose 89.7, code 131.9, structured 157.8** (the kit dashboard, 256 tokens, thinking off) | prose C4 165.9, code C16 456 | real text 3.9-4.9k | **Uses decoder SWA bounded replay** (prompt logprobs return 400); Engram 4 GiB row cache (67-76% hits); qeval 72/75; greedy-continuation KLD 0.003-0.007. voktolom's independent run of v2.3: prose 72.3, code 127.0 |
| **tonyd2wild** [repo](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark), speedrun 2 (09-19) | 4x, vLLM TP4, b12x RoCE one-shot all-reduce, CUDA graphs | Pollard EXL3 3.5 bpw, abliterated | per-stream C1 67.4; code 90.7, tables 98.6, math 87.5, prose 39.0, narrative 31.4, counting 113 | C6 189.3; C12 code 474.7 | 1,939-2,048 (2.9-93K) | k=5. Step: GPU 49-52 ms (~45 ms batch-independent); host bubble 1.0-1.2 ms; Engram page faults cause the spikes. "Lanes age": ~30% slower prefill after 15 h. Engram on NFS 5.9-7.8 ms a step vs 2.8 local. Clock latch: 41 -> 77 tok/s. NVFP4 pack never served (14 boots) |
| luxingcom LuZ / ntxf31415 | 4x, SGLang ring | native | code 89.2, prose 51.9 (55.0 with b12x MoE off) | code C16 572.6 | 5.0-5.8k | b12x MoE `a8` nondeterministic |
| ZackO2o | 4x vLLM / SGLang | native | vLLM 50.9; SGLang 104.1 (long answer) | | 1.45k vLLM / 4.7k SGLang | **NVMe prefix tier: 92K prompt in 0.86 s after restart**. Cross-project table with reading rules |
| vcruz305 TP4 [recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe) | 4x vLLM + vllm-exl3 | EXL3 4.75 bpw | dynamic k (4 at c1-4, 3 at c5-10): **c1 code 64.8** | c4 109.6, c10 166.7 | | Grouped padded-MoE kernel (PR #39), bit-identical; measured real-traffic expert reuse 16.1 unique of 24 slots at 4 rows, 44 of 96 at 16; k=2 acceptance code 2.69, prose 1.82 |
| drowzeys TR3 hybrid | 4x vLLM | 320 experts EXL3-TR3 K3 + 64 native (3.22 bpw) | code 37.3, prose 22.2 | | | KL 0.032 vs native (EXL3 3.5: 0.057) |
| nacyot | 4x vLLM | native | prose 45.2, code 86.0 | 4 streams 78.6 | 1.5-1.9k | **disk KV offload: a 493K session restores in 7.8 s vs 392 s cold** |
| rhys101 / sumsliu / im0xMagnus / Lightfoundry | 8x | native / uncensored FP8 | code 131.7, prose 87.3 (rhys101); 100 (sumsliu) | 8-stream code 508.8 | 3.5-4.4k | |
| **tpurtell ds41rt** [repo](https://github.com/tpurtell/ds41rt) v15 | custom Rust / CUDA: RTX PRO 6000 lead (attention, Engram, router, shared, DSpark, head) + 4 Sparks as expert servers | official / NVFP4 / EXL3 | official 1 RTX: **C1 code 146.1**, counting 182.4; EXL3 5090-class + **2 Sparks: code 122.5**, weighted 88.2 | code C16 1,057 | 7.8k (official), 2.0k (EXL3 + 2 Sparks) | **This is NEXT-HETERO-5080's attention/FFN split, built.** Per-position acceptance: code 91% (5.37), fable 55% (1.86), JSON 71-74%. FP4 global KV + FP8 SWA; RAM snapshot tier |
| hughmadden ds41rt-rtx5090 | RTX 5090 + 4 Sparks | | prose 86.9 | | 218K in 34.3 s | **copy-window drafts (credited to TensorFold): +7-18% on near-verbatim rewrites, 95% of copied drafts accepted** |

## 4. One Spark

| Setup | Engine | Quant | Decode | Prefill | Notes |
| --- | --- | --- | --- | --- | --- |
| vcruz305 TP1 | native ExLlamaV3 fork | **SAGE 1.59 bpw** (mixed K1-K6) + 10.6 GiB attn / MTP overlay | fresh prompt **17.5 median** (acc 0.889, conf gate 0.7); warm repeat 33.6; no drafter 15.2 | 154-261 | ~107 GiB resident. **Speculation not output-exact** (int8 GEMV vs GEMM). Our own try (2026-09-17): 0.22-0.88 tok/s safe mode, the fast mode OOM-crashed the box 5 times; "not deployable for agent use" |
| YoungAi ds4 VQ [card](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash) | closed ds4 fork | VQ ~1.5-1.6 bpw experts, q4_K rest | greedy 30.7; speculative 43.0 (3.04 / round) | **1,055 @12.5K (539 with `--decoder-full`)** | Evidence that CED replay ~2x prefill on GB10. Teacher-forced top-1 72.8-82.3% (weak) |
| 0xBakeer / sayyidfareed | own PyTorch + Triton | pruned keep-sets (154-170 of 384 experts), CB3 | 17-37; median code 27.8 | 147-337 | Lossy pruning; decoder SWA replay on by default |
| antirez ds4 | C / CUDA, SSD streaming | Q2 GGUF | 9.3 | ~100-384 | |
| vcruz305 llama.cpp | fork `runtime/deepseek41` | Q2_K | 2.8 | 1.9 | Upstream PR #28696 open |

## 5. Other hardware (for scale, not comparison)

| Setup | Hardware | Decode c1 | Aggregate | Prefill | Notes |
| --- | --- | --- | --- | --- | --- |
| killy-netsphere SGLang | 4x RTX PRO 6000, native | code 365, essay 163 | code 3,081 at 32 | 13.5-16.3k | DSpark 3.2x on prose, 6.6x on code vs 37 tok/s without |
| 0xSero [repo](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) | 4x RTX PRO 6000, native, Engram on NVMe + 64 GiB RAM cache | 196-232 (synthetic repeated input) | C8 600-748 | 7.0-7.4k | Acceptance inflated by repeated input. Also an open PR "tuned V4.1 recipe for 4x DGX Spark (knapcio + adaptive verify)" in `local-ai-registry` #61 (not read) |
| diffbot [card](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) | 2x RTX PRO 6000, EXL3 3.0 + int4 Engram | 123.8 | 4 streams 248-269 | 8.85k @46K | 3.9 tokens a verify. Agent turn (34K cached + 6K new) 1.11 s TTFT |
| tacos4me | 2x RTX PRO 6000, pruned to 224 experts | 141 | | ~9.8k | text-only pruning profile dropped GPQA-D to 67.7% |
| yangqinhuan | 8x RTX PRO 6000 SE, vLLM TP8 | prose 94-108, counting 250-280 | prose 1.1k at 100 | | prose per-position 0.632 / 0.330 / 0.153 / 0.055 / 0.020 |
| JigSawPT | 1x RTX 5090 + NVMe streaming (llama.cpp fork) | 5.1 new, 21.3 cached | | | `token_embd` precision changes DSpark acceptance |
| ds4 M3 Ultra / Jackten 3x M5 Max / mlx-lm PR #1895 | Mac | 42-45 with DSpark / 35.6 / 15-17 (4x M5 Ultra, no Engram) | | 846 @63K (M3 Ultra) | TensorFold has no V4.1 |
| vLLM recipe | GB300 DGX Station TP1 | 90.5 | 420 at 16 | | |
| Dynamo PR #15146 | B200 / GB200, TP4, DSpark 3 | p50 52-82 tok/s/user at 168-184 concurrent | ~1-1.15k tok/s/GPU | | agentic trace, 64K in / 400 out, 90% KV reuse |
| SGLang blog | 4x GB300 | plain 223.5; DSpark 873.6 with **simulated** acceptance 5.5 | | | random input, `match-expected`: not a real acceptance. FP4 KV: 288 B main + 68 B indexer an entry |
| sanjay920 | 4x H100, SGLang, KV + Engram in host RAM | code 203, prose 107-130 at 400K cached | 834-1,154 for 30 x 400K | | 16.9 at 1M |
| DeepSeek API (Artificial Analysis) | | ~209-218 | | TTFT 0.97 s | AA index 39 (max effort); very verbose (250M tokens in the index run) |

## 6. Quality versus bits (V4.1 only)

Reference points, all on the release weights unless noted:

- DeepSeek card (max effort): Terminal-Bench 2.1 **90.6**, DeepSWE v1.1 74.2, TB 3.0 30.0, TB 4.0 31.2, NL2Repo 65.4
  (the card says 64.0), GPQA-D 90.9, Codeforces 3471.
  - Scaffold spread: DeepSWE 65.5-74.2, TB 2.1 84.1-90.6.
  - Effort 25 -> 100: TB 2.1 82.4 -> 90.6, DeepSWE 66.0 -> 74.2. "low" = 50 in the API.
- NVIDIA's own TB 2.1 run on the source checkpoint: **81.6** (NVFP4 82.16). Harness matters more than precision there.

| Pack | Bits (experts) | KL / NLL vs reference | Task scores | Source |
| --- | --- | --- | --- | --- |
| NVIDIA NVFP4 | 4.5 (W4A4 cast) | AtomicChat: KL 0.036 / 0.020 / 0.009 (neutral / code / agentic) vs repeat-run noise 0.016 / 0.010 / 0.005 | GPQA-D 91.3 vs 91.0; TB 2.1 82.2 vs 81.6; SciCode 55.8 vs 54.4 | nvidia card, AtomicChat |
| AMD Quark MXFP4 (W + A) | 4 | | GSM8K 92.3 vs 92.9; GPQA-D 90.4 vs 89.4 | amd card |
| vcruz305 EXL3 4.75 | 4.75 | | none published | |
| bot-lab-21 Pollard | 3.51 (K3 / K4) | ΔNLL -0.003 (calibration rows, not held out) | HumanEval 0.951 / HE+ 0.921 | card |
| vcruz305 SAGE | 3.30 | **KL 0.091, top-1 94.0%** (general 0.159, code 0.035) | | card |
| drowzeys TR3 hybrid | 3.22 | KL 0.032 (EXL3 3.5: 0.057) | battery 25/27 (native 24/27) | repo |
| coolbho3k | 3.0 mul1 | **KL 0.041, top-1 93.6%**, PPL 3.817 vs 3.798 | GSM8K 63/64, ARC 123/128 | card |
| diffbot (coolbho3k experts + int4 Engram) | 3.0 | KL 0.054 / 0.032 / 0.010 | GSM8K-200 98.5, HumanEval 95.7 | card |
| **Mia 2.9 (our pack's base)** | 2.9 mul1 (K3, K2 on L18-22) | satgeze vs API: KL 0.124, top-1 98.4% (method unclear); WikiText-2 PPL +4.05% vs FP8 (jeet0733) | **MMLU-14k 82.15** vs FP8 86.96 (dealignai, cross-card); GSM8K-50 96%; helge TEB 89/100 | dealignai, jeet0733, forum |
| **dealignai UNCENSORED 2.9 (our pack)** | 2.9 | | **MMLU-14k 79.20** (-2.95; non-ethics -0.58); HarmBench comply 99.4% | card |
| jeet0733 C4h36 | 2.9 re-split | PPL +2.46% vs FP8 | | card |
| sfxnz Viterbi | 2.0 MCG | NLL 0.139 (from 0.243) | GSM8K-100 97, MMLU 206/228, tool args 88/88 | repo |
| diffbot 2.0 (old) | 2.0 | KL 0.361 / 0.215 / 0.077 | GSM8K 97.5, HumanEval 91.5 | card |
| vcruz305 SAGE | 1.59 (K1-K6) | **KL 0.189, top-1 91.2%** (general 0.409, code 0.054, math 0.183) | | card |
| antirez Q2 GGUF | ~2 | KL 0.260, top-1 78.3% (Lucebox) | | |
| pipenetwork MLX ladder | 8 / 6 / mixed 4-8 / 4 | layer divergence 0.008 / 0.018 / 0.034 / 0.058; **Engram at 6 bits indistinguishable, 4 bits +7.3%** | | card |

Reading:

- Code and reasoning degrade least, open-ended general text most (SAGE subsets, diffbot neutral / code / agentic).
- **Recipe beats bits at 2-3 bpw:** Viterbi re-encode cut NLL 42% at the same 2.0 bpw, and the usage re-split cut PPL
  loss from 4.05% to 2.46% at 2.9.
- **The quant costs more than the abliteration.** Abliteration costs -0.58 MMLU points outside ethics, against
  ~-4.8 for the 2.9 bpw quant (cross-card).
- **No agentic evidence on quants beyond helge's TEB 89** (Mia 2.9, thinking max). Our tool-call gate (210 cases) and
  MMLU-200 will be among the first.

## 7. Architecture notes others published (beyond the tech report)

From the tech report (51 pages, copied read-only from head to the workstation's `/tmp`):

- **CED.** Layers 0-19 encode; the decoder's global KV is projected from H_19. Decoder SWA Bounded Replay prefills
  only the last n_win = 128 tokens through the decoder: "nearly halving total prefill computation". It was simulated in
  post-training. Encoder SWA Bounded Replay rebuilds a missing SWA state after a global-KV hit by replaying 128 tokens.
  - DeepSeek does not persist SWA KV at all (a 10%-of-DRAM pool with a minutes-long TTL).
  - Global KV persists for 72 h on SSD.
  - Both replays are approximate by design ("not mathematically identical across positions").
- **CSA2.** Full mode at [2, 8, 14, 20], Reindex at [24, 28, 32, 36], Reuse elsewhere. The hierarchical indexer is
  decoder-only and introduced in post-training: layer 20 picks 2,048 blocks of 8 for later Reindex layers. "Reuse Mode
  layers execute with only 15 kernels during prefill and 11 during decode."
- **FP4 main KV.** E2M1 + one E4M3 scale per 16, no global scale, quantized after RoPE; QAT-trained. SWA stays FP8
  ("sensitivity"). 890 B a token global (SGLang: 288 B main + 68 B indexer an entry; 3 ratio-2 producers + 1 ratio-1).
  - In FP8 the same is ~1.65 KB a token. vLLM's real allocation is ~3.4 KB a token (Mia), and MiaAI TP3 counts
    1,670.75 B / token / rank.
- **Single-Pass mHC.** Coefficients shifted by one block: one read and one write of the residual (Mega-mHC).
- **Engram.** Layers 1 and 14; deterministic addressing. DeepSeek prefetches from host memory via background RDMA,
  overlapping the first block.
- **DSpark.** 3 blocks, SWA 128, 5 positions a pass, Markov head, confidence head. DeepSeek's scheduler uses the
  confidence head with profiled throughput curves to pick the verify length per request. Trained after pre-training
  and co-trained through post-training (no gradient into the backbone).
- **No serving speeds and no acceptance numbers** are in the report. EPD disaggregation in production.

From others:

- **SWA geometry on GB10.** FlashInfer has no SM120 sparse-MLA decode kernel for 32-token pages, and DeepGEMM wants 64
  states. Kits use 64-token KV blocks (tonyd2wild boots 4-6, Mia `patch_sm120_block64`, vLLM #56509 / #59385 open).
- **`persistent_topk`** oversubscribes 48 SMs and needs 128 KB of shared memory (GB10: 99 KB); `top_k_per_row_decode`
  is 1.6-3.6x faster (tonyd2wild fix 7).
- **Engram correctness traps.**
  - CUDA-graph replay keyed by the wrong layer fills layer 1 from layer 14's table, and acceptance collapses to 1.00
    (vcruz305 fix 1).
  - A per-rank row-offset bug made ranks 1-3 read rank 0's rows (tonyd2wild).
  - With graphs but no Engram prestage, output is NaN (Libertai).
  - Engram rows are Zipfian: the top 100M rows give 92.7% held-out coverage (bot-lab-21), and a 4 GiB row cache hits
    67-76% (knapcio).
- **Vision on SM12x.** No 1,152-wide sparse-MLA kernel, so in-image bidirectional attention is clamped off (Mia).
  Snail3D saw garbage output from the `mm_prefix` mode at 110K.
- **Draft experts silently zero** under the EP weight filter: acceptance flat, code no faster than prose; fixing it
  gave +9-20% (vcruz305 fix 4).
- **Prefix caching.** DSpark disables vLLM's prefix cache unless `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=64` (satgeze).
  SGLang SWA prefix tails give 0% revisit hits without `--swa-full-tokens-ratio` (MiaAI #31).
- **GB10 hygiene.** GPU clock latch at 630-950 MHz needs a power drain; kernel 7.0.0-1019 has `CmaTotal 0` (NCCL
  ENOMEM; the user's `cma=128M`); "lanes age" (~30% prefill loss after 15 h).

## 8. Techniques: adopt, consider, avoid

| Technique | Evidence | For us |
| --- | --- | --- |
| Decoder SWA bounded replay (CED) as a knob | report; YoungAi 1,055 vs 539; SGLang 1.37-1.56x; knapcio | **Adopt** (`CED_PREFILL=replay\|full`), part of the exactness contract (DSV41-BASELINE section 2.3) |
| Exact speculative decoding (one kernel path for 1..R rows) | vcruz305 proves the kits are not exact; row-invariant EXL3 linear upstream | **Adopt** (the differentiator) |
| Node-local Engram rows, O_DIRECT, prefetch before the forward, hash outside the graph | tonyd2wild 2.8 vs 5.9-7.8 ms; Mia +25-50% prefill; sfxnz WILLNEED 184 -> 838 prefill on novel text | **Adopt** (already planned) |
| Engram RAM row cache (Zipfian) | knapcio 67-76% hits at 4 GiB | **Consider** only if the memory floor allows (we have ~4 GiB, not 40) |
| Small-message pinned-host RDMA collectives | coolbho3k fastcomm +4% bit-exact; tonyd2wild b12x one-shot | **Have** (0230 / 0350) |
| Source-precision drafter (and embedding) | sfxnz 1.33 vs 2.77; JigSawPT embedding; coolbho3k says small | **Measure first**: our pack's DSpark is 4-bit (3.17 GiB/rank); native MXFP4 draft experts are ~3.7 GiB/rank |
| Adaptive verify length (confidence head) | DeepSeek; christopherowen; knapcio verify cap; vcruz305 dynamic k (+15% c1 vs k=3); coolbho3k found adaptive EMA did not pay | **Adopt** via our cost-derived depth with the confidence head as input |
| Prompt lookup / copy drafts on edits | coolbho3k 47.6 vs ~31; hughmadden +7-18% (TensorFold copy-window idea) | **Adopt** (our 0020 suffix lookup) |
| FP4 main KV (the model's QAT format) | report; coolbho3k NVFP4 3.3M tokens; ds41rt | **Option**: halves the pool; exact under our rules if one format throughout |
| Display carve-out memory | coolbho3k +1.8 GiB KV; christopherowen head + embedding; reads ~160 GB/s vs 250 | **Consider** for cold data (embedding, KV) only; requires `nvidia_drm modeset=1` (a GX10 report of a 700 MHz latch) |
| MoE TP vs EP | 1,152 = 9 x 128: pure TP2 needs no padding; SGLang TP4 +2.2% over EP4 | **TP** (what the Mia kit and we do) |
| Grouped expert decode (one tile decode per expert, not per slot) | vcruz305 PR #39: c4 64 -> 85, c10 83 -> 140 | **Have** (upstream grouped kernel) |
| SP prefill, indexer TP-split | christopherowen, knapcio (+~20%), tonyd2wild | **Consider** for prefill |
| SwiGLU clamp fused into the GLU kernel | tonyd2wild +3% decode | **Adopt** (trivial, bit-identical) |
| NVFP4 experts on GB10 | knapcio "no bandwidth saved"; tonyd2wild 14 failed boots; 4.5 bpw > EXL3 2.9 | **Avoid** |
| REAP pruning / top-k routing cuts | PPL +14%, benkatzir 0/24 long-context lookups, tacos4me GPQA 67.7 | **Avoid** |
| b12x MoE `a8` / fused MoE finalize | nondeterministic greedy (ntxf31415, Mia) | **Avoid** |
| k = 10 drafts | -17 to -38% everywhere | **Avoid**; DSpark's block is 5 |
| Eager mode without graphs | 5 -> 41 tok/s with graphs (tonyd2wild) | Graphs are mandatory (ours are) |
| Attention/FFN split with a discrete GPU | ds41rt: 2 Sparks + 5090-class EXL3 code 122.5 | **Proven by others**; our NEXT-HETERO-5080 path, not this project |

## Sources

Local:

- head, read-only: `~/src/dsv41-2x-kit/README.md`, `docs/cooperative-moe.md`;
  `~/models/dsv41-{mialab,uncensored}-2.9bpw/README.md`; `DeepSeek_V41_Tech_Report.pdf`;
  `~/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw/README.md`.
- `~/.cache/rigmark` @ `c5a0db0` and GitHub (no V4.1 receipt; PR #2 open).
- `<upstream 0.6.0 checkout>/docs/recipes/{cuda,deepseek-v4-flash}.md`.
- `private notes of an earlier kit run`.
- `glm53-tensorfold-spark/docs/{DSV41-BASELINE,NEXT-DEEPSEEK-V41-FLASH,ROOFLINE,NEXT-HETERO-5080}.md`.

Web (as listed per row above), principally:

- Kits:
  - [MiaAI 2x](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
  - [MiaAI 3x / 4x](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)
  - [sfxnz](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark)
  - [coolbho3k](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark)
  - [vcruz305](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe)
  - [tonyd2wild](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
  - [christopherowen](https://github.com/christopherowen/spark3-vllm-ds41f)
  - [knapcio](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)
  - [tpurtell ds41rt](https://github.com/tpurtell/ds41rt)
  - [hughmadden ds41rt-rtx5090](https://github.com/hughmadden/ds41rt-rtx5090)
  - [ZackO2o](https://github.com/ZackO2o/DeepSeek-V4.1-Flash-4x-GB10-1M-Full-Recipe)
  - [Libertai](https://github.com/Libertai/dsv41-flash-vllm-2x-spark)
  - [Reederey87](https://github.com/Reederey87/deepseek-v41-flash-reap-2x-dgx-spark)
  - [ivanusto](https://github.com/ivanusto/dsv41-flash-vllm030-2x-gb10)
- Cards:
  - [deepseek-ai](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  - [nvidia NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)
  - [Mia 2.9](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
  - [dealignai](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw)
  - [coolbho3k 3bpw](https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw)
  - [jeet0733](https://huggingface.co/jeet0733/DeepSeek-V4.1-Flash-EXL3-C4h36)
  - [vcruz305 SAGE 1.59](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw)
  - [diffbot](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)
  - [bot-lab-21](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard)
  - [AtomicChat](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia)
  - [pipenetwork](https://huggingface.co/pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit)
- Forums:
  - NVIDIA [382725](https://forums.developer.nvidia.com/t/deepseek-v4-1-flash/382725), [382897](https://forums.developer.nvidia.com/t/382897), [383242](https://forums.developer.nvidia.com/t/383242), [383583](https://forums.developer.nvidia.com/t/383583), [384438](https://forums.developer.nvidia.com/t/384438), [384548](https://forums.developer.nvidia.com/t/384548)
- Engines:
  - [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) and v0.30.0 release notes
  - [SGLang V4.1 kernel blog](https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization)
  - [llama.cpp #28696](https://github.com/ggml-org/llama.cpp/pull/28696)
  - [TensorRT-LLM #19481](https://github.com/NVIDIA/TensorRT-LLM/issues/19481)
  - [mlx-lm #1895](https://github.com/ml-explore/mlx-lm/pull/1895)
  - [Dynamo #15146](https://github.com/ai-dynamo/dynamo/pull/15146)
- Quality: [Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v4-1-flash), arXiv 2609.19969 (tech
  report), arXiv 2607.05147 (DSpark).
