# Results

Everything measured on one pair of DGX Sparks (GB10, 128 GB each, CX7 link, RoCE), 2026-10-01 to 10-03, on
`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`. Development ran in numbered test windows (G1-G11); the window
names are kept so the raw files in [`../results/`](../results/README.md) can be matched to a row. How each cell is
measured: [BENCHMARKS.md](BENCHMARKS.md).

## 1. The baseline: MiaAI-Lab's vLLM kit on the same pair (2026-10-01)

| | |
| --- | --- |
| Kit | MiaAI-Lab 2x DGX Spark kit, vLLM `0.1.dev20904` TP=2, DSpark k=3, moe_x `2,12,8` engaged, max model length 600,000, 4 sequences, prefix caching on, `fp8_ds_mla` KV, `MAX_NUM_BATCHED_TOKENS=2048` |
| KV pool | 773,163 tokens (2.5 GiB pinned) |
| Nodes | both rebooted inside the window; GPU clocks 2,223 MHz |

| metric | kit |
| --- | ---: |
| Code, 1 stream (64-token reply, T=0 / T=1; 512-token reply) | 41.9 / 40.6; 45.0 |
| Prose: 200-token essay; short chat T=0 / T=1 | 32.5; 19.2 / 17.7 |
| Structured: count 1-200; JSON; primes list | 38.0; 50.2; 47.4 |
| Copy-heavy edit (1,024 tokens) | 55.1-56.2 |
| C1 / C2 / C4 aggregate | 32.2 / 46.7 / 37.6 (C4 per stream 9.2-10.2) |
| Cold prefill 8K / 32K / 64K / 128K / 256K | 1,073 / 1,075 / 1,060 / 1,031 / 983 tok/s |
| MemAvailable floor (worst node, every phase) | head 4.94, worker 4.40 GiB |
| Start to `/health` (freshly rebooted nodes) | 378 s |
| DSpark tokens a round, mixed | 2.27 (MMLU 1.46, multiturn 1.83, code+chat ~1.8, structured ~3.1, copy-heavy ~3.8) |
| MMLU-200, 0-shot, thinking off, greedy | 87.5% |

Two findings from the baseline that shaped the work: the kit's C4 is *below* its C2 (4 x 4 verify rows run slower
than 2 x 4), and the kit renders `reasoning_effort: "low"` as effort 25 (vLLM's mapping), not DeepSeek's 50.

## 2. The final configuration (G10 / G11, engine = this repository's patches)

| tok/s (T=0 / T=0.7; C = decode aggregate) | code | prose | structured | C1 | C2 | C4 | 1-row window |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| **final** | **79.0-81.5 / 72** | **40.9-41.2 / 42** | **116.6 / 117.5** | **82-83** | **67.1** | **93.5** | **26.9 ms** |
| kit | 41.9-45 | 32.5 | 38-50 | 32.2 | 46.7 | 37.6 | |
| final / kit | 1.8-1.9x | 1.26x | 2.3-3.1x | 2.6x | 1.44x | 2.5x | |

`exact_all True` (drafted == serial) in every run. DSpark tokens a round: code 3.88, prose 1.57-1.59, structured
5.91.

Ship gates on the final configuration (a test server started by `scripts/serve.sh` from the production config):

| gate | result |
| --- | --- |
| teacher-forced top-1 vs the kit | 0.9963 (first copy 0.9502) |
| MMLU-200 0-shot | 87.5% (kit 87.5%) |
| structured output (json_schema x thinking off / on, tool choice) | 12 + 10 cases pass |
| tool chains (`chains.py`), thinking off / high | 11 / 12, 11 / 12 (pass line 10) |
| tool-eval-bench category C | 8 / 8, score 100 |
| 30-min soak | 529 requests, 0 errors, 77 intentional cancels, drained, 17*23 = 391; MemAvailable min head 5.11 / worker 4.07 GiB |
| stress 4 x 300K (one 299K prefill + three 64K prompts decoding 2,048 tokens, ignore_eos) | every stream complete; first token of the 299K prompt at 217 s; **worker MemAvailable min 4.01 GiB: under the 5 GiB target** |

Earlier gates with the same prefill path (G7 engine commit): MMLU-200 replay / full / kit 88.5 / 88.0 / 87.5% (199 of
200 answers equal); MMLU with a 20-question preamble, replay / full 81.1 / 81.1% (178 of 180 equal); needles at 32K /
128K / 299K found in 19.9 / 74.0 / 195 s (replay) and 32K / 128K in 34.7 / 137.3 s (full).

### Prefill (G7, one slot, cold, fresh random text after a 2K warm-up; tok/s)

| config | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: |
| **replay + every prefill lever (production)** | **1,833** | **2,043** | **2,068** | **1,953** |
| replay, base kernels | 1,343 | 1,421 | 1,407 | 1,379 |
| full (no replay), every lever | 969 | 1,004 | 1,003 | 876 |
| full, base | 756 | 772 | 764 | 739 |
| kit | 1,073 | 1,075 | 1,060 | 1,031 |

TTFT at 128K: 67 s. One run of the production prefill was anomalous (854 tok/s at 128K: both GPUs at ~36 W instead of
~52 W at normal clocks) and did not reproduce in two later runs.

### Start

34-44 s from `docker run` to `/v1/models` on the G10 / G11 test servers (4 slots x 300K pool, prepared folders,
compiled kernels cached, page cache dropped); `m2bench` boots in 37-39 s. The weights are read back from the
prepared folders with parallel O_DIRECT readers; M1 measured 18 s for the weights alone.

## 3. How it got there

| step | code | prose | structured | C1 | C2 | C4 | what changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G4 | 57.8 | 31.3 | 86.0 | 60.5 | 46.1 | 70.7 | the family running with CUDA graphs and RoCE; expert load path; router fix |
| G7 | 69.1 | 37.4 | 102.2 | 70.0 | 48.9 | 67.0 | the router as a CUDA GEMV (1-row window 33.1 -> 29.5 ms); bf16 exchanges, mHC split; prefill levers (1.7-1.95x the kit) |
| G8 | 72.7 | 39.0 | 109.1 | 69.1 | 58.9 | 77.5 | the GPU waits for Engram rows instead of the host; slot-agnostic row graphs (C2 +34%, C4 +12%); round plan over TCP; speculative DSpark pass |
| G9 | 80.2 | 39.9 | 114.5 | 76.0 | 63.0 | 92.8 | decode glue (bit for bit) + bf16 mHC weights; routed-expert pruning p85k (x1.05) |
| G10 (final) | 79.0-81.5 | 40.9-41.2 | 116.6 | 82-83 | 67.1 | 93.5 | cross-op L2 prefetch, 12 MiB a site (code +2.5%, prose +3.5%) |

## 4. Measured and not adopted

| lever | result |
| --- | --- |
| **Lossy verify budget** (`TF_DSV41_VERIFY_BUDGET`: verify drafts with fewer experts) | B=4: code 89.3 / prose 41.1 / C4 100.7, but only 12% of greedy tokens equal the exact reply (first divergence at token ~13), MMLU-gen 84.5% vs 86.0%, tool chains 4 / 12. B=2 failed the server's own canary. B=0: code 184.6 from degenerate loops the drafter accepts, MMLU-gen 79.5%, chains 1 / 12. **Dropped**: even as an opt-in it breaks tool calls. |
| Draft trees (parent-conditioned) | the target's 2nd choice is the draft's 2nd only 17% of the time at position 1 (line 35%); tree replays +0.1%. The drafter is wrong, not near. Not run on the GPU. |
| PDL (programmatic dependent launch) | 1-row window within +-0.1 ms in every part (segments, singles, experts, Triton): no gain |
| x3dn dense kernels (v1 persistent, v2 upstream-grid) | v1 9-24% slower than upstream on every shape in the engine; v2 wins some 1-row shapes, loses at 2 rows (prose -9%) |
| Joint multi-slot draft depth | C2 -2.1%: the windows do not get cheaper enough |
| Long projection plan | no 1-row gain |
| 4-bit / trimmed draft head | -0.27 ms a pass but code acceptance -0.11 tokens a round; trim: pass unchanged |
| More DSpark candidates (K 256 / 1,024 a rank) | prose +0.6% / +0.1% |
| DSpark self-distillation (LoRA deltas on our own drafting logs) | delta A: prose **+5.5%** (43.3) but code -4.4%; delta B: +17.6% tokens a round offline, only +11% in the engine, code -5.7%; a balanced re-capture: prose +4.1%, code -2.9%. The training port disagrees with the engine's drafter after position 1 (agreement 0.39); that comes first. |
| x3pf prefill experts, x3tc tensor-core experts | slower at every size (x3tc 3-5x) |
| Streaming top-k from 4K keys | -2% |

## 5. Memory

| run | head | worker |
| --- | ---: | ---: |
| decode benchmarks (1-4 streams) | >= 10.3 | >= 7.3 |
| prefill to 128K | 7.6 | 5.66 |
| soak, 30 min | 5.11 | 4.07 |
| stress 4 x 300K (six runs on the final engine) | 5.5 | 3.6-4.4 |
| stress, the G8 engine | | 4.07 |
| stress, the G4 engine | 5.46 | 4.38 |

MemAvailable minimum, GiB, 0.5-1 s samplers. The dip is rank 1's host anonymous memory growing from ~0.9 GB at boot
to 2.7-4.7 GB during the 299K prefill (and ~7 GB later), not returned. Excluded: the session tier, malloc arenas
(`MALLOC_ARENA_MAX=2`: same), the Engram row cache (17 MB), prefetch-ahead, L2 prefetch, pruning. Open.
