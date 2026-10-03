# G2: M1 bring-up and gates on both Sparks (2026-10-02, 03:04-03:51)

DeepSeek-V4.1-Flash on our engine (branch `dsv41-060`, TP=2, one slot, serial path), against the kit's oracle from
the baseline window (`results/BASELINE-20261001/oracle-kit.json`: 8 prompts x 2,048 tokens of `prompt_logprobs`).
G2 ran straight after G1 in the same campaign, with no GLM prod in between (G1-RESULTS.md has the harness).
Raw files are in [`results/G2-20261002/`](../../results/campaign/G2-20261002/). The final gate run is `m1final-*`; `m1-*` is
the first run, before the fix.

## Gates

| # | Gate | Result | Pass |
| --- | --- | --- | --- |
| 1 | Teacher-forced top-1 vs the kit's `prompt_logprobs` argmax >= 99% | **99.63%** over all 16,376 positions (first run: 92.27%, see the fix below). First copy of each prompt: **95.35%** (287 / 301). 11 of the 14 first-copy misses are positions where the kit's own top-2 logprobs are within 0.125 (7 exactly tied); without those, 99.31%. Kit top-1 in our top-5: 99.98%. Median \|logprob error\| 5e-6 | **whole: PASS**; first copy: FAIL as a raw number (95.4%), PASS once the kit's own near-ties are excluded |
| 2 | 256-token greedy replies == the kit's on >= 6 of 8 | **1 of 8** identical (prompt 7). First divergences on the others at tokens 18 / 55 / 21 / 1 / 26 / 6 / 38 (kit captured at the end of the campaign) | **FAIL** |
| 3 | Row invariance on the real weights: prefill windows (128 rows) == windows of 8 == one row at a time, both ranks, 259 rows | bit for bit | **PASS** |
| 4 | Serial decode >= the kit's no-drafting speed (23 tok/s) | **19.39 tok/s** median (19.11-19.57 over 8 prompts). Eager one-slot path, no CUDA graphs. First run 11.42 | **FAIL** |
| 5 | Boot from the prepared folder <= 90 s, then <= 60 s | **17.9 / 18.1 s** engine construction (21.1 / 21.5 s from `docker run`) with the extension cache warm. 76.6-79.7 s on the first start after a restage, of which about 60 s is rebuilding the CUDA extensions | **PASS** (both) |

**M1 overall: gates 1 (whole), 3 and 5 pass. Gates 2 (greedy replies, 1 of 8) and 4 (19.4 tok/s) fail.**

### Per prompt (final run `m1final`)

| Prompt | top-1 % | first copy (tokens) | first-copy top-1 % | misses | prefill 2,048 (s) | decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 prose | 99.90 | 38 | 94.7 | 2 | 9.3 | 19.57 |
| 1 code | 100.00 | 44 | 100.0 | 0 | 9.5 | 19.49 |
| 2 math | 99.80 | 49 | 93.9 | 4 | 9.9 | 19.44 |
| 3 chat template | 97.85 | 21 | 95.2 | 44 | 7.6 | 19.18 |
| 4 multilingual | 99.80 | 45 | 93.3 | 4 | 10.0 | 19.11 |
| 5 JSON | 99.85 | 38 | 92.1 | 3 | 9.2 | 19.11 |
| 6 long instruction | 99.95 | 44 | 97.7 | 1 | 10.0 | 19.17 |
| 7 repetition | 99.90 | 22 | 95.5 | 2 | 6.4 | 19.39 |

There are 60 misses in all. The kit's margin between its top-1 and top-2 logprob at those positions has quantiles
0 / 0.125 / 0.25 / 0.5 / 0.75 / 1.56 (min, 25%, median, 75%, 90%, max). 25 of the 60 are kit near-ties (<= 0.125);
excluding them, the whole-prompt agreement is 99.79%. 44 of the misses are in prompt 3, the chat-template prompt,
right after `<｜Assistant｜>` / `<｜User｜>`. There the kit itself is unsure: at position 337, for example, it gives
`\n\n` -1.89, "Explain" -2.39 and `<` -2.39, and we pick "Explain" (the copy).

## The bug G2 found: V4.1 has no per-head query norm

The first run scored 92.27% top-1 (85.7% on first copies; 73% on the repetition prompt). The misses showed a
pattern. On the repetition prompt, at every period restart (positions 45, 67, 89, 111, ...), the kit predicted the
glued `one` that continues "... ten one two" + "one two three", with margins of 2-3 nats, while we predicted
` three`. That is a copy / induction failure inside the 128-token sliding window. Steps:

1. **The reference has the same misses.** `engine/reference` (kit numerics, on the GPU, layer by layer; the
   diagnostic is under `results/G2-20261002/`) on the first 128 tokens of prompts 7 and 0 missed exactly the same
   positions (7: 45 / 67 / 89 / 111; 0: 7, 12, 13, 17, 18, 19, ...). So the kernels were not at fault. The
   architecture as both implement it differs from the kit.
2. **Ablations on the reference.** Without Engram the misses got worse (prompt 0: 25 -> 29 of 63). Exact numerics
   left them unchanged. Neither is the cause.
3. **The kit's source** (vLLM `models/deepseek_v4_1`, copied from the kit image) calls the fused q-norm / RoPE /
   KV insert with `apply_q_norm=False`: "qr is normed before wq_b". V4.0's code calls the same op with the
   default `True`. Our forward and the reference normalised each query head (weightless RMS). That pins every
   head's |q| to sqrt(512) and flattens the copy heads. ENGINE-PLAN section 13.1 had listed this as an open
   detail.
4. **Fix** (TF `9fd3764`, reference `b778b1f`, drafter `6c4a69a`): `q = rope(wq_b(qr))`. Top-1 went from 92.27% to
   **99.63%**; the repetition prompt from 73.3% to 99.90%. CPU test: doubling `wq_b`'s output must change the
   logits, which fails with the norm (it cancels). The staging repo's exact-numerics family tests still match the
   reference at 1e-12. The kit-numerics noise bound went 0.90 -> 0.85, because sharper attention amplifies bf16
   noise (26 of 29 rows under 3%).

## Serial decode speed (gate 4): 11.4 -> 19.4 tok/s, short of 23

`gate --profile 20` (torch profiler, 20 serial steps after a 256-token prompt, rank 0):

| State | ms a step | tok/s | Top GPU time a step |
| --- | ---: | ---: | --- |
| First run | 87.9 | 11.4 | `_route` 40.5 ms (1.01 ms a layer: the router ran as **one program**, grid = row blocks, at R = 1), x3ld 8.6, EXL3 linears ~12, all-gathers 2.8 |
| Router split (`ccffe64`): logits as (row block, 32-expert block) programs, then the selection | 57.1 | 17.5 | `_logits` 8.1 ms (202 us a layer) |
| 16-expert blocks, k steps of 64, 3 stages | 53.9 | 18.6 | `_logits` 4.3 ms (106 us a layer) |
| Final gate run (8 prompts x 256 tokens) | ~51.6 | 19.4 | |

- The split router is **bit-identical** to the one-program router. Every one of 72 tile shapes tried gave the same
  bits, because each logit is its own fp32 FMA chain over k ascending. A GPU test checks split == one program at
  R = 1-128 for the target (384 x top-6) and DSpark (128 x top-3) routers.
- The remaining gap to 23 tok/s (43.5 ms a step) is host time, not GPU time. The GPU is busy about 39 ms of 54.
  The Engram inputs take **5.75 ms a step** on the host before any GPU work (n-gram hash, O_DIRECT row reads,
  gather). The rest is `choose`'s D2H synchronisation and about 2,500 kernel launches a step. M2's
  `DecodeForward` has the two fixes (Engram prefetch at drafter end, CUDA graphs per window mix). The M1
  one-slot path the gate uses has neither. G3 measures the serial speed through that path.

## Memory

| | head (rank 0) | worker (rank 1) |
| --- | ---: | ---: |
| Weights on the GPU (prepared folder) | 95.4 GiB allocated (101.8 GB) | 95.4 GiB |
| After boot: MemAvailable | 9.8 GiB | 8.9 GiB |
| Peak allocated / reserved (4K context, 128-row windows) | 98.48 / 98.84 GiB | -- / 98.84 GiB |
| **Workspace** (peak allocated above the post-boot allocation) | **2.98 GiB** | |
| **MemAvailable minimum, 0.5 s samplers, the whole of G2 (03:04-03:51)** | **6.24 GiB** (MemFree 4.95) | **5.23 GiB** (MemFree 4.12) |
| The same during the final gate run alone | 6.82 GiB | 5.31 GiB |

**95 GiB a rank with 4 x 300K does not fit at the 5 GiB floor.** At 4K context, with M1's one slot and contiguous
stores, the worker already sits at 5.2 GiB. A 4 x 300K pool adds 2.35 GiB (bf16 index keys; 2.00 with FP8). That
puts the worker at about 2.9 GiB and the head at about 3.9 GiB, under the 4 GiB hard stop, before M2's drafter
(about 3.4 GiB a rank) and graph pools. The plan's 4.0 GiB workspace estimate is close to the measured 3.0 GiB
plus graphs. The rest of the gap comes from the nodes: after boot only ~9-10 GiB is available, against the plan's
112 GiB "available" point. Options: FP8 index keys (-0.35 GiB), a smaller 2,048-row workspace, 2 x 300K or 4 x 150K
streams, or the head on a 2.0 bpw pack.

## FP8 index keys vs bf16 (the kit's FP8 indexer keys)

`TF_DSV41_INDEX_KEY_SIM=fp8` (`e8c9365`; the gate harness only) rounds every stored index key through e4m3 with a
power-of-two scale a key, to emulate the kit's FP8 key cache. Top-1 vs the kit: **99.646% with FP8 keys vs
99.634% with bf16** (2 more positions of 16,376 agree). First copies are unchanged (95.35%). The selection barely
feels the key precision at 2K. Keeping bf16 costs 0.35 GiB at 4 x 300K and buys nothing measurable here.

## Gate 2 (greedy replies vs the kit)

At the end of the campaign, after G3, Mia's kit was booted as `scripts/dsv41/baseline.sh` does: `start-guarded.sh`,
`MAX_NUM_BATCHED_TOKENS=2048`, `SKIP_SYNC=1`. It was healthy in about 6 min (05:09-05:15). Note that it listened on
its own port 8888 from its `.env`, not :8001; the kit has no env override for the port, and its `.env` was not
edited. 8 x 256 greedy tokens were captured from each prompt's first copy with the new `oracle_prompt_logprobs.py
greedy` (token ids, `ignore_eos`), then the kit was stopped.
Files: `results/campaign-20261002/kit-greedy/kit-greedy.json`, scored in `results/G2-20261002/gate2-greedy-vs-kit.json`
against `m1final`'s replies.

| Prompt | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First divergence (token) | 18 | 55 | 21 | 1 | 26 | 6 | 38 | none (256 / 256) |

**1 of 8 identical: FAIL** (the gate wants 6 of 8). Notes:

- Teacher-forced agreement is 99.6% overall but 95% on first copies, so a 256-token free run is expected to cross
  a disagreeing position early.
- Prompt 3 diverges at its 2nd token (ours 372, kit 5), and both go on with the same tokens (8850, 10531).
- The kit's capture ran with DSpark on (k = 3), and vLLM's batched verify is not bit-exact with its own serial
  greedy. A kit capture with `SPEC_METHOD=none` would be the cleaner reference.
- **One real lead:** on prompt 5, both emit BOS (token 0) mid-reply. The kit continues with text, while we keep
  emitting 0 (`0, 5, 0, 0, 0`). That points at how a mid-sequence BOS is handled (Engram's n-gram boundary / DEAD
  ids, or the hasher's token map for id 0) and should be checked against the kit's `common/engram.py` before
  re-running gate 2.

## Other G2 changes on the branch

- `gate` records the memory, the knobs, each prompt's misses with the kit's margins, `--no-invariance`, `--trace`,
  `--profile` and `--kit-replies` (`e8c9365`, `fde46e7`, `f5dd903`, `e420088`).
- G2's harness is `campaign.sh` plus `G2-m1.sh pair TAG MIN ARGS` (one more gate run inside the held window) and
  `DSV_ENV` (extra environment for the ranks).
