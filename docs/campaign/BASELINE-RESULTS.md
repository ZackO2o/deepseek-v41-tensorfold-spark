# Baseline: the MiaAI kit on our 2 Sparks (window 2026-10-01, 20:56-21:26 +07)

The first measured baseline of DeepSeek-V4.1-Flash on our pair. It replaces the "kit anchor" column of
[`TARGETS.md`](TARGETS.md) wherever a cell below covers it. Raw files are in
[`results/BASELINE-20261001/`](../../results/campaign/BASELINE-20261001/). The harness is `scripts/dsv41/baseline.sh` in the glm53
repo (DSV41-BASELINE.md section 3), with the changes listed under "Window" below.

**RigMark was not run** (user change on the day). Every number here comes from our own clients: `bench/glmbench.py`,
`bench/multiturn.py`, `bench/quality.py` and `scripts/dsv41/longprefill.py`. The single-stream cells are therefore not
RigMark code / prose / structured. They are the closest of our own cells, and each row says which one.

## Setup (as the user last ran it, plus the research settings)

| | |
| --- | --- |
| Checkpoint | `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, 39/39 shards, index md5 `db5906a6...`, byte-identical shards on both nodes |
| Kit | MiaAI 2x kit @ `8404ac7` + local changes; image `...-2x-dgx-sparks:2.9bpw` (head `6c944f26`, worker `704645c5`: the same images the 09-20 run used); `./start-guarded.sh` (detect-gids -> oomwrap 105 GiB -> `start.sh`) |
| Engine | vLLM `0.1.dev20904` TP=2, DSpark k=3, **moe_x `2,12,8` engaged** (3 lines a rank), `MAX_MODEL_LEN` 600000, 4 sequences, vision on, prefix caching on, fp8_ds_mla, 64-token blocks |
| KV pool | pinned 2.5 GiB = **773,163 tokens** (09-20 at 1,536: 785,676) |
| Research settings | `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` (already in the kit's `.env`: no deadlock seen at 4 streams); **`MAX_NUM_BATCHED_TOKENS=2048`** through `start.sh`'s env override (issue #28 multi-image floor; the user's `.env` has 1,536 and was not edited; vLLM logged "max_num_scheduled_tokens is set to 2048"); `LONG_PREFILL_TOKEN_THRESHOLD` 1280 unchanged |
| Only `.env` change | detect-gids rewrote `WORKER_GID` 4 -> 3 after the reboot (its job) |
| Nodes | **both rebooted inside the window** after GLM was stopped (worker first, then head; new boot ids), because W24 found ~3% drift from node state that a reboot clears. The lease was kept fresh and the watchdog stayed stopped; boot-start stood down on the fresh lease. GPU clocks 2,223 MHz on both after the reboot (gpucheck's warnings were the pre-reboot W24 power clamp and the reboot gap, both cleared) |

## Results against the targets

Means of 3 reps (glmbench prints medians of 3). "Anchor" = the column TARGETS.md section 2 used: (a) helge's forum
measurement, (b) our own 09-20 session, (c) the kit README with moe_x.

| Metric | **Measured (this window)** | Cell used | Anchor in TARGETS | Published (LANDSCAPE) | **Target** | Target / measured |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| Code c1 | **41.9** (T=0) / 40.6 (T=1); 45.0 at 512 tokens; hashmap 30.3 | glmbench tf code (64 tok, raw completion); tweet code; kit hashmap (200 tok, thinking off) | 36.4 (a); 56-57 (c) | 37.4 coolbho3k; 36.4 helge | 72 | 1.6-1.7x (2.4x on hashmap) |
| Prose c1 | **32.5** hard essay; 19.2 short chat (T=0), 17.7 (T=1) | kit essay (200 tok); tf chat (64 tok) | 25.1 (a); 32 hard / 63-66 easy (c) | 42.6 sfxnz 2.0 bpw | 45 | 1.4x (essay) |
| Structured c1 | **38.0** count 1-200; 50.2 JSON; 47.4 primes list | kit structured; tweet json / sequence (512 tok) | 40 (c, pre-moe_x) | 84.2 sfxnz | 90 | 1.8-2.4x |
| Copy-heavy edit c1 (1,024 tok) | **55.1-56.2** | glmbench edit (3 cells, 1 rep each) | | 47.6 coolbho3k with prompt lookup | (no cell) | |
| Concurrent C1 / C2 / C4 aggregate | **32.2 / 46.7 / 37.6** (C4 per stream 9.2-10.2) | multiturn concurrent, tf/kit prompts, 3 reps | x2 ~78, **x4 53-63** (b); x2 97 (c) | C6 62.1 coolbho3k | C4 120 | **3.2x** |
| Cold prefill 8K / 32K / 64K | **1,073 / 1,075 / 1,060** tok/s (TTFT 7.6 / 30.5 / 61.8 s) | longprefill.py, exact token-id prompt, nonce + RigMark filler | ~1,050 (b, c) | 971 / 1,055 / 1,023 (kit the kit dashboard); 1,054-1,136 @32K coolbho3k | 1,500 full decoder; 2,200 CED | 1.4x / 2.1x |
| Cold prefill 128K / 256K | **1,031 / 983** tok/s (TTFT 127.2 / 266.7 s) | same | 961 / 873 (c) | | | |
| 300K cold prompt end to end | ~**5.1 min** extrapolated (983 tok/s at 256K) | | ~850 tok/s at that length | | ≤ 2.5 min | ≥ 2x |
| Decode at depth | 58-62 tok/s after 8K-256K prompts (filler content: drafts ~100% accepted, not a real-text cell) | longprefill decode (128 tok) | 19-24 at 100-601K (c) | 32.5 at 1M coolbho3k | ≥ 85% of short-context | not measurable from this cell |
| **Memory floor** (MemAvailable, worst node, all bench phases) | **head 4.94 / worker 4.40 GiB**; during the 256K prefill 5.01 / 4.46 | 0.5 s samplers both nodes | 2.1 GiB at 601K (kit README) | 0.887 coolbho3k | 4-6 GiB | already 4.4 at ≤ 256K |
| **Boot to healthy** | **378 s** (6 min 18 s: `start-guarded.sh` 21:00:13 -> /health 21:06:31) | baseline.sh kit-up | ~6 min | helge 6 min 30 s with rsync | ≤ 2 min | 3.2x |
| DSpark tokens a round (mixed) | **2.27** over the window (log: 9,849 / 23,178 drafts = 42.5%); per workload below | vLLM SpecDecoding | **3.11** (b, 3 h of real use, thinking on) | ~2 at 45% (dealignai card) | ≥ 3.1 at k=3 | |
| MMLU-200 (thinking off, greedy) | **87.5%** (175/200); refusals 0/10 | bench/quality.py | | dealignai full MMLU 79.20% | within 1 point of the kit | GLM prod on the same 200: 88.0% |
| reasoning_effort scalar | **25** for `chat_template_kwargs.reasoning_effort: "low"` | `/tokenize` + `/detokenize` render | | | | see below |

### DSpark acceptance per workload (k=3; tokens a round = 1 + accepted / drafts)

From `/metrics` counter deltas between phase marks (`summary.txt`); the glmbench suites are split by the kit log's 10 s
SpecDecoding windows (the suites run in order tf, tweet, kit, edit; boundaries are ±10 s).

| Workload | Tokens a round | Accepted / drafted tokens | Per position (1 / 2 / 3) |
| --- | ---: | ---: | --- |
| MMLU-200 (one-letter answers, thinking off) | 1.46 | 15.2% | 0.34 / 0.07 / 0.04 |
| multiturn concurrent (1 / 2 / 4 streams, tf + kit prompts) | 1.83 | 27.8% | 0.37 / 0.27 / 0.20 |
| glmbench tf (64-token code + chat, T=1 and T=0) | ~1.78 | ~26% | |
| glmbench kit (hashmap, count 1-200, essay; 200 tok) | ~2.17 | ~39% | |
| glmbench tweet (primes, code, JSON; 512 tok) | ~3.08 | ~69% | |
| glmbench edit (copy-heavy rewrites, 1,024 tok) | ~3.84 | ~95% | |
| glmbench, all suites | 2.79 | 59.8% | 0.68 / 0.60 / 0.52 |
| long-prefill decode (RigMark filler) | 3.94-4.00 | 98-100% | (filler is repetitive) |

## Findings

1. **The kit sends effort 25, not 50, for our RigMark body.** With `chat_template_kwargs: {"reasoning_effort": "low"}`
   the kit renders `Reasoning Effort: 25`. The kit's own `files/chat_template.jinja` would say 50, so the served
   prompt does not go through that template's mapping. vLLM's renderer maps low = 25. With no kwargs it renders 50
   (vLLM's default), and a top-level `reasoning_effort: "low"` is ignored (50). This confirms ARCH-LEVERAGE section 7:
   a kit cell at `low` is a b = 25 workload. Our engine must send the same scalar (or the receipt must name it), or
   RigMark cells are not like for like. The glmbench / multiturn / MMLU cells above run with thinking off, so they are
   not affected.
2. **C4 is below C2 on the kit** (37.6 vs 46.7 aggregate; 9.6 a stream). There was no deadlock (the shared-experts
   stream is off), but 4 x 4 verify rows run slower than 2 x 4. Acceptance in that phase was 1.83 tokens a round. This
   is the cheapest large target: 120 is 3.2x the measured kit.
3. **Prefill is ~1.03-1.08k tok/s up to 128K and 983 at 256K**, at 2,048-token chunks. That is the kit README's
   level (+2-13% at the same lengths). The 1,500 / 2,200 targets stand. Caveat: the prompt is RigMark's repeated
   filler (LANDSCAPE: repeated fillers can inflate prefill by hitting hot Engram rows), so a novel-text prompt may be
   slower. Our build is measured with the same file, so the ratio is fair.
4. **Memory floor 4.40 GiB** (worker) at ≤ 256K with MNBT 2048; head 4.94. The kit's 2.1 GiB was at 601K, which this
   window did not run (`LONG_LIST` stopped at 256K). The target (4-6 GiB in the 4 x 300K worst case) has to hold more
   live state than this measurement did.
5. **Single-stream decode is lower than the README's moe_x cells.** Code is 42-45 vs 56-57, and essay 32.5 vs "hard
   prose 32", which matches. Count 1-200 is 38.0, about the pre-moe_x 40. Our own 3.11 tokens a round (thinking
   on, real use) is above everything here except tweet / edit. For decode targets, the "mixed" input should come from
   the per-workload rows above, not the 3.11.
6. **Boot 6 min 18 s** from clean (just rebooted) nodes and from local EXT4 (no rsync: the worker marker was current).
   The ≤ 2 min target is a 3.2x cut.
7. **MMLU-200 87.5%**, half a point under GLM prod's 88.0% on the same 200 questions.

## Oracle capture for M1

`results/BASELINE-20261001/oracle-kit.json` (3.7 MB): 8 built-in prompts x 2,048 tokens (BOS + text, repeated to
length), `prompt_logprobs` k = 5 for 2,047 positions each. Captured with the kit at MNBT 2048 / moe_x on, at 21:06:33.
Compare offline:

```bash
python tests/reference/oracle_prompt_logprobs.py compare --capture results/BASELINE-20261001/oracle-kit.json \
    --model /models/dsv41-uncensored-2.9bpw --engram /models/dsv41-engram-src --device cuda --report ...
```

The prompts repeat their seed text to fill 2,048 tokens. So the kit's top-1 equals the actual next token at 90-99.7%
of positions, and most positions are easy. M1 should also report top-1 agreement over the first copy of each prompt,
where the model is not just copying.

## Window

| Time (+07) | Step |
| --- | --- |
| 20:56:57 | lease, watchdog timer stopped, `serve.sh stop` (GLM prod down) |
| 20:57-20:58 | worker rebooted (back in 80 s) |
| 20:58-20:59 | head rebooted (back in 80 s); boot-start stood down on the fresh lease; watchdog timer stopped again |
| 21:00:00 | `baseline.sh run` (FORCE=1 for our own lease, SKIP_RIGMARK=1, KIT_MNBT=2048, LONG_LIST 8K-256K, glmbench tf,tweet,kit,edit, multiturn concurrent); deadman armed at +225 min |
| 21:00:13 | memory ok: MemFree 117.0 / 115.7, MemAvailable 116.6 / 115.1 GiB |
| 21:06:31 | kit healthy (378 s); smoke 17*19 = 323 |
| 21:06-21:24 | oracle, MMLU-200, long 8K / 32K / 64K / 128K / 256K, multiturn, glmbench |
| 21:24:46 | kit stopped, containers gone on both nodes |
| 21:25:48 | **GLM prod up: b13-060 (config/prod.env, md5 `7b5759d8`), 4 request slots**; `:8000` and https list `GLM-5.3-Flash-EXL3`; 17*23 = 391; canary rc 0; page cache dropped after the canary |
| 21:26:02 | watchdog timer active, lease deleted, deadman and refresher killed; no `dsv41mem` sampler left on either node |

GLM prod was down for 29 min.

Harness changes made for this window (glm53 repo, `scripts/dsv41/`):

- `KIT_MNBT` env override;
- the effort render probe;
- an oracle capture phase;
- `/metrics` snapshots at every phase, with per-phase DSpark acceptance in `summary.txt`;
- `LONG_LIST`, `GLMBENCH_SUITES` and `MULTITURN_MODES`;
- phases ordered with the must-have cells first;
- `check` compares shard bytes only (the worker's sync marker added 2,144 B);
- `restore` kills the sampler loops on both nodes;
- `metadata.py` records the MNBT override.

## Not measured, and why

- **RigMark x3** (user change): the first V4.1 RigMark receipt is still open. The effort finding above decides its
  body.
- **The 601K floor** (`LONG2_TOKENS=580000`, ~13 min). The 256K floor was enough for the targets' 300K-a-stream plan.
- **Sessions / follow-up replay** (multiturn modes left out by the user's list). TARGETS' replay anchor (0.46 s) stays
  the user's 09-20 number.
- **Multi-image** (the reason for MNBT 2048): not exercised.
