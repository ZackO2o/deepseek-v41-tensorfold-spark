Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# DeepSeek-V4.1-Flash on TensorFold, 2x NVIDIA DGX Spark

Serve DeepSeek-V4.1-Flash (the 2.9 bpw EXL3 pack) across two NVIDIA DGX Sparks, tensor-parallel over the 200 Gb/s
CX7 link, behind an OpenAI-compatible API. The engine is [TensorFold](https://github.com/ashhart/TensorFold) 0.6.0
(pinned, unmodified submodule) plus two patches applied at image build: the two-Spark engine stack from the
[GLM-5.3-Flash recipe](https://github.com/jayleaton/glm53-tensorfold-spark) and a new DeepSeek-V4.1-Flash family
written for this model: CSA2 attention with FP8 KV rows and the lightning indexer, Single-Pass mHC, Engram rows read
from local NVMe, EXL3 expert and dense kernels tuned for GB10, **exact** DSpark speculative decoding, CED
bounded-replay prefill, 4 request slots over a shared FP8 KV pool (4 x 300K tokens), sessions with an NVMe tier,
prepared per-rank folders for ~40 s restarts, structured output and DSML tool calls.

> **Work in progress.** Measured on one pair of Sparks, against one baseline. Knobs, defaults and numbers may change.
> Read [What is not solved](#what-is-not-solved) before relying on it.

SPDX-License-Identifier: Apache-2.0 (this project's own code, scripts, benchmarks and docs; see [Licensing](#licensing)).

## Results

Hardware: two DGX Sparks (GB10, 128 GB unified memory each), one QSFP cable between their CX7 ports, RoCE. Weights:
[`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw),
the uncensored variant of [Mia-AiLab's 2.9 bpw EXL3 pack](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
(same layout). Baseline: [MiaAI-Lab's 2x DGX Spark vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
on the same pair and the same weights (vLLM TP=2, DSpark k=3, its moe_x kernel engaged), measured with our own
clients on 2026-10-01. Our side: the configuration in [`config/prod.env.example`](config/prod.env.example) on the
engine these patches build. Raw files: [`results/`](results/README.md).

### Speed (tok/s)

| | **TensorFold (this recipe)** | MiaAI-Lab vLLM kit | ratio |
| --- | ---: | ---: | ---: |
| Code, 1 stream, greedy | **79.0-81.5** | 41.9-45.0 | **1.8-1.9x** |
| Prose (essay), 1 stream, greedy | **40.9-41.2** | 32.5 | 1.26x |
| Structured (count 1-200), 1 stream, greedy | **116.6** | 38.0 (50.2 on a JSON task) | **2.3-3.1x** |
| 1 stream, decode aggregate | **82-83** | 32.2 | **2.6x** |
| 2 streams, decode aggregate | **67.1** | 46.7 | 1.44x |
| 4 streams, decode aggregate | **93.5** | 37.6 | **2.5x** |
| Cold prefill 8K / 32K / 64K / 128K (CED replay, the default) | **1,833 / 2,043 / 2,068 / 1,953** | 1,073 / 1,075 / 1,060 / 1,031 | **1.7-1.95x** |
| Cold prefill, `TF_DSV41_PREFILL=full` (no replay) | 969 / 1,004 / 1,003 / 876 | same | 0.85-0.95x |
| Start to ready | **34-44 s** | 378 s | ~9x |

At temperature 0.7 the single-stream cells are code 72, prose 42, structured 117.5.

### Quality and robustness

| | **TensorFold** | kit |
| --- | --- | --- |
| Teacher-forced top-1 agreement with the kit (8 prompts x 2,048 positions) | 0.9963 (first copy of each prompt: 0.9502) | 1 by definition |
| MMLU-200, 0-shot, greedy, thinking off | **87.5%** | 87.5% |
| MMLU-200 with a 20-question preamble (~2.1K-token prompts), replay vs full prefill | 81.1% / 81.1% (178 of 180 answers equal) | - |
| Needle at 32K / 128K / 299K (replay prefill) | found (19.9 s / 74.0 s / 195 s) | - |
| Multi-step tool chains (`bench/tooleval/chains.py`, 6 scenarios, 12 points), thinking off / high | 11 / 12, 11 / 12 | - |
| tool-eval-bench category C (multi-step), thinking off | 8 / 8 (score 100) | - |
| Structured output: 12 JSON-schema cases + 10 tool-choice cases (drafted == serial == batched, valid, no markup) | 22 / 22 | - |
| 30-minute soak (1-4 streams, cancels, disconnects, long prompts) | 529 requests, 0 errors, drained | - |
| Stress: one 299K prefill + three 64K prompts decoding 2,048 tokens each | every stream completes; 299K prefill **175 s**; host RSS growth ~0.7 GiB a rank; worker MemAvailable minimum 3.0-3.7 GiB (the boot budget, not growth) | 4.40 GiB at <= 256K (lighter load) |

Top-1, MMLU-200, tool calling, structured output and soak ran on the final configuration (engine `38f6500`); the
stress row is the current engine (`a6f5792`, G12); the 20-question MMLU and the needles ran on the G7 engine commit
with the same prefill path. On the current engine a 299K prefill takes 175 s inside the stress (three other streams
decoding), against 181-185 s before the G12 memory fixes; the needles were not re-run.

### What measures what

- **Decode cells, ours:** `m2bench` (inside the engine, both ranks, no HTTP; `scripts/serve.sh run`), tok/s from the
  first to the last token of a 384-token reply, the median of the repetitions (2 by default), request slots sized
  for 16K tokens. Prompts: an LRU-cache
  class with tests (code), a 400-word essay (prose), counting 1 to 200 (structured). The ranges are the spread over
  the runs of the final configuration (G10 / G11 in the development log).
- **Decode cells, kit:** HTTP clients (`glmbench`, `multiturn` of the GLM recipe): code = a 64-token code reply
  (41.9) and a 512-token one (45.0), prose = a 200-token essay, structured = count 1-200, thinking off.
- **2 / 4 streams:** both sides report decode aggregate = all tokens / (last token - first token). Ours mixes code,
  prose, a JSON task and a copy-heavy edit, every other stream at T = 0.7; the kit's mix is its chat / code prompts,
  256 tokens each. Same metric, different prompts.
- **Prefill:** one request, cold, after a 2K warm-up. Ours: fresh random text through `m2bench --prefill` (measured
  at the G7 engine commit; G8-G11 changed decode paths only, and G12's host-memory fixes were measured only on the
  299K stress prefill: 175 s). The kit: a repeated filler document over HTTP.
  Repeated fillers can make Engram reads look cheaper, so the kit's cells are not pessimistic.
- **Start to ready:** ours = `docker run` to `/v1/models` answering, with prepared folders and compiled kernels
  cached, page cache dropped first. The kit's = its start script to `/health` on freshly rebooted nodes. The first
  start of a new image is slower (kernel builds, ~80 s) and the very first writes the prepared folders (~95 GB a node).
- **Top-1:** the kit's `prompt_logprobs` (top 5) over 8 built-in prompts repeated to 2,048 tokens
  ([`results/kit-baseline/oracle-kit.json`](results/kit-baseline/oracle-kit.json)), our engine teacher-forced on the
  same token ids. "First copy" counts only each prompt's first pass, before the model can copy itself.
- **Not RigMark.** No RigMark receipt exists for this model yet; every cell comes from the clients above.

[`docs/RESULTS.md`](docs/RESULTS.md) has every table, the lever-by-lever history and the negatives;
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) how to rerun each cell.

### What "exact" means here, and what is approximate

- **Exact:** speculative decoding never changes a reply. Every verify row is the serial step at its position (row-
  invariant kernels) and its token is the request's keyed choice at that absolute position, so drafted == serial at
  T = 0 and at T > 0, and batched == alone. Every benchmark run checks it (`exact_all`).
- **Approximate by design, on in the measured config:**
  - CED decoder bounded replay for prompts (`TF_DSV41_PREFILL=replay`, DeepSeek's own technique: the decoder half
    runs only over a prompt's last 128 tokens). `full` is the exact prefill at about the kit's speed.
  - Routed-expert pruning in decode (`TF_DSV41_EXPERT_TOPP=0.85`, at least 3 experts, renormalized): +5% decode,
    MMLU-200 88.5% alone. Delete three lines of the config to serve the unpruned model.
  - mHC mixing weights in bf16 (`TF_DSV41_MHC_FN=bf16`): +2-5% prose, top-1 vs the kit 0.9963.
- **Not bit-identical to the kit.** Different kernels and summation orders; the agreement is the top-1 row above.

### Where we are not at 2x

Prose (1.26x) and 2 streams (1.44x). Prose drafts poorly: DSpark keeps ~1.6 tokens a round on prose against ~3.9 on
code, so prose speed is the verify window's cost. A 1-row window is 26.9 ms against a bandwidth floor of ~17 ms a
rank (the 2.9 bpw weights read once), and the second row costs ~6 ms more because a second token brings ~5 new
experts a layer. 2 streams pair a code stream with a T = 0.7 prose stream, so the prose stream sets the pace.
[`docs/DECODE.md`](docs/DECODE.md) has the roofline and what was tried.

## Quick start

Requirements:

- two DGX Sparks with their CX7 ports cabled and addressed (one link subnet), Docker with the NVIDIA runtime on both,
  and passwordless ssh from the head to the worker over the link;
- on each node's local NVMe: ~100 GB for the weights, ~95 GB for the Engram shards, ~95 GB for the prepared
  folders, and room for the session tier (`TF_DSV41_SESSION_DISK_GIB`, 128 GB by default);
- nothing else on the GPUs: the stack plans for a 4-5 GiB MemAvailable floor out of 128 GB a node.

**1. Clone and configure** (on the head):

```bash
git clone --recurse-submodules https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
cp config/prod.env.example config/prod.env
$EDITOR config/prod.env      # every <placeholder>: WORKER_SSH, HEAD_IP, the paths on each node; check the NIC names
```

**2. Weights** (on both nodes, byte-identical):

```bash
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw --local-dir <HEAD_MODEL>    # and <WORKER_MODEL>
```

**3. Engram shards.** The EXL3 packs do not carry the Engram tables (layers 1 and 14, ~101 GB each). They come from
DeepSeek's original checkpoint; each rank keeps its half of the hash heads (~47 GiB a layer) on local NVMe:

```bash
mkdir -p <src> && hf download deepseek-ai/DeepSeek-V4.1-Flash model.safetensors.index.json --local-dir <src>
python3 scripts/pack_engram.py --src <src> --list                 # the shard files that hold the tables
hf download deepseek-ai/DeepSeek-V4.1-Flash <those files> --local-dir <src>
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --out <HEAD_ENGRAM> --rank 0
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --out <dir> --rank 1   # copy to <WORKER_ENGRAM> on the worker
python3 scripts/pack_engram.py --src <src> --config <HEAD_MODEL>/config.json --check <HEAD_ENGRAM> --rank 0
```

`pack_engram.py` writes the format the engine reads (`engram-l{1,14}-r{rank}of2.bin`) from the source tensors'
raw bytes; it is tested against the engine's reader on synthetic tables. The measured runs used shards packed by
the MiaAI-Lab kit (`./start.sh pack`), which writes the same format; `--check` compares either with the source.

**4. Build, check, start:**

```bash
scripts/serve.sh build        # docker/Dockerfile: TensorFold v0.6.0 + patches/, shipped to the worker
scripts/serve.sh preflight    # image on both nodes, weights, Engram shards, RoCE ports, free ports, idle GPUs
scripts/serve.sh start        # memory gate, rank 1 then rank 0, /v1/models, slot check, canary
```

The first start compiles the CUDA / Triton kernels into the `CACHE_VOL` volume and writes the prepared rank folders
(`TF_DSV41_PREPARED_WRITE=1`, ~95 GB a node, several minutes); later starts read them back in ~40 s. Then:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}], "reasoning_effort": "low"}'
scripts/serve.sh status | logs [0|1] | canary | stop | restart
```

**5. Run it unattended** (optional): a watchdog tick every minute (heals after 3 bad ticks, at most every 30 min)
and a start at boot.

```bash
mkdir -p ~/.config/systemd/user && cp scripts/systemd/dsv41-* ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/dsv41-*.service        # WorkingDirectory= this checkout
loginctl enable-linger "$USER"
systemctl --user daemon-reload && systemctl --user enable --now dsv41-tf-watchdog.timer && systemctl --user enable dsv41-boot-start.service
```

`scripts/serve.sh` drops the page cache on both nodes around a start (`DROP_CACHES=1`: needs `sudo -n` or root for
`/proc/sys/vm/drop_caches`; otherwise it logs and continues). [`docs/OPERATIONS.md`](docs/OPERATIONS.md) covers the
knobs, the memory gates, the watchdog, and how to turn each lever off.

## API

OpenAI-compatible on `HOST:PORT` (`127.0.0.1:8000` by default: put your own proxy and authentication in front):
`/v1/chat/completions` (streaming, tool calls, `response_format`), `/v1/completions` (text or token ids),
`/tokenize`, `/v1/models` (`max_model_len`), `/health`, `/metrics`.

Thinking follows DeepSeek-V4.1's encoding (`Reasoning Effort: N (range 1-100)`), on by default
(`TF_DSV41_THINKING=0` turns it off). Both the top-level `reasoning_effort` and
`chat_template_kwargs.reasoning_effort` are read (the kwargs win):

| value | thinking | effort |
| --- | --- | ---: |
| `none`, `minimal` | off | - |
| `low` | on | 50 |
| `medium`, `high` (default: `TF_DSV41_DEFAULT_EFFORT`) | on | 75 |
| `xhigh`, `max` | on | 100 |
| an integer 1-100 | on | that |

`chat_template_kwargs.enable_thinking` (or `thinking`) true / false sets the mode directly. Note: the kit's vLLM
path renders `low` as 25; we keep DeepSeek's 50.

## The engine

`vendor/TensorFold` is upstream TensorFold v0.6.0, unmodified. `patches/` holds two patches, applied in order by the
Dockerfile:

| patch | what | licence |
| --- | --- | --- |
| [`0001-spark-stack-060.patch`](patches/0001-spark-stack-060.patch) | the GLM-5.3-Flash two-Spark engine (`families/glm5_next/spark/`) rebased onto 0.6.0, the CUDA communicator interface (`cuda/comm.py`), the family `CUDA_SERVE` hook (`cli.py`, `families/glm5_next/__init__.py`), the server's descriptor fix (`server/cancellation.py`), packaging (`pyproject.toml`), recipes and tests |
| [`0002-deepseek-v41-family.patch`](patches/0002-deepseek-v41-family.patch) | `families/deepseek_v41/` and its tests, the EXL3 linear's device-side skip (`cuda/exl3/linear.*`), fp64 in `cuda/comm.py`, `--kv-dtype fp8` (`cli_args.py`), model aliases in the GLM server, packaging, NOTICE entries |

Together they are every engine change production runs (development commit `a6f5792`, 356 files over v0.6.0): applying
them to v0.6.0 reproduces that tree except for reworded comments, a later test-only fix and the excluded
draft-vocabulary files
([`docs/ENGINE.md`](docs/ENGINE.md) lists each difference).

[`docs/ENGINE.md`](docs/ENGINE.md) explains how the patches were produced, how to get the same tree as a git branch,
and how to run the engine's test suites. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) summarises the model and the
TP=2 split.

## What is not solved

- **The worker's memory floor.** In the 4 x 300K stress the worker's MemAvailable bottoms out at 3.0-3.7 GiB early in
  the 299K prefill, under our own 5 GiB target and the 4 GiB admission floor (nothing new is admitted below it). It
  comes from the boot budget (the 4 x 300K KV pool) and the first segments' CUDA reservations, not from host growth
  (fixed in G12). Open. `CONTEXT=196608` lowers the exposure.
- **Prose and 2 streams** are not at 2x (above).
- **The speed table predates G12.** It was measured on `38f6500`; the current engine has the same decode kernels, so
  no change is expected, but it has not been re-measured.
- **`/health`** reports `drafted_total` / `accepted_total` as 0 for this family (the counters are not wired).
- **Vision** is not wired for this family.
- **No trimmed draft-head vocabulary is shipped** (`TF_DSV41_DRAFT_HEAD=trim`, off by default and not adopted). The
  development ranking was counted from private chat transcripts and is excluded, so `trim` needs
  `TF_DSV41_DRAFT_VOCAB=<file>` and `tests/test_dsv41_draft_head.py::test_shipped_ranking` fails. A ranking of your
  own traffic (`scripts/campaign/draftvocab.py`) or of public text (the GLM recipe's `bench/draftvocab_public.py`
  method on this tokenizer) can be used.

## Layout

| path | what |
| --- | --- |
| `vendor/TensorFold` | upstream TensorFold v0.6.0 (submodule) |
| `patches/` | the engine changes ([`docs/ENGINE.md`](docs/ENGINE.md)) |
| `docker/Dockerfile` | the image: NVIDIA PyTorch 26.07 + xgrammar + TensorFold with the patches |
| `config/prod.env.example` | the measured configuration, with placeholders for your hosts and paths |
| `scripts/serve.sh` | build / preflight / start / stop / status / watchdog / `run` (engine benchmarks on both ranks) |
| `scripts/pack_engram.py` | the per-rank Engram shards from DeepSeek's checkpoint |
| `scripts/canary.py`, `scripts/boot-start.sh`, `scripts/systemd/` | post-start canary, start at boot, watchdog units |
| `scripts/check-public.sh` | the sanitizer this repository was checked with |
| `bench/` | HTTP clients: quality (MMLU, needles), structured output, soak, stress, tool calling |
| `docs/` | results, benchmark method, architecture, decode roofline and lessons, operations, the engine |
| `results/` | the raw files behind the tables ([`results/README.md`](results/README.md)); `results/campaign/` every tracked result of the development windows G1-G12 |
| `docs/campaign/` | the development log: plans, targets, the landscape study, the results of every window ([`docs/campaign/README.md`](docs/campaign/README.md)) |
| `engine/`, `tests/` | the development staging tree: the PyTorch reference model (the correctness oracle), the kernels and serving layer before they were ported into the TensorFold family, and their tests |
| `scripts/campaign/` | analysis tools of the windows (nsys window / idle / skew breakdowns, summaries), the draft-vocabulary study tool, the porting script |

## Licensing

| Part | License |
| --- | --- |
| This project's code, patches, scripts, benchmarks and docs | **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). Redistributions, modified or not, must keep the copyright line and the NOTICE attributions and state their changes. |
| TensorFold (`vendor/TensorFold`) | Apache License 2.0 from 0.6.0 (code written before 0.6.0 keeps its MIT notice), Copyright 2026 TensorFold contributors; unmodified submodule, the patches are applied at build time. The TensorFold code the patches modify stays under its license. Its third-party notices: `vendor/TensorFold/THIRD_PARTY_NOTICES.md` (the patches extend it). |
| Files the patches add | keep the SPDX notice written in them: the DeepSeek-V4.1-Flash family (`families/deepseek_v41/`, its tests) is MIT, Copyright (c) 2026 Jay Leaton; the GLM Spark engine (`families/glm5_next/spark/`) is MIT ([`NOTICE`](NOTICE)). |
| RoCE all-gather and fast-prefill kernels in `patches/0001` | adapted from / re-implementing [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0, Luke Alonso and the b12x contributors); details in [`NOTICE`](NOTICE). |
| Fat-expert MoE kernel structure in `patches/0001` | adapted from the Apache-2.0 [Reederey87 kit](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (code MiaAI-Lab contributed under MIT before 2026-09-07); its NOTICE is reproduced in [`NOTICE`](NOTICE). |
| Ported upstream code in `patches/0001` | from later TensorFold releases (0.3.6.2, 0.5.0), MIT, Copyright (c) 2026 TensorFold contributors; each ported piece names its source commit. |
| xgrammar (structured output) | Apache-2.0 ([mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)), installed into the image by pip, not vendored. |
| Docker base image | NVIDIA Deep Learning Container License (`nvcr.io/nvidia/pytorch:26.07-py3`) |
| Model weights (not included) | `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (measured here), its base `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`, and `deepseek-ai/DeepSeek-V4.1-Flash` (for the Engram tables): each under its model card's terms. The uncensored weights have refusals removed; you are responsible for how you use them. |

Nothing from the MiaAI-Lab DeepSeek kit's AGPL-3.0 code is included: the engine reads the pack and the Engram shard
format (file-format facts), and implements DeepSeek's prompt encoding from DeepSeek's own MIT `encoding.py`.

## Credits

- [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): DeepSeek-V4.1-Flash and its prompt encoding;
  the ideas this engine implements come from DeepSeek's papers and code: the
  [V4.1-Flash technical report](https://arxiv.org/abs/2609.19969) (CED and bounded replay, CSA2, mHC),
  [DSpark](https://arxiv.org/abs/2607.05147) with [DeepSpec](https://github.com/deepseek-ai/DeepSpec), and
  [Engram](https://arxiv.org/abs/2601.07372) with [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram).
- [Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold): the engine, the EXL3 kernels, the drafting
  and the server this recipe builds on.
- Mia / MiaAI-Lab: the [DeepSeek-V4.1-Flash 2x DGX Spark vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
  this recipe is measured against (whose packed Engram shards the measured runs used), and the
  [2.9 bpw EXL3 pack](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) on Hugging Face
  ([Mia-AiLab](https://huggingface.co/Mia-AiLab)); their GLM-5.3 kit's fat-expert MoE design also lives on in
  `patches/0001`.
- [dealignai](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw): the uncensored 2.9 bpw
  variant measured here.
- [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3): the EXL3 format.
- [Cruz (vcruz305)](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe): the finding that DSpark
  verify and plain decode take different EXL3 paths in the vLLM kits (the case for exact speculative decoding), the
  expert-union counts per verify row behind our round model, and the host-side Engram hashing fix
  ([`docs/campaign/LANDSCAPE.md`](docs/campaign/LANDSCAPE.md), [`docs/campaign/TARGETS.md`](docs/campaign/TARGETS.md));
  his [SAGE 1.59 bpw pack](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw) was evaluated
  ([`docs/campaign/PARKED-SAGE-1.59.md`](docs/campaign/PARKED-SAGE-1.59.md)).
- PCTree, [arXiv 2608.02123](https://arxiv.org/abs/2608.02123): the parent-conditioned draft trees implemented as
  `TF_DSV41_TREE_PC` (measured, not adopted: [`docs/DECODE.md`](docs/DECODE.md)).
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (Luke Alonso and contributors): the
  "RoCEnante" one-shot RoCE all-gather that `patches/0001`'s RoCE path (used by this family) adapts, and the prefill
  kernel designs it re-implements.
- [Reederey87](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark): the Apache-2.0 kernel code the GLM
  stack's fat-expert kernels adapt.
- [The vLLM project](https://github.com/vllm-project/vllm): the DeepSeek-V4 / V4.1 implementation our kernels' math
  follows (cited per file; no code copied).
- [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar): the grammar engine behind structured output.
- [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) and
  [Weschera/spark-bench](https://github.com/Weschera/spark-bench): the tool-calling benchmarks.
- [MMLU](https://github.com/hendrycks/test) (Hendrycks et al.): the 200 questions in `bench/data/`.
- NVIDIA: the DGX Spark and the PyTorch container.

## Current focus

**Fixed in G12** (engine `a6f5792`, [`docs/campaign/G12-RESULTS.md`](docs/campaign/G12-RESULTS.md)):

- **Host memory growth in long prompts.** torch's CPU allocator in NVIDIA's PyTorch build is an embedded mimalloc,
  and it kept the memory of large blocks freed on another thread: a prompt segment's Engram rows, made on a prefetch
  thread and freed on the round thread. The rows now live in reused NumPy-owned buffers. Growth per rank during a
  299K prefill went from 4.1-5.6 GiB to ~0.7 GiB, and the 299K prefill takes 175 s (was 181-185 s).
- **The 19-minute stall.** It was transparent-huge-page faults compacting memory synchronously under memory pressure,
  not a hang. NumPy's huge-page request is off in the engine, and `MIMALLOC_ALLOW_THP=0` is in the config
  (`scripts/serve.sh` passes `MIMALLOC_*` to both ranks). A stall watchdog and deadlines were added as a safety net.
- **Fast-prefill segmentation dependence.** The RoPE tables are built in fixed blocks, so the segment split no longer
  changes bits.

**Still open:** the worker's boot-time memory floor of ~3-3.7 GiB in the stress, and the speed table, which has not
been re-measured on `a6f5792` (the decode path is unchanged, so no change is expected).

**Next:** the optimisation round, with engine and kernel rewrites.
