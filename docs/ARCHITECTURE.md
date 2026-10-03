# DeepSeek-V4.1-Flash on two Sparks: the model, the split, the bytes

A summary of what the `deepseek_v41` family computes and how it divides the work between two GB10s. Sources: the
checkpoint's `config.json` and shard headers, DeepSeek's technical report, and vLLM's DeepSeek-V4 / V4.1 model code
(Apache-2.0), which the family's math follows file by file (each module names the vLLM file it follows).

## 1. Shape

| | |
| --- | --- |
| Vocabulary | 129,280 |
| Hidden | 5,120, carried as **4 hyper-connection streams** [T, 4, 5,120] bf16 |
| Blocks | 40 = CED encoder 0-19 + decoder 20-39, then 3 DSpark draft blocks |
| Attention | 64 query heads x 512, **one K = V head of 512** (448 NoPE + 64 RoPE), attention sinks, a 128-token sliding window on every layer, CSA2 compressed rows on 36 layers |
| MoE | 384 routed experts, top-6 (sqrt-softplus scores, bias selection, renormalized x 1.5) + 1 shared expert; expert 5,120 -> 2,304 -> 5,120 |
| Engram | layers 1 and 14: n-gram hash tables, ~384M rows of 264 bytes each |
| Head | untied, 5,120 -> 129,280 |

The 2.9 bpw EXL3 pack: routed experts mostly 3 bits (2 bits on layers 18-22), attention 5 bits, head 6 bits.

## 2. The parts that are new compared with a classic MoE

- **Single-Pass mHC.** Every sublayer mixes the 4 streams with learned, input-dependent weights (a 24-wide mix, a
  20-step Sinkhorn on a 4 x 4 matrix). V4.1's "delayed" detail: a sublayer's input is collapsed with the *previous*
  sublayer's pre-mix. The family fuses the RMSNorm and the residual add of the gathered partials into the mHC site.
- **CSA2 attention.** One latent K = V row a token per layer, sliding-window rows on every layer, compressed rows
  (ratio 1-2) produced by a compressor on a few layers and reused by the next ones, and a lightning indexer that picks
  the compressed rows each query attends to (top-k over FP8 index keys). KV rows are 584 bytes (448 FP8 + 64 bf16 RoPE
  + scales), ~1.8 KB a token over all caches: 1M pooled tokens = 1.75 GiB a rank.
- **CED: causal encoder-decoder.** The decoder's global KV is a per-token projection of the encoder's output (layer
  19), so a prompt needs the decoder only for its sliding-window rings. DeepSeek's *decoder bounded replay* runs the
  decoder over a prompt's last 128 tokens only (the model was post-trained for it): ~2x prefill. Every prefill
  replays, whatever its size, so the decode state is a function of the token sequence alone (resumed == fresh,
  batched == alone). `TF_DSV41_PREFILL=full` runs all 40 layers over the prompt instead.
- **Engram.** At each position, the last 2-4 token ids (after a tokenizer compression to 99,092 ids) hash into 24
  heads of a ~384M-row table; the rows (256 FP8 values + 8 scales) are gathered and gated into the streams. Addresses
  are known as soon as the tokens are, so rows are read from NVMe ahead of time: a drafted token's rows while the
  draft pass runs, a prompt chunk's rows under the previous chunk's forward. The tables never enter RAM.
- **DSpark.** Three draft blocks fed by the target's last layers draft a block of 5 positions in one pass, a rank-256
  Markov head chains them and a confidence head estimates acceptance. The family verifies drafts **exactly** (below)
  and chooses the draft depth from the confidence head and the measured cost of each verify-window size.

## 3. TP = 2

| part | rank 0 / rank 1 | exchange |
| --- | --- | --- |
| embedding, head | vocabulary halves | gather of each rank's top candidates |
| mHC, norms, router, `wq_a`, `wkv`, compressor, indexer | replicated | - |
| sliding-window rows, compressed rows, index keys | replicated (one KV head cannot be split by head) | - |
| `wq_b`, sparse attention core | heads 0-31 / 32-63 | - |
| `wo_a` / `wo_b` | groups 0-3 / 4-7, K halves | **reduce** [rows, 5,120] |
| routed + shared experts | intermediate halves | **reduce** [rows, 5,120] |
| Engram tables | hash heads 0-11 / 12-23, from each node's own NVMe shard | gather [rows, 24, 256] bf16 |
| DSpark | as a backbone block | 2 a block |

Two reductions a block, ~83 exchanges a forward, 10 KB a row: a verify window moves well under the 1 MiB RoCE
one-shot limit. Exchanges are bf16 (the partials are already bf16-rounded). The residual streams never cross the link.

Resident weights: ~98.9 GiB a rank (routed experts 91.3, DSpark 3.5, the rest under 1 GiB each).

## 4. Exact speculative decoding

- Every kernel on the verify path is **row-invariant**: a row's logits do not depend on the other rows of the window,
  the window's size, or the other requests in it.
- A verify row's token is the request's **keyed** choice at that absolute position (a sampler whose randomness is a
  function of the request seed and the position), so at T > 0 too the serial reply is defined independently of
  drafting.
- A draft is kept up to the first position where it differs from that choice; a round emits accepted + 1 tokens.

So any drafter, depth or acceptance gives the serial reply, and batched == alone. vLLM and the kit use rejection
sampling, so their T > 0 replies depend on the drafts and the batch, and their verify and plain-decode paths take
different EXL3 kernels.

## 5. Serving

- 4 request slots over one shared FP8 KV pool (4 x 300K tokens), prompts admitted in 2,048-row segments shared by
  the prefilling slots, long prompts one at a time.
- Slot-agnostic CUDA graphs keyed by the padded row count (per-row metadata in device tables), for windows and DSpark
  passes, so a window mixing slots replays a graph instead of launching thousands of kernels.
- Sessions: snapshots of a request's state (pool pages + ~3.3 MB of rings and carries) in RAM and on an NVMe tier,
  so a follow-up turn resumes instead of re-prefilling.
- Prepared folders: each rank's weight tree written once (~95 GB a node), keyed by everything that decides its bytes,
  and read back with parallel O_DIRECT readers at start.
- OpenAI API (rank 0): DeepSeek's prompt encoding (from DeepSeek's MIT `encoding.py`), DSML tool calls (streamed),
  structured output through xgrammar (`response_format`, strict / required / named tools) checked token by token and
  compatible with drafting.
