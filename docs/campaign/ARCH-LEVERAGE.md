# What DeepSeek built into V4.1-Flash, and how we can use it on 2 Sparks (2026-10-01)

DeepSeek's own material on how DeepSeek-V4.1-Flash was designed and how it is meant to be served, read for one
question: which architectural features can our 2x DGX Spark TensorFold build use to go faster without breaking the
exactness contract? That contract is drafted == serial, resumed == fresh, batched == alone; approximations are allowed
only as explicit opt-in knobs.

Desk research only. No GPU was used. The Sparks were read over ssh (head: the tech report, `config.json` of the
packs, safetensors headers of the source Engram shards). Companion files: [`TARGETS.md`](TARGETS.md) (the numbers we
must hit), [`../research/LANDSCAPE.md`](LANDSCAPE.md) (what others run). Where those files already cover a
point, this one only adds DeepSeek's mechanism and the leverage.

**Sources, cited below by tag:**

- **[TR]** DeepSeek-V4.1-Flash tech report, arXiv [2609.19969](https://arxiv.org/abs/2609.19969) (PDF on head:
  `~/models/dsv41-mialab-2.9bpw/DeepSeek_V41_Tech_Report.pdf`). Sections given as TR 2.2 and so on.
- **[DS]** DSpark, arXiv [2607.05147](https://arxiv.org/abs/2607.05147) (HTML v1), and the reference code
  [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec).
- **[EG]** Engram, arXiv [2601.07372](https://arxiv.org/abs/2601.07372) v2, and
  [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) (`engram_demo_v1.py`).
- **[CFG]** `config.json` of the release (identical text_config in our pack) and the safetensors headers: Engram source
  shards (`~/models/dsv41-engram-src`) and the uncensored 2.9 bpw pack.
- **[vLLM]** vLLM's `models/deepseek_v4_1/common/engram.py` (copy in tonyd2wild's patch set), the
  [recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) and the PRs named inline.
- **[SGL]** [LMSYS V4.1 day-0 post](https://www.lmsys.org/blog/2026-09-10-deepseek-v41/) and the
  [SGLang kernel blog](https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization).
- **[K]** DeepSeek's kernel repos: [FlashMLA](https://github.com/deepseek-ai/FlashMLA),
  [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (incl. the `nv_dev` SM120 branch,
  [PR #447](https://github.com/deepseek-ai/DeepGEMM/pull/447)), [DeepEP](https://github.com/deepseek-ai/DeepEP),
  TileKernels, DeepSelect (all named in TR 3.2).
- **[REF]** DeepSeek's reference inference code in the HF repo (`inference/model.py`, `kernel.py`, `engram.py`;
  [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)), read for this file.
- **[API]** [DeepSeek API pricing](https://api-docs.deepseek.com/quick_start/pricing).

---

## 0. The short version

1. **CED replay is exact under our contract, not just "close".** The decoder's global KV is a per-token projection of
   the encoder output H_19, and the encoder is run exactly over every token. So the only prompt-boundary-dependent
   state is the decoder SWA rings, and DeepSeek rebuilds those from the last 128 prompt tokens at every prefill. Result:
   the decode start state is a **pure function of the token sequence**, whatever the turn history or cache-hit point.
   resumed == fresh and batched == alone hold inside `CED_PREFILL=replay`. It halves prefill and nothing else in the
   design gets close to that.
2. **Between turns, a session needs no decoder state at all.** A parked session = global KV (890 B/token in FP4, ~1.63
   KB in our FP8) + the 20 encoder SWA rings (1.5 MB) + 3 Engram lookback ids. Decoder rings (another 1.5 MB) are
   needed only if a session is parked mid-generation. DeepSeek drops encoder rings too ("Encoder SWA Bounded Replay")
   but that one *is* approximate: we keep the rings and offer the replay only as an opt-in.
3. **Engram addresses are pure functions of the last 4 token ids.** A verify round's rows are known the moment the
   drafter emits its block, ~6 ms before the target's layer 1 needs them. Fetch them then: NVMe latency (~0.1 ms) hides
   completely. In prefill, rows are known as soon as the prompt is tokenized, and cached prefix tokens need no rows.
4. **DSpark's own scheduler is the verify-length policy we planned** (confidence head x profiled cost curve, greedy on
   prefix survival). It also tells us the drafter is cheap to change: anything on the draft side (vocab trim, Markov
   candidate pruning, lookup drafts, recalibration) only moves acceptance, never output.
5. **The KV that the model was trained with is FP4** (QAT in post-training, E2M1 + one E4M3 scale per 16). Our FP8 rows
   are a deviation toward *more* precision. FP4 halves the pool and the park files, and is DeepSeek's reference numerics.
6. **CSA2 reuse means one KV gather per group, not per layer.** 8 index producers serve 38 sparse layers. Gathering and
   dequantizing the 512 selected rows once per group (4-6 layers) is bit-identical and cuts attention gather traffic
   ~4.75x.
7. **"EPD disaggregation" is Encoder = vision encoder.** It is not the causal encoder, and with TP2 weight placement
   there is no prefill/decode node split on 2 Sparks. The 2-node analogues: a ViT side stream with an image-embedding
   cache, and mixing decode rows into prefill chunks. During a prefill chunk, decode rows ride through layers 0-19 on
   weights the chunk already streams.
8. **Nobody else serves V4.1 as a function of the tokens alone.**
   - vLLM's encoder replay runs after *every* prefix hit (#56227: "not bit-identical to a cold run, by design").
   - vLLM's decoder replay applies only to eager prefill steps; steps of ≤ 1,024 tokens that fit a CUDA graph run the
     full decoder (#58132). So vLLM's output depends on the step size.
   - SGLang: "output is not bitwise stable across batch composition", and it refuses `--enable-deterministic-inference`.
   - DeepGEMM's 26/09 release made determinism opt-in (default off).

   Our contract is unclaimed ground on this model too.
9. **The EXL3 packs dropped the image routing bias.** The release has `gate.bias_vl` (43 tensors: 40 layers + 3
   DSpark), which REF uses for image tokens; both 2.9 bpw packs have none. So the kit routes image tokens with the text
   bias. Load `bias_vl` from the source shards (66 KB). This is a vision-quality fix, not a speed item.

The ranked list is in [section 9](#9-ranked-opportunities).

---

## 1. The model in numbers (from TR 4.2.1 and CFG)

| Item | Value | Where it matters |
| --- | --- | --- |
| Layers | 40 = causal encoder 0-19 + decoder 20-39; d = 5,120 | CED (section 2) |
| Attention | 64 query heads x 512 (RoPE on the last 64), 1 KV head (MLA-style latent, K = V), q_lora 1,280, output in 8 groups of 1,024 | KV row = 512 values |
| Layers 0-1 | SWA only (window 128) | no global KV |
| Encoder 2-19 | CSA2 ratio 2; three groups of six: Full at 2, 8, 14, Reuse the other 15 | 3 KV producers at N/2 entries |
| Decoder 20-39 | CSA2 ratio 1; Full at 20, Reindex at 24 / 28 / 32 / 36, Reuse the other 15 | 1 KV producer at N entries, KV from H_19 |
| Indexer | 32 heads x 128, FP4 QAT (since V4); top-k 512 every layer; hierarchical pool 2,048 blocks x 8 = 16,384 candidates (layer 20 builds it, decoder Reindex layers search it) | section 4 |
| SWA | every layer, window 128, FP8 (kept FP8: "sensitivity") | rings |
| Global KV | 890 B/token: encoder 3 x (288 + 68) / 2 + decoder 288 + 68 (main row FP4 E2M1 + E4M3/16 = 288 B, indexer K 68 B) | section 4 |
| MoE | every layer: 1 shared + 384 routed (top-6) of 5,120 x 2,304; `sqrtsoftplus` scores, `noaux_tc` bias for selection only, no expert groups, `norm_topk_prob`, scale 1.5; SwiGLU clamp 10 | section 6 |
| mHC | 4 residual streams, 20 Sinkhorn-Knopp iterations, Single-Pass (coefficients shifted one block) | section 7 |
| Engram | 2 modules, layers 1 and 14; orders 2 / 3 / 4 x 8 heads; ~16M-row prime tables; 256-dim FP8 rows (+ 8 UE8M0 scales = 264 B); 384M rows a module; 196B params | section 3 |
| DSpark | 3 blocks (SWA 128, 128 experts top-3 + shared, mHC), block 5, Markov head rank 256, confidence head; taps the inputs of layers 37-39 | section 5 |
| Active params | 8B per prefill token (encoder only), 16B per decode token (all 40) | section 2.5 |
| Vision | DeepSeek-ViT 32 x 1,024, patch 14, 2D-RoPE; 3x3 pixel-unshuffle; up to 1,024 tokens an image (1,344 px) | section 8 |

---

## 2. CED: Causal Encoder-Decoder (TR 2.2, 3.2.1, 3.2.2)

### 2.1 Mechanism, precisely

- **Global attention.** For decoder layers l > 20 the global KV is not computed from the layer's own hidden state. It is
  projected from the encoder output: `C_l = H_19 W_l^KV`, `Z_l = H_19 W_l^Z` (TR eq. 1). With CSA2 only layer 20 is a
  Full (KV-producing) decoder layer, so **the decoder's whole global KV is one ratio-1 cache, built per token from H_19**,
  plus its indexer K (projected from that main KV, TR 2.3). Ratio 1 means no compressor carry across tokens.
- **Local attention.** SWA KV in *every* layer comes from that layer's own hidden state, encoder and decoder alike.
- **So, for a prompt, the decoder needs exactly two things:**
  1. H_19 of every prompt token, through layer 20's projection: the decoder global KV and indexer K. This needs only
     the encoder plus one small projection, and it is what makes prefill "8B active".
  2. The decoder SWA rings at the end of the prompt: 20 layers x the last 128 positions. Building them exactly needs the
     decoder run over the last 20 x 128 = 2,560 prompt tokens (each SWA layer widens the receptive field by 127).
- **Decoder SWA Bounded Replay** (TR 3.2.2) replaces (2) by a cheap approximation. At every prefill:
  - run the decoder over the last n_win = 128 prompt tokens only;
  - truncate SWA to the replay segment: a query at i attends SWA keys in [max(s, i - 127), i], s = replay start;
  - use the rings for decoding only, never for prefix caching.

  DeepSeek simulated the same replay in post-training ("train-aware adaptation") and reports a negligible quality
  impact. Prefill becomes O(N L/2 + 128 L/2) instead of O(N L): "nearly halving total prefill computation".
- **Encoder SWA Bounded Replay** (TR 3.2.2) is a *different* knob. When a prefix hits in the global-KV cache but its
  encoder SWA rings were evicted, DeepSeek replays the last 128 tokens of the cached prefix through the encoder with
  truncated SWA, then continues with the suffix. TR says outright that this makes the suffix's KV depend on the hit
  position ("not mathematically identical across positions").
- **What DeepSeek stores** (TR 3.2.1):
  - global KV in the persistent SSD cache, at least 72 h, LRU;
  - SWA KV in a host-DRAM pool (10% of DRAM a machine) with a minutes TTL;
  - decoder SWA never.

  V4 had stored SWA at two points (end of prompt, end of output). V4.1 dropped that, with persistent KV at 1/8 of V4's.

### 2.2 The exactness consequence (our key finding)

The encoder is causal and exact, so its outputs for a token sequence do not depend on where prefills started or
stopped, provided chunked prefill is chunk-invariant (an engine property we already require). The decoder's global KV
is a per-token function of H_19. So, at the start of decoding after any prefill:

- decoder global KV = f(tokens), identical whether a token was prefilled or generated in an earlier turn;
- decoder SWA rings = g(H_19 of the last 128 prompt tokens), also f(tokens).

**In `CED_PREFILL=replay` the state is a deterministic function of the full token sequence.** It does not depend on
turn boundaries, earlier cache hits or whether a session was resumed. That is resumed == fresh by construction, as long
as two rules hold:

1. **Always replay.** Never keep the previous turn's decoder rings and run the decoder incrementally over a short new
   suffix. That would make the state depend on turn history. Follow DeepSeek: every prefill rebuilds the decoder rings
   from the last 128 prompt tokens. The cost is a 128-row decoder pass, the size of 20 verify rounds' rows, tens of ms.
2. **Exact encoder state on resume.** Keep the encoder rings with the session (1.5 MB). Do not use Encoder Bounded
   Replay, except as an opt-in knob (`ENC_REPLAY=approx`) for the case where only global KV survived (a crash, or an
   NVMe tier that lost the ring file).

drafted == serial is unaffected: decode and verify rows always run all 40 layers. batched == alone needs the
128-row replay to use the same row-invariant kernels as everything else.

Apply the replay rule to **every** prefill, whatever its size. vLLM's implementation (#58132) runs it only in eager
prefill steps and keeps the full decoder for steps of ≤ 1,024 tokens that fit a CUDA graph. That makes its result depend
on the step size, which is exactly what our rule avoids. The replay segment is 128 rows, so it fits our decode graphs
(capture an R = 128 graph for the decoder half). It is an unusual shape, not an eager-only one.

`full` mode (decoder over every prompt token) stays the M1 oracle against the kit and vLLM. The two modes give
different numbers, so each mode has its own contract and tests, as DSV41-BASELINE 2.3 proposes.

### 2.3 What the decoder state costs, and prefix sharing across sessions

| State | Size | Needed when |
| --- | --- | --- |
| Encoder global KV (3 producers, ratio 2) | 534 B/token FP4 (978 B in our FP8 rows) | always |
| Decoder global KV (layer 20, ratio 1) | 356 B/token FP4 (652 B FP8) | always |
| Encoder SWA rings | 20 x 128 x 584 B = **1.5 MB** a boundary (FP8) | resume from that boundary |
| Ratio-2 compressor carry | at most 1 token x 3 producers | at odd boundaries |
| Engram lookback | 3 token ids | any boundary |
| Decoder SWA rings | 1.5 MB | only mid-generation park / resume |
| DSpark window | 3 blocks x 128 rows | only mid-generation (it holds target taps of committed tokens) |

**Prefix sharing across sessions** (system prompts, a shared repo context for 4 agent streams) becomes cheap and exact:

- The shareable prefix is just global KV pages. They are identical for every session with the same token prefix, the
  decoder part included, because it comes from H_19, not from any session's decoder.
- The sharing boundary needs encoder rings. Store them at **checkpoint boundaries** (end of every prompt, plus every
  8K tokens of long prompts: 1.5 MB per 8K = 190 B/token, ~21% on top of FP4 global KV). A new session that shares a
  prefix resumes from the last checkpoint at or below the match point. It recomputes at most 8K tokens of *encoder
  only*, exactly, about 3-4 s at the 2,200 target; less with tighter checkpoints.
- Without CED this needs the full model's SWA state at the boundary. With CED replay it needs half of it, and no
  decoder state ever.

### 2.4 "Encoder-Prefill-Decode disaggregation" and 2 nodes

In TR 3.2, **E is the vision encoder**: "enabling vision encoding, prefill, and decoding to scale independently and
overlap in execution". It is not the causal encoder.

- **The datacenter version of "encoder-only prefill" exists.** vLLM PRs #57872 (Mooncake) and #58289 (NIXL), both
  open, build a prefill instance that runs L0-L19 plus layer 20's pre-mix and ships main KV + indexer K. The decode
  instance replays the bounded tail and owns DSpark. A prefill instance therefore needs only the encoder half of the
  weights.
- **A prefill node and a decode node still do not work for us.** The 2.9 bpw pack is 99.5 GiB a rank at TP2, so neither
  node can hold the model alone. Even an encoder-only prefill node would leave decode (all 40 layers) on one node.
- **Pipeline split (encoder on node A, decoder on node B): rejected.** Prefill would run only on A: replay prefill is
  encoder-only, so B would idle. At c1, decode would serialize the two halves.
- **What does carry over:**
  1. **Mixed prefill + decode steps get cheaper under CED.** A prefill chunk only goes through layers 0-19 (plus the
     128-row replay on its last chunk). Decode rows batched into that step share layers 0-19's expert and attention
     weight reads with the chunk, which streams those weights anyway at prefill arithmetic intensity, so they cost
     ~nothing extra there. They pay full price only in layers 20-39.

     Effect: while one stream prefills, the other 3 streams keep decoding at roughly half their normal round cost per
     step, instead of stalling or paying a full round. The scheduler should therefore co-schedule (chunk + all decode
     rows) as one step, not alternate them. Exact: rows are independent (batched == alone).
  2. **Vision encoder on a side stream** (section 8).

### 2.5 Why 16B active at decode but 8B at prefill

- Prefill computes layers 0-19 for every token (8B: 20 layers x (6 routed + 1 shared experts of 35.4M params = 248M,
  plus attention), plus Engram and layer 20's KV projection).
- Decode must compute all 40 layers for each new token, because its next-token logits come from layer 39 (16B).

Implications:

- prefill FLOPs and MoE weight traffic halve;
- decode bytes do not change. The decode round is still all 40 layers, so CED does nothing for TARGETS 3.1's decode
  model;
- the "persistent KV 1/8 of V4" is two factors: no SWA in the persistent cache (~1/2) times global KV at 1/4 of V4's
  (CSA2 sharing x FP4).

### 2.6 Leverage summary

| Idea | Impact | Exactness | Effort |
| --- | --- | --- | --- |
| `CED_PREFILL=replay` with always-replay and exact encoder rings | prefill ~1.9x (YoungAi 1,055 vs 539 on GB10; SGLang 1.37-1.56x); TARGETS 2,200 vs 1,500 tok/s | exact within the mode (2.2) | M (already planned) |
| Session park without decoder state between turns | park files 1.5 MB smaller; no decoder anything on resume | exact | S |
| Encoder-ring checkpoints every 8K for cross-session prefix share | shared system prompts / repo context at 4 streams: TTFT of a new session drops to the uncached suffix + <= 8K encoder tokens | exact | M |
| Co-scheduled chunk + decode rows | C4 decode does not stall during a 300K prefill; ~half-cost rounds during chunks | exact | M (scheduler) |
| `ENC_REPLAY=approx` opt-in | resume when only global KV survived | approximate (opt-in) | S |

---

## 3. Engram (TR 2.4.2, 3.1.3; EG; vLLM)

### 3.1 Mechanism, precisely

- **Tokenizer compression.** Each token id is mapped to a canonical id: NFKC, NFD, strip accents, lowercase, whitespace
  runs to one space, strip. A lone-space token is kept through a sentinel; partial-UTF-8 tokens are keyed raw. 129,280
  ids become 99,092 (`engram_compressed_vocab_size`). vLLM asserts that size because the hash multipliers derive from
  it.
- **Hash.** For position t and module (layer) L:
  - h_1 = x'_t * m_0, h_2 = h_1 XOR x'_(t-1) * m_1, h_3 = h_2 XOR x'_(t-2) * m_2, h_4 = h_3 XOR x'_(t-3) * m_3;
  - the n-gram hash is h_n (n = 2, 3, 4), with x' the compressed ids;
  - multipliers are odd int64 from `np.random.default_rng(10007 * layer_id)`;
  - head j of order n reads row `h_n mod p_(n,j) + offset_(n,j)`;
  - the p are 24 distinct primes just above 16,000,000 (none reused across orders or modules);
  - positions before the sequence start (and across image tokens) use the pad id.

  Heads differ only in modulus.
- **Per position and module: 24 rows** (3 orders x 8 heads) of 256 FP8 values + 8 UE8M0 scales = 264 B, so 6.3 KB a
  module and **12.7 KB a token** for both. Tables: 384,006,168 and 384,016,682 rows, 101.5 GB each on disk.
- **Fusion** (EG 2.3-2.4, vLLM `Engram`):
  - one `wkv` linear maps the 6,144-wide concatenation to 5 x 5,120: four keys, one per mHC stream, plus one shared
    value. It is FP8 in the release and EXL3 (~5 bpw) in our pack: [384, 1600, 80] trellis, ~98 MB;
  - the gate per stream is `sigmoid(signed_sqrt(RMSNorm(h_m) . RMSNorm(k_m) / sqrt(d)))`, with `q_weight` and `k_weight`
    [4, 5,120] as the norm weights;
  - output: `h_m + gate_m * v`, before the block's attention and MoE;
  - V4.1 drops the paper's causal conv (TR 2.4.2).
- **Image tokens** get no Engram contribution (gate masked) and break n-grams: a lookback across an image token hashes
  as pad (REF `forward`, vLLM `dead_mask`).
- **REF sharding.** The reference shards the table by contiguous row ranges, masks out-of-range indices to zero, and
  all-reduces the gathered rows (a `ParallelEmbedding`). vLLM shards by complete hash heads and all-gathers.
- **Placement.** Layers 1 and 14 are **both encoder layers**: the decoder has no Engram. DeepSeek prefetches "from host
  memory via background RDMA transfers, with prefetching for the first module overlapping computation in the first
  Transformer block" (TR 2.4.2).
- **Engram paper offload result** (EG 6.4, Table 4): a 100B table entirely in host DRAM costs **-1.9% / -2.8%**
  throughput (4B / 8B dense, H800, nano-vLLM, no HBM cache). That is the "<3%" figure. EG 2.5 mentions an HBM/DRAM-hot,
  NVMe-tail cache hierarchy qualitatively. **DeepSeek publishes no coverage statistic.** The "top 100M rows = 92.7%"
  figure is bot-lab-21's measurement, and "4 GiB cache hits 67-76%" is knapcio's.

### 3.2 Locality and what can be prefetched

- **The address of position t needs only tokens t-3..t.** For a verify round, the R rows are the bonus token plus the
  drafts. Their addresses are known when the drafter's block is sampled, which is before the verify forward is launched.
  For the bonus (row 0) token, the address is known at the end of the previous verify.
  - tonyd2wild: "step N+1's rows hash over the draft tokens produced at the very end of step N... speculative prefetch
    is not possible". True, but the window is not empty. The DSpark pass (~5.6 ms in TARGETS 3.1) sits *between* the
    bonus token and the verify, and the drafts appear at its end. So: issue the bonus row's 24 reads a rank at verify
    end, under the DSpark pass, and the draft rows' 24 x k at drafter end, under host bookkeeping, layer 0 and layer 1's
    attention-input prep.
  - Layer 14's rows have half a forward of slack.
- **Prefill.** Every row of a chunk is known when the request is tokenized. Pipeline chunk c+1's reads under chunk c,
  and deduplicate within a chunk: frequent bigrams repeat, while 3/4-grams mostly don't.
- **Tokens that hit the prefix cache need no Engram at all.** The output at t affects only position t's residual, and
  that is captured in the KV. Engram cost is proportional to new tokens only, which matters for agent turns.
- **Read amplification.** One 264 B row costs a 4 KB NVMe read (15x). Two layouts could fix it:
  - **(a) Hot-row cache:** a static, frequency-profiled set (top N rows from a corpus pass) in a GPU-resident hash, sized
    from the memory margin. 256 MB holds ~1M rows. It is exact (same bytes), and deterministic if static.
  - **(b) Repacked shards:** rows permuted by frequency, so hot rows share pages. This needs a row-id map: 768M x 4 B =
    3 GB, too big for RAM, so only a hot-set map is practical, and that is (a) again.

### 3.3 Two-node sharding

vLLM already shards Engram by **complete hash heads** across TP ranks (`part_n_hash_cols` = 12 of 24 a rank), then
all-gathers the rows (12 x 256 x 2 B = 6 KB a token a module).

- **Store only a rank's half on each node.** The table is laid out head-contiguous, prime range by prime range, so a
  rank's half is one contiguous byte range: ~101 GB a node instead of 203 GB. The kit's packed
  `engram-l{1,14}-r{0,1}of2.bin` already does this. Halving the IOPS a node needs is the main effect.
- **Option: row-parallel `wkv`.** Each rank multiplies its 12 heads' 3,072 columns, then one all-reduce of the 25,600
  partial outputs, instead of all-gathering rows and running a replicated wkv. It saves ~49 MB a module a rank a round
  (~0.4 ms a round for both modules), at the cost of an all-reduce of ~50-100 KB instead of a 6 KB all-gather. Exact
  under our rules: a fixed 2-term sum is order-free. The gain is small; measure the collective first.

### 3.4 Boot

- The tables are never loaded; only the 2 x 98 MB `wkv` and the norm weights are.
- Precompute the token map (129,280 tokenizer decodes, ~1-3 s in Python), the primes and the multipliers into the
  prepared folder.
- The kit's boot has to verify the packed Engram files on both nodes. Ours should check a header plus a small row
  checksum sample, not hash 100 GB.

### 3.5 Leverage summary

| Idea | Impact | Exactness | Effort |
| --- | --- | --- | --- |
| Issue verify-round reads at drafter end (bonus row at verify end), io_uring O_DIRECT on node-local half-tables | removes the 2.8 ms a step tonyd2wild measured unhidden: **~5% of a 51 ms round**; removes cold-page spikes (5.2 / 11.8 ms p50 / p90) | exact | M (planned "Engram off the critical path") |
| Hash on the host, outside the graph; GPU staging buffer per layer | needed for graphs (vcruz305 / tonyd2wild fixes) | exact | S |
| Prefill: chunk-ahead prefetch + within-chunk dedupe; skip cached prefix tokens | keeps Engram I/O (~786K reads a rank per 32K tokens, ~1 s) under compute; sfxnz saw 184 -> 838 tok/s from fixing exactly this | exact | S-M |
| Static hot-row cache in spare memory (only above the 4-6 GiB floor) | cuts IOPS by the hit rate (third-party 67-76% at 4 GiB); mainly prefill | exact | M |
| Per-node half tables | 101 GB a node of NVMe; half the IOPS | exact | S (the kit has the files) |
| Row-parallel `wkv` | ~0.4 ms a round | exact under our rules | M, low priority |

---

## 4. CSA2 and the indexers (TR 2.3, 2.4.4, 3.2; SGL; K)

### 4.1 Mechanism, precisely

- **Three modes.** Every CSA2 layer computes its own main Q and SWA KV.
  - **Full** computes main KV (with the compressor at ratio m) and indexer Q, projects indexer K from the main KV, and
    runs the indexer to pick the top-512.
  - **Reindex** reuses the latest main KV and indexer K, computes its own indexer Q, and rescores to a fresh top-512.
  - **Reuse** reuses both the KV and the latest top-512 indices and runs sparse attention directly.
  - The core attention reads the selected main KV + the 128-token SWA KV.
- **Compressor.** CSA2 removes CSA's overlapping 2m-token windows and absolute position embedding. Ratio 2: each entry
  is a gated sum of 2 tokens, non-overlapping. Indexer K is a projection of the main KV, not a separate path.
- **Hierarchical Sparse Indexer** (decoder only, post-training).
  - Layer 20 scores all causally visible decoder entries (ratio 1: all N tokens). It keeps its top-512, assigns each
    8-position block the max score of its positions, and keeps the 2,048 best blocks: a 16,384-position pool.
  - Reindex layers 24, 28, 32, 36 score only the pool.
  - Per-query indexer cost of deeper layers is constant in N. Only layers 2, 8, 14 (N/2 entries) and 20 (N entries)
    scan the full context.
- **Indexer (REF).** Q and K are MXFP4 (block 32, E8M0 scale); the score is the sum over heads of
  `relu(q . k) x w_head`, with per-query head weights from `weights_proj`. The candidate pool always pins the block
  holding the query's newest position, so a partly filled block cannot be outscored.
- **FP4 KV.**
  - Main KV is E2M1 with one E4M3 scale per 16 channels and no global scale; the bound is 448 x 6 = 2,688 vs a
    theoretical sqrt(512) = 22.6 and an observed max ~10.
  - Quantized **after** RoPE, RoPE and non-RoPE parts alike.
  - QAT'd in post-training. The indexer Q/K were FP4 QAT already in V4.
  - REF quantizes at inference too: `fp4_act_quant(latent, 16, scale_dtype=e4m3)` right after RoPE, before the cache
    write. FP4 is the model's reference numerics, not a lossy option.
  - SWA stays FP8: REF and FlashMLA use a 528 B MXFP8 record (512 + 16 UE8M0 scales per 32). On SM12x vLLM falls back
    to the V4 `fp8_ds_mla` 584 B record for *both* caches; its NVFP4 main-KV path (`nvfp4_ds_mla`) runs only through
    FlashMLA on SM100.
  - 288 B a main entry, 68 B an indexer entry (SGL; REF arithmetic agrees).
- **Kernels for GB10.**
  - DeepSeek's FlashMLA (SM100/103 only since 2026-09-30), Mega-MoE / Mega-Gate / Mega-mHC and TileKernels do not cover
    SM12x.
  - DeepGEMM's `nv_dev` branch (PR #447, merged 2026-09-23) has native SM120 BF16 / FP8 / FP4 GEMMs, HC prenorm,
    paged MQA logits and the DSv4.1 sparse MQA (indexer) kernels, and fixes three GB10 bugs (#417 fp8 read as fp4,
    #425 varlen paged-MQA IMA on sm_121a, #443 JIT). It explicitly has no SM120 Mega kernels. It is the one first-party
    reference for our csa2 indexer on GB10.
  - DeepSelect (top-k, tuned for k = 512) states no arch requirement; unverified on SM12x.
- **Kernel budget.** DeepSeek runs a Reuse-mode layer in **15 kernels at prefill and 11 at decode**: FlashMLA's fused
  RoPE-attention-RoPE-cast, DeepGEMM's Mega-Gate / Mega-mHC / Mega-MoE, TileKernels, DeepSelect TopK (TR 3.2).

### 4.2 What dominates at long context (our arithmetic, 2 ranks, indexer heads split 16 / 16)

| At N = 300K | Per decode row | Shared by the R rows of a round? |
| --- | --- | --- |
| Indexer K scanned (layers 2, 8, 14 at N/2; 20 at N) | 750K entries x 68 B = **51 MB** (0.22 ms at 230 GB/s) | yes (same K, R queries) |
| Indexer FLOPs | 750K x 32 x 128 x 2 = 6.1 GFLOP | no, x R |
| Reindex scans (4 x 16K) | 4.5 MB | per row (pools differ) |
| Selected main KV (38 sparse layers x 512 x 288-584 B) | 5.6-11 MB, gathered | per row |
| SWA (40 x 128 rows) | 3 MB | per row |
| **Weights per round** | **~5-7.5 GB** | yes |

- **Decode at 300K is still weight-bound.** The attention-side extras are ~1-3 ms a round, which matches TR Figure 2
  (decode FLOPs +25% from 4K to 1M). The "≥ 85% of short-context rate at 300K" target is architecturally easy. The risk
  is selection kernels: top-512 of 150-300K scores, and top-2,048 blocks of 37.5K, for each row and producer layer. On
  GB10, `persistent_topk` misfits SM count and smem (tonyd2wild: `top_k_per_row_decode` 1.6-3.6x faster).
- **Prefill at long context.**
  - The indexer is O(N^2): per Full encoder layer 2,048 N^2 FLOP, so ~550 TFLOP for 3 layers at 300K. At FP8/FP4 MMA
    that is ~1-2 s a node, against ~136 s of total prefill at the 2,200 target.
  - Small, *if* scoring and top-k are fused and streaming. Materializing the scores is 300K x 150K x 2 B = **90 GB a
    layer**, so scores must never hit memory (DeepSeek's DeepSelect / FlashMLA path).
  - In replay mode, layer 20's N^2 scan (~370 TFLOP at 300K) and the Reindex layers run only for the 128 replay rows: a
    second, silent CED saving.
  - At 1M the indexer becomes ~11x larger (~15-20 s a node) but MoE still dominates.

### 4.3 Leverage

| Idea | Impact | Exactness | Effort |
| --- | --- | --- | --- |
| **Gather-once per CSA2 group.** The 512 selected rows of a Full / Reindex layer are identical for its following Reuse layers (same KV, same indices). Gather and dequantize them once into a per-row bf16 tile (512 x 512 x 2 B = 512 KB a row; R = 6 is 3 MB, fits the 24 MB L2 or a scratch buffer) and let the next 3-5 layers read it | ~4.75x fewer scattered KV reads (38 -> 8 gathers a row); ~0.5-1 ms a round at R = 6, more at C4 (rows x layers) | bit-identical | M (csa2 kernels' interface) |
| **FP4 main KV in the model's QAT format** (E2M1 + E4M3/16, after RoPE) | pool 1.63 -> 0.89 KB/token: **-0.9 GiB a rank at 4 x 300K** (the KV is whole on both ranks: 1 KV head); park files and resume I/O halve; matches DeepSeek's reference numerics better than FP8 | exact if one format throughout (row-local quantization) | M (new row format in `csa2/rows.py`, dequant in the attention tile loader) |
| Fused score -> top-k for prefill indexers (never materialize scores); row-invariant tie order | keeps 300K prefill compute-bound; a precondition for the 2.5-min 300K target | exact (fixed tie order) | M-L (the csa2 index kernel) |
| GB10-shaped top-k (per-row, smem ≤ 99 KB, no oversubscription) | removes a known 1.6-3.6x selection penalty | exact | S-M |
| Replay mode also skips decoder indexers for prompt tokens | prefill at 300K: ~370 TFLOP less | exact in mode | free with CED replay |
| Kernel budget: aim at DeepSeek's 11 kernels a decode layer (fuse RoPE / inverse RoPE / cast into attention; mHC into GEMV pro/epilogues) | in graphs, ~2-4 µs a small kernel; 40 x (25 -> 11) kernels is ~1-2 ms a round | exact if the fused math is the same op order | M |

---

## 5. DSpark (TR 2.4.3; DS; DeepSpec)

### 5.1 Mechanism, precisely

- **Drafter.**
  - 3 Transformer blocks with mHC, SWA window 128, MoE (128 routed experts top-3 + shared).
  - Target conditioning: the *inputs* of target layers 37, 38, 39 (residual before the layer, mean over the 4 mHC
    streams) are concatenated (15,360), projected by `main_proj` to 5,120 and RMS-normed. Each drafter block then writes
    them into its own 128-slot SWA ring, one entry per **committed** token (KV injection, DFlash style).
  - The block input is the anchor (last committed token) + 4 noise tokens (`dspark_noise_token_id` 128,799) through the
    shared, frozen target embedding. It attends the ring plus the 5 block positions (bidirectional within the block).
  - The LM head is the target's.
- **Semi-autoregressive head.** One drafter pass gives base logits U_k for k = 1..5 in parallel. The **Markov head**
  adds a rank-256 bigram bias, B(x_(k-1), .) = W1[x_(k-1)] W2, with W1 = `markov_head.embed` [129,280 x 256] and W2 =
  `markov_head.head` [129,280 x 256]. The 5 tokens are then sampled left to right. This sequential part is the only
  serial work: 5 small full-vocab GEMVs over a 66 MB matrix.
- **Confidence head.** c_k = sigmoid(w . [h_k ; W1[x_(k-1)]]); `confidence_head.proj` is [1 x 5,376]. It predicts the
  conditional acceptance of position k given acceptance of 1..k-1. It was trained toward 1 - TV(p_draft, p_target),
  then calibrated by Sequential Temperature Scaling: raw AUC 0.81-0.90, ECE 3-8% -> ~1%.
- **Scheduler** (DS Algorithm 1).
  - Prefix survival a_(r,j) = product of c_(r,1..j).
  - Expected tokens tau = sum over requests of (1 + sum of a_(r,j) for j ≤ l_r). The objective is tau x SPS(B), where
    SPS is steps/s at B verify tokens, profiled at engine init.
  - Greedily admit (r, j) in decreasing a while the objective improves.
  - Production runs the scheduler asynchronously, with a capacity K from 2-step-old confidences (keeps CUDA graphs).
  - Below ~200 concurrent requests the verify budget grows from MTP-1's 2 to **4-6 tokens a request**. At batch 1 it
    verifies nearly the whole block, because the cost curve is flat.
- **Verification.** Chain drafts (no tree), standard rejection sampling (lossless) plus the bonus token. vLLM:
  `rejection_sample_method: block`. REF samples with the Gumbel-max / exponential-race trick (`probs / Exp(1)`,
  argmax), the same family as our keyed-noise sampling, but with unkeyed noise. REF ships no verify loop.
- **State.** The drafter ring holds only committed tokens' target taps. On rejection nothing is rolled back; the block's
  own KV is never cached.
- **Training** (DS 3.3, TR 2.4.3).
  - Target, embedding and head frozen. Loss 0.1 CE + 0.9 TV + 1.0 confidence BCE, position weights exp(-(k-1)/gamma).
  - Trained after pre-training, then co-trained through RL without gradients into the backbone, so it tracks the
    released policy.
- **Published acceptance.**
  - Paper (Qwen3, block 7, T = 1): tau 3.3-6.1, +16-18% over DFlash, +27-31% over EAGLE-3.
  - Production vs MTP-1: **+60-85% per-user speed** on V4-Flash at matched throughput.
  - TR gives **no V4.1 acceptance numbers.** SGLang's 5.5 tau on 4x GB300 is a best case. Third-party V4.1 numbers
    (LANDSCAPE 0.4) range from ~1.0 accepted on prose to ~4.8 on counting.

### 5.2 What this means for us

- **Everything on the draft side is exactness-free.**
  - Our verification decides by the target's keyed sampling (DSV41-BASELINE 2.3 point 5), so output equals serial
    output for *any* drafts.
  - Drafter numerics, vocab, Markov pruning and lookup drafts only change acceptance.
  - (Under DeepSeek's own rejection sampling, a pruned draft distribution must still be the distribution actually
    sampled from. That is not a constraint for us.)
- **Draft-side byte cuts** (the DSpark pass is ~0.9 GB a rank, ~5.6 ms, a fixed cost every round):
  - **Markov head on candidates only.** Compute the bias for the union of the base logits' top-k and nothing else (vLLM
    PR #56694 does a variant: +5-8% throughput, lossless). That replaces up to 5 sequential 66 MB GEMVs (~1.4 ms) with
    a gather of a few hundred W2 rows.
  - **Draft vocab trim.** Run the draft LM head over the top ~32K tokens by frequency: ~186 of 248 MB a rank saved
    (glm53 DRAFT-VOCAB.md has the method).
  - Together: **~1.5-2 ms a round, ~3-4% of a 51 ms round.** Every workload gains, prose most, since prose spends the
    largest share of its time on the drafter.
- **Verify length = DeepSeek's scheduler with our cost table.**
  - At c1, choose l to maximize (1 + sum of a_j for j ≤ l) / T(1 + l), with T from TARGETS 3.1: 31 / 47 / 51 / 60 ms at
    R = 1 / 3 / 4 / 6. At C4, the global greedy over all requests' (r, j) is the same algorithm.
  - Prose with conditional 0.63 / 0.33 / 0.15 gives a = 0.63 / 0.21 / 0.03, so l = 2. Code typically fills the block.
  - **Recalibrate the confidence head on our pack with STS:** a per-position temperature grid search on logged
    (c_k, accepted) pairs. Our target is the 2.9 bpw abliterated model, not the one the head was calibrated on. Hours,
    no training.
- **Raising acceptance**, in order of cost:
  1. Feed the drafter the *exact* taps: layer 37-39 inputs, mean over streams, from the verify forward's accepted rows
     only. Wrong taps silently cost acceptance, and two kits had such bugs.
  2. Source-precision drafter experts / embedding if memory allows (LANDSCAPE: 1.33 vs 2.77 claimed, 53 vs 54% counter).
     Measure first.
  3. Suffix / prompt-lookup drafts mixed in when the lookup's match is long (code edits, structured output). DS does not
     propose this; others measured +7-18% (hughmadden) and 47.6 vs ~31 (coolbho3k).
  4. On-policy refit of `main_proj`, the Markov and confidence heads (small) to our quantized, abliterated target, using
     DS's loss. The 3 x 128-expert blocks are ~13.6B params: refitting those is a later-round project.
- **No block-size change.** The drafter emits 5 per pass. k > 5 needs a second pass, and every source found k = 10
  worse.

---

## 6. MoE (TR 2.1.1, 4.2.1; CFG)

- **Routing.**
  - Scores s = sqrt(softplus(x W_g)), `scoring_func: sqrtsoftplus`.
  - Selection = top-6 of s + b, where b is the per-expert aux-loss-free correction bias (`gate.bias` [384]). Weights use
    the uncorrected s, normalized over the 6 and scaled by 1.5.
  - **No group-limited routing** (no `n_group` / `topk_group` in CFG) and no hash-routed layers: every layer, encoder
    and decoder, has a dense `gate.weight` + `gate.bias`.
  - TR trains separate text and image bias vectors. The release ships both: `gate.bias` and `gate.bias_vl`, and REF
    selects `bias_vl` for image positions. **Both 2.9 bpw EXL3 packs dropped `bias_vl`** (index checked on head), so
    our loader must take it from the source shards.
  - The sequence-level balance loss is tiny (1e-4), so per-sequence imbalance and locality are allowed.
- **Expert locality across adjacent tokens.**
  - DeepSeek publishes none. vcruz305 measured 16.1 unique experts for 24 slots at 4 rows, and 44 for 96 slots at 16
    rows, on real k = 3 traffic. Uniform random would give ~23.3 and ~84.
  - So verify rows share ~1/3 of their experts. The grouped kernel already turns that into fewer bytes (TARGETS uses
    U(R)).
  - The DSpark drafter routes the same 5 positions through 128 experts top-3, and its choices are not the target's.
- **Activation sparsity.** None designed in. SwiGLU with a clamp at 10 (fuse the clamp: tonyd2wild +3%, bit-identical).
- **Prefill under CED.** Only 20 MoE layers per prompt token, so EXL3 prefill expert throughput (TARGETS 3.3) counts
  half as much.
- **Leverage:**
  - fixed tie order in the top-6 on s + b (batched == alone);
  - routing-aware verify length: the expected unique-expert count enters T(R);
  - nothing else architectural to exploit. No groups means no node-local routing trick, and TP (not EP) stays right for
    2 nodes (LANDSCAPE 8).

---

## 7. Single-Pass mHC, MTP, and the rest

- **Single-Pass mHC is part of the model, not an optional speed-up** (TR 2.4.1).
  - It uses X_(l+1) = B_l X_l + C_l F_l(A_(l-1) X_l): each block mixes its input with the *previous* block's
    coefficients. The model was trained this way (the shift is a semantics change), so an engine that uses A_l is wrong,
    not slower.
  - The fused form (Mega-mHC) reads the 4 x 5,120 residual once and writes it once per block: (2n + 2)d vs (4n + 4)d.
  - At decode this is negligible. At prefill it is ~8 MB of activation traffic a token across 20 encoder layers (fp32
    residual, two mHC blocks a layer) in the unfused form, ~1 s per 32K tokens on one GB10, so **fusing saves ~3-4% of
    prefill**.
  - The coefficient projection `hc_*_fn` [24 x 20,480] is fp32, ~2 MB a block, 157 MB a decode step a rank. That is a
    fixed ~0.7 ms a round, worth keeping in mind for the byte model. Do not downcast; that changes numerics.
- **MTP.** None in the backbone ("We omit the MTP module", TR 2.1). The `mtp.0-2` tensors *are* the DSpark blocks
  (`num_nextn_predict_layers: 3`).
- **Sinks.** `attn_sink` per layer; DSV41-BASELINE 2.3 point 6 has these.
- **Reasoning effort.** The API maps max / high / low to a scalar b = 100 / 75 / 50 (TR Table 2). Lower effort means
  shorter outputs: TR B.3 shows 2.0-3.1x more tokens from low to max on AIME. That is the cheapest end-to-end speed-up
  per task, and it is the user's choice, not ours. Our RigMark protocol pins `low`.
  - **Engines disagree on what "low" means.** vLLM maps low / high / xhigh / max to 25 / 50 / 75 / 100 (default 50).
    SGLang maps them to 50 / 75 / 75 / 100 (default 75). DeepSeek's API uses low = 50.
  - The baseline window should record which b the kit actually sent. A RigMark cell at b = 25 against one at b = 50 is
    not the same workload, since output length is a direct multiplier on wall time.
  - **Recorded (baseline window 2026-10-01, [BASELINE-RESULTS](BASELINE-RESULTS.md)):** the kit renders
    `chat_template_kwargs.reasoning_effort: "low"` as **b = 25** (vLLM's map, not its own jinja template's 50). With no
    kwargs it renders 50, and a top-level `reasoning_effort` is ignored. Our RigMark body therefore means b = 25 on the
    kit; our engine must send the same scalar for a like-for-like receipt.

---

## 8. Vision path (TR 2.1.1, 3.1.1)

- DeepSeek-ViT: 32 layers x 1,024, 16 heads, patch 14, 2D-RoPE, SwiGLU, RMSNorm. Then a 3x3 pixel-unshuffle (9x fewer
  tokens) and a 2-layer MLP projector to 5,120. Embeddings are inserted at image-token positions. At most 1,024 tokens
  an image (~1,344 px).
- Production uses EPD: the ViT is its own pool. Training also disaggregates it.
- **For us:**
  - Run the ViT (~0.4B params, ~0.8 GB BF16) on one rank's side stream when the request is admitted, overlapping other
    streams' decode, then send the ≤ 10 MB of embeddings to the other rank.
  - **Cache image embeddings by content hash.** Agent loops resend the same screenshots, and the prefix cache only
    helps when the image sits in the cached prefix. Exact (deterministic ViT).
  - Load the ViT lazily after "ready", to keep it off the boot path.
- Image tokens use `bias_vl` in the router (section 6: missing from our pack) and get no Engram (section 3.1).
- REF requires image spans to be prefilled within one chunk starting at position 0. The in-image bidirectional mask is
  an open SM12x kernel gap (LANDSCAPE 7).

---

## 8.5 DeepSeek's own serving numbers and recommendations

| Item | Value | Source |
| --- | --- | --- |
| Sampling | temperature 1.0, top_p 0.95 or 1.0; max_tokens ≥ 256K; evals at effort 100 | model card |
| Reference deployment | TP8 torchrun, FP4 experts, TileLang kernels (`tilelang==0.1.8`); no verify loop in REF | REF README |
| Production | EPD (vision / prefill / decode pools); Reuse layers 15 kernels at prefill, 11 at decode | TR 3.2 |
| KV tiers | global KV on SSD ≥ 72 h (LRU); SWA in a 10%-of-DRAM pool, minutes TTL; decoder SWA never stored | TR 3.2.1 |
| Speculation | DSpark: verify budget 4-6 tokens a request below ~200 concurrent; +60-85% per-user speed vs MTP-1 at matched throughput (V4-Flash) | DS 5.4 |
| Acceptance | none published for V4.1. vLLM recipe: "golden acceptance length 3.51" | vLLM recipe |
| Prefix caching economics | cache-hit input $0.003 / M vs cache-miss $0.15 / M (50x), output $0.60 / M (off-peak) | API |
| Speeds | **none in TR**. Third-party: API ~209-218 tok/s, TTFT 0.97 s (Artificial Analysis); SGLang 4x GB300 BS1 203 tok/s plain, 874 with simulated acceptance | LANDSCAPE 5, SGL |
| Known limits (DeepSeek's own) | "Potential selection errors in CSA2 and approximate state reconstruction in SWA Bounded Replay may still cause capability degradation in untested boundary cases" | TR 6 |

For us:

- The 50x price gap says DeepSeek's whole stack is built around prefix reuse. Our resume / park / share work (TARGETS
  2: replay TTFT, NVMe sessions) follows the same design priority.
- DeepSeek's own limitations line is the best argument for keeping `CED_PREFILL=full` as a tested mode, and for running
  needle tests at 64K / 128K / 300K in replay mode (TARGETS 2: quality, CED replay mode).

---

## 9. Ranked opportunities

Ranked by (gain on our TARGETS) / (effort), for the round-1 priorities of TARGETS 2.1: boot ≤ 2 min and blanket speed.
"Round" = one decode verify round (~51 ms at R = 4).

### Round 1: boot and blanket speed

| # | Opportunity | Section | Expected impact | Exactness | Effort |
| ---: | --- | --- | --- | --- | --- |
| 1 | **CED replay prefill**, always-replay rule, exact encoder rings | 2.2 | prefill ~1.9x (the 2,200 target); TTFT of agent turns; the decoder indexers' N^2 cost disappears from prefill | exact in mode (proved in 2.2) | M |
| 2 | **Engram reads issued at drafter end / verify end**, io_uring O_DIRECT on per-node half tables, host hash outside the graph | 3.2, 3.3 | ~2.8 ms a round off the critical path (~5%), no cold-page spikes; every workload | exact | M |
| 3 | **Draft-side byte cuts:** Markov head on candidates + draft vocab trim | 5.2 | ~1.5-2 ms a round (~3-4%); prose gains most | exact (draft side) | S-M |
| 4 | **Gather-once per CSA2 group** | 4.3 | 38 -> 8 KV gathers a row; ~0.5-1 ms a round, more at C4 and depth | bit-identical | M |
| 5 | **Kernel budget toward 11 a decode layer** (fused RoPE/cast in attention, mHC into GEMV pro/epilogues, SwiGLU clamp fused) | 4.3, 7 | ~1-2 ms a round | exact if op order is kept | M |
| 6 | **Boot:** Engram never loaded; token map, primes, multipliers precomputed in the prepared folder; Engram files checked by header + sample; ViT loaded lazily; decode graphs only for the R set used | 3.4, 8 | removes the kit's Engram staging / verification from boot; helps the ≤ 2 min target | n/a | S |
| 7 | **Prefill Engram pipeline:** chunk-ahead reads, in-chunk dedupe, skip cached tokens | 3.2 | prevents I/O-bound prefill on novel text (others: 184 -> 838 tok/s) | exact | S-M |
| 8 | **Fused score -> top-k indexer + GB10-shaped top-k** | 4.3 | precondition for 300K prefill in ≤ 2.5 min and ≥ 85% decode at depth | exact (fixed ties) | M-L (in progress in `engine/kernels/csa2`) |

Items 2-5 stack on the decode round. Together they are worth roughly **6-10 ms of a ~51 ms round (12-20%)**. That
closes most of the gap between the kit's ~59% and our assumed 75% of the bandwidth floor, without touching acceptance.

These ride along with round 1 and are correctness, not speed. They are cheap, so do them while the loader and CED code
are being written:

- load `gate.bias_vl` from the source shards (section 6);
- apply the decoder replay rule to every prefill size (2.2);
- tap the *inputs* of layers 37-39 for DSpark (5.2);
- mask Engram on image tokens (3.1);
- record the effort scalar the kit sends (7).

### Round 2 and later

| # | Opportunity | Section | Expected impact | Exactness | Effort |
| ---: | --- | --- | --- | --- | --- |
| 9 | Confidence-scheduled verify length (DS Algorithm 1 with our T(R)) + **STS recalibration** on our pack | 5.2 | per workload: prose R = 3 instead of 4 (~8% a round), code fills the block; C4 global greedy | exact (any l) | S-M |
| 10 | **FP4 main KV** in the QAT format | 4.3 | -0.9 GiB a rank at 4 x 300K (memory floor, or an Engram hot-row cache), half-size parks, faster resume; closer to the reference | exact in format | M |
| 11 | **Session park without decoder state** + encoder-ring checkpoints every 8K for cross-session prefix sharing | 2.3 | resume ≤ 1 s with less I/O; new sessions sharing a long prefix skip it exactly | exact | M |
| 12 | Co-scheduled prefill chunk + decode rows (CED makes decode rows half price during chunks) | 2.4 | C4 keeps decoding during long prefills | exact | M |
| 13 | Suffix / prompt-lookup drafts mixed with DSpark | 5.2 | structured / code edits (+7-18% others) | exact | M (0020 exists) |
| 14 | Static Engram hot-row cache from any memory above the floor (more if FP4 KV lands) | 3.2 | IOPS down by the hit rate; mostly prefill | exact | M |
| 15 | Single-Pass mHC fused for prefill (one residual read / write a block, sequence-parallel) | 7 | ~3-4% prefill | exact if op order is kept | M |
| 16 | ViT side stream + image-embedding cache | 8 | multimodal track loops | exact | M |
| 17 | Row-parallel Engram `wkv` | 3.3 | ~0.4 ms a round | exact under our rules | M |
| 18 | DSpark refit to our target (`main_proj`, Markov, confidence first; blocks later) | 5.2 | acceptance; the only prose lever besides round time | exact (draft side) | L |
| 19 | `ENC_REPLAY=approx` opt-in | 2.1 | resume when only global KV survived | approximate, opt-in | S |

**Not worth pursuing (architecturally):**

- a prefill node / decode node split, or an encoder / decoder pipeline split on 2 nodes (2.4);
- k > 5 drafts (5.2);
- downcasting the fp32 mHC coefficient weights (7);
- expert-group or node-local routing tricks: the router has no groups (6);
- Encoder SWA Bounded Replay as a default: it breaks resumed == fresh (2.1).
