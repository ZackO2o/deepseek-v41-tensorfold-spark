# DeepSeek-V4.1-Flash: architecture, TP=2 plan and byte budget (2026-10-01)

What the model computes, layer by layer, with the shapes and bit widths of the checkpoint we serve
(`dsv41-uncensored-2.9bpw`, the MiaAI / dealignai EXL3 mul1 pack, layout identical to the mialab pack), how it splits
over two GB10s, what a token costs in KV and state, what a decode step reads, and which parts of our GLM-5.3 engine
carry over.

Sources, in order of authority:

1. vLLM's DeepSeek-V4.1 model code shipped in the kit image (`vllm/models/deepseek_v4_1/`, plus the V4 MoE base,
   `model_executor/kernels/mhc/`, `layers/sparse_attn_indexer.py`, `kernels/attention/dsa/candidate_blocks.py`,
   `v1/worker/gpu/spec_decode/{dflash,dspark}/`; Apache-2.0), extracted read-only from the stopped container
   `dsv41-exl3-head` on head. This is the M1 oracle's code, so it is the spec.
2. The kit's EXL3 overlay (`/opt/dsv41/exl3.py`, `engram_layout.py`, `pack_engram.py`) for how the pack is applied.
3. The checkpoint itself: `config.json`, all 39 shard headers (192,452 tensors), the original FP8 Engram shards
   47-48 (`~/models/dsv41-engram-src`).
4. `glm53-tensorfold-spark/docs/DSV41-BASELINE.md` and `NEXT-DEEPSEEK-V41-FLASH.md` (corrections in section 9).

The executable form of this document is `engine/reference/` (pure PyTorch, CPU or CUDA); every formula below is in it,
with the vLLM file it follows named in each module's docstring. `python -m engine.reference.budget` reproduces the
byte tables from the headers.

Upstream TensorFold v0.6.0 has a `families/deepseek_v4` (V4-Flash, MLX): it has no Engram, no CED, no CSA2 sources,
so it is not used. Its `cuda/exl3/format.py` (Apache-2.0 from 0.6.0) is vendored as the EXL3 oracle.

## 1. Global shape

| | |
| --- | --- |
| Vocabulary | 129,280 (BOS 0, EOS 1, pad 2, DSpark noise 128,799, image 129,264) |
| Hidden D | 5,120, carried as **4 hyper-connection streams** [T, 4, 5,120] bf16 |
| Blocks | 40 = CED encoder 0-19 + decoder 20-39; then 3 DSpark blocks (`mtp.0-2`) |
| Attention | 64 query heads x 512, **one K=V head of 512** (448 NoPE + 64 RoPE, GPT-J pairs on the last 64 dims), sinks, 128-token window on every layer, CSA2 compressed rows on 36 layers |
| MoE | 384 routed experts top-6 (sqrt-softplus, bias selection, normalized, x1.5) + 1 shared; SwiGLU clamped at 10; expert 5,120 -> 2,304 -> 5,120 |
| Engram | layers 1 and 14 |
| Norm eps | 1e-20 everywhere (RMSNorm, hc mixes, q per-head norm) |
| RoPE | ratio-0 layers (0, 1, DSpark): theta 10,000 plain. ratio > 0 layers: theta 160,000, DeepSeek YaRN x16 over 65,536 (beta 32 / 1, mscale 1) |
| Head | untied, EXL3 6-bit, 5,120 -> 129,280 |

Per token, block L (`deepseek_v4_1/nvidia/model.py`; `engine/reference/model.py: Block`):

```
streams [T,4,D] bf16                                      (block 0: the embedding copied 4 times)
if L in {1, 14}:  streams = engram(streams)              (before the attention pre, on all 4 streams)
post, comb, x, pre_a = hc_pre(streams, hc_attn, pre_in = previous block's FFN pre-mix | stream 0 at L = 0, attn_norm)
streams = hc_post(attention_L(x), streams, post, comb)
post, comb, x, pre_f = hc_pre(streams, hc_ffn, pre_in = pre_a, ffn_norm)
streams = hc_post(moe_L(x), streams, post, comb)
```

Final: `h = sum_i pre_f(39)[i] * streams[i]` (bf16), `logits = head(norm(h))`. V4.1 has no learned `hc_head`.

## 2. Hyper-connections (mHC, Single-Pass form)

Per sublayer: `fn [24, 20,480]` fp32, `base [24]`, `scale [3]` (`hc_{attn,ffn}_{fn,base,scale}`).

```
mixes = (streams.flat @ fn.T) * rsqrt(mean(streams.flat^2) + 1e-20)          [T, 24] fp32
pre   = sigmoid(mixes[0:4]  * scale[0] + base[0:4]) + 1e-6                     -> the NEXT sublayer's input mix
post  = sigmoid(mixes[4:8]  * scale[1] + base[4:8]) * 2
comb  = Sinkhorn_20(softmax_row(mixes[8:24].view(4,4) * scale[2] + base[8:24]))   (+1e-6, column-normalize first,
        then 19 x (row, column); ends column-stochastic, rows ~1)
x     = RMSNorm(bf16(sum_i pre_in[i] * streams[i]), norm_weight)              the sublayer input, bf16
streams'[j] = bf16(sum_i comb[i, j] * streams[i] + post[j] * out)
```

The "delayed" pre-mix is the V4.1 detail: a sublayer's input is collapsed with the **previous** sublayer's `pre`
(attention uses the last block's FFN `pre`, the FFN uses this block's attention `pre`), while its own `pre` is handed
on. Block 0's attention input is stream 0 (= the embedding). vLLM feeds block 0 the plain embedding with `fn` summed
over the 4 copies, which gives the same mixes (tested). The bf16 rounding of the collapsed input before its RMSNorm is
part of the kernel and of the reference.

## 3. Attention (CSA2)

### 3.1 Every layer

| step | shapes | weight (K in -> N out, bits) |
| --- | --- | --- |
| `qr = q_norm(wq_a x)` | [T, 1,280] | `attn.wq_a` 5,120 -> 1,280, 5 bit (layer 0: 6) |
| `kv = kv_norm(wkv x)` | [T, 512] | `attn.wkv` 5,120 -> 512, 5 bit (layer 0: 6) |
| `q = rope(rms(wq_b qr))` per head, weightless RMSNorm | [T, 64, 512] | `attn.wq_b` 1,280 -> 32,768, 5 bit |
| SWA row = `rope(kv)` at the token's position, stored fp8_ds_mla | [T, 512] | |
| scores over (window rows U selected compressed rows), scale 1/sqrt(512), **sink logit per head**, V = K | [T, 64, 512] | `attn.attn_sink` [64] fp32 |
| inverse RoPE on the output's last 64 dims (at the query position) | [T, 64, 512] | |
| `z_g = wo_a[g] o[:, 8g:8g+8]` for g = 0..7 | [T, 8, 1,024] | `attn.wo_a.slice.g` 4,096 -> 1,024, 5 bit, x8 |
| `out = wo_b z` | [T, 5,120] | `attn.wo_b` 8,192 -> 5,120, 5 bit |

Sink softmax: `p_j = exp(s_j - m) / (sum_k exp(s_k - m) + exp(sink_h - m))`, `out = sum_j p_j K_j`. The window is the
token itself and the 127 before it. On SM12x the kit clamps vision's in-image bidirectional window off; the text path
never needs it.

### 3.2 Layer schedule

| layers | compress ratio | mode | KV rows from | top-k from | extra weights |
| --- | ---: | --- | --- | --- | --- |
| 0, 1 | 0 | SWA only | - | - | (Engram on 1) |
| 2, 8, 14 | 2 | **Full**: compressor + indexer with K | own | own | compressor `wkv`, `wgate` 5,120 -> 512 (5 bit), `norm` [512]; indexer `wq_b` 1,280 -> 4,096 (5 bit), `wk` 512 -> 128 (8 bit), `k_norm` [128], `weights_proj` [32, 5,120] fp16 (Engram on 14) |
| 3-7, 9-13, 15-19 | 2 | Reuse | 2 / 8 / 14 | 2 / 8 / 14 | - |
| 20 | 1 | **Full**, candidate source | own | own | compressor `wkv` only (no gate at ratio 1); indexer as above |
| 21-23 | 1 | Reuse | 20 | 20 | - |
| 24, 28, 32, 36 | 1 | **Reindex** (own query over layer 20's keys, masked to the candidates) | 20 | own | indexer `wq_b`, `weights_proj` |
| 25-27, 29-31, 33-35, 37-39 | 1 | Reuse | 20 | 24 / 28 / 32 / 36 | - |
| DSpark 0-2 | 0 | SWA (non-causal over the draft block) | - | - | - |

So the decoder's global KV is layer 20's compressor applied to layer 20's own input (the stream that leaves the
encoder): there is no separate "H_19 projection" tensor. Every block keeps its own SWA rows.

### 3.3 Compressor (KV sources)

```
kv_c  = bf16(wkv_c x), score = bf16(wgate_c x)                    fp32 math, [T, 512] each
ratio 2: for each closed group g = (2g, 2g+1):  pooled = sum_{t in g} softmax_t(score[t, d]) * kv_c[t, d]   (per dim, no APE)
ratio 1: pooled = kv_c
latent = bf16(RMSNorm(pooled, compressor.norm))                    [C, 512]
compressed row g = fp8_ds_mla(rope(latent_g) at position g * ratio)
index key g      = fp8(rope(bf16(k_norm(bf16(wk latent_g)))) at position g * ratio)   [C, 128]
```

A ratio-2 group is published when its second token arrives; the open group's first token waits in a per-slot carry
(two fp32 rows of 1,024). A query at position p sees compressed rows `g < (p + 1) // ratio`.

### 3.4 Indexer and selection

```
qi[t, h] = fp8(bf16(rope(wq_b_i qr)))             [T, 32, 128]
w[t, h]  = bf16(weights_proj x) / sqrt(128) / sqrt(32)
score[t, s] = sum_h w[t, h] * relu(qi[t, h] . key[s]),   s < (pos_t + 1) // ratio
```

- If the batch's longest sequence has at most 512 compressed rows, every valid row is selected (vLLM's short path).
- **Candidate source (layer 20)**: block score = max of `score` over 8 consecutive compressed positions; the newest
  block is pinned to +inf; the top **2,048 blocks** (16,384 positions) are kept per row.
- **Reindex layers (24, 28, 32, 36)** set every score outside the candidate blocks to -inf, then select.
- Top-512 (fewer when fewer are valid). Reuse layers take the latest index source's selection.
- FP8 on SM12x (MXFP4 is SM10x only): keys with one power-of-two scale a token, queries one a (token, head), the
  query scale folded into `w`.

## 4. Engram (layers 1 and 14)

```
id'   = compress(token id)              tokenizer normalizer (NFKC, NFD, strip accents, lowercase, whitespace) -> 99,092 ids
for n = 2..4 (shift s = 0..n-1):  rolling ^= value_s * mult[layer, s]      value = pad id once a slot is before the
                                                                              sequence start or an image token (and for
                                                                              every older slot after that)
row[n, h] = rolling % prime[layer, n, h] + offset[layer, n, h]                24 rows a token (3 orders x 8 heads)
e   = bf16(fp8 row x 2^(ue8m0 - 127) per 32)                                  [T, 24, 256]
kv  = wkv(e.flat)                                                             6,144 -> 25,600 (5 bit on L1, 4 on L14)
key_h = kv[:, h*5120:(h+1)*5120] (h = 0..3), value = kv[:, 4*5120:]
dot_h = sum(x_h * q_h * k_h * key_h) * rsqrt(ms(x_h)) * rsqrt(ms(key_h)) / sqrt(5120)
gate_h = sigmoid(sign(dot_h) * sqrt(max(|dot_h|, 1e-6)))                      (0 on image tokens)
x_h  <- bf16(x_h + gate_h * value)
```

- Primes are the next unused primes above 15,999,999, drawn in (layer, order, head) order. Their sums are exactly the
  checkpoint's 384,006,168 / 384,016,682 table rows (tested), so the reference regenerates the layout from the
  config.
- `mult` comes from `numpy.random.default_rng(10007 * layer)`, odd and bounded so `id * mult` fits int64.
- `q_weight`, `k_weight` [4, 5,120] (fp32 in the pack, bf16 parameters in vLLM).
- The tables (FP8 [rows, 256] + UE8M0 [rows, 8], 98 / 98 GB) are never quantized. They live only in the original
  shards 47-48 and in the kit's packed per-rank files (`engram-l{1,14}-r{0,1}of2.bin`: 4 KiB header, then 264-B
  records of 256 weight bytes + 8 scale bytes for the rank's complete hash heads 0-11 or 12-23).
- A token's rows depend only on its own id and the 3 before it, so the reads of a decode round, drafted rows included,
  are known before the forward starts.

## 5. MoE

| | backbone (L 0-39) | DSpark |
| --- | --- | --- |
| router | `ffn.gate.weight` [384, 5,120] (fp16 in the pack, bf16 parameter in vLLM), `gate.bias` [384] | [128, 5,120] |
| routing | `s = sqrt(softplus(x @ G.T))` (fp32); experts = top-6 of `s + bias`; weights = their `s` / sum x 1.5 | top-3 |
| expert | `w1`, `w3` 5,120 -> 2,304; `w2` 2,304 -> 5,120; `down(silu(min(g, 10)) * clamp(u, +-10))` | same |
| bits | 3 (2 on layers 18-22) | 4 |
| shared | same shapes; 5 bit on 0-10 and 30-39, 4 bit on 11-28, layer 29 w1/w3 4 + w2 5 | 4 |
| one expert | 13.27 MB at 3 bit, 8.85 MB at 2 bit | 17.7 MB |

The kit's EXL3 linears cast their input to fp16, accumulate in fp32 and return the activation dtype; the routed
experts keep fp32 between `w1/w3` and `w2` and sum in fp32 (`apply_exl3_python_loop`). The reference emulates this
(`Numerics.kit()`).

## 6. DSpark (checkpoint `mtp.0-2`)

- **Taps**: the mean over the 4 streams of the stream **entering** layers 37, 38, 39 (that is, after layers 36-38;
  vLLM's eagle3 utils use the V4.1 ids as capture-at-entry).
- `main_x = main_norm(main_proj(cat(tap37, tap38, tap39)))`, `mtp.0.main_proj` 15,360 -> 5,120 (4 bit).
- One pass drafts N positions (the kit runs N = 3; `dspark_block_size` 5): inputs `[anchor, noise x (N - 1)]` at
  positions P..P+N-1 (P = the anchor's position), every position predicting the next token.
- Each of the 3 blocks is a full block (hc, attention, 128-expert MoE). Its attention is SWA and **non-causal**: every
  draft row sees all N draft rows plus the context rows P-128..P-1. A context row's K/V is that block's own
  `rope(kv_norm(wkv(main_x)))` at that position, so a slot keeps 3 rings of 128 rows that the target's taps refresh
  each round.
- Head: `logits_i = head(norm(h_i)) + markov_w2 @ markov_w1[previous token]` (`markov_head.embed` [129,280, 256]
  bf16, `markov_head.head` [129,280, 256]), sampled left to right from the anchor.
- Confidence: `sigmoid(proj([h_i, markov_w1[prev]]))`, `confidence_head.proj` [1, 5,376]: the acceptance estimate a
  cost model can use for the verify depth.

## 7. TP=2 split

| part | rank 0 / rank 1 | exchange after it (decode, R rows) |
| --- | --- | --- |
| embedding | vocab halves (64,640 rows) | one all-reduce / all-gather of [R, 5,120] at the start |
| hc mixes, norms, router | replicated | - |
| `wq_a`, `wkv` | replicated (vLLM fuses them with `disable_tp`) | - |
| `wq_b` | heads 0-31 / 32-63 (N split 16,384, a multiple of 128 so the Hadamard blocks stay local) | - |
| SWA rows, compressed rows, index keys, compressor, indexer | **replicated** (one KV head: the cache cannot be split by head; both ranks compute the same latent and selection) | - |
| sparse attention core | 32 heads a rank | - |
| `wo_a` | slices 0-3 / 4-7 | - |
| `wo_b` | K 4,096 / 4,096 | **1: reduce [R, 5,120]** |
| routed + shared experts | intermediate 1,152 / 1,152 (w1 / w3 N split, w2 K split; 9 Hadamard blocks each) | **2: reduce [R, 5,120]** (routed + shared summed first) |
| Engram tables | hash heads 0-11 / 12-23 (the kit's packed shards), 12 rows a token a layer a rank | gather [R, 24, 256] bf16 (12 KB a row) |
| Engram `wkv` | N 12,800 / 12,800 | gather [R, 25,600] bf16 (vLLM replicates it; splitting saves 88 MB a step for 2 small exchanges) |
| head | vocab halves | gather of each rank's top candidates (or [R, 129,280] when full logits are needed) |
| DSpark | as a backbone block; `main_proj` and the Markov / confidence heads replicated | 2 a block |

Two exchanges a block, 80 a forward plus the embedding, head and 2 Engram gathers. A row is 10 KB (bf16 [5,120]); a
6-row verify moves 60 KB, inside the 256 KiB RoCE one-shot path. The residual streams never cross the link.

Resident weights a rank (`budget.py`):

| | GiB a rank |
| --- | ---: |
| routed experts (backbone) | 91.29 |
| attention `wq_b` / `wo_b` / `wo_a` (halves) | 0.49 / 0.49 / 0.39 |
| shared experts (halves) | 0.37 |
| `wq_a` + `wkv` (replicated) | 0.22 |
| hc (fp32) / router (bf16) | 0.15 / 0.15 |
| Engram `wkv` (halves) | 0.08 |
| indexer + compressor (replicated) | 0.04 |
| embedding / head (vocab halves) | 0.62 / 0.23 |
| DSpark (experts 3.17, rest 0.30) | 3.47 |
| vision (rank 0 only) | 0.90 |
| **total** | **98.9** (98.0 without vision) |

## 8. KV, state and bytes

### 8.1 Per token (pooled, grows with context; identical on both ranks)

| cache | rows a token | bytes a row | bytes a token |
| --- | ---: | ---: | ---: |
| compressed KV of layers 2, 8, 14 (ratio 2) | 3 x 0.5 | 584 (fp8_ds_mla: 448 FP8 + 64 bf16 RoPE + 8 UE8M0 scales) | 876 |
| compressed KV of layer 20 (ratio 1) | 1 | 584 | 584 |
| index keys of the same 4 sources | 2.5 | 132 (128 FP8 + fp32 scale) | 330 |
| **total** | | | **1,790 B** (vLLM pads rows to 576-B alignment: ~1.8 KB) |

1M pooled tokens = 1.75 GiB a rank. The model was trained for FP4 main KV (E2M1 + E4M3 per 16). Assuming FP4 NoPE and
bf16 RoPE (448 / 2 + 28 + 128 = 380 B a row), that is ~1.28 KB a token with FP8 index keys, if we ever want it (exact
within one format).

### 8.2 Per slot (bounded, independent of context)

| state | size |
| --- | ---: |
| SWA rings: 40 blocks x 128 rows x 584 B | 2.99 MB |
| DSpark SWA rings: 3 x 128 x 584 B | 0.22 MB |
| compressor carries: layers 2, 8, 14, the open group's [kv, score] fp32 (vLLM keeps an 8-row ring: 3 x 8 x 4 KB) | 12-96 KB |
| Engram lookback: the last 3 token ids | 12 B |
| hc / pre-mix carry between blocks | none across steps |

A session snapshot is the pooled pages plus these ~3.3 MB; storing the SWA rings is what makes resumed == fresh
exact (DeepSeek drops them and replays).

### 8.3 Bytes a rank for one forward (decode / verify of R rows; `budget.py`)

Non-expert reads are the same for any R: **2.80 GB**:

- attention 1.70 GB (the `wq_b` / `wo_a` / `wo_b` halves and the replicated `wq_a` / `wkv`);
- shared expert halves 0.40, hc 0.16 (fp32) and router 0.16 (bf16), head half 0.25, Engram `wkv` half 0.09, indexer
  and compressor 0.04.

Experts: 6.38 MB an expert half (mean over layers), U(R) distinct experts a layer (assumed from GLM's measured U,
scaled to 384 experts; to be replaced by a routing trace).

| R | U | experts | KV + indexer at 64k | total | floor at 230 GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 1.53 GB | 44 MB | 4.37 GB | 19.0 ms |
| 4 (k = 3) | 18.5 | 4.72 GB | 76 MB | 7.60 GB | 33.0 ms |
| 6 (k = 5) | 25 | 6.38 GB | 98 MB | 9.28 GB | 40.3 ms |
| 8 | 30 | 7.66 GB | 120 MB | 10.58 GB | 46.0 ms |

- KV: the 128-row window on 40 layers (3 MB) plus each row's 512 selected compressed rows on 36 layers (10.8 MB a
  row, gathered).
- Indexer keys are scanned once a forward: `(3 x N/2 + N) x 132 B` for the four sources plus 4 x 16,384 x 132 B for
  the Reindex layers. That is 30 MB at 64k and **355 MB at 1M** (+8% of R = 1), the one context-dependent term.
- A DSpark pass: ~0.19 GB of block weights (attention, shared, hc, router, `main_proj`) + the head half 0.25 GB + up
  to 3 x 15 distinct 4-bit experts (8.85 MB halves, 0.40 GB): ~0.85 GB. Plus the Markov bias: `markov_head.head` is
  66 MB, read once a drafted position unless the bias is applied to the base logits' top candidates only (vLLM's
  `dspark_draft_topk`).
- Prefill FLOPs a token a rank: routed experts 8.5 GFLOP, attention projections 5.4, shared 1.4, attention core 1.5
  (640 keys x 32 heads x 36 layers + 128 keys on 4), router + hc 0.2; indexer `32 x 128 x 2 x keys` on the 8 index
  sources: 1.9 GFLOP at a 64k context. ~17.0 GFLOP plus the indexer, like GLM's 18.4.

## 9. Carry-over onto our GLM engine (`glm53-tensorfold-spark`)

| V4.1 piece | GLM component | verdict |
| --- | --- | --- |
| 512-wide K=V latent row, 64 query heads | latent MLA rows (0060), FP8 latent rows (0220: 528-B rows) | **adapt the row format**: 448 FP8 NoPE + 64 bf16 RoPE + 8 UE8M0 scales (584 B) instead of 528-B FP8 rows. No absorb / expand stage (the row *is* the head) |
| sparse attention over selected rows | b12x sparse attention (0360 / 0410): 64 heads over gathered 512-wide FP8 rows | **adapt**: two sources (128 SWA rows from the block's own ring + up to 512 rows from the group's compressed cache), a sink term in the softmax denominator, inverse RoPE on the output's last 64 dims, scale 1/sqrt(512) |
| indexer 32 x 128, FP8, ReLU-weighted | DSA indexer (0004 / 0050 / 0065: 32 x 128, top-2,048 over kpool-4) | **adapt**: top-512; keys = `k_norm(wk(latent))` of the source's compressor, not a separate K projection; 4 Full + 4 Reindex scorers; block-max candidate pool of 2,048 x 8; the fixed tie order of our selection carries over |
| KV pool (0290) | 11 latent caches + MTP caches in 256-row pages | **adapt**: 4 compressed caches + 4 key caches, pages counted in compressed rows (ratio-2 caches fill at half speed); 1.8 KB a token instead of 7.4 |
| hyper-connections (0320 / 0520: D 4,096, 4 streams, 20 Sinkhorn, fn bf16) | `hc_pre` / `hc_post`, the fused boundary kernel | **adapt (small)**: D 5,120, fn fp32, the delayed pre-mix (the input collapse uses the previous sublayer's pre, which 0520's fused boundary already has in hand), bf16 rounding before the input RMSNorm, eps 1e-20 |
| routed experts (GLM: 288 x 4,096 x 2,048, top-8, EXL3 4-bit mcg, 6.29 MB halves) | grouped expert kernels, streaming loads, expert prefill (0006 / 0260 / 0440 / 0580 / fat) | **adapt**: 384 x 5,120 x 2,304 top-6, mul1 at 3 bit (2 on 18-22), 6.64 MB halves (4.42 at 2 bit); DSpark 128 x top-3 at 4 bit. The mcg-only kernels need mul1 (upstream's `cuda/exl3/` grouped kernel); tile shapes 5,120 -> 1,152 and 1,152 -> 5,120 |
| router | GLM sigmoid router + bias | **adapt**: sqrt-softplus scores, bias for selection only, renormalize x 1.5 |
| non-expert q4mse matrices (0001 / 0470) | | **n/a**: the pack's non-expert matrices are already EXL3 (5-6 bit); serve them through upstream's row-invariant EXL3 linear |
| MTP / DFlash2 drafters, keyed sampling, cost-derived depth | DFlash2 (0010 / 0430), 0071 | **adapt**: DSpark replaces DFlash2 (block of N with a sequential Markov head, its own SWA rings fed from the taps); keyed sampling makes drafted == serial as before; the confidence head feeds 0071 |
| Engram | O_DIRECT row reader (0250) | **new**: hashing on the CPU or GPU, 12 rows a token a layer a rank from the kit's packed shards, issued before the verify starts |
| CED | - | **new** as a policy: the forward is the plain layer loop; decoder bounded replay (decoder over the prompt's last 128 tokens) is a prefill knob, off for M1 |
| RoCE all-gather, batching, sessions, prefix share, replay, memory safety, server | 0230 / 0350, 0120 / 0200, 0110 / 0180 / 0250, 0310, 0540, 0550, 0160 ... | **as-is** in mechanism (see DSV41-BASELINE.md section 2) |

### Corrections to DSV41-BASELINE.md / NEXT-DEEPSEEK

1. **DSpark taps** are the streams entering layers 37-39 (after layers 36-38), not "after layers 37-39".
2. **CED's "layer-specific W_KV / W_Z" from H_19** is layer 20's compressor (`wkv` only; ratio 1 has no gate) over
   layer 20's own input. vLLM has no other projection.
3. **Widths**: attention is 5-bit except layer 0's `wq_a` / `wkv` (6-bit); the kit's `exl3_k_map.json` says
   `attn_default 5`. The loader reads every width from the trellis shape, as the kit does.
4. **Non-expert bytes a rank** under the split above are 2.80 GB, not 3.41: `wq_b`, `wo_a`, `wo_b` split in two, only
   `wq_a` / `wkv`, the compressor and the indexer replicated. The hc `fn` (fp32) and router (bf16) are 0.31 GB of
   that, read every step.
5. **The indexer** is replicated in vLLM (no TP split), so at 1M context its key scan is the largest context-dependent
   read (355 MB a step).

## 10. The reference implementation

`engine/reference/` (CPU or CUDA, fp32 math with bf16 storage boundaries):

| module | what |
| --- | --- |
| `config.py` | the config and the CSA2 topology rules (`attention_mode`, `kv_source`, `index_source`, candidates) |
| `exl3.py` | EXL3 decode for every codebook / width: TensorFold v0.6.0's numpy oracle (vendored) + a bit-identical torch port |
| `ops.py` | RMSNorm, clamped SwiGLU, the kit's quantization emulations (`Numerics.kit()` / `.exact()`), linears |
| `rope.py`, `hc.py` | RoPE flavours, mHC |
| `attention.py` | CSA2: projections, compressor, indexer, candidates, sink attention, grouped output |
| `engram.py` | prime layout, hashing, token-map compression, rows, gate |
| `moe.py` | router and experts |
| `model.py` | blocks, layer-major multi-sequence forward (weights streamed and released per layer), DSpark |
| `loader.py` | local or ssh (read-only) safetensors access, the checkpoint namespace -> weights |
| `budget.py` | the byte tables above |

Validation so far (no GPU):

- 76 unit tests on tiny synthetic configs (`tests/reference/`): EXL3 bit-exactness against the oracle for all 27
  codebook x width pairs; every formula above against an independent re-derivation; causality and prefix invariance
  of the whole model with and without the kit's quantization; layer-major batching == one by one; the loader on a
  synthetic EXL3 checkpoint in the real namespace == the directly built model; DSpark's window and Markov bias; the
  oracle script against a fake vLLM server.
- On the real pack (headers + ssh reads, `DSV41_REAL=1`): all 47,900 EXL3 groups well-formed mul1 with the K map's
  widths; every tensor the loader needs present with the expected shape; Engram tables match the prime layout; the
  dequantized `layers.1.engram.wkv` (5-bit) against its original FP8 weights: **3.9% relative error, per-row
  cosine >= 0.9991** (a layout or rotation error would give ~0); blocks 0-2 (SWA, Engram, Full ratio 2 with the
  compressor and indexer) run on a real prompt with finite, well-scaled streams and doubly-stochastic `comb`.
- Not yet: the end-to-end logits against the kit (`tests/reference/oracle_prompt_logprobs.py`, M1's oracle; needs
  a GPU window, see its docstring).
