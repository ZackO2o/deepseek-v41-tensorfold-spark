# DSpark draft acceptance on prose (overnight research track, 2026-10-02)

Goal: more accepted DSpark tokens a round on prose and in multi-stream decode, without changing a reply (verification
stays exact). Work was offline (no GPU). The GPU steps are in `scripts/windows/G10-draft.sh` and have **not run**.
Engine code is on `dsv41-060` behind knobs, all default off. The tok/s figures below use G9's cost table: verify
29.5 / 36.3 / 40.25 / 44.2 / 48.15 ms for 1-5 rows, plus 5.9 ms a round for the DSpark pass and GPU idle.

## 1. Summary

| # | Approach | Status | Prose tokens a round | Prose tok/s (G9 costs) | Confidence |
| --- | --- | --- | --- | --- | --- |
| 1 | **Re-align the drafter to our target.** Precision (the native FP8 / MXFP4 DSpark is on head in the SAGE pack) and/or self-distilling the blocks | planned; capture hook and replay inputs built | 1.65 -> 1.82-2.02 if position 1 goes from ~0.49 (pooled) to 0.565-0.63 | **+7 to +15%** | medium: the gap is measured against others' native-weight runs, not ours |
| 2 | **Parent-conditioned trees** (PCTree-style: branches at positions 1-3, scored by the Markov head given each parent, with an expert-union cost) | **built**: `TF_DSV41_TREE_PC=1`, tests pass | +6 to +26% | **+0.2 to +5%** at today's row costs; +3 to +13% if rows get 25-50% cheaper or sibling rows share experts | decided by one unmeasured number, s2 (section 4) |
| 3 | Self-distil the heads only (Markov delta + confidence head, folded into the checkpoint's tables, no kernel change) | **built**: `drafttrain.py`, `TF_DSV41_DSPARK_HEADS` | +1 to +4% (guess) | +1 to +3% | low |
| 4 | Confidence-head temperature scaling (sequential, a temperature for each position) | **built**: `TF_DSV41_CONF_CAL=temp` | ~0 | 0 to +1% (ECE 4.9 -> 4.0% on q_1) | medium |
| 5 | Markov-bias scale / logit temperature | offline sweep built (`draftsim`) | unknown, probably ~0 | ~0 | needs the log |
| 6 | Phrase table / n-grams (from the prompt or a background corpus) | measured offline: **no** | ~0 | ~0 | high |
| 7 | "Self-draft" from the target's top-2 at the last verified row | **no**, by construction | 0 | 0 | high |
| 8 | SGLang #40513 (Markov over the backbone top-K ∪ bias top-M) | not applicable: we already bias only the top-128 candidates | 0 | 0 | the log measures misses |

**What merits GPU time tonight:**
- `G10-draft.sh tests` (~40 min) and `capture` (~30 min). Every estimate above then becomes a measurement from one
  log: s2, near ties, the Markov / n-gram / temperature sweeps, heads-training data, and the taps needed for an
  offline precision A/B.
- `speed` for `off pc2 pc2w3` (~1 h), only if `analyze` shows s2 >= 0.35.
- No multi-hour training without the lead's go-ahead. The heads-only fit takes minutes, but it is still
  gated (`TRAIN_OK=1`).

## 2. Where drafts fail (G8 / G9 m2bench JSON, `results/G8-20261002`, `G9-20261002`)

**Per position.** The prose rows below are G8 `m2-ab-g8on`, T = 0. Conditional = kept / reached, for drafts the
depth chose to verify.

| workload | reached (1..5) | kept | conditional 1 / 2 / 3 / 4 / 5 |
| --- | --- | --- | --- |
| prose T0 | 204 / 65 / 8 / 1 / 0 | 118 / 29 / 5 / 0 / 0 | **0.58 / 0.45** / 0.63 / - / - |
| prose T0.7 | 178 / 69 / 16 / 2 / 0 | 105 / 39 / 11 / 0 / 0 | 0.59 / 0.57 / 0.69 / - / - |
| code T0 | 99 / 84 / 65 / 46 / 35 | 91 / 68 / 52 / 41 / 30 | 0.92 / 0.81 / 0.80 / 0.89 / 0.86 |
| structured | 65 / 65 / 64 / 63 / 63 | 65 / 65 / 63 / 63 / 63 | ~1.0 |

- **Prose stops after position 1-2.** Prose verifies 1.4 drafts a round and keeps 0.64. Position 1 is unconditional
  0.54 once the zero-depth counterfactuals are counted (125 / 231).
- **Native weights do better.** LANDSCAPE section 4 has native-weight prose at 0.63 / 0.33 / 0.15 / 0.06 / 0.02
  cumulative, which is conditional 0.63 / 0.52 / 0.45 / 0.40 / 0.33. Ours is lower at both positions measured.
- **The keyed noise works.** T0.7 accepts at least as well as T0.

**By the drafter's own confidence.** These are q_1 bins pooled over the 15 G8 / G9 prose runs: 6,602 rounds, the
calibrated q_1 of `depth.Stats`.

| calibrated q_1 | .0-.1 | .1-.2 | .2-.3 | .3-.4 | .4-.5 | .5-.6 | .6-.7 | .7-.8 | .8-.9 | .9-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rounds | 36 | 614 | 1073 | 1285 | 886 | 698 | 566 | 471 | 457 | 516 |
| first draft kept | .17 | .16 | .18 | .30 | .51 | .54 | .70 | .87 | .88 | .93 |

- **Prose failures are uncertain drafts, not confident ones.** 68% of first-draft rejections come from rounds where
  the drafter said q_1 < 0.4 (46% of rounds). Only 4.4% come from q_1 >= 0.7.
- The draft distribution is wide where the text is wide, so a better drafter has to sharpen it, not re-rank a
  confident mistake.
- Calibration is decent. A temperature fit has slope 1.15 (the factored q is slightly too flat), and ECE goes from
  4.9% to 4.0%. That fit is now `TF_DSV41_CONF_CAL=temp`.

**Near ties vs genuinely wrong (target top-2), and the draft's 2nd choice.** The JSON has no per-round target logits
or draft distributions, so neither question can be answered from G8 / G9. The new log
(`TF_DSV41_DRAFT_LOG`, section 5) records both, and `draftsim.near_ties` / `draftsim.ranks` compute them.

- **Target's 2nd choice equals the draft.** At T = 0 nothing can be bought from this: the greedy choice is exact.
  At T > 0 the keyed draft noise already shares the target's uniforms (T0.7 >= T0 above). The log gives the rate and
  the logit margin, for the record.
- **Draft's 2nd choice equals the target's.** This is s2, the number that prices width (section 4).

**Offline checks that need no GPU** (the prose replies with the Markov head and tokenizer streamed read-only from
head):
- **The Markov head is a residual, not a language model.** On its own (argmax of w2 · w1[prev] over the vocabulary)
  it predicts the next prose token 2.9% of the time (code 9.9%, structured 0%). It only makes sense on top of the
  blocks' base logits, so there is no cheap Markov-only drafting trick.
- **In-reply n-grams do nothing for prose.** Prompt / suffix lookup is correct on 0% of positions (code: 6-9%).
- **A background corpus does little better.** We built a trigram + bigram table from WikiText-103 (57.6M tokens,
  DSV4 tokenizer). On our prose replies its top-1 is right 13.6% of the time. Its confident predictions (share >= 0.5)
  cover 16% of positions at 29% precision. That is far below DSpark's own 0.45-0.58, and the hits are mostly inside
  words (`keros|ene`, `windswe|pt`), which the Markov head already covers. **A phrase table is not worth building
  for prose.**
- **Self-draft from the target's top-2 at the last verified row adds nothing.** That row gives one distribution,
  for the next token, and its keyed choice is the bonus token, which is already committed exactly. Nothing predicts
  the token after it, and the top-2 alternative at that row can never be kept. Value 0.

## 3. The tree, recomputed (lead's leads: PCTree 2608.02123, Baidu 2609.24698, SGLang #40513)

**What changed from G8's tree.** G8's `tree.py` branches only at position 1, with marginal sibling probabilities.

- **`pctree.py`** (`TF_DSV41_TREE_PC=1`) expands best-first over the pass's whole draft distribution.
- **Children.** A node's children are the keyed order of z_i(· | parent) = base_i + w2 · w1[parent], so siblings
  share the backbone logits and only the Markov bias follows the parent (PCTree). Branches are allowed at depths
  1 .. `TF_DSV41_TREE_DEPTHS` (default 3), with `TF_DSV41_TREE_WIDTH` children (default 2), at most B chains.
- **Values.** A keyed first child takes the confidence head's value given its own parent. The head is
  sigmoid(cw_h · h + cw_m · w1[t]), so the parent term is swapped on the host. That value then goes through the
  depth's calibration. Later children take their draft probability times a decayed per-(depth, rank) factor,
  learned from the serial tokens.
- **Expert-union cost.** Unique rows are priced on the measured verify table, whose row-2 premium is G9's +5 ms of
  new experts. A chain re-runs its shared prefix on a shadow: same tokens, same routing, no new experts, only
  `dup_ms` (0.4). Each extra chain also pays `Costs.slot`. `TF_DSV41_TREE_SIB_EXPERT` scales the expert cost of
  off-chain rows, for when siblings turn out to share experts.
- **Verification.** Each chain is a chain window (exactness unchanged). The window presents the chain with the
  longest accepted prefix. The main chain's own accepted length still feeds the depth calibration.

**Expectations** (`treesim.synth`, 4,000 synthetic rounds):
- The draft distributions match G8 prose: the q_1 histogram and conditional 0.45 / 0.40 / 0.35 / 0.30 after
  position 1. The chain baseline reproduces prose (1.65 tokens a round, 38.2-38.6 tok/s).
- The unknown is **s2 = P(serial token = the draft's 2nd | not its 1st)**.
- Each cell is prose tok/s against the chain, with tokens a round in parentheses:

| s2 | row costs | tree d1 (G8) | **pctree d2 w2 B3** | pctree d3 w2 B4 | pctree d2 w3 B4 |
| --- | --- | --- | --- | --- | --- |
| 0.25 | G9 | +0.1% (1.67) | +0.2% (1.68) | +0.2% | +0.2% |
| 0.35 | G9 | +1.0% (1.83) | +1.1% (1.86) | +1.1% | +1.0% (1.87) |
| 0.45 | G9 | +3.9% (1.98) | +4.7% (2.03) | +4.7% | +5.3% (2.11) |
| 0.35 | G9, sibling rows at 0.6 of a row's experts | +4.2% | +4.7% | +4.9% | +5.2% (2.07) |
| 0.35 | 25% cheaper rows | +3.0% | +3.4% | +3.5% | +3.3% |
| 0.35 | 50% cheaper rows (pruning / dense) | +5.3% | +5.9% | +6.1% | +6.8% (2.14) |
| 0.45 | 50% cheaper rows | +9.7% | +11.5% | +11.7% | +13.5% (2.37) |

- **Why literature gains do not transfer.** Acceptance rises like the literature (+12 to +26%, against PCTree's
  +26-32% and Baidu's +17-19%). Tok/s does not, because a verify row on GB10 is a full routed-expert read, ~4-6 ms
  against a 43 ms round. At λ = 0.038 tok/ms, a sibling row must add >= 0.15 expected tokens to pay. With
  q_1 ≈ 0.5 that needs s2 >= ~0.3.
- **Width or depth?** The planner already spends rows where they pay. On prose it picks width at positions 1-2 over
  depth 3+, which matches the collapse after positions 1-2. When the drafter gets better (section 1, row 1) the tree
  matters less: at position-1 0.63, PC trees add +0.1%.
- **Multi-stream.** Trees stay per slot and are off when a grammar mask or nucleus row is in the round. In a joint
  C2 window a row costs its marginal ~3.5-4 ms at a higher aggregate λ, so trees pay even less there. Acceptance
  gains (row 1) help C2 directly: prose's rows ride code's window.
- **SGLang #40513.** Our chain already biases only the gathered top-K (64 a rank, 128 total) candidates.
  `draftsim.ranks` counts how often the serial token falls outside them. Only a non-zero count would make a Markov
  top-M union worth building.

## 4. Training plan (self-distillation on our own target)

**Why train at all.**
- Our target is the dealigned pack at 2.9 bpw with a 4-bit EXL3 drafter (`mtp_bits 4`). DSpark was trained against
  DeepSeek's FP8 / FP4 model.
- sfxnz measured 1.33 vs 2.77 tokens a round for 4-bit vs source-precision DSpark. The evidence is contested:
  coolbho3k saw 53.2 vs 54.2%.
- Native-weight prose is 0.63 at position 1 against our ~0.49-0.54.
- On G9 costs, **each +0.05 of position-1 acceptance (later positions scaled alike) is worth ~+4-5% prose tok/s**:
  - pooled 0.485 -> 1.65 tokens a round, 38.2 tok/s;
  - 0.565 -> 1.82 tokens, 40.9 tok/s (+7%);
  - 0.58 -> 1.87 tokens, 41.7 tok/s (+9%);
  - 0.63 -> 2.02 tokens, 43.8 tok/s (+15%).

**Literature** (as reported; not re-measured):
- EAGLE-3 is ~1.4x over EAGLE-2, through training-time test and multi-layer features.
- HASS gives +8-16% acceptance length over EAGLE-2 (harmonized context alignment).
- DistillSpec's on-policy KD is worth 10-45% speed over plain speculative decoding.
- DSpark's paper reports sequential temperature scaling cutting calibration error from 3-8% to ~1%.
- All of these train the drafter body on 10^7-10^8 target-generated tokens. Heads-only fits (Medusa-1 style) gain
  much less.

**Phase 0 (no training, measure first): a precision A/B, offline.**
- The SAGE 1.59 bpw pack on head (`~/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw`, shards 14-15) holds DSpark at
  **native precision**: FP8 E4M3 attention and main_proj, MXFP4 experts as I8 + E8M0 scales, BF16 router.
- With `TF_DSV41_DRAFT_LOG_TAPS=1` the log records the drafter's exact input (30 KB a token).
- `engine/reference`'s DSpark can replay each logged pass with the 4-bit EXL3 weights and with the native ones, and
  score both against the same serial tokens. This needs a native-weight reader for the reference: FP8 dequant exists
  (`ops.fp8_e4m3_dequant`), MXFP4 does not yet.
- If native wins by >= 0.05 at position 1, serving it costs memory. Re-quantizing at 6 bpw is +~1.6 GiB a rank,
  against a worker floor of 4.42 GiB, so trimmed experts or NVFP4 would be needed. Decide after the A/B.

**Phase 1 (built): heads only, from the log, no model loaded.**
- Script: `drafttrain.py`.
- **Model.** Markov interaction w2' · (w1' [prev]), with w1' = (w1 + w1 M) · α and w2' = w2 + w2 Q (M, Q 256 x 256).
  That is 131K parameters, folded into new `markov_head.embed / .head` tables. Then a logistic refit of the
  confidence head on [h ; w1'[prev]].
- **Loss.** Cross-entropy over the row's candidates against the serial token (T = 0 rows: the target's own greedy
  choice), teacher-forced parent, held-out sessions.
- **Deploy.** `TF_DSV41_DSPARK_HEADS=heads.safetensors` (no kernel change; both ranks load the file).
- **Option `--ctx r`.** Adds a hidden-conditioned term u_i = A B rms(h_i) to the Markov input. It is not foldable
  (it needs one vector add in the chain kernel), so it is trained only to price that kernel work.
- **Data.** 20-60K passes (100-300K labelled rows) is a 30-60 min C4 capture, 1-3 GB with hidden rows.
- **Time.** Minutes on a GB10 (a 128 x 256 bilinear a row).
- **Risk.** Low: drafts only propose, held-out gating, the knob is off by default. The expected gain is small
  because the blocks, not the heads, make the distribution.

**Phase 2 (plan, not built): LoRA on the 3 DSpark blocks.**
- **Adapters.** Attention projections and the shared expert, r = 16, ~3-6M parameters. The 128-expert banks stay
  frozen at 4 bits.
- **Training.** The reference DSpark in PyTorch, with experts dequantized on the fly, trained on captured
  `main_x` / taps.
- **Labels.** The target's top-K logits at every position: a teacher-forced prefill of target-generated text gives
  them for every token, not just verified rows.
- **Data.** 2-5M target-generated tokens: ~200-500 prose / chat prompts x 384 tokens per 1M tokens, generated at C4
  ~80 tok/s, so ~3.5 h a 1M tokens. Taps are 30 KB a token (main_x after main_proj: 10 KB), so 20-50 GB on head's
  NVMe.
- **Training time.** Our estimate is ~0.4G active parameters a row and ~15 GFLOP a 5-row sample with its context.
  At a realistic 20-30 TFLOPS, 1M samples is roughly 1-3 h on one GB10. The GPU must be free of prod: it needs a
  window.
- **Risk.** Medium-high engineering (2-3 days):
  - a differentiable EXL3 path, or dequantized bf16 for only the active experts;
  - matching the engine's numerics on the frozen parts;
  - the train / serve gap of the 4-bit experts.
- **Gain.** Closing half to all of the gap to native-weight acceptance is +7 to +15% prose tok/s (the table above).

## 5. What was built (engine `dsv41-060`, all default off)

| file | what |
| --- | --- |
| `cuda/pctree.py` (new) | parent-conditioned tree: `Expander` (children and the confidence head for any parent), `grow` (best-first, at most B chains), `select` (surplus with the expert-union cost), `chains_of`, `learn`, `Cal`. Knobs `TF_DSV41_TREE_PC`, `_TREE_DEPTHS`, `_TREE_WIDTH`, `_TREE_SIB_EXPERT` |
| `cuda/tree.py` | the PC path (`_pc_draft`, `_pc_resolve`: the longest accepted chain, main keep for calibration); G8's first-position path unchanged when PC is off |
| `cuda/depth.py` | `choose_pc`; calibration from `confcal.make` |
| `cuda/confcal.py` (new) | sequential temperature / Platt scaling of the confidence head, online or fixed (`TF_DSV41_CONF_CAL`, `TF_DSV41_CONF_TEMP`) |
| `cuda/draftlog.py` (new) | `TF_DSV41_DRAFT_LOG=<dir>`: per pass the drafts, raw confidences, candidates and base logits [5, 128]. Optionally head hidden (`_HID`) and taps (`_TAPS`). The target's top-K a verified row (`_TOPK`, 8), the chain presented, the bonus. Rank 0 only; drafts and acceptance unchanged |
| `cuda/drafter.py` | keeps the hidden for the log; `on_ingest` taps hook; `heads_override` (`TF_DSV41_DSPARK_HEADS`) |
| `cuda/decode.py` | log hooks (reset, draft, window, commit, bonus) |
| `cuda/draftsim.py` (new) | offline analysis of a log: positions, ranks / recall@m, near ties, Markov and n-gram sweeps, temperature fit; CLI |
| `cuda/treesim.py` (new) | round simulator with the engine's planners and calibrations: `replay` (logs) and `synth` (the tables above) |
| `cuda/drafttrain.py` (new) | phase-1 heads self-distillation, export, CLI |
| `cuda/draftcap.py` (new) | GPU capture driver: 184 built-in prompts or JSONL, C4, logs on |
| tests | `test_dsv41_pctree.py`, `test_dsv41_pctree_e2e.py` (drafted == serial with oracle depth-1 / depth-2 branches at T = 0 / T > 0 on the CPU twin, and the log read back; the log changes nothing), `test_dsv41_drafttrain.py`, `tests/cuda/test_dsv41_pctree_gpu.py` (GPU, row + pass graphs; not run) |

GPU steps: `scripts/windows/G10-draft.sh tests | capture | analyze | speed | pick | train (TRAIN_OK=1) | all`.

## 6. G11: self-distillation of the drafter (lead GO, 2026-10-03 00:45 +07)

**What G10 measured** (`results/G10-20261002`: 34,753 passes, `analyze.txt`):
- conditional acceptance by position: 0.506 / 0.415 / 0.359 / 0.349 / 0.385;
- s2 = 0.174, so trees are dead (treesim: below +0.2%);
- 28% of first-position serial tokens (49% at position 2) are **not among the 128 base candidates at all**. At
  rejections the target's top-1 margin has a median of 2.5 logits.

The drafter is wrong, not near: its weights are the lever.

**Pipeline** (engine `dsv41-060`; GPU steps in `scripts/windows/G11-train.sh`):

| piece | what |
| --- | --- |
| `draftlog` + `batch.py` | with `TF_DSV41_DRAFT_LOG`, every verify row fetches the target's top-`TOPK` (32) candidates. The candidates are sorted, and greedy / top-k choices read the same first columns, so replies are unchanged (tested). `_TAPS=1` logs the drafter's input rows |
| `draftcap.py` | C4 capture: 175 prompts (prose in 4 forms x 40 topics, chat, code, tool calls) x `--repeat` (repeats at T = 0.7), `--max`, `--minutes` time box, a new seed for each sampled request |
| `dsparkdata.py` | logs -> per-session arrays: tokens, taps, the target's top-K of each committed position (from the row that chose it), and the logged drafts. G10's log (top-8) is added |
| `dsparktrain.py` | the drafter in PyTorch, block-parallel as DSpark drafts, from the pack (EXL3 decoded once to frozen bf16, ~32 GB). **Stage A**: Markov / confidence heads, main / final norms, LoRA on main_proj. **Stage B**: + LoRA (rank 32) on every block's wq_a / wkv / wq_b / wo_a / wo_b and its norms. Loss: forward KL from the target's top-K distribution, a weight 0.8^i by position, + confidence BCE. Held-out sessions: teacher-forced top-1, the engine's T = 0 chain (top-128 candidates + Markov), and agreement with the engine's logged drafts |
| `dsparkdelta.py` | `TF_DSV41_DSPARK_DELTA=<file>` (default off): replacement vectors / tables copied in place, LoRA wrappers split like their matrices under TP (wq_b by output rows, wo_b by input columns, wo_a by groups). The pack is never modified |

**Tests** (CPU twin, synthetic checkpoint):
- the port reproduces the engine's base logits (correlation 0.9998);
- full-chain agreement with the logged drafts is 0.67 on random weights, where many rows are near ties;
- stage B lowers the loss;
- the exported delta loads in the engine, and replies with it == serial;
- LoRA shards sum to the whole delta.

**Not covered:**
- **Shared experts.** On the GPU the shared expert rides in the routed launch, so a LoRA there needs kernel work.
- **The 4-bit routed experts** stay frozen.
- **LK-loss** (2602.23881) is not implemented: KL is the stand-in.

**Data and time.**
- ~90 min of C4 capture gives ~300-400K target tokens, not 2M: decode is the bottleneck at ~80 tok/s aggregate.
- 2M tokens would need a teacher-forced prefill capture (the engine computing every row's top-K and taps at
  prefill speed). That is the next step if stage B shows a held-out gain.
- Expect stage A +0-3% prose. Stage B is unknown, with a real risk of overfitting at this data size: the held-out
  chain metric decides before any engine run is trusted.

**The 07:30 time box** (`G11-train.sh all`): tests 15 min, capture 90 + data 10, trainA ~30, evalA 20, trainB
(shrinks to the deadline, at most 70 min of training), evalB 20, speed (off) 20, pick.
