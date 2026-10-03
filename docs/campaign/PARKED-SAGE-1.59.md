# PARKED: SAGE EXL3 1.59 bpw routed experts (2026-10-02)

User decision: not for this build. Kept here for a possible separate build. Nothing was committed to dsv41-060, and
no GPU time was used. All evidence comes from safetensors headers and sampled tensor hashes, read over read-only ssh
on head (`~/models/DSV4.1-Flash-SAGE-EXL3-1.59bpw`, by vcruz305).

## What the pack is

- 307.7 GiB in total. The two Engram shards (16, 17) are 189 GiB of native FP8, the same tables we already read
  from NVMe. Shards 1-15 hold about 119 GiB.
- **Routed experts are EXL3 with the mul1 codebook.** They use the same names as Mia's pack:
  `layers.N.ffn.experts.E.{w1,w2,w3}.{trellis,suh,svh,mul1}`, all 184,320 names on both sides. The suh, svh and mul1
  dtypes and shapes match, and so does the mul1 scalar. In total they are 101.4 GiB, against 182.6 GiB in Mia's pack
  (x 0.555).
- **Everything else is native: FP8 E4M3 + E8M0 scales.** That covers attention, the shared expert and Engram wkv.
  The MTP / DSpark experts are FP4 (I8 + E8M0). The router is bf16 (ours is fp16, and its hash differs). There is
  also an extra `ffn.gate.bias_vl`. Our engine has no FP8 dense path, so the pack cannot be loaded as it is.
- **Width mix (bits = trellis last dim / 16).** By tensor: K1 27,210, K2 11,811, K3 5,774, K4 1,221, K5 62, K6 2.
  Of the 15,360 experts, 834 have w1 != w3. The average per layer is 1.31 bpw (L38) to 1.95 bpw (L19), with the
  middle layers the heaviest.
- **`-a64` variant.** It has the same tensors, dtypes and shapes as the base pack, and the sampled data hashes are
  identical. It only adds 46,253 `__align_pad__` entries (64-byte alignment).

## Compatibility with our engine

- **"K1" is already in range.** TensorFold's `K2` is half-bits, so 1 bit is K2 = 2:
  - upstream `grouped_kernel` instantiates 2..10 and 2..16;
  - `x3ld`'s switch has `case 2` and `RANGES = ((8,8),(2,10))`;
  - `linear_kernel` has a dedicated 1 / 2-bit shuffled load path.

  So no new decode kernel is needed. Open items:
  - a CPU lane-emulator and GPU `Z(ld) == Z(grouped)` test at K2 = 2 (a step of NT x 8 = 64 words, so only half the
    lanes issue v4 loads);
  - adding `(2,16)` to x3ld's `RANGES`, because K6 (K2 = 12) on layers 1 and 15 would otherwise fall back to
    upstream;
  - prefill fallbacks. `x3gm.K2S = (4,6,8)` needs gate width == up width, and `x3tc` needs one width for all three
    matrices. So K1 / K5 experts and the 834 mixed-width experts take upstream's grouped kernel in prefill, which is
    slower for TTFT but not wrong.
- **Graft, not load.** The workable build is Mia's pack with only the routed-expert tensors redirected to SAGE shards:
  - an overlay folder with symlinks and a merged `index.json`;
  - `experts.plan_layer` already reads a width per table entry.

  The sampled hashes show that the dealignai abliteration did not touch routed experts. mialab and uncensored are
  identical on experts at L0 / 10 / 20 / 30 / 39. It changed `shared_experts.w2` and `attn.wo_b` in the middle
  layers. So a graft keeps the abliteration.
- **TP2 per rank.** Routed experts drop from ~91.3 to ~50.7 GiB, which frees about 40 GiB a rank for KV.
- **TP1 on one Spark does not fit at a useful context.** The breakdown:
  - SAGE experts 101.4 GiB;
  - Mia's non-routed weights about 13.1 GiB: attention 3.25, embed 1.23, shared 0.75, head 0.46, MTP 6.75, router
    and other 0.45, Engram wkv 0.17;
  - total about 114.5 GiB of weights, against roughly 119-121 GiB of GB10 memory that CUDA can use.

  That leaves no room for KV plus the 4-6 GiB floor. Dropping DSpark's 6.3 GiB makes it fit, but it removes
  speculation.

## Predicted speed: graft at TP2, from G9's per-class table, an untested model

The routed share is about 8.0 of the 9.35 ms of experts in a 1-row window, scaled by 0.555:

| case | today | predicted |
| --- | --- | --- |
| 1-row window | 29.5 ms | ~26 ms |
| 2nd-row premium | +6.3 ms | ~+4.1 ms |
| prose 1 stream | 38 tok/s | ~43-44 (+13-15%) |
| code 1 stream, 6-row window | 53.5 ms | ~42 ms (+20-25%) |
| C2 joint windows (experts 59-60% of the window) | | +25-30% |

Two risks would shrink these gains:
- SAGE probably gives more bits to frequently routed or sensitive experts, so the bytes actually read per token
  could be above 0.555 of today's;
- at 1 bit, `decode_tile` costs the same ALU work a value on half the bytes, so x3ld may stop being
  bandwidth-bound.

**Two TP1 replicas, one per Spark: no.** They do not fit (see above). Even with DSpark dropped, every token would read
all of the dense weights (~3.8 GB with no TP split) from one GPU, against ~1.9 GB a rank today.

## Predicted quality: likely fails our gate

- The README reports top-1 vs the FP4 release of 91.2%, KL 0.19. By subset: general 82.0%, code 97.9%, math 91.7%,
  reasoning 93.2%. The 3.30 bpw sibling gets 94.0%.
- Our gate is top-1 vs the kit (Mia's 2.9 pack) at >= 96%. The kit is itself only about 94% vs FP4 (by analogy with
  the 3.30 sibling), and the two packs' errors are mostly independent.
- **Estimate vs the kit: ~88-92% top-1**, and lowest on prose, the workload we wanted to speed up.
- **MMLU** is likely more than 1 point below 87.5.

## Hybrids considered

- **SAGE experts for prose verify rows only.** Verification is exact against whichever experts it uses, so this
  would change the output distribution for prose, the subset where SAGE is weakest (82% top-1 vs FP4). Not
  recommended.
- **SAGE experts for the DSpark drafter or as a self-draft.** This is lossless, because the target verifies with
  Mia's experts. But it holds both expert sets in memory (+~25 GiB a rank at TP2, which fits), and DSpark already has
  its own 128 experts. The gain depends on acceptance and is unmeasured.

## If revived as a separate build

1. Build the graft folder, then run the K2 = 2 emulator and GPU Z tests.
2. Run the M1 top-1 gate vs the kit oracle, and MMLU-200.
3. Measure the speed table (code / prose / structured at C1 / C2 / C4).

Estimated at about 2.5-3 h of a two-Spark window. Go only if the top-1 gate passes. On the current evidence, it is
expected not to.
