# G9: where the decode window goes (profiling for prose and 2 streams), 2026-10-02 19:30-

User direction: "We can't go live till we get prose and 2 stream better."

- **Window.** DeepSeek prod was brought down. `campaign.sh open` at 19:30 (it now also stops DeepSeek prod and both
  stacks' timers), deadman 01:30. `the prod-stack marker` = dsv41, so closing the window restores DeepSeek prod.
- **Branch.** `dsv41-060` staged at `3e20a5f`: `5ace28b` + m2bench `--nsys code | prose | c2`.
- **Config.** The final G8 prod config: every `TF_DSV41_*` / `GLM53_TF_*` line of config/prod.env (`prod-knobs.txt`),
  RoCE.
- **Traces.** nsys on **both ranks**: rank 0 inside cudaProfilerStart / Stop, rank 1 for its whole run. Graph nodes
  traced.
- **Tools.** `scripts/windows/nsys_window.py` (new): a window = one graph replay; rows = `_finish_k`'s grid; a DSpark
  pass = a graph with < 20 `_finish_k`. Also `nsys_idle.py`, `nsys_decode.py` and `nsys_skew.py`.
- **Raw files.** `results/G9-20261002/` (`window-<tag>-r<rank>.txt / .json`, `idle-*`, `decode-*`, `skew-*`).

## 1. Prose, 1 stream (`p1`): 37.2 tok/s under nsys, 1.64 DSpark tokens a round

Windows by rows on rank 0 (rank 1 within 0.4 ms on every row count): 98 rounds, 18 / 43 / 25 / 11 / 1 windows at
1 / 2 / 3 / 4 / 6 rows. A round is **43.3 ms** (`nsys_idle.py`): window ~38.9 + DSpark pass 3.5 + GPU idle 2.4 (5.5%).

**The 1-row window (31.0 ms under nsys; 29.5 ms calibrated without it), per class, ms a window:**

| class | ms | detail |
| --- | ---: | --- |
| routed + shared experts (`x3ld` + gate/up epilogue + down/combine + bf16 rot_in) | **9.35** | 7 experts x 40 layers, ~1.6 GB at ~170 GB/s |
| dense EXL3, `linear_kernel` (unfused shapes) | **6.79** | wo_b (grid 40 x 8) **2.60**; wq_b of non-index layers (grid 128) **2.20**; head (grid 505, K6) **1.00**; Engram wkv (grid 200, layers 1 / 14) **0.82**; their rot_in ~0.11 |
| dense EXL3, x3seg groups | **4.69** | o = wo_a x 4 (grid 256) **2.28**; x = wq_a + wkv (grid 112) **1.27** (+ layer 0 0.04); + compressor (144 / 176) 0.21; q = wq_b + indexer wq_b (grid 192) **0.72**; seg_rot_in 0.14 |
| mHC site | 2.18 | 83 a window, 26 us each |
| exchanges (83 RoCE gathers) | 1.72 | **transfer 0.62** (p10 7.5 us each) + **wait 1.10** (the peer later) |
| attention (`_chunks`, `_merge`, `_kv_store`, `_rope`, `_fuse`) | 1.29 | |
| router (gemv) | 1.26 | 40 target + 3 drafter routers |
| copies / elementwise (torch) | 1.03 | |
| torch topk / sort (`gatherTopK`, `radixSortKVInPlace`) | **0.87** | the indexer's top-k on 8 index layers |
| mHC finish | 0.81 | |
| norms (`_rms`, `_pool_norm`) | 0.40 | |
| indexer (`_scores`, `_keys`, `_index_k`, `_plain`, `_block_keys`) | 0.18 | |
| other | 0.14 | |
| **GPU idle inside the window** | **0.30** | |
| **span** | **31.0** | kernels 30.7 |

- **Between windows (rank 0).** Gaps median 1.21 ms. Per round, the GPU is idle 2.39 ms. That is 1.69 in the window
  phase: `forward` 1.42 (graph.replay launch 1.27) and `prefetch` 0.34, `draft` 0.31.
- **DSpark pass (98, one a round).** 3.47 ms: the full head 1.02, drafter dense 0.35, drafter experts 0.96,
  exchanges 0.17 (8), mHC 0.22.

**Rows cost (rank 0 span, ms):**

| rows | 1 | 2 | 3 | 4 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| span | 31.0 | 37.3 | 40.4 | 44.8 | 53.2 |
| experts | 9.35 | 14.34 | 17.84 | 21.76 | 28.43 |
| exchanges (transfer + wait) | 0.62 + 1.10 | 0.63 + 1.57 | 0.64 + 1.23 | 0.65 + 1.35 | 0.68 + 1.40 |
| dense (both kinds) | 11.48 | 11.77 | 12.03 | 12.18 | 12.69 |
| everything else | 9.55 | 10.29 | 9.94 | 10.25 | 11.11 |

**The second-row premium (+6.3 ms, against ~+3.7 for each later row)** is **experts +5.0 ms**: a second token
brings ~5-6 new experts a layer. Later rows overlap more with experts already loaded (+3.5 / +3.9). The rest is the
exchange wait (+0.47) and torch copies (+0.23). Dense is flat (+0.3). So for prose (1.64 tokens a round) the levers
are:
- the expert bytes of row 2 (cheaper experts, or expert-overlap-aware drafting);
- the fixed ~21 ms of a 1-row window that is not experts: dense 11.5, glue 7.6, exchange 1.7;
- the ~6 ms a round outside the target window: DSpark pass 3.5 + idle 2.4.

**Rank skew** (`skew-p1.txt`, 98 aligned windows): busy r0 38.89 / r1 38.87 ms, kernel ratio 1.003. One segment
(11, layer ~5) is consistently rank 0 +73 us. The exchange wait (1.1 ms a 1-row window) is jitter, as in G7.

## 2. Dense x3dn in the engine (`p1attn`, `p1all`): why it is slower

Same prose 1-stream nsys with `TF_DSV41_DENSE=attn` / `all`. Under nsys: prose 37.2 tok/s off, 32.6 attn, 36.0 all.
1-row window span: 31.0 off, 33.2 attn, 33.2 all.

Every x3dn launch has the same grid (144 persistent CTAs = 48 SMs x 3), so its shapes were matched to the baseline's
by position in the window. Both have 160 attention dense launches a 1-row window in the same order, once the indexer
`wk` is set aside. Rank 0, 18 / 31 windows:

| shape (MB a rank) | launches a window | upstream / x3seg us (GB/s) | x3dn us (GB/s) | ms a window |
| --- | ---: | ---: | ---: | --- |
| wo_b (13.1) | 40 | 65.1 (201) | 80.9 (162) | 2.60 -> 3.24 |
| o = wo_a x 4 (10.5) | 40 | 57.0 (184) | 67.6 (155) | 2.28 -> 2.71 |
| wq_b (13.1) | 32 | 68.8 (191) | 80.4 (163) | 2.20 -> 2.57 |
| x = wq_a + wkv (5.7) | 36 | 36.4 (157) | 42.2 (135) | 1.31 -> 1.52 |
| q = wq_b + indexer wq_b (16.4) | 8 | 90.4 (181) | 98.9 (166) | 0.72 -> 0.79 |
| x + compressor | 4 | 43-55 | 52-60 | 0.21 -> 0.23 |
| **attention dense** | **160** | | | **9.32 -> 11.06 (+1.74)** |

- **Per shape, x3dn is 9-24% slower in the engine**, against the design's -15 to -25%. Its own `rot_in` adds
  ~0.33 ms against x3seg's 0.14 + upstream's 0.11.
- **`all` also moves the head and Engram** (grid 505 / 200 upstream, 1.00 + 0.82 ms) into x3dn: 1.35 ms for the K12
  launches, about the same as before. The drafter's 4-bit matrices cost 0.42 ms.
- **The design's input was L2-hot.** The G8 dense test's own timing showed x3seg at 288-443 GB/s on 9 MB (above
  DRAM's ~240), so that test measured an L2-hot replay. There, x3dn's persistent dynamic-atom loop loses to
  upstream's split-K grid even when hot (x4: 21.1 vs 24.5 us at R = 1). In the engine (cold, in graphs) upstream
  reaches 157-201 GB/s and x3dn 135-166.

### 2b. x3dn's schedule knobs on the real window (prose, 1 stream, prod knobs; `ab-dn-*`; none changes a bit)

| config | verify 1 / 2 / 4 / 16 rows ms | prose tok/s |
| --- | --- | ---: |
| DENSE=0 (upstream / x3seg) | **29.3** / 35.6 / 44.0 / 80.0 | **38.2** |
| attn (PD 3, PDL, 144 CTAs) | 30.9 / 37.9 / 46.2 / 83.9 | 35.1 |
| attn, PDL off | 31.1 / 37.9 / 46.3 / 85.0 | 34.7 |
| attn, 48 CTAs | 31.1 / 37.7 / 46.3 / 84.8 | 34.5 |
| attn, 96 CTAs | 30.8 / 38.1 / 46.4 / 83.7 | 34.8 |
| attn, ring depth 4 | 31.0 / 37.4 / 46.2 / 83.3 | 34.6 |
| attn, ring depth 2 | 30.7 / 38.0 / 46.5 / 84.9 | 35.0 |

**No schedule knob closes the gap.** Every x3dn variant is +1.4 to +1.8 ms on a 1-row window, so it is not PDL, not
occupancy / CTA count and not the ring. What differs structurally from upstream is the reduction:

- a strip's K is cut into S = K/256 atoms (wo_b and wq_b: 16; x at K 5,120: 20), against upstream's split-K of 8 (or
  1 for wq_b);
- every atom writes an fp32 partial, takes a strip-counter atomic and the dynamic-schedule atomic;
- the last arriver sums all S partials and runs the epilogue.

At 1-4 rows the weight stream is the same bytes, but there are 2-16x more partial round trips and serialized
last-arriver epilogues a matrix. The design's GB/s targets came from an L2-hot test, where upstream itself reached
288-443 GB/s.

**Not fixable in this window.** The fix is a larger atom: fewer partials, which changes the bits (one constant,
`ATOM`; G8's gate rule allows it), or a split-K-free strip ownership at small R. Both are kernel work for the dense
agent. `TF_DSV41_DENSE` stays 0.

**Implication for the fixed window cost.** The dense EXL3 matrices run at 157-201 GB/s in the engine:

- attention dense 9.3 ms + head 1.0 + Engram 0.8 = 11.5 ms of a 29.5 ms 1-row window;
- at 240 GB/s they would be ~8.9 ms (-2.6 ms);
- the rest of the non-expert time is glue: mHC 3.0, attention / indexer / topk 2.3, router 1.3, copies 1.0,
  norms 0.4, exchange 1.7.

## 3. Two streams, code + prose (`c2`): 51.5 aggregate / 58.0 decode aggregate tok/s under nsys (code 50.0, prose 28.8)

- **Who shares a window.** The two streams share a window only while both decode: code ends its 160 tokens in 44
  rounds (3.64 tokens a round), and prose then runs alone (95 rounds, 1.68 tokens a round).
- **Round.** 93 rounds, **57.5 ms a round**, GPU idle 3.40 ms (5.9%). Of the idle: window 1.77 (forward / graph
  replay launch 1.21), draft 0.58, commit 0.58 (commit wall 1.02 ms), nucleus 0.23 (T = 0.7 sampling of one stream),
  prefetch 0.45.
- **The speculative DSpark pass does not run at C2** (`spec` 0.002 ms a round): it is a one-stream feature.

Windows by rows (rank 0):

| rows | windows | span ms | experts | dense | exchange transfer + wait | other |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 6 | 31.1 | | | 0.62 + 1.41 | |
| 2 | 28 | 36.8 | 14.1 | 11.9 | 0.63 + 1.63 | 8.0 |
| 3 | 16 | 41.1 | | | 0.65 + 1.70 | |
| 7 | 8 | 60.7 | | | 0.77 + 2.29 | |
| 8 | 14 | 66.0 | 39.2 | 13.5 | 0.78 + 2.52 | 10.5 |
| 12 | 10 | 74.5 | 44.2 | 14.5 | 0.81 + 3.43 | 11.6 |
| DSpark pass, 2 slots (10 rows) | 43 | 5.42 | 1.60 | 1.70 (head 1.04) | 0.08 + **1.09** | 1.0 |
| DSpark pass, 1 slot (5 rows) | 51 | 4.34 | | | 0.06 + **0.84** | |

- **The joint window is the cost.** An 8-12-row joint window (66-75 ms) is set by code's depth: prose's ~1.7 tokens
  ride a window built for code's ~4.8 rows. That is 66 ms for prose's 1.7 tokens, against 37 ms alone. Experts are
  59-60% of a joint window (union of ~two streams' experts).
- **Exchange wait grows with rows** (1.4 -> 3.4 ms). At 2 slots the DSpark pass waits 0.8-1.1 ms of its ~4-5 ms in
  only 8 exchanges: the ranks enter the pass out of step (host-side round work differs between ranks at C2: commit
  / nucleus).
- **Levers for 2 streams:**
  - the joint depth policy (cap the code stream's rows while a low-acceptance stream shares the window: the coming
    `joint`);
  - the DSpark pass's rank wait (~1 ms a round);
  - commit / nucleus host time (~0.8 ms of idle a round).
- **Rank skew at C2** (`skew-c2.txt`): busy equal (50.58 / 50.62 ms), but exchanges 3.79 ms on rank 0 vs 2.57 on
  rank 1. Rank 0 waits more, because rank 1 finishes its host work later.

## 4. Code, 1 stream (`c1code`, staged `665a1e1`, prod knobs): 67.3 tok/s under nsys, 3.70 DSpark tokens a round

44 rounds; windows at 1 / 2 / 3 / 4 / 5 / 6 rows: 1 / 4 / 4 / 7 / 9 / 19. A round is **53.8 ms**:

- the window phase 49.6 ms;
- the speculative DSpark pass (3.5 ms, overlapped: `spec` 2.4 ms of wall);
- GPU idle 2.21 ms (4.1%): forward / graph-replay launch 1.30, draft 0.29, candidates 0.28;
- the Engram gate's stall kernel, 1.36 ms a round.

Rank 0, ms a window:

| class | 1 row | 4 rows | 6 rows |
| --- | ---: | ---: | ---: |
| span | 31.9 | 43.3 | **53.5** |
| experts | 9.43 | 20.42 | **28.20** |
| dense linear: wo_b / wq_b / head / Engram wkv | 6.79 (2.62 / 2.19 / 1.01 / 0.81) | 7.33 (2.87 / 2.42 / 1.03 / 0.85) | 7.80 (3.07 / 2.67 / 1.05 / 0.86) |
| dense x3seg: o / x / q + ix / compressor | 4.68 (2.28 / 1.26 / 0.73 / 0.16) | 4.94 (2.42 / 1.26 / 0.82 / 0.17) | 5.12 (2.46 / 1.32 / 0.88 / 0.18) |
| mHC site + finish | 2.98 | 3.12 | 3.22 |
| exchanges (transfer + wait) | 2.90 (0.63 + 2.27, one window) | 1.79 (0.65 + 1.15) | 2.08 (0.67 + 1.41) |
| router | 1.27 | 1.34 | 1.57 |
| attention | 1.34 | 1.35 | 1.47 |
| torch copies / elementwise | 0.58 | 0.73 | 1.51 |
| torch topk / sort | 0.87 | 0.93 | 0.93 |
| norms | 0.40 | 0.40 | 0.40 |
| indexer | 0.18 | 0.27 | 0.34 |
| other | 0.14 | 0.36 | 0.51 |
| GPU idle inside the window | 0.30 | 0.30 | 0.32 |

- **Experts are code's cost.** At 6 rows they are 28.2 of 53.5 ms (53%). Dense (12.9) and glue (~10) are nearly
  flat in rows. Each row past the first adds ~3.8 ms, ~3.1 of it experts.
- **Two-rank alignment was not possible for this trace.** The ranks' windows had different segment counts, so there
  is no `skew-c1code`. The lead asked to skip the skew re-run.

## 5. Joint DSpark depth (`G9.sh joint all`, staged `665a1e1`, prod knobs): not adopted

- **Tests.** 72 passed (joint allocation == brute force, drafted == serial at 2 / 4 slots, joint on == off replies,
  calibration by total rows).
- **Speed.** m2bench code + prose, 1 rep, C1 / C2 / C4, `exact_all True` in all four runs. Per stream: tok/s / rows a
  round / tokens a round.

| config | C1 | C2 decode agg (code / prose) | C4 decode agg |
| --- | --- | --- | ---: |
| off, mixed | 75.56 | **62.37** (55.32 / 31.11) | 87.74 |
| on, mixed | 75.01 | 61.06 (53.95 / 30.45) | 88.27 |
| off, T = 0 | 74.67 | 61.77 (56.08 / 30.80) | 87.50 |
| on, T = 0 | 74.99 | 61.83 (55.96 / 30.83) | 87.99 |

- **Calibration.** The boot calibration now spans 1-64 rows: 29.4 / 35.9 / 44.0 / 52.0 / 60.1 / 78.8 / 110.3 /
  125.8 / 161.0 / 213.9 ms at 1 / 2 / 4 / 6 / 8 / 16 / 24 / 32 / 48 / 64 rows. A further slot's overhead is ~1.0-1.15
  ms (0.40 in one run).
- **Gate: FAIL.** C2 -2.1% and prose-in-C2 -2.1% under mixed sampling (pass lines: -1% / -3%). C4 +0.6%, C1 -0.7%.
  At T = 0 the two are equal.
- **Why.** As its own model predicted, the joint depth trims prose's rows (2.48 -> 2.38) and code's (4.78 -> 4.45)
  at nearly the same tokens a round. The windows do not get cheaper enough to pay for it.
- **Adoption.** `TF_DSV41_DEPTH_JOINT=0` in config/prod.env (the branch default is 1 since `665a1e1`). Replies are
  unaffected either way.
- **Note.** C2 at 62.4 decode aggregate (this window, joint off) is up from G8's 58.9: the same knobs at a later
  stage, with the `5ace28b` -> `665a1e1` commits.

## 6. Dense x3dn v2 (`e961a3c`, staged `24925da`, prod knobs + joint off): not adopted

- **Tests.** 66 passed, 5 skipped.
- **Cold-L2 bench, us, x3seg / upstream -> x3dn** (the test's [G9] lines; the bench and the split sweep agree):

| shape | R = 1 | R = 2 | R = 4 | R = 8 |
| --- | --- | --- | --- | --- |
| x4 (wq_a + wkv + compressor, 9.0 MB) | 61.6 -> 59.9 | 65.6 -> 60.7 | 67.1 -> 62.9 | 67.3 -> 45.6 |
| q (wq_b + indexer, 16.4 MB) | 89.8 -> **106.4** | 92.8 -> 90.6 | 92.5 -> 86.1 | 94.0 -> **105.3** |
| o (wo_a x 4, 10.5 MB) | 67.5 -> 55.4 | 72.8 -> **87.3** | 74.7 -> 79.3 | 80.9 -> 81.8 |
| wo_b (13.1 MB) | 86.7 -> 65.1 | 86.4 -> 89.8 | 89.6 -> 88.3 | 101.8 -> 80.6 |
| drafter wo_b (10.5 MB) | 54.0 -> **77.6** | 56.2 -> 74.6 | 57.7 -> 64.5 | 58.5 -> 45.2 |

- **Split sweep.** Best a shape (`dense-best-split.txt`): 5120x512:8, 5120x1280:8, 4096x5120:2, 4096x1024:4,
  1280x4096:4, 1280x16384:1. Summed over the bench layers, x3dn wins on x (340 vs 442 us) and loses on q (522 vs
  384), q + ix (950 vs 821) and wo_b (1,295 vs 1,067).
- **In the engine** (the calibration, best split):

| | verify 1 / 2 / 4 / 16 rows ms | code / prose tok/s (1 stream) |
| --- | --- | --- |
| off | 30.8 / 35.9 / 44.6 / 81.8 | 74.9 / 37.8 |
| attn | 31.4 / 38.2 / 46.2 / 82.0 | 73.3 / 34.4 |

- **Gate.** Fast kernels 0.9958, exact kernels 0.9958. Gate passes.
- **Verdict: not adopted.** The window line was -1.5 ms; it measured +0.6 ms at 1 row and +2.3 ms at 2 rows, and
  prose -9%. v2 helps some shapes at single rows (o, wo_b, x at R = 1) but loses badly at R = 2 (o +20%) and on q and
  the drafter. Shape-specific choices (v2 only for x, and o / wo_b at R = 1) might net ~0.5 ms: not worth a
  per-row-count kernel switch tonight. `TF_DSV41_DENSE=0`.

## 7. Expert pruning (`TF_DSV41_EXPERT_TOPP / MIN_W / MIN_K`, staged `2905959`, prod knobs + joint off)

- **First run (stage `6e0daab`) failed.** The tests, every gate and every speed run crashed: `grouped(): incompatible
  function arguments`. Cause: the PDL plumbing (`7633509`) did `from . import pdl as P` inside
  `expert_loads.grouped`, rebinding the local pair count `P` to the module. Fixed in `2905959` (`pdl_mod`); the other
  `pdl as P` imports have no such clash. The glue run on that stage was stopped and rerun.
- **Tests (rerun).** 79 passed, 10 skipped, 5 failed. All 5 are `test_a_2048_row_window_runs_in_blocks_under_the_
  group_limit`: a **test bug**, the stand-in `route()` did not take the router's new `keep` keyword (`1f46bc7`).
  Fixed in `38f6500` (test only).
- **Gate and speed.** Gate: exact kernels with prompts pruned. Speed: decode pruning, m2bench code + prose, C1 / C2 /
  C4 mixed. `exact_all True` in every run (pruning applies to drafting and verification alike).

| setting | top-1 / first copy | code C1 | prose C1 | C2 | C4 | speed (pick) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| off | 0.9958 / 0.9601 | 73.91 | 38.18 | 62.47 | 88.16 | 1.000 |
| p90 (top-p 0.90, min-k 3) | 0.9952 / 0.9402 | 72.45 | 38.64 | 64.56 | 89.63 | 1.010 |
| w10 (min weight 0.10, min-k 4) | 0.9949 / 0.9402 | 74.84 | 38.97 | 64.26 | 90.59 | 1.022 |
| p85 (top-p 0.85, min-k 3) | 0.9949 / 0.9336 | 73.69 | 41.92 | 63.02 | 90.17 | 1.031 |
| **p85k** (p85, RENORM=kept) | 0.9944 / 0.9169 | **77.32** | 40.71 | **65.48** | **92.14** | **1.051** |

- **Gate.** Every setting passes top-1 >= 0.97. First-copy agreement falls 0.960 -> 0.917-0.940.
- **MMLU.** For the fastest, p85k: below (queued after glue).
- **MMLU-200 0-shot, p85k** (server, prompts pruned too): **88.5%** (kit 87.5; G7 replay 88.5). **Adopted: p85k**,
  decode only (`TF_DSV41_EXPERT_TOPP=0.85 MIN_K=3 RENORM=kept` in config/prod.env).

## 8. Glue (`G9.sh glue all`, staged `2905959`, prod knobs + joint off)

- **Tests.** Pass: GPU glue / decode glue / router narrow / dtopk / Engram / EXL3 linear suites + interpreter. The
  only failures were the expert_block stand-in (the test bug of section 7, fixed in `38f6500`).
- **dtopk.** At 32K keys: select 244 -> 92 us, reindex top-k 190 -> 74 us. At 75K keys the candidate blocks are
  slower (123 -> 262 us) but reindex is 172 -> 60.
- **Window and speed** (calibration verify ms; prose 1-stream window run; speed run: code / prose T0, C2 decode
  aggregate):

| config | verify 1 / 2 / 4 / 16 rows | prose (window run) | code / prose / C2 (speed run) | gate top-1 / first copy |
| --- | --- | ---: | --- | --- |
| off (every glue knob at its old path) | 29.5 / 36.2 / 44.4 / 79.0 | 37.75 | 71.85 / 37.47 / 61.6 | |
| on (the bit-for-bit set) | 28.7 / 35.5 / 43.7 / 80.3 | 38.54 | 73.55 / 38.31 / 62.4 | 0.9958 / 0.9601 (== off) |
| **fn16** (on + mHC mixing weights bf16) | **28.7 / 35.0** / 43.3 / 80.2 | **39.61** | 73.31 / 39.24 / 61.x | **0.9963** / 0.9502 |
| gsplit (on + Engram gate split) | 28.9 / 35.4 / 43.7 / 79.9 | 38.58 | | |

- **Verdict: adopted, on + fn16.** -0.8 ms at 1 row, -1.2 at 2 rows, prose +4.9%. That is a third of the -2.5 to -3
  ms the glue model expected. The nsys side by side (`glue-nsys.txt`) shows the classes moved less than modelled.
