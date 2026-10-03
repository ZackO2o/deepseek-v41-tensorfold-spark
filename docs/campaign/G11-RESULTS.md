# G11: DSpark self-distillation (2026-10-03, 01:48-04:20 +07, inside the G9 window)

- **What it trains.** A separate drafter delta (LoRA + heads) behind `TF_DSV41_DSPARK_DELTA`, default off. Drafts
  never change a reply (exact verification).
- **Setup.** `dsv41-060` staged at `7031b8c`, then `7d940f0` (my dsparkdata fix). Best exact config (config/prod.env
  knobs), `TF_DSV41_DRAFT_K=64`. Raw files: `results/G11-20261003/` (deltas, logs and data stay on head).

## Verdict: **no delta adopted** (pick: none)

| config (exact everywhere) | code T0 / T0.7 (tokens a round) | prose T0 / T0.7 (tokens a round) | structured | C1 | C2 | C4 | draft ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| off (no delta) | **81.5** / 71.2 (3.88) | 41.0 / 41.7 (1.59) | 116.0 | **82.2** | **67.2** | **93.3** | 3.61 |
| delta A (heads + main_proj LoRA, 15 min) | 78.0 / 71.8 (3.69) | **43.3** / 42.8 (1.70) | **117.3** | 78.7 | 66.6 | 93.2 | 3.68 |
| delta B (A + rank-32 LoRA on the 3 blocks, 60 min) | 76.9 / 68.5 (3.84) | 42.3 / 40.3 (1.76) | 112.4 | 77.1 | 62.8 | 89.7 | 4.04 |

- **Gates.** Prose +2% and code / structured / C2 no worse than -1%. A: prose **+5.5%** but code **-4.4%** (its code
  acceptance fell 3.88 -> 3.69 tokens a round). B: prose +3.1%, code -5.7%, C2 -6.4%: its 25 LoRA adapters make the
  pass 0.4 ms longer.
- **Offline.** Held-out 25,320 rows, conditional acceptance a position:

| | 1 | 2 | 3 | 4 | 5 | tokens a round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.521 | 0.423 | 0.375 | 0.344 | 0.319 | 1.862 |
| A | 0.543 | 0.448 | 0.391 | 0.357 | 0.329 | 1.927 |
| B | **0.602** | **0.530** | **0.503** | **0.470** | **0.440** | **2.190** |

- **Offline vs engine.** B's offline +17.6% shows up in the engine as only +11% prose tokens a round (1.59 -> 1.76).
  The port's agreement with the engine's logged drafts is 0.39 overall (0.88 at position 1) at base, so the training
  port and the engine's drafter disagree after position 1. Fixing that port fidelity is the first thing before more
  training.
- **Code regressed.** The captured data is prose-heavy (175 prompts: prose in 4 forms, chat, code, tools), so a
  delta tuned on it trades code acceptance for prose. A per-workload delta (prose-only requests) or code-weighted
  data are the obvious next tries.
- **Where it leaves prose.** Delta A reaches 43.3 tok/s, 1.33x the kit's 32.5, and shows that drafter training is the
  prose lever. It needs more data, a balanced mix, and the port fidelity fixed.

## Run notes (what went wrong tonight)

- **The first `all` run.** The capture succeeded (525 requests, 6.6 GB of logs, 45 min), then `dsparkdata` raised
  IndexError: a session's array did not span its final-pass taps. Fixed in `7d940f0`; `G11-train.sh data` / `resume`
  added (dsv41 `b4ff143`). Merged with G10's log: 695 sessions, 253,612 labelled positions.
- **evalA's first try.** It could not copy the delta to the worker: the trainer writes it as root, mode 0600. Fixed in
  the script (`61daa5b`).
- **My mistakes.**
  - A queued evalA started early: it keyed on a stale `G11-DONE` in the log. Its container cleanup killed the first
    trainB after 2 min, and two benches overlapped for ~20 s. Everything was stopped and re-run cleanly from 03:00.
  - The first clean evals died building CUDA extensions at boot (no prebuild after the 02:39 restage). Re-run after a
    prebuild. All numbers above come from the clean runs.

## G11b: balanced re-capture + delta A (2026-10-03 05:20-06:07): not adopted

- **Capture.** 80 new prompts (56 code in Python / TypeScript / Go / Rust / Bash, 8 fix / SQL / ops tasks, 20 tool
  asks, 4 technical chats) x 4 repeats = 320 requests (`results/G11b-20261003/code-tool-prompts.jsonl`). Merged with
  G10's and G11's logs: 1,015 sessions, 360,573 labelled positions (~30% code / tool, up from ~5%). Stage A (heads,
  15 min, 0.71 epochs).
- **Two script fixes** (dsv41 `1478dde`, the nofile commit): the capture options `CAP_PROMPTS` / `CAP_REPEAT` /
  `EXTRA_LOGS`, and `--ulimit nofile=65536`. The first trainA died of EMFILE: 1,015 memory-mapped sessions.

| | code T0 (tokens a round) | prose T0 (tokens a round) | structured | C2 | C4 | exact |
| --- | --- | --- | ---: | ---: | ---: | --- |
| off | 81.2 (3.88) | 41.3 (1.59) | 116.8 | 66.9 | 93.9 | yes |
| balanced delta A | 78.8 (3.66) | **43.0** (1.65) | 116.9 | 65.7 | 93.1 | yes |

- **Gate.** Prose **+4.1%** (line +2%) but code **-2.9%** (line -1%): **not adopted**. Balancing roughly halved the
  code loss of tonight's delta A (-4.4%) while keeping most of the prose gain.
- **Offline.** Held-out conditional acceptance at position 1: 0.629 -> 0.639. Tokens a round 2.587 -> 2.617.
- **The delta still costs code acceptance (3.88 -> 3.66).** The port's agreement with the engine's logged drafts
  drops from 0.45 to 0.13 after training (first position 0.90 -> 0.75): the trained heads drift from what the engine
  computes. Port fidelity is the blocker before more training, as before.
