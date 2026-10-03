# The development log (G1-G12)

The documents written while the DeepSeek-V4.1-Flash family was built and measured, 2026-10-01 to 10-03, kept as
they were except for machine names, link addresses, local paths and a few process words, which were replaced. They
describe the work in the order it happened, so later windows correct earlier plans; the summaries in
[`../RESULTS.md`](../RESULTS.md), [`../DECODE.md`](../DECODE.md) and [`../ARCHITECTURE.md`](../ARCHITECTURE.md) are
the current picture. Raw files: [`../../results/campaign/`](../../results/campaign/). Paths such as
`scripts/windows/*.sh` and `config/prod.env` refer to the development harness, which is not published (it managed
the private test windows and the switch with another production service); `scripts/serve.sh` and
`config/prod.env.example` are its public form.

| document | what |
| --- | --- |
| [LANDSCAPE.md](LANDSCAPE.md) | what others ran and measured on this model (1-4 Sparks and other hardware), quality vs bits |
| [ARCH-LEVERAGE.md](ARCH-LEVERAGE.md) | what DeepSeek built into V4.1 (CED, Engram, CSA2, DSpark, MoE, mHC) and how to use it on two Sparks |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the model layer by layer, the TP=2 split, the byte budget |
| [TARGETS.md](TARGETS.md), [BASELINE-RESULTS.md](BASELINE-RESULTS.md) | the targets and the measured kit baseline |
| [ENGINE-PLAN.md](ENGINE-PLAN.md), [M1-STATUS.md](M1-STATUS.md) | the engine plan and its status through the windows |
| [G1-RESULTS.md](G1-RESULTS.md) ... [G12-RESULTS.md](G12-RESULTS.md) | each test window's results (G12: the host-memory growth, the stall and the segmentation dependence, diagnosed and fixed) |
| [DECODE-ROOFLINE.md](DECODE-ROOFLINE.md), [DRAFT-ACCEPTANCE.md](DRAFT-ACCEPTANCE.md) | the decode roofline and the draft-acceptance study |
| [PARKED-SAGE-1.59.md](PARKED-SAGE-1.59.md) | the evaluated (and parked) 1.59 bpw SAGE pack |
