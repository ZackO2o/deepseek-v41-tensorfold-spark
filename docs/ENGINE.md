# The engine: TensorFold v0.6.0 + two patches

## What ships

| | |
| --- | --- |
| `vendor/TensorFold` | upstream [TensorFold](https://github.com/ashhart/TensorFold) at tag `v0.6.0` (commit `c464617`), unmodified submodule |
| `patches/0001-spark-stack-060.patch` | `families/glm5_next/spark/`: the GLM-5.3-Flash two-Spark engine (TensorFold 0.3.4's GLM CUDA engine with the [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) patch series 0001-0620) rebased onto 0.6.0; the CUDA communicator interface (`cuda/comm.py`); a family `CUDA_SERVE` hook; a server fix for sockets past descriptor 1023. The DeepSeek family reuses its RoCE all-gather, communicator and HTTP server. |
| `patches/0002-deepseek-v41-family.patch` | `families/deepseek_v41/` (MIT) and its tests; a device-side skip in the EXL3 linear (graph-safe), fp64 in the communicator's dtype table, `--kv-dtype fp8`, more model ids in `/v1/models`; NOTICE and THIRD_PARTY_NOTICES entries |

`docker/Dockerfile` applies both with `git apply` on a copy of the submodule and installs the result with pip.

## How the patches were made, and what differs from production

The engine was developed on a private branch: 219 commits on `v0.6.0` up to the first publication (85 for the GLM
stack, 134 for the DeepSeek family), then 9 more (G11's drafter self-distillation tooling and G12's host-memory, stall
and RoPE fixes). Production runs commit `a6f5792` of that branch (it ran `38f6500` until G12). For publication the
branch was squashed into the two commits above and re-authored; the G12 update regenerated `0002` the same way (`0001`
is unchanged). Every engine change production runs is in them, plus `a20be14`, a later test-only commit (two G12
tests made independent of the suite's order).

Applying `patches/` to v0.6.0 and diffing the result against `a6f5792` leaves exactly these differences:

| file | difference | why |
| --- | --- | --- |
| 25 Python files in `families/deepseek_v41/` (14), `families/glm5_next/spark/` (`decode_stream.py`, `l2pf.py`, `roce.py`) and `tests/` (8) | comments and docstrings only (checked: identical syntax trees once docstrings are dropped) | machine names, development paths, link addresses and agent / development-process wording reworded |
| `tests/test_dsv41_long_prefill_memory.py` | `a20be14`'s version | the test-only fix after the production commit (the huge-page test in a fresh interpreter, every tensor group) |
| `families/deepseek_v41/cuda/draft_vocab.txt` | absent | counted from private chat transcripts. Only `TF_DSV41_DRAFT_HEAD=trim` reads it (off, not adopted); `test_dsv41_draft_head.py::test_shipped_ranking` fails without it |
| `families/glm5_next/spark/draft_vocab.txt` | replaced | same origin; replaced by the public-text ranking the GLM recipe publishes (its `patches/0420`, `bench/draftvocab_public.py`). Read only with `GLM53_TF_DRAFT_VOCAB` (GLM engine) |
| `NOTICE`, `THIRD_PARTY_NOTICES.md` | an entry for the DeepSeek family; the deployment names | licensing record |

No production code differs. Against `a20be14` the same list holds without the test row.

## The same tree as a git branch

```bash
git clone https://github.com/ashhart/TensorFold.git && cd TensorFold
git checkout -b deepseek-v41-tensorfold-spark v0.6.0
git am /path/to/deepseek-v41-tensorfold-spark/patches/*.patch
```

A public fork carrying this branch (suggested: `<you>/TensorFold`, branch `deepseek-v41-tensorfold-spark`) would let the
Dockerfile build from it directly (`TF_SRC=<export of the branch> PATCHES=none`); none is published yet.

## Tests

The family's tests are in the patched tree (`tests/test_dsv41_*.py`, `tests/cuda/test_dsv41_*.py`). CPU suites (fake
forward, Triton interpreter, emulators) run anywhere with torch, numpy, safetensors and tokenizers; the CUDA suites
need a GB10 and some the real pack (`TF_DSV41_TEST_MODEL=/model`). In the image, against an export of the patched tree:

```bash
mkdir -p build/tf && git -C vendor/TensorFold archive HEAD | tar -x -C build/tf
(cd build/tf && for p in ../../patches/*.patch; do git apply "$p"; done)
docker run --rm --gpus all -v "$PWD/build/tf:/tf" -w /tf -e PYTHONPATH=/tf/src --entrypoint python \
    dsv41-tensorfold:060 -m pytest -q tests/test_dsv41_serving_app.py tests/test_dsv41_serving_batch.py
```

## Upstream

The communicator interface and the descriptor fix in `0001` were offered upstream separately. The DeepSeek family is
deployment code for two Sparks and this pack; it is not proposed for upstream as is.
