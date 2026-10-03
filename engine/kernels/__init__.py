"""sm_121 (GB10) kernels for DeepSeek-V4.1-Flash on TensorFold 0.6.0: EXL3 experts, CSA2 attention / indexer, router.

Staging area for ``tensorfold/families/deepseek_v41/cuda/`` (docs/ENGINE-PLAN.md). Every kernel here keeps the
exactness rules of the GLM Spark engine: a row's bits never depend on the window it runs in (row invariance), sums
run in a fixed order, no atomics. Torch / Triton are imported inside the submodules that need them.
"""

from .tf import tensorfold_src  # noqa: F401
