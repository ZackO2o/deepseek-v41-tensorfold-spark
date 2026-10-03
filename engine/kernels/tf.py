"""Where TensorFold 0.6.0 comes from: the installed package, or a source tree named by ``TF_SRC`` (its ``src`` dir).

The kernels build on the branch ``glm-spark-stack-060`` (TensorFold 0.6.0 + our GLM Spark engine). Upstream's
``tensorfold.cuda.exl3`` is used as is (no copy): ``experts_grouped.cuh`` for the device code, ``experts.py`` /
``format.py`` for the host side.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_TREES = (Path.home() / "<engine checkout>/src", Path.home() / "<upstream 0.6.0 checkout>/src")


def tensorfold_src() -> Path | None:
    """The ``src`` directory holding ``tensorfold`` (added to sys.path if needed), or None when nothing is found."""

    env = os.environ.get("TF_SRC", "").strip()
    cands = [Path(env)] if env else []
    try:
        import tensorfold  # noqa: F401

        return Path(sys.modules["tensorfold"].__file__).resolve().parents[1]
    except ImportError:
        pass
    for c in [*cands, *DEFAULT_TREES]:
        if (c / "tensorfold" / "cuda" / "exl3" / "experts_grouped.cuh").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    return None


def exl3_dir() -> Path:
    """Upstream's ``tensorfold/cuda/exl3`` (the include directory of ``x3ld.cu``)."""

    src = tensorfold_src()
    if src is None:
        raise ImportError("TensorFold 0.6.0 not found: pip install it or set TF_SRC=<tree>/src")
    return src / "tensorfold" / "cuda" / "exl3"
