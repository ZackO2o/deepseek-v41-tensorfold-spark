"""The TensorFold family (``tensorfold.families.deepseek_v41`` on the branch dsv41-060) against engine/reference.

Run from the repo root with the work tree's sources (engine/kernels/tf.py finds ``<engine checkout>/src``):

    TF_SRC=<engine checkout>/src python -m pytest -q tests/family
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, Path(__file__).parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from engine.kernels.tf import tensorfold_src  # noqa: E402

TF = tensorfold_src()
if TF is not None and str(TF.parent / "tests") not in sys.path:
    sys.path.insert(0, str(TF.parent / "tests"))      # dsv41_fakes.py: the synthetic EXL3 checkpoint


def pytest_collection_modifyitems(config, items):
    if TF is None or not (TF / "tensorfold" / "families" / "deepseek_v41").is_dir():
        skip = pytest.mark.skip(reason="the TensorFold work tree with families/deepseek_v41 (TF_SRC)")
        for it in items:
            if "family" in str(it.path):
                it.add_marker(skip)
