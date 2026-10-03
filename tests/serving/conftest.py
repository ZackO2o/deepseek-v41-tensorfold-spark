"""CPU tests of the V4.1 serving layer (engine/serving) against a fake forward (``fake_forward.py``).

    TF_SRC=<engine checkout>/src python -m pytest -q tests/serving

Needs torch (CPU), numpy, tokenizers, jinja2 and xgrammar 0.2.8 (structured output); the template / tokenizer tests
read the V4.1 tokenizer files from ``TF_DSV41_TOKENIZER_DIR`` (default ``.cache/dsv41-tok``, copied read-only from
the model folder) and skip without them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, Path(__file__).parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from engine.kernels.tf import tensorfold_src

TF = tensorfold_src()

TEXT = {
    "model_type": "deepseek_v41_text", "vocab_size": 129280, "hidden_size": 5120, "num_hidden_layers": 40,
    "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64, "sliding_window": 128,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "kv_source_layer_ids": [2, 8, 14, 20], "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512, "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048, "candidate_block_size": 8, "engram_layer_ids": [1, 14],
    "num_nextn_predict_layers": 3, "dspark_target_layer_ids": [37, 38, 39],
}
CONFIG = {"model_type": "deepseek_v41", "quantization_config": {"quant_method": "exl3", "codebook": "mul1"},
          "text_config": TEXT}


def tok_dir() -> Path:
    return Path(os.environ.get("TF_DSV41_TOKENIZER_DIR", str(ROOT / ".cache" / "dsv41-tok")))


@pytest.fixture(scope="session", name="topo")
def topo_fixture():
    from engine.serving import topology

    return topology.build(CONFIG)


@pytest.fixture(scope="session", name="model_dir")
def model_dir_fixture():
    d = tok_dir()
    if not (d / "tokenizer.json").is_file():
        pytest.skip(f"no V4.1 tokenizer files in {d} (TF_DSV41_TOKENIZER_DIR)")
    return d


def pytest_configure(config):
    """Temporary files on the home filesystem (O_DIRECT works there; /tmp is a small tmpfs on the workstation)."""

    if not config.option.basetemp:
        config.option.basetemp = str(ROOT / ".cache" / "pytest-serving")
