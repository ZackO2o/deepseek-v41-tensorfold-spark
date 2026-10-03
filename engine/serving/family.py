"""``tensorfold/families/deepseek_v41/__init__.py`` as it will read on our 0.6.0 branch (``glm-spark-stack-060``):
what TensorFold's family discovery and CUDA CLI need (upstream's ``docs/recipes/adding-a-cuda-family.md`` contract;
the GLM Spark family's ``CUDA_SERVE`` hook from our PR 1, commit ``48886c5``).

Importable without torch (family discovery runs on Macs too). The engine and the app load lazily.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v41",)
TITLE = "DeepSeek-V4.1-Flash"
MODELS = ("dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw", "MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2.9bpw")
QUANT_METHODS = ("exl3",)
CUDA_QUANTIZATION = ("exl3",)
CUDA_KV_DTYPES = ("fp8",)            # the 584-byte FP8 row (engine/kernels/csa2/rows.py); fp4 (the QAT format) later
DRAFTER = "dspark"                    # built in: the checkpoint's mtp.* blocks
REQUIRED_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json")
ENGINES = ("spark",)

# Engine switches (TF_<FAMILY>_* as upstream names them); the CLI's --context / --kv-dtype / --parallel apply.
KNOBS = {
    "TF_DSV41_PREFILL": "full | replay: CED decoder bounded replay for prompts (approximate by design; own tag)",
    "TF_DSV41_ENGRAM_DIR": "the packed per-rank Engram shards on local NVMe (engram-l{1,14}-r{rank}of2.bin)",
    "TF_DSV41_ENGRAM_CACHE_ROWS": "RAM row cache (0 = off): only from memory above the floor",
    "TF_DSV41_EXPERT_LOADS": "the 0580 load path for the grouped mul1 experts (engine/kernels/exl3/loads.py)",
    "TF_DSV41_INDEX_KV": "bf16 | fp8 index keys in the pool (bf16 default: the keys decide the selection)",
    "TF_DSV41_VISION": "0 | 1: the DeepSeek-ViT tower on rank 0 (0.9 GiB)",
    # serving (M3: engine/serving/stack.py, batch.py, sessions.py, sessdisk.py, memory.py, app.py, structured.py)
    "TF_DSV41_POOL_TOKENS": "KV pool tokens (default --parallel x (--context + 64), whole 256-token pages)",
    "TF_DSV41_PREFILL_ROWS": "prompt rows a round shared by the prefilling slots (2048; 16-token grid)",
    "TF_DSV41_DRAFT_DEPTH": "DSpark drafts a round (3) when the forward drafts",
    "TF_DSV41_FLOOR_GIB": "MemAvailable target a node (5): the load-time budget",
    "TF_DSV41_FLOOR_HARD_GIB": "hard floor (4): admission waits below it",
    "TF_DSV41_PACK": "the weight pack the load-time budget assumes (mia29)",
    "TF_DSV41_SESSIONS": "1 | 0: the session store (RAM tier on pool pages, replay point, turn ends)",
    "TF_DSV41_SESSION_RAM_MIB": "the RAM tier's bounded state (256)",
    "TF_DSV41_SESSION_DISK": "the NVMe tier's directory (unset: none)",
    "TF_DSV41_SESSION_DISK_GIB": "the NVMe tier's budget (64)",
    "TF_DSV41_SESSION_DISK_MIN": "shorter entries are dropped, not parked (1024 tokens)",
    "TF_DSV41_GRAMMAR": "1 | 0: structured output (response_format, guided_*, strict / required / named tools)",
    "TF_DSV41_TOOL_GRAMMAR": "0 | 1: every tools request held to the schema-typed DSML grammar",
    "TF_DSV41_GRAMMAR_THREADS": "mask-fill threads (4)",
    "TF_DSV41_THINKING": "1 | 0: thinking by default",
    "TF_DSV41_DEFAULT_EFFORT": "low (50) | high (75) | max (100) | 1-100: a thinking request's default effort",
    "TF_DSV41_SCHEMA_PROMPT": "1 | 0: response_format's schema in the system block",
    "TF_DSV41_REASONING_MEMORY": "reasoning kept for history messages that come without it (1024 replies)",
    "TF_DSV41_REQUEST_LOG": "one JSON line a request (no text) to this file",
    "TF_DSV41_TOKCACHE": "the prompt-token cache (GLM 0210)",
    "TF_DSV41_DISCONNECT": "1 | 0: stop requests whose client has gone",
    "TF_DSV41_HEALTH": "basic | strict: /health after a fatal error",
    "TF_DSV41_REASONING_FIELDS": "both | reasoning_content | reasoning",
}


def check(model_dir: str | Path) -> str | None:
    """None when ``model_dir`` is a V4.1-Flash EXL3 checkpoint this family serves, else why not."""

    p = Path(model_dir)
    try:
        cfg = json.loads((p / "config.json").read_text())
    except (OSError, ValueError) as e:
        return f"no readable config.json ({e})"
    t = cfg.get("text_config", cfg)
    if cfg.get("model_type") != "deepseek_v41" and t.get("model_type") not in ("deepseek_v41", "deepseek_v41_text"):
        return f"model_type {cfg.get('model_type')!r} is not deepseek_v41"
    q = cfg.get("quantization_config") or {}
    if str(q.get("quant_method", "")).lower() != "exl3":
        return "this family serves the EXL3 packs (quant_method exl3)"
    if str(q.get("codebook", "mul1")).lower() not in ("mul1", "mcg", "3inst"):
        return f"unknown EXL3 codebook {q.get('codebook')!r}"
    need = {"compress_ratios", "kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"}
    missing = sorted(need - set(t))
    if missing:
        return f"config.json lacks {', '.join(missing)} (not a V4.1 layout)"
    return None


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    """The CLI's engine factory (GLM's ``cuda_engine`` signature): TP=2 over two Sparks only."""

    if tp != 2 or not master:
        raise ValueError("DeepSeek-V4.1-Flash runs on two ranks (--tp 2 --master <rank 0 address>)")
    from .engine import Dsv41Engine

    context = options.get("context", 0) if options.get("context_explicit") else 0
    return Dsv41Engine(Path(model_dir), rank=rank, master=master, port=master_port, context=context,
                       serial_only=no_drafts, draft_depth=mtp_drafts)


def __getattr__(name: str):
    if name == "CUDA_APP":
        from .app import Dsv41App

        return Dsv41App
    if name == "CUDA_SERVE":
        from tensorfold.families.glm5_next.spark.server import serve       # our server (0150-0620), as GLM's

        return serve
    raise AttributeError(name)
