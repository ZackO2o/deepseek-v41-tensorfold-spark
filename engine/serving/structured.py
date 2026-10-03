# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""Structured output for DeepSeek-V4.1 on GLM 0610's grammar module (``glm5_next/spark/grammar.py``): request
parsing (``response_format`` json_object / json_schema, vLLM's ``guided_*`` / ``structured_outputs``, strict /
required / named tool calls), xgrammar compilers with a cache, the per-reply ``Constraint`` (window cuts, mask fills,
follow) and the two-rank ``pack`` / ``follow``: all reused as is.

V4.1's own:

- tool calls are DSML: the ``tools`` view keeps the ``｜DSML｜`` token and compiles xgrammar 0.2.8's built-in
  ``deepseek_v4_1`` structural tag (``<｜DSML｜ calls>`` / `` invoke`` / `` parameter``, string parameters raw, others
  JSON by the tool's schema) instead of GLM's ``glm_4_7``;
- the grammar starts after ``</think>`` in thinking mode (GLM's ``Bound.active`` rule; the same token strings);
- the logits have 129,280 columns (``config.json``'s vocab_size), the tokenizer 128,000 + 1,283 added ids.

``Host`` is what GLM's ``grammar.check`` / ``for_request`` expect on the engine (``grammar_on``, ``grammars``,
``grammar_why``); the app owns one, the batcher's executor gets ``host.grammars`` (rank 1: the same, built at load).
Knobs: ``TF_DSV41_GRAMMAR`` (1: on; default 1), ``TF_DSV41_TOOL_GRAMMAR`` (1: every tools request is held to the
schema-typed DSML grammar, GLM 0620's ``grammar`` fix; default 0: only required / named / strict tools).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tensorfold.families.glm5_next.spark import grammar as gm

DSML_TOKENS = ("｜DSML｜",)
MODEL_TAG = "deepseek_v4_1"


def enabled() -> bool:
    v = os.environ.get("TF_DSV41_GRAMMAR", "1").strip() or "1"
    if v not in ("0", "1"):
        raise ValueError(f"TF_DSV41_GRAMMAR={v!r}: expected 0 or 1")
    return v == "1"


def tool_grammar() -> bool:
    return (os.environ.get("TF_DSV41_TOOL_GRAMMAR", "0").strip() or "0") == "1"


class Dsv41Grammars(gm.Grammars):
    """GLM's compilers with the DSML tool tag."""

    @classmethod
    def from_model(cls, model_dir: str | Path, vocab_size: int, stop_ids: Sequence[int]) -> Dsv41Grammars:
        import xgrammar as xgr

        path = Path(model_dir) / "tokenizer.json"
        infos = []
        for keep in ((), DSML_TOKENS):
            enc, backend, t_open, t_end = gm._vocab(path, vocab_size, keep)
            meta = xgr.TokenizerInfo._detect_metadata_from_hf(backend)
            infos.append(xgr.TokenizerInfo(enc, meta["vocab_type"], vocab_size=vocab_size,
                                           stop_token_ids=list(stop_ids), add_prefix_space=meta["add_prefix_space"]))
        return cls(xgr, infos[0], infos[1], vocab_size, stop_ids, t_open, t_end)

    def compile(self, spec: gm.Spec):
        if spec.kind != "tools":
            return super().compile(spec)
        try:
            with self.lock:
                d = json.loads(spec.text)              # a named function: xgrammar forces that tool
                tag = self.xgr.get_model_structural_tag(MODEL_TAG, tools=d["tools"], tool_choice=d["tool_choice"],
                                                         reasoning="disabled", max_whitespace_cnt=gm.BLANKS,
                                                         parallel_tool_calls=bool(d["parallel_tool_calls"]))
                return self.compilers["tools"].compile_structural_tag(tag)
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{spec.field}: the grammar cannot be enforced: {gm._message(exc)}") from None


class Host:
    """The engine-side attributes GLM's grammar functions read, for this checkpoint."""

    def __init__(self, model_dir: str | Path | None, vocab_size: int, eos: Sequence[int], *, on: bool | None = None,
                 quiet: bool = False) -> None:
        self.grammar_on = enabled() if on is None else bool(on)
        self.grammars = None
        self.grammar_why = None
        if not self.grammar_on or model_dir is None:
            return
        t0 = time.perf_counter()
        try:
            self.grammars = Dsv41Grammars.from_model(model_dir, vocab_size, eos)
        except ImportError:
            self.grammar_why = f"structured output needs {gm.EXTRA} (pip install {gm.EXTRA})"
        except Exception as exc:                       # noqa: BLE001  (served without it, said at load)
            self.grammar_why = f"structured output is unavailable: {exc}"
        if not quiet:
            what = (f"on (xgrammar, {vocab_size} columns, DSML tool tag {MODEL_TAG}; built in "
                    f"{time.perf_counter() - t0:.1f} s)") if self.grammars is not None else f"off: {self.grammar_why}"
            print(f"[tensorfold] structured output: {what}", flush=True)

    def spec(self, body: dict[str, Any]) -> gm.Spec | None:
        """The request's grammar spec (``TF_DSV41_TOOL_GRAMMAR``: every tools request), or None."""

        spec = gm.request_spec(body)
        if spec is None and tool_grammar() and body.get("tools") and body.get("tool_choice") in (None, "auto"):
            fns = [dict(t["function"] if isinstance(t.get("function"), dict) else t, strict=True)
                   for t in body["tools"] if isinstance(t, dict)]
            spec = gm._tool_spec(dict(body, tools=[{"type": "function", "function": f} for f in fns]))
        return spec

    def check(self, body: dict[str, Any]) -> str | None:
        """Why the request's grammar cannot run (HTTP 400), compiling it (cached), or None."""

        if not self.grammar_on:
            return None
        try:
            spec = self.spec(body)
            if spec is None:
                return None
            if self.grammars is None:
                return self.grammar_why or "structured output is unavailable on this server"
            self.grammars.compile(spec)
        except ValueError as exc:
            return str(exc)
        return None

    def for_request(self, body: dict[str, Any]):
        """(spec, compiled) for ``engine.request.grammar``, or None."""

        if not self.grammar_on or self.grammars is None:
            return None
        spec = self.spec(body)
        return None if spec is None else (spec, self.grammars.compile(spec))

    def bind(self, req, prompt: Sequence[int]):
        """``engine.request.grammar`` bound to a prompt (the think state): the batcher job's ``grammar``."""

        if req is None or self.grammars is None:
            return None
        spec, compiled = req
        return self.grammars.bind(spec, compiled, prompt)


def schema_for_prompt(body: dict[str, Any]) -> Any:
    """What the V4.1 prompt's ``## Response Format`` block shows for ``response_format`` (json_schema: its schema;
    json_object: ``{"type": "object"}``), or None."""

    rf = body.get("response_format")
    if not isinstance(rf, dict):
        return None
    kind = rf.get("type")
    if kind == "json_schema":
        js = rf.get("json_schema") or {}
        schema = js.get("schema", js.get("json_schema")) if isinstance(js, dict) else None
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except json.JSONDecodeError:
                return None
        return schema
    if kind == "json_object":
        return {"type": "object"}
    return None
