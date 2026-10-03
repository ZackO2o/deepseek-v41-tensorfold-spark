# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""DeepSeek-V4.1's chat encoding (``encode_messages`` of the checkpoint's ``encoding/encoding.py``, MIT, DeepSeek):
the prompt format rendered from OpenAI-shaped messages, without the kit's (AGPL) Jinja template. Same strings as
the reference for every case it renders (``tests/serving/test_template.py`` compares them), plus the serving rules
upstream TensorFold's V4 family uses (``deepseek_v4/prompts.py``):

- tools and ``response_format`` join the first system message (an empty one is added when the conversation has
  none; the reference renders them only on a system message);
- OpenAI shapes: ``content`` as a list of text parts, ``content: null``, tool-call ``arguments`` as a JSON string,
  ``reasoning`` for ``reasoning_content``; images are refused (vision is not served by this build).

The format (V4.1): ``<｜begin▁of▁sentence｜>``, an optional ``<｜System｜>`` lead with ``Reasoning Effort: N (range
1-100, ...)`` in thinking mode, ``<｜User｜>`` turns (tool results merged into them as ``<tool_result>`` blocks in
call order), ``<｜Assistant｜>`` + ``<think>`` (thinking) or ``</think>`` (chat), assistant turns as ``reasoning</think>
content`` + ``\\n\\n<｜DSML｜ calls>`` blocks + ``<｜end▁of▁sentence｜>``. Reasoning of turns before the last user turn
is dropped unless the conversation has tools.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
THINK_OPEN, THINK_END = "<think>", "</think>"
DSML = "｜DSML｜"
SYSTEM, USER, ASSISTANT, REMINDER = "<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "<｜latest_reminder｜>"
CALLS, INVOKE, PARAM = " calls", " invoke", " parameter"
CALLS_OPEN, CALLS_CLOSE = f"<{DSML}{CALLS}>", f"</{DSML}{CALLS}>"
EFFORTS = {"low": 50, "high": 75, "max": 100}
DEFAULT_EFFORT = "high"

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{d}{c}>" block like the following:

<{d}{c}>
<{d}{i} name="$TOOL_NAME">
<{d}{p} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{d}{p}>
...
</{d}{i}>
<{d}{i} name="$TOOL_NAME2">
...
</{d}{i}>
</{d}{c}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {o}), you MUST output your complete reasoning inside {o}...{e} BEFORE any tool calls or final response.

Otherwise, output directly after {e} with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=True)


def effort_value(effort: Any) -> int:
    """``low`` / ``high`` / ``max`` -> 50 / 75 / 100, an int 1-100 as is (None: the default, high)."""

    if effort is None:
        effort = DEFAULT_EFFORT
    if type(effort) is int and 1 <= effort <= 100:
        return effort
    if isinstance(effort, str) and effort in EFFORTS:
        return EFFORTS[effort]
    raise ValueError(f"Invalid reasoning effort for deepseek_v41: {effort!r}, expected an int 1-100 or "
                     f"{', '.join(EFFORTS)}")


def _text(content: Any, where: str) -> str:
    """OpenAI ``content``: a string, null, or a list of text parts (joined with blank lines, as the reference)."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
                out.append(str(part.get("text") or ""))
            elif isinstance(part, dict) and part.get("type") in ("image_url", "image", "input_image"):
                raise ValueError(f"{where}: images are not served by this DeepSeek-V4.1 build")
            else:
                raise ValueError(f"{where}: unsupported content part {part!r:.80}")
        return "\n\n".join(out)
    return to_json(content)


def _tool_name(tool: dict) -> str:
    ns = tool.get("namespace")
    if isinstance(ns, dict):
        ns = ns.get("name")
    name = str(tool["name"])
    pre, sep, bare = name.partition("::")
    if sep:
        ns, name = pre, bare
    return name if not ns else f"{ns}::{name}"


def tool_functions(tools: Sequence[dict]) -> list[dict]:
    """OpenAI tools -> the function definitions the system block lists (namespace-qualified names)."""

    out = []
    for t in tools:
        fn = dict(t["function"]) if isinstance(t.get("function"), dict) else dict(t)
        if t.get("namespace") is not None:
            fn["namespace"] = t["namespace"]
        ns = fn.get("namespace")
        fn["name"] = _tool_name(fn)
        fn.pop("namespace", None)
        if isinstance(ns, dict) and ns.get("description"):
            fn["description"] = ns["description"] + "\n" + (fn.get("description") or "")
        out.append(fn)
    return out


def render_tools(tools: Sequence[dict]) -> str:
    schemas = "\n".join(to_json(f) for f in tool_functions(tools))
    return TOOLS_TEMPLATE.format(d=DSML, c=CALLS, i=INVOKE, p=PARAM, o=THINK_OPEN, e=THINK_END, schemas=schemas)


def arguments_dict(arguments: Any) -> dict:
    """Tool-call arguments as a dict (JSON strings, even double-encoded, are read; else {"arguments": raw})."""

    raw = arguments
    for _ in range(2):
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except (json.JSONDecodeError, ValueError):
                break
    return arguments if isinstance(arguments, dict) else {"arguments": raw}


def render_call(name: str, arguments: Any) -> str:
    params = [f'<{DSML}{PARAM} name="{k}" string="{"true" if isinstance(v, str) else "false"}">'
              f'{v if isinstance(v, str) else to_json(v)}</{DSML}{PARAM}>' for k, v in arguments_dict(arguments).items()]
    return f'<{DSML}{INVOKE} name="{name}">\n' + "\n".join(params) + f"\n</{DSML}{INVOKE}>"


def render_calls(calls: Sequence[dict]) -> str:
    body = "\n".join(render_call(_tool_name(c.get("function") or c), (c.get("function") or c).get("arguments"))
                     for c in calls)
    return f"\n\n{CALLS_OPEN}\n{body}\n{CALLS_CLOSE}"


def normalize(messages: Sequence[dict], tools: Sequence[dict] | None, response_format: Any = None) -> list[dict]:
    """OpenAI messages -> the reference's message dicts, with tools / response format on the first system one."""

    out: list[dict] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m:
            raise ValueError(f"messages[{i}]: an object with a role")
        role = m["role"]
        if role == "developer":
            role = "system"
        msg: dict[str, Any] = {"role": role}
        if role == "assistant":
            msg["content"] = _text(m.get("content"), f"messages[{i}].content")
            rc = m.get("reasoning_content")
            rc = m.get("reasoning") if rc is None else rc
            if rc is not None:
                msg["reasoning_content"] = _text(rc, f"messages[{i}].reasoning_content")
            if m.get("tool_calls"):
                msg["tool_calls"] = copy.deepcopy(m["tool_calls"])
        elif role == "tool":
            msg["content"] = _text(m.get("content"), f"messages[{i}].content")
            msg["tool_call_id"] = m.get("tool_call_id") or ""
        elif role in ("system", "user", "latest_reminder"):
            msg["content"] = _text(m.get("content"), f"messages[{i}].content")
        else:
            raise ValueError(f"messages[{i}]: unknown role {role!r}")
        out.append(msg)
    if tools or response_format:
        if not out or out[0]["role"] != "system":
            out.insert(0, {"role": "system", "content": ""})
        if tools:
            out[0]["tools"] = list(tools)
        if response_format:
            out[0]["response_format"] = response_format
    return out


def _merge_tools(messages: list[dict]) -> list[dict]:
    """Tool results into user turns (``merge_tool_messages``), sorted by the calling turn's order."""

    merged: list[dict] = []
    order: dict[str, int] = {}
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            order = {}
            for k, tc in enumerate(m["tool_calls"]):
                tc_id = tc.get("id") or (tc.get("function") or {}).get("id", "")
                if tc_id:
                    order[tc_id] = k
        if role in ("tool", "user"):
            block = {"tool": role == "tool", "id": m.get("tool_call_id", ""), "text": m.get("content", ""),
                     "rank": order.get(m.get("tool_call_id", ""), 0) if role == "tool" else 0}
            if merged and merged[-1]["role"] == "user":
                merged[-1]["blocks"].append(block)
            else:
                merged.append({"role": "user", "blocks": [block], "order": dict(order)})
        else:
            merged.append(dict(m))
    for m in merged:
        if m["role"] != "user":
            continue
        tools = [b for b in m["blocks"] if b["tool"]]
        if len(tools) > 1 and m["order"]:
            ranked = iter(sorted(tools, key=lambda b: b["rank"]))
            m["blocks"] = [next(ranked) if b["tool"] else b for b in m["blocks"]]
    return merged


def _last_user(messages: list[dict]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx]["role"]
        if role == "user" or (role == "system" and idx > 0):
            return idx
    return -1


def encode(messages: Sequence[dict], *, tools: Sequence[dict] | None = None, thinking: bool = True,
           effort: Any = None, response_format: Any = None, drop_thinking: bool = True,
           add_generation_prompt: bool = True) -> str:
    """The V4.1 prompt for OpenAI-shaped ``messages`` (``encode_messages(..., thinking_mode, reasoning_effort)``)."""

    msgs = _merge_tools(normalize(messages, tools, response_format))
    budget = effort_value(effort)
    drop = drop_thinking and not any(m.get("tools") for m in msgs)
    last = _last_user(msgs)
    if thinking and drop:                     # ``_drop_thinking_messages``: reasoning before the last user turn
        msgs = [dict(m, reasoning_content=None) if m["role"] == "assistant" and i < last else m
                for i, m in enumerate(msgs)]
    out = [BOS]
    for idx, m in enumerate(msgs):
        role = m["role"]
        effort_text = (f"Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the "
                       "reasoning)\n\n") if idx == 0 and thinking else ""
        if idx == 0 and (effort_text or role == "system"):
            out.append(SYSTEM)
        out.append(effort_text)
        if role == "system":
            if idx > 0:
                out.append(SYSTEM)
            out.append(m.get("content") or "")
            if m.get("tools"):
                out.append("\n\n" + render_tools(m["tools"]))
            if m.get("response_format"):
                out.append("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
                           + to_json(m["response_format"]))
        elif role == "user":
            out.append(USER)
            out.append("\n\n".join(f"<tool_result>{b['text']}</tool_result>" if b["tool"] else b["text"]
                                   for b in m["blocks"]))
        elif role == "latest_reminder":
            out.append(REMINDER + (m.get("content") or ""))
        elif role == "assistant":
            if thinking and (not drop or idx > last):
                out.append((m.get("reasoning_content") or "") + THINK_END)
            out.append(m.get("content") or "")
            if m.get("tool_calls"):
                out.append(render_calls(m["tool_calls"]))
            out.append(EOS)
        nxt = msgs[idx + 1]["role"] if idx + 1 < len(msgs) else None
        if nxt is not None and nxt not in ("assistant", "latest_reminder"):
            continue
        if nxt is None and not add_generation_prompt and role != "assistant":
            continue
        if role == "user" or (role == "system" and idx > 0):
            out.append(ASSISTANT)
            out.append(THINK_OPEN if thinking and (not drop or idx >= last) else THINK_END)
    return "".join(out)


def generation_prefix(thinking: bool) -> str:
    return ASSISTANT + (THINK_OPEN if thinking else THINK_END)
