# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""DeepSeek-V4.1 replies: reasoning, content and DSML tool calls, whole (``parse``) or while they stream (``Stream``).

A V4.1 reply (thinking mode; the prompt ends with ``<think>``):

    reasoning</think>content

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="get_weather">
    <｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
    <｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls><｜end▁of▁sentence｜>

(chat mode: no reasoning part). The reference parser (``parse_message_from_completion_text``) is strict; this one is
lenient where models slip, as GLM 0620's fixes are:

- a calls block inside an unclosed think block ends the reasoning there (``thinkcalls``);
- ``<｜DSML｜calls>`` / V4's ``tool_calls`` / ``function_calls`` block names, missing blank lines, an unclosed last
  invoke or block (a reply cut by max_tokens) are read;
- values: ``string="true"`` keeps the text (read by the schema when the tool's schema does not allow a string,
  GLM 0620 ``typed_value``); ``string="false"`` is JSON, closed when it stops one bracket short (#87), else read by
  the schema, else kept as text;
- an invoke of a tool the request did not offer stays in the content (as GLM's parser leaves unknown calls).

Arguments are compact JSON (``{"city":"Paris","days":3}``); while streaming, each completed parameter is sent as one
fragment (``{`` + key + value, ``,`` + key + value ..., ``}``), so the fragments concatenate to the final arguments.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tensorfold.tool_parameters import closed_json

from .encoding import DSML, THINK_END

_BLOCK_OPEN = re.compile(r"<｜DSML｜\s?(?:calls|tool_calls|function_calls)>")
_BLOCK_CLOSE = re.compile(r"</｜DSML｜\s?(?:calls|tool_calls|function_calls)>")
_INVOKE = re.compile(r'<｜DSML｜\s?invoke\s+name="([^"]*)"\s*>')
_INVOKE_END = re.compile(r"</｜DSML｜\s?invoke>")
_PARAM = re.compile(r'<｜DSML｜\s?parameter\s+name="([^"]*)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜\s?parameter>',
                    re.DOTALL)
MARK = "<" + DSML                         # every DSML tag starts here: text that may become one is held back
OPEN_TAGS = tuple(f"{sep}<{DSML}{name}>" for sep in ("", "\n\n") for name in (" calls", "calls", "tool_calls",
                                                                                  "function_calls"))


def _partial(text: str, tag: str) -> int:
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def _hold(text: str) -> int:
    """Trailing characters that may still become a calls block's opening tag (with its blank line)."""

    return max(_partial(text, tag) for tag in OPEN_TAGS)


def _schemas(tools: Sequence[dict] | None) -> dict[str, dict]:
    out = {}
    for t in tools or ():
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn, dict) and fn.get("name"):
            out[str(fn["name"])] = ((fn.get("parameters") or {}).get("properties") or {})
    return out


def _known(name: str, schemas: dict[str, dict]) -> str | None:
    if name in schemas:
        return name
    low = {k.lower(): k for k in schemas}
    return low.get(name.lower()) or low.get(name.split("::")[-1].lower())


def value(raw: str, string: str, prop: Any) -> Any:
    """A parameter's value (see the module docstring)."""

    from tensorfold.families.glm5_next.spark.server import schema_types, typed_value

    types = schema_types(prop)
    if string == "true":
        if types is None or "string" in types:
            return raw
        return typed_value(raw, prop)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    fixed = closed_json(raw.strip())
    if fixed is not None:
        try:
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass
    return typed_value(raw, prop) if types is not None else raw


def fragment(first: bool, key: str, val: Any) -> str:
    return ("{" if first else ",") + json.dumps(key, ensure_ascii=False) + ":" + \
        json.dumps(val, ensure_ascii=False, separators=(",", ":"))


def new_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


@dataclass
class Call:
    name: str
    args: list[tuple[str, Any]] = field(default_factory=list)
    closed: bool = False
    id: str = field(default_factory=new_id)

    def arguments(self) -> str:
        if not self.args:
            return "{}"
        return "".join(fragment(i == 0, k, v) for i, (k, v) in enumerate(self.args)) + "}"

    def openai(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments()}}


def _calls(block: str, schemas: dict[str, dict] | None) -> tuple[list[Call], int]:
    """Invokes of a calls block body (complete parameters only); -> (calls, chars fully read)."""

    out: list[Call] = []
    pos = 0
    read = 0
    while True:
        m = _INVOKE.search(block, pos)
        if m is None:
            break
        name = m.group(1)
        end = _INVOKE_END.search(block, m.end())
        nxt = _INVOKE.search(block, m.end())
        stop = end.start() if end is not None and (nxt is None or end.start() < nxt.start()) else \
            (nxt.start() if nxt is not None else len(block))
        call = Call(name)
        props = (schemas or {}).get(_known(name, schemas or {}) or name, {})
        seen = set()
        for p in _PARAM.finditer(block, m.end(), stop):
            key = p.group(1)
            if key in seen:
                continue
            seen.add(key)
            call.args.append((key, value(p.group(3), p.group(2), props.get(key))))
        call.closed = end is not None and stop == end.start()
        out.append(call)
        pos = end.end() if call.closed else stop
        if call.closed:
            read = pos
        if not call.closed and nxt is None:
            break
    return out, read


@dataclass
class Reply:
    reasoning: str
    content: str
    calls: list[Call]
    raw_calls: str = ""                    # the calls block's text (kept in content when no call is usable)


def _split(text: str, thinking: bool) -> tuple[str, str, int]:
    """(reasoning, the rest, where the rest starts)."""

    if not thinking:
        return "", text, 0
    end = text.find(THINK_END)
    blk = _BLOCK_OPEN.search(text)
    if end >= 0 and (blk is None or end < blk.start()):
        return text[:end], text[end + len(THINK_END):], end + len(THINK_END)
    if blk is not None:                      # calls inside an unclosed think block: the reasoning ends there
        cut = blk.start()
        head = text[:cut]
        reasoning = head[:-2] if head.endswith("\n\n") else head.rstrip("\n")
        return reasoning, text[len(reasoning):], len(reasoning)
    return text, "", len(text)


def parse(text: str, *, thinking: bool, tools: Sequence[dict] | None = None) -> Reply:
    """A finished reply's parts (EOS already removed)."""

    reasoning, rest, _ = _split(text, thinking)
    blk = _BLOCK_OPEN.search(rest) if tools else None
    if blk is None:
        return Reply(reasoning, rest, [])
    content = rest[:blk.start()]
    content = content.removesuffix("\n\n")
    close = _BLOCK_CLOSE.search(rest, blk.end())
    body = rest[blk.end():close.start() if close else len(rest)]
    schemas = _schemas(tools)
    calls, _ = _calls(body, schemas)
    usable = []
    for c in calls:
        known = _known(c.name, schemas)
        if known is not None:
            c.name = known
            usable.append(c)
    if not usable:
        return Reply(reasoning, rest, [])
    return Reply(reasoning, content, usable, rest[blk.start():])


class Stream:
    """A reply's deltas while it grows: ``feed(text)`` (the whole decoded text so far) -> OpenAI delta dicts
    (``reasoning`` / ``content`` text, ``tool_calls`` entries); ``finish(text)`` releases what was held back.
    Concatenating the deltas gives ``parse(text)``'s parts (tool-call arguments included)."""

    def __init__(self, *, thinking: bool, tools: Sequence[dict] | None) -> None:
        self.thinking = thinking
        self.tools = list(tools or [])
        self.schemas = _schemas(self.tools)
        self.sent_reasoning = 0
        self.sent_content = 0
        self.calls: list[Call] = []
        self.sent_args: list[int] = []          # parameters sent a call
        self.sent_close: list[bool] = []
        self.in_calls = False

    def _content_limit(self, rest: str, finished: bool) -> tuple[int, Any]:
        """How much of the rest is content now, and the calls block's match (or None)."""

        blk = _BLOCK_OPEN.search(rest) if self.tools else None
        if blk is not None:
            cut = blk.start()
            if rest[:cut].endswith("\n\n"):
                cut -= 2
            return cut, blk
        if finished:
            return len(rest), None
        return len(rest) - (_hold(rest) if self.tools else 0), None

    def feed(self, text: str, finished: bool = False) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.thinking:
            end = text.find(THINK_END)
            blk = _BLOCK_OPEN.search(text) if self.tools else None
            if end < 0 and blk is None:
                hold = _partial(text, THINK_END)
                if self.tools:
                    hold = max(hold, _hold(text))
                upto = len(text) if finished else len(text) - hold
                if upto > self.sent_reasoning:
                    out.append({"reasoning": text[self.sent_reasoning:upto]})
                    self.sent_reasoning = upto
                return out
        reasoning, rest, _ = _split(text, self.thinking)
        if len(reasoning) > self.sent_reasoning:
            out.append({"reasoning": reasoning[self.sent_reasoning:]})
            self.sent_reasoning = len(reasoning)
        limit, blk = self._content_limit(rest, finished)
        if limit > self.sent_content and not self.in_calls:
            out.append({"content": rest[self.sent_content:limit]})
            self.sent_content = limit
        if blk is None:
            return out
        self.in_calls = True
        close = _BLOCK_CLOSE.search(rest, blk.end())
        body = rest[blk.end():close.start() if close else len(rest)]
        calls, _ = _calls(body, self.schemas)
        k = -1
        for c in calls:
            known = _known(c.name, self.schemas)
            if known is None:
                continue                          # an unknown tool: left out (``parse`` keeps it as content)
            c.name = known
            k += 1
            if k == len(self.calls):
                self.calls.append(c)
                self.sent_args.append(0)
                self.sent_close.append(False)
                out.append({"tool_calls": [{"index": k, "id": c.id, "type": "function",
                                            "function": {"name": c.name, "arguments": ""}}]})
            mine = self.calls[k]
            mine.args = c.args
            frag = "".join(fragment(j == 0, key, val) for j, (key, val) in enumerate(c.args[self.sent_args[k]:],
                                                                                       start=self.sent_args[k]))
            self.sent_args[k] = len(c.args)
            if (c.closed or finished) and not self.sent_close[k]:
                frag += "}" if c.args else "{}"
                self.sent_close[k] = True
                mine.closed = True
            if frag:
                out.append({"tool_calls": [{"index": k, "function": {"arguments": frag}}]})
        return out

    def finish(self, text: str) -> list[dict[str, Any]]:
        return self.feed(text, finished=True)
