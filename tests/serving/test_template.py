"""The V4.1 prompt: ``engine/serving/encoding.py`` == DeepSeek's reference ``encode_messages`` (vendored as test data)
on chat, thinking, efforts, multi-step tool calls with out-of-order results, mid-conversation systems, merged users,
response formats; == the checkpoint folder's Jinja template where both define the case; tokenized as the model's
special tokens."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from engine.serving import encoding

REF_PATH = Path(__file__).parent / "data" / "encoding_dsv41_ref.py"
_spec = importlib.util.spec_from_file_location("encoding_dsv41_ref", REF_PATH)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                      "parameters": {"type": "object", "properties": {
                                          "city": {"type": "string"}, "days": {"type": "integer"}},
                                          "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Web search",
                                      "parameters": {"type": "object", "properties": {
                                          "q": {"type": "string"}, "filters": {"type": "object"},
                                          "top": {"type": "array", "items": {"type": "integer"}}}}}},
]


def _call(i, name, args, as_string=False):
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args) if as_string else args}}


def cases():
    yield "chat", [{"role": "user", "content": "Hello"}], None
    yield "system", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], None
    yield "multi-turn", [
        {"role": "system", "content": "S"}, {"role": "user", "content": "one"},
        {"role": "assistant", "content": "A1", "reasoning_content": "r1"},
        {"role": "user", "content": "two"}, {"role": "assistant", "content": "A2", "reasoning_content": "r2"},
        {"role": "user", "content": "three"}], None
    yield "merged-users", [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], None
    yield "mid-system", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                         {"role": "system", "content": "new rules"}], None
    yield "tools-3-steps", [
        {"role": "system", "content": "You are an track."},
        {"role": "user", "content": "Weather in Paris and news?"},
        {"role": "assistant", "content": "", "reasoning_content": "I need two tools.",
         "tool_calls": [_call(1, "get_weather", {"city": "Paris", "days": 3}),
                        _call(2, "search", {"q": "news", "filters": {"lang": "fr"}, "top": [1, 2]}, True)]},
        {"role": "tool", "tool_call_id": "call_2", "content": "news result"},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "assistant", "content": "Checking more.", "reasoning_content": "One more.",
         "tool_calls": [_call(3, "search", {"q": "paris events"})]},
        {"role": "tool", "tool_call_id": "call_3", "content": "a fair"},
    ], TOOLS
    yield "tools-final", [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "reasoning_content": "r", "tool_calls": [_call(9, "get_weather", {})]},
        {"role": "tool", "tool_call_id": "call_9", "content": "ok"},
        {"role": "assistant", "content": "Done.", "reasoning_content": "fine"},
        {"role": "user", "content": "thanks"}], TOOLS
    yield "reminder", [{"role": "user", "content": "x"}, {"role": "latest_reminder", "content": "be nice"}], None


def _ref_messages(messages, tools, response_format=None):
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if m.get("role") == "assistant" and m.get("content") is None:
            m["content"] = ""
    if tools or response_format:
        if msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        if tools:
            msgs[0]["tools"] = tools
        if response_format:
            msgs[0]["response_format"] = response_format
    return msgs


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("effort", [None, "low", "max", 37])
def test_matches_reference(thinking, effort):
    for name, messages, tools in cases():
        ours = encoding.encode(copy.deepcopy(messages), tools=tools, thinking=thinking, effort=effort)
        theirs = ref.encode_messages(_ref_messages(messages, tools), thinking_mode="thinking" if thinking else "chat",
                                     reasoning_effort=effort)
        assert ours == theirs, name


def test_response_format_and_tools_without_system():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    msgs = [{"role": "user", "content": "give json"}]
    ours = encoding.encode(msgs, tools=TOOLS[:1], thinking=True, response_format=schema)
    theirs = ref.encode_messages(_ref_messages(msgs, TOOLS[:1], schema), thinking_mode="thinking")
    assert ours == theirs
    assert "## Response Format:" in ours and "## Tools" in ours
    assert ours.startswith(encoding.BOS + encoding.SYSTEM + "Reasoning Effort: 75")


def test_openai_shapes_and_refusals():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": None, "reasoning": "kept", "tool_calls": [_call(1, "search", "{}")]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "r"}]}]
    text = encoding.encode(msgs, tools=TOOLS, thinking=True)
    assert "<｜User｜>a\n\nb" in text and "kept</think>" in text and "<tool_result>r</tool_result>" in text
    with pytest.raises(ValueError, match="images"):
        encoding.encode([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])
    with pytest.raises(ValueError):
        encoding.encode([{"role": "user", "content": "x"}], effort=0)
    with pytest.raises(ValueError):
        encoding.encode([{"role": "wizard", "content": "x"}])


def test_generation_prefix():
    assert encoding.encode([{"role": "user", "content": "x"}], thinking=True).endswith("<｜Assistant｜><think>")
    assert encoding.encode([{"role": "user", "content": "x"}], thinking=False).endswith("<｜Assistant｜></think>")


def test_jinja_template_agrees(model_dir):
    from tensorfold.families.glm5_next.spark.server import ChatTemplate

    if not (model_dir / "chat_template.jinja").is_file():
        pytest.skip("no chat_template.jinja")
    jinja = ChatTemplate(model_dir)
    for name, messages, tools in cases():
        if messages[0]["role"] != "system":
            continue                                # the Jinja port renders tools without a system differently
        for thinking in (True, False):
            for eff in (None, "max"):
                ours = encoding.encode(copy.deepcopy(messages), tools=tools, thinking=thinking, effort=eff)
                msgs = copy.deepcopy(messages)
                for m in msgs:
                    if m.get("role") == "assistant" and m.get("content") is None:
                        m["content"] = ""
                extra = {"reasoning_effort": eff} if eff else {}
                theirs = jinja.render(msgs, tools=tools, enable_thinking=thinking, extra=extra)
                assert ours == theirs, (name, thinking, eff)


def test_special_tokens(model_dir):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    _, messages, tools = list(cases())[5]
    text = encoding.encode(messages, tools=tools, thinking=True)
    ids = tok.encode(text, add_special_tokens=False).ids
    assert ids[0] == 0 and ids[1] == tok.token_to_id("<｜System｜>")
    for s in ("<｜User｜>", "<｜Assistant｜>", "｜DSML｜", "</think>", "<｜end▁of▁sentence｜>"):
        assert tok.token_to_id(s) in ids, s
    assert tok.decode(ids, skip_special_tokens=False) == text
