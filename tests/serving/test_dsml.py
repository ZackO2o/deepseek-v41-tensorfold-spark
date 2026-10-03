"""DSML replies: parse == the reference parser on well-formed replies; render -> parse round trips; streaming in any
chunking (characters, the real tokenizer's tokens) gives the same reasoning, content and call arguments as the whole
parse; the lenient cases (calls in an unclosed think block, a call cut by max_tokens, V4-style tags, typed values,
unknown tools); a multi-step conversation stays a prefix of its next prompt."""

from __future__ import annotations

import json

import pytest
from test_template import TOOLS, ref

from engine.serving import dsml, encoding

EOS = encoding.EOS


def reply_text(reasoning, content, calls, thinking=True):
    head = (reasoning + encoding.THINK_END) if thinking else ""
    return head + content + (encoding.render_calls(calls) if calls else "")


CALLS = [{"function": {"name": "get_weather", "arguments": {"city": "Paris, \"FR\"", "days": 3}}},
         {"function": {"name": "search", "arguments": {"q": "naïve <b>", "filters": {"lang": "fr", "n": None},
                                                       "top": [1, 2]}}}]


def collect(stream: dsml.Stream, pieces) -> dict:
    out = {"reasoning": "", "content": "", "calls": {}}
    text = ""
    events = []
    for p in pieces:
        text += p
        events += stream.feed(text)
    events += stream.finish(text)
    for e in events:
        if "reasoning" in e:
            out["reasoning"] += e["reasoning"]
        if "content" in e:
            out["content"] += e["content"]
        for tc in e.get("tool_calls", []):
            c = out["calls"].setdefault(tc["index"], {"name": None, "args": "", "ids": set()})
            if "id" in tc:
                c["ids"].add(tc["id"])
                c["name"] = tc["function"]["name"]
            c["args"] += tc["function"]["arguments"]
    return out


def test_parse_matches_reference():
    text = reply_text("Plan: two calls.", "Let me check.", CALLS)
    got = dsml.parse(text, thinking=True, tools=TOOLS)
    want = ref.parse_message_from_completion_text(text + EOS, "thinking")
    assert got.reasoning == want["reasoning_content"] and got.content == want["content"]
    assert [c.name for c in got.calls] == [c["function"]["name"] for c in want["tool_calls"]]
    for c, w in zip(got.calls, want["tool_calls"]):
        assert json.loads(c.arguments()) == json.loads(w["function"]["arguments"])


@pytest.mark.parametrize("thinking", [True, False])
def test_round_trip_and_rerender(thinking):
    text = reply_text("why", "", CALLS, thinking)
    r = dsml.parse(text, thinking=thinking, tools=TOOLS)
    assert [json.loads(c.arguments()) for c in r.calls] == [c["function"]["arguments"] for c in CALLS]
    # the history message made from the parse renders back to the reply's exact text (session prefixes hold)
    calls = [c.openai() for c in r.calls]
    again = encoding.render_calls(calls)
    assert text.endswith(again)


@pytest.mark.parametrize("step", [1, 2, 3, 7])
def test_stream_chunks_equal_parse(step):
    text = reply_text("thinking about <｜DSML maybe", "Answer:\n\nhere", CALLS)
    whole = dsml.parse(text, thinking=True, tools=TOOLS)
    got = collect(dsml.Stream(thinking=True, tools=TOOLS), [text[i:i + step] for i in range(0, len(text), step)])
    assert got["reasoning"] == whole.reasoning and got["content"] == whole.content
    assert [got["calls"][i]["name"] for i in sorted(got["calls"])] == [c.name for c in whole.calls]
    for i, c in enumerate(whole.calls):
        assert json.loads(got["calls"][i]["args"]) == json.loads(c.arguments())
        assert len(got["calls"][i]["ids"]) == 1


def test_stream_with_tokenizer(model_dir):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    text = reply_text("I will call it.", "", CALLS[:1])
    ids = tok.encode(text, add_special_tokens=False).ids
    assert tok.token_to_id("｜DSML｜") in ids
    pieces, prev = [], ""
    for k in range(1, len(ids) + 1):
        cur = tok.decode(ids[:k], skip_special_tokens=False)
        pieces.append(cur[len(prev):])
        prev = cur
    got = collect(dsml.Stream(thinking=True, tools=TOOLS), pieces)
    assert got["calls"][0]["name"] == "get_weather"
    assert json.loads(got["calls"][0]["args"]) == {"city": "Paris, \"FR\"", "days": 3}


def test_lenient_cases():
    # calls inside an unclosed think block
    t = "I should call.\n\n" + encoding.render_calls(CALLS[:1])[2:]
    r = dsml.parse(t, thinking=True, tools=TOOLS)
    assert r.reasoning == "I should call." and r.calls[0].name == "get_weather"
    # cut by max_tokens inside the second parameter: the first one is kept
    full = reply_text("", "", CALLS[:1])
    cut = full[:full.index('name="days"') + 15]
    r = dsml.parse(cut, thinking=True, tools=TOOLS)
    assert json.loads(r.calls[0].arguments()) == {"city": "Paris, \"FR\""}
    # V4-style tags (no space, tool_calls)
    v4 = '</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">x' \
         '</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
    assert dsml.parse(v4, thinking=True, tools=TOOLS).calls[0].name == "search"
    # typed values: a quoted integer, a JSON array one bracket short
    odd = '</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n<｜DSML｜ parameter name="days" ' \
          'string="true">4</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name="search">\n<｜DSML｜ ' \
          'parameter name="top" string="false">[1, 2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    r = dsml.parse(odd, thinking=True, tools=TOOLS)
    assert json.loads(r.calls[0].arguments()) == {"days": 4}
    assert json.loads(r.calls[1].arguments()) == {"top": [1, 2]}
    # an unknown tool stays content; no tools: nothing is parsed
    unk = reply_text("", "x", [{"function": {"name": "rm_rf", "arguments": {}}}])
    r = dsml.parse(unk, thinking=True, tools=TOOLS)
    assert not r.calls and "rm_rf" in r.content
    assert not dsml.parse(reply_text("", "x", CALLS), thinking=True, tools=None).calls


def test_multi_step_prompt_prefix():
    """Three tool steps: each next prompt starts with the previous prompt + the model's exact reply + EOS."""

    msgs = [{"role": "system", "content": "agent"}, {"role": "user", "content": "plan a trip"}]
    prompt = encoding.encode(msgs, tools=TOOLS, thinking=True)
    replies = [reply_text("step one", "", CALLS[:1]), reply_text("step two", "Now search.", CALLS[1:]),
               reply_text("done", "All set.", [])]
    for k, text in enumerate(replies):
        r = dsml.parse(text, thinking=True, tools=TOOLS)
        msg = {"role": "assistant", "content": r.content, "reasoning_content": r.reasoning}
        if r.calls:
            msg["tool_calls"] = [c.openai() for c in r.calls]
        msgs.append(msg)
        for c in r.calls:
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": f"result {k}"})
        if not r.calls:
            msgs.append({"role": "user", "content": "thanks"})
        nxt = encoding.encode(msgs, tools=TOOLS, thinking=True)
        assert nxt.startswith(prompt + text + EOS), k
        prompt = nxt
