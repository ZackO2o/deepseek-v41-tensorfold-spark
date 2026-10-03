"""The V4.1 app on our server: replies parsed (reasoning, content, DSML calls), streamed tool calls, multi-step tool
calling with the reasoning restored, effort / thinking, /tokenize, context errors, stop strings, cancellation, the
HTTP routes; structured output end to end through the batcher (JSON schema, a required DSML tool call)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from tensorfold.server.cancellation import RequestCancelled
from test_template import TOOLS

from engine.serving import encoding
from engine.serving.app import Dsv41App, options

pytestmark = pytest.mark.filterwarnings("ignore")


class ScriptEngine:
    """Replies with a scripted text (tokenized), a few tokens a callback; records the prompts."""

    eos = (1,)

    def __init__(self, tok, limit=8192):
        self.tok = tok
        self.limit = limit
        self.request = threading.local()
        self.batch = None
        self.script = ""
        self.prompts = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        ids = self.tok.encode(self.script, add_special_tokens=False).ids + [1]
        ids = ids[:max_tokens]
        for i in range(0, len(ids), 3):
            if on_tokens(ids[i:i + 3]):
                break
        return {"cached": 0}


@pytest.fixture(scope="module")
def app(model_dir):
    from tokenizers import Tokenizer

    eng = ScriptEngine(Tokenizer.from_file(str(model_dir / "tokenizer.json")))
    return Dsv41App(eng, model_dir, "deepseek-v4.1-flash")


def chat(app, body, script):
    app.engine.script = script
    deltas = []
    body = dict(body)
    res = app.run(body, True, lambda d: deltas.append(d) or True)
    return res, deltas


CALL = encoding.render_calls([{"function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}}])


def test_reply_parts(app):
    res, _ = chat(app, {"messages": [{"role": "user", "content": "hi"}]}, "Let me think.</think>Hello!")
    assert res["reasoning"] == "Let me think." and res["content"] == "Hello!" and res["finish"] == "stop"
    assert app.engine.prompts[-1][0] == 0                        # BOS, then the effort line (thinking default on)
    res, _ = chat(app, {"messages": [{"role": "user", "content": "w?"}], "tools": TOOLS}, "plan</think>" + CALL)
    assert res["finish"] == "tool_calls" and res["content"] == ""
    call = res["calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 2}


def test_streamed_tool_calls(app):
    body = {"messages": [{"role": "user", "content": "w?"}], "tools": TOOLS, "stream": True}
    res, deltas = chat(app, body, "plan</think>Checking." + CALL)
    assert res["calls"] is None and res["finish"] == "tool_calls"
    content = "".join(d.get("content", "") for d in deltas)
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    calls = [tc for d in deltas for tc in d.get("tool_calls", [])]
    assert content == "Checking." and reasoning == "plan"
    assert calls[0]["function"]["name"] == "get_weather" and calls[0]["id"].startswith("call_")
    assert json.loads("".join(c["function"]["arguments"] for c in calls)) == {"city": "Paris", "days": 2}
    assert "｜DSML｜" not in content


def test_multi_step_with_dropped_reasoning(app):
    msgs = [{"role": "system", "content": "agent"}, {"role": "user", "content": "weather?"}]
    body = {"messages": msgs, "tools": TOOLS}
    script = "I need the weather tool.</think>" + CALL
    res, _ = chat(app, body, script)
    first_prompt = app.engine.prompts[-1]
    call = res["calls"][0]
    # the client sends the history back without the reasoning (as many agents do) and the tool's result
    hist = msgs + [{"role": "assistant", "content": None, "tool_calls": [call]},
                   {"role": "tool", "tool_call_id": call["id"], "content": "sunny"}]
    res2, _ = chat(app, {"messages": hist, "tools": TOOLS}, "Sunny.</think>It is sunny.")
    second = app.engine.prompts[-1]
    generated = app.engine.tok.encode(script, add_special_tokens=False).ids + [1]
    assert second[:len(first_prompt) + len(generated)] == first_prompt + generated     # a session prefix
    assert res2["content"] == "It is sunny." and res2["finish"] == "stop"


def test_effort_and_thinking(app):
    assert options({"reasoning_effort": "low"}, True, 75) == (True, 50)
    assert options({"reasoning_effort": "none"}, True, 75) == (False, 75)
    assert options({"reasoning_effort": "xhigh"}, False, 75) == (True, 100)
    assert options({"reasoning_effort": 37}, True, 75) == (True, 37)
    assert options({"chat_template_kwargs": {"enable_thinking": False}}, True, 75) == (False, 75)
    with pytest.raises(ValueError):
        options({"reasoning_effort": "extreme"}, True, 75)
    text, thinking = app.render({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "max"})
    assert thinking and "Reasoning Effort: 100" in text and text.endswith("<think>")
    text, thinking = app.render({"messages": [{"role": "user", "content": "x"}],
                                 "chat_template_kwargs": {"enable_thinking": False}})
    assert not thinking and text.endswith("</think>") and "Reasoning Effort" not in text
    res, _ = chat(app, {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "none"}, "plain")
    assert res["content"] == "plain" and res["reasoning"] == ""


def test_tokenize_and_context(app):
    body = {"messages": [{"role": "user", "content": "count me"}], "tools": TOOLS}
    ids = app.tokenize(dict(body))
    assert ids == app.tok.encode(app.render(dict(body))[0], add_special_tokens=False).ids
    assert app.check(dict(body)) is None
    big = dict(body, max_tokens=app.engine.limit)
    problem = app.check(big)
    assert problem is not None and getattr(problem, "code", None) == "context_length_exceeded"
    assert app.check({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "bogus"})
    assert app.check({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]})


def test_stop_strings(app):
    res, deltas = chat(app, {"messages": [{"role": "user", "content": "x"}], "stop": ["END"], "stream": True},
                       "r</think>one two END three")
    assert "".join(d.get("content", "") for d in deltas) == "one two " and res["finish"] == "stop"
    res, _ = chat(app, {"messages": [{"role": "user", "content": "x"}], "stop": "END"}, "r</think>a END b")
    assert res["content"] == "a " and res["finish"] == "stop"


def test_cancellation(app):
    app.engine.script = "r</think>" + "word " * 200
    with pytest.raises(RequestCancelled):
        app.run({"messages": [{"role": "user", "content": "x"}], "stream": True}, True, lambda d: False)
    left = {"n": 0}

    def gone():
        left["n"] += 1
        return left["n"] > 3

    with pytest.raises(RequestCancelled):
        app.run({"messages": [{"role": "user", "content": "x"}]}, True, lambda d: True, cancelled=gone)


def test_http_routes(app):
    from tensorfold.families.glm5_next.spark.server import make_handler

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        app.engine.script = "plan</think>" + CALL
        body = {"model": "x", "messages": [{"role": "user", "content": "w?"}], "tools": TOOLS, "stream": True}
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        events = [json.loads(line[6:]) for line in urllib.request.urlopen(req, timeout=30).read().decode().splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        calls = [tc for e in events for tc in e["choices"][0]["delta"].get("tool_calls", [])]
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert json.loads("".join(c["function"]["arguments"] for c in calls)) == {"city": "Paris", "days": 2}
        assert len([c for c in calls if "id" in c]) == 1
        tok = json.loads(urllib.request.urlopen(urllib.request.Request(
            url + "/tokenize", json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode()),
            timeout=30).read())
        assert tok["count"] == len(tok["tokens"]) and tok["tokens"][0] == 0
        models = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=30).read())
        assert models["data"][0]["max_model_len"] == app.engine.limit
    finally:
        srv.shutdown()


# -- structured output through the batcher -----------------------------------------------------------------------
class MiniEngine:
    """What M1's ``Dsv41Engine.generate`` does with the batcher (``Batcher.generate_request``)."""

    eos = (1,)

    def __init__(self, batch, host, limit):
        self.batch, self.grammar_host, self.limit = batch, host, limit
        self.request = threading.local()

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        return self.batch.generate_request(prompt, max_tokens, sampling, on_tokens, draft, request=self.request,
                                           host=self.grammar_host)


@pytest.fixture(scope="module")
def stack(model_dir, topo):
    from fake_forward import FakeForward

    from engine.serving.batch import Batcher
    from engine.serving.pool import Pool
    from engine.serving.structured import Host

    host = Host(model_dir, 129280, (1,), on=True, quiet=True)
    if host.grammars is None:
        pytest.skip(host.grammar_why)
    pool = Pool(topo, 8192)
    fwd = FakeForward(topo, pool, 2, 129280)
    b = Batcher(fwd, pool, n_slots=2, capacity=4096, grammars=host.grammars, session_min=16)
    eng = MiniEngine(b, host, 4000)
    yield Dsv41App(eng, model_dir, "dsv41"), b
    b.stop()


def test_json_schema_output(stack):
    app, _ = stack
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "n": {"enum": [1, 2, 3]}},
              "required": ["ok", "n"], "additionalProperties": False}
    body = {"messages": [{"role": "user", "content": "json please"}], "max_tokens": 300,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}}
    assert app.check(dict(body)) is None
    text, _ = app.render(dict(body))
    assert "## Response Format:" in text
    for seed in range(3):
        res = app.run(dict(body, seed=seed, temperature=1.0), True, lambda d: True)
        value = json.loads(res["content"])
        assert set(value) == {"ok", "n"} and isinstance(value["ok"], bool) and value["n"] in (1, 2, 3)
        assert res["finish"] == "stop"
    bad = dict(body, response_format={"type": "json_schema", "json_schema": {"schema": {"type": "nope"}}})
    assert app.check(bad)


def test_required_tool_call_output(stack):
    app, _ = stack
    tools = [{"type": "function", "function": {"name": "set_mode", "parameters": {
        "type": "object", "properties": {"mode": {"enum": ["fast", "safe"]}, "on": {"type": "boolean"}},
        "required": ["mode", "on"], "additionalProperties": False}}}]
    body = {"messages": [{"role": "user", "content": "go"}], "tools": tools, "tool_choice": "required",
            "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}}
    assert app.check(dict(body)) is None
    res = app.run(dict(body, seed=1, temperature=1.0), True, lambda d: True)
    assert res["finish"] == "tool_calls", res
    args = json.loads(res["calls"][0]["function"]["arguments"])
    assert args["mode"] in ("fast", "safe") and isinstance(args["on"], bool)
    named = dict(body, tool_choice={"type": "function", "function": {"name": "set_mode"}})
    assert app.check(named) is None
    res = app.run(dict(named, seed=2, temperature=1.0), True, lambda d: True)
    assert res["calls"][0]["function"]["name"] == "set_mode"
