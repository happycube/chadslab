"""Check the OpenAI compatible server, the sampler, and the chat template use.

    python scripts/check_server.py

The script uses a fake backend, so it needs no model. It starts the server on
a free port and exercises the paths with an HTTP client.
"""
from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.chat import parse_output  # noqa: E402
from np_gemma.sampling import Sampler  # noqa: E402
from np_gemma.server import Backend, make_server  # noqa: E402


class FakeTokenizer:
    """Map a token id to a letter, or to a text from a table."""

    def __init__(self, table=None):
        self.stop_ids = {99}
        self.table = table or {}

    def apply_chat_template(self, messages, add_generation_prompt=True, thinking=False,
                            tools=None, preserve_thinking=False):
        parts = ["<bos>"]
        if tools:
            parts.append("<tools>")
        parts.extend(m["content"] for m in messages)
        return "".join(parts)

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            if skip_special_tokens and i in self.stop_ids:
                continue
            out.append(self.table.get(i, chr(65 + (i % 26))))
        return "".join(out)


class FakeBackend(Backend):
    def __init__(self, tokens, table=None):
        super().__init__(None, FakeTokenizer(table), model_id="fake-model")
        self.script = list(tokens)

    def prompt_ids(self, messages, thinking=None, tools=None):
        return [1, 2, 3]

    def completion_ids(self, prompt):
        return [1, 2, 3]

    def stop_ids(self):
        return set()

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        for token in self.script[:max_tokens]:
            yield token


def request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    headers = {"Content-Type": "application/json"}
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data.decode("utf-8", "replace")


def parse_sse(text):
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def serve(backend):
    httpd = make_server(backend, port=0, quiet=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def check_sampling():
    logits = np.array([0.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float32)
    ok = Sampler(temperature=0.0)(logits) == 1
    ok &= Sampler(temperature=0.0, top_k=1)(logits) == 1
    ok &= Sampler(temperature=1.0, seed=7)(logits.copy()) == \
        Sampler(temperature=1.0, seed=7)(logits.copy())
    s = Sampler(temperature=0.0, repetition_penalty=2.0)
    s.reset([1, 1, 1])
    ok &= s(np.array([0.0, 5.0, 4.0])) == 2
    print("sampling: %s" % ("ok" if ok else "FAIL"))
    return ok


def check_parse():
    raw = ('<|channel>thought\nI reason\n<channel|>'
           '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,days:3}<tool_call|>'
           '<|tool_response><|tool_response>')
    got = parse_output(raw)
    ok = got["reasoning"] == "I reason"
    ok &= got["content"] == ""
    ok &= len(got["tool_calls"]) == 1
    ok &= got["tool_calls"][0]["function"]["name"] == "get_weather"
    args = json.loads(got["tool_calls"][0]["function"]["arguments"])
    ok &= args == {"location": "Paris", "days": 3}
    print("parse: %s calls=%s reason=%r" % ("ok" if ok else "FAIL", args, got["reasoning"]))
    # A stream sends deltas that the reader joins. The reasoning must only
    # ever grow, or the reader cannot correct an earlier delta.
    pieces = ["<|channel>", "thought", "\n", "The", " user", " is", " asking"]
    full = "".join(pieces)
    want = parse_output(full)["reasoning"]
    prev = ""
    grown = True
    for i in range(len(pieces)):
        r = parse_output("".join(pieces[:i + 1]))["reasoning"]
        if not r.startswith(prev):
            grown = False
        prev = r
    ok &= grown and want == parse_output(full)["reasoning"]
    print("parse stream: %s final=%r" % ("ok" if grown else "FAIL", want))
    return ok


def check_http(port):
    ok = True
    status, body = request(port, "GET", "/v1/models")
    obj = json.loads(body)
    ok &= status == 200 and obj["data"][0]["id"] == "fake-model"
    print("models: %s %s" % (status, obj["data"][0]["id"]))

    status, body = request(port, "GET", "/health")
    ok &= status == 200
    print("health: %s" % status)

    status, body = request(port, "POST", "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 3})
    obj = json.loads(body)
    text = obj["choices"][0]["message"]["content"]
    ok &= status == 200 and text == "KLM" and obj["usage"]["completion_tokens"] == 3
    print("chat: %s text=%r usage=%s" % (status, text, obj["usage"]))

    status, body = request(port, "POST", "/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
    ok &= status == 200 and json.loads(body)["object"] == "chat.completion"
    print("bare path: %s" % status)

    status, body = request(port, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3,
        "stream": True, "stream_options": {"include_usage": True}})
    events = parse_sse(body)
    ok &= status == 200 and events[-1] == "[DONE]"
    content = ""
    finish = usage = None
    for e in events[:-1]:
        chunk = json.loads(e)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for c in chunk.get("choices", []):
            content += (c.get("delta") or {}).get("content", "") or ""
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    ok &= content == "KLM" and finish == "length" and usage is not None
    print("stream: %s content=%r finish=%s usage=%s" % (status, content, finish, usage))

    status, body = request(port, "POST", "/v1/completions", {
        "model": "fake-model", "prompt": "hi", "max_tokens": 3})
    obj = json.loads(body)
    ok &= status == 200 and obj["choices"][0]["text"] == "KLM"
    print("completions: %s text=%r" % (status, obj["choices"][0]["text"]))
    return ok


def check_tools(port):
    ok = True
    body_in = {"messages": [{"role": "user", "content": "weather?"}],
               "tools": [{"type": "function", "function": {"name": "get_weather"}}],
               "max_tokens": 4}
    status, body = request(port, "POST", "/v1/chat/completions", body_in)
    obj = json.loads(body)
    choice = obj["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    ok &= status == 200 and len(calls) == 1
    ok &= choice["finish_reason"] == "tool_calls"
    ok &= choice["message"].get("content") is None
    if calls:
        args = json.loads(calls[0]["function"]["arguments"])
        ok &= calls[0]["function"]["name"] == "get_weather"
        ok &= args == {"location": "Paris", "days": 3}
        print("tools: %s finish=%s name=%s args=%s"
              % (status, choice["finish_reason"], calls[0]["function"]["name"], args))

    status, body = request(port, "POST", "/v1/chat/completions",
                           dict(body_in, stream=True))
    events = parse_sse(body)
    got = None
    finish = None
    for e in events[:-1]:
        chunk = json.loads(e)
        for c in chunk.get("choices", []):
            for tc in (c.get("delta") or {}).get("tool_calls", []):
                got = tc
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    ok &= got is not None and finish == "tool_calls"
    if got:
        ok &= json.loads(got["function"]["arguments"]) == {"location": "Paris", "days": 3}
    print("tools stream: finish=%s got=%s" % (finish, got["function"]["name"] if got else None))
    return bool(ok)


def check_reasoning(port):
    body_in = {"messages": [{"role": "user", "content": "think"}], "max_tokens": 4,
               "thinking": True}
    status, body = request(port, "POST", "/v1/chat/completions", body_in)
    obj = json.loads(body)
    msg = obj["choices"][0]["message"]
    ok = status == 200 and msg.get("reasoning_content") == "I think"
    ok &= msg.get("content") == "The answer"
    print("reasoning: %s reason=%r content=%r"
          % (status, msg.get("reasoning_content"), msg.get("content")))

    status, body = request(port, "POST", "/v1/chat/completions",
                           dict(body_in, stream=True))
    events = parse_sse(body)
    reason = ""
    content = ""
    for e in events[:-1]:
        for c in json.loads(e).get("choices", []):
            d = c.get("delta") or {}
            reason += d.get("reasoning_content", "") or ""
            content += d.get("content", "") or ""
    ok &= reason == "I think" and content == "The answer"
    print("reasoning stream: reason=%r content=%r" % (reason, content))
    return bool(ok)


def check_keepalive():
    """A slow prompt read must keep the stream alive with empty deltas."""
    import np_gemma.server as srv

    class SlowBackend(FakeBackend):
        def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
            time.sleep(1.0)
            for token in [10, 11]:
                yield token

    old = srv.KEEPALIVE_SECONDS
    srv.KEEPALIVE_SECONDS = 0.2
    httpd = serve(SlowBackend([]))
    try:
        status, body = request(httpd.server_address[1], "POST",
                               "/v1/chat/completions", {
                                   "messages": [{"role": "user", "content": "hi"}],
                                   "max_tokens": 2, "stream": True})
        chunks = [json.loads(e) for e in parse_sse(body)[:-1]]
        keep = 0
        content = ""
        for c in chunks:
            for ch in c.get("choices", []):
                d = ch.get("delta") or {}
                if not d and ch.get("finish_reason") is None:
                    keep += 1
                content += d.get("content", "") or ""
        ok = keep >= 2 and content == "KL"
        print("keepalive: sent=%d content=%r %s" % (keep, content, "ok" if ok else "FAIL"))
        return ok
    finally:
        srv.KEEPALIVE_SECONDS = old
        httpd.shutdown()
        httpd.server_close()


def check_stop():
    httpd = serve(FakeBackend([10, 11, 12, 13]))
    try:
        status, body = request(httpd.server_address[1], "POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stop": "M"})
        obj = json.loads(body)
        text = obj["choices"][0]["message"]["content"]
        finish = obj["choices"][0]["finish_reason"]
        ok = text == "KL" and finish == "stop"
        print("stop: text=%r finish=%s %s" % (text, finish, "ok" if ok else "FAIL"))
        return ok
    finally:
        httpd.shutdown()
        httpd.server_close()


def main():
    ok = check_sampling()
    ok = check_parse() and ok
    httpd = serve(FakeBackend([10, 11, 12]))
    try:
        ok = check_http(httpd.server_address[1]) and ok
    finally:
        httpd.shutdown()
        httpd.server_close()

    table = {40: '<|tool_call>call:get_weather{',
             41: 'location:<|"|>Paris<|"|>,days:3}<tool_call|>'}
    httpd = serve(FakeBackend([40, 41], table))
    try:
        ok = check_tools(httpd.server_address[1]) and ok
    finally:
        httpd.shutdown()
        httpd.server_close()

    # Split the channel name from the newline: a partial name must not leak.
    rtable = {50: '<|channel>', 51: 'thought', 52: '\nI think',
              53: '\n<channel|>The answer'}
    httpd = serve(FakeBackend([50, 51, 52, 53], rtable))
    try:
        ok = check_reasoning(httpd.server_address[1]) and ok
    finally:
        httpd.shutdown()
        httpd.server_close()

    ok = check_keepalive() and ok
    ok = check_stop() and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
