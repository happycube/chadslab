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


def raw_request(port, method, path, headers, payload=b""):
    """A request with the headers as given (no Content-Type added)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.putrequest(method, path)
    for k, v in headers.items():
        conn.putheader(k, v)
    conn.endheaders(payload or None)
    resp = conn.getresponse()
    data = resp.read()
    hdrs = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, hdrs, data.decode("utf-8", "replace")


def check_security():
    """The API key, the CORS headers, the limit of the body, and the raw
    answers only with RAW_DIR."""
    import tempfile

    import np_gemma.server as srv
    ok = True
    httpd = make_server(FakeBackend([10, 11]), port=0, quiet=True, api_key="test-key",
                        max_body=4096)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}).encode()
    js = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    tests = [
        ("no key: 401", "POST", js, body, 401),
        ("wrong key: 401", "POST", dict(js, Authorization="Bearer nope"), body, 401),
        ("Bearer key: 200", "POST", dict(js, Authorization="Bearer test-key"), body, 200),
        ("x-api-key: 200", "POST", dict(js, **{"x-api-key": "test-key"}), body, 200),
        ("negative Content-Length: 400", "POST",
         {"Authorization": "Bearer test-key", "Content-Length": "-1"}, b"", 400),
        ("Content-Length over the limit: 413", "POST",
         {"Authorization": "Bearer test-key", "Content-Length": "999999999"}, b"", 413),
    ]
    try:
        for name, method, hdrs, payload, want in tests:
            status, rh, _ = raw_request(port, method, "/v1/chat/completions", hdrs, payload)
            good = status == want and "access-control-allow-origin" not in rh
            print("  %-36s %s" % (name, "ok" if good else "FAIL (%d)" % status))
            ok &= good
        status, _, _ = raw_request(port, "GET", "/v1/models", {})
        status2, _, _ = raw_request(port, "GET", "/health", {})
        good = status == 401 and status2 == 200
        print("  %-36s %s" % ("models needs the key, health not", "ok" if good else "FAIL"))
        ok &= good
    finally:
        httpd.shutdown()
        httpd.server_close()
    # CORS only for --cors-origin
    httpd = make_server(FakeBackend([10]), port=0, quiet=True, cors_origin="http://x.test")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        status, rh, _ = raw_request(httpd.server_address[1], "OPTIONS", "/v1/chat/completions", {})
        good = rh.get("access-control-allow-origin") == "http://x.test"
        print("  %-36s %s" % ("CORS for the given origin", "ok" if good else "FAIL"))
        ok &= good
    finally:
        httpd.shutdown()
        httpd.server_close()
    # the raw answers: no file without RAW_DIR, owner-only files with it
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            srv.RAW_DIR = None
            srv.log_answer(1, 1, 0, 0, "raw")
            srv.log_empty(1, "raw")
            none = os.listdir(d) == []
            sub = os.path.join(d, "dbg")
            os.mkdir(sub)
            srv.RAW_DIR = sub
            srv.log_answer(1, 1, 0, 0, "raw")
            srv.log_empty(1, "raw")
            srv.log_empty(1, "raw")
            files = os.listdir(sub)
            modes = {f: os.stat(os.path.join(sub, f)).st_mode & 0o777 for f in files}
            good = none and len(files) == 3 and all(m == 0o600 for m in modes.values())
        finally:
            srv.RAW_DIR = None
            os.chdir(cwd)
    print("  %-36s %s" % ("raw answers only with RAW_DIR, 0600", "ok" if good else "FAIL %s" % modes))
    ok &= good
    return ok


def check_limits():
    """max_tokens below 1 and not a number get 400; a floor of the backend
    gives X-Max-Tokens-Applied; a stream of /v1/completions keeps alive from
    its start; a long stream (the parse every few tokens) gives the whole
    text."""
    import np_gemma.server as srv
    ok = True

    class FloorBackend(FakeBackend):
        def applied_max_tokens(self, n):
            return max(n, 5) if n >= 3 else n

    httpd = serve(FloorBackend([10, 11, 12, 13, 14, 15]))
    port = httpd.server_address[1]
    msgs = [{"role": "user", "content": "hi"}]
    try:
        for name, mt, want in (("max_tokens 0: 400", 0, 400), ("max_tokens 'x': 400", "x", 400)):
            status, _ = request(port, "POST", "/v1/chat/completions", {"messages": msgs, "max_tokens": mt})
            print("  %-36s %s" % (name, "ok" if status == want else "FAIL (%d)" % status))
            ok &= status == want
        body = json.dumps({"messages": msgs, "max_tokens": 3}).encode()
        status, hdrs, text = raw_request(port, "POST", "/v1/chat/completions",
                                         {"Content-Type": "application/json",
                                          "Content-Length": str(len(body))}, body)
        good = (status == 200 and hdrs.get("x-max-tokens-applied") == "5"
                and hdrs.get("x-max-tokens-requested") == "3"
                and json.loads(text)["choices"][0]["message"]["content"] == "KLMNO")
        print("  %-36s %s" % ("a floor: X-Max-Tokens-Applied", "ok" if good else "FAIL %s" % hdrs))
        ok &= good
        status, hdrs, _ = raw_request(port, "POST", "/v1/chat/completions",
                                      {"Content-Type": "application/json", "Content-Length": str(len(
                                          json.dumps({"messages": msgs, "max_tokens": 2})))},
                                      json.dumps({"messages": msgs, "max_tokens": 2}).encode())
        good = status == 200 and "x-max-tokens-applied" not in hdrs
        print("  %-36s %s" % ("no floor: no header", "ok" if good else "FAIL"))
        ok &= good
    finally:
        httpd.shutdown()
        httpd.server_close()

    class SlowBackend(FakeBackend):
        def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
            time.sleep(0.6)
            for token in [10, 11]:
                yield token
    old = srv.KEEPALIVE_SECONDS
    srv.KEEPALIVE_SECONDS = 0.2
    httpd = serve(SlowBackend([]))
    try:
        status, body = request(httpd.server_address[1], "POST", "/v1/completions",
                               {"prompt": "hi", "max_tokens": 2, "stream": True})
        text = "".join(json.loads(e)["choices"][0]["text"] for e in parse_sse(body)[:-1]
                       if json.loads(e).get("choices"))     # the form of the completions stream
        good = status == 200 and text == "KL"
        print("  %-36s %s" % ("a stream of /v1/completions", "ok" if good else "FAIL %r" % body[:200]))
        ok &= good
    finally:
        srv.KEEPALIVE_SECONDS = old
        httpd.shutdown()
        httpd.server_close()

    n = 5000
    ids = [int(x) for x in np.random.default_rng(3).integers(0, 26, n)]   # no loop in it
    httpd = serve(FakeBackend(ids))
    try:
        want = "".join(chr(65 + i) for i in ids)
        status, body = request(httpd.server_address[1], "POST", "/v1/chat/completions",
                               {"messages": msgs, "max_tokens": n, "stream": True})
        chunks = [json.loads(e) for e in parse_sse(body)[:-1]]
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
        good = text == want
        print("  %-36s %s" % ("a stream of %d tokens" % n, "ok" if good else "FAIL (%d chars)" % len(text)))
        ok &= good
    finally:
        httpd.shutdown()
        httpd.server_close()
    return ok


def check_hooks():
    """The handler parses with the parse_output of the backend (the Qwen
    backend has its own), in a stream and in a whole answer."""
    from np_gemma.chat import parse_output as base

    class HookBackend(FakeBackend):
        def parse_output(self, text):
            p = base(text)
            return dict(p, content=p["content"].lower())

    httpd = serve(HookBackend([10, 11, 12]))
    ok = True
    try:
        msgs = [{"role": "user", "content": "hi"}]
        for stream in (False, True):
            status, body = request(httpd.server_address[1], "POST", "/v1/chat/completions",
                                   {"messages": msgs, "max_tokens": 3, "stream": stream})
            if stream:
                text = "".join(json.loads(e)["choices"][0]["delta"].get("content") or ""
                               for e in parse_sse(body)[:-1] if json.loads(e).get("choices"))
            else:
                text = json.loads(body)["choices"][0]["message"]["content"]
            good = text == "klm"
            print("  %-36s %s" % ("the parse of the backend (%s)" % ("stream" if stream else "whole"),
                                  "ok" if good else "FAIL %r" % text))
            ok &= good
    finally:
        httpd.shutdown()
        httpd.server_close()
    return ok


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

    # The prompt ends in an open thought channel (the Gemma 4 template after
    # a tool result): the answer starts in the thought with no opener.
    otable = {60: '<|channel>', 61: 'thought\n', 62: 'I think',
              63: '\n<channel|>The answer'}
    backend = FakeBackend([62, 63], otable)
    backend.prompt_ids = lambda messages, thinking=None, tools=None: [1, 60, 61]
    httpd = serve(backend)
    try:
        print("the prompt opens the thought channel:")
        ok = check_reasoning(httpd.server_address[1]) and ok
    finally:
        httpd.shutdown()
        httpd.server_close()

    ok = check_keepalive() and ok
    ok = check_stop() and ok
    print("the key, CORS, the body, the raw answers:")
    ok = check_security() and ok
    print("max_tokens, the stream of completions, a long stream:")
    ok = check_limits() and ok
    print("the hooks of the backend:")
    ok = check_hooks() and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
