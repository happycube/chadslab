"""Check the OpenAI server end to end with a real model.

    python scripts/check_server_live.py --gguf PATH [--dtype int4]

The script loads the model, starts the server on a free port, and asks the
chat path for the capital of France with greedy selection. It prints the text
and the token ids of the answer.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import Model, Tokenizer  # noqa: E402
from np_gemma.config import Config  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma.server import Backend, make_server  # noqa: E402


def post(port, path, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    conn.request("POST", path, body=json.dumps(body).encode(),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read().decode("utf-8", "replace")
    conn.close()
    return resp.status, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dtype", default="int4")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype=args.dtype)
    backend = Backend(model, tok, cfg, model_id="gemma", temperature=0.0)
    httpd = make_server(backend, port=0, quiet=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    ok = True
    try:
        status, body = post(port, "/v1/chat/completions", {
            "model": "gemma",
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens, "temperature": 0})
        obj = json.loads(body)
        text = obj["choices"][0]["message"]["content"]
        print("chat   status=%d finish=%s" % (status, obj["choices"][0]["finish_reason"]))
        print("text   %r" % text)
        print("usage  %s" % obj["usage"])
        ok &= status == 200 and "Paris" in text

        status, body = post(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}})
        streamed = ""
        done = False
        usage = None
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                continue
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                streamed += (c.get("delta") or {}).get("content", "") or ""
        print("stream done=%s usage=%s" % (done, usage))
        print("stream text %r" % streamed)
        ok &= done and streamed == text

        status, body = post(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens, "temperature": 1.2,
            "top_k": 40, "top_p": 0.95, "seed": 3})
        obj = json.loads(body)
        print("sampled %r" % obj["choices"][0]["message"]["content"])

        status, body = post(port, "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "What is the weather in Paris? Use the tool."}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "Get the weather",
                "parameters": {"type": "object",
                               "properties": {"location": {"type": "string",
                                                           "description": "The city"}},
                               "required": ["location"]}}}],
            "max_tokens": 64, "temperature": 0})
        obj = json.loads(body)
        choice = obj["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        print("tools finish=%s calls=%s" % (choice["finish_reason"],
              [(c["function"]["name"], c["function"]["arguments"]) for c in calls]))
        print("tools text %r" % choice["message"].get("content"))
        ok &= len(calls) >= 1

        status, body = post(port, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "What is 17 times 24?"}],
            "max_tokens": 96, "temperature": 0, "thinking": True})
        obj = json.loads(body)
        msg = obj["choices"][0]["message"]
        print("reasoning %r" % (msg.get("reasoning_content") or "")[:160])
        print("answer %r" % msg.get("content"))
        ok &= bool(msg.get("reasoning_content"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        g.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
