#!/usr/bin/env python3
"""Golden answers of a running server: the check of a change of the server
code (REFACTOR_PLAN.md, P6). Greedy requests (temperature 0) over HTTP:

- a short answer with no think part, and the same as a stream (SSE);
- an answer with a think part;
- a tool call, then the turn with the tool result (the cache of the first
  turn is reused);
- a chat of three turns (each reuses the cache of the turn before);
- an image (--image: a model with --mmproj).

    python tests/golden/server_golden.py record --url http://127.0.0.1:8081/v1 --out A.json
    python tests/golden/server_golden.py compare A.json B.json

The API key comes from --key, else RECLANK_API_KEY. compare checks the
content, the reasoning, the tool calls, the finish reason, and the token
counts of each answer.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The current weather of a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "the city"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}


def post(url, key, body, stream=False):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=1800) as r:
        if not stream:
            return json.loads(r.read())
        content, reasoning, finish, usage = [], [], None, None
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning_content") or "")
                finish = ch.get("finish_reason") or finish
            usage = ev.get("usage") or usage
        return {"choices": [{"message": {"content": "".join(content),
                                         "reasoning_content": "".join(reasoning)},
                             "finish_reason": finish}], "usage": usage}


def answer(resp):
    ch = resp["choices"][0]
    m = ch["message"]
    u = resp.get("usage") or {}
    return {"content": m.get("content"), "reasoning": m.get("reasoning_content"),
            "tool_calls": [{"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
                           for t in (m.get("tool_calls") or [])],
            "finish": ch.get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}


def image_uri():
    """A small picture: a red circle on white, above a blue bar."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((64, 32, 192, 160), fill=(220, 30, 30))
    d.rectangle((32, 196, 224, 228), fill=(30, 60, 200))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def record(args):
    key = args.key or os.environ.get("RECLANK_API_KEY", "")
    url, model = args.url, args.model
    out = {}

    def run(name, body, stream=False):
        body = dict(body, model=model, temperature=0)
        t0 = time.time()
        r = post(url, key, body, stream)
        out[name] = answer(r)
        out[name]["seconds"] = round(time.time() - t0, 2)
        print("%-12s %5.1f s  %s" % (name, time.time() - t0, (out[name]["content"] or "")[:70].replace("\n", " ")),
              flush=True)
        return r

    q = [{"role": "user", "content": "Name three primary colors, one per line, and nothing else."}]
    run("short", dict(NO_THINK, messages=q, max_tokens=64))
    run("short_sse", dict(NO_THINK, messages=q, max_tokens=64, stream=True), stream=True)
    run("think", {"messages": [{"role": "user", "content":
                                "A train leaves at 9:40 and arrives at 13:05. How long is the trip? "
                                "Answer in one sentence."}], "max_tokens": 1200})
    msgs = [{"role": "user", "content": "What is the weather in Lisbon right now?"}]
    r = run("tool", dict(NO_THINK, messages=msgs, tools=TOOLS, max_tokens=256))
    m = r["choices"][0]["message"]
    if m.get("tool_calls"):
        msgs = msgs + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"]},
                       {"role": "tool", "tool_call_id": m["tool_calls"][0].get("id", "call_0"),
                        "content": json.dumps({"city": "Lisbon", "temperature": 21, "unit": "celsius",
                                               "sky": "clear"})}]
        run("tool_result", dict(NO_THINK, messages=msgs, tools=TOOLS, max_tokens=128))
    chat = [{"role": "user", "content": "Pick a number from 1 to 10 and tell me only the number."}]
    for i, follow in enumerate(["Now double it. Only the number.", "Now subtract three. Only the number."]):
        r = run("chat%d" % (i + 1), dict(NO_THINK, messages=chat, max_tokens=32))
        chat = chat + [{"role": "assistant", "content": r["choices"][0]["message"].get("content") or ""},
                       {"role": "user", "content": follow}]
    run("chat3", dict(NO_THINK, messages=chat, max_tokens=32))
    if args.image:
        run("image", dict(NO_THINK, messages=[{"role": "user", "content": [
            {"type": "text", "text": "Describe this picture in one short sentence."},
            {"type": "image_url", "image_url": {"url": image_uri()}}]}], max_tokens=64))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


def compare(args):
    a, b = json.load(open(args.a)), json.load(open(args.b))
    bad = 0
    for name in a:
        if name not in b:
            print("%-12s missing" % name)
            bad += 1
            continue
        diff = [k for k in a[name] if k != "seconds" and a[name][k] != b[name].get(k)]
        if diff:
            bad += 1
            print("%-12s DIFFERS in %s" % (name, ", ".join(diff)))
            for k in diff:
                print("    %s: %r\n    %s  %r" % (k, a[name][k], " " * len(k), b[name].get(k)))
        else:
            print("%-12s same (%s s -> %s s)" % (name, a[name]["seconds"], b[name]["seconds"]))
    print("%d of %d answers differ" % (bad, len(a)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--url", required=True)
    r.add_argument("--key", default=None)
    r.add_argument("--model", default="x")
    r.add_argument("--image", action="store_true")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
