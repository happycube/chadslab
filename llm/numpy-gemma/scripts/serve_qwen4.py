#!/usr/bin/env python3
"""Run an OpenAI compatible server for Qwen3.8-Flash-Next (the GGUF file of
scripts/convert_nvfp4_gguf.py) on the GPU, or on the CPU.

    python scripts/serve_qwen4.py [--ctx 98304] [--port 8081] [--hot-gb 0.5]

Point an OpenAI client at http://127.0.0.1:8081/v1 . The HTTP part is that of
np_gemma/server.py. This file gives it the Qwen parts:

- the chat template of the model (chat_template.jinja, with jinja2): the
  tools, the tool results, and the tool calls of the earlier turns; the stop
  tokens <|im_end|> and <|endoftext|>;
- the <think> part of the answer as the reasoning (reasoning_content): the
  text goes to the channel form of the Gemma output that the parser reads;
- the tool calls of the answer (<tool_call><function=NAME><parameter=P>
  value</parameter>...) as the tool_calls of the OpenAI API, with the types
  of the schema of each tool;
- one cache of --ctx tokens. A prompt that starts with all the tokens in the
  cache reads only its new tokens. At the end of each prompt the server keeps
  the recurrent state (DeltaNet, the n-gram layer): a prompt that starts
  with the last prompt (the next turn of an agent, whose earlier answer
  comes back in the form of the template) reads only the tokens after it.
  Any other prompt starts again at position 0.

--thinking opens the think part by default; a request can set "thinking".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402
from np_gemma import server as S  # noqa: E402
from np_gemma.chat import parse_output as gemma_parse_output  # noqa: E402
from np_gemma.server import Backend, serve  # noqa: E402

MODEL = "models2/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-bf16.gguf"


class QwenTok:
    """The tokenizer interface of np_gemma/server.py on QwenTokenizer, with
    the chat template of the model."""

    def __init__(self, path, template):
        import jinja2
        self.t = QwenTokenizer(path)
        self.stop_ids = list(self.t.stop_ids)
        self.think_open = False     # the prompt of the request ends in <think>
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                 extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
        env.filters["items"] = lambda d: list(d.items()) if isinstance(d, dict) else []

        def raise_exception(msg):
            raise ValueError(msg)
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(open(template, encoding="utf-8").read())

    def encode(self, text):
        return self.t.encode(text)

    def decode(self, ids, skip_special_tokens=False):
        text = self.t.decode(ids)
        # The think part to the channel of the Gemma output (np_gemma/chat.py,
        # parse_output): the reasoning, then the answer.
        if self.think_open:
            text = "<|channel>thought\n" + text
        text = text.replace("<think>", "<|channel>thought\n").replace("</think>", "<channel|>")
        return text

    def apply_chat_template(self, messages, add_generation_prompt=True, thinking=False,
                            tools=None, empty_thought_block=True):
        text = self.template.render(messages=messages, tools=tools or None,
                                    add_generation_prompt=add_generation_prompt,
                                    enable_thinking=bool(thinking))
        self.think_open = bool(thinking)
        return text


# The tool calls of the answer (the form of the template).
TOOLS = {}      # the tools of the request: name -> the properties of its parameters
CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
OPEN = "<tool_call>"


def _value(text, prop):
    """A parameter in the type of the schema: a string as it is, else JSON."""
    if (prop or {}).get("type") == "string":
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def qwen_parse_output(text):
    """parse_output of np_gemma/chat.py (the reasoning), and the tool calls of
    Qwen. A tool call that is not complete is pending: a stream holds it."""
    p = gemma_parse_output(text)
    calls = []

    def call(m):
        name, props = m.group(1), TOOLS.get(m.group(1), {})
        args = {k: _value(v, props.get(k)) for k, v in PARAM.findall(m.group(2))}
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        return ""
    body = CALL.sub(call, p["content"])
    pending = ""
    i = body.find(OPEN)
    if i >= 0:
        body, pending = body[:i], body[i:]
    else:
        for k in range(min(len(body), len(OPEN) - 1), 0, -1):
            if OPEN.startswith(body[-k:]):
                body, pending = body[:-k], body[-k:]
                break
    return {"reasoning": p["reasoning"], "content": body.strip(),
            "tool_calls": p["tool_calls"] + calls, "pending": pending}


# np_gemma/server.py calls these by name
S.parse_output = qwen_parse_output
S.tool_done = lambda text: False


def _message(m):
    """A message of the request for the template: the arguments of a tool
    call as a dict."""
    msg = dict(m)
    msg["role"] = m.get("role", "user")
    calls = []
    for c in m.get("tool_calls") or []:
        c = json.loads(json.dumps(c))
        f = c.get("function", c)
        if isinstance(f.get("arguments"), str):
            try:
                f["arguments"] = json.loads(f["arguments"]) if f["arguments"].strip() else {}
            except ValueError:
                f["arguments"] = {}
        calls.append(c)
    if calls:
        msg["tool_calls"] = calls
    return msg


class QwenBackend(Backend):
    """Backend with Qwen4GPU (or Qwen4CPU) and one cache of ctx tokens."""

    def __init__(self, model, dev, tok, ctx, **kw):
        super().__init__(model, tok, model_id=kw.pop("model_id"), **kw)
        self.dev = dev
        self.ctx = ctx
        self.cache = None
        self.ids = []           # the tokens in the cache
        self.snap_ids = None    # the last prompt, and the recurrent state after it
        self.snap = None

    def prompt_ids(self, messages, thinking=None, tools=None):
        TOOLS.clear()
        for tl in tools or []:
            f = tl.get("function", tl)
            TOOLS[f.get("name")] = (f.get("parameters") or {}).get("properties") or {}
        think = self.thinking if thinking is None else bool(thinking)
        text = self.tokenizer.apply_chat_template([_message(m) for m in messages], thinking=think,
                                                  tools=tools)
        return self.tokenizer.encode(text)

    def completion_ids(self, prompt):
        self.tokenizer.think_open = False
        return self.tokenizer.encode(prompt)

    def _fresh(self):
        self.cache = Qwen4Cache(self.model.cfg, self.ctx)
        if self.dev is not None:
            self.dev.attach(self.cache)
        self.ids = []
        self.snap_ids = self.snap = None

    def _state(self):
        """The arrays of the recurrent state: DeltaNet (conv, state) and the
        convolution of the n-gram layer. The keys and values and the keys of
        the indexer are per position: a later prompt writes them again."""
        c = self.cache
        return list(c.conv.values()) + list(c.state.values()) + list(c.ple_conv.values())

    def _save(self, ids, logits):
        snap = []
        for a in self._state():
            h = np.empty_like(a)
            if self.dev is not None:
                self.dev.cache_dev.bufs[id(a)][1].download(h)
            else:
                h[...] = a
            snap.append(h)
        self.snap, self.snap_ids = (snap, self.cache.ple_ids.copy(), logits.copy()), list(ids)

    def _restore(self):
        snap, ple_ids, logits = self.snap
        for a, h in zip(self._state(), snap):
            if self.dev is not None:
                self.dev.cache_dev.bufs[id(a)][1].upload(h)
            else:
                a[...] = h
        self.cache.ple_ids = ple_ids.copy()
        self.cache.n = len(self.snap_ids)
        self.ids = list(self.snap_ids)
        return logits

    def _prompt(self, ids, pos):
        """Read ids from pos. Return the logits of the last token."""
        if self.dev is not None:
            for c0 in range(0, len(ids), 16384):
                self.dev.prefill(ids[c0:c0 + 16384], pos=pos + c0)
            return np.asarray(self.dev.logits()).reshape(-1)
        h = None
        for c0 in range(0, len(ids), 16384):
            h = self.model.forward(ids[c0:c0 + 16384], self.cache, start_pos=pos + c0)
        return self.model.logits(h[-1:])[0]

    def _step(self, token, pos):
        if self.dev is not None:
            self.dev.step(token, pos)
            return np.asarray(self.dev.logits()).reshape(-1)
        return self.model.logits(self.model.forward([token], self.cache, start_pos=pos))[0]

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        n = len(prompt_ids)
        if n + 1 > self.ctx:
            raise ValueError("the prompt has %d tokens; the context is %d" % (n, self.ctx))
        max_tokens = min(max_tokens, self.ctx - n - 1)
        k = len(self.ids)
        s = len(self.snap_ids) if self.snap_ids is not None else 0
        if self.cache is not None and 0 < k < n and prompt_ids[:k] == self.ids:
            pass                                            # the cache as it is
        elif self.cache is not None and 0 < s <= n and prompt_ids[:s] == self.snap_ids:
            logits = self._restore()                        # the state after the last prompt
            k = s
        else:
            self._fresh()
            k = 0
        t0 = time.time()
        if k < n:
            logits = self._prompt(prompt_ids[k:], k)
            self.ids = list(prompt_ids)
            self._save(prompt_ids, logits)
        dt = time.time() - t0
        print("[qwen4] reuse=%d new=%d prompt %.1f s (%.0f tok/s)" % (
            k, n - k, dt, (n - k) / max(dt, 1e-9)), file=sys.stderr, flush=True)
        sampler.reset(prompt_ids)
        pos, count, t0 = n, 0, time.time()
        try:
            while count < max_tokens:
                token = int(sampler(logits))
                count += 1
                yield token
                if token in eos_ids or count >= max_tokens:
                    break
                logits = self._step(token, pos)
                self.ids.append(token)
                pos += 1
        finally:
            dt = time.time() - t0
            print("[qwen4] decode %d tokens in %.1f s (%.2f tok/s); %d tokens in the cache" % (
                count, dt, count / max(dt, 1e-9), len(self.ids)), file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-m", "--model", default=MODEL)
    ap.add_argument("--tok", default=None, help="tokenizer.json (default: next to the model)")
    ap.add_argument("--template", default=None,
                    help="chat_template.jinja (default: next to the model, else in the checkpoint)")
    ap.add_argument("--ctx", type=int, default=98304, help="the tokens of the context")
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--hot-gb", type=float, default=0.5, help="gpu: GB of hot experts")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--model-id", default="qwen3.8-flash-next")
    ap.add_argument("--max-tokens", type=int, default=4096, help="the default of a request")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--thinking", action="store_true", help="open the think part by default")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = args.model if os.path.isabs(args.model) else os.path.join(root, args.model)

    template = args.template or os.path.join(os.path.dirname(path), "chat_template.jinja")
    if not os.path.exists(template):
        template = os.path.join(root, "models/Qwen3.8-Flash-Next-NVFP4/chat_template.jinja")
    tok = QwenTok(args.tok or os.path.join(os.path.dirname(path), "tokenizer.json"), template)
    print("loading %s ..." % path, flush=True)
    t0 = time.time()
    model = Qwen4CPU(path)
    dev = None
    if args.backend == "gpu":
        from np_gemma.qwen4_gpu import Qwen4GPU
        dev = Qwen4GPU(model, hot_gb=args.hot_gb)
    backend = QwenBackend(model, dev, tok, args.ctx, model_id=args.model_id,
                          thinking=args.thinking, max_tokens=args.max_tokens,
                          temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    # the cache and the programs of a prompt now, so the first request does
    # not wait for them and a context that does not fit fails here
    backend._fresh()
    if dev is not None:
        backend._prompt(tok.encode("<|im_start|>user\nHello<|im_end|>\n") * 160, 0)
        backend._fresh()
    print("ready in %.0f s: dense %s, context %d%s; sampling temperature=%s top_k=%s top_p=%s" % (
        time.time() - t0, model.dense, args.ctx,
        ", %d hot experts in each layer" % dev.n_slots if dev else "",
        args.temperature, args.top_k, args.top_p), flush=True)
    serve(backend, host=args.host, port=args.port, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    np.seterr(over="ignore")
    raise SystemExit(main())
