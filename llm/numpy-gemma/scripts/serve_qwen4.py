#!/usr/bin/env python3
"""Run an OpenAI compatible server for Qwen3.8-Flash-Next (the GGUF file of
scripts/convert_nvfp4_gguf.py) on the GPU, or on the CPU.

    python scripts/serve_qwen4.py [--ctx 98304] [--port 8081] [--hot-gb 0.5]

Point an OpenAI client at http://127.0.0.1:8081/v1 . The HTTP part is that of
np_gemma/server.py. This file gives it the Qwen parts:

- the chat template of Qwen (ChatML), and the stop tokens <|im_end|> and
  <|endoftext|>;
- the <think> part of the answer as the reasoning (reasoning_content): the
  text goes to the channel form of the Gemma output that the parser reads;
- one cache of --ctx tokens. A prompt that starts with all the tokens in the
  cache (the next turn of the same chat) reads only its new tokens; any other
  prompt starts again at position 0.

Tools are not supported. --thinking opens the think part by default; a
request can set "thinking".
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402
from np_gemma.server import Backend, content_text, serve  # noqa: E402

MODEL = "models2/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-bf16.gguf"
EMPTY_THINK = "<think>\n\n</think>\n\n"


class QwenTok:
    """The tokenizer interface of np_gemma/server.py on QwenTokenizer."""

    def __init__(self, path):
        self.t = QwenTokenizer(path)
        self.stop_ids = list(self.t.stop_ids)
        self.think_open = False     # the prompt of the request ends in <think>

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
        out = []
        for m in messages:
            role, content = m["role"], m["content"] or ""
            if role == "assistant":
                # the form of the answers of this server (no think part), so
                # the next turn starts with the tokens of the cache
                content = EMPTY_THINK + content
            out.append("<|im_start|>%s\n%s<|im_end|>\n" % (role, content))
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n" + ("<think>\n" if thinking else EMPTY_THINK))
        self.think_open = bool(thinking)
        return "".join(out)


class QwenBackend(Backend):
    """Backend with Qwen4GPU (or Qwen4CPU) and one cache of ctx tokens."""

    def __init__(self, model, dev, tok, ctx, **kw):
        super().__init__(model, tok, model_id=kw.pop("model_id"), **kw)
        self.dev = dev
        self.ctx = ctx
        self.cache = None
        self.ids = []           # the tokens in the cache

    def prompt_ids(self, messages, thinking=None, tools=None):
        msgs = [{"role": m.get("role", "user"), "content": content_text(m.get("content"))}
                for m in messages]
        think = self.thinking if thinking is None else bool(thinking)
        return self.tokenizer.encode(self.tokenizer.apply_chat_template(msgs, thinking=think))

    def completion_ids(self, prompt):
        self.tokenizer.think_open = False
        return self.tokenizer.encode(prompt)

    def _fresh(self):
        self.cache = Qwen4Cache(self.model.cfg, self.ctx)
        if self.dev is not None:
            self.dev.attach(self.cache)
        self.ids = []

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
        if self.cache is None or k == 0 or k >= n or prompt_ids[:k] != self.ids:
            self._fresh()
            k = 0
        t0 = time.time()
        logits = self._prompt(prompt_ids[k:], k)
        dt = time.time() - t0
        self.ids = list(prompt_ids)
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

    tok = QwenTok(args.tok or os.path.join(os.path.dirname(path), "tokenizer.json"))
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
