"""Nemotron 3.5 Bolt in the server (reclank/models/base.py, P3 of docs/NEMOTRON_BOLT_PLAN.md): the
model on the CPU (NemotronKQ), the chat of the checkpoint's template (the Qwen kind: QwenTok,
qwen_parse_output for the <tool_call><function=...> calls), and the contexts of a few
conversations between the turns (core/session.py, RowCache).

The points of a context: the Mamba layers' state (the conv rows and the SSM state, 170 MB) cannot
go back, so a point holds it; it is taken before the last <|im_start|> of each prompt (the
assistant turn of this request), where the next turn's prompt goes on (the template rewrites the
earlier answer: its reasoning dropped once a later user message comes). The keys and values of the
8 attention layers are the rows of the context (HostRows). A prompt that goes on from all the
tokens the cache read (the last answer as the model wrote it: an agent's tool loop) needs no point.

Thinking: the template's enable_thinking (on by default); low and medium efforts set its
medium_effort (the last user message gets "{reasoning effort: efficient}").
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from reclank.core.session import HostRows, RowCache
from reclank.core.think_guard import think_left_open
from reclank.models.base import Engine
from reclank.models.qwen.chat import QwenTok, TOOLS, _escaped, _message, _plain, qwen_parse_output, thinking_of

FAMILY = "nemotron"
MODEL_TYPES = ("nemotron_h_omni", "nemotron_h_puzzle", "nemotron_h")
THOUGHT_OPEN = "<|channel>thought\n"     # the opener of the parser (models/gemma4/chat.py)


def detect(meta):
    t = meta.get("model_type")
    if t in MODEL_TYPES:
        return True
    return (meta.get("llm_config") or {}).get("model_type") in MODEL_TYPES


class NemotronTok(QwenTok):
    """QwenTok with the switches of the Nemotron template (enable_thinking, medium_effort)."""

    def apply_chat_template(self, messages, add_generation_prompt=True, thinking=False, tools=None,
                            empty_thought_block=True, effort=None, preserve=None):
        kw = {}
        if thinking and effort in ("low", "medium"):
            kw["medium_effort"] = True
        if preserve is not None:
            kw["truncate_history_thinking"] = not preserve
        return self.template.render(messages=messages, tools=tools or None,
                                    add_generation_prompt=add_generation_prompt,
                                    enable_thinking=bool(thinking), bos_token=self.bos_token, **kw)


class CpuRunner:
    """The model on the CPU (NemotronKQ) with its cache, for the engine."""

    def __init__(self, m, ctx):
        self.m, self.cache = m, m.cache(ctx)

    @property
    def n(self):
        return self.cache.n

    def forward(self, ids):
        return self.m.forward(ids, self.cache, logits="last")[0]

    def rows(self):
        c = self.cache
        return [HostRows(a, 1) for i in sorted(c.K) for a in (c.K[i], c.V[i])]

    def save(self):
        c = self.cache
        return {"conv": {i: a.copy() for i, a in c.conv.items()}, "S": {i: a.copy() for i, a in c.S.items()}}

    def restore(self, st, k):
        c = self.cache
        for i in c.conv:
            c.conv[i][:] = st["conv"][i]
            c.S[i][:] = st["S"][i]
        c.n = k

    def go_on(self, k):
        self.cache.n = k

    def fresh(self):
        self.restore({"conv": {i: 0 for i in self.cache.conv}, "S": {i: 0 for i in self.cache.S}}, 0)


class GpuRunner:
    """The model on the GPU (NemotronGPU: its cache on the device), for the engine."""

    def __init__(self, g):
        self.g = g
        g.reset()

    @property
    def n(self):
        return self.g.cache.n

    def forward(self, ids):
        return self.g.prefill(ids) if len(ids) > 1 else self.g.step(ids[0])

    def _buf(self, a):
        return self.g.cache_dev.buf(a)

    def rows(self):
        from reclank.core.session import BufferRows
        c = self.g.cache
        return [BufferRows(self._buf(a), 1, a.nbytes // a.shape[0]) for i in sorted(c.K) for a in (c.K[i], c.V[i])]

    def save(self):
        c = self.g.cache
        st = {"conv": {}, "S": {}}
        for name, d in (("conv", c.conv), ("S", c.S)):
            for i, a in d.items():
                self._buf(a).download(a)
                st[name][i] = a.copy()
        return st

    def restore(self, st, k):
        c = self.g.cache
        for name, d in (("conv", c.conv), ("S", c.S)):
            for i, a in d.items():
                a[:] = st[name][i]
                self._buf(a).upload(a)
        c.n = k

    def go_on(self, k):
        self.g.cache.n = k

    def fresh(self):
        self.g.reset()


class NemotronEngine(Engine):
    family = FAMILY

    def __init__(self, run, tok, ctx, contexts=4, contexts_gb=16.0):
        self.run = run                  # CpuRunner or GpuRunner
        self.tokenizer = tok
        self.max_context = ctx
        self.ctxs = RowCache(self, contexts, contexts_gb * 1e9, "nemotron")
        self.as_is = []                 # the keys the cache has read (the prompt, the answer)
        self.last_cached = 0
        self.last_text = ""
        self.im_start = tok.t.special["<|im_start|>"]

    # ---- the prompt

    def think_level(self, req, default):
        return thinking_of(req, default)

    def think_off(self):
        return None

    def prompt_ids(self, messages, level, tools=None, preserve=None, source=None):
        TOOLS.clear()
        for tl in tools or []:
            f = tl.get("function", tl)
            TOOLS[f.get("name")] = (f.get("parameters") or {}).get("properties") or {}
        msgs = [_message(m) for m in messages]
        for m in msgs:
            if isinstance(m.get("content"), list):
                if any(isinstance(p, dict) and p.get("type") not in ("text", None) for p in m["content"]):
                    raise ValueError("this model takes text only (the vision tower is P7 of the plan)")
                m["content"] = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in m["content"])
        esc = self.tokenizer.t.escape
        text = self.tokenizer.apply_chat_template(_escaped(msgs, esc), thinking=level is not None,
                                                  tools=_escaped(tools, esc), effort=level, preserve=preserve)
        self.last_text = text
        return self.tokenizer.encode(text)

    def open_channel(self, prompt_ids):
        return THOUGHT_OPEN if think_left_open(prompt_ids, self.tokenizer.t.special) else ""

    def parse_output(self, text):
        return qwen_parse_output(text)

    def clean_raw(self, raw):
        return _plain(raw)

    def turn_fields(self, rec):
        return {"prompt_text": self.tokenizer.t.unescape(self.last_text),
                "raw": self.clean_raw(rec.get("raw", "")), "cached_tokens": self.last_cached}

    # ---- the contexts (core/session.py, RowCache)

    def rows(self):
        return self.run.rows()

    def reserve(self, n):
        pass

    def _save(self, keys):
        self.ctxs.point(keys, self.run.save())

    def _restore(self, st, k):
        self.run.restore(st, k)

    def _fresh(self):
        self.run.fresh()

    def _cut(self, ids):
        """The position of the last <|im_start|> of the prompt (the start the next turn keeps)."""
        return max((i for i, t in enumerate(ids) if t == self.im_start), default=0)

    # ---- the generation

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        ids = list(prompt_ids)
        n = len(ids)
        if n + 1 > self.max_context:
            raise ValueError("the prompt has %d tokens; the context is %d" % (n, self.max_context))
        max_tokens = min(int(max_tokens), self.max_context - n - 1)
        t0 = time.time()
        k, point = self.ctxs.begin(ids, lambda st: False, as_is=self.as_is)
        if k == 0:
            self._fresh()
        elif point is not None:
            self._restore(point, k)
        else:
            self.run.go_on(k)               # go on from what the cache read
        self.as_is = []
        cut = self._cut(ids)
        if k < cut < n:
            self.run.forward(ids[k:cut])
            self._save(ids[:cut])
            k0 = cut
        else:
            k0 = k
        lg = self.run.forward(ids[k0:])
        dt = time.time() - t0
        self.last_cached = k
        print("[nemotron] cache hit %d of %d prompt tokens; new %d in %.1f s (%.0f tok/s); %s" % (
            k, n, n - k, dt, (n - k) / max(dt, 1e-9), self.ctxs.note(n, k)), file=sys.stderr, flush=True)
        sampler.reset(prompt_ids)
        t1, out = time.time(), []
        try:
            y = sampler(lg)
            while True:
                out.append(y)
                yield y
                if y in eos_ids or len(out) >= max_tokens:
                    break
                y = sampler(self.run.forward([y]))
        finally:
            # the cache has read the prompt and the answer but its last token
            self.as_is = ids + out[:-1]
            dt = time.time() - t1
            print("[nemotron] decode %d tokens in %.1f s (%.1f tok/s)" % (len(out), dt, len(out) / max(dt, 1e-9)),
                  file=sys.stderr, flush=True)


def load(opts, meta, worker=None):
    """Load the model with the server options (server/options.py), on the model thread. Return
    (engine, defaults)."""
    from reclank.models.nemotron.kq import NemotronKQ
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    path = opts.model
    t0 = time.time()
    print("loading %s ..." % path, flush=True)
    template = opts.template or os.path.join(path, "chat_template.jinja")
    tok = NemotronTok(opts.tokenizer or os.path.join(path, "tokenizer.json"), template)
    m = NemotronKQ(path)
    ctx = int(opts.ctx or 65536)
    backend = getattr(opts, "backend", "auto") or "auto"
    gpu = None
    if backend != "cpu":
        from reclank.models.nemotron.gpu import task_threads
        os.environ.setdefault("RECLANK_GPU_CPU_THREADS", str(task_threads()))
        try:
            from reclank.backends.cuda.runtime import lib
            lib()
            gpu = True
        except Exception as exc:        # no GPU, no CUDA: the CPU engine
            if backend == "cuda":
                raise
            print("no GPU (%s): the CPU engine" % exc, flush=True)
    if gpu:
        from reclank.models.nemotron.gpu import NemotronGPU, load_counts
        g = NemotronGPU(m, ctx=ctx, hot_gb=opts.hot_gb, counts=load_counts(path))
        run = GpuRunner(g)
        where = "GPU, %d%% of the experts hot (%.1f GB); the host link %.1f GB/s: a prompt's cold experts on the %s" % (
            round(100 * g.hot_share), g.hot_bytes / 1e9, g.link_gbps, "GPU (mapped)" if g.zc_tables else "CPU")
    else:
        run = CpuRunner(m, ctx)
        where = "CPU, tier %s" % __import__("reclank.backends.cpu.cops", fromlist=["TIER"]).TIER
    eng = NemotronEngine(run, tok, ctx, contexts=getattr(opts, "contexts", 4),
                         contexts_gb=getattr(opts, "contexts_gb", 16.0))
    # the experts of every layer and the programs of a prompt and of a step now
    warm = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Hello. " * 40}], thinking=False))
    run.forward(warm)
    run.forward([warm[-1]])
    eng._fresh()
    thinking = opts.thinking or "high"
    level = None if thinking == "off" else thinking_of({"reasoning_effort": thinking}, "xhigh")
    defaults = {"temperature": 1.0, "top_p": 0.95, "top_k": None, "max_tokens": 32768,
                "model_id": "nemotron-3.5-bolt", "thinking": level}
    print("ready in %.0f s: context %d (%s); thinking %s" % (time.time() - t0, ctx, where, thinking), flush=True)
    return eng, defaults
