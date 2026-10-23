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

import contextlib
import json
import os
import sys
import time

import numpy as np

from reclank.core.session import HostRows, RowCache
from reclank.core.speculative import DeviceOp
from reclank.core.think_guard import ThinkGuard, newline_test, report, think_budget, think_left_open, wrap_at_of
from reclank.models.base import Engine
from reclank.models.qwen.chat import (THINK_CLOSE, TOOLS, WRAP_ANSWER, WRAP_THINK, QwenTok, _escaped, _message,
                                      _plain, qwen_parse_output, thinking_of)

FAMILY = "nemotron"
IMAGE_TYPES = ("image_url", "image", "input_image")
MODEL_TYPES = ("nemotron_h_omni", "nemotron_h_puzzle", "nemotron_h")
# the GPU memory the hot experts leave to the image encoder (its bfloat16 weights, 1.6 GB, and the
# buffers of an image of 1024 rows)
VISION_GB = float(os.environ.get("RECLANK_NM_VISION_GB", "2.6"))
# and to the MTP block's programs (Super: its prompt parts of up to 256 rows, its drafts, the verify group)
MTP_GB = float(os.environ.get("RECLANK_NM_MTP_GB", "0.6"))
# and to lane B (--lanes 2): its cache (~1.5 GB at 131072 positions with the Mamba states' two sets) and its
# programs
LANE_GB = float(os.environ.get("RECLANK_NM_LANE_GB", "1.5"))
THOUGHT_OPEN = "<|channel>thought\n"     # the opener of the parser (models/gemma4/chat.py)
# the lanes (--lanes 2): a prompt in parts of this many tokens, a pause after each but the last (the
# other lane decodes between them; RECLANK_LANE_PART, server/worker.py's)
LANE_PART = int(os.environ.get("RECLANK_LANE_PART", "4096"))


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
        keys = sorted(i for i in c.K if i != "mtp") + (["mtp"] if "mtp" in c.K else [])
        return [HostRows(a, 1) for i in keys for a in (c.K[i], c.V[i])]

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
    """The model on the GPU (NemotronGPU: its cache on the device), for the engine. mtp: Super's
    rounds of drafts by its MTP block (spec.NemotronMTPSpec), or None."""

    def __init__(self, g, mtp=None):
        self.g = g
        self.mtp = mtp
        self.ids = []                   # (with mtp: the tokens of the positions so far)
        g.reset()

    @property
    def n(self):
        return self.g.cache.n

    def forward(self, ids, relax=True):
        if self.mtp is None:
            return self.g.prefill(ids, relax=relax) if len(ids) > 1 else self.g.step(ids[0])
        pos = self.g.cache.n
        del self.ids[pos:]
        self.ids += [0] * (pos - len(self.ids)) + list(ids)     # (positions before a restored point: unknown)
        lg = self.g.prefill(ids, rows=self.mtp.after_prompt, relax=False)
        self.mtp.flush(self.ids)
        if relax:
            self.g.relax()          # (after the MTP block's pass: its big parts go too)
        return lg

    def stepped(self, token, pos):
        """After a step of token at pos run by the lanes' scheduler (paired or alone): the token in the
        runner's ids, and (mtp) the step's row for the MTP block (read by the next round's flush)."""
        del self.ids[pos:]
        self.ids += [0] * (pos - len(self.ids)) + [int(token)]
        if self.mtp is not None:
            self.mtp.after_prompt(pos, self.g.last_hidden(1))

    def resume_rounds(self):
        """Before an MTP round after plain steps: the block reads their positions (but the last)."""
        if self.mtp is not None and self.mtp.pend is not None and len(self.mtp.pend[1]) > 1:
            self.mtp.flush(self.ids)

    def round(self, last, sample):
        """A round of MTP drafts and their verify group after the token last: the tokens taken."""
        got = self.mtp.round(last, sample)
        n0 = self.g.cache.n - len(got)
        del self.ids[n0:]
        self.ids += [int(last)] + got[:-1]
        return got

    def _buf(self, a):
        return self.g.cache_dev.buf(a)

    def rows(self):
        from reclank.core.session import BufferRows
        c = self.g.cache
        keys = sorted(i for i in c.K if i != "mtp") + (["mtp"] if "mtp" in c.K else [])
        return [BufferRows(self._buf(a), 1, a.nbytes // a.shape[0]) for i in keys for a in (c.K[i], c.V[i])]

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
        if self.mtp is not None:
            self.mtp.pend = None        # (the block has read the point's positions but its last)

    def go_on(self, k):
        self.g.cache.n = k
        if self.mtp is not None and (self.mtp.pend is None or self.mtp.pend[0] + len(self.mtp.pend[1]) != k):
            self.mtp.pend = None

    def fresh(self):
        self.g.reset()
        self.ids = []
        if self.mtp is not None:
            self.mtp.pend = None


class NemotronEngine(Engine):
    family = FAMILY

    def __init__(self, run, tok, ctx, contexts=4, contexts_gb=16.0, vision=None, worker=None):
        self.run = run                  # CpuRunner or GpuRunner
        # Super's image input (radio.RadioGPU, or None), the model thread's worker for it, the rows of
        # the last images by key, the most rows of an image
        self.vision = vision
        self.worker = worker
        self._media_rows = {}
        self.image_budget = 1024
        self.last_media = []
        self.tokenizer = tok
        self.max_context = ctx
        # the think budget (--think-budget), the words to wrap up (--wrap-left) and their tokens (load):
        # the request's ThinkGuard (think_guard)
        self.think_budget, self.wrap_left = 6000, 500
        self.think_force = self.wrap_think = self.wrap_answer = ()
        self.ends_line = None
        self.ctxs = RowCache(self, contexts, contexts_gb * 1e9, "nemotron")
        self.as_is = []                 # the keys the cache has read (the prompt, the answer)
        self.last_cached = 0
        self.last_text = ""
        self.im_start = tok.t.special["<|im_start|>"]
        # the lanes (--lanes 2: _lanes_up): the engines of both, this one's NmLane (None: lane A, the model's
        # own state), coop (the runs handed to the scheduler as DeviceOps), decoding (a generation of this
        # lane is in its decode), the logits of the last run the scheduler ran for it
        self.lanes = [self]
        self.lane = None
        self.coop = False
        self.decoding = False
        self.last_logits = None

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
        images = []
        for m in msgs:
            if isinstance(m.get("content"), list):
                kinds = [p.get("type") for p in m["content"] if isinstance(p, dict)]
                if any(k in IMAGE_TYPES for k in kinds) and self.vision is not None:
                    # the image parts stay for the template (it writes <image>, or <image k><image>
                    # when the message has several); the text parts as parts
                    parts = []
                    for p in m["content"]:
                        if isinstance(p, str):
                            parts.append({"type": "text", "text": p})
                        elif p.get("type") in IMAGE_TYPES:
                            images.append(p)
                            parts.append({"type": "image"})
                        elif p.get("type") in ("text", "input_text", "output_text"):
                            parts.append({"type": "text", "text": p.get("text", "")})
                    m["content"] = parts
                    continue
                if any(k not in ("text", "input_text", "output_text", None) for k in kinds):
                    raise ValueError("this model takes text and images only" if self.vision is not None else
                                     "this server takes no image input (Bolt, or --mmproj off)")
                m["content"] = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in m["content"])
        esc = self.tokenizer.t.escape
        text = self.tokenizer.apply_chat_template(_escaped(msgs, esc), thinking=level is not None,
                                                  tools=_escaped(tools, esc), effort=level, preserve=preserve)
        self.last_text = text
        ids = self.tokenizer.encode(text)
        return self._expand(ids, images, source) if images else ids

    def _expand(self, ids, parts, source):
        """Each <image> of ids as the span <img> + <image> x n + </img> of its image (the processor's
        expansion): a PromptIds whose media.Span gives the n rows of the <image> positions."""
        from reclank.core.messages import PromptIds
        from reclank.encoders import media as MD
        from reclank.models.nemotron import radio
        if sum(1 for t in ids if t == radio.IMG_CONTEXT) != len(parts):
            raise ValueError("the prompt has %d <image> for %d images" % (
                sum(1 for t in ids if t == radio.IMG_CONTEXT), len(parts)))
        out, spans, it = [], [], iter(parts)
        for t in ids:
            if t != radio.IMG_CONTEXT:
                out.append(t)
                continue
            rows, key = self._image(next(it), source, len(parts))
            out.append(radio.IMG_START)
            spans.append(MD.Span(len(out), rows, False, "image", key, (rows.shape[0], 1)))
            out += [radio.IMG_CONTEXT] * rows.shape[0] + [radio.IMG_END]
        res = PromptIds(out)
        res.spans = spans
        self.last_media = [(sp.kind, sp.rows.shape[0], sp.key) for sp in spans]
        return res

    def _image(self, part, source, n_images):
        """The rows of an image part (the encoder on the model thread; the last ones kept by key)."""
        from reclank.encoders import media as MD
        if source is None:
            from server.media import MediaSource
            source = MediaSource()
        if part.get("type") == "input_image":           # (the Responses form: image_url a string)
            part = {"type": "image_url", "image_url": part.get("image_url")}
        data, budget = source.image(part, self.image_budget)
        budget = max(64, min(int(budget), self.image_budget))
        key = MD.media_key(data) + ":%d:%d" % (budget, n_images)
        rows = self._media_rows.pop(key, None)
        if rows is None:
            t0 = time.time()

            def call():
                try:
                    return self.vision.image(data, n_images, budget)[0]
                except (MemoryError, RuntimeError) as exc:
                    # (no room for the encoder: the hot experts give back its share, then once more)
                    pool = getattr(getattr(self.run, "g", None), "pool", None)
                    if pool is None or "memory" not in str(exc).lower() and "cudaMalloc" not in str(exc):
                        raise
                    n = pool.shrink(VISION_GB * 1e9)
                    print("[nemotron] the image encoder found no room (%s): the hot experts -%.1f GB" % (exc, n / 1e9),
                          file=sys.stderr, flush=True)
                    return self.vision.image(data, n_images, budget)[0]
            rows = self.worker.call(call) if self.worker is not None else call()
            print("[nemotron] image of %d rows in %.2f s" % (rows.shape[0], time.time() - t0), file=sys.stderr,
                  flush=True)
        self._media_rows[key] = rows
        while len(self._media_rows) > 1 and sum(r.nbytes for r in self._media_rows.values()) > 4e9:
            self._media_rows.pop(next(iter(self._media_rows)))
        return rows, key

    @staticmethod
    def _keys(ids):
        """The keys of the positions of a prompt: the ids; the first position of each image span its key
        (two images of the same size have the same ids)."""
        keys = list(ids)
        for sp in getattr(ids, "spans", ()) or ():
            keys[sp.start] = ("image", sp.key)
        return keys

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

    # ---- the lanes (server/worker.py LaneScheduler)

    @contextlib.contextmanager
    def on(self):
        """The model with this engine's lane in place (NmLane.view; lane A: as it is), and the device the
        scheduler runs a lone step on (its step(token, pos))."""
        with self.lane.view() if self.lane is not None else contextlib.nullcontext():
            yield _LaneDev(self)

    def pair(self, a, b):
        """The runs of lane A (this engine) and lane B (lanes[1]) at once (lanes.run_both): two steps.
        Not verify groups (a lane alone decodes with MTP rounds; with both decoding, plain steps)."""
        if len(a.tokens) != 1 or len(b.tokens) != 1 or len(self.lanes) < 2:
            return False
        from reclank.models.nemotron.lanes import run_both
        eb = self.lanes[1]
        t0 = time.perf_counter()
        self.last_logits, eb.last_logits = run_both(self.run.g, eb.lane, a.tokens[0], a.pos, b.tokens[0], b.pos)
        if os.environ.get("RECLANK_NM_LANE_STATS"):
            st = self.__dict__.setdefault("_pst", [0, 0.0, time.perf_counter()])
            st[0] += 1
            st[1] += time.perf_counter() - t0
            if st[0] == 100:
                wall = time.perf_counter() - st[2]
                print("[lanes] 100 pairs: %.1f ms a pair, %.1f ms between pairs" % (10 * st[1], 10 * (wall - st[1])),
                      file=sys.stderr, flush=True)
                self._pst = [0, 0.0, time.perf_counter()]
        return True

    def hit(self, prompt_ids):
        """What this lane's contexts would give a prompt (RowCache.peek): (the tokens they hold, True when
        that is the context on this lane's device). On the model thread."""
        return self.ctxs.peek(self._keys(prompt_ids), lambda st: False)

    def _other_decodes(self):
        return any(e is not self and e.decoding for e in self.lanes)

    def _forward_co(self, ids):
        """(yield from) run.forward(ids); with coop in parts of LANE_PART tokens, a pause after each but
        the last (the other lane decodes between them; the big groups' programs kept until the last)."""
        if not self.coop or len(ids) <= LANE_PART:
            return self.run.forward(ids)
        out = None
        for c0 in range(0, len(ids), LANE_PART):
            if c0:
                yield DeviceOp("pause")
            last = c0 + LANE_PART >= len(ids)
            out = self.run.forward(ids[c0:c0 + LANE_PART], relax=last)
        return out

    def _step_co(self, y):
        """(yield from) a plain step of y handed to the scheduler (paired with the other lane's when it
        steps too): its logits."""
        pos = self.run.n
        self.last_logits = None
        yield DeviceOp("run", [int(y)], pos)
        self.run.stepped(y, pos)
        return self.last_logits

    def _plain(self, y, coop):
        """(yield from) a plain step of y on the MTP engine (the lanes: handed to the scheduler,
        _step_co): its logits; the step's row to the MTP block, as after the lanes' steps."""
        if coop:
            return (yield from self._step_co(y))
        pos = self.run.n
        lg = self.run.g._run([int(y)], pos)
        self.run.stepped(y, pos)
        return lg

    # ---- the generation

    def think_guard(self, prompt_ids, max_tokens, eos_ids):
        """The ThinkGuard of a request (core/think_guard.py) when the prompt ends inside <think> or the
        answer may reach the end of the context, else None: the think part closes at think_budget
        tokens (and at most half of max_tokens), and the words to wrap up go in when fewer than
        wrap_left positions of the context are left. A request of max_tokens 32768 (a three.js page)
        thought for all of them: no answer, and the client's "continue" started the thinking again."""
        force = self.think_force
        wrap_at = wrap_at_of(self.wrap_left, self.max_context - len(prompt_ids) - 1, max_tokens)
        return ThinkGuard.of(prompt_ids, self.tokenizer.t.special, eos_ids,
                             think_budget(self.think_budget, max_tokens, len(force)), force, wrap_at,
                             self.wrap_think, self.wrap_answer, self.ends_line)

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        spans = list(getattr(prompt_ids, "spans", ()) or ())
        if spans and not isinstance(self.run, GpuRunner):
            raise ValueError("images need the GPU engine")
        if isinstance(self.run, GpuRunner):
            self.run.g.media = [(sp.start, sp.rows) for sp in spans]
        keys = self._keys(prompt_ids)
        ids = list(prompt_ids)
        n = len(ids)
        if n + 1 > self.max_context:
            raise ValueError("the prompt has %d tokens; the context is %d" % (n, self.max_context))
        max_tokens = min(int(max_tokens), self.max_context - n - 1)
        t0 = time.time()
        k, point = self.ctxs.begin(keys, lambda st: False, as_is=self.as_is)
        if k == 0:
            self._fresh()
        elif point is not None:
            self._restore(point, k)
        else:
            self.run.go_on(k)               # go on from what the cache read
        self.as_is = []
        cut = self._cut(ids)
        if k < cut < n:
            yield from self._forward_co(ids[k:cut])
            self._save(keys[:cut])
            k0 = cut
        else:
            k0 = k
        lg = yield from self._forward_co(ids[k0:])
        dt = time.time() - t0
        self.last_cached = k
        print("[nemotron] cache hit %d of %d prompt tokens; new %d in %.1f s (%.0f tok/s); %s" % (
            k, n, n - k, dt, (n - k) / max(dt, 1e-9), self.ctxs.note(n, k)), file=sys.stderr, flush=True)
        sampler.reset(prompt_ids)
        guard = sampler.guard = self.think_guard(prompt_ids, max_tokens, eos_ids)

        def pick(row):
            return sampler.fixed(sampler(row))

        t1, out = time.time(), []
        rounds = 0
        coop = self.coop and isinstance(self.run, GpuRunner)
        plain = 0
        try:
            self.decoding = True
            y = pick(lg)
            if getattr(self.run, "mtp", None) is not None:
                out.append(y)
                yield y
                while y not in eos_ids and len(out) < max_tokens:
                    # (the last tokens of max_tokens in plain steps: a round would read drafts past the
                    # cut that the client never gets, and the Mamba states cannot go back: a "continue"
                    # after the cut, the answer's prompt and tokens, would find no start in the cache)
                    tail = max_tokens - len(out) <= self.run.mtp.k
                    if (coop and self._other_decodes()) or tail:
                        # (the other lane decodes: a plain step, paired with its)
                        y = pick((yield from self._plain(y, coop)))
                        plain += 1
                        out.append(y)
                        yield y
                        continue
                    if coop or plain:
                        self.run.resume_rounds()
                    got = self.run.round(y, sampler)
                    rounds += 1
                    for tok in got:
                        out.append(tok)
                        yield tok
                        if tok in eos_ids or len(out) >= max_tokens:
                            break
                    if coop:
                        # (the scheduler between the rounds: the other lane's step, or its prompt's next
                        # part when its share of the time has passed)
                        yield DeviceOp("yield")
                    y = got[-1]
                    if out[-1] != y:            # (stopped inside the round)
                        break
                return
            while True:
                out.append(y)
                yield y
                if y in eos_ids or len(out) >= max_tokens:
                    break
                if coop:
                    y = pick((yield from self._step_co(y)))
                else:
                    y = pick(self.run.forward([y]))
        finally:
            self.decoding = False
            sampler.guard = None
            report(guard, "nemotron", self.wrap_left)
            # the cache has read the prompt and the answer but its last token (MTP rounds: what the
            # runner's tokens say it read, the drafts taken past a stop too)
            self.as_is = keys + out[:-1]
            if rounds or plain:
                self.as_is = keys + list(self.run.ids[len(keys):self.run.n])
            dt = time.time() - t1
            extra = "; MTP %d rounds, %.2f tokens a round" % (rounds, (len(out) - 1 - plain) / rounds) if rounds else ""
            extra += "; %d plain steps (the lanes)" % plain if plain else ""
            extra += " [lane %s]" % "AB"[self.lanes.index(self)] if len(self.lanes) > 1 else ""
            print("[nemotron] decode %d tokens in %.1f s (%.1f tok/s)%s" % (len(out), dt, len(out) / max(dt, 1e-9), extra),
                  file=sys.stderr, flush=True)


class _LaneDev:
    """What LaneScheduler runs a lone step on (engine.on()): the step, its logits to the engine."""

    def __init__(self, eng):
        self.eng = eng

    def step(self, token, pos):
        self.eng.last_logits = self.eng.run.g._run([int(token)], pos)

    def verify(self, tokens, pos):
        raise RuntimeError("the Nemotron lanes hand over single steps only")


def _lanes_up(eng, ctx, drafts, warm):
    """--lanes 2: lane B (lanes.NmLane: a second sequence, its own state on the GPU, its cache as long
    as the model's so that the lanes share the contexts in host memory) and its engine; both hand their
    runs to the server's scheduler (coop). warm: a prompt for lane B's programs."""
    import copy
    from reclank.models.nemotron.lanes import NmLane
    t0 = time.time()
    g = eng.run.g
    lane = NmLane(g, ctx=ctx)
    eb = copy.copy(eng)             # (the model, the tokenizer, the encoder, the images' rows shared)
    with lane.view():
        mtp = None
        if eng.run.mtp is not None:
            from reclank.models.nemotron.spec import NemotronMTPSpec
            mtp = NemotronMTPSpec(g, drafts)
        eb.run = GpuRunner(g, mtp)
    eb.ctxs = RowCache(eb, eng.ctxs.max_contexts, eng.ctxs.max_bytes, "nemotron B")
    eb.ctxs.share_with(eng.ctxs)
    eb.as_is, eb.last_cached, eb.last_text, eb.last_media = [], 0, "", []
    eb._media_rows = eng._media_rows
    eb.lane = lane
    eng.lanes = eb.lanes = [eng, eb]
    eng.coop = eb.coop = True
    eb.decoding = eng.decoding = False
    with lane.view():
        # its programs of a prompt and of a step now (as the model's at the load), its first steps alone
        eb.run.forward(warm)
        eb.run.forward([warm[-1]])
        if mtp is not None:
            eb.run.round(warm[-1], lambda lg: int(np.argmax(lg)))
        eb._fresh()
    from reclank.backends.cuda.gpumm import mem_info
    print("lane B: %d positions, its programs in %.0f s; GPU %.1f GB free" % (
        ctx, time.time() - t0, mem_info()[0] / 1e9), flush=True)


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
    # (Super: 524288 positions, its bfloat16 cache of 8 attention layers and the MTP block's ~4.8 GB on the
    # GPU, out of the hot experts; Bolt 65536)
    ctx = int(opts.ctx or (524288 if m.cfg.mtp else 65536))
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
    vision = None
    images = gpu and str(opts.mmproj or "on").lower() not in ("off", "none", "0") and \
        os.path.exists(os.path.join(path, "modeling_radio.py"))
    lanes = int(getattr(opts, "lanes", 1) or 1) if gpu else 1
    if gpu:
        from reclank.models.nemotron.gpu import NemotronGPU, load_counts
        # (Super's image encoder: its weights and a 1024-row image on the GPU while it runs)
        drafts = 3 if opts.mtp is None else int(opts.mtp)
        reserve = (VISION_GB if images else 0.0) + (MTP_GB if m.cfg.mtp and drafts > 0 else 0.0)
        if lanes > 1:
            # (lane B's thread on a CPU of its own, out of the CPU team planned at the first run; its cache
            # and programs out of the hot experts)
            from reclank.models.nemotron.lanes import reserve_lane_cpu
            reserve_lane_cpu()
            reserve += LANE_GB
        g = NemotronGPU(m, ctx=ctx, hot_gb=opts.hot_gb, counts=load_counts(path), reserve_gb=reserve)
        mtp = None
        if m.cfg.mtp and drafts > 0:
            from reclank.models.nemotron.spec import NemotronMTPSpec
            mtp = NemotronMTPSpec(g, drafts)
        run = GpuRunner(g, mtp)
        if images:
            from reclank.models.nemotron.radio import RadioGPU
            vision = RadioGPU(path, max_patches=4 * (opts.image_budget or 1024))
        where = "GPU, %d%% of the experts hot (%.1f GB); the host link %.1f GB/s: a prompt's cold experts on the %s%s%s" % (
            round(100 * g.hot_share), g.hot_bytes / 1e9, g.link_gbps, "GPU (mapped)" if g.zc_tables else "CPU",
            "; MTP %d drafts" % drafts if mtp else "", "; images on" if vision else "")
    else:
        run = CpuRunner(m, ctx)
        where = "CPU, tier %s" % __import__("reclank.backends.cpu.cops", fromlist=["TIER"]).TIER
    eng = NemotronEngine(run, tok, ctx, contexts=getattr(opts, "contexts", 4),
                         contexts_gb=getattr(opts, "contexts_gb", 16.0), vision=vision, worker=worker)
    if opts.image_budget:
        eng.image_budget = int(opts.image_budget)
    eng.think_budget, eng.wrap_left = max(0, opts.think_budget), max(0, opts.wrap_left)
    eng.think_force, eng.wrap_think, eng.wrap_answer = (tok.encode(THINK_CLOSE), tok.encode(WRAP_THINK),
                                                        tok.encode(WRAP_ANSWER))
    eng.ends_line = newline_test(tok.decode)
    # the experts of every layer and the programs of a prompt and of a step now
    warm = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Hello. " * 40}], thinking=False))
    run.forward(warm)
    run.forward([warm[-1]])
    eng._fresh()
    if lanes > 1:
        _lanes_up(eng, ctx, drafts, warm)
    thinking = opts.thinking or "high"
    level = None if thinking == "off" else thinking_of({"reasoning_effort": thinking}, "xhigh")
    defaults = {"temperature": 1.0, "top_p": 0.95, "top_k": None, "max_tokens": 32768,
                "model_id": "nemotron-3.5-super" if m.cfg.mtp else "nemotron-3.5-bolt", "thinking": level}
    print("ready in %.0f s: context %d (%s); thinking %s" % (time.time() - t0, ctx, where, thinking), flush=True)
    return eng, defaults
