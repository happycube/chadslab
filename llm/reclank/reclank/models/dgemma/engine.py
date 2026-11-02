"""DiffusionGemma in the server (reclank/models/base.py; D4 of docs/DIFFUSIONGEMMA_PLAN.md): the BF12
GGUF on the GPU (DGemmaGPU), the chat of Gemma 4 (the checkpoint's template is Gemma 4's with no
empty thought block in the generation prompt: models/gemma4/chat.py render_chat with
empty_thought_block False; the thought channel and the tool calls parse as Gemma 4's).

A response template: when the last message of a chat request is the assistant's, its text is the
answer to come, with holes [[n]] of n free tokens (sampler.parse_template): the text stands where
it is, the model writes the holes and goes on after it. Text at its start is a plain prefix. The
answer (the content) holds the whole of it, the pinned text in place.

The answer streams a canvas (256 tokens) at a time. The request's sampling fields do not apply:
the checkpoint's sampler does (generation_config.json); a seed gives the same answer.
"diffusion_drafts": true in a request adds a record after each denoising step (draft_record: the
draft canvas, each position's probability (processed and raw) and entropy, the kept positions): a stream sends each
as an event of its own (object "diffusion.draft"), a response without a stream holds them in
"diffusion_drafts".

The cache keeps the last request's positions: a prompt that starts with them (the next turn of
a chat: the prompt before it, and the answer as the template gives it back) reads only the rest.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from reclank.core.messages import content_text, has_media
from reclank.models.base import Engine
from reclank.models.dgemma import sampler as S
from reclank.models.gemma4.engine import request_thinking, tool_done

FAMILY = "dgemma"


def detect(meta):
    return str(meta.get("general.architecture", "")) == "diffusion-gemma"


class DGPrompt(list):
    """The prompt's ids, the response template (or None), and the request's fields (options: the
    server sets it; "diffusion_drafts")."""
    template = None
    options = None


class DGemmaEngine(Engine):
    family = FAMILY

    def __init__(self, model, tokenizer, settings, max_context):
        self.model = model
        self.tokenizer = tokenizer
        self.s = settings
        self.max_context = max_context
        self.last_text = None

    def think_level(self, req, default):
        v = request_thinking(req)
        return bool(default) if v is None else bool(v)

    def prompt_ids(self, messages, level, tools=None, preserve=None, source=None):
        from reclank.models.gemma4.chat import render_chat
        msgs, template = [], None
        for m in messages:
            if has_media(m.get("content")):
                raise ValueError("this server takes no image input (DiffusionGemma's vision is not in reclank)")
            msg = dict(m)
            msg["role"] = m.get("role", "user")
            msg["content"] = content_text(m.get("content"))
            msgs.append(msg)
        if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"):
            template = msgs.pop()["content"] or None
        text = render_chat(msgs, tools=tools, add_generation_prompt=True, enable_thinking=bool(level),
                           preserve_thinking=bool(preserve), empty_thought_block=False)
        self.last_text = text if template is None else text + "[the response template] " + template
        out = DGPrompt(self.tokenizer.encode(text))
        out.template = template
        return out

    def stop_ids(self):
        """The checkpoint's end tokens (generation_config.json: <eos>, <turn|>, and 50) and the
        tokenizer's."""
        return set(self.s.eos) | set(getattr(self.tokenizer, "stop_ids", ()) or ())

    def completion_ids(self, prompt):
        return self.tokenizer.encode(prompt)

    def parse_output(self, text):
        from reclank.models.gemma4.chat import parse_output
        return parse_output(text)

    def tool_done(self, text):
        return tool_done(text)

    def turn_fields(self, rec):
        return {"prompt_text": self.last_text}

    def draft_record(self, ev):
        """A step's record for the client: the canvas, the steps left, the draft (ids, text, and
        each token's own text), each position's probability of its draft token (prob: the
        processed logits', the step's temperature and the end ban; raw_prob: the raw logits', the
        model's own), its entropy, and the positions the sampler kept (pinned positions: 1 for
        both)."""
        ids = [int(v) for v in ev["draft"]]
        r = {"canvas": int(ev["canvas"]), "step": int(ev["step"]), "tokens": ids,
             "text": self.tokenizer.decode(ids, skip_special_tokens=False),
             "pieces": [self.tokenizer.decode([v], skip_special_tokens=False) for v in ids],
             "entropy": [round(float(v), 4) for v in ev["entropy"]], "kept": [int(v) for v in ev["kept"]]}
        if ev.get("prob") is not None:
            r["prob"] = [round(float(v), 4) for v in ev["prob"]]
        if ev.get("raw") is not None:
            r["raw_prob"] = [round(float(v), 4) for v in ev["raw"]]
        return r

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        """Yield the answer's tokens, a canvas at a time; with the request's "diffusion_drafts", a
        dict (draft_record) after each denoising step too. The cache keeps the positions that the
        last request left and that this prompt starts with (the turns of a chat)."""
        m = self.model
        req = getattr(prompt_ids, "options", None) or {}
        events = bool(req.get("diffusion_drafts"))
        template = getattr(prompt_ids, "template", None)
        fixed = S.parse_template(template, self.tokenizer.encode) if template else None
        if len(prompt_ids) + max_tokens + m.C > m.max_len:
            max_tokens = max(1, m.max_len - m.C - len(prompt_ids))
        s = S.Settings(**{**self.s.__dict__, "eos": tuple(sorted(set(self.s.eos) | set(eos_ids or ())))})
        # the request's random numbers (its seed: core/sampling.py Sampler.rng)
        rng = getattr(sampler, "rng", None) or np.random.default_rng()
        # the ids the encoder reads first (the prompt and a pinned start): those the cache holds
        # from the last request are not read again
        head = []
        if fixed is not None:
            for v in fixed:
                if v < 0:
                    break
                head.append(int(v))
        target = [int(v) for v in prompt_ids] + head
        held = m.reuse(target)
        self.last_cached = held
        state = {"skip": held}

        def encode(ids):
            ids = ids[state["skip"]:]
            state["skip"] = 0
            if len(ids):
                m.encode(ids)

        st = S.Stats()
        t0 = time.time()
        try:
            for got in S.generate_iter(encode, None, list(prompt_ids), max_tokens, s, rng, fixed=fixed,
                                       steps=m.steps, stats=st, events=events):
                if isinstance(got, dict):
                    yield self.draft_record(got)
                    continue
                for tok in got:
                    yield int(tok)
        finally:
            t2 = time.time()
            print("[reclank] dgemma: prompt %d tokens (%d from the cache); %d tokens in %.1f s (%.1f tok/s), "
                  "%d passes (%s), %.1f tokens a pass%s" % (
                      len(prompt_ids), held, st.tokens, t2 - t0, st.tokens / max(t2 - t0, 1e-9), sum(st.steps),
                      " ".join(map(str, st.steps)), st.per_pass, "; a template" if template else ""),
                  file=sys.stderr, flush=True)


def load(opts, meta=None, worker=None):
    """The engine of a BF12 GGUF (scripts/convert_dgemma_bf12.py's file; config.json, the
    generation config and tokenizer.json beside it): (engine, the defaults of the Backend)."""
    import json
    import os
    from reclank.models.dgemma.gpu import DGemmaGPU
    from reclank.models.gemma4.tokenizer import Tokenizer
    d = os.path.dirname(os.path.abspath(opts.model))
    ctx = int(opts.ctx or 32768)
    m = DGemmaGPU(opts.model, max_len=ctx)
    tok = Tokenizer(opts.tokenizer or os.path.join(d, "tokenizer.json"))
    s = S.Settings.from_config(json.load(open(os.path.join(d, "generation_config.json"))),
                               json.load(open(os.path.join(d, "config.json"))))
    eng = DGemmaEngine(m, tok, s, ctx)
    # the programs now (the first request would build them)
    t0 = time.time()
    m.reset()
    for _ in S.generate_iter(m.encode, None, tok.encode("<bos><|turn>user\nHello.<turn|>\n<|turn>model\n"), 8, s,
                             np.random.default_rng(0), steps=m.steps):
        pass
    print("dgemma: ready in %.0f s (the programs built); a cache of %d positions" % (time.time() - t0, ctx),
          flush=True)
    thinking = getattr(opts, "thinking", None)
    # (the request's sampling fields do not apply: the checkpoint's sampler does)
    defaults = {"max_tokens": 4096, "temperature": 1.0, "top_k": None, "top_p": None,
                "model_id": "diffusiongemma-26b-a4b", "thinking": thinking in ("on", "low", "medium", "high", "xhigh")}
    return eng, defaults
