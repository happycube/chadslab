"""The Qwen models in the server (reclank/models/base.py): the Qwen3.5 family (Qwen3.6 35B A3B)
(qwen35moe: QwenGGUFProgram, QwenGPU) and Qwen3.8-Flash-Next (qwen4exp:
Qwen4CPU, Qwen4GPU), with one cache of --ctx tokens, snapshots of the
recurrent state between the turns (core/session.py), MTP drafts (Qwen3.8 on
the GPU), and the image encoder.

The cache: a prompt reads the tokens after the longest start that the cache
or a snapshot holds; at the end of each prompt, and after the last
<|im_start|>assistant\n of each prompt, the state goes to a snapshot. The
template of the next turn writes the earlier answer in another form than
the model wrote it (Qwen3.6 drops its think part; Qwen3.8 keeps only the
reasoning_content that the client sends back), so the next prompt differs
from the cache there.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback

import numpy as np

from reclank.core.messages import MEDIA_TYPES, PromptIds
from reclank.core.session import SnapshotList
from reclank.core.think_guard import ThinkGuard, think_budget, think_left_open
from reclank.encoders import media as MD
from reclank.models.base import Engine
from reclank.models.qwen.chat import (THINK_CLOSE, TOOLS, WRAP_ANSWER, WRAP_THINK, QwenTok, _escaped,
                                      _message, _plain, qwen_parse_output, thinking_of)

FAMILY = "qwen"
THOUGHT_OPEN = "<|channel>thought\n"     # the opener of the parser (models/gemma4/chat.py)
# The CUDA errors that leave the context unusable: after one the process
# exits (server/worker.py, EXIT_FATAL), so a supervisor
# (scripts/serve_forever.sh) can start it again. Out of memory is not one.
FATAL = re.compile(r"illegal (memory access|instruction|address)|unspecified launch failure|"
                   r"misaligned address|device-side assert|invalid program counter|"
                   r"hardware stack error|uncorrectable ECC|context is destroyed|"
                   r"cudaErrorLaunchFailure|CUDA_ERROR_LAUNCH_FAILED", re.I)
ARCHS = ("qwen35moe", "qwen4exp")


def detect(meta):
    return meta.get("general.architecture") in ARCHS or meta.get("model_type") == "qwen4_exp"


class QwenEngine(Engine):
    """Qwen4GPU (or Qwen4CPU), or QwenGPU (or QwenGGUFProgram), and one cache
    of ctx tokens."""

    family = FAMILY

    def __init__(self, model, dev, tok, ctx, cache_cls):
        self.model = model
        self.dev = dev
        self.tokenizer = tok
        self.ctx = ctx
        self.cache_cls = cache_cls
        self.worker = None      # the model thread (server/worker.py): the encoders run there
        self.cache = None
        self.ids = []           # the keys of the tokens in the cache (media.keys)
        # Snapshots of the recurrent state: (keys, state, n-gram ids, logits
        # or None, stream), the oldest first. The state of the DeltaNet cannot
        # go back, so a prompt can reuse the cache only from the end of the
        # cache or from a snapshot (core/session.py).
        self.snaps = SnapshotList(4)
        # The head of the answer of a chat prompt (<|im_start|>assistant\n):
        # each prompt has a snapshot after its last one.
        self.answer_head = None
        self.last_cached = 0    # the prompt tokens of the last request that the cache held
        self.hits = [0, 0, 0]   # requests, prompt tokens, cached tokens
        self.last_text = ""
        self.mtp = 0            # --mtp: the drafts of a round (0: no MTP)
        self.h_end = None       # the stream of the model at the last position of the cache
        self.mtp_stats = [0, 0]  # drafts, accepted
        # the think part: the budget, and the words that close it and wrap up
        self.think_budget = 0
        self.wrap_left = 0
        self.think_force = ()
        self.wrap_think = ()
        self.wrap_answer = ()
        self.crash_dir = None
        # media input (--mmproj): the embedder, the default budgets, and a
        # cache of soft rows by media key
        self.embedder = None
        self.image_budget = 1024
        self.video_budget = 128
        self.video_frames = 32
        self._media_rows = {}
        self.last_media = []

    # ---- the prompt

    def think_level(self, req, default):
        return thinking_of(req, default)

    def think_off(self):
        return None

    def turn_fields(self, rec):
        return {"prompt_text": self.tokenizer.t.unescape(self.last_text),
                "raw": self.clean_raw(rec.get("raw", "")), "cached_tokens": self.last_cached}

    def prompt_ids(self, messages, level, tools=None, preserve=None, source=None):
        thinking, effort = level is not None, level
        TOOLS.clear()
        for tl in tools or []:
            f = tl.get("function", tl)
            TOOLS[f.get("name")] = (f.get("parameters") or {}).get("properties") or {}
        think = bool(thinking)
        msgs = [_message(m) for m in messages]
        parts = [p for m in msgs if isinstance(m.get("content"), list) for p in m["content"]
                 if isinstance(p, dict) and p.get("type") in MEDIA_TYPES]
        for m in msgs:
            if isinstance(m.get("content"), list):
                # the template knows a video part by a key "video"
                m["content"] = [dict(p, video=True) if isinstance(p, dict)
                                and p.get("type") in ("video", "video_url") else p
                                for p in m["content"]]
        # The special tokens of the chat in the text of the messages and the
        # tools stay text (QwenTokenizer.escape): a file that holds
        # "<|im_end|>" must not end a turn of the prompt.
        esc = self.tokenizer.t.escape
        msgs = _escaped(msgs, esc)
        text = self.tokenizer.apply_chat_template(msgs, thinking=think, tools=_escaped(tools, esc),
                                                  effort=effort, preserve=preserve)
        if not parts:
            self.last_text = text
            return self.tokenizer.encode(text)
        items, videos = self._media_items(parts, source)
        if videos:
            # each <|video_pad|> of the template becomes the times and one pad
            # for each pair of frames (Qwen3VLProcessor.replace_video_token)
            pieces = text.split("<|video_pad|>")
            if len(pieces) != len(videos) + 1:
                raise ValueError("the prompt has %d video placeholders for %d videos"
                                 % (len(pieces) - 1, len(videos)))
            text = pieces[0] + "".join(v + p for v, p in zip(videos, pieces[1:]))
        self.last_text = text
        out_ids, spans = MD.expand_qwen(self.tokenizer.encode(text), items)
        out = PromptIds(out_ids)
        out.spans = spans
        self.last_media = [(sp.kind, sp.rows.shape[0], sp.key) for sp in spans]
        return out

    def _media_items(self, parts, source):
        """The rows of the media parts: the items of expand_qwen (one for an
        image, one for each pair of frames of a video), and the text of each
        video. The encoder runs on the thread of the model."""
        from reclank.models.qwen.vision import video_text
        if self.embedder is None:
            raise ValueError("this server takes no image or video input (start it with --mmproj)")
        items, videos = [], []
        for part in parts:
            kind = part.get("type")
            t0 = time.time()
            if kind in ("video", "video_url"):
                data = source.video(part)
                key = MD.media_key(data) + ":v%d:%d" % (self.video_budget, self.video_frames)
                got = self._media_rows.get(key)
                if got is None:
                    got = self._on_model_thread(
                        lambda: self.embedder.video(data, self.video_budget, self.video_frames))
                    print("[qwen4] video %d pairs of frames of %dx%d tokens in %.2f s"
                          % (len(got), got[0][2][0], got[0][2][1], time.time() - t0),
                          file=sys.stderr, flush=True)
                videos.append(video_text(got))
                items += [MD.Media("video", r, "%s:%d" % (key, j), grid=g)
                          for j, (_t, r, g) in enumerate(got)]
            elif kind in ("image", "image_url"):
                data, budget = source.image(part, self.image_budget)
                key = MD.media_key(data) + ":%d" % budget
                got = self._media_rows.get(key)
                if got is None:
                    got = self._on_model_thread(lambda: self.embedder.image(data, budget))
                    print("[qwen4] image %dx%d tokens in %.2f s" % (got[1][0], got[1][1],
                                                                  time.time() - t0),
                          file=sys.stderr, flush=True)
                items.append(MD.Media("image", got[0], key, grid=got[1]))
            else:
                raise ValueError("this model takes images and video, not %r" % kind)
            self._media_rows[key] = got
            while len(self._media_rows) > 64:
                self._media_rows.pop(next(iter(self._media_rows)))
        return items, videos

    def _on_model_thread(self, fn):
        """Run fn (an encoder) on the thread of the model (its OpenMP team,
        the GPU) and return its value. An encoder on the GPU first frees the
        buffer of the copies of the mixed groups (QwenGPU.free_ring), which
        holds all the free memory but 0.8 GB."""
        def job():
            if self.dev is not None and getattr(self.embedder, "gpu", False):
                if getattr(self.embedder, "lent", False):
                    self.dev.release_warm()     # the encoder takes its room back
                else:
                    self.dev.free_ring()
            return fn()
        return self.worker.call(job) if self.worker is not None else job()

    def completion_ids(self, prompt):
        return self.tokenizer.encode(prompt)

    def open_channel(self, prompt_ids):
        """The opener of the think part for the parser of the answer of this
        request (reclank/server.py keeps it with the request): "<|channel>
        thought\n" when its prompt leaves <think> open. It was a field of the
        tokenizer, set by each prompt: a client's title request (max_tokens
        64: no think part) made just after its main request set it off, and
        the main answer (its reasoning, 20669 chars) came as content."""
        return THOUGHT_OPEN if think_left_open(prompt_ids, self.tokenizer.t.special) else ""

    def parse_output(self, text):
        return qwen_parse_output(text)

    def tool_done(self, text):
        return False

    def clean_raw(self, raw):
        return _plain(raw)

    def note_error(self, exc):
        """An error of a job: a fatal one (FATAL: the CUDA context is lost)
        makes the model thread end the process after the job."""
        if FATAL.search(str(exc)):
            self.fatal = exc

    def crash_report(self, exc, where_dir=None):
        """A fatal error: write a report to a new directory under
        --crash-dir (report.json: the error, its traceback, the record of
        the GPU program queued last (reclank.backends.cuda.runtime.where), the request, the
        cache, the memory of the GPU as GpuMem knows it; prompt.npy: the
        tokens, for scripts/replay_crash.py). No CUDA call: the context is
        lost. Return the directory."""
        base = where_dir or self.crash_dir
        if not base:
            return None
        d = os.path.join(base, time.strftime("crash-%Y%m%d-%H%M%S"))
        os.makedirs(d, exist_ok=True)
        rep = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "error": str(exc),
               "traceback": "".join(traceback.format_exception(exc)),
               "context": self.ctx, "request": getattr(self, "last_request", None)}
        try:
            from reclank.backends.cuda import runtime as G
            rep["where"] = G.where()
        except Exception as e:
            rep["where"] = "unknown: %s" % e
        try:
            dev = self.dev
            cd = getattr(dev, "cache_dev", None)
            rep["device"] = {
                "last_run": getattr(dev, "last_run", None),
                "share_to": getattr(cd, "share_to", None),
                "mix_cap": getattr(dev, "mix_cap", None),
                "programs": [str(k) for k in getattr(dev, "_lru", ())],
            }
            from reclank.backends.cuda.gpumm import mem
            m = mem()
            kinds = {}
            for b in list(m.blocks.values()):
                k = kinds.setdefault(b.kind, [0, 0])
                k[0] += 1
                k[1] += b.nbytes
            rep["gpu_memory"] = {
                "kinds": {k: {"blocks": v[0], "GB": round(v[1] / 1e9, 3)} for k, v in kinds.items()},
                "lent": {str(o): [[hex(p), n] for p, n in r] for o, r in m.shared.items()},
                "placed": len(m.placed), "fences": len(m.fences), "c_reserve": m.c_reserve,
            }
        except Exception as e:
            rep["device_error"] = repr(e)
        with open(os.path.join(d, "report.json"), "w") as f:
            json.dump(rep, f, indent=1, default=str)
        ids = getattr(self, "last_prompt", None)
        if ids is not None:
            np.save(os.path.join(d, "prompt.npy"), np.asarray(ids, np.int32))
        return d

    def _fresh(self):
        self.cache = self.cache_cls(self.model.cfg, self.ctx)
        if self.dev is not None:
            self.dev.attach(self.cache)
        self.ids = []
        self.snaps.clear()
        self.h_end = None

    def _state(self):
        """The arrays of the recurrent state: DeltaNet (conv, state) and the
        convolution of the n-gram layer. The keys and values and the keys of
        the indexer are per position: a later prompt writes them again."""
        c = self.cache
        return (list(c.conv.values()) + list(c.state.values())
                + list(getattr(c, "ple_conv", {}).values()))

    def _save(self, keys, logits, stream=None):
        """A snapshot of the state after the tokens keys (logits: those of
        the last token, or None; stream: the stream of the model at the last
        token, for the MTP layer). A snapshot at the same place goes."""
        state = []
        for a in self._state():
            h = np.empty_like(a)
            if self.dev is not None:
                self.dev.cache_dev.bufs[id(a)][1].download(h)
            else:
                h[...] = a
            state.append(h)
        ple = getattr(self.cache, "ple_ids", None)
        self.snaps.add(keys, state, None if ple is None else ple.copy(),
                       None if logits is None else logits.copy(),
                       None if stream is None else stream.copy())

    def _restore(self, snap):
        """Go back to a snapshot. Return its logits (or None)."""
        keys, state, ple_ids, logits, self.h_end = snap
        for a, h in zip(self._state(), state):
            if self.dev is not None:
                self.dev.cache_dev.bufs[id(a)][1].upload(h)
            else:
                a[...] = h
        if ple_ids is not None:
            self.cache.ple_ids = ple_ids.copy()
        self.cache.n = len(keys)
        self.ids = list(keys)
        # the later snapshots of another branch stay: their keys decide
        return logits

    def _answer_cut(self, ids):
        """The position after the last <|im_start|>assistant\n of ids (the
        start of the answer), or 0. The template of the next turn writes the
        earlier answers in another form than the model wrote them (Qwen3.6
        drops the think part; Qwen3.8 has only the reasoning_content that
        the client sends back), so the next prompt differs from the cache
        from there."""
        if self.answer_head is None:
            self.answer_head = list(self.tokenizer.encode("<|im_start|>assistant\n"))
        h = self.answer_head
        m = len(h)
        for i in range(len(ids) - m, -1, -1):
            if list(ids[i:i + m]) == h:
                return i + m
        return 0

    def _prompt(self, ids, pos, media=None):
        """Read ids from pos. Return the logits of the last token. media: the
        spans of the images (cache positions)."""
        kw = {"media": media} if media else {}
        if self.dev is not None:
            for c0 in range(0, len(ids), 16384):
                self.dev.prefill(ids[c0:c0 + 16384], pos=pos + c0, **kw)
            return np.asarray(self.dev.logits()).reshape(-1)
        h = None
        for c0 in range(0, len(ids), 16384):
            h = self.model.forward(ids[c0:c0 + 16384], self.cache, start_pos=pos + c0, **kw)
        return self.model.logits(h[-1:])[0]

    def _mtp_on(self):
        return self.mtp > 0 and self.dev is not None and getattr(self.dev, "has_mtp", False)

    def _mtp_rows(self, Hp, ids, pos):
        """The MTP layer on the tokens ids at pos.., with the streams of the
        model at the positions before them (Hp). Return the streams of the
        layer; dev.logits() then gives the draft of the last row."""
        hm = None
        for c0 in range(0, len(ids), 256):
            hm = self.dev.mtp(Hp[c0:c0 + 256], ids[c0:c0 + 256], pos + c0)
        return hm

    def _prompt_mtp(self, ids, pos, media=None):
        """_prompt with the MTP layer: the rows of the prompt go to the cache
        of the MTP layer too. Return the logits of the last token and the
        stream of the model there."""
        kw = {"media": media} if media else {}
        hidden = self.dev.cfg.hidden_size * self.dev.cfg.hc_count
        h0 = self.h_end if (pos > 0 and self.h_end is not None) else np.zeros(hidden, np.float32)
        logits = None
        for c0 in range(0, len(ids), 4096):
            part = ids[c0:c0 + 4096]
            H = self.dev.prefill(part, pos=pos + c0, streams=True, **kw)
            if c0 + 4096 >= len(ids):
                logits = np.asarray(self.dev.logits()).reshape(-1)
            Hp = np.concatenate([h0.reshape(1, -1), H[:-1]])
            self._mtp_rows(Hp, part, pos + c0)
            h0 = H[-1].copy()
        return logits, h0

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        n = len(prompt_ids)
        # for a crash report: the request as it ran
        self.last_prompt = list(prompt_ids)
        self.last_request = {"prompt_tokens": n, "max_tokens": int(max_tokens),
                             "cached_before": len(self.ids), "time": time.strftime("%H:%M:%S")}
        if n + 1 > self.ctx:
            raise ValueError("the prompt has %d tokens; the context is %d" % (n, self.ctx))
        max_tokens = min(max_tokens, self.ctx - n - 1)
        spans = list(getattr(prompt_ids, "spans", ()) or ())
        keys = MD.keys(list(prompt_ids), spans) if spans else list(prompt_ids)
        # The longest start of the prompt that the cache holds: the cache as
        # it is (the last prompt and its answer), or a snapshot. A snapshot
        # with no logits must leave a token to run.
        k, snap = 0, None
        if self.cache is not None:
            k, snap = self.snaps.best(keys, self.ids, lambda sn: sn[3] is not None)
        if k == 0:
            self._fresh()
        elif snap is not None:
            logits = self._restore(snap)
        t0 = time.time()
        if k < n:
            # The M-RoPE positions of the prompt (the rows of an image), and of
            # the answer after it.
            self.cache.set_rope(MD.mrope_positions(n, spans))
            ids = list(prompt_ids)
            cut = self._answer_cut(ids)
            mtp = self._mtp_on()
            if k < cut < n:
                # a snapshot at the start of the answer, for the next turn
                if mtp:
                    _, self.h_end = self._prompt_mtp(ids[k:cut], k, spans)
                else:
                    self._prompt(ids[k:cut], k, spans)
                self._save(keys[:cut], None, self.h_end)
                k0 = cut
            else:
                k0 = k
            if mtp:
                logits, self.h_end = self._prompt_mtp(ids[k0:], k0, spans)
            else:
                logits = self._prompt(ids[k0:], k0, spans)
            self.ids = list(keys)
            self._save(keys, logits, self.h_end)
        dt = time.time() - t0
        self.last_cached = k
        self.last_request["cache_hit"] = k
        h = self.hits
        h[0], h[1], h[2] = h[0] + 1, h[1] + n, h[2] + k
        print("[qwen4] cache hit %d of %d prompt tokens (%.0f%%); new %d in %.1f s (%.0f tok/s); "
              "all %d requests: %d of %d cached (%.0f%%)" % (
                  k, n, 100.0 * k / n, n - k, dt, (n - k) / max(dt, 1e-9), h[0], h[2], h[1],
                  100.0 * h[2] / max(h[1], 1)), file=sys.stderr, flush=True)
        sampler.reset(prompt_ids)
        force = self.think_force
        # --wrap-left: the token of the answer from which fewer than that
        # many positions of the context are left (when max_tokens gets there)
        left, wrap_at = self.wrap_left, None
        room = self.ctx - n - 1
        if left > 0 and max_tokens > room - left:
            wrap_at = max(1, room - left + 1)
        guard = ThinkGuard.of(prompt_ids, self.tokenizer.t.special, eos_ids,
                              think_budget(self.think_budget, max_tokens, len(force)),
                              force, wrap_at, self.wrap_think, self.wrap_answer)
        sampler.guard = guard
        try:
            yield from self._decode(logits, n, max_tokens, sampler, eos_ids)
        finally:
            sampler.guard = None
            if guard is not None and guard.fixes:
                print("[qwen4] the model ended the turn inside <think>: </think> in its place",
                      file=sys.stderr, flush=True)
            if guard is not None and guard.wrapped:
                print("[qwen4] fewer than %d tokens of the context left at token %d of the answer: "
                      "the words to wrap up" % (self.wrap_left, guard.wrap_at), file=sys.stderr, flush=True)
            if guard is not None and guard.budget_hits:
                print("[qwen4] the think part reached its budget (%d tokens): closed" % guard.budget,
                      file=sys.stderr, flush=True)

    def _decode(self, logits, n, max_tokens, sampler, eos_ids):
        pos, count, t0 = n, 0, time.time()
        if self._mtp_on() and self.h_end is not None:
            s0 = list(self.mtp_stats)
            try:
                for token in self._decode_mtp(logits, n, max_tokens, sampler, eos_ids):
                    count += 1
                    yield token
            finally:
                dt = time.time() - t0
                dr, ac = self.mtp_stats[0] - s0[0], self.mtp_stats[1] - s0[1]
                print("[qwen4] decode %d tokens in %.1f s (%.2f tok/s); MTP %d of %d drafts "
                      "(%.0f%%); %d tokens in the cache" % (
                          count, dt, count / max(dt, 1e-9), ac, dr, 100.0 * ac / max(dr, 1),
                          len(self.ids)), file=sys.stderr, flush=True)
            return
        try:
            while count < max_tokens:
                token = sampler.fixed(sampler(logits))
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

    def _decode_mtp(self, logits, n, max_tokens, sampler, eos_ids):
        """The tokens of the answer with MTP drafts (see the module text):
        the loop of reclank.core.speculative, with the sampler of the plain
        decode (mtp_accept "in_set" keeps drafts that its settings allow)."""
        from reclank.core.speculative import QwenMTPDrafter, QwenTarget, stream
        if max_tokens <= 0:
            return
        tok = int(sampler(logits))

        def on_commit(tokens, H):
            # the cache keeps these tokens; the stream of the last is the
            # input of the MTP layer for the next token (h_end)
            self.ids.extend(tokens)
            self.h_end = H[-1].copy()

        # the MTP layer runs the rows of the tokens the model keeps; flush:
        # those of all but the last token at the end (the next prompt goes on
        # from there)
        drafter = QwenMTPDrafter(self.dev, flush_rows=True)
        drafter.observe([tok], self.h_end.reshape(1, -1), n)
        st = {}
        try:
            yield from stream(QwenTarget(self.dev, on_commit), drafter, tok, None, n, self.mtp,
                              sampler, eos_ids, max_tokens, st, room=lambda p: self.ctx - p - 2)
        finally:
            self.mtp_stats[0] += st.get("drafts", 0)
            self.mtp_stats[1] += st.get("accepted", 0)

    def _step(self, token, pos):
        if self.dev is not None:
            self.dev.step(token, pos)
            return np.asarray(self.dev.logits()).reshape(-1)
        return self.model.logits(self.model.forward([token], self.cache, start_pos=pos))[0]


def dense_label(model):
    """The form of the dense matrices for the banner: bf12 for a file of
    convert_q8_gguf.py --dense bf12 (Qwen4CPU.K keeps them unless
    RECLANK_DENSE is q8, rq8 or bf16), else the dense mode of the model."""
    g = getattr(model, "g", None)
    mode = os.environ.get("RECLANK_DENSE", "bf12")
    if g is not None and mode not in ("q8", "bf16") and any(
            t == 57 for _d, t, _o in g.tensors.values()):
        return "rq8" if mode == "rq8" else "bf12"
    return getattr(model, "dense", "-")


def load(opts, meta, worker=None):
    """Load a Qwen model with the server options (server/options.py), on the
    model thread. Return (engine, defaults)."""
    np.seterr(over="ignore")
    if getattr(opts, "moe_rows", None):
        os.environ["RECLANK_GPU_MOE_ROWS"] = opts.moe_rows      # (read as the programs compile)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path = opts.model if os.path.isabs(opts.model) else os.path.join(root, opts.model)
    arch = meta.get("general.architecture")
    # a checkpoint directory (the original Qwen3.8 in the weightkeeper): its
    # template and tokenizer inside it
    here = path if os.path.isdir(path) else os.path.dirname(path)
    if os.path.isdir(path) and meta.get("model_type") == "qwen4_exp":
        arch = "qwen4exp"
    template = opts.template or os.path.join(here, "chat_template.jinja")
    text = None
    if not os.path.exists(template):
        if arch == "qwen35moe":
            text = meta["tokenizer.chat_template"]          # the template of the GGUF
        else:
            template = os.path.join(root, "models/Qwen3.8-Flash-Next-NVFP4/chat_template.jinja")
    tok = QwenTok(opts.tokenizer or os.path.join(here, "tokenizer.json"), template, text)
    if opts.mmproj and arch not in ("qwen35moe", "qwen4exp"):
        raise ValueError("--mmproj: %s has no image input" % arch)
    gpu = opts.backend != "cpu"
    if getattr(opts, "resident", False):
        raise SystemExit("--resident: the Gemma models only for now (the Qwen experts stay in "
                         "HotCache; --hot-gb sets their memory)")
    ctx = opts.ctx or 98304
    image_budget = opts.image_budget or 1024
    print("loading %s ..." % path, flush=True)
    t0 = time.time()
    dev = None
    mode = opts.mmproj_gpu or "auto"
    if mode == "auto":
        mode = "keep" if arch == "qwen35moe" else "off"
    embedder = None
    placeholder = None
    if opts.mmproj and mode == "lend" and gpu:
        # the room of the encoder: measured, then held by a placeholder
        # while the model sizes its hot experts, then lent to the model
        from reclank.backends.cuda.gpumm import Buffer
        from reclank.models.qwen.vision import QwenEmbedder
        t1 = time.time()
        embedder = QwenEmbedder(opts.mmproj, gpu=True)
        lent = embedder.lend_gpu(max(image_budget, 1120))
        placeholder = Buffer(lent, "lend", "encoder")
        print("image encoder: %.2f GB of the GPU for images of at most %d tokens, lent to "
              "the experts between them (%.0f s)" % (lent / 1e9, embedder.budget_max,
                                                    time.time() - t1), flush=True)
    if opts.mmproj and mode == "reserve" and gpu:
        # the room of the encoder first: the hot experts are sized with the
        # memory that is free after it
        from reclank.models.qwen.vision import QwenEmbedder
        t1 = time.time()
        embedder = QwenEmbedder(opts.mmproj, gpu=True)
        held = embedder.reserve_gpu(max(image_budget, 1120))
        print("image encoder: %.2f GB of the GPU reserved (images of at most %d tokens) "
              "in %.0f s" % (held / 1e9, embedder.budget_max, time.time() - t1), flush=True)
    if arch == "qwen35moe":
        from reclank.models.qwen.qwen35 import QwenCache, QwenGGUFProgram
        model, cache_cls = QwenGGUFProgram(path), QwenCache
        if gpu:
            from reclank.models.qwen.gpu35 import QwenGPU
            dev = QwenGPU(model, hot_gb=opts.hot_gb)
    else:
        from reclank.models.qwen.qwen38 import Qwen4Cache, Qwen4CPU
        model, cache_cls = Qwen4CPU(path, experts=opts.experts), Qwen4Cache
        if gpu:
            from reclank.models.qwen.gpu38 import Qwen4GPU
            dev = Qwen4GPU(model, hot_gb=opts.hot_gb, ctx=ctx)
            if placeholder is not None:
                placeholder.free()
                dev.lend_warm(lent)
    eng = QwenEngine(model, dev, tok, ctx, cache_cls)
    eng.worker = worker
    eng.think_budget = max(0, opts.think_budget)
    eng.wrap_left = max(0, opts.wrap_left)
    eng.crash_dir = opts.crash_dir
    eng.wrap_think = tok.encode(WRAP_THINK)
    eng.wrap_answer = tok.encode(WRAP_ANSWER)
    eng.think_force = tok.encode(THINK_CLOSE)
    # MTP is on by default (3 drafts) with the layer: on the 2-socket Xeon
    # and the 3090 with the RQ8_0 file, 1 draft was best (greedy 27.7 ->
    # 30.4 tok/s; 2 or 3 drafts 29.6 and 29.0); with the MIX-BF12 file, the
    # BF12 verify groups on the tensor cores (k_kq_bf12_tc) and int16
    # experts, team 24 on a hot day: plain 16.2, 1 draft 19.7, 2 drafts
    # 21.4, 3 drafts 22.1 tok/s
    mtp = 3 if opts.mtp is None else opts.mtp
    eng.mtp = mtp if getattr(dev, "has_mtp", False) else 0
    if opts.mtp and not eng.mtp:
        print("--mtp: the model has no MTP layer on the GPU; plain steps", file=sys.stderr, flush=True)
    if opts.mmproj:
        from reclank.models.qwen.vision import QwenEmbedder
        t1 = time.time()
        if embedder is not None:
            eng.embedder = embedder
        else:
            eng.embedder = QwenEmbedder(opts.mmproj,
                                        gpu=dev is not None and mode not in ("off", "reserve", "lend"))
            eng.embedder.keep_gpu = mode == "keep"
            if mode == "keep":
                eng.embedder.warm()
        eng.image_budget = image_budget
        eng.video_budget = opts.video_budget or 128
        eng.video_frames = opts.video_frames or 32
        print("image encoder %s (%s) in %.0f s" % (
            opts.mmproj, ("GPU, " + mode) if eng.embedder.gpu else "CPU", time.time() - t1), flush=True)
    # the cache and the programs of a prompt now, so the first request does
    # not wait for them and a context that does not fit fails here
    eng._fresh()
    if dev is not None:
        eng._prompt(tok.encode("<|im_start|>user\nHello<|im_end|>\n") * 160, 0)
        eng._fresh()
    # from here on, an OpenMP team outside the planned ones is reported
    # (reclank.backends.cpu.cops.team_warn: RECLANK_TEAM_WARN, default 5 threads)
    from reclank.backends.cpu import cops as _cops
    _cops.team_warn()
    thinking = opts.thinking or "medium"
    think = {"auto": "medium", "on": "xhigh", "high": "xhigh"}.get(thinking, thinking)
    defaults = {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "max_tokens": 4096,
                "model_id": "qwen3.8-flash-next", "thinking": None if think == "off" else think}
    print("ready in %.0f s: dense %s, context %d%s; thinking %s" % (
        time.time() - t0, dense_label(model), ctx,
        ", %d hot experts in each layer" % dev.n_slots if dev else "", thinking), flush=True)
    return eng, defaults
