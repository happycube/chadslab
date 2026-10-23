"""DeepSeek V4.1 Flash in the server (reclank/models/base.py, DEEPSEEK_V41_PLAN.md
Phase 4): DSV41GPU with the routed experts in the parts on the weightkeeper's rows,
DSpark drafts (sampled or greedy: DSpark.stream), the chat of the model's own
encoding.py (the checkpoint's encoding/), its tokenizer (DSTokenizer: the BPE of
QwenTokenizer with the pre-tokenizer of tokenizer.json), and the contexts of a
few conversations between the turns (core/session.py, RowCache).

The points of a context: the window rings (128 positions) overwrite their
rows, so the state cannot go back to an earlier position; the compressed rows
and the index keys of the positions before a point stay (no later position
writes them). A point holds what goes back: the rings of every layer, the
compressors' open groups, DSpark's rings; it is taken after the last
<｜Assistant｜> of each prompt, where the next turn's prompt leaves it (the
earlier answer comes back in another form: its reasoning dropped). A prompt
starts from the longest point that is a strict start of it; the compressed
rows and the index keys of a context that another one replaced on the device
wait in host memory (RowCache) and go back with it.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

import numpy as np

from reclank.core.session import BufferRows, RowCache
from reclank.core.speculative import DeviceOp
from reclank.core.think_guard import ThinkGuard, newline_test, report, think_budget, wrap_at_of
from reclank.models.base import Engine
from reclank.models.qwen.tokenizer import QwenTokenizer

FAMILY = "deepseek41"
MODEL_TYPES = ("deepseek_v41",)
EOS = "<｜end▁of▁sentence｜>"
BOS = "<｜begin▁of▁sentence｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
# the markers of the parser of reclank/models/gemma4/chat.py (as QwenTok.decode)
THOUGHT_OPEN, THOUGHT_CLOSE = "<|channel>thought\n", "<channel|>"
# the reasoning effort of the model (encoding.py: 1 to 100; low 50, high 75, max 100)
EFFORT = {"low": 50, "medium": 62, "high": 75, "xhigh": 100, "max": 100, "on": 75, "auto": 75,
          "true": 75, "enabled": 75, "1": 75}
OFF = ("off", "false", "none", "disabled", "0", "no")
# the GPU memory the image encoder takes while it runs (an image of 1024 tokens:
# the bfloat16 weights, 0.93 GB, and the buffers of 8547 patches: 1.78 GB in all);
# the hot experts leave it free when the server takes images
VISION_GB = float(os.environ.get("RECLANK_DS_VISION_GB", "1.8"))
# the encoded rows of the images kept in host memory (a session of screenshots sends every image of
# its history each turn: a cache of 8 encoded 17 again each turn, ~20 s; an image of 1024 tokens is 21 MB)
IMAGE_CACHE_GB = float(os.environ.get("RECLANK_DS_IMAGE_CACHE_GB", "4"))
# the prompt of a lane (--lanes 2) in parts of this many tokens, with a pause after each (the other
# lane decodes between them: server/worker.py; as the Qwen engine's)
LANE_PART = int(os.environ.get("RECLANK_LANE_PART", "4096"))
FATAL_WORDS = ("illegal memory access", "unspecified launch failure", "misaligned address",
               "device-side assert", "context is destroyed", "CUDA_ERROR_LAUNCH_FAILED")


def detect(meta):
    return meta.get("model_type") in MODEL_TYPES


def load_meta(path):
    """The config of a checkpoint directory (the metadata of detect)."""
    return json.load(open(os.path.join(path, "config.json")))


class DSTokenizer(QwenTokenizer):
    """The tokenizer of tokenizer.json of DeepSeek V4.1: the BPE and the byte
    table of QwenTokenizer, no normalizer, and the pre-tokenizer of the file:
    a Sequence of Split (Isolated) patterns, each over the pieces of the one
    before (the regex module: \\p{N}, \\p{L}, \\p{P}, \\p{S})."""

    def __init__(self, path):
        super().__init__(path)
        import regex
        data = json.load(open(path))
        pre = data["pre_tokenizer"]
        steps = pre["pretokenizers"] if pre.get("type") == "Sequence" else [pre]
        self._pats = []
        for st in steps:
            if st.get("type") == "Split":
                if st.get("behavior") != "Isolated" or st.get("invert"):
                    raise ValueError("a Split of the pre-tokenizer that is not Isolated")
                self._pats.append(regex.compile(st["pattern"]["Regex"]))
            elif st.get("type") == "ByteLevel":
                if st.get("use_regex"):
                    raise ValueError("a ByteLevel with its own pattern")
            else:
                raise ValueError("a pre-tokenizer step %r" % st.get("type"))
        if data.get("normalizer") and data["normalizer"].get("normalizers"):
            raise ValueError("a normalizer in tokenizer.json")
        self.stop_ids = [self.special[EOS]]

    def split(self, text):
        pieces = [text]
        for pat in self._pats:
            out = []
            for p in pieces:
                at = 0
                for m in pat.finditer(p):
                    if m.start() > at:
                        out.append(p[at:m.start()])
                    if m.end() > m.start():
                        out.append(m.group())
                    at = m.end()
                if at < len(p):
                    out.append(p[at:])
            pieces = out
        return pieces

    def encode(self, text):
        out = []
        words, tr = self._words, self._bytes_tr
        for special, part in self._split_special(text):
            if special:
                out.append(self.special[part])
                continue
            for w in self.split(part):
                ids = words.get(w)
                if ids is None:
                    ids = self._bpe(w.encode("utf-8").decode("latin-1").translate(tr))
                    if len(words) < 500000:
                        words[w] = ids
                out += ids
        return out


class DSTok:
    """The tokenizer interface of the server: encode, decode (the think markers
    as the channel of the Gemma parser), stop_ids."""

    def __init__(self, path):
        self.t = DSTokenizer(path)
        self.stop_ids = list(self.t.stop_ids)

    def encode(self, text):
        return self.t.encode(text)

    def decode(self, ids, skip_special_tokens=False):
        t = self.t
        parts, run = [], []

        def flush():
            if run:
                parts.append(t.decode(run))
                run.clear()
        for i in ids:
            tok = t.id_to_token.get(int(i), "")
            if tok in t.special:
                flush()
                if tok == THINK_OPEN:
                    parts.append(THOUGHT_OPEN)
                elif tok == THINK_CLOSE:
                    parts.append(THOUGHT_CLOSE)
                elif tok == EOS:
                    pass
                elif not skip_special_tokens or "DSML" in tok:
                    parts.append(tok)
            else:
                run.append(i)
        flush()
        return "".join(parts)


IMAGE_TYPES = ("image_url", "image", "input_image")


def _content(content, images):
    """The content of a message for encoding.py: its text (a string, or a list
    of parts joined), or, with images, the blocks of encoding.py (text, and
    image_url with the index of the part in images as its url; encoding.py
    gives one placeholder for each, in the order of the prompt)."""
    if content is None or isinstance(content, str):
        return content
    if not any(isinstance(p, dict) and p.get("type") in IMAGE_TYPES for p in content):
        out = []
        for p in content:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"):
                out.append(p.get("text", ""))
            elif isinstance(p, dict) and p.get("type") in ("video_url", "video", "input_audio"):
                raise ValueError("DeepSeek V4.1 takes images, not %r" % p.get("type"))
        return "".join(out)
    blocks = []
    for p in content:
        if isinstance(p, str):
            blocks.append({"type": "text", "text": p})
        elif not isinstance(p, dict):
            continue
        elif p.get("type") in ("text", "input_text", "output_text"):
            blocks.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") in IMAGE_TYPES:
            if p.get("type") == "input_image":      # (the Responses form: image_url a string)
                p = {"type": "image_url", "image_url": p.get("image_url")}
            blocks.append({"type": "image_url", "image_url": {"url": "#image%d" % len(images)}})
            images.append(p)
        else:
            raise ValueError("DeepSeek V4.1 takes images, not %r" % p.get("type"))
    return blocks


class DSEngine(Engine):
    """DSV41GPU, DSpark (or plain steps), the contexts of the turns."""

    family = FAMILY

    def __init__(self, m, g, ds, tok, enc, ctx, vision=None, worker=None, contexts=4, contexts_gb=16.0):
        self.m, self.g, self.ds = m, g, ds
        self.vision = vision
        self.worker = worker
        self.image_budget = 1024        # the most tokens of an image (the model's max_image_tokens)
        self._media_rows = {}           # the rows of the last images by key
        self.last_media = []
        self.tokenizer = tok
        self.enc = enc
        self.max_context = ctx
        self.ctxs = RowCache(self, contexts, contexts_gb * 1e9, "dsv41")
        self.last_cached = 0
        self.spec = "auto"       # DSpark: "auto", True (a draft every round)
        self.assistant_id = tok.t.special[ASSISTANT]
        self.think_open_id = tok.t.special[THINK_OPEN]
        self.think_close_id = tok.t.special[THINK_CLOSE]
        self.last_text = ""
        # the think budget (--think-budget), the words to wrap up (--wrap-left) and their tokens (load):
        # the request's ThinkGuard (think_guard)
        self.think_budget, self.wrap_left = 6000, 500
        self.think_force = self.wrap_think = self.wrap_answer = ()
        self.ends_line = None
        # the lanes (--lanes 2, server/worker.py; lanes.py): coop hands the runs of a generation to
        # the scheduler (DeviceOp); lane is the DSLane of this engine (None: lane A, the model's own
        # state); lanes are the engines of all the lanes (this one alone: [self])
        self.coop = False
        self.lane = None
        self.lanes = [self]
        self.index = 0

    # ---- the prompt

    def think_level(self, req, default):
        """The reasoning effort of a request (1 to 100), or None (the chat
        mode): reasoning_effort (a level or a number), reasoning.effort,
        chat_template_kwargs (thinking, enable_thinking, reasoning_effort),
        thinking ({"type": "enabled"}), else default."""
        kw = req.get("chat_template_kwargs") or {}
        level = default
        reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
        for v in (kw.get("reasoning_effort"), reasoning.get("effort"), req.get("reasoning_effort")):
            if v is None:
                continue
            level = self._effort(v, level)
        for v in (kw.get("thinking"), kw.get("enable_thinking"), req.get("thinking")):
            if isinstance(v, dict):
                v = v.get("type")
            if v is None:
                continue
            if str(v).lower() in OFF:
                level = None
            elif level is None:
                level = EFFORT["high"]
        return level

    @staticmethod
    def _effort(v, default):
        s = str(v).lower()
        if s in OFF:
            return None
        if s in EFFORT:
            return EFFORT[s]
        try:
            n = int(float(s))
        except ValueError:
            return default
        return min(100, max(1, n))

    def think_off(self):
        return None

    def prompt_ids(self, messages, level, tools=None, preserve=None, source=None):
        msgs, images = [], []
        for m in messages:
            m = dict(m)
            if "content" in m:
                m["content"] = _content(m["content"], images)
            if m.get("role") == "developer":
                m["role"] = "system"
            msgs.append(m)
        if tools:
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = dict(msgs[0], tools=tools)
            else:
                msgs.insert(0, {"role": "system", "content": "", "tools": tools})
        mode = "thinking" if level else "chat"
        drop = not preserve
        text, media = self.enc.encode_messages(msgs, thinking_mode=mode, drop_thinking=drop,
                                               reasoning_effort=level if level else None,
                                               return_multi_modal_data=True)
        self.last_text = text
        ids = self.tokenizer.encode(text)
        if not images:
            return ids
        order = [int(r["url"][len("#image"):]) for r in media["images"]]
        return self._expand(ids, [images[i] for i in order], source)

    def _expand(self, ids, parts, source):
        """Each placeholder of ids (image_token_id) as the span of the image of
        its part (in order): a PromptIds with a media.Span for each (its rows
        the input of layer 0 at its positions)."""
        from reclank.core.messages import PromptIds
        from reclank.encoders import media as MD
        if self.vision is None:
            raise ValueError("this server takes no image input (started with --mmproj off)")
        tid = self.vision.image_token_id
        if sum(1 for t in ids if t == tid) != len(parts):
            raise ValueError("the prompt has %d image placeholders for %d images"
                             % (sum(1 for t in ids if t == tid), len(parts)))
        out, spans, it = [], [], iter(parts)
        for t in ids:
            if t != tid:
                out.append(t)
                continue
            rows, grid, key = self._image(next(it), source)
            spans.append(MD.Span(len(out), rows, False, "image", key, grid))
            out += [tid] * rows.shape[0]
        res = PromptIds(out)
        res.spans = spans
        self.last_media = [(sp.kind, sp.rows.shape[0], sp.key) for sp in spans]
        return res

    def _image(self, part, source):
        """The rows of the span of an image part, its grid of tokens, its key."""
        from reclank.encoders import media as MD
        if source is None:
            from server.media import MediaSource
            source = MediaSource()
        data, budget = source.image(part, self.image_budget)
        budget = max(8, min(int(budget), self.vision.max_tokens))
        key = MD.media_key(data) + ":%d" % budget
        got = self._media_rows.get(key)
        if got is None:
            t0 = time.time()
            got = self._on_model_thread(lambda: self._encode(data, budget))
            print("[dsv41] image %dx%d tokens (%d in the span) in %.2f s" % (
                got[1][0], got[1][1], got[0].shape[0], time.time() - t0), file=sys.stderr, flush=True)
        self._media_rows.pop(key, None)
        self._media_rows[key] = got
        while len(self._media_rows) > 1 and \
                sum(r.nbytes for r, _g in self._media_rows.values()) > IMAGE_CACHE_GB * 1e9:
            self._media_rows.pop(next(iter(self._media_rows)))
        return got[0], got[1], key

    def _encode(self, data, budget):
        """The encoder on the GPU; its memory freed after (the model's hot
        experts and programs hold the rest of the GPU). The programs of a long
        session grow with the context (1.3 GB at 48K tokens), so when the room
        the encoder needs (VISION_GB for an image of max_tokens) is not free,
        or the encoder runs out of memory, the model's programs go first (made
        again at the next run: fit does the same as cap grows)."""
        from reclank.backends.cuda.gpumm import mem_info
        need = VISION_GB * 1e9 * max(budget, 256) / self.vision.max_tokens + 0.2e9
        if mem_info()[0] < need:
            self.g.free_programs()
        try:
            while True:
                try:
                    return self.vision.image(data, budget)
                except (RuntimeError, MemoryError) as e:
                    if "memory" not in str(e).lower():
                        raise
                    print("[dsv41] the image encoder out of GPU memory (%s): room made, again" % e,
                          file=sys.stderr, flush=True)
                    self.vision.release()
                    # the programs, then (if they went for room just before) a slot of the hot experts
                    if not self.g.make_room(e):
                        raise
        finally:
            self.vision.release()

    def _on_model_thread(self, fn):
        return self.worker.call(fn) if self.worker is not None else fn()

    def completion_ids(self, prompt):
        """The ids of a plain completion, after <｜begin▁of▁sentence｜> (the
        model starts every sequence with it; tokenizer.json does not add it)."""
        ids = self.tokenizer.encode(prompt)
        bos = self.tokenizer.t.special[BOS]
        return ids if ids[:1] == [bos] else [bos] + ids

    def open_channel(self, prompt_ids):
        """The opener of the think part when the prompt ends inside it (thinking
        mode: ...<｜Assistant｜><think>)."""
        ids = list(prompt_ids)
        start = max((i for i, t in enumerate(ids) if t == self.assistant_id), default=-1)
        tail = ids[start + 1:]
        if self.think_open_id in tail and self.think_close_id not in tail[tail.index(self.think_open_id):]:
            return THOUGHT_OPEN
        return ""

    def parse_output(self, text):
        """The reasoning (the channel), the content, the DSML tool calls (the
        checkpoint's parse_tool_calls); a block of calls that is not closed is
        pending."""
        from reclank.models.gemma4.chat import parse_output as gemma_parse_output
        head = "<" + self.enc.dsml_token + self.enc.tool_calls_block_name
        # The block of calls is cut from the text before the Gemma parser,
        # which drops what looks like its special tokens (<|...|>, <|word>,
        # <word|>) from the content: the code of a call keeps them. A block
        # counts after the reasoning (not in a channel still open).
        c = text.rfind(THOUGHT_CLOSE)
        after = c + len(THOUGHT_CLOSE) if c >= 0 else 0
        i = text.find(head, after) if text.find(THOUGHT_OPEN, after) < 0 else -1
        block = text[i:] if i >= 0 else ""
        p = gemma_parse_output(text[:i] if i >= 0 else text)
        body = p["content"]
        calls, pending = [], ""
        if i >= 0:
            j = len(head)
            try:
                _idx, stop, raw = self.enc.parse_tool_calls(j, block)
                ok = stop == "</" + self.enc.dsml_token + self.enc.tool_calls_block_name + ">"
            except (ValueError, AssertionError, IndexError):
                ok, raw = False, []
            if ok:
                for c in raw:
                    calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}})
            else:
                pending = block
        else:
            for k in range(min(len(body), len(head) - 1), 0, -1):
                if head.startswith(body[-k:]):
                    body, pending = body[:-k], body[-k:]
                    break
        return {"reasoning": p["reasoning"], "content": body.strip(),
                "tool_calls": p["tool_calls"] + calls, "pending": pending}

    def tool_done(self, text):
        return False

    def turn_fields(self, rec):
        return {"prompt_text": self.last_text, "cached_tokens": self.last_cached}

    def note_error(self, exc):
        if any(w.lower() in str(exc).lower() for w in FATAL_WORDS):
            self.fatal = exc

    # ---- the state and the contexts (core/session.py, RowCache)

    def rows(self):
        """The compressed rows and the index keys of each layer: a row for each
        block of ratio positions."""
        g, m = self.g, self.m
        out = []
        for d in (g.ckv, g.kcache):
            for i, a in d.items():
                out.append(BufferRows(g.mirror.buffer_of(a), m.max_len // a.shape[0], a.nbytes // a.shape[0]))
        return out

    def reserve(self, n):
        pass

    def _state_arrays(self):
        g = self.g
        arrs = list(g.window) + list(g.kstate.values()) + list(g.sstate.values())
        if self.ds is not None:
            arrs += list(self.ds.rings)
        return arrs

    def _save(self, ids):
        """A point at len(ids) (ids: the keys of the positions, _keys): the
        state arrays, and the engram hashes' ids of the last positions (the
        look-back of the next tokens: a request between two turns writes
        others there)."""
        g = self.g
        saved = []
        for a in self._state_arrays():
            g.mirror.buffer_of(a).download(a)
            saved.append(a.copy())
        s = len(ids)
        back = self.m.hashes.cache[max(0, s - 8):s].copy()
        self.ctxs.point(ids, (saved, back))

    def _restore(self, point, s):
        g = self.g
        for a, v in zip(self._state_arrays(), point[0]):
            a[:] = v
            g.mirror.buffer_of(a).upload(a)
        self.m.hashes.cache[max(0, s - 8):s] = point[1]

    @staticmethod
    def _keys(ids):
        """The keys of the positions of a prompt: the ids, the first position of
        each image span its key (two images of the same size have the same
        ids)."""
        keys = list(ids)
        for sp in getattr(ids, "spans", ()) or ():
            keys[sp.start] = ("image", sp.key)
        return keys

    def _fresh(self):
        self.g.reset()
        if self.ds is not None:
            self.ds.reset()

    def _answer_cut(self, ids):
        """The position after the last <｜Assistant｜> of the prompt (the start
        that the next turn keeps), or 0."""
        k = max((i for i, t in enumerate(ids) if t == self.assistant_id), default=-1)
        return k + 1

    def _prefill(self, ids, pos, media=None):
        if self.ds is not None:
            return self.ds.prefill(ids, pos=pos, media=media)
        return self.g.prefill(ids, pos, media=media), None

    def _prefill_co(self, ids, pos, media=None):
        """(yield from) _prefill; with coop in parts of LANE_PART tokens, a pause (DeviceOp) after
        each but the last, so that the other lane runs between them (a prompt in parts gives the
        bits of one: the cut at the answer does the same)."""
        if not self.coop or len(ids) <= LANE_PART:
            return self._prefill(ids, pos, media)
        out = None
        for c0 in range(0, len(ids), LANE_PART):
            if c0:
                yield DeviceOp("pause")
            out = self._prefill(ids[c0:c0 + LANE_PART], pos + c0, media)
        return out

    # ---- the lanes (server/worker.py LaneScheduler)

    def on(self):
        """The model with this engine's lane in place (DSLane.view; lane A: as it is)."""
        import contextlib
        return self.lane.view() if self.lane is not None else contextlib.nullcontext(self.g)

    def pair(self, a, b):
        """The runs of lane A (this engine: the model's state) and lane B (lanes[1]) at once
        (lanes.run_both; their outputs are then each lane's g.last): two steps. Not verify
        groups (False: each alone): a group already overlaps its halves, and two groups bound the
        CPU (check_dsv41_lanes.py: plain pairs 1.34x the tokens of one after the other, DSpark
        pairs 0.92x); DSpark's auto then weighs the paired plain steps against its rounds."""
        if len(a.tokens) != 1 or len(b.tokens) != 1:
            return False
        from reclank.models.deepseek41.lanes import run_both
        run_both(self.g, self.lanes[1].lane, a.tokens, a.pos, b.tokens, b.pos)
        return True

    def hit(self, prompt_ids):
        """What this lane's contexts would give a prompt (RowCache.peek): (the tokens they hold,
        True when that is the context on this lane's device). On the model thread."""
        return self.ctxs.peek(self._keys(prompt_ids), lambda p: False)

    # ---- the generation

    def think_guard(self, prompt_ids, max_tokens, eos_ids):
        """The ThinkGuard of a request (core/think_guard.py) when the prompt ends inside <think>
        (open_channel) or the answer may reach the end of the context, else None: the think part
        closes at think_budget tokens (and at most half of max_tokens), and the words to wrap up go in
        when fewer than wrap_left positions of the context are left. (No tool markers: a DSML call
        is not one token.)"""
        think = bool(self.open_channel(prompt_ids))
        wrap_at = wrap_at_of(self.wrap_left, self.max_context - len(prompt_ids) - 1, max_tokens)
        if not think and wrap_at is None:
            return None
        force = self.think_force
        return ThinkGuard(eos_ids, self.think_close_id, think_budget(self.think_budget, max_tokens, len(force)),
                          force, think=think, wrap_at=wrap_at, wrap_think=self.wrap_think,
                          wrap_answer=self.wrap_answer, ends_line=self.ends_line, para=True)

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        ids = list(prompt_ids)
        keys = self._keys(prompt_ids)
        media = [(sp.start, sp.rows) for sp in getattr(prompt_ids, "spans", ()) or ()] or None
        n = len(ids)
        if n + 1 > self.max_context:
            raise ValueError("the prompt has %d tokens; the context is %d" % (n, self.max_context))
        max_tokens = min(int(max_tokens), self.max_context - n - 1)
        t0 = time.time()
        # after a longer sequence the programs' cap comes back down and the hot experts that
        # make_room took grow back while the GPU has room (no run is half done here)
        self.g.relax(n)
        # the longest point that is a strict start of the prompt (its context
        # brought to the device)
        k, point = self.ctxs.begin(keys, lambda p: False)
        cut = self._answer_cut(ids)

        def prefill():
            if k == 0:
                self._fresh()
            else:
                self._restore(point, k)
            if k < cut < n:
                yield from self._prefill_co(ids[k:cut], k, media)
                self._save(keys[:cut])
                k0 = cut
            else:
                k0 = k
            return (yield from self._prefill_co(ids[k0:], k0, media))

        try:
            lg, main_last = yield from prefill()
        except MemoryError as e:
            # the programs of a long session take more as the context grows (as _encode): a new one
            # (an image in the window: the vis program) can find the GPU full. The programs go (made
            # again when run), the context back to the point, and the prompt again.
            from reclank.backends.cuda.gpumm import mem_info
            free0 = mem_info()[0]
            self.g.free_programs()
            print("[dsv41] %s: the programs freed (GPU %.2f -> %.2f GB free), the prompt again" % (
                e, free0 / 1e9, mem_info()[0] / 1e9), file=sys.stderr, flush=True)
            lg, main_last = yield from prefill()
        dt = time.time() - t0
        self.last_cached = k
        print("[dsv41] cache hit %d of %d prompt tokens; new %d in %.1f s (%.0f tok/s); %s" % (
            k, n, n - k, dt, (n - k) / max(dt, 1e-9), self.ctxs.note(n, k)), file=sys.stderr, flush=True)
        sampler.reset(prompt_ids)
        guard = sampler.guard = self.think_guard(prompt_ids, max_tokens, eos_ids)
        t1, count, stats = time.time(), 0, {}
        try:
            if self.ds is not None:
                for tok in self.ds.stream(lg, main_last, n, sampler, max_tokens=max_tokens, eos=eos_ids,
                                          spec=self.spec, stats=stats, coop=self.coop):
                    if not isinstance(tok, DeviceOp):
                        count += 1
                    yield tok
            else:
                pos, y = n, sampler.fixed(sampler(lg))
                while True:
                    count += 1
                    yield y
                    if y in eos_ids or count >= max_tokens:
                        break
                    if self.coop:
                        yield DeviceOp("run", [y], pos)
                        y, pos = sampler.fixed(sampler(self.g.last.logits[0].copy())), pos + 1
                    else:
                        y, pos = sampler.fixed(sampler(self.g.step(y, pos))), pos + 1
        finally:
            sampler.guard = None
            report(guard, "dsv41", self.wrap_left)
            dt = time.time() - t1
            extra = ""
            if stats.get("rounds"):
                extra = "; DSpark %d rounds (%d plain steps), %.0f%% of %d drafts" % (
                    stats["rounds"], stats["plain"], 100.0 * stats["accepted"] / max(stats["drafted"], 1),
                    stats["drafted"])
            print("[dsv41] decode %d tokens in %.1f s (%.1f tok/s)%s" % (count, dt, count / max(dt, 1e-9), extra),
                  file=sys.stderr, flush=True)


def load(opts, meta, worker=None):
    """Load DeepSeek V4.1 Flash with the server options (server/options.py), on
    the model thread. Return (engine, defaults)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from reclank.models.deepseek41.dspark import DSpark
    from reclank.models.deepseek41.gpu import HOT_GB, HOT_GB_DSPARK, DSV41GPU
    from reclank.models.deepseek41.model import DSV41CPU
    path = opts.model
    t0 = time.time()
    print("loading %s ..." % path, flush=True)
    sys.path.insert(0, os.path.join(path, "encoding"))
    import encoding as enc
    tok = DSTok(opts.tokenizer or os.path.join(path, "tokenizer.json"))
    mtp = 1 if opts.mtp is None else opts.mtp
    m = DSV41CPU(path, max_len=opts.ctx or None)
    images = str(opts.mmproj or "").lower() not in ("off", "none", "0")
    hot = opts.hot_gb if opts.hot_gb is not None else (HOT_GB_DSPARK if mtp else HOT_GB) - (VISION_GB if images else 0)
    # (spec: the prefetch of the cold experts, one lane alone: gpu.py SPEC)
    g = DSV41GPU(m, keep_layers=False, hot_gb=hot, spec=int(getattr(opts, "lanes", 1) or 1) == 1)
    ds = DSpark(g) if mtp else None
    vision = None
    if images:
        # the ViT and the aligner are in the checkpoint (--mmproj off: no image input)
        from reclank.models.deepseek41.vision import DSVision
        vision = DSVision(m.w.src, meta)
    eng = DSEngine(m, g, ds, tok, enc, m.max_len, vision, worker, contexts=getattr(opts, "contexts", 4),
                   contexts_gb=getattr(opts, "contexts_gb", 16.0))
    if opts.image_budget:
        eng.image_budget = min(int(opts.image_budget), vision.max_tokens if vision else 1024)
    from reclank.models.qwen.chat import THINK_CLOSE as CLOSE_WORDS, WRAP_ANSWER, WRAP_THINK
    eng.think_budget, eng.wrap_left = max(0, opts.think_budget), max(0, opts.wrap_left)
    eng.think_force, eng.wrap_think, eng.wrap_answer = (tok.encode(CLOSE_WORDS), tok.encode(WRAP_THINK),
                                                        tok.encode(WRAP_ANSWER))
    eng.ends_line = newline_test(tok.decode)
    # the programs of a prompt and of the loop now (the first request does not wait for them)
    warm = tok.encode(enc.encode_messages([{"role": "user", "content": "Hello. " * 300}], thinking_mode="chat"))
    eng._fresh()
    if ds is not None:
        ds.prefill(warm)
        ds.warm()
    else:
        g.prefill(warm)
    eng._fresh()
    if int(getattr(opts, "lanes", 1) or 1) > 1:
        _lanes_up(eng, warm)
        _lanes_warm(eng, tok, enc)
    from reclank.backends.cpu import cops as _cops
    _cops.team_warn()
    thinking = opts.thinking or "high"
    level = None if thinking == "off" else EFFORT.get(thinking, EFFORT["high"])
    if os.environ.get("RECLANK_DS_EFFORT"):
        # a default effort as a number (1 to 100; --thinking takes the names only), as a request's
        # reasoning_effort may give
        level = DSEngine._effort(os.environ["RECLANK_DS_EFFORT"], level)
    defaults = {"temperature": 1.0, "top_p": 0.95, "top_k": None, "max_tokens": 65536,
                "model_id": "deepseek-v4.1-flash", "thinking": level}
    from reclank.backends.cuda.gpumm import mem_info
    print("ready in %.0f s: context %d, %d hot experts in each layer (%.1f GB), DSpark %s; thinking %s "
          "(effort %s); images %s; GPU %.1f GB free" % (
              time.time() - t0, eng.max_context, g.n_slots, g.n_slots * m.L * g.per / 1e9, "on" if ds else "off",
              thinking, level, "on (at most %d tokens)" % eng.image_budget if vision else "off",
              mem_info()[0] / 1e9), flush=True)
    return eng, defaults


def _lanes_up(eng, warm):
    """--lanes 2: lane B (lanes.py DSLane: a second sequence of the model, its own state on the
    GPU, its caches as long as the model's so that the lanes share the contexts in host memory)
    and its engine; both hand their runs to the scheduler of the server (coop). warm: a prompt for
    lane B's prompt programs."""
    import copy
    from reclank.models.deepseek41.lanes import DSLane
    t0 = time.time()
    g, ds = eng.g, eng.ds
    lane = DSLane(g, ds)
    eb = copy.copy(eng)             # (the model, the tokenizer, the encoder, the images' rows shared)
    eb.ctxs = RowCache(eb, eng.ctxs.max_contexts, eng.ctxs.max_bytes, "dsv41 B")
    eb.ctxs.share_with(eng.ctxs)
    eb.last_cached, eb.last_text, eb.last_media = 0, "", []
    eb.lane, eb.index = lane, 1
    eng.lanes = eb.lanes = [eng, eb]
    eng.coop = eb.coop = True
    with eb.on():
        # its programs of a prompt and of the loop now (as the model's at the load)
        eb._fresh()
        if ds is not None:
            ds.prefill(warm)
            ds.warm()
        else:
            g.prefill(warm)
        eb._fresh()
    from reclank.backends.cuda.gpumm import mem_info
    print("lane B: %d positions, its programs in %.0f s; GPU %.1f GB free" % (
        lane.state["max_pos"], time.time() - t0, mem_info()[0] / 1e9), flush=True)


def _lanes_warm(eng, tok, enc):
    """Two short answers at once through the scheduler of the lanes (server/worker.py): the first
    runs of lane B's programs on its stream, and a check of the pairs."""
    import queue
    import threading
    from reclank.core.sampling import Sampler
    from server.worker import LaneScheduler
    t0 = time.time()
    sch = LaneScheduler(eng.lanes)
    qs = []
    for i, text in enumerate(("Name three colors.", "Count from one to five.")):
        e = eng.lanes[i]
        ids = tok.encode(enc.encode_messages([{"role": "user", "content": text}], thinking_mode="chat"))
        q = queue.Queue()
        sch.admit(i, (lambda e=e, ids=ids: e.generate(ids, 32, Sampler(temperature=0.0), e.stop_ids())),
                  q, threading.Event())
        qs.append(q)
    errs = []
    while sch.busy():
        sch.tick(errs.append)
    if errs:
        raise errs[0]
    outs = []
    for q in qs:
        out = []
        while not q.empty():
            kind, v = q.get()
            if kind == "item":
                out.append(v)
        outs.append(tok.decode(out))
    for e in eng.lanes:
        with e.on():
            e._fresh()
    print("lanes: a warm pair in %.1f s (%d pairs, %d runs alone): %r / %r" % (
        time.time() - t0, sch.pairs, sch.solos, outs[0][:40], outs[1][:40]), flush=True)
