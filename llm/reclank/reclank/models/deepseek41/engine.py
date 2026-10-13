"""DeepSeek V4.1 Flash in the server (reclank/models/base.py, DEEPSEEK_V41_PLAN.md
Phase 4): DSV41GPU with the routed experts in the parts on the weightkeeper's rows,
DSpark drafts (sampled or greedy: DSpark.stream), the chat of the model's own
encoding.py (the checkpoint's encoding/), its tokenizer (DSTokenizer: the BPE of
QwenTokenizer with the pre-tokenizer of tokenizer.json), and snapshots of the
state between the turns.

The snapshots: the window rings (128 positions) overwrite their rows, so the
state cannot go back to an earlier position; the compressed rows and the
index keys of the positions before a point stay (no later position writes
them). A snapshot holds what goes back: the rings of every layer, the
compressors' open groups, DSpark's rings; it is taken after the last
<｜Assistant｜> of each prompt, where the next turn's prompt leaves it (the
earlier answer comes back in another form: its reasoning dropped). A prompt
starts from the longest snapshot that is a strict start of it.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

import numpy as np

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
    """DSV41GPU, DSpark (or plain steps), the snapshots of the turns."""

    family = FAMILY

    def __init__(self, m, g, ds, tok, enc, ctx, vision=None, worker=None):
        self.m, self.g, self.ds = m, g, ds
        self.vision = vision
        self.worker = worker
        self.image_budget = 1024        # the most tokens of an image (the model's max_image_tokens)
        self._media_rows = {}           # the rows of the last images by key
        self.last_media = []
        self.tokenizer = tok
        self.enc = enc
        self.max_context = ctx
        self.snaps = []          # (prompt start ids, the state), the oldest first
        self.n_snaps = 4
        self.last_cached = 0
        self.hits = [0, 0, 0]    # requests, prompt tokens, cached tokens
        self.spec = "auto"       # DSpark: "auto", True (a draft every round)
        self.assistant_id = tok.t.special[ASSISTANT]
        self.think_open_id = tok.t.special[THINK_OPEN]
        self.think_close_id = tok.t.special[THINK_CLOSE]
        self.last_text = ""

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
        while len(self._media_rows) > 8:
            self._media_rows.pop(next(iter(self._media_rows)))
        return got[0], got[1], key

    def _encode(self, data, budget):
        """The encoder on the GPU; its memory freed after (the model's hot
        experts and programs hold the rest of the GPU)."""
        try:
            return self.vision.image(data, budget)
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

    # ---- the state and the snapshots

    def _state_arrays(self):
        g = self.g
        arrs = list(g.window) + list(g.kstate.values()) + list(g.sstate.values())
        if self.ds is not None:
            arrs += list(self.ds.rings)
        return arrs

    def _save(self, ids):
        """A snapshot at len(ids) (ids: the keys of the positions, _keys): the
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
        self.snaps = [sn for sn in self.snaps if sn[0] != ids]
        self.snaps.append((list(ids), saved, back))
        if len(self.snaps) > self.n_snaps:
            self.snaps.pop(0)

    def _restore(self, snap):
        g = self.g
        for a, v in zip(self._state_arrays(), snap[1]):
            a[:] = v
            g.mirror.buffer_of(a).upload(a)
        s = len(snap[0])
        self.m.hashes.cache[max(0, s - 8):s] = snap[2]

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

    # ---- the generation

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        ids = list(prompt_ids)
        keys = self._keys(prompt_ids)
        media = [(sp.start, sp.rows) for sp in getattr(prompt_ids, "spans", ()) or ()] or None
        n = len(ids)
        if n + 1 > self.max_context:
            raise ValueError("the prompt has %d tokens; the context is %d" % (n, self.max_context))
        max_tokens = min(int(max_tokens), self.max_context - n - 1)
        best = None
        for snap in self.snaps:
            s = len(snap[0])
            if s < n and keys[:s] == snap[0] and (best is None or s > len(best[0])):
                best = snap
        if best is None:
            self._fresh()
            k = 0
        else:
            self._restore(best)
            k = len(best[0])
        t0 = time.time()
        cut = self._answer_cut(ids)
        if k < cut < n:
            self._prefill(ids[k:cut], k, media)
            self._save(keys[:cut])
            k0 = cut
        else:
            k0 = k
        lg, main_last = self._prefill(ids[k0:], k0, media)
        dt = time.time() - t0
        self.last_cached = k
        h = self.hits
        h[0], h[1], h[2] = h[0] + 1, h[1] + n, h[2] + k
        print("[dsv41] cache hit %d of %d prompt tokens; new %d in %.1f s (%.0f tok/s); all %d requests: "
              "%d of %d cached" % (k, n, n - k, dt, (n - k) / max(dt, 1e-9), h[0], h[2], h[1]),
              file=sys.stderr, flush=True)
        sampler.reset(prompt_ids)
        t1, count, stats = time.time(), 0, {}
        try:
            if self.ds is not None:
                for tok in self.ds.stream(lg, main_last, n, sampler, max_tokens=max_tokens, eos=eos_ids,
                                          spec=self.spec, stats=stats):
                    count += 1
                    yield tok
            else:
                pos, y = n, sampler(lg)
                while True:
                    count += 1
                    yield y
                    if y in eos_ids or count >= max_tokens:
                        break
                    y, pos = sampler(self.g.step(y, pos)), pos + 1
        finally:
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
    g = DSV41GPU(m, keep_layers=False, hot_gb=hot)
    ds = DSpark(g) if mtp else None
    vision = None
    if images:
        # the ViT and the aligner are in the checkpoint (--mmproj off: no image input)
        from reclank.models.deepseek41.vision import DSVision
        vision = DSVision(m.w.src, meta)
    eng = DSEngine(m, g, ds, tok, enc, m.max_len, vision, worker)
    if opts.image_budget:
        eng.image_budget = min(int(opts.image_budget), vision.max_tokens if vision else 1024)
    # the programs of a prompt and of the loop now (the first request does not wait for them)
    warm = tok.encode(enc.encode_messages([{"role": "user", "content": "Hello. " * 300}], thinking_mode="chat"))
    eng._fresh()
    if ds is not None:
        ds.prefill(warm)
        ds.warm()
    else:
        g.prefill(warm)
    eng._fresh()
    from reclank.backends.cpu import cops as _cops
    _cops.team_warn()
    thinking = opts.thinking or "high"
    level = None if thinking == "off" else EFFORT.get(thinking, EFFORT["high"])
    defaults = {"temperature": 1.0, "top_p": 0.95, "top_k": None, "max_tokens": 65536,
                "model_id": "deepseek-v4.1-flash", "thinking": level}
    from reclank.backends.cuda.gpumm import mem_info
    print("ready in %.0f s: context %d, %d hot experts in each layer (%.1f GB), DSpark %s; thinking %s "
          "(effort %s); images %s; GPU %.1f GB free" % (
              time.time() - t0, eng.max_context, g.n_slots, g.n_slots * m.L * g.per / 1e9, "on" if ds else "off",
              thinking, level, "on (at most %d tokens)" % eng.image_budget if vision else "off",
              mem_info()[0] / 1e9), flush=True)
    return eng, defaults
