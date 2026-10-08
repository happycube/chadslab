"""An OpenAI compatible HTTP server for the NumPy Gemma runtime.

The server uses only the Python standard library. It gives these paths:

    GET  /v1/models
    GET  /v1/models/{id}
    POST /v1/chat/completions
    POST /v1/completions
    GET  /health

The chat path takes the fields that the OpenAI clients send: model, messages,
max_tokens, temperature, top_k, top_p, min_p, stop, stream, seed, and the three
penalties. A stream uses the server-sent-event format of the OpenAI API. The
server also takes the bare paths with no /v1.

The model is not thread safe and shares one key and value cache. The server
runs one request at a time. Other requests wait.
"""
from __future__ import annotations

import hmac
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .chat import parse_output
from .model import Session
from .sampling import Sampler


# A stream sends this often while it waits for a token. A prompt read can take
# minutes, and a client closes a stream that stays silent for too long.
KEEPALIVE_SECONDS = 15.0


def content_text(content):
    """Return the text of a message content. Take a string or a part list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("text") is not None:
                    out.append(str(part["text"]))
            else:
                out.append(str(part))
        return "".join(out)
    return str(content)


def stop_strings(value):
    """Return the list of stop strings. Take a string or a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def truncate(text, stops):
    """Cut the text at the first stop string. Return the text and the cut."""
    cut = None
    for s in stops:
        i = text.find(s)
        if i >= 0 and (cut is None or i < cut):
            cut = i
    if cut is None:
        return text, False
    return text[:cut], True


REPEAT_MIN = 3
REPEAT_MAX = 256
# The text of an answer is decoded and parsed again for each new token. A
# long answer does it every 1 + len // PARSE_STEP tokens: the cost of a
# parse grows with the text (33 ms at 30000 tokens), and it holds the GIL
# that the model thread needs. The last parse, at the end, is the same.
PARSE_STEP = 2000


def parse_due(n):
    """True when the answer of n tokens is parsed again now."""
    return n % (1 + n // PARSE_STEP) == 0


# A loop repeats a phrase: a block of fewer distinct characters (a line of
# "=", the "| --- " of a table) is text, not a loop.
REPEAT_CHARS = 8
# Inside a tool call that is not complete (the file of a write), a line of
# code can repeat a few times, so a loop needs more copies there.
REPEAT_MIN_CALL = 8


def open_call(text):
    """Return True when text ends inside a tool call (Qwen <tool_call>, or
    Gemma <|tool_call>)."""
    return (text.rfind("<tool_call>") > text.rfind("</tool_call>")
            or text.rfind("<|tool_call>") > text.rfind("<tool_call|>"))


def repeat_len(text):
    """Return the length of a repeated tail, or 0 when the text is fine.

    A greedy decode can fall into a loop and repeat one phrase to the token
    limit. A tail that holds the same block (REPEAT_CHARS distinct
    characters or more) three times or more is a loop; REPEAT_MIN_CALL
    times inside a tool call.
    """
    n = len(text)
    copies = REPEAT_MIN_CALL if open_call(text) else REPEAT_MIN
    for size in range(16, REPEAT_MAX + 1, 4):
        need = size * copies
        if n < need:
            continue
        block = text[n - size:]
        if len(set(block)) >= REPEAT_CHARS and text[n - need:] == block * copies:
            return need
    return 0


def think_loop(sampler):
    """A repeated tail in an open think part: True when the guard of the
    sampler (np_gemma/think_guard.py) closes the think part (once), so the
    model writes the answer; False: the turn stops at the loop."""
    g = getattr(sampler, "guard", None)
    if g is None or not hasattr(g, "close_now"):
        return False
    before = g.loops
    if g.close_now():
        if g.loops > before:        # (once: with MTP the tokens of a group came before)
            print("[np-gemma] a loop in the think part: closed", file=sys.stderr, flush=True)
        return True
    return False


def cut_repeat(text):
    """Remove a repeated tail, keeping the first copy of the block."""
    need = repeat_len(text)
    if not need:
        return text
    copies = REPEAT_MIN_CALL if open_call(text) else REPEAT_MIN
    return text[:len(text) - need + need // copies]


def tool_done(text):
    """Return True when the model asks the caller for a tool result.

    The model writes <|tool_response> to ask for the result. Without this test
    the model repeats the marker until it reaches the token limit. The marker
    also stops a turn that waits for a result with no new call."""
    return "<|tool_response>" in text


# The key of the API: a request needs "Authorization: Bearer <key>" (or the
# header x-api-key). The scripts take --api-key, else NP_GEMMA_API_KEY, else
# this; an empty key turns the check off.
DEFAULT_API_KEY = "change-me"
# The largest body of a request (a prompt with images as data URIs).
MAX_BODY = int(os.environ.get("NP_GEMMA_MAX_BODY", str(64 << 20)))
# The directory of the raw answers (--debug), or None: the raw text of an
# answer (its reasoning too) goes to no file.
RAW_DIR = None


def _write_raw(name, raw):
    """raw to RAW_DIR/name, readable by the owner only. Return the path, or
    None (no RAW_DIR, or an error)."""
    if not RAW_DIR:
        return None
    path = os.path.join(RAW_DIR, name)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(raw)
        return path
    except OSError:
        return None


def log_answer(reasoning, content, calls, pending, raw):
    """The parts of an answer to the log, and (with RAW_DIR) its raw text to
    last-answer.txt there."""
    _write_raw("last-answer.txt", raw)
    print('[np-gemma] answer: reasoning %d chars, content %d chars, %d tool calls, unparsed %d chars; '
          'end %r' % (reasoning, content, calls, pending, raw[-120:]), file=sys.stderr, flush=True)


def log_empty(reasoning_chars, raw):
    """An answer with no content and no tool call: (with RAW_DIR) the raw
    text to a file there, and its start and end to the log."""
    name = _write_raw("empty-%s-%s.txt" % (time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:6]),
                      raw) or "(not written: no --debug)"
    print('[np-gemma] empty answer (reasoning %d chars): raw %d chars in %s; start %r; end %r' % (
        reasoning_chars, len(raw), name, raw[:160], raw[-160:]), file=sys.stderr, flush=True)


def usage_of(backend, prompt_ids, out):
    """The usage of a response. A backend with last_cached (the prompt
    tokens that the cache held) gives prompt_tokens_details.cached_tokens."""
    usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(out),
             "total_tokens": len(prompt_ids) + len(out)}
    cached = getattr(backend, "last_cached", None)
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": int(cached)}
    return usage


MEDIA_TYPES = ("image", "image_url", "audio", "input_audio", "video", "video_url")


def has_media(content):
    """Return True when a content part list has an image, audio, or video part."""
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in MEDIA_TYPES for p in content)


def message_parts(content):
    """Return a content part list for render_chat: the text parts (type
    text) and the media parts, in order."""
    out = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") in MEDIA_TYPES:
                out.append(part)
            elif part.get("text") is not None:
                out.append({"type": "text", "text": str(part["text"])})
        else:
            out.append({"type": "text", "text": str(part)})
    return out


class PromptIds(list):
    """The token ids of a prompt, with the soft-token spans of its media
    (media.Span) in .spans."""
    spans = ()


def common_len(a, b):
    """Return the count of the first items that a and b share."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


THOUGHT_OPEN = "<|channel>thought\n"
OFF = ("none", "off", "disabled", "disable", "false", "0", "no")


def request_thinking(req):
    """The think part that a request asks for: True, False, or None (the
    default of the server). A request sets it with "thinking" (true, false,
    or {"type": "enabled"} as the DeepSeek API), "reasoning_effort",
    "reasoning": {"effort": ...}, or "chat_template_kwargs": {"enable_thinking":
    ...} or {"reasoning_effort": ...}."""
    kw = req.get("chat_template_kwargs") or {}
    reasoning = req.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    for v in (kw.get("enable_thinking"), req.get("thinking"),
              kw.get("reasoning_effort"), effort, req.get("reasoning_effort")):
        if isinstance(v, dict):
            v = v.get("type")
        if v is None:
            continue
        return str(v).lower() not in OFF
    return None


class Backend:
    """Hold the model, the tokenizer, and the default settings."""

    def __init__(self, model, tokenizer, cfg=None, model_id="np-gemma",
                 thinking=False, max_tokens=1024, temperature=1.0, top_k=None,
                 top_p=None, drafter=None, n_draft=2, empty_thought_block=True,
                 max_context=None, mtp_accept="exact", mtp_floor=None):
        self.model = model
        # The most tokens of a request (the prompt and the answer); None has
        # no limit. A longer prompt gets an error, and max_tokens gets the
        # room that is left.
        self.max_context = max_context
        # False for the E2B and E4B models (Tokenizer.apply_chat_template).
        self.empty_thought_block = empty_thought_block
        # The MTP drafter and the count of drafts for each step. None turns
        # MTP off.
        self.drafter = drafter
        self.n_draft = n_draft
        # The rule of the MTP drafts with sampling (Sampler mtp_accept and
        # mtp_floor). A request can give "mtp_accept" and "mtp_floor".
        self.mtp_accept = mtp_accept
        self.mtp_floor = mtp_floor
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model_id = model_id
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.lock = threading.Lock()
        # Keep the key and value cache between the turns of a chat. The model
        # then reads only the new tokens of a turn, not the whole history.
        self.sessions = []
        self.max_sessions = 4
        # A shared prefix below this count is not worth a cache hit, and it
        # would evict the conversation that the cache should keep.
        self.min_shared = 16
        # --debug: the directory of the records of the turns (on_turn).
        self.debug = None
        self.turn_n = 0
        self.last_think = None
        self.last_text = None
        # Image and audio input (--mmproj): the embedder of the soft rows, the
        # default image budget, the directory of local media files (None:
        # only data URIs), and a cache of soft rows by media key.
        self.embedder = None
        self.image_budget = 280
        self.media_dir = None
        self.video_budget = 70
        self.video_frames = 32
        self._media_rows = {}
        self.last_media = []

    def model_info(self, name=None):
        """Return one model description."""
        info = {"id": name or self.model_id, "object": "model", "created": 0,
                "owned_by": "np-gemma"}
        if self.max_context:
            info["context_length"] = self.max_context
        return info

    def prompt_ids(self, messages, thinking=None, tools=None):
        """Return the token ids of a chat prompt."""
        think = self.thinking if thinking is None else bool(thinking)
        # Keep every field. tool_calls on an assistant message and tool_call_id
        # on a tool message carry the tool result to the model.
        msgs = []
        for m in messages:
            msg = dict(m)
            msg["role"] = m.get("role", "user")
            c = m.get("content")
            msg["content"] = message_parts(c) if has_media(c) else content_text(c)
            msgs.append(msg)
        parts = []
        text = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, thinking=think, tools=tools,
            empty_thought_block=self.empty_thought_block, media=parts)
        if parts:
            return self.media_prompt(text, parts, think)
        self.last_think, self.last_text = think, text
        return self.tokenizer.encode(text)

    def on_turn(self, rec):
        """--debug: write the record of a turn (_turn) to a file, and a line
        to turns.log."""
        if not self.debug:
            return
        self.turn_n += 1
        rec = dict(rec, think=self.last_think, prompt_text=self.last_text,
                   sampling={"temperature": self.temperature, "top_k": self.top_k,
                             "top_p": self.top_p})
        name = os.path.join(self.debug, "%05d-%s.json" % (self.turn_n, time.strftime("%H%M%S")))
        with open(name, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        req = rec["request"] or {}
        with open(os.path.join(self.debug, "turns.log"), "a", encoding="utf-8") as f:
            f.write("%s %s messages=%d tools=%d prompt=%d out=%d finish=%s think=%s "
                    "reasoning=%d content=%d calls=%s %.0fs\n" % (
                        rec["time"], os.path.basename(name), len(req.get("messages") or []),
                        len(req.get("tools") or []), rec["prompt_tokens"], rec["out_tokens"],
                        rec["finish"], rec["think"], len(rec["reasoning"]),
                        len(rec["content"] or ""),
                        ",".join(c["function"]["name"] for c in rec["tool_calls"]) or "-",
                        rec["seconds"]))

    def media_prompt(self, text, parts, think):
        """Return the PromptIds of a prompt with media parts: the soft rows of
        each part (the embedder), and their tokens in the ids."""
        from . import media as MD
        if self.embedder is None:
            raise ValueError("this server takes no image or audio input "
                             "(start it with --mmproj)")
        items = []
        videos = []
        for part in parts:
            kind = part.get("type")
            if kind in ("video", "video_url"):
                # The frames and their times go into the text (media.video_text),
                # and each frame is one item.
                data = self._video_source(part)
                key = MD.media_key(data) + ":v%d:%d" % (self.video_budget, self.video_frames)
                frames = self._media_rows.get(key)
                if frames is None:
                    frames = self.embedder.video(data, self.video_budget, self.video_frames)
                self._media_rows[key] = frames
                videos.append(MD.video_text(frames))
                for j, (_t, rows) in enumerate(frames):
                    items.append(MD.Media("video", rows, "%s:%d" % (key, j),
                                          bidir=getattr(self.embedder, "bidir", True)))
                continue
            if kind in ("image", "image_url"):
                data, budget = self._image_source(part)
                key = MD.media_key(data) + ":%d" % budget
                rows = self._media_rows.get(key)
                if rows is None:
                    rows, _ = self.embedder.image(data, budget)
                items.append(MD.Media("image", rows, key,
                                      bidir=getattr(self.embedder, "bidir", True)))
            else:
                data = self._audio_source(part)
                key = MD.media_key(data)
                rows = self._media_rows.get(key)
                if rows is None:
                    rows = self.embedder.audio(data)
                items.append(MD.Media("audio", rows, key))
            self._media_rows[key] = rows
            while len(self._media_rows) > 64:
                self._media_rows.pop(next(iter(self._media_rows)))
        if videos:
            pieces = text.split("<|video|>")
            if len(pieces) != len(videos) + 1:
                raise ValueError("the prompt has %d video placeholders for %d videos"
                                 % (len(pieces) - 1, len(videos)))
            text = pieces[0] + "".join(v + p for v, p in zip(videos, pieces[1:]))
        ids, spans = MD.expand(self.tokenizer.encode(text), items)
        out = PromptIds(ids)
        out.spans = spans
        self.last_media = [(sp.kind, sp.rows.shape[0], sp.key) for sp in spans]
        self.last_think, self.last_text = think, text
        print("[np-gemma] media: %s" % ", ".join(
            "%s %d tokens" % (sp.kind, sp.rows.shape[0]) for sp in spans),
            file=sys.stderr, flush=True)
        return out

    def _source_bytes(self, ref):
        """Return the bytes of a data URI, or of a file under media_dir."""
        import base64
        import os
        ref = str(ref or "")
        if ref.startswith("data:"):
            head, _, body = ref.partition(",")
            return base64.b64decode(body) if ";base64" in head else body.encode()
        if ref.startswith(("http://", "https://")):
            raise ValueError("this server does not fetch media from the network; "
                             "send a data URI")
        if self.media_dir is None:
            raise ValueError("send the media as a data URI (the server has no --media-dir)")
        path = os.path.realpath(ref[7:] if ref.startswith("file://") else ref)
        root = os.path.realpath(self.media_dir)
        if not path.startswith(root + os.sep):
            raise ValueError("the media path is not under --media-dir")
        with open(path, "rb") as f:
            return f.read()

    def _image_source(self, part):
        """Return (bytes, budget) of an image part."""
        v = part.get("image_url", part.get("image", part.get("url")))
        detail = None
        if isinstance(v, dict):
            detail = v.get("detail")
            v = v.get("url")
        budget = {"low": 70, "high": 1120}.get(detail, self.image_budget)
        if part.get("max_soft_tokens") is not None:
            budget = int(part["max_soft_tokens"])
        return self._source_bytes(v), budget

    def _video_source(self, part):
        """Return the bytes of a video part (video_url or video: a data URI or
        a path under media_dir)."""
        v = part.get("video_url", part.get("video", part.get("url")))
        if isinstance(v, dict):
            v = v.get("url")
        return self._source_bytes(v)

    def _audio_source(self, part):
        """Return the bytes of an audio part (input_audio data, or a URI)."""
        import base64
        v = part.get("input_audio", part.get("audio", part.get("url")))
        if isinstance(v, dict):
            if v.get("data") is not None:
                return base64.b64decode(v["data"])
            v = v.get("url")
        return self._source_bytes(v)

    def open_channel(self, prompt_ids):
        """Return "<|channel>thought\n" if the prompt ends in that open
        channel, else "". The Gemma 4 template ends the prompt so after a
        tool result when the thought channel is on; the model then writes
        its thought with no opener and closes it with <channel|>."""
        tail = self.tokenizer.decode(list(prompt_ids[-4:]), skip_special_tokens=False)
        return THOUGHT_OPEN if tail.endswith(THOUGHT_OPEN) else ""

    def completion_ids(self, prompt):
        """Return the token ids of a raw completion prompt."""
        return self.tokenizer.encode(prompt)

    def sampler(self, req):
        """Build the sampler from the request fields and the defaults."""
        def pick(name, default):
            v = req.get(name)
            return default if v is None else v
        return Sampler(temperature=pick("temperature", self.temperature),
                       top_k=pick("top_k", self.top_k), top_p=pick("top_p", self.top_p),
                       min_p=req.get("min_p"),
                       repetition_penalty=req.get("repetition_penalty"),
                       presence_penalty=req.get("presence_penalty"),
                       frequency_penalty=req.get("frequency_penalty"),
                       seed=req.get("seed"),
                       mtp_accept=pick("mtp_accept", self.mtp_accept),
                       mtp_floor=pick("mtp_floor", self.mtp_floor))

    def stop_ids(self):
        """Return the token ids that end a generation."""
        return set(getattr(self.tokenizer, "stop_ids", ()) or ())

    # The hooks of the output of a model family (the Qwen backend of
    # scripts/serve_qwen4.py has its own): the parse of the raw text, the
    # test for a turn that waits for a tool result, and the raw text for the
    # log and the files.
    def parse_output(self, text):
        return parse_output(text)

    def tool_done(self, text):
        return tool_done(text)

    def clean_raw(self, raw):
        return raw

    def applied_max_tokens(self, n):
        """The max_tokens that a request of n gets (a backend with a floor
        raises it; the response says so in X-Max-Tokens-Applied)."""
        return n

    def pick_session(self, prompt_ids, max_tokens):
        """Return the session with the longest shared prefix, or a new one.

        A hit moves the session to the front, so a smaller cache keeps the
        conversation that is in use and drops the one that is not. The soft
        tokens of media match only the same media (Session.common).
        """
        best = None
        best_n = 0
        spans = getattr(prompt_ids, "spans", None)
        for session in self.sessions:
            n = (session.common(prompt_ids, spans) if hasattr(session, "common")
                 else common_len(session.ids, prompt_ids))
            if n > best_n:
                best = session
                best_n = n
        if best is not None and best_n >= self.min_shared:
            self.sessions.remove(best)
            self.sessions.insert(0, best)
            return best, best_n
        best = Session(self.model, max_len=len(prompt_ids) + max_tokens + 8,
                       drafter=self.drafter, n_draft=self.n_draft)
        self.sessions.insert(0, best)
        del self.sessions[self.max_sessions:]
        return best, 0

    def think_guard(self, prompt_ids, max_tokens, eos_ids):
        """The ThinkGuard of a request (np_gemma/think_guard.py) when the
        server set think_tokens (serve.py: the ids of the thought channel
        of Gemma 4, the words of the budget and of the wrap-up), else None:
        the think part closes at think_budget tokens (and at most half of
        max_tokens), and fewer than wrap_left tokens of the context left
        (max_context, else 262144) put in the words to wrap up."""
        tt = getattr(self, "think_tokens", None)
        if tt is None:
            return None
        from .think_guard import ThinkGuard, think_budget
        left, wrap_at = getattr(self, "wrap_left", 0), None
        room = (self.max_context or 262144) - len(prompt_ids) - 1
        if left > 0 and max_tokens > room - left:
            wrap_at = max(1, room - left + 1)
        return ThinkGuard(eos_ids, tt["close"],
                          think_budget(getattr(self, "think_budget", 0), max_tokens, len(tt["force"])),
                          tt["force"], think=bool(self.open_channel(prompt_ids)), wrap_at=wrap_at,
                          wrap_think=tt["wrap_think"], wrap_answer=tt["wrap_answer"],
                          tool=(tt["tool_open"], tt["tool_close"]), open_id=tt["open"])

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        """Yield one token id at a time for a prompt."""
        session, shared = self.pick_session(prompt_ids, max_tokens)
        print('[np-gemma] reuse=%d new=%d' % (shared, len(prompt_ids) - shared),
              file=sys.stderr, flush=True)
        guard = self.think_guard(prompt_ids, max_tokens, eos_ids)
        sampler.guard = guard
        # the rates: the prompt (its new tokens, to the first token of the
        # answer) and the decode (the other tokens)
        t0, t1, n = time.time(), None, 0
        try:
            spans = getattr(prompt_ids, "spans", None)
            kw = {"media": spans} if spans else {}
            for token in session.generate_stream(prompt_ids, max_new_tokens=max_tokens,
                                                 eos_ids=eos_ids, sampler=sampler, **kw):
                if t1 is None:
                    t1 = time.time()
                n += 1
                yield token
        finally:
            if t1 is not None:
                # the tokens the session read (Session.prefill: a truncate
                # that fails reads all of them, whatever pick_session found)
                t2, new = time.time(), getattr(session, "prefilled", len(prompt_ids) - shared)
                print("[np-gemma] prompt %d new tokens in %.2f s (%.0f tok/s); decode %d tokens in "
                      "%.1f s (%.1f tok/s)" % (new, t1 - t0, new / max(t1 - t0, 1e-9), n - 1, t2 - t1,
                                               (n - 1) / max(t2 - t1, 1e-9)), file=sys.stderr, flush=True)
            # also when the caller stops at a stop token and closes the
            # generator
            sampler.guard = None
            if guard is not None and guard.fixes:
                print("[np-gemma] the model ended the turn inside the thought channel: <channel|> "
                      "in its place", file=sys.stderr, flush=True)
            if guard is not None and guard.wrapped:
                print("[np-gemma] fewer than %d tokens of the context left at token %d of the answer: "
                      "the words to wrap up" % (self.wrap_left, guard.wrap_at), file=sys.stderr,
                      flush=True)
            if guard is not None and guard.budget_hits:
                print("[np-gemma] the think part reached its budget (%d tokens): closed" % guard.budget,
                      file=sys.stderr, flush=True)
            st = session.mtp_stats
            if self.drafter is not None and st.get("drafts"):
                print('[np-gemma] mtp steps=%d drafts=%d accepted=%d (%d%%)' % (
                    st["steps"], st["drafts"], st["accepted"],
                    100 * st["accepted"] // st["drafts"]), file=sys.stderr, flush=True)


class _BodyError(Exception):
    """A body that the server does not read (status: 400 or 413)."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "np-gemma/1.0"

    # ---- small helpers -----------------------------------------------------

    def log_message(self, fmt, *args):
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    def _cors(self):
        """The CORS header, only for the origin of --cors-origin."""
        origin = getattr(self.server, "cors_origin", None)
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            if origin != "*":
                self.send_header("Vary", "Origin")

    def _authorized(self):
        """True when the server has no key or the request gives it (Bearer,
        or x-api-key). Else send 401."""
        key = getattr(self.server, "api_key", None)
        if not key:
            return True
        auth = self.headers.get("Authorization") or ""
        given = auth[7:].strip() if auth[:7].lower() == "bearer " else (self.headers.get("x-api-key") or "")
        if hmac.compare_digest(given.encode("utf-8"), key.encode("utf-8")):
            return True
        data = json.dumps({"error": {"message": "invalid or missing API key",
                                     "type": "invalid_request_error",
                                     "code": "invalid_api_key"}}).encode("utf-8")
        self.send_response(401)
        self._cors()
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return False

    def _max_header(self):
        """X-Max-Tokens-Applied when the backend changed max_tokens (a floor:
        backend.applied_max_tokens), so a client sees the limit that ran."""
        note = getattr(self, "_max_note", None)
        if note:
            self.send_header("X-Max-Tokens-Applied", str(note[1]))
            self.send_header("X-Max-Tokens-Requested", str(note[0]))

    def _json(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self._max_header()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message, kind="invalid_request_error", code=None):
        self._json(status, {"error": {"message": message, "type": kind, "code": code}})

    def _body(self):
        """The JSON of the body. A Content-Length that is not a number, is
        negative, or is above the limit (MAX_BODY) gives _BodyError; the
        server does not read such a body and closes the connection."""
        text = self.headers.get("Content-Length")
        try:
            n = int(text) if text is not None else 0
        except ValueError:
            n = -1
        limit = getattr(self.server, "max_body", MAX_BODY)
        if n < 0:
            self.close_connection = True
            raise _BodyError(400, "invalid Content-Length: %r" % text)
        if n > limit:
            self.close_connection = True
            raise _BodyError(413, "the body has %d bytes; the limit is %d" % (n, limit))
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw) if raw.strip() else {}

    def _sse_start(self):
        self._sent_at = time.monotonic()        # the keepalive counts from here
        self.send_response(200)
        self._cors()
        self._max_header()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse(self, obj):
        data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
        self.wfile.write(("%x\r\n" % len(data)).encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()
        self._sent_at = time.monotonic()

    def _sse_data(self, text):
        data = ("data: " + text + "\n\n").encode("utf-8")
        self.wfile.write(("%x\r\n" % len(data)).encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _sse_end(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # ---- routes ------------------------------------------------------------

    def do_OPTIONS(self):
        """The CORS preflight: allowed only for --cors-origin."""
        self.send_response(204)
        if getattr(self.server, "cors_origin", None):
            self._cors()
            self.send_header("Access-Control-Allow-Headers", "authorization, content-type, x-api-key")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/")
        backend = self.server.backend
        if path in ("", "/health", "/healthz", "/v1/health"):
            self._json(200, {"status": "ok"})          # no key: a check of the process
            return
        if not self._authorized():
            return
        if path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [backend.model_info()]})
        elif path.startswith("/v1/models/") or path.startswith("/models/"):
            self._json(200, backend.model_info(path.rsplit("/", 1)[-1]))
        else:
            self._error(404, "unknown path: " + path)

    def do_POST(self):
        if not self._authorized():
            self.close_connection = True       # the body stays unread
            return
        path = urlsplit(self.path).path.rstrip("/")
        if path.endswith("/chat/completions"):
            self._chat()
        elif path.endswith("/completions"):
            self._completions()
        else:
            self._error(404, "unknown path: " + path)

    def _chat(self):
        try:
            req = self._body()
        except _BodyError as exc:
            self._error(exc.status, str(exc))
            return
        except Exception as exc:
            self._error(400, "invalid JSON: %s" % exc)
            return
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            self._error(400, "messages must be a non-empty list")
            return
        backend = self.server.backend
        thinking = request_thinking(req)
        print('[np-gemma] messages=%d tools=%d' % (len(messages),
              len(req.get("tools") or [])), file=sys.stderr, flush=True)
        below = getattr(backend, "no_think_below", 0)
        try:
            lim = int(req.get("max_tokens", req.get("max_completion_tokens")))
        except (TypeError, ValueError):
            lim = None          # (the error comes later, from _run)
        if below and lim is not None and lim < below and thinking is not False and \
                not hasattr(backend, "chat_prompt_ids"):         # (serve_qwen4 has its own)
            # --no-think-below: a small request (a client's title, 64 tokens)
            # answers with no think part; the thought took all its tokens
            print("[np-gemma] max_tokens %d < %d: no think part" % (lim, below), file=sys.stderr,
                  flush=True)
            thinking = False
        try:
            if hasattr(backend, "chat_prompt_ids"):
                # a backend with more fields of the request (the Qwen server:
                # reasoning_effort, chat_template_kwargs)
                prompt_ids = backend.chat_prompt_ids(req)
            else:
                prompt_ids = backend.prompt_ids(messages, thinking, req.get("tools"))
        except Exception as exc:
            self._error(400, "prompt failed: %s" % exc)
            return
        self._run(req, prompt_ids, chat=True)

    def _completions(self):
        try:
            req = self._body()
        except _BodyError as exc:
            self._error(exc.status, str(exc))
            return
        except Exception as exc:
            self._error(400, "invalid JSON: %s" % exc)
            return
        prompt = req.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not isinstance(prompt, str):
            prompt = str(prompt)
        prompt_ids = self.server.backend.completion_ids(prompt)
        self._run(req, prompt_ids, chat=False)

    def _turn(self, backend, prompt_ids, out, raw, parsed, content, calls, finish):
        """A record of the turn for backend.on_turn (a debug mode), if the
        backend has it."""
        hook = getattr(backend, "on_turn", None)
        if hook is None:
            return
        try:
            hook({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - self._t0, 2),
                  "request": self._req, "prompt_tokens": len(prompt_ids), "out_tokens": len(out),
                  "finish": finish, "raw": backend.clean_raw(raw), "reasoning": parsed["reasoning"], "content": content,
                  "tool_calls": calls, "unparsed": parsed.get("pending") or ""})
        except Exception as exc:
            print("[np-gemma] on_turn failed: %s" % exc, file=sys.stderr, flush=True)

    def _run(self, req, prompt_ids, chat):
        backend = self.server.backend
        self._req, self._t0 = req, time.time()
        # A prompt can end in an open thought channel (the Gemma 4 template
        # after a tool result). The answer then starts in the thought, with
        # no opener: parse it with the opener in front.
        opener = getattr(backend, "open_channel", None)
        self._open = opener(prompt_ids) if (chat and opener) else ""
        max_tokens = req.get("max_tokens")
        if max_tokens is None:
            max_tokens = req.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = backend.max_tokens
        self._max_note = None
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            self._error(400, "max_tokens must be an integer")
            return
        if max_tokens < 1:
            self._error(400, "max_tokens must be at least 1")
            return
        applied = backend.applied_max_tokens(max_tokens)
        if applied != max_tokens:
            print("[np-gemma] max_tokens %d -> %d (the floor of the server)" % (max_tokens, applied),
                  file=sys.stderr, flush=True)
            self._max_note = (max_tokens, applied)
            max_tokens = applied
        if backend.max_context:
            room = backend.max_context - len(prompt_ids)
            if room <= 0:
                # the words and the code of the OpenAI API: a client that
                # knows them compacts its history and sends it again
                self._error(400, "This model's maximum context length is %d tokens. However, your "
                            "messages resulted in %d tokens. Please reduce the length of the "
                            "messages." % (backend.max_context, len(prompt_ids)),
                            "invalid_request_error", "context_length_exceeded")
                return
            max_tokens = min(max_tokens, room)
        print('[np-gemma] max_tokens=%d stream=%s' % (max_tokens, bool(req.get('stream'))),
              file=sys.stderr, flush=True)
        stops = stop_strings(req.get("stop"))
        sampler = backend.sampler(req)
        eos = backend.stop_ids()
        model_id = req.get("model") or backend.model_id
        if req.get("stream"):
            self._stream(req, prompt_ids, max_tokens, sampler, stops, eos, model_id, chat)
        else:
            self._block(prompt_ids, max_tokens, sampler, stops, eos, model_id, chat)

    def _ids(self, prompt_ids, max_tokens, sampler, eos):
        """Run the generation. Return the token ids and the finish reason."""
        backend = self.server.backend
        tok = backend.tokenizer
        out = []
        finish = "length"
        for token in backend.generate(prompt_ids, max_tokens, sampler, eos):
            out.append(token)
            if token in eos:
                finish = "stop"
                break
            if not parse_due(len(out)):
                continue
            raw = self._raw(tok, out)
            if backend.tool_done(raw):
                finish = ("tool_calls" if backend.parse_output(raw)["tool_calls"]
                          else "stop")
                break
            if repeat_len(raw):
                if think_loop(sampler):
                    continue        # the think part closes; the answer comes
                finish = "stop"
                break
        return out, finish

    def _raw(self, tok, out):
        """The text of the answer, with the opener of a thought channel that
        the prompt left open (see _run)."""
        text = tok.decode(out, skip_special_tokens=False) if out else ""
        return getattr(self, "_open", "") + text

    def _block(self, prompt_ids, max_tokens, sampler, stops, eos, model_id, chat):
        backend = self.server.backend
        try:
            with backend.lock:
                out, finish = self._ids(prompt_ids, max_tokens, sampler, eos)
        except Exception as exc:
            # an answer, not a closed connection (the client sees why); the
            # log gets the trace
            print("[np-gemma] generation failed:", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            self._error(500, "generation failed: %s: %s" % (type(exc).__name__, exc), "server_error")
            return
        print('[np-gemma] prompt=%d out=%d finish=%s' % (len(prompt_ids), len(out), finish),
              file=sys.stderr, flush=True)
        raw = self._raw(backend.tokenizer, out)
        parsed = backend.parse_output(raw)
        content = parsed["content"]
        if not parsed["tool_calls"] and parsed.get("pending") and finish != "length":
            content = (content + "\n" + parsed["pending"]).strip()   # a call that did not close
        text, cut = truncate(cut_repeat(content), stops)
        calls = parsed["tool_calls"]
        log_answer(len(parsed["reasoning"]), len(text), len(calls), len(parsed.get("pending") or ""),
                   backend.clean_raw(raw))
        self._turn(backend, prompt_ids, out, raw, parsed, text, calls, finish)
        if not text and not calls:
            log_empty(len(parsed["reasoning"]), backend.clean_raw(raw))
        if calls:
            finish = "tool_calls"
        elif cut:
            finish = "stop"
        created = int(time.time())
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        usage = usage_of(backend, prompt_ids, out)
        if chat:
            obj = "chat.completion"
            message = {"role": "assistant",
                       "content": (text or None) if calls else text}
            if parsed["reasoning"]:
                message["reasoning_content"] = parsed["reasoning"]
            if calls:
                message["tool_calls"] = calls
            choice = {"index": 0, "message": message, "finish_reason": finish}
        else:
            obj = "text_completion"
            choice = {"index": 0, "text": text, "logprobs": None,
                      "finish_reason": "stop" if cut else finish}
        self._json(200, {"id": cid, "object": obj, "created": created,
                         "model": model_id, "choices": [choice], "usage": usage})

    def _stream(self, req, prompt_ids, max_tokens, sampler, stops, eos, model_id, chat):
        backend = self.server.backend
        tok = backend.tokenizer
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        obj = "chat.completion.chunk" if chat else "text_completion"

        def event(delta, finish):
            # chat: the delta of the message; a completion: the text in the
            # choice (the form of the OpenAI completions stream)
            choice = ({"index": 0, "delta": delta, "finish_reason": finish} if chat else
                      {"index": 0, "text": delta.get("text", ""), "logprobs": None,
                       "finish_reason": finish})
            return {"id": cid, "object": obj, "created": created,
                    "model": model_id, "choices": [choice]}

        def piece(kind, text):
            if kind == "reasoning_content":
                return {"reasoning_content": text}
            return {"content": text} if chat else {"text": text}

        self._sse_start()
        out = []
        sent_reason = ""
        sent_content = ""
        calls = []
        finish = "length"

        def advance(sent, value, kind):
            """Send the part of value that sent does not hold. Return the new sent.

            A reader joins the deltas, so a shorter value cannot correct one that
            already went out. Forget sent when the value no longer extends it.
            """
            if not value.startswith(sent):
                sent = ""
            if len(value) > len(sent):
                self._sse(event(piece(kind, value[len(sent):]), None))
                sent = value
            return sent

        # Read the prompt and generate on a worker, so this thread can keep the
        # connection alive. A long prompt read sends no token for minutes, and a
        # client closes a stream that stays silent for too long.
        q = queue.Queue()
        done = threading.Event()

        def worker():
            try:
                with backend.lock:
                    for token in backend.generate(prompt_ids, max_tokens, sampler, eos):
                        q.put(("token", token))
                        if done.is_set():
                            break
                q.put(("end", None))
            except BaseException as exc:
                # The stream sends only the message; the log gets the trace.
                print("[np-gemma] generation failed:", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                q.put(("error", exc))

        threading.Thread(target=worker, daemon=True).start()
        t_start = time.monotonic()
        closed = False
        empty = None
        shape = None
        try:
            if chat:
                self._sse(event({"role": "assistant", "content": ""}, None))
            while True:
                if time.monotonic() - self._sent_at >= KEEPALIVE_SECONDS:
                    # Tokens can come for minutes with nothing to send (a
                    # tool call that is not complete). A client closes a
                    # stream that stays silent, and a write finds a closed
                    # stream.
                    self._sse(event({}, None))
                try:
                    kind, value = q.get(timeout=KEEPALIVE_SECONDS)
                except queue.Empty:
                    continue
                if kind == "end":
                    break
                if kind == "error":
                    raise value
                token = value
                out.append(token)
                if token in eos:
                    finish = "stop"
                    done.set()
                    break
                if not parse_due(len(out)):
                    continue
                raw = self._raw(tok, out)
                parsed = backend.parse_output(raw)
                calls = parsed["tool_calls"]
                if backend.tool_done(raw):
                    finish = "tool_calls" if calls else "stop"
                    done.set()
                    break
                if repeat_len(raw) and not think_loop(sampler):
                    finish = "stop"
                    sent_content = advance(sent_content,
                                           truncate(cut_repeat(parsed["content"]), stops)[0],
                                           "content")
                    done.set()
                    break
                if chat:
                    sent_reason = advance(sent_reason, parsed["reasoning"],
                                          "reasoning_content")
                text, cut = truncate(cut_repeat(parsed["content"]), stops)
                sent_content = advance(sent_content, text, "content")
                if cut:
                    finish = "stop"
                    done.set()
                    break
            raw = self._raw(tok, out)
            parsed = backend.parse_output(raw)
            calls = parsed["tool_calls"]
            # the tokens after the last parse of the loop (parse_due)
            if chat:
                sent_reason = advance(sent_reason, parsed["reasoning"], "reasoning_content")
            sent_content = advance(sent_content, truncate(cut_repeat(parsed["content"]), stops)[0],
                                   "content")
            if not calls and parsed.get("pending") and finish != "length":
                # A tool call that the parser could not close: its text, not
                # nothing (a client shows an empty answer as an error). A call
                # that max_tokens cut is not text: finish is "length".
                sent_content = advance(sent_content, (parsed["content"] + "\n" + parsed["pending"]).strip(),
                                       "content")
            if not sent_content and not calls:
                empty = (len(parsed["reasoning"]), raw)
            shape = (len(parsed["reasoning"]), len(sent_content), len(calls),
                     len(parsed.get("pending") or ""), backend.clean_raw(raw))
            if calls:
                finish = "tool_calls"
                for i, call in enumerate(calls):
                    self._sse(event({"tool_calls": [{
                        "index": i, "id": call["id"], "type": "function",
                        "function": {"name": call["function"]["name"],
                                     "arguments": call["function"]["arguments"]}}]}, None))
            self._sse(event({}, finish))
            if include_usage:
                self._sse({"id": cid, "object": obj, "created": created,
                           "model": model_id, "choices": [],
                           "usage": usage_of(backend, prompt_ids, out)})
            self._sse_data("[DONE]")
            self._sse_end()
        except (BrokenPipeError, ConnectionResetError):
            closed = True
        except Exception as exc:
            try:
                self._sse({"error": {"message": str(exc), "type": "server_error"}})
                self._sse_data("[DONE]")
                self._sse_end()
            except Exception:
                pass
        finally:
            # A closed stream or an error stops the generation (the worker
            # checks done after each token).
            done.set()
        print('[np-gemma] stream prompt=%d out=%d finish=%s %.0f s%s' % (
            len(prompt_ids), len(out), "closed" if closed else finish,
            time.monotonic() - t_start, " (the client closed the stream: stopped)" if closed else ""),
            file=sys.stderr, flush=True)
        if shape is not None:
            log_answer(*shape)
        raw = self._raw(tok, out)
        parsed = backend.parse_output(raw)
        self._turn(backend, prompt_ids, out, raw, parsed, sent_content, parsed["tool_calls"],
                   "closed" if closed else finish)
        if empty is not None and not closed:
            log_empty(empty[0], backend.clean_raw(empty[1]))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(backend, host="127.0.0.1", port=8080, quiet=False, api_key=None,
                cors_origin=None, max_body=None):
    """Make the HTTP server. Use port 0 for a free port. api_key: the key
    that each request must give (None or "": no check); cors_origin: the
    origin of the CORS headers (None: no CORS headers)."""
    httpd = _Server((host, port), _Handler)
    httpd.backend = backend
    httpd.quiet = quiet
    httpd.api_key = api_key or None
    httpd.cors_origin = cors_origin or None
    httpd.max_body = MAX_BODY if max_body is None else max_body
    return httpd


def add_http_args(ap):
    """The options of the HTTP side: --api-key, --cors-origin, --max-body-mb."""
    ap.add_argument("--api-key", default=os.environ.get("NP_GEMMA_API_KEY", DEFAULT_API_KEY),
                    help="the key of each request (Authorization: Bearer KEY); default "
                         "NP_GEMMA_API_KEY, else %r; \"\" turns the check off" % DEFAULT_API_KEY)
    ap.add_argument("--cors-origin", default=None,
                    help="send CORS headers for this origin (\"*\" for any); default none, so a "
                         "web page cannot call the server from a browser")
    ap.add_argument("--max-body-mb", type=float, default=MAX_BODY / (1 << 20),
                    help="the largest request body in MB (default %(default)s)")


def serve(backend, host="127.0.0.1", port=8080, quiet=False, api_key=None, cors_origin=None,
          max_body=None):
    """Run the HTTP server until the process stops."""
    httpd = make_server(backend, host, port, quiet, api_key, cors_origin, max_body)
    where = "http://%s:%d/v1" % (host, httpd.server_address[1])
    print("np-gemma server %s  model=%s  key %s  CORS %s" % (
        where, backend.model_id, "required" if httpd.api_key else "OFF",
        httpd.cors_origin or "off"), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
