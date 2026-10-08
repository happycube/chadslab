#!/usr/bin/env python3
"""Run an OpenAI compatible server for Qwen3.8-Flash-Next (the GGUF file of
scripts/convert_nvfp4_gguf.py) or Qwen3.6-35B-A3B (a GGUF of llama.cpp, with
images through --mmproj) on the GPU, or on the CPU.

    python scripts/serve_qwen4.py [--ctx 98304] [--port 8081] [--hot-gb 0.5]
    python scripts/serve_qwen4.py -m models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
        --mmproj models/Qwen3.6-35B-A3B-GGUF/mmproj-BF16.gguf --media-dir DIR
    python scripts/serve_qwen4.py --mmproj models/Qwen3.8-Flash-Next-NVFP4 --media-dir DIR

The architecture of the GGUF (general.architecture) picks the model:
qwen4exp (Qwen4CPU and Qwen4GPU) or qwen35moe (QwenGGUFProgram and
QwenGPU).

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
- images and video (--mmproj; the ViT of Qwen3.6 and of Qwen3.8): the
  image and video parts of the messages (a data URI, or a path under
  --media-dir) go through the encoder of np_gemma/vision_qwen.py on the
  thread of the model. The <|image_pad|> of the template becomes one token
  for each row, the rows take the place of their embeddings, and the
  prompt gets the M-RoPE positions of Qwen (QwenCache.set_rope). A video
  gives 2 frames for each second (at most --video-frames), in pairs of
  at most --video-budget tokens; its <|video_pad|> becomes <T seconds>
  <|vision_start|> pads <|vision_end|> for each pair. The cache compares the
  keys of the tokens (media.keys), so another image of the same size is not
  a hit;
- one cache of --ctx tokens. A prompt that starts with all the tokens in the
  cache reads only its new tokens. The recurrent state (DeltaNet, the
  n-gram layer) cannot go back, so the server keeps snapshots of it (the
  last 4): at the end of each prompt, and after the last
  <|im_start|>assistant\n of each prompt. The template of the next turn
  writes the earlier answer in another form than the model wrote it
  (Qwen3.6 drops its think part; Qwen3.8 keeps only the reasoning_content
  that the client sends back), so the next prompt differs from the cache
  there. A prompt reads the tokens after the longest start that the cache
  or a snapshot holds; else it starts again at position 0.
  chat_template_kwargs.preserve_thinking goes to the template (Qwen3.8
  keeps the think parts of the earlier answers by default, Qwen3.6 only
  with it).
- --mtp N: N drafts of the MTP layer for each step (Qwen3.8 on the GPU,
  when the file has the layer; 1 by default, 0 turns it off). The prompt also runs the MTP layer, so its
  cache holds a row for each position of the cache of the model; a
  snapshot keeps the last stream of the model for the next row. A draft
  stays only when the sample of its row picks it (Sampler mtp_accept
  "exact"), so the text has the distribution of the settings; "in_set"
  keeps a draft that the settings allow. A round commits its tokens before
  it gives them to the client.
- one thread for all the work of the model (the load, the prompts, the
  steps): the CPU part of a step runs in the thread that runs the step, and
  each thread has a team of OpenMP threads of its own. With a team for each
  request as well as that of the load and that of GP_CPU_START, libgomp has
  more threads than CPUs, and its barriers sleep and wake (a step of 58 ms,
  not 46).

The think part: --thinking off|low|medium|xhigh is the default (medium). A
request sets it with "thinking" (true, false, or {"type": "enabled"}),
"reasoning_effort" (none, minimal, low, medium, high, xhigh), "reasoning":
{"effort": ...}, or "chat_template_kwargs": {"enable_thinking": ...,
"reasoning_effort": ...}. The template of the model has the levels low,
medium, and xhigh (minimal is low; high is xhigh). A request with a
max_tokens below --no-think-below (500) gets no think part; a think part
gets at most --think-budget tokens, and at most half of max_tokens.

--debug DIR writes each turn to DIR/NNNNN-<time>.json: the request (the
messages, the tools, the settings), the prompt as the template made it, the
cache hits, the raw answer, its reasoning, content, and tool calls, and the
finish. DIR/turns.log has one line for each turn.

The usage of a response has prompt_tokens_details.cached_tokens: the tokens
of the prompt that the cache held. The log gives the hits of each request
and of all the requests.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.qwen_tok import ESCAPE, QwenTokenizer  # noqa: E402
from np_gemma import media as MD  # noqa: E402
from np_gemma import server as S  # noqa: E402
from np_gemma.chat import parse_output as gemma_parse_output  # noqa: E402
from np_gemma.server import Backend, serve  # noqa: E402
from np_gemma.think_guard import ThinkGuard, think_budget, think_left_open  # noqa: E402,F401

MODEL = "models2/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-bf16.gguf"


# The words that close a think part at --think-budget (then </think>).
THINK_CLOSE = "\n\nI have thought about this long enough; now I write the answer.\n</think>\n\n"
# The words when fewer than --wrap-left tokens of the context are left: in a
# think part (then </think>), and in the answer.
WRAP_THINK = "\n\nThe context is almost full, so I stop thinking and answer briefly now.\n</think>\n\n"
WRAP_ANSWER = "\n\n(The context is almost full, so I wrap up briefly now.)\n\n"


class QwenTok:
    """The tokenizer interface of np_gemma/server.py on QwenTokenizer, with
    the chat template of the model."""

    def __init__(self, path, template, text=None):
        import jinja2
        self.t = QwenTokenizer(path)
        self.stop_ids = list(self.t.stop_ids)
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                 extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
        env.filters["items"] = lambda d: list(d.items()) if isinstance(d, dict) else []

        def raise_exception(msg):
            raise ValueError(msg)
        env.globals["raise_exception"] = raise_exception
        if text is None:
            text = open(template, encoding="utf-8").read()
        self.template = env.from_string(text)

    def encode(self, text):
        return self.t.encode(text)

    def decode(self, ids, skip_special_tokens=False):
        """The text of the answer for the parsers: the special tokens <think>
        and </think> as the channel of the Gemma output (np_gemma/chat.py,
        parse_output), the others as their text. A marker that the model
        wrote as plain text (a quote of code: "<think>", "</tool_call>",
        "<|im_end|>") has ESCAPE in place of its "<", so no parser takes it;
        qwen_parse_output gives the "<" back."""
        t = self.t
        parts = []
        run = []

        def flush():
            if run:
                parts.append(PLAIN_MARK.sub(ESCAPE, t.decode(run)))
                run.clear()
        for i in ids:
            tok = t.id_to_token.get(int(i), "")
            if tok in t.special:
                flush()
                parts.append({"<think>": "<|channel>thought\n", "</think>": "<channel|>"}.get(tok, tok))
            else:
                run.append(i)
        flush()
        return "".join(parts)

    def apply_chat_template(self, messages, add_generation_prompt=True, thinking=False,
                            tools=None, empty_thought_block=True, effort=None, preserve=None):
        kw = {"reasoning_effort": effort} if thinking and effort else {}
        if preserve is not None:
            kw["preserve_thinking"] = bool(preserve)
        text = self.template.render(messages=messages, tools=tools or None,
                                    add_generation_prompt=add_generation_prompt,
                                    enable_thinking=bool(thinking), **kw)
        return text


# The levels of the request to those of the template (None: no think part).
EFFORT = {"none": None, "off": None, "disabled": None, "false": None,
          "minimal": "low", "low": "low", "medium": "medium",
          "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "on": "xhigh", "true": "xhigh",
          "enabled": "xhigh"}


def thinking_of(req, default):
    """The think level of a request (None: off), else default."""
    kw = req.get("chat_template_kwargs") or {}
    level = default
    for v in (kw.get("reasoning_effort"), (req.get("reasoning") or {}).get("effort")
              if isinstance(req.get("reasoning"), dict) else None, req.get("reasoning_effort")):
        if v is not None:
            level = EFFORT.get(str(v).lower(), level)
    for v in (kw.get("enable_thinking"), req.get("thinking")):
        if isinstance(v, dict):
            v = v.get("type")
        if v is None:
            continue
        on = EFFORT.get(str(v).lower(), "xhigh") is not None
        if not on:
            level = None
        elif level is None:
            level = "xhigh"
    return level


# The tool calls of the answer (the form of the template).
TOOLS = {}      # the tools of the request: name -> the properties of its parameters
# The CUDA errors that leave the context unusable: after one the process
# exits with EXIT_FATAL, so a supervisor (scripts/serve_forever.sh) can start
# it again. Out of memory is not one of them.
FATAL = re.compile(r"illegal (memory access|instruction|address)|unspecified launch failure|"
                   r"misaligned address|device-side assert|invalid program counter|"
                   r"hardware stack error|uncorrectable ECC|context is destroyed|"
                   r"cudaErrorLaunchFailure|CUDA_ERROR_LAUNCH_FAILED", re.I)
EXIT_FATAL = 70
# The "<" of a marker in plain text of the answer (QwenTok.decode): the tags
# of Qwen and the markers of the Gemma parser (SPECIAL of np_gemma/chat.py).
MARK_TAIL = r"\|[^>\n]*?\|>|\|[a-z_]+>|[a-z_]+\|>|/?think>|/?tool_call>|/?tool_response>"
PLAIN_MARK = re.compile("<(?=" + MARK_TAIL + ")")
# ESCAPE stands for "<" only in front of a marker (a U+E000 of the answer
# itself stays): _plain gives the "<" back.
UNMARK = re.compile(ESCAPE + "(?=" + MARK_TAIL + ")")
CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
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


def _plain(x):
    """x with the "<" of the markers that QwenTok.decode escaped."""
    if isinstance(x, str):
        return UNMARK.sub("<", x) if ESCAPE in x else x
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {_plain(k): _plain(v) for k, v in x.items()}
    return x


def _call(name, args):
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": _plain(name), "arguments": json.dumps(_plain(args),
                                                                       ensure_ascii=False)}}


def qwen_parse_output(text):
    """parse_output of np_gemma/chat.py (the reasoning), and the tool calls of
    Qwen: the XML form of the template (<function=...><parameter=...>, also
    with no </function>), and the JSON form ({"name": ..., "arguments": ...}).
    A tool call that is not complete is pending: a stream holds it."""
    p = gemma_parse_output(text)
    calls = []

    def call(m):
        inner = m.group(1)
        f = FUNC.search(inner)
        if f:
            name, props = f.group(1), TOOLS.get(f.group(1), {})
            args = {k: _value(v, props.get(k)) for k, v in PARAM.findall(f.group(2))}
            calls.append(_call(name, args))
            return ""
        try:
            obj = json.loads(inner.strip())
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(_call(obj["name"], args))
            return ""
        except (ValueError, KeyError, TypeError, AttributeError):
            return m.group(0)                       # not a call: the text stays
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
    return {"reasoning": _plain(p["reasoning"]), "content": _plain(body.strip()),
            "tool_calls": _plain(p["tool_calls"]) + calls, "pending": _plain(pending)}




def _escaped(x, esc):
    """A copy of the messages (or the tools) x with esc on each string; the
    media parts keep their data."""
    if isinstance(x, str):
        return esc(x)
    if isinstance(x, list):
        return [_escaped(v, esc) for v in x]
    if isinstance(x, dict):
        if x.get("type") in S.MEDIA_TYPES:
            return x
        return {k: _escaped(v, esc) for k, v in x.items()}
    return x


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
    """Backend with Qwen4GPU (or Qwen4CPU), or QwenGPU (or QwenGGUFProgram),
    and one cache of ctx tokens."""

    def __init__(self, model, dev, tok, ctx, cache_cls, **kw):
        super().__init__(model, tok, model_id=kw.pop("model_id"), **kw)
        self.dev = dev
        self.ctx = ctx
        self.cache_cls = cache_cls
        self.cache = None
        self.ids = []           # the keys of the tokens in the cache (media.keys)
        # Snapshots of the recurrent state: (keys, state, n-gram ids, logits
        # or None), the oldest first. The state of the DeltaNet cannot go
        # back, so a prompt can reuse the cache only from the end of the
        # cache or from a snapshot.
        self.snaps = []
        self.max_snaps = 4
        # The head of the answer of a chat prompt (<|im_start|>assistant\n):
        # each prompt has a snapshot after its last one.
        self.answer_head = None
        self.last_cached = 0    # the prompt tokens of the last request that the cache held
        self.hits = [0, 0, 0]   # requests, prompt tokens, cached tokens
        self.think = None       # the default think level (None: off)
        self.max_floor = 0      # --max-tokens-floor
        self.floor_min = 8192   # --max-tokens-floor-min
        self.debug = None       # --debug: the directory of the records of the turns
        self.turn_n = 0
        self.last_think = None
        self.last_text = ""
        self.mtp = 0            # --mtp: the drafts of a round (0: no MTP)
        self.fatal = None       # an error that leaves the GPU unusable (note_error)
        self.h_end = None       # the stream of the model at the last position of the cache
        self.mtp_stats = [0, 0]  # drafts, accepted

    def chat_prompt_ids(self, req):
        """The prompt of a chat request, with its think level, and
        chat_template_kwargs.preserve_thinking (the reasoning of the earlier
        answers in the prompt; the default of the template when absent)."""
        level = thinking_of(req, self.think)
        lim = req.get("max_tokens", req.get("max_completion_tokens"))
        try:
            lim = int(lim) if lim is not None else None
        except (TypeError, ValueError):
            lim = None          # np_gemma/server.py answers the error
        if level is not None and lim is not None and lim < getattr(self, "no_think_below", 0):
            # --no-think-below: a small request (a client's title, 64
            # tokens) answers with no think part
            print("[qwen4] max_tokens %d < %d: no think part" % (lim, self.no_think_below),
                  file=sys.stderr, flush=True)
            level = None
        self.last_think = level
        preserve = (req.get("chat_template_kwargs") or {}).get("preserve_thinking")
        return self.prompt_ids(req["messages"], level is not None, req.get("tools"), effort=level,
                               preserve=preserve)

    def on_turn(self, rec):
        """--debug: the record of a turn (np_gemma/server.py) to a file."""
        if not self.debug:
            return
        self.turn_n += 1
        rec = dict(rec, think=self.last_think, prompt_text=self.tokenizer.t.unescape(self.last_text),
                   raw=self.clean_raw(rec.get("raw", "")),
                   cached_tokens=self.last_cached,
                   sampling={"temperature": self.temperature, "top_k": self.top_k, "top_p": self.top_p})
        name = os.path.join(self.debug, "%05d-%s.json" % (self.turn_n, time.strftime("%H%M%S")))
        with open(name, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        req = rec["request"] or {}
        with open(os.path.join(self.debug, "turns.log"), "a", encoding="utf-8") as f:
            f.write("%s %s messages=%d tools=%d prompt=%d cached=%d out=%d finish=%s think=%s "
                    "reasoning=%d content=%d calls=%s %.0fs\n" % (
                        rec["time"], os.path.basename(name), len(req.get("messages") or []),
                        len(req.get("tools") or []), rec["prompt_tokens"], rec["cached_tokens"],
                        rec["out_tokens"], rec["finish"], rec["think"], len(rec["reasoning"]),
                        len(rec["content"] or ""),
                        ",".join(c["function"]["name"] for c in rec["tool_calls"]) or "-", rec["seconds"]))

    def prompt_ids(self, messages, thinking=None, tools=None, effort=None, preserve=None):
        TOOLS.clear()
        for tl in tools or []:
            f = tl.get("function", tl)
            TOOLS[f.get("name")] = (f.get("parameters") or {}).get("properties") or {}
        think = self.thinking if thinking is None else bool(thinking)
        msgs = [_message(m) for m in messages]
        parts = [p for m in msgs if isinstance(m.get("content"), list) for p in m["content"]
                 if isinstance(p, dict) and p.get("type") in S.MEDIA_TYPES]
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
        items, videos = self._media_items(parts)
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
        out = S.PromptIds(out_ids)
        out.spans = spans
        self.last_media = [(sp.kind, sp.rows.shape[0], sp.key) for sp in spans]
        return out

    def _media_items(self, parts):
        """The rows of the media parts: the items of expand_qwen (one for an
        image, one for each pair of frames of a video), and the text of each
        video. The encoder runs on the thread of the model."""
        from np_gemma.vision_qwen import video_text
        if self.embedder is None:
            raise ValueError("this server takes no image or video input (start it with --mmproj)")
        items, videos = [], []
        for part in parts:
            kind = part.get("type")
            t0 = time.time()
            if kind in ("video", "video_url"):
                data = self._video_source(part)
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
                data, budget = self._image_source(part)
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
        q = queue.Queue()

        def job():
            try:
                if self.dev is not None and getattr(self.embedder, "gpu", False):
                    if getattr(self.embedder, "lent", False):
                        self.dev.release_warm()     # the encoder takes its room back
                    else:
                        self.dev.free_ring()
                q.put((True, fn()))
            except BaseException as exc:
                self.note_error(exc)
                q.put((False, exc))
        self.jobs.put(job)
        ok, value = q.get()
        if not ok:
            raise value
        return value

    # The hooks of np_gemma/server.py for the output of Qwen: the parse,
    # no tool-result marker (that is the Gemma <|tool_response>), and the raw
    # text for the log and its files without ESCAPE (QwenTok.decode).
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

    def completion_ids(self, prompt):
        return self.tokenizer.encode(prompt)

    def open_channel(self, prompt_ids):
        """The opener of the think part for the parser of the answer of this
        request (np_gemma/server.py keeps it with the request): "<|channel>
        thought\n" when its prompt leaves <think> open. It was a field of the
        tokenizer, set by each prompt: a client's title request (max_tokens
        64: no think part) made just after its main request set it off, and
        the main answer (its reasoning, 20669 chars) came as content."""
        return S.THOUGHT_OPEN if think_left_open(prompt_ids, self.tokenizer.t.special) else ""

    def _fresh(self):
        self.cache = self.cache_cls(self.model.cfg, self.ctx)
        if self.dev is not None:
            self.dev.attach(self.cache)
        self.ids = []
        self.snaps = []
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
        keys = list(keys)
        self.snaps = [sn for sn in self.snaps if sn[0] != keys]
        self.snaps.append((keys, state, None if ple is None else ple.copy(),
                           None if logits is None else logits.copy(),
                           None if stream is None else stream.copy()))
        del self.snaps[:-self.max_snaps]

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

    def _decode_mtp(self, logits, n, max_tokens, sampler, eos_ids):
        """The tokens of the answer with MTP drafts (see the module text):
        the loop of np_gemma.speculative, with the sampler of the plain
        decode (mtp_accept "in_set" keeps drafts that its settings allow)."""
        from np_gemma.speculative import QwenMTPDrafter, QwenTarget, stream
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

    def applied_max_tokens(self, n):
        """--max-tokens-floor: a request of at least --max-tokens-floor-min
        tokens (the work of an agent: the reasoning and a file in a tool
        call need more than the 8192 of a client) gets the floor. A smaller
        request (a title, a summary) keeps its limit. The response gives
        the limit in X-Max-Tokens-Applied."""
        if self.max_floor and self.floor_min <= n < self.max_floor:
            return self.max_floor
        return n

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        """The tokens of _generate, from the model thread. A caller that
        stops early stops the generation. (np_gemma/server.py applies
        applied_max_tokens before.)"""
        q, stop = queue.Queue(), threading.Event()

        def job():
            try:
                for token in self._generate(prompt_ids, max_tokens, sampler, eos_ids):
                    q.put(("token", token))
                    if stop.is_set():
                        break
                q.put(("end", None))
            except BaseException as exc:
                self.note_error(exc)
                q.put(("error", exc))
        self.jobs.put(job)
        try:
            while True:
                kind, value = q.get()
                if kind == "end":
                    return
                if kind == "error":
                    raise value
                yield value
        finally:
            stop.set()

    def crash_report(self, exc, where_dir=None):
        """A fatal error: write a report to a new directory under
        --crash-dir (report.json: the error, its traceback, the record of
        the GPU program queued last (np_gemma.gpu.where), the request, the
        cache, the memory of the GPU as GpuMem knows it; prompt.npy: the
        tokens, for scripts/replay_crash.py). No CUDA call: the context is
        lost. Return the directory."""
        base = where_dir or getattr(self, "crash_dir", None)
        if not base:
            return None
        d = os.path.join(base, time.strftime("crash-%Y%m%d-%H%M%S"))
        os.makedirs(d, exist_ok=True)
        rep = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "error": str(exc),
               "traceback": "".join(traceback.format_exception(exc)),
               "context": self.ctx, "request": getattr(self, "last_request", None)}
        try:
            from np_gemma import gpu as G
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
            from np_gemma.gpumm import mem
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

    def _generate(self, prompt_ids, max_tokens, sampler, eos_ids):
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
            c = len(self.ids)
            if 0 < c < n and keys[:c] == self.ids:
                k = c
            for sn in self.snaps:
                s = len(sn[0])
                if s > k and (s < n or (s == n and sn[3] is not None)) and keys[:s] == sn[0]:
                    k, snap = s, sn
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
        force = getattr(self, "think_force", ())
        # --wrap-left: the token of the answer from which fewer than that
        # many positions of the context are left (when max_tokens gets there)
        left, wrap_at = getattr(self, "wrap_left", 0), None
        room = self.ctx - n - 1
        if left > 0 and max_tokens > room - left:
            wrap_at = max(1, room - left + 1)
        guard = ThinkGuard.of(prompt_ids, self.tokenizer.t.special, eos_ids,
                              think_budget(getattr(self, "think_budget", 0), max_tokens, len(force)),
                              force, wrap_at, getattr(self, "wrap_think", ()),
                              getattr(self, "wrap_answer", ()))
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


def dense_label(model):
    """The form of the dense matrices for the banner: bf12 for a file of
    convert_q8_gguf.py --dense bf12 (Qwen4CPU.K keeps them unless
    NP_GEMMA_DENSE is q8, rq8 or bf16), else the dense mode of the model."""
    g = getattr(model, "g", None)
    mode = os.environ.get("NP_GEMMA_DENSE", "bf12")
    if g is not None and mode not in ("q8", "bf16") and any(
            t == 57 for _d, t, _o in g.tensors.values()):
        return "rq8" if mode == "rq8" else "bf12"
    return getattr(model, "dense", "-")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-m", "--model", default=MODEL)
    ap.add_argument("--tok", default=None, help="tokenizer.json (default: next to the model)")
    ap.add_argument("--template", default=None,
                    help="chat_template.jinja (default: next to the model, else in the checkpoint)")
    ap.add_argument("--ctx", type=int, default=98304, help="the tokens of the context")
    ap.add_argument("--experts", default=None,
                    help="a file of the routed experts in another form that takes the place of "
                         "those of the model (scripts/convert_q8_gguf.py --experts-only)")
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--hot-gb", type=float, default=None,
                    help="gpu: GB of hot experts (default: the free memory after the dense part "
                         "and the cache of --ctx tokens)")
    ap.add_argument("--mtp", type=int, default=None,
                    help="gpu: the drafts of the MTP layer for each step (0: off; default 3 "
                         "when the file has the layer: Qwen3.8)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--model-id", default="qwen3.8-flash-next")
    ap.add_argument("--max-tokens", type=int, default=4096, help="the default of a request")
    ap.add_argument("--max-tokens-floor", type=int, default=0,
                    help="the least max_tokens of a request of at least --max-tokens-floor-min "
                         "(the reasoning and a file in a tool call need more than 8192); the "
                         "context bounds it; the response header X-Max-Tokens-Applied gives it")
    ap.add_argument("--max-tokens-floor-min", type=int, default=8192,
                    help="a request below this keeps its max_tokens (a title, a summary)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--think-budget", type=int, default=6000,
                    help="the most tokens of a think part: then the server closes it (THINK_CLOSE) "
                         "and the model writes the answer (0: no limit); at most half of the "
                         "max_tokens of the request too (think_budget)")
    ap.add_argument("--crash-dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "np-gemma-crash"),
                    help="a fatal error (a lost CUDA context) writes a report here: report.json "
                         "and prompt.npy ('': none)")
    ap.add_argument("--wrap-left", type=int, default=500,
                    help="when fewer tokens of the context are left, the answer gets words that "
                         "tell the model to wrap up (WRAP_THINK, WRAP_ANSWER; 0: none)")
    ap.add_argument("--no-think-below", type=int, default=500,
                    help="a request with a smaller max_tokens gets no think part (a title, a "
                         "summary; 0: none)")
    ap.add_argument("--thinking", choices=("off", "low", "medium", "xhigh"), default="medium",
                    help="the think part by default (a request can set it)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--debug", default=None, metavar="DIR",
                    help="write each turn (request, prompt, raw answer, parts) to DIR")
    ap.add_argument("--mmproj", default=None,
                    help="image input: the mmproj GGUF of the encoder (Qwen3.6), or the "
                         "checkpoint directory with model.visual.* (Qwen3.8: "
                         "models/Qwen3.8-Flash-Next-NVFP4)")
    ap.add_argument("--image-budget", type=int, default=1024,
                    help="the most tokens of an image (a request can ask for detail low: 70, "
                         "high: 1120, or max_soft_tokens)")
    ap.add_argument("--mmproj-gpu", choices=("auto", "keep", "reserve", "lend", "release", "off"),
                    default="auto",
                    help="the encoder on the GPU: keep its weights there (0.9 GB, loaded before "
                         "the buffers of the model), reserve the memory of the largest image "
                         "before the model sizes its hot experts (the weights and the tensors of "
                         "an image of max(--image-budget, 1120) tokens, about 1.6 GB for "
                         "Qwen3.8; a request then gets at most that many tokens), lend (that room, "
                         "but between media requests the model keeps more of its experts there: "
                         "the encoder loads for each image, about 0.3 s more), release them "
                         "after each media request, or off (the CPU: 2.2 s for an image, 6.7 s "
                         "for 9 s of video). auto: keep for Qwen3.6; off for Qwen3.8, whose "
                         "programs leave no room for it on a GPU of 16 GB with a desktop")
    ap.add_argument("--video-budget", type=int, default=128,
                    help="the most tokens of each pair of frames of a video")
    ap.add_argument("--video-frames", type=int, default=32,
                    help="the most frames of a video (2 for each second up to it)")
    ap.add_argument("--media-dir", default=None,
                    help="the directory of local media files a request may name (default: "
                         "data URIs only)")
    S.add_http_args(ap)
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = args.model if os.path.isabs(args.model) else os.path.join(root, args.model)

    from np_gemma.gguf import GGUF
    meta = GGUF(path, stage=False).meta      # the metadata only: no copy of a file on Optane
    arch = meta.get("general.architecture")
    template = args.template or os.path.join(os.path.dirname(path), "chat_template.jinja")
    text = None
    if not os.path.exists(template):
        if arch == "qwen35moe":
            text = meta["tokenizer.chat_template"]          # the template of the GGUF
        else:
            template = os.path.join(root, "models/Qwen3.8-Flash-Next-NVFP4/chat_template.jinja")
    tok = QwenTok(args.tok or os.path.join(os.path.dirname(path), "tokenizer.json"), template, text)
    if args.mmproj and arch not in ("qwen35moe", "qwen4exp"):
        ap.error("--mmproj: %s has no image input" % arch)
    jobs, ready = queue.Queue(), queue.Queue()

    def model_thread():
        """The load, the warm-up, and then the jobs of generate."""
        try:
            print("loading %s ..." % path, flush=True)
            t0 = time.time()
            dev = None
            mode = args.mmproj_gpu
            if mode == "auto":
                mode = "keep" if arch == "qwen35moe" else "off"
            embedder = None
            placeholder = None
            if args.mmproj and mode == "lend" and args.backend == "gpu":
                # the room of the encoder: measured, then held by a placeholder
                # while the model sizes its hot experts, then lent to the model
                from np_gemma.gpumm import Buffer
                from np_gemma.vision_qwen import QwenEmbedder
                t1 = time.time()
                embedder = QwenEmbedder(args.mmproj, gpu=True)
                lent = embedder.lend_gpu(max(args.image_budget, 1120))
                placeholder = Buffer(lent, "lend", "encoder")
                print("image encoder: %.2f GB of the GPU for images of at most %d tokens, lent to "
                      "the experts between them (%.0f s)" % (lent / 1e9, embedder.budget_max,
                                                            time.time() - t1), flush=True)
            if args.mmproj and mode == "reserve" and args.backend == "gpu":
                # the room of the encoder first: the hot experts are sized
                # with the memory that is free after it
                from np_gemma.vision_qwen import QwenEmbedder
                t1 = time.time()
                embedder = QwenEmbedder(args.mmproj, gpu=True)
                held = embedder.reserve_gpu(max(args.image_budget, 1120))
                print("image encoder: %.2f GB of the GPU reserved (images of at most %d tokens) "
                      "in %.0f s" % (held / 1e9, embedder.budget_max, time.time() - t1), flush=True)
            if arch == "qwen35moe":
                from np_gemma.qwen import QwenCache, QwenGGUFProgram
                model, cache_cls = QwenGGUFProgram(path), QwenCache
                if args.backend == "gpu":
                    from np_gemma.qwen_gpu import QwenGPU
                    dev = QwenGPU(model, hot_gb=args.hot_gb)
            else:
                from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU
                model, cache_cls = Qwen4CPU(path, experts=args.experts), Qwen4Cache
                if args.backend == "gpu":
                    from np_gemma.qwen4_gpu import Qwen4GPU
                    dev = Qwen4GPU(model, hot_gb=args.hot_gb, ctx=args.ctx)
                    if placeholder is not None:
                        placeholder.free()
                        dev.lend_warm(lent)
            backend = QwenBackend(model, dev, tok, args.ctx, cache_cls, model_id=args.model_id,
                                  thinking=args.thinking != "off", max_tokens=args.max_tokens,
                                  temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            backend.jobs = jobs
            backend.think = None if args.thinking == "off" else args.thinking
            backend.think_budget = max(0, args.think_budget)
            backend.no_think_below = max(0, args.no_think_below)
            backend.wrap_left = max(0, args.wrap_left)
            backend.crash_dir = args.crash_dir
            backend.wrap_think = tok.encode(WRAP_THINK)
            backend.wrap_answer = tok.encode(WRAP_ANSWER)
            backend.think_force = tok.encode(THINK_CLOSE)
            backend.max_floor = args.max_tokens_floor
            backend.floor_min = args.max_tokens_floor_min
            # MTP is on by default (3 drafts) with the layer: on the 2-socket
            # Xeon and the 3090 with the RQ8_0 file, 1 draft was best (greedy
            # 27.7 -> 30.4 tok/s; 2 or 3 drafts 29.6 and 29.0); with the
            # MIX-BF12 file, the BF12 verify groups on the tensor cores
            # (k_kq_bf12_tc) and int16 experts, team 24 on a hot day: plain
            # 16.2, 1 draft 19.7, 2 drafts 21.4, 3 drafts 22.1 tok/s
            mtp = 3 if args.mtp is None else args.mtp
            backend.mtp = mtp if getattr(dev, "has_mtp", False) else 0
            if args.mtp and not backend.mtp:
                print("--mtp: the model has no MTP layer on the GPU; plain steps",
                      file=sys.stderr, flush=True)
            if args.debug:
                os.makedirs(args.debug, exist_ok=True)
                backend.debug = args.debug
                S.RAW_DIR = args.debug   # the raw answers (last-answer.txt, empty-*.txt)
            if args.mmproj:
                from np_gemma.vision_qwen import QwenEmbedder
                t1 = time.time()
                if embedder is not None:
                    backend.embedder = embedder
                else:
                    backend.embedder = QwenEmbedder(args.mmproj,
                                                    gpu=dev is not None and mode not in ("off", "reserve",
                                                                                         "lend"))
                    backend.embedder.keep_gpu = mode == "keep"
                    if mode == "keep":
                        backend.embedder.warm()
                backend.image_budget = args.image_budget
                backend.video_budget = args.video_budget
                backend.video_frames = args.video_frames
                backend.media_dir = args.media_dir
                print("image encoder %s (%s) in %.0f s" % (
                    args.mmproj, ("GPU, " + mode) if backend.embedder.gpu else "CPU",
                    time.time() - t1), flush=True)
            # the cache and the programs of a prompt now, so the first request
            # does not wait for them and a context that does not fit fails here
            backend._fresh()
            if dev is not None:
                backend._prompt(tok.encode("<|im_start|>user\nHello<|im_end|>\n") * 160, 0)
                backend._fresh()
            # from here on, an OpenMP team outside the planned ones is reported
            # (np_gemma.cops.team_warn: NP_GEMMA_TEAM_WARN, default 5 threads)
            from np_gemma import cops as _cops
            _cops.team_warn()
            print("ready in %.0f s: dense %s, context %d%s; thinking %s; sampling temperature=%s "
                  "top_k=%s top_p=%s" % (time.time() - t0, dense_label(model), args.ctx,
                                         ", %d hot experts in each layer" % dev.n_slots if dev else "",
                                         args.thinking, args.temperature, args.top_k, args.top_p),
                  flush=True)
            ready.put(backend)
        except BaseException as exc:
            ready.put(exc)
            return
        # Each job reports its own errors to its caller; this is the last
        # guard, so the thread that serves every request does not end.
        while True:
            job = jobs.get()
            try:
                job()
            except BaseException as exc:
                print("[qwen4] a job failed:", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                backend.note_error(exc)
            if backend.fatal is not None:
                try:
                    d = backend.crash_report(backend.fatal)
                    if d:
                        print("[qwen4] the crash report: %s (scripts/replay_crash.py %s)" % (d, d),
                              file=sys.stderr, flush=True)
                except Exception as exc:
                    print("[qwen4] the crash report failed: %r" % exc, file=sys.stderr, flush=True)
                print("[qwen4] fatal error, the process ends (exit %d): %s" % (
                    EXIT_FATAL, backend.fatal), file=sys.stderr, flush=True)
                time.sleep(1.0)        # the caller sends the error to its client
                os._exit(EXIT_FATAL)

    threading.Thread(target=model_thread, daemon=True).start()
    backend = ready.get()
    if isinstance(backend, BaseException):
        raise backend
    serve(backend, host=args.host, port=args.port, quiet=args.quiet, api_key=args.api_key,
          cors_origin=args.cors_origin, max_body=int(args.max_body_mb * (1 << 20)))
    return 0


if __name__ == "__main__":
    np.seterr(over="ignore")
    raise SystemExit(main())
