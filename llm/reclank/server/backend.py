"""The backend of the server: what the HTTP handler (server/http.py) asks of
a model, on top of an engine (reclank/models/base.py) and the model thread
(server/worker.py).

The backend holds the settings of the requests: the sampling defaults, the
think default and --no-think-below, max_tokens and its floor, the debug
records of the turns, and the media sources. The engine renders the chat,
parses the answer, and runs the decode of its family.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

from reclank.core.sampling import Sampler
from server.media import MediaSource


class Backend:
    """The model of the server, for the HTTP handler."""

    def __init__(self, engine, worker, model_id="reclank", thinking=None, max_tokens=1024,
                 temperature=1.0, top_k=None, top_p=None, mtp_accept="exact", mtp_floor=None,
                 media_dir=None):
        self.engine = engine
        self.worker = worker
        self.model_id = model_id
        # the think setting of a request that sets none (engine.think_level)
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # The rule of the MTP drafts with sampling (Sampler mtp_accept and
        # mtp_floor). A request can give "mtp_accept" and "mtp_floor".
        self.mtp_accept = mtp_accept
        self.mtp_floor = mtp_floor
        # --no-think-below: a request with a smaller max_tokens gets no think
        # part (a client's title or a short check; 0: no rule)
        self.no_think_below = 0
        # --max-tokens-floor: a request of at least floor_min tokens gets at
        # least max_floor (the work of an agent); 0: none
        self.max_floor = 0
        self.floor_min = 8192
        self.lock = threading.Lock()
        self.source = MediaSource(media_dir)
        # --debug: the directory of the records of the turns (on_turn)
        self.debug = None
        self.turn_n = 0
        self.last_think = None

    # ---- what the handler reads of the model

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def max_context(self):
        return self.engine.max_context

    @property
    def last_cached(self):
        return self.engine.last_cached

    def model_info(self, name=None):
        """Return one model description."""
        info = {"id": name or self.model_id, "object": "model", "created": 0,
                "owned_by": "reclank"}
        if self.max_context:
            info["context_length"] = self.max_context
        return info

    # ---- the prompt

    def chat_prompt_ids(self, req):
        """The prompt of a chat request: its think setting (the request, else
        the default; --no-think-below), the tools, and the media."""
        eng = self.engine
        level = eng.think_level(req, self.thinking)
        lim = req.get("max_tokens", req.get("max_completion_tokens"))
        try:
            lim = int(lim) if lim is not None else None
        except (TypeError, ValueError):
            lim = None          # (the error comes later, from the handler)
        if level and lim is not None and self.no_think_below and lim < self.no_think_below:
            # a small request (a client's title, 64 tokens) answers with no
            # think part; the thought took all its tokens
            print("[reclank] max_tokens %d < %d: no think part" % (lim, self.no_think_below),
                  file=sys.stderr, flush=True)
            level = eng.think_off()
        self.last_think = level
        preserve = (req.get("chat_template_kwargs") or {}).get("preserve_thinking")
        return eng.prompt_ids(req["messages"], level, req.get("tools"), preserve=preserve,
                              source=self.source)

    def completion_ids(self, prompt):
        return self.engine.completion_ids(prompt)

    def open_channel(self, prompt_ids):
        return self.engine.open_channel(prompt_ids)

    # ---- the generation

    def applied_max_tokens(self, n):
        """--max-tokens-floor: a request of at least --max-tokens-floor-min
        tokens (the reasoning and a file in a tool call need more than the
        8192 of a client) gets the floor. A smaller request keeps its limit.
        The response gives the limit in X-Max-Tokens-Applied."""
        if self.max_floor and self.floor_min <= n < self.max_floor:
            return self.max_floor
        return n

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
        return self.engine.stop_ids()

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        """Yield the tokens of the answer; the engine runs on the model
        thread."""
        return self.worker.stream(
            lambda: self.engine.generate(prompt_ids, max_tokens, sampler, eos_ids))

    # ---- the answer

    def parse_output(self, text):
        return self.engine.parse_output(text)

    def tool_done(self, text):
        return self.engine.tool_done(text)

    def clean_raw(self, raw):
        return self.engine.clean_raw(raw)

    def on_turn(self, rec):
        """--debug: write the record of a turn to a file, and a line to
        turns.log."""
        if not self.debug:
            return
        self.turn_n += 1
        rec = dict(rec, think=self.last_think,
                   sampling={"temperature": self.temperature, "top_k": self.top_k, "top_p": self.top_p})
        rec.update(self.engine.turn_fields(rec))
        name = os.path.join(self.debug, "%05d-%s.json" % (self.turn_n, time.strftime("%H%M%S")))
        with open(name, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        req = rec["request"] or {}
        cached = " cached=%d" % rec["cached_tokens"] if rec.get("cached_tokens") is not None else ""
        with open(os.path.join(self.debug, "turns.log"), "a", encoding="utf-8") as f:
            f.write("%s %s messages=%d tools=%d prompt=%d%s out=%d finish=%s think=%s "
                    "reasoning=%d content=%d calls=%s %.0fs\n" % (
                        rec["time"], os.path.basename(name), len(req.get("messages") or []),
                        len(req.get("tools") or []), rec["prompt_tokens"], cached, rec["out_tokens"],
                        rec["finish"], rec["think"], len(rec["reasoning"]),
                        len(rec["content"] or ""),
                        ",".join(c["function"]["name"] for c in rec["tool_calls"]) or "-",
                        rec["seconds"]))
