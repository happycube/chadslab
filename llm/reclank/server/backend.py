"""The backend of the server: what the HTTP handler (server/http.py) asks of
a model, on top of an engine (reclank/models/base.py) and the model thread
(server/worker.py).

The backend holds the settings of the requests: the sampling defaults, the
think default and --no-think-below, max_tokens and its floor, the debug
records of the turns, and the media sources. The engine renders the chat,
parses the answer, and runs the decode of its family.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time

from reclank.core.messages import PromptIds, content_text
from reclank.core.sampling import Sampler
from server.media import MediaSource

# The id of a client's session or conversation, for the lanes (--lanes 2): the headers and the
# fields of the body that the clients send, in this order.
#   x-session-affinity, session_id: pi-ai (dsh) with compat.sendSessionAffinityHeaders (its
#     x-client-request-id is the same id, but other clients send a new one with each request)
#   x-session-id: pi-ai's OpenRouter form, OpenRouter, vLLM's router (--session-key)
#   x-claude-code-session-id: Claude Code
#   conversation_id: Codex (older releases)
#   prompt_cache_key: the OpenAI API (Codex, OpenCode, pi-ai with long cache retention)
#   metadata.session_id; metadata.user_id with "session_<id>" (the Anthropic form of Claude Code)
# "user" and a user_id without a session are a user's, not a conversation's: not used.
SESSION_HEADERS = ("x-session-affinity", "session_id", "session-id", "x-session-id",
                   "x-claude-code-session-id", "conversation_id", "x-conversation-id")
SESSION_FIELDS = ("prompt_cache_key", "session_id", "conversation_id")
# A lane that the client names: llama.cpp's id_slot, or "lane" (body), or the header x-reclank-lane.
LANE_FIELDS = ("lane", "id_slot")


def session_of(req, headers):
    """The session id of a request (SESSION_HEADERS, SESSION_FIELDS, metadata), or None."""
    for h in SESSION_HEADERS:
        v = headers.get(h) if headers is not None else None
        if v:
            return "h:" + str(v).strip()[:128]
    for f in SESSION_FIELDS:
        v = req.get(f)
        if isinstance(v, str) and v:
            return "b:" + v[:128]
    meta = req.get("metadata")
    if isinstance(meta, dict):
        v = meta.get("session_id")
        if isinstance(v, str) and v:
            return "b:" + v[:128]
        u = meta.get("user_id")
        if isinstance(u, str) and "session_" in u:
            return "b:" + u[u.index("session_"):][:128]
    return None


def lane_of(req, headers, n):
    """The lane that a request names (LANE_FIELDS, x-reclank-lane), or None."""
    vals = [req.get(f) for f in LANE_FIELDS] + [headers.get("x-reclank-lane") if headers is not None else None]
    for v in vals:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            return i
    return None


# A request to go on after an answer that max_tokens cut (finish "length"): its last message is the
# user's bare "continue" (or these words), after that answer (continuation)
CONTINUE_RE = re.compile(r"^(please[\s,]+)?(continue|go on|keep going|carry on|resume|proceed|go ahead)"
                         r"([\s,]+please)?[\s.!]*$", re.I)
# the cut answers kept for a continue (the latest)
CUTS_KEPT = 8


def conversation_key(messages, tools):
    """A key of the messages of a chat request and its tools: the roles, the texts of the contents,
    the tool calls and their ids (the fields a client sends back as it got them; not the media, nor
    a reasoning_content some clients send and others drop)."""
    rows = [[m.get("role"), content_text(m.get("content")).strip(), m.get("tool_calls") or None,
             m.get("tool_call_id")] for m in messages]
    return hashlib.sha1(json.dumps([rows, tools or None], sort_keys=True, default=str).encode()).hexdigest()


# a lane busy with the request's conversation or session wins a wait by this many prompt tokens
# (a context comes back on the other lane in a copy: 160K positions in about 0.5 s)
WAIT_MARGIN = 512


def choose_lane(hits, live, busy, last_used, sticky, named=None, wait=None):
    """The lane of a request (--lanes 2). hits[i]: the prompt tokens lane i's cache would give it
    (a context in host memory counts on every lane: either brings it back in a copy); live[i]:
    that is the context on lane i's device; busy[i]: lane i runs or has requests waiting;
    last_used[i]: when lane i's last request ended; sticky: the lane of the request's session
    before, or None; named: a lane the client named; wait[i]: what a request waits for on lane i
    (LaneScheduler.wait, prompt tokens), or None. Return (lane, why).

    1. the lane the client named;
    2. the lane whose device holds the conversation, if it is free (on a busy one it is the
       request in flight: a title or a request beside it goes to the other lane);
    3. the lane of the session, if it is free (a conversation whose start changed, compacted or
       edited, takes the place of its own context there, not another's);
    4. a free lane: the one whose last request ended first;
    5. all busy: the lane with the least wait (a decode ends sooner than a long prompt; the
       conversation's or the session's lane by a margin, its context's copy, else the oldest
       end); with no waits, the conversation's lane, else the session's, else the oldest end."""
    n = len(hits)
    if named is not None:
        return named, "named"
    own = [i for i in range(n) if live[i] and hits[i] > 0]
    for i in own:
        if not busy[i]:
            return i, "its context"
    if sticky is not None and not busy[sticky]:
        return sticky, "its session"
    free = [i for i in range(n) if not busy[i]]
    if free:
        return min(free, key=lambda i: last_used[i]), "free"
    if wait is not None:
        mine = set(own) | ({sticky} if sticky is not None else set())
        lane = min(range(n), key=lambda i: (wait[i] - (WAIT_MARGIN if i in mine else 0), last_used[i]))
        return lane, ("its context (waits)" if lane in own else
                      "its session (waits)" if lane == sticky else "waits")
    if own:
        return own[0], "its context (waits)"
    if sticky is not None:
        return sticky, "its session (waits)"
    return min(range(n), key=lambda i: last_used[i]), "waits"


class Backend:
    """The model of the server, for the HTTP handler."""

    def __init__(self, engine, worker, model_id="reclank", thinking=None, max_tokens=1024,
                 temperature=1.0, top_k=None, top_p=None, mtp_accept="exact", mtp_floor=None,
                 media_dir=None, repetition_penalty=None):
        self.engine = engine
        self.worker = worker
        self.model_id = model_id
        # the think setting of a request that sets none (engine.think_level)
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # the repetition penalty of a request that gives none (LFM2.5: 1.05)
        self.repetition_penalty = repetition_penalty
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
        # the lanes (--lanes 2): the engine of each; the lane of each session; the lane of the
        # request of this thread (usage: its cached tokens)
        self.engines = list(getattr(engine, "lanes", None) or [engine])
        self.sessions = {}
        self._tls = threading.local()
        # the chat answers that max_tokens cut, by the key of their requests' messages
        # (conversation_key): their prompts and tokens, for a "continue" (continuation)
        self.cuts = {}
        self._cuts_lock = threading.Lock()

    # ---- what the handler reads of the model

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def max_context(self):
        return self.engine.max_context

    @property
    def last_cached(self):
        return self.engines[getattr(self._tls, "lane", 0) or 0].last_cached

    # ---- the lanes

    @property
    def lanes(self):
        return len(self.engines)

    def guard(self, lane):
        """What a generation holds: the lock of the model (one lane), or nothing (the lanes:
        the scheduler queues the requests of each lane)."""
        import contextlib
        return self.lock if self.lanes == 1 else contextlib.nullcontext()

    def route(self, req, headers, prompt_ids):
        """The lane of a request (choose_lane), on the model thread; None with one lane."""
        if self.lanes == 1 or getattr(self.worker, "lanes", None) is None:
            return None
        sid = session_of(req, headers)
        named = lane_of(req, headers, self.lanes)
        sch = self.worker.lanes

        def pick():
            hl = [e.hit(prompt_ids) for e in self.engines]
            hits, live = [h[0] for h in hl], [h[1] for h in hl]
            busy = [sch.lane_busy(i) for i in range(self.lanes)]
            wait = [sch.wait(i) for i in range(self.lanes)]
            lane, why = choose_lane(hits, live, busy, sch.last_used, self.sessions.get(sid), named,
                                    wait)
            new = len(prompt_ids) - hits[lane]
            sch.reserved[lane] += 1
            sch.reserved_tokens[lane] += new
            if sid is not None:
                self.sessions[sid] = lane
                while len(self.sessions) > 512:
                    self.sessions.pop(next(iter(self.sessions)))
            print("[lanes] lane %s (%s): cached %s of %d tokens, busy %s%s%s" % (
                "AB"[lane], why, "/".join(str(h) for h in hits), len(prompt_ids),
                "".join("AB"[i] for i in range(self.lanes) if busy[i]) or "-",
                " (waits %s)" % "/".join(str(w) for w in wait) if all(busy) else "",
                ", session %s" % sid[:24] if sid else ""), file=sys.stderr, flush=True)
            return lane, new
        lane, new = self.worker.call(pick)
        self._tls.lane = lane
        self._tls.new = new
        return lane

    def route_tokens(self):
        """The prompt tokens the request that route() placed on this thread reads (not cached)."""
        return getattr(self._tls, "new", 0)

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

    def remember_cut(self, req, prompt_ids, out, content):
        """A chat answer that max_tokens cut (finish "length"): the prompt and the answer's tokens,
        and the content the client got, for a "continue" after it (continuation)."""
        if not out:
            return
        ids = PromptIds(list(prompt_ids) + [int(t) for t in out])
        ids.spans = getattr(prompt_ids, "spans", ())
        key = conversation_key(req["messages"], req.get("tools"))
        with self._cuts_lock:
            self.cuts.pop(key, None)
            self.cuts[key] = {"ids": ids, "content": (content or "").strip(), "out": len(out)}
            while len(self.cuts) > CUTS_KEPT:
                self.cuts.pop(next(iter(self.cuts)))

    def continuation(self, req):
        """The prompt of a "continue" after an answer that max_tokens cut, or None: the request's
        messages are those of the cut answer's request, then that answer (its content as the client
        got it, no tool calls), then the user's bare "continue" (CONTINUE_RE). The prompt is the cut
        answer's prompt and all its tokens, so the model goes on where it stopped: in its think part,
        if that was open (a template drops the reasoning of an earlier turn: a Super answer of 32768
        tokens of reasoning, cut, and the "continue" started the thinking again from nothing). A
        continuation that is cut again is kept in its turn (a second "continue" goes on from it)."""
        msgs = req.get("messages") or []
        if len(msgs) < 3 or not self.cuts:
            return None
        last, prev = msgs[-1], msgs[-2]
        if (not isinstance(last, dict) or not isinstance(prev, dict) or last.get("role") != "user"
                or prev.get("role") != "assistant" or prev.get("tool_calls")):
            return None
        if not CONTINUE_RE.match(content_text(last.get("content")).strip()):
            return None
        with self._cuts_lock:
            cut = self.cuts.get(conversation_key(msgs[:-2], req.get("tools")))
        got = content_text(prev.get("content")).strip()
        # (a client's text of a stream that sent its content again ends in the content sent)
        if cut is None or not (got == cut["content"] or (cut["content"] and got.endswith(cut["content"]))):
            print("[reclank] continue: %s; the prompt as the template renders it" % (
                "no answer that max_tokens cut has these messages" if cut is None else
                "the answer before it is not the cut one as sent (%d chars, kept %d)" % (
                    len(got), len(cut["content"]))), file=sys.stderr, flush=True)
            return None
        self.last_think = self.engine.think_level(req, self.thinking)
        print("[reclank] continue: the answer that max_tokens cut goes on (its prompt and its %d tokens)"
              % cut["out"], file=sys.stderr, flush=True)
        return cut["ids"]

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
                       repetition_penalty=pick("repetition_penalty", self.repetition_penalty),
                       presence_penalty=req.get("presence_penalty"),
                       frequency_penalty=req.get("frequency_penalty"),
                       seed=req.get("seed"),
                       mtp_accept=pick("mtp_accept", self.mtp_accept),
                       mtp_floor=pick("mtp_floor", self.mtp_floor))

    def stop_ids(self):
        """Return the token ids that end a generation."""
        return self.engine.stop_ids()

    # generate takes session (server/http.py passes the client's conversation id)
    takes_session = True

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids, lane=None, tokens=0, session=None):
        """Yield the tokens of the answer; the engine runs on the model
        thread (lane: on that lane, beside the other's; tokens: the prompt
        tokens it reads, route_tokens()). session: the conversation id of the
        client (session_of), for an engine that takes it (session_ids)."""
        eng = self.engines[lane] if lane is not None else self.engine
        kw = {"session": session} if getattr(eng, "session_ids", False) else {}
        if lane is None:
            return self.worker.stream(lambda: eng.generate(prompt_ids, max_tokens, sampler, eos_ids, **kw))
        return self.worker.stream(lambda: eng.generate(prompt_ids, max_tokens, sampler, eos_ids, **kw),
                                  lane=lane, tokens=tokens)

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
