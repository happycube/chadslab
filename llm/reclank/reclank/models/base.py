"""The interface between a model family and the server (REFACTOR_PLAN.md,
section 5.1).

A family module (reclank/models/<family>/engine.py) gives:

    detect(meta) -> bool       the GGUF metadata (or the HF config) is of this family
    load(opts) -> Engine       load the model with the server options

The server (server/backend.py) talks to an Engine only. The Engine owns
the decode loop of its family (its caches, the reuse of a cache between
the turns of a chat, MTP), so the answers stay those of the family code;
the server owns the request: the sampling settings, the limits of the
tokens, the think budget words, the debug records, and the one model
thread (server/worker.py) on which load() and generate() run.
"""
from __future__ import annotations


class Engine:
    """The defaults of an engine; a family overrides what it has."""

    family = ""
    # encode(text), decode(ids, skip_special_tokens=False), stop_ids
    tokenizer = None
    # The most tokens of a request (the prompt and the answer), or None
    # (no limit). A longer prompt gets an error; max_tokens gets the room.
    max_context = None
    # The prompt tokens that the cache held in the last request (the usage
    # of the response), or None when the engine does not count them.
    last_cached = None
    # An error that leaves the device unusable (note_error); the model
    # thread then writes a crash report and ends the process.
    fatal = None

    def think_level(self, req, default):
        """The think setting of a request (False or None: off; the family
        knows the levels), from the request fields, else default."""
        raise NotImplementedError

    def think_off(self):
        """The value of think_level for "no think part"."""
        return False

    def prompt_ids(self, messages, level, tools=None, preserve=None, source=None):
        """The token ids of a chat prompt (a list; a PromptIds with .spans
        for media). source (server/media.py) reads the bytes of media parts."""
        raise NotImplementedError

    def completion_ids(self, prompt):
        return list(self.tokenizer.encode(prompt))

    def open_channel(self, prompt_ids):
        """The opener of the think part when the prompt leaves it open (the
        answer starts inside it), else ""."""
        return ""

    def stop_ids(self):
        return set(getattr(self.tokenizer, "stop_ids", ()) or ())

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        """Yield the token ids of the answer (on the model thread)."""
        raise NotImplementedError

    def parse_output(self, text):
        """{"reasoning", "content", "tool_calls", "pending"} of the raw text."""
        raise NotImplementedError

    def tool_done(self, text):
        """True when the answer waits for a tool result (the turn ends)."""
        return False

    def clean_raw(self, raw):
        """The raw text for the logs and the debug files."""
        return raw

    def turn_fields(self, rec):
        """More fields of the debug record of a turn."""
        return {}

    def note_error(self, exc):
        """An error of a job; a family marks the fatal ones (fatal)."""

    def crash_report(self, exc, where_dir=None):
        """Write a report of a fatal error; return its directory or None."""
        return None


def reload_lanes(eng, make):
    """A reload's start of the lanes and its warm-up (reload_device, docs/GPU_SWAP_PLAN.md 3.6):
    make() runs with the contexts in host memory set aside (an empty store, as at the load: the
    warm-up's prompts do not go on from the requests' contexts), then each lane gets its store
    back; lane B's engine stays the same object, with the new state (a request the server routed
    to it before the reload runs on it)."""
    old = list(eng.lanes)
    real = [e.ctxs for e in old]
    c0 = real[0]
    eng.ctxs = type(c0)(eng, c0.max_contexts, c0.max_bytes, c0.name)
    eng.lanes = [eng]
    make()
    new = eng.lanes
    for i in range(1, min(len(old), len(new))):
        o, n = old[i], new[i]
        if o is not n:
            o.__dict__.clear()
            o.__dict__.update(n.__dict__)
            new[i] = o
    for e, c in zip(new, real):
        e.ctxs = c
