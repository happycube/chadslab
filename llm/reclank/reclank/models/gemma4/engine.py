"""The Gemma 4 models in the server (reclank/models/base.py): the 26B A4B and
the 12B (Model), the E2B and the E4B (E4B), on the CPU or the GPU, with the
MTP drafter and the media encoders.

The chat template is the canonical Gemma 4 one (chat_template.jinja), with
the thought channel (<|channel>thought ... <channel|>). The sessions keep the
key and value caches between the turns of a chat (core/session.py).
"""
from __future__ import annotations

import os
import sys
import time

from reclank.core.messages import PromptIds, content_text, has_media, message_parts
from reclank.core.session import SessionPool
from reclank.models.base import Engine

FAMILY = "gemma4"
THOUGHT_OPEN = "<|channel>thought\n"
OFF = ("none", "off", "disabled", "disable", "false", "0", "no")
# The words that close a thought channel at --think-budget (then <channel|>),
# and when fewer than --wrap-left tokens of the context are left: in the
# thought channel (then <channel|>), and in the answer.
THINK_CLOSE = "\n\nI have thought about this long enough; now I write the answer.<channel|>"
WRAP_THINK = "\n\nThe context is almost full, so I stop thinking and answer briefly now.<channel|>"
WRAP_ANSWER = "\n\n(The context is almost full, so I wrap up briefly now.)\n\n"
# The least context of a server for the thought channel by default (--thinking auto).
THINK_MIN_CONTEXT = 32768


def detect(meta):
    return str(meta.get("general.architecture", "")).startswith("gemma")


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


def tool_done(text):
    """True when the answer waits for a tool result: the model wrote
    <|tool_response> to ask for it. Without this test the model repeats the
    marker until it reaches the token limit. The marker also stops a turn
    that waits for a result with no new call."""
    return "<|tool_response>" in text


class GemmaEngine(Engine):
    family = FAMILY
    # generate takes the conversation id of the client (server/backend.py
    # session_of): the name of its context on disk (--context-dir)
    session_ids = True

    def __init__(self, model, tokenizer, cfg, drafter=None, n_draft=2, empty_thought_block=True,
                 max_context=None, contexts=4, store=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        # The MTP drafter and the count of drafts for each step (None: off).
        self.drafter = drafter
        self.n_draft = n_draft
        # False for the E2B and E4B models (Tokenizer.apply_chat_template).
        self.empty_thought_block = empty_thought_block
        self.max_context = max_context
        # the conversations kept (core/session.py): each session has its cache
        # in host memory; the model puts the one in use on the GPU
        # in host memory; the model puts the one in use on the GPU; store
        # (core/session.py ContextDir, --context-dir): on disk too
        self.pool = SessionPool(lambda max_len: self._session(max_len), contexts, store=store)
        # the thought channel: the ids of its markers and the words of the
        # budget and of the wrap-up (load); None: no guard
        self.think_tokens = None
        self.think_budget = 0
        self.wrap_left = 0
        # media input (--mmproj): the embedder of the soft rows, the default
        # budgets, and a cache of soft rows by media key
        self.embedder = None
        self.image_budget = 280
        self.video_budget = 70
        self.video_frames = 32
        self._media_rows = {}
        self.last_media = []
        self.last_text = None

    def _session(self, max_len):
        from reclank.models.gemma4.model import Session
        return Session(self.model, max_len=max_len, drafter=self.drafter, n_draft=self.n_draft)

    # ---- the prompt

    def think_level(self, req, default):
        v = request_thinking(req)
        return bool(default) if v is None else bool(v)

    def prompt_ids(self, messages, level, tools=None, preserve=None, source=None):
        """Return the token ids of a chat prompt."""
        think = bool(level)
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
            return self.media_prompt(text, parts, source)
        self.last_text = text
        return self.tokenizer.encode(text)

    def media_prompt(self, text, parts, source):
        """Return the PromptIds of a prompt with media parts: the soft rows of
        each part (the embedder), and their tokens in the ids."""
        from reclank.encoders import media as MD
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
                data = source.video(part)
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
                data, budget = source.image(part, self.image_budget)
                key = MD.media_key(data) + ":%d" % budget
                rows = self._media_rows.get(key)
                if rows is None:
                    rows, _ = self.embedder.image(data, budget)
                items.append(MD.Media("image", rows, key,
                                      bidir=getattr(self.embedder, "bidir", True)))
            else:
                data = source.audio(part)
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
        self.last_text = text
        print("[reclank] media: %s" % ", ".join(
            "%s %d tokens" % (sp.kind, sp.rows.shape[0]) for sp in spans),
            file=sys.stderr, flush=True)
        return out

    def open_channel(self, prompt_ids):
        """Return "<|channel>thought\n" if the prompt ends in that open
        channel, else "". The Gemma 4 template ends the prompt so after a
        tool result when the thought channel is on; the model then writes
        its thought with no opener and closes it with <channel|>."""
        tail = self.tokenizer.decode(list(prompt_ids[-4:]), skip_special_tokens=False)
        return THOUGHT_OPEN if tail.endswith(THOUGHT_OPEN) else ""

    def completion_ids(self, prompt):
        return self.tokenizer.encode(prompt)

    # ---- the answer

    def parse_output(self, text):
        from reclank.models.gemma4.chat import parse_output
        return parse_output(text)

    def tool_done(self, text):
        return tool_done(text)

    def turn_fields(self, rec):
        return {"prompt_text": self.last_text}

    # ---- the decode

    def think_guard(self, prompt_ids, max_tokens, eos_ids):
        """The ThinkGuard of a request (reclank/core/think_guard.py) when the
        engine has think_tokens, else None: the think part closes at
        think_budget tokens (and at most half of max_tokens), and fewer than
        wrap_left tokens of the context left (max_context, else 262144) put
        in the words to wrap up."""
        tt = self.think_tokens
        if tt is None:
            return None
        from reclank.core.think_guard import ThinkGuard, think_budget
        left, wrap_at = self.wrap_left, None
        room = (self.max_context or 262144) - len(prompt_ids) - 1
        if left > 0 and max_tokens > room - left:
            wrap_at = max(1, room - left + 1)
        return ThinkGuard(eos_ids, tt["close"],
                          think_budget(self.think_budget, max_tokens, len(tt["force"])),
                          tt["force"], think=bool(self.open_channel(prompt_ids)), wrap_at=wrap_at,
                          wrap_think=tt["wrap_think"], wrap_answer=tt["wrap_answer"],
                          tool=(tt["tool_open"], tt["tool_close"]), open_id=tt["open"])

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids, session=None):
        """Yield one token id at a time for a prompt. session: the
        conversation id of the client, or None (core/session.py ContextDir)."""
        sid = session
        session, shared = self.pool.pick(prompt_ids, max_tokens, sid=sid)
        print('[reclank] reuse=%d new=%d; %s' % (shared, len(prompt_ids) - shared,
                                                 self.pool.note(len(prompt_ids), shared)),
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
                # that fails reads all of them, whatever the pool found)
                t2, new = time.time(), getattr(session, "prefilled", len(prompt_ids) - shared)
                print("[reclank] prompt %d new tokens in %.2f s (%.0f tok/s); decode %d tokens in "
                      "%.1f s (%.1f tok/s)" % (new, t1 - t0, new / max(t1 - t0, 1e-9), n - 1, t2 - t1,
                                               (n - 1) / max(t2 - t1, 1e-9)), file=sys.stderr, flush=True)
            # also when the caller stops at a stop token and closes the
            # generator
            sampler.guard = None
            self.pool.done(session, sid=sid)    # a long prompt to disk (--context-dir)
            if guard is not None and guard.fixes:
                print("[reclank] the model ended the turn inside the thought channel: <channel|> "
                      "in its place", file=sys.stderr, flush=True)
            if guard is not None and guard.wrapped:
                print("[reclank] fewer than %d tokens of the context left at token %d of the answer: "
                      "the words to wrap up" % (self.wrap_left, guard.wrap_at), file=sys.stderr,
                      flush=True)
            if guard is not None and guard.budget_hits:
                print("[reclank] the think part reached its budget (%d tokens): closed" % guard.budget,
                      file=sys.stderr, flush=True)
            st = session.mtp_stats
            if self.drafter is not None and st.get("drafts"):
                print('[reclank] mtp steps=%d drafts=%d accepted=%d (%d%%)' % (
                    st["steps"], st["drafts"], st["accepted"],
                    100 * st["accepted"] // st["drafts"]), file=sys.stderr, flush=True)


# The room of the GPU after the weights (gpu.room) under which the default
# is 2 drafts, not 3. (3.0e9 before the programs packed their temporaries,
# ir/alloc.py: the 12B with 10 GB, 2.6 GB of room, then ran 3 drafts at
# 95 tok/s; packed 128.0, 2 drafts 127.2.)
SMALL_ROOM = 1.5e9
# The memory of a card (the free memory before the weights, at most its
# size) under which MTP is off by default: a card of less than 8 GB (one of
# 8 GB has about 8.0e9 bytes free, one of 6 GB at most 6.4e9).
SMALL_CARD = 7.5e9
# The default context (--ctx) of the 26B on a card of less than 12 GB: the
# cache, the programs of the prompt (groups of 2048 tokens on 8 GB:
# gpu.chunk_for) and the output head (0.61 GB, on the GPU at the first
# prompt) must fit beside the hot experts, which give way. The GTX 1070
# (8.5 GB): a cache of 163840 tokens and a prompt of 8192 fit (the cache
# grows by 11.6 KB a token: the 5 global layers; the sliding ones keep their
# window). With 4 GB of it held (4.4 GB free, a small card: groups of 1024
# tokens): a prompt of 40000 tokens in a cache of 40960 fits with 0.07 GB to
# spare, 48K does not (the head finds no 0.61 GB); 32768 leaves 0.33 GB.
CTX_8GB = 163840
CTX_SMALL = 32768


def load(opts, meta=None):
    """Load a Gemma 4 model with the server options (server/options.py).
    Return (engine, defaults): the defaults of the Backend (the sampling of
    the model file, the think default)."""
    from reclank.formats.gguf import GGUF
    from reclank.models.gemma4 import model as model_mod
    from reclank.models.gemma4.config import Config
    from reclank.models.gemma4.tokenizer import Tokenizer

    # The cache copy was int8 with an int8 query. Its error sent a long greedy
    # generation into a loop, so the server used the float cache. The copy is
    # now int16 with a float query, which is as accurate as the float cache
    # and faster at a long context.
    kv = opts.kv_attn or "int16"
    os.environ["RECLANK_ATTN"] = "0" if kv == "float" else "1"
    if kv in ("int8", "k16v8", "rq8", "k16vr8"):
        model_mod.KV_FORM = kv
    from reclank.models.qwen.keep import gguf_or_kept
    g = gguf_or_kept(opts.model)        # (the weightkeeper's tensors when one holds the file)
    tok = Tokenizer(opts.tokenizer) if opts.tokenizer else Tokenizer.from_gguf(g)
    tc = g.text_config()
    e4b = bool(tc.get("hidden_size_per_layer_input"))
    print("loading %s ..." % opts.model, flush=True)
    gpu_on = opts.backend == "cuda"
    if opts.backend == "auto":
        # the Gemma default: the CPU unless asked (as serve.py --gpu off)
        gpu_on = bool(getattr(opts, "resident", False))
    if getattr(opts, "resident", False) and not gpu_on:
        raise SystemExit("--resident needs the GPU (--backend cuda)")
    small_card = False
    if gpu_on:
        from reclank.backends.cuda.gpumm import mem_info
        free0, total0 = mem_info()
        small_card = min(free0, total0) < SMALL_CARD
    if e4b:
        from reclank.models.gemma4.e4b import E4B, E4BConfig
        cfg = E4BConfig({"text_config": tc})
        model = E4B(g, cfg, mode="int4" if (opts.dtype or "int4") == "int4" else "f32")
        if opts.drafter and gpu_on and not small_card:
            # The E4B with its drafter on the GPU is faster with MTP than
            # without it (SPLIT_PLAN.md, phase 5).
            os.environ.setdefault("RECLANK_MTP", "1")
    else:
        cfg = Config({"text_config": tc})
        model = model_mod.Model(g, cfg).load_all(dtype=opts.dtype or "int4")
    if gpu_on:
        from reclank.models.gemma4 import gpu
        if not e4b:
            # an expert device (RECLANK_EXPERT_GPUS) copies its experts while the
            # weights go to this GPU
            from reclank.backends.cuda.devices import start_expert_device
            start_expert_device(model)
        print("copying the weights to the GPU ...", flush=True)
        dev = gpu.offload(model, 0.0 if opts.gpu_dense else opts.hot_gb,
                          resident=getattr(opts, "resident", False))
        print(gpu.describe(dev), flush=True)
    # The model file gives the sampling that the model wants. Use it unless the
    # caller asks for another value. A greedy default is not the wish of the
    # model and it hides the setting from the client.
    rec = {k: g.meta.get("general.sampling." + k) for k in ("temp", "top_p", "top_k")}
    defaults = {
        "temperature": float(rec["temp"]) if rec["temp"] is not None else 1.0,
        "top_p": float(rec["top_p"]) if rec["top_p"] is not None else None,
        "top_k": int(rec["top_k"]) if rec["top_k"] is not None else None,
        "max_tokens": 1024, "image_budget": 280, "video_budget": 70,
        "model_id": os.path.basename(opts.model).rsplit(".", 1)[0],
    }
    drafter = None
    # The drafts of a step and RECLANK_MTP_STOP (they end where the product
    # of their probabilities falls under it): with the drafter on the GPU 3
    # drafts and 0.9, the best (or within 1% of it) of each model on the
    # 3090 (MTP tok/s of a chat, 512 tokens; 2 drafts with no stop in
    # brackets): the 26B 165 with hot experts (163), 169 resident (165); the
    # 12B 129 (126); the E4B 182 (175). On the CPU the stop gained little
    # (within the noise): 2 drafts, no stop. --mtp and the variable win.
    n_def, stop_def = (3, 0.9) if gpu_on else (2, 0.0)
    if gpu_on and gpu.room(dev) < SMALL_ROOM:
        # Little memory after the weights: the programs of 3 drafts and of
        # a prompt do not all fit above the reserve, and each prompt makes
        # them again (before the packing of ir/alloc.py the 12B on 10 GB,
        # 2.6 GB of room: 95 tok/s with 3 drafts, 126 with 2).
        n_def = 2
    if small_card:
        # a card of less than 8 GB (SMALL_CARD): the memory of the drafter
        # and of the programs of the groups goes to the cache
        n_def = 0
    n_draft = opts.mtp if opts.mtp is not None else n_def
    from reclank.core.speculative import mtp_stop
    stop = mtp_stop(stop_def)
    if opts.drafter and n_draft <= 0:
        # --mtp 0, or a card of less than 8 GB: no drafter
        print("MTP off%s: the drafter is not loaded" % (
            " (less than %.1f GB of the GPU)" % (SMALL_CARD / 1e9) if small_card and opts.mtp is None else ""),
            flush=True)
    elif opts.drafter and "RECLANK_MTP" not in os.environ:
        # With a GPU the default of assistant.mtp_enabled is off; the drafter
        # was asked for, so turn MTP on.
        os.environ["RECLANK_MTP"] = "1"
    if opts.drafter and n_draft > 0:
        print("loading the MTP drafter %s ..." % opts.drafter, flush=True)
        if gpu_on:
            # The drafter on the GPU reads the cache on the GPU.
            from reclank.models.gemma4.gpu import GPUDrafter
            drafter = GPUDrafter(opts.drafter, model, stop=stop)
        else:
            from reclank.models.gemma4.assistant import Assistant
            drafter = Assistant(opts.drafter, dtype=opts.mtp_dtype or "int4")
            drafter.stop = stop
        print("MTP: %d drafts a step%s" % (n_draft, ", they end where the product of their "
                                           "probabilities falls under %g" % stop if stop > 0 else ""),
              flush=True)
    ctx = opts.ctx
    if ctx is None and gpu_on and getattr(cfg, "enable_moe_block", False) and min(free0, total0) < 12e9:
        ctx = CTX_SMALL if small_card else CTX_8GB
        print("context: %d tokens by default on a card of %.1f GB (--ctx sets it)" % (ctx, total0 / 1e9),
              flush=True)
    thinking = opts.thinking or "auto"
    defaults["thinking"] = thinking in ("on", "low", "medium", "high", "xhigh") or (
        thinking == "auto" and (ctx is None or ctx >= THINK_MIN_CONTEXT))
    print("thinking %s by default (--thinking %s)" % ("on" if defaults["thinking"] else "off", thinking),
          flush=True)
    store = None
    cdir = getattr(opts, "context_dir", None) or os.environ.get("RECLANK_CONTEXT_DIR")
    if cdir and not e4b:
        from reclank.core.session import ContextDir
        # the tag: the model file and the form of the host cache (a file of
        # another one is never loaded)
        tag = "%s:%d:%s" % (os.path.basename(opts.model), os.path.getsize(opts.model), model_mod.KV_FORM)
        store = ContextDir(cdir, tag, max_bytes=getattr(opts, "context_dir_gb", 64.0) * 1e9,
                           min_tokens=int(os.environ.get("RECLANK_CONTEXT_SAVE_MIN", "4096")))
        print("contexts on disk: %s (at most %.0f GB; prompts of %d tokens or more)" % (
            cdir, store.max_bytes / 1e9, store.min_tokens), flush=True)
    eng = GemmaEngine(model, tok, cfg, drafter=drafter, n_draft=n_draft, empty_thought_block=not e4b,
                      max_context=ctx, contexts=getattr(opts, "contexts", 4), store=store)
    # the thought channel of Gemma 4 (<|channel>thought ... <channel|>): the
    # budget of a think part and the words to wrap up (core/think_guard.py)
    one = lambda t: tok.encode(t)[0] if len(tok.encode(t)) == 1 else None  # noqa: E731
    eng.think_budget, eng.wrap_left = opts.think_budget, max(0, opts.wrap_left)
    if one("<|channel>") is not None and one("<channel|>") is not None:
        eng.think_tokens = {
            "open": one("<|channel>"), "close": one("<channel|>"),
            "force": tok.encode(THINK_CLOSE), "wrap_think": tok.encode(WRAP_THINK),
            "wrap_answer": tok.encode(WRAP_ANSWER),
            "tool_open": one("<|tool_call>"), "tool_close": one("<tool_call|>")}
    if opts.mmproj:
        from reclank.models.gemma4.encoders import load_embedder
        # The encoders on the GPU when the model is (RECLANK_MEDIA_GPU=0: CPU).
        media_gpu = gpu_on and os.environ.get("RECLANK_MEDIA_GPU", "1") != "0"
        q8 = opts.mmproj_q8 == "on" or (opts.mmproj_q8 in (None, "auto") and e4b)
        eng.embedder = load_embedder(opts.mmproj, gpu=media_gpu, q8=q8)
        eng.image_budget = opts.image_budget or defaults["image_budget"]
        eng.video_frames = opts.video_frames or 32
        eng.video_budget = opts.video_budget or defaults["video_budget"]
        print("media input: %s (images%s), image budget %d%s" % (
            opts.mmproj, " and audio" if eng.embedder.has_audio else "",
            eng.image_budget, ", Q8_0 encoder weights" if getattr(eng.embedder, "q8", False)
            else ""), flush=True)
    return eng, defaults
