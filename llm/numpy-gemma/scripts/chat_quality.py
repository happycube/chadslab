#!/usr/bin/env python3
"""The quality of a form of Qwen3.8-Flash-Next on chat transcripts.

TEST_PLAN.md, tier 3: on README or source text a model is unsure of many
tokens, and the top-token agreement of two forms says little; on chat text
(the answers of the model) it is sure of most of them. Three steps:

    make  TEXTS.json      the transcripts: chat prompts (code, a fix of code,
                          prose, an explanation with a follow-up, math,
                          German, a tool call and its result, a chat of three
                          turns), the answers greedy from --gguf (the form
                          closest to the original: RQ8_0)
    score TEXTS.json OUT  each position of the answers under --gguf: the top
                          64 log-probs, the log-prob of the next token, and
                          (--ref REF.npz) the log-probs at the top 64 of REF;
                          --decode: the answers through the kernels of the
                          decode (verify groups of 4 rows)
    compare REF A B ...   for each run against REF: the KL over the top 64 of
                          REF, the top-token agreement, the NLL of the
                          answers, and the disagreements where REF is sure

    python scripts/chat_quality.py make tests/texts/qwen38_chat.json --gguf RQ8.gguf
    NP_GEMMA_GPU_BF16_TC=0 NP_GEMMA_GPU_QSA_TC=0 python scripts/chat_quality.py \\
        score tests/texts/qwen38_chat.json ref.npz --gguf RQ8.gguf
    python scripts/chat_quality.py score tests/texts/qwen38_chat.json nvfp4.npz \\
        --gguf NVFP4.gguf --ref ref.npz
    python scripts/chat_quality.py compare ref.npz rq8.npz nvfp4.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

K = 64

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The current weather and a 3-day forecast for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "The city, with its country"},
        "units": {"type": "string", "enum": ["metric", "imperial"]}},
        "required": ["city"]}}}]

CONVS = [
    ("code", [
        "Write a Python function that merges overlapping intervals in a list of (start, end) "
        "pairs. Include a docstring and a few tests with assert."]),
    ("code_fix", [
        "This function should return the second largest distinct number in a list, but it has "
        "bugs. Find them and give a fixed version.\n\n```python\ndef second_largest(xs):\n"
        "    first = second = 0\n    for x in xs:\n        if x > first:\n            first = x\n"
        "        elif x > second:\n            second = x\n    return second\n```"]),
    ("prose", [
        "Write a short story (about 250 words) about a lighthouse keeper who finds a message in "
        "a bottle that seems to be written by herself."]),
    ("explain", [
        "Explain how public-key cryptography works to a high-school student.",
        "How does that relate to the padlock icon in my web browser?"]),
    ("math", [
        "A train leaves city A at 9:00 at 80 km/h. Another leaves city B, 300 km away, at 10:00 "
        "toward A at 100 km/h. When and where do they meet? Show the steps."]),
    ("german", [
        "Erkläre in einfachen Worten, wie ein Kühlschrank funktioniert.",
        "Und warum wird es hinter dem Kühlschrank warm?"]),
    ("tool", [
        "What's the weather like in Lisbon right now, and should I pack an umbrella for the "
        "weekend?",
        {"role": "tool", "content": json.dumps({
            "city": "Lisbon, Portugal", "now": {"temp_c": 19, "sky": "partly cloudy", "wind_kmh": 14},
            "forecast": [{"day": "Fri", "high_c": 21, "rain_pct": 10},
                         {"day": "Sat", "high_c": 18, "rain_pct": 70},
                         {"day": "Sun", "high_c": 17, "rain_pct": 80}]})}]),
    ("chat", [
        "I have a free weekend and about 300 euros. Any ideas for a short trip from Munich?",
        "I like hiking but I don't have a car.",
        "Great, which of those would you pick for late October, and why?"]),
]


def template(gdir):
    import jinja2
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                             extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    env.filters["items"] = lambda d: list(d.items()) if isinstance(d, dict) else []

    def raise_exception(msg):
        raise ValueError(msg)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(open(os.path.join(gdir, "chat_template.jinja"), encoding="utf-8").read())


def load(gguf, ctx):
    from np_gemma.qwen4 import Qwen4CPU
    from np_gemma.qwen4_gpu import Qwen4GPU
    from np_gemma.qwen_tok import QwenTokenizer
    tp = os.path.join(os.path.dirname(gguf), "tokenizer.json")
    tok = QwenTokenizer(tp) if os.path.exists(tp) else None     # score needs none
    # EXPERTS: an overlay of the experts (GGUFOverlay), e.g. the NVFP4 ones
    m = Qwen4CPU(gguf, experts=os.environ.get("EXPERTS") or None)
    return tok, m, Qwen4GPU(m, ctx=ctx)


def make(args):
    """The transcripts: the answers greedy from the model."""
    from np_gemma.qwen4 import Qwen4Cache
    tok, m, g = load(args.gguf, 8192)
    tpl = template(os.path.dirname(args.gguf))
    stop = set(tok.stop_ids)
    out = []
    for topic, turns in CONVS:
        msgs = []
        t0 = time.time()
        for turn in turns:
            msgs.append(turn if isinstance(turn, dict) else {"role": "user", "content": turn})
            text = tpl.render(messages=msgs, tools=TOOLS if topic == "tool" else None,
                              add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(text)
            c = Qwen4Cache(m.cfg, len(ids) + args.tokens + 64)
            g.attach(c)
            g.prefill(ids)
            nxt, pos, ans = int(g.argmax()[0]), len(ids), []
            while True:
                ans.append(nxt)
                if nxt in stop or len(ans) >= args.tokens:
                    break
                g.step(nxt, pos)
                nxt = int(g.argmax()[0])
                pos += 1
            msgs.append({"role": "assistant", "content": tok.decode([a for a in ans if a not in stop])})
        # the whole conversation, and the spans of the answers in it
        text = tpl.render(messages=msgs, tools=TOOLS if topic == "tool" else None,
                          add_generation_prompt=False, enable_thinking=False)
        ids = tok.encode(text)
        spans = []
        for a in (x for x in msgs if x["role"] == "assistant"):
            a_ids = tok.encode(a["content"])
            # the first match after the last span
            start = spans[-1][1] if spans else 0
            for i in range(start, len(ids) - len(a_ids) + 1):
                if ids[i:i + len(a_ids)] == a_ids:
                    spans.append((i, i + len(a_ids)))
                    break
            else:
                raise RuntimeError("%s: an answer is not in the rendered conversation" % topic)
        out.append({"topic": topic, "ids": ids, "spans": spans, "messages": msgs})
        print("%-9s %4d tokens, answers %s (%.0f s)" % (
            topic, len(ids), [b - a for a, b in spans], time.time() - t0), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.texts)), exist_ok=True)
    json.dump({"source": os.path.basename(args.gguf), "convs": out}, open(args.texts, "w"),
              ensure_ascii=False, indent=1)
    g.close()


def logprobs(g, rows):
    lg = g.logits(x=np.ascontiguousarray(rows)).astype(np.float64).reshape(rows.shape[0], -1)
    lg -= lg.max(1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(1, keepdims=True))


def decode_logprobs(g, m, ids, spans):
    """--decode: the log-probs of the positions of the answers through the
    kernels of the decode: the prompt before each answer as a prompt, the
    answer in MTP verify groups of 4 rows (a step for a last single row),
    each kept (commit), the logits from the head of the GPU."""
    from np_gemma.qwen4 import Qwen4Cache
    g.attach(Qwen4Cache(m.cfg, len(ids) + 64))
    out, p = [], 0
    for a, b in spans:
        if a - 1 > p:
            g.prefill(ids[p:a - 1], pos=p)
            p = a - 1
        while p < b - 1:
            n = min(4, b - 1 - p)
            if n > 1:
                g.verify(ids[p:p + n], p)
            else:
                g.step(ids[p], p)
            lg = np.asarray(g.logits(rows=n), np.float64).reshape(n, -1)
            if n > 1:
                g.commit(n)
            lg -= lg.max(1, keepdims=True)
            out.append(lg - np.log(np.exp(lg).sum(1, keepdims=True)))
            p += n
    return np.concatenate(out)


def score(args):
    """The log-probs of the positions of the answers."""
    from np_gemma.qwen4 import Qwen4Cache
    data = json.load(open(args.texts))
    convs = data["convs"]
    n_max = max(len(c["ids"]) for c in convs)
    tok, m, g = load(args.gguf, n_max + 64)
    ref = np.load(args.ref) if args.ref else None
    top_i, top_p, nxt_p, conv_of, ref_p = [], [], [], [], []
    k0 = 0
    for ci, c in enumerate(convs):
        ids = c["ids"]
        # position p predicts ids[p + 1]: the answers' tokens a .. b - 1
        pos = np.concatenate([np.arange(a - 1, b - 1) for a, b in c["spans"]])
        if args.decode:
            lps = decode_logprobs(g, m, ids, c["spans"])
        else:
            # whole groups of 2048 rows: a group's padded rows need their place
            cache = Qwen4Cache(m.cfg, -(-len(ids) // 2048) * 2048 + 64)
            g.attach(cache)
            h = np.concatenate([g.mix(ids[c0:c0 + 2048], c0, 2048) for c0 in range(0, len(ids), 2048)])
        for s in range(0, len(pos), 64):
            pp = pos[s:s + 64]
            lp = lps[s:s + 64] if args.decode else logprobs(g, h[pp])
            ti = np.argsort(-lp, axis=1)[:, :K]
            top_i.append(ti.astype(np.int32))
            top_p.append(np.take_along_axis(lp, ti, 1).astype(np.float32))
            nxt_p.append(lp[np.arange(len(pp)), np.asarray(ids)[pp + 1]].astype(np.float32))
            conv_of.append(np.full(len(pp), ci, np.int16))
            if ref is not None:
                ri = ref["top_i"][k0:k0 + len(pp)]
                ref_p.append(np.take_along_axis(lp, ri.astype(np.int64), 1).astype(np.float32))
            k0 += len(pp)
        print("%-9s %4d positions" % (c["topic"], len(pos)), flush=True)
    out = dict(top_i=np.concatenate(top_i), top_p=np.concatenate(top_p),
               nxt_p=np.concatenate(nxt_p), conv=np.concatenate(conv_of),
               topics=np.array([c["topic"] for c in convs]), gguf=os.path.basename(args.gguf),
               env=json.dumps({k: v for k, v in os.environ.items() if k.startswith("NP_GEMMA")}))
    if ref is not None:
        out["ref_p"] = np.concatenate(ref_p)
    np.savez(args.out, **out)
    g.close()


def compare(args):
    ref = np.load(args.ref)
    rp = ref["top_p"].astype(np.float64)
    pr = np.exp(rp)
    sure = pr[:, 0] > 0.9
    topics = list(ref["topics"])
    print("reference %s: %d positions, %.0f%% of them with p(top) > 0.9; NLL %.4f"
          % (ref["gguf"], len(rp), 100 * sure.mean(), -ref["nxt_p"].mean()))
    print("%-34s %8s %8s %8s %9s  %s" % ("run", "KL64", "top-1", "NLL", "flip|sure", "KL by topic"))
    for path in args.runs:
        r = np.load(path)
        q = r["ref_p"].astype(np.float64)
        kl = (pr * (rp - q)).sum(1)
        agree = r["top_i"][:, 0] == ref["top_i"][:, 0]
        by = " ".join("%s %.4f" % (t[:5], kl[r["conv"] == i].mean()) for i, t in enumerate(topics))
        print("%-34s %8.4f %7.2f%% %8.4f %8.2f%%  %s" % (
            os.path.basename(path)[:34], kl.mean(), 100 * agree.mean(), -r["nxt_p"].mean(),
            100 * (~agree[sure]).mean(), by))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make")
    a.add_argument("texts")
    a.add_argument("--gguf", required=True)
    a.add_argument("--tokens", type=int, default=400, help="the most tokens of an answer")
    a = sub.add_parser("score")
    a.add_argument("texts")
    a.add_argument("out")
    a.add_argument("--gguf", required=True)
    a.add_argument("--ref", default=None)
    a.add_argument("--decode", action="store_true",
                   help="the answers through the decode (verify groups of 4), not prompt groups")
    a = sub.add_parser("compare")
    a.add_argument("ref")
    a.add_argument("runs", nargs="+")
    args = ap.parse_args()
    {"make": make, "score": score, "compare": compare}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
