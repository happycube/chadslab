#!/usr/bin/env python3
"""The quality of a model's answers through the server API (tests/quality/suite.py).

The items run against an OpenAI-compatible server (serve_qwen4 or any other):
the chat template, the think part, and the parse of tool calls are part of
what is measured, as a client (dsh) sees them. Each item has a grader:
exact numbers (math), unit tests in a subprocess (code), the tool and its
arguments, or no tool (tools), a check of the form (instruct), one letter
(mcq), a passphrase in a long text (needle). Every answer is also checked
for the faults of a server: a finish by length, an empty answer, tags of
the template in the content (<think>, </think>, <tool_call>, <|im_...),
and (with a think part asked for) a missing reasoning.

    # a run (greedy; the server's own think level unless --effort)
    NP_GEMMA_API_KEY=... python scripts/quality_suite.py run --label nvfp4
    python scripts/quality_suite.py run --label mix --only math,tools --effort low
    python scripts/quality_suite.py run --label long --needle 4000,32000,128000 --only needle
    python scripts/quality_suite.py run --label nvfp4-hard --tier hard     # tests/quality/hard.py
    python scripts/quality_suite.py run --label nvfp4-x --tier expert      # tests/quality/expert.py
    NP_GEMMA_API_KEY=sk-or-... python scripts/quality_suite.py run --label or-qwen-flash \
        --url https://openrouter.ai/api/v1 --model qwen/qwen3.8-flash          # OpenRouter
    ... --url https://openrouter.ai/api/v1 --model qwen/qwen3.8-flash --provider alibaba  # one provider

    # the scores of runs side by side, and the items that changed
    python scripts/quality_suite.py compare ~/quality-runs/nvfp4-*.json ~/quality-runs/mix-*.json

The runs go to --out (default ~/quality-runs, not the repository): the
answers in full, the grade and note of each item, the faults, the tokens
and the seconds. For the distribution of the logits (KL against a
reference, close to a form's noise), see scripts/chat_quality.py,
scripts/check_qwen4_kl.py, and scripts/check_mtp_accuracy.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests", "quality"))

MAX_TOKENS = {"math": 6000, "code": 6000, "tools": 4000, "instruct": 4000, "mcq": 4000,
              "needle": 3000, "hmath": 12000, "hreason": 12000, "hcode": 12000, "htools": 6000,
              "hinstruct": 6000, "hneedle": 6000, "xmath": 16000, "xreason": 16000, "xcode": 16000,
              "xtools": 8000, "xneedle": 12000}
AGENT_TURNS = 12        # the most model turns of an agent task (htools)
RETRIES = 8             # the retries of a request after a 429 (about 4 min in all)
LEAKS = re.compile(r"<think>|</think>|<tool_call>|</tool_call>|<\|im_(?:start|end)\|>|<\|channel>")


def reasoning(msg):
    """The think part: reasoning_content (serve_qwen4, vLLM, llama.cpp) or reasoning (OpenRouter)."""
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def post(url, key, body, timeout):
    """One request; a 429 (too many requests: an HTTP status, or an error in
    the body as OpenRouter gives for its providers) waits and asks again:
    the server's Retry-After, else 2, 4, 8 ... 60 s with a jitter, at most
    RETRIES times."""
    data = json.dumps(body).encode()
    hdr = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
    for n in range(RETRIES + 1):
        req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data, hdr)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            err = d.get("error") if not d.get("choices") else None
            if not (isinstance(err, dict) and err.get("code") == 429) or n == RETRIES:
                return d
            wait = None
        except urllib.error.HTTPError as e:
            if e.code != 429 or n == RETRIES:
                raise
            wait = e.headers.get("Retry-After")
        try:
            wait = float(wait)
        except (TypeError, ValueError):
            wait = min(60.0, 2.0 * 2 ** n) * random.uniform(0.75, 1.25)
        print("    429: waiting %.0fs (retry %d/%d)" % (wait, n + 1, RETRIES), flush=True)
        time.sleep(wait)


def faults(msg, finish, effort):
    out = []
    if finish == "length":
        out.append("length")
    c = msg.get("content") or ""
    if not c.strip() and not msg.get("tool_calls"):
        out.append("empty")
    if LEAKS.search(c):
        out.append("leak:" + LEAKS.search(c).group(0))
    if effort not in (None, "none") and not reasoning(msg).strip() \
            and not msg.get("tool_calls"):
        out.append("no-reasoning")
    return out


def run(args):
    import suite
    lengths = tuple(int(x) for x in args.needle.split(",") if x)
    items = suite.all_items(lengths) if args.tier in ("basic", "all") else []
    if args.tier in ("hard", "all"):
        import hard
        items += hard.hard_items(tuple(int(x) for x in args.needle_hard.split(",") if x))
    if args.tier in ("expert", "all"):
        import expert
        items += expert.expert_items(tuple(int(x) for x in args.needle_expert.split(",") if x),
                                     tuple(int(x) for x in args.needle_expert_count.split(",") if x))
    if args.only:
        cats = set(args.only.split(","))
        items = [it for it in items if it["cat"] in cats or it["id"] in cats]
    if args.limit:
        by = {}
        items = [it for it in items if by.setdefault(it["cat"], []).append(it) or
                 len(by[it["cat"]]) <= args.limit]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "%s-%s.json" % (args.label, time.strftime("%Y%m%d-%H%M%S")))
    meta = {"label": args.label, "url": args.url, "effort": args.effort, "provider": args.provider,
            "temperature": args.temperature, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": len(items)}
    res = []
    t_all = time.time()
    for i, it in enumerate(items):
        body = {"model": args.model, "messages": it["messages"], "temperature": args.temperature,
                "max_tokens": it.get("max_tokens", MAX_TOKENS[it["cat"]]) +
                (0 if args.effort in ("none",) else 0)}
        if args.temperature == 0:
            body["top_k"] = 1
        if it.get("tools"):
            body["tools"] = it["tools"]
        if args.effort and "openrouter.ai" in args.url:     # OpenRouter's form of the effort
            body["reasoning"] = {"effort": args.effort}
        elif args.effort:
            body["reasoning_effort"] = args.effort
        if args.provider:       # OpenRouter: only these providers, in this order, no fallback to others
            body["provider"] = {"order": args.provider.split(","), "allow_fallbacks": False}
        t0 = time.time()
        trace = []
        try:
            if "env" in it:
                msg, finish, usage, trace, score, note = agent(args, it, body)
            else:
                d = post(args.url, args.key, body, args.timeout)
                ch = d["choices"][0]
                msg, finish, usage = ch.get("message", {}), ch.get("finish_reason"), d.get("usage", {})
                usage["provider"] = d.get("provider")
                score, note = it["grade"](msg)
            err = None
        except Exception as e:      # the server failed or timed out: a zero
            msg, finish, usage, score, note, err = {}, "error", {}, 0.0, "error: %r" % e, repr(e)
        dt = time.time() - t0
        fl = faults(msg, finish, args.effort) if err is None else ["error"]
        r = {"id": it["id"], "cat": it["cat"], "score": score, "note": note, "faults": fl,
             "finish": finish, "seconds": round(dt, 2),
             "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
             "content": msg.get("content"), "reasoning_chars": len(reasoning(msg)),
             "tool_calls": msg.get("tool_calls"), "trace": trace, "provider": usage.get("provider")}
        res.append(r)
        print("[%3d/%d] %-26s %s %-40s %s %.0fs" % (i + 1, len(items), it["id"],
              "PASS" if score >= 1 else ("part" if score > 0 else "FAIL"), note[:40],
              ",".join(fl), dt), flush=True)
        with open(path, "w") as f:      # after each item: a run that stops keeps its results
            json.dump({"meta": dict(meta, seconds=round(time.time() - t_all, 1)), "results": res}, f,
                      indent=1)
    print()
    summary(res)
    print("\nthe run: %s" % path)


def agent(args, it, body):
    """An agent task: the model's tool calls go to the item's environment
    (it["env"]()), their results back as tool messages, until an answer with
    no call or AGENT_TURNS turns. Returns the last message, its finish, the
    usage summed over the turns, the trace (name, arguments, result), and
    the grade of it["grade_env"](msg, env, trace)."""
    env = it["env"]()
    msgs = list(it["messages"])
    trace, usage = [], {"prompt_tokens": 0, "completion_tokens": 0}
    msg, finish = {}, None
    for _turn in range(AGENT_TURNS):
        d = post(args.url, args.key, dict(body, messages=msgs), args.timeout)
        ch = d["choices"][0]
        msg, finish = ch.get("message", {}), ch.get("finish_reason")
        u = d.get("usage", {})
        usage["prompt_tokens"] = u.get("prompt_tokens", 0)      # the last turn's whole prompt
        usage["completion_tokens"] += u.get("completion_tokens", 0) or 0
        usage["provider"] = d.get("provider")                   # OpenRouter: who served the turn
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for k, c in enumerate(calls):
            f = c.get("function", {})
            try:
                a = json.loads(f.get("arguments") or "{}")
                out = env.call(f.get("name"), a if isinstance(a, dict) else {})
            except Exception as e:
                a, out = f.get("arguments"), "error: arguments are not a JSON object (%s)" % e
            trace.append([f.get("name"), a, out[:300]])
            msgs.append({"role": "tool", "tool_call_id": c.get("id") or "call_%d" % k, "content": out})
    else:
        score, note = it["grade_env"](msg, env, trace)
        return msg, finish, usage, trace, score, "turn limit; " + note
    score, note = it["grade_env"](msg, env, trace)
    return msg, finish, usage, trace, score, note + " (%d calls)" % len(trace)


def summary(res):
    cats = []
    for r in res:
        if r["cat"] not in cats:
            cats.append(r["cat"])
    tot = 0.0
    print("%-9s %6s %8s %7s %9s  %s" % ("category", "items", "score", "faults", "tokens/it", ""))
    for c in cats:
        rs = [r for r in res if r["cat"] == c]
        s = sum(r["score"] for r in rs)
        tot += s
        nf = sum(1 for r in rs if r["faults"])
        tk = [r["completion_tokens"] for r in rs if r["completion_tokens"]]
        print("%-9s %6d %7.1f%% %7d %9.0f" % (c, len(rs), 100 * s / len(rs), nf,
                                            sum(tk) / max(1, len(tk))))
    print("%-9s %6d %7.1f%%" % ("all", len(res), 100 * tot / max(1, len(res))))
    fl = {}
    for r in res:
        for f in r["faults"]:
            fl[f.split(":")[0]] = fl.get(f.split(":")[0], 0) + 1
    if fl:
        print("faults:", ", ".join("%s %d" % kv for kv in sorted(fl.items())))


def compare(args):
    runs = []
    for p in args.runs:
        for q in sorted(glob.glob(os.path.expanduser(p))):
            runs.append((os.path.basename(q)[:-5], json.load(open(q))["results"]))
    if len(runs) < 2:
        sys.exit("compare: two runs or more")
    cats = []
    for _n, res in runs:
        for r in res:
            if r["cat"] not in cats:
                cats.append(r["cat"])
    w = max(len(n) for n, _ in runs)
    print("%-9s " % "category" + " ".join("%*s" % (max(w, 8), n[:w]) for n, _ in runs))
    for c in cats + ["all"]:
        row = []
        for _n, res in runs:
            rs = [r for r in res if c == "all" or r["cat"] == c]
            row.append("%*.1f%%" % (max(w, 8) - 1, 100 * sum(r["score"] for r in rs) / max(1, len(rs)))
                       if rs else "%*s" % (max(w, 8), "-"))
        print("%-9s " % c + " ".join(row))
    # the tokens of the answers (reasoning and content): the mean of an item
    # of each category, and the sum of all
    tok = lambda r: r.get("completion_tokens") or 0  # noqa: E731
    print("\n%-9s " % "tokens/it" + " ".join("%*s" % (max(w, 8), n[:w]) for n, _ in runs))
    for c in cats + ["all", "total"]:
        row = []
        for _n, res in runs:
            rs = [r for r in res if c in ("all", "total") or r["cat"] == c]
            if not rs:
                row.append("%*s" % (max(w, 8), "-"))
                continue
            v = sum(tok(r) for r in rs) / (1 if c == "total" else len(rs))
            row.append("%*.0f" % (max(w, 8), v))
        print("%-9s " % c + " ".join(row))
    base_n, base = runs[0]
    bid = {r["id"]: r for r in base}
    for n, res in runs[1:]:
        print("\n%s against %s:" % (n, base_n))
        for r in res:
            b = bid.get(r["id"])
            if b is None or (b["score"] >= 1) == (r["score"] >= 1):
                continue
            print("  %-26s %s -> %s  %6d -> %6d tokens  %s" % (
                r["id"], "PASS" if b["score"] >= 1 else "FAIL", "PASS" if r["score"] >= 1 else "FAIL",
                tok(b), tok(r), r["note"][:60]))
        nf = sum(1 for r in res if r["faults"])
        print("  faults: %d (base %d)" % (nf, sum(1 for r in base if r["faults"])))
        # the tokens of the items of both runs, and of those both pass (the
        # same work: the cost of an answer, not of a wrong path)
        both = [(bid[r["id"]], r) for r in res if r["id"] in bid]
        for label, pairs in (("all items", both),
                             ("both pass", [(b, r) for b, r in both if b["score"] >= 1 and r["score"] >= 1])):
            tb, tr = sum(tok(b) for b, _ in pairs), sum(tok(r) for _, r in pairs)
            print("  tokens, %-9s (%3d): %8d (base %8d, %+.1f%%)" % (
                label, len(pairs), tr, tb, 100.0 * (tr - tb) / max(1, tb)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--url", default=os.environ.get("NP_GEMMA_URL", "http://127.0.0.1:8081/v1"))
    a.add_argument("--key", default=os.environ.get("NP_GEMMA_API_KEY"))
    a.add_argument("--model", default="qwen3.8-flash-next")
    a.add_argument("--label", required=True, help="the name of the run (the configuration)")
    a.add_argument("--effort", default=None,
                   help="reasoning_effort for every item (none, low, medium, high); default: the server's")
    a.add_argument("--temperature", type=float, default=0.0, help="0: greedy (repeatable)")
    a.add_argument("--only", default="", help="categories or item ids, comma separated")
    a.add_argument("--limit", type=int, default=0, help="the first N items of each category")
    a.add_argument("--needle", default="4000,16000,64000", help="the lengths of the needle texts")
    a.add_argument("--tier", choices=("basic", "hard", "expert", "all"), default="basic",
                   help="basic: tests/quality/suite.py; hard: hard.py; expert: expert.py; all: the three")
    a.add_argument("--needle-hard", default="32000,128000", help="the lengths of the hard needle texts")
    a.add_argument("--needle-expert", default="64000,128000", help="the lengths of the expert needle texts")
    a.add_argument("--needle-expert-count", default="256000",
                   help="the lengths of the expert needle texts with only the count item")
    a.add_argument("--provider", default=None,
                   help="OpenRouter providers to pin, comma separated (in order, no fallback), e.g. deepinfra")
    a.add_argument("--timeout", type=float, default=1800)
    a.add_argument("--out", default=os.path.expanduser(os.environ.get("NP_GEMMA_QUALITY_DIR",
                                                                       "~/quality-runs")))
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="run files (globs allowed); the first is the base")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
