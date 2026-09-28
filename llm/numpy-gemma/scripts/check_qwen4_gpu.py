#!/usr/bin/env python3
"""Check Qwen3.8-Flash-Next on the GPU (np_gemma.qwen4_gpu.Qwen4GPU).

QWEN38_PLAN.md, phase 5. The GPU products read x in float32, so the values
are close to those of the CPU program, not the same. The script:

1. runs a chat prompt and a greedy decode on the GPU and on the CPU
   program: the logits of the prompt, and the tokens;
2. runs a verify group and commit(2) on the GPU against steps;
3. runs the greedy decode with MTP drafts (the MTP layer on the CPU).

    python scripts/check_qwen4_gpu.py --hot-gb 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU  # noqa: E402
from np_gemma.qwen4_gpu import Qwen4GPU, generate_mtp_gpu  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = ("models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/"
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
MTP = "models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
TOK = "models/Qwen3.6-35B-A3B-OptiQ-4bit/tokenizer.json"
PROMPT = ("<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--mtp", default=MTP)
    ap.add_argument("--tok", default=TOK)
    ap.add_argument("--hot-gb", type=float, default=1.0)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--draft", type=int, default=3)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU reference")
    args = ap.parse_args()
    tok = QwenTokenizer(args.tok)
    stop = set(tok.stop_ids)
    ids = tok.encode(args.prompt)
    max_len = len(ids) + args.tokens + 64
    m = Qwen4CPU(args.path, mtp=args.mtp)
    ok = True

    ref_tokens = None
    if not args.no_cpu:
        cache = Qwen4Cache(m.cfg, max_len)
        ref0 = m.logits(m.forward(ids, cache)[-1:])[0]
        nxt, pos, ref_tokens = int(np.argmax(ref0)), len(ids), []
        ref_tokens.append(nxt)
        while len(ref_tokens) < args.tokens and nxt not in stop:
            nxt = int(np.argmax(m.logits(m.forward([nxt], cache, start_pos=pos))[0]))
            ref_tokens.append(nxt)
            pos += 1

    t0 = time.time()
    g = Qwen4GPU(m, hot_gb=args.hot_gb)
    print("Qwen4GPU: %d hot experts in each layer, %.1f s" % (g.n_slots, time.time() - t0))
    cache = Qwen4Cache(m.cfg, max_len)
    g.attach(cache)
    t0 = time.time()
    g.prefill(ids)
    lg = g.logits()
    t_prompt = time.time() - t0
    if ref_tokens is not None:
        print("prompt of %d tokens: logits max rel %.2e against the CPU" % (len(ids), rel(lg, ref0)))
    nxt, pos, out, ts = int(np.argmax(lg)), len(ids), [], []
    out.append(nxt)
    while len(out) < args.tokens and nxt not in stop:
        t1 = time.time()
        g.step(nxt, pos)
        nxt = int(np.argmax(g.logits()))
        ts.append(time.time() - t1)
        out.append(nxt)
        pos += 1
    print("prompt %.2f s; decode %.1f ms for each token (median), %.2f tok/s" % (
        t_prompt, 1e3 * np.median(ts), 1 / np.median(ts)))
    print(repr(tok.decode(out)))
    if ref_tokens is not None:
        same = next((i for i, (a, b) in enumerate(zip(out, ref_tokens)) if a != b),
                    min(len(out), len(ref_tokens)))
        print("the same tokens as the CPU: the first %d of %d" % (same, len(ref_tokens)))
        ok &= same >= min(8, len(ref_tokens))

    # 2. A verify group of 4 and commit(2), then a step, against steps. The
    # hot experts stay (HotCache off): the split of the experts between the
    # GPU and the CPU changes the values.
    hc, g.hot_cache = g.hot_cache, None
    n = len(ids)
    sub = ids[:n - 5]
    for label in ("steps", "verify"):
        c = Qwen4Cache(m.cfg, max_len)
        g.attach(c)
        g.prefill(sub)
        if label == "steps":
            rows = []
            for p in range(n - 5, n):
                g.step(ids[p], p)
                rows.append(g.logits())
            steps = np.stack(rows)
        else:
            g.verify(ids[n - 5:n - 1], n - 5)
            v = g.logits(rows=4)
            g.commit(2)
            after = []
            for p in range(n - 3, n):
                g.step(ids[p], p)
                after.append(g.logits())
            r1, r2 = rel(v[:2], steps[:2]), rel(np.stack(after), steps[2:])
            print("verify group of 4 against steps: max rel %.2e; after commit(2): %.2e" % (r1, r2))
            ok &= r1 < 1e-3 and r2 < 1e-3

    g.hot_cache = hc

    # 3. MTP: the MTP layer on the GPU, and on the CPU.
    for where in ("GPU", "CPU"):
        c = Qwen4Cache(m.cfg, max_len)
        g.attach(c)
        st = {}
        got = generate_mtp_gpu(g, ids, len(out), draft=args.draft, stop=stop, stats=st,
                               mtp_cpu=where == "CPU")
        print("MTP layer on the %s (%d drafts): %d tokens, %d/%d drafts accepted (%.0f%%), "
              "%.2f tokens a round" % (
                  where, args.draft, len(got), st["accepted"], st["drafted"],
                  100 * st["accepted"] / max(1, st["drafted"]), len(got) / max(1, st["rounds"])))
        print("  decode: %.2f tok/s; for each round: draft %.0f ms, verify %.0f ms" % (
            (len(got) - 1) / st["decode_s"],
            *(1e3 * st[k] / max(1, st["rounds"]) for k in ("draft", "verify"))))
        same = next((i for i, (a, b) in enumerate(zip(got, out)) if a != b),
                    min(len(got), len(out)))
        print("  the same tokens as the GPU decode: the first %d of %d" % (same, len(out)))
    g.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    np.seterr(over="ignore")
    raise SystemExit(main())
