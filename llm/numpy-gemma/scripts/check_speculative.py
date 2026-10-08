#!/usr/bin/env python3
"""Check the shared MTP loop (np_gemma.speculative) with a sampler on Qwen3.8.

1. The plain decode with a seeded Sampler (temperature, top_k, top_p).
2. MTP (generate_mtp_gpu) with the same Sampler and mtp_accept "exact":
   the tokens must be those of 1 (the sampler runs once for each emitted
   token, in the same order).
3. MTP with mtp_accept "in_set": drafts that the settings allow stay too;
   the text may differ, so only the acceptance is reported.

    GF=model.gguf python scripts/check_speculative.py [--tokens 96] [--draft 2]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import np_gemma  # noqa: E402,F401
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--draft", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU
    from np_gemma.qwen4_gpu import Qwen4GPU, generate_mtp_gpu
    from np_gemma.qwen_tok import QwenTokenizer
    from np_gemma.sampling import Sampler
    gf = os.environ["GF"]
    tok = QwenTokenizer(os.path.join(os.path.dirname(gf), "tokenizer.json"))
    ids = tok.encode("<|im_start|>user\nWrite a short story about a lighthouse keeper.<|im_end|>\n"
                     "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    m = Qwen4CPU(gf)
    g = Qwen4GPU(m, hot_gb=1)
    # the hot experts stay (a hot expert gives values a little different from
    # a cold one, so another split could pick another token)
    hc, g.hot_cache = g.hot_cache, None

    def sampler(accept):
        s = Sampler(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                    seed=args.seed, mtp_accept=accept)
        s.reset(ids)
        return s

    ctx = len(ids) + args.tokens + 64
    # 1. plain
    s = sampler("exact")
    c = Qwen4Cache(m.cfg, ctx)
    g.attach(c)
    g.prefill(ids)
    plain, pos = [], len(ids)
    t = int(s(np.asarray(g.logits()).reshape(-1)))
    while len(plain) < args.tokens:
        plain.append(t)
        g.step(t, pos)
        pos += 1
        t = int(s(np.asarray(g.logits()).reshape(-1)))
    ok = True
    for accept in ("exact", "in_set"):
        c = Qwen4Cache(m.cfg, ctx)
        g.attach(c)
        st = {}
        got = generate_mtp_gpu(g, ids, args.tokens, draft=args.draft, stats=st, pick=sampler(accept))
        same = next((i for i, (a, b) in enumerate(zip(got, plain)) if a != b), min(len(got), len(plain)))
        print("MTP %d drafts, %-6s: %d tokens, %d/%d drafts kept (%.0f%%; %d by the in_set rule); "
              "the same tokens as the plain decode: the first %d of %d" % (
                  args.draft, accept, len(got), st["accepted"], st["drafted"],
                  100 * st["accepted"] / max(1, st["drafted"]), st["in_set"],
                  same, len(plain)))
        if accept == "exact":
            ok &= same == len(plain)
    g.hot_cache = hc
    g.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
