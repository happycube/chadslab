#!/usr/bin/env python3
"""The rate of Qwen3.6 (GGUF) on the GPU for some budgets of hot experts.

For each budget (GB): the decode of --tokens tokens from a new HotCache,
then again with the warm HotCache. Then a prompt pass of --prompt tokens. A
budget of 0 keeps one slot in each layer (the programs need one).

    OMP_NUM_THREADS=18 python scripts/bench_qwen_hot.py --budgets 0,0.5,1,1.5,2
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import ops  # noqa: E402
from np_gemma.qwen import QwenCache, QwenGGUFProgram  # noqa: E402
from np_gemma.qwen_gpu import QwenGPU  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TOK = "models/Qwen3.6-35B-A3B-OptiQ-4bit/tokenizer.json"
PROMPT = ("<|im_start|>user\nWrite a short Python function that checks if a number is prime, "
          "and explain how it works.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def decode(g, m, tok, ids, n):
    cache = QwenCache(m.cfg, len(ids) + n + 64)
    g.attach(cache)
    h = g.prefill(ids)
    nxt = ops.argmax(g.logits())
    pos, ts = len(ids), []
    for _ in range(n):
        t = time.perf_counter()
        g.step(nxt, pos)
        nxt = ops.argmax(g.logits())
        ts.append(time.perf_counter() - t)
        pos += 1
    g.detach(cache)
    return n / sum(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budgets", default="0,0.5,1,1.5,2")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompt", type=int, default=1500)
    args = ap.parse_args()
    tok = QwenTokenizer(TOK)
    m = QwenGGUFProgram(PATH)
    ids = tok.encode(PROMPT)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    long_ids = tok.encode(open(os.path.join(root, "QWEN_PLAN.md")).read() +
                          open(os.path.join(root, "README.md")).read())[:args.prompt]
    print("budget  slots  decode new  decode warm  cold/layer  prompt of %d" % len(long_ids))
    for b in [float(x) for x in args.budgets.split(",")]:
        g = QwenGPU(m, hot_gb=b)
        r1 = decode(g, m, tok, ids, args.tokens)
        hc = g.hot_cache
        r2 = decode(g, m, tok, ids, args.tokens)
        cold = hc.cold / max(1, hc.steps) / len(hc.layers)
        cache = QwenCache(m.cfg, len(long_ids) + 1100)
        g.attach(cache)
        g.prefill(long_ids)             # builds the programs
        g.detach(cache)
        cache = QwenCache(m.cfg, len(long_ids) + 1100)
        g.attach(cache)
        t0 = time.time()
        g.prefill(long_ids)
        g.logits()
        tp = time.time() - t0
        g.detach(cache)
        print("%4.1f GB  %5d  %7.1f tok/s  %7.1f tok/s  %8.2f  %7.0f tok/s" % (
            b, g.n_slots, r1, r2, cold, len(long_ids) / tp), flush=True)
        g.close()
        del g, hc, cache
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
