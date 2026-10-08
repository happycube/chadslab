#!/usr/bin/env python3
"""Measure the cost of a verify batch of the 26B target.

An MTP step runs the target on the last token and the drafts in one batch.
This script gives the time of a batch of 1, 2, 4, and 7 tokens after a
context of --context tokens. It also gives the count of distinct experts that
the batch selects in each layer, because the expert reads grow with the
batch.

Run from the numpy-gemma directory:

    PYTHONPATH=. python scripts/bench_verify.py
    NP_GEMMA_INT4_Q8=0 PYTHONPATH=. python scripts/bench_verify.py

The second form uses the float kernels for a small batch.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from np_gemma import KVCache, Model
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 7])
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")

    text = open("README.md").read()
    ids = tok.encode(text)[:args.context + max(args.sizes)]
    ctx = ids[:args.context]
    cache = KVCache(cfg, max_len=len(ids) + 8)
    model.prefill(ctx, cache)

    print("INT4_Q8=%s  context=%d" % (os.environ.get("NP_GEMMA_INT4_Q8", "1"), args.context))
    print("%6s %10s %10s %10s %12s" % ("tokens", "fwd ms", "head ms", "ms/token", "experts"))
    base = None
    for t in args.sizes:
        batch = ids[args.context:args.context + t]
        fwd, head = [], []
        for _ in range(args.reps):
            cache.truncate(args.context)
            t0 = time.perf_counter()
            x = model.forward(batch, cache=cache, start_pos=args.context)
            t1 = time.perf_counter()
            model.logits(x)
            t2 = time.perf_counter()
            fwd.append(t1 - t0)
            head.append(t2 - t1)
        # One more run with a hook to count the distinct experts of each layer.
        union = []

        def hook(key, value):
            if key.endswith("router.top_idx"):
                union.append(len(np.unique(value)))
        cache.truncate(args.context)
        model.forward(batch, cache=cache, start_pos=args.context, hook=hook)
        f = 1000 * float(np.median(fwd))
        h = 1000 * float(np.median(head))
        if base is None:
            base = f + h
        print("%6d %10.1f %10.1f %10.1f %8.1f/layer   %.2fx one token" % (
            t, f, h, (f + h) / t, float(np.mean(union)), (f + h) / base))
    g.close()


if __name__ == "__main__":
    main()
