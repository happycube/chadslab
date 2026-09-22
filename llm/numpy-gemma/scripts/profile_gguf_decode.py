#!/usr/bin/env python3
"""Find the bottleneck in one decode step of a GGUF model.

The script loads a GGUF model, warms the cache with a short prompt, times
some decode steps, and then profiles one step. The hook records the time
between the emit points of the forward pass. Thus the report shows which
stage uses the time.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from collections import defaultdict

import numpy as np

from np_gemma import KVCache, Model
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer


def group_of(key):
    """Return the stage name for one emit key."""
    if "self_attn" in key:
        return "attn"
    if "experts" in key or "router" in key:
        return "moe"
    if "mlp" in key:
        return "dense_mlp"
    if "norm" in key:
        return "norm"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dtype", default="int4")
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype=args.dtype)
    ids = tok.encode("The capital of France is")
    cache = KVCache(cfg, max_len=len(ids) + args.steps + 4)
    t0 = time.perf_counter()
    x = model.prefill(ids, cache)
    print("prefill %d tokens %.3f s" % (len(ids), time.perf_counter() - t0), flush=True)
    nxt = int(np.argmax(model.logits(x[-1:])[0]))

    # Warm up three steps. The page tables and the prefetchers need the warm-up.
    times = []
    for k in range(args.steps + 3):
        t0 = time.perf_counter()
        x = model.forward([nxt], cache=cache, start_pos=len(ids) + k)
        nxt = int(np.argmax(model.logits(x)[0]))
        dt = time.perf_counter() - t0
        if k >= 3:
            times.append(dt)
        print("step %2d %.3f s" % (k, dt), flush=True)
    times.sort()
    print("decode median %.3f s  best %.3f s  %.2f tok/s"
          % (times[len(times) // 2], times[0], 1.0 / times[len(times) // 2]))

    # The stage report for one step.
    last = [None]
    stage = defaultdict(float)
    def hook(key, value):
        now = time.perf_counter()
        if last[0] is not None:
            # The time between two emit points is the work of the stage that
            # emitted the second point.
            stage[group_of(key)] += now - last[0][1]
        last[0] = (key, now)

    cache2 = KVCache(cfg, max_len=len(ids) + 2)
    model.prefill(ids[:1], cache2)
    x = model.forward([nxt], cache=cache2, start_pos=1, hook=hook)
    total = sum(stage.values())
    print("stage report (sum %.1f ms)" % (total * 1000.0))
    for k in sorted(stage, key=lambda s: -stage[s]):
        print("  %-10s %7.1f ms  %4.0f%%" % (k, stage[k] * 1000.0, 100.0 * stage[k] / total))

    # The output head alone. It reads the Q6_K table for each token.
    cache2 = KVCache(cfg, max_len=len(ids) + 2)
    model.prefill(ids[:1], cache2)
    x = model.forward([nxt], cache=cache2, start_pos=1)
    head = []
    for _ in range(3):
        t0 = time.perf_counter()
        model.logits(x)
        head.append(time.perf_counter() - t0)
    print("output head %.1f ms" % (min(head) * 1000.0))

    # The profile of one step.
    cache2 = KVCache(cfg, max_len=len(ids) + 2)
    model.prefill(ids[:1], cache2)
    pr = cProfile.Profile()
    pr.enable()
    x = model.forward([nxt], cache=cache2, start_pos=1)
    model.logits(x)
    pr.disable()
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats("tottime").print_stats(22)
    print(out.getvalue())


if __name__ == "__main__":
    main()
