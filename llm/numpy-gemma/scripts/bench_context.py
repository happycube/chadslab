#!/usr/bin/env python3
"""Measure the cost of a long context. Remove the effect of the machine load.

The script fills the key and value cache with random data for a long context.
It does not run a long prefill. Then it measures a decode step at a short
position and at a long position. The two measurements change in turn. Thus the
machine load is the same for both.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from np_gemma import Config, KVCache, Model, SafeTensors


def resolve_paths(args):
    """Return the config path and the weights path."""
    if args.snapshot:
        s = Path(args.snapshot)
        return (args.config or str(s / "config.json"),
                args.weights or str(s / "model.safetensors"))
    return args.config, args.weights


def fill(cache, cfg, n, seed):
    """Write n random key and value rows into every layer of the cache."""
    rng = np.random.default_rng(seed)
    for i in range(cfg.num_hidden_layers):
        plan = cfg.plan[i]
        k = rng.standard_normal((n, plan.num_kv_heads, plan.head_dim)).astype(np.float32)
        v = rng.standard_normal((n, plan.num_kv_heads, plan.head_dim)).astype(np.float32)
        cache.write(i, 0, k, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--dtype", default="int8")
    ap.add_argument("--short", type=int, default=16)
    ap.add_argument("--long", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()
    config_path, weights_path = resolve_paths(args)

    cfg = Config.load(config_path)
    st = SafeTensors(weights_path)
    model = Model(st, cfg)
    model.load_all(dtype=args.dtype)

    cache_s = KVCache(cfg, max_len=args.short + 4)
    cache_l = KVCache(cfg, max_len=args.long + 4)
    fill(cache_s, cfg, args.short, 1)
    fill(cache_l, cfg, args.long, 2)

    x = np.zeros((1, cfg.hidden_size), dtype=np.float32)
    tok = 50429
    ts, tl = [], []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        model.forward([tok], cache=cache_s, start_pos=args.short)
        ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        model.forward([tok], cache=cache_l, start_pos=args.long)
        tl.append(time.perf_counter() - t0)
    print("short pos %5d: min %.3f  median %.3f s" % (args.short, min(ts), statistics.median(ts)))
    print("long  pos %5d: min %.3f  median %.3f s" % (args.long, min(tl), statistics.median(tl)))
    print("context cost  : %.3f s per token" % (min(tl) - min(ts)))
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
