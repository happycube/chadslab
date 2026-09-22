#!/usr/bin/env python3
"""Time the real attention of one layer at a long context."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from np_gemma import Config, KVCache, Model, SafeTensors


def resolve_paths(args):
    if args.snapshot:
        s = Path(args.snapshot)
        return (args.config or str(s / "config.json"),
                args.weights or str(s / "model.safetensors"))
    return args.config, args.weights


def fill(cache, cfg, n, seed=1):
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
    ap.add_argument("--long", type=int, default=1024)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 5, 6])
    args = ap.parse_args()
    config_path, weights_path = resolve_paths(args)

    cfg = Config.load(config_path)
    st = SafeTensors(weights_path)
    model = Model(st, cfg)
    model.load_all(dtype=args.dtype)
    cache = KVCache(cfg, max_len=args.long + 4)
    fill(cache, cfg, args.long)
    x = np.random.default_rng(0).standard_normal((1, cfg.hidden_size)).astype(np.float32)

    for lay in args.layers:
        plan = cfg.plan[lay]
        w = model._layers[lay]
        pos = np.array([args.long], dtype=np.int64)
        cos, sin = model._rope(plan, pos)
        model._attention(x, w, plan, cos, sin, pos, lay, "", None, cache)
        best = 1e9
        for _ in range(6):
            t0 = time.perf_counter()
            model._attention(x, w, plan, cos, sin, pos, lay, "", None, cache)
            best = min(best, time.perf_counter() - t0)
        print("layer %2d sliding=%-5s hd=%3d kv=%d q=%d  %.3f s" % (
            lay, plan.is_sliding, plan.head_dim, plan.num_kv_heads, plan.num_q_heads, best))
    # Estimate the full-model attention cost.
    print()
    for n in (16, 128, 512, 1024):
        cache2 = KVCache(cfg, max_len=n + 4)
        fill(cache2, cfg, n)
        tot = 0.0
        for lay in range(cfg.num_hidden_layers):
            plan = cfg.plan[lay]
            pos = np.array([n], dtype=np.int64)
            cos, sin = model._rope(plan, pos)
            t0 = time.perf_counter()
            model._attention(x, model._layers[lay], plan, cos, sin, pos, lay, "", None, cache2)
            tot += time.perf_counter() - t0
        print("all 48 layers of attention at n=%4d: %.3f s" % (n, tot))
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
