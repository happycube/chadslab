#!/usr/bin/env python3
"""Profile one decode step with a long context. Fill the cache with random data."""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
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
    args = ap.parse_args()
    config_path, weights_path = resolve_paths(args)

    cfg = Config.load(config_path)
    print("sliding_window:", cfg.sliding_window)
    st = SafeTensors(weights_path)
    model = Model(st, cfg)
    model.load_all(dtype=args.dtype)
    cache = KVCache(cfg, max_len=args.long + 4)
    fill(cache, cfg, args.long)
    plan0 = cfg.plan[0]
    print("layer0: head_dim %d q_heads %d kv_heads %d sliding %s" % (
        plan0.head_dim, plan0.num_q_heads, plan0.num_kv_heads, plan0.is_sliding))
    plang = cfg.plan[5]
    print("layer5: head_dim %d q_heads %d kv_heads %d sliding %s" % (
        plang.head_dim, plang.num_q_heads, plang.num_kv_heads, plang.is_sliding))

    model.forward([50429], cache=cache, start_pos=args.long)
    pr = cProfile.Profile()
    pr.enable()
    model.forward([50429], cache=cache, start_pos=args.long)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
    print(s.getvalue())
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
