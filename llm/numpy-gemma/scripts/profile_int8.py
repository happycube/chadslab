#!/usr/bin/env python3
"""Show the time and the bandwidth of each int8 matrix in one decode step.

The script does four tasks:
1. Load the model in the int8 mode.
2. Time one multiply for each matrix with one token as input.
3. Group the results by the projection name.
4. Print the bytes, the time, and the bandwidth of each group.

The output shows which matrices are slow. A matrix below the memory bandwidth
of the machine has room for improvement.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from np_gemma import Config, Model, SafeTensors
from np_gemma import ops
from np_gemma.model import _PROJ_KEYS

try:
    from np_gemma import cops
except Exception:
    cops = None


def resolve_paths(args):
    """Return the config path and the weights path."""
    if args.snapshot:
        snap = Path(args.snapshot)
        return (args.config or str(snap / "config.json"),
                args.weights or str(snap / "model.safetensors"))
    if not (args.config and args.weights):
        raise SystemExit("provide --snapshot, or both --config and --weights")
    return args.config, args.weights


def best_time(fn, repeats):
    """Return the best time of one call. Call the function one time first."""
    fn()
    out = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    config_path, weights_path = resolve_paths(args)

    cfg = Config.load(config_path)
    st = SafeTensors(weights_path)
    model = Model(st, cfg)
    t0 = time.perf_counter()
    model.load_all(dtype="int8")
    print("load_all: %.1f s" % (time.perf_counter() - t0))
    print("C kernel: %s | AVX-512: %s" % (
        cops is not None and cops.available(),
        getattr(cops, "AVX512", None) if cops is not None else None))
    print()

    x = np.random.default_rng(0).standard_normal((1, cfg.hidden_size)).astype(np.float32)

    # key -> [bytes, seconds, count]
    agg = defaultdict(lambda: [0, 0.0, 0])
    shapes = {}
    for i in sorted(model._layers):
        w = model._layers[i]
        for key in list(_PROJ_KEYS) + ["self_attn.v_proj"]:
            item = w.get(key)
            if not isinstance(item, tuple):
                continue
            q, s = item
            group = ops.int8_group(q, s)
            nbytes = q.nbytes + s.nbytes

            def run(q=q, s=s, group=group):
                if cops is not None and cops.available():
                    return cops.linear_int8_float(x, q, s, group)
                return ops.linear_int8_numpy(x, q, s)

            dt = best_time(run, args.repeats)
            a = agg[key]
            a[0] += nbytes
            a[1] += dt
            a[2] += 1
            shapes[key] = (int(q.shape[0]), int(q.shape[1]))

    # The output head. The model uses the int8 embedding table.
    if model._embed_q is not None:
        q, s = model._embed_q, model._embed_s
        group = ops.int8_group(q, s)

        def run_embed():
            return cops.linear_int8_float(x, q, s, group)

        dt = best_time(run_embed, args.repeats)
        agg["embed_tokens (logits)"] = [q.nbytes + s.nbytes, dt, 1]
        shapes["embed_tokens (logits)"] = (int(q.shape[0]), int(q.shape[1]))

    print("%-26s %4s %-14s %8s %9s %8s" % ("projection", "n", "shape", "MB", "ms/tok", "GB/s"))
    total_b = total_t = 0
    for key, (nbytes, dt, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        r, c = shapes[key]
        print("%-26s %4d %-14s %8.1f %9.2f %8.1f" % (
            key, n, "%dx%d" % (r, c), nbytes / 1e6, dt * 1e3,
            nbytes / max(dt, 1e-12) / 1e9))
        total_b += nbytes
        total_t += dt

    print("-" * 78)
    print("%-26s %4s %-14s %8.1f %9.2f %8.1f" % (
        "TOTAL", "", "", total_b / 1e6, total_t * 1e3, total_b / total_t / 1e9))

    # Compare the sum of the matrices with the real forward pass.
    ids = [2, 105, 9731, 107, 3048, 659, 496, 11045, 16326, 236761, 106, 107,
           105, 2364, 107, 4377, 699, 236743, 236770, 531, 236743, 236770,
           236771, 236764, 15914, 684, 162760, 236761, 106, 107, 105, 4368,
           107, 100, 45518, 107, 101]
    from np_gemma import KVCache
    cache = KVCache(cfg, max_len=len(ids) + 8)
    model.forward(ids, cache=cache)
    tok = 50429
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        model.forward([tok], cache=cache, start_pos=len(ids))
        ts.append(time.perf_counter() - t0)
    print()
    print("real decode forward: %.3f s per token (best of %d)" % (min(ts), args.repeats))

    # Split one real pass into kernel time and other work.
    orig = ops.linear_int8
    state = {"t": 0.0, "n": 0}

    def timed(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        state["t"] += time.perf_counter() - t0
        state["n"] += 1
        return r

    ops.linear_int8 = timed
    t0 = time.perf_counter()
    model.forward([tok], cache=cache, start_pos=len(ids))
    total = time.perf_counter() - t0
    ops.linear_int8 = orig
    print("one pass: %.3f s total, %.3f s in %d int8 calls, %.3f s other" % (
        total, state["t"], state["n"], total - state["t"]))
    print("kernel rate: %.1f GB/s" % (total_b / max(state["t"], 1e-12) / 1e9))
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
