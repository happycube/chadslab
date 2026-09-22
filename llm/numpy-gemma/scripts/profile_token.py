#!/usr/bin/env python3
"""Find the bottlenecks in one decode step.

The script does five tasks:
1. Measure the memory bandwidth of the machine.
2. Load the model with bfloat16 weights and the C kernel.
3. Warm the key and value cache with one prefill.
4. Time the forward pass and the output head for some decode steps.
5. Profile one decode step and print the top functions.

Weights are read from memory for each token. Thus the bandwidth gives a lower
bound for the time of one token.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np

from np_gemma import Config, KVCache, Model, SafeTensors, Tokenizer


def resolve_paths(args):
    """Return the config path, the weights path, and the tokenizer path."""
    if args.snapshot:
        s = Path(args.snapshot)
        return (args.config or str(s / "config.json"),
                args.weights or str(s / "model.safetensors"),
                args.tokenizer or str(s / "tokenizer.json"))
    if not (args.config and args.weights and args.tokenizer):
        raise SystemExit("provide --snapshot, or all of --config/--weights/--tokenizer")
    return args.config, args.weights, args.tokenizer


def measure_bandwidth(mb=512):
    """Return the memory bandwidth in GB per second. Use one copy operation."""
    n = mb * 1024 * 1024 // 4
    a = np.ones(n, dtype=np.float32)
    b = np.empty_like(a)
    a.sum()  # page in
    t0 = time.perf_counter()
    b[:] = a
    dt = time.perf_counter() - t0
    return (a.nbytes + b.nbytes) / dt / 1e9


def weight_bytes(cfg):
    """Return the number of bytes in the projection weights and the embedding table."""
    total = 0
    for plan in cfg.plan:
        q = plan.q_dim * cfg.hidden_size
        kv = plan.kv_dim * cfg.hidden_size
        o = cfg.hidden_size * plan.q_dim
        up = cfg.intermediate_size * cfg.hidden_size
        down = cfg.hidden_size * cfg.intermediate_size
        total += q + o + up + down + (kv if plan.k_eq_v else 2 * kv)
    total += cfg.vocab_size * cfg.hidden_size
    return total * 2  # bfloat16


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default="Count from 1 to 10, separated by commas.")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()
    config_path, weights_path, tok_path = resolve_paths(args)

    bandwidth = measure_bandwidth()
    print("memory bandwidth: %.1f GB/s" % bandwidth)

    tokenizer = Tokenizer(tok_path)
    cfg = Config.load(config_path)
    wb = weight_bytes(cfg)
    print("weights read for each token: %.1f GB (bfloat16)" % (wb / 1e9))

    with SafeTensors(weights_path) as st:
        model = Model(st, cfg)
        t0 = time.perf_counter()
        model.load_all(dtype=args.dtype)
        print("load_all: %.1f s" % (time.perf_counter() - t0))

        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": args.prompt}],
            add_generation_prompt=True, thinking=False)
        ids = tokenizer.encode(text)
        cache = KVCache(cfg, max_len=len(ids) + args.steps + 8)
        t0 = time.perf_counter()
        x = model.forward(ids, cache=cache)
        print("prefill (%d tokens): %.1f s" % (len(ids), time.perf_counter() - t0))

        t0 = time.perf_counter()
        logits = model.logits(x[-1:])
        t_logits = time.perf_counter() - t0
        tok = int(np.argmax(logits[0]))
        print("logits: %.3f s" % t_logits)

        fwd_times, log_times = [], []
        for i in range(args.steps):
            pos = len(ids) + i
            t0 = time.perf_counter()
            x = model.forward([tok], cache=cache, start_pos=pos)
            t_fwd = time.perf_counter() - t0
            t0 = time.perf_counter()
            logits = model.logits(x)
            t_log = time.perf_counter() - t0
            tok = int(np.argmax(logits[0]))
            fwd_times.append(t_fwd)
            log_times.append(t_log)
        print("decode forward: %.3f s per token" % (sum(fwd_times) / len(fwd_times)))
        print("decode logits : %.3f s per token" % (sum(log_times) / len(log_times)))
        total = sum(fwd_times) / len(fwd_times) + sum(log_times) / len(log_times)
        print("decode total  : %.3f s per token" % total)
        print("effective bandwidth: %.1f GB/s" % ((wb / 1e9) / total))

        print()
        print("profile of one decode step:")
        pr = cProfile.Profile()
        pr.enable()
        x = model.forward([tok], cache=cache, start_pos=len(ids) + args.steps)
        logits = model.logits(x)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(16)
        for line in s.getvalue().splitlines():
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
