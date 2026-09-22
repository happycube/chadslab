#!/usr/bin/env python3
"""Time each decode step in the int8 mode. Show the warm-up and the spread.

The first decode steps can be slower than the later steps. The system page
tables and the memory prefetchers need some steps to become warm. This script
prints one line for each step.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from np_gemma import Config, KVCache, Model, SafeTensors, Tokenizer


def resolve_paths(args):
    """Return the config path, the weights path, and the tokenizer path."""
    if args.snapshot:
        s = Path(args.snapshot)
        return (args.config or str(s / "config.json"),
                args.weights or str(s / "model.safetensors"),
                args.tokenizer or str(s / "tokenizer.json"))
    return args.config, args.weights, args.tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dtype", default="int8")
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    config_path, weights_path, tok_path = resolve_paths(args)

    tok = Tokenizer(tok_path)
    cfg = Config.load(config_path)
    st = SafeTensors(weights_path)
    model = Model(st, cfg)
    model.load_all(dtype=args.dtype)
    text = tok.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Count from 1 to 10, separated by commas."}],
        add_generation_prompt=True, thinking=False)
    ids = tok.encode(text)
    cache = KVCache(cfg, max_len=len(ids) + args.steps + 4)
    t0 = time.perf_counter()
    x = model.forward(ids, cache=cache)
    print("prefill (%d tokens): %.2f s" % (len(ids), time.perf_counter() - t0))
    nxt = int(model.logits(x[-1:]).argmax())

    times = []
    for k in range(args.steps):
        t0 = time.perf_counter()
        x = model.forward([nxt], cache=cache, start_pos=len(ids) + k)
        dt = time.perf_counter() - t0
        times.append(dt)
        nxt = int(model.logits(x).argmax())
        print("step %2d: %.3f s" % (k, dt), flush=True)
    print("min %.3f  median %.3f  mean %.3f  max %.3f" % (
        min(times), statistics.median(times), statistics.mean(times), max(times)))
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
