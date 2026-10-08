#!/usr/bin/env python3
"""Measure the decode rate of the 26B the way llama-bench measures tg128.

The script starts from a cache that holds one token, and it runs --tokens
decode steps with the output head and a greedy choice. The first step is a
warm-up and is not in the time. --gpu gives the place of the weights, as in
scripts/serve.py.

    PYTHONPATH=. python scripts/bench_decode.py --gpu dense
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from np_gemma import KVCache, Model
from np_gemma.config import Config
from np_gemma.gguf import GGUF

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--gpu", choices=("off", "dense", "hot"), default="off")
    ap.add_argument("--gpu-experts-gb", type=float, default=None)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    g = GGUF(args.gguf)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    if args.gpu != "off":
        from np_gemma import gpu
        dev = gpu.offload(model, 0.0 if args.gpu == "dense" else args.gpu_experts_gb)
        print(gpu.describe(dev))
    rates = []
    for _ in range(args.repeat):
        cache = KVCache(cfg, max_len=args.tokens + 8)
        x = model.prefill([2], cache)
        nxt = int(np.argmax(model.logits(x[-1:])[0]))
        x = model.forward([nxt], cache=cache, start_pos=1)
        nxt = int(np.argmax(model.logits(x)[0]))
        t0 = time.perf_counter()
        for pos in range(2, args.tokens + 2):
            x = model.forward([nxt], cache=cache, start_pos=pos)
            nxt = int(np.argmax(model.logits(x)[0]))
        rates.append(args.tokens / (time.perf_counter() - t0))
    print("gpu=%s tg%d: %.2f tokens/s (runs: %s)" % (
        args.gpu, args.tokens, np.mean(rates), " ".join("%.2f" % r for r in rates)))


if __name__ == "__main__":
    raise SystemExit(main())
