#!/usr/bin/env python3
"""Run decode steps of the 26B at a long context on the CPU and on the GPU.

SPLIT_PLAN.md, the cache on the GPU. A prompt pass of 64k tokens takes a
long time on the CPU.

Thus the script builds the cache. It runs a real
prompt pass of --real tokens. Then it writes the same key and value rows
again and again up to --context positions. The rows past --real are not the rows
of a real text, but they have the size and the form of a real cache. The CPU
and the GPU read the same cache, so their results can be compared.

The script runs --steps decode steps with the CPU program (the int16 cache
of the CPU), and with the GPU for each form of --kv. It prints these values:

- the time of a step with the output head;
- the size of the cache on the GPU;
- the largest difference of the logits from the CPU;
- the share of the steps with the same top token.

    OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=ACTIVE PYTHONPATH=. \\
        python scripts/check_gpu_long.py --context 65536
"""
from __future__ import annotations

import argparse
import copy
import gc
import time

import numpy as np

from np_gemma import KVCache, Model, gpu, program
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def build_cache(model, cfg, ids, real, context):
    """Return a KVCache of context positions: a real prompt pass of real
    tokens, then copies of its rows."""
    cache = KVCache(cfg, max_len=context + 64)
    model.prefill(ids[:real], cache)
    for i in range(cfg.num_hidden_layers):
        k, v, _ = cache.read(i, real)          # dequantized rows (new arrays)
        pos = cache.end[i]
        while pos < context:
            n = min(len(k), context - pos)
            cache.write(i, pos, k[:n], v[:n])
            pos += n
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--real", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--kv", nargs="+", default=["int16", "float"])
    ap.add_argument("--hot-gb", type=float, default=0.0,
                    help="The GPU memory for the hot experts. 0 keeps the experts on the CPU.")
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    ids = tok.encode(open("README.md").read())
    t0 = time.perf_counter()
    base = build_cache(model, cfg, ids, args.real, args.context)
    print("cache of %d positions (%d real) in %.0f s" % (
        args.context, args.real, time.perf_counter() - t0), flush=True)
    toks = ids[args.real:args.real + args.steps]

    # The CPU program.
    cache = copy.deepcopy(base)
    ref, t_cpu = [], []
    for k, t in enumerate(toks):
        t0 = time.perf_counter()
        ref.append(model.logits(program.decode_step(model, cache, [t], args.context + k))[0])
        t_cpu.append(time.perf_counter() - t0)
    print("CPU program: %.1f ms for each step (median)" % (1000 * np.median(t_cpu)), flush=True)
    del cache
    hot = None
    if args.hot_gb > 0:
        hot = gpu.pick_hot(model, gpu.hot_counts(model), args.hot_gb * 1e9)
    ok = True
    for kv in args.kv:
        free0 = gpu.mem_info()[0]
        dev = gpu.ModelGPU(model, hot=hot or {}, kv=kv)
        dev.logits()
        cache = copy.deepcopy(base)
        dev.attach(cache)
        dl, top, ts = [], [], []
        for k, t in enumerate(toks):
            t0 = time.perf_counter()
            dev.step([t], args.context + k)
            lg = dev.logits()[0]
            ts.append(time.perf_counter() - t0)
            dl.append(float(np.abs(lg - ref[k]).max()))
            top.append(int(lg.argmax()) == int(ref[k].argmax()))
        free1 = gpu.mem_info()[0]
        print("GPU, %s cache: %.1f ms for each step (median), cache %.2f GB, all the GPU "
              "memory of the model %.2f GB, logits max |d| %.3f, same top token %d/%d" % (
                  kv, 1000 * np.median(ts), dev.kv.nbytes() / 1e9, (free0 - free1) / 1e9,
                  max(dl), sum(top), len(top)), flush=True)
        ok = ok and sum(top) >= len(top) - 1
        dev.g.close()
        del dev, cache
        gc.collect()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
