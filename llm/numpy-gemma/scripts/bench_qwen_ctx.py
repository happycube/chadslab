#!/usr/bin/env python3
"""The decode rate of Qwen3.6 (GGUF) on the GPU at long positions.

The step at position p reads the keys and values of p positions. The rate
does not depend on the values of the cache, so the cache is not filled.
Each step runs at a position far into it.

    NP_GEMMA_QWEN_KV=int16 python scripts/bench_qwen_ctx.py --positions 1000,32000,131000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.gpu import mem_info  # noqa: E402
from np_gemma.qwen import QwenCache, QwenGGUFProgram  # noqa: E402
from np_gemma.qwen_gpu import QwenGPU  # noqa: E402

PATH = "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--positions", default="1000,32000,131000")
    ap.add_argument("--hot-gb", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=32)
    args = ap.parse_args()
    m = QwenGGUFProgram(PATH)
    pos = [int(x) for x in args.positions.split(",")]
    g = QwenGPU(m, hot_gb=args.hot_gb)
    cache = QwenCache(m.cfg, max(pos) + args.steps + 8)
    kv = sum(a.nbytes for v in cache.kv.values() for a in v)
    f0 = mem_info()[0]
    g.attach(cache)
    print("kv form %s: the cache of %d positions takes %.2f GB (GPU free %.2f -> %.2f GB)" % (
        m.cfg.kv_form, cache.max_len, kv / 1e9, f0 / 1e9, mem_info()[0] / 1e9))
    for p in pos:
        ts = []
        for s in range(args.steps):
            t = time.perf_counter()
            g.step(1000 + s, p + s)
            g.logits()
            ts.append(time.perf_counter() - t)
        print("position %7d: %.1f ms for each token, %.1f tok/s" % (
            p, 1e3 * np.median(ts[4:]), 1 / np.median(ts[4:])), flush=True)
    g.detach(cache)
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
