"""Measure a prompt GEMM against the token count.

The int4 kernel decodes each weight once and uses it for a block of tokens. The
decode is fixed work, so more tokens per weight make it cheaper. The MoE layer
gives each expert a small group of tokens, so this curve says how much the
prefill schedule costs.

Run:  PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/bench_gemm_tokens.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from np_gemma import ops

TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def best(fn, reps=3):
    fn()
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


def main():
    rng = np.random.default_rng(0)
    shapes = ((2816, 2816, "dense 2816x2816"),
              (1408, 2816, "moe gate+up 1408x2816"))
    for rows, cols, label in shapes:
        w = rng.standard_normal((rows, cols)).astype(np.float32)
        packed, scales4 = ops.quantize_int4(w)
        q8, s8 = ops.quantize_int8(w)
        w16 = (w.view(np.uint32) >> 16).astype(np.uint16)
        print("== %s" % label)
        print("   %-7s %10s %10s %10s" % ("tokens", "int4", "int8", "bf16"))
        for t in TOKENS:
            x = rng.standard_normal((t, cols)).astype(np.float32)
            fl = 2.0 * rows * cols * t
            out = []
            for name in ("int4", "int8", "bf16"):
                try:
                    if name == "int4":
                        fn = lambda: ops.linear_int4(x, packed, scales4)
                    elif name == "int8":
                        fn = lambda: ops.linear_int8(x, q8, s8)
                    else:
                        fn = lambda: ops.linear_bf16(x, w16)
                    out.append("%9.0f" % (fl / best(fn) / 1e9))
                except Exception as exc:
                    out.append("   err")
            print("   %-7d %s" % (t, " ".join(out)), flush=True)


main()
