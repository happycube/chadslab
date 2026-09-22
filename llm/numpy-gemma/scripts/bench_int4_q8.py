"""Compare the int4 kernels with float32 and int8 activations.

    python scripts/bench_int4_q8.py [--threads N] [--reps N]

The script times the float32 tile and the int8 tile for the shapes of the 26B
mixture-of-experts layers. The int8 time has two parts: the quantization of the
activations and the tile.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import cops, ops  # noqa: E402

# The 26B model. hidden 2816, the expert feed-forward width 704, 128 experts.
SHAPES = [
    ("moe gate_up", 1408, 2816),
    ("moe down", 2816, 704),
    ("attn q", 8192, 2816),
    ("attn kv", 1024, 2816),
]


def best(fn, reps):
    t = 1e9
    for _ in range(reps):
        a = time.perf_counter()
        fn()
        b = time.perf_counter()
        t = min(t, b - a)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    rng = np.random.default_rng(7)
    print("threads=%d reps=%d" % (args.threads, args.reps))
    print("%-12s %6s %7s  %9s %9s  %9s %9s  %9s %9s" % (
        "shape", "tokens", "GFLOP", "float-tile", "int8-quant", "int8-tile",
        "float-GF", "int8-GF", "speedup"))
    for name, rows, cols in SHAPES:
        for tokens in (16, 64, 256):
            w = (rng.standard_normal((rows, cols)).astype(np.float32) * 0.02)
            packed, scales = ops.quantize_int4(w, group=32)
            x = (rng.standard_normal((tokens, cols)).astype(np.float32) * 0.5)
            xt = np.ascontiguousarray(x.T)
            flop = 2.0 * rows * cols * tokens / 1e9

            fout = np.empty((tokens, rows), dtype=np.float32)
            tf = best(lambda: cops._lib.gemma_int4_gemm_tile_run(
                packed.ctypes.data, scales.ctypes.data, x.ctypes.data, xt.ctypes.data,
                fout.ctypes.data, rows, cols, tokens), args.reps)
            tq = best(lambda: cops.quantize_q8_t(x), args.reps)
            qxt, sx, sumx = cops.quantize_q8_t(x)
            ti = best(lambda: cops.int4_q8_tile(qxt, sx, sumx, packed, scales, 32, tokens),
                      args.reps)
            t8 = tq + ti
            print("%-12s %6d %7.2f  %9.3f %9.3f  %9.3f %9.1f  %9.1f %9.2fx" % (
                name, tokens, flop, tf * 1e3, tq * 1e3, ti * 1e3,
                flop / tf, flop / t8, tf / t8))


if __name__ == "__main__":
    main()
