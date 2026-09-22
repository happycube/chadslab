#!/usr/bin/env python3
"""Measure the prompt GEMM of the int8 and int4 paths.

The script makes a random matrix with the size of one MLP projection. It does
not load the model weights. Set NP_GEMMA_ARCH=avx2 or NP_GEMMA_ARCH=avx512 to
select one C library.
"""
from __future__ import annotations

import time

import numpy as np

from np_gemma import cops, ops


def best_time(fn, repeats=3):
    """Return the smallest time of several calls."""
    fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    if cops is None or not cops.available():
        print("The C path is not available.")
        return 1
    rng = np.random.default_rng(0)
    shapes = ((15360, 3840), (3840, 15360), (3840, 3840))
    tokens_list = (32, 64, 128, 256)
    print("AVX-512 library:", cops.AVX512)
    print("%-5s %-12s %6s %10s %10s" % ("mode", "shape", "tokens", "ms", "GFLOP/s"))
    for rows, cols in shapes:
        q = rng.integers(-127, 128, size=(rows, cols), dtype=np.int8)
        scale8 = np.ones((rows, 1), dtype=np.float32)
        w = rng.standard_normal((rows, cols)).astype(np.float32)
        packed, scale4 = ops.quantize_int4(w)
        for t in tokens_list:
            x = rng.standard_normal((t, cols)).astype(np.float32)
            xt = np.ascontiguousarray(x.T)
            dt = best_time(lambda: cops.linear_int8_gemm(x, xt, q, scale8))
            print("int8  %5dx%5d %6d %10.3f %10.0f"
                  % (rows, cols, t, dt * 1e3, 2 * rows * cols * t / dt / 1e9))
            dt = best_time(lambda: ops.linear_int4(x, packed, scale4))
            print("int4  %5dx%5d %6d %10.3f %10.0f"
                  % (rows, cols, t, dt * 1e3, 2 * rows * cols * t / dt / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
