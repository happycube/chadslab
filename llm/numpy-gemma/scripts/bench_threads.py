#!/usr/bin/env python3
"""Measure the memory and the kernel speed against the OpenMP thread count.

The thread count must be set before the OpenMP library starts. Run this script
one time for each value of OMP_NUM_THREADS.
"""
from __future__ import annotations

import os
import time

import numpy as np

from np_gemma import cops

COLS = 3840
ROWS = 15360


def best(fn, reps=5):
    fn()
    out = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out


def main():
    threads = os.environ.get("OMP_NUM_THREADS", "?")
    rng = np.random.default_rng(0)
    q = rng.integers(-127, 128, size=(ROWS, COLS)).astype(np.int8)
    s = (np.abs(q).max(axis=1) / 127.0).astype(np.float32)[:, None]
    print("threads %2s  weight %5.1f MB" % (threads, q.nbytes / 1e6))
    for t in (1, 8, 64, 256):
        x = np.random.default_rng(1).standard_normal((t, COLS)).astype(np.float32)
        dt = best(lambda: cops.linear_int8_float(x, q, s, COLS))
        macs = ROWS * COLS * t
        print("    T=%3d  %8.3f ms   weight %6.1f GB/s   %7.1f GFLOP/s" % (
            t, dt * 1e3, q.nbytes / dt / 1e9, 2.0 * macs / dt / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
