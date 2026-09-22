#!/usr/bin/env python3
"""Measure the int8 multiply for different stream lengths.

A single call reads one large int8 matrix one time. Thus the test gives the
single-pass rate of the kernel. A short matrix gives the rate for one model
layer. The test shows if the kernel or the many short reads limit the speed.

The weights use anonymous memory with large pages. The test then measures the
best rate of the kernel alone.
"""
from __future__ import annotations

import ctypes
import mmap as mmapmod
import time

import numpy as np

from np_gemma import cops

COLS = 3840


def timed(x, q, s):
    """Return the best time of one int8 multiply."""
    run = lambda: cops.linear_int8_float(x, q, s, COLS)
    run()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, COLS)).astype(np.float32)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)

    print("%10s %10s %10s %8s" % ("rows", "MB", "ms", "GB/s"))
    for rows in (3840, 15360, 38400, 153600, 384000):
        q = np.empty((rows, COLS), dtype=np.int8)
        q.fill(1)
        s = np.ones((rows, 1), dtype=np.float32)
        try:
            libc.madvise(ctypes.c_void_p(q.ctypes.data), ctypes.c_size_t(q.nbytes),
                         ctypes.c_int(mmapmod.MADV_HUGEPAGE))
        except Exception:
            pass
        dt = timed(x, q, s)
        print("%10d %10.1f %10.3f %8.1f" % (rows, q.nbytes / 1e6, dt * 1e3,
                                            q.nbytes / dt / 1e9))
        del q, s
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
