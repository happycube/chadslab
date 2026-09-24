"""Measure the int4 int8-activation tile: 16-token block against 32-token block.

The wide tile decodes a weight block one time for 32 tokens. It should cut the
instructions that do not multiply, which the VNNI test says are the larger part.

Run:  PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/bench_tile.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from np_gemma import cops, ops


def best(fn, reps=7):
    fn()
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


def main():
    rng = np.random.default_rng(3)
    print("AVX-512:", cops.AVX512, " VNNI:", cops.have_vnni())
    for rows, cols, label in ((2816, 2816, "dense 2816x2816"),
                              (704, 2816, "expert 704x2816")):
        w = rng.standard_normal((rows, cols)).astype(np.float32)
        packed, scales = ops.quantize_int4(w)
        print("== %s" % label)
        for t in (16, 32, 64, 128, 256, 512):
            x = rng.standard_normal((t, cols)).astype(np.float32)
            a = best(lambda: ops.linear_int4_q8(x, packed, scales))
            b = best(lambda: ops.linear_int4_q8_wide(x, packed, scales))
            fl = 2.0 * rows * cols * t
            if t == 64:
                ya = ops.linear_int4_q8(x, packed, scales)
                yb = ops.linear_int4_q8_wide(x, packed, scales)
                d = float(np.abs(ya - yb).max()) / (float(np.abs(ya).max()) or 1.0)
                print("   check t=64 relative maxdiff %.2e" % d)
            print("   t=%-5d block16 %8.0f  block32 %8.0f  %5.2fx"
                  % (t, fl / a / 1e9, fl / b / 1e9, a / b), flush=True)


main()
