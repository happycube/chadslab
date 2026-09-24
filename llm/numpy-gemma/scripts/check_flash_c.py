"""Check the C flash attention kernels against the NumPy reference.

The reference is the plain path. The three C versions must agree with it. The
value is the largest difference over the largest reference value.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from np_gemma import cops, ops
from np_gemma.flash import attention_reference

NAMES = {1: "c", 2: "avx2", 3: "avx512"}


def run_case(t, n, qh, kvh, hd, window, base, scale, rng):
    q = (rng.standard_normal((t, qh, hd)) * scale).astype(np.float32)
    k = (rng.standard_normal((n, kvh, hd)) * scale).astype(np.float32)
    v = rng.standard_normal((n, kvh, hd)).astype(np.float32)
    first = base + n - t
    positions = np.arange(first, first + t, dtype=np.int32)
    ref = attention_reference(q, k, v, positions, base, window)
    denom = float(np.abs(ref).max()) or 1.0
    out = []
    for impl in (1, 2, 3):
        cops.attn_prefill_impl(impl)
        got = ops.flash_prefill(q, k, v, positions, base, window)
        out.append("%s %.2e" % (NAMES[impl], float(np.abs(got - ref).max()) / denom))
    cops.attn_prefill_impl(0)
    print("t=%-4d n=%-4d qh=%-3d kvh=%-2d hd=%-3d win=%-4d base=%-4d | %s"
          % (t, n, qh, kvh, hd, window, base, "  ".join(out)))
    return out


def main():
    rng = np.random.default_rng(1234)
    print("relative max difference against the NumPy reference")
    for scale in (0.04, 1.0):
        print("-- query and key scale", scale)
        run_case(64, 64, 16, 2, 512, 0, 0, scale, rng)
        run_case(64, 64, 16, 8, 256, 1024, 0, scale, rng)
        run_case(64, 300, 16, 8, 256, 64, 0, scale, rng)
        run_case(64, 100, 16, 8, 256, 0, 0, scale, rng)
        run_case(1, 200, 16, 8, 256, 64, 0, scale, rng)
        run_case(37, 200, 16, 8, 256, 0, 0, scale, rng)
        run_case(64, 128, 16, 8, 256, 0, 64, scale, rng)
        run_case(64, 128, 16, 2, 512, 32, 100, scale, rng)
        run_case(5, 5, 16, 2, 512, 0, 0, scale, rng)


main()
