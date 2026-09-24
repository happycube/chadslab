"""Check the GELU kernel against a float64 reference.

The C kernel uses a vector form of tanh in the AVX-512 builds. This script
measures the error of that form over the full range of the model.

Run it from the numpy-gemma directory:

    PYTHONPATH=. $PY scripts/check_gelu.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import cops  # noqa: E402

C = 0.7978845608028654


def ref64(x):
    """The reference GELU in float64."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(C * (x + 0.044715 * x * x * x)))


def ref32(x):
    """The reference GELU in float32. This is the form of the old kernel."""
    x = np.asarray(x, dtype=np.float32)
    return (0.5 * x * (1.0 + np.tanh(C * (x + 0.044715 * x * x * x)))).astype(np.float32)


def report(name, got, want):
    d = np.abs(got.astype(np.float64) - want)
    big = np.abs(want) > 1e-6
    rel = (d[big] / np.abs(want)[big]).max() if big.any() else 0.0
    print("  %-28s max|diff| %.3e   max rel %.3e" % (name, d.max(), rel))
    return d.max()


def main():
    print("AVX512=%s VNNI=%s" % (cops.AVX512, cops.VNNI))
    if not cops.available():
        print("no C kernel: nothing to check")
        return 0
    bad = 0

    # The full range of the model, and beyond it.
    for name, x in (
        ("linspace -40..40", np.linspace(-40.0, 40.0, 200001).astype(np.float32)),
        ("linspace -800..800", np.linspace(-800.0, 800.0, 100001).astype(np.float32)),
        ("normal 0, 8", np.random.default_rng(3).standard_normal(100003).astype(np.float32) * 8.0),
        ("zeros", np.zeros(10240, np.float32)),
    ):
        got = cops.gelu(x)
        w = ref64(x)
        d = report(name, got, w)
        if d > 1e-5:
            bad += 1

    # A length that is not a multiple of 16 checks the tail of the loop.
    print("  tail lengths:")
    worst = 0.0
    for n in range(1, 200):
        x = (np.random.default_rng(n).standard_normal(n) * 6.0).astype(np.float32)
        d = np.abs(cops.gelu(x).astype(np.float64) - ref64(x)).max()
        worst = max(worst, d)
    print("  n = 1..199                  max|diff| %.3e" % worst)
    if worst > 1e-5:
        bad += 1

    # The new kernel against the float32 NumPy form that the model used before.
    x = (np.random.default_rng(11).standard_normal(100000) * 4.0).astype(np.float32)
    d = np.abs(cops.gelu(x) - ref32(x)).max()
    print("  new kernel against float32  max|diff| %.3e" % d)
    if d > 1e-5:
        bad += 1

    # The speed. The E4B model calls this function 84 times for each token with
    # 10240 values in each call.
    import time
    x = (np.random.default_rng(5).standard_normal(10240) * 4.0).astype(np.float32)
    best = 1e9
    for _ in range(20):
        t0 = time.perf_counter()
        for _ in range(42):
            cops.gelu(x)
        best = min(best, time.perf_counter() - t0)
    print("  42 calls of 10240 values: %.4f s  (%.1f us a call)"
          % (best, best / 42 * 1e6))

    print("PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
