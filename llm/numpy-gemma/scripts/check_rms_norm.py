"""Check the rms_norm kernel against a float64 reference.

The real kernel sums the squares with several accumulators at the same time,
so its result differs from a left to right sum in the last bits. This script
gives the size of that difference and the distance from the true value.

    PYTHONPATH=. $PY scripts/check_rms_norm.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import cops, ops  # noqa: E402

EPS = 1e-6


def ref(x, w):
    """The reference in float64, as the model definition gives it."""
    xd = x.astype(np.float64)
    mean_sq = np.mean(xd * xd, axis=-1, keepdims=True) + EPS
    y = xd * np.power(mean_sq, -0.5)
    if w is not None:
        y = y * w.astype(np.float64)
    return y


def check(rng, rows, cols, with_w, label):
    x = (rng.standard_normal((rows, cols)) * 2.5).astype(np.float32)
    w = (rng.standard_normal(cols) * 1.5 + 1.0).astype(np.float32) if with_w else None
    got = cops.rms_norm(x, w, EPS)
    exp = ref(x, w)

    # Compare with the exact value and with the float32 rounding of it.
    scale = np.max(np.abs(exp)) + 1e-30
    err = np.max(np.abs(got - exp)) / scale
    # The best that float32 can do for this input.
    floor = np.max(np.abs(exp.astype(np.float32).astype(np.float64) - exp)) / scale
    # The same kernel with a left to right sum, for the size of the change.
    n = cols
    ss = np.zeros(rows, np.float32)
    for k in range(n):
        ss += x[:, k] * x[:, k]
    old = x * (1.0 / np.sqrt(ss / np.float32(n) + np.float32(EPS)))[:, None]
    if w is not None:
        old = old * w
    change = np.max(np.abs(got - old)) / scale

    print("%-26s rows=%-5d cols=%-5d  vs float64 %.2e  float32 floor %.2e"
          "  vs old sum %.2e" % (label, rows, cols, err, floor, change))
    return err < 4.0 * max(floor, 1e-8) + 1e-7


def main():
    if not cops.available():
        print("no C library")
        return 1
    print("AVX512 build:", cops.AVX512)
    rng = np.random.default_rng(7)
    ok = True
    for cols in (1, 7, 16, 32, 63, 64, 65, 256, 512, 704, 2816, 3072):
        ok = check(rng, 3, cols, True, "weight") and ok
    ok = check(rng, 1, 2816, False, "no weight") and ok
    ok = check(rng, 40, 2816, True, "many rows") and ok
    # A large value in one place must not change the result of the others.
    x = rng.standard_normal((2, 2816)).astype(np.float32)
    x[1, 0] = 0.0
    got = cops.rms_norm(x, None, EPS)
    exp = ref(x, None)
    err = np.max(np.abs(got - exp)) / (np.max(np.abs(exp)) + 1e-30)
    print("%-26s rows=2     cols=2816   vs float64 %.2e" % ("zero term", err))
    ok = err < 1e-6 and ok
    # The NumPy path must agree with the C path.
    x = rng.standard_normal((5, 704)).astype(np.float32)
    w = rng.standard_normal(704).astype(np.float32)
    a = cops.rms_norm(x, w, EPS)
    b = ops.rms_norm(x, w, EPS)
    d = np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30)
    print("%-26s C against the ops wrapper      %.2e" % ("paths", d))
    ok = d < 1e-6 and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
