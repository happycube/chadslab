#!/usr/bin/env python3
"""Compare the kernel paths on one large matrix. Do not load the model weights.

The matrix has the size of the MLP gate projection. The test shows the speed of
the multiply only.
"""
from __future__ import annotations

import time

import numpy as np

from np_gemma import ops

try:
    from np_gemma import cops
except Exception:
    cops = None
try:
    from np_gemma import numba_ops
except Exception:
    numba_ops = None


def timeit(fn, repeats=3):
    """Return the average time of one call. Call the function one time first."""
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def main():
    rng = np.random.default_rng(0)
    rows, cols, tokens = 15360, 3840, 1
    w32 = rng.standard_normal((rows, cols)).astype(np.float32)
    u16 = (w32.view(np.uint32) >> 16).astype(np.uint16)
    x = rng.standard_normal((tokens, cols)).astype(np.float32)
    w32 = ops.bf16_to_f32(u16)

    print("matrix %dx%d, tokens %d" % (rows, cols, tokens))
    print("C available:", cops is not None and cops.available())
    print("Numba available:", numba_ops is not None and numba_ops.enabled())
    print()

    ref = ops.linear_bf16_numpy(x, u16)
    t = timeit(lambda: ops.linear_bf16_numpy(x, u16))
    print("numpy block dequant : %7.3f s" % t)

    if numba_ops is not None and numba_ops.enabled():
        got = numba_ops.linear_bf16(x, u16)
        t = timeit(lambda: numba_ops.linear_bf16(x, u16))
        print("numba fused bf16    : %7.3f s  (max abs diff %.2e)" % (t, float(np.abs(ref - got).max())))
    if cops is not None and cops.available():
        got = cops.linear_bf16(x, u16)
        t = timeit(lambda: cops.linear_bf16(x, u16))
        print("C fused bf16        : %7.3f s  (max abs diff %.2e)" % (t, float(np.abs(ref - got).max())))

    q8, sc8 = ops.quantize_int8(w32)
    g8 = ops.int8_group(q8, sc8)
    ref8 = ops.linear_int8_numpy(x, q8, sc8)
    t = timeit(lambda: ops.linear_int8_numpy(x, q8, sc8))
    print("numpy dequant int8  : %7.3f s  (group %d)" % (t, g8))
    if cops is not None and cops.available():
        got8 = cops.linear_int8_s8(x, q8, sc8)
        t = timeit(lambda: cops.linear_int8_s8(x, q8, sc8))
        print("C integer int8      : %7.3f s  (max abs diff %.2e)" % (t, float(np.abs(ref8 - got8).max())))
    print()
    t = timeit(lambda: x @ w32.T)
    print("blas float32        : %7.3f s" % t)
    if cops is not None and cops.available():
        t = timeit(lambda: cops.linear_f32(x, w32))
        print("C float32           : %7.3f s" % t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
