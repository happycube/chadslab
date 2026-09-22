#!/usr/bin/env python3
"""Compare the NumPy path and the Numba path. Do not load the model weights.

This test uses one large matrix. The matrix has the size of the MLP gate
projection. The test shows the speed of the multiply only.
"""
from __future__ import annotations

import time

import numpy as np

from np_gemma import ops

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
    # Make valid weights. Convert float32 bits to bfloat16 bits.
    w32 = rng.standard_normal((rows, cols)).astype(np.float32)
    u16 = (w32.view(np.uint32) >> 16).astype(np.uint16)
    x = rng.standard_normal((tokens, cols)).astype(np.float32)
    w32 = ops.bf16_to_f32(u16)

    print("matrix %dx%d, tokens %d" % (rows, cols, tokens))
    print("numba available:", numba_ops is not None and numba_ops.enabled())
    print()

    t_np = timeit(lambda: ops.linear_bf16_numpy(x, u16))
    print("numpy block dequant : %7.3f s" % t_np)
    if numba_ops is not None and numba_ops.enabled():
        t_nb = timeit(lambda: numba_ops.linear_bf16(x, u16))
        print("numba fused bf16    : %7.3f s  (%.2fx)" % (t_nb, t_np / t_nb))
        ref = ops.linear_bf16_numpy(x, u16)
        got = numba_ops.linear_bf16(x, u16)
        print("numba compared with numpy: max abs diff %.3e" % float(np.abs(ref - got).max()))
    print()
    t_blas = timeit(lambda: x @ w32.T)
    print("blas float32        : %7.3f s" % t_blas)
    if numba_ops is not None and numba_ops.enabled():
        t_nf = timeit(lambda: numba_ops.linear_f32(x, w32))
        print("numba float32       : %7.3f s  (blas is %.2fx)" % (t_nf, t_nf / t_blas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
