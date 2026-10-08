#!/usr/bin/env python3
"""The products of a prompt pass alone: a KQ_Q4X matrix (int8 x, kq_linear)
on a block of tokens, the rate in int8 products a second.

    numactl -N0 -m0 env OMP_NUM_THREADS=24 PYTHONPATH=. \\
        python scripts/bench_q4x_gemm.py --rows 15360 --cols 3840 --tokens 256

With --check, the result is compared with the kernel of one token (the
same operations for each token, so the same bits).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from np_gemma import cops, ops


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=15360)
    ap.add_argument("--cols", type=int, default=3840)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--q16", action="store_true", help="int16 x (kq_linear16)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows, cols, t = args.rows, args.cols, args.tokens
    nb = cols // 32
    # Q4_0 blocks (18 bytes: a float16 scale, then 16 bytes of codes) and
    # float32 scales of the same values, as the int4 matrices of the model.
    packed = rng.integers(0, 256, size=(rows, nb, 18), dtype=np.uint8)
    sc = (rng.random((rows, nb), dtype=np.float32) * 0.02 + 0.001).astype(np.float16)
    packed[:, :, :2] = sc.view(np.uint8).reshape(rows, nb, 2)
    scales = sc.astype(np.float32)
    qx = cops.kq_q4x_pack(packed, scales)
    x = rng.standard_normal((t, cols), dtype=np.float32)
    run = (lambda: cops.kq_linear16(qx, rows, cols, x)) if args.q16 else (lambda: ops._q4x_linear(x, qx))
    out = run()                                   # warm, and the pages of the result
    t0 = time.perf_counter()
    for _ in range(args.reps):
        out = run()
    dt = (time.perf_counter() - t0) / args.reps
    mac = rows * cols * t
    print("rows %d cols %d tokens %d: %.2f ms, %.2f T int8 products/s, %.1f GB/s of weights"
          % (rows, cols, t, 1e3 * dt, mac / dt / 1e12, qx.nbytes / dt / 1e9))
    if args.check and args.q16:
        # against float32 x and the float32 weights of the same codes
        nb_ = cols // 32
        codes = np.concatenate([packed[:, :, 2:] & 15, packed[:, :, 2:] >> 4], axis=2).astype(np.float32) - 8
        wf = (codes * scales[:, :, None]).reshape(rows, cols)
        ref = x.astype(np.float64) @ wf.T.astype(np.float64)
        err = np.abs(out - ref).max() / np.abs(ref).max()
        r8 = ops._q4x_linear(x, qx)
        print("int16 x: max |d| / max of float64 %.2e (int8 x %.2e)" % (err, np.abs(r8 - ref).max() / np.abs(ref).max()))
    elif args.check:
        one = np.concatenate([ops._q4x_linear(x[j:j + 1], qx) for j in range(min(t, 8))])
        print("the first 8 tokens against the kernel of one token: same bits %s"
              % np.array_equal(one, out[:min(t, 8)]))


if __name__ == "__main__":
    raise SystemExit(main())
