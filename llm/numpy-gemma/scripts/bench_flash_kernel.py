"""Benchmark the attention kernel alone, without the model.

Compare the three C flash versions with the batched matmul path that the model
uses. That path builds the whole score matrix, so the flash kernels are the
only ones that can skip the hidden keys.

Run:  PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/bench_flash_kernel.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from np_gemma import cops, ops

NAMES = {1: "c", 2: "avx2", 3: "avx512"}


def blas_attn(q, k, v, positions, base, window):
    """The batched matmul path of the model, with the key slice."""
    t, qh, hd = q.shape
    n, kvh, _ = k.shape
    n_rep = qh // kvh
    k2, v2, base2 = k, v, base
    if window:
        kpos = base + np.arange(n)
        lo = int(np.searchsorted(kpos, positions.min() - window + 1, 'left'))
        hi = int(np.searchsorted(kpos, positions.max(), 'right'))
        k2, v2, base2 = k[lo:hi], v[lo:hi], base + lo
    m = k2.shape[0]
    qb = q.reshape(t, kvh, n_rep, hd).transpose(1, 0, 2, 3).reshape(kvh, t * n_rep, hd)
    scores = np.matmul(qb, k2.transpose(1, 2, 0)).reshape(kvh, t, n_rep, m)
    probs = ops.softmax_mask(scores, positions, n_rep, base2, window)
    out = np.matmul(probs.reshape(kvh, t * n_rep, m), v2.transpose(1, 0, 2))
    return out.reshape(kvh, t, n_rep, hd).transpose(1, 0, 2, 3).reshape(t, qh, hd)


def best(fn, reps=3):
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


def main():
    rng = np.random.default_rng(7)
    t = 256
    configs = [
        ("global ", 16, 2, 512, 0),
        ("sliding", 16, 8, 256, 1024),
    ]
    for n in (1024, 2048, 4096, 8192):
        for name, qh, kvh, hd, window in configs:
            q = (rng.standard_normal((t, qh, hd)) * 0.04).astype(np.float32)
            k = (rng.standard_normal((n, kvh, hd)) * 0.04).astype(np.float32)
            v = rng.standard_normal((n, kvh, hd)).astype(np.float32)
            positions = np.arange(n - t, n, dtype=np.int32)
            tb = best(lambda: blas_attn(q, k, v, positions, 0, window))
            ref = None
            row = ["n=%-5d %s hd=%-3d win=%-4d blas %7.1fms" % (n, name, hd, window, tb * 1e3)]
            for impl in ((1, 2, 3) if n <= 1024 else (2, 3)):
                cops.attn_prefill_impl(impl)
                got = ops.flash_prefill(q, k, v, positions, 0, window)
                if ref is None:
                    ref = got
                tc = best(lambda: ops.flash_prefill(q, k, v, positions, 0, window))
                row.append("%s %7.1fms %4.2fx" % (NAMES[impl], tc * 1e3, tb / tc))
            cops.attn_prefill_impl(0)
            print("  ".join(row), flush=True)
    cops.attn_prefill_impl(0)


main()
