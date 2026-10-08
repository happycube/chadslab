"""Test whether a frequency domain transform can compress the weights.

A 2D separable transform (FFT, DCT, wavelet) writes a matrix as a sum of rank
one outer products. The low frequency corner is the natural sparse form: a
K x K corner costs K * K coefficients and the index needs no storage. Compare
that with the SVD, which costs r * (m + n) for rank r.

A random matrix of the same shape is the control. If the model's matrices are
no more concentrated than random, the training made no spatial structure.

Run:  PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/check_freq.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from np_gemma import Model, ops
from np_gemma.config import Config
from np_gemma.gguf import GGUF

P = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def best_toeplitz(w):
    m, n = w.shape
    t = np.zeros_like(w)
    for d in range(-(m - 1), n):
        i = np.arange(max(0, -d), min(m, n - d))
        t[i, i + d] = w[i, i + d].mean()
    return t


def analyze(name, w, rng=None):
    if w.ndim != 2 or min(w.shape) < 256:
        return
    m, n = w.shape
    tot2 = float(np.sum(w * w))
    c = np.fft.fftshift(np.fft.fft2(w))
    mag = np.abs(c) ** 2 / (m * n * tot2)
    cm, cn = m // 2, n // 2
    s = np.linalg.svd(w, compute_uv=False)
    cum = np.cumsum((s * s)[::-1])
    tail = np.sqrt(np.concatenate([cum[::-1], [0.0]])) / np.sqrt(tot2)
    print("  %-18s %d x %d" % (name, m, n))
    print("      %-14s %10s %10s %12s %10s" % ("budget", "dct corner", "dct topk",
                                               "svd (same budget)", "toeplitz"))
    tolr = float(np.abs(w - best_toeplitz(w)).max()) / (float(np.abs(w).max()) or 1.0)
    for budget in (16384, 65536, 262144, 1048576):
        k = int(np.sqrt(budget))
        h = max(1, k // 2)
        r0, r1 = max(0, cm - h), min(m, cm + h)
        c0, c1 = max(0, cn - h), min(n, cn + h)
        corner = np.sqrt(max(0.0, 1.0 - float(np.sum(mag[r0:r1, c0:c1]))))
        flat = mag.ravel()
        idx = np.argpartition(flat, -budget)[-budget:]
        topk = np.sqrt(max(0.0, 1.0 - float(np.sum(flat[idx]))))
        r = max(1, int(budget / (m + n)))
        svd = tail[r]
        print("      %-14d %9.1f%% %9.1f%% %11.1f%% %9.1f%%"
              % (budget, 100 * corner, 100 * topk, 100 * svd, 100 * tolr))


def main():
    g = GGUF(P)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    rng = np.random.default_rng(0)
    w = model.load_layer(15)
    packed, scales = w["self_attn.q_proj"]
    analyze("attn.q_proj", ops.dequantize_int4(packed, scales))
    packed, scales = w["experts.gate_up_proj"]
    analyze("expert0 gate_up", ops.dequantize_int4(packed[0], scales[0]))
    model.free_layer(15)
    m, n = 2816, 2816
    print("  %-18s %d x %d" % ("random control", m, n))
    analyze("random control", rng.standard_normal((m, n)).astype(np.float32))


main()
