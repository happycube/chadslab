"""Measure how low-rank each kind of weight matrix is.

For each matrix, report the error at a few ranks, and the rank that a tolerance
needs. A factorised matrix of rank r uses r * (m + n) numbers and r * (m + n)
multiplies for each token, against m * n for the dense form.

Run:  PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/check_spectrum.py
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


def analyze(name, w):
    if w.ndim != 2 or min(w.shape) < 256:
        return
    m, n = w.shape
    s = np.linalg.svd(w, compute_uv=False)
    tot = float(np.sqrt(np.sum(s * s)))
    cum = np.cumsum((s * s)[::-1])
    tail = np.sqrt(np.concatenate([cum[::-1], [0.0]]))
    print("  %-20s %5d x %-5d" % (name, m, n))
    for pct in (0.10, 0.25, 0.50):
        r = max(1, int(round(pct * min(m, n))))
        frac = r * (m + n) / float(m * n)
        print("      rank %3.0f%% of n -> weight err %5.1f%%   storage %5.2fx   flops %5.2fx"
              % (pct * 100, 100.0 * tail[r] / tot, frac, 1.0 / frac))
    for tol in (0.01, 0.05, 0.10):
        r = int(np.argmax(tail <= tol * tot))
        frac = r * (m + n) / float(m * n)
        print("      weight err <= %3.0f%% needs rank %5d (%5.1f%% of n)  storage %5.2fx  flops %5.2fx"
              % (tol * 100, r, 100.0 * r / min(m, n), frac, 1.0 / frac))


def main():
    g = GGUF(P)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    for li in (0, 15):
        w = model.load_layer(li)
        print("=== layer %d" % li)
        for key in ("self_attn.q_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.down_proj"):
            if key in w:
                packed, scales = w[key]
                analyze(key, ops.dequantize_int4(packed, scales))
        packed, scales = w["experts.gate_up_proj"]
        analyze("expert0 gate_up", ops.dequantize_int4(packed[0], scales[0]))
        packed, scales = w["experts.down_proj"]
        analyze("expert0 down", ops.dequantize_int4(packed[0], scales[0]))
        model.free_layer(li)


main()
