"""Measure how much the 128 experts of one layer share a common basis.

Each expert is a large matrix. If the experts are small variations on a common
set of directions, then a shared basis of k directions plus one small
coefficient vector for each expert stores them in far less space than 128
separate matrices.

Run:  PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/check_expert_basis.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from np_gemma import Model, ops
from np_gemma.config import Config
from np_gemma.gguf import GGUF

P = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"


def report(name, key, packed, scales):
    ne = packed.shape[0]
    m = packed.shape[1]
    n = packed.shape[2] * 32
    E = np.empty((ne, m * n), dtype=np.float32)
    for e in range(ne):
        E[e] = ops.dequantize_int4(packed[e], scales[e]).reshape(-1)
    mu = E.mean(axis=0)
    tot2 = float(np.sum(E * E))
    mean2 = float(np.sum(mu * mu)) * ne
    E -= mu
    dev2 = float(np.sum(E * E))
    G = E @ E.T
    ev = np.linalg.eigvalsh(G)[::-1]
    ev = np.maximum(ev, 0.0)
    print("  %-18s %d experts, %d x %d   (mean energy %5.1f%%, deviation %5.1f%%)"
          % (name, ne, m, n, 100.0 * mean2 / tot2, 100.0 * dev2 / tot2))
    cum = np.cumsum(ev) / float(np.sum(ev))
    for k in (1, 2, 4, 8, 16, 32, 64):
        frac = (k * m * n + ne * k) / float(ne * m * n)
        print("      basis %3d -> %5.1f%% of the deviation energy, storage %6.3fx"
              % (k, 100.0 * cum[k - 1], frac))
    del E, G


def main():
    g = GGUF(P)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    for li in (0, 15):
        w = model.load_layer(li)
        print("=== layer %d" % li)
        for key, nm in (("experts.gate_up_proj", "gate_up"),
                        ("experts.down_proj", "down")):
            report(nm, key, w[key][0], w[key][1])
        model.free_layer(li)


main()
