#!/usr/bin/env python3
"""Check that a token of a small group (an MTP verify group) gets the bits of
a decode step in the int8 products of Q4_0 matrices on the GPU (the dense
Gemma models, the E4B and the 12B): GP_KQ_LINEAR, GP_KQ_MULTI of three
matrices, and GP_KQ_MULTI of a gate and an up matrix with GP_GELU_MUL_ROWS
(the fused k_kq_glu_i8), for groups of 1 to 8 tokens against each token
alone, on matrices of 2048 rows or more (two rows a warp,
RECLANK_GPU_KQ_R2) with an odd count (the last warp has one row; a warp
of GP_KQ_MULTI takes the last row of a matrix and the first of the next)
and on a small one (one row a warp). The values are also near a float64
product (x is int8 on the GPU).

    python scripts/check_kq_group.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.ir import program as P  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402

Q4_0 = 2        # KQ_Q4_0 of backends/cuda/csrc


def q4_0(rng, rows, cols):
    """A random Q4_0 matrix (rows of cols / 32 blocks of 18 bytes) and its
    float64 values."""
    nb = cols // 32
    q = rng.integers(0, 16, (rows, nb, 32)).astype(np.uint8)
    d = (rng.random((rows, nb)) * 0.02 + 0.001).astype(np.float16)
    blk = np.zeros((rows, nb, 18), np.uint8)
    blk[:, :, :2] = d.view(np.uint8).reshape(rows, nb, 2)
    blk[:, :, 2:] = q[:, :, :16] | (q[:, :, 16:] << 4)
    vals = (q.astype(np.float64) - 8) * d.astype(np.float64)[:, :, None]
    return np.ascontiguousarray(blk.reshape(rows, nb * 18)), vals.reshape(rows, cols)


def run(prog, names, mirror):
    g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
    g.run()
    for n in names:
        g.download(n)
    g.close()


def linear(x, w, rows, cols, mirror):
    t = x.shape[0]
    out = np.zeros((t, rows), np.float32)
    p = P.Program()
    p.emit(P.KQ_LINEAR, None, None, None, x, w, Q4_0, rows, cols, t, out)
    p.names["out"] = out
    run(p, ["out"], mirror)
    return out


def multi(x, ws, rows, cols, mirror, glu=False):
    t = x.shape[0]
    outs = [np.zeros((t, r), np.float32) for r in rows]
    p = P.Program()
    args = [x, cols, t, len(ws)]
    for w, r, o in zip(ws, rows, outs):
        args += [w, Q4_0, r, o]
    p.emit(P.KQ_MULTI, *args)
    names = {}
    if glu:
        h = np.zeros((t, rows[0]), np.float32)
        p.emit(P.GELU_MUL_ROWS, outs[0], outs[1], h, t, rows[0])
        names["h"] = h
    else:
        for i, o in enumerate(outs):
            names["o%d" % i] = o
    p.names.update(names)
    run(p, list(names), mirror)
    return [names["h"]] if glu else outs


def gelu(v):
    return 0.5 * v * (1.0 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)))


def main():
    rng = np.random.default_rng(9)
    mirror = Mirror()
    cols = 2560
    bad = 0
    cases = [("linear 2049 rows", "linear", [2049]), ("linear 1000 rows", "linear", [1000]),
             ("multi 2049+1025+2047 rows", "multi", [2049, 1025, 2047]),
             ("gate/up 2049 rows", "glu", [2049, 2049]), ("gate/up 1000 rows", "glu", [1000, 1000])]
    for name, kind, rows in cases:
        mats = [q4_0(rng, r, cols) for r in rows]
        ws = [m[0] for m in mats]
        for t in range(1, 9):
            x = rng.standard_normal((t, cols)).astype(np.float32)

            def go(xx):
                if kind == "linear":
                    return [linear(xx, ws[0], rows[0], cols, mirror)]
                return multi(xx, ws, rows, cols, mirror, glu=kind == "glu")
            group = go(x)
            alone = [go(x[j:j + 1].copy()) for j in range(t)]
            same = all(np.array_equal(group[k][j], alone[j][k][0])
                       for k in range(len(group)) for j in range(t))
            # near float64 (x is int8 with a scale a block on the GPU)
            ref = [x.astype(np.float64) @ m[1].T for m in mats]
            if kind == "glu":
                ref = [gelu(ref[0]) * ref[1]]
            err = max(float(np.abs(g - r).max() / (np.abs(r).max() + 1e-12)) for g, r in zip(group, ref))
            ok = same and err < 2e-2
            bad += not ok
            print("%-27s t %d: each token as alone: %s, rel err %.1e %s" % (
                name, t, "same bits" if same else "DIFFERENT", err, "ok" if ok else "BAD"), flush=True)
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
