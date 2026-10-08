#!/usr/bin/env python3
"""Check the TQ6 cache form (np_gemma/tq6.py) in its three forms: NumPy, the
C library (csrc/bf16_linear.c), and the GPU (csrc/gpu.cu).

1. The tables: csrc/tq6_tables.h is the one copy. bf16_linear.c and gpu.cu
   include it and have no table of their own; tq6.py reads it.
2. The C quantizer against NumPy on random vectors: the same indices, the
   norms and the rotation within float rounding.
3. The GPU (if there is one): GP_KV_WRITETQ and GP_TQ_ROT against the CPU
   records on the same rows give the same bytes and norms (the two use the
   same order of the operations).

    python scripts/check_tq6.py [--no-gpu]
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from np_gemma import cops, tq6  # noqa: E402
from np_gemma import program as P  # noqa: E402


def check_tables():
    ok = tq6.CODEBOOK.size == 64 and tq6.EDGES.size == 63
    ok &= bool(np.all(np.diff(tq6.CODEBOOK) > 0)) and np.array_equal(tq6.CODEBOOK, -tq6.CODEBOOK[::-1])
    ok &= bool(np.all((tq6.EDGES > tq6.CODEBOOK[:-1]) & (tq6.EDGES < tq6.CODEBOOK[1:])))
    csrc = os.path.join(ROOT, "np_gemma", "csrc")
    for name in ("bf16_linear.c", "gpu.cu"):
        text = open(os.path.join(csrc, name), encoding="utf-8").read()
        inc = '#include "tq6_tables.h"' in text
        own = re.search(r"tq6_(cb|edges)(_d)?\[\d+\]\s*=\s*\{", text) is not None or \
            re.search(r"#define\s+TQ6_SIGNS", text) is not None
        print("  %-14s includes tq6_tables.h: %s, a table of its own: %s" % (name, inc, own))
        ok &= inc and not own
    print("tables: %s" % ("ok" if ok else "FAIL"))
    return ok


def check_cpu(rng):
    x = (rng.standard_normal((4000, 256)) * 3).astype(np.float32)
    x[0] = 0.0                                          # a group of zeros
    b, s = cops.tq6_quantize(x)
    bn, sn = tq6.quantize(x)
    ic, inp = tq6.indices(b.reshape(-1, 24)), tq6.indices(bn.reshape(-1, 24))
    mism = int((ic != inp).sum())
    nrm = float(np.abs(s - sn.reshape(-1)).max() / np.abs(sn).max())
    deq = float(np.abs(cops.tq6_dequantize_rotated(b, s) - tq6.dequantize_rotated(bn, sn).reshape(-1)).max())
    rot = float(np.abs(cops.tq6_rotate(x) - tq6.rotate(x)).max())
    back = float(np.abs(cops.tq6_rotate(cops.tq6_rotate(x), True) - x).max())
    rel = float(((tq6.dequantize(bn, sn) - x) ** 2).sum() / (x ** 2).sum())
    ok = mism <= 2 and nrm < 1e-6 and deq < 1e-4 and rot < 1e-5 and back < 1e-4 and rel < 1e-3
    print("C against NumPy: %d of %d indices differ, norms %.1e, dequant %.1e, rotation %.1e, "
          "round trip %.1e, relative MSE %.2e: %s" % (mism, ic.size, nrm, deq, rot, back, rel,
                                                       "ok" if ok else "FAIL"))
    return ok


def check_gpu(rng):
    from np_gemma import gpu
    if not gpu.available():
        print("GPU: none, skipped")
        return True
    n = 64 * 512
    k = (rng.standard_normal(n) * 2).astype(np.float32)
    v = (rng.standard_normal(n) * 2).astype(np.float32)
    outs = {}
    for side in ("cpu", "gpu"):
        prog = P.Program()
        kq, vq = np.zeros(n * 3 // 4, np.uint8), np.zeros(n * 3 // 4, np.uint8)
        ks, vs = np.zeros(n // 32, np.float32), np.zeros(n // 32, np.float32)
        q = k.copy()
        prog.names.update(k=k, v=v, kq=kq, ks=ks, vq=vq, vs=vs, q=q)
        prog.emit(P.KV_WRITETQ, k, v, None, None, kq, ks, vq, vs, n)
        prog.emit(P.TQ_ROT, q, n // 32, 0)
        prog = prog.finish()
        if side == "cpu":
            prog.run()
        else:
            g = gpu.GPUProgram(prog, graph=False)
            for nm in ("k", "v", "q"):
                g.upload(nm)
            g.run()
            for nm in ("kq", "ks", "vq", "vs", "q"):
                g.download(nm)
            g.close()
        outs[side] = [a.copy() for a in (kq, ks, vq, vs, q)]
    c, g = outs["cpu"], outs["gpu"]
    same = [np.array_equal(a, b) for a, b in zip(c, g)]
    ok = all(same)
    print("GPU against the C records: bytes of the keys %s, norms %s, bytes of the values %s, "
          "norms %s, rotation %s: %s" % (*same, "ok" if ok else "FAIL"))
    return ok


def main():
    rng = np.random.default_rng(0)
    ok = check_tables()
    ok = check_cpu(rng) and ok
    if "--no-gpu" not in sys.argv:
        ok = check_gpu(rng) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
