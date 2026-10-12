#!/usr/bin/env python3
"""Check the int8 products of large groups on dp4a (the kernels of a GPU
without the tensor cores; this GPU acts as one: RECLANK_GPU_CC=61) against a
float reference on random weights and the same int8 x:

- GP_INT4_LINEAR_MT (k_gemm_dp4a): int4 blocks of 18 bytes, x as int8 (i8 1)
  or in the int16 form (i8 2); rows that end in a part of a tile, columns
  that end in a part of a step;
- GP_KQ_LINEAR (k_kq_dp4a): Q8_0, Q4_K, Q5_K, Q6_K (the reference: the rows
  of the CPU library, x as its kq_quant_x), and the same with x in the int16
  form (RECLANK_GPU_KQ_X2=1; the form of the experts of Qwen, RECLANK_MOE_X16).

    python scripts/check_dp4a.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("RECLANK_GPU_CC", "61")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.ir import program as P  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402
from reclank.backends.cpu import cops  # noqa: E402

TYPES = {8: ("Q8_0", 34, 32), 12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256)}


def q4_blocks(rng, rows, cols):
    """int4 blocks (a float16 scale, 16 bytes of codes), their scales as
    float32 (the program checks them against the float16 ones), and their
    floats."""
    nb = cols // 32
    w = rng.integers(0, 256, (rows, nb, 18), dtype=np.uint8)
    d = (rng.random((rows, nb)) * 0.02 + 0.001).astype(np.float16)
    w[:, :, 0:2] = d.view(np.uint8).reshape(rows, nb, 2)
    codes = w[:, :, 2:]
    v = np.concatenate([(codes & 15).astype(np.float32) - 8, (codes >> 4).astype(np.float32) - 8], -1)
    return (w.reshape(rows, -1), d.astype(np.float32),
            (v * d.astype(np.float32)[..., None]).reshape(rows, cols))


def quant_x(x, qmax):
    """x as the GPU quantizes it (k_quant_q8: 127; k_quant_x2: 16256)."""
    t, cols = x.shape
    b = x.reshape(t, -1, 32)
    s = np.abs(b).max(-1) / qmax
    q = np.rint(b / np.where(s > 0, s, 1)[..., None])
    return (q * s[..., None]).reshape(t, cols)


def kq_weights(type_, rng, rows, cols):
    """Random raw rows of a GGUF type with sane float16 scales (check_kq_tiers)."""
    _, bb, bv = TYPES[type_]
    nb = cols // bv
    w = rng.integers(0, 256, (rows, nb, bb), dtype=np.uint8)
    d = (rng.random((rows, nb)) * 0.02 + 0.001).astype(np.float16).view(np.uint8).reshape(rows, nb, 2)
    if type_ == 8:
        w[:, :, 0:2] = d
    elif type_ in (12, 13):
        w[:, :, 0:2] = d
        w[:, :, 2:4] = (rng.random((rows, nb)) * 0.01).astype(np.float16).view(np.uint8).reshape(rows, nb, 2)
    else:
        w[:, :, 208:210] = d
    return np.ascontiguousarray(w.reshape(rows, -1))


def run(emit, out, mirror, i8=False):
    prog = P.Program()
    emit(prog)
    prog.names["out"] = out
    g = GPUProgram(prog.finish(), graph=False, mirror=mirror, i8=i8)
    g.run()
    g.download("out")
    res = out.copy()
    g.close()
    return res


def main():
    rng = np.random.default_rng(3)
    mirror = Mirror()
    bad = 0
    for rows, cols, t in ((384, 1024, 200), (300, 1056, 77)):
        w, s, wf = q4_blocks(rng, rows, cols)
        x = rng.standard_normal((t, cols)).astype(np.float32)
        for i8, qmax in ((1, 127.0), (2, 16256.0)):
            out = np.zeros((t, rows), np.float32)
            got = run(lambda p: p.emit(P.INT4_LINEAR_MT, x, w, s, out, rows, cols, t), out, mirror, i8)
            ref = quant_x(x, qmax).astype(np.float64) @ wf.astype(np.float64).T
            err = float(np.abs(got - ref).max() / np.abs(ref).max())
            ok = err < 1e-5
            bad += not ok
            print("int4 %4d x %4d, t %3d, x %-5s: vs ref %.1e %s" % (
                rows, cols, t, "int8" if i8 == 1 else "int16", err, "ok" if ok else "BAD"))
    rows, cols, t = 320, 1024, 150
    x = rng.standard_normal((t, cols)).astype(np.float32)
    xq = np.empty((t, cols), np.int8)
    xs = np.empty((t, cols // 32), np.float32)
    xm = np.empty((t, cols // 16), np.float32)
    cops.kq_quant_x(x, xq, xs, xm)
    xd = xq.astype(np.float64) * np.repeat(xs, 32, axis=1)
    x16 = quant_x(x, 16256.0).astype(np.float64)
    for type_, (name, _, _) in TYPES.items():
        w = kq_weights(type_, rng, rows, cols)
        wf = cops.kq_rows(w, type_, cols, np.arange(rows)).astype(np.float64)
        for form, xr in (("int8", xd), ("int16", x16)):
            os.environ["RECLANK_GPU_KQ_X2"] = "1" if form == "int16" else "0"
            out = np.zeros((t, rows), np.float32)
            got = run(lambda p: p.emit(P.KQ_LINEAR, xq, xs, xm, x, w, type_, rows, cols, t, out),
                      out, mirror)
            ref = xr @ wf.T
            err = float(np.abs(got - ref).max() / np.abs(ref).max())
            ok = err < 1e-5
            bad += not ok
            print("%-5s %4d x %4d, t %3d, x %-5s: vs ref %.1e %s" % (name, rows, cols, t, form, err,
                                                                 "ok" if ok else "BAD"))
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
