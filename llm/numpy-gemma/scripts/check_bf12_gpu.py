#!/usr/bin/env python3
"""Check the products of KQ_BF12 rows on the GPU (csrc/gpu.cu).

A matrix of bfloat16 values with the spread of a dense matrix of a model
goes to BF12 (cops.kq_bf16_to_bf12); GP_KQ_LINEAR and GP_KQ_MULTI of the
BF12 rows and of their bfloat16 bits (cops.kq_bf12_to_bf16, type 30) for
groups of 1 to 512 tokens:

- 1 token (kq_row_part): against float64 products of the bfloat16 bits;
- 2 to 8 tokens (k_kq_bf12_tc for 1024 rows or more; else kq_row_bf16_nt,
  with and without the split of a long row) and GP_KQ_MULTI: the bits of
  each token alone (MTP verify groups);
- more than 16 tokens (gg_gemm_bf12: chunks of bfloat16 rows, then the
  tensor cores): the bits of the bfloat16 matrix (gg_gemm_bf16), with a
  matrix of more than one chunk;
- NP_GEMMA_GPU_BF16_TC=0 (k_kq_gemm, kq_dequant8): against float64.

    python scripts/check_bf12_gpu.py [--bench]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from np_gemma import cops  # noqa: E402
from np_gemma import program as P  # noqa: E402
from np_gemma.gpu import GPUProgram, Mirror  # noqa: E402


def bf16_matrix(rng, rows, cols):
    """bfloat16 bits (uint16) of normal values with a spread of scales by
    row, and a few zeros."""
    a = rng.standard_normal((rows, cols)).astype(np.float32) * 0.02
    a *= np.exp(rng.standard_normal((rows, 1)).astype(np.float32) * 0.5)
    a[rng.random((rows, cols)) < 1e-3] = 0.0
    return (a.view(np.uint32) >> 16).astype(np.uint16)


def run_linear(mats, x, mirror, multi=False, reps=0):
    """GP_KQ_LINEAR (or one GP_KQ_MULTI) of the matrices [(w, type, rows,
    cols)] with the rows of x; return the outputs. reps > 0: return the
    median ms of each record over reps runs (GPUProgram.profile)."""
    t, cols = x.shape
    prog = P.Program()
    outs = [np.zeros((t, r), np.float32) for _w, _ty, r, _c in mats]
    if multi:
        args = [x, cols, t, len(mats)]
        for (w, ty, r, _c), o in zip(mats, outs):
            args += [w, ty, r, o]
        prog.emit(P.KQ_MULTI, *args)
    else:
        for (w, ty, r, c), o in zip(mats, outs):
            prog.emit(P.KQ_LINEAR, None, None, None, x, w, ty, r, c, t, o)
    for k, o in enumerate(outs):
        prog.names["o%d" % k] = o
    prog = prog.finish()
    g = GPUProgram(prog, graph=False, mirror=mirror)
    if reps:
        g.run()
        ms = np.median([g.profile() for _ in range(reps)], axis=0)
        g.close()
        return ms
    g.run()
    res = []
    for k in range(len(outs)):
        g.download("o%d" % k)
        res.append(prog.names["o%d" % k].copy())
    g.close()
    return res


def bench(rng, mirror):
    """The ms of GP_KQ_LINEAR of the dense shapes of Qwen3.8 (BF12 and
    bfloat16) for prompt groups; run with NP_GEMMA_GPU_BF12_FUSED=0 for the
    scratch path."""
    print("rows x cols, tokens: BF12 ms, bfloat16 ms (BF12 / bf16)")
    for rows, cols in ((10240, 2560), (6144, 2560), (2560, 6144), (2560, 4096), (10240, 320),
                       (320, 10240)):
        bits = bf16_matrix(rng, rows, cols)
        b12 = cops.kq_bf16_to_bf12(bits, cols)[0].reshape(-1)
        w16 = cops.kq_bf12_to_bf16(b12, rows, cols).reshape(-1).view(np.uint8)
        for t in (512, 2048):
            x = rng.standard_normal((t, cols)).astype(np.float32)
            # the smallest median of 3 alternating rounds (the clocks vary)
            a = b = 1e9
            for _ in range(3):
                a = min(a, float(run_linear([(b12, 57, rows, cols)], x, mirror, reps=20)[0]))
                b = min(b, float(run_linear([(w16, 30, rows, cols)], x, mirror, reps=20)[0]))
            print("%5d x %5d, %4d: %.3f, %.3f (%.2f)" % (rows, cols, t, a, b, a / b), flush=True)


def main():
    rng = np.random.default_rng(5)
    mirror = Mirror()
    if "--bench" in sys.argv:
        bench(rng, mirror)
        return 0
    ok = True
    for rows, cols in ((512, 4096), (256, 8192), (2048, 2560), (4096, 8192), (320, 10240),
                       (10240, 320), (1056, 2592)):
        bits = bf16_matrix(rng, rows, cols)
        b12, rep = cops.kq_bf16_to_bf12(bits, cols)
        b12 = b12.reshape(-1)
        back = cops.kq_bf12_to_bf16(b12, rows, cols)
        w16 = back.reshape(-1).view(np.uint8)
        wf = (back.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        for t in (1, 2, 3, 8, 64, 200, 512):
            if (rows * cols > 6e6 or t == 200) and t not in (1, 200, 512):
                continue
            x = rng.standard_normal((t, cols)).astype(np.float32)
            ref = x.astype(np.float64) @ wf.T
            g12 = run_linear([(b12, 57, rows, cols)], x, mirror)[0]
            g16 = run_linear([(w16, 30, rows, cols)], x, mirror)[0]
            rel = float(np.abs(g12 - ref).max() / np.abs(ref).max())
            line = "%5d x %5d, %3d tokens: BF12 against float64 %.2e" % (rows, cols, t, rel)
            good = rel < (1e-5 if t <= 16 else 1e-2)
            if t > 16:
                same = np.array_equal(g12, g16)
                line += ", the bits of bfloat16 (tensor cores): %s" % same
                good &= same
            if 2 <= t <= 8:
                alone = np.concatenate([run_linear([(b12, 57, rows, cols)], x[j:j + 1], mirror)[0]
                                        for j in range(t)])
                same = np.array_equal(g12, alone)
                # GP_KQ_MULTI against itself for each token (a step merges
                # the products of one x too; a long row is not split there)
                two = [(b12, 57, rows, cols), (b12, 57, rows, cols)]
                m2 = run_linear(two, x, mirror, multi=True)
                m1 = [run_linear(two, x[j:j + 1], mirror, multi=True) for j in range(t)]
                same_m = all(np.array_equal(m2[k], np.concatenate([m[k] for m in m1]))
                             for k in range(2))
                line += ", the bits of each token alone: %s, GP_KQ_MULTI: %s" % (same, same_m)
                good &= same and same_m
            print(line + ("" if good else "  FAIL"), flush=True)
            ok &= good
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
