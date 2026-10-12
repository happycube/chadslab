#!/usr/bin/env python3
"""The two weight types of DeepSeek V4.1 Flash on the GPU (kernels/dsv41/types.cu)
against the CPU (csrc/kernels/kq/mxx.c), on real weights (DEEPSEEK_V41_PLAN.md
Phase 3): GP_DS_MXX_LINEAR must give the bits of the CPU's near product (an
expert gives the same output on either); GP_DS_FP8_LINEAR is compared in dB.

    python scripts/check_dsv41_gpu_types.py /home/cpage/models/DeepSeek-V4.1-Flash [--layer 3]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats import dsv41_st as D  # noqa: E402
from reclank.ir import program as P  # noqa: E402


def gpu_run(build):
    from reclank.backends.cuda.runtime import GPUProgram
    prog = P.Program()
    out = build(prog)
    prog.names["out"] = out
    g = GPUProgram(prog.finish(), graph=False)
    g.run()
    g.download("out")
    g.close()
    return out.copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()
    src = D.DSV41Source(a.ckpt)
    rng = np.random.default_rng(1)
    bad = 0
    for e in (0, 200):
        ex = src.expert(a.layer, e)
        for w in ("w1", "w2"):
            codes, sc = ex[w]
            rows, cols = codes.shape[0], codes.shape[1] * 2
            mxx, _ = cops.kq_mxx_pack(codes, sc, rows, cols)
            for t in (1, 4):
                x = (rng.standard_normal((t, cols)) * np.exp(rng.standard_normal((t, cols)))).astype(np.float32)
                cpu = cops.kq_mxx_linear(mxx, rows, cols, x)

                def build(prog):
                    out = np.zeros((t, rows), np.float32)
                    prog.emit(P.DS_MXX_LINEAR, x, mxx, out, rows, cols, t, np.zeros((t, cols), np.int8),
                              np.zeros((t, cols), np.int8), np.zeros((t, cols // 32), np.float32))
                    return out
                gpu = gpu_run(build)
                nd = int(np.sum(gpu.view(np.uint32) != cpu.view(np.uint32)))
                bad += nd != 0
                print("MXX expert %d %s (%d x %d), %d tokens: GPU vs CPU %s" % (
                    e, w, rows, cols, t, "the same bits" if nd == 0 else "%d of %d DIFFER (max %.3g)" % (
                        nd, cpu.size, np.abs(gpu - cpu).max())))
    for name in ("layers.%d.attn.wq_b.weight" % a.layer, "layers.%d.attn.wo_b.weight" % a.layer):
        codes, sc = src.fp8(name)
        rows, cols = codes.shape
        for t in (1, 3, 40, 129):
            x = (rng.standard_normal((t, cols))).astype(np.float32)
            ref = x.astype(np.float64) @ D.dequant_fp8(codes, sc).astype(np.float64).T

            def build(prog):
                out = np.zeros((t, rows), np.float32)
                prog.emit(P.DS_FP8_LINEAR, x, np.ascontiguousarray(codes), sc, out, rows, cols, t)
                return out
            gpu = gpu_run(build)
            err = 10 * np.log10(((gpu - ref) ** 2).sum() / (ref ** 2).sum())
            bad += err > -120
            print("FP8 %s (%d x %d), %d tokens: GPU vs float64 %.1f dB" % (name, rows, cols, t, err))
    print("all good" if not bad else "%d checks failed" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
