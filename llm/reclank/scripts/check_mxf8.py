#!/usr/bin/env python3
"""The GPU's products of KQ_MXF8 matrices (FP4-Local's MXFP8 rows, kernels/quant/mxf8.cu) against
float64: GP_KQ_LINEAR of 1-8 tokens (k_mxf8_gemv, k_mxf8_gemv_s), of a group (k_mxf8_tc) and of a
large group (the bfloat16 rows and gg_gemm_bf16). x rounds to bfloat16 and so does the output, so a
value may differ from the rounded float64 sum by one bfloat16 step (or, where the sum cancels, by
float32's error: 3e-5 of the row's rms); and the staged kernel of 2-8
tokens (k_mxf8_gemv_s) must give the bits of k_mxf8_gemv (RECLANK_GPU_MXF8_STAGE=0, in a process of
its own: the switch is read once).

    python scripts/check_mxf8.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.ir import program as P  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402
from reclank.formats.fp4local_st import bf16r, mxf8_pack, mxfp8_dequant  # noqa: E402

KQ_MXF8 = 66
SHAPES = ((768, 2560), (320, 6144), (640, 1152))
TOKENS = (1, 2, 3, 4, 5, 8, 64, 600)


def weights(rng, rows, cols):
    codes = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x3F          # (no NaN codes)
    scales = rng.integers(112, 124, (rows, cols // 32), dtype=np.uint8)
    return mxf8_pack(codes, scales), mxfp8_dequant(codes, scales).astype(np.float64)


def products(mirror):
    """The GPU's outputs of every shape and count of tokens (float32 arrays), and the float64 sums."""
    rng = np.random.default_rng(5)
    got, ref = {}, {}
    for rows, cols in SHAPES:
        w, wf = weights(rng, rows, cols)
        for t in TOKENS:
            x = rng.standard_normal((t, cols)).astype(np.float32)
            out = np.zeros((t, rows), np.float32)
            prog = P.Program()
            prog.emit(P.KQ_LINEAR, None, None, None, x, w, KQ_MXF8, rows, cols, t, out)
            prog.names["out"] = out
            g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
            g.run()
            g.download("out")
            g.close()
            got[(rows, cols, t)] = out.copy()
            ref[(rows, cols, t)] = bf16r(x).astype(np.float64) @ wf.T
    return got, ref


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--dump":
        got, _ref = products(Mirror())
        np.savez(sys.argv[2], **{"%d_%d_%d" % k: v for k, v in got.items()})
        return 0
    got, ref = products(Mirror())
    bad = 0
    for (rows, cols, t), o in got.items():
        r = ref[(rows, cols, t)]
        # the bfloat16 step at each value (8 significant bits); where the sum cancels (|r| far under
        # the row's scale) the float32 sums of the tensor cores may differ by more steps, so a value
        # is right within one step or within 3e-5 of the row's rms
        ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(r), 1e-30))) - 7)
        diff = np.abs(o.astype(np.float64) - r)
        rms = np.sqrt((r * r).mean(-1, keepdims=True))
        step = np.where(diff <= 3e-5 * rms, 0.0, diff / ulp)
        ok = float(step.max()) <= 1.0
        bad += not ok
        print("%4d x %4d, t %3d: the largest difference %.2f bfloat16 steps, %.1e of the row's rms %s" % (
            rows, cols, t, float(step.max()), float((diff / rms).max()), "ok" if ok else "BAD"))
    # the staged kernel against k_mxf8_gemv, bit for bit
    here = os.path.dirname(os.path.abspath(__file__))
    fd, tmp = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    env = dict(os.environ, RECLANK_GPU_MXF8_STAGE="0")
    subprocess.run([sys.executable, os.path.join(here, "check_mxf8.py"), "--dump", tmp], env=env, check=True)
    old = np.load(tmp)
    for (rows, cols, t), o in got.items():
        if 2 <= t <= 8:
            same = np.array_equal(o.view(np.uint32), old["%d_%d_%d" % (rows, cols, t)].view(np.uint32))
            bad += not same
            print("%4d x %4d, t %3d: k_mxf8_gemv_s %s k_mxf8_gemv" % (
                rows, cols, t, "==" if same else "BAD: differs from"))
    os.unlink(tmp)
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
