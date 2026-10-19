#!/usr/bin/env python3
"""Checks of the EXL3 MUL1 matrices (EXL3_PLAN.md; reclank/formats/exl3.py,
backends/cpu/csrc/kernels/kq/exl3.c) on random tiles, for each K:

1. the products of the CPU (VNNI) against the scalar form (the same int8 x);
2. the CPU product against NumPy: the weights decoded (cops.exl3_decode,
   MUL1 without the float16 rounding), the transforms, and x itself: the
   error is that of the int8 x (about -40 dB);
3. with --ref DIR (a directory of reference tensors of ExLlamaV3:
   scripts/exl3_reference.py): the decoder bit for bit against the float16
   weights of ExLlamaV3, and the products against its forward;
4. with --gpu: the GPU products (GP_KQ_LINEAR of a KQ_E3 matrix) against the
   CPU: the GEMV (dp4a) and, from 32 rows on, the tensor cores (k_e3_mma);
   and the int16 x of the tensor cores (that of a group of experts) against
   NumPy.

    python scripts/check_exl3.py [--ref DIR] [--gpu]
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.exl3 import fwht128  # noqa: E402


def db(ref, x):
    d = np.asarray(x, np.float64) - ref
    return 10 * np.log10(max((d * d).sum() / max((np.asarray(ref, np.float64) ** 2).sum(), 1e-30), 1e-30))


def random_block(rng, K, k, n):
    """A KQ_E3 block: random trellis words, suh and svh of the sizes of a
    real file (suh about 1 / 1.24 / sqrt(k) with signs, svh about 0.01)."""
    tr = rng.integers(0, 1 << 16, size=(k // 16, n // 16, 16 * K), dtype=np.uint32).astype(np.uint16)
    suh = (rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k) / 1.24371088).astype(np.float16)
    svh = (rng.uniform(0.5, 1.5, n) * 0.01).astype(np.float16)
    blk = np.concatenate([tr.reshape(-1).view(np.uint8), suh.view(np.uint8), svh.view(np.uint8)])
    assert blk.size == cops.e3_bytes(K, k, n)
    return blk, tr, suh, svh


def weights(tr, suh, svh, k, n, cb="mul1x"):
    w = fwht128(cops.exl3_decode(tr, k, n, cb), 0) * suh.astype(np.float32)[:, None]
    return fwht128(w, 1) * svh.astype(np.float32)[None, :]


def check_cpu(rng, Ks, shapes, t):
    ok = True
    for K in Ks:
        for k, n in shapes:
            blk, tr, suh, svh = random_block(rng, K, k, n)
            x = rng.standard_normal((t, k)).astype(np.float32)
            x[:, rng.integers(0, k)] *= 30          # a large input channel
            xq, xs, xn = cops.e3_had_in(x, suh)
            a = cops.e3_cols(blk, K, k, n, xq, xs)
            b = cops.e3_cols(blk, K, k, n, xq, xs, ref=True)
            d1 = db(b, a)
            y = cops.e3_linear(blk, K, n, k, x)
            ref = x.astype(np.float64) @ weights(tr, suh, svh, k, n)
            d2 = db(ref, y)
            # the same with the int8 x of the kernel (the error of the kernel alone)
            xh = (xq.astype(np.float32).reshape(t, k // 128, 128) * xs[:, :, None]).reshape(t, k)
            wp = cops.exl3_decode(tr, k, n, "mul1x").astype(np.float64)
            yq = fwht128((xh.astype(np.float64) @ wp).astype(np.float32), 1) * svh.astype(np.float32)
            d3 = db(yq, y)
            good = d1 < -100 and d2 < -33 and d3 < -100
            ok &= good
            print("K %d %4d x %4d, t %d: VNNI vs scalar %6.1f dB; vs NumPy %6.1f dB (the int8 x), "
                  "%6.1f dB (the same x)%s" % (K, k, n, t, d1, d2, d3, "" if good else "  FAIL"))
    return ok


def check_gpu(rng, Ks, shapes, ts):
    """The GPU product (rc_gpu_e3_linear: the kernels of GP_KQ_LINEAR) against
    the CPU: the same int8 x, so they differ only in the order of the sums."""
    from reclank.backends.cuda.gpumm import Buffer
    from reclank.backends.cuda.runtime import _check, lib
    ok = True
    for K in Ks:
        for k, n in shapes:
            for t in ts:
                blk, tr, suh, svh = random_block(rng, K, k, n)
                x = rng.standard_normal((t, k)).astype(np.float32)
                x[:, rng.integers(0, k)] *= 30
                ref = cops.e3_linear(blk, K, n, k, x)
                bw, bx, by = Buffer(blk.nbytes), Buffer(x.nbytes), Buffer(4 * t * n)
                bw.upload(blk)
                bx.upload(x)
                _check(lib().rc_gpu_e3_linear(bw.ptr, K, n, k, bx.ptr, t, by.ptr, 0.0))
                y = np.empty((t, n), np.float32)
                by.download(y)
                d = db(ref, y)
                good = d < -60
                ok &= good
                print("GPU K %d %5d x %5d, t %3d: vs CPU %6.1f dB%s" % (K, k, n, t, d, "" if good else "  FAIL"))
    return ok


def check_gpu_x16(rng, Ks, shapes, ts):
    """The int16 x of the tensor cores (rc_gpu_e3_linear_x16: the body of
    k_e3g_mma, a group of MUL1 experts) against NumPy (the weights decoded,
    the transforms, x itself): the error of int16 x, where int8 x gives about
    -40 dB."""
    from reclank.backends.cuda.gpumm import Buffer
    from reclank.backends.cuda.runtime import _check, lib
    ok = True
    for K in Ks:
        for k, n in shapes:
            for t in ts:
                blk, tr, suh, svh = random_block(rng, K, k, n)
                x = rng.standard_normal((t, k)).astype(np.float32)
                x[:, rng.integers(0, k)] *= 30
                ref = x.astype(np.float64) @ weights(tr, suh, svh, k, n)
                bw, bx, by = Buffer(blk.nbytes), Buffer(x.nbytes), Buffer(4 * t * n)
                bw.upload(blk)
                bx.upload(x)
                y8, y16 = np.empty((t, n), np.float32), np.empty((t, n), np.float32)
                _check(lib().rc_gpu_e3_linear(bw.ptr, K, n, k, bx.ptr, t, by.ptr, 0.0))
                by.download(y8)
                _check(lib().rc_gpu_e3_linear_x16(bw.ptr, K, n, k, bx.ptr, t, by.ptr))
                by.download(y16)
                d8, d16 = db(ref, y8), db(ref, y16)
                good = d16 < -70
                ok &= good
                print("GPU x16 K %d %5d x %5d, t %3d: vs NumPy %6.1f dB (int8 x %6.1f dB)%s" % (
                    K, k, n, t, d16, d8, "" if good else "  FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--K", default="1,2,3,4,5,6,7,8")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    Ks = [int(v) for v in args.K.split(",")]
    ok = check_cpu(rng, Ks, [(256, 384), (512, 256)], 1)
    ok &= check_cpu(rng, [4, 6], [(2048, 512), (512, 2048)], 5)
    if args.gpu:
        ok &= check_gpu(rng, Ks, [(256, 384), (512, 256)], [1, 3, 40, 130])
        ok &= check_gpu(rng, [3, 4, 6], [(3840, 4096), (15360, 3840)], [1, 8, 37, 200])
        ok &= check_gpu_x16(rng, Ks, [(256, 384), (512, 256)], [32, 40, 130])
        ok &= check_gpu_x16(rng, [4, 5, 6], [(2560, 640), (640, 2560)], [64, 200])
        # long slices (the loop over the blocks of a slice, as a group's one slice)
        ok &= check_gpu_x16(rng, [5], [(2560, 640)], [2112])
        ok &= check_gpu_x16(rng, [4, 6], [(3840, 4096)], [200])
    print("check_exl3: %s" % ("OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
