"""Check the Q6_K kernel against two dequantize references.

This script compares the C kernel with the GGUF reader and with the NumPy
fallback. It also checks one block against a direct scalar decode.

Use PYTHONPATH=. and give the path of a GGUF file that keeps the tied
embedding table in Q6_K. Set NP_GEMMA_ARCH to avx2 or avx512 to test one
kernel path.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from np_gemma import cops, gguf, ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--name", default="model.language_model.embed_tokens.weight")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("C library", cops.HAVE_C, "AVX-512", cops.AVX512)
    with gguf.GGUF(args.path) as g:
        cols = g.shape(args.name)[1]
        rows = min(args.rows, g.shape(args.name)[0])
        blocks = g.q6k_blocks(args.name)
        data = g.q6k_bytes(args.name)
        print("table", g.shape(args.name), g.dtype(args.name), "cols", cols)

        rng = np.random.default_rng(args.seed)
        ids = rng.choice(g.shape(args.name)[0], rows, replace=False)

        t0 = time.time()
        ref = g.q6k_dequant(blocks[ids], cols)
        print("gguf dequant  %.3f s" % (time.time() - t0))

        alt = ops.dequantize_q6k(np.ascontiguousarray(data[ids]), cols).reshape(rows, cols)
        print("gguf vs ops dequant   max abs diff %.3e" % np.abs(ref - alt).max())

        # One block against a direct scalar decode.
        blk = np.ascontiguousarray(data[ids[0], :210])
        ql = blk[0:128]
        qh = blk[128:192]
        sc = blk[192:208].view(np.int8).astype(np.int64)
        d = blk[208:210].copy().view("<f2").astype(np.float32)[0]
        man = np.empty(256, np.float32)
        for h in range(2):
            for l in range(32):
                isx = l >> 4
                q1 = int((ql[h * 64 + l] & 0x0F) | (((qh[h * 32 + l] >> 0) & 3) << 4)) - 32
                q2 = int((ql[h * 64 + l + 32] & 0x0F) | (((qh[h * 32 + l] >> 2) & 3) << 4)) - 32
                q3 = int((ql[h * 64 + l] >> 4) | (((qh[h * 32 + l] >> 4) & 3) << 4)) - 32
                q4 = int((ql[h * 64 + l + 32] >> 4) | (((qh[h * 32 + l] >> 6) & 3) << 4)) - 32
                man[h * 128 + l] = d * sc[h * 8 + isx + 0] * q1
                man[h * 128 + l + 32] = d * sc[h * 8 + isx + 2] * q2
                man[h * 128 + l + 64] = d * sc[h * 8 + isx + 4] * q3
                man[h * 128 + l + 96] = d * sc[h * 8 + isx + 6] * q4
        got = ops.dequantize_q6k(np.ascontiguousarray(data[ids[0]:ids[0] + 1, :210]), 256)
        print("scalar decode         max abs diff %.3e" % np.abs(man - got).max())

        # The kernel for a few token rows against the dequantize and the matmul.
        # One token uses the SIMD path. Three tokens use the scalar path.
        sub = np.ascontiguousarray(data[ids])
        for nt in (1, 3):
            x = rng.standard_normal((nt, cols)).astype(np.float32)
            want = x @ ref.T
            t0 = time.time()
            out = ops.linear_q6k(x, sub, cols)
            dt = time.time() - t0
            err = np.abs(out - want).max()
            print("kernel %d rows x %d   %.4f s  max abs diff %.3e  rel %.3e"
                  % (rows, nt, dt, err, err / np.abs(want).max()))
            if not np.allclose(out, want, rtol=2e-4, atol=2e-4):
                raise SystemExit("FAIL: the kernel does not match the matmul")

        # One all-ones row checks every value of each row.
        ones = ops.linear_q6k(np.ones((1, cols), np.float32), sub, cols)[0]
        err1 = np.abs(ones - ref.sum(axis=1)).max()
        print("ones row              max abs diff %.3e" % err1)
        if err1 > 2e-2:
            raise SystemExit("FAIL: the ones row does not match")
    print("PASS")


if __name__ == "__main__":
    main()
