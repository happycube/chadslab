"""Check the int4 kernel with int8 activations.

Run this script from the numpy-gemma directory:

    python scripts/check_int4_q8.py

The script compares the C kernels with a NumPy reference. The tile must agree
with the reference to float32 precision. It must stay near the float path.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import cops, ops  # noqa: E402

TOKENS = [1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65, 128]


def ref_quantize_groups(x):
    """Reference for quantize_q8_groups."""
    tokens, cols = x.shape
    groups = cols // 32
    flat = x.reshape(tokens, groups, 32)
    amax = np.max(np.abs(flat), axis=2)
    scale = np.where(amax > 0.0, amax / 127.0, 1e-12).astype(np.float32)
    q = np.rint(flat / scale[:, :, None]).clip(-127.0, 127).astype(np.int8)
    return q.reshape(tokens, cols), scale, q.astype(np.int32).sum(axis=2)


def ref_quantize_t(x):
    """Reference for quantize_q8_t."""
    tokens, cols = x.shape
    groups = cols // 32
    q, scale, sumx = ref_quantize_groups(x)
    qg = q.reshape(tokens, groups, 32).reshape(tokens, groups, 8, 4)
    qxt = qg.transpose(1, 2, 0, 3).reshape(groups * 8, tokens, 4)
    return qxt, scale.T.copy(), sumx.T.copy()


def check_groups(rng):
    x = rng.standard_normal((37, 96)).astype(np.float32) * 3.7
    qx, sx, sumx = cops.quantize_q8_groups(x)
    rq, rs, rsum = ref_quantize_groups(x)
    ok1 = np.array_equal(qx, rq)
    ok2 = np.array_equal(sx, rs)
    ok3 = np.array_equal(sumx, rsum)
    print("quantize_q8_groups: qx=%s sx=%s sumx=%s" % (ok1, ok2, ok3))
    return ok1 and ok2 and ok3


def check_t(rng):
    x = rng.standard_normal((23, 64)).astype(np.float32) * 2.0
    qxt, sx, sumx = cops.quantize_q8_t(x)
    rqxt, rsx, rsum = ref_quantize_t(x)
    ok1 = np.array_equal(qxt, rqxt)
    ok2 = np.array_equal(sx, rsx)
    ok3 = np.array_equal(sumx, rsum)
    print("quantize_q8_t:      qxt=%s sx=%s sumx=%s" % (ok1, ok2, ok3))
    if not ok1:
        print("  first difference", np.argwhere(qxt != rqxt)[:5])
    return ok1 and ok2 and ok3


def check_tile(rng, rows, cols, tokens):
    w = rng.standard_normal((rows, cols)).astype(np.float32)
    packed, scales = ops.quantize_int4(w, group=32)
    x = rng.standard_normal((tokens, cols)).astype(np.float32) * 1.5
    qxt, sx, sumx = cops.quantize_q8_t(x)
    out = cops.int4_q8_tile(qxt, sx, sumx, packed, scales, 32, tokens)
    # The reference uses the same quantized activations. Thus only the kernel
    # arithmetic is under test.
    qd = qxt.transpose(1, 0, 2).reshape(tokens, cols)
    qd = (qd * np.repeat(sx.T, 32, axis=1)).astype(np.float32)
    ref = qd @ ops.dequantize_int4(packed, scales).T
    err = np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-30)
    float_out = ops.linear_int4_numpy(x, packed, scales)
    ferr = np.max(np.abs(out - float_out)) / (np.max(np.abs(float_out)) + 1e-30)
    print("tile rows=%d cols=%d tokens=%d: q8 rel=%.3e float rel=%.3e"
          % (rows, cols, tokens, err, ferr))
    return err < 2e-5


def main():
    if not cops.available():
        print("no C library")
        return 1
    print("AVX512 build:", cops.AVX512)
    rng = np.random.default_rng(1234)
    ok = check_groups(rng)
    ok = check_t(rng) and ok
    for tokens in TOKENS:
        ok = check_tile(rng, 34, 128, tokens) and ok
    ok = check_tile(rng, 17, 64, 5) and ok
    ok = check_tile(rng, 8, 32, 16) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
