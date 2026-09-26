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
    stride = -(-tokens // 16) * 16
    qxt, sx, sumx = cops.quantize_q8_t(x, stride)
    out = cops.int4_q8_tile(qxt, sx, sumx, packed, scales, 32, tokens)
    # The reference uses the same quantized activations. Thus only the kernel
    # arithmetic is under test. The token stride is padded to a full block.
    qd = qxt[:, :tokens, :].transpose(1, 0, 2).reshape(tokens, cols)
    qd = (qd * np.repeat(sx[:, :tokens].T, 32, axis=1)).astype(np.float32)
    ref = qd @ ops.dequantize_int4(packed, scales).T
    err = np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-30)
    float_out = ops.linear_int4_numpy(x, packed, scales)
    ferr = np.max(np.abs(out - float_out)) / (np.max(np.abs(float_out)) + 1e-30)
    print("tile rows=%d cols=%d tokens=%d: q8 rel=%.3e float rel=%.3e"
          % (rows, cols, tokens, err, ferr))
    return err < 2e-5


def check_moe(rng):
    """Compare the fused mixture of experts with the per-expert int8 path."""
    experts, cols, inner = 8, 64, 32
    tokens, top_k = 6, 3
    rows_gu = 2 * inner
    gu_p, gu_s = ops.quantize_int4(
        rng.standard_normal((experts * rows_gu, cols)).astype(np.float32), 32)
    dn_p, dn_s = ops.quantize_int4(
        rng.standard_normal((experts * cols, inner)).astype(np.float32), 32)
    gu_p = gu_p.reshape(experts, rows_gu, cols // 32, 18)
    gu_s = gu_s.reshape(experts, rows_gu, cols // 32)
    dn_p = dn_p.reshape(experts, cols, inner // 32, 18)
    dn_s = dn_s.reshape(experts, cols, inner // 32)
    h = rng.standard_normal((tokens, cols)).astype(np.float32)
    idx = np.stack([rng.choice(experts, size=top_k, replace=False)
                    for _ in range(tokens)]).astype(np.int64)
    val = rng.random((tokens, top_k)).astype(np.float32)
    ref = np.zeros_like(h)
    for e in np.unique(idx):
        tok, slot = np.nonzero(idx == e)
        act = ops.linear_int4_q8(h[tok], gu_p[e], gu_s[e])
        act = ops.gelu_tanh(act[:, :inner]) * act[:, inner:]
        de = ops.linear_int4_q8(act, dn_p[e], dn_s[e])
        ref[tok] += de * val[tok, slot, None]
    out = ops.moe_int4_q8(h, (gu_p, gu_s), (dn_p, dn_s), val, idx, inner)
    err = np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-30)
    print("fused moe:          rel=%.3e" % err)
    return err < 2e-5


def check_gemv(rng, rows, cols):
    """Compare the one-token int8 dot product with the same reference."""
    w = rng.standard_normal((rows, cols)).astype(np.float32)
    packed, scales = ops.quantize_int4(w, group=32)
    x = rng.standard_normal((1, cols)).astype(np.float32) * 1.5
    qx, sx, sumx = cops.quantize_q8_groups(x)
    out = cops.int4_q8_gemv(qx[0], sx[0], sumx[0], packed, scales)
    # The reference uses the same quantized activations, so only the kernel
    # arithmetic is under test.
    qd = (qx[0].astype(np.float32) * np.repeat(sx[0], 32)).astype(np.float32)
    ref = qd @ ops.dequantize_int4(packed, scales).T
    err = np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-30)
    disp = ops.linear_int4_numpy(x, packed, scales)[0]
    derr = np.max(np.abs(out - disp)) / (np.max(np.abs(disp)) + 1e-30)
    print("gemv rows=%d cols=%d: q8 rel=%.3e float rel=%.3e"
          % (rows, cols, err, derr))
    return err < 2e-5


def check_gemv_paths(rng):
    """Compare the fused entry points with one call for each matrix."""
    mats = []
    for rows in (4096, 1024, 1024, 2112):
        w = rng.standard_normal((rows, 2816)).astype(np.float32)
        mats.append(ops.quantize_int4(w, group=32))
    x = rng.standard_normal((1, 2816)).astype(np.float32)
    fused = cops.int4_q8_gemv_x(x, mats[0][0], mats[0][1])
    qx, sx, sumx = cops.quantize_q8_groups(x)
    split = cops.int4_q8_gemv(qx[0], sx[0], sumx[0], mats[0][0], mats[0][1])
    ok = np.array_equal(fused, split)
    print("gemv_x:             fused==split %s" % ok)
    outs = cops.int4_q8_multi4(mats[:3] + [None], x, 2816)
    for i in range(3):
        ref = ops.linear_int4_numpy(x, mats[i][0], mats[i][1])[0]
        rel = np.max(np.abs(outs[i] - ref)) / (np.max(np.abs(ref)) + 1e-30)
        print("multi4 slot %d:      float rel=%.3e" % (i, rel))
        ok = rel < 2e-2 and ok
    ok = outs[3] is None and ok
    return ok


def check_moe_gemv(rng):
    """Compare the int8 expert kernel with one call for each expert."""
    ne, rows, cols = 4, 1408, 2816
    w = rng.standard_normal((ne * rows, cols)).astype(np.float32)
    p4, s4 = ops.quantize_int4(w, group=32)
    p4 = p4.reshape(ne, rows, cols // 32, 18)
    s4 = s4.reshape(ne, rows, cols // 32)
    ids = np.array([0, 2, 3], dtype=np.int32)
    x = rng.standard_normal((3, cols)).astype(np.float32)
    ok = True
    for stride, src in ((cols, x), (0, x[:1])):
        out = cops.int4_q8_moe_gemv(p4, s4, src, ids, rows, cols, stride)
        for j in range(ids.size):
            # A stride of zero gives row 0 to every job.
            row = src[:1] if stride == 0 else src[j:j + 1]
            ref = ops.linear_int4_numpy(row, p4[ids[j]], s4[ids[j]])[0]
            rel = np.max(np.abs(out[j] - ref)) / (np.max(np.abs(ref)) + 1e-30)
            print("moe_gemv stride=%d j=%d: float rel=%.3e" % (stride, j, rel))
            ok = rel < 2e-2 and ok
    return ok


def main():
    if not cops.available():
        print("no C library")
        return 1
    print("AVX512 build:", cops.AVX512)
    rng = np.random.default_rng(1234)
    ok = check_groups(rng)
    ok = check_t(rng) and ok
    ok = check_gemv(rng, 34, 128) and ok
    ok = check_gemv(rng, 8, 32) and ok
    ok = check_gemv(rng, 71, 96) and ok
    ok = check_gemv(rng, 512, 704) and ok
    ok = check_gemv_paths(rng) and ok
    ok = check_moe_gemv(rng) and ok
    for tokens in TOKENS:
        ok = check_tile(rng, 34, 128, tokens) and ok
    ok = check_tile(rng, 17, 64, 5) and ok
    ok = check_tile(rng, 8, 32, 16) and ok
    ok = check_moe(rng) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
