#!/usr/bin/env python3
"""The two weight types of DeepSeek V4.1 Flash on real weights (DEEPSEEK_V41_PLAN.md
Phase 1, step 4): csrc/kernels/kq/mxx.c against NumPy forms written here.

- KQ_MXX: the pack of an expert's rows, its dequantized values against the
  E2M1 and E8M0 tables (bit for bit); the near product (int16 x) against a
  NumPy form of the same operations (the exact integer sums, the exact
  product with the scale, an fma for each block: computed in float64 and
  rounded); the exact product against float64 sums of the same rounded
  values; the act rounding (E4M3, power-of-two scales) against a NumPy form
  that picks the nearest of the 256 E4M3 values.
- the rows of a part from the zygote's segments (if one runs) against the
  rows from the shards.
- KQ_FP8B32: the dequantized values; the near and exact products against
  float64.
- the error of near and exact against float64 products of the unrounded x.

    python scripts/check_dsv41_types.py /home/cpage/models/DeepSeek-V4.1-Flash [--layer 3] [--zygote]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats import dsv41_st as D  # noqa: E402

E4M3_POS = np.unique(D.E4M3[np.isfinite(D.E4M3) & (D.E4M3 >= 0)])


def act_quant_np(x):
    """The reference's act_quant (kernel.py) on rows of x, in NumPy: the rounded
    values and the scales."""
    g = x.astype(np.float32).reshape(x.shape[0], -1, 32)
    amax = np.maximum(np.abs(g).max(-1, keepdims=True), np.float32(1e-4))
    v = (amax * np.float32(1.0 / 448.0)).astype(np.float32)
    m, e = np.frexp(v)                      # v = m 2^e, m in [0.5, 1)
    e = np.where(m == 0.5, e - 1, e)        # an exact power of two stays
    s = np.ldexp(np.float32(1.0), e).astype(np.float32)
    y = np.clip(g / s, -448, 448).astype(np.float32)
    a = np.abs(y)
    i = np.searchsorted(E4M3_POS, a).clip(1, len(E4M3_POS) - 1)
    lo, hi = E4M3_POS[i - 1], E4M3_POS[i]
    # ties to the even code: the codes of E4M3_POS are 0, 1, 2, ... in order
    pick = np.where(a - lo < hi - a, lo, np.where(hi - a < a - lo, hi, np.where((i - 1) % 2 == 0, lo, hi)))
    pick = np.where(a >= E4M3_POS[-1], E4M3_POS[-1], pick)
    q = np.where(y < 0, -pick, pick).astype(np.float32)
    return q.reshape(x.shape), s.reshape(x.shape[0], -1)


def quant16_np(x):
    """kq_quant_part16 in NumPy: v (int64, 128 hi + lo) and xs for each 32."""
    g = x.astype(np.float32).reshape(x.shape[0], -1, 32)
    mv = np.abs(g).max(-1, keepdims=True)
    iv = np.where(mv > 0, np.float32(16192.0) / np.where(mv > 0, mv, 1), 0).astype(np.float32)
    q = np.clip(np.rint((g * iv).astype(np.float32)), -16192, 16192).astype(np.int64)
    return q, (mv / np.float32(16192.0)).astype(np.float32)[..., 0]


def near_np(codes, scales, x):
    """The near product in NumPy: per block a = sum(2 e2m1 v), p = a 2^(e-127),
    f = fma(p, xs, f) (float64 then float32: exact for the product, one
    rounding for the sum), 0.5 f."""
    v, xs = quant16_np(x)
    r = codes.shape[0]
    w2 = np.stack([D.E2M1[codes & 15], D.E2M1[codes >> 4]], -1).reshape(r, -1, 32) * 2
    a = np.einsum("rbk,tbk->trb", w2.astype(np.int64), v)            # exact integers
    p = (a.astype(np.float64) * D.E8M0[scales].astype(np.float64)[None]).astype(np.float32)
    f = np.zeros((x.shape[0], r), np.float32)
    for b in range(a.shape[2]):
        f = (p[:, :, b].astype(np.float64) * xs[:, None, b].astype(np.float64) + f.astype(np.float64)
             ).astype(np.float32)
    return (f * np.float32(0.5)).astype(np.float32)


def db(err, ref):
    return 10 * np.log10(np.sum(err.astype(np.float64) ** 2) / max(np.sum(ref.astype(np.float64) ** 2), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", default="0,17,383")
    ap.add_argument("--tokens", type=int, default=5)
    ap.add_argument("--zygote", action="store_true", help="compare the rows of a running zygote too")
    a = ap.parse_args()
    src = D.DSV41Source(a.ckpt)
    rng = np.random.default_rng(0)
    bad = 0
    for e in [int(v) for v in a.experts.split(",")]:
        ex = src.expert(a.layer, e)
        for w in ("w1", "w3", "w2"):
            codes, sc = ex[w]
            rows, cols = codes.shape[0], codes.shape[1] * 2
            mxx, odd = cops.kq_mxx_pack(codes, sc, rows, cols)
            deq = cops.kq_mxx_dequant(mxx, rows, cols)
            ref = D.dequant_mxfp4(codes, sc)
            same = np.array_equal(deq.view(np.uint32), ref.view(np.uint32))
            # x like an activation: heavy tails
            x = (rng.standard_normal((a.tokens, cols)) * np.exp(rng.standard_normal((a.tokens, cols)))).astype(np.float32)
            y16 = cops.kq_mxx_linear(mxx, rows, cols, x)
            yn = near_np(codes, sc, x)
            nbit = int(np.sum(y16.view(np.uint32) != yn.view(np.uint32)))
            ye = cops.kq_mxx_linear(mxx, rows, cols, x, exact=True)
            q, s = act_quant_np(x)
            xq = (q.reshape(a.tokens, -1, 32) * s[..., None]).reshape(a.tokens, cols)
            aq = cops.dsv41_act_quant(x)
            aq_same = np.array_equal(aq.view(np.uint32), xq.astype(np.float32).view(np.uint32))
            ye64 = xq.astype(np.float64) @ ref.astype(np.float64).T
            y64 = x.astype(np.float64) @ ref.astype(np.float64).T
            ok = same and odd == 0 and nbit == 0 and aq_same
            bad += not ok
            print("expert %3d %s (%d x %d): dequant %s, odd scales %d; near vs NumPy: %d of %d differ; act_quant %s; "
                  "exact vs float64 of its values %.1f dB; vs unrounded x: near %.1f dB, exact %.1f dB" % (
                      e, w, rows, cols, "equal" if same else "DIFFERS", odd, nbit, y16.size,
                      "equal" if aq_same else "DIFFERS", db(ye - ye64, ye64), db(y16 - y64, y64),
                      db(ye - y64, y64)))
    for name in ("layers.%d.attn.wq_a.weight" % a.layer, "layers.%d.attn.wo_b.weight" % a.layer,
                 "layers.%d.ffn.shared_experts.w2.weight" % a.layer):
        codes, sc = src.fp8(name)
        rows, cols = codes.shape
        ref = D.dequant_fp8(codes, sc)
        x = (rng.standard_normal((a.tokens, cols)) * np.exp(rng.standard_normal((a.tokens, cols)))).astype(np.float32)
        yn = cops.kq_fp8b32_linear(codes, sc, rows, cols, x)
        ye = cops.kq_fp8b32_linear(codes, sc, rows, cols, x, exact=True)
        xq = cops.dsv41_act_quant(x)
        y64 = x.astype(np.float64) @ ref.astype(np.float64).T
        ye64 = xq.astype(np.float64) @ ref.astype(np.float64).T
        print("FP8 %s (%d x %d): near vs float64 %.1f dB; exact vs float64 of its values %.1f dB, vs unrounded x %.1f dB"
              % (name, rows, cols, db(yn - y64, y64), db(ye - ye64, ye64), db(ye - y64, y64)))
    if a.zygote:
        from reclank.runtime import zygote as Z
        wz = Z.attach()
        ix = wz.index
        for part in range(len(ix["segments"])):
            pl = wz.planes(a.layer, part)
            for e in [int(v) for v in a.experts.split(",")]:
                ex = src.expert(a.layer, e)
                for name, _off, r0, nr, rb in ix["planes"][part]:
                    w, kind = name.split(".")
                    ref = ex[w][0 if kind == "weight" else 1][r0:r0 + nr]
                    if not np.array_equal(pl[name][e], ref.reshape(nr, -1)):
                        bad += 1
                        print("zygote part %d expert %d %s DIFFERS" % (part, e, name))
        print("the zygote's rows of layer %d: %s" % (a.layer, "checked" if not bad else "see above"))
    print("all equal" if not bad else "%d checks failed" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
