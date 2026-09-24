"""Check the packed 4-bit and 2-bit C kernel against the decoded weights.

The kernel `cops.ct_linear` reads the packed words of the compressed-tensors
file. It unpacks the values in place and applies one scale for each row. This
script proves that the unpack agrees with `CompressedTensors.dequant`, which
`check_ct.py` already proved against the `compressed_tensors` library.

Two checks:

1.  An element check. Give the kernel the one-hot vector e_k and it returns
    one column of the matrix. The script does this for every column k of a
    small group of rows, so every bit position of the row is tested. This is
    what catches a wrong bit order or a wrong sign bias.

2.  A token check. Multiply a random block of tokens and compare with the
    decoded matrix. This tests the token loop and the vector tail.

Run:

    PYTHONPATH=. $PY scripts/check_ct_kernel.py --snapshot "$SNAP4B"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

CASES = [
    ("model.language_model.layers.0.self_attn.q_proj", 4, 0, 4),
    ("model.language_model.layers.5.mlp.down_proj", 4, 0, 2),
    ("model.language_model.layers.23.mlp.up_proj", 4, 0, 2),
    ("lm_head", 2, 0, 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--columns", type=int, default=0,
                    help="0 checks every column, otherwise a sample")
    args = ap.parse_args()

    from np_gemma import cops
    from np_gemma.ct import CompressedTensors

    if not cops.available():
        print("FAIL: the C library did not build")
        return 1
    print("C library ready; AVX512=%s VNNI=%s" % (cops.AVX512, cops.VNNI))

    ct = CompressedTensors(os.path.join(args.snapshot, "model.safetensors"))
    rng = np.random.default_rng(1234)
    bad = 0

    for name, want_bits, r0, r1 in CASES:
        bits = ct.num_bits(name)
        strategy, group = ct.strategy(name)
        packed = ct.packed_words(name)
        scale = ct.channel_scale(name)
        if bits != want_bits or packed is None or scale is None:
            print("SKIP %s: bits=%s strategy=%s packed=%s scale=%s"
                  % (name, bits, strategy, packed is not None, scale is not None))
            bad += 1
            continue
        rows, words = packed.shape
        cols = words * 32 // bits
        ref = ct.rows(name, r0, r1)
        pk = packed[r0:r1]
        sc = scale[r0:r1]
        print("\n%-52s bits=%d %s rows=%d cols=%d" %
              (name.split("language_model.")[-1][:52], bits, strategy, rows, cols))

        # 1. the element check, one column at a time
        if args.columns:
            picks = rng.choice(cols, size=min(args.columns, cols), replace=False)
        else:
            picks = np.arange(cols)
        x = np.zeros((1, cols), dtype=np.float32)
        worst = 0.0
        worst_at = None
        t0 = time.time()
        for k in picks:
            x[0, k] = 1.0
            got = cops.ct_linear(x, pk, sc, bits, cols)[0]
            x[0, k] = 0.0
            d = np.abs(got - ref[:, k]).max()
            if d > worst:
                worst = d
                worst_at = int(k)
        dt = time.time() - t0
        print("  every column k, %d of %d: max|diff| = %.3g at k=%s  (%.2f s)"
              % (len(picks), cols, worst, worst_at, dt))
        if worst > 1e-5:
            print("  FAIL: the unpack does not match the decoded weight")
            bad += 1

        # 2. the token check
        xt = (rng.standard_normal((args.tokens, cols)) * 0.05).astype(np.float32)
        got = cops.ct_linear(xt, pk, sc, bits, cols)
        want = xt @ ref.T
        denom = max(1e-9, float(np.abs(want).max()))
        rel = float(np.abs(got - want).max()) / denom
        print("  %d tokens: max relative difference = %.3g" % (args.tokens, rel))
        if rel > 1e-5:
            print("  FAIL: the token loop does not match the decoded weight")
            bad += 1

    print("\n%s" % ("FAIL" if bad else "OK: the kernel matches the decoded weights"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
