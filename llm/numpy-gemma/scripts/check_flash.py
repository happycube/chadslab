"""Check the flash attention reference against the plain path.

    python scripts/check_flash.py

No model is needed. The script builds random queries, keys, and values and
compares the tile path with the full-score path for many shapes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.flash import attention_reference, flash_attention  # noqa: E402

CASES = [
    # tokens, keys, kv heads, head dim, window, base, block_q, block_k
    (1, 1, 1, 64, 0, 0, 1, 8),
    (8, 8, 1, 64, 0, 0, 4, 4),
    (32, 100, 2, 128, 0, 0, 16, 32),
    (64, 500, 2, 128, 128, 0, 32, 64),
    (17, 4096, 8, 256, 1024, 3072, 64, 128),
    (256, 256, 2, 512, 0, 0, 64, 96),
    (33, 65, 1, 32, 0, 0, 7, 11),
    (5, 200, 4, 256, 64, 195, 2, 32),
]


def main():
    rng = np.random.default_rng(11)
    ok = True
    for (t, n, kvh, hd, window, base, bq, bk) in CASES:
        n_rep = 4
        q = rng.standard_normal((t, kvh * n_rep, hd)).astype(np.float32)
        k = rng.standard_normal((n, kvh, hd)).astype(np.float32)
        v = rng.standard_normal((n, kvh, hd)).astype(np.float32)
        positions = base + np.arange(n - t, n)
        ref = attention_reference(q, k, v, positions, base, window)
        got = flash_attention(q, k, v, positions, base, window, block_q=bq, block_k=bk)
        scale = np.abs(ref).max() + 1e-30
        err = np.abs(got - ref).max() / scale
        good = err < 2e-5
        ok &= good
        print("t=%-4d n=%-5d kvh=%d hd=%-4d win=%-5d base=%-5d bq=%-3d bk=%-4d rel=%.2e %s"
              % (t, n, kvh, hd, window, base, bq, bk, err, "ok" if good else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
