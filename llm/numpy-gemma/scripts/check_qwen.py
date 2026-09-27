#!/usr/bin/env python3
"""Compare np_gemma.qwen with the outputs of transformers.

QWEN_PLAN.md, phase 1. scripts/qwen_reference.py makes the reference file
with the same count of layers. The script compares:

1. the output of each layer and the logits, for the whole prompt in one
   pass;
2. the same after a prompt pass of the first --split tokens and then one
   token at a time (the cache: the convolution, the state, the keys).

    python scripts/check_qwen.py --ref ref4.npz --layers 4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.qwen import Qwen, QwenCache, QwenConfig  # noqa: E402

PATH = "models/Qwen3.6-35B-A3B-OptiQ-4bit"


def rel(a, b):
    assert np.abs(b).max() > 0, "the reference is zero"
    return float(np.abs(a - b).max() / np.abs(b).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--split", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    ref = np.load(args.ref)
    ids = [int(x) for x in ref["ids"]]
    cfg = QwenConfig(args.path)
    model = Qwen(args.path, cfg, layers=args.layers)
    ok = True

    got = {}
    h = model.forward(ids, QwenCache(cfg, len(ids) + 8), hook=lambda k, v: got.__setitem__(k, v))
    for i in range(args.layers - 1):
        r = rel(got["layer.%d" % i], ref["layer.%d" % i])
        print("one pass, layer %d (%s): max rel %.2e" % (i, cfg.layer_types[i], r))
        ok &= r < args.tol
    lg = model.logits(h)
    r = rel(lg, ref["logits"])
    same = float((lg.argmax(-1) == ref["logits"].argmax(-1)).mean())
    print("one pass, logits: max rel %.2e, same top token %.0f%%" % (r, 100 * same))
    ok &= r < args.tol

    cache = QwenCache(cfg, len(ids) + 8)
    rows = [model.forward(ids[:args.split], cache)]
    for p in range(args.split, len(ids)):
        rows.append(model.forward([ids[p]], cache, start_pos=p))
    h2 = np.concatenate(rows)
    lg2 = model.logits(h2)
    r = rel(lg2, ref["logits"])
    same = float((lg2.argmax(-1) == ref["logits"].argmax(-1)).mean())
    print("prompt of %d, then steps: logits max rel %.2e, same top token %.0f%%" % (
        args.split, r, 100 * same))
    ok &= r < args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
