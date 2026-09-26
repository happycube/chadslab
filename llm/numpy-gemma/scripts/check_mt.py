#!/usr/bin/env python3
"""Check that a small token group gives the same bits as the decode steps.

An MTP verify step runs the target on a small group of tokens. The group must
give each token the same hidden state and the same logits as a decode step
for that token. Then the MTP decode gives the same text as the plain decode.

For each context length and group size, the script runs the group in one
forward pass, then runs the same tokens one at a time, and compares the
final hidden states and the logits bit for bit. It also prints the time of
the group and of the single steps.

A context of 64 uses the float32 attention. A context of 300 uses the int8
cache of the fused attention.

    OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=ACTIVE PYTHONPATH=. python scripts/check_mt.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from np_gemma import KVCache, Model
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--contexts", type=int, nargs="+", default=[64, 300])
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5, 8])
    ap.add_argument("--e4b", action="store_true", help="The GGUF file is an E4B model.")
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    if args.e4b:
        from np_gemma.e4b import E4B, E4BCache, E4BConfig
        cfg = E4BConfig({"text_config": g.text_config()})
        model = E4B(g, cfg, mode="int4")
        new_cache = lambda n: E4BCache(cfg, max_len=n)  # noqa: E731
    else:
        cfg = Config({"text_config": g.text_config()})
        model = Model(g, cfg).load_all(dtype="int4")
        new_cache = lambda n: KVCache(cfg, max_len=n)  # noqa: E731
    ids = tok.encode(open("README.md").read())

    ok = True
    print("%7s %5s %6s %10s %10s %9s %9s" % ("context", "group", "same", "hidden", "logits",
                                             "group ms", "single ms"))
    for n in args.contexts:
        cache = new_cache(n + max(args.sizes) + 8)
        model.prefill(ids[:n], cache)
        for t in args.sizes:
            batch = ids[n:n + t]
            cache.truncate(n)
            t0 = time.perf_counter()
            xg = model.forward(batch, cache=cache, start_pos=n)
            lg = model.logits(xg)
            tg = time.perf_counter() - t0
            cache.truncate(n)
            xs, ls = [], []
            t0 = time.perf_counter()
            for j, b in enumerate(batch):
                x = model.forward([b], cache=cache, start_pos=n + j)
                xs.append(x)
                ls.append(model.logits(x))
            ts = time.perf_counter() - t0
            xs = np.concatenate(xs)
            ls = np.concatenate(ls)
            same = np.array_equal(xg, xs) and np.array_equal(lg, ls)
            ok = ok and same
            print("%7d %5d %6s %10.1e %10.1e %9.1f %9.1f" % (
                n, t, "yes" if same else "NO", float(np.abs(xg - xs).max()),
                float(np.abs(lg - ls).max()), 1000 * tg, 1000 * ts))
    print("PASS" if ok else "FAIL")
    g.close()


if __name__ == "__main__":
    main()
