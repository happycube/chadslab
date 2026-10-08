#!/usr/bin/env python3
"""Check a step in parts against the program of one part, bit for bit.

SPLIT_PLAN.md, phase 1. The script runs the prompt pass, then decode steps
of one token. For each step it first runs the program of one part
(program.decode_step). Then it removes the new cache row and runs the same
step in parts (parts.decode_step). It compares the hidden state and the
logits.

The step in parts writes the cache row that the next step reads.
Thus a wrong cache row shows in the next steps.

On jackal, the parts are teams on one NUMA node. The check shows that the
bits are the same, not the speed of NUMA. It prints the time of each form.

    OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=ACTIVE OMP_PLACES=cores \\
        PYTHONPATH=. python scripts/check_parts.py

Run it with NP_GEMMA_ATTN=0 as well, for the float cache.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from np_gemma import KVCache, Model
from np_gemma import parts as parts_mod
from np_gemma import program
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--contexts", type=int, nargs="+", default=[200, 1100])
    ap.add_argument("--parts", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    g = GGUF(args.gguf)
    cfg = Config({"text_config": g.text_config()})
    tok = Tokenizer.from_gguf(g)
    model = Model(g, cfg).load_all(dtype="int4")
    ids = tok.encode(open("README.md").read())
    ok = True
    for n in args.parts:
        for ctx in args.contexts:
            cache = KVCache(cfg, max_len=ctx + args.steps + 8)
            model.prefill(ids[:ctx], cache)
            attn = program.ready(model, cache)
            same, t1, tn = True, [], []
            for k in range(args.steps):
                pos = ctx + k
                tokens = [ids[pos]]
                t0 = time.perf_counter()
                x1 = program.decode_step(model, cache, tokens, pos)
                t1.append(time.perf_counter() - t0)
                cache.truncate(pos)
                t0 = time.perf_counter()
                xn = parts_mod.decode_step(model, cache, tokens, pos, attn, n)
                tn.append(time.perf_counter() - t0)
                if not (np.array_equal(x1, xn)
                        and np.array_equal(model.logits(x1), model.logits(xn))):
                    same = False
                    print("  step %d: max |d| of the hidden state %.3e" % (
                        k, float(np.abs(x1 - xn).max())))
            ok = ok and same
            print("%d parts, context %4d, attention %s, %d steps: %s. "
                  "One part %.1f ms, %d parts %.1f ms (median, without the head)" % (
                      n, ctx, attn, args.steps, "same" if same else "DIFFERENT",
                      1000 * np.median(t1), n, 1000 * np.median(tn)))
    print("PASS" if ok else "FAIL")
    g.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
