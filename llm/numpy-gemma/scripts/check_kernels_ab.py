#!/usr/bin/env python3
"""Save or compare the output of a fixed decode workload, bit for bit.

Use this script for a change to the C kernels that must keep the bits. An
example is the split of a kernel into a body and a wrapper. Run it two times.
The first run, on the old code, saves the output. The second run, on the new
code, compares it.

The workload covers the paths of a decode step. It has a prompt pass, single
decode steps, and token groups of the MTP verify step. It uses a context
past the sliding window of 1024, and it runs the output head.

    PYTHONPATH=. python scripts/check_kernels_ab.py --save base.npz
    PYTHONPATH=. python scripts/check_kernels_ab.py --compare base.npz

Run it with NP_GEMMA_ATTN=0 as well, for the float attention. Use --e4b and
the E4B GGUF file for the E4B model.
"""
from __future__ import annotations

import argparse

import numpy as np

from np_gemma import KVCache, Model
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def workload(model, new_cache, ids, contexts):
    out = {}
    for n in contexts:
        cache = new_cache(n + 32)
        x = model.prefill(ids[:n], cache)
        out["c%d.prefill" % n] = model.logits(x[-1:])
        pos = n
        for k in range(3):
            x = model.forward([ids[pos]], cache=cache, start_pos=pos)
            out["c%d.step%d.x" % (n, k)] = x
            out["c%d.step%d.logits" % (n, k)] = model.logits(x)
            pos += 1
        for t in (2, 3, 5):
            cache.truncate(pos)
            x = model.forward(ids[pos:pos + t], cache=cache, start_pos=pos)
            out["c%d.group%d.x" % (n, t)] = x
            out["c%d.group%d.logits" % (n, t)] = model.logits(x)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--e4b", action="store_true")
    ap.add_argument("--contexts", type=int, nargs="+", default=[40, 300, 1100])
    g_ = ap.add_mutually_exclusive_group(required=True)
    g_.add_argument("--save")
    g_.add_argument("--compare")
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
    out = workload(model, new_cache, ids, args.contexts)
    if args.save:
        np.savez(args.save, **out)
        print("saved %d arrays to %s" % (len(out), args.save))
        return 0
    ref = np.load(args.compare)
    bad = 0
    for k in sorted(out):
        same = np.array_equal(out[k], ref[k])
        if not same:
            bad += 1
            print("DIFF %-24s max %.3e" % (k, float(np.abs(out[k] - ref[k]).max())))
    print("%d arrays, %d differ -> %s" % (len(out), bad, "PASS" if bad == 0 else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
