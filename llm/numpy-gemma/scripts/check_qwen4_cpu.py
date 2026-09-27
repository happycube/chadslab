#!/usr/bin/env python3
"""Check the CPU program of Qwen3.8-Flash-Next (np_gemma.qwen4.Qwen4CPU).

QWEN38_PLAN.md, phase 2. The products of the program quantize x to 8 bits
for each 32 values. Thus its values are close to those of the NumPy model
(Qwen4), not the same. The script:

1. compares the logits of the first --layers layers with Qwen4: one pass,
   and a prompt of --split tokens, then steps;
2. runs a chat prompt through the whole model: the answer, and the time of
   a step.

    OPENBLAS_NUM_THREADS=1 python scripts/check_qwen4_cpu.py --layers 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import ops  # noqa: E402
from np_gemma.qwen4 import Qwen4, Qwen4Cache, Qwen4CPU  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = ("models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/"
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
TOK = "models/Qwen3.6-35B-A3B-OptiQ-4bit/tokenizer.json"
TEXT = ("The quick brown fox jumps over the lazy dog. In 2024, 17 * 23 = 391, "
        "and the cache of a CPU keeps recent data close to the core.")
PROMPT = ("<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--tok", default=TOK)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--split", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=48)
    # The error of int8 x is spread over the layers, and a router of 512
    # experts turns a small change into another expert more often than the
    # 256 of Qwen3.6. With 4 layers, about 85% of the top tokens agree
    # (making any one group of products exact changes that little).
    ap.add_argument("--min-same", type=float, default=0.8)
    args = ap.parse_args()
    tok = QwenTokenizer(args.tok)
    ok = True
    if args.layers:
        ids = tok.encode(TEXT)
        ref_m = Qwen4(args.path, layers=args.layers)
        ref = ref_m.logits(ref_m.forward(ids, Qwen4Cache(ref_m.cfg, len(ids) + 8)))
        m = Qwen4CPU(args.path, ref_m.cfg, layers=args.layers)
        lg = m.logits(m.forward(ids, Qwen4Cache(m.cfg, len(ids) + 8)))
        r = float(np.abs(lg - ref).max() / np.abs(ref).max())
        same = float((lg.argmax(-1) == ref.argmax(-1)).mean())
        print("%d layers, one pass: logits max rel %.2e, same top token %.0f%%" % (
            args.layers, r, 100 * same))
        cache = Qwen4Cache(m.cfg, len(ids) + 8)
        rows = [m.forward(ids[:args.split], cache)]
        for p in range(args.split, len(ids)):
            rows.append(m.forward([ids[p]], cache, start_pos=p))
        lg2 = m.logits(np.concatenate(rows))
        same2 = float((lg2.argmax(-1) == ref.argmax(-1)).mean())
        print("%d layers, prompt of %d then steps: same top token %.0f%%" % (
            args.layers, args.split, 100 * same2))
        ok &= same >= args.min_same and same2 >= args.min_same
    m = Qwen4CPU(args.path)
    ids = tok.encode(PROMPT)
    cache = Qwen4Cache(m.cfg, len(ids) + args.tokens + 8)
    m.forward(ids[:4], Qwen4Cache(m.cfg, 16))            # compile the programs of 4 and of 1
    m.forward([ids[0]], Qwen4Cache(m.cfg, 16))
    t0 = time.time()
    h = m.forward(ids, cache)
    t1 = time.time()
    nxt = ops.argmax(m.logits(h[-1:])[0])
    out, pos, ts = [nxt], len(ids), []
    while len(out) < args.tokens and nxt not in tok.stop_ids:
        t = time.time()
        nxt = ops.argmax(m.logits(m.forward([nxt], cache, start_pos=pos))[0])
        ts.append(time.time() - t)
        out.append(nxt)
        pos += 1
    print("prompt of %d tokens: %.2f s; decode: %.1f ms for each token (median), %.2f tok/s" % (
        len(ids), t1 - t0, 1e3 * np.median(ts), 1 / np.median(ts)))
    print(repr(tok.decode(out)))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    np.seterr(over="ignore")
    raise SystemExit(main())
