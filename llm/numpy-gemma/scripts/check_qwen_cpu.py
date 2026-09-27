#!/usr/bin/env python3
"""Check the C path of the Qwen3.5 MoE model (np_gemma.qwen.QwenCPU).

QWEN_PLAN.md, phase 2. The products of the C path quantize x to int8 for
each group of 64, so the values are close to the reference, not the same.
The script:

1. compares the logits of the first --layers layers with the reference of
   scripts/qwen_reference.py (one pass, and a prompt pass with steps);
2. runs the whole model on a chat prompt: the answer, and the time of a
   decode step.

    OPENBLAS_NUM_THREADS=1 python scripts/check_qwen_cpu.py --ref ref4.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import ops  # noqa: E402
from np_gemma.qwen import QwenCache, QwenConfig, QwenCPU, QwenProgram  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = "models/Qwen3.6-35B-A3B-OptiQ-4bit"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--split", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--impl", choices=("program", "cpu"), default="program",
                    help="program: the step as one program (QwenProgram); cpu: one call "
                         "for each operation (QwenCPU).")
    args = ap.parse_args()
    cfg = QwenConfig(args.path)
    Model = QwenProgram if args.impl == "program" else QwenCPU
    ok = True
    if args.ref:
        ref = np.load(args.ref)
        ids = [int(x) for x in ref["ids"]]
        m = Model(args.path, cfg, layers=args.layers)
        lg = m.logits(m.forward(ids, QwenCache(cfg, len(ids) + 8)))
        r = float(np.abs(lg - ref["logits"]).max() / np.abs(ref["logits"]).max())
        same = float((lg.argmax(-1) == ref["logits"].argmax(-1)).mean())
        print("%d layers, one pass: logits max rel %.2e, same top token %.0f%%" % (
            args.layers, r, 100 * same))
        cache = QwenCache(cfg, len(ids) + 8)
        rows = [m.forward(ids[:args.split], cache)]
        for p in range(args.split, len(ids)):
            rows.append(m.forward([ids[p]], cache, start_pos=p))
        lg2 = m.logits(np.concatenate(rows))
        same2 = float((lg2.argmax(-1) == ref["logits"].argmax(-1)).mean())
        print("%d layers, prompt of %d then steps: same top token %.0f%%" % (
            args.layers, args.split, 100 * same2))
        ok &= same >= 0.9 and same2 >= 0.9

    m = Model(args.path, cfg)
    tok = QwenTokenizer(os.path.join(args.path, "tokenizer.json"))
    prompt = ("<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = tok.encode(prompt)
    cache = QwenCache(cfg, len(ids) + args.tokens + 8)
    t0 = time.time()
    h = m.forward(ids, cache)
    t1 = time.time()
    nxt = ops.argmax(m.logits(h[-1:])[0])
    out, pos = [nxt], len(ids)
    ts = []
    while len(out) < args.tokens and nxt not in tok.stop_ids:
        t = time.time()
        h = m.forward([nxt], cache, start_pos=pos)
        nxt = ops.argmax(m.logits(h)[0])
        ts.append(time.time() - t)
        out.append(nxt)
        pos += 1
    print("prompt of %d tokens: %.2f s; decode: %.1f ms for each token (median), %.1f tok/s" % (
        len(ids), t1 - t0, 1e3 * np.median(ts), 1 / np.median(ts)))
    print(repr(tok.decode(out)))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
