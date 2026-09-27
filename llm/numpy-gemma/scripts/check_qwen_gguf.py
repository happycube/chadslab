#!/usr/bin/env python3
"""Check the C path of the Qwen3.5 MoE model on a GGUF file of llama.cpp.

QWEN_PLAN.md, the GGUF path. The reference is the NumPy model on the same
file (np_gemma.qwen.QwenGGUF, float32 on the dequantized blocks). The C
path quantizes x to int8 for each 32 values, so the values are close to
the reference, not the same. The script:

1. compares the logits of the first --layers layers with the reference (one
   pass, and a prompt pass with steps);
2. checks that an MTP verify group and commit give the same bits as plain
   steps (the program only);
3. runs the whole model on a chat prompt: the answer, and the time of a
   decode step.

    OPENBLAS_NUM_THREADS=1 python scripts/check_qwen_gguf.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import ops  # noqa: E402
from np_gemma.qwen import QwenCache, QwenGGUF, QwenGGUFCPU, QwenGGUFProgram  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TOK = "models/Qwen3.6-35B-A3B-OptiQ-4bit/tokenizer.json"
TEXT = ("The quick brown fox jumps over the lazy dog. In 2024, 17 * 23 = 391, "
        "and the cache of a CPU keeps recent data close to the core.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--tok", default=TOK)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--split", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--impl", choices=("program", "cpu"), default="program")
    ap.add_argument("--skip-ref", action="store_true")
    args = ap.parse_args()
    Model = QwenGGUFProgram if args.impl == "program" else QwenGGUFCPU
    tok = QwenTokenizer(args.tok)
    ok = True
    if not args.skip_ref:
        ids = tok.encode(TEXT)
        ref_m = QwenGGUF(args.path, layers=args.layers)
        cfg = ref_m.cfg
        ref = ref_m.logits(ref_m.forward(ids, QwenCache(cfg, len(ids) + 8)))
        m = Model(args.path, cfg, layers=args.layers)
        lg = m.logits(m.forward(ids, QwenCache(cfg, len(ids) + 8)))
        r = float(np.abs(lg - ref).max() / np.abs(ref).max())
        same = float((lg.argmax(-1) == ref.argmax(-1)).mean())
        print("%d layers, one pass: logits max rel %.2e, same top token %.0f%%" % (
            args.layers, r, 100 * same))
        cache = QwenCache(cfg, len(ids) + 8)
        rows = [m.forward(ids[:args.split], cache)]
        steps = QwenCache(cfg, len(ids) + 8)
        m.forward(ids[:args.split], steps)
        for p in range(args.split, len(ids)):
            rows.append(m.forward([ids[p]], cache, start_pos=p))
        lg2 = m.logits(np.concatenate(rows))
        same2 = float((lg2.argmax(-1) == ref.argmax(-1)).mean())
        print("%d layers, prompt of %d then steps: same top token %.0f%%" % (
            args.layers, args.split, 100 * same2))
        ok &= same >= 0.9 and same2 >= 0.9
        if args.impl == "program":
            # A verify group of 4 tokens, commit 2, then 2 plain steps: the
            # same bits as 4 plain steps.
            p0 = args.split
            hv = m.verify(ids[p0:p0 + 4], steps, p0)
            m.commit(2)
            h3 = m.forward(ids[p0 + 2:p0 + 4], steps, start_pos=p0 + 2)
            plain = np.concatenate(rows[1:5])
            exact = np.array_equal(hv, plain) and np.array_equal(h3, plain[2:4])
            print("verify group and commit equal to plain steps: %s" % exact)
            ok &= exact

    m = Model(args.path)
    cfg = m.cfg
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
