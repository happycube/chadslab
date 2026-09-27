#!/usr/bin/env python3
"""Check the NumPy model of Qwen3.8-Flash-Next (np_gemma.qwen4).

QWEN38_PLAN.md, phase 1. Two checks:

1. With --dump: the sum of the streams after each of the first --layers
   layers, against the sum of l_last-N of llama.cpp. The dump comes from
   llama-eval-callback of the branch qwen4exp, for the text --text.
   The products of llama.cpp quantize x to 8 bits, so the sums are close,
   not the same.
2. A chat prompt through the whole model: the first tokens of the answer.

    llama-eval-callback -m MODEL -ngl 0 -p "The capital of France is" -n 1 > dump.txt
    OPENBLAS_NUM_THREADS=18 python scripts/check_qwen4.py --dump dump.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.qwen4 import Qwen4, Qwen4Cache  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = ("models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/"
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
TOK = "models/Qwen3.6-35B-A3B-OptiQ-4bit/tokenizer.json"


def dump_sums(path):
    """The sum of each tensor of a llama-eval-callback dump (the first
    time of each name)."""
    sums, name = {}, None
    for ln in open(path):
        m = re.match(r"common_debug_cb_eval:\s+(.+?) = \(f32\)", ln)
        if m:
            name = m.group(1)
            continue
        m = re.match(r"\s+sum = (\S+)", ln)
        if m and name and name not in sums:
            sums[name] = float(m.group(1))
    return sums


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--tok", default=TOK)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--tokens", type=int, default=12)
    args = ap.parse_args()
    tok = QwenTokenizer(args.tok)
    ok = True
    if args.dump:
        sums = dump_sums(args.dump)
        m = Qwen4(args.path, layers=args.layers)
        got = {}
        m.forward(tok.encode(args.text), Qwen4Cache(m.cfg, 64),
                  hook=lambda k, v: got.__setitem__(k, v))
        for i in range(args.layers):
            ours, ref = float(got["layer.%d" % i].sum()), sums["l_last-%d" % i]
            r = abs(ours - ref) / max(abs(ref), 1e-6)
            print("layer %2d (%s): sum %9.4f, llama.cpp %9.4f (%.1f%%)" % (
                i, m.cfg.layer_types[i], ours, ref, 100 * r))
            ok &= r < args.tol
    m = Qwen4(args.path)
    ids = tok.encode("<|im_start|>user\nWhat is the capital of France? Answer in one sentence."
                     "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    cache = Qwen4Cache(m.cfg, len(ids) + args.tokens + 8)
    nxt = int(m.logits(m.forward(ids, cache)[-1:])[0].argmax())
    out, pos = [nxt], len(ids)
    while len(out) < args.tokens and nxt not in tok.stop_ids:
        nxt = int(m.logits(m.forward([nxt], cache, start_pos=pos))[0].argmax())
        out.append(nxt)
        pos += 1
    text = tok.decode(out)
    print(repr(text))
    ok &= "Paris" in text
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    np.seterr(over="ignore")
    raise SystemExit(main())
