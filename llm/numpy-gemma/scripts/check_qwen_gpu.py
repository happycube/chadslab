#!/usr/bin/env python3
"""Check the decode of Qwen3.6 (GGUF) on the GPU with the experts split.

QWEN_PLAN.md, phase 4. The prompt runs on the CPU (QwenGGUFProgram). Then:

1. one step on the GPU and one on the CPU from the same cache: the logits
   and the top token;
2. the greedy answer on the GPU (HotCache on) and on the CPU: the rates,
   and the count of the same tokens at the start.

    OPENBLAS_NUM_THREADS=1 python scripts/check_qwen_gpu.py --tokens 128
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import ops  # noqa: E402
from np_gemma.qwen import QwenCache, QwenGGUFProgram  # noqa: E402
from np_gemma.qwen_gpu import QwenGPU  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TOK = "models/Qwen3.6-35B-A3B-OptiQ-4bit/tokenizer.json"
PROMPT = ("<|im_start|>user\nWrite a short Python function that checks if a number is prime, "
          "and explain how it works.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--tok", default=TOK)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--hot-gb", type=float, default=None)
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU decode")
    args = ap.parse_args()
    tok = QwenTokenizer(args.tok)
    m = QwenGGUFProgram(args.path)
    cfg = m.cfg
    ids = tok.encode(PROMPT)
    cache = QwenCache(cfg, len(ids) + args.tokens + 8)
    t0 = time.time()
    h = m.forward(ids, cache)
    print("prompt of %d tokens on the CPU: %.2f s" % (len(ids), time.time() - t0))
    first = ops.argmax(m.logits(h[-1:])[0])
    base = copy.deepcopy(cache)

    t0 = time.time()
    g = QwenGPU(m, hot_gb=args.hot_gb)
    print("QwenGPU: %.1f s, %d hot experts in each layer" % (time.time() - t0, g.n_slots))

    # 1. One step from the same cache.
    c1 = copy.deepcopy(base)
    lc = m.logits(m.forward([first], c1, start_pos=len(ids)))[0]
    c2 = copy.deepcopy(base)
    g.attach(c2)
    g.step(first, len(ids))
    lg = g.logits()
    g.detach(c2)
    r = float(np.abs(lg - lc).max() / np.abs(lc).max())
    top = np.argsort(-lc)[:5]
    print("one step: logits max rel %.2e; top token GPU %d, CPU %d; top 5 of the CPU in the "
          "top 5 of the GPU: %d" % (r, int(lg.argmax()), int(lc.argmax()),
                                    len(set(top) & set(np.argsort(-lg)[:5]))))
    for i in (0, 3):
        name = "state" if cfg.layer_types[i] != "full_attention" else "kv"
        a = c2.state[i] if name == "state" else c2.kv[i][0]
        b = c1.state[i] if name == "state" else c1.kv[i][0]
        print("  layer %d %s: max rel %.2e" % (i, name, np.abs(a - b).max() / np.abs(b).max()))
    ok = int(lg.argmax()) == int(lc.argmax())

    # 2. The greedy answers.
    def run_gpu():
        c = copy.deepcopy(base)
        g.attach(c)
        out, pos, nxt, ts = [first], len(ids), first, []
        while len(out) < args.tokens and nxt not in tok.stop_ids:
            t = time.perf_counter()
            g.step(nxt, pos)
            nxt = ops.argmax(g.logits())
            ts.append(time.perf_counter() - t)
            out.append(nxt)
            pos += 1
        g.detach(c)
        return out, ts

    out_g, ts = run_gpu()
    hc = g.hot_cache
    print("GPU: %d tokens, %.1f tok/s (median %.1f ms)%s" % (
        len(out_g), len(ts) / sum(ts), 1e3 * np.median(ts),
        "" if hc is None else "; copies %d, cold experts %.2f for each layer" % (
            hc.copies, hc.cold / max(1, hc.steps) / len(hc.layers))))
    out_g2, ts2 = run_gpu()
    print("GPU again (warm HotCache): %.1f tok/s (median %.1f ms); same tokens: %s" % (
        len(ts2) / sum(ts2), 1e3 * np.median(ts2), out_g2 == out_g))
    print(repr(tok.decode(out_g)))
    if not args.no_cpu:
        c = copy.deepcopy(base)
        out_c, pos, nxt, ts = [first], len(ids), first, []
        while len(out_c) < args.tokens and nxt not in tok.stop_ids:
            t = time.perf_counter()
            nxt = ops.argmax(m.logits(m.forward([nxt], c, start_pos=pos))[0])
            ts.append(time.perf_counter() - t)
            out_c.append(nxt)
            pos += 1
        same = 0
        while same < min(len(out_c), len(out_g)) and out_c[same] == out_g[same]:
            same += 1
        print("CPU: %.1f tok/s; the first %d of %d tokens are the same" % (
            len(ts) / sum(ts), same, len(out_c)))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
