#!/usr/bin/env python3
"""Check Qwen3.8-Flash-Next on the NVFP4 safetensors checkpoint
(np_gemma/st_qwen4.py, NVFP4Source).

1. The first --layers layers: the NumPy model (Qwen4) and the CPU program
   (Qwen4CPU) on the checkpoint: the logits and the top tokens.
2. The whole model: a chat answer on the CPU program (and with --gpu, on
   Qwen4GPU), with the rate.
3. With --gguf: the top token of each position of a text on the GGUF model
   and on the checkpoint (two quantizations of one model: they agree on most
   tokens, not on all).

    python scripts/check_qwen4_st.py --layers 4 --gpu
    NP_GEMMA_ST_DENSE=q8 python scripts/check_qwen4_st.py --layers 0 --gpu
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.qwen4 import Qwen4, Qwen4Cache, Qwen4CPU  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = "models/Qwen3.8-Flash-Next-NVFP4"
GGUF = ("models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/"
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
TEXT = ("The quick brown fox jumps over the lazy dog. In 2024, 17 * 23 = 391, "
        "and the cache of a CPU keeps recent data close to the core.")
PROMPT = ("<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--tok", default=None, help="tokenizer.json (default: the one of the checkpoint)")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--hot-gb", type=float, default=None)
    ap.add_argument("--gguf", default=None, help="compare the top tokens with this GGUF model")
    args = ap.parse_args()
    tok = QwenTokenizer(args.tok or os.path.join(args.path, "tokenizer.json"))
    stop = set(tok.stop_ids)
    ok = True

    if args.layers:
        ids = tok.encode(TEXT)
        ref_m = Qwen4(args.path, layers=args.layers)
        ref = ref_m.logits(ref_m.forward(ids, Qwen4Cache(ref_m.cfg, len(ids) + 8)))
        m = Qwen4CPU(args.path, ref_m.cfg, layers=args.layers)
        lg = m.logits(m.forward(ids, Qwen4Cache(m.cfg, len(ids) + 8)))
        r = float(np.abs(lg - ref).max() / np.abs(ref).max())
        same = float((lg.argmax(-1) == ref.argmax(-1)).mean())
        print("%d layers: CPU program against NumPy: logits max rel %.2e, same top token %.0f%%" % (
            args.layers, r, 100 * same))
        ok &= same >= 0.8

    t0 = time.time()
    m = Qwen4CPU(args.path)
    print("Qwen4CPU: dense %s" % m.g.dense)
    ids = tok.encode(PROMPT)
    cache = Qwen4Cache(m.cfg, len(ids) + args.tokens + 8)
    h = m.forward(ids, cache)
    print("the first prompt (with the repack of the experts): %.1f s" % (time.time() - t0))
    nxt, pos, out, ts = int(np.argmax(m.logits(h[-1:])[0])), len(ids), [], []
    out.append(nxt)
    while len(out) < args.tokens and nxt not in stop:
        t1 = time.time()
        nxt = int(np.argmax(m.logits(m.forward([nxt], cache, start_pos=pos))[0]))
        ts.append(time.time() - t1)
        out.append(nxt)
        pos += 1
    print("CPU: %.2f tok/s; %r" % (1 / np.median(ts), tok.decode(out)))

    if args.gpu:
        from np_gemma.qwen4_gpu import Qwen4GPU
        g = Qwen4GPU(m, hot_gb=args.hot_gb)
        cache = Qwen4Cache(m.cfg, len(ids) + args.tokens + 8)
        g.attach(cache)
        g.prefill(ids)
        nxt, pos, gout, ts = int(np.argmax(g.logits())), len(ids), [], []
        gout.append(nxt)
        while len(gout) < args.tokens and nxt not in stop:
            t1 = time.time()
            g.step(nxt, pos)
            nxt = int(np.argmax(g.logits()))
            ts.append(time.time() - t1)
            gout.append(nxt)
            pos += 1
        same = next((i for i, (a, b) in enumerate(zip(gout, out)) if a != b), min(len(gout), len(out)))
        print("GPU (%d hot experts in each layer): %.2f tok/s; the same tokens as the CPU: "
              "the first %d; %r" % (g.n_slots, 1 / np.median(ts), same, tok.decode(gout)))
        g.close()

    if args.gguf:
        ids = tok.encode(TEXT)
        top_st = m.logits(m.forward(ids, Qwen4Cache(m.cfg, len(ids) + 8))).argmax(-1)
        mg = Qwen4CPU(args.gguf)
        top_gg = mg.logits(mg.forward(ids, Qwen4Cache(mg.cfg, len(ids) + 8))).argmax(-1)
        print("the top token of each position, checkpoint against GGUF: %.0f%% the same" % (
            100 * float((top_st == top_gg).mean())))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    np.seterr(over="ignore")
    raise SystemExit(main())
