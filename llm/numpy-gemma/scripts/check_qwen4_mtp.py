#!/usr/bin/env python3
"""Check MTP on the CPU for Qwen3.8-Flash-Next (QWEN38_PLAN.md, phase 3).

1. The MTP layer as a program (Qwen4CPU.mtp_step) against the NumPy layer
   (Qwen4.mtp), on the streams of the first --layers layers.
2. A verify group and commit(n) against steps, on --layers layers.
3. The whole model: greedy decode with MTP drafts against plain greedy
   decode. The script gives the accepted drafts and the speed.

    OPENBLAS_NUM_THREADS=1 python scripts/check_qwen4_mtp.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.qwen4 import Qwen4, Qwen4Cache, Qwen4CPU, Qwen4MTPCache  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = ("models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/"
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
MTP = "models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
TOK = "models/Qwen3.6-35B-A3B-OptiQ-4bit/tokenizer.json"
TEXT = ("The quick brown fox jumps over the lazy dog. In 2024, 17 * 23 = 391, "
        "and the cache of a CPU keeps recent data close to the core.")
PROMPT = ("<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--mtp", default=MTP)
    ap.add_argument("--tok", default=TOK)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--draft", type=int, default=3)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()
    tok = QwenTokenizer(args.tok)
    ok = True
    if args.layers:
        ids = tok.encode(TEXT)
        n = len(ids)
        m = Qwen4CPU(args.path, layers=args.layers, mtp=args.mtp)
        _xn, H = m.forward(ids, Qwen4Cache(m.cfg, n + 8), streams=True)
        Hin = np.concatenate([np.zeros((1, H.shape[1]), np.float32), H[:-1]])
        hc, hid = m.cfg.hc_count, m.cfg.hidden_size

        # 1. The MTP layer: a pass over the tokens, then two steps.
        ref_m = Qwen4(args.path, m.cfg, layers=args.layers, mtp=args.mtp)
        split = n - 2
        mc = Qwen4MTPCache(m.cfg, n + 8)
        ref = [ref_m.mtp(Hin[:split].reshape(split, hc, hid), ids[:split], mc, 0)]
        for p in range(split, n):
            ref.append(ref_m.mtp(Hin[p:p + 1].reshape(1, hc, hid), ids[p:p + 1], mc, p))
        ref = ref_m.logits(np.concatenate(ref))
        mc = Qwen4MTPCache(m.cfg, n + 8)
        got = [m.mtp_step(Hin[:split], ids[:split], mc, 0)[0]]
        for p in range(split, n):
            got.append(m.mtp_step(Hin[p:p + 1], ids[p:p + 1], mc, p)[0])
        got = m.logits(np.concatenate(got))
        same = float((got.argmax(-1) == ref.argmax(-1)).mean())
        print("MTP layer: logits max rel %.2e, same top token %.0f%%" % (rel(got, ref),
                                                                       100 * same))
        ok &= same >= 0.8

        # 2. A verify group of 4 and commit(2), then a step, against steps.
        c1 = Qwen4Cache(m.cfg, n + 8)
        m.forward(ids[:n - 5], c1)
        steps = [m.forward([ids[p]], c1, start_pos=p) for p in range(n - 5, n)]
        c2 = Qwen4Cache(m.cfg, n + 8)
        m.forward(ids[:n - 5], c2)
        xv, _Hv = m.verify(ids[n - 5:n - 1], c2, n - 5)
        m.commit(2)
        after = [m.forward([ids[p]], c2, start_pos=p) for p in range(n - 3, n)]
        r1 = rel(xv[:2], np.concatenate(steps[:2]))
        r2 = rel(np.concatenate(after), np.concatenate(steps[2:]))
        print("verify group of 4 against steps: max rel %.2e; after commit(2): %.2e" % (r1, r2))
        ok &= r1 < 2e-2 and r2 < 2e-2

    # 3. The whole model.
    m = Qwen4CPU(args.path, mtp=args.mtp)
    ids = tok.encode(args.prompt)
    stop = set(tok.stop_ids)
    m.program(1)
    m.program(args.draft + 1, "verify")
    for t in range(1, args.draft + 2):
        m.program(t, "mtp")
    cache = Qwen4Cache(m.cfg, len(ids) + args.tokens + 8)
    h = m.forward(ids, cache)
    nxt = int(np.argmax(m.logits(h[-1:])[0]))
    plain, pos = [nxt], len(ids)
    t0 = time.time()
    while len(plain) < args.tokens and nxt not in stop:
        nxt = int(np.argmax(m.logits(m.forward([nxt], cache, start_pos=pos))[0]))
        plain.append(nxt)
        pos += 1
    t_plain = (time.time() - t0) / max(1, len(plain) - 1)
    st = {}
    t0 = time.time()
    out = m.generate_mtp(ids, len(plain), draft=args.draft,
                         max_len=len(ids) + args.tokens + 16, stop=stop, stats=st)
    t_mtp = time.time() - t0
    print("plain: %.2f tok/s" % (1 / t_plain))
    print("MTP (%d drafts): %d tokens in %.1f s with the prompt; %d rounds, %d/%d drafts "
          "accepted (%.0f%%), %.2f tokens a round" % (
              args.draft, len(out), t_mtp, st["rounds"], st["accepted"], st["drafted"],
              100 * st["accepted"] / max(1, st["drafted"]), len(out) / max(1, st["rounds"])))
    print("decode: %.2f tok/s; for each round: draft %.0f ms, verify %.0f ms"
          % ((len(out) - 1) / st["decode_s"], *(1e3 * st[k] / max(1, st["rounds"])
                                                for k in ("draft", "verify"))))
    same = sum(1 for a, b in zip(out, plain) if a == b)
    first = next((i for i, (a, b) in enumerate(zip(out, plain)) if a != b), len(plain))
    print("same tokens as plain decode: %d/%d (first change at %d)" % (same, len(plain), first))
    print(repr(tok.decode(out)))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    np.seterr(over="ignore")
    raise SystemExit(main())
