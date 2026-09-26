#!/usr/bin/env python3
"""Compare the MTP decode with the plain greedy decode of the 26B model.

For each prompt, the script runs the plain greedy decode and the MTP decode
with the assistant drafter. It prints the decode rate of each, the drafts, the
accepted drafts, and a check that the token ids are the same. The prompts are
the prompts of scripts/bench_mtp_llamacpp.py.

    OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=ACTIVE PYTHONPATH=. \\
        python scripts/check_mtp.py --n-draft 3
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from np_gemma import KVCache, Model
from np_gemma.assistant import Assistant, mtp_generate
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"
HUB = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub"
REPO = "models--google--gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"

PROMPTS = [
    ("code", "Write a Python function that returns the n-th Fibonacci number, "
             "then explain it in two sentences."),
    ("prose", "Explain why the sky is blue in one paragraph."),
    ("list", "List the planets of the solar system with one fact about each."),
    ("math", "Solve 17 * 23 step by step, then check the result by division."),
]


def plain(model, ids, cache, max_new, eos):
    t0 = time.perf_counter()
    x = model.prefill(ids, cache)
    t1 = time.perf_counter()
    nxt = int(np.argmax(model.logits(x[-1:])[0]))
    out = []
    pos = len(ids)
    while True:
        out.append(nxt)
        if nxt in eos or len(out) >= max_new:
            break
        x = model.forward([nxt], cache=cache, start_pos=pos)
        pos += 1
        nxt = int(np.argmax(model.logits(x)[0]))
    return out, t1 - t0, time.perf_counter() - t1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--assistant", default=None, help="The snapshot directory.")
    ap.add_argument("--drafter-dtype", default="int4")
    ap.add_argument("--n-draft", type=int, nargs="+", default=[3])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--prompts", nargs="+", default=[p[0] for p in PROMPTS])
    args = ap.parse_args()
    path = args.assistant or sorted(glob.glob(os.path.join(HUB, REPO, "snapshots", "*")))[-1]

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    drafter = Assistant(path, dtype=args.drafter_dtype)
    eos = set(tok.stop_ids)

    print("%-6s %4s %8s %8s %6s %7s %9s  %s" % ("prompt", "n", "plain", "mtp", "gain",
                                               "drafts", "accepted", "same ids"))
    tot = {}
    for name, text in PROMPTS:
        if name not in args.prompts:
            continue
        msg = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, thinking=False)
        ids = tok.encode(msg)
        cache = KVCache(cfg, max_len=len(ids) + args.max_new_tokens + 16)
        ref, _p, dt_plain = plain(model, ids, cache, args.max_new_tokens, eos)
        r_plain = (len(ref) - 1) / dt_plain
        for n in args.n_draft:
            cache = KVCache(cfg, max_len=len(ids) + args.max_new_tokens + 16)
            st = {}
            out = mtp_generate(model, drafter, ids, cache, args.max_new_tokens, n, eos, st)
            r_mtp = (len(out) - 1) / st["decode_s"]
            same = "yes" if out == ref else "NO (first diff at %d)" % next(
                (i for i, (a, b) in enumerate(zip(out, ref)) if a != b), min(len(out), len(ref)))
            print("%-6s %4d %8.2f %8.2f %6.2f %7d %4d (%2d%%)  %s" % (
                name, n, r_plain, r_mtp, r_mtp / r_plain, st["drafts"], st["accepted"],
                100 * st["accepted"] // max(1, st["drafts"]), same), flush=True)
            a = tot.setdefault(n, [0, 0.0, 0, 0.0, 0, 0])
            a[0] += len(ref) - 1
            a[1] += dt_plain
            a[2] += len(out) - 1
            a[3] += st["decode_s"]
            a[4] += st["drafts"]
            a[5] += st["accepted"]
    for n, a in tot.items():
        rp, rm = a[0] / a[1], a[2] / a[3]
        print("%-6s %4d %8.2f %8.2f %6.2f %7d %4d (%2d%%)" % (
            "ALL", n, rp, rm, rm / rp, a[4], a[5], 100 * a[5] // max(1, a[4])))
    g.close()


if __name__ == "__main__":
    main()
