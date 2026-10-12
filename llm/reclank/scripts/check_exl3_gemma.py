#!/usr/bin/env python3
"""Gemma 4 from an EXL3 directory of ExLlamaV3 (MUL1 matrices; the weight
mode exl3 of reclank/models/gemma4/model.py): a greedy answer to a chat
prompt, and the NLL of a text (teacher forcing).

    python scripts/check_exl3_gemma.py DIR [--gpu] [--tokens 48] [--nll 512]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--nll", type=int, default=512, help="tokens of text for the NLL (0: none)")
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    args = ap.parse_args()

    from reclank.formats.exl3 import EXL3Dir
    from reclank.models.gemma4 import model as model_mod
    from reclank.models.gemma4.chat import render_chat
    from reclank.models.gemma4.config import Config
    from reclank.models.gemma4.tokenizer import Tokenizer

    st = EXL3Dir(args.path)
    tok = Tokenizer(os.path.join(args.path, "tokenizer.json"))
    cfg = Config({"text_config": st.text_config()})
    t0 = time.time()
    model = model_mod.Model(st, cfg).load_all(dtype="exl3")
    print("loaded in %.1f s" % (time.time() - t0), flush=True)
    if args.gpu:
        from reclank.models.gemma4 import gpu
        dev = gpu.offload(model, 0.0)
        print(gpu.describe(dev), flush=True)
    text = render_chat([{"role": "user", "content": args.prompt}])
    ids = tok.encode(text, add_special_tokens=False)
    sess = model_mod.Session(model, max_len=len(ids) + args.tokens + 16)
    t0 = time.time()
    out = sess.generate(ids, max_new_tokens=args.tokens, eos_ids=(1, 106))
    dt = time.time() - t0
    new = list(out)[len(ids):] if len(out) > len(ids) else list(out)
    print("answer (%d tokens, %.1f s):" % (len(new), dt))
    print(tok.decode(new))
    if args.nll:
        doc = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(os.path.dirname(HERE), "*.md"))))
        tids = [2] + tok.encode(doc)[:args.nll]
        cache = model_mod.KVCache(cfg, max_len=len(tids) + 8)
        h = model.forward(np.asarray(tids[:-1]), cache=cache)
        lg = model.logits(h)
        lg = lg - lg.max(axis=1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
        nll = -lp[np.arange(len(tids) - 1), np.asarray(tids[1:])]
        print("NLL of %d tokens of text: %.3f (ppl %.2f)" % (len(nll), nll.mean(), np.exp(nll.mean())))


if __name__ == "__main__":
    main()
