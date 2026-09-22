#!/usr/bin/env python3
"""Run a GGUF model and generate text.

The script maps the GGUF tensor names to the names of this runtime. It reads
the quantized weights directly from the GGUF. It does not build the on-disk
weight cache.

Use --tokenizer to give a tokenizer.json file. The Gemma 4 tokenizer is the
same for each model size.

Use --raw to send the prompt with no chat template. Use this mode to compare
the token ids with llama.cpp.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from np_gemma import KVCache, Model
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dtype", default="int4")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--raw", action="store_true", help="Do not use the chat template.")
    args = ap.parse_args()

    tok = Tokenizer(args.tokenizer)
    g = GGUF(args.gguf)
    cfg = Config({"text_config": g.text_config()})
    t0 = time.perf_counter()
    model = Model(g, cfg).load_all(dtype=args.dtype)
    print("load %.1f s" % (time.perf_counter() - t0), flush=True)

    if args.raw:
        ids = tok.encode(args.prompt)
    else:
        text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                       add_generation_prompt=True, thinking=False)
        ids = tok.encode(text)
    cache = KVCache(cfg, max_len=len(ids) + args.max_new_tokens + 4)
    t0 = time.perf_counter()
    x = model.prefill(ids, cache)
    prefill_s = time.perf_counter() - t0
    nxt = int(np.argmax(model.logits(x[-1:])[0]))
    out = list(ids)
    gens = []
    t0 = time.perf_counter()
    for k in range(args.max_new_tokens):
        gens.append(nxt)
        out.append(nxt)
        if k + 1 == args.max_new_tokens:
            break
        x = model.forward([nxt], cache=cache, start_pos=len(out) - 1)
        nxt = int(np.argmax(model.logits(x)[0]))
    decode_s = time.perf_counter() - t0
    print("prefill %4d tokens %7.2f s  %6.2f tok/s" % (len(ids), prefill_s, len(ids) / prefill_s), flush=True)
    print("decode  %4d tokens %7.2f s  %6.2f tok/s" % (len(gens), decode_s, len(gens) / decode_s), flush=True)
    print("prompt_ids %s" % ids)
    print("gen_ids %s" % gens)
    print("text %r" % tok.decode(gens))
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
