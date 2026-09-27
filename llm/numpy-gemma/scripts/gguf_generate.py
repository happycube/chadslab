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
    ap.add_argument("--tokenizer", default=None,
                    help="A tokenizer.json file. The GGUF data is the default.")
    ap.add_argument("--dtype", default="int4")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--raw", "--no-chat-template", action="store_true",
                    help="Do not use the chat template.")
    ap.add_argument("--system", default=None, help="A system message for the chat template.")
    ap.add_argument("--mtp", default=None, metavar="DIR",
                    help="The snapshot directory of the Gemma 4 assistant model. The"
                         " decode then uses MTP. The token ids do not change.")
    ap.add_argument("--mtp-n", type=int, default=2, help="The count of drafts for each step.")
    ap.add_argument("--mtp-dtype", choices=("int4", "int8", "f32"), default="int4")
    ap.add_argument("--gpu", choices=("off", "dense", "hot"), default="off",
                    help="dense puts the weights outside the experts and the output head on"
                         " the GPU; the experts stay on the CPU. hot also puts the most used"
                         " experts on the GPU. Needs nvcc. MTP is then off.")
    ap.add_argument("--gpu-experts-gb", type=float, default=None,
                    help="With --gpu hot, the GPU memory for the experts. The default is the"
                         " free memory less 6 GB.")
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer(args.tokenizer) if args.tokenizer else Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    t0 = time.perf_counter()
    model = Model(g, cfg).load_all(dtype=args.dtype)
    print("load %.1f s" % (time.perf_counter() - t0), flush=True)
    if args.gpu != "off":
        from np_gemma import gpu
        t0 = time.perf_counter()
        dev = gpu.offload(model, 0.0 if args.gpu == "dense" else args.gpu_experts_gb)
        print("GPU copy %.1f s. %s" % (time.perf_counter() - t0, gpu.describe(dev)), flush=True)

    if args.raw:
        ids = tok.encode(args.prompt)
    else:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        text = tok.apply_chat_template(messages, add_generation_prompt=True, thinking=False)
        ids = tok.encode(text)
    cache = KVCache(cfg, max_len=len(ids) + args.max_new_tokens + 4)
    if args.mtp:
        from np_gemma.assistant import Assistant, mtp_generate
        drafter = Assistant(args.mtp, dtype=args.mtp_dtype)
        st = {}
        gens = mtp_generate(model, drafter, ids, cache, args.max_new_tokens,
                            args.mtp_n, set(tok.stop_ids), st)
        if gens and gens[-1] in tok.stop_ids:
            gens = gens[:-1]
        print("prefill %4d tokens %7.2f s  %6.2f tok/s" % (
            len(ids), st["prefill_s"], len(ids) / st["prefill_s"]), flush=True)
        print("decode  %4d tokens %7.2f s  %6.2f tok/s" % (
            len(gens), st["decode_s"], len(gens) / st["decode_s"]), flush=True)
        print("mtp steps %d drafts %d accepted %d (%d%%)" % (
            st["steps"], st["drafts"], st["accepted"],
            100 * st["accepted"] // max(1, st["drafts"])))
        print("prompt_ids %s" % ids)
        print("gen_ids %s" % gens)
        print("text %r" % tok.decode(gens))
        g.close()
        return 0
    t0 = time.perf_counter()
    x = model.prefill(ids, cache)
    prefill_s = time.perf_counter() - t0
    nxt = int(np.argmax(model.logits(x[-1:])[0]))
    out = list(ids)
    gens = []
    t0 = time.perf_counter()
    # Stop at an end token, or at the token limit.
    while len(gens) < args.max_new_tokens and nxt not in tok.stop_ids:
        gens.append(nxt)
        out.append(nxt)
        if len(gens) == args.max_new_tokens:
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
