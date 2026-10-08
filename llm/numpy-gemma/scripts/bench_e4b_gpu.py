#!/usr/bin/env python3
"""Measure the E4B model on the GPU: the prompt pass, the decode, and MTP.

SPLIT_PLAN.md, phase 5. The script measures:

- the prompt pass of --prompt tokens on the CPU and on the GPU, with the
  tensor cores and with the float32 kernels;
- the plain decode on the GPU (128 tokens, greedy);
- MTP with the drafter on the GPU (GPUDrafter), for each count of --drafts.

It checks that MTP gives the tokens of the plain decode. Compare the values
with llama-bench of llama.cpp (build-cuda, -ngl 99).

    NP_GEMMA_GPU=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \\
        python scripts/bench_e4b_gpu.py --drafter ASSISTANT_DIR
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("NP_GEMMA_GPU", "1")

from np_gemma import gpu, ops  # noqa: E402
from np_gemma.assistant import mtp_generate  # noqa: E402
from np_gemma.e4b import E4B, E4BCache, E4BConfig  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma.tokenizer import Tokenizer  # noqa: E402

GGUF_PATH = "models2/gemma-4-E4B-unsloth-UD-Q4_K_XL/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--drafter", default=None, help="The snapshot directory of the assistant.")
    ap.add_argument("--prompt", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--drafts", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--cpu", action="store_true", help="Also time the prompt pass on the CPU.")
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = E4BConfig({"text_config": g.text_config()})
    model = E4B(g, cfg, mode="int4")
    text = tok.encode(open(os.path.join(os.path.dirname(__file__), "..", "README.md")).read())

    dev = None
    for n in args.prompt:
        ids = text[:n]
        if args.cpu:
            import np_gemma.e4b as e4b_mod
            e4b_mod._GPU = False
            c = E4BCache(cfg, max_len=n + 8)
            t0 = time.perf_counter()
            model.forward(ids, cache=c)
            print("pp%d CPU: %.0f tokens/s" % (n, n / (time.perf_counter() - t0)), flush=True)
            e4b_mod._GPU = True
        for tc in (1, 0):
            gpu.lib().gg_set_tc(tc)
            dev = gpu.E4BGPU(model)       # a new runner, so the graphs use this setting
            rates = []
            for rep in range(3):
                c = E4BCache(cfg, max_len=n + 8)
                dev._ensure(c)
                dev.attach(c)
                t0 = time.perf_counter()
                dev.prefill(ids, 0, c)
                gpu.lib().gg_sync()
                rates.append(n / (time.perf_counter() - t0))
            print("pp%d GPU %s: %.0f tokens/s" % (n, "tensor cores" if tc else "float32",
                                                  max(rates[1:])), flush=True)
    gpu.lib().gg_set_tc(1)

    msgs = [{"role": "user", "content": "Explain in a few paragraphs how a CPU cache works "
                                        "and why it matters for performance."}]
    ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, thinking=False))
    eos = set(tok.stop_ids)
    n = args.tokens

    def plain():
        cache = E4BCache(cfg, max_len=len(ids) + n + 8)
        x = model.forward(ids, cache=cache)
        nxt = ops.argmax(model.logits(x[-1:])[0])
        out, pos = [nxt], len(ids)
        t0 = time.perf_counter()
        while len(out) < n and nxt not in eos:
            x = model.forward([nxt], cache=cache, start_pos=pos)
            pos += 1
            nxt = ops.argmax(model.logits(x)[0])
            out.append(nxt)
        return out, (len(out) - 1) / (time.perf_counter() - t0)

    plain()
    ref, rate = plain()
    print("tg%d GPU: %.1f tokens/s" % (n, rate), flush=True)
    if not args.drafter:
        return 0
    dr = gpu.GPUDrafter(args.drafter, model)
    ok = True
    for nd in args.drafts:
        cache = E4BCache(cfg, max_len=len(ids) + n + 16)
        mtp_generate(model, dr, ids, cache, n, nd, eos, {})
        cache = E4BCache(cfg, max_len=len(ids) + n + 16)
        st = {}
        out = mtp_generate(model, dr, ids, cache, n, nd, eos, st)
        same = out == ref
        ok = ok and same
        print("tg%d GPU, MTP with %d drafts: %.1f tokens/s, %d of %d drafts accepted (%d%%), "
              "same tokens: %s" % (n, nd, len(out) / st["decode_s"], st["accepted"], st["drafts"],
                                   100 * st["accepted"] // max(1, st["drafts"]), same), flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
