#!/usr/bin/env python3
"""Measure MTP of the 26B model with the drafter on the GPU.

SPLIT_PLAN.md, phase 5. The script runs the plain greedy decode on the GPU,
then MTP with GPUDrafter for each count of --drafts. It checks that MTP
gives the tokens of the plain decode, and it gives the rate of each.

    NP_GEMMA_GPU=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/bench_mtp_gpu.py
"""
from __future__ import annotations

import argparse
import glob
import os
import time

os.environ.setdefault("NP_GEMMA_GPU", "1")

from np_gemma import KVCache, Model, gpu  # noqa: E402
from np_gemma.assistant import mtp_generate  # noqa: E402
from np_gemma.config import Config  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma.tokenizer import Tokenizer  # noqa: E402

GGUF_PATH = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"
HUB = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub"
REPO = "models--google--gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--drafter", default=None, help="The snapshot directory of the assistant.")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--drafts", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    drafter = args.drafter or glob.glob(os.path.join(HUB, REPO, "snapshots", "*"))[0]

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    msgs = [{"role": "user", "content": "Explain in a few paragraphs how a CPU cache works "
                                        "and why it matters for performance."}]
    ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, thinking=False))
    eos = set(tok.stop_ids)
    n = args.tokens

    def plain():
        cache = KVCache(cfg, max_len=len(ids) + n + 8)
        x = model.prefill(ids, cache)
        nxt = int(model.logits(x[-1:])[0].argmax())
        out, pos = [nxt], len(ids)
        t0 = time.perf_counter()
        while len(out) < n and nxt not in eos:
            x = model.forward([nxt], cache=cache, start_pos=pos)
            pos += 1
            nxt = int(model.logits(x)[0].argmax())
            out.append(nxt)
        return out, (len(out) - 1) / (time.perf_counter() - t0)

    plain()
    ref, rate = plain()
    print("tg%d GPU: %.1f tokens/s" % (n, rate), flush=True)
    dr = gpu.GPUDrafter(drafter, model)
    ok = True
    for nd in args.drafts:
        cache = KVCache(cfg, max_len=len(ids) + n + 16)
        mtp_generate(model, dr, ids, cache, n, nd, eos, {})
        cache = KVCache(cfg, max_len=len(ids) + n + 16)
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
