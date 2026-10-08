#!/usr/bin/env python3
"""A test: MTP of the 26B on the GPU, with the reuse of the experts of the
first token for the drafts of a verify group (gg_set_reuse).

SPLIT_PLAN.md, phase 5. The verify group of MTP runs the cold experts of
each of its tokens on the CPU. With reuse, a draft token selects only from
the experts of the first token and the experts that the GPU holds, so the
CPU runs only the cold experts of the first token. The result is then not
the result of the model. The script gives, for each count of drafts, the
rate, the share of drafts that the target accepts, and how many tokens are
the tokens of the plain decode.

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/bench_mtp_reuse.py --hot-gb 2
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

os.environ.setdefault("NP_GEMMA_GPU", "1")

from np_gemma import KVCache, Model, gpu, ops  # noqa: E402
from np_gemma.assistant import mtp_generate  # noqa: E402
from np_gemma.config import Config  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma.tokenizer import Tokenizer  # noqa: E402

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
HUB = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub"
REPO = "models--google--gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
PROMPTS = [
    "Explain in a few paragraphs how a CPU cache works and why it matters for performance.",
    "Write a Python function that returns the n-th Fibonacci number, then explain it.",
    "List the planets of the solar system with one fact about each.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--drafter", default=None)
    ap.add_argument("--hot-gb", type=float, default=2.0, help="GPU memory for hot experts.")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--drafts", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--keep", type=int, nargs="+", default=[0],
                    help="With reuse, each draft token also keeps its own best KEEP experts.")
    args = ap.parse_args()
    drafter = args.drafter or glob.glob(os.path.join(HUB, REPO, "snapshots", "*"))[0]
    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    dev = gpu.offload(model, args.hot_gb)
    print(gpu.describe(dev), flush=True)
    dr = gpu.GPUDrafter(drafter, model)
    # The reuse is for the verify groups only. A prompt of fewer than
    # PREFILL_MIN tokens runs as groups of up to 16 tokens, so the prompt
    # pass turns the reuse off.
    state = {"reuse": 0}
    prefill = model.prefill

    def prefill_exact(*a, **k):
        gpu.lib().gg_set_reuse(0)
        try:
            return prefill(*a, **k)
        finally:
            gpu.lib().gg_set_reuse(state["reuse"])

    model.prefill = prefill_exact
    eos = set(tok.stop_ids)
    n = args.tokens
    rows = {}
    for prompt in PROMPTS:
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                 add_generation_prompt=True, thinking=False))

        def plain():
            cache = KVCache(cfg, max_len=len(ids) + n + 8)
            x = model.prefill(ids, cache)
            nxt = ops.argmax(model.logits(x[-1:])[0])
            out, pos = [nxt], len(ids)
            t0 = time.perf_counter()
            while len(out) < n and nxt not in eos:
                x = model.forward([nxt], cache=cache, start_pos=pos)
                pos += 1
                nxt = ops.argmax(model.logits(x)[0])
                out.append(nxt)
            return out, (len(out) - 1) / (time.perf_counter() - t0)

        def forced(ids, out):
            """Run the exact model on the prompt and out. Return the share of
            the tokens of out that are the best token of the exact model, and
            the mean of log p(best) - log p(token) in nats."""
            gpu.lib().gg_set_reuse(0)
            cache = KVCache(cfg, max_len=len(ids) + len(out) + 8)
            x = prefill(ids + out[:-1], cache)[len(ids) - 1:]
            agree, gap = 0, 0.0
            for i in range(0, len(out), 16):
                lg = model.logits(x[i:i + 16])
                for r, t in zip(lg, out[i:i + 16]):
                    lp = r - r.max() - np.log(np.exp(r - r.max()).sum())
                    agree += int(ops.argmax(r) == t)
                    gap += float(lp.max() - lp[t])
            gpu.lib().gg_set_reuse(state["reuse"])
            return agree / len(out), gap / len(out)

        plain()
        ref, rate = plain()
        agree, gap = forced(ids, ref)
        rows.setdefault(("plain", 0), []).append((rate, 0, 0, len(ref), len(ref), len(ref),
                                                  agree, gap))
        print("    plain text: %r" % tok.decode(ref)[:300])
        for reuse in [0] + [1 + m for m in args.keep]:
            state["reuse"] = reuse
            gpu.lib().gg_set_reuse(reuse)
            for nd in args.drafts:
                cache = KVCache(cfg, max_len=len(ids) + n + 16)
                mtp_generate(model, dr, ids, cache, n, nd, eos, {})
                cache = KVCache(cfg, max_len=len(ids) + n + 16)
                st = {}
                out = mtp_generate(model, dr, ids, cache, n, nd, eos, st)
                m = min(len(out), len(ref))
                prefix = next((i for i in range(m) if out[i] != ref[i]), m)
                same = sum(a == b for a, b in zip(out, ref))
                agree, gap = forced(ids, out)
                rows.setdefault(("reuse top %d" % (reuse - 1) if reuse else "exact", nd), []).append(
                    (len(out) / st["decode_s"], st["accepted"], st["drafts"], prefix, same, len(out),
                     agree, gap))
                if reuse and nd == 2 and reuse == 1 + args.keep[-1]:
                    print("    reuse text: %r" % tok.decode(out)[:300])
                print("%-30.30s %-5s drafts %d: %5.1f tok/s, accepted %d/%d, same prefix %d, "
                      "same tokens %d/%d" % (prompt, "reuse top %d" % (reuse - 1) if reuse else "exact", nd,
                                             len(out) / st["decode_s"], st["accepted"],
                                             st["drafts"], prefix, same, len(out)), flush=True)
        state["reuse"] = 0
        gpu.lib().gg_set_reuse(0)
    print("\nmean over %d prompts:" % len(PROMPTS))
    for (kind, nd), v in rows.items():
        r = sum(x[0] for x in v) / len(v)
        acc = sum(x[1] for x in v) / max(1, sum(x[2] for x in v))
        pre = sum(x[3] for x in v) / len(v)
        same = sum(x[4] for x in v) / max(1, sum(x[5] for x in v))
        agree = sum(x[6] for x in v) / len(v)
        gap = sum(x[7] for x in v) / len(v)
        print("  %-11s drafts %d: %5.1f tok/s, accepted %3.0f%%, same prefix %5.1f, same tokens "
              "%3.0f%%, best token of the exact model %5.1f%%, log p gap %.3f nats"
              % (kind, nd, r, 100 * acc, pre, 100 * same, 100 * agree, gap))


if __name__ == "__main__":
    raise SystemExit(main())
