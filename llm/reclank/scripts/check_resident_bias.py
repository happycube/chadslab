#!/usr/bin/env python3
"""The cost and the gain of RECLANK_GPU_RESIDENT_BIAS and _HOT_SWAP on Qwen3.8 on the GPU.

The text of bench_prompt_real (the *.md files and reclank/*.py): a prompt of
--prompt tokens, then --tokens decode steps that take the tokens of the text
(teacher forcing), so every run sees the same tokens whatever it predicts.
It prints the NLL of the text, the top token against the text, the cold
experts of a step (HotCache), and the rate of the steps; with --ref (the
--out of a run with no bias), the KL of each step against it (over the top
--top tokens of the reference), the top token the same, and the overlap of
the experts HotCache holds in its pool at the end (the share of the union
that both hold).

    GF=model.gguf EXPERTS=experts.gguf python scripts/check_resident_bias.py --out b0.npz
    GF=... EXPERTS=... RECLANK_GPU_RESIDENT_BIAS=1 python scripts/check_resident_bias.py --ref b0.npz
    GF=... EXPERTS=... RECLANK_GPU_HOT_SWAP=0.7 python scripts/check_resident_bias.py --ref b0.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reclank  # noqa: E402,F401  (the NUMA policy before NumPy)
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--lend-gb", type=float, default=1.12, help="room the encoder lends (0: none)")
    ap.add_argument("--top", type=int, default=256, help="the tokens of each step that --out keeps")
    ap.add_argument("--out", default=None, help="save the top tokens of each step (a reference)")
    ap.add_argument("--ref", default=None, help="the --out of a run to compare with")
    args = ap.parse_args()
    from reclank.backends.cuda.gpumm import Buffer
    from reclank.models.qwen.qwen38 import Qwen4Cache, Qwen4CPU
    from reclank.models.qwen import gpu38
    from reclank.models.qwen.tokenizer import QwenTokenizer
    gf = os.environ["GF"]
    tok = QwenTokenizer(os.path.join(os.path.dirname(gf), "tokenizer.json"))
    text = ""
    for f in sorted(glob.glob("*.md")) + sorted(glob.glob("reclank/*.py")):
        text += "\n\n===== %s =====\n" % f + open(f, encoding="utf-8", errors="replace").read()
    ids = tok.encode("<|im_start|>user\n" + text)[:args.prompt + args.tokens + 1]
    P, N = args.prompt, args.tokens
    lent = int(args.lend_gb * 1e9)
    ph = Buffer(lent, "lend", "encoder") if lent else None
    m = Qwen4CPU(gf, experts=os.environ.get("EXPERTS") or None)
    g = gpu38.Qwen4GPU(m, ctx=args.ctx)
    if ph is not None:
        ph.free()
        g.lend_warm(lent)
    ref = np.load(args.ref) if args.ref else None
    c = Qwen4Cache(m.cfg, args.ctx)
    g.attach(c)
    g.prefill(ids[:P])
    hc = g.hot_cache
    s0, c0 = hc.steps, hc.cold
    nll = np.zeros(N)
    hit = np.zeros(N, bool)
    kl = np.zeros(N)
    same = np.zeros(N, bool)
    top_ids = np.zeros((N, args.top), np.int32)
    top_lp = np.zeros((N, args.top), np.float32)
    dt = 0.0
    for i in range(N):
        lg = np.asarray(g.logits()).reshape(-1).astype(np.float64)
        lp = lg - lg.max()
        lp -= np.log(np.exp(lp).sum())
        want = ids[P + i]
        nll[i] = -lp[want]
        hit[i] = int(lp.argmax()) == want
        o = np.argpartition(-lp, args.top)[:args.top]
        o = o[np.argsort(-lp[o])]
        top_ids[i], top_lp[i] = o, lp[o]
        if ref is not None:
            ri, rl = ref["top_ids"][i], ref["top_lp"][i].astype(np.float64)
            kl[i] = float((np.exp(rl) * (rl - lp[ri])).sum())
            same[i] = int(ri[0]) == int(o[0])
        t0 = time.time()
        g.step(want, P + i)
        dt += time.time() - t0
    cold = (hc.cold - c0) / max(1, hc.steps - s0)
    held = np.sort(g.pool.owner[g.pool.owner >= 0]) if g.pool is not None else np.zeros(0, np.int64)
    print("bias %g swap %g: %d steps after %d tokens: NLL %.4f, top token = text %.1f%%, %.1f cold "
          "experts a step, %.1f steps/s" % (gpu38.RESIDENT_BIAS, gpu38.HOT_SWAP, N, P, nll.mean(),
                                            100 * hit.mean(), cold, N / dt), flush=True)
    if ref is not None:
        both = np.intersect1d(held, ref["held"]).size / max(1, np.union1d(held, ref["held"]).size) \
            if "held" in ref.files else float("nan")
        print("  against %s: NLL %.4f -> %.4f, KL mean %.5f median %.5f p99 %.4f max %.4f, top token the same "
              "%.1f%%, pool overlap %.1f%% (%d experts)"
              % (os.path.basename(args.ref), float(ref["nll"].mean()), nll.mean(), kl.mean(), np.median(kl),
                 np.quantile(kl, 0.99), kl.max(), 100 * same.mean(), 100 * both, held.size), flush=True)
    if args.out:
        np.savez(args.out, nll=nll, top_ids=top_ids, top_lp=top_lp, held=held)
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
