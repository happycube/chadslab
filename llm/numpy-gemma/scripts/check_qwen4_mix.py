#!/usr/bin/env python3
"""Check the mixed groups of a prompt of Qwen3.8 on the GPU against the CPU.

A prompt of real text (the notes and sources of this project) of --tokens
runs on the CPU (Qwen4CPU.forward) and on the GPU in mixed groups of --group
rows (NP_GEMMA_GPU_MIX); the logits of the last token are compared (max rel,
the KL, the top token). The split of the experts between the GPU and the CPU
changes the values a little (the GPU products read x in float32), so the
check is the size of the difference: run it with NP_GEMMA_GPU_PREFETCH=0 and
=1 (prefetch starts with the second group) and compare. The CPU reference is
kept in --ref (made when missing).

    GF=model.gguf python scripts/check_qwen4_mix.py --ref /tmp/ref.npy
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=1536)
    ap.add_argument("--group", type=int, default=512)
    ap.add_argument("--ref", required=True, help="the logits of the CPU (an .npy, made when missing)")
    ap.add_argument("--max-kl", type=float, default=0.05)
    ap.add_argument("--ctx", type=int, default=0, help="the cache of the GPU (0: the prompt)")
    ap.add_argument("--no-graph", action="store_true", help="no CUDA graphs (to find a failing kernel)")
    ap.add_argument("--out", default=None, help="save the logits of the GPU (an .npy)")
    args = ap.parse_args()
    os.environ.setdefault("NP_GEMMA_GPU_MIX", str(args.group))
    import np_gemma  # noqa: F401
    import numpy as np
    from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU
    from np_gemma.qwen_tok import QwenTokenizer
    gf = os.environ["GF"]
    tok = QwenTokenizer(os.path.join(os.path.dirname(gf), "tokenizer.json"))
    text = ""
    for f in sorted(glob.glob("*.md")) + sorted(glob.glob("np_gemma/*.py")):
        text += "\n\n===== %s =====\n" % f + open(f, encoding="utf-8", errors="replace").read()
    ids = tok.encode("<|im_start|>user\n" + text)[:args.tokens]
    m = Qwen4CPU(gf)
    if not os.path.exists(args.ref):
        c = Qwen4Cache(m.cfg, len(ids) + 64)
        h = m.forward(ids, c, 0)
        np.save(args.ref, np.asarray(m.logits(h[-1:])[0], np.float32))
        print("the CPU reference: %s" % args.ref, flush=True)
    ref = np.load(args.ref).astype(np.float64)
    from np_gemma.qwen4_gpu import Qwen4GPU
    ctx = args.ctx or len(ids) + 64
    g = Qwen4GPU(m, ctx=ctx, graph=not args.no_graph)
    c = Qwen4Cache(m.cfg, ctx)
    g.attach(c)
    g.prefill(ids)
    got = np.asarray(g.logits(), np.float64).reshape(-1)
    if args.out:
        np.save(args.out, got.astype(np.float32))
    st = np.array([g.mix_stats[i] for i in sorted(g.mix_stats)])
    g.close()

    def lp(x):
        x = x - x.max()
        return x - np.log(np.exp(x).sum())
    pr, pg = lp(ref), lp(got)
    kl = float((np.exp(pr) * (pr - pg)).sum())
    rel = float(np.abs(got - ref).max() / np.abs(ref).max())
    print("%d tokens in groups of %d, prefetch %s: logits max rel %.3g, KL %.4f, top token %d / %d "
          "(CPU); last group: copied %.1f, prefetched %.1f (%.1f used) experts a layer" % (
              len(ids), args.group, os.environ.get("NP_GEMMA_GPU_PREFETCH", "1"), rel, kl,
              int(got.argmax()), int(ref.argmax()), st[:, 0].mean(), st[:, 6].mean(),
              st[:, 5].mean()))
    ok = kl < args.max_kl and got.argmax() == ref.argmax()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
