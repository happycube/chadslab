#!/usr/bin/env python3
"""The routing profile of GLM-5.3-Flash on the GPU: the uses of each routed expert in the decode steps
of real text (HotCache.profile: (layers + 1, experts) counts), saved as an .npy, the first hot
experts of a later start (GLM53GPU reads <model>/expert_profile.npy, or RECLANK_EXPERT_COUNTS).

The text: the files of this repository (the docs, the Python and C sources), --prompts of them, each
a user message of about --tokens tokens of one file (GLM's template: Reasoning Effort: Max, the
answer opened with <think>), and --steps greedy decode steps after it. --exclude: files not to use
(a held-out text for the measure of the profile).

    RECLANK_GPU_PARTS=1 OMP_NUM_THREADS=1 RECLANK_GPU_PART_THREADS=22 RECLANK_GPU_CPU_THREADS=44 \\
        taskset -c 0-46 python scripts/glm53_expert_profile.py /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw \\
        --exclude docs/MOE_PLAN.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

TEMPLATE = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>%s<|assistant|><think>"
ASKS = ("Explain what this file does and how it works.", "Summarize the main points of this document.",
        "Review this code: what could go wrong?", "What would you change in this design, and why?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default=None, help="default: <model>/expert_profile.npy")
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=192)
    ap.add_argument("--exclude", default="", help="files not to use, comma separated (paths in the repo)")
    a = ap.parse_args()
    from reclank.models.glm53.gpu import GLM53GPU
    from reclank.models.glm53.program import GLM53Prog
    from reclank.models.qwen.tokenizer import QwenTokenizer
    tok = QwenTokenizer(os.path.join(a.model, "tokenizer.json"))
    skip = {os.path.join(ROOT, x) for x in a.exclude.split(",") if x}
    files = [f for pat in ("docs/*.md", "reclank/**/*.py", "reclank/backends/**/*.c", "reclank/backends/**/*.cu",
                           "scripts/*.py")
             for f in sorted(glob.glob(os.path.join(ROOT, pat), recursive=True))
             if f not in skip and os.path.getsize(f) > 8000]
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.choice(len(files), min(a.prompts, len(files)), replace=False)]
    ctx = a.tokens + a.steps + 256
    m = GLM53Prog(a.model)
    dev = GLM53GPU(m, ctx=ctx, counts=False)
    t0 = time.time()
    for n, f in enumerate(files):
        body = tok.decode(tok.encode(open(f).read())[:a.tokens])
        text = TEMPLATE % ("%s\n\n%s" % (body, ASKS[n % len(ASKS)]))
        ids = tok.encode(text)
        dev.attach(m.cache(ctx))
        dev.prefill(ids)
        y, pos = int(dev.argmax(1)[0]), len(ids)
        for _ in range(a.steps):
            dev.step(y, pos)
            y, pos = int(dev.argmax(1)[0]), pos + 1
        print("%d/%d %s: %d tokens, %d steps (%.0f s)" % (n + 1, len(files), os.path.relpath(f, ROOT), len(ids),
                                                         a.steps, time.time() - t0), flush=True)
    prof = dev.hot_cache.profile()
    out = a.out or os.path.join(a.model, "expert_profile.npy")
    np.save(out, prof)
    print("written %s: %s, %d uses" % (out, prof.shape, int(prof.sum())))
    dev.close()


if __name__ == "__main__":
    main()
