#!/usr/bin/env python3
"""The speed of GLM-5.3-Flash on the GPU (GLM53GPU, the experts from the weightkeeper when it holds
the directory): prefill of real text (the files of docs/, through the chat template) at each
--sizes, twice (the first run compiles the programs), then --steps greedy decode steps at that
depth (the last half timed).

    RECLANK_GPU_PARTS=1 OMP_NUM_THREADS=1 RECLANK_GPU_PART_THREADS=22 RECLANK_GPU_CPU_THREADS=44 \\
        taskset -c 0-46 python scripts/bench_glm53.py \\
        /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw --sizes 2048,8192
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def text_ids(path, n):
    """n tokens of the docs of this repository (real text), as a user message of the template."""
    from reclank.models.qwen.tokenizer import QwenTokenizer
    tok = QwenTokenizer(os.path.join(path, "tokenizer.json"))
    doc = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, "docs", "*.md"))))
    ids = tok.encode(doc)
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sizes", default="2048,8192")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--runs", type=int, default=2, help="prefill runs at each size (the first compiles)")
    a = ap.parse_args()
    from reclank.models.glm53.gpu import GLM53GPU
    from reclank.models.glm53.program import GLM53Prog
    sizes = [int(x) for x in a.sizes.split(",")]
    ctx = max(sizes) + a.steps + 16
    t0 = time.time()
    m = GLM53Prog(a.model)
    dev = GLM53GPU(m, ctx=ctx)
    print("the GPU model in %.0f s (%d hot experts a layer)" % (time.time() - t0, dev.n_slots), flush=True)
    for n in sizes:
        ids = text_ids(a.model, n)
        for run in range(a.runs):
            cache = m.cache(ctx)
            dev.attach(cache)
            t0 = time.time()
            dev.prefill(ids)
            tok = int(dev.argmax(1)[0])
            dt = time.time() - t0
            print("PREFILL %d tokens run %d: %.2f s (%.0f tok/s)" % (n, run, dt, n / dt), flush=True)
        t1, half = None, a.steps // 2
        for j in range(a.steps):
            if j == half:
                t1 = time.time()
            dev.step(tok, n + j)
            tok = int(dev.argmax(1)[0])
        print("DECODE at %d: last half %.1f tok/s" % (n, (a.steps - half) / (time.time() - t1)), flush=True)
        from reclank.backends.cuda.gpumm import mem_info
        print("GPU memory free: %.2f GB" % (mem_info()[0] / 1e9), flush=True)
    dev.close()


if __name__ == "__main__":
    main()
