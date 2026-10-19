#!/usr/bin/env python3
"""The DSpark drafter of GLM-5.3-Flash on the GPU (reclank/models/glm53/dspark.py) against plain greedy
decoding: from the prompt of a trace of the reference, N greedy tokens plain (GLM53GPU.step), with 8
drafts every round (--fixed), and with the verify length the loop chooses: the tokens of plain
decoding, but for near ties (the bits of an expert differ on the GPU and on the CPU, and HotCache
moves the experts: two plain runs part at a top-2 margin of 0.30); the rates and the acceptance (the replay on the trace, scripts/
glm53_dspark_reference.py: 4.02 tokens a verify step with 8 drafts).

    RECLANK_GPU_PARTS=1 OMP_NUM_THREADS=1 RECLANK_GPU_PART_THREADS=22 RECLANK_GPU_CPU_THREADS=44 taskset -c 0-46 \\
        python scripts/check_glm53_dspark.py /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw \\
        ~/reclank-golden/glm53/trace_256_256.npz --tokens 128 \\
        --draft /home/cpage/models/GLM-5.3-Flash-speculator.dspark-preview
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("trace")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--fixed", type=int, default=8)
    ap.add_argument("--draft", default=None, help="the drafter's directory (RECLANK_GLM_DSPARK)")
    ap.add_argument("--modes", default=None, help="the drafts of each run, comma separated (c: chosen); "
                                                     "default: --fixed, c, c")
    a = ap.parse_args()
    from reclank.models.glm53.dspark import GLM53DSpark
    from reclank.models.glm53.gpu import GLM53GPU
    from reclank.models.glm53.program import GLM53Prog
    ids = np.load(a.trace)["ids"].tolist()
    ctx = len(ids) + a.tokens + 64
    t0 = time.time()
    ds = GLM53DSpark(a.draft)
    m = GLM53Prog(a.model)
    dev = GLM53GPU(m, ctx=ctx, dspark=ds)
    print("the GPU model and the drafter in %.0f s (%d hot experts a layer)" % (time.time() - t0, dev.n_slots), flush=True)
    runs = []
    for _ in range(3):          # (the first run makes the programs; the others: plain against plain)
        dev.attach(m.cache(ctx))
        dev.prefill(ids)
        lg = dev.logits(1)
        y, pos, plain, margin = int(lg.argmax()), len(ids), [], []
        t0 = time.time()
        while len(plain) < a.tokens:
            top = np.sort(lg)[-2:]
            margin.append(float(top[1] - top[0]))
            dev.step(y, pos)
            plain.append(y)
            lg = dev.logits(1)
            y, pos = int(lg.argmax()), pos + 1
        tp = time.time() - t0
        runs.append(list(plain))
    pa = next((i for i, (x, y2) in enumerate(zip(runs[1], runs[2])) if x != y2), -1)
    print("plain: %d tokens, %.1f tok/s; two plain runs %s" % (
        a.tokens, a.tokens / tp, "the same" if pa < 0 else "differ at %d (margin %.4f)" % (pa, margin[pa])), flush=True)
    ok = True
    modes = [(("%d drafts" % int(x), int(x)) if x != "c" else ("chosen", None))
             for x in (a.modes or "%d,c,c" % a.fixed).split(",")]
    for name, fixed in modes:
        dev.attach(m.cache(ctx))
        out, st = ds.generate(dev, ids, a.tokens, fixed=fixed)
        same = out == plain
        at = next((i for i, (x, y2) in enumerate(zip(out, plain)) if x != y2), -1)
        # (the experts on the GPU and on the CPU differ in their last bits, and HotCache moves them:
        # two plain runs part at a margin of 0.30; a margin of more than 0.5 would be the loop's)
        ok &= same or margin[at] < 0.5
        rounds = max(st["rounds"], 1)
        print("%s: %.1f tok/s, %s plain; %d rounds (%d plain steps), %.2f drafts accepted of %.2f, %.2f tokens a "
              "round; verify ms %s, draft %.1f ms" % (
                  name, len(out) / st["seconds"], "the tokens of" if same else
                  "differs from (at %d: plain's top-2 margin %.4f)" % (at, margin[at]), st["rounds"], st["plain"],
                  st["accepted"] / rounds, st["drafted"] / rounds, 1 + st["accepted"] / rounds, st["t_verify"],
                  st["t_draft"]), flush=True)
    print("check_glm53_dspark: %s" % ("OK" if ok else "FAILED"))
    dev.close()


if __name__ == "__main__":
    main()
