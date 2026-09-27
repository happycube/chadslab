#!/usr/bin/env python3
"""Compare the decode step of the 26B model on the GPU with the CPU program.

SPLIT_PLAN.md, phase 4. The GPU runs the attention, the dense feed-forward
part, the router, and the output head. The CPU runs the experts. The script
runs the prompt pass on the CPU into two caches. Then it runs decode steps of
one token with the true next token as input. The CPU program uses the first
cache, and the GPU uses the second.

The script compares the logits of each step. The GPU adds the values in a
different order. A small difference can make the router select a different
expert, and the difference then grows. Thus the script gives the share of
the steps with the same most probable token, as PERF_PLAN.md does.

    OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=ACTIVE OMP_PLACES=cores OMP_PROC_BIND=close \\
        PYTHONPATH=. python scripts/check_gpu_split.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from np_gemma import KVCache, Model, gpu, program
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--contexts", type=int, nargs="+", default=[200, 1100])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--hot", default=None,
                    help="A file of scripts/expert_use.py. The GPU then holds the most used experts.")
    ap.add_argument("--hot-gb", type=float, default=4.0, help="The memory of the hot experts.")
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    ids = tok.encode(open("README.md").read())
    free0, total = gpu.mem_info()
    t0 = time.perf_counter()
    hot = None
    if args.hot:
        f = np.load(args.hot)
        hot = gpu.pick_hot(model, sum(f[k] for k in f.files), args.hot_gb * 1e9)
    dev = gpu.ModelGPU(model, graph=not args.no_graph, hot=hot)
    print("GPU program: %.1f s to build and copy, %d hot experts" % (
        time.perf_counter() - t0, sum(len(v) for v in dev.hot.values())))
    ok = True
    for ctx in args.contexts:
        c_cpu = KVCache(cfg, max_len=ctx + args.steps + 8)
        c_gpu = KVCache(cfg, max_len=ctx + args.steps + 8)
        model.prefill(ids[:ctx], c_cpu)
        model.prefill(ids[:ctx], c_gpu)
        dev.attach(c_gpu)
        dl, top, t_cpu, t_gpu = [], [], [], []
        for k in range(args.steps):
            pos = ctx + k
            t0 = time.perf_counter()
            x1 = program.decode_step(model, c_cpu, [ids[pos]], pos)
            l1 = model.logits(x1)[0]
            t_cpu.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            dev.step([ids[pos]], pos)
            l2 = dev.logits()[0]
            t_gpu.append(time.perf_counter() - t0)
            dl.append(float(np.abs(l1 - l2).max()))
            top.append(int(l1.argmax()) == int(l2.argmax()))
        dev.detach(c_gpu)
        share = sum(top) / len(top)
        ok = ok and share >= 0.9
        print("context %4d, %d steps: logits max |d| %.3f (median %.3f), same top token "
              "%d/%d. CPU %.1f ms, GPU with the experts on the CPU %.1f ms (median, with "
              "the head)" % (ctx, args.steps, max(dl), float(np.median(dl)), sum(top),
                             len(top), 1000 * np.median(t_cpu), 1000 * np.median(t_gpu)))
        # The host cache now has the rows of the GPU steps. A CPU step on it
        # must agree with the CPU cache.
        pos = ctx + args.steps
        a = model.logits(program.decode_step(model, c_cpu, [ids[pos]], pos))[0]
        b = model.logits(program.decode_step(model, c_gpu, [ids[pos]], pos))[0]
        print("  a CPU step after detach: max |d| of the logits %.3f, same top token %s"
              % (float(np.abs(a - b).max()), int(a.argmax()) == int(b.argmax())))
    free1, _ = gpu.mem_info()
    print("GPU memory: %.2f GB (weights and buffers %.2f GB); free %.1f of %.1f GB" % (
        (free0 - free1) / 1e9, dev.g.mirror.nbytes() / 1e9, free1 / 1e9, total / 1e9))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
