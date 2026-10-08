#!/usr/bin/env python3
"""Compare the decode step of the E4B model on the GPU with the CPU program.

SPLIT_PLAN.md, phase 3. The script runs the prompt pass on the CPU into two
caches. Then it runs decode steps of one token with the true next token as
input. The CPU program uses the first cache, and the GPU uses the second.

The script compares the hidden state after the final norm and the logits. Each side
runs its own output head. The GPU adds
the values in a different order, so the bits are not the same. The script
prints the largest difference and the share of the steps with the same most
probable token.

    OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=ACTIVE PYTHONPATH=. \\
        python scripts/check_gpu.py \\
        --gguf models2/gemma-4-E4B-unsloth-UD-Q4_K_XL/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from np_gemma import gpu, program
from np_gemma.e4b import E4B, E4BCache, E4BConfig
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models2/gemma-4-E4B-unsloth-UD-Q4_K_XL/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--contexts", type=int, nargs="+", default=[200, 1100])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--no-graph", action="store_true", help="Launch the kernels one at a time.")
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = E4BConfig({"text_config": g.text_config()})
    model = E4B(g, cfg, mode="int4")
    ids = tok.encode(open("README.md").read())
    free0, total = gpu.mem_info()
    dev = gpu.E4BGPU(model, graph=not args.no_graph)
    ok = True
    for ctx in args.contexts:
        c_cpu = E4BCache(cfg, max_len=ctx + args.steps + 8)
        c_gpu = E4BCache(cfg, max_len=ctx + args.steps + 8)
        model.forward(ids[:ctx], cache=c_cpu)
        model.forward(ids[:ctx], cache=c_gpu)
        dev.attach(c_gpu)
        dx, dl, top, t_cpu, t_gpu = [], [], [], [], []
        for k in range(args.steps):
            pos = ctx + k
            t0 = time.perf_counter()
            x1 = program.decode_step_e4b(model, c_cpu, [ids[pos]], pos)
            model.logits(x1)
            t_cpu.append(time.perf_counter() - t0)
            l1 = model.logits(x1)[0]
            t0 = time.perf_counter()
            x2 = dev.step([ids[pos]], pos, c_gpu)
            l2 = dev.logits()[0]
            t_gpu.append(time.perf_counter() - t0)
            dx.append(float(np.abs(x1 - x2).max() / np.abs(x1).max()))
            dl.append(float(np.abs(l1 - l2).max()))
            top.append(int(l1.argmax()) == int(l2.argmax()))
        dev.detach(c_gpu)
        good = min(top) and max(dl) < 0.5
        ok = ok and good
        print("context %4d, %d steps: hidden max rel %.1e, logits max |d| %.4f, same top token "
              "%d/%d. CPU %.1f ms, GPU %.1f ms (median, with the head)" % (
                  ctx, args.steps, max(dx), max(dl), sum(top), len(top),
                  1000 * np.median(t_cpu), 1000 * np.median(t_gpu)))
    free1, _ = gpu.mem_info()
    print("GPU memory of the program: %.2f GB (weights and buffers %.2f GB); free %.1f of %.1f GB"
          % ((free0 - free1) / 1e9, dev.g.mirror.nbytes() / 1e9, free1 / 1e9, total / 1e9))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
