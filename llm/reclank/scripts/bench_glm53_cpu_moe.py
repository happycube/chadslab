#!/usr/bin/env python3
"""The CPU experts of GLM-5.3-Flash in parts (GP_RUN_PARTS of GP_KQ_MOE_PART, reclank/runtime/
moe_parts.py), as the GPU model's steps and verify groups run them, on real rows: the inputs of the
MoE layers and the router's picks of t consecutive decode rows of a trace of the reference
(scripts/glm53_reference.py --trace-prefill N --trace-decode M: mlp_in<i>, topk<i>, topw<i>), every
expert cold (on the CPU), the experts of the weightkeeper (whole on each node) read in place.

For each t: the time of a layer (the mean over the MoE layers and --reps runs), the unique experts of
the t rows, the rate in G weights/s of the unique experts (the decode of each expert's trellis once
for its rows); with RECLANK_PART_PROF=1 the phases of moe_part.c (the most over the threads / the
mean, us a layer).

    RECLANK_GPU_PARTS=1 OMP_NUM_THREADS=1 taskset -c 0-19,24-43 python scripts/bench_glm53_cpu_moe.py \\
        /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw ~/reclank-golden/glm53/trace_256_256.npz --t 1,2,3,5,9
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RECLANK_GPU_PARTS", "1")

import numpy as np  # noqa: E402

from reclank.backends.cpu import cops  # noqa: E402
from reclank.runtime import moe_parts  # noqa: E402

PHASES = ("A", "Aw", "pairs", "pw", "B", "Bw", "C", "Cx", "D", "Dw", "E", "Ew", "F")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("trace")
    ap.add_argument("--t", default="1,2,3,5,9")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--layers", default=None, help="the MoE layers (default all)")
    ap.add_argument("--flags", type=int, default=32 | 512)
    ap.add_argument("--files", type=int, default=None,
                    help="the first N layers from the files (the stacks on huge pages), not the weightkeeper")
    a = ap.parse_args()
    from reclank.models.glm53.program import GLM53Prog
    m = GLM53Prog(a.model, layers=a.files)
    cfg = m.cfg
    E, H, I, k = cfg.num_experts, cfg.hidden_size, cfg.moe_inter, cfg.top_k
    layers = [int(x) for x in a.layers.split(",")] if a.layers else \
        [i for i in range(m.n_layers) if cfg.mlp_types[i] == "moe"]
    names = ("gate", "up", "down")
    mats = {i: tuple(m.K("blk.%d.ffn_%s_exps.weight" % (i, x)) for x in names) for i in layers}

    def on_node(i, j, nd):
        for node, arr, ids in getattr(mats[i][j], "kept", ()):
            if node == nd and len(ids) == E:
                return arr.reshape(-1)
        return None
    parts = moe_parts.ExpertParts(mats, E, H, I, moe_parts.part_cpus(), on_node=on_node)
    z = np.load(a.trace)
    T = len(z["ids"])
    per = 3 * H * I
    prof = os.environ.get("RECLANK_PART_PROF") == "1"
    for t in (int(x) for x in a.t.split(",")):
        progs, uniq = [], []
        bufs = {}
        for n, i in enumerate(layers):
            r0 = T + 8 * n % 200                       # (other rows in each layer)
            hf = np.ascontiguousarray(z["mlp_in%d" % i][r0:r0 + t], np.float32)
            sel = z["topk%d" % i][r0:r0 + t].astype(np.int32)
            val = np.ascontiguousarray(z["topw%d" % i][r0:r0 + t], np.float32)
            uniq.append(len(np.unique(sel)))
            out = np.zeros((t, H), np.float32)
            if t == 1:
                ids = np.zeros(2 * k + 1, np.int32)
                ids[:k], ids[k] = sel[0], k
                kc = ids[k:]
            else:
                ids, kc = np.ascontiguousarray(sel), None
            progs.append((parts.program(i, t, k, hf, ids, val, out, kc, a.flags, bufs), hf, ids, val, out))
        for q in progs:
            cops.gp_run_task(q[0].buf)
        if prof:
            cops.part_prof(True)
        t0 = time.time()
        for _ in range(a.reps):
            for q in progs:
                assert cops.gp_run_task(q[0].buf) == 0
        dt = (time.time() - t0) / (a.reps * len(progs))
        u = float(np.mean(uniq))
        print("t %d: %.0f us a layer (%.1f ms for %d layers), %.1f unique experts, %.0f G weights/s" % (
            t, dt * 1e6, dt * 1e3 * len(layers), len(layers), u, u * per / dt / 1e9), flush=True)
        if prof:
            mx, mean = cops.part_prof(True)
            nl = a.reps * len(progs)
            print("  phases us (most / mean): " + ", ".join(
                "%s %.0f/%.0f" % (nm, 1e6 * x / nl, 1e6 * y / nl) for nm, x, y in zip(PHASES, mx, mean)), flush=True)


if __name__ == "__main__":
    main()
