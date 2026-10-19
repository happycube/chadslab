#!/usr/bin/env python3
"""The cold experts in parts (GP_RUN_PARTS of GP_KQ_MOE_PART,
reclank/runtime/moe_parts.py) against GP_KQ_MOE, on the CPU alone: the bits,
and the time of a layer.

The experts have the shapes of Qwen3.8 (512 experts, hidden 2560, inner
640; gate and up Q6_K, down Q8_0 by default), in the memory of node 0 (as
the copy that the GGUF reader stages); the parts copy the rows of node 1.
GP_KQ_MOE runs as a GPU program runs it (rc_run_task: a team of
RECLANK_GPU_CPU_THREADS spread over the nodes, the float rows of the small
form); the parts on the first RECLANK_GPU_PART_THREADS CPUs of each node.

    RECLANK_GPU_PARTS=1 taskset -c 0-47 python scripts/check_moe_parts.py [--layers 4]

--exl3 DIR: the routed experts of the first layers of an EXL3 directory of
Qwen3.8-Flash-Next (the MUL1 blocks, KQ_E3; the parts copy the tile columns
of their outputs), int16 x as the GPU step gives it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("RECLANK_GPU_PARTS", "1")

import numpy as np  # noqa: E402

from reclank.backends.cpu import cops  # noqa: E402
from reclank.ir import program as P  # noqa: E402
from reclank.runtime import moe_parts, numa  # noqa: E402


class Mat:
    def __init__(self, data, type_, rows, cols):
        self.data, self.type, self.rows, self.cols = data, type_, rows, cols

    def c(self):
        return (self.data, self.type)


def make_layer(rng, E, hidden, inner, gtype, dtype, node, distinct=8):
    """gate, up, down of E experts: distinct experts quantized, repeated."""
    q = {12: None, 14: cops.kq_to_q6_k, 8: cops.kq_to_q8_0}
    out = []
    for rows, cols, t in ((inner, hidden, gtype), (inner, hidden, gtype), (hidden, inner, dtype)):
        if t == 53:
            from check_nvx_hot import nvx_experts
            base = nvx_experts(rng, distinct, rows, cols)
        else:
            base = np.concatenate([q[t](rng.standard_normal((rows, cols)).astype(np.float32) * 0.05, cols)
                                   for _ in range(distinct)])
        per = base.nbytes // distinct
        a = numa.empty_on(E * per, np.uint8, node)
        for e in range(0, E, distinct):
            n = min(distinct, E - e)
            a[e * per:(e + n) * per] = base[:n * per]
        out.append(Mat(a, t, rows, cols))
    return tuple(out)


def exl3_layers(path, n, node):
    """gate, up, down of the first n MoE layers of an EXL3 directory, in the
    memory of node."""
    from reclank.formats.exl3_qwen4 import EXL3Source
    src = EXL3Source(path)
    out = {}
    for i in range(src.config().num_hidden_layers):
        if len(out) == n:
            break
        names = ["blk.%d.ffn_%s_exps.weight" % (i, x) for x in ("gate", "up", "down")]
        if names[0] not in src.tensors:
            continue
        ms = []
        for name in names:
            blocks, dims, t = src.raw(name)
            a = numa.empty_on(blocks.nbytes, np.uint8, node)
            a[:] = np.ascontiguousarray(blocks).view(np.uint8).reshape(-1)
            ms.append(Mat(a, t, int(dims[1]), int(dims[0])))
        out[i] = tuple(ms)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--E", type=int, default=512)
    ap.add_argument("--k", type=int, default=6, help="cold experts of a token")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--task-threads", type=int, default=int(os.environ.get("RECLANK_GPU_CPU_THREADS", "24")))
    ap.add_argument("--nvx", action="store_true", help="NVFP4 experts (KQ_NVX) in place of Q6_K and Q8_0")
    ap.add_argument("--exl3", help="the experts of this EXL3 directory (KQ_E3)")
    ap.add_argument("--dump", help="save the outputs of GP_KQ_MOE to this .npz (a check of another build)")
    args = ap.parse_args()
    E, hidden, inner, k = args.E, 2560, 640, args.k
    rng = np.random.default_rng(0)
    t0 = time.time()
    gt, dt = (53, 53) if args.nvx else (14, 8)
    if args.exl3:
        layers = exl3_layers(args.exl3, args.layers, 0)
    else:
        layers = {i: make_layer(rng, E, hidden, inner, gt, dt, 0) for i in range(args.layers)}
    print("%d layers of %d experts in %.0f s" % (args.layers, E, time.time() - t0), flush=True)
    parts = moe_parts.ExpertParts(layers, E, hidden, inner, moe_parts.part_cpus())
    cops._lib.rc_set_task_threads(args.task_threads)
    ok = True
    dumps = {}
    for t in (1, 3):
        hf = rng.standard_normal((t, hidden)).astype(np.float32)
        val = rng.uniform(0.05, 0.2, (t, k)).astype(np.float32)
        if t == 1:
            ids = np.zeros(2 * k + 1, np.int32)
            ids[:k] = rng.choice(E, k, replace=False)
            ids[k] = k
            kcount = ids[k:]
        else:
            ids = np.stack([rng.choice(E, k, replace=False) for _ in range(t)]).astype(np.int32)
            ids[1, min(2, k - 1)] = -1      # a hot expert of a token: not on the CPU
            kcount = None
        ref, out = np.zeros((t, hidden), np.float32), np.zeros((t, hidden), np.float32)
        progs_ref, progs_par = [], []
        bufs = {}
        for i, (G, U, D) in layers.items():
            q = P.Program()
            mats = cops.kq_moe_mats(G.c(), U.c(), D.c(), None)
            q.emit(P.KQ_MOE, np.zeros((t, hidden), np.int8), np.zeros((t, hidden // 32), np.float32),
                   np.zeros((t, hidden // 16), np.float32), ids, val, t, k, E, mats, None, hidden, inner,
                   cops.kq_moe_scratch(t, k, E, hidden, inner), ref, kcount, 8 | 32, hf, None, None, None)
            q.keep.append(mats)
            progs_ref.append(q.finish())
            progs_par.append(parts.program(i, t, k, hf, ids, val, out, kcount, 32, bufs))
        for name, progs in (("KQ_MOE", progs_ref), ("parts", progs_par)):
            for q in progs[:1]:
                cops.gp_run_task(q.buf)
            if name == "parts" and os.environ.get("RECLANK_PART_PROF") == "1":
                cops.part_prof(True)
            t1 = time.time()
            for r in range(args.reps):
                for q in progs:
                    rc = cops.gp_run_task(q.buf)
                    assert rc == 0, rc
            dt = (time.time() - t1) / (args.reps * len(progs))
            print("t %d, %d cold experts a token: %-7s %7.1f us a layer" % (t, k, name, dt * 1e6), flush=True)
            if name == "parts" and os.environ.get("RECLANK_PART_PROF") == "1":
                mx, mean = cops.part_prof(True)
                nl = args.reps * len(progs)
                print("  phases, us a layer (max / mean): " + ", ".join(
                    "%s %.1f/%.1f" % (n, 1e6 * a / nl, 1e6 * b / nl) for n, a, b in zip(
                        ("A", "Aw", "pairs", "pw", "B", "Bw", "C", "Cx", "D", "Dw", "E", "Ew", "F"), mx, mean)))
        if args.dump:
            dumps["t%d" % t] = ref.copy()
        same = np.array_equal(ref.view(np.uint32), out.view(np.uint32))
        ok &= same
        print("t %d: the parts %s GP_KQ_MOE (max |diff| %.3g)" % (
            t, "give the bits of" if same else "DIFFER from", np.abs(ref - out).max()), flush=True)
    if args.dump:
        np.savez(args.dump, **dumps)
    print("check_moe_parts: %s" % ("OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
