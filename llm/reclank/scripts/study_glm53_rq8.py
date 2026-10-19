#!/usr/bin/env python3
"""A study of RQ8_0 experts for GLM-5.3-Flash (the rotated Q8_0 of docs/RQ8_EXPERTS_PLAN.md, from the
original FP8 checkpoint) against turboderp's EXL3 4.05bpw experts: the quality and the speed of the
CPU parts, on real rows of a trace of the reference (scripts/glm53_reference.py --trace-prefill N
--trace-decode M: mlp_in<i>, topk<i>, topw<i>).

--quality LAYERS: for 32 decode rows of each layer, the experts their routing uses, read from the FP8
checkpoint (E4M3 times the 128 x 128 scales: the reference's weights); each matrix as RQ8_0 (the
TQ6 rotation of each 32, Q8_0, back), as plain Q8_0, and as the EXL3 matrix decoded: the error
against FP8 (dB), and the routed experts' output of the rows (float math, the clamped SwiGLU, the
router's weights) against that of the FP8 weights.

--speed LAYERS: all 288 experts of those layers from the FP8 checkpoint to Q8_0 (KQ_Q8_0 stacks on
huge pages: the products of RQ8_0 are those of Q8_0, x and the act rotated before), the parts as
the GPU model's steps and verify groups run them (GP_KQ_MOE_PART, each node its rows: the rows of
node 1 copied to node 1), every expert cold, t rows of the trace: the time of a layer, as
scripts/bench_glm53_cpu_moe.py measures the EXL3 experts.

    python scripts/study_glm53_rq8.py /space/models/GLM-5.3-Flash /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw \\
        ~/reclank-golden/glm53/trace_256_256.npz --quality 3,20,44
    RECLANK_GPU_PARTS=1 OMP_NUM_THREADS=1 taskset -c 0-46 python scripts/study_glm53_rq8.py ... --speed 3,4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault("RECLANK_GPU_PARTS", "1")

import numpy as np  # noqa: E402

from reclank.backends.cpu import cops  # noqa: E402

MAIN = "model.language_model."
KQ_Q8_0 = 8


def db(got, want):
    e = np.sum((np.asarray(got, np.float64) - want) ** 2)
    return 10 * np.log10(max(e, 1e-300) / np.sum(np.asarray(want, np.float64) ** 2))


def q8_back(w, rotate):
    """The values of w (rows x cols) as Q8_0 (rotate: in the TQ6 rotation of each 32, then back)."""
    rows, cols = w.shape
    a = cops.tq6_rotate(w) if rotate else np.ascontiguousarray(w, np.float32)
    q = cops.kq_to_q8_0(a.reshape(rows, cols), cols)
    v = cops.kq_rows(q, KQ_Q8_0, cols, np.arange(rows)).reshape(rows, cols)
    return cops.tq6_rotate(v, inverse=True).reshape(rows, cols) if rotate else v


def swiglu(g, u, lim=10.0):
    g = np.minimum(g, lim)
    return g / (1 + np.exp(-g)) * np.clip(u, -lim, lim)


def quality(orig, src, z, layers, nrows=32):
    T = len(z["ids"])
    for i in layers:
        x = z["mlp_in%d" % i][T:T + nrows].astype(np.float64)
        ids, w = z["topk%d" % i][T:T + nrows].astype(np.int64), z["topw%d" % i][T:T + nrows].astype(np.float64)
        used = np.unique(ids)
        outs = {k: np.zeros((nrows, x.shape[1])) for k in ("fp8", "rq8", "q8", "exl3")}
        mats = {k: [] for k in ("rq8", "q8", "exl3")}
        t0 = time.time()
        for e in used:
            W = {}
            for part in ("gate", "up", "down"):
                f = orig.get(MAIN + "layers.%d.mlp.experts.%d.%s_proj.weight" % (i, e, part))
                ex = src.dequant("blk.%d.ffn_%s_exps.weight" % (i, part), rows=[int(e)])[0]
                W[part] = {"fp8": f, "rq8": q8_back(f, True), "q8": q8_back(f, False), "exl3": ex}
                for k in mats:
                    mats[k].append(db(W[part][k], f))
            tok, slot = np.nonzero(ids == e)
            for k in outs:
                a = swiglu(x[tok] @ W["gate"][k].T.astype(np.float64), x[tok] @ W["up"][k].T.astype(np.float64))
                outs[k][tok] += (a @ W["down"][k].T.astype(np.float64)) * w[tok, slot][:, None]
        print("layer %d: %d experts of %d rows (%.0f s); matrices against FP8 (mean dB): RQ8_0 %.1f, Q8_0 %.1f, "
              "EXL3 %.1f; the experts' output against FP8: RQ8_0 %.1f dB, Q8_0 %.1f dB, EXL3 %.1f dB" % (
                  i, len(used), nrows, time.time() - t0, np.mean(mats["rq8"]), np.mean(mats["q8"]),
                  np.mean(mats["exl3"]), db(outs["rq8"], outs["fp8"]), db(outs["q8"], outs["fp8"]),
                  db(outs["exl3"], outs["fp8"])), flush=True)


class _Mat:
    def __init__(self, data, rows, cols):
        self.data, self.type, self.rows, self.cols, self.rot = data, KQ_Q8_0, rows, cols, False

    def c(self):
        return (self.data, self.type)


def speed(orig, cfg, z, layers, ts, reps=10, flags=512):
    from reclank.runtime import moe_parts, numa
    E, H, I, k = cfg.num_experts, cfg.hidden_size, cfg.moe_inter, cfg.top_k
    mats = {}
    t0 = time.time()
    for i in layers:
        ms = []
        for part, (rows, cols) in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
            rb = cols // 32 * 34
            st = numa.empty_huge(E * rows * rb).reshape(E, rows * rb)
            for e in range(E):
                f = orig.get(MAIN + "layers.%d.mlp.experts.%d.%s_proj.weight" % (i, e, part))
                st[e] = cops.kq_to_q8_0(np.ascontiguousarray(f, np.float32), cols)
            ms.append(_Mat(st.reshape(-1), rows, cols))
        mats[i] = tuple(ms)
        print("layer %d: Q8_0 stacks in %.0f s (%.1f GB)" % (i, time.time() - t0, sum(m.data.nbytes for m in ms) / 1e9),
              flush=True)
    parts = moe_parts.ExpertParts(mats, E, H, I, moe_parts.part_cpus())
    T = len(z["ids"])
    prof = os.environ.get("RECLANK_PART_PROF") == "1"
    per = 3 * H * I
    for t in ts:
        progs, uniq, bufs = [], [], {}
        for n, i in enumerate(layers):
            r0 = T + 8 * n % 200
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
            progs.append((parts.program(i, t, k, hf, ids, val, out, kc, flags, bufs), hf, ids, val, out))
        for q in progs:
            cops.gp_run_task(q[0].buf)
        if prof:
            cops.part_prof(True)
        t1 = time.time()
        for _ in range(reps):
            for q in progs:
                assert cops.gp_run_task(q[0].buf) == 0
        dt = (time.time() - t1) / (reps * len(progs))
        u = float(np.mean(uniq))
        print("Q8_0 t %d: %.0f us a layer (%.1f ms for 42 layers), %.1f unique experts, %.0f G weights/s, %.0f GB/s" % (
            t, dt * 1e6, dt * 42e3, u, u * per / dt / 1e9, u * per * 34 / 32 / dt / 1e9), flush=True)
        if prof:
            mx, mean = cops.part_prof(True)
            nl = reps * len(progs)
            names = ("A", "Aw", "pairs", "pw", "B", "Bw", "C", "Cx", "D", "Dw", "E", "Ew", "F")
            print("  phases us (most / mean): " + ", ".join(
                "%s %.0f/%.0f" % (nm, 1e6 * a / nl, 1e6 * b / nl) for nm, a, b in zip(names, mx, mean)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig", help="the FP8 checkpoint")
    ap.add_argument("exl3", help="the EXL3 directory")
    ap.add_argument("trace")
    ap.add_argument("--quality", default=None)
    ap.add_argument("--speed", default=None)
    ap.add_argument("--t", default="1,2,5,9")
    a = ap.parse_args()
    from check_glm53_source import Orig
    from reclank.formats.exl3_glm53 import GLM53Source
    orig = Orig(a.orig)
    src = GLM53Source(a.exl3, mtp=False)
    z = np.load(a.trace)
    if a.quality:
        quality(orig, src, z, [int(x) for x in a.quality.split(",")])
    if a.speed:
        speed(orig, src.config(), z, [int(x) for x in a.speed.split(",")], [int(x) for x in a.t.split(",")])


if __name__ == "__main__":
    main()
