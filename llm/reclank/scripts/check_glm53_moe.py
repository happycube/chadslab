#!/usr/bin/env python3
"""The CPU ops of GLM-5.3-Flash's MoE (P3 of docs/GLM53_PLAN.md) on real rows: the inputs of the
MoE layers and the router's picks from a trace of the reference (scripts/glm53_reference.py
--trace-prefill N --trace-decode M: mlp_in<i>, topk<i>, topw<i>), the EXL3 experts of the
directory.

- GP_KQ_MOE (kq_e3_moe_body, int16 x: act 8 | 32) with act bit 512, the clamped SwiGLU
  silu(min(g, 10)) clamp(u, -10, 10), against the float32 experts (the decoded weights, the clamp
  in NumPy): dB; and without bit 512, to show what the clamp changes on these rows;
- the parts (GP_RUN_PARTS of GP_KQ_MOE_PART, flags 32 | 512) against GP_KQ_MOE: the same bits;
- the shared expert without a logit (weight 1, ungated): GP_KQ_MOE with the shared matrices and
  no shared_logit, against the float32 routed + shared experts.

The rows: one decode row (t 1, kcount: the cold experts of a GPU step) and a group of 64.

    RECLANK_GPU_PARTS=1 taskset -c 0-47 python scripts/check_glm53_moe.py \\
        /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw ~/reclank-golden/glm53/trace_256_256.npz --layers 3,20
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("RECLANK_GPU_PARTS", "1")

import numpy as np  # noqa: E402

from reclank.backends.cpu import cops  # noqa: E402
from reclank.ir import program as P  # noqa: E402
from reclank.models.glm53.model import swiglu  # noqa: E402
from reclank.runtime import moe_parts, numa  # noqa: E402


class Mat:
    def __init__(self, data, type_, rows, cols):
        self.data, self.type, self.rows, self.cols = data, type_, rows, cols

    def c(self):
        return (self.data, self.type)


def db(got, want):
    e = np.sum((got.astype(np.float64) - want) ** 2)
    return 10 * np.log10(max(e, 1e-300) / np.sum(want.astype(np.float64) ** 2))


def mat(src, name, node=0):
    blocks, dims, t = src.raw(name)
    a = numa.empty_on(blocks.nbytes, np.uint8, node)
    a[:] = np.ascontiguousarray(blocks).view(np.uint8).reshape(-1)
    src.drop(name)
    return Mat(a, t, int(dims[1]), int(dims[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("trace")
    ap.add_argument("--layers", default="3")
    ap.add_argument("--task-threads", type=int, default=int(os.environ.get("RECLANK_GPU_CPU_THREADS", "24")))
    a = ap.parse_args()
    from reclank.formats.exl3_glm53 import GLM53Source
    src = GLM53Source(a.model)
    cfg = src.config()
    E, H, I, k = cfg.num_experts, cfg.hidden_size, cfg.moe_inter, cfg.top_k
    z = np.load(a.trace)
    T = len(z["ids"])
    cops._lib.rc_set_task_threads(a.task_threads)
    layers = [int(x) for x in a.layers.split(",")]
    mats = {i: tuple(mat(src, "blk.%d.ffn_%s_exps.weight" % (i, x)) for x in ("gate", "up", "down")) for i in layers}
    parts = moe_parts.ExpertParts(mats, E, H, I, moe_parts.part_cpus())
    ok = True
    for i in layers:
        G, U, D = mats[i]
        b = "blk.%d." % i
        sh = tuple(mat(src, b + "ffn_%s_shexp.weight" % x) for x in ("gate", "up", "down"))
        wsh = [src.dequant(b + "ffn_%s_shexp.weight" % x) for x in ("gate", "up", "down")]
        xs, ids_all, w_all = z["mlp_in%d" % i], z["topk%d" % i].astype(np.int32), z["topw%d" % i]
        for name, rows in (("decode row", [T + 5 if len(xs) > T + 5 else T]), ("64 prompt rows", list(range(64)))):
            t = len(rows)
            hf = np.ascontiguousarray(xs[rows], np.float32)
            ids, val = np.ascontiguousarray(ids_all[rows]), np.ascontiguousarray(w_all[rows], np.float32)
            # the float32 experts
            ref = np.zeros((t, H), np.float32)
            nobig = np.zeros((t, H), np.float32)
            big = [0, 0]
            for e in np.unique(ids):
                tok, slot = np.nonzero(ids == e)
                wg, wu, wd = (src.dequant(b + "ffn_%s_exps.weight" % x, rows=[int(e)])[0] for x in ("gate", "up", "down"))
                g, u = hf[tok] @ wg.T, hf[tok] @ wu.T
                big[0] += int((g > cfg.swiglu_limit).sum() + (np.abs(u) > cfg.swiglu_limit).sum())
                big[1] += g.size
                ref[tok] += (swiglu(g, u, cfg.swiglu_limit) @ wd.T) * val[tok, slot][:, None]
                nobig[tok] += ((g / (1 + np.exp(-g)) * u) @ wd.T) * val[tok, slot][:, None]
            shared = swiglu(hf @ wsh[0].T, hf @ wsh[1].T, cfg.swiglu_limit) @ wsh[2].T
            if t == 1:
                kids = np.zeros(2 * k + 1, np.int32)
                kids[:k] = ids[0]
                kids[k] = k
                kcount, ids_in = kids[k:], kids
            else:
                kcount, ids_in = None, ids

            def kq_moe(act, with_shared=False):
                out = np.zeros((t, H), np.float32)
                q = P.Program()
                m = cops.kq_moe_mats(G.c(), U.c(), D.c(), tuple(x.c() for x in sh) if with_shared else None)
                q.emit(P.KQ_MOE, np.zeros((t, H), np.int8), np.zeros((t, H // 32), np.float32),
                       np.zeros((t, H // 16), np.float32), ids_in, val, t, k, E, m, None, H, I,
                       cops.kq_moe_scratch(t, k, E, H, I), out, kcount, act, hf, None, None, None)
                q.keep.append(m)
                assert cops.gp_run_task(q.finish().buf) == 0
                return out
            whole = kq_moe(8 | 32 | 512)
            plain = kq_moe(8 | 32)
            out = np.zeros((t, H), np.float32)
            prog = parts.program(i, t, k, hf, ids_in, val, out, kcount, 32 | 512, {})
            assert cops.gp_run_task(prog.buf) == 0
            same = np.array_equal(whole.view(np.uint32), out.view(np.uint32))
            wsh_out = kq_moe(8 | 32 | 512, with_shared=True)
            d_whole, d_sh = db(whole, ref), db(wsh_out, ref + shared)
            print("layer %d, %s: %d of %d gate/up values past 10; GP_KQ_MOE (bit 512) against float %.1f dB "
                  "(without bit 512: %.1f dB; float without the clamp vs with: %.1f dB); parts %s; "
                  "with the ungated shared expert %.1f dB" % (
                      i, name, big[0], 2 * big[1], d_whole, db(plain, ref), db(nobig, ref),
                      "the same bits" if same else "DIFFER (max %.3g)" % np.abs(whole - out).max(), d_sh), flush=True)
            ok &= same and d_whole < -40 and d_sh < -40
    print("check_glm53_moe: %s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
