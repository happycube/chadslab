#!/usr/bin/env python3
"""GLM-5.3-Flash's indexer on the GPU (P4.3 of docs/GLM53_PLAN.md) against the NumPy model
(GLM53CPU.index_keys) on real rows: the inputs of a DSA layer from a trace of the reference
(scripts/glm53_reference.py --trace-prefill N: attn_in<i>), the index queries from the NumPy model
(idx_q_b of q_resid), then on the GPU

    F16_GLINEAR   wk x, the gate of the pools, weights_proj x (float16 matrices);
    GLM_POOL      the keys' LayerNorm, the pools of 4 (softmax of gate + ape) into the cache;
    DS_IDX_SCORE  sum over 32 heads of relu(q . pool) w / sqrt(128 x 32);
    DS_TOPK       the best --pools pools of each query (ties to the lower index);
    GLM_EXPAND    their tokens and the open tail;

for the prompt rows as one group, then steps of one row. The context of the trace (512 rows) is
below GLM's 512 pools, so the selection runs with fewer pools (--pools, the NumPy model the same).
The pooled keys and the scores in dB, the key sets of each query (the same set, or how many
differ; a tie of scores may pick another pool of the same score).

    python scripts/check_glm53_indexer.py /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw \\
        ~/reclank-golden/glm53/trace_256_256.npz --layers 3,43
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402
from reclank.ir import program as P  # noqa: E402
from reclank.models.glm53.model import GLM53CPU  # noqa: E402


def db(got, want):
    e = np.sum((np.asarray(got, np.float64) - want) ** 2)
    return 10 * np.log10(max(e, 1e-300) / np.sum(np.asarray(want, np.float64) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("trace")
    ap.add_argument("--layers", default="3")
    ap.add_argument("--prompt-rows", type=int, default=None)
    ap.add_argument("--pools", type=int, default=16, help="the pools a query keeps (GLM: 512)")
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    m = GLM53CPU(a.model)
    cfg = m.cfg
    ratio, D, NH = cfg.index_pool, cfg.index_dim, cfg.index_heads
    cfg.index_topk = a.pools * ratio
    maxk = cfg.index_topk + ratio - 1
    z = np.load(a.trace)
    T = a.prompt_rows or len(z["ids"])
    mirror = Mirror()
    worst, ok = -1e9, True
    for i in (int(x) for x in a.layers.split(",")):
        assert cfg.layer_types[i] == "dsa", i
        b = "blk.%d." % i
        x = z["attn_in%d" % i]
        N = min(len(x), T + a.steps)
        x = np.ascontiguousarray(x[:N])
        q_resid, _q, _c = m.dsa_inputs(i, x)
        qi = m.lin(b + "idx_q_b.weight", q_resid)                         # [N, 32 x 128]
        wk, wg, ww = (np.ascontiguousarray(m.g.raw(b + n)[0]) for n in ("idx_k.weight", "idx_gate.weight",
                                                                         "idx_w.weight"))
        ape, lnw, lnb = m.F(b + "idx_ape"), m.F(b + "idx_k_norm.weight"), m.F(b + "idx_k_norm.bias")
        kstate, sstate = np.zeros((ratio, D), np.float32), np.zeros((ratio, D), np.float32)
        maxn = N // ratio + 1
        pooled = np.zeros((maxn, D), np.float32)
        for w in (wk, wg, ww, ape, lnw, lnb, kstate, sstate, pooled):
            mirror.add(w)
        cache = m.cache()
        res, nsame, ndiff = [], 0, 0
        for lo, hi in [(0, T)] + [(T + j, T + j + 1) for j in range(N - T)]:
            t = hi - lo
            m.index_scores = {}
            keys = m.index_keys(i, x[lo:hi], q_resid[lo:hi], cache, lo)    # [t, hi]
            kr, gt, wt = (np.zeros((t, n), np.float32) for n in (D, D, NH))
            scores = np.full((t, maxn), -np.inf, np.float32)
            psel, pcnt = np.zeros((t, a.pools), np.int32), np.zeros(t, np.int32)
            sel, cnt = np.zeros((t, maxk), np.int32), np.zeros(t, np.int32)
            xv, qv = np.ascontiguousarray(x[lo:hi]), np.ascontiguousarray(qi[lo:hi])
            prog = P.Program()
            prog.emit(P.F16_GLINEAR, xv, wk, kr, D, cfg.hidden_size, t, 1)
            prog.emit(P.F16_GLINEAR, xv, wg, gt, D, cfg.hidden_size, t, 1)
            prog.emit(P.F16_GLINEAR, xv, ww, wt, NH, cfg.hidden_size, t, 1)
            prog.emit(P.GLM_POOL, kr, gt, kstate, sstate, ape, pooled, lo, ratio, D, 0, t, 1e-6, None, None, lnw, lnb)
            prog.emit(P.DS_IDX_SCORE, qv, wt, float((D * NH) ** -0.5), pooled, lo, ratio, None, 1, scores, NH, D,
                      maxn, t, 0, 0, 0)
            prog.emit(P.DS_TOPK, scores, lo, ratio, 1, a.pools, psel, pcnt, None, maxn, t)
            prog.emit(P.GLM_EXPAND, psel, pcnt, sel, cnt, lo, ratio, a.pools, maxk, t)
            for k, v in dict(scores=scores, sel=sel, cnt=cnt).items():
                prog.names[k] = v
            g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
            g.run()
            for k in ("scores", "sel", "cnt"):
                g.download(k)
            g.close()
            Pn = hi // ratio
            if Pn and i in m.index_scores:
                ref = m.index_scores[i]                                    # [t, Pn]
                vis = (np.arange(Pn)[None, :] * ratio + ratio - 1) <= (lo + np.arange(t))[:, None]
                res.append(db(scores[:, :Pn][vis], ref[vis]))
            for j in range(t):
                got = set(sel[j, :cnt[j]].tolist())
                want = set(np.nonzero(keys[j])[0].tolist())
                if got == want:
                    nsame += 1
                else:
                    ndiff += 1
        mirror.buffer_of(pooled).download(pooled)
        P_all = N // ratio
        lg = cache.ig[i][:P_all * ratio].reshape(P_all, ratio, -1) + ape[None]
        pr = np.exp(lg - lg.max(axis=1, keepdims=True))
        pr /= pr.sum(axis=1, keepdims=True)
        want_pool = (pr * cache.ik[i][:P_all * ratio].reshape(P_all, ratio, -1)).sum(axis=1)
        dp = db(pooled[:P_all], want_pool)
        print("layer %d: pooled keys %.1f dB; scores (worst of the groups) %.1f dB; key sets: %d the same, %d differ"
              % (i, dp, max(res), nsame, ndiff), flush=True)
        worst = max(worst, dp, max(res))
        ok &= ndiff == 0
    ok &= worst < -80
    print("check_glm53_indexer: %s (worst %.1f dB)" % ("OK" if ok else "FAILED", worst))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
