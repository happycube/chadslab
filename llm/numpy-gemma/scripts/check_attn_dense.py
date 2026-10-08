#!/usr/bin/env python3
"""Check GP_ATTN_QSA of a large group whose queries all see every position
before them (maxsel 0: k_attn_dense_tc, csrc/gpu.cu; the dense attention of
the MTP layer) against k_attn_qsa_tc (the same record with a maxsel) and a
float64 reference, on an int8 cache of random keys and values (Qwen3.8: 24
query heads, 2 key heads, head_dim 256), and the time of both kernels.

    python scripts/check_attn_dense.py [--pos 4000,200000] [--rows 256,61]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from np_gemma import program as P  # noqa: E402
from np_gemma.gpu import GPUProgram, Mirror  # noqa: E402

NQ, NK, HD = 24, 2, 256


def cache(rng, n):
    """int8 rows of n positions (NK * HD values) and their scales (each 32)."""
    x = rng.standard_normal((n, NK * HD)).astype(np.float32)
    s = np.abs(x.reshape(n, -1, 32)).max(-1) / 127.0
    q = np.round(x.reshape(n, -1, 32) / s[..., None]).astype(np.int8).reshape(n, -1)
    return q, s.astype(np.float32)


def run(q, kq, ks, vq, vs, pos, maxsel, mirror, reps=0, rows=None):
    """rows: the selected rows of a query of one row (cnt the count; else -1)."""
    t = q.shape[0]
    prog = P.Program()
    out = np.zeros((t, NQ * HD), np.float32)
    sel = np.zeros((t, max(maxsel, 1)), np.int32)
    cnt = np.full(t, -1, np.int32)
    if rows is not None:
        sel[0, :len(rows)] = rows
        cnt[0] = len(rows)
    scores = np.zeros(NQ * (pos + t) + 64, np.float32)
    prog.emit(P.ATTN_QSA, q, kq, ks, vq, vs, scores, out, NQ, NK, HD, t, pos, sel, cnt, maxsel, 1, 0)
    prog.names["out"] = out
    prog = prog.finish()
    g = GPUProgram(prog, graph=False, mirror=mirror)
    g.run()
    ms = float(np.median([g.profile()[0] for _ in range(reps)])) if reps else 0.0
    g.download("out")
    res = out.copy()
    g.close()
    return res, ms


def reference(q, kq, ks, vq, vs, pos, rows, sel=None):
    k = (kq.astype(np.float64).reshape(len(kq), -1, 32) * ks[..., None]).reshape(len(kq), NK, HD)
    v = (vq.astype(np.float64).reshape(len(vq), -1, 32) * vs[..., None]).reshape(len(vq), NK, HD)
    out = {}
    for j in rows:
        keys = np.arange(pos + j + 1) if sel is None else np.asarray(sel)
        qj = q[j].astype(np.float64).reshape(NQ, HD)
        o = np.zeros((NQ, HD))
        for h in range(NQ):
            kv = h // (NQ // NK)
            s = k[keys, kv] @ qj[h]
            p = np.exp(s - s.max())
            o[h] = p @ v[keys, kv] / p.sum()
        out[j] = o.reshape(-1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pos", default="4000,200000")
    ap.add_argument("--rows", default="256,61")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    mirror = Mirror()
    bad = 0
    for pos in (int(x) for x in args.pos.split(",")):
        for t in (int(x) for x in args.rows.split(",")):
            kq, ks = cache(rng, pos + t)
            vq, vs = cache(rng, pos + t)
            # the scale of the queries of the model (ATTN_PREP: hd^-0.5)
            q = (rng.standard_normal((t, NQ * HD)) * 0.6 / np.sqrt(HD)).astype(np.float32)
            dense, ms_d = run(q, kq, ks, vq, vs, pos, 0, mirror, args.reps)
            old, ms_o = run(q, kq, ks, vq, vs, pos, 2051, mirror, args.reps)
            rows = [0, 1, t // 2, t - 1]
            ref = reference(q, kq, ks, vq, vs, pos, rows)
            e_d = max(np.abs(dense[j] - ref[j]).max() / np.abs(ref[j]).max() for j in rows)
            e_o = max(np.abs(old[j] - ref[j]).max() / np.abs(ref[j]).max() for j in rows)
            diff = np.abs(dense - old).max() / np.abs(old).max()
            ok = e_d < 2e-3 and diff < 2e-3
            bad += not ok
            print("pos %6d rows %3d: dense vs ref %.2e, old vs ref %.2e, dense vs old %.2e; "
                  "%.2f ms (old %.2f ms, %.1fx) %s" % (pos, t, e_d, e_o, diff, ms_d, ms_o,
                                                    ms_o / max(ms_d, 1e-9), "ok" if ok else "BAD"))
    # one query (a step: k_attn_part_tc, or k_attn_part with
    # NP_GEMMA_GPU_PART_TC=0): all the positions (the MTP layer), and the
    # rows of a QSA selection
    for pos in (int(x) for x in args.pos.split(",")):
        kq, ks = cache(rng, pos + 1)
        vq, vs = cache(rng, pos + 1)
        q = (rng.standard_normal((1, NQ * HD)) * 0.6 / np.sqrt(HD)).astype(np.float32)
        sel = np.sort(rng.choice(pos + 1, min(pos + 1, 2051), replace=False)).astype(np.int32)
        for name, rows in (("all", None), ("sel", sel)):
            got, ms = run(q, kq, ks, vq, vs, pos, 2051, mirror, args.reps, rows=rows)
            ref = reference(q, kq, ks, vq, vs, pos, [0], sel=rows)[0]
            err = np.abs(got[0] - ref).max() / np.abs(ref).max()
            ok = err < 2e-3
            bad += not ok
            print("pos %6d one query, %s: vs ref %.2e; %.3f ms %s" % (pos, name, err, ms, "ok" if ok else "BAD"))
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
