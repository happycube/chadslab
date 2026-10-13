#!/usr/bin/env python3
"""The cold experts of DeepSeek V4.1 Flash in parts (GP_RUN_PARTS of
GP_KQ_MOE_PART with KQ_MXX and flag 128, DEEPSEEK_V41_PLAN.md Phase 3) against
the products of each expert (cops.kq_mxx_linear, int16 x), on real experts.

The experts of a layer are packed from the shards, each part's rows in the
memory of its node, as the weightkeeper's layout v2 holds them; the tokens take 6
of 16 experts. Flag 128: the act silu(min(g, 10)) clamp(u, -10, 10) times
the weight of the pair before the down rows (the reference's Expert).

    RECLANK_GPU_PARTS=1 taskset -c 0-47 python scripts/check_dsv41_parts.py /home/cpage/models/DeepSeek-V4.1-Flash
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("RECLANK_GPU_PARTS", "1")

import numpy as np  # noqa: E402

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats import dsv41_st as D  # noqa: E402
from reclank.ir import program as P  # noqa: E402
from reclank.runtime import moe_parts, numa  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--experts", type=int, default=16)
    a = ap.parse_args()
    src = D.DSV41Source(a.ckpt)
    H, I, E, k = src.hidden, src.inner, a.experts, 6
    cpus = moe_parts.part_cpus()
    nodes = moe_parts.nodes()[:len(cpus)]
    nparts = len(cpus)
    gu = moe_parts.split(I, nparts, share=[1.0 / nparts] * nparts)
    dn = moe_parts.split(H, nparts, share=[1.0 / nparts] * nparts)
    rng = np.random.default_rng(5)
    real = rng.choice(src.E, E, replace=False)
    full = {}
    pm, keep = [], []
    for p, nd in enumerate(nodes):
        d = []
        for w, (r0, nr), cols in (("w1", gu[p], H), ("w3", gu[p], H), ("w2", dn[p], I)):
            per = nr * cols // 32 * 17
            arr = numa.empty_on((E, per), np.uint8, nd, strict=True)
            for i, e in enumerate(real):
                codes, sc = src.expert(a.layer, int(e))[w]
                arr[i] = cops.kq_mxx_pack(codes[r0:r0 + nr], sc[r0:r0 + nr], nr, cols)[0]
                if p == 0:
                    full[(w, i)] = cops.kq_mxx_pack(codes, sc, codes.shape[0], cols)[0]
            keep.append(arr)
            d += [arr.ctypes.data, cops.KQ_MXX, per, 0]
        pm += d
    pm = np.array(pm, np.int64)
    ok = True
    for t in (1, 4):
        x = (rng.standard_normal((t, H)) * 0.5).astype(np.float32)
        ids = np.stack([rng.choice(E, k, replace=False) for _ in range(t)]).astype(np.int32)
        val = rng.uniform(0.1, 0.5, (t, k)).astype(np.float32)
        out = np.zeros((t, H), np.float32)
        act2 = np.zeros((t * k, I), np.float32)
        scr = [numa.empty_on(cops.kq_moe_part_scratch(t, k, E, H, I, gu[p][1], dn[p][1]), np.uint8, nodes[p],
                             strict=True) for p in range(nparts)]
        q = P.Program()
        q.emit(P.RUN_PARTS, nparts, len(cpus[0]), np.array([c for cs in cpus for c in cs], np.int32))
        q.emit(P.KQ_MOE_PART, x, ids, val, t, k, E, pm, H, I, np.array(gu, np.int32).reshape(-1),
               np.array(dn, np.int32).reshape(-1), act2, out, nparts, len(cpus[0]), None, 32 | 128,
               np.array([s.ctypes.data for s in scr], np.int64))
        q.keep += scr + keep
        q = q.finish()
        t0 = time.time()
        assert cops.gp_run_task(q.buf) == 0
        dt = time.time() - t0
        ref = np.zeros((t, H), np.float64)
        for j in range(t):
            for s in range(k):
                i = int(ids[j, s])
                g = cops.kq_mxx_linear(full[("w1", i)], I, H, x[j:j + 1])
                u = cops.kq_mxx_linear(full[("w3", i)], I, H, x[j:j + 1])
                g = np.minimum(g, 10)
                u = np.clip(u, -10, 10)
                act = (g / (1 + np.exp(-g)) * u * val[j, s]).astype(np.float32)
                ref[j] += cops.kq_mxx_linear(full[("w2", i)], H, I, act)[0]
        err = 10 * np.log10(((out - ref) ** 2).sum() / (ref ** 2).sum())
        ok &= err < -80
        print("%d tokens, %d parts (nodes %s, %d threads each): parts vs each expert's products %.1f dB (%.2f ms)" % (
            t, nparts, nodes, len(cpus[0]), err, dt * 1e3))
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
