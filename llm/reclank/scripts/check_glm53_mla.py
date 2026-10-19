#!/usr/bin/env python3
"""GLM-5.3-Flash's absorbed MLA on the GPU (P4.2 of docs/GLM53_PLAN.md) against the NumPy model
(GLM53CPU.mla) on real rows: the inputs of a DSA layer from a trace of the reference
(scripts/glm53_reference.py --trace-prefill N: attn_in<i>), its queries, latents and the keys of its
indexer from the NumPy model, then on the GPU

    F16_GLINEAR   W_UK^T q: 64 groups of 512 x 256 (attn_k_b transposed), the latent queries;
    DS_ATTN       the latent queries over the selected latents (no window, the latent cache as
                  the selected entries, a sink of -inf: no sink; scale 1/16);
    F16_GLINEAR   W_UV o: 64 groups of 256 x 512, the values of each head;

for the prompt rows as one group (k_f16g_tile, k_ds_attn_g), a step of one row (k_f16g_gemv,
k_ds_attn) and a verify group of 4. The cache of latents in float32 (form 0) and bfloat16 (form 1;
the latents rounded as the cache would hold them; its prompt group on the tensor cores, GLM_ATTN,
q and P in float16). dB against NumPy on the same latents.

    python scripts/check_glm53_mla.py /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw \\
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


def bf16r(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("trace")
    ap.add_argument("--layers", default="3")
    ap.add_argument("--prompt-rows", type=int, default=None)
    a = ap.parse_args()
    m = GLM53CPU(a.model)
    cfg = m.cfg
    z = np.load(a.trace)
    T = a.prompt_rows or len(z["ids"])
    H, dk, dl = cfg.num_heads, cfg.qk_head_dim, cfg.kv_lora_rank
    maxk = cfg.index_topk + cfg.index_pool - 1
    mirror = Mirror()
    worst = -1e9
    for i in (int(x) for x in a.layers.split(",")):
        assert cfg.layer_types[i] == "dsa", i
        b = "blk.%d." % i
        x = z["attn_in%d" % i]
        N = min(len(x), T + 5)
        x = x[:N]
        q_resid, q, c = m.dsa_inputs(i, x)
        cache = m.cache()
        cache.lat[i] = c
        m.index_scores = {}
        keys = m.index_keys(i, x, q_resid, cache, 0)                     # [N, N]
        wukt = np.ascontiguousarray(m.g.raw(b + "attn_k_b.weight")[0].transpose(0, 2, 1)).reshape(H * dl, dk)
        wuv = np.ascontiguousarray(m.g.raw(b + "attn_v_b.weight")[0]).reshape(H * cfg.v_head_dim, dl)
        sink = np.full(H, -np.inf, np.float32)
        for w in (wukt, wuv, sink):
            mirror.add(w)
        res = []
        for form in (0, 1):
            C = c if form == 0 else bf16r(c)
            ckv = np.ascontiguousarray(C) if form == 0 else (C.view(np.uint32) >> 16).astype(np.uint16)
            mirror.add(ckv)

            def gpu(lo, hi):
                t = hi - lo
                sel = np.zeros((t, maxk), np.int32)
                cnt = np.zeros(t, np.int32)
                for j in range(t):
                    ks = np.nonzero(keys[lo + j])[0]
                    sel[j, :len(ks)] = ks
                    cnt[j] = len(ks)
                qv = np.ascontiguousarray(q[lo:hi].reshape(t, H * dk))
                ql = np.zeros((t, H * dl), np.float32)
                ol = np.zeros((t, H * dl), np.float32)
                out = np.zeros((t, H * cfg.v_head_dim), np.float32)
                prog = P.Program()
                prog.emit(P.F16_GLINEAR, qv, wukt, ql, H * dl, dk, t, H)
                if t > 8 and form == 1:
                    # a prompt group of the program: the tensor cores (k_glm_attn_tc)
                    prog.emit(P.GLM_ATTN, ql, ckv, sel, cnt, ol, H, dl, float(dk ** -0.5), maxk, t)
                else:
                    prog.emit(P.DS_ATTN, ql, None, lo, 0, ckv, sel, cnt, sink, ol, H, dl, float(dk ** -0.5), maxk,
                              None, t, None, None, form, None, None)
                prog.emit(P.F16_GLINEAR, ol, wuv, out, H * cfg.v_head_dim, dl, t, H)
                prog.names["out"] = out
                g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
                g.run()
                g.download("out")
                g.close()
                return out

            for name, lo, hi in (("prompt %d rows" % T, 0, T), ("a step", T, T + 1), ("verify group of 4", T + 1, T + 5)):
                want = m.mla(i, q[lo:hi], C, keys[lo:hi])
                res.append(("form %d %s" % (form, name), db(gpu(lo, hi), want)))
        print("layer %d: " % i + "; ".join("%s %.1f dB" % r for r in res), flush=True)
        worst = max(worst, *(r[1] for r in res))
    # (the prompt group of form 1 on the tensor cores: q and P float16, about -70 dB)
    ok = worst < -60
    print("check_glm53_mla: %s (worst %.1f dB)" % ("OK" if ok else "FAILED", worst))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
