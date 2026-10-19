#!/usr/bin/env python3
"""GP_GDN with flag 4 (KDA: the decay of each channel; P4.1 of docs/GLM53_PLAN.md) against the
NumPy model (GLM53CPU.kda_core) on real rows: the inputs of KDA layers from a trace of the
reference (scripts/glm53_reference.py --trace-prefill N: attn_in<i>), their products by the NumPy
model (the operands qkv, a, b, z of the record), then

- the prompt: the N prompt rows as one group (the parallel forms: k_gdn_conv_par, k_gdn_prep,
  k_gdn_scan<.., CH>, k_gdn_out);
- 8 steps of one row (k_gdn_heads);
- a verify group of 4 rows with a log (k_gdn_heads), its first 2 committed (rc_gpu_gdn_commit with
  flags 4), and a step of the third row on the committed state;

each output and the state at the end against NumPy run through the same rows (dB).

    python scripts/check_glm53_kda.py /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw \\
        ~/reclank-golden/glm53/trace_256_256.npz --layers 0,21,44
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.backends.cuda.runtime import GPUProgram, Mirror, lib  # noqa: E402
from reclank.ir import program as P  # noqa: E402
from reclank.models.glm53.model import GLM53CPU  # noqa: E402


def db(got, want):
    e = np.sum((np.asarray(got, np.float64) - want) ** 2)
    return 10 * np.log10(max(e, 1e-300) / np.sum(np.asarray(want, np.float64) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("trace")
    ap.add_argument("--layers", default="0")
    ap.add_argument("--prompt-rows", type=int, default=None,
                    help="the rows of the prompt group (default: the trace's prompt)")
    a = ap.parse_args()
    m = GLM53CPU(a.model)
    cfg = m.cfg
    z = np.load(a.trace)
    T = a.prompt_rows or len(z["ids"])
    nh, d, K = cfg.kda_heads, cfg.kda_dim, cfg.conv_kernel
    cd, vd = 3 * nh * d, nh * d
    mirror = Mirror()
    worst = -1e9
    for i in (int(x) for x in a.layers.split(",")):
        assert cfg.layer_types[i] == "kda", i
        b = "blk.%d." % i
        x = z["attn_in%d" % i]
        assert len(x) >= T + 14, "the trace needs 14 decode rows"
        qkv, f, beta, gate = m.kda_inputs(i, x)
        ref_cache = m.cache()
        conv = np.zeros((K - 1, cd), np.float32)
        S = np.zeros((nh, d, d), np.float32)
        weights = (m.F(b + "ssm_conv1d.weight"), m.F(b + "ssm_a_log"), m.F(b + "ssm_dt.bias"),
                   m.F(b + "ssm_norm.weight"))
        for w in weights + (conv, S):
            mirror.add(w)

        def record(lo, hi, log=None):
            t = hi - lo
            out = np.zeros((t, vd), np.float32)
            prog = P.Program()
            prog.emit(P.GDN, np.ascontiguousarray(qkv[lo:hi]), conv, weights[0], K,
                      np.ascontiguousarray(gate[lo:hi]), np.ascontiguousarray(f[lo:hi]),
                      np.ascontiguousarray(beta[lo:hi]), weights[1], weights[2], weights[3], S, out,
                      np.zeros((t, cd), np.float32), t, nh, nh, d, d, cfg.rms_norm_eps, log, 2 | 4, None)
            prog.names["out"] = out
            g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
            g.run()
            g.download("out")
            g.close()
            return out

        def ref(lo, hi):
            return m.kda_core(i, qkv[lo:hi], f[lo:hi], beta[lo:hi], gate[lo:hi], ref_cache)

        res = [("prompt %d rows" % T, db(record(0, T), ref(0, T)))]
        st = [db(record(T + j, T + j + 1), ref(T + j, T + j + 1)) for j in range(8)]
        res.append(("8 steps (worst)", max(st)))
        # a verify group of 4 rows with a log; commit 2; a step of the third row
        lrow = cd + nh * 3 * d
        log = np.zeros(4 * lrow, np.float32)
        mirror.add(log)
        v0 = T + 8
        got = record(v0, v0 + 4, log)
        snap = (ref_cache.conv[i].copy(), ref_cache.state[i].copy())
        want = ref(v0, v0 + 4)
        res.append(("verify group of 4", db(got, want)))
        ref_cache.conv[i], ref_cache.state[i] = snap
        ref(v0, v0 + 2)
        from reclank.backends.cuda.runtime import _check
        _check(lib().rc_gpu_gdn_commit(mirror.buffer_of(conv).ptr, mirror.buffer_of(S).ptr,
                                       mirror.buffer_of(log).ptr, 2, K, nh, nh, d, d, 4))
        res.append(("step after a commit of 2", db(record(v0 + 2, v0 + 3), ref(v0 + 2, v0 + 3))))
        mirror.buffer_of(S).download(S)
        mirror.buffer_of(conv).download(conv)
        res.append(("state at the end", db(S, ref_cache.state[i])))
        res.append(("conv rows at the end", db(conv, ref_cache.conv[i])))
        print("layer %d: " % i + "; ".join("%s %.1f dB" % r for r in res), flush=True)
        worst = max(worst, *(r[1] for r in res))
    ok = worst < -60
    print("check_glm53_kda: %s (worst %.1f dB)" % ("OK" if ok else "FAILED", worst))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
