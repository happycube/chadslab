#!/usr/bin/env python3
"""The GPU model of GLM-5.3-Flash (reclank/models/glm53/gpu.py, GLM53GPU) against the NumPy model
(GLM53CPU with the EXL3 matrices decoded to float32: the math of the reference on the same
weights) and against the reference (P5 of docs/GLM53_PLAN.md), on the tokens of a trace of the
reference (scripts/glm53_reference.py --trace-prefill N --trace-decode M).

The prompt of the trace runs as one group on the GPU, then --steps decode steps (the trace's
greedy tokens). The NumPy model runs first in a process of its own (--numpy-only, many threads: its
rows go to --ref; the GPU run keeps OMP_NUM_THREADS=1 and reads them). The input of the head of each row against the NumPy model's (dB, the prompt rows
and the step rows apart); with all the layers, the logits against the trace's (KL on the full
vocabulary, argmax agreement: the EXL3 error) and against the NumPy model's.

    M=/home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw Z=~/reclank-golden/glm53/trace_256_256.npz
    python scripts/check_glm53_gpu.py $M $Z --layers 5 --ref ref5.npz --numpy-only
    RECLANK_GPU_PARTS=1 OMP_NUM_THREADS=1 python scripts/check_glm53_gpu.py $M $Z --layers 5 --ref ref5.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def db(got, want):
    e = np.sum((np.asarray(got, np.float64) - want) ** 2)
    return 10 * np.log10(max(e, 1e-300) / np.sum(np.asarray(want, np.float64) ** 2))


def kl(ref, got):
    p = np.exp(ref - ref.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    lq = got - got.max(-1, keepdims=True)
    lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    return (p * (np.log(np.maximum(p, 1e-30)) - lq)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("trace")
    ap.add_argument("--layers", type=int, default=None, help="(a test) the first N layers only")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ref", default=None, help="the rows of the NumPy model (.npz): written by --numpy-only")
    ap.add_argument("--numpy-only", action="store_true")
    ap.add_argument("--topk", type=int, default=None,
                    help="(a test of the indexer) the keys a query keeps, in both models (GLM: 2048)")
    ap.add_argument("--int8-only", action="store_true", help="--numpy-only: the CPU's products alone (quick)")
    ap.add_argument("--verify", type=int, default=None,
                    help="rows R: a group of the first R rows, a verify group of the next 4, a commit of 2, then "
                         "steps of the rest (each against the NumPy rows)")
    ap.add_argument("--prefill", action="store_true",
                    help="the prompt through prefill (the mixed groups of 256 rows and more): its last row")
    a = ap.parse_args()
    z = np.load(a.trace)
    ids, gen = z["ids"].tolist(), z["gen"].tolist()
    T, S = len(ids), a.steps
    if a.numpy_only and a.int8_only:
        from reclank.models.glm53.model import GLM53CPU
        m8 = GLM53CPU(a.model, layers=a.layers)
        if a.topk:
            m8.cfg.index_topk = a.topk
        got8 = {}
        rl8 = m8.forward(ids + gen[:S], m8.cache(), hook=lambda k, v: got8.__setitem__(k, v) if k == "norm" else None)
        np.savez(a.ref, norm=got8["norm"], logits=rl8, norm8=got8["norm"], logits8=rl8, steps=S)
        return
    if a.numpy_only:
        from reclank.models.glm53.model import GLM53CPU
        ref = GLM53CPU(a.model, layers=a.layers, float_weights=True)
        if a.topk:
            ref.cfg.index_topk = a.topk
        got = {}
        t0 = time.time()
        rl = ref.forward(ids + gen[:S], ref.cache(), hook=lambda k, v: got.__setitem__(k, v) if k == "norm" else None)
        print("NumPy (float weights): %d rows in %.0f s" % (T + S, time.time() - t0), flush=True)
        # and with the EXL3 products of the CPU (int8 x): the products the GPU's are closest to
        m8 = GLM53CPU(ref.g, layers=a.layers)
        if a.topk:
            m8.cfg.index_topk = a.topk
        got8 = {}
        rl8 = m8.forward(ids + gen[:S], m8.cache(), hook=lambda k, v: got8.__setitem__(k, v) if k == "norm" else None)
        np.savez(a.ref, norm=got["norm"], logits=rl, norm8=got8["norm"], logits8=rl8, steps=S)
        return
    from reclank.models.glm53.gpu import GLM53GPU
    from reclank.models.glm53.program import GLM53Prog
    t0 = time.time()
    m = GLM53Prog(a.model, layers=a.layers)
    if a.topk:
        m.cfg.index_topk = a.topk
    dev = GLM53GPU(m, ctx=a.ctx)
    print("the GPU model in %.0f s (%d hot experts a layer)" % (time.time() - t0, dev.n_slots), flush=True)
    cache = m.cache(a.ctx)
    dev.attach(cache)
    if a.verify:
        r = np.load(a.ref)
        toks, R = ids + gen[:S], a.verify
        out = [("group of %d rows" % R, dev.group(toks[:R], 0), slice(0, R))]
        out.append(("verify group of 4", dev.verify(toks[R:R + 4], R), slice(R, R + 4)))
        dev.commit(2)
        st = []
        for j in range(R + 2, len(toks)):
            dev.step(toks[j], j)
            prog, g = dev.last_prog
            g.download("xn")
            st.append(prog.names["xn"][:1].copy())
        out.append(("steps after a commit of 2", np.concatenate(st), slice(R + 2, len(toks))))
        for name, x, sl in out:
            print("%s: against NumPy %.1f dB, against the CPU's products %.1f dB" % (
                name, db(x, r["norm"][sl]), db(x, r["norm8"][sl])))
        dev.close()
        return
    t0 = time.time()
    xn = [dev.prefill(ids) if a.prefill else dev.group(ids, 0)]
    lg = [dev.logits(T)] if a.layers is None else []
    t1 = time.time()
    for j in range(S):
        dev.step(gen[j], T + j)
        prog, g = dev.last_prog
        g.download("xn")
        xn.append(prog.names["xn"][:1].copy())
        if a.layers is None:
            lg.append(dev.logits(1)[None])
    print("GPU: the prompt of %d rows in %.2f s, %d steps in %.2f s" % (T, t1 - t0, S, time.time() - t1), flush=True)
    xn = np.concatenate(xn)
    if a.prefill:
        # the last prompt row alone: the rows before it are those of row T - 1
        xn = np.concatenate([np.zeros((T - 1, xn.shape[1]), np.float32), xn])
    if a.ref:
        r = np.load(a.ref)
        assert int(r["steps"]) == S
        got, rl = {"norm": r["norm"]}, r["logits"]
        P0 = T - 1 if a.prefill else 0
        print("head input, GPU against NumPy: prompt rows %.1f dB, step rows %.1f dB" % (
            db(xn[P0:T], got["norm"][P0:T]), db(xn[T:], got["norm"][T:])))
        if "norm8" in r.files:
            n8 = r["norm8"]
            print("head input, GPU against NumPy with the CPU's EXL3 products: prompt rows %.1f dB, step rows %.1f dB; "
                  "the CPU's products against float: %.1f / %.1f dB" % (
                      db(xn[P0:T], n8[P0:T]), db(xn[T:], n8[T:]), db(n8[:T], got["norm"][:T]),
                      db(n8[T:], got["norm"][T:])))
    if a.layers is None:
        lg = np.concatenate(lg)
        lz = z["logits"][:T + S]
        k = kl(lz, lg)
        am = lg.argmax(-1) == lz.argmax(-1)
        print("logits, GPU against the reference: KL prompt rows %.2e, step rows %.2e; argmax %.1f / %.1f%%" % (
            k[:T].mean(), k[T:].mean(), 100 * am[:T].mean(), 100 * am[T:].mean()))
        if a.ref:
            k2 = kl(rl, lg)
            print("logits, GPU against NumPy: KL prompt rows %.2e, step rows %.2e" % (k2[:T].mean(), k2[T:].mean()))
    dev.close()


if __name__ == "__main__":
    main()
