#!/usr/bin/env python3
"""The NumPy model of GLM-5.3-Flash (reclank/models/glm53/model.py, GLM53CPU) against the
reference of scripts/glm53_reference.py (P2 of docs/GLM53_PLAN.md).

--trace trace_N_M.npz (scripts/glm53_reference.py --trace-prefill N --trace-decode M): ids + gen
in one pass, each value of each layer against the trace's (dB), the N prompt rows and the M decode
rows apart, and the router's picks; the KL of the logits.

--layers NAME_layers.npz: the prompt of the dump, the streams after each layer against the
reference's (relative error, dB) and the router's picks (the share of the 8 of each token that
agree); the EXL3 error grows through the layers, the router flips a few picks.

GOLDEN.npz ...: each prompt teacher-forced (ids + gen in one pass); KL(reference || model) on the
reference's top 64 of each row (both renormalized over those ids), the prompt rows and the decode
rows apart (the decode rows are the steadiest); argmax agreement. --save DIR writes the model's
top 64 and lse of each row (<name>_np.npz, the dump form of the reference).

    OMP_NUM_THREADS=24 python scripts/check_glm53_model.py /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw \\
        --layers ~/reclank-golden/glm53/think_layers.npz ~/reclank-golden/glm53/{doc,think}.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from reclank.models.glm53.model import GLM53CPU  # noqa: E402


def db(got, want):
    e = np.sum((got.astype(np.float64) - want) ** 2)
    return 10 * np.log10(max(e, 1e-300) / np.sum(want.astype(np.float64) ** 2))


def kl_rows(z, logits):
    """KL(reference || model) of each row on the reference's top 64."""
    p = np.exp(z["top_vals"] - z["lse"][:, None])
    p /= p.sum(-1, keepdims=True)
    q = np.take_along_axis(logits, z["top_ids"].astype(np.int64), axis=1)
    q = np.exp(q - q.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)
    return (p * (np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(q, 1e-30)))).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("golden", nargs="*")
    ap.add_argument("--layers", default=None, help="a <name>_layers.npz of the reference")
    ap.add_argument("--trace", default=None, help="a trace_N_M.npz of the reference")
    ap.add_argument("--n-layers", type=int, default=None, help="(a test) the first N layers only")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    m = GLM53CPU(a.model, layers=a.n_layers)
    cfg = m.cfg

    if a.trace:
        z = np.load(a.trace)
        T = len(z["ids"])
        toks = z["ids"].tolist() + z["gen"][:-1].tolist()
        got = {}
        t0 = time.time()
        logits = m.forward(toks, m.cache(), hook=lambda k, v: got.__setitem__(k, np.array(v, np.float32)))
        print("%s: %d + %d rows in %.0f s; dB of each value, prompt rows / decode rows" % (
            os.path.basename(a.trace), T, len(toks) - T, time.time() - t0))
        both = lambda k: "%6.1f /%6.1f" % (db(got[k][:T], z[k][:T]), db(got[k][T:], z[k][T:]))  # noqa: E731
        print("  embed %s" % both("embed"))
        for i in range(m.n_layers):
            line = "  layer %2d %s %-5s attn_in %s  attn_out %s  mlp_in %s  mlp_out %s  h %s" % (
                i, cfg.layer_types[i], cfg.mlp_types[i], both("attn_in%d" % i), both("attn_out%d" % i),
                both("mlp_in%d" % i), both("mlp_out%d" % i), both("h%d" % i))
            if "topk%d" % i in z.files:
                ids, want = m.picks[i][0], z["topk%d" % i].astype(np.int64)
                same = [len(set(ids[j]) & set(want[j])) / cfg.top_k for j in range(len(ids))]
                line += "  picks %.1f / %.1f%%" % (100 * np.mean(same[:T]), 100 * np.mean(same[T:]))
            print(line, flush=True)
        if m.n_layers == cfg.num_hidden_layers:
            print("  norm %s" % both("norm"))
            lz = z["logits"]
            p = np.exp(lz - lz.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            lq = logits - logits.max(-1, keepdims=True)
            lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
            lp = np.log(np.maximum(p, 1e-30))
            kl = (p * (lp - lq)).sum(-1)
            am = logits.argmax(-1) == lz.argmax(-1)
            print("  logits: KL (full vocab) prompt rows %.2e, decode rows %.2e (p99 %.2e); argmax %.1f / %.1f%%" % (
                kl[:T].mean(), kl[T:].mean(), np.percentile(kl[T:], 99), 100 * am[:T].mean(), 100 * am[T:].mean()))

    if a.layers:
        z = np.load(a.layers)
        got = {}
        t0 = time.time()
        m.forward(z["ids"].tolist(), m.cache(), hook=lambda k, v: got.__setitem__(k, v.copy()))
        print("%s: %d tokens in %.0f s" % (os.path.basename(a.layers), len(z["ids"]), time.time() - t0))
        print("  embed: %.1f dB" % db(got["embed"], z["embed"]))
        for i in range(m.n_layers):
            if "h%d" % i not in z.files:
                break
            line = "  layer %2d %s %s: streams %6.1f dB" % (i, cfg.layer_types[i], cfg.mlp_types[i],
                                                           db(got["h%d" % i], z["h%d" % i]))
            if "topk%d" % i in z.files:
                ids = m.picks[i][0]
                want = z["topk%d" % i].astype(np.int64)
                same = np.mean([len(set(ids[j]) & set(want[j])) / cfg.top_k for j in range(len(ids))])
                line += ", router picks %.1f%% the same" % (100 * same)
            print(line)
        if "norm" in z.files and m.n_layers == cfg.num_hidden_layers:
            print("  head input: %.1f dB" % db(got["norm"], z["norm"]))

    for path in a.golden:
        z = np.load(path)
        ids, gen = z["ids"].tolist(), z["gen"].tolist()
        toks = ids + gen[:-1]
        t0 = time.time()
        logits = m.forward(toks, m.cache())
        T = len(ids)
        kl = kl_rows(z, logits)
        am = logits.argmax(-1) == z["argmax"]
        print("%s: %d rows in %.0f s; prompt rows KL mean %.2e (argmax %.1f%%), decode rows (%d) KL mean %.2e "
              "p99 %.2e max %.2e (argmax %.1f%%)" % (
                  os.path.basename(path), len(toks), time.time() - t0, kl[:T - 1].mean(), 100 * am[:T - 1].mean(),
                  len(toks) - T + 1, kl[T - 1:].mean(), np.percentile(kl[T - 1:], 99), kl[T - 1:].max(),
                  100 * am[T - 1:].mean()))
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            ti = np.argsort(-logits, axis=1)[:, :64]
            mx = logits.max(-1, keepdims=True)
            np.savez(os.path.join(a.save, os.path.basename(path)[:-4] + "_np.npz"), ids=z["ids"], gen=z["gen"],
                     top_ids=ti.astype(np.int32), top_vals=np.take_along_axis(logits, ti, 1),
                     lse=(mx[:, 0] + np.log(np.exp(logits - mx).sum(-1))).astype(np.float32),
                     argmax=logits.argmax(-1).astype(np.int32), kl=kl)


if __name__ == "__main__":
    main()
