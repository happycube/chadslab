#!/usr/bin/env python3
"""A study (no change to the runtime): the forms of the KV cache of the
global layers of the Gemma 4 26B at a long context, by the error of the
attention output with the real queries.

The prompt (--tokens of the sources of this repository) goes through the GPU;
the cache then comes to the host (the int16 form, about 4e-5 of the output:
the reference here), and --steps decode steps run on the CPU, where hooks
take the query of each global layer and its rows. Each form quantizes the keys
and the values (groups of 32 values of a row) and the attention runs again;
the error is |o - o_ref| / |o_ref| for each query head (16 a layer):

    q8      int8, a scale max|x| / 127 for 32 values (the form "int8")
    k16v8   int16 keys, q8 values (the form "k16v8")
    rq8     q8 of the rows rotated in each 32 values (the TQ6 rotation:
            signs, then Walsh-Hadamard), the query rotated, the output back
    k16vr8  int16 keys, rq8 values; rk8v8: rq8 keys, q8 values
    q6/rq6  6 bits, a scale for 32 values (max|x| / 31), plain and rotated
    tq6     the TQ6 form of the Qwen cache (np_gemma/tq6.py: rotated, a norm
            and the Lloyd-Max codebook)
    fp8     e4m3, a scale max|x| / 448 for 32 values
    q4/rq4  4 bits (max|x| / 7), plain and rotated

    NP_GEMMA_GPU=1 python scripts/study_kv_forms.py [--tokens 100000] [--steps 4]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("NP_GEMMA_GPU", "1")
os.environ.setdefault("NP_GEMMA_ATTN", "1")      # the int16 attention of the CPU path (ops.attn_decode)

import numpy as np  # noqa: E402

from np_gemma import model as M  # noqa: E402
from np_gemma import ops, tq6  # noqa: E402
from np_gemma.config import Config  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma.tokenizer import Tokenizer  # noqa: E402

GGUF_PATH = "/space/models/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def groups(f):
    """f on rows (..., d) by groups of 32 values."""
    def g(x):
        sh = x.shape
        return f(x.reshape(-1, 32)).reshape(sh)
    return g


def uniform(bits):
    m = 2 ** (bits - 1) - 1

    def f(x):
        sc = np.abs(x).max(-1, keepdims=True) / m
        sc = np.where(sc > 0, sc, 1e-30)
        return np.rint(x / sc).clip(-m, m) * sc
    return f


def fp8(x):
    """e4m3 (no subnormal care beyond the grid) with a scale max|x| / 448."""
    sc = np.abs(x).max(-1, keepdims=True) / 448.0
    sc = np.where(sc > 0, sc, 1e-30)
    y = x / sc
    e = np.floor(np.log2(np.maximum(np.abs(y), 2.0 ** -6)))
    step = 2.0 ** (e - 3)
    return np.rint(y / step) * step * sc


def rotated(f):
    def g(x):
        return tq6.unrotate(f(tq6.rotate(x)))
    return g


def tq6_form(x):
    b, nrm = tq6.quantize(x.astype(np.float32))
    return tq6.dequantize(b, nrm).astype(np.float64)


FORMS = {
    "q8": (groups(uniform(8)), groups(uniform(8))),
    "k16v8": (None, groups(uniform(8))),
    "rq8": (groups(rotated(uniform(8))), groups(rotated(uniform(8)))),
    "k16vr8": (None, groups(rotated(uniform(8)))),
    "rk8v8": (groups(rotated(uniform(8))), groups(uniform(8))),
    "q6": (groups(uniform(6)), groups(uniform(6))),
    "rq6": (groups(rotated(uniform(6))), groups(rotated(uniform(6)))),
    "tq6": (groups(tq6_form), groups(tq6_form)),
    "fp8": (groups(fp8), groups(fp8)),
    "q4": (groups(uniform(4)), groups(uniform(4))),
    "rq4": (groups(rotated(uniform(4))), groups(rotated(uniform(4)))),
}


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cap", default=os.path.expanduser("~/npg-run/study_kv_cap.npz"),
                    help="the captures (queries, rows): made when missing, else read (no model)")
    args = ap.parse_args()
    if os.path.exists(args.cap):
        z = np.load(args.cap, allow_pickle=True)
        layers = [int(L) for L in z["layers"]]
        kv = {L: tuple(z["kv_%d_%d" % (L, j)] for j in range(4)) for L in layers}
        report(list(z["got"]), kv, layers, int(z["P"]), int(z["steps"]))
        return
    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    m = M.Model(g, cfg).load_all(dtype="int4")
    from np_gemma import gpu
    gpu.offload(m, None)
    text = ""
    for f in sorted(glob.glob(ROOT + "/*.md")) + sorted(glob.glob(ROOT + "/np_gemma/*.py")):
        text += "\n\n===== %s =====\n" % os.path.basename(f) + open(f, errors="replace").read()
    ids = tok.encode(text)[:args.tokens + args.steps]
    P = len(ids) - args.steps
    cache = M.KVCache(cfg, max_len=len(ids) + 8)
    m.prefill(ids[:P], cache)
    m._gpu_release(cache)                 # the rows to the host
    M._GPU = M._PROGRAM = False           # the decode steps on the CPU in Python, with the hooks
    glob_layers = [i for i, p in enumerate(cfg.plan) if not p.is_sliding]
    cap = {"cur": -1, "got": []}
    orig_attn, orig_read = ops.attn_decode, M.KVCache.read_qc

    def read_hook(self, layer, end):
        cap["cur"] = layer
        return orig_read(self, layer, end)

    def attn_hook(q, kq, ks, vq, vs, q_heads, kv_heads, head_dim, n, *a, **k):
        if cap["cur"] in glob_layers:
            # the rows of a layer once (those of the last step), a query and
            # its count of rows for each capture
            cap["got"].append((cap["cur"], np.array(q, np.float64).reshape(q_heads, head_dim), n,
                               q_heads, kv_heads, head_dim))
            cap.setdefault("kv", {})[cap["cur"]] = tuple(np.array(a[:n]) for a in (kq, ks, vq, vs))
        return orig_attn(q, kq, ks, vq, vs, q_heads, kv_heads, head_dim, n, *a, **k)
    M.KVCache.read_qc, ops.attn_decode = read_hook, attn_hook
    for j in range(args.steps):
        m.forward(ids[P + j:P + j + 1], cache=cache, start_pos=P + j)
    M.KVCache.read_qc, ops.attn_decode = orig_read, orig_attn
    os.makedirs(os.path.dirname(args.cap), exist_ok=True)
    got = np.empty(len(cap["got"]), dtype=object)
    got[:] = cap["got"]
    kv = {"kv_%d_%d" % (L, j): a for L, t in cap["kv"].items() for j, a in enumerate(t)}
    np.savez(args.cap, got=got, layers=np.array(glob_layers), P=P, steps=args.steps, **kv)
    report(cap["got"], cap["kv"], glob_layers, P, args.steps)


def report(captures, kv, glob_layers, P, steps):
    print("context %d tokens, %d decode steps, global layers %s; %d captures" % (
        P, steps, glob_layers, len(captures)), flush=True)
    errs = {name: [] for name in FORMS}
    by_layer = {}
    for layer, q, n, qh, kvh, hd in captures:
        kq, ks, vq, vs = (a[:n] for a in kv[layer])
        rep = qh // kvh
        for h in range(kvh):
            K = kq[:, h].astype(np.float64) * np.repeat(ks[:, h].astype(np.float64), 32, axis=-1)
            V = vq[:, h].astype(np.float64) * np.repeat(vs[:, h].astype(np.float64), 32, axis=-1)
            Q = q[h * rep:(h + 1) * rep]
            ref = softmax(Q @ K.T) @ V
            for name, (fk, fv) in FORMS.items():
                Kh = K if fk is None else fk(K)
                Vh = V if fv is None else fv(V)
                out = softmax(Q @ Kh.T) @ Vh
                e = np.linalg.norm(out - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
                errs[name] += list(e)
                by_layer.setdefault((name, layer), []).extend(e)
    print("\n%-7s %12s %12s   %s" % ("form", "mean err", "worst head", "mean by layer " + str(glob_layers)))
    for name in FORMS:
        a = np.array(errs[name])
        lay = " ".join("%.2e" % np.mean(by_layer[(name, L)]) for L in glob_layers)
        print("%-7s %12.3e %12.3e   %s" % (name, a.mean(), a.max(), lay))


if __name__ == "__main__":
    main()
