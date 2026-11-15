#!/usr/bin/env python3
"""Check the GPU products of the block formats of a GGUF file (kernels/quant/iq.cuh: Q3_K, IQ3_XXS,
IQ4_NL, IQ3_S, IQ4_XS, and the other types of the file) on its own matrices: GP_KQ_LINEAR of a step
and of groups of 3 and 8 tokens (each token gets the bits of a step) and of a large group (the tiles:
the tensor cores, the dp4a tiles under RECLANK_GPU_CC=61, the float tiles under RECLANK_GPU_CC=52),
against a float64 product of the dequantized rows (reclank/formats/gguf.py). x is real: the embeddings
of the tokens of a text (the matrices of hidden_size columns), and for a down matrix silu(gate x) (up x)
of its layer. x is int8 on the GPU, so the error is that of int8 x.

    python scripts/check_kq_iq.py MODEL.gguf [--rows 1024] [--types IQ4_XS,Q3_K]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.ir import program as P  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402
from reclank.formats.gguf import GGUF, _TYPE_NAME  # noqa: E402

TEXT = ("The river had been rising for three days before anyone in the village thought to move the "
        "grain from the low barns. By the time the elders met, the water stood at the second step of "
        "the mill, and the miller's daughter had already carried the ledgers to the church loft. ")


def linear(x, w, type_, rows, cols, mirror):
    t = x.shape[0]
    out = np.zeros((t, rows), np.float32)
    p = P.Program()
    p.emit(P.KQ_LINEAR, None, None, None, x, w, type_, rows, cols, t, out)
    p.names["out"] = out
    g = GPUProgram(p.finish(), graph=False, mirror=mirror)
    g.run()
    g.download("out")
    g.close()
    return out


def rows_of(g, name, n):
    """The raw blocks (n rows) and the float values of the first n rows of a matrix."""
    blocks, dims, t = g.raw(name)
    cols, rows = int(dims[0]), int(dims[1])
    n = min(n, rows)
    raw = np.ascontiguousarray(np.ascontiguousarray(blocks).view(np.uint8).reshape(rows, -1)[:n]).reshape(-1)
    return raw, g.dequant(name, rows=np.arange(n)).reshape(n, cols), t, cols, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf")
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--types", default="", help="type names (default: all the matrix types of the file)")
    ap.add_argument("--save", default="", help="an npz of the GPU's outputs (to compare two builds or kernels bit for bit)")
    ap.add_argument("--names", default="", help="these matrices (GGUF names) in place of one of each type; a matrix "
                    "of other columns takes the values of the embeddings in rows of its length")
    args = ap.parse_args()
    g = GGUF(args.gguf, stage=False)
    hidden = int(g.tensors["token_embd.weight"][0][0])
    from reclank.models.qwen.tokenizer import QwenTokenizer
    tok = QwenTokenizer(os.path.join(os.path.dirname(os.path.abspath(args.gguf)), "tokenizer.json")) \
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(args.gguf)), "tokenizer.json")) else None
    ids = np.array(tok.encode(TEXT * 4) if tok else np.arange(1000, 1400), np.int64)
    emb = g.dequant("token_embd.weight", rows=ids).reshape(len(ids), hidden).astype(np.float32)
    want = set(args.types.split(",")) if args.types else None
    # the first matrix of each type (of hidden_size columns, or a down matrix)
    picks = {}
    for n, (d, t, _o) in g.tensors.items():
        if len(d) != 2 or t == 0 or not n.startswith("blk.") or "nextn" in n or n.startswith("blk.64"):
            continue
        name = _TYPE_NAME.get(t, str(t))
        if (want and name not in want) or name in picks:
            continue
        if int(d[0]) == hidden or n.endswith("ffn_down.weight"):
            picks[name] = n
    if args.names:
        picks = {"%s:%d" % (_TYPE_NAME.get(g.tensors[n][1], g.tensors[n][1]), i): n
                 for i, n in enumerate(args.names.split(","))}
    mirror = Mirror()
    bad = 0
    saved = {}
    for name, n in sorted(picks.items()):
        raw, W, t, cols, nr = rows_of(g, n, args.rows)
        if cols == hidden:
            X = emb
        elif not n.endswith("ffn_down.weight"):
            X = emb.reshape(-1)[:len(emb) * hidden // cols * cols].reshape(-1, cols)
        else:
            blk = n.rsplit(".", 2)[0]
            _r, Wg, _t, _c, _n = rows_of(g, blk + ".ffn_gate.weight", cols)
            _r, Wu, _t, _c, _n = rows_of(g, blk + ".ffn_up.weight", cols)
            a = emb.astype(np.float64) @ Wg.T.astype(np.float64)
            X = (a / (1 + np.exp(-a)) * (emb.astype(np.float64) @ Wu.T.astype(np.float64))).astype(np.float32)
        for tt in (1, 3, 8, 160):
            x = np.ascontiguousarray(X[:tt])
            y = linear(x, raw, t, nr, cols, mirror)
            saved["%s|%d" % (n, tt)] = y
            ref = x.astype(np.float64) @ W.T.astype(np.float64)
            err = float(np.abs(y - ref).max() / (np.abs(ref).max() + 1e-30))
            same = True
            if 1 < tt <= 8:
                alone = [linear(x[j:j + 1].copy(), raw, t, nr, cols, mirror) for j in range(tt)]
                same = all(np.array_equal(y[j], alone[j][0]) for j in range(tt))
            ok = same and err < 2e-2 and np.isfinite(y).all()
            bad += not ok
            print("%-8s %-24s %5d x %5d t %3d: rel err %.2e%s %s" % (
                name, n, nr, cols, tt, err, "" if same else ", tokens DIFFER from steps",
                "ok" if ok else "BAD"), flush=True)
    if args.save:
        np.savez(args.save, **saved)
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
