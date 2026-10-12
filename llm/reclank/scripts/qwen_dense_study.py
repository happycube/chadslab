#!/usr/bin/env python3
"""The error of each dense matrix of Qwen3.6 in each type of the dense
plan (QWEN_DENSE_QUANT.md, steps 1 and 2).

collect: the mean square of each input channel of each dense product
(E[x^2]) on real text, through the Python path of a GGUF (QwenGGUFCPU.lin):

    python scripts/qwen_dense_study.py collect --gguf FILE --out act.npz [--tokens 2048]

score: for each matrix of the plan (convert_q36_dense.DENSE) and each type,
its bytes, the error of its weights against ORIG (dB, "plain"), and the
error of its output weighted by the inputs (dB, "weighted":
sum dW^2 E[x^2] / sum W^2 E[x^2], each input channel independent). The
types: Q4_K, Q5_K, Q6_K, Q8_0, their forms of the rotated rows (RQ4_K,
RQ5_K, RQ6_K, RQ8_0: the TQ6 rotation of each 32 values), BF12 (the
bfloat16 values, 0 error but for the values it zeroes; -inf when none):

    python scripts/qwen_dense_study.py score --orig DIR --gguf FILE --act act.npz --out study.json

The template GGUF gives the order of the rows (the DeltaNet value heads,
convert_q36_dense.to_gguf_order) and the shapes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

# type name -> (bits a weight of a block of 256 values, the block of the cols)
TYPES = {"Q4_K": (4.5, 256), "RQ4_K": (4.5, 256), "Q5_K": (5.5, 256), "RQ5_K": (5.5, 256),
         "Q6_K": (6.5625, 256), "RQ6_K": (6.5625, 256), "Q8_0": (8.5, 32), "RQ8_0": (8.5, 32),
         "BF12": (12.25, 32)}


def type_bytes(t, rows, cols):
    if t == "BF12":
        from reclank.backends.cpu import cops
        return rows * cops.bf12_row_bytes(cols)
    return int(rows * cols * TYPES[t][0] / 8)


def roundtrip(t, w):
    """The values of the float32 rows w after type t (None: the cols are not
    whole blocks of it)."""
    from reclank.backends.cpu import cops
    rows, cols = w.shape
    if cols % TYPES[t][1]:
        return None
    rot = t.startswith("R")
    base = t[1:] if rot else t
    x = cops.tq6_rotate(w) if rot else w
    r = np.arange(rows)
    if base == "Q4_K":
        v = cops.kq_rows(cops.kq_to_qk(x, cols, 4), 12, cols, r)
    elif base == "Q5_K":
        v = cops.kq_rows(cops.kq_to_qk(x, cols, 5), 13, cols, r)
    elif base == "Q6_K":
        v = cops.kq_rows(cops.kq_to_q6_k(x, cols), 14, cols, r)
    elif base == "Q8_0":
        v = cops.kq_rows(cops.kq_to_q8_0(x, cols), 8, cols, r)
    elif t == "BF12":
        u = (w.view(np.uint32) >> 16).astype(np.uint16)     # (w holds bfloat16 values)
        q, _rep = cops.kq_bf16_to_bf12(u, cols)
        return cops.kq_rows(q.reshape(-1), 57, cols, r)
    else:
        raise ValueError(t)
    return cops.tq6_rotate(v, inverse=True) if rot else v


def collect(args):
    os.environ.setdefault("RECLANK_GPU", "0")
    from reclank.formats.gguf import GGUF
    from reclank.models.qwen import chat as QC
    from reclank.models.qwen.qwen35 import QwenCache, QwenGGUFCPU, gguf_name
    d = os.path.dirname(args.gguf)
    model = QwenGGUFCPU(args.gguf)
    tok = QC.QwenTok(os.path.join(d, "tokenizer.json"), None,
                     GGUF(args.gguf, stage=False).meta["tokenizer.chat_template"])
    text = "".join(open(f).read() for f in sorted(glob.glob(args.text)))
    ids = tok.encode(text)[:args.tokens]
    acc = {}
    lin = QwenGGUFCPU.lin

    def hook(self, name, qx, full=None):
        x = qx.x.astype(np.float64)
        s = acc.setdefault(gguf_name(name), [np.zeros(x.shape[1]), 0])
        s[0] += (x * x).sum(0)
        s[1] += x.shape[0]
        return lin(self, name, qx, full)
    QwenGGUFCPU.lin = hook
    cache = QwenCache(model.cfg, len(ids) + 64)
    for c0 in range(0, len(ids), 512):
        model.forward(ids[c0:c0 + 512], cache, start_pos=c0)
        print("collect: %d of %d tokens" % (min(c0 + 512, len(ids)), len(ids)), flush=True)
    np.savez(args.out, **{k: v[0] / v[1] for k, v in acc.items()})


def score(args):
    from convert_q36_dense import DENSE, SafeTensors, bf16f, to_gguf_order
    from reclank.formats.gguf import GGUF
    act = np.load(args.act)
    st = SafeTensors(args.orig)
    cfg = json.load(open(os.path.join(args.orig, "config.json")))
    g = GGUF(args.gguf, stage=False)
    types = args.types.split(",")
    out = []
    for name in g.tensors:
        m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
        if not m or m.group(2) not in DENSE:
            continue
        w = np.ascontiguousarray(to_gguf_order(m.group(2), bf16f(st.get(
            "model.language_model.layers.%s.%s.weight" % (m.group(1), DENSE[m.group(2)]))[0]), cfg))
        rows, cols = w.shape
        mx = act[name] if name in act.files else np.ones(cols)
        w64 = w.astype(np.float64)
        den_w, den_p = (w64 ** 2 @ mx).sum(), (w64 ** 2).sum()
        e = {"name": name, "kind": m.group(2), "layer": int(m.group(1)), "rows": rows, "cols": cols,
             "input_peak": float(mx.max() / mx.mean()), "types": {}}
        for t in types:
            v = roundtrip(t, w)
            if v is None:
                continue
            d = (v.astype(np.float64) - w64) ** 2
            ew, ep = (d @ mx).sum(), d.sum()
            e["types"][t] = {"bytes": type_bytes(t, rows, cols),
                             "weighted": float(10 * np.log10(ew / den_w)) if ew > 0 else None,
                             "plain": float(10 * np.log10(ep / den_p)) if ep > 0 else None}
        out.append(e)
        print("%-26s %s" % (name, "  ".join("%s %s" % (t, "%.1f" % x["weighted"] if x["weighted"] is not None
                                                        else "-inf") for t, x in e["types"].items())),
              flush=True)
    json.dump({"orig": args.orig, "gguf": args.gguf, "act": args.act, "matrices": out},
              open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--gguf", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tokens", type=int, default=2048)
    c.add_argument("--text", default="*.md", help="a glob of the text files (default: *.md here)")
    s = sub.add_parser("score")
    s.add_argument("--orig", required=True)
    s.add_argument("--gguf", required=True)
    s.add_argument("--act", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--types", default=",".join(TYPES))
    args = ap.parse_args()
    (collect if args.cmd == "collect" else score)(args)


if __name__ == "__main__":
    main()
