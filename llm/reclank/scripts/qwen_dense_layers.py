#!/usr/bin/env python3
"""The sensitivity of the output of Qwen3.6 to the dense matrices of each
layer (QWEN_DENSE_QUANT.md, step 3), in one process: no file to write for
each test.

The model is a file with every dense matrix BF12 (convert_q36_dense.py
--type BF12), the reference. A run: a prompt of --tokens tokens of text
(fewer than the rows of a mixed group: the split groups, the same bits in
every run), then --steps steps of the next tokens of the text (teacher
forcing). For each layer (or each matrix, --per matrix), its dense
matrices take the values of --type (the quantization of the ORIG values:
qwen_dense_study.roundtrip), still in BF12 (the bfloat16 of those values:
the arrays keep their size, so the device copies change in place and the
programs stay), a run, the KL of the reference run to it at each position
(computed as the steps go: no logits on the disk), and the matrices back.
The first run is the reference; a second run of the reference checks that
the runs repeat (KL 0).

    python scripts/qwen_dense_layers.py --gguf BF12FILE --tokenizer-gguf UD.gguf \\
        --orig ORIG --type RQ4_K --out layers.json

The JSON: for each test, the matrices, their summed weighted error
(qwen_dense_study, E), and the mean KL with its standard error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ.setdefault("RECLANK_GPU_HOT_DYN", "0")

import numpy as np  # noqa: E402


def logsm(x):
    x = np.asarray(x, np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the file with every dense matrix BF12")
    ap.add_argument("--tokenizer-gguf", default=None)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--type", default="RQ4_K")
    ap.add_argument("--per", choices=("layer", "matrix"), default="layer")
    ap.add_argument("--only", default=None, help="the tests whose names match this regex")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4096)
    ap.add_argument("--text", default="*.md")
    ap.add_argument("--study", default=None, help="study.json (qwen_dense_study): E of each test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from convert_q36_dense import DENSE, SafeTensors, bf16f, to_gguf_order
    from qwen_dense_study import roundtrip
    from reclank.backends.cpu import cops
    from reclank.encoders import media as MD
    from reclank.formats.gguf import GGUF
    from reclank.models.qwen import chat as QC
    from reclank.models.qwen.gpu35 import QwenGPU
    from reclank.models.qwen.qwen35 import QwenCache, QwenGGUFProgram, gguf_name

    tg = args.tokenizer_gguf or args.gguf
    tok = QC.QwenTok(os.path.join(os.path.dirname(tg), "tokenizer.json"), None,
                     GGUF(tg, stage=False).meta["tokenizer.chat_template"])
    text = "".join(open(f).read() for f in sorted(glob.glob(args.text)))
    T, N = args.tokens, args.steps
    base = tok.encode(text)
    ids = (base * ((T + N + 1) // len(base) + 1))[:T + N + 1]
    st = SafeTensors(args.orig)
    cfg = json.load(open(os.path.join(args.orig, "config.json")))
    model = QwenGGUFProgram(args.gguf)
    dev = QwenGPU(model, hot_gb=None, ctx=T + N + 64)
    E = {}
    if args.study:
        for m in json.load(open(args.study))["matrices"]:
            db = m["types"].get(args.type, {}).get("weighted")
            E[m["name"]] = 0.0 if db is None else 10 ** (db / 10)

    def run(ref=None):
        cache = QwenCache(model.cfg, T + N + 64)
        dev.attach(cache)
        cache.set_rope(MD.mrope_positions(T + N + 64, []))
        dev.prefill(ids[:T], pos=0)
        out = []
        for i in range(N + 1):
            if i:
                dev.step(ids[T + i - 1], T + i - 1)
            ls = logsm(dev.logits())
            if ref is None:
                out.append(ls.astype(np.float32))
            else:
                lr = ref[i].astype(np.float64)
                out.append(((np.exp(lr) * (lr - ls)).sum(), -ls[ids[T + i]]))
        return out

    t0 = time.time()
    ref = np.stack(run())
    print("reference: %d positions in %.0f s" % (len(ref), time.time() - t0), flush=True)
    again = np.array(run(ref))
    print("a second run of the reference: KL %.2e (0: the runs repeat)" % again[:, 0].mean(), flush=True)

    # the dense matrices: their module names (dev.dense keys), GGUF names, ORIG names
    mats = []
    for name in GGUF(args.gguf, stage=False).tensors:
        m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
        if m and m.group(2) in DENSE:
            mats.append((name, int(m.group(1)), m.group(2)))
    module_of = {}
    for i in range(model.n_layers):
        for mod in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
            key = "layers.%d.%s" % (i, mod)
            try:
                module_of[gguf_name(key)] = key
            except KeyError:
                pass
    if args.per == "layer":
        tests = [("layer %d" % L, [n for n, l, _k in mats if l == L]) for L in range(model.n_layers)]
    else:
        tests = [(n, [n]) for n, _l, _k in mats]
    if args.only:
        tests = [t for t in tests if re.search(args.only, t[0])]
    mir = dev.g.mirror
    results = []
    for label, names in tests:
        saved = []
        for n in names:
            key = module_of[n]
            arr, t = dev.dense(key)
            assert t == 57, (n, t)          # BF12
            _l, li, kind = n.split(".")[:3]
            w = np.ascontiguousarray(to_gguf_order(kind, bf16f(st.get(
                "model.language_model.layers.%s.%s.weight" % (li, DENSE[kind]))[0]), cfg))
            v = roundtrip(args.type, w)
            u = (np.ascontiguousarray(v, np.float32).view(np.uint32) + 0x7FFF +
                 ((np.ascontiguousarray(v, np.float32).view(np.uint32) >> 16) & 1)) >> 16
            q, _rep = cops.kq_bf16_to_bf12(u.astype(np.uint16), w.shape[1])
            saved.append((arr, arr.copy()))
            arr[:] = q.reshape(-1)
            mir.buffer_of(arr).upload(arr)
        t1 = time.time()
        r = np.array(run(ref))
        for arr, old in saved:
            arr[:] = old
            mir.buffer_of(arr).upload(arr)
        kl = r[:, 0]
        res = {"test": label, "matrices": names, "E": float(sum(E.get(n, 0.0) for n in names)),
               "kl": float(kl.mean()), "se": float(kl.std(ddof=1) / np.sqrt(len(kl))),
               "nll_minus_ref": float(np.mean(r[:, 1] + np.array([ref[i][ids[T + i]] for i in range(len(r))])))}
        results.append(res)
        print("%-26s KL %.3e +- %.1e  E %.2e  (%.0f s)" % (label, res["kl"], res["se"], res["E"],
                                                          time.time() - t1), flush=True)
        json.dump({"type": args.type, "tokens": T, "steps": N, "repeat_kl": float(again[:, 0].mean()),
                   "tests": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
