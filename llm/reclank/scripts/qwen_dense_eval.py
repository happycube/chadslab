#!/usr/bin/env python3
"""The output of a Qwen3.6 GGUF against a reference file, end to end
(QWEN_DENSE_QUANT.md, steps 3 and 5).

run: on the GPU, a prompt of T tokens of text, then N steps of the next
tokens of the text (teacher forcing: each step reads the whole cache); the
logits of the N + 1 positions and the text tokens go to OUT.npz. No
HotCache (RECLANK_GPU_HOT_DYN=0), so two runs differ only by their files:

    python scripts/qwen_dense_eval.py run --gguf FILE --out OUT.npz [--tokens 2048] [--steps 4096]

compare: for each run, the KL of the reference run to it (the whole
vocabulary), its difference with the base run (the first given: the same
positions, so the paired error is small), the top token as the
reference's, and the NLL of the text:

    python scripts/qwen_dense_eval.py compare REF.npz BASE.npz RUN.npz...

The tokenizer: tokenizer.json in the directory of --tokenizer-gguf (default:
--gguf), and the chat template of that file.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def run(args):
    os.environ.setdefault("RECLANK_GPU_HOT_DYN", "0")
    from reclank.encoders import media as MD
    from reclank.formats.gguf import GGUF
    from reclank.models.qwen import chat as QC
    from reclank.models.qwen.gpu35 import QwenGPU
    from reclank.models.qwen.qwen35 import QwenCache, QwenGGUFProgram
    tg = args.tokenizer_gguf or args.gguf
    tok = QC.QwenTok(os.path.join(os.path.dirname(tg), "tokenizer.json"), None,
                     GGUF(tg, stage=False).meta["tokenizer.chat_template"])
    text = "".join(open(f).read() for f in sorted(glob.glob(args.text)))
    T, N = args.tokens, args.steps
    base = tok.encode(text)
    ids = (base * ((T + N + 1) // len(base) + 1))[:T + N + 1]
    model = QwenGGUFProgram(args.gguf)
    dev = QwenGPU(model, hot_gb=None, ctx=T + N + 64)
    cache = QwenCache(model.cfg, T + N + 64)
    print("dense on the GPU %.2f GB; %d hot experts a layer; cache %s" % (
        dev._dense_bytes() / 1e9, dev.n_slots, cache.kv_form), flush=True)
    dev.attach(cache)
    cache.set_rope(MD.mrope_positions(T + N + 64, []))
    t0 = time.time()
    dev.prefill(ids[:T], pos=0)
    t1 = time.time()
    logits = [np.asarray(dev.logits(), np.float32).reshape(-1)]
    for i in range(N):
        dev.step(ids[T + i], T + i)
        logits.append(np.asarray(dev.logits(), np.float32).reshape(-1))
    t2 = time.time()
    np.savez(args.out, logits=np.stack(logits), next=np.array(ids[T:T + N + 1]))
    print("prompt %d tokens %.0f tok/s, %d steps %.1f tok/s" % (T, T / (t1 - t0), N, N / (t2 - t1)),
          flush=True)


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def compare(args):
    ref = np.load(args.ref)
    lr = logsm(ref["logits"])
    nxt = ref["next"]
    n = len(nxt)
    i = np.arange(n)
    pr = np.exp(lr)
    nr = -lr[i, nxt]
    print("%d positions; the reference %s (NLL %.4f)" % (n, os.path.basename(args.ref), nr.mean()))
    print("%-36s %-22s %-24s %-7s %s" % ("run", "KL (mean +- se)", "KL - base (paired)", "top-1",
                                          "NLL - ref"))
    base = None
    for f in args.runs:
        r = np.load(f)
        assert (r["next"] == nxt).all(), "%s: another text" % f
        l = logsm(r["logits"])
        kl = (pr * (lr - l)).sum(-1)
        if base is None:
            base = kl
        d, dn = kl - base, -l[i, nxt] - nr
        se = lambda a: a.std(ddof=1) / np.sqrt(n)  # noqa: E731
        print("%-36s %.3e +- %.1e   %+.3e +- %.1e   %5.1f%%  %+.4f +- %.4f" % (
            os.path.basename(f)[:36], kl.mean(), se(kl), d.mean(), se(d),
            100 * (l.argmax(-1) == lr.argmax(-1)).mean(), dn.mean(), se(dn)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gguf", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tokens", type=int, default=2048)
    r.add_argument("--steps", type=int, default=4096)
    r.add_argument("--text", default="*.md", help="a glob of the text files (default: *.md here)")
    r.add_argument("--tokenizer-gguf", default=None,
                   help="the GGUF whose directory has tokenizer.json (default: --gguf)")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("runs", nargs="+", help="the base run first")
    args = ap.parse_args()
    (run if args.cmd == "run" else compare)(args)


if __name__ == "__main__":
    main()
