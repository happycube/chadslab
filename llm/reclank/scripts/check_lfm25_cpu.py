#!/usr/bin/env python3
"""LFM2.5-24B-A2B on the CPU against the reference dumps of
scripts/lfm25_reference.py (transformers, float32; docs/LFM25_24B_PLAN.md
phase 1).

Two paths on the BF12 GGUF of scripts/convert_lfm25_gguf.py:

- numpy (QwenGGUF, float32 NumPy, a whole matrix at a time): the hidden
  state after each layer against hs of the dump, on the first rows of a
  prompt (--numpy-rows);
- program (QwenGGUFProgram, compile_qwen_step: the C records): the prompt
  as chunks, then each token of gen as a decode step (the tokens of the
  dump, so the rows stay aligned). The logits of each row against the top
  64 of the dump: the argmax, the KL of the top 64 (the dump's softmax
  restricted to its top 64, against ours at the same ids), the largest
  difference of a top-64 logit.

    RECLANK_QWEN_KV=f32 python scripts/check_lfm25_cpu.py GGUF ~/reclank-golden/lfm25 [--only think]

The same for another model of the Qwen classes and its dumps, e.g.
Qwen3.8-27B (scripts/qwen38_27b_reference.py: the last row of hs is after
the final norm, --hs-last-normed):

    python scripts/check_lfm25_cpu.py Qwen3.8-27B-bf12.gguf ~/reclank-golden/qwen38_27b --hs-last-normed

An EXL3 directory (QwenEXL3Program: the MUL1 matrices of ExLlamaV3) in
place of the GGUF takes the program path alone (--numpy-rows 0).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def rel(a, b):
    """The relative RMS difference of a against b, per row: (rows,)."""
    return np.sqrt(((a - b) ** 2).mean(-1) / np.maximum((b ** 2).mean(-1), 1e-30))


def row_stats(logits, z, rows):
    """Compare our logits (n, vocab) of the dump rows rows against its top 64."""
    ti, tv, lse, am = z["top_ids"][rows], z["top_vals"][rows], z["lse"][rows], z["argmax"][rows]
    m = logits.max(-1, keepdims=True)
    our_lse = (m[:, 0] + np.log(np.exp(logits - m).sum(-1)))
    ours = np.take_along_axis(logits, ti, axis=-1)
    p = np.exp(tv - lse[:, None])
    p = p / p.sum(-1, keepdims=True)
    q = np.exp(ours - our_lse[:, None])
    q = q / q.sum(-1, keepdims=True)
    kl = (p * (np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(q, 1e-30)))).sum(-1)
    return dict(agree=(logits.argmax(-1) == am), kl=kl, dmax=np.abs(ours - tv).max(-1))


def text_nll(logits, ids):
    """The loss (nats) of the tokens ids[1:] of a prompt (the real text) from
    the logits of its rows (row r predicts token r + 1)."""
    lg = logits[:len(ids) - 1]
    m = lg.max(-1)
    lse = m + np.log(np.exp(lg - m[:, None]).sum(-1))
    return lse - lg[np.arange(len(ids) - 1), np.asarray(ids[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("dumps")
    ap.add_argument("--only", default=None)
    ap.add_argument("--numpy-rows", type=int, default=24, help="0: skip the NumPy path")
    ap.add_argument("--steps", type=int, default=None, help="at most this many decode steps a prompt")
    ap.add_argument("--save-nll", default=None, help="save the loss of each token of the prompts (.npy)")
    ap.add_argument("--max-kl", type=float, default=1e-3,
                    help="the largest mean KL that passes (a file of 4 to 6 bits, EXL3: about 1e-2)")
    ap.add_argument("--hs-last-normed", action="store_true",
                    help="the last row of hs of the dumps is after the final norm (transformers)")
    a = ap.parse_args()
    os.environ.setdefault("RECLANK_DAX_STAGE", "0")
    from reclank.models.qwen.qwen35 import QwenCache, QwenGGUF, QwenGGUFProgram

    files = sorted(glob.glob(os.path.join(a.dumps, "*.npz")))
    if a.only:
        files = [f for f in files if os.path.basename(f)[:-4] in a.only.split(",")]
    fails = 0

    if a.numpy_rows:
        ref = QwenGGUF(a.gguf)
        cfg = ref.cfg
        z = np.load(files[0])
        n = min(a.numpy_rows, len(z["ids"]))
        t0 = time.time()
        got = {}
        cache = QwenCache(cfg, max_len=n + 8, kv="f32")
        h = ref.forward(z["ids"][:n], cache, hook=lambda name, x: got.__setitem__(name, x.copy()))
        worst = 0.0
        L = cfg.num_hidden_layers
        for i in range(L):
            if i == L - 1 and a.hs_last_normed:
                r = rel(h, z["hs"][L, :n])
            else:
                r = rel(got["layer.%d" % i], z["hs"][i + 1, :n])
            worst = max(worst, float(r.max()))
            if i < 3 or i % 8 == 7 or i == cfg.num_hidden_layers - 1:
                print("numpy %s layer %2d (%s): rel diff max %.2e mean %.2e" % (
                    os.path.basename(files[0])[:-4], i, cfg.layer_types[i], r.max(), r.mean()), flush=True)
        st = row_stats(ref.logits(h), z, np.arange(n))
        print("numpy: %d rows, the layers worst %.2e; argmax %d/%d, KL max %.2e, top-64 logit diff max %.3f "
              "(%.0f s)" % (n, worst, st["agree"].sum(), n, st["kl"].max(), st["dmax"].max(), time.time() - t0),
              flush=True)
        fails += int(worst > 1e-2 or not st["agree"].all())
        del ref

    from reclank.formats.exl3 import is_exl3
    if is_exl3(a.gguf):
        from reclank.models.qwen.qwen35 import QwenEXL3Program
        model = QwenEXL3Program(a.gguf)
    else:
        model = QwenGGUFProgram(a.gguf)
    cfg = model.cfg
    print("program: kv form %s" % cfg.kv_form, flush=True)
    tot = {"agree": [], "kl": [], "dmax": []}
    nll = []
    for f in files:
        name = os.path.basename(f)[:-4]
        z = np.load(f)
        ids, gen = z["ids"].tolist(), z["gen"].tolist()
        T = len(ids)
        steps = len(gen) - 1 if a.steps is None else min(a.steps, len(gen) - 1)
        cache = QwenCache(cfg, max_len=T + steps + 8)
        t0 = time.time()
        h = model.forward(ids, cache)
        tp = time.time() - t0
        lg = model.logits(h)
        sp = row_stats(lg, z, np.arange(T))
        nll.append(text_nll(lg, ids))
        rows, t1 = [], time.time()
        for s in range(steps):
            hs = model.forward([gen[s]], cache, start_pos=T + s)
            rows.append(model.logits(hs)[0])
        td = time.time() - t1
        sd = row_stats(np.array(rows), z, np.arange(T, T + steps)) if steps else None
        for st in (sp, sd):
            if st is not None:
                for k in tot:
                    tot[k].append(st[k])
        msg = "%-6s prompt %4d rows: argmax %d/%d, KL mean %.2e max %.2e, dmax %.3f (%.1f s)" % (
            name, T, sp["agree"].sum(), T, sp["kl"].mean(), sp["kl"].max(), sp["dmax"].max(), tp)
        if sd is not None:
            msg += " | decode %3d: argmax %d/%d, KL mean %.2e max %.2e, dmax %.3f (%.0f ms a step)" % (
                steps, sd["agree"].sum(), steps, sd["kl"].mean(), sd["kl"].max(), sd["dmax"].max(),
                1000 * td / max(steps, 1))
        print(msg, flush=True)
        bad = np.nonzero(~sp["agree"])[0].tolist() + [T + j for j in np.nonzero(~sd["agree"])[0]] if sd is not None \
            else np.nonzero(~sp["agree"])[0].tolist()
        if bad:
            print("        rows with another argmax: %s" % bad[:20], flush=True)
    ag = np.concatenate(tot["agree"])
    kl = np.concatenate(tot["kl"])
    print("all: %d rows, argmax %.3f%%, KL mean %.2e p99 %.2e max %.2e, dmax max %.3f" % (
        ag.size, 100 * ag.mean(), kl.mean(), np.percentile(kl, 99), kl.max(), np.concatenate(tot["dmax"]).max()))
    nl = np.concatenate(nll)
    print("the loss of the real text of the prompts: %.4f nats over %d tokens" % (nl.mean(), nl.size))
    if a.save_nll:
        np.save(a.save_nll, nl)
    fails += int(not kl.mean() <= a.max_kl)      # (a NaN fails)
    print("PASS" if not fails else "FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
