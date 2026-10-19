#!/usr/bin/env python3
"""LFM2.5-24B-A2B on the GPU (QwenGPU) against the reference dumps of
scripts/lfm25_reference.py (docs/LFM25_24B_PLAN.md phase 2), as
check_lfm25_cpu.py for the CPU.

For each prompt: the prompt by prefill (the groups of the GPU: mixed groups
of a long prompt, split groups of a short one), the logits of its last row,
then each token of gen as a step (the tokens of the dump, so the rows stay
aligned). With --rows N the first N rows of each prompt also run as steps
from an empty cache (the attention of one query and the short convolution of
a step, row by row). The logits of each row against the top 64 of the dump:
the argmax, the KL of the top 64, the largest difference of a top-64 logit;
and the decode rate.

    python scripts/check_lfm25_gpu.py GGUF ~/reclank-golden/lfm25 [--only think] [--steps 64]

Another model of the Qwen classes and its dumps the same way, and an EXL3
directory in place of the GGUF (QwenEXL3Program; a dense model on QwenGPU:
no experts):

    python scripts/check_lfm25_gpu.py ~/models/Qwen3.8-27B-exl3-SC_5.00bpw_H6_V6 ~/reclank-golden/qwen38_27b
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
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("dumps")
    ap.add_argument("--only", default=None)
    ap.add_argument("--steps", type=int, default=None, help="at most this many decode steps a prompt")
    ap.add_argument("--rows", type=int, default=0, help="the first rows of each prompt as steps too")
    ap.add_argument("--hot-gb", type=float, default=None)
    a = ap.parse_args()
    os.environ.setdefault("RECLANK_DAX_STAGE", "0")
    from check_lfm25_cpu import row_stats
    from reclank.models.qwen.gpu35 import QwenGPU
    from reclank.models.qwen.qwen35 import QwenCache, QwenGGUFProgram

    files = sorted(glob.glob(os.path.join(a.dumps, "*.npz")))
    if a.only:
        files = [f for f in files if os.path.basename(f)[:-4] in a.only.split(",")]
    from reclank.formats.exl3 import is_exl3
    if is_exl3(a.gguf):
        from reclank.models.qwen.qwen35 import QwenEXL3Program
        model = QwenEXL3Program(a.gguf)
    else:
        model = QwenGGUFProgram(a.gguf)
    cfg = model.cfg
    t0 = time.time()
    dev = QwenGPU(model, hot_gb=a.hot_gb)
    print("QwenGPU: kv form %s, %d hot experts a layer (%.0f s)" % (cfg.kv_form, dev.n_slots, time.time() - t0),
          flush=True)
    tot = {"agree": [], "kl": [], "dmax": []}
    rates = []
    for f in files:
        name = os.path.basename(f)[:-4]
        z = np.load(f)
        ids, gen = z["ids"].tolist(), z["gen"].tolist()
        T = len(ids)
        steps = len(gen) - 1 if a.steps is None else min(a.steps, len(gen) - 1)
        if a.rows:
            n = min(a.rows, T)
            cache = QwenCache(cfg, max_len=n + 8)
            dev.attach(cache)
            rows = []
            for p in range(n):
                dev.step(ids[p], p)
                rows.append(dev.logits())
            dev.detach(cache)
            st = row_stats(np.array(rows), z, np.arange(n))
            print("%-6s rows 0-%d as steps: argmax %d/%d, KL mean %.2e max %.2e, dmax %.3f" % (
                name, n - 1, st["agree"].sum(), n, st["kl"].mean(), st["kl"].max(), st["dmax"].max()), flush=True)
            for k in tot:
                tot[k].append(st[k])
        cache = QwenCache(cfg, max_len=T + steps + 8)
        dev.attach(cache)
        t1 = time.time()
        dev.prefill(ids)
        rows = [dev.logits()]
        tp = time.time() - t1
        t1 = time.time()
        for s in range(steps):
            dev.step(gen[s], T + s)
            rows.append(dev.logits())
        td = time.time() - t1
        dev.detach(cache)
        st = row_stats(np.array(rows), z, np.arange(T - 1, T + steps))
        for k in tot:
            tot[k].append(st[k])
        if steps:
            rates.append(steps / td)
        bad = [T - 1 + j for j in np.nonzero(~st["agree"])[0]]
        print("%-6s prompt %4d in %.2f s (%.0f tok/s); last row + %d steps: argmax %d/%d, KL mean %.2e max %.2e, "
              "dmax %.3f; decode %.1f tok/s%s" % (
                  name, T, tp, T / tp, steps, st["agree"].sum(), steps + 1, st["kl"].mean(), st["kl"].max(),
                  st["dmax"].max(), steps / max(td, 1e-9), "; other argmax at %s" % bad[:12] if bad else ""),
              flush=True)
    ag, kl = np.concatenate(tot["agree"]), np.concatenate(tot["kl"])
    print("all: %d rows, argmax %.3f%%, KL mean %.2e p99 %.2e max %.2e; decode %.1f tok/s (mean)" % (
        ag.size, 100 * ag.mean(), kl.mean(), np.percentile(kl, 99), kl.max(), np.mean(rates) if rates else 0))
    ok = kl.mean() < 2e-3
    print("PASS" if ok else "FAIL")
    dev.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
