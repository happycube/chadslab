#!/usr/bin/env python3
"""DeepSeek V4.1 Flash on the GPU (reclank/models/deepseek41/gpu.py: the prompt in
groups, then steps) against the reference (scripts/dsv41_reference.py,
DEEPSEEK_V41_PLAN.md Phase 3): the KL and the top token of the prompt's last
logits, then of each step on the reference's tokens (teacher forced). With
--cpu, the CPU model in near mode the same way, for a side by side.

    RECLANK_GPU_PART_THREADS=18 taskset -c 0-19,24-43 python scripts/check_dsv41_gpu_ref.py \\
        /home/cpage/models/DeepSeek-V4.1-Flash /home/cpage/npg-run/dsv41-ref/v1/prompt1.npz [--cpu]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# DeepSeek V4.1 on the GPU runs with OMP_NUM_THREADS=1 only (gpu.check_omp_threads), set before reclank loads
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.deepseek41.gpu import HOT_GB, DSV41GPU  # noqa: E402
from reclank.models.deepseek41.model import DSV41CPU  # noqa: E402


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(ref, mine):
    p = logsm(ref)
    return float((np.exp(p) * (p - logsm(mine))).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("ref", nargs="+")
    ap.add_argument("--group", type=int, default=None, help="the tokens of a group of a prompt (default the model's, RECLANK_DS_GROUP)")
    ap.add_argument("--max-len", type=int, default=None, help="the most positions (default the model's, 1M)")
    ap.add_argument("--cpu", action="store_true", help="the CPU model (near) too")
    ap.add_argument("--hot-gb", type=float, default=HOT_GB, help="the GPU memory of the hot experts (DSHot; default 12)")
    ap.add_argument("--cache", choices=("min", "bf16"), default="min", help="the form of the caches")
    ap.add_argument("--rq8", action="store_true", help="the bfloat16 matrices in RQ8_0 (not BF12)")
    ap.add_argument("--x8-prefill", action="store_true", help="8-bit inputs of the routed experts in a prompt (a measure)")
    ap.add_argument("--full-prefill", action="store_true",
                    help="every layer on every token of the prompt (not the decoder's bounded replay)")
    ap.add_argument("--logits-out", help="save the logits of the prompt and the steps (a .npz), to compare runs")
    ap.add_argument("--steps", type=int, default=0, help="more steps on the GPU's own tokens after the reference's")
    a = ap.parse_args()
    m = DSV41CPU(a.ckpt, exact=False, max_len=a.max_len)
    g = DSV41GPU(m, keep_layers=False, hot_gb=a.hot_gb, cache=a.cache, group=a.group, rq8=a.rq8, x8_prefill=a.x8_prefill,
                 replay=not a.full_prefill)
    saved = {}
    if g.n_slots:
        print("hot experts: %d slots a layer (%.1f GB)" % (g.n_slots, g.n_slots * g.L if False else
                                                          g.n_slots * m.L * g.per / 1e9))
    for path in a.ref:
        r = np.load(path)
        ids, gen, sl = r["ids"], r["gen"], r["step_logits"]
        T = len(ids)
        rows = {}
        for name in ("gpu", "cpu") if a.cpu else ("gpu",):
            m.reset()
            g.reset()
            t0 = time.time()
            if name == "gpu":
                lg = g.prefill(ids, 0, group=a.group)
            else:
                lg = m.forward(ids, 0)[0][-1]
            tp = time.time() - t0
            kls, tops = [kl(r["logits"][-1:], lg[None])], [int(lg.argmax()) == int(r["logits"][-1].argmax())]
            mine = [lg.copy()]
            t0 = time.time()
            hits = []
            for s in range(len(sl)):
                tok = int(gen[s])
                lg = g.step(tok, T + s) if name == "gpu" else m.forward([tok], T + s)[0][-1]
                if name == "gpu" and g.n_slots:
                    hits.append(g.hits())
                kls.append(kl(sl[s:s + 1], lg[None]))
                tops.append(int(lg.argmax()) == int(sl[s].argmax()))
                mine.append(lg.copy())
            ts = (time.time() - t0) / max(len(sl), 1)
            saved[os.path.basename(path)] = np.stack(mine)
            if name == "gpu" and a.steps:
                tok, pos, t1 = int(lg.argmax()), T + len(sl), time.time()
                for s in range(a.steps):
                    lg = g.step(tok, pos + s)
                    if g.n_slots:
                        hits.append(g.hits())
                    tok = int(lg.argmax())
                print("  %d more steps on its own tokens: %.1f ms a step" % (a.steps, (time.time() - t1) / a.steps * 1e3))
            if hits:
                print("  hot: the experts held %.2f of the pairs (the last 8 runs %.2f); %d copies" % (
                    np.mean(hits), np.mean(hits[-8:]), g.hot.copies if g.hot else 0))
            rows[name] = (kls, tops)
            print("%s %s: the prompt (%d tokens) %.2f s, a step %.0f ms; KL to the reference: prompt %.2e, steps "
                  "mean %.2e max %.2e; top token %d of %d" % (
                      os.path.basename(path), name, T, tp, ts * 1e3, kls[0], np.mean(kls[1:]) if len(kls) > 1 else 0,
                      max(kls[1:]) if len(kls) > 1 else 0, sum(tops), len(tops)))
        if a.cpu:
            print("  per step KL gpu: %s" % " ".join("%.1e" % v for v in rows["gpu"][0]))
            print("  per step KL cpu: %s" % " ".join("%.1e" % v for v in rows["cpu"][0]))
    if a.logits_out:
        np.savez(a.logits_out, **saved)
    from reclank.backends.cuda.gpumm import mem_info
    f, tot = mem_info()
    print("GPU memory in use: %.2f GB" % ((tot - f) / 1e9))
    g.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
