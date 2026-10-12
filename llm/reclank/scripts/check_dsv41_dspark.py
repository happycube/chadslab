#!/usr/bin/env python3
"""DSpark of DeepSeek V4.1 Flash on the GPU (reclank/models/deepseek41/dspark.py,
DEEPSEEK_V41_PLAN.md Phase 5) on real prompts: n tokens greedy with plain steps
and with the speculative loop; the tokens must be the same (a row of a verify
group has the bits of a step). The rate of each, the drafts accepted, the
verify lengths chosen, and the confidence against the acceptance.

    RECLANK_GPU_PART_THREADS=18 taskset -c 0-19,24-43 python scripts/check_dsv41_dspark.py \\
        /home/cpage/models/DeepSeek-V4.1-Flash /home/cpage/npg-run/dsv41-ref/v1/prompt2.npz --tokens 64
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

from reclank.models.deepseek41.dspark import DSpark  # noqa: E402
from reclank.models.deepseek41.gpu import HOT_GB_DSPARK, DSV41GPU  # noqa: E402
from reclank.models.deepseek41.model import DSV41CPU  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("ref", nargs="+")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--hot-gb", type=float, default=HOT_GB_DSPARK,
                    help="the GPU memory of the hot experts (default 10: 2 GB less with DSpark)")
    ap.add_argument("--max-len", type=int, default=None, help="the most positions (default the model's, 1M)")
    ap.add_argument("--fixed", type=int, help="always this many drafts (else chosen by the confidence)")
    ap.add_argument("--where", choices=("cpu1", "gpu"), default="cpu1", help="the DSpark experts")
    ap.add_argument("--mode", default="auto", help="auto (a draft while it pays) or always")
    ap.add_argument("--full-prefill", action="store_true",
                    help="every layer on every token of the prompt (not the decoder's bounded replay)")
    a = ap.parse_args()
    a.mode = True if a.mode == "always" else "auto"
    t0 = time.time()
    m = DSV41CPU(a.ckpt, exact=False, max_len=a.max_len)
    g = DSV41GPU(m, keep_layers=False, hot_gb=a.hot_gb, replay=not a.full_prefill)
    ds = DSpark(g, where=a.where)
    print("loaded in %.1f s (hot experts: %d slots a layer; the DSpark experts on %s)" % (
        time.time() - t0, g.n_slots, a.where))
    bad = 0
    for path in a.ref:
        r = np.load(path)
        ids = [int(v) for v in r["ids"]]
        plain, sp = ds.generate(ids, a.tokens, spec=False)
        tp = sp["seconds"]
        spec, st = ds.generate(ids, a.tokens, spec=a.mode if a.fixed is None else True, fixed=a.fixed)
        ts = st["seconds"]
        same = plain == spec
        bad += not same
        first = next((i for i in range(len(plain)) if plain[i] != spec[i]), None)
        ref_gen = [int(v) for v in r["gen"]]
        agree_ref = sum(int(x == y) for x, y in zip(plain, ref_gen))
        print("%s (%d tokens): plain %.1f tok/s, DSpark %.1f tok/s (%.2fx); the same tokens: %s%s; the reference's "
              "first %d: %d agree" % (
                  os.path.basename(path), len(ids), a.tokens / tp, a.tokens / ts, tp / ts, same,
                  "" if same else " (first difference at %d)" % first, len(ref_gen), agree_ref))
        acc = st["accepted"] / max(st["drafted"], 1)
        print("  %d rounds of DSpark (%d plain steps): %.2f tokens a round, %d drafts verified, %.2f accepted; "
              "lengths %s" % (st["rounds"], st.get("plain", 0),
                              (a.tokens - st.get("plain", 0)) / max(st["rounds"], 1), st["drafted"], acc,
                              np.bincount(st["lengths"], minlength=6) if st["lengths"] else []))
        # the confidence of each position against its acceptance (the positions verified)
        pc, ok = [], []
        for conf, res in st["conf"]:
            p = 1.0 / (1.0 + np.exp(-conf))
            for i, v in enumerate(res):
                pc.append(p[i])
                ok.append(v)
        if pc:
            pc, ok = np.array(pc), np.array(ok, float)
            for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
                sel = (pc >= lo) & (pc < hi)
                if sel.any():
                    print("  confidence %.2f-%.2f: %d positions, accepted %.2f" % (lo, min(hi, 1), sel.sum(),
                                                                                  ok[sel].mean()))
        print("  times (ms): draft %.1f, verify %s" % (ds.t_draft, " ".join(
            "%d:%.0f" % (t, v) for t, v in sorted(ds.t_verify.items()) if t <= 6)))
    print("OK" if not bad else "%d prompts differ" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
