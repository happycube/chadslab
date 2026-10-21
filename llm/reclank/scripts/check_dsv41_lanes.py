#!/usr/bin/env python3
"""Lanes of DeepSeek V4.1 Flash (reclank/models/deepseek41/lanes.py, docs/OVERLAP_PLAN.md 2.1): two
real prompts, greedy, n tokens each, alone (one after the other on the model) and as a pair (lane B
beside the model, their runs at once); plain steps, then DSpark rounds. Each lane's tokens must be
those of its prompt alone; with --fixed-hot (no copies of the hot experts) bit for bit, else up to
a near tie (an expert on the GPU or on the CPU sums in another order). The tokens a second of the
pair against one alone.

    RECLANK_GPU_PART_THREADS=21 taskset -c 0-21,24-45 python scripts/check_dsv41_lanes.py \\
        /home/cpage/models/DeepSeek-V4.1-Flash-1017 /home/cpage/npg-run/dsv41-ref/v1/prompt0.npz \\
        /home/cpage/npg-run/dsv41-ref/v1/prompt2.npz --tokens 96
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cuda.gpumm import mem_info  # noqa: E402
from reclank.models.deepseek41.dspark import DSpark  # noqa: E402
from reclank.models.deepseek41.gpu import HOT_GB_DSPARK, DSV41GPU  # noqa: E402
from reclank.models.deepseek41.lanes import DSLane, greedy_alone, greedy_pair  # noqa: E402
from reclank.models.deepseek41.model import DSV41CPU  # noqa: E402


def first_diff(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("ref_a")
    ap.add_argument("ref_b")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--hot-gb", type=float, default=HOT_GB_DSPARK - 1.8,
                    help="the hot experts' GPU memory (default the server's with images)")
    ap.add_argument("--fixed-hot", action="store_true", help="no copies of the hot experts (the bits exact)")
    ap.add_argument("--lane-ctx", type=int, default=262144, help="lane B's positions")
    ap.add_argument("--modes", default="plain,dspark")
    a = ap.parse_args()
    t0 = time.time()
    m = DSV41CPU(a.ckpt, exact=False)
    g = DSV41GPU(m, keep_layers=False, hot_gb=a.hot_gb, hot_dyn=not a.fixed_hot)
    ds = DSpark(g) if "dspark" in a.modes else None
    lane = DSLane(g, ds, max_len=a.lane_ctx)
    print("loaded in %.1f s: %d hot experts a layer (%s); GPU %.2f GB free" % (
        time.time() - t0, g.n_slots, "fixed" if a.fixed_hot else "DSHot", mem_info()[0] / 1e9))
    ids_a = [int(v) for v in np.load(a.ref_a)["ids"]]
    ids_b = [int(v) for v in np.load(a.ref_b)["ids"]]
    n = a.tokens
    ok = True
    for mode in a.modes.split(","):
        d = ds if mode == "dspark" else None
        t1 = time.time()
        ra = greedy_alone(g, ids_a, n, d)
        rb = greedy_alone(g, ids_b, n, d)
        t_alone = time.time() - t1
        st = {}
        t2 = time.time()
        pa, pb = greedy_pair(g, lane, ids_a, ids_b, n, d, st)
        t_pair = time.time() - t2
        same_a, same_b = pa == ra, pb == rb
        ok = ok and same_a and same_b
        print("%s: alone %.1f s, pair %.1f s (prompts included); lane A the same tokens: %s%s, lane B: %s%s" % (
            mode, t_alone, t_pair, same_a, "" if same_a else " (first difference at %d)" % first_diff(pa, ra),
            same_b, "" if same_b else " (first difference at %d)" % first_diff(pb, rb)))
        print("  %d pairs; drafts taken A %d of %d, B %d of %d; GPU %.2f GB free" % (
            st["pairs"], st["taken_a"], st["drafted_a"], st["taken_b"], st["drafted_b"], mem_info()[0] / 1e9))
        # the decode alone: the rate of the pairs without the prompts (one more pair run, timed)
        g.reset()
        t3 = time.time()
        greedy_pair(g, lane, ids_a[-64:], ids_b[-64:], n, d)
        t_p = time.time() - t3
        t4 = time.time()
        greedy_alone(g, ids_a[-64:], n, d)
        greedy_alone(g, ids_b[-64:], n, d)
        t_s = time.time() - t4
        print("  short prompts (64 tokens each): the pair %.1f tok/s for both, one after the other %.1f tok/s"
              " (%.2fx)" % (2 * n / t_p, 2 * n / t_s, t_s / t_p))
    lane.close()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
