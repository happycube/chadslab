#!/usr/bin/env python3
"""The room of DeepSeek V4.1 Flash on the GPU (gpu.py make_room, shrink_hot): a program that finds the
GPU full frees the other programs, then the hot experts give slots. On a real prompt, greedy with
DSpark: the tokens of a plain run against a run where, between the rounds of its decode, the programs
go (make_room), then a slot of each layer (make_room again within ROOM_S), then two more
(shrink_hot); then the prompt again on the smaller hot set; then on a GPU filled with ballast (a
program that does not fit makes the room itself); then relax after a long sequence's cap (the cap
down, the hot experts back to their slots at the load). All the tokens must be the same (the experts
give the same bits on the GPU and on the CPU).

    RECLANK_GPU_PART_THREADS=21 taskset -c 0-21,24-45 python scripts/check_dsv41_room.py \\
        /home/cpage/models/DeepSeek-V4.1-Flash-1017 /home/cpage/npg-run/dsv41-ref/v1/prompt2.npz
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
from reclank.models.deepseek41.engine import VISION_GB  # noqa: E402
from reclank.models.deepseek41.gpu import HOT_GB_DSPARK, DSV41GPU  # noqa: E402
from reclank.models.deepseek41.model import DSV41CPU  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("ref")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--hot-gb", type=float, default=HOT_GB_DSPARK - VISION_GB,
                    help="the hot experts' GPU memory (default the server's with images)")
    ap.add_argument("--leave", type=float, default=0.3, help="the GPU memory the ballast leaves free (GB)")
    a = ap.parse_args()
    t0 = time.time()
    m = DSV41CPU(a.ckpt, exact=False)
    g = DSV41GPU(m, keep_layers=False, hot_gb=a.hot_gb)
    ds = DSpark(g)
    n0 = g.n_slots
    print("loaded in %.1f s: %d hot experts a layer; GPU %.2f GB free" % (time.time() - t0, n0, mem_info()[0] / 1e9))
    ids = [int(v) for v in np.load(a.ref)["ids"]]

    base, st = ds.generate(ids, a.tokens, spec=True)
    print("plain run: %d tokens, %d rounds; GPU %.2f GB free" % (len(base), st["rounds"], mem_info()[0] / 1e9))

    events = {a.tokens // 4: "room", a.tokens // 2: "room", 3 * a.tokens // 4: "shrink2"}
    seen = []

    def on_token(_tok):
        k = len(seen)
        seen.append(_tok)
        ev = events.get(k)
        if ev is None:
            return
        free0, n = mem_info()[0], g.n_slots
        ok = g.make_room("check %d" % k) if ev == "room" else g.shrink_hot(2) > 0
        print("  token %d: %s -> %s; hot %d -> %d a layer; GPU %.2f -> %.2f GB free" % (
            k, ev, ok, n, g.n_slots, free0 / 1e9, mem_info()[0] / 1e9))

    room, st = ds.generate(ids, a.tokens, spec=True, on_token=on_token)
    same = room == base
    first = next((i for i in range(min(len(room), len(base))) if room[i] != base[i]), None)
    print("run with room made: the same tokens: %s%s" % (same, "" if same else " (first difference at %d)" % first))

    again, st = ds.generate(ids, a.tokens, spec=True)
    same2 = again == base
    print("again on %d hot experts a layer (was %d): the same tokens: %s; GPU %.2f GB free" % (
        g.n_slots, n0, same2, mem_info()[0] / 1e9))
    ok = same and same2 and g.n_slots == max(n0 - 3, 0)

    # the real trigger: the GPU filled with ballast but leave GB, so that making a program fails by
    # itself (its arrays or its graphs) and run makes the room
    from reclank.backends.cuda.gpumm import Buffer
    g.free_programs()
    n1 = g.n_slots
    ballast = Buffer(int(mem_info()[0] - a.leave * 1e9), "program")
    print("ballast %.2f GB: GPU %.2f GB free" % (ballast.nbytes / 1e9, mem_info()[0] / 1e9))
    full, st = ds.generate(ids, a.tokens, spec=True)
    same3 = full == base
    print("on a full GPU: the same tokens: %s; hot %d -> %d a layer; GPU %.2f GB free" % (
        same3, n1, g.n_slots, mem_info()[0] / 1e9))
    ballast.free()
    ok = ok and same3 and g.n_slots < n1

    # back: a long sequence's cap (as a session past 100K), then a short one: relax brings the cap
    # down and the hot experts grow back to their slots at the load (the GPU keeps GROW_MARGIN_GB)
    n2 = g.n_slots
    g.fit(100000)
    cap_long = g.cap
    added = g.relax(len(ids))
    print("relax: cap %d -> %d; hot %d -> %d a layer (+%d); GPU %.2f GB free" % (
        cap_long, g.cap, n2, g.n_slots, added, mem_info()[0] / 1e9))
    back, st = ds.generate(ids, a.tokens, spec=True)
    same4 = back == base
    print("after the growth: the same tokens: %s; GPU %.2f GB free" % (same4, mem_info()[0] / 1e9))
    ok = ok and same4 and g.n_slots == n0 and g.cap < cap_long
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
