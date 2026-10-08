#!/usr/bin/env python3
"""Where the time of a prompt goes: the time of each function of np_gemma.ops
(its own time, without the functions that it calls) and of the attention of
the cache, and the traffic of each socket (perf stat, uncore).

    PYTHONPATH=. python scripts/profile_prefill.py --gguf G --mode one-team
    PYTHONPATH=. python scripts/profile_prefill.py --gguf G --mode one-node

one-team: all the cores. one-node: the cores and the memory of node 0
(set_mempolicy MPOL_BIND before the load). NP_GEMMA_PREFILL_CHUNK gives the
block of the prompt, and NP_GEMMA_PARTS=2 the prompt of the parts.
"""
from __future__ import annotations

import argparse
import functools
import os
import signal
import subprocess
import tempfile
import sys
import time
from collections import defaultdict

GGUF_PATH = "models2/gemma-4-12B-unsloth-UD-Q4_K_XL/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
EV = ["unc_m_cas_count.rd", "unc_upi_txl_flits.all_data", "unc_cha_requests.reads_local",
      "unc_cha_requests.reads_remote"]


def perf_start(out):
    return subprocess.Popen(["perf", "stat", "-a", "--per-socket", "-x,", "-o", out,
                             "-e", ",".join(EV)], preexec_fn=os.setpgrp)


def perf_stop(p, out):
    p.send_signal(signal.SIGINT)
    p.wait()
    r = {}
    for line in open(out):
        f = line.strip().split(",")
        if len(f) > 4 and f[4] in EV:
            r[(f[0], f[4])] = float(f[2])
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--mode", choices=("one-team", "one-node"), default="one-team")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--perf", action="store_true", help="the uncore counters (perf stat)")
    args = ap.parse_args()
    if args.mode == "one-node":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from profile_parts import bind_node0
        bind_node0()

    import numpy as np
    from np_gemma import KVCache, Model, ops
    from np_gemma.config import Config
    from np_gemma.gguf import GGUF
    from np_gemma.tokenizer import Tokenizer

    g = GGUF(args.gguf)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    ids = Tokenizer.from_gguf(g).encode(open("README.md").read() * 4)[:args.tokens]

    # The own time of each function: a stack of the calls that run.
    own = defaultdict(float)
    calls = defaultdict(int)
    stack = []

    def timed(name, fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            t0 = time.perf_counter()
            stack.append(0.0)
            try:
                return fn(*a, **k)
            finally:
                inner = stack.pop()
                dt = time.perf_counter() - t0
                own[name] += dt - inner
                calls[name] += 1
                if stack:
                    stack[-1] += dt
        return wrap

    for name in dir(ops):
        fn = getattr(ops, name)
        if callable(fn) and getattr(fn, "__module__", None) == ops.__name__ and not isinstance(fn, type):
            setattr(ops, name, timed("ops." + name, fn))
    KV = type(KVCache(cfg, max_len=8))
    for name in ("write", "read", "read_qc", "prefill_attention"):
        if hasattr(KV, name):
            setattr(KV, name, timed("cache." + name, getattr(KV, name)))

    chunk = model.prefill_chunk
    print("%s, %s, %d tokens, blocks of %d, threads %s" % (
        args.mode, os.path.basename(args.gguf), len(ids), chunk,
        os.environ.get("OMP_NUM_THREADS", "all")))
    for rep in range(args.reps):
        own.clear()
        calls.clear()
        cache = KVCache(cfg, max_len=len(ids) + 8)
        tmp = os.path.join(tempfile.gettempdir(), "prefill_perf_%d.txt" % os.getpid())
        p = perf_start(tmp) if args.perf else None
        t0 = time.perf_counter()
        model.prefill(ids, cache)
        wall = time.perf_counter() - t0
        run = perf_stop(p, tmp) if p else {}
        print("rep %d: %.2f s, %.1f tok/s" % (rep, wall, len(ids) / wall))
    total = sum(own.values())
    print("  %-34s %8s %6s %7s" % ("function (own time)", "s", "share", "calls"))
    for name, t in sorted(own.items(), key=lambda kv: -kv[1])[:16]:
        print("  %-34s %8.3f %5.1f%% %7d" % (name, t, 100 * t / wall, calls[name]))
    print("  %-34s %8.3f %5.1f%%" % ("(Python and the rest)", wall - total, 100 * (wall - total) / wall))
    if run:
        print("  per socket over the last rep      socket 0    socket 1")
        for e, name, scale in ((EV[0], "DRAM read (GB)", 64e-9), (EV[1], "UPI tx data (GB)", 64e-9 / 9),
                               (EV[2], "CHA reads local (M)", 1e-6),
                               (EV[3], "CHA reads remote (M)", 1e-6)):
            print("  %-34s %9.2f  %9.2f" % (name, run.get(("S0", e), 0) * scale,
                                              run.get(("S1", e), 0) * scale))


if __name__ == "__main__":
    raise SystemExit(main())
