#!/usr/bin/env python3
"""Where the time of a decode step goes: the parts, one team, or one node.

    PYTHONPATH=. python scripts/profile_parts.py --gguf G --mode parts
    PYTHONPATH=. python scripts/profile_parts.py --gguf G --mode one-team
    PYTHONPATH=. python scripts/profile_parts.py --gguf G --mode one-node

parts: NP_GEMMA_PARTS=2 (with a PartKVCache). Each part adds the time of each
of its records (gemma_run_parts_prof, no added barrier), and the barriers of
the parts give the wait of each team for the other parts
(gemma_xbar_stats). one-team: the program of one team on all cores, with a
barrier after each record (gemma_profile). one-node: the same on the cores
and the memory of node 0 (set_mempolicy MPOL_BIND before the load).

The report gives, for each kind of operation, the ms of a token, the MB of
weights that it reads (the operands of 1 MB or more), and the rate.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from collections import defaultdict

GGUF_PATH = "models2/gemma-4-12B-unsloth-UD-Q4_K_XL/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"


def bind_node0():
    """The cores and the memory of node 0, before the load."""
    cpus = []
    with open("/sys/devices/system/node/node0/cpulist") as f:
        for part in f.read().strip().split(","):
            a, _, b = part.partition("-")
            cpus += range(int(a), int(b or a) + 1)
    os.sched_setaffinity(0, cpus)
    os.environ["OMP_NUM_THREADS"] = str(len(cpus))
    libc = ctypes.CDLL(None, use_errno=True)
    mask = (ctypes.c_ulong * 16)()
    mask[0] = 1
    # set_mempolicy(MPOL_BIND, node 0) on x86_64
    if libc.syscall(238, 2, mask, ctypes.c_ulong(16 * 64)) != 0:
        raise OSError(ctypes.get_errno(), "set_mempolicy")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--mode", choices=("parts", "one-team", "one-node"), default="parts")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=32)
    args = ap.parse_args()
    if args.mode == "one-node":
        bind_node0()
    if args.mode == "parts":
        os.environ["NP_GEMMA_PARTS"] = "2"

    import numpy as np
    from np_gemma import KVCache, Model, cops
    from np_gemma import parts as parts_mod
    from np_gemma import program as P
    from np_gemma.config import Config
    from np_gemma.gguf import GGUF
    from np_gemma.tokenizer import Tokenizer

    # The operands of each record, from the compile: (op, layer, arrays).
    cur = {"layer": None}
    orig_emit, orig_compile = P.Program.emit, P.Compiler.compile

    def emit(self, op, *a):
        self.__dict__.setdefault("_ops", []).append(
            (op, cur["layer"], parts_mod._arrays(a)))
        return orig_emit(self, op, *a)

    def compile_(self, form):
        if isinstance(form, tuple) and form and form[0] == "layer":
            cur["layer"] = form[1]
        return orig_compile(self, form)

    P.Program.emit, P.Compiler.compile = emit, compile_

    g = GGUF(args.gguf)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    ids = Tokenizer.from_gguf(g).encode(open("README.md").read() * 4)
    n = args.context
    cache = KVCache(cfg, max_len=n + args.steps + 16)
    model.prefill(ids[:n], cache)
    nxt = ids[n]
    # The compile and a warm step; in parts also the steps of the balance
    # (NP_GEMMA_PART_BALANCE) and the compile with its shares.
    warm = 2
    if args.mode == "parts":
        warm += 2 + int(os.environ.get("NP_GEMMA_PART_BALANCE", "8")) + 1
    for pos in range(n, n + warm):
        x = model.forward([nxt], cache=cache, start_pos=pos)
        nxt = int(np.argmax(model.logits(x)[0]))
    start = n + warm
    attn = P.ready(model, cache)

    if args.mode == "parts":
        paired = os.environ.get("NP_GEMMA_PART_PAIRED", "0") == "1"
        pp = model._programs[("parts", attn, 2, cache.split, paired)]
        progs = pp.progs
        ms = np.zeros((len(progs), max(len(p.recs) for p in progs)), dtype=np.float64)
        cops.gp_xbar_stats()                 # reset
        t0 = time.perf_counter()
        for pos in range(start, start + args.steps):
            pp.bind(model, cache, pos)
            x = model.embed([nxt])
            for p in progs:
                p.names["x"][:] = x
            pp.run_prof(ms)
            xn = progs[0].names["xn"].copy()
            model._parts_xn, model._parts_logits = xn, pp.logits
            nxt = int(np.argmax(model.logits(xn)[0]))
        wall = (time.perf_counter() - t0) / args.steps * 1e3
        xs = cops.gp_xbar_stats()[:len(progs)]
        skip = {id(pp.logits)}
    else:
        prog = model._programs[(attn, 1)]
        progs = [prog]
        ms = np.zeros((1, len(prog.recs)), dtype=np.float64)
        t0 = time.perf_counter()
        for pos in range(start, start + args.steps):
            P.bind_step(prog, model, cache, pos)
            prog.names["x"][:] = model.embed([nxt])
            ms[0] += prog.profile()
            nxt = int(np.argmax(model.logits(prog.names["xn"].copy())[0]))
        wall = (time.perf_counter() - t0) / args.steps * 1e3
        xs = None
        skip = set()

    steps = args.steps
    print("%s, %s, context %d, %d steps: %.1f ms a token (with the head and Python)" % (
        args.mode, os.path.basename(args.gguf), n, steps, wall))
    bal = model.__dict__.get("_parts_balance")
    if args.mode == "parts":
        print("shares of the rows: %s%s" % (
            pp.weights and ", ".join("%.3f" % v for v in pp.weights) or "the same",
            "" if bal is None else "  (measured: fixed %s ms, rows %s ms a step)" % (
                ", ".join("%.1f" % v for v in bal["fixed_ms"]),
                ", ".join("%.1f" % v for v in bal["rows_ms"]))))
    # The kind of each record: the operation, and sliding or global for the
    # attention.
    att = {"ATTN_QC_H", "ATTN_QC", "KV_WRITE", "QKV_NORM_ROPE", "ATTN_F32_H", "COPY"}
    for k, p in enumerate(progs):
        rows = defaultdict(lambda: [0.0, 0.0, 0])      # kind -> ms, MB, count
        for pc, (op, layer, arrs) in enumerate(p._ops[-len(p.recs):]):
            name = P.OP_NAMES.get(op, str(op))
            if name in att and layer is not None:
                name += " (global)" if not cfg.plan[layer].is_sliding else " (sliding)"
            mb = sum(a.nbytes for a in arrs if a.nbytes >= 1 << 20 and id(a) not in skip) / 1e6
            r = rows[name]
            r[0] += ms[k, pc] / steps
            r[1] += mb
            r[2] += 1
        total = sum(r[0] for r in rows.values())
        label = "part %d" % k if len(progs) > 1 else args.mode
        print("\n%s: %.1f ms a token in the records" % (label, total))
        print("  %-26s %8s %6s %9s %8s %6s" % ("operation", "ms", "share", "MB", "GB/s", "recs"))
        for name, (t, mb, c) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
            if t < 0.05 and mb < 1:
                continue
            rate = "%8.1f" % (mb / t) if t > 0 and mb > 0 else "%8s" % "-"
            print("  %-26s %8.2f %5.1f%% %9.1f %s %6d" % (name, t, 100 * t / total, mb, rate, c))
        if xs is not None:
            print("  barriers: %d a token; the wait for the team %.2f ms, for the other "
                  "part %.2f ms (in XBAR and MOE_PART)" % (
                      xs[k, 2] / steps, 1e3 * xs[k, 1] / steps, 1e3 * xs[k, 0] / steps))


if __name__ == "__main__":
    raise SystemExit(main())
