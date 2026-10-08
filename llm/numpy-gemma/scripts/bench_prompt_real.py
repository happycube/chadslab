#!/usr/bin/env python3
"""The prompt rate of Qwen3.8 on the GPU on real text, in server settings.

A prompt of random tokens routes differently from text: on the 2-socket
Xeon an 8K prompt of random tokens ran at 652 tok/s and one of the notes and
sources of this project at 216 to 270 (the CPU took most of each mixed
group). This prompt is that text (the *.md files and np_gemma/*.py), cut to
--tokens; the image encoder lends its room as with serve_qwen4 --mmproj-gpu
lend, and the cache holds --ctx tokens. It prints the rate of each run (the
copy cost of the mixed groups calibrates over the first runs), the pool, and
the memory of GpuMem.

    GF=model.gguf [EXPERTS=experts.gguf] python scripts/bench_prompt_real.py \\
        [--tokens 8192] [--reps 3] [--ctx 262144] [--out logits.npy]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import np_gemma  # noqa: E402,F401  (the NUMA policy before NumPy)
import numpy as np  # noqa: E402


CHAT = ("<|im_start|>user\nWrite a detailed explanation of how a CPU cache hierarchy works, with L1, "
        "L2 and L3, and why it matters for performance.<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n")


def decode(g, m, tok, args):
    """Plain decode and MTP with --draft drafts of a chat prompt, for each team."""
    import ctypes
    from np_gemma import cops
    from np_gemma.qwen4 import Qwen4Cache
    from np_gemma.qwen4_gpu import generate_mtp_gpu
    ids = tok.encode(CHAT)
    n = args.decode

    def plain():
        c = Qwen4Cache(m.cfg, args.ctx)
        g.attach(c)
        g.prefill(ids)
        hc = g.hot_cache
        s0, c0 = hc.steps, hc.cold
        nxt, pos = int(g.argmax()[0]), len(ids)
        t0 = time.time()
        for _ in range(n - 1):
            g.step(nxt, pos)
            nxt = int(g.argmax()[0])
            pos += 1
        return (n - 1) / (time.time() - t0), (hc.cold - c0) / max(1, hc.steps - s0)

    def mtp(draft):
        c = Qwen4Cache(m.cfg, args.ctx)
        g.attach(c)
        st = {}
        out = generate_mtp_gpu(g, ids, n, draft=draft, stats=st)
        return (len(out) - 1) / st["decode_s"], st["accepted"] / max(1, st["drafted"])

    plain()                                 # warm
    for th in [int(x) for x in args.threads.split(",") if x] or [0]:
        if th:
            cops._lib.gemma_set_task_threads(ctypes.c_int(th))
        r = [plain() for _ in range(2)]
        held = int((g.pool.owner >= 0).sum()) if g.pool is not None else 0
        print("team %s: plain %.2f tok/s (%.2f), %.0f cold experts a step; %d experts in the pool" % (
            th or "default", r[1][0], r[0][0], r[1][1], held), flush=True)
        for d in [int(x) for x in str(args.draft).split(",") if x]:
            q = [mtp(d) for _ in range(2)]
            print("team %s: MTP %d draft %.2f tok/s (%.2f), %.0f%% accepted" % (
                th or "default", d, q[1][0], q[0][0], 100 * q[1][1]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--lend-gb", type=float, default=1.12, help="room the encoder lends (0: none)")
    ap.add_argument("--out", default=None, help="the logits of the last token of the last run")
    ap.add_argument("--decode", type=int, default=0,
                    help="then decode this many tokens of a chat prompt: plain, and MTP")
    ap.add_argument("--draft", default="3", help="the drafts of a round of MTP (a list: each)")
    ap.add_argument("--threads", default="", help="teams of the decode to try (gemma_set_task_threads), e.g. 24,40")
    args = ap.parse_args()
    from np_gemma.gpumm import Buffer, mem, mem_info
    from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU
    from np_gemma.qwen4_gpu import Qwen4GPU
    from np_gemma.qwen_tok import QwenTokenizer
    gf = os.environ["GF"]
    tok = QwenTokenizer(os.path.join(os.path.dirname(gf), "tokenizer.json"))
    text = ""
    for f in sorted(glob.glob("*.md")) + sorted(glob.glob("np_gemma/*.py")):
        text += "\n\n===== %s =====\n" % f + open(f, encoding="utf-8", errors="replace").read()
    ids = tok.encode("<|im_start|>user\n" + text)[:args.tokens]
    lent = int(args.lend_gb * 1e9)
    ph = Buffer(lent, "lend", "encoder") if lent else None     # the encoder's room
    m = Qwen4CPU(gf, experts=os.environ.get("EXPERTS") or None)
    g = Qwen4GPU(m, ctx=args.ctx)
    if ph is not None:
        ph.free()
        g.lend_warm(lent)
    print("affinity %d CPUs; %d tokens" % (len(os.sched_getaffinity(0)), len(ids)), flush=True)
    lg = None
    for _ in range(args.reps):
        c = Qwen4Cache(m.cfg, args.ctx)
        g.attach(c)
        t0 = time.time()
        g.prefill(ids)
        lg = np.asarray(g.logits()).reshape(-1)
        dt = time.time() - t0
        print("prompt of %d tokens %.2f s (%.0f tok/s)" % (len(ids), dt, len(ids) / dt), flush=True)
    if args.out:
        np.save(args.out, lg)
    if args.decode:
        decode(g, m, tok, args)
    pool = g.pool
    print("pool %.2f GB (hot %.2f, lend %.2f), copies to %d blocks, free %.2f GB" % (
        pool.nbytes() / 1e9, pool.nbytes("hot") / 1e9, pool.nbytes("lend") / 1e9, g.mix_cap,
        mem_info()[0] / 1e9), flush=True)
    print(mem().report(), flush=True)
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
