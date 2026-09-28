#!/usr/bin/env python3
"""Measure Qwen3.8-Flash-Next as llama-bench does (the prompt and the decode).

The method is that of tools/llama-bench/llama-bench.cpp of llama.cpp, so the
numbers compare with it:

- The tokens are random: glibc rand() % n_vocab, with no srand (seed 1), in
  one sequence for the whole run. The model adds no BOS token.
- A test ppN decodes a prompt of N tokens from an empty cache (the logits of
  its last token). A test tgN decodes N tokens one at a time from an empty
  cache; the next token is rand(), not a sampled token.
- Before the reps of each test, a warm-up run: the prompt, or 1 generated
  token (it also compiles the programs). --no-warmup skips it.
- The rate of a rep is N / (its time); the table gives the mean and the
  sample std over the reps.
- The tests run in the order of llama-bench: each -p size, then each -n size.

    python scripts/bench_qwen4.py                       # the CPU program, pp512, tg128
    python scripts/bench_qwen4.py -p 512,2048 -n 32 -r 3
    python scripts/bench_qwen4.py --backend gpu --hot-gb 2
    ../llama.cpp-qwen4exp/build-cpu/bin/llama-bench -m MODEL -t 18 -p 512 -n 32 -r 2
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

PATH = ("models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/"
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")


def ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-m", "--model", default=PATH)
    ap.add_argument("-p", "--n-prompt", type=ints, default=[512],
                    help="prompt sizes, comma separated (0 for none; default 512)")
    ap.add_argument("-n", "--n-gen", type=ints, default=[128],
                    help="decode sizes, comma separated (0 for none; default 128)")
    ap.add_argument("-r", "--reps", type=int, default=5, help="reps of each test (default 5)")
    ap.add_argument("-t", "--threads", type=int, default=None,
                    help="OpenMP threads (default: one for each physical core)")
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu",
                    help="cpu: Qwen4CPU; gpu: Qwen4GPU (the experts split)")
    ap.add_argument("--hot-gb", type=float, default=None,
                    help="gpu: GB of hot experts (default: from the free memory)")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    # The thread count must be set before the libraries load.
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import numpy as np
    from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU

    libc = ctypes.CDLL("libc.so.6")
    m = Qwen4CPU(args.model)
    n_vocab = int(m.cfg.vocab_size)     # the tokens of llama-bench: rand() % n_vocab
    rand = lambda: libc.rand() % n_vocab  # noqa: E731
    g = None
    if args.backend == "gpu":
        from np_gemma.qwen4_gpu import Qwen4GPU
        g = Qwen4GPU(m, hot_gb=args.hot_gb)
    max_len = max(args.n_prompt + args.n_gen) + 16

    def fresh():
        c = Qwen4Cache(m.cfg, max_len)
        if g is not None:
            g.attach(c)
        return c

    def decode(tokens, c, pos):
        """llama_decode of one batch, with the logits of its last token."""
        if g is None:
            m.logits(m.forward(tokens, c, start_pos=pos)[-1:])
        elif len(tokens) == 1:
            g.step(tokens[0], pos)
            g.logits()
        else:
            g.prefill(tokens, pos)
            g.logits()

    def test_prompt(n):
        decode([rand() for _ in range(n)], fresh(), 0)

    def test_gen(n):
        c = fresh()
        token = rand()
        for i in range(n):
            decode([token], c, i)
            token = rand()

    tests = [("pp%d" % n, test_prompt, n, n) for n in args.n_prompt if n > 0]
    tests += [("tg%d" % n, test_gen, n, 1) for n in args.n_gen if n > 0]
    backend = "GPU %s" % ("%.1f GB hot" % args.hot_gb if args.hot_gb is not None else
                          "%d hot/layer" % g.n_slots) if g is not None else "CPU"
    if hasattr(m.g, "file_bytes"):
        size = m.g.file_bytes() / 2 ** 30         # the safetensors checkpoint
    else:
        size = sum(t[0].nbytes for t in (m.g.raw(n) for n in m.g.tensors)) / 2 ** 30
    print("| model                      |       size | backend          | threads |   test |"
          "              t/s |")
    print("| -------------------------- | ---------: | ---------------- | ------: | -----: |"
          " ---------------: |")
    for name, fn, n, warm in tests:
        if not args.no_warmup:
            fn(warm)
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn(n)
            ts.append(n / (time.perf_counter() - t0))
        ts = np.array(ts)
        std = ts.std(ddof=1) if len(ts) > 1 else 0.0
        print("| %-26s | %6.2f GiB | %-16s | %7s | %6s | %8.2f ± %5.2f |" % (
            "qwen4exp (np_gemma)", size, backend, os.environ.get("OMP_NUM_THREADS", "?"), name,
            ts.mean(), std), flush=True)
    if g is not None:
        g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
