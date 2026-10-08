#!/usr/bin/env python3
"""Measure the Gemma 4 26B A4B or Qwen3.6 35B A3B as llama-bench does.

The method is that of scripts/bench_qwen4.py (tools/llama-bench of
llama.cpp), so the numbers compare with llama-bench:

- The tokens are random: glibc rand() % n_vocab, seed 1, one sequence for
  the whole run. No BOS token.
- A test ppN runs a prompt of N tokens from an empty cache (with the logits
  of its last token). A test tgN runs N tokens one at a time from an empty
  cache; the next token is rand(), not a sampled token.
- A warm-up run before the reps of each test (the prompt, or one token).
- The table gives the mean and the sample std of the reps.

    python scripts/bench_llama_method.py --arch gemma --backend cpu -p 512,2048 -n 128
    python scripts/bench_llama_method.py --arch qwen --backend gpu -p 512,2048,8192
    ../llama.cpp/build-vnni/bin/llama-bench -m MODEL -t 18 -p 512,2048 -n 128 -r 3
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

GEMMA = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
QWEN = "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=("gemma", "qwen"), required=True)
    ap.add_argument("-m", "--model", default=None)
    ap.add_argument("-p", "--n-prompt", type=ints, default=[512])
    ap.add_argument("-n", "--n-gen", type=ints, default=[128])
    ap.add_argument("-r", "--reps", type=int, default=3)
    ap.add_argument("-t", "--threads", type=int, default=None)
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--hot-gb", type=float, default=None,
                    help="GB of hot experts on the GPU (default: the default budget)")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import numpy as np

    path = args.model or (GEMMA if args.arch == "gemma" else QWEN)
    max_len = max(args.n_prompt + args.n_gen) + 16
    g = None
    if args.arch == "gemma":
        from np_gemma.config import Config
        from np_gemma.gguf import GGUF
        from np_gemma.model import KVCache, Model
        src = GGUF(path)
        cfg = Config({"text_config": src.text_config()})
        m = Model(src, cfg).load_all(dtype="int4")
        new_cache = lambda: KVCache(cfg, max_len=max_len)  # noqa: E731
        if args.backend == "gpu":
            from np_gemma import gpu
            if args.hot_gb is not None:
                os.environ["NP_GEMMA_GPU_HOT_GB"] = str(args.hot_gb)
            g = gpu.ModelGPU(m)
        size = os.path.getsize(path) / 2 ** 30
    else:
        from np_gemma.qwen import QwenCache, QwenGGUFProgram
        m = QwenGGUFProgram(path)
        cfg = m.cfg
        new_cache = lambda: QwenCache(cfg, max_len)  # noqa: E731
        if args.backend == "gpu":
            from np_gemma.qwen_gpu import QwenGPU
            g = QwenGPU(m, hot_gb=args.hot_gb) if args.hot_gb is not None else QwenGPU(m)
        size = os.path.getsize(path) / 2 ** 30

    libc = ctypes.CDLL("libc.so.6")
    n_vocab = int(cfg.vocab_size)
    rand = lambda: libc.rand() % n_vocab  # noqa: E731

    def fresh():
        c = new_cache()
        if g is not None:
            g.attach(c)
        return c

    def decode(tokens, c, pos):
        """llama_decode of one batch, with the logits of its last token."""
        if g is None:
            if args.arch == "gemma" and len(tokens) > 1:
                h = m.prefill(tokens, c)
            else:
                h = m.forward(tokens, cache=c, start_pos=pos)
            m.logits(h[-1:])
        elif len(tokens) == 1:
            g.step([tokens[0]] if args.arch == "gemma" else tokens[0], pos)
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
    backend = "CPU"
    if g is not None:
        backend = "GPU %s" % ("%.1f GB hot" % args.hot_gb if args.hot_gb is not None else
                              "default hot")
    name = "gemma4 26B A4B (np_gemma)" if args.arch == "gemma" else "qwen3.6 35B A3B (np_gemma)"
    print("| model                      |       size | backend          | threads |   test |"
          "              t/s |")
    print("| -------------------------- | ---------: | ---------------- | ------: | -----: |"
          " ---------------: |")
    for tname, fn, n, warm in tests:
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
            name[:26], size, backend, os.environ.get("OMP_NUM_THREADS", "?"), tname,
            ts.mean(), std), flush=True)
    if g is not None and hasattr(g, "close"):
        g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
