"""Benchmark a GGUF model in the same form as llama-bench.

The script times a prompt pass of --prompt tokens, then a generation of --gen
tokens. Give the same numbers to llama-bench -p and -n for a comparison:

    llama-bench -m FILE -p 512 -n 128 -r 3

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=18 OMP_WAIT_POLICY=ACTIVE \
        PYTHONPATH=. $PY scripts/bench_gguf_models.py --gguf FILE \
        --prompt 512 --gen 128 --reps 3

The script selects the model class from the file. A file with a per-layer
embedding width uses the E4B class. Every other file uses the 12B class.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A repeating prompt gives a prompt of any length without a tokenizer.
SEED = [2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107]


def build(path, dtype):
    """Return (source, model, config, kind)."""
    from np_gemma.gguf import GGUF

    g = GGUF(path)
    tc = g.text_config()
    if tc.get("hidden_size_per_layer_input"):
        from np_gemma.e4b import E4B, E4BConfig

        cfg = E4BConfig({"text_config": tc})
        return g, E4B(g, cfg, mode=dtype), cfg, "e4b"
    from np_gemma.config import Config
    from np_gemma.model import Model

    cfg = Config({"text_config": tc})
    return g, Model(g, cfg).load_all(dtype=dtype), cfg, "model"


def make_cache(cfg, kind, max_len):
    if kind == "e4b":
        from np_gemma.e4b import E4BCache

        return E4BCache(cfg, max_len=max_len)
    from np_gemma.model import KVCache

    return KVCache(cfg, max_len=max_len)


def prompt_pass(model, kind, ids, cache):
    if kind == "e4b":
        return model.forward(ids, cache=cache, start_pos=0)
    return model.prefill(ids, cache)


def one_step(model, tok, cache, pos):
    return model.forward([tok], cache=cache, start_pos=pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dtype", default="int4")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    print("OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s"
          % (os.environ.get("OMP_NUM_THREADS", "unset"),
             os.environ.get("OPENBLAS_NUM_THREADS", "unset")))

    g, model, cfg, kind = build(os.path.expanduser(args.gguf), args.dtype)
    label = args.label or os.path.basename(args.gguf)
    ids = (SEED * (args.prompt // len(SEED) + 1))[:args.prompt]
    max_len = args.prompt + args.gen + 8

    # Warm the file cache and the weight cache.
    prompt_pass(model, kind, ids[:32], make_cache(cfg, kind, max_len))

    best = 1e9
    for _ in range(args.reps):
        cache = make_cache(cfg, kind, max_len)
        t0 = time.perf_counter()
        hidden = prompt_pass(model, kind, ids, cache)
        best = min(best, time.perf_counter() - t0)
    pp = args.prompt / best

    # The decode continues from the cache of the last prompt pass.
    nxt = int(np.argmax(model.logits(hidden[-1:])[0]))
    pos = args.prompt
    t0 = time.perf_counter()
    for _ in range(args.gen):
        hidden = one_step(model, nxt, cache, pos)
        pos += 1
        nxt = int(np.argmax(model.logits(hidden[-1:])[0]))
    gen_s = time.perf_counter() - t0
    tg = args.gen / gen_s

    print("%-40s pp%-4d %8.2f t/s   tg%-4d %8.2f t/s"
          % (label, args.prompt, pp, args.gen, tg))
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
