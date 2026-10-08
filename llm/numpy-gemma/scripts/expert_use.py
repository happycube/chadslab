#!/usr/bin/env python3
"""Measure how often the router of the 26B selects each expert.

SPLIT_PLAN.md, phase 0, the hot experts on the GPU. The GPU can hold only a
part of the experts. If the router selects some experts much more often
than others, the GPU can hold those. This script runs the prompt pass on
several texts and counts the selections of each expert in each layer.

Then it selects the most used experts on one text and finds the share of
the selections on the other texts that those experts get. Compare it with
the share of the experts themselves, for example 1700 of 3840 (44%). If the
two shares are near, the use is uniform, and hot experts do not help.

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/expert_use.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from np_gemma import Model
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
HERE = os.path.join(os.path.dirname(__file__), "..")
TEXTS = {
    "readme": os.path.join(HERE, "README.md"),
    "python": os.path.join(HERE, "np_gemma", "model.py"),
    "c": os.path.join(HERE, "np_gemma", "csrc", "bf16_linear.c"),
    "notes": os.path.join(HERE, "..", "Gemma LLM Runtime Learning Plan.md"),
}


def count(model, cfg, ids):
    """Return the selections of each expert, shape (layers, experts)."""
    cnt = np.zeros((cfg.num_hidden_layers, cfg.num_experts), np.int64)

    def hook(key, value):
        if key.endswith("router.top_idx"):
            i = int(key.split(".")[1])
            np.add.at(cnt[i], value.astype(np.int64).reshape(-1), 1)

    model.forward(ids, hook=hook)
    return cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--tokens", type=int, default=800)
    ap.add_argument("--out", default=None, help="Save the counts to this .npz file.")
    ap.add_argument("--budgets", type=int, nargs="+", default=[480, 960, 1700])
    args = ap.parse_args()

    g = GGUF(args.gguf)
    cfg = Config({"text_config": g.text_config()})
    tok = Tokenizer.from_gguf(g)
    model = Model(g, cfg).load_all(dtype="int4")
    counts = {}
    for name, path in TEXTS.items():
        ids = tok.encode(open(path).read())[:args.tokens]
        counts[name] = count(model, cfg, ids)
        print("%-7s %d tokens" % (name, len(ids)), flush=True)
    if args.out:
        np.savez(args.out, **counts)

    total = cfg.num_hidden_layers * cfg.num_experts
    print("\nthe share of the selections that the most used experts of each layer get")
    print("(one text; the same count of experts in each layer)")
    for name, c in counts.items():
        s = np.sort(c, axis=1)[:, ::-1]
        share = s.cumsum(axis=1) / s.sum(axis=1, keepdims=True)
        row = "  ".join("top %3d: %3.0f%%" % (k, 100 * share[:, k - 1].mean())
                        for k in (8, 16, 32, 57, 64))
        print("%-7s %s" % (name, row))

    print("\nthe experts selected on one text, the share on the other texts")
    print("(a global selection over all layers; uniform use gives the share of experts)")
    names = list(counts)
    for b in args.budgets:
        print("%d of %d experts (%.0f%%):" % (b, total, 100 * b / total))
        for src in names:
            hot = np.argsort(counts[src].reshape(-1))[::-1][:b]
            cells = []
            for dst in names:
                c = counts[dst].reshape(-1)
                cells.append("%s %3.0f%%" % (dst, 100 * c[hot].sum() / c.sum()))
            print("  from %-7s -> %s" % (src, "  ".join(cells)))
    g.close()


if __name__ == "__main__":
    main()
