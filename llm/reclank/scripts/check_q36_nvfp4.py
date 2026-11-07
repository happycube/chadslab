#!/usr/bin/env python3
"""Qwen3.6-35B-A3B NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4: reclank/formats/q35nv_st.py, from the
weightkeeper when it holds the checkpoint) against transformers' float32 on the same weights
dequantized (an .npz of each prompt: its ids, rows and the logits of those rows).

- the CPU program (QwenGGUFProgram, one pass over the prompt): the logits of the rows;
- the GPU (QwenGPU): the prompt but its last 16 tokens as a prefill, then those 16 as steps: the
  logits of the last 16 rows.

KL(transformers || reclank) over the vocabulary (float64) and the top-1 agreement of each set.

    python scripts/check_q36_nvfp4.py MODELS/Qwen3.6-35B-A3B-NVFP4 q36_short.npz q36_long.npz [--cpu-only]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.qwen.qwen35 import QwenCache, QwenGGUFProgram  # noqa: E402


def kl(ref, got):
    """The mean KL(ref || got) of rows of logits, and the top-1 agreement."""
    a, b = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    la = a - a.max(-1, keepdims=True)
    la -= np.log(np.exp(la).sum(-1, keepdims=True))
    lb = b - b.max(-1, keepdims=True)
    lb -= np.log(np.exp(lb).sum(-1, keepdims=True))
    return float((np.exp(la) * (la - lb)).sum(-1).mean()), float((a.argmax(-1) == b.argmax(-1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    m = QwenGGUFProgram(a.model)
    cfg = m.cfg
    print("load %.0f s (%s)" % (time.time() - t0, type(m.g).__name__), flush=True)
    dev = None
    if not a.cpu_only:
        from reclank.models.qwen.gpu35 import QwenGPU
        dev = QwenGPU(m)
    for path in a.refs:
        h = np.load(path)
        ids, rows, ref = h["ids"].astype(np.int64), h["rows"], h["logits"]
        n = len(ids)
        name = os.path.basename(path)
        if not a.gpu_only:
            t0 = time.time()
            hs = m.forward(ids, QwenCache(cfg, n + 8))
            lg = m.logits(hs[rows])
            k, t1 = kl(ref, lg)
            print("%s: %d tokens; CPU one pass (%.0f s): KL %.5f, top-1 %.3f over %d rows" % (
                name, n, time.time() - t0, k, t1, len(rows)), flush=True)
        if dev is not None:
            last = [j for j, r in enumerate(rows) if r >= n - 16]
            cache = QwenCache(cfg, n + 8)
            dev.attach(cache)
            t0 = time.time()
            dev.prefill(ids[:n - 16])
            out = []
            for p in range(n - 16, n):
                dev.step(int(ids[p]), p)
                out.append(np.array(dev.logits()).reshape(-1)[:ref.shape[1]])
            dev.detach(cache)
            k, t1 = kl(ref[last], np.stack(out))
            print("%s: GPU prefill of %d + 16 steps (%.0f s): KL %.5f, top-1 %.3f" % (
                name, n - 16, time.time() - t0, k, t1), flush=True)


if __name__ == "__main__":
    main()
