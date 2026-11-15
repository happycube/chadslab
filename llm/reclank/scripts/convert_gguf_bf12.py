#!/usr/bin/env python3
"""Convert a GGUF file of bfloat16 matrices to one with the matrices in BF12 (the bfloat16 values in
12.25 bits, plan-scripts/BF12_PLAN.md), as convert_dgemma_bf12.py does from safetensors: the same
names, metadata and order; each BF16 tensor of 2 or more dims whose rows are a multiple of 32 goes
to BF12 unless BF12 cannot hold it (a zeroed value above 2^-8 of its RMS, or an Inf or NaN: it stays
bfloat16); the other tensors are copied as they are. Two passes: the BF12 report of every matrix
first (the types go in the header), then the write, converting each tensor again when the writer
comes to it (one tensor in memory).

    python scripts/convert_gguf_bf12.py MODELS/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-BF16.gguf \\
        MODELS/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-BF12.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.gguf import BF12, BF16, GGUF, tensor_bytes, write_gguf  # noqa: E402


def plain(v):
    """A metadata value as write_gguf takes it."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    g = GGUF(a.src, stage=False)
    t0 = time.time()

    def bits(n):
        raw, dims, _t = g.raw(n)
        return np.ascontiguousarray(raw).view(np.uint16).reshape(-1, int(dims[0]))

    # pass 1: the BF12 report of each bfloat16 matrix
    kind, report = {}, {}
    for n, (dims, t, _o) in g.tensors.items():
        kind[n] = t
        if t != BF16 or len(dims) < 2 or int(dims[0]) % 32:
            continue
        _rows, r = cops.kq_bf16_to_bf12(bits(n), int(dims[0]))
        report[n] = r
        if not (r["nonfinite"] > 0 or r["largest"] > r["rms"] * 2.0 ** -8):
            kind[n] = BF12
    kept = [n for n in report if kind[n] == BF16]
    zeroed = sum(r["zeroed"] for r in report.values())
    vals = sum(int(np.prod(g.tensors[n][0])) for n in report)
    print("pass 1: %d matrices (%.2f G values) %.0f s: %d zeroed values (%.5f%%), %d kept bfloat16 %s"
          % (len(report), vals / 1e9, time.time() - t0, zeroed, 100.0 * zeroed / max(vals, 1), len(kept), kept),
          flush=True)

    def maker(n):
        if kind[n] == BF12:
            return lambda: cops.kq_bf16_to_bf12(bits(n), int(g.tensors[n][0][0]))[0].reshape(-1)
        return lambda: np.ascontiguousarray(g.raw(n)[0]).view(np.uint8).reshape(-1)

    tensors = [(n, tuple(int(d) for d in dims), kind[n], maker(n)) for n, (dims, _t, _o) in g.tensors.items()]
    total = sum(tensor_bytes(d, t) for _n, d, t, _m in tensors)
    print("%d tensors, %.2f GB -> %s" % (len(tensors), total / 1e9, a.out), flush=True)

    meta = {}
    for k, v in g.meta.items():
        if k in ("general.alignment", "general.file_type", "general.quantization_version"):
            continue
        if isinstance(v, dict) and "__array__" in v:
            v = g.meta_strings(k)
        meta[k] = plain(v)
    meta["general.name"] = "%s, BF12 (reclank)" % g.meta.get("general.name", os.path.basename(a.src))
    meta["general.source"] = os.path.basename(a.src)
    meta["reclank.bf12.kept_bf16"] = ",".join(kept)
    meta["reclank.bf12.zeroed"] = int(zeroed)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"
    write_gguf(tmp, list(meta.items()), tensors)
    os.replace(tmp, a.out)
    json.dump({n: {k: v for k, v in r.items() if k != "worst"} for n, r in report.items()},
              open(a.out + ".bf12.json", "w"), indent=0)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
