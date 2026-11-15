#!/usr/bin/env python3
"""Convert a Nemotron 3.5 Embed or Rerank checkpoint (the bfloat16 safetensors) to the file reclank
loads (docs/NEMOTRON_EMBED_PLAN.md): one GGUF file (models/nemotron_embed/model.py BF12_FILE) with each
matrix in BF12 (the bfloat16 values in 12.25 bits), the vectors (norms; the reranker's score row) as
float32 (their bfloat16 values exactly), and the small files of the checkpoint beside it:

    OUT/model-bf12.gguf (BF12_FILE)
    OUT/config.json, processor_config.json, tokenizer.json, ... (copies)

The tensors keep the checkpoint's names. Two take the layout of the programs (the values the same):
the patch conv [1024, 3, 14, 14] as the linear [1024, 608] (the 588 columns, then 20 of zeros: BF12
takes columns in multiples of 32), and the patch merger's columns in the order of ENC_GATHER's rows
(the block position, then the channel; the checkpoint's unfold has the channel first). A matrix that
BF12 cannot hold (a zeroed value above 2^-8 of its RMS, an Inf or NaN) stays bfloat16, and the script
says so. A float32 matrix of a checkpoint would go to F20 (these have none). Two passes: the BF12
report of every matrix first (the types go in the header), then the write, converting each tensor
again when the writer comes to it (one tensor in memory).

    python scripts/convert_nemotron_embed_bf12.py MODELS/Nemotron-3.5-Embed-1B-BF16 \\
        MODELS/Nemotron-3.5-Embed-1B-BF12
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.gguf import BF12, BF16, F20, F32, tensor_bytes, write_gguf  # noqa: E402
from reclank.formats.safetensors import Shards  # noqa: E402
from reclank.models.nemotron_embed.model import BF12_FILE, program_layout  # noqa: E402

COPY = ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "config_sentence_transformers.json", "README.md", "LICENSE", "NOTICE")


def fits(rep):
    return not (rep["nonfinite"] > 0 or rep["largest"] > rep["rms"] * 2.0 ** -8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    st = Shards(a.src)
    name = os.path.basename(os.path.normpath(a.out))
    t0 = time.time()

    def tensor(n):
        """(the tensor in the programs' layout: uint16 bfloat16 bits or float32, its dtype)."""
        if st.dtype(n) == "BF16":
            return program_layout(n, np.ascontiguousarray(st.get_bf16(n))), "bf16"
        return program_layout(n, np.ascontiguousarray(st.get(n), np.float32)), "f32"

    def matrix(x):
        return x.ndim == 2 and x.shape[0] >= 2 and x.shape[1] % 32 == 0

    # pass 1: the report of each matrix
    kind, report, vals = {}, {}, 0
    for n in st.names():
        x, dt = tensor(n)
        if not matrix(x):
            kind[n] = F32
            continue
        vals += x.size
        _rows, r = cops.kq_bf16_to_bf12(x, x.shape[1]) if dt == "bf16" else cops.kq_f32_to_f20(x, x.shape[1])
        report[n] = r
        kind[n] = (BF12 if dt == "bf16" else F20) if fits(r) else (BF16 if dt == "bf16" else F32)
    kept = [n for n in report if kind[n] in (BF16, F32)]
    zeroed = sum(r["zeroed"] for r in report.values())
    print("pass 1: %d matrices (%.2f G values) %.0f s: %d zeroed values (%.6f%%), %d kept as stored %s"
          % (len(report), vals / 1e9, time.time() - t0, zeroed, 100.0 * zeroed / max(vals, 1), len(kept), kept),
          flush=True)

    def maker(n):
        def make():
            x, _dt = tensor(n)
            if kind[n] == BF12:
                return cops.kq_bf16_to_bf12(x, x.shape[1])[0].reshape(-1)
            if kind[n] == F20:
                return cops.kq_f32_to_f20(x, x.shape[1])[0].reshape(-1)
            if kind[n] == BF16:
                return x.reshape(-1)
            return (x.astype(np.uint32) << 16).view(np.float32).reshape(-1) if x.dtype == np.uint16 else \
                x.reshape(-1)
        return make

    tensors = [(n, tuple(reversed(tensor(n)[0].shape)), kind[n], maker(n)) for n in st.names()]
    total = sum(tensor_bytes(d, t) for _n, d, t, _m in tensors)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, BF12_FILE)
    print("%d tensors, %.2f GB -> %s" % (len(tensors), total / 1e9, path), flush=True)
    meta = [("general.name", "%s (reclank)" % name),
            ("general.source", "%s (bfloat16 safetensors)" % os.path.basename(os.path.normpath(a.src))),
            ("reclank.layout", "vision_tower.patch_conv.weight: [1024, 608], the 588 values of a patch then zeros; "
                               "multi_modal_projector.patch_merger.merging_layer.weight: columns (block position, "
                               "channel)"),
            ("reclank.bf12.kept", ",".join(kept)), ("reclank.bf12.zeroed", int(zeroed))]
    tmp = path + ".part"
    write_gguf(tmp, meta, tensors)
    os.replace(tmp, path)
    json.dump({n: {k: v for k, v in r.items() if k != "worst"} for n, r in report.items()},
              open(path + ".bf12.json", "w"), indent=0)
    for f in COPY:
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
