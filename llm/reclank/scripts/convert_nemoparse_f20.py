#!/usr/bin/env python3
"""Convert the Nemotron 3.5 Parse checkpoint (float32 safetensors) and its DFlash2 drafter (bfloat16) to
the files reclank loads (docs/NEMOTRON_PARSE.md): each matrix in the form of its original dtype, F20 for a
float32 one (16 significant bits), BF12 for a bfloat16 one (its values exactly); the vectors (biases, norms)
as stored. Two GGUF files and the configs:

    OUT/Nemotron-3.5-Parse-F20.gguf         the model
    OUT/dflash2/Nemotron-3.5-Parse-DFlash2-BF12.gguf
    OUT/config.json, preprocessor_config.json, generation_config.json, tokenizer*.json, ... (copies)

The model's tensors keep the checkpoint's names; four take a layout of the programs (the values the same):
conv1 [1024, 1152, 1] as [1024, 1152]; conv2 [1024, 1024, 1, 4] as the linear [1024, 4096] of 4 patches'
rows side by side (tap, then channel); pos_embed [1, 16384, 1152] as [16384, 1152]; RADIO's mlp.fc1 (and
its bias) padded with zero rows from 4304 to 4320 and mlp.fc2 with zero columns (F20 takes columns in
multiples of 32). A matrix that F20 (or BF12) cannot hold (a zeroed value above 2^-8 of its RMS, an Inf or
NaN) stays as stored, and the script says so.

    python scripts/convert_nemoparse_f20.py MODELS/Nemotron-3.5-Parse MODELS/Nemotron-3.5-Parse-F20
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.gguf import BF12, BF16, F20, F32, write_gguf  # noqa: E402
from reclank.formats.safetensors import SafeTensors  # noqa: E402
from reclank.models.nemoparse.model import F20_FILE, MLP_PAD, program_layout  # noqa: E402

COPY = ("config.json", "preprocessor_config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "LICENSE",
        "TOKENIZER_ATTRIBUTION.md")


def model_tensors(W):
    """(name, float32 array in the programs' layout: model.program_layout) of each tensor, one at a time."""
    for n in W.names():
        if n.endswith("summary_idxs"):
            assert list(W.get(n, dtype=None)) == [0, 1], "a summary of other tokens than 0 and 1"
            continue
        yield n, program_layout(n, W.get(n, dtype=np.float32))


def fits(rep):
    return not (rep["nonfinite"] > 0 or rep["largest"] > rep["rms"] * 2.0 ** -8)


def convert(entries, label):
    """GGUF tensors of entries [(name, kind ("f32" or "bf16"), array)]: (name, dims, type, make)."""
    out, kept = [], []
    for name, kind, a in entries:
        dims = list(reversed(a.shape))
        if a.ndim == 2 and a.shape[1] % 32 == 0 and a.shape[0] >= 2:
            if kind == "f32":
                rows, rep = cops.kq_f32_to_f20(a, a.shape[1])
                if fits(rep):
                    out.append((name, dims, F20, lambda r=rows: r))
                    continue
            else:
                rows, rep = cops.kq_bf16_to_bf12(a, a.shape[1])
                if fits(rep):
                    out.append((name, dims, BF12, lambda r=rows: r))
                    continue
            kept.append(name)
        out.append((name, dims, F32 if kind == "f32" else BF16, lambda v=a: v))
    print("%s: %d tensors, %d matrices %s, %d kept as stored%s" % (
        label, len(out), sum(t in (F20, BF12) for _n, _d, t, _m in out), "F20" if label == "model" else "BF12",
        len(kept), (": " + ", ".join(kept)) if kept else ""), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(a.out, "dflash2"), exist_ok=True)
    for f in COPY:
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    W = SafeTensors(os.path.join(a.src, "model.safetensors"))
    # (summary_idxs, int64 [0, 1], is not a weight: the programs gather rows 0 and 1)
    assert all(W.dtype(n) == "F32" for n in W.names() if not n.endswith("summary_idxs")), "float32 tensors"
    meta = [("general.architecture", "nemotron_parse"), ("general.name", "Nemotron-3.5-Parse"),
            ("nemotron_parse.mlp_padded_to", MLP_PAD), ("nemotron_parse.conv2_layout", "tap_channel")]
    tensors = convert(((n, "f32", v) for n, v in model_tensors(W)), "model")
    path = os.path.join(a.out, F20_FILE)
    write_gguf(path, meta, tensors)
    W.release_pages()
    print("%s: %.2f GB" % (path, os.path.getsize(path) / 1e9), flush=True)
    dsrc = os.path.join(a.src, "dflash2")
    if os.path.isdir(dsrc):
        shutil.copy(os.path.join(dsrc, "config.json"), os.path.join(a.out, "dflash2", "config.json"))
        D = SafeTensors(os.path.join(dsrc, "model.safetensors"))
        ent = []
        for n in D.names():
            dt = D.dtype(n)
            assert dt in ("BF16", "F32"), (n, dt)
            v = D.get(n, dtype=None)
            if dt == "BF16":
                v = np.ascontiguousarray(v).view(np.uint32) >> 16
                v = v.astype(np.uint16)
            ent.append((n, "bf16" if dt == "BF16" else "f32", v))
        path = os.path.join(a.out, "dflash2", "Nemotron-3.5-Parse-DFlash2-BF12.gguf")
        write_gguf(path, [("general.architecture", "dflash2"), ("general.name", "Nemotron-3.5-Parse-DFlash2")],
                   convert(ent, "drafter"))
        D.release_pages()
        print("%s: %.2f GB" % (path, os.path.getsize(path) / 1e9), flush=True)
    print("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
