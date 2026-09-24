"""Check the NumPy compressed-tensors reader against the library.

The library `compressed_tensors` gives the reference decode. This script reads
the same tensors with `np_gemma.ct` and compares.

It checks four kinds of weight:

    a 2-bit weight with one scale for each row      embed_tokens
    a 2-bit weight with one scale for each group    embed_tokens_per_layer
    a 4-bit weight with one scale for each row      the decoder projections
    an 8-bit weight with one scale for each row     per_layer_input_gate

It also checks that `row()` equals the matching row of `dequant()`.

Run:

    PYTHONPATH=. PY=../gemma4-12b-qat-pytorch/.venv/bin/python \
        $PY scripts/check_ct.py --snapshot "$SNAP"
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from np_gemma.ct import CompressedTensors

CASES = [
    ("model.language_model.embed_tokens", "2-bit channel"),
    ("model.language_model.embed_tokens_per_layer", "2-bit group-256"),
    ("lm_head", "2-bit channel"),
    ("model.language_model.layers.0.mlp.gate_proj", "4-bit channel"),
    ("model.language_model.layers.5.self_attn.q_proj", "4-bit channel"),
    ("model.language_model.layers.0.per_layer_input_gate", "8-bit channel"),
    ("model.language_model.layers.0.per_layer_projection", "8-bit channel"),
]


def reference(snapshot, name):
    """Decode one weight with the compressed_tensors library."""
    import torch
    from safetensors import safe_open
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization.lifecycle.forward import dequantize

    path = os.path.join(snapshot, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        if name + ".weight_packed" in keys:
            packed = f.get_tensor(name + ".weight_packed")
            scale = f.get_tensor(name + ".weight_scale").to(torch.float32)
            shape = torch.Size([int(x) for x in f.get_tensor(name + ".weight_shape")])
            bits = (packed.shape[-1] * 32) // shape[-1]
            q = unpack_from_int32(packed, bits, shape)
            return dequantize(q, scale, None).numpy()
        w = f.get_tensor(name + ".weight")
        if w.dtype == torch.int8:
            scale = f.get_tensor(name + ".weight_scale").to(torch.float32)
            return dequantize(w, scale, None).numpy()
        return w.to(torch.float32).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--rows", type=int, default=8,
                    help="how many rows to compare for the row path")
    args = ap.parse_args()

    ct = CompressedTensors(os.path.join(args.snapshot, "model.safetensors"))
    bad = 0
    for name, label in CASES:
        bits = ct.num_bits(name)
        strat, group = ct.strategy(name)
        mine = ct.dequant(name)
        ref = reference(args.snapshot, name)
        ok_shape = mine.shape == ref.shape
        diff = np.abs(mine - ref).max() if ok_shape else float("nan")
        same = ok_shape and np.array_equal(mine, ref)
        close = ok_shape and np.allclose(mine, ref, rtol=0, atol=1e-5)
        print("%-56s %-14s bits=%s %-8s group=%-4d shape=%s" %
              (name.split("language_model.")[-1][:56], label, bits, strat, group,
               tuple(mine.shape)))
        print("    exact=%s close=%s max|diff|=%.3g shape_ok=%s" %
              (same, close, diff, ok_shape))
        if not close:
            bad += 1

    # The row path must equal the matching rows of the whole-matrix path.
    print("\n--- row() against dequant() ---")
    rng = np.random.default_rng(0)
    for name, label in CASES:
        full = ct.dequant(name)
        n = full.shape[0]
        picks = rng.choice(n, size=min(args.rows, n), replace=False)
        worst = 0.0
        exact = True
        for i in picks:
            r = ct.row(name, int(i))
            if r.shape != full[i].shape:
                print("    %s row %d shape %s != %s" % (name, i, r.shape, full[i].shape))
                bad += 1
                continue
            worst = max(worst, float(np.abs(r - full[i]).max()))
            exact = exact and np.array_equal(r, full[i])
        print("  %-56s rows=%d exact=%s max|diff|=%.3g" %
              (name.split("language_model.")[-1][:56], len(picks), exact, worst))
        if not exact:
            bad += 1

    # The rows path must equal a range of the whole-matrix path.
    print("\n--- rows() against dequant() ---")
    for name, label in CASES:
        full = ct.dequant(name)
        stop = min(5, full.shape[0])
        block = ct.rows(name, 0, stop)
        ok = np.array_equal(block, full[:stop])
        print("  %-56s rows 0..%d exact=%s" %
              (name.split("language_model.")[-1][:56], stop, ok))
        if not ok:
            bad += 1

    print("\n%s" % ("FAIL" if bad else "OK: every decode matches the library"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
