"""Measure the packed-weight kernel against the float32 multiply.

For each matrix of the E4B text model this script reports:

    the size of the packed weight and of the float32 copy
    the time of `cops.ct_linear`, which reads the packed words
    the time of `ops.linear` on the float32 copy
    the throughput of each

It then runs every matrix once, which is what one decode token costs, and
reports the total. Point it at a copy of the checkpoint in two places to see
what the storage under the file does to the packed kernel: the packed kernel
reads the file on every token, and the float32 kernel does not.

Set OPENBLAS_NUM_THREADS=1. The kernel opens one OpenMP region for each
matrix, and a BLAS pool with more than one thread fights those regions. The
difference measured on a six-core machine is a factor of eight.

Run:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=ACTIVE \
        PYTHONPATH=. $PY scripts/bench_ct_kernel.py --snapshot "$SNAP4B" --label local
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 \
        PYTHONPATH=. $PY scripts/bench_ct_kernel.py --snapshot "$SNAP" --label sshfs
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PREFIX = "model.language_model."
HEAD = "lm_head"


def matrix_modules(cfg):
    """Return the matrix modules of the text model, in the order it reads them."""
    out = [PREFIX + "per_layer_model_projection", HEAD]
    for i in range(cfg.num_hidden_layers):
        p = PREFIX + "layers." + str(i) + "."
        plan = cfg.plan[i]
        for m in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                  "self_attn.q_proj", "self_attn.o_proj",
                  "per_layer_input_gate", "per_layer_projection"):
            out.append(p + m)
        if not plan.shared:
            out.append(p + "self_attn.k_proj")
            out.append(p + "self_attn.v_proj")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-f32", action="store_true")
    args = ap.parse_args()

    from np_gemma import cops, ops
    from np_gemma.ct import CompressedTensors
    from np_gemma.e4b import E4BConfig

    snap = os.path.expanduser(args.snapshot)
    path = os.path.join(snap, "model.safetensors")
    cfg = E4BConfig.load(os.path.join(snap, "config.json"))
    ct = CompressedTensors(path)
    mods = matrix_modules(cfg)
    rng = np.random.default_rng(7)

    label = args.label or snap
    print("=== %s ===" % label)
    print("  %s" % path)
    print("  %-46s %8s %9s %9s %9s" %
          ("matrix", "packed", "kernel", "f32", "speedup"))

    total_packed = 0
    total_f32 = 0
    total_kernel = 0.0
    total_f32_time = 0.0
    total_macs = 0
    per_bits = {}

    for name in mods:
        bits = ct.num_bits(name)
        packed = ct.packed_words(name)
        scale = ct.channel_scale(name)
        if packed is None or scale is None:
            continue
        rows, words = packed.shape
        cols = words * 32 // bits
        x = rng.standard_normal((args.tokens, cols)).astype(np.float32) * 0.05
        cand = [p for p in (2, 4) if p == bits]
        if not cand:
            continue

        t0 = time.time()
        for _ in range(args.repeats):
            got = cops.ct_linear(x, packed, scale, bits, cols)
        tk = (time.time() - t0) / args.repeats

        tf = float("nan")
        if not args.skip_f32:
            w = ct.dequant(name)
            t0 = time.time()
            for _ in range(args.repeats):
                want = ops.linear(x, w)
            tf = (time.time() - t0) / args.repeats
            err = float(np.abs(got - want).max()) / max(1e-9, float(np.abs(want).max()))
            del w
        else:
            err = float("nan")

        pb = ct.packed_bytes(name)
        total_packed += pb
        total_f32 += rows * cols * 4
        total_kernel += tk
        total_f32_time += tf if tf == tf else 0.0
        total_macs += rows * cols
        per_bits[bits] = per_bits.get(bits, 0) + rows * cols

        print("  %-46s %8.1f %9.3f %9.3f %8s" %
              (name.split("language_model.")[-1][:46], pb / 1e6, tk, tf,
               ("%.2fx" % (tf / tk)) if tf == tf else "-"))
        if err == err and err > 1e-5:
            print("      WARNING: relative error %.3g" % err)

    print("\n  one pass over every matrix, %d token(s)" % args.tokens)
    print("    packed bytes read   %8.3f GB" % (total_packed / 1e9))
    print("    float32 bytes read  %8.3f GB" % (total_f32 / 1e9))
    print("    multiply and add    %8.3f G" % (total_macs / 1e9))
    print("    packed kernel       %8.3f s  (%.1f GFLOP/s, %.2f GB/s packed)"
          % (total_kernel, 2 * total_macs / total_kernel / 1e9,
             total_packed / total_kernel / 1e9))
    if not args.skip_f32:
        print("    float32 multiply    %8.3f s  (%.1f GFLOP/s, %.2f GB/s)"
              % (total_f32_time, 2 * total_macs / total_f32_time / 1e9,
                 total_f32 / total_f32_time / 1e9))
        print("    speedup             %8.2fx" % (total_f32_time / total_kernel))
    for bits in sorted(per_bits):
        print("    %d-bit values         %8.3f G" % (bits, per_bits[bits] / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
