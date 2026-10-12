"""Show the tensor layout of the E4B mobile-ct compressed-tensors checkpoint.

The checkpoint uses the compressed-tensors "pack-quantized" format. Each
quantized weight is stored as a packed integer tensor plus a scale, and often
a shape tensor that records the logical shape.

This script prints, for a chosen layer:

    the tensor name, the dtype, and the shape

Use it to learn the names that the loader must read.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter

import numpy as np

NP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U64": np.uint64, "U32": np.uint32, "U16": np.uint16, "U8": np.uint8,
    "BOOL": np.bool_,
}


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors")
    ap.add_argument("--layer", type=int, action="append", default=None)
    ap.add_argument("--grep", default=None)
    ap.add_argument("--prefixes", action="store_true")
    args = ap.parse_args()

    h = read_header(args.safetensors)

    if args.prefixes:
        pref = Counter()
        for name, t in h.items():
            parts = name.split(".")
            pref[".".join(parts[:4])] += 1
        print("--- prefixes ---")
        for k, v in sorted(pref.items(), key=lambda x: -x[1]):
            print("  %-64s %d" % (k, v))

    dt = Counter(t["dtype"] for t in h.values())
    total_bytes = sum(t["data_offsets"][1] - t["data_offsets"][0] for t in h.values())
    print("\n--- totals ---")
    print("  tensors:", len(h), " bytes:", total_bytes, "(%.2f GB)" % (total_bytes / 1e9))
    for k, v in dt.items():
        nb = sum(t["data_offsets"][1] - t["data_offsets"][0]
                 for t in h.values() if t["dtype"] == k)
        print("  %-6s %8d tensors  %10d bytes  (%.3f GB)" % (k, v, nb, nb / 1e9))

    # Group by the "kind" suffix, so the reader learns which suffixes matter.
    kinds = Counter()
    for name in h:
        parts = name.split(".")
        kinds[".".join(parts[-2:]) if parts[-1] in
              ("weight", "weight_packed", "weight_scale", "weight_shape", "weight_zero_point",
               "input_scale", "output_scale", "bias") else parts[-1]] += 1
    print("\n--- suffix kinds ---")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1])[:40]:
        print("  %-46s %d" % (k, v))

    for layer in (args.layer or []):
        p = "model.language_model.layers.%d." % layer
        print("\n--- layer %d ---" % layer)
        for name in sorted(n for n in h if n.startswith(p)):
            t = h[name]
            print("  %-72s %-5s %s" % (name[len(p):], t["dtype"], t["shape"]))

    if args.grep:
        print("\n--- grep %r ---" % args.grep)
        for name in sorted(n for n in h if args.grep in n):
            t = h[name]
            print("  %-78s %-5s %s" % (name, t["dtype"], t["shape"]))

    # Always show the top-level tensors, they are the interesting ones.
    print("\n--- top level (no .layers.) ---")
    for name in sorted(n for n in h if ".layers." not in n):
        t = h[name]
        print("  %-78s %-5s %s" % (name, t["dtype"], t["shape"]))


if __name__ == "__main__":
    sys.exit(main())
