#!/usr/bin/env python3
"""
Print the safetensors header of a Gemma 4 checkpoint: dtype histogram and the
largest tensors. Useful for confirming what a checkpoint actually stores.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from huggingface_hub import hf_hub_download


def header_of(path: Path) -> dict:
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=_bootstrap.MODEL_ID)
    ap.add_argument("--filename", default="model.safetensors")
    args = ap.parse_args()

    path = Path(hf_hub_download(args.model, args.filename))
    hdr = header_of(path)
    hdr.pop("__metadata__", None)

    dtypes = collections.Counter(t["dtype"] for t in hdr.values())
    total = sum(t["data_offsets"][1] - t["data_offsets"][0] for t in hdr.values())
    print(f"{args.model}/{args.filename}")
    print(f"  tensors : {len(hdr)}")
    print(f"  bytes   : {total / 1e9:.2f} GB")
    print(f"  dtypes  : {dict(dtypes)}")
    print("  largest :")
    for name, t in sorted(hdr.items(),
                          key=lambda kv: kv[1]["data_offsets"][1] - kv[1]["data_offsets"][0],
                          reverse=True)[:10]:
        nbytes = t["data_offsets"][1] - t["data_offsets"][0]
        print(f"    {nbytes / 1e6:9.1f} MB  {t['dtype']:8} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
