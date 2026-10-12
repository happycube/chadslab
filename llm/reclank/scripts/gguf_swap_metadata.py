#!/usr/bin/env python3
"""Write a GGUF with the tensors of one file and the metadata of another.

The Unsloth Q4_0 file of the 26B has the weights of the unquantized QAT
release, but its own metadata: the canonical template (2026-07-09) with a
change for tool arguments in a string, add_bos_token 0, and its own names.
The Google file has the canonical template as Google gives it and
add_bos_token 1. This script copies all the keys of --meta (the template, the
tokenizer, the model keys) and all the tensors of --tensors:

    PYTHONPATH=../llama.cpp/gguf-py python scripts/gguf_swap_metadata.py \\
        --meta models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf \\
        --tensors models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \\
        --out models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-Q4_0-fixed.gguf

The two files must have the same architecture keys; a key that differs
outside general.* and tokenizer.* stops the script. The tensor names and
shapes of --tensors must be the names and shapes that --meta describes.
"""
from __future__ import annotations

import argparse

import gguf


def keys(reader):
    """The metadata of a reader as {name: (type, sub type, value)}."""
    out = {}
    for f in reader.fields.values():
        if f.name.startswith("GGUF."):
            continue
        t = f.types[0]
        sub = f.types[-1] if t == gguf.GGUFValueType.ARRAY else None
        out[f.name] = (t, sub, f.contents())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", required=True, help="the file that gives the metadata")
    ap.add_argument("--tensors", required=True, help="the file that gives the tensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="a new general.name")
    args = ap.parse_args()
    meta, tens = gguf.GGUFReader(args.meta), gguf.GGUFReader(args.tensors)
    km, kt = keys(meta), keys(tens)
    arch = km[gguf.Keys.General.ARCHITECTURE][2]
    if kt[gguf.Keys.General.ARCHITECTURE][2] != arch:
        raise SystemExit("the architectures differ")
    for k in sorted(set(km) | set(kt)):
        if k.startswith(("general.", "tokenizer.")):
            continue
        if km.get(k, (None, None, None))[2] != kt.get(k, (None, None, None))[2]:
            raise SystemExit("the key %s differs" % k)
    shapes = {t.name: (tuple(t.shape), t.tensor_type) for t in meta.tensors}
    for t in tens.tensors:
        if t.name not in shapes or shapes[t.name][0] != tuple(t.shape):
            raise SystemExit("the tensor %s is not in --meta with that shape" % t.name)
    if len(shapes) != len(tens.tensors):
        raise SystemExit("the tensor counts differ")

    w = gguf.GGUFWriter(args.out, arch=arch, endianess=tens.endianess)
    for name, (t, sub, value) in km.items():
        if name == gguf.Keys.General.ARCHITECTURE:
            continue
        if name == gguf.Keys.General.NAME and args.name:
            value = args.name
        w.add_key_value(name, value, t, sub_type=sub)
    for t in tens.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in tens.tensors:
        w.write_tensor_data(t.data, tensor_endianess=tens.endianess)
    w.close()
    print("wrote %s: %d keys of %s, %d tensors of %s"
          % (args.out, len(km), args.meta, len(tens.tensors), args.tensors))


if __name__ == "__main__":
    main()
