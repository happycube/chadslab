#!/usr/bin/env python3
"""Dump the metadata of a GGUF file. Do not read the tensor data."""
from __future__ import annotations

import struct
import sys


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B",
       10: "<Q", 11: "<q", 12: "<d"}


def read_value(f, t):
    if t in FMT:
        return struct.unpack(FMT[t], f.read(struct.calcsize(FMT[t])))[0]
    if t == 8:
        return read_string(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(n):
                read_string(f)
            return "<array of %d strings>" % n
        size = FMT.get(et)
        if size is None:
            raise ValueError("array type %d" % et)
        f.read(struct.calcsize(size) * n)
        return "<array of %d type %d>" % (n, et)
    raise ValueError("type %d" % t)


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        magic = f.read(4)
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print("magic %s version %d tensors %d kv %d" % (magic, version, n_tensors, n_kv))
        for _ in range(n_kv):
            key = read_string(f)
            t = struct.unpack("<I", f.read(4))[0]
            val = read_value(f, t)
            print("%-48s %s" % (key, val))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
