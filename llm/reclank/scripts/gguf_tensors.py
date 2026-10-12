#!/usr/bin/env python3
"""List the tensors of a GGUF file. Group the names to show the layer layout."""
from __future__ import annotations

import re
import struct
import sys


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B",
       10: "<Q", 11: "<q", 12: "<d"}


def skip_value(f, t):
    if t in FMT:
        f.read(struct.calcsize(FMT[t]))
    elif t == 8:
        read_string(f)
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(n):
                read_string(f)
        else:
            f.read(struct.calcsize(FMT[et]) * n)


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        struct.unpack("<I", f.read(4))
        struct.unpack("<I", f.read(4))
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            read_string(f)
            t = struct.unpack("<I", f.read(4))[0]
            skip_value(f, t)
        groups = {}
        for _ in range(n_tensors):
            name = read_string(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            ttype = struct.unpack("<I", f.read(4))[0]
            struct.unpack("<Q", f.read(8))
            key = re.sub(r"blk\.\d+\.", "blk.N.", name)
            groups.setdefault((key, dims, ttype), 0)
            groups[(key, dims, ttype)] += 1
    print("tensors: %d" % n_tensors)
    for (key, dims, ttype), count in sorted(groups.items()):
        print("%-44s x%-4d dims=%-22s type=%d" % (key, count, str(tuple(dims)), ttype))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
