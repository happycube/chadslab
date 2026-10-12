#!/usr/bin/env python3
"""The size of the GPU programs of a Gemma 4 model, by component: the
records (224 bytes each), the slots, and the device memory of the arrays
the compiler made for the program (its temporaries), as compiled without
packing (RECLANK_PACK=0: the buffers of PoolCompiler) and with it
(ir/alloc.py: one arena by liveness), and the opcodes the programs use.
The programs are compiled on the host (no GPU run): a step, and groups of
the sizes given.

    python scripts/program_size.py GGUF [T ...]      (default 4 16 1024 2048)

The 26B compiles its groups as on a small GPU: the hot experts in the pool
(stub stores), the cold ones on the CPU (no buffers of cold experts).
"""
from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RECLANK_GPU", "1")
os.environ.setdefault("RECLANK_DAX_STAGE", "0")

import numpy as np  # noqa: E402

from reclank import Model  # noqa: E402
from reclank.formats.gguf import GGUF  # noqa: E402
from reclank.ir import alloc, opcodes  # noqa: E402
from reclank.models.gemma4 import gpu  # noqa: E402
from reclank.models.gemma4.config import Config  # noqa: E402

NAMES = {code: name for name, code, _runners, _text in opcodes.TABLE}


def used_bytes(prog):
    """The bytes of the temporaries the records use (each array once)."""
    temps = {a.ctypes.data: a.nbytes for a in prog.temps if a.nbytes}
    arena = getattr(prog, "arena", None)
    if arena is not None:
        temps[arena.ctypes.data] = arena.nbytes
    starts = np.array(sorted(temps), dtype=np.int64)
    ends = starts + np.array([temps[a] for a in starts], dtype=np.int64)
    seen = set()
    for r in prog.code:
        for k in range(len(r["tag"])):
            if r["tag"][k] != opcodes.T_INT:
                continue
            v = int(r["v"][k])
            j = int(np.searchsorted(starts, v, side="right")) - 1
            if j >= 0 and v < ends[j]:
                seen.add(int(starts[j]))
    return sum(temps[a] for a in seen)


def compiled(fn, pack):
    alloc.PACK = pack
    try:
        return fn()
    finally:
        alloc.PACK = True


def main():
    path = sys.argv[1]
    sizes = [int(x) for x in sys.argv[2:]] or [4, 16, 1024, 2048]
    g = GGUF(path)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    moe = cfg.enable_moe_block
    ops = collections.Counter()
    step = compiled(lambda: gpu.compile_split_step(model, {}, "int16", pooled=moe), True)
    hot_host = step.hot_stores if moe else {}
    builds = [("step", lambda: gpu.compile_split_step(model, {}, "int16", pooled=moe))]
    for t in sizes:
        builds.append(("group %d" % t, lambda t=t: gpu.compile_split_group(model, t, kv="int16",
                                                                           hot_host=hot_host)))
    for name, fn in builds:
        plain, packed = compiled(fn, False), compiled(fn, True)
        for r in packed.code:
            ops[int(r["op"])] += 1
        print("%-11s %5d records (%5.0f KB), %4d slots; temporaries %8.1f MB, packed %7.1f MB (%s)" % (
            name, len(packed.code), len(packed.code) * 224 / 1e3, len(packed.env), used_bytes(plain) / 1e6,
            used_bytes(packed) / 1e6, getattr(packed, "pack_info", None)), flush=True)
    print("opcodes: " + ", ".join("%s %d" % (NAMES.get(k, k), v) for k, v in sorted(ops.items())))


if __name__ == "__main__":
    main()
