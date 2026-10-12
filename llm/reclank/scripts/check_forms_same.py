#!/usr/bin/env python3
"""Dump the records of the programs of the Gemma 4 models in a canonical
form, to compare the programs of two versions of the code (a change of the
compiler or of the forms that must not change a record):

    python scripts/check_forms_same.py OUT MODEL...     (MODEL: g12 g26 e4b)
    diff OUT_a OUT_b

An address in an array of the program becomes the index of that array in
the order the records first use it, its offset, and a fingerprint of its
bytes (shape, type, and samples: the weights must be the same matrices);
a slot becomes its name; a float its bits (an int64 table and the scratch
of KQ_MOE keep only their shape: they hold addresses and garbage that
differ between runs). Packing is off (RECLANK_PACK=0).
The programs: the CPU step of 1 and 4 tokens, the prompt block of 64, the
parts step (dense models), and the GPU step and groups of 16 and 1024
(compiled on the host: no GPU run).
"""
from __future__ import annotations

import hashlib
import os
import sys

os.environ["RECLANK_PACK"] = "0"
os.environ.setdefault("RECLANK_GPU", "1")
os.environ.setdefault("RECLANK_DAX_STAGE", "0")
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from reclank import Model  # noqa: E402
from reclank.formats.gguf import GGUF  # noqa: E402
from reclank.ir import program as P  # noqa: E402
from reclank.models.gemma4.config import Config  # noqa: E402

PATHS = {"g12": "/home/cpage/llm/gemma-4-12B-unsloth-UD-Q4_K_XL/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
         "g26": "/space/models/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
         "e4b": "/space/models/gemma-4-E4B-unsloth-UD-Q4_K_XL/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"}


def fingerprint(a, content=True):
    # the count of values, not the shape: the array that owns a large
    # temporary is the flat map of program.lazy_zeros
    if not content:
        return "%d/%s" % (a.size, a.dtype.str)
    b = a.view(np.uint8).reshape(-1)
    idx = np.linspace(0, b.size - 1, num=min(b.size, 4096)).astype(np.int64)
    h = hashlib.sha1(b[idx].tobytes()).hexdigest()[:10]
    return "%d/%s/%s" % (a.size, a.dtype.str, h)


def canonical(prog):
    """The lines of a program: one for each record."""
    arrays = {}
    for a in prog.keep:
        if isinstance(a, np.ndarray) and a.nbytes:
            while isinstance(a.base, np.ndarray):
                a = a.base
            arrays[a.ctypes.data] = a
    starts = np.array(sorted(arrays), dtype=np.int64)
    ends = starts + np.array([arrays[s].nbytes for s in starts], dtype=np.int64)
    ids, names = {}, {s.index: n for n, s in prog.by_name.items()}
    lines = []
    for r in prog.code:
        ops = []
        for k in range(len(r["tag"])):
            tag, v = int(r["tag"][k]), int(r["v"][k])
            if tag == P.T_NONE:
                continue
            if tag == P.T_SLOT:
                ops.append("$" + names.get(v, "#%d" % v))
            elif tag == P.T_F32:
                ops.append("f%08x" % (v & 0xffffffff))
            else:
                j = int(np.searchsorted(starts, v, side="right")) - 1
                if j >= 0 and v < ends[j]:
                    s = int(starts[j])
                    if s not in ids:
                        # no bytes for what differs between two runs: the
                        # int64 tables (CPU programs, pointers, ranges) and
                        # the scratch of KQ_MOE (np.empty)
                        a = arrays[s]
                        content = a.dtype != np.int64 and not (int(r["op"]) == P.KQ_MOE and k == 12)
                        ids[s] = "A%d:%s" % (len(ids), fingerprint(a, content))
                    ops.append("%s+%d" % (ids[s], v - s))
                else:
                    ops.append(str(v))
        lines.append("%d %s" % (int(r["op"]), " ".join(ops)))
    return lines


def main():
    out = open(sys.argv[1], "w")
    from reclank.models.gemma4 import forms, gpu, prompt
    for name in sys.argv[2:]:
        g = GGUF(PATHS[name])
        tc = g.text_config()
        progs = []
        if name == "e4b":
            from reclank.models.gemma4.e4b import E4B, E4BConfig
            model = E4B(g, E4BConfig({"text_config": tc}), mode="int4")
            progs += [("e4b step", lambda: forms.compile_e4b_step(model, 1)),
                      ("e4b step fused", lambda: forms.compile_e4b_step(model, 1, fused=True)),
                      ("e4b group 16", lambda: gpu.compile_e4b_group(model, 16))]
        else:
            model = Model(g, Config({"text_config": tc})).load_all(dtype="int4")
            moe = model.cfg.enable_moe_block
            progs += [("cpu step 1", lambda: forms.compile_step(model, "qc", 1)),
                      ("cpu step 4", lambda: forms.compile_step(model, "qc", 4)),
                      ("prompt 64", lambda: prompt.compile_prompt(model, "qc", 64))]
            if not moe:
                from reclank.models.gemma4 import parts
                progs.append(("parts 2", lambda: parts.compile_parts(model, "qc", 2)))
            step = gpu.compile_split_step(model, {}, "int16", pooled=moe)
            hot_host = step.hot_stores if moe else {}
            progs += [("gpu step", lambda: gpu.compile_split_step(model, {}, "int16", pooled=moe)),
                      ("gpu group 16", lambda: gpu.compile_split_group(model, 16, kv="int16",
                                                                        hot_host=hot_host)),
                      ("gpu group 1024", lambda: gpu.compile_split_group(model, 1024, kv="int16",
                                                                          hot_host=hot_host))]
        for label, fn in progs:
            prog = fn()
            if not hasattr(prog, "code"):          # (compile_parts gives a PartsProgram)
                prog = prog.progs[0]
            lines = canonical(prog)
            out.write("== %s %s: %d records\n" % (name, label, len(lines)))
            out.write("\n".join(lines) + "\n")
            print("%s %s: %d records" % (name, label, len(lines)), flush=True)
    out.close()


if __name__ == "__main__":
    main()
