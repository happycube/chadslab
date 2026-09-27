#!/usr/bin/env python3
"""Write the programs of Qwen3.6 (GGUF) as S-expressions, for reading.

Each record becomes (op operand ...). An operand that points into an array
gets the name of the array. The names are:

- a weight of the file (w:blk.0.attn_qkv), or its copy on the GPU (gpu:...);
- a buffer of the program with its shape (%3:f32[1x8192]);
- a pinned buffer, or a CPU program.

A slot is $name, and a float is 1e-06f. A comment marks each layer.
sexp/README.md explains the notation and the operations.

    python scripts/export_sexp.py --out sexp
"""
from __future__ import annotations

import argparse
import bisect
import os
import sys

import numpy as np

os.environ.setdefault("NP_GEMMA_GPU_HOT_DYN", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import program as P  # noqa: E402
from np_gemma.qwen import QwenGGUFProgram, gguf_name  # noqa: E402

PATH = "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


class Names:
    """The name of the array that holds an address."""

    def __init__(self):
        self.starts, self.items = [], {}
        self.anon = 0

    def add(self, a, name):
        while isinstance(a, np.ndarray) and isinstance(a.base, np.ndarray) and \
                a.base.ctypes.data == a.ctypes.data and a.base.nbytes == a.nbytes:
            a = a.base
        start = a.ctypes.data
        if start in self.items or a.nbytes == 0:
            return
        bisect.insort(self.starts, start)
        self.items[start] = (start + a.nbytes, name)

    def name(self, addr):
        k = bisect.bisect_right(self.starts, addr) - 1
        if k < 0:
            return None
        start = self.starts[k]
        end, name = self.items[start]
        if addr >= end:
            return None
        return name if addr == start else "%s+%d" % (name, addr - start)


def shape_name(n, a):
    return "%%%d:%s[%s]" % (n, {"float32": "f32", "int32": "i32", "int8": "i8", "int16": "i16",
                                "uint8": "u8", "int64": "i64"}.get(a.dtype.name, a.dtype.name),
                            "x".join(str(d) for d in a.shape))


def export(prog, title, names, extra_arrays=()):
    """The S-expression text of a program."""
    for a in list(prog.keep) + list(extra_arrays):
        if isinstance(a, np.ndarray) and names.name(a.ctypes.data) is None:
            names.add(a, shape_name(names.anon, a))
            names.anon += 1
    slot_names = {s.index: s.name for s in prog.slots}
    out = [";; %s: %d records, %d slots" % (title, len(prog.recs), len(prog.slots)),
           "(program %s" % title.split()[0]]
    out.append("  (env " + " ".join("$%s" % s.name for s in prog.slots) + ")")
    layer = None
    for op, args in prog.recs:
        ops = []
        for tag, v in args:
            if tag == P.T_SLOT:
                ops.append("$" + slot_names[v])
            elif tag == P.T_F32:
                ops.append("%gf" % P._bits_f32(v))
            elif v > 1 << 20:
                nm = names.name(v)
                ops.append(nm if nm else "#x%x" % v)
            else:
                ops.append(str(v))
        # the layer of the record: the first weight name with blk.N
        for o in ops:
            if "blk." in o:
                n = int(o.split("blk.")[1].split(".")[0])
                if n != layer:
                    layer = n
                    out.append("")
                    out.append("  ;; ---- layer %d ----" % n)
                break
        name = P.OP_NAMES.get(op, str(op)).lower().replace("_", "-")
        out.append("  (%s %s)" % (name, " ".join(ops)))
    out.append(")")
    return "\n".join(out) + "\n"


def model_names(model, names):
    for name, m in model._m.items():
        names.add(m.data, "w:" + gguf_name(name).replace(".weight", ""))
    for name, a in model._f.items():
        names.add(a, "w:" + gguf_name(name).replace(".weight", "") if "A_log" not in name
                  else "w:blk.%s.A_log" % name.split(".")[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="sexp")
    ap.add_argument("--gpu", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m = QwenGGUFProgram(PATH)

    # The CPU step of one token.
    prog = m.program(1)
    names = Names()
    model_names(m, names)
    names.add(prog.names["x"], "x")
    names.add(prog.names["xn"], "xn")
    open(os.path.join(args.out, "cpu_step.sexp"), "w").write(
        export(prog, "qwen-cpu-step (the CPU program of one token)", names))

    from np_gemma.qwen_gpu import QwenGPU
    g = QwenGPU(m, hot_gb=0.3)
    names = Names()
    model_names(m, names)
    for name, (a, _t) in g._dense.items():
        names.add(a, "gpu:" + gguf_name(name).replace(".weight", ""))
    for i, st in g.stores.items():
        for k in ("gate", "up", "down"):
            names.add(st[k], "gpu:blk.%d.hot_%s" % (i, k))
        names.add(st["slots"], "gpu:blk.%d.slots" % i)
    for n, cpu in enumerate(g.cpu_progs):
        names.add(cpu.buf, "#<cpu-program %d>" % n)
    names.add(g.prog.names["x"], "x")
    names.add(g.prog.names["xn"], "xn")
    open(os.path.join(args.out, "gpu_step.sexp"), "w").write(
        export(g.prog, "qwen-gpu-step (the GPU program of one token; cold experts on the CPU)",
               names))
    # The CPU program of the cold experts of layer 0 of the step.
    open(os.path.join(args.out, "cpu_cold_experts.sexp"), "w").write(
        export(g.cpu_progs[0], "qwen-cpu-cold-experts (layer 0; run by gp-cpu-join)", names))
    # An MTP verify group of 4 tokens, and a large group of the prompt.
    for t, verify, fetch, fname, title in (
            (4, True, False, "gpu_verify4.sexp", "qwen-gpu-verify (4 tokens, MTP verify)"),
            (1024, False, True, "gpu_prompt1024.sexp",
             "qwen-gpu-prompt (1024 rows, the experts copied to the GPU)")):
        n0 = len(g.cpu_progs)
        prog, _gp = g._group(t, verify, fetch)
        for n, cpu in enumerate(g.cpu_progs[n0:]):
            names.add(cpu.buf, "#<cpu-program %d>" % (n0 + n))
        for i, tabs in g.tables.items():
            for k, a in zip(("gate", "up", "down", "fetch_ranges"), tabs):
                names.add(a, "gpu:blk.%d.table_%s" % (i, k))
        names.add(prog.names["x"], "x%d" % t)
        names.add(prog.names["xn"], "xn%d" % t)
        open(os.path.join(args.out, fname), "w").write(export(prog, title, names))
    g.close()
    for f in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, f)
        print("%-28s %6d lines" % (p, sum(1 for _ in open(p))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
