#!/usr/bin/env python3
"""Requantize the dense matrices of a Qwen3.6 GGUF (llama.cpp, e.g. the
UD-Q4_K_M of Unsloth) from the original bfloat16 checkpoint
(Qwen/Qwen3.6-35B-A3B, "ORIG"), each in a type of its own
(QWEN_DENSE_QUANT.md): Q4_K, Q5_K, Q6_K, Q8_0, their forms of the rotated
rows (RQ4_K, RQ5_K, RQ6_K, RQ8_0: the blocks of the rows after the TQ6
rotation of each 32 values; the runtime rotates the x of their products the
same way, KMat.rot, compile_qwen_step), or BF12 (the bfloat16 values in
12.25 bits, BF12_PLAN.md; a matrix that BF12 cannot hold, a zeroed value
above 2^-8 of its RMS or an Inf or NaN, stays bfloat16).

--type: the type of every dense matrix (default RQ6_K); --plan FILE: a JSON
{tensor name: type} of scripts/qwen_dense_plan.py, which takes precedence;
--bf12 REGEX: the matrices whose names match it in BF12.

The other tensors of the template stay as they are: the experts, the
shared experts (the MoE kernels of Qwen3.6 do not rotate), the router, the
token table, the head, the norms. A matrix whose rows are not whole blocks
of its type (256 values for the K types, 32 for Q8_0 and BF12) stays too.

The value heads of the DeltaNet: llama.cpp stores them in the tiled order
(value head j = r kh + k reads key head k), ORIG in the grouped order
(value head k rep + r, rep = vh / kh): the rows of attn_gate and of the
value part of attn_qkv, and the columns of ssm_out, take that order
(v_order).

Each matrix of ORIG is checked against the template (its values from the
blocks of the file): the same tensor in the same order gives the error of
the quantization of the template (about -45 dB for Q8_0); worse than
--max-db stops the conversion (another name, or rows in another order).
reclank.quant_error.dense: the mean error in dB of the requantized
matrices against ORIG, and of the template's.

    python scripts/convert_q36_dense.py ORIG Qwen3.6-35B-A3B-UD-Q4_K_M.gguf OUT.gguf --plan plan.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.gguf import BF12, GGUF, ROT_BASE, Q8_0, rotation_meta, write_gguf  # noqa: E402

# GGUF name (blk.N.<name>.weight) -> the module of ORIG (model.language_model.layers.N.<module>.weight)
DENSE = {
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_gate": "linear_attn.in_proj_z",
    "ssm_out": "linear_attn.out_proj",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
}
BF16 = 30
# the types of the plan: name -> (GGUF type, the block of the cols)
TYPES = {"Q4_K": (12, 256), "Q5_K": (13, 256), "Q6_K": (14, 256), "Q8_0": (8, 32),
         "RQ4_K": (58, 256), "RQ5_K": (59, 256), "RQ6_K": (56, 256), "RQ8_0": (55, 32),
         "BF12": (BF12, 32)}


def quantize(tname, wf):
    """The blocks of the float32 rows wf in type tname (not BF12), and their
    values back."""
    t = TYPES[tname][0]
    base = ROT_BASE.get(t, t)
    x = cops.tq6_rotate(wf) if t in ROT_BASE else wf
    rows, cols = wf.shape
    q = {12: lambda: cops.kq_to_qk(x, cols, 4), 13: lambda: cops.kq_to_qk(x, cols, 5),
         14: lambda: cops.kq_to_q6_k(x, cols), 8: lambda: cops.kq_to_q8_0(x, cols)}[base]()
    v = cops.kq_rows(q, base, cols, np.arange(rows))
    return q, (cops.tq6_rotate(v, inverse=True) if t in ROT_BASE else v)


class SafeTensors:
    """The tensors of the shards of a checkpoint (model.safetensors.index.json)
    as memory maps; bfloat16 as uint16."""

    def __init__(self, path, index=None):
        self.path = path
        wm = json.load(open(index or os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.where = wm
        self._hdr = {}

    def _header(self, f):
        h = self._hdr.get(f)
        if h is None:
            with open(os.path.join(self.path, f), "rb") as fh:
                n = int(np.frombuffer(fh.read(8), np.uint64)[0])
                h = (json.loads(fh.read(n)), 8 + n)
            self._hdr[f] = h
        return h

    def get(self, name):
        f = self.where[name]
        hdr, base = self._header(f)
        e = hdr[name]
        a, b = e["data_offsets"]
        dt = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}[e["dtype"]]
        mm = np.memmap(os.path.join(self.path, f), dtype=np.uint8, mode="r", offset=base + a,
                       shape=(b - a,))
        return mm.view(dt).reshape(e["shape"]), e["dtype"]


def v_order(cfg):
    """The ORIG value head of each value head of the GGUF (the tiled order)."""
    tc = cfg.get("text_config", cfg)
    vh, kh = tc["linear_num_value_heads"], tc["linear_num_key_heads"]
    rep = vh // kh
    return np.array([(j % kh) * rep + j // kh for j in range(vh)]), tc["linear_value_head_dim"]


def to_gguf_order(kind, w, cfg):
    """The matrix w (rows, cols) of ORIG in the order of the GGUF."""
    perm, vd = v_order(cfg)
    vh = len(perm)
    if kind == "attn_gate":
        return w.reshape(vh, vd, -1)[perm].reshape(w.shape)
    if kind == "attn_qkv":
        v0 = w.shape[0] - vh * vd
        return np.concatenate([w[:v0], w[v0:].reshape(vh, vd, -1)[perm].reshape(vh * vd, -1)])
    if kind == "ssm_out":
        return w.reshape(w.shape[0], vh, vd)[:, perm].reshape(w.shape)
    return w


def bf16f(a):
    return (np.asarray(a, np.uint16).astype(np.uint32) << 16).view(np.float32)


def db(ref, x):
    """The error of x against ref in dB (the energy of the difference over
    that of ref)."""
    d = np.asarray(x, np.float64) - ref
    return 10 * np.log10(max((d * d).sum() / max((ref * ref).sum(), 1e-30), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orig", help="the ORIG checkpoint directory")
    ap.add_argument("template", help="the GGUF whose other tensors and metadata stay")
    ap.add_argument("out", help="the GGUF file to write")
    ap.add_argument("--type", default="RQ6_K",
                    help="the type of the dense matrices (%s; rq6, rq8 as RQ6_K, RQ8_0)" % ", ".join(TYPES))
    ap.add_argument("--plan", default=None, help="a JSON {tensor name: type} (qwen_dense_plan.py)")
    ap.add_argument("--bf12", default=None, metavar="REGEX",
                    help="the matrices whose names match it in BF12 (or bfloat16 when BF12 "
                         "cannot hold them)")
    ap.add_argument("--max-db", type=float, default=-35.0,
                    help="the worst error of ORIG against the template (default -35 dB)")
    ap.add_argument("--only", default=None,
                    help="a test: requantize only the tensors whose names match this regex")
    ap.add_argument("--index", default=None,
                    help="the model.safetensors.index.json (default: in ORIG)")
    ap.add_argument("--check-only", action="store_true",
                    help="measure the errors of the matrices of --only, write nothing")
    args = ap.parse_args()
    deflt = {"rq6": "RQ6_K", "rq8": "RQ8_0", "rq5": "RQ5_K", "rq4": "RQ4_K"}.get(args.type, args.type.upper())
    assert deflt in TYPES, args.type
    plan_of = json.load(open(args.plan)) if args.plan else {}
    assert all(v in TYPES for v in plan_of.values()), set(plan_of.values()) - set(TYPES)
    st = SafeTensors(args.orig, args.index)
    cfg = json.load(open(os.path.join(args.orig, "config.json")))
    g = GGUF(args.template, stage=False)

    # (the reader gives arrays as NumPy arrays, a large string array as its
    # place in the file: the writer takes lists)
    def value(k, v):
        if isinstance(v, dict) and "__array__" in v:
            return g.meta_strings(k)
        return v.tolist() if isinstance(v, np.ndarray) else v
    meta = [(k, value(k, v)) for k, v in g.meta.items() if k != "general.alignment"]
    meta += list(rotation_meta().items())
    errs = {"new": [], "template": []}
    plan = []           # (name, dims, type name or None: as the template, ORIG name)
    for name, (dims, t, _o) in g.tensors.items():
        m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
        cols = int(dims[0])
        if m and m.group(2) in DENSE and t == Q8_0 and (args.only is None or re.search(args.only, name)):
            tn = "BF12" if args.bf12 and re.search(args.bf12, name) else plan_of.get(name, deflt)
            if cols % TYPES[tn][1]:
                print("  %-28s cols %d are not blocks of %s: as the template" % (name, cols, tn))
                plan.append((name, dims, None, None))
                continue
            hf = "model.language_model.layers.%s.%s.weight" % (m.group(1), DENSE[m.group(2)])
            plan.append((name, dims, tn, hf))
        else:
            plan.append((name, dims, None, None))
    counts = {}
    for p in plan:
        if p[2]:
            counts[p[2]] = counts.get(p[2], 0) + 1
    print("%d tensors; dense: %s" % (len(plan), ", ".join("%d %s" % (n, t) for t, n in sorted(counts.items()))),
          flush=True)
    bf16_kept = set()

    def make(name, dims, tn, hf):
        if hf is None:
            blocks, _d, _t = g.raw(name)
            return np.ascontiguousarray(blocks).view(np.uint8).reshape(-1)
        w, dt = st.get(hf)
        cols, rows = int(dims[0]), int(dims[1])
        assert dt == "BF16" and tuple(w.shape) == (rows, cols), (hf, w.shape, dims, dt)
        wf = np.ascontiguousarray(to_gguf_order(name.split(".")[2], bf16f(w), cfg))
        tq = np.ascontiguousarray(g.raw(name)[0]).view(np.uint8).reshape(-1)
        e_t = db(wf, cops.kq_rows(tq, Q8_0, cols, np.arange(rows)))
        if e_t > args.max_db:
            raise SystemExit("%s: ORIG %s is %.1f dB from the template: another tensor or "
                             "another order of its rows" % (name, hf, e_t))
        if tn == "BF12":
            u16 = np.ascontiguousarray(to_gguf_order(name.split(".")[2], np.asarray(w), cfg))
            bq, rep_ = cops.kq_bf16_to_bf12(u16, cols)
            if rep_["nonfinite"] > 0 or rep_["largest"] > rep_["rms"] * 2.0 ** -8:
                print("  %-28s BF12 cannot hold it: bfloat16" % name, flush=True)
                bf16_kept.add(name)
                return u16.view(np.uint8).reshape(-1)
            print("  %-28s %5d x %-5d Q8_0 %.1f dB -> BF12 (%d zeroed values)" % (
                name, rows, cols, e_t, rep_["zeroed"]), flush=True)
            return bq.reshape(-1)
        q, back = quantize(tn, wf)
        e_q = db(wf, back)
        errs["new"].append(e_q)
        errs["template"].append(e_t)
        print("  %-28s %5d x %-5d Q8_0 %.1f dB -> %s %.1f dB" % (name, rows, cols, e_t, tn, e_q),
              flush=True)
        return q

    t0 = time.time()
    # the errors go into the metadata, so they are measured before the write
    # (the dense matrices are 2 GB of the file)
    made = {}
    for name, dims, tn, hf in plan:
        if hf is not None:
            made[name] = make(name, dims, tn, hf)
    if errs["new"]:
        meta.append(("reclank.quant_error.dense",
                     "%s: %.2f dB, the template Q8_0 %.2f dB (mean of %d matrices against ORIG)" % (
                         ", ".join(sorted(counts)), float(np.mean(errs["new"])),
                         float(np.mean(errs["template"])), len(errs["new"]))))
        print("mean: %.2f dB, Q8_0 %.2f dB" % (np.mean(errs["new"]), np.mean(errs["template"])), flush=True)
    if args.check_only:
        return

    def gtype(name, tn, t0_):
        if tn is None:
            return t0_
        return BF16 if name in bf16_kept else TYPES[tn][0]
    tensors = [(name, dims, gtype(name, tn, g.tensors[name][1]),
                (lambda n=name: made[n]) if hf is not None
                else (lambda n=name, d=dims: make(n, d, None, None)))
               for name, dims, tn, hf in plan]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_gguf(args.out, meta, tensors)
    print("wrote %s (%.1f GB) in %.0f s" % (args.out, os.path.getsize(args.out) / 1e9,
                                             time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
