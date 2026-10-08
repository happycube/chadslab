#!/usr/bin/env python3
"""Check every tensor of a --dense bf12 GGUF of scripts/convert_q8_gguf.py.

BF12_PLAN.md, step 4 (a). For each tensor of the file:

- a BF12 matrix: the decoded bits (cops.kq_bf12_to_bf16) against the
  bfloat16 of ORIG in the order of the file (the DeltaNet value heads
  tiled): equal but the zeroed values, each 0 in the file and at most 2^-8
  of the RMS of its matrix in ORIG; the count of those against the report
  of the converter (<out>.bf12.json);
- a routed expert matrix of --experts-from: the bytes of that file;
- a matrix that the rule kept bfloat16 (np_gemma.bf12.kept_bf16): the bits
  of ORIG.

The file is read in place (NP_GEMMA_DAX_STAGE=0: no copy into memory).

    python scripts/check_bf12.py OUT.gguf /space/models/Qwen3.8-Flash-Next \\
        [--experts-from EXPERTS.gguf]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("NP_GEMMA_DAX_STAGE", "0")
os.environ.setdefault("NP_GEMMA_QUIET", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from convert_q8_gguf import OrigSource, reorder  # noqa: E402
from np_gemma.cops import bf12_row_bytes, kq_bf12_to_bf16  # noqa: E402
from np_gemma.gguf import BF12, BF16, GGUF, RQ6_K, RQ8_0  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="the GGUF of --dense bf12")
    ap.add_argument("src", help="the ORIG checkpoint directory")
    ap.add_argument("--experts-from", default=None, help="the GGUF of the routed experts")
    ap.add_argument("--only", default=None, help="the tensors whose names hold this text")
    args = ap.parse_args()
    t0 = time.time()
    g = GGUF(args.out)
    xt = {(int(n.split(".")[1]), n.split("_")[1]): t for n, (_d, t, _o) in g.tensors.items()
          if "_exps." in n}
    src = OrigSource(args.src, rot=any(t in (RQ8_0, RQ6_K) for t in xt.values()), xtypes=xt)
    cfg = src.config()
    hk, hv, dk, dv = cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim
    rep, qk = hv // hk, 2 * hk * dk
    report = {}
    if os.path.exists(args.out + ".bf12.json"):
        report = json.load(open(args.out + ".bf12.json"))
    kept = [n for n in str(g.meta.get("np_gemma.bf12.kept_bf16", "")).split(",") if n]

    def orig_bits(gname):
        """The bfloat16 bits of ORIG in the order of the file (as bf16_of of
        the converter)."""
        _kind, s = src._map[gname]
        if gname.endswith(("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight")):
            a = np.asarray(src._bf16(s))
            if gname.endswith("attn_qkv.weight"):
                a = np.concatenate([a[:qk], reorder(a[qk:], 0, hk, rep, dv)])
            elif gname.endswith("attn_gate.weight"):
                a = reorder(a, 0, hk, rep, dv)
            else:
                a = reorder(a, 1, hk, rep, dv)
        else:
            a = np.asarray(src._make(gname))
        return np.ascontiguousarray(a).view(np.uint16)

    xf = GGUF(args.experts_from) if args.experts_from else None
    good = True
    n_bf12 = n_x = n_kept = zeroed = 0
    for name in g.tensors:
        if args.only and args.only not in name:
            continue
        dims, t, _o = g.tensors[name]
        if t == BF12:
            cols = dims[0]
            raw = np.asarray(g.raw(name)[0]).view(np.uint8).reshape(-1)
            rows = raw.size // bf12_row_bytes(cols)
            got = kq_bf12_to_bf16(raw, rows, cols).reshape(-1)
            ref = orig_bits(name).reshape(-1)
            diff = got != ref
            rf = (ref[diff].astype(np.uint32) << 16).view(np.float32)
            rr = (ref.astype(np.uint32) << 16).view(np.float32)
            rms = float(np.sqrt(np.mean(np.square(rr, dtype=np.float64))))
            nz = int(diff.sum())
            ok = bool(np.all(got[diff] == 0) and np.all(np.abs(rf) <= rms / 256))
            want = report.get(name, {}).get("zeroed")
            if want is not None:
                ok &= nz == int(want) + int(report[name].get("collisions", 0)) or nz == int(want)
            zeroed += nz
            n_bf12 += 1
            if not ok or n_bf12 % 50 == 1:
                print("%-36s BF12 %6d x %5d: %5d zeroed (report %s), the largest %.3g of the RMS %s" % (
                    name, rows, cols, nz, want, float(np.abs(rf).max() / rms) if nz else 0.0,
                    "" if ok else "FAIL"), flush=True)
            good &= ok
        elif xf is not None and "_exps." in name and name in xf.tensors:
            a = np.asarray(g.raw(name)[0]).view(np.uint8).reshape(-1)
            b = np.asarray(xf.raw(name)[0]).view(np.uint8).reshape(-1)
            ok = a.size == b.size and xf.tensors[name][1] == t
            for o in range(0, a.size if ok else 0, 1 << 28):
                ok &= np.array_equal(a[o:o + (1 << 28)], b[o:o + (1 << 28)])
            n_x += 1
            if not ok:
                print("%-36s the bytes of --experts-from: FAIL" % name, flush=True)
            good &= ok
        elif name in kept and t == BF16:
            ok = np.array_equal(np.asarray(g.raw(name)[0]).view(np.uint16).reshape(-1),
                                orig_bits(name).reshape(-1))
            n_kept += 1
            print("%-36s kept bfloat16: the bits of ORIG %s" % (name, ok), flush=True)
            good &= ok
    print("%d BF12 matrices (%d zeroed values), %d kept bfloat16, %d expert matrices of "
          "--experts-from; %.0f s" % (n_bf12, zeroed, n_kept, n_x, time.time() - t0))
    print("RESULT", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
