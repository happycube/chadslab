#!/usr/bin/env python3
"""Check the Q8_0 GGUF of scripts/convert_q8_gguf.py against ORIG (the
original bfloat16 checkpoint) and the NVFP4 GGUF.

RQ8_EXPERTS_PLAN.md, section 6. For a sample of each part, g.dequant of the
file against ORIG:

- the routed experts of some layers and of the MTP layer (gate, up from the
  fused gate_up_proj, down): Q8_0, about -45 dB;
- the n-gram table (rows of some shards): about -45.5 dB;
- the dense matrices (bfloat16): the same bits as the NVFP4 GGUF (both are
  the bfloat16 of ORIG; the DeltaNet value heads in the tiled order there);
  BF12 (--dense bf12): those bits but the zeroed values.

    python scripts/check_q8_gguf.py OUT.gguf /space/models/Qwen3.8-Flash-Next \\
        [--nvfp4 models2/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-bf16.gguf]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from convert_q8_gguf import OrigSource, bf16f, db  # noqa: E402
from np_gemma.cops import bf12_row_bytes, kq_bf12_to_bf16  # noqa: E402
from np_gemma.gguf import BF12, BF16, GGUF, RQ6_K, RQ8_0  # noqa: E402

NVFP4 = "models2/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-bf16.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="the Q8_0 GGUF")
    ap.add_argument("src", help="the ORIG checkpoint directory")
    ap.add_argument("--nvfp4", default=NVFP4, help="the NVFP4 GGUF ('' to skip)")
    ap.add_argument("--min-db", type=float, default=-44.0, help="the worst error that passes")
    ap.add_argument("--min-db-rq6", type=float, default=-34.0,
                    help="the worst error of an RQ6_K matrix that passes")
    args = ap.parse_args()
    g = GGUF(args.out)
    xt = {(int(n.split(".")[1]), n.split("_")[1]): t for n, (_d, t, _o) in g.tensors.items()
          if "_exps." in n}
    src = OrigSource(args.src, rot=any(t in (RQ8_0, RQ6_K) for _d, t, _o in g.tensors.values()),
                     xtypes=xt)
    good = True
    names = set(g.tensors)
    missing = [n for n in src.tensors if n not in names]
    if g.meta.get("np_gemma.experts_only"):
        missing = [n for n in missing if "_exps." in n and int(n.split(".")[1]) < src.L]
    if missing:
        print("the file lacks %d tensors of the map, e.g. %s" % (len(missing), missing[:3]))
        good = len(missing) == 0 and good
    # the experts
    for i in sorted({0, 13, 26, src.L - 1, src.L}):
        for x in ("gate", "up", "down"):
            n = "blk.%d.ffn_%s_exps.weight" % (i, x)
            if n not in names:
                continue
            hf, lo, hi, _t = src._map[n][1]
            w = src._bf16(hf)
            v = []
            for e in (1, 255, 510):
                ref = bf16f(w[e, lo:hi])
                got = g.dequant(n, rows=[e])[0]
                v.append(db(ref, got))
            ok = max(v) < (args.min_db_rq6 if g.tensors[n][1] == RQ6_K else args.min_db)
            good &= ok
            print("%-28s %-5s experts 1, 255, 510: %s dB %s" % (
                n, {RQ6_K: "RQ6_K", RQ8_0: "RQ8_0"}.get(g.tensors[n][1], "Q8_0"),
                " ".join("%.2f" % a for a in v), "" if ok else "FAIL"))
    # the n-gram table
    n = "per_layer_token_embd.weight"
    if n in names:
        starts = src._ple_start
        for k in (0, 64, len(src._ple_shards) - 1):
            w = src._bf16(src._ple_shards[k])
            r0 = int(starts[k]) + w.shape[0] // 3
            ref = bf16f(w[w.shape[0] // 3:w.shape[0] // 3 + 5000])
            got = g.dequant(n, rows=np.arange(r0, r0 + 5000))
            d = db(ref, got)
            ok = d < args.min_db
            good &= ok
            print("%-28s shard %3d, 5000 rows: %.2f dB %s" % (n, k, d, "" if ok else "FAIL"))
    # the shared experts in RQ8_0 (--experts rq8): against ORIG
    for i in sorted({0, 26, src.L}):
        for x in ("gate", "up", "down"):
            n = "blk.%d.ffn_%s_shexp.weight" % (i, x)
            if n not in names or g.tensors[n][1] != RQ8_0:
                continue
            ref = bf16f(np.asarray(src._bf16(src._map[n][1])))
            d = db(ref, g.dequant(n))
            ok = d < args.min_db
            good &= ok
            print("%-28s %.2f dB %s" % (n, d, "" if ok else "FAIL"))
    # the dense matrices against the NVFP4 GGUF (the same bfloat16 of ORIG)
    if args.nvfp4 and os.path.exists(args.nvfp4):
        h = GGUF(args.nvfp4)
        for n in ("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
                  "blk.3.attn_q.weight", "blk.0.ssm_a", "blk.0.ssm_conv1d.weight",
                  "blk.48.nextn.eh_proj.weight", "output.weight", "token_embd.weight",
                  "blk.10.ffn_gate_shexp.weight", "blk.1.ple_key.weight"):
            if n not in names or n not in h.tensors or g.tensors[n][1] == RQ8_0:
                continue
            a, _d, ta = g.raw(n)
            b, _d2, tb = h.raw(n)
            if ta == BF12 and tb == BF16:
                # --dense bf12: the bits of the NVFP4 GGUF but the zeroed
                # values (each below 2^-8 of the RMS of its matrix)
                cols = g.tensors[n][0][0]
                rows = a.nbytes // bf12_row_bytes(cols)
                got = kq_bf12_to_bf16(np.asarray(a).view(np.uint8), rows, cols).reshape(-1)
                ref = np.asarray(b).view(np.uint16).reshape(-1)
                diff = got != ref
                rf = bf16f(ref)
                rms = float(np.sqrt(np.mean(rf.astype(np.float64) ** 2)))
                same = bool(np.all(got[diff] == 0) and np.all(np.abs(rf[diff]) <= rms / 256))
                good &= same
                print("%-28s BF12: the bits of the NVFP4 GGUF but %d zeroed values (%.4f%%): %s" % (
                    n, int(diff.sum()), 100 * diff.mean(), same))
                continue
            same = ta == tb and a.nbytes == b.nbytes and np.array_equal(
                np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8))
            good &= same
            print("%-28s the bits of the NVFP4 GGUF: %s" % (n, same))
    print("RESULT", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
