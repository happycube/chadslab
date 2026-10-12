#!/usr/bin/env python3
"""Can the dense matrices of DeepSeek V4.1 Flash be BF12 (DEEPSEEK_V41_PLAN.md 3)?

For each bfloat16 matrix of the checkpoint (two dims): the report of the BF12
converter (cops.kq_bf16_to_bf12: a value more than 15 binades under the
largest of its group of 32 decodes to 0) and the rule of the runtime
(scripts/convert_q8_gguf.py --dense bf12): cols a multiple of 32, more than
one row, no Inf or NaN, the largest zeroed value at most 2^-8 of the RMS of
its matrix; else it stays bfloat16. --fp8: the FP8 matrices (E4M3, an E8M0
scale for each 32 x 32 block) too, as their exact bfloat16 values, to see
whether BF12 could hold them (the plan keeps them FP8).

    python scripts/check_dsv41_bf12.py /home/cpage/models/DeepSeek-V4.1-Flash [--fp8] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.nvfp4_st import E4M3  # noqa: E402
from reclank.runtime.zygote import Shards  # noqa: E402


def read(sh, name):
    f, dt, shape, o0, o1 = sh.t[name]
    with open(sh.file(f), "rb") as fh:
        fh.seek(o0)
        return np.frombuffer(fh.read(o1 - o0), np.uint8), dt, shape


def fp8_bits(sh, name):
    """The exact bfloat16 bits of an FP8 matrix with its 32 x 32 E8M0 scales."""
    raw, _dt, (r, c) = read(sh, name)
    sc, _sdt, (sr, scc) = read(sh, name[:-len("weight")] + "scale")
    v = E4M3[raw].reshape(r, c)
    s = np.exp2(sc.reshape(sr, scc).astype(np.float32) - 127.0)
    v = v * np.repeat(np.repeat(s, 32, 0), 32, 1)[:r, :c]
    bits = v.view(np.uint32)
    assert not (bits & 0xFFFF).any(), name          # exact in bfloat16
    return (bits >> 16).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    sh = Shards(a.ckpt)
    rows = []
    t0 = time.time()
    for name, (_f, dt, shape, o0, o1) in sorted(sh.t.items()):
        if len(shape) != 2 or ".experts." in name or ".engram.embed." in name:
            continue
        if dt == "BF16":
            raw, _dt, _shape = read(sh, name)
            bits = raw.view(np.uint16).reshape(shape)
            kind = "bf16"
        elif dt == "F8_E4M3" and a.fp8 and name.endswith(".weight"):
            bits = fp8_bits(sh, name)
            kind = "fp8"
        else:
            continue
        r, c = shape
        if c % 32 or r < 2:
            rows.append(dict(name=name, kind=kind, shape=shape, ok=False,
                             why="cols %% 32 = %d" % (c % 32) if c % 32 else "one row"))
            continue
        _out, rep = cops.kq_bf16_to_bf12(bits, c)
        rel = rep["largest"] / rep["rms"] if rep["rms"] else 0.0
        ok = rep["nonfinite"] == 0 and rel <= 2.0 ** -8
        rows.append(dict(name=name, kind=kind, shape=shape, ok=ok, zeroed=rep["zeroed"],
                         collisions=rep["collisions"], nonfinite=rep["nonfinite"],
                         largest=rep["largest"], rms=rep["rms"], rel=rel, n=r * c,
                         why="" if ok else ("Inf/NaN" if rep["nonfinite"] else "zeroed %.2e of RMS" % rel)))
    for kind in ("bf16", "fp8"):
        rs = [x for x in rows if x["kind"] == kind]
        if not rs:
            continue
        n = sum(x.get("n", 0) for x in rs)
        z = sum(x.get("zeroed", 0) for x in rs)
        bad = [x for x in rs if not x["ok"]]
        worst = max((x for x in rs if "rel" in x), key=lambda x: x["rel"], default=None)
        print("%s: %d matrices, %.3f G values; %d zeroed (%.5f%%), %d collisions; the worst largest "
              "zeroed %.2e of its RMS (%s); %d fail the rule" % (
                  kind, len(rs), n / 1e9, z, 100.0 * z / max(n, 1), sum(x.get("collisions", 0) for x in rs),
                  worst["rel"] if worst else 0, worst["name"] if worst else "-", len(bad)))
        for x in bad:
            print("  stays %s: %s %s: %s" % (kind, x["name"], x["shape"], x["why"]))
        top = sorted((x for x in rs if x.get("zeroed")), key=lambda x: -x["zeroed"])[:8]
        for x in top:
            print("  %-48s %-16s zeroed %9d (%.4f%%), largest %.3g = %.2e of RMS" % (
                x["name"], x["shape"], x["zeroed"], 100.0 * x["zeroed"] / x["n"], x["largest"], x["rel"]))
    print("(%.0f s)" % (time.time() - t0))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=list)


if __name__ == "__main__":
    main()
