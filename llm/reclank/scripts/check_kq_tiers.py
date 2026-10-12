#!/usr/bin/env python3
"""Check the GGUF products (cops.kq_linear) of each CPU tier on random rows of
Q8_0, Q4_K, Q5_K, Q6_K, and F32, each tier in its own process (RECLANK_CPU):

- each tier against the float reference (the rows dequantized, x as the
  products see it): a relative error near the rounding of float32;
- avx2_vnni against avx2: the same bits (vpdpbusd and vpmaddubsw give the same
  lanes). On a CPU with AVX-512 VNNI and no AVX-VNNI, the avx2_vnni build takes
  the EVEX form (RECLANK_CPU_TEST_EVEX=1).

    python scripts/check_kq_tiers.py [--tiers scalar,avx2,avx2_vnni,avx512,avx512_vnni]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = {8: ("Q8_0", 34, 32), 12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256),
         0: ("F32", 4, 1)}
ROWS, COLS, T = 48, 1024, 3


def weights(type_, rng):
    """Random raw rows of a type, with sane fp16 scales."""
    name, bb, bv = TYPES[type_]
    if type_ == 0:
        return rng.standard_normal((ROWS, COLS)).astype(np.float32).view(np.uint8).reshape(ROWS, -1)
    nb = COLS // bv
    w = rng.integers(0, 256, (ROWS, nb, bb), dtype=np.uint8)
    d = (rng.random((ROWS, nb)) * 0.02 + 0.001).astype(np.float16).view(np.uint8).reshape(ROWS, nb, 2)
    if type_ == 8:
        w[:, :, 0:2] = d
    elif type_ in (12, 13):
        w[:, :, 0:2] = d
        w[:, :, 2:4] = (rng.random((ROWS, nb)) * 0.01).astype(np.float16).view(np.uint8).reshape(ROWS, nb, 2)
    elif type_ == 14:
        w[:, :, 208:210] = d
    return w.reshape(ROWS, -1)


def run_tier(out):
    """In the process of one tier: the products of each type."""
    sys.path.insert(0, ROOT)
    from reclank.backends.cpu import cops
    rng = np.random.default_rng(7)
    x = rng.standard_normal((T, COLS)).astype(np.float32)
    xq = np.empty((T, COLS), np.int8)
    xs = np.empty((T, COLS // 32), np.float32)
    xm = np.empty((T, COLS // 16), np.float32)
    cops.kq_quant_x(x, xq, xs, xm)
    res = {"tier": np.array(cops.TIER)}
    for type_ in TYPES:
        w = np.ascontiguousarray(weights(type_, rng))
        o = np.empty((T, ROWS), np.float32)
        cops.kq_linear(w, type_, ROWS, COLS, xq, xs, xm, x, T, o)
        res["out%d" % type_] = o
        # the reference: the rows as floats, on x as the products see it
        wf = cops.kq_rows(w, type_, COLS, np.arange(ROWS)) if type_ else w.view(np.float32)
        xd = x if type_ == 0 else (xq.astype(np.float32) * np.repeat(xs, 32, axis=1))
        res["ref%d" % type_] = (xd.astype(np.float64) @ wf.astype(np.float64).T).astype(np.float32)
    np.savez(out, **res)


def cpu_flags():
    try:
        return open("/proc/cpuinfo").read()
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiers", default="scalar,avx2,avx2_vnni,avx512,avx512_vnni")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        run_tier(args.one)
        return 0
    flags = cpu_flags()
    have = {"scalar": True, "avx2": " avx2 " in flags,
            "avx2_vnni": " avx_vnni " in flags or " avx512_vnni " in flags,
            "avx512": " avx512bw " in flags, "avx512_vnni": " avx512_vnni " in flags}
    got = {}
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for t in args.tiers.split(","):
            if not have.get(t):
                print("%-12s skipped (the CPU does not run it)" % t)
                continue
            env = dict(os.environ, RECLANK_CPU=t)
            if t == "avx2_vnni" and " avx_vnni " not in flags:
                env["RECLANK_CPU_TEST_EVEX"] = "1"
            p = os.path.join(d, t + ".npz")
            r = subprocess.run([sys.executable, __file__, "--one", p], env=env, capture_output=True, text=True)
            if r.returncode:
                print("%-12s FAIL to run:\n%s" % (t, r.stderr[-2000:]))
                ok = False
                continue
            got[t] = dict(np.load(p))
            errs = []
            for type_, (name, _, _) in TYPES.items():
                o, ref = got[t]["out%d" % type_], got[t]["ref%d" % type_]
                e = float(np.abs(o - ref).max() / max(np.abs(ref).max(), 1e-30))
                errs.append("%s %.1e" % (name, e))
                ok &= e < 2e-5
            print("%-12s against the reference: %s" % (t, ", ".join(errs)))
    if "avx2" in got and "avx2_vnni" in got:
        same = all(np.array_equal(got["avx2"]["out%d" % k], got["avx2_vnni"]["out%d" % k]) for k in TYPES)
        print("avx2_vnni and avx2: %s" % ("the same bits" if same else "DIFFERENT"))
        ok &= same
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
