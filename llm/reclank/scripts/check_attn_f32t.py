#!/usr/bin/env python3
"""Check the attention of a large group over the quantized caches
(GP_ATTN_QC_MT int16, GP_ATTN_Q8_MT int8, GP_ATTN_V8_MT int16 keys and int8
values, GP_ATTN_TQ_MT TQ6: the rotated values, the queries as given) and over
the float32 cache of the E4B (GP_ATTN_F32H, base 0) against
a float64 reference, on random caches: k_flash_f32t, the
kernel of a GPU without the tensor cores (this GPU acts as one:
RECLANK_GPU_CC=61; RECLANK_GPU_F32T=0 for the kernels before it).
Shapes: the 12B (16 query heads, 8 key heads, 256), the global layers of the
26B (16, 2, 512), Qwen3.6 (16, 2, 256); a window and a base; a group that
ends in a part of a tile.

    python scripts/check_attn_f32t.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("RECLANK_GPU_CC", "61")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.formats import tq6  # noqa: E402
from reclank.ir import program as P  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402


def quant(rng, n, per, bits):
    x = rng.standard_normal((n, per)).astype(np.float32)
    if bits == "tq":
        return tq6.quantize(x)          # 24 bytes and a norm for each 32 values
    if bits == 32:
        return x, np.ones((n, per // 32), np.float32)
    qmax = 127.0 if bits == 8 else 32767.0
    s = np.abs(x.reshape(n, -1, 32)).max(-1) / qmax
    q = np.round(x.reshape(n, -1, 32) / s[..., None]).reshape(n, per)
    return q.astype(np.int8 if bits == 8 else np.int16), s.astype(np.float32)


def run(op, q, kq, ks, vq, vs, qh, kvh, hd, pos, base, window, mirror):
    t = q.shape[0]
    prog = P.Program()
    out = np.zeros((t, qh * hd), np.float32)
    scores = np.zeros(qh * (pos + t) + 64, np.float32)
    if op == P.ATTN_F32H:
        # position-major rows (head stride hd)
        prog.emit(op, q, kq, vq, scores, out, qh, kvh, hd, t, pos, hd, window, 0)
    else:
        prog.emit(op, q, kq, ks, vq, vs, scores, out, qh, kvh, hd, t, pos, base, window,
                  np.zeros(t, np.int32), np.zeros(t, np.int32), None)
    prog.names["out"] = out
    g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
    g.run()
    g.download("out")
    res = out.copy()
    g.close()
    return res


def values(cq, cs, kvh, hd):
    """The float64 rows (n, kvh, hd) of a cache: q times its scales, or TQ6
    (uint8 bytes) in the rotated form."""
    n = len(cq)
    if cq.dtype == np.uint8:
        return tq6.dequantize_rotated(cq, cs).astype(np.float64).reshape(n, kvh, hd)
    return (cq.astype(np.float64).reshape(n, -1, 32) * cs[..., None]).reshape(n, kvh, hd)


def reference(q, kq, ks, vq, vs, qh, kvh, hd, pos, base, window, rows):
    k, v = values(kq, ks, kvh, hd), values(vq, vs, kvh, hd)
    out = {}
    for j in rows:
        p = pos + j
        lo = max(base, p - window + 1) if window else base
        keys = np.arange(lo, p + 1) - base
        qj = q[j].astype(np.float64).reshape(qh, hd)
        o = np.zeros((qh, hd))
        for h in range(qh):
            kv = h // (qh // kvh)
            s = k[keys, kv] @ qj[h]
            e = np.exp(s - s.max())
            o[h] = e @ v[keys, kv] / e.sum()
        out[j] = o.reshape(-1)
    return out


def main():
    rng = np.random.default_rng(0)
    mirror = Mirror()
    bad = 0
    cases = [  # name, qh, kvh, hd, t, pos, base, window, the scale of q
        ("12B", 16, 8, 256, 300, 0, 0, 0, 2.0),
        ("12B chunk", 16, 8, 256, 512, 0, 0, 0, 2.0),
        ("12B peaked", 16, 8, 256, 232, 512, 0, 0, 30.0),
        ("12B pos", 16, 8, 256, 200, 333, 0, 0, 2.0),
        ("12B window", 16, 8, 256, 300, 700, 100, 512, 2.0),
        ("26B global", 16, 2, 512, 150, 77, 0, 0, 2.0),
        ("26B peaked", 16, 2, 512, 512, 0, 0, 0, 30.0),
        ("12B global", 16, 1, 512, 512, 0, 0, 0, 30.0),
        ("12B glob 2", 16, 1, 512, 232, 512, 0, 0, 30.0),
        ("12B slide", 16, 8, 256, 232, 512, 0, 1024, 30.0),
        ("Qwen3.6", 16, 2, 256, 260, 1000, 0, 0, 2.0),
    ]
    forms = (("int16", P.ATTN_QC_MT, 16, 16), ("int8", P.ATTN_Q8_MT, 8, 8), ("k16v8", P.ATTN_V8_MT, 16, 8),
             ("f32", P.ATTN_F32H, 32, 32), ("tq6", P.ATTN_TQ_MT, "tq", "tq"))
    if os.environ.get("RECLANK_GPU_F32T") == "0":
        forms = forms[:1]       # (the int8 forms have no other kernel without the tensor cores)
    for name, qh, kvh, hd, t, pos, base, window, qs in cases:
        n = pos + t - base
        q = (rng.standard_normal((t, qh * hd)) * qs / np.sqrt(hd)).astype(np.float32)
        for form, op, kb, vb in forms:
            if op == P.ATTN_F32H and base:
                continue        # (its cache starts at position 0)
            kq, ks = quant(rng, n, kvh * hd, kb)
            vq, vs = quant(rng, n, kvh * hd, vb)
            if kb == vb and "glob" in name:
                vq, vs = kq, ks         # attention_k_eq_v: the values are the keys
            got = run(op, q, kq, ks, vq, vs, qh, kvh, hd, pos, base, window, mirror)
            rows = sorted({0, 1, 63, 64, t // 2, t - 2, t - 1})
            ref = reference(q, kq, ks, vq, vs, qh, kvh, hd, pos, base, window, rows)
            err = max(np.abs(got[j] - ref[j]).max() / np.abs(ref[j]).max() for j in rows)
            ok = err < 1e-4
            bad += not ok
            print("%-11s %-6s t %3d pos %4d: vs ref %.2e %s" % (name, form, t, pos, err,
                                                              "ok" if ok else "BAD"))
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
