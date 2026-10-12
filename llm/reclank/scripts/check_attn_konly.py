#!/usr/bin/env python3
"""Check the attention over a cache with keys only (the global layers of
Gemma 4 on the GPU, RECLANK_GPU_KONLY; ko_tab of kernels/operands.cu)
against a float64 reference, on random caches: V random, K = c rope(V)
(the rope of the global layers: head_dim 512, theta 1e6, a quarter of the
pairs turn) in int16 with ks, the values' scales vs = ks / c, the records
with the address of the keys for the values. The reference reads K and
V = unrope(K) / c in float64.

Records: a step (GP_ATTN_QC; k_attn_part for 16 query heads a key head,
k_attn_fdt for 8), a group of 2 to 4 queries (operand t; k_attn_fdtc, the
tensor cores), a prompt (GP_ATTN_QC_MT: k_flash_qc_h; RECLANK_GPU_FLASH512=0
k_flash_qc_tc; RECLANK_GPU_CC=61 k_flash_f32t, with RECLANK_GPU_F32T=0
k_flash_qc_mt), and the write of a row (GP_KV_WRITE with 1 / c).

    python scripts/check_attn_konly.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.ir import program as P  # noqa: E402
from reclank.backends.cuda.gpumm import Buffer  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror, _check, has_tc, lib  # noqa: E402
from reclank.models.gemma4 import rope as rope_mod  # noqa: E402

HD, C = 512, 0.0625
INV = rope_mod.proportional_inv_freq(HD, 1.0e6, 0.25)
RD = int(np.count_nonzero(INV))
_TABS = []


def set_table():
    """The tables of ko_tab, as gpu.ko_table makes them."""
    nhi = 4096
    for n, step in ((nhi, 256.0), (256, 1.0)):
        a = np.arange(n, dtype=np.float64)[:, None] * step * INV[None, :RD]
        t = np.ascontiguousarray(np.stack([np.cos(a), np.sin(a)], -1).astype(np.float32))
        b = Buffer(t.nbytes, "weights")
        b.upload(t)
        _TABS.append(b)
    _check(lib().rc_gpu_ko_table(_TABS[0].ptr, _TABS[1].ptr, RD, HD // 2, nhi))


def rope(x, pos, inverse=False):
    """x (n, kvh, hd) float64 turned by the angles of the positions."""
    f = np.outer(pos.astype(np.float64), INV)
    cos, sin = np.cos(np.concatenate([f, f], -1))[:, None], np.sin(np.concatenate([f, f], -1))[:, None]
    return x * cos - rope_mod.rotate_half(x) * sin if inverse else x * cos + rope_mod.rotate_half(x) * sin


def quant16(x):
    n = x.shape[0]
    s = np.abs(x.reshape(n, -1, 32)).max(-1) / 32767.0
    s = s.astype(np.float32)
    q = np.round(x.reshape(n, -1, 32) / s[..., None]).reshape(x.shape)
    return q.astype(np.int16).reshape(n, -1), s


def cache(rng, n, kvh):
    """kq, ks, vs of n rows from position 0, and the float64 K and V of the reference."""
    v = rng.standard_normal((n, kvh, HD))
    v /= np.sqrt((v ** 2).mean(-1, keepdims=True))          # the norm of no weight
    k = C * rope(v, np.arange(n))
    kq, ks = quant16(k)
    vs = (ks * np.float32(1.0 / C)).astype(np.float32)
    kd = (kq.astype(np.float64).reshape(n, -1, 32) * ks[..., None]).reshape(n, kvh, HD)
    return kq, ks, vs, kd, rope(kd, np.arange(n), inverse=True) / C


def attend(q, kd, vd, qh, kvh, keys):
    qj = q.astype(np.float64).reshape(qh, HD)
    o = np.zeros((qh, HD))
    for h in range(qh):
        kv = h // (qh // kvh)
        s = kd[keys, kv] @ qj[h]
        e = np.exp(s - s.max())
        o[h] = e @ vd[keys, kv] / e.sum()
    return o.reshape(-1)


def run(prog, out, mirror):
    prog.names["out"] = out
    g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
    g.run()
    g.download("out")
    res = out.copy()
    g.close()
    return res


def step(q, kq, ks, vs, qh, kvh, n, t, mirror):
    """t queries (rows of q), query j over n + j rows: GP_ATTN_QC."""
    prog = P.Program()
    out = np.zeros((t, qh * HD), np.float32)
    scores = np.zeros(qh * (n + t) + 64, np.float32)
    prog.emit(P.ATTN_QC, q, kq, ks, kq, vs, scores, out, qh, kvh, HD, n, *((t,) if t > 1 else ()))
    return run(prog, out, mirror)


def prompt(q, kq, ks, vs, qh, kvh, pos, mirror):
    """t queries from position pos: GP_ATTN_QC_MT."""
    t = q.shape[0]
    prog = P.Program()
    out = np.zeros((t, qh * HD), np.float32)
    scores = np.zeros(qh * (pos + t) + 64, np.float32)
    prog.emit(P.ATTN_QC_MT, q, kq, ks, kq, vs, scores, out, qh, kvh, HD, t, pos, 0, 0,
              np.zeros(t, np.int32), np.zeros(t, np.int32), None)
    return run(prog, out, mirror)


def write(rng, kvh, mirror):
    """GP_KV_WRITE with keys only: kq and ks as the int16 cache, vs = ks / c."""
    per = kvh * HD
    k = rng.standard_normal((1, per)).astype(np.float32)
    kq = np.zeros(per, np.int16)
    ks = np.zeros(per // 32, np.float32)
    vs = np.zeros(per // 32, np.float32)
    prog = P.Program()
    prog.emit(P.KV_WRITE, k, k, 0, 0, kq, ks, kq, vs, per, float(np.float32(1.0 / C)))
    prog.names.update(kq=kq, ks=ks, vs=vs)
    g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
    g.run()
    for name in ("kq", "ks", "vs"):
        g.download(name)
    g.close()
    rq, rs = quant16(k.reshape(1, kvh, HD).astype(np.float64))
    ok_q = np.abs(kq.astype(np.int32) - rq.reshape(-1)).max() <= 1
    ok_s = np.allclose(ks, rs.reshape(-1), rtol=1e-6) and np.array_equal(vs, ks * np.float32(1.0 / C))
    return ok_q and ok_s


def main():
    rng = np.random.default_rng(0)
    mirror = Mirror()
    set_table()
    bad = 0

    def report(name, got, ref):
        nonlocal bad
        err = max(np.abs(g - r).max() / np.abs(r).max() for g, r in zip(got, ref))
        ok = err < 2e-3      # (the tensor cores take float16 values)
        bad += not ok
        print("%-26s vs ref %.2e %s" % (name, err, "ok" if ok else "BAD"), flush=True)

    shapes = (("12B", 16, 1), ("26B", 16, 2))
    for name, qh, kvh in shapes:
        for n in (1, 100, 3000, 70000):
            kq, ks, vs, kd, vd = cache(rng, n, kvh)
            q = (rng.standard_normal((1, qh * HD)) * 30.0 / np.sqrt(HD)).astype(np.float32)
            got = step(q, kq, ks, vs, qh, kvh, n, 1, mirror)
            report("%s step n %d" % (name, n), [got[0]], [attend(q[0], kd, vd, qh, kvh, np.arange(n))])
    if has_tc():
        for t in (2, 3, 4):
            n = 2000
            kq, ks, vs, kd, vd = cache(rng, n + t, 2)
            q = (rng.standard_normal((t, 16 * HD)) * 30.0 / np.sqrt(HD)).astype(np.float32)
            got = step(q, kq, ks, vs, 16, 2, n, t, mirror)
            report("26B group t %d" % t, list(got),
                   [attend(q[j], kd, vd, 16, 2, np.arange(n + j)) for j in range(t)])
    for name, qh, kvh in shapes:
        for t, pos in ((300, 0), (200, 333), (64, 5000)):
            n = pos + t
            kq, ks, vs, kd, vd = cache(rng, n, kvh)
            q = (rng.standard_normal((t, qh * HD)) * 30.0 / np.sqrt(HD)).astype(np.float32)
            got = prompt(q, kq, ks, vs, qh, kvh, pos, mirror)
            rows = sorted({0, 1, t // 2, t - 1})
            report("%s prompt t %d pos %d" % (name, t, pos), [got[j] for j in rows],
                   [attend(q[j], kd, vd, qh, kvh, np.arange(pos + j + 1)) for j in rows])
    for kvh in (1, 2):
        ok = write(rng, kvh, mirror)
        bad += not ok
        print("kv write, %d key heads       %s" % (kvh, "ok" if ok else "BAD"))
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
