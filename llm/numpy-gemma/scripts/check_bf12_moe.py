#!/usr/bin/env python3
"""CPU experts in KQ_BF12 (kq_rows_bf12_f in kq_moe_small_body and kq_moe_body)
against float64 products of the BF12 values."""
import ctypes, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import numpy as np
from np_gemma import cops
L = cops._lib
rng = np.random.default_rng(1)
E, H, I, k = 32, 2560, 640, 8
def bf12(rows, cols):
    a = (rng.standard_normal((rows, cols)) * 0.03).astype(np.float32)
    bits = (a.view(np.uint32) >> 16).astype(np.uint16)
    b12, _ = cops.kq_bf16_to_bf12(bits, cols)
    back = cops.kq_bf12_to_bf16(b12.reshape(-1), rows, cols)
    return b12.reshape(-1), (back.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
G, Gf = bf12(E * I, H); U, Uf = bf12(E * I, H); D, Df = bf12(E * H, I)
mats = cops.kq_moe_mats((G, 57), (U, 57), (D, 57), None)
vp, ci = ctypes.c_void_p, ctypes.c_int
L.kq_moe_numa_act.argtypes = [vp] * 5 + [ci] * 3 + [vp, vp] + [ci] * 2 + [vp, vp] + [ci] + [vp, vp, vp, vp]
L.kq_moe_numa_act.restype = None
def silu(x): return x / (1 + np.exp(-x))
ok = True
for t, act, label in ((1, 8, "one token (small body)"), (4, 8, "verify group of 4 (small body)"), (12, 0, "group of 12 (kq_moe_body)"), (12, 32, "group of 12, int16 flag (kq_moe_body)")):
    hf = rng.standard_normal((t, H)).astype(np.float32)
    hq = np.zeros((t, H), np.int8); hs = np.zeros((t, H // 32), np.float32); hm = np.zeros((t, H // 16), np.float32)
    cops.kq_quant_x(hf, hq, hs, hm)
    ids = np.stack([rng.choice(E, k, replace=False) for _ in range(t)]).astype(np.int32)
    val = rng.random((t, k)).astype(np.float32)
    sc = cops.kq_moe_scratch(t, k, E, H, I); out = np.zeros((t, H), np.float32)
    L.kq_moe_numa_act(hq.ctypes.data, hs.ctypes.data, hm.ctypes.data, ids.ctypes.data, val.ctypes.data,
                      t, k, E, mats.ctypes.data, None, H, I, sc.ctypes.data, out.ctypes.data, act,
                      hf.ctypes.data, None, None, None)
    ref = np.zeros((t, H))
    for j in range(t):
        x = hf[j].astype(np.float64)
        for q in range(k):
            e = ids[j, q]
            g = Gf[e * I:(e + 1) * I] @ x; u = Uf[e * I:(e + 1) * I] @ x
            ref[j] += val[j, q] * (Df[e * H:(e + 1) * H] @ (silu(g) * u))
    rel = float(np.abs(out - ref).max() / np.abs(ref).max())
    good = rel < 1e-4
    ok &= good
    print("%-36s max rel %.2e %s" % (label, rel, "" if good else "FAIL"))
print("PASS" if ok else "FAIL")
