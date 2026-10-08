#!/usr/bin/env python3
"""A study (no change to the runtime): the experts of Qwen3.8-Flash-Next
(the MTP layer first) in TQ6, rotated Q6_K and Q8_K, and other forms,
quantized from the original weights (RQ8_EXPERTS_PLAN.md; scripts/study_orig_q8.py
runs the forms on the original weights themselves).

Only weights that exist unquantized are test data: the shared experts, which
ModelOpt keeps in bfloat16 (the routed experts of the checkpoint are FP8 or
NVFP4, so they are not used). Each form is measured against the bfloat16
values: the same thing as quantizing the original model.

The error is ||Wq - W||^2 / ||W||^2 in dB; "output" is that of W x with a
Gaussian x (64 rows).

    PYTHONPATH=. python scripts/study_tq6_experts.py [models/Qwen3.8-Flash-Next-NVFP4]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import tq6  # noqa: E402
from np_gemma.st import SafeTensors  # noqa: E402
from np_gemma.st_qwen4 import E4M3  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.8-Flash-Next-NVFP4"
WHERE = json.load(open(os.path.join(D, "model.safetensors.index.json")))["weight_map"]
_files = {}


def get(name, dtype=np.float32):
    f = WHERE[name]
    if f not in _files:
        _files[f] = SafeTensors(os.path.join(D, f))
    return _files[f].get(name, dtype=dtype)


# ---- the forms (rows: outputs, cols: inputs; the groups along the inputs) ----

E4M3_POS = np.unique(np.abs(E4M3[np.isfinite(E4M3)]))
E4M3_POS = E4M3_POS[E4M3_POS <= 448]


def to_e4m3(v):
    a = np.minimum(np.abs(v), 448.0)
    i = np.clip(np.searchsorted(E4M3_POS, a), 1, len(E4M3_POS) - 1)
    lo, hi = E4M3_POS[i - 1], E4M3_POS[i]
    return np.sign(v) * np.where(a - lo <= hi - a, lo, hi)


def fp8_block(w, b=128):
    """FP8 with a scale for each 128 x 128 (FP8_PB_WO of ModelOpt)."""
    out = np.empty_like(w)
    for i in range(0, w.shape[0], b):
        for j in range(0, w.shape[1], b):
            blk = w[i:i + b, j:j + b]
            s = np.abs(blk).max() / 448.0 or 1.0
            out[i:i + b, j:j + b] = to_e4m3(blk / s) * s
    return out


def q_absmax(w, bits, g=32):
    """Q8_0 (bits 8), or the same with fewer bits: a float16 scale for each 32."""
    r, c = w.shape
    x = w.reshape(r, c // g, g)
    qmax = 2 ** (bits - 1) - 1
    d = np.abs(x).max(-1, keepdims=True) / qmax
    d = np.where(d == 0, 1, d).astype(np.float16).astype(np.float32)
    return (np.clip(np.round(x / d), -qmax - 1, qmax) * d).reshape(r, c)


def lloyd(bits, d=32, it=300):
    """The Lloyd-Max values of one coordinate of a random unit vector in d
    dimensions (for TQ4; TQ6 takes tq6.CODEBOOK)."""
    xs = np.linspace(-1, 1, 200001)[1:-1]
    p = (1 - xs * xs) ** ((d - 3) / 2)
    p /= p.sum()
    n = 2 ** bits
    cb = np.interp((np.arange(n) + 0.5) / n, np.cumsum(p), xs)
    for _ in range(it):
        k = np.searchsorted((cb[1:] + cb[:-1]) / 2, xs)
        cb = np.bincount(k, p * xs, n) / np.maximum(np.bincount(k, p, n), 1e-30)
    return cb.astype(np.float32), ((cb[1:] + cb[:-1]) / 2).astype(np.float32)


CB4, EDGES4 = lloyd(4)
CB6_I16 = np.round(tq6.CODEBOOK / np.abs(tq6.CODEBOOK).max() * 32767) / 32767 * \
    np.abs(tq6.CODEBOOK).max()


def tq(w, cb=tq6.CODEBOOK, edges=tq6.EDGES, mse=True, norm=np.float16, dec=None):
    """TQ6 (or TQ4 with CB4) on groups of 32 inputs: the rotation of tq6.py,
    the nearest value of the codebook for each value of the unit group, and a
    scale for each group: the L2 norm (the KV cache), or the MSE-optimal scale
    of the chosen values (mse). dec: the codebook of the decode (int16)."""
    sh = w.shape
    y = tq6.rotate(w.reshape(-1, 32))
    n = np.sqrt((y * y).sum(1))
    q = cb[np.searchsorted(edges, y / np.maximum(n, 1e-30)[:, None], side="left")]
    if mse:
        n = (y * q).sum(1) / np.maximum((q * q).sum(1), 1e-30)
    if dec is not None:
        q = dec[np.searchsorted(cb, q)]
    n = n.astype(norm).astype(np.float32)
    return tq6.unrotate(q * n[:, None]).reshape(sh)


GRID = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)


def nvfp4(w, search=True):
    """NVFP4: E2M1 values, an E4M3 scale for each 16, a float32 scale of the
    matrix. search: 8 scales for each block, the least squared error."""
    r, c = w.shape
    g = np.abs(w).max() / (448 * 6)
    x = w.reshape(r, c // 16, 16)
    base = np.abs(x).max(-1, keepdims=True) / 6 / g
    best = berr = None
    for f in (1.0, .95, .9, .85, .8, .75, 1.05, 1.1) if search else (1.0,):
        s = to_e4m3(base * f) * g
        s = np.where(s == 0, 1e-30, s)
        i = np.abs(np.abs(x / s)[..., None] - GRID).argmin(-1)
        q = np.sign(x) * GRID[i] * s
        e = ((q - x) ** 2).sum(-1, keepdims=True)
        if best is None:
            best, berr = q, e
        else:
            best, berr = np.where(e < berr, q, best), np.minimum(e, berr)
    return best.reshape(r, c)


def qx_scales(x, nmax=32):
    """make_qx_quants of llama.cpp (rmse_type 1, weights x^2) for the
    sub-blocks of 16 (the last axis): the float scale of each."""
    mx = np.take_along_axis(x, np.abs(x).argmax(-1)[..., None], -1)
    w = x * x
    best = best_s = None
    for i in range(-9, 10):
        iscale = -(nmax + 0.1 * i) / np.where(mx == 0, 1, mx)
        L = np.clip(np.round(iscale * x), -nmax, nmax - 1)
        suml2 = (w * L * L).sum(-1, keepdims=True)
        s = np.where(suml2 > 0, (w * x * L).sum(-1, keepdims=True) / np.maximum(suml2, 1e-30), 0)
        err = (w * (x - s * L) ** 2).sum(-1, keepdims=True)
        if best is None:
            best, best_s = err, s
        else:
            m = err < best
            best, best_s = np.where(m, err, best), np.where(m, s, best_s)
    return best_s


def q6k(w):
    """Q6_K as llama.cpp quantizes it: blocks of 256 (16 sub-blocks of 16 with
    int8 scales, a float16 d), and a last block of 128 (8 sub-blocks: the
    first half of a Q6_K block, 106 bytes) when cols % 256 == 128 (down, 640)."""
    r, c = w.shape
    assert c % 256 in (0, 128)
    out = np.empty_like(w)
    for b in range(0, c, 256):
        n = min(256, c - b)
        x = w[:, b:b + n].reshape(r, n // 16, 16)
        s = qx_scales(x)
        smax = np.take_along_axis(s, np.abs(s).argmax(1)[:, None], 1)
        d = np.where(smax == 0, 1e-30, smax / -128.0).astype(np.float16).astype(np.float32)
        si = np.clip(np.round(s / d), -128, 127) * d
        L = np.clip(np.round(x / np.where(si == 0, 1e-30, si)), -32, 31)
        out[:, b:b + n] = (L * si).reshape(r, n)
    return out


def q8k(w):
    """Q8_K as quantize_row_q8_K (as weights: no bsums): blocks of 256 (a
    last one of 128 for 640), d = signed max / -127 (float32)."""
    r, c = w.shape
    out = np.empty_like(w)
    for b in range(0, c, 256):
        x = w[:, b:b + 256]
        mx = np.take_along_axis(x, np.abs(x).argmax(1)[:, None], 1)
        d = np.where(mx == 0, 1e-30, mx) / -127.0
        out[:, b:b + 256] = np.clip(np.round(x / d), -128, 127) * d
    return out


def rotated(f):
    """The form f on the rotated weights (the rotation of tq6.py on each 32
    inputs): the product then takes the rotated x."""
    return lambda w: tq6.unrotate(f(tq6.rotate(w.reshape(-1, 32)).reshape(w.shape))
                                  .reshape(-1, 32)).reshape(w.shape)


def rel(a, ref):
    return float(((a - ref) ** 2).sum() / (ref ** 2).sum())


def db(e):
    return "%8.2f dB" % (10 * np.log10(e))


FORMS = [
    ("Q8_0 (8.5)", lambda w: q_absmax(w, 8)),
    ("rotated Q8_0 (8.5)", rotated(lambda w: q_absmax(w, 8))),
    ("Q8_K, 128 tail (8.125)", q8k),
    ("rotated Q8_K (8.125)", rotated(q8k)),
    ("FP8 128x128 (8.0)", fp8_block),
    ("TQ6, L2 norm, f32 (7.0)", lambda w: tq(w, mse=False, norm=np.float32)),
    ("TQ6, MSE norm, f16 (6.5)", tq),
    ("TQ6, int16 codebook", lambda w: tq(w, dec=CB6_I16)),
    ("Q6 absmax/32 (6.5)", lambda w: q_absmax(w, 6)),
    ("Q6_K, 128 tail (6.56)", q6k),
    ("rotated Q6_K (6.56)", rotated(q6k)),
    ("TQ4, MSE norm (4.5)", lambda w: tq(w, CB4, EDGES4)),
    ("NVFP4, amax (4.5)", lambda w: nvfp4(w, False)),
    ("NVFP4, searched (4.5)", nvfp4),
]



def main():
    layers = [None] + list(range(0, 48, 4)) + [47]
    name = lambda i: ("mtp.layers.0.mlp.shared_expert.%s_proj.weight" if i is None else  # noqa: E731
                      "model.language_model.layers.%d.mlp.shared_expert.%%s_proj.weight" % i)
    rng = np.random.default_rng(0)
    err = {n: {x: [] for x in ("gate", "up", "down")} for n, _ in FORMS}
    worst = {n: -1e9 for n, _ in FORMS}
    kurt = {x: [] for x in ("gate", "up", "down")}
    for i in layers:
        for x in ("gate", "up", "down"):
            w = get(name(i) % x)
            kurt[x].append(((w - w.mean()) ** 4).mean() / w.var() ** 2)
            X = rng.standard_normal((64, w.shape[1])).astype(np.float32)
            for n, f in FORMS:
                wq = f(w)
                e = rel(wq, w)
                err[n][x].append((e, rel(X @ wq.T, X @ w.T)))
                worst[n] = max(worst[n], 10 * np.log10(e))
    print("The bfloat16 shared experts of %d layers (the MTP layer, layers %s), against"
          " the bfloat16 values." % (len(layers), ",".join(str(i) for i in layers[1:])))
    print("kurtosis: " + ", ".join("%s %.0f-%.0f" % (x, min(k), max(k)) for x, k in kurt.items()))
    print("%-26s %10s %10s %10s %10s %10s %10s" % ("", "gate", "up", "down", "mean", "worst",
                                                    "output"))
    for n, _ in FORMS:
        a = {x: np.array(err[n][x]) for x in err[n]}
        allw = np.concatenate([a[x][:, 0] for x in a])
        allo = np.concatenate([a[x][:, 1] for x in a])
        print("%-26s %s %s %s %s %7.2f dB %s" % (n, db(a["gate"][:, 0].mean()),
                                                 db(a["up"][:, 0].mean()), db(a["down"][:, 0].mean()),
                                                 db(allw.mean()), worst[n], db(allo.mean())))


if __name__ == "__main__":
    main()
