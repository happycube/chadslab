"""Ported in part from Practical-RIFE (model/warplayer.py's grid; MIT License; Copyright (c) 2021
hzwer). See THIRD_PARTY_NOTICES.md.

NumPy references of the frame interpolation records (the Python interpreter's steps and the unit
tests): ENC_WARP, ENC_PRELU and ENC_MIX, the same float32 operations in the same order as interp.c
and encoders.cu (NumPy's float32 products and sums are not fused), so the three give the same bits
(ENC_MIX's exp apart: the C library's expf, CUDA's expf and NumPy's exp may differ in the last bit).
ENC_RESIZE's reference is the stereo port's (models/depth/ref.resize)."""
from __future__ import annotations

import numpy as np

f32 = np.float32


def grid(n):
    """torch.linspace(-1, 1, n) in float32 as its CPU kernel computes it: start + i step below the
    halfway point, end - (n - 1 - i) step from it (RIFE's warplayer.py grid)"""
    if n == 1:
        return np.array([-1], np.float32)
    step = f32(2) / f32(n - 1)
    i = np.arange(n)
    lo = (f32(-1) + step * i.astype(np.float32)).astype(np.float32)
    hi = (f32(1) - step * (n - 1 - i).astype(np.float32)).astype(np.float32)
    return np.where(i < n // 2, lo, hi).astype(np.float32)


def warp(x, fl, gx, gy, dx, dy):
    """x (h, w, C) sampled at each pixel plus the flow fl (h, w, 2), as ENC_WARP: torch's CUDA
    grid_sampler_2d (bilinear, padding "border", align_corners): the grid gx, gy plus the flow over
    dx, dy ((size - 1) / 2), unnormalized ((g + 1) / 2 (size - 1)), clipped to the image, the
    corners in bounds summed in the order nw, ne, sw, se."""
    x = np.asarray(x, np.float32)
    h, w, C = x.shape
    one, two = f32(1), f32(2)
    gxv = (gx[None, :] + (fl[..., 0] / f32(dx)).astype(np.float32)).astype(np.float32)
    gyv = (gy[:, None] + (fl[..., 1] / f32(dy)).astype(np.float32)).astype(np.float32)
    ix = (((gxv + one) / two).astype(np.float32) * f32(w - 1)).astype(np.float32)
    iy = (((gyv + one) / two).astype(np.float32) * f32(h - 1)).astype(np.float32)
    ix = np.minimum(f32(w - 1), np.maximum(ix, f32(0))).astype(np.float32)
    iy = np.minimum(f32(h - 1), np.maximum(iy, f32(0))).astype(np.float32)
    xw, yn = np.floor(ix), np.floor(iy)
    xe, ys = (xw + one).astype(np.float32), (yn + one).astype(np.float32)
    wts = (((xe - ix) * (ys - iy)).astype(np.float32), ((ix - xw) * (ys - iy)).astype(np.float32),
           ((xe - ix) * (iy - yn)).astype(np.float32), ((ix - xw) * (iy - yn)).astype(np.float32))
    out = np.zeros((h, w, C), np.float32)
    for (cx, cy), wt in zip(((xw, yn), (xe, yn), (xw, ys), (xe, ys)), wts):
        ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        xi = np.where(ok, cx, 0).astype(np.int64)
        yi = np.where(ok, cy, 0).astype(np.int64)
        out = np.where(ok[..., None], (out + (x[yi, xi] * wt[..., None]).astype(np.float32)).astype(np.float32), out)
    return out


def prelu(x, a):
    """x above 0, else a[col] x (ENC_PRELU)"""
    x = np.asarray(x, np.float32)
    return np.where(x > 0, x, (x * np.asarray(a, np.float32)).astype(np.float32)).astype(np.float32)


def mix(a, b, m):
    """a s + b (1 - s), s = 1 / (1 + exp(-m)) a row (ENC_MIX)"""
    s = (f32(1) / (f32(1) + np.exp(-np.asarray(m, np.float32)).astype(np.float32))).astype(np.float32)[:, None]
    t = (f32(1) - s).astype(np.float32)
    return ((a * s).astype(np.float32) + (b * t).astype(np.float32)).astype(np.float32)
