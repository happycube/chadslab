"""torchaudio's resampling (torchaudio.functional.resample, sinc_interp_hann)
on the host: the Higgs codec's 24 kHz to HuBERT's 16 kHz.

Ported from torchaudio (src/torchaudio/functional/functional.py, resample; BSD
2-Clause License; Copyright (c) 2017 Facebook Inc. (Soumith Chintala)). See
THIRD_PARTY_NOTICES.md.

    the rates over their gcd (o in, n out); the filters of the n phases at
    base = min(o, n) rolloff, width = ceil(lpw o / base) inputs each side:
    t = (-j / n + i / o) base for i in [-width, width + o), clamped to
    [-lpw, lpw]; the Hann window cos(t pi / lpw / 2)^2 times sinc(pi t),
    times base / o; computed in float32 as torchaudio does for a float32
    waveform
    the samples padded with width zeros before and width + o after; output
    frame f of phase j the filter's sum over the o f th inputs on; the
    phases interleaved; ceil(n L / o) samples kept

The sums are in float64 (torch's conv1d sums in float32).
"""
from __future__ import annotations

import math

import numpy as np


def sinc_kernel(orig, new, lowpass_filter_width=6, rolloff=0.99):
    """The filters (n, 2 width + o) float32 and width of a rate change, as
    torchaudio's _get_sinc_resample_kernel of a float32 waveform."""
    g = math.gcd(int(orig), int(new))
    o, n = int(orig) // g, int(new) // g
    lpw = lowpass_filter_width
    base = min(o, n) * rolloff
    width = math.ceil(lpw * o / base)
    f32 = np.float32
    idx = np.arange(-width, width + o, dtype=f32)[None] / f32(o)
    t = np.arange(0, -n, -1, dtype=f32)[:, None] / f32(n) + idx
    t = (t * f32(base)).astype(f32)
    t = np.clip(t, f32(-lpw), f32(lpw))
    window = np.cos(t * f32(math.pi) / f32(lpw) / f32(2)) ** 2
    t = t * f32(math.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(t == 0, f32(1.0), np.sin(t) / t).astype(f32)
    k = k * (window * f32(base / o))
    return k.astype(f32), width


def resample(x, orig, new, lowpass_filter_width=6, rolloff=0.99):
    """The samples x (float32, 1-D) at rate orig resampled to rate new."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if orig == new:
        return x.copy()
    g = math.gcd(int(orig), int(new))
    o, n = int(orig) // g, int(new) // g
    k, width = sinc_kernel(orig, new, lowpass_filter_width, rolloff)
    L = len(x)
    xp = np.concatenate([np.zeros(width, np.float64), x.astype(np.float64), np.zeros(width + o, np.float64)])
    K = k.shape[1]
    frames = (len(xp) - K) // o + 1
    win = np.lib.stride_tricks.as_strided(xp, (frames, K), (xp.strides[0] * o, xp.strides[0]))
    y = (win @ k.astype(np.float64).T).reshape(-1)
    return y[:math.ceil(n * L / o)].astype(np.float32)
