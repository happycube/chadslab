"""Rotary positional embeddings: default (sliding) and proportional (global)."""
from __future__ import annotations

import numpy as np


def default_inv_freq(head_dim, base):
    i = np.arange(0, head_dim, 2, dtype=np.float64)
    return 1.0 / np.power(base, i / head_dim)


def proportional_inv_freq(head_dim, base, partial_rotary_factor):
    """Inverse frequencies padded with zeros so the encoding is head_dim wide.

    Only the leading rope_angles pairs rotate; the rest get cos=1, sin=0.
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    rot = 1.0 / np.power(base, np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / head_dim)
    nope = np.zeros(head_dim // 2 - rope_angles, dtype=np.float64)
    return np.concatenate([rot, nope]) if nope.size else rot


def cos_sin(inv_freq, positions):
    freqs = np.outer(np.asarray(positions, dtype=np.float64), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply(x, cos, sin):
    # cos/sin are (..., head_dim); broadcast them over the head axis of x.
    while cos.ndim < x.ndim:
        cos = np.expand_dims(cos, -2)
        sin = np.expand_dims(sin, -2)
    return x * cos + rotate_half(x) * sin
