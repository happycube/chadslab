"""Make rotary position embeddings (RoPE).

The model uses two RoPE types:
    default       Use this type for the sliding-window layers.
    proportional  Use this type for the global layers.

Proportional RoPE turns only the first angle pairs. The other pairs stay
unchanged.
"""
from __future__ import annotations

import numpy as np


def default_inv_freq(head_dim, base):
    """Return the inverse frequencies for the default RoPE type."""
    i = np.arange(0, head_dim, 2, dtype=np.float64)
    return 1.0 / np.power(base, i / head_dim)


def proportional_inv_freq(head_dim, base, partial_rotary_factor):
    """Return the inverse frequencies for the proportional RoPE type.

    The result has head_dim // 2 values. Add zeros to the end. Thus the
    encoding has the full head width.

    Only the first angle pairs turn. The other pairs get cos=1 and sin=0.
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    rot = 1.0 / np.power(base, np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / head_dim)
    nope = np.zeros(head_dim // 2 - rope_angles, dtype=np.float64)
    return np.concatenate([rot, nope]) if nope.size else rot


def cos_sin(inv_freq, positions):
    """Return the cosine and sine tables for the given positions.

    Join the frequency vector to itself. Thus the table pairs dimension i with
    dimension i + head_dim // 2.
    """
    freqs = np.outer(np.asarray(positions, dtype=np.float64), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    """Split the last axis into two halves. Swap the halves and negate the first."""
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply(x, cos, sin):
    """Apply RoPE to x.

    cos and sin have the shape (..., head_dim). Expand them over the head axis
    of x.
    """
    while cos.ndim < x.ndim:
        cos = np.expand_dims(cos, -2)
        sin = np.expand_dims(sin, -2)
    return x * cos + rotate_half(x) * sin
