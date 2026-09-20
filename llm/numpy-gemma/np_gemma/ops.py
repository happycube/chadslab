"""Numerically explicit primitives, NumPy only."""
from __future__ import annotations

import numpy as np

GELU_C = 0.7978845608028654  # sqrt(2/pi)


def rms_norm(x, weight=None, eps=1e-6):
    """Gemma RMSNorm: x * pow(mean(x^2) + eps, -0.5) * weight (weight is NOT 1+weight)."""
    x32 = np.asarray(x, dtype=np.float32)
    mean_sq = np.mean(x32 * x32, axis=-1, keepdims=True) + eps
    y = x32 * np.power(mean_sq, -0.5)
    if weight is not None:
        y = y * np.asarray(weight, dtype=np.float32)
    return y


def linear(x, w):
    """y = W x, with W stored as (out_features, in_features)."""
    return np.asarray(x, dtype=np.float32) @ np.asarray(w, dtype=np.float32).T


def gelu_tanh(x):
    """PyTorch gelu(approximate='tanh')."""
    x = np.asarray(x, dtype=np.float32)
    return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + 0.044715 * x * x * x)))


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def softcap(logits, cap):
    return np.tanh(np.asarray(logits, dtype=np.float32) / cap) * cap
