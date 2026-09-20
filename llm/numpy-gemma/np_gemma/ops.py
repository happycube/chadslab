"""Give the numeric functions for the model.

All functions use NumPy only.

Functions:
    rms_norm      Normalize the last axis. Multiply by a weight.
    linear        Multiply x by W.
    bf16_to_f32   Convert bfloat16 data to float32 data.
    linear_bf16   Multiply x by W. W is in bfloat16 format.
    gelu_tanh     Apply the GELU activation function.
    softmax       Change scores into probabilities.
    softcap       Limit the size of the logits.
"""
from __future__ import annotations

import os

import numpy as np

# The constant sqrt(2/pi). The GELU function uses it.
GELU_C = 0.7978845608028654


def rms_norm(x, weight=None, eps=1e-6):
    """Normalize the last axis of x. Multiply by the weight.

    The formula is x * pow(mean(x * x) + eps, -0.5) * weight.

    Note: the weight is the full scale. Do not add 1 to the weight.
    """
    x32 = np.asarray(x, dtype=np.float32)
    # Calculate the mean of the squares. Add eps for stability.
    mean_sq = np.mean(x32 * x32, axis=-1, keepdims=True) + eps
    y = x32 * np.power(mean_sq, -0.5)
    if weight is not None:
        y = y * np.asarray(weight, dtype=np.float32)
    return y


def linear(x, w):
    """Multiply x by W. Use the transpose of W.

    W has the shape (out_features, in_features).
    """
    return np.asarray(x, dtype=np.float32) @ np.asarray(w, dtype=np.float32).T


def bf16_to_f32(u16):
    """Convert bfloat16 data to float32 data.

    Move the 16 data bits to the top of a 32-bit word. The shift occurs in
    place. Thus only one float32 temporary is necessary.
    """
    raw = np.asarray(u16, dtype=np.uint16)
    out = raw.astype(np.uint32)   # One temporary. Then shift in place.
    out <<= 16
    return out.view(np.float32)


# The number of output rows in one dequant block. A small block keeps the
# float32 data in cache. A large block lowers the Python work. Change the value
# with the environment variable NP_GEMMA_BF16_CHUNK.
LINEAR_BF16_CHUNK = int(os.environ.get("NP_GEMMA_BF16_CHUNK", "8192"))


def linear_bf16(x, w_u16, chunk=LINEAR_BF16_CHUNK):
    """Multiply x by W. W is raw bfloat16 data.

    Convert one block of output rows at a time. Only one float32 block exists at
    a time. Thus the memory stays at the bfloat16 size.
    """
    x = np.asarray(x, dtype=np.float32)
    out_dim = w_u16.shape[0]
    out = np.empty((x.shape[0], out_dim), dtype=np.float32)
    for start in range(0, out_dim, chunk):
        stop = min(start + chunk, out_dim)
        out[:, start:stop] = x @ bf16_to_f32(w_u16[start:stop]).T
    return out


def gelu_tanh(x):
    """Apply the tanh approximation of GELU.

    This function agrees with torch.nn.functional.gelu(approximate="tanh").
    """
    x = np.asarray(x, dtype=np.float32)
    return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + 0.044715 * x * x * x)))


def softmax(x, axis=-1):
    """Change scores into probabilities.

    Subtract the maximum value before the exponent. This step prevents
    overflow.
    """
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def softcap(logits, cap):
    """Limit the size of the logits. Apply tanh(logits / cap) * cap."""
    return np.tanh(np.asarray(logits, dtype=np.float32) / cap) * cap
