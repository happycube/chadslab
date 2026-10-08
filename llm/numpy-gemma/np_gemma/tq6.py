"""The TQ6 form of the KV cache (TurboQuant, 6 bits, the form of the MSE paper
with no QJL stage), for the Qwen models (NP_GEMMA_QWEN_KV=tq6).

A group of 32 values gets a fixed rotation: the signs SIGNS, then the
Walsh-Hadamard transform, divided by sqrt(32). The cache keeps the L2 norm of
the rotated group (float32, in the place of the scale of the int8 form) and
6 bits for each rotated value: the index of the nearest value of CODEBOOK.
CODEBOOK has the Lloyd-Max values for one coordinate of a random unit vector
in 32 dimensions (density (1 - x^2)^14.5). The rotated value is then
norm * CODEBOOK[index].

The attention works in the rotated form: the query gets the rotation (the
scores do not change, because the rotation is orthonormal), the values add
in the rotated form, and the output gets the inverse rotation (GP_TQ_ROT).

The 24 bytes of a group: byte j (0 to 15) has the low 4 bits of index j and
(in its high half) of index j + 16. Byte 16 + j (j 0 to 7) has the high 2
bits of indices j, j + 8, j + 16, j + 24 at bits 0, 2, 4, 6.

The tables are in csrc/tq6_tables.h (bf16_linear.c and gpu.cu include it)."""
import os
import re

import numpy as np


def _tables(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "tq6_tables.h")):
    """TQ6_SIGNS, TQ6_CODEBOOK, and TQ6_EDGES of csrc/tq6_tables.h, the one
    copy of the tables (the C and the CUDA code include it)."""
    text = open(path, encoding="utf-8").read()
    signs = int(re.search(r"#define\s+TQ6_SIGNS\s+(0x[0-9A-Fa-f]+)u?", text).group(1), 16)

    def floats(name):
        body = re.search(r"#define\s+%s\s*\{(.*?)\}" % name, text, re.S).group(1)
        vals = [v.strip().rstrip("fF") for v in body.replace("\\", " ").split(",")]
        return np.array([float(v) for v in vals if v], np.float32)
    return signs, floats("TQ6_CODEBOOK"), floats("TQ6_EDGES")


SIGN_BITS, CODEBOOK, EDGES = _tables()
assert CODEBOOK.size == 64 and EDGES.size == 63
# the codebook is symmetric, as tq.codebook(6, 32) of the study (.cache/work/tq.py)
# made symmetric; index i takes the values in (EDGES[i - 1], EDGES[i]]
SIGNS = np.where((SIGN_BITS >> np.arange(32)) & 1, -1.0, 1.0).astype(np.float32)


def wht32(x):
    """The Walsh-Hadamard transform of groups of 32 values (the last axis), in
    float32, in the order of the butterflies of the C code: stride 1 first."""
    x = np.array(x, np.float32).reshape(-1, 32)
    h = 1
    while h < 32:
        y = x.reshape(-1, 32 // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :], y[:, :, 1, :] = a + b, a - b
        x = y.reshape(-1, 32)
        h *= 2
    return x


_R = np.float32(1.0 / np.sqrt(32.0))


def _c_rotate():
    """cops.tq6_rotate when the library is there (the same bits, about 11
    times the speed: the rotated KV forms rotate the rows, the queries and
    the outputs of each block of a prompt), else None."""
    try:
        from . import cops
    except ImportError:
        return None
    return cops.tq6_rotate if cops.available() else None


def rotate(x):
    """The rotation of each group of 32 values of x (the last axis)."""
    f = _c_rotate()
    if f is not None and np.size(x) % 32 == 0:
        return f(x).reshape(np.shape(x))
    sh = np.shape(x)
    return (wht32(np.asarray(x, np.float32).reshape(-1, 32) * SIGNS) * _R).reshape(sh)


def unrotate(y):
    """The inverse of rotate."""
    f = _c_rotate()
    if f is not None and np.size(y) % 32 == 0:
        return f(y, inverse=True).reshape(np.shape(y))
    sh = np.shape(y)
    return (wht32(np.asarray(y, np.float32).reshape(-1, 32)) * _R * SIGNS).reshape(sh)


def quantize(x):
    """x (..., d), d a multiple of 32. Return the bytes (..., d * 3 / 4)
    (uint8) and the norms (..., d / 32) (float32)."""
    sh = np.shape(x)
    y = rotate(np.asarray(x, np.float32).reshape(-1, 32))
    nrm = np.sqrt((y * y).sum(1, dtype=np.float32)).astype(np.float32)
    u = y / np.maximum(nrm, np.float32(1e-30))[:, None]
    idx = np.searchsorted(EDGES, u, side="left").astype(np.uint8)
    lo, hi = idx & 15, idx >> 4
    b = np.empty((idx.shape[0], 24), np.uint8)
    b[:, :16] = lo[:, :16] | (lo[:, 16:] << 4)
    b[:, 16:] = hi[:, 0:8] | (hi[:, 8:16] << 2) | (hi[:, 16:24] << 4) | (hi[:, 24:32] << 6)
    return (b.reshape(*sh[:-1], sh[-1] // 32 * 24),
            nrm.reshape(*sh[:-1], sh[-1] // 32))


def indices(b):
    """The 6-bit indices (..., d) of the bytes (..., d * 3 / 4)."""
    sh = np.shape(b)
    g = np.asarray(b, np.uint8).reshape(-1, 24)
    lo = np.concatenate([g[:, :16] & 15, g[:, :16] >> 4], 1)
    hb = g[:, 16:]
    hi = np.concatenate([hb & 3, (hb >> 2) & 3, (hb >> 4) & 3, hb >> 6], 1)
    return (lo | (hi << 4)).reshape(*sh[:-1], sh[-1] // 24 * 32)


def dequantize_rotated(b, nrm):
    """The rotated values (..., d) of the bytes and the norms."""
    v = CODEBOOK[indices(b)]
    sh = v.shape
    return (v.reshape(-1, 32) * np.asarray(nrm, np.float32).reshape(-1, 1)).reshape(sh)


def dequantize(b, nrm):
    """The values (..., d) of the bytes and the norms."""
    return unrotate(dequantize_rotated(b, nrm))
