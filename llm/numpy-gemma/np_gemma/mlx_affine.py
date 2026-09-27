"""The MLX affine weight format (mlx-community files, OptiQ quants).

MLX quantizes each matrix in groups of 64 values along a row:
w = scale * q + bias, with scale and bias in bfloat16 and q of 4 or 8 bits,
packed in uint32 words from the low bits up. OptiQ selects the bits for
each matrix, so the bits come from the shape of the data.

QMat holds one matrix (or a stack: the experts of a layer) as views into
the memory map. dequant() gives float32 (the NumPy reference). QX and
linear() use the C kernels of csrc/mlx_affine.c (cops): x is quantized to
int8 for each group of 64, and the products run with VNNI.
"""
from __future__ import annotations

import numpy as np

from . import cops


def bf16(raw):
    """float32 values of raw bfloat16 bits (uint16)."""
    return (np.asarray(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class QMat:
    """One MLX-quantized matrix, or a stack of them (the experts).

    q is the uint32 data, (..., rows, cols * bits / 32). scales and biases are
    the bfloat16 bits (uint16), (..., rows, cols / 64). The arrays are views
    into the memory maps of the files.
    """

    group = 64

    def __init__(self, q, scales, biases):
        self.q = q
        self.scales = scales
        self.biases = biases
        self.rows = q.shape[-2]
        self.cols = scales.shape[-1] * self.group
        self.bits = q.shape[-1] * 32 // self.cols
        assert self.bits in (4, 8), "bits %d" % self.bits

    @property
    def nbytes(self):
        return self.q.nbytes + self.scales.nbytes + self.biases.nbytes

    def c(self):
        """(q, scales, biases, bits) for the C kernels."""
        return (self.q, self.scales, self.biases, self.bits)

    def values(self, q):
        """The integer values of uint32 words q, (..., rows, cols) as uint8."""
        b = np.ascontiguousarray(q).view(np.uint8)
        if self.bits == 8:
            return b
        out = np.empty(b.shape[:-1] + (b.shape[-1] * 2,), dtype=np.uint8)
        out[..., 0::2] = b & 15
        out[..., 1::2] = b >> 4
        return out

    def dequant(self, rows=None, expert=None):
        """Return float32 weights: the matrix, one expert of a stack, or some
        rows of it."""
        q, s, b = self.q, self.scales, self.biases
        if expert is not None:
            q, s, b = q[expert], s[expert], b[expert]
        if rows is not None:
            q, s, b = q[rows], s[rows], b[rows]
        v = self.values(q).astype(np.float32)
        g = v.reshape(v.shape[:-1] + (-1, self.group))
        w = g * bf16(s)[..., None] + bf16(b)[..., None]
        return w.reshape(v.shape)


class QX:
    """Rows of x, quantized for the products: one int8 array for each bit
    width (the order of 4-bit weights differs), with shared scales and sums."""

    def __init__(self, x):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.t, self.cols = self.x.shape
        ng = self.cols // 64
        self.xs = np.empty((self.t, ng), np.float32)
        self.xsum = np.empty((self.t, ng), np.float32)
        self.q = {}

    def get(self, bits):
        q = self.q.get(bits)
        if q is None:
            q = self.q[bits] = np.empty((self.t, self.cols), np.int8)
            cops.ma_quant_x(self.x, bits, q, self.xs, self.xsum)
        return q


def linear(mat, qx):
    """qx.x @ W^T for a QMat (one matrix), with the C kernel."""
    q = qx.get(mat.bits)
    out = np.empty((qx.t, mat.rows), np.float32)
    cops.ma_linear(mat.q, mat.scales, mat.biases, mat.bits, mat.rows, mat.cols, q, qx.xs,
                   qx.xsum, qx.t, out)
    return out
