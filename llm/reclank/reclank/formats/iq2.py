"""IQ2_XXS (ggml's 2.06-bit i-quant) in NumPy: the decoder of the reference and of the checks.

The tables are those of reclank/backends/include/iq2_tables.h (from ggml), read from that file. A
block of 256 values is 66 bytes (the layout is in the header). A value is
d * (2s + 1) / 8 * g * sign, g one of 8, 25, 43: so each group of 32 values is an int8 vector
(+-g) times one float32 scale, which the int8 kernels (dp4a, VNNI) use as it is.

    w = dequant(blocks)           # blocks: uint8 [..., n_blocks, 66] -> float32 [..., n_blocks * 256]
    q, s = int8_form(blocks)      # int8 [..., n_blocks * 256], float32 [..., n_blocks * 8]
    r = to_rows(blocks)           # KQ_IQ2R rows: uint8 [rows, row_bytes(n_blocks * 256)]

KQ_IQ2R (type 67, this runtime's rows of IQ2_XXS: the same bytes, placed for aligned loads): a row
of cols values is the 8 parts of each block (block b, part j at byte 64 b + 8 j: its 4 grid indices
and the uint32 of signs and scale), then the fp16 d of each block (at cols / 4 + 2 b), then zeros
to a multiple of 8 bytes. As stored, a block is 66 bytes, so its parts are only 2-byte aligned and
the kernels read them in bytes and halves; here a part is one 8-byte load, a warp's 32 parts 256
contiguous bytes.
"""
from __future__ import annotations

import os
import re

import numpy as np

BLOCK, BYTES = 256, 66
KQ_IQ2R = 67
_H = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backends", "include", "iq2_tables.h")


def _grid():
    text = open(_H).read()
    body = text[text.index("#define IQ2XXS_GRID"):]
    body = body[:body.index("}")]
    vals = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{16})ull", body)]
    assert len(vals) == 256
    return np.array(vals, np.uint64).view(np.uint8).reshape(256, 8)


GRID = _grid()                                                  # [256, 8] bytes 8, 25, 43
SIGNS = np.array([[-1 if (i | ((bin(i).count("1") & 1) << 7)) >> j & 1 else 1 for j in range(8)]
                  for i in range(128)], np.int8)                # [128, 8]


def int8_form(blocks):
    """The int8 values (+-g) and the float32 scale of each group of 32."""
    b = np.ascontiguousarray(blocks, dtype=np.uint8)
    assert b.shape[-1] == BYTES, b.shape
    lead = b.shape[:-1]
    d = b[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)          # [..., nb]
    qs = b[..., 2:].reshape(*lead, 8, 8)                                        # [..., nb, 8 groups, 8]
    idx = qs[..., :4]                                                           # [..., nb, 8, 4]
    aux = qs[..., 4:].copy().view(np.uint32)[..., 0]                            # [..., nb, 8]
    sgn = (aux[..., None] >> (7 * np.arange(4, dtype=np.uint32))) & 127         # [..., nb, 8, 4]
    q = GRID[idx].astype(np.int8) * SIGNS[sgn]                                  # [..., nb, 8, 4, 8]
    scale = d[..., None] * (2 * (aux >> 28).astype(np.float32) + 1) / 8         # [..., nb, 8]
    return q.reshape(*lead[:-1], -1), scale.reshape(*lead[:-1], -1)


def dequant(blocks):
    q, s = int8_form(blocks)
    return (q.reshape(*q.shape[:-1], -1, 32).astype(np.float32) * s[..., None]).reshape(q.shape)


def row_bytes(cols):
    """The bytes of a KQ_IQ2R row of cols values (a multiple of 256)."""
    return (cols // 4 + cols // 128 + 7) // 8 * 8


def to_rows(blocks, out=None):
    """IQ2_XXS blocks [rows, nb, 66] (or [nb * 66] bytes of each of rows rows) to KQ_IQ2R rows
    [rows, row_bytes(nb * 256)] (into out when given)."""
    b = np.asarray(blocks, np.uint8)
    rows = b.shape[0]
    b = b.reshape(rows, -1, BYTES)
    nb = b.shape[1]
    rb = row_bytes(nb * BLOCK)
    if out is None:
        out = np.empty((rows, rb), np.uint8)
    out = out.reshape(rows, rb)
    out[:, :64 * nb] = b[:, :, 2:].reshape(rows, 64 * nb)
    out[:, 64 * nb:66 * nb] = b[:, :, :2].reshape(rows, 2 * nb)
    out[:, 66 * nb:] = 0
    return out


def from_rows(rows_, cols):
    """KQ_IQ2R rows back to the blocks [rows, nb, 66]."""
    r = np.asarray(rows_, np.uint8).reshape(-1, row_bytes(cols))
    nb = cols // BLOCK
    out = np.empty((r.shape[0], nb, BYTES), np.uint8)
    out[:, :, 2:] = r[:, :64 * nb].reshape(-1, nb, 64)
    out[:, :, :2] = r[:, 64 * nb:66 * nb].reshape(-1, nb, 2)
    return out
