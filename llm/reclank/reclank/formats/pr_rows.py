"""The planar GPU rows of ggml's i-quants for GPUs without the tensor cores (Pascal: kernels/quant/pr_fmt.cuh),
and the decoders of Q2_K, Q3_K, IQ2_XS, IQ3_XXS and IQ3_S in NumPy that make and check them.

Ported from ggml (ggml/src/ggml-quants.c: dequantize_row_q2_K, _q3_K, _iq2_xs, _iq3_xxs,
_iq3_s; ggml/src/ggml-common.h: the blocks and the grids; MIT License; Copyright (c) 2023-2026
The ggml authors). See THIRD_PARTY_NOTICES.md.

The ShapeLearn GGUF files of Qwen3.8-27B (byteshape) mix these with IQ4_XS, Q4_K, Q5_K, Q6_K and
Q8_0. The grids are those of reclank/backends/include/iq_tables.h (from ggml), read from that file;
IQ2_XXS's is in iq2.py. Each type is an int8 value for each weight times a float scale for each
16 (Q2_K, Q3_K, IQ2_XS) or 32 (IQ3_XXS, IQ3_S), less a float offset for each 16 (Q2_K):

    q, s, m = int8_form(t, blocks)   # blocks uint8 [n, nbytes]: int8 [n, 256], scales, offsets (or None)
    w = dequant(t, blocks)           # float32 [n * 256]: (q * s - m) for each group

A block of 256 values (the byte layouts, ggml-common.h):

- Q2_K (84 bytes): 16 bytes of scales (low 4 bits: the scale, high: the min of each 16), 64 bytes
  of 2-bit codes (part n of 128: byte l of 32, bits 2j, the values 32j + l of the part), d, dmin.
  value = d sc q - dmin m.
- Q3_K (110 bytes): 32 bytes of high bits (bit j of byte l: value 32j + l of the part... of 128,
  as Q2_K's codes), 64 bytes of 2-bit codes, 12 bytes of 6-bit scales (as Q4_K's), d.
  value = d (sc - 32) (q - (high bit ? 0 : 4)).
- IQ2_XS (74 bytes): d, 32 uint16 (a 9-bit grid index of 8 values and a 7-bit sign index), 8
  bytes of scales (4 bits for each 16). value = d (0.5 + s) / 4 grid sign.
- IQ3_XXS (98 bytes): d, 64 bytes of 8-bit grid indices (4 values each), then a uint32 for each
  32 (four 7-bit sign indices, the 4-bit scale in bits 28-31). value = d (0.5 + s) / 2 grid sign.
- IQ3_S (110 bytes): d, 64 bytes of the low 8 bits of the grid indices, 8 bytes of their 9th
  bits, 32 bytes of signs (one bit a value), 4 bytes of scales (4 bits for each 32).
  value = d (1 + 2 s) grid sign.

The signs of IQ2_XS and IQ3_XXS: a 7-bit index i gives the 8 signs i | parity(i) << 7 (ggml's
ksigns_iq2xs).
"""
from __future__ import annotations

import os
import re

import numpy as np

Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S = 10, 11, 16, 17, 18, 21
BYTES = {Q2_K: 84, Q3_K: 110, IQ2_XS: 74, IQ3_XXS: 98, IQ3_S: 110}
TYPES = tuple(BYTES)
_H = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backends", "include", "iq_tables.h")


def _table(name, n, width):
    text = open(_H).read()
    body = text[text.index("#define " + name):]
    body = body[:body.index("}")]
    vals = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)", body)]
    assert len(vals) == n, (name, len(vals))
    return np.array(vals, np.uint64 if width == 8 else np.uint32).view(np.uint8).reshape(n, width)


IQ2XS_GRID = _table("IQ2XS_GRID", 512, 8)          # [512, 8] bytes 8, 25, 43
IQ3XXS_GRID = _table("IQ3XXS_GRID", 256, 4)        # [256, 4] bytes 4 ... 62
IQ3S_GRID = _table("IQ3S_GRID", 512, 4)            # [512, 4] bytes 1 ... 15
# the 8 signs (+-1) of a 7-bit sign index (the 8th the parity)
KSIGNS = np.array([[-1 if (i | ((bin(i).count("1") & 1) << 7)) >> j & 1 else 1 for j in range(8)]
                   for i in range(128)], np.int8)   # [128, 8]


def _f16(b, at):
    return b[:, at:at + 2].copy().view(np.float16)[:, 0].astype(np.float32)


def _k_scales12(sc):
    """The 16 6-bit scales of Q3_K's 12 bytes (ggml's kmask shuffle), as int."""
    a = sc.copy().view(np.uint32).astype(np.uint32)                 # [n, 3]
    k1, k2 = np.uint32(0x03030303), np.uint32(0x0f0f0f0f)
    t = a[:, 2]
    out = np.stack([(a[:, 0] & k2) | (((t >> 0) & k1) << 4), (a[:, 1] & k2) | (((t >> 2) & k1) << 4),
                    ((a[:, 0] >> 4) & k2) | (((t >> 4) & k1) << 4),
                    ((a[:, 1] >> 4) & k2) | (((t >> 6) & k1) << 4)], axis=1).astype(np.uint32)
    return out.view(np.uint8).reshape(-1, 16).astype(np.int32)


def _parts2(q):
    """The 2-bit codes of Q2_K / Q3_K (64 bytes: 2 parts of 128 values, byte l of 32, bits 2j for the
    values 32j + l) in value order, [n, 256]."""
    q = q.reshape(-1, 2, 1, 32)
    sh = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 4, 1)
    return ((q >> sh) & 3).reshape(-1, 256)


def int8_form(t, blocks):
    """(q int8 [n, 256], s float32 [n, groups], m float32 [n, groups] or None) of blocks uint8
    [n, BYTES[t]]: value = q s - m in each group of 256 / groups."""
    b = np.ascontiguousarray(blocks, np.uint8).reshape(-1, BYTES[t])
    n = b.shape[0]
    if t == Q2_K:
        sc = b[:, :16].astype(np.int32)
        q = _parts2(b[:, 16:80]).astype(np.int8)
        d, dm = _f16(b, 80), _f16(b, 82)
        return q, d[:, None] * (sc & 15), dm[:, None] * (sc >> 4)
    if t == Q3_K:
        hm = b[:, :32].reshape(n, 1, 32)
        bit = np.arange(8, dtype=np.uint8).reshape(1, 8, 1)
        hi = ((hm >> bit) & 1).reshape(n, 2, 4, 32).reshape(n, 256)       # bit j of byte l: value 32j + l
        q = _parts2(b[:, 32:96]).astype(np.int8) - 4 * (1 - hi.astype(np.int8))
        sc = _k_scales12(b[:, 96:108]) - 32
        return q, _f16(b, 108)[:, None] * sc, None
    if t == IQ2_XS:
        d = _f16(b, 0)
        qs = b[:, 2:66].copy().view(np.uint16).astype(np.int32)           # [n, 32]
        sc = b[:, 66:74].astype(np.int32)
        g = IQ2XS_GRID[qs & 511].astype(np.int8) * KSIGNS[qs >> 9]       # [n, 32, 8]
        s4 = np.stack([sc & 15, sc >> 4], axis=2).reshape(n, 16)          # one for each 16
        return g.reshape(n, 256), d[:, None] * (0.5 + s4) * 0.25, None
    if t == IQ3_XXS:
        d = _f16(b, 0)
        qs = b[:, 2:66].reshape(n, 8, 8)
        aux = b[:, 66:98].copy().view(np.uint32).reshape(n, 8)
        sg = (aux[..., None] >> (7 * np.arange(4, dtype=np.uint32))) & 127   # [n, 8, 4]
        g = IQ3XXS_GRID[qs].reshape(n, 8, 4, 8).astype(np.int8) * KSIGNS[sg]  # [n, 8, 4, 8]
        s = d[:, None] * (0.5 + (aux >> 28).astype(np.float32)) * 0.5
        return g.reshape(n, 256), s, None
    if t == IQ3_S:
        d = _f16(b, 0)
        qs = b[:, 2:66].astype(np.int32).reshape(n, 8, 8)                 # [n, part of 32, 8 indices]
        qh = b[:, 66:74].astype(np.int32).reshape(n, 8, 1)
        idx = qs | (((qh >> np.arange(8)) & 1) << 8)
        signs = b[:, 74:106].reshape(n, 8, 4, 1)
        bit = np.arange(8, dtype=np.uint8).reshape(1, 1, 1, 8)
        sg = 1 - 2 * ((signs >> bit) & 1).astype(np.int8)                 # [n, 8, 4, 8]
        g = IQ3S_GRID[idx].reshape(n, 8, 4, 8).astype(np.int8) * sg
        sc = b[:, 106:110].astype(np.int32)
        s8 = np.stack([sc & 15, sc >> 4], axis=2).reshape(n, 8)
        return g.reshape(n, 256), d[:, None] * (1 + 2 * s8), None
    raise ValueError("type %d" % t)


def dequant(t, blocks):
    """float32 values of blocks (uint8 [n, BYTES[t]] or flat) of type t."""
    q, s, m = int8_form(t, blocks)
    n, groups = s.shape
    v = q.reshape(n, groups, -1).astype(np.float32) * s[:, :, None].astype(np.float32)
    if m is not None:
        v -= m[:, :, None].astype(np.float32)
    return v.reshape(-1)


# ---- the rows of the GPU (backends/cuda/csrc/kernels/quant/pr_fmt.cuh) ----
# The same bits as the blocks, in planes, so each part of 32 values is aligned loads. IQ4_XS keeps its
# blocks (136 bytes, 8-aligned) under an id of its own (107: the kernels of pr.cu, not iq.cuh's of type
# 23); Q2_K goes to float32 rows (its tensors in byteshape's 27B are small: ssm_beta, 48 rows).
IQ4_XS = 23
KQ_IQ3SR, KQ_IQ3XR, KQ_Q3KR, KQ_IQ2SR, KQ_IQ2XR, KQ_Q6KR, KQ_IQ4XR = 101, 102, 103, 104, 105, 106, 107
ROW_TYPE = {IQ3_S: KQ_IQ3SR, IQ3_XXS: KQ_IQ3XR, Q3_K: KQ_Q3KR, IQ2_XS: KQ_IQ2SR, IQ2_XXS: KQ_IQ2XR, Q2_K: 0,
            IQ4_XS: KQ_IQ4XR}


def _r16(n):
    return (n + 15) // 16 * 16


def row_bytes(rtype, cols):
    """The bytes of a GPU row of cols values (a multiple of 256) of a runtime type of ROW_TYPE."""
    if rtype == KQ_IQ4XR:
        return cols // 256 * 136
    if rtype == KQ_IQ3SR:
        return _r16(cols // 4 + cols // 8 + cols // 32 + cols // 64 + cols // 128)
    if rtype == KQ_IQ3XR:
        return _r16(cols // 4 + cols // 8 + cols // 128)
    if rtype == KQ_Q3KR:
        return _r16(cols // 4 + cols // 8 + cols // 16 + cols // 128)
    if rtype == KQ_IQ2SR:
        return _r16(cols // 4 + cols // 32 + cols // 128)
    if rtype == KQ_IQ2XR:
        return (cols // 4 + cols // 128 + 7) // 8 * 8
    if rtype == KQ_Q6KR:
        return _r16(cols // 2 + cols // 4 + cols // 16 + cols // 128)
    raise ValueError("row type %d" % rtype)


def _q6k_codes(k):
    """The 6-bit codes (0 .. 63, the value + 32) of Q6_K blocks k [rows, nb, 210] in value order [rows, nb, 256]."""
    ql = k[..., :128].reshape(*k.shape[:-1], 2, 64).astype(np.int32)
    qh = k[..., 128:192].reshape(*k.shape[:-1], 2, 32).astype(np.int32)
    q1 = (ql[..., 0:32] & 15) | ((qh & 3) << 4)
    q2 = (ql[..., 32:64] & 15) | (((qh >> 2) & 3) << 4)
    q3 = (ql[..., 0:32] >> 4) | (((qh >> 4) & 3) << 4)
    q4 = (ql[..., 32:64] >> 4) | (((qh >> 6) & 3) << 4)
    return np.concatenate([q1, q2, q3, q4], axis=-1).reshape(*k.shape[:-1], 256)


def q6k_rows(blocks, rows):
    """Q6_K blocks (uint8, rows x the blocks of a row) as KQ_Q6KR rows (pr_fmt.cuh KQ_PR_Q6K: the Q6_K matrices
    of the 27B on Pascal)."""
    k = np.ascontiguousarray(blocks, np.uint8).reshape(rows, -1, 210)
    nb = k.shape[1]
    cols = 256 * nb
    c = _q6k_codes(k).astype(np.uint32).reshape(rows, cols // 32, 32)       # [rows, parts, 32]
    lo, hi = c & 15, ((c >> 4) & 3) ^ 2                                      # (the top bit inverted)
    lo = lo.reshape(rows, cols // 32, 2, 4, 4)                               # [part, half, k, b]
    lw = (lo[:, :, 0] | (lo[:, :, 1] << 4)) << (8 * np.arange(4, dtype=np.uint32))   # [part, k, b]
    lw = lw.sum(axis=3, dtype=np.uint64).astype(np.uint32)                   # [rows, parts, 4]
    hv = hi.reshape(rows, cols // 32, 2, 4, 4)                               # [part, u, s, b]
    sh = 8 * np.arange(4, dtype=np.uint32)[None, None, None, None, :] + 2 * np.arange(4, dtype=np.uint32)[None, None, None, :, None]
    hw = (hv << sh).sum(axis=(3, 4), dtype=np.uint64).astype(np.uint32)      # [rows, parts, 2]
    sc = k[:, :, 192:208]
    return _planes(rows, cols, KQ_Q6KR, [lw.view(np.uint8), hw.view(np.uint8), sc, k[:, :, 208:210]])


def _planes(rows, cols, rtype, parts):
    """Rows of planes: parts [(bytes uint8 [rows, n]), ...] in order, zeros to the row's bytes."""
    rb = row_bytes(rtype, cols)
    out = np.zeros((rows, rb), np.uint8)
    o = 0
    for p in parts:
        p = p.reshape(rows, -1)
        out[:, o:o + p.shape[1]] = p
        o += p.shape[1]
    assert o <= rb
    return out


def to_rows(t, blocks, rows):
    """(GPU rows uint8 [rows, row bytes] or float32 [rows, cols], the runtime type) of the blocks of a
    matrix of type t (uint8, rows x the blocks of a row)."""
    b = np.ascontiguousarray(blocks, np.uint8).reshape(rows, -1)
    if t == IQ4_XS:
        return b, KQ_IQ4XR
    if t == Q2_K:
        return dequant(t, b).reshape(rows, -1).astype(np.float32), 0
    if t == IQ2_XXS:
        from reclank.formats import iq2
        return iq2.to_rows(b.reshape(rows, -1, iq2.BYTES)), KQ_IQ2XR
    nb = b.shape[1] // BYTES[t]
    cols = nb * 256
    k = b.reshape(rows, nb, BYTES[t])
    if t == IQ3_S:
        return _planes(rows, cols, KQ_IQ3SR, [k[:, :, 2:66], k[:, :, 74:106], k[:, :, 66:74], k[:, :, 106:110],
                                              k[:, :, 0:2]]), KQ_IQ3SR
    if t == IQ3_XXS:
        return _planes(rows, cols, KQ_IQ3XR, [k[:, :, 2:66], k[:, :, 66:98], k[:, :, 0:2]]), KQ_IQ3XR
    if t == IQ2_XS:
        return _planes(rows, cols, KQ_IQ2SR, [k[:, :, 2:66], k[:, :, 66:74], k[:, :, 0:2]]), KQ_IQ2SR
    if t == Q3_K:
        q, _s, _m = int8_form(t, k.reshape(-1, BYTES[t]))
        c = (q.astype(np.int16) + 4).astype(np.uint32).reshape(rows, cols // 32, 2, 4, 4)   # [part, u, s, b]
        q2, hinv = c & 3, 1 - (c >> 2)
        sh = 8 * np.arange(4, dtype=np.uint32)[None, None, None, None, :] + \
            2 * np.arange(4, dtype=np.uint32)[None, None, None, :, None]
        codes = (q2 << sh).sum(axis=(3, 4), dtype=np.uint64).astype(np.uint32)            # [rows, parts, 2]
        hv = hinv.reshape(rows, cols // 32, 8, 4)                                            # [part, s', b]
        hs = 8 * np.arange(4, dtype=np.uint32)[None, None, None, :] + np.arange(8, dtype=np.uint32)[None, None, :, None]
        hi = (hv << hs).sum(axis=(2, 3), dtype=np.uint64).astype(np.uint32)                 # [rows, parts]
        sc = (_k_scales12(k[:, :, 96:108].reshape(-1, 12)) - 32).astype(np.int8)            # [rows nb, 16]
        return _planes(rows, cols, KQ_Q3KR, [codes.view(np.uint8), hi.view(np.uint8), sc.view(np.uint8),
                                             k[:, :, 108:110]]), KQ_Q3KR
    raise ValueError("type %d" % t)


def rows_dequant(rtype, r, cols):
    """float32 [rows, cols] of GPU rows (the decoders of pr_fmt.cuh in NumPy: the checks of to_rows)."""
    r = np.ascontiguousarray(r, np.uint8).reshape(-1, row_bytes(rtype, cols))
    n, P = r.shape[0], cols // 32
    o = [0]

    def take(nbytes):
        a = r[:, o[0]:o[0] + nbytes]
        o[0] += nbytes
        return np.ascontiguousarray(a)
    if rtype == KQ_IQ2XR:
        from reclank.formats import iq2
        nb = cols // 256
        parts = r[:, :nb * 64].reshape(n, nb, 64)
        d = r[:, cols // 4:cols // 4 + 2 * nb].reshape(n, nb, 2)
        return iq2.dequant(np.concatenate([d, parts], axis=2)).reshape(n, cols)
    if rtype == KQ_IQ3SR:
        qs, sg, qh, sc, d = take(cols // 4), take(cols // 8), take(cols // 32), take(cols // 64), take(cols // 128)
        k = np.concatenate([d.reshape(n, -1, 2), qs.reshape(n, -1, 64), qh.reshape(n, -1, 8), sg.reshape(n, -1, 32),
                            sc.reshape(n, -1, 4)], axis=2)
        return dequant(IQ3_S, k).reshape(n, cols)
    if rtype == KQ_IQ3XR:
        qs, aux, d = take(cols // 4), take(cols // 8), take(cols // 128)
        k = np.concatenate([d.reshape(n, -1, 2), qs.reshape(n, -1, 64), aux.reshape(n, -1, 32)], axis=2)
        return dequant(IQ3_XXS, k).reshape(n, cols)
    if rtype == KQ_IQ2SR:
        qs, sc, d = take(cols // 4), take(cols // 32), take(cols // 128)
        k = np.concatenate([d.reshape(n, -1, 2), qs.reshape(n, -1, 64), sc.reshape(n, -1, 8)], axis=2)
        return dequant(IQ2_XS, k).reshape(n, cols)
    if rtype == KQ_Q6KR:
        lw = take(cols // 2).view(np.uint32).reshape(n, P, 4)
        hw = take(cols // 4).view(np.uint32).reshape(n, P, 2)
        sc = take(cols // 16).view(np.int8).reshape(n, P, 2).astype(np.float32)
        d = take(cols // 128).view(np.float16).reshape(n, cols // 256).astype(np.float32)
        b = np.arange(4)
        lo = np.stack([(lw >> (8 * bb)) & 15 for bb in b], -1)                 # [n, P, k(4), b] values 4k + b
        lo2 = np.stack([(lw >> (8 * bb + 4)) & 15 for bb in b], -1)            # values 16 + 4k + b
        lov = np.concatenate([lo, lo2], axis=2).reshape(n, P, 32)
        sh = (2 * np.arange(4, dtype=np.uint32))[:, None] + (8 * np.arange(4, dtype=np.uint32))[None, :]   # [s, b]
        hv = np.stack([(hw[:, :, u, None, None] >> sh[None, None]) & 3 for u in range(2)], 2).reshape(n, P, 32)
        v = (lov | ((hv ^ 2) << 4)).astype(np.int32) - 32
        s = np.repeat(sc, 16, axis=2) * np.repeat(d, 8, axis=1)[:, :, None]
        return (v * s).reshape(n, cols)
    if rtype == KQ_Q3KR:
        codes = take(cols // 4).view(np.uint32).reshape(n, P, 2)
        hi = take(cols // 8).view(np.uint32).reshape(n, P)
        sc = take(cols // 16).view(np.int8).reshape(n, P, 2).astype(np.float32)
        d = take(cols // 128).view(np.float16).reshape(n, cols // 256).astype(np.float32)
        k = np.arange(8)
        b = np.arange(4)
        q2 = (codes[:, :, k // 4, None] >> (2 * (k % 4))[None, None, :, None] >> (8 * b)[None, None, None, :]) & 3
        hb = (hi[:, :, None, None] >> k[None, None, :, None] >> (8 * b)[None, None, None, :]) & 1
        v = (q2.astype(np.int32) - 4 * hb.astype(np.int32)).reshape(n, P, 32).astype(np.float32)
        s = np.repeat(sc, 16, axis=2) * np.repeat(d, 8, axis=1)[:, :, None]
        return (v * s).reshape(n, cols)
    raise ValueError("row type %d" % rtype)
