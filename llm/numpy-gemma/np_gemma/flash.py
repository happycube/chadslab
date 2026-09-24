"""Flash attention in NumPy: a reference implementation.

The block path of model._attention builds the whole (queries x keys) score
matrix, masks it, normalises it, and multiplies by V. That is fine for a short
prompt and it is not fine for a long one: at 25000 keys, one layer of one chunk
holds a score tensor of about 400 MB, and the mask, the softmax, and the second
matmul each read it again.

This module gives the same result with a tile loop and the online softmax. It
holds one key tile for one query tile at a time, and it walks only the key tiles
that the causal mask and the sliding window leave visible. For this model 25 of
30 layers are sliding with a window of 1024, so the plain path computes most of
its scores to throw them away.

The name follows the published algorithm. This version is a reference: it is
correct and simple, not fast. The C kernel follows the same shape.
"""
from __future__ import annotations

import numpy as np
from . import ops


def _masked(s):
    """Replace the non-finite entries (a fully masked row) with 0."""
    return np.where(np.isfinite(s), s, 0.0)


def attention_reference(q, k, v, positions, base, window=0):
    """The plain path. Build the full scores and normalise them.

    q is (tokens, query heads, head_dim). k and v are (keys, kv heads,
    head_dim). positions gives the position of each query. base is the position
    of key 0. window of 0 turns the sliding window off. Return (tokens, query
    heads, head_dim).
    """
    t, qh, hd = q.shape
    n, kvh, _ = k.shape
    n_rep = qh // kvh
    kpos = base + np.arange(n)
    out = np.empty_like(q)
    for kv in range(kvh):
        h0 = kv * n_rep
        kb = k[:, kv, :]
        vb = v[:, kv, :]
        qb = q[:, h0:h0 + n_rep, :]
        s = np.einsum('thd,kd->thk', qb, kb)
        mask = kpos[None, None, :] <= positions[:, None, None]
        if window:
            mask &= (positions[:, None, None] - kpos[None, None, :]) < window
        s = np.where(mask, s, -np.inf)
        m = s.max(axis=-1, keepdims=True)
        p = _masked(np.exp(s - m))
        l = p.sum(axis=-1, keepdims=True)
        out[:, h0:h0 + n_rep, :] = np.einsum('thk,kd->thd', p, vb) / np.maximum(l, 1e-30)
    return out


def flash_attention(q, k, v, positions, base, window=0, block_q=64, block_k=128):
    """The tile path with the online softmax.

    The arguments match attention_reference. Block sizes are in queries and in
    keys. Return the same shape.
    """
    t, qh, hd = q.shape
    n, kvh, _ = k.shape
    n_rep = qh // kvh
    positions = np.asarray(positions)
    kpos = base + np.arange(n)
    out = np.empty_like(q)
    for kv in range(kvh):
        h0 = kv * n_rep
        kb = np.ascontiguousarray(k[:, kv, :])
        vb = np.ascontiguousarray(v[:, kv, :])
        for q0 in range(0, t, block_q):
            q1 = min(q0 + block_q, t)
            qb = q[q0:q1, h0:h0 + n_rep, :]
            pos = positions[q0:q1]
            bq = q1 - q0
            # Walk only the keys that any query in this block can see.
            lo = 0
            if window:
                lo = int(np.searchsorted(kpos, pos.min() - window + 1, side='left'))
            hi = int(np.searchsorted(kpos, pos.max(), side='right'))
            m = np.full((bq, n_rep), -np.inf)
            l = np.zeros((bq, n_rep))
            acc = np.zeros((bq, n_rep, hd))
            for j0 in range(max(lo, 0), hi, block_k):
                j1 = min(j0 + block_k, hi)
                s = np.einsum('bhd,kd->bhk', qb, kb[j0:j1])
                kp = kpos[j0:j1]
                mask = kp[None, None, :] <= pos[:, None, None]
                if window:
                    mask &= (pos[:, None, None] - kp[None, None, :]) < window
                s = np.where(mask, s, -np.inf)
                m_new = np.maximum(m, s.max(axis=-1))
                p = _masked(np.exp(s - m_new[:, :, None]))
                alpha = _masked(np.exp(m - m_new))
                l = l * alpha + p.sum(axis=-1)
                acc = acc * alpha[:, :, None] + np.einsum('bhk,kd->bhd', p, vb[j0:j1])
                m = m_new
            out[q0:q1, h0:h0 + n_rep, :] = acc / np.maximum(l, 1e-30)[:, :, None]
    return out
