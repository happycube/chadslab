"""models/flashvsr's records and host parts: ENC_ATTN_BLOCKS (block-sparse attention) on the CPU (C) and in the
Python interpreter against a NumPy softmax of each row over its listed keys, and on the GPU (marked gpu) with K and V
float32 and float16; the block lists of a mask. The checks against the official pipeline on real footage (outside
the repository) are scripts/check_flashvsr.py's.

The rows: tests/unit/data/vfi_real.npz's two frames (a clip that Wan 2.2 TI2V-5B made here): 8 x 8 patches of
frame 0 the queries and values, of frame 1 the keys (384 rows of 192 values: two heads of 96)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from reclank.encoders.runtime import _EncCompiler

DATA = os.path.join(os.path.dirname(__file__), "data", "vfi_real.npz")
HEADS, HD, N = 2, 96, 384


def rows():
    fr = np.load(DATA)["frames"].astype(np.float32) / np.float32(255)
    def patches(f):
        h, w, _ = f.shape
        p = f[:h // 8 * 8, :w // 8 * 8].reshape(h // 8, 8, w // 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 192)
        return np.ascontiguousarray(p[:N] - np.float32(0.5), np.float32)
    return patches(fr[0]), patches(fr[1]), np.ascontiguousarray(patches(fr[0])[::-1], np.float32)


# query block b of head h: its key blocks (-1 ends them); one block with none
LISTS = np.array([[[0, -1, -1], [0, 2, -1], [-1, -1, -1]],
                  [[0, 1, 2], [2, -1, -1], [1, 2, -1]]], np.int32)


def want(q, k, v, hd=HD):
    o = np.zeros((N, HEADS * hd), np.float64)
    for h in range(HEADS):
        c = slice(h * hd, (h + 1) * hd)
        for b in range(N // 128):
            sel = LISTS[h, b][LISTS[h, b] >= 0]
            if sel.size == 0:
                continue
            keys = (sel[:, None] * 128 + np.arange(128)).reshape(-1)
            s = q[128 * b:128 * b + 128, c].astype(np.float64) @ k[keys, c].astype(np.float64).T
            p = np.exp(s - s.max(1, keepdims=True))
            o[128 * b:128 * b + 128, c] = (p / p.sum(1, keepdims=True)) @ v[keys, c]
    return o


def program(q, k, v, gpu=False):
    c = _EncCompiler(1e-6, pack=not gpu)
    c.env.update(q=q.copy(), k=k.copy(), v=v.copy(), o=np.full((N, HEADS * HD), 7, np.float32))
    c.compile(("set", "o", ("enc_attn_blocks", "q", "k", "v", LISTS, HEADS, HD)))
    p = c.p.finish()
    if not gpu:
        p.threads = p.buf[3] = 4
    return c.env, p


def test_attn_blocks_cpu():
    q, k, v = rows()
    env, p = program(q, k, v)
    p.run()
    w = want(q, k, v)
    assert np.abs(env["o"] - w).max() < 2e-6 and (env["o"][256:, :HD] == 0).all()


def test_attn_blocks_tile():
    """a tile of the rows (query blocks 1 and 2) over the lists of all the blocks (qb0)"""
    q, k, v = rows()
    c = _EncCompiler(1e-6, pack=True)
    c.env.update(q=q[128:].copy(), k=k.copy(), v=v.copy(), o=np.full((N - 128, HEADS * HD), 7, np.float32))
    c.compile(("set", "o", ("enc_attn_blocks", "q", "k", "v", LISTS, HEADS, HD, 1)))
    p = c.p.finish()
    p.threads = p.buf[3] = 4
    p.run()
    assert np.abs(c.env["o"] - want(q, k, v)[128:]).max() < 2e-6
    c.env["o"][...] = 7
    p.run_py()
    assert np.abs(c.env["o"] - want(q, k, v)[128:]).max() < 2e-6


def test_attn_blocks_python_steps():
    q, k, v = rows()
    env, p = program(q, k, v)
    p.run_py()
    assert np.abs(env["o"] - want(q, k, v)).max() < 2e-6


def mask_case():
    """2 heads of 96, slices of 6 blocks (a 2 x 3 grid, a window of 3 x 3: some blocks outside it), 2 query slices
    over 3 key slices (their rows in the key array in another order), a top-k of half"""
    from reclank.models.flashvsr import dit as D
    q, k, _ = rows()
    nb = 6
    qm, km = q[:2 * nb].copy(), k[:4 * nb].copy()
    kr = np.array([18 + j for j in range(6)] + list(range(12)), np.int32)          # key block j at row kr[j]
    ids = np.arange(100, 100 + 3 * nb, dtype=np.int32)
    local = D.local_mask(2, 3, r=3)
    topk = int(nb * nb * 0.5) - 1
    want = D.block_lists(D.draft_mask(qm, km[kr], nb, local, topk, heads=HEADS, hd=HD), ids)
    return qm, km, kr, local.astype(np.uint8), ids, nb, topk, want


def mask_program(gpu=False):
    qm, km, kr, local, ids, nb, topk, want = mask_case()
    c = _EncCompiler(1e-6, pack=not gpu)
    c.env.update(qm=qm, km=km, kr=kr, local=local, ids=ids, p=np.zeros(HEADS * 2 * nb * 3 * nb, np.float32),
                 thr=np.zeros(HEADS * 2, np.float32), bl=np.full((HEADS, 2 * nb, 3 * nb), 7, np.int32))
    c.compile(("enc_block_mask", "qm", "km", "kr", "local", "ids", "p", "thr", "bl", nb, HEADS, HD, topk))
    p = c.p.finish()
    if not gpu:
        p.threads = p.buf[3] = 4
    return c.env, p, want


def test_block_mask_cpu():
    env, p, want = mask_program()
    p.run()
    assert (env["bl"] == want).all() and (want == -1).any() and (want >= 0).any()
    env["bl"][...] = 7
    p.run_py()
    assert (env["bl"] == want).all()


@pytest.mark.gpu
def test_block_mask_gpu():
    from reclank.backends.cuda.runtime import GPUProgram
    env, p, want = mask_program(gpu=True)
    g = GPUProgram(p)
    g.run()
    g.download("bl")
    g.close()
    assert (env["bl"] == want).all()


def test_block_lists():
    from reclank.models.flashvsr.dit import block_lists
    m = np.zeros((2, 3, 5), bool)
    m[0, 0, [1, 3]] = True
    m[1, 2, :] = True
    bl = block_lists(m, np.array([4, 5, 6, 7, 8]))
    assert bl.shape == (2, 3, 5) and bl[0, 0, :3].tolist() == [5, 7, -1] and (bl[0, 1] == -1).all()
    assert bl[1, 2].tolist() == [4, 5, 6, 7, 8]


@pytest.mark.gpu
def test_attn_blocks_gpu():
    from reclank.backends.cuda.runtime import GPUProgram
    q, k, v = rows()
    w = want(q, k, v)
    env, p = program(q, k, v, gpu=True)
    g = GPUProgram(p)
    g.run()
    g.download("o")
    assert np.abs(env["o"] - w).max() < 2e-6
    k16, v16 = (a.astype(np.float16) for a in (k, v))
    env, p = program(q, k16.view(np.uint16), v16.view(np.uint16), gpu=True)
    g = GPUProgram(p)
    g.run()
    g.download("o")
    assert np.abs(env["o"] - want(q, k16.astype(np.float32), v16.astype(np.float32))).max() < 2e-6
    g.close()


@pytest.mark.gpu
def test_attn_blocks_gpu_tc():
    """the tensor cores' form (hd 128; K and V float16): q's rows two heads of 128 (the patches of frame 0 and 64
    values of frame 1's, the keys the other way round), against the softmax of q's float16 values (its weights
    float16 too: within 2e-3)"""
    from reclank.backends.cuda.runtime import GPUProgram
    a, b, c_ = rows()
    q, k = np.hstack([a, b[:, :64]]), np.hstack([b, a[:, :64]])
    v = np.hstack([c_, a[:, 64:128]])
    k16, v16 = (np.ascontiguousarray(x, np.float16) for x in (k, v))
    c = _EncCompiler(1e-6, pack=False)
    c.env.update(q=q.copy(), k=k16.view(np.uint16), v=v16.view(np.uint16), o=np.full(q.shape, 7, np.float32))
    c.compile(("set", "o", ("enc_attn_blocks", "q", "k", "v", LISTS, HEADS, 128, 0, 1)))
    g = GPUProgram(c.p.finish())
    g.run()
    g.download("o")
    w = want(q.astype(np.float16).astype(np.float32), k16.astype(np.float32), v16.astype(np.float32), hd=128)
    err = np.abs(c.env["o"] - w).max()
    assert err < 2e-3, err
    assert np.abs(c.env["o"][256:, :128]).max() == 0          # (head 0's block 2: no keys)
    g.close()
