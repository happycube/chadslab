"""models/realbasicvsr's host parts and ENC_WARP's padding "zeros" (BasicVSR's feature warp): the record on the CPU and
in the Python interpreter against vfi/ref.warp (the same bits), its points past the image zeros and its border mode
unchanged; the k x k im2col index, SPyNet's working size, the official frames' float32. The checks against the
official network on real footage (outside the repository) are scripts/check_realbasicvsr.py's.

The image and flow: tests/unit/data/vfi_real.npz (test_vfi.py's: a crop of a frame Wan 2.2 TI2V-5B made here and
RIFE's real flow there, the flow times 8 so that some of its points leave the image)."""
from __future__ import annotations

import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler
from reclank.models.realbasicvsr import model as M
from reclank.models.vfi import ref

DATA = os.path.join(os.path.dirname(__file__), "data", "vfi_real.npz")
f32 = np.float32


def bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def warp_program():
    real = np.load(DATA)
    y0, x0 = real["crop"]
    h, w = 64, 128
    img = real["frames"][0, y0:y0 + h, x0:x0 + w].astype(np.float32) / f32(255)
    fl = (real["flow"][..., 0:2] * f32(8)).astype(np.float32)
    gx, gy = ref.grid(w), ref.grid(h)
    c = _EncCompiler(1e-6, pack=True)
    c.env.update(img=img.reshape(-1, 3).copy(), fl=fl.reshape(-1, 2).copy(), Z=np.zeros((h * w, 5), np.float32),
                 B=np.zeros((h * w, 3), np.float32))
    c.compile(("seq", ("enc_warp", "img", "fl", "Z", gx, gy, h, w, 3, 3, 0, 2, 0, 5, 2, True),
               ("enc_warp", "img", "fl", "B", gx, gy, h, w, 3)))
    p = c.p.finish()
    p.threads = p.buf[3] = 4
    dx, dy = f32((w - 1.0) / 2.0), f32((h - 1.0) / 2.0)
    want_z = ref.warp(img, fl, gx, gy, dx, dy, zeros=True)
    want_b = ref.warp(img, fl, gx, gy, dx, dy)
    return c.env, p, want_z, want_b, fl


def check(env, want_z, want_b, fl):
    h, w = 64, 128
    assert (bits(env["Z"][:, 2:5]) == bits(want_z.reshape(-1, 3))).all()
    assert (env["Z"][:, :2] == 0).all()
    assert (bits(env["B"]) == bits(want_b.reshape(-1, 3))).all()
    # the points past the image: zeros (or a partial sum) with "zeros", the clamped edge with "border"
    px = np.arange(w)[None, :] + fl[..., 0]
    py = np.arange(h)[:, None] + fl[..., 1]
    out = ((px < -1) | (px > w) | (py < -1) | (py > h)).reshape(-1)
    assert out.sum() > 50
    assert (env["Z"][out, 2:5] == 0).all() and (env["B"][out] != 0).any()


def test_warp_zeros_cpu_same_bits():
    env, p, want_z, want_b, fl = warp_program()
    p.run()
    check(env, want_z, want_b, fl)


def test_warp_zeros_python_steps():
    env, p, want_z, want_b, fl = warp_program()
    p.run_py()
    check(env, want_z, want_b, fl)


def test_conv_index():
    i = M.conv_index(5, 6, 7)
    assert i.shape == (30, 49) and i[0, 24] == 0 and i[0, 0] == -1 and i[29, 24] == 29
    assert (i[14] >= 0).sum() == 5 * 6          # a centre pixel of 5 x 6: every pixel in its 7 x 7
    assert (M.conv_index(4, 5) == M.conv_index(4, 5, 3)).all() and M.conv_index(4, 5).shape == (20, 9)


def test_sizes_and_frames():
    assert M.flow_size(480, 640) == (480, 640) and M.flow_size(190, 250) == (192, 256)
    u = np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1).repeat(3, 3)
    x = M.frames_f32(u)
    assert x.dtype == np.float32 and (x == (u.astype(np.float64) / 255.).astype(np.float32)).all()
    assert M.frames_f32(x) is x or (M.frames_f32(x) == x).all()
