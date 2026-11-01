"""reclank/models/seedvr2: the host parts (windows, sizes, padding, the resize, the VAE's indexes and
its arena), no weights."""
from __future__ import annotations

import numpy as np

from reclank.models.seedvr2 import pipeline as P
from reclank.models.seedvr2.dit import patchify, unpatchify, window_slices
from reclank.models.seedvr2.vae import _Arena, conv_index, down_index, up_index


def _cover(t, h, w, shifted):
    """each token in exactly one window"""
    seen = np.zeros((t, h, w), np.int32)
    for s in window_slices(t, h, w, shifted):
        seen[s] += 1
    return (seen == 1).all()


def test_windows():
    # 2048 x 2048: 128 x 128 tokens in windows of 20 x 20 (7 x 7); 1024: 64 x 64 in 4 x 4
    assert len(window_slices(1, 128, 128, False)) == 49 and len(window_slices(1, 128, 128, True)) == 49
    assert len(window_slices(1, 64, 64, False)) == 16
    assert len(window_slices(5, 60, 104, False)) == 48 and len(window_slices(5, 60, 104, True)) == 75
    for size in [(1, 50, 80), (1, 32, 32), (5, 60, 104), (2, 90, 160), (1, 135, 240), (1, 7, 300)]:
        assert _cover(*size, False) and _cover(*size, True), size


def test_patch():
    x = np.random.default_rng(0).standard_normal((2, 6, 8, 16)).astype(np.float32)
    rows = patchify(x)
    assert rows.shape == (2 * 3 * 4, 64)
    np.testing.assert_array_equal(unpatchify(rows, 2, 3, 4), x)


def test_sizes_and_padding():
    assert P.output_size(400, 640, 800, 1280) == ((800, 1280), (800, 1280))
    assert P.output_size(684, 512, 1368, 1024) == ((1368, 1024), (1360, 1024))
    x = np.zeros((3, 17, 16, 16), np.float32)
    assert P.pad_frames(x).shape[1] == 17
    assert P.pad_frames(x[:, :1]).shape[1] == 1
    for t, n in [(2, 5), (4, 5), (6, 9), (9, 9), (10, 13), (21, 21)]:
        y = np.arange(t, dtype=np.float32)[None, :, None, None] * np.ones((3, t, 2, 2), np.float32)
        p = P.pad_frames(y)
        assert p.shape[1] == n and (p[:, t:] == t - 1).all()


def test_resize_weights():
    for n_in, n_out in [(400, 800), (640, 1280), (512, 2048), (7, 3), (100, 37)]:
        M = P._aa_bicubic_weights(n_in, n_out)
        np.testing.assert_allclose(M.sum(1), 1.0, rtol=1e-6)
    # an upscale by 2 of a constant stays constant; of a ramp, a ramp inside
    img = np.full((1, 8, 8, 3), 128, np.uint8)
    x = P.transform(img, 16, 16)
    np.testing.assert_allclose(x, 128 / 255 * 2 - 1, atol=1e-6)


def test_wavelet():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 40, 48)).astype(np.float32)
    # the reconstruction of an image with itself is the image (high + low = image)
    np.testing.assert_allclose(P.wavelet_reconstruction(a, a), a, atol=1e-5)
    b = np.full_like(a, 0.25)
    # a constant style: the content's high frequencies plus the constant
    r = P.wavelet_reconstruction(a, b)
    np.testing.assert_allclose(r - P.wavelet_reconstruction(a, np.zeros_like(a)), 0.25, atol=1e-6)


def test_indexes():
    i = conv_index(4, 5)
    assert i.shape == (20, 9) and i[0, 4] == 0 and i[0, 0] == -1 and i[19, 8] == -1
    d = down_index(8, 6)
    assert d.shape == (12, 9) and d[0].tolist() == [0, 1, 2, 6, 7, 8, 12, 13, 14]
    assert (d[-1] == -1).sum() == 5            # the bottom row and the right column of padding
    u = up_index(2, 3)                          # 4 x 6 output over (2 3 4, C)
    assert u.shape == (24, 9)
    # the center tap of output pixel (Y, X): the (Y % 2, X % 2) block of input pixel (Y / 2, X / 2)
    for Y in range(4):
        for X in range(6):
            assert u[Y * 6 + X, 4] == ((Y // 2) * 3 + X // 2) * 4 + 2 * (Y % 2) + X % 2


def test_arena():
    ar = _Arena()
    a, b, c, d = (ar.new(n, 10, 7) for n in "abcd")
    forms = [("set", a, ("x",)), ("set", b, ("f", a)), ("set", c, ("f", b)), ("set", d, ("g", c, b))]
    env = {}
    arena = ar.place(forms, env)
    # a dies when b is made from it... b is read with c by d: b, c, d are live together
    starts = {n: env[n].__array_interface__["data"][0] for n in "abcd"}
    assert len({starts["b"], starts["c"], starts["d"]}) == 3
    assert arena.size <= 3 * 128           # (each 70 floats rounded to 64s: 128) a reused
    pool = np.zeros(arena.size + 10, np.float32)
    env2 = {}
    assert ar.place(forms, env2, pool=pool) is pool
