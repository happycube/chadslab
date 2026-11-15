"""The 16-bit forms of float32 W for the tensor cores (encoders.runtime: W16H, W16I): their layouts and the
values they keep."""
import numpy as np

from reclank.encoders import runtime as R
from reclank.ir import program as P


def _w():
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((48, 256)) * np.logspace(-6, 1, 48)[:, None]).astype(np.float32)
    w[5] = 0                                    # a zero row
    w[7, :64] = 0                               # a zero group
    w[9, 3] = 3e4                               # a large value
    return w


def test_w16h():
    w = _w()
    m, k = w.shape
    d = R.w16h_pack(w)
    assert d.dtype == np.uint8 and d.size == 2 * m * k + 4 * m
    ws = d[2 * m * k:].view(np.float32)
    e = np.log2(ws)
    assert np.all(e == np.round(e))             # powers of two
    mx = np.abs(w).max(1)
    nz = mx > 0
    assert np.all((mx[nz] * ws[nz] >= 2 ** 14) & (mx[nz] * ws[nz] < 2 ** 15))
    v = R.w16h_values(d, m, k)
    # float16's spacing: 2^-11 of a normal value, 2^-24 (of W ws) among the subnormals
    assert np.all(np.abs(v - w) <= np.abs(w) * 2.0 ** -11 + 2.0 ** -25 / ws[:, None])


def test_w16i():
    w = _w()
    m, k = w.shape
    d = R.w16i_pack(w)
    assert d.shape == (m, P.i16_row_bytes(k)) and P.i16_row_bytes(k) % 16 == 0
    hi = d[:, :k].view(np.int8).astype(np.int32)
    lo = d[:, k:2 * k].view(np.int8).astype(np.int32)
    q = hi * 256 + lo
    assert np.abs(hi).max() <= 127 and np.abs(q).max() <= 32639
    sc = np.ascontiguousarray(d[:, 2 * k:2 * k + 4 * (k // 64)]).view(np.float32)
    assert sc[5].max() == 0 and sc[7, 0] == 0 and sc[7, 1] > 0
    v = R.w16i_values(d, m, k)
    # half a step of the group's scale (and float32's rounding of 32639 / mx)
    assert np.all(np.abs(v - w) <= np.repeat(sc, 64, axis=1) * 0.5 + np.abs(w) * 2.0 ** -21)
    g = np.abs(w).reshape(m, k // 64, 64).max(2)
    assert np.allclose(sc, g / 32639, rtol=1e-6)


def test_to_w16():
    w = _w()
    for fmt, t in (("f16", R.KQ_W16H), ("i16", R.KQ_W16I)):
        lin = R._Plain(w.copy())
        assert R.to_w16(lin, fmt) and lin.w is None and lin.wq.type == t and (lin.wq.rows, lin.wq.cols) == w.shape
        assert R.lin_q(lin) == ("wq", lin.wq)
    bad = R._Plain(np.ones((4, 48), np.float32))          # k not a multiple of 64
    assert not R.to_w16(bad, "i16") and bad.w is not None
