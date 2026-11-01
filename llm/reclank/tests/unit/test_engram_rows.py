"""models/deepseek41/engram_rows.py: engram's rows read through io_uring and through pread (cops.RowsPool, many in
flight) are those of numpy's fancy indexing of the file's map (rows across a 4 KB page, reads of a byte, the
fallback to pread when io_uring cannot be set up), and a prompt's rows read ahead on a thread are those of each group
in turn (groups across the chunks, a range it does not cover refused)."""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

er = pytest.importorskip("reclank.models.deepseek41.engram_rows")
cops = pytest.importorskip("reclank.backends.cpu.cops")


@pytest.fixture
def table(tmp_path):
    """A shard of two layers' tables (codes [R, 256], scales [R, 8]) after a header, and a source over it (a row of
    the codes across a 4 KB page every 16: the header's 664 bytes)."""
    rng = np.random.default_rng(3)
    R, head = 5000, 664
    codes = [rng.integers(0, 256, (R, 256), dtype=np.uint8) for _ in range(2)]
    scales = [rng.integers(0, 256, (R, 8), dtype=np.uint8) for _ in range(2)]
    t, at, blob = {}, head, [np.zeros(head, np.uint8)]
    for i, (c, s) in zip((1, 14), zip(codes, scales)):
        for name, a in (("weight", c), ("scale", s)):
            t["layers.%d.engram.embed.%s" % (i, name)] = ("x.safetensors", "U8", a.shape, at, at + a.nbytes)
            blob.append(a.reshape(-1))
            at += a.nbytes
    np.concatenate(blob).tofile(tmp_path / "x.safetensors")
    src = types.SimpleNamespace(path=str(tmp_path), t=t)
    return src, codes, scales, R


def check(src, outs, hashes):
    mm = np.memmap(src.path + "/x.safetensors", np.uint8, "r")
    for j, i in enumerate((1, 14)):
        _f, _dt, shape, o0, o1 = src.t["layers.%d.engram.embed.weight" % i]
        _f, _sdt, sshape, s0, s1 = src.t["layers.%d.engram.embed.scale" % i]
        flat = hashes[:, j].reshape(-1)
        assert np.array_equal(outs[j][0], np.asarray(mm[o0:o1]).reshape(shape)[flat])
        assert np.array_equal(outs[j][1], np.asarray(mm[s0:s1]).reshape(sshape)[flat])


@pytest.mark.parametrize("uring", [True, False])
def test_read_equals_fancy_indexing(table, uring):
    src, codes, scales, R = table
    rd = er.EngramReader(src, [1, 14], uring=uring, quiet=True)
    rng = np.random.default_rng(5)
    T, heads = 37, 24
    hashes = rng.integers(0, R, (T, 2, heads))
    hashes[0, :, :4] = [15, 31, 47, 63]             # (rows across a 4 KB page: offset 664 + 256 r, r = 15 mod 16)
    for ahead in (False, True):
        outs = rd.outs(T * heads)
        rd.read(hashes, outs, ahead=ahead)
        check(src, outs, hashes)
        if not uring:
            assert not rd.pool(ahead).uring


@pytest.mark.parametrize("uring", [True, False])
def test_pool_bytes_and_errors(table, uring):
    """single bytes, reads across a page, a read past the end of the file (an error, not a wrong row)"""
    src, _codes, _scales, _R = table
    path = src.path + "/x.safetensors"
    data = np.fromfile(path, np.uint8)
    fd = os.open(path, os.O_RDONLY)
    pool = cops.RowsPool(3, uring, 8)
    offs = np.array([0, 1, 4095, 4090, 8191, 12345, len(data) - 1], np.int64)
    lens = np.array([1, 1, 2, 300, 9, 4096, 1], np.int64)
    out = np.zeros(int(lens.sum()), np.uint8)
    dst = out.ctypes.data + np.concatenate([[0], np.cumsum(lens)[:-1]])
    pool.read(np.full(len(offs), fd), offs, lens, dst)
    assert np.array_equal(out, np.concatenate([data[o:o + n] for o, n in zip(offs, lens)]))
    with pytest.raises(OSError):
        pool.read(np.full(1, fd), np.array([len(data) - 4]), np.array([64]), np.array([out.ctypes.data]))
    os.close(fd)


@pytest.mark.parametrize("uring", [True, False])
def test_pool_after_an_error(table, uring):
    """a batch with a read past the end of the file in its middle fails as a whole, and the same pool's next read
    (its rings with nothing queued or in flight) gives the right bytes"""
    src, _codes, _scales, _R = table
    path = src.path + "/x.safetensors"
    data = np.fromfile(path, np.uint8)
    fd = os.open(path, os.O_RDONLY)
    pool = cops.RowsPool(2, uring, 4)
    rng = np.random.default_rng(11)
    n = 200
    offs = rng.integers(0, len(data) - 300, n).astype(np.int64)
    lens = rng.integers(1, 300, n).astype(np.int64)
    lens[n // 2] = 64
    bad = offs.copy()
    bad[n // 2] = len(data) - 2                     # (a read past the end of the file in the middle)
    out = np.zeros(int(lens.sum()), np.uint8)
    dst = out.ctypes.data + np.concatenate([[0], np.cumsum(lens)[:-1]])
    with pytest.raises(OSError):
        pool.read(np.full(n, fd), bad, lens, dst)
    out[:] = 0
    pool.read(np.full(n, fd), offs, lens, dst)
    assert np.array_equal(out, np.concatenate([data[o:o + m] for o, m in zip(offs, lens)]))
    os.close(fd)


def test_fallback_to_pread(table, monkeypatch):
    """when io_uring cannot be set up the reader takes pread (the same rows)"""
    src, _codes, _scales, R = table
    real = cops.RowsPool

    class NoUring(real):
        def __init__(self, workers, uring=True, qd=64, cpus=None):
            super().__init__(workers, False, qd, cpus)      # (as rc_rows_pool_new when io_uring_setup fails)
    monkeypatch.setattr(cops, "RowsPool", NoUring)
    rd = er.EngramReader(src, [1, 14], uring=True, quiet=True)
    hashes = np.random.default_rng(9).integers(0, R, (11, 2, 24))
    outs = rd.outs(11 * 24)
    rd.read(hashes, outs)
    check(src, outs, hashes)
    assert not rd.pool(False).uring and rd.pool(False).workers == 8


def test_ahead_groups(table):
    src, codes, scales, R = table
    rd = er.EngramReader(src, [1, 14], quiet=True)
    rng = np.random.default_rng(7)
    T, heads, pos = 300, 24, 1000
    hashes = rng.integers(0, R, (T, 2, heads))
    tokens = rng.integers(0, 1000, T)
    ah = er.EngramAhead(rd, hashes, tokens, pos, most=64, chunk=32)
    assert not ah.covers(pos - 1, tokens[:4])                       # (before its first position)
    assert not ah.covers(pos, tokens[1:5])                          # (other tokens)
    assert not ah.covers(pos, tokens[:4], dead=np.array([0, 1, 0, 0], bool))      # (an image's tokens)
    at = 0
    for n in (5, 40, 64, 1, 90, 100):                               # (groups across the chunks, not aligned)
        assert ah.covers(pos + at, tokens[at:at + n])
        outs = rd.outs(n * heads)
        assert ah.take(pos + at, n, outs)
        for j in range(2):
            flat = hashes[at:at + n, j].reshape(-1)
            assert np.array_equal(outs[j][0], codes[j][flat]) and np.array_equal(outs[j][1], scales[j][flat])
        at += n
    assert at == T and (not ah.chunks or max(ah.chunks) + 32 > at)  # (the chunks taken gone)
    ah.close()
    assert not ah.covers(pos, tokens[:1])
