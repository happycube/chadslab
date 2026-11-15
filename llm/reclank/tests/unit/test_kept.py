"""A model's host arrays in the weightkeeper (reclank/runtime/kept.py; docs/GPU_SWAP_PLAN.md 3.7, step 4):
a load with no entry makes them and puts them in the keeper; the next maps them (make not called): the
same bytes, the linears rebuilt; other options another entry; a keeper started again with --save
restores the entry from its saved copy; no keeper: made as before."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from reclank.core import ops
from reclank.runtime import kept as K
from reclank.runtime import weightkeeper as Z

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
WK = os.path.join(ROOT, "scripts", "weightkeeper.py")


@pytest.fixture
def sockdir(tmp_path, monkeypatch):
    d = tmp_path / "socks"
    d.mkdir()
    monkeypatch.setenv("RECLANK_WEIGHTKEEPER_DIR", str(d))
    monkeypatch.delenv("RECLANK_WEIGHTKEEPER", raising=False)
    monkeypatch.setenv("RECLANK_KEPT", "1")
    return str(d)


@pytest.fixture
def ckpt(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    return str(d)


def _keeper(save):
    p = subprocess.Popen([sys.executable, WK, "--serve", "--threads", "2", "--save", save],
                         env=dict(os.environ, PYTHONPATH=ROOT), stderr=subprocess.PIPE, text=True)
    for _ in range(300):
        if Z._answers(Z.socket_path()) or p.poll() is not None:
            break
        time.sleep(0.1)
    assert p.poll() is None, p.stderr.read()
    return p


def _stop(p):
    try:
        Z.request("stop").close()
    except OSError:
        pass
    p.wait(timeout=30)


def _load(path, gpu, made):
    """A model's load: three F20 linears (to_f20 of seeded float32 W) with biases, and two arrays."""
    from reclank.encoders.runtime import _Plain, to_f20
    k = K.Kept("toy", path, dict(gpu=gpu), code=[__name__])

    def lin(i):
        def make():
            made.append(i)
            rng = np.random.default_rng(i)
            w = rng.standard_normal((32, 64)).astype(np.float32)
            ln = _Plain(w)
            ln.shape = w.shape
            ln.b = rng.standard_normal(32).astype(np.float32)
            ln.imin = np.float32(-1.5)
            assert to_f20(ln, gpu)
            return ln
        return k.linear("lin%d" % i, make)
    lins = [lin(i) for i in range(3)]
    arrs = [k.array("a%d" % i, lambda i=i: (made.append("a%d" % i), np.arange(10 * (i + 1), dtype=np.float32))[1])
            for i in range(2)]
    k.done()
    return k, lins, arrs


def _same(a, b):
    la, aa = a
    lb, ab = b
    for x, y in zip(la, lb):
        assert type(x) is type(y) and x.shape == y.shape and x.w is None and y.w is None
        assert np.array_equal(x.b, y.b) and type(y.imin) is np.float32 and y.imin == x.imin
        assert (x.f20.type, x.f20.rows, x.f20.cols) == (y.f20.type, y.f20.rows, y.f20.cols)
        assert np.asarray(x.f20.data).tobytes() == np.asarray(y.f20.data).tobytes()
    for x, y in zip(aa, ab):
        assert np.array_equal(x, y) and x.dtype == y.dtype


def _entries():
    return {e["id"]: e for e in json.loads(Z._recv_msg(Z.request("list")))}


@pytest.mark.skipif(not ops._COPS_READY, reason="the C library")
def test_kept(sockdir, ckpt, tmp_path, monkeypatch):
    save = str(tmp_path / "saved")
    # no keeper: made, nothing kept
    made = []
    k, *ref = _load(ckpt, True, made)
    assert k.mode is None and len(made) == 5
    p = _keeper(save)
    try:
        made = []
        k1, *r1 = _load(ckpt, True, made)
        assert k1.mode == "write" and len(made) == 5
        _same(ref, r1)
        assert not r1[1][0].flags.writeable and not np.asarray(r1[0][0].f20.data).flags.writeable
        es = _entries()
        assert es[k1.id]["source"] == "put" and es[k1.id]["leases"] == [os.getpid()]
        made = []
        k2, *r2 = _load(ckpt, True, made)
        assert k2.mode == "read" and made == [] and k2.id == k1.id
        _same(ref, r2)
        # the CPU's forms: another entry
        made = []
        k3, *r3 = _load(ckpt, False, made)
        assert k3.mode == "write" and len(made) == 5 and k3.id != k1.id
        for _ in range(300):
            if all(os.path.isfile(os.path.join(save, i, "index.json")) for i in (k1.id, k3.id)):
                break
            time.sleep(0.1)
        assert os.path.isfile(os.path.join(save, k1.id, "seg0.bin"))
    finally:
        _stop(p)
    # a new keeper: restored from the saved copy
    p = _keeper(save)
    try:
        made = []
        k4, *r4 = _load(ckpt, True, made)
        assert k4.mode == "read" and made == []
        _same(ref, r4)
        assert _entries()[k4.id]["source"] == "saved"
        # the checkpoint's files changed: another entry
        with open(os.path.join(ckpt, "config.json"), "w") as f:
            f.write('{"x": 1}')
        made = []
        k5, *_r = _load(ckpt, True, made)
        assert k5.mode == "write" and k5.id != k1.id and len(made) == 5
        # not asked (RECLANK_KEPT unset): made, nothing kept
        monkeypatch.delenv("RECLANK_KEPT")
        made = []
        k6, *_r = _load(ckpt, True, made)
        assert k6.mode is None and len(made) == 5
    finally:
        _stop(p)
