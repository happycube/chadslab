"""encoders.runtime.Residency: the GPU runners' LRU eviction, pins, scopes, the budget, the retry
after a failed cudaMalloc, and EVICT_ALWAYS; with a fake GPU (no CUDA)."""
from __future__ import annotations

import numpy as np
import pytest

from reclank.encoders import runtime as R


class FakeGPU:
    """The free memory of a card of total bytes, less what the fakes hold."""

    def __init__(self, total):
        self.total, self.used = total, 0

    def info(self):
        return self.total - self.used, self.total


class FakeRunner:
    """A runner of nbytes once loaded (load), none after release."""

    def __init__(self, gpu, label, nbytes):
        self.gpu, self.label, self.nbytes, self.held, self.releases = gpu, label, nbytes, 0, 0

    def load(self):
        if not self.held:
            self.gpu.used += self.nbytes
            self.held = self.nbytes

    def gpu_bytes(self):
        return self.held

    def release(self, keep_host=False):
        assert keep_host
        self.gpu.used -= self.held
        self.held = 0
        self.releases += 1


def _setup(total=100, reserve=0, budget=None, always=False, sizes=(30, 30, 30)):
    gpu = FakeGPU(total)
    res = R.Residency(mem_info=gpu.info, reserve=reserve, budget=budget, always=always, enabled=True)
    runners = []
    for i, n in enumerate(sizes):
        r = FakeRunner(gpu, "r%d" % i, n)
        res.register(r)
        runners.append(r)
    return gpu, res, runners


def _run(res, r):
    """A GPU run of r: before_run, then its load through make_room."""
    res.before_run(r)
    if not r.held:
        res.load(r, r.nbytes, r.load)


def test_lru_order():
    gpu, res, (a, b, c) = _setup(sizes=(30, 30, 30))
    for r in (a, b, c):
        _run(res, r)
    _run(res, a)                    # a is the most recent now: b is the oldest
    d = FakeRunner(gpu, "d", 30)
    res.register(d)
    _run(res, d)
    assert b.held == 0 and a.held and c.held and d.held
    e = FakeRunner(gpu, "e", 50)
    res.register(e)
    _run(res, e)                    # 50 more: c then a go (d is newer)
    assert c.held == 0 and a.held == 0 and d.held and e.held
    assert [k for k, *_ in res.log].count("evicted") == 3


def test_reserve_counts():
    gpu, res, (a, b) = _setup(total=100, reserve=20, sizes=(50, 40))
    _run(res, a)
    _run(res, b)                    # 50 + 40 + 20 > 100: a goes
    assert a.held == 0 and b.held == 40


def test_pins():
    gpu, res, (a, b, c) = _setup(sizes=(40, 40, 40))
    _run(res, a)
    res.pin(a)
    _run(res, b)
    _run(res, c)                    # a pinned: b goes, though a is older
    assert a.held and b.held == 0 and c.held
    res.unpin(a)
    _run(res, b)
    assert a.held == 0 and b.held and c.held


def test_scope_holds_from_first_run():
    gpu, res, (a, b, c) = _setup(sizes=(40, 40, 40))
    _run(res, a)
    _run(res, b)
    with res.use(a, c) as use:
        # a was run before the scope: not held yet, so c's room takes the oldest, a
        _run(res, c)
        assert a.held == 0 and b.held and c.held
        assert res.pinned(c) and not res.pinned(a)
        _run(res, a)                # c is held now (its first run in the scope): b goes
        assert b.held == 0 and a.held and c.held
        assert res.pinned(a)
    assert not res.pinned(a) and not res.pinned(c)


def test_early_unpin():
    gpu, res, (dit, vae) = _setup(sizes=(60, 60))
    with res.use(dit, vae) as use:
        _run(res, dit)
        use.done(dit)               # Wan: the DiT's part is over before the VAE
        assert not res.pinned(dit)
        _run(res, vae)
        assert dit.held == 0 and vae.held
        _run(res, dit)              # a run after done: not held again
        assert not res.pinned(dit)


def test_nested_scopes():
    gpu, res, (lm, codec) = _setup(sizes=(40, 40))
    with res.use(lm, codec) as outer:
        with res.use(lm):
            _run(res, lm)
            assert res.state[lm][1] == 2
        assert res.pinned(lm)
        outer.done(lm)
        assert not res.pinned(lm)


def test_budget():
    gpu, res, (a, b) = _setup(total=1000, budget=70, sizes=(40, 40))
    _run(res, a)
    _run(res, b)                    # free memory plenty, the budget not: a goes
    assert a.held == 0 and b.held == 40
    res.pin(b)
    c = FakeRunner(gpu, "c", 40)
    res.register(c)
    with pytest.raises(MemoryError):
        _run(res, c)                # b pinned, 40 + 40 > 70


def test_budget_estimate():
    # a first build's estimate of the C side may pass the budget: it evicts, but refuses only the known bytes
    gpu, res, (a,) = _setup(total=1000, budget=70, sizes=(40,))
    _run(res, a)
    b = FakeRunner(gpu, "b", 60)
    res.register(b)
    res.before_run(b)
    res.load(b, 90, b.load, sure=60)
    assert a.held == 0 and b.held == 60
    res.pin(b)
    with pytest.raises(MemoryError):
        res.make_room(a, 90, sure=40)       # b pinned: 60 + 40 > 70


def test_oom_retry():
    gpu, res, (a, b, c) = _setup(total=100, sizes=(30, 30, 30))
    for r in (a, b, c):
        _run(res, r)
    res.pin(c)
    d = FakeRunner(gpu, "d", 10)
    res.register(d)
    calls = []

    def make():
        calls.append(gpu.used)
        if len(calls) == 1:
            raise MemoryError("cudaMalloc")     # fragmentation, a C side's scratch: more than estimated
        d.load()
    res.before_run(d)
    res.load(d, 10, make)
    assert len(calls) == 2 and calls[1] == 30   # every unpinned runner went (c stays), then d
    assert a.held == 0 and b.held == 0 and c.held and d.held


def test_oom_retry_once():
    gpu, res, (a,) = _setup(sizes=(30,))
    _run(res, a)

    def make():
        raise MemoryError("cudaMalloc")
    with pytest.raises(MemoryError):
        res.load(a, 10, make)


def test_oom_keeps_the_runner():
    gpu, res, (a, b) = _setup(sizes=(30, 30))
    _run(res, a)
    _run(res, b)
    n = []

    def make():
        n.append(1)
        if len(n) == 1:
            raise MemoryError("cudaMalloc")
    res.load(b, 10, make)           # b's own memory (its uploads so far) is not evicted
    assert b.held and a.held == 0


def test_evict_always():
    gpu, res, (a, b, c) = _setup(total=1000, always=True, sizes=(30, 30, 30))
    _run(res, a)
    _run(res, b)
    assert a.held == 0 and b.held   # room for both, evicted all the same
    _run(res, b)
    assert b.releases == 1          # b itself too: its run uploads again
    with res.use(b, c):
        _run(res, b)
        _run(res, c)                # b held by the scope: stays
        assert b.held and c.held
    assert b.releases == 2


def test_victims_skip_empty():
    gpu, res, (a, b) = _setup(sizes=(30, 30))
    _run(res, b)
    assert res.victims() == [b]     # a holds nothing
    assert res.evict(a) == 0 and [k for k, *_ in res.log] == ["loaded"]


# ---- a real _Runner with a fake device side

class _Prog:
    def __init__(self, nbytes):
        self.keep = [np.zeros(nbytes // 4, np.float32)]
        self.names = {}


class _Buf:
    def __init__(self, gpu, n):
        self.gpu, self.nbytes = gpu, n
        gpu.used += n

    def free(self):
        self.gpu.used -= self.nbytes
        self.nbytes = 0


class _Mirror:
    def __init__(self):
        self.bufs, self.arrays, self.starts = {}, {}, []

    def nbytes(self):
        return sum(b.nbytes for b in self.bufs.values())


class _G:
    named = {}

    def close(self):
        pass


def test_disabled():
    # off (the default outside the generation server): no eviction, no budget refusal, no retry
    gpu = FakeGPU(100)
    res = R.Residency(mem_info=gpu.info, reserve=0, budget=50)
    assert not res.enabled
    a, b = FakeRunner(gpu, "a", 60), FakeRunner(gpu, "b", 60)
    for r in (a, b):
        res.register(r)
        res.before_run(r)
        res.load(r, r.nbytes, r.load)
    assert a.held and b.held and a.releases == 0 and gpu.used == 120
    calls = []

    def make():
        calls.append(1)
        raise MemoryError("cudaMalloc")
    with pytest.raises(MemoryError):
        res.load(a, 10, make)
    assert calls == [1] and b.held          # no retry, nothing evicted
    assert not [k for k, *_ in res.log if k == "evicted"]


def test_defaults(monkeypatch):
    for env, on in (({}, False), ({"RECLANK_RESIDENCY": "1"}, True), ({"RECLANK_EVICT_ALWAYS": "1"}, True)):
        for k in ("RECLANK_RESIDENCY", "RECLANK_EVICT_ALWAYS"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert R.Residency().enabled is on


def test_runner_get(monkeypatch):
    gpu = FakeGPU(150)
    res = R.Residency(mem_info=gpu.info, reserve=0, enabled=True)
    monkeypatch.setattr(R, "RESIDENCY", res)
    builds = []

    def load(self, key, prog):
        if self.mirror is None:
            self.mirror = _Mirror()
        a = prog.keep[0]
        if a.ctypes.data not in self.mirror.bufs:
            self.mirror.bufs[a.ctypes.data] = _Buf(gpu, a.nbytes)
        self.scratch[key] = self.seen[key] = 0
        builds.append((self.label, key))
        return _G()
    monkeypatch.setattr(R._Runner, "_load", load)
    monkeypatch.setattr(R._Runner, "C_EXTRA", 0)
    x = R._Runner(lambda n, g: _Prog(40), keep=2, keep_gpu=1)
    y = R._Runner(lambda n, g: _Prog(40), keep=2, keep_gpu=1)
    x.label, y.label = "x", "y"
    p, g = x.get(1, True)
    assert x.gpu_bytes() == 40 and x._need((1, True), p) == (0, 0)    # on the device, no scratch
    y.get(1, True)
    x.get(1, True)                  # cached: no build
    assert builds == [("x", (1, True)), ("y", (1, True))]
    z = R._Runner(lambda n, g: _Prog(40), keep=2, keep_gpu=1)
    z.label = "z"
    # z's first need: its 40 bytes and the estimate of the C side (the largest float32 array, 40)
    z.get(1, True)                  # 80 + 80 > 150: y, the oldest, goes; its host program is kept
    assert y.gpu_bytes() == 0 and y.mirror is None and (1, True) in y.hosts and x.gpu_bytes() == 40
    y.get(1, True)                  # reloaded from its host program (its need now 40: room for it)
    assert builds[-1] == ("y", (1, True)) and x.gpu_bytes() == 40 and gpu.used == 120
    assert [k for k, *_ in res.log if k != "loaded"] == ["evicted", "reloaded"]


def test_runner_get_disabled(monkeypatch):
    # off: _Runner.get builds straight away (no need, no room made, no eviction), as before phase 1
    gpu = FakeGPU(50)
    res = R.Residency(mem_info=gpu.info, reserve=0)
    monkeypatch.setattr(R, "RESIDENCY", res)
    monkeypatch.setattr(R._Runner, "_need", lambda self, key, prog: pytest.fail("_need called"))

    def load(self, key, prog):
        if self.mirror is None:
            self.mirror = _Mirror()
        self.mirror.bufs[prog.keep[0].ctypes.data] = _Buf(gpu, prog.keep[0].nbytes)
        self.scratch[key] = 0
        return _G()
    monkeypatch.setattr(R._Runner, "_load", load)
    x = R._Runner(lambda n, g: _Prog(40), keep=2, keep_gpu=1)
    y = R._Runner(lambda n, g: _Prog(40), keep=2, keep_gpu=1)
    x.get(1, True)
    y.get(1, True)                  # 80 of 50: no eviction (the cudaMalloc would decide)
    assert x.gpu_bytes() == 40 and y.gpu_bytes() == 40 and not res.log


def test_need():
    r = R._Runner(lambda n, g: None)
    prog = _Prog(400)
    view = prog.keep[0][10:20]
    prog.keep += [view, np.zeros(8, np.uint16)]     # a view counts as its array; uint16 is no float32
    assert r._need((1, True), prog) == (400 + 16 + 400 + R._Runner.C_EXTRA, 400 + 16)
    r.seen[(1, True)] = 7
    assert r._need((1, True), prog) == (400 + 16 + 7, 400 + 16 + 7)
