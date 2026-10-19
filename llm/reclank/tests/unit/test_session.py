"""The session layer (reclank/core/session.py) and the option mapping of the
old server scripts, with no model."""
from __future__ import annotations

import os
import sys

import random

import numpy as np

from reclank.core.session import HostRows, RowCache, SessionPool, common_len

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeSession:
    def __init__(self, max_len):
        self.max_len = max_len
        self.ids = []


def test_pool_reuses_the_longest_start():
    pool = SessionPool(FakeSession, max_sessions=2, min_shared=2)
    a, n = pool.pick(list(range(10)), 5)
    assert n == 0 and a.max_len == 23
    a.ids = list(range(10))
    b, n = pool.pick([0, 1, 2, 3, 99], 5)
    assert b is a and n == 4
    # a start below min_shared makes a new session; the pool keeps two
    c, n = pool.pick([7, 7, 7], 5)
    assert c is not a and n == 0
    d, n = pool.pick([8, 8], 5)
    assert len(pool.sessions) == 2 and a not in pool.sessions


class FakeDevice:
    """A device cache of max_len positions for RowCache: an array of a row a
    position and one of a row a block of 4, each row a value of the keys up
    to it (as a model's keys and values) and the tag of the run that wrote it
    (a model's rows of the same keys differ in their last bits when another
    prompt computed them: the kernels of another group size); the state is
    the keys read and the tags of their rows."""

    def __init__(self, max_len=256):
        self.max_len = max_len
        self.a1 = np.zeros((max_len, 2), np.int64)
        self.a4 = np.zeros((max_len // 4 + 1, 3), np.int32)
        self.keys = []
        self.reserved = 0

    def rows(self):
        return [HostRows(self.a1, 1), HostRows(self.a4, 4)]

    def reserve(self, n):
        self.reserved = max(self.reserved, n)

    @staticmethod
    def value(keys, p):
        return hash(tuple(keys[:p + 1])) & 0x3FFFFFFF

    def run(self, keys, k, tag=0):
        """Read keys from k (the rows of 0 .. k - 1 are there)."""
        for p in range(k, len(keys)):
            self.a1[p] = self.value(keys, p), tag
            if p % 4 == 3:
                self.a4[p // 4, :2] = self.value(keys, p), tag
        self.keys = list(keys)

    def state(self, n):
        return self.keys[:n], self.a1[:n, 1].copy(), self.a4[:n // 4, 1].copy()

    def check(self, keys, k, st=None):
        for p in range(k):
            assert self.a1[p, 0] == self.value(keys, p), (p, k)
            if p % 4 == 3:
                assert self.a4[p // 4, 0] == self.value(keys, p), (p, k)
        if st is not None:
            # the rows the context wrote itself
            assert (self.a1[:k, 1] == st[1][:k]).all() and (self.a4[:k // 4, 1] == st[2][:k // 4]).all(), k


def test_contexts_bring_back_their_rows():
    """Random conversations that share starts, branch and come back: after
    begin() the device has the rows of the start it returns, the very rows
    that the context wrote (not those of another one with the same keys)."""
    rng = random.Random(5)
    dev = FakeDevice()
    rc = RowCache(dev, max_contexts=4, max_bytes=0)
    convs = []
    hits = 0
    for it in range(400):
        r = rng.random()
        if convs and r < 0.7:
            # the next turn of a conversation (the answer, a new message)
            c = rng.choice(convs)
            keys = c + [rng.randrange(3) for _ in range(rng.randrange(1, 12))]
        elif convs and r < 0.85:
            # a branch from an earlier turn
            c = rng.choice(convs)
            keys = c[:rng.randrange(1, len(c) + 1)] + [7, 7] + [rng.randrange(3) for _ in range(5)]
        else:
            # a new conversation (some with the same start)
            keys = ([1, 2, 3, 4, 5, 6] if rng.random() < 0.5 else []) + [rng.randrange(9) for _ in range(8)]
        if len(keys) > dev.max_len - 4:
            continue
        k, st = rc.begin(keys, lambda s: True, as_is=dev.keys if rng.random() < 0.5 else None)
        if k == 0:
            dev.a1[:] = 0
            dev.a4[:] = 0
        else:
            hits += 1
            assert keys[:k] == (st[0] if st is not None else dev.keys[:k])
            dev.check(keys, k, st)
        dev.run(keys, k, tag=it + 1)
        cut = len(keys) - 2
        if k < cut:
            rc.point(keys[:cut], dev.state(cut))
        rc.point(keys, dev.state(len(keys)))
        convs = [cv for cv in convs if cv != keys][-6:] + [keys]
        assert len(rc.items) <= 4
    assert hits > 150


def test_contexts_keep_the_limits():
    dev = FakeDevice()
    rc = RowCache(dev, max_contexts=2, max_bytes=0, max_points=2)
    for j in range(4):
        keys = [j] * 20
        k, st = rc.begin(keys, lambda s: True)
        assert k == 0
        dev.run(keys, 0, tag=j + 1)
        for s in (5, 10, 15, 20):
            rc.point(keys[:s], dev.state(s))
    assert len(rc.items) == 2 and sorted(rc.live.points) == [15, 20]
    # the one before the last is in host memory: its rows come back
    k, st = rc.begin([2] * 20 + [9], lambda s: True)
    assert k == 20 and rc.live.keys == [2] * 20
    dev.check([2] * 20, 20, st)
    # the oldest is gone
    k, st = rc.begin([0] * 20 + [9], lambda s: True)
    assert k == 0


def test_common_len():
    a = list(range(10000))
    b = a[:9000] + [-1]
    assert common_len(a, b) == 9000 and common_len(a, a) == 10000 and common_len([], a) == 0


def test_old_scripts_map_their_options():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import serve
    import serve_qwen4
    assert serve.translate(["--gguf", "a", "--gpu", "hot", "--mtp", "D", "--mtp-n", "2"]) == \
        ["--family", "gemma4", "--gguf", "a", "--backend", "cuda", "--drafter", "D", "--mtp", "2"]
    assert serve.translate(["--gguf", "a"])[-2:] == ["--backend", "cpu"]
    assert serve.translate(["--gguf", "a", "--gpu=dense"])[4:] == ["--backend", "cuda", "--gpu-dense"]
    out = serve_qwen4.translate(["-m", "b", "--backend", "gpu"])
    assert out == ["--family", "qwen", "-m", "b", "--backend", "cuda", "--port", "8081"]
    from server.options import parser
    o = parser().parse_args(serve.translate(["--gguf", "a", "--gpu", "hot", "--max-context", "4096"]))
    assert o.model == "a" and o.backend == "cuda" and o.ctx == 4096
