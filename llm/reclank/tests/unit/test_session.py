"""The session layer (reclank/core/session.py) and the option mapping of the
old server scripts, with no model."""
from __future__ import annotations

import os
import sys

from reclank.core.session import SessionPool, SnapshotList

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


def test_snapshots_pick_the_longest_held_start():
    s = SnapshotList(2)
    s.add([1, 2, 3], "state3", None, None, None)
    s.add([1, 2, 3, 4, 5], "state5", None, "logits", None)
    has = lambda sn: sn[3] is not None  # noqa: E731
    # the cache holds 1..4 as it is; a snapshot of 5 tokens is longer
    k, snap = s.best([1, 2, 3, 4, 5, 6], [1, 2, 3, 4], has)
    assert k == 5 and snap[1] == "state5"
    # a snapshot of the whole prompt needs its logits
    k, snap = s.best([1, 2, 3], [], has)
    assert k == 0 and snap is None
    k, snap = s.best([1, 2, 3, 4, 5], [], has)
    assert k == 5
    # the same place replaces the old one; at most max_snaps
    s.add([1, 2, 3], "new3", None, None, None)
    s.add([9], "s9", None, None, None)
    assert [sn[1] for sn in s.snaps] == ["new3", "s9"]


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
