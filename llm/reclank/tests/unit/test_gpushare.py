"""server/gpushare.py: the peer files, the admin routes (over HTTP), the asking of the peers for room
(the order, the hold, busy peers, the locks), with fake servers on a fake GPU (no CUDA)."""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

import pytest

from reclank.encoders import runtime as R
from server import gpushare as G
from server import http as H

KEY = "test-key"


class FakeGPU:
    def __init__(self, total):
        self.total, self.used = total, 0

    def info(self):
        return self.total - self.used, self.total


class FakeShare(G.Share):
    """A server of nbytes on the fake GPU: locked or not, its gate (a GPU job) held or not."""

    kind = "fake"

    def __init__(self, gpu, nbytes, locked=False, **kw):
        super().__init__(**kw)
        self.gpu, self.held, self.is_locked = gpu, nbytes, locked
        gpu.used += nbytes
        self.gate = threading.Lock()
        self.released = 0

    def gates(self):
        return [self.gate]

    def bytes(self):
        return self.held

    def locked_bytes(self):
        return self.held if self.is_locked else 0

    def models(self):
        return {"m%d" % self.port: self.is_locked}

    def set_lock(self, model, on):
        self.is_locked = on

    def release_now(self):
        if self.is_locked or not self.held:
            return 0, False
        n, self.held = self.held, 0
        self.gpu.used -= n
        self.released += 1
        return n, True


@pytest.fixture
def gdir(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "GPU_DIR", str(tmp_path))
    return tmp_path


def _serve(make):
    """A peer: an HTTP server of the admin routes and its share (make(port))."""
    httpd = H._Server(("127.0.0.1", 0), H._Handler)
    httpd.backend, httpd.quiet, httpd.api_key, httpd.cors_origin, httpd.max_body = None, True, KEY, None, 1 << 20
    httpd.gpu_share = make(httpd.server_address[1])
    httpd.gpu_share.write(force=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _post(httpd, path, body=None):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (httpd.server_address[1], path), method="POST",
                                 data=json.dumps(body or {}).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _asker(gpu, **kw):
    """The server short of memory: a Residency on the fake GPU with a share of its own (no HTTP)."""
    share = G.Share(port=1, api_key=KEY, quiet=True, **kw)
    res = R.Residency(mem_info=gpu.info, reserve=0, budget=None, always=False, enabled=True)
    res.peers = share
    return res, share


class Need:
    """A runner of the asker that needs nbytes."""

    def __init__(self, gpu, nbytes):
        self.gpu, self.nbytes, self.held, self.label = gpu, nbytes, 0, "need"

    def gpu_bytes(self):
        return self.held

    def load(self):
        if self.gpu.used + self.nbytes > self.gpu.total:
            raise MemoryError("cudaMalloc failed (the fake card is full)")
        self.gpu.used += self.nbytes
        self.held = self.nbytes

    def release(self, keep_host=False):
        self.gpu.used -= self.held
        self.held = 0


def test_peer_file_and_status(gdir):
    gpu = FakeGPU(100)
    s = _serve(lambda port: FakeShare(gpu, 30, port=port, api_key=KEY, quiet=True))
    sh = s.gpu_share
    p = json.load(open(os.path.join(gdir, "%d.json" % sh.port)))
    assert p["bytes"] == 30 and p["locked"] == 0 and p["pid"] == os.getpid() and not p["busy"]
    # another server sees it; a file of a dead pid is ignored
    json.dump(dict(p, pid=2 ** 22 + 12345, port=9), open(os.path.join(gdir, "9.json"), "w"))
    other = G.Share(port=2, quiet=True)
    seen = [q["port"] for q in other.peers()]
    assert seen == [sh.port] or seen == []          # (pid == ours is skipped by peers(): same process)
    st = urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:%d/v1/admin/gpu" % sh.port, headers={"Authorization": "Bearer " + KEY}), timeout=10)
    assert json.loads(st.read())["bytes"] == 30
    sh.close()
    assert not os.path.exists(os.path.join(gdir, "%d.json" % sh.port))
    s.shutdown()


def test_release_lock_routes(gdir):
    gpu = FakeGPU(100)
    s = _serve(lambda port: FakeShare(gpu, 40, locked=True, port=port, api_key=KEY, quiet=True))
    a = _post(s, "/v1/admin/release")
    assert a["status"] == "locked" and a["freed"] == 0 and a["locked"] == ["m%d" % s.gpu_share.port]
    b = _post(s, "/v1/admin/lock", {"on": False})
    assert b["models"] == {"m%d" % s.gpu_share.port: False}
    c = _post(s, "/v1/admin/release")
    assert c["status"] == "released" and c["freed"] == 40 and c["context"]
    assert _post(s, "/v1/admin/release")["status"] == "idle"
    # busy: a GPU job holds the gate
    s.gpu_share.held, gpu.used = 10, gpu.used + 10
    with s.gpu_share.gate:
        assert _post(s, "/v1/admin/release")["status"] == "busy"
    assert s.gpu_share.held == 10
    s.shutdown()


def _peers(gdir, gpu, specs):
    """Fake peers in-process: their peer files carry another pid (ours is skipped by peers())."""
    out = []
    for nbytes, locked, last_use in specs:
        srv = _serve(lambda port: FakeShare(gpu, nbytes, locked=locked, port=port, api_key=KEY, quiet=True))
        srv.gpu_share.last_use = last_use
        out.append(srv)
    return out


def _as_others(monkeypatch):
    """The peers' files as another process's (peers() skips its own pid)."""
    real = G.Share.state
    monkeypatch.setattr(G.Share, "state", lambda self: dict(real(self), pid=1))
    monkeypatch.setattr(G, "_alive", lambda pid: True)


def test_room_lru_and_hold(gdir, monkeypatch):
    _as_others(monkeypatch)
    gpu = FakeGPU(100)
    now = time.time()
    # the oldest goes first; the one used just now (within the hold) only if nothing else is left
    a, b, c = _peers(gdir, gpu, [(30, False, now - 100), (30, False, now - 500), (30, False, now)])
    for p in (a, b, c):
        p.gpu_share.write(force=True)
    res, share = _asker(gpu, hold=30)
    r = Need(gpu, 45)                               # (10 free: two peers' 30)
    res.register(r)
    res.load(r, r.nbytes, r.load)
    assert r.held == 45
    assert (b.gpu_share.released, a.gpu_share.released, c.gpu_share.released) == (1, 1, 0)
    assert share.arbiter.n == 0                    # (the arbiter let go after the load)
    for p in (a, b, c):
        p.shutdown()


def test_locked_fails_at_once(gdir, monkeypatch):
    _as_others(monkeypatch)
    gpu = FakeGPU(100)
    (a,) = _peers(gdir, gpu, [(70, True, 0)])
    a.gpu_share.write(force=True)
    res, share = _asker(gpu, wait=30)
    r = Need(gpu, 50)
    res.register(r)
    t0 = time.time()
    with pytest.raises(MemoryError, match="locked"):
        res.load(r, r.nbytes, r.load)
    assert time.time() - t0 < 5 and a.gpu_share.held == 70
    a.shutdown()


def test_busy_peer_waited_for(gdir, monkeypatch):
    _as_others(monkeypatch)
    gpu = FakeGPU(100)
    (a,) = _peers(gdir, gpu, [(80, False, 0)])
    a.gpu_share.gate.acquire()                      # in a GPU job
    a.gpu_share.write(force=True)
    threading.Timer(1.5, a.gpu_share.gate.release).start()
    res, share = _asker(gpu, wait=30)
    r = Need(gpu, 50)
    res.register(r)
    res.load(r, r.nbytes, r.load)
    assert r.held == 50 and a.gpu_share.released == 1
    a.shutdown()


def test_busy_too_long(gdir, monkeypatch):
    _as_others(monkeypatch)
    gpu = FakeGPU(100)
    (a,) = _peers(gdir, gpu, [(80, False, 0)])
    a.gpu_share.gate.acquire()
    a.gpu_share.write(force=True)
    res, share = _asker(gpu, wait=1.5)
    r = Need(gpu, 50)
    res.register(r)
    with pytest.raises(MemoryError, match="fake card"):         # (the build tried, and failed)
        res.load(r, r.nbytes, r.load)
    assert a.gpu_share.held == 80 and share.arbiter.n == 0
    a.gpu_share.gate.release()
    a.shutdown()


def test_remake_keeps_the_host_half():
    from reclank.backends.cuda.runtime import remake

    class Dev:
        def __init__(self, host):
            self.host = host                # the host half
            self._device_state()
            self._host_keys = set(self.__dict__)
            self._device_init()

        def _device_state(self):
            self.progs = {}

        def _device_init(self):
            self.made = getattr(self, "made", 0) + 1
            self.progs["step"] = object()

    d = Dev([1, 2, 3])
    d.later = "a device table set after the load"
    old_prog = d.progs["step"]
    n = remake(d)
    assert type(n) is Dev and n is not d
    assert n.host is d.host and n.made == 1 and n.progs["step"] is not old_prog
    assert not hasattr(n, "later")
    m = remake(n)                               # (again: a second release and reload)
    assert m.host is d.host and m.progs["step"] is not n.progs["step"]


def test_estimate_short_builds_anyway(gdir, monkeypatch):
    # need is an estimate over what the card can give: the peer releases, the build tries and fits
    _as_others(monkeypatch)
    gpu = FakeGPU(100)
    (a,) = _peers(gdir, gpu, [(30, False, 0)])
    a.gpu_share.write(force=True)
    res, share = _asker(gpu)
    r = Need(gpu, 80)
    res.register(r)
    res.load(r, 110, r.load, sure=80)             # (110: the estimate; 80 the known part)
    assert r.held == 80 and a.gpu_share.released == 1 and share.arbiter.n == 0
    a.shutdown()


def test_reload_lanes():
    from reclank.models.base import reload_lanes

    class Store:
        def __init__(self, io, max_contexts=4, max_bytes=1e9, name="s"):
            self.io, self.max_contexts, self.max_bytes, self.name = io, max_contexts, max_bytes, name

    class Eng:
        pass

    a, b = Eng(), Eng()
    a.ctxs, b.ctxs = Store(a, name="A"), Store(b, name="B")
    a.lanes = b.lanes = [a, b]
    b.state = "old"
    seen = []

    def make():
        seen.append(a.ctxs.name)            # (the warm-up's store: a new, empty one)
        nb = Eng()
        nb.ctxs, nb.state = Store(nb, name="tmp B"), "new"
        a.lanes = nb.lanes = [a, nb]
    reload_lanes(a, make)
    assert seen == ["A"] and a.lanes[1] is b and b.state == "new" and b.lanes is a.lanes
    assert a.ctxs.name == "A" and b.ctxs.name == "B" and a.ctxs.io is a and b.ctxs.io is b
