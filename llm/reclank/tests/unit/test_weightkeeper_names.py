"""Several weightkeepers (reclank/runtime/weightkeeper.py: --name, --list, the memory check, the root
keepers' shared state) on small checkpoints of the shape of DeepSeek V4.1: one on the default socket
and one named, side by side, each found by its checkpoint, stopped on its own. The sockets in a
directory of the test (RECLANK_WEIGHTKEEPER_DIR)."""
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from reclank.runtime import weightkeeper as Z

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
WK = os.path.join(ROOT, "scripts", "weightkeeper.py")


def _write_shard(path, tensors):
    hdr, off, blobs = {}, 0, []
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _dsv41(d, seed):
    rng = np.random.default_rng(seed)
    t = {}
    for e in range(2):
        p = "layers.0.ffn.experts.%d." % e
        for w, (r, c) in (("w1", (64, 128)), ("w3", (64, 128)), ("w2", (128, 64))):
            t[p + w + ".weight"] = ("I8", rng.integers(-128, 128, (r, c // 2), dtype=np.int8))
            t[p + w + ".scale"] = ("F8_E8M0", rng.integers(0, 256, (r, c // 32), dtype=np.uint8))
    _write_shard(d / "model-00001-of-00001.safetensors", t)
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in t}},
              open(d / "model.safetensors.index.json", "w"))
    json.dump({"text_config": {"num_hidden_layers": 1, "n_routed_experts": 2, "hidden_size": 128,
                               "moe_intermediate_size": 64}}, open(d / "config.json", "w"))
    return str(d)


@pytest.fixture(scope="module")
def dsv41(tmp_path_factory):
    return _dsv41(tmp_path_factory.mktemp("dsv41"), 0)


@pytest.fixture(scope="module")
def dsv41b(tmp_path_factory):
    return _dsv41(tmp_path_factory.mktemp("dsv41b"), 1)


@pytest.fixture
def sockdir(tmp_path, monkeypatch):
    d = tmp_path / "socks"
    d.mkdir()
    monkeypatch.setenv("RECLANK_WEIGHTKEEPER_DIR", str(d))
    monkeypatch.delenv("RECLANK_WEIGHTKEEPER", raising=False)
    return str(d)


def test_names(sockdir):
    uid = os.getuid()
    d = os.path.join(sockdir, "reclank-weightkeeper-%d.sock" % uid)
    assert Z.default_socket() == d and Z.socket_path() == d and Z.socket_path(None) == d
    p = Z.socket_path("wan-a14b")
    assert p == os.path.join(sockdir, "reclank-weightkeeper-%d-wan-a14b.sock" % uid)
    assert Z.name_of(p) == "wan-a14b" and Z.name_of(d) is None
    for bad in ("", "../x", "a/b", "-x", "x" * 65, "a b"):
        if bad:
            with pytest.raises(ValueError):
                Z.socket_path(bad)
    assert Z.sockets() == [] and Z.keepers() == []


def test_check_memory():
    Z.check_memory({0: 1 << 20}, numa_on=False)
    with pytest.raises(SystemExit) as e:
        Z.check_memory({0: 1 << 62}, numa_on=False)
    assert "not enough memory" in str(e.value)


def _stop(p, sock):
    """A keeper the test started: stop it on its socket, else kill it (never left behind)."""
    try:
        Z.request("stop", sock).close()
    except OSError:
        pass
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def _start(args, sock, env, started):
    """A keeper on sock (in started at once, so the test's teardown stops it whatever happens)."""
    p = subprocess.Popen([sys.executable, WK] + args, env=env, stderr=subprocess.PIPE, text=True)
    started.append((p, sock))
    for _ in range(1200):
        if os.path.exists(sock) or p.poll() is not None:
            break
        time.sleep(0.1)
    assert p.poll() is None, p.stderr.read()
    assert os.path.exists(sock), "the keeper did not serve within 120 s"
    return p


@pytest.fixture
def keepers():
    """The keepers a test starts: stopped (or killed) at its end."""
    started = []
    yield started
    for p, sock in started:
        if p.poll() is None:
            _stop(p, sock)


def test_two_keepers(dsv41, dsv41b, sockdir, keepers):
    from reclank.backends.cpu import cops
    assert cops.KQ_F20      # (the C library built here before the keepers start)
    env = dict(os.environ, PYTHONPATH=ROOT)
    a = _start([dsv41, "--pages", "4k", "--threads", "2"], Z.socket_path(), env, keepers)
    b = _start([dsv41b, "--pages", "4k", "--threads", "2", "--name", "b"], Z.socket_path("b"), env, keepers)
    ks = {name: st for name, _p, st in Z.keepers()}
    assert set(ks) == {None, "b"} and ks["b"]["checkpoint"] == os.path.abspath(dsv41b)
    out = subprocess.run([sys.executable, WK, "--list"], env=env, capture_output=True, text=True).stdout
    assert "(default)" in out and "b " in out and dsv41b in out
    # a second keeper on a socket that answers: refused before it loads
    r = subprocess.run([sys.executable, WK, dsv41, "--pages", "4k", "--name", "b"], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "answers on" in r.stderr
    # each found by its checkpoint; the default socket still the default keeper
    assert Z.find(dsv41)[0] == Z.socket_path() and Z.find(dsv41b)[0] == Z.socket_path("b")
    Z.attach(ckpt=dsv41).check(dsv41)
    Z.attach(ckpt=dsv41b).check(dsv41b)
    c = Z.request("status")
    with c:
        assert json.loads(Z._recv_msg(c))["checkpoint"] == os.path.abspath(dsv41)
    os.environ["RECLANK_WEIGHTKEEPER"] = "b"
    try:
        c = Z.request("status")
        with c:
            assert json.loads(Z._recv_msg(c))["checkpoint"] == os.path.abspath(dsv41b)
        assert Z.find(dsv41) == (None, None) and Z.find(dsv41b)[0] == Z.socket_path("b")
    finally:
        del os.environ["RECLANK_WEIGHTKEEPER"]
    # --check finds the keeper of its checkpoint (no --name)
    r = subprocess.run([sys.executable, WK, "--check", dsv41b, "--check-n", "4"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "all equal" in r.stdout, r.stdout + r.stderr
    # stopped on its own: the other answers
    r = subprocess.run([sys.executable, WK, "--stop", "--name", "b"], env=env, capture_output=True, text=True)
    assert "stopping" in r.stdout
    b.wait(timeout=30)
    assert [n for n, _p, _s in Z.keepers()] == [None]
    subprocess.run([sys.executable, WK, "--stop"], env=env, capture_output=True, text=True)
    a.wait(timeout=30)
    assert Z.keepers() == []


def test_root_shared_state(sockdir, tmp_path, monkeypatch):
    """Two root keepers: the pool pages each added go back alone; the settings go back with the
    last one, to the values before the first (the parents' logic, without root: files of the test)."""
    pool = tmp_path / "hugepages-1048576kB"
    pool.mkdir()
    # 10 pages before the keepers, keeper A added 4, then keeper B 6; A exits: B's 6 stay
    (pool / "nr_hugepages").write_text("20")
    Z.give_back_pools({str(pool): 4})
    assert int((pool / "nr_hugepages").read_text()) == 16
    Z.give_back_pools({str(pool): 6})
    assert int((pool / "nr_hugepages").read_text()) == 10
    restored = []
    monkeypatch.setattr(Z, "restore_settings", lambda orig: restored.append(orig))
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        uid = os.getuid()
        st = Z.settings_join(uid, {"shmem": "never", "compact": 1}, other.pid)
        assert st["orig"] == {"shmem": "never", "compact": 1}
        # the second sees the first's changes as now: the first's record stays
        st = Z.settings_join(uid, {"shmem": "advise", "compact": 0}, os.getpid())
        assert st["orig"] == {"shmem": "never", "compact": 1} and len(st["pids"]) == 2
        assert not Z.settings_leave(uid, os.getpid()) and restored == []
        assert Z.settings_leave(uid, other.pid) and restored == [{"shmem": "never", "compact": 1}]
        assert not os.path.exists(Z._settings_path(uid))
        # a holder that died (no leave): the next keeper is the first again
        Z.settings_join(uid, {"shmem": "never", "compact": 1}, 2 ** 22 + 12345)
        st = Z.settings_join(uid, {"shmem": "advise", "compact": 0}, os.getpid())
        assert st["orig"] == {"shmem": "advise", "compact": 0} and st["pids"] == [os.getpid()]
    finally:
        other.kill()
        other.wait()
