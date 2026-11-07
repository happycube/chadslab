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
    t["layers.0.attn.wq.weight"] = ("BF16", rng.integers(0, 1 << 16, (16, 128), dtype=np.uint16))   # (a dense one)
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


def _wk(args, env):
    return subprocess.run([sys.executable, WK] + args, env=env, capture_output=True, text=True)


def test_catalog(dsv41, dsv41b, sockdir, keepers):
    """One keeper, two entries (docs/GPU_SWAP_PLAN.md 3.7, step 1): an empty catalog (--serve), each
    checkpoint added in the background (--add) and found by its checkpoint; the same load again its
    entry; a drop: the entry gone from the catalog, a round that mapped it reads it still."""
    from reclank.backends.cpu import cops
    assert cops.KQ_F20      # (the C library built here before the keeper starts)
    env = dict(os.environ, PYTHONPATH=ROOT)
    _start(["--serve", "--threads", "2"], Z.socket_path(), env, keepers)
    st = json.loads(Z._recv_msg(Z.request("status")))
    assert st["checkpoint"] is None and st["entries"] == []
    ids = []
    for ck in (dsv41, dsv41b):
        r = _wk(["--add", ck, "--pages", "4k", "--threads", "2"], env)
        assert r.returncode == 0, r.stdout + r.stderr
        ids.append(json.loads(r.stdout)["id"])
    assert ids[0] != ids[1] and all(i.startswith("deepseek_v41-") for i in ids)
    for _ in range(600):
        ents = json.loads(Z._recv_msg(Z.request("list")))
        if [e.get("status") for e in ents] == [None, None]:
            break
        assert not any(e.get("status") == "failed" for e in ents), ents
        time.sleep(0.1)
    assert [e["id"] for e in ents] == ids
    # the same load again: its entry, no second copy
    r = _wk(["--add", dsv41, "--pages", "4k", "--threads", "2"], env)
    assert json.loads(r.stdout) == {"id": ids[0], "status": "ready"}
    # each found by its checkpoint (one socket), mapped and checked against its shards
    assert Z.find(dsv41) == (Z.socket_path(), Z.find(dsv41)[1]) and Z.find(dsv41)[1]["id"] == ids[0]
    assert Z.find(dsv41b)[1]["id"] == ids[1]
    wa, wb = Z.attach(ckpt=dsv41), Z.attach(ckpt=dsv41b)
    wa.check(dsv41)
    wb.check(dsv41b)
    assert wa.index["checkpoint"] != wb.index["checkpoint"]
    # the first entry answers a plain "get" and "status" (the servers before the catalog)
    assert Z.attach().index["checkpoint"] == os.path.abspath(dsv41)
    out = _wk(["--list"], env).stdout
    assert ids[0] in out and ids[1] in out
    r = _wk(["--status", ids[1]], env)
    assert json.loads(r.stdout)["checkpoint"] == os.path.abspath(dsv41b)
    # dropped: gone from the catalog; the round's map reads the same bytes
    before = wb.mats(0, 0)["w1"].copy()
    r = _wk(["--drop", ids[1]], env)
    assert json.loads(r.stdout) == {"dropped": ids[1], "mapped_by": [os.getpid()]}
    assert Z.find(dsv41b) == (None, None) and Z.find(dsv41)[1]["id"] == ids[0]
    assert "error" in json.loads(Z._recv_msg(Z.request("get " + ids[1])))
    assert np.array_equal(wb.mats(0, 0)["w1"], before)
    assert json.loads(_wk(["--drop", ids[1]], env).stdout) == {"dropped": None}


@pytest.fixture(scope="module")
def dsv41c(tmp_path_factory):
    return _dsv41(tmp_path_factory.mktemp("dsv41c"), 2)


def test_leases(dsv41, dsv41b, dsv41c, sockdir, keepers):
    """Leases and the host budget (docs/GPU_SWAP_PLAN.md 3.7, step 2): a budget for two entries of
    three; a round's get its lease until it ends it or exits; a load over the budget drops the least
    recently leased entry with no lease, never a leased or a kept one (the load fails then)."""
    from reclank.backends.cpu import cops
    assert cops.KQ_F20
    env = dict(os.environ, PYTHONPATH=ROOT)
    n = sum(Z.Entry(Z.Options(ckpt=dsv41, maxpage=Z.K4)).need().values())
    _start(["--serve", "--threads", "2", "--host-budget", "%.9f" % (2.5 * n / 1e9)], Z.socket_path(), env, keepers)

    def ls():
        return {e["id"]: e for e in json.loads(Z._recv_msg(Z.request("list")))}

    def wait(pred):
        for _ in range(600):
            es = ls()
            if pred(es):
                return es
            time.sleep(0.05)
        raise AssertionError(es)

    def add(ck, *more):
        r = _wk(["--add", ck, "--pages", "4k", "--threads", "2"] + list(more), env)
        assert r.returncode == 0, r.stdout + r.stderr
        i = json.loads(r.stdout)["id"]
        return i, wait(lambda es: es.get(i, {}).get("status") != "loading")

    a, es = add(dsv41)
    b, es = add(dsv41b)
    assert [es[a].get("status"), es[b].get("status")] == [None, None]
    assert es[a]["leases"] == [] and es[a]["idle_s"] is not None
    # A leased, then its lease ended: A the more recently leased of the two
    wa = Z.attach(ckpt=dsv41)
    assert ls()[a]["leases"] == [os.getpid()] and ls()[a]["idle_s"] is None
    wa.end_lease()
    wait(lambda es: es[a]["leases"] == [])
    c, es = add(dsv41c)
    assert set(es) == {a, c}, es        # B dropped: the least recently leased
    # C leased, A kept: a load over the budget fails, both stay
    wc = Z.attach(ckpt=dsv41c)
    assert json.loads(_wk(["--keep", a], env).stdout) == {"kept": a}
    b, es = add(dsv41b)
    assert es[b]["status"] == "failed" and "leased or kept" in es[b]["error"] and es[a]["kept"]
    assert json.loads(_wk(["--unkeep", a], env).stdout) == {"unkept": a}
    b, es = add(dsv41b)
    assert set(es) == {b, c} and es[b].get("status") is None, es    # A dropped
    # B leased by another process, until it exits
    p = subprocess.Popen([sys.executable, "-c", "import sys; from reclank.runtime import weightkeeper as Z; "
                          "Z.attach(ckpt=sys.argv[1]); print('ok', flush=True); sys.stdin.read()", dsv41b],
                         env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "ok"
        assert ls()[b]["leases"] == [p.pid] and ls()[c]["leases"] == [os.getpid()]
        a, es = add(dsv41)
        assert es[a]["status"] == "failed" and set(es) == {a, b, c}
    finally:
        p.kill()
        p.wait()
    wait(lambda es: es[b]["leases"] == [])
    a, es = add(dsv41)
    assert set(es) == {a, c} and es[a].get("status") is None, es    # B dropped, C leased still
    # the round's maps of the dropped entries read as before
    wa.check(dsv41)
    assert np.array_equal(wc.mats(0, 0)["w1"], Z.attach(ckpt=dsv41c).mats(0, 0)["w1"])
    out = _wk(["--list"], env).stdout
    assert "leased by %d" % os.getpid() in out and "the host budget" in out


def test_save(dsv41, dsv41b, sockdir, keepers, tmp_path):
    """The converted-weights cache (docs/GPU_SWAP_PLAN.md 3.7, step 3): --save DIR writes an entry once
    (its expert rows and its dense segment); a second keeper reads it back instead of the checkpoint:
    the rows and the dense tensors those of the shards (--check); other options (another ID) read the
    checkpoint; a copy of other code is not read."""
    from reclank.backends.cpu import cops
    assert cops.KQ_F20
    env = dict(os.environ, PYTHONPATH=ROOT)
    save = str(tmp_path / "saved")
    opts = ["--pages", "4k", "--threads", "2", "--dense"]

    def entry(ck, *more):
        r = _wk(["--add", ck] + opts + list(more), env)
        assert r.returncode == 0, r.stdout + r.stderr
        i = json.loads(r.stdout)["id"]
        for _ in range(600):
            es = {e["id"]: e for e in json.loads(Z._recv_msg(Z.request("list")))}
            if es[i].get("status") != "loading":
                assert es[i].get("status") is None, es[i]
                return es[i]
            time.sleep(0.05)
        raise AssertionError(es)

    def keeper():
        p = _start(["--serve", "--threads", "2", "--save", save], Z.socket_path(), env, keepers)
        return p

    p = keeper()
    e = entry(dsv41)
    assert e["source"] == "checkpoint"
    dirs = [d for d in os.listdir(save) if not d.startswith(".")]
    assert len(dirs) == 1 and dirs[0].startswith(e["id"] + "-")
    files = sorted(os.listdir(os.path.join(save, dirs[0])))
    assert files == ["index.json", "seg0.bin", "seg1.bin", "seg2.bin"][:len(files)] and "seg1.bin" in files
    first = Z.attach(ckpt=dsv41)
    rows, dense = first.mats(0, 0)["w1"].copy(), first.dense(next(iter(first.index["dense"]["tensors"]))).copy()
    _stop(p, Z.socket_path())
    p = keeper()
    e2 = entry(dsv41)
    assert e2["id"] == e["id"] and e2["source"] == "saved"
    r = _wk(["--check", dsv41], env)
    assert r.returncode == 0 and "all equal" in r.stdout, r.stdout + r.stderr
    w = Z.attach(ckpt=dsv41)
    assert np.array_equal(w.mats(0, 0)["w1"], rows)
    assert np.array_equal(w.dense(next(iter(w.index["dense"]["tensors"]))), dense)
    # another checkpoint, other options: their own entries, from the checkpoint
    assert entry(dsv41b)["source"] == "checkpoint"
    assert entry(dsv41, "--layout", "planes")["source"] == "checkpoint"
    assert len([d for d in os.listdir(save) if not d.startswith(".")]) == 3
    _stop(p, Z.socket_path())
    # a copy of other code (its key renamed): not read, the checkpoint read (and saved again)
    os.rename(os.path.join(save, dirs[0]), os.path.join(save, e["id"] + "-00000000"))
    keeper()
    assert entry(dsv41)["source"] == "checkpoint"
    assert sorted(d for d in os.listdir(save) if d.startswith(e["id"] + "-")) == sorted(
        [e["id"] + "-00000000", dirs[0]])
