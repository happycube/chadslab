"""The GPU shared between the servers (docs/GPU_SWAP_PLAN.md 3.6): each GPU server's peer file, its
admin routes and locks, and the asking of the others for room.

Each GPU server writes DIR/<port>.json (DIR: RECLANK_GPU_DIR, default ~/.cache/reclank/gpu): its
pid, url and kind, the device bytes it holds and the part of them locked, its last use and whether
it is in a GPU job; it keeps the file current (on a change, and every few seconds) and removes it
at exit. A file whose pid is gone is ignored.

The routes (the API key):

    GET  /v1/admin/gpu      the server's residency: bytes, locked bytes, each model's lock, busy
    POST /v1/admin/release  an idle server releases all that is not locked (its CUDA context too
                            when nothing is left): {"status": "released" | "locked" | "idle" |
                            "busy", "freed": bytes, "context": bool, "locked": [ids]}; a server in
                            a GPU job answers "busy" at once
    POST /v1/admin/lock     {"model": ID (none: all of the server's), "on": true | false}

A server short of memory (Residency.make_room, its own victims gone) asks the peers that are not
busy, the least recently used first (one used within --gpu-hold seconds only when nothing else is
left), until the free memory covers the need; busy ones are asked again every second up to
--gpu-wait seconds, then the request fails with the out-of-memory error. When the card less the
locked bytes of every server cannot hold the need, it fails at once and names the locks. Two
servers short at once take DIR/arbiter.lock (flock) around the asking and the load, so one
finishes its load first; while it waits for a busy peer it lets the lock go, and its own parts not
in use may go to a peer's request then.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

GPU_DIR = os.environ.get("RECLANK_GPU_DIR") or os.path.expanduser("~/.cache/reclank/gpu")
# the device memory of a CUDA context (a bare one is 258 MiB on the 3090), counted while it is up
CTX_BYTES = 300e6
BEAT = 5.0          # the peer file written again at least this often (s)
# a check: an LLM engine's device half (and the context) released before every generation
SWAP_ALWAYS = os.environ.get("RECLANK_SWAP_ALWAYS", "0") == "1"


def add_gpu_args(ap, models=True):
    """The options of the sharing: --gpu-peers, --gpu-wait, --gpu-hold, --gpu-lock (models: the
    servers of several models take ids)."""
    ap.add_argument("--gpu-peers", choices=("on", "off"), default="on",
                    help="share the GPU with the other reclank servers (default on): a peer file in "
                         "RECLANK_GPU_DIR (~/.cache/reclank/gpu), the admin routes, and room asked of "
                         "the peers when short")
    ap.add_argument("--gpu-wait", type=float, default=600.0, metavar="S",
                    help="how long a request waits for busy peers to give room (default %(default)s)")
    ap.add_argument("--gpu-hold", type=float, default=30.0, metavar="S",
                    help="a peer used within this many seconds is asked last (default %(default)s)")
    if models:
        ap.add_argument("--gpu-lock", "--pin", dest="pin", nargs="?", const="*", default="", metavar="ID,...",
                        help="lock these models on the GPU (all their parts; no ids: all of them): never "
                             "freed for room, neither this server's nor a peer's")
    else:
        ap.add_argument("--gpu-lock", dest="gpu_lock", action="store_true",
                        help="lock the model on the GPU: never released for a peer's room")


def _alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError):
        return True
    return True


class _Arbiter:
    """DIR/arbiter.lock: one server at a time asks its peers and loads (flock), reentrant in the
    process."""

    def __init__(self, path):
        self.path = path
        self.mu = threading.RLock()
        self.n = 0
        self.fd = None

    def acquire(self):
        self.mu.acquire()
        if self.n == 0:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        self.n += 1

    def release(self):
        self.n -= 1
        if self.n == 0:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
        self.mu.release()


class Share:
    """A server's part of the sharing. A kind of server gives: busy() (in a GPU job), bytes() and
    locked_bytes(), models() ({id: locked}), release_now() (all that is not locked: (bytes freed,
    context released); the gates held), gates() (the locks of its GPU jobs), and set_lock(id, on)."""

    kind = "server"

    def __init__(self, port, host="127.0.0.1", api_key=None, wait=600.0, hold=30.0, quiet=False):
        self.port = int(port)
        self.url = "http://%s:%d" % ("127.0.0.1" if host in ("0.0.0.0", "") else host, self.port)
        self.api_key = api_key or None
        self.wait = float(wait)
        self.hold = float(hold)
        self.quiet = quiet
        os.makedirs(GPU_DIR, mode=0o700, exist_ok=True)
        self.path = os.path.join(GPU_DIR, "%d.json" % self.port)
        self.arbiter = _Arbiter(os.path.join(GPU_DIR, "arbiter.lock"))
        self.waiting = False        # a request of this server waits for its peers (room)
        self.last_use = 0.0
        self.mu = threading.Lock()
        self._last = None
        self._stop = threading.Event()

    # ---- what a kind of server gives

    def gates(self):
        return []

    def busy(self):
        return any(g.locked() for g in self.gates())

    def bytes(self):
        return 0

    def locked_bytes(self):
        return 0

    def models(self):
        return {}

    def release_now(self):
        return 0, False

    def set_lock(self, model, on):
        raise ValueError("this server has no models to lock")

    # ---- the peer file

    def state(self):
        return {"pid": os.getpid(), "port": self.port, "url": self.url, "kind": self.kind,
                "bytes": int(self.bytes()), "locked": int(self.locked_bytes()), "models": self.models(),
                "last_use": self.last_use, "busy": bool(self.busy()), "waiting": self.waiting}

    def write(self, force=False):
        with self.mu:
            st = self.state()
            key = json.dumps(st, sort_keys=True)
            if not force and key == self._last:
                return
            tmp = self.path + ".%d.tmp" % os.getpid()
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(key)
            os.replace(tmp, self.path)
            self._last = key

    def start(self):
        """Write the peer file now and keep it current (a thread); remove it at exit."""
        import atexit
        self.write(force=True)
        atexit.register(self.close)

        def beat():
            n = 0
            while not self._stop.wait(1.0):
                n += 1
                try:
                    self.write(force=n % int(BEAT) == 0)
                except Exception as e:      # (a full disk: the next beat tries again)
                    print("gpu: the peer file: %s" % e, file=sys.stderr, flush=True)
        threading.Thread(target=beat, name="gpu-peer", daemon=True).start()
        return self

    def close(self):
        self._stop.set()
        try:
            with open(self.path) as f:
                if json.load(f).get("pid") == os.getpid():
                    os.unlink(self.path)
        except (OSError, ValueError):
            pass

    def used(self):
        self.last_use = time.time()

    # ---- the routes

    def status(self):
        return dict(self.state(), object="gpu")

    def release(self):
        """POST /v1/admin/release: all that is not locked, when no GPU job runs."""
        got = []
        for g in self.gates():
            if not g.acquire(blocking=False):
                for x in got:
                    x.release()
                # a request that waits for room runs nothing: its parts not in use may go
                freed = self.release_waiting() if self.waiting else 0
                return {"status": "busy", "freed": int(freed), "context": False, "locked": self.locked_ids()}
            got.append(g)
        try:
            freed, ctx = self.release_now()
        finally:
            for x in got:
                x.release()
        self.write()
        locked = self.locked_ids()
        status = "released" if freed or ctx else ("locked" if locked else "idle")
        if not self.quiet and (freed or ctx):
            print("gpu: released %.2f GB for a peer%s%s" % (
                freed / 1e9, ", the context too" if ctx else "",
                "; locked: %s" % ",".join(locked) if locked else ""), flush=True)
        return {"status": status, "freed": int(freed), "context": bool(ctx), "locked": locked}

    def release_waiting(self):
        return 0

    def locked_ids(self):
        return sorted(k for k, v in self.models().items() if v)

    def lock(self, body):
        """POST /v1/admin/lock {"model": ID or none (all), "on": bool}."""
        if not isinstance(body, dict):
            raise ValueError("the body must be a JSON object")
        on = body.get("on", True)
        if not isinstance(on, bool):
            raise ValueError("\"on\" must be true or false")
        model = body.get("model")
        ids = list(self.models()) if model in (None, "", "*") else [model]
        for m in ids:
            if m not in self.models():
                raise ValueError("no model %s (the models: %s)" % (m, ", ".join(self.models()) or "none"))
            self.set_lock(m, on)
        self.write()
        return {"object": "gpu.lock", "models": self.models()}

    # ---- asking the peers

    def peers(self):
        """The other servers' states (their pids alive)."""
        out = []
        try:
            names = os.listdir(GPU_DIR)
        except OSError:
            return out
        for n in names:
            if not n.endswith(".json") or n == "%d.json" % self.port:
                continue
            try:
                with open(os.path.join(GPU_DIR, n)) as f:
                    p = json.load(f)
            except (OSError, ValueError):
                continue
            if p.get("pid") != os.getpid() and _alive(p.get("pid")):
                out.append(p)
        return out

    def ask(self, peer, timeout=120.0):
        """POST the peer's /v1/admin/release; its answer (None when it does not answer)."""
        req = urllib.request.Request(peer["url"] + "/v1/admin/release", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            if not self.quiet:
                print("gpu: the peer %s did not answer (%s)" % (peer.get("url"), e), flush=True)
            return None

    def room(self, short, need, label, total, reserve, own_locked=0, sure=None):
        """Ask the peers until short() (the bytes missing) is <= 0, or none has more to give, or
        busy ones stayed busy for --gpu-wait seconds: the build then tries (need may be an estimate;
        a build short of memory fails as before). Return with the arbiter held (done() lets it go after
        the load). MemoryError at once when the card less the locked bytes cannot hold sure (the part
        of need that is known; None: all of it)."""
        sure = need if sure is None else sure
        deadline = time.time() + self.wait
        self.note = None            # (the locks that kept memory, for the build's out-of-memory error)
        self.waiting = True
        self.write()
        try:
            while True:
                self.arbiter.acquire()
                held = False
                try:
                    if short() <= 0:
                        held = True
                        return
                    peers = self.peers()
                    locked = own_locked + sum(int(p.get("locked", 0)) for p in peers)
                    names = ["%s (%s, %.1f GB)" % (p.get("url"), ",".join(k for k, v in p.get("models", {}).items()
                                                                        if v) or "all", p.get("locked", 0) / 1e9)
                             for p in peers if p.get("locked")]
                    if total - locked - reserve < sure:
                        raise MemoryError("%s needs %.2f GB: the GPU's %.2f GB less the locked %.2f GB cannot hold it "
                                          "(locked: %s)" % (label, sure / 1e9, total / 1e9, locked / 1e9,
                                                            "; ".join(names) or "this server's"))
                    now = time.time()
                    order = sorted((p for p in peers if int(p.get("bytes", 0)) > int(p.get("locked", 0))),
                                   key=lambda p: (now - float(p.get("last_use", 0)) < self.hold,
                                                  float(p.get("last_use", 0))))
                    busy = False
                    for p in order:
                        t0 = time.time()
                        ans = self.ask(p)
                        if ans is None:
                            continue
                        if ans.get("status") == "busy":
                            busy = True
                        elif not self.quiet:
                            print("gpu: %s for %s: %s %.2f GB in %.2f s%s" % (
                                p.get("url"), label, ans.get("status"), ans.get("freed", 0) / 1e9,
                                time.time() - t0, ", its context too" if ans.get("context") else ""), flush=True)
                        if short() <= 0:
                            held = True
                            return
                    if not busy or time.time() > deadline:
                        if not self.quiet:
                            print("gpu: %s is %.2f GB short by its estimate: the peers %s; the build tries%s" % (
                                label, short() / 1e9, "stayed busy (--gpu-wait)" if busy else "have nothing more to give",
                                " (locked: %s)" % "; ".join(names) if names else ""), flush=True)
                        self.note = "the peers' locks: %s" % "; ".join(names) if names else None
                        held = True
                        return
                finally:
                    if not held:
                        self.arbiter.release()
                time.sleep(1.0)
        finally:
            self.waiting = False
            self.write()

    def clear(self, label):
        """Before a load whose plan takes the free memory (an LLM's): every peer releases what is
        not locked; busy ones are asked again every second up to --gpu-wait. The arbiter stays held
        (done() after the load)."""
        self.arbiter.acquire()
        deadline = time.time() + self.wait
        while True:
            busy = False
            for p in self.peers():
                if int(p.get("bytes", 0)) <= int(p.get("locked", 0)):
                    continue
                t0 = time.time()
                ans = self.ask(p)
                if ans is None:
                    continue
                if ans.get("status") == "busy":
                    busy = True
                elif not self.quiet:
                    print("gpu: %s for %s: %s %.2f GB in %.2f s%s" % (
                        p.get("url"), label, ans.get("status"), ans.get("freed", 0) / 1e9, time.time() - t0,
                        ", its context too" if ans.get("context") else ""), flush=True)
            if not busy or time.time() > deadline:
                return
            time.sleep(1.0)

    def done(self):
        """After the load that room() made room for: the arbiter goes."""
        self.arbiter.release()


class RunnerShare(Share):
    """The servers of _Runners (generation, embeddings): RESIDENCY holds the parts; models
    ({id: the model, its runners()}), locks (ids, or "*"), the gates of the GPU jobs."""

    def __init__(self, kind, models, gates, locks=(), **kw):
        super().__init__(**kw)
        self.kind = kind
        self._models = models           # callable: {id: model}
        self._gates = gates             # callable: [threading.Lock]
        self.locks = set(locks)

    def gates(self):
        return self._gates()

    def _runners(self, mid):
        m = self._models().get(mid)
        return list(m.runners().values()) if m is not None else []

    def is_locked(self, mid):
        return "*" in self.locks or mid in self.locks

    def models(self):
        return {mid: self.is_locked(mid) for mid in self._models()}

    def context_up(self):
        from reclank.backends.cuda import gpumm
        return gpumm._UP[0] == gpumm._EPOCH[0]

    def bytes(self):
        """The runners' device bytes, and the context's while it is up (a release frees it too)."""
        from reclank.encoders.runtime import RESIDENCY
        return RESIDENCY.total() + (CTX_BYTES if self.context_up() else 0)

    def locked_bytes(self):
        """The locked models' bytes (and the context, which stays with them)."""
        seen, n = set(), 0
        for mid in self._models():
            if self.is_locked(mid):
                for r in self._runners(mid):
                    if id(r) not in seen:
                        seen.add(id(r))
                        n += r.gpu_bytes()
        return n + (CTX_BYTES if n else 0)

    def apply(self):
        """Pin the runners of the locked models in RESIDENCY (again after new ones are made)."""
        from reclank.encoders.runtime import RESIDENCY
        for mid in self._models():
            for r in self._runners(mid):
                if self.is_locked(mid):
                    RESIDENCY.pin(r)

    def set_lock(self, mid, on):
        from reclank.encoders.runtime import RESIDENCY
        if "*" in self.locks and not on:
            self.locks.update(self._models())
        self.locks.discard("*")
        (self.locks.add if on else self.locks.discard)(mid)
        if on:
            self.apply()
        else:
            for r in self._runners(mid):
                RESIDENCY.unpin(r)

    def release_now(self):
        from reclank.encoders.runtime import RESIDENCY
        return RESIDENCY.release_all(context=True)

    def release_waiting(self):
        from reclank.encoders.runtime import RESIDENCY
        return RESIDENCY.evict_unpinned()


def process_bytes(pid=None):
    """The device memory of a process by the driver (nvidia-smi: its context too), or None."""
    import subprocess
    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 2 and f[0] == str(pid) and f[1].isdigit():
            return int(f[1]) << 20
    return 0


class EngineShare(Share):
    """An LLM server: one engine on the model thread (worker). Its device half goes and comes back
    whole (engine.release_device / reload_device: docs/GPU_SWAP_PLAN.md 3.6) when the engine has
    them and is not locked (--gpu-lock, the route); an engine without them keeps its memory (all of
    it locked)."""

    kind = "llm"

    def __init__(self, backend, worker, locked=False, **kw):
        super().__init__(**kw)
        self.backend = backend
        self.worker = worker
        self.locked = bool(locked)
        self.held = process_bytes() or 0       # (measured after the load; again after each change)
        self.released = False

    def engine(self):
        return self.worker.engine

    def can_release(self):
        eng = self.engine()
        ok = getattr(eng, "can_release_device", None)
        return hasattr(eng, "release_device") and not self.locked and (ok is None or bool(ok()))

    def busy(self):
        return self.worker.busy()

    def bytes(self):
        return 0 if self.released else self.held

    def locked_bytes(self):
        return 0 if self.can_release() else self.bytes()

    def models(self):
        return {self.backend.model_id: not self.can_release()}

    def set_lock(self, model, on):
        self.locked = bool(on)

    def start(self):
        self.worker.before = self.ensure
        return super().start()

    def release(self):
        if self.busy():
            return {"status": "busy", "freed": 0, "context": False, "locked": self.locked_ids()}
        if not self.can_release():
            return {"status": "locked", "freed": 0, "context": False, "locked": self.locked_ids()}
        if self.released:
            return {"status": "idle", "freed": 0, "context": False, "locked": []}
        before = self.held
        t0 = time.time()
        ctx = self.worker.call(self._release, device=False)
        if not self.quiet:
            print("gpu: released %.2f GB for a peer in %.2f s%s" % (
                before / 1e9, time.time() - t0, ", the context too" if ctx else ""), flush=True)
        return {"status": "released", "freed": int(before), "context": bool(ctx), "locked": []}

    def _release(self):
        """On the model thread: the engine's device half, then the CUDA context (it stays while
        anything of the process holds device memory: the reason printed)."""
        from reclank.backends.cuda import gpumm
        from reclank.backends.cuda.runtime import close_all
        self.engine().release_device()
        self.released = self.worker.released = True
        progs, blocks = close_all()     # (the engine's device objects, forgotten: freed here)
        if (progs or blocks) and not self.quiet:
            print("gpu: %d programs and %d blocks freed" % (progs, blocks), flush=True)
        try:
            gpumm.context_release()
            ctx = True
        except RuntimeError as e:
            print("gpu: the context stays (%s)" % e, flush=True)
            ctx = False
        self.write()
        return ctx

    def ensure(self, generation=False):
        """On the model thread before a job: after a release, room for the device half (the
        peers asked) and the device half again; its time printed. RECLANK_SWAP_ALWAYS=1 (a check):
        a release (the context too) before every generation."""
        self.used()
        if generation and SWAP_ALWAYS and not self.released and self.can_release():
            t0 = time.time()
            ctx = self._release()
            if not self.quiet:
                print("gpu: released %.2f GB (RECLANK_SWAP_ALWAYS) in %.2f s%s" % (
                    self.held / 1e9, time.time() - t0, ", the context too" if ctx else ""), flush=True)
        if not self.released:
            return
        from reclank.backends.cuda.gpumm import mem_info
        from reclank.encoders.runtime import RESIDENCY
        need, label = self.held, self.backend.model_id
        reserve = RESIDENCY.reserve
        t0 = time.time()
        total = mem_info()[1]
        self.room(lambda: need + reserve - mem_info()[0], need, label, total, reserve)
        try:
            self.engine().reload_device()
        finally:
            self.done()
        self.released = self.worker.released = False
        self.held = process_bytes() or need
        self.used()
        self.write()
        if not self.quiet:
            print("gpu: %s back on the GPU (%.2f GB) in %.2f s" % (label, self.held / 1e9, time.time() - t0), flush=True)


def attach(share):
    """Make share the peers of RESIDENCY (its make_room asks them) and start its peer file."""
    from reclank.encoders.runtime import RESIDENCY
    RESIDENCY.peers = share
    return share.start()
