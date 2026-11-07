"""One port for every model: the gateway in front of the reclank servers.

    python -m server.gateway --config gateway.json [--port 8080] [--host 127.0.0.1]

A request names its model ("model" of the JSON body, or the field "model" of a multipart form);
the gateway sends it to the server of that model, starting the server first when it is not
running (the request waits for it). The servers are the usual ones (python -m server,
server.embed, server.generate, scripts/serve_*.py), each on a port of its own that only the
gateway uses; they share the GPU among themselves as before (server/gpushare.py: the peers ask
each other for room). Host memory is the gateway's: a server's "ram_gb" counts against
"host_ram_gb", and a server to start stops the least recently used idle ones until it fits.

The page (GET /) shows every server and model, the GPUs and the host memory, and acts on them:
start, stop, restart, the GPU's release, the locks of the models (a locked model is never freed:
its server is not stopped for room or when idle, and its GPU memory is locked on the server again
at each start), keep (the server never stopped), and which GPUs a server may use (or the CPU). A change of the
GPUs restarts a running server when it is idle (CUDA_VISIBLE_DEVICES, and the server's own
"gpu_args" / "cpu_args"). The page asks for the API key and keeps it in the browser.

The routes (the API key, as the servers'; GET /health and GET / without):

    GET  /v1/models                        every model of the config, its server and state
    GET  /gateway/state                    the servers, the GPUs, the host memory
    GET  /gateway/servers/{id}/log         the end of the server's log
    POST /v1/admin/release                 every running server's release (its GPU memory; as a
                                           server's own route: what a GPU job asks first)
    GET  /v1/admin/gpu                     the running servers' /v1/admin/gpu, by server
    POST /gateway/servers/{id}/{action}    start | stop | restart | release;
                                           lock {"model": ID (none: all), "on": bool} | unlock;
                                           keep {"on": bool}; gpus {"gpus": [0, 1] | "cpu"}
    any other path                         to the server of the request's model; a request
                                           with no model to the server of its path's kind
                                           (chat and completions: llm, embeddings: embed,
                                           images, audio, music, videos: generate), the
                                           default model of the kind first

The config (JSON):

    {"host_ram_gb": 340,              the host memory the servers may hold (default: 90% of it)
     "log_dir": "logs",               the servers' logs (relative: to the config's directory)
     "backend_key": "...",            the key the servers take (default: the gateway's)
     "gpu_dir": "...",                RECLANK_GPU_DIR of the servers (the peers' files)
     "default": {"llm": "model-id"},  the model of a request with none, by kind
     "room_wait": 600,                how long a start waits for busy servers to give room (s)
     "servers": [
       {"id": "dg", "kind": "llm",            llm | embed | generate | service
        "models": ["diffusiongemma-26b-a4b"], the ids it serves (none: asked of the server once
                                              it runs, and kept in the state file)
        "port": 8091,
        "cmd": ["python", "-m", "server", "-m", "/models/dg/dg.gguf", "--port", "{port}"],
        "cwd": "/path/to/reclank",            (default: the gateway's own tree)
        "env": {"OMP_NUM_THREADS": "1"},
        "ram_gb": 45,                         its host memory when loaded
        "gpus": "all",                        the default: "all", [0, 1], or "cpu"
        "cpu": false,                         whether it can run with no GPU (default false)
        "gpu_args": [], "cpu_args": [],       added to cmd with GPUs / on the CPU
        "gpu_env": {}, "cpu_env": {},         added to env with GPUs / on the CPU
        "start_timeout": 900,                 seconds to wait for its /health
        "idle_stop": 0,                       stop it after this many idle seconds (0: never)
        "keep": false,                        never stop it for room or when idle
        "lock": false,                        true (all its models) or [ids]: locked models
        "requires": ["keeper"]}]}             services started first (and kept while it runs)

A service (kind "service": a weightkeeper, say) serves no requests: it is ready when its
"ready_cmd" exits 0 (a command run every few seconds), "stop_cmd" stops it (else SIGTERM), and it
goes for room only with the last running server that requires it (or when none does).

The page's changes (the GPUs, keep, the locks) and the models asked of the servers are kept in
CONFIG.state.json beside the config.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from server import http as H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway.html")
KINDS = ("llm", "embed", "generate")
# the kind of a path, for a request that names no model
PATH_KINDS = (("/chat/completions", "llm"), ("/completions", "llm"), ("/embeddings", "embed"),
              ("/images/", "generate"), ("/audio/", "generate"), ("/music/", "generate"),
              ("/videos", "generate"))
# the headers not passed on (hop by hop, and the ones the gateway sets)
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
       "transfer-encoding", "upgrade", "host", "authorization", "x-api-key", "content-length", "server", "date"}


def path_kind(path):
    """The kind of server of a path (llm, embed, generate), or None."""
    for part, kind in PATH_KINDS:
        if part in path:
            return kind
    return None


def request_model(body, content_type):
    """The model a request names: "model" of a JSON body, or the field "model" of a multipart
    form; None when it names none."""
    ct = (content_type or "").lower()
    if not body:
        return None
    if ct.startswith("multipart/form-data"):
        m = re.search(r'boundary="?([^";]+)"?', content_type or "")
        if not m:
            return None
        sep = b"--" + m.group(1).encode("latin-1")
        for part in body.split(sep):
            head, _, value = part.partition(b"\r\n\r\n")
            if re.search(rb'name="model"(;|\r\n|$)', head, re.I) and b"filename=" not in head.lower():
                return value.rstrip(b"\r\n").decode("utf-8", "replace").strip() or None
        return None
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    v = obj.get("model") if isinstance(obj, dict) else None
    return str(v) if v else None


RUNNING = ("starting", "ready", "stopping")


def room_victims(servers, need_gb, budget_gb, exclude=None):
    """The servers to stop so that need_gb more fits in budget_gb: the ready ones that are not held
    (kept or locked), have no request and are not exclude, the least recently used first; a service
    goes after the last running server that requires it (never one that exclude requires). None
    when even all of them are not enough."""
    running = [s for s in servers if s.state in RUNNING]
    held = sum(s.ram_gb for s in running)
    if held + need_gb <= budget_gb:
        return []
    needed = set(getattr(exclude, "requires", None) or ())
    gone, out = set(), []

    def unused_services():
        nonlocal held
        for v in servers:
            if (v.kind == "service" and v.state == "ready" and v.id not in gone and v.id not in needed
                    and not v.held and not any(v.id in d.requires and d.id not in gone for d in running)):
                out.append(v)
                gone.add(v.id)
                held -= v.ram_gb

    unused_services()
    for s in sorted(servers, key=lambda s: s.last_used):
        if held + need_gb <= budget_gb:
            break
        if s is exclude or s.kind == "service" or s.held or s.inflight or s.state != "ready":
            continue
        out.append(s)
        gone.add(s.id)
        held -= s.ram_gb
        unused_services()
    return out if held + need_gb <= budget_gb else None


def host_ram_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1 << 20)
    return 0.0


def gpu_list():
    """The GPUs (nvidia-smi): index, name, memory used and total (MiB), utilization (%)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        f = [v.strip() for v in line.split(",")]
        if len(f) == 5:
            num = lambda v: int(v) if v.isdigit() else None   # noqa: E731
            gpus.append({"index": int(f[0]), "name": f[1], "used_mib": num(f[2]), "total_mib": num(f[3]),
                         "util": num(f[4])})
    return gpus


class Server:
    """A server of the config: its process, its state (stopped, starting, ready, stopping,
    failed), its requests in flight and its last use."""

    def __init__(self, cfg, gw):
        self.gw = gw
        self.id = str(cfg["id"])
        self.kind = cfg.get("kind", "llm")
        if self.kind not in KINDS + ("service",):
            raise ValueError("server %s: kind %r is not one of %s" % (self.id, self.kind, KINDS + ("service",)))
        self.models = [str(m) for m in cfg.get("models") or []]
        self.port = int(cfg.get("port") or 0)
        if not self.port and self.kind != "service":
            raise ValueError("server %s has no port" % self.id)
        self.requires = [str(v) for v in cfg.get("requires") or []]
        self.ready_cmd = [str(v) for v in cfg["ready_cmd"]] if cfg.get("ready_cmd") else None
        self.stop_cmd = [str(v) for v in cfg["stop_cmd"]] if cfg.get("stop_cmd") else None
        lk = cfg.get("lock", False)
        self.locked = {"*"} if lk is True else {str(m) for m in lk} if isinstance(lk, list) else set()
        self.cmd = [str(v) for v in cfg["cmd"]]
        self.cwd = cfg.get("cwd") or ROOT
        self.env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
        self.ram_gb = float(cfg.get("ram_gb", 0))
        self.cpu_ok = bool(cfg.get("cpu", False))
        self.gpu_args = [str(v) for v in cfg.get("gpu_args") or []]
        self.cpu_args = [str(v) for v in cfg.get("cpu_args") or []]
        self.gpu_env = {str(k): str(v) for k, v in (cfg.get("gpu_env") or {}).items()}
        self.cpu_env = {str(k): str(v) for k, v in (cfg.get("cpu_env") or {}).items()}
        self.start_timeout = float(cfg.get("start_timeout", 900))
        self.idle_stop = float(cfg.get("idle_stop", 0))
        self.gpus = cfg.get("gpus", "all")
        self.keep = bool(cfg.get("keep", False))
        self.state = "stopped"
        self.error = None
        self.proc = None
        self.inflight = 0
        self.last_used = 0.0
        self.started = None
        self.pending_restart = False    # the GPUs changed while it ran: restart when idle
        self.cv = threading.Condition()
        self.log_path = os.path.join(gw.log_dir, "%s.log" % self.id)

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.port

    @property
    def held(self):
        """Never stopped by the gateway (kept, or a model locked)."""
        return self.keep or bool(self.locked)

    def info(self):
        return {"id": self.id, "kind": self.kind, "models": self.models, "port": self.port, "state": self.state,
                "error": self.error, "pid": self.proc.pid if self.proc and self.proc.poll() is None else None,
                "ram_gb": self.ram_gb, "gpus": self.gpus, "cpu": self.cpu_ok, "keep": self.keep,
                "inflight": self.inflight, "last_used": self.last_used or None, "started": self.started,
                "pending_restart": self.pending_restart, "idle_stop": self.idle_stop, "requires": self.requires,
                "locked": sorted(self.locked)}

    # ---- the process -------------------------------------------------------------------------------
    def command(self):
        args = self.cpu_args if self.gpus == "cpu" else self.gpu_args
        return [v.replace("{port}", str(self.port)) for v in self.cmd + args]

    def environment(self):
        env = dict(os.environ)
        env.update(self.env)
        env.update(self.cpu_env if self.gpus == "cpu" else self.gpu_env)
        if self.gw.backend_key is not None:
            env["RECLANK_API_KEY"] = self.gw.backend_key
        if self.gw.gpu_dir:
            env["RECLANK_GPU_DIR"] = self.gw.gpu_dir
        if self.gpus == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif isinstance(self.gpus, list):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(i)) for i in self.gpus)
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        return env

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def ensure(self):
        """Running and ready when this returns (it starts the server, and waits for a start in
        progress); RuntimeError when it cannot start."""
        for r in self.requires:
            self.gw.servers[r].ensure()
        with self.cv:
            while self.state in ("starting", "stopping"):
                self.cv.wait(1.0)
            if self.state == "ready" and self.alive():
                return
        self.gw.make_room(self)
        with self.cv:
            if self.state == "ready" and self.alive():
                return
            if self.state == "starting":
                while self.state == "starting":
                    self.cv.wait(1.0)
                if self.state != "ready":
                    raise RuntimeError("server %s did not start: %s" % (self.id, self.error))
                return
            self.state, self.error = "starting", None
        try:
            if self.kind != "service" and self.gpus != "cpu":
                # a server's first load does not ask its peers for room (only a reload does): the GPU
                # servers running give back what is not locked first
                self.gw.release_all(but=self)
            self._spawn()
            self._wait_ready()
        except Exception as exc:
            self._kill()
            with self.cv:
                self.state, self.error = "failed", str(exc)
                self.cv.notify_all()
            self.gw.log("%s failed to start: %s" % (self.id, exc))
            raise RuntimeError("server %s did not start: %s" % (self.id, exc))
        with self.cv:
            self.state, self.started = "ready", time.time()
            self.cv.notify_all()
        self.gw.log("%s ready (pid %d)" % (self.id, self.proc.pid))
        if self.kind == "service":
            return
        if not self.models:
            self._ask_models()
        for m in sorted(self.locked):
            self.apply_lock(m, True)

    def apply_lock(self, model, on):
        """The lock of a model ("*": all of the server's) on the running server's GPU."""
        out = self.gw.backend_post(self, "/v1/admin/lock", {"model": None if model == "*" else model, "on": on})
        if out.get("status") == "error":
            self.gw.log("%s: lock %s: %s" % (self.id, model, out.get("message")))
        return out

    def _spawn(self):
        cmd = self.command()
        self.gw.log("%s start: %s%s" % (self.id, " ".join(cmd), "" if self.gpus in ("all", None) else
                                         "  (GPUs %s)" % self.gpus))
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        log = open(self.log_path, "ab")
        log.write(("\n==== %s start: %s\n" % (time.strftime("%F %T"), " ".join(cmd))).encode())
        log.flush()
        try:
            self.proc = subprocess.Popen(cmd, cwd=self.cwd, env=self.environment(), stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
        finally:
            log.close()

    def _wait_ready(self):
        t_end = time.time() + self.start_timeout
        while time.time() < t_end:
            if not self.alive():
                raise RuntimeError("it exited (status %s); its log: %s" % (self.proc.returncode, self.log_path))
            if self.ready_cmd:
                try:
                    r = subprocess.run(self.ready_cmd, cwd=self.cwd, env=self.environment(), capture_output=True,
                                       timeout=60)
                    if r.returncode == 0:
                        return
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(3.0)
                continue
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (OSError, urllib.error.URLError, http.client.HTTPException):
                pass
            time.sleep(0.5)
        raise RuntimeError("no /health after %.0f s" % self.start_timeout)

    def _ask_models(self):
        try:
            data = self.gw.backend_get(self, "/v1/models", timeout=10)
            ids = [str(m["id"]) for m in data.get("data", []) if m.get("id")]
        except Exception as exc:
            self.gw.log("%s: /v1/models: %s" % (self.id, exc))
            return
        if ids:
            self.models = ids
            self.gw.save_state()
            self.gw.log("%s serves %s" % (self.id, ", ".join(ids)))

    def stop(self, wait=True, only_idle=False):
        """Stop the process (SIGTERM to its group, SIGKILL after 60 s). only_idle: only when it is
        ready with no request (checked with the state's change, so a request that came first keeps
        it); False when it was not stopped."""
        with self.cv:
            if only_idle and (self.inflight or self.state != "ready"):
                return False
            if self.state in ("stopped", "failed") and not self.alive():
                self.state = "stopped"
                return True
            self.state = "stopping"
        self.gw.log("%s stop" % self.id)
        self._kill(wait=wait)
        with self.cv:
            self.state, self.pending_restart = "stopped", False
            self.cv.notify_all()
        return True

    def _kill(self, wait=True):
        p = self.proc
        if p is None or p.poll() is not None:
            return
        if self.stop_cmd:
            try:
                subprocess.run(self.stop_cmd, cwd=self.cwd, env=self.environment(), capture_output=True, timeout=300)
                p.wait(120)
                return
            except subprocess.SubprocessError:
                pass
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if not wait:
            return
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(10)

    def begin(self):
        with self.cv:
            self.inflight += 1
            self.last_used = time.time()
        for r in self.requires:
            self.gw.servers[r].last_used = self.last_used

    def end(self):
        with self.cv:
            self.inflight -= 1
            self.last_used = time.time()
            self.cv.notify_all()
        with self.gw.room_cv:
            self.gw.room_cv.notify_all()

    def log_tail(self, n=200):
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 256 * 1024))
                return b"\n".join(f.read().splitlines()[-n:]).decode("utf-8", "replace")
        except OSError:
            return ""


class Gateway:
    def __init__(self, config_path, api_key=None):
        self.config_path = os.path.abspath(config_path)
        cfg = json.load(open(self.config_path))
        base = os.path.dirname(self.config_path)
        self.log_dir = os.path.join(base, cfg.get("log_dir", "logs"))
        self.backend_key = cfg.get("backend_key", api_key)
        self.gpu_dir = cfg.get("gpu_dir")
        self.ram_budget = float(cfg.get("host_ram_gb") or 0.9 * host_ram_gb())
        self.defaults = dict(cfg.get("default") or {})
        self.lock = threading.Lock()        # starts and stops for room, one at a time
        self.room_cv = threading.Condition()     # a request's end (room may be free)
        self.room_wait = float(cfg.get("room_wait", 600))
        self.servers = {}
        for sc in cfg.get("servers", []):
            s = Server(sc, self)
            if s.id in self.servers:
                raise ValueError("server id %s twice" % s.id)
            self.servers[s.id] = s
        ports = [s.port for s in self.servers.values() if s.port]
        if len(set(ports)) != len(ports):
            raise ValueError("two servers on one port")
        for s in self.servers.values():
            for r in s.requires:
                if r not in self.servers or r == s.id:
                    raise ValueError("server %s requires %r: no such server" % (s.id, r))
        self.state_path = self.config_path + ".state.json"
        self._load_state()
        self._gpus = (0.0, [])

    # ---- the state the page changes -------------------------------------------------------------
    def _load_state(self):
        try:
            st = json.load(open(self.state_path))
        except (OSError, ValueError):
            return
        for sid, v in st.get("servers", {}).items():
            s = self.servers.get(sid)
            if s is None:
                continue
            if "gpus" in v:
                s.gpus = v["gpus"]
            if "keep" in v:
                s.keep = bool(v["keep"])
            if "locked" in v:
                s.locked = {str(m) for m in v["locked"]}
            if v.get("models") and not s.models:
                s.models = [str(m) for m in v["models"]]

    def save_state(self):
        st = {"servers": {s.id: {"gpus": s.gpus, "keep": s.keep, "models": s.models, "locked": sorted(s.locked)}
                          for s in self.servers.values()}}
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, self.state_path)

    def log(self, msg):
        print("[gateway] %s %s" % (time.strftime("%F %T"), msg), file=sys.stderr, flush=True)

    # ---- routing --------------------------------------------------------------------------------
    def server_of(self, model, path):
        """The server of a request: by its model, else by its path's kind (the kind's default
        model, else the kind's one running server, else its only server). (server, model) or
        raises LookupError."""
        if model:
            for s in self.servers.values():
                if model in s.models:
                    return s, model
            raise LookupError("the model %r is not served here (GET /v1/models lists them)" % model)
        kind = path_kind(path)
        if kind is None:
            raise LookupError("the request names no model, and its path %s no kind of server" % path)
        if self.defaults.get(kind):
            return self.server_of(self.defaults[kind], path)
        of_kind = [s for s in self.servers.values() if s.kind == kind]
        running = [s for s in of_kind if s.state == "ready"]
        if len(running) == 1:
            return running[0], None
        if len(of_kind) == 1:
            return of_kind[0], None
        raise LookupError("the request names no model (%d %s servers)" % (len(of_kind), kind))

    def make_room(self, server):
        """Stop idle servers (least recently used first) until server's ram_gb fits the budget. While
        the ones in the way have requests, wait for them (room_wait seconds; a request's end wakes
        it); kept servers are never stopped, so with them in the way it fails at once."""
        t_end = time.time() + self.room_wait
        while True:
            with self.lock:
                if server.state in ("ready", "starting") and server.alive():
                    return
                servers = list(self.servers.values())
                victims = room_victims(servers, server.ram_gb, self.ram_budget, exclude=server)
                if victims is not None:
                    for v in victims:
                        if v.stop(only_idle=True):
                            self.log("%s stopped for room (%s)" % (v.id, server.id))
                    if room_victims(servers, server.ram_gb, self.ram_budget, exclude=server) == []:
                        return
                    continue        # one took a request meanwhile: choose again
                # the kept servers stay; the busy ones go when their requests end
                kept = [s for s in servers if s is not server and s.held and s.state in ("starting", "ready")]
                busy = [s.id for s in servers if s is not server and s.inflight and not s.held
                        and server.id not in s.requires]       # (not the one asking for it)
                if sum(s.ram_gb for s in kept) + server.ram_gb > self.ram_budget or time.time() > t_end:
                    raise RuntimeError("no host memory for %s (%.0f GB of the budget's %.0f; held: %s; busy: %s)" % (
                        server.id, server.ram_gb, self.ram_budget, ", ".join(s.id for s in kept) or "none",
                        ", ".join(busy) or "none"))
            with self.room_cv:
                self.room_cv.wait(1.0)

    # ---- the servers' own routes ----------------------------------------------------------------
    def _backend_headers(self):
        h = {"Content-Type": "application/json"}
        if self.backend_key:
            h["Authorization"] = "Bearer " + self.backend_key
        return h

    def backend_get(self, s, path, timeout=5):
        req = urllib.request.Request(s.url + path, headers=self._backend_headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def backend_post(self, s, path, obj, timeout=60):
        req = urllib.request.Request(s.url + path, data=json.dumps(obj).encode(), method="POST",
                                     headers=self._backend_headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return {"status": "error", "code": exc.code, "message": exc.read().decode("utf-8", "replace")[:500]}

    # ---- the page's actions ---------------------------------------------------------------------
    def action(self, sid, act, body):
        s = self.servers.get(sid)
        if s is None:
            raise LookupError("no server %r" % sid)
        if act == "start":
            s.ensure()
            return s.info()
        if act == "stop":
            s.stop()
            return s.info()
        if act == "restart":
            s.stop()
            s.ensure()
            return s.info()
        if act == "release":
            if s.state != "ready" or s.kind == "service":
                raise RuntimeError("server %s is not running" % sid)
            return self.backend_post(s, "/v1/admin/release", {})
        if act in ("lock", "unlock"):
            if s.kind == "service":
                raise ValueError("a service has no models to lock")
            on = act == "lock" and bool(body.get("on", True))
            m = body.get("model") or "*"
            if m != "*" and m not in s.models:
                raise LookupError("server %s has no model %r" % (sid, m))
            if m == "*":
                s.locked = {"*"} if on else set()
            elif on:
                s.locked.add(m)
            else:
                s.locked = (set(s.models) if "*" in s.locked else s.locked) - {m, "*"}
            self.save_state()
            if s.state == "ready":
                s.apply_lock(m, on)
            return s.info()
        if act == "keep":
            s.keep = bool(body.get("on", True))
            self.save_state()
            return s.info()
        if act == "gpus":
            g = body.get("gpus")
            if g == "cpu":
                if not s.cpu_ok:
                    raise ValueError("server %s cannot run on the CPU" % sid)
            elif g in ("all", None):
                g = "all"
            elif isinstance(g, list) and g and all(isinstance(i, int) and i >= 0 for i in g):
                g = sorted(set(g))
            else:
                raise ValueError('gpus: "all", "cpu", or a list of GPU indices')
            changed = g != s.gpus
            s.gpus = g
            self.save_state()
            if changed and s.alive():
                s.pending_restart = True
                self.idle_tick()
            return s.info()
        raise LookupError("no action %r" % act)

    def release_all(self, but=None):
        """Every running server's POST /v1/admin/release (but that one): {id: its answer}."""
        out = {}
        for s in self.servers.values():
            if s.state == "ready" and s.kind != "service" and s is not but and s.gpus != "cpu":
                try:
                    out[s.id] = self.backend_post(s, "/v1/admin/release", {}, timeout=60)
                except Exception as exc:
                    out[s.id] = {"status": "error", "message": str(exc)}
        return out

    def idle_tick(self):
        """Stop the servers idle past their idle_stop, and restart (stop: the next request starts
        it) the ones whose GPUs changed, when they have no request."""
        now = time.time()
        for s in self.servers.values():
            if s.state == "ready" and not s.alive():
                self.log("%s exited (status %s)" % (s.id, s.proc.returncode))
                with s.cv:
                    s.state, s.error = "failed", "exited (status %s)" % s.proc.returncode
                continue
            if s.state != "ready" or s.inflight:
                continue
            if s.pending_restart:
                if s.stop(only_idle=True):
                    threading.Thread(target=self._start_quietly, args=(s,), daemon=True).start()
            elif (s.idle_stop and not s.held and now - max(s.last_used, s.started or 0) > s.idle_stop
                  and not any(s.id in d.requires and d.state in RUNNING for d in self.servers.values())):
                if s.stop(only_idle=True):
                    self.log("%s idle %.0f s: stopped" % (s.id, now - max(s.last_used, s.started or 0)))

    def _start_quietly(self, s):
        try:
            s.ensure()
        except Exception:
            pass

    def gpus(self):
        t, g = self._gpus
        if time.time() - t > 2.0:
            g = gpu_list()
            self._gpus = (time.time(), g)
        return g

    def state(self):
        servers = []
        for s in self.servers.values():
            info = s.info()
            if s.state == "ready" and s.kind != "service":
                try:
                    info["gpu"] = self.backend_get(s, "/v1/admin/gpu", timeout=2)
                except Exception:
                    info["gpu"] = None
            servers.append(info)
        held = sum(s.ram_gb for s in self.servers.values() if s.state in ("starting", "ready", "stopping"))
        return {"servers": servers, "gpus": self.gpus(), "ram": {"budget_gb": self.ram_budget, "held_gb": held,
                                                              "total_gb": host_ram_gb()},
                "defaults": self.defaults}

    def models(self):
        out = []
        for s in self.servers.values():
            for m in s.models:
                out.append({"id": m, "object": "model", "created": 0, "owned_by": "reclank", "server": s.id,
                            "kind": s.kind, "state": s.state})
        return out

    def stop_all(self):
        for s in self.servers.values():
            if s.alive():
                s.stop()


class _Handler(H._Handler):
    server_version = "reclank-gateway/1.0"

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/")
        gw = self.server.gw
        if path in ("/health", "/healthz", "/v1/health"):
            self._json(200, {"status": "ok"})
            return
        if path in ("", "/gateway", "/index.html"):
            self._page()
            return
        if not self._authorized():
            return
        if path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": gw.models()})
            return
        if path == "/gateway/state":
            self._json(200, gw.state())
            return
        if path in ("/v1/admin/gpu", "/admin/gpu"):
            self._json(200, {s["id"]: s.get("gpu") for s in gw.state()["servers"] if s.get("gpu") is not None})
            return
        m = re.fullmatch(r"/gateway/servers/([^/]+)/log", path)
        if m:
            s = gw.servers.get(m.group(1))
            if s is None:
                self._error(404, "no server %r" % m.group(1))
                return
            self._text(s.log_tail())
            return
        self._proxy(b"")

    def do_POST(self):
        if not self._authorized():
            self.close_connection = True
            return
        path = urlsplit(self.path).path.rstrip("/")
        try:
            body = self._raw_body()
        except H._BodyError as exc:
            self._error(exc.status, str(exc))
            return
        if path in ("/v1/admin/release", "/admin/release"):
            self._json(200, self.server.gw.release_all())
            return
        m = re.fullmatch(r"/gateway/servers/([^/]+)/([a-z]+)", path)
        if m:
            try:
                arg = json.loads(body) if body.strip() else {}
                self._json(200, self.server.gw.action(m.group(1), m.group(2), arg))
            except LookupError as exc:
                self._error(404, str(exc))
            except ValueError as exc:
                self._error(400, str(exc))
            except RuntimeError as exc:
                self._error(503, str(exc), kind="server_error")
            return
        self._proxy(body)

    def do_DELETE(self):
        if not self._authorized():
            self.close_connection = True
            return
        try:
            body = self._raw_body()
        except H._BodyError as exc:
            self._error(exc.status, str(exc))
            return
        self._proxy(body)

    def _page(self):
        try:
            data = open(PAGE, "rb").read()
        except OSError:
            self._error(404, "the page is missing")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _text(self, text):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self, body):
        """The request to its model's server (started first if need be); the answer back as it
        comes (a stream's events as they come)."""
        gw = self.server.gw
        path = urlsplit(self.path).path
        model = request_model(body, self.headers.get("Content-Type"))
        try:
            s, _ = gw.server_of(model, path)
        except LookupError as exc:
            self._error(404, str(exc), code="model_not_found")
            return
        s.begin()
        try:
            try:
                s.ensure()
            except RuntimeError as exc:
                self._error(503, str(exc), kind="server_error")
                return
            self._forward(s, body)
        finally:
            s.end()

    def _forward(self, s, body):
        gw = self.server.gw
        conn = http.client.HTTPConnection("127.0.0.1", s.port, timeout=None)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if gw.backend_key:
            headers["Authorization"] = "Bearer " + gw.backend_key
        headers["Content-Length"] = str(len(body))
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            r = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            self._error(502, "server %s: %s" % (s.id, exc), kind="server_error")
            return
        try:
            self.send_response(r.status, r.reason)
            length = r.getheader("Content-Length")
            for k, v in r.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            if length is not None:
                self.send_header("Content-Length", length)
            else:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            if length is not None:
                left = int(length)
                while left > 0:
                    chunk = r.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            else:
                while True:
                    chunk = r.read1(1 << 16)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True     # the client went: the server sees its connection go
        finally:
            conn.close()


class _Server(H._Server):
    pass


def make_server(gw, host="127.0.0.1", port=8080, api_key=None, max_body=None, quiet=True):
    httpd = _Server((host, port), _Handler)
    httpd.gw = gw
    httpd.api_key = api_key or None
    httpd.cors_origin = None
    httpd.quiet = quiet
    httpd.max_body = max_body or (1 << 30)
    return httpd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="the servers (JSON; the module text has its form)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("RECLANK_API_KEY", H.DEFAULT_API_KEY),
                    help="the key of each request and of the page; default RECLANK_API_KEY, else %r; \"\" "
                         "turns the check off" % H.DEFAULT_API_KEY)
    ap.add_argument("--max-body-mb", type=float, default=1024.0)
    ap.add_argument("--start", default="", metavar="ID,...", help="servers to start at once")
    ap.add_argument("--verbose", action="store_true", help="log each request")
    a = ap.parse_args(argv)
    gw = Gateway(a.config, api_key=a.api_key or None)
    httpd = make_server(gw, a.host, a.port, a.api_key, int(a.max_body_mb * (1 << 20)), quiet=not a.verbose)

    def tick():
        while True:
            time.sleep(2.0)
            try:
                gw.idle_tick()
            except Exception as exc:
                gw.log("idle tick: %s" % exc)
    threading.Thread(target=tick, daemon=True).start()

    def down(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, down)
    for sid in [v for v in a.start.split(",") if v]:
        threading.Thread(target=gw._start_quietly, args=(gw.servers[sid],), daemon=True).start()
    print("reclank gateway http://%s:%d  (the page: /)  %d servers, %d models  key %s" % (
        a.host, httpd.server_address[1], len(gw.servers), len(gw.models()),
        "required" if a.api_key else "OFF"), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        gw.log("stopping the servers")
        gw.stop_all()


if __name__ == "__main__":
    main()
