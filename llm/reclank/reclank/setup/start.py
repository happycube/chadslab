"""Start a server: the logic of ./start.sh (python -m reclank.setup.start).

    ./start.sh                         the models downloaded here, and how each starts
    ./start.sh gemma4-26b              a model of the setup table, by its id
    ./start.sh ~/models/x.gguf         a model file or folder
    ./start.sh gemma4-26b -d           detached: waits until it serves, then returns
    ./start.sh gemma4-26b --ctx 8192   the arguments this script does not know go to the server
    ./start.sh --stop                  stop the reclank server on the port

The id of a model in the table (catalog.py) gives its folder under the models
folder (RECLANK_MODELS, else ~/models), its server (chat 8080, embeddings
8082, generation 8084) and its flags. The mode is the GPU when the model fits
the card (fit()), else the CPU; --gpu or --cpu chooses it. A file or a folder
runs on python -m server, on the GPU when the machine has an NVIDIA GPU and
nvcc, else on the CPU.

plan() decides everything and touches nothing; main() prints the plan, checks
the port, then runs it in the foreground (exec) or detached.
"""
from __future__ import annotations

import glob
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

from reclank import setup as S
from reclank.setup import hardware
from reclank.setup.catalog import MODELS, by_id

USAGE = """usage: ./start.sh [MODEL] [options] [server arguments ...]

MODEL   an id of the setup table (./start.sh --list; python -m reclank.setup
        shows every model and what this machine runs), or a model file
        (.gguf) or folder

options
  -d, --detach     run in the background: wait until it serves, print its pid and log
  --gpu, --cpu     the mode (default: the GPU when the model fits it, else the CPU)
  --port N         the port (default: chat 8080, embeddings 8082, generation 8084)
  --host H         the address it listens on (default 127.0.0.1; 0.0.0.0: every one)
  --restart        stop the reclank server on the port first
  --keeper         start the model's weightkeeper first when none runs (DeepSeek
                   V4.1, Qwen3.8-Flash-Next-FP4-Local, Qwen3.6 NVFP4)
  --log FILE       the log with -d (default ~/.cache/reclank/MODEL.log)
  -n, --dry-run    print what would run; start nothing
  --stop           stop the reclank server on the port (--port N, or MODEL's port)
  --list           the models of the table downloaded here, and their mode here
  -h, --help       this text

Any other argument goes to the server as it is (python -m server --help lists them).

settings (environment)
  KEY              the API key clients send; else RECLANK_API_KEY; else the file
                   ~/.config/reclank/api_key; else the server's default, change-me
  HOST, PORT       the same as --host, --port
  RECLANK_MODELS   the models folder (default ~/models)
  PY               the Python to run (default: the active venv, else .venv, else
                   ~/.venvs/reclank, else python3)
  ~/.config/reclank/env (or the file RECLANK_ENV names) is sourced first: the
  settings of this machine (CUDA_HOME or NVCC, CUDA_VISIBLE_DEVICES, ...)

examples
  ./start.sh gemma4-26b                       chat on http://127.0.0.1:8080/v1
  ./start.sh lfm25-24b --cpu -d               on the CPU, in the background
  HOST=0.0.0.0 KEY=secret ./start.sh gemma4-12b --drafter ~/models/gemma-4-12B-unsloth-UD-Q4_K_XL/mtp-gemma-4-12B-it.gguf
  ./start.sh ~/models/some-model.gguf --ctx 32768
  ./start.sh --stop
"""

# the line each server prints once it serves (server/http.py, embed.py, generate.py, gateway.py)
READY = re.compile(r"^reclank (\S+ )?(server|gateway) ")
CACHE = os.path.expanduser("~/.cache/reclank")
OURS = {"-d", "--detach", "--gpu", "--cpu", "--restart", "--keeper", "-n", "--dry-run", "--stop", "--list",
        "-h", "--help"}
WITH_VALUE = {"--port", "--host", "--log", "-m", "--model"}


@dataclass
class Plan:
    name: str                      # what runs, for the user
    tag: str                       # a short name: the log's
    args: list                     # after the python executable
    env: dict = field(default_factory=dict)
    port: int = 8080
    mode: str = ""                 # gpu, cpu, or "" (the server's choice)
    keeper: str | None = None      # the folder its weightkeeper holds
    notes: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def split_args(argv):
    """argv -> (options of ours: {name: value or True}, the model or None, the server's arguments)."""
    opts, target, rest = {}, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        name, eq, val = a.partition("=")
        if a in OURS:
            opts[a.lstrip("-")] = True
        elif name in WITH_VALUE and (eq or i + 1 < len(argv)):
            if not eq:
                i += 1
                val = argv[i]
            opts["model" if name in ("-m", "--model") else name.lstrip("-")] = val
        elif target is None and not a.startswith("-") and not opts.get("model") and \
                (by_id(a) or not rest or not rest[-1].startswith("-")):
            target = a
        else:
            rest.append(a)
        i += 1
    if "model" in opts:
        target = opts.pop("model")
    opts["d"] = opts.pop("detach", opts.get("d"))
    opts["n"] = opts.pop("dry-run", opts.get("n"))
    opts["h"] = opts.pop("help", opts.get("h"))
    return opts, target, rest


def _missing_files(args):
    """The paths among the arguments that do not exist (a model's files)."""
    return [a for a in args if a.startswith("/") and not os.path.exists(a)]


def _beside(path):
    """The one mmproj file beside a GGUF model (or in a folder), else None."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    found = [p for p in glob.glob(os.path.join(d, "*mmproj*.gguf")) if os.path.abspath(p) != os.path.abspath(path)]
    return found[0] if len(found) == 1 else None


def plan(target, m, root, mode=None, port=None, rest=()):
    """What to run for target (a table id, or a model file or folder) on machine m."""
    model = by_id(target)
    if model is None and not os.path.exists(os.path.expanduser(target)):
        p = Plan(target, "model", [])
        near = [x.id for x in MODELS if target.lower() in x.id]
        p.errors.append("no model %r: neither an id of the table%s nor a file or folder" % (
            target, " (the near ones: %s)" % ", ".join(near) if near else ""))
        p.errors.append("./start.sh --list: the downloaded ones; python -m reclank.setup: every one")
        return p
    if model is None:
        return _plan_path(os.path.abspath(os.path.expanduser(target)), m, mode, port, rest)
    p = Plan(model.name, model.id, [], port=port or S.PORTS[model.server])
    d = os.path.join(root, model.dir)
    if model.status == "not-yet":
        p.errors.append("%s is not supported yet: %s" % (model.id, model.why))
        return p
    f = S.fit(model, m)
    if mode is None:
        mode = "gpu" if "gpu" in f["modes"] else "cpu" if "cpu" in f["modes"] else None
        if mode is None:
            p.errors.append("%s does not run on this machine: %s" % (model.id, "; ".join(f["why"])))
            p.errors.append("(--gpu or --cpu tries it anyway)")
            return p
    elif not getattr(model, mode):
        p.errors.append("%s has no %s mode (it has: %s)" % (
            model.id, mode.upper(), ", ".join(k.upper() for k in ("gpu", "cpu") if getattr(model, k))))
        return p
    elif mode not in f["modes"]:
        p.notes.append("warning: %s on the %s may not fit this machine: %s" % (
            model.id, mode.upper(), "; ".join(f["why"])))
    p.mode = mode
    p.env, p.args = S.serve_argv(model, root, mode, p.port)
    p.notes += ["%s: %s" % (mode.upper(), n) for n in f["modes"].get(mode, [])]
    p.notes += ["warning: " + w for w in f["warn"] if not w.startswith("the download is")]
    if model.keeper:
        p.keeper = d
    missing = [d] if not os.path.isdir(d) else _missing_files(p.args)
    if missing:
        for x in missing:
            p.errors.append("not downloaded: %s" % x)
        if model.status == "manual":
            p.errors.append("prepare it first: " + model.why)
        else:
            p.errors.append("download it (%.1f GB): python -m reclank.setup --model %s --download" % (
                model.gb, model.id))
            p.errors.append("   or: %s" % S.download_cmd(model, root, m.hf_cli))
    p.args += list(rest)
    return p


def _plan_path(path, m, mode, port, rest):
    """A model file or folder on python -m server."""
    p = Plan(path, os.path.basename(path.rstrip("/")).rsplit(".", 1)[0] or "model", [], port=port or 8080)
    if mode is None:
        mode = "gpu" if m.gpus and m.toolkits else "cpu"
        if m.gpus and not m.toolkits:
            p.notes.append("the CPU: the machine has an NVIDIA GPU but no CUDA toolkit (nvcc)")
    p.mode = mode
    p.env = {"OMP_NUM_THREADS": "1"} if mode == "gpu" else {}
    p.args = ["-m", "server", "--model", path, "--backend", "cuda" if mode == "gpu" else "cpu",
              "--port", str(p.port)]
    mm = _beside(path)
    if mm and "--mmproj" not in rest:
        p.args += ["--mmproj", mm]
        p.notes.append("images: --mmproj %s (beside the model)" % mm)
    p.args += list(rest)
    return p


def api_key():
    """(key, where it came from) or (None, "the server's default")."""
    for name in ("KEY", "RECLANK_API_KEY"):
        if os.environ.get(name):
            return os.environ[name], name
    f = os.path.expanduser("~/.config/reclank/api_key")
    if os.path.isfile(f):
        with open(f) as fh:
            k = fh.readline().strip()
        if k:
            return k, f
    return None, "the default"


def port_owner(port):
    """(pid, its command line) of the process listening on port, (None, "") when the port is
    free, (0, "") when it is taken but its process is not to be seen."""
    try:
        out = subprocess.run(["ss", "-ltnpH", "sport = :%d" % port], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        out = None
    if out is not None:
        if not out.strip():
            return None, ""
        mm = re.search(r"pid=(\d+)", out)
        if not mm:
            return 0, ""
        pid = int(mm.group(1))
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = ""
        return pid, cmd
    import socket
    with socket.socket() as s:
        return (0, "") if s.connect_ex(("127.0.0.1", port)) == 0 else (None, "")


def alive(pid):
    """pid runs (a zombie, ended but not yet reaped, does not)."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def is_reclank(cmd):
    return " -m server" in " " + cmd


def stop(port):
    """Stop the reclank server on port; 0 when none is left."""
    pid, cmd = port_owner(port)
    if pid is None:
        print("no server on port %d" % port)
        return 0
    if not pid or not is_reclank(cmd):
        print("port %d is not a reclank server's: pid %s %s" % (port, pid or "?", cmd[:120]), file=sys.stderr)
        return 1
    print("stopping the server on port %d (pid %d)" % (port, pid))
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not alive(pid):
            return 0
        time.sleep(0.5)
    print("it did not stop in 30 s: SIGKILL", file=sys.stderr)
    os.kill(pid, signal.SIGKILL)
    while alive(pid):
        time.sleep(0.5)
    return 0


def _wait(proc, log, pattern, what):
    """Wait until log shows pattern (True) or proc ends (False)."""
    t0 = time.time()
    try:
        while True:
            try:
                with open(log, errors="replace") as f:
                    for line in f:
                        if pattern.search(line):
                            print("%s ready after %.0f s: %s" % (what, time.time() - t0, line.strip()[:160]))
                            return True
            except OSError:
                pass
            if proc.poll() is not None:
                sys.stdout.flush()
                print("%s stopped (exit %s) after %.0f s; the end of %s:" % (
                    what, proc.returncode, time.time() - t0, log), file=sys.stderr)
                with open(log, errors="replace") as f:
                    sys.stderr.write("".join(f.readlines()[-20:]))
                return False
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopped waiting; %s keeps running (pid %d, log %s)" % (what, proc.pid, log))
        raise SystemExit(130)


def _spawn(args, env, log):
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    with open(log, "w") as out:
        return subprocess.Popen(args, env=env, cwd=S.ROOT, stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)


def start_keeper(folder, env, tag):
    """Start the weightkeeper of folder and wait until it holds it."""
    log = os.path.join(CACHE, "weightkeeper-%s.log" % tag)
    print("starting the weightkeeper of %s (its log: %s); it reads the experts into memory once" % (folder, log))
    proc = _spawn([sys.executable, "scripts/weightkeeper.py", folder], env, log)
    return _wait(proc, log, re.compile(r"ready in"), "the weightkeeper")


def keeper_running():
    try:
        from reclank.runtime import weightkeeper
        return bool(weightkeeper.keepers())
    except Exception:
        return False


def list_models(m, root):
    print("Models of the table in %s:" % root)
    n = 0
    for model in MODELS:
        if model.status == "not-yet" or not os.path.isdir(os.path.join(root, model.dir)):
            continue
        f = S.fit(model, m)
        modes = ", ".join(k.upper() for k in ("gpu", "cpu") if k in f["modes"]) or "does not run here"
        p = plan(model.id, m, root)
        miss = [e[len("not downloaded: "):] for e in p.errors if e.startswith("not downloaded: ")]
        if miss:
            modes += "  (incomplete: no %s)" % ", ".join(os.path.basename(x) for x in miss)
        print("  %-22s %-44s %s" % (model.id, model.name[:44], modes))
        n += 1
    if not n:
        print("  none yet")
    print("Every model, what this machine runs, and the downloads: python -m reclank.setup")
    return 0


def main(argv=None):
    opts, target, rest = split_args(sys.argv[1:] if argv is None else argv)
    if opts.get("h"):
        print(USAGE)
        return 0
    port = opts.get("port") or os.environ.get("PORT")
    port = int(port) if port else None
    host = opts.get("host") or os.environ.get("HOST")
    if opts.get("stop"):
        if port is None and target and by_id(target):
            port = S.PORTS[by_id(target).server]
        return stop(port or 8080)
    root = S.models_dir()
    m = hardware.detect(root)
    if opts.get("list") or target is None:
        if target is None and not opts.get("list"):
            print(USAGE.split("\noptions")[0] + "\n(./start.sh --help: every option)\n")
        return list_models(m, root)
    mode = "gpu" if opts.get("gpu") else "cpu" if opts.get("cpu") else None
    p = plan(target, m, root, mode, port, rest)
    if host and "--host" not in rest:
        p.args += ["--host", host]
    if p.errors:
        for e in p.errors:
            print(e, file=sys.stderr)
        return 2
    if not m.numpy:
        print("%s has no NumPy: pip install numpy (in reclank's venv; PY=... picks another Python)"
              % sys.executable, file=sys.stderr)
        return 2
    env = dict(os.environ)
    env.update(p.env)
    env["PYTHONPATH"] = S.ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    key, where = api_key()
    if key:
        env["RECLANK_API_KEY"] = key
    else:
        env.pop("RECLANK_API_KEY", None)   # (an empty one turns the check off)
    cmd = " ".join("%s=%s" % kv for kv in p.env.items()) + (" " if p.env else "") + \
        " ".join(shlex.quote(a) for a in [sys.executable] + p.args)
    url = "http://%s:%d/v1" % ("127.0.0.1" if host in (None, "", "0.0.0.0") else host, p.port)
    print("model   %s%s" % (p.name, " on the %s" % p.mode.upper() if p.mode else ""))
    print("serves  %s  (key: %s)" % (url, "from " + where if key else "change-me, the default"))
    for n in p.notes:
        print("note    " + n)
    have_keeper = keeper_running() if p.keeper else True
    if not have_keeper and not opts.get("keeper"):
        print("note    this model takes its experts from a weightkeeper, and none runs: --keeper starts one "
              "first (or: python scripts/weightkeeper.py %s)" % p.keeper)
    print("runs    cd %s && %s" % (shlex.quote(S.ROOT), cmd))
    if opts.get("n"):
        return 0

    pid, owner = port_owner(p.port)
    if pid is not None:
        if opts.get("restart") and pid and is_reclank(owner):
            if stop(p.port):
                return 1
        else:
            print("port %d is in use: pid %s %s" % (p.port, pid or "?", owner[:120]), file=sys.stderr)
            print("stop it (./start.sh --stop --port %d), restart (--restart), or take another port (--port N)"
                  % p.port, file=sys.stderr)
            return 1
    if not have_keeper and opts.get("keeper") and not start_keeper(p.keeper, env, p.tag):
        return 1
    if not opts.get("d"):
        print("The first start builds the kernels (C about a minute, CUDA some minutes). Ctrl-C stops the server.")
        sys.stdout.flush()
        os.chdir(S.ROOT)
        os.execve(sys.executable, [sys.executable] + p.args, env)
    log = os.path.abspath(os.path.expanduser(opts.get("log") or os.path.join(CACHE, p.tag + ".log")))
    proc = _spawn([sys.executable] + p.args, env, log)
    print("started pid %d, log %s; waiting until it serves (the first start builds the kernels)" % (proc.pid, log),
          flush=True)
    if not _wait(proc, log, READY, "the server"):
        return 1
    print("stop it with: ./start.sh --stop --port %d" % p.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
