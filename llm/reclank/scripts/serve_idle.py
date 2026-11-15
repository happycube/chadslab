"""The launcher of a server that exits when idle (docs/GPU_SWAP_PLAN.md 3.7, step 5): it binds the port and
keeps the listening socket, and runs the server command with it (RECLANK_LISTEN_FD, server/idle.py).

    scripts/serve_idle.py --port 8081 -- python -m server -m MODEL --port 8081 --idle-exit 600 ...

- the server exits idle (status 75): its CUDA context and all its own memory gone; the launcher waits
  for the next connection (without taking it: it waits in the socket's backlog) and starts the server
  again, which takes it;
- a clean exit (0) or a stop (SIGINT, SIGTERM: given to the server too) ends the launcher;
- another status (a crash; 70, a fatal CUDA error) starts the server again after a pause, as
  scripts/serve_forever.sh (doubling up to 60 s when it ends often).
"""
from __future__ import annotations

import argparse
import os
import select
import signal
import socket
import subprocess
import sys
import time

EXIT_IDLE = 75      # (server/idle.py)
EXIT_STOP = (0, 130, -signal.SIGINT, -signal.SIGTERM, 128 + signal.SIGTERM)


def log(msg):
    print("[serve_idle] %s %s" % (time.strftime("%F %T"), msg), file=sys.stderr, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--backlog", type=int, default=128)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- the server's command")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("the server's command (after --)")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((a.host, a.port))
    s.listen(a.backlog)
    s.set_inheritable(True)
    env = dict(os.environ, RECLANK_LISTEN_FD=str(s.fileno()))
    child = [None]
    stopping = [False]

    def stop(sig, _frame):
        stopping[0] = True
        if child[0] is not None and child[0].poll() is None:
            child[0].send_signal(sig)
        else:
            sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log("pid %d on %s:%d: %s" % (os.getpid(), a.host, a.port, " ".join(cmd)))
    wait = 5
    while True:
        start = time.time()
        child[0] = subprocess.Popen(cmd, env=env, pass_fds=(s.fileno(),))
        log("the server: pid %d" % child[0].pid)
        while True:
            try:
                code = child[0].wait()
                break
            except InterruptedError:
                continue
        child[0] = None
        if stopping[0] or code in EXIT_STOP:
            log("the server ended (%d): done" % code)
            return 0 if code in EXIT_STOP else code
        if code == EXIT_IDLE:
            log("the server exited idle after %.0f s: waiting for a connection" % (time.time() - start))
            while not stopping[0]:
                try:
                    r, _w, _x = select.select([s], [], [], 1.0)
                except InterruptedError:
                    continue
                if r:
                    break
            if stopping[0]:
                return 0
            log("a connection: starting the server")
            wait = 5
            continue
        log("the server ended (%d); again in %d s" % (code, wait))
        time.sleep(wait)
        wait = 5 if time.time() - start > 600 else min(60, wait * 2)


if __name__ == "__main__":
    sys.exit(main())
