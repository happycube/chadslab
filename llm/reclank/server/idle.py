"""A server that exits when idle (docs/GPU_SWAP_PLAN.md 3.7, step 5: --idle-exit S), and the socket it
may take from its launcher (scripts/serve_idle.py).

- **The socket.** With RECLANK_LISTEN_FD (the launcher's), the server takes that listening socket as
  it is (bound, listening) instead of binding its port: while the server is gone the launcher keeps
  it, and a client's connection waits in its backlog until the next server takes it.
- **Idle.** --idle-exit S: no request in flight and none for S seconds, and nothing busy (busy: the
  server's own work with no request, as the generation server's queued jobs) -> the server stops
  accepting, serves the requests of the connections it has (an idle keep-alive connection ends at
  CONN_TIMEOUT), and exits with EXIT_IDLE; the launcher starts it again at the next connection. A
  request or a job that comes during that drain, or a busy server then, serves on (the timer from 0).
  The weights of a server that exits are best in a weightkeeper (its next start attaches in seconds).

    httpd = idle.server(_Server, (host, port), Handler)
    ...
    return idle.serve(httpd, a.idle_exit, busy=lambda: ...)     # serve_forever, or EXIT_IDLE
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time

EXIT_IDLE = 75          # (EX_TEMPFAIL) the launcher starts the server again at the next connection
# with --idle-exit: an idle keep-alive connection ends after this
CONN_TIMEOUT = float(os.environ.get("RECLANK_IDLE_CONN_TIMEOUT", "5"))


def add_idle_args(ap):
    ap.add_argument("--idle-exit", type=float, default=None, metavar="S",
                    help="exit (status %d) after S seconds with no request and nothing busy; with "
                         "scripts/serve_idle.py, which keeps the socket and starts the server again at the "
                         "next connection (the weights best in a weightkeeper)" % EXIT_IDLE)


def server(cls, addr, handler):
    """An HTTP server of cls: on the launcher's socket (RECLANK_LISTEN_FD), else bound to addr."""
    fd = os.environ.get("RECLANK_LISTEN_FD")
    if not fd:
        return cls(addr, handler)
    httpd = cls(addr, handler, bind_and_activate=False)
    httpd.socket.close()
    httpd.socket = socket.socket(fileno=int(fd))
    httpd.server_address = httpd.socket.getsockname()[:2]
    os.environ.pop("RECLANK_LISTEN_FD", None)       # (not handed to the server's own children)
    return httpd


class _Watch:
    """The requests in flight and the connections open of an HTTP server; the time of the last."""

    def __init__(self):
        self.mu = threading.Lock()
        self.requests = self.conns = self.begun = 0
        self.last = time.time()         # the last request's start or end

    def change(self, which, d):
        with self.mu:
            setattr(self, which, getattr(self, which) + d)
            if which == "requests":
                self.last = time.time()
                self.begun += d > 0


def _wrap(handler, watch, timeout):
    """handler's class counting its requests (parse_request to the end of handle_one_request) and
    connections (handle), its idle keep-alive connections ended after timeout (the wait for a request
    line only)."""
    class Counted(handler):
        def handle(self):
            watch.change("conns", 1)
            try:
                super().handle()
            finally:
                watch.change("conns", -1)

        def parse_request(self):
            ok = super().parse_request()
            if ok:
                self._counted = True
                watch.change("requests", 1)
                self.connection.settimeout(None)    # (the request's body and answer: no timeout)
            return ok

        def handle_one_request(self):
            # the wait for the next request line of a keep-alive connection: at most timeout
            self.connection.settimeout(timeout)
            try:
                super().handle_one_request()
            finally:
                if getattr(self, "_counted", False):
                    self._counted = False
                    watch.change("requests", -1)
    Counted.__name__ = Counted.__qualname__ = handler.__name__
    return Counted


def serve(httpd, idle_exit=None, busy=None, log=None):
    """httpd.serve_forever(); with idle_exit (seconds), until the server is idle that long: then the
    drain and EXIT_IDLE (the module's text). Return 0 otherwise (the server stopped)."""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if not idle_exit:
        httpd.serve_forever()
        return 0
    watch = _Watch()
    httpd.RequestHandlerClass = _wrap(httpd.RequestHandlerClass, watch, CONN_TIMEOUT)
    busy = busy or (lambda: False)
    stop, fired = threading.Event(), threading.Event()

    def watcher():
        while not stop.wait(1.0):
            with watch.mu:
                quiet = watch.requests == 0 and time.time() - watch.last >= idle_exit
            if quiet and not busy():
                fired.set()
                httpd.shutdown()        # (serve_forever returns; the backlog waits for the next server)
                return
    while True:
        stop.clear()
        fired.clear()
        t = threading.Thread(target=watcher, name="idle-exit", daemon=True)
        t.start()
        httpd.serve_forever()
        stop.set()
        if not fired.is_set():
            return 0                    # (stopped by another: not idle)
        t.join()
        t0 = time.time()
        with watch.mu:
            begun = watch.begun
        while True:
            with watch.mu:
                left = watch.conns
            if left == 0:
                break
            time.sleep(0.05)
        if watch.begun > begun or busy():
            log("idle exit: a request came during the drain: serving on")
            watch.last = time.time()
            continue
        log("idle exit: %.0f s with no request; exit %d (the drain %.1f s)" % (idle_exit, EXIT_IDLE,
                                                                             time.time() - t0))
        return EXIT_IDLE
