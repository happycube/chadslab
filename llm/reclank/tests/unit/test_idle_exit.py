"""The idle exit (server/idle.py) and its launcher (scripts/serve_idle.py; docs/GPU_SWAP_PLAN.md 3.7,
step 5) with a small HTTP server that answers its pid: idle, it exits (75) and the launcher keeps the
socket; the next request starts it again and is answered by the new server; a request longer than
the idle time is not cut; an idle keep-alive connection ends at its timeout and does not keep the
server; busy (a file) keeps it; SIGTERM to the launcher ends both."""
import http.client
import os
import signal
import socket
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SERVER = '''
import os, sys, time
from http.server import BaseHTTPRequestHandler
from server import http as H, idle

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def do_GET(self):
        if self.path.startswith("/sleep"):
            time.sleep(float(self.path.split("/")[-1]))
        b = str(os.getpid()).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

httpd = idle.server(H._Server, ("127.0.0.1", int(sys.argv[1])), Handler)
busy = sys.argv[3]
sys.exit(idle.serve(httpd, float(sys.argv[2]), lambda: os.path.exists(busy)))
'''


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _get(port, path="/", conn=None):
    c = conn or http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("GET", path)
    r = c.getresponse()
    assert r.status == 200
    return int(r.read())


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # (a zombie of the launcher counts as gone)
    try:
        return open("/proc/%d/stat" % pid).read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def _gone(pid, within):
    for _ in range(int(within * 20)):
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_idle_exit(tmp_path):
    port = _port()
    srv = tmp_path / "srv.py"
    srv.write_text(SERVER)
    busy = str(tmp_path / "busy")
    env = dict(os.environ, PYTHONPATH=ROOT, RECLANK_IDLE_CONN_TIMEOUT="1")
    log = open(tmp_path / "launcher.log", "w")
    lp = subprocess.Popen([sys.executable, os.path.join(ROOT, "scripts", "serve_idle.py"), "--port", str(port),
                           "--", sys.executable, str(srv), str(port), "2", busy], env=env, stderr=log)
    try:
        for _ in range(200):
            try:
                a = _get(port)
                break
            except OSError:
                time.sleep(0.05)
        assert _alive(a)
        # a request longer than the idle time: answered by the same server
        assert _get(port, "/sleep/3") == a
        # idle: it exits; the next request starts another, which answers
        assert _gone(a, 6)
        b = _get(port)
        assert b != a and _alive(b)
        # an idle keep-alive connection ends at its timeout: the server exits all the same
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        assert _get(port, conn=c) == b
        assert _gone(b, 8)
        # busy: no exit
        open(busy, "w").close()
        d = _get(port)
        time.sleep(4)
        assert _alive(d) and _get(port) == d
        os.unlink(busy)
        assert _gone(d, 8)
        e = _get(port)
        # a stop of the launcher: both end
        lp.send_signal(signal.SIGTERM)
        assert lp.wait(timeout=20) == 0 and _gone(e, 5)
    finally:
        if lp.poll() is None:
            lp.kill()
            lp.wait()
    text = open(tmp_path / "launcher.log").read()
    assert text.count("exited idle") == 3 and "done" in text, text
