"""A stand-in server for tests/unit/test_gateway.py: python gateway_stub.py PORT MODEL.

/health; /v1/models (its model, and the CUDA_VISIBLE_DEVICES it was started with); a chat
completion that echoes the model and, with "stream", sends three events; /v1/admin/gpu and
/v1/admin/release and /v1/admin/lock as server/gpushare.py answers them. It checks the gateway's key."""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, MODEL = int(sys.argv[1]), sys.argv[2]
KEY = os.environ.get("RECLANK_API_KEY", "")
LOCKS = {}      # model (or "*") -> on


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _ok(self):
        if KEY and self.headers.get("Authorization") != "Bearer " + KEY:
            self._send({"error": "key"}, 401)
            return False
        return True

    def do_GET(self):
        if self.path == "/health":
            return self._send({"status": "ok"})
        if not self._ok():
            return
        if self.path == "/v1/models":
            return self._send({"data": [{"id": MODEL, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}]})
        if self.path == "/v1/admin/gpu":
            on = bool(LOCKS.get(MODEL, LOCKS.get("*", False)))
            return self._send({"bytes": 1 << 30, "locked": (1 << 30) if on else 0, "models": {MODEL: on},
                               "busy": False})
        self._send({"error": "no"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if not self._ok():
            return
        if self.path == "/v1/admin/lock":
            LOCKS[body.get("model") or "*"] = bool(body.get("on"))
            return self._send({"status": "ok"})
        if self.path == "/v1/admin/release":
            return self._send({"status": "released", "freed": 1 << 30, "context": True, "locked": []})
        if not body.get("stream"):
            return self._send({"model": MODEL, "asked": body.get("model"),
                               "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i in range(3):
            data = ("data: " + json.dumps({"model": MODEL, "i": i}) + "\n\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
