"""The gateway (server/gateway.py): the model of a request, the room by host memory, and a session
against two stand-in servers (tests/unit/gateway_stub.py): each started by its model's first
request, a stream passed on as it comes, one stopped for the other's room, a change of GPUs."""
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from server import gateway as G

STUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway_stub.py")


def test_request_model_json_and_multipart():
    assert G.request_model(b'{"model": "a", "messages": []}', "application/json") == "a"
    assert G.request_model(b'{"messages": []}', "application/json") is None
    assert G.request_model(b"", "application/json") is None
    body = (b'--XyZ\r\nContent-Disposition: form-data; name="image"; filename="model.png"\r\n\r\nPNG...\r\n'
            b'--XyZ\r\nContent-Disposition: form-data; name="model"\r\n\r\nflux2-klein\r\n--XyZ--\r\n')
    assert G.request_model(body, "multipart/form-data; boundary=XyZ") == "flux2-klein"


def test_path_kind():
    assert G.path_kind("/v1/chat/completions") == "llm"
    assert G.path_kind("/v1/completions") == "llm"
    assert G.path_kind("/v1/embeddings") == "embed"
    assert G.path_kind("/v1/images/generations") == "generate"
    assert G.path_kind("/v1/videos/abc") == "generate"
    assert G.path_kind("/v1/other") is None


class _S:
    def __init__(self, sid, ram, last, state="ready", keep=False, inflight=0, kind="llm", requires=()):
        self.id, self.ram_gb, self.last_used, self.state = sid, ram, last, state
        self.keep, self.inflight = keep, inflight
        self.kind, self.requires, self.locked = kind, list(requires), set()

    @property
    def held(self):
        return self.keep or bool(self.locked)


def test_room_victims_least_recently_used_first():
    a, b, c = _S("a", 100, 3), _S("b", 100, 1), _S("c", 100, 2)
    assert G.room_victims([a, b, c], 50, 400) == []
    assert G.room_victims([a, b, c], 150, 400) == [b]
    assert G.room_victims([a, b, c], 250, 400) == [b, c]
    b.keep = True
    c.inflight = 1
    assert G.room_victims([a, b, c], 150, 400) == [a]
    assert G.room_victims([a, b, c], 250, 400) is None
    b.keep = False
    b.locked = {"*"}        # a locked model's server is not stopped either
    assert G.room_victims([a, b, c], 250, 400) is None


def test_room_victims_a_service_goes_with_its_last_server():
    k = _S("keeper", 200, 0, kind="service")
    d1, d2 = _S("d1", 20, 5, requires=["keeper"]), _S("d2", 20, 1, requires=["keeper"])
    x = _S("x", 50, 9)
    new = _S("new", 150, 10)
    # d2, then d1 (the keeper's last user), then the keeper
    assert G.room_victims([k, d1, d2, x], 150, 300) == [d2, d1, k]
    # a server that needs the keeper does not free it
    new.requires = ["keeper"]
    assert G.room_victims([k, d1, d2, x], 30, 300) == [d2]
    # a service nobody uses goes first
    d1.state = d2.state = "stopped"
    assert G.room_victims([k, d1, d2, x], 150, 300) == [k]


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _req(url, path, body=None, key="k"):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read()


@pytest.fixture
def gateway(tmp_path):
    pa, pb = _port(), _port()
    cfg = {"host_ram_gb": 150, "log_dir": str(tmp_path / "logs"), "backend_key": "inner",
           "default": {"llm": "model-a"},
           "servers": [
               {"id": "a", "kind": "llm", "models": ["model-a"], "port": pa, "ram_gb": 100, "cpu": True,
                "cmd": [sys.executable, STUB, "{port}", "model-a"], "start_timeout": 30},
               {"id": "b", "kind": "llm", "port": pb, "ram_gb": 100,
                "cmd": [sys.executable, STUB, "{port}", "model-b"], "start_timeout": 30}]}
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(cfg))
    gw = G.Gateway(str(path), api_key="k")
    httpd = G.make_server(gw, port=0, api_key="k")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % httpd.server_address[1]
    yield gw, url
    httpd.shutdown()
    gw.stop_all()


def test_session(gateway):
    gw, url = gateway
    # no key: 401; the page and /health need none
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(url, "/v1/models", key="wrong")
    assert e.value.code == 401
    assert _req(url, "/health", key="")[0] == 200
    assert b"reclank models" in _req(url, "/", key="")[1]
    # a's first request starts it (the gateway's own key passed on as the inner one)
    st, body = _req(url, "/v1/chat/completions", {"model": "model-a", "messages": [1]})
    assert st == 200 and json.loads(body)["model"] == "model-a"
    assert gw.servers["a"].state == "ready" and gw.servers["b"].state == "stopped"
    # a stream passes on as events
    st, body = _req(url, "/v1/chat/completions", {"model": "model-a", "messages": [1], "stream": True})
    assert body.count(b"data: ") == 3
    # no model: the kind's default
    assert json.loads(_req(url, "/v1/chat/completions", {"messages": [1]})[1])["model"] == "model-a"
    # b's models are not known before it runs; started by hand, it is asked; a goes for its room
    gw.action("b", "start", {})
    assert gw.servers["b"].models == ["model-b"]
    assert gw.servers["a"].state == "stopped"
    ids = [m["id"] for m in json.loads(_req(url, "/v1/models")[1])["data"]]
    assert ids == ["model-a", "model-b"]
    # back to a: b goes
    assert json.loads(_req(url, "/v1/chat/completions", {"model": "model-a", "messages": [1]})[1])["model"] == "model-a"
    assert gw.servers["b"].state == "stopped"
    # an unknown model: 404
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(url, "/v1/chat/completions", {"model": "nope", "messages": [1]})
    assert e.value.code == 404
    # the page's actions: release, a change of GPUs (a restart when idle), keep
    assert json.loads(_req(url, "/gateway/servers/a/release", {})[1])["status"] == "released"
    # the servers' own release route on the gateway: every running server's
    assert json.loads(_req(url, "/v1/admin/release", {})[1])["a"]["status"] == "released"
    assert json.loads(_req(url, "/v1/admin/gpu")[1])["a"]["bytes"] == 1 << 30
    _req(url, "/gateway/servers/a/gpus", {"gpus": "cpu"})
    for _ in range(100):
        if gw.servers["a"].state == "ready" and not gw.servers["a"].pending_restart:
            break
        time.sleep(0.1)
    out = json.loads(_req(url, "/v1/chat/completions", {"model": "model-a", "messages": [1]})[1])
    assert out["cuda"] == ""
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(url, "/gateway/servers/b/gpus", {"gpus": "cpu"})
    assert e.value.code == 400
    # a lock: kept with the server's state, applied on the server, and again after a restart
    _req(url, "/gateway/servers/a/lock", {"model": "model-a", "on": True})
    assert gw.servers["a"].locked == {"model-a"}
    assert json.loads(_req(url, "/gateway/state")[1])["servers"][0]["gpu"]["models"]["model-a"] is True
    gw.action("a", "restart", {})
    assert json.loads(_req(url, "/gateway/state")[1])["servers"][0]["gpu"]["models"]["model-a"] is True
    _req(url, "/gateway/servers/a/unlock", {"model": "model-a"})
    assert gw.servers["a"].locked == set()
    _req(url, "/gateway/servers/a/keep", {"on": True})
    st = json.loads(_req(url, "/gateway/state")[1])
    a = [s for s in st["servers"] if s["id"] == "a"][0]
    assert a["keep"] and a["gpus"] == "cpu" and a["gpu"]["bytes"] == 1 << 30
    # kept: b cannot have a's room
    with pytest.raises(urllib.error.HTTPError) as e:
        _req(url, "/v1/chat/completions", {"model": "model-b", "messages": [1]})
    assert e.value.code == 503
    # the state file keeps the page's changes and b's models
    saved = json.load(open(gw.state_path))["servers"]
    assert saved["a"]["gpus"] == "cpu" and saved["a"]["keep"] and saved["b"]["models"] == ["model-b"]
    assert saved["a"]["locked"] == []
