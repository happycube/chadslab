"""reclank/setup: the model table and the fit of a model on a machine (no Hub, no GPU)."""
from __future__ import annotations

from reclank import setup as S
from reclank.setup import hardware
from reclank.setup.catalog import KINDS, MODELS, by_id


def _machine(ram, vram=0, cc=8.6, tier="avx2"):
    m = hardware.Machine(os="Linux", arch="x86_64", cpu="test", threads=8, cores=8, usable=8, tier=tier,
                         cc_compiler="/usr/bin/cc", toolkits=[("/usr/local/cuda-12/bin/nvcc", "12.9")],
                         models_dir="/models", disk_free_gb=1000.0)
    return hardware.assume(m, ram=ram, vram=vram, cc=cc if vram else None)


def test_table():
    ids = [m.id for m in MODELS]
    assert len(ids) == len(set(ids))
    for m in MODELS:
        assert m.kind in KINDS and m.status in ("ready", "manual", "not-yet") and m.server in S.PORTS
        assert m.gb > 0 and m.repo.count("/") == 1 and m.license
        if m.status == "ready":
            assert m.gpu or m.cpu, m.id
            assert "{d}" in m.args, m.id
            assert S.request_cmd(m), m.id
        if m.status == "not-yet":
            assert m.why, m.id


def test_laptop_cpu_only():
    m = _machine(16)
    assert m.gpus == []
    f = S.fit(by_id("gemma4-e4b"), m)
    assert list(f["modes"]) == ["cpu"]
    assert not S.fit(by_id("qwen38-27b"), m)["modes"]
    assert not S.fit(by_id("nemotron-bolt-e4b"), m)["modes"]
    assert S.suggest(m) is not None and S.suggest(m).kind == "chat"


def test_small_gpu():
    m = _machine(32, vram=8, cc=6.1)
    assert "gpu" in S.fit(by_id("gemma4-26b"), m)["modes"]
    assert "gpu" not in S.fit(by_id("ace-step-1.5"), m)["modes"]
    old = _machine(32, vram=8, cc=5.2)
    assert "gpu" not in S.fit(by_id("gemma4-26b"), old)["modes"]
    assert "cpu" in S.fit(by_id("gemma4-26b"), old)["modes"]


def test_commands():
    m = by_id("whisper-large-v3")
    assert "--include" in S.download_cmd(m, "/models")
    assert "--include" not in S.download_cmd(by_id("gemma4-26b"), "/models")
    assert "--revision 5.05bpw_h6_ng6" in S.download_cmd(by_id("qwen38-fn-exl3"), "/models")
    gpu = S.serve_cmd(by_id("gemma4-26b"), "/models", "gpu")
    assert "OMP_NUM_THREADS=1" in gpu and "--backend cuda" in gpu and "/models/gemma-4-26B" in gpu
    assert "OMP_NUM_THREADS" not in S.serve_cmd(by_id("gemma4-26b"), "/models", "cpu")


def test_hidden_rows():
    m = _machine(64, vram=24)
    short = S.report_text(m)
    full = S.report_text(m, all_rows=True)
    hidden = [x for x in MODELS if x.status != "ready"]
    assert hidden and all(" %s " % x.id not in short for x in hidden)
    assert all(" %s " % x.id in full for x in hidden)
    assert "more not yet supported" in short and "--all" in short
    assert "more not yet supported" not in full
    assert {d["status"] for d in S.report_json(m)["models"]} >= {"ready", "manual", "not-yet"}
    assert S.fit(by_id("ace-step-1.5"), _machine(32, vram=16))["modes"].get("gpu") is not None


def test_serve_argv():
    env, args = S.serve_argv(by_id("gemma4-26b"), "/models", "gpu")
    assert env == {"OMP_NUM_THREADS": "1"}
    assert args[:2] == ["-m", "server"] and "--backend" in args and args[-2:] == ["--port", "8080"]
    env, args = S.serve_argv(by_id("embeddinggemma2"), "/models", "cpu", port=9000)
    assert env == {} and args[:2] == ["-m", "server.embed"] and args[-2:] == ["--port", "9000"]


def test_start_args():
    from reclank.setup.start import split_args
    o, t, r = split_args(["gemma4-26b", "-d", "--ctx", "8192", "--port=9000", "--cpu"])
    assert t == "gemma4-26b" and o["d"] and o["cpu"] and o["port"] == "9000" and r == ["--ctx", "8192"]
    o, t, r = split_args(["--ctx", "8192", "gemma4-12b", "--host", "0.0.0.0", "-n"])
    assert t == "gemma4-12b" and o["host"] == "0.0.0.0" and o["n"] and r == ["--ctx", "8192"]
    o, t, r = split_args(["--stop", "--port", "8082"])
    assert t is None and o["stop"] and o["port"] == "8082" and r == []
    o, t, r = split_args(["-m", "/x/y.gguf", "--drafter", "/x/d.gguf"])
    assert t == "/x/y.gguf" and r == ["--drafter", "/x/d.gguf"]


def test_start_plan(tmp_path):
    from reclank.setup.start import plan
    m = _machine(64, vram=24)
    root = str(tmp_path)
    p = plan("gemma4-12b", m, root)
    assert p.errors and "not downloaded" in p.errors[0] and "--download" in p.errors[1]
    d = tmp_path / by_id("gemma4-12b").dir
    d.mkdir()
    for f in by_id("gemma4-12b").files:
        (d / f).write_bytes(b"")
    p = plan("gemma4-12b", m, root, rest=["--ctx", "8192"])
    assert not p.errors and p.mode == "gpu" and p.env == {"OMP_NUM_THREADS": "1"}
    assert p.args[-2:] == ["--ctx", "8192"] and "--backend" in p.args and p.port == 8080
    p = plan("gemma4-12b", _machine(64), root)
    assert not p.errors and p.mode == "cpu" and p.env == {}
    assert "no CPU mode" in plan("qwen38-27b", m, root, mode="cpu").errors[0]
    assert "not supported yet" in plan("glm53-flash", m, root).errors[0]
    assert "near ones" in plan("gemma", m, root).errors[0]
    assert plan("deepseek41", _machine(384, vram=24), root).keeper
    assert not plan("gemma4-12b", m, root).keeper
    # a file of one's own, with its mmproj beside it
    g = tmp_path / "own" / "x.gguf"
    g.parent.mkdir()
    g.write_bytes(b"")
    (g.parent / "x-mmproj.gguf").write_bytes(b"")
    p = plan(str(g), m, root, port=9001)
    assert p.args[:4] == ["-m", "server", "--model", str(g)] and p.port == 9001
    assert p.args[p.args.index("--mmproj") + 1] == str(g.parent / "x-mmproj.gguf")


def test_start_ready_lines():
    from reclank.setup.start import READY
    for line in ("reclank server http://127.0.0.1:8080/v1  model=x  key required  CORS off",
                 "reclank embedding server http://127.0.0.1:8082/v1  model=x  key required",
                 "reclank generation server http://127.0.0.1:8084/v1  speech=-",
                 "reclank gateway http://127.0.0.1:8081  (the page: /)"):
        assert READY.search(line), line
    assert not READY.search("[reclank] prompt=3 out=4 finish=stop")
