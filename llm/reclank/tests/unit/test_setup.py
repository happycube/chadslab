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
    assert "gpu" not in S.fit(by_id("wan2.2-ti2v-5b"), m)["modes"]
    old = _machine(32, vram=8, cc=5.2)
    assert "gpu" not in S.fit(by_id("gemma4-26b"), old)["modes"]
    assert "cpu" in S.fit(by_id("gemma4-26b"), old)["modes"]


def test_commands():
    m = by_id("flux2-klein")
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
    assert S.fit(by_id("wan2.2-ti2v-5b"), _machine(32, vram=16))["modes"].get("gpu") is not None
