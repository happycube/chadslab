"""The Wan A14B weightkeeper (reclank/models/wan/keep.py) on a small checkpoint of the shape of its
two experts: the keeper's bytes (F20 rows of to_f20's function, float32 as stored), the blocks'
layout that WanDiT's streamed blocks read in place, a named keeper beside the default one."""
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from reclank.runtime import weightkeeper as Z

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
WK = os.path.join(ROOT, "scripts", "weightkeeper.py")


def _write_shard(path, tensors):
    hdr, off, blobs = {}, 0, []
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


W, FF, HEADS, LAYERS, TXT, FREQ, IN = 64, 96, 2, 2, 32, 32, 16


def _wan_expert(d, seed):
    rng = np.random.default_rng(seed)
    f32 = lambda *s: ("F32", rng.standard_normal(s).astype(np.float32) * 0.05)  # noqa: E731
    t = {"patch_embedding.weight": f32(W, IN, 1, 2, 2), "patch_embedding.bias": f32(W),
         "scale_shift_table": f32(1, 2, W), "proj_out.weight": f32(4 * IN, W), "proj_out.bias": f32(4 * IN)}
    e = "condition_embedder."
    for n, (o, i) in (("time_embedder.linear_1", (W, FREQ)), ("time_embedder.linear_2", (W, W)),
                      ("time_proj", (6 * W, W)), ("text_embedder.linear_1", (W, TXT)),
                      ("text_embedder.linear_2", (W, W))):
        t[e + n + ".weight"], t[e + n + ".bias"] = f32(o, i), f32(o)
    for b in range(LAYERS):
        p = "blocks.%d." % b
        for a in ("attn1.", "attn2."):
            for n in ("to_q", "to_k", "to_v", "to_out.0"):
                t[p + a + n + ".weight"], t[p + a + n + ".bias"] = f32(W, W), f32(W)
            t[p + a + "norm_q.weight"], t[p + a + "norm_k.weight"] = f32(W), f32(W)
        t[p + "norm2.weight"], t[p + "norm2.bias"] = f32(W), f32(W)
        t[p + "ffn.net.0.proj.weight"], t[p + "ffn.net.0.proj.bias"] = f32(FF, W), f32(FF)
        t[p + "ffn.net.2.weight"], t[p + "ffn.net.2.bias"] = f32(W, FF), f32(W)
        t[p + "scale_shift_table"] = f32(1, 6, W)
    os.makedirs(d, exist_ok=True)
    names = sorted(t)
    files = {}
    for k, part in enumerate((names[:len(names) // 2], names[len(names) // 2:])):
        f = "diffusion_pytorch_model-%05d-of-00002.safetensors" % (k + 1)
        _write_shard(os.path.join(d, f), {n: t[n] for n in part})
        files.update({n: f for n in part})
    json.dump({"weight_map": files}, open(os.path.join(d, "diffusion_pytorch_model.safetensors.index.json"), "w"))
    json.dump({"patch_size": [1, 2, 2], "qk_norm": "rms_norm_across_heads", "cross_attn_norm": True,
               "image_dim": None, "added_kv_proj_dim": None, "num_attention_heads": HEADS,
               "attention_head_dim": W // HEADS, "ffn_dim": FF, "eps": 1e-6, "in_channels": IN, "out_channels": IN,
               "text_dim": TXT, "freq_dim": FREQ, "num_layers": LAYERS}, open(os.path.join(d, "config.json"), "w"))
    return t


@pytest.fixture(scope="module")
def wan(tmp_path_factory):
    d = tmp_path_factory.mktemp("wan")
    json.dump({"_class_name": "WanPipeline", "boundary_ratio": 0.875}, open(d / "model_index.json", "w"))
    ts = [_wan_expert(str(d / s), k) for k, s in enumerate(("transformer", "transformer_2"))]
    return str(d), ts


@pytest.fixture
def sockdir(tmp_path, monkeypatch):
    d = tmp_path / "socks"
    d.mkdir()
    monkeypatch.setenv("RECLANK_WEIGHTKEEPER_DIR", str(d))
    monkeypatch.delenv("RECLANK_WEIGHTKEEPER", raising=False)
    return str(d)


def test_wan_keeper(wan, sockdir):
    from reclank.backends.cpu import cops
    from reclank.models.wan import keep
    from reclank.models.wan.dit import WanDiT
    env = dict(os.environ, PYTHONPATH=ROOT)
    assert keep.detect(wan[0])
    p = subprocess.Popen([sys.executable, WK, wan[0], "--pages", "4k", "--threads", "2", "--name", "wan"], env=env,
                         stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(1200):
            if os.path.exists(Z.socket_path("wan")) or p.poll() is not None:
                break
            time.sleep(0.1)
        assert p.poll() is None, p.stderr.read()
        assert os.path.exists(Z.socket_path("wan")), "the keeper did not serve within 120 s"
        assert Z.find(wan[0])[0] == Z.socket_path("wan")
        w = keep.attach(wan[0])
        assert w is not None and keep.attach(wan[0], name="other") is None
        for k, sub in enumerate(keep.SUBS):
            ke = keep.KeptExpert(w, sub)
            for name, (_dt, arr) in wan[1][k].items():
                d = w.index["tensors"]["%s/%s" % (sub, name)]
                if d["kind"] == "f20":
                    ref = cops.kq_f32_to_f20(arr.reshape(arr.shape[0], -1), d["cols"])[0]
                    assert ke.linear(name[:-len(".weight")]).f20.data.tobytes() == ref.tobytes(), name
                else:
                    assert np.array_equal(ke.get(name), arr), name
            # the time embedder stays float32 (WanDiT's host vectors)
            assert w.index["tensors"]["%s/condition_embedder.time_proj.weight" % sub]["kind"] == "f32"
            # WanDiT on the keeper's rows (no GPU work): its streamed blocks read the keeper's in place
            dit = WanDiT(os.path.join(wan[0], sub), gpu=True, stream=True, keeper=ke, log=lambda m: None)
            assert len(dit.packed) == LAYERS and all(v[2] is None for v in dit.packed.values())
        r = subprocess.run([sys.executable, WK, "--check", wan[0], "--check-n", "8"], env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0 and "all equal" in r.stdout, r.stdout + r.stderr
    finally:
        # (stopped on its socket, else killed: never left behind)
        try:
            Z.request("stop", Z.socket_path("wan")).close()
        except OSError:
            pass
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
