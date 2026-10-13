"""The weightkeeper (reclank/runtime/weightkeeper.py) on a small checkpoint of the shape of
DeepSeek V4.1 Flash: the rows of each part, the dense tensors, the seals."""
import fcntl
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
L, E, HIDDEN, INNER = 2, 3, 128, 64


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


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("dsv41")
    rng = np.random.default_rng(0)
    where = {}
    for layer in range(L):
        t = {}
        for e in range(E):
            p = "layers.%d.ffn.experts.%d." % (layer, e)
            t[p + "w1.weight"] = ("I8", rng.integers(-128, 128, (INNER, HIDDEN // 2), dtype=np.int8))
            t[p + "w1.scale"] = ("F8_E8M0", rng.integers(0, 256, (INNER, HIDDEN // 32), dtype=np.uint8))
            t[p + "w3.weight"] = ("I8", rng.integers(-128, 128, (INNER, HIDDEN // 2), dtype=np.int8))
            t[p + "w3.scale"] = ("F8_E8M0", rng.integers(0, 256, (INNER, HIDDEN // 32), dtype=np.uint8))
            t[p + "w2.weight"] = ("I8", rng.integers(-128, 128, (HIDDEN, INNER // 2), dtype=np.int8))
            t[p + "w2.scale"] = ("F8_E8M0", rng.integers(0, 256, (HIDDEN, INNER // 32), dtype=np.uint8))
        t["layers.%d.attn.wkv.weight" % layer] = ("F8_E4M3", rng.integers(0, 256, (16, HIDDEN), dtype=np.uint8))
        if layer == 1:
            t["layers.1.engram.embed.weight"] = ("F8_E4M3", rng.integers(0, 256, (50, 32), dtype=np.uint8))
        f = "model-%05d-of-%05d.safetensors" % (layer + 1, L)
        _write_shard(d / f, t)
        where.update({k: f for k in t})
    json.dump({"weight_map": where}, open(d / "model.safetensors.index.json", "w"))
    json.dump({"text_config": {"num_hidden_layers": L, "n_routed_experts": E, "hidden_size": HIDDEN,
                               "moe_intermediate_size": INNER}}, open(d / "config.json", "w"))
    return str(d)


def test_layout(ckpt):
    sh = Z.Shards(ckpt)
    nodes = [0, 1]
    ix = Z.expert_layout(sh, nodes, [0.5, 0.5], range(L), "planes")
    assert ix["gu"] == [[0, 32], [32, 32]] and ix["dn"] == [[0, 64], [64, 64]]
    for part in range(2):
        offs = [off for _p, off, _r0, _nr, _rb in ix["planes"][part]]
        assert all(o % Z.ALIGN == 0 for o in offs)
    v2 = Z.expert_layout(sh, nodes, [0.5, 0.5], range(L))
    assert v2["layout"] == Z.LAYOUT and v2["stride"] == ix["stride"]      # the same bytes, packed
    for part in range(2):
        assert all(off % Z.ALIGN == 0 for _w, off, _r0, _nr, _c, _n in v2["mats"][part])
    assert "layers.1.engram.embed.weight" not in Z.dense_names(sh)
    assert "layers.0.attn.wkv.weight" in Z.dense_names(sh)


@pytest.mark.parametrize("layout", ["planes", "mxx"])
def test_serve_attach(ckpt, tmp_path, layout):
    sock = str(tmp_path / "z.sock")
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "scripts", "weightkeeper.py"), ckpt,
                          "--pages", "4k", "--dense", "--threads", "4", "--socket", sock, "--layout", layout],
                         env=env, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(300):
            if os.path.exists(sock) or p.poll() is not None:
                break
            time.sleep(0.1)
        assert p.poll() is None, p.stderr.read()
        w = Z.attach(sock)
        w.check(ckpt)
        sh = Z.Shards(ckpt)
        ix = w.index
        if layout == "mxx":
            from reclank.backends.cpu import cops
            for layer in range(L):
                for part in range(len(ix["segments"])):
                    ms = w.mats(layer, part)
                    for name, _off, r0, nr, cols, _n in ix["mats"][part]:
                        for e in range(E):
                            base = Z.EXPERT % (layer, e, name)
                            f, _dt, shp, o0, o1 = sh.t[base + ".weight"]
                            sf, _sdt, sshp, s0, s1 = sh.t[base + ".scale"]
                            c = np.fromfile(sh.file(f), np.uint8, o1 - o0, offset=o0).reshape(shp)[r0:r0 + nr]
                            sc = np.fromfile(sh.file(sf), np.uint8, s1 - s0, offset=s0).reshape(sshp)[r0:r0 + nr]
                            ref, _odd = cops.kq_mxx_pack(c, sc, nr, cols)
                            assert ms[name][e].tobytes() == ref.tobytes(), (layer, part, name, e)
            pl = None
        for layer in range(L if layout == "planes" else 0):
            for part in range(len(ix["segments"])):
                pl = w.planes(layer, part)
                for name, _off, r0, nr, rb in ix["planes"][part]:
                    for e in range(E):
                        f, _dt, _shape, o0, _o1 = sh.t[Z.EXPERT % (layer, e, name)]
                        with open(sh.file(f), "rb") as fh:
                            fh.seek(o0 + r0 * rb)
                            assert pl[name][e].tobytes() == fh.read(nr * rb), (layer, part, name, e)
        f, _dt, _shape, o0, o1 = sh.t["layers.0.attn.wkv.weight"]
        with open(sh.file(f), "rb") as fh:
            fh.seek(o0)
            assert w.dense("layers.0.attn.wkv.weight").tobytes() == fh.read(o1 - o0)
        if pl is not None:
            with pytest.raises(ValueError):
                pl[name][0, 0, 0] = 1
        c = Z.request("status", sock)
        with c:
            st = json.loads(Z._recv_msg(c))
        assert st["layers"] == L and not st["locked"]
    finally:
        try:
            Z.request("stop", sock).close()
        except OSError:
            pass
        p.wait(timeout=30)
    assert not os.path.exists(sock)


def test_seals(tmp_path):
    seg = Z.Segment("t", 0, [(Z.K4, Z.M2)], numa_on=False)
    seg.array(0, 16)[:] = 7
    seg.seal()
    assert fcntl.fcntl(seg.fds[0], Z.F_GET_SEALS) & Z.SEALS == Z.SEALS
    with pytest.raises(OSError):
        os.ftruncate(seg.fds[0], 2 * Z.M2)
    with pytest.raises(OSError):
        Z._mmap(None, Z.M2, Z.PROT_READ | Z.PROT_WRITE, Z.MAP_SHARED, seg.fds[0])
