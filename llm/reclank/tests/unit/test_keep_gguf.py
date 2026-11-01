"""The weightkeeper of a GGUF file (reclank/models/qwen/keep.py, layout_gguf and KeptGGUF) on a small
file of the shape of Gemma 4: the experts held (as KQ_Q4X), the dense and small tensors in the file
unless --dense; and the KQ_Q4X copies of the dense Q4_0 matrices that the loader makes at their first
read (ops.q4x_pack_model)."""
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from reclank.core import ops
from reclank.formats.gguf import F32, GGUF, Q4_0, Q4X, write_gguf
from reclank.models.qwen import keep
from reclank.runtime import weightkeeper as Z

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
HIDDEN, ROWS, E = 64, 64, 2
DENSE, STACK, NORM = "blk.0.attn_q.weight", "blk.0.ffn_gate_up_exps.weight", "blk.0.attn_norm.weight"


def _q40(rng, rows, cols):
    """Q4_0 blocks (rows, cols / 32, 18) with float16 scales of a normal size."""
    b = rng.integers(0, 256, (rows, cols // 32, 18), dtype=np.uint8)
    b[..., :2] = rng.uniform(0.01, 0.1, (rows, cols // 32)).astype(np.float16)[..., None].view(np.uint8)
    return b


def _pack(blocks):
    scales = np.ascontiguousarray(blocks[..., :2]).view(np.float16)[..., 0].astype(np.float32)
    return ops._cops.kq_q4x_pack(np.ascontiguousarray(blocks), scales)


@pytest.fixture(scope="module")
def gguf(tmp_path_factory):
    rng = np.random.default_rng(0)
    data = {DENSE: _q40(rng, 16, HIDDEN), STACK: _q40(rng, E * ROWS, HIDDEN),
            NORM: rng.standard_normal(HIDDEN).astype(np.float32),
            "token_embd.weight": rng.standard_normal((8, HIDDEN)).astype(np.float32)}
    tensors = [(DENSE, (HIDDEN, 16), Q4_0), (STACK, (HIDDEN, ROWS, E), Q4_0), (NORM, (HIDDEN,), F32),
               ("token_embd.weight", (HIDDEN, 8), F32)]
    meta = [("general.architecture", "gemma4"), ("gemma4.block_count", 1), ("gemma4.expert_count", E),
            ("gemma4.embedding_length", HIDDEN), ("gemma4.expert_feed_forward_length", ROWS // 2)]
    path = str(tmp_path_factory.mktemp("gemma4") / "tiny.gguf")
    write_gguf(path, meta, [(n, d, t, (lambda n=n: data[n])) for n, d, t in tensors])
    return path, data


def _serve(path, sock, *args):
    # one node (--nodes 0): a machine of two sockets would split the stack's experts between them, which
    # KeptGGUF.raw() does not join
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "scripts", "weightkeeper.py"), path,
                          "--pages", "4k", "--threads", "2", "--nodes", "0", "--socket", sock] + list(args),
                         env=dict(os.environ, PYTHONPATH=ROOT), stderr=subprocess.PIPE, text=True)
    for _ in range(300):
        if os.path.exists(sock) or p.poll() is not None:
            break
        time.sleep(0.1)
    assert p.poll() is None, p.stderr.read()
    return p


@pytest.mark.skipif(not ops._COPS_READY, reason="the C library")
@pytest.mark.parametrize("dense", [False, True])
def test_kept_gguf(gguf, tmp_path, dense):
    path, data = gguf
    sock = str(tmp_path / "k.sock")
    p = _serve(path, sock, *(["--dense"] if dense else []))
    try:
        g = keep.KeptGGUF(path, Z.attach(sock))
        f = GGUF(path, stage=False)
        tens = g.index["tensors"]
        assert tens[STACK]["place"] == "experts" and tens[STACK]["type"] == Q4X
        assert g.raw(STACK)[0].tobytes() == _pack(data[STACK]).tobytes()
        assert tens[DENSE]["place"] == ("node" if dense else "file")
        assert tens[NORM]["place"] == ("node" if dense else "file")
        assert np.array_equal(g.dequant(NORM), f.dequant(NORM))
        packed, s = g.int4_packed(g._name(DENSE))
        if dense:
            # held as KQ_Q4X, its only copy
            assert s is None and packed.tobytes() == _pack(data[DENSE]).tobytes()
        else:
            # read in the file as it is: Q4_0 and its scales
            assert tens[DENSE]["type"] == Q4_0 and g.index["types"]["dense"] == "Q4_0 in the file"
            assert packed.tobytes() == data[DENSE].tobytes()
            assert np.array_equal(s, data[DENSE][..., :2].copy().view(np.float16)[..., 0].astype(np.float32))
        g.close()
        f.close()
    finally:
        try:
            Z.request("stop", sock).close()
        except OSError:
            pass
        p.wait(timeout=30)


@pytest.mark.skipif(not ops._COPS_READY, reason="the C library")
def test_dense_copy_at_first_read(gguf):
    """The KQ_Q4X experts (the weightkeeper's) and a dense Q4_0 matrix (the file's): the model is
    q4x_only; the copy of the dense matrix is made when a reader asks for it, and is its KQ_Q4X."""
    _path, data = gguf
    q = np.ascontiguousarray(data[DENSE])
    s = np.ascontiguousarray(q[..., :2]).view(np.float16)[..., 0].astype(np.float32)
    gu = _pack(data[STACK]).reshape(E, ROWS, HIDDEN // 32, 18)
    dn = _pack(_q40(np.random.default_rng(1), E * HIDDEN, ROWS // 2)).reshape(E, HIDDEN, ROWS // 2 // 32, 18)

    class M:
        _layers = {0: {"self_attn.q_proj": (q, s), "experts.gate_up_proj": (gu, None),
                       "experts.down_proj": (dn, None)}}
    keys = [q.ctypes.data, gu.ctypes.data]
    try:
        ops.q4x_pack_model(M)
        assert M._q4x_only
        assert dict.get(ops._Q4X, q.ctypes.data) is None and q.ctypes.data in ops._Q4X.todo
        assert ops._Q4X
        got = ops._Q4X.get(q.ctypes.data)
        assert got.tobytes() == _pack(data[DENSE]).tobytes()
        assert ops._Q4X.get(q.ctypes.data) is got and q.ctypes.data not in ops._Q4X.todo
        assert q.ctypes.data not in ops._Q4X_SELF
    finally:
        # (the tables are by address: leave none behind for another test)
        for k in keys:
            ops._Q4X.pop(k, None)
            ops._Q4X.todo.pop(k, None)
            ops._Q4X_MOE.pop(k, None)
            ops._Q4X_SELF.pop(k, None)
