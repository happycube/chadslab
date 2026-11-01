"""Wan 2.2 T2V-A14B in the weightkeeper (reclank/runtime/weightkeeper.py, the tensors layout): both
experts' DiT weights in the forms WanDiT makes at its start, made once from the float32 shards and
handed to every server or check (models/wan/dit.py keeper=).

- each matrix WanDiT makes a linear of, of both transformers (transformer/, transformer_2/: the
  blocks' 10, the text embedder's 2, proj_out, the patch embedding's conv as its (out, in k)
  matrix) in F20 rows (encoders.runtime.to_f20's conversion and
  rule: cops.kq_f32_to_f20, the same bytes; a matrix the rule refuses stops the keeper), a block's
  10 matrices side by side in the order of dit.MATS, each at a multiple of dit.ALIGN (the layout
  of WanDiT._pack, so the server's copies read the keeper's rows in place);
- every other tensor (biases, norms, scale_shift_tables, the time embedder) its float32 bytes.

One segment, on --dense-node (default 0), sealed for size only: a round maps it writable and
registers it with CUDA (the copies of the streamed blocks: the pull kernel). 71.4 GB for both
experts. The shards are read with O_DIRECT (no page cache).

    python scripts/weightkeeper.py /space/models/Wan2.2-T2V-A14B-Diffusers --name wan-a14b
    WanT2V(path)        # takes a keeper that holds this checkpoint (any name), else loads
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time

import numpy as np

MODEL = "wan-a14b"
LAYOUT = "tensors-v1"
SUBS = ("transformer", "transformer_2")
INDEX = "diffusion_pytorch_model.safetensors.index.json"


def detect(path):
    """A diffusers directory of WanPipeline with two experts (boundary_ratio)."""
    try:
        mi = json.load(open(os.path.join(path, "model_index.json")))
    except (OSError, ValueError):
        return False
    return mi.get("_class_name") == "WanPipeline" and mi.get("boundary_ratio") is not None and \
        all(os.path.isfile(os.path.join(path, s, INDEX)) for s in SUBS)


def _headers(path):
    """{sub/file: (header dict, data start)} of the experts' shards."""
    out = {}
    for sub in SUBS:
        where = json.load(open(os.path.join(path, sub, INDEX)))["weight_map"]
        for f in sorted(set(where.values())):
            with open(os.path.join(path, sub, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            out["%s/%s" % (sub, f)] = (h, 8 + n)
    return out


def ident(path):
    """The size and the sha256 of the header of each shard of both experts (Weights.check)."""
    out = {}
    for sub in SUBS:
        where = json.load(open(os.path.join(path, sub, INDEX)))["weight_map"]
        for f in sorted(set(where.values())):
            p = os.path.join(path, sub, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                out["%s/%s" % (sub, f)] = [os.path.getsize(p), hashlib.sha256(fh.read(n)).hexdigest()]
    return out


# the weights WanDiT makes linears of (WanDiT.linear) beside the blocks' MATS; the rest (the time
# embedder and time_proj too: WanDiT's host vectors read them as float32) stay float32
LINEARS = ("patch_embedding.weight", "condition_embedder.text_embedder.linear_1.weight",
           "condition_embedder.text_embedder.linear_2.weight", "proj_out.weight")


def layout(path, nodes, share, o):
    """The index: "<sub>/<name>" of each tensor: kind "f20" (rows, cols: the F20 rows) or "f32"
    (shape: float32 as stored), place node, seg 0, off, bytes."""
    from reclank.backends.cpu import cops
    from reclank.models.wan.dit import ALIGN, MATS
    node = o.dense_node if o.dense_node in nodes else nodes[0]
    used = [0]
    tens, order = {}, []

    def put(key, nbytes, align, **kw):
        off = (used[0] + align - 1) // align * align
        used[0] = off + nbytes
        tens[key] = dict(place="node", seg=0, off=off, bytes=nbytes, **kw)
        order.append(key)
    blockmats = {"q": "attn1.to_q", "k": "attn1.to_k", "v": "attn1.to_v", "o": "attn1.to_out.0",
                 "q2": "attn2.to_q", "k2": "attn2.to_k", "v2": "attn2.to_v", "o2": "attn2.to_out.0",
                 "up": "ffn.net.0.proj", "down": "ffn.net.2"}
    for sub in SUBS:
        cfg = json.load(open(os.path.join(path, sub, "config.json")))
        shapes = {}
        for f, (h, _s) in _headers(path).items():
            if f.startswith(sub + "/"):
                for n, t in h.items():
                    assert t["dtype"] == "F32", (n, t["dtype"])
                    shapes[n] = (f, tuple(t["shape"]))

        def mat(name):
            shape = shapes[name][1]
            rows, cols = shape[0], int(np.prod(shape[1:]))
            assert cols % 32 == 0, (name, shape)
            return rows, cols
        done = set()
        # the blocks' matrices: block after block, in the order of WanDiT._pack
        for i in range(cfg["num_layers"]):
            for k, m in enumerate(MATS):
                n = "blocks.%d.%s.weight" % (i, blockmats[m])
                rows, cols = mat(n)
                put("%s/%s" % (sub, n), rows * cops.f20_row_bytes(cols), ALIGN, kind="f20", rows=rows, cols=cols)
                done.add(n)
        for n, (f, shape) in sorted(shapes.items()):
            if n in done:
                continue
            if n in LINEARS:
                rows, cols = mat(n)
                put("%s/%s" % (sub, n), rows * cops.f20_row_bytes(cols), 64, kind="f20", rows=rows, cols=cols)
            else:
                put("%s/%s" % (sub, n), int(np.prod(shape)) * 4, 64, kind="f32", shape=list(shape))
    return dict(layout=LAYOUT, model=MODEL, source="wan", file=os.path.abspath(path), nodes=[node],
                share=[1.0], types=dict(dense="F20 rows (the matrices), float32 (the rest)", experts="none",
                                        ngram="none"),
                tensors=tens, order=order, meta=[], part_bytes=[used[0]])


def fill(sh, index, segs, threads, log):
    """The tensors from the shards (O_DIRECT, in the order of each file), each matrix converted to
    F20 by to_f20's function and rule."""
    from reclank.backends.cpu import cops
    from reclank.runtime.weightkeeper import Reader
    path = index["file"]
    items = []
    for f, (h, start) in _headers(path).items():
        sub = f.split("/")[0]
        for n, t in h.items():
            o0, o1 = t["data_offsets"]
            items.append((f, start + o0, start + o1, "%s/%s" % (sub, n)))
    items.sort()
    missing = set(index["tensors"]) - {it[3] for it in items}
    assert not missing, "tensors of the index not in the shards: %s" % sorted(missing)[:4]

    class _Files:
        def file(self, f):
            return os.path.join(path, f)
    rd = Reader(_Files(), max(it[2] - it[1] for it in items))
    seg = segs[0]
    lock = threading.Lock()
    pos, done, errs = [0], [0], []
    total = sum(it[2] - it[1] for it in items)
    t0 = time.time()

    def work():
        try:
            while True:
                with lock:
                    if pos[0] >= len(items) or errs:
                        return
                    f, o0, o1, key = items[pos[0]]
                    pos[0] += 1
                src = rd.read(f, o0, o1).view(np.float32)
                d = index["tensors"][key]
                dst = seg.array(d["off"], d["bytes"])
                if d["kind"] == "f20":
                    data, rep = cops.kq_f32_to_f20(src.reshape(d["rows"], d["cols"]), d["cols"])
                    if rep["nonfinite"] or rep["largest"] > rep["rms"] / 256:
                        raise ValueError("%s: F20's rule refuses it (%s)" % (key, rep))
                    dst[:] = data.reshape(-1)
                else:
                    dst[:] = src.view(np.uint8)
                with lock:
                    done[0] += o1 - o0
        except BaseException as e:      # noqa: BLE001 - reported by the caller
            errs.append(e)
    ts = [threading.Thread(target=work, daemon=True) for _ in range(max(1, min(2, threads)))]
    [t.start() for t in ts]
    last = t0
    while any(t.is_alive() for t in ts):
        time.sleep(0.5)
        if time.time() - last >= 30:
            last = time.time()
            log("read %.1f of %.1f GB (%.0f MB/s)" % (done[0] / 1e9, total / 1e9, done[0] / 1e6 / (last - t0)))
    rd.close()
    if errs:
        raise errs[0]
    log("the experts: %.1f GB of float32 read and converted in %.0f s (%.0f MB/s)" % (
        total / 1e9, time.time() - t0, total / 1e6 / max(time.time() - t0, 1e-9)))


class _Plugin:
    layout = staticmethod(layout)
    fill = staticmethod(fill)
    ident = staticmethod(ident)


PLUGIN = _Plugin()


class KeptExpert:
    """One transformer (sub) of a weightkeeper's segments as WanDiT reads its weights: get(name)
    (float32, read-only), linear(name) (F20 rows, or float32 W), segments()."""

    def __init__(self, w, sub):
        assert w.index.get("model") == MODEL, w.index.get("model")
        self.w, self.sub = w, sub
        self.t = w.index["tensors"]

    def _view(self, name):
        d = self.t["%s/%s" % (self.sub, name)]
        return d, self.w.tensor("%s/%s" % (self.sub, name))

    def get(self, name):
        d, a = self._view(name)
        assert d["kind"] == "f32", name
        return a.view(np.float32).reshape(d["shape"])

    def linear(self, name):
        from reclank.backends.cpu import cops
        from reclank.encoders.runtime import _Plain
        from reclank.models.gemma4.e4b import _KQ
        d, a = self._view(name + ".weight")
        assert d["kind"] == "f20", name
        lin = _Plain(None)
        lin.f20 = _KQ(a.reshape(d["rows"], -1), cops.KQ_F20, d["rows"], d["cols"])
        lin.shape = (d["rows"], d["cols"])
        lin.b = np.ascontiguousarray(self.get(name + ".bias"))
        return lin

    def segments(self):
        """(address, bytes) of each segment (to register with CUDA)."""
        return list(zip(self.w.base, self.w.size))


def attach(path, name=None):
    """The weightkeeper's segments when one holds this checkpoint (name: that keeper's, else any of
    this user's), else None."""
    try:
        from reclank.runtime import weightkeeper as W
        sock = W.socket_path(name) if name else None
        w = W.attach(sock, ckpt=path)
    except (OSError, ValueError):
        return None
    if w.index.get("model") != MODEL:
        return None
    return w
