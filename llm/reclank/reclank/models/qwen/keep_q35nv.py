"""An NVFP4 checkpoint of Qwen3.5 / Qwen3.6 MoE of NVIDIA ModelOpt (nvidia/Qwen3.6-35B-A3B-NVFP4;
reclank/formats/q35nv_st.py) in the weightkeeper (reclank/runtime/weightkeeper.py; the tensors layout
of reclank/models/qwen/keep.py, the model qwen35moe that reclank/models/qwen/qwen35.py reads).

The tensors are those of Q35NVSource, in its forms:

- the routed experts: KQ_NVX stacks, whole on every node when they fit (--copies auto, as
  keep_exl3), so the CPU parts of each node read their rows in their own memory; else split between
  the nodes by expert;
- the rest on the GPU's node as Q35NVSource.raw makes it (the FP8 and NVFP4 matrices as their
  float32 values, the embeddings bfloat16, the small tensors float32).

    python scripts/weightkeeper.py models/Qwen3.6-35B-A3B-NVFP4
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from reclank.formats.gguf import tensor_bytes
from reclank.models.qwen.keep import LAYOUT, KeptGGUF, _dense_ok, _homes, _up
from reclank.models.qwen.keep_exl3 import copies_mode
from reclank.models.qwen.keep_fp4l import _pool

MODEL = "qwen35moe"


def detect(path):
    from reclank.formats.q35nv_st import is_q35nv
    return os.path.isdir(path) and is_q35nv(path)


def _source(path):
    from reclank.formats.q35nv_st import Q35NVSource
    return Q35NVSource(path)


def layout(path, nodes, share, o):
    """The index of the tensors of the checkpoint in the weightkeeper."""
    src = _source(path)
    cfg = src.config()
    parts = len(nodes)
    dpart = nodes.index(o.dense_node) if o.dense_node in nodes else 0
    counts = np.load(o.counts) if getattr(o, "counts", None) else None
    stacks = [n for n in src.tensors if src._map[n][0] == "nv4exps"]
    ebytes = sum(tensor_bytes(src.tensors[n][0], src.tensors[n][1]) for n in stacks)
    copies = copies_mode(o, ebytes, nodes)
    tens, used, order, homes = {}, [0] * parts, [], {}

    def at(p, n):
        off = used[p]
        used[p] = _up(off + n)
        return off
    for gname in src.tensors:
        dims, t, _o = src.tensors[gname]
        dims = tuple(int(x) for x in dims)
        if src._map[gname][0] == "nv4exps":
            E = dims[2]
            nb = tensor_bytes(dims[:2], t)
            i = int(gname.split(".")[1])
            if copies == "both":
                ps = [[p, at(p, E * nb), list(range(E))] for p in range(parts)]
            else:
                ids = homes.setdefault(i, _homes(i, E, parts, counts))
                ps = [[p, at(p, len(ids[p]) * nb), [int(e) for e in ids[p]]] for p in range(parts)]
            tens[gname] = dict(dims=list(dims), type=int(t), place="experts", E=E, rows=dims[1], rb=0, nb=nb,
                               parts=ps, kind="experts")
        else:
            n = tensor_bytes(dims, t)
            tens[gname] = dict(dims=list(dims), type=int(t), place="node", seg=dpart, off=at(dpart, n), bytes=n,
                               kind="dense" if _dense_ok(gname, dims) else "small")
        order.append(gname)
    out = dict(layout=LAYOUT, model=MODEL, source="q35nv", file=os.path.abspath(path), nodes=list(nodes),
               share=list(share), gu=[], dn=[], L=int(cfg.num_hidden_layers), E=int(cfg.num_experts),
               hidden=int(cfg.hidden_size), inner=int(cfg.moe_inter), dense_part=dpart, copies=copies,
               types=dict(dense="float32 (FP8 and NVFP4 dequantized)", experts="nvx", ngram="none"), tensors=tens,
               order=order, meta=[["general.architecture", MODEL]], part_bytes=list(used))
    src.close()
    return out


def fill(sh, index, segs, threads, log):
    """The tensors into the segments: the small and dense ones through Q35NVSource.raw, the stacks
    of experts packed, then copied to their nodes."""
    src = _source(index["file"])
    t0 = time.time()
    n = 0
    for gname in index["order"]:
        d = index["tensors"][gname]
        if d["place"] != "node":
            continue
        a = np.ascontiguousarray(src.raw(gname)[0]).reshape(-1).view(np.uint8)
        assert a.nbytes == d["bytes"], (gname, a.nbytes, d["bytes"])
        segs[d["seg"]].array(d["off"], a.nbytes)[:] = a
        src._cache.pop(gname, None)
        n += a.nbytes
    log("the dense and small tensors: %.1f GB in %.0f s" % (n / 1e9, time.time() - t0))
    stacks = [g for g in index["order"] if index["tensors"][g]["place"] == "experts"]
    t0 = time.time()

    def one(gname):
        d = index["tensors"][gname]
        stack = src.pack_experts(gname, np.empty(d["E"] * d["nb"], np.uint8)).reshape(d["E"], d["nb"])
        for p, off, ids in d["parts"]:
            dst = segs[p].array(off, len(ids) * d["nb"]).reshape(len(ids), d["nb"])
            if len(ids) == d["E"]:
                dst[:] = stack
            else:
                dst[:] = stack[np.asarray(ids)]
    _pool(stacks, max(1, min(6, threads // 4)), one, log, "the stacks of experts")
    log("the experts: %d stacks in %.0f s" % (len(stacks), time.time() - t0))
    src.close()


def ident(path):
    """The size and the sha256 of the header of each safetensors file of the checkpoint."""
    where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for f in sorted(set(where.values())):
        p = os.path.join(path, f)
        with open(p, "rb") as fh:
            n = int(np.frombuffer(fh.read(8), np.uint64)[0])
            out[f] = [os.path.getsize(p), hashlib.sha256(fh.read(n)).hexdigest()]
    return out


class _Plugin:
    layout = staticmethod(layout)
    fill = staticmethod(fill)
    ident = staticmethod(ident)


PLUGIN = _Plugin()


class KeptQ35NV(KeptGGUF):
    """The tensors of the checkpoint in the weightkeeper (KeptGGUF), and the checkpoint's config."""

    f20_dense = True        # (its float32 products go to the GPU as F20 rows: models/qwen/gpu35.py)

    def __init__(self, path, w=None):
        super().__init__(path, w)
        if self.index.get("source") != "q35nv":
            raise ValueError("the weightkeeper holds %s, not an NVFP4 Qwen3.5 / 3.6" % self.index.get("source"))

    def config(self):
        from reclank.models.qwen.qwen35 import QwenConfig
        cfg = QwenConfig(self.path)
        cfg.v_tiled = False
        return cfg
