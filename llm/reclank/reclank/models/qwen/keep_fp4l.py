"""nvidia/Qwen3.8-Flash-Next-FP4-Local (reclank/formats/fp4local_st.py) in the weightkeeper
(reclank/runtime/weightkeeper.py; the tensors layout of reclank/models/qwen/keep.py;
docs/FP4LOCAL_PLAN.md P6).

The tensors are those of FP4LSource, in its forms:

- the routed experts (the CSF experts of the 48 layers and the plain ones of the MTP layer):
  KQ_NVX stacks, whole on every node when they fit (--copies auto, as keep_exl3: 69 GB on each
  of two nodes of 192 GB), so the CPU parts of each node read their rows in their own memory;
  else split between the nodes by expert;
- the n-gram table: KQ_NV4 rows of 160 (96 bytes: the codes, the 10 E4M3 scales, the float32
  global scale), split between the nodes by rows so they hold about the same bytes;
- the rest on the GPU's node as FP4LSource.raw makes it (the MXFP8 matrices as their exact
  bfloat16 values, the norms rounded to bfloat16 plus one, the bfloat16 tensors).

The fill: each stack of experts packed by FP4LSource.pack_experts (the CSF scales decoded)
on a few threads at once, then copied to the nodes that hold it; the n-gram shards packed
the same way and their rows copied to their nodes.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import numpy as np

from reclank.backends.cpu import cops
from reclank.formats.gguf import tensor_bytes
from reclank.models.qwen.keep import LAYOUT, MODEL, KeptGGUF, _dense_ok, _homes, _up
from reclank.models.qwen.keep_exl3 import copies_mode

KQ_NV4, KQ_NVX = 51, 53


def detect(path):
    from reclank.formats.fp4local_st import is_fp4local
    return os.path.isdir(path) and is_fp4local(path)


def _source(path):
    from reclank.formats.fp4local_st import FP4LSource
    return FP4LSource(path)


def layout(path, nodes, share, o):
    """The index of the tensors of FP4-Local in the weightkeeper."""
    src = _source(path)
    cfg = src.config()
    parts = len(nodes)
    dpart = nodes.index(o.dense_node) if o.dense_node in nodes else 0
    counts = np.load(o.counts) if getattr(o, "counts", None) else None
    stacks = [n for n in src.tensors if src._map[n][0] in ("csfexps", "nv4exps")]
    ebytes = sum(tensor_bytes(src.tensors[n][0], src.tensors[n][1]) for n in stacks)
    copies = copies_mode(o, ebytes, nodes)
    tens, used, order, homes = {}, [0] * parts, [], {}

    def at(p, n):
        off = used[p]
        used[p] = _up(off + n)
        return off
    ple = None
    for gname in src.tensors:
        kind = src._map[gname][0]
        dims, t, _o = src.tensors[gname]
        dims = tuple(int(x) for x in dims)
        if kind == "ple":
            ple = (gname, dims)
            continue
        if kind in ("csfexps", "nv4exps"):
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
    if ple is not None:
        gname, dims = ple
        cols, rows = int(dims[0]), int(dims[1])
        rb = cops.kq_nv4_row_bytes(cols)
        total = sum(used) + rows * rb
        left, r0, ps = rows, 0, []
        for p in range(parts):
            nr = left if p == parts - 1 else max(0, min(left, int((total / parts - used[p]) // rb)))
            ps.append([p, at(p, nr * rb), r0, nr])
            r0 += nr
            left -= nr
        tens[gname] = dict(dims=list(dims), type=KQ_NV4, place="rows", E=1, rows=rows, rb=rb, parts=ps,
                           kind="ngram")
        order.append(gname)
    out = dict(layout=LAYOUT, model=MODEL, source="fp4l", file=os.path.abspath(path), nodes=list(nodes),
               share=list(share), gu=[], dn=[], L=int(cfg.num_hidden_layers), E=int(cfg.num_experts),
               hidden=int(cfg.hidden_size), inner=int(cfg.moe_inter), dense_part=dpart, copies=copies,
               types=dict(dense="as stored (MXFP8 as bf16)", experts="nvx", ngram="nv4"), tensors=tens,
               order=order, meta=[], part_bytes=list(used))
    src.close()
    return out


def _pool(jobs, threads, fn, log, what):
    """fn(job) for each job on threads threads (the first error raised)."""
    it = iter(list(jobs))
    lock = threading.Lock()
    errs, done = [], [0]
    t0 = time.time()

    def work():
        while not errs:
            with lock:
                job = next(it, None)
            if job is None:
                return
            try:
                fn(job)
            except BaseException as e:      # noqa: BLE001
                errs.append(e)
                return
            with lock:
                done[0] += 1
                if done[0] % 16 == 0:
                    log("%s: %d done (%.0f s)" % (what, done[0], time.time() - t0))
    ts = [threading.Thread(target=work) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]


def fill(sh, index, segs, threads, log):
    """The tensors into the segments: the small and dense ones through FP4LSource.raw, the
    stacks of experts and the n-gram rows packed, then copied to their nodes."""
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
    # (each pack runs its OpenMP team; a few at once keep the NVMe busy while others decode)
    _pool(stacks, max(1, min(6, threads // 4)), one, log, "the stacks of experts")
    log("the experts: %d stacks in %.0f s" % (len(stacks), time.time() - t0))
    g = "per_layer_token_embd.weight"
    if g in index["tensors"]:
        d = index["tensors"][g]
        t0 = time.time()

        def shard(k):
            rows, r0 = src.pack_ple(k)
            rows = rows.reshape(-1, d["rb"])
            r1 = r0 + rows.shape[0]
            for p, off, q0, nr in d["parts"]:
                a, b = max(r0, q0), min(r1, q0 + nr)
                if a < b:
                    segs[p].array(off + (a - q0) * d["rb"], (b - a) * d["rb"])[:] = rows[a - r0:b - r0].reshape(-1)
        _pool(range(len(src._ple_shards)), max(1, min(6, threads // 4)), shard, log, "the n-gram shards")
        log("the n-gram table: %d rows in %.0f s" % (d["rows"], time.time() - t0))
    src.close()


def ident(path):
    """The size and the sha256 of the header of each safetensors file of the checkpoint (its
    index names them; Weights.check)."""
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


class KeptFP4L(KeptGGUF):
    """The tensors of FP4-Local in the weightkeeper (KeptGGUF), and the checkpoint itself
    (FP4LSource) for the config and the input scales of the experts."""

    as_stored = True
    _mtp_names = frozenset()        # (fp4local_st.attach_mtp)

    def __init__(self, path, w=None):
        super().__init__(path, w)
        if self.index.get("source") != "fp4l":
            raise ValueError("the weightkeeper holds %s, not FP4-Local" % self.index.get("source"))
        self.src = _source(path)
        # tensors the checkpoint makes that the weightkeeper does not hold (a keeper of an earlier
        # layout: the MTP drafter's head): from FP4LSource
        self._extra = [n for n in self.src.tensors if n not in self.index["tensors"] and self.src._map[n][0] != "ple"]
        for n in self._extra:
            self.tensors[n] = self.src.tensors[n]

    def raw(self, gname):
        if gname in self._mtp_names:
            return self._mtp.raw(gname)
        if gname in self._extra:
            return self.src.raw(gname)
        return super().raw(gname)

    def dequant(self, gname, rows=None):
        if gname in self._mtp_names:
            return self._mtp.dequant(gname, rows)
        if gname in self._extra:
            return self.src.dequant(gname, rows)
        return super().dequant(gname, rows)

    def config(self):
        return self.src.config()

    def input_scales(self, layer):
        return self.src.input_scales(layer)
