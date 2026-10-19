"""Qwen3.8-Flash-Next from an EXL3 directory (reclank/formats/exl3_qwen4.py), or GLM-5.3-Flash
(reclank/formats/exl3_glm53.py: the same forms, its tensors without the MTP layer), in the
weightkeeper (reclank/runtime/weightkeeper.py; the tensors layout of
reclank/models/qwen/keep.py).

The tensors are those of EXL3Source, in its forms: the MUL1 matrices and the
stacks of the routed experts as the blocks of the products (the trellis,
suh, svh of each matrix, KQ_E3_BASE + K), the other tensors as the GGUF
view makes them (the norms plus one, the bfloat16 matrices). The n-gram
table stays in its file (EXL3Source.ple_rows reads the rows it needs from
the NVMe).

The places: the stacks of the routed experts either split between the nodes
by expert (as keep.layout), or, when they are small enough (--copies, auto:
the experts of all the layers within COPY_SHARE of the memory of the
smallest node), whole on every node, so the threads of each node read the
experts in their own memory (gpu35.QwenGPU._kept_split: slot0 and slot1 of
every expert; kernels/kq/exl3_moe.c reads the copy of the thread's node);
the rest on the GPU's node.

The fill: the four tensors of an EXL3 module (mul1, suh, svh, trellis) sit
together in their file, so each expert matrix is one read, its parts placed
as the block (trellis, suh, svh) on each node that holds it; the other
tensors through EXL3Source.raw.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from reclank.formats.gguf import tensor_bytes
from reclank.models.qwen.keep import LAYOUT, MODEL, KeptGGUF, _dense_ok, _homes, _stream, _up

COPY_SHARE = 0.45           # auto: the experts on every node when they take at most this of a node


def detect(path):
    from reclank.formats.exl3_glm53 import is_glm53_exl3
    from reclank.formats.exl3_qwen4 import is_qwen4_exl3
    return is_qwen4_exl3(path) or is_glm53_exl3(path)


def _source(path):
    """The GGUF view of the directory: GLM53Source (no MTP layer) or EXL3Source."""
    from reclank.formats.exl3_glm53 import GLM53Source, is_glm53_exl3
    if is_glm53_exl3(path):
        return GLM53Source(path, mtp=False)
    from reclank.formats.exl3_qwen4 import EXL3Source
    return EXL3Source(path)


def _node_bytes(node):
    """The memory of a NUMA node (MemTotal), or None."""
    try:
        for line in open("/sys/devices/system/node/node%d/meminfo" % node):
            if "MemTotal:" in line:
                return int(line.split()[-2]) * 1024
    except OSError:
        pass
    return None


def copies_mode(o, expert_bytes, nodes):
    """"both" (the experts whole on every node) or "split" (each expert on one)."""
    want = getattr(o, "copies", None) or "auto"
    if want in ("both", "split") or len(nodes) < 2:
        return want if len(nodes) > 1 else "split"
    mem = [_node_bytes(n) for n in nodes]
    if any(m is None for m in mem):
        return "split"
    return "both" if expert_bytes <= COPY_SHARE * min(mem) else "split"


def layout(path, nodes, share, o):
    """The index of the tensors of an EXL3 directory in the weightkeeper."""
    src = _source(path)
    cfg = src.config()
    parts = len(nodes)
    dpart = nodes.index(o.dense_node) if o.dense_node in nodes else 0
    counts = np.load(o.counts) if getattr(o, "counts", None) else None
    stacks = [n for n in src.tensors if src._map[n][0] == "e3exps"]
    ebytes = sum(tensor_bytes(src.tensors[n][0], src.tensors[n][1]) for n in stacks)
    copies = copies_mode(o, ebytes, nodes)
    tens, used, order, homes = {}, [0] * parts, [], {}

    def at(p, n):
        off = used[p]
        used[p] = _up(off + n)
        return off
    for gname in src.tensors:
        kind = src._map[gname][0]
        if kind == "ple":
            continue            # (the n-gram table: from its file)
        dims, t, _o = src.tensors[gname]
        dims = tuple(int(x) for x in dims)
        if kind == "e3exps":
            E = dims[2]
            nb = tensor_bytes(dims[:2], t)
            i = int(gname.split(".")[1])
            if copies == "both":
                ps = [[p, at(p, E * nb), list(range(E))] for p in range(parts)]
            else:
                ids = homes.setdefault(i, _homes(i, E, parts, counts))
                ps = [[p, at(p, len(ids[p]) * nb), [int(e) for e in ids[p]]] for p in range(parts)]
            m, x = src._map[gname][1]
            tens[gname] = dict(dims=list(dims), type=int(t), place="experts", E=E, rows=dims[1], rb=0, nb=nb,
                               parts=ps, kind="experts", src=[m, x])
        else:
            n = tensor_bytes(dims, t)
            tens[gname] = dict(dims=list(dims), type=int(t), place="node", seg=dpart, off=at(dpart, n), bytes=n,
                               kind="dense" if kind == "e3" or _dense_ok(gname, dims) else "small")
        order.append(gname)
    out = dict(layout=LAYOUT, model=MODEL, source="exl3", file=os.path.abspath(path), nodes=list(nodes),
               share=list(share), gu=[], dn=[], L=int(cfg.num_hidden_layers), E=int(cfg.num_experts),
               hidden=int(cfg.hidden_size), inner=int(cfg.moe_inter), dense_part=dpart, copies=copies,
               types=dict(dense="exl3", experts="exl3", ngram="file"), tensors=tens, order=order, meta=[],
               part_bytes=list(used))
    src.close()
    return out


def _spans(d, key):
    """The spans of the trellis, suh, and svh of an EXL3 module in their files:
    [(path, start, end)] in that order (most modules in one file; a few of
    GLM-5.3's span two shards)."""
    out = []
    for name in (key + ".trellis", key + ".suh", key + ".svh"):
        st = d._of(name)
        a, b = st.header[name]["data_offsets"]
        out.append((os.path.join(d.path, d.where[name]), st._base + a, st._base + b))
    return out


def fill(sh, index, segs, threads, log):
    """The tensors into the segments: the small and the dense ones through
    EXL3Source.raw, the stacks of experts by spans of their files."""
    import time
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
        n += a.nbytes
    log("the dense and small tensors: %.1f GB in %.0f s" % (n / 1e9, time.time() - t0))
    merged = {}
    for gname in index["order"]:
        d = index["tensors"][gname]
        if d["place"] != "experts":
            continue
        m, x = d["src"]
        where = {}          # expert -> [(part, offset of its block)]
        for p, off, ids in d["parts"]:
            for slot, e in enumerate(ids):
                where.setdefault(e, []).append((p, off + slot * d["nb"]))
        for e, dsts in where.items():
            sp = _spans(src.d, m + "experts.%d.%s_proj" % (e, x))
            runs, at = [], 0
            for path, s, e2 in sp:      # the block: trellis, suh, svh, read as contiguous runs of a file
                if runs and runs[-1][0] == path and runs[-1][2] == s:
                    runs[-1][2] = e2
                    runs[-1][3].append((s - runs[-1][1], e2 - s, at))
                else:
                    runs.append([path, s, e2, [(0, e2 - s, at)]])
                at += e2 - s
            assert at == d["nb"], (gname, e, at, d["nb"])
            for path, a, b, pieces in runs:
                merged[(path, a, b)] = (dsts, pieces)

    def handle(v, h):
        dsts, pieces = h
        for p, base in dsts:
            for rel, nbytes, dr in pieces:
                segs[p].array(base + dr, nbytes)[:] = v[rel:rel + nbytes]
    _stream(merged, threads, handle, log, "the experts of %s" % os.path.basename(index["file"]))
    src.close()


def ident(path):
    """The size and the sha256 of the header of each safetensors file of the
    directory (Weights.check)."""
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            p = os.path.join(path, f)
            with open(p, "rb") as fh:
                n = int(np.frombuffer(fh.read(8), np.uint64)[0])
                out[f] = [os.path.getsize(p), hashlib.sha256(fh.read(n)).hexdigest()]
    return out


class _Plugin:
    """The weightkeeper's plugin for an EXL3 directory of Qwen3.8."""
    layout = staticmethod(layout)
    fill = staticmethod(fill)
    ident = staticmethod(ident)


PLUGIN = _Plugin()


class KeptEXL3(KeptGGUF):
    """The tensors of an EXL3 directory in the weightkeeper (KeptGGUF), and
    the directory itself (EXL3Source) for the config and the n-gram rows."""

    exl3 = True

    def __init__(self, path, w=None):
        super().__init__(path, w)
        if self.index.get("source") != "exl3":
            raise ValueError("the weightkeeper holds %s, not an EXL3 directory" % self.index.get("source"))
        self.src = _source(path)
        name = "per_layer_token_embd.weight"
        if name in self.src.tensors:
            self.tensors[name] = self.src.tensors[name]

    def config(self):
        return self.src.config()

    def raw(self, gname):
        d = self.index["tensors"][gname]
        dims, t, _o = self.tensors[gname]
        if 70 < t <= 78 and d["place"] == "node":
            return self.w.tensor(gname), dims, t       # (the block of an EXL3 matrix)
        return super().raw(gname)

    def ple_rows(self, rows):
        return self.src.ple_rows(rows)

    def embed_rows(self, ids):
        return self.src.embed_rows(ids)

    def dequant(self, gname, rows=None):
        if gname == "per_layer_token_embd.weight":
            return self.src.ple_rows(np.arange(self.tensors[gname][0][1]) if rows is None else rows)
        if gname not in self.index["tensors"]:
            return self.src.dequant(gname, rows)
        dims, t, _o = self.tensors[gname]
        if 70 < t <= 78:
            return self.src.dequant(gname, rows)        # (the weights of its trellis)
        return super().dequant(gname, rows)

