"""Qwen3.8-Flash-Next in the weightkeeper (reclank/runtime/weightkeeper.py):
the original bfloat16 checkpoint ("ORIG", Qwen/Qwen3.8-Flash-Next) converted
once into the GGUF tensors of the runtime in memory, handed to the rounds.

The types are minimums, chosen for each kind of tensor (Options of the
weightkeeper: dense, experts, ngram):

- **dense** (the attention, the DeltaNet, the shared experts, the hyper
  connections, the head, the MTP projections): bf12 (the default; a matrix
  that BF12 cannot hold under the runtime's rule stays bfloat16), bf16, or q8.
- **experts** (the routed experts of the layers and of the MTP layer): bf12
  (the default), bf16, q8, rq8.
- **ngram** (the n-gram table, per_layer_token_embd): bf12 (the default),
  bf16, q8.
- the embeddings, the indexer, and the small tensors stay as the GGUF view of
  ORIG has them (bfloat16, float32).

The places: the routed experts of each layer (and of the MTP layer) split
between the nodes by expert, each expert whole on one node (the experts of a
node one after another in the order of their index): the split of
gpu35.QwenGPU._numa_split, whose CPU part runs each expert on the threads of
its node and whose copies to the GPU read each expert where it is; the
n-gram table split by rows so that the nodes hold about the same bytes; the
rest on the GPU's node (dense_node). index["tensors"][name]: dims (ggml
order), type, place "node" (seg, off, bytes), "experts" (E, rows, rb, nb the
bytes of an expert, parts [[seg, off, [the experts there]]]) or "rows" (the
n-gram table: E 1, rows, rb, parts [[seg, off, r0, nr]]).

The fill: the small tensors through the map of OrigSource (the value heads
of the DeltaNet in the tiled order, the norms plus one), then the bulk (the
experts, the n-gram table) read in the order of the files by one thread
(the HDD streams) and converted by the others.
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import threading
import time

import numpy as np

from reclank.backends.cpu import cops
from reclank.formats.gguf import BF12, BF16, F32, GGUF, Q8_0, RQ8_0, tensor_bytes

LAYOUT = "tensors-v1"
MODEL = "qwen4exp"
ALIGN = 64
TYPES = {"bf12": BF12, "bf16": BF16, "q8": Q8_0, "rq8": RQ8_0}
CHUNK = 64 << 20              # the bytes of an item of the bulk read (the n-gram rows)


def _up(n, a=ALIGN):
    return (n + a - 1) // a * a


def row_bytes(t, cols):
    return tensor_bytes((cols, 1), t)


def detect(cfg):
    return cfg.get("model_type") == "qwen4_exp"


def convert(rows, t):
    """Rows of bfloat16 (uint16, [n, cols]) in type t: (uint8 bytes, the BF12
    report or None)."""
    rows = np.ascontiguousarray(rows)
    cols = rows.shape[-1]
    if t == BF12:
        out, rep = cops.kq_bf16_to_bf12(rows, cols)
        return out.reshape(-1), rep
    if t == BF16:
        return rows.reshape(-1).view(np.uint8), None
    if t == Q8_0:
        return cops.kq_to_q8_0(rows, cols).reshape(-1), None
    if t == RQ8_0:
        f = (rows.astype(np.uint32) << 16).view(np.float32)
        return cops.kq_to_q8_0(cops.tq6_rotate(f), cols).reshape(-1), None
    raise ValueError("no conversion to type %d" % t)


def bf12_bad(rep):
    """The rule of the runtime (convert_q8_gguf.py --dense bf12): an Inf or a
    NaN, or a zeroed value above 2^-8 of the RMS of the matrix."""
    return rep["nonfinite"] > 0 or rep["largest"] > rep["rms"] * 2.0 ** -8


class Report:
    """The BF12 reports of a kind of tensor, added up (many threads)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.k = {}

    def add(self, kind, rep, name):
        if rep is None:
            return
        with self.lock:
            r = self.k.setdefault(kind, dict(mats=0, zeroed=0, collisions=0, bad=0, worst=0.0,
                                             worst_name=None))
            r["mats"] += 1
            r["zeroed"] += rep["zeroed"]
            r["collisions"] += rep["collisions"]
            ratio = rep["largest"] / max(rep["rms"], 1e-30)
            if ratio > r["worst"]:
                r["worst"], r["worst_name"] = ratio, name
            if bf12_bad(rep):
                r["bad"] += 1

    def text(self):
        return "; ".join("%s: %d matrices, %d zeroed, %d collisions, the largest zeroed %.2e of its "
                         "RMS (%s), %d over the rule" % (k, r["mats"], r["zeroed"], r["collisions"],
                                                         r["worst"], r["worst_name"], r["bad"])
                         for k, r in sorted(self.k.items()))


def _orig(ckpt):
    from reclank.formats.qwen38_orig import OrigSource
    return OrigSource(ckpt)


def layout(ckpt, nodes, share, o):
    """The index of the tensors of ORIG in the weightkeeper (without the
    segments): the types of the options, the places, the bytes of each node."""
    from reclank.formats.qwen38_orig import metadata
    from reclank.runtime.moe_parts import split
    src = _orig(ckpt)
    cfg = src.config()
    want = {k: TYPES[getattr(o, k + "_type")] for k in ("dense", "experts", "ngram")}
    parts = len(nodes)
    L, E = src.L, src.E
    hidden, inner = cfg.hidden_size, cfg.moe_inter
    gu = split(inner, parts, share=share) if parts > 1 else [(0, inner)]
    dn = split(hidden, parts, share=share) if parts > 1 else [(0, hidden)]
    dpart = nodes.index(o.dense_node) if o.dense_node in nodes else 0
    tens, used = {}, [0] * parts
    order = []
    homes = {}
    counts = np.load(o.counts) if getattr(o, "counts", None) else None

    def at(p, n):
        off = used[p]
        used[p] = _up(off + n)
        return off
    ple = None
    for gname, (kind, _s) in src._map.items():
        dims, t0, _ = src.tensors[gname]
        if kind == "q8exps":
            t = want["experts"]
            cols, rows, ne = (int(x) for x in dims)
            rb = row_bytes(t, cols)
            i = int(gname.split(".")[1])
            ids = homes.setdefault(i, _homes(i, ne, parts, counts))
            nb = rows * rb
            ps = [[p, at(p, len(ids[p]) * nb), [int(e) for e in ids[p]]] for p in range(parts)]
            tens[gname] = dict(dims=list(dims), type=t, place="experts", E=ne, rows=rows, rb=rb, nb=nb,
                               parts=ps, kind="experts")
        elif kind == "q8ple":
            ple = (gname, dims)
            continue
        elif kind in ("dense", "ehproj") and _dense_ok(gname, dims):
            t = want["dense"]
            # the bytes of bfloat16 at most: a matrix that BF12 cannot hold stays so
            n = max(tensor_bytes(dims, t), tensor_bytes(dims, BF16))
            tens[gname] = dict(dims=list(dims), type=t, place="node", seg=dpart, off=at(dpart, n),
                               bytes=tensor_bytes(dims, t), kind="dense")
        else:
            t = F32 if kind == "f32" else BF16 if t0 in (BF16, 30) else t0
            tens[gname] = dict(dims=list(dims), type=t, place="node", seg=dpart,
                               off=at(dpart, tensor_bytes(dims, t)), bytes=tensor_bytes(dims, t),
                               kind="small")
        order.append(gname)
    if ple is not None:
        gname, dims = ple
        t = want["ngram"]
        cols, rows = int(dims[0]), int(dims[1])
        rb = row_bytes(t, cols)
        # the rows of each node: the nodes about even in bytes
        total = sum(used) + rows * rb
        left = rows
        ps = []
        r0 = 0
        for p in range(parts):
            nr = left if p == parts - 1 else max(0, min(left, int((total / parts - used[p]) // rb)))
            ps.append([p, at(p, nr * rb), r0, nr])
            r0 += nr
            left -= nr
        tens[gname] = dict(dims=list(dims), type=t, place="rows", E=1, rows=rows, rb=rb, parts=ps,
                           kind="ngram")
        order.append(gname)
    meta = [list(kv) for kv in metadata(cfg, src)]
    meta[1][1] = "Qwen3.8-Flash-Next (reclank weightkeeper: dense %s, experts %s, n-gram %s)" % (
        o.dense_type, o.experts_type, o.ngram_type)
    meta[2][1] = "Qwen/Qwen3.8-Flash-Next (original bfloat16)"
    if want["experts"] == RQ8_0:
        from reclank.formats.gguf import rotation_meta
        meta += [list(kv) for kv in rotation_meta()]
    src.close()
    return dict(layout=LAYOUT, model=MODEL, nodes=list(nodes), share=list(share), gu=[list(x) for x in gu],
                dn=[list(x) for x in dn], L=L, E=E, hidden=hidden, inner=inner, dense_part=dpart,
                types=dict(dense=o.dense_type, experts=o.experts_type, ngram=o.ngram_type),
                tensors=tens, order=order, meta=meta, part_bytes=list(used))


def _homes(i, E, parts, counts):
    """The experts of layer i on each part (sorted): by their counts (the
    profile of the router on real text, RECLANK_EXPERT_COUNTS) in snake order,
    so the parts have about the same share of the picks; else in turn."""
    if counts is not None and i < counts.shape[0]:
        order = np.argsort(-np.asarray(counts[i]), kind="stable")
    else:
        order = np.arange(E)
    k = np.arange(E) % (2 * parts)
    part = np.where(k < parts, k, 2 * parts - 1 - k)
    return [np.sort(order[part == p]) for p in range(parts)]


def _dense_ok(gname, dims):
    """A matrix that takes the dense type (the rule of convert_q8_gguf.py
    bf12_ok): two dims, more than one row, cols a multiple of 32, not the
    indexer, not the embeddings."""
    return (len(dims) == 2 and dims[1] > 1 and dims[0] % 32 == 0 and ".indexer." not in gname
            and gname != "token_embd.weight")


class _Bufs:
    """A pool of page-aligned buffers for the bulk read (O_DIRECT)."""

    def __init__(self, n, size):
        from reclank.runtime import weightkeeper as W
        self.size = size
        self.free = queue.Queue()
        for _ in range(n):
            addr = W._mmap(None, size, W.PROT_READ | W.PROT_WRITE, W.MAP_PRIVATE | W.MAP_ANONYMOUS)
            self.free.put(np.frombuffer((ctypes.c_uint8 * size).from_address(addr), np.uint8))


def _pread_into(fd, buf, start, end, direct):
    """The bytes [start, end) of fd into buf: a view of them."""
    K4 = 4096
    a0 = start & ~(K4 - 1) if direct else start
    a1 = (end + K4 - 1) // K4 * K4 if direct else end
    mv = memoryview(buf)[:a1 - a0]
    got = 0
    while a0 + got < end:
        r = os.preadv(fd, [mv[got:]], a0 + got)
        if r <= 0:
            raise OSError("short read at %d" % (a0 + got))
        got += r
    if not direct:
        os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_DONTNEED)
    return buf[start - a0:end - a0]


def fill(sh, index, segs, threads, log):
    """The tensors of the index into the segments (segs[p]: the Segment of part
    p): the small ones first, then the bulk."""
    src = _orig(sh.path)
    cfg = src.config()
    from reclank.formats.qwen38_orig import tiler
    tiled = tiler(cfg)
    tens = index["tensors"]
    rep = Report()
    t0 = time.time()

    def put(p, off, data):
        segs[p].array(off, data.nbytes)[:] = data.reshape(-1).view(np.uint8)

    # ---- the small tensors and the dense matrices (the map of OrigSource)
    small = [n for n in index["order"] if tens[n]["place"] == "node"]
    done_small = [0]
    lock = threading.Lock()
    work = list(small)
    errs = []

    def do_small():
        try:
            while True:
                with lock:
                    if not work or errs:
                        return
                    gname = work.pop(0)
                d = tens[gname]
                kind, s = src._map[gname]
                if kind == "dense" and gname.endswith(("attn_qkv.weight", "attn_gate.weight",
                                                       "ssm_out.weight")):
                    a = tiled(gname, np.asarray(src._bf16(s)))
                elif kind == "f32":
                    a = tiled(gname, src._make(gname))
                else:
                    a = np.asarray(src._make(gname))
                if d["kind"] == "dense":
                    bits = np.ascontiguousarray(a).view(np.uint16)
                    data, r = convert(bits, d["type"])
                    rep.add("dense", r, gname)
                    if r is not None and bf12_bad(r):
                        data, d["type"] = bits.reshape(-1).view(np.uint8), BF16
                        d["bytes"] = data.nbytes
                        log("%s stays bfloat16 (BF12: the largest zeroed %.2e of its RMS, %d Inf/NaN)" % (
                            gname, r["largest"] / max(r["rms"], 1e-30), r["nonfinite"]))
                else:
                    want = np.float32 if d["type"] == F32 else None
                    data = np.ascontiguousarray(a, want) if want else np.ascontiguousarray(a)
                assert data.nbytes == d["bytes"], (gname, data.nbytes, d["bytes"], d["type"])
                put(d["seg"], d["off"], data)
                with lock:
                    done_small[0] += data.nbytes
        except BaseException as e:      # noqa: BLE001 - reported below
            errs.append(e)
    ts = [threading.Thread(target=do_small, daemon=True) for _ in range(min(threads, 8))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if errs:
        raise errs[0]
    log("the dense and small tensors: %.1f GB in %.1f s" % (done_small[0] / 1e9, time.time() - t0))

    # ---- the bulk: the experts (an item an expert) and the n-gram rows (an item a chunk)
    items = []                  # (file, start, end, handler args)
    for gname in index["order"]:
        d = tens[gname]
        if d["place"] not in ("rows", "experts"):
            continue
        kind, s = src._map[gname]
        if kind == "q8exps":
            hf, lo, hi, _t = s
            f, start, end = src.span(hf)
            per = (end - start) // d["E"]
            where = {}
            for p, off, ids in d["parts"]:
                for slot, e in enumerate(ids):
                    where[e] = (p, off + slot * d["nb"])
            for e in range(d["E"]):
                p, at_ = where[e]
                items.append((f, start + e * per, start + (e + 1) * per, ("x", gname, e, lo, hi, p, at_)))
        else:
            g0 = 0
            for shard in src._ple_shards:
                f, start, end = src.span(shard)
                nrow, cols = src._shape(shard)
                rb16 = 2 * cols
                step = max(1, CHUNK // rb16)
                for r in range(0, nrow, step):
                    nr = min(step, nrow - r)
                    items.append((f, start + r * rb16, start + (r + nr) * rb16, ("p", gname, g0 + r, nr, cols)))
                g0 += nrow
    # the gate and the up rows of an expert come from the same span: one read, two items
    merged = {}
    for f, a, b, h in items:
        merged.setdefault((f, a, b), []).append(h)

    def handle(v, hs):
        for h in hs:
            _bulk(h, v, tens, put, rep)
    _stream(merged, threads, handle, log, "the experts and the n-gram table")
    if rep.k:
        log("BF12: " + rep.text())
    index["bf12"] = rep.k
    src.close()


def _stream(merged, threads, handle, log, what):
    """Read the spans of merged ({(file, start, end): payload}) in the order of
    the files on one thread (O_DIRECT into page-aligned buffers; the disk
    streams) and give each to handle(view, payload) on threads others."""
    reads = sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    total = sum(b - a for (f, a, b), _h in reads)
    big = max(b - a for (f, a, b), _h in reads)
    pool = _Bufs(2 * threads + 2, (big + 2 * 4096 + 4095) // 4096 * 4096)
    q = queue.Queue(maxsize=2 * threads)
    done, errs, lock = [0], [], threading.Lock()
    t1 = time.time()

    def reader():
        fds = {}
        try:
            for (f, a, b), hs in reads:
                if errs:
                    break
                if f not in fds:
                    try:
                        fds[f] = (os.open(f, os.O_RDONLY | os.O_DIRECT), True)
                    except OSError:
                        fds[f] = (os.open(f, os.O_RDONLY), False)
                fd, direct = fds[f]
                buf = pool.free.get()
                try:
                    v = _pread_into(fd, buf, a, b, direct)
                except OSError as e:
                    if direct and e.errno == 22:
                        os.close(fd)
                        fds[f] = (os.open(f, os.O_RDONLY), False)
                        v = _pread_into(fds[f][0], buf, a, b, False)
                    else:
                        raise
                q.put((buf, v, hs))
        except BaseException as e:      # noqa: BLE001
            errs.append(e)
        finally:
            for _ in range(threads):
                q.put(None)
            for fd, _d in fds.values():
                os.close(fd)

    def worker():
        try:
            while True:
                job = q.get()
                if job is None:
                    return
                buf, v, hs = job
                if not errs:
                    handle(v, hs)
                with lock:
                    done[0] += v.nbytes
                pool.free.put(buf)
        except BaseException as e:      # noqa: BLE001
            errs.append(e)
            while q.get() is not None:      # let the reader finish
                pass
    rt = threading.Thread(target=reader, daemon=True)
    ws = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    rt.start()
    [w.start() for w in ws]
    last = time.time()
    while any(w.is_alive() for w in ws):
        time.sleep(0.5)
        if time.time() - last >= 30:
            last = time.time()
            log("read %.1f of %.1f GB (%.0f MB/s)" % (done[0] / 1e9, total / 1e9,
                                                       done[0] / 1e6 / (last - t1)))
    rt.join()
    if errs:
        raise errs[0]
    dt = time.time() - t1
    log("%s: %.1f GB in %.0f s (%.0f MB/s)" % (what, total / 1e9, dt, total / 1e6 / max(dt, 1e-9)))


def _bulk(h, v, tens, put, rep):
    """One item of the bulk: the rows of an expert (gate or up of the fused
    gate_up, or down) or a chunk of n-gram rows, to the parts."""
    if h[0] == "x":
        _x, gname, e, lo, hi, p, at_ = h
        d = tens[gname]
        cols = int(d["dims"][0])
        data, r = convert(v.view(np.uint16).reshape(-1, cols)[lo:hi], d["type"])
        rep.add("experts", r, "%s[%d]" % (gname, e))
        assert data.nbytes == d["nb"], (gname, data.nbytes, d["nb"])
        put(p, at_, data)
        return
    _p, gname, g0, n, cols = h
    d = tens[gname]
    w = v.view(np.uint16).reshape(n, cols)
    for p, off, r0, nr in d["parts"]:
        a, b = max(g0, r0), min(g0 + n, r0 + nr)
        if a >= b:
            continue
        data, r = convert(w[a - g0:b - g0], d["type"])
        rep.add("ngram", r, gname)
        put(p, off + (a - r0) * d["rb"], data)


# ---------------------------------------------------------------- a round

def holds(ckpt):
    """True when the weightkeeper of this user holds the checkpoint at ckpt in
    this layout (no map: the status)."""
    from reclank.runtime import weightkeeper as W
    try:
        c = W.request("status")
        with c:
            st = json.loads(W._recv_msg(c))
    except (OSError, ValueError):
        return False
    return st.get("layout") == LAYOUT and os.path.realpath(st.get("checkpoint", "")) == os.path.realpath(ckpt)


def is_orig(path):
    """The original bfloat16 checkpoint (its fused experts), not the NVFP4 one
    of ModelOpt (reclank/formats/nvfp4_st.py)."""
    try:
        where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    except OSError:
        return False
    return any(k.endswith("mlp.experts.gate_up_proj") for k in where)


class KeptGGUF(GGUF):
    """The GGUF view of the tensors in the weightkeeper (the interface of
    reclank.formats.gguf.GGUF: tensors, meta, raw, dequant): the tensors
    whole on a node are views of the segments; the routed experts and the
    n-gram table are split by rows between the nodes (rows(): the parts)."""

    use_cache = False
    keep_embedding_bf16 = True

    def __init__(self, ckpt, w=None):
        from reclank.runtime import weightkeeper as W
        self.path = ckpt
        self.w = w or W.attach(ckpt=ckpt)
        ix = self.w.index
        if ix.get("layout") != LAYOUT or ix.get("model") != MODEL:
            raise ValueError("the weightkeeper holds %s (%s), not Qwen3.8" % (ix.get("model"), ix.get("layout")))
        self.index = ix
        self.meta = {k: v for k, v in ix["meta"]}
        self.tensors = {n: (tuple(int(x) for x in d["dims"]), int(d["type"]), 0)
                        for n, d in ix["tensors"].items()}
        self._order = list(ix["order"])
        self._mm = None
        self.staged = False
        self.stage_node = None
        self.expert_node = None
        self.parts = []
        self.kept = True

    def raw(self, gname):
        d = self.index["tensors"][gname]
        dims, t, _o = self.tensors[gname]
        if d["place"] != "node":
            raise ValueError("%s is split between the nodes (experts(), row_addrs())" % gname)
        a = self.w.tensor(gname)
        from reclank.formats.gguf import _BLOCK_DT, _ROW_BYTES
        if t in _ROW_BYTES:
            return a, dims, t
        return a.view(_BLOCK_DT[t]), dims, t

    def experts(self, gname):
        """A stack of experts split between the nodes: [(node, uint8 (n, nb), the
        n experts there)] (each expert whole, nb bytes)."""
        d = self.index["tensors"][gname]
        nodes = self.index["nodes"]
        return [(nodes[p], self.w._arr(self.w.base[p] + off, len(ids) * d["nb"]).reshape(len(ids), d["nb"]),
                 np.asarray(ids, np.int64)) for p, off, ids in d["parts"]]

    def row_addrs(self, gname, rows):
        """The addresses of rows of the n-gram table (the part of each row)."""
        d = self.index["tensors"][gname]
        rows = np.asarray(rows, np.int64).reshape(-1)
        out = np.empty(rows.size, np.int64)
        for p, off, r0, nr in d["parts"]:
            m = (rows >= r0) & (rows < r0 + nr)
            out[m] = self.w.base[p] + off + (rows[m] - r0) * d["rb"]
        return out

    def dequant(self, gname, rows=None):
        d = self.index["tensors"][gname]
        if d["place"] == "node":
            return super().dequant(gname, rows)
        if d["E"] != 1:
            raise ValueError("%s: the experts are read by their parts" % gname)
        dims, t, _o = self.tensors[gname]
        cols = int(dims[0])
        idx = np.arange(d["rows"]) if rows is None else np.asarray(rows, np.int64).reshape(-1)
        raw = cops.kq_gather(self.row_addrs(gname, idx), d["rb"])
        return cops.kq_rows(raw.reshape(-1), t, cols, np.arange(idx.size))

    def advise_random(self, gname):
        pass

    def drop_cache(self):
        pass

    def close(self):
        pass


def export_gguf(g, out, log=print):
    """The tensors that the weightkeeper holds (g: KeptGGUF) as one GGUF file at
    out, in their types (BF12 by default), each tensor whole: the experts of a
    stack in the order of their index (from the nodes that hold them), the
    n-gram table in the order of its rows. The runtime reads it as any GGUF
    file of qwen4exp (Qwen4CPU, Qwen4GPU)."""
    from reclank.formats.gguf import write_gguf
    ix = g.index
    tens = ix["tensors"]

    def maker(name):
        d = tens[name]
        if d["place"] == "node":
            return lambda: g.w.tensor(name)
        if d["place"] == "experts":
            def experts():
                at = {}
                for _node, arr, ids in g.experts(name):
                    for k, e in enumerate(ids):
                        at[int(e)] = arr[k]
                for e in range(d["E"]):
                    yield at[e]
            return experts

        def rows():
            for _p, arr, _r0, _nr in sorted(g.w.tensor(name), key=lambda x: x[2]):
                yield arr.reshape(-1)
        return rows
    tensors = [(n, tuple(int(x) for x in tens[n]["dims"]), int(tens[n]["type"]), maker(n)) for n in ix["order"]]
    meta = [tuple(kv) for kv in ix["meta"]]
    meta.append(("reclank.weightkeeper.types", "dense %s, experts %s, n-gram %s" % (
        ix["types"]["dense"], ix["types"]["experts"], ix["types"]["ngram"])))
    if ix.get("bf12"):
        meta.append(("reclank.bf12.zeroed", int(sum(r["zeroed"] for r in ix["bf12"].values()))))
    total = sum(tensor_bytes(d, t) for _n, d, t, _m in tensors)
    t0 = time.time()
    done = [0, 0]

    def progress(name, n):
        done[0] += n
        if done[0] - done[1] >= 20e9 or done[0] == total:
            done[1] = done[0]
            dt = time.time() - t0
            log("  %.1f of %.1f GB, %.0f s, %.2f GB/s (%s)" % (done[0] / 1e9, total / 1e9, dt,
                                                             done[0] / 1e9 / max(dt, 1e-9), name))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".part"
    write_gguf(tmp, meta, tensors, progress=progress)
    os.replace(tmp, out)
    import shutil
    for f in ("tokenizer.json", "chat_template.jinja"):
        p = os.path.join(g.path, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(os.path.dirname(os.path.abspath(out)), f))
    log("%s: %d tensors, %.1f GB in %.0f s" % (out, len(tensors), total / 1e9, time.time() - t0))


# ---------------------------------------------------------------- from a GGUF file

def detect_gguf(path):
    """A GGUF file of qwen4exp (the file of export_gguf, or of convert_q8_gguf.py)."""
    if not (path.endswith(".gguf") and os.path.isfile(path)):
        return False
    g = GGUF(path, stage=False)
    try:
        return g.meta.get("general.architecture") == MODEL
    finally:
        g.close()


def gguf_ident(path):
    """The size and the sha256 of the header (the metadata and the tensor
    infos) of a GGUF file: the check of a round (Weights.check)."""
    import hashlib
    g = GGUF(path, stage=False)
    base = g._base
    g.close()
    with open(path, "rb") as f:
        h = hashlib.sha256(f.read(base)).hexdigest()
    return {os.path.basename(path): [os.path.getsize(path), h]}


def _plain(v):
    """A metadata value as JSON takes it."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def _tname(t):
    from reclank.formats.gguf import _TYPE_NAME
    return {BF12: "bf12", BF16: "bf16", F32: "f32", Q8_0: "q8", RQ8_0: "rq8"}.get(t, _TYPE_NAME.get(t, str(t)))


def layout_gguf(path, nodes, share, o):
    """The index of the tensors of a qwen4exp GGUF file in the weightkeeper: the
    places of layout() (the experts of each layer split between the nodes by
    expert, the n-gram table by rows, the rest on the GPU's node), the types
    as the file has them (a copy, no conversion)."""
    g = GGUF(path, stage=False)
    parts = len(nodes)
    dpart = nodes.index(o.dense_node) if o.dense_node in nodes else 0
    counts = np.load(o.counts) if getattr(o, "counts", None) else None
    tens, used, order, homes, offs = {}, [0] * parts, [], {}, {}

    def at(p, n):
        off = used[p]
        used[p] = _up(off + n)
        return off
    ple = None
    for gname in g._order:
        dims, t, off = g.tensors[gname]
        dims = tuple(int(x) for x in dims)
        offs[gname] = int(off)
        if "_exps." in gname and len(dims) == 3:
            cols, rows, ne = dims
            rb = row_bytes(t, cols)
            i = int(gname.split(".")[1])
            ids = homes.setdefault(i, _homes(i, ne, parts, counts))
            nb = rows * rb
            ps = [[p, at(p, len(ids[p]) * nb), [int(e) for e in ids[p]]] for p in range(parts)]
            tens[gname] = dict(dims=list(dims), type=int(t), place="experts", E=ne, rows=rows, rb=rb, nb=nb,
                               parts=ps, kind="experts")
        elif gname == "per_layer_token_embd.weight":
            ple = (gname, dims, t)
            continue
        else:
            n = tensor_bytes(dims, t)
            tens[gname] = dict(dims=list(dims), type=int(t), place="node", seg=dpart, off=at(dpart, n), bytes=n,
                               kind="dense" if _dense_ok(gname, dims) else "small")
        order.append(gname)
    if ple is not None:
        gname, dims, t = ple
        cols, rows = dims
        rb = row_bytes(t, cols)
        total = sum(used) + rows * rb
        left, r0, ps = rows, 0, []
        for p in range(parts):
            nr = left if p == parts - 1 else max(0, min(left, int((total / parts - used[p]) // rb)))
            ps.append([p, at(p, nr * rb), r0, nr])
            r0 += nr
            left -= nr
        tens[gname] = dict(dims=list(dims), type=int(t), place="rows", E=1, rows=rows, rb=rb, parts=ps,
                           kind="ngram")
        order.append(gname)

    def kind_type(k):
        ts = sorted({_tname(d["type"]) for d in tens.values() if d["kind"] == k})
        return "/".join(ts) or "-"
    meta = [[k, _plain(v)] for k, v in g.meta.items() if k != "general.alignment"]
    a = MODEL + "."
    out = dict(layout=LAYOUT, model=MODEL, source="gguf", file=os.path.abspath(path), base=int(g._base),
               offs=offs, nodes=list(nodes), share=list(share), gu=[], dn=[],
               L=int(g.meta.get(a + "block_count", 0)), E=int(g.meta.get(a + "expert_count", 0)),
               hidden=int(g.meta.get(a + "embedding_length", 0)),
               inner=int(g.meta.get(a + "expert_feed_forward_length", 0)), dense_part=dpart,
               types=dict(dense=kind_type("dense"), experts=kind_type("experts"), ngram=kind_type("ngram")),
               tensors=tens, order=order, meta=meta, part_bytes=list(used))
    g.close()
    return out


def fill_gguf(sh, index, segs, threads, log):
    """The tensors of the GGUF file into the segments: bytes as they are, read
    in the order of the file (O_DIRECT; the NVMe streams), copied by threads."""
    path, base = index["file"], index["base"]
    tens, offs = index["tensors"], index["offs"]
    merged = {}

    def add(a, b, p, dst):
        for x in range(a, b, CHUNK):
            merged[(path, x, min(b, x + CHUNK))] = (p, dst + (x - a))
    for gname in index["order"]:
        d = tens[gname]
        src = base + offs[gname]
        if d["place"] == "node":
            add(src, src + d["bytes"], d["seg"], d["off"])
        elif d["place"] == "experts":
            for p, off, ids in d["parts"]:
                for slot, e in enumerate(ids):
                    add(src + e * d["nb"], src + (e + 1) * d["nb"], p, off + slot * d["nb"])
        else:
            for p, off, r0, nr in d["parts"]:
                add(src + r0 * d["rb"], src + (r0 + nr) * d["rb"], p, off)

    def handle(v, h):
        p, dst = h
        segs[p].array(dst, v.nbytes)[:] = v
    _stream(merged, threads, handle, log, "the tensors of %s" % os.path.basename(path))


class _GGUFPlugin:
    """The weightkeeper's plugin for a GGUF file (reclank/runtime/weightkeeper.py)."""
    layout = staticmethod(layout_gguf)
    fill = staticmethod(fill_gguf)
    ident = staticmethod(gguf_ident)


GGUF_PLUGIN = _GGUFPlugin()
