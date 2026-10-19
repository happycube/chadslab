"""The weightkeeper: the expert rows of DeepSeek V4.1 Flash loaded once for many
rounds (DEEPSEEK_V41_PLAN.md 4.2; the command: scripts/weightkeeper.py).

A process that reads the routed experts of the checkpoint once (289 GB,
about 100 s from the NVMe) into a segment for each NUMA node, and hands the
segments to the rounds (a new build, a new script, the server) over a Unix
socket, so a round starts in seconds:

- **The segments.** The rows of each expert are split between the parts as
  moe_parts.split splits them (the rows of inner for w1 and w3, of hidden for
  w2; a part for each node): node p holds its rows of every expert of every
  layer. A segment is made of memfds in the largest pages it gets (pages()):
  1 GB hugetlb pages, then 2 MB hugetlb pages, then shmem pages (2 MB
  transparent huge pages when shmem_enabled allows them, else 4 KB). The
  memfds are mapped side by side at one address (map_chunks), bound to their
  node with mbind before a page is touched, and filled by MADV_POPULATE_WRITE
  (an error, not a SIGBUS, when the pool runs dry) from threads on the node.
- **The layout** (LAYOUT, v2, the default): for each layer (in the order of
  index["layers"]), each expert, the matrices w1, w3, w2 of the part's rows
  in KQ_MXX (csrc/kernels/kq/mxx.c: the E2M1 codes and E8M0 scales of the
  checkpoint in groups of 16 rows, the form the products read; packed at the
  load, the values unchanged), each at a multiple of 64 bytes:
  index["mats"][part] = [name, offset, first row, rows, cols, bytes].
  LAYOUT_V1 (layout="planes"): the planes w1.weight, w1.scale, ... of the
  part's rows as the shard has them (index["planes"]); a round reads both.
- **The handoff.** A round connects (attach()), checks the uid of the peer,
  gets the index (JSON) and the memfds (SCM_RIGHTS), and maps them at one
  address in the same order; no copy, the pages stay where the weightkeeper
  put them. Each memfd is sealed once it is filled: no resize, and with
  seal="write" no write (the rounds map it read-only); with seal="size" (the
  default) the rounds map it writable (index["writable"]), so CUDA can
  page-lock it for DMA (cudaHostRegister of a read-only map: "operation not
  supported" on the 3090; the copies then stage, a prompt of 8192 tokens of
  the EXL3 Flash-Next 654 tok/s against 871 page-locked). The arrays of a
  round stay read-only (Weights._arr); the seal is the guard against a
  stray write of C or of the GPU.
- **The checkpoint** of the segments: the size and a sha256 of the header of
  each shard in the index (Weights.check against the files of a round).
- **The dense weights** (18.7 GB: the attention, the shared experts, the
  DSpark experts, embed and head, vision) stay in the shards by default: next
  to 144.4 GB of expert rows they do not fit the RAM of a node (4.1); a round
  streams them straight to the GPU (the FP8 bytes as they are, the bfloat16
  matrices as BF12; about 9 s from the NVMe). dense=True puts their bytes in a
  segment on the GPU's node too (index["dense"]["tensors"]: name -> offset,
  dtype, shape), for a smaller test or a machine with room.

As root (the root parent, run()): the weightkeeper fills the pools of hugetlb pages
it needs (drop the page cache, compact the node, ask for the pages), and a
parent that stays root gives them back when the weightkeeper exits. lock=True
(--lock) also locks the memory of each node in full: mlock of every segment,
vm.compact_unevictable_allowed 0, the MPOL_BIND policy (no NUMA balancing), a
check of /proc/self/numa_maps, all or nothing. Then the weightkeeper drops to the
invoking user (SUDO_UID) before it serves.

    run(Options(ckpt="/path/DeepSeek-V4.1-Flash", lock=True))   # the weightkeeper
    w = attach()                                                # a round
    planes = w.planes(layer=3, part=0)   # dict: name -> uint8 (E, rows, bytes)
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import re
import resource
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from reclank.runtime import numa

LAYOUT = "dsv41-expert-mxx-v2"
LAYOUT_V1 = "dsv41-expert-planes-v1"
LAYOUT_TENSORS = "tensors-v1"       # the GGUF tensors of a model (reclank/models/qwen/keep.py)
MATS = ("w1", "w3", "w2")
VERSION = 1
PLANES = ("w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale")
EXPERT = "layers.%d.ffn.experts.%d.%s"

G1, M2, K4 = 1 << 30, 2 << 20, 4096
ALIGN = 64

MFD_CLOEXEC, MFD_ALLOW_SEALING, MFD_HUGETLB = 1, 2, 4
MFD_HUGE = {G1: 30 << 26, M2: 21 << 26}
PROT_NONE, PROT_READ, PROT_WRITE = 0, 1, 2
MAP_SHARED, MAP_PRIVATE, MAP_FIXED, MAP_ANONYMOUS = 0x01, 0x02, 0x10, 0x20
MAP_NORESERVE, MAP_POPULATE = 0x4000, 0x8000
MADV_HUGEPAGE, MADV_POPULATE_READ, MADV_POPULATE_WRITE = 14, 22, 23
# F_ADD_SEALS and the seals (not in the fcntl of every Python build)
F_ADD_SEALS, F_GET_SEALS = 1033, 1034
SEALS = 1 | 2 | 4 | 8           # F_SEAL_SEAL, SHRINK, GROW, WRITE
# F_SEAL_FUTURE_WRITE: no new writable map or write(), the maps there stay.
# A hugetlb memfd with its pages refuses F_SEAL_WRITE (EBUSY: the pin check
# of memfd_wait_for_pins counts a hugetlb folio once in the page cache and
# compares it with its count of small pages), so it gets this one instead.
SEALS_FUTURE = 1 | 2 | 4 | 16
SEALS_SIZE = 1 | 2 | 4          # (seal="size": no resize, writes allowed)
PR_SET_PDEATHSIG = 1

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_long)
_libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
_libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
_libc.mlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
_libc.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
_libc.memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)


def memfd_create(name, flags):
    """memfd_create of libc (os.memfd_create is not in every Python build)."""
    return _check(_libc.memfd_create(name.encode(), flags), "memfd_create")


def _check(r, what):
    if r == -1 or r == ctypes.c_void_p(-1).value:
        e = ctypes.get_errno()
        raise OSError(e, "%s: %s" % (what, os.strerror(e)))
    return r


def _mmap(addr, size, prot, flags, fd=-1):
    return _check(_libc.mmap(addr, size, prot, flags, fd, 0), "mmap")


def _up(n, a):
    return (n + a - 1) // a * a


def log(msg):
    print("weightkeeper: " + msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- the checkpoint

class Shards:
    """The headers of the shards: name -> (file, dtype, shape, start, end), the
    offsets in the file; and the size and the sha256 of the header of each."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        self.hf = self.cfg.get("text_config", self.cfg)
        where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.t = {}
        self.ident = {}
        for f in sorted(set(where.values())):
            fp = os.path.join(path, f)
            with open(fp, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                raw = fh.read(n)
            h = json.loads(raw)
            h.pop("__metadata__", None)
            for k, v in h.items():
                o0, o1 = v["data_offsets"]
                self.t[k] = (f, v["dtype"], tuple(v["shape"]), 8 + n + o0, 8 + n + o1)
            self.ident[f] = [os.path.getsize(fp), hashlib.sha256(raw).hexdigest()]

    def file(self, f):
        return os.path.join(self.path, f)


def identity(path):
    """The size and the sha256 of the header of each shard of the checkpoint at
    path (the check of a round: Weights.check), or of a GGUF file."""
    if path.endswith(".gguf"):
        from reclank.models.qwen.keep import gguf_ident
        return gguf_ident(path)
    from reclank.models.qwen import keep_exl3
    if keep_exl3.detect(path):
        return keep_exl3.ident(path)    # (an EXL3 directory: every safetensors file)
    where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for f in sorted(set(where.values())):
        fp = os.path.join(path, f)
        with open(fp, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            out[f] = [os.path.getsize(fp), hashlib.sha256(fh.read(n)).hexdigest()]
    return out


def expert_layout(sh, nodes, share, layers, layout="mxx"):
    """The index of the expert segments (without the chunks): the rows of each
    part, the matrices (v2) or planes (v1) of an expert of each part, the
    stride of an expert and of a layer, the bytes of each part."""
    from reclank.runtime.moe_parts import split
    hf = sh.hf
    L, E = int(hf["num_hidden_layers"]), int(hf["n_routed_experts"])
    hidden, inner = int(hf["hidden_size"]), int(hf["moe_intermediate_size"])
    rb = {}
    for p in PLANES:
        _f, dt, shape, o0, o1 = sh.t[EXPERT % (0, 0, p)]
        rb[p] = (o1 - o0) // shape[0]
        assert shape[0] == (inner if p[:2] in ("w1", "w3") else hidden), (p, shape)
    parts = len(nodes)
    gu = split(inner, parts, share=share) if parts > 1 else [(0, inner)]
    dn = split(hidden, parts, share=share) if parts > 1 else [(0, hidden)]
    planes, mats, stride = [], [], []
    for k in range(parts):
        off, pl = 0, []
        if layout == "planes":
            for p in PLANES:
                r0, nr = gu[k] if p[:2] in ("w1", "w3") else dn[k]
                pl.append([p, off, r0, nr, rb[p]])
                off = _up(off + nr * rb[p], ALIGN)
            planes.append(pl)
        else:
            for w in MATS:
                r0, nr = gu[k] if w != "w2" else dn[k]
                cols = hidden if w != "w2" else inner
                assert nr % 16 == 0, (w, nr)
                n = nr * (cols // 32) * 17
                pl.append([w, off, r0, nr, cols, n])
                off = _up(off + n, ALIGN)
            mats.append(pl)
        stride.append(off)
    return dict(version=VERSION, layout=LAYOUT if layout != "planes" else LAYOUT_V1, model="deepseek_v41",
                checkpoint=sh.path, shards=sh.ident, L=L, E=E, hidden=hidden, inner=inner,
                layers=list(layers), nodes=list(nodes), share=list(share), gu=[list(x) for x in gu],
                dn=[list(x) for x in dn], planes=planes, mats=mats, stride=stride,
                layer_stride=[E * s for s in stride],
                part_bytes=[len(layers) * E * s for s in stride])


def dense_names(sh):
    """The tensors that are not routed experts of the main layers nor engram
    tables, in the order of the files."""
    out = []
    for k, v in sorted(sh.t.items(), key=lambda kv: (kv[1][0], kv[1][3])):
        if re.match(r"layers\.\d+\.ffn\.experts\.", k) or re.match(r"layers\.\d+\.engram\.embed\.", k):
            continue
        out.append(k)
    return out


# ---------------------------------------------------------------- pages and pools

def _node_dir(node, page):
    return "/sys/devices/system/node/node%d/hugepages/hugepages-%dkB" % (node, page // 1024)


def _read_int(path, default=0):
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return default


def _write(path, value):
    with open(path, "w") as f:
        f.write(str(value))


def _sysfs_choice(path):
    """The [chosen] word of a sysfs choice file, or None."""
    try:
        m = re.search(r"\[(\w+)\]", open(path).read())
        return m.group(1) if m else None
    except OSError:
        return None


SHMEM_THP = "/sys/kernel/mm/transparent_hugepage/shmem_enabled"
COMPACT_UNEVICT = "/proc/sys/vm/compact_unevictable_allowed"


def system_state(nodes):
    """What the root parent restores: the pools of each node and page size,
    shmem_enabled, compact_unevictable_allowed."""
    pools = {}
    for nd in nodes:
        for pg in (G1, M2):
            d = _node_dir(nd, pg)
            if os.path.isdir(d):
                pools[d] = _read_int(d + "/nr_hugepages")
    return dict(pools=pools, shmem=_sysfs_choice(SHMEM_THP), compact=_read_int(COMPACT_UNEVICT, None))


def restore_system(st):
    """The pools and the settings of st back, each write checked by reading it
    back; what failed is logged."""
    for d, n in st["pools"].items():
        try:
            _write(d + "/nr_hugepages", n)
        except OSError as e:
            log("could not give back the pool %s: %s" % (d, e))
    for path, want, now in ((SHMEM_THP, st["shmem"], lambda: _sysfs_choice(SHMEM_THP)),
                            (COMPACT_UNEVICT, st["compact"], lambda: _read_int(COMPACT_UNEVICT, None))):
        if want is None:
            continue
        try:
            _write(path, want)
        except OSError as e:
            log("could not restore %s to %s: %s" % (path, want, e))
        if str(now()) != str(want):
            log("%s is %s, not %s" % (path, now(), want))


def pages(need, node, root, maxpage=G1):
    """The chunks of a segment of need bytes on node: [(page, bytes)], the
    largest pages first. As root it asks the pool of the node for the pages
    it needs (for 1 GB pages after dropping the page cache and compacting the
    node); else it takes the free pages of pools filled beforehand. The rest
    in shmem pages (K4: 2 MB transparent huge pages where allowed)."""
    chunks, rest = [], need
    for pg in (G1, M2):
        d = _node_dir(node, pg)
        if pg > maxpage or rest <= 0 or not os.path.isdir(d):
            continue
        want = (rest + pg - 1) // pg
        free = _read_int(d + "/free_hugepages")
        if root and free < want:
            if pg == G1:
                try:
                    _write("/proc/sys/vm/drop_caches", 1)
                    _write("/sys/devices/system/node/node%d/compact" % node, 1)
                except OSError:
                    pass
            nr = _read_int(d + "/nr_hugepages")
            try:
                _write(d + "/nr_hugepages", nr + want - free)
            except OSError as e:
                log("node %d: the pool of %s pages: %s" % (node, _pname(pg), e))
            free = _read_int(d + "/free_hugepages")
        n = min(want, free)
        if n > 0:
            chunks.append((pg, n * pg))
            rest -= n * pg
    if rest > 0:
        chunks.append((K4, _up(rest, M2)))
    return chunks


def _pname(pg):
    return {G1: "1 GB", M2: "2 MB", K4: "4 KB"}[pg]


# ---------------------------------------------------------------- the segments

class Segment:
    """Memfds side by side at one address: chunks [(page, bytes)]."""

    def __init__(self, name, node, chunks, numa_on):
        self.name, self.node, self.numa_on = name, node, numa_on
        self.chunks = []        # [page, bytes, offset]
        self.fds = []
        off = 0
        for pg, n in chunks:
            flags = MFD_CLOEXEC | MFD_ALLOW_SEALING
            if pg in MFD_HUGE:
                flags |= MFD_HUGETLB | MFD_HUGE[pg]
            fd = memfd_create("%s-%s" % (name, _pname(pg).replace(" ", "")), flags)
            os.ftruncate(fd, n)
            self.fds.append(fd)
            self.chunks.append([pg, n, off])
            off += n
        self.size = off
        self.base = map_chunks(self.fds, self.chunks, PROT_READ | PROT_WRITE)
        if numa_on:
            numa._mbind(self.base, self.size, node, policy=numa.MPOL_BIND)
        for (pg, n, o) in self.chunks:
            if pg == K4:
                _libc.madvise(self.base + o, n, MADV_HUGEPAGE)

    def array(self, off, n):
        return np.frombuffer((ctypes.c_uint8 * n).from_address(self.base + off), np.uint8)

    def populate(self, threads):
        """Every page of the segment, by threads on the CPUs of its node (the
        kernel zeroes a page in the thread that faults it): MADV_POPULATE_WRITE
        gives an error where a fault in the fill would give a SIGBUS (a pool
        short of pages). The shmem pages too: faulted in the fill, 4 layers
        of 4 KB pages took 31.7 s against 14.4 + 13.1 s populated first (the
        copies of the fill wait on the faults; the reads alone run at the
        2.2 GB/s of the NVMe)."""
        cpus = sorted(numa.node_cpus(self.node) & os.sched_getaffinity(0)) if self.numa_on else []
        pieces = []
        for pg, n, o in self.chunks:
            step = max(pg, 64 * M2)
            pieces += [(o + x, min(step, n - x)) for x in range(0, n, step)]
        errs, lock = [], threading.Lock()

        def work(k):
            if cpus:
                os.sched_setaffinity(0, cpus)
            while True:
                with lock:
                    if not pieces:
                        return
                    o, n = pieces.pop()
                if _libc.madvise(self.base + o, n, MADV_POPULATE_WRITE) != 0:
                    errs.append((o, os.strerror(ctypes.get_errno())))
        ts = [threading.Thread(target=work, args=(k,)) for k in range(threads)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if errs:
            raise OSError("%s: node %d: %d pieces not populated (%s at %d)" % (
                self.name, self.node, len(errs), errs[0][1], errs[0][0]))

    def seal(self, write=True):
        """write=False: the seals of the size alone (the rounds may map the
        segment writable), then a read-only map of the weightkeeper's own.
        Else the seals, then a read-only map in place of the writable one. A
        seal of the writes needs no shared map that may write (a read-only
        MAP_SHARED of the fd counts too), so the range holds an empty
        reservation while the seals go on; after the seal a shared map can
        never be made writable. A hugetlb chunk that refuses F_SEAL_WRITE
        takes F_SEAL_FUTURE_WRITE (SEALS_FUTURE). Then the check: a writable
        shared map of each fd must fail. Returns the chunks whose seal is not
        enforced (empty when all are)."""
        weak = []
        for fd, (pg, n, o) in zip(self.fds, self.chunks):
            if not write:
                fcntl.fcntl(fd, F_ADD_SEALS, SEALS_SIZE)
                _mmap(self.base + o, n, PROT_READ, MAP_SHARED | MAP_FIXED | MAP_POPULATE, fd)
                continue
            _mmap(self.base + o, n, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED)
            try:
                fcntl.fcntl(fd, F_ADD_SEALS, SEALS)
            except OSError as e:
                if e.errno != 16:
                    raise OSError(e.errno, "%s: the seals of the %s chunk: %s" % (
                        self.name, _pname(pg), e.strerror))
                fcntl.fcntl(fd, F_ADD_SEALS, SEALS_FUTURE)
            _mmap(self.base + o, n, PROT_READ, MAP_SHARED | MAP_FIXED | MAP_POPULATE, fd)
            w = _libc.mmap(None, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
            if w != ctypes.c_void_p(-1).value:
                _libc.munmap(w, n)
                weak.append(_pname(pg))
        if self.numa_on:
            numa._mbind(self.base, self.size, self.node, policy=numa.MPOL_BIND)
        return weak

    def lock(self):
        return _libc.mlock(self.base, self.size) == 0 or ctypes.get_errno()

    def numa_pages(self):
        """The bytes of the segment on each node (/proc/self/numa_maps)."""
        out = {}
        starts = {self.base + o: pg for pg, _n, o in self.chunks}
        for line in open("/proc/self/numa_maps"):
            f = line.split()
            a = int(f[0], 16)
            if a < self.base or a >= self.base + self.size:
                continue
            kb = next((int(x.split("=")[1]) for x in f if x.startswith("kernelpagesize_kB=")),
                      starts.get(a, K4) // 1024)
            for x in f:
                m = re.match(r"N(\d+)=(\d+)$", x)
                if m:
                    out[int(m.group(1))] = out.get(int(m.group(1)), 0) + int(m.group(2)) * kb * 1024
        return out

    def describe(self):
        return dict(node=self.node, bytes=self.size, chunks=self.chunks)


def map_chunks(fds, chunks, prot, populate=False):
    """The memfds of chunks ([page, bytes, offset]) at one address aligned to
    the largest page (a hugetlb map needs an address of its page size)."""
    size = sum(n for _pg, n, _o in chunks)
    align = max([pg for pg, _n, _o in chunks] + [M2])
    resv = _mmap(None, size + align, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE)
    base = _up(resv, align)
    if base > resv:
        _libc.munmap(resv, base - resv)
    tail = resv + size + align - (base + size)
    if tail:
        _libc.munmap(base + size, tail)
    flags = MAP_SHARED | MAP_FIXED | (MAP_POPULATE if populate else 0)
    for fd, (pg, n, o) in zip(fds, chunks):
        _mmap(base + o, n, prot, flags, fd)
    return base


# ---------------------------------------------------------------- the fill

class Reader:
    """Reads byte ranges of the shards without the page cache (O_DIRECT into a
    bounce buffer of each thread; a buffered read and POSIX_FADV_DONTNEED when
    the file system has no O_DIRECT)."""

    def __init__(self, sh, maxbytes):
        self.sh = sh
        self.fds = {}
        self.direct = {}
        self.local = threading.local()
        self.maxbytes = _up(maxbytes, K4) + 2 * K4
        self.lock = threading.Lock()

    def _fd(self, f):
        with self.lock:
            if f not in self.fds:
                p = self.sh.file(f)
                try:
                    self.fds[f] = os.open(p, os.O_RDONLY | os.O_DIRECT)
                    self.direct[f] = True
                except OSError:
                    self.fds[f] = os.open(p, os.O_RDONLY)
                    self.direct[f] = False
            return self.fds[f], self.direct[f]

    def _buf(self, slot):
        bufs = getattr(self.local, "bufs", None)
        if bufs is None:
            bufs = self.local.bufs = {}
        b = bufs.get(slot)
        if b is None:
            addr = _mmap(None, self.maxbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS)
            b = bufs[slot] = (ctypes.c_uint8 * self.maxbytes).from_address(addr)
        return b

    def read(self, f, start, end, slot=0):
        """The bytes [start, end) of shard f: a uint8 view of the bounce buffer
        slot of the thread (valid until its next read into that slot)."""
        fd, direct = self._fd(f)
        buf = self._buf(slot)
        mv = memoryview(buf).cast("B")
        a0 = start & ~(K4 - 1) if direct else start
        a1 = _up(end, K4) if direct else end
        got = 0
        while a0 + got < end:
            try:
                r = os.preadv(fd, [mv[got:a1 - a0]], a0 + got)
            except OSError as e:
                if direct and e.errno == 22:
                    with self.lock:
                        os.close(self.fds.pop(f))
                        self.fds[f] = os.open(self.sh.file(f), os.O_RDONLY)
                        self.direct[f] = False
                    return self.read(f, start, end, slot)
                raise
            if r <= 0:
                raise OSError("%s: short read at %d" % (f, a0 + got))
            got += r
        if not direct:
            os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_DONTNEED)
        return np.frombuffer(buf, np.uint8, count=end - a0)[start - a0:]

    def close(self):
        for fd in self.fds.values():
            os.close(fd)


def fill(sh, index, segs, dense_seg, threads):
    """The expert rows of every part (and the dense tensors) from the shards,
    by threads; each expert matrix read once (v2: its codes and its scales,
    then each part's rows packed into its segment)."""
    from reclank.backends.cpu import cops
    v2 = index["layout"] == LAYOUT
    items = []
    for k, layer in enumerate(index["layers"]):
        for e in range(index["E"]):
            if v2:
                for mi, w in enumerate(MATS):
                    f, _dt, _shape, o0, o1 = sh.t[EXPERT % (layer, e, w + ".weight")]
                    items.append((f, o0, o1, "m", (k, e, mi)))
                continue
            for pi, p in enumerate(PLANES):
                f, _dt, _shape, o0, o1 = sh.t[EXPERT % (layer, e, p)]
                items.append((f, o0, o1, "x", (k, e, pi)))
    if dense_seg is not None:
        for name, (off, _dt, _shape) in index["dense"]["tensors"].items():
            f, _dt, _shape, o0, o1 = sh.t[name]
            items.append((f, o0, o1, "d", off))
    items.sort(key=lambda x: (x[0], x[1]))
    total = sum(it[2] - it[1] for it in items)
    rd = Reader(sh, max(it[2] - it[1] for it in items))
    pos, done, lock, errs = [0], [0], threading.Lock(), []
    t0 = time.time()

    def work():
        try:
            while True:
                with lock:
                    if pos[0] >= len(items) or errs:
                        return
                    f, o0, o1, kind, where = items[pos[0]]
                    pos[0] += 1
                src = rd.read(f, o0, o1)
                if kind == "d":
                    dense_seg.array(where, o1 - o0)[:] = src
                elif kind == "m":
                    k, e, mi = where
                    w = MATS[mi]
                    sf, _sdt, _sshape, s0, s1 = sh.t[EXPERT % (index["layers"][k], e, w + ".scale")]
                    scl = rd.read(sf, s0, s1, slot=1)
                    for part, seg in enumerate(segs):
                        _w, off, r0, nr, cols, n = index["mats"][part][mi]
                        at = k * index["layer_stride"][part] + e * index["stride"][part] + off
                        cops._lib.rc_kq_mxx_pack_serial(src.ctypes.data + r0 * (cols // 2),
                                                        scl.ctypes.data + r0 * (cols // 32), nr, cols,
                                                        seg.base + at)
                    with lock:
                        done[0] += s1 - s0
                else:
                    k, e, pi = where
                    for part, seg in enumerate(segs):
                        _p, off, r0, nr, rb = index["planes"][part][pi]
                        at = k * index["layer_stride"][part] + e * index["stride"][part] + off
                        seg.array(at, nr * rb)[:] = src[r0 * rb:(r0 + nr) * rb]
                with lock:
                    done[0] += o1 - o0
        except BaseException as ex:      # noqa: BLE001 - reported by the caller
            errs.append(ex)

    ts = [threading.Thread(target=work, daemon=True) for _ in range(threads)]
    [t.start() for t in ts]
    last = t0
    while any(t.is_alive() for t in ts):
        time.sleep(0.5)
        if time.time() - last >= 10:
            last = time.time()
            log("read %.1f of %.1f GB (%.2f GB/s)" % (done[0] / 1e9, total / 1e9,
                                                       done[0] / 1e9 / (last - t0)))
    rd.close()
    if errs:
        raise errs[0]
    dt = time.time() - t0
    log("read %.1f GB in %.1f s (%.2f GB/s; O_DIRECT %s)" % (
        total / 1e9, dt, total / 1e9 / max(dt, 1e-9), "yes" if all(rd.direct.values()) else "no"))


# ---------------------------------------------------------------- the weightkeeper

@dataclass
class Options:
    ckpt: str
    share: float = 0.5
    nodes: list = None          # None: the nodes with CPUs (all of them)
    layers: list = None         # None: all
    threads: int = 16
    lock: bool = False
    dense: bool = False
    dense_node: int = 0
    maxpage: int = G1
    layout: str = "mxx"         # "mxx" (v2) or "planes" (v1)
    # Qwen3.8-Flash-Next (reclank/models/qwen/keep.py): the types of ORIG's
    # bfloat16 tensors, the least they keep: bf12 (the default), bf16, q8 (rq8
    # for the experts)
    dense_type: str = "bf12"
    experts_type: str = "bf12"
    ngram_type: str = "bf12"
    counts: str = None          # the counts of the router (layers x experts, .npy): the experts' nodes
    # an EXL3 directory (reclank/models/qwen/keep_exl3.py): the stacks of
    # experts "split" between the nodes, "both" (whole on every node), or
    # "auto" (both when they fit)
    copies: str = "auto"
    # the seals of the segments: "size" (no resize: the rounds map them
    # writable, so CUDA page-locks them) or "write" (no write either: the
    # rounds map them read-only)
    seal: str = "size"
    sock: str = None
    user: int = None            # the uid to drop to as root (default SUDO_UID)
    group: int = None
    extra: dict = field(default_factory=dict)


def default_socket(uid=None):
    uid = os.getuid() if uid is None else uid
    return "/tmp/reclank-weightkeeper-%d.sock" % uid


def plugin(ckpt):
    """The module of a model that the weightkeeper takes in the tensors layout
    (layout(), fill()), or None: DeepSeek V4.1 (this module's layouts)."""
    from reclank.models.qwen import keep as qwen_keep
    from reclank.models.qwen import keep_exl3, keep_fp4l
    from reclank.models.nemotron import keep as nemotron_keep
    if nemotron_keep.detect(ckpt):
        return nemotron_keep.PLUGIN         # Nemotron 3.5 Bolt (the expert stacks, the KQ_NV4 rows)
    if keep_exl3.detect(ckpt):
        return keep_exl3.PLUGIN             # an EXL3 directory of Qwen3.8 (its MUL1 blocks)
    if keep_fp4l.detect(ckpt):
        return keep_fp4l.PLUGIN             # nvidia/Qwen3.8-Flash-Next-FP4-Local (KQ_NVX experts)
    if qwen_keep.detect_gguf(ckpt):
        return qwen_keep.GGUF_PLUGIN        # a GGUF file of Qwen3.8 (its tensors as they are)
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    if qwen_keep.detect(cfg):
        return qwen_keep
    return None


class Weightkeeper:
    def __init__(self, o: Options):
        self.o = o
        self.root = os.geteuid() == 0
        self.numa_on = numa.enabled()
        t0 = time.time()
        self.plugin = plugin(o.ckpt)
        if hasattr(self.plugin, "ident"):
            # a file of its own (a GGUF): its identity, no shards
            self.sh = Shards.__new__(Shards)
            self.sh.path, self.sh.ident = os.path.abspath(o.ckpt), self.plugin.ident(o.ckpt)
        else:
            self.sh = Shards(o.ckpt)
        all_nodes = sorted(set(numa.node_of_cpu().values())) if self.numa_on else [0]
        nodes = list(o.nodes) if o.nodes else all_nodes
        if self.plugin is not None:
            share = [o.share] + [(1.0 - o.share) / (len(nodes) - 1)] * (len(nodes) - 1) \
                if len(nodes) > 1 else [1.0]
            ix = self.index = self.plugin.layout(o.ckpt, nodes, share, o)
            ix.update(version=VERSION, checkpoint=self.sh.path, shards=self.sh.ident, dense=None)
            log("%s: %d files, %s (dense %s, experts %s, n-gram %s), the experts of each layer %s "
                "nodes %s: %s GB a node (%.1f s)" % (
                    o.ckpt, len(self.sh.ident), ix["model"], ix["types"]["dense"], ix["types"]["experts"],
                    ix["types"]["ngram"], "on all the" if ix.get("copies") == "both" else "split between",
                    nodes, ["%.1f" % (b / 1e9) for b in ix["part_bytes"]], time.time() - t0))
            return
        L = int(self.sh.hf["num_hidden_layers"])
        layers = list(o.layers) if o.layers is not None else list(range(L))
        share = [o.share] + [(1.0 - o.share) / (len(nodes) - 1)] * (len(nodes) - 1) \
            if len(nodes) > 1 else [1.0]
        self.index = expert_layout(self.sh, nodes, share, layers, o.layout)
        if o.dense:
            tens, off = {}, 0
            for name in dense_names(self.sh):
                _f, dt, shape, o0, o1 = self.sh.t[name]
                tens[name] = [off, dt, list(shape)]
                off = _up(off + o1 - o0, ALIGN)
            self.index["dense"] = dict(node=o.dense_node, bytes=off, tensors=tens)
        else:
            self.index["dense"] = None
        log("the checkpoint: %d shards, the rows of %d layers x %d experts on nodes %s "
            "(inner %s, hidden %s): %s GB a node; dense %s (%.1f s)" % (
                len(self.sh.ident), len(layers), self.index["E"], nodes,
                [n for _a, n in self.index["gu"]], [n for _a, n in self.index["dn"]],
                ["%.1f" % (b / 1e9) for b in self.index["part_bytes"]],
                "%.1f GB on node %d" % (self.index["dense"]["bytes"] / 1e9, o.dense_node)
                if o.dense else "in the shards", time.time() - t0))

    def load(self):
        o, ix = self.o, self.index
        t0 = time.time()
        self.segs = []
        name = "%s-n%%d" % ("dsv41-experts" if self.plugin is None else ix["model"])
        for part, nd in enumerate(ix["nodes"]):
            ch = pages(ix["part_bytes"][part], nd, self.root, o.maxpage)
            self.segs.append(Segment(name % nd, nd, ch, self.numa_on))
        self.dense = None
        if ix["dense"]:
            ch = pages(ix["dense"]["bytes"], o.dense_node, self.root, o.maxpage)
            self.dense = Segment("dsv41-dense-n%d" % o.dense_node, o.dense_node, ch, self.numa_on)
        every = self.segs + ([self.dense] if self.dense else [])
        log("the pages in %.1f s" % (time.time() - t0))
        t0 = time.time()
        for s in every:
            log("%s: %s" % (s.name, ", ".join("%s %.1f GB" % (_pname(pg), n / 1e9) for pg, n, _o in s.chunks)))
        per = max(1, o.threads // max(1, len(every)))
        errs = []

        def run(s):
            try:
                s.populate(per)
            except OSError as e:
                errs.append(e)
        ts = [threading.Thread(target=run, args=(s,)) for s in every]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if errs:
            raise errs[0]
        log("populated in %.1f s" % (time.time() - t0))
        if self.plugin is not None:
            self.plugin.fill(self.sh, ix, self.segs, o.threads, log)
        else:
            fill(self.sh, ix, self.segs, self.dense, o.threads)
        ix["writable"] = o.seal == "size"
        for s in every:
            weak = s.seal(write=not ix["writable"])
            seals = {fcntl.fcntl(fd, F_GET_SEALS) for fd in s.fds}
            log("%s: sealed (%s)%s" % (
                s.name, ", ".join("write" if x & 8 else "future write" if x & 16
                                  else "size: the rounds map it writable" if ix["writable"] else "no write seal"
                                  for x in sorted(seals)),
                "; WARNING: a writable map still works for the %s chunk" % ", ".join(weak) if weak else ""))
        bad = []
        for s in every:
            where = s.numa_pages()
            off = {n: b for n, b in where.items() if n != s.node}
            log("%s: on the nodes %s" % (s.name, {n: "%.1f GB" % (b / 1e9) for n, b in sorted(where.items())}))
            if off and self.numa_on:
                bad.append("%s: %.1f GB off node %d" % (s.name, sum(off.values()) / 1e9, s.node))
        if o.lock:
            for s in every:
                r = s.lock()
                if r is not True:
                    bad.append("%s: mlock: %s" % (s.name, os.strerror(r)))
            if bad:
                raise SystemExit("weightkeeper: --lock: " + "; ".join(bad))
            log("locked: %.1f GB" % (sum(s.size for s in every) / 1e9))
        elif bad:
            log("warning: " + "; ".join(bad))
        ix["segments"] = [s.describe() for s in self.segs]
        if self.dense:
            ix["dense"].update(self.dense.describe())
        ix["locked"] = bool(o.lock)
        ix["pid"] = os.getpid()
        log("ready in %.1f s" % (time.time() - t0))

    def fds(self):
        out = [fd for s in self.segs for fd in s.fds]
        if self.dense:
            out += self.dense.fds
        return out

    def serve(self, path):
        if os.path.exists(path):
            try:
                c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                c.connect(path)
                c.close()
                raise SystemExit("weightkeeper: a weightkeeper answers on %s" % path)
            except (ConnectionRefusedError, FileNotFoundError):
                os.unlink(path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        srv.bind(path)
        os.umask(old)
        srv.listen(8)
        self.path = path
        signal.signal(signal.SIGTERM, lambda *_a: sys.exit(0))
        signal.signal(signal.SIGINT, lambda *_a: sys.exit(0))
        log("serving on %s" % path)
        blob = json.dumps(self.index).encode()
        try:
            while True:
                c, _ = srv.accept()
                with c:
                    try:
                        uid = struct.unpack("3i", c.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                               12))[1]
                        if uid != os.getuid():
                            continue
                        req = _recv_msg(c).decode().strip()
                        if req == "get":
                            _send_msg(c, blob)
                            socket.send_fds(c, [b"F"], self.fds())
                        elif req == "status":
                            _send_msg(c, json.dumps(self.status()).encode())
                        elif req == "stop":
                            try:
                                _send_msg(c, b"stopping")
                            except OSError:
                                pass
                            return
                    except (OSError, ValueError) as e:
                        log("a client: %s" % e)
        finally:
            srv.close()
            try:
                os.unlink(path)
            except OSError:
                pass

    def status(self):
        ix = self.index
        return dict(pid=os.getpid(), model=ix.get("model", "deepseek_v41"), layout=ix["layout"],
                    types=ix.get("types"), locked=ix["locked"], layers=len(ix.get("layers", [])) or ix.get("L"),
                    nodes=ix["nodes"],
                    segments=[dict(node=s.node, gb=round(s.size / 1e9, 2),
                                   chunks=[[_pname(pg), round(n / 1e9, 2)] for pg, n, _o in s.chunks])
                              for s in self.segs + ([self.dense] if self.dense else [])],
                    checkpoint=ix["checkpoint"])


def _send_msg(c, b):
    c.sendall(struct.pack("<Q", len(b)) + b)


def _recv_exact(c, n):
    out = bytearray()
    while len(out) < n:
        b = c.recv(n - len(out))
        if not b:
            raise ConnectionError("the weightkeeper closed the socket")
        out += b
    return bytes(out)


def _recv_msg(c):
    n = struct.unpack("<Q", _recv_exact(c, 8))[0]
    return _recv_exact(c, n)


def _drop(uid, gid):
    os.setgroups([gid])
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)
    try:
        os.setuid(0)
    except PermissionError:
        pass
    else:
        raise SystemExit("weightkeeper: could not drop root")
    _pdeathsig()


def _pdeathsig():
    try:
        ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except (AttributeError, OSError):
        pass


def run(o: Options):
    """The weightkeeper, as the module text says: as root a parent that stays root
    and gives back the pools and the settings when the weightkeeper exits."""
    root = os.geteuid() == 0
    if o.lock and not root:
        raise SystemExit("weightkeeper: --lock needs root (sudo): no partial lock")
    if not root:
        z = Weightkeeper(o)
        z.load()
        return z.serve(o.sock or default_socket())
    uid = o.user if o.user is not None else int(os.environ.get("SUDO_UID", "0"))
    gid = o.group if o.group is not None else int(os.environ.get("SUDO_GID", "0"))
    if uid == 0:
        raise SystemExit("weightkeeper: as root it serves as the invoking user: run it with sudo, or give --user")
    path = o.sock or default_socket(uid)
    nodes = sorted(set(numa.node_of_cpu().values())) or [0]
    st = system_state(nodes)
    log("the settings now: shmem_enabled %s, compact_unevictable_allowed %s (given back at the end)" % (
        st["shmem"], st["compact"]))
    try:
        if st["shmem"] == "never":
            _write(SHMEM_THP, "advise")
        if o.lock:
            _write(COMPACT_UNEVICT, 0)
    except OSError as e:
        log("the settings: %s" % e)
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            _pdeathsig()
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            z = Weightkeeper(o)
            z.load()
            _drop(uid, gid)
            log("dropped to uid %d" % uid)
            z.serve(path)
        except SystemExit as e:
            if e.code not in (None, 0):
                print(e.code, file=sys.stderr)
                code = 1
        except BaseException as e:      # noqa: BLE001
            log("failed: %r" % e)
            code = 1
        os._exit(code)

    def fwd(sig, _frame):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    signal.signal(signal.SIGINT, fwd)
    signal.signal(signal.SIGTERM, fwd)
    while True:
        try:
            _p, status = os.waitpid(pid, 0)
            break
        except ChildProcessError:
            status = 0
            break
        except InterruptedError:
            continue
    restore_system(st)
    log("the pools and the settings given back")
    return os.waitstatus_to_exitcode(status)


# ---------------------------------------------------------------- a round

def request(op, path=None):
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(path or default_socket())
    _send_msg(c, op.encode())
    return c


def find(ckpt):
    """(the socket, the status) of the weightkeeper of this user that holds
    ckpt, or (None, None): the default socket, then those beside it named
    reclank-weightkeeper-<uid>-<name>.sock (scripts/weightkeeper.py --socket:
    a keeper for each model, Qwen3.6 and Gemma 26B on one machine)."""
    import glob
    d = default_socket()
    want = os.path.realpath(ckpt)
    for path in [d] + sorted(glob.glob(d[:-len(".sock")] + "-*.sock")):
        try:
            c = request("status", path)
            with c:
                st = json.loads(_recv_msg(c))
        except (OSError, ValueError):
            continue
        if os.path.realpath(st.get("checkpoint", "")) == want:
            return path, st
    return None, None


class Weights:
    """The segments of a weightkeeper, mapped in this process: read-only, or
    writable when the weightkeeper sealed only their size (self.writable:
    CUDA can page-lock them; the arrays stay read-only)."""

    def __init__(self, path=None, populate=True, ckpt=None):
        """ckpt: the checkpoint the round wants (with no path, the keeper that
        holds it: find); another one raises ValueError before anything is
        mapped."""
        if path is None and ckpt is not None:
            path = find(ckpt)[0]
        c = request("get", path)
        with c:
            self.index = json.loads(_recv_msg(c))
            nfd = sum(len(s["chunks"]) for s in self.index["segments"])
            if self.index.get("dense"):
                nfd += len(self.index["dense"]["chunks"])
            _msg, fds, _flags, _addr = socket.recv_fds(c, 16, nfd)
        if self.index["version"] != VERSION or self.index["layout"] not in (LAYOUT, LAYOUT_V1, LAYOUT_TENSORS):
            for fd in fds:
                os.close(fd)
            raise ValueError("the weightkeeper has the layout %r (version %s)" % (
                self.index["layout"], self.index["version"]))
        if ckpt is not None and identity(ckpt) != self.index["shards"]:
            for fd in fds:
                os.close(fd)
            raise ValueError("the weightkeeper holds another checkpoint (%s)" % self.index["checkpoint"])
        self.fds = list(fds)
        self.writable = bool(self.index.get("writable"))
        prot = PROT_READ | (PROT_WRITE if self.writable else 0)
        self.base, self.size, k = [], [], 0
        for s in self.index["segments"]:
            n = len(s["chunks"])
            self.base.append(map_chunks(self.fds[k:k + n], s["chunks"], prot, populate))
            self.size.append(s["bytes"])
            k += n
        self.dense_base = None
        if self.index.get("dense"):
            d = self.index["dense"]
            self.dense_base = map_chunks(self.fds[k:], d["chunks"], prot, populate)
        for fd in self.fds:
            os.close(fd)

    def check(self, ckpt):
        """Raise when the checkpoint at ckpt is not the one of the segments."""
        mine = identity(ckpt)
        if mine != self.index["shards"]:
            raise ValueError("the weightkeeper holds another checkpoint (%s)" % self.index["checkpoint"])

    def _arr(self, addr, n):
        a = np.frombuffer((ctypes.c_uint8 * n).from_address(addr), np.uint8)
        a.flags.writeable = False
        return a

    def planes(self, layer, part):
        """The planes of the rows of part for every expert of layer: name ->
        uint8 (E, rows, bytes of a row), strided views of the segment."""
        ix = self.index
        assert ix["layout"] == LAYOUT_V1, ix["layout"]
        k = ix["layers"].index(layer)
        base = self.base[part] + k * ix["layer_stride"][part]
        E, st = ix["E"], ix["stride"][part]
        seg = self._arr(base, E * st).reshape(E, st)
        return {p: seg[:, off:off + nr * rb].reshape(E, nr, rb) for p, off, _r0, nr, rb in ix["planes"][part]}

    def mats(self, layer, part):
        """v2: the KQ_MXX matrices of the rows of part for every expert of layer:
        name (w1, w3, w2) -> uint8 (E, bytes), strided views of the segment."""
        ix = self.index
        assert ix["layout"] == LAYOUT, ix["layout"]
        k = ix["layers"].index(layer)
        base = self.base[part] + k * ix["layer_stride"][part]
        E, st = ix["E"], ix["stride"][part]
        seg = self._arr(base, E * st).reshape(E, st)
        return {w: seg[:, off:off + n] for w, off, _r0, _nr, _cols, n in ix["mats"][part]}

    def tensor(self, name):
        """The tensors layout: a tensor whole on one node (uint8, its bytes), or
        for one split by rows, [(part, uint8 (E, nr x rb), first row, rows)]."""
        d = self.index["tensors"][name]
        if d["place"] == "node":
            return self._arr(self.base[d["seg"]] + d["off"], d["bytes"])
        if d["place"] == "experts":
            return [(p, self._arr(self.base[p] + off, len(ids) * d["nb"]).reshape(len(ids), d["nb"]), ids)
                    for p, off, ids in d["parts"]]
        return [(p, self._arr(self.base[p] + off, d["E"] * nr * d["rb"]).reshape(d["E"], nr * d["rb"]), r0, nr)
                for p, off, r0, nr in d["parts"]]

    def dense(self, name):
        off, dt, shape = self.index["dense"]["tensors"][name]
        from reclank.formats.safetensors import _NP
        dtype = np.dtype(np.uint16 if dt == "BF16" else _NP.get(dt, np.uint8))
        n = int(np.prod(shape)) * dtype.itemsize
        return self._arr(self.dense_base + off, n).view(dtype).reshape(shape)


def attach(path=None, populate=True, ckpt=None):
    """The segments of the weightkeeper of this user (or at path), mapped
    read-only; ckpt: ValueError when it holds another checkpoint."""
    return Weights(path, populate, ckpt)
