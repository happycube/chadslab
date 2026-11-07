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

**Several keepers.** A keeper with a name (Options.name, --name NAME) serves on a socket of its
own, socket_path(NAME) = /tmp/reclank-weightkeeper-<uid>-NAME.sock; without one, the default
socket as before. keepers() lists the ones that answer (--list). A round finds the keeper that
holds its checkpoint among all of them (find); RECLANK_WEIGHTKEEPER=NAME restricts it (and the
requests with no socket) to that keeper. What two keepers share: the host's memory (each checks
what it will hold against the memory available, per node and in all, before it takes a page:
check_memory), and as root the hugetlb pools and two settings (shmem_enabled,
compact_unevictable_allowed). A root keeper gives back only the pages it added to a pool (its
child reports them: the pools of another keeper stay), and the settings go back to the values
before the first root keeper when the last one exits (a state file /tmp/reclank-weightkeeper-
<uid>.settings.json under flock: the original values and the pids holding them). The memfds,
their names and the NUMA binding are per process. RECLANK_WEIGHTKEEPER_DIR: the directory of the
sockets and the state file (the tests).

**The catalog** (docs/GPU_SWAP_PLAN.md 3.7): a keeper holds entries (Entry: a checkpoint's index,
segments and plugin), each with an ID (its model and a hash of the checkpoint and the options that
shape it). The checkpoint it starts with is its first entry (none with --serve); "load OPTIONS"
(--add) makes another and reads it in the background, one load at a time, each checking the memory
before it takes a page; "drop ID" closes an entry (a round that maps it keeps its pages); "list"
and "status ID" tell them; "get ID|CKPT" hands one out. A plain "get" and "status" are the first
entry's, as before the catalog, so the rounds attach unchanged; find() looks through every entry.
As root only the first entry takes the pools' pages and the lock: later loads run as the user.

**Leases and the host budget** (3.7, step 2): a round's get keeps its socket open as its lease (the
keeper knows who maps what: the peer's pid; the lease ends when the round exits). With a host budget
(--host-budget GB) a load that would take the entries over it, or that the memory cannot take,
first drops the entries no round leases, the least recently leased first; a kept entry (--keep: an
ID or a checkpoint, "keep KEY" at run time) is never dropped. Without a budget nothing is dropped
but by "drop", and a load that does not fit fails, as before.

**The converted-weights cache** (3.7, step 3; off unless --save DIR): an entry's segments (the bytes
in use) and its index written once to DIR/<ID>-<code key>/ (in a directory of its own, renamed when
complete; not when the disk has no room); a load of the same entry reads them back with O_DIRECT
(no conversion, no page cache) before it reads the checkpoint. The key: the entry's ID (the
checkpoint's identity and the options) and code_key (the plugin's source, the CPU library's hash of
every C source, VERSION), and the layout made again must be the saved one; a copy of other code is
not read (and not removed: the log names it).
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import inspect
import json
import os
import re
import resource
import selectors
import signal
import socket
import struct
import sys
import threading
import time
import types
from dataclasses import dataclass, field

import numpy as np

from reclank.runtime import numa

LAYOUT = "dsv41-expert-mxx-v2"
LAYOUT_V1 = "dsv41-expert-planes-v1"
LAYOUT_TENSORS = "tensors-v1"       # the GGUF tensors of a model (reclank/models/qwen/keep.py)
LAYOUT_STORE = "store-v1"           # a model's host arrays, put by the process that made them (runtime/kept.py)
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


def _mmap(addr, size, prot, flags, fd=-1, offset=0):
    return _check(_libc.mmap(addr, size, prot, flags, fd, offset), "mmap")


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
    from reclank.models.wan import keep as wan_keep
    if wan_keep.detect(path):
        return wan_keep.ident(path)     # (Wan A14B: both experts' shards)
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


ADDED = {}      # the pages this process added to each pool (root): what its parent gives back


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
            ADDED[d] = ADDED.get(d, 0) + max(0, _read_int(d + "/nr_hugepages") - nr)
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

    @classmethod
    def of_fd(cls, name, node, fd, size, numa_on):
        """A segment of one memfd a round made and sealed (runtime/kept.py's put), mapped read-only."""
        s = cls.__new__(cls)
        s.name, s.node, s.numa_on = name, node, numa_on
        s.chunks, s.fds, s.size = [[K4, size, 0]], [fd], size
        s.base = map_chunks(s.fds, s.chunks, PROT_READ)
        return s

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
    # a Gemma 4 GGUF file: its Q4_0 matrices as KQ_Q4X ("auto" or "on"), or as the file has them ("off")
    q4x: str = "auto"
    # an EXL3 directory (reclank/models/qwen/keep_exl3.py): the stacks of
    # experts "split" between the nodes, "both" (whole on every node), or
    # "auto" (both when they fit)
    copies: str = "auto"
    # the seals of the segments: "size" (no resize: the rounds map them
    # writable, so CUDA page-locks them) or "write" (no write either: the
    # rounds map them read-only)
    seal: str = "size"
    sock: str = None
    name: str = None            # the keeper's name: its socket socket_path(name) (None: the default)
    user: int = None            # the uid to drop to as root (default SUDO_UID)
    group: int = None
    # the host budget (docs/GPU_SWAP_PLAN.md 3.7, step 2): the entries together at most host_budget
    # GB; a load over it (or one the memory cannot take) drops the entries no round leases, the least
    # recently leased first, but never a kept one: keep (this entry), keep_keys (the keeper's: IDs or
    # checkpoints). None: nothing is dropped but by "drop"
    host_budget: float = None
    keep: bool = False
    keep_keys: list = None
    # the converted-weights cache (3.7, step 3): the keeper's directory of the entries' segments (None:
    # none, the default): a load reads them there before the checkpoint, and writes them there once
    save: str = None
    extra: dict = field(default_factory=dict)


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MEM_MARGIN = 2 << 30        # what check_memory leaves free on a node


def _dir():
    return os.environ.get("RECLANK_WEIGHTKEEPER_DIR", "/tmp")


def default_socket(uid=None):
    uid = os.getuid() if uid is None else uid
    return os.path.join(_dir(), "reclank-weightkeeper-%d.sock" % uid)


def socket_path(name=None, uid=None):
    """The socket of the keeper named name (None: the default keeper's)."""
    d = default_socket(uid)
    if not name:
        return d
    if not _NAME.match(name):
        raise ValueError("a keeper's name: letters, digits, '_', '.', '-' (at most 64), not %r" % name)
    return d[:-len(".sock")] + "-" + name + ".sock"


def env_name():
    """RECLANK_WEIGHTKEEPER: the keeper a round asks (its name), or None: any of this user's."""
    return os.environ.get("RECLANK_WEIGHTKEEPER") or None


def name_of(path, uid=None):
    """The name of the keeper of a socket path (None: the default one)."""
    d = default_socket(uid)
    if path == d:
        return None
    pre = d[:-len(".sock")] + "-"
    return path[len(pre):-len(".sock")] if path.startswith(pre) and path.endswith(".sock") else path


def sockets(uid=None):
    """The sockets of this user's keepers that exist: the default one first, then the named."""
    import glob
    d = default_socket(uid)
    named = sorted(p for p in glob.glob(glob.escape(d[:-len(".sock")]) + "-*.sock")
                   if _NAME.match(name_of(p, uid) or ""))
    return [p for p in [d] + named if os.path.exists(p)]


def keepers(uid=None):
    """[(name, socket, status)] of the keepers that answer (name None: the default)."""
    out = []
    for path in sockets(uid):
        try:
            c = request("status", path)
            with c:
                st = json.loads(_recv_msg(c))
        except (OSError, ValueError):
            continue
        out.append((name_of(path, uid), path, st))
    return out


def _answers(path):
    try:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(path)
        c.close()
        return True
    except OSError:
        return False


def _meminfo(path):
    out = {}
    try:
        for line in open(path):
            f = line.replace(":", " ").split()
            # (/proc/meminfo: "Key: n kB"; a node's: "Node 0 Key: n kB")
            if f and f[0] == "Node":
                f = f[2:]
            if len(f) >= 2 and f[1].isdigit():
                out[f[0]] = int(f[1]) * (1024 if len(f) > 2 and f[2] == "kB" else 1)
    except OSError:
        pass
    return out


def _free_huge(node=None):
    """The bytes of the free hugetlb pages (of a node, or of all)."""
    n = 0
    for pg in (G1, M2):
        if node is None:
            n += _read_int("/sys/kernel/mm/hugepages/hugepages-%dkB/free_hugepages" % (pg // 1024)) * pg
        else:
            n += _read_int(_node_dir(node, pg) + "/free_hugepages") * pg
    return n


def memory_available(node=None):
    """What a keeper can take: the free memory and the page cache that can go (FilePages less
    Shmem: another keeper's memfds are Shmem, not available), and the free hugetlb pages; of a
    node, or of the host (MemAvailable)."""
    if node is None:
        m = _meminfo("/proc/meminfo")
        return m.get("MemAvailable", 0) + _free_huge()
    m = _meminfo("/sys/devices/system/node/node%d/meminfo" % node)
    return m.get("MemFree", 0) + max(0, m.get("FilePages", 0) - m.get("Shmem", 0)) + _free_huge(node)


def memory_short(need, numa_on=True, margin=MEM_MARGIN):
    """need: {node: bytes}. [(the node, None: the host; what it says)] where it does not fit the
    memory available."""
    bad = []
    total = sum(need.values())
    host = memory_available()
    if total > host - margin:
        bad.append((None, "the host: %.1f GB to hold, %.1f GB available" % (total / 1e9, host / 1e9)))
    for nd, n in sorted(need.items()) if numa_on else ():
        a = memory_available(nd)
        if n > a - margin:
            bad.append((nd, "node %d: %.1f GB to hold, %.1f GB available" % (nd, n / 1e9, a / 1e9)))
    return bad


def check_memory(need, numa_on=True, margin=MEM_MARGIN):
    """need: {node: bytes}. Raise SystemExit with what is missing when it does not fit the memory
    available (each node, and the host)."""
    bad = [m for _nd, m in memory_short(need, numa_on, margin)]
    if bad:
        held = ["%s %s %.1f GB" % (nm or "default", e.get("id"), sum(x["gb"] for x in e.get("segments", [])))
                for nm, _p, st in keepers() for e in st.get("entries") or [st] if e.get("segments")]
        raise SystemExit("weightkeeper: not enough memory: %s (with %.1f GB left free a node; keepers running: "
                         "%s)" % ("; ".join(bad), margin / 1e9, ", ".join(held) or "none"))


def plugin(ckpt):
    """The module of a model that the weightkeeper takes in the tensors layout
    (layout(), fill()), or None: DeepSeek V4.1 (this module's layouts)."""
    from reclank.models.qwen import keep as qwen_keep
    from reclank.models.qwen import keep_exl3, keep_fp4l, keep_q35nv
    from reclank.models.nemotron import keep as nemotron_keep
    from reclank.models.wan import keep as wan_keep
    if wan_keep.detect(ckpt):
        return wan_keep.PLUGIN              # Wan 2.2 T2V-A14B (both experts' F20 rows)
    if nemotron_keep.detect(ckpt):
        return nemotron_keep.PLUGIN         # Nemotron 3.5 Bolt (the expert stacks, the KQ_NV4 rows)
    if keep_exl3.detect(ckpt):
        return keep_exl3.PLUGIN             # an EXL3 directory of Qwen3.8 (its MUL1 blocks)
    if keep_fp4l.detect(ckpt):
        return keep_fp4l.PLUGIN             # nvidia/Qwen3.8-Flash-Next-FP4-Local (KQ_NVX experts)
    if keep_q35nv.detect(ckpt):
        return keep_q35nv.PLUGIN            # an NVFP4 Qwen3.5 / 3.6 MoE of ModelOpt (KQ_NVX experts)
    if qwen_keep.detect_gguf(ckpt):
        return qwen_keep.GGUF_PLUGIN        # a GGUF file of Qwen3.8 (its tensors as they are)
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    if qwen_keep.detect(cfg):
        return qwen_keep
    return None


def entry_id(o, model, ident):
    """The ID of an entry: its model and a hash of its checkpoint (the path, the files' identity) and
    of the options that shape what it holds."""
    keys = {k: v for k, v in vars(o).items() if k not in ("sock", "name", "user", "group", "threads", "lock",
                                                          "host_budget", "keep", "keep_keys", "save")}
    keys["ckpt"] = os.path.realpath(o.ckpt)
    h = hashlib.sha1(json.dumps([keys, ident], sort_keys=True, default=str).encode()).hexdigest()[:8]
    return "%s-%s" % (model, h)


RUNTIME_KEYS = ("segments", "locked", "pid", "writable")     # an index's keys of one load, not saved
SAVE_MARGIN = 4 << 30       # what a save leaves free on its disk
_PIECE = 64 << 20


def code_key(plugin):
    """A hash of the code that makes an entry's bytes: the plugin's module (DeepSeek: fill), the CPU
    library's name (a hash of every C source and the compiler) and VERSION. A saved entry is read back
    only with the same."""
    from reclank.backends.cpu import cops
    if plugin is None:
        src = inspect.getsource(fill)
    else:
        mod = plugin if isinstance(plugin, types.ModuleType) else sys.modules[type(plugin).__module__]
        src = open(mod.__file__).read()
    return hashlib.sha1(json.dumps([src, os.path.basename(str(cops._path)), VERSION]).encode()).hexdigest()[:8]


def _layout_hash(ix):
    return hashlib.sha1(json.dumps(ix, sort_keys=True, default=str).encode()).hexdigest()


def _direct_io(path, write, spans, threads):
    """spans [(address, bytes, file offset)] (each a multiple of 4 KB at an address of 4 KB) written
    to, or read from, path: O_DIRECT (no page cache) by threads, else buffered and the pages
    dropped after. Return whether it was O_DIRECT."""
    flags = (os.O_WRONLY | os.O_CREAT) if write else os.O_RDONLY
    try:
        fd, direct = os.open(path, flags | os.O_DIRECT, 0o644), True
    except OSError:
        fd, direct = os.open(path, flags, 0o644), False
    pieces = [(a + x, min(_PIECE, n - x), o + x) for a, n, o in spans for x in range(0, n, _PIECE)]
    errs, lock = [], threading.Lock()

    def work():
        while True:
            with lock:
                if not pieces or errs:
                    return
                a, n, o = pieces.pop()
            mv = memoryview((ctypes.c_uint8 * n).from_address(a)).cast("B")
            try:
                got = 0
                while got < n:
                    r = os.pwritev(fd, [mv[got:]], o + got) if write else os.preadv(fd, [mv[got:]], o + got)
                    if r <= 0:
                        raise OSError("%s: short %s at %d" % (path, "write" if write else "read", o + got))
                    got += r
            except OSError as e:
                errs.append(e)
    try:
        ts = [threading.Thread(target=work) for _ in range(max(1, threads))]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if errs:
            raise errs[0]
        if write:
            os.fsync(fd)
        if not direct:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return direct


class Entry:
    """A checkpoint the keeper holds (docs/GPU_SWAP_PLAN.md 3.7: an entry of the catalog): its index,
    its segments, its plugin. Made (the layout) then load()ed."""

    def __init__(self, o: Options):
        self.o = o
        self.root = os.geteuid() == 0
        self.kept = bool(o.keep)
        self.last = time.time()     # its last lease's start or end (the order of the drops)
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
                    ix["nodes"], ["%.1f" % (b / 1e9) for b in ix["part_bytes"]], time.time() - t0))
            self.id = entry_id(o, ix["model"], self.sh.ident)
            self.layout_hash = _layout_hash(ix)
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
        self.id = entry_id(o, "deepseek_v41", self.sh.ident)
        self.layout_hash = _layout_hash(self.index)

    def need(self):
        """{node: bytes} it holds (loaded or not)."""
        o, ix = self.o, self.index
        need = {}
        for part, nd in enumerate(ix["nodes"]):
            need[nd] = need.get(nd, 0) + ix["part_bytes"][part]
        if ix.get("dense"):
            need[o.dense_node] = need.get(o.dense_node, 0) + ix["dense"]["bytes"]
        return need

    def saved_path(self, save):
        return os.path.join(save, "%s-%s" % (self.id, code_key(self.plugin)))

    def _spans(self):
        """[(address, bytes)] of what the segments hold (each segment's bytes in use, to 4 KB)."""
        ix = self.index
        out = [(s.base, _up(ix["part_bytes"][k], K4)) for k, s in enumerate(self.segs)]
        if self.dense:
            out.append((self.dense.base, _up(ix["dense"]["bytes"], K4)))
        return out

    def _read_saved(self, save):
        """The segments from the saved copy of this entry (the same key and layout); False when there
        is none."""
        d = self.saved_path(save)
        try:
            meta = json.load(open(os.path.join(d, "index.json")))
        except (OSError, ValueError):
            others = [x for x in (os.listdir(save) if os.path.isdir(save) else [])
                      if x.startswith(self.id + "-") and not x.startswith(".")]
            if others:
                log("%s: saved copies of other code, not read: %s" % (save, ", ".join(sorted(others))))
            return False
        spans = self._spans()
        if meta.get("layout") != self.layout_hash or meta.get("bytes") != [n for _a, n in spans]:
            log("%s: another layout, not read" % d)
            return False
        t0 = time.time()
        direct = all([_direct_io(os.path.join(d, "seg%d.bin" % k), False, [(a, n, 0)], min(8, self.o.threads))
                      for k, (a, n) in enumerate(spans)])
        dt = time.time() - t0
        total = sum(n for _a, n in spans)
        self.index.update(meta["index"])
        self.source = "saved"
        log("read the saved copy %s: %.1f GB in %.1f s (%.2f GB/s; O_DIRECT %s)" % (
            d, total / 1e9, dt, total / 1e9 / max(dt, 1e-9), "yes" if direct else "no"))
        return True

    def _save(self, save):
        """The segments and the index written to save (once: in a directory of its own, renamed
        when complete); skipped, said, when the disk has no room."""
        d = self.saved_path(save)
        if os.path.exists(d):
            return
        spans = self._spans()
        total = sum(n for _a, n in spans)
        os.makedirs(save, exist_ok=True)
        st = os.statvfs(save)
        if st.f_bavail * st.f_frsize < total + SAVE_MARGIN:
            log("not saved to %s: %.1f GB to write, %.1f GB free" % (save, total / 1e9,
                                                                      st.f_bavail * st.f_frsize / 1e9))
            return
        t0 = time.time()
        tmp = os.path.join(save, ".%s.%d" % (os.path.basename(d), os.getpid()))
        os.makedirs(tmp)
        for k, (a, n) in enumerate(spans):
            _direct_io(os.path.join(tmp, "seg%d.bin" % k), True, [(a, n, 0)], min(8, self.o.threads))
        ix = {k: v for k, v in self.index.items() if k not in RUNTIME_KEYS}
        if ix.get("dense"):
            ix["dense"] = {k: v for k, v in ix["dense"].items() if k != "chunks"}
        with open(os.path.join(tmp, "index.json"), "w") as f:
            json.dump(dict(layout=self.layout_hash, bytes=[n for _a, n in spans], index=ix,
                           checkpoint=self.index["checkpoint"]), f)
        if self.root and (self.o.user is not None or os.environ.get("SUDO_UID")):
            uid = self.o.user if self.o.user is not None else int(os.environ["SUDO_UID"])
            gid = self.o.group if self.o.group is not None else int(os.environ.get("SUDO_GID", uid))
            for x in [tmp] + [os.path.join(tmp, f) for f in os.listdir(tmp)]:
                os.chown(x, uid, gid)
        os.rename(tmp, d)
        dt = time.time() - t0
        log("saved to %s: %.1f GB in %.1f s (%.2f GB/s)" % (d, total / 1e9, dt, total / 1e9 / max(dt, 1e-9)))

    def load(self, save=None):
        """The segments made and filled: from the saved copy in save when there is one, else from the
        checkpoint (then written to save)."""
        o, ix = self.o, self.index
        check_memory(self.need(), self.numa_on)
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
        self.source = "checkpoint"
        if not (save and self._read_saved(save)):
            if self.plugin is not None:
                self.plugin.fill(self.sh, ix, self.segs, o.threads, log)
            else:
                fill(self.sh, ix, self.segs, self.dense, o.threads)
            if save:
                try:
                    self._save(save)
                except OSError as e:
                    log("not saved to %s: %s" % (save, e))
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

    def status(self):
        ix = self.index
        return dict(pid=os.getpid(), model=ix.get("model", "deepseek_v41"), layout=ix["layout"],
                    types=ix.get("types"), locked=ix["locked"], layers=len(ix.get("layers", [])) or ix.get("L"),
                    nodes=ix["nodes"],
                    segments=[dict(node=s.node, gb=round(s.size / 1e9, 2),
                                   chunks=[[_pname(pg), round(n / 1e9, 2)] for pg, n, _o in s.chunks])
                              for s in self.segs + ([self.dense] if self.dense else [])],
                    checkpoint=ix["checkpoint"], id=self.id, source=getattr(self, "source", None))

    def close(self):
        """Its memfds closed and its maps gone (a round that maps them keeps their pages until it
        unmaps them: the kernel counts the maps)."""
        for seg in self.segs + ([self.dense] if self.dense else []):
            for fd in seg.fds:
                os.close(fd)
            _libc.munmap(seg.base, seg.size)
        self.segs, self.dense = [], None


class PutEntry(Entry):
    """An entry a round made and put (runtime/kept.py: a model's host arrays, made by its own loader):
    one memfd and its index, its ID the round's (the label and a hash of its key). Saved with --save
    as any entry (DIR/<ID>/) and read back by "restore ID"."""

    def __init__(self, index, seg, source="put"):
        self.o = Options(ckpt=None)
        self.root = os.geteuid() == 0
        self.numa_on = numa.enabled()
        self.kept, self.last, self.plugin = False, time.time(), None
        self.index, self.id, self.source = index, index["id"], source
        self.layout_hash = _layout_hash({k: v for k, v in index.items() if k not in RUNTIME_KEYS})
        self.segs, self.dense = [seg], None
        self.saver = None
        index.update(segments=[seg.describe()], locked=False, pid=os.getpid(), writable=True)

    def saved_path(self, save):
        return os.path.join(save, self.id)

    @classmethod
    def restore(cls, save, i, threads=8):
        """The entry i from its saved copy in save (O_DIRECT into new pages on its node), or None."""
        d = os.path.join(save, i)
        try:
            meta = json.load(open(os.path.join(d, "index.json")))
        except (OSError, ValueError):
            return None
        ix = meta["index"]
        if ix.get("layout") != LAYOUT_STORE or ix.get("id") != i or meta.get("layout") != _layout_hash(ix):
            log("%s: not a saved copy of %s, not read" % (d, i))
            return None
        node, n = ix["nodes"][0], meta["bytes"][0]
        numa_on = numa.enabled()
        check_memory({node: n}, numa_on)
        t0 = time.time()
        seg = Segment("%s-n%d" % (ix["model"], node), node, pages(n, node, os.geteuid() == 0, K4), numa_on)
        seg.populate(threads)
        direct = _direct_io(os.path.join(d, "seg0.bin"), False, [(seg.base, n, 0)], min(8, threads))
        seg.seal(write=False)
        dt = time.time() - t0
        log("read the saved copy %s: %.1f GB in %.1f s (%.2f GB/s; O_DIRECT %s)" % (
            d, n / 1e9, dt, n / 1e9 / max(dt, 1e-9), "yes" if direct else "no"))
        return cls(ix, seg, "saved")

    def close(self):
        if self.saver is not None:
            self.saver.join()
        super().close()


class Keeper:
    """The catalog (docs/GPU_SWAP_PLAN.md 3.7, steps 1 and 2): the entries by ID on one socket. The
    requests: "get [KEY]" (the index and the memfds; KEY an ID or a checkpoint's path, none: the first
    entry; the connection stays open as the round's lease), "status [KEY]" (an entry's status; with no
    KEY the first's, and every entry's under "entries"), "list", "load OPTIONS" (the JSON of Options:
    made at once, read in the background; one load at a time), "drop KEY", "keep KEY", "unkeep KEY",
    "put INDEX" (a round's memfd and its index: runtime/kept.py; the connection its lease), "restore ID"
    (an entry from the saved copy, in the background), "stop". budget (bytes, None: none): a load over
    it, or one the memory cannot take, drops the entries with no lease that are not kept, the least
    recently leased first."""

    def __init__(self, budget=None, keep_keys=(), save=None):
        self.entries = {}           # id -> Entry (ready), in the order they came
        self.loading = {}           # id -> {"checkpoint", "since"}
        self.failed = {}            # id -> the error of its load
        self.leases = {}            # the socket of a round's get -> (the entry's id, the round's pid)
        self.budget = budget
        self.keep_keys = list(keep_keys or ())
        self.save = save
        self.mu = threading.RLock()     # (a get hands out the fds under it: no drop closes them meanwhile)
        self.load_mu = threading.Lock()     # (one load at a time: each checks the memory before it takes a page)

    def add(self, e):
        with self.mu:
            e.last = time.time()
            self.entries[e.id] = e

    def pick(self, key=None):
        """The entry of key: an ID, a checkpoint's path, or None (the first); None when none."""
        with self.mu:
            if not key:
                return next(iter(self.entries.values()), None)
            if key in self.entries:
                return self.entries[key]
            want = os.path.realpath(key)
            return next((e for e in self.entries.values()
                         if os.path.realpath(e.index["checkpoint"]) == want), None)

    def lessees(self, e):
        """The pids of the rounds that lease e."""
        with self.mu:
            return sorted(pid for i, pid in self.leases.values() if i == e.id)

    def is_kept(self, e):
        return e.kept or any(k == e.id or os.path.realpath(k) == os.path.realpath(e.index["checkpoint"])
                             for k in self.keep_keys)

    def entry_status(self, e):
        st = e.status()
        st["leases"] = self.lessees(e)
        st["kept"] = self.is_kept(e)
        st["idle_s"] = None if st["leases"] else round(time.time() - e.last, 1)
        return st

    def listing(self):
        with self.mu:
            out = [self.entry_status(e) for e in self.entries.values()]
            out += [dict(id=i, status="loading", **d) for i, d in self.loading.items()]
            out += [dict(id=i, status="failed", error=m) for i, m in self.failed.items()]
        return out

    def held(self):
        with self.mu:
            return sum(n for e in self.entries.values() for n in e.need().values())

    def status(self, key=None):
        e = self.pick(key)
        st = self.entry_status(e) if e is not None else dict(pid=os.getpid(), checkpoint=None, segments=[])
        if key is None:
            st["entries"] = self.listing()
            st["host_budget_gb"] = None if self.budget is None else round(self.budget / 1e9, 2)
            st["held_gb"] = round(self.held() / 1e9, 2)
            st["keep"] = self.keep_keys
            st["save"] = self.save
        return st

    def make_room(self, e, memory=True):
        """Before e's load, with a budget: the entries no round leases and none keeps dropped, the least
        recently leased first, while e would take the held bytes over the budget or does not fit the
        memory (on a node short of it, the entries with pages there). SystemExit when the budget is
        still short; a memory still short is left to the load's check (it says what is missing)."""
        if self.budget is None:
            return
        need = e.need()
        with self.mu:
            while True:
                held = self.held()
                over = held + sum(need.values()) > self.budget
                short = memory_short(need, e.numa_on) if memory else []
                if not over and not short:
                    return
                free = [x for x in self.entries.values() if not self.lessees(x) and not self.is_kept(x)
                        and not (getattr(x, "saver", None) and x.saver.is_alive())]
                nodes = {nd for nd, _m in short}
                if not over and None not in nodes:
                    free = [x for x in free if nodes & set(x.need())]
                if not free:
                    if over:
                        raise SystemExit(
                            "weightkeeper: the host budget %.1f GB: %.1f GB held, %.1f GB to load, and every entry "
                            "leased or kept (%s)" % (
                                self.budget / 1e9, held / 1e9, sum(need.values()) / 1e9,
                                ", ".join("%s %.1f GB %s" % (x.id, sum(x.need().values()) / 1e9,
                                                             "kept" if self.is_kept(x) else
                                                             "leased by %s" % self.lessees(x))
                                          for x in self.entries.values())))
                    return
                v = min(free, key=lambda x: x.last)
                self.entries.pop(v.id)
                v.close()
                log("dropped %s (%.1f GB, no lease for %.0f s) for %s: %s" % (
                    v.id, sum(v.need().values()) / 1e9, time.time() - v.last, e.id,
                    "over the budget" if over else "; ".join(m for _nd, m in short)))

    def load(self, o):
        """An entry of o: made now (its layout and ID), loaded in a thread. Return its ID and its
        status ("ready" when the catalog had it)."""
        e = Entry(o)
        with self.mu:
            if e.id in self.entries:
                if o.keep:
                    self.entries[e.id].kept = True
                return e.id, "ready"
            if e.id in self.loading:
                return e.id, "loading"
            self.failed.pop(e.id, None)
            self.loading[e.id] = dict(checkpoint=e.index["checkpoint"], since=time.time())

        def work():
            try:
                with self.load_mu:
                    self.make_room(e)
                    e.load(self.save)
                self.add(e)
            except BaseException as exc:    # noqa: BLE001 (SystemExit of check_memory too)
                log("the load of %s: %r" % (e.id, exc))
                with self.mu:
                    self.failed[e.id] = str(exc)
            finally:
                with self.mu:
                    self.loading.pop(e.id, None)
        threading.Thread(target=work, daemon=True).start()
        return e.id, "loading"

    def drop(self, key):
        """The entry of key closed (leased or not: a round that maps it keeps its pages). Return
        its ID and the pids that lease it."""
        with self.mu:
            e = self.pick(key)
            if e is None:
                return None, []
            pids = self.lessees(e)
            self.entries.pop(e.id, None)
            e.close()
        log("dropped %s%s" % (e.id, " (mapped still by %s)" % pids if pids else ""))
        return e.id, pids

    def keep(self, key, on=True):
        """key kept (on) or not: an entry's flag, and the keeper's keys (a checkpoint not loaded
        yet kept when it comes)."""
        with self.mu:
            e = self.pick(key)
            if e is not None:
                e.kept = on
            if on and key not in self.keep_keys and e is None:
                self.keep_keys.append(key)
            if not on:
                self.keep_keys = [k for k in self.keep_keys if k != key and not (
                    e is not None and (k == e.id or os.path.realpath(k) == os.path.realpath(e.index["checkpoint"])))]
            return e.id if e is not None else key

    def put(self, c, index, fd, pid):
        """A round's memfd (sealed for size, the size of its index) and index as an entry, c its lease:
        its ID, or an error (the round keeps its own)."""
        sz = os.fstat(fd).st_size
        if fcntl.fcntl(fd, F_GET_SEALS) & SEALS_SIZE != SEALS_SIZE or sz != index["part_bytes"][0]:
            os.close(fd)
            return dict(error="a memfd not sealed for size, or not of the index's size")
        index["id"] = str(index["id"])
        if os.sep in index["id"] or index["id"].startswith("."):
            os.close(fd)
            return dict(error="an ID of a path")
        e = PutEntry(index, Segment.of_fd("%s-n%d" % (index["model"], index["nodes"][0]), index["nodes"][0], fd,
                                          sz, numa.enabled()))
        with self.mu:
            if e.id in self.entries:
                e.close()
                return dict(error="the keeper has %s" % e.id)
            try:
                self.make_room(e, memory=False)
            except SystemExit as exc:
                e.close()
                return dict(error=str(exc))
            self.entries[e.id] = e
            self.leases[c] = (e.id, pid)
        log("put %s: %.1f GB by pid %d" % (e.id, sz / 1e9, pid))
        if self.save:
            def save():
                try:
                    e._save(self.save)
                except OSError as exc:
                    log("not saved to %s: %s" % (self.save, exc))
            e.saver = threading.Thread(target=save, daemon=True)
            e.saver.start()
        return dict(id=e.id)

    def restore(self, i):
        """"ready" (the entry is held), "loading" (read from the saved copy in the background), or
        "none"."""
        with self.mu:
            if i in self.entries:
                return "ready"
            if i in self.loading:
                return "loading"
            if not self.save or os.sep in i or i.startswith(".") or \
                    not os.path.isfile(os.path.join(self.save, i, "index.json")):
                return "none"
            self.failed.pop(i, None)
            self.loading[i] = dict(checkpoint=None, since=time.time())

        def work():
            try:
                with self.load_mu:
                    e = PutEntry.restore(self.save, i)
                if e is None:
                    return
                with self.mu:
                    self.make_room(e, memory=False)
                    self.entries[e.id] = e
            except BaseException as exc:    # noqa: BLE001
                log("the restore of %s: %r" % (i, exc))
                with self.mu:
                    self.failed[i] = str(exc)
            finally:
                with self.mu:
                    self.loading.pop(i, None)
        threading.Thread(target=work, daemon=True).start()
        return "loading"

    def _get(self, c, arg, pid):
        """A round's get: the index and the fds, and c its lease (True), or an error (False)."""
        with self.mu:
            # (a checkpoint the catalog knows by another path: its only entry, which the round
            # checks by the files' identity, as before the catalog)
            e = self.pick(arg) or (self.pick() if len(self.entries) == 1 and arg and os.sep in arg else None)
            if e is None:
                _send_msg(c, json.dumps(dict(error="no entry %s" % (arg or ""))).encode())
                return False
            _send_msg(c, json.dumps(e.index).encode())
            socket.send_fds(c, [b"F"], e.fds())
            self.leases[c] = (e.id, pid)
            e.last = time.time()
        log("a lease of %s: pid %d" % (e.id, pid))
        return True

    def _end_lease(self, c):
        with self.mu:
            i, pid = self.leases.pop(c, (None, None))
            e = self.entries.get(i)
            if e is not None:
                e.last = time.time()
        if i is not None:
            log("the lease of %s ended: pid %d" % (i, pid))

    def _request(self, c):
        """One request on c. Return "lease" (c stays open), "stop", or None (c closed)."""
        c.settimeout(60)
        uid, pid = struct.unpack("3i", c.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1::-1]
        if uid != os.getuid():
            return None
        req, _sp, arg = _recv_msg(c).decode().strip().partition(" ")
        arg = arg.strip() or None
        if req == "get":
            return "lease" if self._get(c, arg, pid) else None
        if req == "put":
            _msg, fds, _f, _a = socket.recv_fds(c, 16, 1)
            if len(fds) != 1:
                [os.close(fd) for fd in fds]
                _send_msg(c, json.dumps(dict(error="put: one memfd")).encode())
                return None
            r = self.put(c, json.loads(arg), fds[0], pid)
            _send_msg(c, json.dumps(r).encode())
            return "lease" if "id" in r else None
        if req == "restore":
            st = self.restore(arg or "")
            if st == "none" and arg in self.failed:
                st = "failed"
            _send_msg(c, json.dumps(dict(status=st)).encode())
            return None
        if req == "status":
            _send_msg(c, json.dumps(self.status(arg)).encode())
        elif req == "list":
            _send_msg(c, json.dumps(self.listing()).encode())
        elif req == "load":
            d = json.loads(arg or "{}")
            try:
                i, st = self.load(Options(**d))
                _send_msg(c, json.dumps(dict(id=i, status=st)).encode())
            except (SystemExit, Exception) as exc:  # noqa: BLE001 (the layout's)
                _send_msg(c, json.dumps(dict(error=str(exc))).encode())
        elif req == "drop":
            i, pids = self.drop(arg)
            _send_msg(c, json.dumps(dict(dropped=i, **(dict(mapped_by=pids) if pids else {}))).encode())
        elif req in ("keep", "unkeep"):
            if not arg:
                _send_msg(c, json.dumps(dict(error="%s: an ID or a checkpoint" % req)).encode())
            else:
                _send_msg(c, json.dumps({"kept" if req == "keep" else "unkept": self.keep(arg, req == "keep")})
                          .encode())
        elif req == "stop":
            try:
                _send_msg(c, b"stopping")
            except OSError:
                pass
            return "stop"
        return None

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
        log("serving on %s (%d entries%s)" % (path, len(self.entries), "; the host budget %.1f GB" % (
            self.budget / 1e9) if self.budget is not None else ""))
        sel = selectors.DefaultSelector()
        sel.register(srv, selectors.EVENT_READ)
        try:
            while True:
                for key, _ev in sel.select():
                    if key.fileobj is srv:
                        c, _ = srv.accept()
                        try:
                            r = self._request(c)
                        except (OSError, ValueError) as e:
                            log("a client: %s" % e)
                            r = None
                        if r == "lease":
                            c.settimeout(None)
                            sel.register(c, selectors.EVENT_READ)
                            continue
                        c.close()
                        if r == "stop":
                            return
                        continue
                    # a lease's socket: the round closed it (or exited), or sent what a lease ignores
                    c = key.fileobj
                    try:
                        b = c.recv(4096)
                    except OSError:
                        b = b""
                    if not b:
                        sel.unregister(c)
                        self._end_lease(c)
                        c.close()
        finally:
            for c in list(self.leases):
                c.close()
            sel.close()
            srv.close()
            try:
                os.unlink(path)
            except OSError:
                pass


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


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _settings_path(uid):
    return os.path.join(_dir(), "reclank-weightkeeper-%d.settings.json" % uid)


def _shared_settings(uid, fn):
    """The state of the root keepers' settings (the values before the first: "orig", the pids
    holding them: "pids"): fn(state or None) -> the new state (None: the file goes), under flock."""
    p = _settings_path(uid)
    fd = os.open(p + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            st = json.load(open(p))
        except (OSError, ValueError):
            st = None
        new = fn(st)
        if new is None:
            try:
                os.unlink(p)
            except OSError:
                pass
        else:
            with open(p + ".tmp", "w") as f:
                json.dump(new, f)
            os.replace(p + ".tmp", p)
        return new
    finally:
        os.close(fd)


def settings_join(uid, now, me):
    """A root keeper (pid me) joins the holders of the settings: the first (no live holder) records
    now's values as the ones to give back. Returns the state."""
    def join(st):
        if st is None or not any(_alive(p) for p in st["pids"]):
            return dict(orig=dict(shmem=now["shmem"], compact=now["compact"]), pids=[me])
        st["pids"] = [p for p in st["pids"] if _alive(p)] + [me]
        return st
    return _shared_settings(uid, join)


def settings_leave(uid, me):
    """A root keeper (pid me) leaves: the last live holder restores the values the first recorded.
    Returns True when it restored them."""
    done = [False]

    def leave(st):
        if st is None:
            return None
        st["pids"] = [p for p in st["pids"] if p != me and _alive(p)]
        if st["pids"]:
            log("the settings stay: root keepers %s hold them" % st["pids"])
            return st
        restore_settings(st["orig"])
        done[0] = True
        return None
    _shared_settings(uid, leave)
    return done[0]


def restore_settings(orig):
    """shmem_enabled and compact_unevictable_allowed back to orig, each write checked."""
    for path, want, now in ((SHMEM_THP, orig["shmem"], lambda: _sysfs_choice(SHMEM_THP)),
                            (COMPACT_UNEVICT, orig["compact"], lambda: _read_int(COMPACT_UNEVICT, None))):
        if want is None:
            continue
        try:
            _write(path, want)
        except OSError as e:
            log("could not restore %s to %s: %s" % (path, want, e))
        if str(now()) != str(want):
            log("%s is %s, not %s" % (path, now(), want))


def give_back_pools(added):
    """The pages a keeper added to each pool (its ADDED), taken out of the pool again (the
    pages of another keeper stay)."""
    for d, n in added.items():
        if n <= 0:
            continue
        nr = _read_int(d + "/nr_hugepages")
        try:
            _write(d + "/nr_hugepages", max(0, nr - n))
        except OSError as e:
            log("could not give back the pool %s: %s" % (d, e))


def _first(o):
    """The catalog with the entry of o.ckpt loaded (none: an empty catalog, --serve)."""
    k = Keeper(None if o.host_budget is None else int(o.host_budget * 1e9), o.keep_keys,
               os.path.abspath(o.save) if o.save else None)
    if o.ckpt:
        e = Entry(o)
        k.make_room(e)
        e.load(k.save)
        k.add(e)
    return k


def run(o: Options):
    """The weightkeeper, as the module text says: as root a parent that stays root and gives back
    the pages it added to the pools and (the last root keeper) the settings when it exits."""
    root = os.geteuid() == 0
    if o.lock and not root:
        raise SystemExit("weightkeeper: --lock needs root (sudo): no partial lock")
    uid = os.getuid() if not root else o.user if o.user is not None else int(os.environ.get("SUDO_UID", "0"))
    if root and uid == 0:
        raise SystemExit("weightkeeper: as root it serves as the invoking user: run it with sudo, or give --user")
    path = o.sock or socket_path(o.name, uid)
    if _answers(path):
        raise SystemExit("weightkeeper: a weightkeeper answers on %s (--list; another --name for a second one)" % path)
    if not root:
        return _first(o).serve(path)
    gid = o.group if o.group is not None else int(os.environ.get("SUDO_GID", "0"))
    nodes = sorted(set(numa.node_of_cpu().values())) or [0]
    now = system_state(nodes)
    me = os.getpid()
    shared = settings_join(uid, now, me)
    log("the settings before the root keepers: shmem_enabled %s, compact_unevictable_allowed %s (given back by "
        "the last; root keepers now: %d)" % (shared["orig"]["shmem"], shared["orig"]["compact"], len(shared["pids"])))
    try:
        if now["shmem"] == "never":
            _write(SHMEM_THP, "advise")
        if o.lock:
            _write(COMPACT_UNEVICT, 0)
    except OSError as e:
        log("the settings: %s" % e)
    rp, wp = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rp)
        code, told = 0, [False]

        def tell():
            # the pages added to the pools, to the parent (once)
            if not told[0]:
                told[0] = True
                try:
                    os.write(wp, json.dumps(ADDED).encode())
                    os.close(wp)
                except OSError:
                    pass
        try:
            _pdeathsig()
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            z = _first(o)
            tell()
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
        tell()
        os._exit(code)
    os.close(wp)

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
    raw = b""
    while True:
        b = os.read(rp, 65536)
        if not b:
            break
        raw += b
    os.close(rp)
    try:
        added = json.loads(raw.decode()) if raw else {}
    except ValueError:
        added = {}
    give_back_pools(added)

    settings_leave(uid, me)
    log("the pages added to the pools given back (%s)" % (
        ", ".join("%s %d" % (os.path.basename(os.path.dirname(d + "/")), n) for d, n in added.items() if n) or "none"))
    return os.waitstatus_to_exitcode(status)


# ---------------------------------------------------------------- a round

def request(op, path=None):
    """op to the keeper at path (None: RECLANK_WEIGHTKEEPER's, else the default one)."""
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(path or socket_path(env_name()))
    _send_msg(c, op.encode())
    return c


def find(ckpt):
    """(the socket, the status) of the weightkeeper of this user that holds ckpt, or (None, None):
    the default socket, then the named ones (socket_path: a keeper for each model); with
    RECLANK_WEIGHTKEEPER=NAME, that keeper's only."""
    want = os.path.realpath(ckpt)
    for path in [socket_path(env_name())] if env_name() else sockets():
        try:
            c = request("status", path)
            with c:
                st = json.loads(_recv_msg(c))
        except (OSError, ValueError):
            continue
        for e in st.get("entries") or [st]:
            if e.get("checkpoint") and os.path.realpath(e["checkpoint"]) == want and e.get("status") is None:
                return path, e
    return None, None


_LEASES = []        # the sockets of this process's gets: open for its life (the maps are never unmapped)


class Weights:
    """The segments of a weightkeeper, mapped in this process: read-only, or
    writable when the weightkeeper sealed only their size (self.writable:
    CUDA can page-lock them; the arrays stay read-only). The get's socket
    stays open as the lease (self.lease: the keeper drops no leased entry for
    its budget), until the process exits or end_lease()."""

    def __init__(self, path=None, populate=True, ckpt=None, key=None):
        """ckpt: the checkpoint the round wants (with no path, the keeper that
        holds it: find); another one raises ValueError before anything is
        mapped."""
        if path is None and ckpt is not None:
            path = find(ckpt)[0]
        c = request("get" + (" " + key if key is not None else " " + os.path.realpath(ckpt) if ckpt is not None
                             else ""), path)
        try:
            self.index = json.loads(_recv_msg(c))
            if "error" in self.index:
                raise ValueError("the weightkeeper: %s" % self.index["error"])
            nfd = sum(len(s["chunks"]) for s in self.index["segments"])
            if self.index.get("dense"):
                nfd += len(self.index["dense"]["chunks"])
            _msg, fds, _flags, _addr = socket.recv_fds(c, 16, nfd)
        except BaseException:
            c.close()
            raise
        if self.index["version"] != VERSION or self.index["layout"] not in (LAYOUT, LAYOUT_V1, LAYOUT_TENSORS,
                                                                                LAYOUT_STORE):
            for fd in fds:
                os.close(fd)
            c.close()
            raise ValueError("the weightkeeper has the layout %r (version %s)" % (
                self.index["layout"], self.index["version"]))
        if ckpt is not None and identity(ckpt) != self.index["shards"]:
            for fd in fds:
                os.close(fd)
            c.close()
            raise ValueError("the weightkeeper holds another checkpoint (%s)" % self.index["checkpoint"])
        self.lease = c
        _LEASES.append(c)
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

    def end_lease(self):
        """The lease ended (the keeper may drop the entry for its budget: only when this process
        reads its maps no more, or the memory stays taken)."""
        if self.lease in _LEASES:
            _LEASES.remove(self.lease)
        self.lease.close()

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
