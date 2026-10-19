"""The cold experts of a GPU step in parts: the GPU+parts mode of QwenGPU
(RECLANK_GPU_PARTS=1).

The CPU part of a step (the cold experts, GP_KQ_MOE in a program of
GP_CPU_TASK) runs in one team spread over the two sockets; each expert is
read from the memory of one node, so half the threads read the other node
(or the experts take twice the memory: QwenGPU._numa_copy_experts). In
parts, the program of the CPU part starts with GP_RUN_PARTS: it runs in a
team of a part for each NUMA node, the threads of a part pinned to CPUs of
its node, and each part computes its rows of every cold expert
(GP_KQ_MOE_PART,
backends/cpu/csrc/kernels/kq/moe_part.c): the gate and the up rows of its
part of inner, then (after one barrier across the parts, the act) the down
rows of its part of hidden. The rows a part reads are in the memory of its
node: the experts themselves where they are (the node of their pages), and
for the other node a copy of only its rows (half the bytes of the experts,
not all of them). The output has the bits of GP_KQ_MOE.

The CPUs: the helper threads (the model thread, the copy workers of the
GPU) on the two highest CPUs of the taskset on the GPU's node
(helper_cpus), the teams of the parts on the other CPUs of each node, the
same count on each (part_cpus). taskset 0-19,24-43 (the 40 cores of the
server) with the GPU on node 0: the helpers on 18 and 19, the parts on
0-17 and 24-41; 42 and 43 stay free for other processes.

    parts = ExpertParts.make(mats, E, hidden, inner)   # None when it is off
    prog = parts.program(i, t, k, hf, ids, val, out, kcount, flags, bufs)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from reclank.ir import program as P
from reclank.runtime import numa


def allowed():
    """The CPUs the process may use: the taskset, as the places of OpenMP
    have it (the main thread is bound to one CPU when the runtime starts,
    OMP_PROC_BIND, and the model thread pins itself later)."""
    from reclank.backends.cpu import cops
    s = cops.place_cpus()
    return s or set(os.sched_getaffinity(0))


def wanted():
    return os.environ.get("RECLANK_GPU_PARTS", "0") == "1"


def nodes():
    """The NUMA nodes with CPUs, in order."""
    return sorted(set(numa.node_of_cpu().values()))


def helper_cpus(gnode):
    """The CPUs of the helper threads in the parts mode: the model thread and
    the copy workers on the two highest allowed CPUs of the GPU's node (taskset
    0-19,24-43 and the GPU on node 0: 18 and 19), or [] when the node has
    fewer than 3."""
    s = sorted(allowed() & numa.node_cpus(gnode))
    return s[-2:] if len(s) >= 3 else []


def _env_cpus():
    out = set()
    for v in ("RECLANK_MAIN_CPU", "RECLANK_GPU_COPY_CPU"):
        try:
            c = int(os.environ.get(v, "-1"))
        except ValueError:
            continue
        if c >= 0:
            out.add(c)
    return out


def part_cpus(team=None):
    """The CPUs of the team of each part: the allowed CPUs of its node that no
    helper thread takes (RECLANK_MAIN_CPU, RECLANK_GPU_COPY_CPU), the lowest
    first, the same count on each node (the fewest of a node, or
    RECLANK_GPU_PART_THREADS when it is less). taskset 0-19,24-43 with the
    helpers on 18 and 19: 0-17 and 24-41."""
    helpers = _env_cpus()
    ok = allowed()
    avail = [sorted((ok & numa.node_cpus(nd)) - helpers) for nd in nodes()]
    n = min(len(a) for a in avail)
    want = team or os.environ.get("RECLANK_GPU_PART_THREADS")
    if want:
        n = min(n, int(want))
    if n < 1:
        raise ValueError("a node has no CPU for its part (taskset, RECLANK_GPU_PART_THREADS)")
    return [a[:n] for a in avail]


def node_budget(node):
    """The bytes of rows of experts a part may copy to node: RECLANK_GPU_NODE<n>_GB,
    else 85% of its memory less 8 GB (the copy is bound to the node, MPOL_BIND:
    the kernel takes back its page cache, not the memory of the processes)."""
    v = os.environ.get("RECLANK_GPU_NODE%d_GB" % node)
    if v not in (None, "", "auto"):
        return float(v) * 1e9
    return 0.85 * numa.node_bytes(node) - 8e9


# The share of part 0 of two by default. On the 2-socket Xeon (6 memory
# channels on node 0, 4 on node 1; 18 threads a part) a row of the cold
# experts took 1.2 times as long on node 1: Qwen3.8, an MTP verify group of
# 4 rows on real text, the CPU task of a layer (node 0 / node 1 gate and up,
# down): even 495 us (304 / 364, 177 / 218), 0.55 459 us (330 / 331, 195 /
# 197), 0.6 492 us (the down rows 212 / 175).
AUTO_SHARE0 = 0.55


def shares(parts, total=0, home=None, nodes_=None, s0_auto=AUTO_SHARE0):
    """The share of the rows of each part. RECLANK_GPU_PART_SHARE: "auto" (the
    default) AUTO_SHARE0 to part 0 of two (equal shares of more) when the rows
    a part copies to its node fit there (node_budget; total: the bytes of the
    experts; home: the node that holds them, whose part reads them in place),
    else that part takes the rows that fit and part 0 the rest; a number: the
    share of part 0 (node 0), the rest equally to the others."""
    v = os.environ.get("RECLANK_GPU_PART_SHARE", "auto")
    if parts < 2:
        return [1.0]
    if v != "auto":
        s0 = min(max(float(v), 0.05), 0.95)
        return [s0] + [(1.0 - s0) / (parts - 1)] * (parts - 1)
    share = [1.0 / parts] * parts
    if parts == 2:
        share = [s0_auto, 1.0 - s0_auto]
    for p, nd in enumerate(nodes_ or []):
        if p == 0 or nd == home or total <= 0:
            continue
        fit = max(0.05, node_budget(nd) / total)
        if fit < share[p]:
            share[0] += share[p] - fit
            share[p] = fit
    return share


def split(n, parts, align=32, share=None):
    """The first row and the count of each part of n rows: the shares of
    share (or equal), multiples of align; the parts after the first round
    down (their rows must fit their nodes), part 0 takes the rest."""
    share = share or [1.0 / parts] * parts
    rest = [max(align, int(n * f / align + 1e-9) * align) for f in share[1:]]
    sizes = [n - sum(rest)] + rest
    if sizes[0] < align:
        raise ValueError("the parts of %d rows (shares %s) leave part 0 %d" % (n, share, sizes[0]))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int).tolist()
    return list(zip(starts, sizes))


def home_node(a):
    """The node of the pages of the array a (three pages of it), or None when
    they differ or are not known."""
    if a.nbytes < 3 * 4096:
        return None
    flat = a.reshape(-1).view(np.uint8)
    found = set()
    for off in (0, flat.size // 2, flat.size - 4096):
        try:
            st = numa.page_nodes(flat[off:off + 1])
        except OSError:
            return None
        if st.size == 0 or st[0] < 0:
            return None
        found.add(int(st[0]))
    return found.pop() if len(found) == 1 else None


def is_e3(t):
    """A KQ_E3 type (the MUL1 blocks of EXL3, kernels/kq/exl3.c)."""
    return 70 < int(t) <= 78


def e3_bytes(K, k, n):
    """The bytes of the block of a k x n KQ_E3 matrix: the trellis, suh, svh."""
    return k * n * K // 8 + 2 * (k + n)


def part_rows(m, E, r0, nr, nd):
    """A copy on node nd of the rows r0 to r0 + nr of each of the E experts
    of m: (array, the bytes of an expert there). The rows of a KQ_E3 matrix
    are its outputs, the columns of its trellis (k / 16 x n / 16 tiles of 32
    K bytes, row by row): the copy is the block of a k x nr matrix (the tile
    columns r0 / 16 on, in each row of tiles; all of suh; svh r0 to r0 + nr)."""
    if not is_e3(m.type):
        rb = m.data.nbytes // (E * m.rows)
        dst = numa.empty_on((E, nr * rb), np.uint8, nd, strict=True)
        dst[:] = m.data.reshape(E, m.rows * rb)[:, r0 * rb:(r0 + nr) * rb]
        return dst, nr * rb
    K, k, n = int(m.type) - 70, m.cols, m.rows
    per, sub, tb = e3_bytes(K, k, n), e3_bytes(K, k, nr), 32 * K
    tr, trs = k * n * K // 8, k * nr * K // 8
    src = m.data.reshape(E, per)
    dst = numa.empty_on((E, sub), np.uint8, nd, strict=True)
    st = np.lib.stride_tricks.as_strided
    tiles = st(src[:, (r0 // 16) * tb:], shape=(E, k // 16, nr // 16, tb), strides=(per, n // 16 * tb, tb, 1),
               writeable=False)
    st(dst, shape=(E, k // 16, nr // 16, tb), strides=(sub, nr // 16 * tb, tb, 1))[:] = tiles
    dst[:, trs:trs + 2 * k] = src[:, tr:tr + 2 * k]
    dst[:, trs + 2 * k:] = src[:, tr + 2 * k + 2 * r0:tr + 2 * k + 2 * (r0 + nr)]
    return dst, sub


class ExpertParts:
    """The rows of the experts of each part and the programs of GP_RUN_PARTS."""

    def __init__(self, mats, E, hidden, inner, cpus, quiet=False, on_node=None):
        """mats: layer -> (gate, up, down) matrices of the experts (KMat: data,
        type, rows, cols; data has the E experts). cpus: the CPUs of each
        part (part_cpus). on_node(i, j, node): all the experts of matrix j
        (0 gate, 1 up, 2 down) of layer i in the memory of node, or None (the
        weightkeeper's experts whole on each node: the parts read them there,
        no copy)."""
        self.E, self.hidden, self.inner = E, hidden, inner
        self.cpus = cpus
        self.nparts = len(cpus)
        self.team = len(cpus[0])
        self.nodes = nodes()[:self.nparts]
        total = sum(m.data.nbytes for ms in mats.values() for m in ms)
        first = next(iter(mats.values()))[0] if mats else None
        home = home_node(first.data) if first is not None else None
        # KQ_E3: the rows of a part in blocks of 128 (the transform of the
        # outputs), those of gate and up as one range of their 2 inner outputs
        # (moe_part.c; two parts: all of gate, all of up), even shares by
        # default (the decode of the trellis, not the reads, takes the time)
        self.e3 = first is not None and is_e3(first.type)
        self.share = shares(self.nparts, total, home, self.nodes, 0.5 if self.e3 else AUTO_SHARE0)
        align = 128 if self.e3 else 32
        self.gu = split(2 * inner if self.e3 else inner, self.nparts, align=align, share=self.share)
        self.dn = split(hidden, self.nparts, align=align, share=self.share)
        self.desc = {}          # layer -> [int64 12 for each part]
        self.keep = []
        t0 = time.time()
        copied = 0
        for i, (G, U, D) in mats.items():
            descs = []
            for p, nd in enumerate(self.nodes):
                d = []
                gu = [self.gu[p], self.gu[p]]
                if self.e3:
                    a0, n = self.gu[p]
                    gu = [(min(a0, inner), min(a0 + n, inner) - min(a0, inner)),
                          (max(a0, inner) - inner, max(a0 + n, inner) - max(a0, inner))]
                for j, (m, (r0, nr)) in enumerate(((G, gu[0]), (U, gu[1]), (D, self.dn[p]))):
                    if nr == 0:
                        # (none of the matrix in this part: an empty block)
                        d += [0, m.type, e3_bytes(int(m.type) - 70, m.cols, 0), 0]
                        continue
                    full = on_node(i, j, nd) if on_node is not None else None
                    if full is None and home_node(m.data) == nd:
                        full = m.data
                    if full is not None:
                        d += [full.ctypes.data, m.type, full.nbytes // E, r0]
                        continue
                    dst, per = part_rows(m, E, r0, nr, nd)
                    self.keep.append(dst)
                    copied += dst.nbytes
                    d += [dst.ctypes.data, m.type, per, 0]
                descs.append(np.array(d, dtype=np.int64))
            self.desc[i] = descs
        self.cpus_arr = np.array([c for cs in cpus for c in cs], dtype=np.int32)
        self.pm = {i: np.concatenate(d) for i, d in self.desc.items()}
        self.gu_arr = np.array(self.gu, dtype=np.int32).reshape(-1)
        self.dn_arr = np.array(self.dn, dtype=np.int32).reshape(-1)
        if not quiet and os.environ.get("RECLANK_QUIET") != "1":
            print("the experts in %d parts (nodes %s, CPUs %s): the rows of %s %s, of hidden %s; "
                  "%.1f GB of rows copied in %.1f s" % (
                      self.nparts, self.nodes, ["%d-%d" % (c[0], c[-1]) for c in cpus],
                      "gate and up" if self.e3 else "inner", [n for _a, n in self.gu], [n for _a, n in self.dn], copied / 1e9,
                      time.time() - t0), file=sys.stderr)

    @classmethod
    def from_desc(cls, desc, gu, dn, E, hidden, inner, cpus, part_nodes=None):
        """ExpertParts on rows already in place on their nodes (the segments of
        the weightkeeper, reclank/runtime/weightkeeper.py): desc is layer -> [12 int64 for
        each part] (address, type, the bytes of an expert, the first row; gate,
        up, down), gu and dn the rows of each part; part_nodes: the node of
        each part (its scratch; default the first nodes)."""
        self = cls.__new__(cls)
        self.E, self.hidden, self.inner = E, hidden, inner
        self.cpus, self.nparts, self.team = cpus, len(cpus), len(cpus[0])
        self.nodes = list(part_nodes) if part_nodes is not None else nodes()[:self.nparts]
        self.share = None
        self.e3 = False
        self.gu, self.dn = [tuple(x) for x in gu], [tuple(x) for x in dn]
        self.desc = {i: [np.asarray(d, np.int64) for d in ds] for i, ds in desc.items()}
        self.keep = []
        self.cpus_arr = np.array([c for cs in cpus for c in cs], dtype=np.int32)
        self.pm = {i: np.concatenate(d) for i, d in self.desc.items()}
        self.gu_arr = np.array(self.gu, dtype=np.int32).reshape(-1)
        self.dn_arr = np.array(self.dn, dtype=np.int32).reshape(-1)
        return self

    @classmethod
    def make(cls, mats, E, hidden, inner, on_node=None):
        """ExpertParts when RECLANK_GPU_PARTS=1 and the machine has two or
        more NUMA nodes, else None."""
        if not wanted() or not numa.enabled() or len(nodes()) < 2:
            return None
        return cls(mats, E, hidden, inner, part_cpus(), on_node=on_node)

    def program(self, i, t, k, hf, ids, val, out, kcount, flags, bufs):
        """The CPU program of the cold experts of layer i: GP_RUN_PARTS (the
        flat team of the parts, pinned) and GP_KQ_MOE_PART. bufs: a dict for
        the scratch that the layers share (they run one at a time)."""
        hidden, inner, E = self.hidden, self.inner, self.E
        act2 = bufs.get("act2")
        w = 2 * inner if self.e3 else inner     # (KQ_E3: the outputs of gate and up)
        if act2 is None or act2.shape[0] < t * k or act2.shape[1] != w:
            act2 = bufs["act2"] = np.zeros((t * k, w), np.float32)
        scr = []
        for p in range(self.nparts):
            (_a0, ni), (_c0, nd) = self.gu[p], self.dn[p]
            key = "scratch%d" % p
            need = _scratch_bytes(t, k, E, hidden, inner, ni, nd)
            a = bufs.get(key)
            if a is None or a.nbytes < need:
                a = bufs[key] = numa.empty_on(need, np.uint8, self.nodes[p], strict=True)
            scr.append(a)
        q = P.Program()
        q.emit(P.RUN_PARTS, self.nparts, self.team, self.cpus_arr)
        q.emit(P.KQ_MOE_PART, hf, ids, val, t, k, E, self.pm[i], hidden, inner, self.gu_arr, self.dn_arr,
               act2, out, self.nparts, self.team, kcount, flags,
               np.array([a.ctypes.data for a in scr], dtype=np.int64))
        q.keep += scr
        return q.finish()


def _scratch_bytes(t, k, E, hidden, inner, ni, nd):
    from reclank.backends.cpu import cops
    return cops.kq_moe_part_scratch(t, k, E, hidden, inner, ni, nd)
