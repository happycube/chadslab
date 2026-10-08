"""The memory of a NUMA node, for the parts of a step (np_gemma/parts.py).

A part of a step runs on the cores of one node (gemma_run_parts binds its
teams). The rows of the weights that a part reads should be in the memory of
that node. This module gives:

    node_of_cpu()            the node of each CPU (/sys/devices/system/node)
    part_nodes(n, team)      the node of each part of gemma_run_parts
    empty_on(shape, dtype, node)   an array in the memory of a node
    copy_on(a, node)         a copy of a in the memory of a node
    pin_thread(node)         keep the calling thread on the CPUs of a node
    page_nodes(a)            the node of each page of an array (a check)

An array of empty_on is an anonymous mmap. A call to mbind with
MPOL_PREFERRED binds its range before the first write, so the kernel puts
each page on the node when the copy touches it, also when the thread of the
copy runs on another node. With a full node the kernel takes another node in
place of a failure. The syscalls go through libc, so libnuma is not needed.

On a machine with one node, or without the syscalls, the functions give
plain arrays and node 0. NP_GEMMA_NUMA=0 turns the placement off.
"""
from __future__ import annotations

import ctypes
import glob
import mmap
import os
import platform
import re

import numpy as np

MPOL_PREFERRED = 1
MPOL_INTERLEAVE = 3
MPOL_MF_MOVE = 1 << 1

# The numbers of the syscalls: (mbind, move_pages)
_SYSCALLS = {"x86_64": (237, 279), "aarch64": (235, 239)}

_libc = None


def _syscalls():
    global _libc
    nums = _SYSCALLS.get(platform.machine())
    if nums is None:
        return None
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.syscall.restype = ctypes.c_long
    return nums


def _cpulist(text):
    out = []
    for part in text.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def node_of_cpu():
    """Return a dict: CPU -> node. Empty without /sys/devices/system/node."""
    out = {}
    for d in glob.glob("/sys/devices/system/node/node[0-9]*"):
        node = int(re.search(r"node(\d+)$", d).group(1))
        with open(os.path.join(d, "cpulist")) as f:
            for cpu in _cpulist(f.read()):
                out[cpu] = node
    return out


def enabled():
    """True when the machine has two or more nodes, the syscalls are known,
    and NP_GEMMA_NUMA is not 0."""
    if os.environ.get("NP_GEMMA_NUMA", "1") == "0" or _syscalls() is None:
        return False
    return len(set(node_of_cpu().values())) > 1


def part_nodes(n, team=0):
    """Return the node of each of the n parts of gemma_run_parts: the node of
    most of the CPUs of its team. All 0 when the placement is off."""
    if not enabled():
        return [0] * n
    from . import cops
    cpus = cops.gp_part_cpus(n, team)
    of = node_of_cpu()
    out = []
    for row in cpus:
        nodes = [of[c] for c in row.tolist() if c in of]
        out.append(max(set(nodes), key=nodes.count) if nodes else 0)
    return out


def _mbind(addr, size, node, flags=0):
    nums = _syscalls()
    mask = (ctypes.c_ulong * 16)()
    mask[node // 64] = 1 << (node % 64)
    r = _libc.syscall(ctypes.c_long(nums[0]), ctypes.c_void_p(addr), ctypes.c_ulong(size),
                      ctypes.c_int(MPOL_PREFERRED), mask, ctypes.c_ulong(16 * 64),
                      ctypes.c_uint(flags))
    if r != 0:
        e = ctypes.get_errno()
        raise OSError(e, "mbind: %s" % os.strerror(e))


def _huge(m):
    """Huge pages for a large anonymous map (transparent_hugepage "madvise"):
    fewer faults when it fills (the copies of the experts) and fewer TLB
    misses when the CPU part of a step reads it."""
    try:
        m.madvise(mmap.MADV_HUGEPAGE)
    except (AttributeError, OSError):
        pass


def empty_interleaved(nbytes):
    """An uninitialized uint8 array of nbytes whose pages go to the nodes in
    turn (MPOL_INTERLEAVE): all the nodes read it at about the same rate. A
    plain array on one node or without the syscalls."""
    if nbytes == 0 or not enabled():
        return np.empty(nbytes, dtype=np.uint8)
    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    _huge(m)
    a = np.frombuffer(m, dtype=np.uint8, count=nbytes)
    nums = _syscalls()
    nodes = sorted(set(node_of_cpu().values()))
    mask = (ctypes.c_ulong * 16)()
    for n in nodes:
        mask[n // 64] |= 1 << (n % 64)
    r = _libc.syscall(ctypes.c_long(nums[0]), ctypes.c_void_p(a.ctypes.data), ctypes.c_ulong(nbytes),
                      ctypes.c_int(MPOL_INTERLEAVE), mask, ctypes.c_ulong(16 * 64), ctypes.c_uint(0))
    if r != 0:
        e = ctypes.get_errno()
        raise OSError(e, "mbind: %s" % os.strerror(e))
    return a


def empty_on(shape, dtype, node, huge=True):
    """Return an uninitialized array in the memory of node. Its pages come
    when they are first written (huge False: 4 KB pages, for an array of
    which only some parts are written)."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if nbytes == 0 or not enabled():
        return np.empty(shape, dtype=dtype)
    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if huge:
        _huge(m)
    a = np.frombuffer(m, dtype=dtype, count=nbytes // dtype.itemsize).reshape(shape)
    _mbind(a.ctypes.data, nbytes, node)
    return a


def copy_on(a, node):
    """Return a C-contiguous copy of a in the memory of node."""
    out = empty_on(a.shape, a.dtype, node)
    out[...] = a
    return out


MPOL_LOCAL = 4
_SET_MEMPOLICY = 238        # x86_64


def no_numa_balancing():
    """Keep the automatic NUMA balancing of the kernel off this process: the
    task policy MPOL_LOCAL (allocations on the local node, as the default
    policy) has no MPOL_F_MOF, so task_numa_work skips the address space and
    no page of it is unmapped for a hinting fault or migrated. With the
    default policy the scans ran in the threads of the CPU part of a step
    (milliseconds over the maps of 100+ GB): a decode step of Qwen3.8 with
    the experts of an overlay file waited 3 to 7 ms for one thread in about
    one call of 20 (RQ6_MIX_PLAN.md). Call it before the threads start (they
    take the policy of the thread that makes them). NP_GEMMA_NUMA_BALANCE=1:
    keep the default. Return True when set."""
    if os.environ.get("NP_GEMMA_NUMA_BALANCE") == "1" or not enabled():
        return False
    try:
        r = _libc.syscall(ctypes.c_long(_SET_MEMPOLICY), ctypes.c_int(MPOL_LOCAL), None,
                          ctypes.c_ulong(0))
    except (AttributeError, OSError):
        return False
    return r == 0


def node_bytes(node):
    """Return the memory of node (MemTotal), in bytes, or 0."""
    try:
        with open("/sys/devices/system/node/node%d/meminfo" % node) as f:
            for line in f:
                if "MemTotal:" in line:
                    return int(line.split()[-2]) * 1024
    except OSError:
        pass
    return 0


def node_cpus(node):
    """Return the set of the CPUs of node."""
    return {c for c, n in node_of_cpu().items() if n == node}


def pin_thread(node):
    """Keep the calling thread on the CPUs of node. A thread that is already
    on CPUs of that node only (such as the first thread of OpenMP, bound by
    OMP_PROC_BIND) keeps its CPUs. Return the CPUs of the thread."""
    cpus = node_cpus(node)
    now = os.sched_getaffinity(0)
    if cpus and not now <= cpus:
        os.sched_setaffinity(0, (now & cpus) or cpus)
    return os.sched_getaffinity(0)


def page_nodes(a):
    """Return the node of each page of the array a (move_pages with no
    target), an int32 array; a negative value is an error, such as a page
    that is not there yet. For a check."""
    nums = _syscalls()
    if nums is None:
        return np.zeros(0, dtype=np.int32)
    page = mmap.PAGESIZE
    start = a.ctypes.data // page * page
    n = (a.ctypes.data + a.nbytes - start + page - 1) // page
    pages = (ctypes.c_void_p * n)(*[start + k * page for k in range(n)])
    status = np.zeros(n, dtype=np.int32)
    r = _libc.syscall(ctypes.c_long(nums[1]), ctypes.c_int(0), ctypes.c_ulong(n), pages,
                      None, ctypes.c_void_p(status.ctypes.data), ctypes.c_int(0))
    if r < 0:
        e = ctypes.get_errno()
        raise OSError(e, "move_pages: %s" % os.strerror(e))
    return status
