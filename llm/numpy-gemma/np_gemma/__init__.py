"""NumPy-only Gemma 4 runtime."""
import os as _os


def _default_threads():
    """Return a default OpenMP thread count for this machine.

    Use one thread for each physical core. A memory-bound kernel gives a small
    return from a second thread on the same core. The count follows the CPUs
    that the process may use. Thus a container limit is respected.
    """
    try:
        allowed = len(_os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = _os.cpu_count() or 1
    cores = 0
    try:
        pairs = set()
        pid = cid = None
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    pid = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    cid = line.split(":", 1)[1].strip()
                elif not line.strip() and pid is not None and cid is not None:
                    pairs.add((pid, cid))
                    pid = cid = None
        cores = len(pairs)
    except OSError:
        pass
    if cores:
        return min(allowed, cores)
    return allowed


# A long prompt spends a large part of its time in the attention matrix
# products. One BLAS thread is too few for them. Many threads fight the OpenMP
# threads of the int4 kernel. A small count is the compromise: it gives the
# prompt about 7 per cent and leaves the decode unchanged. Set the values
# before NumPy loads. OMP_NUM_THREADS follows the physical core count. Set it
# yourself to override the value, for example OMP_NUM_THREADS=6 on a small
# machine.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", str(min(8, _default_threads())))
_os.environ.setdefault("OMP_NUM_THREADS", str(_default_threads()))
# Keep each OpenMP thread on one core. The threads then keep their caches and
# the pages that they read, and the OS does not move them. The runner of the
# parts (np_gemma/parts.py) needs it: on jackal, a step of the 26B in 2 parts
# takes 45.5 ms with it and 52.1 ms without it. A step of one program changes
# by less than the noise. Set the variables yourself to override the values.
_os.environ.setdefault("OMP_PLACES", "cores")
_os.environ.setdefault("OMP_PROC_BIND", "close")
# The CPUs of the process before any thread is pinned (the main thread is
# pinned later): the GPU library takes the CPU of its copy workers from them
# (gpu.cu gg_worker_bind: the last of the node of the GPU, outside the teams).
try:
    _os.environ.setdefault("NP_GEMMA_START_CPUS",
                           ",".join(str(c) for c in sorted(_os.sched_getaffinity(0))))
except (AttributeError, OSError):
    pass


def _no_numa_balancing():
    """numa.no_numa_balancing before NumPy and OpenMP make their threads (a
    thread takes the memory policy of the thread that makes it)."""
    import ctypes
    import glob
    import platform
    if _os.environ.get("NP_GEMMA_NUMA_BALANCE") == "1" or _os.environ.get("NP_GEMMA_NUMA") == "0":
        return
    if platform.machine() != "x86_64" or len(glob.glob("/sys/devices/system/node/node[0-9]*")) < 2:
        return
    try:
        ctypes.CDLL(None, use_errno=True).syscall(238, 4, None, ctypes.c_ulong(0))  # MPOL_LOCAL
    except (AttributeError, OSError):
        pass


_no_numa_balancing()

from .config import Config
from .st import SafeTensors
from .ct import CompressedTensors
from .e4b import E4B, E4BConfig, E4BCache
from .model import Model, KVCache, Session
from .sampling import Sampler
from .tokenizer import Tokenizer

__all__ = ["Config", "SafeTensors", "CompressedTensors", "E4B", "E4BConfig",
           "E4BCache", "Model", "KVCache", "Session", "Sampler", "Tokenizer"]
