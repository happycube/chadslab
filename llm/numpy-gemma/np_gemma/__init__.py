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


# The attention uses small matrix products. The BLAS library must use one
# thread for them. Many BLAS threads fight the OpenMP threads of the int4
# kernel and make a long decode slow. Set the values before NumPy loads.
# OMP_NUM_THREADS follows the physical core count. Set it yourself to override
# the value, for example OMP_NUM_THREADS=6 on a small machine.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", str(_default_threads()))

from .config import Config
from .st import SafeTensors
from .model import Model, KVCache, Session
from .tokenizer import Tokenizer

__all__ = ["Config", "SafeTensors", "Model", "KVCache", "Session", "Tokenizer"]
