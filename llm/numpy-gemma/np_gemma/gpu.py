"""Run the program of a decode step on a CUDA GPU.

SPLIT_PLAN.md, phase 3. The file np_gemma/csrc/gpu.cu has a kernel for each
operation of a program. It also has a runner that records the launches in a
CUDA graph. This module builds that file with nvcc. It copies the data of a
program to the GPU and changes the addresses of the records to device
addresses.

CUDA is optional. Without nvcc or a GPU, available() gives False and the
rest of the package does not change.

The data of a program has three kinds:

- Weights and buffers of the program (Program.keep). A record operand that
  points into such an array gets the device address of the same place. The
  module copies an array to the GPU the first time an operand points into
  it. The int4 kernels read the float16 scale of each block, so the float32
  scales stay on the host. The module checks that the two scales are equal.
- Parameters of a step (the bind). The position is an integer. The rope
  tables change for each step: each has one device buffer, and each step
  copies the new values. The scores are scratch on the device.
- The cache. GPUCache keeps a device copy of each buffer of the cache. The
  copy on the device is the true one while the GPU runs the steps. Call
  GPUCache.to_host before the CPU reads the cache again.

The first version runs a decode step of one token of the E4B model. It also
runs the output head (a Q6_K table) with the soft cap.

    g = gpu.E4BGPU(model)
    g.attach(cache)                  # after the prompt pass on the CPU
    xn = g.step([token], pos, cache) # the hidden state after the final norm
    logits = g.logits()              # the head of the last step
    g.detach(cache)                  # the cache is on the host again
"""
from __future__ import annotations

import bisect
import ctypes
import hashlib
import os
import weakref
from collections import OrderedDict
import time
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import program as P
from .gpumm import (Buffer, DeviceCache, ExpertPool, GpuMem, ProgramLRU, _alloc_retry,  # noqa: F401
                    mem, mem_info, pinned)
from .model import KV_KEEP

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "csrc" / "gpu.cu"
_LIB_DIR = _HERE / "_libs"
_lib = None
_error = None

# The operands that keep their host address. The GPU kernels do not read the
# float32 scales of the int4 matrices: they read the float16 scale in each
# block. The records of the handoff to the CPU hold host addresses: pinned
# buffers and the address of a CPU program.
SKIP = {P.INT4_LINEAR: (2,), P.INT4_MULTI4: (3, 7, 11, 15),
        P.INT4_LINEAR_MT: (2,), P.INT4_MULTI4_MT: (4, 8, 12, 16),
        P.RMS_NORM_MULTI4: (6, 10, 14, 18), P.GELU_MUL_INT4: (5,),
        P.TO_HOST: (1, 4, 7), P.CPU_JOIN: (0,), P.TO_DEV: (0,), P.FETCH: (0,),
        P.CPU_START: (0,), P.SIGNAL: (0, 3, 6, 9), P.AWAIT: (0, 2), P.D2H: (1,), P.H2D: (0,),
        P.CPU_TASK: (0, 1, 2)}


def _nvcc():
    """Return the path of nvcc, or None."""
    for c in (os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and Path(c).exists():
            return c
    return None


def _build():
    """Build the library when necessary. Return its path. The name has a hash
    of the source, the compiler version, and the flags, as in cops."""
    nvcc = _nvcc()
    if nvcc is None:
        raise RuntimeError("no nvcc: set NVCC or put nvcc on the PATH")
    flags = ["-O3", "-arch=native", "-shared", "-Xcompiler", "-fPIC"]
    version = subprocess.check_output([nvcc, "--version"], text=True).strip().splitlines()[-1]
    # the hash covers the headers that gpu.cu includes (the TQ6 tables)
    text = _SRC.read_text() + (_SRC.parent / "tq6_tables.h").read_text()
    key = hashlib.sha256((text + "|" + version + "|" + " ".join(flags))
                         .encode()).hexdigest()[:16]
    lib = _LIB_DIR / ("libgemma_gpu_" + key + ".so")
    from .cops import build_lib, lib_ready
    if lib_ready(lib):
        return lib
    return build_lib(lambda tmp: [nvcc] + flags + ["-o", str(tmp), str(_SRC)], lib,
                     "libgemma_gpu_", 8)


def lib():
    """Return the library. Build and load it on the first call."""
    global _lib, _error
    if _lib is not None:
        return _lib
    if _error is not None:
        raise RuntimeError(_error)
    try:
        L = ctypes.CDLL(str(_build()))
    except Exception as exc:  # no nvcc, or the build failed
        _error = "the GPU library is not available: %s" % exc
        raise RuntimeError(_error) from exc
    vp, sz, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    L.gg_last_error.restype = ctypes.c_char_p
    L.gg_where.argtypes = [vp]
    L.gg_where.restype = vp
    L.gg_init.argtypes = [i]
    L.gg_mem_info.argtypes = [ctypes.POINTER(sz), ctypes.POINTER(sz)]
    L.gg_malloc.argtypes = [sz]
    L.gg_malloc.restype = vp
    L.gg_free.argtypes = [vp]
    L.gg_h2d.argtypes = [vp, vp, sz]
    L.gg_d2h.argtypes = [vp, vp, sz]
    L.gg_mix_stats_on.argtypes = [i]
    L.gg_mix_stats.argtypes = [vp, i]
    L.gg_load.argtypes = [vp, i]
    L.gg_load.restype = vp
    L.gg_run.argtypes = [vp, vp]
    L.gg_unload.argtypes = [vp]
    L.gg_prepare.argtypes = [vp]
    L.gg_clear_error.argtypes = []
    L.gg_profile.argtypes = [vp, vp, vp]
    L.gg_q6k_head.argtypes = [vp, vp, vp, i, i, ctypes.c_float, i]
    L.gg_q4_head.argtypes = [vp, vp, vp, i, i, ctypes.c_float, i]
    L.gg_topk.argtypes = [vp, i, i, i, ctypes.c_float, vp, vp, vp]
    L.gg_argmax_rows.argtypes = [vp, i, i, vp]
    L.gg_embed_q4.argtypes = [vp, vp, vp, i, ctypes.c_float]
    L.gg_embed_q4_rows.argtypes = [vp, vp, vp, i, i, ctypes.c_float]
    L.gg_set_cpu_runner.argtypes = [vp]
    L.gg_host_register_rw.argtypes = [vp, ctypes.c_size_t]
    L.gg_set_moe_rot.argtypes = [ctypes.c_int]
    L.gg_set_moe_rot.restype = None
    L.gg_set_pool.argtypes = [vp]
    L.gg_fence.restype = ctypes.c_int64
    L.gg_fence_done.argtypes = [ctypes.c_int64, i]
    L.gg_pool_dims.argtypes = [vp, vp]
    L.gg_host_register_rw.restype = ctypes.c_int64
    L.gg_set_tc.argtypes = [i]
    L.gg_host_alloc.argtypes = [sz]
    L.gg_cache_copy.argtypes = [vp, i]
    L.gg_cache_query.argtypes = [i, i]
    L.gg_gdn_commit.argtypes = [vp, vp, vp] + [i] * 6
    L.gg_host_alloc.restype = vp
    L.gg_d2d.argtypes = [vp, vp, sz]
    L.gg_zero.argtypes = [vp, sz]
    # The products of a large group: NP_GEMMA_GPU_TC=1 (the default) runs
    # them on the tensor cores with float16 inputs, 8 with int8 inputs (the
    # Q8_0 form: faster, less exact), and 0 with float32 kernels.
    L.gg_set_tc(int(os.environ.get("NP_GEMMA_GPU_TC", "1")))
    if L.gg_init(int(os.environ.get("NP_GEMMA_GPU_DEVICE", "0"))) != 0:
        _error = L.gg_last_error().decode()
        raise RuntimeError(_error)
    # A GP_CPU_JOIN record runs a CPU program with gemma_run of the CPU library.
    from . import cops
    if cops._lib is not None:
        # gemma_run_task: a team of NP_GEMMA_GPU_CPU_THREADS bound spread;
        # by default half the cores on a machine of two or more NUMA nodes
        # (the 2-socket Xeon: Qwen3.8 26.0 tok/s, all 48 cores 21.3), else
        # OMP_NUM_THREADS
        n = os.environ.get("NP_GEMMA_GPU_CPU_THREADS")
        if n is None:
            from . import numa
            n = int(os.environ.get("OMP_NUM_THREADS", "0") or 0) // 2 if numa.enabled() else 0
        cops._lib.gemma_set_task_threads(ctypes.c_int(int(n)))
        L.gg_set_cpu_runner(ctypes.cast(cops._lib.gemma_run_task, ctypes.c_void_p))
        _place_threads(L)
    _lib = L
    return L


def _place_threads(L):
    """The CPUs around the teams of gemma_run_task (gg_cpu_roles): the
    thread that loads the library (the thread of the model: serve_qwen4's
    model thread, else the main thread) alone on the first CPU of the GPU's
    node (NP_GEMMA_MAIN_CPU; -1: no pin), the copy workers on its last; the
    teams leave both out (NP_GEMMA_RESERVED_CPUS), so the master of the teams
    (the GPU runner, 84-91% of a CPU in a decode) takes a CPU of its own too.
    OMP_PROC_BIND close had put the model thread and the runner on CPU 0
    together (67% + 34% of a decode)."""
    if not hasattr(L, "gg_cpu_roles"):
        return
    copy_cpu, main_cpu = ctypes.c_int(-1), ctypes.c_int(-1)
    L.gg_cpu_roles(ctypes.byref(copy_cpu), ctypes.byref(main_cpu))
    m = int(os.environ.get("NP_GEMMA_MAIN_CPU", main_cpu.value))
    rsv = [c for c in (m, copy_cpu.value) if c >= 0]
    if rsv:
        os.environ.setdefault("NP_GEMMA_RESERVED_CPUS", ",".join(str(c) for c in rsv))
    if m >= 0:
        try:
            os.sched_setaffinity(0, {m})
        except OSError:
            pass


def available():
    """Return True when the library builds and a GPU is present."""
    try:
        lib()
        return True
    except RuntimeError:
        return False


def _check(rc):
    if rc != 0:
        raise RuntimeError(lib().gg_last_error().decode() + where_text())


# The GPUPrograms by handle (where: the program of the last run).
_PROGS = weakref.WeakValueDictionary()


def _op_names():
    from . import program as P
    names = {}
    for k, v in vars(P).items():
        if k.isupper() and isinstance(v, int) and not k.startswith("T_"):
            names.setdefault(v, k)
    return names


def where():
    """The place of the last run (gg_where; no CUDA call, so it works after
    a lost context): the program (its label, e.g. ("mix_progs", 256)), the
    record that the host queued last, its operation, the segment of the
    launch, and sync (NP_GEMMA_GPU_SYNC_CHECK). A kernel error shows up at a
    record at or after the one at fault. None before the first run."""
    if _lib is None:
        return None
    out = (ctypes.c_int * 5)()
    h = _lib.gg_where(out)
    if not h:
        return None
    g = _PROGS.get(h)
    pc, op = out[0], out[1]
    rec = {"handle": hex(h), "label": getattr(g, "label", None), "record": pc, "op": op,
           "op_name": _op_names().get(op), "segment": [out[2], out[3]], "sync": bool(out[4])}
    if g is not None:
        recs = getattr(g.prog, "recs", ())
        rec["records"] = len(recs)
        rec["params"] = getattr(g, "last_params", None)
        if 0 <= pc < len(recs):
            rec["op_args"] = [v for _t, v in recs[pc][1]][:24]
    return rec


def where_text():
    try:
        w = where()
    except Exception:
        return ""
    if not w:
        return ""
    return " [the last record queued: %s record %d (%s), segment %d-%d%s]" % (
        w["label"], w["record"], w["op_name"] or w["op"], w["segment"][0], w["segment"][1],
        "" if w["sync"] else "; the one at fault is at or before it")


class Mirror:
    """The device copies of the host arrays of a program.

    add() makes the host range of an array known. translate() returns the
    device address of a host address in a known range. It copies the array
    to the device the first time.
    """

    def __init__(self):
        self.starts = []    # sorted host start addresses
        self.arrays = {}    # host start -> array
        self.bufs = {}      # host start -> Buffer
        self.checked = set()  # the int4 matrices whose scales are checked
        # the kind of the new device copies (GpuMem), and the key of the
        # program they are made for (the reclaimers keep it)
        self.kind = "program"
        self.keep = None

    def add(self, a):
        # A view (such as one row of a buffer) registers the array that owns
        # its memory. Else the view and its array can have the same start,
        # and the device copy has the size of the view.
        while isinstance(a.base, np.ndarray):
            a = a.base
        start = a.ctypes.data
        if start in self.arrays or a.nbytes == 0:
            return
        bisect.insort(self.starts, start)
        self.arrays[start] = a

    def owner(self, addr):
        """Return the host start of the array that holds addr, or None."""
        k = bisect.bisect_right(self.starts, addr) - 1
        if k < 0:
            return None
        start = self.starts[k]
        return start if addr < start + self.arrays[start].nbytes else None

    def forget(self, start):
        """Drop the array at start and its device copy (a program that held
        it is freed); the place of the call is kept for an error later."""
        b = self.bufs.pop(start, None)
        if b is not None:
            b.free()
        if start in self.arrays:
            del self.arrays[start]
            self.starts.remove(start)
            import traceback
            self.__dict__.setdefault("forgotten", {})[start] = "".join(traceback.format_stack(limit=10)[:-1])

    def device(self, start):
        """Return the Buffer of an array. Copy the array the first time."""
        b = self.bufs.get(start)
        if b is None:
            if start not in self.arrays:
                gone = getattr(self, "forgotten", {}).get(start)
                raise KeyError("host array %#x has no device copy: %s" % (
                    start, ("dropped at:\n" + gone) if gone else "never added to the mirror"))
            a = np.ascontiguousarray(self.arrays[start])
            # pinned: the programs hold its address in their code
            b = self.bufs[start] = Buffer(a.nbytes, self.kind, keep=self.keep, pinned=True)
            b.upload(a)
        return b

    def translate(self, addr):
        start = self.owner(addr)
        if start is None:
            return None
        return self.device(start).ptr + (addr - start)

    def buffer_of(self, a):
        """Return the Buffer of a known array."""
        return self.device(a.ctypes.data)

    def nbytes(self):
        return sum(b.nbytes for b in self.bufs.values())


def _check_scales(prog, done=None):
    """Check that the float16 scale of each int4 block equals the float32
    scale that the CPU kernels read. done is a set of the matrices that an
    earlier check covered."""
    done = set() if done is None else done
    for op, args in prog.recs:
        # (w, s, rows, cols) of each matrix of the record.
        if op == P.INT4_LINEAR:
            pairs = [(1, 2, 4, 5)]
        elif op == P.INT4_MULTI4:
            pairs = [(2 + 4 * m, 3 + 4 * m, 5 + 4 * m, 1) for m in range(4)]
        elif op == P.RMS_NORM_MULTI4:
            pairs = [(5 + 4 * m, 6 + 4 * m, 8 + 4 * m, 3) for m in range(4)]
        elif op == P.GELU_MUL_INT4:
            pairs = [(4, 5, 7, 8)]
        elif op == P.INT4_LINEAR_MT:
            pairs = [(1, 2, 4, 5)]
        elif op == P.INT4_MULTI4_MT:
            pairs = [(3 + 4 * m, 4 + 4 * m, 6 + 4 * m, 1) for m in range(4)]
        else:
            continue
        for wk, sk, rk, ck in pairs:
            w, s = args[wk][1], args[sk][1]
            if not w or w in done:
                continue
            done.add(w)
            rows, cols = args[rk][1], args[ck][1]
            if not rows:
                continue
            n = rows * (cols // 32)
            blocks = P._arr(w, n * 18, ctypes.c_uint8).reshape(n, 18)
            half = blocks[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
            if not np.array_equal(half, P._arr(s, n)):
                raise ValueError("an int4 matrix has float32 scales that are not its "
                                 "float16 scales; the GPU kernels read the float16 scale")


class GPUProgram:
    """A Program on the GPU: its records with device addresses, its data, and
    the device buffers of its parameters."""

    def __init__(self, prog, graph=True, mirror=None, tc=True, i8=False, atc=False):
        """mirror is the Mirror of an earlier program of the same model. The
        programs then share the device copies of the weights. tc False keeps
        float32 products for the large groups of this program, in place of
        the tensor cores. i8 True gives int8 activations to the int4 products
        of the large groups (k_gemm_q8) also with tc False; i8 2 gives them
        the int16 form (k_quant_x2: two int8 planes), and to the KQ_Q4X
        experts of a prompt. atc True runs the
        attention of the large groups on the tensor cores also with tc False."""
        self.run_blocks = None    # the model's blocks for each run (GpuMem.lock)
        self.fence = 0
        self.prog = prog
        self.mirror = mirror if mirror is not None else Mirror()
        for a in prog.keep:
            if isinstance(a, np.ndarray):
                self.mirror.add(a)
        _check_scales(prog, self.mirror.checked)
        buf = prog.buf.copy()
        n_env = int(buf[1])
        code = buf[4 + n_env:].view(P.REC)
        for i, (op, args) in enumerate(prog.recs):
            skip = SKIP.get(op, ())
            for k, (tag, val) in enumerate(args):
                if tag != P.T_INT or val == 0 or k in skip:
                    continue
                d = self.mirror.translate(val)
                if d is not None:
                    code[i]["v"][k] = d
        self.handle = lib().gg_load(buf.ctypes.data, (1 if graph else 0) | (0 if tc else 2) | (4 if i8 else 0) |
                                  (8 if atc else 0) | (16 if i8 == 2 else 0))
        if not self.handle:
            msg = lib().gg_last_error().decode()
            lib().gg_clear_error()
            # gg_load fails for memory (its scratch buffers): the caller can
            # free some and try again
            raise (MemoryError if "memory" in msg else RuntimeError)(msg)
        self.env = np.array(prog.buf[4:4 + n_env], dtype=np.int64)
        self.named = {}     # the device buffer of each bound array name
        self.label = None   # what the program is (where; ProgramLRU sets it)
        _PROGS[self.handle] = self

    def upload(self, name):
        """Copy the host values of a buffer of the compiler to the GPU."""
        self.mirror.buffer_of(self.prog.names[name]).upload(self.prog.names[name])

    def download(self, name):
        """Copy a buffer of the compiler from the GPU to its host array."""
        self.mirror.buffer_of(self.prog.names[name]).download(self.prog.names[name])

    def bind(self, kw, cache=None, scratch=("scores",)):
        """Write the parameters of a step into the environment.

        An array of the cache gets its device buffer. A name in scratch gets
        a device buffer with no copy. Any other array gets a device buffer for
        its name, and the step copies its values.
        """
        by_name = self.prog.by_name
        # the parameters of the run, for where (a crash report)
        self.last_params = {name: (list(v.shape) if isinstance(v, np.ndarray) else
                                   v if isinstance(v, (int, float, str)) else repr(v)[:80])
                            for name, v in kw.items()}
        for name, v in kw.items():
            s = by_name.get(name)
            if s is None:
                continue
            if isinstance(v, np.ndarray):
                d = cache.device(v) if cache is not None else None
                if d is None:
                    b = self.named.get(name)
                    if b is None or b.nbytes < v.nbytes:
                        if b is not None:
                            b.free()
                        # keep: the memory for it must not come from closing
                        # this program (ProgramLRU: lru_key); a mixed group of
                        # 4096 rows at 98304 of context closed its own
                        # program here, then ran it (gg_run of NULL)
                        b = self.named[name] = Buffer(max(v.nbytes, 2 * (b.nbytes if b else 0)),
                                                      "program", keep=getattr(self, "lru_key", None))
                    if name not in scratch:
                        b.upload(np.ascontiguousarray(v))
                    d = b.ptr
                self.env[s.index] = d
            elif isinstance(v, (float, np.floating)):
                self.env[s.index] = P._f32_bits(v)
            else:
                self.env[s.index] = int(v)

    def _alive(self):
        """A run of a closed program (gg_run of NULL: a segfault) is an
        error that names the program and where it was closed."""
        if not self.handle:
            raise RuntimeError("the GPU program %s was closed before this run; closed at:\n%s" % (
                self.label, getattr(self, "closed_by", "?")))

    def run(self):
        self._alive()
        s = self.prog.by_name.get("seq")
        if s is not None:
            # The value of the flags of this run (GP_SIGNAL, GP_AWAIT,
            # GP_CPU_TASK): it grows, so the flags need no reset.
            self.seq = getattr(self, "seq", 0) + 1
            self.env[s.index] = self.seq
        # GpuMem: the blocks of the run stay where they are until the GPU is
        # done with it (a fence; gg_run returns before): its parameters, and
        # those of its model (run_blocks: the cache, the pool)
        mem().poll()            # (the fences of the runs before: their locks)
        blocks = list(self.named.values())
        if self.run_blocks is not None:
            blocks += self.run_blocks()
        mem().lock(blocks)
        try:
            _check(lib().gg_run(self.handle, self.env.ctypes.data))
        except RuntimeError:
            # the fence fails too after a lost context: the error of the run
            # is the one to report
            try:
                self.fence = mem().fence(blocks)
            except RuntimeError:
                pass
            raise
        self.fence = mem().fence(blocks)

    def profile(self):
        """Run the program with a timer on each record. Return the time of
        each record in ms. Bind the step first."""
        ms = np.zeros(len(self.prog.recs), dtype=np.float32)
        self._alive()
        _check(lib().gg_profile(self.handle, self.env.ctypes.data, ms.ctypes.data))
        return ms

    def prepare(self):
        """Record the CUDA graphs now and upload them (the first run does it
        otherwise). Raise MemoryError when the GPU has no room for them."""
        if self.handle and lib().gg_prepare(self.handle) != 0:
            msg = lib().gg_last_error().decode()
            lib().gg_clear_error()
            raise MemoryError(msg)

    def close(self):
        if self.handle:
            lib().gg_unload(self.handle)
            self.handle = None
            # who closed it: a run after this is an error that names it
            import traceback
            self.closed_by = "".join(traceback.format_stack(limit=12)[:-1])


class GPUCache(DeviceCache):
    """The device copy of the buffers of an E4BCache (gpumm.DeviceCache:
    locked; the rows past the positions in use are lent to GpuMem).

    Each buffer of the cache holds (capacity, heads, head_dim) values. The
    device buffer has the same shape. attach() copies the host buffers to the
    GPU. From then on, the device copy is the true one. to_host() copies it
    back. A cache that must grow first comes back to the host, grows, and
    goes to the GPU again.
    """

    def __init__(self):
        super().__init__("cache0")
        self.bufs = {}     # id of a host array -> (array, Buffer)
        self.oom = None    # (GpuMem reclaims memory for the buffers)

    def attach(self, cache):
        self.release()
        for store in list(cache.kv.values()):
            for a in store:
                if a is not None and id(a) not in self.bufs:
                    b = self._new(a.nbytes, rows=a.shape[0])
                    b.upload(a)
                    self.bufs[id(a)] = (a, b)

    def device(self, a):
        e = self.bufs.get(id(a))
        return None if e is None else e[1].ptr

    def blocks(self):
        return [b for _a, b in self.bufs.values()]

    def to_host(self):
        for a, b in self.bufs.values():
            b.download(a)

    def reserve(self, cache, need):
        """Make the cache large enough for need positions."""
        if need <= cache.cap:
            return
        self.to_host()
        cache._reserve(need)
        self.attach(cache)

    def release(self):
        self.unshare()
        for _a, b in self.bufs.values():
            self._free(b)
        self.bufs = {}


class PoolCompiler(P.Compiler):
    q4x = False     # the GPU reads the int4 matrices of the model, not KQ_Q4X

    """A compiler that reuses the buffers of a layer in every layer.

    With pool, the n-th buffer of a shape in a layer is the same array in
    every layer. A layer does not read the buffers of an earlier layer, and
    the GPU runs the layers in order, so this is safe. A buffer outside the
    layers is a new array. A group of 1024 tokens of the 26B then needs about
    0.3 GB, not 8 GB.
    """

    def __init__(self, model, pool=False):
        super().__init__(model)
        self.pool_on = pool
        self.pool = {}
        self.pool_n = {}
        self.in_layer = False

    def compile(self, form):
        if form[0] == "layer":
            self.pool_n = {}
            self.in_layer = True
            try:
                return super().compile(form)
            finally:
                self.in_layer = False
        return super().compile(form)

    def buffer(self, shape, dtype=np.float32):
        if not (self.pool_on and self.in_layer):
            return super().buffer(shape, dtype)
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        k = (shape, np.dtype(dtype).str)
        n = self.pool_n.get(k, 0)
        self.pool_n[k] = n + 1
        b = self.pool.get((k, n))
        if b is None:
            b = self.pool[(k, n)] = np.zeros(shape, dtype=dtype)
        return b


class E4BGroupCompiler(PoolCompiler):
    """The compiler of a group of the E4B model. A small group (at most MT_CPU
    tokens, such as an MTP verify group) runs the attention of each query
    with the record of a decode step. That kernel computes in float32, as
    the decode step does, so the verify group agrees with the decode."""

    def kernel(self, head, vals, out=None):
        if head == "attn_e4b" and vals[1].shape[0] <= MT_CPU:
            return self.attn_small(*vals)
        return super().kernel(head, vals, out)

    def attn_small(self, layer, q):
        plan = self.cfg.plan[layer]
        hd, qh = plan.head_dim, plan.num_q_heads
        t = q.shape[0]
        out = self.buffer((t, qh * hd))
        s = lambda name: self.p.slot("%s.%d" % (name, plan.source))  # noqa: E731
        slide = 1 if os.environ.get("NP_GEMMA_SLIDE", "1") == "1" else 0
        for j in range(t):
            pos_j = self.scalar("+", [self.p.slot("pos"), j])
            self.p.emit(P.ATTN_F32H, q[j:j + 1], s("k"), s("v"), self.p.slot("scores"),
                        out[j:j + 1], qh, plan.num_kv_heads, hd, 1, pos_j, s("hs"),
                        plan.window, slide)
        return out


def _fuse_kq(prog):
    """Fewer launches of the GGUF products of an E4B program on the GPU:
    drop GP_KQ_QUANT (the GPU products read x), and make up to 5 GP_KQ_LINEAR
    of a few tokens on the same x one GP_KQ_MULTI (the query, the key, and
    the value; the gate and the up matrix). The values are the same."""
    recs = [r for r in prog.recs if r[0] != P.KQ_QUANT]
    out, i = [], 0
    while i < len(recs):
        op, a = recs[i]
        if op == P.KQ_LINEAR and a[8][0] == P.T_INT and a[8][1] <= MT_CPU:
            group, j = [a], i + 1
            while (j < len(recs) and recs[j][0] == P.KQ_LINEAR and recs[j][1][3] == a[3]
                   and len(group) < 5):
                group.append(recs[j][1])
                j += 1
            if len(group) > 1:
                args = [a[3], a[7], a[8], (P.T_INT, len(group))]
                for g in group:
                    args += [g[4], g[5], g[6], g[9]]
                out.append((P.KQ_MULTI, args))
                i = j
                continue
        out.append((op, a))
        i += 1
    prog.recs = out
    return prog.finish()


# 1: the Q4_0 matrices of an E4B step and of a verify group run with int8 x
# and dp4a (E4B.kq_q4, GP_KQ_LINEAR), as the K quants. With float32 x, the
# loads of x for each token made a group of 3 tokens cost 1.75 steps
# (MTP_PLAN.md). 0 keeps float32 x (GP_INT4_LINEAR). The prompt keeps the
# int4 records either way.
Q4_I8 = os.environ.get("NP_GEMMA_GPU_Q4_I8", "1") != "0"

# The dense Q4_0 matrices of a step and of a verify group of the 12B and the
# 26B with int8 x and dp4a (program._Q4KQ), as Q4_I8 for the E4B. Unset: on
# for a dense model (the 12B: a verify group of 3 tokens 35.4 -> 24.9 ms, KL
# 0.00012 -> 0.0004), off for the 26B (its activations have larger outliers:
# KL 0.0001 -> 0.0009; MTP_PLAN.md). 1 or 0 sets it for both.
Q4_I8_DENSE = os.environ.get("NP_GEMMA_GPU_Q4_I8_DENSE")


def q4_i8_dense(model):
    """Return True when the Q4_0 matrices of the GPU step and small groups of
    model take int8 x (Q4_I8_DENSE)."""
    if Q4_I8_DENSE is not None:
        return Q4_I8_DENSE == "1"
    return not model.cfg.enable_moe_block

# The groups of up to this many tokens start the head right after the step
# (E4BGPU._head_ahead); 0 turns it off.
HEAD_AHEAD = int(os.environ.get("NP_GEMMA_GPU_HEAD_AHEAD", "8"))


def compile_e4b_group(model, t):
    """Compile a step of t tokens of the E4B model for the GPU. A small group
    (at most MT_CPU tokens, an MTP verify group) has the form of the decode
    step (fused 2, no reuse of the buffers), so each token gets the bits of a
    decode step and MTP gives the tokens of the plain decode. A larger group
    (a part of a prompt) reuses the buffers of the layers (PoolCompiler)."""
    cfg = model.cfg
    small = t <= MT_CPU
    c = E4BGroupCompiler(model, pool=not small)
    c.q4_kq = small and Q4_I8
    c.env["x"] = np.zeros((t, cfg.hidden_size), dtype=np.float32)
    c.env["tok"] = np.zeros((t, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input),
                            dtype=np.float32)
    c.p.slot("pos")
    c.compile(P.e4b_step_form(model, fused=2 if small else True))
    c.p.tokens = t
    return _fuse_kq(c.p.finish())


class E4BGPU(ProgramLRU):
    """The decode step, the groups of tokens, and the prompt pass of the
    E4B model on the GPU. See the module text."""

    def __init__(self, model, graph=True):
        self.model = model
        progs = model.__dict__.setdefault("_programs", {})
        # The GPU step has the fused operations; the CPU step (key 1) has not.
        prog = progs.get("gpu1")
        if prog is None:
            prog = progs["gpu1"] = _fuse_kq(P.compile_e4b_step(model, 1, fused=2, q4_kq=Q4_I8))
        self.prog = prog
        self.g = GPUProgram(prog, graph=graph)
        self.graph = graph
        self.cache = GPUCache()
        self.head = None
        self.groups = {}      # t -> (Program, GPUProgram)
        self._lru_init()
        self.cache.oom = self._evict
        self.last = self.g.mirror.buffer_of(prog.names["xn"]).ptr
        self.rows = 1
        self._mm_freeze()       # GpuMem: the weights and the step stay
        self._run_blocks = self.cache.blocks

    def attach(self, cache):
        self.cache.attach(cache)

    def detach(self, cache):
        self.cache.to_host()
        self.cache.release()

    def step(self, tokens, pos, cache):
        """Run a step of one token. Return the hidden state after the final
        norm, shape (1, hidden)."""
        model, cfg = self.model, self.model.cfg
        ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
        assert ids.size == 1, "the GPU runs a step of one token"
        self.cache.reserve(cache, pos + 1)
        self.cache.share_tail(pos + 1)          # (GpuMem: the rows past the run are lent)
        kw = P.e4b_step_params(self.prog, model, cache, pos, self._rope)
        names = self.prog.names
        names["x"][:] = model.embed_rows(P.E4B_PREFIX + "embed_tokens", ids) * cfg.embed_scale
        tok = model.embed_rows(P.E4B_PREFIX + "embed_tokens_per_layer", ids)
        names["tok"][:] = (tok * cfg.per_layer_embed_scale).reshape(1, -1)
        self.g.upload("x")
        self.g.upload("tok")
        self.g.bind(kw, self.cache)
        self.g.run()
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.rows = 1
        self._head_ahead(1)
        self.g.download("xn")
        return names["xn"].copy()

    def _group(self, t):
        e = self.groups.get(t)
        if e is None:
            prog = compile_e4b_group(self.model, t)
            e = self.groups[t] = (prog, self._build_program(prog, t))
        self._used(t)
        return e

    def group(self, tokens, pos, cache, size=None, media=None):
        """Run a group of tokens from position pos. Return the hidden states
        after the final norm, shape (len(tokens), hidden). size pads the group
        to a program of that many tokens. The padding rows write cache rows
        after the group, and a later step writes them again.

        media is a list of media.Span (absolute positions): their rows take
        the place of the token rows, and the per-layer token rows of those
        positions are those of the pad token (id 0), as in transformers."""
        model, cfg = self.model, self.model.cfg
        ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
        t = ids.size
        size = size or t
        prog, g = self._group(size)
        if self._ensure(cache):
            self.cache.attach(cache)
        self.cache.reserve(cache, pos + size)
        self.cache.share_tail(pos + size)       # (GpuMem: the rows past the run are lent)
        kw = P.e4b_step_params(prog, model, cache, pos, self._rope)
        cache.n = pos + t
        names = prog.names
        names["x"][:t] = model.embed_rows(P.E4B_PREFIX + "embed_tokens", ids) * cfg.embed_scale
        names["x"][t:] = 0.0
        tok_ids = ids
        for sp in media or ():
            lo, hi = max(sp.start, pos), min(sp.end, pos + t)
            if lo < hi:
                names["x"][lo - pos:hi - pos] = sp.rows[lo - sp.start:hi - sp.start]
                if tok_ids is ids:
                    tok_ids = ids.copy()
                tok_ids[lo - pos:hi - pos] = 0
        tok = model.embed_rows(P.E4B_PREFIX + "embed_tokens_per_layer", tok_ids)
        names["tok"][:t] = (tok * cfg.per_layer_embed_scale).reshape(t, -1)
        names["tok"][t:] = 0.0
        g.upload("x")
        g.upload("tok")
        g.bind(kw, self.cache)
        g.run()
        self.last = g.mirror.buffer_of(names["xn"]).ptr + (t - 1) * cfg.hidden_size * 4
        self.rows = t
        if size == t and t <= HEAD_AHEAD:
            self._head_ahead(t)       # an MTP verify group reads every row
        g.download("xn")
        return names["xn"][:t].copy()

    def _ensure(self, cache):
        """Make the buffers of every layer that stores a key, as
        E4BCache.append does. Return True when a buffer is new."""
        new = False
        for plan in self.model.cfg.plan:
            if plan.shared or plan.idx in cache.kv:
                continue
            shape = (cache.cap, plan.num_kv_heads, plan.head_dim)      # position-major
            store = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
            cache.kv[plan.idx] = store
            if plan.stores:
                cache.shared[plan.kind] = store
            new = True
        return new

    def prefill(self, ids, pos, cache, media=None):
        """Run a prompt from position pos in chunks of CHUNK tokens. A group
        of up to MT_CPU tokens runs as it is. A longer chunk goes to a
        program of the next power of two. Return the hidden states of every
        token. media: see group (the E4B is causal, so a chunk can split an
        image)."""
        ids = list(ids)
        out = []
        for c0 in range(0, len(ids), CHUNK):
            chunk = ids[c0:c0 + CHUNK]
            size = None
            if len(chunk) > MT_CPU:
                size = min(CHUNK, 1 << (len(chunk) - 1).bit_length())
            spans = [sp for sp in media or () if sp.start < pos + c0 + len(chunk)
                     and sp.end > pos + c0]
            if spans:
                out.append(self.group(chunk, pos + c0, cache, size, media=spans))
            else:
                out.append(self.group(chunk, pos + c0, cache, size))
        return np.concatenate(out)

    def logits(self, rows=1):
        """Return the logits of the last rows of the last step or group, with
        the soft cap, shape (rows, vocabulary). The first call copies the
        head to the GPU."""
        model = self.model
        if self.head is None and model._gpu_head() is None:
            return self._kq_logits(rows)
        vocab = self._run_head(rows)
        _check(lib().gg_d2h(self.host_logits.ctypes.data, self.out.ptr, rows * vocab * 4))
        return self.host_logits[:rows].copy()

    def topk(self, rows, k, temperature):
        """Return the candidates of sampling of the last rows (_topk_rows), or
        None for a head that is not Q6_K or Q4_0 (GPU._kq_head)."""
        if self.head is None and self.model._gpu_head() is None:
            return None
        self._run_head(rows)
        return _topk_rows(self, rows, k, temperature)

    def _run_head(self, rows):
        """Run the head on the last rows into self.out. The first call copies
        the head to the GPU. Return the vocabulary size."""
        model, cfg = self.model, self.model.cfg
        if self.head is None:
            kind, w = model._gpu_head()
            w = np.ascontiguousarray(w)
            self.head_fn = lib().gg_q6k_head if kind == "q6k" else lib().gg_q4_head
            vocab = w.shape[0]
            self.head = Buffer(w.nbytes, "weights")
            self.head.upload(w)
            self.out = Buffer(4 * vocab * MT_CPU, "weights")
            self.host_logits = pinned((MT_CPU, vocab))     # a faster copy from the GPU
        assert 1 <= rows <= min(self.rows, MT_CPU)
        vocab = self.host_logits.shape[1]
        cap = float(cfg.final_logit_softcapping or 0.0)
        first = self.last - (rows - 1) * cfg.hidden_size * 4
        _check(self.head_fn(self.head.ptr, first, self.out.ptr, vocab, cfg.hidden_size,
                            cap, rows))
        return vocab

    def _rope(self, kind, pos, t):
        """The GPU addresses of the cosine and the sine rows of positions pos
        to pos + t - 1 for the layers of a kind ("s", "f"): tables of all the
        positions up to a limit, made once (rope.cos_sin, the same bits as a
        table of those positions). A step then copies no table."""
        tabs = self.__dict__.setdefault("rope_tabs", {})
        e = tabs.get(kind)
        if e is None or pos + t > e[0]:
            if e is not None:
                e[1].free()
                e[2].free()
            n = max(4096, 2 * (e[0] if e else 0), pos + t)
            plan = next(p for p in self.model.cfg.plan if p.is_sliding == (kind == "s"))
            from . import rope as rope_mod
            cos, sin = rope_mod.cos_sin(self.model.cfg.rope_inv_freq(plan), np.arange(n))
            bc, bs = Buffer(cos.nbytes), Buffer(sin.nbytes)
            bc.upload(np.ascontiguousarray(cos))
            bs.upload(np.ascontiguousarray(sin))
            e = tabs[kind] = (n, bc, bs, cos.shape[1] * 4)
        return e[1].ptr + pos * e[3], e[2].ptr + pos * e[3]

    def _head_ahead(self, rows):
        """Start the head of the last rows rows now, before the copy of the
        hidden state to the host, so the GPU runs it right after the step.
        logits() and argmax() then only copy the result. A GGUF head that is
        not Q6_K or Q4_0 only (_kq_head)."""
        self.head_done = None
        model = self.model
        if HEAD_AHEAD and self.head is None and model._gpu_head() is None:
            self._kq_head(rows)

    def _kq_head(self, rows):
        """Run the head of a GGUF file whose token table is not Q6_K (the UD
        files of the E2B and the E4B keep it in Q5_K) on the last rows rows:
        a program of one GP_KQ_LINEAR, the soft cap (GP_SOFTCAP), and the
        best token of each row (GP_ARGMAX, the first index of the largest
        value, as np.argmax). Return the Program and the GPUProgram; the
        logits are "out" and the tokens "tok"."""
        model, cfg = self.model, self.model.cfg
        progs = self.__dict__.setdefault("kq_heads", {})
        e = progs.get(rows)
        if e is None:
            k = model.kq(model.head)
            if k is None:
                raise RuntimeError("the GPU head needs a Q6_K or a K quant head")
            hp = P.Program()
            hx = np.zeros((rows, cfg.hidden_size), np.float32)
            out = np.zeros((rows, k.rows), np.float32)
            tok = np.zeros(rows, np.int32)
            hp.names.update(x=hx, out=out, tok=tok)
            hp.emit(P.KQ_LINEAR, None, None, None, hx, k.data, k.type, k.rows, k.cols, rows, out)
            cap = cfg.final_logit_softcapping
            if cap:
                hp.emit(P.SOFTCAP, out, out.size, float(cap))
            for j in range(rows):
                hp.emit(P.ARGMAX, out[j], k.rows, tok[j:j + 1])
            hp = hp.finish()
            e = progs[rows] = (hp, GPUProgram(hp, graph=self.graph, mirror=self.g.mirror))
        hp, hg = e
        assert 1 <= rows <= self.rows
        if getattr(self, "head_done", None) == (self.last, rows):
            return hp, hg             # _head_ahead ran it
        step = cfg.hidden_size * 4
        _check(lib().gg_d2d(hg.mirror.buffer_of(hp.names["x"]).ptr,
                            self.last - (rows - 1) * step, rows * step))
        hg.run()
        self.head_done = (self.last, rows)
        return hp, hg

    def _kq_logits(self, rows):
        """The logits of _kq_head, with the soft cap of the GPU, through a
        pinned buffer (a faster copy)."""
        hp, hg = self._kq_head(rows)
        out = hp.names["out"]
        pin = self.__dict__.setdefault("kq_pinned", {})
        if rows not in pin:
            pin[rows] = pinned(out.shape)
        hg.mirror.buffer_of(out).download(pin[rows])
        return pin[rows].copy()

    def argmax(self, rows=1):
        """Return the best token of each of the last rows of the last step or
        group: the tokens of np.argmax on logits(rows), with a copy of rows
        integers from the GPU and not of the logits."""
        model = self.model
        if self.head is None and model._gpu_head() is None:
            hp, hg = self._kq_head(rows)
            hg.download("tok")
            return [int(v) for v in hp.names["tok"]]
        self._run_head(rows)
        return _argmax_rows(self, rows)


# ---- the 26B model: the experts on the CPU (SPLIT_PLAN.md, phase 4) -----------

def expert_host(model, layer):
    """The bytes of the experts of a layer that the GPU reads (its hot slots,
    the copies of GP_FETCH and of HotCache): (gate and up, down, q4x), each
    (experts, bytes). q4x: the KQ_Q4X copies (ops.q4x_pack_model; the bytes
    of an expert are those of the int4 expert, in groups of 16 rows), unless
    NP_GEMMA_GPU_Q4X=0; the GPU kernels then read that layout (GP_HOT_MOE
    and GP_MOE_GPU with the flag)."""
    w = model._layers[layer]
    gu, dn = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
    n = gu.shape[0]
    e = P.ops._Q4X_MOE.get(gu.ctypes.data)
    if e is not None and os.environ.get("NP_GEMMA_GPU_Q4X", "1") != "0":
        return e[0].reshape(n, -1), e[1].reshape(n, -1), True
    return gu.reshape(n, -1), dn.reshape(n, -1), False


def _arrays(vals):
    out = []
    for v in vals:
        if isinstance(v, np.ndarray):
            out.append(v)
        elif isinstance(v, tuple):
            out += _arrays(v)
    return out


class SplitCompiler(PoolCompiler):
    """The compiler of a step of the 26B model for the GPU, with the experts
    on the CPU.

    The operation moe becomes a GP_TO_HOST record: the copy of its input, its
    weights, and its indices to pinned host memory. Its CPU program is the
    MOE record of the program of one part, on those pinned buffers. The first
    operation that reads the output of the experts gets two records before
    it: GP_CPU_JOIN, which runs the CPU program, and GP_TO_DEV, which copies
    the output to the GPU. The operations between the two parts run on the
    GPU while the CPU computes the experts.
    """

    def __init__(self, model, hot=None, kv="int16", pool=False, stage=None):
        super().__init__(model, pool)
        self.kv = kv
        # The two device buffers for the weights of the experts of a layer
        # (see moe_group_gpu), or None.
        self.stage = stage
        # id of an output of the experts -> (CPU program, host output, event,
        # device buffer of the CPU part, device buffer of the GPU part).
        self.pending = {}
        self.n_events = 0
        self.cpu_progs = []
        self.hot = hot or {}   # layer -> the experts that the GPU holds
        # For a step program: layer -> (hot experts, gate and up blocks, down
        # blocks, slots). slots gives the slot of each expert in the blocks,
        # or -1; HotCache changes it. For a group program: (slots, device
        # address of the gate and up blocks, of the down blocks), see
        # moe_group_gpu.
        self.hot_stores = {}
        # layer -> the entry of hot_stores of the step program, for a small
        # group (see moe_hot_group).
        self.hot_host = {}
        # layer -> the pinned array that GP_HOT_SPLIT fills: the cold
        # experts, their count, and the selection of the step (HotCache).
        self.hot_ip = {}
        # layer -> (tgu, tdn, ranges) of a large group (see tables).
        self.tables_of = {}
        # (layers, experts) int32, or None: a group counts the selections of
        # its routers there (GP_COUNT, HotCache.seed).
        self.counts = None
        # The mixed form of a large group (see moe_group_mix): layer -> the
        # array that marks the experts on the GPU (hot or copied), and the
        # pinned buffers and the scratch that the CPU parts of all the
        # layers share (one CPU part runs at a time).
        self.mix = False
        self.gslots_of = {}
        self.mix_bufs = None
        # The flag form of the handoff to the CPU (a step): no boundary
        # record, so the step is one graph (to_cpu, join).
        self.flags = False
        self.fused = False      # the fused forms of the step (add_norm2, ffn_out)
        self.flag_count = {}    # id of the flags -> the device int of the count

    def kernel(self, head, vals, out=None):
        for a in _arrays(vals):
            if id(a) in self.pending:
                self.join(a)
        if head == "router" and self.counts is not None and vals[0].shape[0] > 1:
            val, idx = super().kernel(head, vals, out)
            layer = vals[1]
            self.p.emit(P.COUNT, idx, self.counts[layer], idx.shape[1], self.p.slot("nreal"),
                        self.counts.shape[1])
            return val, idx
        if head == "moe":
            assert out is None
            return self.moe(*vals)
        if head == "ffn_out" and self.fused:
            m, w1, e, w2, w, x, s, wn = vals
            out2 = self.buffer(x.shape) if wn is not None else None
            self.p.emit(P.FFN_OUT, m, w1, e, w2, w, x, x, x.shape[0], x.shape[1],
                        float(self.eps), float(np.asarray(s).reshape(-1)[0]), wn, out2)
            return out2 if out2 is not None else x
        if head == "kv_write" and self.kv in ("int16", "int8", "k16v8"):
            return self.kv_write16(*vals)
        if head in ("attn_rows_qc", "attn_rows_q8", "attn_rows_qv") and vals[1].shape[0] <= MT_CPU:
            return self.attn_rows_small(*vals)
        if head in ("attn_qc", "attn_q8", "attn_qv") and self.fdts(vals[0]):
            # a step over a window: its rows from row 0 (k_attn_fdts, as a group)
            layer, q, lo, n = vals
            return P.k_attn_qc(self, layer, q, 0, self.value(("+", lo, n)),
                               q8={"attn_q8": True, "attn_qv": "v"}.get(head, False),
                               window=self.cfg.plan[layer].sliding_window)
        return super().kernel(head, vals, out)

    def router_slots(self, layer):
        """The slot of each expert of a layer on the GPU, -1 for a cold one.
        The router of a group reads it for the test of gg_set_reuse."""
        if layer in self.hot_host:
            return self.hot_host[layer][3]
        n = self.cfg.num_experts
        slots = np.full(n, -1, dtype=np.int32)
        hot = self.hot.get(layer, [])
        slots[hot] = np.arange(len(hot), dtype=np.int32)
        self.__dict__.setdefault("_router_slots", []).append(slots)
        return slots

    def fdts(self, layer):
        """A sliding layer of the 26B (head_dim 256, 2 query heads a key head)
        takes the record of a window (k_attn_fdts) in a step and in a group."""
        plan = self.cfg.plan[layer]
        return (FDT_TS and plan.is_sliding and bool(plan.sliding_window) and plan.head_dim == 256 and
                plan.num_q_heads == 2 * plan.num_kv_heads)

    def attn_rows_small(self, layer, q):
        """The attention of a small group, one query at a time with the
        record of a decode step (GP_ATTN_QC). Its kernel splits the keys of
        each head into chunks, and it is faster for a few queries than the
        group kernel. Query j has the position pos + j. A global layer of
        the 26B (head_dim 512, 8 query heads a key head) takes the group of
        up to FDT_TMAX queries in one record (operand 11: t; k_attn_fdtc
        reads the rows once, with the numbers of a step for each query), and
        a sliding layer (fdts) too, with its window (k_attn_fdts)."""
        plan = self.cfg.plan[layer]
        hd, qh = plan.head_dim, plan.num_q_heads
        t = q.shape[0]
        base = "base.%d" % layer
        q8 = {"int8": True, "k16v8": "v"}.get(self.kv, False)
        if t <= FDT_TMAX and (self.fdts(layer) or (
                FDT_TC and not plan.is_sliding and hd == 512 and qh == 8 * plan.num_kv_heads)):
            n_v = self.value(("-", ("+", "pos", 1), base))
            return P.k_attn_qc(self, layer, q, 0, n_v, q8=q8, t=t,
                               window=plan.sliding_window if self.fdts(layer) else 0)
        out = self.buffer((t, qh * hd))
        for j in range(t):
            p = ("+", "pos", j)
            if plan.is_sliding:
                lo = ("max", 0, ("-", p, plan.sliding_window - 1, base))
            else:
                lo = 0
            lo_v = self.value(lo)
            n_v = self.value(("-", ("+", p, 1), base, lo_v))
            a = P.k_attn_qc(self, layer, q[j:j + 1], lo_v, n_v, q8=q8)
            self.p.emit(P.COPY, a, out[j:j + 1], a.nbytes)
        return out

    def kv_write16(self, layer, k, v, row, qc=1):
        """As k_kv_write, for a cache with the int16 copy only (or the int8
        cache: KV_WRITE8). The float addresses are null, so the kernel does
        not store float rows."""
        plan = self.cfg.plan[layer]
        per = plan.num_kv_heads * plan.head_dim
        kesz = 1 if self.kv == "int8" else 2
        vesz = 1 if self.kv in ("int8", "k16v8") else 2
        sl = lambda name: self.p.slot("%s.%d" % (name, layer))  # noqa: E731
        q = [P._addr(self, sl("kq"), row, kesz * per), P._addr(self, sl("ks"), row, per // 8),
             P._addr(self, sl("vq"), row, vesz * per), P._addr(self, sl("vs"), row, per // 8)]
        op = {"int8": P.KV_WRITE8, "k16v8": P.KV_WRITEV8}.get(self.kv, P.KV_WRITE)
        self.p.emit(op, k, v, 0, 0, *q, k.size)

    def to_cpu(self, *copies, count=None):
        """Hand work to the CPU: copy each (device array, pinned array) to the
        host. Return the event (boundary form) or the flags (flag form: two
        int64 in pinned memory, in and out) for join. count (or None) is the
        device int of the length of the work of the CPU: with 0, the GPU does
        not wait for the CPU (GP_AWAIT)."""
        if self.flags:
            # GP_SIGNAL writes the data to the pinned arrays itself.
            ops = []
            for src, dst in copies:
                ops += [src, dst, dst.nbytes]
            ops += [0, 0, 0] * (3 - len(copies))
            fl = pinned((2,), np.int64)
            self.p.keep.append(fl)
            self.p.emit(P.SIGNAL, fl[0:1], self.p.slot("seq"), *ops)
            self.flag_count[id(fl)] = count
            return fl
        ev = self.n_events
        self.n_events += 1
        ops = []
        for src, dst in copies:
            ops += [src, dst, dst.nbytes]
        ops += [0, 0, 0] * (3 - len(copies))
        self.p.emit(P.TO_HOST, *ops, ev)
        return ev

    def join(self, a):
        cpu, host, ev, part, gpu_part, *started = self.pending.pop(id(a))
        if started and started[0]:
            self.p.emit(P.CPU_WAIT)         # a CPU_START runs it (moe_group_mix)
        elif isinstance(ev, np.ndarray):
            # The flag form: the runner runs the CPU program when the GPU
            # signals, and the GPU waits for its flag.
            seq = self.p.slot("seq")
            self.p.emit(P.CPU_TASK, cpu.buf, ev[0:1], ev[1:2], seq)
            count = self.flag_count.pop(id(ev), None)
            self.p.emit(P.AWAIT, ev[1:2], seq, host, part, part.nbytes,
                        count if count is not None else 0)
            if gpu_part is not None:
                self.p.emit(P.ADD, gpu_part, part, a, a.size)
            return
        else:
            self.p.emit(P.CPU_JOIN, cpu.buf, ev)
        self.p.emit(P.TO_DEV, host, part, part.nbytes)
        if gpu_part is not None:
            self.p.emit(P.ADD, gpu_part, part, a, a.size)

    def moe(self, h, val, idx, layer):
        if h.shape[0] > 1 and self.stage is not None and self.mix:
            return self.moe_group_mix(h, val, idx, layer)
        if h.shape[0] > 1 and self.stage is not None:
            return self.moe_group_gpu(h, val, idx, layer)
        if h.shape[0] > 1 and layer in self.hot_host:
            return self.moe_hot_group(h, val, idx, layer)
        if h.shape[0] > 1:
            return self.moe_group_cpu(h, val, idx, layer)
        if self.hot.get(layer):
            return self.moe_hot(h, val, idx, layer, self.hot[layer])
        hp, vp, ip = pinned(h.shape), pinned(val.shape), pinned(idx.shape, np.int32)
        ev = self.to_cpu((h, hp), (val, vp), (idx, ip))
        cc = P.Compiler(self.model)
        host_out = P.k_moe(cc, hp, vp, ip, layer)
        if self.flags:
            # a copy node of a graph reads pinned memory only
            po = pinned(host_out.shape)
            cc.p.emit(P.COPY, host_out, po, po.nbytes)
            host_out = po
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        dev_out = self.buffer(h.shape)
        self.pending[id(dev_out)] = (cpu, host_out, ev, dev_out, None)
        return dev_out

    def tables(self, layer):
        """Return the tables of the device addresses of the experts of a
        layer (gate and up, down), and the list of the copies of the cold
        experts: (host address, device address, bytes) for each run of
        adjacent cold experts. The arrays are made one time for each layer;
        fill_tables fills them again when the hot experts change."""
        if layer not in self.tables_of:
            n = self.model._layers[layer]["experts.gate_up_proj"][0].shape[0]
            tgu = np.zeros(n, dtype=np.int64)
            tdn = np.zeros(n, dtype=np.int64)
            ranges = np.zeros((2 * n, 3), dtype=np.int64)
            self.tables_of[layer] = (tgu, tdn, ranges)
            self.gslots_of[layer] = np.full(n, -1, dtype=np.int32)
            fill_tables(self.model, layer, self.hot_stores.get(layer), self.stage, tgu, tdn,
                        ranges, self.gslots_of[layer])
        return self.tables_of[layer]

    def fetch(self, layer):
        """Emit the copy of the weights of the cold experts of a layer to the
        device buffer layer % 2."""
        _tgu, _tdn, ranges = self.tables(layer)
        self.p.emit(P.FETCH, ranges, len(ranges), layer, layer % 2)

    def moe_group_gpu(self, h, val, idx, layer):
        """The experts of a large group on the GPU (GP_MOE_GPU). All the
        experts of the layer come to a device buffer with GP_FETCH. The copy
        of layer l + 2 starts when the experts of layer l are done with the
        buffer."""
        w = self.model._layers[layer]
        gu_q, dn_q = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
        inner = self.cfg.moe_intermediate_size
        t, top_k = idx.shape
        pairs = t * top_k
        hidden = h.shape[1]
        tgu, tdn, _ranges = self.tables(layer)
        i32 = lambda n: self.buffer((n,), np.int32)  # noqa: E731
        out = self.buffer(h.shape)
        self.p.emit(P.FETCH_WAIT, layer)
        q4x = 1 if expert_host(self.model, layer)[2] else 0
        self.p.emit(P.MOE_GPU, h, val, idx, t, top_k, q4x, 0, gu_q.shape[1], hidden,
                    dn_q.shape[1], inner, i32(256), i32(257), i32(256), i32(pairs), i32(pairs),
                    i32(1 + 2 * (pairs // 64 + 129)), self.buffer((pairs, gu_q.shape[1])),
                    self.buffer((pairs, inner)), self.buffer((pairs, dn_q.shape[1])), out,
                    tgu, tdn)
        self.p.emit(P.FETCH_DONE, layer % 2)
        if layer + 2 < self.cfg.num_hidden_layers:
            self.fetch(layer + 2)
        return out

    def moe_group_mix(self, h, val, idx, layer):
        """The experts of a large group, split between the GPU and the CPU
        (a mixed group). GP_FETCH copies only some cold experts: the ones
        that ModelGPU.plan_mix predicts, from the routers of the group
        before. gslots marks the experts on the GPU (hot or copied).
        GP_HOT_SPLIT_MT gives the pairs of the other experts to the CPU, and
        the rest to GP_MOE_GPU (-1 for a pair of the CPU). The CPU computes
        its pairs (KQ_MOE of the KQ_Q4X experts, int8 activations, as the
        prompt of the CPU) while the GPU computes its pairs. The output is
        the sum of the two parts, at the first reader (join)."""
        w = self.model._layers[layer]
        gu_q, dn_q = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
        inner = self.cfg.moe_intermediate_size
        t, top_k = idx.shape
        pairs = t * top_k
        hidden = h.shape[1]
        E = gu_q.shape[0]
        tgu, tdn, _ranges = self.tables(layer)
        gslots = self.gslots_of[layer]
        i32 = lambda n: self.buffer((n,), np.int32)  # noqa: E731
        if self.mix_bufs is None:
            from . import cops
            self.mix_bufs = dict(
                hp=pinned((t, hidden)), vp=pinned((t, top_k)), ip=pinned((t, top_k), np.int32),
                xq=np.zeros((t, hidden), np.int8), xs=np.zeros((t, hidden // 32), np.float32),
                xm=np.zeros((t, hidden // 16), np.float32), out=pinned((t, hidden)),
                scratch=cops.kq_moe_scratch(t, top_k, E, hidden, inner))
        b = self.mix_bufs
        cold, cold_val, gidx = i32(pairs), self.buffer((pairs,)), i32(pairs)
        # The split and the copy to the host do not need the copies of the
        # experts, so the CPU part can start before they end.
        self.p.emit(P.HOT_SPLIT_MT, idx, val, gslots, cold, cold_val, pairs,
                    self.p.slot("nreal"), top_k, gidx)
        ev = self.n_events
        self.n_events += 1
        self.p.emit(P.TO_HOST, h, b["hp"], b["hp"].nbytes, cold_val, b["vp"], b["vp"].nbytes,
                    cold, b["ip"], b["ip"].nbytes, ev)
        # The CPU part, on the helper thread (GP_CPU_START).
        cc = P.Compiler(self.model)
        mats = P.ops.q4x_moe_mats(P.ops._Q4X_MOE[gu_q.ctypes.data])
        cc.p.keep.append(mats)
        cc.p.emit(P.KQ_QUANT, b["hp"], t, hidden, b["xq"], b["xs"], b["xm"], b["ip"], top_k)
        cc.p.emit(P.KQ_MOE, b["xq"], b["xs"], b["xm"], b["ip"], b["vp"], t, top_k, E, mats, None,
                  hidden, inner, b["scratch"], b["out"], None, 1, None)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        self.p.emit(P.CPU_START, cpu.buf, ev)
        self.p.emit(P.FETCH_WAIT, layer)
        gpu_part = self.buffer(h.shape)
        self.p.emit(P.MOE_GPU, h, val, gidx, t, top_k, 1, 0, gu_q.shape[1], hidden,
                    dn_q.shape[1], inner, i32(256), i32(257), i32(256), i32(pairs), i32(pairs),
                    i32(1 + 2 * (pairs // 64 + 129)), self.buffer((pairs, gu_q.shape[1])),
                    self.buffer((pairs, inner)), self.buffer((pairs, dn_q.shape[1])), gpu_part,
                    tgu, tdn)
        self.p.emit(P.FETCH_DONE, layer % 2)
        if layer + 2 < self.cfg.num_hidden_layers:
            self.fetch(layer + 2)
        out = self.buffer(h.shape)
        self.pending[id(out)] = (cpu, b["out"], ev, self.buffer(h.shape), gpu_part, True)
        return out

    def moe_hot_group(self, h, val, idx, layer):
        """The experts of a small group (at most MT_CPU tokens) when the GPU
        holds some of them. GP_HOT_SPLIT_MT marks each pair (token, slot):
        the GPU computes the pairs of the hot experts with GP_HOT_MOE, and
        the CPU computes the other pairs with GP_MOE_MT. The output is the
        sum of the two parts. The hot experts are the arrays of the step
        program, so the GPU holds them one time."""
        _hot, gu_store, dn_store, slots = self.hot_host[layer]
        w = self.model._layers[layer]
        dn_rows = w["experts.down_proj"][0].shape[1]
        inner = self.cfg.moe_intermediate_size
        t, top_k = idx.shape
        hidden = h.shape[1]
        cold = self.buffer((t, top_k), np.int32)
        cold_val = self.buffer((t, top_k))
        self.p.emit(P.HOT_SPLIT_MT, idx, val, slots, cold, cold_val, t * top_k)
        hp, vp, ip = pinned(h.shape), pinned((t, top_k)), pinned((t, top_k), np.int32)
        ev = self.n_events
        self.n_events += 1
        self.p.emit(P.TO_HOST, h, hp, h.nbytes, cold_val, vp, vp.nbytes, cold, ip, ip.nbytes, ev)
        gpu_part = self.buffer(h.shape)
        pairs = t * top_k
        self.p.emit(P.HOT_MOE, h, val, idx, slots, gu_store, dn_store,
                    self.buffer((pairs, 2 * inner)), self.buffer((pairs, inner)),
                    self.buffer((pairs, hidden)), gpu_part, top_k, 2 * inner, hidden,
                    dn_rows, inner, t, 1 if expert_host(self.model, layer)[2] else 0)
        cc = P.Compiler(self.model)
        host_out = P.k_moe(cc, hp, vp, ip, layer)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        out = self.buffer(h.shape)
        self.pending[id(out)] = (cpu, host_out, ev, self.buffer(h.shape), gpu_part)
        return out

    def moe_group_cpu(self, h, val, idx, layer):
        """The experts of a group of tokens on the CPU: the MOE_MT record of
        the CPU interpreter on pinned copies of the input."""
        hp, vp, ip = pinned(h.shape), pinned(val.shape), pinned(idx.shape, np.int32)
        ev = self.n_events
        self.n_events += 1
        self.p.emit(P.TO_HOST, h, hp, h.nbytes, val, vp, val.nbytes, idx, ip, idx.nbytes, ev)
        cc = P.Compiler(self.model)
        host_out = P.k_moe(cc, hp, vp, ip, layer)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        dev_out = self.buffer(h.shape)
        self.pending[id(dev_out)] = (cpu, host_out, ev, dev_out, None)
        return dev_out

    def moe_hot(self, h, val, idx, layer, hot):
        """The experts of a layer when the GPU holds some of them.

        GP_HOT_SPLIT writes the selected experts that the GPU does not hold
        (the cold experts), with their weights and their count. GP_TO_HOST
        copies them and the input to the host. GP_HOT_MOE computes the hot
        experts on the GPU, while the CPU computes the cold experts with
        GP_MOE_N. The output is the sum of the two parts.
        """
        w = self.model._layers[layer]
        gu_q, gu_s = w["experts.gate_up_proj"]
        dn_q, dn_s = w["experts.down_proj"]
        inner = self.cfg.moe_intermediate_size
        top_k = idx.size
        hidden = h.shape[1]
        slots = np.full(gu_q.shape[0], -1, dtype=np.int32)
        slots[hot] = np.arange(len(hot), dtype=np.int32)
        hgu, hdn, q4x = expert_host(self.model, layer)
        gu_store = np.ascontiguousarray(hgu[hot])
        dn_store = np.ascontiguousarray(hdn[hot])
        self.hot_stores[layer] = (list(hot), gu_store, dn_store, slots)
        if not q4x:
            for store, scales in ((gu_q[hot], gu_s), (dn_q[hot], dn_s)):
                half = store[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
                if not np.array_equal(half, scales[hot]):
                    raise ValueError("an expert has float32 scales that are not its float16 "
                                     "scales")
        # The cold experts, their count, and the selection of the step.
        cold = np.zeros(2 * top_k + 1, dtype=np.int32)
        cold_val = self.buffer(top_k)
        self.p.emit(P.HOT_SPLIT, idx, val, slots, cold, cold_val, top_k, 1)
        hp, vp, ip = pinned(h.shape), pinned((top_k,)), pinned((2 * top_k + 1,), np.int32)
        self.hot_ip[layer] = ip
        ev = self.to_cpu((h, hp), (cold_val, vp), (cold, ip), count=cold[top_k:top_k + 1])
        gpu_part = self.buffer(h.shape)
        self.p.emit(P.HOT_MOE, h, val, idx, slots, gu_store, dn_store,
                    self.buffer((top_k, 2 * inner)), self.buffer((top_k, inner)),
                    self.buffer((top_k, hidden)), gpu_part, top_k, 2 * inner, hidden,
                    dn_q.shape[1], inner, 1, 1 if q4x else 0)
        cc = P.Compiler(self.model)
        host_out = pinned(h.shape) if self.flags else np.zeros(h.shape, dtype=np.float32)
        q4x = P.ops._Q4X_MOE.get(gu_q.ctypes.data)
        if q4x is not None:
            # The cold experts in groups of 16 rows (KQ_Q4X), float32
            # activations: GP_KQ_MOE with the count in memory (cold_idx[top_k])
            # gives the products of GP_MOE_N to 4e-6, in less time.
            from . import cops
            E = gu_q.shape[0]
            mats = P.ops.q4x_moe_mats(q4x)
            cc.p.keep.append(mats)
            cc.p.emit(P.KQ_MOE, np.zeros((1, hidden), np.int8), np.zeros((1, hidden // 32), np.float32),
                      np.zeros((1, hidden // 16), np.float32), ip, vp, 1, top_k, E, mats, None,
                      hidden, inner, cops.kq_moe_scratch(1, top_k, E, hidden, inner), host_out,
                      ip[top_k:], 3, hp)
        else:
            cc.p.emit(P.MOE_N, hp, vp, ip, ip[top_k:], gu_q, gu_s, dn_q, dn_s, gu_q.shape[1],
                      hidden, dn_q.shape[1], inner, np.zeros(top_k, dtype=np.int32),
                      np.zeros((top_k, 2 * inner), np.float32), np.zeros((top_k, inner), np.float32),
                      np.zeros((top_k, dn_q.shape[1]), np.float32), host_out)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        out = self.buffer(h.shape)
        self.pending[id(out)] = (cpu, host_out, ev, self.buffer(h.shape), gpu_part)
        return out


def fill_tables(model, layer, hot, stage, tgu, tdn, ranges, gslots=None, pre=None):
    """Fill the tables of a large group for one layer (SplitCompiler.tables).

    hot is (slots, device address of the hot gate and up blocks, of the hot
    down blocks), or None. stage is the two device buffers of the cold
    experts. tgu and tdn get the device address of each expert. ranges gets
    the copies of the cold experts to buffer layer % 2: the runs of the gate
    and up blocks, then those of the down blocks. The count of ranges is
    fixed (twice the experts), so a program keeps it; the other rows copy
    nothing.

    pre (or None for all) is the set of the cold experts to copy (a mixed
    group). The other cold experts get no place: address 0, and -1 in
    gslots. gslots (or None) gets 0 or more for the experts on the GPU."""
    gu, dn, _q4x = expert_host(model, layer)
    n = gu.shape[0]
    egu, edn = gu.nbytes // n, dn.nbytes // n
    dgu, ddn = stage[layer % 2]
    slots, hgu, hdn = hot if hot is not None else (np.full(n, -1), 0, 0)
    runs = []
    rank = 0
    if gslots is not None:
        gslots[:] = -1
    for x in range(n):
        if slots[x] >= 0:
            tgu[x] = hgu + int(slots[x]) * egu
            tdn[x] = hdn + int(slots[x]) * edn
            if gslots is not None:
                gslots[x] = slots[x]
            continue
        if pre is not None and x not in pre:
            tgu[x] = tdn[x] = 0
            continue
        if gslots is not None:
            gslots[x] = rank
        tgu[x] = dgu.ptr + rank * egu
        tdn[x] = ddn.ptr + rank * edn
        if runs and runs[-1][3] == x - 1:
            runs[-1][2] += egu
            runs[-1][6] += edn
            runs[-1][3] = x
        else:
            runs.append([gu.ctypes.data + x * egu, int(tgu[x]), egu, x,
                         dn.ctypes.data + x * edn, int(tdn[x]), edn])
        rank += 1
    ranges[:] = 0
    for k, r in enumerate(runs):
        ranges[k] = r[0:3]
        ranges[len(runs) + k] = r[4:7]


def pick_hot(model, counts, budget):
    """Return the experts that the GPU holds, as a dict layer -> the sorted
    expert indices. counts has shape (layers, experts): the selections of each
    expert on a text (scripts/expert_use.py). The set takes the most used
    experts over all layers, up to budget bytes."""
    w = model._layers[0]
    per = (w["experts.gate_up_proj"][0][0].nbytes + w["experts.down_proj"][0][0].nbytes)
    n = int(budget // per)
    flat = np.asarray(counts).reshape(-1)
    order = np.argsort(flat, kind="stable")[::-1][:n]
    order = order[flat[order] > 0]
    e = counts.shape[1]
    hot = {}
    for k in order:
        hot.setdefault(int(k // e), []).append(int(k % e))
    return {layer: sorted(v) for layer, v in hot.items()}


# The counts of the experts of the 26B QAT model on 800 tokens of each of four
# texts. The texts are the README, Python code, C code, and notes in English
# (SPLIT_PLAN.md).
HOT_COUNTS = _HERE / "data" / "gemma-4-26B-expert-counts.npz"


def hot_counts(model):
    """Return the counts of the experts for the hot experts, or None."""
    path = os.environ.get("NP_GEMMA_GPU_HOT")
    if path == "0":
        return None
    f = np.load(path or HOT_COUNTS)
    counts = sum(f[k] for k in f.files)
    shape = (model.cfg.num_hidden_layers, model.cfg.num_experts)
    if counts.shape != shape:
        if path:
            raise ValueError("the counts of %s have the shape %s, not %s"
                             % (path, counts.shape, shape))
        return None
    return counts


# The tokens of a chunk of a prompt pass on the GPU. Each chunk copies the
# weights of the cold experts to the GPU. For the 26B that is up to 11 GB,
# about 1.6 s over PCIe 3 x8 (6.7 GB/s). A long chunk pays for the copy with
# more tokens. With 2048 the copy is hidden behind the other work of the
# layers even with 2 GB of hot experts (the wait is 0.3% of a group); with
# 1024 the wait is 32% of a group at 2 GB.
CHUNK = int(os.environ.get("NP_GEMMA_GPU_CHUNK", "2048"))
# The tokens of an image see each other in every layer on the GPU, as on the
# CPU (np_gemma.model._BIDIR_ALL); NP_GEMMA_BIDIR_ALL=0 is not on the GPU yet.
_BIDIR_ALL_OK = os.environ.get("NP_GEMMA_BIDIR_ALL", "1") == "1"
# The largest group whose experts run on the CPU (GP_MOE_MT of the CPU
# interpreter takes at most 16 tokens).
MT_CPU = 16
# The record of a group of up to FDT_TMAX queries for a global layer of the
# 26B (attn_rows_small; FT_TMAX of csrc/gpu.cu); NP_GEMMA_GPU_FDT_TC=0 (the
# old kernel, one query a record) one record for each query.
FDT_TMAX = 3
FDT_TC = os.environ.get("NP_GEMMA_GPU_FDT_TC", "1") != "0"
# the sliding layers (SplitCompiler.fdts, k_attn_fdts); NP_GEMMA_GPU_FDT_TS=0
# the record of a step for each query, k_attn_fdt (a test)
FDT_TS = FDT_TC and os.environ.get("NP_GEMMA_GPU_FDT_TS", "1") != "0"
# The shortest part of a prompt that runs with the experts on the GPU. The
# copy of their weights costs about 1.9 s, and the CPU takes about as long for
# 128 tokens.
PREFILL_MIN = int(os.environ.get("NP_GEMMA_GPU_PREFILL_MIN", "128"))
# The mixed groups of a prompt of the 26B (SplitCompiler.moe_group_mix):
# each layer copies some of its cold experts (the most used in the group
# before), and the CPU computes the pairs of the other cold experts.
# NP_GEMMA_GPU_MIX=0 turns the mixed groups off. NP_GEMMA_GPU_MIX_PRE gives
# the count of copied experts of each layer (-1: all); the default (auto)
# takes the count of the least time in the model of ModelGPU.plan_mix.
MIX = os.environ.get("NP_GEMMA_GPU_MIX", "1") != "0"   # (the Qwen models read a size)
# The handoff to the CPU of a decode step inside its graph (SplitCompiler.
# to_cpu and join, GP_SIGNAL and GP_AWAIT): the step is one graph, and the
# runner runs the CPU parts as the GPU signals them. 0 keeps the boundary
# records (GP_TO_HOST, GP_CPU_JOIN, GP_TO_DEV).
FLAGS = os.environ.get("NP_GEMMA_GPU_FLAGS", "1") == "1"
# The fused forms of the step of the 26B (program.layer_form, fused): the
# norms and the adds of the end of the attention and of the end of each
# layer in two kernels (GP_ADD_NORM, GP_FFN_OUT) in place of about ten.
FUSED = os.environ.get("NP_GEMMA_GPU_FUSED", "1") == "1"
# 1: a small group (an MTP verify group of the 26B) takes the fused forms
# too (compile_split_group). 0 keeps the separate norms, for a test.
GROUP_FUSED = os.environ.get("NP_GEMMA_GPU_GROUP_FUSED", "1") == "1"
# The fused dense layer (program._dense_fused_layer) for the large groups of
# a dense model too (a prompt pass of the 12B): fewer kernels for the norms,
# the adds, and the int8 x of the products.
PREFILL_FUSED = os.environ.get("NP_GEMMA_GPU_PREFILL_FUSED", "1") == "1"
# The rows of the tokens of a large group (a prompt pass) from the Q4_0 token
# table on the GPU (ModelGPU._embed_gpu), not from Model.embed and a copy of
# the rows: about 30 ms for each chunk of 2048 tokens of the 12B.
EMBED_GPU = os.environ.get("NP_GEMMA_GPU_EMBED", "1") == "1"
MIX_PRE = os.environ.get("NP_GEMMA_GPU_MIX_PRE", "auto")
# The model of the time of a layer of a mixed group, in ms: the other GPU
# work of the layer for each token (gpu_tok; with the copies of the CPU
# part to the host and back), the GPU experts for each pair (gpu_pair), the
# copy rate in bytes for each ms (rate), and the CPU part: cpu_a for each
# layer, cpu_e for each expert (the read of its weights), and cpu_b for each
# pair (KQ_Q4X, int8). The first values are those
# of an RTX 5060 Ti on PCIe 3 x8 with 18 threads; each mixed group measures
# them again (ModelGPU.calibrate_mix), so the plan follows the machine and
# its other load. NP_GEMMA_GPU_MIX_CAL=0 keeps the first values.
MIX_START = dict(gpu_tok=0.007, gpu_pair=0.65e-3, rate=3.3e6, cpu_a=1.0, cpu_e=0.06,
                 cpu_b=4e-3)
MIX_CAL = os.environ.get("NP_GEMMA_GPU_MIX_CAL", "1") == "1"
MIX_LOG = os.environ.get("NP_GEMMA_GPU_MIX_LOG", "0") == "1"


def compile_split_group(model, t, hot=None, kv="int16", stage=None, hot_stores=None,
                        hot_host=None, counts=None, mix=False):
    """Compile a step of t tokens for the GPU: a chunk of a prompt, or the
    group of an MTP verify step. The form is the group form of the layers.

    A group of more than MT_CPU tokens computes the experts on the GPU, with
    the weights in the two device buffers of stage. A smaller group sends
    them to the CPU, as a step does."""
    assert kv in ("int16", "int8", "k16v8"), "a group on the GPU reads a quantized cache"
    gpu_experts = model.cfg.enable_moe_block and t > MT_CPU and stage is not None
    # A small group (an MTP verify group) takes the fused forms of the step
    # (add_norm2, ffn_out; GP_ADD_NORM and GP_FFN_OUT take rows), so it runs
    # the norms of the step and fewer kernels. It keeps its buffers, as the
    # step does: the fused layer reads the norm "hn" of the layer before.
    small = t <= MT_CPU and not gpu_experts and FUSED and GROUP_FUSED
    big_dense = (t > MT_CPU and not model.cfg.enable_moe_block and FUSED and GROUP_FUSED
                 and PREFILL_FUSED)
    c = SplitCompiler(model, hot, kv, pool=not small, stage=stage if gpu_experts else None)
    c.fused = small or big_dense
    # int8 x for the dense Q4_0 matrices of a small group, as the step (its
    # own switch, not that of the fused forms).
    c.q4_kq = t <= MT_CPU and not gpu_experts and q4_i8_dense(model)
    c.hot_stores = hot_stores or {}
    c.hot_host = hot_host or {}
    c.counts = counts
    c.mix = mix and gpu_experts and all(
        P.ops._Q4X_MOE.get(model._layers[i]["experts.gate_up_proj"][0].ctypes.data) is not None
        for i in range(model.cfg.num_hidden_layers))
    c.env["x"] = np.zeros((t, model.cfg.hidden_size), dtype=np.float32)
    # The last key of each query less pos (the attention records): row j sees
    # up to its position unless it is in an image (ModelGPU.group). A group of
    # MT_CPU tokens or fewer (an MTP verify group) runs the attention of a
    # step for each query, with no such record, and a prompt puts each image
    # in a larger group (ModelGPU._prefill).
    if t > MT_CPU:
        c.env["lim"] = np.arange(t, dtype=np.int32)
    c.p.slot("pos")
    if gpu_experts:
        c.fetch(0)
        c.fetch(1)
    mode = {"int8": "q8", "k16v8": "qv"}.get(kv, "qc")
    c.compile(P.step_form(model, mode, t, fused=c.fused))
    assert not c.pending, "an output of the experts has no reader"
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = mode
    c.p.tokens = t
    c.p.cpu_progs = c.cpu_progs
    c.p.tables_of = c.tables_of
    c.p.gslots_of = c.gslots_of
    c.p.mix = c.mix
    return _fuse_kq(c.p.finish()) if c.q4_kq else c.p.finish()


def compile_split_step(model, hot=None, kv="int16"):
    """Compile a step of one token of the 26B model (or of a dense model) for
    the GPU. "x" is the input and "xn" the result. hot gives the experts that
    the GPU holds (see pick_hot). kv is the form of the cache on the GPU:
    "int16" (the form of the CPU program, a scale for each group of 32
    values) or "float"."""
    c = SplitCompiler(model, hot, kv)
    c.flags = FLAGS
    c.fused = FUSED
    c.q4_kq = q4_i8_dense(model)
    c.env["x"] = np.zeros((1, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    mode = {"int16": "qc", "int8": "q8", "k16v8": "qv"}.get(kv, "f32")
    c.compile(P.step_form(model, mode, 1, fused=FUSED))
    assert not c.pending, "an output of the experts has no reader"
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = mode
    c.p.tokens = 1
    c.p.cpu_progs = c.cpu_progs
    c.p.hot_stores = c.hot_stores
    c.p.hot_ip = c.hot_ip
    return _fuse_kq(c.p.finish()) if c.q4_kq else c.p.finish()


class GPUKV(DeviceCache):
    """The cache of the 26B model on the GPU (gpumm.DeviceCache: locked; the
    rows past the positions in use are lent to GpuMem).

    Each layer has buffers of (rows, kv_heads, head_dim) values. Row 0 has the
    position base, as in KVCache. The form "int16" keeps the int16 copy of
    KVCache only: kq and vq, and ks and vs, one float32 scale for each group
    of 32 values. That is about half of the float form. The form "float"
    keeps k and v.

    A layer with a window drops its oldest rows when the buffer holds more
    than two windows, as KVCache.prepare does, with a copy on the GPU.
    attach() copies the rows of a KVCache to the GPU. detach() writes the rows
    that only the GPU has into the KVCache with KVCache.write. With the form
    int16, those rows come back as the int16 values times their scales.
    """

    def __init__(self, cfg, kv="int16", max_chunk=512):
        super().__init__("cache0")
        self.cfg = cfg
        self.form = kv
        self.max_chunk = max_chunk
        self.window = cfg.sliding_window or 0
        n = cfg.num_hidden_layers
        self.names = ("kq", "ks", "vq", "vs") if kv != "float" else ("k", "v")
        # the bytes of a value of kq and of vq
        self.kesz = 1 if kv == "int8" else 2
        self.vesz = 1 if kv in ("int8", "k16v8") else 2
        self.bufs = [dict() for _ in range(n)]
        self.cap = [0] * n
        self.base = [0] * n
        self.end = [0] * n
        self.host_end = [0] * n

    def _row(self, i, name):
        """Return the bytes of one row of the buffer name of layer i."""
        plan = self.cfg.plan[i]
        per = plan.num_kv_heads * plan.head_dim
        return {"k": 4 * per, "v": 4 * per, "kq": self.kesz * per, "vq": self.vesz * per,
                "ks": per // 8, "vs": per // 8}[name]

    def nbytes(self):
        return sum(b.nbytes for d in self.bufs for b in d.values())

    def blocks(self):
        return [b for d in self.bufs for b in d.values()]

    def _alloc(self, i, cap, keep_rows=0):
        """Give layer i buffers of cap rows. Keep the first keep_rows rows.
        When the memory runs out, self.oom frees programs and it tries again;
        else the new buffers go and MemoryError comes up."""
        new = {}
        if self.bufs[i]:
            self.unshare()      # the old buffers go: the memory they lent first
        try:
            for name in self.names:
                rb = self._row(i, name)
                new[name] = self._new(cap * rb, rows=cap, base=lambda i=i: self.base[i])
                if keep_rows:
                    _check(lib().gg_d2d(new[name].ptr, self.bufs[i][name].ptr, keep_rows * rb))
        except MemoryError:
            for b in new.values():
                self._free(b)
            raise
        if self.bufs[i]:
            lib().gg_sync()
            for b in self.bufs[i].values():
                self._free(b)
        self.bufs[i], self.cap[i] = new, cap

    def _host_rows(self, cache, i, rows):
        """Return the host rows of layer i in the form of the GPU."""
        if self.form == "float":
            # the host keeps quantized rows only: their float values
            k, v, _ = cache.read(i, cache.base[i] + rows)
            return {"k": k, "v": v}
        if getattr(cache, "kv", "int16") != self.form:
            raise ValueError("the GPU cache is %s and the host cache %s"
                             % (self.form, getattr(cache, "kv", "int16")))
        return {"kq": cache.kq[i][:rows], "ks": cache.ks[i][:rows],
                "vq": cache.vq[i][:rows], "vs": cache.vs[i][:rows]}

    def attach(self, cache, max_len):
        self.unshare()          # the rows of this cache go where the last lent memory
        for i in range(self.cfg.num_hidden_layers):
            rows = cache.end[i] - cache.base[i]
            plan = self.cfg.plan[i]
            cap = ((2 * self.window + KV_KEEP + self.max_chunk + 64) if plan.is_sliding
                   else max(max_len, rows + 64))
            cap = max(cap, rows + 64)
            if not self.bufs[i] or self.cap[i] < cap:
                self._alloc(i, cap)
            self.base[i], self.end[i] = cache.base[i], cache.end[i]
            self.host_end[i] = cache.end[i]
            if rows > 0:
                for name, a in self._host_rows(cache, i, rows).items():
                    self.bufs[i][name].upload(np.ascontiguousarray(a))

    def prepare(self, i, pos, t=1):
        """Make room for the rows of positions pos to pos + t - 1 in layer i."""
        if self.cfg.plan[i].is_sliding:
            w = self.window
            if pos - self.base[i] > 2 * w + KV_KEEP:
                keep = pos - w + 1 - KV_KEEP
                off = keep - self.base[i]
                rows = self.end[i] - keep
                if rows > 0:
                    for name in self.names:
                        rb = self._row(i, name)
                        b = self.bufs[i][name]
                        _check(lib().gg_d2d(b.ptr, b.ptr + off * rb, rows * rb))
                self.base[i] = keep
                # The host may lack some rows that the GPU dropped; only the
                # rows of the window matter to a later step, and detach starts
                # at base. host_end stays the end of the host rows: sync()
                # reads a host end below it as a truncate, and it then cut end
                # (a prompt pass on the GPU past two windows, or a long decode)
                # while the next drop moved base with no rows: the rows of the
                # window no longer matched their positions.
        need = pos + t - self.base[i]
        if need > self.cap[i]:
            self._alloc(i, max(need + 64, 2 * self.cap[i]), self.end[i] - self.base[i])
        self.end[i] = max(self.end[i], pos + t)

    def detach(self, cache, layers=None):
        """Write the rows that the GPU made into the host cache: of every
        layer, or of the layers in the list layers."""
        for i in (range(self.cfg.num_hidden_layers) if layers is None else layers):
            start = max(self.host_end[i], cache.end[i], self.base[i])
            n = self.end[i] - start
            if n <= 0:
                continue
            plan = self.cfg.plan[i]
            shape = (n, plan.num_kv_heads, plan.head_dim)
            # The int16 rows go straight into the host cache when it keeps
            # an int16 copy of them (KVCache.rows_q).
            views = (cache.rows_q(i, start, n) if self.form != "float"
                     and hasattr(cache, "rows_q") else None)
            got = {}
            for j, name in enumerate(self.names):
                rb = self._row(i, name)
                if views is not None:
                    a = views[j]
                    assert a.flags.c_contiguous and a.nbytes == n * rb
                else:
                    dt = {"kq": np.int8 if self.kesz == 1 else np.int16,
                          "vq": np.int8 if self.vesz == 1 else np.int16}.get(name, np.float32)
                    a = np.empty(n * rb // np.dtype(dt).itemsize, dtype=dt)
                _check(lib().gg_d2h(a.ctypes.data,
                                    self.bufs[i][name].ptr + (start - self.base[i]) * rb, a.nbytes))
                got[name] = a
            if views is not None:
                cache.rows_q_done(i, start, n)
            elif self.form == "float":
                cache.write(i, start, got["k"].reshape(shape), got["v"].reshape(shape))
            elif hasattr(cache, "write_q"):
                sshape = shape[:2] + (shape[2] // 32,)
                cache.write_q(i, start, got["kq"].reshape(shape), got["ks"].reshape(sshape),
                              got["vq"].reshape(shape), got["vs"].reshape(sshape))
            else:
                k = (got["kq"].reshape(-1, 32) * got["ks"][:, None]).reshape(shape)
                v = (got["vq"].reshape(-1, 32) * got["vs"][:, None]).reshape(shape)
                cache.write(i, start, k, v)
            self.host_end[i] = self.end[i]

    def window_get(self, i, start, n):
        """The rows of positions start .. start + n - 1 of layer i (host
        arrays by name), or None when the GPU does not hold them all
        (Model.window_snapshot)."""
        if start < self.base[i] or start + n > self.end[i]:
            return None
        plan = self.cfg.plan[i]
        out = {}
        for name in self.names:
            rb = self._row(i, name)
            dt = np.float32 if name in ("ks", "vs", "k", "v") else (np.int8 if rb == plan.num_kv_heads * plan.head_dim else np.int16)
            a = np.empty(n * rb // np.dtype(dt).itemsize, dt)
            if n:
                _check(lib().gg_d2h(a.ctypes.data, self.bufs[i][name].ptr + (start - self.base[i]) * rb,
                                    n * rb))
            out[name] = a
        return out

    def window_put(self, i, rows, start, end):
        """Layer i holds the rows of positions start .. end - 1 (rows of
        window_get) from buffer row 0: base start, end end
        (Model.window_restore)."""
        n = end - start
        if self.cap[i] < n:
            self._alloc(i, max(n + 64, self.cap[i]))
        for name in self.names:
            a = np.ascontiguousarray(rows[name])
            _check(lib().gg_h2d(self.bufs[i][name].ptr, a.ctypes.data, a.nbytes))
        self.base[i], self.end[i] = start, end
        self.host_end[i] = end

    def truncate(self, n):
        """Cut the rows back to n positions, as KVCache.truncate does. Return
        False, with no change, when a layer with a window dropped a row that
        a token at n sees (the rows before base are gone on the GPU)."""
        w = self.window
        for i in range(self.cfg.num_hidden_layers):
            first = max(0, n - w + 1) if (self.cfg.plan[i].is_sliding and w) else n
            if self.base[i] > first:
                return False
        for i in range(self.cfg.num_hidden_layers):
            self.end[i] = min(self.end[i], n)
            self.host_end[i] = min(self.host_end[i], n)
        return True

    def sync(self, cache):
        """Follow a truncate of the host cache (KVCache.truncate). A row
        after the end of the host is not a row of the history any more."""
        for i in range(self.cfg.num_hidden_layers):
            if cache.end[i] < self.host_end[i]:
                self.host_end[i] = cache.end[i]
                self.end[i] = min(self.end[i], cache.end[i])

    def params(self):
        kw = {}
        for i in range(self.cfg.num_hidden_layers):
            kw["base.%d" % i] = self.base[i]
            for name in self.names:
                kw["%s.%d" % (name, i)] = self.bufs[i][name].ptr
        return kw


class ModelGPU(ProgramLRU):
    """A decode step of the 26B model on the GPU, with the experts on the CPU.
    A dense model (the 12B) runs wholly on the GPU if it fits. See the module
    text and GPUKV.

        g = gpu.ModelGPU(model)
        g.attach(cache)                  # after the prompt pass on the CPU
        xn = g.step([token], pos)        # the hidden state after the final norm
        logits = g.logits()              # the head of the last step
        g.detach(cache)                  # the host cache has the new rows
    """

    def __init__(self, model, graph=True, hot=None, kv=None):
        """hot is a dict layer -> experts (see pick_hot), or None.

        Without hot, a model with experts gets the most used experts of a
        file of counts (scripts/expert_use.py). The file is NP_GEMMA_GPU_HOT,
        or else HOT_COUNTS when its shape agrees with the model.
        NP_GEMMA_GPU_HOT=0 turns the hot experts off. NP_GEMMA_GPU_HOT_GB
        gives the budget in GB. The default budget is the free memory of the
        GPU less 6 GB: the step, its head, and a cache of 4096 tokens take
        about 2.7 GB, a prompt pass takes about 1.3 GB more, and the display
        needs some memory too.
        """
        self.model = model
        if hot is None and model.cfg.enable_moe_block:
            counts = hot_counts(model)
            if counts is not None:
                gb = os.environ.get("NP_GEMMA_GPU_HOT_GB")
                budget = float(gb) * 1e9 if gb else max(0.0, mem_info()[0] - 6.0e9)
                hot = pick_hot(model, counts, budget)
        self.hot = hot or {}
        from . import model as _model_mod
        kv = kv or (_model_mod.KV_FORM if _model_mod.KV_FORM != "int16"
                    else os.environ.get("NP_GEMMA_GPU_KV", "int16"))
        kv = _model_mod.kv_base(kv)     # (rq8, k16vr8: the step form rotates; model.kv_rot)
        self.kv_form = kv
        self.graph = graph
        self.prog = compile_split_step(model, self.hot, kv)
        self.g = GPUProgram(self.prog, graph=graph)
        self.kv = GPUKV(model.cfg, kv, max_chunk=CHUNK)
        self.groups = {}      # t -> (Program, GPUProgram) of a group of t tokens
        self._lru_init()
        self.kv.oom = self._evict
        self.head = None
        self.rows = 1
        self.max_len = 4096
        # The device address of the hidden state of the last row of the last
        # step or group, for the output head.
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        # The hot experts follow the text (HotCache). NP_GEMMA_GPU_HOT_DYN=0
        # keeps the first set.
        self.hot_cache = None
        if self.prog.hot_stores and os.environ.get("NP_GEMMA_GPU_HOT_DYN", "1") != "0":
            self.hot_cache = HotCache(self)
        self._mm_freeze()       # GpuMem: the weights, the stores, the step stay
        self._run_blocks = self.kv.blocks

    def attach(self, cache):
        self.kv.attach(cache, max(self.max_len, cache.max_len))

    def detach(self, cache):
        self.kv.detach(cache)

    def _params(self, pos, t):
        """Prepare the cache for t rows from pos. Return the parameters."""
        model, cfg = self.model, self.model.cfg
        for i in range(cfg.num_hidden_layers):
            self.kv.prepare(i, pos, t)
        self.kv.share_tail(pos + t)             # (GpuMem: the rows past the run are lent)
        kw = {"pos": pos}
        kw.update(self.kv.params())
        positions = np.arange(pos, pos + t)
        for kind, sliding in (("s", True), ("f", False)):
            plan = next((p for p in cfg.plan if p.is_sliding == sliding), None)
            if plan is None:
                continue
            cos, sin, _ca, _sa = model._rope(plan, positions)
            kw["cos." + kind] = np.ascontiguousarray(cos, dtype=np.float32)
            kw["sin." + kind] = np.ascontiguousarray(sin, dtype=np.float32)
        # The attention of a large group needs no scores buffer on the GPU.
        # A small group runs the attention of a decode step for each query.
        need = max(p.num_q_heads * (pos + t) for p in cfg.plan) if t <= MT_CPU else 16
        kw["scores"] = np.empty(need, dtype=np.float32)
        return kw

    def step(self, tokens, pos):
        """Run a step of one token. Return the hidden state after the final
        norm, shape (1, hidden)."""
        assert len(tokens) == 1, "step runs one token; group runs more"
        if self.hot_cache is not None:
            self.hot_cache.prepare()
        kw = self._params(pos, 1)
        self.prog.names["x"][:] = self.model.embed(tokens)
        self.g.upload("x")
        self.g.bind(kw)
        self.g.run()
        self.g.download("xn")
        if self.hot_cache is not None:
            # logits() runs it while the GPU runs the head, or else the next
            # prepare().
            self.hot_cache.due = True
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.rows = 1
        return self.prog.names["xn"].copy()

    def _hot_devices(self):
        """Return layer -> (slots, device address of the gate and up blocks
        of the hot experts, of their down blocks), from the step program."""
        out = {}
        for layer, (_hot, gu, dn, slots) in self.prog.hot_stores.items():
            out[layer] = (slots, self.g.mirror.buffer_of(gu).ptr, self.g.mirror.buffer_of(dn).ptr)
        return out

    def _stage(self):
        """Return the two device buffers for the weights of the cold experts
        of a layer, for a large group. The GPU holds the hot experts already,
        so a buffer holds only the cold experts of the layer with the most."""
        if getattr(self, "stage", None) is None:
            w = self.model._layers[0]
            gu, dn = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
            n = gu.shape[0]
            cold = max(n - len(self.hot.get(i, ())) for i in range(self.model.cfg.num_hidden_layers))
            self.stage = [(Buffer(cold * gu.nbytes // n), Buffer(cold * dn.nbytes // n))
                          for _ in range(2)]
        return self.stage

    def _group(self, t):
        e = self.groups.get(t)
        if e is None:
            stage = self._stage() if (t > MT_CPU and self.model.cfg.enable_moe_block) else None
            prog = compile_split_group(self.model, t, kv=self.kv_form, stage=stage,
                                       hot_stores=self._hot_devices() if stage else None,
                                       hot_host=self.prog.hot_stores,
                                       counts=self.hot_cache.counts if self.hot_cache else None,
                                       mix=MIX)
            # The tensor cores round the input of a product to float16. The
            # router of the experts then selects another expert more often:
            # 93% of the top tokens agree with the CPU, against 98% with the
            # float32 products. The prompt pass of the 26B waits for the copy
            # of the experts anyway, so it keeps float32 products.
            tc = os.environ.get("NP_GEMMA_GPU_TC_MOE", "0") == "1" or \
                not self.model.cfg.enable_moe_block
            # The int4 dense products take int8 x all the same (k_gemm_q8): an
            # int8 block has its own scale, so it has no overflow. On a chat
            # text, 98.9% of the top tokens agree with float32 (KL 0.002), and
            # the prompt pass is 1.6 times as fast (README). With the int16
            # prompt of the CPU (Model.prompt_act "16", the default of the
            # 26B) they and the KQ_Q4X experts take the int16 form: x as two
            # int8 planes, q = 128 hi + lo, and two int8 products of the
            # tensor cores for each block (k_quant_x2).
            i8 = os.environ.get("NP_GEMMA_GPU_I8_MOE", "1") == "1"
            if i8 and self.model.prompt_act == "16":
                i8 = 2
            # The attention runs on the tensor cores all the same
            # (k_flash_qc_tc): the queries and the keys have a norm, so
            # float16 has no overflow there. It is ten times as fast as the
            # float32 kernel, and on a chat text the KL to float32 goes from
            # 0.0020 to 0.0024.
            atc = os.environ.get("NP_GEMMA_GPU_ATTN_TC", "1") == "1"
            e = self.groups[t] = (prog, self._build_program(prog, t, tc=tc, i8=i8, atc=atc))
        self._used(t)
        return e

    def group(self, tokens, pos, size=None, media=None):
        """Run a group of tokens from position pos. Return the hidden states
        after the final norm, shape (len(tokens), hidden).

        size pads the group to a program of that many tokens. The padding
        rows write cache rows after the group. The cache then forgets them:
        a later step writes those rows again.

        media is a list of media.Span (absolute positions): their rows take
        the place of the token rows, and the tokens of a span with bidir see
        each other (the whole span must be in the group).
        """
        t = len(tokens)
        size = size or t
        prog, g = self._group(size)
        if self.hot_cache is not None:
            # A large group copies the cold experts to buffers of a fixed
            # size, so it needs every slot full: wait for the copies.
            self.hot_cache.prepare(wait=size > MT_CPU)
            if size > MT_CPU and not getattr(prog, "mix", False):
                self.hot_cache.refresh(prog, g)
        mixed = size > MT_CPU and getattr(prog, "mix", False)
        if mixed:
            self.plan_mix(prog, g)
            if MIX_CAL:
                _check(lib().gg_mix_stats_on(1))
        kw = self._params(pos, size)
        kw["nreal"] = t
        x = prog.names["x"]
        on_gpu = size > MT_CPU and not media and self._embed_gpu(g, prog, tokens, size)
        if not on_gpu:
            x[:t] = self.model.embed(tokens)
            x[t:] = 0.0
        lim = prog.names.get("lim")
        if lim is not None:
            lim[:] = np.arange(lim.shape[0], dtype=np.int32)
        for sp in media or ():
            lo, hi = max(sp.start, pos), min(sp.end, pos + t)
            if lo >= hi:
                continue
            x[lo - pos:hi - pos] = sp.rows[lo - sp.start:hi - sp.start]
            if sp.bidir and lim is not None and _BIDIR_ALL_OK:
                if sp.start < pos or sp.end > pos + t:
                    raise ValueError("a group splits the image at %d to %d" % (sp.start, sp.end))
                lim[lo - pos:hi - pos] = sp.end - 1 - pos
        if lim is not None:
            g.upload("lim")
        if not on_gpu:
            g.upload("x")
        g.bind(kw)
        t0 = time.perf_counter()
        g.run()
        g.download("xn")
        wall = time.perf_counter() - t0
        if self.hot_cache is not None:
            self.hot_cache.seed(g, t, prompt=getattr(self, "in_prompt", False))
        if mixed and MIX_CAL:
            self.calibrate_mix(prog, size, wall)
            _check(lib().gg_mix_stats_on(0))
        for i in range(self.model.cfg.num_hidden_layers):
            self.kv.end[i] = min(self.kv.end[i], pos + t)
        hidden = self.model.cfg.hidden_size
        self.first = g.mirror.buffer_of(prog.names["xn"]).ptr
        self.last = self.first + (t - 1) * hidden * 4
        self.rows = t
        return prog.names["xn"][:t].copy()

    def plan_mix(self, prog, g):
        """Choose the cold experts that each layer of a mixed group copies to
        the GPU: the MIX_PRE cold experts with the most selections in the
        group before (HotCache.last), or else in the file of counts. The
        CPU takes the other cold experts that the routers select. Fill the
        tables and the marks of the experts on the GPU, and upload them."""
        last = self.hot_cache.last if self.hot_cache is not None else None
        if last is None or not last.any():
            if getattr(self, "_static_counts", None) is None:
                c = hot_counts(self.model)
                self._static_counts = c if c is not None else np.zeros(
                    (self.model.cfg.num_hidden_layers, self.model.cfg.num_experts))
            last = self._static_counts
        hot = self._hot_devices()
        t = prog.tokens
        cfg = self.model.cfg
        cal = self.mix_cal
        per = sum(a.nbytes for a in expert_host(self.model, 0)[:2]) / cfg.num_experts
        a_ms = cal["gpu_tok"] * t
        self.mix_plan = []
        for layer, (tgu, tdn, ranges) in prog.tables_of.items():
            h = hot.get(layer)
            slots = h[0] if h is not None else np.full(tgu.shape[0], -1)
            pred = np.asarray(last[layer], dtype=np.float64)
            pred = pred * (t * cfg.top_k_experts / max(pred.sum(), 1.0))
            order = np.argsort(-pred, kind="stable")
            cold = [int(x) for x in order if slots[x] < 0]
            if MIX_PRE != "auto":
                m = len(cold) if int(MIX_PRE) < 0 else int(MIX_PRE)
            else:
                # The time of the layer for m copied experts: the copies go on
                # during the work of the layers before, and the CPU part runs
                # during the copy wait and the GPU experts.
                cp = pred[cold]
                rest = cp.sum() - np.concatenate(([0.0], np.cumsum(cp)))
                used = cp > 0
                rest_e = used.sum() - np.concatenate(([0], np.cumsum(used)))
                ms = np.arange(len(cold) + 1)
                gpu_pairs = pred.sum() - rest
                cpu = cal["cpu_a"] + rest_e * cal["cpu_e"] + rest * cal["cpu_b"]
                tm = np.maximum(ms * per / cal["rate"],
                                a_ms + np.maximum(gpu_pairs * cal["gpu_pair"],
                                                  np.where(rest > 0, cpu, 0.0)))
                # The model is flat where the copies stay behind the other
                # work and the CPU part ends before the GPU experts. There the
                # most copies leave the CPU the least work: a margin for the
                # errors of the prediction and for other load on the CPU.
                m = int(ms[tm <= tm.min() + 0.25].max())
            self.mix_plan.append(m)
            pre = set(cold[:m])
            gslots = prog.gslots_of[layer]
            fill_tables(self.model, layer, h, self.stage, tgu, tdn, ranges, gslots, pre)
            for a in (tgu, tdn, gslots):
                _check(lib().gg_h2d(g.mirror.buffer_of(a).ptr, a.ctypes.data, a.nbytes))
        prog.hot_version = None

    @property
    def mix_cal(self):
        """The model of the time of a mixed group (MIX_START, then the
        measures of calibrate_mix)."""
        if "_mix_cal" not in self.__dict__:
            self._mix_cal = dict(MIX_START)
        return self._mix_cal

    def calibrate_mix(self, prog, t, wall):
        """Fit the model of plan_mix to the measures of the mixed group that
        just ran (gg_mix_stats; wall is its time in s): the copy rate, the
        time of the CPU part against its pairs in each layer, and the other
        GPU work for each token. Each new value is the mean of the old one
        and the measure."""
        buf = np.zeros(4096)
        n = lib().gg_mix_stats(buf.ctypes.data, buf.size)
        if n < 0:
            raise RuntimeError(lib().gg_last_error().decode())
        nbytes, secs, nf, nj = buf[0], buf[1], int(buf[2]), int(buf[3])
        waits = buf[4:4 + 3 * (nf + nj)].reshape(-1, 3)     # kind, wait, work before
        cpu_s = buf[4 + 3 * (nf + nj):4 + 3 * (nf + nj) + nj]
        cal = self.mix_cal
        new = {}
        if secs > 0 and nbytes > 0:
            new["rate"] = nbytes / (1e3 * secs)
        last = self.hot_cache.last if self.hot_cache is not None else None
        layers = sorted(prog.gslots_of)
        # Each layer gives a wait of the copies, then a wait of the CPU part.
        # The work before a copy wait is the other work of the layer (from the
        # wait of the CPU part of the layer before); the work before a CPU
        # wait is the GPU experts.
        kinds = waits[:, 0] if len(waits) else np.zeros(0)
        other = waits[1:][kinds[1:] == 0, 2]
        if other.size:
            new["gpu_tok"] = 1e3 * float(np.median(other)) / t
        if last is not None and nj == len(layers):
            cpu_pairs = np.array([float(last[l][prog.gslots_of[l] < 0].sum()) for l in layers])
            cpu_exp = np.array([float((last[l][prog.gslots_of[l] < 0] > 0).sum())
                                for l in layers])
            gpu_pairs = np.array([float(last[l][prog.gslots_of[l] >= 0].sum()) for l in layers])
            ms = 1e3 * cpu_s
            # ms = cpu_a + cpu_e experts + cpu_b pairs, with no term below 0:
            # the terms that come out below 0 go, and the fit runs again.
            cols = {"cpu_a": np.ones_like(ms), "cpu_e": cpu_exp, "cpu_b": cpu_pairs}
            keys = [k for k in cols if np.ptp(cols[k]) > 0 or k == "cpu_a"]
            while keys:
                coef = np.linalg.lstsq(np.stack([cols[k] for k in keys], 1), ms, rcond=None)[0]
                if (coef >= 0).all():
                    new.update({k: float(c) for k, c in zip(keys, coef)})
                    new.update({k: 0.0 for k in cols if k not in keys})
                    break
                keys = [k for k, c in zip(keys, coef) if c >= 0]
            gw = 1e3 * waits[kinds == 1, 2]
            if gw.size == len(layers) and gpu_pairs.sum() > 0:
                new["gpu_pair"] = float(gw.sum() / gpu_pairs.sum())
        for k, v in new.items():
            cal[k] = 0.5 * (cal[k] + v)
        if MIX_LOG:
            fw = 1e3 * float(waits[kinds == 0, 1].sum()) if len(waits) else 0.0
            cw = 1e3 * float(waits[kinds == 1, 1].sum()) if len(waits) else 0.0
            print("mix: %d tokens %.0f ms: copy waits %.0f ms, CPU waits %.0f ms, plan %.1f; "
                  "measured %s -> %s" % (t, 1e3 * wall, fw, cw, np.mean(self.mix_plan),
                                         {k: round(v, 5) for k, v in new.items()},
                                         {k: round(v, 5) for k, v in cal.items()}))

    def prefill(self, ids, pos=0, media=None):
        """Run a prompt from position pos. Return the hidden states after the
        final norm of every token. media: see group.

        A part of PREFILL_MIN tokens or more goes in chunks of CHUNK tokens,
        with the experts on the GPU. A short chunk goes to a program of the
        next power of two. Each such chunk copies the weights of the experts
        to the GPU. A shorter part goes in groups of 16 tokens, with the
        experts on the CPU, which costs less than that copy.
        """
        self.in_prompt = True       # for HotCache.seed
        try:
            return self._prefill(ids, pos, media or ())
        finally:
            self.in_prompt = False

    def _prefill(self, ids, pos, media=()):
        out = []
        c0 = 0
        while c0 < len(ids):
            end = min(c0 + CHUNK, len(ids))
            # A chunk ends before an image that it would split (an image has
            # at most 1120 tokens, less than a chunk).
            for sp in media:
                if sp.bidir and pos + c0 < sp.end and sp.start < pos + end < sp.end:
                    if sp.start > pos + c0:
                        end = sp.start - pos
                    else:
                        end = min(len(ids), sp.end - pos)
            chunk = ids[c0:end]
            spans = [sp for sp in media if sp.start < pos + end and sp.end > pos + c0]
            if (len(chunk) >= PREFILL_MIN or not self.model.cfg.enable_moe_block
                    or any(sp.bidir for sp in spans)):
                size = max(MT_CPU + 1, 1 << (len(chunk) - 1).bit_length())
                out.append(self.group(chunk, pos + c0, min(size, CHUNK), media=spans))
            else:
                chunk = ids[c0:c0 + MT_CPU]
                spans = [sp for sp in media if sp.start < pos + c0 + len(chunk)
                         and sp.end > pos + c0]
                out.append(self.group(chunk, pos + c0, MT_CPU, media=spans))
            c0 += len(chunk)
        return np.concatenate(out)

    def logits(self, rows=1):
        """Return the logits of the last rows of the last step or group, with
        the soft cap, shape (rows, vocabulary). An MTP verify group needs the
        logits of each of its rows. At most 16 rows."""
        vocab = self._run_head(rows)
        _check(lib().gg_d2h(self.host_logits.ctypes.data, self.out.ptr, rows * vocab * 4))
        return self.host_logits[:rows].copy()

    def topk(self, rows, k, temperature):
        """Return the candidates of sampling of the last rows (_topk_rows)."""
        self._run_head(rows)
        return _topk_rows(self, rows, k, temperature)

    def argmax(self, rows=1):
        """Return the best token of each of the last rows (_argmax_rows): the
        tokens of np.argmax on logits(rows), with a copy of rows integers."""
        self._run_head(rows)
        return _argmax_rows(self, rows)

    def _load_head(self):
        """Copy the head (the tied token table) to the GPU, on the first
        call."""
        if self.head is not None:
            return
        model = self.model
        if model._embed_q6k is not None:
            w = np.ascontiguousarray(model._embed_q6k_bytes)
            self.head_fn = lib().gg_q6k_head
        elif model._embed_q is not None and model._dtype == "int4":
            # A file that keeps the head in Q4_0: the blocks of the file
            # (gguf.int4_packed), 18 bytes for 32 weights.
            w = np.ascontiguousarray(model._embed_q).reshape(model._embed_q.shape[0], -1)
            self.head_fn = lib().gg_q4_head
        else:
            raise RuntimeError("the GPU head needs a Q6_K or a Q4_0 head")
        vocab = w.shape[0]
        self.head = Buffer(w.nbytes, "weights")
        self.head.upload(w)
        self.out = Buffer(4 * vocab * MT_CPU, "weights")
        self.host_logits = pinned((MT_CPU, vocab))     # a faster copy from the GPU

    def _embed_gpu(self, g, prog, tokens, size):
        """Write the rows of the tokens (and zeros for the padding to size)
        into x of the program on the GPU, with the Q4_0 token table of the
        head (gg_embed_q4_rows). Return False when the head is not Q4_0."""
        model = self.model
        if not EMBED_GPU or model._embed_q6k is not None or model._embed_q is None \
                or model._dtype != "int4":
            return False
        self._load_head()
        if getattr(self, "embed_ids", None) is None or self.embed_ids.nbytes < 4 * size:
            self.embed_ids = Buffer(4 * max(size, CHUNK))
            self.embed_ids_host = pinned((max(size, CHUNK),), np.int32)
        ids = self.embed_ids_host[:size]
        ids[:len(tokens)] = tokens
        ids[len(tokens):] = -1
        _check(lib().gg_h2d(self.embed_ids.ptr, ids.ctypes.data, 4 * size))
        x = g.mirror.buffer_of(prog.names["x"]).ptr
        _check(lib().gg_embed_q4_rows(self.head.ptr, self.embed_ids.ptr, x, size,
                                      model.cfg.hidden_size,
                                      float(np.float32(model.cfg.embed_scale))))
        return True

    def _run_head(self, rows):
        """Run the head on the last rows into self.out. The first call copies
        the head to the GPU. Return the vocabulary size."""
        model, cfg = self.model, self.model.cfg
        self._load_head()
        assert 1 <= rows <= min(self.rows, MT_CPU)
        vocab = self.host_logits.shape[1]
        cap = float(cfg.final_logit_softcapping or 0.0)
        step = cfg.hidden_size * 4
        _check(self.head_fn(self.head.ptr, self.last - (rows - 1) * step, self.out.ptr,
                            vocab, cfg.hidden_size, cap, rows))
        if self.hot_cache is not None and self.hot_cache.due:
            self.hot_cache.observe()
        return vocab


def _argmax_rows(g, rows):
    """The best token of each row of g.out (the logits that the head just
    made), as np.argmax (gg_argmax_rows): rows integers cross to the host, not
    rows of the vocabulary."""
    vocab = g.host_logits.shape[1]
    if getattr(g, "argmax_buf", None) is None:
        g.argmax_buf = Buffer(4 * MT_CPU, "program")
        g.argmax_host = pinned((MT_CPU,), np.int32)
    _check(lib().gg_argmax_rows(g.out.ptr, rows, vocab, g.argmax_buf.ptr))
    _check(lib().gg_d2h(g.argmax_host.ctypes.data, g.argmax_buf.ptr, 4 * rows))
    return [int(v) for v in g.argmax_host[:rows]]


# The most candidates of sampling that the GPU selects for a row (_topk_rows).
TOPK_MAX = 1024


def _topk_rows(g, rows, k, temperature):
    """The candidates of sampling of the rows of g.out (the logits that the
    head just made): (ids, values) of the k largest logits of each row, in no
    order, shape (rows, k), and (max, sum of exp((l - max) / temperature)) of
    each row, shape (rows, 2). The host copies these in place of the rows of
    the vocabulary (Sampler.sample_sparse). k is at most TOPK_MAX."""
    vocab = g.host_logits.shape[1]
    k = int(min(k, TOPK_MAX, vocab))
    if getattr(g, "topk_buf", None) is None:
        # ids, values, and stats of up to MT_CPU rows in one buffer: one copy.
        g.topk_buf = Buffer(8 * MT_CPU * TOPK_MAX + 8 * MT_CPU, "program")
        g.topk_host = pinned((2 * MT_CPU * TOPK_MAX + 2 * MT_CPU,))
    inv_t = 1.0 / float(temperature) if temperature > 0 else 1.0
    n = rows * k
    base = g.topk_buf.ptr
    _check(lib().gg_topk(g.out.ptr, rows, vocab, k, inv_t, base, base + 4 * n, base + 8 * n))
    h = g.topk_host
    _check(lib().gg_d2h(h.ctypes.data, base, 8 * n + 8 * rows))
    ids = h[:n].view(np.int32).reshape(rows, k).copy()
    vals = h[n:2 * n].reshape(rows, k).copy()
    stat = h[2 * n:2 * n + 2 * rows].reshape(rows, 2).copy()
    return ids, vals, stat


# The size of a piece of a copy of the cache. The copies of each layer of a
# step are small and on the path of the step; a piece holds the link for
# PIECE / 6.9 GB/s, about 0.15 ms. (Pieces of 128 KB and of the whole
# expert gave the same rate; 32 KB was slower.)
PIECE = 1 << 20


def _pieces(src, dst, n):
    """Return the rows (src, dst, bytes) of a copy of n bytes in PIECE parts."""
    return [(src + o, dst + o, min(PIECE, n - o)) for o in range(0, n, PIECE)]


class HotCache:
    """The hot experts of a ModelGPU follow the text.

    The GPU holds a fixed count of experts in each layer, in slots (see
    SplitCompiler.moe_hot). After each decode step, the host reads the
    selection of each layer (GP_HOT_SPLIT writes it to the pinned array of
    the step) and keeps a score of each expert: score = decay * score + 1 for
    a selection. When a selected expert that the GPU does not hold has a
    higher score than the lowest expert in the slots, it takes that slot:

    1. The slot table marks the old expert cold at once. The GPU is idle
       between steps, so no kernel reads the slot after that.
    2. A worker thread copies the new expert to the slot (gg_cache_copy).
    3. When the copy is done, a later step marks the new expert hot.

    At most max_ins experts change in each step. NP_GEMMA_GPU_HOT_DECAY and
    NP_GEMMA_GPU_HOT_INS change decay and max_ins. See SPLIT_PLAN.md.
    """

    def __init__(self, dev, decay=None, max_ins=None, layers=None, top_k=None):
        """layers (or None for a ModelGPU) gives one dict for each layer
        with slots: layer, slots (the host array), dslots (its device
        address), ip (the pinned array of GP_HOT_SPLIT), and parts: for each
        part of an expert, (host address of expert 0, bytes of one expert,
        device address of slot 0). top_k (or None for a ModelGPU) is the
        count of experts of a token."""
        self.dev = dev
        model = dev.model
        cfg = model.cfg
        self.decay = float(decay or os.environ.get("NP_GEMMA_GPU_HOT_DECAY", "0.97"))
        self.max_ins = int(max_ins or os.environ.get("NP_GEMMA_GPU_HOT_INS", "8"))
        # A cold expert goes to the GPU only from its admit-th use while it
        # is cold. Its first uses run on the CPU.
        self.admit = int(os.environ.get("NP_GEMMA_GPU_HOT_ADMIT", "2"))
        self.top_k = top_k or cfg.top_k_experts
        self.layers = layers if layers is not None else []  # the layers with hot experts
        for layer, (_hot, gu_store, dn_store, slots) in (
                sorted(dev.prog.hot_stores.items()) if layers is None else ()):
            gq, dq, _q4x = expert_host(model, layer)
            n = gq.shape[0]
            self.layers.append(dict(
                layer=layer, slots=slots, dslots=dev.g.mirror.buffer_of(slots).ptr,
                parts=[(gq.ctypes.data, gq.nbytes // n, dev.g.mirror.buffer_of(gu_store).ptr),
                       (dq.ctypes.data, dq.nbytes // n, dev.g.mirror.buffer_of(dn_store).ptr)],
                ip=dev.prog.hot_ip[layer]))
        m = len(self.layers)
        # One row for each entry of layers: the score of each expert, the
        # experts in the slots, and the experts of the pending copies.
        self.score = np.zeros((m, cfg.num_experts), dtype=np.float32)
        self.held = np.stack([e["slots"] >= 0 for e in self.layers]) if m else self.score > 0
        self.incoming = np.zeros_like(self.held)
        self.uses = np.zeros(self.score.shape, dtype=np.int32)   # uses while cold
        self.total = np.zeros(self.score.shape, dtype=np.int64)  # all the uses (profile())
        # The selections of the routers of a group (GP_COUNT), for all the
        # layers; seed() reads the rows of the layers with slots.
        self.counts = np.zeros((cfg.num_hidden_layers, cfg.num_experts), dtype=np.int32)
        self.count_rows = [e["layer"] for e in self.layers]
        self.last = None        # the counts of the last large group
        # The experts that a prompt pass can change; 0 (the default) only
        # reads the counts. A test gave no gain (SPLIT_PLAN.md).
        self.seed_ins = int(os.environ.get("NP_GEMMA_GPU_HOT_SEED", "0"))
        self.rows = np.arange(m)[:, None]
        self.pending = []       # (job id, [(row, expert, slot)], ranges, locked segments)
        self.due = False        # a step is done, and observe has not run
        self.version = 0        # changes when a slot table changes
        # Counts for a test: the copies, the steps, and the cold experts of
        # the steps (the mean over the layers is cold / steps / layers).
        self.copies = 0
        self.steps = 0
        self.cold = 0
        # The pool of warm experts (set_pool): None, or an ExpertPool, and
        # the rows whose experts fit its blocks
        self.pool = None
        self.pool_rows = np.zeros(m, dtype=bool)

    WARM = 1 << 24          # gpu.cu KQH_WARM: slot WARM + i is block i of the pool

    def set_pool(self, pool):
        """Use the blocks of pool (an ExpertPool) for warm experts: a row whose
        experts fit takes a free block, or the block of the warm expert of
        the lowest score of any row, when its new expert scores higher."""
        self.pool = pool
        pool.evict_hook = self.pool_evict
        self.pool_rows = np.array([len(e["parts"]) == 3 and all(
            nb <= pool.stride[p] for p, (_s, nb, _d) in enumerate(e["parts"])) for e in self.layers])

    def _pool_free(self):
        return self.pool is not None and bool((self.pool.owner == -1).any())

    def pool_evict(self, blocks):
        """The warm experts of blocks go cold (after the copies that run);
        the blocks are free."""
        pool = self.pool
        blocks = np.asarray(blocks, dtype=np.int64)
        held = blocks[pool.owner[blocks] >= 0]
        if held.size == 0:
            return
        self.prepare(wait=True)
        E = self.score.shape[1]
        changed = set()
        for i in held:
            r, x = divmod(int(pool.owner[i]), E)
            e = self.layers[r]
            if e["slots"][x] == self.WARM + i:
                e["slots"][x] = -1
                self.held[r, x] = False
                changed.add(r)
            pool.owner[i] = -1
        self._upload(sorted(changed))

    def pool_reserve(self, n):
        """n blocks for the copies of a mixed group (or fewer, if the pool
        has fewer): the free blocks first, then those of the warm experts of
        the lowest scores. Return their indices (int32, in order)."""
        pool = self.pool
        if pool is None:
            return np.zeros(0, np.int32)
        self.pool_unreserve()
        free = np.flatnonzero(pool.owner == -1)
        if free.size < n:
            held = np.flatnonzero(pool.owner >= 0)
            E = self.score.shape[1]
            r, x = np.divmod(pool.owner[held], E)
            out = held[np.argsort(self.score[r, x], kind="stable")[:n - free.size]]
            self.pool_evict(out)
            free = np.flatnonzero(pool.owner == -1)
        take = np.sort(free[:n])
        pool.owner[take] = -2
        return take.astype(np.int32)

    def pool_unreserve(self):
        if self.pool is not None:
            self.pool.owner[self.pool.owner == -2] = -1

    def pool_shrink(self, src, nbytes):
        """Give back segments of src until about nbytes are free, those with
        the fewest warm experts first (GpuMem). Return the bytes freed."""
        pool = self.pool
        if pool is None or nbytes <= 0:
            return 0
        # not a segment with blocks reserved for a mixed group (its plan and
        # copies use them) or locked (a run, a copy)
        segs = [sg for sg in pool.segments(src)
                if not (pool.owner[sg * pool.K:(sg + 1) * pool.K] == -2).any()
                and not pool.bufs[sg].locks]
        held = [int((pool.owner[sg * pool.K:(sg + 1) * pool.K] >= 0).sum()) for sg in segs]
        take = [sg for _h, sg in sorted(zip(held, segs))][:-(-int(nbytes) // pool.seg_bytes)]
        if not take:
            return 0
        blocks = np.concatenate([np.arange(sg * pool.K, (sg + 1) * pool.K) for sg in take])
        self.pool_evict(blocks)
        pool.drop(take)
        return len(take) * pool.seg_bytes

    def pool_drop(self, src=None):
        """Give back the segments of src (None: all): their warm experts go
        cold first."""
        pool = self.pool
        if pool is None:
            return
        segs = pool.segments(src)
        if not segs:
            return
        blocks = np.concatenate([np.arange(sg * pool.K, (sg + 1) * pool.K) for sg in segs])
        self.prepare(wait=True)
        self.pool_evict(blocks)
        pool.drop(segs)

    def _slot_addr(self, part, dst, nb, slot):
        """The device address of slot of a part (dst: slot 0 of the store)."""
        if slot >= self.WARM:
            return self.pool.addr(part, slot - self.WARM)
        return dst + slot * nb

    def _upload(self, rows):
        for r in rows:
            e = self.layers[r]
            _check(lib().gg_h2d(e["dslots"], e["slots"].ctypes.data, e["slots"].nbytes))
        if rows:
            self.version += 1

    def prepare(self, wait=False):
        """Before a step or a group: mark the experts of the finished copies
        hot. wait waits for all the copies."""
        if self.due:
            self.observe()
        if not self.pending:
            return
        done, still = set(), []
        for job, items, ranges, segs in self.pending:
            r = lib().gg_cache_query(job, 1 if wait else 0)
            if r < 0:
                raise RuntimeError(lib().gg_last_error().decode())
            if r == 0:
                still.append((job, items, ranges, segs))
                continue
            mem().unlock(segs)
            for row, x, slot in items:
                self.layers[row]["slots"][x] = slot
                self.held[row, x] = True
                self.incoming[row, x] = False
                done.add(row)
        self.pending = still
        self._upload(done)

    def profile(self):
        """The uses of each expert so far (decode steps, MTP rows, groups):
        an array (the layers + 1, experts), 0 for the layers with no hot
        experts. Saved, it is NP_GEMMA_EXPERT_COUNTS of a later start."""
        L = max([e["layer"] for e in self.layers] + [0]) + 1
        out = np.zeros((L, self.score.shape[1]), np.int64)
        for r, e in enumerate(self.layers):
            out[e["layer"]] = self.total[r]
        return out

    def observe(self):
        """After a decode step: score the selection, and start the copies of
        the experts that take the place of others. The slot tables change on
        the stream of the programs, after the work that is in it; no kernel
        of that work reads the slots. The next step comes after them."""
        self.due = False
        k = self.top_k
        rows = [r for r, e in enumerate(self.layers) if e.get("step", True)]
        sel = np.stack([self.layers[r]["ip"][k + 1:2 * k + 1] for r in rows])
        self.steps += 1
        self.cold += int(sum(int(self.layers[r]["ip"][k]) for r in rows))
        self.score_rows(rows, sel)

    def score_rows(self, rows, sel):
        """Score the selection sel (len(rows) x top_k experts; -1 for none)
        of one token in the layers of the rows of layers, as a step does,
        and start the copies of the experts that take the place of others.
        A group (an MTP verify group, the MTP layer) gives the selections of
        its tokens in turn (or all at once: score_group)."""
        self.score_group(rows, np.asarray(sel)[None])

    def score_group(self, rows, sels):
        """Score the selections of n tokens in turn, sels (n x len(rows) x
        top_k; -1 for none), in the layers of the rows of layers: the scores
        of n calls of score_rows (score decay^n + the sum of decay^(n-1-j)
        for the selections of token j), then one pass of copies for the
        group, of up to max_ins experts for each token. Only the rows of
        rows change: a pass over the whole table took about 1.5 ms of Python
        for each token of an MTP verify group of Qwen3.8 (49 x 512)."""
        rows = np.asarray(rows)
        sels = np.asarray(sels)
        n = sels.shape[0]
        if n == 0 or rows.size == 0:
            return
        score, d = self.score, self.decay
        sub = score[rows]
        sub *= d ** n
        uses = self.uses[rows]
        ri = np.broadcast_to(np.arange(rows.size)[:, None], sels.shape[1:])
        hit = np.zeros(sub.shape, dtype=bool)
        for j in range(n):
            ok = sels[j] >= 0
            rr, ss = ri[ok], sels[j][ok]
            sub[rr, ss] += d ** (n - 1 - j)       # the experts of a row differ
            uses[rr, ss] += 1
            self.total[rows[rr], ss] += 1
            hit[rr, ss] = True
        held = self.held[rows]
        uses[held] = 0
        score[rows] = sub
        self.uses[rows] = uses
        low = np.where(held, sub, np.inf).min(axis=1)
        if self._pool_free():
            # a row that can take a free block of the pool takes the next
            # best experts too
            low[self.pool_rows[rows]] = -np.inf
        cand = hit & ~held & ~self.incoming[rows] & (sub > low[:, None]) & (uses >= self.admit)
        if not cand.any():
            return
        full = np.zeros_like(self.held)
        full[rows] = cand
        self._replace(full, self.max_ins * n)

    def seed(self, g, t, prompt):
        """After a group of t tokens (a part of a prompt, or an MTP verify
        group): add the selections of its routers to the scores, as if its
        tokens were the last t steps, then change the slots. After a part of
        a prompt, up to seed_ins experts change (not max_ins), and one use is
        enough, so the slots fit the prompt before the decode; the copies go
        on during the decode. With NP_GEMMA_GPU_HOT_SEED=0 (the default) a
        prompt changes nothing. An MTP verify group always counts: MTP runs
        no decode steps, so without it the slots would not change."""
        if not self.layers:
            return
        dev = g.mirror.buffer_of(self.counts).ptr
        _check(lib().gg_d2h(self.counts.ctypes.data, dev, self.counts.nbytes))
        if t > MT_CPU:
            self.last = self.counts.copy()      # for ModelGPU.plan_mix
        c = self.counts[self.count_rows].astype(np.float32)
        self.total += self.counts[self.count_rows]
        self.counts[:] = 0
        _check(lib().gg_h2d(dev, self.counts.ctypes.data, self.counts.nbytes))
        if prompt and self.seed_ins == 0:
            return
        # Each of the t tokens gets the mean weight that the steps would give.
        f = self.decay ** t
        w = (1.0 - f) / ((1.0 - self.decay) * t)
        self.score *= f
        self.score += w * c
        self.uses += c.astype(np.int32)
        self.uses[self.held] = 0
        low = np.where(self.held, self.score, np.inf).min(axis=1)
        if self._pool_free():
            low[self.pool_rows] = -np.inf
        cand = (c > 0) & ~self.held & ~self.incoming & (self.score > low[:, None]) & \
            (self.uses >= (1 if prompt else self.admit))
        if cand.any():
            self._replace(cand, self.seed_ins if prompt else self.max_ins)

    def _replace(self, cand, limit):
        """Move up to limit experts of the mask cand (rows, experts) to the
        slots of the lowest held experts, best gain first."""
        score = self.score
        low = np.where(self.held, score, np.inf).min(axis=1)
        ri, xi = np.nonzero(cand)
        order = np.argsort(low[ri] - score[ri, xi], kind="stable")
        items, rows, changed = [], [], set()
        pool, E = self.pool, score.shape[1]
        # The free blocks of the pool, and its warm experts by score, once for
        # the call (the scores do not change in it; a block that a copy takes
        # gets an owner with a copy on the way, which the walk skips): the
        # same choices as a pass over the pool for each candidate, which took
        # 1.7 ms a call (2.9 ms a token of MTP decode with the NVFP4 experts).
        free, pv, pk = [], None, 0
        if pool is not None and self.pool_rows[ri].any():
            free = list(np.flatnonzero(pool.owner == -1))
            pb = np.flatnonzero(pool.owner >= 0)
            if pb.size:
                pr, px = np.divmod(pool.owner[pb], E)
                ps = np.where(self.incoming[pr, px], np.inf, score[pr, px])
                pv = (pb, pool.owner[pb].copy(), ps, np.argsort(ps, kind="stable"))
        for o in order[:4 * limit]:
            if len(items) >= limit:
                break
            r, x = int(ri[o]), int(xi[o])
            e = self.layers[r]
            if pool is not None and self.pool_rows[r] and free:
                slot = self.WARM + int(free.pop(0))    # a free block: no victim
            else:
                held = np.where(self.held[r], score[r], np.inf)
                victim = int(held.argmin())
                vr, vs = r, held[victim]
                if pool is not None and self.pool_rows[r] and pv is not None:
                    # or the warm expert of the lowest score of any row (not
                    # one with a copy on the way)
                    pb, own, ps, po = pv
                    while pk < po.size:
                        j = int(po[pk])
                        if pool.owner[pb[j]] == own[j] and ps[j] < np.inf:
                            break
                        pk += 1
                    if pk < po.size and ps[j] < vs:
                        vr, victim, vs = int(own[j] // E), int(own[j] % E), ps[j]
                if score[r, x] <= vs:
                    continue
                ve = self.layers[vr]
                slot = int(ve["slots"][victim])
                ve["slots"][victim] = -1
                self.held[vr, victim] = False
                changed.add(vr)
            if slot >= self.WARM:
                pool.owner[slot - self.WARM] = r * E + x
            self.incoming[r, x] = True
            items.append((r, x, slot))
            for part, (src, nb, dst) in enumerate(e["parts"]):
                # src: the host address of expert 0 (expert x at x nb), or the
                # address of each expert (an int64 array: QwenGPU._expert_srcs)
                a = int(src[x]) if isinstance(src, np.ndarray) else src + x * nb
                rows += _pieces(a, self._slot_addr(part, dst, nb, slot), nb)
        if not items:
            return
        # The old experts are cold on the GPU before the copies start.
        self._upload(changed)
        ranges = np.array(rows, dtype=np.int64)
        # GpuMem: the segments of the pool that the copies write stay until
        # they land (prepare)
        segs = []
        if pool is not None:
            segs = list({id(b): b for b in (pool.bufs[(s - self.WARM) // pool.K]
                                           for _r, _x, s in items if s >= self.WARM)}.values())
            mem().lock(segs)
        job = lib().gg_cache_copy(ranges.ctypes.data, len(rows))
        if job < 0:
            mem().unlock(segs)
            raise RuntimeError(lib().gg_last_error().decode())
        self.pending.append((job, items, ranges, segs))
        self.copies += len(items)

    def refresh(self, prog, g):
        """Fill the tables of a large group again if the slots changed."""
        if getattr(prog, "hot_version", None) == self.version:
            return
        hot = self.dev._hot_devices()
        for layer, (tgu, tdn, ranges) in prog.tables_of.items():
            fill_tables(self.dev.model, layer, hot.get(layer), self.dev.stage, tgu, tdn, ranges)
            for a in (tgu, tdn):
                _check(lib().gg_h2d(g.mirror.buffer_of(a).ptr, a.ctypes.data, a.nbytes))
        prog.hot_version = self.version


def offload(model, experts_gb=0.0):
    """Put the weights of a Model outside the experts, and its output head, on
    the GPU. The decode steps of one token then run there (NP_GEMMA_GPU=1).

    experts_gb 0 keeps every expert on the CPU. A value above 0 also puts the
    most used experts on the GPU, up to that many GB. None takes the default
    budget of ModelGPU. The function copies the weights now, so the first
    request does not wait for them. Return the ModelGPU.

    For an E4B model, put the whole model on the GPU (E4BGPU). experts_gb
    has no effect, because the E4B has no experts. Return the E4BGPU.
    """
    from . import model as model_mod
    os.environ["NP_GEMMA_GPU"] = "1"
    if hasattr(model, "mode"):
        from . import e4b as e4b_mod
        e4b_mod._GPU = True
        g = E4BGPU(model)
        g.logits()      # copies the head to the GPU
        model._gpu, model._gpu_cache, model._gpu_xn = g, None, None
        return g
    model_mod._GPU = True
    hot = {}
    if experts_gb is None:
        hot = None
    elif experts_gb > 0 and model.cfg.enable_moe_block:
        counts = hot_counts(model)
        hot = pick_hot(model, counts, experts_gb * 1e9) if counts is not None else {}
    g = ModelGPU(model, hot=hot)
    g.logits()          # copies the head to the GPU
    model._gpu = g
    model._gpu_cache = None
    model._gpu_xn = None
    return g


def describe(g):
    """Return one line about the GPU part of a ModelGPU or an E4BGPU."""
    free, total = mem_info()
    if not hasattr(g, "hot"):
        return "GPU: %.2f GB of weights and buffers; %.1f of %.1f GB free" % (
            g.g.mirror.nbytes() / 1e9, free / 1e9, total / 1e9)
    return ("GPU: %.2f GB of weights and buffers, %d hot experts, head %.2f GB; "
            "%.1f of %.1f GB free" % (g.g.mirror.nbytes() / 1e9,
                                       sum(len(v) for v in g.hot.values()),
                                       (g.head.nbytes if g.head else 0) / 1e9,
                                       free / 1e9, total / 1e9))


# ---- the MTP drafter on the GPU (SPLIT_PLAN.md, phase 5) --------------------

def quantize_q4_0(w):
    """Quantize a float32 matrix to int4 blocks with float16 scales. Return
    (blocks, scales), as ops.quantize_int4 does: the scale of the QAT grid
    when the block has one. The float32 scales equal the float16 scales of the
    blocks, which the GPU kernels read."""
    from . import ops
    blocks, scale = ops.quantize_int4(np.asarray(w, dtype=np.float32))
    scale = scale.astype(np.float16).astype(np.float32)
    return np.ascontiguousarray(blocks), scale


def compile_drafter(a, target_cfg, cache_form="e4b", rot=(False, False)):
    """Compile one draft step of the assistant a (an Assistant with float32
    weights) for the GPU. The steps are those of Assistant.step and of its
    head: the centroid head (E4B), or the full head and an argmax (26B).

    cache_form is the cache of the target: "e4b", the float cache of the E4B
    (heads, positions, head_dim), or "qc", the int16 cache of the 26B on the
    GPU (GPUKV). For "qc" the step binds the addresses of the first key row
    and the count of the rows (the slots kq.s, ks.s, vq.s, vs.s, n.s and the
    same with .f), as Assistant._attention finds them.

    The input "xin" holds the embedding of the token and the hidden state h,
    with the backbone size each. The step writes the next h into the second
    half of xin, so the next draft step finds it there. "token" gets the
    draft token. The attention of a layer reads the cache of the target:
    the query at pos sees the rows before pos. The record GP_ATTN_F32H sees
    the rows to its pos operand, so the step gives it pos - 1 (the slot
    "pd"), and a window one smaller.
    """
    cfg = a.cfg
    bk = a.backbone
    c = P.Compiler(a)
    xin = np.zeros((1, 2 * bk), dtype=np.float32)
    c.env["xin"] = xin
    q4 = lambda w: quantize_q4_0(w)  # noqa: E731
    u = P.k_int4(c, q4(a.pre), xin)
    tplan = {True: next(p for p in target_cfg.plan if p.is_sliding),
             False: next(p for p in target_cfg.plan if not p.is_sliding)}
    for i, w in enumerate(a.layers):
        plan = cfg.plan[i]
        kind = "s" if plan.is_sliding else "f"
        hd, nq = plan.head_dim, plan.num_q_heads
        h = P.k_rms_norm(c, u, w["input_layernorm"])
        q = P.k_int4(c, q4(w["self_attn.q_proj"]), h)
        q = P.k_rms_norm_rows(c, q, hd, w["self_attn.q_norm"])
        P.k_rope(c, q, None, c.p.slot("cos." + kind), c.p.slot("sin." + kind), i)
        att = c.buffer((1, nq * hd))
        window = plan.sliding_window or 0
        kvh = tplan[plan.is_sliding].num_kv_heads
        if cache_form in ("qc", "q8", "qv"):
            sl = lambda n: c.p.slot("%s.%s" % (n, kind))  # noqa: E731
            op = {"q8": P.ATTN_Q8, "qv": P.ATTN_V8}.get(cache_form, P.ATTN_QC)
            if rot[0]:      # the rotated keys of the target (rq8): the query too
                c.p.emit(P.TQ_ROT, q, q.size // 32, 0)
            c.p.emit(op, q, sl("kq"), sl("ks"),
                     sl("vq"), sl("vs"), c.p.slot("scores"), att, nq, kvh, hd, sl("n"))
            if rot[1]:      # rotated values (rq8, k16vr8): the output back
                c.p.emit(P.TQ_ROT, att, att.size // 32, 1)
        else:
            c.p.emit(P.ATTN_F32H, q, c.p.slot("k." + kind), c.p.slot("v." + kind),
                     c.p.slot("scores"), att, nq, kvh, hd, 1,
                     c.p.slot("pd"), c.p.slot("hs." + kind), max(0, window - 1), 1)
        o = P.k_int4(c, q4(w["self_attn.o_proj"]), att)
        u = P.k_add(c, u, P.k_rms_norm(c, o, w["post_attention_layernorm"]))
        m = P.k_rms_norm(c, u, w["pre_feedforward_layernorm"])
        g = P.k_int4(c, q4(w["mlp.gate_proj"]), m)
        up = P.k_int4(c, q4(w["mlp.up_proj"]), m)
        mm = P.k_int4(c, q4(w["mlp.down_proj"]), P.k_mul_v(c, P.k_gelu(c, g), up))
        u = P.k_add(c, u, P.k_rms_norm(c, mm, w["post_feedforward_layernorm"]))
        u = P.k_mul(c, u, w["layer_scalar"])
    un = P.k_rms_norm(c, u, a.norm)
    hnext = P.k_int4(c, q4(a.post), un)
    token = np.zeros(1, dtype=np.int32)
    if a.centroids is not None:
        n_cent, per = a.order.shape
        cent = np.ascontiguousarray(a.centroids, dtype=np.float32)
        clog = c.buffer(n_cent)
        c.p.emit(P.F32_LINEAR, un, cent, clog, n_cent, cfg.hidden_size)
        c.p.emit(P.DRAFT_HEAD, un, clog, np.ascontiguousarray(a.order, dtype=np.int32),
                 np.ascontiguousarray(a.head_f32, dtype=np.float32), c.buffer(a.top_k * per),
                 np.zeros(a.top_k, dtype=np.int32), token, n_cent, per, a.top_k,
                 cfg.hidden_size)
    else:
        # The full head: the int4 product with the table of the drafter, and
        # the best token. The assistant has no soft cap.
        logits = P.k_int4(c, q4(a.head), un)
        c.p.emit(P.ARGMAX, logits, logits.size, token)
    c.p.emit(P.COPY, hnext, xin[:, bk:], bk * 4)
    c.env["token"] = token
    return c.p.finish()


# 1: the draft steps of a round run one after another on the GPU, with no
# copy to the host between them (GPUDrafter._draft_chain), when the target
# keeps its token table on the GPU as a Q4_0 head. 0 keeps a copy of the
# token and of the embedding for each step.
DRAFT_CHAIN = os.environ.get("NP_GEMMA_GPU_DRAFT_CHAIN", "1") == "1"


class GPUDrafter:
    """The MTP drafter on the GPU, for the E4B model or the 26B model. draft()
    has the arguments of Assistant.draft, so mtp_stream takes either.

    The drafter reads the cache of the target on the GPU. For the E4B, those
    are the buffers that its shared layers reuse. For the 26B, they are the
    int16 buffers of two layers of GPUKV (assistant.shared_layers). Thus it
    needs no copy of the cache to the host. Its weights are its own int4
    blocks with float16 scales (quantize_q4_0), so its drafts can differ a
    little from the CPU drafter. The target checks each draft, so the text
    does not change.
    """

    on_gpu = True

    def __init__(self, path, target, weights=None):
        from .assistant import Assistant
        self.a = Assistant(path, dtype="f32", weights=weights)
        self.target = target
        self.p_min = 0.0
        self.e4b = hasattr(target, "mode")      # E4B has a mode; Model does not
        from . import model as _model_mod
        form = "e4b" if self.e4b else {"int8": "q8", "k16v8": "qv"}.get(
            _model_mod.kv_base(_model_mod.KV_FORM), "qc")
        self.kesz = 1 if form == "q8" else 2
        self.vesz = 1 if form in ("q8", "qv") else 2
        self.prog = compile_drafter(self.a, target.cfg, form,
                                    rot=(False, False) if self.e4b else _model_mod.kv_rot())
        self.g = None

    def draft(self, target, token, h, pos, cache, n, eos_ids=()):
        target._gpu_attach(cache)
        tg = target._gpu
        if self.g is None:
            # Its arrays share the mirror of the target: keep them on the GPU
            # when the target frees a program (ProgramLRU), and free the
            # programs of the target used least recently when the memory
            # runs out.
            tg.__dict__.setdefault("extra_progs", []).append(self.prog)
            while True:
                try:
                    self.g = GPUProgram(self.prog, mirror=tg.g.mirror)
                    break
                except MemoryError:
                    # The drafts read h from the host, so the output of the
                    # last group can go too.
                    if not (hasattr(tg, "_evict") and tg._evict(last=True)):
                        raise
        a, bk = self.a, self.a.backbone
        kw = {"pd": pos - 1}
        if not self.e4b:
            from .assistant import shared_layers
            layers = dict(zip(("s", "f"), shared_layers(target.cfg)))
        for kind, sliding in (("s", True), ("f", False)):
            plan = next(p for p in a.cfg.plan if p.is_sliding == sliding)
            if self.e4b:
                store = cache.shared["sliding_attention" if sliding else "full_attention"]
                kw["k." + kind], kw["v." + kind] = store[0], store[1]
                kw["hs." + kind] = store[0].shape[2]       # position-major
            else:
                # The rows of positions lo to pos - 1 of the target layer, as
                # Assistant._attention reads them.
                i = layers[kind]
                kv = tg.kv
                base = kv.base[i]
                lo = max(0, pos - a.cfg.sliding_window + 1 - base) if sliding else 0
                tp = target.cfg.plan[i]
                per = tp.num_kv_heads * tp.head_dim
                b = kv.bufs[i]
                assert (kv.kesz, kv.vesz) == (self.kesz, self.vesz), \
                    "the drafter and the GPU cache differ in form"
                kw.update({"kq." + kind: b["kq"].ptr + lo * kv.kesz * per,
                           "ks." + kind: b["ks"].ptr + lo * per // 8,
                           "vq." + kind: b["vq"].ptr + lo * kv.vesz * per,
                           "vs." + kind: b["vs"].ptr + lo * per // 8,
                           "n." + kind: pos - base - lo})
            cos, sin = a._cos_sin(plan, pos)
            kw["cos." + kind] = np.ascontiguousarray(cos, dtype=np.float32)
            kw["sin." + kind] = np.ascontiguousarray(sin, dtype=np.float32)
        kw["scores"] = np.empty(max(p.num_q_heads for p in a.cfg.plan) * (pos + 1), np.float32)
        self.g.bind(kw, tg.cache if self.e4b else None)
        xin = self.g.mirror.buffer_of(self.prog.names["xin"]).ptr
        tokbuf = self.g.mirror.buffer_of(self.prog.names["token"])
        hh = np.ascontiguousarray(h, dtype=np.float32).reshape(-1)
        _check(lib().gg_h2d(xin + bk * 4, hh.ctypes.data, bk * 4))
        if (DRAFT_CHAIN and getattr(tg, "head", None) is not None
                and getattr(tg, "head_fn", None) is lib().gg_q4_head):
            return self._draft_chain(target, tg, token, n, xin, tokbuf, eos_ids)
        out = []
        got = np.zeros(1, dtype=np.int32)
        for _ in range(n):
            emb = np.ascontiguousarray(target.embed([token]), dtype=np.float32).reshape(-1)
            _check(lib().gg_h2d(xin, emb.ctypes.data, bk * 4))
            self.g.run()
            tokbuf.download(got)
            token = int(got[0])
            out.append(token)
            if token in eos_ids:
                break
        return out

    def _draft_chain(self, target, tg, token, n, xin, tokbuf, eos_ids):
        """The n draft steps with no copy to the host between them: the
        embedding of each draft token comes from the Q4_0 head of the target
        on the GPU (gg_embed_q4, the values of target.embed), and each token
        goes to a device array. One copy at the end brings the n tokens. The
        drafts after an end token are dropped."""
        bk = self.a.backbone
        if getattr(self, "chain_out", None) is None:
            self.chain_out = Buffer(4 * 64)
            self.chain_host = pinned((64,), np.int32)
        n = min(n, 64)
        scale = float(np.float32(target.cfg.embed_scale))
        first = np.array([token], dtype=np.int32)
        _check(lib().gg_h2d(tokbuf.ptr, first.ctypes.data, 4))
        for j in range(n):
            _check(lib().gg_embed_q4(tg.head.ptr, tokbuf.ptr, xin, bk, scale))
            self.g.run()
            _check(lib().gg_d2d(self.chain_out.ptr + 4 * j, tokbuf.ptr, 4))
        _check(lib().gg_d2h(self.chain_host.ctypes.data, self.chain_out.ptr, 4 * n))
        out = []
        for t in self.chain_host[:n]:
            out.append(int(t))
            if int(t) in eos_ids:
                break
        return out
