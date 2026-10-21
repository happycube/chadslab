"""Run the program of a decode step on a CUDA GPU.

SPLIT_PLAN.md, phase 3. The CUDA code (backends/cuda/csrc) has a kernel for each
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
import os
import threading
import weakref

import numpy as np

from reclank.ir import program as P
from reclank.backends.cuda.gpumm import Buffer, DeviceCache, ExpertPool, mem, mem_info, pinned, weak_method


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
        P.CPU_START: (0,), P.SIGNAL: (0, 3, 6, 9, 11), P.AWAIT: (0, 2), P.D2H: (1,), P.H2D: (0,),
        P.CPU_TASK: (0, 1, 2), P.MOE_COMBINE: (5,),     # (MOE_COMBINE: the rows of the CPU, pinned)
        P.DS_COMBINE: (3,), P.DS_EMBED: (0,),     # (DS_EMBED: the table in pinned host memory)
        P.DS_HC_POST: (12,), P.DS_HC_STEP: (2,)}  # (the CPU's sums of its experts, pinned)


def lib():
    """Return the library. Build and load it on the first call."""
    global _lib, _error
    if _lib is not None:
        return _lib
    if _error is not None:
        raise RuntimeError(_error)
    try:
        from reclank.backends.cuda import build as _cuda_build
        from reclank.backends.cuda.devices import prebuild
        prebuild()      # (the expert devices' libraries, beside this build)
        L = ctypes.CDLL(str(_cuda_build.build()))
    except Exception as exc:  # no nvcc, or the build failed
        _error = "the GPU library is not available: %s" % exc
        raise RuntimeError(_error) from exc
    vp, sz, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    L.rc_gpu_last_error.restype = ctypes.c_char_p
    L.rc_gpu_lane.argtypes = [i]
    L.rc_gpu_scratch_own.argtypes = [i]
    L.rc_gpu_lane_serve.argtypes = [vp]
    L.rc_gpu_where.argtypes = [vp]
    L.rc_gpu_where.restype = vp
    L.rc_gpu_init.argtypes = [i]
    L.rc_gpu_mem_info.argtypes = [ctypes.POINTER(sz), ctypes.POINTER(sz)]
    L.rc_gpu_malloc.argtypes = [sz]
    L.rc_gpu_malloc.restype = vp
    L.rc_gpu_free.argtypes = [vp]
    L.rc_gpu_h2d.argtypes = [vp, vp, sz]
    L.rc_gpu_h2d_rows.argtypes = [i, vp, vp, sz]
    L.rc_gpu_d2h.argtypes = [vp, vp, sz]
    L.rc_gpu_mix_stats_on.argtypes = [i]
    L.rc_gpu_mix_stats.argtypes = [vp, i]
    L.rc_gpu_load.argtypes = [vp, i]
    L.rc_gpu_load.restype = vp
    L.rc_gpu_run.argtypes = [vp, vp]
    L.rc_gpu_unload.argtypes = [vp]
    L.rc_gpu_prepare.argtypes = [vp]
    L.rc_gpu_clear_error.argtypes = []
    L.rc_gpu_profile.argtypes = [vp, vp, vp]
    L.rc_gpu_q6k_head.argtypes = [vp, vp, vp, i, i, ctypes.c_float, i]
    L.rc_gpu_q4_head.argtypes = [vp, vp, vp, i, i, ctypes.c_float, i]
    L.rc_gpu_e3_linear.argtypes = [vp, i, i, i, vp, i, vp, ctypes.c_float]
    L.rc_gpu_e3_linear_x16.argtypes = [vp, i, i, i, vp, i, vp]
    L.rc_gpu_topk.argtypes = [vp, i, i, i, ctypes.c_float, vp, vp, vp]
    L.rc_gpu_argmax_rows.argtypes = [vp, i, i, vp]
    L.rc_gpu_embed_q4.argtypes = [vp, vp, vp, i, ctypes.c_float]
    L.rc_gpu_embed_q4_rows.argtypes = [vp, vp, vp, i, i, ctypes.c_float]
    L.rc_gpu_set_cpu_runner.argtypes = [vp]
    L.rc_gpu_host_register_rw.argtypes = [vp, ctypes.c_size_t]
    L.rc_gpu_set_moe_rot.argtypes = [ctypes.c_int]
    L.rc_gpu_set_moe_rot.restype = None
    L.rc_gpu_set_moe_clamp.argtypes = [ctypes.c_int]
    L.rc_gpu_set_moe_clamp.restype = None
    L.rc_gpu_set_moe_contract.argtypes = [ctypes.c_int]
    L.rc_gpu_set_moe_contract.restype = None
    L.rc_gpu_set_kv_e4m3.argtypes = [ctypes.c_int]
    L.rc_gpu_set_kv_e4m3.restype = ctypes.c_int
    L.rc_gpu_set_pool.argtypes = [vp]
    L.rc_gpu_fence.restype = ctypes.c_int64
    L.rc_gpu_fence_done.argtypes = [ctypes.c_int64, i]
    L.rc_gpu_pool_dims.argtypes = [vp]
    L.rc_gpu_host_register_rw.restype = ctypes.c_int64
    L.rc_gpu_dev_addr.argtypes = [vp]
    L.rc_gpu_dev_addr.restype = ctypes.c_int64
    L.rc_gpu_task_log.argtypes = [i, vp, i]
    L.rc_gpu_set_tc.argtypes = [i]
    L.rc_gpu_host_alloc.argtypes = [sz]
    L.rc_gpu_cache_copy.argtypes = [vp, i]
    L.rc_gpu_host_register_ro.argtypes = [vp, ctypes.c_size_t]
    L.rc_gpu_host_register_ro.restype = i
    L.rc_gpu_cache_query.argtypes = [i, i]
    L.rc_gpu_gdn_commit.argtypes = [vp, vp, vp] + [i] * 7
    L.rc_gpu_host_alloc.restype = vp
    L.rc_gpu_d2d.argtypes = [vp, vp, sz]
    L.rc_gpu_ko_table.argtypes = [vp, vp, i, i, i]
    L.rc_gpu_zero.argtypes = [vp, sz]
    L.rc_gpu_caps_get.argtypes = [ctypes.POINTER(i), i]
    # The products of a large group: RECLANK_GPU_TC=1 (the default) runs
    # them on the tensor cores with float16 inputs, 8 with int8 inputs (the
    # Q8_0 form: faster, less exact), and 0 with float32 kernels.
    L.rc_gpu_set_tc(int(os.environ.get("RECLANK_GPU_TC", "1")))
    if L.rc_gpu_init(int(os.environ.get("RECLANK_GPU_DEVICE", "0"))) != 0:
        _error = L.rc_gpu_last_error().decode()
        raise RuntimeError(_error)
    # A GP_CPU_JOIN record runs a CPU program with gemma_run of the CPU library.
    from reclank.backends.cpu import cops
    if cops._lib is not None:
        # gemma_run_task: a team of RECLANK_GPU_CPU_THREADS bound spread;
        # by default half the cores on a machine of two or more NUMA nodes
        # (the 2-socket Xeon: Qwen3.8 26.0 tok/s, all 48 cores 21.3), else
        # OMP_NUM_THREADS
        n = os.environ.get("RECLANK_GPU_CPU_THREADS")
        if n is None:
            from reclank.runtime import numa
            n = int(os.environ.get("OMP_NUM_THREADS", "0") or 0) // 2 if numa.enabled() else 0
        cops._lib.rc_set_task_threads(ctypes.c_int(int(n)))
        L.rc_gpu_set_cpu_runner(ctypes.cast(cops._lib.rc_run_task, ctypes.c_void_p))
        _place_threads(L)
    _lib = L
    return L


CAPS_FIELDS = ("cc", "tc", "fp8_mma", "fp4_mma", "pdl", "nanosleep", "smem_block_max", "warp",
               "sm_count", "unified", "dp4a")


def caps():
    """The capabilities of the GPU (backends/cuda/csrc/runtime/caps.cu) as a
    dict: cc (86 for 8.6, or RECLANK_GPU_CC), tc (the tensor cores), ... The
    compilers read them for their limits."""
    L = lib()
    out = (ctypes.c_int * 32)()
    n = L.rc_gpu_caps_get(out, 32)
    return dict(zip(CAPS_FIELDS, out[:n]))


_TC = None


def has_tc():
    """True when the GPU has the tensor cores that the kernels use (sm_80 up;
    RECLANK_GPU_CC=61 says no on any GPU). Without them (Pascal) the
    compilers leave out the records that only have tensor-core kernels."""
    global _TC
    if _TC is None:
        _TC = bool(caps().get("tc", 1))
    return _TC


def int8_tiles():
    """True when the int8 products of large groups have a kernel: the tensor
    cores, or dp4a (sm_61 up, the Pascal kernels; RECLANK_GPU_DP4A=0 turns
    them off; Maxwell, RECLANK_GPU_CC below 61: no dp4a)."""
    return has_tc() or (bool(caps().get("dp4a", 0)) and os.environ.get("RECLANK_GPU_DP4A", "1") != "0")


def _place_threads(L):
    """The CPUs around the teams of gemma_run_task (gg_cpu_roles): the
    thread that loads the library (the thread of the model: serve_qwen4's
    model thread, else the main thread) alone on the first CPU of the GPU's
    node (RECLANK_MAIN_CPU; -1: no pin), the copy workers on its last; the
    teams leave both out (RECLANK_RESERVED_CPUS), so the master of the teams
    (the GPU runner, 84-91% of a CPU in a decode) takes a CPU of its own too.
    OMP_PROC_BIND close had put the model thread and the runner on CPU 0
    together (67% + 34% of a decode)."""
    if not hasattr(L, "rc_gpu_cpu_roles"):
        return
    copy_cpu, main_cpu = ctypes.c_int(-1), ctypes.c_int(-1)
    L.rc_gpu_cpu_roles(ctypes.byref(copy_cpu), ctypes.byref(main_cpu))
    from reclank.runtime import moe_parts
    if moe_parts.wanted() and main_cpu.value >= 0:
        # the parts mode (RECLANK_GPU_PARTS): the model thread and the copy
        # workers on the two highest CPUs of the taskset on the GPU's node;
        # the teams of the parts leave them out (moe_parts.part_cpus)
        from reclank.runtime import numa
        helpers = moe_parts.helper_cpus(numa.node_of_cpu().get(main_cpu.value, 0))
        if helpers:
            os.environ.setdefault("RECLANK_MAIN_CPU", str(helpers[0]))
            os.environ.setdefault("RECLANK_GPU_COPY_CPU", str(helpers[1]))
            copy_cpu = ctypes.c_int(int(os.environ["RECLANK_GPU_COPY_CPU"]))
    m = int(os.environ.get("RECLANK_MAIN_CPU", main_cpu.value))
    rsv = [c for c in (m, copy_cpu.value) if c >= 0]
    if rsv:
        os.environ.setdefault("RECLANK_RESERVED_CPUS", ",".join(str(c) for c in rsv))
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
        raise RuntimeError(lib().rc_gpu_last_error().decode() + where_text())


# The GPUPrograms by handle (where: the program of the last run).
_PROGS = weakref.WeakValueDictionary()


def _op_names():
    from reclank.ir import program as P
    names = {}
    for k, v in vars(P).items():
        if k.isupper() and isinstance(v, int) and not k.startswith("T_"):
            names.setdefault(v, k)
    return names


def where():
    """The place of the last run (gg_where; no CUDA call, so it works after
    a lost context): the program (its label, e.g. ("mix_progs", 256)), the
    record that the host queued last, its operation, the segment of the
    launch, and sync (RECLANK_GPU_SYNC_CHECK). A kernel error shows up at a
    record at or after the one at fault. None before the first run."""
    if _lib is None:
        return None
    out = (ctypes.c_int * 5)()
    h = _lib.rc_gpu_where(out)
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
        self.handle = lib().rc_gpu_load(buf.ctypes.data, (1 if graph else 0) | (0 if tc else 2) | (4 if i8 else 0) |
                                  (8 if atc else 0) | (16 if i8 == 2 else 0))
        if not self.handle:
            msg = lib().rc_gpu_last_error().decode()
            lib().rc_gpu_clear_error()
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
            _check(lib().rc_gpu_run(self.handle, self.env.ctypes.data))
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
        _check(lib().rc_gpu_profile(self.handle, self.env.ctypes.data, ms.ctypes.data))
        return ms

    def prepare(self):
        """Record the CUDA graphs now and upload them (the first run does it
        otherwise). Raise MemoryError when the GPU has no room for them."""
        if self.handle and lib().rc_gpu_prepare(self.handle) != 0:
            msg = lib().rc_gpu_last_error().decode()
            lib().rc_gpu_clear_error()
            raise MemoryError(msg)

    def close(self):
        if self.handle:
            lib().rc_gpu_unload(self.handle)
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


class WarmPool:
    """The pool of the warm experts of HotCache (gpumm.ExpertPool) of a
    model on the GPU: one cache system for the Qwen and the Gemma models.

    The hot experts of a layer are blocks of the pool too (its "hot"
    segments, _pool_place), and the pool takes more memory for warm
    experts: "lend" the room that the image encoder lends (lend_warm), and
    the memory that the cache lends (the rows past those in use: "shared");
    with warm_mode "free" also the free memory at a decode step (_warm_on).
    The programs, the cache and a new prompt take the memory back: GpuMem
    frees the free segments (rank 10), then the hot ones (rank 15), the
    blocks with the fewest warm experts first, before the programs in use.

    The class sets self.per (the bytes of each of the 3 parts of an expert;
    0 for a part it has not), self.warm_mode, self.pool (None) and
    self.hot_cache, and calls _warm_reclaimers() after _mm_init, and
    _warm_close() when it closes."""

    warm_mode = "lend"
    # warm_mode "free": the free memory the pool leaves at a decode step:
    # None the reserve of GpuMem (GpuMem.room), or these bytes (into the
    # reserve: the free segments go first when a Buffer needs it); when its
    # free segments and the free memory are less than warm_min bytes,
    # _warm_programs frees the programs of the prompts first
    warm_keep = None
    warm_min = 0
    # True: _warm_programs also when the free segments and the free memory
    # are less than the blocks of the experts the GPU does not hold (the end
    # of a prompt on a card too small for all of them: the warm experts are
    # worth more to the decode than the programs to the next prompt, which
    # makes them again); a card that holds them all keeps the programs
    warm_short = False
    # True: a decode step adds free segments whenever the memory has room for
    # one (the memory a prompt let go); False: only when the pool has no free
    # segment (they stay until something needs the memory)
    warm_regrow = False

    def _warm_programs(self):
        """Free the programs (and buffers) of the prompts for the pool: none
        by default."""

    def _warm_prompt_held(self):
        """True while programs (or buffers) of a prompt that _warm_programs
        frees are held: none by default."""
        return False

    def _warm_need(self, hc, pool):
        """The bytes of segments for the experts the GPU does not hold, less
        the free blocks (the 26B on the 3090 holds 3766 of its 3840 experts:
        421 segments, 11 GB, made at the first decode step for nothing, then
        freed by the next prompt)."""
        need = int((~hc.held[hc.pool_rows]).sum()) - int((pool.owner == -1).sum())
        return max(0, -(-need // pool.K)) * pool.seg_bytes

    def _warm_want(self, hc, pool):
        """The bytes of free segments worth adding: with warm_regrow those
        of _warm_need, else all there is."""
        return self._warm_need(hc, pool) if self.warm_regrow else float("inf")

    def _warm_hc(self):
        """The HotCache that uses the pool (the slot tables point into its
        blocks even when a test takes hot_cache away), or None."""
        hc = getattr(self, "warm_owner", None) or self.hot_cache
        return hc if hc is not None and hc.pool is not None else None

    def _ensure_pool(self):
        """The pool of HotCache (blocks of the size of the experts of the
        layers), made once."""
        if self.pool is None:
            self.pool = ExpertPool(self.per)
        hc = self.hot_cache
        if hc is not None and hc.pool is None:
            hc.set_pool(self.pool)
            if self.warm_mode in ("0", "off"):
                hc.pool_rows[:] = False         # the copies of the prompts only
            self.warm_owner = hc
        # the lent room first: release_warm gives exactly it back
        lend = getattr(self, "warm_lend", 0)
        if lend > 0 and not self.pool.has("lend"):
            self.pool.grow(lend, "lend")
        return self.pool

    def lend_warm(self, nbytes):
        """Room that the image encoder lends (serve_qwen4 --mmproj-gpu lend):
        nbytes of the GPU for the pool until release_warm. Its segments come
        at the next decode step (_warm_on)."""
        self.warm_lend = int(nbytes)

    def release_warm(self):
        """Give the lent room back (an image to encode): the warm experts of
        its segments go cold, and they are freed. The next decode step takes
        it again."""
        hc = self._warm_hc()
        if hc is not None:
            hc.pool_drop("lend")
        elif self.pool is not None:
            self.pool.drop(self.pool.segments("lend"))

    def free_ring(self):
        """Give back all the segments of the pool from the free memory
        (ExpertPool "free"): their warm experts go cold. For a user outside
        GpuMem (an image encoder that allocates with torch or cudaMalloc);
        the Buffers of GpuMem reclaim them as they need. The next mixed group
        (or decode step) takes the free memory again."""
        hc = self._warm_hc()
        if hc is not None:
            hc.pool_drop("free")
        elif self.pool is not None:
            self.pool.drop(self.pool.segments("free"))

    def _grow_shared(self):
        """Segments of the pool in the memory that the cache lends (GpuMem
        shared: the rows past those in use and a margin)."""
        pool = self.pool
        if pool is None:
            return
        k = mem().shared_room(pool.seg_bytes)
        if k:
            pool.grow(k * pool.seg_bytes, "shared")

    def _warm_on(self):
        """Before a decode step, a small group, or the MTP layer: the blocks
        that the last mixed group reserved go back to the pool, the pool
        takes the lent room (lend_warm) and the room the cache lends, and
        with warm_mode "free" the free memory (less the reserve of GpuMem,
        without freeing programs for it). Its segments of free memory stay
        until something needs the memory."""
        hc = self.hot_cache
        if hc is None:
            return
        pool = self._ensure_pool()
        hc.pool_unreserve()
        self._grow_shared()
        if self.warm_mode == "free" and self.warm_short and self._warm_prompt_held() and \
                pool.count.get("free", 0) * pool.seg_bytes + mem_info()[0] < self._warm_need(hc, pool):
            # the end of a prompt (warm_short), also when the pool has free
            # segments (those the mixed groups of the prompt took): their
            # memory goes to the pool at once
            self._warm_programs()
            room = mem().room() if self.warm_keep is None else max(0, mem_info()[0] - self.warm_keep)
            room = min(room, self._warm_need(hc, pool))
            if room >= pool.seg_bytes:
                pool.grow(room, "free")
        if self.warm_mode == "free" and (self.warm_regrow or not pool.has("free")):
            # (the cheap test first: on the 3090 the 26B wants no block in
            # most steps, and the query of the free memory waits for nothing)
            want = self._warm_want(hc, pool)
            if want < pool.seg_bytes:
                return

            def avail():
                return mem().room() if self.warm_keep is None else max(0, mem_info()[0] - self.warm_keep)
            if pool.count.get("free", 0) * pool.seg_bytes + avail() < self.warm_min:
                # (the pool and the free memory: a pool that has taken the
                # memory already keeps the programs)
                self._warm_programs()
            want = min(avail(), want)
            if want >= pool.seg_bytes:
                pool.grow(want, "free")

    def _reclaim_pool(self, nbytes, keep):
        """GpuMem: the free-memory segments of the pool, those with the
        fewest warm experts first."""
        hc = self._warm_hc()
        if hc is not None:
            return hc.pool_shrink("free", nbytes)
        pool = self.pool
        if pool is None:
            return 0
        segs = [sg for sg in pool.segments("free")
                if not (pool.owner[sg * pool.K:(sg + 1) * pool.K] == -2).any()
                and not pool.bufs[sg].locks][:-(-int(nbytes) // pool.seg_bytes)]
        pool.drop(segs)
        return len(segs) * pool.seg_bytes

    def _reclaim_hot(self, nbytes, keep):
        """GpuMem: segments of the hot experts, those with the fewest warm
        experts first (after the free segments and the stale programs,
        before the programs in use: see ProgramLRU.STALE)."""
        hc = self._warm_hc()
        return hc.pool_shrink("hot", nbytes) if hc is not None else 0

    def _warm_reclaimers(self):
        mem().add_reclaimer(10, self._mm_name + "-pool", weak_method(self._reclaim_pool))
        mem().add_reclaimer(15, self._mm_name + "-hot", weak_method(self._reclaim_hot))

    def _warm_close(self):
        """Give back the pool (the warm experts go cold) and the reclaimers."""
        hc = self._warm_hc()
        if hc is not None:
            hc.pool_drop()
        if self.pool is not None:
            self.pool.close()
        mem().remove_reclaimer(self._mm_name + "-pool")
        mem().remove_reclaimer(self._mm_name + "-hot")

    def _pool_place(self, entries):
        """The first hot experts in the pool: entries (key, init, parts,
        slots) give the experts init of each layer (parts: host address of
        expert 0 or of each expert, bytes, device store); its "hot"
        segments get a block for each, the parts copied there, and its slot
        HotCache.WARM + block (owner -4: held; HotCache sets the owner).
        Return (key, expert, block) of each, in order (fewer when the
        memory runs out)."""
        need = sum(len(init) for _k, init, _p, _s in entries)
        placed = []
        if not need:
            return placed
        pool = self.pool
        pool.grow(-(-need // pool.K) * pool.seg_bytes, "hot")
        free = list(np.flatnonzero(pool.owner == -1))
        for key, init, parts, slots in entries:
            for x in init:
                if not free:
                    return placed
                b = int(free.pop(0))
                for part, (srcs, nb, _dst) in enumerate(parts):
                    a = int(srcs[int(x)]) if isinstance(srcs, np.ndarray) else srcs + int(x) * nb
                    _check(lib().rc_gpu_h2d(pool.addr(part, b), a, nb))
                slots[x] = HotCache.WARM + b
                pool.owner[b] = -4
                placed.append((key, int(x), b))
        return placed


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

    def layer_begin(self, i):
        self.pool_n = {}
        self.in_layer = True

    def layer_end(self, i):
        self.in_layer = False

    def buffer(self, shape, dtype=np.float32):
        if not (self.pool_on and self.in_layer):
            return super().buffer(shape, dtype)
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        k = (shape, np.dtype(dtype).str)
        n = self.pool_n.get(k, 0)
        self.pool_n[k] = n + 1
        b = self.pool.get((k, n))
        if b is None:
            b = self.pool[(k, n)] = P.lazy_zeros(shape, dtype)
            self.p.temps.append(b)
        return b


# ---- the 26B model: the experts on the CPU (SPLIT_PLAN.md, phase 4) -----------

def expert_host(model, layer):
    """The bytes of the experts of a layer that the GPU reads (its hot slots,
    the copies of GP_FETCH and of HotCache): (gate and up, down, q4x), each
    (experts, bytes). q4x: the KQ_Q4X copies (ops.q4x_pack_model; the bytes
    of an expert are those of the int4 expert, in groups of 16 rows), unless
    RECLANK_GPU_Q4X=0; the GPU kernels then read that layout (GP_HOT_MOE
    and GP_MOE_GPU with the flag). Without the int8 products (Maxwell: no
    tensor cores, no dp4a) the GPU takes the int4 layout, whose float
    kernels it has (int4_row, k_moe_gemm); the CPU keeps its KQ_Q4X copies."""
    w = model._layers[layer]
    gu, dn = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
    n = gu.shape[0]
    e = P.ops._Q4X_MOE.get(gu.ctypes.data)
    if e is not None and os.environ.get("RECLANK_GPU_Q4X", "1") != "0" and int8_tiles():
        return e[0].reshape(n, -1), e[1].reshape(n, -1), True
    return gu.reshape(n, -1), dn.reshape(n, -1), False


def fill_tables(model, layer, hot, stage, tgu, tdn, ranges, gslots=None, pre=None, pool=None, buf=None):
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
    gslots. gslots (or None) gets 0 or more for the experts on the GPU. A
    warm expert (slot HotCache.WARM + block) has the address of its block
    of pool (the tables change when a block moves: table_version).

    The buffers hold the cold experts of the layer with the most when they
    were made (ModelGPU._stage); the copies may not pass their end (a long
    cache took memory from the hot experts since, so more are cold: the 26B
    with a cache of 131072 tokens on the GTX 1070 copied past it and read
    there). A mixed group asks for at most stage_room; a group of all the
    experts with more cold ones than that raises MemoryError. buf (or None:
    layer % 2) is the buffer of the copies (a piece of fill_chunks)."""
    gu, dn, _q4x = expert_host(model, layer)
    n = gu.shape[0]
    egu, edn = gu.nbytes // n, dn.nbytes // n
    dgu, ddn = stage[layer % 2 if buf is None else buf]
    room = min(dgu.nbytes // egu, ddn.nbytes // edn)
    slots, hgu, hdn = hot if hot is not None else (np.full(n, -1), 0, 0)
    runs = []
    rank = 0
    if gslots is not None:
        gslots[:] = -1
    for x in range(n):
        if slots[x] >= HotCache.WARM:
            tgu[x] = pool.addr(0, int(slots[x]) - HotCache.WARM)
            tdn[x] = pool.addr(1, int(slots[x]) - HotCache.WARM)
            if gslots is not None:
                gslots[x] = slots[x]
            continue
        if slots[x] >= 0:
            tgu[x] = hgu + int(slots[x]) * egu
            tdn[x] = hdn + int(slots[x]) * edn
            if gslots is not None:
                gslots[x] = slots[x]
            continue
        if pre is not None and x not in pre:
            tgu[x] = tdn[x] = 0
            continue
        if rank >= room:
            raise MemoryError("layer %d: more cold experts to copy than the %d places of the buffers" % (layer, room))
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


def fill_chunks(model, layer, hot, stage, chunks, pool=None):
    """Fill the tables of the pieces of a layer of a large group whose
    buffers hold fewer experts than the layer has (chunked staging,
    SplitCompiler.moe_group_chunks). chunks is a list of (tgu, tdn, ranges,
    gslots, k) for each piece: the cold experts of the layer go to the
    pieces in even runs (each at most stage_room), piece c to buffer k % 2
    (k: its fetch); the hot and warm experts are those of piece 0. Raise
    MemoryError when the pieces have too few places."""
    n = expert_host(model, layer)[0].shape[0]
    slots = hot[0] if hot is not None else np.full(n, -1)
    cold = [x for x in range(n) if slots[x] < 0]
    C = len(chunks)
    room = stage_room(model, stage)
    if len(cold) > C * room:
        raise MemoryError("layer %d: %d cold experts, %d pieces of %d places" % (layer, len(cold), C, room))
    per = -(-len(cold) // C)
    for c, (tgu, tdn, ranges, gslots, k) in enumerate(chunks):
        fill_tables(model, layer, hot if c == 0 else None, stage, tgu, tdn, ranges, gslots,
                    pre=set(cold[c * per:(c + 1) * per]), pool=pool if c == 0 else None, buf=k % 2)


def stage_room(model, stage):
    """The cold experts that the buffers of a large group hold (fill_tables)."""
    gu, dn, _q4x = expert_host(model, 0)
    n = gu.shape[0]
    return min(min(b[0].nbytes for b in stage) // (gu.nbytes // n), min(b[1].nbytes for b in stage) // (dn.nbytes // n))


# The largest group whose experts run on the CPU (GP_MOE_MT of the CPU
# interpreter takes at most 16 tokens).
MT_CPU = 16


def _argmax_rows(g, rows):
    """The best token of each row of g.out (the logits that the head just
    made), as np.argmax (gg_argmax_rows): rows integers cross to the host, not
    rows of the vocabulary."""
    vocab = g.host_logits.shape[1]
    if getattr(g, "argmax_buf", None) is None:
        g.argmax_buf = Buffer(4 * MT_CPU, "program")
        g.argmax_host = pinned((MT_CPU,), np.int32)
    _check(lib().rc_gpu_argmax_rows(g.out.ptr, rows, vocab, g.argmax_buf.ptr))
    _check(lib().rc_gpu_d2h(g.argmax_host.ctypes.data, g.argmax_buf.ptr, 4 * rows))
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
    _check(lib().rc_gpu_topk(g.out.ptr, rows, vocab, k, inv_t, base, base + 4 * n, base + 8 * n))
    h = g.topk_host
    _check(lib().rc_gpu_d2h(h.ctypes.data, base, 8 * n + 8 * rows))
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


# The most copy jobs of HotCache on the way (the queue of the GPU holds 64:
# experts.cu GG_CACHE_JOBS).
HOT_PENDING_MAX = 48

# RECLANK_GPU_HOT_C (1): HotCache.score_group and the choice of _replace in C
# (backends/cpu/csrc/common/hotcache.c: the same sums and orders, so the same
# slots); 0 the NumPy code (a test).
HOT_C = os.environ.get("RECLANK_GPU_HOT_C", "1") != "0"
# RECLANK_GPU_H2D_ROWS (1): the slot tables HotCache changes in one copy (rc_gpu_h2d_rows); 0 a copy
# each (a test).
H2D_ROWS = os.environ.get("RECLANK_GPU_H2D_ROWS", "1") != "0"


def _hot_lib():
    """The CPU library with the HotCache functions, or None (HOT_C off, or no library)."""
    if not HOT_C:
        return None
    from reclank.backends.cpu import cops
    lb = cops._lib
    if lb is None or not hasattr(lb, "rc_hot_score"):
        return None
    if lb.rc_hot_score.argtypes is None:
        vp, i = ctypes.c_void_p, ctypes.c_int
        lb.rc_hot_score.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, i, i, i, i, ctypes.c_double, i,
                                    i, vp, i, vp]
        lb.rc_hot_replace.argtypes = [vp, vp, vp, vp, vp, ctypes.c_int64, vp, vp, vp, i, i, i, i,
                                      ctypes.c_int64, vp, vp]
    return lb


class LaneRunner:
    """The thread of lane B (docs/LANES_PLAN.md): a stream of its own
    (rc_gpu_lane), and the run of a program there while the model thread runs
    lane A's (run_pair). Its CPU tasks run on the model thread."""

    def __init__(self, cpus=None):
        self.cpus = cpus        # its CPUs (not those of the parts' teams, not the model thread's)
        self.done = np.zeros(1, np.int64)       # set after lane B's run and its stream
        self.job = self.err = None
        self.go, self.fin = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="lane-b")
        self.thread.start()

    def _loop(self):
        if self.cpus:
            # (a thread starts with the affinity of the one that made it: the model
            # thread, pinned to one CPU by the parts; two spinning threads on one
            # CPU took a time slice for each hand-off, 5.5 ms a layer)
            os.sched_setaffinity(0, self.cpus)
        lib().rc_gpu_lane(1)
        while True:
            self.go.wait()
            self.go.clear()
            g = self.job
            if g is None:
                return
            try:
                if callable(g):
                    # (a job of Python: run_pair_job)
                    self.result = g()
                else:
                    _check(lib().rc_gpu_run(g.handle, g.env.ctypes.data))
                _check(lib().rc_gpu_lane_sync())
            except BaseException as exc:      # (the model thread raises it)
                self.err = exc
            self.done[0] = 1
            self.fin.set()

    def solo(self, g, extra=()):
        """Run g on lane B's stream with nothing else (its first run records
        its graphs there; no capture while another stream launches)."""
        mem().poll()
        blocks = run_blocks_of(g) + list(extra)
        mem().lock(blocks)
        self._start(g)
        _check(lib().rc_gpu_lane_serve(self.done.ctypes.data))
        self.fin.wait()
        g.fence = mem().fence(blocks)
        if self.err is not None:
            raise self.err

    def _start(self, g):
        s = g.prog.by_name.get("seq") if not callable(g) else None
        if s is not None:
            g.seq = getattr(g, "seq", 0) + 1
            g.env[s.index] = g.seq
        self.result = None
        self.done[0] = 0
        self.err = None
        self.job = g
        self.fin.clear()
        self.go.set()

    def close(self):
        self.job = None
        self.go.set()
        self.thread.join(timeout=5)


def run_blocks_of(g):
    """GpuMem: the blocks a run of g uses (its parameters, its model's)."""
    return list(g.named.values()) + (g.run_blocks() if g.run_blocks is not None else [])


def run_pair(ga, gb, runner, locked):
    """Run ga (lane A) on the model thread and gb (lane B) on runner's thread
    at once: the GPU work of the two on two streams, their CPU tasks in turn
    on the model thread (rc_gpu_lane_serve after ga). locked: the blocks of
    both (run_blocks_of), locked by the caller right after each bind (the
    parameters of one move if the bind of the other reclaims memory: the
    run then read the old place, an illegal address); they unlock when the
    GPU is done. Lane B's stream is done when this returns."""
    ga._alive()
    gb._alive()
    s = ga.prog.by_name.get("seq")
    if s is not None:
        ga.seq = getattr(ga, "seq", 0) + 1
        ga.env[s.index] = ga.seq
    blocks = locked
    runner._start(gb)
    err = None
    try:
        _check(lib().rc_gpu_run(ga.handle, ga.env.ctypes.data))
    except RuntimeError as exc:
        err = exc
    try:
        _check(lib().rc_gpu_lane_serve(runner.done.ctypes.data))
    except RuntimeError as exc:
        err = err or exc
    runner.fin.wait()
    ga.fence = gb.fence = mem().fence(blocks)
    if err is not None:
        raise err
    if runner.err is not None:
        raise runner.err


def run_pair_job(ga, job, runner, locked):
    """run_pair with a job of Python on runner's thread in place of a program: job() runs there (its
    GPU work on lane B's stream, its CPU tasks on the model thread) and returns (the programs it ran
    whose last run the fence covers, the blocks it locked); ga runs on the model thread at the same
    time. The job must not allocate, free, or change what the model thread's run reads (HotCache's
    slots). Return what job returned."""
    ga._alive()
    s = ga.prog.by_name.get("seq")
    if s is not None:
        ga.seq = getattr(ga, "seq", 0) + 1
        ga.env[s.index] = ga.seq
    runner._start(job)
    err = None
    try:
        _check(lib().rc_gpu_run(ga.handle, ga.env.ctypes.data))
    except RuntimeError as exc:
        err = exc
    try:
        _check(lib().rc_gpu_lane_serve(runner.done.ctypes.data))
    except RuntimeError as exc:
        err = err or exc
    runner.fin.wait()
    progs, blocks = runner.result if runner.result is not None else ((), [])
    f = mem().fence(list(locked) + list(blocks))
    ga.fence = f
    for g in progs:
        g.fence = f
    if err is not None:
        raise err
    if runner.err is not None:
        raise runner.err
    return runner.result


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

    At most max_ins experts change in each step. RECLANK_GPU_HOT_DECAY and
    RECLANK_GPU_HOT_INS change decay and max_ins. See SPLIT_PLAN.md.
    """

    # (rows x experts) bool, or None: the experts an expert device holds (set in __init__ and by
    # devices.py; here too for the caches made without __init__, as scripts/check_hotcache.py does)
    exclude = None

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
        self.decay = float(decay or os.environ.get("RECLANK_GPU_HOT_DECAY", "0.97"))
        self.max_ins = int(max_ins or os.environ.get("RECLANK_GPU_HOT_INS", "8"))
        # A cold expert goes to the GPU only from its admit-th use while it
        # is cold. Its first uses run on the CPU.
        self.admit = int(os.environ.get("RECLANK_GPU_HOT_ADMIT", "2"))
        # (rows x experts) bool, or None: experts another GPU holds (an expert
        # device, devices.py), which the slots do not take: their selections
        # do not count (exclusive caching: the slots go to experts that would
        # run on the CPU)
        self.exclude = None
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
        # The uses of each expert in this conversation (new_conversation sets
        # them to 0): with run, they decide the copies in place of the score
        # that decays (a cold expert takes the place of a held one only when
        # the conversation used it more), so the slots stay with the experts
        # the conversation uses and change less. RECLANK_GPU_HOT_RUN (1/0)
        # sets it; else the model's hot_run (the Gemma 26B: on).
        self.run = np.zeros(self.score.shape, dtype=np.float32)
        env = os.environ.get("RECLANK_GPU_HOT_RUN")
        self.run_mode = (env != "0") if env is not None else bool(getattr(dev, "hot_run", False))
        # The selections of the routers of a group (GP_COUNT), for all the
        # layers; seed() reads the rows of the layers with slots.
        self.counts = np.zeros((cfg.num_hidden_layers, cfg.num_experts), dtype=np.int32)
        self.count_rows = [e["layer"] for e in self.layers]
        self.last = None        # the counts of the last large group
        # The experts that a prompt pass can change; 0 (the default) only
        # reads the counts. A test gave no gain (SPLIT_PLAN.md).
        self.seed_ins = int(os.environ.get("RECLANK_GPU_HOT_SEED", "0"))
        self.rows = np.arange(m)[:, None]
        self.pending = []       # (job id, [(row, expert, slot)], ranges, locked segments)
        self.skipped = 0        # the passes of copies skipped with the queue full (_replace)
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

    def _key(self):
        """What the copies compare: the uses of the conversation (run_mode)
        or the score that decays."""
        return self.run if self.run_mode else self.score

    def new_conversation(self):
        """A new conversation (a new cache on the GPU): its uses start at 0."""
        self.run[:] = 0

    def set_pool(self, pool):
        """Use the blocks of pool (an ExpertPool) for warm experts: a row whose
        experts fit takes a free block, or the block of the warm expert of
        the lowest score of any row, when its new expert scores higher."""
        self.pool = pool
        pool.evict_hook = self.pool_evict
        # (3 parts of an expert of Qwen: gate, up, down; 2 of Gemma: gate
        # and up, down)
        self.pool_rows = np.array([1 < len(e["parts"]) <= 3 and all(
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
            out = held[np.argsort(self._key()[r, x], kind="stable")[:n - free.size]]
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
        # copies use them) or locked (a run, a copy). The runs that are done
        # unlock theirs first; with none to give then, wait for the GPU: the
        # next reclaimers free programs, and a program built after them can
        # free the last one, whose output logits() still reads (with
        # RECLANK_GPU_WARM=free the head after a short prompt freed it, and
        # the copy from QwenGPU.last crashed)
        def ready():
            return [sg for sg in pool.segments(src)
                    if not (pool.owner[sg * pool.K:(sg + 1) * pool.K] == -2).any()
                    and not pool.bufs[sg].locks]
        mem().poll()
        segs = ready()
        if not segs and pool.has(src):
            # the copies into its blocks hold their segments until they land
            # (the 26B on 4 GB: a prompt after a decode found every free
            # segment locked by a copy, and its program failed)
            self._land(wait=True)
            mem().poll(wait=True)
            segs = ready()
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
        """The slot tables of rows to the GPU (one call: rc_gpu_h2d_rows)."""
        if not rows:
            return
        dst, src, nb = [], [], None
        for r in rows:
            e = self.layers[r]
            a = e["slots"]
            sp = e.get("_sp")
            if sp is None or sp[0] is not a:      # (the host address, kept: .ctypes.data costs us)
                sp = e["_sp"] = (a, a.ctypes.data)
            dst.append(e["dslots"])
            src.append(sp[1])
            nb = a.nbytes if nb is None or nb == a.nbytes else -1
        if nb > 0 and H2D_ROWS:
            d, s_ = np.array(dst, np.uint64), np.array(src, np.uint64)
            _check(lib().rc_gpu_h2d_rows(len(dst), d.ctypes.data, s_.ctypes.data, nb))
        else:
            for r in rows:
                e = self.layers[r]
                _check(lib().rc_gpu_h2d(e["dslots"], e["slots"].ctypes.data, e["slots"].nbytes))
        self.version += 1

    def prepare(self, wait=False):
        """Before a step or a group: mark the experts of the finished copies
        hot. wait waits for all the copies."""
        if self.due:
            self.observe()
        self._land(wait)

    def _land(self, wait=False):
        """Mark the experts of the finished copies hot (wait: all of them),
        and unlock their segments of the pool; no new copies."""
        if not self.pending:
            return
        done, still = set(), []
        for job, items, ranges, segs in self.pending:
            r = lib().rc_gpu_cache_query(job, 1 if wait else 0)
            if r < 0:
                raise RuntimeError(lib().rc_gpu_last_error().decode())
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
        experts. Saved, it is RECLANK_EXPERT_COUNTS of a later start."""
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
        if self.exclude is not None:
            sels = sels.copy()
            ex = self.exclude[rows]
            for j in range(n):
                s = sels[j]
                s[(s >= 0) & np.take_along_axis(ex, np.maximum(s, 0), axis=1)] = -1
        lb = _hot_lib()
        if lb is not None:
            p = self._cptrs()
            rows = np.ascontiguousarray(rows, np.int64)
            sels = np.ascontiguousarray(sels, np.int32)
            nc = lb.rc_hot_score(p["score"], p["uses"], p["total"], p["run"], p["held"], p["incoming"],
                                 rows.ctypes.data, sels.ctypes.data, n, rows.size, self.score.shape[1],
                                 sels.shape[2], self.decay, int(self.run_mode), int(self._pool_free()),
                                 p["pool_rows"], self.admit, p["cand"])
            if nc and self._queue_room():
                self._replace_c(lb, p, rows, self.max_ins * n)
            return
        score, d = self.score, self.decay
        sub = score[rows]
        sub *= d ** n
        R, E = sub.shape
        # (one flat index of each selection: the scores in the order of the tokens, as n calls of
        # score_rows; the counts of uses at once, a bincount: 2 ms of Python a verify group before)
        ok = sels >= 0
        flat = (np.arange(R)[None, :, None] * E + sels)
        fs = sub.reshape(-1)
        for j in range(n):
            fs[flat[j][ok[j]]] += d ** (n - 1 - j)       # the experts of a row differ
        cnt = np.bincount(flat[ok], minlength=R * E).reshape(R, E)
        uses = self.uses[rows] + cnt.astype(self.uses.dtype)
        self.total[rows] += cnt.astype(self.total.dtype)
        self.run[rows] += cnt.astype(self.run.dtype)
        hit = cnt > 0
        held = self.held[rows]
        uses[held] = 0
        score[rows] = sub
        self.uses[rows] = uses
        if self.run_mode:
            sub = self.run[rows]
        low = np.where(held, sub, np.inf).min(axis=1)
        if self._pool_free():
            # a row that can take a free block of the pool takes the next
            # best experts too
            low[self.pool_rows[rows]] = -np.inf
        cand = hit & ~held & ~self.incoming[rows] & (sub > low[:, None]) & (uses >= self.admit)
        if not cand.any():
            return
        self._replace(cand, self.max_ins * n, rows)

    def seed(self, g, t, prompt):
        """After a group of t tokens (a part of a prompt, or an MTP verify
        group): add the selections of its routers to the scores, as if its
        tokens were the last t steps, then change the slots. After a part of
        a prompt, up to seed_ins experts change (not max_ins), and one use is
        enough, so the slots fit the prompt before the decode; the copies go
        on during the decode. With RECLANK_GPU_HOT_SEED=0 (the default) a
        prompt changes nothing. An MTP verify group always counts: MTP runs
        no decode steps, so without it the slots would not change."""
        if not self.layers:
            return
        dev = g.mirror.buffer_of(self.counts).ptr
        _check(lib().rc_gpu_d2h(self.counts.ctypes.data, dev, self.counts.nbytes))
        if t > MT_CPU:
            self.last = self.counts.copy()      # for ModelGPU.plan_mix
        c = self.counts[self.count_rows].astype(np.float32)
        self.total += self.counts[self.count_rows]
        self.run += c
        self.counts[:] = 0
        _check(lib().rc_gpu_h2d(dev, self.counts.ctypes.data, self.counts.nbytes))
        if prompt and self.seed_ins == 0:
            return
        # Each of the t tokens gets the mean weight that the steps would give.
        f = self.decay ** t
        w = (1.0 - f) / ((1.0 - self.decay) * t)
        self.score *= f
        self.score += w * c
        self.uses += c.astype(np.int32)
        self.uses[self.held] = 0
        key = self._key()
        low = np.where(self.held, key, np.inf).min(axis=1)
        if self._pool_free():
            low[self.pool_rows] = -np.inf
        cand = (c > 0) & ~self.held & ~self.incoming & (key > low[:, None]) & \
            (self.uses >= (1 if prompt else self.admit))
        if cand.any():
            self._replace(cand, self.seed_ins if prompt else self.max_ins)

    @staticmethod
    def _low_first(ps, k):
        """The indices of ps in the order of np.argsort(ps, kind="stable"),
        at least its first k (all those up to the k-th smallest value, ties
        too), and whether that is all of them: a partition and a sort of a
        few, in place of a sort of the whole pool (0.4 ms a call)."""
        if k >= ps.size:
            return np.argsort(ps, kind="stable"), True
        v = np.partition(ps, k - 1)[k - 1]
        idx = np.flatnonzero(ps <= v)
        return idx[np.argsort(ps[idx], kind="stable")], idx.size == ps.size

    def _replace(self, cand, limit, cand_rows=None):
        """Move up to limit experts of the mask cand (rows, experts) to the
        slots of the lowest held experts, best gain first. cand_rows: cand
        holds only those rows, in their order."""
        if not self._queue_room():
            return
        score = self._key()
        lb = _hot_lib()
        if lb is not None:
            rows_ = np.arange(self.held.shape[0], dtype=np.int64) if cand_rows is None else \
                np.ascontiguousarray(cand_rows, np.int64)
            if np.unique(rows_).size == rows_.size:
                p = self._cptrs()
                p["cand_a"][:rows_.size] = cand
                self._replace_c(lb, p, rows_, limit)
                return
        if cand_rows is not None and np.any(np.diff(cand_rows) <= 0):
            full = np.zeros_like(self.held)     # (the order of the whole table)
            full[cand_rows] = cand
            cand, cand_rows = full, None
        if cand_rows is None:
            ri, xi = np.nonzero(cand)
        else:
            si, xi = np.nonzero(cand)
            ri = np.asarray(cand_rows)[si]
        if ri.size == 0:
            return
        # the lowest held score of the rows of the candidates only (ri is sorted, from nonzero:
        # the unique rows and the inverse of np.unique from where it changes, no sort)
        step = np.empty(ri.size, dtype=bool)
        step[0] = True
        np.not_equal(ri[1:], ri[:-1], out=step[1:])
        ur, inv = ri[step], np.cumsum(step) - 1
        hs = np.where(self.held[ur], score[ur], np.inf)
        low = hs.min(axis=1)
        order = np.argsort(low[inv] - score[ri, xi], kind="stable")
        # the held expert of the lowest score of each of those rows (as the loop's np.where and
        # argmin for a row whose slots this call has not changed)
        va = hs.argmin(axis=1)
        urow = {int(r): k for k, r in enumerate(ur.tolist())}
        items, changed = [], set()
        pool, E = self.pool, score.shape[1]
        # The free blocks of the pool, and its warm experts by score, once for
        # the call (the scores do not change in it; a block that a copy takes
        # gets an owner with a copy on the way, which the walk skips): the
        # same choices as a pass over the pool for each candidate, which took
        # 1.7 ms a call (2.9 ms a token of MTP decode with the NVFP4 experts).
        free, pv, pk = [], None, 0
        use_pool = pool is not None and bool(self.pool_rows[ri].any())
        if use_pool:
            # (a copy takes at most one free block: the first limit; the table of the warm experts
            # only when they run out, pool_scan: the two took 0.4 ms a call, most of _replace)
            free = np.flatnonzero(pool.owner == -1)[:limit].tolist()

        def pool_scan():
            pb = np.flatnonzero(pool.owner >= 0)
            if not pb.size:
                return None
            own = pool.owner[pb]            # (row E + expert: the flat index of the tables)
            ps = np.where(self.incoming.reshape(-1)[own], np.inf, score.reshape(-1)[own])
            # the walk below takes at most one block a copy (and skips
            # the blocks this call took): the lowest 2 limit + 8 do
            return [pb, own, ps, *self._low_first(ps, 2 * limit + 8)]
        scanned = not use_pool
        lowest = {}         # row -> its held expert of the lowest score, until one goes
        for o in order[:4 * limit]:
            if len(items) >= limit:
                break
            r, x = int(ri[o]), int(xi[o])
            if pool is not None and self.pool_rows[r] and free:
                slot = self.WARM + int(free.pop(0))    # a free block: no victim
            else:
                if r not in lowest:
                    k = urow.get(r)
                    if k is not None and r not in changed:
                        victim = int(va[k])
                        lowest[r] = (victim, hs[k, victim])
                    else:
                        held = np.where(self.held[r], score[r], np.inf)
                        victim = int(held.argmin())
                        lowest[r] = (victim, held[victim])
                victim, vs = lowest[r]
                vr = r
                if pool is not None and self.pool_rows[r] and not scanned:
                    pv, scanned = pool_scan(), True
                if pool is not None and self.pool_rows[r] and pv is not None:
                    # or the warm expert of the lowest score of any row (not
                    # one with a copy on the way)
                    pb, own, ps, po, whole = pv
                    while True:
                        while pk < po.size:
                            j = int(po[pk])
                            if pool.owner[pb[j]] == own[j] and ps[j] < np.inf:
                                break
                            pk += 1
                        if pk < po.size or whole:
                            break
                        po, whole = np.argsort(ps, kind="stable"), True   # past the part
                        pv[3:] = po, whole
                    if pk < po.size and ps[j] < vs:
                        vr, victim, vs = int(own[j] // E), int(own[j] % E), ps[j]
                if score[r, x] <= vs:
                    continue
                ve = self.layers[vr]
                slot = int(ve["slots"][victim])
                ve["slots"][victim] = -1
                self.held[vr, victim] = False
                lowest.pop(vr, None)
                changed.add(vr)
            if slot >= self.WARM:
                pool.owner[slot - self.WARM] = r * E + x
            self.incoming[r, x] = True
            items.append((r, x, slot))
        self._start_copies(items, changed)

    def _queue_room(self):
        """The copy queue of the GPU holds GG_CACHE_JOBS (64) jobs. Large
        experts (LFM2.5: 14 MB in BF12, 8 a step) can come faster than PCIe
        takes them: land the finished ones, and with the queue still near full
        no new copies this time (the candidates stay candidates)."""
        if len(self.pending) >= HOT_PENDING_MAX:
            self._land(False)
            if len(self.pending) >= HOT_PENDING_MAX:
                self.skipped += 1
                return False
        return True

    def _cptrs(self):
        """The addresses of the tables for the C functions (_hot_lib) and the
        buffers of their results, kept while the arrays stay (NumPy's
        .ctypes.data took several us each, about 0.1 ms a call)."""
        owner = self.pool.owner if self.pool is not None else None
        objs = (self.score, self.uses, self.total, self.run, self.held, self.incoming,
                self.pool_rows, owner) + tuple(e["slots"] for e in self.layers)
        c = self.__dict__.get("_cp")
        if c is None or len(c[0]) != len(objs) or any(a is not b for a, b in zip(c[0], objs)):
            m, E = self.score.shape
            sp = np.array([e["slots"].ctypes.data for e in self.layers], np.int64)
            cand, ch = np.empty((m, E), bool), np.zeros(m, np.uint8)
            p = {nm: getattr(self, nm).ctypes.data for nm in
                 ("score", "uses", "total", "run", "held", "incoming", "pool_rows")}
            p.update(owner=owner.ctypes.data if owner is not None else None,
                     nowner=owner.size if owner is not None else 0, slots=sp.ctypes.data,
                     cand=cand.ctypes.data, ch=ch.ctypes.data, sp=sp, cand_a=cand, ch_a=ch)
            c = self._cp = (objs, p)
        p = c[1]
        p["key"] = p["run"] if self.run_mode else p["score"]
        return p

    def _replace_c(self, lb, p, rows, limit):
        """_replace in C (hotcache.c) of the candidates in p's buffer cand
        (a row for each of rows, in order)."""
        m, E = self.held.shape
        out = self.__dict__.get("_cp_out")
        if out is None or out.shape[0] < limit:
            out = self._cp_out = np.empty((max(limit, 64), 3), np.int64)
            self._cp_outp = out.ctypes.data
        ni = lb.rc_hot_replace(p["key"], p["held"], p["incoming"], p["slots"], p["owner"],
                               p["nowner"], p["pool_rows"], p["cand"], rows.ctypes.data, rows.size,
                               m, E, limit, self.WARM, self._cp_outp, p["ch"])
        if ni:
            self._start_copies([tuple(r) for r in out[:ni].tolist()],
                               set(np.flatnonzero(p["ch_a"]).tolist()))

    def _start_copies(self, items, changed):
        """The copies of items ((row, expert, slot) of _replace) to their slots,
        after the slot tables of the rows changed (their old experts cold)."""
        if not items:
            return
        pool, rows = self.pool, []
        for r, x, slot in items:
            for part, (src, nb, dst) in enumerate(self.layers[r]["parts"]):
                # src: the host address of expert 0 (expert x at x nb), or the
                # address of each expert (an int64 array: QwenGPU._expert_srcs)
                a = int(src[x]) if isinstance(src, np.ndarray) else src + x * nb
                rows += _pieces(a, self._slot_addr(part, dst, nb, slot), nb)
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
        job = lib().rc_gpu_cache_copy(ranges.ctypes.data, len(rows))
        if job < 0:
            mem().unlock(segs)
            raise RuntimeError(lib().rc_gpu_last_error().decode())
        self.pending.append((job, items, ranges, segs))
        self.copies += len(items)

    def table_version(self):
        """The version of the tables of a large group: the slots, and the
        places of the segments of the pool (a warm expert has the address of
        its block)."""
        return (self.version, self.pool.moves if self.pool is not None else 0)

    def refresh(self, prog, g):
        """Fill the tables of a large group again if the slots changed."""
        if getattr(prog, "hot_version", None) == self.table_version():
            return
        hot = self.dev._hot_devices()
        for layer, chunks in (getattr(prog, "chunks_of", None) or {}).items():
            fill_chunks(self.dev.model, layer, hot.get(layer), self.dev.stage, chunks, pool=self.pool)
            for tgu, tdn, _ranges, gslots, _k in chunks:
                for a in (tgu, tdn, gslots):
                    _check(lib().rc_gpu_h2d(g.mirror.buffer_of(a).ptr, a.ctypes.data, a.nbytes))
        for layer, (tgu, tdn, ranges) in prog.tables_of.items():
            fill_tables(self.dev.model, layer, hot.get(layer), self.dev.stage, tgu, tdn, ranges,
                        pool=self.pool)
            for a in (tgu, tdn):
                _check(lib().rc_gpu_h2d(g.mirror.buffer_of(a).ptr, a.ctypes.data, a.nbytes))
        prog.hot_version = self.table_version()
