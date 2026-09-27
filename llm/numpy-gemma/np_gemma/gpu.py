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
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import program as P

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
        P.RMS_NORM_MULTI4: (6, 10, 14, 18), P.GELU_MUL_INT4: (5,),
        P.TO_HOST: (1, 4, 7), P.CPU_JOIN: (0,), P.TO_DEV: (0,)}


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
    key = hashlib.sha256((_SRC.read_text() + "|" + version + "|" + " ".join(flags))
                         .encode()).hexdigest()[:16]
    lib = _LIB_DIR / ("libgemma_gpu_" + key + ".so")
    if lib.exists():
        return lib
    _LIB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = lib.with_suffix(".so.tmp")
    subprocess.run([nvcc] + flags + ["-o", str(tmp), str(_SRC)], check=True,
                   capture_output=True)
    tmp.replace(lib)
    return lib


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
    L.gg_init.argtypes = [i]
    L.gg_mem_info.argtypes = [ctypes.POINTER(sz), ctypes.POINTER(sz)]
    L.gg_malloc.argtypes = [sz]
    L.gg_malloc.restype = vp
    L.gg_free.argtypes = [vp]
    L.gg_h2d.argtypes = [vp, vp, sz]
    L.gg_d2h.argtypes = [vp, vp, sz]
    L.gg_load.argtypes = [vp, i]
    L.gg_load.restype = vp
    L.gg_run.argtypes = [vp, vp]
    L.gg_unload.argtypes = [vp]
    L.gg_profile.argtypes = [vp, vp, vp]
    L.gg_q6k_head.argtypes = [vp, vp, vp, i, i, ctypes.c_float]
    L.gg_set_cpu_runner.argtypes = [vp]
    L.gg_host_alloc.argtypes = [sz]
    L.gg_host_alloc.restype = vp
    L.gg_d2d.argtypes = [vp, vp, sz]
    if L.gg_init(int(os.environ.get("NP_GEMMA_GPU_DEVICE", "0"))) != 0:
        _error = L.gg_last_error().decode()
        raise RuntimeError(_error)
    # A GP_CPU_JOIN record runs a CPU program with gemma_run of the CPU library.
    from . import cops
    if cops._lib is not None:
        L.gg_set_cpu_runner(ctypes.cast(cops._lib.gemma_run, ctypes.c_void_p))
    _lib = L
    return L


def available():
    """Return True when the library builds and a GPU is present."""
    try:
        lib()
        return True
    except RuntimeError:
        return False


def _check(rc):
    if rc != 0:
        raise RuntimeError(lib().gg_last_error().decode())


def mem_info():
    """Return the free and the total memory of the GPU, in bytes."""
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    _check(lib().gg_mem_info(ctypes.byref(f), ctypes.byref(t)))
    return f.value, t.value


class Buffer:
    """One block of device memory."""

    def __init__(self, nbytes):
        self.nbytes = int(nbytes)
        self.ptr = lib().gg_malloc(max(self.nbytes, 1))
        if not self.ptr:
            raise MemoryError(lib().gg_last_error().decode())

    def upload(self, a):
        assert a.flags.c_contiguous and a.nbytes <= self.nbytes
        _check(lib().gg_h2d(self.ptr, a.ctypes.data, a.nbytes))

    def download(self, a):
        assert a.flags.c_contiguous and a.nbytes <= self.nbytes
        _check(lib().gg_d2h(a.ctypes.data, self.ptr, a.nbytes))

    def free(self):
        if self.ptr:
            lib().gg_free(self.ptr)
            self.ptr = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


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

    def add(self, a):
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

    def device(self, start):
        """Return the Buffer of an array. Copy the array the first time."""
        b = self.bufs.get(start)
        if b is None:
            a = np.ascontiguousarray(self.arrays[start])
            b = self.bufs[start] = Buffer(a.nbytes)
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


def _check_scales(prog):
    """Check that the float16 scale of each int4 block equals the float32
    scale that the CPU kernels read."""
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
        else:
            continue
        for wk, sk, rk, ck in pairs:
            w, s = args[wk][1], args[sk][1]
            if not w:
                continue
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

    def __init__(self, prog, graph=True):
        self.prog = prog
        self.mirror = Mirror()
        for a in prog.keep:
            if isinstance(a, np.ndarray):
                self.mirror.add(a)
        _check_scales(prog)
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
        self.handle = lib().gg_load(buf.ctypes.data, 1 if graph else 0)
        if not self.handle:
            raise RuntimeError(lib().gg_last_error().decode())
        self.env = np.array(prog.buf[4:4 + n_env], dtype=np.int64)
        self.named = {}     # the device buffer of each bound array name

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
                        b = self.named[name] = Buffer(max(v.nbytes, 2 * (b.nbytes if b else 0)))
                    if name not in scratch:
                        b.upload(np.ascontiguousarray(v))
                    d = b.ptr
                self.env[s.index] = d
            elif isinstance(v, (float, np.floating)):
                self.env[s.index] = P._f32_bits(v)
            else:
                self.env[s.index] = int(v)

    def run(self):
        _check(lib().gg_run(self.handle, self.env.ctypes.data))

    def profile(self):
        """Run the program with a timer on each record. Return the time of
        each record in ms. Bind the step first."""
        ms = np.zeros(len(self.prog.recs), dtype=np.float32)
        _check(lib().gg_profile(self.handle, self.env.ctypes.data, ms.ctypes.data))
        return ms

    def close(self):
        if self.handle:
            lib().gg_unload(self.handle)
            self.handle = None


class GPUCache:
    """The device copy of the buffers of an E4BCache.

    Each buffer of the cache holds (heads, capacity, head_dim) values. The
    device buffer has the same shape. attach() copies the host buffers to the
    GPU. From then on, the device copy is the true one. to_host() copies it
    back. A cache that must grow first comes back to the host, grows, and
    goes to the GPU again.
    """

    def __init__(self):
        self.bufs = {}     # id of a host array -> (array, Buffer)

    def attach(self, cache):
        self.release()
        for store in list(cache.kv.values()):
            for a in store:
                if a is not None and id(a) not in self.bufs:
                    b = Buffer(a.nbytes)
                    b.upload(a)
                    self.bufs[id(a)] = (a, b)

    def device(self, a):
        e = self.bufs.get(id(a))
        return None if e is None else e[1].ptr

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
        for _a, b in self.bufs.values():
            b.free()
        self.bufs = {}


class E4BGPU:
    """The decode step of the E4B model on the GPU. See the module text."""

    def __init__(self, model, graph=True):
        self.model = model
        progs = model.__dict__.setdefault("_programs", {})
        prog = progs.get(1)
        if prog is None:
            prog = progs[1] = P.compile_e4b_step(model, 1)
        self.prog = prog
        self.g = GPUProgram(prog, graph=graph)
        self.cache = GPUCache()
        self.head = None

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
        kw = P.e4b_step_params(self.prog, model, cache, pos)
        names = self.prog.names
        names["x"][:] = model.embed_rows(P.E4B_PREFIX + "embed_tokens", ids) * cfg.embed_scale
        tok = model.embed_rows(P.E4B_PREFIX + "embed_tokens_per_layer", ids)
        names["tok"][:] = (tok * cfg.per_layer_embed_scale).reshape(1, -1)
        self.g.upload("x")
        self.g.upload("tok")
        self.g.bind(kw, self.cache)
        self.g.run()
        self.g.download("xn")
        return names["xn"].copy()

    def logits(self):
        """Return the logits of the last step, with the soft cap, shape
        (1, vocabulary). The first call copies the head to the GPU."""
        model, cfg = self.model, self.model.cfg
        if self.head is None:
            if not (model._q4 and model._head_is_q6k()):
                raise RuntimeError("the GPU head needs a Q6_K head")
            w = np.ascontiguousarray(model._head_q6k)
            rows = w.shape[0]
            self.head = Buffer(w.nbytes)
            self.head.upload(w)
            self.out = Buffer(4 * rows)
            self.host_logits = np.empty((1, rows), dtype=np.float32)
        xn = self.g.mirror.buffer_of(self.prog.names["xn"])
        cap = float(cfg.final_logit_softcapping or 0.0)
        _check(lib().gg_q6k_head(self.head.ptr, xn.ptr, self.out.ptr,
                                 self.host_logits.shape[1], cfg.hidden_size, cap))
        self.out.download(self.host_logits)
        return self.host_logits.copy()


# ---- the 26B model: the experts on the CPU (SPLIT_PLAN.md, phase 4) -----------

def pinned(shape, dtype=np.float32):
    """Return a NumPy array in pinned host memory. A copy between the GPU and
    pinned memory does not wait for the host. The memory is not freed."""
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    ptr = lib().gg_host_alloc(max(n, 1))
    if not ptr:
        raise MemoryError(lib().gg_last_error().decode())
    raw = (ctypes.c_uint8 * max(n, 1)).from_address(ptr)
    return np.frombuffer(raw, dtype=np.uint8, count=n).view(dtype).reshape(shape)


def _arrays(vals):
    out = []
    for v in vals:
        if isinstance(v, np.ndarray):
            out.append(v)
        elif isinstance(v, tuple):
            out += _arrays(v)
    return out


class SplitCompiler(P.Compiler):
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

    def __init__(self, model):
        super().__init__(model)
        self.pending = {}      # id of an output -> (CPU program, host output, event)
        self.n_events = 0
        self.cpu_progs = []

    def kernel(self, head, vals, out=None):
        for a in _arrays(vals):
            if id(a) in self.pending:
                self.join(a)
        if head == "moe":
            assert out is None
            return self.moe(*vals)
        return super().kernel(head, vals, out)

    def join(self, a):
        cpu, host, ev = self.pending.pop(id(a))
        self.p.emit(P.CPU_JOIN, cpu.buf, ev)
        self.p.emit(P.TO_DEV, host, a, a.nbytes)

    def moe(self, h, val, idx, layer):
        assert h.shape[0] == 1, "the GPU runs a step of one token"
        hp, vp, ip = pinned(h.shape), pinned(val.shape), pinned(idx.shape, np.int32)
        ev = self.n_events
        self.n_events += 1
        self.p.emit(P.TO_HOST, h, hp, h.nbytes, val, vp, val.nbytes, idx, ip, idx.nbytes, ev)
        cc = P.Compiler(self.model)
        host_out = P.k_moe(cc, hp, vp, ip, layer)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        dev_out = self.buffer(h.shape)
        self.pending[id(dev_out)] = (cpu, host_out, ev)
        return dev_out


def compile_split_step(model):
    """Compile a step of one token of the 26B model (or of a dense model) for
    the GPU, with the float cache. "x" is the input and "xn" the result."""
    c = SplitCompiler(model)
    c.env["x"] = np.zeros((1, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    c.compile(P.step_form(model, "f32", 1))
    assert not c.pending, "an output of the experts has no reader"
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = "f32"
    c.p.tokens = 1
    c.p.cpu_progs = c.cpu_progs
    return c.p.finish()


class GPUKV:
    """The float cache of the 26B model on the GPU.

    Each layer has a key and a value buffer of (rows, kv_heads, head_dim)
    values. Row 0 has the position base, as in KVCache. A sliding layer drops
    its oldest rows when the buffer holds more than two windows, as
    KVCache.prepare does, with a copy on the GPU. attach() copies the rows of
    a KVCache to the GPU. detach() writes the rows that only the GPU has into
    the KVCache with KVCache.write, which also makes their int16 copy.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.window = cfg.sliding_window or 0
        n = cfg.num_hidden_layers
        self.k = [None] * n
        self.v = [None] * n
        self.cap = [0] * n
        self.base = [0] * n
        self.end = [0] * n
        self.host_end = [0] * n

    def _row(self, i):
        plan = self.cfg.plan[i]
        return plan.num_kv_heads * plan.head_dim * 4

    def _alloc(self, i, cap, keep_rows=0):
        """Give layer i a buffer of cap rows. Keep the first keep_rows rows."""
        rb = self._row(i)
        k, v = Buffer(cap * rb), Buffer(cap * rb)
        if keep_rows:
            _check(lib().gg_d2d(k.ptr, self.k[i].ptr, keep_rows * rb))
            _check(lib().gg_d2d(v.ptr, self.v[i].ptr, keep_rows * rb))
        if self.k[i] is not None:
            lib().gg_sync()
            self.k[i].free()
            self.v[i].free()
        self.k[i], self.v[i], self.cap[i] = k, v, cap

    def attach(self, cache, max_len):
        for i in range(self.cfg.num_hidden_layers):
            rows = cache.end[i] - cache.base[i]
            plan = self.cfg.plan[i]
            cap = (2 * self.window + 64) if plan.is_sliding else max(max_len, rows + 64)
            cap = max(cap, rows + 64)
            if self.k[i] is None or self.cap[i] < cap:
                self._alloc(i, cap)
            self.base[i], self.end[i] = cache.base[i], cache.end[i]
            self.host_end[i] = cache.end[i]
            if rows > 0:
                Buffer.upload(self.k[i], np.ascontiguousarray(cache.k[i][:rows]))
                Buffer.upload(self.v[i], np.ascontiguousarray(cache.v[i][:rows]))

    def prepare(self, i, pos):
        """Make room for the row of position pos in layer i."""
        rb = self._row(i)
        if self.cfg.plan[i].is_sliding:
            w = self.window
            if pos - self.base[i] > 2 * w:
                keep = pos - w + 1
                off = keep - self.base[i]
                rows = self.end[i] - keep
                if rows > 0:
                    _check(lib().gg_d2d(self.k[i].ptr, self.k[i].ptr + off * rb, rows * rb))
                    _check(lib().gg_d2d(self.v[i].ptr, self.v[i].ptr + off * rb, rows * rb))
                self.base[i] = keep
                # The host now lacks some rows that the GPU dropped. Only the
                # rows of the window matter to a later step.
                self.host_end[i] = max(self.host_end[i], keep)
        need = pos + 1 - self.base[i]
        if need > self.cap[i]:
            self._alloc(i, max(need + 64, 2 * self.cap[i]), self.end[i] - self.base[i])
        self.end[i] = max(self.end[i], pos + 1)

    def detach(self, cache):
        """Write the rows that the GPU made into the host cache."""
        for i in range(self.cfg.num_hidden_layers):
            start = max(self.host_end[i], cache.end[i], self.base[i])
            n = self.end[i] - start
            if n <= 0:
                continue
            plan = self.cfg.plan[i]
            shape = (n, plan.num_kv_heads, plan.head_dim)
            k, v = np.empty(shape, np.float32), np.empty(shape, np.float32)
            off = (start - self.base[i]) * self._row(i)
            _check(lib().gg_d2h(k.ctypes.data, self.k[i].ptr + off, k.nbytes))
            _check(lib().gg_d2h(v.ctypes.data, self.v[i].ptr + off, v.nbytes))
            cache.write(i, start, k, v)
            self.host_end[i] = self.end[i]

    def params(self):
        kw = {}
        for i in range(self.cfg.num_hidden_layers):
            kw["base.%d" % i] = self.base[i]
            kw["k.%d" % i] = self.k[i].ptr
            kw["v.%d" % i] = self.v[i].ptr
        return kw


class ModelGPU:
    """A decode step of the 26B model on the GPU, with the experts on the CPU.
    A dense model (the 12B) runs wholly on the GPU if it fits. See the module
    text and GPUKV.

        g = gpu.ModelGPU(model)
        g.attach(cache)                  # after the prompt pass on the CPU
        xn = g.step([token], pos)        # the hidden state after the final norm
        logits = g.logits()              # the head of the last step
        g.detach(cache)                  # the host cache has the new rows
    """

    def __init__(self, model, graph=True):
        self.model = model
        self.prog = compile_split_step(model)
        self.g = GPUProgram(self.prog, graph=graph)
        self.kv = GPUKV(model.cfg)
        self.head = None
        self.max_len = 4096

    def attach(self, cache):
        self.kv.attach(cache, max(self.max_len, cache.max_len))

    def detach(self, cache):
        self.kv.detach(cache)

    def step(self, tokens, pos):
        model, cfg = self.model, self.model.cfg
        assert len(tokens) == 1, "the GPU runs a step of one token"
        for i in range(cfg.num_hidden_layers):
            self.kv.prepare(i, pos)
        kw = {"pos": pos}
        kw.update(self.kv.params())
        positions = np.array([pos])
        for kind, sliding in (("s", True), ("f", False)):
            plan = next((p for p in cfg.plan if p.is_sliding == sliding), None)
            if plan is None:
                continue
            cos, sin, _ca, _sa = model._rope(plan, positions)
            kw["cos." + kind] = np.ascontiguousarray(cos, dtype=np.float32)
            kw["sin." + kind] = np.ascontiguousarray(sin, dtype=np.float32)
        need = max(p.num_q_heads * (pos + 1) for p in cfg.plan)
        kw["scores"] = np.empty(need, dtype=np.float32)
        self.prog.names["x"][:] = model.embed(tokens)
        self.g.upload("x")
        self.g.bind(kw)
        self.g.run()
        self.g.download("xn")
        return self.prog.names["xn"].copy()

    def logits(self):
        """Return the logits of the last step, with the soft cap."""
        model, cfg = self.model, self.model.cfg
        if self.head is None:
            if model._embed_q6k is None:
                raise RuntimeError("the GPU head needs a Q6_K head")
            w = np.ascontiguousarray(model._embed_q6k_bytes)
            rows = w.shape[0]
            self.head = Buffer(w.nbytes)
            self.head.upload(w)
            self.out = Buffer(4 * rows)
            self.host_logits = np.empty((1, rows), dtype=np.float32)
        xn = self.g.mirror.buffer_of(self.prog.names["xn"])
        cap = float(cfg.final_logit_softcapping or 0.0)
        _check(lib().gg_q6k_head(self.head.ptr, xn.ptr, self.out.ptr,
                                 self.host_logits.shape[1], cfg.hidden_size, cap))
        self.out.download(self.host_logits)
        return self.host_logits.copy()
