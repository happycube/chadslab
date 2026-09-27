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
        P.INT4_LINEAR_MT: (2,), P.INT4_MULTI4_MT: (4, 8, 12, 16),
        P.RMS_NORM_MULTI4: (6, 10, 14, 18), P.GELU_MUL_INT4: (5,),
        P.TO_HOST: (1, 4, 7), P.CPU_JOIN: (0,), P.TO_DEV: (0,), P.FETCH: (0,)}


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
    L.gg_q6k_head.argtypes = [vp, vp, vp, i, i, ctypes.c_float, i]
    L.gg_set_cpu_runner.argtypes = [vp]
    L.gg_set_tc.argtypes = [i]
    L.gg_host_alloc.argtypes = [sz]
    L.gg_host_alloc.restype = vp
    L.gg_d2d.argtypes = [vp, vp, sz]
    # NP_GEMMA_GPU_TC=0 gives the float32 kernels for the products of a
    # large group, in place of the tensor cores.
    L.gg_set_tc(int(os.environ.get("NP_GEMMA_GPU_TC", "1")))
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
        self.checked = set()  # the int4 matrices whose scales are checked

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

    def __init__(self, prog, graph=True, mirror=None, tc=True):
        """mirror is the Mirror of an earlier program of the same model. The
        programs then share the device copies of the weights. tc False keeps
        float32 products for the large groups of this program, in place of
        the tensor cores."""
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
        self.handle = lib().gg_load(buf.ctypes.data, (1 if graph else 0) | (0 if tc else 2))
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


class PoolCompiler(P.Compiler):
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


def compile_e4b_group(model, t):
    """Compile a step of t tokens of the E4B model for the GPU, with the
    buffers of the layers reused (PoolCompiler)."""
    cfg = model.cfg
    c = E4BGroupCompiler(model, pool=True)
    c.env["x"] = np.zeros((t, cfg.hidden_size), dtype=np.float32)
    c.env["tok"] = np.zeros((t, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input),
                            dtype=np.float32)
    c.p.slot("pos")
    c.compile(P.e4b_step_form(model))
    c.p.tokens = t
    return c.p.finish()


class E4BGPU:
    """The decode step, the groups of tokens, and the prompt pass of the
    E4B model on the GPU. See the module text."""

    def __init__(self, model, graph=True):
        self.model = model
        progs = model.__dict__.setdefault("_programs", {})
        prog = progs.get(1)
        if prog is None:
            prog = progs[1] = P.compile_e4b_step(model, 1)
        self.prog = prog
        self.g = GPUProgram(prog, graph=graph)
        self.graph = graph
        self.cache = GPUCache()
        self.head = None
        self.groups = {}      # t -> (Program, GPUProgram)
        self.last = self.g.mirror.buffer_of(prog.names["xn"]).ptr
        self.rows = 1

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
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.rows = 1
        return names["xn"].copy()

    def _group(self, t):
        e = self.groups.get(t)
        if e is None:
            prog = compile_e4b_group(self.model, t)
            e = self.groups[t] = (prog, GPUProgram(prog, graph=self.graph, mirror=self.g.mirror))
        return e

    def group(self, tokens, pos, cache, size=None):
        """Run a group of tokens from position pos. Return the hidden states
        after the final norm, shape (len(tokens), hidden). size pads the group
        to a program of that many tokens. The padding rows write cache rows
        after the group, and a later step writes them again."""
        model, cfg = self.model, self.model.cfg
        ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
        t = ids.size
        size = size or t
        prog, g = self._group(size)
        if self._ensure(cache):
            self.cache.attach(cache)
        self.cache.reserve(cache, pos + size)
        kw = P.e4b_step_params(prog, model, cache, pos)
        cache.n = pos + t
        names = prog.names
        names["x"][:t] = model.embed_rows(P.E4B_PREFIX + "embed_tokens", ids) * cfg.embed_scale
        names["x"][t:] = 0.0
        tok = model.embed_rows(P.E4B_PREFIX + "embed_tokens_per_layer", ids)
        names["tok"][:t] = (tok * cfg.per_layer_embed_scale).reshape(t, -1)
        names["tok"][t:] = 0.0
        g.upload("x")
        g.upload("tok")
        g.bind(kw, self.cache)
        g.run()
        g.download("xn")
        self.last = g.mirror.buffer_of(names["xn"]).ptr + (t - 1) * cfg.hidden_size * 4
        self.rows = t
        return names["xn"][:t].copy()

    def _ensure(self, cache):
        """Make the buffers of every layer that stores a key, as
        E4BCache.append does. Return True when a buffer is new."""
        new = False
        for plan in self.model.cfg.plan:
            if plan.shared or plan.idx in cache.kv:
                continue
            shape = (plan.num_kv_heads, cache.cap, plan.head_dim)
            store = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
            cache.kv[plan.idx] = store
            if plan.stores:
                cache.shared[plan.kind] = store
            new = True
        return new

    def prefill(self, ids, pos, cache):
        """Run a prompt from position pos in chunks of CHUNK tokens. A group
        of up to MT_CPU tokens runs as it is. A longer chunk goes to a
        program of the next power of two. Return the hidden states of every
        token."""
        ids = list(ids)
        out = []
        for c0 in range(0, len(ids), CHUNK):
            chunk = ids[c0:c0 + CHUNK]
            size = None
            if len(chunk) > MT_CPU:
                size = min(CHUNK, 1 << (len(chunk) - 1).bit_length())
            out.append(self.group(chunk, pos + c0, cache, size))
        return np.concatenate(out)

    def logits(self, rows=1):
        """Return the logits of the last rows of the last step or group, with
        the soft cap, shape (rows, vocabulary). The first call copies the
        head to the GPU."""
        model, cfg = self.model, self.model.cfg
        if self.head is None:
            if not (model._q4 and model._head_is_q6k()):
                raise RuntimeError("the GPU head needs a Q6_K head")
            w = np.ascontiguousarray(model._head_q6k)
            vocab = w.shape[0]
            self.head = Buffer(w.nbytes)
            self.head.upload(w)
            self.out = Buffer(4 * vocab * MT_CPU)
            self.host_logits = np.empty((MT_CPU, vocab), dtype=np.float32)
        assert 1 <= rows <= min(self.rows, MT_CPU)
        vocab = self.host_logits.shape[1]
        cap = float(cfg.final_logit_softcapping or 0.0)
        first = self.last - (rows - 1) * cfg.hidden_size * 4
        _check(lib().gg_q6k_head(self.head.ptr, first, self.out.ptr, vocab, cfg.hidden_size,
                                 cap, rows))
        _check(lib().gg_d2h(self.host_logits.ctypes.data, self.out.ptr, rows * vocab * 4))
        return self.host_logits[:rows].copy()


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
        # blocks). For a group program: the device addresses of those blocks
        # (see moe_group_gpu).
        self.hot_stores = {}
        # layer -> (hot experts, gate and up blocks, down blocks) of the step
        # program, for a small group (see moe_hot_group).
        self.hot_host = {}

    def kernel(self, head, vals, out=None):
        for a in _arrays(vals):
            if id(a) in self.pending:
                self.join(a)
        if head == "moe":
            assert out is None
            return self.moe(*vals)
        if head == "kv_write" and self.kv == "int16":
            return self.kv_write16(*vals)
        if head == "attn_rows_qc" and vals[1].shape[0] <= MT_CPU:
            return self.attn_rows_small(*vals)
        return super().kernel(head, vals, out)

    def attn_rows_small(self, layer, q):
        """The attention of a small group, one query at a time with the
        record of a decode step (GP_ATTN_QC). Its kernel splits the keys of
        each head into chunks, and it is faster for a few queries than the
        group kernel. Query j has the position pos + j."""
        plan = self.cfg.plan[layer]
        hd, qh = plan.head_dim, plan.num_q_heads
        t = q.shape[0]
        out = self.buffer((t, qh * hd))
        base = "base.%d" % layer
        for j in range(t):
            p = ("+", "pos", j)
            if plan.is_sliding:
                lo = ("max", 0, ("-", p, plan.sliding_window - 1, base))
            else:
                lo = 0
            lo_v = self.value(lo)
            n_v = self.value(("-", ("+", p, 1), base, lo_v))
            a = P.k_attn_qc(self, layer, q[j:j + 1], lo_v, n_v)
            self.p.emit(P.COPY, a, out[j:j + 1], a.nbytes)
        return out

    def kv_write16(self, layer, k, v, row, qc=1):
        """As k_kv_write, for a cache with the int16 copy only. The float
        addresses are null, so the kernel does not store float rows."""
        plan = self.cfg.plan[layer]
        per = plan.num_kv_heads * plan.head_dim
        sl = lambda name: self.p.slot("%s.%d" % (name, layer))  # noqa: E731
        q = [P._addr(self, sl("kq"), row, 2 * per), P._addr(self, sl("ks"), row, per // 8),
             P._addr(self, sl("vq"), row, 2 * per), P._addr(self, sl("vs"), row, per // 8)]
        self.p.emit(P.KV_WRITE, k, v, 0, 0, *q, k.size)

    def join(self, a):
        cpu, host, ev, part, gpu_part = self.pending.pop(id(a))
        self.p.emit(P.CPU_JOIN, cpu.buf, ev)
        self.p.emit(P.TO_DEV, host, part, part.nbytes)
        if gpu_part is not None:
            self.p.emit(P.ADD, gpu_part, part, a, a.size)

    def moe(self, h, val, idx, layer):
        if h.shape[0] > 1 and self.stage is not None:
            return self.moe_group_gpu(h, val, idx, layer)
        if h.shape[0] > 1 and layer in self.hot_host:
            return self.moe_hot_group(h, val, idx, layer)
        if h.shape[0] > 1:
            return self.moe_group_cpu(h, val, idx, layer)
        if self.hot.get(layer):
            return self.moe_hot(h, val, idx, layer, self.hot[layer])
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

    def tables(self, layer):
        """Return the tables of the device addresses of the experts of a
        layer (gate and up, down), and the list of the copies of the cold
        experts: (host address, device address, bytes) for each run of
        adjacent cold experts."""
        w = self.model._layers[layer]
        gu, dn = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
        n = gu.shape[0]
        egu, edn = gu.nbytes // n, dn.nbytes // n
        dgu, ddn = self.stage[layer % 2]
        hot, hgu, hdn = self.hot_stores.get(layer, ([], 0, 0))
        slot = {x: k for k, x in enumerate(hot)}
        tgu = np.zeros(n, dtype=np.int64)
        tdn = np.zeros(n, dtype=np.int64)
        ranges = []
        rank = 0
        for x in range(n):
            if x in slot:
                tgu[x] = hgu + slot[x] * egu
                tdn[x] = hdn + slot[x] * edn
                continue
            tgu[x] = dgu.ptr + rank * egu
            tdn[x] = ddn.ptr + rank * edn
            if ranges and ranges[-1][3] == x - 1:
                ranges[-1][2] += egu
                ranges[-1][6] += edn
                ranges[-1][3] = x
            else:
                # host gate and up, device, bytes, last expert; host down,
                # device, bytes
                ranges.append([gu.ctypes.data + x * egu, int(tgu[x]), egu, x,
                               dn.ctypes.data + x * edn, int(tdn[x]), edn])
            rank += 1
        flat = []
        for r in ranges:
            flat += [r[0], r[1], r[2]]
        for r in ranges:
            flat += [r[4], r[5], r[6]]
        return tgu, tdn, np.array(flat, dtype=np.int64).reshape(-1, 3)

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
        self.p.emit(P.MOE_GPU, h, val, idx, t, top_k, 0, 0, gu_q.shape[1], hidden,
                    dn_q.shape[1], inner, i32(256), i32(257), i32(256), i32(pairs), i32(pairs),
                    i32(1 + 2 * (pairs // 64 + 129)), self.buffer((pairs, gu_q.shape[1])),
                    self.buffer((pairs, inner)), self.buffer((pairs, dn_q.shape[1])), out,
                    tgu, tdn)
        self.p.emit(P.FETCH_DONE, layer % 2)
        if layer + 2 < self.cfg.num_hidden_layers:
            self.fetch(layer + 2)
        return out

    def moe_hot_group(self, h, val, idx, layer):
        """The experts of a small group (at most MT_CPU tokens) when the GPU
        holds some of them. GP_HOT_SPLIT_MT marks each pair (token, slot):
        the GPU computes the pairs of the hot experts with GP_HOT_MOE, and
        the CPU computes the other pairs with GP_MOE_MT. The output is the
        sum of the two parts. The hot experts are the arrays of the step
        program, so the GPU holds them one time."""
        hot, gu_store, dn_store = self.hot_host[layer]
        w = self.model._layers[layer]
        n = w["experts.gate_up_proj"][0].shape[0]
        dn_rows = w["experts.down_proj"][0].shape[1]
        inner = self.cfg.moe_intermediate_size
        t, top_k = idx.shape
        hidden = h.shape[1]
        slots = np.full(n, -1, dtype=np.int32)
        slots[hot] = np.arange(len(hot), dtype=np.int32)
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
                    dn_rows, inner, t)
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
        gu_store = np.ascontiguousarray(gu_q[hot])
        dn_store = np.ascontiguousarray(dn_q[hot])
        self.hot_stores[layer] = (list(hot), gu_store, dn_store)
        for store, scales in ((gu_store, gu_s), (dn_store, dn_s)):
            half = store[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
            if not np.array_equal(half, scales[hot]):
                raise ValueError("an expert has float32 scales that are not its float16 scales")
        cold = np.zeros(top_k + 1, dtype=np.int32)
        cold_val = self.buffer(top_k)
        self.p.emit(P.HOT_SPLIT, idx, val, slots, cold, cold_val, top_k)
        hp, vp, ip = pinned(h.shape), pinned((top_k,)), pinned((top_k + 1,), np.int32)
        ev = self.n_events
        self.n_events += 1
        self.p.emit(P.TO_HOST, h, hp, h.nbytes, cold_val, vp, vp.nbytes, cold, ip, ip.nbytes, ev)
        gpu_part = self.buffer(h.shape)
        self.p.emit(P.HOT_MOE, h, val, idx, slots, gu_store, dn_store,
                    self.buffer((top_k, 2 * inner)), self.buffer((top_k, inner)),
                    self.buffer((top_k, hidden)), gpu_part, top_k, 2 * inner, hidden,
                    dn_q.shape[1], inner, 1)
        cc = P.Compiler(self.model)
        host_out = np.zeros(h.shape, dtype=np.float32)
        cc.p.emit(P.MOE_N, hp, vp, ip, ip[top_k:], gu_q, gu_s, dn_q, dn_s, gu_q.shape[1],
                  hidden, dn_q.shape[1], inner, np.zeros(top_k, dtype=np.int32),
                  np.zeros((top_k, 2 * inner), np.float32), np.zeros((top_k, inner), np.float32),
                  np.zeros((top_k, dn_q.shape[1]), np.float32), host_out)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        out = self.buffer(h.shape)
        self.pending[id(out)] = (cpu, host_out, ev, self.buffer(h.shape), gpu_part)
        return out


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
# about 1.9 s over PCIe 3 x8. A long chunk pays for the copy with more tokens.
CHUNK = int(os.environ.get("NP_GEMMA_GPU_CHUNK", "1024"))
# The largest group whose experts run on the CPU (GP_MOE_MT of the CPU
# interpreter takes at most 16 tokens).
MT_CPU = 16
# The shortest part of a prompt that runs with the experts on the GPU. The
# copy of their weights costs about 1.9 s, and the CPU takes about as long for
# 128 tokens.
PREFILL_MIN = int(os.environ.get("NP_GEMMA_GPU_PREFILL_MIN", "128"))


def compile_split_group(model, t, hot=None, kv="int16", stage=None, hot_stores=None,
                        hot_host=None):
    """Compile a step of t tokens for the GPU: a chunk of a prompt, or the
    group of an MTP verify step. The form is the group form of the layers.

    A group of more than MT_CPU tokens computes the experts on the GPU, with
    the weights in the two device buffers of stage. A smaller group sends
    them to the CPU, as a step does."""
    assert kv == "int16", "a group on the GPU reads the int16 cache"
    gpu_experts = model.cfg.enable_moe_block and t > MT_CPU and stage is not None
    c = SplitCompiler(model, hot, kv, pool=True, stage=stage if gpu_experts else None)
    c.hot_stores = hot_stores or {}
    c.hot_host = hot_host or {}
    c.env["x"] = np.zeros((t, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    if gpu_experts:
        c.fetch(0)
        c.fetch(1)
    c.compile(P.step_form(model, "qc", t))
    assert not c.pending, "an output of the experts has no reader"
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = "qc"
    c.p.tokens = t
    c.p.cpu_progs = c.cpu_progs
    return c.p.finish()


def compile_split_step(model, hot=None, kv="int16"):
    """Compile a step of one token of the 26B model (or of a dense model) for
    the GPU. "x" is the input and "xn" the result. hot gives the experts that
    the GPU holds (see pick_hot). kv is the form of the cache on the GPU:
    "int16" (the form of the CPU program, a scale for each group of 32
    values) or "float"."""
    c = SplitCompiler(model, hot, kv)
    c.env["x"] = np.zeros((1, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    c.compile(P.step_form(model, "qc" if kv == "int16" else "f32", 1))
    assert not c.pending, "an output of the experts has no reader"
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = "qc" if kv == "int16" else "f32"
    c.p.tokens = 1
    c.p.cpu_progs = c.cpu_progs
    c.p.hot_stores = c.hot_stores
    return c.p.finish()


class GPUKV:
    """The cache of the 26B model on the GPU.

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
        self.cfg = cfg
        self.form = kv
        self.max_chunk = max_chunk
        self.window = cfg.sliding_window or 0
        n = cfg.num_hidden_layers
        self.names = ("kq", "ks", "vq", "vs") if kv == "int16" else ("k", "v")
        self.bufs = [dict() for _ in range(n)]
        self.cap = [0] * n
        self.base = [0] * n
        self.end = [0] * n
        self.host_end = [0] * n

    def _row(self, i, name):
        """Return the bytes of one row of the buffer name of layer i."""
        plan = self.cfg.plan[i]
        per = plan.num_kv_heads * plan.head_dim
        return {"k": 4 * per, "v": 4 * per, "kq": 2 * per, "vq": 2 * per,
                "ks": per // 8, "vs": per // 8}[name]

    def nbytes(self):
        return sum(b.nbytes for d in self.bufs for b in d.values())

    def _alloc(self, i, cap, keep_rows=0):
        """Give layer i buffers of cap rows. Keep the first keep_rows rows."""
        new = {}
        for name in self.names:
            rb = self._row(i, name)
            new[name] = Buffer(cap * rb)
            if keep_rows:
                _check(lib().gg_d2d(new[name].ptr, self.bufs[i][name].ptr, keep_rows * rb))
        if self.bufs[i]:
            lib().gg_sync()
            for b in self.bufs[i].values():
                b.free()
        self.bufs[i], self.cap[i] = new, cap

    def _host_rows(self, cache, i, rows):
        """Return the host rows of layer i in the form of the GPU."""
        if self.form == "float":
            return {"k": cache.k[i][:rows], "v": cache.v[i][:rows]}
        if cache._qc_on[i]:
            return {"kq": cache.kq[i][:rows], "ks": cache.ks[i][:rows],
                    "vq": cache.vq[i][:rows], "vs": cache.vs[i][:rows]}
        from . import ops
        out = {}
        for name, a in (("k", cache.k[i][:rows]), ("v", cache.v[i][:rows])):
            q, sc = ops.quantize_i16(a.reshape(rows, -1, 32))
            out[name + "q"], out[name + "s"] = q, sc
        return out

    def attach(self, cache, max_len):
        for i in range(self.cfg.num_hidden_layers):
            rows = cache.end[i] - cache.base[i]
            plan = self.cfg.plan[i]
            cap = ((2 * self.window + self.max_chunk + 64) if plan.is_sliding
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
            if pos - self.base[i] > 2 * w:
                keep = pos - w + 1
                off = keep - self.base[i]
                rows = self.end[i] - keep
                if rows > 0:
                    for name in self.names:
                        rb = self._row(i, name)
                        b = self.bufs[i][name]
                        _check(lib().gg_d2d(b.ptr, b.ptr + off * rb, rows * rb))
                self.base[i] = keep
                # The host now lacks some rows that the GPU dropped. Only the
                # rows of the window matter to a later step.
                self.host_end[i] = max(self.host_end[i], keep)
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
            got = {}
            for name in self.names:
                rb = self._row(i, name)
                dt = {"kq": np.int16, "vq": np.int16}.get(name, np.float32)
                a = np.empty(n * rb // np.dtype(dt).itemsize, dtype=dt)
                _check(lib().gg_d2h(a.ctypes.data,
                                    self.bufs[i][name].ptr + (start - self.base[i]) * rb, a.nbytes))
                got[name] = a
            if self.form == "float":
                k, v = got["k"].reshape(shape), got["v"].reshape(shape)
            else:
                k = (got["kq"].reshape(-1, 32) * got["ks"][:, None]).reshape(shape)
                v = (got["vq"].reshape(-1, 32) * got["vs"][:, None]).reshape(shape)
            cache.write(i, start, k.astype(np.float32), v.astype(np.float32))
            self.host_end[i] = self.end[i]

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
        kv = kv or os.environ.get("NP_GEMMA_GPU_KV", "int16")
        self.kv_form = kv
        self.graph = graph
        self.prog = compile_split_step(model, self.hot, kv)
        self.g = GPUProgram(self.prog, graph=graph)
        self.kv = GPUKV(model.cfg, kv, max_chunk=CHUNK)
        self.groups = {}      # t -> (Program, GPUProgram) of a group of t tokens
        self.head = None
        self.rows = 1
        self.max_len = 4096
        # The device address of the hidden state of the last row of the last
        # step or group, for the output head.
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr

    def attach(self, cache):
        self.kv.attach(cache, max(self.max_len, cache.max_len))

    def detach(self, cache):
        self.kv.detach(cache)

    def _params(self, pos, t):
        """Prepare the cache for t rows from pos. Return the parameters."""
        model, cfg = self.model, self.model.cfg
        for i in range(cfg.num_hidden_layers):
            self.kv.prepare(i, pos, t)
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
        kw = self._params(pos, 1)
        self.prog.names["x"][:] = self.model.embed(tokens)
        self.g.upload("x")
        self.g.bind(kw)
        self.g.run()
        self.g.download("xn")
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.rows = 1
        return self.prog.names["xn"].copy()

    def _hot_devices(self):
        """Return layer -> (hot experts, device address of their gate and up
        blocks, of their down blocks), from the step program."""
        out = {}
        for layer, (hot, gu, dn) in self.prog.hot_stores.items():
            out[layer] = (hot, self.g.mirror.buffer_of(gu).ptr, self.g.mirror.buffer_of(dn).ptr)
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
                                       hot_host=self.prog.hot_stores)
            # The tensor cores round the input of a product to float16. The
            # router of the experts then selects another expert more often:
            # 93% of the top tokens agree with the CPU, against 98% with the
            # float32 products. The prompt pass of the 26B waits for the copy
            # of the experts anyway, so it keeps float32 products.
            tc = os.environ.get("NP_GEMMA_GPU_TC_MOE", "0") == "1" or \
                not self.model.cfg.enable_moe_block
            e = self.groups[t] = (prog, GPUProgram(prog, graph=self.graph, mirror=self.g.mirror,
                                                   tc=tc))
        return e

    def group(self, tokens, pos, size=None):
        """Run a group of tokens from position pos. Return the hidden states
        after the final norm, shape (len(tokens), hidden).

        size pads the group to a program of that many tokens. The padding
        rows write cache rows after the group. The cache then forgets them:
        a later step writes those rows again.
        """
        t = len(tokens)
        size = size or t
        prog, g = self._group(size)
        kw = self._params(pos, size)
        x = prog.names["x"]
        x[:t] = self.model.embed(tokens)
        x[t:] = 0.0
        g.upload("x")
        g.bind(kw)
        g.run()
        g.download("xn")
        for i in range(self.model.cfg.num_hidden_layers):
            self.kv.end[i] = min(self.kv.end[i], pos + t)
        hidden = self.model.cfg.hidden_size
        self.first = g.mirror.buffer_of(prog.names["xn"]).ptr
        self.last = self.first + (t - 1) * hidden * 4
        self.rows = t
        return prog.names["xn"][:t].copy()

    def prefill(self, ids, pos=0):
        """Run a prompt from position pos. Return the hidden states after the
        final norm of every token.

        A part of PREFILL_MIN tokens or more goes in chunks of CHUNK tokens,
        with the experts on the GPU. A short chunk goes to a program of the
        next power of two. Each such chunk copies the weights of the experts
        to the GPU. A shorter part goes in groups of 16 tokens, with the
        experts on the CPU, which costs less than that copy.
        """
        out = []
        c0 = 0
        while c0 < len(ids):
            chunk = ids[c0:c0 + CHUNK]
            if len(chunk) >= PREFILL_MIN or not self.model.cfg.enable_moe_block:
                size = max(MT_CPU + 1, 1 << (len(chunk) - 1).bit_length())
                out.append(self.group(chunk, pos + c0, min(size, CHUNK)))
            else:
                chunk = ids[c0:c0 + MT_CPU]
                out.append(self.group(chunk, pos + c0, MT_CPU))
            c0 += len(chunk)
        return np.concatenate(out)

    def logits(self, rows=1):
        """Return the logits of the last rows of the last step or group, with
        the soft cap, shape (rows, vocabulary). An MTP verify group needs the
        logits of each of its rows. At most 16 rows."""
        model, cfg = self.model, self.model.cfg
        if self.head is None:
            if model._embed_q6k is None:
                raise RuntimeError("the GPU head needs a Q6_K head")
            w = np.ascontiguousarray(model._embed_q6k_bytes)
            vocab = w.shape[0]
            self.head = Buffer(w.nbytes)
            self.head.upload(w)
            self.out = Buffer(4 * vocab * MT_CPU)
            self.host_logits = np.empty((MT_CPU, vocab), dtype=np.float32)
        assert 1 <= rows <= min(self.rows, MT_CPU)
        vocab = self.host_logits.shape[1]
        cap = float(cfg.final_logit_softcapping or 0.0)
        step = cfg.hidden_size * 4
        _check(lib().gg_q6k_head(self.head.ptr, self.last - (rows - 1) * step, self.out.ptr,
                                 vocab, cfg.hidden_size, cap, rows))
        _check(lib().gg_d2h(self.host_logits.ctypes.data, self.out.ptr, rows * vocab * 4))
        return self.host_logits[:rows].copy()


def offload(model, experts_gb=0.0):
    """Put the weights of a Model outside the experts, and its output head, on
    the GPU. The decode steps of one token then run there (NP_GEMMA_GPU=1).

    experts_gb 0 keeps every expert on the CPU. A value above 0 also puts the
    most used experts on the GPU, up to that many GB. None takes the default
    budget of ModelGPU. The function copies the weights now, so the first
    request does not wait for them. Return the ModelGPU.
    """
    from . import model as model_mod
    os.environ["NP_GEMMA_GPU"] = "1"
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
    """Return one line about the GPU part of a ModelGPU."""
    free, total = mem_info()
    return ("GPU: %.2f GB of weights and buffers, %d hot experts, head %.2f GB; "
            "%.1f of %.1f GB free" % (g.g.mirror.nbytes() / 1e9,
                                       sum(len(v) for v in g.hot.values()),
                                       (g.head.nbytes if g.head else 0) / 1e9,
                                       free / 1e9, total / 1e9))
