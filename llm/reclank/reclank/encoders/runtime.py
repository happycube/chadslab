"""The image and audio encoders of the Gemma 4 E2B, E4B, and 26B (gemma4v, gemma4a).

Ported from transformers (src/transformers/models/gemma4/modeling_gemma4.py,
Copyright 2026 the HuggingFace Team; models/gemma4/feature_extraction_gemma4.py,
Copyright 2026 Google LLC; Apache License 2.0). See THIRD_PARTY_NOTICES.md.

The weights come from the mmproj GGUF (llama.cpp). The graphs follow
transformers (models/gemma4/modeling_gemma4.py): Gemma4VisionModel and
Gemma4AudioModel, then Gemma4MultimodalEmbedder (an RMS norm with no weight
and a projection to the width of the text model).

Vision (MULTIMODAL_PLAN.md, section 3.2):

    patches of 16 x 16 (row, column, channel), 2x - 1 -> the patch linear ->
    + the x row and the y row of the position tables -> layers (RMS norm,
    q k v with an RMS norm of each head, and one with no weight of v, a 2D
    RoPE of theta 100: the first half of a head takes x and the second y,
    attention of all patches with scale 1, the output linear, a post norm;
    RMS norm, the gated FFN with gelu tanh, a post norm) -> the average of
    each 3 x 3 block of patches -> times sqrt(width) -> (the 26B)
    (x - std_bias) * std_scale -> RMS norm -> mm.input_projection

Audio (section 3.3): the mel of transformers (Gemma4AudioFeatureExtractor),
two Conv2d 3 x 3 of stride 2 with a LayerNorm of the channels and ReLU, a
linear, 12 Conformer layers (a half FFN, the local attention of 12 keys with
a relative position term and a softcap of 50, the conv module, a half FFN,
an RMS norm), the output linear, RMS norm, mm.a.input_projection.

A linear of a "clippable" layer clamps its input and its output with the
four scalars of the GGUF (input_min, input_max, output_min, output_max).

The GGUF changes of the llama.cpp converter: the patch linear is a conv of
(channel, row, column); per_dim_scale holds softplus(per_dim_scale); the
depthwise conv has no middle axis. The names of the lconv1d norms: the GGUF
conv_norm is pre_layer_norm of transformers, and norm_conv is conv_norm.
"""
from __future__ import annotations

import itertools
import platform as _platform
import os
import threading
import time
import weakref

import numpy as np

from reclank.core import ops
from reclank.ir import program as P
from reclank.formats.gguf import GGUF


# The encoders run as programs (reclank/ir/program.py, the records ENC_*): in C
# on the CPU, or as one CUDA graph on the GPU. RECLANK_ENC_PY=1 runs the
# NumPy layers of this module instead (the reference of the programs).
_ENC_PY = os.environ.get("RECLANK_ENC_PY", "0") == "1"


def _bf16_exact(a):
    """a as raw bfloat16 (uint16) if every value is a bfloat16 (the float32
    tensors of the mmproj GGUF are upcasts of bfloat16), else a float32."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    if np.any(u & 0xFFFF):
        return a
    return (u >> 16).astype(np.uint16)


# ---- the weights ------------------------------------------------------------

class _Weights:
    """The tensors of an mmproj GGUF: float32 arrays, or raw bfloat16 for the
    large matrices (the C products of ops.linear_bf16)."""

    def __init__(self, path):
        self.g = GGUF(path)
        self.meta = self.g.meta

    def has(self, name):
        return name in self.g.tensors

    def f32(self, name):
        return np.ascontiguousarray(np.asarray(self.g.dequant(name), dtype=np.float32))

    def scalar(self, name):
        return float(self.f32(name).reshape(-1)[0]) if self.has(name) else None

    def matrix(self, name):
        """A (out, in) matrix: raw bfloat16 (uint16) if the GGUF has BF16,
        else float32."""
        raw, dims, t = self.g.raw(name)
        out, inn = int(dims[1]), int(dims[0])
        if t == 30:     # BF16
            return np.frombuffer(raw, dtype=np.uint16).reshape(out, inn)
        return _bf16_exact(self.f32(name).reshape(out, inn))


class _Plain:
    """A linear with no clamps and no bias (the patch linear)."""

    def __init__(self, w):
        self.w = w
        self.b = None
        self.imin = self.imax = self.omin = self.omax = None


class _EncCompiler(P.Compiler):
    """The compiler of an encoder program: no model, only the eps of its
    norms."""

    q4x = False

    def __init__(self, eps, pack=False, q8=False, rows=0, kmax=0):
        self.model = None
        self.cfg = None
        self.eps = eps
        # pack: a CPU program, whose linears read W in groups of 16 rows.
        self.pack = pack
        # q8: the linears with Q8_0 weights (lin.q8) take the int8 records;
        # a CPU program quantizes x into the shared buffers _xq, _xs, _xm.
        self.q8 = q8
        self.p = P.Program()
        self.env = self.p.names
        self.frames = []            # the scopes of P.Compiler (an encoder has none)
        self.frame_kinds = []
        self.n_gensym = 0
        if q8 and pack:
            self.env.update(_xq=np.zeros(rows * kmax, np.int8),
                            _xs=np.zeros(rows * kmax // 32, np.float32),
                            _xm=np.zeros(rows * kmax // 16, np.float32))


def quantize_q8(lin):
    """Give a linear (bfloat16 or float32 W, cols of 32k values) Q8_0
    weights: lin.q8 (rows, for the GPU and a CPU without VNNI) and lin.q8x16
    (groups of 16 rows, for a CPU with VNNI). Return False when W does not
    fit Q8_0 (the 26B FFN has 4304 cols)."""
    from reclank.backends.cpu import cops
    from reclank.models.gemma4.e4b import _KQ
    rows, cols = lin.w.shape
    if cols % 32:
        return False
    data = cops.kq_to_q8_0(lin.w, cols)
    lin.q8 = _KQ(data, cops.KQ_Q8_0, rows, cols)
    lin.q8x16 = None
    # KQ_Q8X16 has a kernel for VNNI and for AVX2 (backends/cpu/csrc/kernels/kq)
    if rows % 16 == 0 and _platform.machine().lower() in ("x86_64", "amd64"):
        lin.q8x16 = _KQ(cops.kq_pack_q8x16(data, rows, cols), cops.KQ_Q8X16, rows, cols)
    return True


def _shape(lin):
    """(rows, cols) of a linear's W, also after to_bf12, to_bf12_dev or to_f20 (W dropped)."""
    if lin.w is not None:
        return lin.w.shape
    q = getattr(lin, "kq", None) or getattr(lin, "bf12", None) or lin.f20
    return (q.rows, q.cols)


def to_bf12(lin):
    """Give a linear of bfloat16 W the BF12 rows of W (lin.kq, for a GPU
    program; the bfloat16 values in 12.25 bits) and drop W, under the rule of
    the runtime (convert_q8_gguf.py --dense bf12): cols a multiple of 32, more
    than one row, no Inf or NaN, the largest zeroed value at most 2^-8 of the
    RMS of the matrix. Return False (W kept) when W does not pass."""
    from reclank.backends.cpu import cops
    from reclank.formats.gguf import BF12
    from reclank.models.gemma4.e4b import _KQ
    rows, cols = lin.w.shape
    if lin.w.dtype != np.uint16 or cols % 32 or rows < 2:
        return False
    data, rep = cops.kq_bf16_to_bf12(lin.w, cols)
    if rep["nonfinite"] or rep["largest"] > rep["rms"] / 256:
        return False
    lin.kq = _KQ(data, BF12, rows, cols)
    lin.w = None
    return True


def to_bf12_dev(lin, gpu, src=None):
    """Give a linear of bfloat16 rows (lin.w, uint16; or lin.kq of KQ_BF16 rows, models.qwen3.dense's
    lin16 on the GPU) their BF12 (the bfloat16 values in 12.25 bits) and drop W: BF12 rows (lin.kq, a
    GPU program's KQ_LINEAR) or KQ_BF12X16 (lin.bf12, a CPU program's ENC_LINEAR: float32 products of
    the BF12 values), under the rule of to_bf12 (cols a multiple of 32, more than one row, no Inf or
    NaN, the largest zeroed value at most 2^-8 of the RMS of the matrix). The pages of the
    checkpoint's map that W, or src, views are dropped (drop_pages). Return False (W kept) when W does
    not pass."""
    from reclank.backends.cpu import cops
    from reclank.formats.gguf import BF12, BF16
    from reclank.formats.safetensors import drop_pages
    from reclank.models.gemma4.e4b import _KQ
    kq = getattr(lin, "kq", None)
    if lin.w is not None and lin.w.dtype == np.uint16:
        w = lin.w
    elif lin.w is None and kq is not None and kq.type == BF16:
        w = np.asarray(kq.data).view(np.uint16).reshape(kq.rows, kq.cols)
    else:
        return False
    rows, cols = w.shape
    if cols % 32 or rows < 2:
        return False
    data, rep = cops.kq_bf16_to_bf12(w, cols)
    if rep["nonfinite"] or rep["largest"] > rep["rms"] / 256:
        return False
    if gpu:
        lin.kq = _KQ(data, BF12, rows, cols)
    else:
        lin.kq = None
        lin.bf12 = _KQ(cops.kq_pack_bf12x16(data, rows, cols), cops.KQ_BF12X16, rows, cols)
        del data
    lin.w = None
    lin.x16 = None
    drop_pages(w if src is None else src)
    return True


def to_f20(lin, gpu, src=None):
    """Give a linear of float32 W its F20 (lin.f20: F20 rows for a GPU program, KQ_F20X16 for a CPU
    program; 16 significant bits a value in 20.25 bits) and drop W, under the rule of to_bf12: cols
    a multiple of 32, no Inf or NaN, the largest zeroed value at most 2^-8 of the RMS of the matrix.
    W's memory goes with it: a copy of W is freed (when nothing else holds it), and the pages of
    the checkpoint's map that W, or src (the tensor W was copied from), views are dropped
    (drop_pages). Return False (W kept) when W does not pass."""
    from reclank.backends.cpu import cops
    from reclank.formats.safetensors import drop_pages
    from reclank.models.gemma4.e4b import _KQ
    w = lin.w
    if w is None or w.dtype != np.float32 or w.ndim != 2 or w.shape[1] % 32:
        return False
    rows, cols = w.shape
    data, rep = cops.kq_f32_to_f20(w, cols)
    if rep["nonfinite"] or rep["largest"] > rep["rms"] / 256:
        return False
    if gpu:
        lin.f20 = _KQ(data, cops.KQ_F20, rows, cols)
    else:
        lin.f20 = _KQ(cops.kq_pack_f20x16(data, rows, cols), cops.KQ_F20X16, rows, cols)
    del data
    lin.w = None
    lin.x16 = None
    drop_pages(w if src is None else src)
    return True


# The 16-bit forms of float32 W for the GPU's tensor cores (to_w16; lin.wq, ENC_LINEAR wbf 7, ENC_CONV_MM
# wbf 3 and 4): one uint8 array of m rows of k values, then a tail of scales (16-byte aligned: k a multiple
# of 32).
#   W16H: the rows float16 of W ws (to nearest even), then ws (m float32): a power of two of each row,
#     max |w| ws in [2^14, 2^15) (no float16 overflow, the least of its subnormals); the products divided
#     by ws, exactly. 11 significant bits a value.
#   W16I: I16 rows (program.i16_row_bytes: hi and lo int8 of each value's int16, then a float32 scale of
#     each group of 64; k a multiple of 64): q = rint(w 32639 / mx), mx the group's largest |w|. 15 bits
#     and a sign a value within its group (ENC_CONV_MM wbf 5 and 6, ENC_LINEAR wbf 8).
KQ_W16H = 90
KQ_W16I = 91


def w16h_pack(w, threads=16):
    """W16H of float32 W (m, k): uint8 (2 m k + 4 m). Blocks of rows on threads (NumPy's float16 casts are slow
    and let the GIL go; each row its own)."""
    w = np.ascontiguousarray(w, np.float32)
    m, k = w.shape
    out = np.empty(2 * m * k + 4 * m, np.uint8)
    h = out[:2 * m * k].view(np.float16).reshape(m, k)
    ws = out[2 * m * k:].view(np.float32)

    def rows(r0, r1):
        mx = np.abs(w[r0:r1]).max(axis=1)
        e = np.where(mx > 0, 14 - np.floor(np.log2(np.where(mx > 0, mx, 1.0))), 0.0)
        ws[r0:r1] = np.ldexp(np.float32(1), e.astype(np.int32)).astype(np.float32)
        h[r0:r1] = (w[r0:r1] * ws[r0:r1, None]).astype(np.float16)
    n = max(1, min(threads, m * k >> 18, m))
    if n == 1:
        rows(0, m)
        return out
    from concurrent.futures import ThreadPoolExecutor

    from reclank.models.media import widen
    cut = np.linspace(0, m, n + 1).astype(int)
    with ThreadPoolExecutor(n, initializer=widen) as pool:
        list(pool.map(rows, cut[:-1], cut[1:]))
    return out


def w16h_values(data, m, k):
    """The float32 W of W16H data (the values the tensor cores multiply, divided by ws)."""
    h = data[:2 * m * k].view(np.float16).reshape(m, k).astype(np.float32)
    return h / data[2 * m * k:].view(np.float32)[:m, None]


def w16i_pack(w):
    """W16I of float32 W (m, k; k a multiple of 64): uint8 (m, i16_row_bytes(k)), k_enc_i16_pack's rule."""
    w = np.ascontiguousarray(w, np.float32)
    m, k = w.shape
    assert k % 64 == 0, k
    g = w.reshape(m, k // 64, 64)
    mx = np.abs(g).max(axis=2)
    ok = (mx > 0) & np.isfinite(mx)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(ok, np.float32(32639) / np.where(ok, mx, 1), 0).astype(np.float32)
        sc = np.where(ok, mx / np.float32(32639), 0).astype(np.float32)
    q = np.clip(np.rint(g * inv[:, :, None]), -32639, 32639).astype(np.int32).reshape(m, k)
    hi = (q + 128) >> 8
    lo = q - (hi << 8)
    out = np.zeros((m, P.i16_row_bytes(k)), np.uint8)
    out[:, :k] = hi.astype(np.int8).view(np.uint8)
    out[:, k:2 * k] = lo.astype(np.int8).view(np.uint8)
    out[:, 2 * k:2 * k + 4 * (k // 64)] = sc.view(np.uint8).reshape(m, -1)
    return out


def w16i_values(data, m, k):
    """The float32 W of W16I data (q s: the values the tensor cores multiply)."""
    d = np.asarray(data).reshape(m, -1)
    q = d[:, :k].view(np.int8).astype(np.int32) * 256 + d[:, k:2 * k].view(np.int8).astype(np.int32)
    sc = np.ascontiguousarray(d[:, 2 * k:2 * k + 4 * (k // 64)]).view(np.float32)
    return (q.reshape(m, k // 64, 64) * sc[:, :, None]).reshape(m, k).astype(np.float32)


W16_FORMATS = ("f16", "i16")


def to_w16(lin, fmt, src=None):
    """Give a linear of float32 W (a GPU program's) its 16-bit form fmt (lin.wq: W16H for "f16", W16I for
    "i16") and drop W (and its pages, as to_f20). k a multiple of 32 (64 for "i16"), no Inf or NaN.
    Return False (W kept) when W does not pass."""
    from reclank.formats.safetensors import drop_pages
    from reclank.models.gemma4.e4b import _KQ
    assert fmt in W16_FORMATS, fmt
    w = lin.w
    if w is None or w.dtype != np.float32 or w.ndim != 2 or w.shape[1] % (64 if fmt == "i16" else 32) or \
            not np.isfinite(w).all():
        return False
    m, k = w.shape
    lin.wq = _KQ(w16i_pack(w), KQ_W16I, m, k) if fmt == "i16" else _KQ(w16h_pack(w), KQ_W16H, m, k)
    lin.w = None
    lin.x16 = None
    drop_pages(w if src is None else src)
    return True


def lin_fmt():
    """The form of the GPU's float32 matrices made by F20Lins: RECLANK_W16 ("f20", the default; "f16",
    "i16")."""
    f = os.environ.get("RECLANK_W16", "f20")
    assert f == "f20" or f in W16_FORMATS, "RECLANK_W16=%s: f20 or one of %s" % (f, W16_FORMATS)
    return f


def lin_q(lin):
    """(attribute, matrix) of a linear's reduced W: ("f20", lin.f20) or ("wq", lin.wq), else (None, None)."""
    for a in ("f20", "wq"):
        q = getattr(lin, a, None)
        if q is not None:
            return a, q
    return None, None


class F20Lins:
    """Linears to F20 one at a time, as a model makes them (add: to_f20, so that the float32 W of
    only one matrix is in memory beside the F20 ones); done prints the count and the memory. fmt (a
    GPU's: "f16", "i16"; None: lin_fmt()) one of the 16-bit forms instead (to_w16)."""

    def __init__(self, gpu, label, fmt=None):
        self.gpu, self.label = gpu, label
        self.fmt = (lin_fmt() if gpu else "f20") if fmt is None else fmt
        assert self.fmt == "f20" or gpu, "the 16-bit forms are the GPU's"
        self.done = self.kept = self.before = self.after = 0

    def add(self, lin, src=None):
        w = lin.w
        if w is None or lin_q(lin)[1] is not None:
            return
        nbytes = w.nbytes
        del w
        if to_f20(lin, self.gpu, src) if self.fmt == "f20" else to_w16(lin, self.fmt, src):
            self.done += 1
            self.before += nbytes
            self.after += lin_q(lin)[1].data.nbytes
        else:
            self.kept += 1

    def report(self):
        print("%s: %d matrices %s (%.2f -> %.2f GB)%s" % (
            self.label, self.done, self.fmt.upper(), self.before / 1e9, self.after / 1e9,
            ", %d stay float32" % self.kept if self.kept else ""), flush=True)


def f20_lins(lins, gpu, label):
    """The float32 matrices of lins in F20 (to_f20, each W dropped after its F20); a matrix that the
    rule refuses keeps its float32 W."""
    f = F20Lins(gpu, label)
    for lin in lins:
        f.add(lin)
    f.report()


class _Use:
    """A scope of Residency.use: its runners, pinned from their first GPU run in it (held) until
    done or its end."""

    def __init__(self, res, runners):
        self.res = res
        self.runners = set(r for r in runners if r is not None)
        self.held, self.finished = set(), set()

    def __enter__(self):
        self.res._scopes().append(self)
        return self

    def __exit__(self, *exc):
        with self.res.lock:
            stack = self.res._scopes()
            if self in stack:
                stack.remove(self)
            for r in self.held:
                self.res._unpin(r)
            self.held.clear()
        return False

    def hold(self, r):
        if r in self.runners and r not in self.held and r not in self.finished:
            self.held.add(r)
            self.res._pin(r)

    def done(self, *runners):
        """These runners' part of the request is over (Wan's DiT before its VAE): they may go from
        now on."""
        with self.res.lock:
            for r in runners:
                if r is None:
                    continue
                self.finished.add(r)
                if r in self.held:
                    self.held.discard(r)
                    self.res._unpin(r)


class Residency:
    """Which GPU runners hold device memory: one manager for the process (RESIDENCY).

    Every _Runner registers. A runner's resident bytes (gpu_bytes) are its mirror's Buffers (the
    weights and the buffers of its programs) and what the C side took for its programs (their
    scratch, code and graphs: the drop of the free memory at a build, less the mirror's growth).
    Before a GPU build (_Runner.get), make_room(runner, need) evicts the unpinned runners, the
    least recently used first, with release(keep_host=True) (the next run uploads the host arrays
    again), until cudaMemGetInfo's free memory covers need + reserve, and the runners' bytes +
    need fit budget (None: no budget; over it with nothing left to evict, MemoryError). A
    cudaMalloc that fails in a build anyway (MemoryError): every other unpinned runner goes, and
    the build runs once more.

    Pins: pin(runner) keeps a runner for good (the server's --keep, --pin). use(*runners) is a
    request's scope: each of its runners is pinned from its first GPU run in the scope (its device
    state, a cache or a text's keys and values, lives from then on) until use.done(runner) or the
    scope's end; a runner not run yet in the scope can still go. A run outside any scope pins
    nothing: a part whose state spans several runs (a prefix, then the steps) runs in a scope.

    enabled: the manager acts (evictions, the budget, the retry; a build also records its graphs
    then, g.prepare). Off by default: a _Runner then runs as it did before, and the registry only
    counts (the LLM servers, the scripts). The generation and embedding servers turn it on, and so
    do RECLANK_RESIDENCY=1 and always. peers: the other servers on the GPU (server/gpushare.py),
    asked for room when this process's victims are not enough.

    always (RECLANK_EVICT_ALWAYS=1): every unpinned runner goes before every GPU run (and the
    runner itself, when unpinned: its run uploads again), to check that a swap keeps the bits;
    RECLANK_EVICT_ALWAYS=2: and the CUDA context too when nothing is left (release_all), so every
    run is in a new context (docs/GPU_SWAP_PLAN.md 3.6).
    verbose (RECLANK_RESIDENCY_LOG=1, the server unless --quiet) prints each eviction and each
    load with its time; log keeps the last 256 of them (kind, label, bytes, seconds)."""

    def __init__(self, mem_info=None, reserve=None, budget=None, always=None, enabled=None):
        self._mem_info = mem_info
        self.reserve = int(float(os.environ.get("RECLANK_GPU_RESERVE", "512e6"))) if reserve is None else reserve
        self.budget = budget
        ev = os.environ.get("RECLANK_EVICT_ALWAYS", "0")
        self.always = ev in ("1", "2") if always is None else always
        self.always_context = ev == "2"
        self.enabled = (os.environ.get("RECLANK_RESIDENCY", "0") == "1" or self.always) if enabled is None else enabled
        self.verbose = os.environ.get("RECLANK_RESIDENCY_LOG", "0") == "1"
        self.lock = threading.RLock()
        self.state = weakref.WeakKeyDictionary()    # runner: [last use, pins]
        self.kept = weakref.WeakSet()               # pinned for good
        self.evicted = weakref.WeakSet()            # gone since its last load (its next load is a swap)
        self.ticks = itertools.count(1)
        self.local = threading.local()              # the scopes of use of this thread
        self.log = []
        self.peers = None                           # the other servers (server/gpushare.py Share)

    def free(self):
        return self._info()[0]

    def _info(self):
        if self._mem_info is None:
            from reclank.backends.cuda.gpumm import mem_info
            self._mem_info = mem_info
        return self._mem_info()

    def _scopes(self):
        s = getattr(self.local, "scopes", None)
        if s is None:
            s = self.local.scopes = []
        return s

    def register(self, r):
        with self.lock:
            if r not in self.state:
                self.state[r] = [0, 0]

    def label(self, r):
        return getattr(r, "label", None) or type(r).__name__

    def total(self):
        """The resident bytes of all runners."""
        return sum(r.gpu_bytes() for r in list(self.state.keys()))

    def pinned(self, r):
        return r in self.kept or self.state.get(r, (0, 0))[1] > 0

    def pin(self, r):
        """Keep r on the GPU for good (its memory is never evicted)."""
        self.register(r)
        self.kept.add(r)

    def unpin(self, r):
        self.kept.discard(r)

    def _pin(self, r):
        self.register(r)
        self.state[r][1] += 1

    def _unpin(self, r):
        st = self.state.get(r)
        if st is not None and st[1] > 0:
            st[1] -= 1

    def use(self, *runners):
        """with use(*runners) as u: a request's scope (u.done(runner): an early end of a part)."""
        return _Use(self, runners)

    def before_run(self, r):
        """At a GPU run of r (_Runner.get): with always, every unpinned runner goes first (r too);
        r is held by the scopes that name it; its last use is now."""
        with self.lock:
            self.register(r)
            if self.always and self.enabled:
                if self.always_context:
                    self.release_all(context=True)
                else:
                    self.evict_unpinned()
            for s in self._scopes():
                s.hold(r)
            self.state[r][0] = next(self.ticks)
            if self.peers is not None:
                self.peers.used()

    def _note(self, kind, r, nbytes, secs):
        self.log = self.log[-255:] + [(kind, self.label(r), int(nbytes), secs)]
        if self.verbose:
            print("gpu: %s %s (%.2f GB) in %.2f s" % (kind, self.label(r), nbytes / 1e9, secs), flush=True)

    def evict(self, r):
        """Free r's GPU memory (its host programs stay). Return the bytes it held."""
        with self.lock:
            nb = r.gpu_bytes()
            if nb <= 0:
                return 0
            t0 = time.time()
            r.release(keep_host=True)
            self.evicted.add(r)
            self._note("evicted", r, nb, time.time() - t0)
            return nb

    def victims(self, keep=None):
        """The unpinned runners that hold memory (not keep), the least recently used first."""
        with self.lock:
            out = [(st[0], i, r) for i, (r, st) in enumerate(list(self.state.items()))
                   if r is not keep and not self.pinned(r) and r.gpu_bytes() > 0]
        return [r for _t, _i, r in sorted(out, key=lambda x: (x[0], x[1]))]

    def evict_unpinned(self, keep=None):
        """Evict every unpinned runner but keep. Return the bytes freed."""
        return sum(self.evict(r) for r in self.victims(keep))

    def release_all(self, context=True):
        """Evict every unpinned runner; with context, when no runner holds device memory then, the
        CUDA context too (gpumm.context_release: it refuses, and the context stays, while any Buffer
        or program of the process is left). Return (the bytes freed, the context released)."""
        with self.lock:
            nb = self.evict_unpinned()
            if not context or any(r.gpu_bytes() > 0 for r in list(self.state.keys())):
                return nb, False
            from reclank.backends.cuda import gpumm
            t0 = time.time()
            try:
                gpumm.mem().poll(wait=True)
                gpumm.context_release()
            except RuntimeError as e:
                if self.verbose:
                    print("gpu: the context stays (%s)" % e, flush=True)
                return nb, False
            self.log = self.log[-255:] + [("context", "released", 0, time.time() - t0)]
            if self.verbose:
                print("gpu: the context released in %.2f s" % (time.time() - t0), flush=True)
            return nb, True

    def short(self, need):
        """The bytes missing for need: of the free memory (with the reserve), and of the budget."""
        s = need + self.reserve - self.free()
        if self.budget is not None:
            s = max(s, self.total() + need - self.budget)
        return s

    def make_room(self, runner, need, sure=None):
        """Evict unpinned runners (not runner), the least recently used first, until need bytes
        fit (short). MemoryError when sure (the part of need that is not an estimate; None: all of
        it) does not fit the budget with all of them gone. Still short with peers (the other
        servers: server/gpushare.py), they are asked for the rest; True then, the arbiter held until
        the load ends (peers.done)."""
        sure = need if sure is None else sure
        with self.lock:
            for v in self.victims(runner):
                if self.short(need) <= 0:
                    break
                self.evict(v)
            if self.budget is not None and self.total() + sure > self.budget:
                raise MemoryError("%s needs %.2f GB: the GPU budget is %.2f GB, %.2f GB of it pinned or in use" % (
                    self.label(runner), sure / 1e9, self.budget / 1e9, self.total() / 1e9))
            short = self.short(need) > 0 and self.peers is not None
            if short:
                card = self._info()[1]
                mine = sum(r.gpu_bytes() for r in list(self.state.keys()) if self.pinned(r) or r is runner)
        if short:
            # the peers (outside the lock: a peer's request may take this server's parts not in use
            # meanwhile); the arbiter stays held until the load is done (load)
            self.peers.room(lambda: self.short(need), need, self.label(runner), card, self.reserve, mine, sure)
        return short

    def load(self, runner, need, make, sure=None):
        """make() (a GPU build of runner's, need bytes, sure of them known) after make_room; when it
        fails for memory, every other unpinned runner goes and make runs once more. Not enabled:
        make() alone."""
        if not self.enabled:
            return make()
        asked = self.make_room(runner, need, sure)
        t0 = time.time()
        try:
            try:
                out = make()
            except MemoryError:
                with self.lock:
                    self.evict_unpinned(keep=runner)
                t0 = time.time()
                try:
                    out = make()
                except MemoryError as e:
                    note = getattr(self.peers, "note", None) if asked else None
                    if note:
                        raise MemoryError("%s (%s)" % (e, note)) from e
                    raise
        finally:
            if asked:
                self.peers.done()
        self._note("reloaded" if runner in self.evicted else "loaded", runner, getattr(runner, "last_load", need),
                   time.time() - t0)
        self.evicted.discard(runner)
        return out


RESIDENCY = Residency()


def gpu_runners(gpu, **parts):
    """{name: the part's _Runner} of the parts that are on the GPU (not None; none on the CPU): a
    pipeline's runners()."""
    return {k: p.runner for k, p in parts.items() if p is not None} if gpu else {}


class _Runner:
    """The programs of an encoder by size: in C on the CPU, or on the GPU
    (gpu.GPUProgram, with one mirror of the weights for all sizes). Its GPU
    memory is in RESIDENCY's hands: it may go between runs (release with
    keep_host) unless pinned."""

    C_EXTRA = 64 << 20      # (_need: the C side's bytes of a first build, past the largest float32 array)

    def __init__(self, build, keep=4, keep_gpu=2):
        self.build = build
        self.keep = keep
        self.keep_gpu = keep_gpu    # GPU programs hold device memory: fewer of them
        self.progs = {}
        self.mirror = None
        self.hosts = {}             # release(keep_host): the programs whose GPU copies went
        self.label = getattr(build, "__qualname__", None) or "runner"
        self.scratch = {}           # key: the C side's bytes of its GPU program (code, graphs, scratch)
        self.seen = {}              # key: the same at its last build (the need of the next)
        self.last_load = 0
        RESIDENCY.register(self)

    def gpu_bytes(self):
        """The device memory of the runner: its mirror's and its programs'."""
        return (self.mirror.nbytes() if self.mirror is not None else 0) + sum(self.scratch.values())

    def _need(self, key, prog):
        """The device bytes a GPU build of prog takes, and the part of them that is known (not an
        estimate): the arrays it reads that are not on the device, and the C side's of its last
        build at this key (the first time an estimate: the largest float32 array, about the float16
        planes of a product's rows, plus 64 MB)."""
        mir = self.mirror
        seen, new, largest = set(), 0, 0
        for a in prog.keep:
            if not isinstance(a, np.ndarray):
                continue
            while isinstance(a.base, np.ndarray):
                a = a.base
            s = a.ctypes.data
            if s in seen:
                continue
            seen.add(s)
            if a.dtype == np.float32:
                largest = max(largest, a.nbytes)
            if mir is None or s not in mir.bufs:
                new += a.nbytes
        c = self.seen.get(key)
        return (new + c, new + c) if c is not None else (new + largest + self.C_EXTRA, new)

    def _load(self, key, prog):
        """The GPU program of prog: its arrays uploaded, its scratch and graphs made."""
        from reclank.backends.cuda.gpumm import mem_info
        from reclank.backends.cuda.runtime import GPUProgram, Mirror
        if self.mirror is None:
            self.mirror = Mirror()
            self.mirror.kind = "encoder"     # GpuMem
        free0, mir0 = mem_info()[0], self.mirror.nbytes()
        g = GPUProgram(prog, graph=True, mirror=self.mirror)
        if RESIDENCY.enabled:
            try:
                g.prepare()     # (its graphs now: their memory is the build's)
            except MemoryError:
                g.close()
                raise
        grew = self.mirror.nbytes() - mir0
        self.scratch[key] = self.seen[key] = max(0, free0 - mem_info()[0] - grew)
        self.last_load = grew + self.scratch[key]
        return g

    def get(self, n, gpu):
        key = (n, bool(gpu))
        if gpu:
            RESIDENCY.before_run(self)
        entry = self.progs.pop(key, None)
        if entry is None:
            prog = self.hosts.pop(key, None)
            prog = self.build(n, gpu) if prog is None else prog
            g = None
            if gpu:
                # room first: the GPU programs over keep_gpu - 1 go before the
                # new one takes its memory (the weights it reads stay)
                gpu_keys = [k for k in self.progs if k[1]]
                while gpu_keys and len(gpu_keys) >= self.keep_gpu:
                    self._drop(gpu_keys.pop(0), keep=prog)
                if RESIDENCY.enabled:
                    need, sure = self._need(key, prog)
                    g = RESIDENCY.load(self, need, lambda: self._load(key, prog), sure)
                else:
                    g = self._load(key, prog)
            entry = (prog, g)
        self.progs[key] = entry
        while len(self.progs) > self.keep:
            self._drop(next(iter(self.progs)))
        gpu_keys = [k for k in self.progs if k[1]]
        while len(gpu_keys) > self.keep_gpu:
            self._drop(gpu_keys.pop(0))
        return entry

    def _drop(self, key, keep=None):
        """Forget a program; a GPU program frees its device memory now (it
        has no finalizer that unloads it): its scratch, and the device copies
        of its arrays (in the mirror of all sizes) that no other program, nor
        keep (a program about to load), reads. Else each new size of an
        image kept the buffers of the old sizes until release()."""
        prog, g = self.progs.pop(key)
        if g is None:
            return
        self.scratch.pop(key, None)
        g.close()
        for b in getattr(g, "named", {}).values():
            b.free()
        if self.mirror is None:
            return

        def starts(p):
            out = set()
            for a in p.keep:
                if isinstance(a, np.ndarray):
                    while isinstance(a.base, np.ndarray):
                        a = a.base
                    out.add(a.ctypes.data)
            return out
        live = starts(keep) if keep is not None else set()
        for p, g2 in self.progs.values():
            if g2 is not None:
                live |= starts(p)
        mir = self.mirror
        for s in starts(prog) - live:
            b = mir.bufs.pop(s, None)
            if b is not None:
                b.free()
            if s in mir.arrays:
                del mir.arrays[s]
                mir.starts.remove(s)

    def release(self, keep_host=False):
        """Free the GPU memory of the programs and of the weights (a model
        that needs the room between media requests). The next GPU run
        builds them again (keep_host: from the programs kept on the host, a
        VAE's of 2.8 s)."""
        if RESIDENCY.enabled and (self.mirror is not None or self.scratch):
            from reclank.backends.cuda.gpumm import mem
            mem().poll(wait=True)       # (the runs queued before: their memory goes now)
        for key in [k for k in self.progs if k[1]]:
            if keep_host:
                self.hosts.pop(key, None)
                self.hosts[key] = self.progs[key][0]
            self._drop(key)
        while RESIDENCY.enabled and len(self.hosts) > self.keep:      # (the oldest sizes' host programs)
            self.hosts.pop(next(iter(self.hosts)))
        if self.mirror is not None:
            for b in self.mirror.bufs.values():
                b.free()
            self.mirror = None
        self.scratch.clear()

    def run(self, n, gpu, inputs, output):
        """Write the inputs (name: array), run, and return the output."""
        prog, g = self.get(n, gpu)
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
            g.download(output)
        else:
            prog.run()
        return prog.names[output].copy()


class _Linear:
    """A linear layer with the clamps of a clippable linear, and a bias."""

    def __init__(self, W, name, bias=False):
        self.w = W.matrix(name + ".weight")
        self.b = W.f32(name + ".bias").reshape(-1) if bias else None
        base = name
        self.imin, self.imax = W.scalar(base + ".input_min"), W.scalar(base + ".input_max")
        self.omin, self.omax = W.scalar(base + ".output_min"), W.scalar(base + ".output_max")

    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.imin is not None:
            x = np.clip(x, self.imin, self.imax)
        if self.w.dtype == np.uint16:
            y = ops.linear_bf16(x, self.w)
        else:
            y = x @ self.w.T
        y = y.astype(np.float32, copy=False)
        if self.b is not None:
            y += self.b
        if self.omin is not None:
            np.clip(y, self.omin, self.omax, out=y)
        return y


def _rms(x, w=None, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    y = x * np.power((x * x).mean(axis=-1, keepdims=True) + eps, -0.5, dtype=np.float32)
    return y * w if w is not None else y


def _gelu_tanh(x):
    return ops.gelu_tanh(x)


def _silu(x):
    # x * sigmoid(x), with no overflow of exp for a large negative x
    return x * (np.float32(0.5) * (1.0 + np.tanh(np.float32(0.5) * x)))


def _softmax(s, axis=-1):
    m = s.max(axis=axis, keepdims=True)
    e = np.exp(s - m)
    return e / e.sum(axis=axis, keepdims=True)
