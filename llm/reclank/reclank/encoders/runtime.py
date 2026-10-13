"""The image and audio encoders of the Gemma 4 E2B, E4B, and 26B (gemma4v, gemma4a).

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

import platform as _platform
import os

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


class _Runner:
    """The programs of an encoder by size: in C on the CPU, or on the GPU
    (gpu.GPUProgram, with one mirror of the weights for all sizes)."""

    def __init__(self, build, keep=4, keep_gpu=2):
        self.build = build
        self.keep = keep
        self.keep_gpu = keep_gpu    # GPU programs hold device memory: fewer of them
        self.progs = {}
        self.mirror = None

    def get(self, n, gpu):
        key = (n, bool(gpu))
        entry = self.progs.pop(key, None)
        if entry is None:
            prog = self.build(n, gpu)
            g = None
            if gpu:
                from reclank.backends.cuda.runtime import GPUProgram, Mirror
                # room first: the GPU programs over keep_gpu - 1 go before the
                # new one takes its memory (the weights it reads stay)
                gpu_keys = [k for k in self.progs if k[1]]
                while gpu_keys and len(gpu_keys) >= self.keep_gpu:
                    self._drop(gpu_keys.pop(0), keep=prog)
                if self.mirror is None:
                    self.mirror = Mirror()
                    self.mirror.kind = "encoder"     # GpuMem
                g = GPUProgram(prog, graph=True, mirror=self.mirror)
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

    def release(self):
        """Free the GPU memory of the programs and of the weights (a model
        that needs the room between media requests). The next GPU run
        builds them again."""
        for key in [k for k in self.progs if k[1]]:
            self._drop(key)
        if self.mirror is not None:
            for b in self.mirror.bufs.values():
                b.free()
            self.mirror = None

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
