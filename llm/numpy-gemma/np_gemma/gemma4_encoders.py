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

import math
import platform as _platform

import numpy as np

import os

from . import ops
from . import program as P
from .gguf import GGUF

# The encoders run as programs (np_gemma/program.py, the records ENC_*): in C
# on the CPU, or as one CUDA graph on the GPU. NP_GEMMA_ENC_PY=1 runs the
# NumPy layers of this module instead (the reference of the programs).
_ENC_PY = os.environ.get("NP_GEMMA_ENC_PY", "0") == "1"


def _bf16_exact(a):
    """a as raw bfloat16 (uint16) if every value is a bfloat16 (the float32
    tensors of the mmproj GGUF are upcasts of bfloat16), else a float32."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    if np.any(u & 0xFFFF):
        return a
    return (u >> 16).astype(np.uint16)

PATCH = 16
POOL = 3


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
        if q8 and pack:
            self.env.update(_xq=np.zeros(rows * kmax, np.int8),
                            _xs=np.zeros(rows * kmax // 32, np.float32),
                            _xm=np.zeros(rows * kmax // 16, np.float32))


def quantize_q8(lin):
    """Give a linear (bfloat16 or float32 W, cols of 32k values) Q8_0
    weights: lin.q8 (rows, for the GPU and a CPU without VNNI) and lin.q8x16
    (groups of 16 rows, for a CPU with VNNI). Return False when W does not
    fit Q8_0 (the 26B FFN has 4304 cols)."""
    from . import cops
    from .e4b import _KQ
    rows, cols = lin.w.shape
    if cols % 32:
        return False
    data = cops.kq_to_q8_0(lin.w, cols)
    lin.q8 = _KQ(data, cops.KQ_Q8_0, rows, cols)
    lin.q8x16 = None
    # KQ_Q8X16 has a kernel for VNNI and for AVX2 (csrc/kquants.c)
    if rows % 16 == 0 and _platform.machine().lower() in ("x86_64", "amd64"):
        lin.q8x16 = _KQ(cops.kq_pack_q8x16(data, rows, cols), cops.KQ_Q8X16, rows, cols)
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
                from .gpu import GPUProgram, Mirror
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


# ---- vision -------------------------------------------------------------------

class Gemma4Vision:
    """The gemma4v encoder of an mmproj GGUF."""

    def __init__(self, W, eps=None):
        m = W.meta
        if m.get("clip.vision.projector_type") != "gemma4v":
            raise ValueError("not a gemma4v mmproj (projector %r)" % m.get("clip.vision.projector_type"))
        self.width = int(m["clip.vision.embedding_length"])
        self.layers = int(m["clip.vision.block_count"])
        self.heads = int(m["clip.vision.attention.head_count"])
        self.hd = self.width // self.heads
        self.eps = eps or float(m.get("clip.vision.attention.layer_norm_epsilon", 1e-6))
        # The patch linear: the GGUF conv (width, channel, row, column) back
        # to the linear of transformers (width, row * column * channel).
        pw = W.f32("v.patch_embd.weight").reshape(self.width, 3, PATCH, PATCH)
        self.patch_w = np.ascontiguousarray(pw.transpose(0, 2, 3, 1).reshape(self.width, -1))
        self.patch_lin = _Plain(_bf16_exact(self.patch_w))
        self.runner = _Runner(self.program)
        self.pos = W.f32("v.position_embd.weight").reshape(2, -1, self.width)
        self.std = ((W.f32("v.std_bias").reshape(-1), W.f32("v.std_scale").reshape(-1))
                    if W.has("v.std_bias") else None)
        self.blk = []
        for i in range(self.layers):
            p = "v.blk.%d." % i
            self.blk.append(dict(
                ln1=W.f32(p + "ln1.weight").reshape(-1),
                q=_Linear(W, p + "attn_q"), k=_Linear(W, p + "attn_k"),
                v=_Linear(W, p + "attn_v"), o=_Linear(W, p + "attn_out"),
                qn=W.f32(p + "attn_q_norm.weight").reshape(-1),
                kn=W.f32(p + "attn_k_norm.weight").reshape(-1),
                post_attn=W.f32(p + "attn_post_norm.weight").reshape(-1),
                ln2=W.f32(p + "ln2.weight").reshape(-1),
                gate=_Linear(W, p + "ffn_gate"), up=_Linear(W, p + "ffn_up"),
                down=_Linear(W, p + "ffn_down"),
                post_ffn=W.f32(p + "ffn_post_norm.weight").reshape(-1)))
        self.proj = _Linear(W, "mm.input_projection")
        # The axial rope: hd / 4 frequencies of theta 100 (transformers
        # Gemma4VisionRotaryEmbedding).
        half = self.hd // 2
        self.inv = (1.0 / (100.0 ** (np.arange(0, half, 2, dtype=np.float32) / half))).astype(np.float32)
        # The program on the GPU (True) or in C on the CPU (False); the
        # weights go to the GPU at the first image.
        self.gpu = False
        self.q8 = False

    def _rope(self, x, cos, sin):
        """x (n, heads, hd): the first half of each head rotates with x
        positions, the second with y (NEOX halves inside each part)."""
        h2 = self.hd // 2
        out = np.empty_like(x)
        for part in range(2):
            a = x[..., part * h2:(part + 1) * h2]
            c = cos[part][:, None, :]
            s = sin[part][:, None, :]
            q = h2 // 2
            rot = np.concatenate([-a[..., q:], a[..., :q]], axis=-1)
            out[..., part * h2:(part + 1) * h2] = a * c + rot * s
        return out

    def _attention(self, q, k, v):
        """Attention of all patches, scale 1. q, k, v (n, heads, hd). The C
        flash kernel of the prompt serves it: with every query at the last
        position, every key is visible (0.05 s for 2394 patches of the E4B,
        against 2.1 s in NumPy on one thread)."""
        n = q.shape[0]
        if ops.flash_ready():
            from . import cops
            hd = q.shape[-1]
            pad = (-hd) % 16
            if pad:
                # The C kernel takes a head of 16k values (the 26B has 72):
                # zeros change neither the scores nor the values.
                z = ((0, 0), (0, 0), (0, pad))
                q, k, v = np.pad(q, z), np.pad(k, z), np.pad(v, z)
            o = cops.attn_prefill(q, k, v, np.full(n, n - 1, np.int32), 0, 0)
            return np.ascontiguousarray(o[..., :hd]) if pad else o
        out = np.empty_like(q)
        step = max(1, (1 << 26) // max(1, n * n))          # heads at a time
        for h0 in range(0, self.heads, step):
            h1 = min(self.heads, h0 + step)
            qs = q[:, h0:h1].transpose(1, 0, 2)
            ks = k[:, h0:h1].transpose(1, 2, 0)
            p = _softmax(np.matmul(qs, ks))
            out[:, h0:h1] = np.matmul(p, v[:, h0:h1].transpose(1, 0, 2)).transpose(1, 0, 2)
        return out

    def encode(self, patches, pos):
        """patches (n, 768) in the order row, column, channel, values in
        [0, 1]; pos (n, 2) of (x, y). Return the soft rows (n / 9, text
        width), in the order of the 3 x 3 blocks (rows of blocks)."""
        pos = np.asarray(pos, dtype=np.int64)
        if _ENC_PY:
            x = self._layers(patches, pos)
        else:
            n = patches.shape[0]
            pe = self.pos[0][pos[:, 0]] + self.pos[1][pos[:, 1]]
            x = self.runner.run(n, self.gpu, {
                "p": 2.0 * np.asarray(patches, np.float32) - 1.0, "pe": pe,
                "pos": pos.astype(np.int32)}, "x")
        return self._pool_project(x, pos)

    def program(self, n, gpu=False):
        """The program of the layers for n patches (the records ENC_*):
        inputs "p" (the patches, 2x - 1), "pe" (the rows of the position
        tables), "pos" (x, y); output "x" (n, width), before the pool."""
        w, H, hd = self.width, self.heads, self.hd
        inter = self.blk[0]["gate"].w.shape[0]
        kp = self.patch_w.shape[1]
        c = _EncCompiler(self.eps, pack=not gpu, q8=self.q8, rows=n, kmax=max(kp, w, inter))
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(p=z(n, kp), pe=z(n, w), pos=np.zeros((n, 2), np.int32),
                     x=z(n, w), h=z(n, w), q=z(n, w), k=z(n, w), v=z(n, w), o=z(n, w),
                     t=z(n, w), g=z(n, inter), u=z(n, inter),
                     _scratch=np.zeros(n * max(kp, w, inter), np.float32))
        forms = [("set", "x", ("enc_linear", self.patch_lin, "p")), ("enc_add", "x", "pe")]
        for b in self.blk:
            forms += [
                ("set", "h", ("enc_rms", "x", b["ln1"], w)),
                ("set", "q", ("enc_linear", b["q"], "h")),
                ("set", "q", ("enc_rms", "q", b["qn"], hd)),
                ("set", "k", ("enc_linear", b["k"], "h")),
                ("set", "k", ("enc_rms", "k", b["kn"], hd)),
                ("set", "v", ("enc_linear", b["v"], "h")),
                ("set", "v", ("enc_rms", "v", None, hd)),
                ("enc_rope2d", "q", "pos", self.inv, H, hd),
                ("enc_rope2d", "k", "pos", self.inv, H, hd),
                ("set", "o", ("enc_attn", "q", "k", "v", H, hd)),
                ("set", "t", ("enc_linear", b["o"], "o")),
                ("set", "t", ("enc_rms", "t", b["post_attn"], w)),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", b["ln2"], w)),
                ("set", "g", ("enc_linear", b["gate"], "h")),
                ("set", "u", ("enc_linear", b["up"], "h")),
                ("set", "g", ("enc_gelu_mul", "g", "u")),
                ("set", "t", ("enc_linear", b["down"], "g")),
                ("set", "t", ("enc_rms", "t", b["post_ffn"], w)),
                ("enc_add", "x", "t"),
            ]
        c.compile(("seq", *forms))
        return c.p.finish()

    def _layers(self, patches, pos):
        """The patch linear and the layers on the CPU: (n, width)."""
        n = patches.shape[0]
        x = (2.0 * np.asarray(patches, np.float32) - 1.0) @ self.patch_w.T
        x = x + self.pos[0][pos[:, 0]] + self.pos[1][pos[:, 1]]
        fx = pos[:, 0:1].astype(np.float32) * self.inv[None, :]
        fy = pos[:, 1:2].astype(np.float32) * self.inv[None, :]
        cos = [np.concatenate([np.cos(f), np.cos(f)], axis=-1) for f in (fx, fy)]
        sin = [np.concatenate([np.sin(f), np.sin(f)], axis=-1) for f in (fx, fy)]
        H, hd = self.heads, self.hd
        for b in self.blk:
            h = _rms(x, b["ln1"], self.eps)
            q = _rms(b["q"](h).reshape(n, H, hd), b["qn"], self.eps)
            k = _rms(b["k"](h).reshape(n, H, hd), b["kn"], self.eps)
            v = _rms(b["v"](h).reshape(n, H, hd), None, self.eps)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            a = self._attention(q, k, v).reshape(n, H * hd)
            x = x + _rms(b["o"](a), b["post_attn"], self.eps)
            h = _rms(x, b["ln2"], self.eps)
            f = b["down"](_gelu_tanh(b["gate"](h)) * b["up"](h))
            x = x + _rms(f, b["post_ffn"], self.eps)
        return x

    def _pool_project(self, x, pos):
        """The average of each 3 x 3 block, sqrt(width), the standardizing,
        and the projection to the text model."""
        # The average of each 3 x 3 block, in the order kx + (cols / 3) ky.
        cols = int(pos[:, 0].max()) + 1
        kid = (pos[:, 0] // POOL) + (cols // POOL) * (pos[:, 1] // POOL)
        m = int(kid.max()) + 1
        pooled = np.zeros((m, self.width), dtype=np.float32)
        np.add.at(pooled, kid, x)
        pooled = pooled / np.float32(POOL * POOL) * np.float32(math.sqrt(self.width))
        if self.std is not None:
            pooled = (pooled - self.std[0]) * self.std[1]
        return self.proj(_rms(pooled, None, self.eps))


def image_patches16(pixels):
    """Split pixels (h, w, 3) into patches of 16 x 16 (row, column, channel)
    in raster order, with their (x, y), as Gemma4ImageProcessor."""
    h, w, _ = pixels.shape
    rows, cols = h // PATCH, w // PATCH
    p = pixels[:rows * PATCH, :cols * PATCH].reshape(rows, PATCH, cols, PATCH, 3)
    p = p.transpose(0, 2, 1, 3, 4).reshape(rows * cols, PATCH * PATCH * 3)
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    pos = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.int64)
    return np.ascontiguousarray(p), pos


# ---- audio ------------------------------------------------------------------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filters(n_freq=257, n_mel=128, fmin=0.0, fmax=8000.0, rate=16000):
    """The triangular HTK mel filters of transformers mel_filter_bank (norm
    None), shape (n_freq, n_mel)."""
    mel = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mel + 2)
    ff = _mel_to_hz(mel)
    fft = np.linspace(0, rate // 2, n_freq)
    diff = np.diff(ff)
    slopes = np.expand_dims(ff, 0) - np.expand_dims(fft, 1)
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down, up))


def audio_features(samples, rate=16000, max_samples=480000):
    """The log mel of Gemma4AudioFeatureExtractor: return (mel (T, 128)
    float32 with the invalid frames zero, valid (T,) bool)."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)[:max_samples]
    n = len(x)
    padded = -(-n // 128) * 128
    wav = np.zeros(padded, dtype=np.float32)
    wav[:n] = x
    mask = np.zeros(padded, dtype=np.int32)
    mask[:n] = 1
    frame, hop, nfft = 320, 160, 512
    wav = np.concatenate([np.zeros(frame // 2, np.float32), wav])
    mask = np.concatenate([np.zeros(frame // 2, np.int32), mask])
    size = frame + 1
    count = (len(wav) - size) // hop + 1
    idx = np.arange(count)[:, None] * hop + np.arange(size)[None, :]
    frames = wav[idx][:, :-1]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, n=nfft, axis=-1))
    mel = np.log(spec @ mel_filters() + np.float64(1e-3))
    valid = mask[np.arange(count) * hop + size - 1].astype(bool)
    mel = mel.astype(np.float32) * valid[:, None]
    return mel, valid


def _conv2d_s2(x, w):
    """x (c_in, H, W), w (c_out, c_in, 3, 3): stride 2, padding 1."""
    c, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    Ho, Wo = (H + 1) // 2, (Wd + 1) // 2
    cols = np.empty((c, 3, 3, Ho, Wo), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            cols[:, i, j] = xp[:, i:i + 2 * Ho:2, j:j + 2 * Wo:2]
    out = w.reshape(w.shape[0], -1) @ cols.reshape(c * 9, Ho * Wo)
    return out.reshape(w.shape[0], Ho, Wo)


def _layer_norm_nobias(x, w, eps):
    mu = x.mean(axis=-1, keepdims=True)
    d = x - mu
    return d / np.sqrt((d * d).mean(axis=-1, keepdims=True) + eps) * w


class Gemma4Audio:
    """The gemma4a encoder (Conformer) of an mmproj GGUF."""

    CHUNK, LEFT = 12, 13            # attention_chunk_size, attention_context_left

    def __init__(self, W):
        m = W.meta
        if m.get("clip.audio.projector_type") != "gemma4a":
            raise ValueError("not a gemma4a mmproj")
        self.width = int(m["clip.audio.embedding_length"])
        self.layers = int(m["clip.audio.block_count"])
        self.heads = int(m["clip.audio.attention.head_count"])
        self.hd = self.width // self.heads
        self.eps = 1e-6        # the GGUF says 1e-5 for some files; transformers uses 1e-6
        self.conv = []
        for i in range(2):
            w = W.f32("a.conv1d.%d.weight" % i)
            cout = w.size // (9 * (1 if i == 0 else 128))
            self.conv.append((w.reshape(cout, -1, 3, 3), W.f32("a.conv1d.%d.norm.weight" % i).reshape(-1)))
        self.in_proj = _Linear(W, "a.input_projection")
        self.blk = []
        for i in range(self.layers):
            p = "a.blk.%d." % i
            self.blk.append(dict(
                ff1=(W.f32(p + "ffn_norm.weight").reshape(-1), _Linear(W, p + "ffn_up"),
                     _Linear(W, p + "ffn_down"), W.f32(p + "ffn_post_norm.weight").reshape(-1)),
                ff2=(W.f32(p + "ffn_norm_1.weight").reshape(-1), _Linear(W, p + "ffn_up_1"),
                     _Linear(W, p + "ffn_down_1"), W.f32(p + "ffn_post_norm_1.weight").reshape(-1)),
                pre_attn=W.f32(p + "attn_pre_norm.weight").reshape(-1),
                post_attn=W.f32(p + "attn_post_norm.weight").reshape(-1),
                q=_Linear(W, p + "attn_q"), k=_Linear(W, p + "attn_k"),
                v=_Linear(W, p + "attn_v"), o=_Linear(W, p + "attn_out"),
                rel=_Linear(W, p + "attn_k_rel"),
                pds=W.f32(p + "per_dim_scale.weight").reshape(-1),     # softplus applied
                conv_pre=W.f32(p + "conv_norm.weight").reshape(-1),
                conv_post=W.f32(p + "norm_conv.weight").reshape(-1),
                pw1=_Linear(W, p + "conv_pw1"), pw2=_Linear(W, p + "conv_pw2"),
                dw=W.f32(p + "conv_dw.weight").reshape(self.width, -1),   # (width, 5)
                out=W.f32(p + "ln2.weight").reshape(-1)))
        self.out_proj = _Linear(W, "a.pre_encode.out", bias=True)
        self.proj = _Linear(W, "mm.a.input_projection")
        # The relative positions: sin and cos of the distances 12 .. 0.
        half = self.width // 2
        inc = math.log(10000.0) / max(half - 1, 1)
        inv = np.exp(np.arange(half) * -inc)
        ctx = self.CHUNK + self.LEFT - 1
        d = np.arange(ctx // 2, -1, -1)[:, None] * inv[None, :]
        self.pos_emb = np.concatenate([np.sin(d), np.cos(d)], axis=-1).astype(np.float32)
        self.q_scale = np.float32(self.hd ** -0.5 / math.log(2))
        self.k_scale = np.float32(math.log(1 + math.e) / math.log(2))
        # The program on the GPU (True) or in C on the CPU (False).
        self.gpu = False
        self.q8 = False
        self.runner = _Runner(self.program)

    def program(self, T, gpu=False):
        """The program of the Conformer for T rows (the records ENC_*):
        inputs "s" (the rows of the subsample convs, T x 1024) and "valid"
        (T int32); output "y" (T, text width)."""
        W, H, hd, span = self.width, self.heads, self.hd, self.LEFT - 1
        ffi = self.blk[0]["ff1"][1].w.shape[0]
        mid, outw = self.out_proj.w.shape[0], self.proj.w.shape[0]
        kin = self.in_proj.w.shape[1]
        c = _EncCompiler(self.eps, pack=not gpu, q8=self.q8, rows=max(T, span + 1),
                         kmax=max(kin, W, ffi, 2 * W, mid))
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(s=z(T, kin), valid=np.zeros(T, np.int32), x=z(T, W), h=z(T, W),
                     f=z(T, ffi), q=z(T, W), k=z(T, W), v=z(T, W), o=z(T, W), t=z(T, W),
                     r=z(span + 1, W), c2=z(T, 2 * W), cg=z(T, W), cd=z(T, W),
                     m1=z(T, mid), y=z(T, outw), pe13=np.ascontiguousarray(self.pos_emb),
                     _scratch=np.zeros(T * max(kin, W, ffi, 2 * W, mid) + (span + 1) * W,
                                       np.float32))
        kvec = np.full(W, self.k_scale, np.float32)
        forms = [("set", "x", ("enc_linear", self.in_proj, "s"))]

        def ffn(ff):
            pre, up, down, post = ff
            return [("set", "h", ("enc_rms", "x", pre, W)),
                    ("set", "f", ("enc_linear", up, "h")),
                    ("set", "f", ("enc_silu", "f")),
                    ("set", "t", ("enc_linear", down, "f")),
                    ("set", "t", ("enc_rms", "t", post, W)),
                    ("enc_add", "x", "t", 0.5)]
        for b in self.blk:
            qvec = np.ascontiguousarray(np.tile(self.q_scale * b["pds"], H), dtype=np.float32)
            forms += ffn(b["ff1"])
            forms += [
                ("set", "h", ("enc_rms", "x", b["pre_attn"], W)),
                ("set", "q", ("enc_linear", b["q"], "h")),
                ("set", "q", ("enc_mul_vec", "q", qvec)),
                ("set", "k", ("enc_linear", b["k"], "h")),
                ("set", "k", ("enc_mul_vec", "k", kvec)),
                ("set", "v", ("enc_linear", b["v"], "h")),
                ("set", "r", ("enc_linear", b["rel"], "pe13")),
                ("set", "o", ("enc_local_attn", "q", "k", "v", "r", "valid", H, hd, span, 50.0)),
                ("set", "t", ("enc_linear", b["o"], "o")),
                ("set", "t", ("enc_rms", "t", b["post_attn"], W)),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", b["conv_pre"], W)),
                ("set", "c2", ("enc_linear", b["pw1"], "h")),
                ("set", "cg", ("enc_glu", "c2")),
                ("set", "cd", ("enc_dwconv", "cg", b["dw"])),
                ("set", "cd", ("enc_rms", "cd", b["conv_post"], W)),
                ("set", "cd", ("enc_silu", "cd")),
                ("set", "t", ("enc_linear", b["pw2"], "cd")),
                ("enc_add", "x", "t"),
            ]
            forms += ffn(b["ff2"])
            forms += [("set", "x", ("enc_rms", "x", b["out"], W))]
        forms += [("set", "m1", ("enc_linear", self.out_proj, "x")),
                  ("set", "m1", ("enc_rms", "m1", None, mid)),
                  ("set", "y", ("enc_linear", self.proj, "m1"))]
        c.compile(("seq", *forms))
        return c.p.finish()

    def _ffn(self, x, ff):
        pre, up, down, post = ff
        h = down(_silu(up(_rms(x, pre, self.eps))))
        return x + np.float32(0.5) * _rms(h, post, self.eps)

    def _attention(self, x, b, valid):
        T = x.shape[0]
        H, hd = self.heads, self.hd
        q = b["q"](x).reshape(T, H, hd) * (self.q_scale * b["pds"])
        k = b["k"](x).reshape(T, H, hd) * self.k_scale
        v = b["v"](x).reshape(T, H, hd)
        R = b["rel"](self.pos_emb).reshape(-1, H, hd)       # (13, H, hd): distance 12 - p
        span = self.LEFT - 1                                # 12 keys: t - 11 .. t
        # Key j of query t is at t - span + 1 + j, at the distance span - 1 - j,
        # so its row of R is 1 + j for every t (the relative shift of
        # transformers). A key before 0 or not valid gets -1e9 after the cap.
        idx = np.arange(T)[:, None] - (span - 1) + np.arange(span)[None, :]     # (T, span)
        ok = (idx >= 0) & valid[np.clip(idx, 0, T - 1)]
        ci = np.clip(idx, 0, T - 1)
        kk, vv = k[ci], v[ci]                               # (T, span, H, hd)
        s = np.einsum("thd,tjhd->thj", q, kk) + np.einsum("thd,jhd->thj", q, R[1:span + 1])
        s = np.tanh(s / np.float32(50.0)) * np.float32(50.0)
        s = np.where(ok[:, None, :], s, np.float32(-1e9))
        p = _softmax(s)
        out = np.einsum("thj,tjhd->thd", p, vv)
        return b["o"](out.reshape(T, H * hd))

    def _conv(self, x, b):
        h = _rms(x, b["conv_pre"], self.eps)
        h = b["pw1"](h)
        a, g = h[:, :self.width], h[:, self.width:]
        h = a * (np.float32(0.5) * (1.0 + np.tanh(np.float32(0.5) * g)))
        kw = b["dw"].shape[1]
        hp = np.concatenate([np.zeros((kw - 1, self.width), np.float32), h], axis=0)
        c = np.zeros_like(h)
        for j in range(kw):
            c += hp[j:j + h.shape[0]] * b["dw"][:, j]
        c = _silu(_rms(c, b["conv_post"], self.eps))
        return x + b["pw2"](c)

    def encode(self, mel, valid):
        """mel (T, 128), valid (T,). Return the soft rows of the valid tokens."""
        x = np.asarray(mel, np.float32)[None]               # (1, T, 128)
        m = np.asarray(valid, bool)
        for w, nw in self.conv:
            x = x * m[None, :, None]
            x = _conv2d_s2(x, w)                            # (c, T', F')
            x = _layer_norm_nobias(x.transpose(1, 2, 0), nw, self.eps).transpose(2, 0, 1)
            x = np.maximum(x, 0.0)
            m = m[::2]
        c, T, F = x.shape
        x = x.transpose(1, 2, 0).reshape(T, F * c)
        if not _ENC_PY:
            y = self.runner.run(T, self.gpu, {"s": x, "valid": m.astype(np.int32)}, "y")
            return y[m]
        x = self.in_proj(x)
        for b in self.blk:
            x = self._ffn(x, b["ff1"])
            x = x + _rms(self._attention(_rms(x, b["pre_attn"], self.eps), b, m),
                         b["post_attn"], self.eps)
            x = self._conv(x, b)
            x = self._ffn(x, b["ff2"])
            x = _rms(x, b["out"], self.eps)
        x = self.out_proj(x)[m]
        return self.proj(_rms(x, None, self.eps))


class Gemma4Embedder:
    """The media embedder of the E2B, E4B, and 26B: images (and video
    frames) with gemma4v, audio with gemma4a when the mmproj has it. The
    interface of unified.UnifiedEmbedder."""

    def __init__(self, path, gpu=False, q8=False):
        W = _Weights(path)
        self.path = path
        self.vision = Gemma4Vision(W)
        self.has_audio = bool(W.meta.get("clip.has_audio_encoder"))
        self.audio_enc = Gemma4Audio(W) if self.has_audio else None
        self.gpu = gpu
        self.vision.gpu = gpu
        if self.audio_enc is not None:
            self.audio_enc.gpu = gpu
        # q8: the linears with Q8_0 weights (half the bytes of bfloat16) and
        # int8 activations, in the programs of the CPU and of the GPU.
        self.q8 = q8
        if q8:
            V = self.vision
            lins = [V.patch_lin] + [b[k] for b in V.blk
                                    for k in ("q", "k", "v", "o", "gate", "up", "down")]
            V.q8 = True
            if self.audio_enc is not None:
                A = self.audio_enc
                lins += [A.in_proj, A.out_proj]
                for b in A.blk:
                    lins += [b[k] for k in ("q", "k", "v", "o", "rel", "pw1", "pw2")]
                    lins += [b["ff1"][1], b["ff1"][2], b["ff2"][1], b["ff2"][2]]
                A.q8 = True
            for lin in lins:
                quantize_q8(lin)
        self.hidden = self.vision.proj.w.shape[0]
        # The tokens of an image see each other in the larger models
        # (use_bidirectional_attention "vision"); the E2B (1536) and the E4B
        # (2560) are causal (their config, and mtmd_decode_use_non_causal of
        # llama.cpp).
        self.bidir = self.hidden not in (1536, 2560)

    def image(self, src, budget=280):
        from .unified import image_pixels
        pixels = image_pixels(src, budget)
        patches, pos = image_patches16(pixels)
        rows = self.vision.encode(patches, pos)
        return rows, (pixels.shape[0] // 48, pixels.shape[1] // 48)

    def audio(self, src):
        from .unified import load_audio, AUDIO_RATE, AUDIO_MAX_S
        if self.audio_enc is None:
            raise ValueError("%s has no audio encoder" % self.path)
        x = src if isinstance(src, np.ndarray) else load_audio(src)
        if len(x) > AUDIO_MAX_S * AUDIO_RATE:
            raise ValueError("the clip has %.1f s; the model takes at most %d s"
                             % (len(x) / AUDIO_RATE, AUDIO_MAX_S))
        mel, valid = audio_features(x)
        return self.audio_enc.encode(mel, valid)

    def video(self, src, budget=70, num_frames=32):
        from PIL import Image
        from .unified import load_video
        frames, times = load_video(src, num_frames)
        return [(t, self.image(Image.fromarray(fr), budget)[0]) for fr, t in zip(frames, times)]


def load_embedder(path, gpu=False, q8=False):
    """Return the embedder of an mmproj GGUF: UnifiedEmbedder for the 12B
    (gemma4uv), Gemma4Embedder for gemma4v (and gemma4a), QwenEmbedder for
    qwen3vl_merger (or a Qwen3.8 checkpoint directory). gpu runs the encoder
    programs on the GPU."""
    if os.path.isdir(path) or GGUF(path, stage=False).meta.get("clip.projector_type") == "qwen3vl_merger":
        # Qwen3.6 (an mmproj GGUF) or Qwen3.8 (the checkpoint directory)
        from .vision_qwen import QwenEmbedder
        return QwenEmbedder(path, gpu=gpu, q8=q8)
    g = GGUF(path)
    kind = g.meta.get("clip.vision.projector_type")
    if kind == "gemma4uv":
        from .unified import UnifiedEmbedder
        return UnifiedEmbedder(path)
    if kind == "gemma4v":
        return Gemma4Embedder(path, gpu=gpu, q8=q8)
    raise ValueError("%s: the projector %r is not supported" % (path, kind))
