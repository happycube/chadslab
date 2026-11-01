"""Qwen-Image-2.1's DiT (diffusers' QwenImage21Transformer2DModel: 32 single
stream blocks, width 4096, 32 heads of 128, causal_condition) as two programs:
the text's keys and values once an image, and a step of the image's rows.

Ported from diffusers
(src/diffusers/models/transformers/transformer_qwenimage21.py; Apache License
2.0; Copyright 2026 Qwen-Image Team, The HuggingFace Team). See
THIRD_PARTY_NOTICES.md.

    the step's vectors, on the host in float32: the timestep's sinusoid (256,
        [cos, sin]), the MLP (linear, SiLU, linear: temb), and from SiLU(temb)
        the modulation shared by all blocks (no shift: the scale and the gate
        of the attention and of the MLP) and the scale of the last norm
    the rows: the text's (txt_in: the zero-centered RMS norm of the text
        encoder's rows, times 1 + w; linear, tanh GELU, linear) then the
        image's (img_in of the latents)
    a block: LN (no weights) times 1 + scale (ENC_LNORM with the step's vector
        as its weight), q k v, the RMS norm of each head (q's weight times
        hd^-1/2), RoPE, the attention, the output, times tanh(gate), + x; LN
        times 1 + scale, SiLU(gate_layer) proj, out, times tanh(gate), + x
    the image rows: LN times 1 + scale, proj_out

The attention is block causal: a text row sees the text rows up to it, an
image row sees all the text and all the image. With causal_condition the text
rows take the modulation of t = 0, so they do not depend on the step: the
prefix program runs them once an image (txt_in, then each block up to its
keys and values, kept in the cache kc, vc: (blocks, rows, width), after the
norms and RoPE), as the pipeline's use_kv_cache. A step runs the image's rows
alone, their keys and values in rows [0, I) of kk, vv and the text's copied
from the cache into the rows after them: the attention of each image row over
the rows [0, I + T) (the order of the keys does not change the sums' terms).
The queries of q, kk and vv have the rows of the keys (ENC_ATTN_X2 takes q, k,
v of the same rows): the rows past I are not used. The text rows are padded
to a multiple of PAD (the programs are made for that many rows; padding at the
end, which the causal text rows do not see, and the image rows' range ends
at I + T).

RoPE: the pairs of a head (interleaved) at the angles of three axes of 16, 56
and 56 values (theta 10000), from tables of positions (QwenImage21Rope): a
text row l at (l, l, l); an image of h x w tokens at frame T (the text's
length) and the centered (y - (h - h // 2), x - (w - w // 2)). The angles in
float32 as torch's tables.

Editing (the pipeline's joint sequence: prefix's media): the prefix holds the
text rows and, at each image's slots of the text, the condition image's
latent tokens (img_in of them in place of txt_in's rows: an <|image_pad|>
slot stands for 2 x 2 tokens), each image a block that sees the rows before
it and its own (bidirectional), modulated at t = 0 as the text. The
positions: a text row at its frame, the frame then + 1; an image block at the
frame, centered, the frame then + max(h, w); the target image at the frame
after the prefix. The prefix program takes its positions' tables and its rows'
ranges of keys as inputs; a step's are as before, the target at its frame.
An edit's cache is F20 by default (f20_media; the pipeline's cache_f20,
RECLANK_QWEN_CACHE_F20=0: float32): the prefix computes a block's keys and
values in float32 (its own attention reads them), then ENC_F20_PACK into the
cache (blocks, rows, F20 row bytes: 0.63 MB a row, float32's 1 MB); a step
ENC_F20_UNPACK of a block's into its text keys and values (the values a
cache of the format's: 16 significant bits in groups of 32). A text's cache
(text to image) stays float32.

On the GPU an edit's F20 cache is in host memory (host_cache: registered
with CUDA; RECLANK_QWEN_CACHE_HOST=0, or no tensor cores: on the GPU as on
the CPU): the prefix packs a block's keys and values into two staging rows
on the GPU and GP_TO_HOST copies them to the host; a step fetches block j's
into slot j % 2 of two slots on the GPU (GP_FETCH, as Wan A14B's streamed
blocks: block j + 2 after FETCH_DONE of block j; the prefix waits for its
copies before it returns) and ENC_ATTN_X2C attends the
image's queries alone over the image's keys and the slot's F20 keys,
decoded into the attention's bfloat16 planes (no float32 rows of the text's
keys and values, no query rows of the text): the same bits as the cache on
the GPU. The GPU then holds two blocks' slots, not the cache. close() (or the
DiT's finalizer) waits for the GPU and unregisters the host caches before
their memory goes.

The matrices: the checkpoint's bfloat16 rows (on the GPU KQ_BF16; on the CPU
KQ_BF16X16, the same values in float32 products).
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import weakref

import numpy as np

import mmap

from reclank.backends.cpu import cops
from reclank.encoders.runtime import _EncCompiler, _Runner
from reclank.ir import program as P
from reclank.formats.safetensors import Shards
from reclank.models.qwen3.dense import lin16
from reclank.runtime.threads import team_threads

PAD = 64            # the text rows of a program: a multiple of PAD
MLP_ROWS = 2048     # the rows of the MLP's products at a time (its buffers)
_REGISTERED = set()     # (address, bytes) of the host memory registered with CUDA here (host_cache)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def timestep_sinusoid(t, dim=256, max_period=10000):
    """QwenImage21TemporalTimesteps of one timestep (0 to 1000: the pipeline's
    t / 1000 times its time_factor), [cos, sin], float32."""
    half = dim // 2
    ex = np.exp((-math.log(max_period) * np.arange(half, dtype=np.float32) / np.float32(half)).astype(np.float32))
    e = (np.float32(t) * ex).astype(np.float32)
    return np.concatenate([np.cos(e), np.sin(e)]).astype(np.float32)


def rope_table(pos, axes, theta):
    """cos and sin (rows, sum(axes) / 2) of the positions pos (rows,
    len(axes)): each axis its own frequencies (float32, as QwenImage21Rope's
    rope_params: the angle a float32 product), the cos and sin of the float32
    angle."""
    cs, sn = [], []
    for i, d in enumerate(axes):
        freqs = (1.0 / np.power(np.float32(theta), (np.arange(0, d, 2).astype(np.float32) / np.float32(d)))).astype(np.float32)
        ang = (pos[:, i].astype(np.float32)[:, None] * freqs[None, :]).astype(np.float32)
        cs.append(np.cos(ang.astype(np.float64)))
        sn.append(np.sin(ang.astype(np.float64)))
    return np.concatenate(cs, 1).astype(np.float32), np.concatenate(sn, 1).astype(np.float32)


def _unregister(hcache, which=None):
    """Wait for the GPU, then unregister host caches (hcache's of which, or all) from CUDA and drop
    them (their memory goes when nothing else holds it)."""
    if not hcache:
        return
    from reclank.backends.cuda.runtime import lib
    lib().rc_gpu_sync()                 # (the copies queued before are done with them)
    for w in [which] if which is not None else list(hcache):
        flat = hcache.pop(w)[1]
        a = flat.ctypes.data
        if (a, flat.nbytes) in _REGISTERED:
            lib().rc_gpu_host_unregister(ctypes.c_void_p(a))
            _REGISTERED.discard((a, flat.nbytes))


def padded(T):
    """The text rows of the programs of T tokens."""
    return max(PAD, -(-T // PAD) * PAD)


class QwenImageDiT:
    """The DiT of a transformer directory (config.json and its safetensors).
    prefix(text) runs the text rows into a cache; velocity(latents, t,
    cache) runs a step."""

    def __init__(self, path, gpu=False, cpu_threads=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg["patch_size"] == 1 and cfg["causal_condition"]
        self.W = W = Shards(path, "diffusion_pytorch_model.safetensors.index.json")
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.heads, self.hd = cfg["num_attention_heads"], cfg["attention_head_dim"]
        self.width = w = self.heads * self.hd
        self.inter = w * cfg["mlp_ratio"]
        self.eps = float(cfg["eps"])
        self.axes = list(cfg["axes_dims_rope"])
        self.in_ch, self.out_ch = cfg["in_channels"], cfg["out_channels"]
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        sq = np.float32(self.hd ** -0.5)
        L = lambda n: lin16(W, n, gpu, bias=False, cpu_bf16=True)  # noqa: E731
        # the host's matrices of the step's vectors
        e = "time_text_embed.timestep_embedder."
        self.t1, self.t2 = f(e + "linear_1.weight"), f(e + "linear_2.weight")
        self.mod = f("modulation.1.weight")
        self.mod_out = f("norm_out.linear.weight")
        self.tnorm = (f("txt_in.text_norm.weight") + 1).astype(np.float32)
        self.t_in, self.t_out = L("txt_in.in_layer"), L("txt_in.out_layer")
        self.x_emb, self.proj = L("img_in"), L("proj_out")
        self.blocks = []
        for i in range(cfg["num_layers"]):
            p = "transformer_blocks.%d." % i
            a = p + "attn."
            self.blocks.append(dict(
                q=L(a + "to_q"), k=L(a + "to_k"), v=L(a + "to_v"), o=L(a + "to_out.0"),
                nq=(f(a + "norm_q.weight") * sq).astype(np.float32), nk=f(a + "norm_k.weight"),
                g=L(p + "img_mlp.gate_layer"), u=L(p + "img_mlp.proj"), d=L(p + "img_mlp.out")))
        self.runner = _Runner(self._build, keep=6, keep_gpu=4)     # a text and a step of each cache
        self.caches = {}        # (rows, which, f20): (kc, vc), the arrays the programs share
        self.f20_media = True   # an edit's cache in F20 (prefix's media)
        self.f20 = {}           # (rows, which): the form of the cache of the last prefix (F20 or not)
        self.acts = {}          # (rows, hh, ww, form): the buffers of the steps of the prompt and of the negative
        self.hcache = {}        # which: (buffer, its bytes as uint8): host_cache
        self.slots = {}         # rows: the two slots' host placeholders (their device copies are the slots)
        # (the host caches unregistered before their memory goes: a range left registered after its
        # unmap fails the copies of the memory mapped there next)
        weakref.finalize(self, _unregister, self.hcache)

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def cache(self, rows, which=0, f20=False):
        """The keys and values (blocks, rows, width) of a text of rows (a
        multiple of PAD): which 0 the prompt's, 1 the negative prompt's (CFG);
        f20: F20 rows (blocks, rows, F20 row bytes), uint8. The prefix program
        writes them, the step programs read them (on the GPU the same device
        copies: one mirror)."""
        key = (rows, which, bool(f20))
        if key not in self.caches:
            shape = (len(self.blocks), rows) + ((cops.f20_row_bytes(self.width),) if f20 else (self.width,))
            dt = np.uint8 if f20 else np.float32
            self.caches[key] = (np.zeros(shape, dt), np.zeros(shape, dt))
        return self.caches[key]

    def cache_on_host(self):
        """Whether an edit's F20 cache is in host memory (host_cache): on the GPU with its tensor
        cores (ENC_ATTN_X2C), unless RECLANK_QWEN_CACHE_HOST=0 or RECLANK_GPU_ENC_X2=0."""
        if not self.gpu or os.environ.get("RECLANK_QWEN_CACHE_HOST", "1") == "0" or \
                os.environ.get("RECLANK_GPU_ENC_X2", "1") == "0":
            return False
        from reclank.backends.cuda.runtime import has_tc
        return has_tc()

    def host_cache(self, rows, which=0):
        """The F20 keys and values (2, blocks, rows, F20 row bytes) of an edit's prefix of rows in
        host memory registered with CUDA (the prefix's GP_TO_HOST writes them, the steps' GP_FETCH
        reads them): the start of one buffer of each which, which only grows (registered once a
        size: 3 s for 5.5 GB). A larger one replaces it after the GPU is done with it, and the
        prefix programs that wrote to it go (their GP_TO_HOST holds its address)."""
        from reclank.backends.cuda.runtime import lib
        n = 2 * len(self.blocks) * rows * cops.f20_row_bytes(self.width)
        old = self.hcache.get(which)
        if old is None or old[1].nbytes < n:
            if old is not None:
                lib().rc_gpu_sync()
                for d in (self.runner.progs, self.runner.hosts):
                    for k in [k for k in d if k[0][0] == "text" and k[0][2] == which and k[0][4:] == ("host",)]:
                        if d is self.runner.progs:
                            self.runner._drop(k)
                        else:
                            del d[k]
                _unregister(self.hcache, which)
            buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            flat = np.frombuffer(buf, np.uint8)
            if lib().rc_gpu_host_register_rw(ctypes.c_void_p(flat.ctypes.data), n) == 0:
                raise RuntimeError("cudaHostRegister of %.2f GB: %s" % (n / 1e9, lib().rc_gpu_last_error().decode()))
            _REGISTERED.add((flat.ctypes.data, n))
            old = self.hcache[which] = (buf, flat)
        return old[1][:n].reshape(2, len(self.blocks), rows, -1)

    def close(self):
        """Free the GPU memory and the programs (on the host too: a prefix program's GP_TO_HOST holds
        a host cache's address), wait for the GPU, then unregister the host caches from CUDA and free
        them. The next run builds them again."""
        self.runner.release(keep_host=False)
        self.runner.progs.clear()
        self.runner.hosts.clear()
        self.slots.clear()
        self.caches.clear()
        self.acts.clear()
        _unregister(self.hcache)

    def slot_pair(self, rows):
        """The two slots (2, rows, F20 row bytes) of a block's keys and values, host placeholders
        of their device copies (never written on the host)."""
        if rows not in self.slots:
            while len(self.slots) >= 2:
                # (a program of an older size keeps its own)
                self.slots.pop(next(iter(self.slots)))
            rb = cops.f20_row_bytes(self.width)
            self.slots[rows] = [np.frombuffer(mmap.mmap(-1, 2 * rows * rb, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS),
                                              np.uint8).reshape(2, rows, rb) for _ in range(2)]
        return self.slots[rows]

    # ---- the step's vectors (host) ----------------------------------------

    def vectors(self, t):
        """The modulation vectors (name: (width,)) of timestep t (0 to
        1000): 1 + scale and tanh(gate) of the attention (s1, g1) and of the
        MLP (s2, g2), 1 + scale of the last norm (so)."""
        w = self.width
        temb = (self.t2 @ _silu(self.t1 @ timestep_sinusoid(t))).astype(np.float32)
        s = _silu(temb).astype(np.float32)
        sc1, g1, sc2, g2 = (self.mod @ s).astype(np.float32).reshape(4, w)
        so = (self.mod_out @ s).astype(np.float32)
        out = dict(s1=1 + sc1, g1=np.tanh(g1), s2=1 + sc2, g2=np.tanh(g2), so=1 + so)
        return {k: v.astype(np.float32) for k, v in out.items()}

    def rope_text(self, rows):
        """cos and sin of text rows 0 .. rows - 1 (each axis at l)."""
        return rope_table(np.repeat(np.arange(rows)[:, None], 3, 1), self.axes, 10000)

    def rope_image(self, T, hh, ww):
        """cos and sin of an image of hh x ww tokens after T text tokens."""
        pos = np.zeros((hh * ww, 3), np.int64)
        pos[:, 0] = T
        pos[:, 1] = np.repeat(np.arange(hh) - (hh - hh // 2), ww)
        pos[:, 2] = np.tile(np.arange(ww) - (ww - ww // 2), hh)
        return rope_table(pos, self.axes, 10000)

    # ---- the programs ------------------------------------------------------

    def _block_forms(self, B, x, h, tmp, q, k, v, att, lo, hi, rows, vec, last=False):
        """The forms of a block on the rows x (names): q k v into q, k, v
        (the rows of x: views), the attention of att's rows (q, k, v, att:
        the names of the whole buffers) over [lo, hi); last: up to the keys
        and values (the prefix's last block)."""
        w, H, hd = self.width, self.heads, self.hd
        forms = [("set", h, ("enc_lnorm", x, vec["s1"], None, w)),
                 ("set", tmp, ("enc_linear", B["q"], h)),
                 ("set", q, ("enc_rms", tmp, B["nq"], hd)),
                 ("set", tmp, ("enc_linear", B["k"], h)),
                 ("set", k, ("enc_rms", tmp, B["nk"], hd)),
                 ("set", v, ("enc_linear", B["v"], h)),
                 ("enc_rope_tab", q, "cos", "sin", H, hd),
                 ("enc_rope_tab", k, "cos", "sin", H, hd)]
        if last:
            return forms
        forms += [("set", att[3], ("enc_attn_range_x2", att[0], att[1], att[2], lo, hi, H, hd)),
                  ("set", tmp, ("enc_linear", B["o"], att[4])),
                  ("set", tmp, ("enc_mul_vec", tmp, vec["g1"])),
                  ("enc_add", x, tmp),
                  ("set", h, ("enc_lnorm", x, vec["s2"], None, w))]
        for r0 in range(0, rows, MLP_ROWS):
            r1 = min(rows, r0 + MLP_ROWS)
            n = "_%d_%d" % (r0, r1)
            forms += [("set", "g" + n, ("enc_linear", B["g"], h + n)),
                      ("set", "u" + n, ("enc_linear", B["u"], h + n)),
                      ("set", "u" + n, ("enc_silu_mul", "g" + n, "u" + n)),
                      ("set", tmp + n, ("enc_linear", B["d"], "u" + n))]
        return forms + [("set", tmp, ("enc_mul_vec", tmp, vec["g2"])),
                        ("enc_add", x, tmp)]

    def _mlp_views(self, env, h, tmp, rows):
        """The views of the MLP's chunks of rows: h and tmp's rows, the g and
        u buffers (MLP_ROWS rows)."""
        for r0 in range(0, rows, MLP_ROWS):
            r1 = min(rows, r0 + MLP_ROWS)
            n = "_%d_%d" % (r0, r1)
            env[h + n], env[tmp + n] = env[h][r0:r1], env[tmp][r0:r1]
            env["g" + n], env["u" + n] = env["g"][:r1 - r0], env["u"][:r1 - r0]

    def program(self, kind, *key, gpu=False):
        """kind "text": the prefix program of rows text rows (key (rows,
        which)): input "txt" (rows, joint dim), the cache of (rows, which);
        key (rows, which, blocks): an edit's, the condition images at the
        rows (start, end) of blocks, inputs "cimg" (their latents), "cos",
        "sin", "ahi".
        kind "step": a step of an image of hh x ww tokens after a text of
        rows (key (rows, which, hh, ww)): inputs "img" (hh ww, in channels),
        "cos", "sin" (the image's RoPE), "ahi" (the end of the keys: I + T),
        the vectors of vectors(); output "vel" (hh ww, out channels)."""
        w, inter = self.width, self.inter
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        if kind == "text":
            rows, which = key[:2]
            blocks, f20 = key[2:] if len(key) > 2 else ((), False)
            host = f20 == "host"
            kc, vc = self.host_cache(rows, which) if host else self.cache(rows, which, f20)
            c = _EncCompiler(self.eps, pack=not gpu, rows=rows, kmax=inter)
            env = c.env
            cos, sin = self.rope_text(rows)
            env.update(txt=z(rows, self.t_in.shape[1]), x=z(rows, w), h=z(rows, w), tmp=z(rows, w), q=z(rows, w),
                       g=z(min(rows, MLP_ROWS), inter), u=z(min(rows, MLP_ROWS), inter), cos=cos, sin=sin,
                       alo=np.zeros(rows, np.int32), ahi=(np.arange(rows) + 1).astype(np.int32))
            self._mlp_views(env, "h", "tmp", rows)
            vec = self.vectors(0.0)
            forms = [("set", "h", ("enc_rms", "txt", self.tnorm, self.t_in.shape[1])),
                     ("set", "tmp", ("enc_linear", self.t_in, "h")),
                     ("set", "tmp", ("enc_gelu", "tmp")),
                     ("set", "x", ("enc_linear", self.t_out, "tmp"))]
            if blocks:
                # the condition images' rows: img_in of their latents (input "cimg") over txt_in's
                env["cimg"] = z(sum(e - s_ for s_, e in blocks), self.in_ch)
                r0 = 0
                for b, (s_, e) in enumerate(blocks):
                    env["x_%d" % b], env["ci_%d" % b] = env["x"][s_:e], env["cimg"][r0:r0 + e - s_]
                    forms.append(("set", "x_%d" % b, ("enc_linear", self.x_emb, "ci_%d" % b)))
                    r0 += e - s_
            if f20:
                # a block's keys and values in float32 (its attention's), then F20 into the cache
                env.update(kf=z(rows, w), vf=z(rows, w))
            if host:
                # (the cache in host memory: F20 into two staging rows, then GP_TO_HOST)
                rb = cops.f20_row_bytes(w)
                env.update(kst=np.zeros((rows, rb), np.uint8), vst=np.zeros((rows, rb), np.uint8))
            for i, B in enumerate(self.blocks):
                ki, vi = ("kf", "vf") if f20 else ("k%d" % i, "v%d" % i)
                forms += self._block_forms(B, "x", "h", "tmp", "q", ki, vi, ("q", ki, vi, "h", "h"),
                                           "alo", "ahi", rows, vec, last=i == len(self.blocks) - 1)
                if host:
                    forms += [("enc_f20_pack", "kf", "kst"), ("enc_f20_pack", "vf", "vst")]
                    c.compile(("seq", *forms))
                    forms = []
                    c.p.emit(P.TO_HOST, env["kst"], kc[i].ctypes.data, kc[i].nbytes, env["vst"], vc[i].ctypes.data,
                             vc[i].nbytes, 0, 0, 0, 0)
                    continue
                env["k%d" % i], env["v%d" % i] = kc[i], vc[i]
                if f20:
                    forms += [("enc_f20_pack", "kf", "k%d" % i), ("enc_f20_pack", "vf", "v%d" % i)]
        elif key[4:] == ("f20h",):
            return self._host_step(*key[:4], gpu=gpu)
        else:
            rows, which, hh, ww = key[:4]
            f20 = len(key) > 4
            kc, vc = self.cache(rows, which, f20)
            I = hh * ww
            R = I + rows
            c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=inter)
            env = c.env
            # the same buffers for the steps of both caches (CFG: on the GPU one device copy)
            acts = self.acts.get((rows, hh, ww, "dev"))
            if acts is None:
                while len(self.acts) >= 2:
                    # (a program kept of an older size keeps its own)
                    self.acts.pop(next(iter(self.acts)))
                acts = self.acts[(rows, hh, ww, "dev")] = dict(
                    img=z(I, self.in_ch), vel=z(I, self.out_ch), x=z(I, w), tmp=z(I, w),
                    h=z(R, w), q=z(R, w), kk=z(R, w), vv=z(R, w),
                    g=z(min(I, MLP_ROWS), inter), u=z(min(I, MLP_ROWS), inter),
                    cos=z(I, self.hd // 2), sin=z(I, self.hd // 2),
                    alo=np.zeros(R, np.int32), ahi=np.full(R, R, np.int32),
                    **{k_: z(w) for k_ in ("s1", "g1", "s2", "g2", "so")})
            env.update(acts)
            env["hi"], env["qi"], env["ki"], env["vi"] = env["h"][:I], env["q"][:I], env["kk"][:I], env["vv"][:I]
            env["kt"], env["vt"] = env["kk"][I:], env["vv"][I:]
            self._mlp_views(env, "hi", "tmp", I)
            vec = {k_: k_ for k_ in ("s1", "g1", "s2", "g2")}
            forms = [("set", "x", ("enc_linear", self.x_emb, "img"))]
            for i, B in enumerate(self.blocks):
                env["kc%d" % i], env["vc%d" % i] = kc[i], vc[i]
                f = self._block_forms(B, "x", "hi", "tmp", "qi", "ki", "vi", ("q", "kk", "vv", "h", "hi"),
                                      "alo", "ahi", I, vec)
                # the text's keys and values after the image's, before the attention
                at = [j for j, x in enumerate(f) if x[0] == "set" and x[2][0] == "enc_attn_range_x2"][0]
                if f20:
                    f[at:at] = [("set", "kt", ("enc_f20_unpack", "kc%d" % i, w)),
                                ("set", "vt", ("enc_f20_unpack", "vc%d" % i, w))]
                else:
                    f[at:at] = [("set", "kt", ("copy", "kc%d" % i)), ("set", "vt", ("copy", "vc%d" % i))]
                forms += f
            forms += [("set", "hi", ("enc_lnorm", "x", "so", None, w)),
                      ("set", "vel", ("enc_linear", self.proj, "hi"))]
        c.compile(("seq", *forms))
        return c.p.finish()

    def _host_step(self, rows, which, hh, ww, gpu=True):
        """The step program of an edit's cache in host memory (key (rows, which, hh, ww, "f20h")):
        the image's rows alone (I), each block's attention ENC_ATTN_X2C over the image's keys and
        slot j % 2's (GP_FETCH j of the host cache's block j: prog.fetch, its ranges filled at each
        run, _fill). The inputs and output of the other step programs; "ahi" (I,)."""
        assert gpu
        w, inter, H, hd = self.width, self.inter, self.heads, self.hd
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        I = hh * ww
        n = len(self.blocks)
        c = _EncCompiler(self.eps, pack=False, rows=I, kmax=inter)
        env = c.env
        acts = self.acts.get((rows, hh, ww, "host"))
        if acts is None:
            while len(self.acts) >= 2:
                self.acts.pop(next(iter(self.acts)))
            acts = self.acts[(rows, hh, ww, "host")] = dict(
                img=z(I, self.in_ch), vel=z(I, self.out_ch), x=z(I, w), tmp=z(I, w),
                h=z(I, w), q=z(I, w), kk=z(I, w), vv=z(I, w),
                g=z(min(I, MLP_ROWS), inter), u=z(min(I, MLP_ROWS), inter),
                cos=z(I, self.hd // 2), sin=z(I, self.hd // 2),
                alo=np.zeros(I, np.int32), ahi=np.full(I, I, np.int32),
                **{k_: z(w) for k_ in ("s1", "g1", "s2", "g2", "so")})
        env.update(acts)
        slots = self.slot_pair(rows)
        for s_ in range(2):
            env["ks%d" % s_], env["vs%d" % s_] = slots[s_][0], slots[s_][1]
        self._mlp_views(env, "h", "tmp", I)
        vec = {k_: k_ for k_ in ("s1", "g1", "s2", "g2")}
        c.compile(("seq", ("set", "x", ("enc_linear", self.x_emb, "img"))))
        # the ranges of each block's fetch: (host, device, bytes) of its keys and of its values
        rng = [np.zeros((2, 3), np.int64) for _ in range(n)]
        for j in range(min(2, n)):
            c.p.emit(P.FETCH, rng[j], 2, j, j % 2)
        for j, B in enumerate(self.blocks):
            s_ = j % 2
            f = self._block_forms(B, "x", "h", "tmp", "q", "kk", "vv", ("q", "kk", "vv", "h", "h"), "alo", "ahi", I,
                                  vec)
            at = [i_ for i_, x in enumerate(f) if x[0] == "set" and x[2][0] == "enc_attn_range_x2"][0]
            c.compile(("seq", *f[:at]))
            c.p.emit(P.FETCH_WAIT, j)
            c.compile(("seq", ("set", "h", ("enc_attn_x2c", "q", "kk", "vv", "ks%d" % s_, "vs%d" % s_, "alo", "ahi",
                                            rows, H, hd))))
            c.p.emit(P.FETCH_DONE, s_)
            if j + 2 < n:
                c.p.emit(P.FETCH, rng[j + 2], 2, j + 2, s_)
            c.compile(("seq", *f[at + 1:]))
        c.compile(("seq", ("set", "h", ("enc_lnorm", "x", "so", None, w)),
                   ("set", "vel", ("enc_linear", self.proj, "h"))))
        prog = c.p.finish()
        prog.fetch = rng
        return prog

    def _fill(self, prog, rows, which):
        """The ranges of a host step program's fetches: the host cache's blocks of now, the slots'
        device copies (in the runner's mirror; new after a release)."""
        kv = self.host_cache(rows, which)
        slots = self.slot_pair(rows)
        mir = self.runner.mirror
        dev = [(mir.translate(sl[0].ctypes.data), mir.translate(sl[1].ctypes.data)) for sl in slots]
        for j, r in enumerate(prog.fetch):
            for a in range(2):
                r[a] = (kv[a, j].ctypes.data, dev[j % 2][a], kv[a, j].nbytes)

    def prefix(self, txt, which=0, media=None, f20=None):
        """Run the prefix program of the text rows txt (T, joint dim): its
        keys and values into the cache of which. Return its rows. media (an
        edit's joint sequence, txt's rows of the images unused): a dict of
        "blocks" the (start, end) rows of each condition image, "lat" their
        packed latents (rows, in channels), "pos" (T, 3) the rows'
        positions, "ahi" (T,) the end of each row's keys. f20: the cache in
        F20 (None: f20_media for an edit, float32 for a text)."""
        T = txt.shape[0]
        rows = padded(T)
        f20 = bool(media is not None and self.f20_media) if f20 is None else bool(f20)
        assert media is not None or not f20, "an F20 cache is an edit's"
        if f20 and self.cache_on_host():
            f20 = "host"
        self.f20[(rows, which)] = f20
        key = ("text", rows, which)
        inputs = {}
        if media is not None:
            key += (tuple((int(s_), int(e)) for s_, e in media["blocks"]), f20)
            pos = np.zeros((rows, 3), np.int64)
            pos[:T] = media["pos"]
            pos[T:] = pos[T - 1] + 1 + np.arange(rows - T)[:, None]
            inputs["cos"], inputs["sin"] = rope_table(pos, self.axes, 10000)
            ahi = (np.arange(rows) + 1).astype(np.int32)
            ahi[:T] = media["ahi"]
            inputs["ahi"], inputs["cimg"] = ahi, media["lat"]
        prog, g = self.runner.get(key, self.gpu)
        x = np.zeros((rows, txt.shape[1]), np.float32)
        x[:T] = txt
        inputs["txt"] = x
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
        else:
            prog.run()
        if f20 == "host":
            # the copies to the host done before a step's fetches read them (another stream)
            from reclank.backends.cuda.runtime import lib
            lib().rc_gpu_sync()
        return rows

    def velocity(self, img, t, T, hh, ww, which=0, frame=None):
        """The DiT's output (hh ww, out channels) of the packed latents img at
        timestep t (0 to 1000), after the prefix of a text of T tokens (its
        cache of which); frame: the image's (default T: a text's)."""
        rows = padded(T)
        form = self.f20.get((rows, which))
        key = ("step", rows, which, hh, ww) + (("f20h",) if form == "host" else ("f20",) if form else ())
        texts = [k for k in self.runner.progs if k[1] and k[0][0] == "text" and k[0][1:3] == (rows, which)]
        if self.gpu and texts and ((key, True) not in self.runner.progs or form == "host"):
            # the prefix program has run: its buffers leave the GPU before the step's take their
            # room (an edit's prefix of 8000 rows holds 0.9 GB), all but the weights and the cache
            # on the GPU, which the step reads (a new step program's host program first); a text's
            # host program stays for the next prefix, and one of the host cache (its address
            # stays); another edit's, of the layout of its prompt, is made again
            host = None
            if (key, True) not in self.runner.progs:
                host = self.runner.hosts.pop((key, True), None) or self._build(key, True)
                self.runner.hosts[(key, True)] = host
            for k in texts:
                if len(k[0]) == 3 or k[0][4:] == ("host",):
                    self.runner.hosts[k] = self.runner.progs[k][0]
                self.runner._drop(k, keep=host)
        prog, g = self.runner.get(key, self.gpu)
        cos, sin = self.rope_image(T if frame is None else frame, hh, ww)
        I = hh * ww
        inputs = dict(img=img, cos=cos, sin=sin, ahi=np.full(I if form == "host" else I + rows, I + T, np.int32))
        if form == "host":
            self._fill(prog, rows, which)
        inputs.update({k: v for k, v in self.vectors(t).items()})
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
            g.download("vel")
        else:
            prog.run()
        return prog.names["vel"].copy()
