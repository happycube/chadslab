"""Z-Image's DiT (diffusers' ZImageTransformer2DModel; Turbo: S3-DiT, width 3840, 30 heads of
128, the SwiGLU FFN 10240, 2 noise refiner and 2 context refiner blocks, 30 main blocks;
patches of 2 x 2 of 16 latent channels) as two programs: the caption's rows once a prompt, and
a step over the image's and the caption's rows in one stream.

Ported from diffusers (src/diffusers/models/transformers/transformer_z_image.py; Apache License 2.0;
Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

    the step's vectors, on the host in float32: the timestep's sinusoid (256, [cos, sin]) of
        t times t_scale, the t_embedder MLP (linear, SiLU, linear: adaln_input, 256), each
        modulated block's adaLN_modulation of it (no SiLU: 1 + scale and tanh(gate) of the
        attention and of the FFN), folded into the weights of the block's RMS norms (norm1's
        times 1 + scale, norm2's times tanh(gate)); the final layer's 1 + scale (adaLN of
        SiLU(adaln_input))
    the caption (prefix program, the prompt's T rows padded to Tp, a multiple of 32): the
        RMS norm and linear of cap_embedder, cap_pad_token in the padding rows, the two context
        refiner blocks (no modulation): the rows "cap" (Tp, 3840), which do not depend on the
        step
    a step (the image's I tokens padded to Ip, a multiple of 32, then the caption's Tp rows:
        diffusers' unified [x, cap] of a batch of one, no mask: the padding rows are keys too):
        the patch linear (all_x_embedder: 64 values a token, (y, x, channel) of its 2 x 2),
        x_pad_token in the image's padding rows, the two noise refiner blocks over the image's
        rows, the caption's rows copied after them, the 30 blocks over all, the final layer's
        LayerNorm (no weights, eps 1e-6) times 1 + scale and its linear of the image's rows
    a block: RMS norm (times 1 + scale), q k v, the RMS norm of each head (q's weight times
        hd^-1/2: the attention's scale), RoPE, the attention, the output, RMS norm (times
        tanh(gate)), + x; RMS norm (times 1 + scale), SiLU(w1) w3, w2, RMS norm (times
        tanh(gate)), + x. The RMS norms' eps 1e-5.

RoPE (RopeEmbedder): the pairs of a head (interleaved) at the angles of three axes of 32, 48 and
48 values (theta 256) of a row's (frame, y, x): the caption's row l at (1 + l, 0, 0) (its padding
too), the image's token (y, x) at (Tp + 1, y, x), the image's padding at (0, 0, 0); the angle a
float64 product rounded to float32 (as precompute_freqs_cis), its cos and sin in float32 by the C
library's cosf and sinf (torch.polar's on the CPU: the same bits).
The attention: float32 on the GPU (ENC_ATTN_TILE; x2, RECLANK_ZIMAGE_X2=1: ENC_ATTN_X2), ENC_ATTN_RANGE
on the CPU.

The matrices: on the GPU F20 (encoders.runtime.F20Lins: each matrix as it is made, its float32 W
and the checkpoint's pages dropped; 16 significant bits a value, float32 products); on the CPU
the checkpoint's float32 W as stored, or F20 (f20). The biases, the norms' weights, the pad
tokens, the t_embedder and the adaLN matrices (host) float32 as stored.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os

import numpy as np

from reclank.encoders.runtime import F20Lins, _EncCompiler, _Plain, _Runner
from reclank.formats.safetensors import Shards
from reclank.models.flux2.dit import timestep_sinusoid
from reclank.runtime.threads import team_threads

SEQ = 32            # the caption's and the image's rows a multiple of SEQ (SEQ_MULTI_OF)
MLP_ROWS = 4096     # the rows of the FFN's products at a time (its buffers)


def _silu(x):
    return (x / (np.float32(1) + np.exp(-x))).astype(np.float32)


def padded(n):
    """n rounded up to a multiple of SEQ."""
    return -(-n // SEQ) * SEQ


def patchify(lat):
    """The DiT's input rows (h/2 w/2, 4 C) of latents (C, h, w): each patch's (y, x, channel)."""
    C, h, w = lat.shape
    x = lat.reshape(C, h // 2, 2, w // 2, 2).transpose(1, 3, 2, 4, 0)
    return np.ascontiguousarray(x.reshape((h // 2) * (w // 2), 4 * C), dtype=np.float32)


def unpatchify(rows, C, hh, ww):
    """The latents (C, 2 hh, 2 ww) of the DiT's output rows (hh ww, 4 C: (y, x, channel))."""
    x = rows.reshape(hh, ww, 2, 2, C).transpose(4, 0, 2, 1, 3)
    return np.ascontiguousarray(x.reshape(C, 2 * hh, 2 * ww), dtype=np.float32)


def _libm():
    """The C library's cosf and sinf (ctypes), or None."""
    try:
        m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        for f in (m.cosf, m.sinf):
            f.restype, f.argtypes = ctypes.c_float, [ctypes.c_float]
        return m
    except (OSError, AttributeError):
        return None


_LIBM = _libm()


def _cos_sin(ang):
    """cos and sin of float32 angles in float32 as torch.polar of complex64 on the CPU: the C
    library's cosf and sinf (glibc's; each distinct angle once), else the float64 values rounded
    (1 ulp apart in a few)."""
    if _LIBM is None:
        a = ang.astype(np.float64)
        return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)
    u, inv = np.unique(ang, return_inverse=True)
    c = np.array([_LIBM.cosf(float(x)) for x in u], np.float32)
    s = np.array([_LIBM.sinf(float(x)) for x in u], np.float32)
    return c[inv].reshape(ang.shape), s[inv].reshape(ang.shape)


def rope_table(pos, axes, theta):
    """cos and sin (rows, sum(axes) / 2) of the positions pos (rows, len(axes)): each axis its
    own frequencies (float64), the angle the float64 product rounded to float32, its cos and sin
    in float32 (RopeEmbedder.precompute_freqs_cis: torch.polar)."""
    cs, sn = [], []
    for i, d in enumerate(axes):
        freqs = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        c, s = _cos_sin((pos[:, i].astype(np.float64)[:, None] * freqs[None, :]).astype(np.float32))
        cs.append(c)
        sn.append(s)
    return np.concatenate(cs, 1), np.concatenate(sn, 1)


class ZImageDiT:
    """The DiT of a transformer directory (config.json and its safetensors). prefix(emb) runs the
    caption's rows; velocity(latents, t) runs a step over them."""

    def __init__(self, path, gpu=False, cpu_threads=None, f20=None, x2=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert list(cfg["all_patch_size"]) == [2] and list(cfg["all_f_patch_size"]) == [1] and cfg["qk_norm"]
        assert cfg["n_kv_heads"] == cfg["n_heads"] and not cfg.get("siglip_feat_dim")
        self.W = W = Shards(path, "diffusion_pytorch_model.safetensors.index.json")
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        f20 = self.gpu if f20 is None else bool(f20)
        assert f20 or not self.gpu, "the GPU's DiT is F20 (float32 W: 24.6 GB)"
        self.f20 = F20Lins(self.gpu, "z-image dit") if f20 else None
        # the attention on the GPU: float32 (ENC_ATTN_TILE), or ENC_ATTN_X2 by x2 (None: RECLANK_ZIMAGE_X2=1)
        self.x2 = os.environ.get("RECLANK_ZIMAGE_X2", "0") == "1" if x2 is None else bool(x2)
        self.width = w = cfg["dim"]
        self.heads = cfg["n_heads"]
        self.hd = hd = w // self.heads
        self.eps = float(cfg["norm_eps"])
        self.in_ch = cfg["in_channels"]
        self.cap_dim = cfg["cap_feat_dim"]
        self.axes, self.theta = list(cfg["axes_dims"]), float(cfg["rope_theta"])
        self.t_scale = np.float32(cfg["t_scale"])
        assert sum(self.axes) == hd
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        self.f = f
        sq = np.float32(hd ** -0.5)
        e = "t_embedder.mlp."
        self.t1, self.t1b = f(e + "0.weight"), f(e + "0.bias")
        self.t2, self.t2b = f(e + "2.weight"), f(e + "2.bias")
        fl = "all_final_layer.2-1."
        self.mod_out, self.mod_outb = f(fl + "adaLN_modulation.1.weight"), f(fl + "adaLN_modulation.1.bias")
        L = self.linear
        self.cap_norm = f("cap_embedder.0.weight")
        self.cap_emb = L("cap_embedder.1")
        self.x_emb = L("all_x_embedder.2-1")
        self.proj = L(fl + "linear")
        self.x_pad, self.cap_pad = f("x_pad_token").reshape(w), f("cap_pad_token").reshape(w)

        def block(p, mod):
            a = p + "attention."
            B = dict(q=L(a + "to_q", False), k=L(a + "to_k", False), v=L(a + "to_v", False),
                     o=L(a + "to_out.0", False),
                     nq=(f(a + "norm_q.weight") * sq).astype(np.float32), nk=f(a + "norm_k.weight"),
                     n1=f(p + "attention_norm1.weight"), n2=f(p + "attention_norm2.weight"),
                     m1=f(p + "ffn_norm1.weight"), m2=f(p + "ffn_norm2.weight"),
                     w1=L(p + "feed_forward.w1", False), w2=L(p + "feed_forward.w2", False),
                     w3=L(p + "feed_forward.w3", False))
            if mod:
                B.update(ada=f(p + "adaLN_modulation.0.weight"), adab=f(p + "adaLN_modulation.0.bias"))
            return B
        self.ctx = [block("context_refiner.%d." % i, False) for i in range(cfg["n_refiner_layers"])]
        self.noise = [block("noise_refiner.%d." % i, True) for i in range(cfg["n_refiner_layers"])]
        self.layers = [block("layers.%d." % i, True) for i in range(cfg["n_layers"])]
        self.inter = self.layers[0]["w1"].shape[0]
        if self.f20 is not None:
            self.f20.report()
        self.runner = _Runner(self._build, keep=4, keep_gpu=3)     # a caption's and a step's
        self.caps = {}          # Tp: the caption's rows (Tp, width), written by the prefix, read by the steps
        self.T = None           # the tokens of the last prefix

    def linear(self, name, bias=True):
        """A linear (out, in) of the checkpoint with its bias; in F20 when f20 (its float32 W and
        the checkpoint's pages dropped then)."""
        src = self.W.get(name + ".weight")
        lin = _Plain(np.ascontiguousarray(src, dtype=np.float32))
        lin.shape = lin.w.shape
        lin.b = np.ascontiguousarray(self.W.get(name + ".bias"), dtype=np.float32) if bias else None
        if self.f20 is not None:
            self.f20.add(lin, src)
        return lin

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    # ---- the step's vectors (host) ----------------------------------------

    def vectors(self, t):
        """The vectors (name: (width,)) of timestep t (the pipeline's (1000 - t) / 1000 times
        t_scale): each modulated block's RMS norm weights (n<i>a, n<i>g, n<i>f, n<i>m: norm1's
        times 1 + scale_msa, norm2's times tanh(gate_msa), ffn_norm1's times 1 + scale_mlp,
        ffn_norm2's times tanh(gate_mlp); the noise refiner's blocks first), the final layer's
        1 + scale (so)."""
        w = self.width
        e = (self.t1 @ timestep_sinusoid(t, self.t1.shape[1]) + self.t1b).astype(np.float32)
        adaln = (self.t2 @ _silu(e) + self.t2b).astype(np.float32)
        out = {}
        one = np.float32(1)
        for i, B in enumerate(self.noise + self.layers):
            sc_a, g_a, sc_f, g_f = (B["ada"] @ adaln + B["adab"]).astype(np.float32).reshape(4, w)
            out.update({"n%da" % i: B["n1"] * (one + sc_a), "n%dg" % i: B["n2"] * np.tanh(g_a),
                        "n%df" % i: B["m1"] * (one + sc_f), "n%dm" % i: B["m2"] * np.tanh(g_f)})
        out["so"] = one + (self.mod_out @ _silu(adaln) + self.mod_outb).astype(np.float32)
        return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}

    def rope(self, Tp, hh, ww):
        """cos and sin (Ip + Tp, hd / 2) of the unified rows: an image of hh x ww tokens (its
        padding at 0) after a caption of Tp rows, then the caption's rows."""
        I = hh * ww
        Ip = padded(I)
        pos = np.zeros((Ip + Tp, 3), np.int64)
        pos[:I, 0] = Tp + 1
        pos[:I, 1] = np.repeat(np.arange(hh), ww)
        pos[:I, 2] = np.tile(np.arange(ww), hh)
        pos[Ip:, 0] = 1 + np.arange(Tp)
        return rope_table(pos, self.axes, self.theta)

    # ---- the programs ------------------------------------------------------

    def _block(self, B, x, rows, att, vec=None):
        """The forms of a block on the rows x (a name; rows of them), q k v in the views of att
        (q, k, v, o, lo, hi: names of the rows of x), the RMS norms' weights vec (n<i>a,
        n<i>g, n<i>f, n<i>m) or the block's own (no modulation)."""
        w, H, hd = self.width, self.heads, self.hd
        q, k, v, o, lo, hi, cos, sin = att
        na, ng, nf, nm = vec if vec is not None else (B["n1"], B["n2"], B["m1"], B["m2"])
        sfx = x[1:] if x.startswith("x") else x
        h, t = "h" + sfx, "t" + sfx
        forms = [("set", h, ("enc_rms", x, na, w)),
                 ("set", t, ("enc_linear", B["q"], h)),
                 ("set", q, ("enc_rms", t, B["nq"], hd)),
                 ("set", t, ("enc_linear", B["k"], h)),
                 ("set", k, ("enc_rms", t, B["nk"], hd)),
                 ("set", v, ("enc_linear", B["v"], h)),
                 ("enc_rope_tab", q, cos, sin, H, hd),
                 ("enc_rope_tab", k, cos, sin, H, hd),
                 ("set", o, ("enc_attn_range_x2", q, k, v, lo, hi, H, hd)),
                 ("set", t, ("enc_linear", B["o"], o)),
                 ("set", t, ("enc_rms", t, ng, w)),
                 ("enc_add", x, t),
                 ("set", h, ("enc_rms", x, nf, w))]
        for r0 in range(0, rows, MLP_ROWS):
            n = "%s_%d_%d" % (sfx, r0, min(rows, r0 + MLP_ROWS))
            forms += [("set", "g" + n, ("enc_linear", B["w1"], "h" + n)),
                      ("set", "u" + n, ("enc_linear", B["w3"], "h" + n)),
                      ("set", "u" + n, ("enc_silu_mul", "g" + n, "u" + n)),
                      ("set", "t" + n, ("enc_linear", B["w2"], "u" + n))]
        return forms + [("set", t, ("enc_rms", t, nm, w)),
                        ("enc_add", x, t)]

    def _views(self, env, sfx, rows, r0=0):
        """The names of the rows [r0, r0 + rows) of the buffers x h t q k v o (x<sfx>, ...),
        and of the FFN's chunks (h t g u <sfx>_<a>_<b>)."""
        for name in ("x", "h", "t", "q", "k", "v", "o"):
            env[name + sfx] = env[name][r0:r0 + rows]
        for a in range(0, rows, MLP_ROWS):
            b = min(rows, a + MLP_ROWS)
            n = "%s_%d_%d" % (sfx, a, b)
            env["h" + n], env["t" + n] = env["h" + sfx][a:b], env["t" + sfx][a:b]
            env["g" + n], env["u" + n] = env["g"][:b - a], env["u"][:b - a]

    def program(self, kind, *key, gpu=False):
        """kind "text" (key (T, Tp)): the prefix program: input "txt" (Tp, cap dim: the prompt's T
        rows, the rest any), output the caption's rows (self.caps[Tp]). kind "step" (key (Tp, hh,
        ww)): a step of an image of hh x ww tokens and a caption of Tp rows: input "img" (I, 64),
        the vectors of vectors(); output "vel" (I, 64)."""
        w, inter = self.width, self.inter
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c = _EncCompiler(self.eps, pack=not gpu)
        c.attn_f32 = not self.x2
        env = c.env
        if kind == "text":
            T, Tp = key
            cap = self.caps.setdefault(Tp, z(Tp, w))
            env.update(txt=z(Tp, self.cap_dim), tn=z(Tp, self.cap_dim), x=cap, h=z(Tp, w), t=z(Tp, w), q=z(Tp, w),
                       k=z(Tp, w), v=z(Tp, w), o=z(Tp, w), g=z(min(Tp, MLP_ROWS), inter),
                       u=z(min(Tp, MLP_ROWS), inter), lo=np.zeros(Tp, np.int32), ahi=np.full(Tp, Tp, np.int32))
            cos, sin = self.rope(Tp, 0, 0)
            env.update(cos=cos, sin=sin)
            self._views(env, "", Tp)
            forms = [("set", "tn", ("enc_rms", "txt", self.cap_norm, self.cap_dim)),
                     ("set", "x", ("enc_linear", self.cap_emb, "tn"))]
            if Tp > T:
                env["xpad"] = cap[T:]
                env["pad"] = np.ascontiguousarray(np.broadcast_to(self.cap_pad, (Tp - T, w)))
                forms.append(("set", "xpad", ("copy", "pad")))
            att = ("q", "k", "v", "o", "lo", "ahi", "cos", "sin")
            for B in self.ctx:
                forms += self._block(B, "x", Tp, att)
            c.compile(("seq", *forms))
            return c.p.finish()
        Tp, hh, ww = key
        I = hh * ww
        Ip = padded(I)
        R = Ip + Tp
        cos, sin = self.rope(Tp, hh, ww)
        env.update(img=z(I, 4 * self.in_ch), vel=z(I, 4 * self.in_ch),
                   x=z(R, w), h=z(R, w), t=z(R, w), q=z(R, w), k=z(R, w), v=z(R, w), o=z(R, w),
                   g=z(min(R, MLP_ROWS), inter), u=z(min(R, MLP_ROWS), inter), cos=cos, sin=sin,
                   cap=self.caps.setdefault(Tp, z(Tp, w)), lo=np.zeros(R, np.int32),
                   ahi=np.full(R, R, np.int32), ilo=np.zeros(Ip, np.int32), ihi=np.full(Ip, Ip, np.int32),
                   **{k_: np.zeros_like(v_) for k_, v_ in self.vectors(0.0).items()})
        self._views(env, "", R)
        self._views(env, "i", Ip)
        env["xc"], env["cosi"], env["sini"] = env["x"][Ip:], cos[:Ip], sin[:Ip]
        env["xe"], env["hf"] = env["x"][:I], env["h"][:I]
        forms = [("set", "xe", ("enc_linear", self.x_emb, "img"))]
        if Ip > I:
            env["xpad"] = env["x"][I:Ip]
            env["pad"] = np.ascontiguousarray(np.broadcast_to(self.x_pad, (Ip - I, w)))
            forms.append(("set", "xpad", ("copy", "pad")))
        att = ("qi", "ki", "vi", "oi", "ilo", "ihi", "cosi", "sini")
        for i, B in enumerate(self.noise):
            forms += self._block(B, "xi", Ip, att, ["n%d%s" % (i, s) for s in "agfm"])
        forms.append(("set", "xc", ("copy", "cap")))
        att = ("q", "k", "v", "o", "lo", "ahi", "cos", "sin")
        for j, B in enumerate(self.layers):
            i = len(self.noise) + j
            forms += self._block(B, "x", R, att, ["n%d%s" % (i, s) for s in "agfm"])
        forms += [("set", "hf", ("enc_lnorm", "xe", "so", None, w, 1e-6)),
                  ("set", "vel", ("enc_linear", self.proj, "hf"))]
        c.compile(("seq", *forms))
        return c.p.finish()

    def prefix(self, emb):
        """Run the prefix program of the caption's rows emb (T, cap dim: the pipeline's
        prompt_embeds): its rows for the steps."""
        T = emb.shape[0]
        Tp = padded(T)
        prog, g = self.runner.get(("text", T, Tp), self.gpu)
        txt = prog.names["txt"]
        txt[:T] = emb
        txt[T:] = emb[-1:]
        if g is not None:
            g.upload("txt")
            g.run()
        else:
            prog.run()
        self.T = T

    def velocity(self, lat, t):
        """The DiT's output (C, h, w) of latents lat (C, h, w) at timestep t (the pipeline's (1000
        - t) / 1000, times t_scale inside), over the caption of the last prefix."""
        C, h, w = lat.shape
        hh, ww = h // 2, w // 2
        prog, g = self.runner.get(("step", padded(self.T), hh, ww), self.gpu)
        inputs = dict(img=patchify(lat))
        inputs.update(self.vectors(np.float32(np.float32(t) * self.t_scale)))
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
            g.download("vel")
        else:
            prog.run()
        return unpatchify(prog.names["vel"], self.in_ch, hh, ww)
