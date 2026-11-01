"""FLUX.2's DiT (diffusers' Flux2Transformer2DModel; [klein] 4B: 5 double
stream blocks, 20 single stream blocks, width 3072, 24 heads of 128) as a
program of one step: the velocity of the packed latents.

Ported from diffusers (src/diffusers/models/transformers/transformer_flux2.py,
Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team;
models/embeddings.py, Copyright 2026 The HuggingFace Team; Apache License 2.0).
See THIRD_PARTY_NOTICES.md.

    the step's vectors, on the host in float32: the timestep's sinusoid (256,
        [cos, sin]), the MLP (linear, SiLU, linear: temb), and from SiLU(temb)
        the modulations shared by all blocks: shift, scale and gate of the
        attention and of the MLP for the image and the text streams of the
        double blocks, of the single blocks, and the scale and shift of the
        last norm
    the rows: the text's (512, context_embedder of the text encoder's taps)
        then the image's (x_embedder of the latents); views of one buffer, so
        the single blocks take them as they are
    a double block: each stream LN (no weights) times 1 + scale plus shift
        (ENC_LNORM with the step's vectors as its weights), its own q k v, the
        RMS norm of each head (q's weight times hd^-1/2), RoPE, the attention
        of all rows over all rows, its own output, times the gate, + x; LN
        and modulation, its own SwiGLU MLP (the halves of linear_in: gate and
        up), times the gate, + x
    a single block: LN and modulation, q k v and the MLP's gate and up (the
        rows of to_qkv_mlp_proj), the RMS norms, RoPE, the attention, SiLU(gate)
        up; the output (the columns of to_out: the attention's and the MLP's,
        added), times the gate, + x
    the image rows: LN times 1 + scale plus shift, proj_out

RoPE: the pairs of a head (interleaved) at the angles of four axes of 32
values each (theta 2000): text rows (0, 0, 0, l), image rows (0, h, w, 0)
(ENC_ROPE_TAB, the table from the host in float64 as diffusers).

Reference images (editing): their packed latents are image rows after the
noisy ones (the image stream's weights, the attention over all rows), at
RoPE (10 + 10 i, h, w, 0) for reference i; the velocity is the noisy rows'.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.qwen3.dense import lin16
from reclank.runtime.threads import team_threads


def _silu(x):
    return x / (1.0 + np.exp(-x))


def timestep_sinusoid(t, dim=256, max_period=10000):
    """diffusers' get_timestep_embedding(flip_sin_to_cos=True, shift 0) of
    one timestep, float32."""
    half = dim // 2
    ex = np.exp((-math.log(max_period) * np.arange(half, dtype=np.float32) / np.float32(half)).astype(np.float32))
    e = (np.float32(t) * ex).astype(np.float32)
    return np.concatenate([np.cos(e), np.sin(e)]).astype(np.float32)


def rope_table(ids, axes, theta):
    """cos and sin (rows, sum(axes) / 2) of the position ids (rows, len(axes))
    (diffusers' Flux2PosEmbed: float64 angles, each axis its own)."""
    cs, sn = [], []
    for i, d in enumerate(axes):
        freqs = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        ang = ids[:, i].astype(np.float64)[:, None] * freqs[None, :]
        cs.append(np.cos(ang))
        sn.append(np.sin(ang))
    return np.concatenate(cs, 1).astype(np.float32), np.concatenate(sn, 1).astype(np.float32)


class Flux2DiT:
    """The DiT of a transformer directory (config.json and its safetensors).
    velocity(latents, timestep, text) runs a step."""

    def __init__(self, path, gpu=False, cpu_threads=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg["patch_size"] == 1 and not cfg["guidance_embeds"]
        self.W = W = SafeTensors(os.path.join(path, "diffusion_pytorch_model.safetensors"))
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.heads, self.hd = cfg["num_attention_heads"], cfg["attention_head_dim"]
        self.width = w = self.heads * self.hd
        self.inter = inter = int(w * cfg["mlp_ratio"])
        self.eps = float(cfg["eps"])
        self.axes, self.theta = list(cfg["axes_dims_rope"]), float(cfg["rope_theta"])
        self.in_ch = cfg["in_channels"]
        self.tdim = cfg["timestep_guidance_channels"]
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        sq = np.float32(self.hd ** -0.5)
        L = lambda n, **kw: lin16(W, n, gpu, bias=False, **kw)  # noqa: E731
        # the host's matrices of the step's vectors
        e = "time_guidance_embed.timestep_embedder."
        self.t1, self.t2 = f(e + "linear_1.weight"), f(e + "linear_2.weight")
        self.mod_img = f("double_stream_modulation_img.linear.weight")
        self.mod_txt = f("double_stream_modulation_txt.linear.weight")
        self.mod_one = f("single_stream_modulation.linear.weight")
        self.mod_out = f("norm_out.linear.weight")
        self.x_emb, self.ctx = L("x_embedder"), L("context_embedder")
        self.proj = L("proj_out")
        self.double = []
        for i in range(cfg["num_layers"]):
            p = "transformer_blocks.%d." % i
            a = p + "attn."
            self.double.append(dict(
                q_i=L(a + "to_q"), k_i=L(a + "to_k"), v_i=L(a + "to_v"), o_i=L(a + "to_out.0"),
                q_t=L(a + "add_q_proj"), k_t=L(a + "add_k_proj"), v_t=L(a + "add_v_proj"), o_t=L(a + "to_add_out"),
                nq_i=(f(a + "norm_q.weight") * sq).astype(np.float32), nk_i=f(a + "norm_k.weight"),
                nq_t=(f(a + "norm_added_q.weight") * sq).astype(np.float32), nk_t=f(a + "norm_added_k.weight"),
                fg_i=L(p + "ff.linear_in", rows=slice(0, inter)), fu_i=L(p + "ff.linear_in", rows=slice(inter, 2 * inter)),
                fo_i=L(p + "ff.linear_out"),
                fg_t=L(p + "ff_context.linear_in", rows=slice(0, inter)),
                fu_t=L(p + "ff_context.linear_in", rows=slice(inter, 2 * inter)),
                fo_t=L(p + "ff_context.linear_out")))
        self.single = []
        for i in range(cfg["num_single_layers"]):
            a = "single_transformer_blocks.%d.attn." % i
            m = a + "to_qkv_mlp_proj"
            self.single.append(dict(
                q=L(m, rows=slice(0, w)), k=L(m, rows=slice(w, 2 * w)), v=L(m, rows=slice(2 * w, 3 * w)),
                g=L(m, rows=slice(3 * w, 3 * w + inter)), u=L(m, rows=slice(3 * w + inter, 3 * w + 2 * inter)),
                oa=L(a + "to_out", cols=slice(0, w)), om=L(a + "to_out", cols=slice(w, w + inter)),
                nq=(f(a + "norm_q.weight") * sq).astype(np.float32), nk=f(a + "norm_k.weight")))
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    # ---- the step's vectors (host) ----------------------------------------

    def vectors(self, t):
        """The modulation vectors (name: (width,)) of timestep t (0 to 1000)."""
        w = self.width
        temb = self.t2 @ _silu(self.t1 @ timestep_sinusoid(t, self.tdim))
        s = _silu(temb).astype(np.float32)
        out = {}
        for name, mat in (("i", self.mod_img), ("t", self.mod_txt)):
            sh_a, sc_a, g_a, sh_f, sc_f, g_f = (mat @ s).astype(np.float32).reshape(6, w)
            out.update({"ma_w_" + name: 1 + sc_a, "ma_b_" + name: sh_a, "ga_" + name: g_a,
                        "mf_w_" + name: 1 + sc_f, "mf_b_" + name: sh_f, "gf_" + name: g_f})
        sh, sc, g = (self.mod_one @ s).astype(np.float32).reshape(3, w)
        out.update(s_w=1 + sc, s_b=sh, s_g=g)
        sc, sh = (self.mod_out @ s).astype(np.float32).reshape(2, w)
        out.update(no_w=1 + sc, no_b=sh)
        return {k: v.astype(np.float32) for k, v in out.items()}

    def rope(self, T, hh, ww, refs=()):
        """cos and sin of T text rows, an image of hh x ww tokens and the
        reference images of refs ((h, w) tokens each)."""
        txt = np.zeros((T, 4), np.int64)
        txt[:, 3] = np.arange(T)
        ids = [txt]
        for t, (h, w) in [(0, (hh, ww))] + [(10 + 10 * i, r) for i, r in enumerate(refs)]:
            img = np.zeros((h * w, 4), np.int64)
            img[:, 0] = t
            img[:, 1] = np.repeat(np.arange(h), w)
            img[:, 2] = np.tile(np.arange(w), h)
            ids.append(img)
        return rope_table(np.concatenate(ids), self.axes, self.theta)

    # ---- the program -------------------------------------------------------

    def program(self, T, hh, ww, refs=(), gpu=False):
        """The program of a step of T text rows, an image of hh x ww tokens
        and the reference images of refs ((h, w) tokens each). Inputs: "txt"
        (T, joint dim), "img" (the image's and the references' rows, in
        channels), the vectors of vectors(); output "vel" (hh ww, in
        channels)."""
        w, H, hd, inter = self.width, self.heads, self.hd, self.inter
        I = hh * ww
        J = sum(h * w_ for h, w_ in refs)
        R = T + I + J
        c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=inter)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        env = c.env
        cos, sin = self.rope(T, hh, ww, refs)
        env.update(txt=z(T, self.ctx.shape[1]), img=z(I + J, self.in_ch), vel=z(I, self.in_ch), cos=cos, sin=sin,
                   alo=np.zeros(R, np.int32), ahi=np.full(R, R, np.int32), t2=z(R, w))
        for name in ("x", "h", "q", "k", "v", "qn", "kn", "o", "t"):
            env[name] = z(R, w)
            env[name + "t"], env[name + "i"] = env[name][:T], env[name][T:]
        # the noisy rows (the output's) of the image stream
        xo, ho = ("xi", "hi") if not J else ("xo", "ho")
        if J:
            env["xo"], env["ho"] = env["x"][T:T + I], env["h"][T:T + I]
        for name in ("g", "u"):
            env[name] = z(R, inter)
            env[name + "t"], env[name + "i"] = env[name][:T], env[name][T:]
        for k_, v_ in self.vectors(1000.0).items():
            env[k_] = np.zeros_like(v_)
        forms = [("set", "xt", ("enc_linear", self.ctx, "txt")),
                 ("set", "xi", ("enc_linear", self.x_emb, "img"))]

        def attention(heads_t, heads_i):
            """q k v of h by stream (or one), the norms, RoPE, the attention"""
            out = []
            for s, B in (("t", heads_t), ("i", heads_i)) if heads_i is not None else (("", heads_t),):
                out += [("set", "q" + s, ("enc_linear", B["q"], "h" + s)),
                        ("set", "k" + s, ("enc_linear", B["k"], "h" + s)),
                        ("set", "v" + s, ("enc_linear", B["v"], "h" + s)),
                        ("set", "qn" + s, ("enc_rms", "q" + s, B["nq"], hd)),
                        ("set", "kn" + s, ("enc_rms", "k" + s, B["nk"], hd))]
            return out + [("enc_rope_tab", "qn", "cos", "sin", H, hd),
                          ("enc_rope_tab", "kn", "cos", "sin", H, hd),
                          ("set", "o", ("enc_attn_range_x2", "qn", "kn", "v", "alo", "ahi", H, hd))]

        for B in self.double:
            bt = dict(q=B["q_t"], k=B["k_t"], v=B["v_t"], nq=B["nq_t"], nk=B["nk_t"])
            bi = dict(q=B["q_i"], k=B["k_i"], v=B["v_i"], nq=B["nq_i"], nk=B["nk_i"])
            forms += [("set", "ht", ("enc_lnorm", "xt", "ma_w_t", "ma_b_t", w)),
                      ("set", "hi", ("enc_lnorm", "xi", "ma_w_i", "ma_b_i", w))]
            forms += attention(bt, bi)
            forms += [("set", "tt", ("enc_linear", B["o_t"], "ot")),
                      ("set", "ti", ("enc_linear", B["o_i"], "oi")),
                      ("set", "tt", ("enc_mul_vec", "tt", "ga_t")),
                      ("set", "ti", ("enc_mul_vec", "ti", "ga_i")),
                      ("enc_add", "x", "t"),
                      ("set", "ht", ("enc_lnorm", "xt", "mf_w_t", "mf_b_t", w)),
                      ("set", "hi", ("enc_lnorm", "xi", "mf_w_i", "mf_b_i", w))]
            for s in ("t", "i"):
                forms += [("set", "g" + s, ("enc_linear", B["fg_" + s], "h" + s)),
                          ("set", "u" + s, ("enc_linear", B["fu_" + s], "h" + s)),
                          ("set", "u" + s, ("enc_silu_mul", "g" + s, "u" + s)),
                          ("set", "t" + s, ("enc_linear", B["fo_" + s], "u" + s)),
                          ("set", "t" + s, ("enc_mul_vec", "t" + s, "gf_" + s))]
            forms.append(("enc_add", "x", "t"))
        for S in self.single:
            forms.append(("set", "h", ("enc_lnorm", "x", "s_w", "s_b", w)))
            forms += attention(S, None)
            forms += [("set", "g", ("enc_linear", S["g"], "h")),
                      ("set", "u", ("enc_linear", S["u"], "h")),
                      ("set", "u", ("enc_silu_mul", "g", "u")),
                      ("set", "t", ("enc_linear", S["oa"], "o")),
                      ("set", "t2", ("enc_linear", S["om"], "u")),
                      ("enc_add", "t", "t2"),
                      ("set", "t", ("enc_mul_vec", "t", "s_g")),
                      ("enc_add", "x", "t")]
        forms += [("set", ho, ("enc_lnorm", xo, "no_w", "no_b", w)),
                  ("set", "vel", ("enc_linear", self.proj, ho))]
        c.compile(("seq", *forms))
        return c.p.finish()

    def velocity(self, img, t, txt, hh, ww, ref=None, refs=()):
        """The DiT's output (hh ww, in channels) of the packed latents img at
        timestep t (0 to 1000), the text rows txt (T, joint dim); ref: the
        reference images' packed latents (their rows, in channels), refs
        their (h, w)."""
        T = txt.shape[0]
        refs = tuple(tuple(r) for r in refs) if ref is not None else ()
        prog, g = self.runner.get((T, hh, ww, refs), self.gpu)
        inputs = dict(img=img if ref is None else np.concatenate([img, ref]), txt=txt)
        inputs.update(self.vectors(t))
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
