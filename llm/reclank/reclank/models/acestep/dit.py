"""ACE-Step 1.5's DiT (the turbo model of the authors' modeling_acestep_v15_turbo.py:
AceStepConditionGenerationModel without the 5 Hz codes) as two programs.

Ported from ACE-Step 1.5 (acestep/models/turbo/modeling_acestep_v15_turbo.py; MIT
License, Copyright (c) 2026 ACEStep; its model files Apache License 2.0, Copyright
2025 The ACESTEO Team). See THIRD_PARTY_NOTICES.md.

The conditions, once a request ("cond"):
    the lyric encoder: the lyrics' embedding rows (the text encoder's table)
        through a linear (1024 -> 2048), 8 Qwen3 layers (bidirectional; the
        layers of "sliding_attention" see the rows within 128), its norm
    the timbre encoder: the reference latents (none given: the silence
        latent's first 750 frames) through a linear (64 -> 2048), 4 such
        layers, its norm; its first row
    the text: the text encoder's last hidden state through text_projector
    packed: the lyrics' rows, the timbre row, the text's rows; the DiT's
        condition_embedder (with bias); then each DiT layer's keys (k_proj,
        the RMS norm of each head) and values of them, into the caches
A step ("step"):
    the input: the context latents (the silence latent and the chunk mask: 128
        columns) beside the noisy latents (64), two frames a token (proj_in:
        a conv of kernel and stride 2)
    24 layers: RMS norm times 1 + scale plus shift (the step's vectors: the
        layer's scale_shift_table plus the timestep's projection), the self
        attention (GQA 16 / 8 heads of 128, q and k RMS norms, RoPE; the
        layers of "sliding_attention" see the tokens within 128), times the
        gate, + x; RMS norm, the cross attention over the conditions' cached
        keys, + x; RMS norm and modulation, SwiGLU, times the gate, + x
    the output: RMS norm times 1 + scale plus shift, proj_out (a transposed
        conv of kernel and stride 2: two frames of 64 a token)

The step's vectors are made on the host in float32 (the sinusoid of t 1000,
the MLPs of time_embed and time_embed_r, the latter at t - r = 0). The
matrices: float32 on the CPU, the checkpoint's bfloat16 rows on the GPU
(KQ_BF16).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.audio.dac import conv_lin, mat
from reclank.models.qwen3.dense import lin16, qwen3_layers, rope_inv
from reclank.runtime.threads import team_threads


def _silu(x):
    return x / (1.0 + np.exp(-x))


class AceStepDiT:
    """The DiT of an acestep-v15-turbo directory (config.json and its
    safetensors, silence_latent.npy)."""

    def __init__(self, path, gpu=False, cpu_threads=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg.get("is_turbo") and cfg["patch_size"] == 2
        self.cfg = cfg
        self.W = W = SafeTensors(os.path.join(path, "model.safetensors"))
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.width, self.H, self.KV = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.eps = float(cfg["rms_norm_eps"])
        self.window = cfg["sliding_window"]
        self.types = cfg["layer_types"]
        self.inv = rope_inv(self.hd, cfg["rope_theta"])
        self.latent = cfg["audio_acoustic_hidden_dim"]          # 64
        self.silence = np.load(os.path.join(path, "silence_latent.npy")).astype(np.float32)   # (15000, 64)
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        L = lambda n, **kw: lin16(W, n, gpu, **kw)  # noqa: E731
        hd = self.hd
        e = "encoder."
        self.lyric_in = L(e + "lyric_encoder.embed_tokens")
        self.lyric = qwen3_layers(W, e + "lyric_encoder.layers.", cfg["num_lyric_encoder_hidden_layers"], hd, gpu)
        self.lyric_norm = f(e + "lyric_encoder.norm.weight")
        self.timbre_in = L(e + "timbre_encoder.embed_tokens")
        self.timbre = qwen3_layers(W, e + "timbre_encoder.layers.", cfg["num_timbre_encoder_hidden_layers"], hd, gpu)
        self.timbre_norm = f(e + "timbre_encoder.norm.weight")
        self.text_proj = L(e + "text_projector", bias=False)
        d = "decoder."
        self.cond_emb = L(d + "condition_embedder")
        sq = np.float32(hd ** -0.5)
        self.layers = []
        for i in range(cfg["num_hidden_layers"]):
            p = d + "layers.%d." % i
            a, x = p + "self_attn.", p + "cross_attn."
            self.layers.append(dict(
                n_a=f(p + "self_attn_norm.weight"), n_c=f(p + "cross_attn_norm.weight"), n_m=f(p + "mlp_norm.weight"),
                table=f(p + "scale_shift_table").reshape(6, -1),
                q=L(a + "q_proj", bias=False), k=L(a + "k_proj", bias=False), v=L(a + "v_proj", bias=False),
                o=L(a + "o_proj", bias=False),
                qn=(f(a + "q_norm.weight") * sq).astype(np.float32), kn=f(a + "k_norm.weight"),
                cq=L(x + "q_proj", bias=False), ck=L(x + "k_proj", bias=False), cv=L(x + "v_proj", bias=False),
                co=L(x + "o_proj", bias=False),
                cqn=(f(x + "q_norm.weight") * sq).astype(np.float32), ckn=f(x + "k_norm.weight"),
                gate=L(p + "mlp.gate_proj", bias=False), up=L(p + "mlp.up_proj", bias=False),
                down=L(p + "mlp.down_proj", bias=False)))
        self.inter = self.layers[0]["gate"].shape[0]
        self.norm_out = f(d + "norm_out.weight")
        self.table_out = f(d + "scale_shift_table").reshape(2, -1)
        self.proj_in = conv_lin(W.get_bf16(d + "proj_in.1.weight"), f(d + "proj_in.1.bias"), gpu)   # (2048, 2 192)
        wo = W.get_bf16(d + "proj_out.1.weight")                                                  # (2048, 64, 2)
        self.proj_out = mat(np.concatenate([wo[:, :, k].T for k in range(wo.shape[2])]), gpu)     # (2 64, 2048)
        self.proj_out.b = np.tile(f(d + "proj_out.1.bias"), wo.shape[2])
        self.time = [dict(l1=(f(t + "linear_1.weight"), f(t + "linear_1.bias")),
                          l2=(f(t + "linear_2.weight"), f(t + "linear_2.bias")),
                          proj=(f(t + "time_proj.weight"), f(t + "time_proj.bias")))
                     for t in (d + "time_embed.", d + "time_embed_r.")]
        # the 5 Hz codes of the planner LM (planner.py): FSQ's levels and project_out on the host, the
        # detokenizer (a linear, 5 learned rows a code, 2 bidirectional layers, a norm, 2048 -> 64)
        self.fsq_levels = np.array(cfg["fsq_input_levels"])
        self.fsq_out = (f("tokenizer.quantizer.project_out.weight"), f("tokenizer.quantizer.project_out.bias"))
        self.pool = cfg["pool_window_size"]
        k = "detokenizer."
        self.dt_in = L(k + "embed_tokens")
        self.dt_special = f(k + "special_tokens").reshape(self.pool, -1)
        self.dt_layers = qwen3_layers(W, k + "layers.", cfg["num_attention_pooler_hidden_layers"], hd, gpu)
        self.dt_norm = f(k + "norm.weight")
        self.dt_out = L(k + "proj_out")
        self.caches = {}
        self.runner = _Runner(self._build, keep=6, keep_gpu=3)

    # ---- the host's parts -------------------------------------------------

    def _temb(self, T, t):
        """TimestepEmbedding: temb (width,) and its projection (6, width)."""
        half = 128
        freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / np.float32(half)).astype(np.float32)
        a = (np.float32(t * 1000.0) * freqs).astype(np.float32)
        tf = np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)
        h = _silu(T["l1"][0] @ tf + T["l1"][1])
        temb = (T["l2"][0] @ h + T["l2"][1]).astype(np.float32)
        proj = (T["proj"][0] @ _silu(temb) + T["proj"][1]).astype(np.float32).reshape(6, -1)
        return temb, proj

    def mods(self, t):
        """The step's vectors (24 6 + 2, width): each layer's RMS weights times
        1 + scale, shifts and gates (n_a (1 + scale), shift, gate, n_m (1 +
        c_scale), c_shift, c_gate), then the output's."""
        te0, pr0 = self._temb(self.time[0], t)
        te1, pr1 = self._temb(self.time[1], 0.0)
        temb, proj = te0 + te1, pr0 + pr1
        rows = []
        for Ly in self.layers:
            sh, sc, g, csh, csc, cg = Ly["table"] + proj
            rows += [Ly["n_a"] * (1 + sc), sh, g, Ly["n_m"] * (1 + csc), csh, cg]
        sh, sc = self.table_out + temb
        rows += [self.norm_out * (1 + sc), sh]
        return np.stack(rows).astype(np.float32)

    def context(self, frames, hints=None):
        """The context latents (frames, 128): the source latents (the
        planner's hints, padded with the silence latent; none: the silence
        latent alone) beside the chunk mask (ones)."""
        src = self.silence[:frames].copy()
        if hints is not None:
            n = min(len(hints), frames)
            src[:n] = hints[:n]
        return np.concatenate([src, np.ones((frames, self.latent), np.float32)], 1)

    # ---- the programs ------------------------------------------------------

    def cache(self, Te):
        c = self.caches.get(Te)
        if c is None:
            c = {}
            for i in range(len(self.layers)):
                for name in ("ck", "cv"):
                    c["%s%d" % (name, i)] = np.zeros((Te, self.KV * self.hd), np.float32)
            self.caches[Te] = c
        return c

    def _build(self, key, gpu):
        build = dict(cond=self.cond_program, step=self.step_program, detok=self.detok_program)[key[0]]
        p = build(*key[1:], gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def _ranges(self, n, kind):
        i = np.arange(n)
        if kind == "sliding_attention":
            return (np.maximum(i - self.window, 0).astype(np.int32),
                    np.minimum(i + self.window + 1, n).astype(np.int32))
        return np.zeros(n, np.int32), np.full(n, n, np.int32)

    def _encoder(self, c, forms, x, n, layers, pfx):
        """the forms of bidirectional Qwen3 layers over the buffer x (n rows)"""
        w, H, KV, hd = self.width, self.H, self.KV, self.hd
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        env = c.env
        for name, cols in (("h", w), ("q", H * hd), ("k", KV * hd), ("v", KV * hd), ("qn", H * hd), ("kn", KV * hd),
                           ("o", H * hd), ("t", w), ("g", self.inter), ("u", self.inter)):
            env[pfx + name] = z(n, cols)
        env[pfx + "pos"] = np.arange(n, dtype=np.int32)
        for kind in ("sliding_attention", "full_attention"):
            env[pfx + kind + "_lo"], env[pfx + kind + "_hi"] = self._ranges(n, kind)
        P = lambda s: pfx + s  # noqa: E731
        for i, Ly in enumerate(layers):
            kind = self.types[i]
            forms += [
                ("set", P("h"), ("enc_rms", x, Ly["ln_a"], w)),
                ("set", P("q"), ("enc_linear", Ly["q"], P("h"))),
                ("set", P("k"), ("enc_linear", Ly["k"], P("h"))),
                ("set", P("v"), ("enc_linear", Ly["v"], P("h"))),
                ("set", P("qn"), ("enc_rms", P("q"), Ly["qn"], hd)),
                ("set", P("kn"), ("enc_rms", P("k"), Ly["kn"], hd)),
                ("enc_rope1d", P("qn"), P("pos"), self.inv, H, hd),
                ("enc_rope1d", P("kn"), P("pos"), self.inv, KV, hd),
                ("set", P("o"), ("enc_attn_range", P("qn"), P("kn"), P("v"), P(kind + "_lo"), P(kind + "_hi"),
                                 H, KV, hd)),
                ("set", P("t"), ("enc_linear", Ly["o"], P("o"))),
                ("enc_add", x, P("t")),
                ("set", P("h"), ("enc_rms", x, Ly["ln_m"], w)),
                ("set", P("g"), ("enc_linear", Ly["gate"], P("h"))),
                ("set", P("u"), ("enc_linear", Ly["up"], P("h"))),
                ("set", P("u"), ("enc_silu_mul", P("g"), P("u"))),
                ("set", P("t"), ("enc_linear", Ly["down"], P("u"))),
                ("enc_add", x, P("t")),
            ]

    def detok_program(self, n, gpu=False):
        """The detokenizer's program of n codes: input "q" (n, width: FSQ's
        vectors after project_out); output "hints" (n pool, 64): each code's
        pool frames (the code's rows attend to each other only, at positions
        0 to pool - 1)."""
        w = self.width
        P = self.pool
        R = n * P
        c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=self.inter)
        env = c.env
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        env.update(q=z(n, w), e=z(n, w), x=z(R, w), special=np.tile(self.dt_special, (n, 1)),
                   hints=z(R, self.dt_out.shape[0]))
        forms = [("set", "e", ("enc_linear", self.dt_in, "q")),
                 ("set", "x", ("enc_gather", "e", (np.arange(R) // P).astype(np.int32)[:, None], w)),
                 ("enc_add", "x", "special")]
        self._encoder(c, forms, "x", R, self.dt_layers, "D_")
        i = np.arange(R)
        env["D_pos"] = (i % P).astype(np.int32)
        for kind in ("sliding_attention", "full_attention"):
            env["D_" + kind + "_lo"] = (i - i % P).astype(np.int32)
            env["D_" + kind + "_hi"] = (i - i % P + P).astype(np.int32)
        forms += [("set", "x", ("enc_rms", "x", self.dt_norm, w)),
                  ("set", "hints", ("enc_linear", self.dt_out, "x"))]
        c.compile(("seq", *forms))
        return c.p.finish()

    def detokenize(self, codes):
        """The 25 Hz hints (len(codes) pool, 64) of the planner's codes: FSQ's
        levels of each code (preserve_symmetry: l 2 / (L - 1) - 1), project_out
        (host, float32), the detokenizer's program."""
        codes = np.asarray(codes, np.int64)
        basis = np.cumprod(np.concatenate([[1], self.fsq_levels[:-1]]))
        lv = (codes[:, None] // basis) % self.fsq_levels
        zq = (lv * (2.0 / (self.fsq_levels - 1)) - 1).astype(np.float32)
        q = (zq @ self.fsq_out[0].T + self.fsq_out[1]).astype(np.float32)
        hints, = self._run(("detok", len(codes)), dict(q=q), ["hints"])
        return hints

    def cond_program(self, Tt, Tl, gpu=False):
        """The conditions' program: inputs "text" (Tt, 1024: the text
        encoder's last hidden state), "lyric" (Tl, 1024: the lyrics'
        embedding rows), "refer" (750, 64); outputs "enc" (Tl + 1 + Tt,
        width) and the caches of the keys and values."""
        w, KV, hd = self.width, self.KV, self.hd
        Te = Tl + 1 + Tt
        R = 750
        c = _EncCompiler(self.eps, pack=not gpu, rows=max(R, Te), kmax=self.inter)
        env = c.env
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        env.update(self.cache(Te))
        env.update(text=z(Tt, self.text_proj.shape[1]), lyric=z(Tl, self.lyric_in.shape[1]),
                   refer=z(R, self.timbre_in.shape[1]), lx=z(Tl, w), tx=z(R, w), tn=z(R, w), packed=z(Te, w),
                   enc=z(Te, w), ek=z(Te, KV * hd))
        env["p_lyric"], env["p_timbre"], env["p_text"] = (env["packed"][:Tl], env["packed"][Tl:Tl + 1],
                                                          env["packed"][Tl + 1:])
        env["tn0"] = env["tn"][:1]
        forms = [("set", "lx", ("enc_linear", self.lyric_in, "lyric"))]
        self._encoder(c, forms, "lx", Tl, self.lyric, "L_")
        forms += [("set", "p_lyric", ("enc_rms", "lx", self.lyric_norm, w)),
                  ("set", "tx", ("enc_linear", self.timbre_in, "refer"))]
        self._encoder(c, forms, "tx", R, self.timbre, "T_")
        forms += [("set", "tn", ("enc_rms", "tx", self.timbre_norm, w)),
                  ("set", "p_timbre", ("copy", "tn0")),
                  ("set", "p_text", ("enc_linear", self.text_proj, "text")),
                  ("set", "enc", ("enc_linear", self.cond_emb, "packed"))]
        for i, Ly in enumerate(self.layers):
            forms += [("set", "ek", ("enc_linear", Ly["ck"], "enc")),
                      ("set", "ck%d" % i, ("enc_rms", "ek", Ly["ckn"], hd)),
                      ("set", "cv%d" % i, ("enc_linear", Ly["cv"], "enc"))]
        c.compile(("seq", *forms))
        return c.p.finish()

    def step_program(self, F, Te, gpu=False):
        """A step's program of F frames (F / 2 tokens) and Te condition rows:
        inputs "xin" (F / 2, 2 192: the frames' context and latents, two a
        row), "mods" (the vectors of mods()); output "vel" (F / 2, 2 64)."""
        w, H, KV, hd = self.width, self.H, self.KV, self.hd
        n = F // 2
        c = _EncCompiler(self.eps, pack=not gpu, rows=n, kmax=self.inter)
        env = c.env
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        env.update(self.cache(Te))
        env.update(xin=z(n, self.proj_in.shape[1]), mods=z(6 * len(self.layers) + 2, w), x=z(n, w), h=z(n, w),
                   q=z(n, H * hd), k=z(n, KV * hd), v=z(n, KV * hd), qn=z(n, H * hd), kn=z(n, KV * hd),
                   o=z(n, H * hd), t=z(n, w), g=z(n, self.inter), u=z(n, self.inter),
                   vel=z(n, self.proj_out.shape[0]), pos=np.arange(n, dtype=np.int32),
                   clo=np.zeros(n, np.int32), chi=np.full(n, Te, np.int32))
        for kind in ("sliding_attention", "full_attention"):
            env[kind + "_lo"], env[kind + "_hi"] = self._ranges(n, kind)
        mods = env["mods"]
        forms = [("set", "x", ("enc_linear", self.proj_in, "xin"))]
        for i, Ly in enumerate(self.layers):
            m = ["m%d_%d" % (i, j) for j in range(6)]
            for j in range(6):
                env[m[j]] = mods[6 * i + j]
            kind = self.types[i]
            forms += [
                ("set", "h", ("enc_rms", "x", m[0], w)),
                ("enc_bias", "h", m[1]),
                ("set", "q", ("enc_linear", Ly["q"], "h")),
                ("set", "k", ("enc_linear", Ly["k"], "h")),
                ("set", "v", ("enc_linear", Ly["v"], "h")),
                ("set", "qn", ("enc_rms", "q", Ly["qn"], hd)),
                ("set", "kn", ("enc_rms", "k", Ly["kn"], hd)),
                ("enc_rope1d", "qn", "pos", self.inv, H, hd),
                ("enc_rope1d", "kn", "pos", self.inv, KV, hd),
                ("set", "o", ("enc_attn_range", "qn", "kn", "v", kind + "_lo", kind + "_hi", H, KV, hd)),
                ("set", "t", ("enc_linear", Ly["o"], "o")),
                ("set", "t", ("enc_mul_vec", "t", m[2])),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", Ly["n_c"], w)),
                ("set", "q", ("enc_linear", Ly["cq"], "h")),
                ("set", "qn", ("enc_rms", "q", Ly["cqn"], hd)),
                ("set", "o", ("enc_attn_kv", "qn", "ck%d" % i, "cv%d" % i, "clo", "chi", H, hd, Te, KV)),
                ("set", "t", ("enc_linear", Ly["co"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", m[3], w)),
                ("enc_bias", "h", m[4]),
                ("set", "g", ("enc_linear", Ly["gate"], "h")),
                ("set", "u", ("enc_linear", Ly["up"], "h")),
                ("set", "u", ("enc_silu_mul", "g", "u")),
                ("set", "t", ("enc_linear", Ly["down"], "u")),
                ("set", "t", ("enc_mul_vec", "t", m[5])),
                ("enc_add", "x", "t"),
            ]
        env["mo_w"], env["mo_b"] = mods[-2], mods[-1]
        forms += [("set", "h", ("enc_rms", "x", "mo_w", w)),
                  ("enc_bias", "h", "mo_b"),
                  ("set", "vel", ("enc_linear", self.proj_out, "h"))]
        c.compile(("seq", *forms))
        return c.p.finish()

    def _run(self, key, inputs, outputs):
        prog, g = self.runner.get(key, self.gpu)
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
            for name in outputs:
                g.download(name)
        else:
            prog.run()
        return [prog.names[name].copy() for name in outputs]

    # ---- the steps -----------------------------------------------------------

    def condition(self, text, lyric, refer=None):
        """Run the conditions' program (text (Tt, 1024), lyric (Tl, 1024),
        refer (750, 64): default the silence latent's); return the packed
        conditions (Te, width) after the condition embedder (self.packed:
        before it)."""
        refer = self.silence[:750] if refer is None else refer
        enc, self.packed = self._run(("cond", len(text), len(lyric)), dict(text=text, lyric=lyric, refer=refer),
                                     ["enc", "packed"])
        return enc

    def velocity(self, xt, t, context, Te):
        """The DiT's output (F, 64) of latents xt (F, 64) at t, the context
        latents (F, 128), after condition() of Te rows."""
        F = xt.shape[0]
        assert F % 2 == 0
        xin = np.concatenate([context, xt], 1).reshape(F // 2, -1)
        vel, = self._run(("step", F, Te), dict(xin=xin, mods=self.mods(t)), ["vel"])
        return vel.reshape(F, -1)
