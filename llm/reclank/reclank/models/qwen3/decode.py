"""A Qwen3 dense model as step programs with a KV cache (as Whisper's decoder,
models/whisper/decoder.py): the backbone of the TTS models (Higgs TTS 3) and
of ACE-Step's planner.

Ported from transformers (src/transformers/models/qwen3/modeling_qwen3.py;
Apache License 2.0; Copyright 2025 The Qwen team, Alibaba Group and the
HuggingFace Inc. team). See THIRD_PARTY_NOTICES.md.

A step program takes r rows of each of n sequences (a prompt: r its tokens,
padded; a step: r = 1). The host gives each row's input embedding (the text
embedding's row, or a model's own: Higgs's sum of the codebooks' rows), its
row in the caches ("kvi": sequence max_tokens + position; -1: none, a row of
padding), its keys (the sequence's rows of the cache up to its own,
included), and the cos and sin of its position (the host's table, so a
model trained with RoPE rounded to bfloat16 gets its rounding). A layer as
qwen3.dense: RMS norm, q k v, the RMS norms of the heads, RoPE (NEOX halves,
ENC_ROPE_TAB), the keys and values to the cache (ENC_SCATTER), the attention
over the cache (ENC_ATTN_KV with GQA), the output, + x; RMS norm, SwiGLU, + x.
Then the last norm and the heads (linears of the normed rows: logits).

The matrices: float32 on the CPU, the checkpoint's bfloat16 rows on the GPU
(KQ_BF16); the caches float32.
"""
from __future__ import annotations

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Runner
from reclank.models.qwen3.dense import qwen3_layers, rope_inv
from reclank.runtime.threads import team_threads


def bf16_round(x):
    """x (float32) rounded to bfloat16 (to nearest, even), as float32."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


class Qwen3Decoder:
    """The layers of a Qwen3 dense checkpoint (W: SafeTensors or Shards;
    cfg: its text config) as step programs. heads: {name: linear} of the
    normed rows (e.g. the logits)."""

    def __init__(self, W, cfg, layers_prefix, norm_name, heads, gpu=False, cpu_threads=None, max_tokens=4096,
                 rope_bfloat16=False):
        self.W, self.gpu, self.cpu_threads = W, bool(gpu), cpu_threads
        self.width = cfg["hidden_size"]
        self.H, self.KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim") or self.width // self.H
        self.eps = float(cfg["rms_norm_eps"])
        theta = (cfg.get("rope_parameters") or {}).get("rope_theta") or cfg.get("rope_theta") or 1e6
        self.inv = rope_inv(self.hd, theta)
        self.rope_bf16 = rope_bfloat16
        self.layers = qwen3_layers(W, layers_prefix, cfg["num_hidden_layers"], self.hd, gpu)
        self.norm = np.ascontiguousarray(W.get(norm_name), dtype=np.float32)
        self.heads = dict(heads)
        self.inter = self.layers[0]["gate"].shape[0]
        self.max_tokens = max_tokens
        self.caches = {}
        self.runner = _Runner(self._build, keep=6, keep_gpu=3)

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def cache(self, n):
        """The caches of n sequences: "sk<l>", "sv<l>" (n max_tokens, kv hd)."""
        c = self.caches.get(n)
        if c is None:
            c = {}
            for i in range(len(self.layers)):
                for name in ("sk", "sv"):
                    c["%s%d" % (name, i)] = np.zeros((n * self.max_tokens, self.KV * self.hd), np.float32)
            self.caches[n] = c
        return c

    def rope(self, pos):
        """cos and sin (rows, hd / 2) of positions (transformers' float32
        angles; rounded to bfloat16 for a model trained so)."""
        ang = (np.asarray(pos, np.float32)[:, None] * self.inv[None, :]).astype(np.float32)
        c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        if self.rope_bf16:
            c, s = bf16_round(c), bf16_round(s)
        return c, s

    def program(self, n, r, gpu=False):
        """The program of r rows of each of n sequences: inputs "x" (n r,
        width), "kvi", "slo", "shi" (n r) int32, "cos", "sin" (n r, hd / 2);
        outputs "hn" (the normed rows) and each head's name."""
        w, H, KV, hd = self.width, self.H, self.KV, self.hd
        R = n * r
        c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=self.inter)
        c.lnorm_rows = True
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        zi = lambda *shape: np.zeros(shape, np.int32)  # noqa: E731
        c.env.update(self.cache(n))
        c.env.update(x=z(R, w), h=z(R, w), hn=z(R, w), q=z(R, H * hd), k=z(R, KV * hd), v=z(R, KV * hd),
                     qn=z(R, H * hd), kn=z(R, KV * hd), o=z(R, H * hd), t=z(R, w),
                     g=z(R, self.inter), u=z(R, self.inter), kvi=zi(R), slo=zi(R), shi=zi(R),
                     cos=z(R, hd // 2), sin=z(R, hd // 2))
        for name, lin in self.heads.items():
            c.env[name] = z(R, lin.shape[0])
        forms = []
        for i, L in enumerate(self.layers):
            sk, sv = "sk%d" % i, "sv%d" % i
            forms += [
                ("set", "h", ("enc_rms", "x", L["ln_a"], w)),
                ("set", "q", ("enc_linear", L["q"], "h")),
                ("set", "k", ("enc_linear", L["k"], "h")),
                ("set", "v", ("enc_linear", L["v"], "h")),
                ("set", "qn", ("enc_rms", "q", L["qn"], hd)),
                ("set", "kn", ("enc_rms", "k", L["kn"], hd)),
                ("enc_rope_tab", "qn", "cos", "sin", H, hd, True),
                ("enc_rope_tab", "kn", "cos", "sin", KV, hd, True),
                ("set", sk, ("enc_scatter", "kn", "kvi")),
                ("set", sv, ("enc_scatter", "v", "kvi")),
                ("set", "o", ("enc_attn_kv", "qn", sk, sv, "slo", "shi", H, hd, self.max_tokens, KV)),
                ("set", "t", ("enc_linear", L["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", L["ln_m"], w)),
                ("set", "g", ("enc_linear", L["gate"], "h")),
                ("set", "u", ("enc_linear", L["up"], "h")),
                ("set", "u", ("enc_silu_mul", "g", "u")),
                ("set", "t", ("enc_linear", L["down"], "u")),
                ("enc_add", "x", "t"),
            ]
        forms.append(("set", "hn", ("enc_rms", "x", self.norm, w)))
        for name, lin in self.heads.items():
            forms.append(("set", name, ("enc_linear", lin, "hn")))
        c.compile(("seq", *forms))
        return c.p.finish()

    def _caches(self):
        """The cache arrays of one sequence (n = 1 programs), and on the GPU
        the device Buffer of each (a step program loaded first, so that the
        caches are in its mirror)."""
        c = self.cache(1)
        if not self.gpu:
            return c, None
        self.runner.get((1, 1), True)
        return c, {k: self.runner.mirror.buffer_of(a) for k, a in c.items()}

    def save_prefix(self, rows):
        """A copy (name: (rows, kv hd)) of the first rows of the cache (n =
        1): a prompt's prefix to reuse (load_prefix)."""
        c, bufs = self._caches()
        out = {}
        for k, a in c.items():
            dst = np.empty((rows, a.shape[1]), np.float32)
            if bufs is not None:
                bufs[k].download_at(dst, 0)
            else:
                dst[:] = a[:rows]
            out[k] = dst
        return out

    def load_prefix(self, snap):
        """Put a save_prefix copy back at the start of the cache (n = 1): its
        rows then need no prefill (the next rows' positions follow them)."""
        c, bufs = self._caches()
        for k, a in c.items():
            if bufs is not None:
                bufs[k].upload_at(snap[k], 0)
            else:
                a[:len(snap[k])] = snap[k]

    def step(self, x, pos, n=1, outputs=("hn",)):
        """Run r rows of each of n sequences: x (n r, width) their input
        embeddings, pos (n, r) their positions (-1: a row of padding).
        Return the outputs (each (n r, ...)). A sequence's earlier rows must
        be in the cache (its earlier steps)."""
        pos = np.asarray(pos).reshape(n, -1)
        r = pos.shape[1]
        seq = np.repeat(np.arange(n), r)
        p = pos.reshape(-1)
        real = p >= 0
        assert p.max() < self.max_tokens
        kvi = np.where(real, seq * self.max_tokens + p, -1).astype(np.int32)
        slo = (seq * self.max_tokens).astype(np.int32)
        shi = np.where(real, kvi + 1, slo + 1).astype(np.int32)
        cs, sn = self.rope(np.maximum(p, 0))
        inputs = dict(x=x, kvi=kvi, slo=slo, shi=shi, cos=cs, sin=sn)
        prog, g = self.runner.get((n, r), self.gpu)
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
