"""CLIP's text model (transformers' CLIPTextModel) as an encoder program: the
last hidden state (after the final LayerNorm) of n prompts of 77 tokens.
SD 1.5: CLIP ViT-L/14's (12 layers of 768, 12 heads, quick_gelu); SD 2.0:
OpenCLIP ViT-H/14's less its last layer (23 layers of 1024, 16 heads,
gelu), as diffusers' checkpoint holds it.

Ported from transformers (src/transformers/models/clip/modeling_clip.py; Apache
License 2.0; Copyright 2021 The OpenAI Team Authors and The HuggingFace Team).
See THIRD_PARTY_NOTICES.md.

    the token's row of token_embedding plus the position's of
        position_embedding (on the host, float32)
    a layer: LayerNorm, q k v (with bias), q times hd^-1/2, the causal
        attention (row i over the keys [0, i + 1): no padding mask, as the
        pipeline gives none), out_proj; + x; LayerNorm, fc1, the activation,
        fc2; + x
    final_layer_norm

quick_gelu x sigmoid(1.702 x) is SiLU(1.702 x) / 1.702 (ENC_MUL_VEC, ENC_SILU,
ENC_MUL_VEC: a few roundings of float32 from torch's). The matrices float32
as stored (ENC_LINEAR), on the CPU and the GPU.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.qwen3.dense import lin16
from reclank.runtime.threads import team_threads

QUICK = np.float32(1.702)


class CLIPText:
    """The text model of a text_encoder directory (config.json and
    model.safetensors). hidden(ids) gives the last hidden state."""

    def __init__(self, path, gpu=False, cpu_threads=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg["model_type"] == "clip_text_model" and cfg["hidden_act"] in ("quick_gelu", "gelu")
        self.W = W = SafeTensors(os.path.join(path, "model.safetensors"))
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.width, self.heads = cfg["hidden_size"], cfg["num_attention_heads"]
        self.hd = self.width // self.heads
        self.inter = cfg["intermediate_size"]
        self.eps = float(cfg["layer_norm_eps"])
        self.quick = cfg["hidden_act"] == "quick_gelu"
        self.length = cfg["max_position_embeddings"]
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        p = "text_model."
        self.pos = f(p + "embeddings.position_embedding.weight")
        self.layers = []
        for i in range(cfg["num_hidden_layers"]):
            r = p + "encoder.layers.%d." % i
            L = lambda n: lin16(W, r + n, gpu)  # noqa: E731
            self.layers.append(dict(
                ln1=(f(r + "layer_norm1.weight"), f(r + "layer_norm1.bias")),
                ln2=(f(r + "layer_norm2.weight"), f(r + "layer_norm2.bias")),
                q=L("self_attn.q_proj"), k=L("self_attn.k_proj"), v=L("self_attn.v_proj"),
                o=L("self_attn.out_proj"), fc1=L("mlp.fc1"), fc2=L("mlp.fc2")))
        self.final = (f(p + "final_layer_norm.weight"), f(p + "final_layer_norm.bias"))
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)

    def _build(self, n, gpu):
        p = self.program(n, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def embed(self, ids):
        """The input rows (n 77, width) of ids (n, 77): token plus position."""
        ids = np.asarray(ids).reshape(-1, self.length)
        name = "text_model.embeddings.token_embedding.weight"
        u, inv = np.unique(ids, return_inverse=True)
        rows = np.stack([self.W.get_row(name, int(t)) for t in u])[inv.reshape(-1)]
        return (rows.reshape(ids.shape[0], self.length, -1) + self.pos[None]).astype(np.float32).reshape(
            -1, self.width)

    def program(self, n, gpu=False):
        """The program of n prompts: inputs "x" (n 77, width), "lo", "hi" (n 77,
        int32: a row's keys); output "out" (n 77, width)."""
        w, H, hd = self.width, self.heads, self.hd
        R = n * self.length
        c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=self.inter)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(x=z(R, w), h=z(R, w), q=z(R, w), k=z(R, w), v=z(R, w), o=z(R, w), t=z(R, w),
                     g=z(R, self.inter), out=z(R, w), lo=np.zeros(R, np.int32), hi=np.zeros(R, np.int32),
                     _scale=np.full(w, hd ** -0.5, np.float32), _quick=np.full(self.inter, QUICK, np.float32),
                     _iquick=np.full(self.inter, np.float32(1) / QUICK, np.float32))
        forms = []
        for L in self.layers:
            forms += [("set", "h", ("enc_lnorm", "x", L["ln1"][0], L["ln1"][1], w)),
                      ("set", "q", ("enc_linear", L["q"], "h")),
                      ("set", "q", ("enc_mul_vec", "q", "_scale")),
                      ("set", "k", ("enc_linear", L["k"], "h")),
                      ("set", "v", ("enc_linear", L["v"], "h")),
                      ("set", "o", ("enc_attn_range_x2", "q", "k", "v", "lo", "hi", H, hd)),
                      ("set", "t", ("enc_linear", L["o"], "o")),
                      ("enc_add", "x", "t"),
                      ("set", "h", ("enc_lnorm", "x", L["ln2"][0], L["ln2"][1], w)),
                      ("set", "g", ("enc_linear", L["fc1"], "h"))]
            if self.quick:
                forms += [("set", "g", ("enc_mul_vec", "g", "_quick")),
                          ("set", "g", ("enc_silu", "g")),
                          ("set", "g", ("enc_mul_vec", "g", "_iquick"))]
            else:
                forms.append(("set", "g", ("enc_gelu", "g", True)))
            forms += [("set", "t", ("enc_linear", L["fc2"], "g")),
                      ("enc_add", "x", "t")]
        forms.append(("set", "out", ("enc_lnorm", "x", self.final[0], self.final[1], w)))
        c.compile(("seq", *forms))
        return c.p.finish()

    def hidden(self, ids):
        """The last hidden state (n, 77, width) of n prompts' ids (n, 77)."""
        ids = np.asarray(ids).reshape(-1, self.length)
        n, T = ids.shape
        i = np.arange(T)
        lo = np.repeat(np.arange(n) * T, T).astype(np.int32)
        hi = (lo + np.tile(i + 1, n)).astype(np.int32)
        out = self.runner.run(n, self.gpu, dict(x=self.embed(ids), lo=lo, hi=hi), "out")
        return out.reshape(n, T, -1)
