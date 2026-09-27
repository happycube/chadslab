"""The Qwen3.5 / Qwen3.6 mixture-of-experts text model (qwen3_5_moe), in NumPy.

QWEN_PLAN.md, phase 1. This module reads the MLX files of the model
(mlx-community, affine 4-bit and 8-bit weights) and runs the text model in
float32 NumPy. It is the reference of this runtime for this model; the C and
GPU paths come later and compare with it. scripts/check_qwen.py compares it
with transformers.

The model has two kinds of layers (config.layer_types):

- linear_attention: a Gated DeltaNet. A causal convolution of 4 on q, k, v,
  then a recurrent state of 128 x 128 for each value head. The state does
  not grow with the context.
- full_attention: attention with 16 query heads and 2 key and value heads of
  256, RoPE on the first 64 values of each head, and a sigmoid gate on the
  output.

Each layer then has 256 experts (8 for each token) and a shared expert with
a sigmoid gate.

The weights are in the MLX affine format (np_gemma/mlx_affine.py).
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import cops, mlx_affine, ops
from .mlx_affine import QMat
from .st import SafeTensors

PREFIX = "language_model.model."


class QwenConfig:
    """The text config of a qwen3_5_moe model (config.json, text_config)."""

    def __init__(self, path):
        c = json.load(open(os.path.join(path, "config.json")))
        t = c.get("text_config", c)
        self.raw = t
        self.hidden_size = t["hidden_size"]
        self.num_hidden_layers = t["num_hidden_layers"]
        self.layer_types = t["layer_types"]
        self.rms_norm_eps = t["rms_norm_eps"]
        self.vocab_size = t["vocab_size"]
        # full attention
        self.num_heads = t["num_attention_heads"]
        self.num_kv_heads = t["num_key_value_heads"]
        self.head_dim = t["head_dim"]
        rp = t.get("rope_parameters", {})
        self.rope_theta = rp.get("rope_theta", t.get("rope_theta", 10000.0))
        self.rotary_dim = int(self.head_dim * rp.get("partial_rotary_factor",
                                                      t.get("partial_rotary_factor", 1.0)))
        # linear attention
        self.lin_k_heads = t["linear_num_key_heads"]
        self.lin_v_heads = t["linear_num_value_heads"]
        self.lin_k_dim = t["linear_key_head_dim"]
        self.lin_v_dim = t["linear_value_head_dim"]
        self.conv_kernel = t["linear_conv_kernel_dim"]
        # experts
        self.num_experts = t["num_experts"]
        self.top_k = t["num_experts_per_tok"]
        self.moe_inter = t["moe_intermediate_size"]
        self.shared_inter = t["shared_expert_intermediate_size"]
        self.eos_token_ids = c.get("eos_token_id", t.get("eos_token_id"))

    @property
    def lin_key_dim(self):
        return self.lin_k_heads * self.lin_k_dim

    @property
    def lin_value_dim(self):
        return self.lin_v_heads * self.lin_v_dim

    @property
    def conv_dim(self):
        return 2 * self.lin_key_dim + self.lin_value_dim


def rms_norm(x, w, eps, offset=0.0):
    """RMSNorm with the weight (offset + w). transformers keeps the norms of
    Qwen3.5 as w - 1 and adds 1; the MLX files keep the full weight (the
    converter of mlx-lm adds the 1), so the default offset is 0. The gated
    norm of the linear layers has no offset in both."""
    x = x.astype(np.float32)
    s = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * s * (offset + w)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


class QwenCache:
    """The state of a sequence: keys and values of the full layers, and the
    convolution inputs and the recurrent state of the linear layers."""

    def __init__(self, cfg, max_len=4096):
        self.cfg = cfg
        self.n = 0
        self.kv = {}
        self.conv = {}
        self.state = {}
        for i, t in enumerate(cfg.layer_types):
            if t == "full_attention":
                shape = (cfg.num_kv_heads, max_len, cfg.head_dim)
                self.kv[i] = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
            else:
                self.conv[i] = np.zeros((cfg.conv_kernel - 1, cfg.conv_dim), np.float32)
                self.state[i] = np.zeros((cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim),
                                         np.float32)


class Qwen:
    """The text model. Weights stay quantized in the memory maps; the NumPy
    path dequantizes each matrix when it uses it.

        cfg = QwenConfig(path); model = Qwen(path, cfg)
        cache = QwenCache(cfg)
        h = model.forward(ids, cache)              # the hidden states after the final norm
        logits = model.logits(h[-1:])
    """

    def __init__(self, path, cfg=None, layers=None):
        self.path = path
        self.cfg = cfg or QwenConfig(path)
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.files = {f: SafeTensors(os.path.join(path, f)) for f in sorted(set(idx.values()))}
        self.where = idx
        # layers limits the model to its first layers (a test against a
        # reference with the same count).
        self.n_layers = layers or self.cfg.num_hidden_layers
        self._deq = {}

    def raw(self, name, dtype=None):
        return self.files[self.where[name]].get(name, dtype=dtype)

    def t(self, name):
        """A float32 tensor (a norm, A_log, the convolution)."""
        return self.files[self.where[PREFIX + name]].get(PREFIX + name)

    def mat(self, name, full=None):
        """A quantized matrix (name without .weight)."""
        n = full or PREFIX + name
        f = self.files[self.where[n + ".weight"]]
        # The scales and biases stay raw bfloat16 (uint16 views, no copy).
        return QMat(f.get(n + ".weight", dtype=None), f.get_bf16(n + ".scales"),
                    f.get_bf16(n + ".biases"))

    def W(self, name):
        """A dequantized dense matrix, kept after the first use."""
        w = self._deq.get(name)
        if w is None:
            w = self._deq[name] = self.mat(name).dequant()
        return w

    # ---- the layers ----------------------------------------------------------

    def embed(self, ids):
        m = self.mat("embed_tokens")
        return m.dequant(rows=np.asarray(ids, dtype=np.int64))

    def rope(self, positions):
        cfg = self.cfg
        d = cfg.rotary_dim
        inv = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        f = np.outer(np.asarray(positions, dtype=np.float64), inv)
        f = np.concatenate([f, f], axis=-1)
        return np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)

    def full_attention(self, i, h, cache, pos):
        cfg = self.cfg
        p = "layers.%d.self_attn." % i
        t = h.shape[0]
        nq, nk, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        qg = (h @ self.W(p + "q_proj").T).reshape(t, nq, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(t, nq * hd)
        k = (h @ self.W(p + "k_proj").T).reshape(t, nk, hd)
        v = (h @ self.W(p + "v_proj").T).reshape(t, nk, hd)
        q = rms_norm(q, self.t(p + "q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm(k, self.t(p + "k_norm.weight"), cfg.rms_norm_eps)
        cos, sin = self.rope(np.arange(pos, pos + t))
        d = cfg.rotary_dim

        def rot(x):
            xr = x[..., :d]
            half = d // 2
            rh = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
            return np.concatenate([xr * cos[:, None] + rh * sin[:, None], x[..., d:]], axis=-1)

        q, k = rot(q), rot(k)
        K, V = cache.kv[i]
        K[:, pos:pos + t] = k.transpose(1, 0, 2)
        V[:, pos:pos + t] = v.transpose(1, 0, 2)
        n = pos + t
        rep = nq // nk
        out = np.empty((t, nq, hd), np.float32)
        scale = hd ** -0.5
        for hq in range(nq):
            kh = hq // rep
            s = (q[:, hq] @ K[kh, :n].T) * scale                    # (t, n)
            mask = np.arange(n)[None, :] > (pos + np.arange(t))[:, None]
            s[mask] = -np.inf
            s -= s.max(axis=-1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(axis=-1, keepdims=True)
            out[:, hq] = w @ V[kh, :n]
        o = out.reshape(t, nq * hd) * sigmoid(gate)
        return o @ self.W(p + "o_proj").T

    def linear_attention(self, i, h, cache, pos):
        cfg = self.cfg
        p = "layers.%d.linear_attn." % i
        t = h.shape[0]
        qkv = h @ self.W(p + "in_proj_qkv").T                      # (t, conv_dim)
        z = (h @ self.W(p + "in_proj_z").T).reshape(t, cfg.lin_v_heads, cfg.lin_v_dim)
        b = h @ self.W(p + "in_proj_b").T                          # (t, v heads)
        a = h @ self.W(p + "in_proj_a").T
        # The causal convolution of kernel 4, with the last 3 inputs of the
        # cache before the new rows.
        w = self.t(p + "conv1d.weight").reshape(cfg.conv_dim, cfg.conv_kernel)
        prev = cache.conv[i]
        xs = np.concatenate([prev, qkv], axis=0)                   # (3 + t, conv_dim)
        kk = cfg.conv_kernel
        conv = np.zeros_like(qkv)
        for j in range(kk):
            conv += xs[j:j + t] * w[:, j]
        conv = silu(conv)
        cache.conv[i] = xs[-(kk - 1):].copy()
        kd, vd = cfg.lin_key_dim, cfg.lin_value_dim
        q = conv[:, :kd].reshape(t, cfg.lin_k_heads, cfg.lin_k_dim)
        k = conv[:, kd:2 * kd].reshape(t, cfg.lin_k_heads, cfg.lin_k_dim)
        v = conv[:, 2 * kd:].reshape(t, cfg.lin_v_heads, cfg.lin_v_dim)
        beta = sigmoid(b)
        g = -np.exp(self.t(p + "A_log")) * softplus(a + self.t(p + "dt_bias"))
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        q = q / np.sqrt(cfg.lin_k_dim)
        rep = cfg.lin_v_heads // cfg.lin_k_heads
        S = cache.state[i]
        o = np.empty((t, cfg.lin_v_heads, cfg.lin_v_dim), np.float32)
        for j in range(t):
            for hv in range(cfg.lin_v_heads):
                hk = hv // rep
                Sh = S[hv]
                Sh *= np.exp(g[j, hv])
                kv_mem = k[j, hk] @ Sh                             # S^T k
                delta = (v[j, hv] - kv_mem) * beta[j, hv]
                Sh += np.outer(k[j, hk], delta)
                o[j, hv] = q[j, hk] @ Sh
        o = rms_norm(o, self.t(p + "norm.weight"), cfg.rms_norm_eps) * silu(z)
        return o.reshape(t, vd) @ self.W(p + "out_proj").T

    def moe(self, i, h):
        cfg = self.cfg
        p = "layers.%d.mlp." % i
        logits = h @ self.W(p + "gate").T                          # (t, experts)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
        top = np.argsort(-probs, axis=-1, kind="stable")[:, :cfg.top_k]
        val = np.take_along_axis(probs, top, axis=-1)
        val = val / val.sum(axis=-1, keepdims=True)
        gm, um, dm = (self.mat(p + "switch_mlp." + n) for n in ("gate_proj", "up_proj", "down_proj"))
        out = np.zeros_like(h)
        for x in np.unique(top):
            rows, slots = np.nonzero(top == x)
            hx = h[rows]
            act = silu(hx @ gm.dequant(expert=x).T) * (hx @ um.dequant(expert=x).T)
            out[rows] += (act @ dm.dequant(expert=x).T) * val[rows, slots][:, None]
        sp = p + "shared_expert."
        sh = silu(h @ self.W(sp + "gate_proj").T) * (h @ self.W(sp + "up_proj").T)
        sh = sh @ self.W(sp + "down_proj").T
        sg = sigmoid(h @ self.W(p + "shared_expert_gate").T)       # (t, 1)
        return out + sg * sh

    def layer(self, i, x, cache, pos):
        cfg = self.cfg
        p = "layers.%d." % i
        h = rms_norm(x, self.t(p + "input_layernorm.weight"), cfg.rms_norm_eps)
        if cfg.layer_types[i] == "full_attention":
            x = x + self.full_attention(i, h, cache, pos)
        else:
            x = x + self.linear_attention(i, h, cache, pos)
        h = rms_norm(x, self.t(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps)
        return x + self.moe(i, h)

    def forward(self, ids, cache, start_pos=0, hook=None):
        """Run tokens from start_pos. Return the hidden states after the final
        norm, (t, hidden). hook(name, array) gets the output of each layer."""
        x = self.embed(ids)
        for i in range(self.n_layers):
            x = self.layer(i, x, cache, start_pos)
            if hook is not None:
                hook("layer.%d" % i, x)
        cache.n = start_pos + len(ids)
        return rms_norm(x, self.t("norm.weight"), self.cfg.rms_norm_eps)

    def logits(self, h, chunk=16384):
        """The head on rows of hidden states, (t, vocabulary)."""
        m = self.mat(None, full="language_model.lm_head")
        out = np.empty((h.shape[0], m.rows), np.float32)
        for r0 in range(0, m.rows, chunk):
            rows = np.arange(r0, min(m.rows, r0 + chunk))
            out[:, rows] = h @ m.dequant(rows=rows).T
        return out


# ---- the CPU path with the C kernels (QWEN_PLAN.md, phase 2) ----------------

class QwenCPU(Qwen):
    """The model with the C kernels of cops: the MLX affine products
    (csrc/mlx_affine.c), the Gated DeltaNet (csrc/deltanet.c), and the shared
    ops (rms_norm, the attention of one token). The weights stay in the MLX
    blocks. The products quantize x to int8 for each group of 64 (as the CPU
    path of llama.cpp does), so the result is close to the NumPy model, not
    the same. scripts/check_qwen_cpu.py compares them."""

    def __init__(self, path, cfg=None, layers=None):
        super().__init__(path, cfg, layers)
        self._m = {}
        self._f = {}

    def M(self, name, full=None):
        m = self._m.get(name)
        if m is None:
            m = self._m[name] = self.mat(name, full)
        return m

    def F(self, name, shape=None):
        a = self._f.get(name)
        if a is None:
            a = np.ascontiguousarray(self.t(name), dtype=np.float32)
            a = self._f[name] = a.reshape(shape) if shape is not None else a
        return a

    def lin(self, name, qx, full=None):
        return mlx_affine.linear(self.M(name, full), qx)

    def norm(self, x, name):
        return ops.rms_norm(x, self.F(name), self.cfg.rms_norm_eps)

    def layer(self, i, x, cache, pos):
        p = "layers.%d." % i
        h = self.norm(x, p + "input_layernorm.weight")
        if self.cfg.layer_types[i] == "full_attention":
            x = x + self.full_attention(i, h, cache, pos)
        else:
            x = x + self.linear_attention(i, h, cache, pos)
        return x + self.moe(i, self.norm(x, p + "post_attention_layernorm.weight"))

    def forward(self, ids, cache, start_pos=0, hook=None):
        x = self.embed(ids)
        for i in range(self.n_layers):
            x = self.layer(i, x, cache, start_pos)
            if hook is not None:
                hook("layer.%d" % i, x)
        cache.n = start_pos + len(ids)
        return self.norm(x, "norm.weight")

    def linear_attention(self, i, h, cache, pos):
        cfg = self.cfg
        p = "layers.%d.linear_attn." % i
        t = h.shape[0]
        qx = mlx_affine.QX(h)
        qkv = self.lin(p + "in_proj_qkv", qx)
        z = self.lin(p + "in_proj_z", qx)
        b = self.lin(p + "in_proj_b", qx)
        a = self.lin(p + "in_proj_a", qx)
        out = np.empty((t, cfg.lin_value_dim), np.float32)
        cops.gdn_step(qkv, cache.conv[i], self.F(p + "conv1d.weight", (cfg.conv_dim, cfg.conv_kernel)),
                      z, a, b, self.F(p + "A_log"), self.F(p + "dt_bias"), self.F(p + "norm.weight"),
                      cache.state[i], out, np.empty((t, cfg.conv_dim), np.float32),
                      cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim,
                      cfg.rms_norm_eps)
        return self.lin(p + "out_proj", mlx_affine.QX(out))

    def full_attention(self, i, h, cache, pos):
        cfg = self.cfg
        p = "layers.%d.self_attn." % i
        t = h.shape[0]
        nq, nk, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        qx = mlx_affine.QX(h)
        qg = self.lin(p + "q_proj", qx).reshape(t, nq, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(t, nq * hd)
        k = self.lin(p + "k_proj", qx).reshape(t, nk, hd)
        v = self.lin(p + "v_proj", qx).reshape(t, nk, hd)
        q = ops.rms_norm(q, self.F(p + "q_norm.weight"), cfg.rms_norm_eps)
        k = ops.rms_norm(k, self.F(p + "k_norm.weight"), cfg.rms_norm_eps)
        cos, sin = self.rope(np.arange(pos, pos + t))
        d = cfg.rotary_dim
        half = d // 2

        def rot(x):
            xr = x[..., :d]
            rh = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
            return np.concatenate([xr * cos[:, None] + rh * sin[:, None], x[..., d:]], axis=-1)

        q, k = rot(q), rot(k)
        K, V = cache.kv[i]
        K[:, pos:pos + t] = k.transpose(1, 0, 2)
        V[:, pos:pos + t] = v.transpose(1, 0, 2)
        # The attention kernel of cops has no scale (Gemma uses 1), so the
        # query takes the scale of Qwen.
        q = np.ascontiguousarray(q * np.float32(hd ** -0.5))
        o = np.empty((t, nq, hd), np.float32)
        for j in range(t):
            n = pos + j + 1
            o[j] = ops.attn_decode_f32(q[j], K[:, :n], V[:, :n], pos + j)
        o = o.reshape(t, nq * hd) * sigmoid(gate)
        return self.lin(p + "o_proj", mlx_affine.QX(o))

    def moe(self, i, h):
        cfg = self.cfg
        p = "layers.%d.mlp." % i
        t = h.shape[0]
        qx = mlx_affine.QX(h)
        logits = self.lin(p + "gate", qx)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
        top = np.argsort(-probs, axis=-1, kind="stable")[:, :cfg.top_k].astype(np.int32)
        val = np.take_along_axis(probs, top, axis=-1)
        val = (val / val.sum(axis=-1, keepdims=True)).astype(np.float32)
        sgate = sigmoid(self.lin(p + "shared_expert_gate", qx))[:, 0]
        experts = [self.M(p + "switch_mlp." + n).c() for n in ("gate_proj", "up_proj", "down_proj")]
        shared = [self.M(p + "shared_expert." + n).c() for n in ("gate_proj", "up_proj", "down_proj")]
        k, inner, hid = cfg.top_k, cfg.moe_inter, cfg.hidden_size
        ne = k + 1
        scratch = (np.empty(ne * 2 * inner, np.float32), np.empty(ne * inner, np.int8),
                   np.empty(ne * inner // 64, np.float32), np.empty(ne * inner // 64, np.float32),
                   np.empty(ne * hid, np.float32))
        out = np.empty((t, hid), np.float32)
        q4, q8 = qx.get(4), qx.get(8)
        for j in range(t):
            cops.ma_moe_step(q4[j], q8[j], qx.xs[j], qx.xsum[j], np.ascontiguousarray(top[j]),
                             np.ascontiguousarray(val[j]), *experts, shared, sgate[j], hid,
                             inner, scratch, out[j])
        return out

    def embed(self, ids):
        return self.M("embed_tokens").dequant(rows=np.asarray(ids, dtype=np.int64))

    def logits(self, h, chunk=None):
        return mlx_affine.linear(self.M("lm_head", "language_model.lm_head"), mlx_affine.QX(h))
