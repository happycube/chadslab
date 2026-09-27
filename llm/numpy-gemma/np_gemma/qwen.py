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
        # The order of the value heads of the linear layers. The MLX files
        # (and transformers) keep the heads of one key head together:
        # value head h reads key head h // (v_heads / k_heads). The llama.cpp
        # converter puts them in tiled order: head h reads key head
        # h % k_heads. All the tensors of a head move together, so only
        # this rule changes.
        self.v_tiled = False

    @classmethod
    def from_gguf(cls, g):
        """The config of a qwen35moe GGUF file (the metadata of llama.cpp)."""
        m = g.meta
        a = m.get("general.architecture", "qwen35moe")

        def k(name, default=None):
            return m.get("%s.%s" % (a, name), default)

        cfg = cls.__new__(cls)
        n = int(k("block_count"))
        every = int(k("full_attention_interval", 4))
        cfg.raw = {}
        cfg.hidden_size = int(k("embedding_length"))
        cfg.num_hidden_layers = n
        cfg.layer_types = ["full_attention" if (i + 1) % every == 0 else "linear_attention"
                           for i in range(n)]
        cfg.rms_norm_eps = float(k("attention.layer_norm_rms_epsilon", 1e-6))
        cfg.num_heads = int(k("attention.head_count"))
        cfg.num_kv_heads = int(k("attention.head_count_kv"))
        cfg.head_dim = int(k("attention.key_length"))
        cfg.rope_theta = float(k("rope.freq_base"))
        cfg.rotary_dim = int(k("rope.dimension_count"))
        cfg.lin_k_heads = int(k("ssm.group_count"))
        cfg.lin_v_heads = int(k("ssm.time_step_rank"))
        cfg.lin_k_dim = int(k("ssm.state_size"))
        cfg.lin_v_dim = int(k("ssm.inner_size")) // cfg.lin_v_heads
        cfg.conv_kernel = int(k("ssm.conv_kernel"))
        cfg.num_experts = int(k("expert_count"))
        cfg.top_k = int(k("expert_used_count"))
        cfg.moe_inter = int(k("expert_feed_forward_length"))
        cfg.shared_inter = int(k("expert_shared_feed_forward_length"))
        cfg.vocab_size = int(g.tensors["output.weight"][0][1])
        cfg.eos_token_ids = None
        cfg.v_tiled = True
        return cfg

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


def _cache_snapshot(self):
    """A copy of the state at position n: the convolution inputs and the
    states of the linear layers (about 60 MB for the 35B). The keys and
    values need no copy: the rows before n do not change."""
    return (self.n, {i: a.copy() for i, a in self.conv.items()},
            {i: a.copy() for i, a in self.state.items()})


def _cache_restore(self, snap):
    """Go back to a snapshot: the state of its position. Rows of keys and
    values after it are written again by the next tokens."""
    n, conv, state = snap
    for i, a in conv.items():
        self.conv[i][...] = a
    for i, a in state.items():
        self.state[i][...] = a
    self.n = n


QwenCache.snapshot = _cache_snapshot
QwenCache.restore = _cache_restore


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
                hk = hv % cfg.lin_k_heads if cfg.v_tiled else hv // rep
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
        g, u, d = (self.M(p + "switch_mlp." + n).c() for n in ("gate_proj", "up_proj", "down_proj"))
        shared = [self.M(p + "shared_expert." + n).c() for n in ("gate_proj", "up_proj", "down_proj")]
        mats = cops.ma_moe_mats(g, u, d, shared)
        slog = np.ascontiguousarray(self.lin(p + "shared_expert_gate", qx)[:, 0])
        k, inner, hid = cfg.top_k, cfg.moe_inter, cfg.hidden_size
        out = np.empty((t, hid), np.float32)
        cops.ma_moe(qx.get(4), qx.get(8), qx.xs, qx.xsum, np.ascontiguousarray(top),
                    np.ascontiguousarray(val), cfg.num_experts, mats, slog, hid, inner,
                    cops.ma_moe_scratch(t, k, cfg.num_experts, hid, inner), out)
        return out

    def embed(self, ids):
        return self.M("embed_tokens").dequant(rows=np.asarray(ids, dtype=np.int64))

    def logits(self, h, chunk=None):
        return mlx_affine.linear(self.M("lm_head", "language_model.lm_head"), mlx_affine.QX(h))


# ---- the step as a program of records (gemma_run, one parallel region) --------

def compile_qwen_step(model, t, verify=False):
    """Compile a step of t tokens of the model (a QwenCPU) into a program of
    records: one parallel region for the whole step, as the Gemma step.

    The input is names["x"] (t x hidden, the rows of the embeddings) and the
    output names["xn"] (after the final norm). bind_qwen_step writes the
    parameters: pos, the RoPE tables, and the arrays of the cache.

    verify makes an MTP verify group: the linear layers do not change their
    state; each writes a log (names["log.<layer>"]) that QwenProgram.commit
    applies for the accepted tokens."""
    from . import program as P
    cfg = model.cfg
    prog = P.Program()
    hid, eps = cfg.hidden_size, float(cfg.rms_norm_eps)
    nq, nk, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    kd, vd, cd = cfg.lin_key_dim, cfg.lin_value_dim, cfg.conv_dim
    k, E, inner = cfg.top_k, cfg.num_experts, cfg.moe_inter
    f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
    x, h, xn = f32(t, hid), f32(t, hid), f32(t, hid)
    wide = max(cd, nq * 2 * hd, vd, nq * hd)
    xq4, xq8 = np.zeros((t, wide), np.int8), np.zeros((t, wide), np.int8)
    xs, xsum = f32(t, wide // 64), f32(t, wide // 64)
    o1, o2, o3, o4 = f32(t, wide), f32(t, wide), f32(t, wide), f32(t, wide)
    att, gate, qout = f32(t, nq * hd), f32(t, nq * hd), f32(t, nq * hd)
    mo, logits = f32(t, hid), f32(t, E)
    val, idx, slog = f32(t, k), np.zeros((t, k), np.int32), f32(t, 1)
    scratch = cops.ma_moe_scratch(t, k, E, hid, inner)
    gscr = np.zeros(t * cd + (cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim if verify else 0),
                    np.float32)
    prog.names.update(x=x, xn=xn)
    pos = prog.slot("pos")
    cos, sin, scores = prog.slot("cos"), prog.slot("sin"), prog.slot("scores")
    hs = prog.slot("hs")

    def quant(src, cols):
        prog.emit(P.MA_QUANT, src, t, cols, xq4, xq8, xs, xsum)

    def lin(name, out, full=None):
        m = model.M(name, full)
        prog.emit(P.MA_LINEAR, xq4 if m.bits == 4 else xq8, xs, xsum, m.q, m.scales, m.biases,
                  m.bits, m.rows, m.cols, t, out)
        return m.rows

    def log_of(i):
        if not verify:
            return None
        log = np.zeros(cops.gdn_log_floats(t, cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim,
                                           cfg.lin_v_dim), np.float32)
        prog.names["log.%d" % i] = log
        return log

    for i in range(model.n_layers):
        p = "layers.%d." % i
        prog.emit(P.RMS_NORM, x, model.F(p + "input_layernorm.weight"), h, t, hid, eps)
        quant(h, hid)
        if cfg.layer_types[i] == "full_attention":
            a = p + "self_attn."
            lin(a + "q_proj", o1)
            lin(a + "k_proj", o2)
            lin(a + "v_proj", o3)
            prog.emit(P.ATTN_PREP, o1, o2, o3, model.F(a + "q_norm.weight"),
                      model.F(a + "k_norm.weight"), cos, sin, prog.slot("K.%d" % i),
                      prog.slot("V.%d" % i), hs, pos, t, nq, nk, hd, cfg.rotary_dim, eps,
                      float(hd ** -0.5), qout, gate)
            prog.emit(P.ATTN_F32H, qout, prog.slot("K.%d" % i), prog.slot("V.%d" % i), scores,
                      att, nq, nk, hd, t, pos, hs, 0, 0)
            prog.emit(P.SIGMUL, att, gate, att, t * nq * hd)
            quant(att, nq * hd)
            lin(a + "o_proj", o4)
        else:
            a = p + "linear_attn."
            lin(a + "in_proj_qkv", o1)
            lin(a + "in_proj_z", o2)
            lin(a + "in_proj_b", o3)
            lin(a + "in_proj_a", att)
            prog.emit(P.GDN, o1, prog.slot("conv.%d" % i),
                      model.F(a + "conv1d.weight", (cd, cfg.conv_kernel)), cfg.conv_kernel, o2,
                      att, o3, model.F(a + "A_log"), model.F(a + "dt_bias"),
                      model.F(a + "norm.weight"), prog.slot("S.%d" % i), gate, gscr, t,
                      cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim, eps,
                      log_of(i))
            quant(gate, vd)
            lin(a + "out_proj", o4)
        # o4 (t x hidden) has the output of the attention; x += o4.
        prog.emit(P.ADD, x, o4, x, t * hid)
        prog.emit(P.RMS_NORM, x, model.F(p + "post_attention_layernorm.weight"), h, t, hid, eps)
        quant(h, hid)
        m = p + "mlp."
        lin(m + "gate", logits)
        prog.emit(P.ROUTER_TOPK, logits, t, E, k, val, idx)
        lin(m + "shared_expert_gate", slog)
        g, u, d = (model.M(m + "switch_mlp." + n).c() for n in ("gate_proj", "up_proj", "down_proj"))
        shared = [model.M(m + "shared_expert." + n).c() for n in ("gate_proj", "up_proj", "down_proj")]
        mats = cops.ma_moe_mats(g, u, d, shared)
        prog.emit(P.MA_MOE, xq4, xq8, xs, xsum, idx, val, t, k, E, mats, slog, hid, inner,
                  scratch, mo)
        prog.emit(P.ADD, x, mo, x, t * hid)
    prog.emit(P.RMS_NORM, x, model.F("norm.weight"), xn, t, hid, eps)
    prog.tokens = t
    return prog.finish()


def bind_qwen_step(prog, model, cache, pos):
    """Write the parameters of a step of prog.tokens tokens from pos."""
    cfg = model.cfg
    t = prog.tokens
    cos, sin = model.rope(np.arange(pos, pos + t))
    kw = {"pos": pos, "cos": np.ascontiguousarray(cos), "sin": np.ascontiguousarray(sin),
          "scores": np.empty(cfg.num_heads * (pos + t) + 64, np.float32)}
    for i in range(model.n_layers):
        if cfg.layer_types[i] == "full_attention":
            K, V = cache.kv[i]
            kw["K.%d" % i], kw["V.%d" % i] = K, V
            kw["hs"] = K.shape[1] * K.shape[2]
        else:
            kw["conv.%d" % i], kw["S.%d" % i] = cache.conv[i], cache.state[i]
    prog.bind(**{k: v for k, v in kw.items() if k in prog.by_name})


class QwenProgram(QwenCPU):
    """QwenCPU with the step as one program (compile_qwen_step). A prompt
    runs in chunks of at most CHUNK tokens; each token count has its own
    program.

    verify() and commit() are the MTP verify group: verify runs the tokens
    and keeps the state of the linear layers; commit(n) applies the first n
    tokens to the state (QWEN_PLAN.md, phase 3)."""

    CHUNK = 512

    def __init__(self, path, cfg=None, layers=None):
        super().__init__(path, cfg, layers)
        self.programs = {}

    def program(self, t):
        prog = self.programs.get(t)
        if prog is None:
            prog = self.programs[t] = compile_qwen_step(self, t)
        return prog

    def verify(self, ids, cache, start_pos):
        """Run a group of tokens from start_pos as an MTP verify group.
        Return the hidden states of every token. The keys and values of the
        full layers are written for all the tokens; the state of the linear
        layers does not change until commit()."""
        t = len(ids)
        prog = self.programs.get(("verify", t))
        if prog is None:
            prog = self.programs[("verify", t)] = compile_qwen_step(self, t, verify=True)
        bind_qwen_step(prog, self, cache, start_pos)
        prog.names["x"][:] = self.embed(ids)
        prog.run()
        self._pending = (prog, cache, start_pos)
        return prog.names["xn"].copy()

    def commit(self, n):
        """Keep the first n tokens of the last verify group."""
        prog, cache, start = self._pending
        cfg = self.cfg
        for i in range(self.n_layers):
            if cfg.layer_types[i] != "full_attention":
                cops.gdn_commit(cache.conv[i], cache.state[i], prog.names["log.%d" % i], n,
                                cfg.conv_kernel, cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim,
                                cfg.lin_v_dim)
        cache.n = start + n
        self._pending = None

    def forward(self, ids, cache, start_pos=0, hook=None):
        ids = list(ids)
        out = []
        c0 = 0
        while c0 < len(ids):
            chunk = ids[c0:c0 + self.CHUNK]
            prog = self.program(len(chunk))
            bind_qwen_step(prog, self, cache, start_pos + c0)
            prog.names["x"][:] = self.embed(chunk)
            prog.run()
            out.append(prog.names["xn"].copy())
            c0 += len(chunk)
        cache.n = start_pos + len(ids)
        return np.concatenate(out)


class QwenSession:
    """Keep the cache between the turns of a chat, as model.Session does for
    Gemma. The state of the linear layers cannot go back to an earlier
    position, so the session keeps a snapshot at the end of each prompt. A
    new turn that shares a prefix with the tokens of the cache restarts from
    the last snapshot at or before the end of that prefix."""

    def __init__(self, model, max_len=8192, snapshots=4):
        self.model = model
        self.max_len = max_len
        self.cache = QwenCache(model.cfg, max_len)
        self.ids = []
        self.snaps = []            # (position, snapshot), the oldest first
        self.max_snaps = snapshots
        self.prefilled = 0

    def prefill(self, ids):
        """Put ids in the cache. Return the hidden state of the last token."""
        ids = list(ids)
        common = 0
        while common < min(len(ids), len(self.ids)) and ids[common] == self.ids[common]:
            common += 1
        start = common if common == len(self.ids) else 0
        if start == 0 and common > 0:
            snap = [sp for sp in self.snaps if sp[0] <= common]
            if snap:
                pos, sn = snap[-1]
                self.cache.restore(sn)
                start = pos
                self.snaps = [sp for sp in self.snaps if sp[0] <= pos]
        if start == len(ids):
            # The prompt is the cache: run its last token again.
            start = len(ids) - 1
            snap = [sp for sp in self.snaps if sp[0] <= start]
            if not snap:
                start = 0
            else:
                self.cache.restore(snap[-1][1])
                start = snap[-1][0]
        if start == 0:
            self.cache = QwenCache(self.model.cfg, self.max_len)
            self.snaps = []
        self.prefilled = len(ids) - start
        h = self.model.forward(ids[start:], self.cache, start_pos=start)
        self.ids = ids
        self.snaps.append((len(ids), self.cache.snapshot()))
        del self.snaps[:-self.max_snaps]
        return h[-1:]

    def step(self, token):
        """Run one generated token. Return its hidden state."""
        pos = len(self.ids)
        h = self.model.forward([token], self.cache, start_pos=pos)
        self.ids.append(int(token))
        return h


# ---- the GGUF files of llama.cpp (QWEN_PLAN.md) --------------------------------

# The GGUF name of each weight of this module ("layers.N." is "blk.N.").
_GGUF_NAMES = {
    "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha", "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out", "linear_attn.norm.weight": "ssm_norm",
    "linear_attn.dt_bias": "ssm_dt.bias", "linear_attn.conv1d.weight": "ssm_conv1d",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output", "self_attn.q_norm.weight": "attn_q_norm",
    "self_attn.k_norm.weight": "attn_k_norm",
    "input_layernorm.weight": "attn_norm", "post_attention_layernorm.weight": "post_attention_norm",
    "mlp.gate": "ffn_gate_inp", "mlp.shared_expert_gate": "ffn_gate_inp_shexp",
    "mlp.shared_expert.gate_proj": "ffn_gate_shexp", "mlp.shared_expert.up_proj": "ffn_up_shexp",
    "mlp.shared_expert.down_proj": "ffn_down_shexp",
    "mlp.switch_mlp.gate_proj": "ffn_gate_exps", "mlp.switch_mlp.up_proj": "ffn_up_exps",
    "mlp.switch_mlp.down_proj": "ffn_down_exps",
}


def gguf_name(name):
    """The GGUF name of a weight of this module (without .weight for the
    matrices)."""
    if name in ("norm.weight",):
        return "output_norm.weight"
    if name == "embed_tokens":
        return "token_embd.weight"
    if name == "lm_head":
        return "output.weight"
    _l, i, rest = name.split(".", 2)
    g = _GGUF_NAMES[rest]
    if not g.endswith(".bias"):
        g += ".weight"
    return "blk.%s.%s" % (i, g)


class _GExperts:
    """The stacked experts of a GGUF tensor, with dequant(expert=e)."""

    def __init__(self, g, gname):
        self.g = g
        self.gname = gname

    def dequant(self, expert):
        return self.g.dequant(self.gname, rows=[int(expert)])[0]


class QwenGGUF(Qwen):
    """The NumPy model on a GGUF file of llama.cpp (UD-Q4_K_M: Q8_0, Q4_K,
    Q5_K, Q6_K, F32). The weights are dequantized when the model uses them.
    The converter of llama.cpp changed some tensors; this class undoes the
    changes where the model needs it:

    - ssm_a holds -exp(A_log), so A_log is log(-ssm_a);
    - the norm weights have the 1 added, as in the MLX files;
    - the value heads are in tiled order (config.v_tiled)."""

    def __init__(self, path, cfg=None, layers=None):
        from .gguf import GGUF
        self.path = path
        self.g = GGUF(path)
        self.cfg = cfg or QwenConfig.from_gguf(self.g)
        self.n_layers = layers or self.cfg.num_hidden_layers
        self._deq = {}

    def W(self, name):
        w = self._deq.get(name)
        if w is None:
            w = self.g.dequant(gguf_name(name))
            if w.ndim == 1:
                w = w.reshape(1, -1)
            w = self._deq[name] = w
        return w

    def t(self, name):
        if name.endswith("linear_attn.A_log"):
            i = name.split(".")[1]
            return np.log(-self.g.dequant("blk.%s.ssm_a" % i)).astype(np.float32)
        return self.g.dequant(gguf_name(name)).astype(np.float32)

    def mat(self, name, full=None):
        return _GExperts(self.g, gguf_name(name))

    def embed(self, ids):
        return self.g.dequant("token_embd.weight", rows=np.asarray(ids, dtype=np.int64))

    def logits(self, h, chunk=16384):
        rows = self.g.tensors["output.weight"][0][1]
        out = np.empty((h.shape[0], rows), np.float32)
        for r0 in range(0, rows, chunk):
            r = np.arange(r0, min(rows, r0 + chunk))
            out[:, r] = h @ self.g.dequant("output.weight", rows=r).T
        return out
