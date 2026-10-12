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

The weights come from a GGUF file of llama.cpp (QwenGGUF, QwenGGUFCPU,
QwenGGUFProgram). Qwen and QwenCPU hold the parts that do not depend on the
form of the weights; the GGUF classes give the weights.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.core import ops
from reclank.formats import tq6

PREFIX = "language_model.model."


def default_kv_form():
    """RECLANK_QWEN_KV, else tq6 when the dense matrices stay in bfloat16
    (RECLANK_DENSE=bf16: the smaller cache leaves the GPU room for more hot
    experts, 18.7 KB a token of Qwen3.8 in place of 21.8; the first full
    attention layer keeps float32 rows, RECLANK_QWEN_KV_FIRST), else int8."""
    v = os.environ.get("RECLANK_QWEN_KV")
    if v:
        return v
    return "tq6" if os.environ.get("RECLANK_DENSE") == "bf16" else "int8"


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
        self.mrope_section = rp.get("mrope_section")
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
        # The form of the keys and values of the cache: "int8" (the default:
        # a scale of max |x| / 127 for each 32 values), "int16" (max |x| /
        # 32767), "k16v8" (int16 keys, int8 values), or "f32". The first full
        # attention layer of a quantized form keeps float32 rows (see
        # layer_kv_form). The default is int8, and tq6 with RECLANK_DENSE=bf16
        # (default_kv_form).
        self.kv_form = default_kv_form()

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
        sec = k("rope.dimension_sections")
        cfg.mrope_section = [int(v) for v in sec] if sec is not None else None
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
        cfg.kv_form = default_kv_form()
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


# The quantized forms of the cache (the C kernels of the 26B read them; tq6:
# reclank/formats/tq6.py, 6 bits and a norm for each 32 rotated values; rq8:
# int8 of the rows rotated in each 32 values as tq6, the queries rotated and
# the output back, as the rq8 cache of Gemma; rk8v8: the keys rq8, the values
# int8 (the rotation spreads the outlier channels of the keys; the values have
# none to spread); Qwen3.6).
QUANT_FORMS = ("int16", "int8", "k16v8", "tq6", "rq8", "rk8v8")
KEY_ROT = ("tq6", "rq8", "rk8v8")       # rotated keys: the queries rotated
VAL_ROT = ("tq6", "rq8")                # rotated values: the output back
# RECLANK_QWEN_KV_FIRST: the form of the first full attention layer under a
# quantized form of the cache: "f32" (the default), a quantized form, or
# "same". Rounding that one layer gives most of the error of the cache: over
# 25766 positions of a chat text with Qwen3.6-35B-A3B, the KL (top 64)
# against a float cache is 3.10e-3 for int8 with this layer float32, 3.64e-3
# with it int16, 3.81e-3 with it float16, and 4.32e-3 for int8 in every
# layer; int16 in every layer gives 3.68e-3 (QWEN_PLAN.md).
KV_FIRST = os.environ.get("RECLANK_QWEN_KV_FIRST", "f32")


def layer_kv_form(form, cfg, i):
    """The form of the cache of the full attention layer i under the form
    of the cache: the first full layer takes KV_FIRST."""
    if form in QUANT_FORMS and KV_FIRST != "same":
        first = next(j for j, t in enumerate(cfg.layer_types) if t == "full_attention")
        if i == first:
            return KV_FIRST
    return form


def layer_quant(cache, i):
    """True when layer i of the cache keeps quantized rows (else float32
    rows, (positions, heads, head_dim))."""
    return cache.layer_form.get(i, cache.kv_form) in QUANT_FORMS


def kv_dtypes(form):
    """The dtypes of the keys and of the values of a quantized form."""
    if form == "tq6":
        return np.uint8, np.uint8
    return (np.int8 if form in ("int8", "rq8", "rk8v8") else np.int16,
            np.int8 if form in ("int8", "rq8", "rk8v8", "k16v8") else np.int16)


def kv_row_bytes(form, per):
    """The bytes of a row of per keys, and of per values, of a quantized form."""
    if form == "tq6":
        return per * 3 // 4, per * 3 // 4
    return tuple(np.dtype(d).itemsize * per for d in kv_dtypes(form))


def kv_arrays(form, max_len, per):
    """The arrays (kq, ks, vq, vs) of a quantized form: a row for each
    position, the heads in order, a float32 scale (tq6: the norm) for each 32
    values."""
    kd, vd = kv_dtypes(form)
    kb, vb = kv_row_bytes(form, per)
    return [np.zeros((max_len, kb // np.dtype(kd).itemsize), kd),
            np.zeros((max_len, per // 32), np.float32),
            np.zeros((max_len, vb // np.dtype(vd).itemsize), vd),
            np.zeros((max_len, per // 32), np.float32)]


class QwenCache:
    """The state of a sequence: keys and values of the full layers, and the
    convolution inputs and the recurrent state of the linear layers."""

    def __init__(self, cfg, max_len=4096, kv=None):
        self.cfg = cfg
        self.n = 0
        self.max_len = max_len
        self.kv_form = kv or getattr(cfg, "kv_form", "f32")
        self.kv = {}
        self.conv = {}
        self.state = {}
        per = cfg.num_kv_heads * cfg.head_dim
        self.layer_form = {}
        for i, t in enumerate(cfg.layer_types):
            lf = layer_kv_form(self.kv_form, cfg, i) if t == "full_attention" else None
            if lf is not None:
                self.layer_form[i] = lf
            if lf in QUANT_FORMS:
                # (kq, ks, vq, vs) of the form of the layer (layer_kv_form)
                self.kv[i] = kv_arrays(lf, max_len, per)
            elif t == "full_attention":
                # position-major (positions, kv heads, head_dim), as all the caches
                shape = (max_len, cfg.num_kv_heads, cfg.head_dim)
                self.kv[i] = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
            else:
                self.conv[i] = np.zeros((cfg.conv_kernel - 1, cfg.conv_dim), np.float32)
                self.state[i] = np.zeros((cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim),
                                         np.float32)
        # The M-RoPE positions (3, max_len) of the rows, after a prompt with
        # an image (set_rope); None: the position of a row is its index.
        self.rpos = None

    def set_rope(self, pos):
        """Set the M-RoPE positions (3, n) of a prompt of n tokens from row 0
        (media.mrope_positions). The rows after it (the answer, the next
        prompt) continue from the largest position + 1. A prompt of text
        only leaves the positions as they are (the row index)."""
        n = pos.shape[1]
        if self.rpos is None:
            if np.array_equal(pos, np.broadcast_to(np.arange(n), pos.shape)):
                return
            self.rpos = np.empty((3, self.max_len), np.int64)
        self.rpos[:, :n] = pos
        nxt = int(pos.max()) + 1 if n else 0
        self.rpos[:, n:] = np.arange(nxt, nxt + self.max_len - n)


def rope_positions(cache, pos, t):
    """The rope positions of the rows pos .. pos + t - 1: (t,), or (3, t)
    where the rows of M-RoPE differ (the tokens of an image)."""
    rp = getattr(cache, "rpos", None)
    if rp is None:
        return np.arange(pos, pos + t)
    r = rp[:, pos:pos + t]
    if r.shape[1] < t:          # the padding rows of a group past the end
        r = np.concatenate([r, r[:, -1:] + np.arange(1, t - r.shape[1] + 1)], axis=1)
    if np.array_equal(r[0], r[1]) and np.array_equal(r[0], r[2]):
        return r[0]
    return r


def media_inputs(x, pos, media):
    """Put the rows of the images (media: spans at cache positions) into x,
    the input rows of the tokens at pos .. pos + len(x) - 1. Return x."""
    t = x.shape[0]
    for sp in media or ():
        a, b = max(sp.start, pos), min(sp.end, pos + t)
        if a < b:
            x[a - pos:b - pos] = sp.rows[a - sp.start:b - sp.start]
    return x


def gdn_flags(cfg):
    """The flags of the GDN record: 1 for the tiled order of the value heads,
    2 for the sigmoid gate of its norm (qwen4exp)."""
    return (1 if cfg.v_tiled else 0) | (2 if getattr(cfg, "lin_gate", "silu") == "sigmoid" else 0)


def kv_store(cache, i, k, v, pos):
    """Write the keys and values k, v (t x heads x head_dim) of layer i at
    positions pos .. pos + t - 1."""
    t = k.shape[0]
    if layer_quant(cache, i):
        kq, ks, vq, vs = cache.kv[i]
        lf = cache.layer_form.get(i, cache.kv_form)
        if lf in ("rq8", "rk8v8"):
            k = tq6.rotate(k)                       # the rows rotated in each 32 values
        if lf == "rq8":
            v = tq6.rotate(v)
        for src, q, sc in ((k, kq, ks), (v, vq, vs)):
            quant = {np.dtype(np.int8): cops.quantize_i8_groups, np.dtype(np.uint8): tq6.quantize,
                     np.dtype(np.int16): cops.quantize_i16_groups}[q.dtype]
            a, b = quant(np.ascontiguousarray(src, np.float32))
            q[pos:pos + t] = a.reshape(t, -1)
            sc[pos:pos + t] = b.reshape(t, -1)
        return
    K, V = cache.kv[i]
    K[pos:pos + t] = k
    V[pos:pos + t] = v


def kv_rows(cache, i, n):
    """The keys and values of the first n positions of layer i, float32,
    (heads, n, head_dim) each."""
    if layer_quant(cache, i):
        cfg = cache.cfg
        out = []
        lf = cache.layer_form.get(i, cache.kv_form)
        for key, (q, sc) in ((True, cache.kv[i][0:2]), (False, cache.kv[i][2:4])):
            if q.dtype == np.uint8:
                x = tq6.dequantize(q[:n], sc[:n])
            else:
                x = q[:n].astype(np.float32).reshape(n, -1, 32) * sc[:n, :, None]
                if lf == "rq8" or (key and lf == "rk8v8"):
                    x = tq6.unrotate(x.reshape(n, -1))
            out.append(x.reshape(n, cfg.num_kv_heads, cfg.head_dim).transpose(1, 0, 2))
        return out
    K, V = cache.kv[i]
    return K[:n].transpose(1, 0, 2), V[:n].transpose(1, 0, 2)


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
    """The text model in NumPy, without its weights: a subclass (QwenGGUF)
    gives t, mat, W, embed, and logits. The weights stay quantized in the
    memory maps; the NumPy path dequantizes each matrix when it uses it.

        model = QwenGGUF(path)
        cache = QwenCache(model.cfg)
        h = model.forward(ids, cache)              # the hidden states after the final norm
        logits = model.logits(h[-1:])
    """

    # ---- the layers ----------------------------------------------------------

    def rope(self, positions):
        """cos and sin (t, rotary_dim) of positions: (t,), or (3, t) of M-RoPE
        (time, row, column). The frequency pairs j take the row if j % 3 ==
        1, the column if j % 3 == 2, else the time (the interleaved M-RoPE
        of transformers, sections [11, 11, 10])."""
        cfg = self.cfg
        d = cfg.rotary_dim
        inv = 1.0 / (cfg.rope_theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        positions = np.asarray(positions)
        if positions.ndim == 2:
            f = positions[0].astype(np.float64)[:, None] * inv[None, :]
            sec = getattr(cfg, "mrope_section", None) or [11, 11, 10]
            for dim, off in ((1, 1), (2, 2)):
                j = np.arange(off, sec[dim] * 3, 3)
                f[:, j] = positions[dim].astype(np.float64)[:, None] * inv[j][None, :]
        else:
            f = np.outer(positions.astype(np.float64), inv)
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
        cos, sin = self.rope(rope_positions(cache, pos, t))
        d = cfg.rotary_dim

        def rot(x):
            xr = x[..., :d]
            half = d // 2
            rh = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
            return np.concatenate([xr * cos[:, None] + rh * sin[:, None], x[..., d:]], axis=-1)

        q, k = rot(q), rot(k)
        kv_store(cache, i, k, v, pos)
        n = pos + t
        K, V = kv_rows(cache, i, n)
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
        # The gate of the norm: silu for Qwen3.5, sigmoid for qwen4exp
        # (config.json output_gate_type).
        gate = sigmoid(z) if getattr(cfg, "lin_gate", "silu") == "sigmoid" else silu(z)
        o = rms_norm(o, self.t(p + "norm.weight"), cfg.rms_norm_eps) * gate
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

    def forward(self, ids, cache, start_pos=0, hook=None, media=None):
        """Run tokens from start_pos. Return the hidden states after the final
        norm, (t, hidden). hook(name, array) gets the output of each layer.
        media: spans (cache positions) whose rows replace the embeddings."""
        x = media_inputs(self.embed(ids), start_pos, media)
        for i in range(self.n_layers):
            x = self.layer(i, x, cache, start_pos)
            if hook is not None:
                hook("layer.%d" % i, x)
        cache.n = start_pos + len(ids)
        return rms_norm(x, self.t("norm.weight"), self.cfg.rms_norm_eps)

# ---- the CPU path with the C kernels (QWEN_PLAN.md, phase 2) ----------------

class QwenCPU(Qwen):
    """The model with the C kernels of cops: the Gated DeltaNet
    (backends/cpu/csrc/common/deltanet.c) and the shared ops (rms_norm, the attention of one
    token). A subclass (QwenGGUFCPU) gives the products of its weights: M,
    QX, lin, the experts, embed, logits, and the emit_* records."""

    def F(self, name, shape=None):
        a = self._f.get(name)
        if a is None:
            a = np.ascontiguousarray(self.t(name), dtype=np.float32)
            a = self._f[name] = a.reshape(shape) if shape is not None else a
        return a

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

    def forward(self, ids, cache, start_pos=0, hook=None, media=None):
        x = media_inputs(self.embed(ids), start_pos, media)
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
        qx = self.QX(h)
        qkv = self.lin(p + "in_proj_qkv", qx)
        z = self.lin(p + "in_proj_z", qx)
        b = self.lin(p + "in_proj_b", qx)
        a = self.lin(p + "in_proj_a", qx)
        out = np.empty((t, cfg.lin_value_dim), np.float32)
        cops.gdn_step(qkv, cache.conv[i], self.F(p + "conv1d.weight", (cfg.conv_dim, cfg.conv_kernel)),
                      z, a, b, self.F(p + "A_log"), self.F(p + "dt_bias"), self.F(p + "norm.weight"),
                      cache.state[i], out, np.empty((t, cfg.conv_dim), np.float32),
                      cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim,
                      cfg.rms_norm_eps, gdn_flags(cfg))
        return self.lin(p + "out_proj", self.QX(out))

    def full_attention(self, i, h, cache, pos):
        cfg = self.cfg
        p = "layers.%d.self_attn." % i
        t = h.shape[0]
        nq, nk, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        qx = self.QX(h)
        qg = self.lin(p + "q_proj", qx).reshape(t, nq, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(t, nq * hd)
        k = self.lin(p + "k_proj", qx).reshape(t, nk, hd)
        v = self.lin(p + "v_proj", qx).reshape(t, nk, hd)
        q = ops.rms_norm(q, self.F(p + "q_norm.weight"), cfg.rms_norm_eps)
        k = ops.rms_norm(k, self.F(p + "k_norm.weight"), cfg.rms_norm_eps)
        cos, sin = self.rope(rope_positions(cache, pos, t))
        d = cfg.rotary_dim
        half = d // 2

        def rot(x):
            xr = x[..., :d]
            rh = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
            return np.concatenate([xr * cos[:, None] + rh * sin[:, None], x[..., d:]], axis=-1)

        q, k = rot(q), rot(k)
        kv_store(cache, i, k, v, pos)
        # The attention kernel of cops has no scale (Gemma uses 1), so the
        # query takes the scale of Qwen.
        q = np.ascontiguousarray(q * np.float32(hd ** -0.5))
        lf = cache.layer_form.get(i, cache.kv_form)
        if lf in ("rq8", "rk8v8"):
            q = np.ascontiguousarray(tq6.rotate(q.reshape(t, -1)).reshape(t, nq, hd))
        o = np.empty((t, nq, hd), np.float32)
        for j in range(t):
            n = pos + j + 1
            if layer_quant(cache, i):
                kq, ks, vq, vs = cache.kv[i]
                o[j] = ops.attn_decode(np.ascontiguousarray(q[j]), kq, ks, vq, vs, nq, nk,
                                       hd, n)
            else:
                K, V = cache.kv[i]
                o[j] = ops.attn_decode_f32(q[j], K[:n].transpose(1, 0, 2),
                                           V[:n].transpose(1, 0, 2), pos + j)
        if lf == "rq8":
            o = tq6.unrotate(o.reshape(t, -1))
        o = o.reshape(t, nq * hd) * sigmoid(gate)
        return self.lin(p + "o_proj", self.QX(o))

    def moe(self, i, h):
        cfg = self.cfg
        p = "layers.%d.mlp." % i
        t = h.shape[0]
        qx = self.QX(h)
        logits = self.lin(p + "gate", qx)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
        top = np.argsort(-probs, axis=-1, kind="stable")[:, :cfg.top_k].astype(np.int32)
        val = np.take_along_axis(probs, top, axis=-1)
        val = (val / val.sum(axis=-1, keepdims=True)).astype(np.float32)
        slog = np.ascontiguousarray(self.lin(p + "shared_expert_gate", qx)[:, 0])
        out = np.empty((t, cfg.hidden_size), np.float32)
        self.experts(p, qx, np.ascontiguousarray(top), np.ascontiguousarray(val), slog, out)
        return out

    # ---- the records of the program (compile_qwen_step) ----

# ---- the step as a program of records (gemma_run, one parallel region) --------

def compile_qwen_step(model, t, verify=False):
    """Compile a step of t tokens of the model (a QwenCPU) into a program of
    records: one parallel region for the whole step, as the Gemma step.

    The input is names["x"] (t x hidden, the rows of the embeddings) and the
    output names["xn"] (after the final norm). bind_qwen_step writes the
    parameters: pos, the RoPE tables, and the arrays of the cache.

    verify makes an MTP verify group: the linear layers do not change their
    state; each writes a log (names["log.<layer>"]) that QwenGGUFProgram.commit
    applies for the accepted tokens."""
    from reclank.ir import program as P
    cfg = model.cfg
    prog = P.Program()
    hid, eps = cfg.hidden_size, float(cfg.rms_norm_eps)
    nq, nk, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    kd, vd, cd = cfg.lin_key_dim, cfg.lin_value_dim, cfg.conv_dim
    k, E, inner = cfg.top_k, cfg.num_experts, cfg.moe_inter
    f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
    x, xn = f32(t, hid), f32(t, hid)
    wide = max(cd, nq * 2 * hd, vd, nq * hd)
    xb = model.x_buffers(t, wide)
    scratch = model.moe_scratch(t)
    # model.pack_temps (the GPU, ir/alloc.pack): each layer has buffers of
    # its own (prog.temps), which the packer then places by liveness; else
    # every layer reuses the same ones
    fresh = getattr(model, "pack_temps", False)

    def layer_buffers(i):
        """h, o1 to o4, att, gate, qout, mo, logits, val, idx, slog, gscr,
        kbuf: the widest of all the layers, or (fresh) those of layer i (the
        records read them as flat rows of their own width)."""
        def tmp(*sh, dtype=np.float32):
            a = P.lazy_zeros(sh, dtype)
            if fresh:
                prog.temps.append(a)
            return a
        gn = t * cd + (cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim if verify else 0)
        if not fresh:
            o = [wide] * 4 + [nq * hd] * 3
        elif cfg.layer_types[i] == "full_attention":
            o = [nq * 2 * hd, nk * hd, nk * hd, hid, nq * hd, nq * hd, nq * hd]
            gn = 0
        else:
            o = [cd, vd, cfg.lin_v_heads, hid, cfg.lin_v_heads, vd, 0]
        return (tmp(t, hid), *[tmp(t, n) if n else None for n in o], tmp(t, hid), tmp(t, E),
                tmp(t, k), tmp(t, k, dtype=np.int32), tmp(t, 1), tmp(gn) if gn else None,
                tmp(t, nk * hd) if kv16 and (not fresh or quant_layer(i)) else None)
    prog.names.update(x=x, xn=xn)
    pos = prog.slot("pos")
    cos, sin, scores = prog.slot("cos"), prog.slot("sin"), prog.slot("scores")
    hs = prog.slot("hs")

    # The matrices of RQ8_0 / RQ6_K (model.dense_rot) read x rotated in each
    # 32 values: a copy made at the first such product of each x (as
    # compile_qwen4_step)
    rot = getattr(model, "dense_rot", None)
    cur = {}
    xr = xbr = None

    def quant(src, cols):
        model.emit_quant(prog, xb, src, cols)
        cur.update(src=src, cols=cols, rot=False)

    def lin(name, out):
        nonlocal xr, xbr
        if rot is None or not rot(name):
            model.emit_lin(prog, xb, name, out)
            return
        if xr is None:
            xr = np.zeros((t, wide), np.float32)
            xbr = model.x_buffers(t, wide)
        if not cur["rot"]:
            n = t * cur["cols"]
            prog.emit(P.COPY, cur["src"], xr, n * 4)
            prog.emit(P.TQ_ROT, xr, n // 32, 0)
            model.emit_quant(prog, xbr, xr, cur["cols"])
            cur["rot"] = True
        model.emit_lin(prog, xbr, name, out)

    form = getattr(cfg, "kv_form", "f32")
    # the quantized layers (layer_kv_form): the first full layer can keep
    # float32 rows
    quant_layer = lambda i: layer_kv_form(form, cfg, i) in QUANT_FORMS  # noqa: E731
    kv16 = any(quant_layer(i) for i, t in enumerate(cfg.layer_types) if t == "full_attention")
    per = nk * hd
    h, o1, o2, o3, o4, att, gate, qout, mo, logits, val, idx, slog, gscr, kbuf = layer_buffers(0)
    lo_n = (np.zeros(t, np.int32), np.zeros(t, np.int32)) if kv16 else None

    def scalar(op, a, b):
        r = prog.temp()
        prog.emit(op, r, a, b)
        return r

    def emit_attn16(i):
        """The int16 cache (the form of the 26B): ATTN_PREP writes the key
        to kbuf, GP_KV_WRITE stores the rows of the key and the value at
        pos, and GP_ATTN_QC reads the cache. A group of at most 16 tokens
        runs each query as a step does (the same bits); a larger group
        uses GP_ATTN_QC_MT."""
        a = "layers.%d.self_attn." % i
        prog.emit(P.ATTN_PREP, o1, o2, o3, model.F(a + "q_norm.weight"),
                  model.F(a + "k_norm.weight"), cos, sin, None, None, 0, pos, t, nq, nk, hd,
                  cfg.rotary_dim, eps, float(hd ** -0.5), qout, gate, kbuf)
        base = [prog.slot("%s.%d" % (nm, i)) for nm in ("kq", "ks", "vq", "vs")]
        # the int8 forms (layer_kv_form): 1 byte for a value, other records
        lf = layer_kv_form(form, cfg, i)
        kb, vb = kv_row_bytes(lf, per)
        write, one, many = {"int16": (P.KV_WRITE, P.ATTN_QC, P.ATTN_QC_MT),
                            "int8": (P.KV_WRITE8, P.ATTN_Q8, P.ATTN_Q8_MT),
                            "rq8": (P.KV_WRITE8, P.ATTN_Q8, P.ATTN_Q8_MT),
                            "rk8v8": (P.KV_WRITE8, P.ATTN_Q8, P.ATTN_Q8_MT),
                            "k16v8": (P.KV_WRITEV8, P.ATTN_V8, P.ATTN_V8_MT),
                            "tq6": (P.KV_WRITETQ, P.ATTN_TQ, P.ATTN_TQ_MT)}[lf]
        rows = [scalar(P.S_ADD, b, scalar(P.S_MUL, pos, step))
                for b, step in zip(base, (kb, per // 8, vb, per // 8))]
        if lf in ("rq8", "rk8v8"):
            # the rows of the key (and of the value) rotated, then int8
            prog.emit(P.TQ_ROT, kbuf, t * per // 32, 0)
        if lf == "rq8":
            prog.emit(P.TQ_ROT, o3, t * per // 32, 0)
        prog.emit(write, kbuf, o3, None, None, *rows, t * per)
        if lf in KEY_ROT:
            # the queries and the output in the rotated form (tq6.py)
            prog.emit(P.TQ_ROT, qout, t * nq * hd // 32, 0)
        # model.attn_mt False (QwenGPU without the tensor cores): the int8
        # caches and the TQ6 cache take the float32 tiles (k_flash_f32t);
        # RECLANK_GPU_F32T=0: each query as a step, in a group of any size
        one_by_one = not getattr(model, "attn_mt", True) and (
            lf in ("int8", "k16v8", "tq6", "rq8", "rk8v8") and not getattr(model, "attn_f32t", True))
        if t <= 16 or one_by_one:
            for j in range(t):
                nj = scalar(P.S_ADD, pos, j + 1)
                prog.emit(one, qout[j:j + 1], *base, scores, att[j:j + 1], nq, nk, hd, nj)
        else:
            prog.emit(many, qout, *base, scores, att, nq, nk, hd, t, pos, 0, 0, *lo_n)
        if lf in VAL_ROT:
            prog.emit(P.TQ_ROT, att, t * nq * hd // 32, 1)

    def log_of(i):
        if not verify:
            return None
        log = np.zeros(cops.gdn_log_floats(t, cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim,
                                           cfg.lin_v_dim), np.float32)
        prog.names["log.%d" % i] = log
        return log

    layer_begin = getattr(model, "layer_begin", None)
    for i in range(model.n_layers):
        p = "layers.%d." % i
        if fresh and i > 0:
            h, o1, o2, o3, o4, att, gate, qout, mo, logits, val, idx, slog, gscr, kbuf = \
                layer_buffers(i)
        if layer_begin is not None:
            layer_begin(prog, i)
        prog.emit(P.RMS_NORM, x, model.F(p + "input_layernorm.weight"), h, t, hid, eps)
        quant(h, hid)
        if cfg.layer_types[i] == "full_attention":
            a = p + "self_attn."
            lin(a + "q_proj", o1)
            lin(a + "k_proj", o2)
            lin(a + "v_proj", o3)
            if quant_layer(i):
                emit_attn16(i)
            else:
                prog.emit(P.ATTN_PREP, o1, o2, o3, model.F(a + "q_norm.weight"),
                          model.F(a + "k_norm.weight"), cos, sin, prog.slot("K.%d" % i),
                          prog.slot("V.%d" % i), hs, pos, t, nq, nk, hd, cfg.rotary_dim, eps,
                          float(hd ** -0.5), qout, gate)
            if quant_layer(i):
                pass
            elif hasattr(model, "emit_attn"):
                model.emit_attn(prog, qout, prog.slot("K.%d" % i), prog.slot("V.%d" % i), scores,
                                att, nq, nk, hd, t, pos, hs)
            else:
                prog.emit(P.ATTN_F32H, qout, prog.slot("K.%d" % i), prog.slot("V.%d" % i),
                          scores, att, nq, nk, hd, t, pos, hs, 0, 0)
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
                      log_of(i), gdn_flags(cfg), prog.slot("nreal"))
            quant(gate, vd)
            lin(a + "out_proj", o4)
        # o4 (t x hidden) has the output of the attention; x += o4.
        prog.emit(P.ADD, x, o4, x, t * hid)
        prog.emit(P.RMS_NORM, x, model.F(p + "post_attention_layernorm.weight"), h, t, hid, eps)
        quant(h, hid)
        m = p + "mlp."
        lin(m + "gate", logits)
        lin(m + "shared_expert_gate", slog)
        prog.emit(P.ROUTER_TOPK, logits, t, E, k, val, idx)
        model.emit_moe(prog, xb, m, idx, val, slog, scratch, mo)
        prog.emit(P.ADD, x, mo, x, t * hid)
    prog.emit(P.RMS_NORM, x, model.F("norm.weight"), xn, t, hid, eps)
    prog.tokens = t
    return prog.finish()


def bind_qwen_step(prog, model, cache, pos):
    """Write the parameters of a step of prog.tokens tokens from pos."""
    cfg = model.cfg
    t = prog.tokens
    cos, sin = model.rope(rope_positions(cache, pos, t))
    kw = {"pos": pos, "cos": np.ascontiguousarray(cos), "sin": np.ascontiguousarray(sin),
          "scores": scores_buffer(cfg, pos + t)}
    kw.update(cache_params(model, cache))
    prog.bind(**{k: v for k, v in kw.items() if k in prog.by_name})


def scores_buffer(cfg, n):
    """The scratch of the attention of n positions: a row for each head (a
    step), or for each thread (GP_ATTN_QC_MT of a group)."""
    return np.empty(max(cfg.num_heads, os.cpu_count() or 1) * n + 64, np.float32)


def cache_params(model, cache):
    """The slots of the arrays of the cache."""
    cfg = model.cfg
    assert cache.kv_form == getattr(cfg, "kv_form", "f32"), "the cache has another form"
    kw = {}
    for i in range(model.n_layers):
        if cfg.layer_types[i] == "full_attention" and layer_quant(cache, i):
            for nm, a in zip(("kq", "ks", "vq", "vs"), cache.kv[i]):
                kw["%s.%d" % (nm, i)] = a
        elif cfg.layer_types[i] == "full_attention":
            K, V = cache.kv[i]
            kw["K.%d" % i], kw["V.%d" % i] = K, V
            kw["hs"] = K.shape[2]           # position-major: a head stride of head_dim
        else:
            kw["conv.%d" % i], kw["S.%d" % i] = cache.conv[i], cache.state[i]
    return kw


class _QwenRuns:
    """The runs of a model with the step as one program (compile_qwen_step). A prompt
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
        layers does not change until commit().

        The rule of verify (all the backends, and the MTP cache): the rows
        at positions >= cache.n (the drafts that commit drops) stay in the
        cache until the next run writes them again. That is safe because no
        reader takes a row at or after cache.n: attention reads the rows
        before pos + t of its own run, and QSA_SELECT the blocks before it.
        A new reader of such rows needs a rollback here first."""
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

    def forward(self, ids, cache, start_pos=0, hook=None, media=None):
        ids = list(ids)
        out = []
        c0 = 0
        while c0 < len(ids):
            chunk = ids[c0:c0 + self.CHUNK]
            prog = self.program(len(chunk))
            bind_qwen_step(prog, self, cache, start_pos + c0)
            prog.names["x"][:] = media_inputs(self.embed(chunk), start_pos + c0, media)
            prog.run()
            out.append(prog.names["xn"].copy())
            c0 += len(chunk)
        cache.n = start_pos + len(ids)
        return np.concatenate(out)



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
        from reclank.formats.gguf import GGUF
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


class KMat:
    """One GGUF matrix (or a stack: the experts) as its raw blocks: data
    (uint8, a view into the memory map), the ggml type, rows and cols of
    one matrix."""

    def __init__(self, g, gname):
        blocks, dims, self.type = g.raw(gname)
        self.data = np.ascontiguousarray(blocks).view(np.uint8).reshape(-1)
        self.cols = int(dims[0])
        self.rows = int(dims[1]) if len(dims) > 1 else 1
        # RQ8_0 / RQ6_K / RQ5_K / RQ4_K: the blocks of Q8_0 / Q6_K / Q5_K / Q4_K
        # of the rows rotated in each 32 values (the TQ6 rotation); the
        # products read x rotated the same way (rot: compile_qwen_step,
        # QwenGGUFCPU.lin)
        from reclank.formats.gguf import ROT_BASE
        self.rot = self.type in ROT_BASE
        if self.rot:
            self.type = ROT_BASE[self.type]

    def c(self):
        return (self.data, self.type)


class KX:
    """Rows of x, quantized for the GGUF products (backends/cpu/csrc/kernels/kq): int8
    for each value, a scale for each 32, a sum for each 16. The F32
    matrices use x itself."""

    def __init__(self, x):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.t, cols = self.x.shape
        self.xq = np.empty((self.t, cols), np.int8)
        self.xs = np.empty((self.t, cols // 32), np.float32)
        self.xm = np.empty((self.t, cols // 16), np.float32)
        cops.kq_quant_x(self.x, self.xq, self.xs, self.xm)


class QwenGGUFCPU(QwenCPU):
    """QwenCPU on a GGUF file (QwenGGUF): the products run on the blocks of
    the file with the kernels of backends/cpu/csrc/kernels/kq. The router and the other
    F32 matrices use x without quantization."""

    t = QwenGGUF.t

    def __init__(self, path, cfg=None, layers=None):
        QwenGGUF.__init__(self, path, cfg, layers)
        self._m = {}
        self._f = {}

    def M(self, name, full=None):
        m = self._m.get(name)
        if m is None:
            m = self._m[name] = KMat(self.g, gguf_name(name))
        return m

    def QX(self, h):
        return KX(h)

    def dense_rot(self, name):
        """True when the products of the matrix read x rotated in each 32
        values (an RQ8_0, RQ6_K, RQ5_K, or RQ4_K matrix of the file)."""
        return self.M(name).rot

    def lin(self, name, qx, full=None):
        m = self.M(name)
        if m.rot:
            qx = KX(tq6.rotate(qx.x))
        out = np.empty((qx.t, m.rows), np.float32)
        cops.kq_linear(m.data, m.type, m.rows, m.cols, qx.xq, qx.xs, qx.xm, qx.x, qx.t, out)
        return out

    def moe_mats(self, p):
        ms = [self.M(p + "switch_mlp." + n) for n in ("gate_proj", "up_proj", "down_proj")]
        ss = [self.M(p + "shared_expert." + n) for n in ("gate_proj", "up_proj", "down_proj")]
        if any(m.rot for m in ms + ss):
            raise ValueError("Qwen3.6: rotated experts (RQ8_0, RQ6_K, ...) are not supported; only "
                             "the dense matrices may be rotated")
        g, u, d = (m.c() for m in ms)
        shared = [m.c() for m in ss]
        return cops.kq_moe_mats(g, u, d, shared)

    def moe_scratch(self, t):
        cfg = self.cfg
        return cops.kq_moe_scratch(t, cfg.top_k, cfg.num_experts, cfg.hidden_size, cfg.moe_inter)

    def experts(self, p, qx, top, val, slog, out):
        cfg = self.cfg
        cops.kq_moe(qx.xq, qx.xs, qx.xm, top, val, cfg.num_experts, self.moe_mats(p), slog,
                    cfg.hidden_size, cfg.moe_inter, self.moe_scratch(top.shape[0]), out)

    def embed(self, ids):
        m = self.M("embed_tokens")
        return cops.kq_rows(m.data, m.type, m.cols, ids)

    def logits(self, h, chunk=None):
        return self.lin("lm_head", KX(h))

    def x_buffers(self, t, wide):
        return dict(xq=np.zeros((t, wide), np.int8), xs=np.zeros((t, wide // 32), np.float32),
                    xm=np.zeros((t, wide // 16), np.float32), t=t, src=None)

    def emit_quant(self, prog, xb, src, cols):
        from reclank.ir import program as P
        prog.emit(P.KQ_QUANT, src, xb["t"], cols, xb["xq"], xb["xs"], xb["xm"])
        # The F32 products that follow read the float rows.
        xb["src"] = src

    def emit_lin(self, prog, xb, name, out):
        from reclank.ir import program as P
        m = self.M(name)
        prog.emit(P.KQ_LINEAR, xb["xq"], xb["xs"], xb["xm"], xb["src"], m.data, m.type, m.rows,
                  m.cols, xb["t"], out)

    def emit_moe(self, prog, xb, p, idx, val, slog, scratch, out):
        from reclank.ir import program as P
        cfg = self.cfg
        prog.emit(P.KQ_MOE, xb["xq"], xb["xs"], xb["xm"], idx, val, xb["t"], cfg.top_k,
                  cfg.num_experts, self.moe_mats(p), slog, cfg.hidden_size, cfg.moe_inter,
                  scratch, out)


class QwenGGUFProgram(_QwenRuns, QwenGGUFCPU):
    """QwenGGUFCPU with the step as one program."""
