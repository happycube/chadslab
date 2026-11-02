"""DiffusionGemma 26B-A4B on the CPU in NumPy: the forward of transformers'
modeling_diffusion_gemma.py op for op, in float32, from the bf16 checkpoint.
The slow reference of the runtime: the engines' checks compare against it.

Ported from transformers (src/transformers/models/diffusion_gemma/
modeling_diffusion_gemma.py and generation_diffusion_gemma.py, Copyright 2026
the HuggingFace Team; Apache License 2.0). See THIRD_PARTY_NOTICES.md.

The model is Gemma 4 26B-A4B's 30 layers (25 with a window of 1024, 5 global;
the shared MLP and 128 experts, 8 a token) used twice, with the same weights:

- the encoder: the layers as a causal model over the prompt and then over each
  finished canvas; it only fills the KV cache (its own layer_scalar of each
  layer; its final norm and head are not used);
- the decoder: the layers over a canvas of 256 tokens at the positions after
  the cache, bidirectional within the canvas, reading the cache and never
  writing it (the decoder's layer_scalar). The canvas embedding goes through
  the self-conditioning block: rmsnorm(embed(canvas) + mlp(rmsnorm(soft) w)),
  the last norm with no weight, where soft = softmax(the last step's logits) @
  the token table * sqrt(hidden) (zeros at the first step). The logits are the
  tied head with the soft cap of 30.

The keys a canvas sees: in a global layer every cached row and the canvas; in
a layer with a window the last window - 1 cached rows (transformers'
DynamicSlidingWindowLayer keeps 1023) and the canvas. The encoder over a new
block is plain Gemma 4: causal, a query sees the 1024 positions up to its own.

    m = DGemmaRef("MODELS/diffusiongemma-26B-A4B-it")
    cache = m.cache()
    m.encode(prompt_ids, cache)                 # fills the cache
    logits = m.decode(canvas, cache)            # [256, vocab], cache unchanged
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.core import ops
from reclank.formats.safetensors import Shards
from reclank.models.gemma4 import rope as rope_mod

DEC = "model.decoder."
ENC = "model.encoder.language_model."


def inv_freq_f32(head_dim, base, partial):
    """The inverse frequencies as transformers makes them: float32 throughout
    (the default rope of the windowed layers is the proportional one with a
    factor of 1). The angles pos * inv_freq are float32 too (cos_sin_f32). At
    the positions of a long prompt their rounding moves the phases by about
    1e-4 rad (2000 tokens), which float64 tables do not; the reference follows
    transformers."""
    angles = int(partial * head_dim // 2)
    rot = np.float32(1.0) / np.power(np.float32(base),
                                     np.arange(0, 2 * angles, 2).astype(np.float32) / np.float32(head_dim))
    return np.concatenate([rot, np.zeros(head_dim // 2 - angles, np.float32)]).astype(np.float32)


def cos_sin_f32(inv_freq, positions):
    f = np.asarray(positions, dtype=np.float32)[:, None] * inv_freq[None, :]
    e = np.concatenate([f, f], axis=-1)
    return np.cos(e), np.sin(e)


class Config:
    """The sizes of config.json's text_config, a plan for each layer."""

    def __init__(self, cfg):
        tc = cfg["text_config"]
        self.canvas = cfg["canvas_length"]
        self.hidden = tc["hidden_size"]
        self.inter = tc["intermediate_size"]
        self.moe_inter = tc["moe_intermediate_size"]
        self.layers = tc["num_hidden_layers"]
        self.heads = tc["num_attention_heads"]
        self.kv_heads = tc["num_key_value_heads"]
        self.head_dim = tc["head_dim"]
        self.g_head_dim = tc["global_head_dim"]
        self.g_kv_heads = tc["num_global_key_value_heads"]
        self.eps = tc["rms_norm_eps"]
        self.vocab = tc["vocab_size"]
        self.window = tc["sliding_window"]
        self.softcap = tc["final_logit_softcapping"]
        self.experts = tc["num_experts"]
        self.top_k = tc["top_k_experts"]
        self.sliding = [t == "sliding_attention" for t in tc["layer_types"]]
        rp = tc["rope_parameters"]
        self.inv_freq = {
            True: inv_freq_f32(self.head_dim, rp["sliding_attention"]["rope_theta"], 1.0),
            False: inv_freq_f32(self.g_head_dim, rp["full_attention"]["rope_theta"],
                                rp["full_attention"].get("partial_rotary_factor", 1.0)),
        }
        self.embed_scale = np.float32(self.hidden ** 0.5)

    def dims(self, i):
        """(head_dim, kv_heads) of layer i."""
        return (self.head_dim, self.kv_heads) if self.sliding[i] else (self.g_head_dim, self.g_kv_heads)


class Cache:
    """The encoder's keys and values: float32 rows (t, kv_heads, head_dim) of
    every position for each layer (the window is applied at the read)."""

    def __init__(self, cfg):
        self.k = [None] * cfg.layers
        self.v = [None] * cfg.layers
        self.n = 0

    def append(self, i, k, v):
        self.k[i] = k if self.k[i] is None else np.concatenate([self.k[i], k])
        self.v[i] = v if self.v[i] is None else np.concatenate([self.v[i], v])


def _f32(st, name):
    if st.dtype(name) == "BF16":
        return ops.bf16_to_f32(st.get_bf16(name))
    return st.get(name)


# the GGUF names (llama.cpp's diffusion-gemma) of the reference's tensors
_GG_LAYER = {
    "input_layernorm": "attn_norm.weight", "post_attention_layernorm": "post_attention_norm.weight",
    "pre_feedforward_layernorm": "ffn_norm.weight", "post_feedforward_layernorm": "post_ffw_norm.weight",
    "post_feedforward_layernorm_1": "post_ffw_norm_1.weight",
    "post_feedforward_layernorm_2": "post_ffw_norm_2.weight",
    "pre_feedforward_layernorm_2": "pre_ffw_norm_2.weight", "self_attn.q_norm": "attn_q_norm.weight",
    "self_attn.k_norm": "attn_k_norm.weight", "self_attn.q_proj": "attn_q.weight",
    "self_attn.k_proj": "attn_k.weight", "self_attn.v_proj": "attn_v.weight",
    "self_attn.o_proj": "attn_output.weight", "mlp.gate_proj": "ffn_gate.weight",
    "mlp.up_proj": "ffn_up.weight", "mlp.down_proj": "ffn_down.weight",
    "router.proj": "ffn_gate_inp.weight", "router.scale": "ffn_gate_inp.scale",
    "router.per_expert_scale": "ffn_down_exps.scale", "experts.gate_up_proj": "ffn_gate_up_exps.weight",
    "experts.down_proj": "ffn_down_exps.weight", "dec_scalar": "layer_output_scale.weight",
    "enc_scalar": "enc_layer_output_scale.weight",
}
_GG_SC = {"pre_norm": "self_cond_pre_norm.weight", "gate_proj": "self_cond_gate.weight",
          "up_proj": "self_cond_up.weight", "down_proj": "self_cond_down.weight"}
# the matrices (the rest, the norms and the router, stay float32 in every form)
_MATS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "experts.gate_up_proj", "experts.down_proj")


def q4_0_round(a):
    """a through our Q4_0 and back (ops.quantize_int4: the runtime's int4 weights)."""
    flat = a.reshape(-1, a.shape[-1])
    q, sc = ops.quantize_int4(flat)
    return ops.dequantize_int4(q, sc).reshape(a.shape).astype(np.float32)


def q8_0_round(a):
    """a through Q8_0 (one scale for each 32 values, max |w| / 127) and back."""
    flat = a.reshape(-1, a.shape[-1])
    q, sc = ops.quantize_int8(flat, group=32)
    return (q.reshape(flat.shape[0], -1, 32).astype(np.float32) * sc[:, :, None]).reshape(a.shape)


# weights -> (the round trip of the experts, of the other matrices, of the self-conditioning)
_FORMS = {"q4_0": (q4_0_round, q4_0_round, q4_0_round), "q8_0": (q8_0_round, q8_0_round, q8_0_round),
          "mix": (q4_0_round, None, None)}


class DGemmaRef:
    """The float32 model. All weights stay in memory as float32 (about 103 GB),
    or, with lazy=True, as the mapped bf16 file, each layer converted when it
    runs.

    weights: "bf16" (the checkpoint), "q4_0" (each matrix through our Q4_0
    and back: the runtime's weights in float32 arithmetic; head "int4" the
    token table too), "q8_0" (the same in Q8_0), "mix" (the experts in Q4_0,
    the other matrices bf16), or "gguf:PATH" (a GGUF of the checkpoint,
    dequantized: unsloth's Q4_K_M)."""

    def __init__(self, path, lazy=False, weights="bf16", head="bf16"):
        self.path = path
        self.cfg = Config(json.load(open(os.path.join(path, "config.json"))))
        self.st = Shards(path)
        self.lazy = lazy
        self.weights = weights
        self.gg = None
        if weights.startswith("gguf:"):
            from reclank.formats.gguf import GGUF
            self.gg = GGUF(weights[5:])
        self._layers = {}
        p = DEC + "self_conditioning."
        if self.gg is not None:
            self.embed_w = self.gg.dequant("token_embd.weight")
            self.norm_w = self.gg.dequant("output_norm.weight")
            self.sc = {k: self.gg.dequant(g) for k, g in _GG_SC.items()}
        else:
            self.embed_w = _f32(self.st, DEC + "embed_tokens.weight")
            self.norm_w = _f32(self.st, DEC + "norm.weight")
            self.sc = {k: _f32(self.st, p + k + ".weight") for k in ("pre_norm", "gate_proj", "up_proj", "down_proj")}
            if weights in _FORMS and _FORMS[weights][2] is not None:
                for k in ("gate_proj", "up_proj", "down_proj"):
                    self.sc[k] = _FORMS[weights][2](self.sc[k])
            if weights in _FORMS and head == "int4":
                self.embed_w = q4_0_round(self.embed_w)
        if not lazy:
            for i in range(self.cfg.layers):
                self._layers[i] = self._load(i)

    def _load(self, i):
        if self.gg is not None:
            w = {}
            for n, g in _GG_LAYER.items():
                name = "blk.%d.%s" % (i, g)
                if n == "self_attn.v_proj" and not self.cfg.sliding[i]:
                    w[n] = None
                    continue
                w[n] = self.gg.dequant(name)
            w["dec_scalar"] = float(w["dec_scalar"].reshape(-1)[0])
            w["enc_scalar"] = float(w["enc_scalar"].reshape(-1)[0])
            return w
        p = DEC + "layers.%d." % i
        g = lambda n: _f32(self.st, p + n)
        w = {}
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                  "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2",
                  "self_attn.q_norm", "self_attn.k_norm", "self_attn.q_proj", "self_attn.k_proj",
                  "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"):
            w[n] = g(n + ".weight")
        w["self_attn.v_proj"] = g("self_attn.v_proj.weight") if self.cfg.sliding[i] else None
        w["router.scale"] = g("router.scale")
        w["router.per_expert_scale"] = g("router.per_expert_scale")
        w["experts.gate_up_proj"] = g("experts.gate_up_proj")
        w["experts.down_proj"] = g("experts.down_proj")
        w["dec_scalar"] = float(g("layer_scalar")[0])
        w["enc_scalar"] = float(_f32(self.st, ENC + "layers.%d.layer_scalar" % i)[0])
        if self.weights in _FORMS:
            ex, rest, _ = _FORMS[self.weights]
            for n in _MATS:
                f = ex if n.startswith("experts.") else rest
                if w[n] is not None and f is not None:
                    w[n] = f(w[n])
        return w

    def layer(self, i):
        return self._layers[i] if i in self._layers else self._load(i)

    def cache(self):
        return Cache(self.cfg)

    # ---- the pieces --------------------------------------------------------
    def embed(self, ids):
        return self.embed_w[np.asarray(ids, dtype=np.int64)] * self.cfg.embed_scale

    def _rope(self, i, positions):
        return cos_sin_f32(self.cfg.inv_freq[self.cfg.sliding[i]], positions)

    def _qkv(self, h, w, i, positions):
        """The query, key and value rows of h (t, heads, hd): the norms, the
        rope (the value: the raw key of a global layer, a norm with no weight)."""
        cfg = self.cfg
        hd, nkv = cfg.dims(i)
        t = h.shape[0]
        eps = cfg.eps
        q = (h @ w["self_attn.q_proj"].T).reshape(t, cfg.heads, hd)
        k = (h @ w["self_attn.k_proj"].T).reshape(t, nkv, hd)
        v = k if w["self_attn.v_proj"] is None else (h @ w["self_attn.v_proj"].T).reshape(t, nkv, hd)
        cos, sin = self._rope(i, positions)
        q = rope_mod.apply(ops.rms_norm(q, w["self_attn.q_norm"], eps), cos, sin)
        k = rope_mod.apply(ops.rms_norm(k, w["self_attn.k_norm"], eps), cos, sin)
        v = ops.rms_norm(v, None, eps)
        return q, k, v

    def _attend(self, q, K, V, mask):
        """Softmax attention with scale 1; mask (t, n) True where a key is seen."""
        t, nh, hd = q.shape
        nkv = K.shape[1]
        rep = nh // nkv
        qb = q.reshape(t, nkv, rep, hd).transpose(1, 2, 0, 3)          # nkv rep t hd
        s = np.matmul(qb, K.transpose(1, 2, 0)[:, None])               # nkv rep t n
        s = np.where(mask[None, None], s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        o = np.matmul(p, V.transpose(1, 0, 2)[:, None])                # nkv rep t hd
        return o.transpose(2, 0, 1, 3).reshape(t, nh * hd)

    def _router(self, x, w):
        cfg = self.cfg
        r = ops.rms_norm(x, None, cfg.eps) * w["router.scale"] * np.float32(cfg.hidden ** -0.5)
        logits = r @ w["router.proj"].T
        logits = logits - logits.max(-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(-1, keepdims=True)
        idx = np.argsort(-probs, axis=-1, kind="stable")[:, :cfg.top_k]
        val = np.take_along_axis(probs, idx, -1)
        val = val / val.sum(-1, keepdims=True)
        return val * w["router.per_expert_scale"][idx], idx

    def _moe(self, h, w, val, idx):
        inner = self.cfg.moe_inter
        out = np.zeros_like(h)
        for e in np.unique(idx):
            tok, slot = np.nonzero(idx == e)
            a = h[tok] @ w["experts.gate_up_proj"][e].T
            a = ops.gelu_tanh(a[:, :inner]) * a[:, inner:]
            out[tok] += (a @ w["experts.down_proj"][e].T) * val[tok, slot, None]
        return out

    def _ffn(self, x, w):
        """The second half of a layer: the shared MLP and the experts, both on
        the residual x; returns x + the norms of their sum."""
        eps = self.cfg.eps
        h = ops.rms_norm(x, w["pre_feedforward_layernorm"], eps)
        m = (ops.gelu_tanh(h @ w["mlp.gate_proj"].T) * (h @ w["mlp.up_proj"].T)) @ w["mlp.down_proj"].T
        h1 = ops.rms_norm(m, w["post_feedforward_layernorm_1"], eps)
        val, idx = self._router(x, w)
        h2 = self._moe(ops.rms_norm(x, w["pre_feedforward_layernorm_2"], eps), w, val, idx)
        h2 = ops.rms_norm(h2, w["post_feedforward_layernorm_2"], eps)
        return x + ops.rms_norm(h1 + h2, w["post_feedforward_layernorm"], eps)

    # ---- the encoder -------------------------------------------------------
    def encode(self, ids, cache, hook=None):
        """Run the encoder over ids at the positions after the cache; append
        their keys and values to it. Returns the last hidden state (normed)."""
        cfg = self.cfg
        t = len(ids)
        start = cache.n
        positions = np.arange(start, start + t)
        x = self.embed(ids)
        for i in range(cfg.layers):
            w = self.layer(i)
            h = ops.rms_norm(x, w["input_layernorm"], cfg.eps)
            q, k, v = self._qkv(h, w, i, positions)
            cache.append(i, k, v)
            K, V = cache.k[i], cache.v[i]
            kpos = np.arange(K.shape[0])
            mask = kpos[None, :] <= positions[:, None]
            if cfg.sliding[i]:
                mask &= kpos[None, :] > positions[:, None] - cfg.window
            a = self._attend(q, K, V, mask) @ w["self_attn.o_proj"].T
            x = x + ops.rms_norm(a, w["post_attention_layernorm"], cfg.eps)
            x = self._ffn(x, w) * np.float32(w["enc_scalar"])
            if hook is not None:
                hook("enc.%d" % i, x)
        cache.n = start + t
        return ops.rms_norm(x, self.norm_w, cfg.eps)

    # ---- the decoder -------------------------------------------------------
    def soft_embed(self, sc_logits):
        """softmax(the last step's processed logits) @ the token table * sqrt(hidden)."""
        z = sc_logits - sc_logits.max(-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(-1, keepdims=True)
        return (p @ self.embed_w) * self.cfg.embed_scale

    def decoder_input(self, canvas, sc_logits=None):
        eps = self.cfg.eps
        x = self.embed(canvas)
        if sc_logits is not None:
            s = ops.rms_norm(self.soft_embed(sc_logits), self.sc["pre_norm"], eps)
            s = (ops.gelu_tanh(s @ self.sc["gate_proj"].T) * (s @ self.sc["up_proj"].T)) @ self.sc["down_proj"].T
            x = x + s
        # With no logits the signal is mlp(rmsnorm(0)) = 0 (gelu(0) = 0).
        return ops.rms_norm(x, None, eps)

    def decode(self, canvas, cache, sc_logits=None, hook=None):
        """The decoder over a canvas at the positions after the cache: the raw
        logits (with the soft cap) [canvas, vocab]. The cache is not changed."""
        cfg = self.cfg
        t = len(canvas)
        n = cache.n
        positions = np.arange(n, n + t)
        x = self.decoder_input(canvas, sc_logits)
        for i in range(cfg.layers):
            w = self.layer(i)
            h = ops.rms_norm(x, w["input_layernorm"], cfg.eps)
            q, k, v = self._qkv(h, w, i, positions)
            lo = max(0, n - (cfg.window - 1)) if cfg.sliding[i] else 0
            K = np.concatenate([cache.k[i][lo:n], k]) if n else k
            V = np.concatenate([cache.v[i][lo:n], v]) if n else v
            mask = np.ones((t, K.shape[0]), dtype=bool)
            a = self._attend(q, K, V, mask) @ w["self_attn.o_proj"].T
            x = x + ops.rms_norm(a, w["post_attention_layernorm"], cfg.eps)
            x = self._ffn(x, w) * np.float32(w["dec_scalar"])
            if hook is not None:
                hook("dec.%d" % i, x)
        x = ops.rms_norm(x, self.norm_w, cfg.eps)
        logits = x @ self.embed_w.T
        return np.tanh(logits / cfg.softcap) * cfg.softcap
