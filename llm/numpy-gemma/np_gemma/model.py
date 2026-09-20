"""Gemma 4 12B NumPy forward pass with a KV cache (no PyTorch)."""
from __future__ import annotations

import numpy as np

from . import ops
from . import rope as rope_mod

PREFIX = "model.language_model."


def emit(hook, key, value):
    if hook is not None:
        hook(key, np.asarray(value, dtype=np.float32))


class KVCache:
    """Per-layer key/value history. Layers stay full-length (trimming is a later optimization)."""

    def __init__(self, num_layers):
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def append(self, layer, k, v):
        if self.k[layer] is None:
            self.k[layer] = k
            self.v[layer] = v
        else:
            self.k[layer] = np.concatenate([self.k[layer], k], axis=0)
            self.v[layer] = np.concatenate([self.v[layer], v], axis=0)

    def length(self, layer):
        return 0 if self.k[layer] is None else self.k[layer].shape[0]


class Model:
    def __init__(self, st, cfg):
        self.st = st
        self.cfg = cfg
        self._layers = {}
        self.keep_weights = False

    # ---- weights -----------------------------------------------------------
    def load_layer(self, i):
        if i in self._layers:
            return self._layers[i]
        plan = self.cfg.plan[i]
        p = PREFIX + "layers." + str(i) + "."
        w = {
            "input_layernorm": self.st.get(p + "input_layernorm.weight"),
            "post_attention_layernorm": self.st.get(p + "post_attention_layernorm.weight"),
            "pre_feedforward_layernorm": self.st.get(p + "pre_feedforward_layernorm.weight"),
            "post_feedforward_layernorm": self.st.get(p + "post_feedforward_layernorm.weight"),
            "self_attn.q_proj": self.st.get(p + "self_attn.q_proj.weight"),
            "self_attn.q_norm": self.st.get(p + "self_attn.q_norm.weight"),
            "self_attn.k_proj": self.st.get(p + "self_attn.k_proj.weight"),
            "self_attn.k_norm": self.st.get(p + "self_attn.k_norm.weight"),
            "self_attn.o_proj": self.st.get(p + "self_attn.o_proj.weight"),
            "mlp.gate_proj": self.st.get(p + "mlp.gate_proj.weight"),
            "mlp.up_proj": self.st.get(p + "mlp.up_proj.weight"),
            "mlp.down_proj": self.st.get(p + "mlp.down_proj.weight"),
            "layer_scalar": float(self.st.get(p + "layer_scalar")[0]),
            "self_attn.v_proj": None if plan.k_eq_v else self.st.get(p + "self_attn.v_proj.weight"),
        }
        self._layers[i] = w
        return w

    def free_layer(self, i):
        if not self.keep_weights:
            self._layers.pop(i, None)

    def _rope(self, plan, positions):
        return rope_mod.cos_sin(self.cfg.rope_inv_freq(plan), positions)

    # ---- forward -----------------------------------------------------------
    def embed(self, input_ids):
        rows = [self.st.get_row(PREFIX + "embed_tokens.weight", int(t)) for t in input_ids]
        return np.stack(rows, axis=0) * self.cfg.embed_scale

    def forward(self, input_ids, hook=None, max_layers=None, cache=None, start_pos=0):
        cfg = self.cfg
        x = self.embed(input_ids)
        if start_pos == 0:
            emit(hook, "embed_tokens", x)
            emit(hook, "inputs_embeds", x)
        n = cfg.num_hidden_layers if max_layers is None else min(max_layers, cfg.num_hidden_layers)
        positions = np.arange(start_pos, start_pos + len(input_ids))
        for i in range(n):
            plan = cfg.plan[i]
            w = self.load_layer(i)
            cos, sin = self._rope(plan, positions)
            x = self._decoder_layer(x, w, plan, cos, sin, positions, i, hook, cache)
            self.free_layer(i)
        if n == cfg.num_hidden_layers:
            x = ops.rms_norm(x, self.st.get(PREFIX + "norm.weight"), cfg.rms_norm_eps)
            emit(hook, "norm", x)
            emit(hook, "last_hidden_state", x)
        return x

    def _decoder_layer(self, x, w, plan, cos, sin, positions, i, hook, cache):
        eps = self.cfg.rms_norm_eps
        p = "layers." + str(i) + "."
        residual = x
        h = ops.rms_norm(x, w["input_layernorm"], eps)
        emit(hook, p + "input_layernorm", h)
        h = self._attention(h, w, plan, cos, sin, positions, i, p, hook, cache)
        h = ops.rms_norm(h, w["post_attention_layernorm"], eps)
        emit(hook, p + "post_attention_layernorm", h)
        x = residual + h
        residual = x
        h = ops.rms_norm(x, w["pre_feedforward_layernorm"], eps)
        emit(hook, p + "pre_feedforward_layernorm", h)
        g = ops.linear(h, w["mlp.gate_proj"])
        emit(hook, p + "mlp.gate_proj", g)
        u = ops.linear(h, w["mlp.up_proj"])
        emit(hook, p + "mlp.up_proj", u)
        m = ops.linear(ops.gelu_tanh(g) * u, w["mlp.down_proj"])
        emit(hook, p + "mlp.down_proj", m)
        m = ops.rms_norm(m, w["post_feedforward_layernorm"], eps)
        emit(hook, p + "post_feedforward_layernorm", m)
        x = residual + m
        x = x * w["layer_scalar"]
        emit(hook, p + "out", x)
        return x

    def _attention(self, x, w, plan, cos, sin, positions, i, p, hook, cache):
        eps = self.cfg.rms_norm_eps
        hd = plan.head_dim
        t = x.shape[0]

        q = ops.linear(x, w["self_attn.q_proj"])
        emit(hook, p + "self_attn.q_proj", q)
        q = q.reshape(t, plan.num_q_heads, hd)
        q = ops.rms_norm(q, w["self_attn.q_norm"], eps)
        emit(hook, p + "self_attn.q_norm", q)

        k_raw = ops.linear(x, w["self_attn.k_proj"])
        emit(hook, p + "self_attn.k_proj", k_raw)
        k = k_raw.reshape(t, plan.num_kv_heads, hd)
        k = ops.rms_norm(k, w["self_attn.k_norm"], eps)
        emit(hook, p + "self_attn.k_norm", k)

        if plan.k_eq_v:
            v = ops.rms_norm(k_raw.reshape(t, plan.num_kv_heads, hd), None, eps)
        else:
            v = ops.linear(x, w["self_attn.v_proj"])
            emit(hook, p + "self_attn.v_proj", v)
            v = ops.rms_norm(v.reshape(t, plan.num_kv_heads, hd), None, eps)

        q = rope_mod.apply(q, cos, sin)
        k = rope_mod.apply(k, cos, sin)

        if cache is not None:
            cache.append(i, k, v)
            K, V = cache.k[i], cache.v[i]
        else:
            K, V = k, v

        n_rep = plan.num_q_heads // plan.num_kv_heads
        qr = q.reshape(t, plan.num_kv_heads, n_rep, hd)
        scores = np.einsum("tkrh,skh->tkrs", qr, K, optimize=True)
        kpos = np.arange(K.shape[0])
        mask = kpos[None, :] <= positions[:, None]
        if plan.sliding_window:
            mask &= (positions[:, None] - kpos[None, :]) < plan.sliding_window
        scores = np.where(mask[:, None, None, :], scores, np.float32(-1e30))
        probs = ops.softmax(scores, axis=-1)
        out = np.einsum("tkrs,skh->tkrh", probs, V, optimize=True).reshape(t, plan.q_dim)
        out = ops.linear(out, w["self_attn.o_proj"])
        emit(hook, p + "self_attn.o_proj", out)
        return out

    # ---- output head -------------------------------------------------------
    def logits(self, x, chunk=32768, apply_softcap=True):
        name = PREFIX + "embed_tokens.weight"
        out = np.empty((x.shape[0], self.cfg.vocab_size), dtype=np.float32)
        for start in range(0, self.cfg.vocab_size, chunk):
            stop = min(start + chunk, self.cfg.vocab_size)
            out[:, start:stop] = x @ self.st.get_rows(name, start, stop).T
        if apply_softcap and self.cfg.final_logit_softcapping:
            out = ops.softcap(out, self.cfg.final_logit_softcapping)
        return out

    # ---- generation --------------------------------------------------------
    def generate(self, input_ids, max_new_tokens=1, eos_ids=(), cache_weights=False):
        self.keep_weights = cache_weights
        cache = KVCache(self.cfg.num_hidden_layers)
        ids = list(input_ids)
        x = self.forward(ids, cache=cache)
        nxt = int(np.argmax(self.logits(x[-1:])[0]))
        out = ids + [nxt]
        for _ in range(max_new_tokens - 1):
            if nxt in eos_ids:
                break
            x = self.forward([nxt], cache=cache, start_pos=len(out) - 1)
            nxt = int(np.argmax(self.logits(x)[0]))
            out.append(nxt)
        return out
