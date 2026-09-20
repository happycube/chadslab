"""Run the Gemma 4 12B model with NumPy only.

The model has 48 decoder layers. This module gives two classes:
    KVCache  Store the keys and values of each layer.
    Model    Load the weights and run the model.

Two weight modes are available:
    f32   Keep the weights in float32 format. This mode is fast. It uses about
          70 GB of memory.
    bf16  Keep the weights in bfloat16 format. Convert the weights during each
          multiply. This mode uses about 24 GB of memory. It is slower.

Call load_all() one time. Then run many prompts. Use dtype "f32" for speed.
Use dtype "bf16" when memory is small.
"""
from __future__ import annotations

import numpy as np

from . import ops
from . import rope as rope_mod

PREFIX = "model.language_model."

# These tensors are small. Keep them in float32 format.
_NORM_KEYS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)
# These tensors are large. Keep them in float32 or bfloat16 format.
_PROJ_KEYS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def emit(hook, key, value):
    """Send one intermediate tensor to the hook. Do nothing when hook is None."""
    if hook is not None:
        hook(key, np.asarray(value, dtype=np.float32))


class KVCache:
    """Store the key and value tensors of each layer.

    The cache keeps the full sequence. A later version can cut the sequence to
    the sliding window.
    """

    def __init__(self, num_layers):
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def append(self, layer, k, v):
        """Add one block of keys and values to one layer."""
        if self.k[layer] is None:
            self.k[layer] = k
            self.v[layer] = v
        else:
            self.k[layer] = np.concatenate([self.k[layer], k], axis=0)
            self.v[layer] = np.concatenate([self.v[layer], v], axis=0)

    def length(self, layer):
        """Return the number of cached positions in one layer."""
        return 0 if self.k[layer] is None else self.k[layer].shape[0]


class Model:
    """Load the weights and run the Gemma 4 12B model.

    The class keeps the weights in a dictionary. The key is the layer number.
    Set keep_weights to True to hold the weights after each forward pass.
    """

    def __init__(self, st, cfg):
        self.st = st
        self.cfg = cfg
        self._layers = {}
        self._embed = None
        self._embed_bf16 = None
        self._norm_w = None
        self._bf16 = False
        self.keep_weights = False

    # ---- weights -----------------------------------------------------------
    def _load_layer(self, i, dtype):
        """Load the weights of one layer. Use the given dtype for the projections."""
        plan = self.cfg.plan[i]
        p = PREFIX + "layers." + str(i) + "."
        proj_get = self.st.get_bf16 if dtype == "bf16" else self.st.get
        w = {key: self.st.get(p + key + ".weight") for key in _NORM_KEYS}
        for key in _PROJ_KEYS:
            w[key] = proj_get(p + key + ".weight")
        w["self_attn.v_proj"] = None if plan.k_eq_v else proj_get(p + "self_attn.v_proj.weight")
        w["layer_scalar"] = float(self.st.get(p + "layer_scalar")[0])
        return w

    def load_all(self, dtype="f32"):
        """Load all layers and the embedding table. Keep the data in memory.

        Use dtype "f32" for speed. Use dtype "bf16" for a smaller memory use.
        """
        dtype = dtype.lower()
        if dtype not in ("f32", "bf16"):
            raise ValueError("dtype must be f32 or bf16")
        for i in range(self.cfg.num_hidden_layers):
            self._layers[i] = self._load_layer(i, dtype)
        self._norm_w = self.st.get(PREFIX + "norm.weight")
        if dtype == "bf16":
            self._embed_bf16 = self.st.get_bf16(PREFIX + "embed_tokens.weight")
            self._embed = None
        else:
            self._embed = self.st.get(PREFIX + "embed_tokens.weight")
            self._embed_bf16 = None
        self._bf16 = dtype == "bf16"
        self.keep_weights = True
        return self

    def free_all(self):
        """Remove all weights from memory."""
        self._layers.clear()
        self._embed = None
        self._embed_bf16 = None
        self._norm_w = None
        self._bf16 = False
        self.keep_weights = False

    def load_layer(self, i):
        """Load one layer if it is not in memory. Return the layer."""
        if i not in self._layers:
            self._layers[i] = self._load_layer(i, "bf16" if self._bf16 else "f32")
        return self._layers[i]

    def free_layer(self, i):
        """Remove one layer from memory. Do nothing when keep_weights is true."""
        if not self.keep_weights:
            self._layers.pop(i, None)

    def linear(self, x, w):
        """Multiply x by W. Use the bfloat16 function when the weights are bfloat16."""
        return ops.linear_bf16(x, w) if self._bf16 else ops.linear(x, w)

    def _rope(self, plan, positions):
        """Return the cosine and sine tables for one layer at the given positions."""
        return rope_mod.cos_sin(self.cfg.rope_inv_freq(plan), positions)

    # ---- forward -----------------------------------------------------------
    def embed(self, input_ids):
        """Return the input embeddings for the token ids. Multiply by the embedding scale."""
        ids = np.asarray(input_ids, dtype=np.int64)
        if self._embed_bf16 is not None:
            return ops.bf16_to_f32(self._embed_bf16[ids]) * self.cfg.embed_scale
        if self._embed is not None:
            return self._embed[ids] * self.cfg.embed_scale
        rows = [self.st.get_row(PREFIX + "embed_tokens.weight", int(t)) for t in input_ids]
        return np.stack(rows, axis=0) * self.cfg.embed_scale

    def forward(self, input_ids, hook=None, max_layers=None, cache=None, start_pos=0):
        """Run the forward pass. Return the final hidden states.

        Set max_layers to stop after a number of layers. Use this option for a
        quick test.
        """
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
            norm_w = self._norm_w if self._norm_w is not None else self.st.get(PREFIX + "norm.weight")
            x = ops.rms_norm(x, norm_w, cfg.rms_norm_eps)
            emit(hook, "norm", x)
            emit(hook, "last_hidden_state", x)
        return x

    def _decoder_layer(self, x, w, plan, cos, sin, positions, i, hook, cache):
        """Run one decoder layer. Use four normalization steps and two residual adds."""
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
        g = self.linear(h, w["mlp.gate_proj"])
        emit(hook, p + "mlp.gate_proj", g)
        u = self.linear(h, w["mlp.up_proj"])
        emit(hook, p + "mlp.up_proj", u)
        m = self.linear(ops.gelu_tanh(g) * u, w["mlp.down_proj"])
        emit(hook, p + "mlp.down_proj", m)
        m = ops.rms_norm(m, w["post_feedforward_layernorm"], eps)
        emit(hook, p + "post_feedforward_layernorm", m)
        x = residual + m
        x = x * w["layer_scalar"]
        emit(hook, p + "out", x)
        return x

    def _attention(self, x, w, plan, cos, sin, positions, i, p, hook, cache):
        """Run the attention part of one layer.

        For a sliding layer, mask keys that are older than the window. For a
        global layer, use the raw key projection for the value.
        """
        eps = self.cfg.rms_norm_eps
        hd = plan.head_dim
        t = x.shape[0]

        q = self.linear(x, w["self_attn.q_proj"])
        emit(hook, p + "self_attn.q_proj", q)
        q = q.reshape(t, plan.num_q_heads, hd)
        q = ops.rms_norm(q, w["self_attn.q_norm"], eps)
        emit(hook, p + "self_attn.q_norm", q)

        k_raw = self.linear(x, w["self_attn.k_proj"])
        emit(hook, p + "self_attn.k_proj", k_raw)
        k = k_raw.reshape(t, plan.num_kv_heads, hd)
        k = ops.rms_norm(k, w["self_attn.k_norm"], eps)
        emit(hook, p + "self_attn.k_norm", k)

        if plan.k_eq_v:
            # The global layers have no v_proj. Use the raw key data. Apply RMSNorm.
            v = ops.rms_norm(k_raw.reshape(t, plan.num_kv_heads, hd), None, eps)
        else:
            v = self.linear(x, w["self_attn.v_proj"])
            emit(hook, p + "self_attn.v_proj", v)
            v = ops.rms_norm(v.reshape(t, plan.num_kv_heads, hd), None, eps)

        q = rope_mod.apply(q, cos, sin)
        k = rope_mod.apply(k, cos, sin)

        if cache is not None:
            cache.append(i, k, v)
            K, V = cache.k[i], cache.v[i]
        else:
            K, V = k, v

        # The attention scale is 1.0. Do not divide by sqrt(head_dim).
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
        out = self.linear(out, w["self_attn.o_proj"])
        emit(hook, p + "self_attn.o_proj", out)
        return out

    # ---- output head -------------------------------------------------------
    def logits(self, x, chunk=32768, apply_softcap=True):
        """Return the logits for the hidden states x.

        Use the embedding table. The embeddings are tied to the output head.
        Apply the softcap when requested.
        """
        if self._embed_bf16 is not None:
            out = ops.linear_bf16(x, self._embed_bf16, chunk=ops.LINEAR_BF16_CHUNK)
        elif self._embed is not None:
            out = x @ self._embed.T
        else:
            name = PREFIX + "embed_tokens.weight"
            out = np.empty((x.shape[0], self.cfg.vocab_size), dtype=np.float32)
            for start in range(0, self.cfg.vocab_size, chunk):
                stop = min(start + chunk, self.cfg.vocab_size)
                out[:, start:stop] = x @ self.st.get_rows(name, start, stop).T
        if apply_softcap and self.cfg.final_logit_softcapping:
            out = ops.softcap(out, self.cfg.final_logit_softcapping)
        return out

    # ---- generation --------------------------------------------------------
    def generate(self, input_ids, max_new_tokens=1, eos_ids=(), cache_weights=False,
                 load_all=False, dtype="f32"):
        """Generate tokens with greedy selection.

        Step 1: run the prompt one time. Step 2: select the most probable token.
        Step 3: add the token to the cache. Repeat step 2 and step 3.
        Stop at an end token or at max_new_tokens.
        """
        if load_all:
            self.load_all(dtype=dtype)
        self.keep_weights = self.keep_weights or cache_weights
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
