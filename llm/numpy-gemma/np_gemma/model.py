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

import os

import numpy as np

from . import ops
from . import rope as rope_mod
from .weight_cache import WeightCache

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

    A global layer keeps the full sequence. A sliding layer keeps the last
    window. The buffers have a fixed size. The code writes in place. Thus the
    cache does not copy the full sequence for each token.
    """

    def __init__(self, cfg, max_len=4096):
        self.cfg = cfg
        self.window = cfg.sliding_window or 0
        self.max_len = max_len
        n = cfg.num_hidden_layers
        self.k = [None] * n
        self.v = [None] * n
        self.base = [0] * n      # absolute position of buffer row 0
        self.end = [0] * n       # absolute position after the last stored row

    def _shape(self, layer, cap):
        plan = self.cfg.plan[layer]
        return (cap, plan.num_kv_heads, plan.head_dim)

    def _alloc(self, layer, cap):
        self.k[layer] = np.empty(self._shape(layer, cap), dtype=np.float32)
        self.v[layer] = np.empty(self._shape(layer, cap), dtype=np.float32)

    def _grow(self, layer, cap):
        old = 0 if self.k[layer] is None else self.k[layer].shape[0]
        nk = np.empty(self._shape(layer, cap), dtype=np.float32)
        nv = np.empty(self._shape(layer, cap), dtype=np.float32)
        if old:
            nk[:old] = self.k[layer][:old]
            nv[:old] = self.v[layer][:old]
        self.k[layer] = nk
        self.v[layer] = nv

    def write(self, layer, start_pos, k, v):
        """Store a block of keys and values. start_pos is the position of k[0]."""
        t = k.shape[0]
        end = start_pos + t
        if self.cfg.plan[layer].is_sliding:
            w = self.window
            if self.k[layer] is None:
                self._alloc(layer, 2 * w)
            # A decode step compacts the buffer when it holds two windows.
            # A prefill block does not compact, because early queries in the
            # block need keys from the start of the block.
            if t == 1 and end - self.base[layer] > 2 * w:
                keep = end - w
                off = keep - self.base[layer]
                rows = self.end[layer] - keep
                if rows > 0:
                    self.k[layer][:rows] = self.k[layer][off:off + rows]
                    self.v[layer][:rows] = self.v[layer][off:off + rows]
                self.base[layer] = keep
            need = end - self.base[layer]
            if need > self.k[layer].shape[0]:
                self._grow(layer, max(need, 2 * w))
            start = start_pos - self.base[layer]
            self.k[layer][start:start + t] = k
            self.v[layer][start:start + t] = v
        else:
            if self.k[layer] is None or self.k[layer].shape[0] < end:
                cap = max(self.max_len, end)
                if self.k[layer] is not None:
                    cap = max(cap, self.k[layer].shape[0] * 2)
                self._grow(layer, cap)
            self.k[layer][start_pos:end] = k
            self.v[layer][start_pos:end] = v
        self.end[layer] = end

    def read(self, layer, end):
        """Return the keys, the values, and the position of the first row."""
        base = self.base[layer]
        return self.k[layer][:end - base], self.v[layer][:end - base], base

    def length(self, layer):
        """Return the number of stored positions in one layer."""
        if self.k[layer] is None:
            return 0
        return self.end[layer] - self.base[layer]

    def truncate(self, n):
        """Keep only the positions before n. Return False when that is not possible.

        A sliding layer drops the oldest rows. If n is before the first row of
        a layer, that layer cannot go back. The caller must then start again.
        """
        for i in range(len(self.k)):
            if self.k[i] is not None and n < self.base[i]:
                return False
        for i in range(len(self.k)):
            if self.k[i] is not None and n < self.end[i]:
                self.end[i] = n
        return True


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
        self._embed_q = None
        self._embed_s = None
        self._norm_w = None
        self._dtype = "f32"
        self._cache = None
        self._cache_write = None
        # The w4a16 checkpoint keeps the packed 4-bit weights and the scales
        # from the quantization-aware training.
        self._w4a16 = (PREFIX + "layers.0.mlp.gate_proj.weight_packed") in st.names()
        self.keep_weights = False
        # The prompt pass uses the int8 GEMM. The GEMM is fastest for a block
        # of about 256 tokens. A longer prompt is cut into blocks of this size.
        # Set NP_GEMMA_PREFILL_CHUNK to change the size.
        self.prefill_chunk = max(1, int(os.environ.get("NP_GEMMA_PREFILL_CHUNK", "256")))
        # The prompt GEMM uses a packed copy of the int8 weights when this is
        # on. The copy is the transpose of the data, so it is the same size.
        self._packed = os.environ.get("NP_GEMMA_PACKED") == "1"

    # ---- weights -----------------------------------------------------------
    def _load_layer(self, i, dtype):
        """Load the weights of one layer. Use the given dtype for the projections.

        dtype "f32" keeps float32 values. dtype "bf16" keeps raw bfloat16
        values. dtype "int8" keeps int8 values. dtype "int4" keeps packed
        4-bit values.
        """
        plan = self.cfg.plan[i]
        p = PREFIX + "layers." + str(i) + "."
        w = {key: self.st.get(p + key + ".weight") for key in _NORM_KEYS}
        if dtype in ("int8", "int4"):
            # Read the quantized weights from the cache. Quantize the weights
            # and write the cache when the cache is not ready.
            quant = ops.quantize_int8 if dtype == "int8" else ops.quantize_int4
            suffix = ".q" if dtype == "int8" else ".q4"
            src_keys = list(_PROJ_KEYS) + ([] if plan.k_eq_v else ["self_attn.v_proj"])
            for key in src_keys:
                src = p + key + ".weight"
                if self._cache is not None:
                    w[key] = (self._cache.read(src + suffix), self._cache.read(src + ".scale"))
                else:
                    if dtype == "int4" and self._w4a16:
                        packed = self.st.get(src + "_packed", dtype=None)
                        scale = ops.bf16_to_f32(self.st.get_bf16(src + "_scale"))
                        parts = ops.convert_w4a16(packed, scale)
                    else:
                        parts = quant(self.st.get(src))
                    if self._cache_write is not None:
                        self._cache_write.write(src + suffix, parts[0])
                        self._cache_write.write(src + ".scale", parts[1])
                    w[key] = parts
            if self._packed:
                for key in src_keys:
                    q, s = w[key]
                    if q.shape[0] % 16 == 0 and q.shape[1] % 16 == 0:
                        w[key] = (q, s, ops.pack_int8_16x16(q))
            if plan.k_eq_v:
                w["self_attn.v_proj"] = None
        else:
            proj_get = self.st.get_bf16 if dtype == "bf16" else self.st.get
            for key in _PROJ_KEYS:
                w[key] = proj_get(p + key + ".weight")
            w["self_attn.v_proj"] = None if plan.k_eq_v else proj_get(p + "self_attn.v_proj.weight")
        w["layer_scalar"] = float(self.st.get(p + "layer_scalar")[0])
        return w

    def load_all(self, dtype="f32"):
        """Load all layers and the embedding table. Keep the data in memory.

        Use dtype "f32" for speed. Use dtype "bf16", "int8", or "int4" for a
        smaller memory use. The int8 and int4 modes read the converted weights
        from a local cache. The first load writes the cache. Later loads read
        it.
        """
        dtype = dtype.lower()
        if dtype not in ("f32", "bf16", "int8", "int4"):
            raise ValueError("dtype must be f32, bf16, int8, or int4")
        if dtype in ("int8", "int4"):
            cache = WeightCache(self.st.path, dtype)
            if cache.ready():
                self._cache = cache
            else:
                cache.open_write()
                self._cache_write = cache
        for i in range(self.cfg.num_hidden_layers):
            self._layers[i] = self._load_layer(i, dtype)
        self._norm_w = self.st.get(PREFIX + "norm.weight")
        self._embed = None
        self._embed_bf16 = None
        self._embed_q = None
        self._embed_s = None
        if dtype == "bf16":
            self._embed_bf16 = self.st.get_bf16(PREFIX + "embed_tokens.weight")
        elif dtype in ("int8", "int4"):
            if dtype == "int4" and self._w4a16:
                # The w4a16 checkpoint does not quantize the embedding table.
                self._embed_bf16 = self.st.get_bf16(PREFIX + "embed_tokens.weight")
            else:
                quant = ops.quantize_int8 if dtype == "int8" else ops.quantize_int4
                suffix = ".q" if dtype == "int8" else ".q4"
                src = PREFIX + "embed_tokens.weight"
                if self._cache is not None:
                    self._embed_q = self._cache.read(src + suffix)
                    self._embed_s = self._cache.read(src + ".scale")
                else:
                    self._embed_q, self._embed_s = quant(self.st.get(src))
                    if self._cache_write is not None:
                        self._cache_write.write(src + suffix, self._embed_q)
                        self._cache_write.write(src + ".scale", self._embed_s)
        else:
            self._embed = self.st.get(PREFIX + "embed_tokens.weight")
        if self._cache_write is not None:
            self._cache_write.close_write()
            self._cache_write = None
            self._cache = WeightCache(self.st.path, dtype)
        self._dtype = dtype
        self.keep_weights = True
        if dtype in ("int8", "int4"):
            # The copy holds all the weights. Drop the mapped bf16 pages.
            self.st.release_pages()
        return self

    def free_all(self):
        """Remove all weights from memory."""
        self._layers.clear()
        self._embed = None
        self._embed_bf16 = None
        self._embed_q = None
        self._embed_s = None
        self._norm_w = None
        self._dtype = "f32"
        self.keep_weights = False

    def load_layer(self, i):
        """Load one layer if it is not in memory. Return the layer."""
        if i not in self._layers:
            self._layers[i] = self._load_layer(i, self._dtype)
        return self._layers[i]

    def free_layer(self, i):
        """Remove one layer from memory. Do nothing when keep_weights is true."""
        if not self.keep_weights:
            self._layers.pop(i, None)

    def linear(self, x, w):
        """Multiply x by W. Use the kernel of the resident dtype."""
        if self._dtype == "int8":
            if len(w) > 2:
                return ops.linear_int8(x, w[0], w[1], w[2])
            return ops.linear_int8(x, w[0], w[1])
        if self._dtype == "int4":
            return ops.linear_int4(x, w[0], w[1])
        if self._dtype == "bf16":
            return ops.linear_bf16(x, w)
        return ops.linear(x, w)

    def _rope(self, plan, positions):
        """Return the cosine and sine tables for one layer at the given positions."""
        return rope_mod.cos_sin(self.cfg.rope_inv_freq(plan), positions)

    # ---- forward -----------------------------------------------------------
    def embed(self, input_ids):
        """Return the input embeddings for the token ids. Multiply by the embedding scale."""
        ids = np.asarray(input_ids, dtype=np.int64)
        if self._embed_q is not None:
            if self._dtype == "int4":
                w = ops.dequantize_int4(self._embed_q[ids], self._embed_s[ids])
            else:
                group = ops.int8_group(self._embed_q, self._embed_s)
                w = self._embed_q[ids].astype(np.float32) * np.repeat(self._embed_s[ids], group, axis=1)
            return w * self.cfg.embed_scale
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
            start = positions[0]
            cache.write(i, start, k, v)
            K, V, base = cache.read(i, start + t)
        else:
            K, V, base = k, v, positions[0]

        # The attention scale is 1.0. Do not divide by sqrt(head_dim).
        n_rep = plan.num_q_heads // plan.num_kv_heads
        nk = plan.num_kv_heads
        n = K.shape[0]
        # Use a batched matrix multiply. The code makes one matrix for each
        # group of query heads. matmul is faster than einsum here, because
        # einsum looks for a contraction path at each call.
        qb = q.reshape(t, nk, n_rep, hd).transpose(1, 0, 2, 3).reshape(nk, t * n_rep, hd)
        kb = K.transpose(1, 2, 0)
        scores = np.matmul(qb, kb).reshape(nk, t, n_rep, n)
        # A decode step needs no mask. The cache holds only earlier positions.
        # A sliding layer needs the mask when the buffer holds more keys than
        # the window. A prompt of many tokens always needs the causal mask.
        if t > 1 or (plan.sliding_window and n > plan.sliding_window):
            kpos = base + np.arange(n)
            mask = kpos[None, :] <= positions[:, None]
            if plan.sliding_window:
                mask &= (positions[:, None] - kpos[None, :]) < plan.sliding_window
            scores = np.where(mask[None, :, None, :], scores, np.float32(-1e30))
        probs = ops.softmax(scores, axis=-1)
        vb = V.transpose(1, 0, 2)
        out = np.matmul(probs.reshape(nk, t * n_rep, n), vb)
        out = out.reshape(nk, t, n_rep, hd).transpose(1, 0, 2, 3).reshape(t, plan.q_dim)
        out = self.linear(out, w["self_attn.o_proj"])
        emit(hook, p + "self_attn.o_proj", out)
        return out

    # ---- output head -------------------------------------------------------
    def logits(self, x, chunk=32768, apply_softcap=True):
        """Return the logits for the hidden states x.

        Use the embedding table. The embeddings are tied to the output head.
        Apply the softcap when requested.
        """
        if self._embed_q is not None:
            if self._dtype == "int4":
                out = ops.linear_int4(x, self._embed_q, self._embed_s)
            else:
                out = ops.linear_int8(x, self._embed_q, self._embed_s)
        elif self._embed_bf16 is not None:
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
    def prefill(self, ids, cache, start=0):
        """Run the prompt. Use blocks to keep the GEMM in its fast range.

        The key and value cache holds the earlier blocks. The result is the
        same as one forward pass over the full prompt. start is the position
        of ids[0]. Use it to add tokens to a cache that already has data.
        """
        chunk = self.prefill_chunk
        x = None
        for off in range(0, len(ids), chunk):
            x = self.forward(ids[off:off + chunk], cache=cache, start_pos=start + off)
        return x

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
        ids = list(input_ids)
        cache = KVCache(self.cfg, max_len=len(ids) + max(max_new_tokens, 1) + 4)
        x = self.prefill(ids, cache)
        nxt = int(np.argmax(self.logits(x[-1:])[0]))
        out = ids + [nxt]
        for _ in range(max_new_tokens - 1):
            if nxt in eos_ids:
                break
            x = self.forward([nxt], cache=cache, start_pos=len(out) - 1)
            nxt = int(np.argmax(self.logits(x)[0]))
            out.append(nxt)
        return out


class Session:
    """Keep the key and value cache between the turns of a chat.

    The class holds the token ids that are in the cache. A new turn sends the
    full conversation. The class finds the common prefix and runs the forward
    pass only for the new tokens. Thus a chat does not read the history again.

    Use reset() to start a new conversation.
    """

    def __init__(self, model, max_len=8192):
        self.model = model
        self.max_len = max_len
        self.cache = None
        self.ids = []
        self._x = None
        # The count of tokens that went through the model in the last turn.
        self.prefilled = 0
        self.reset()

    def reset(self):
        """Start a new conversation. Remove the keys and the values."""
        self.cache = KVCache(self.model.cfg, max_len=self.max_len)
        self.ids = []
        self._x = None

    def _common(self, ids):
        """Return the count of the first ids that are already in the cache."""
        n = min(len(ids), len(self.ids))
        i = 0
        while i < n and ids[i] == self.ids[i]:
            i += 1
        return i

    def prefill(self, ids):
        """Put the tokens in the cache. Run the forward pass for the new tokens."""
        ids = list(ids)
        common = self._common(ids)
        if common < len(self.ids):
            # The history changed at position common. Drop the rows after it.
            if self.cache.truncate(common):
                self.ids = self.ids[:common]
            else:
                self.reset()
                common = 0
        new = ids[common:]
        self.prefilled = len(new)
        if new:
            self._x = self.model.prefill(new, self.cache, start=common)
            self.ids = ids
        else:
            self._x = None
        return common

    def generate(self, ids, max_new_tokens=1, eos_ids=()):
        """Run the new tokens. Then generate tokens with greedy selection."""
        ids = list(ids)
        self.prefill(ids)
        if self._x is None:
            # The prompt is the same as the cache. Run the last token again.
            self._x = self.model.forward(ids[-1:], cache=self.cache,
                                         start_pos=len(ids) - 1)
        nxt = int(np.argmax(self.model.logits(self._x[-1:])[0]))
        out = ids + [nxt]
        pos = len(ids)
        for _ in range(max_new_tokens - 1):
            if nxt in eos_ids:
                break
            x = self.model.forward([nxt], cache=self.cache, start_pos=pos)
            self.ids.append(nxt)
            self._x = x
            pos += 1
            nxt = int(np.argmax(self.model.logits(x)[0]))
            out.append(nxt)
        return out
