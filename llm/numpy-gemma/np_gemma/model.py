"""Run the Gemma 4 12B model with NumPy only.

The model has 48 decoder layers. This module gives two classes:
    KVCache  Store the keys and values of each layer.
    Model    Load the weights and run the model.

Four weight modes are available:
    f32   Keep float32 weights. This mode uses about 70 GB of memory.
    bf16  Keep bfloat16 weights. This mode uses about 24 GB of memory.
    int8  Keep int8 weights. This mode uses about 12 GB of memory.
    int4  Keep packed 4-bit weights. This mode uses about 9 GB of memory.

For int4, reuse packed source weights when the input format supports them.
Otherwise, quantize the source weights during loading. Call load_all() once,
then run many prompts.
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
# The extra normalization tensors of the mixture-of-experts block.
_MOE_NORM_KEYS = (
    "post_feedforward_layernorm_1",
    "post_feedforward_layernorm_2",
    "pre_feedforward_layernorm_2",
)


# The attention of one query over the float cache. "c" uses the C kernel,
# which the program of a decode step also uses. "numpy" uses the batched
# matrix product of the prompt path.
_F32_ATTN_C = os.environ.get("NP_GEMMA_F32_ATTN", "c") != "numpy"

# Run a decode step as one program in C. Set NP_GEMMA_PROGRAM=0 for the Python
# loop over the layers. The program needs NP_GEMMA_F32_ATTN=c for the float
# cache, so "numpy" also turns it off.
_PROGRAM = os.environ.get("NP_GEMMA_PROGRAM", "1") != "0" and _F32_ATTN_C


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
        self.kq = [None] * n     # int16 copy of k for the fused attention
        self.ks = [None] * n     # one float32 scale for each group of 32
        self.vq = [None] * n     # int16 copy of v for the fused attention
        self.vs = [None] * n
        self.base = [0] * n      # absolute position of buffer row 0
        self.end = [0] * n       # absolute position after the last stored row
        self._qc_on = [False] * n
        self.attn_min = ops.ATTN_MIN
        # The int16 copy is only useful to the fused attention. The float path
        # never reads it, so do not build it and do not spend the memory.
        self.attn_on = ops.attn_ready()

    def _shape(self, layer, cap):
        plan = self.cfg.plan[layer]
        return (cap, plan.num_kv_heads, plan.head_dim)

    def _shape_q(self, layer, cap):
        plan = self.cfg.plan[layer]
        return (cap, plan.num_kv_heads, plan.head_dim)

    def _shape_s(self, layer, cap):
        plan = self.cfg.plan[layer]
        return (cap, plan.num_kv_heads, plan.head_dim // 32)

    def _alloc_q(self, layer, cap):
        self.kq[layer] = np.empty(self._shape_q(layer, cap), dtype=np.int16)
        self.ks[layer] = np.empty(self._shape_s(layer, cap), dtype=np.float32)
        self.vq[layer] = np.empty(self._shape_q(layer, cap), dtype=np.int16)
        self.vs[layer] = np.empty(self._shape_s(layer, cap), dtype=np.float32)

    def _alloc(self, layer, cap):
        self.k[layer] = np.empty(self._shape(layer, cap), dtype=np.float32)
        self.v[layer] = np.empty(self._shape(layer, cap), dtype=np.float32)
        if self.attn_on:
            self._alloc_q(layer, cap)

    def _grow(self, layer, cap):
        old = 0 if self.k[layer] is None else self.k[layer].shape[0]
        nk = np.empty(self._shape(layer, cap), dtype=np.float32)
        nv = np.empty(self._shape(layer, cap), dtype=np.float32)
        if old:
            nk[:old] = self.k[layer][:old]
            nv[:old] = self.v[layer][:old]
        self.k[layer] = nk
        self.v[layer] = nv
        if not self.attn_on:
            return
        ok = 0 if self.kq[layer] is None else self.kq[layer].shape[0]
        qk = np.empty(self._shape_q(layer, cap), dtype=np.int16)
        qks = np.empty(self._shape_s(layer, cap), dtype=np.float32)
        qv = np.empty(self._shape_q(layer, cap), dtype=np.int16)
        qvs = np.empty(self._shape_s(layer, cap), dtype=np.float32)
        if ok:
            qk[:ok] = self.kq[layer][:ok]
            qks[:ok] = self.ks[layer][:ok]
            qv[:ok] = self.vq[layer][:ok]
            qvs[:ok] = self.vs[layer][:ok]
        self.kq[layer] = qk
        self.ks[layer] = qks
        self.vq[layer] = qv
        self.vs[layer] = qvs

    def prepare(self, layer, start_pos, t):
        """Make room for t rows at start_pos. Return the buffer row of start_pos.

        A sliding layer drops its oldest rows here, and a buffer grows here.
        write calls this first. The program of a decode step (np_gemma.program)
        calls it before the step and then writes the rows in C.
        """
        end = start_pos + t
        if self.cfg.plan[layer].is_sliding:
            w = self.window
            if self.k[layer] is None:
                self._alloc(layer, 2 * w)
            # Drop the oldest rows when the buffer holds more than two
            # windows. A query at position p sees back to p - window + 1, so
            # the first query of the new block sees back to
            # start_pos - window + 1. Every row before that is hidden for the
            # whole block. The buffer then holds about the window and the
            # block, and it does not grow with the context. A decode step and
            # a prompt block both compact. The old code compacted only for a
            # decode step, so a prompt block grew the buffer to the full
            # sequence.
            if start_pos - self.base[layer] > 2 * w:
                keep = start_pos - w + 1
                off = keep - self.base[layer]
                rows = self.end[layer] - keep
                if rows > 0:
                    self.k[layer][:rows] = self.k[layer][off:off + rows]
                    self.v[layer][:rows] = self.v[layer][off:off + rows]
                    if self._qc_on[layer]:
                        self.kq[layer][:rows] = self.kq[layer][off:off + rows]
                        self.ks[layer][:rows] = self.ks[layer][off:off + rows]
                        self.vq[layer][:rows] = self.vq[layer][off:off + rows]
                        self.vs[layer][:rows] = self.vs[layer][off:off + rows]
                self.base[layer] = keep
            need = end - self.base[layer]
            if need > self.k[layer].shape[0]:
                self._grow(layer, max(need, 2 * w))
        else:
            if self.k[layer] is None or self.k[layer].shape[0] < end:
                cap = max(self.max_len, end)
                if self.k[layer] is not None:
                    cap = max(cap, self.k[layer].shape[0] * 2)
                self._grow(layer, cap)
        return start_pos - self.base[layer]

    def write(self, layer, start_pos, k, v):
        """Store a block of keys and values. start_pos is the position of k[0]."""
        t = k.shape[0]
        end = start_pos + t
        start = self.prepare(layer, start_pos, t)
        self.k[layer][start:start + t] = k
        self.v[layer][start:start + t] = v
        self.end[layer] = end
        # Keep an int16 copy for the fused attention of a decode step. Build it
        # only when the cache is long enough that the fused path pays for the
        # work of the quantization.
        if self.attn_on and end - self.base[layer] >= self.attn_min:
            if not self._qc_on[layer]:
                self._quantize_all(layer)
                self._qc_on[layer] = True
            else:
                self._store_qc(layer, start_pos - self.base[layer], k, v)

    def _store_qc(self, layer, start, k, v):
        """Store the int16 copy of a block of keys and values."""
        t = k.shape[0]
        plan = self.cfg.plan[layer]
        g = plan.head_dim // 32
        nkv = plan.num_kv_heads
        kq, ks = ops.quantize_i16(k.reshape(t, nkv, g, 32))
        vq, vs = ops.quantize_i16(v.reshape(t, nkv, g, 32))
        self.kq[layer][start:start + t] = kq.reshape(t, nkv, g * 32)
        self.ks[layer][start:start + t] = ks
        self.vq[layer][start:start + t] = vq.reshape(t, nkv, g * 32)
        self.vs[layer][start:start + t] = vs

    def read(self, layer, end):
        """Return the keys, the values, and the position of the first row."""
        base = self.base[layer]
        return self.k[layer][:end - base], self.v[layer][:end - base], base

    def _quantize_all(self, layer):
        """Quantize every stored row of one layer from the float32 copy."""
        n = self.end[layer] - self.base[layer]
        if n > 0:
            self._store_qc(layer, 0, self.k[layer][:n], self.v[layer][:n])

    def qc_ready(self, layer):
        """Return True when the int16 copy of one layer is ready."""
        return self._qc_on[layer]

    def read_qc(self, layer, end):
        """Return the int16 keys and values, their scales, and the position of
        row 0."""
        base = self.base[layer]
        n = end - base
        return (self.kq[layer][:n], self.ks[layer][:n],
                self.vq[layer][:n], self.vs[layer][:n], base)

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
        self._embed_q6k = None
        self._embed_q6k_bytes = None
        self._norm_w = None
        self._dtype = "f32"
        self._cache = None
        self._cache_write = None
        # The cosine and sine tables of the rope, by layer type and position.
        self._rope_cache = {}
        # The w4a16 checkpoint keeps the packed 4-bit weights and the scales
        # from the quantization-aware training.
        self._w4a16 = (PREFIX + "layers.0.mlp.gate_proj.weight_packed") in st.names()
        # A GGUF source gives the quantized data directly. Do not build the
        # on-disk weight cache for it.
        self._use_cache = bool(getattr(st, "use_cache", True))
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
        if self.cfg.enable_moe_block:
            for key in _MOE_NORM_KEYS:
                w[key] = self.st.get(p + key + ".weight")
        if dtype in ("int8", "int4"):
            # Read the runtime weights from the cache when available. Otherwise,
            # reuse or repack source int4 data, or quantize unquantized weights.
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
                    elif dtype == "int4" and hasattr(self.st, "int4_packed"):
                        # The source already gives the int4 data in the runtime
                        # layout. Do not convert the data to float32 and
                        # quantize it again.
                        parts = self.st.int4_packed(src)
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
        if self.cfg.enable_moe_block:
            # The router tensors stay in float32. They are small.
            w["router.proj"] = self.st.get(p + "router.proj.weight")
            w["router.scale"] = self.st.get(p + "router.scale")
            w["router.per_expert_scale"] = self.st.get(p + "router.per_expert_scale")
            for key in ("experts.gate_up_proj", "experts.down_proj"):
                w[key] = self._load_expert(p + key, dtype)
        return w

    def _load_expert(self, src, dtype):
        """Load one 3-D expert tensor.

        The int4 mode reuses packed source data when the reader supports it.
        Otherwise, it quantizes the source values. Int8 also quantizes the
        source values. Float modes keep their source precision.
        """
        n = self.cfg.num_experts
        if dtype == "int4" and hasattr(self.st, "int4_row_slice"):
            if self._cache is not None:
                return (self._cache.read(src + ".q4"), self._cache.read(src + ".scale"))
            parts = self.st.int4_row_slice(src, 0, n)
            if self._cache_write is not None:
                self._cache_write.write(src + ".q4", parts[0])
                self._cache_write.write(src + ".scale", parts[1])
            return parts
        if dtype in ("int8", "int4"):
            suffix = ".q" if dtype == "int8" else ".q4"
            if self._cache is not None:
                return (self._cache.read(src + suffix), self._cache.read(src + ".scale"))
            arr = self.st.get(src)
            flat = arr.reshape(-1, arr.shape[-1])
            quant = ops.quantize_int8 if dtype == "int8" else ops.quantize_int4
            q, s = quant(flat)
            parts = (q.reshape(arr.shape[:-1] + (q.shape[-1],)),
                     s.reshape(arr.shape[:-1] + (s.shape[-1],)))
            if self._cache_write is not None:
                self._cache_write.write(src + suffix, parts[0])
                self._cache_write.write(src + ".scale", parts[1])
            return parts
        if dtype == "bf16":
            return self.st.get_bf16(src)
        return self.st.get(src)

    def _is_q6k_embed(self, hf_name):
        """Return True when the source table is Q6_K.

        Only a GGUF file sets keep_embedding_bf16. The QAT GGUFs keep the tied
        output head in Q6_K.
        """
        if not getattr(self.st, "keep_embedding_bf16", False):
            return False
        try:
            return self.st.dtype(hf_name) == "Q6_K"
        except (KeyError, AttributeError, ValueError):
            return False

    def load_all(self, dtype="f32"):
        """Load all layers and the embedding table. Keep the data in memory.

        Use dtype "f32", "bf16", "int8", or "int4" to select the weight
        format. Int8 and int4 use the local weight cache when the source allows
        it. A cache miss writes the runtime representation; later loads reuse
        it. Int4 readers can reuse packed source weights without quantizing
        them again.
        """
        dtype = dtype.lower()
        if dtype not in ("f32", "bf16", "int8", "int4"):
            raise ValueError("dtype must be f32, bf16, int8, or int4")
        if dtype in ("int8", "int4") and self._use_cache:
            # The int4 layout changed to the block layout of Q4_0. Use a new
            # cache key.
            cache = WeightCache(self.st.path, dtype, extra="blk" if dtype == "int4" else "")
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
        self._embed_q6k = None
        self._embed_q6k_bytes = None
        src = PREFIX + "embed_tokens.weight"
        keep_head = getattr(self.st, "keep_embedding_bf16", False)
        if dtype in ("bf16", "int8", "int4") and self._is_q6k_embed(src):
            # The file keeps the tied output head in Q6_K. Hold the blocks in
            # place. The load step then does no dequantize of the table, and
            # the kernel reads 6.05 bits for each weight.
            self._embed_q6k = self.st.q6k_blocks(src)
            self._embed_q6k_bytes = self.st.q6k_bytes(src)
        elif dtype == "bf16":
            self._embed_bf16 = self.st.get_bf16(src)
        elif dtype == "int4" and (self._w4a16 or keep_head):
            # The source keeps the embedding at a higher precision. Do not
            # quantize the tied output head to 4 bits.
            self._embed_bf16 = self.st.get_bf16(src)
        elif dtype in ("int8", "int4"):
            quant = ops.quantize_int8 if dtype == "int8" else ops.quantize_int4
            suffix = ".q" if dtype == "int8" else ".q4"
            if self._cache is not None:
                self._embed_q = self._cache.read(src + suffix)
                self._embed_s = self._cache.read(src + ".scale")
            else:
                self._embed_q, self._embed_s = quant(self.st.get(src))
                if self._cache_write is not None:
                    self._cache_write.write(src + suffix, self._embed_q)
                    self._cache_write.write(src + ".scale", self._embed_s)
        else:
            self._embed = self.st.get(src)
        if self._cache_write is not None:
            self._cache_write.close_write()
            self._cache_write = None
            self._cache = WeightCache(self.st.path, dtype, extra="blk" if dtype == "int4" else "")
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
        self._embed_q6k = None
        self._embed_q6k_bytes = None
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

    # ---- mixture of experts ------------------------------------------------
    def _router(self, x, w):
        """Return the expert weights and indices for each token.

        The router reads the residual. The router RMSNorm has no weight. The
        softmax uses float32.
        """
        eps = self.cfg.rms_norm_eps
        if x.shape[0] == 1 and ops.router_ready():
            return ops.router(x, w["router.scale"], w["router.proj"],
                              w["router.per_expert_scale"], self.cfg.top_k_experts,
                              eps, self.cfg.hidden_size ** -0.5)
        if self._dtype == "int4" and ops.mt_ready(x.shape[0]) and ops.router_ready():
            # A small group: the steps of the fused router for each token, in
            # one call. A matrix product over the group sums in another order
            # and can select another expert.
            return ops.router_mt(x, w["router.scale"], w["router.proj"],
                                 w["router.per_expert_scale"], self.cfg.top_k_experts,
                                 eps, self.cfg.hidden_size ** -0.5)
        r = ops.rms_norm(x, None, eps)
        r = r * w["router.scale"] * (self.cfg.hidden_size ** -0.5)
        logits = ops.linear(r, w["router.proj"])
        probs = ops.softmax(logits.astype(np.float32), axis=-1)
        val, idx = ops.topk_k(probs, self.cfg.top_k_experts)
        val = val / val.sum(axis=-1, keepdims=True)
        val = val * w["router.per_expert_scale"][idx]
        return val, idx

    def _moe(self, h, w, val, idx):
        """Run the selected experts. Return the sum with the router weights.

        The code groups the tokens by expert. Thus one expert runs one matrix
        for all of its tokens.
        """
        if self._dtype == "int4" and h.shape[0] == 1 and ops.int4_moe_ready():
            return self._moe_one_token(h, w, val, idx)
        if self._dtype == "int4" and ops.mt_ready(h.shape[0]):
            return self._moe_mt(h, w, val, idx)
        if self._dtype == "int4" and h.shape[0] >= 2 and ops.moe_prompt_ready():
            # One parallel region covers every expert of the layer. The
            # activations are int8, int16, or float32 (NP_GEMMA_INT4_Q8).
            return ops.moe_prompt(h, w["experts.gate_up_proj"],
                                  w["experts.down_proj"], val, idx,
                                  self.cfg.moe_intermediate_size)
        inner = self.cfg.moe_intermediate_size
        out = np.zeros_like(h)
        gu = w["experts.gate_up_proj"]
        dn = w["experts.down_proj"]
        packed = self._dtype in ("int8", "int4")
        # Visit the selected experts only. A decode step selects eight experts.
        # The scan over all 128 experts costs a large part of the layer time.
        for e in np.unique(idx):
            e = int(e)
            tok, slot = np.nonzero(idx == e)
            if tok.size == 0:
                continue
            xe = h[tok]
            gu_e = (gu[0][e], gu[1][e]) if packed else gu[e]
            dn_e = (dn[0][e], dn[1][e]) if packed else dn[e]
            act = self.linear(xe, gu_e)
            gate = act[:, :inner]
            up = act[:, inner:]
            act = ops.gelu_tanh(gate) * up
            de = self.linear(act, dn_e)
            out[tok] += de * val[tok, slot, None]
        return out

    def _moe_one_token(self, h, w, val, idx):
        """Run the selected experts for one token with the fused kernel.

        One call serves all of the selected experts. The kernel starts one
        thread team for the whole layer. The result matches the grouped code,
        because the experts run in the same sorted order.
        """
        inner = self.cfg.moe_intermediate_size
        gu_q, gu_s = w["experts.gate_up_proj"]
        dn_q, dn_s = w["experts.down_proj"]
        ids = np.unique(idx).astype(np.int32)
        cols = h.shape[1]
        act = ops.moe_gemv_gelu(gu_q, gu_s, h, ids, gu_q.shape[1], cols, 0,
                                inner)
        de = ops.int4_moe_gemv(dn_q, dn_s, act, ids, dn_q.shape[1], inner, inner)
        out = np.zeros_like(h)
        for j in range(ids.size):
            slot = np.nonzero(idx[0] == ids[j])[0]
            out[0] += de[j] * val[0, slot[0]]
        return out

    def _moe_mt(self, h, w, val, idx):
        """Run the selected experts for a small group of tokens.

        Each selected expert is read one time for all of its tokens. A pair is
        one (expert, token). The pairs of one expert are adjacent. Each token
        adds its experts in the order of the expert index, as
        _moe_one_token does, so each token gets the same bits.
        """
        inner = self.cfg.moe_intermediate_size
        gu_q, gu_s = w["experts.gate_up_proj"]
        dn_q, dn_s = w["experts.down_proj"]
        t, k = idx.shape
        flat = idx.reshape(-1)
        # Sort the pairs by expert. The stable sort keeps the tokens of one
        # expert in order.
        order = np.argsort(flat, kind="stable")
        tok = order // k
        slot = order % k
        ids, counts = np.unique(flat, return_counts=True)
        poff = np.concatenate([[0], np.cumsum(counts)])
        cols = h.shape[1]
        act = ops.moe_gemv_mt(gu_q, gu_s, h, ids, poff, tok, gu_q.shape[1],
                              cols, cols, inner)
        de = ops.moe_gemv_mt(dn_q, dn_s, act, ids, poff, np.arange(tok.size),
                             dn_q.shape[1], inner, inner)
        contrib = de * val[tok, slot][:, None]
        # Row j of m gives the pairs of token j in the order of the expert
        # index. Add them in that order, as _moe_one_token does.
        m = np.argsort(tok, kind="stable").reshape(t, k)
        out = np.zeros_like(h)
        for s in range(k):
            out += contrib[m[:, s]]
        return out

    def _rope(self, plan, positions):
        """Return the cosine and sine tables for one layer at the positions.

        Every layer of the same type uses the same table, so a prompt of 30
        layers makes two tables in place of 30. One table costs about 55
        microseconds. A long generation makes a new table for each token, so
        the cache holds a few entries.
        """
        key = (plan.is_sliding, plan.head_dim, int(positions[0]),
               int(positions.size))
        entry = self._rope_cache.get(key)
        if entry is None:
            if len(self._rope_cache) > 8:
                self._rope_cache.clear()
            cos, sin = rope_mod.cos_sin(self.cfg.rope_inv_freq(plan), positions)
            # Keep the address with the table. Every layer of the same type
            # uses this table, and the read of the address costs about 1.5
            # microseconds, so 30 layers must not read it 30 times. The cache
            # holds the table, so the address stays good.
            entry = (cos, sin, cos.ctypes.data, sin.ctypes.data)
            self._rope_cache[key] = entry
        return entry

    # ---- forward -----------------------------------------------------------
    def embed(self, input_ids):
        """Return the input embeddings for the token ids. Multiply by the embedding scale."""
        ids = np.asarray(input_ids, dtype=np.int64)
        if self._embed_q6k is not None:
            return self.st.q6k_dequant(self._embed_q6k[ids], self.cfg.hidden_size) * self.cfg.embed_scale
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
        t = len(input_ids)
        if (_PROGRAM and (t == 1 or ops.mt_ready(t)) and hook is None
                and max_layers is None and isinstance(cache, KVCache)
                and self._dtype == "int4" and self.keep_weights):
            # One decode step, or the group of an MTP verify step, as one
            # program in C (np_gemma/program.py). The result has the bits of
            # the Python loop below.
            from . import program
            if program.ready(self, cache) is not None:
                return program.decode_step(self, cache, input_ids, int(start_pos))
        x = self.embed(input_ids)
        if start_pos == 0:
            emit(hook, "embed_tokens", x)
            emit(hook, "inputs_embeds", x)
        n = cfg.num_hidden_layers if max_layers is None else min(max_layers, cfg.num_hidden_layers)
        positions = np.arange(start_pos, start_pos + len(input_ids))
        for i in range(n):
            plan = cfg.plan[i]
            w = self.load_layer(i)
            cos, sin, cos_a, sin_a = self._rope(plan, positions)
            x = self._decoder_layer(x, w, plan, cos, sin, positions, i, hook,
                                    cache, cos_a, sin_a)
            self.free_layer(i)
        if n == cfg.num_hidden_layers:
            norm_w = self._norm_w if self._norm_w is not None else self.st.get(PREFIX + "norm.weight")
            x = ops.rms_norm(x, norm_w, cfg.rms_norm_eps)
            emit(hook, "norm", x)
            emit(hook, "last_hidden_state", x)
        return x

    def _decoder_layer(self, x, w, plan, cos, sin, positions, i, hook, cache,
                       cos_a=None, sin_a=None):
        """Run one decoder layer. Use four normalization steps and two residual adds."""
        eps = self.cfg.rms_norm_eps
        p = "layers." + str(i) + "."
        residual = x
        h = ops.rms_norm(x, w["input_layernorm"], eps)
        emit(hook, p + "input_layernorm", h)
        h = self._attention(h, w, plan, cos, sin, positions, i, p, hook, cache,
                            cos_a, sin_a)
        h = ops.rms_norm(h, w["post_attention_layernorm"], eps)
        emit(hook, p + "post_attention_layernorm", h)
        x = residual + h
        residual = x
        fuse_norm = (x.shape[0] == 1 and self._dtype == "int4"
                     and ops.int4_multi4_ready())
        mt = self._dtype == "int4" and ops.mt_ready(x.shape[0])
        if fuse_norm:
            # The norm of the row, then the gate and the up projection, in one
            # call. The two projections also share the input row: one kernel
            # call runs both. One read of 6.7 MB is faster than two reads of
            # 3.3 MB: 38.8 GB/s against 31.5 on the 26B model.
            h = None
            g, u = ops.rms_norm_multi4(
                x, w["pre_feedforward_layernorm"], eps,
                [w["mlp.gate_proj"], w["mlp.up_proj"]],
                self.cfg.hidden_size)[:2]
            g = g.reshape(1, -1)
            u = u.reshape(1, -1)
        elif mt:
            # A small group: the same kernels for each token, with one read of
            # the weights for the whole group.
            h = ops.rms_norm(x, w["pre_feedforward_layernorm"], eps)
            g, u = ops.int4_multi4_mt([w["mlp.gate_proj"], w["mlp.up_proj"]],
                                      h, self.cfg.hidden_size)[:2]
        else:
            h = ops.rms_norm(x, w["pre_feedforward_layernorm"], eps)
            emit(hook, p + "pre_feedforward_layernorm", h)
            g = self.linear(h, w["mlp.gate_proj"])
            u = self.linear(h, w["mlp.up_proj"])
        emit(hook, p + "mlp.gate_proj", g)
        emit(hook, p + "mlp.up_proj", u)
        if fuse_norm:
            # gelu(g) * u, then the down projection, in one call. The down
            # projection reads the inner values, which is the width of the
            # gate and the up projection together.
            m = ops.gelu_mul_int4(g.reshape(-1), u.reshape(-1),
                                  w["mlp.down_proj"][0], w["mlp.down_proj"][1],
                                  self.cfg.hidden_size, self.cfg.intermediate_size)
            m = m.reshape(1, -1)
        elif mt:
            m = ops.linear_int4_mt(ops.gelu_mul_rows(g, u), *w["mlp.down_proj"])
        else:
            m = self.linear(ops.gelu_tanh(g) * u, w["mlp.down_proj"])
        emit(hook, p + "mlp.down_proj", m)
        if self.cfg.enable_moe_block:
            # The mixture-of-experts block is additive and parallel to the
            # shared MLP. The router and the experts read the residual, not the
            # MLP output.
            h1 = ops.rms_norm(m, w["post_feedforward_layernorm_1"], eps)
            emit(hook, p + "post_feedforward_layernorm_1", h1)
            val, idx = self._router(residual, w)
            emit(hook, p + "router.top_idx", idx.astype(np.float32))
            emit(hook, p + "router.probs", val)
            h2 = ops.rms_norm(residual, w["pre_feedforward_layernorm_2"], eps)
            emit(hook, p + "pre_feedforward_layernorm_2", h2)
            h2 = self._moe(h2, w, val, idx)
            emit(hook, p + "experts.out", h2)
            h2 = ops.rms_norm(h2, w["post_feedforward_layernorm_2"], eps)
            emit(hook, p + "post_feedforward_layernorm_2", h2)
            m = h1 + h2
        m = ops.rms_norm(m, w["post_feedforward_layernorm"], eps)
        emit(hook, p + "post_feedforward_layernorm", m)
        x = residual + m
        x = x * w["layer_scalar"]
        emit(hook, p + "out", x)
        return x

    def _attention(self, x, w, plan, cos, sin, positions, i, p, hook, cache,
                   cos_a=None, sin_a=None):
        """Run the attention part of one layer.

        For a sliding layer, keep only the keys inside the window and mask the
        ones that the causal mask hides. For a global layer, use the raw key
        projection for the value.
        """
        eps = self.cfg.rms_norm_eps
        hd = plan.head_dim
        t = x.shape[0]

        mt = self._dtype == "int4" and ops.mt_ready(t)
        if t == 1 and self._dtype == "int4" and ops.int4_multi4_ready():
            # One call serves the query, the key, and the value projection.
            vp = None if plan.k_eq_v else w["self_attn.v_proj"]
            qf, kf, vf, _ = ops.int4_multi4(
                [w["self_attn.q_proj"], w["self_attn.k_proj"], vp], x, self.cfg.hidden_size)
        elif mt:
            vp = None if plan.k_eq_v else w["self_attn.v_proj"]
            qf, kf, vf, _ = ops.int4_multi4_mt(
                [w["self_attn.q_proj"], w["self_attn.k_proj"], vp], x, self.cfg.hidden_size)
        else:
            qf = self.linear(x, w["self_attn.q_proj"])
            kf = self.linear(x, w["self_attn.k_proj"])
            vf = None if plan.k_eq_v else self.linear(x, w["self_attn.v_proj"])

        emit(hook, p + "self_attn.q_proj", qf)
        emit(hook, p + "self_attn.k_proj", kf)
        q2 = np.ascontiguousarray(qf).reshape(t * plan.num_q_heads, hd)
        k2 = np.ascontiguousarray(kf).reshape(t * plan.num_kv_heads, hd)
        if plan.k_eq_v:
            # The global layers have no v_proj. The value is the raw key data
            # with RMSNorm and no weight. Copy the key before the key norm.
            v2 = k2.copy()
        else:
            emit(hook, p + "self_attn.v_proj", vf)
            v2 = np.ascontiguousarray(vf).reshape(t * plan.num_kv_heads, hd)

        if ops.qkv_ready():
            # One call for the three norms and the two rotations.
            ops.qkv_norm_rope(q2, w["self_attn.q_norm"], k2,
                              w["self_attn.k_norm"], v2, cos, sin,
                              plan.num_q_heads, plan.num_kv_heads, hd, eps,
                              cos_a, sin_a)
            emit(hook, p + "self_attn.q_norm", q2)
            emit(hook, p + "self_attn.k_norm", k2)
            q = q2.reshape(t, plan.num_q_heads, hd)
            k = k2.reshape(t, plan.num_kv_heads, hd)
            v = v2.reshape(t, plan.num_kv_heads, hd)
        else:
            q = ops.rms_norm(q2.reshape(t, plan.num_q_heads, hd), w["self_attn.q_norm"], eps)
            emit(hook, p + "self_attn.q_norm", q)
            k = ops.rms_norm(k2.reshape(t, plan.num_kv_heads, hd), w["self_attn.k_norm"], eps)
            emit(hook, p + "self_attn.k_norm", k)
            if plan.k_eq_v:
                v = ops.rms_norm(k2.reshape(t, plan.num_kv_heads, hd), None, eps)
            else:
                v = ops.rms_norm(v2.reshape(t, plan.num_kv_heads, hd), None, eps)
            q = rope_mod.apply(q, cos, sin)
            k = rope_mod.apply(k, cos, sin)

        if cache is not None:
            start = int(positions[0])
            qc_before = cache.qc_ready(i)
            cache.write(i, start, k, v)
            if t == 1 or mt:
                # A decode step, or a small group that repeats the decode step
                # for each token. Row j sees the cache rows up to its position.
                o = None
                if mt and ops.attn_ready() and cache.qc_ready(i) and (
                        qc_before or start + 1 - cache.base[i] >= cache.attn_min):
                    # Every row uses the int16 cache. One call serves the group.
                    kq, ks, vq, vs, base = cache.read_qc(i, start + t)
                    pos = start + np.arange(t)
                    window = plan.sliding_window or 0
                    lo = np.maximum(0, pos - window + 1 - base) if window else np.zeros(t, np.int64)
                    o = ops.attn_decode_mt(q, kq, ks, vq, vs, plan.num_q_heads,
                                           plan.num_kv_heads, hd, lo, pos + 1 - base - lo)
                    o = o.reshape(t, plan.q_dim)
                if o is None:
                    o = np.empty((t, plan.q_dim), dtype=np.float32)
                    for j in range(t):
                        o[j] = self._attend_one(q[j:j + 1], plan, i, start + j, cache,
                                                qc_before)
                if mt:
                    out = ops.linear_int4_mt(o, *w["self_attn.o_proj"])
                else:
                    out = self.linear(o, w["self_attn.o_proj"])
                emit(hook, p + "self_attn.o_proj", out)
                return out
            K, V, base = cache.read(i, start + t)
        else:
            K, V, base = k, v, positions[0]

        flash = os.environ.get("NP_GEMMA_FLASH", "0")
        window = plan.sliding_window or 0
        # "slide" combines the two paths: the kernel serves a sliding layer,
        # where the window caps the work, and the batched matmul serves a
        # global layer, where OpenBLAS tiles the score matrix better than a
        # small register tile.
        use_flash = flash != "0" and (flash != "slide" or window > 0)
        if t > 1 and use_flash:
            # The flash path. It keeps the scores of one block at a time and it
            # walks only the keys that the mask leaves visible. Use the C
            # kernel. The NumPy reference stays for a comparison and for a
            # target with no C kernel.
            from .flash import flash_attention
            if flash == "ref" or not ops.flash_ready():
                fo = flash_attention(q, K, V, positions, base, window)
            else:
                fo = ops.flash_prefill(q, K, V, positions, base, window)
            out = self.linear(fo.reshape(t, plan.q_dim), w["self_attn.o_proj"])
            emit(hook, p + "self_attn.o_proj", out)
            return out
        out = self._attend_rows(q, K, V, base, positions, plan)
        out = self.linear(out, w["self_attn.o_proj"])
        emit(hook, p + "self_attn.o_proj", out)
        return out

    def _attend_one(self, q, plan, i, pos, cache, qc_before):
        """Run the attention of one query at pos over the cache. Return (q_dim,).

        Use the int16 cache when a decode step at pos uses it. That is true
        when the int16 copy is on before the write. It is also true when the
        write turns the copy on at pos + 1 rows. Otherwise use the float cache.

        A sliding layer reads only the rows of the window. The cache can hold
        more rows than the window, because it drops old rows in large steps.
        """
        hd = plan.head_dim
        window = plan.sliding_window or 0
        if ops.attn_ready() and cache.qc_ready(i) and (
                qc_before or pos + 1 - cache.base[i] >= cache.attn_min):
            kq, ks, vq, vs, base = cache.read_qc(i, pos + 1)
            lo = max(0, pos - window + 1 - base) if window else 0
            o = ops.attn_decode(q[0], kq[lo:], ks[lo:], vq[lo:], vs[lo:],
                                plan.num_q_heads, plan.num_kv_heads, hd,
                                kq.shape[0] - lo)
            return o.reshape(plan.q_dim)
        K, V, base = cache.read(i, pos + 1)
        if _F32_ATTN_C:
            # The C kernel of the float cache. The program of a decode step
            # uses the same kernel, so the two give the same bits.
            lo = max(0, pos - window + 1 - base) if window else 0
            o = ops.attn_decode_f32s(q[0], K[lo:], V[lo:], pos, base + lo, window)
            return o.reshape(plan.q_dim)
        return self._attend_rows(q, K, V, base, np.array([pos]), plan)[0]

    def _attend_rows(self, q, K, V, base, positions, plan):
        """Run the attention of the queries q over K and V. Return (t, q_dim).

        The causal mask and the window mask come from the positions.
        """
        t = q.shape[0]
        hd = plan.head_dim
        # The attention scale is 1.0. Do not divide by sqrt(head_dim).
        n_rep = plan.num_q_heads // plan.num_kv_heads
        nk = plan.num_kv_heads
        n = K.shape[0]
        # Drop the keys that are invisible for every query in the block. The
        # causal mask hides the keys after the last query. The window hides
        # the keys before the first query less the window. A key that is
        # hidden for every query would become zero in the softmax, so dropping
        # it does not change the result. This is what keeps a long prompt
        # affordable: 25 of the 30 layers slide with a window of 1024, so they
        # read the window and not the whole context.
        window = plan.sliding_window or 0
        if window and os.environ.get("NP_GEMMA_SLIDE", "1") == "1":
            kpos = base + np.arange(n)
            lo = int(np.searchsorted(kpos, positions.min() - window + 1, side='left'))
            hi = int(np.searchsorted(kpos, positions.max(), side='right'))
            if lo or hi < n:
                K = K[lo:hi]
                V = V[lo:hi]
                base = base + lo
                n = hi - lo
        # Use a batched matrix multiply. The code makes one matrix for each
        # group of query heads. matmul is faster than einsum here, because
        # einsum looks for a contraction path at each call.
        qb = q.reshape(t, nk, n_rep, hd).transpose(1, 0, 2, 3).reshape(nk, t * n_rep, hd)
        kb = K.transpose(1, 2, 0)
        scores = np.matmul(qb, kb).reshape(nk, t, n_rep, n)
        # The kernel applies the causal mask, the window mask, and the softmax
        # over the last axis in one pass. The values that the mask hides become
        # zero, so a decode step needs no special case.
        probs = ops.softmax_mask(scores, positions, n_rep, base, window)
        vb = V.transpose(1, 0, 2)
        out = np.matmul(probs.reshape(nk, t * n_rep, n), vb)
        return out.reshape(nk, t, n_rep, hd).transpose(1, 0, 2, 3).reshape(t, plan.q_dim)

    # ---- output head -------------------------------------------------------
    def logits(self, x, chunk=32768, apply_softcap=True):
        """Return the logits for the hidden states x.

        Use the embedding table. The embeddings are tied to the output head.
        Apply the softcap when requested.
        """
        if self._embed_q6k is not None:
            out = ops.linear_q6k(x, self._embed_q6k_bytes, self.cfg.hidden_size)
        elif self._embed_q is not None:
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
    def prefill(self, ids, cache, start=0, hook=None):
        """Run the prompt. Use blocks to keep the GEMM in its fast range.

        The key and value cache holds the earlier blocks. The result is the
        same as one forward pass over the full prompt. start is the position
        of ids[0]. Use it to add tokens to a cache that already has data. The
        hook gives the time of each stage of every block.
        """
        chunk = self.prefill_chunk
        x = None
        for off in range(0, len(ids), chunk):
            x = self.forward(ids[off:off + chunk], cache=cache, start_pos=start + off,
                             hook=hook)
        return x

    def prefill_layer_major(self, ids, cache, start=0, hook=None, chunk=None):
        """Run the prompt one layer at a time instead of one block at a time.

        The hidden state of every token stays in memory. The code applies one
        layer to the whole prompt, then the next layer. Thus a layer is loaded
        one time and its weights are used for every token, and the mixture of
        experts sees the whole prompt for each expert. The result matches
        prefill, but the routing may choose a different expert, because the
        token grouping changes the order of the sums.

        A chunk of zero means the whole prompt in one step. That step needs the
        tile attention, because the score matrix of a global layer would
        otherwise hold tokens * keys values.
        """
        cfg = self.cfg
        if chunk is None:
            chunk = self.prefill_chunk
        if chunk <= 0:
            chunk = len(ids)
        ids = np.asarray(ids)
        n_tok = len(ids)
        x = self.embed(ids)
        if start == 0:
            emit(hook, "embed_tokens", x)
            emit(hook, "inputs_embeds", x)
        pos = np.arange(start, start + n_tok)
        for i in range(cfg.num_hidden_layers):
            plan = cfg.plan[i]
            w = self.load_layer(i)
            for off in range(0, n_tok, chunk):
                stop = off + chunk
                if stop > n_tok:
                    stop = n_tok
                cos, sin, cos_a, sin_a = self._rope(plan, pos[off:stop])
                x[off:stop] = self._decoder_layer(
                    x[off:stop], w, plan, cos, sin, pos[off:stop], i, hook,
                    cache, cos_a, sin_a)
            self.free_layer(i)
        norm_w = self._norm_w if self._norm_w is not None else self.st.get(PREFIX + "norm.weight")
        x = ops.rms_norm(x, norm_w, cfg.rms_norm_eps)
        emit(hook, PREFIX + "norm", x)
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

    def __init__(self, model, max_len=8192, drafter=None, n_draft=2):
        self.model = model
        self.max_len = max_len
        # The MTP drafter (np_gemma.assistant.Assistant). None turns MTP off.
        self.drafter = drafter
        self.n_draft = n_draft
        # The counts of the last MTP generation: steps, drafts, accepted.
        self.mtp_stats = {}
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

    def generate_stream(self, ids, max_new_tokens=1, eos_ids=(), sampler=None):
        """Yield one token id at a time.

        Run the new prompt tokens, then select one token for each step. The
        sampler holds the sampling settings. The default sampler selects the
        most probable token. The generator stops at an end token.
        """
        from .sampling import Sampler

        ids = list(ids)
        self.prefill(ids)
        if self._x is None:
            # The prompt is the same as the cache. Run the last token again.
            self._x = self.model.forward(ids[-1:], cache=self.cache,
                                         start_pos=len(ids) - 1)
        if sampler is None:
            sampler = Sampler(temperature=0.0)
        sampler.reset(ids)
        x = self._x
        pos = len(ids)
        if self.drafter is not None and max_new_tokens > 0:
            from .assistant import mtp_enabled, mtp_stream
            if mtp_enabled():
                # The drafter proposes tokens and one pass of the model checks
                # them. The sampler picks each token, as below, so the tokens
                # are the tokens of the plain loop.
                nxt = sampler(self.model.logits(x[-1:])[0])
                self.mtp_stats = {}
                yield from mtp_stream(self.model, self.drafter, self.cache, self.ids,
                                      x[-1:], nxt, self.n_draft, eos_ids, sampler,
                                      max_new_tokens, self.mtp_stats)
                return
        for _ in range(max_new_tokens):
            nxt = sampler(self.model.logits(x[-1:])[0])
            yield nxt
            if nxt in eos_ids:
                return
            x = self.model.forward([nxt], cache=self.cache, start_pos=pos)
            self.ids.append(nxt)
            pos += 1
