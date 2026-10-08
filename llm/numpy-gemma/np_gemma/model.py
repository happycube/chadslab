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
import weakref

import numpy as np

from . import ops
from . import rope as rope_mod
from . import tq6
from .weight_cache import WeightCache

PREFIX = "model.language_model."

# The rows that a layer with a window keeps before the window when it drops
# old rows (KVCache.prepare, GPUKV.prepare). A chat turn cuts the cache back
# to the start of the last answer, and the first new token needs the window
# before the cut. Without this margin a cut soon after a drop found the rows
# gone, and the Session read the whole history again (a video of 2325 tokens:
# 5 s for each question). A cut back of up to KV_KEEP tokens is always
# possible. The cost: KV_KEEP more rows in each layer with a window.
KV_KEEP = int(os.environ.get("NP_GEMMA_KV_KEEP", "1024"))

# NP_GEMMA_KV_INT8=1: the cache is int8 only (KVCache kv="int8"), with no
# float32 rows. A step of the 26B then reads half the bytes of the int16
# copy (about 150 MB in place of 300 MB a token at 4300 tokens).
# NP_GEMMA_KV_INT8=v: int16 keys and int8 values (kv="k16v8"), also with no
# float32 rows: three quarters of the bytes of the int16 copy.
# NP_GEMMA_KV_INT8=r: rq8, the int8 cache of the rows rotated in each 32
# values (the TQ6 rotation, np_gemma/tq6.py); =vr: k16vr8, int16 keys and
# rotated int8 values. The storage of int8 and k16v8 (KV_BASE): the step
# rotates the keys and the values before their write, the query for rotated
# keys (the scores do not change) and the output back for rotated values.
# On the 26B at 100K tokens (scripts/study_kv_forms.py), the error of the
# attention output: int8 7.4e-4, rq8 5.5e-4, k16v8 5.2e-4.
KV_FORM = {"1": "int8", "v": "k16v8", "r": "rq8", "vr": "k16vr8"}.get(
    os.environ.get("NP_GEMMA_KV_INT8", "0"), "int16")
KV_BASE = {"rq8": "int8", "k16vr8": "k16v8"}


def kv_base(form):
    """The storage form of a KV form (rq8: int8, k16vr8: k16v8)."""
    return KV_BASE.get(form, form)


def kv_rot(form=None):
    """(keys rotated, values rotated) of a KV form (else of KV_FORM)."""
    f = KV_FORM if form is None else form
    return f == "rq8", f in ("rq8", "k16vr8")

# The tokens of one image see each other in every layer. The docstring of
# create_masks_for_vision_model (transformers) says that the global layers
# stay causal, but the logits of generate() and of forward() both agree bit
# for bit with the mask in every layer (scripts/check_mm_prompt.py, the 12B in
# float32), and llama.cpp does the same. NP_GEMMA_BIDIR_ALL=0 keeps the
# global layers causal (KL 0.008 to transformers on a prompt with an image).
_BIDIR_ALL = os.environ.get("NP_GEMMA_BIDIR_ALL", "1") == "1"
# A prompt with media runs on the GPU when the model does (ModelGPU.group);
# NP_GEMMA_MEDIA_GPU=0 runs it on the CPU in Python.
_MEDIA_GPU = os.environ.get("NP_GEMMA_MEDIA_GPU", "1") == "1"

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
# Run a decode step of one token and the output head on a CUDA GPU, with the
# experts on the CPU (np_gemma/gpu.py, SPLIT_PLAN.md, phase 4).
_GPU = os.environ.get("NP_GEMMA_GPU", "0") == "1"


def emit(hook, key, value):
    """Send one intermediate tensor to the hook. Do nothing when hook is None."""
    if hook is not None:
        hook(key, np.asarray(value, dtype=np.float32))


class KVCache:
    """Store the key and value tensors of each layer.

    A global layer keeps the full sequence. A sliding layer keeps the last
    window. The buffers have a fixed size. The code writes in place. Thus the
    cache does not copy the full sequence for each token.

    The cache keeps only quantized rows, with a float32 scale for each group
    of 32 values: no float32 rows. The form "int16" (a scale of max |x| /
    32767), "int8" (max |x| / 127: half the bytes for each step to read),
    or "k16v8" (int16 keys, int8 values). read() gives dequantized rows to
    the readers of float rows (the prompt attention). The default form comes
    from NP_GEMMA_KV_INT8.

    With NP_GEMMA_PARTS=2 (or more) and the int16 form, KVCache(...) gives a
    parts.PartKVCache: a cache for each part of a step, in the memory of its
    node, with the KV heads of that part. NP_GEMMA_PART_KV=0 turns this off.
    """

    split = False      # True for a parts.PartKVCache

    def __new__(cls, cfg=None, max_len=4096, kv=None, **kw):
        if (cls is KVCache and int(os.environ.get("NP_GEMMA_PARTS", "1")) > 1
                and os.environ.get("NP_GEMMA_PART_KV", "1") != "0"
                and (kv or KV_FORM) == "int16"):
            from .parts import PartKVCache
            cls = PartKVCache
        return super().__new__(cls)

    def __init__(self, cfg, max_len=4096, kv=None):
        self.cfg = cfg
        self.window = cfg.sliding_window or 0
        self.max_len = max_len
        if kv is None:
            kv = KV_FORM
        # rq8, k16vr8: rotated rows in the storage of int8, k16v8 (kv_rot)
        self.rot_k, self.rot_v = kv_rot(kv)
        kv = kv_base(kv)
        if kv not in ("int16", "int8", "k16v8"):
            raise ValueError("kv must be int16, int8, k16v8, rq8 or k16vr8, not %r" % (kv,))
        if not ops._COPS_READY:
            raise ValueError("the quantized cache needs the C kernels")
        self.kv = kv
        # the dtypes of the keys and of the values
        self.kdtype = np.int8 if kv == "int8" else np.int16
        self.vdtype = np.int8 if kv in ("int8", "k16v8") else np.int16
        n = cfg.num_hidden_layers
        self.kq = [None] * n     # the keys, quantized
        self.ks = [None] * n     # one float32 scale for each group of 32
        self.vq = [None] * n     # the values, quantized
        self.vs = [None] * n
        self.base = [0] * n      # absolute position of buffer row 0
        self.end = [0] * n       # absolute position after the last stored row

    def _cap(self, layer):
        """The rows of the buffers of a layer (0 before the first write)."""
        a = self.kq[layer]
        return 0 if a is None else a.shape[0]

    def _shape_q(self, layer, cap):
        plan = self.cfg.plan[layer]
        return (cap, plan.num_kv_heads, plan.head_dim)

    def _shape_s(self, layer, cap):
        plan = self.cfg.plan[layer]
        return (cap, plan.num_kv_heads, plan.head_dim // 32)

    def _grow(self, layer, cap):
        """Give layer i buffers of cap rows; keep the rows it has."""
        old = self._cap(layer)
        new = [np.empty(self._shape_q(layer, cap), dtype=self.kdtype),
               np.empty(self._shape_s(layer, cap), dtype=np.float32),
               np.empty(self._shape_q(layer, cap), dtype=self.vdtype),
               np.empty(self._shape_s(layer, cap), dtype=np.float32)]
        if old:
            for dst, src in zip(new, (self.kq[layer], self.ks[layer], self.vq[layer],
                                      self.vs[layer])):
                dst[:old] = src[:old]
        self.kq[layer], self.ks[layer], self.vq[layer], self.vs[layer] = new

    def _buffers(self, layer):
        """The arrays of a layer, with one row for each position."""
        return (self.kq[layer], self.ks[layer], self.vq[layer], self.vs[layer])

    def prepare(self, layer, start_pos, t):
        """Make room for t rows at start_pos. Return the buffer row of start_pos.

        A sliding layer drops its oldest rows here, and a buffer grows here.
        write calls this first. The program of a decode step (np_gemma.program)
        calls it before the step and then writes the rows in C.
        """
        end = start_pos + t
        if self.cfg.plan[layer].is_sliding:
            w = self.window
            if self._cap(layer) == 0:
                self._grow(layer, 2 * w)
            if start_pos > self.end[layer]:
                # Rows after a gap (GPUKV.detach: the GPU dropped the rows of
                # the window before its base, and the host had none of them):
                # the layer starts at start_pos. The drop below had set base to
                # start_pos - window - KV_KEEP, over rows that no one wrote, and
                # a truncate then took them for the window of the prompt (the
                # 26B, a session of 108K tokens back after another session:
                # garbage in the window, and the GPU hung).
                self.base[layer] = self.end[layer] = start_pos
            # Drop the oldest rows when the buffer holds more than two
            # windows. A query at position p sees back to p - window + 1, so
            # the first query of the new block sees back to
            # start_pos - window + 1. Every row before that is hidden for the
            # whole block. The buffer then holds about the window and the
            # block, and it does not grow with the context. A decode step and
            # a prompt block both compact. The old code compacted only for a
            # decode step, so a prompt block grew the buffer to the full
            # sequence.
            if start_pos - self.base[layer] > 2 * w + KV_KEEP:
                keep = start_pos - w + 1 - KV_KEEP
                off = keep - self.base[layer]
                rows = self.end[layer] - keep
                if rows > 0:
                    for a in self._buffers(layer):
                        a[:rows] = a[off:off + rows]
                self.base[layer] = keep
            need = end - self.base[layer]
            if need > self._cap(layer):
                # The rows stay below 2 w + KV_KEEP + t (the drop above), so
                # one growth to that size serves every later decode step. A
                # growth of only the rows of the step copied the whole buffer
                # of each sliding layer for each token from 2 w to 2 w +
                # KV_KEEP rows: the 26B went from 42 to 210 ms a step.
                self._grow(layer, max(need, 2 * w + KV_KEEP + 1))
        else:
            if self._cap(layer) < end:
                cap = max(self.max_len, end)
                if self._cap(layer):
                    cap = max(cap, self._cap(layer) * 2)
                self._grow(layer, cap)
        return start_pos - self.base[layer]

    def write(self, layer, start_pos, k, v):
        """Store a block of keys and values. start_pos is the position of k[0]."""
        t = k.shape[0]
        start = self.prepare(layer, start_pos, t)
        self.end[layer] = start_pos + t
        self._store_qc(layer, start, k, v)

    def rows_q(self, layer, start_pos, t):
        """Make room for t rows at start_pos, for a writer that stores the
        quantized rows of a GPU cache there with no copy. Return the views
        (kq, ks, vq, vs) of the rows, of the shapes (t, kv heads, head_dim)
        and (t, kv heads, head_dim / 32). Then the writer calls
        rows_q_done."""
        start = self.prepare(layer, start_pos, t)
        r = slice(start, start + t)
        return self.kq[layer][r], self.ks[layer][r], self.vq[layer][r], self.vs[layer][r]

    def rows_q_done(self, layer, start_pos, t):
        """The rows of rows_q are written."""
        self.end[layer] = start_pos + t

    def write_q(self, layer, start_pos, kq, ks, vq, vs):
        """Store a block of quantized keys and values (shapes as rows_q), as a
        GPU cache holds them. They are not quantized a second time."""
        t = kq.shape[0]
        for dst, src in zip(self.rows_q(layer, start_pos, t), (kq, ks, vq, vs)):
            dst[...] = src
        self.rows_q_done(layer, start_pos, t)

    def _store_qc(self, layer, start, k, v):
        """Quantize and store a block of keys and values from buffer row
        start."""
        t = k.shape[0]
        plan = self.cfg.plan[layer]
        g = plan.head_dim // 32
        nkv = plan.num_kv_heads
        quant = {np.dtype(np.int8): ops.quantize_i8, np.dtype(np.int16): ops.quantize_i16}
        if getattr(self, "rot_k", False):
            k = tq6.rotate(k)
        if getattr(self, "rot_v", False):
            v = tq6.rotate(v)
        kq, ks = quant[np.dtype(self.kdtype)](k.reshape(t, nkv, g, 32))
        vq, vs = quant[np.dtype(self.vdtype)](v.reshape(t, nkv, g, 32))
        self.kq[layer][start:start + t] = kq.reshape(t, nkv, g * 32)
        self.ks[layer][start:start + t] = ks
        self.vq[layer][start:start + t] = vq.reshape(t, nkv, g * 32)
        self.vs[layer][start:start + t] = vs

    def read(self, layer, end, lo=0, rotated=False):
        """Return the keys, the values (new float32 arrays of the dequantized
        rows), and the position of the first row. lo skips the first lo rows
        of the buffer (their position is base + lo). rotated: the rows of
        the rotated forms (rq8, k16vr8) as stored (the reader rotates its
        queries and unrotates its output)."""
        base = self.base[layer]
        n = end - base - lo
        shape = (n,) + self.kq[layer].shape[1:]
        out = []
        rots = (getattr(self, "rot_k", False), getattr(self, "rot_v", False))
        for (q, sc), rot in zip(((self.kq[layer], self.ks[layer]), (self.vq[layer], self.vs[layer])), rots):
            x = np.empty(shape, np.float32)
            q, sc = q[lo:lo + n], sc[lo:lo + n]
            if q.dtype == np.int8:
                ops._cops.dequantize_i8_groups(q, sc, x)
            else:
                ops._cops.dequantize_i16_groups(q, sc, x)
            out.append(tq6.unrotate(x) if rot and not rotated else x)    # (rq8, k16vr8: the rows back)
        return out[0], out[1], base + lo

    def qc_ready(self, layer):
        """Return True: the quantized rows are the cache (an old test)."""
        return True

    def read_qc(self, layer, end):
        """Return the quantized keys and values, their scales, and the
        position of row 0."""
        base = self.base[layer]
        n = end - base
        return (self.kq[layer][:n], self.ks[layer][:n],
                self.vq[layer][:n], self.vs[layer][:n], base)

    def length(self, layer):
        """Return the number of stored positions in one layer."""
        if self._cap(layer) == 0:
            return 0
        return self.end[layer] - self.base[layer]

    def truncate(self, n):
        """Keep only the positions before n. Return False when that is not possible.

        A sliding layer drops the oldest rows. The next token at n sees back
        to n - window + 1, so a sliding layer must still hold that row. If it
        does not, that layer cannot go back. The caller must then start again.
        """
        layers = range(self.cfg.num_hidden_layers)
        for i in layers:
            if self._cap(i) and self.base[i] > self.first_row(i, n):
                return False
        for i in layers:
            if self._cap(i) and n < self.end[i]:
                self.end[i] = n
        return True

    def first_row(self, layer, n):
        """Return the first position that a token at n sees in the layer."""
        if self.cfg.plan[layer].is_sliding and self.window:
            return max(0, n - self.window + 1)
        return n


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
        # The soft-token spans of the prompt that prefill runs now (media.Span).
        self._media = None
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
        # The activations of the int4 products of a prompt pass: "1" (int8),
        # "16" (int16; the default of a model with experts), or "0" (float).
        # See ops.prompt_act.
        self.prompt_act = ops.prompt_act(cfg.enable_moe_block)

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

    def _embed_type(self, hf_name):
        """Return the GGUF type name of the source table, or None.

        Only a GGUF file sets keep_embedding_bf16. The QAT GGUFs of Google keep
        the tied output head in Q6_K; other files (Unsloth) keep it in Q4_0.
        """
        if not getattr(self.st, "keep_embedding_bf16", False):
            return None
        try:
            return self.st.dtype(hf_name)
        except (KeyError, AttributeError, ValueError):
            return None

    def _is_q6k_embed(self, hf_name):
        """Return True when the source table is Q6_K."""
        return self._embed_type(hf_name) == "Q6_K"

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
            # The int4 layout changed to the block layout of Q4_0, then the
            # quantizer found the QAT grid (ops.quantize_int4). Use a new
            # cache key.
            cache = WeightCache(self.st.path, dtype, extra="grid" if dtype == "int4" else "")
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
        elif (dtype == "int4" and self._embed_type(src) == "Q4_0"
              and hasattr(self.st, "int4_packed")):
            # The file keeps the tied output head in Q4_0. Use the blocks in
            # place, as for the layers: the head reads 4.5 bits for each
            # weight, not the 16 of a bfloat16 copy, and has the same values.
            self._embed_q, self._embed_s = self.st.int4_packed(src)
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
            self._cache = WeightCache(self.st.path, dtype, extra="grid" if dtype == "int4" else "")
        self._dtype = dtype
        self.keep_weights = True
        if dtype in ("int8", "int4"):
            # The copy holds all the weights. Drop the mapped bf16 pages.
            self.st.release_pages()
        if dtype == "int4":
            # The int4 matrices in groups of 16 rows too, for the prompt
            # (ops.q4x_pack_model; NP_GEMMA_Q4X=0 turns it off).
            ops.q4x_pack_model(self)
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
            return ops.linear_int4(x, w[0], w[1], q8=self.prompt_act == "1",
                                   x16=self.prompt_act == "16")
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
        if self._dtype == "int4" and h.shape[0] >= 2 and ops.moe_prompt_ready(self.prompt_act):
            # One parallel region covers every expert of the layer. The
            # activations are int8, int16, or float32 (self.prompt_act).
            return ops.moe_prompt(h, w["experts.gate_up_proj"],
                                  w["experts.down_proj"], val, idx,
                                  self.cfg.moe_intermediate_size, self.prompt_act)
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
        media = self._media_in(start_pos, t)
        if (_GPU and t <= 16 and hook is None and max_layers is None and not media
                and isinstance(cache, KVCache) and self._dtype == "int4" and self.keep_weights):
            # One decode step, or the verify group of an MTP step.
            if t == 1:
                return self._gpu_step(input_ids, cache, int(start_pos))
            return self._gpu_group(input_ids, cache, int(start_pos))
        self._gpu_release(cache)
        if (_PROGRAM and (t == 1 or ops.mt_ready(t)) and hook is None and not media
                and max_layers is None and isinstance(cache, KVCache)
                and self._dtype == "int4" and self.keep_weights
                and not (getattr(cache, "split", False) and t > 1)):
            # One decode step, or the group of an MTP verify step, as one
            # program in C (np_gemma/program.py). The result has the bits of
            # the Python loop below.
            from . import program
            if program.ready(self, cache) is not None:
                return program.decode_step(self, cache, input_ids, int(start_pos))
        if t > 1 and hook is None and max_layers is None:
            # A prompt block as one program (np_gemma/prompt.py): the kernels
            # of the loop below, so the same bits, with no Python between them.
            # The soft rows of media and their attention too.
            from . import prompt
            if prompt.prompt_ready(self, cache, t):
                return prompt.prompt_step(self, cache, input_ids, int(start_pos), media)
        x = self.embed(input_ids)
        for sp in media:
            # A soft token takes the row of its image or clip, with no scale.
            lo, hi = max(sp.start, start_pos), min(sp.end, start_pos + t)
            x[lo - start_pos:hi - start_pos] = sp.rows[lo - sp.start:hi - sp.start]
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

        # The tokens of an image see each other (use_bidirectional_attention
        # "vision"): the last key of each query. See _BIDIR_ALL.
        limit = self._media_limit(positions) if (plan.sliding_window or _BIDIR_ALL) else None
        if cache is not None:
            start = int(positions[0])
            flash = os.environ.get("NP_GEMMA_FLASH", "1")
            window = plan.sliding_window or 0
            if (getattr(cache, "split", False) and t > 1 and not mt and limit is None
                    and flash not in ("0", "ref") and (flash != "slide" or window > 0)
                    and ops.flash_ready()):
                # The parts write and read the cache of their own heads, each
                # in its team on its node (parts.PartKVCache): the bits of the
                # flash path below.
                fo = cache.prefill_attention(i, start, q, k, v, positions, window)
                out = self.linear(fo.reshape(t, plan.q_dim), w["self_attn.o_proj"])
                emit(hook, p + "self_attn.o_proj", out)
                return out
            cache.write(i, start, k, v)
            if (t == 1 or mt) and limit is None:
                # A decode step, or a small group that repeats the decode step
                # for each token. Row j sees the cache rows up to its position.
                o = None
                if mt and ops.attn_ready():
                    # Every row uses the int16 cache. One call serves the group.
                    kq, ks, vq, vs, base = cache.read_qc(i, start + t)
                    pos = start + np.arange(t)
                    window = plan.sliding_window or 0
                    lo = np.maximum(0, pos - window + 1 - base) if window else np.zeros(t, np.int64)
                    rk, rv = getattr(cache, "rot_k", False), getattr(cache, "rot_v", False)
                    o = ops.attn_decode_mt(tq6.rotate(q) if rk else q, kq, ks, vq, vs, plan.num_q_heads,
                                           plan.num_kv_heads, hd, lo, pos + 1 - base - lo)
                    o = o.reshape(t, plan.q_dim)
                    if rv:
                        o = tq6.unrotate(o)
                if o is None:
                    o = np.empty((t, plan.q_dim), dtype=np.float32)
                    for j in range(t):
                        o[j] = self._attend_one(q[j:j + 1], plan, i, start + j, cache)
                if mt:
                    out = ops.linear_int4_mt(o, *w["self_attn.o_proj"])
                else:
                    out = self.linear(o, w["self_attn.o_proj"])
                emit(hook, p + "self_attn.o_proj", out)
                return out
            if (t > 1 and flash not in ("0", "ref")
                    and (flash != "slide" or window > 0) and ops.flash_ready()
                    and cache.kdtype == np.int16 and cache.vdtype == np.int16):
                # The flash kernel on the int16 cache itself (no float copy of
                # the cache): the bits of KVCache.read and flash_prefill. With
                # media, limit gives the last key of each query (the tokens
                # of an image see each other), as the prompt program.
                kq, ks, vq, vs, base = cache.read_qc(i, start + t)
                fo = ops.flash_prefill_qc(q, kq, ks, vq, vs, positions, base, window, limit)
                if fo is not None:
                    out = self.linear(fo.reshape(t, plan.q_dim), w["self_attn.o_proj"])
                    emit(hook, p + "self_attn.o_proj", out)
                    return out
            # the rotated forms (rq8, k16vr8): the rows as stored, the
            # queries rotated and the output back, as the decode (an unrotate
            # of every row for each block of a prompt took 2.3 times the
            # prompt of int16 at 4300 tokens)
            rk, rv = getattr(cache, "rot_k", False), getattr(cache, "rot_v", False)
            K, V, base = cache.read(i, start + t, rotated=True)
            if rk:
                q = tq6.rotate(q)
        else:
            K, V, base = k, v, positions[0]
            rv = False

        # The C flash kernel is the default: a prompt of the 26B of 512 tokens
        # takes 2.2 s, not 3.1 s, and one of 16384 tokens 156 s, not 318
        # (README.md, "The int4 matrices in groups of 16 rows").
        flash = os.environ.get("NP_GEMMA_FLASH", "1")
        window = plan.sliding_window or 0
        # "slide" combines the two paths: the kernel serves a sliding layer,
        # where the window caps the work, and the batched matmul serves a
        # global layer, where OpenBLAS tiles the score matrix better than a
        # small register tile.
        use_flash = flash != "0" and (flash != "slide" or window > 0) and limit is None
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
            if rv:
                fo = tq6.unrotate(fo)
            out = self.linear(fo.reshape(t, plan.q_dim), w["self_attn.o_proj"])
            emit(hook, p + "self_attn.o_proj", out)
            return out
        out = self._attend_rows(q, K, V, base, positions, plan, limit)
        if rv:
            out = tq6.unrotate(out)
        out = self.linear(out, w["self_attn.o_proj"])
        emit(hook, p + "self_attn.o_proj", out)
        return out

    def _media_in(self, start, t):
        """Return the media spans that overlap positions start to start + t - 1."""
        media = self.__dict__.get("_media")
        if not media:
            return ()
        return [sp for sp in media if sp.start < start + t and sp.end > start]

    def _media_limit(self, positions):
        """Return the last key position of each query, or None when no query
        is in a span whose tokens see each other."""
        spans = [sp for sp in self._media_in(int(positions[0]), len(positions)) if sp.bidir]
        if not spans:
            return None
        limit = np.array(positions, dtype=np.int64)
        for sp in spans:
            inside = (limit >= sp.start) & (np.asarray(positions) < sp.end)
            limit[inside] = sp.end - 1
        return limit

    def _attend_one(self, q, plan, i, pos, cache):
        """Run the attention of one query at pos over the cache. Return (q_dim,).

        The fused kernel reads the quantized rows (NP_GEMMA_ATTN=1, the
        default). NP_GEMMA_ATTN=0 reads dequantized rows (KVCache.read).

        A sliding layer reads only the rows of the window. The cache can hold
        more rows than the window, because it drops old rows in large steps.
        """
        hd = plan.head_dim
        window = plan.sliding_window or 0
        if ops.attn_ready():
            kq, ks, vq, vs, base = cache.read_qc(i, pos + 1)
            lo = max(0, pos - window + 1 - base) if window else 0
            rk, rv = getattr(cache, "rot_k", False), getattr(cache, "rot_v", False)
            o = ops.attn_decode(tq6.rotate(q[0]) if rk else q[0], kq[lo:], ks[lo:], vq[lo:], vs[lo:],
                                plan.num_q_heads, plan.num_kv_heads, hd,
                                kq.shape[0] - lo)
            return (tq6.unrotate(o) if rv else o).reshape(plan.q_dim)
        K, V, base = cache.read(i, pos + 1)
        if _F32_ATTN_C:
            # The C kernel of the float cache. The program of a decode step
            # uses the same kernel, so the two give the same bits.
            lo = max(0, pos - window + 1 - base) if window else 0
            o = ops.attn_decode_f32s(q[0], K[lo:], V[lo:], pos, base + lo, window)
            return o.reshape(plan.q_dim)
        return self._attend_rows(q, K, V, base, np.array([pos]), plan)[0]

    def _attend_rows(self, q, K, V, base, positions, plan, limit=None):
        """Run the attention of the queries q over K and V. Return (t, q_dim).

        The causal mask and the window mask come from the positions. limit
        gives the last key position of each query in place of its position
        (the tokens of one image, _media_limit).
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
            top = positions.max() if limit is None else max(positions.max(), limit.max())
            hi = int(np.searchsorted(kpos, top, side='right'))
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
        if limit is None:
            probs = ops.softmax_mask(scores, positions, n_rep, base, window)
        else:
            probs = ops.softmax_mask_limit(scores, positions, limit, base, window)
        vb = V.transpose(1, 0, 2)
        out = np.matmul(probs.reshape(nk, t * n_rep, n), vb)
        return out.reshape(nk, t, n_rep, hd).transpose(1, 0, 2, 3).reshape(t, plan.q_dim)

    # ---- output head -------------------------------------------------------
    def _gpu_cached(self):
        """The host cache that is on the GPU, or None. The model keeps a weak
        reference to it: a cache that the caller no longer holds is gone."""
        r = self.__dict__.get("_gpu_cache")
        return r() if r is not None else None

    def _gpu_attach(self, cache):
        """Return the GPU runner with this cache on the GPU. The first use of
        a cache copies it to the GPU. From then on, the GPU has the new rows,
        until _gpu_release (or gpu_sync) writes them into the host cache.

        The cache that was on the GPU before gets its new rows only when the
        caller still holds it (a Session of the server does). A cache that no
        one holds (a new KVCache for each request, or each rep of a
        benchmark) needs no copy: the copy of 2048 rows of the 12B took 0.6 s
        before each prompt."""
        g = self.__dict__.get("_gpu")
        if g is None:
            from . import gpu
            g = self._gpu = gpu.ModelGPU(self)
            self._gpu_cache = None
        prev = self._gpu_cached()
        if prev is not cache:
            if prev is not None:
                g.detach(prev)
            g.attach(cache)
            self._gpu_cache = weakref.ref(cache)
        else:
            g.kv.sync(cache)
        return g

    def gpu_sync(self, cache):
        """Write the rows that the GPU made into the host cache, and keep the
        cache on the GPU: a server reads or saves a cache with it while the
        GPU goes on with it (the next step needs no new copy to the GPU). A
        cache that is not on the GPU is up to date already."""
        g = self.__dict__.get("_gpu")
        if g is not None and self._gpu_cached() is cache:
            g.kv.sync(cache)
            g.kv.detach(cache)

    def _gpu_step(self, ids, cache, pos):
        """Run a decode step on the GPU."""
        self._gpu_xn = self._gpu_attach(cache).step(ids, pos)
        self._gpu_mirror_rows(cache)
        return self._gpu_xn

    def _gpu_group(self, ids, cache, pos):
        """Run a group of up to 16 tokens on the GPU, such as the verify
        group of an MTP step."""
        self._gpu_xn = self._gpu_attach(cache).group(list(ids), pos)
        self._gpu_mirror_rows(cache)
        return self._gpu_xn

    def _gpu_prefill(self, ids, cache, start, media=None):
        """Run a prompt on the GPU. Return the hidden states of every
        token (see ModelGPU.prefill)."""
        g = self._gpu_attach(cache)
        if media:
            self._gpu_xn = g.prefill(list(ids), start, media=media)
        else:
            self._gpu_xn = g.prefill(list(ids), start)
        self._gpu_mirror_rows(cache)
        return self._gpu_xn

    def gpu_mirror(self, cache, layers):
        """Keep the rows of some layers of the host cache up to date while
        the GPU has the cache. The MTP drafter reads the cache of two layers
        on the CPU (assistant.shared_layers). The GPU then writes the new
        rows of those layers into the host cache after each step. None
        stops it."""
        self._gpu_mirror = layers
        self._gpu_mirror_rows(cache)

    def _gpu_mirror_rows(self, cache):
        layers = self.__dict__.get("_gpu_mirror")
        g = self.__dict__.get("_gpu")
        if layers and g is not None and self._gpu_cached() is cache:
            g.kv.sync(cache)
            g.kv.detach(cache, layers)

    def truncate_cache(self, cache, n):
        """Cut the cache back to n positions (KVCache.truncate), and its copy
        on the GPU too when the GPU has it. Return False when a layer lacks
        the rows that a token at n sees; the caller must then start again."""
        g = self.__dict__.get("_gpu")
        if g is not None and self._gpu_cached() is cache:
            # The GPU copy drops its rows on its own, and while it holds the
            # cache the host rows lag behind it: sync() cannot find a cut.
            if not g.kv.truncate(n):
                return False
        return cache.truncate(n)

    def window_snapshot(self, cache, n):
        """The rows of the window of each sliding layer that a token at n
        sees (positions n - window + 1 .. n - 1), for window_restore: a
        decode past two windows drops them (KVCache.prepare, GPUKV.prepare),
        and the next turn of a chat, whose history goes back to n (the
        template drops the thought of the answer), then read the whole prompt
        again (the 26B: 84 s for 108K tokens). None for a model with no
        window. The GPU copy when the GPU holds the cache."""
        w = self.cfg.sliding_window or 0
        if not w or n < 1:
            return None
        start = max(0, n - w + 1)
        g = self.__dict__.get("_gpu")
        on_gpu = g is not None and self._gpu_cached() is cache
        rows = {}
        for i, plan in enumerate(self.cfg.plan):
            if not plan.is_sliding:
                continue
            if on_gpu:
                got = g.kv.window_get(i, start, n - start)
            elif cache.base[i] <= start and cache.end[i] >= n and cache._cap(i):
                r = slice(start - cache.base[i], n - cache.base[i])
                got = {name: a[r].copy() for name, a in zip(("kq", "ks", "vq", "vs"), cache._buffers(i))}
            else:
                got = None
            if got is None:
                return None
            rows[i] = got
        return {"n": n, "start": start, "rows": rows}

    def window_restore(self, cache, snap):
        """Cut the cache back to snap["n"] positions with the rows of the
        window of window_snapshot: the global layers keep their rows (end n),
        the sliding layers get the rows of the snapshot (base start). The
        GPU copy too when the GPU holds the cache. Return True."""
        n, start = snap["n"], snap["start"]
        g = self.__dict__.get("_gpu")
        on_gpu = g is not None and self._gpu_cached() is cache
        for i, plan in enumerate(self.cfg.plan):
            if plan.is_sliding:
                got = snap["rows"][i]
                k = n - start
                if cache._cap(i) < k:
                    cache._grow(i, max(k, 2 * cache.window + KV_KEEP + 1))
                for dst, name in zip(cache._buffers(i), ("kq", "ks", "vq", "vs")):
                    dst[:k] = np.asarray(got[name]).reshape(dst[:k].shape)
                cache.base[i], cache.end[i] = start, n
                if on_gpu:
                    g.kv.window_put(i, got, start, n)
            else:
                cache.end[i] = min(cache.end[i], n)
                if on_gpu:
                    g.kv.end[i] = min(g.kv.end[i], n)
                    g.kv.host_end[i] = min(g.kv.host_end[i], cache.end[i])
        return True

    def _gpu_release(self, cache):
        """Write the rows of the GPU cache into the host cache before the CPU
        uses it."""
        g = self.__dict__.get("_gpu")
        if g is not None and self._gpu_cached() is cache:
            g.detach(cache)
            self._gpu_cache = None
            self._gpu_xn = None

    def argmax_rows(self, x):
        """The greedy token of each row of x: np.argmax of each row of
        logits(x). For the last rows of a GPU step, the GPU picks them and
        copies only the tokens (ModelGPU.argmax)."""
        xn = self.__dict__.get("_gpu_xn")
        if xn is not None and x.shape[0] <= 16 and (
                x is xn or (getattr(x, "base", None) is xn
                            and x.ctypes.data + x.nbytes == xn.ctypes.data + xn.nbytes)):
            return self._gpu.argmax(x.shape[0])
        return [int(v) for v in np.argmax(self.logits(x), axis=1)]

    def logits_topk(self, x, k, temperature):
        """Return the candidates of sampling of the rows x from the GPU
        (ModelGPU.topk), or None when x is not the last rows of a GPU step."""
        xn = self.__dict__.get("_gpu_xn")
        if xn is not None and x.shape[0] <= 16 and (
                x is xn or (getattr(x, "base", None) is xn
                            and x.ctypes.data + x.nbytes == xn.ctypes.data + xn.nbytes)):
            return self._gpu.topk(x.shape[0], k, temperature)
        return None

    def logits(self, x, chunk=32768, apply_softcap=True):
        """Return the logits for the hidden states x.

        Use the embedding table. The embeddings are tied to the output head.
        Apply the softcap when requested.
        """
        xn = self.__dict__.get("_gpu_xn")
        if xn is not None and apply_softcap and x.shape[0] <= 16:
            # The last rows of the last GPU step or group: the GPU runs the
            # head.
            if x is xn or (getattr(x, "base", None) is xn
                           and x.ctypes.data + x.nbytes == xn.ctypes.data + xn.nbytes):
                return self._gpu.logits(x.shape[0])
        px = self.__dict__.get("_parts_xn")
        if px is not None and (x is px or (getattr(x, "base", None) is px
                                           and x.ctypes.data + x.nbytes
                                           == px.ctypes.data + px.nbytes)):
            # The last step of the parts ran the head (np_gemma/parts.py).
            out = self._parts_logits[-x.shape[0]:].copy()
        elif self._embed_q6k is not None:
            out = ops.linear_q6k(x, self._embed_q6k_bytes, self.cfg.hidden_size)
        elif self._embed_q is not None:
            if self._dtype == "int4":
                out = ops.linear_int4(x, self._embed_q, self._embed_s,
                                      q8=self.prompt_act == "1")
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
    def prefill(self, ids, cache, start=0, hook=None, media=None):
        """Run the prompt. Use blocks to keep the GEMM in its fast range.

        The key and value cache holds the earlier blocks. The result is the
        same as one forward pass over the full prompt. start is the position
        of ids[0]. Use it to add tokens to a cache that already has data. The
        hook gives the time of each stage of every block.

        media is a list of media.Span (absolute positions): the soft rows of
        images and audio. A block does not split a span whose tokens see each
        other. With media on the CPU the prompt takes the program
        (np_gemma/prompt.py) as without, or the Python path.
        """
        media = [sp for sp in (media or ()) if sp.end > start]
        if (_GPU and hook is None and len(ids) > 1 and isinstance(cache, KVCache)
                and self._dtype == "int4" and self.keep_weights
                and (not media or (_MEDIA_GPU and _BIDIR_ALL))):
            if media:
                return self._gpu_prefill(ids, cache, start, media)
            return self._gpu_prefill(ids, cache, start)
        if media:
            self._gpu_release(cache)
        chunk = self.prefill_chunk
        x = None
        self._media = media or None
        try:
            off = 0
            while off < len(ids):
                end = min(off + chunk, len(ids))
                for sp in media:
                    if sp.bidir and sp.start < start + end < sp.end:
                        end = sp.end - start
                x = self.forward(ids[off:end], cache=cache, start_pos=start + off, hook=hook)
                off = end
        finally:
            self._media = None
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
        # The E4B model has its own cache (E4B.new_cache).
        make = getattr(self.model, "new_cache", None)
        self.cache = make(self.max_len) if make else KVCache(self.model.cfg, max_len=self.max_len)
        self.ids = []
        # The media keys of the soft tokens in the cache: {position: key}
        # (media.keys). A soft token matches only the same image or clip.
        self.media_keys = {}
        self._x = None

    def common(self, ids, media=None):
        """Return the count of the first ids that are already in the cache.
        media gives the soft-token spans of ids (media.Span)."""
        want = {}
        for sp in media or ():
            for j in range(sp.start, sp.end):
                want[j] = (sp.key, j - sp.start)
        n = min(len(ids), len(self.ids))
        i = 0
        while i < n and ids[i] == self.ids[i] and want.get(i) == self.media_keys.get(i):
            i += 1
        return i

    def _common(self, ids):
        return self.common(ids)

    def prefill(self, ids, media=None):
        """Put the tokens in the cache. Run the forward pass for the new tokens.
        media is a list of media.Span of ids: the soft rows of images and
        audio (Model.prefill)."""
        ids = list(ids)
        common = self.common(ids, media)
        if common < len(self.ids):
            # The history changed at position common. Drop the rows after it.
            cut = getattr(self.model, "truncate_cache", None)
            snap = getattr(self, "snap", None)
            if cut(self.cache, common) if cut else self.cache.truncate(common):
                self.ids = self.ids[:common]
            elif snap is not None and snap["n"] <= common and \
                    self.model.window_restore(self.cache, snap):
                # the window of the start of the last answer (window_snapshot):
                # read again only the tokens after it
                common = snap["n"]
                self.ids = self.ids[:common]
            else:
                self.reset()
                common = 0
            self.media_keys = {j: k for j, k in self.media_keys.items() if j < common}
        new = ids[common:]
        self.prefilled = len(new)
        if new:
            if media:
                self._x = self.model.prefill(new, self.cache, start=common, media=media)
            else:
                self._x = self.model.prefill(new, self.cache, start=common)
            self.ids = ids
            for sp in media or ():
                for j in range(sp.start, sp.end):
                    self.media_keys[j] = (sp.key, j - sp.start)
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

    def generate_stream(self, ids, max_new_tokens=1, eos_ids=(), sampler=None, media=None):
        """Yield one token id at a time.

        Run the new prompt tokens, then select one token for each step. The
        sampler holds the sampling settings. The default sampler selects the
        most probable token. The generator stops at an end token.
        """
        from .assistant import RowPicker
        from .sampling import Sampler

        ids = list(ids)
        self.prefill(ids, media)
        if self._x is None:
            # The prompt is the same as the cache. Run the last token again.
            self._x = self.model.forward(ids[-1:], cache=self.cache,
                                         start_pos=len(ids) - 1)
        # the window at the start of the answer: the next turn goes back here
        snap_of = getattr(self.model, "window_snapshot", None)
        self.snap = snap_of(self.cache, len(ids)) if snap_of else None
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
                nxt = RowPicker(self.model, x[-1:], sampler).token(0)
                self.mtp_stats = {}
                yield from mtp_stream(self.model, self.drafter, self.cache, self.ids,
                                      x[-1:], nxt, self.n_draft, eos_ids, sampler,
                                      max_new_tokens, self.mtp_stats)
                return
        for _ in range(max_new_tokens):
            # The cheapest path to the token of the sampler (GPU argmax, or
            # the candidates of the GPU; assistant.RowPicker).
            nxt = RowPicker(self.model, x[-1:], sampler).token(0)
            yield nxt
            if nxt in eos_ids:
                return
            x = self.model.forward([nxt], cache=self.cache, start_pos=pos)
            self.ids.append(nxt)
            pos += 1
