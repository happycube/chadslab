"""Run the Gemma 4 E4B text model from the mobile-ct checkpoint.

The E4B model is a dense Gemma 4 with 42 layers. It adds one idea that the
12B and 26B models do not have: Per-Layer Embeddings (PLE). PLE gives each
decoder layer its own small embedding for every token. The embedding table is
large, but a token reads one row of it, so the memory map keeps the table on
the disk and reads 2.7 KiB for each token.

The forward pass follows the reference implementation in
`transformers/modeling_gemma4.py`. The order of one decoder layer is:

    residual = x
    x = input_layernorm(x)
    x = attention(x)
    x = post_attention_layernorm(x)
    x = residual + x

    residual = x
    x = pre_feedforward_layernorm(x)
    x = mlp(x)
    x = post_feedforward_layernorm(x)
    x = residual + x

    residual = x
    x = per_layer_input_gate(x)
    x = gelu(x) * per_layer_input
    x = per_layer_projection(x)
    x = post_per_layer_input_norm(x)
    x = residual + x

    x = x * layer_scalar

Three details of this model differ from the 12B model:

1.  The head size changes with the layer. A sliding layer has head_dim 256.
    A global layer has head_dim 512. The reference calls this
    `per_layer_config`.
2.  The last 18 layers share the key and value projections of earlier layers.
    There is no `k_proj`, no `v_proj`, and no `k_norm` in those layers. The
    file still holds a `k_proj` and a `v_proj` for them; they are unused and
    this module does not read them.
3.  The key and the value are separate in every layer. The 12B model reuses
    the key as the value in the global layers. This model sets
    `attention_k_eq_v` to false.

The attention scale is 1.0. The query norm already gives the query unit root
mean square, so the reference does not divide by sqrt(head_dim).
"""
from __future__ import annotations

import json
import os
import warnings
import weakref
from dataclasses import dataclass

import numpy as np

from . import cops
from . import gguf as _gg
from . import ops
from . import rope as rope_mod


class _KQ:
    """A matrix in the blocks of the GGUF products: data (uint8), the ggml
    type, rows, and cols (as qwen.KMat)."""

    def __init__(self, data, type_, rows, cols):
        self.data, self.type, self.rows, self.cols = data, type_, rows, cols

    def c(self):
        return (self.data, self.type)


def _iq4xs_to_q8_0(blocks):
    """IQ4_XS blocks (256 values) as Q8_0 blocks (32 values): for each 32,
    the float16 scale d (ls - 32) and the int8 table values of the codes."""
    nb = blocks.shape[0]
    j = np.arange(8)
    sl = blocks["sl"].astype(np.int32)[:, j // 2] >> (4 * (j % 2)) & 0xF
    sh = blocks["sh"].astype(np.int32)[:, None] >> (2 * j) & 3
    ls = (sl | (sh << 4)) - 32
    out = np.empty(nb * 8, dtype=_gg._BLOCK_DT[_gg.Q8_0])
    out["d"] = (blocks["d"].astype(np.float32)[:, None] * ls).reshape(-1).astype(np.float16)
    q = blocks["qs"].reshape(nb, 8, 16)
    vals = _gg._IQ4_NL_VALUES.astype(np.int8)
    out["qs"] = np.concatenate([vals[q & 0x0F], vals[q >> 4]], axis=2).reshape(nb * 8, 32)
    return out


def _kq_linear(x, k):
    """x times the GGUF matrix k (see E4B.kq), with the int8 rows of x."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, k.cols)
    t = x.shape[0]
    xq = np.empty((t, k.cols), np.int8)
    xs = np.empty((t, k.cols // 32), np.float32)
    xm = np.empty((t, k.cols // 16), np.float32)
    cops.kq_quant_x(x, xq, xs, xm)
    out = np.empty((t, k.rows), np.float32)
    cops.kq_linear(k.data, k.type, k.rows, k.cols, xq, xs, xm, x, t, out)
    return out

PREFIX = "model.language_model."

# Run a decode step as one program in C. Set NP_GEMMA_PROGRAM=0 for the Python
# loop over the layers.
_PROGRAM = os.environ.get("NP_GEMMA_PROGRAM", "1") != "0"
# Run a decode step of one token and the output head on a CUDA GPU
# (np_gemma/gpu.py, SPLIT_PLAN.md, phase 3). NP_GEMMA_GPU=1 turns it on.
_GPU = os.environ.get("NP_GEMMA_GPU", "0") == "1"
# The output head is a top-level tensor. It is not inside the language model.
# This checkpoint sets tie_word_embeddings to false, so the head is its own
# quantized matrix and not the embedding table.
HEAD = "lm_head"

# The smallest value count for a bfloat16 copy of a weight. A smaller matrix
# gives a lower rate from the C kernel than from the BLAS path, because the
# start of the kernel costs more than the read. The per-layer model projection
# has 27.5 M values and gains 2.3 times. The 8-bit per-layer maps of the
# mobile-ct file have 0.66 M values and lose a little.
_BF16_MIN = int(os.environ.get("NP_GEMMA_E4B_BF16_MIN", str(1 << 20)))


def round_bf16(x):
    """Round a number to bfloat16. Return the result as a Python float.

    Gemma 4 casts two scale constants to the weight dtype before the multiply.
    The checkpoint is bfloat16, so the constants lose precision. sqrt(2560) is
    50.596443 in float32 and 50.5 in bfloat16. Apply the same rounding, or the
    output drifts.
    """
    u = np.array([x], dtype=np.float32).view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    u = (u + (np.uint32(0x7FFF) + lsb)) & np.uint32(0xFFFF0000)
    return float(u.view(np.float32)[0])


@dataclass
class LayerPlan:
    """Give the geometry of one attention layer."""

    idx: int
    is_sliding: bool
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    window: int
    shared: bool
    source: int
    stores: bool

    @property
    def kind(self):
        """Return the key that the cache uses for this layer type."""
        return "sliding_attention" if self.is_sliding else "full_attention"

    @property
    def q_dim(self):
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self):
        return self.num_kv_heads * self.head_dim


class E4BConfig:
    """Read the text configuration and make a plan for each layer."""

    def __init__(self, cfg):
        tc = cfg["text_config"]
        self.raw = cfg
        self.hidden_size = tc["hidden_size"]
        self.intermediate_size = tc["intermediate_size"]
        self.num_hidden_layers = tc["num_hidden_layers"]
        self.num_attention_heads = tc["num_attention_heads"]
        self.num_key_value_heads = tc["num_key_value_heads"]
        self.head_dim = tc["head_dim"]
        self.global_head_dim = tc["global_head_dim"]
        self.attention_k_eq_v = bool(tc.get("attention_k_eq_v", False))
        self.num_global_key_value_heads = tc.get("num_global_key_value_heads")
        self.num_kv_shared_layers = int(tc.get("num_kv_shared_layers") or 0)
        self.rms_norm_eps = tc["rms_norm_eps"]
        self.vocab_size = tc["vocab_size"]
        self.vocab_size_per_layer_input = tc.get("vocab_size_per_layer_input") or tc["vocab_size"]
        self.hidden_size_per_layer_input = int(tc.get("hidden_size_per_layer_input") or 0)
        self.sliding_window = tc["sliding_window"]
        self.max_position_embeddings = tc["max_position_embeddings"]
        self.final_logit_softcapping = tc.get("final_logit_softcapping")
        self.rope_parameters = tc["rope_parameters"]
        self.layer_types = list(tc["layer_types"])
        self.hidden_activation = tc.get("hidden_activation", "gelu_pytorch_tanh")
        self.tie_word_embeddings = bool(tc.get("tie_word_embeddings", False))

        # The constants that the reference casts to bfloat16.
        self.embed_scale = round_bf16(self.hidden_size ** 0.5)
        self.per_layer_input_scale = round_bf16(2.0 ** -0.5)
        self.per_layer_model_projection_scale = round_bf16(self.hidden_size ** -0.5)
        self.per_layer_embed_scale = round_bf16(self.hidden_size_per_layer_input ** 0.5)

        self.plan = self._plan_all()
        self._inv_freq = {}

    def _plan_all(self):
        """Make the plan for every layer.

        The key sharing starts at the first shared layer. The last layer of
        each attention type before that point holds the key and the value that
        the shared layers reuse.
        """
        first = self.num_hidden_layers - self.num_kv_shared_layers
        last_of_type = {}
        for i in range(max(first, 0)):
            last_of_type[self.layer_types[i]] = i
        plans = []
        for i, layer_type in enumerate(self.layer_types):
            sliding = layer_type == "sliding_attention"
            if sliding:
                head_dim = self.head_dim
                num_kv = self.num_key_value_heads
            else:
                head_dim = self.global_head_dim
                num_kv = (self.num_global_key_value_heads
                          if self.attention_k_eq_v and self.num_global_key_value_heads
                          else self.num_key_value_heads)
            shared = i >= first > 0
            plans.append(LayerPlan(
                idx=i,
                is_sliding=sliding,
                head_dim=head_dim,
                num_q_heads=self.num_attention_heads,
                num_kv_heads=num_kv,
                window=self.sliding_window if sliding else 0,
                shared=shared,
                source=last_of_type.get(layer_type, -1) if shared else i,
                stores=(not shared) and last_of_type.get(layer_type) == i,
            ))
        return plans

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def rope_inv_freq(self, plan):
        """Return the inverse frequencies for one layer. Cache by type and head size."""
        key = (plan.kind, plan.head_dim)
        if key in self._inv_freq:
            return self._inv_freq[key]
        params = self.rope_parameters[plan.kind]
        if plan.is_sliding:
            inv = rope_mod.default_inv_freq(plan.head_dim, params["rope_theta"])
        else:
            inv = rope_mod.proportional_inv_freq(
                plan.head_dim, params["rope_theta"],
                params.get("partial_rotary_factor", 1.0))
        self._inv_freq[key] = inv
        return inv

    def describe(self):
        """Return a short report about the model."""
        n_shared = sum(1 for p in self.plan if p.shared)
        return {
            "layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "ple_dim": self.hidden_size_per_layer_input,
            "shared_kv_layers": n_shared,
            "full_attention_layers": [p.idx for p in self.plan if not p.is_sliding],
            "k_eq_v": self.attention_k_eq_v,
            "sliding_window": self.sliding_window,
        }


class E4BCache:
    """Keep the key and the value of every layer.

    The layers before the sharing point keep their own key and value. The
    layers after it reuse the data of the last layer of the same type.

    The buffers are made one time and a new key is written into its place. A
    concatenate for each step copies the whole history: at a context of 512
    tokens that is 63 MB and 6 ms for each token.

    The key and the value of one head lie together, and the keys of one head
    follow each other. The attention kernel of a decode step then reads them
    in order. The other order costs 2.4 times in the dot product, because the
    hardware prefetch jumps over the other head for each key.

    The cache keeps the full history. A sliding layer applies the window
    through the mask, not by throwing the old keys away. The two forms agree:
    a key outside the window has a weight of zero.
    """

    def __init__(self, cfg, max_len=None):
        self.cfg = cfg
        self.cap = max(256, int(max_len or 0))
        self.n = 0
        self.kv = {}
        self.shared = {}

    def reset(self):
        self.n = 0
        self.kv.clear()
        self.shared.clear()

    def length(self):
        """Return the number of positions the cache holds."""
        return self.n

    def _reserve(self, need):
        """Make every buffer large enough for `need` positions."""
        if need <= self.cap:
            return
        cap = self.cap
        while cap < need:
            cap *= 2
        seen = set()
        for store in list(self.kv.values()) + list(self.shared.values()):
            if id(store) in seen:
                continue
            seen.add(id(store))
            if store[0] is None:
                continue
            for i in (0, 1):
                old = store[i]
                new = np.empty((cap,) + old.shape[1:], np.float32)
                new[:self.n] = old[:self.n]
                store[i] = new
        self.cap = cap

    def append(self, plan, k, v, start):
        """Write new keys and values for one layer. Return the full history."""
        count = k.shape[0]
        self._reserve(start + count)
        store = self.kv.get(plan.idx)
        if store is None:
            store = [None, None]
            self.kv[plan.idx] = store
        if store[0] is None:
            # position-major (positions, kv heads, head_dim), as all the
            # caches (gpumm shares the part past the positions in use)
            shape = (self.cap, k.shape[1], k.shape[2])
            store[0] = np.empty(shape, np.float32)
            store[1] = np.empty(shape, np.float32)
        store[0][start:start + count] = k
        store[1][start:start + count] = v
        if plan.stores:
            # The shared layers read the same buffer, so the data stays in one
            # place.
            self.shared[plan.kind] = store
        if start + count > self.n:
            self.n = start + count
        return self._heads(store)

    def shared_kv(self, plan):
        """Return the key and the value that a shared layer reuses."""
        store = self.shared.get(plan.kind)
        if store is None:
            raise RuntimeError(
                "layer %d wants the shared %s key and value, but layer %d has "
                "not stored them" % (plan.idx, plan.kind, plan.source))
        return self._heads(store)

    def _heads(self, store):
        """The keys and the values of positions 0 .. n - 1 as (kv heads, n,
        head_dim) views of the position-major store."""
        return store[0][:self.n].transpose(1, 0, 2), store[1][:self.n].transpose(1, 0, 2)

    def truncate(self, n):
        """Cut the cache back to n positions. Use it to reuse a prefix.
        Return True, as KVCache.truncate does when it can cut the cache."""
        self.n = min(self.n, int(n))
        return True


class E4B:
    """Load the weights and run the Gemma 4 E4B text model.

    Set the mode to choose how the model keeps the weights:

        "f32"     Copy every weight into float32 memory one time. The decode
                  work happens once. The memory holds about 15 GB.
        "stream"  Keep the file mapped and decode a weight each time the model
                  uses it. The memory holds one layer. The decode work repeats
                  for every token.
        "int4"    Keep the 4-bit and the 2-bit matrices packed and let the C
                  kernel read the packed words. Keep the rest in float32. The
                  memory holds the packed weights, 2.2 GB, and a token reads
                  them instead of a float32 copy.

    The argument `resident` is the older spelling of the mode: True is "f32"
    and False is "stream".
    """

    def __init__(self, ct, cfg, mode="f32", resident=None):
        if resident is not None:
            mode = "f32" if resident else "stream"
        if mode not in ("f32", "stream", "int4"):
            raise ValueError("mode must be f32, stream, or int4")
        self.ct = ct
        self.cfg = cfg
        self.mode = mode
        # Keep a float32 copy of a weight when the mode allows it. The int4
        # mode keeps only the small tensors: the norms, the 8-bit per-layer
        # gates, and the per-layer model projection.
        self.resident = mode in ("f32", "int4")
        self._w = {}
        self._packed = {}
        self._kq = {}           # the GGUF matrices of the GGUF products (kq)
        # The bfloat16 copy of a weight that the quantization did not touch.
        self._bf16 = {}
        # The cosine and sine tables of the rope, by layer type and position.
        self._rope = {}
        # A GGUF file gives the block layout of Q4_0, which is the layout of
        # the int4 kernel of the 12B model. A compressed-tensors file gives its
        # own packed layout. The two need different kernels.
        self._q4 = hasattr(ct, "int4_packed")
        # The GGUF file has no output head. The head is the token embedding:
        # the two matrices hold the same values in the checkpoint.
        self.head = (PREFIX + "embed_tokens") if self._q4 else HEAD
        self._head_q6k = None
        self._head_q4 = None
        # The GPU runner, the cache that is on the GPU, and the hidden state
        # of the last GPU step (see _gpu_step).
        self._gpu = None
        self._gpu_cache = None
        self._gpu_xn = None
        self._embed_tables = (PREFIX + "embed_tokens",
                              PREFIX + "embed_tokens_per_layer")
        if mode == "int4" and os.environ.get("OPENBLAS_NUM_THREADS", "1") != "1":
            # The C kernel runs one OpenMP region for each matrix, and the
            # model makes about 300 calls for each layer pass. A BLAS pool with
            # more than one thread fights those regions. The measured cost on
            # a six-core machine is 2.5 s a token against 0.33 s. Set
            # OPENBLAS_NUM_THREADS=1 before the process starts; NumPy reads the
            # variable when it loads, so a later change has no effect.
            warnings.warn(
                "the int4 mode is about 8 times slower with more than one "
                "BLAS thread; set OPENBLAS_NUM_THREADS=1 (it is %s)"
                % os.environ.get("OPENBLAS_NUM_THREADS"), stacklevel=2)

    def close(self):
        self._w.clear()
        self._packed.clear()
        self._bf16.clear()
        self._rope.clear()
        self.ct.close()

    def rope_tables(self, plan, start_pos, ntok):
        """Return the cosine and sine tables of the rope for one layer.

        Every layer of the same type uses the same table. A prompt has two
        types, so the model makes two tables in place of 42. A new table costs
        about 2.5 microseconds for each 1000 values, and the layer loop calls
        it 42 times. The cache holds a few entries, because a long generation
        makes a new table for each token.
        """
        key = (plan.kind, plan.head_dim, start_pos, ntok)
        entry = self._rope.get(key)
        if entry is None:
            if len(self._rope) > 8:
                self._rope.clear()
            entry = rope_mod.cos_sin(self.cfg.rope_inv_freq(plan),
                                     np.arange(start_pos, start_pos + ntok))
            self._rope[key] = entry
        return entry

    # ---- weights -----------------------------------------------------------
    def W(self, module):
        """Return a weight matrix as float32. `module` has no `.weight` suffix."""
        w = self._w.get(module)
        if w is None:
            if self._q4:
                w = self.ct.get(module + ".weight")
            else:
                w = self.ct.dequant(module)
            if self.resident:
                self._w[module] = w
        return w

    def q4(self, module):
        """Return the Q4_0 blocks of a weight in the int4 layout, or None.

        This is the path of a GGUF file. The layout is the block layout of
        Q4_0, which is the layout that the int4 kernel of the 12B model takes.
        """
        entry = self._packed.get(module)
        if entry is not None:
            return entry if entry is not False else None
        try:
            entry = self.ct.int4_packed(module + ".weight")
        except (ValueError, KeyError):
            entry = False
        self._packed[module] = entry
        return entry if entry is not False else None

    def packed(self, module):
        """Return the packed words and the row scale of a weight, or None.

        Return None when the weight is not in the packed form that the C kernel
        takes: a 4-bit or a 2-bit weight with one scale for each row. A group
        scale, an int8 weight, and a bfloat16 weight take the decoded path.
        """
        entry = self._packed.get(module)
        if entry is not None:
            return entry if entry is not False else None
        bits = self.ct.num_bits(module)
        words = self.ct.packed_words(module) if bits in (2, 4) else None
        scale = self.ct.channel_scale(module) if words is not None else None
        if words is None or scale is None:
            self._packed[module] = False
            return None
        entry = (words, scale, bits)
        self._packed[module] = entry
        return entry

    def W16(self, module):
        """Return a weight as bfloat16 values for the C kernel, or None.

        A packed 4-bit weight stays packed and does not take this path. Every
        other weight becomes float32 in the mode "int4". Two bytes for each
        value read by the full thread team beat four bytes read by one BLAS
        thread, so the function makes a bfloat16 copy one time.

        The bfloat16 kernel needs a large matrix to pay for its start. A small
        matrix keeps the float32 path. Set NP_GEMMA_E4B_BF16=0 to compare the
        two paths, and NP_GEMMA_E4B_BF16_MIN to change the size limit.
        """
        if os.environ.get("NP_GEMMA_E4B_BF16", "1") != "1":
            return None
        entry = self._bf16.get(module)
        if entry is not None:
            return entry if entry is not False else None
        w16 = None
        try:
            w = self.W(module)
        except (KeyError, ValueError):
            w = None
        if w is not None and w.size >= _BF16_MIN:
            w16 = ops.to_bf16(w)
            # The float32 copy is no longer necessary.
            self._w.pop(module, None)
        self._bf16[module] = w16 if w16 is not None else False
        return w16

    def kq(self, module):
        """Return a matrix of a GGUF file in the blocks of the GGUF products
        (csrc/kquants.c, GP_KQ_LINEAR), or None: the K quants (Q4_K, Q5_K,
        Q6_K) and Q8_0 of the UD files of unsloth. An IQ4_XS matrix becomes
        Q8_0: the codes become their int8 table values, and the scale of
        each 32 values, d (ls - 32), becomes a float16 (a relative change of
        at most 2^-11). The object has data, type, rows, cols, and c(), as
        qwen.KMat."""
        e = self._kq.get(module)
        if e is not None:
            return e if e is not False else None
        e = False
        if self._q4:
            try:
                gname = self.ct._gguf(module + ".weight")
                dims, t, _off = self.ct.tensors[gname]
            except (KeyError, AttributeError):
                t = None
            if t in (_gg.Q4_K, _gg.Q5_K, _gg.Q6_K, _gg.Q8_0, _gg.IQ4_XS) and len(dims) == 2:
                blocks, dims, t = self.ct.raw(gname)
                if t == _gg.IQ4_XS:
                    blocks, t = _iq4xs_to_q8_0(blocks), _gg.Q8_0
                e = _KQ(np.ascontiguousarray(blocks).view(np.uint8).reshape(-1), t,
                        int(dims[1]), int(dims[0]))
        self._kq[module] = e
        return e if e is not False else None

    def kq_q4(self, module):
        """Return a Q4_0 matrix of a GGUF file as a matrix of the GGUF products
        (ggml type 2, as kq), or None. The GPU then runs it with int8 x and
        dp4a (GP_KQ_LINEAR, kq_rows_i8), as the K quants of the UD files. The
        data is a view of the int4 blocks (q4), so the GPU holds one copy."""
        q = self.q4(module)
        if q is None:
            return None
        packed = q[0]
        return _KQ(packed.reshape(-1), _gg.Q4_0, int(packed.shape[0]), int(packed.shape[1]) * 32)

    def linear(self, x, module):
        """Multiply x by a weight matrix of the model.

        In the mode "int4" a packed weight stays packed. A GGUF file gives the
        block layout of Q4_0, so the model uses the int4 kernel of the 12B
        model. A compressed-tensors file gives its own layout, so the model
        uses the packed kernel of that layout. Every other weight uses a
        bfloat16 copy and the bfloat16 kernel.
        """
        if self.mode == "int4":
            # A small token group (an MTP verify step) uses the kernels that
            # give each token the bits of a decode step.
            mt = ops.mt_ready(x.shape[0])
            if self._q4:
                entry = self.q4(module)
                if entry is not None:
                    packed, scales = entry
                    if mt:
                        return ops.linear_int4_mt(x, packed, scales)
                    return ops.linear_int4(x, packed, scales)
                k = self.kq(module)
                if k is not None:
                    return _kq_linear(x, k)
            else:
                entry = self.packed(module)
                if entry is not None:
                    words, scale, bits = entry
                    return cops.ct_linear(x, words, scale, bits)
            w16 = self.W16(module)
            if w16 is not None:
                if mt:
                    return ops.linear_bf16_mt(x, w16)
                return ops.linear_bf16(x, w16)
        return ops.linear(x, self.W(module))

    def linear_multi(self, x, modules):
        """Multiply one row of x by two or more packed weights.

        The matrices share the input row, so one kernel call runs them all.
        The model then opens one OpenMP region instead of one for each matrix.
        Return a list of arrays in the order of `modules`.
        """
        mt = ops.mt_ready(x.shape[0])
        if (self.mode != "int4" or not self._q4 or len(modules) < 2
                or (x.shape[0] != 1 and not mt) or not ops.int4_multi4_ready()):
            return [self.linear(x, m) for m in modules]
        entries = [self.q4(m) for m in modules]
        if any(e is None for e in entries):
            return [self.linear(x, m) for m in modules]
        if mt:
            return ops.int4_multi4_mt(entries, x, x.shape[-1])[:len(modules)]
        outs = ops.int4_multi4(entries, x, x.shape[-1])[:len(modules)]
        return [None if o is None else o.reshape(1, -1) for o in outs]

    def T(self, key):
        """Return a tensor that the quantization did not touch. Use the full key."""
        w = self._w.get(key)
        if w is None:
            if self._q4:
                w = self.ct.get(key)
            else:
                w = self.ct.plain(key)
            if self.resident:
                self._w[key] = w
        return w

    def embed_rows(self, module, ids):
        """Read the rows of an embedding table. Read only the bytes of the rows.

        The memory map makes this operation cheap: ask for the bytes of one
        row, not for the whole table.
        """
        ids = np.asarray(ids).reshape(-1)
        if self._q4 and hasattr(self.ct, "take_rows"):
            return self.ct.take_rows(module + ".weight", ids)
        if self._q4:
            return np.stack([self.ct.get_row(module + ".weight", int(t)) for t in ids])
        return np.stack([self.ct.row(module, int(t)) for t in ids])

    def load_all(self):
        """Prepare every weight for the current mode, before the first token."""
        for i in range(self.cfg.num_hidden_layers):
            self.load_layer(i)

    def warm(self, module):
        """Prepare one weight for the current mode.

        In the int4 mode a packed weight stays packed: only the scale is read
        out of the file, one value for each row. Every other weight becomes
        bfloat16, and the C kernel reads it with the full thread team.
        """
        if self.mode == "int4":
            if self._q4:
                if self.q4(module) is not None:
                    return
            elif self.packed(module) is not None:
                return
            if self.W16(module) is not None:
                return
        self.W(module)

    def load_layer(self, i):
        """Prepare the weights of one layer for the current mode."""
        p = PREFIX + "layers." + str(i) + "."
        plan = self.cfg.plan[i]
        modules = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                   "self_attn.q_proj", "self_attn.o_proj",
                   "per_layer_input_gate", "per_layer_projection"]
        keys = ["input_layernorm.weight", "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                "post_per_layer_input_norm.weight", "layer_scalar",
                "self_attn.q_norm.weight"]
        if not plan.shared:
            modules += ["self_attn.k_proj", "self_attn.v_proj"]
            keys += ["self_attn.k_norm.weight"]
        for m in modules:
            self.warm(p + m)
        for k in keys:
            self.T(p + k)
        self.warm(PREFIX + "per_layer_model_projection")
        self.T(PREFIX + "norm.weight")
        self.T(PREFIX + "per_layer_projection_norm.weight")
        if not self._q4:
            self.warm(HEAD)

    # ---- the forward pass --------------------------------------------------
    def per_layer_inputs(self, ids, emb):
        """Build the per-layer input for every layer.

        The token part comes from `embed_tokens_per_layer`. The context part
        comes from a projection of the main embedding. The model adds the two
        and scales by 1/sqrt(2).
        """
        cfg = self.cfg
        n = cfg.hidden_size_per_layer_input
        n_layers = cfg.num_hidden_layers
        ntok = len(ids)
        tok = self.embed_rows(PREFIX + "embed_tokens_per_layer", ids)
        tok = (tok * cfg.per_layer_embed_scale).reshape(ntok, n_layers, n)
        proj = self.linear(emb, PREFIX + "per_layer_model_projection")
        proj = proj * cfg.per_layer_model_projection_scale
        proj = proj.reshape(ntok, n_layers, n)
        proj = ops.rms_norm(proj, self.T(PREFIX + "per_layer_projection_norm.weight"),
                            cfg.rms_norm_eps)
        return (proj + tok) * cfg.per_layer_input_scale

    def mlp(self, h, p):
        """Run the feed-forward block of one layer.

        The gate and the up projection share the input row. One kernel call
        runs both, so the model opens one OpenMP region.
        """
        gate, up = self.linear_multi(h, [p + "mlp.gate_proj", p + "mlp.up_proj"])
        return self.linear(ops.gelu_tanh(gate) * up, p + "mlp.down_proj")

    def attention(self, h, p, plan, cache, start_pos, hook=None):
        """Run the attention block of one layer."""
        cfg = self.cfg
        ntok = h.shape[0]
        head_dim = plan.head_dim

        # One call gives the norm of the query, the key, and the value, and a
        # second call gives the two rope rotations. The kernels work in place
        # and use the OpenMP pool. Three separate norms and two separate
        # rotations make five calls and five regions for each layer, and the
        # NumPy rotation also builds a second array for each call.
        if plan.shared:
            # The key and the value of a shared layer belong to the layer that
            # stored them, and they are already normal and turned.
            q = self.linear(h, p + "self_attn.q_proj")
            q = q.reshape(ntok * plan.num_q_heads, head_dim)
            ops.qkv_norm(q, self.T(p + "self_attn.q_norm.weight"),
                         None, None, None, cfg.rms_norm_eps)
        else:
            # The query, the key, and the value projection share the input
            # row. One kernel call runs all three.
            q, k, v = self.linear_multi(
                h, [p + "self_attn.q_proj", p + "self_attn.k_proj",
                    p + "self_attn.v_proj"])
            q = q.reshape(ntok * plan.num_q_heads, head_dim)
            k = k.reshape(ntok * plan.num_kv_heads, head_dim)
            v = v.reshape(ntok * plan.num_kv_heads, head_dim)
            # The value norm has no scale. Give None for its weight.
            ops.qkv_norm(q, self.T(p + "self_attn.q_norm.weight"),
                         k, self.T(p + "self_attn.k_norm.weight"), v,
                         cfg.rms_norm_eps)

        cos, sin = self.rope_tables(plan, start_pos, ntok)
        if plan.shared:
            ops.rope_apply(q, None, cos, sin, plan.num_q_heads, 0, head_dim)
            q = q.reshape(ntok, plan.num_q_heads, head_dim)
            k, v = cache.shared_kv(plan)
        else:
            ops.rope_apply(q, k, cos, sin, plan.num_q_heads,
                           plan.num_kv_heads, head_dim)
            q = q.reshape(ntok, plan.num_q_heads, head_dim)
            k = k.reshape(ntok, plan.num_kv_heads, head_dim)
            v = v.reshape(ntok, plan.num_kv_heads, head_dim)
            k, v = cache.append(plan, k, v, start_pos)

        npos = k.shape[1]
        if npos != start_pos + ntok:
            raise RuntimeError("layer %d sees %d positions, expected %d"
                               % (plan.idx, npos, start_pos + ntok))

        slide = bool(plan.window) and os.environ.get("NP_GEMMA_SLIDE", "1") == "1"
        if (1 < ntok and ops.attn_ready() and self.mode == "int4"
                and ops.mt_ready(ntok)):
            # A small token group: the decode-step attention for each token,
            # over the keys that a decode step of that token sees.
            out = np.empty((ntok, plan.num_q_heads, head_dim), dtype=np.float32)
            for j in range(ntok):
                pos = start_pos + j
                lo = max(0, pos - plan.window + 1) if slide else 0
                out[j] = ops.attn_decode_f32(q[j:j + 1], k[:, lo:pos + 1, :],
                                             v[:, lo:pos + 1, :], pos, lo, plan.window)[0]
            return self.linear(out.reshape(ntok, plan.q_dim), p + "self_attn.o_proj")

        positions = np.arange(start_pos, start_pos + ntok, dtype=np.int32)
        base = 0
        # A sliding layer sees only the last window keys. Drop the keys that no
        # query in this block can see, so the products and the mask are
        # smaller. The dropped keys lie outside the window of every query, so
        # the result does not change. A prompt of 256 tokens is shorter than
        # the window, and then the test does nothing.
        if plan.window and os.environ.get("NP_GEMMA_SLIDE", "1") == "1":
            lo = max(0, int(positions.min()) - plan.window + 1)
            hi = min(npos, int(positions.max()) + 1)
            if lo or hi < npos:
                k = k[:, lo:hi, :]
                v = v[:, lo:hi, :]
                base = lo
                npos = hi - lo

        group = plan.num_q_heads // plan.num_kv_heads
        if ntok == 1 and ops.attn_ready():
            # One C call for the scores, the softmax, and the output. The
            # kernel reads the cache in place. The batched matmul of the path
            # below needs a transpose of the key for each layer, and a copy of
            # 2 MB costs more than the arithmetic.
            out = ops.attn_decode_f32(q, k, v, start_pos, base, plan.window)
            return self.linear(out.reshape(ntok, plan.q_dim),
                               p + "self_attn.o_proj")
        # The C flash kernel walks only the keys that the mask leaves visible,
        # and it uses the OpenMP pool in place of one BLAS thread. The batched
        # matmul gives the score matrix to OpenBLAS, which tiles a large matrix
        # better than the small register tile of the kernel. E4B measures the
        # kernel ahead at every prompt length: 1.13 times at 256 tokens and
        # 1.24 times at 1024. Thus the default here is 1, and the 12B model
        # keeps its own default of 0. The value "slide" uses the kernel for a
        # sliding layer and the matmul for a global layer.
        flash = os.environ.get("NP_GEMMA_FLASH", "1")
        use_flash = flash != "0" and (flash != "slide" or plan.window > 0)
        if ntok > 1 and use_flash and ops.flash_ready():
            # The flash kernel wants the keys and the values in the order of
            # position, so the two arrays change their shape for this path.
            out = ops.flash_prefill(q, k.transpose(1, 0, 2), v.transpose(1, 0, 2),
                                    positions, base, plan.window)
            out = out.reshape(ntok, plan.q_dim)
            return self.linear(out, p + "self_attn.o_proj")
        # Use a batched matrix multiply. The code makes one matrix for each
        # key and value head. matmul is faster than einsum here, because einsum
        # looks for a contraction path at each call: a prompt of 256 tokens is
        # 18 times faster. The einsum form also needs a repeat of the key and
        # the value, which matmul does not.
        qb = q.reshape(ntok, plan.num_kv_heads, group, head_dim)
        qb = qb.transpose(1, 0, 2, 3).reshape(plan.num_kv_heads, ntok * group, head_dim)
        kb = k.transpose(0, 2, 1)
        # The attention scale is 1.0. Do not divide by sqrt(head_dim).
        scores = np.matmul(qb, kb).reshape(plan.num_kv_heads, ntok, group, npos)
        # The kernel applies the causal mask, the window mask, and the softmax
        # over the last axis in one pass.
        probs = ops.softmax_mask(scores, positions, group, base, plan.window)
        out = np.matmul(probs.reshape(plan.num_kv_heads, ntok * group, npos), v)
        out = out.reshape(plan.num_kv_heads, ntok, group, head_dim)
        out = out.transpose(1, 0, 2, 3).reshape(ntok, plan.q_dim)
        return self.linear(out, p + "self_attn.o_proj")

    def layer(self, x, per_layer_input, i, cache, start_pos, hook=None):
        """Run one decoder layer."""
        cfg = self.cfg
        p = PREFIX + "layers." + str(i) + "."
        plan = cfg.plan[i]

        residual = x
        h = ops.rms_norm(x, self.T(p + "input_layernorm.weight"), cfg.rms_norm_eps)
        h = self.attention(h, p, plan, cache, start_pos, hook)
        h = ops.rms_norm(h, self.T(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps)
        x = residual + h

        residual = x
        h = ops.rms_norm(x, self.T(p + "pre_feedforward_layernorm.weight"), cfg.rms_norm_eps)
        h = self.mlp(h, p)
        h = ops.rms_norm(h, self.T(p + "post_feedforward_layernorm.weight"), cfg.rms_norm_eps)
        x = residual + h

        # The per-layer embedding gives this layer its own residual signal.
        residual = x
        h = self.linear(x, p + "per_layer_input_gate")
        h = ops.gelu_tanh(h)
        h = h * per_layer_input
        h = self.linear(h, p + "per_layer_projection")
        h = ops.rms_norm(h, self.T(p + "post_per_layer_input_norm.weight"), cfg.rms_norm_eps)
        x = residual + h

        x = x * self.T(p + "layer_scalar")
        if hook is not None:
            hook(i, x)
        return x

    def forward(self, input_ids, cache=None, start_pos=0, hook=None, media=None):
        """Run the model. Return the final hidden state.

        input_ids  The token ids. A list or an array.
        cache      An E4BCache. Make one when the argument is None.
        start_pos  The position of the first token. Use it after a prefill.
        hook       A callable (layer, hidden) for a trace.
        media      A list of media.Span (absolute positions): the soft rows of
                   images and audio. A soft token takes its row in place of
                   the token row (no scale), and the per-layer token row of
                   the pad token (id 0), as Gemma4Model of transformers.
        """
        cfg = self.cfg
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        t0 = int(start_pos)
        media = [sp for sp in media or () if sp.start < t0 + ids.size and sp.end > t0]
        if _GPU and hook is None and isinstance(cache, E4BCache):
            from . import program
            if ids.size == 1 and not media and program.e4b_ready(self, cache):
                return self._gpu_step(ids, cache, int(start_pos))
            if ids.size > 1 and self._q4 and self.mode == "int4":
                # A prompt, or the verify group of an MTP step.
                return self._gpu_prefill(ids, cache, int(start_pos), media)
        self._gpu_release(cache)
        if (_PROGRAM and hook is None and isinstance(cache, E4BCache) and not media
                and (ids.size == 1 or ops.mt_ready(ids.size))):
            # One decode step, or the group of an MTP verify step, as one
            # program in C (np_gemma/program.py). The result has the bits of
            # the Python loop below.
            from . import program
            if program.e4b_ready(self, cache):
                return program.decode_step_e4b(self, cache, ids, int(start_pos))
        if cache is None:
            cache = E4BCache(cfg)
        emb = self.embed_rows(PREFIX + "embed_tokens", ids)
        x = emb * cfg.embed_scale
        tok_ids = ids
        for sp in media:
            lo, hi = max(sp.start, t0), min(sp.end, t0 + ids.size)
            x[lo - t0:hi - t0] = sp.rows[lo - sp.start:hi - sp.start]
            if tok_ids is ids:
                tok_ids = ids.copy()
            tok_ids[lo - t0:hi - t0] = 0
        per_layer = self.per_layer_inputs(tok_ids, x)
        for i in range(cfg.num_hidden_layers):
            x = self.layer(x, per_layer[:, i, :], i, cache, start_pos, hook)
        return ops.rms_norm(x, self.T(PREFIX + "norm.weight"), cfg.rms_norm_eps)

    def _gpu_cached(self):
        """The host cache that is on the GPU, or None (a weak reference, as
        Model._gpu_cached)."""
        r = self.__dict__.get("_gpu_cache")
        return r() if r is not None else None

    def _gpu_attach(self, cache):
        """Put the cache on the GPU. The first use of a cache copies it to the
        GPU. From then on, the copy on the GPU is the true one, until
        _gpu_release (or gpu_sync) copies it back. The cache that was on the
        GPU before gets its rows only when the caller still holds it (see
        Model._gpu_attach)."""
        if self._gpu is None:
            from . import gpu
            self._gpu = gpu.E4BGPU(self)
        prev = self._gpu_cached()
        if prev is not cache:
            if prev is not None:
                self._gpu.detach(prev)
            self._gpu.attach(cache)
            self._gpu_cache = weakref.ref(cache)

    def gpu_sync(self, cache):
        """Copy the cache on the GPU into the host cache, and keep it on the
        GPU (as Model.gpu_sync)."""
        if self._gpu is not None and self._gpu_cached() is cache:
            self._gpu.cache.to_host()

    def _gpu_step(self, ids, cache, pos):
        """Run a decode step on the GPU."""
        self._gpu_attach(cache)
        self._gpu_xn = self._gpu.step(ids, pos, cache)
        return self._gpu_xn

    def _gpu_prefill(self, ids, cache, pos, media=None):
        """Run a group or a prompt on the GPU (E4BGPU.prefill)."""
        self._gpu_attach(cache)
        if media:
            self._gpu_xn = self._gpu.prefill(ids, pos, cache, media=media)
        else:
            self._gpu_xn = self._gpu.prefill(ids, pos, cache)
        return self._gpu_xn

    def _gpu_release(self, cache):
        """Copy the cache back to the host before the CPU uses it."""
        if self._gpu is not None and self._gpu_cached() is cache:
            self._gpu.detach(cache)
            self._gpu_cache = None
            self._gpu_xn = None

    def embed(self, input_ids):
        """Return the scaled token embeddings, the input of layer 0."""
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        return self.embed_rows(PREFIX + "embed_tokens", ids) * self.cfg.embed_scale

    def new_cache(self, max_len):
        """Return an empty cache for this model. Session uses it."""
        return E4BCache(self.cfg, max_len=max_len)

    def prefill(self, ids, cache, start=0, media=None):
        """Run the prompt into the cache. Return the final hidden states.
        media: see forward."""
        return self.forward(ids, cache=cache, start_pos=start, media=media)

    def logits(self, hidden, softcap=True):
        """Project the hidden state onto the vocabulary."""
        xn = self._gpu_xn
        if (xn is not None and softcap and hidden.shape[0] <= 16
                and (hidden is xn or (getattr(hidden, "base", None) is xn and
                                      hidden.ctypes.data + hidden.nbytes ==
                                      xn.ctypes.data + xn.nbytes))):
            # The last rows of the last GPU step or group: the GPU runs the
            # head.
            return self._gpu.logits(hidden.shape[0])
        out = None
        if self._q4 and self._head_is_q6k():
            # The head of a GGUF file is the token embedding, and the file
            # keeps that table in Q6_K. The kernel reads the blocks in place,
            # so the head never becomes a float32 matrix.
            out = ops.linear_q6k(hidden, self._head_q6k, self.cfg.hidden_size)
        if out is None:
            out = self.linear(hidden, self.head)
        cap = self.cfg.final_logit_softcapping
        if softcap and cap:
            out = ops.softcap(out, cap)
        return out

    def logits_topk(self, hidden, k, temperature):
        """Return the candidates of sampling of the rows of hidden from the GPU
        (E4BGPU.topk: the k largest logits of each row and the row stats), or
        None when the rows are not the last rows of a GPU step."""
        xn = self._gpu_xn
        if (xn is not None and hidden.shape[0] <= 16
                and (hidden is xn or (getattr(hidden, "base", None) is xn and
                                      hidden.ctypes.data + hidden.nbytes ==
                                      xn.ctypes.data + xn.nbytes))):
            return self._gpu.topk(hidden.shape[0], k, temperature)
        return None

    def argmax_rows(self, hidden):
        """The greedy token of each row of hidden: np.argmax of each row of
        logits(hidden). For the last rows of a GPU step or group, the GPU
        picks them and copies only the tokens."""
        xn = self._gpu_xn
        if (xn is not None and hidden.shape[0] <= 16
                and (hidden is xn or (getattr(hidden, "base", None) is xn and
                                      hidden.ctypes.data + hidden.nbytes ==
                                      xn.ctypes.data + xn.nbytes))):
            return self._gpu.argmax(hidden.shape[0])
        return [int(v) for v in np.argmax(self.logits(hidden), axis=1)]

    def _head_is_q6k(self):
        """Return True when the output head is a Q6_K table. Prepare it."""
        if self._head_q6k is not None:
            return self._head_q6k is not False
        key = self.head + ".weight"
        try:
            if self.ct.dtype(key) != "Q6_K":
                self._head_q6k = False
            else:
                self._head_q6k = self.ct.q6k_bytes(key)
        except (KeyError, AttributeError, ValueError):
            self._head_q6k = False
        return self._head_q6k is not False

    def _gpu_head(self):
        """Return (kind, blocks) of a GGUF head that a GPU head kernel reads in
        place: ("q6k", the Q6_K blocks) for gg_q6k_head, or ("q4", the Q4_0
        blocks, one row of bytes for each token) for gg_q4_head (the QAT files
        of unsloth keep the token table in Q4_0). None for another head
        (GPU._kq_head)."""
        if not self._q4:
            return None
        if self._head_is_q6k():
            return "q6k", self._head_q6k
        if self._head_q4 is None:
            self._head_q4 = False
            key = self.head + ".weight"
            try:
                if self.ct.dtype(key) == "Q4_0":
                    blocks, dims, _t = self.ct.raw(self.ct._gguf(key))
                    self._head_q4 = blocks.view(np.uint8).reshape(int(dims[1]), -1)
            except (KeyError, AttributeError, ValueError):
                pass
        return ("q4", self._head_q4) if self._head_q4 is not False else None

    def generate(self, input_ids, max_new_tokens=8, eos_ids=(), cache=None,
                 sampler=None, hook=None):
        """Generate tokens. Return the new ids.

        The first call runs the whole prompt. The later calls run one token.
        """
        ids = list(np.asarray(input_ids, dtype=np.int64).reshape(-1))
        if cache is None:
            cache = E4BCache(self.cfg)
        hidden = self.forward(ids, cache=cache, start_pos=0, hook=hook)
        out = []
        pos = len(ids)
        step = 0
        while step < max_new_tokens:
            logits = self.logits(hidden[-1:])[0]
            if sampler is None:
                nxt = int(np.argmax(logits))
            else:
                nxt = int(sampler(logits, ids + out))
            out.append(nxt)
            step += 1
            if nxt in eos_ids or step >= max_new_tokens:
                break
            hidden = self.forward([nxt], cache=cache, start_pos=pos, hook=hook)
            pos += 1
        return out
