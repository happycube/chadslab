"""Read GGUF files with NumPy only.

The GGUF format holds quantized blocks. This module does four tasks:

1. Read the metadata and the tensor directory.
2. Dequantize one tensor on demand to float32.
3. Map the GGUF tensor names to the names of this runtime.
4. Give the Q4_0 tensors in the runtime int4 layout.

The reader maps the file into memory. It reads the small block headers as
needed. It does not copy the whole tensor.

Only the quant types of the Gemma 4 QAT files are implemented: F32, F16, BF16,
Q4_0, and Q6_K. A different type raises an error.
"""
from __future__ import annotations

import mmap
import re
import struct

import numpy as np

# The GGML data types.
F32, F16, Q4_0, Q4_1 = 0, 1, 2, 3
Q5_0, Q5_1, Q8_0, Q8_1 = 6, 7, 8, 9
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K = 10, 11, 12, 13, 14, 15
BF16 = 30

# (values in one block, bytes in one block) for each type.
_BLOCK = {
    F32: (1, 4), F16: (1, 2), BF16: (1, 2),
    Q4_0: (32, 18), Q4_1: (32, 20), Q5_0: (32, 22), Q5_1: (32, 24),
    Q8_0: (32, 34), Q8_1: (32, 36), Q6_K: (256, 210),
}
_TYPE_NAME = {
    F32: "F32", F16: "F16", BF16: "BF16", Q4_0: "Q4_0", Q4_1: "Q4_1",
    Q5_0: "Q5_0", Q5_1: "Q5_1", Q8_0: "Q8_0", Q8_1: "Q8_1", Q6_K: "Q6_K",
}

# The NumPy dtype of one block for the implemented types.
_BLOCK_DT = {
    F32: np.dtype("<f4"),
    F16: np.dtype("<f2"),
    BF16: np.dtype("<u2"),
    Q4_0: np.dtype([("d", "<f2"), ("qs", "u1", (16,))]),
    Q6_K: np.dtype([("ql", "u1", (128,)), ("qh", "u1", (64,)),
                    ("sc", "i1", (16,)), ("d", "<f2")]),
}

# The scalar metadata types of the GGUF format.
_NP_META = {
    0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32,
    5: np.int32, 6: np.float32, 7: np.uint8, 10: np.uint64, 11: np.int64,
    12: np.float64,
}
_SCALAR = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B",
    10: "<Q", 11: "<q", 12: "<d",
}
_STRING = 8
_ARRAY = 9
# Arrays with more elements than this keep a file offset instead of the values.
_KEEP = 4096

_PREFIX = "model.language_model."

# Map the name of one block tensor to the name of this runtime.
_BLOCK_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    "post_ffw_norm_1.weight": "post_feedforward_layernorm_1.weight",
    "post_ffw_norm_2.weight": "post_feedforward_layernorm_2.weight",
    "pre_ffw_norm_2.weight": "pre_feedforward_layernorm_2.weight",
    "layer_output_scale.weight": "layer_scalar",
    "ffn_gate_inp.weight": "router.proj.weight",
    "ffn_gate_inp.scale": "router.scale",
    "ffn_gate_up_exps.weight": "experts.gate_up_proj",
    "ffn_down_exps.weight": "experts.down_proj",
    "ffn_down_exps.scale": "router.per_expert_scale",
}
_GLOBAL_NAMES = {
    "token_embd.weight": "embed_tokens.weight",
    "output_norm.weight": "norm.weight",
    "rope_freqs.weight": "rope_freqs.weight",
}


def _read_string(f):
    """Read one string from the stream."""
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def _read_meta(f, t):
    """Read one metadata value from the stream."""
    if t in _SCALAR:
        return struct.unpack(_SCALAR[t], f.read(struct.calcsize(_SCALAR[t])))[0]
    if t == _STRING:
        return _read_string(f)
    if t == _ARRAY:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == _STRING:
            if n <= _KEEP:
                return [_read_string(f) for _ in range(n)]
            start = f.tell()
            for _ in range(n):
                _read_string(f)
            return {"__array__": n, "elem": et, "offset": start}
        dt = _NP_META.get(et)
        if dt is None:
            raise ValueError("metadata array element type %d" % et)
        itemsize = np.dtype(dt).itemsize
        raw = f.read(itemsize * n)
        return np.frombuffer(raw, dtype=dt).copy()
    raise ValueError("metadata type %d" % t)


def _dequant(raw, t, count):
    """Return float32 values from the blocks in raw.

    raw is a structured array of blocks. count is the number of values.
    """
    if t == F32:
        return np.asarray(raw, dtype=np.float32).reshape(-1)[:count]
    if t == F16:
        return raw.astype(np.float32).reshape(-1)[:count]
    if t == BF16:
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(-1)[:count]
    if t == Q4_0:
        # One byte holds value j in the low nibble and value j+16 in the high
        # nibble. The value is the nibble minus 8. The block has one scale.
        d = raw["d"].astype(np.float32)
        q = raw["qs"]
        lo = (q & 0x0F).astype(np.float32) - 8.0
        hi = (q >> 4).astype(np.float32) - 8.0
        out = np.concatenate([lo, hi], axis=1) * d[:, None]
        return out.reshape(-1)[:count]
    if t == Q6_K:
        # 256 values in one block. ql holds the low 4 bits, qh the top 2 bits,
        # sc one 8-bit scale for each group of 16, and d the block scale.
        # Use int16 for the work. The scale of a group of 16 is a repeat, not a
        # gather. Thus the temporary arrays stay small.
        nb = raw.shape[0]
        ql = raw["ql"].reshape(nb, 2, 64)
        qh = raw["qh"].reshape(nb, 2, 32)
        sc = raw["sc"].astype(np.int16).reshape(nb, 2, 8)
        d = raw["d"].astype(np.float32)
        q1 = ((ql[:, :, 0:32] & 0x0F) | (((qh >> 0) & 3) << 4)).astype(np.int16) - 32
        q2 = ((ql[:, :, 32:64] & 0x0F) | (((qh >> 2) & 3) << 4)).astype(np.int16) - 32
        q3 = ((ql[:, :, 0:32] >> 4) | (((qh >> 4) & 3) << 4)).astype(np.int16) - 32
        q4 = ((ql[:, :, 32:64] >> 4) | (((qh >> 6) & 3) << 4)).astype(np.int16) - 32
        y1 = q1 * np.repeat(sc[:, :, 0:2], 16, axis=2)
        y2 = q2 * np.repeat(sc[:, :, 2:4], 16, axis=2)
        y3 = q3 * np.repeat(sc[:, :, 4:6], 16, axis=2)
        y4 = q4 * np.repeat(sc[:, :, 6:8], 16, axis=2)
        out = np.concatenate([y1, y2, y3, y4], axis=2)
        out = out.reshape(nb, 256).astype(np.float32) * d[:, None]
        return out.reshape(-1)[:count]
    raise ValueError("dequant for type %s is not implemented" % _TYPE_NAME.get(t, t))


def to_bf16(x):
    """Round a float32 array to bfloat16. Return the raw uint16 values."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    # Round to the nearest even value before the shift.
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


class GGUF:
    """Read a GGUF file from a read-only memory map.

    The class gives the same read methods as SafeTensors. Thus the model can
    use a GGUF file in place of a SafeTensors file. The tensor names are
    mapped to the names of this runtime.
    """

    # A GGUF file already holds the quantized weights. Do not build the
    # on-disk weight cache for it.
    use_cache = False
    # The embedding table of the QAT files is Q6_K. Keep the tied output head
    # at that precision. A 4-bit output head changes the first token.
    keep_embedding_bf16 = True

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        magic = self._fh.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file: %s" % path)
        self.version = struct.unpack("<I", self._fh.read(4))[0]
        n_tensors = struct.unpack("<Q", self._fh.read(8))[0]
        n_kv = struct.unpack("<Q", self._fh.read(8))[0]
        self.meta = {}
        for _ in range(n_kv):
            key = _read_string(self._fh)
            t = struct.unpack("<I", self._fh.read(4))[0]
            self.meta[key] = _read_meta(self._fh, t)
        self.tensors = {}
        self._order = []
        for _ in range(n_tensors):
            name = _read_string(self._fh)
            nd = struct.unpack("<I", self._fh.read(4))[0]
            dims = struct.unpack("<%dQ" % nd, self._fh.read(8 * nd))
            t = struct.unpack("<I", self._fh.read(4))[0]
            off = struct.unpack("<Q", self._fh.read(8))[0]
            self.tensors[name] = (dims, t, off)
            self._order.append(name)
        align = int(self.meta.get("general.alignment", 32))
        self._base = (self._fh.tell() + align - 1) // align * align
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            self._mm.madvise(mmap.MADV_HUGEPAGE)
        except (AttributeError, OSError):
            pass
        # Map the runtime names to the GGUF names.
        self._to_gguf = {}
        for gname in self._order:
            self._to_gguf[self._name(gname)] = gname

    # ---- name mapping -----------------------------------------------------

    @staticmethod
    def _name(gname):
        """Return the runtime name of one GGUF tensor name."""
        if gname in _GLOBAL_NAMES:
            return _PREFIX + _GLOBAL_NAMES[gname]
        m = re.match(r"blk\.(\d+)\.(.+)", gname)
        if m:
            idx, suffix = m.group(1), m.group(2)
            target = _BLOCK_NAMES.get(suffix)
            if target is None:
                raise KeyError("unknown block tensor %s" % gname)
            if target == "layer_scalar":
                return _PREFIX + "layers.%s.layer_scalar" % idx
            return _PREFIX + "layers.%s.%s" % (idx, target)
        raise KeyError("unknown tensor %s" % gname)

    def _gguf(self, hf_name):
        try:
            return self._to_gguf[hf_name]
        except KeyError:
            raise KeyError("no GGUF tensor for %s" % hf_name) from None

    # ---- read methods (the SafeTensors interface) -------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self._mm.close()
        except BufferError:
            pass
        self._fh.close()

    def release_pages(self):
        try:
            self._mm.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, OSError):
            pass

    def names(self):
        """Return all tensor names of this runtime."""
        return list(self._to_gguf)

    def shape(self, hf_name):
        """Return the shape of one tensor. The GGUF order is reversed."""
        dims, _t, _o = self.tensors[self._gguf(hf_name)]
        return tuple(reversed(dims))

    def dtype(self, hf_name):
        """Return the GGUF type name of one tensor."""
        _d, t, _o = self.tensors[self._gguf(hf_name)]
        return _TYPE_NAME[t]

    def _blocks(self, hf_name, first, nblk):
        """Return the structured block array for a block range."""
        dims, t, off = self.tensors[self._gguf(hf_name)]
        dt = _BLOCK_DT.get(t)
        if dt is None:
            raise ValueError("type %s is not implemented" % _TYPE_NAME.get(t, t))
        return np.frombuffer(self._mm, dtype=dt, count=nblk,
                             offset=self._base + off + first * dt.itemsize)

    def get(self, hf_name, dtype=np.float32):
        """Return one tensor as a float32 array in the runtime shape."""
        dims, t, _o = self.tensors[self._gguf(hf_name)]
        count = int(np.prod(dims))
        bv, _bb = _BLOCK[t]
        nblk = count // bv
        raw = self._blocks(hf_name, 0, nblk)
        arr = _dequant(raw, t, count).reshape(tuple(reversed(dims)))
        if dtype is not None and arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype)
        return arr

    def get_bf16(self, hf_name):
        """Return the tensor as raw bfloat16 values."""
        return to_bf16(self.get(hf_name))

    def get_rows(self, hf_name, start, stop, dtype=np.float32):
        """Return a row range of a 2-D tensor."""
        dims, t, _o = self.tensors[self._gguf(hf_name)]
        if len(dims) != 2:
            raise ValueError("%s is not 2-D" % hf_name)
        shape = tuple(reversed(dims))
        bv, _bb = _BLOCK[t]
        nblk_row = shape[-1] // bv
        raw = self._blocks(hf_name, start * nblk_row, (stop - start) * nblk_row)
        arr = _dequant(raw, t, (stop - start) * shape[-1]).reshape((stop - start,) + shape[1:])
        if dtype is not None and arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype)
        return arr

    def get_row(self, hf_name, row, dtype=np.float32):
        return self.get_rows(hf_name, row, row + 1, dtype=dtype)[0]

    # ---- the int4 fast path ----------------------------------------------

    def int4_packed(self, hf_name):
        """Return the Q4_0 tensor in the int4 layout of this runtime.

        Return (packed, scales). packed is uint8 with two values in each byte.
        scales is float32 with one value for each group of 32 columns.

        The Q4_0 nibble is the two's complement nibble flipped at bit 3. The
        runtime nibble is the two's complement nibble. Thus the function flips
        bit 3 of each nibble, that is bit 0x88 of each byte. The group scale of
        Q4_0 is then the group scale of the runtime. No other change is needed.
        """
        dims, t, _o = self.tensors[self._gguf(hf_name)]
        if t != Q4_0:
            raise ValueError("%s is %s, not Q4_0" % (hf_name, _TYPE_NAME.get(t, t)))
        shape = tuple(reversed(dims))
        count = int(np.prod(dims))
        nblk = count // 32
        raw = self._blocks(hf_name, 0, nblk)
        packed = (raw["qs"] ^ 0x88).reshape(shape[:-1] + (shape[-1] // 2,))
        scales = raw["d"].astype(np.float32).reshape(shape[:-1] + (shape[-1] // 32,))
        return np.ascontiguousarray(packed), np.ascontiguousarray(scales)

    def int4_row_slice(self, hf_name, start, stop):
        """Return the Q4_0 rows [start, stop) in the int4 layout.

        Use this method for a large tensor. A full 3-D expert tensor is too
        large for the memory.
        """
        dims, t, _o = self.tensors[self._gguf(hf_name)]
        if t != Q4_0 or len(dims) != 3:
            raise ValueError("%s is not a 3-D Q4_0 tensor" % hf_name)
        shape = tuple(reversed(dims))          # (experts, out, in)
        nblk_row = shape[-1] // 32
        first = start * shape[1] * nblk_row
        raw = self._blocks(hf_name, first, (stop - start) * shape[1] * nblk_row)
        packed = (raw["qs"] ^ 0x88).reshape((stop - start, shape[1], shape[-1] // 2))
        scales = raw["d"].astype(np.float32).reshape((stop - start, shape[1], shape[-1] // 32))
        return packed, scales

    # ---- the configuration ------------------------------------------------

    def text_config(self):
        """Return a Hugging Face text_config dict from the GGUF metadata."""
        m = self.meta
        head_kv = np.asarray(m["gemma4.attention.head_count_kv"])
        pattern = np.asarray(m["gemma4.attention.sliding_window_pattern"])
        layer_types = ["sliding_attention" if bool(p) else "full_attention" for p in pattern]
        sliding = [i for i, t in enumerate(layer_types) if t == "sliding_attention"]
        return {
            "hidden_size": int(m["gemma4.embedding_length"]),
            "intermediate_size": int(m["gemma4.feed_forward_length"]),
            "num_hidden_layers": int(m["gemma4.block_count"]),
            "num_attention_heads": int(m["gemma4.attention.head_count"]),
            "num_key_value_heads": int(head_kv[sliding[0]]) if sliding else int(head_kv[0]),
            "head_dim": int(m["gemma4.attention.key_length_swa"]),
            "global_head_dim": int(m["gemma4.attention.key_length"]),
            "num_global_key_value_heads": int(head_kv[5]),
            "rms_norm_eps": float(m["gemma4.attention.layer_norm_rms_epsilon"]),
            "vocab_size": 262144,
            "max_position_embeddings": int(m["gemma4.context_length"]),
            "sliding_window": int(m["gemma4.attention.sliding_window"]),
            "final_logit_softcapping": float(m["gemma4.final_logit_softcapping"]),
            "num_experts": int(m["gemma4.expert_count"]),
            "top_k_experts": int(m["gemma4.expert_used_count"]),
            "moe_intermediate_size": int(m["gemma4.expert_feed_forward_length"]),
            "layer_types": layer_types,
            "rope_parameters": {
                "sliding_attention": {"rope_theta": float(m["gemma4.rope.freq_base_swa"])},
                "full_attention": {
                    "rope_theta": float(m["gemma4.rope.freq_base"]),
                    "partial_rotary_factor": 0.25,
                },
            },
        }

    # ---- the tokenizer ----------------------------------------------------

    def meta_strings(self, key):
        """Return one metadata string array. Read the data from the file.

        The reader keeps a large string array on the disk. It stores the file
        offset in the metadata. This method reads the array.
        """
        item = self.meta[key]
        if not isinstance(item, dict) or "__array__" not in item:
            return list(item)
        n = item["__array__"]
        pos = item["offset"]
        mm = self._mm
        out = []
        for _ in range(n):
            ln = struct.unpack_from("<Q", mm, pos)[0]
            pos += 8
            out.append(mm[pos:pos + ln].decode("utf-8", "replace"))
            pos += ln
        return out

    def tokenizer_json(self):
        """Return the tokenizer data in the form of a tokenizer.json dict.

        The GGUF token type gives the kind of each token. Type 3 is a control
        token and type 4 is a user token. These are the added tokens of the
        Hugging Face file. Type 6 is a byte token, so the byte fallback is on.
        """
        m = self.meta
        tokens = self.meta_strings("tokenizer.ggml.tokens")
        merges = self.meta_strings("tokenizer.ggml.merges")
        types = np.asarray(m["tokenizer.ggml.token_type"])
        vocab = {t: i for i, t in enumerate(tokens)}
        added = []
        for i, t in enumerate(tokens):
            if i < types.size and int(types[i]) in (3, 4):
                added.append({"content": t, "id": i, "special": True})
        unk_id = int(m.get("tokenizer.ggml.unknown_token_id", 0))
        return {
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": merges,
                "byte_fallback": bool((types == 6).any()),
                "unk_token": tokens[unk_id],
            },
            "added_tokens": added,
        }
