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

from .ops import to_bf16

# The GGML data types.
F32, F16, Q4_0, Q4_1 = 0, 1, 2, 3
Q5_0, Q5_1, Q8_0, Q8_1 = 6, 7, 8, 9
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K = 10, 11, 12, 13, 14, 15
IQ4_NL = 20
BF16 = 30

# The 16 values of the 4-bit codes of IQ4_NL (ggml kvalues_iq4nl).
_IQ4_NL_VALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89,
                           113], dtype=np.float32)

# (values in one block, bytes in one block) for each type.
_BLOCK = {
    F32: (1, 4), F16: (1, 2), BF16: (1, 2),
    Q4_0: (32, 18), Q4_1: (32, 20), Q5_0: (32, 22), Q5_1: (32, 24),
    Q8_0: (32, 34), Q8_1: (32, 36), Q6_K: (256, 210), Q4_K: (256, 144), Q5_K: (256, 176),
    IQ4_NL: (32, 18),
}
_TYPE_NAME = {
    F32: "F32", F16: "F16", BF16: "BF16", Q4_0: "Q4_0", Q4_1: "Q4_1",
    Q5_0: "Q5_0", Q5_1: "Q5_1", Q8_0: "Q8_0", Q8_1: "Q8_1", Q6_K: "Q6_K",
    Q4_K: "Q4_K", Q5_K: "Q5_K", IQ4_NL: "IQ4_NL",
}

# The NumPy dtype of one block for the implemented types.
_BLOCK_DT = {
    F32: np.dtype("<f4"),
    F16: np.dtype("<f2"),
    BF16: np.dtype("<u2"),
    Q4_0: np.dtype([("d", "<f2"), ("qs", "u1", (16,))]),
    Q6_K: np.dtype([("ql", "u1", (128,)), ("qh", "u1", (64,)),
                    ("sc", "i1", (16,)), ("d", "<f2")]),
    Q8_0: np.dtype([("d", "<f2"), ("qs", "i1", (32,))]),
    Q5_1: np.dtype([("d", "<f2"), ("m", "<f2"), ("qh", "<u4"), ("qs", "u1", (16,))]),
    IQ4_NL: np.dtype([("d", "<f2"), ("qs", "u1", (16,))]),
    Q4_K: np.dtype([("d", "<f2"), ("dmin", "<f2"), ("sc", "u1", (12,)), ("qs", "u1", (128,))]),
    Q5_K: np.dtype([("d", "<f2"), ("dmin", "<f2"), ("sc", "u1", (12,)), ("qh", "u1", (32,)),
                    ("qs", "u1", (128,))]),
}


def _k_scales(sc):
    """The 8 scales and 8 mins (6 bits each) of the 12 bytes of a Q4_K or
    Q5_K block, as get_scale_min_k4 of ggml does it. sc is (blocks, 12)."""
    sc = sc.astype(np.int32)
    s = np.empty((sc.shape[0], 8), np.int32)
    m = np.empty((sc.shape[0], 8), np.int32)
    s[:, :4] = sc[:, 0:4] & 63
    m[:, :4] = sc[:, 4:8] & 63
    s[:, 4:] = (sc[:, 8:12] & 0xF) | ((sc[:, 0:4] >> 6) << 4)
    m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    return s, m

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
    # The per-layer embeddings of the E4B model.
    "inp_gate.weight": "per_layer_input_gate.weight",
    "proj.weight": "per_layer_projection.weight",
    "post_norm.weight": "post_per_layer_input_norm.weight",
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
    "per_layer_model_proj.weight": "per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": "per_layer_projection_norm.weight",
    "per_layer_token_embd.weight": "embed_tokens_per_layer.weight",
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
    if t == Q5_1:
        # Value j (j < 16): the low 4 bits of byte j and bit j of qh; value
        # j + 16: the high 4 bits of byte j and bit j + 16 of qh. d q + m.
        q = raw["qs"]
        qh = raw["qh"].astype(np.uint32)[:, None]
        bits = np.arange(16, dtype=np.uint32)[None, :]
        lo = (q & 0x0F).astype(np.uint32) | (((qh >> bits) & 1) << 4)
        hi = (q >> 4).astype(np.uint32) | (((qh >> (bits + 16)) & 1) << 4)
        v = np.concatenate([lo, hi], axis=1).astype(np.float32)
        return (v * raw["d"].astype(np.float32)[:, None] +
                raw["m"].astype(np.float32)[:, None]).reshape(-1)[:count]
    if t == IQ4_NL:
        # 4-bit codes into a table of 16 values: the low 4 bits of byte j are
        # value j, the high 4 bits value j + 16.
        q = raw["qs"]
        v = np.concatenate([_IQ4_NL_VALUES[q & 0x0F], _IQ4_NL_VALUES[q >> 4]], axis=1)
        return (v * raw["d"].astype(np.float32)[:, None]).reshape(-1)[:count]
    if t == Q8_0:
        return (raw["qs"].astype(np.float32) * raw["d"].astype(np.float32)[:, None]).reshape(-1)[:count]
    if t in (Q4_K, Q5_K):
        # 256 values in 4 parts of 64: the low 4 bits of 32 bytes are the
        # first 32 values of a part, the high 4 bits the next 32. Q5_K adds
        # a fifth bit from qh: bit 2j for the low half of part j, 2j + 1 for
        # the high half. Value = d * sc * q - dmin * m for each 32 values.
        nb = raw.shape[0]
        qs = raw["qs"].reshape(nb, 4, 32)
        lo = (qs & 0xF).astype(np.int32)
        hi = (qs >> 4).astype(np.int32)
        if t == Q5_K:
            qh = raw["qh"].astype(np.int32)                    # (nb, 32)
            for j in range(4):
                lo[:, j] |= ((qh >> (2 * j)) & 1) << 4
                hi[:, j] |= ((qh >> (2 * j + 1)) & 1) << 4
        q = np.stack([lo, hi], axis=2).reshape(nb, 8, 32)      # sub-block 2j, 2j + 1
        sc, mn = _k_scales(raw["sc"])
        d = raw["d"].astype(np.float32)[:, None, None]
        dmin = raw["dmin"].astype(np.float32)[:, None, None]
        out = d * sc[:, :, None] * q - dmin * mn[:, :, None]
        return out.astype(np.float32).reshape(-1)[:count]
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
        # Map the runtime names to the GGUF names. A model that this module
        # does not map (Qwen3.5) reads its tensors by GGUF name (raw).
        self._to_gguf = {}
        for gname in self._order:
            try:
                self._name(gname)
            except KeyError:
                continue
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

    def raw(self, gname):
        """The blocks of a tensor by its GGUF name, as a structured array
        (a view of the map), with its dims (ggml order) and type."""
        dims, t, off = self.tensors[gname]
        dt = _BLOCK_DT.get(t)
        if dt is None:
            raise ValueError("type %s is not implemented" % _TYPE_NAME.get(t, t))
        bv, _bb = _BLOCK[t]
        n = int(np.prod(dims)) // bv
        return np.frombuffer(self._mm, dtype=dt, count=n, offset=self._base + off), dims, t

    def dequant(self, gname, rows=None):
        """float32 values of a tensor by GGUF name, in the shape of NumPy
        (dims reversed). rows selects rows of a 2-D tensor (or the first
        index of a 3-D one)."""
        raw, dims, t = self.raw(gname)
        shape = tuple(reversed(dims))
        if rows is None:
            return _dequant(raw, t, int(np.prod(dims))).reshape(shape)
        bv, _bb = _BLOCK[t]
        per = int(np.prod(shape[1:])) // bv
        rows = np.asarray(rows).reshape(-1)
        blocks = raw.reshape(-1, per)[rows].reshape(-1)
        return _dequant(blocks, t, len(rows) * per * bv).reshape((len(rows),) + shape[1:])

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
        """Do nothing.

        The int4 data of the model is a view into the file map. The pages must
        stay in the map. The system can drop a clean page at any time and read
        it again when the code needs it.
        """

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

    def tensor_bytes(self, hf_name):
        """Return the byte count that one tensor takes in the file.

        A quantized tensor holds whole blocks. The count uses the block table,
        so the method reads no tensor data.
        """
        dims, t, _o = self.tensors[self._gguf(hf_name)]
        count = 1
        for d in dims:
            count *= int(d)
        values, block = _BLOCK[t]
        return count // values * block

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
        """Return the tensor as raw bfloat16 values.

        For a Q6_K tensor, dequantize a block range and convert it to bfloat16
        at once. Then the float32 data stays in the cache.
        """
        dims, t, _o = self.tensors[self._gguf(hf_name)]
        if t != Q6_K:
            return to_bf16(self.get(hf_name))
        nb = int(np.prod(dims)) // 256
        out = np.empty(nb * 256, dtype=np.uint16)
        step = 1 << 12
        for b0 in range(0, nb, step):
            n = min(step, nb - b0)
            f = _dequant(self._blocks(hf_name, b0, n), Q6_K, n * 256)
            out[b0 * 256:(b0 + n) * 256] = to_bf16(f)
        return out.reshape(tuple(reversed(dims)))

    def q6k_blocks(self, hf_name):
        """Return the Q6_K blocks of a 2-D tensor as a view.

        The result has shape (rows, blocks in one row). The data points into
        the file map. Do not change it.
        """
        dims, t, _o = self.tensors[self._gguf(hf_name)]
        if t != Q6_K:
            raise ValueError("%s is %s, not Q6_K" % (hf_name, _TYPE_NAME.get(t, t)))
        if len(dims) != 2:
            raise ValueError("%s is not a 2-D tensor" % hf_name)
        cols = int(dims[0])
        rows = int(dims[1])
        if cols % 256:
            raise ValueError("%s has %d columns, not a multiple of 256" % (hf_name, cols))
        bpr = cols // 256
        return self._blocks(hf_name, 0, rows * bpr).reshape(rows, bpr)

    def q6k_bytes(self, hf_name):
        """Return the Q6_K blocks of a 2-D tensor as bytes.

        The result has shape (rows, blocks in one row * 210). Give it to the C
        kernel. The data points into the file map. Do not change it.
        """
        blocks = self.q6k_blocks(hf_name)
        return blocks.view(np.uint8).reshape(blocks.shape[0], blocks.shape[1] * 210)

    def q6k_dequant(self, blocks, cols):
        """Return float32 values from a Q6_K block slice.

        blocks has one row for each output row. cols is the value count in one
        row.
        """
        flat = np.asarray(blocks).reshape(-1)
        return _dequant(flat, Q6_K, flat.size * 256).reshape(-1, cols)

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

    def take_rows(self, hf_name, rows, dtype=np.float32):
        """Return the rows of a 2-D tensor in the order of the list rows.

        The function gathers the blocks of every row, then dequantizes them in
        one operation. An embedding lookup of a prompt then does not do one
        dequant for each token.
        """
        dims, t, _o = self.tensors[self._gguf(hf_name)]
        if len(dims) != 2:
            raise ValueError("%s is not 2-D" % hf_name)
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = int(dims[0])
        bv, _bb = _BLOCK[t]
        nblk_row = cols // bv
        table = self._blocks(hf_name, 0, int(dims[1]) * nblk_row)
        if t == Q6_K and dtype == np.float32:
            from . import cops
            if cops.available():
                return cops.q6k_rows(table.view(np.uint8), rows, cols)
        raw = table.reshape(-1, nblk_row)[rows].reshape(-1)
        arr = _dequant(raw, t, rows.size * cols).reshape(rows.size, cols)
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

        The runtime uses the block layout of Q4_0. One block holds 32 values in
        18 bytes: a float16 scale and 16 nibble bytes. The value of a nibble is
        the nibble minus 8. The layout is the same. Thus the function copies no
        data. packed is a view into the file map.
        """
        dims, t, _o = self.tensors[self._gguf(hf_name)]
        if t != Q4_0:
            raise ValueError("%s is %s, not Q4_0" % (hf_name, _TYPE_NAME.get(t, t)))
        shape = tuple(reversed(dims))
        count = int(np.prod(dims))
        nblk = count // 32
        raw = self._blocks(hf_name, 0, nblk)
        groups = shape[-1] // 32
        packed = raw.view(np.uint8).reshape(shape[:-1] + (groups, 18))
        scales = raw["d"].astype(np.float32).reshape(shape[:-1] + (groups,))
        return packed, np.ascontiguousarray(scales)

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
        packed = raw.view(np.uint8).reshape((stop - start, shape[1], nblk_row, 18))
        scales = raw["d"].astype(np.float32).reshape((stop - start, shape[1], nblk_row))
        return packed, np.ascontiguousarray(scales)

    # ---- the configuration ------------------------------------------------

    def text_config(self):
        """Return a Hugging Face text_config dict from the GGUF metadata."""
        m = self.meta
        head_kv = np.asarray(m["gemma4.attention.head_count_kv"])
        if head_kv.ndim == 0:
            # The E4B file gives one value for every layer.
            head_kv = np.repeat(head_kv, int(m["gemma4.block_count"]))
        pattern = np.asarray(m["gemma4.attention.sliding_window_pattern"])
        layer_types = ["sliding_attention" if bool(p) else "full_attention" for p in pattern]
        sliding = [i for i, t in enumerate(layer_types) if t == "sliding_attention"]
        full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        # The 12B and the 26B reuse the key as the value in a global layer, so
        # those layers have no value projection. The E4B has one there.
        has_v = any(("blk.%d.attn_v.weight" % i) in self.tensors for i in full)
        cfg = {
            "hidden_size": int(m["gemma4.embedding_length"]),
            "intermediate_size": int(m["gemma4.feed_forward_length"]),
            "num_hidden_layers": int(m["gemma4.block_count"]),
            "num_attention_heads": int(m["gemma4.attention.head_count"]),
            "num_key_value_heads": int(head_kv[sliding[0]]) if sliding else int(head_kv[0]),
            "head_dim": int(m["gemma4.attention.key_length_swa"]),
            "global_head_dim": int(m["gemma4.attention.key_length"]),
            "num_global_key_value_heads": int(head_kv[full[0]]) if full else int(head_kv[0]),
            "rms_norm_eps": float(m["gemma4.attention.layer_norm_rms_epsilon"]),
            "vocab_size": 262144,
            "max_position_embeddings": int(m["gemma4.context_length"]),
            "sliding_window": int(m["gemma4.attention.sliding_window"]),
            "final_logit_softcapping": float(m["gemma4.final_logit_softcapping"]),
            "attention_k_eq_v": not has_v,
            "num_kv_shared_layers": int(m.get("gemma4.attention.shared_kv_layers", 0) or 0),
            "hidden_size_per_layer_input": int(
                m.get("gemma4.embedding_length_per_layer_input", 0) or 0),
            "vocab_size_per_layer_input": 262144,
            "layer_types": layer_types,
            "rope_parameters": {
                "sliding_attention": {"rope_theta": float(m["gemma4.rope.freq_base_swa"])},
                "full_attention": {
                    "rope_theta": float(m["gemma4.rope.freq_base"]),
                    "partial_rotary_factor": 0.25,
                },
            },
        }
        if "gemma4.expert_count" in m:
            cfg["num_experts"] = int(m["gemma4.expert_count"])
            cfg["top_k_experts"] = int(m["gemma4.expert_used_count"])
            cfg["moe_intermediate_size"] = int(m["gemma4.expert_feed_forward_length"])
        return cfg

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
            # The ids that the GGUF metadata declares. The end token of the
            # metadata is the end of a turn, not the <eos> token.
            "bos_id": int(m.get("tokenizer.ggml.bos_token_id", 0)),
            "eos_id": int(m.get("tokenizer.ggml.eos_token_id", 0)),
            "pad_id": int(m.get("tokenizer.ggml.padding_token_id", 0)),
            # The end of a turn ends a chat answer. The metadata declares the
            # <eos> token, so add the end-of-turn tokens as well.
            "stop_ids": [int(m.get("tokenizer.ggml.eos_token_id", 0))] + [
                i for i, s in enumerate(tokens) if s in ("<turn|>", "<end_of_turn>")
            ],
        }


class GGUFSplit(GGUF):
    """A model in split GGUF files (name-00001-of-0000N.gguf, as llama.cpp
    gguf-split writes them). The first file has the metadata; each file has
    its own tensors. raw() and dequant() read a tensor from its file.

        g = open_gguf("model-00001-of-00004.gguf")
    """

    def __init__(self, path):
        m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
        if m is None:
            raise ValueError("not the name of a split GGUF file: %s" % path)
        prefix, count = m.group(1), int(m.group(3))
        self.path = path
        self.parts = [GGUF("%s-%05d-of-%05d.gguf" % (prefix, i, count))
                      for i in range(1, count + 1)]
        self.meta = self.parts[0].meta
        self.version = self.parts[0].version
        self.tensors, self._where, self._order = {}, {}, []
        for p in self.parts:
            for name in p._order:
                self.tensors[name] = p.tensors[name]
                self._where[name] = p
                self._order.append(name)
        n = int(self.meta.get("split.tensors.count", len(self.tensors)))
        if n != len(self.tensors):
            raise ValueError("the split files have %d tensors, not %d" % (len(self.tensors), n))
        self._to_gguf = {}

    def raw(self, gname):
        return self._where[gname].raw(gname)

    def attach(self, path):
        """Add the tensors of another file (the MTP layer of a model in its own
        file). The names of the file must be new."""
        p = GGUF(path)
        for name in p._order:
            if name in self.tensors:
                raise ValueError("%s has the tensor %s of the model" % (path, name))
            self.tensors[name] = p.tensors[name]
            self._where[name] = p
            self._order.append(name)
        self.parts.append(p)
        return p

    def close(self):
        for p in self.parts:
            p.close()


def open_gguf(path):
    """A GGUF file, or the first file of a split model (GGUFSplit)."""
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", path):
        return GGUFSplit(path)
    return GGUF(path)
