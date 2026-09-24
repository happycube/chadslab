"""Read a compressed-tensors checkpoint with NumPy only.

The E4B mobile-ct checkpoint is a SafeTensors file in the compressed-tensors
"pack-quantized" format. This module reads that format from a memory map.

The format has four kinds of weight:

1. A packed sub-byte weight. The file keeps `weight_packed`, `weight_scale`,
   and `weight_shape`. The packed data is int32. Each int32 holds several
   values. The scale is a bfloat16 tensor. The shape tensor gives the logical
   shape before the packing.
2. An int8 weight. The file keeps `weight` as int8 and `weight_scale` as
   bfloat16. This is the 8-bit group of the checkpoint.
3. A plain float weight. The quantization ignores some tensors, for example
   `per_layer_model_projection`. The file keeps these as bfloat16.
4. A norm weight. The file keeps these as bfloat16.

The packing is dense. Element i of a row starts at bit i * num_bits, counted
from the start of the row. When num_bits divides 32, each element lives inside
one int32 word: element i is at bit (i mod 32/num_bits) * num_bits of word
i // (32/num_bits). The functions below use that fact. The window and the
compressed-tensors library agree on this layout; `check_ct.py` proves it.

The reader does not copy a whole tensor unless the caller asks for a whole
tensor. `row()` reads one row. That is the operation that the embedding
lookups need, and it is the operation that makes the memory map useful: a
token reads 2.7 KiB of the per-layer table, not 704 MiB.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .st import SafeTensors


def bf16_to_f32(raw):
    """Turn bfloat16 bits into float32 values.

    Move the 16 data bits to the top of a 32-bit word. The low 16 bits are zero.
    """
    raw = np.asarray(raw)
    if raw.dtype == np.float32:
        return raw
    return (raw.astype(np.uint32) << 16).view(np.float32)


def unpack_int32(words, num_bits, ncols):
    """Unpack densely packed values from int32 words.

    words   An int32 or uint32 array. The last axis holds the words of one row.
    num_bits The number of bits for each value. It must divide 32.
    ncols   The number of values in the row. The last word can hold padding.

    Return an int8 array with the shape of the input, but the last axis is
    ncols. The values are signed: the packing stored them with an offset of
    2 ** (num_bits - 1), and this function removes that offset.
    """
    if 32 % num_bits:
        raise ValueError("num_bits must divide 32, got %r" % (num_bits,))
    words = np.ascontiguousarray(words)
    if words.dtype != np.uint32:
        words = words.view(np.uint32)
    per_word = 32 // num_bits
    mask = np.uint32((1 << num_bits) - 1)
    full = words.shape[-1] * per_word
    # Write one value position inside the word at a time. Each step reads the
    # whole packed row and writes a strided slice of the output. This form
    # avoids a second full-size integer array, which is what made the first
    # version slow: it built a uint32 array, an int16 array, and then an int8
    # array, so it wrote four times the size of the result.
    out = np.empty(words.shape[:-1] + (full,), dtype=np.uint8)
    flat = out.reshape(-1, per_word)
    src = words.reshape(-1, 1)
    for j in range(per_word):
        np.bitwise_and(np.right_shift(src, np.uint32(j * num_bits)), mask,
                       out=flat[:, j:j + 1], casting="unsafe")
    vals = out[..., :ncols]
    if ncols != full:
        vals = np.ascontiguousarray(vals)
    # The packing stored the values with an offset. Remove the offset in the
    # unsigned domain. The low 8 bits are then the signed value, so the view
    # gives the answer without a cast.
    return (vals - np.uint8(1 << (num_bits - 1))).view(np.int8)


# The module names the quantization ignores. The file keeps these in bfloat16.
_UNQUANTIZED_SUFFIX = (".weight",)


class CompressedTensors:
    """Read one compressed-tensors SafeTensors file through a memory map.

    Use `dequant()` for a whole matrix. Use `row()` for one row of a matrix.
    Use `has()` to ask whether a tensor exists.
    """

    def __init__(self, path, st=None):
        self.path = path
        self.st = st if st is not None else SafeTensors(path)
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")
        self.quant_config = None
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                self.quant_config = json.load(fh).get("quantization_config")
        # Remember the bit width of each packed tensor. The width follows from
        # the packed shape and the logical shape, so the configuration file is
        # not needed. Keep the result so the lookup is cheap.
        self._bits = {}

    def close(self):
        self.st.close()

    def names(self):
        return self.st.names()

    def has(self, name):
        """Return True when the file holds this tensor."""
        return name in self.st.header

    # ---- the shape and the bit width ---------------------------------------
    def _logical_shape(self, name):
        """Return the logical shape of a weight. Read weight_shape when it exists.

        A packed weight carries its shape in a separate tensor. An int8 weight
        and a float weight carry it in the tensor itself.
        """
        shape_name = name + ".weight_shape"
        if shape_name in self.st.header:
            return tuple(int(v) for v in self.st.get(shape_name, dtype=np.int64))
        if name in self.st.header:
            return self.st.shape(name)
        return self.st.shape(name + ".weight")

    def num_bits(self, name):
        """Return the bit width of a weight. Return None for a float weight."""
        key = name + ".weight_packed"
        if key in self.st.header:
            if name not in self._bits:
                shape = self._logical_shape(name)
                words = self.st.shape(key)[-1]
                ncols = shape[-1]
                n = words * 32
                if n % ncols:
                    raise ValueError("%s: packed size %d is not a multiple of %d"
                                     % (name, n, ncols))
                self._bits[name] = n // ncols
            return self._bits[name]
        key = name + ".weight"
        if key in self.st.header and self.st.dtype(key) == "I8":
            return 8
        return None

    def strategy(self, name):
        """Return ('channel', 1) or ('group', group_size) for a quantized weight.

        A scale of shape [rows, 1] is one scale for each row. A scale of shape
        [rows, groups] is one scale for each group of columns.
        """
        scale = self._logical_shape(name + ".weight_scale")
        ncols = self._logical_shape(name)[-1]
        groups = scale[-1]
        if groups == 1:
            return ("channel", ncols)
        if ncols % groups:
            raise ValueError("%s: %d columns do not divide into %d groups"
                             % (name, ncols, groups))
        return ("group", ncols // groups)

    def _stored_key(self, name):
        """Return the header key of the stored form of a weight."""
        for suffix in (".weight_packed", ".weight", ""):
            key = name + suffix
            if key in self.st.header:
                return key
        raise KeyError(name)

    def packed_bytes(self, name):
        """Return the number of bytes the stored form of this weight uses."""
        t = self.st.header[self._stored_key(name)]
        return t["data_offsets"][1] - t["data_offsets"][0]

    def raw_view(self, name):
        """Return a uint8 view of the stored bytes of a weight.

        The view points into the memory map. This is the entry point for a
        kernel that reads the packed data directly, and for a measurement of
        the read cost of the file without the decode.
        """
        t = self.st.header[self._stored_key(name)]
        o0, o1 = t["data_offsets"]
        return np.frombuffer(self.st._mm, dtype=np.uint8, count=o1 - o0,
                             offset=self.st._base + o0)

    def dtype(self, name):
        """Return the dtype string of the stored form of a weight."""
        return self.st.header[self._stored_key(name)]["dtype"]

    # ---- the packed form, for a kernel -------------------------------------
    def packed_words(self, name):
        """Return the int32 words of a packed weight as a view into the map.

        Do not unpack. The C kernel reads these words where the file put them,
        so the model can keep the weights packed. Return None when the weight
        is not in the packed form, for example when it is int8 or bfloat16.
        """
        key = name + ".weight_packed"
        if key not in self.st.header:
            return None
        return self.st.get(key, dtype=None)

    def channel_scale(self, name):
        """Return the scale of a channel-strategy weight as float32, one per row.

        Return None when the weight is not channel strategy. A group strategy
        needs one scale for each group of columns, and the kernel does not
        take that form. Return None as well for a weight that the quantization
        did not touch, which has no scale at all.
        """
        if name + ".weight_scale" not in self.st.header:
            return None
        strategy, _group = self.strategy(name)
        if strategy != "channel":
            return None
        return self._scale(name).reshape(-1).astype(np.float32)

    # ---- the decode --------------------------------------------------------
    def _scale(self, name, dtype=np.float32):
        t = self.st.header[name + ".weight_scale"]
        if t["dtype"] == "BF16":
            return bf16_to_f32(self.st.get_bf16(name + ".weight_scale")).astype(dtype)
        return self.st.get(name + ".weight_scale", dtype=dtype)

    def _apply_scale(self, q, scale, shape):
        """Multiply the integer values by the scale. q is float32."""
        if scale.shape[-1] == 1:
            return q * scale.reshape(scale.shape[0], 1)
        groups = scale.shape[-1]
        group_size = shape[-1] // groups
        q = q.reshape(*shape[:-1], groups, group_size)
        q = q * scale.reshape(scale.shape[0], groups, 1)
        return q.reshape(shape)

    def dequant(self, name, dtype=np.float32):
        """Return a whole weight as a float array.

        This function reads the full weight and builds a float copy. Use it to
        build a resident set of weights. Use `row()` to read one row.
        """
        key = name + ".weight_packed"
        if key in self.st.header:
            shape = self._logical_shape(name)
            bits = self.num_bits(name)
            packed = self.st.get(key, dtype=None)
            q = unpack_int32(packed, bits, shape[-1]).astype(np.float32)
            q = q.reshape(shape)
            return self._apply_scale(q, self._scale(name), shape).astype(dtype)
        key = name + ".weight"
        if key in self.st.header:
            if self.st.dtype(key) == "I8":
                shape = self._logical_shape(name)
                q = self.st.get(key, dtype=np.int8).astype(np.float32)
                return self._apply_scale(q, self._scale(name), shape).astype(dtype)
            return self.st.get(key, dtype=dtype)
        return self.st.get(name, dtype=dtype)

    def row(self, name, i, dtype=np.float32):
        """Return one row of a weight as a float array.

        Read only the bytes of that row. This is the operation that an
        embedding lookup needs. It keeps the memory map useful: the caller
        touches 2.7 KiB of the per-layer table instead of 704 MiB.
        """
        shape = self._logical_shape(name)
        ncols = shape[-1]
        key = name + ".weight_packed"
        if key in self.st.header:
            bits = self.num_bits(name)
            packed = self.st.get_row(key, i, dtype=None)
            q = unpack_int32(packed, bits, ncols).astype(np.float32)
            scale = self._scale_row(name, i)
            if scale.shape[0] == 1:
                return (q * scale[0]).astype(dtype)
            group_size = ncols // scale.shape[0]
            q = q.reshape(scale.shape[0], group_size) * scale[:, None]
            return q.reshape(ncols).astype(dtype)
        key = name + ".weight"
        if key in self.st.header and self.st.dtype(key) == "I8":
            q = self.st.get_row(key, i, dtype=np.int8).astype(np.float32)
            scale = self._scale_row(name, i)
            return (q * scale[0]).astype(dtype)
        if key in self.st.header:
            return self.st.get_row(key, i, dtype=dtype)
        return self.st.get_row(name, i, dtype=dtype)

    def _scale_row(self, name, i):
        """Return the scale row for one row of a weight, as float32."""
        key = name + ".weight_scale"
        if self.st.dtype(key) == "BF16":
            raw = self.st.get_row(key, i, dtype=None)
            return bf16_to_f32(raw)
        return self.st.get_row(key, i, dtype=np.float32)

    def rows(self, name, start, stop, dtype=np.float32):
        """Return a row range of a weight as a float array.

        This function reads the bytes of the range only. Use it for a small
        batch of tokens.
        """
        shape = self._logical_shape(name)
        ncols = shape[-1]
        key = name + ".weight_packed"
        if key in self.st.header:
            bits = self.num_bits(name)
            packed = self.st.get_rows(key, start, stop, dtype=None)
            q = unpack_int32(packed, bits, ncols).astype(np.float32)
            scale = self._scale_rows(name, start, stop)
            if scale.shape[-1] == 1:
                return (q * scale).astype(dtype)
            group_size = ncols // scale.shape[-1]
            q = q.reshape(stop - start, scale.shape[-1], group_size)
            q = q * scale[:, :, None]
            return q.reshape(stop - start, ncols).astype(dtype)
        key = name + ".weight"
        if key in self.st.header and self.st.dtype(key) == "I8":
            q = self.st.get_rows(key, start, stop, dtype=np.int8).astype(np.float32)
            scale = self._scale_rows(name, start, stop)
            return (q * scale).astype(dtype)
        if key in self.st.header:
            return self.st.get_rows(key, start, stop, dtype=dtype)
        return self.st.get_rows(name, start, stop, dtype=dtype)

    def _scale_rows(self, name, start, stop):
        key = name + ".weight_scale"
        if self.st.dtype(key) == "BF16":
            raw = self.st.get_rows(key, start, stop, dtype=None)
            return bf16_to_f32(raw)
        return self.st.get_rows(key, start, stop, dtype=np.float32)

    # ---- helpers the model uses -------------------------------------------
    def plain(self, name, dtype=np.float32):
        """Return a tensor that the quantization did not touch.

        A norm weight and a bias take this path.
        """
        if self.st.dtype(name) == "BF16":
            return bf16_to_f32(self.st.get_bf16(name)).astype(dtype)
        return self.st.get(name, dtype=dtype)

    def scalar(self, name):
        """Return a one-element tensor as a Python float."""
        return float(self.plain(name).reshape(-1)[0])
