"""Minimal NumPy-only safetensors reader: mmap the file, convert bf16 on the fly."""
from __future__ import annotations

import json
import mmap
import struct

import numpy as np

_NP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U64": np.uint64, "U32": np.uint32, "U16": np.uint16, "U8": np.uint8,
    "BOOL": np.bool_,
}


class SafeTensors:
    """Read a .safetensors file directly from a read-only memory map."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header = json.loads(self._fh.read(n))
        self.header.pop("__metadata__", None)
        self._base = 8 + n
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._mm.close()
        self._fh.close()

    def names(self):
        return list(self.header)

    def shape(self, name):
        return tuple(self.header[name]["shape"])

    def dtype(self, name):
        return self.header[name]["dtype"]

    def _decode(self, name, start_byte, count):
        t = self.header[name]
        dt = t["dtype"]
        if dt == "BF16":
            raw = np.frombuffer(self._mm, dtype=np.uint16, count=count, offset=self._base + start_byte)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return np.frombuffer(self._mm, dtype=_NP[dt], count=count, offset=self._base + start_byte)

    def get(self, name, dtype=np.float32):
        t = self.header[name]
        o0, o1 = t["data_offsets"]
        if t["dtype"] == "BF16":
            arr = self._decode(name, o0, (o1 - o0) // 2)
        else:
            itemsize = np.dtype(_NP[t["dtype"]]).itemsize
            arr = self._decode(name, o0, (o1 - o0) // itemsize)
        arr = arr.reshape(t["shape"])
        if dtype is not None and arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype)
        return arr

    def get_rows(self, name, start, stop, dtype=np.float32):
        t = self.header[name]
        shape = t["shape"]
        o0 = t["data_offsets"][0]
        ncols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        dt = t["dtype"]
        itemsize = 2 if dt == "BF16" else np.dtype(_NP[dt]).itemsize
        count = (stop - start) * ncols
        arr = self._decode(name, o0 + start * ncols * itemsize, count)
        arr = arr.reshape((stop - start,) + tuple(shape[1:]))
        if dtype is not None and arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype)
        return arr

    def get_row(self, name, row, dtype=np.float32):
        return self.get_rows(name, row, row + 1, dtype=dtype)[0]
