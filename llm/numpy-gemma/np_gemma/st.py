"""Read SafeTensors files with NumPy only.

This module does three tasks:
1. Read the file header.
2. Map the file into memory.
3. Convert bfloat16 data to float32 data on demand.

The reader does not copy the weights. It returns views into the memory map.
"""
from __future__ import annotations

import json
import mmap
import struct

import numpy as np

# Map the dtype names in the header to NumPy types.
_NP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U64": np.uint64, "U32": np.uint32, "U16": np.uint16, "U8": np.uint8,
    "BOOL": np.bool_,
}


class SafeTensors:
    """Read a .safetensors file from a read-only memory map.

    A SafeTensors file has two parts:
    1. An 8-byte integer. The integer gives the header length.
    2. A JSON header. Then the raw tensor data.

    The header gives the name, the shape, the dtype, and the byte offsets of
    each tensor.
    """

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        # Read the header length. Then read the header.
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header = json.loads(self._fh.read(n))
        self.header.pop("__metadata__", None)
        self._base = 8 + n
        # Map the file. A map is faster than a read. A map also uses less memory.
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Release the memory map and the file handle.

        A bf16-resident model keeps NumPy views into the map. In this case, the
        unmap operation can fail with BufferError. Ignore this error. The
        operating system releases the map at process exit.
        """
        try:
            self._mm.close()
        except BufferError:
            pass
        self._fh.close()

    def names(self):
        """Return all tensor names."""
        return list(self.header)

    def shape(self, name):
        """Return the shape of one tensor."""
        return tuple(self.header[name]["shape"])

    def dtype(self, name):
        """Return the dtype string of one tensor. For example: BF16."""
        return self.header[name]["dtype"]

    def _decode(self, name, start_byte, count):
        """Make a NumPy array from one byte range of the memory map.

        Convert bfloat16 data to float32 data. Move the 16 data bits to the top
        of a 32-bit word.
        """
        t = self.header[name]
        dt = t["dtype"]
        if dt == "BF16":
            raw = np.frombuffer(self._mm, dtype=np.uint16, count=count, offset=self._base + start_byte)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return np.frombuffer(self._mm, dtype=_NP[dt], count=count, offset=self._base + start_byte)

    def get(self, name, dtype=np.float32):
        """Return one tensor as a NumPy array.

        Read the full tensor. Convert bfloat16 data to float32 data.
        """
        t = self.header[name]
        o0, o1 = t["data_offsets"]
        if t["dtype"] == "BF16":
            arr = self._decode(name, o0, (o1 - o0) // 2)
        else:
            itemsize = np.dtype(_NP[t["dtype"]]).itemsize
            arr = self._decode(name, o0, (o1 - o0) // itemsize)
        arr = arr.reshape(t["shape"])
        # astype makes a copy only when the dtype is different.
        if dtype is not None and arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype)
        return arr

    def get_bf16(self, name):
        """Return the raw bfloat16 data as a uint16 view.

        Do not convert the data. Do not copy the data. Keep the file open while
        you use the view.
        """
        t = self.header[name]
        if t["dtype"] != "BF16":
            return self.get(name)
        o0, o1 = t["data_offsets"]
        raw = np.frombuffer(self._mm, dtype=np.uint16, count=(o1 - o0) // 2, offset=self._base + o0)
        return raw.reshape(t["shape"])

    def get_rows(self, name, start, stop, dtype=np.float32):
        """Return a row range of a 2-D tensor as a NumPy array.

        Read only the necessary bytes. Convert bfloat16 data to float32 data.
        """
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
        """Return one row of a 2-D tensor as a NumPy array."""
        return self.get_rows(name, row, row + 1, dtype=dtype)[0]
