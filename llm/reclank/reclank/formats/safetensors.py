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
    # FP8 (the NVFP4 scales, the n-gram table of Qwen3.8): the raw bytes;
    # reclank/formats/nvfp4_st.py decodes them.
    "F8_E4M3": np.uint8,
}


def drop_pages(arr):
    """Drop from the process's memory the pages of a NumPy view of a SafeTensors map (the map is
    read-only and shared: a page used again is read again from the file). The pages at its ends
    may hold the neighbours' data: they are read again too when used. Return the bytes of the view
    (0: not a view of a map)."""
    x = arr
    while isinstance(x, np.ndarray) and x.base is not None:
        x = x.base
    if isinstance(x, memoryview):
        x = x.obj
    if not isinstance(x, mmap.mmap) or arr.size == 0:
        return 0
    bounds = getattr(np.lib, "array_utils", np).byte_bounds(arr)
    base = np.frombuffer(x, np.uint8).ctypes.data
    lo = (bounds[0] - base) // mmap.PAGESIZE * mmap.PAGESIZE
    try:
        x.madvise(mmap.MADV_DONTNEED, lo, bounds[1] - base - lo)
    except (AttributeError, OSError, ValueError):
        return 0
    return arr.nbytes


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
        # Ask for large pages. A large page holds more data in the translation
        # lookaside buffer. The call does nothing when the system says no.
        try:
            self._mm.madvise(mmap.MADV_HUGEPAGE)
        except (AttributeError, OSError):
            pass

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

    def release_pages(self):
        """Tell the operating system to drop the mapped pages from memory.

        Use this method after the code copies all the data. Use it after an
        int8 load. Do not use it when NumPy views still point into the map.
        """
        try:
            self._mm.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, OSError):
            pass

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


class Shards:
    """The SafeTensors files of a directory as one: the weight map of its
    model.safetensors.index.json (or the one model.safetensors), each name
    read from its file (get, get_bf16, get_rows, get_row, shape, dtype)."""

    def __init__(self, path, index="model.safetensors.index.json"):
        import os
        ip = os.path.join(path, index)
        if os.path.exists(ip):
            where = json.load(open(ip))["weight_map"]
        else:
            one = [f for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
            assert len(one) == 1, "no index and %d safetensors files in %s" % (len(one), path)
            where = {n: one[0] for n in SafeTensors(os.path.join(path, one[0])).names()}
        self.files = {f: SafeTensors(os.path.join(path, f)) for f in sorted(set(where.values()))}
        self.where = {n: self.files[f] for n, f in where.items()}

    def names(self):
        return list(self.where)

    def close(self):
        for f in self.files.values():
            f.close()

    def __getattr__(self, attr):
        if attr in ("get", "get_bf16", "get_rows", "get_row", "shape", "dtype"):
            return lambda name, *a, **kw: getattr(self.where[name], attr)(name, *a, **kw)
        raise AttributeError(attr)
