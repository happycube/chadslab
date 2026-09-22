"""Store converted weights in a local cache directory.

The int8 conversion reads the full model and quantizes six billion parameters.
This work takes some minutes. The cache keeps the result on disk. The next load
reads the cache instead of the full conversion.

The default cache directory is ~/.cache/np_gemma/weights. The home directory is
on a local disk. The project directory can be a network mount. Set the variable
NP_GEMMA_CACHE to use a different directory. Set the variable
NP_GEMMA_CACHE_RAM=1 to copy the cache into local memory. The copy can use large
pages. The copy uses more memory.

The cache has two files:
    manifest.json  Gives the dtype, the shape, and the offset of each tensor.
    data.bin       Holds the raw tensor bytes.

The cache key covers the source path, the source size, the source time, and the
dtype. A new source file or a new dtype gives a new directory. Thus the code
never uses a stale cache.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path

import numpy as np


def default_root():
    """Return the cache root directory.

    Use the directory in the variable NP_GEMMA_CACHE when it is set.
    Otherwise use ~/.cache/np_gemma/weights. The home directory is a local
    disk. The project directory can be a network mount. A fast local disk
    gives a faster first token.
    """
    override = os.environ.get("NP_GEMMA_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "np_gemma" / "weights"


class WeightCache:
    """Read and write converted weights in one cache directory."""

    def __init__(self, source, dtype, extra="", root=None, ram=None):
        self.root = Path(root) if root is not None else default_root()
        if ram is None:
            ram = os.environ.get("NP_GEMMA_CACHE_RAM") == "1"
        self.ram = bool(ram)
        self._buffer = None
        stat = os.stat(source)
        key = hashlib.sha256(
            ("%s|%d|%d|%s|%s" % (os.path.abspath(source), stat.st_size, int(stat.st_mtime), dtype, extra)).encode()
        ).hexdigest()[:16]
        self.directory = self.root / key
        self.manifest_path = self.directory / "manifest.json"
        self.data_path = self.directory / "data.bin"
        self.manifest = None
        self._fh = None
        self._offset = 0

    def ready(self):
        """Return True when a complete cache exists. Read the manifest."""
        if not (self.manifest_path.exists() and self.data_path.exists()):
            return False
        try:
            self.manifest = json.loads(self.manifest_path.read_text())
            self.manifest["tensors"]
        except Exception:
            return False
        return True

    def load_buffer(self):
        """Read the data file into one local memory block. Ask for large pages.

        The first call reads the file. Later calls return the same block. The
        start of the block is aligned to 2 MiB, so the system can use large
        pages. A large page covers 512 small pages. Thus the kernel needs fewer
        TLB entries for a long read.
        """
        if self._buffer is not None:
            return self._buffer
        size = os.path.getsize(self.data_path)
        align = 2 << 20
        raw = np.empty(size + align, dtype=np.uint8)
        base = raw.ctypes.data
        start = (base + align - 1) & ~(align - 1)
        buffer = raw[start - base:start - base + size]
        with open(self.data_path, "rb") as fh:
            got = fh.readinto(memoryview(buffer))
        if got != size:
            raise OSError("short read: %d of %d bytes" % (got or -1, size))
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.madvise(ctypes.c_void_p(buffer.ctypes.data), ctypes.c_size_t(size),
                         ctypes.c_int(mmap.MADV_HUGEPAGE))
        except Exception:
            pass
        self._buffer = buffer
        return self._buffer

    def read(self, name):
        """Return one tensor from the cache."""
        entry = self.manifest["tensors"][name]
        if self.ram:
            buffer = self.load_buffer()
            return np.ndarray(tuple(entry["shape"]), dtype=np.dtype(entry["dtype"]),
                              buffer=buffer, offset=entry["offset"])
        array = np.memmap(self.data_path, dtype=np.dtype(entry["dtype"]), mode="r",
                          offset=entry["offset"], shape=tuple(entry["shape"]))
        # Ask for large pages. The call does nothing when the system says no.
        try:
            array._mmap.madvise(mmap.MADV_HUGEPAGE)
        except (AttributeError, OSError):
            pass
        return array

    def close(self):
        """Release the local memory copy."""
        self._buffer = None

    def open_write(self):
        """Start a write operation. Make the cache directory."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.data_path, "wb")
        self._offset = 0
        self.manifest = {"tensors": {}}

    def write(self, name, array):
        """Write one tensor. Record the dtype, the shape, and the offset."""
        array = np.ascontiguousarray(array)
        array.tofile(self._fh)
        self.manifest["tensors"][name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "offset": self._offset,
        }
        self._offset += array.nbytes

    def close_write(self):
        """Close the data file. Write the manifest."""
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.manifest = None
