"""The EXL3 files of ExLlamaV3 (EXL3_PLAN.md): a directory of safetensors
files with quantization_config.json (quant_method "exl3").

An EXL3 matrix of k inputs and n outputs has the tensors <key>.trellis
(int16, (k / 16, n / 16, 16 K): the codes of the tiles), <key>.suh (float16,
k), <key>.svh (float16, n), and a marker of its codebook (<key>.mcg or
<key>.mul1; none: 3INST). Its weights (k x n, y = x W) are

    W = diag(suh) H_k W' H_n diag(svh)

with W' the decoded trellis (cops.exl3_decode: the float16 steps of
ExLlamaV3) and H the Hadamard matrix of 128 (Sylvester, scaled by
1 / sqrt(128)) on each part of 128 rows (H_k) or columns (H_n)
(modules/quant/exl3.py, get_weight_tensor). The other tensors (the norms,
the embeddings, the router) are as in the original checkpoint.

The products of this runtime take the MUL1 matrices as they are: a block
of the trellis, suh, and svh (EXL3Matrix.block; type KQ_E3_BASE + K of
backends/cpu/csrc/kernels/kq/exl3.c, and of the GPU). EXL3Dir reads the
directory with the read methods of SafeTensors (the names of the
checkpoint), so a model that reads a checkpoint reads it, and e3(key)
gives the matrices.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.formats.safetensors import SafeTensors

KQ_E3_BASE = 70


def is_exl3(path):
    """True for a directory of an EXL3 model."""
    if not os.path.isdir(path):
        return False
    for f in ("quantization_config.json", "config.json"):
        p = os.path.join(path, f)
        if os.path.exists(p):
            c = json.load(open(p))
            c = c.get("quantization_config", c) if f == "config.json" else c
            if c.get("quant_method") == "exl3":
                return True
    return False


def fwht128(x, axis):
    """The Hadamard transform of 128 (Sylvester order, scaled by 1 / sqrt(128))
    on each part of 128 values of x along axis (float32)."""
    x = np.moveaxis(np.asarray(x, np.float32), axis, -1)
    sh = x.shape
    y = x.reshape(-1, 128).copy()
    h = 1
    while h < 128:
        y = y.reshape(-1, 128 // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :], y[:, :, 1, :] = a + b, a - b
        h *= 2
    y = y.reshape(sh) * np.float32(1 / np.sqrt(128))
    return np.moveaxis(y, -1, axis)


class EXL3Matrix:
    """One EXL3 matrix: trellis, suh, svh, k (cols), n (rows), K, cb, bias.
    As a matrix of the products (KMat): data (the block), type, rows, cols."""

    def __init__(self, d, key):
        self.key = key
        self.trellis = d.get(key + ".trellis", dtype=None)
        self.suh = d.get(key + ".suh", dtype=None)
        self.svh = d.get(key + ".svh", dtype=None)
        self.bias = d.get(key + ".bias") if d.has(key + ".bias") else None
        self.k, self.n = self.trellis.shape[0] * 16, self.trellis.shape[1] * 16
        w = self.trellis.shape[2]
        assert w % 16 == 0, "%s: a half-integer K (%d words a tile)" % (key, w)
        self.K = w // 16
        self.cb = "mul1" if d.has(key + ".mul1") else ("mcg" if d.has(key + ".mcg") else "3inst")
        self.rows, self.cols = self.n, self.k
        self.rot = False
        self._block = None

    @property
    def type(self):
        if self.cb != "mul1":
            raise ValueError("%s: the products of this runtime take MUL1 matrices only (this one is %s)"
                             % (self.key, self.cb))
        return KQ_E3_BASE + self.K

    @property
    def data(self):
        return self.block()

    def block(self):
        """The trellis, suh, and svh in one array (uint8): the form of the
        products."""
        if self._block is None:
            self._block = np.concatenate([np.ascontiguousarray(self.trellis).reshape(-1).view(np.uint8),
                                          np.ascontiguousarray(self.suh, np.float16).view(np.uint8),
                                          np.ascontiguousarray(self.svh, np.float16).view(np.uint8)])
        return self._block

    def c(self):
        return (self.data, self.type)

    def inner(self, cb=None):
        """W' (k x n, float32) in the rotated basis (cb: the codebook of the
        decode; "mul1x" for the values of the products of this runtime)."""
        from reclank.backends.cpu import cops
        return cops.exl3_decode(self.trellis, self.k, self.n, cb or self.cb)

    def weights(self, cb=None):
        """W (k x n, float32): y = x W."""
        w = fwht128(self.inner(cb), 0) * self.suh.astype(np.float32)[:, None]
        return fwht128(w, 1) * self.svh.astype(np.float32)[None, :]


class EXL3Dir:
    """The tensors of an EXL3 model directory: the read methods of
    SafeTensors over its files, and e3(key) for the EXL3 matrices."""

    # the matrices are the bytes of the file: no weight cache
    use_cache = False
    keep_embedding_bf16 = True

    def __init__(self, path):
        self.path = path
        qc = os.path.join(path, "quantization_config.json")
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.config = cfg
        self.qconfig = json.load(open(qc)) if os.path.exists(qc) else cfg["quantization_config"]
        assert self.qconfig.get("quant_method") == "exl3", path
        self._files = {}
        self.where = {}
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.where = json.load(open(idx))["weight_map"]
        else:
            for f in sorted(os.listdir(path)):
                if f.endswith(".safetensors"):
                    for name in self._st(f).header:
                        self.where[name] = f
        gen = os.path.join(path, "generation_config.json")
        g = json.load(open(gen)) if os.path.exists(gen) else {}
        self.meta = {"general.sampling.temp": g.get("temperature"),
                     "general.sampling.top_p": g.get("top_p"),
                     "general.sampling.top_k": g.get("top_k")}
        self._mats = {}

    def _st(self, f):
        st = self._files.get(f)
        if st is None:
            st = self._files[f] = SafeTensors(os.path.join(self.path, f))
        return st

    def _of(self, name):
        return self._st(self.where[name])

    # ---- the read methods of SafeTensors ----

    @property
    def header(self):
        return self.where

    def names(self):
        return list(self.where)

    def has(self, name):
        return name in self.where

    def shape(self, name):
        return self._of(name).shape(name)

    def dtype(self, name):
        return self._of(name).dtype(name)

    def get(self, name, dtype=np.float32):
        return self._of(name).get(name, dtype=dtype)

    def get_bf16(self, name):
        st = self._of(name)
        if st.dtype(name) == "BF16":
            return st.get_bf16(name)
        a = np.ascontiguousarray(st.get(name, dtype=np.float32))
        u = a.view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

    def get_rows(self, name, start, stop, dtype=np.float32):
        return self._of(name).get_rows(name, start, stop, dtype=dtype)

    def get_row(self, name, row, dtype=np.float32):
        return self._of(name).get_row(name, row, dtype=dtype)

    def release_pages(self):
        """Do nothing: the blocks of the matrices are copies."""

    def close(self):
        for st in self._files.values():
            st.close()

    # ---- the configuration ----

    def text_config(self):
        return self.config.get("text_config", self.config)

    # ---- the EXL3 matrices ----

    def is_e3(self, key):
        return self.has(key + ".trellis")

    def e3(self, key):
        """The EXL3 matrix of a module key (without .weight)."""
        m = self._mats.get(key)
        if m is None:
            m = self._mats[key] = EXL3Matrix(self, key)
        return m

    def matrix(self, key):
        return self.e3(key)
