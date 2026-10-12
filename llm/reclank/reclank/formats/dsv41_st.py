"""DeepSeek V4.1 Flash from its safetensors checkpoint (DEEPSEEK_V41_PLAN.md 1,
Phase 1): the view of the shards, the forms of this runtime, nothing written.

    src = DSV41Source("/home/cpage/models/DeepSeek-V4.1-Flash")
    src.cfg["hidden_size"]                 # the text config
    src.raw(name) -> (uint8 bytes, dtype, shape)      # as stored
    src.fp8(name) -> (codes uint8 [r, c], scales uint8 [r/32, c/32])   # FP8, 32 x 32 blocks
    src.bf16(name) -> uint16 [r, c]
    src.expert(layer, e) -> {"w1": (codes, scales), "w3": ..., "w2": ...}   # MXFP4 rows
    src.expert_mxx(layer, e, w, rows=None) -> KQ_MXX bytes of the rows (all, or a range)

The values: E4M3 (e4m3_values: the 256 codes), E2M1 (E2M1: the 16 codes),
E8M0 (e8m0_values: 2^(e - 127)). dequant_fp8 and dequant_mxfp4 give float32
(the checks; the runtime keeps the codes).
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def e4m3_values():
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 512.0, (1 + m / 8.0) * 2.0 ** (e - 7))
    v = np.where(b & 128, -v, v)
    v[(b & 127) == 127] = np.nan
    return v.astype(np.float32)


def e8m0_values():
    v = np.ldexp(1.0, np.arange(256) - 127)
    v[255] = np.nan
    return v.astype(np.float32)


E4M3 = e4m3_values()
E8M0 = e8m0_values()


def dequant_fp8(codes, scales):
    """float32 [r, c] of FP8 codes with E8M0 scales of each 32 x 32 block."""
    r, c = codes.shape
    s = E8M0[scales].repeat(32, 0).repeat(32, 1)[:r, :c]
    return E4M3[codes] * s


def dequant_mxfp4(codes, scales):
    """float32 [r, 2 c] of MXFP4 rows (two codes a byte, the low half first)
    with an E8M0 scale for each 32 values."""
    r = codes.shape[0]
    v = np.stack([E2M1[codes & 15], E2M1[codes >> 4]], -1).reshape(r, -1)
    return (v.reshape(r, -1, 32) * E8M0[scales][:, :, None]).reshape(r, -1)


class DSV41Source:
    def __init__(self, path):
        self.path = path
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.cfg = cfg.get("text_config", cfg)
        self.L = int(self.cfg["num_hidden_layers"])
        self.E = int(self.cfg["n_routed_experts"])
        self.hidden = int(self.cfg["hidden_size"])
        self.inner = int(self.cfg["moe_intermediate_size"])
        where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.t = {}
        for f in sorted(set(where.values())):
            with open(os.path.join(path, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            for k, v in h.items():
                o0, o1 = v["data_offsets"]
                self.t[k] = (f, v["dtype"], tuple(v["shape"]), 8 + n + o0, 8 + n + o1)
        self._maps = {}

    def names(self):
        return list(self.t)

    def _map(self, f):
        m = self._maps.get(f)
        if m is None:
            m = self._maps[f] = np.memmap(os.path.join(self.path, f), np.uint8, "r")
        return m

    def raw(self, name, copy=True):
        """The bytes of a tensor (a copy, or a view of the map of its shard)."""
        f, dt, shape, o0, o1 = self.t[name]
        a = self._map(f)[o0:o1]
        return (np.array(a) if copy else a), dt, shape

    def fp8(self, name):
        codes, dt, shape = self.raw(name)
        assert dt == "F8_E4M3", (name, dt)
        sc, sdt, sshape = self.raw(name[:-len("weight")] + "scale")
        return codes.reshape(shape), sc.reshape(sshape)

    def bf16(self, name):
        b, dt, shape = self.raw(name)
        assert dt == "BF16", (name, dt)
        return b.view(np.uint16).reshape(shape)

    def f32(self, name):
        """A float32 copy of a tensor stored as BF16 or F32."""
        b, dt, shape = self.raw(name)
        if dt == "BF16":
            return (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        assert dt == "F32", (name, dt)
        return b.view(np.float32).reshape(shape).copy()

    def expert(self, layer, e, prefix="layers"):
        out = {}
        for w in ("w1", "w3", "w2"):
            base = "%s.%d.ffn.experts.%d.%s." % (prefix, layer, e, w)
            codes, dt, shape = self.raw(base + "weight")
            sc, _sdt, sshape = self.raw(base + "scale")
            out[w] = (codes.reshape(shape), sc.reshape(sshape))
        return out

    def expert_mxx(self, layer, e, w, rows=None, prefix="layers"):
        """The KQ_MXX bytes of rows (r0, n) (all by default) of matrix w of an expert."""
        from reclank.backends.cpu import cops
        codes, sc = self.expert(layer, e, prefix)[w]
        r0, n = rows or (0, codes.shape[0])
        return cops.kq_mxx_pack(codes[r0:r0 + n], sc[r0:r0 + n], n, codes.shape[1] * 2)
