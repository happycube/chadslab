"""Qwen3.8-Flash-Next from an EXL3 directory of ExLlamaV3 (turboderp/Qwen3.8-Flash-Next-exl3,
e.g. 5.05bpw_h6_ng6; docs/EXL3_PLAN.md).

EXL3Source is NVFP4Source (reclank/formats/nvfp4_st.py: the GGUF view of a
checkpoint that reclank/models/qwen/qwen38.py reads) on the tensors of the
directory (reclank/formats/exl3.py, EXL3Dir):

- the EXL3 matrices (MUL1: the attention, the DeltaNet, the shared expert,
  the head) as they are: the blocks of the products, type KQ_E3_BASE + K;
- the routed experts: one stack of the blocks of the 512 experts for each
  layer and matrix (the form of kernels/kq/exl3_moe.c);
- a matrix that the model reads as float32 (the indexer's index_qk_proj,
  split into q and k): the weights of its trellis, decoded;
- the float16 matrices (the hyper-connections, the n-gram projections):
  bfloat16 (they came from bfloat16: the round trip is exact) or float32;
- the norms: bfloat16 to float32, plus 1 as in nvfp4_st;
- the n-gram table: its own trellis (ngram_embedding.safetensors, the
  exl3_ngram_trellis form of ExLlamaV3, modules/quant/exl3_lib/ngram_codec.py):
  each row 1 + 160 K / 16 words, word 0 a float16 scale, then a ring of 160
  K-bit symbols; the 16-bit state of position i is the symbols of i, i - 1,
  ... (K bits each, from bit 0 up, mod 160), and the value is
  mul1(state) * scale + head_bias[head of the row]. ple_rows decodes the
  rows it reads.

The MTP layer as nvfp4_st makes it (layer L): its EXL3 matrices and stack
of experts as those of the other layers; eh_proj, one matrix of
fc_embedding and fc_hidden (two EXL3 matrices with transforms of their
own), from their weights decoded, in bfloat16; the hyper-connection mixer
from mtp_hyper_connection_mixer_patch.safetensors.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.formats.exl3 import KQ_E3_BASE, EXL3Dir
from reclank.formats.nvfp4_st import KQ_BF16, KQ_F32, MAIN, NVFP4Source

# the names of the n-gram parameters in nvfp4_st (config) and in the EXL3 directory
_PLE_ALIAS = {"ple_embedding.layer_multipliers": "ple_embedding.ngram_embedding.layer_multipliers",
              "ple_embedding.ngram_heads_offsets": "ple_embedding.ngram_embedding.head_offsets",
              "ple_embedding.ngram_heads_vocab_sizes": "ple_embedding.ngram_embedding.head_vocab_sizes"}


def mul1_table():
    """The 65536 MUL1 values (float16 steps, as the codebook of ExLlamaV3), float32."""
    s = np.arange(65536, dtype=np.uint64)
    p = (s * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)
    b = sum((p >> np.uint64(8 * i)) & np.uint64(255) for i in range(4)).astype(np.float32)
    k_inv = np.array([0x1eee], np.uint16).view(np.float16)[0]
    k_bias = np.array([0xc931], np.uint16).view(np.float16)[0]
    return ((1024 + b).astype(np.float32) * np.float32(k_inv) + np.float32(k_bias)).astype(np.float16) \
        .astype(np.float32)


MUL1 = None


def ngram_decode(packed, K, bias):
    """The float32 rows (n x dim) of packed n-gram rings (n x words, int16) with the bias of
    each row (n x dim)."""
    global MUL1
    if MUL1 is None:
        MUL1 = mul1_table()
    n, words = packed.shape
    dim = (words - 1) * 16 // K
    scale = packed[:, 0].copy().view(np.float16).astype(np.float32)
    stream = packed[:, 1:].view(np.uint16).astype(np.int64)
    bit = np.arange(dim) * K
    nw = words - 1
    win = stream[:, bit >> 4] | (stream[:, ((bit >> 4) + 1) % nw] << 16)
    sym = (win >> (bit & 15)) & ((1 << K) - 1)
    st = np.zeros_like(sym)
    for j in range((15 + K) // K):
        st |= np.roll(sym, j, axis=1) << (j * K)
    return MUL1[st & 0xFFFF] * scale[:, None] + bias


class EXL3Source(NVFP4Source):
    """The GGUF view of an EXL3 directory of Qwen3.8-Flash-Next (see the module text)."""

    exl3 = True     # (qwen38.dense_mode: the bfloat16 matrices as they are)
    # the stacks of the experts are in memory of the process (_make: the blocks
    # read and joined), not a map of the files: QwenGPU page-locks them for the
    # copies of HotCache and of the prompt groups (_register_experts; a prompt
    # of 512 tokens 297 tok/s against 154 from pageable memory)
    staged = True

    def __init__(self, path):
        self.d = EXL3Dir(path)
        # the files outside the index (the n-gram table, the patch of the MTP mixer)
        indexed = set(self.d.where.values())
        for f in sorted(os.listdir(path)):
            if f.endswith(".safetensors") and f not in indexed:
                for name in self.d._st(f).header:
                    self.d.where.setdefault(name, f)
        # <key>.weight for each EXL3 module (nvfp4_st finds the parts of a
        # layer by those names, the MTP layer's too; _get gives their weights)
        for name, f in list(self.d.where.items()):
            if name.endswith(".trellis") and ".ngram_embedding." not in name:
                self.d.where.setdefault(name[:-8] + ".weight", f)
        super().__init__(path)

    # ---- the files (EXL3Dir) ----

    def _module(self, name):
        """The EXL3 module of a name <key>.weight, or None."""
        if name.endswith(".weight") and self.d.is_e3(name[:-7]):
            return name[:-7]
        return None

    def _get(self, name, dtype=np.float32):
        for a, b in _PLE_ALIAS.items():
            if name.endswith(a):
                name = name[:-len(a)] + b
        key = self._module(name)
        if key is not None:
            # (n x k, the form of the checkpoint) from the weights of the trellis
            return np.ascontiguousarray(self.d.e3(key).weights().T)
        return self.d.get(name, dtype=dtype)

    def _bf16(self, name):
        return self.d.get_bf16(name)

    def _shape(self, name):
        key = self._module(name)
        if key is not None:
            m = self.d.e3(key)
            return (m.n, m.k)
        return tuple(self.d.shape(name))

    @property
    def where(self):
        return self.d.where

    @where.setter
    def where(self, v):
        pass        # (NVFP4Source.__init__ sets it from the index: the EXL3Dir map holds more)

    # ---- the map of the names ----

    def _dense_mat(self, gname, hf):
        key = self._module(hf)
        if key is not None:
            m = self.d.e3(key)
            self._add(gname, "e3", key, (m.k, m.n), KQ_E3_BASE + m.K)
        else:
            rows, cols = self._shape(hf)
            self._add(gname, "dense", hf, (cols, rows), KQ_BF16)

    def _experts(self, b, m, mtp):
        for x in ("gate", "up", "down"):
            key = m + "experts.0.%s_proj" % x
            e0 = self.d.e3(key)
            self._add(b + "ffn_%s_exps.weight" % x, "e3exps", (m, x), (e0.k, e0.n, self.E), KQ_E3_BASE + e0.K)

    def _build(self):
        super()._build()
        import re
        tr = [n for n in self.d.where if re.search(r"\.ple_embedding\.ngram_embedding(\.shard_\d+)?\.trellis$", n)]
        if tr:
            # the n-gram table: one trellis tensor, or shards of it (rows on
            # demand: ple_rows)
            tr.sort(key=lambda n: int(n.rsplit(".shard_", 1)[1].split(".")[0]) if ".shard_" in n else 0)
            self._ple_shards = tr
            self._ple_key = tr[0].split(".ngram_embedding.")[0] + ".ngram_embedding"
            sizes = [self.d.shape(n)[0] for n in tr]
            self._ple_start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            words = self.d.shape(tr[0])[1]
            self._ple_K = (words - 1) * 16 // 160
            self._ple_words = words
            self._add("per_layer_token_embd.weight", "ple", None, (160, int(self._ple_start[-1])), KQ_F32)

    # ---- the tensors ----

    def _make(self, gname):
        kind, src = self._map[gname]
        if kind == "e3":
            return self.d.e3(src).block()
        if kind == "e3exps":
            m, x = src
            blocks = [self.d.e3(m + "experts.%d.%s_proj" % (e, x)) for e in range(self.E)]
            K = blocks[0].K
            assert all(b.K == K for b in blocks), "%s: the experts of a matrix have one K" % gname
            # on huge pages: the CPU part of a step reads a tile of each row of
            # tiles of a column, a page apart (TLB misses on 4 KB pages)
            from reclank.runtime import numa
            first = blocks[0].block()
            out = numa.empty_huge(first.nbytes * len(blocks)).reshape(len(blocks), -1)
            for e, b in enumerate(blocks):
                out[e] = b.block().reshape(-1).view(np.uint8)
                b._block = None         # (the stack holds the bytes)
            return out.reshape(-1).view(first.dtype)
        return super()._make(gname)

    def dequant(self, gname, rows=None):
        dims, type_, _ = self.tensors[gname]
        if self._map[gname][0] == "e3":
            w = self.d.e3(self._map[gname][1]).weights().T            # (n x k)
            return w if rows is None else w[np.asarray(rows)]
        if self._map[gname][0] == "e3exps":
            raise ValueError("%s: the EXL3 experts have no rows of their own" % gname)
        return super().dequant(gname, rows)

    def ple_rows(self, rows):
        """The float32 rows of the n-gram table, decoded."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        base = self._cache.get("_ple_base")
        if base is None:
            self._ple_hold = [self.d.get(n, dtype=None) for n in self._ple_shards]
            base = self._cache["_ple_base"] = np.array([t.ctypes.data for t in self._ple_hold], np.int64)
            self._ple_bias = np.asarray(self.d.get(self._ple_key + ".head_bias"), np.float32)
            self._ple_offs = np.asarray(self.d.get(self._ple_key + ".head_offsets", dtype=None), np.int64)
        rb = 2 * self._ple_words
        k = np.searchsorted(self._ple_start, rows, side="right") - 1
        addrs = base[k] + (rows - self._ple_start[k]) * rb
        raw = cops.kq_gather(addrs, rb).view(np.int16).reshape(len(rows), self._ple_words)
        head = np.clip(np.searchsorted(self._ple_offs, rows, side="right") - 1, 0, len(self._ple_offs) - 1)
        global MUL1
        if MUL1 is None:
            MUL1 = mul1_table()
        return cops.e3_ngram_decode(raw, self._ple_K, MUL1, self._ple_bias, head)     # (ngram_decode in C)

    def _ple_random(self):
        pass

    def file_bytes(self):
        return sum(os.path.getsize(os.path.join(self.path, f)) for f in set(self.d.where.values()))


def is_qwen4_exl3(path):
    """True for an EXL3 directory of Qwen3.8-Flash-Next (qwen4_exp)."""
    from reclank.formats.exl3 import is_exl3
    if not is_exl3(path):
        return False
    c = json.load(open(os.path.join(path, "config.json")))
    return c.get("text_config", c).get("model_type", c.get("model_type", "")).startswith("qwen4_exp")


__all__ = ["EXL3Source", "is_qwen4_exl3", "ngram_decode", "MAIN"]
