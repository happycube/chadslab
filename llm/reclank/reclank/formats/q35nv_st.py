"""Qwen3.5 / Qwen3.6 MoE from an NVFP4 checkpoint of NVIDIA ModelOpt (nvidia/Qwen3.6-35B-A3B-NVFP4:
quant_algo MIXED_PRECISION, model_type qwen3_5_moe).

Q35NVSource gives the GGUF view of the checkpoint that reclank/models/qwen/qwen35.py reads (the
tensors of llama.cpp's qwen35moe: tensors, meta, raw, dequant), for the weightkeeper
(reclank/models/qwen/keep_q35nv.py) and the model itself. The forms:

- the routed experts: NVFP4 (E2M1 codes, an E4M3 scale for each 16 values, the float32
  weight_scale_2 of each matrix), KQ_NVX stacks of each layer and matrix (lossless);
- the FP8 matrices (the attention's q, k, v, o and the DeltaNet's qkv, z and out: E4M3 codes and
  one float32 weight_scale): their float32 values, the code times the scale, as ModelOpt
  dequantizes them. The static input_scale of each (the A8 of W8A8) is not applied: the products
  read float32 x;
- the NVFP4 shared experts: KQ_NVX too (the kernels of the experts take x as int16 only when every
  matrix of the layer does); the NVFP4 head: its float32 values (the code times its block scale,
  then times the matrix's scale);
- the bfloat16 tensors: the embeddings as they are (KQ_BF16, rows on demand); the routers, the
  shared experts' gates and the DeltaNet's a and b float32; the norms float32 plus 1 where the
  converter of llama.cpp adds it (all but the DeltaNet's gated norm), ssm_a -exp(A_log);
- the value heads of the DeltaNet in the checkpoint's order (grouped by key head): the config
  has v_tiled False.

On the GPU the float32 matrices of the products (the FP8 ones and the head) go as F20 rows
(models/qwen/gpu35.py QwenGPU.dense, f20_dense; RECLANK_GPU_F20=0 keeps float32): 4.5 GB in place
of 7.2, more hot experts. The weightkeeper and the CPU keep the float32 values.

The MTP layer of this checkpoint has no experts (exclude_modules): it is not part of the view.

    src = Q35NVSource("models/Qwen3.6-35B-A3B-NVFP4")
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.formats.nvfp4_st import E4M3, KQ_BF16, KQ_F32, KQ_NVX
from reclank.formats.safetensors import SafeTensors

MAIN = "model.language_model."
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def is_q35nv(path):
    """True for a checkpoint directory of qwen3_5_moe with NVFP4 experts of ModelOpt."""
    try:
        c = json.load(open(os.path.join(path, "config.json")))
        q = json.load(open(os.path.join(path, "hf_quant_config.json")))["quantization"]
    except (OSError, ValueError, KeyError):
        return False
    return (c.get("model_type") == "qwen3_5_moe" and q.get("quant_algo") == "MIXED_PRECISION"
            and any(k.endswith("mlp.experts") and v.get("quant_algo") == "W4A16_NVFP4"
                    for k, v in q.get("quantized_layers", {}).items()))


def nvfp4_dequant(codes, scales, gs):
    """The float32 values of an NVFP4 matrix (codes: uint8 [rows, cols / 2], the low nibble the even
    value; scales: E4M3 bytes [rows, cols / 16]; gs: the float32 scale of the matrix): the E2M1
    value times its E4M3 scale (exact in float32), then times gs."""
    codes = np.asarray(codes, np.uint8)
    rows, half = codes.shape
    v = np.empty((rows, 2 * half), np.float32)
    v[:, 0::2] = E2M1[codes & 15]
    v[:, 1::2] = E2M1[codes >> 4]
    v = (v.reshape(rows, -1, 16) * E4M3[np.asarray(scales, np.uint8)][:, :, None]).reshape(rows, -1)
    return v * np.float32(gs)


class Q35NVSource:
    """The GGUF view of the checkpoint (see the module text)."""

    f20_dense = True        # (its float32 products go to the GPU as F20 rows: models/qwen/gpu35.py)

    def __init__(self, path):
        self.path = path
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.hf = cfg.get("text_config", cfg)
        self.where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self._files = {}
        self._cache = {}
        self.L = int(self.hf["num_hidden_layers"])
        self.E = int(self.hf["num_experts"])
        self.meta = {"general.architecture": "qwen35moe"}
        self.tensors = {}           # GGUF name -> (dims (ggml order), type, 0)
        self._map = {}              # GGUF name -> (kind, source)
        self._build()

    # ---- the files ----

    def _file(self, name):
        f = self.where[name]
        st = self._files.get(f)
        if st is None:
            st = self._files[f] = SafeTensors(os.path.join(self.path, f))
        return st

    def _get(self, name, dtype=np.float32):
        return self._file(name).get(name, dtype=dtype)

    def _shape(self, name):
        return tuple(self._file(name).shape(name))

    def _scalar(self, name):
        return float(np.asarray(self._get(name)).reshape(-1)[0])

    # ---- the map of the names ----

    def _add(self, gname, kind, src, dims, type_):
        self._map[gname] = (kind, src)
        self.tensors[gname] = (tuple(int(d) for d in dims), type_, 0)

    def _f32(self, gname, hf, fn=None, dims=None):
        self._add(gname, "f32", (hf, fn), dims or tuple(reversed(self._shape(hf))), KQ_F32)

    def _fp8(self, gname, hf):
        """An FP8 matrix (hf: its prefix), float32."""
        rows, cols = self._shape(hf + ".weight")
        self._add(gname, "fp8", hf, (cols, rows), KQ_F32)

    def _nvfp4(self, gname, hf):
        """An NVFP4 matrix (hf: its prefix), float32."""
        rows, half = self._shape(hf + ".weight")
        self._add(gname, "nvfp4", hf, (2 * half, rows), KQ_F32)

    def _layer(self, i):
        b, base = "blk.%d." % i, MAIN + "layers.%d." % i
        one = lambda a: a + 1.0  # noqa: E731
        self._f32(b + "attn_norm.weight", base + "input_layernorm.weight", one)
        self._f32(b + "post_attention_norm.weight", base + "post_attention_layernorm.weight", one)
        m = base + "mlp."
        self._f32(b + "ffn_gate_inp.weight", m + "gate.weight")
        self._f32(b + "ffn_gate_inp_shexp.weight", m + "shared_expert_gate.weight",
                  dims=(self._shape(m + "shared_expert_gate.weight")[-1],))
        for x in ("gate", "up", "down"):
            hf = m + "shared_expert.%s_proj" % x
            rows, half = self._shape(hf + ".weight")
            self._add(b + "ffn_%s_shexp.weight" % x, "nv4mat", hf, (2 * half, rows), KQ_NVX)
            rows, half = self._shape(m + "experts.0.%s_proj.weight" % x)
            self._add(b + "ffn_%s_exps.weight" % x, "nv4exps", (m, x), (2 * half, rows, self.E), KQ_NVX)
        a = base + "self_attn."
        if a + "q_proj.weight" in self.where:
            for x, g in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output")):
                self._fp8(b + g + ".weight", a + "%s_proj" % x)
            self._f32(b + "attn_q_norm.weight", a + "q_norm.weight", one)
            self._f32(b + "attn_k_norm.weight", a + "k_norm.weight", one)
        la = base + "linear_attn."
        if la + "in_proj_qkv.weight" in self.where:
            self._fp8(b + "attn_qkv.weight", la + "in_proj_qkv")
            self._fp8(b + "attn_gate.weight", la + "in_proj_z")
            self._fp8(b + "ssm_out.weight", la + "out_proj")
            self._f32(b + "ssm_alpha.weight", la + "in_proj_a.weight")
            self._f32(b + "ssm_beta.weight", la + "in_proj_b.weight")
            self._f32(b + "ssm_norm.weight", la + "norm.weight")
            self._f32(b + "ssm_a", la + "A_log", lambda a_: -np.exp(a_))
            self._f32(b + "ssm_dt.bias", la + "dt_bias")
            conv = la + "conv1d.weight"
            ch, _one, k = self._shape(conv)
            self._f32(b + "ssm_conv1d.weight", conv, lambda w: w.reshape(ch, k), dims=(k, ch))

    def _build(self):
        for i in range(self.L):
            self._layer(i)
        self._add("token_embd.weight", "bf16", MAIN + "embed_tokens.weight",
                  tuple(reversed(self._shape(MAIN + "embed_tokens.weight"))), KQ_BF16)
        self._f32("output_norm.weight", MAIN + "norm.weight", lambda a: a + 1.0)
        self._nvfp4("output.weight", "lm_head")

    # ---- the config ----

    def config(self):
        """The QwenConfig of the model (config.json; the value heads in the checkpoint's order)."""
        from reclank.models.qwen.qwen35 import QwenConfig
        cfg = QwenConfig(self.path)
        cfg.v_tiled = False
        return cfg

    # ---- the tensors ----

    def _make(self, gname):
        kind, src = self._map[gname]
        if kind == "f32":
            hf, fn = src
            a = self._get(hf)
            if fn is not None:
                a = fn(a)
            return np.ascontiguousarray(a, dtype=np.float32)
        if kind == "bf16":
            return self._file(src).get_bf16(src)
        if kind == "fp8":
            w = E4M3[self._get(src + ".weight", dtype=None)]
            return np.ascontiguousarray(w * np.float32(self._scalar(src + ".weight_scale")), np.float32)
        if kind == "nvfp4":
            return nvfp4_dequant(self._get(src + ".weight", dtype=None), self._get(src + ".weight_scale", dtype=None),
                                 self._scalar(src + ".weight_scale_2"))
        if kind == "nv4exps":
            return self.pack_experts(gname)
        if kind == "nv4mat":
            cols, rows = self.tensors[gname][0]
            out = np.empty(rows * cops.kq_nvx_row_bytes(cols), np.uint8)
            cops.kq_nvx_pack([np.ascontiguousarray(self._get(src + ".weight", dtype=None))],
                             [np.ascontiguousarray(self._get(src + ".weight_scale", dtype=None))],
                             [self._scalar(src + ".weight_scale_2")], rows, cols, out)
            return out
        raise KeyError(gname)

    def pack_experts(self, gname, out=None):
        """The KQ_NVX stack of a layer's matrix of the routed experts (uint8; out: its room)."""
        m, x = self._map[gname][1]
        dims = self.tensors[gname][0]
        cols, rows = int(dims[0]), int(dims[1])
        ws, ss, gs = [], [], []
        for e in range(self.E):
            p = m + "experts.%d.%s_proj." % (e, x)
            ws.append(np.ascontiguousarray(self._get(p + "weight", dtype=None)))
            ss.append(np.ascontiguousarray(self._get(p + "weight_scale", dtype=None)))
            gs.append(self._scalar(p + "weight_scale_2"))
        if out is None:
            out = np.empty(self.E * rows * cops.kq_nvx_row_bytes(cols), np.uint8)
        cops.kq_nvx_pack(ws, ss, gs, rows, cols, out)
        return out

    def raw(self, gname):
        """(the data, dims in the ggml order, the type), as GGUF.raw."""
        a = self._cache.get(gname)
        if a is None:
            a = self._cache[gname] = self._make(gname)
        dims, type_, _ = self.tensors[gname]
        return a, dims, type_

    def dequant(self, gname, rows=None):
        """float32 values in the NumPy shape (rows selects the first index), as GGUF.dequant."""
        dims, type_, _ = self.tensors[gname]
        shape = tuple(reversed(dims))
        a, _dims, _t = self.raw(gname)
        if type_ == KQ_F32:
            a = a.reshape(shape)
            return a if rows is None else a[np.asarray(rows)]
        cols = dims[0]
        per = int(np.prod(shape[1:-1])) if len(shape) > 2 else 1   # the rows of one index
        idx = np.arange(shape[0]) if rows is None else np.asarray(rows).reshape(-1)
        ids = (idx[:, None] * per + np.arange(per)[None, :]).reshape(-1)
        out = cops.kq_rows(np.asarray(a).reshape(-1).view(np.uint8), type_, cols, ids)
        return out.reshape((len(idx),) + shape[1:])

    def file_bytes(self):
        """The bytes of the safetensors files."""
        return sum(os.path.getsize(os.path.join(self.path, f)) for f in set(self.where.values()))

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}
