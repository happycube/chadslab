"""Qwen3.8-Flash-Next from the safetensors checkpoint of NVIDIA ModelOpt
(nvidia/Qwen3.8-Flash-Next-NVFP4).

The code of the model (np_gemma/qwen4.py: Qwen4, Qwen4CPU, Qwen4GPU) reads
the tensors of a GGUF file of llama.cpp: by GGUF name, in the GGUF forms.
NVFP4Source gives the same view of the checkpoint (raw, dequant, tensors),
and makes the forms at startup:

- the routed experts (NVFP4: 4-bit E2M1 codes, an E4M3 scale for each 16
  values, a float32 scale for each matrix): one stack for each layer and
  matrix, in the rows of KQ_NV4 (csrc/kquants.c). The file keeps all the
  codes of a shard together, then all the scales, in the order of the names
  (expert 0, 1, 10, 100, ...), so the kernels cannot read it in place;
- the large bfloat16 matrices: as they are (KQ_BF16). The model can
  requantize them to Q8_0 (Qwen4CPU dense, qwen4.dense_mode);
- the small matrices (the routers, the gates, the inject weights, the
  indexer): float32;
- the norms: float32, plus 1 where the converter of llama.cpp adds it (the
  weights of the file are centred on 0);
- the MTP experts (FP8 with a scale for each 128 x 128): Q8_0;
- the n-gram table (FP8 in 128 shards, one scale): dequant(rows=...) and
  ple_rows read the rows from the shards;
- the embeddings: bfloat16 (KQ_BF16), rows on demand.

The value heads of the DeltaNet stay in the order of the checkpoint (grouped
by key head): the config has v_tiled False, and the GDN records take that
order.

    m = Qwen4CPU("models/Qwen3.8-Flash-Next-NVFP4")   # a directory: this source
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import cops
from .st import SafeTensors

KQ_F32, KQ_Q8_0, KQ_BF16, KQ_NV4 = 0, 8, 30, 51
MAIN = "model.language_model."


def is_checkpoint(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "model.safetensors.index.json"))


def _e4m3_table():
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 512.0, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where(b & 128, -v, v).astype(np.float32)


E4M3 = _e4m3_table()


class NVFP4Source:
    """The GGUF view of the checkpoint (see the module text)."""

    def __init__(self, path, headers=None):
        """headers (a test): a dict of the safetensors headers of all the
        files, for the shapes before the files are all there."""
        self.path = path
        self._headers = headers
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.hf = cfg.get("text_config", cfg)
        self.where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self._files = {}
        self._cache = {}
        self.L = int(self.hf["num_hidden_layers"])
        self.E = int(self.hf["num_experts"])
        self.meta = {"general.architecture": "qwen4exp"}
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

    def _bf16(self, name):
        return self._file(name).get_bf16(name)

    def _shape(self, name):
        if self._headers is not None:
            return tuple(self._headers[name]["shape"])
        return self._file(name).shape(name)

    # ---- the choices ----

    # ---- the map of the names ----

    def _add(self, gname, kind, src, dims, type_):
        self._map[gname] = (kind, src)
        self.tensors[gname] = (tuple(dims), type_, 0)

    def _dense_mat(self, gname, hf):
        rows, cols = self._shape(hf)
        self._add(gname, "dense", hf, (cols, rows), KQ_BF16)

    def _f32(self, gname, hf, fn=None, dims=None):
        shape = self._shape(hf)
        self._add(gname, "f32", (hf, fn), dims or tuple(reversed(shape)), KQ_F32)

    def _layer(self, i, base, mtp=False):
        b = "blk.%d." % i
        one = lambda a: a + 1.0  # noqa: E731
        for hc, g in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            p = base + hc + "."
            self._f32(b + g + "_norm.weight", p + "hc_norm.weight", one)
            self._dense_mat(b + g + "_down.weight", p + "input_mix_weight_down.weight")
            self._dense_mat(b + g + "_up.weight", p + "input_mix_weight_up.weight")
            self._f32(b + g + "_inject.weight", p + "block_inject_weight.weight")
        m = base + "mlp."
        self._f32(b + "ffn_gate_inp.weight", m + "gate.weight")
        self._f32(b + "ffn_gate_inp_shexp.weight", m + "shared_expert_gate.weight",
                  dims=(self._shape(m + "shared_expert_gate.weight")[-1],))
        for x in ("gate", "up", "down"):
            self._dense_mat(b + "ffn_%s_shexp.weight" % x, m + "shared_expert.%s_proj.weight" % x)
            e0 = m + "experts.0.%s_proj.weight" % x
            if mtp:
                rows, cols = self._shape(e0)
                self._add(b + "ffn_%s_exps.weight" % x, "fp8exps", (m, x), (cols, rows, self.E),
                          KQ_Q8_0)
            else:
                rows, half = self._shape(e0)
                self._add(b + "ffn_%s_exps.weight" % x, "nv4exps", (m, x), (half * 2, rows, self.E),
                          KQ_NV4)
        a = base + "self_attn."
        if a + "q_proj.weight" in self.where:
            for x, g in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output")):
                self._dense_mat(b + g + ".weight", a + "%s_proj.weight" % x)
            self._f32(b + "attn_q_norm.weight", a + "q_norm.weight", one)
            self._f32(b + "attn_k_norm.weight", a + "k_norm.weight", one)
            n_q = int(self.hf["indexer_n_heads"]) * int(self.hf["indexer_head_dim"])
            qk = a + "indexer.index_qk_proj.weight"
            rows, cols = self._shape(qk)
            self._add(b + "indexer.q_proj.weight", "f32", (qk, lambda w: w[:n_q]), (cols, n_q), KQ_F32)
            self._add(b + "indexer.k_proj.weight", "f32", (qk, lambda w: w[n_q:]), (cols, rows - n_q),
                      KQ_F32)
            self._f32(b + "indexer.q_norm.weight", a + "indexer.q_layernorm.weight", one)
            self._f32(b + "indexer.k_norm.weight", a + "indexer.k_layernorm.weight", one)
        la = base + "linear_attn."
        if la + "in_proj_qkv.weight" in self.where:
            self._dense_mat(b + "attn_qkv.weight", la + "in_proj_qkv.weight")
            self._dense_mat(b + "attn_gate.weight", la + "in_proj_z.weight")
            self._dense_mat(b + "ssm_out.weight", la + "out_proj.weight")
            self._f32(b + "ssm_alpha.weight", la + "in_proj_a.weight")
            self._f32(b + "ssm_beta.weight", la + "in_proj_b.weight")
            self._f32(b + "ssm_norm.weight", la + "norm.weight")
            self._f32(b + "ssm_a", la + "A_log", lambda a_: -np.exp(a_))
            self._f32(b + "ssm_dt.bias", la + "dt_bias")
            conv = la + "conv1d.weight"
            ch, _one, k = self._shape(conv)
            self._f32(b + "ssm_conv1d.weight", conv, lambda w: w.reshape(ch, k), dims=(k, ch))
        pl = base + "ple."
        if pl + "key_proj.weight" in self.where:
            self._dense_mat(b + "ple_key.weight", pl + "key_proj.weight")
            self._dense_mat(b + "ple_value.weight", pl + "value_proj.weight")
            for x in ("key", "query", "conv"):
                self._f32(b + "ple_norm_%s.weight" % x, pl + "norm_%s.weight" % x, one)
            conv = pl + "conv1d.weight"
            ch, _one, k = self._shape(conv)
            self._f32(b + "ple_conv1d.weight", conv, lambda w: w.reshape(ch, k), dims=(k, ch))

    def _build(self):
        for i in range(self.L):
            self._layer(i, MAIN + "layers.%d." % i)
        self._add("token_embd.weight", "bf16", MAIN + "embed_tokens.weight",
                  tuple(reversed(self._shape(MAIN + "embed_tokens.weight"))), KQ_BF16)
        self._dense_mat("output.weight", "lm_head.weight")
        self._f32("output_hc_norm.weight", MAIN + "hyper_connection_mixer.hc_norm.weight",
                  lambda a: a + 1.0)
        self._dense_mat("output_hc_down.weight", MAIN + "hyper_connection_mixer.input_mix_weight_down.weight")
        self._dense_mat("output_hc_up.weight", MAIN + "hyper_connection_mixer.input_mix_weight_up.weight")
        # The n-gram table: rows from the shards.
        shards = sorted((n for n in self.where if ".ngram_embedding.shard_" in n),
                        key=lambda n: int(n.rpartition(".shard_")[2].partition(".")[0]))
        if shards:
            self._ple_shards = shards
            sizes = [self._shape(n)[0] for n in shards]
            self._ple_start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            row = self._shape(shards[0])[1]
            self._add("per_layer_token_embd.weight", "ple", None, (row, int(self._ple_start[-1])),
                      KQ_F32)
        # The MTP layer, as layer L (the names of llama.cpp).
        if "mtp.fc_embedding.weight" in self.where:
            L = self.L
            self._layer(L, "mtp.layers.0.", mtp=True)
            b = "blk.%d.nextn." % L
            rows, cols = self._shape("mtp.fc_embedding.weight")
            self._add(b + "eh_proj.weight", "ehproj", None, (2 * cols, rows), KQ_BF16)
            self._f32(b + "enorm.weight", "mtp.pre_fc_norm_embedding.weight", lambda a: a + 1.0)
            self._f32(b + "hnorm.weight", "mtp.pre_fc_norm_hidden.weight", lambda a: a + 1.0)
            self._f32(b + "hc_head_norm.weight", "mtp.hyper_connection_mixer.hc_norm.weight",
                      lambda a: a + 1.0)
            self._dense_mat(b + "hc_head_down.weight", "mtp.hyper_connection_mixer.input_mix_weight_down.weight")
            self._dense_mat(b + "hc_head_up.weight", "mtp.hyper_connection_mixer.input_mix_weight_up.weight")

    # ---- the config ----

    def config(self):
        """The QwenConfig of the model, with the fields of qwen4.config_from_gguf."""
        from .qwen import QwenConfig
        h = self.hf
        cfg = QwenConfig.__new__(QwenConfig)
        cfg.raw = h
        n = self.L
        cfg.hidden_size = int(h["hidden_size"])
        cfg.num_hidden_layers = n
        cfg.layer_types = list(h["layer_types"])
        cfg.rms_norm_eps = float(h["rms_norm_eps"])
        cfg.num_heads = int(h["num_attention_heads"])
        cfg.num_kv_heads = int(h["num_key_value_heads"])
        cfg.head_dim = int(h["head_dim"])
        rp = h.get("rope_parameters", {})
        cfg.rope_theta = float(rp.get("rope_theta", h.get("rope_theta", 1e7)))
        cfg.rotary_dim = int(cfg.head_dim * float(rp.get("partial_rotary_factor",
                                                         h.get("partial_rotary_factor", 0.25))))
        cfg.lin_k_heads = int(h["linear_num_key_heads"])
        cfg.lin_v_heads = int(h["linear_num_value_heads"])
        cfg.lin_k_dim = int(h["linear_key_head_dim"])
        cfg.lin_v_dim = int(h["linear_value_head_dim"])
        cfg.conv_kernel = int(h["linear_conv_kernel_dim"])
        cfg.num_experts = self.E
        cfg.top_k = int(h["num_experts_per_tok"])
        cfg.moe_inter = int(h["moe_intermediate_size"])
        cfg.shared_inter = int(h["shared_expert_intermediate_size"])
        cfg.vocab_size = int(h["vocab_size"])
        cfg.eos_token_ids = None
        cfg.v_tiled = False             # the order of the checkpoint
        cfg.kv_form = os.environ.get("NP_GEMMA_QWEN_KV", "int16")
        cfg.hc_count = int(h["hc_count"])
        cfg.hc_lowrank = int(h["hc_lowrank"])
        cfg.ple_layers = [int(x) - 1 for x in h.get("ple_layer_ids", [])]
        cfg.ple_ngram = int(h["ngram_size"])
        cfg.ple_heads_per_ngram = int(h["heads_per_ngram"])
        cfg.ple_conv_kernel = int(h["ple_conv_kernel_size"])
        eos = h.get("eos_token_id")
        cfg.ple_eos = int(eos[-1] if isinstance(eos, list) else eos)
        if cfg.ple_layers:
            pl = MAIN + "layers.%d.ple.ple_embedding." % cfg.ple_layers[0]
            ints = lambda s: [int(x) for x in self._get(pl + s, dtype=None).reshape(-1)]  # noqa: E731
            cfg.ple_multipliers = ints("layer_multipliers")
            cfg.ple_offsets = ints("ngram_heads_offsets")
            cfg.ple_sizes = ints("ngram_heads_vocab_sizes")
            cfg.ple_row = int(self.tensors["per_layer_token_embd.weight"][0][0])
        cfg.indexer_heads = int(h["indexer_n_heads"])
        cfg.indexer_dim = int(h["indexer_head_dim"])
        cfg.indexer_top_k = int(h["indexer_budget"])
        ratio = int(h["indexer_compress_ratio"])
        cfg.compress_ratios = [ratio if t == "full_attention" else 0 for t in cfg.layer_types] + [0]
        cfg.lin_gate = h.get("output_gate_type", "sigmoid")
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
            return self._bf16(src)
        if kind == "dense":
            return self._bf16(src)
        if kind == "ehproj":
            e, hd = self._get("mtp.fc_embedding.weight"), self._get("mtp.fc_hidden.weight")
            w = np.concatenate([e, hd], axis=1)           # embedding first (the graph of llama.cpp)
            return (np.ascontiguousarray(w, np.float32).view(np.uint32) >> 16).astype(np.uint16)
        if kind == "nv4exps":
            m, x = src
            p = m + "experts.%d.%s_proj." % (0, x)
            rows, half = self._shape(p + "weight")
            cols = 2 * half
            ws, ss, gs = [], [], []
            for e in range(self.E):
                p = m + "experts.%d.%s_proj." % (e, x)
                ws.append(self._get(p + "weight", dtype=None))
                ss.append(self._get(p + "weight_scale", dtype=None))
                gs.append(float(np.asarray(self._get(p + "weight_scale_2")).reshape(-1)[0]))
            out = np.empty(self.E * rows * cops.kq_nv4_row_bytes(cols), np.uint8)
            cops.kq_nv4_pack(ws, ss, gs, rows, cols, out)
            return out
        if kind == "fp8exps":
            m, x = src
            parts = []
            for e in range(self.E):
                p = m + "experts.%d.%s_proj." % (e, x)
                w = E4M3[self._get(p + "weight", dtype=None)]
                s = self._get(p + "weight_scale_inv")
                rows, cols = w.shape
                br, bc = -(-rows // s.shape[0]), -(-cols // s.shape[1])
                w = w * np.repeat(np.repeat(s, br, axis=0), bc, axis=1)[:rows, :cols]
                parts.append(cops.kq_to_q8_0(np.ascontiguousarray(w, np.float32), cols))
            return np.concatenate(parts)
        raise KeyError(gname)

    def raw(self, gname):
        """(the data, dims in the ggml order, the type), as GGUF.raw."""
        a = self._cache.get(gname)
        if a is None:
            if self._map[gname][0] == "ple":
                raise ValueError("the n-gram table has rows only (dequant, ple_rows)")
            a = self._cache[gname] = self._make(gname)
        dims, type_, _ = self.tensors[gname]
        return a, dims, type_

    def ple_rows(self, rows):
        """The float32 rows of the n-gram table."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        n = self._ple_shards
        scale = self._ple_scale()
        k = np.searchsorted(self._ple_start, rows, side="right") - 1
        width = self.tensors["per_layer_token_embd.weight"][0][0]
        base = self._cache.get("_ple_base")
        if base is None:
            base = self._cache["_ple_base"] = np.array(
                [self._get(x, dtype=None).ctypes.data for x in n], dtype=np.int64)
            self._ple_random()
        # The address of each row in the maps; many threads read them (the
        # rows are random, and most are not in the page cache).
        addrs = base[k] + (rows - self._ple_start[k]) * width
        raw = cops.kq_gather(addrs, width)
        return E4M3[raw] * scale

    def _ple_random(self):
        """No read-ahead on the bytes of the n-gram table: a page fault of a
        random row then reads one page, not a large window."""
        import mmap
        for f in {self.where[x] for x in self._ple_shards}:
            st = self._files[f]
            spans = [st.header[x]["data_offsets"] for x in self._ple_shards if self.where[x] == f]
            lo = st._base + min(a for a, _b in spans)
            hi = st._base + max(b for _a, b in spans)
            lo -= lo % mmap.PAGESIZE
            for adv in ("MADV_NOHUGEPAGE", "MADV_RANDOM"):    # no 2 MB reads for a row
                try:
                    st._mm.madvise(getattr(mmap, adv), lo, hi - lo)
                except (AttributeError, OSError, ValueError):
                    pass

    def _ple_scale(self):
        v = self._cache.get("_ple_scale")
        if v is None:
            name = self._ple_shards[0].split(".shard_")[0] + ".weight_scale"
            v = self._cache["_ple_scale"] = float(self._get(name).reshape(-1)[0])
        return v

    def dequant(self, gname, rows=None):
        """float32 values in the NumPy shape (rows selects the first index),
        as GGUF.dequant."""
        dims, type_, _ = self.tensors[gname]
        shape = tuple(reversed(dims))
        if self._map[gname][0] == "ple":
            return self.ple_rows(np.arange(shape[0]) if rows is None else rows)
        a, _dims, _t = self.raw(gname)
        if type_ == KQ_F32:
            a = a.reshape(shape)
            return a if rows is None else a[np.asarray(rows)]
        cols = dims[0]
        rb = cops_row_bytes(type_, cols)
        mat = a.reshape(-1).view(np.uint8)
        per = int(np.prod(shape[1:-1])) if len(shape) > 2 else 1   # the rows of one index
        idx = np.arange(shape[0]) if rows is None else np.asarray(rows).reshape(-1)
        ids = (idx[:, None] * per + np.arange(per)[None, :]).reshape(-1)
        out = cops.kq_rows(mat, type_, cols, ids)
        return out.reshape((len(idx),) + shape[1:])

    def file_bytes(self):
        """The bytes of the safetensors files."""
        return sum(os.path.getsize(os.path.join(self.path, f)) for f in set(self.where.values()))

    def close(self):
        for f in self._files.values():
            f.close()


def cops_row_bytes(type_, cols):
    return {KQ_F32: 4 * cols, KQ_Q8_0: cols // 32 * 34, KQ_BF16: 2 * cols,
            KQ_NV4: cops.kq_nv4_row_bytes(cols)}[type_]
