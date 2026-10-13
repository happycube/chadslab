"""The original bfloat16 checkpoint of Qwen3.8-Flash-Next (Qwen/Qwen3.8-Flash-Next,
"ORIG") in the GGUF view of the runtime: OrigSource (the map of NVFP4Source
with the fused bfloat16 experts and the bfloat16 n-gram table), the order of
the value heads of the DeltaNet (tiled), the GGUF metadata. Shared by
scripts/convert_q8_gguf.py (a GGUF file) and the weightkeeper
(reclank/runtime/weightkeeper.py: the tensors in memory)."""
from __future__ import annotations

import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.formats.gguf import Q8_0, RQ6_K, RQ8_0
from reclank.formats.nvfp4_st import NVFP4Source


def reorder(a, dim, hk, rep, hd):
    """The heads of axis dim from grouped by key head to the tiled order of
    llama.cpp (_reorder_v_heads of its converter)."""
    a = np.moveaxis(a, dim, 0)
    sh = a.shape
    a = a.reshape(hk, rep, hd, *sh[1:]).swapaxes(0, 1).reshape(sh)
    return np.ascontiguousarray(np.moveaxis(a, 0, dim))


def metadata(cfg, src):
    a = "qwen4exp."
    m = [("general.architecture", "qwen4exp"),
         ("general.name", "Qwen3.8-Flash-Next NVFP4 (reclank)"),
         ("general.source", "nvidia/Qwen3.8-Flash-Next-NVFP4"),
         (a + "block_count", cfg.num_hidden_layers),
         (a + "nextn_predict_layers", 1 if "blk.%d.nextn.eh_proj.weight" % cfg.num_hidden_layers
          in src.tensors else 0),
         (a + "full_attention_interval", int(src.hf.get("full_attention_interval", 4))),
         (a + "embedding_length", cfg.hidden_size),
         (a + "attention.layer_norm_rms_epsilon", float(cfg.rms_norm_eps)),
         (a + "attention.head_count", cfg.num_heads),
         (a + "attention.head_count_kv", cfg.num_kv_heads),
         (a + "attention.key_length", cfg.head_dim),
         (a + "attention.value_length", cfg.head_dim),
         (a + "rope.freq_base", float(cfg.rope_theta)),
         (a + "rope.dimension_count", cfg.rotary_dim),
         (a + "ssm.group_count", cfg.lin_k_heads),
         (a + "ssm.time_step_rank", cfg.lin_v_heads),
         (a + "ssm.state_size", cfg.lin_k_dim),
         (a + "ssm.inner_size", cfg.lin_v_heads * cfg.lin_v_dim),
         (a + "ssm.conv_kernel", cfg.conv_kernel),
         (a + "expert_count", cfg.num_experts),
         (a + "expert_used_count", cfg.top_k),
         (a + "expert_feed_forward_length", cfg.moe_inter),
         (a + "expert_shared_feed_forward_length", cfg.shared_inter),
         (a + "hyper_connection.count", cfg.hc_count),
         (a + "hyper_connection.low_rank", cfg.hc_lowrank),
         (a + "ple.layers", list(cfg.ple_layers)),
         (a + "ple.ngram_size", cfg.ple_ngram),
         (a + "ple.heads_per_ngram", cfg.ple_heads_per_ngram),
         (a + "ple.conv_kernel", cfg.ple_conv_kernel),
         (a + "ple.eos_token_id", cfg.ple_eos),
         (a + "embedding_length_per_layer_input", cfg.ple_row),
         (a + "ple.layer_multipliers", list(cfg.ple_multipliers)),
         (a + "ple.head_offsets", list(cfg.ple_offsets)),
         (a + "ple.head_vocab_sizes", list(cfg.ple_sizes)),
         (a + "attention.indexer.head_count", cfg.indexer_heads),
         (a + "attention.indexer.key_length", cfg.indexer_dim),
         (a + "attention.indexer.top_k", cfg.indexer_top_k),
         (a + "attention.compress_ratios", list(cfg.compress_ratios))]
    return m


def tiler(cfg):
    """The function (gname, array) -> the array with the value heads of a
    DeltaNet tensor in the tiled order of llama.cpp (the others as they are)."""
    hk, hv, dv = cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_v_dim
    rep = hv // hk
    qk = 2 * hk * cfg.lin_k_dim

    def tiled(gname, a):
        if gname.endswith(("attn_qkv.weight", "ssm_conv1d.weight")):
            return np.concatenate([a[:qk], reorder(a[qk:], 0, hk, rep, dv)])
        if gname.endswith("attn_gate.weight"):
            return reorder(a, 0, hk, rep, dv)
        if gname.endswith(("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias")):
            return reorder(a, 0, hk, rep, 1)
        if gname.endswith("ssm_out.weight"):
            return reorder(a, 1, hk, rep, dv)
        return a
    return tiled


def bf16f(a):
    return (np.asarray(a, np.uint16).astype(np.uint32) << 16).view(np.float32)


def to_q8(w, rot):
    """Q8_0 blocks of bfloat16 rows (uint16), or RQ8_0 (rot: the rows rotated
    in each 32 values first)."""
    if not rot:
        return cops.kq_to_q8_0(w, w.shape[-1])
    return cops.kq_to_q8_0(cops.tq6_rotate(bf16f(w)), w.shape[-1])


def to_type(w, t):
    """The blocks of bfloat16 rows in type t: Q8_0, RQ8_0, or RQ6_K (Q6_K of
    the rotated rows, as llama.cpp quantizes Q6_K)."""
    if t == RQ6_K:
        return cops.kq_to_q6_k(cops.tq6_rotate(bf16f(w)), w.shape[-1])
    return to_q8(w, t == RQ8_0)


def back_type(q, cols, t):
    """The values of blocks of rows of cols values in type t."""
    kt = {RQ6_K: 14}.get(t, Q8_0)
    rb = cols // 256 * 210 if kt == 14 else cols // 32 * 34
    v = cops.kq_rows(q, kt, cols, np.arange(q.size // rb))
    return cops.tq6_rotate(v, inverse=True) if t in (RQ8_0, RQ6_K) else v


def ud_map(path, L):
    """The types of the routed experts of RQ6_MIX_PLAN.md from the choices of
    a GGUF of Unsloth (UD-Q4_K_XL): {(layer, "gate"|"up"|"down"): type}.
    Where it paid for Q8_0, RQ8_0; else RQ6_K if the rows are whole blocks
    of 256 (gate and up: 2560 inputs), else RQ8_0 (down: 640 inputs, M1)."""
    from reclank.formats.gguf import open_gguf
    g = open_gguf(path)
    out = {}
    for i in range(L):
        for x in ("gate", "up", "down"):
            dims, t, _o = g.tensors["blk.%d.ffn_%s_exps.weight" % (i, x)]
            out[(i, x)] = RQ8_0 if t == Q8_0 or int(dims[0]) % 256 else RQ6_K
    g.close()
    return out


class OrigSource(NVFP4Source):
    """The GGUF view of ORIG: the map of NVFP4Source, with the fused bfloat16
    experts and the bfloat16 n-gram table in Q8_0. rot: the experts (and the
    shared experts) in RQ8_0."""

    def __init__(self, path, headers=None, rot=False, xtypes=None):
        """xtypes: {(layer, "gate"|"up"|"down"): type} of the routed experts
        (ud_map; the others RQ8_0 with rot, else Q8_0)."""
        self.rot = rot
        self.xtypes = xtypes or {}
        super().__init__(path, headers=headers)

    def _experts(self, b, m, mtp):
        gu, dn = m + "experts.gate_up_proj", m + "experts.down_proj"
        E, rows2, cols = self._shape(gu)
        inter = rows2 // 2
        assert E == self.E and tuple(self._shape(dn)) == (E, cols, inter), (gu, self._shape(dn))
        i = int(b.split(".")[1])
        t0 = RQ8_0 if self.rot else Q8_0
        tg, tu, td = (self.xtypes.get((i, x), t0) for x in ("gate", "up", "down"))
        self._add(b + "ffn_gate_exps.weight", "q8exps", (gu, 0, inter, tg), (cols, inter, E), tg)
        self._add(b + "ffn_up_exps.weight", "q8exps", (gu, inter, rows2, tu), (cols, inter, E), tu)
        self._add(b + "ffn_down_exps.weight", "q8exps", (dn, 0, cols, td), (inter, cols, E), td)

    def _build(self):
        super()._build()
        g = "per_layer_token_embd.weight"
        dims, _t, _ = self.tensors[g]
        self._add(g, "q8ple", None, dims, Q8_0)
        if self.rot:
            # the shared experts too: the MoE then rotates its x for all its
            # experts
            for n, (kind, hf) in list(self._map.items()):
                if kind == "dense" and n.endswith(("ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                                   "ffn_down_shexp.weight")):
                    self._add(n, "rq8dense", hf, self.tensors[n][0], RQ8_0)

    def _make(self, gname):
        kind, src = self._map[gname]
        if kind == "q8exps":
            hf, lo, hi, t = src
            return self._q8_experts(hf, lo, hi, t)
        if kind == "q8ple":
            return self._q8_ple()
        if kind == "rq8dense":
            return to_q8(np.asarray(self._bf16(src)), True)
        return super()._make(gname)

    def _q8_experts(self, hf, lo, hi, t):
        """Rows lo to hi of each expert of a fused stack, one expert at a
        time, in type t."""
        w = self._bf16(hf)
        for e in range(self.E):
            yield to_type(w[e, lo:hi], t)

    def _q8_ple(self):
        for n in self._ple_shards:
            w = self._bf16(n)
            yield cops.kq_to_q8_0(w, w.shape[1])

    def sources(self, gname):
        """The safetensors names that gname reads."""
        kind, src = self._map[gname]
        if kind == "f32":
            return [src[0]]
        if kind in ("bf16", "dense", "rq8dense"):
            return [src]
        if kind == "ehproj":
            return ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"]
        if kind == "q8exps":
            return [src[0]]  # (hf, lo, hi, type)
        if kind == "q8ple":
            return list(self._ple_shards)
        raise KeyError(gname)

    def span(self, hf):
        """(path, start, end) of the bytes of a safetensors tensor."""
        st = self._file(hf)
        a, b = st.header[hf]["data_offsets"]
        return os.path.join(self.path, self.where[hf]), st._base + a, st._base + b
