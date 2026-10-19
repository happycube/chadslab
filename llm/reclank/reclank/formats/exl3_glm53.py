"""GLM-5.3-Flash from an EXL3 directory of ExLlamaV3 (turboderp/GLM-5.3-Flash-exl3, e.g. 4.05bpw;
docs/GLM53_PLAN.md).

GLM53Source is the GGUF view of the directory (the interface of EXL3Source: tensors, raw,
dequant; GGUF names, ggml order of dims) on its tensors (reclank/formats/exl3.py, EXL3Dir). Nothing
is converted beyond the forms of the products:

- the EXL3 matrices (MUL1) as they are: the blocks of the products, type KQ_E3_BASE + K;
- the routed experts: one stack of the blocks of the 288 experts for each layer and matrix (the
  form of kernels/kq/exl3_moe.c), on huge pages;
- the float16 tensors (kv_b, the router, the KDA low-rank pairs, b_proj, the indexer's wk, gate and
  weights): float16 (KQ_F16) or float32 as stored (they came from bfloat16);
- the norms: bfloat16 to float32, as they are (GLM's RMSNorm is w * x: no "+1"); A_log as it is
  (the KDA decay is -5 sigmoid(exp(A_log) (f + dt_bias)), not -exp(A_log) softplus);
- kv_b_proj [64 x (256 k + 256 v), 512]: split into attn_k_b and attn_v_b, [64, 256, 512] each
  (head h: rows 512 h .. 512 h + 255 the keys, the next 256 the values);
- the embeddings: bfloat16, rows on demand.

The names (blk.<i>.):

    hc_{attn,ffn}_fn [24, 16384], _base [24], _scale [3]     the hyper-connections (f32)
    attn_norm, ffn_norm                                      input / post-attention RMSNorm
    KDA (34 layers):  attn_qkv (e3: q | k | v, 3 x 8192), ssm_conv1d [24576, 4], ssm_f_a [128, 4096],
                      ssm_f_b [8192, 128], ssm_dt.bias [8192], ssm_a_log [64], ssm_beta [64, 4096],
                      ssm_g_a [128, 4096], ssm_g_b [8192, 128], ssm_norm [128], attn_output (e3)
    DSA (11 layers):  attn_q_a (e3), attn_q_a_norm, attn_q_b (e3), attn_kv_a_mqa (e3), attn_kv_a_norm,
                      attn_k_b, attn_v_b, attn_output (e3); the indexer: idx_q_b (e3), idx_k [128, 4096],
                      idx_k_norm.weight / .bias [128], idx_w [32, 4096], idx_gate [128, 4096], idx_ape [4, 128]
    dense MLP (0-2):  ffn_gate, ffn_up, ffn_down (e3)
    MoE (3-44):       ffn_gate_inp [288, 4096], exp_probs_b.bias [288], ffn_{gate,up,down}_exps (stacks),
                      ffn_{gate,up,down}_shexp (e3)

and token_embd, output (e3), output_norm. The MTP layer (mtp.safetensors, outside the index) is
layer 45: a DSA + MoE layer without hyper-connections, and nextn.eh_proj (e3: [embedding |
hidden] 8192 -> 4096), nextn.enorm, nextn.hnorm, nextn.shared_head_norm.

    src = GLM53Source("/home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw")
    src.config()                         # GLM53Config
    src.raw("blk.3.ffn_up_exps.weight")  # (bytes, dims, type), as GGUF.raw
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from reclank.formats.exl3 import KQ_E3_BASE, EXL3Dir

KQ_F32, KQ_F16, KQ_BF16 = 0, 1, 30
MAIN = "model.language_model."


@dataclass
class GLM53Config:
    """The fields of the text config that the model reads."""
    hidden_size: int = 4096
    num_hidden_layers: int = 45
    vocab_size: int = 154880
    rms_norm_eps: float = 1e-5
    layer_types: list = field(default_factory=list)      # "kda" / "dsa"
    mlp_types: list = field(default_factory=list)        # "dense" / "moe"
    # KDA
    kda_heads: int = 64
    kda_dim: int = 128
    conv_kernel: int = 4
    gate_lower_bound: float = -5.0
    # MLA (no RoPE)
    num_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 256
    v_head_dim: int = 256
    # the indexer
    index_heads: int = 32
    index_dim: int = 128
    index_topk: int = 2048
    index_pool: int = 4
    index_tail: bool = True
    # mHC
    hc_mult: int = 4
    hc_iters: int = 20
    hc_eps: float = 1e-6
    # MoE
    num_experts: int = 288
    top_k: int = 8
    moe_inter: int = 2048
    dense_inter: int = 12288
    routed_scale: float = 2.5
    swiglu_limit: float = 10.0
    eos_token_ids: list = field(default_factory=list)
    mtp_layers: int = 0


class GLM53Source:
    """The GGUF view of an EXL3 directory of GLM-5.3-Flash (see the module text)."""

    exl3 = True
    staged = True       # (the stacks of the experts are copies in memory: QwenGPU page-locks them)

    def __init__(self, path, mtp=True, layers=None):
        """mtp False: no layer 45 (the GPU model without the MTP layer: its experts are not
        copied, nor split into parts). layers (a test): the tensors of the first layers only."""
        self.path = path
        self.mtp = mtp
        self.layers = layers
        self.d = EXL3Dir(path)
        indexed = set(self.d.where.values())
        for f in sorted(os.listdir(path)):
            if f.endswith(".safetensors") and f not in indexed:     # mtp.safetensors
                for name in self.d._st(f).header:
                    self.d.where.setdefault(name, f)
        self.hf = self.d.text_config()
        self.L = int(self.hf["num_hidden_layers"])
        self.E = int(self.hf["n_routed_experts"])
        self.meta = dict(self.d.meta, **{"general.architecture": "glm53"})
        self.tensors = {}           # GGUF name -> (dims (ggml order), type, 0)
        self._map = {}              # GGUF name -> (kind, source)
        self._cache = {}
        self._build()

    # ---- the map of the names ----

    def _add(self, gname, kind, src, dims, type_):
        self._map[gname] = (kind, src)
        self.tensors[gname] = (tuple(int(x) for x in dims), type_, 0)

    def _shape(self, name):
        return tuple(self.d.shape(name))

    def _e3(self, gname, key):
        m = self.d.e3(key)
        self._add(gname, "e3", key, (m.k, m.n), KQ_E3_BASE + m.K)

    def _plain(self, gname, name, fn=None, dims=None):
        """A tensor as stored: float16 stays float16 (KQ_F16), the rest float32 (fn: a reshape)."""
        shape = self._shape(name)
        t = KQ_F16 if self.d.dtype(name) == "F16" and fn is None else KQ_F32
        self._add(gname, "plain", (name, fn), dims or tuple(reversed(shape)), t)

    def _layer(self, i, base, mtp=False):
        b = "blk.%d." % i
        if not mtp:
            for g in ("hc_attn", "hc_ffn"):
                for x in ("fn", "base", "scale"):
                    self._plain(b + "%s_%s" % (g, x), base + "%s_%s" % (g, x))
        self._plain(b + "attn_norm.weight", base + "input_layernorm.weight")
        self._plain(b + "ffn_norm.weight", base + "post_attention_layernorm.weight")
        a = base + "self_attn."
        if self.d.is_e3(a + "qkv_proj"):
            self._e3(b + "attn_qkv.weight", a + "qkv_proj")
            conv = a + "conv1d.weight"
            ch, _one, k = self._shape(conv)
            self._plain(b + "ssm_conv1d.weight", conv, lambda w: w.reshape(ch, k), (k, ch))
            for x, g in (("f_a_proj.weight", "ssm_f_a.weight"), ("f_b_proj.weight", "ssm_f_b.weight"),
                         ("dt_bias", "ssm_dt.bias"), ("A_log", "ssm_a_log"), ("b_proj.weight", "ssm_beta.weight"),
                         ("g_a_proj.weight", "ssm_g_a.weight"), ("g_b_proj.weight", "ssm_g_b.weight"),
                         ("o_norm.weight", "ssm_norm.weight")):
                self._plain(b + g, a + x)
        else:
            self._e3(b + "attn_q_a.weight", a + "q_a_proj")
            self._plain(b + "attn_q_a_norm.weight", a + "q_a_layernorm.weight")
            self._e3(b + "attn_q_b.weight", a + "q_b_proj")
            self._e3(b + "attn_kv_a_mqa.weight", a + "kv_a_proj_with_mqa")
            self._plain(b + "attn_kv_a_norm.weight", a + "kv_a_layernorm.weight")
            kvb = a + "kv_b_proj.weight"
            rows, lat = self._shape(kvb)
            H = int(self.hf["num_attention_heads"])
            dk, dv = int(self.hf["qk_nope_head_dim"]), int(self.hf["v_head_dim"])
            assert rows == H * (dk + dv), (kvb, rows)
            self._add(b + "attn_k_b.weight", "kvb", (kvb, 0, dk), (lat, dk, H), KQ_F16)
            self._add(b + "attn_v_b.weight", "kvb", (kvb, dk, dv), (lat, dv, H), KQ_F16)
            x = a + "indexer."
            self._e3(b + "idx_q_b.weight", x + "wq_b")
            for s, g in (("wk.weight", "idx_k.weight"), ("k_norm.weight", "idx_k_norm.weight"),
                         ("k_norm.bias", "idx_k_norm.bias"), ("weights_proj.weight", "idx_w.weight"),
                         ("index_kpool_compress_gate", "idx_gate.weight"), ("index_kpool_compress_ape", "idx_ape")):
                self._plain(b + g, x + s)
        self._e3(b + "attn_output.weight", a + "o_proj")
        m = base + "mlp."
        if self.d.is_e3(m + "gate_proj"):
            for x in ("gate", "up", "down"):
                self._e3(b + "ffn_%s.weight" % x, m + "%s_proj" % x)
        else:
            self._plain(b + "ffn_gate_inp.weight", m + "gate.weight")
            self._plain(b + "exp_probs_b.bias", m + "gate.e_score_correction_bias")
            for x in ("gate", "up", "down"):
                self._e3(b + "ffn_%s_shexp.weight" % x, m + "shared_experts.%s_proj" % x)
                e0 = self.d.e3(m + "experts.0.%s_proj" % x)
                self._add(b + "ffn_%s_exps.weight" % x, "e3exps", (m, x), (e0.k, e0.n, self.E), KQ_E3_BASE + e0.K)
        if mtp:
            n = b + "nextn."
            self._e3(n + "eh_proj.weight", base + "eh_proj")
            self._plain(n + "enorm.weight", base + "enorm.weight")
            self._plain(n + "hnorm.weight", base + "hnorm.weight")
            self._plain(n + "shared_head_norm.weight", base + "shared_head.norm.weight")

    def _build(self):
        for i in range(self.layers or self.L):
            self._layer(i, MAIN + "layers.%d." % i)
        mtp = MAIN + "layers.%d." % self.L
        if self.mtp and not self.layers and self.d.is_e3(mtp + "eh_proj"):
            self._layer(self.L, mtp, mtp=True)
        emb = MAIN + "embed_tokens.weight"
        self._add("token_embd.weight", "bf16", emb, tuple(reversed(self._shape(emb))), KQ_BF16)
        self._e3("output.weight", "lm_head")
        self._plain("output_norm.weight", MAIN + "norm.weight")

    # ---- the config ----

    def config(self):
        h = self.hf
        la = h["linear_attn_config"]
        c = GLM53Config()
        c.hidden_size = int(h["hidden_size"])
        c.num_hidden_layers = self.L
        c.vocab_size = int(h["vocab_size"])
        c.rms_norm_eps = float(h["rms_norm_eps"])
        c.layer_types = ["kda" if t == "linear_attention" else "dsa" for t in h["layer_types"]]
        c.mlp_types = ["moe" if t == "sparse" else "dense" for t in h["mlp_layer_types"]]
        assert c.layer_types.count("kda") == len(la["kda_layers"])
        c.kda_heads = int(la["num_heads"])
        c.kda_dim = int(la["head_dim"])
        c.conv_kernel = int(la["short_conv_kernel_size"])
        c.gate_lower_bound = float(la["gate_lower_bound"])
        c.num_heads = int(h["num_attention_heads"])
        c.q_lora_rank = int(h["q_lora_rank"])
        c.kv_lora_rank = int(h["kv_lora_rank"])
        c.qk_head_dim = int(h["qk_nope_head_dim"])
        assert int(h["qk_rope_head_dim"]) == 0 and h.get("mla_use_nope", True)
        c.v_head_dim = int(h["v_head_dim"])
        c.index_heads = int(h["index_n_heads"])
        c.index_dim = int(h["index_head_dim"])
        c.index_topk = int(h["index_topk"])
        c.index_pool = int(h["index_kpool"])
        c.index_tail = bool(h["index_kpool_always_select_tail"])
        assert set(h["indexer_types"]) == {"full"}, "a shared indexer layer (not in GLM-5.3-Flash)"
        c.hc_mult = int(h["hc_mult"])
        c.hc_iters = int(h["hc_sinkhorn_iters"])
        c.hc_eps = float(h["hc_eps"])
        c.num_experts = self.E
        c.top_k = int(h["num_experts_per_tok"])
        c.moe_inter = int(h["moe_intermediate_size"])
        c.dense_inter = int(h["intermediate_size"])
        c.routed_scale = float(h["routed_scaling_factor"])
        c.swiglu_limit = float(h["swiglu_limit"])
        assert h["scoring_func"] == "sigmoid" and h["norm_topk_prob"] and int(h["n_group"]) == 1
        assert int(h["n_shared_experts"]) == 1
        eos = h.get("eos_token_id")
        c.eos_token_ids = list(eos) if isinstance(eos, list) else [int(eos)]
        c.mtp_layers = 1 if "blk.%d.nextn.eh_proj.weight" % self.L in self.tensors else 0
        return c

    # ---- the tensors ----

    def _make(self, gname):
        kind, src = self._map[gname]
        if kind == "e3":
            return self.d.e3(src).block()
        if kind == "plain":
            name, fn = src
            if fn is None and self.d.dtype(name) == "F16":
                return np.ascontiguousarray(self.d.get(name, dtype=None))
            a = self.d.get(name, dtype=np.float32)
            return np.ascontiguousarray(fn(a) if fn is not None else a, dtype=np.float32)
        if kind == "bf16":
            return self.d.get_bf16(src)
        if kind == "kvb":
            name, off, n = src
            dims = self.tensors[gname][0]
            H, lat = dims[2], dims[0]
            w = np.asarray(self.d.get(name, dtype=None)).reshape(H, -1, lat)
            return np.ascontiguousarray(w[:, off:off + n])
        if kind == "e3exps":
            m, x = src
            blocks = [self.d.e3(m + "experts.%d.%s_proj" % (e, x)) for e in range(self.E)]
            K = blocks[0].K
            assert all(b.K == K for b in blocks), "%s: the experts of a matrix have one K" % gname
            from reclank.runtime import numa
            first = blocks[0].block()
            out = numa.empty_huge(first.nbytes * len(blocks)).reshape(len(blocks), -1)
            for e, b in enumerate(blocks):
                out[e] = b.block().reshape(-1).view(np.uint8)
                b._block = None         # (the stack holds the bytes)
            return out.reshape(-1).view(first.dtype)
        raise KeyError(gname)

    def raw(self, gname):
        """(the data, dims in the ggml order, the type), as GGUF.raw."""
        a = self._cache.get(gname)
        if a is None:
            a = self._cache[gname] = self._make(gname)
        dims, type_, _ = self.tensors[gname]
        return a, dims, type_

    def drop(self, gname):
        """Forget the copy of a tensor (the stacks of experts: once the model holds them)."""
        self._cache.pop(gname, None)

    def dequant(self, gname, rows=None):
        """float32 values in the NumPy shape (rows selects the first index), as GGUF.dequant."""
        dims, type_, _ = self.tensors[gname]
        kind, src = self._map[gname]
        shape = tuple(reversed(dims))
        if kind == "e3":
            w = self.d.e3(src).weights().T            # (n x k)
        elif kind == "e3exps":
            m, x = src
            idx = range(self.E) if rows is None else np.asarray(rows).reshape(-1)
            from reclank.formats.exl3 import EXL3Matrix
            return np.stack([EXL3Matrix(self.d, m + "experts.%d.%s_proj" % (int(e), x)).weights().T for e in idx])
        elif kind == "bf16":
            if rows is not None:
                return self.embed_rows(rows)
            w = self.d.get(src)
        else:
            w = np.asarray(self.raw(gname)[0], np.float32).reshape(shape)
        return w if rows is None else w[np.asarray(rows)]

    def expert_block(self, layer, e, x):
        """(the block, K) of matrix x ("gate", "up", "down") of one routed expert, read from the
        files and not kept (the NumPy model: the experts a step routes to)."""
        from reclank.formats.exl3 import EXL3Matrix
        m = EXL3Matrix(self.d, MAIN + "layers.%d.mlp.experts.%d.%s_proj" % (layer, e, x))
        return m.block(), m.K

    def embed_rows(self, ids):
        """The float32 rows of token_embd for token ids."""
        name = self._map["token_embd.weight"][1]
        return np.stack([self.d.get_row(name, int(t)) for t in np.asarray(ids).reshape(-1)])

    def file_bytes(self):
        return sum(os.path.getsize(os.path.join(self.path, f)) for f in set(self.d.where.values()))

    def close(self):
        self.d.close()


def is_glm53_exl3(path):
    """True for an EXL3 directory of GLM-5.3-Flash (glm5_next)."""
    from reclank.formats.exl3 import is_exl3
    if not is_exl3(path):
        return False
    c = json.load(open(os.path.join(path, "config.json")))
    return c.get("text_config", c).get("model_type", c.get("model_type", "")).startswith("glm5_next")


__all__ = ["GLM53Source", "GLM53Config", "is_glm53_exl3", "MAIN", "KQ_F16"]
