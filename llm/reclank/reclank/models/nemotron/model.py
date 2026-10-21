"""Nemotron 3.5 Bolt 74B-A9B (NemotronH Puzzle) and Super 120B-A12B (NemotronH, uniform layers)
on the CPU in NumPy (P2 of docs/NEMOTRON_BOLT_PLAN.md, S1 of docs/NEMOTRON_SUPER_PLAN.md): the
forward of the checkpoint's modeling_nemotron_h.py op for op, in float32, from the shipped tensors
decoded in NumPy (IQ2_XXS: reclank/formats/iq2.py; NVFP4, FP8 with a tensor scale and Q8_0 here).
The slow reference of the runtime: the engines' checks compare against it.

- each of the 88 blocks: x + mixer(rmsnorm(x));
- Mamba-2 (40): in_proj -> z, xBC, dt; a causal conv of 4 + bias + silu over xBC; 128 heads of
  64 with B and C of 8 groups (16 heads a group), state 128; dt = softplus(dt + dt_bias) (no
  floor: time_step_limit (0, inf), as the fast path), h = exp(dt A) h + dt x B^T, y = h C + D x (the recurrent form for the prompt too:
  transformers' chunked SSD is the same sums); the gated RMSNorm in 8 groups of 1024,
  rmsnorm(y silu(z)) w; out_proj;
- attention (8): GQA, 32 query heads, 2 KV heads of 128, no RoPE;
- latent MoE (40): sigmoid router + correction bias for the choice, 9 experts, normalized, x 5;
  fc1_latent 4096 -> 1024, the experts down(relu(up x)^2) in the latent, fc2_latent; the shared
  expert relu^2 in the full width, on the same normed input.

    m = NemotronCPU("/home/cpage/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB")
    cache = m.cache()
    logits = m.forward(ids, cache)              # [t, vocab]
    logits = m.forward([tok], cache)
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.formats import iq2
from reclank.formats.safetensors import SafeTensors

LM = "language_model."
BB = LM + "backbone."


class NemotronConfig:
    """The text model's config (config.json llm_config): the kind of each block in kinds[i]
    ("mamba", "attention", "moe") and moe[i] = (experts, inner, top), from the per-block configs
    of the Puzzle model (Bolt) or the uniform layers of NemotronH (Super); mtp: the blocks of the
    MTP layer (Super: attention, moe), else none."""

    def __init__(self, path):
        c = json.load(open(os.path.join(path, "config.json")))
        t = c.get("llm_config", c)
        self.raw = t
        self.hidden = t["hidden_size"]
        self.vocab = t["vocab_size"]
        self.eps = t["layer_norm_epsilon"]
        blocks = t.get("block_configs")
        if blocks:
            self.kinds = [b["block_type"] for b in blocks]
            self.moe = {i: (b["n_routed_experts"], b["moe_intermediate_size"], b["num_experts_per_tok"])
                        for i, b in enumerate(blocks) if b["block_type"] == "moe"}
        else:
            self.kinds = list(t["layers_block_type"])
            self.moe = {i: (t["n_routed_experts"], t["moe_intermediate_size"], t["num_experts_per_tok"])
                        for i, kind in enumerate(self.kinds) if kind == "moe"}
        self.layers = len(self.kinds)
        self.mtp = list(t.get("mtp_layers_block_type") or []) if t.get("num_nextn_predict_layers") else []
        self.latent = t["moe_latent_size"]
        self.shared_inner = t["moe_shared_expert_intermediate_size"]
        self.route_scale = t["routed_scaling_factor"]
        self.norm_topk = t["norm_topk_prob"]
        assert t["n_group"] == 1 and t["topk_group"] == 1 and t.get("scoring_func", "sigmoid") == "sigmoid"
        assert t["mlp_hidden_act"] == "relu2" and not t["mlp_bias"]
        self.heads = t["num_attention_heads"]
        self.kv_heads = t["num_key_value_heads"]
        self.head_dim = t["head_dim"]
        self.m_heads = t["mamba_num_heads"]
        self.m_dim = t["mamba_head_dim"]
        self.m_inner = self.m_heads * self.m_dim
        self.m_groups = t["n_groups"]
        self.m_state = t["ssm_state_size"]
        self.conv = t["conv_kernel"]
        self.conv_dim = self.m_inner + 2 * self.m_groups * self.m_state
        # the floor of dt: time_step_limit[0] (0), as the fast path of modeling_nemotron_h.py
        # (mamba_chunk_scan_combined, dt_limit), vLLM and llama.cpp; the slow torch path clamps
        # at time_step_min (1e-4), which ~10% of the dt values of a real prompt fall under (against
        # llama.cpp: KL 0.083 -> 0.069, top-1 90% -> 94% without it)
        self.dt_min = float((t.get("time_step_limit") or [0.0])[0] or 0.0)
        assert t["mamba_hidden_act"] == "silu" and t["use_conv_bias"] and not t["use_bias"]
        self.eos = [2, 11]


def rms_norm(x, w, eps):
    x = np.asarray(x, np.float32)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps)) * w).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20)))).astype(np.float32)


def relu2(x):
    r = np.maximum(x, 0)
    return (r * r).astype(np.float32)


def q8_0(blocks):
    """ggml Q8_0 blocks (uint8 [rows, n, 34]: d fp16, 32 int8) -> float32 [rows, n * 32]."""
    b = np.ascontiguousarray(blocks)
    d = b[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
    q = b[..., 2:].view(np.int8).astype(np.float32)
    return (q * d[..., None]).reshape(*b.shape[:-2], -1)


E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def _e4m3():
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 512.0, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where(b & 128, -v, v).astype(np.float32)


E4M3 = _e4m3()


def fp8(w, s):
    """ModelOpt FP8 (w: E4M3 bytes [rows, cols]; s: the tensor's float32 scale) -> float32."""
    return E4M3[np.asarray(w)] * np.float32(s)


def nvfp4(w, s, g):
    """ModelOpt NVFP4 (w: uint8 [rows, cols / 2], value 2i in the low 4 bits of byte i; s: E4M3
    bytes [rows, cols / 16]; g: the float32 scale) -> float32 [rows, cols]."""
    w = np.asarray(w)
    v = np.empty((w.shape[0], w.shape[1] * 2), np.float32)
    v[:, 0::2] = E2M1[w & 15]
    v[:, 1::2] = E2M1[w >> 4]
    # (code x scale) is exact; then x g: one rounding, as the kernels (KQ_NV4)
    sc = E4M3[np.asarray(s)]
    return ((v.reshape(w.shape[0], -1, 16) * sc[..., None]) * np.float32(g)).reshape(v.shape)


class Checkpoint:
    """The tensors of the checkpoint by name (views of the memory maps of the shards)."""

    def __init__(self, path):
        self.path = path
        self.where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self._files = {}

    def file(self, name):
        f = self.where[name]
        st = self._files.get(f)
        if st is None:
            st = self._files[f] = SafeTensors(os.path.join(self.path, f))
        return st

    def raw(self, name):
        """The tensor in its stored dtype (bfloat16: widened to float32 by SafeTensors.get)."""
        return self.file(name).get(name, dtype=None)

    def bytes(self, name):
        """The bytes of the tensor as the shard has them (a view of the map)."""
        st = self.file(name)
        o0, o1 = st.header[name]["data_offsets"]
        return np.frombuffer(st._mm, np.uint8, count=o1 - o0, offset=st._base + o0)

    def f32(self, name):
        return np.ascontiguousarray(self.file(name).get(name, dtype=np.float32))

    def shape(self, name):
        return tuple(self.file(name).shape(name))

    def close(self):
        for st in self._files.values():
            st.close()
        self._files = {}


class NemotronCache:
    """The state of one sequence: the conv rows (3 x 10240) and the SSM state (128 x 64 x 128) of
    each Mamba layer, the keys and values of each attention layer (n tokens so far)."""

    def __init__(self, cfg):
        self.n = 0
        self.conv, self.ssm, self.k, self.v = {}, {}, {}, {}
        for i, kind in enumerate(cfg.kinds):
            if kind == "mamba":
                self.conv[i] = np.zeros((cfg.conv - 1, cfg.conv_dim), np.float32)
                self.ssm[i] = np.zeros((cfg.m_heads, cfg.m_dim, cfg.m_state), np.float32)
            elif kind == "attention":
                self.k[i] = np.zeros((0, cfg.kv_heads, cfg.head_dim), np.float32)
                self.v[i] = np.zeros((0, cfg.kv_heads, cfg.head_dim), np.float32)


class NemotronCPU:
    """The reference forward. The dense matrices are decoded once to float32 on first use (about
    15 GB for all); the experts on each use (1.23 MB each as stored)."""

    def __init__(self, path, keep_dense=True):
        self.path = path
        self.cfg = NemotronConfig(path)
        self.ck = Checkpoint(path)
        self.keep_dense = keep_dense
        self._dense = {}
        self.hook = None                    # hook(layer, x) after each block (the checks)

    def cache(self):
        return NemotronCache(self.cfg)

    # ---- the weights

    def _mat(self, name):
        """A matrix as float32 [out, in]: Q8_0, NVFP4, IQ2_XXS or bf16 by what is stored."""
        m = self._dense.get(name)
        if m is not None:
            return m
        ck = self.ck
        if ck.file(name + ".weight").dtype(name + ".weight") == "F8_E4M3":
            # FP8 with a tensor scale (Super's Mamba, attention, lm_head)
            m = fp8(ck.raw(name + ".weight"), float(np.asarray(ck.f32(name + ".weight_scale")).reshape(-1)[0]))
        elif name + ".weight_scale" in ck.where:
            m = nvfp4(ck.raw(name + ".weight"), ck.raw(name + ".weight_scale"),
                      float(np.asarray(ck.f32(name + ".weight_scale_2")).reshape(-1)[0]))
        else:
            w = ck.raw(name + ".weight")
            if w.dtype == np.uint8 and w.ndim == 3 and w.shape[-1] == 34:
                m = q8_0(w)
            elif w.dtype == np.uint8 and w.ndim == 3 and w.shape[-1] == iq2.BYTES:
                m = iq2.dequant(w)
            else:
                m = ck.f32(name + ".weight")
        m = np.ascontiguousarray(m, np.float32)
        if self.keep_dense and ".experts." not in name:
            self._dense[name] = m
        return m

    def _vec(self, name):
        v = self._dense.get(name)
        if v is None:
            v = self._dense[name] = self.ck.f32(name)
        return v

    def embed(self, ids):
        st = self.ck.file(BB + "embeddings.weight")
        return np.stack([st.get_row(BB + "embeddings.weight", int(t)) for t in ids]).astype(np.float32)

    # ---- the blocks

    def mamba(self, i, x, cache):
        c = self.cfg
        p = BB + "layers.%d.mixer." % i
        zxd = x @ self._mat(p + "in_proj").T                    # [t, 18560]
        z = zxd[:, :c.m_inner]
        xbc = zxd[:, c.m_inner:c.m_inner + c.conv_dim]
        dt = zxd[:, c.m_inner + c.conv_dim:]
        cw = self._vec(p + "conv1d.weight").reshape(c.conv_dim, c.conv)    # [10240, 4]
        cb = self._vec(p + "conv1d.bias")
        rows = np.concatenate([cache.conv[i], xbc])            # [3 + t, 10240]
        t = x.shape[0]
        conv = np.zeros((t, c.conv_dim), np.float32)
        for k in range(c.conv):
            conv += rows[k:k + t] * cw[:, k]
        cache.conv[i] = rows[-(c.conv - 1):].copy()
        xbc = silu(conv + cb)
        xs = xbc[:, :c.m_inner].reshape(t, c.m_heads, c.m_dim)
        rep = c.m_heads // c.m_groups
        B = np.repeat(xbc[:, c.m_inner:c.m_inner + c.m_groups * c.m_state].reshape(t, c.m_groups, c.m_state), rep, axis=1)
        C = np.repeat(xbc[:, c.m_inner + c.m_groups * c.m_state:].reshape(t, c.m_groups, c.m_state), rep, axis=1)
        A = -np.exp(self._vec(p + "A_log"))
        D = self._vec(p + "D")
        dt = np.maximum(softplus(dt + self._vec(p + "dt_bias")), np.float32(c.dt_min))   # [t, 128]
        h = cache.ssm[i]
        y = np.empty((t, c.m_heads, c.m_dim), np.float32)
        for s in range(t):
            dA = np.exp(dt[s] * A)                                       # [128]
            h = h * dA[:, None, None] + (dt[s][:, None] * xs[s])[:, :, None] * B[s][:, None, :]
            y[s] = np.einsum("hdn,hn->hd", h, C[s]) + D[:, None] * xs[s]
        cache.ssm[i] = h.astype(np.float32)
        y = y.reshape(t, c.m_inner) * silu(z)
        g = c.m_inner // c.m_groups
        y = rms_norm(y.reshape(t, c.m_groups, g), self._vec(p + "norm.weight").reshape(c.m_groups, g), c.eps)
        return y.reshape(t, c.m_inner) @ self._mat(p + "out_proj").T

    def attention(self, i, x, cache, p=None):
        """p: the prefix of the mixer's tensors (default block i's; the MTP block's for the
        drafts, its keys and values in cache under i)."""
        c = self.cfg
        p = p or BB + "layers.%d.mixer." % i
        t = x.shape[0]
        q = (x @ self._mat(p + "q_proj").T).reshape(t, c.heads, c.head_dim)
        k = (x @ self._mat(p + "k_proj").T).reshape(t, c.kv_heads, c.head_dim)
        v = (x @ self._mat(p + "v_proj").T).reshape(t, c.kv_heads, c.head_dim)
        K = cache.k[i] = np.concatenate([cache.k[i], k])
        V = cache.v[i] = np.concatenate([cache.v[i], v])
        n0 = K.shape[0] - t
        rep = c.heads // c.kv_heads
        Kh = np.repeat(K, rep, axis=1)                                   # [n, 32, 128]
        Vh = np.repeat(V, rep, axis=1)
        s = np.einsum("thd,nhd->htn", q, Kh) / np.sqrt(np.float32(c.head_dim))
        mask = np.arange(K.shape[0])[None, :] > (n0 + np.arange(t))[:, None]
        s = np.where(mask[None], -np.inf, s)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("htn,nhd->thd", s.astype(np.float32), Vh).reshape(t, -1)
        return o @ self._mat(p + "o_proj").T

    def route(self, i, x, p=None):
        """(ids [t, k], weights [t, k]) of the experts of each token (p: the mixer's prefix)."""
        c = self.cfg
        p = (p or BB + "layers.%d.mixer." % i) + "gate."
        E, _, top = c.moe[i] if i in c.moe else c.moe[min(c.moe)]
        scores = sigmoid(x @ self._vec(p + "weight").T)                 # [t, E]
        choice = scores + self._vec(p + "e_score_correction_bias")
        ids = np.argsort(-choice, axis=-1, kind="stable")[:, :top]
        w = np.take_along_axis(scores, ids, axis=-1)
        if c.norm_topk:
            w = w / (w.sum(-1, keepdims=True) + 1e-20)
        return ids, (w * c.route_scale).astype(np.float32)

    def moe(self, i, x, p=None):
        c = self.cfg
        p = p or BB + "layers.%d.mixer." % i
        ids, w = self.route(i, x, p)
        lat = x @ self._mat(p + "fc1_latent_proj").T                    # [t, 1024]
        out = np.zeros_like(lat)
        for e in np.unique(ids):
            rows, cols = np.nonzero(ids == e)
            up = self._mat(p + "experts.%d.up_proj" % e)
            down = self._mat(p + "experts.%d.down_proj" % e)
            out[rows] += (relu2(lat[rows] @ up.T) @ down.T) * w[rows, cols][:, None]
        y = out @ self._mat(p + "fc2_latent_proj").T
        sh = relu2(x @ self._mat(p + "shared_experts.up_proj").T) @ self._mat(p + "shared_experts.down_proj").T
        return (y + sh).astype(np.float32)

    def head(self, x):
        """The logits of the rows of x (normed by norm_f), in slices of the lm_head."""
        h = rms_norm(x, self._vec(BB + "norm_f.weight"), self.cfg.eps)
        self.last_hidden = h            # (the rows after norm_f: the input of the MTP block)
        return self._head_rows(h)

    def _head_rows(self, h):
        """h (normed) times the lm_head, in slices of its rows: Q8_0 (Bolt) or FP8 (Super)."""
        ck = self.ck
        w = ck.raw(LM + "lm_head.weight")
        f8 = ck.file(LM + "lm_head.weight").dtype(LM + "lm_head.weight") == "F8_E4M3"
        sc = float(np.asarray(ck.f32(LM + "lm_head.weight_scale")).reshape(-1)[0]) if f8 else None
        out = np.empty((h.shape[0], w.shape[0]), np.float32)
        for r0 in range(0, w.shape[0], 8192):
            out[:, r0:r0 + 8192] = h @ (fp8(w[r0:r0 + 8192], sc) if f8 else q8_0(w[r0:r0 + 8192])).T
        return out

    def mtp(self, token, h, cache):
        """One draft of the MTP block (Super; vLLM nemotron_h_mtp.py): token, the token after the
        target's last position, and h [hidden], the target's hidden there after norm_f (or, for the
        drafts after the first, the out of the draft before). u = eh_proj(cat(enorm(embed),
        hnorm(h))); the attention block on its own keys and values (cache under "mtp"); s = a + u;
        the latent MoE block; out = final_layernorm(m + s); the logits by the lm_head (no norm_f).
        Return (logits [vocab], out [hidden])."""
        c = self.cfg
        p0, p1 = LM + "mtp.layers.0.", LM + "mtp.layers.1."
        if "mtp" not in cache.k:
            cache.k["mtp"] = np.zeros((0, c.kv_heads, c.head_dim), np.float32)
            cache.v["mtp"] = np.zeros((0, c.kv_heads, c.head_dim), np.float32)
        e = rms_norm(self.embed([token]), self._vec(p0 + "enorm.weight"), c.eps)
        hh = rms_norm(np.asarray(h, np.float32).reshape(1, -1), self._vec(p0 + "hnorm.weight"), c.eps)
        u = np.concatenate([e, hh], -1) @ self._mat(p0 + "eh_proj").T
        a = self.attention("mtp", rms_norm(u, self._vec(p0 + "norm.weight"), c.eps), cache, p0 + "mixer.")
        s_ = a + u
        m = self.moe("mtp", rms_norm(s_, self._vec(p1 + "norm.weight"), c.eps), p1 + "mixer.")
        out = rms_norm(m + s_, self._vec(p1 + "final_layernorm.weight"), c.eps)
        return self._head_rows(out)[0], out[0]

    def forward(self, ids, cache, logits="all"):
        """The logits of each token of ids (logits="last": of the last only), after cache.n
        tokens; the cache moves on by len(ids)."""
        c = self.cfg
        x = self.embed(ids)
        for i, kind in enumerate(c.kinds):
            h = rms_norm(x, self._vec(BB + "layers.%d.norm.weight" % i), c.eps)
            if kind == "mamba":
                x = x + self.mamba(i, h, cache)
            elif kind == "attention":
                x = x + self.attention(i, h, cache)
            else:
                x = x + self.moe(i, h)
            if self.hook is not None:
                self.hook(i, x)
            if not self.keep_dense:
                # (the maps of the shards closed: the pages read stay out of the process, and the
                # page cache can let them go; Super reads 65 GB of experts)
                self.ck.close()
        cache.n += len(ids)
        return self.head(x[-1:] if logits == "last" else x)
