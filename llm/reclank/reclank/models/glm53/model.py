"""GLM-5.3-Flash on the CPU in NumPy (P2 of docs/GLM53_PLAN.md): the forward of transformers'
glm5_next (Glm5NextTextModel) op for op, in float32, with the EXL3 products of this runtime
(cops.e3_linear: x through the Hadamard to int8, the trellis as it is) for the EXL3 matrices and
the float16 / float32 tensors as stored. The slow reference of the runtime: the GPU model's checks
compare against it.

- KDA (Kimi Delta Attention, 34 layers): qkv, a short causal conv (4) + SiLU, l2-normed q and k,
  a decay for each channel g = -5 sigmoid(exp(A_log) (f_b f_a x + dt_bias)), beta = sigmoid(b x),
  the recurrent form of the delta rule on a 128 x 128 state of each of 64 heads (the prompt too:
  transformers' chunked form is the same sums), the RMSNorm of each head times sigmoid(g_b g_a x);
- DSA (11 layers): MLA without RoPE on the 512 latent (absorbed: W_UK^T q against the latents,
  W_UV after the weighted sum), the keys an indexer picks: 32 heads of relu(q k) over keys pooled
  by 4 (softmax of gate + ape in each pool), the best 512 complete pools and the open tail;
- mHC: 4 streams, Sinkhorn of the comb matrix (20), the head the mean of the streams;
- MoE (3-44): sigmoid + correction bias for the choice, 8 of 288, normalized, x 2.5; SwiGLU with
  silu(min(g, 10)) * clamp(u, -10, 10); the shared expert (ungated); layers 0-2 a dense MLP.

    m = GLM53CPU("/home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw")
    cache = m.cache()
    logits = m.forward(ids, cache)              # [t, vocab]; the streams of each layer: hook
    logits = m.forward([tok], cache)
"""
from __future__ import annotations

import numpy as np

from reclank.backends.cpu import cops
from reclank.formats.exl3 import KQ_E3_BASE


def rms_norm(x, w, eps):
    x = np.asarray(x, np.float32)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps)) * w).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def l2norm(x, eps=1e-6):
    return (x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + np.float32(eps))).astype(np.float32)


def swiglu(g, u, limit):
    return (silu(np.minimum(g, limit)) * np.clip(u, -limit, limit)).astype(np.float32)


class GLM53Cache:
    """The state of one sequence: the conv rows and the delta-rule state of each KDA layer, the
    latents, index keys and pool gates of each DSA layer (n tokens so far)."""

    def __init__(self, cfg, layers):
        C = 3 * cfg.kda_heads * cfg.kda_dim
        self.n = 0
        self.conv, self.state = {}, {}
        self.lat, self.ik, self.ig = {}, {}, {}
        for i in layers:
            if cfg.layer_types[i] == "kda":
                self.conv[i] = np.zeros((cfg.conv_kernel - 1, C), np.float32)
                self.state[i] = np.zeros((cfg.kda_heads, cfg.kda_dim, cfg.kda_dim), np.float32)
            else:
                self.lat[i] = np.zeros((0, cfg.kv_lora_rank), np.float32)
                self.ik[i] = np.zeros((0, cfg.index_dim), np.float32)
                self.ig[i] = np.zeros((0, cfg.index_dim), np.float32)


class GLM53CPU:
    """The NumPy model on a GLM53Source (see the module text)."""

    def __init__(self, path_or_src, layers=None, float_weights=False):
        """float_weights (a check of the math): every EXL3 matrix decoded to float32, float
        products."""
        from reclank.formats.exl3_glm53 import GLM53Source
        self.g = GLM53Source(path_or_src) if isinstance(path_or_src, str) else path_or_src
        self.cfg = self.g.config()
        self.n_layers = layers or self.cfg.num_hidden_layers
        self.float_weights = float_weights
        self._f = {}
        self.picks = {}             # the router's choice of each MoE layer in the last forward

    # ---- the tensors ----

    def F(self, gname):
        """A float32 tensor (NumPy shape), kept."""
        w = self._f.get(gname)
        if w is None:
            w = self._f[gname] = self.g.dequant(gname).astype(np.float32)
        return w

    def lin(self, gname, x):
        """x W^T of a matrix by its GGUF name: the EXL3 product, or float32."""
        dims, t, _ = self.g.tensors[gname]
        if t >= KQ_E3_BASE and not self.float_weights:
            w, _d, _t = self.g.raw(gname)
            return cops.e3_linear(w, t - KQ_E3_BASE, dims[1], dims[0], x)
        return np.asarray(x, np.float32) @ self.F(gname).T

    def cache(self):
        return GLM53Cache(self.cfg, range(self.n_layers))

    # ---- mHC ----

    def hc(self, H, prefix):
        """(post [t, 4], comb [t, 4, 4], the block input [t, hidden]) of the streams H [t, 4, hidden]."""
        cfg = self.cfg
        t, n, hid = H.shape
        eps = np.float32(cfg.hc_eps)
        flat = H.reshape(t, n * hid)
        flat = flat / np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + np.float32(cfg.rms_norm_eps))
        mix = flat @ self.F(prefix + "_fn").T                              # [t, 24]
        base, scale = self.F(prefix + "_base"), self.F(prefix + "_scale")
        pre = sigmoid(mix[:, :n] * scale[0] + base[:n]) + eps
        post = 2 * sigmoid(mix[:, n:2 * n] * scale[1] + base[n:2 * n])
        c = mix[:, 2 * n:].reshape(t, n, n) * scale[2] + base[2 * n:].reshape(n, n)
        c = np.exp(c - c.max(axis=-1, keepdims=True))
        comb = c / c.sum(axis=-1, keepdims=True) + eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
        for _ in range(cfg.hc_iters - 1):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
        x = (pre[:, :, None] * H).sum(axis=1)
        return post.astype(np.float32), comb.astype(np.float32), x.astype(np.float32)

    @staticmethod
    def hc_post(y, H, post, comb):
        """The new streams: post (x) y + comb^T H."""
        return (post[:, :, None] * y[:, None, :] + np.einsum("tij,tid->tjd", comb, H)).astype(np.float32)

    # ---- KDA ----

    def kda_inputs(self, i, x):
        """(qkv [t, 3 x 8192], f [t, 8192], b [t, 64], gate [t, 8192]) of KDA layer i: the products
        of each row (the operands of the GDN record: qkv, a, b, z)."""
        b_ = "blk.%d." % i
        qkv = self.lin(b_ + "attn_qkv.weight", x)
        f = self.lin(b_ + "ssm_f_b.weight", self.lin(b_ + "ssm_f_a.weight", x))
        beta = self.lin(b_ + "ssm_beta.weight", x)
        gate = self.lin(b_ + "ssm_g_b.weight", self.lin(b_ + "ssm_g_a.weight", x))
        return qkv, f, beta, gate

    def kda_core(self, i, qkv, f, beta, gate, cache):
        """The output of KDA layer i before o_proj [t, 8192], in order through the cache."""
        cfg = self.cfg
        b = "blk.%d." % i
        t = qkv.shape[0]
        nh, d = cfg.kda_heads, cfg.kda_dim
        K = cfg.conv_kernel
        hist = np.concatenate([cache.conv[i], qkv])
        cache.conv[i] = hist[-(K - 1):].copy()
        w = self.F(b + "ssm_conv1d.weight")                                # [C, K]
        conv = np.zeros_like(qkv)
        for k in range(K):
            conv += hist[k:k + t] * w[:, k]
        conv = silu(conv)
        q, k_, v = (conv[:, j * nh * d:(j + 1) * nh * d].reshape(t, nh, d) for j in range(3))
        f = (f + self.F(b + "ssm_dt.bias")).reshape(t, nh, d)
        g = cfg.gate_lower_bound * sigmoid(np.exp(self.F(b + "ssm_a_log"))[None, :, None] * f)
        beta = sigmoid(beta)                                               # [t, nh]
        q = l2norm(q) * np.float32(d ** -0.5)
        k_ = l2norm(k_)
        S = cache.state[i]
        out = np.empty((t, nh, d), np.float32)
        for j in range(t):
            S = S * np.exp(g[j])[:, :, None]
            mem = np.einsum("hkv,hk->hv", S, k_[j])
            delta = (v[j] - mem) * beta[j][:, None]
            S = S + k_[j][:, :, None] * delta[:, None, :]
            out[j] = np.einsum("hkv,hk->hv", S, q[j])
        cache.state[i] = S.astype(np.float32)
        o = rms_norm(out, self.F(b + "ssm_norm.weight"), cfg.rms_norm_eps) * sigmoid(gate.reshape(t, nh, d))
        return o.reshape(t, nh * d)

    def kda(self, i, x, cache):
        return self.lin("blk.%d.attn_output.weight" % i, self.kda_core(i, *self.kda_inputs(i, x), cache))

    # ---- DSA ----

    def index_keys(self, i, x, q_resid, cache, pos):
        """The keys each query reads ([t, n] bool), from the indexer (transformers'
        Glm5NextTextIndexer on one unpadded sequence)."""
        cfg = self.cfg
        b = "blk.%d." % i
        t = x.shape[0]
        k = x @ self.F(b + "idx_k.weight").T
        mu = k.mean(axis=-1, keepdims=True)
        var = ((k - mu) ** 2).mean(axis=-1, keepdims=True)
        k = (k - mu) / np.sqrt(var + np.float32(1e-6)) * self.F(b + "idx_k_norm.weight") + self.F(b + "idx_k_norm.bias")
        cache.ik[i] = np.concatenate([cache.ik[i], k.astype(np.float32)])
        cache.ig[i] = np.concatenate([cache.ig[i], x @ self.F(b + "idx_gate.weight").T])
        n = pos + t
        P, ratio = n // cfg.index_pool, cfg.index_pool                   # the complete pools
        keys = np.zeros((t, n), bool)
        causal = np.arange(n)[None, :] <= (pos + np.arange(t))[:, None]
        sel_k = min(cfg.index_topk // ratio, P)
        if P == 0:
            return causal
        lg = cache.ig[i][:P * ratio].reshape(P, ratio, -1) + self.F(b + "idx_ape")[None]
        pr = np.exp(lg - lg.max(axis=1, keepdims=True))
        pr = pr / pr.sum(axis=1, keepdims=True)
        pool = (pr * cache.ik[i][:P * ratio].reshape(P, ratio, -1)).sum(axis=1)   # [P, 128]
        nh, d = cfg.index_heads, cfg.index_dim
        q = self.lin(b + "idx_q_b.weight", q_resid).reshape(t, nh, d)
        sc = np.maximum(np.einsum("thd,pd->thp", q, pool) * np.float32(d ** -0.5), 0)
        wt = (x @ self.F(b + "idx_w.weight").T) * np.float32(nh ** -0.5)
        score = np.einsum("th,thp->tp", wt, sc)                            # [t, P]
        self.index_scores[i] = score
        for j in range(t):
            pj = pos + j
            vis = (np.arange(P) * ratio + ratio - 1) <= pj
            s = np.where(vis, score[j], np.finfo(np.float32).min)
            # relu gives many equal scores: the set is what counts (P4 compares sets by score)
            top = np.argsort(-s, kind="stable")[:sel_k]
            top = top[vis[top]]
            keys[j, (top[:, None] * ratio + np.arange(ratio)).reshape(-1)] = True
            tail = (pj + 1) % ratio
            keys[j, pj + 1 - tail:pj + 1] = True
        return keys & causal

    def dsa_inputs(self, i, x):
        """(q_resid [t, 1536], q [t, H, 256], the latents [t, 512]) of DSA layer i."""
        cfg = self.cfg
        b = "blk.%d." % i
        q_resid = rms_norm(self.lin(b + "attn_q_a.weight", x), self.F(b + "attn_q_a_norm.weight"), cfg.rms_norm_eps)
        q = self.lin(b + "attn_q_b.weight", q_resid).reshape(x.shape[0], cfg.num_heads, cfg.qk_head_dim)
        c = rms_norm(self.lin(b + "attn_kv_a_mqa.weight", x), self.F(b + "attn_kv_a_norm.weight"), cfg.rms_norm_eps)
        return q_resid, q, c

    def mla(self, i, q, C, keys):
        """The absorbed MLA of DSA layer i: queries q [t, H, 256] over the latents C [n, 512] that
        keys [t, n] marks; the output before o_proj [t, H x 256]."""
        cfg = self.cfg
        b = "blk.%d." % i
        t, H, dk = q.shape
        wk, wv = self.F(b + "attn_k_b.weight"), self.F(b + "attn_v_b.weight")   # [H, 256, 512]
        ql = np.einsum("thd,hdc->thc", q, wk)                              # W_UK^T q
        s = np.einsum("thc,nc->thn", ql, C) * np.float32(dk ** -0.5)
        s = np.where(keys[:, None, :], s, -np.inf)
        s = np.exp(s - s.max(axis=-1, keepdims=True))
        p = s / s.sum(axis=-1, keepdims=True)
        ol = np.einsum("thn,nc->thc", p, C)
        o = np.einsum("thc,hdc->thd", ol, wv).astype(np.float32)          # W_UV
        return o.reshape(t, H * cfg.v_head_dim)

    def dsa(self, i, x, cache, pos):
        q_resid, q, c = self.dsa_inputs(i, x)
        cache.lat[i] = np.concatenate([cache.lat[i], c])
        keys = self.index_keys(i, x, q_resid, cache, pos)
        return self.lin("blk.%d.attn_output.weight" % i, self.mla(i, q, cache.lat[i], keys))

    # ---- the MLPs ----

    def mlp(self, i, x):
        b = "blk.%d." % i
        g, u = self.lin(b + "ffn_gate.weight", x), self.lin(b + "ffn_up.weight", x)
        return self.lin(b + "ffn_down.weight", swiglu(g, u, self.cfg.swiglu_limit))

    def route(self, i, x):
        """(ids [t, 8], weights [t, 8]) of the router of MoE layer i."""
        cfg = self.cfg
        b = "blk.%d." % i
        s = sigmoid(x @ self.F(b + "ffn_gate_inp.weight").T)              # [t, 288]
        choice = s + self.F(b + "exp_probs_b.bias")
        ids = np.argsort(-choice, axis=-1, kind="stable")[:, :cfg.top_k]
        w = np.take_along_axis(s, ids, axis=-1)
        w = w / (w.sum(axis=-1, keepdims=True) + np.float32(1e-20)) * np.float32(cfg.routed_scale)
        return ids, w.astype(np.float32)

    def moe(self, i, x):
        cfg = self.cfg
        b = "blk.%d." % i
        ids, w = self.route(i, x)
        self.picks[i] = (ids, w)
        out = np.zeros_like(x)
        H, I = cfg.hidden_size, cfg.moe_inter
        for e in np.unique(ids):
            tok, slot = np.nonzero(ids == e)
            xe = np.ascontiguousarray(x[tok])
            if self.float_weights:
                wg, wu, wd = (self.g.dequant(b + "ffn_%s_exps.weight" % m, rows=[int(e)])[0]
                              for m in ("gate", "up", "down"))
                a = swiglu(xe @ wg.T, xe @ wu.T, cfg.swiglu_limit)
                out[tok] += (a @ wd.T) * w[tok, slot][:, None]
                continue
            (wg, Kg), (wu, Ku), (wd, Kd) = (self.g.expert_block(i, int(e), m) for m in ("gate", "up", "down"))
            a = swiglu(cops.e3_linear(wg, Kg, I, H, xe), cops.e3_linear(wu, Ku, I, H, xe), cfg.swiglu_limit)
            out[tok] += cops.e3_linear(wd, Kd, H, I, a) * w[tok, slot][:, None]
        g, u = self.lin(b + "ffn_gate_shexp.weight", x), self.lin(b + "ffn_up_shexp.weight", x)
        return out + self.lin(b + "ffn_down_shexp.weight", swiglu(g, u, cfg.swiglu_limit))

    # ---- the model ----

    def layer(self, i, H, cache, pos, hook=None):
        """The streams after layer i. hook(name, value): attn_in<i>, attn_out<i>, mid<i> (the
        streams after the attention), mlp_in<i>, mlp_out<i> (the names of the reference's trace)."""
        cfg = self.cfg
        b = "blk.%d." % i
        hook = hook or (lambda k, v: None)
        post, comb, x = self.hc(H, b + "hc_attn")
        x = rms_norm(x, self.F(b + "attn_norm.weight"), cfg.rms_norm_eps)
        hook("attn_in%d" % i, x)
        y = self.kda(i, x, cache) if cfg.layer_types[i] == "kda" else self.dsa(i, x, cache, pos)
        hook("attn_out%d" % i, y)
        H = self.hc_post(y, H, post, comb)
        hook("mid%d" % i, H)
        post, comb, x = self.hc(H, b + "hc_ffn")
        x = rms_norm(x, self.F(b + "ffn_norm.weight"), cfg.rms_norm_eps)
        hook("mlp_in%d" % i, x)
        y = self.moe(i, x) if cfg.mlp_types[i] == "moe" else self.mlp(i, x)
        hook("mlp_out%d" % i, y)
        return self.hc_post(y, H, post, comb)

    def forward(self, ids, cache, hook=None):
        """The logits [t, vocab] of tokens at cache.n.. . hook(name, value): "embed", "h<i>" (the
        streams after layer i), "norm" (the input of the head)."""
        cfg = self.cfg
        pos = cache.n
        self.index_scores = {}
        x = self.g.embed_rows(ids).astype(np.float32)
        if hook:
            hook("embed", x)
        H = np.repeat(x[:, None, :], cfg.hc_mult, axis=1)
        for i in range(self.n_layers):
            H = self.layer(i, H, cache, pos, hook)
            if hook:
                hook("h%d" % i, H)
        cache.n = pos + len(ids)
        h = rms_norm(H.mean(axis=1), self.F("output_norm.weight"), cfg.rms_norm_eps)
        if hook:
            hook("norm", h)
        return self.lin("output.weight", h)
