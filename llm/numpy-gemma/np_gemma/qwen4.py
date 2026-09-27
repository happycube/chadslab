"""Qwen3.8-Flash-Next (qwen4exp, a preview of Qwen4), in NumPy.

QWEN38_PLAN.md, phase 1. This is the reference of this runtime for the
model. It reads the GGUF file of Unsloth with open_gguf (split files). It
dequantizes each weight when it uses it. transformers (models/qwen4_exp)
is the reference for it; scripts/check_qwen4.py compares them.

Some parts of Qwen3.6 stay (np_gemma/qwen.py, QwenGGUF):

- the Gated DeltaNet (48 value heads in tiled order, a sigmoid gate of its
  norm);
- the gated attention;
- the MoE with a shared expert (512 experts, 10 for each token).

The new parts:

- the gated residual. The residual has 4 streams of 2560 values. The
  function hc_pre makes the input of a block: a grouped norm, a low-rank
  gate, and the mean over the streams. The block output goes back to each
  stream with its inject weight;
- the n-gram table (PLE) at one layer. Hashes of the last 2 and 3 tokens
  read 16 rows of 160 values. A gate for each stream and a dilated
  convolution add them to the streams;
- QSA attention. An indexer scores blocks of 4 keys and keeps the best 512
  blocks (2048 tokens) and the tail for each query (Qwen4.qsa_mask). For a
  context of at most 2048 + 3 tokens it keeps all the keys;
- no final norm: a last mixer (hc_pre with no inject) makes the input of
  the head.
"""
from __future__ import annotations

import numpy as np

from .gguf import open_gguf
from .qwen import QwenCache, QwenConfig, QwenGGUF, kv_rows, kv_store, rms_norm, sigmoid, silu



def config_from_gguf(g):
    """The QwenConfig of a qwen4exp file, with the fields of the new parts."""
    cfg = QwenConfig.from_gguf(g)
    m = g.meta
    a = m.get("general.architecture", "qwen4exp")

    def k(name, default=None):
        return m.get("%s.%s" % (a, name), default)

    cfg.hc_count = int(k("hyper_connection.count"))
    cfg.hc_lowrank = int(k("hyper_connection.low_rank"))
    cfg.ple_layers = [int(x) for x in k("ple.layers", [])]
    cfg.ple_ngram = int(k("ple.ngram_size", 3))
    cfg.ple_heads_per_ngram = int(k("ple.heads_per_ngram", 8))
    cfg.ple_conv_kernel = int(k("ple.conv_kernel", 4))
    cfg.ple_eos = int(k("ple.eos_token_id"))
    cfg.ple_row = int(k("embedding_length_per_layer_input"))
    cfg.ple_multipliers = [int(x) for x in k("ple.layer_multipliers")]
    cfg.ple_offsets = [int(x) for x in k("ple.head_offsets")]
    cfg.ple_sizes = [int(x) for x in k("ple.head_vocab_sizes")]
    cfg.indexer_heads = int(k("attention.indexer.head_count"))
    cfg.indexer_dim = int(k("attention.indexer.key_length"))
    cfg.indexer_top_k = int(k("attention.indexer.top_k"))
    cfg.compress_ratios = [int(x) for x in k("attention.compress_ratios")]
    # The gate of the norm of the DeltaNet is sigmoid (output_gate_type; llama.cpp
    # qwen4exp.cpp), not the silu of Qwen3.5.
    cfg.lin_gate = "sigmoid"
    return cfg


class Qwen4Cache(QwenCache):
    """QwenCache with the state of the n-gram layer: the last ngram - 1
    tokens, and the last (kernel - 1) * dilation inputs of its convolution."""

    def __init__(self, cfg, max_len=4096, kv=None):
        super().__init__(cfg, max_len, kv)
        hc_dim = cfg.hc_count * cfg.hidden_size
        self.ple_ids = np.full(cfg.ple_ngram - 1, cfg.ple_eos, dtype=np.int64)
        self.ple_conv = {i: np.zeros(((cfg.ple_conv_kernel - 1) * cfg.ple_ngram, hc_dim),
                                     np.float32) for i in cfg.ple_layers}
        # The raw keys of the indexer of each QSA layer (one for each position).
        self.idx_k = {i: np.zeros((max_len, cfg.indexer_dim), np.float32)
                      for i, t in enumerate(cfg.layer_types) if t == "full_attention"}


def grouped_norm(x, w, hidden, eps):
    """RMS norm of each group of hidden values of the last axis, times w (the
    weights of the file have the 1 in them)."""
    s = x.reshape(*x.shape[:-1], -1, hidden)
    s = s / np.sqrt(np.mean(s * s, axis=-1, keepdims=True) + eps)
    return (s.reshape(x.shape) * w).astype(np.float32)


class Qwen4(QwenGGUF):
    """The NumPy model of Qwen3.8-Flash-Next on a GGUF file.

        m = Qwen4(path)
        cache = Qwen4Cache(m.cfg)
        h = m.forward(ids, cache)            # the input of the head
        logits = m.logits(h[-1:])
    """

    def __init__(self, path, cfg=None, layers=None):
        self.path = path
        self.g = open_gguf(path)
        self.cfg = cfg or config_from_gguf(self.g)
        self.n_layers = layers or self.cfg.num_hidden_layers
        self._deq = {}

    def G(self, gname):
        """A dequantized tensor by its GGUF name, kept after the first use."""
        w = self._deq.get(gname)
        if w is None:
            w = self._deq[gname] = self.g.dequant(gname).astype(np.float32)
        return w

    # ---- the gated residual ----

    def hc_pre(self, H, prefix, inject=True):
        """The input of a block from the streams H (t x hc x hidden).
        prefix is blk.N.hc_attn, blk.N.hc_ffn, or output_hc. Return the
        input (t x hidden) and, with inject, the weight of each stream
        (t x hc)."""
        cfg = self.cfg
        t, hc, hid = H.shape
        hn = grouped_norm(H.reshape(t, hc * hid), self.G(prefix + "_norm.weight"), hid,
                          cfg.rms_norm_eps)
        lo = silu((hn @ self.G(prefix + "_down.weight").T) / hc)
        m = sigmoid(lo @ self.G(prefix + "_up.weight").T)
        mixed = (m * hn).reshape(t, hc, hid).mean(axis=1)
        if not inject:
            return mixed.astype(np.float32)
        w = 2.0 * sigmoid((hn @ self.G(prefix + "_inject.weight").T) / hc)
        return mixed.astype(np.float32), w.astype(np.float32)

    # ---- the n-gram table ----

    def ple_ids(self, ids, cache):
        """The rows of the n-gram table of each token (t x 16): the hashes
        of the last 2 and 3 tokens (transformers, Qwen4ExpTextNGramEmbedding).
        An n-gram does not reach back over the eos token of PLE."""
        cfg = self.cfg
        n, eos = cfg.ple_ngram, cfg.ple_eos
        ctx = n - 1
        hist = np.concatenate([cache.ple_ids, np.asarray(ids, dtype=np.int64)])
        cache.ple_ids = hist[-ctx:].copy()
        L = len(hist)
        pos = np.arange(L)
        eos_pos = np.where(hist == eos, pos, -1)
        prev = np.maximum.accumulate(eos_pos)
        prev_eos = np.concatenate([[-1], prev[:-1]])
        in_seg = pos - (prev_eos + 1)
        shifted = []
        for s in range(n):
            src = np.clip(pos - s, 0, None)
            valid = (in_seg >= s) & (pos - s >= 0)
            shifted.append(np.where(valid, hist[src], eos))
        mult = cfg.ple_multipliers
        blocks = []
        hpn = cfg.ple_heads_per_ngram
        for gram in range(2, n + 1):
            mixed = shifted[0] * np.int64(mult[0])
            for p in range(1, gram):
                mixed = np.bitwise_xor(mixed, shifted[p] * np.int64(mult[p]))
            h0 = (gram - 2) * hpn
            sizes = np.array(cfg.ple_sizes[h0:h0 + hpn], dtype=np.int64)
            offs = np.array(cfg.ple_offsets[h0:h0 + hpn], dtype=np.int64)
            blocks.append(np.remainder(mixed[:, None], sizes[None, :]) + offs[None, :])
        return np.concatenate(blocks, axis=1)[ctx:]

    def ple(self, i, H, ids, cache):
        """The output of the n-gram layer i for the streams H: add it to H."""
        cfg = self.cfg
        t, hc, hid = H.shape
        rows = self.ple_ids(ids, cache)                              # (t, 16)
        emb = self.g.dequant("per_layer_token_embd.weight", rows=rows.reshape(-1))
        emb = emb.reshape(t, -1)                                     # (t, 16 x 160)
        p = "blk.%d.ple_" % i
        eps = cfg.rms_norm_eps
        key = grouped_norm(emb @ self.G(p + "key.weight").T, self.G(p + "norm_key.weight"),
                           hid, eps).reshape(t, hc, hid)
        value = emb @ self.G(p + "value.weight").T                   # (t, hid)
        query = grouped_norm(H.reshape(t, hc * hid), self.G(p + "norm_query.weight"), hid,
                             eps).reshape(t, hc, hid)
        s = (key * query).sum(axis=-1) / np.sqrt(hid)                # (t, hc)
        s = np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6))
        gated = sigmoid(s)[:, :, None] * value[:, None, :]           # (t, hc, hid)
        gn = grouped_norm(gated.reshape(t, hc * hid), self.G(p + "norm_conv.weight"), hid, eps)
        # A depthwise causal convolution: kernel K, dilation ngram.
        K, dil = cfg.ple_conv_kernel, cfg.ple_ngram
        w = self.G(p + "conv1d.weight").reshape(hc * hid, K)         # (channels, K)
        hist = np.concatenate([cache.ple_conv[i], gn])
        cache.ple_conv[i] = hist[-(K - 1) * dil:].copy()
        base = (K - 1) * dil
        conv = np.zeros_like(gn)
        for k in range(K):
            back = (K - 1 - k) * dil
            conv += hist[base - back:base - back + t] * w[:, k]
        out = gated.reshape(t, hc * hid) + silu(conv)
        return (H + out.reshape(t, hc, hid)).astype(np.float32)

    # ---- QSA attention ----

    def _rot(self, x, cos, sin):
        """RoPE on the first rotary_dim values of the last axis (x: t x ... x
        d; cos, sin: t x rotary_dim)."""
        d = self.cfg.rotary_dim
        half = d // 2
        extra = (slice(None),) + (None,) * (x.ndim - 2)
        c, s = cos[extra], sin[extra]
        xr = x[..., :d]
        rh = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
        return np.concatenate([xr * c + rh * s, x[..., d:]], axis=-1)

    def qsa_mask(self, i, h, cache, pos):
        """The keys that each query of the QSA layer i reads: a bool array
        (t x (pos + t)), or None for all the keys before it (a context of at
        most budget blocks). The indexer (transformers Qwen4ExpTextQSAIndexer)
        keeps the raw key of each token. A block of `ratio` tokens gets the
        mean of its keys, the norm, and RoPE at its first position. A query
        scores each complete block with the sum over its heads of relu(q . k),
        and keeps the best budget blocks and the tail. qsa_scores gets the
        scores of the queries that drop blocks."""
        cfg = self.cfg
        t, n = h.shape[0], pos + h.shape[0]
        nh, d = cfg.indexer_heads, cfg.indexer_dim
        p = "blk.%d.indexer." % i
        cache.idx_k[i][pos:n] = h @ self.G(p + "k_proj.weight").T
        ratio = cfg.compress_ratios[i]
        budget = cfg.indexer_top_k // ratio
        if n // ratio <= budget:
            return None
        cos, sin = self.rope(np.arange(0, n))
        q = (h @ self.G(p + "q_proj.weight").T).reshape(t, nh, d)
        q = self._rot(rms_norm(q, self.G(p + "q_norm.weight"), cfg.rms_norm_eps),
                      cos[pos:n], sin[pos:n])
        nb_all = n // ratio
        pooled = cache.idx_k[i][:nb_all * ratio].reshape(nb_all, ratio, d).mean(axis=1)
        starts = np.arange(nb_all) * ratio
        kb = self._rot(rms_norm(pooled, self.G(p + "k_norm.weight"), cfg.rms_norm_eps),
                       cos[starts], sin[starts])
        mask = np.arange(n)[None, :] <= (pos + np.arange(t))[:, None]
        self.qsa_scores = {}
        for j in range(t):
            pj = pos + j
            nb = (pj + 1) // ratio
            if nb <= budget:
                continue
            score = np.maximum(q[j] @ kb[:nb].T, 0.0).sum(axis=0) / np.sqrt(d)
            self.qsa_scores[pj] = score
            # relu gives many scores of 0, so equal scores are common. Keep the
            # most recent of equal blocks (torch.topk has no fixed order).
            top = np.lexsort((-np.arange(nb), -score))[:budget]
            keep = np.zeros(n, bool)
            keep[(top[:, None] * ratio + np.arange(ratio)[None, :]).reshape(-1)] = True
            keep[nb * ratio:pj + 1] = True
            mask[j] = keep
        return mask

    def full_attention(self, i, h, cache, pos):
        """The gated attention of Qwen3.5 with the key mask of QSA."""
        cfg = self.cfg
        p = "layers.%d.self_attn." % i
        t = h.shape[0]
        nq, nk, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        sel = self.qsa_mask(i, h, cache, pos)
        qg = (h @ self.W(p + "q_proj").T).reshape(t, nq, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(t, nq * hd)
        k = (h @ self.W(p + "k_proj").T).reshape(t, nk, hd)
        v = (h @ self.W(p + "v_proj").T).reshape(t, nk, hd)
        q = rms_norm(q, self.t(p + "q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm(k, self.t(p + "k_norm.weight"), cfg.rms_norm_eps)
        cos, sin = self.rope(np.arange(pos, pos + t))
        q, k = self._rot(q, cos, sin), self._rot(k, cos, sin)
        kv_store(cache, i, k, v, pos)
        n = pos + t
        K, V = kv_rows(cache, i, n)
        rep = nq // nk
        if sel is None:
            sel = np.arange(n)[None, :] <= (pos + np.arange(t))[:, None]
        out = np.empty((t, nq, hd), np.float32)
        for hq in range(nq):
            kh = hq // rep
            s_ = (q[:, hq] @ K[kh].T) * hd ** -0.5                  # (t, n)
            s_[~sel] = -np.inf
            s_ -= s_.max(axis=-1, keepdims=True)
            w = np.exp(s_)
            w /= w.sum(axis=-1, keepdims=True)
            out[:, hq] = w @ V[kh]
        o = out.reshape(t, nq * hd) * sigmoid(gate)
        return o @ self.W(p + "o_proj").T

    # ---- the model ----

    def layer(self, i, H, ids, cache, pos):
        cfg = self.cfg
        if i in cfg.ple_layers:
            H = self.ple(i, H, ids, cache)
        x, w = self.hc_pre(H, "blk.%d.hc_attn" % i)
        if cfg.layer_types[i] == "full_attention":
            out = self.full_attention(i, x, cache, pos)
        else:
            out = self.linear_attention(i, x, cache, pos)
        H = H + out[:, None, :] * w[:, :, None]
        x, w = self.hc_pre(H, "blk.%d.hc_ffn" % i)
        out = self.moe(i, x)
        return (H + out[:, None, :] * w[:, :, None]).astype(np.float32)

    def forward(self, ids, cache, start_pos=0, hook=None):
        """Run tokens from start_pos. Return the input of the head (t x
        hidden). hook(name, value) gets the streams after each layer."""
        ids = list(ids)
        cfg = self.cfg
        x = self.embed(ids)
        H = np.repeat(x[:, None, :], cfg.hc_count, axis=1).astype(np.float32)
        for i in range(self.n_layers):
            H = self.layer(i, H, ids, cache, start_pos)
            if hook is not None:
                hook("layer.%d" % i, H)
        cache.n = start_pos + len(ids)
        self.last_streams = H
        return self.hc_pre(H, "output_hc", inject=False)
