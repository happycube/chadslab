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

import os

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
        # The raw keys of the indexer of each QSA layer (one for each position),
        # and the key of each complete block (the C program keeps it).
        self.idx_k = {i: np.zeros((max_len, cfg.indexer_dim), np.float32)
                      for i, t in enumerate(cfg.layer_types) if t == "full_attention"}
        self.idx_blk = {i: np.zeros((max_len // 4 + 1, cfg.indexer_dim), np.float32)
                        for i in self.idx_k}


class Qwen4MTPCache:
    """The keys and values of the MTP layer (a dense attention layer after
    the last layer of the model), in the form of QwenCache."""

    def __init__(self, cfg, max_len=4096, kv=None):
        self.cfg, self.n, self.max_len = cfg, 0, max_len
        self.kv_form = kv or getattr(cfg, "kv_form", "f32")
        self.layer = cfg.num_hidden_layers
        per = cfg.num_kv_heads * cfg.head_dim
        if self.kv_form == "int16":
            self.kv = {self.layer: [np.zeros((max_len, per), np.int16),
                                    np.zeros((max_len, per // 32), np.float32),
                                    np.zeros((max_len, per), np.int16),
                                    np.zeros((max_len, per // 32), np.float32)]}
        else:
            shape = (cfg.num_kv_heads, max_len, cfg.head_dim)
            self.kv = {self.layer: [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]}


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

    def __init__(self, path, cfg=None, layers=None, mtp=None):
        self.path = path
        self.g = open_gguf(path)
        if mtp is not None:
            # The MTP layer (blk.<layers>) is in its own file.
            self.g.attach(mtp)
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
        ratio = cfg.compress_ratios[i] if i < len(cfg.compress_ratios) else 0
        if ratio == 0:
            return None                  # a dense layer (the MTP layer)
        cache.idx_k[i][pos:n] = h @ self.G(p + "k_proj.weight").T
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

    # ---- the MTP layer ----

    def mtp(self, Hs, ids, mcache, pos):
        """The MTP layer (llama.cpp qwen4exp.cpp, graph_mtp). Row j pairs
        the streams of the model at position pos + j - 1 (Hs: t x hc x
        hidden; zeros before position 0) with the token at pos + j. Return
        the input of the head (the prediction of the token at pos + j + 1);
        mtp_streams gets the streams of the layer (the Hs of the next draft
        step)."""
        cfg = self.cfg
        L = cfg.num_hidden_layers
        b = "blk.%d." % L
        t, hc, hid = Hs.shape
        eps = cfg.rms_norm_eps
        e = rms_norm(self.embed(ids), self.G(b + "nextn.enorm.weight"), eps)
        hn = grouped_norm(Hs.reshape(t, hc * hid), self.G(b + "nextn.hnorm.weight"), hid, eps)
        # For each stream: the embedding, then the stream (ggml_concat of
        # enorm and hnorm), then eh_proj.
        cat = np.concatenate([np.repeat(e[:, None, :], hc, axis=1), hn.reshape(t, hc, hid)],
                             axis=-1)
        R = (cat @ self.G(b + "nextn.eh_proj.weight").T).astype(np.float32)
        x, w = self.hc_pre(R, b + "hc_attn")
        R = R + self.full_attention(L, x, mcache, pos)[:, None, :] * w[:, :, None]
        x, w = self.hc_pre(R, b + "hc_ffn")
        R = (R + self.moe(L, x)[:, None, :] * w[:, :, None]).astype(np.float32)
        mcache.n = pos + t
        self.mtp_streams = R
        return self.hc_pre(R, b + "nextn.hc_head", inject=False)


# ---- the CPU path: the step as a program of records (QWEN38_PLAN.md, phase 2) ----

def compile_qwen4_step(model, t, verify=False, mtp=False):
    """Compile a step of t tokens of a Qwen4CPU into a program of records.

    The inputs are names["H"] (t x hc * hid: the embeddings in each stream)
    and names["ple"] (t x hid: the rows of the n-gram table). The output is
    names["xn"], the input of the head. The products are KQ_QUANT and
    KQ_LINEAR (csrc/kquants.c); the gated residual and the n-gram layer are
    the records of csrc/hyperconn.c.
    The QSA layers have QSA_SELECT (the indexer) and ATTN_QSA (csrc/qsa.c).

    verify makes an MTP verify group: the GDN records write a log
    (names["log.<layer>"]) and do not change the state; Qwen4CPU.commit
    applies the first n tokens. names["gn.<layer>"] has the inputs of the
    convolution of each n-gram layer, for the commit of its state.

    mtp compiles the MTP layer (Qwen4.mtp) in place of the model: the
    inputs are names["e"] (t x hid: the embeddings) and names["h"] (t x
    hc * hid: the streams of the model); names["H"] gets the streams of the
    layer, and names["xn"] the input of the head."""
    from . import cops
    from . import program as P
    cfg = model.cfg
    prog = P.Program()
    hid, hc, eps = cfg.hidden_size, cfg.hc_count, float(cfg.rms_norm_eps)
    HD, lr = hc * hid, cfg.hc_lowrank
    nq, nk, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    kd, vd, cd = cfg.lin_key_dim, cfg.lin_value_dim, cfg.conv_dim
    k, E, inner = cfg.top_k, cfg.num_experts, cfg.moe_inter
    f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
    H, ple, xn = f32(t, HD), f32(t, hid), f32(t, hid)
    hn, g, lo, loa, mixed, inj = f32(t, HD), f32(t, HD), f32(t, lr), f32(t, lr), f32(t, hid), f32(t, hc)
    wide = max(HD, cd, nq * 2 * hd, vd, nq * hd, 2 * HD if mtp else 0)
    xq, xs, xm = np.zeros((t, wide), np.int8), f32(t, wide // 32), f32(t, wide // 16)
    o1, o2, o3, o4, o5 = f32(t, wide), f32(t, wide), f32(t, wide), f32(t, wide), f32(t, wide)
    att, gate, qout, kbuf = f32(t, nq * hd), f32(t, nq * hd), f32(t, nq * hd), f32(t, nk * hd)
    ratio = lambda i: cfg.compress_ratios[i] if i < len(cfg.compress_ratios) else 0  # noqa: E731
    budget = lambda i: cfg.indexer_top_k // cfg.compress_ratios[i]  # noqa: E731
    r0 = max(r for r in cfg.compress_ratios if r)
    maxsel = cfg.indexer_top_k + r0 - 1
    iq, ik = f32(t, cfg.indexer_heads * cfg.indexer_dim), f32(t, cfg.indexer_dim)
    sel, cnt = np.zeros((t, maxsel), np.int32), np.zeros(t, np.int32)
    dense = np.full(t, -1, np.int32)      # cnt of a dense layer: all the positions
    qscr = prog.slot("qsa_scratch")
    keyn, qn, gated = f32(t, HD), f32(t, HD), f32(t, HD)
    mo, logits = f32(t, hid), f32(t, E)
    val, idx, slog = f32(t, k), np.zeros((t, k), np.int32), f32(t, 1)
    scratch = cops.kq_moe_scratch(t, k, E, hid, inner)
    gscr = f32(t * cd + (cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim if verify else 0))
    prog.names.update(H=H, ple=ple, xn=xn)
    pos, cos, sin, scores = prog.slot("pos"), prog.slot("cos"), prog.slot("sin"), prog.slot("scores")
    cur = {}

    def quant(src, cols):
        prog.emit(P.KQ_QUANT, src, t, cols, xq, xs, xm)
        cur["src"] = src

    def lin(gname, out, src=None):
        m = model.K(gname)
        prog.emit(P.KQ_LINEAR, xq, xs, xm, cur["src"] if src is None else src, m.data, m.type,
                  m.rows, m.cols, t, out)

    def scalar(op, a, b):
        r = prog.temp()
        prog.emit(op, r, a, b)
        return r

    def hc_pre(prefix, out, inject=True):
        prog.emit(P.HC_NORM, H, model.F(prefix + "_norm.weight"), hn, t, hc, hid, eps)
        quant(hn, HD)
        lin(prefix + "_down.weight", lo)
        prog.emit(P.HC_ACT, lo, loa, t * lr, 1.0 / hc)
        quant(loa, lr)
        lin(prefix + "_up.weight", g)
        prog.emit(P.HC_MIX, hn, g, out, t, hc, hid)
        if inject:
            lin(prefix + "_inject.weight", inj, src=hn)

    def attention(i):
        b = "blk.%d." % i
        lin(b + "attn_q.weight", o1)
        lin(b + "attn_k.weight", o2)
        lin(b + "attn_v.weight", o3)
        prog.emit(P.ATTN_PREP, o1, o2, o3, model.F(b + "attn_q_norm.weight"),
                  model.F(b + "attn_k_norm.weight"), cos, sin, None, None, 0, pos, t, nq, nk, hd,
                  cfg.rotary_dim, eps, float(hd ** -0.5), qout, gate, kbuf)
        base = [prog.slot("%s.%d" % (nm, i)) for nm in ("kq", "ks", "vq", "vs")]
        per = nk * hd
        rows = [scalar(P.S_ADD, bs, scalar(P.S_MUL, pos, step))
                for bs, step in zip(base, (2 * per, per // 8, 2 * per, per // 8))]
        prog.emit(P.KV_WRITE, kbuf, o3, None, None, *rows, t * per)
        if ratio(i) == 0:
            # A dense layer (the MTP layer).
            prog.emit(P.ATTN_QSA, qout, *base, scores, att, nq, nk, hd, t, pos, sel, dense,
                      maxsel)
            prog.emit(P.SIGMUL, att, gate, att, t * nq * hd)
            quant(att, nq * hd)
            lin(b + "attn_output.weight", o5)
            return
        # QSA: the indexer selects the keys of each query (csrc/qsa.c).
        lin(b + "indexer.q_proj.weight", iq)
        lin(b + "indexer.k_proj.weight", ik)
        prog.emit(P.QSA_SELECT, iq, ik, prog.slot("idxk.%d" % i), prog.slot("blk.%d" % i),
                  model.F(b + "indexer.q_norm.weight"), model.F(b + "indexer.k_norm.weight"),
                  cos, sin, pos, t, cfg.indexer_heads, cfg.indexer_dim, ratio(i), budget(i),
                  cfg.rotary_dim, float(cfg.rope_theta), eps, sel, cnt, maxsel, qscr,
                  prog.slot("nbmax"))
        prog.emit(P.ATTN_QSA, qout, *base, scores, att, nq, nk, hd, t, pos, sel, cnt, maxsel)
        prog.emit(P.SIGMUL, att, gate, att, t * nq * hd)
        quant(att, nq * hd)
        lin(b + "attn_output.weight", o5)

    def deltanet(i):
        b = "blk.%d." % i
        lin(b + "attn_qkv.weight", o1)
        lin(b + "attn_gate.weight", o2)
        lin(b + "ssm_beta.weight", o3)
        lin(b + "ssm_alpha.weight", o4)
        from .qwen import gdn_flags
        log = None
        if verify:
            log = prog.names["log.%d" % i] = np.zeros(cops.gdn_log_floats(
                t, cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim), np.float32)
        prog.emit(P.GDN, o1, prog.slot("conv.%d" % i), model.F(b + "ssm_conv1d.weight", (cd, cfg.conv_kernel)),
                  cfg.conv_kernel, o2, o4, o3, model.A_log(i), model.F(b + "ssm_dt.bias"),
                  model.F(b + "ssm_norm.weight"), prog.slot("S.%d" % i), att, gscr, t,
                  cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim, eps, log,
                  gdn_flags(cfg), prog.slot("nreal"))
        quant(att, vd)
        lin(b + "ssm_out.weight", o5)

    def ple_layer(i):
        b = "blk.%d.ple_" % i
        gn = f32(t, HD)
        quant(ple, hid)
        lin(b + "key.weight", o1)
        lin(b + "value.weight", o2)
        prog.emit(P.HC_NORM, o1, model.F(b + "norm_key.weight"), keyn, t, hc, hid, eps)
        prog.emit(P.HC_NORM, H, model.F(b + "norm_query.weight"), qn, t, hc, hid, eps)
        prog.emit(P.PLE_GATE, keyn, qn, o2, gated, t, hc, hid)
        prog.emit(P.HC_NORM, gated, model.F(b + "norm_conv.weight"), gn, t, hc, hid, eps)
        prog.names["gn.%d" % i] = gn
        prog.emit(P.PLE_CONV, gn, gated, H, prog.slot("pleconv.%d" % i),
                  model.F(b + "conv1d.weight", (HD, cfg.ple_conv_kernel)), t, HD,
                  cfg.ple_conv_kernel, cfg.ple_ngram)

    layers, head = range(model.n_layers), "output_hc"
    if mtp:
        L = cfg.num_hidden_layers
        b = "blk.%d.nextn." % L
        e, hin, en, cat = f32(t, hid), f32(t, HD), f32(t, hid), f32(t, 2 * HD)
        prog.names.update(e=e, h=hin)
        prog.emit(P.RMS_NORM, e, model.F(b + "enorm.weight"), en, t, hid, eps)
        prog.emit(P.HC_NORM, hin, model.F(b + "hnorm.weight"), hn, t, hc, hid, eps)
        prog.emit(P.HC_CAT, en, hn, cat, t, hc, hid)
        # eh_proj on each stream: t * hc rows of 2 hid values.
        prog.emit(P.KQ_QUANT, cat, t * hc, 2 * hid, xq, xs, xm)
        m = model.K(b + "eh_proj.weight")
        prog.emit(P.KQ_LINEAR, xq, xs, xm, cat, m.data, m.type, m.rows, m.cols, t * hc, H)
        layers, head = [L], b + "hc_head"
    for i in layers:
        b = "blk.%d." % i
        if i in cfg.ple_layers:
            ple_layer(i)
        hc_pre(b + "hc_attn", mixed)
        quant(mixed, hid)
        if i >= len(cfg.layer_types) or cfg.layer_types[i] == "full_attention":
            attention(i)
        else:
            deltanet(i)
        prog.emit(P.HC_ADD, H, o5, inj, t, hc, hid, 1.0 / hc)
        hc_pre(b + "hc_ffn", mixed)
        quant(mixed, hid)
        lin(b + "ffn_gate_inp.weight", logits)
        lin(b + "ffn_gate_inp_shexp.weight", slog)
        prog.emit(P.ROUTER_TOPK, logits, t, E, k, val, idx)
        mats = cops.kq_moe_mats(*(model.K(b + "ffn_%s_exps.weight" % n).c()
                                  for n in ("gate", "up", "down")),
                                [model.K(b + "ffn_%s_shexp.weight" % n).c()
                                 for n in ("gate", "up", "down")])
        prog.emit(P.KQ_MOE, xq, xs, xm, idx, val, t, k, E, mats, slog, hid, inner, scratch, mo)
        prog.keep.append(mats)
        prog.emit(P.HC_ADD, H, mo, inj, t, hc, hid, 1.0 / hc)
    hc_pre(head, xn, inject=False)
    prog.keep.append(dense)
    prog.tokens = t
    return prog.finish()


class Qwen4CPU(Qwen4):
    """Qwen3.8-Flash-Next with the step as one program of records (the C
    kernels of cops). The weights stay in the blocks of the file."""

    CHUNK = 512

    def __init__(self, path, cfg=None, layers=None, mtp=None):
        super().__init__(path, cfg, layers, mtp)
        self._k = {}
        self._f = {}
        self.programs = {}

    def K(self, gname):
        from .qwen import KMat
        m = self._k.get(gname)
        if m is None:
            m = KMat(self.g, gname)
            if m.type == 30:
                # BF16 (the indexer): the products take it as F32.
                a = self.G(gname)
                m.data, m.type = np.ascontiguousarray(a, np.float32).view(np.uint8).reshape(-1), 0
            self._k[gname] = m
        return m

    def F(self, gname, shape=None):
        a = self._f.get((gname, shape))
        if a is None:
            a = np.ascontiguousarray(self.G(gname), dtype=np.float32)
            a = self._f[(gname, shape)] = a.reshape(shape) if shape is not None else a
        return a

    def A_log(self, i):
        key = ("A_log", i)
        a = self._f.get(key)
        if a is None:
            a = self._f[key] = np.log(-self.G("blk.%d.ssm_a" % i)).astype(np.float32)
        return a

    def embed(self, ids):
        from . import cops
        m = self.K("token_embd.weight")
        return cops.kq_rows(m.data, m.type, m.cols, ids)

    def logits(self, h, chunk=None):
        from . import cops
        from .qwen import KX
        m, qx = self.K("output.weight"), KX(h)
        out = np.empty((qx.t, m.rows), np.float32)
        cops.kq_linear(m.data, m.type, m.rows, m.cols, qx.xq, qx.xs, qx.xm, qx.x, qx.t, out)
        return out

    def ple_rows(self, ids, cache):
        """The 16 rows of the n-gram table of each token (t x hid)."""
        from . import cops
        rows = self.ple_ids(ids, cache)
        m = self.K("per_layer_token_embd.weight")
        return cops.kq_rows(m.data, m.type, m.cols, rows.reshape(-1)).reshape(len(ids), -1)

    def program(self, t, kind=None):
        """The program of t tokens: a step (kind None), an MTP verify group
        ("verify"), or the MTP layer ("mtp")."""
        key = t if kind is None else (kind, t)
        prog = self.programs.get(key)
        if prog is None:
            prog = self.programs[key] = compile_qwen4_step(
                self, t, verify=kind == "verify", mtp=kind == "mtp")
        return prog

    def _bind(self, prog, cache, pos, t):
        from .qwen import cache_params, scores_buffer
        cfg = self.cfg
        cos, sin = self.rope(np.arange(pos, pos + t))
        kw = {"pos": pos, "nreal": t, "cos": np.ascontiguousarray(cos),
              "sin": np.ascontiguousarray(sin), "scores": scores_buffer(cfg, pos + t)}
        kw.update(cache_params(self, cache))
        for i in cfg.ple_layers:
            kw["pleconv.%d" % i] = cache.ple_conv[i]
        for i, a in cache.idx_k.items():
            kw["idxk.%d" % i], kw["blk.%d" % i] = a, cache.idx_blk[i]
        nbmax = cache.max_len // min(r for r in cfg.compress_ratios if r) + 1
        need = (os.cpu_count() or 1) * nbmax * 12
        if getattr(self, "_qscr", None) is None or self._qscr.size < need:
            self._qscr = np.zeros(need, np.uint8)
        kw["qsa_scratch"], kw["nbmax"] = self._qscr, nbmax
        prog.bind(**{k: v for k, v in kw.items() if k in prog.by_name})

    def _run(self, prog, ids, cache, pos):
        cfg = self.cfg
        self._bind(prog, cache, pos, len(ids))
        x = self.embed(ids)
        prog.names["H"][:] = np.repeat(x[:, None, :], cfg.hc_count, axis=1).reshape(len(ids), -1)
        if cfg.ple_layers:
            prog.names["ple"][:] = self.ple_rows(ids, cache)
        prog.run()

    def forward(self, ids, cache, start_pos=0, hook=None, streams=False):
        """Run tokens from start_pos. Return the input of the head (t x
        hidden); with streams, also the streams after the last layer (t x
        hc * hidden, the input of the MTP layer)."""
        ids = list(ids)
        out, hs = [], []
        c0 = 0
        while c0 < len(ids):
            chunk = ids[c0:c0 + self.CHUNK]
            prog = self.program(len(chunk))
            self._run(prog, chunk, cache, start_pos + c0)
            out.append(prog.names["xn"].copy())
            if streams:
                hs.append(prog.names["H"].copy())
            c0 += len(chunk)
        cache.n = start_pos + len(ids)
        if streams:
            return np.concatenate(out), np.concatenate(hs)
        return np.concatenate(out)

    # ---- MTP: the verify group and the MTP layer (QWEN38_PLAN.md, phase 3) ----

    def verify(self, ids, cache, start_pos):
        """Run a group of tokens from start_pos as an MTP verify group.
        Return the input of the head and the streams of each token. The
        keys (and the keys of the indexer) are written for all the tokens;
        the state of the linear layers and of the n-gram layer does not
        change until commit(n)."""
        cfg = self.cfg
        prog = self.program(len(ids), "verify")
        snap = (cache.ple_ids.copy(), {i: a.copy() for i, a in cache.ple_conv.items()})
        self._run(prog, list(ids), cache, start_pos)
        self._pending = (prog, cache, start_pos, list(ids), snap)
        return prog.names["xn"].copy(), prog.names["H"].copy()

    def commit(self, n):
        """Keep the first n tokens of the last verify group."""
        from . import cops
        prog, cache, start, ids, (ple_ids, ple_conv) = self._pending
        cfg = self.cfg
        for i in range(self.n_layers):
            if cfg.layer_types[i] != "full_attention":
                cops.gdn_commit(cache.conv[i], cache.state[i], prog.names["log.%d" % i], n,
                                cfg.conv_kernel, cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim,
                                cfg.lin_v_dim)
        ctx = cfg.ple_ngram - 1
        cache.ple_ids = np.concatenate([ple_ids, np.asarray(ids[:n], np.int64)])[-ctx:].copy()
        for i, old in ple_conv.items():
            hist = old.shape[0]
            cache.ple_conv[i][:] = np.concatenate([old, prog.names["gn.%d" % i][:n]])[-hist:]
        cache.n = start + n
        self._pending = None

    def mtp_step(self, Hs, ids, mcache, pos):
        """The MTP layer as a program (Qwen4.mtp). Return the input of the
        head and the streams of the layer."""
        from .qwen import scores_buffer
        cfg = self.cfg
        t = len(ids)
        prog = self.program(t, "mtp")
        L = cfg.num_hidden_layers
        cos, sin = self.rope(np.arange(pos, pos + t))
        kw = {"pos": pos, "nreal": t, "cos": np.ascontiguousarray(cos),
              "sin": np.ascontiguousarray(sin), "scores": scores_buffer(cfg, pos + t)}
        for nm, a in zip(("kq", "ks", "vq", "vs"), mcache.kv[L]):
            kw["%s.%d" % (nm, L)] = a
        prog.bind(**{k: v for k, v in kw.items() if k in prog.by_name})
        prog.names["e"][:] = self.embed(ids)
        prog.names["h"][:] = Hs.reshape(t, -1)
        prog.run()
        mcache.n = pos + t
        return prog.names["xn"].copy(), prog.names["H"].copy()

    def generate_mtp(self, prompt, n_new, draft=3, max_len=8192, stop=None, stats=None):
        """Greedy generation with the MTP layer as the drafter. Each round:
        the MTP layer on the rows it has not seen (the accepted tokens with
        the streams of the model) and the first draft in one group, the
        next drafts on the streams of the MTP layer, then one verify group
        of the model. Return the new tokens."""
        import time
        cfg = self.cfg
        cache = Qwen4Cache(cfg, max_len)
        mcache = Qwen4MTPCache(cfg, max_len)
        prompt = list(prompt)
        n = len(prompt)
        xn, H = self.forward(prompt, cache, 0, streams=True)
        tok = int(np.argmax(self.logits(xn[-1:])[0]))
        out, pos = [tok], n                # tok is at pos
        # The rows of the MTP layer before the first draft: token j with the
        # streams of j - 1 (zeros for token 0), up to (tok, the streams at n - 1).
        m_ids = prompt + [tok]
        m_H = np.concatenate([np.zeros((1, H.shape[1]), np.float32), H])
        m_pos = 0
        rounds = acc_total = 0
        tm = {"draft": 0.0, "verify": 0.0}
        t_start = time.time()
        while len(out) < n_new and (stop is None or tok not in stop):
            t0 = time.time()
            for c0 in range(0, len(m_ids), self.CHUNK):
                c1 = min(len(m_ids), c0 + self.CHUNK)
                xm, hm = self.mtp_step(m_H[c0:c1], m_ids[c0:c1], mcache, m_pos + c0)
            drafts = []
            while True:
                drafts.append(int(np.argmax(self.logits(xm[-1:])[0])))
                if len(drafts) == draft:
                    break
                xm, hm = self.mtp_step(hm[-1:], drafts[-1:], mcache, pos + len(drafts))
            t1 = time.time()
            # Verify tok and the drafts.
            xv, Hv = self.verify([tok] + drafts, cache, pos)
            best = np.argmax(self.logits(xv), axis=1)
            a = 0
            while a < draft and best[a] == drafts[a]:
                a += 1
            self.commit(a + 1)
            tok = int(best[a])
            out.extend(drafts[:a] + [tok])
            # The keys of the MTP layer at pos + 1 .. pos + a came from its own
            # streams: make them again with the streams of the model.
            m_ids, m_H, m_pos = drafts[:a] + [tok], Hv[:a + 1], pos + 1
            tm["draft"] += t1 - t0
            tm["verify"] += time.time() - t1
            rounds += 1
            acc_total += a
            pos += a + 1
        if stats is not None:
            stats.update(rounds=rounds, accepted=acc_total, drafted=rounds * draft,
                         decode_s=time.time() - t_start, **tm)
        return out[:n_new]
