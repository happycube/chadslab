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
- QSA attention: for a context of at most 2048 + 3 tokens the indexer keeps
  all the keys, and QSA is the full attention. This file has the full
  attention only (the indexer comes later; forward() refuses a longer
  context);
- no final norm: a last mixer (hc_pre with no inject) makes the input of
  the head.
"""
from __future__ import annotations

import numpy as np

from .gguf import open_gguf
from .qwen import QwenCache, QwenConfig, QwenGGUF, sigmoid, silu

# The keys the indexer can pass (QWEN38_PLAN.md): with more positions the
# indexer drops blocks, and this file has no indexer yet.
FULL_LIMIT = 2048 + 3


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
        if start_pos + len(ids) > FULL_LIMIT:
            raise NotImplementedError("the QSA indexer is not in this file yet: at most %d "
                                      "positions" % FULL_LIMIT)
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
