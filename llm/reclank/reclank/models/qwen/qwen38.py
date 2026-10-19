"""Qwen3.8-Flash-Next (qwen4exp, a preview of Qwen4), in NumPy.

QWEN38_PLAN.md, phase 1. This is the reference of this runtime for the
model. It reads the GGUF file of Unsloth with open_gguf (split files). It
dequantizes each weight when it uses it. transformers (models/qwen4_exp)
is the reference for it; scripts/check_qwen4.py compares them.

Some parts of Qwen3.6 stay (reclank/models/qwen/qwen35.py, QwenGGUF):

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

from reclank.formats.gguf import open_gguf
from reclank.models.qwen.qwen35 import QUANT_FORMS, QwenCache, QwenConfig, QwenGGUF, kv_arrays, kv_row_bytes, kv_rows, kv_store, layer_kv_form, media_inputs, rms_norm, rope_positions, sigmoid, silu



def dense_mode(g, dense=None):
    """The form of the large bfloat16 matrices of a model (the NVFP4
    checkpoint and its GGUF): "bf16" (as they are) or "q8" (requantized to
    Q8_0 at the first use: half the bytes to read, for the CPU and for a
    GPU of little memory). dense, else RECLANK_DENSE, else "q8". q8 is
    also faster on a GPU with room for bf16: the products of a prompt use
    the tensor cores, and the buffer of the expert copies keeps more room
    (a prompt of 32K: 351 tok/s with bf16; 131072 tokens: 424 with q8).
    "auto": q8 when a GPU is there and its free memory is less than the
    bfloat16 dense part, the head, and 3 GB; else bf16. The BF12 matrices
    of a file of convert_q8_gguf.py --dense bf12 stay BF12 unless
    RECLANK_DENSE is q8, rq8 or bf16 (Qwen4CPU.K; rq8: rotated Q8_0, the
    other bfloat16 matrices as q8)."""
    if getattr(g, "kept", False) or getattr(g, "as_stored", False):
        # the weightkeeper chose the types (its least ones), or a checkpoint whose
        # types are its contract (FP4-Local, fp4local_st): as they are
        return "bf16"
    # an EXL3 directory (exl3_qwen4.EXL3Source): its few bfloat16 matrices as they are
    dense = dense or os.environ.get("RECLANK_DENSE", "bf16" if getattr(g, "exl3", False) else "q8")
    if dense != "auto":
        return dense
    from reclank.formats.gguf import tensor_bytes
    big = [(n, d) for n, (d, t, _o) in g.tensors.items()
           if t == 30 and len(d) == 2 and d[1] > 1 and n != "token_embd.weight"]
    if not big:
        return "bf16"
    try:
        from reclank.backends.cuda import runtime as gpu
        from reclank.backends.cuda import gpumm
        if not gpu.available():
            return "bf16"
        free = gpumm.mem_info()[0]
    except Exception:
        return "bf16"
    need = sum(tensor_bytes(d, 30) for _n, d in big) + 3e9
    return "q8" if free < need else "bf16"


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



def mrope_sec(cfg):
    """The M-RoPE sections of the row and the column for QSA_SELECT: s1 |
    s2 << 8 (transformers mrope_section [11, 11, 10])."""
    sec = getattr(cfg, "mrope_section", None) or [11, 11, 10]
    return int(sec[1]) | int(sec[2]) << 8

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
        # and the key of each complete block (the C program keeps it). float16:
        # the scores only rank the blocks.
        self.idx_k = {i: np.zeros((max_len, cfg.indexer_dim), np.float16)
                      for i, t in enumerate(cfg.layer_types) if t == "full_attention"}
        self.idx_blk = {i: np.zeros((max_len // 4 + 1, cfg.indexer_dim), np.float16)
                        for i in self.idx_k}
        # The M-RoPE positions of the rows as (max_len, 3) int32 for the keys
        # of the blocks of QSA_SELECT (None: the row index); qpos_version
        # counts the changes (the GPU copy).
        self.qpos = None
        self.qpos_version = 0

    def set_rope(self, pos):
        super().set_rope(pos)
        if self.rpos is not None:
            if self.qpos is None:
                self.qpos = np.empty((self.max_len, 3), np.int32)
            self.qpos[:] = self.rpos.T
            self.qpos_version += 1


class Qwen4MTPCache:
    """The keys and values of the MTP layer (a dense attention layer after
    the last layer of the model), in the form of QwenCache."""

    def __init__(self, cfg, max_len=4096, kv=None):
        self.cfg, self.n, self.max_len = cfg, 0, max_len
        self.kv_form = kv or getattr(cfg, "kv_form", "f32")
        self.layer = cfg.num_hidden_layers
        per = cfg.num_kv_heads * cfg.head_dim
        if self.kv_form in QUANT_FORMS:
            lf = layer_kv_form(self.kv_form, cfg, self.layer)
            self.layer_form = {self.layer: lf}
            self.kv = {self.layer: kv_arrays(lf, max_len, per)}
        else:
            shape = (max_len, cfg.num_kv_heads, cfg.head_dim)      # position-major
            self.layer_form = {}
            self.kv = {self.layer: [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]}


def grouped_norm(x, w, hidden, eps):
    """RMS norm of each group of hidden values of the last axis, times w (the
    weights of the file have the 1 in them)."""
    s = x.reshape(*x.shape[:-1], -1, hidden)
    s = s / np.sqrt(np.mean(s * s, axis=-1, keepdims=True) + eps)
    return (s.reshape(x.shape) * w).astype(np.float32)


def default_experts(path):
    """The experts when none are given (--experts, RECLANK_EXPERTS): the one file
    *nvfp4-experts*.gguf next to the model (scripts/convert_nvfp4_gguf.py
    --experts-only), else those of the model. With the MIX-BF12 file on the
    3090 (a chat reply of 256 tokens, plain and MTP 3 drafts with
    RECLANK_MTP_STOP 0.9, the parts): 35.0 and 49.0 tok/s with its own
    experts, 40.9 and 59.4 with the NVFP4 ones (NLL of real text 2.0289 and
    2.0297); their shared experts (bfloat16) load as BF12 (Qwen4CPU.K).
    "none" keeps the experts of the model file."""
    import glob
    from reclank.formats.nvfp4_st import is_checkpoint
    if is_checkpoint(path) or os.path.isdir(path):
        return None
    d = os.path.dirname(os.path.abspath(path))
    found = [f for f in sorted(glob.glob(os.path.join(d, "*nvfp4-experts*.gguf")))
             if os.path.abspath(f) != os.path.abspath(path)]
    if len(found) != 1:
        if found:
            print("%d files *nvfp4-experts*.gguf next to %s: the experts of the model (give --experts)" % (
                len(found), os.path.basename(path)), flush=True)
        return None
    print("the experts of %s (the NVFP4 experts next to the model; --experts none: those of the model)" %
          os.path.basename(found[0]), flush=True)
    return found


class Qwen4(QwenGGUF):
    """The NumPy model of Qwen3.8-Flash-Next on a GGUF file.

        m = Qwen4(path)
        cache = Qwen4Cache(m.cfg)
        h = m.forward(ids, cache)            # the input of the head
        logits = m.logits(h[-1:])
    """

    def __init__(self, path, cfg=None, layers=None, mtp=None, experts=None):
        """experts: a file (or list) of experts in another form that takes
        the place of those of path (scripts/convert_q8_gguf.py
        --experts-only; RECLANK_EXPERTS when None)."""
        from reclank.formats.nvfp4_st import NVFP4Source, is_checkpoint
        self.path = path
        if experts is None and os.environ.get("RECLANK_EXPERTS"):
            experts = os.environ["RECLANK_EXPERTS"].split(",")
        if experts is None:
            experts = default_experts(path)
        if experts in ("none", ["none"]):
            experts = None          # (--experts none: the experts of the model file)
        if isinstance(experts, str):
            # several files (a,b): the later ones over the earlier (GGUFOverlay),
            # e.g. the NVFP4 experts, then the BF12 experts of the MTP layer
            experts = [e for e in experts.split(",") if e]
        from reclank.models.qwen import keep
        from reclank.formats.exl3_qwen4 import EXL3Source, is_qwen4_exl3
        from reclank.formats.fp4local_st import FP4LSource, is_fp4local
        if is_qwen4_exl3(path):
            # an EXL3 directory of ExLlamaV3 (reclank/formats/exl3_qwen4.py), from
            # the weightkeeper when it holds it (keep_exl3.KeptEXL3)
            if keep.holds(path):
                from reclank.models.qwen.keep_exl3 import KeptEXL3
                self.g = KeptEXL3(path)
            else:
                self.g = EXL3Source(path)
            self.cfg = cfg or self.g.config()
        elif path.endswith(".gguf") and keep.holds(path):
            # a GGUF file that the weightkeeper holds: its tensors from there
            self.g = keep.KeptGGUF(path)
            self.cfg = cfg or config_from_gguf(self.g)
        elif is_checkpoint(path) and keep.is_orig(path):
            # The original bfloat16 checkpoint: its tensors from the weightkeeper
            # (reclank/models/qwen/keep.py; scripts/weightkeeper.py CKPT)
            if not keep.holds(path):
                raise SystemExit("%s: the original bfloat16 checkpoint runs from the weightkeeper: "
                                 "start it first (scripts/weightkeeper.py %s)" % (path, path))
            self.g = keep.KeptGGUF(path)
            self.cfg = cfg or config_from_gguf(self.g)
        elif experts and not is_checkpoint(path):
            from reclank.formats.gguf import GGUFOverlay
            self.g = GGUFOverlay(path, experts)
            if mtp is not None:
                self.g.attach(mtp)
            self.cfg = cfg or config_from_gguf(self.g)
        elif is_fp4local(path):
            # nvidia/Qwen3.8-Flash-Next-FP4-Local (reclank/formats/fp4local_st.py:
            # the CSF experts, MXFP8, the NVFP4 n-gram table; docs/FP4LOCAL_PLAN.md),
            # from the weightkeeper when it holds it (keep_fp4l.KeptFP4L)
            if keep.holds(path):
                from reclank.models.qwen.keep_fp4l import KeptFP4L
                self.g = KeptFP4L(path)
            else:
                self.g = FP4LSource(path)
            if os.environ.get("RECLANK_FP4L_MTP"):
                # another file's MTP layer (the BF12 GGUF's blk.48) as the drafter (plan D8)
                from reclank.formats.fp4local_st import attach_mtp
                attach_mtp(self.g, os.environ["RECLANK_FP4L_MTP"])
            self.cfg = cfg or self.g.config()
        elif is_checkpoint(path):
            # The safetensors checkpoint of ModelOpt (reclank/formats/nvfp4_st.py); it
            # has the MTP layer.
            self.g = NVFP4Source(path)
            self.cfg = cfg or self.g.config()
        else:
            self.g = open_gguf(path)
            if mtp is not None:
                # The MTP layer (blk.<layers>) is in its own file.
                self.g.attach(mtp)
            self.cfg = cfg or config_from_gguf(self.g)
        self.n_layers = layers or self.cfg.num_hidden_layers
        self._deq = {}
        # FP4-Local's FP8 cache: the int8 cache holds E4M3 codes (a state of the process)
        cops_ = __import__("reclank.backends.cpu.cops", fromlist=["cops"])
        cops_.set_kv_e4m3(getattr(self.cfg, "kv_form", "") == "e4m3")
        from reclank.models.qwen.qwen35 import gguf_name
        self.gname = gguf_name          # (QwenGGUF.W: the DeltaNet of qwen35 reads its matrices by it)

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
        if getattr(cfg, "ple_bf16", False):
            from reclank.formats.fp4local_st import bf16r
            emb = bf16r(emb)            # (FP4-Local: the rows rounded to bfloat16, as vLLM)
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
        cos, sin = self.rope(rope_positions(cache, 0, n))
        q = (h @ self.G(p + "q_proj.weight").T).reshape(t, nh, d)
        q = self._rot(rms_norm(q, self.G(p + "q_norm.weight"), cfg.rms_norm_eps),
                      cos[pos:n], sin[pos:n])
        nb_all = n // ratio
        pooled = cache.idx_k[i][:nb_all * ratio].astype(np.float32).reshape(
            nb_all, ratio, d).mean(axis=1)
        starts = np.arange(nb_all) * ratio
        kb = self._rot(rms_norm(pooled, self.G(p + "k_norm.weight"), cfg.rms_norm_eps),
                       cos[starts], sin[starts])
        kb = kb.astype(np.float16).astype(np.float32)     # the block keys of the C program
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
        cos, sin = self.rope(rope_positions(cache, pos, t))
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

    def forward(self, ids, cache, start_pos=0, hook=None, media=None):
        """Run tokens from start_pos. Return the input of the head (t x
        hidden). hook(name, value) gets the streams after each layer.
        media: spans (cache positions) whose rows replace the embeddings."""
        ids = list(ids)
        cfg = self.cfg
        x = media_inputs(self.embed(ids), start_pos, media)
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
    KQ_LINEAR (backends/cpu/csrc/kernels/kq); the gated residual and the n-gram layer are
    the records of backends/cpu/csrc/common/hyperconn.c.
    The QSA layers have QSA_SELECT (the indexer) and ATTN_QSA (backends/cpu/csrc/common/qsa.c).

    verify makes an MTP verify group: the GDN records write a log
    (names["log.<layer>"]) and do not change the state; Qwen4CPU.commit
    applies the first n tokens. names["gn.<layer>"] has the inputs of the
    convolution of each n-gram layer, for the commit of its state.

    mtp compiles the MTP layer (Qwen4.mtp) in place of the model: the
    inputs are names["e"] (t x hid: the embeddings) and names["h"] (t x
    hc * hid: the streams of the model); names["H"] gets the streams of the
    layer, and names["xn"] the input of the head."""
    from reclank.backends.cpu import cops
    from reclank.ir import program as P
    cfg = model.cfg
    prog = P.Program()
    hid, hc, eps = cfg.hidden_size, cfg.hc_count, float(cfg.rms_norm_eps)
    HD, lr = hc * hid, cfg.hc_lowrank
    nq, nk, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    kd, vd, cd = cfg.lin_key_dim, cfg.lin_value_dim, cfg.conv_dim
    k, E, inner = cfg.top_k, cfg.num_experts, cfg.moe_inter
    f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
    H, ple, xn = f32(t, HD), f32(t, hid), f32(t, hid)
    wide = max(HD, cd, nq * 2 * hd, vd, nq * hd, 2 * HD if mtp else 0)
    xq, xs, xm = np.zeros((t, wide), np.int8), f32(t, wide // 32), f32(t, wide // 16)
    ratio = lambda i: cfg.compress_ratios[i] if i < len(cfg.compress_ratios) else 0  # noqa: E731
    budget = lambda i: cfg.indexer_top_k // cfg.compress_ratios[i]  # noqa: E731
    r0 = max(r for r in cfg.compress_ratios if r)
    maxsel = cfg.indexer_top_k + r0 - 1
    dense = np.full(t, -1, np.int32)      # cnt of a dense layer: all the positions
    qscr = prog.slot("qsa_scratch")
    scratch = model.moe_scratch4(t)
    # model.pack_temps (the GPU, ir/alloc.pack): each layer has buffers of
    # its own (prog.temps), which the packer then places by liveness; else
    # every layer reuses the same ones (as compile_qwen_step)
    fresh = getattr(model, "pack_temps", False)
    rot_x = getattr(model, "rot_experts", False)

    def layer_buffers():
        def tmp(*sh, dtype=np.float32):
            a = P.lazy_zeros(sh, dtype)
            if fresh:
                prog.temps.append(a)
            return a
        return (tmp(t, HD), tmp(t, HD), tmp(t, lr), tmp(t, lr), tmp(t, hid), tmp(t, hc),
                tmp(t, wide), tmp(t, wide), tmp(t, wide), tmp(t, wide), tmp(t, wide),
                tmp(t, nq * hd), tmp(t, nq * hd), tmp(t, nq * hd), tmp(t, nk * hd),
                tmp(t, cfg.indexer_heads * cfg.indexer_dim), tmp(t, cfg.indexer_dim),
                tmp(t, maxsel, dtype=np.int32), tmp(t, dtype=np.int32),
                tmp(t, HD), tmp(t, HD), tmp(t, HD), tmp(t, hid), tmp(t, E),
                # RQ8_0 experts (model.rot_experts): the experts read x rotated
                # in each 32 values (gguf.RQ8_0); the routers read x
                tmp(t, hid) if rot_x else None,
                tmp(t, k), tmp(t, k, dtype=np.int32), tmp(t, 1),
                tmp(t * cd + (cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim if verify else 0)))
    (hn, g, lo, loa, mixed, inj, o1, o2, o3, o4, o5, att, gate, qout, kbuf, iq, ik, sel, cnt,
     keyn, qn, gated, mo, logits, mixr, val, idx, slog, gscr) = layer_buffers()
    prog.names.update(H=H, ple=ple, xn=xn)
    pos, cos, sin, scores = prog.slot("pos"), prog.slot("cos"), prog.slot("sin"), prog.slot("scores")
    cur = {}

    # RECLANK_DENSE=rq8 (model.dense_rot): those matrices read x rotated in
    # each 32 values, a copy made at the first such product of each x
    rot = getattr(model, "dense_rot", None)
    xr, xqr, xsr, xmr = None, None, None, None

    def quant(src, cols, rows=t):
        prog.emit(P.KQ_QUANT, src, rows, cols, xq, xs, xm)
        cur.update(src=src, rows=rows, cols=cols, rot=False)

    def lin(gname, out, src=None, rows=t):
        nonlocal xr, xqr, xsr, xmr
        if rot is not None and rot(gname):
            assert src is None and rows == cur["rows"], gname
            n = cur["rows"] * cur["cols"]
            if xr is None:
                xr, xqr = f32(t, wide), np.zeros((t, wide), np.int8)
                xsr, xmr = f32(t, wide // 32), f32(t, wide // 16)
            if not cur["rot"]:
                prog.emit(P.COPY, cur["src"], xr, n * 4)
                prog.emit(P.TQ_ROT, xr, n // 32, 0)
                prog.emit(P.KQ_QUANT, xr, cur["rows"], cur["cols"], xqr, xsr, xmr)
                cur["rot"] = True
            xb = {"xq": xqr, "xs": xsr, "xm": xmr, "src": xr, "t": rows}
        else:
            xb = {"xq": xq, "xs": xs, "xm": xm, "src": cur["src"] if src is None else src, "t": rows}
        model.emit_lin4(prog, xb, gname, out)

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
        # the form of the cache of the layer (qwen.layer_kv_form): 0 int16, 1
        # int8, 2 int16 keys and int8 values, 3 float32 rows (heads,
        # positions, head_dim; the first full layer by default)
        lf = layer_kv_form(getattr(cfg, "kv_form", "int16"), cfg, i)
        if lf in ("rq8", "rk8v8"):
            raise ValueError("the %s cache is of Qwen3.6 (compile_qwen_step) only" % lf)
        # (e4m3: the int8 form, its bytes E4M3 codes: FP4-Local's FP8 cache)
        form = {"int16": 0, "int8": 1, "k16v8": 2, "f32": 3, "tq6": 4, "e4m3": 1}[lf]
        if form == 3:
            # ATTN_PREP writes the float rows of the key and the value
            prog.emit(P.ATTN_PREP, o1, o2, o3, model.F(b + "attn_q_norm.weight"),
                      model.F(b + "attn_k_norm.weight"), cos, sin, prog.slot("K.%d" % i),
                      prog.slot("V.%d" % i), prog.slot("hs"), pos, t, nq, nk, hd,
                      cfg.rotary_dim, eps, float(hd ** -0.5), qout, gate)
            base = [prog.slot("K.%d" % i), 0, prog.slot("V.%d" % i), 0]
        else:
            prog.emit(P.ATTN_PREP, o1, o2, o3, model.F(b + "attn_q_norm.weight"),
                      model.F(b + "attn_k_norm.weight"), cos, sin, None, None, 0, pos, t, nq, nk,
                      hd, cfg.rotary_dim, eps, float(hd ** -0.5), qout, gate, kbuf)
            base = [prog.slot("%s.%d" % (nm, i)) for nm in ("kq", "ks", "vq", "vs")]
            per = nk * hd
            kb, vb = kv_row_bytes(lf, per)
            rows = [scalar(P.S_ADD, bs, scalar(P.S_MUL, pos, step))
                    for bs, step in zip(base, (kb, per // 8, vb, per // 8))]
            write = {0: P.KV_WRITE, 1: P.KV_WRITE8, 2: P.KV_WRITEV8, 4: P.KV_WRITETQ}[form]
            prog.emit(write, kbuf, o3, None, None, *rows, t * per)

        def attn_qsa(c):
            # TQ6: the queries and the output in the rotated form (tq6.py)
            if form == 4:
                prog.emit(P.TQ_ROT, qout, t * nq * hd // 32, 0)
            model.emit_attn_qsa(prog, qout, base, scores, att, t, pos, sel, c, maxsel, form)
            if form == 4:
                prog.emit(P.TQ_ROT, att, t * nq * hd // 32, 1)
        if ratio(i) == 0:
            # A dense layer (the MTP layer).
            attn_qsa(dense)
            prog.emit(P.SIGMUL, att, gate, att, t * nq * hd)
            quant(att, nq * hd)
            lin(b + "attn_output.weight", o5)
            return
        # QSA: the indexer selects the keys of each query (backends/cpu/csrc/common/qsa.c).
        lin(b + "indexer.q_proj.weight", iq)
        lin(b + "indexer.k_proj.weight", ik)
        prog.emit(P.QSA_SELECT, iq, ik, prog.slot("idxk.%d" % i), prog.slot("blk.%d" % i),
                  model.F(b + "indexer.q_norm.weight"), model.F(b + "indexer.k_norm.weight"),
                  cos, sin, pos, t, cfg.indexer_heads, cfg.indexer_dim, ratio(i), budget(i),
                  cfg.rotary_dim, float(cfg.rope_theta), eps, sel, cnt, maxsel, qscr,
                  prog.slot("nbmax"), prog.slot("qpos"), mrope_sec(cfg))
        attn_qsa(cnt)
        prog.emit(P.SIGMUL, att, gate, att, t * nq * hd)
        quant(att, nq * hd)
        lin(b + "attn_output.weight", o5)

    def deltanet(i):
        b = "blk.%d." % i
        lin(b + "attn_qkv.weight", o1)
        lin(b + "attn_gate.weight", o2)
        lin(b + "ssm_beta.weight", o3)
        lin(b + "ssm_alpha.weight", o4)
        from reclank.models.qwen.qwen35 import gdn_flags
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
        # nreal rows: the padding rows of a GPU group do not go to the state.
        prog.emit(P.PLE_CONV, gn, gated, H, prog.slot("pleconv.%d" % i),
                  model.F(b + "conv1d.weight", (HD, cfg.ple_conv_kernel)), prog.slot("nreal"), HD,
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
        quant(cat, 2 * hid, rows=t * hc)
        lin(b + "eh_proj.weight", H, rows=t * hc)
        layers, head = [L], b + "hc_head"
    layer_begin = getattr(model, "layer_begin", None)
    # FP4-Local (cfg.b12x): the outputs of the hyper-connection mixes (the inputs of the blocks, the
    # MoE and the head) in bfloat16, as vLLM's gate_mean
    b12x = getattr(cfg, "b12x", False)
    for n, i in enumerate(layers):
        b = "blk.%d." % i
        if fresh and n > 0:
            (hn, g, lo, loa, mixed, inj, o1, o2, o3, o4, o5, att, gate, qout, kbuf, iq, ik, sel,
             cnt, keyn, qn, gated, mo, logits, mixr, val, idx, slog, gscr) = layer_buffers()
        if layer_begin is not None:
            layer_begin(prog, i)
        if i in cfg.ple_layers:
            ple_layer(i)
        hc_pre(b + "hc_attn", mixed)
        if b12x:
            prog.emit(P.BF16_ROUND, mixed, t * hid)
        quant(mixed, hid)
        if i >= len(cfg.layer_types) or cfg.layer_types[i] == "full_attention":
            attention(i)
        else:
            deltanet(i)
        prog.emit(P.HC_ADD, H, o5, inj, t, hc, hid, 1.0 / hc)
        hc_pre(b + "hc_ffn", mixed)
        if b12x:
            prog.emit(P.BF16_ROUND, mixed, t * hid)
        quant(mixed, hid)
        lin(b + "ffn_gate_inp.weight", logits)
        lin(b + "ffn_gate_inp_shexp.weight", slog)
        prog.emit(P.ROUTER_TOPK, logits, t, E, k, val, idx, *model.router_bias4(i))
        xin = mixed
        if mixr is not None:
            prog.emit(P.COPY, mixed, mixr, mixed.nbytes)
            prog.emit(P.TQ_ROT, mixr, t * hid // 32, 0)
            quant(mixr, hid)
            xin = mixr
        xb = {"xq": xq, "xs": xs, "xm": xm, "src": xin, "t": t}
        model.emit_moe4(prog, i, xb, idx, val, slog, scratch, mo)
        prog.emit(P.HC_ADD, H, mo, inj, t, hc, hid, 1.0 / hc)
    hc_pre(head, xn, inject=False)
    if b12x:
        prog.emit(P.BF16_ROUND, xn, t * hid)
    prog.keep.append(dense)
    prog.tokens = t
    return prog.finish()


class Qwen4CPU(Qwen4):
    """Qwen3.8-Flash-Next with the step as one program of records (the C
    kernels of cops). The weights stay in the blocks of the file."""

    CHUNK = 512

    def __init__(self, path, cfg=None, layers=None, mtp=None, dense=None, experts=None):
        super().__init__(path, cfg, layers, mtp, experts)
        self.dense = dense_mode(self.g, dense)
        # The RQ8_0 experts (scripts/convert_q8_gguf.py --experts rq8): all the
        # experts of the file (routed, MTP, shared) or none. The kernels of
        # the MoE then rotate the act of each pair before the down product
        # (kq_set_moe_rot: a state of the process), and the program rotates
        # the x of the experts (compile_qwen4_step).
        from reclank.formats.gguf import ROT_BASE
        kinds = {t in ROT_BASE for n, (_d, t, _o) in self.g.tensors.items()
                 if "_exps." in n or "_shexp.weight" in n and "gate_inp" not in n}
        assert len(kinds) <= 1, "rotated experts (RQ8_0, RQ6_K, RQ4_K, ...) must be all the experts of the file, or none"
        self.rot_experts = kinds == {True}
        from reclank.backends.cpu import cops
        cops.kq_set_moe_rot(self.rot_experts)
        self._k = {}
        self._f = {}
        self.programs = {}

    def K(self, gname):
        from reclank.models.qwen.qwen35 import KMat
        m = self._k.get(gname)
        if m is None and getattr(self.g, "kept", False) and \
                self.g.index["tensors"].get(gname, {}).get("place") == "experts":
            # a stack of experts split between the nodes by the weightkeeper:
            # data holds those of the first node; kept, all the nodes (the GPU
            # takes them as its split, gpu35.QwenGPU._kept_split)
            d = self.g.index["tensors"][gname]
            m = KMat.__new__(KMat)
            m.kept = self.g.experts(gname)
            m.data = m.kept[0][1].reshape(-1)
            m.type, m.cols, m.rows, m.rot = int(d["type"]), int(d["dims"][0]), int(d["dims"][1]), False
            m.nb = int(d["nb"])
            self._k[gname] = m
            return m
        if m is None:
            m = KMat(self.g, gname)
            if m.rot:
                # RQ8_0 / RQ6_K (KMat gives the blocks of Q8_0 / Q6_K): the
                # experts of the file, whose x and act the program rotates
                # (rot_experts), not a dense matrix of dense_rot
                assert self.rot_experts, gname
                m.rot = False
            if m.type == 57 and "_exps." not in gname and "_shexp." not in gname:
                # BF12 (the dense matrices of a file of convert_q8_gguf.py
                # --dense bf12): its kernels, unless RECLANK_DENSE asks for
                # q8, rq8 or bf16 (the bfloat16 bits, then as those modes).
                # Not the experts: a stack of routed experts is E matrices
                # (KMat rows: those of one), and the MoE takes BF12 routed
                # and shared experts together (the float rows of x).
                # rq8: Q8_0 of the rows rotated in each 32 values (as RQ8_0),
                # m.rot; the program rotates the x of their products
                # (compile_qwen4_step, dense_rot). Not the head, which runs
                # in its own program.
                mode = os.environ.get("RECLANK_DENSE", "bf12")
                if mode == "rq8" and gname == "output.weight":
                    mode = "bf12"
                if mode in ("q8", "rq8", "bf16"):
                    from reclank.backends.cpu import cops
                    bits = cops.kq_bf12_to_bf16(m.data, m.rows, m.cols)
                    if mode == "q8":
                        m.data, m.type = cops.kq_to_q8_0(bits, m.cols), 8
                    elif mode == "rq8":
                        f = (bits.astype(np.uint32) << 16).view(np.float32)
                        m.data, m.type = cops.kq_to_q8_0(cops.tq6_rotate(f), m.cols), 8
                        m.rot = True
                    else:
                        m.data, m.type = bits.reshape(-1).view(np.uint8), 30
            if m.type == 30 and "_shexp." in gname and m.cols % 32 == 0 and not getattr(self.g, "as_stored", False):
                # A bfloat16 shared expert (the NVFP4 experts file) to BF12,
                # not Q8_0: it runs for every token, on the GPU, and its
                # bytes are few (3 matrices of 2560 x 640 a layer)
                from reclank.backends.cpu import cops
                bits = m.data.view(np.uint16).reshape(m.rows, m.cols)
                m.data, m.type = cops.kq_bf16_to_bf12(bits, m.cols)[0].reshape(-1), 57
            if m.type == 30 and self.dense in ("q8", "rq8") and ".indexer." not in gname and \
                    gname != "token_embd.weight" and m.rows > 1 and m.cols % 32 == 0:
                # A large bfloat16 matrix to Q8_0 (dense_mode): for the CPU and
                # for the GPU (Qwen4GPU takes the matrices of the model).
                from reclank.backends.cpu import cops
                m.data, m.type = cops.kq_to_q8_0(m.data.view(np.uint16), m.cols), 8
            elif m.type == 30 and ".indexer." in gname:
                # BF16 (the indexer of the GGUF file): the products take it as F32.
                a = self.G(gname)
                m.data, m.type = np.ascontiguousarray(a, np.float32).view(np.uint8).reshape(-1), 0
            self._k[gname] = m
        return m

    def dense_rot(self, gname):
        """True when the products of the matrix read x rotated in each 32
        values (RECLANK_DENSE=rq8)."""
        return getattr(self.K(gname), "rot", False)

    def M(self, name, full=None):
        """A matrix by its GGUF name, or by the name of the Qwen3.6 modules
        (for reclank/models/qwen/gpu35.py)."""
        from reclank.models.qwen.qwen35 import gguf_name
        return self.K(name if name in self.g.tensors else gguf_name(name))

    # ---- the records of the program (compile_qwen4_step); Qwen4GPU has
    # its own ----

    def KP(self, gname):
        """The matrix for the CPU products, in groups of 16 rows (csrc/
        kquants.c), made at the first use: Q8_0 (16 n rows) as KQ_Q8X16,
        float32 as KQ_F32X16, bfloat16 as KQ_BF16X16 (the last two take x
        as float32; 64 rows or more: the smaller ones take the threads with
        their tokens). RECLANK_X16=0: as it is. The GPU takes K().

        The cost in RAM: the packed copy is private memory, next to the
        mapped file (the page cache). It has the bytes of the source (a
        bfloat16 matrix stays 2 bytes a value). For Qwen3.8 (the NVFP4 GGUF)
        the head (output.weight) is 1.2 GiB, and all the dense matrices of
        the CPU program are about 8.2 GiB with bfloat16, or 4.5 GiB with Q8_0
        (dense q8). RECLANK_X16=0 saves that memory, and the CPU products
        are then slower."""
        m = self._kp.get(gname) if hasattr(self, "_kp") else None
        if m is None:
            if not hasattr(self, "_kp"):
                self._kp = {}
            m = self.K(gname)
            if os.environ.get("RECLANK_X16", "1") != "0":
                from reclank.backends.cpu import cops
                from reclank.models.qwen.qwen35 import KMat
                data = None
                if m.type == 8 and m.rows % 16 == 0 and m.cols % 32 == 0:
                    data, t = cops.kq_pack_q8x16(m.data, m.rows, m.cols), 60
                elif m.type in (0, 30) and m.rows >= 64:
                    data = cops.kq_pack_x16f(m.data, m.type == 30, m.rows, m.cols)
                    t = 61 if m.type == 30 else 62
                elif m.type == 57 and m.rows >= 64:
                    data, t = cops.kq_pack_bf12x16(m.data, m.rows, m.cols), 63
                if data is not None:
                    p = KMat.__new__(KMat)
                    p.data, p.type, p.rows, p.cols = data, t, m.rows, m.cols
                    m = p
            self._kp[gname] = m
        return m

    def emit_lin4(self, prog, xb, gname, out):
        from reclank.ir import program as P
        m = self.KP(gname)
        prog.emit(P.KQ_LINEAR, xb["xq"], xb["xs"], xb["xm"], xb["src"], m.data, m.type, m.rows,
                  m.cols, xb["t"], out)

    def moe_scratch4(self, t):
        from reclank.backends.cpu import cops
        cfg = self.cfg
        return cops.kq_moe_scratch(t, cfg.top_k, cfg.num_experts, cfg.hidden_size,
                                   cfg.moe_inter)

    def router_bias4(self, i):
        """The extra operands of ROUTER_TOPK (gpu38: RESIDENT_BIAS, HOT_SWAP); none."""
        return ()

    def emit_moe4(self, prog, i, xb, idx, val, slog, scratch, out):
        from reclank.backends.cpu import cops
        from reclank.ir import program as P
        cfg = self.cfg
        b = "blk.%d." % i
        mats = cops.kq_moe_mats(*(self.K(b + "ffn_%s_exps.weight" % n).c()
                                  for n in ("gate", "up", "down")),
                                [self.K(b + "ffn_%s_shexp.weight" % n).c()
                                 for n in ("gate", "up", "down")])
        # RECLANK_MOE_XQ (a test of the precision of the activations of the
        # experts): "8" (the default) the rows of RECLANK_MOE_X16 (int16
        # by default, int8 with 0); "16" the float rows rounded
        # to int16 in each 32 values, and the GELU too, then float products;
        # "f" float32 rows and products
        xqm = os.environ.get("RECLANK_MOE_XQ", "8")
        # the float32 rows always (KQ_BF12 experts read them; no mode without
        # its flag)
        extra = (None, 0, xb["src"]) if xqm == "8" else (None, 2 | (16 if xqm == "16" else 0), xb["src"])
        if xqm == "8" and os.environ.get("RECLANK_MOE_X16", "1") == "1":
            extra = (None, 32, xb["src"])       # two int8 planes (qwen_gpu.MOE_X16)
        prog.emit(P.KQ_MOE, xb["xq"], xb["xs"], xb["xm"], idx, val, xb["t"], cfg.top_k,
                  cfg.num_experts, mats, slog, cfg.hidden_size, cfg.moe_inter, scratch, out, *extra)
        prog.keep.append(mats)

    def emit_attn_qsa(self, prog, q, base, scores, out, t, pos, sel, cnt, maxsel, form=0):
        """form: 0 the int16 cache, 1 int8, 2 int16 keys and int8 values, 3
        float32 rows (heads, positions, head_dim; the head stride is the
        slot hs), 4 TQ6 (q and out rotated)."""
        from reclank.ir import program as P
        cfg = self.cfg
        prog.emit(P.ATTN_QSA, q, *base, scores, out, cfg.num_heads, cfg.num_kv_heads,
                  cfg.head_dim, t, pos, sel, cnt, maxsel, form, prog.slot("hs"))

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
        from reclank.backends.cpu import cops
        m = self.K("token_embd.weight")
        return cops.kq_rows(m.data, m.type, m.cols, ids)

    def logits(self, h, chunk=None):
        from reclank.backends.cpu import cops
        from reclank.models.qwen.qwen35 import KX
        m, qx = self.KP("output.weight"), KX(h)
        out = np.empty((qx.t, m.rows), np.float32)
        cops.kq_linear(m.data, m.type, m.rows, m.cols, qx.xq, qx.xs, qx.xm, qx.x, qx.t, out)
        return out

    def ple_rows(self, ids, cache):
        """The 16 rows of the n-gram table of each token (t x hid); FP4-Local's
        (cfg.ple_bf16) rounded to bfloat16, as vLLM (ple_embedding/reference.py)."""
        out = self._ple_rows(ids, cache)
        if getattr(self.cfg, "ple_bf16", False):
            from reclank.formats.fp4local_st import bf16r
            out = bf16r(out).reshape(out.shape)
        return out

    def _ple_rows(self, ids, cache):
        from reclank.backends.cpu import cops
        rows = self.ple_ids(ids, cache)
        if getattr(self.g, "kept", False) and "per_layer_token_embd.weight" in self.g.index["tensors"]:
            # the weightkeeper (keep.KeptGGUF): the rows split between the nodes,
            # read by their addresses (many threads), then decoded
            g, name = self.g, "per_layer_token_embd.weight"
            d = g.index["tensors"][name]
            r = rows.reshape(-1).astype(np.int64)
            raw = cops.kq_gather(g.row_addrs(name, r), d["rb"])
            return cops.kq_rows(raw.reshape(-1), d["type"], int(d["dims"][0]),
                                np.arange(len(r))).reshape(len(ids), -1)
        if hasattr(self.g, "ple_rows"):
            return self.g.ple_rows(rows.reshape(-1)).reshape(len(ids), -1)
        m = self.K("per_layer_token_embd.weight")
        if m.type == 52:
            # FP8 E4M3 rows (scripts/convert_nvfp4_gguf.py): many threads read
            # the random rows, with no read-ahead.
            from reclank.formats.gguf import E4M3_VALUES
            if not getattr(self, "_ple_advised", False):
                self.g.advise_random("per_layer_token_embd.weight")
                self._ple_scale = float(self.g.dequant("per_layer_token_embd.scale").reshape(-1)[0])
                self._ple_advised = True
            r = rows.reshape(-1).astype(np.int64)
            raw = cops.kq_gather(m.data.ctypes.data + r * m.cols, m.cols)
            return (E4M3_VALUES[raw] * self._ple_scale).reshape(len(ids), -1)
        if m.type == 8:
            # Q8_0 rows (scripts/convert_q8_gguf.py, 170 bytes for 160
            # values): the same gather, then the dequant of the rows
            if not getattr(self, "_ple_advised", False):
                self.g.advise_random("per_layer_token_embd.weight")
                self._ple_advised = True
            rb = m.cols // 32 * 34
            r = rows.reshape(-1).astype(np.int64)
            raw = cops.kq_gather(m.data.ctypes.data + r * rb, rb)
            return cops.kq_rows(raw.reshape(-1), 8, m.cols, np.arange(len(r))).reshape(len(ids), -1)
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
        from reclank.models.qwen.qwen35 import cache_params, scores_buffer
        cfg = self.cfg
        cos, sin = self.rope(rope_positions(cache, pos, t))
        kw = {"pos": pos, "nreal": t, "cos": np.ascontiguousarray(cos),
              "sin": np.ascontiguousarray(sin), "scores": scores_buffer(cfg, pos + t),
              "qpos": cache.qpos if cache.qpos is not None else 0}
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

    def _run(self, prog, ids, cache, pos, media=None):
        cfg = self.cfg
        self._bind(prog, cache, pos, len(ids))
        x = media_inputs(self.embed(ids), pos, media)
        prog.names["H"][:] = np.repeat(x[:, None, :], cfg.hc_count, axis=1).reshape(len(ids), -1)
        if cfg.ple_layers:
            prog.names["ple"][:] = self.ple_rows(ids, cache)
        prog.run()

    def forward(self, ids, cache, start_pos=0, hook=None, streams=False, media=None):
        """Run tokens from start_pos. Return the input of the head (t x
        hidden); with streams, also the streams after the last layer (t x
        hc * hidden, the input of the MTP layer). media: the spans of the
        images (cache positions)."""
        ids = list(ids)
        out, hs = [], []
        c0 = 0
        while c0 < len(ids):
            chunk = ids[c0:c0 + self.CHUNK]
            prog = self.program(len(chunk))
            self._run(prog, chunk, cache, start_pos + c0, media)
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
        """Run a group of tokens from start_pos as an MTP verify group (see
        the rule of qwen._QwenRuns.verify for the rows after the commit).
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
        from reclank.backends.cpu import cops
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
        from reclank.models.qwen.qwen35 import scores_buffer
        cfg = self.cfg
        t = len(ids)
        prog = self.program(t, "mtp")
        L = cfg.num_hidden_layers
        cos, sin = self.rope(rope_positions(mcache, pos, t))
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
