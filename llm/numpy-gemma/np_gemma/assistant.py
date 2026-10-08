"""Run the Gemma 4 assistant model, the MTP drafter, and the MTP decode.

Gemma 4 gives multi-token prediction as a separate small model, the
assistant. The assistant proposes the next few tokens. The target model then
checks them in one batch. See MTP_PLAN.md.

The assistant has four decoder layers. Its attention has a query and no key
or value. A sliding layer reads the keys and the values of the last sliding
layer of the target. The global layer reads the last global layer. The
assistant writes nothing to the cache.

One draft step:

    x      = target_embed(token)          (the target scale is included)
    u      = pre_projection([x, h])
    u      = the four decoder layers, at the position of the token
    u      = rms_norm(u, norm)
    logits = u @ embed_tokens.T           (or the centroid head)
    h      = post_projection(u)           (the h of the next step)

The first h is the hidden state of the target after its final norm, at the
row that predicted the token. Every draft step uses the same position. Only
the token and h change.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import ops
from . import rope as rope_mod
from . import speculative
from .speculative import RowPicker, greedy_pick  # noqa: F401  (they lived here)
from .config import Config
from .st import SafeTensors

_NORMS = ("input_layernorm", "post_attention_layernorm",
          "pre_feedforward_layernorm", "post_feedforward_layernorm",
          "self_attn.q_norm")
_MATS = ("self_attn.q_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


# The names of the tensors of a drafter GGUF (arch gemma4-assistant, as the
# MTP files of unsloth) for the names of the safetensors of the assistant.
_GGUF_LAYER = {"input_layernorm": "attn_norm", "post_attention_layernorm": "post_attention_norm",
               "pre_feedforward_layernorm": "ffn_norm", "post_feedforward_layernorm": "post_ffw_norm",
               "self_attn.q_norm": "attn_q_norm", "self_attn.q_proj": "attn_q",
               "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate",
               "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
_GGUF_TOP = {"model.norm.weight": "output_norm.weight",
             "pre_projection.weight": "nextn.pre_projection.weight",
             "post_projection.weight": "nextn.post_projection.weight",
             "model.embed_tokens.weight": "token_embd.weight"}


def _gguf_getter(path):
    """Return get(name) for the tensors of a drafter GGUF by their names in the
    safetensors of the assistant, as float32 (the blocks dequantized). The
    norms are stored as is (no added 1)."""
    from .gguf import GGUF
    g = GGUF(path)

    def get(name, dtype=np.float32):
        if name in _GGUF_TOP:
            gn = _GGUF_TOP[name]
        else:
            parts = name.split(".")          # model.layers.<i>.<key>...
            i, key = parts[2], ".".join(parts[3:])
            if key == "layer_scalar":
                gn = "blk.%s.layer_output_scale.weight" % i
            else:
                gn = "blk.%s.%s.weight" % (i, _GGUF_LAYER[key[:-len(".weight")]])
        return np.ascontiguousarray(g.dequant(gn), dtype=np.float32)
    get.release_pages = lambda: None
    return get


def shared_layers(cfg):
    """Return the target layers that the assistant reads.

    The result is (last sliding layer, last global layer). Some targets share
    key and value layers (E2B and E4B). For them, the search stops
    before the first shared layer. Those layers reuse the cache of an earlier
    layer.
    """
    n = cfg.num_hidden_layers - (getattr(cfg, "num_kv_shared_layers", 0) or 0)
    types = cfg.layer_types[:n]
    sliding = max(i for i, t in enumerate(types) if t == "sliding_attention")
    full = max(i for i, t in enumerate(types) if t == "full_attention")
    return sliding, full


class Assistant:
    """Load the assistant weights and run one draft step at a time.

    path is the snapshot directory of the Hugging Face checkpoint. It holds
    config.json and model.safetensors. dtype "f32" keeps float32 weights.
    dtype "int4" and "int8" quantize every matrix, the output head too.
    weights, a drafter GGUF (the MTP files of unsloth, arch
    gemma4-assistant), gives the weights in place of model.safetensors; the
    snapshot still gives config.json.
    """

    def __init__(self, path, dtype="int4", q8_attn=True, weights=None):
        with open(os.path.join(path, "config.json")) as fh:
            raw = json.load(fh)
        self.raw = raw
        self.cfg = Config({"text_config": raw["text_config"]})
        self.backbone = raw["backbone_hidden_size"]
        self.dtype = dtype
        # Read the int16 copy of the target cache when it is ready. Set False
        # to read the float copy, as the check against transformers does.
        self.q8_attn = q8_attn
        # Stop a draft when the drafter gives its best token less than this
        # probability. Zero turns the test off. NP_GEMMA_MTP_PMIN sets it.
        self.p_min = float(os.environ.get("NP_GEMMA_MTP_PMIN", "0"))
        if weights:
            if raw.get("use_ordered_embeddings"):
                raise ValueError("a drafter GGUF has no centroid head")
            get = _gguf_getter(weights)
            self.st = get
        else:
            self.st = SafeTensors(os.path.join(path, "model.safetensors"))
            get = self.st.get
        self.layers = []
        for i in range(self.cfg.num_hidden_layers):
            p = "model.layers.%d." % i
            w = {k: get(p + k + ".weight") for k in _NORMS}
            for k in _MATS:
                w[k] = self._quant(get(p + k + ".weight"))
            w["layer_scalar"] = float(get(p + "layer_scalar")[0])
            self.layers.append(w)
        self.norm = get("model.norm.weight")
        self.pre = self._quant(get("pre_projection.weight"))
        self.post = self._quant(get("post_projection.weight"))
        self.head = self._quant(get("model.embed_tokens.weight"))
        self.centroids = None
        if raw.get("use_ordered_embeddings"):
            self.centroids = get("masked_embedding.centroids.weight")
            order = self.st.get("masked_embedding.token_ordering", dtype=None)
            self.top_k = raw["centroid_intermediate_top_k"]
            n = raw["num_centroids"]
            self.order = np.asarray(order, dtype=np.int64).reshape(n, -1)
            # The centroid head reads a few rows of the table. Keep a float32
            # copy for the gather.
            self.head_f32 = get("model.embed_tokens.weight")
        self.st.release_pages()
        self._rope = {}
        # The int4 matrices in groups of 16 rows (KQ_Q4X, as the target:
        # ops.q4x_pack_model), with int8 x for each token count. The head
        # alone (262144 rows of 1024 values for the 26B) took 6.3 ms of the
        # 13.6 ms of a draft step on the int4 path (AVX2, 6 threads).
        self._q4x = {}
        if dtype == "int4" and ops._Q4X_ON:
            mats = [self.pre, self.post, self.head] + [w[k] for w in self.layers for k in _MATS]
            for q, sc in mats:
                if ops._q4x_ok(q, sc):
                    self._q4x[q.ctypes.data] = ops._cops.kq_q4x_pack(q, sc)

    def _quant(self, w):
        if self.dtype == "int4":
            return ops.quantize_int4(w)
        if self.dtype == "int8":
            return ops.quantize_int8(w)
        return np.ascontiguousarray(w, dtype=np.float32)

    def linear(self, x, w):
        """Multiply x by W. Use the kernel of the weight dtype."""
        if self.dtype == "int4":
            qx = self._q4x.get(w[0].ctypes.data)
            if qx is not None:
                return ops._q4x_linear(x, qx)
            return ops.linear_int4(x, w[0], w[1])
        if self.dtype == "int8":
            return ops.linear_int8(x, w[0], w[1])
        return x @ w.T

    def _cos_sin(self, plan, pos):
        key = (plan.is_sliding, pos)
        e = self._rope.get(key)
        if e is None:
            if len(self._rope) > 8:
                self._rope.clear()
            e = rope_mod.cos_sin(self.cfg.rope_inv_freq(plan), np.array([pos]))
            self._rope[key] = e
        return e

    def _attention(self, x, w, plan, pos, cache, layer):
        """Run the query-only attention over the target cache.

        The query at pos sees the target rows before pos. A sliding layer also
        drops the rows at pos - window and before.
        """
        eps = self.cfg.rms_norm_eps
        hd = plan.head_dim
        nq = plan.num_q_heads
        q = self.linear(x, w["self_attn.q_proj"]).reshape(nq, hd)
        q = ops.rms_norm(q, w["self_attn.q_norm"], eps)
        cos, sin = self._cos_sin(plan, pos)
        q = rope_mod.apply(q[None], cos, sin)[0]
        if layer is None:
            # The E4B cache keeps the key and the value that the shared layers
            # reuse, for each layer type, with the shape (keys, heads, dim).
            store = cache.shared[plan.kind if hasattr(plan, "kind") else
                                 ("sliding_attention" if plan.is_sliding else "full_attention")]
            window = self.cfg.sliding_window if plan.is_sliding else 0
            o = ops.attn_decode_f32(np.ascontiguousarray(q), store[0][:pos].transpose(1, 0, 2),
                                    store[1][:pos].transpose(1, 0, 2), pos, 0, window)
            return self.linear(o.reshape(1, nq * hd), w["self_attn.o_proj"])
        if self.q8_attn and ops.attn_ready() and cache.qc_ready(layer):
            # The int16 copy of the target cache, with the fused kernel of the
            # decode step. It reads about half the bytes of the float copy.
            kq, ks, vq, vs, base = cache.read_qc(layer, pos)
            lo = max(0, pos - self.cfg.sliding_window + 1 - base) if plan.is_sliding else 0
            o = ops.attn_decode(np.ascontiguousarray(q), kq[lo:], ks[lo:], vq[lo:], vs[lo:],
                                nq, kq.shape[1], hd, kq.shape[0] - lo)
            return self.linear(o.reshape(1, nq * hd), w["self_attn.o_proj"])
        K, V, base = cache.read(layer, pos)
        lo = 0
        if plan.is_sliding:
            lo = max(0, pos - self.cfg.sliding_window + 1 - base)
        K = K[lo:]
        V = V[lo:]
        nk = K.shape[1]
        rep = nq // nk
        # The attention scale is 1.0, as in the target.
        qb = q.reshape(nk, rep, hd)
        s = np.matmul(qb, K.transpose(1, 2, 0))            # (nk, rep, n)
        s -= s.max(axis=-1, keepdims=True)
        np.exp(s, out=s)
        s /= s.sum(axis=-1, keepdims=True)
        o = np.matmul(s, V.transpose(1, 0, 2))             # (nk, rep, hd)
        return self.linear(o.reshape(1, nq * hd), w["self_attn.o_proj"])

    def step(self, emb, h, pos, cache, layers):
        """Run one draft step. Return the logits and the next h.

        The argument emb is the scaled target embedding of the token, with
        the shape (1, backbone). The argument h is the target hidden state,
        with the same shape. The argument layers is the pair of
        shared_layers() for the target.
        """
        eps = self.cfg.rms_norm_eps
        u = self.linear(np.concatenate([emb, h], axis=-1).astype(np.float32), self.pre)
        for i, w in enumerate(self.layers):
            plan = self.cfg.plan[i]
            layer = None if layers is None else (layers[0] if plan.is_sliding else layers[1])
            a = ops.rms_norm(u, w["input_layernorm"], eps)
            a = self._attention(a, w, plan, pos, cache, layer)
            u = u + ops.rms_norm(a, w["post_attention_layernorm"], eps)
            m = ops.rms_norm(u, w["pre_feedforward_layernorm"], eps)
            g = self.linear(m, w["mlp.gate_proj"])
            up = self.linear(m, w["mlp.up_proj"])
            m = self.linear(ops.gelu_tanh(g) * up, w["mlp.down_proj"])
            u = u + ops.rms_norm(m, w["post_feedforward_layernorm"], eps)
            u = u * w["layer_scalar"]
        u = ops.rms_norm(u, self.norm, eps)
        return self.logits(u), self.linear(u, self.post)

    def logits(self, u):
        """Return the logits of the draft. Use the centroid head when present.

        The centroid head scores the centroids, keeps the best top_k of them,
        and computes the logits of their tokens only. Every other token gets a
        value below the smallest computed logit.
        """
        if self.centroids is None:
            return self.linear(u, self.head)
        c = (u @ self.centroids.T)[0]
        top = np.argpartition(-c, self.top_k)[:self.top_k]
        tok = self.order[top].reshape(-1)
        sel = self.head_f32[tok] @ u[0]
        out = np.full((1, self.head_f32.shape[0]), sel.min() - 1.0, dtype=np.float32)
        out[0, tok] = sel
        return out

    def draft(self, target, token, h, pos, cache, n, eos_ids=()):
        """Propose up to n tokens after token. Stop after an end token.

        token is the last accepted token at position pos. The target cache
        holds the rows before pos. h is the target hidden state of the row
        that predicted token.

        With p_min above zero, stop when the probability of the best draft
        falls below p_min. The first draft is always kept. A draft that the
        drafter itself doubts is often rejected, and it costs a draft step
        and a verify row.
        """
        # The E4B cache gives the shared key and value by layer type. The
        # other cache gives them by layer index.
        layers = None if hasattr(cache, "shared") else shared_layers(target.cfg)
        out = []
        for _ in range(n):
            logits, h = self.step(target.embed([token]), h, pos, cache, layers)
            row = logits[0]
            token = int(np.argmax(row))
            if out and self.p_min > 0.0:
                p = 1.0 / float(np.exp(row - row[token]).sum())
                if p < self.p_min:
                    break
            out.append(token)
            if token in eos_ids:
                break
        return out


def mtp_enabled():
    """Return False when NP_GEMMA_MTP=0 turns the drafter off.

    With NP_GEMMA_GPU=1 the default is off. The verify group of MTP then
    sends about three times the cold experts to the CPU, and a step of MTP
    gives fewer tokens/s than the plain decode on the GPU (SPLIT_PLAN.md).
    NP_GEMMA_MTP=1 turns it on again."""
    v = os.environ.get("NP_GEMMA_MTP")
    if v is None:
        return os.environ.get("NP_GEMMA_GPU", "0") != "1"
    return v != "0"


def mtp_stream(target, drafter, cache, ids, h, nxt, n_draft, eos_ids, pick,
               max_new_tokens, stats=None):
    """Yield the new tokens of an MTP decode, one at a time (the loop of
    speculative.stream, for a Gemma target and its assistant).

    The list ids holds the tokens in the cache. The function adds the tokens
    of the rows that it keeps.

    The token nxt is the first new token. The cache does not hold it yet. The
    array h is the target hidden state of the row that predicted nxt. The
    function pick(logits) selects a token from one row of target logits. It
    is the sampler of the plain decode.

    The target picks its own token at each row of a verify batch, with the
    sampler of the plain decode. A draft is kept while it is the token that
    the target picked. Thus every emitted token is the token that the plain
    decode emits, and pick runs one time for each emitted token, in the same
    order. With greedy selection, or with a sampler that has a seed, the text
    is the same as the text of the plain decode.

    A Sampler with mtp_accept "in_set" also keeps a draft that it did not
    pick when its settings allow the draft (Sampler.draft_ok). That keeps
    more drafts, but the text is no longer the text of the plain decode.
    """
    if (hasattr(target, "gpu_mirror") and not hasattr(cache, "shared")
            and not getattr(drafter, "on_gpu", False)):
        # With the cache on the GPU, the drafter needs the new rows of its
        # two layers in the host cache after each step.
        target.gpu_mirror(cache, shared_layers(target.cfg))
    yield from speculative.stream(speculative.GemmaTarget(target, cache, ids),
                                  speculative.GemmaDrafter(drafter, target, cache), nxt, h,
                                  len(ids), n_draft, pick, eos_ids, max_new_tokens, stats)


def mtp_generate(target, drafter, ids, cache, max_new_tokens, n_draft=2,
                 eos_ids=(), stats=None, pick=greedy_pick):
    """Run the prompt, then generate tokens with MTP. Return the new tokens.

    The result is the same as the plain decode with the same pick. The
    optional dict stats receives the counts of steps, drafts, and accepted
    drafts. It also receives the time of the prompt pass and of the decode.
    """
    ids = list(ids)
    t0 = time.perf_counter()
    x = target.prefill(ids, cache)
    t1 = time.perf_counter()
    h = x[-1:]
    nxt = RowPicker(target, h, pick).token(0)
    st = {} if stats is None else stats
    out = list(mtp_stream(target, drafter, cache, ids, h, nxt, n_draft, eos_ids,
                          pick, max_new_tokens, st))
    st.update(prefill_s=t1 - t0, decode_s=time.perf_counter() - t1)
    return out
