"""Run the Gemma 4 assistant model, the MTP drafter, and the MTP decode.

Gemma 4 gives multi-token prediction as a separate small model, the
assistant. The assistant proposes the next few tokens. The target model then
checks them in one batch. See MTP_PLAN.md.

The assistant has four decoder layers. Its attention has a query and no key
or value. A sliding layer reads the keys and the values of the last sliding
layer of the target, and the global layer reads the last global layer. The
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
from .config import Config
from .st import SafeTensors

_NORMS = ("input_layernorm", "post_attention_layernorm",
          "pre_feedforward_layernorm", "post_feedforward_layernorm",
          "self_attn.q_norm")
_MATS = ("self_attn.q_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def shared_layers(cfg):
    """Return the target layers that the assistant reads.

    The result is (last sliding layer, last global layer). A target with
    shared key and value layers (E2B and E4B) stops before the first shared
    layer, because those layers reuse the cache of an earlier layer.
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
    """

    def __init__(self, path, dtype="int4", q8_attn=True):
        with open(os.path.join(path, "config.json")) as fh:
            raw = json.load(fh)
        self.raw = raw
        self.cfg = Config({"text_config": raw["text_config"]})
        self.backbone = raw["backbone_hidden_size"]
        self.dtype = dtype
        # Read the int8 copy of the target cache when it is ready. Set False
        # to read the float copy, as the check against transformers does.
        self.q8_attn = q8_attn
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

    def _quant(self, w):
        if self.dtype == "int4":
            return ops.quantize_int4(w)
        if self.dtype == "int8":
            return ops.quantize_int8(w)
        return np.ascontiguousarray(w, dtype=np.float32)

    def linear(self, x, w):
        """Multiply x by W. Use the kernel of the weight dtype."""
        if self.dtype == "int4":
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
        if self.q8_attn and ops.attn_ready() and cache.q8_ready(layer):
            # The int8 copy of the target cache, with the fused kernel of the
            # decode step. It reads a quarter of the bytes of the float copy.
            kq, ks, vq, vs, base = cache.read_q8(layer, pos)
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

        emb is the scaled target embedding of the token, shape (1, backbone).
        h is the target hidden state, shape (1, backbone). layers is the pair
        of shared_layers() for the target.
        """
        eps = self.cfg.rms_norm_eps
        u = self.linear(np.concatenate([emb, h], axis=-1).astype(np.float32), self.pre)
        for i, w in enumerate(self.layers):
            plan = self.cfg.plan[i]
            layer = layers[0] if plan.is_sliding else layers[1]
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
        """
        layers = shared_layers(target.cfg)
        out = []
        for _ in range(n):
            logits, h = self.step(target.embed([token]), h, pos, cache, layers)
            token = int(np.argmax(logits[0]))
            out.append(token)
            if token in eos_ids:
                break
        return out


def mtp_generate(target, drafter, ids, cache, max_new_tokens, n_draft=3,
                 eos_ids=(), stats=None):
    """Generate tokens with greedy selection and MTP. Return the new tokens.

    The result is the same as the plain greedy decode when the batch and the
    single-token paths of the target give the same argmax. stats, when given,
    is a dict that receives the counts of steps, drafts, and accepted drafts,
    and the time of the prompt pass and of the decode.
    """
    ids = list(ids)
    t0 = time.perf_counter()
    x = target.prefill(ids, cache)
    t1 = time.perf_counter()
    h = x[-1:]
    nxt = int(np.argmax(target.logits(h)[0]))
    out = []
    pos = len(ids)
    steps = drafts = accepted = 0
    while True:
        out.append(nxt)
        if nxt in eos_ids or len(out) >= max_new_tokens:
            break
        k = min(n_draft, max_new_tokens - len(out))
        d = drafter.draft(target, nxt, h, pos, cache, k, eos_ids) if k > 0 else []
        batch = [nxt] + d
        x = target.forward(batch, cache=cache, start_pos=pos)
        pred = np.argmax(target.logits(x), axis=-1)
        # Row i predicts the token after batch[i]. Keep the drafts while
        # they match the target.
        j = 0
        while j < len(d) and int(pred[j]) == d[j]:
            j += 1
        steps += 1
        drafts += len(d)
        accepted += j
        stop = False
        for t in d[:j]:
            out.append(t)
            if t in eos_ids or len(out) >= max_new_tokens:
                stop = True
                break
        if stop:
            break
        # The rows after the last kept token hold rejected drafts.
        pos += j + 1
        cache.truncate(pos)
        h = x[j:j + 1]
        nxt = int(pred[j])
    if stats is not None:
        stats.update(steps=steps, drafts=drafts, accepted=accepted,
                     prefill_s=t1 - t0, decode_s=time.perf_counter() - t1)
    return out[:max_new_tokens]
