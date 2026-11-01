"""A Qwen3 dense model (transformers' Qwen3ForCausalLM) as an encoder program:
the hidden states of chosen layers over a prompt (FLUX.2 [klein]'s text
encoder: Qwen3-4B, layers 9, 18 and 27).

Ported from transformers (src/transformers/models/qwen3/modeling_qwen3.py;
Apache License 2.0; Copyright 2025 The Qwen team, Alibaba Group and the
HuggingFace Inc. team). See THIRD_PARTY_NOTICES.md.

    a layer: RMS norm, q k v (no bias), the RMS norm of each head of q and of
    k (q's weight times hd^-1/2: the attention's scale), RoPE (NEOX halves,
    theta of the config) at the token's position, the causal attention with
    KV heads shared (GQA), the output; + x; RMS norm, SiLU(gate) up, down; + x

hidden_states[k] of transformers is the residual stream after k layers (0:
the embeddings); a program runs the layers up to the largest k it gives. The
tokens' rows of embed_tokens are gathered on the host. A prompt of fewer
tokens than its rows (padding at the right, as a tokenizer's max_length)
masks the padding as keys (attention_mask): row i attends to the keys [0,
min(i + 1, tokens)), the padding rows too. A program takes n prompts of the
same rows, each its own keys.

The matrices: float32 on the CPU (exact), on the GPU the checkpoint's
bfloat16 rows as they are (KQ_BF16: the tensor cores with x as two bfloat16
planes, float32 sums).

Images (Qwen3-VL's text model, Qwen-Image-2.1's editing: hidden's media, one
prompt): the vision tower's rows in place of the embeddings of each image's
tokens; the interleaved MRoPE (mrope_section) of three positions a token
(Qwen3-VL's get_rope_index: an image's tokens at (s, s + row, s + column)),
frequency j at the position of its axis (H at j % 3 == 1 and W at j % 3 == 2
of the first 3 x section, T else) as cos and sin tables (ENC_ROPE_TAB, NEOX
halves: the angle a float32 product); after each of the first layers the
rows of a deepstack merger added to the image tokens' (_deepstack_process).
The causal attention as for a text.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner
from reclank.formats.gguf import BF16
from reclank.formats.safetensors import Shards
from reclank.runtime.threads import team_threads


class _KQ:
    """A matrix of a GGUF type for KQ_LINEAR (gemma4.e4b._KQ)."""

    def __init__(self, data, type_, rows, cols):
        self.data, self.type, self.rows, self.cols = data, type_, rows, cols


def lin16(W, name, gpu, bias=True, rows=None, cols=None, weight=".weight", cpu_bf16=False):
    """A linear of a checkpoint's bfloat16 (or float32) W (out, in): on the
    GPU the bfloat16 rows as they are (lin.kq, KQ_BF16), else float32 W
    (cpu_bf16: the bfloat16 rows, KQ_BF16X16 of a CPU program: the same
    values in float32 products, half the memory); the bias float32. rows,
    cols: slices of W (a fused matrix split: a DiT's q k v and MLP in one),
    the bias by rows."""
    rows = slice(None) if rows is None else rows
    cols = slice(None) if cols is None else cols
    b = np.ascontiguousarray(W.get(name + ".bias")[rows], dtype=np.float32) if bias else None
    if W.dtype(name + weight) == "BF16":
        raw = np.ascontiguousarray(W.get_bf16(name + weight)[rows, cols])
        if gpu:
            lin = _Plain(None)
            lin.kq = _KQ(raw.view(np.uint8), BF16, raw.shape[0], raw.shape[1])
        elif cpu_bf16:
            lin = _Plain(raw)
        else:
            lin = _Plain((raw.astype(np.uint32) << 16).view(np.float32))
        lin.shape = raw.shape
    else:
        lin = _Plain(np.ascontiguousarray(W.get(name + weight)[rows, cols], dtype=np.float32))
        lin.shape = lin.w.shape
    lin.b = b
    return lin


def qwen3_layers(W, prefix, n, hd, gpu, cpu_bf16=False):
    """The first n layers of a Qwen3 dense checkpoint (names prefix + "%d."):
    each a dict of its norms and linears (lin16), q's norm weight times
    hd^-1/2 (the attention's scale)."""
    f = lambda name: np.ascontiguousarray(W.get(name), dtype=np.float32)  # noqa: E731
    layers = []
    for i in range(n):
        r = prefix + "%d." % i
        L = lambda name: lin16(W, r + name, gpu, bias=False, cpu_bf16=cpu_bf16)  # noqa: E731
        layers.append(dict(
            ln_a=f(r + "input_layernorm.weight"), ln_m=f(r + "post_attention_layernorm.weight"),
            q=L("self_attn.q_proj"), k=L("self_attn.k_proj"), v=L("self_attn.v_proj"), o=L("self_attn.o_proj"),
            qn=(f(r + "self_attn.q_norm.weight") * np.float32(hd ** -0.5)).astype(np.float32),
            kn=f(r + "self_attn.k_norm.weight"),
            gate=L("mlp.gate_proj"), up=L("mlp.up_proj"), down=L("mlp.down_proj")))
    return layers


def rope_inv(hd, theta):
    """The angles of RoPE a position (hd / 2), as transformers' default
    rope (float32)."""
    return (1.0 / (np.float32(theta) ** (np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd)))).astype(np.float32)


class Qwen3Dense:
    """The layers of a Qwen3 dense checkpoint directory (config.json and its
    safetensors). hidden(ids, tokens) gives the hidden states of taps."""

    def __init__(self, path, taps, gpu=False, cpu_threads=None, prefix="model.", final_norm=False, cfg=None,
                 cpu_bf16=False):
        # cfg: the text config of a multimodal checkpoint (Qwen3-VL's text_config: its MRoPE of a
        # text alone, the three positions the same, is this RoPE) in place of path's config.json
        cfg = cfg or json.load(open(os.path.join(path, "config.json")))
        assert cfg["model_type"] in ("qwen3", "qwen3_vl_text") and not cfg.get("attention_bias") and \
            cfg["hidden_act"] == "silu"
        self.W = Shards(path)
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.taps = tuple(taps)
        # final_norm: the tap of the last layer after the model's norm (transformers'
        # last_hidden_state; ACE-Step's text encoder)
        self.final_norm = np.ascontiguousarray(self.W.get(prefix + "norm.weight"), dtype=np.float32) \
            if final_norm else None
        self.width = cfg["hidden_size"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim") or self.width // self.heads
        self.eps = float(cfg["rms_norm_eps"])
        self.inv = rope_inv(self.hd, cfg.get("rope_theta", 10000.0))
        # Qwen3-VL: the axis (0 T, 1 H, 2 W) of each frequency of the interleaved MRoPE
        rs = cfg.get("rope_scaling") or {}
        self.axis = None
        if rs.get("mrope_section"):
            assert rs.get("mrope_interleaved")
            self.axis = np.zeros(self.hd // 2, np.int64)
            for d in (1, 2):
                self.axis[d:3 * rs["mrope_section"][d]:3] = d
        self.prefix = prefix
        n = max(self.taps)
        assert n <= cfg["num_hidden_layers"]
        self.layers = qwen3_layers(self.W, prefix + "layers.", n, self.hd, gpu, cpu_bf16)
        self.inter = self.layers[0]["gate"].shape[0]
        self.runner = _Runner(self._build, keep=3, keep_gpu=2)

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def embed(self, ids):
        """The rows of embed_tokens of ids (float32)."""
        ids = np.asarray(ids).reshape(-1)
        name = self.prefix + "embed_tokens.weight"
        u, inv = np.unique(ids, return_inverse=True)
        rows = np.stack([self.W.get_row(name, int(t)) for t in u])
        return rows[inv].astype(np.float32)

    def program(self, n, rows, media=None, gpu=False):
        """The program of n prompts of rows tokens each: inputs "x" (n rows,
        width: the embeddings), "lo", "hi" (n rows, int32: a row's keys),
        "pos" (n rows, int32); outputs "t<j>" the hidden states after
        self.taps[j] layers. media (one prompt with images): (blocks, deep),
        the (start, end) rows of each image and the deepstack layers: inputs
        "cos", "sin" (rows, hd / 2: the MRoPE in place of pos) and "ds<i>"
        (the images' rows, width) added to them after layer i < deep."""
        w, H, KV, hd = self.width, self.heads, self.kv_heads, self.hd
        R = n * rows
        c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=self.inter)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        zi = lambda *shape: np.zeros(shape, np.int32)  # noqa: E731
        c.env.update(x=z(R, w), h=z(R, w), q=z(R, H * hd), k=z(R, KV * hd), v=z(R, KV * hd),
                     qn=z(R, H * hd), kn=z(R, KV * hd), o=z(R, H * hd), t=z(R, w),
                     g=z(R, self.inter), u=z(R, self.inter), lo=zi(R), hi=zi(R), pos=zi(R))
        for j in range(len(self.taps)):
            c.env["t%d" % j] = z(R, w)
        blocks, deep = media if media is not None else ((), 0)
        if media is not None:
            assert n == 1
            c.env.update(cos=z(R, hd // 2), sin=z(R, hd // 2))
            img = sum(e - s_ for s_, e in blocks)
            for i in range(deep):
                c.env["ds%d" % i] = z(img, w)
                r0 = 0
                for b, (s_, e) in enumerate(blocks):
                    c.env["x_%d" % b] = c.env["x"][s_:e]
                    c.env["ds%d_%d" % (i, b)] = c.env["ds%d" % i][r0:r0 + e - s_]
                    r0 += e - s_
        rope = [("enc_rope_tab", "qn", "cos", "sin", H, hd, True), ("enc_rope_tab", "kn", "cos", "sin", KV, hd, True)] \
            if media is not None else \
            [("enc_rope1d", "qn", "pos", self.inv, H, hd), ("enc_rope1d", "kn", "pos", self.inv, KV, hd)]
        forms = []
        for i, L in enumerate(self.layers):
            forms += [
                ("set", "h", ("enc_rms", "x", L["ln_a"], w)),
                ("set", "q", ("enc_linear", L["q"], "h")),
                ("set", "k", ("enc_linear", L["k"], "h")),
                ("set", "v", ("enc_linear", L["v"], "h")),
                ("set", "qn", ("enc_rms", "q", L["qn"], hd)),
                ("set", "kn", ("enc_rms", "k", L["kn"], hd)),
                *rope,
                ("set", "o", ("enc_attn_range", "qn", "kn", "v", "lo", "hi", H, KV, hd)),
                ("set", "t", ("enc_linear", L["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", L["ln_m"], w)),
                ("set", "g", ("enc_linear", L["gate"], "h")),
                ("set", "u", ("enc_linear", L["up"], "h")),
                ("set", "u", ("enc_silu_mul", "g", "u")),
                ("set", "t", ("enc_linear", L["down"], "u")),
                ("enc_add", "x", "t"),
            ]
            if i < deep:
                forms += [("enc_add", "x_%d" % b, "ds%d_%d" % (i, b)) for b in range(len(blocks))]
            for j, tap in enumerate(self.taps):
                if tap == i + 1:
                    if self.final_norm is not None and tap == len(self.layers):
                        forms.append(("set", "t%d" % j, ("enc_rms", "x", self.final_norm, w)))
                    else:
                        forms.append(("set", "t%d" % j, ("copy", "x")))
        c.compile(("seq", *forms))
        return c.p.finish()

    def mrope(self, pos):
        """cos and sin (rows, hd / 2) of the interleaved MRoPE at the
        positions pos (3, rows): each frequency at its axis's position (a
        float32 product, as transformers')."""
        ang = (pos[self.axis].T.astype(np.float32) * self.inv[None, :]).astype(np.float32)
        a64 = ang.astype(np.float64)
        return np.cos(a64).astype(np.float32), np.sin(a64).astype(np.float32)

    def hidden(self, ids, tokens, media=None):
        """The hidden states of the taps (a list, each (n, rows, width)) of
        n prompts: ids (n, rows), tokens (n,) the real tokens of each (the
        rest padding). media (one prompt with images): a dict of "blocks" the
        (start, end) tokens of each image, "rows" (their tokens, width) the
        vision tower's rows in place of their embeddings, "pos" (3, rows)
        the MRoPE positions, "deep" the deepstack rows (a list, each (their
        tokens, width))."""
        ids = np.asarray(ids).reshape(len(tokens), -1)
        n, rows = ids.shape
        i = np.arange(rows)
        lo = np.concatenate([np.full(rows, j * rows) for j in range(n)]).astype(np.int32)
        hi = np.concatenate([j * rows + np.minimum(i + 1, max(int(t), 1)) for j, t in enumerate(tokens)]).astype(np.int32)
        inputs = dict(x=self.embed(ids), lo=lo, hi=hi, pos=np.tile(i, n).astype(np.int32))
        key = (n, rows)
        if media is not None:
            blocks = tuple((int(s_), int(e)) for s_, e in media["blocks"])
            r0 = 0
            for s_, e in blocks:
                inputs["x"][s_:e] = media["rows"][r0:r0 + e - s_]
                r0 += e - s_
            del inputs["pos"]
            inputs["cos"], inputs["sin"] = self.mrope(media["pos"])
            for j, d in enumerate(media["deep"]):
                inputs["ds%d" % j] = d
            key = (n, rows, (blocks, len(media["deep"])))
        prog, g = self.runner.get(key, self.gpu)
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        outs = ["t%d" % j for j in range(len(self.taps))]
        if g is not None:
            g.run()
            for name in outs:
                g.download(name)
        else:
            prog.run()
        return [prog.names[name].reshape(n, rows, -1).copy() for name in outs]
