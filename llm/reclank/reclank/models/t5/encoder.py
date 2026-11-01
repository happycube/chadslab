"""umT5's encoder (transformers' UMT5EncoderModel: google/umt5-xxl, Wan 2.2's text encoder) as
an encoder program, and the prompt embeddings of diffusers' WanPipeline (_get_t5_prompt_embeds).

Ported from transformers (src/transformers/models/umt5/modeling_umt5.py; Apache License 2.0;
Copyright 2023 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team) and diffusers
(src/diffusers/pipelines/wan/pipeline_wan.py; Apache License 2.0; Copyright 2025 The Wan Team
and The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

    the tokens' rows of shared (the embedding; no scale) on the host
    a block: RMS norm (T5's: no mean, no bias), q k v (no bias), the attention with no scale
        of 1/sqrt(d_kv), each score plus the block's own relative position bias (umT5: every
        block has one, T5 only the first) of the head by the bucket of the distance (key -
        query: bidirectional, 32 buckets, exact under 8 each way, then logarithmic up to 128),
        the padding keys masked; the output; + x; RMS norm, gelu_new(wi_0 h) (the tanh form)
        times wi_1 h, wo; + x
    the final RMS norm: last_hidden_state

The pipeline's prompt_embeds: prompt_clean, the tokens of each prompt (at most max_len, </s>
last), the last hidden state of its tokens, zero rows after them to max_len. The rows of the
padding do not change the tokens' rows (masked as keys), so a program runs only the rows of the
prompt (rounded up to 16; full: max_len rows, as the pipeline). A program takes n prompts of
the same rows, each its own keys; encode runs the prompts of the same rows together, each
other prompt in a program of its own rows (the same bits as in any program).

The matrices: the checkpoint's bfloat16 rows as they are (KQ_BF16X16 of a CPU program: the same
values in float32 products), made at load from the file's map, whose pages are dropped then
(the RAM holds one copy: 9 GB for the XXL's 24 blocks). A CPU program only (ENC_ATTN_REL).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.encoders.runtime import _EncCompiler
from reclank.encoders.runtime import _Runner
from reclank.formats.safetensors import Shards, drop_pages
from reclank.models.qwen3.dense import lin16
from reclank.models.t5.tokenizer import T5Tokenizer, prompt_clean
from reclank.runtime.threads import team_threads


def buckets(dist, num_buckets, max_distance):
    """The relative position buckets of the distances dist (key - query) of an encoder (T5's
    _relative_position_bucket, bidirectional), in transformers' float32 arithmetic."""
    dist = np.asarray(dist, np.int64)
    nb = num_buckets // 2
    out = (dist > 0).astype(np.int64) * nb
    a = np.abs(dist)
    exact = nb // 2
    with np.errstate(divide="ignore"):
        r = np.log(a.astype(np.float32) / np.float32(exact)) / np.float32(math.log(max_distance / exact))
    r = r * np.float32(nb - exact)
    large = exact + np.where(a < exact, 0, r).astype(np.int64)
    large = np.minimum(large, nb - 1)
    return out + np.where(a < exact, a, large)


class UMT5Encoder:
    """The encoder of a UMT5EncoderModel directory (config.json and its safetensors; Wan's
    text_encoder). tokenizer: its tokenizer.json (default: ../tokenizer/tokenizer.json, Wan's
    layout). encode(prompts) gives the pipeline's prompt_embeds."""

    def __init__(self, path, gpu=False, cpu_threads=None, tokenizer=None):
        assert not gpu, "umT5's encoder is a CPU program (ENC_ATTN_REL)"
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg["model_type"] in ("umt5", "t5") and cfg["is_gated_act"] and cfg["dense_act_fn"] == "gelu_new"
        self.cfg = cfg
        self.cpu_threads = cpu_threads
        self.tok = T5Tokenizer(tokenizer or os.path.join(os.path.dirname(os.path.abspath(path)), "tokenizer",
                                                         "tokenizer.json"))
        self.W = W = Shards(path)
        self.width, self.heads, self.hd = cfg["d_model"], cfg["num_heads"], cfg["d_kv"]
        self.inter = cfg["d_ff"]
        self.eps = float(cfg["layer_norm_epsilon"])
        self.nb, self.maxd = cfg["relative_attention_num_buckets"], cfg["relative_attention_max_distance"]
        # umT5: a bias table in every block; T5: the first block's for all
        own = cfg["model_type"] == "umt5"
        f = lambda name: np.ascontiguousarray(W.get(name), dtype=np.float32)  # noqa: E731
        self.layers = []
        for i in range(cfg["num_layers"]):
            r = "encoder.block.%d.layer." % i
            L = lambda name: self._lin(r + name)  # noqa: E731
            self.layers.append(dict(
                ln_a=f(r + "0.layer_norm.weight"), ln_m=f(r + "1.layer_norm.weight"),
                q=L("0.SelfAttention.q"), k=L("0.SelfAttention.k"), v=L("0.SelfAttention.v"),
                o=L("0.SelfAttention.o"),
                rel=f("encoder.block.%d.layer.0.SelfAttention.relative_attention_bias.weight" % (i if own else 0)),
                wi0=L("1.DenseReluDense.wi_0"), wi1=L("1.DenseReluDense.wi_1"), wo=L("1.DenseReluDense.wo")))
        self.final_norm = f("encoder.final_layer_norm.weight")
        self.runner = _Runner(self._build, keep=3)

    def _lin(self, name):
        """A linear of bfloat16 W: KQ_BF16X16 made now (enc_linear's lin.x16), then the pages of
        the map dropped (W stays a view of the map: the record reads x16 alone)."""
        lin = lin16(self.W, name, False, bias=False, cpu_bf16=True)
        assert lin.w.dtype == np.uint16
        lin.x16 = cops.kq_pack_x16f(lin.w, 1, *lin.w.shape)
        drop_pages(lin.w)
        return lin

    def _build(self, key, gpu):
        p = self.program(*key)
        p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def rel_table(self, rel, span):
        """A block's bias (heads, 2 span - 1) of the distances -(span - 1) .. span - 1 (rel:
        the block's (buckets, heads))."""
        b = buckets(np.arange(-(span - 1), span), self.nb, self.maxd)
        return np.ascontiguousarray(rel[b].T, dtype=np.float32)

    def embed(self, ids):
        """The rows of shared of ids (float32)."""
        ids = np.asarray(ids).reshape(-1)
        u, inv = np.unique(ids, return_inverse=True)
        rows = np.stack([self.W.get_row("shared.weight", int(t)) for t in u])
        return rows[inv].astype(np.float32)

    def program(self, n, rows, taps=False):
        """The program of n prompts of rows tokens each: inputs "x" (n rows, width: the
        embeddings), "lo", "hi" (n rows, int32: a row's keys); output "y" the last hidden state
        (after the final norm); taps: "t<i>" the hidden state after block i too."""
        w, H, hd, F = self.width, self.heads, self.hd, self.inter
        R = n * rows
        c = _EncCompiler(self.eps, pack=True, rows=R, kmax=F)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(x=z(R, w), h=z(R, w), q=z(R, H * hd), k=z(R, H * hd), v=z(R, H * hd), o=z(R, H * hd),
                     t=z(R, w), g=z(R, F), u=z(R, F), y=z(R, w), lo=np.zeros(R, np.int32), hi=np.zeros(R, np.int32))
        forms = []
        for i, L in enumerate(self.layers):
            forms += [
                ("set", "h", ("enc_rms", "x", L["ln_a"], w)),
                ("set", "q", ("enc_linear", L["q"], "h")),
                ("set", "k", ("enc_linear", L["k"], "h")),
                ("set", "v", ("enc_linear", L["v"], "h")),
                ("set", "o", ("enc_attn_rel", "q", "k", "v", "lo", "hi", self.rel_table(L["rel"], rows), H, hd, rows)),
                ("set", "t", ("enc_linear", L["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", L["ln_m"], w)),
                ("set", "g", ("enc_linear", L["wi0"], "h")),
                ("set", "u", ("enc_linear", L["wi1"], "h")),
                ("set", "u", ("enc_gelu_mul", "g", "u")),
                ("set", "t", ("enc_linear", L["wo"], "u")),
                ("enc_add", "x", "t"),
            ]
            if taps:
                c.env["t%d" % i] = z(R, w)
                forms.append(("set", "t%d" % i, ("copy", "x")))
        forms.append(("set", "y", ("enc_rms", "x", self.final_norm, w)))
        c.compile(("seq", *forms))
        return c.p.finish()

    def tokenize(self, prompts, max_len=512):
        """The ids of each prompt after prompt_clean (at most max_len, </s> last)."""
        return [np.array(self.tok.encode(prompt_clean(p), max_len), np.int32) for p in prompts]

    def run(self, ids, rows=None, taps=False):
        """The last hidden state (n, rows, width) of the prompts' ids (a list), and with taps
        the hidden states after each block (layers, n, rows, width). rows: the program's
        (default: the longest prompt rounded up to 16)."""
        n = len(ids)
        lens = [len(t) for t in ids]
        rows = rows or -(-max(lens) // 16) * 16
        assert max(lens) <= rows
        x = np.full((n, rows), self.tok.pad_id, np.int32)
        for j, t in enumerate(ids):
            x[j, :len(t)] = t
        # each row (the padding rows too) attends to its prompt's tokens
        lo = np.repeat(np.arange(n) * rows, rows).astype(np.int32)
        hi = np.repeat(np.arange(n) * rows + np.array(lens), rows).astype(np.int32)
        prog, _ = self.runner.get((n, rows, bool(taps)), False)
        prog.names["x"][...] = self.embed(x)
        prog.names["lo"][...] = lo
        prog.names["hi"][...] = hi
        prog.run()
        y = prog.names["y"].reshape(n, rows, -1).copy()
        if not taps:
            return y
        return y, np.stack([prog.names["t%d" % i].reshape(n, rows, -1) for i in range(len(self.layers))])

    def encode(self, prompts, max_len=512, full=False):
        """WanPipeline's prompt_embeds of prompts (a str or a list): float32 (n, max_len, width),
        each prompt's token rows then zeros; and the tokens of each (n,). full: a program of
        max_len rows (as the pipeline; the same rows of the tokens)."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        ids = self.tokenize(prompts, max_len)
        lens = np.array([len(t) for t in ids], np.int64)
        out = np.zeros((len(prompts), max_len, self.width), np.float32)
        # the prompts of the same rows in one program (a prompt's rows have the same bits in a
        # program of others): a short prompt beside a long one would pay for the long one's rows
        groups = {}
        for j, k in enumerate(lens):
            groups.setdefault(max_len if full else -(-int(k) // 16) * 16, []).append(j)
        for rows, js in groups.items():
            y = self.run([ids[j] for j in js], rows)
            for g, j in enumerate(js):
                out[j, :lens[j]] = y[g, :lens[j]]
        return out, lens
