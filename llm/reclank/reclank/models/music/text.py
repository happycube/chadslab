"""The text towers of the music embeddings (docs/MUSIC_EMBED_PLAN.md):
RoBERTa base for CLAP, XLM-R base and MuLan's 8 layers for MuQ-MuLan.

Ported from transformers (src/transformers/models/roberta/modeling_roberta.py, Copyright
2018 The Google AI Language Team Authors and The HuggingFace Inc. team;
models/xlm_roberta/modeling_xlm_roberta.py, Copyright 2019 Facebook AI Research and the
HuggingFace Inc. team; models/clap/modeling_clap.py, Copyright 2023 The LAION-AI Team and
The HuggingFace Team; Apache License 2.0) and MuQ (tencent-ailab/MuQ,
src/muq/muq_mulan/modules/transformer.py; MIT License; Copyright (c) Tencent.). See
THIRD_PARTY_NOTICES.md.

RoBERTa and XLM-R are the same post-LN BERT encoder (transformers'
RobertaModel, XLMRobertaModel):

    the word row + the position row (padding_idx + 1 + the index: 2, 3, ...)
        + the token type row 0, a LayerNorm
    each layer: q k v with bias (scale 1/8: folded into q), the attention of
        the text over itself, the output with bias, + x, a LayerNorm; the FFN
        (a linear with bias, GELU erf, a linear with bias), + x, a LayerNorm

MuLan's layers (muq_mulan/modules/transformer.py) follow XLM-R's last hidden
state, each pre-LN (a LayerNorm with no bias):

    q (8 heads of 64) and k v with no bias, the L2 norm of each head of q
        and k times q_scale and k_scale, scale 8, the output, + x
    a LayerNorm, the linear to the halves x and gate (2048 each), GEGLU
        (gelu_erf(gate) x), the linear back, + x

The L2 norm of a head of 64 is its RMS norm over 8, so q takes the RMS norm
with the weight q_scale / 8 and k the weight k_scale (scale 1 in
ENC_ATTN_RANGE); the eps 1e-24 stands for F.normalize's clamp of the norm at
1e-12.

Each tower runs as one program (the records ENC_*) on the CPU or the GPU,
with the float32 matrices of the checkpoint (as trained) on both: in BF12
(gpu_bf12, what the audio towers take on the GPU) MuQ-MuLan's text vectors
moved by up to 4e-4 (1 - cos; CLAP's 4e-5), and a query costs milliseconds. The texts are
packed, each over its own keys (ENC_ATTN_RANGE); the rows past them are
zeros, each over itself alone. The word table stays in the memory map (the
rows of the ids are read); the pooling and the projections to 512 are on
the host.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.core import ops
from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner, _shape, to_bf12
from reclank.formats.safetensors import SafeTensors
from reclank.models.embedgemma2.model import _bucket
from reclank.runtime.threads import team_threads

# the split pattern of GPT-2 (RoBERTa's ByteLevel pre-tokenizer with use_regex)
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def gpu_bf12(lins, label):
    """The matrices of a GPU program in BF12: the float32 W rounded to
    bfloat16, then BF12 rows (encoders.runtime.to_bf12). A matrix that the
    rule refuses (cols not a multiple of 32: MuQ's first conv, CLAP's patch
    linear) keeps its float32 W. RECLANK_MUSIC_F32=1 keeps them all."""
    if os.environ.get("RECLANK_MUSIC_F32", "0") == "1":
        return
    kept = 0
    for lin in lins:
        w32 = lin.w
        if w32 is None:
            continue
        if w32.shape[1] % 32 or w32.shape[0] < 2:
            kept += 1
            continue
        lin.w = ops.to_bf16(w32)
        if not to_bf12(lin):
            lin.w = w32
            kept += 1
    if kept:
        print("%s: %d of %d matrices stay float32 (not BF12)" % (label, kept, len(lins)), flush=True)


def _lin(W, name, bias=True, scale=None, rows=None):
    """A linear of float32 W (out, in) and its bias; scale multiplies both;
    rows: a slice of the rows of W (and the bias)."""
    w = np.asarray(W.get(name + ".weight"), dtype=np.float32)
    b = np.asarray(W.get(name + ".bias"), dtype=np.float32) if bias else None
    if rows is not None:
        w = w[rows]
        b = b[rows] if b is not None else None
    if scale is not None:
        w = w * np.float32(scale)
        b = b * np.float32(scale) if b is not None else None
    lin = _Plain(np.ascontiguousarray(w))
    lin.b = None if b is None else np.ascontiguousarray(b)
    return lin


class TextTower:
    """A BERT encoder of a checkpoint (prefix: its names, "text_model." or
    "mulan.text.model."), and MuLan's layers after it (mulan: their prefix).
    rows(seqs) gives the rows of the last layer, (sum of tokens, 768)."""

    def __init__(self, W, prefix, layers=12, heads=12, eps=1e-5, mulan=None, mulan_layers=0,
                 gpu=False, cpu_threads=None):
        self.W = W
        self.gpu = bool(gpu)
        self.cpu_threads = cpu_threads
        self.heads, self.eps = heads, float(eps)
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        e = prefix + "embeddings."
        self.word = W.get(e + "word_embeddings.weight")         # a view of the map
        self.pos = f(e + "position_embeddings.weight")
        self.type0 = f(e + "token_type_embeddings.weight")[0]
        self.width = self.pos.shape[1]
        self.max_tokens = self.pos.shape[0] - 2
        self.emb_ln = (f(e + "LayerNorm.weight"), f(e + "LayerNorm.bias"))
        hd = self.width // heads
        self.blk = []
        for i in range(layers):
            q = prefix + "encoder.layer.%d." % i
            self.blk.append(dict(
                q=_lin(W, q + "attention.self.query", scale=hd ** -0.5),
                k=_lin(W, q + "attention.self.key"), v=_lin(W, q + "attention.self.value"),
                o=_lin(W, q + "attention.output.dense"),
                ln1=(f(q + "attention.output.LayerNorm.weight"), f(q + "attention.output.LayerNorm.bias")),
                up=_lin(W, q + "intermediate.dense"), down=_lin(W, q + "output.dense"),
                ln2=(f(q + "output.LayerNorm.weight"), f(q + "output.LayerNorm.bias"))))
        self.mulan = []
        for i in range(mulan_layers):
            q = mulan + "%d." % i
            inner = W.header[q + "0.to_q.weight"]["shape"][0]
            ff = W.header[q + "1.4.weight"]["shape"][1]
            self.mulan.append(dict(
                inner=inner, ff=ff,
                n1=f(q + "0.norm.learned_gamma"),
                q=_lin(W, q + "0.to_q", bias=False),
                k=_lin(W, q + "0.to_kv", bias=False, rows=slice(0, inner)),
                v=_lin(W, q + "0.to_kv", bias=False, rows=slice(inner, 2 * inner)),
                qs=f(q + "0.q_scale") / np.float32(8.0), ks=f(q + "0.k_scale"),
                o=_lin(W, q + "0.to_out.0", bias=False),
                n2=f(q + "1.0.learned_gamma"),
                fx=_lin(W, q + "1.1", bias=False, rows=slice(0, ff)),
                fg=_lin(W, q + "1.1", bias=False, rows=slice(ff, 2 * ff)),
                fo=_lin(W, q + "1.4", bias=False)))
        self.runner = _Runner(self._team, keep=6, keep_gpu=4)

    def _team(self, n, gpu):
        p = self.program(n, gpu)
        if not gpu:     # word 3 of the header: the team (cpu_threads, or the CPUs of the taskset)
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def program(self, n, gpu=False):
        """The program of n rows: inputs "e" (the embeddings), "lo", "hi" (the
        keys of each row); output "x" (n, 768)."""
        w, H = self.width, self.heads
        hd = w // H
        inter = _shape(self.blk[0]["up"])[0]
        mi = max([m["inner"] for m in self.mulan] or [w])
        mf = max([m["ff"] for m in self.mulan] or [w])
        c = _EncCompiler(self.eps, pack=not gpu, rows=n, kmax=max(inter, mf))
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(e=z(n, w), x=z(n, w), t=z(n, w), h=z(n, w), q=z(n, w), k=z(n, w), v=z(n, w), o=z(n, w),
                     f=z(n, inter), mq=z(n, mi), mk=z(n, mi), mv=z(n, mi), mo=z(n, mi), fx=z(n, mf), fg=z(n, mf),
                     lo=np.zeros(n, np.int32), hi=np.zeros(n, np.int32))
        forms = [("set", "x", ("enc_lnorm", "e", self.emb_ln[0], self.emb_ln[1], w))]
        for b in self.blk:
            forms += [
                ("set", "q", ("enc_linear", b["q"], "x")),
                ("set", "k", ("enc_linear", b["k"], "x")),
                ("set", "v", ("enc_linear", b["v"], "x")),
                ("set", "o", ("enc_attn_range", "q", "k", "v", "lo", "hi", H, H, hd)),
                ("set", "t", ("enc_linear", b["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "x", ("enc_lnorm", "x", b["ln1"][0], b["ln1"][1], w)),
                ("set", "f", ("enc_linear", b["up"], "x")),
                ("set", "f", ("enc_gelu", "f", True)),
                ("set", "t", ("enc_linear", b["down"], "f")),
                ("enc_add", "x", "t"),
                ("set", "x", ("enc_lnorm", "x", b["ln2"][0], b["ln2"][1], w)),
            ]
        for m in self.mulan:
            mh = m["inner"] // 64
            forms += [
                ("set", "h", ("enc_lnorm", "x", m["n1"], None, w)),
                ("set", "mq", ("enc_linear", m["q"], "h")),
                ("set", "mq", ("enc_rms", "mq", m["qs"], 64, 1e-24)),
                ("set", "mk", ("enc_linear", m["k"], "h")),
                ("set", "mk", ("enc_rms", "mk", m["ks"], 64, 1e-24)),
                ("set", "mv", ("enc_linear", m["v"], "h")),
                ("set", "mo", ("enc_attn_range", "mq", "mk", "mv", "lo", "hi", mh, mh, 64)),
                ("set", "t", ("enc_linear", m["o"], "mo")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", m["n2"], None, w)),
                ("set", "fx", ("enc_linear", m["fx"], "h")),
                ("set", "fg", ("enc_linear", m["fg"], "h")),
                ("set", "fg", ("enc_gelu_mul", "fg", "fx", True)),
                ("set", "t", ("enc_linear", m["fo"], "fg")),
                ("enc_add", "x", "t"),
            ]
        c.compile(("seq", *forms))
        return c.p.finish()

    def _inputs(self, seqs, n):
        e = np.zeros((n, self.width), np.float32)
        r = np.arange(n, dtype=np.int32)
        lo, hi = r.copy(), r + 1
        a = 0
        for s in seqs:
            m = len(s)
            ids = np.asarray(s, dtype=np.int64)
            e[a:a + m] = self.word[ids] + self.pos[2:2 + m] + self.type0
            lo[a:a + m], hi[a:a + m] = a, a + m
            a += m
        return dict(e=e, lo=lo, hi=hi)

    def rows(self, seqs):
        """The rows of the last layer for lists of ids, packed (sum of tokens,
        768)."""
        for s in seqs:
            if len(s) > self.max_tokens:
                raise ValueError("a text has %d tokens; the model takes at most %d" % (len(s), self.max_tokens))
        m = sum(len(s) for s in seqs)
        n = _bucket(m)
        return self.runner.run(n, self.gpu, self._inputs(seqs, n), "x")[:m]


def _norm(y):
    return y / np.linalg.norm(y, axis=-1, keepdims=True)


class ClapText:
    """CLAP's text tower: RoBERTa, the pooler (dense with tanh on the first
    row), text_projection (linear, ReLU, linear), the L2 norm. path: a
    directory of scripts/convert_music.py --clap."""

    def __init__(self, path, gpu=False, cpu_threads=None, W=None):
        from reclank.models.qwen.tokenizer import QwenTokenizer
        self.W = W or SafeTensors(os.path.join(path, "model.safetensors"))
        tc = json.load(open(os.path.join(path, "config.json"))).get("text_config") or {}
        # transformers' ClapTextConfig: eps 1e-12 (the reference of the trial)
        self.tower = TextTower(self.W, "text_model.", layers=tc.get("num_hidden_layers", 12),
                               heads=tc.get("num_attention_heads", 12), eps=tc.get("layer_norm_eps", 1e-12),
                               gpu=gpu, cpu_threads=cpu_threads)
        f = lambda n: np.ascontiguousarray(self.W.get(n), dtype=np.float32)  # noqa: E731
        self.pool = (f("text_model.pooler.dense.weight").T.copy(), f("text_model.pooler.dense.bias"))
        self.proj = [(f("text_projection.linear%d.weight" % i).T.copy(), f("text_projection.linear%d.bias" % i))
                     for i in (1, 2)]
        data = json.load(open(os.path.join(path, "tokenizer.json")))
        data["pre_tokenizer"] = {"type": "Split", "pattern": {"Regex": GPT2_PATTERN}}
        self.tok = QwenTokenizer(data)
        self.bos, self.eos = self.tok.special["<s>"], self.tok.special["</s>"]

    def tokens(self, text):
        return [self.bos] + self.tok.encode(text) + [self.eos]

    def embed_ids(self, seqs, tokens=False):
        x = self.tower.rows(seqs)
        starts = np.cumsum([0] + [len(s) for s in seqs])[:-1]
        p = np.tanh(x[starts] @ self.pool[0] + self.pool[1])
        h = np.maximum(p @ self.proj[0][0] + self.proj[0][1], 0)
        y = _norm(h @ self.proj[1][0] + self.proj[1][1])
        return (y, x) if tokens else y

    def embed(self, texts):
        return self.embed_ids([self.tokens(t) for t in texts])


class MuqText:
    """MuQ-MuLan's text tower: XLM-R, MuLan's 8 layers, the mean of the rows,
    text_to_latents, the L2 norm. path: a directory of
    scripts/convert_music.py --muq."""

    def __init__(self, path, gpu=False, cpu_threads=None, W=None):
        from reclank.models.music.unigram import UnigramTokenizer
        self.W = W or SafeTensors(os.path.join(path, "model.safetensors"))
        cfg = json.load(open(os.path.join(path, "config.json")))["text_transformer"]
        self.tower = TextTower(self.W, "mulan.text.model.", mulan="mulan.text.transformer.layers.",
                               mulan_layers=int(cfg["tf_depth"]), gpu=gpu, cpu_threads=cpu_threads)
        f = lambda n: np.ascontiguousarray(self.W.get(n), dtype=np.float32)  # noqa: E731
        self.latent = (f("mulan.text_to_latents.weight").T.copy(), f("mulan.text_to_latents.bias"))
        self.tok = UnigramTokenizer(os.path.join(path, "tokenizer.json"))

    def tokens(self, text):
        return self.tok.encode(text)

    def embed_ids(self, seqs, tokens=False):
        x = self.tower.rows(seqs)
        b = np.cumsum([0] + [len(s) for s in seqs])
        means = np.stack([x[b[i]:b[i + 1]].mean(0) for i in range(len(seqs))])
        y = _norm(means @ self.latent[0] + self.latent[1])
        return (y, x) if tokens else y

    def embed(self, texts):
        return self.embed_ids([self.tokens(t) for t in texts])
