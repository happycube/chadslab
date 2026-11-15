"""Nemotron 3.5 Parse as programs (docs/NEMOTRON_PARSE.md): in C on the CPU or on the GPU, from the
converted F20 directory (scripts/convert_nemoparse_f20.py: the matrices in F20, 16 significant bits of
the checkpoint's float32) or from the download (float32 safetensors: float32 matrices on the CPU, F20
made at load on the GPU, the same bits as the file's). The sums, the activations and the attention are
float32. model.py is the reference, op for op.

Three kinds of program, records of the encoders (encoders/runtime.py, ir/program.py ENC_*):

    ("vit", R, C)   a page of R x C patches: inputs "p" [10 + R C, 768] (zeros in the 10 prefix
                    rows) and "x0" [10 + R C, 1152] (the CLS and register rows, then the position
                    rows of the host: model.position_rows); RADIO's 27 blocks (the attention in
                    float32: ENC_ATTN_TILE on the GPU, ENC_ATTN_RANGE in C; fc1 and fc2 padded from
                    4304 to 4320 with zeros: model.program_layout); the neck: conv1 and LN1 over
                    every row, conv2 a linear of the gathered rows of 4 patches, LN2 ("feat"); the
                    summary (rows 0 and 1 gathered), sum_proj, LN3 ("summ")
    ("pre", T)      the decoder over T rows "e" (the visual rows, then the prompt's embeddings x 32,
                    from the host): their keys and values into the caches' rows 0 .. T - 1, the
                    causal attention (ENC_ATTN_RANGE: 16 query heads on 8 KV heads), the logits
                    of the last row ("logits" [1, vocab])
    ("step", r, span, mtp)
                    r tokens ("tok", their rows "kvi" in the caches, the keys "lo" .. "hi" each
                    sees: ENC_ATTN_KV over at most span rows of the cache): "logits" [r, vocab]

With mtp (a GPU's program: GP_ARGMAX) a program also gives each row's draft of the checkpoint's
one-step MTP head (the vLLM plugin's proposer: extra_head(h + extra_proj(embed(a) x 32)) with h the
row's final hidden state and a its logits' argmax, then the tied head and its argmax): "am" (a) and
"draft". The host checks a against its own choice (the repetition penalty can change it), then feeds
the token and the draft as a step of two; a row's logits are the same bits as alone (each kernel of a
step keeps a row's sums apart), so the output is the plain greedy one, in fewer steps (about 1.85
tokens a step on real pages).

The decoder's programs share the caches of keys and values ("k<l>", "v<l>": max_seq rows of 512
float32) through one _Runner (one mirror); the encoder has a runner of its own. q takes the
attention's scale 1/8 after q_proj (a power of two: exact). The repetition penalty and the choice of
a token run on the host (model.Rules).

    e = ParseEngine("MODELS/Nemotron-3.5-Parse", gpu=True)
    r = e.parse("page.png")        # {"ids", "text", "encode_s", "decode_s", ...}

Ported from the Nemotron 3.5 Parse checkpoint's code (hf_nemotron_parse_modeling.py; the MTP proposer of
vllm_tied_patch/vllm_radio_mbart_mtp.py; OpenMDW License 1.1; Copyright (c) 2026, NVIDIA CORPORATION).
See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from reclank.backends.cpu import cops
from reclank.encoders.runtime import RESIDENCY, _EncCompiler, _Plain, _Runner, _shape, to_f20
from reclank.ir import program as P
from reclank.models.gemma4.e4b import _KQ
from reclank.models.nemoparse.model import (MAX_NEW, PATCH, PREFIX, POS_SIDE, PROMPT, REP_PENALTY, VM,
                                            ParseConfig, Rules, Tokens, Weights, blocks, page_pixels,
                                            position_rows)
from reclank.runtime.threads import team_threads

VIT_HEADS = 16
SPAN = 2048                         # a step program's keys: the context rounded up to a multiple of this
# RECLANK_NPARSE_MTP=0: no drafts of the checkpoint's MTP head (a GPU's steps take one by default)
MTP = os.environ.get("RECLANK_NPARSE_MTP", "1") != "0"
# RECLANK_NPARSE_DFLASH: the DFlash2 drafter (dflash2/ of the model's directory): "auto" (the default) on
# the CPU only (on the GPU its rounds' host work made it slower than MTP: 543 against 705 tokens/s on six
# pages of a 3090), "1" on either, "0" off; RECLANK_NPARSE_DRAFTS: the drafts of a round (default 15: a verify step of 16 rows, the
# most that k_kq_f20_gemv takes a row at a time on the GPU, MT_MAX, so each row's bits are a single step's)
DFLASH = os.environ.get("RECLANK_NPARSE_DFLASH", "auto")
DRAFTS = int(os.environ.get("RECLANK_NPARSE_DRAFTS", "15"))
# RECLANK_NPARSE_ATTN=f16: RADIO's attention on the GPU as ENC_ATTN (the tensor cores with q, k and v in
# float16, as Bolt's RADIO) instead of float32 (ENC_ATTN_TILE): a precision the owner has not approved
ATTN_F16 = os.environ.get("RECLANK_NPARSE_ATTN", "f32") == "f16"


def _f20_lin(rows, gpu, b=None):
    """A linear of F20 rows (uint8 [rows, row bytes] of cols values): KQ_F20 rows for KQ_LINEAR on the
    GPU, KQ_F20X16 for ENC_LINEAR's wbf 3 on the CPU (float32 products of the F20 values)."""
    data, m, k = rows
    lin = _Plain(None)
    data = np.ascontiguousarray(data).reshape(-1)
    if gpu:
        lin.kq = _KQ(data, cops.KQ_F20, m, k)
    else:
        lin.f20 = _KQ(cops.kq_pack_f20x16(data, m, k), cops.KQ_F20X16, m, k)
    lin.b = None if b is None else np.ascontiguousarray(b, np.float32)
    return lin


class ParseEngine:
    """The model of a checkpoint directory on the CPU or the GPU. parse() gives the raw output of a
    page; encode() and the decoder's prefill() and step() are its parts (the checks)."""

    def __init__(self, path, gpu=False, cpu_threads=None, drafter=True):
        self.path = path
        self.gpu = bool(gpu)
        self.cpu_threads = cpu_threads
        self.cfg = cfg = ParseConfig(path)
        self.tok = Tokens(path)
        self.W = W = Weights(path)
        f = W.f32
        lins = []

        def lin(name, bias=True, rows=None):
            """The linear of matrix name (and its bias); rows: a slice of its rows."""
            b = f(name.rsplit(".", 1)[0] + ".bias") if bias else None
            q = W.f20(name)
            if q is not None:
                if rows is not None:
                    q = (q[0][rows], rows.stop - rows.start, q[2])
                return _f20_lin(q, self.gpu, None if b is None else b if rows is None else b[rows])
            w = f(name)
            L = _Plain(np.ascontiguousarray(w if rows is None else w[rows]))
            L.b = None if b is None else np.ascontiguousarray(b if rows is None else b[rows])
            lins.append(L)
            return L
        # ---- RADIO and the neck
        self.vw = cfg.vit_width
        self.vhd = self.vw // VIT_HEADS
        nblk = sum(1 for n in W.names() if n.startswith(VM + "blocks.") and n.endswith(".norm1.weight"))
        self.patch_lin = lin(VM + "patch_generator.embedder.weight", bias=False)
        self.pos = f(VM + "patch_generator.pos_embed").reshape(POS_SIDE, POS_SIDE, self.vw)
        self.cls = f(VM + "patch_generator.cls_token.token")
        self.blk = []
        for i in range(nblk):
            b = "%sblocks.%d." % (VM, i)
            g = lambda n: f(b + n)  # noqa: E731
            w = self.vw
            self.blk.append(dict(
                n1=(g("norm1.weight"), g("norm1.bias")), n2=(g("norm2.weight"), g("norm2.bias")),
                q=lin(b + "attn.qkv.weight", rows=slice(0, w)), k=lin(b + "attn.qkv.weight", rows=slice(w, 2 * w)),
                v=lin(b + "attn.qkv.weight", rows=slice(2 * w, 3 * w)), o=lin(b + "attn.proj.weight"),
                fc1=lin(b + "mlp.fc1.weight"), fc2=lin(b + "mlp.fc2.weight")))
        self.vit_scale = np.full(self.vw, self.vhd ** -0.5, np.float32)
        e = "encoder."
        self.conv1 = lin(e + "conv1.weight")
        self.conv2 = lin(e + "conv2.weight", bias=False)
        self.ln1 = (f(e + "layer_norm1.weight"), f(e + "layer_norm1.bias"))
        self.ln2 = (f(e + "layer_norm2.weight"), f(e + "layer_norm2.bias"))
        self.ln3 = (f(e + "layer_norm3.weight"), f(e + "layer_norm3.bias"))
        self.sum_proj = lin(e + "sum_proj.weight")
        # ---- the decoder
        d = "decoder."
        self.width, self.hd = cfg.width, cfg.width // cfg.heads
        self.kvw = cfg.kv_heads * self.hd
        self.embed = f(d + "embed_tokens.weight")                      # the rows of the tokens
        self.head = lin(d + "embed_tokens.weight", bias=False)         # the logits (tied)
        self.ln_emb = (f(d + "layernorm_embedding.weight"), f(d + "layernorm_embedding.bias"))
        self.ln_end = (f(d + "layer_norm.weight"), f(d + "layer_norm.bias"))
        self.layers = []
        self.qscale = np.full(cfg.width, self.hd ** -0.5, np.float32)      # 1/8: exact
        for i in range(cfg.layers):
            b = "%slayers.%d." % (d, i)
            g = lambda n: f(b + n)  # noqa: E731
            self.layers.append(dict(
                ln_a=(g("self_attn_layer_norm.weight"), g("self_attn_layer_norm.bias")),
                q=lin(b + "self_attn.q_proj.weight"), k=lin(b + "self_attn.k_proj.weight"),
                v=lin(b + "self_attn.v_proj.weight"), o=lin(b + "self_attn.out_proj.weight"),
                ln_f=(g("final_layer_norm.weight"), g("final_layer_norm.bias")),
                fc1=lin(b + "fc1.weight"), fc2=lin(b + "fc2.weight")))
        dpath = os.path.join(path, "dflash2")
        self.dflash = (DFLASH == "1" or DFLASH == "auto" and not self.gpu) and drafter and \
            os.path.exists(os.path.join(dpath, "config.json"))
        self.mtp = self.gpu and MTP and cfg.num_extra_heads > 0 and not self.dflash
        if self.mtp:
            self.mproj = lin(d + "extra_proj.0.weight")
            self.mhead = lin(d + "extra_heads.0.weight")
        W.release()
        if self.gpu:
            # (the download's float32 matrices in F20 as the converter makes them) F20 rows read by
            # KQ_LINEAR: k_kq_f20_gemv for a step's few rows, k_enc_sgemm's F20 tiles for more: float32
            # sums of the float32 x
            kept = 0
            for L in lins:
                if to_f20(L, True):
                    L.kq, L.f20 = L.f20, None
                else:
                    kept += 1
            assert not kept, "%d matrices refused F20" % kept
        self.df = None
        if self.dflash:
            from reclank.models.nemoparse.dflash2 import ParseDrafter
            self.df = ParseDrafter(self, dpath, block=DRAFTS + 1)
        self.caches = None
        self.positions = {}
        self.vit_runner = _Runner(self._build_vit, keep=4, keep_gpu=2)
        self.dec_runner = _Runner(self._build_dec, keep=16, keep_gpu=8)
        self.vit_runner.label, self.dec_runner.label = "nemoparse/encoder", "nemoparse/decoder"

    def runners(self):
        """The GPU runners (part: _Runner) of the GPU sharing (server/gpushare.py)."""
        return {"encoder": self.vit_runner, "decoder": self.dec_runner} if self.gpu else {}

    def _team(self, p, gpu):
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    # ---- the encoder

    def _build_vit(self, key, gpu):
        _, R, C = key
        n, w, H = PREFIX + R * C, self.vw, VIT_HEADS
        M = R * C // 4
        kp = _shape(self.patch_lin)[1]
        inter = self.blk[0]["fc1"].b.size
        c = _EncCompiler(1e-6, pack=not gpu, rows=n, kmax=max(kp, w, inter))
        c.attn_f32 = True
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        dw = self.width
        gidx = (PREFIX + np.arange(R * C, dtype=np.int32)).reshape(M, 4)
        c.env.update(p=z(n, kp), x0=z(n, w), x=z(n, w), h=z(n, w), q=z(n, w), k=z(n, w), v=z(n, w), o=z(n, w),
                     t=z(n, w), g=z(n, inter), lo=np.zeros(n, np.int32), hi=np.full(n, n, np.int32),
                     f1=z(n, dw), f1n=z(n, dw), gat=z(M, 4 * dw), f2=z(M, dw), feat=z(M, dw),
                     gidx=gidx, sidx=np.array([[0, 1]], np.int32), sg=z(1, 2 * w), s1=z(1, dw), summ=z(1, dw))
        forms = [("set", "x", ("enc_linear", self.patch_lin, "p")), ("enc_add", "x", "x0")]
        for b in self.blk:
            forms += [
                ("set", "h", ("enc_lnorm", "x", b["n1"][0], b["n1"][1], w)),
                ("set", "q", ("enc_linear", b["q"], "h")),
                ("set", "k", ("enc_linear", b["k"], "h")),
                ("set", "v", ("enc_linear", b["v"], "h")),
                ("set", "q", ("enc_mul_vec", "q", self.vit_scale)),
                ("set", "o", ("enc_attn", "q", "k", "v", H, self.vhd) if ATTN_F16 and gpu else
                 ("enc_attn_range_x2", "q", "k", "v", "lo", "hi", H, self.vhd)),
                ("set", "t", ("enc_linear", b["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", b["n2"][0], b["n2"][1], w)),
                ("set", "g", ("enc_linear", b["fc1"], "h")),
                ("set", "g", ("enc_gelu", "g", True)),
                ("set", "t", ("enc_linear", b["fc2"], "g")),
                ("enc_add", "x", "t"),
            ]
        forms += [
            ("set", "f1", ("enc_linear", self.conv1, "x")),
            ("set", "f1n", ("enc_lnorm", "f1", self.ln1[0], self.ln1[1], dw)),
            ("set", "gat", ("enc_gather", "f1n", "gidx", dw)),
            ("set", "f2", ("enc_linear", self.conv2, "gat")),
            ("set", "feat", ("enc_lnorm", "f2", self.ln2[0], self.ln2[1], dw)),
            ("set", "sg", ("enc_gather", "x", "sidx", w)),
            ("set", "s1", ("enc_linear", self.sum_proj, "sg")),
            ("set", "summ", ("enc_lnorm", "s1", self.ln3[0], self.ln3[1], dw)),
        ]
        c.compile(("seq", *forms))
        return self._team(c.p.finish(), gpu)

    def encode_patches(self, patches, hw, content):
        """The visual rows [(H / 16) (W / 64) + 1, 1024] of a page's patches."""
        R, C = hw[0] // PATCH, hw[1] // PATCH
        n = PREFIX + R * C
        p = np.zeros((n, patches.shape[1]), np.float32)
        p[PREFIX:] = patches
        x0 = np.empty((n, self.vw), np.float32)
        x0[:PREFIX] = self.cls
        key = (tuple(hw), tuple(content))
        pos = self.positions.pop(key, None)
        if pos is None:
            pos = position_rows(self.pos, hw, content)
        self.positions[key] = pos               # (the last few page sizes: a document's pages share one)
        while len(self.positions) > 4:
            self.positions.pop(next(iter(self.positions)))
        x0[PREFIX:] = pos
        prog, g = self.vit_runner.get(("vit", R, C), self.gpu)
        for name, a in (("p", p), ("x0", x0)):
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
            g.download("feat")
            g.download("summ")
        else:
            prog.run()
        return np.concatenate([prog.names["feat"], prog.names["summ"]])

    def encode(self, image):
        """(the visual rows, hw, content) of a page (an image file, bytes or a PIL image)."""
        patches, hw, content = page_pixels(image, self.cfg)
        return self.encode_patches(patches, hw, content), hw, content

    # ---- the decoder

    def cache(self):
        if self.caches is None:
            n = self.cfg.max_seq
            self.caches = {}
            for i in range(self.cfg.layers):
                self.caches["k%d" % i] = np.zeros((n, self.kvw), np.float32)
                self.caches["v%d" % i] = np.zeros((n, self.kvw), np.float32)
        return self.caches

    def _layers(self, c, R, attn):
        """The forms of layernorm_embedding of "e" (R rows of embeddings) into "x" and of the decoder's
        layers over "x" ("h", "t", "u", "q", "k", "v", "o" of the env); attn(i) the attention's form of
        layer i (into "o"). With the drafter, the residual stream after its tap layers into "tp" [R taps,
        width] (row r tap j at r taps + j: the [R, taps x width] rows of the drafter's fc)."""
        w = self.width
        taps = self.df.taps if self.df is not None else []
        if taps:
            c.env["tp"] = np.zeros((R * len(taps), w), np.float32)
            for j in range(len(taps)):
                c.env["tpi%d" % j] = (len(taps) * np.arange(R) + j).astype(np.int32)
        forms = [("set", "x", ("enc_lnorm", "e", self.ln_emb[0], self.ln_emb[1], w))]
        for i, L in enumerate(self.layers):
            forms += [
                ("set", "h", ("enc_lnorm", "x", L["ln_a"][0], L["ln_a"][1], w)),
                ("set", "q", ("enc_linear", L["q"], "h")),
                ("set", "q", ("enc_mul_vec", "q", self.qscale)),
                ("set", "k", ("enc_linear", L["k"], "h")),
                ("set", "v", ("enc_linear", L["v"], "h")),
                ("set", "k%d" % i, ("enc_scatter", "k", "kvi")),
                ("set", "v%d" % i, ("enc_scatter", "v", "kvi")),
                ("set", "o", attn(i)),
                ("set", "t", ("enc_linear", L["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", L["ln_f"][0], L["ln_f"][1], w)),
                ("set", "u", ("enc_linear", L["fc1"], "h")),
                ("set", "u", ("enc_gelu", "u", True)),
                ("set", "t", ("enc_linear", L["fc2"], "u")),
                ("enc_add", "x", "t"),
            ]
            if i in taps:
                forms.append(("set", "tp", ("enc_scatter", "x", "tpi%d" % taps.index(i))))
        return forms

    def _draft(self, c, R, h):
        """The emits of the MTP head after "logits" of R rows of final hidden state h: "am", "draft"."""
        w, V = self.width, self.cfg.vocab
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(am=np.zeros((R, 1), np.int32), draft=np.zeros((R, 1), np.int32), mg=z(R, w), me=z(R, w),
                     mt=z(R, w), dh=z(R, w), dlog=z(R, V))
        if "sc" not in c.env:
            c.env["sc"] = np.full(w, math.sqrt(w), np.float32)
        lg, am, dlog, dr = (c.env[n] for n in ("logits", "am", "dlog", "draft"))
        for j in range(R):
            c.p.emit(P.ARGMAX, lg[j], V, am[j])
        c.compile(("seq", ("set", "mg", ("enc_gather", self.embed, "am", w)), ("set", "me", ("enc_mul_vec", "mg", "sc")),
                   ("set", "mt", ("enc_linear", self.mproj, "me")), ("enc_add", "mt", h),
                   ("set", "dh", ("enc_linear", self.mhead, "mt")), ("set", "dlog", ("enc_linear", self.head, "dh"))))
        for j in range(R):
            c.p.emit(P.ARGMAX, dlog[j], V, dr[j])

    def _build_dec(self, key, gpu):
        if key[0] in ("inj", "drf"):
            return self.df.build(key, gpu)
        kind, R, mtp = key[0], key[1], key[-1]
        cfg, w = self.cfg, self.width
        inter = self.layers[0]["fc1"].b.size
        H, KV, hd = cfg.heads, cfg.kv_heads, self.hd
        c = _EncCompiler(1e-5, pack=not gpu, rows=R, kmax=inter)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        zi = lambda *shape: np.zeros(shape, np.int32)  # noqa: E731
        c.env.update(self.cache())
        c.env.update(e=z(R, w), x=z(R, w), h=z(R, w), t=z(R, w), u=z(R, inter), q=z(R, w), k=z(R, self.kvw),
                     v=z(R, self.kvw), o=z(R, w), kvi=zi(R), lo=zi(R), hi=zi(R))
        if kind == "pre":
            # the causal attention of rows 0 .. R - 1 over the caches' rows (the same rows)
            c.env.update(kvi=np.arange(R, dtype=np.int32), hi=np.arange(1, R + 1, dtype=np.int32),
                         last=np.array([[R - 1]], np.int32), hl=z(1, w), hn=z(1, w), logits=z(1, cfg.vocab))
            forms = self._layers(c, R, lambda i: ("enc_attn_range", "q", "k%d" % i, "v%d" % i, "lo", "hi",
                                                  H, KV, hd))
            forms += [("set", "hl", ("enc_gather", "x", "last", w)),
                      ("set", "hn", ("enc_lnorm", "hl", self.ln_end[0], self.ln_end[1], w)),
                      ("set", "logits", ("enc_linear", self.head, "hn"))]
            h = "hn"
        else:
            c.lnorm_rows = True             # few rows: a block a row on the GPU
            c.env.update(tok=zi(R, 1), g=z(R, w), sc=np.full(w, math.sqrt(w), np.float32), logits=z(R, cfg.vocab))
            forms = [("set", "g", ("enc_gather", self.embed, "tok", w)), ("set", "e", ("enc_mul_vec", "g", "sc"))]
            # (in C the scores' sums in SIMD lanes: the sums in order took 16 ms a layer for one row at
            # 1600 keys on 2 threads)
            forms += self._layers(c, R, lambda i: ("enc_attn_kv", "q", "k%d" % i, "v%d" % i, "lo", "hi", H, hd,
                                                   key[2], KV, 0 if gpu else 1))
            forms += [("set", "h", ("enc_lnorm", "x", self.ln_end[0], self.ln_end[1], w)),
                      ("set", "logits", ("enc_linear", self.head, "h"))]
            h = "h"
        c.compile(("seq", *forms))
        if mtp:
            self._draft(c, c.env[h].shape[0], h)
        return self._team(c.p.finish(), gpu)

    def _run(self, key, inputs, outputs):
        prog, g = self.dec_runner.get(key, self.gpu)
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
            for name in outputs:
                g.download(name)
        else:
            prog.run()
        return [prog.names[name].copy() for name in outputs]

    def text_rows(self, ids):
        return self.embed[np.asarray(ids, np.int64)] * np.float32(math.sqrt(self.width))

    def prefill(self, vis, prompt_ids, mtp=False):
        """The logits [1, vocab] after the visual rows and the prompt (the caches' rows 0 .. T - 1); with
        mtp also "am" and "draft" [1, 1]."""
        x = np.concatenate([vis, self.text_rows(prompt_ids)]).astype(np.float32)
        return self._run(("pre", x.shape[0], mtp), {"e": x}, ["logits", "am", "draft"] if mtp else
                         ["logits", "tp"] if self.df is not None else ["logits"])

    def step(self, tokens, pos, mtp=False):
        """The logits [r, vocab] of r tokens at positions pos .. pos + r - 1 (each over the cache's rows
        up to its own); with mtp also "am" and "draft" [r, 1]."""
        r = len(tokens)
        p = np.arange(pos, pos + r, dtype=np.int32)
        if p[-1] >= self.cfg.max_seq:
            raise ValueError("past the context of %d rows" % self.cfg.max_seq)
        span = min(self.cfg.max_seq, -(-(pos + r) // SPAN) * SPAN)
        return self._run(("step", r, span, mtp), {"tok": np.asarray(tokens, np.int32).reshape(r, 1), "kvi": p,
                                                  "lo": np.zeros(r, np.int32), "hi": p + 1},
                         ["logits", "am", "draft"] if mtp else ["logits", "tp"] if self.df is not None else ["logits"])

    def generate(self, vis, prompt_ids, max_new=MAX_NEW, penalty=REP_PENALTY, forced=None, on_token=None,
                 stats=None, sampler=None):
        """(new ids, logits of each step if forced else None): greedy with model.Rules (sampler: a
        model.Sampler instead, no drafts). forced: the ids fed instead of the chosen ones (teacher
        forcing, no drafts; the logits kept). on_token(id): each new token (streaming). stats (a
        dict): "steps" and "drafted" (the drafts taken) added, and "first_s" (the seconds to the first
        token: the prefill, and the drafter's context)."""
        T = vis.shape[0] + len(prompt_ids)
        max_new = max(1, min(int(max_new), self.cfg.max_seq - T))
        rules = Rules(prompt_ids, max_new, self.cfg.eos, penalty)
        mtp = self.mtp and forced is None and sampler is None
        out, logs = [], []
        steps = drafted = 0

        def emit(t):
            """Take token t; True when the output ends with it."""
            out.append(t)
            rules.add(t)
            if on_token is not None:
                on_token(t)
            return forced is None and (t == self.cfg.eos or len(out) >= max_new)

        if forced is not None:
            lg = self.prefill(vis, prompt_ids)[0][0]
            for t in forced:
                logs.append(lg)
                emit(int(t))
                lg = self.step([int(t)], T + len(out) - 1)[0][0]
            logs.append(lg)
            return out, logs
        t0 = time.time()

        def first():
            if stats is not None:
                stats["first_s"] = stats.get("first_s", 0.0) + time.time() - t0
        if self.df is not None and sampler is None:
            steps, drafted = self._generate_dflash(vis, prompt_ids, rules, emit, max_new, first)
            if stats is not None:
                stats["steps"] = stats.get("steps", 0) + steps
                stats["drafted"] = stats.get("drafted", 0) + drafted
            return out, None
        r = self.prefill(vis, prompt_ids, mtp)
        first()
        pos, t = T, rules.choose(r[0][0], 0, sampler)
        d = int(r[2][0, 0]) if mtp and t == int(r[1][0, 0]) else None
        done = emit(t)
        while not done:
            steps += 1
            if d is None or len(out) + 1 >= max_new:
                r = self.step([t], pos, mtp)
                pos += 1
                t = rules.choose(r[0][0], len(out), sampler)
                d = int(r[2][0, 0]) if mtp and t == int(r[1][0, 0]) else None
                done = emit(t)
                continue
            r = self.step([t, d], pos, True)
            y = rules.choose(r[0][0], len(out))
            done = emit(y)
            if y == d and not done:
                # the draft taken: the second row's logits follow it
                drafted += 1
                pos += 2
                t = rules.choose(r[0][1], len(out))
                d = int(r[2][1, 0]) if t == int(r[1][1, 0]) else None
                done = emit(t)
            else:
                pos += 1        # (the draft's row of the caches: the next step writes it again)
                t = y
                d = int(r[2][0, 0]) if y == int(r[1][0, 0]) else None
        if stats is not None:
            stats["steps"] = stats.get("steps", 0) + steps
            stats["drafted"] = stats.get("drafted", 0) + drafted
        return out, None

    def _generate_dflash(self, vis, prompt_ids, rules, emit, max_new, first):
        """Greedy decoding with the DFlash2 drafter: a round drafts after the last token (ParseDrafter.propose),
        the target runs the token and its drafts as one step, the drafts its own choices confirm are taken
        (and the choice after them), and their taps become the drafter's context. emit(token): the caller's
        (True when the output ends). (steps, drafted)."""
        df = self.df
        T = vis.shape[0] + len(prompt_ids)
        r = self.prefill(vis, prompt_ids)
        df.inject_rows(r[1].reshape(T, -1), np.arange(T))
        first()
        pos, t = T, rules.choose(r[0][0], 0)
        steps = drafted = 0
        done = emit(t)
        while not done:
            steps += 1
            n = min(df.block - 1, max_new - (pos - T) - 2, self.cfg.max_seq - pos - 1)
            d = df.propose(t, pos, n) if n > 0 else []
            r = self.step([t] + d, pos)
            lg, tp = r[0], r[1].reshape(len(d) + 1, -1)
            a = 0
            while True:
                y = rules.choose(lg[a], pos - T + a + 1)
                done = emit(y)
                if done or a == len(d) or y != d[a]:
                    break
                a += 1
            drafted += a
            df.inject_rows(tp[:a + 1], np.arange(pos, pos + a + 1))
            pos += a + 1
            t = y
        return steps, drafted

    def parse(self, image, prompt=PROMPT, max_new=MAX_NEW, penalty=REP_PENALTY, on_token=None, sampler=None):
        """{"ids" (the new tokens), "text" (the prompt and the output, the special tokens kept: the
        card's raw output), "output" (the new tokens' text), "finish_reason", "tokens",
        "prefix_rows", "encode_s", "decode_s", "image_size" (width, height)} of a page."""
        from reclank.models.nemotron.radio import load_rgb
        ids = self.tok.encode(prompt)
        rgb = load_rgb(image)
        with RESIDENCY.use(*self.runners().values()):
            t0 = time.time()
            vis = self.encode(rgb)[0]
            t1 = time.time()
            st = {}
            out, _ = self.generate(vis, ids, max_new, penalty, on_token=on_token, stats=st, sampler=sampler)
            t2 = time.time()
        return {"ids": out, "steps": st.get("steps", 0) + 1, "drafted": st.get("drafted", 0), "text": self.tok.text(ids + out), "output": self.tok.text(out),
                "finish_reason": "stop" if out and out[-1] == self.cfg.eos else "length",
                "tokens": len(out), "prefix_rows": int(vis.shape[0]), "encode_s": t1 - t0, "decode_s": t2 - t1,
                "image_size": (int(rgb.shape[1]), int(rgb.shape[0]))}

    def structured(self, result):
        """model.blocks of a parse() result (the boxes in the page's pixels too)."""
        return blocks(result["text"], result["image_size"])
