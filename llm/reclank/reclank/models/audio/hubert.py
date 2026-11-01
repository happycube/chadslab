"""HuBERT (transformers' HubertModel: feat_extract_norm "group", post-LN
layers) as a program on 16 kHz mono: the Higgs codec's semantic model
(models/higgs/codec.py), whose input is the mean of all its hidden states.

Ported from transformers (src/transformers/models/hubert/modeling_hubert.py;
Apache License 2.0; Copyright 2021 The Fairseq Authors and the HuggingFace Inc.
team). See THIRD_PARTY_NOTICES.md.

    the feature extractor: conv 0 (k 10, stride 5, 1 -> 512, no bias),
        GroupNorm of 512 groups (each channel over all the rows: ENC_GNORM),
        GELU erf; convs 1 to 6 (k 3 3 3 3 2 2, stride 2, no bias), GELU erf
    the projection: LayerNorm (512), linear (512 -> 768)
    the conv position embedding: a conv of k 128 in 16 groups of 48
        channels, padding 64, the last row dropped, weight-normed (W = v g /
        |v|, the norm over all but the taps' axis: folded on the host in
        float32), GELU erf; + x, LayerNorm: hidden state 0
    12 post-LN layers (12 heads of 64): q (scale 1/8 folded: exact) k v
        with bias, the attention of all the rows, the output, + x, LN; fc1,
        GELU erf, fc2, + x, LN: hidden states 1 to 12

A conv: models/audio/conv.py (the gather of its taps, ENC_LINEAR). A group
of the position conv: the gather of its 48 channels' rows (x as rows of
48 values), its linear, ENC_SCATTER of its rows into its columns of the
output. The attention: ENC_ATTN_RANGE of all the rows (ENC_ATTN_X2 on the
GPU). The matrices: the checkpoint's bfloat16 rows on the GPU (KQ_BF16), as
float32 on the CPU; the position conv's float32 (folded) on both.
"""
from __future__ import annotations

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Runner
from reclank.models.audio.conv import Convs, conv_rows
from reclank.models.audio.dac import aligned, conv_lin, mat
from reclank.runtime.threads import team_threads


def _scaled(raw, s):
    """bfloat16 rows (uint16) times a power of two s (exact), or float32."""
    if raw.dtype != np.uint16:
        return (raw.astype(np.float32) * np.float32(s)).astype(np.float32)
    w = ((raw.astype(np.uint32) << 16).view(np.float32) * np.float32(s)).view(np.uint32)
    assert not np.any(w & 0xFFFF)
    return (w >> 16).astype(np.uint16)


class Hubert:
    """The HuBERT of the tensors under prefix in W (Shards or SafeTensors:
    get, get_bf16) and its config (dict). program(n, tail) gives the
    program of n samples; run(x) the mean of the hidden states of 16 kHz
    samples x."""

    def __init__(self, W, prefix, cfg, gpu=False, cpu_threads=None, tail=None):
        assert cfg["feat_extract_norm"] == "group" and not cfg["do_stable_layer_norm"]
        assert cfg["feat_proj_layer_norm"] and not cfg["conv_bias"] and not cfg.get("conv_pos_batch_norm")
        assert cfg["hidden_act"] == "gelu" and cfg["feat_extract_activation"] == "gelu"
        self.gpu, self.cpu_threads, self.tail = bool(gpu), cpu_threads, tail
        f = lambda n: np.ascontiguousarray(W.get(prefix + n), dtype=np.float32)  # noqa: E731
        raw = lambda n: W.get_bf16(prefix + n)  # noqa: E731

        def lin(n, scale=None):
            w = raw(n + ".weight")
            m = mat(w if scale is None else _scaled(w, scale), gpu)
            m.b = f(n + ".bias") * np.float32(1.0 if scale is None else scale)
            return m
        self.width, self.H = cfg["hidden_size"], cfg["num_attention_heads"]
        self.eps = float(cfg["layer_norm_eps"])
        fe = "feature_extractor.conv_layers."
        self.convs = []
        for i, (k, s) in enumerate(zip(cfg["conv_kernel"], cfg["conv_stride"])):
            cl = conv_lin(aligned(raw(fe + "%d.conv.weight" % i)), None, gpu)
            cl.stride = s
            self.convs.append(cl)
        self.gn = (f(fe + "0.layer_norm.weight"), f(fe + "0.layer_norm.bias"))
        self.fp_ln = (f("feature_projection.layer_norm.weight"), f("feature_projection.layer_norm.bias"))
        self.fp = lin("feature_projection.projection")
        e = "encoder."
        pc = e + "pos_conv_embed.conv."
        g = f(pc + "parametrizations.weight.original0")             # (1, 1, k)
        v = f(pc + "parametrizations.weight.original1")             # (out, in / groups, k)
        nrm = np.sqrt((v.astype(np.float64) ** 2).sum(axis=(0, 1), keepdims=True)).astype(np.float32)
        wpos = (v * (g / nrm)).astype(np.float32)
        bpos = f(pc + "bias")
        self.groups = cfg["num_conv_pos_embedding_groups"]
        self.kpos = cfg["num_conv_pos_embeddings"]
        cg = self.width // self.groups
        self.pos = [conv_lin(wpos[j * cg:(j + 1) * cg], bpos[j * cg:(j + 1) * cg], gpu) for j in range(self.groups)]
        self.ln = (f(e + "layer_norm.weight"), f(e + "layer_norm.bias"))
        hd = self.width // self.H
        self.layers = []
        for i in range(cfg["num_hidden_layers"]):
            p = e + "layers.%d." % i
            a = p + "attention."
            self.layers.append(dict(
                q=lin(a + "q_proj", scale=hd ** -0.5), k=lin(a + "k_proj"), v=lin(a + "v_proj"),
                o=lin(a + "out_proj"), ln=(f(p + "layer_norm.weight"), f(p + "layer_norm.bias")),
                fc1=lin(p + "feed_forward.intermediate_dense"), fc2=lin(p + "feed_forward.output_dense"),
                lnf=(f(p + "final_layer_norm.weight"), f(p + "final_layer_norm.bias"))))
        self.inter = self.layers[0]["fc1"].shape[0]
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)

    def _build(self, n, gpu):
        p = self.program(n, gpu=gpu, tail=self.tail)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def frames(self, n):
        """The rows of n samples after the feature extractor."""
        for cl in self.convs:
            n = conv_rows(n, cl.k, cl.stride)
        return n

    def program(self, n, gpu=False, tail=None):
        """The program of n samples: input "wav" (n, 1); outputs "feat" (F,
        512) the feature extractor's rows, "h0" hidden state 0, "x" the last,
        "hid" (F, width) the mean of all of them. tail(cv, F): more forms
        (and their names in cv.env) after "hid" (cv: the Convs)."""
        w, H = self.width, self.H
        hd = w // H
        F0 = conv_rows(n, self.convs[0].k, self.convs[0].stride)
        F = self.frames(n)
        c = _EncCompiler(self.eps, pack=not gpu)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        ch = self.convs[0].shape[0]
        c.env.update(wav=z(n, 1), a=z(F0 * ch), b=z(F0 * ch), feat=z(F, ch), fn=z(F, ch), x=z(F, w), h=z(F, w),
                     t=z(F, w), q=z(F, w), k=z(F, w), v=z(F, w), o=z(F, w), u=z(F, self.inter), pc=z(F, w),
                     h0=z(F, w), hid=z(F, w), cg=z(F, w // self.groups),
                     lo=np.zeros(F, np.int32), hi=np.full(F, F, np.int32),
                     mean=np.full(w, 1.0 / (len(self.layers) + 1), np.float32))
        cv = Convs(c)
        forms = cv.forms
        # the feature extractor
        rows, src, cin, held = n, "wav", 1, None      # held: the buffer ("a" or "b") of src
        for i, cl in enumerate(self.convs):
            nxt = conv_rows(rows, cl.k, cl.stride)
            buf = "b" if held == "a" else "a"
            out = "feat" if i + 1 == len(self.convs) else cv.view(buf, nxt, ch)
            cv.conv(src, rows, cin, cl, out, stride=cl.stride, pad=0)
            if i == 0:
                buf = "b"
                g = cv.view(buf, nxt, ch)
                forms.append(("set", g, ("enc_gnorm", out, self.gn[0], self.gn[1], ch, 1e-5)))
                out = g
            forms.append(("set", out, ("enc_gelu", out, True)))
            rows, src, cin, held = nxt, out, ch, buf
        forms += [("set", "fn", ("enc_lnorm", "feat", self.fp_ln[0], self.fp_ln[1], ch)),
                  ("set", "x", ("enc_linear", self.fp, "fn"))]
        # the conv position embedding, a group at a time
        G, cg, K = self.groups, w // self.groups, self.kpos
        t = np.arange(F)[:, None] - K // 2 + np.arange(K)[None, :]
        xg = cv.view("x", F * G, cg)
        pg = cv.view("pc", F * G, cg)
        for j, lin in enumerate(self.pos):
            idx = np.where((t >= 0) & (t < F), t * G + j, -1).astype(np.int32)
            cv.gather_linear(xg, idx, cg, lin, "cg")
            forms.append(("set", pg, ("enc_scatter", "cg", (np.arange(F) * G + j).astype(np.int32))))
        forms += [("set", "pc", ("enc_gelu", "pc", True)),
                  ("enc_add", "x", "pc"),
                  ("set", "h0", ("enc_lnorm", "x", self.ln[0], self.ln[1], w)),
                  ("set", "x", ("enc_clamp", "h0", float("-inf"), float("inf"))),
                  ("set", "hid", ("enc_clamp", "h0", float("-inf"), float("inf")))]
        for L in self.layers:
            forms += [
                ("set", "q", ("enc_linear", L["q"], "x")),
                ("set", "k", ("enc_linear", L["k"], "x")),
                ("set", "v", ("enc_linear", L["v"], "x")),
                ("set", "o", ("enc_attn_range_x2", "q", "k", "v", "lo", "hi", H, hd)),
                ("set", "t", ("enc_linear", L["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", L["ln"][0], L["ln"][1], w)),
                ("set", "u", ("enc_linear", L["fc1"], "h")),
                ("set", "u", ("enc_gelu", "u", True)),
                ("set", "t", ("enc_linear", L["fc2"], "u")),
                ("enc_add", "h", "t"),
                ("set", "x", ("enc_lnorm", "h", L["lnf"][0], L["lnf"][1], w)),
                ("enc_add", "hid", "x"),
            ]
        forms.append(("set", "hid", ("enc_mul_vec", "hid", "mean")))
        if tail is not None:
            tail(cv, F)
        c.compile(("seq", *forms))
        return c.p.finish()

    def run(self, x, outputs=("hid",)):
        """The outputs (default: the mean of the hidden states) of 16 kHz
        samples x."""
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 1)
        prog, g = self.runner.get(len(x), self.gpu)
        prog.names["wav"][...] = x
        if g is not None:
            g.upload("wav")
            g.run()
            for name in outputs:
                g.download(name)
        else:
            prog.run()
        return [prog.names[name].copy() for name in outputs]
