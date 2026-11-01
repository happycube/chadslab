"""MuQ-MuLan's audio tower (docs/MUSIC_EMBED_PLAN.md): MuQ (OpenMuQ/MuQ-large-
msd-iter) on clips of 10 s of 24 kHz mono, then MuLan's projection.

Ported from MuQ (tencent-ailab/MuQ, src/muq/muq and src/muq/muq_mulan; MIT
License; Copyright (c) Tencent.) and its conformer, transformers
(src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py;
Apache License 2.0; Copyright 2022 The Fairseq Authors and the HuggingFace Inc.
team). See THIRD_PARTY_NOTICES.md.

    the mel of torchaudio (n_fft 2048, hop 240, 128 HTK mels, power 2,
        centered with reflect padding), 10 log10 (floor 1e-10), the last
        frame dropped (1000 frames), (x - 6.768) / 18.418: on the host
    Conv2dSubsampling: two Res2d blocks of 512 channels, stride (2, 2) each:
        relu(bn(conv3x3(relu(bn(conv3x3 / 2(x))))) + bn(conv3x3 / 2(x))); the
        BatchNorms are folded into the convs. Then (channel, band) of each of
        the 250 frames, a linear to 1024
    the Wav2Vec2-Conformer of transformers, 12 layers of width 1024 (16 heads
        of 64, FFN 4096 with swish): FFN1 times 0.5, + x; the attention (LN,
        RoPE on the input of q and k, q k v with bias, scale 1/8, out, + x);
        the conv module (LN, pointwise to 2048, GLU, depthwise k31 "same",
        BatchNorm, swish, pointwise, + x); FFN2 times 0.5, + x; LN. Then the
        encoder's LN
    MuLan: proj 1024 -> 768 (its audio transformer has no layers), the mean
        over the 250 rows, audio_to_latents 768 -> 512, the L2 norm: on the
        host (the mean before proj: proj is linear)

The convs are products over the im2col (each output position's 9 input rows
of C channels), on a grid of rows (band, frame) of C values: ENC_GATHER_MM on
the GPU (the product reads the rows by index), ENC_GATHER and ENC_LINEAR on
the CPU. The depthwise conv of k31 "same" is the causal ENC_DWCONV of the
frames with 15 zero rows after them, read 15 rows later. A program takes
BATCH clips (program()), each clip with the bits of a program of it alone:
on the CPU the float32 matrices as trained, on the GPU their BF12 rows
(text.gpu_bf12).
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner, _shape
from reclank.formats.safetensors import SafeTensors
from reclank.models.music.frontend import muq_mel as mel  # noqa: F401
from reclank.models.music.text import gpu_bf12
from reclank.runtime.threads import team_threads

RATE, CLIP = 24000, 10
N_FFT, HOP, N_MELS = 2048, 240, 128
# the clips of a program: 5 x 256 rows are 10 tiles of 128 rows, so the products of
# 1024 rows of the conformer fill the 82 SMs of a 3090 (80 blocks; 6 clips: 96, two waves)
BATCH = 5
ROWS, CLIP_ROWS = 250, 256   # the conformer's rows of a clip, and their room in a program


def _im2col(F, T, stride):
    """The im2col index (out positions, 9) of a conv 3x3, padding 1, of a
    grid of F x T rows (band, frame), stride (stride, stride)."""
    Fo, To = (F - 1) // stride + 1, (T - 1) // stride + 1
    fo, to = np.meshgrid(np.arange(Fo), np.arange(To), indexing="ij")
    idx = np.empty((Fo, To, 9), np.int32)
    for ky in range(3):
        for kx in range(3):
            fi, ti = fo * stride - 1 + ky, to * stride - 1 + kx
            ok = (fi >= 0) & (fi < F) & (ti >= 0) & (ti < T)
            idx[:, :, ky * 3 + kx] = np.where(ok, fi * T + ti, -1)
    return idx.reshape(Fo * To, 9), Fo, To


class MuqAudio:
    """The audio tower of a directory of scripts/convert_music.py --muq.
    embed(clips) gives the vectors (clips, 512) of clips of 10 s at 24 kHz."""

    def __init__(self, path, gpu=False, cpu_threads=None, W=None):
        self.W = W or SafeTensors(os.path.join(path, "model.safetensors"))
        self.gpu = bool(gpu)
        self.cpu_threads = cpu_threads
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg["mulan"]["sr"] == RATE and cfg["mulan"]["clip_secs"] == CLIP
        assert cfg["audio_transformer"]["tf_depth"] == 0
        W = self.W
        p = "mulan.audio.model.model."
        f = lambda n: np.asarray(W.get(p + n), dtype=np.float32)  # noqa: E731
        self.eps = 1e-5

        def conv(name, bn, relu=False):
            """conv3x3 + its BatchNorm as a linear over the im2col: W (out, 9 in)."""
            w, b = f(name + ".weight"), f(name + ".bias")
            s = f(bn + ".weight") / np.sqrt(f(bn + ".running_var") + np.float32(1e-5))
            w = w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1) * s[:, None]
            lin = _Plain(np.ascontiguousarray(w, dtype=np.float32))
            lin.b = ((b - f(bn + ".running_mean")) * s + f(bn + ".bias")).astype(np.float32)
            if relu:
                lin.omin, lin.omax = 0.0, float("inf")
            return lin
        self.blocks = []
        for i in range(2):
            q = "conv.conv.%d." % i
            self.blocks.append(dict(c1=conv(q + "conv1", q + "bn1", relu=True), c2=conv(q + "conv2", q + "bn2"),
                                    c3=conv(q + "conv3", q + "bn3")))
        self.C = self.blocks[0]["c1"].w.shape[0]
        # the linear of (channel, band) -> 1024, its columns in the order (band, channel) of the gather
        w = f("conv.linear.weight")
        width = w.shape[0]
        bands = w.shape[1] // self.C
        self.bands = bands
        self.sub = _Plain(np.ascontiguousarray(w.reshape(width, self.C, bands).transpose(0, 2, 1)
                                               .reshape(width, -1)))
        self.sub.b = f("conv.linear.bias")
        self.width = width

        def lin(name, scale=None, bias=True):
            w = f(name + ".weight")
            w = w.reshape(w.shape[0], -1)
            b = f(name + ".bias") if bias else None
            if scale is not None:
                w, b = w * np.float32(scale), (b * np.float32(scale) if b is not None else None)
            out = _Plain(np.ascontiguousarray(w))
            out.b = b
            return out
        c = "conformer."
        self.heads = 16
        hd = width // self.heads
        self.inv = (1.0 / (10000.0 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))).astype(np.float32)
        self.layers = []
        i = 0
        while (p + c + "layers.%d.final_layer_norm.weight" % i) in W.header:
            q = c + "layers.%d." % i
            cm = q + "conv_module."
            s = f(cm + "batch_norm.weight") / np.sqrt(f(cm + "batch_norm.running_var") + np.float32(1e-5))
            dw = f(cm + "depthwise_conv.weight").reshape(width, -1) * s[:, None]
            self.layers.append(dict(
                ln_f1=(f(q + "ffn1_layer_norm.weight"), f(q + "ffn1_layer_norm.bias")),
                f1u=lin(q + "ffn1.intermediate_dense"), f1d=lin(q + "ffn1.output_dense"),
                ln_a=(f(q + "self_attn_layer_norm.weight"), f(q + "self_attn_layer_norm.bias")),
                aq=lin(q + "self_attn.linear_q", scale=hd ** -0.5), ak=lin(q + "self_attn.linear_k"),
                av=lin(q + "self_attn.linear_v"), ao=lin(q + "self_attn.linear_out"),
                ln_c=(f(cm + "layer_norm.weight"), f(cm + "layer_norm.bias")),
                pw1=lin(cm + "pointwise_conv1", bias=False), pw2=lin(cm + "pointwise_conv2", bias=False),
                dw=np.ascontiguousarray(dw, dtype=np.float32),
                dwb=(f(cm + "batch_norm.bias") - f(cm + "batch_norm.running_mean") * s).astype(np.float32),
                ln_f2=(f(q + "ffn2_layer_norm.weight"), f(q + "ffn2_layer_norm.bias")),
                f2u=lin(q + "ffn2.intermediate_dense"), f2d=lin(q + "ffn2.output_dense"),
                ln_o=(f(q + "final_layer_norm.weight"), f(q + "final_layer_norm.bias"))))
            i += 1
        self.ln_end = (f(c + "layer_norm.weight"), f(c + "layer_norm.bias"))
        g = lambda n: np.asarray(W.get(n), dtype=np.float32)  # noqa: E731
        self.proj = (g("mulan.audio.proj.weight").T.copy(), g("mulan.audio.proj.bias"))
        self.latent = (g("mulan.audio_to_latents.weight").T.copy(), g("mulan.audio_to_latents.bias"))
        if self.gpu:
            gpu_bf12([b[k] for b in self.blocks for k in ("c1", "c2", "c3")] + [self.sub] +
                     [L[k] for L in self.layers for k in ("f1u", "f1d", "aq", "ak", "av", "ao", "pw1", "pw2",
                                                          "f2u", "f2d")], "muq audio")
        self.runner = _Runner(lambda n, gpu: self._team(self.program(n, gpu), gpu), keep=3, keep_gpu=2)

    def _team(self, p, gpu):
        if not gpu:     # word 3 of the header: the team (cpu_threads, or the CPUs of the taskset)
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def program(self, n_clips=1, gpu=False):
        """The program of n_clips clips: input "m" (each clip's mel as rows
        (band, frame) of 1 value, one clip after another), output "hid"
        (n_clips 256, 1024): the conformer's last hidden state of each clip
        in its first 250 rows.

        The convs are products over the im2col of their input
        (enc_gather_linear: on the GPU without the gathered rows; on the CPU
        590 MB a clip for the first block). The first block runs a clip at a
        time in the same buffers into the clip's rows of b1;
        the second block (3 x 8000 rows a clip: 252 blocks of 128 x 128 on 82
        SMs leave a fourth wave of 6), the linear to 1024 and the conformer
        take the rows of all the clips at once. In the
        conformer a clip has 256 rows, its 250 and 6 of zeros, so that the
        blocks of 8 queries and the tiles of 32 keys of ENC_ATTN_RANGE fall on
        a clip's rows as in a program of that clip alone (a block over two
        clips sums in another order); a clip's rows attend their own 250, take
        the positions 0..249, and have the 15 zero rows of the depthwise conv
        after them. So a clip's rows come out with the bits of a program of
        that clip alone; 5 clips are 10 tiles of 128 rows."""
        C, w = self.C, self.width
        F, T = N_MELS, CLIP * RATE // HOP
        i1, F1, T1 = _im2col(F, T, 2)
        s1, _, _ = _im2col(F1, T1, 1)
        i2, F2, T2 = _im2col(F1, T1, 2)
        s2, _, _ = _im2col(F2, T2, 1)
        assert F2 == self.bands and T2 == ROWS
        n = n_clips
        S = CLIP_ROWS
        R = n * S                                    # the rows of the conformer
        P2 = F2 * T2                                 # the positions of a clip's b2
        clip = np.repeat(np.arange(n), S)
        row = np.tile(np.arange(S), n)
        real = row < T2
        # the rearrange (band, frame) -> rows of the frames, (band, channel), of each clip
        rr = np.where(real[:, None], clip[:, None] * P2 + np.arange(F2)[None, :] * T2 + row[:, None],
                      -1).astype(np.int32)
        H = self.heads
        hd = w // H
        K = self.layers[0]["dw"].shape[1]
        pad = (K - 1) // 2
        ext = np.full((n, T2 + pad), -1, np.int32)
        ext[:, :T2] = np.arange(R).reshape(n, S)[:, :T2]
        ext = ext.reshape(-1, 1)
        back = np.where(real, clip * (T2 + pad) + row + pad, -1).astype(np.int32)[:, None]
        inter = _shape(self.layers[0]["f1u"])[0]
        c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=inter)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        P1 = F1 * T1                                 # the positions of a clip's b1
        # block 2 of all the clips at once: its im2col of each clip's rows of b1
        i2n = np.concatenate([np.where(i2 >= 0, i2 + j * P1, -1) for j in range(n)]).astype(np.int32)
        s2n = np.concatenate([np.where(s2 >= 0, s2 + j * P2, -1) for j in range(n)]).astype(np.int32)
        m, b1 = z(n * F * T, 1), z(n * P1, C)
        c.env.update(m=m, col1=z(P1, 9), a1=z(P1, C), b1=b1, r1=z(P1, C),
                     a2=z(n * P2, C), b2=z(n * P2, C), r2=z(n * P2, C), x=z(R, w), h=z(R, w), hr=z(R, w), t=z(R, w), u=z(R, inter),
                     q=z(R, w), k=z(R, w), v=z(R, w), o=z(R, w), g2=z(R, 2 * w), xe=z(n * (T2 + pad), w),
                     ye=z(n * (T2 + pad), w), hid=z(R, w), pos=row.astype(np.int32),
                     lo=(clip * S).astype(np.int32), hi=(clip * S + T2).astype(np.int32))
        b0, b1w = self.blocks
        forms = []
        for j in range(n):
            # block 1 of a clip (its im2col is 590 MB) into the clip's rows of b1 (views: the
            # records name their rows)
            mj, bj = "m%d" % j, "b1_%d" % j
            c.env[mj], c.env[bj] = m[j * F * T:(j + 1) * F * T], b1[j * P1:(j + 1) * P1]
            forms += [
                # conv1 (1 channel) and the residual conv3 share the im2col
                ("set", "col1", ("enc_gather", mj, i1, 1)),
                ("set", "a1", ("enc_linear", b0["c1"], "col1")),
                ("set", "r1", ("enc_linear", b0["c3"], "col1")),
                ("set", bj, ("enc_gather_linear", b0["c2"], "a1", s1, C)),
                ("enc_add", bj, "r1"),
                ("set", bj, ("enc_clamp", bj, 0.0, float("inf"))),
            ]
        forms += [
            ("set", "a2", ("enc_gather_linear", b1w["c1"], "b1", i2n, C)),
            ("set", "r2", ("enc_gather_linear", b1w["c3"], "b1", i2n, C)),
            ("set", "b2", ("enc_gather_linear", b1w["c2"], "a2", s2n, C)),
            ("enc_add", "b2", "r2"),
            ("set", "b2", ("enc_clamp", "b2", 0.0, float("inf"))),
        ]
        forms += [
            ("set", "x", ("enc_gather_linear", self.sub, "b2", rr, C)),
        ]
        for L in self.layers:
            forms += [
                ("set", "h", ("enc_lnorm", "x", L["ln_f1"][0], L["ln_f1"][1], w)),
                ("set", "u", ("enc_linear", L["f1u"], "h")),
                ("set", "u", ("enc_silu", "u")),
                ("set", "t", ("enc_linear", L["f1d"], "u")),
                ("enc_add", "x", "t", 0.5),
                ("set", "h", ("enc_lnorm", "x", L["ln_a"][0], L["ln_a"][1], w)),
                ("set", "hr", ("copy", "h")),
                ("enc_rope1d", "hr", "pos", self.inv, H, hd),
                ("set", "q", ("enc_linear", L["aq"], "hr")),
                ("set", "k", ("enc_linear", L["ak"], "hr")),
                ("set", "v", ("enc_linear", L["av"], "h")),
                ("set", "o", ("enc_attn_range", "q", "k", "v", "lo", "hi", H, H, hd)),
                ("set", "t", ("enc_linear", L["ao"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", L["ln_c"][0], L["ln_c"][1], w)),
                ("set", "g2", ("enc_linear", L["pw1"], "h")),
                ("set", "h", ("enc_glu", "g2")),
                ("set", "xe", ("enc_gather", "h", ext, w)),
                ("set", "ye", ("enc_dwconv", "xe", L["dw"])),
                ("set", "h", ("enc_gather", "ye", back, w)),
                ("enc_bias", "h", L["dwb"]),
                ("set", "h", ("enc_silu", "h")),
                ("set", "t", ("enc_linear", L["pw2"], "h")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", L["ln_f2"][0], L["ln_f2"][1], w)),
                ("set", "u", ("enc_linear", L["f2u"], "h")),
                ("set", "u", ("enc_silu", "u")),
                ("set", "t", ("enc_linear", L["f2d"], "u")),
                ("enc_add", "x", "t", 0.5),
                ("set", "x", ("enc_lnorm", "x", L["ln_o"][0], L["ln_o"][1], w)),
            ]
        forms.append(("set", "hid", ("enc_lnorm", "x", self.ln_end[0], self.ln_end[1], w)))
        c.compile(("seq", *forms))
        return c.p.finish()

    @staticmethod
    def prep(samples):
        """The host's part of one clip (the mel as the program's input): a
        thread can make it while the program of another clip runs."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(x) != CLIP * RATE:
            raise ValueError("a MuQ clip is %d samples (10 s at 24 kHz), not %d" % (CLIP * RATE, len(x)))
        return mel(x).reshape(-1, 1)

    def run_many(self, preps, batch=BATCH):
        """The conformer's last hidden states (250, 1024) of clips' preps, in
        programs of batch clips (the last one of the rest)."""
        hs = []
        for i in range(0, len(preps), batch):
            part = preps[i:i + batch]
            h = self.runner.run(len(part), self.gpu, {"m": np.concatenate(part)}, "hid")
            hs += list(h.reshape(len(part), CLIP_ROWS, -1)[:, :ROWS])
        return hs

    def run(self, prep):
        """The conformer's last hidden state (250, 1024) of a clip's prep."""
        return self.run_many([prep])[0]

    def vectors(self, hs):
        """The vectors (clips, 512) of the hidden states of clips."""
        means = np.stack([h.mean(0) for h in hs])
        y = (means @ self.proj[0] + self.proj[1]) @ self.latent[0] + self.latent[1]
        return y / np.linalg.norm(y, axis=1, keepdims=True)

    def embed(self, clips, hidden=False):
        """The vectors (len(clips), 512) of clips of 10 s at 24 kHz."""
        hs = self.run_many([self.prep(c) for c in clips])
        means = np.stack([h.mean(0) for h in hs])
        y = (means @ self.proj[0] + self.proj[1]) @ self.latent[0] + self.latent[1]
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return (y, hs) if hidden else y
