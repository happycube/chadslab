"""CLAP's audio tower (docs/MUSIC_EMBED_PLAN.md): LAION-CLAP music_audioset,
HTSAT-base on clips of 10 s of 48 kHz mono, in the names of transformers'
ClapAudioModel (scripts/convert_music.py --clap).

Ported from transformers (src/transformers/models/clap/modeling_clap.py,
Copyright 2023 The LAION-AI Team and The HuggingFace Team;
models/clap/feature_extraction_clap.py, Copyright 2023 The HuggingFace Inc.
team; Apache License 2.0). See THIRD_PARTY_NOTICES.md.

On the host, as ClapFeatureExtractor (rand_trunc; a clip of 480000 samples
is neither cut nor padded) and ClapAudioEncoder before its patch embed:

    the STFT of 1024 (hop 480, periodic Hann, centered with reflect
        padding), power 2, the Slaney mel of 64 bands from 50 to 14000 Hz,
        10 log10 (floor 1e-10): 1001 frames by 64
    the BatchNorm of the 64 bands; the frames stretched to 1024 (bicubic,
        A = -0.75, align_corners: a fixed matrix); the 4 chunks of 256 frames
        stacked as a 256 x 256 image (reshape_mel2img); its patches of 4 x 4

The program (the records ENC_*), on the CPU (the float32 matrices as
trained) or the GPU (their BF12 rows, text.gpu_bf12):

    the patch linear (16 -> 128) and its LayerNorm: 64 x 64 tokens
    4 Swin stages (depths 2 2 12 2, widths 128 256 512 1024, heads of 32,
        window 8; the odd blocks shifted by 4; the last stage, 8 x 8, one
        window with no shift): LN, the rows in window order (ENC_GATHER of the
        cyclic shift and the partition), q k v with bias, the attention of
        each window with its relative position bias and the shift's mask
        (ENC_ATTN_BIAS), the output, the rows back, + x; LN, the MLP (GELU
        erf), + x. Between stages the patch merging: the 4 rows of each 2 x 2
        (ENC_GATHER), LN, a linear to twice the width
    the final LayerNorm (64 rows); on the host their mean (HTSAT's
        "embedding", not the tscam head), audio_projection (linear, ReLU,
        linear), the L2 norm
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.music.frontend import clap_mel as mel, clap_patches  # noqa: F401
from reclank.models.music.text import gpu_bf12
from reclank.runtime.threads import team_threads

RATE, CLIP = 48000, 10
N_FFT, HOP, N_MELS = 1024, 480, 64
SPEC = 256              # the image of HTSAT: 256 x 256
RATIO = 4               # spec_size // num_mel_bins: the chunks of frames


def _window_order(H, W, ws, shift):
    """The grid rows (y * W + x) of the rows in window order (window by
    window, row-major in each) after the cyclic shift by -shift, and the
    shift's mask type of each window with the masks (types, g, g)."""
    rows, regions = [], []
    def region(v, n):
        return 0 if v < n - ws else (1 if v < n - shift else 2)
    for wy in range(H // ws):
        for wx in range(W // ws):
            r, reg = [], []
            for iy in range(ws):
                for ix in range(ws):
                    y, x = wy * ws + iy, wx * ws + ix
                    r.append(((y + shift) % H) * W + (x + shift) % W)
                    reg.append(region(y, H) * 3 + region(x, W) if shift else 0)
            rows += r
            regions.append(reg)
    masks, wtype = [], []
    for reg in regions:
        reg = np.array(reg)
        m = np.where(reg[:, None] != reg[None, :], -100.0, 0.0).astype(np.float32)
        for t, mm in enumerate(masks):
            if np.array_equal(mm, m):
                wtype.append(t)
                break
        else:
            wtype.append(len(masks))
            masks.append(m)
    return np.array(rows, np.int32), np.array(wtype, np.int32), np.stack(masks)


def _rel_index(ws):
    c = np.stack(np.meshgrid(np.arange(ws), np.arange(ws), indexing="ij")).reshape(2, -1)
    rel = (c[:, :, None] - c[:, None, :]).transpose(1, 2, 0) + (ws - 1)
    return rel[:, :, 0] * (2 * ws - 1) + rel[:, :, 1]


class ClapAudio:
    """The audio tower of a directory of scripts/convert_music.py --clap.
    embed(clips) gives the vectors (clips, 512) of clips of 10 s at 48 kHz."""

    def __init__(self, path, gpu=False, cpu_threads=None, W=None):
        self.W = W or SafeTensors(os.path.join(path, "model.safetensors"))
        self.gpu = bool(gpu)
        self.cpu_threads = cpu_threads
        ac = json.load(open(os.path.join(path, "config.json")))["audio_config"]
        self.depths = list(ac.get("depths", [2, 2, 12, 2]))
        self.width0 = int(ac.get("patch_embeds_hidden_size", 128))
        self.ws = 8
        self.eps = 1e-5
        W = self.W
        p = "audio_model.audio_encoder."
        f = lambda n: np.asarray(W.get(p + n), dtype=np.float32)  # noqa: E731

        def lin(name, bias=True):
            out = _Plain(np.ascontiguousarray(f(name + ".weight").reshape(W.header[p + name + ".weight"]["shape"][0], -1)))
            out.b = f(name + ".bias") if bias else None
            return out
        s = f("batch_norm.weight") / np.sqrt(f("batch_norm.running_var") + np.float32(1e-5))
        self.bn = (s, f("batch_norm.bias") - f("batch_norm.running_mean") * s)
        self.patch = lin("patch_embed.proj")
        self.patch_ln = (f("patch_embed.norm.weight"), f("patch_embed.norm.bias"))
        self.stages = []
        grid = SPEC // 4
        for si, depth in enumerate(self.depths):
            C = self.width0 * 2 ** si
            heads = W.header[p + "layers.%d.blocks.0.attention.self.relative_position_bias_table" % si]["shape"][1]
            ws = min(self.ws, grid)
            blocks = []
            for bi in range(depth):
                q = "layers.%d.blocks.%d." % (si, bi)
                shift = 0 if (bi % 2 == 0 or grid <= self.ws) else self.ws // 2
                order, wtype, masks = _window_order(grid, grid, ws, shift)
                table = f(q + "attention.self.relative_position_bias_table")      # (225, heads)
                rel = table[_rel_index(ws).reshape(-1)].reshape(ws * ws, ws * ws, heads).transpose(2, 0, 1)
                bias = np.ascontiguousarray(rel[None] + masks[:, None], dtype=np.float32)
                blocks.append(dict(
                    order=order[:, None], back=np.argsort(order).astype(np.int32)[:, None], wtype=wtype, bias=bias,
                    ln1=(f(q + "layernorm_before.weight"), f(q + "layernorm_before.bias")),
                    q=lin(q + "attention.self.query"), k=lin(q + "attention.self.key"),
                    v=lin(q + "attention.self.value"), o=lin(q + "attention.output.dense"),
                    ln2=(f(q + "layernorm_after.weight"), f(q + "layernorm_after.bias")),
                    up=lin(q + "intermediate.dense"), down=lin(q + "output.dense")))
            st = dict(C=C, heads=heads, grid=grid, ws=ws, blocks=blocks)
            if si < len(self.depths) - 1:
                h2 = grid // 2
                yy, xx = np.meshgrid(np.arange(h2), np.arange(h2), indexing="ij")
                y, x = (2 * yy).reshape(-1), (2 * xx).reshape(-1)
                st["merge"] = np.stack([y * grid + x, (y + 1) * grid + x, y * grid + x + 1,
                                        (y + 1) * grid + x + 1], 1).astype(np.int32)
                d = "layers.%d.downsample." % si
                st["merge_ln"] = (f(d + "norm.weight"), f(d + "norm.bias"))
                st["reduce"] = lin(d + "reduction", bias=False)
                grid = h2
            self.stages.append(st)
        self.norm = (f("norm.weight"), f("norm.bias"))
        g = lambda n: np.asarray(W.get(n), dtype=np.float32)  # noqa: E731
        self.proj = [(g("audio_projection.linear%d.weight" % i).T.copy(), g("audio_projection.linear%d.bias" % i))
                     for i in (1, 2)]
        if self.gpu:
            gpu_bf12([self.patch] + [b[k] for st in self.stages for b in st["blocks"]
                                     for k in ("q", "k", "v", "o", "up", "down")] +
                     [st["reduce"] for st in self.stages if "reduce" in st], "clap audio")
        self.runner = _Runner(self._team, keep=2, keep_gpu=1)

    def _team(self, n, gpu):
        p = self.program(n, gpu)
        if not gpu:     # word 3 of the header: the team (cpu_threads, or the CPUs of the taskset)
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def patches(self, samples):
        """The patches (4096, 16) of one clip: the host's part."""
        return clap_patches(samples, self.bn)

    def program(self, n_clips=1, gpu=False):
        """The program of one clip: input "pe" (4096, 16), output "xn" (64,
        1024), the final LayerNorm."""
        assert n_clips == 1
        c = _EncCompiler(self.eps, pack=not gpu, rows=1, kmax=1)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        n0 = (SPEC // 4) ** 2
        c.env.update(pe=z(n0, 16))
        forms = []
        for si, st in enumerate(self.stages):
            C, n, s_ = st["C"], st["grid"] ** 2, "%d" % si
            nm = lambda k: k + s_  # noqa: E731
            c.env.update(**{nm(k): z(n, C) for k in ("x", "h", "hw", "q", "k", "v", "o", "t", "tg")})
            c.env[nm("u")] = z(n, 4 * C)
            if si == 0:
                forms += [("set", "x0", ("enc_linear", self.patch, "pe")),
                          ("set", "x0", ("enc_lnorm", "x0", self.patch_ln[0], self.patch_ln[1], C))]
            hd = C // st["heads"]
            g = st["ws"] ** 2
            for b in st["blocks"]:
                forms += [
                    ("set", nm("h"), ("enc_lnorm", nm("x"), b["ln1"][0], b["ln1"][1], C)),
                    ("set", nm("hw"), ("enc_gather", nm("h"), b["order"], C)),
                    ("set", nm("q"), ("enc_linear", b["q"], nm("hw"))),
                    ("set", nm("k"), ("enc_linear", b["k"], nm("hw"))),
                    ("set", nm("v"), ("enc_linear", b["v"], nm("hw"))),
                    ("set", nm("o"), ("enc_attn_bias", nm("q"), nm("k"), nm("v"), b["bias"], b["wtype"],
                                      st["heads"], hd, g, hd ** -0.5)),
                    ("set", nm("t"), ("enc_linear", b["o"], nm("o"))),
                    ("set", nm("tg"), ("enc_gather", nm("t"), b["back"], C)),
                    ("enc_add", nm("x"), nm("tg")),
                    ("set", nm("h"), ("enc_lnorm", nm("x"), b["ln2"][0], b["ln2"][1], C)),
                    ("set", nm("u"), ("enc_linear", b["up"], nm("h"))),
                    ("set", nm("u"), ("enc_gelu", nm("u"), True)),
                    ("set", nm("t"), ("enc_linear", b["down"], nm("u"))),
                    ("enc_add", nm("x"), nm("t")),
                ]
            if "merge" in st:
                n2 = n // 4
                c.env.update(**{nm("m"): z(n2, 4 * C), nm("mn"): z(n2, 4 * C)})
                forms += [("set", nm("m"), ("enc_gather", nm("x"), st["merge"], C)),
                          ("set", nm("mn"), ("enc_lnorm", nm("m"), st["merge_ln"][0], st["merge_ln"][1], 4 * C)),
                          ("set", "x%d" % (si + 1), ("enc_linear", st["reduce"], nm("mn")))]
        last = len(self.stages) - 1
        C, n = self.stages[-1]["C"], self.stages[-1]["grid"] ** 2
        c.env["xn"] = z(n, C)
        forms.append(("set", "xn", ("enc_lnorm", "x%d" % last, self.norm[0], self.norm[1], C)))
        c.compile(("seq", *forms))
        return c.p.finish()

    def prep(self, samples):
        """The host's part of one clip (its patches)."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(x) != CLIP * RATE:
            raise ValueError("a CLAP clip is %d samples (10 s at 48 kHz), not %d" % (CLIP * RATE, len(x)))
        return self.patches(x)

    def run(self, prep):
        """The rows of the final LayerNorm (64, 1024) of a clip's prep."""
        return self.runner.run(1, self.gpu, {"pe": prep}, "xn")

    def vectors(self, outs):
        """The vectors (clips, 512) of the final rows of clips."""
        lat = np.stack([o.mean(0) for o in outs])
        h = np.maximum(lat @ self.proj[0][0] + self.proj[0][1], 0)
        y = h @ self.proj[1][0] + self.proj[1][1]
        return y / np.linalg.norm(y, axis=1, keepdims=True)

    def embed(self, clips, rows=False):
        """The vectors (len(clips), 512) of clips of 10 s at 48 kHz."""
        outs = [self.run(self.prep(x)) for x in clips]
        lat = np.stack([o.mean(0) for o in outs])
        h = np.maximum(lat @ self.proj[0][0] + self.proj[0][1], 0)
        y = h @ self.proj[1][0] + self.proj[1][1]
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return (y, outs) if rows else y
