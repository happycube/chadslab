"""ACE-Step 1.5's VAE decoder (diffusers' AutoencoderOobleck, Stable Audio's):
latents (T frames of 64, 25 a second) to 48 kHz stereo, 1920 samples a
frame.

Ported from diffusers
(src/diffusers/models/autoencoders/autoencoder_oobleck.py; Apache License 2.0;
Copyright 2026 The HuggingFace Team) and ACE-Step 1.5
(acestep/core/generation/handler/vae_decode_chunks.py; MIT License, Copyright
(c) 2026 ACEStep; its model files Apache License 2.0, Copyright 2025 The
ACESTEO Team). See THIRD_PARTY_NOTICES.md.

    the DAC-style decoder (models/audio/dac.py): conv1 (k 7, 64 -> 2048), five
    blocks of Snake, ConvTranspose1d (strides 10 6 4 4 2) and three residual
    units, the channels 2048 1024 512 256 128 128; Snake, conv2 (k 7, 128 ->
    2, no bias)

The convs are weight-normed (weight_g, weight_v: W = g v / |v|, the norm over
all but the first axis, as torch's weight_norm), folded on the host in
float32: their W is float32 on the GPU too (not the checkpoint's bfloat16).
Snake takes exp of its alpha and beta (logscale).

decode_tiled: the authors' overlap-discard tiles (vae_decode_chunks.py:
windows of chunk frames, each keeping its core of chunk - 2 overlap frames
and dropping the overlap's samples at each side), which bound the memory of
a long piece.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from reclank.formats.safetensors import SafeTensors
from reclank.models.audio.dac import DacDecoder, conv_lin, tconv_lins


def _wn(W, name):
    """W = g v / |v| of a weight-normed conv (float32), the norm over all
    but the first axis."""
    g = np.asarray(W.get(name + ".weight_g"), dtype=np.float32)
    v = np.asarray(W.get(name + ".weight_v"), dtype=np.float32)
    n = np.sqrt((v.astype(np.float64) ** 2).sum(axis=tuple(range(1, v.ndim)), keepdims=True)).astype(np.float32)
    return (g * (v / n)).astype(np.float32)


class OobleckVAE:
    """The decoder of a vae directory (config.json and its safetensors)."""

    def __init__(self, path, gpu=False, cpu_threads=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg["_class_name"] == "AutoencoderOobleck"
        W = SafeTensors(os.path.join(path, "diffusion_pytorch_model.safetensors"))
        self.rate, self.channels = cfg["sampling_rate"], cfg["audio_channels"]
        ratios = list(reversed(cfg["downsampling_ratios"]))
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        bias = lambda n: f(n + ".bias") if n + ".bias" in W.header else None  # noqa: E731
        conv = lambda n: conv_lin(_wn(W, n), bias(n), gpu)  # noqa: E731
        snake = lambda n: (np.exp(f(n + ".alpha")).reshape(-1), np.exp(f(n + ".beta")).reshape(-1))  # noqa: E731
        d = "decoder."
        blocks = []
        for b, r in enumerate(ratios):
            p = d + "block.%d." % b
            blocks.append(dict(
                snake=snake(p + "snake1"), up=tconv_lins(_wn(W, p + "conv_t1"), bias(p + "conv_t1"), r, gpu),
                res=[dict(s1=snake(p + "res_unit%d.snake1" % u), c1=conv(p + "res_unit%d.conv1" % u),
                          s2=snake(p + "res_unit%d.snake2" % u), c2=conv(p + "res_unit%d.conv2" % u))
                     for u in (1, 2, 3)]))
        self.dac = DacDecoder(conv(d + "conv1"), blocks, snake(d + "snake1"), conv(d + "conv2"), gpu=gpu,
                              cpu_threads=cpu_threads)
        self.hop = self.dac.hop

    def decode(self, z):
        """The audio (channels, T hop) of latents z (T, 64)."""
        return self.dac.decode(np.ascontiguousarray(z, dtype=np.float32)).T.copy()

    def decode_tiled(self, z, chunk=256, overlap=64):
        """decode() in the authors' tiles of chunk frames (overlap frames each
        side dropped): (channels, T hop)."""
        T = z.shape[0]
        if T <= chunk:
            return self.decode(z)
        stride = chunk - 2 * overlap
        out = []
        for i in range(math.ceil(T / stride)):
            c0, c1 = i * stride, min(i * stride + stride, T)
            w0, w1 = max(0, c0 - overlap), min(T, c1 + overlap)
            a = self.decode(z[w0:w1])
            out.append(a[:, (c0 - w0) * self.hop:a.shape[1] - (w1 - c1) * self.hop])
        return np.concatenate(out, 1)
