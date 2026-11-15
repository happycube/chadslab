"""Whisper's audio encoder (openai/whisper-large-v3 as transformers'
WhisperEncoder) on windows of 30 s of 16 kHz mono.

Ported from transformers (src/transformers/models/whisper/modeling_whisper.py, Copyright 2022 The
OpenAI Authors and The HuggingFace Inc. team; models/whisper/feature_extraction_whisper.py,
Copyright 2022 The HuggingFace Inc. team; audio_utils.py, Copyright 2023 The HuggingFace Inc. team
and the librosa & torchaudio authors; Apache License 2.0). See THIRD_PARTY_NOTICES.md.

    the log mel of WhisperFeatureExtractor: the samples padded with zeros to
        30 s, the STFT (n_fft 400, hop 160, periodic Hann, centered with
        reflect padding), power 2, 128 Slaney mels (0 to 8 kHz, Slaney norm),
        log10 (floor 1e-10), the last frame dropped (3000 frames), floored at
        the window's peak - 8, (x + 4) / 4: on the host (log_mel_all: the
        same of longer audio, unpadded, floored at the whole audio's peak)
    conv1 (128 -> 1280, k3, padding 1), GELU erf; conv2 (1280 -> 1280, k3,
        stride 2, padding 1), GELU erf: 1500 rows; + the position table
    32 pre-LN layers of width 1280 (20 heads of 64, FFN 5120 with GELU erf):
        LN, q (with bias, scale 1/8: folded into q) k (no bias) v (with bias),
        the attention of the window over itself, the output with bias, + x;
        LN, fc1, GELU erf, fc2, + x. Then the encoder's LN

The convs are products over the im2col (each output row's 3 input rows):
ENC_GATHER_MM on the GPU with BF12 rows, ENC_GATHER and ENC_LINEAR otherwise.
A program takes n windows, each in 1536 rows (its 1500 and 36 of zeros) so that
the blocks of queries (8 of ENC_ATTN_RANGE on the CPU, 64 of ENC_ATTN_X2 and 128
of ENC_ATTN_TILE on the GPU) and their steps of keys fall on a window's rows as in
a program of it alone. The attention on the GPU is ENC_ATTN_X2 (the tensor cores, q k v and
the weights as two bfloat16 planes; RECLANK_GPU_ENC_X2=0: float32, ENC_ATTN_TILE). The checkpoint's matrices are
float16: a CPU program takes them as float32 (exact), a GPU program as they
are (ENC_LINEAR wbf 2: the tensor cores, each row of x scaled by a power of
two and as two float16 planes, float32 sums; RECLANK_WHISPER_F32=1:
float32), or with bf12=True their BF12 rows.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Runner, _shape
from reclank.formats.safetensors import SafeTensors
from reclank.models.music.frontend import clap_filters
from reclank.models.music.text import _lin, gpu_bf12
from reclank.runtime.threads import team_threads

RATE, N_FFT, HOP, N_MELS = 16000, 400, 160, 128
WINDOW_S = 30
SAMPLES = WINDOW_S * RATE            # 480000
FRAMES = SAMPLES // HOP              # 3000 mel frames
ROWS, WIN_ROWS = FRAMES // 2, 1536   # the encoder's rows of a window, and their room in a program

_FB = None


def log_mel(samples):
    """The input features (128, 3000) of at most 30 s of 16 kHz samples
    (WhisperFeatureExtractor's)."""
    global _FB
    if _FB is None:
        _FB = clap_filters(sr=RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=8000.0)
    x = np.asarray(samples, dtype=np.float64).reshape(-1)[:SAMPLES]
    x = np.pad(x, (0, SAMPLES - len(x)))
    p = N_FFT // 2
    xp = np.pad(x, (p, p), mode="reflect")
    n = 1 + (len(xp) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(xp, (n, N_FFT), (xp.strides[0] * HOP, xp.strides[0]))
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)       # periodic Hann
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    m = np.log10(np.maximum(spec[:-1] @ _FB, 1e-10)).T                   # (mels, 3000)
    m = np.maximum(m, m.max() - 8.0)
    return ((m + 4.0) / 4.0).astype(np.float32)


def log_mel_all(samples):
    """The input features (128, len // 160) of more than 30 s of 16 kHz
    samples, as WhisperFeatureExtractor's of the whole audio (truncation=False,
    padding="longest": transformers' long-form input): log_mel's steps
    without the padding to 30 s, floored at the peak of the whole audio - 8.
    The STFT in pieces of 3000 frames (its frames are independent)."""
    global _FB
    if _FB is None:
        _FB = clap_filters(sr=RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=8000.0)
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    p = N_FFT // 2
    xp = np.pad(x, (p, p), mode="reflect")
    n = len(x) // HOP                                                    # the last frame dropped
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    m = np.empty((N_MELS, n), np.float64)
    for a in range(0, n, FRAMES):
        b = min(a + FRAMES, n)
        frames = np.lib.stride_tricks.as_strided(xp[a * HOP:], (b - a, N_FFT), (xp.strides[0] * HOP, xp.strides[0]))
        spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
        m[:, a:b] = np.log10(np.maximum(spec @ _FB, 1e-10)).T
    m = np.maximum(m, m.max() - 8.0)
    return ((m + 4.0) / 4.0).astype(np.float32)


def _conv_index(T, stride):
    """The im2col index (out rows, 3) of a conv k3, padding 1, over T rows."""
    To = (T - 1) // stride + 1
    t = np.arange(To)[:, None] * stride - 1 + np.arange(3)[None, :]
    return np.where((t >= 0) & (t < T), t, -1).astype(np.int32)


class WhisperEncoder:
    """The encoder of a Whisper checkpoint directory (transformers' layout).
    encode(features) gives the last hidden states (1500, 1280) of windows."""

    def __init__(self, path, gpu=False, bf12=False, cpu_threads=None, W=None, batch=4):
        self.W = W or SafeTensors(os.path.join(path, "model.safetensors"))
        self.gpu, self.cpu_threads, self.batch = bool(gpu), cpu_threads, batch
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg["num_mel_bins"] == N_MELS and cfg["max_source_positions"] == ROWS
        assert cfg["activation_function"] == "gelu" and not cfg.get("scale_embedding")
        self.width, self.heads = cfg["d_model"], cfg["encoder_attention_heads"]
        hd = self.width // self.heads
        p = "model.encoder."
        f = lambda n: np.asarray(self.W.get(p + n), dtype=np.float32)  # noqa: E731
        self.eps = 1e-5

        def conv(name):
            """a conv k3 as a linear over the im2col: W (out, 3 in), the columns (tap, channel)"""
            w = f(name + ".weight")
            lin = _lin(self.W, p + name)
            lin.w = np.ascontiguousarray(w.transpose(0, 2, 1).reshape(w.shape[0], -1))
            return lin
        self.c1, self.c2 = conv("conv1"), conv("conv2")
        self.pos = f("embed_positions.weight")
        self.layers = []
        for i in range(cfg["encoder_layers"]):
            r = "layers.%d." % i
            q = p + r
            self.layers.append(dict(
                ln_a=(f(r + "self_attn_layer_norm.weight"), f(r + "self_attn_layer_norm.bias")),
                aq=_lin(self.W, q + "self_attn.q_proj", scale=hd ** -0.5), ak=_lin(self.W, q + "self_attn.k_proj", bias=False),
                av=_lin(self.W, q + "self_attn.v_proj"), ao=_lin(self.W, q + "self_attn.out_proj"),
                ln_f=(f(r + "final_layer_norm.weight"), f(r + "final_layer_norm.bias")),
                fc1=_lin(self.W, q + "fc1"), fc2=_lin(self.W, q + "fc2")))
        self.ln_end = (f("layer_norm.weight"), f("layer_norm.bias"))
        lins = [self.c1, self.c2] + [L[k] for L in self.layers for k in ("aq", "ak", "av", "ao", "fc1", "fc2")]
        if self.gpu and bf12:
            gpu_bf12(lins, "whisper encoder")
        elif self.gpu and os.environ.get("RECLANK_WHISPER_F32", "0") != "1":
            # the checkpoint's float16 matrices as they are (q's scale 1/8 too): the tensor
            # cores take them with x in two float16 planes (ENC_LINEAR wbf 2)
            for lin in lins:
                lin.w = lin.w.astype(np.float16)
        self.runner = _Runner(lambda n, gpu: self._team(self.program(n, gpu), gpu), keep=3, keep_gpu=2)

    def _team(self, p, gpu):
        """A CPU program's team in its header (word 3): cpu_threads, or the
        CPUs of the taskset (threads.team_threads)."""
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def program(self, n=1, gpu=False, tail=None):
        """The program of n windows: input "m" (each window's features as rows
        of frames (3000, 128), one window after another), output "hid"
        (n 1536, 1280): each window's last hidden state in its first 1500 rows.
        tail(c, n): more forms after "hid" (and the names they need in c.env):
        the decoder's keys and values of the windows (decoder.Whisper)."""
        w, H = self.width, self.heads
        hd = w // H
        S, R = WIN_ROWS, n * WIN_ROWS
        i1 = _conv_index(FRAMES, 1)                                # (3000, 3) of a window's frames
        i2 = _conv_index(FRAMES, 2)                                # (1500, 3) of its conv1 rows
        win = np.repeat(np.arange(n), S)           # a row's window: its keys are the window's 1500 rows
        i1n = np.concatenate([np.where(i1 >= 0, i1 + j * FRAMES, -1) for j in range(n)]).astype(np.int32)
        # conv2 of each window into its 1536 rows (the 36 past its 1500: zeros)
        i2n = np.full((R, 3), -1, np.int32)
        for j in range(n):
            i2n[j * S:j * S + ROWS] = np.where(i2 >= 0, i2 + j * FRAMES, -1)
        pe = np.zeros((R, w), np.float32)
        for j in range(n):
            pe[j * S:j * S + ROWS] = self.pos
        inter = _shape(self.layers[0]["fc1"])[0]
        c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=inter)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(m=z(n * FRAMES, N_MELS), a=z(n * FRAMES, w), x=z(R, w), h=z(R, w), t=z(R, w), u=z(R, inter),
                     q=z(R, w), k=z(R, w), v=z(R, w), o=z(R, w), hid=z(R, w), pe=pe,
                     lo=(win * S).astype(np.int32), hi=(win * S + ROWS).astype(np.int32))
        forms = [
            ("set", "a", ("enc_gather_linear", self.c1, "m", i1n, N_MELS)),
            ("set", "a", ("enc_gelu", "a", True)),
            ("set", "x", ("enc_gather_linear", self.c2, "a", i2n, w)),
            ("set", "x", ("enc_gelu", "x", True)),
            ("enc_add", "x", "pe"),
        ]
        for L in self.layers:
            forms += [
                ("set", "h", ("enc_lnorm", "x", L["ln_a"][0], L["ln_a"][1], w)),
                ("set", "q", ("enc_linear", L["aq"], "h")),
                ("set", "k", ("enc_linear", L["ak"], "h")),
                ("set", "v", ("enc_linear", L["av"], "h")),
                ("set", "o", ("enc_attn_range_x2", "q", "k", "v", "lo", "hi", H, hd)),
                ("set", "t", ("enc_linear", L["ao"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", L["ln_f"][0], L["ln_f"][1], w)),
                ("set", "u", ("enc_linear", L["fc1"], "h")),
                ("set", "u", ("enc_gelu", "u", True)),
                ("set", "t", ("enc_linear", L["fc2"], "u")),
                ("enc_add", "x", "t"),
            ]
        forms.append(("set", "hid", ("enc_lnorm", "x", self.ln_end[0], self.ln_end[1], w)))
        if tail is not None:
            forms += tail(c, n)
        c.compile(("seq", *forms))
        return c.p.finish()

    def encode(self, features):
        """The last hidden states (1500, 1280) of windows' features ((128,
        3000) each, log_mel), in programs of self.batch windows."""
        hs = []
        for i in range(0, len(features), self.batch):
            part = features[i:i + self.batch]
            m = np.concatenate([np.ascontiguousarray(f.T) for f in part])
            h = self.runner.run(len(part), self.gpu, {"m": m}, "hid")
            hs += list(h.reshape(len(part), WIN_ROWS, -1)[:, :ROWS])
        return hs
