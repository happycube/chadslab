"""The decoder of a DAC-style codec as a program: latent frames (rows: time,
columns: channels) to audio. The Higgs codec's acoustic decoder
(models/higgs/codec.py) and the Oobleck VAE of Stable Audio and ACE-Step
(models/acestep/vae.py). And the encoder (DacEncoder: the Higgs codec's
acoustic encoder): audio to frames.

Ported from diffusers
(src/diffusers/models/autoencoders/autoencoder_oobleck.py; Apache License 2.0;
Copyright 2026 The HuggingFace Team) and transformers
(src/transformers/models/dac/modeling_dac.py; Apache License 2.0; Copyright
2024 Descript and The HuggingFace Inc. team). See THIRD_PARTY_NOTICES.md.

    conv1 (k 7); blocks of Snake, ConvTranspose1d (stride r, kernel 2 r,
    padding ceil(r / 2): r rows a row) and three residual units (Snake, conv
    k 7 of dilation 1, 3, 9, Snake, conv k 1, + x); Snake, conv2 (k 7)

A conv is the gather of its taps (ENC_GATHER, -1 past the ends: zeros) and
ENC_LINEAR of W (out, k, in). A transposed conv of stride r: output row t
reads input rows i1 = (t + p) / r with tap k1 = (t + p) % r and i1 - 1 with
tap k1 + r (p the padding): r phases, each the gather of its rows' two
inputs, ENC_LINEAR of its taps' W, and ENC_SCATTER of the rows into their
places. Snake: ENC_SNAKE (alpha, beta). A linear's W: bfloat16 rows on the
GPU when the checkpoint's are (KQ_BF16), else float32.

The encoder: conv1 (k 7); blocks of three residual units (as the
decoder's), Snake, a conv of stride r (kernel 2 r, padding ceil(r / 2));
Snake, conv2 (k 3). Its convs are models/audio/conv.py's (tiles of output
rows: 24 kHz audio of 64 channels).
"""
from __future__ import annotations

import math

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner
from reclank.formats.gguf import BF16
from reclank.models.audio.conv import Convs, conv_rows
from reclank.models.qwen3.dense import _KQ
from reclank.runtime.threads import team_threads


def mat(raw, gpu):
    """A linear of W (out, in) given as bfloat16 (uint16) or float32 rows."""
    raw = np.ascontiguousarray(raw)
    if raw.dtype == np.uint16 and gpu:
        lin = _Plain(None)
        lin.kq = _KQ(raw.view(np.uint8), BF16, raw.shape[0], raw.shape[1])
    elif raw.dtype == np.uint16:
        lin = _Plain((raw.astype(np.uint32) << 16).view(np.float32))
    else:
        lin = _Plain(raw.astype(np.float32))
    lin.shape = raw.shape
    return lin


def aligned(w):
    """A conv's W (out, in, k) as it is when its rows (k in values) are a
    multiple of 32, else as float32 (bfloat16 rows exactly: the GPU's
    bfloat16 products take rows of 32 k values; a first conv of one
    channel)."""
    if w.dtype != np.uint16 or w.shape[1] * w.shape[2] % 32 == 0:
        return w
    return (w.astype(np.uint32) << 16).view(np.float32)


def conv_lin(w, b, gpu):
    """A Conv1d's W (out, in, k) (bfloat16 rows or float32) as a linear of
    its taps: W (out, k in); b its bias (or None)."""
    lin = mat(w.transpose(0, 2, 1).reshape(w.shape[0], -1), gpu)
    lin.b = None if b is None else np.ascontiguousarray(b, dtype=np.float32)
    lin.k = w.shape[2]
    return lin


def tconv_lins(w, b, r, gpu):
    """A ConvTranspose1d's W (in, out, 2 r) as r linears, phase j: W (out, 2
    in) of the taps j + r (input row i1 - 1) and j (row i1)."""
    assert w.shape[2] == 2 * r
    b = None if b is None else np.ascontiguousarray(b, dtype=np.float32)
    out = []
    for j in range(r):
        lin = mat(np.concatenate([w[:, :, j + r].T, w[:, :, j].T], axis=1), gpu)
        lin.b = b
        out.append(lin)
    return dict(phases=out, r=r, cin=w.shape[0], cout=w.shape[1])


class DacDecoder:
    """conv1 (conv_lin), blocks [dict(snake=(alpha, beta), up=tconv_lins,
    res=[dict(s1=(alpha, beta), c1=conv_lin, s2=..., c2=conv_lin)] x 3)],
    snake (alpha, beta), conv2 (conv_lin). decode(z) of z (T, conv1's in)."""

    def __init__(self, conv1, blocks, snake, conv2, gpu=False, cpu_threads=None):
        self.conv1, self.blocks, self.snake, self.conv2 = conv1, blocks, snake, conv2
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.ratios = [B["up"]["r"] for B in blocks]
        self.hop = int(np.prod(self.ratios))
        self.runner = _Runner(self._build, keep=4, keep_gpu=2)

    def _build(self, key, gpu):
        p = self.program(key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def program(self, T, gpu=False):
        """The program of T frames: input "z" (T, in), output "audio" (T hop,
        out channels)."""
        widths = [self.conv1.shape[0]] + [B["up"]["cout"] for B in self.blocks]
        c = _EncCompiler(1e-5, pack=not gpu, rows=T * self.hop, kmax=7 * max(widths))
        env = c.env
        lengths = [T]
        for r in self.ratios:
            lengths.append(lengths[-1] * r)
        # two buffers of activations and two of temps, each of the largest level
        big = max(L * ch for L, ch in zip(lengths, widths))
        col = max(max(L * 7 * ch for L, ch in zip(lengths, widths)),
                  max(lengths[i + 1] // r * 2 * widths[i] for i, r in enumerate(self.ratios)))
        for name in ("A", "B", "S", "U"):
            env[name] = np.zeros(big, np.float32)
        env["col"] = np.zeros(col, np.float32)
        env["z"] = np.zeros((T, self.conv1.shape[1] // self.conv1.k), np.float32)
        env["audio"] = np.zeros((lengths[-1], self.conv2.shape[0]), np.float32)
        count = [0]
        forms = []

        def view(base, rows, cols):
            count[0] += 1
            name = "_v%d" % count[0]
            env[name] = env[base][:rows * cols].reshape(rows, cols)
            return name

        def conv(x, rows, cin, lin, out, dil=1):
            """a conv of k taps (dilation dil, padding (k - 1) dil / 2) of x"""
            k = lin.k
            if k == 1:
                forms.append(("set", out, ("enc_linear", lin, x)))
                return
            t = np.arange(rows)[:, None] + dil * (np.arange(k)[None, :] - (k - 1) // 2)
            idx = np.where((t >= 0) & (t < rows), t, -1).astype(np.int32)
            cv = view("col", rows, k * cin)
            forms.extend([("set", cv, ("enc_gather", x, idx, cin)), ("set", out, ("enc_linear", lin, cv))])

        def tconv(x, rows, U, out):
            """a transposed conv of stride r: rows r rows out"""
            r, cin, cout = U["r"], U["cin"], U["cout"]
            p = math.ceil(r / 2)
            t = np.arange(rows * r)
            ph, i1 = (t + p) % r, (t + p) // r
            for j in range(r):
                ts = t[ph == j]
                ii = i1[ph == j]
                idx = np.stack([ii - 1, ii], 1)
                idx = np.where((idx >= 0) & (idx < rows), idx, -1).astype(np.int32)
                cv = view("col", len(ts), 2 * cin)
                tmp = view("U", len(ts), cout)
                forms.extend([("set", cv, ("enc_gather", x, idx, cin)),
                              ("set", tmp, ("enc_linear", U["phases"][j], cv)),
                              ("set", out, ("enc_scatter", tmp, ts.astype(np.int32)))])

        x = view("A", T, widths[0])
        conv("z", T, env["z"].shape[1], self.conv1, x)
        other = "B"
        for b, B in enumerate(self.blocks):
            rows, cin, cout = lengths[b], widths[b], widths[b + 1]
            s = view("S", rows, cin)
            forms.append(("set", s, ("enc_snake", x, *B["snake"])))
            y = view(other, rows * B["up"]["r"], cout)
            tconv(s, rows, B["up"], y)
            other = "A" if other == "B" else "B"
            x = y
            rows = lengths[b + 1]
            for u, R in enumerate(B["res"]):
                s1 = view("S", rows, cout)
                forms.append(("set", s1, ("enc_snake", x, *R["s1"])))
                h = view(other, rows, cout)
                conv(s1, rows, cout, R["c1"], h, dil=3 ** u)
                s2 = view("S", rows, cout)
                forms.append(("set", s2, ("enc_snake", h, *R["s2"])))
                h2 = view("U", rows, cout)
                conv(s2, rows, cout, R["c2"], h2)
                forms.append(("enc_add", x, h2))
        rows = lengths[-1]
        s = view("S", rows, widths[-1])
        forms.append(("set", s, ("enc_snake", x, *self.snake)))
        conv(s, rows, widths[-1], self.conv2, "audio")
        c.compile(("seq", *forms))
        return c.p.finish()

    def decode(self, z):
        """The audio (T hop, out channels) of frames z (T, in)."""
        T = z.shape[0]
        prog, g = self.runner.get(T, self.gpu)
        prog.names["z"][...] = z
        if g is not None:
            g.upload("z")
            g.run()
            g.download("audio")
        else:
            prog.run()
        return prog.names["audio"].copy()


class DacEncoder:
    """conv1 (conv_lin), blocks [dict(res=[dict(s1=(alpha, beta), c1=conv_lin,
    s2=..., c2=conv_lin)] x 3, snake=(alpha, beta), down=conv_lin, r=stride)],
    snake (alpha, beta), conv2 (conv_lin). encode(x) of samples x (L, in)."""

    def __init__(self, conv1, blocks, snake, conv2, gpu=False, cpu_threads=None):
        self.conv1, self.blocks, self.snake, self.conv2 = conv1, blocks, snake, conv2
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)

    def _build(self, key, gpu):
        p = self.program(key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def lengths(self, L):
        """The rows of each level of L samples: [L, after block 0, ...]
        (conv1, the residual units and conv2 keep the rows)."""
        out = [L]
        for B in self.blocks:
            r = B["r"]
            out.append(conv_rows(out[-1], 2 * r, r, math.ceil(r / 2)))
        return out

    def program(self, L, gpu=False):
        """The program of L samples: input "x" (L, in), output "z" (frames,
        conv2's out)."""
        widths = [self.conv1.shape[0]] + [B["down"].shape[0] for B in self.blocks]
        lengths = self.lengths(L)
        c = _EncCompiler(1e-5, pack=not gpu)
        env = c.env
        big = max(n * ch for n, ch in zip(lengths, widths))
        for name in ("A", "B", "S", "U"):
            env[name] = np.zeros(big, np.float32)
        env["x"] = np.zeros((L, self.conv1.shape[1] // self.conv1.k), np.float32)
        env["z"] = np.zeros((lengths[-1], self.conv2.shape[0]), np.float32)
        cv = Convs(c)
        forms = cv.forms
        x = cv.view("A", L, widths[0])
        cv.conv("x", L, env["x"].shape[1], self.conv1, x)
        other = "B"
        for b, B in enumerate(self.blocks):
            rows, ch = lengths[b], widths[b]
            for u, R in enumerate(B["res"]):
                s1 = cv.view("S", rows, ch)
                forms.append(("set", s1, ("enc_snake", x, *R["s1"])))
                h = cv.view(other, rows, ch)
                cv.conv(s1, rows, ch, R["c1"], h, dil=3 ** u)
                s2 = cv.view("S", rows, ch)
                forms.append(("set", s2, ("enc_snake", h, *R["s2"])))
                h2 = cv.view("U", rows, ch)
                cv.conv(s2, rows, ch, R["c2"], h2)
                forms.append(("enc_add", x, h2))
            s = cv.view("S", rows, ch)
            forms.append(("set", s, ("enc_snake", x, *B["snake"])))
            r = B["r"]
            y = cv.view(other, lengths[b + 1], widths[b + 1])
            cv.conv(s, rows, ch, B["down"], y, stride=r, pad=math.ceil(r / 2))
            x, other = y, ("A" if other == "B" else "B")
        rows = lengths[-1]
        s = cv.view("S", rows, widths[-1])
        forms.append(("set", s, ("enc_snake", x, *self.snake)))
        cv.conv(s, rows, widths[-1], self.conv2, "z")
        c.compile(("seq", *forms))
        return c.p.finish()

    def encode(self, x):
        """The frames (lengths(L)[-1], out) of samples x (L, in)."""
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(len(x), -1)
        return self.runner.run(len(x), self.gpu, {"x": x}, "z")
