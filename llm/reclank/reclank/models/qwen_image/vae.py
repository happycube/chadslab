"""Qwen-Image-2.1's VAE decoder (diffusers' AutoencoderKLQwenImage21: Wan's
causal 3D VAE as an image one, 64 latent channels, 16 x, RGBA out) as a
program: an image of one frame's latents.

Ported from diffusers
(src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py; Apache
License 2.0; Copyright 2026 The Qwen Team and The HuggingFace Team). See
THIRD_PARTY_NOTICES.md.

    post_quant_conv (1 x 1), conv_in (3 x 3, 64 -> 1152)
    the mid block: a residual block, the attention of all pixels over all
        pixels (the RMS norm, q k v by one 1 x 1 conv with bias, one head of
        1152, scale 1152^-1/2, proj, + x), a residual block
    five up blocks (1152 1152 576 288 144 channels out) of three residual
        blocks (the RMS norm, SiLU, conv, the RMS norm, SiLU, conv, + x or a
        1 x 1 conv of x), the first four with a 2 x nearest upsampling and a
        3 x 3 conv after them, plus the block's shortcut (DupUp3D of its
        input: ENC_DUPUP)
    the RMS norm, SiLU, conv_out (3 x 3, 144 -> 4), clamped to [-1, 1]

The norm of a pixel (QwenImage21RMS_norm): x / |x| sqrt(channels) gamma,
the RMS norm of its channels (ENC_RMS with no eps). A frame alone: the
causal convs are 2D convs (the checkpoint's are), the temporal upsampling's
time_conv does not run on the first frame, and DupUp3D keeps the last of its
temporal copies (first_chunk). The rows are pixels (row major), the columns
channels. A 3 x 3 conv is the im2col of its 9 neighbours (ENC_GATHER, the
upsampling in the same gather) into a buffer of at most COL values, then
ENC_LINEAR of W (out, 3, 3, in), a tile of rows at a time (on the GPU the
norm and SiLU before a conv run on the tile's gathered pixels: no buffer of
the normed image). The checkpoint's float32 matrices as they are (CPU and
GPU). The buffers are views of a few flat arrays, given by the order of the
forms (each view's array is free again after its last use): on the GPU 4.2
GB at 1024 x 1024, 13 GB at 2048 x 2048.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.flux2.vae import conv_index
from reclank.runtime.threads import team_threads

COL = 1 << 26       # the values of an im2col tile (256 MB)
LIN_ROWS = 1 << 18  # the rows of a 1 x 1 conv at a time (its buffer; the GPU's product: at most 65535 tiles of 64 rows)
NORM_EPS = 1e-30    # F.normalize's max(|x|, 1e-12): no eps but for a zero pixel


def down_index(h, w):
    """The im2col index (h/2 w/2, 9) of a 3 x 3 conv of stride 2 over an
    image of h x w pixels padded by one row and column at the bottom and the
    right (Resample's ZeroPad2d((0, 1, 0, 1))); -1 past the edge."""
    y = 2 * np.arange(h // 2)[:, None, None, None] + np.arange(3)[None, None, :, None]
    x = 2 * np.arange(w // 2)[None, :, None, None] + np.arange(3)[None, None, None, :]
    ok = (y < h) & (x < w)
    return np.where(ok, y * w + x, -1).reshape(-1, 9).astype(np.int32)


def avg_down(cin, cout, ft, fs):
    """AvgDown3D of one frame as a linear of the fs x fs pixels of each
    output pixel (the gathered rows: pixel s = row fs + column, then the
    channel): diffusers' channels (c, t, s) of factor ft fs^2, in groups of
    cin factor / cout, each output their mean; ft 2 pads a zero frame in
    front (t 0), so its part of a group adds zeros. W (cout, fs^2 cin)
    float32 (the means' 1 / group a power of 2 here: the same values)."""
    s2 = fs * fs
    factor = ft * s2
    group = cin * factor // cout
    assert cin * factor % cout == 0
    w = np.zeros((cout, s2 * cin), np.float32)
    for o in range(cout):
        for v in range(o * group, (o + 1) * group):
            c, r = divmod(v, factor)
            t, sp = divmod(r, s2)
            if t == ft - 1:
                w[o, sp * cin + c] += np.float32(1.0 / group)
    lin = _Plain(w)
    lin.shape = w.shape
    return lin


class _Build:
    """The forms of a VAE program and its buffers: views of a few flat
    arrays, given by the order of the forms (each view's array is free again
    after its last use)."""

    def __init__(self, c, gpu):
        self.c, self.env, self.gpu = c, c.env, gpu
        self.forms = []
        self.views = {}         # a buffer's name: (its array's slot, rows, cols)
        self.slots = []         # the sizes of the flat arrays
        self.free = []          # the slots free now
        self.tiles = {}         # a tile's name: (its buffer's name, r0, r1)
        self.index = {}         # (hh, ww, kind): the im2col index of a level, one array for its convs
        self.n = 0
        self.env["col"] = np.zeros(COL, np.float32)

    def name(self, prefix):
        self.n += 1
        return "%s%d" % (prefix, self.n)

    def take(self, rows, cols):
        """a new buffer (rows, cols): the smallest free slot that holds it,
        else the largest free one (grown), else a new slot"""
        need = rows * cols
        fit = [s for s in self.free if self.slots[s] >= need]
        s = min(fit, key=lambda s: self.slots[s]) if fit else \
            max(self.free, key=lambda s: self.slots[s]) if self.free else None
        if s is None:
            s = len(self.slots)
            self.slots.append(0)
        else:
            self.free.remove(s)
        self.slots[s] = max(self.slots[s], need)
        name = self.name("_b")
        self.views[name] = (s, rows, cols)
        return name

    def drop(self, *names):
        for n in names:
            self.free.append(self.views[n][0])

    def conv3(self, x, hh, ww, cin, lin, out, kind=None, gamma=None):
        """a 3 x 3 conv of x (hh x ww pixels of cin channels) into out (kind
        "up": of x's 2 x nearest upsampling; "down": of stride 2, padded at
        the bottom and the right); gamma: of the norm and SiLU before it: in
        a GPU program on the gathered pixels of a tile (a pixel's own
        channels: the same values, with no buffer of x's size, at 9 times the
        norms' work; the zeros past the edge stay zeros), in a CPU program
        into a buffer of x's size first"""
        if gamma is not None and not self.gpu:
            y = self.take(hh * ww, cin)
            self.forms.extend([("set", y, ("enc_rms", x, gamma, cin)), ("set", y, ("enc_silu", y))])
            self.conv3(y, hh, ww, cin, lin, out, kind)
            self.drop(y)
            return
        idx = self.index.get((hh, ww, kind))
        if idx is None:
            idx = self.index[(hh, ww, kind)] = down_index(hh, ww) if kind == "down" else \
                conv_index(hh, ww, kind == "up")
        rows = idx.shape[0]
        step = max(1, min(rows, COL // (9 * cin)))
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            col, ot = self.name("_c"), self.name("_t")
            self.env[col] = self.env["col"][:(r1 - r0) * 9 * cin].reshape(r1 - r0, 9 * cin)
            self.tiles[ot] = (out, r0, r1)
            self.forms.append(("set", col, ("enc_gather", x, idx[r0:r1], cin)))
            if gamma is not None:
                self.forms.extend([("set", col, ("enc_rms", col, gamma, cin)), ("set", col, ("enc_silu", col))])
            self.forms.append(("set", ot, ("enc_linear", lin, col)))

    def resnet(self, x, hh, ww, cin, cout, R, free_x=True):
        """a residual block of x (freed unless not free_x); returns its
        output"""
        rows = hh * ww
        h1 = self.take(rows, cout)
        self.conv3(x, hh, ww, cin, R["c1"], h1, gamma=R["n1"])
        h2 = self.take(rows, cout)
        self.conv3(h1, hh, ww, cout, R["c2"], h2, gamma=R["n2"])
        self.drop(h1)
        if R["sc"] is not None:
            # the 1 x 1 conv of x a tile of rows at a time, added to its rows of h2
            s = self.take(min(rows, LIN_ROWS), cout)
            for r0 in range(0, rows, LIN_ROWS):
                r1 = min(rows, r0 + LIN_ROWS)
                xt, st, ht = self.name("_x"), self.name("_s"), self.name("_h")
                self.tiles[xt], self.tiles[st], self.tiles[ht] = (x, r0, r1), (s, 0, r1 - r0), (h2, r0, r1)
                self.forms.extend([("set", st, ("enc_linear", R["sc"], xt)), ("enc_add", ht, st)])
            self.drop(s)
        else:
            self.forms.append(("enc_add", h2, x))
        if free_x:
            self.drop(x)
        return h2

    def attention(self, x, rows, C, M):
        """the mid block's attention of x (rows pixels of C channels), added
        to x"""
        nn_ = self.take(rows, C)
        q, k, v, o = self.take(rows, C), self.take(rows, C), self.take(rows, C), self.take(rows, C)
        self.env["_scale"] = np.full(C, C ** -0.5, np.float32)
        self.forms.extend([("set", nn_, ("enc_rms", x, M["n"], C)),
                           ("set", q, ("enc_linear", M["q"], nn_)),
                           ("set", q, ("enc_mul_vec", q, "_scale")),
                           ("set", k, ("enc_linear", M["k"], nn_)),
                           ("set", v, ("enc_linear", M["v"], nn_))])
        self.drop(nn_)
        self.forms.extend([("set", o, ("enc_attn_wide", q, k, v)),
                           ("set", q, ("enc_linear", M["o"], o)),
                           ("enc_add", x, q)])
        self.drop(k, v, o, q)

    def avg_down(self, x, out, hh, ww, cin, cout, lin):
        """out (hh/2 x ww/2 pixels, cout) += AvgDown3D of x (hh x ww pixels,
        cin): the 2 x 2 pixels of each output pixel gathered, times lin
        (avg_down), a tile of rows at a time"""
        key = (hh, ww, "avg")
        idx = self.index.get(key)
        if idx is None:
            y = 2 * np.arange(hh // 2)[:, None, None, None] + np.arange(2)[None, None, :, None]
            xx = 2 * np.arange(ww // 2)[None, :, None, None] + np.arange(2)[None, None, None, :]
            idx = self.index[key] = (y * ww + xx).reshape(-1, 4).astype(np.int32)
        rows = idx.shape[0]
        step = max(1, min(rows, COL // (4 * cin), LIN_ROWS))
        s = self.take(step, cout)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            col, st, ot = self.name("_c"), self.name("_s"), self.name("_o")
            self.env[col] = self.env["col"][:(r1 - r0) * 4 * cin].reshape(r1 - r0, 4 * cin)
            self.tiles[st], self.tiles[ot] = (s, 0, r1 - r0), (out, r0, r1)
            self.forms.extend([("set", col, ("enc_gather", x, idx[r0:r1], cin)),
                               ("set", st, ("enc_linear", lin, col)),
                               ("enc_add", ot, st)])
        self.drop(s)

    def finish(self):
        """the arrays of the views and tiles; the program"""
        arrays = [np.zeros(n, np.float32) for n in self.slots]
        for name, (s, r, cc) in self.views.items():
            self.env[name] = arrays[s][:r * cc].reshape(r, cc)
        for name, (of, r0, r1) in self.tiles.items():
            self.env[name] = self.env[of][r0:r1]
        self.c.compile(("seq", *self.forms))
        return self.c.p.finish()


class QwenImageVAE:
    """The decoder of a vae directory (config.json and its safetensors).
    decode(z) gives the image (4, 16h, 16w) (RGBA) in [-1, 1] of latents
    (64, h, w)."""

    def __init__(self, path, gpu=False, cpu_threads=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg["is_residual"] and cfg["patch_size"] is None
        self.W = W = SafeTensors(os.path.join(path, "diffusion_pytorch_model.safetensors"))
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.latent = cfg["z_dim"]
        self.out_ch = cfg["out_channels"]
        self.mean = np.array(cfg["latents_mean"], np.float32)
        self.std = np.array(cfg["latents_std"], np.float32)
        mult = cfg["dim_mult"]
        base = cfg["decoder_base_dim"] or cfg["base_dim"]
        dims = [base * u for u in [mult[-1]] + mult[::-1]]
        tup = cfg["temperal_downsample"][::-1]
        g = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32).reshape(-1)  # noqa: E731
        d = "decoder."
        self.pq = self.conv("post_quant_conv")
        self.cin = self.conv(d + "conv_in")
        res = lambda p: dict(  # noqa: E731
            n1=g(p + "norm1.gamma"), c1=self.conv(p + "conv1"), n2=g(p + "norm2.gamma"), c2=self.conv(p + "conv2"),
            sc=self.conv(p + "conv_shortcut") if p + "conv_shortcut.weight" in W.header else None)
        m = d + "mid_block."
        a = m + "attentions.0."
        C = dims[0]
        qkv = self.conv(a + "to_qkv")
        q, k, v = (self._rows(qkv, i * C, (i + 1) * C) for i in range(3))
        self.mid = dict(r0=res(m + "resnets.0."), r1=res(m + "resnets.1."), n=g(a + "norm.gamma"),
                        q=q, k=k, v=v, o=self.conv(a + "proj"))
        self.up = []
        for i, (cin, cout) in enumerate(zip(dims[:-1], dims[1:])):
            p = d + "up_blocks.%d." % i
            up = i != len(mult) - 1
            self.up.append(dict(cin=cin, cout=cout, res=[res(p + "resnets.%d." % j) for j in range(cfg["num_res_blocks"] + 1)],
                                up=self.conv(p + "upsampler.resample.1") if up else None,
                                ft=2 if up and tup[i] else 1))
        self.nout = g(d + "norm_out.gamma")
        self.cout = self.conv(d + "conv_out")
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)
        self.cfg, self.in_ch = cfg, cfg["in_channels"]
        self.enc = None
        self.enc_runner = _Runner(self._build_enc, keep=2, keep_gpu=1)

    def conv(self, name):
        """A conv (out, in, k, k) as a linear of its im2col: W (out, k k in)
        float32, the bias."""
        raw = np.asarray(self.W.get(name + ".weight"), dtype=np.float32)
        raw = raw.transpose(0, 2, 3, 1).reshape(raw.shape[0], -1)
        lin = _Plain(np.ascontiguousarray(raw))
        lin.shape = raw.shape
        lin.b = np.ascontiguousarray(self.W.get(name + ".bias"), dtype=np.float32)
        return lin

    @staticmethod
    def _rows(lin, r0, r1):
        """The linear of rows [r0, r1) of lin (q, k, v of a fused conv)."""
        out = _Plain(np.ascontiguousarray(lin.w[r0:r1]))
        out.shape = out.w.shape
        out.b = np.ascontiguousarray(lin.b[r0:r1])
        return out

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def _build_enc(self, key, gpu):
        p = self.encoder_program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def program(self, h, w, gpu=False):
        """The program of latents of h x w: input "z" (h w, 64), output "img"
        (256 h w, 4)."""
        c = _EncCompiler(NORM_EPS, pack=not gpu)
        b = _Build(c, gpu)
        env = c.env
        env["z"] = np.zeros((h * w, self.latent), np.float32)
        H, Wd = h << 4, w << 4
        env["img"] = np.zeros((H * Wd, self.out_ch), np.float32)
        rows = h * w
        x = b.take(rows, self.latent)
        b.forms.append(("set", x, ("enc_linear", self.pq, "z")))
        C = self.up[0]["cin"]
        y = b.take(rows, C)
        b.conv3(x, h, w, self.latent, self.cin, y)
        b.drop(x)
        M = self.mid
        x = b.resnet(y, h, w, C, C, M["r0"])
        b.attention(x, rows, C, M)
        x = b.resnet(x, h, w, C, C, M["r1"])
        hh, ww = h, w
        for B in self.up:
            # the block's input stays for its shortcut
            keep = x if B["up"] is not None else None
            cin = B["cin"]
            for j, R in enumerate(B["res"]):
                x = b.resnet(x, hh, ww, cin, B["cout"], R, free_x=j > 0 or keep is None)
                cin = B["cout"]
            if B["up"] is not None:
                y = b.take(4 * hh * ww, B["cout"])
                b.conv3(x, hh, ww, B["cout"], B["up"], y, kind="up")
                b.drop(x)
                b.forms.append(("enc_dupup", keep, y, hh, ww, B["cin"], B["cout"], B["ft"], 2))
                b.drop(keep)
                x, hh, ww = y, 2 * hh, 2 * ww
        b.conv3(x, hh, ww, cin, self.cout, "img", gamma=self.nout)
        return b.finish()

    def _encoder(self):
        """The encoder's weights (at the first encode: text to image needs
        none)."""
        if self.enc is not None:
            return self.enc
        cfg = self.cfg
        W = self.W
        g = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32).reshape(-1)  # noqa: E731
        e = "encoder."
        res = lambda p: dict(  # noqa: E731
            n1=g(p + "norm1.gamma"), c1=self.conv(p + "conv1"), n2=g(p + "norm2.gamma"), c2=self.conv(p + "conv2"),
            sc=self.conv(p + "conv_shortcut") if p + "conv_shortcut.weight" in W.header else None)
        mult = cfg["dim_mult"]
        dims = [cfg["base_dim"] * u for u in [1] + mult]
        down = []
        for i, (cin, cout) in enumerate(zip(dims[:-1], dims[1:])):
            p = e + "down_blocks.%d." % i
            last = i == len(mult) - 1
            ft = 2 if not last and cfg["temperal_downsample"][i] else 1
            fs = 1 if last else 2
            down.append(dict(cin=cin, cout=cout, res=[res(p + "resnets.%d." % j) for j in range(cfg["num_res_blocks"])],
                             down=None if last else self.conv(p + "downsampler.resample.1"), ft=ft, fs=fs,
                             avg=avg_down(cin, cout, ft, fs) if fs > 1 else None))
        m = e + "mid_block."
        a = m + "attentions.0."
        C = dims[-1]
        qkv = self.conv(a + "to_qkv")
        q, k, v = (self._rows(qkv, i * C, (i + 1) * C) for i in range(3))
        self.enc = dict(cin=self.conv(e + "conv_in"), down=down,
                        mid=dict(r0=res(m + "resnets.0."), r1=res(m + "resnets.1."), n=g(a + "norm.gamma"),
                                 q=q, k=k, v=v, o=self.conv(a + "proj")),
                        nout=g(e + "norm_out.gamma"), cout=self.conv(e + "conv_out"),
                        # the latents' mean: the first z_dim rows of quant_conv (DiagonalGaussian's mode)
                        qc=self._rows(self.conv("quant_conv"), 0, self.latent), C=C)
        return self.enc

    def encoder_program(self, H, W, gpu=False):
        """The program of the encoder of an image of H x W pixels (multiples
        of 16): input "x" (H W, 4) in [-1, 1], output "zm" (H W / 256, 64)
        the latents' mean."""
        E = self._encoder()
        c = _EncCompiler(NORM_EPS, pack=not gpu)
        b = _Build(c, gpu)
        env = c.env
        env["x"] = np.zeros((H * W, self.in_ch), np.float32)
        env["zm"] = np.zeros((H * W >> 8, self.latent), np.float32)
        hh, ww = H, W
        x = b.take(hh * ww, E["down"][0]["cin"])
        b.conv3("x", hh, ww, self.in_ch, E["cin"], x)
        for B in E["down"]:
            # the block's input stays for its shortcut (AvgDown3D, or itself)
            keep, cin = x, B["cin"]
            for j, R in enumerate(B["res"]):
                x = b.resnet(x, hh, ww, cin, B["cout"], R, free_x=j > 0)
                cin = B["cout"]
            if B["down"] is not None:
                y = b.take((hh >> 1) * (ww >> 1), B["cout"])
                b.conv3(x, hh, ww, B["cout"], B["down"], y, kind="down")
                b.drop(x)
                b.avg_down(keep, y, hh, ww, B["cin"], B["cout"], B["avg"])
                x, hh, ww = y, hh >> 1, ww >> 1
            else:
                b.forms.append(("enc_add", x, keep))
            b.drop(keep)
        C = E["C"]
        M = E["mid"]
        rows = hh * ww
        x = b.resnet(x, hh, ww, C, C, M["r0"])
        b.attention(x, rows, C, M)
        x = b.resnet(x, hh, ww, C, C, M["r1"])
        y = b.take(rows, 2 * self.latent)
        b.conv3(x, hh, ww, C, E["cout"], y, gamma=E["nout"])
        b.drop(x)
        b.forms.append(("set", "zm", ("enc_linear", E["qc"], y)))
        return b.finish()

    def unpack(self, latents, hh, ww):
        """The VAE's latents (64, hh, ww) of the DiT's packed latents (hh ww,
        64): times the latents' std plus their mean."""
        x = latents.T.reshape(self.latent, hh, ww)
        return np.ascontiguousarray(x * self.std[:, None, None] + self.mean[:, None, None], dtype=np.float32)

    def decode(self, z):
        """The image (4, 16h, 16w) in [-1, 1] of latents z (64, h, w)."""
        _, h, w = z.shape
        prog, g = self.runner.get((h, w), self.gpu)
        prog.names["z"][...] = z.reshape(self.latent, h * w).T
        if g is not None:
            g.upload("z")
            g.run()
            g.download("img")
        else:
            prog.run()
        img = prog.names["img"].T.reshape(self.out_ch, 16 * h, 16 * w)
        return np.clip(img, -1.0, 1.0)

    def encode(self, img):
        """The latents (64, H/16, W/16) of an image (4, H, W) (RGBA) in [-1,
        1]: the encoder's mean (sample_mode argmax), less the latents' mean,
        over their std (_encode_vae_image)."""
        _, H, W = img.shape
        prog, g = self.enc_runner.get((H, W), self.gpu)
        prog.names["x"][...] = img.reshape(self.in_ch, H * W).T
        if g is not None:
            g.upload("x")
            g.run()
            g.download("zm")
        else:
            prog.run()
        z = prog.names["zm"].T.reshape(self.latent, H >> 4, W >> 4)
        return ((z - self.mean[:, None, None]) / self.std[:, None, None]).astype(np.float32)

    @staticmethod
    def pack(z):
        """The DiT's packed latents (h w, 64) of latents (64, h, w)."""
        return np.ascontiguousarray(z.reshape(z.shape[0], -1).T)
