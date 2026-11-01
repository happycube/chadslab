"""FLUX.2's VAE decoder (diffusers' AutoencoderKLFlux2: the Decoder of
AutoencoderKL with 32 latent channels, blocks of 512 512 256 128 channels, 3
ResNet blocks each, GroupNorm of 32 groups, eps 1e-6) as a program: an image
of the latents. Stable Diffusion's AutoencoderKL (4 latent channels, no batch
norm, the mid attention's old names query key value proj_attn) and FLUX.1's
(16 latent channels, no post_quant_conv: Z-Image's) are the same decoder.

Ported from diffusers
(src/diffusers/models/autoencoders/autoencoder_kl_flux2.py, autoencoder_kl.py
and vae.py; Apache License 2.0; Copyright 2026 The HuggingFace Team). See
THIRD_PARTY_NOTICES.md.

    post_quant_conv (1 x 1; none when use_post_quant_conv is false), conv_in
        (3 x 3, 32 -> 512)
    the mid block: a ResNet block, the attention of all pixels over all
        pixels (GroupNorm, q k v with bias, one head of 512, the output, + x),
        a ResNet block
    four up blocks of three ResNet blocks (GroupNorm, SiLU, conv, GroupNorm,
        SiLU, conv, + x or a 1 x 1 conv of x), the first three with a 2 x
        nearest upsampling and a 3 x 3 conv after them
    GroupNorm, SiLU, conv_out (3 x 3, 128 -> 3)

The rows are pixels (row major), the columns channels. A 3 x 3 conv is the
im2col of its 9 neighbours (zeros past the edge; the upsampling's pixel
(y, x) reads (y / 2, x / 2) of its input, in the same gather) by ENC_GATHER
into a buffer of TILE rows, then ENC_LINEAR of W (out, 3, 3, in), a tile of
rows at a time. The matrices: float32 on the CPU, the bfloat16 rows on the
GPU (KQ_BF16).

The encoder (an image's latents: a reference image of FLUX.2's editing) is
diffusers' Encoder as a program of the same pieces, its weights loaded at
its first use:

    conv_in (3 x 3, 3 -> 128); four down blocks of two ResNet blocks (128 256
        512 512 channels), the first three with Downsample2D after them (one
        row and column of zeros at the bottom and the right, a 3 x 3 conv of
        stride 2: its im2col reads pixel (2y + i, 2x + j))
    the mid block (a ResNet block, the attention, a ResNet block), GroupNorm,
        SiLU, conv_out (3 x 3, 512 -> 64), quant_conv (1 x 1; none when
        use_quant_conv is false): the posterior's mean and log variance, of
        which the mean is the latents (its mode)
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner
from reclank.formats.gguf import BF16
from reclank.formats.safetensors import SafeTensors
from reclank.models.qwen3.dense import _KQ
from reclank.runtime.threads import team_threads

TILE = 32768        # rows of an im2col at a time


def conv_index(h, w, up=False):
    """The im2col index (rows, 9) of a 3 x 3 conv with padding 1 over an
    image of h x w pixels (up: the 2 x nearest upsampling of it, (2h, 2w)
    pixels out, reading the input's pixel (y / 2, x / 2)); -1 past the edge."""
    H, W = (2 * h, 2 * w) if up else (h, w)
    y = np.arange(H)[:, None, None, None] + np.arange(-1, 2)[None, None, :, None]
    x = np.arange(W)[None, :, None, None] + np.arange(-1, 2)[None, None, None, :]
    ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
    if up:
        y, x = y // 2, x // 2
    idx = np.where(ok, y * w + x, -1)
    return idx.reshape(H * W, 9).astype(np.int32)


def down_index(h, w):
    """The im2col index (h/2 w/2, 9) of Downsample2D's conv (3 x 3, stride 2,
    no padding) over an image of h x w pixels (even) padded with a row and a
    column of zeros at the bottom and the right; -1 past the edge."""
    y = 2 * np.arange(h // 2)[:, None, None, None] + np.arange(3)[None, None, :, None]
    x = 2 * np.arange(w // 2)[None, :, None, None] + np.arange(3)[None, None, None, :]
    idx = np.where((y < h) & (x < w), y * w + x, -1)
    return idx.reshape((h // 2) * (w // 2), 9).astype(np.int32)


class Flux2VAE:
    """The decoder of a vae directory (config.json and its safetensors).
    decode(z) gives the image (3, 8h, 8w) in [-1, 1] of latents (latent
    channels, h, w); encode(img) the latents of an image (the encoder)."""

    def __init__(self, path, gpu=False, cpu_threads=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.W = W = SafeTensors(os.path.join(path, "diffusion_pytorch_model.safetensors"))
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.groups = cfg["norm_num_groups"]
        self.chans = list(reversed(cfg["block_out_channels"]))         # 512 512 256 128
        self.lpb = cfg["layers_per_block"] + 1
        self.latent = cfg["latent_channels"]
        self.bn_eps = float(cfg.get("batch_norm_eps", 0))
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        self.bn_mean, self.bn_var = (f("bn.running_mean"), f("bn.running_var")) if "bn.running_mean" in W.header \
            else (None, None)
        d = "decoder."
        self.pq = self.conv("post_quant_conv") if cfg.get("use_post_quant_conv", True) else None
        self.cin = self.conv(d + "conv_in")
        res = lambda p: dict(  # noqa: E731
            n1=(f(p + "norm1.weight"), f(p + "norm1.bias")), c1=self.conv(p + "conv1"),
            n2=(f(p + "norm2.weight"), f(p + "norm2.bias")), c2=self.conv(p + "conv2"),
            sc=self.conv(p + "conv_shortcut") if p + "conv_shortcut.weight" in W.header else None)
        m = d + "mid_block."
        a = m + "attentions.0."
        # the names of diffusers, or of its old checkpoints (SD's)
        lin = lambda n, old: self.conv(a + n if a + n + ".weight" in W.header else a + old)  # noqa: E731
        self.mid = dict(r0=res(m + "resnets.0."), r1=res(m + "resnets.1."),
                        gn=(f(a + "group_norm.weight"), f(a + "group_norm.bias")),
                        q=lin("to_q", "query"), k=lin("to_k", "key"), v=lin("to_v", "value"),
                        o=lin("to_out.0", "proj_attn"))
        self.up = []
        for i in range(len(self.chans)):
            p = d + "up_blocks.%d." % i
            ups = p + "upsamplers.0.conv"
            self.up.append(dict(res=[res(p + "resnets.%d." % j) for j in range(self.lpb)],
                                up=self.conv(ups) if ups + ".weight" in W.header else None))
        self.nout = (f(d + "conv_norm_out.weight"), f(d + "conv_norm_out.bias"))
        self.cout = self.conv(d + "conv_out")
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)
        self.enc_chans = list(cfg["block_out_channels"])        # 128 256 512 512
        self.enc_lpb = cfg["layers_per_block"]
        self.use_quant = cfg.get("use_quant_conv", True)
        self.enc = None
        # the GPU's bfloat16 products take K of a multiple of 8: conv_in's 3 x 3 x 3 inputs as 3 x 3 x 8
        # (the pixels' channels and the weights padded with zeros)
        self.enc_in = 8 if self.gpu else 3
        self.enc_runner = _Runner(self._build_enc, keep=2, keep_gpu=1)

    def conv(self, name, pad_in=None):
        """A conv (out, in, k, k) or a linear (out, in) as a linear of its
        im2col: W (out, k k in), the bias. pad_in: the conv's inputs padded
        with zero weights to that many."""
        W = self.W
        wn = name + ".weight"
        raw = W.get_bf16(wn) if W.dtype(wn) == "BF16" else W.get(wn)
        if pad_in is not None:
            raw = np.pad(raw, ((0, 0), (0, pad_in - raw.shape[1]), (0, 0), (0, 0)))
        if raw.ndim == 4:
            raw = raw.transpose(0, 2, 3, 1).reshape(raw.shape[0], -1)
        raw = np.ascontiguousarray(raw)
        if raw.dtype == np.uint16 and self.gpu:
            lin = _Plain(None)
            lin.kq = _KQ(raw.view(np.uint8), BF16, raw.shape[0], raw.shape[1])
        elif raw.dtype == np.uint16:
            lin = _Plain((raw.astype(np.uint32) << 16).view(np.float32))
        else:
            lin = _Plain(raw.astype(np.float32))
        lin.shape = raw.shape
        lin.b = np.ascontiguousarray(W.get(name + ".bias"), dtype=np.float32)
        return lin

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def program(self, h, w, gpu=False):
        """The program of latents of h x w: input "z" (h w, latent channels), output "img"
        (64 h w, 3)."""
        c = _EncCompiler(1e-6, pack=not gpu, rows=TILE, kmax=9 * max(self.chans))
        env = c.env
        G = self.groups
        levels = [(h << i, w << i) for i in range(len(self.chans))]
        # each level's buffers (roles a b c d) of its widest channels
        widest = [max(self.chans[0], self.latent)]
        for i, ch in enumerate(self.chans[1:], 1):
            widest.append(max(self.chans[i - 1], ch))
        # the buffers of each level's roles (a b c d), each the size of its
        # largest use: the views are named as the forms are made, and made
        # (with the pools) before the forms compile
        uses = {}           # name: (pool key, rows, cols)
        pools = {}
        env["col"] = np.zeros(TILE * 9 * max(widest), np.float32)
        env["z"] = np.zeros((h * w, self.latent), np.float32)
        H, Wd = levels[-1]
        env["img"] = np.zeros((H * Wd, 3), np.float32)
        n_names = [0]

        def view(arr, rows, cols):
            """a name of a view (rows, cols) of the start of a flat buffer"""
            n_names[0] += 1
            name = "_v%d" % n_names[0]
            env[name] = arr[:rows * cols].reshape(rows, cols)
            return name

        def buf(li, role, cols):
            hh, ww = levels[li]
            n_names[0] += 1
            name = "_v%d" % n_names[0]
            uses[name] = ((li, role), hh * ww, cols)
            return name

        forms = []
        rows_of = {}        # a tile's name: (the view's name, r0, r1)

        def conv3(x, li, cin, lin, out, up=False):
            """a 3 x 3 conv of x (level li, cin channels) into the name out"""
            hh, ww = levels[li]
            idx = conv_index(hh, ww, up)
            rows = idx.shape[0]
            for r0 in range(0, rows, TILE):
                r1 = min(rows, r0 + TILE)
                col = view(env["col"], r1 - r0, 9 * cin)
                n_names[0] += 1
                ot = "_v%d" % n_names[0]
                rows_of[ot] = (out, r0, r1)
                forms.extend([("set", col, ("enc_gather", x, np.ascontiguousarray(idx[r0:r1]), cin)),
                              ("set", ot, ("enc_linear", lin, col))])

        def resnet(x, li, cin, cout, R, free):
            """a ResNet block of x (level li); free: three roles to use. The
            name of its output."""
            ra, rb, rc = free
            n1 = buf(li, ra, cin)
            forms.append(("set", n1, ("enc_gnorm", x, R["n1"][0], R["n1"][1], G, 1e-6, True)))
            h1 = buf(li, rb, cout)
            conv3(n1, li, cin, R["c1"], h1)
            n2 = buf(li, ra, cout)
            forms.append(("set", n2, ("enc_gnorm", h1, R["n2"][0], R["n2"][1], G, 1e-6, True)))
            h2 = buf(li, rc, cout)
            conv3(n2, li, cout, R["c2"], h2)
            if R["sc"] is not None:
                s = buf(li, ra, cout)
                forms.append(("set", s, ("enc_linear", R["sc"], x)))
                forms.append(("enc_add", h2, s))
            else:
                forms.append(("enc_add", h2, x))
            return h2, rc

        # post_quant_conv, conv_in
        if self.pq is not None:
            x = buf(0, "a", self.latent)
            forms.append(("set", x, ("enc_linear", self.pq, "z")))
        else:
            x = "z"
        C = self.chans[0]
        y = buf(0, "b", C)
        conv3(x, 0, self.latent, self.cin, y)
        x, role = y, "b"

        def others(r):
            return [q for q in "abcd" if q != r]

        M = self.mid
        x, role = resnet(x, 0, C, C, M["r0"], others(role))
        # the attention of the mid block
        rows = levels[0][0] * levels[0][1]
        ra, rb, rc = others(role)
        nn_ = buf(0, ra, C)
        forms.append(("set", nn_, ("enc_gnorm", x, M["gn"][0], M["gn"][1], G, 1e-6, False)))
        q, k, v = buf(0, rb, C), buf(0, rc, C), "_attn_v"
        env["_attn_v"] = np.zeros((rows, C), np.float32)
        env["_attn_o"] = np.zeros((rows, C), np.float32)
        env["_scale"] = np.full(C, C ** -0.5, np.float32)
        env["_alo"], env["_ahi"] = np.zeros(rows, np.int32), np.full(rows, rows, np.int32)
        forms += [("set", q, ("enc_linear", M["q"], nn_)),
                  ("set", q, ("enc_mul_vec", q, "_scale")),
                  ("set", k, ("enc_linear", M["k"], nn_)),
                  ("set", v, ("enc_linear", M["v"], nn_)),
                  ("set", "_attn_o", ("enc_attn_range", q, k, v, "_alo", "_ahi", 1, 1, C)),
                  ("set", nn_, ("enc_linear", M["o"], "_attn_o")),
                  ("enc_add", x, nn_)]
        x, role = resnet(x, 0, C, C, M["r1"], others(role))
        cin = C
        for i, B in enumerate(self.up):
            cout = self.chans[i]
            for j, R in enumerate(B["res"]):
                x, role = resnet(x, i, cin if j == 0 else cout, cout, R, others(role))
            cin = cout
            if B["up"] is not None:
                nr = others(role)[0]
                y = buf(i + 1, nr, cout)
                conv3(x, i, cout, B["up"], y, up=True)
                x, role = y, nr
        li = len(levels) - 1
        nn_ = buf(li, others(role)[0], cin)
        forms.append(("set", nn_, ("enc_gnorm", x, self.nout[0], self.nout[1], G, 1e-6, True)))
        conv3(nn_, li, cin, self.cout, "img")
        for name, (key, rows, cols) in uses.items():
            pools[key] = max(pools.get(key, 0), rows * cols)
        pools = {k: np.zeros(v, np.float32) for k, v in pools.items()}
        for name, (key, rows, cols) in uses.items():
            env[name] = pools[key][:rows * cols].reshape(rows, cols)
        for name, (of, r0, r1) in rows_of.items():
            env[name] = env[of][r0:r1]
        c.compile(("seq", *forms))
        return c.p.finish()

    # ---- the encoder --------------------------------------------------------

    def encoder(self):
        """The encoder's weights (loaded at the first use)."""
        if self.enc is not None:
            return self.enc
        f = lambda n: np.ascontiguousarray(self.W.get(n), dtype=np.float32)  # noqa: E731
        H = self.W.header
        d = "encoder."
        res = lambda p: dict(  # noqa: E731
            n1=(f(p + "norm1.weight"), f(p + "norm1.bias")), c1=self.conv(p + "conv1"),
            n2=(f(p + "norm2.weight"), f(p + "norm2.bias")), c2=self.conv(p + "conv2"),
            sc=self.conv(p + "conv_shortcut") if p + "conv_shortcut.weight" in H else None)
        m = d + "mid_block."
        a = m + "attentions.0."
        lin = lambda n, old: self.conv(a + n if a + n + ".weight" in H else a + old)  # noqa: E731
        down = []
        for i in range(len(self.enc_chans)):
            p = d + "down_blocks.%d." % i
            ds = p + "downsamplers.0.conv"
            down.append(dict(res=[res(p + "resnets.%d." % j) for j in range(self.enc_lpb)],
                             down=self.conv(ds) if ds + ".weight" in H else None))
        self.enc = dict(
            cin=self.conv(d + "conv_in", pad_in=self.enc_in), down=down,
            mid=dict(r0=res(m + "resnets.0."), r1=res(m + "resnets.1."),
                     gn=(f(a + "group_norm.weight"), f(a + "group_norm.bias")),
                     q=lin("to_q", "query"), k=lin("to_k", "key"), v=lin("to_v", "value"),
                     o=lin("to_out.0", "proj_attn")),
            nout=(f(d + "conv_norm_out.weight"), f(d + "conv_norm_out.bias")), cout=self.conv(d + "conv_out"),
            quant=self.conv("quant_conv") if self.use_quant else None)
        return self.enc

    def _build_enc(self, key, gpu):
        p = self.encoder_program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def encoder_program(self, h, w, gpu=False):
        """The program of an image of h x w pixels (multiples of 8): input "x"
        (h w, 3; enc_in, the rest zeros), output "moments" (h w / 64, 2 latent channels)."""
        E = self.encoder()
        chans = self.enc_chans
        c = _EncCompiler(1e-6, pack=not gpu, rows=TILE, kmax=9 * max(chans))
        env = c.env
        G = self.groups
        levels = [(h >> i, w >> i) for i in range(len(chans))]
        uses = {}           # name: (pool key, rows, cols)
        pools = {}
        rows_of = {}        # a tile's name: (the view's name, r0, r1)
        env["col"] = np.zeros(TILE * 9 * max(chans), np.float32)
        env["x"] = np.zeros((h * w, self.enc_in), np.float32)
        hl, wl = levels[-1]
        env["moments"] = np.zeros((hl * wl, 2 * self.latent), np.float32)
        n_names = [0]
        forms = []

        def name():
            n_names[0] += 1
            return "_v%d" % n_names[0]

        def buf(li, role, cols):
            hh, ww = levels[li]
            n = name()
            uses[n] = ((li, role), hh * ww, cols)
            return n

        def conv3(x, cin, lin, out, idx):
            """a 3 x 3 conv of x (cin channels) by its im2col index into the name out"""
            rows = idx.shape[0]
            for r0 in range(0, rows, TILE):
                r1 = min(rows, r0 + TILE)
                col = name()
                env[col] = env["col"][:(r1 - r0) * 9 * cin].reshape(r1 - r0, 9 * cin)
                ot = name()
                rows_of[ot] = (out, r0, r1)
                forms.extend([("set", col, ("enc_gather", x, np.ascontiguousarray(idx[r0:r1]), cin)),
                              ("set", ot, ("enc_linear", lin, col))])

        def resnet(x, li, cin, cout, R, free):
            """a ResNet block of x (level li); free: three roles to use. The
            name of its output, its role."""
            ra, rb, rc = free
            idx = conv_index(*levels[li])
            n1 = buf(li, ra, cin)
            forms.append(("set", n1, ("enc_gnorm", x, R["n1"][0], R["n1"][1], G, 1e-6, True)))
            h1 = buf(li, rb, cout)
            conv3(n1, cin, R["c1"], h1, idx)
            n2 = buf(li, ra, cout)
            forms.append(("set", n2, ("enc_gnorm", h1, R["n2"][0], R["n2"][1], G, 1e-6, True)))
            h2 = buf(li, rc, cout)
            conv3(n2, cout, R["c2"], h2, idx)
            if R["sc"] is not None:
                s = buf(li, ra, cout)
                forms.append(("set", s, ("enc_linear", R["sc"], x)))
                forms.append(("enc_add", h2, s))
            else:
                forms.append(("enc_add", h2, x))
            return h2, rc

        def others(r):
            return [q for q in "abcd" if q != r]

        x, role = buf(0, "b", chans[0]), "b"
        conv3("x", self.enc_in, E["cin"], x, conv_index(h, w))
        cin = chans[0]
        for i, B in enumerate(E["down"]):
            cout = chans[i]
            for j, R in enumerate(B["res"]):
                x, role = resnet(x, i, cin if j == 0 else cout, cout, R, others(role))
            cin = cout
            if B["down"] is not None:
                nr = others(role)[0]
                y = buf(i + 1, nr, cout)
                conv3(x, cout, B["down"], y, down_index(*levels[i]))
                x, role = y, nr
        li = len(levels) - 1
        M = E["mid"]
        C = cin
        x, role = resnet(x, li, C, C, M["r0"], others(role))
        rows = hl * wl
        ra, rb, rc = others(role)
        nn_ = buf(li, ra, C)
        forms.append(("set", nn_, ("enc_gnorm", x, M["gn"][0], M["gn"][1], G, 1e-6, False)))
        q, k, v = buf(li, rb, C), buf(li, rc, C), "_attn_v"
        env["_attn_v"] = np.zeros((rows, C), np.float32)
        env["_attn_o"] = np.zeros((rows, C), np.float32)
        env["_scale"] = np.full(C, C ** -0.5, np.float32)
        env["_alo"], env["_ahi"] = np.zeros(rows, np.int32), np.full(rows, rows, np.int32)
        forms += [("set", q, ("enc_linear", M["q"], nn_)),
                  ("set", q, ("enc_mul_vec", q, "_scale")),
                  ("set", k, ("enc_linear", M["k"], nn_)),
                  ("set", v, ("enc_linear", M["v"], nn_)),
                  ("set", "_attn_o", ("enc_attn_range", q, k, v, "_alo", "_ahi", 1, 1, C)),
                  ("set", nn_, ("enc_linear", M["o"], "_attn_o")),
                  ("enc_add", x, nn_)]
        x, role = resnet(x, li, C, C, M["r1"], others(role))
        nn_ = buf(li, others(role)[0], C)
        forms.append(("set", nn_, ("enc_gnorm", x, E["nout"][0], E["nout"][1], G, 1e-6, True)))
        if E["quant"] is None:
            conv3(nn_, C, E["cout"], "moments", conv_index(hl, wl))
        else:
            mo = buf(li, others(role)[1], 2 * self.latent)
            conv3(nn_, C, E["cout"], mo, conv_index(hl, wl))
            forms.append(("set", "moments", ("enc_linear", E["quant"], mo)))
        for n, (key, r, cols) in uses.items():
            pools[key] = max(pools.get(key, 0), r * cols)
        pools = {k_: np.zeros(v_, np.float32) for k_, v_ in pools.items()}
        for n, (key, r, cols) in uses.items():
            env[n] = pools[key][:r * cols].reshape(r, cols)
        for n, (of, r0, r1) in rows_of.items():
            env[n] = env[of][r0:r1]
        c.compile(("seq", *forms))
        return c.p.finish()

    def encode(self, img):
        """The latents (latent channels, h / 8, w / 8) of an image (3, h, w) in
        [-1, 1]: the posterior's mean (its mode: diffusers' sample_mode
        "argmax")."""
        _, h, w = img.shape
        prog, g = self.enc_runner.get((h, w), self.gpu)
        prog.names["x"][:, :3] = img.reshape(3, h * w).T
        if g is not None:
            g.upload("x")
            g.run()
            g.download("moments")
        else:
            prog.run()
        f = 1 << (len(self.enc_chans) - 1)
        return np.ascontiguousarray(prog.names["moments"][:, :self.latent].T).reshape(self.latent, h // f, w // f)

    def normalize(self, z):
        """The DiT's packed latents (h/2 w/2, 128) of the VAE's latents (32, h,
        w): the 2 x 2 patches packed (channel c 4 + p), the batch norm applied
        ((x - mean) / std, std = sqrt(var + eps) in float32)."""
        C, h, w = z.shape
        x = z.reshape(C, h // 2, 2, w // 2, 2).transpose(1, 3, 0, 2, 4).reshape((h // 2) * (w // 2), 4 * C)
        std = np.sqrt(self.bn_var + np.float32(self.bn_eps)).astype(np.float32)
        return ((x.astype(np.float32) - self.bn_mean) / std).astype(np.float32)

    def unnormalize(self, packed, hh, ww):
        """The VAE's latents (32, 2 hh, 2 ww) of the DiT's packed latents (hh
        ww, 128): the batch norm undone (x std + mean, std = sqrt(var + eps)
        in float32), the 2 x 2 patches unpacked."""
        std = np.sqrt(self.bn_var + np.float32(self.bn_eps)).astype(np.float32)
        x = (packed * std + self.bn_mean).astype(np.float32)               # (hh ww, 128): channel c 4 + p
        x = x.reshape(hh, ww, self.latent, 2, 2).transpose(2, 0, 3, 1, 4)
        return np.ascontiguousarray(x.reshape(self.latent, 2 * hh, 2 * ww))

    def decode(self, z):
        """The image (3, 8h, 8w) of latents z (latent channels, h, w)."""
        _, h, w = z.shape
        prog, g = self.runner.get((h, w), self.gpu)
        prog.names["z"][...] = z.reshape(self.latent, h * w).T
        if g is not None:
            g.upload("z")
            g.run()
            g.download("img")
        else:
            prog.run()
        return prog.names["img"].T.reshape(3, 8 * h, 8 * w).copy()
