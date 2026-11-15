"""SeedVR2's VAE (video_vae_v3, s8_c16_t4_inflation_sd3: Stable Diffusion 3's AutoencoderKL of
128 256 512 512 channels, 2 ResNet blocks a down block and 3 an up block, GroupNorm of 32 groups
(eps 1e-6), 16 latent channels, inflated to causal 3 x 3 x 3 convs with 4 x temporal compression)
as programs. This file is the image path: one frame.

Ported from SeedVR (ByteDance-Seed/SeedVR, models/video_vae_v3; Apache License 2.0; Copyright (c)
2023 HuggingFace Team; Copyright (c) 2025 ByteDance Ltd. and/or its affiliates). See
THIRD_PARTY_NOTICES.md.

A causal conv pads time with copies of the first frame, so on one frame each 3 x 3 x 3 kernel sees
that frame three times: it is the 2D conv of its kernel summed over time (summed in float32,
(k0 + k1) + k2: the same values, the sums in another order); a temporal stride-2 downsampler on
one frame is again that 2D conv, and the decoder's temporal upsampling keeps its first phase
(remove_head drops the copy). So:

    encoder: conv_in (3 x 3, 3 -> 128); four down blocks of 2 ResNet blocks (GroupNorm, SiLU,
        conv, GroupNorm, SiLU, conv, + x or a 1 x 1 conv of x), the first three with a stride-2
        3 x 3 conv after SD's (0, 1, 0, 1) zero padding; the mid block (a ResNet block, the
        attention of all pixels: GroupNorm, q k v with bias, one head of 512, the output, + x; a
        ResNet block); GroupNorm, SiLU, conv_out (3 x 3, 512 -> 32: the posterior's mean and
        logvar)
    decoder: conv_in (3 x 3, 16 -> 512); the mid block; four up blocks of 3 ResNet blocks, the
        first three with the MAGViT upsampler: a 1 x 1 conv to 4 C (the time phase 0 of the
        8 C of a temporal one), its (x, y) blocks of C the 2 x 2 output pixels (the rearrange
        "b (x y z c) f h w -> b c (f z) (h x) (w y)"), read by the next 3 x 3 conv's im2col
        directly; GroupNorm, SiLU, conv_out (3 x 3, 128 -> 3)

The rows are pixels (row major), the columns channels. A 3 x 3 conv is the im2col of its 9
neighbours (ENC_GATHER, -1 past the edges) in tiles of TILE rows and ENC_LINEAR of W (out, 9
in). GroupNorm (ENC_GNORM, SiLU fused) of the frame's pixels; the attention ENC_ATTN_WIDE on the
GPU (ENC_ATTN_RANGE on the CPU), q times 512^-1/2 first. The weights the checkpoint's float32 as
stored (CPU and GPU). The tensors of a program share one arena, placed by their lifetimes."""
from __future__ import annotations

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner
from reclank.formats.safetensors import drop_pages
from reclank.runtime.threads import team_threads

TILE = 32768        # rows of an im2col at a time
GROUPS, EPS = 32, 1e-6
SCALE = 0.9152      # the latent's scaling_factor (shifting 0)


def conv_index(h, w, src=None):
    """The im2col index (h w, 9) of a 3 x 3 conv with padding 1 over an h x w image; -1 past the
    edge. src (y, x) -> the row of the input pixel (default y w + x)."""
    y = np.arange(h)[:, None, None, None] + np.arange(-1, 2)[None, None, :, None]
    x = np.arange(w)[None, :, None, None] + np.arange(-1, 2)[None, None, None, :]
    ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    yc, xc = np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)
    rows = yc * w + xc if src is None else src(yc, xc)
    return np.where(ok, rows, -1).reshape(h * w, 9).astype(np.int32)


def up_index(h, w):
    """The im2col of a 3 x 3 conv over the 2h x 2w output of a MAGViT upsampler whose 1 x 1 conv
    gave (h w, 4 C) as (h w 4, C): output pixel (Y, X) is row (Y / 2 w + X / 2) 4 + 2 (Y % 2) +
    X % 2."""
    return conv_index(2 * h, 2 * w, lambda y, x: ((y // 2) * w + x // 2) * 4 + 2 * (y % 2) + x % 2)


def down_index(h, w):
    """The im2col (h/2 w/2, 9) of a 3 x 3 stride-2 conv with no padding over an h x w image padded
    with a zero row and column at its bottom and right (SD's (0, 1, 0, 1))."""
    ho, wo = h // 2, w // 2
    y = 2 * np.arange(ho)[:, None, None, None] + np.arange(3)[None, None, :, None]
    x = 2 * np.arange(wo)[None, :, None, None] + np.arange(3)[None, None, None, :]
    ok = (y < h) & (x < w)
    return np.where(ok, np.minimum(y, h - 1) * w + np.minimum(x, w - 1), -1).reshape(ho * wo, 9).astype(np.int32)


class _Arena:
    """The tensors of a program in one float32 arena: each (rows, cols) from its first use to its
    last (by the names in the forms), placed first-fit."""

    ALIGN = 64

    def __init__(self):
        self.shapes = {}
        self.alias = {}         # a tile's name: its tensor's

    def new(self, name, rows, cols):
        self.shapes[name] = (rows, cols)
        return name

    def place(self, forms, env, pool=None):
        """Place the tensors (env gets their views); pool: an arena to use when it is large enough
        (the programs of a size share one). Return the arena."""
        first, last = {}, {}

        def walk(x, i):
            if isinstance(x, str):
                n = x
                while n in self.alias:
                    n = self.alias[n]
                if n in self.shapes:
                    first.setdefault(n, i)
                    last[n] = i
            elif isinstance(x, tuple):
                for y in x:
                    walk(y, i)
        for i, f in enumerate(forms):
            walk(f, i)
        size = lambda n: -(-self.shapes[n][0] * self.shapes[n][1] // self.ALIGN) * self.ALIGN  # noqa: E731
        live, offset = [], {}            # (start, end, name)
        top = 0
        for n in sorted(first, key=lambda n: first[n]):
            live = [s for s in live if last[s[2]] >= first[n]]
            need = size(n)
            pos = 0
            for s0, s1, _ in sorted(live):
                if s0 - pos >= need:
                    break
                pos = max(pos, s1)
            offset[n] = pos
            live.append((pos, pos + need, n))
            top = max(top, pos + need)
        arena = pool if pool is not None and pool.size >= top else np.zeros(max(top, 1), np.float32)
        for n, o in offset.items():
            r, c = self.shapes[n]
            env[n] = arena[o:o + r * c].reshape(r, c)
        return arena


class SeedVR2VAE:
    """The VAE of ema_vae.safetensors (a SafeTensors or a dict of arrays). encode(x) gives the
    posterior's (mean, logvar) (16, H/8, W/8) of an image x (3, H, W) in [-1, 1]; decode(z) the
    image (3, 8h, 8w) of latents z (16, h, w) (already divided by SCALE)."""

    def __init__(self, W, gpu=False, cpu_threads=None):
        self.W = W
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        e, d = "encoder.", "decoder."
        g = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        res = lambda p: dict(  # noqa: E731
            n1=(g(p + "norm1.weight"), g(p + "norm1.bias")), c1=self.conv(p + "conv1"),
            n2=(g(p + "norm2.weight"), g(p + "norm2.bias")), c2=self.conv(p + "conv2"),
            sc=self.conv(p + "conv_shortcut") if p + "conv_shortcut.weight" in W.header else None)
        attn = lambda a: dict(  # noqa: E731
            gn=(g(a + "group_norm.weight"), g(a + "group_norm.bias")), q=self.conv(a + "to_q"),
            k=self.conv(a + "to_k"), v=self.conv(a + "to_v"), o=self.conv(a + "to_out.0"))
        self.enc = dict(cin=self.conv(e + "conv_in"),
                        down=[dict(res=[res(e + "down_blocks.%d.resnets.%d." % (i, j)) for j in range(2)],
                                   down=self.conv(e + "down_blocks.%d.downsamplers.0.conv" % i) if i < 3 else None)
                              for i in range(4)],
                        mid=dict(r0=res(e + "mid_block.resnets.0."), a=attn(e + "mid_block.attentions.0."),
                                 r1=res(e + "mid_block.resnets.1.")),
                        nout=(g(e + "conv_norm_out.weight"), g(e + "conv_norm_out.bias")),
                        cout=self.conv(e + "conv_out"))
        up = []
        for i in range(4):
            p = d + "up_blocks.%d." % i
            B = dict(res=[res(p + "resnets.%d." % j) for j in range(3)], ups=None, upc=None)
            if i < 3:
                B["ups"] = self.upscale(p + "upsamplers.0.upscale_conv", temporal=i < 2)
                B["upc"] = self.conv(p + "upsamplers.0.conv")
            up.append(B)
        self.dec = dict(cin=self.conv(d + "conv_in"),
                        mid=dict(r0=res(d + "mid_block.resnets.0."), a=attn(d + "mid_block.attentions.0."),
                                 r1=res(d + "mid_block.resnets.1.")),
                        up=up, nout=(g(d + "conv_norm_out.weight"), g(d + "conv_norm_out.bias")),
                        cout=self.conv(d + "conv_out"))
        # the decoder's programs (runner) and the encoder's (enc_runner): each stays on the GPU between
        # the calls of a request, RESIDENCY's to evict
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)
        self.enc_runner = _Runner(self._build, keep=2, keep_gpu=1)

    # ---- the weights ----------------------------------------------------------------------------

    def conv(self, name):
        """A conv (out, in, kt, kh, kw) or a linear as the linear of its im2col: W (out, kh kw in) of
        the kernel summed over time (one frame), the bias."""
        src = self.W.get(name + ".weight")
        raw = np.asarray(src, dtype=np.float32)
        if raw.ndim == 5:
            k = raw[:, :, 0]
            for t in range(1, raw.shape[2]):
                k = k + raw[:, :, t]
            raw = k
        if raw.ndim == 4:
            raw = raw.transpose(0, 2, 3, 1).reshape(raw.shape[0], -1)
        lin = _Plain(np.ascontiguousarray(raw, dtype=np.float32))
        lin.shape = lin.w.shape
        lin.b = np.ascontiguousarray(self.W.get(name + ".bias"), dtype=np.float32)
        drop_pages(src)
        return lin

    def upscale(self, name, temporal, phase=0):
        """The MAGViT upsampler's 1 x 1 conv (C x 4 z, C): the rows of its time phase (0: the image's),
        in the order (x, y, c) of the 2 x 2 output pixels."""
        lin = self.conv(name)
        z = 2 if temporal else 1
        C = lin.shape[0] // (4 * z)
        rows = np.array([((xy * z) + phase) * C + c for xy in range(4) for c in range(C)])
        out = _Plain(np.ascontiguousarray(lin.w[rows]))
        out.shape = out.w.shape
        out.b = np.ascontiguousarray(lin.b[rows])
        return out

    def _build(self, key, gpu):
        p = self.encoder_program(*key[1:], gpu=gpu) if key[0] == "enc" else self.decoder_program(*key[1:], gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    # ---- the programs ---------------------------------------------------------------------------

    class _Builder:
        """The forms of a program, its tensors in an arena."""

        def __init__(self, gpu):
            self.gpu = gpu
            self.c = _EncCompiler(EPS, pack=not gpu)
            self.env = self.c.env
            self.ar = _Arena()
            self.forms = []
            self.n = 0
            self.index = {}
            self.col_cols = 0

        def name(self):
            self.n += 1
            return "_v%d" % self.n

        def tensor(self, rows, cols):
            return self.ar.new(self.name(), rows, cols)

        def idx(self, key, make):
            """an index array, one per (kind, size): its tiles views of it"""
            if key not in self.index:
                self.index[key] = make()
            return self.index[key]

        def conv(self, x, cin, lin, idx, rows):
            """the conv of the im2col idx (rows, 9) of x (cin channels): a new tensor"""
            out = self.tensor(rows, lin.shape[0])
            self.col_cols = max(self.col_cols, 9 * cin)
            for r0 in range(0, rows, TILE):
                r1 = min(rows, r0 + TILE)
                col, ot, it = self.name(), self.name(), self.name()
                self.env[it] = idx[r0:r1]
                self.tiles.append((col, r1 - r0, 9 * cin))
                self.ar.alias[ot] = out
                self.views.append((ot, out, r0, r1))
                self.forms += [("set", col, ("enc_gather", x, it, cin)), ("set", ot, ("enc_linear", lin, col))]
            return out

        def linear(self, x, lin, rows):
            out = self.tensor(rows, lin.shape[0])
            self.forms.append(("set", out, ("enc_linear", lin, x)))
            return out

        def gnorm(self, x, gn, rows, cols, silu=True):
            out = self.tensor(rows, cols)
            self.forms.append(("set", out, ("enc_gnorm", x, gn[0], gn[1], GROUPS, EPS, silu)))
            return out

        def resnet(self, x, R, h, w, cin, cout, ix):
            rows = h * w
            n1 = self.gnorm(x, R["n1"], rows, cin)
            h1 = self.conv(n1, cin, R["c1"], ix, rows)
            n2 = self.gnorm(h1, R["n2"], rows, cout)
            h2 = self.conv(n2, cout, R["c2"], ix, rows)
            s = self.linear(x, R["sc"], rows) if R["sc"] is not None else x
            self.forms.append(("enc_add", h2, s))
            return h2

        def attention(self, x, A, rows, C):
            nn_ = self.gnorm(x, A["gn"], rows, C, silu=False)
            q, k, v = (self.linear(nn_, A[n], rows) for n in "qkv")
            self.env["_scale"] = np.full(C, C ** -0.5, np.float32)
            self.forms.append(("set", q, ("enc_mul_vec", q, "_scale")))
            o = self.tensor(rows, C)
            if self.gpu:
                self.forms.append(("set", o, ("enc_attn_wide", q, k, v)))
            else:
                self.env["_alo"], self.env["_ahi"] = np.zeros(rows, np.int32), np.full(rows, rows, np.int32)
                self.forms.append(("set", o, ("enc_attn_range", q, k, v, "_alo", "_ahi", 1, 1, C)))
            y = self.linear(o, A["o"], rows)
            self.forms.append(("enc_add", y, x))
            return y

        def finish(self, inputs, outputs):
            """inputs, outputs: name: (rows, cols) buffers of their own (not in the arena)"""
            for n, (r, cc) in {**inputs, **outputs}.items():
                self.env[n] = np.zeros((r, cc), np.float32)
            self.arena = self.ar.place(self.forms, self.env)
            colbuf = np.zeros(TILE * max(self.col_cols, 1), np.float32)
            for col, r, cc in self.tiles:
                self.env[col] = colbuf[:r * cc].reshape(r, cc)
            for ot, out, r0, r1 in self.views:
                self.env[ot] = self.env[out][r0:r1]
            self.c.compile(("seq", *self.forms))
            return self.c.p.finish()

    def _new(self, gpu):
        b = self._Builder(gpu)
        b.tiles, b.views = [], []
        return b

    def encoder_program(self, H, W, gpu=False):
        """Input "img" (H W, 3), output "post" (H/8 W/8, 32): the posterior's mean, then logvar."""
        assert H % 8 == 0 and W % 8 == 0
        b = self._new(gpu)
        E = self.enc
        h, w = H, W
        x = b.conv("img", 3, E["cin"], b.idx(("c", h, w), lambda: conv_index(h, w)), h * w)
        chans = [128, 256, 512, 512]
        cin = 128
        for i, B in enumerate(E["down"]):
            ix = b.idx(("c", h, w), lambda: conv_index(h, w))
            for j, R in enumerate(B["res"]):
                x = b.resnet(x, R, h, w, cin, chans[i], ix)
                cin = chans[i]
            if B["down"] is not None:
                hh, ww = h, w
                x = b.conv(x, cin, B["down"], b.idx(("d", hh, ww), lambda: down_index(hh, ww)), (h // 2) * (w // 2))
                h, w = h // 2, w // 2
        ix = b.idx(("c", h, w), lambda: conv_index(h, w))
        M = E["mid"]
        x = b.resnet(x, M["r0"], h, w, 512, 512, ix)
        x = b.attention(x, M["a"], h * w, 512)
        x = b.resnet(x, M["r1"], h, w, 512, 512, ix)
        n = b.gnorm(x, E["nout"], h * w, 512)
        y = b.conv(n, 512, E["cout"], ix, h * w)
        b.forms.append(("set", "post", ("copy", y)))
        return b.finish({"img": (H * W, 3)}, {"post": (h * w, 32)})

    def decoder_program(self, h, w, gpu=False):
        """Input "z" (h w, 16), output "out" (8h 8w, 3)."""
        b = self._new(gpu)
        Dd = self.dec
        ix = b.idx(("c", h, w), lambda: conv_index(h, w))
        x = b.conv("z", 16, Dd["cin"], ix, h * w)
        M = Dd["mid"]
        x = b.resnet(x, M["r0"], h, w, 512, 512, ix)
        x = b.attention(x, M["a"], h * w, 512)
        x = b.resnet(x, M["r1"], h, w, 512, 512, ix)
        chans = [512, 512, 256, 128]
        cin = 512
        for i, B in enumerate(Dd["up"]):
            ix = b.idx(("c", h, w), lambda: conv_index(h, w))
            for j, R in enumerate(B["res"]):
                x = b.resnet(x, R, h, w, cin, chans[i], ix)
                cin = chans[i]
            if B["ups"] is not None:
                u = b.linear(x, B["ups"], h * w)                     # (h w, 4 C) = (h w 4, C)
                C = cin
                uv = b.name()
                b.ar.alias[uv] = u
                b.views.append(("_r", uv, u, C))
                hh, ww = h, w
                x = b.conv(uv, C, B["upc"], b.idx(("u", hh, ww), lambda: up_index(hh, ww)), 4 * h * w)
                h, w = 2 * h, 2 * w
        ix = b.idx(("c", h, w), lambda: conv_index(h, w))
        n = b.gnorm(x, Dd["nout"], h * w, cin)
        y = b.conv(n, cin, Dd["cout"], ix, h * w)
        b.forms.append(("set", "out", ("copy", y)))
        return self._finish_views(b, {"z": (h * w // 64, 16)}, {"out": (h * w, 3)})

    @staticmethod
    def _finish_views(b, inputs, outputs):
        """finish, with the reshaped views ("_r": a tensor (rows, k C) read as (rows k, C))"""
        plain = [v for v in b.views if v[0] != "_r"]
        reshaped = [v for v in b.views if v[0] == "_r"]
        b.views = plain
        for n, (r, cc) in {**inputs, **outputs}.items():
            b.env[n] = np.zeros((r, cc), np.float32)
        b.arena = b.ar.place(b.forms, b.env)
        for _, uv, u, C in reshaped:
            b.env[uv] = b.env[u].reshape(-1, C)
        colbuf = np.zeros(TILE * max(b.col_cols, 1), np.float32)
        for col, r, cc in b.tiles:
            b.env[col] = colbuf[:r * cc].reshape(r, cc)
        for ot, out, r0, r1 in b.views:
            b.env[ot] = b.env[out][r0:r1]
        b.c.compile(("seq", *b.forms))
        return b.c.p.finish()

    # ---- the calls -------------------------------------------------------------------------------

    def _run(self, key, inputs, output):
        prog, g = (self.enc_runner if key[0] == "enc" else self.runner).get(key, self.gpu)
        for n, a in inputs.items():
            prog.names[n][...] = a
            if g is not None:
                g.upload(n)
        if g is not None:
            g.run()
            g.download(output)
        else:
            prog.run()
        return prog.names[output]

    def encode(self, img):
        """(mean, logvar) (16, H/8, W/8) of an image (3, H, W) in [-1, 1]; logvar clamped to
        [-30, 20] as diffusers' DiagonalGaussianDistribution."""
        _, H, W = img.shape
        post = self._run(("enc", H, W), {"img": img.reshape(3, -1).T}, "post")
        h, w = H // 8, W // 8
        p = post.T.reshape(32, h, w)
        return p[:16].copy(), np.clip(p[16:], -30.0, 20.0).astype(np.float32)

    def decode(self, z):
        """The image (3, 8h, 8w) of latents z (16, h, w) (the DiT's x0 / SCALE)."""
        _, h, w = z.shape
        out = self._run(("dec", h, w), {"z": z.reshape(16, -1).T}, "out")
        return out.T.reshape(3, 8 * h, 8 * w).copy()

    @staticmethod
    def sample(mean, logvar, noise):
        """The posterior's sample: mean + exp(logvar / 2) noise (float32)."""
        return (mean + np.exp(np.float32(0.5) * logvar).astype(np.float32) * noise).astype(np.float32)
