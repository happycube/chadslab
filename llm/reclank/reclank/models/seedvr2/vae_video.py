"""SeedVR2's VAE on video (vae.py has the image path and the layers): the causal 3 x 3 x 3 convs
frame by frame, each with a cache of the last two frames of its input.

Ported from SeedVR (ByteDance-Seed/SeedVR, models/video_vae_v3; Apache License 2.0; Copyright (c)
2023 HuggingFace Team; Copyright (c) 2025 ByteDance Ltd. and/or its affiliates). See
THIRD_PARTY_NOTICES.md.

A causal conv's output frame t reads its input frames t - 2, t - 1, t (the first frame copied
twice before frame 0: extend_head). Each causal conv keeps a buffer of two frames, [t - 2, t - 1]
(its "cache"): the conv is the product of its kernel's first two time taps with the im2col of the
cache (18 neighbours) plus the last tap's with the im2col of the frame (9), then the cache moves
on ([t - 1, t]); the first frame fills the cache with two copies of itself first. (The two
products sum in another order than one 27-tap product: float32 rounding apart.) GroupNorm and the
mid attention are per frame. So a frame needs only its own buffers and the caches, and the output
equals the whole clip at once (the official code's causal slices: the same values).

    decoder: a program for the first latent frame (one video frame) and one for each later one
        (four): the latent rate (conv_in, the mid block, up block 0's ResNet blocks); up block 0's
        upsampler: the 1 x 1 conv's rows of each time phase (the first latent frame: phase 0
        only, remove_head) to the frames of level 1, each through the upsampler's conv and up
        block 1's ResNet blocks, its upsampler the same to level 2 (the video's rate): up block 1's
        conv, up block 2's ResNet blocks and its spatial upsampler, up block 3, conv_out
    encoder: a program for the first frame and one for each later four: each frame through
        conv_in, down block 0 (its downsampler a 2D stride-2 conv: kernel 1 in time) and down block
        1's ResNet blocks; down block 1's downsampler (3 x 3 x 3 of stride 2 in time and space)
        reads the cache and the frame at the even frames (frame 2k from frames 2k - 2 .. 2k: the
        padded stride), only moves its cache at the odd ones; its output frames through down
        block 2, whose downsampler does the same; its outputs (the latent rate) through down block
        3, the mid block, conv_out

The caches are buffers of the VAE (both programs of a size share them); a program's other
tensors live in an arena by their lifetimes (vae._Arena), one arena for both programs of a size.
Memory (float32, the caches): the decoder 15.9 GB at 1280 x 720, the encoder 7.0 GB; 1664 x 960
needs 27.6 GB of decoder caches (the pinned-host phase)."""
from __future__ import annotations

import numpy as np

from reclank.encoders.runtime import _Plain, _Runner
from reclank.formats.safetensors import drop_pages
from reclank.models.seedvr2.vae import TILE, SeedVR2VAE, conv_index, down_index, up_index
from reclank.runtime.threads import team_threads


class SeedVR2VideoVAE:
    """The video path of the VAE of ema_vae.safetensors (a SafeTensors or a dict of arrays). encode(x) gives
    the posterior (mean, logvar) (16, T', H/8, W/8) of frames x (3, T, H, W) in [-1, 1] (T = 4k +
    1); decode(z) the frames (3, 4 (T' - 1) + 1, 8h, 8w) of latents z (16, T', h, w)."""

    def __init__(self, W, gpu=False, cpu_threads=None, image=None):
        self.W = W
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        # the image path's 2D weights (1 x 1 convs, the norms, the attention, the 2D downsampler)
        self.img = image if image is not None else SeedVR2VAE(W, gpu=gpu, cpu_threads=cpu_threads)
        e, d = "encoder.", "decoder."
        res = lambda p, R: dict(R, c1=self.causal(p + "conv1"), c2=self.causal(p + "conv2"))  # noqa: E731
        E, Dd = self.img.enc, self.img.dec
        self.enc = dict(cin=self.causal(e + "conv_in"),
                        down=[dict(res=[res(e + "down_blocks.%d.resnets.%d." % (i, j), B["res"][j]) for j in range(2)],
                                   down=(B["down"] if i == 0 else self.causal(e + "down_blocks.%d.downsamplers.0.conv" % i))
                                   if i < 3 else None) for i, B in enumerate(E["down"])],
                        mid=dict(r0=res(e + "mid_block.resnets.0.", E["mid"]["r0"]), a=E["mid"]["a"],
                                 r1=res(e + "mid_block.resnets.1.", E["mid"]["r1"])),
                        nout=E["nout"], cout=self.causal(e + "conv_out"))
        up = []
        for i, B in enumerate(Dd["up"]):
            p = d + "up_blocks.%d." % i
            U = dict(res=[res(p + "resnets.%d." % j, B["res"][j]) for j in range(3)], ups=None, upc=None)
            if i < 3:
                U["ups"] = [self.img.upscale(p + "upsamplers.0.upscale_conv", temporal=i < 2, phase=z)
                            for z in range(2 if i < 2 else 1)]
                U["upc"] = self.causal(p + "upsamplers.0.conv")
            up.append(U)
        self.dec = dict(cin=self.causal(d + "conv_in"),
                        mid=dict(r0=res(d + "mid_block.resnets.0.", Dd["mid"]["r0"]), a=Dd["mid"]["a"],
                                 r1=res(d + "mid_block.resnets.1.", Dd["mid"]["r1"])),
                        up=up, nout=Dd["nout"], cout=self.causal(d + "conv_out"))
        self.runner = _Runner(self._build, keep=4, keep_gpu=4)
        self.caches = {}        # (kind, h, w): {conv: (2 P, cin) array}
        self.arenas = {}        # (kind, h, w): the arena both programs share

    def causal(self, name):
        """A causal 3 x 3 x 3 conv as two linears: (out, 2 x 9 in) of its first two time taps (the
        cache's frames, the bias) and (out, 9 in) of the last (the frame)."""
        src = self.W.get(name + ".weight")
        raw = np.asarray(src, dtype=np.float32)
        assert raw.ndim == 5 and raw.shape[2] == 3, name
        w = raw.transpose(0, 2, 3, 4, 1)                                  # (out, kt, kh, kw, in)
        o = raw.shape[0]
        c = _Plain(np.ascontiguousarray(w[:, :2].reshape(o, -1)))
        c.shape = c.w.shape
        c.b = np.ascontiguousarray(self.W.get(name + ".bias"), dtype=np.float32)
        f = _Plain(np.ascontiguousarray(w[:, 2].reshape(o, -1)))
        f.shape = f.w.shape
        f.b = None
        drop_pages(src)
        return dict(cache=c, frame=f, cin=raw.shape[1], name=name)

    def _build(self, key, gpu):
        kind, first, h, w = key
        p = (self._encoder if kind == "enc" else self._decoder)(first, h, w, gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    # ---- the frame-by-frame builder ---------------------------------------------------------------

    class _B(SeedVR2VAE._Builder):
        """The image path's builder with causal convs over caches."""

        def __init__(self, gpu, caches, first):
            super().__init__(gpu)
            self.tiles, self.views = [], []
            self.caches, self.first = caches, first
            self.fill = set()           # the caches a first program has filled

        def cache(self, C, P):
            """The cache of a causal conv C over frames of P pixels: its two frames' views"""
            nm = C["name"]
            if nm not in self.caches:
                self.caches[nm] = np.zeros((2 * P, C["cin"]), np.float32)
            arr = self.caches[nm]
            base = "_c_" + nm
            self.env[base], self.env[base + "_a"], self.env[base + "_b"] = arr, arr[:P], arr[P:]
            return base

        def cconv(self, x, C, h, w, idx9, idx18, rows, emit=True):
            """The causal conv C of frame x (h w pixels, or the rows of idx9's input): a new tensor
            of rows (None when not emit: the cache only moves)."""
            P = h * w
            cb = self.cache(C, P)
            cin = C["cin"]
            if self.first and C["name"] not in self.fill:
                self.fill.add(C["name"])
                self.forms += [("set", cb + "_a", ("copy", x)), ("set", cb + "_b", ("copy", x))]
            out = None
            if emit:
                out = self.tensor(rows, C["cache"].shape[0])
                tmp = self.tensor(min(rows, TILE), C["cache"].shape[0])
                self.col_cols = max(self.col_cols, 18 * cin)
                for r0 in range(0, rows, TILE):
                    r1 = min(rows, r0 + TILE)
                    c18, c9, ot, tt, i18, i9 = (self.name() for _ in range(6))
                    self.env[i18], self.env[i9] = idx18[r0:r1], idx9[r0:r1]
                    self.tiles += [(c18, r1 - r0, 18 * cin), (c9, r1 - r0, 9 * cin)]
                    self.ar.alias[ot], self.ar.alias[tt] = out, tmp
                    self.views += [(ot, out, r0, r1), (tt, tmp, 0, r1 - r0)]
                    self.forms += [("set", c18, ("enc_gather", cb, i18, cin)),
                                   ("set", ot, ("enc_linear", C["cache"], c18)),
                                   ("set", c9, ("enc_gather", x, i9, cin)),
                                   ("set", tt, ("enc_linear", C["frame"], c9)),
                                   ("enc_add", ot, tt)]
            self.forms += [("set", cb + "_a", ("copy", cb + "_b")), ("set", cb + "_b", ("copy", x))]
            return out

        def cresnet(self, x, R, h, w, cin, cout, ix):
            rows = h * w
            i9, i18 = ix
            n1 = self.gnorm(x, R["n1"], rows, cin)
            h1 = self.cconv(n1, R["c1"], h, w, i9, i18, rows)
            n2 = self.gnorm(h1, R["n2"], rows, cout)
            h2 = self.cconv(n2, R["c2"], h, w, i9, i18, rows)
            s = self.linear(x, R["sc"], rows) if R["sc"] is not None else x
            self.forms.append(("enc_add", h2, s))
            return h2

        def finish2(self, inputs, outputs, arena):
            for n, (r, cc) in {**inputs, **outputs}.items():
                self.env[n] = np.zeros((r, cc), np.float32)
            reshaped = [v for v in self.views if v[0] == "_r"]
            self.views = [v for v in self.views if v[0] != "_r"]
            need = self.ar.place(self.forms, self.env, pool=arena)
            for _, uv, u, C in reshaped:
                self.env[uv] = self.env[u].reshape(-1, C)
            colbuf = np.zeros(TILE * max(self.col_cols, 1), np.float32)
            for col, r, cc in self.tiles:
                self.env[col] = colbuf[:r * cc].reshape(r, cc)
            for ot, out, r0, r1 in self.views:
                self.env[ot] = self.env[out][r0:r1]
            self.c.compile(("seq", *self.forms))
            return self.c.p.finish(), need

    def _index(self, b, h, w):
        """(im2col of a frame (h w, 9), of the cache (h w, 18)) of a size"""
        def make():
            i9 = conv_index(h, w)
            P = h * w
            return i9, np.concatenate([i9, np.where(i9 >= 0, i9 + P, -1)], 1).astype(np.int32)
        return b.idx(("cc", h, w), make)

    def _down_index(self, b, h, w):
        """the stride-2 conv's im2col (h/2 w/2, 9) of a frame and (.., 18) of the cache"""
        def make():
            i9 = down_index(h, w)
            P = h * w
            return i9, np.concatenate([i9, np.where(i9 >= 0, i9 + P, -1)], 1).astype(np.int32)
        return b.idx(("cd", h, w), make)

    def _finish(self, b, key, inputs, outputs):
        prog, arena = b.finish2(inputs, outputs, self.arenas.get(key))
        self.arenas[key] = arena
        return prog

    def _decoder(self, first, h, w, gpu):
        """Input "z" (h w, 16), output "out" (frames 8h 8w, 3): one frame (first) or four."""
        key = ("dec", h, w)
        b = self._B(gpu, self.caches.setdefault(key, {}), first)
        Dd = self.dec
        ix = self._index(b, h, w)
        x = b.cconv("z", Dd["cin"], h, w, *ix, h * w)
        M = Dd["mid"]
        x = b.cresnet(x, M["r0"], h, w, 512, 512, ix)
        x = b.attention(x, M["a"], h * w, 512)
        x = b.cresnet(x, M["r1"], h, w, 512, 512, ix)
        chans = [512, 512, 256, 128]
        outs = []

        def level(x, i, h, w, cin, phase_first):
            """up block i's ResNet blocks on a frame x, its upsampler to the next level's frames,
            on through the blocks after it; the output frames appended to outs"""
            ix = self._index(b, h, w)
            for j, R in enumerate(Dd["up"][i]["res"]):
                x = b.cresnet(x, R, h, w, cin, chans[i], ix)
                cin = chans[i]
            U = Dd["up"][i]
            if U["ups"] is None:
                n = b.gnorm(x, Dd["nout"], h * w, cin)
                ixo = self._index(b, h, w)
                outs.append(b.cconv(n, Dd["cout"], h, w, *ixo, h * w))
                return
            phases = U["ups"][:1] if phase_first else U["ups"]
            for z, lin in enumerate(phases):
                u = b.linear(x, lin, h * w)                              # (h w, 4 C) = (h w 4, C)
                uv = b.name()
                b.ar.alias[uv] = u
                b.views.append(("_r", uv, u, cin))
                hh, ww = 2 * h, 2 * w
                # the frame in raster order (the cache keeps frames so): a gather of the 2 x 2 blocks
                perm = b.idx(("perm", h, w), lambda: np.ascontiguousarray(up_index(h, w)[:, 4:5]))
                d = b.tensor(hh * ww, cin)
                b.forms.append(("set", d, ("enc_gather", uv, perm, cin)))
                y = b.cconv(d, U["upc"], hh, ww, *self._index(b, hh, ww), hh * ww)
                # the next level's first frame is phase 0 of this level's first frame
                level(y, i + 1, hh, ww, cin, phase_first and z == 0)
        level(x, 0, h, w, 512, first)
        H, Wd = 8 * h, 8 * w
        n_out = len(outs)
        assert n_out == (1 if first else 4), n_out
        for k, o in enumerate(outs):
            b.forms.append(("set", "out_%d" % k, ("copy", o)))
        prog = self._finish(b, key, {"z": (h * w, 16)}, {"out_%d" % k: (H * Wd, 3) for k in range(n_out)})
        return prog

    def _encoder(self, first, H, W, gpu):
        """Input "img_k" (H W, 3) of the program's frames (one: first; four), output "post" (H/8
        W/8, 32)."""
        key = ("enc", H, W)
        b = self._B(gpu, self.caches.setdefault(key, {}), first)
        E = self.enc
        frames = 1 if first else 4
        posts = []

        def l0(x, idx_in_chunk, abs_even):
            """an input frame through conv_in, down block 0 and down block 1's ResNet blocks, and
            down block 1's downsampler (it emits at the even frames)"""
            h, w = H, W
            ix = self._index(b, h, w)
            x = b.cconv(x, E["cin"], h, w, *ix, h * w)
            cin = 128
            for R in E["down"][0]["res"]:
                x = b.cresnet(x, R, h, w, cin, 128, ix)
            x = b.conv(x, 128, E["down"][0]["down"], b.idx(("d", h, w), lambda: down_index(h, w)), (h // 2) * (w // 2))
            h, w = h // 2, w // 2
            ix = self._index(b, h, w)
            cin = 128
            for R in E["down"][1]["res"]:
                x = b.cresnet(x, R, h, w, cin, 256, ix)
                cin = 256
            return b.cconv(x, E["down"][1]["down"], h, w, *self._down_index(b, h, w), (h // 2) * (w // 2),
                           emit=abs_even)

        def l1(x, emit2):
            h, w = H // 4, W // 4
            ix = self._index(b, h, w)
            cin = 256
            for R in E["down"][2]["res"]:
                x = b.cresnet(x, R, h, w, cin, 512, ix)
                cin = 512
            return b.cconv(x, E["down"][2]["down"], h, w, *self._down_index(b, h, w), (h // 2) * (w // 2),
                           emit=emit2)

        def l2(x):
            h, w = H // 8, W // 8
            ix = self._index(b, h, w)
            for R in E["down"][3]["res"]:
                x = b.cresnet(x, R, h, w, 512, 512, ix)
            M = E["mid"]
            x = b.cresnet(x, M["r0"], h, w, 512, 512, ix)
            x = b.attention(x, M["a"], h * w, 512)
            x = b.cresnet(x, M["r1"], h, w, 512, 512, ix)
            n = b.gnorm(x, E["nout"], h * w, 512)
            posts.append(b.cconv(n, E["cout"], h, w, *ix, h * w))

        # the frames of the program: first: frame 0 (even: L1 frame 0, even: L2 frame 0); later:
        # frames 4j-3 .. 4j: odd, even (L1 frame 2j-1: odd), odd, even (L1 frame 2j: even -> L2)
        if first:
            y = l0("img_0", 0, True)
            l2(l1(y, True))
        else:
            for k in range(4):
                y = l0("img_%d" % k, k, k % 2 == 1)
                if k == 1:
                    l1(y, False)
                elif k == 3:
                    l2(l1(y, True))
        assert len(posts) == 1
        b.forms.append(("set", "post", ("copy", posts[0])))
        return self._finish(b, key, {"img_%d" % k: (H * W, 3) for k in range(frames)}, {"post": (H * W // 64, 32)})

    # ---- the calls ---------------------------------------------------------------------------------

    def _run(self, key, inputs, outputs):
        prog, g = self.runner.get(key, self.gpu)
        for n, a in inputs.items():
            prog.names[n][...] = a
            if g is not None:
                g.upload(n)
        if g is not None:
            g.run()
            for n in outputs:
                g.download(n)
        else:
            prog.run()
        return [prog.names[n] for n in outputs]

    def encode(self, x, progress=None):
        """(mean, logvar) (16, T', H/8, W/8) of frames x (3, T, H, W), T = 4k + 1."""
        _, T, H, W = x.shape
        assert (T - 1) % 4 == 0, T
        if T > 1:
            self.runner.get(("enc", False, H, W), self.gpu)
        posts = []
        rows = lambda f: x[:, f].reshape(3, -1).T  # noqa: E731
        posts.append(self._run(("enc", True, H, W), {"img_0": rows(0)}, ["post"])[0].copy())
        for j in range(1, (T - 1) // 4 + 1):
            posts.append(self._run(("enc", False, H, W), {"img_%d" % k: rows(4 * j - 3 + k) for k in range(4)},
                                   ["post"])[0].copy())
            if progress is not None:
                progress(j, (T - 1) // 4)
        h, w = H // 8, W // 8
        p = np.stack([q.T.reshape(32, h, w) for q in posts], 1)
        return p[:16].copy(), np.clip(p[16:], -30.0, 20.0).astype(np.float32)

    def decode(self, z, progress=None):
        """The frames (3, 4 (T' - 1) + 1, 8h, 8w) of latents z (16, T', h, w)."""
        _, T, h, w = z.shape
        if T > 1:
            self.runner.get(("dec", False, h, w), self.gpu)
        frames = []
        for j in range(T):
            first = j == 0
            n = 1 if first else 4
            outs = self._run(("dec", first, h, w), {"z": z[:, j].reshape(16, -1).T}, ["out_%d" % k for k in range(n)])
            frames += [o.T.reshape(3, 8 * h, 8 * w).copy() for o in outs]
            if progress is not None:
                progress(j + 1, T)
        return np.stack(frames, 1)

    def release(self):
        self.runner.release(keep_host=True)
