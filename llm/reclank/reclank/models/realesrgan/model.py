"""Real-ESRGAN (xinntao/Real-ESRGAN, BSD-3): RRDBNet (RealESRGAN_x4plus, _x2plus, _x4plus_anime_6B)
and SRVGGNetCompact (realesr-general-x4v3, realesr-animevideov3) as programs, with RealESRGANer's
pre- and post-processing at inference_realesrgan.py's defaults (no tiles, no pre-padding).

Ported from Real-ESRGAN (xinntao/Real-ESRGAN, realesrgan/archs/srvgg_arch.py, realesrgan/utils.py,
inference_realesrgan.py; BSD 3-Clause License; Copyright (c) 2021, Xintao Wang) and BasicSR
(XPixelGroup/BasicSR, basicsr/archs/rrdbnet_arch.py; Apache License 2.0; Copyright 2018-2022 BasicSR
Authors). See THIRD_PARTY_NOTICES.md.

    RRDBNet (64 features, growth 32, LeakyReLU 0.2): the x2 model's input pixel-unshuffled by 2;
        conv_first; RRDB blocks (three residual dense blocks: conv1-4 each over the concatenation
        of the block's input and the earlier convs' outputs, LeakyReLU; conv5 over all of them,
        times 0.2, + the input; the block's output times 0.2 + its input); conv_body, + the
        features; two 2 x nearest upsamplings, each read by a conv's im2col, LeakyReLU; conv_hr,
        LeakyReLU, conv_last
    SRVGGNetCompact (64 features): conv, PReLU, then convs and PReLUs, the last conv to 3 x s^2
        channels; on the host the pixel shuffle and + the nearest upsampling of the input
    enhance: RGB / 255; the x2 model's input reflect-padded to even sides (cropped after); the
        output clamped to [0, 1], x 255, rounded (uint8)

The rows are pixels, the columns channels. A dense block's input and its convs' outputs live in one
buffer as rows of 32 values (the input's two rows a pixel, then each output's), so each conv's
im2col is one ENC_GATHER of 32-value blocks in the order of torch.cat. LeakyReLU and PReLU are
max(x, 0) + a min(x, 0): ENC_CLAMP, ENC_ADD (x - max(x, 0)), ENC_MUL_VEC, ENC_ADD: the bits of
torch's (one of the two terms is 0). The weights float32 as stored (params_ema / params, through
the .pth files converted once to safetensors by scripts/pt_to_safetensors.py --key params_ema,params: torch.load
with weights_only=True); realesr-general-x4v3's denoise strength s mixes its weights with
realesr-general-wdn-x4v3's: s a + (1 - s) b in float32 (inference_realesrgan.py's dni, 0.5 by
default). The tensors of a program share one arena (seedvr2.vae._Arena)."""
from __future__ import annotations

import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.seedvr2.vae import TILE, _Arena, conv_index
from reclank.runtime.threads import team_threads

# the models: the converted file, arch, scale, RRDB blocks or SRVGG convs
MODELS = {
    "realesrgan-x4plus": ("RealESRGAN_x4plus", "rrdb", 4, 23),
    "realesrgan-x2plus": ("RealESRGAN_x2plus", "rrdb", 2, 23),
    "realesrgan-x4plus-anime": ("RealESRGAN_x4plus_anime_6B", "rrdb", 4, 6),
    "realesr-general-x4v3": ("realesr-general-x4v3", "srvgg", 4, 32),
    "realesr-animevideov3": ("realesr-animevideov3", "srvgg", 4, 16),
}


def pixel_unshuffle(x, r):
    """(c, h, w) -> (c r r, h / r, w / r): channel c r^2 + i r + j of pixel (y, x) = x[c, r y + i, r x + j]."""
    c, h, w = x.shape
    return np.ascontiguousarray(x.reshape(c, h // r, r, w // r, r).transpose(0, 2, 4, 1, 3).reshape(c * r * r, h // r, w // r))


def pixel_shuffle(x, r):
    c, h, w = x.shape
    return np.ascontiguousarray(x.reshape(c // (r * r), r, r, h, w).transpose(0, 3, 1, 4, 2).reshape(c // (r * r), h * r, w * r))


class RealESRGAN:
    """A model of MODELS from its directory (the converted .safetensors). upscale(img) gives the output of an
    image (h, w, 3) uint8."""

    def __init__(self, path, model_id, gpu=False, cpu_threads=None, denoise=0.5):
        fname, self.arch, self.scale, self.n = MODELS[model_id]
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        R = SafeTensors(os.path.join(path, fname + ".safetensors"))
        W = {k: np.array(R.get(k), dtype=np.float32) for k in R.names()}       # copies: the file closes
        R.close()
        if model_id == "realesr-general-x4v3" and denoise != 1:
            B = SafeTensors(os.path.join(path, "realesr-general-wdn-x4v3.safetensors"))
            s, t = np.float32(denoise), np.float32(1 - denoise)
            W = {k: (s * v + t * B.get(k)).astype(np.float32) for k, v in W.items()}
            B.close()
        self.W = W
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)

    def conv(self, name):
        w = self.W[name + ".weight"]
        lin = _Plain(np.ascontiguousarray(w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1)))
        lin.shape = lin.w.shape
        lin.b = self.W[name + ".bias"]
        return lin

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def program(self, h, w, gpu=False):
        """The network on an input of h x w pixels: input "img" (h w, cin), output "y" (rows, cout):
        RRDB (4h 4w, 3), SRVGG (h w, 3 s^2)."""
        c = _EncCompiler(0.0, pack=not gpu)
        env = c.env
        # the GPU's convs by ENC_CONV_MM (no im2col buffer; LeakyReLU in its epilogue); RECLANK_ESR_IM2COL=1:
        # the ENC_GATHER + ENC_LINEAR tiles (the CPU's form)
        conv_mm = gpu and os.environ.get("RECLANK_ESR_IM2COL", "0") != "1"
        # RRDB's 3 x 3 convs of 32 or 64 outputs on the GPU by ENC_CONV3 (direct tiles: each input value read
        # once a tile; its own order of the sums, so not the CPU's bits); RECLANK_ESR_EXACT=1: ENC_CONV_MM
        direct = conv_mm and os.environ.get("RECLANK_ESR_EXACT", "0") != "1"
        ar = _Arena()
        forms, tiles, views = [], [], []
        n = [0]
        index = {}
        col_cols = [0]

        def name():
            n[0] += 1
            return "_v%d" % n[0]

        def conv(x, cin, lin, idx, rows, out=None, block=None, slope=None):
            """the conv of im2col idx over x (blocks of block values: cin / block a pixel's), then
            LeakyReLU of slope (its value) when given"""
            out = ar.new(name(), rows, lin.shape[0]) if out is None else out
            bl = block or cin
            if conv_mm:
                # (one ENC_CONV_MM: the im2col read through idx, LeakyReLU in its epilogue: the same values)
                tail = () if slope is None else (np.full(lin.shape[0], slope, np.float32),)
                forms.append(("set", out, ("enc_conv_mm", lin, x, np.ascontiguousarray(idx, np.int32), bl) + tail))
                return out
            col_cols[0] = max(col_cols[0], idx.shape[1] * bl)
            for r0 in range(0, rows, TILE):
                r1 = min(rows, r0 + TILE)
                col, ot, it = name(), name(), name()
                env[it] = idx[r0:r1]
                tiles.append((col, r1 - r0, idx.shape[1] * bl))
                ar.alias[ot] = out
                views.append((ot, out, r0, r1))
                forms.extend([("set", col, ("enc_gather", x, it, bl)), ("set", ot, ("enc_linear", lin, col))])
            if slope is not None:
                act(out, rows, lin.shape[0], "_lr32" if lin.shape[0] == 32 else "_lr")
            return out

        def conv3(x, groups, lin, hh, ww, out=None, up=0, slope=None, res=None, rs=1.0):
            """the ENC_CONV3 of the hh x ww image whose channels are groups of x ((offset, row stride,
            channels)), then x rs + res, LeakyReLU of slope (its value) when given"""
            out = ar.new(name(), hh * ww, lin.shape[0]) if out is None else out
            sl = None if slope is None else np.full(lin.shape[0], slope, np.float32)
            forms.append(("set", out, ("enc_conv3", lin, x, np.asarray(groups, np.int32), hh, ww, up, sl, res, rs)))
            return out

        def act(x, rows, cols, slope):
            """in place: max(x, 0) + slope min(x, 0)"""
            pos = ar.new(name(), rows, cols)
            forms.extend([("set", pos, ("enc_clamp", x, 0.0, float("inf"))), ("enc_add", x, pos, -1.0),
                          ("set", x, ("enc_mul_vec", x, slope)), ("enc_add", x, pos)])

        def idx_of(key, make):
            if key not in index:
                index[key] = make()
            return index[key]
        P = h * w
        i9 = idx_of(("c", h, w), lambda: conv_index(h, w))
        if self.arch == "srvgg":
            cin = 3
            env["img"] = np.zeros((P, cin), np.float32)
            x, k = "img", 0
            while "body.%d.weight" % k in self.W:
                wk = self.W["body.%d.weight" % k]
                if wk.ndim == 4:
                    x = conv(x, cin, self.conv("body.%d" % k), i9, P)
                    cin = wk.shape[0]
                else:
                    sl = name()
                    env[sl] = np.ascontiguousarray(wk, dtype=np.float32)
                    act(x, P, cin, sl)
                k += 1
            outs = (P, cin)
        else:
            cin = 3 * (4 if self.scale == 2 else 1)
            env["img"] = np.zeros((P, cin), np.float32)
            env["_lr"] = np.full(64, 0.2, np.float32)
            env["_lr32"] = np.full(32, 0.2, np.float32)
            env["_s02"] = np.full(64, 0.2, np.float32)
            feat = conv("img", cin, self.conv("conv_first"), i9, P)
            body = ar.new(name(), P, 64)
            forms.append(("set", body, ("copy", feat)))
            pools = []
            for q in range(2):
                pool = ar.new(name(), 6 * P, 32)
                A, regs = name(), []
                ar.alias[A] = pool
                views.append(("_rows", A, pool, 0, 2 * P, 64))
                for r in range(4):
                    v = name()
                    ar.alias[v] = pool
                    views.append(("_rows", v, pool, (2 + r) * P, (3 + r) * P, 32))
                    regs.append(v)
                pools.append((pool, A, regs))

            def dense_index(k):
                """the im2col (P, 9 (k + 1)) of conv k (1..5): the input's two rows and the k - 1
                outputs' rows of each of the 9 neighbours, in torch.cat's order"""
                def make():
                    nb = i9
                    parts = [np.where(nb >= 0, 2 * nb, -1), np.where(nb >= 0, 2 * nb + 1, -1)]
                    parts += [np.where(nb >= 0, (2 + r) * P + nb, -1) for r in range(k - 1)]
                    return np.ascontiguousarray(np.stack(parts, 2).reshape(P, -1).astype(np.int32))
                return idx_of(("dense", k), make)

            def rdb(src, dst, p):
                pool, A, regs = src
                if direct:          # (the dense input: A's 64 channels, then the outputs' 32 each, in place)
                    gr = [(0, 64, 64)] + [((2 + r) * P * 32, 32, 32) for r in range(4)]
                    for k in range(1, 5):
                        conv3(pool, gr[:k], self.conv(p + "conv%d" % k), h, w, out=regs[k - 1], slope=0.2)
                    conv3(pool, gr, self.conv(p + "conv5"), h, w, out=dst[1], res=A, rs=0.2)
                    return
                for k in range(1, 5):
                    conv(pool, 64 + 32 * (k - 1), self.conv(p + "conv%d" % k), dense_index(k), P, out=regs[k - 1],
                         block=32, slope=0.2)
                x5 = conv(pool, 192, self.conv(p + "conv5"), dense_index(5), P, block=32)
                forms.extend([("set", dst[1], ("enc_mul_vec", x5, "_s02")), ("enc_add", dst[1], A)])
            for b in range(self.n):
                p = "body.%d." % b
                forms.append(("set", pools[0][1], ("copy", body)))
                rdb(pools[0], pools[1], p + "rdb1.")
                rdb(pools[1], pools[0], p + "rdb2.")
                rdb(pools[0], pools[1], p + "rdb3.")
                t = ar.new(name(), P, 64)
                forms.extend([("set", t, ("enc_mul_vec", pools[1][1], "_s02")), ("enc_add", body, t)])
            if direct:
                conv3(body, [(0, 64, 64)], self.conv("conv_body"), h, w, out=feat, res=feat)
            else:
                bb = conv(body, 64, self.conv("conv_body"), i9, P)
                forms.append(("enc_add", feat, bb))
            x, hh, ww = feat, h, w
            for nm in ("conv_up1", "conv_up2"):
                hq, wq = hh, ww
                iu = idx_of(("up", hh, ww), lambda: conv_index(2 * hq, 2 * wq, lambda y, x_: (y // 2) * wq + x_ // 2))
                hh, ww = 2 * hh, 2 * ww
                if direct:
                    x = conv3(x, [(0, 64, 64)], self.conv(nm), hh, ww, up=1, slope=0.2)
                else:
                    x = conv(x, 64, self.conv(nm), iu, hh * ww, slope=0.2)
            i4 = idx_of(("c", hh, ww), lambda: conv_index(hh, ww))
            if direct:
                x = conv3(x, [(0, 64, 64)], self.conv("conv_hr"), hh, ww, slope=0.2)
            else:
                x = conv(x, 64, self.conv("conv_hr"), i4, hh * ww, slope=0.2)
            x = conv(x, 64, self.conv("conv_last"), i4, hh * ww)
            outs = (hh * ww, 3)
        env["y"] = np.zeros(outs, np.float32)
        forms.append(("set", "y", ("copy", x)))
        arena = ar.place(forms, env)
        for v in views:
            if v[0] == "_rows":
                _, nm, pool, r0, r1, cols = v
                env[nm] = env[pool][r0:r1].reshape(-1, cols)
        colbuf = np.zeros(TILE * max(col_cols[0], 1), np.float32)
        for col, r, cc in tiles:
            env[col] = colbuf[:r * cc].reshape(r, cc)
        for v in views:
            if v[0] != "_rows":
                ot, out, r0, r1 = v
                env[ot] = env[out][r0:r1]
        self.arena = arena
        c.compile(("seq", *forms))
        return c.p.finish()

    def network(self, x):
        """The network's output (3, s h, s w) of its input x (cin, h, w)."""
        cin, h, w = x.shape
        prog, g = self.runner.get((h, w), self.gpu)
        prog.names["img"][...] = x.reshape(cin, -1).T
        if g is not None:
            g.upload("img")
            g.run()
            g.download("y")
        else:
            prog.run()
        y = prog.names["y"]
        if self.arch == "srvgg":
            out = pixel_shuffle(y.T.reshape(-1, h, w), self.scale)
            base = np.repeat(np.repeat(x, self.scale, 1), self.scale, 2)
            return (out + base).astype(np.float32)
        return y.T.reshape(3, 4 * h, 4 * w).copy()

    def upscale(self, img, parts=None):
        """The output (s h, s w, 3) uint8 of an image (h, w, 3) uint8 (RealESRGANer.enhance)."""
        y = self.upscale_f32(img.astype(np.float32) / np.float32(255.0), parts)
        return np.round(np.clip(y, 0, 1) * np.float32(255.0)).astype(np.uint8)

    def upscale_f32(self, img, parts=None):
        """The output (s h, s w, 3) float32 of a float32 image (h, w, 3) (0 to 1 the trained range):
        upscale() before its clamp and rounding (for sources of more than 8 bits)."""
        x = np.ascontiguousarray(img.transpose(2, 0, 1), np.float32)
        h, w = x.shape[1:]
        ph = pw = 0
        if self.scale == 2:
            ph, pw = h % 2, w % 2
            if ph or pw:
                x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            xin = pixel_unshuffle(x, 2)
        else:
            xin = x
        y = self.network(np.ascontiguousarray(xin))[:, :h * self.scale, :w * self.scale]
        if parts is not None:
            parts.update(x=x, y=y)
        return y.transpose(1, 2, 0)

    def release(self):
        if self.gpu:
            self.runner.release(keep_host=True)
