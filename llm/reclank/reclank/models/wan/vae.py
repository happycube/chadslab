"""Wan 2.2's VAE decoder (diffusers' AutoencoderKLWan of TI2V-5B: 48 latent channels, 16 x 16
spatial and 4 x temporal, the decoder of 1024 1024 1024 512 256 channels, patch_size 2) as two
programs: the first latent frame (one video frame) and each later one (four frames), as
diffusers' _decode runs the decoder a latent frame at a time with the cache of its causal convs.
Also Wan 2.1's (T2V-A14B's: 16 latent channels, 8 x 8 x 4, the decoder of 384 384 384 192 96, no
patch, is_residual false): its up blocks after the first take half the channels the block before
gives out (the upsamplers' convs to half), and no DupUp3D shortcuts; the rest the same.

Ported from diffusers (src/diffusers/models/autoencoders/autoencoder_kl_wan.py; Apache License 2.0;
Copyright 2025 The Wan Team and The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

    post_quant_conv (1 x 1 x 1), conv_in (3 x 3 x 3, 48 -> 1024)
    the mid block: a residual block, the attention of a frame's pixels over its pixels (the RMS
        norm, q k v by one 1 x 1 conv with bias, one head of 1024, scale 1024^-1/2, proj, + x),
        a residual block
    four up blocks (1024 1024 512 256 channels out) of three residual blocks (the RMS norm,
        SiLU, conv, the RMS norm, SiLU, conv, + x or a 1 x 1 x 1 conv of x), the first three
        with an upsampler after them, plus the block's shortcut (DupUp3D of its input):
        upsample3d (the first two blocks): time_conv (3 x 1 x 1, c -> 2c: each frame two, the
            channels' halves) but on the first latent frame, then a 2 x nearest upsampling and
            a 3 x 3 conv of each frame
        upsample2d (the third): the upsampling and the conv of each frame
    the RMS norm, SiLU, conv_out (3 x 3 x 3, 256 -> 12), unpatched on the host (12 = 3 colours
        x 2 x 2: the 16 x 16 of a latent pixel), clamped to [-1, 1]

The causal convs (WanCausalConv3d, kernel 3 in time): the input of a chunk after the last two
frames of the input of the chunks before (2 zero frames before the first), as diffusers' cache
(feat_cache: the last CACHE_T = 2 frames); time_conv's frames are those of the latent frames
after the first (its first chunk is skipped, "Rep": zeros before the second). Each tensor that a
causal conv reads lives in a buffer of two frames more than its chunk's: the cache in the first
two, the chunk's frames after them, so the conv's im2col (ENC_GATHER of the 27 neighbours, -1
past the edges) reads both; after the chunk the last two frames move to the front (two copies of
a frame). The first program zeros the caches. The norm before a conv (WanRMS_norm: x / |x|
sqrt(channels) gamma, ENC_RMS with no eps) is of each pixel, so the cache keeps the norm's input:
on the GPU the norm and SiLU run on a conv tile's gathered pixels (as Qwen-Image's VAE), on the
CPU on the whole buffer first. DupUp3D of a later chunk: each input frame to two output frames;
the decoder's channel maps do not depend on the time phase (checked), so ENC_DUPUP (the last
phase, first_chunk) gives both. The rows are pixels (row major) of each frame in turn, the
columns channels; the matrices the checkpoint's float32 as stored (CPU and GPU).

Memory: the buffers of all the causal convs' inputs stay, and do not depend on the frames: 9.0 GB
at 480 x 832, 19.8 GB at 704 x 1280 (the GPU's peak 22.6 GB of the 3090's 24), the last two levels'
(256 and 512 channels at 4 frames and two of cache) most of it. The compact layout (when that
passes COMPACT_GB, RECLANK_WAN_VAE_COMPACT=1 / 0): each causal buffer's cache in an array of its
own (two frames), its chunk in one of three work slots (four with DupUp3D) that the buffers share
as they die (_slots), the cache copied into the slot's front before its conv and its last two
frames back after it: the same bits (the copies only), Wan 2.1's VAE 26.3 -> 10.0 GB at 720 x 1280
(11.5 -> 4.5 GB at 480 x 832), the 5B's 19.8 -> 7.7 GB at 704 x 1280.
No tiles: diffusers' tiled_decode (enable_tiling: latent tiles of 16 x 16 at a stride of 12, the
overlaps blended) gives another video than the untiled decode, the pipeline's default.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import RESIDENCY, _EncCompiler, _Plain, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.flux2.vae import conv_index
from reclank.runtime.threads import team_threads

COL = 1 << 26       # the values of an im2col tile (256 MB)
LIN_ROWS = 1 << 18  # the rows of a 1 x 1 conv at a time (its buffer)
NORM_EPS = 1e-30    # F.normalize's max(|x|, 1e-12): no eps but for a zero pixel
CACHE = 2           # the frames of a causal conv's cache (diffusers' CACHE_T)
# the compact layout when the buffers of the full one pass this (GB); RECLANK_WAN_VAE_COMPACT=1 / 0
COMPACT_GB = 14.0


def causal_index(t, hh, ww):
    """The im2col index (t hh ww, 27) of a 3 x 3 x 3 causal conv over a buffer of 2 + t frames of
    hh x ww pixels (the cache's two, then the chunk's): output frame f reads frames f .. f + 2;
    -1 past the edge."""
    P = hh * ww
    sp = conv_index(hh, ww)                       # (P, 9)
    f = np.arange(t)[:, None, None, None]
    kt = np.arange(3)[None, None, :, None]
    idx = np.where(sp[None, :, None, :] >= 0, sp[None, :, None, :] + (f + kt) * P, -1)
    return idx.reshape(t * P, 27).astype(np.int32)


def frames_index(idx, t, rows_in):
    """An index of one frame for t frames (frame f's rows after f rows_in of the input)."""
    return np.concatenate([np.where(idx >= 0, idx + f * rows_in, -1) for f in range(t)]).astype(np.int32)


class WanVAE:
    """The decoder of a vae directory (config.json and its safetensors). decode(z) gives the
    video (3, frames, 16 h, 16 w) in [-1, 1] of latents (48, latent frames, h, w)."""

    def __init__(self, path, gpu=False, cpu_threads=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        # the 5B's (is_residual, patch 2) or Wan 2.1's (the A14B's: no DupUp3D shortcuts, the up
        # blocks after the first with half the channels in, the upsamplers to half, no patch)
        self.residual = bool(cfg.get("is_residual", False))
        assert (cfg.get("patch_size") or 1) == (2 if self.residual else 1) and not cfg["attn_scales"]
        self.W = W = SafeTensors(os.path.join(path, "diffusion_pytorch_model.safetensors"))
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.latent = cfg["z_dim"]
        self.patch = cfg.get("patch_size") or 1
        # (the 5B's 12: 3 colours x 2 x 2; Wan 2.1's config has none: AutoencoderKLWan's 3)
        self.out_ch = cfg.get("out_channels", 3)
        self.mean = np.array(cfg["latents_mean"], np.float32)
        self.std = np.array(cfg["latents_std"], np.float32)
        mult = cfg["dim_mult"]
        base = cfg.get("decoder_base_dim") or cfg["base_dim"]
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
        us = "upsampler." if self.residual else "upsamplers.0."
        for i, (cin, cout) in enumerate(zip(dims[:-1], dims[1:])):
            p = d + "up_blocks.%d." % i
            if i > 0 and not self.residual:
                cin //= 2       # (Wan 2.1: the upsampler before gave half the channels)
            up = i != len(mult) - 1
            t3 = up and tup[i]
            B = dict(cin=cin, cout=cout, res=[res(p + "resnets.%d." % j) for j in range(cfg["num_res_blocks"] + 1)],
                     up=self.conv(p + us + "resample.1") if up else None, ft=2 if t3 else 1)
            if t3:
                tc = self.conv(p + us + "time_conv")
                B["tc"] = [self._rows(tc, 0, cout), self._rows(tc, cout, 2 * cout)]
            if up and self.residual:
                self._dupup_phase_free(cin, cout, B["ft"], 2)
            self.up.append(B)
        self.nout = g(d + "norm_out.gamma")
        self.cout = self.conv(d + "conv_out")
        self.runner = _Runner(self._build, keep=2, keep_gpu=2)
        self.bufs = {}          # (h, w): the buffers both programs share (the caches)

    @staticmethod
    def _dupup_phase_free(cin, cout, ft, fs):
        """DupUp3D's input channel of each output channel and spatial phase does not depend on the
        time phase (ENC_DUPUP gives the last one)."""
        F = ft * fs * fs
        rep = cout * F // cin
        o = np.arange(cout)[:, None]
        sp = np.arange(fs * fs)[None, :]
        ch = [(o * F + a * fs * fs + sp) // rep for a in range(ft)]
        assert all(np.array_equal(ch[0], c) for c in ch), "DupUp3D's channels depend on the time phase"

    def conv(self, name):
        """A conv (out, in, [kt,] kh, kw) as a linear of its im2col: W (out, kt kh kw in) float32,
        the bias."""
        raw = np.asarray(self.W.get(name + ".weight"), dtype=np.float32)
        raw = raw.transpose(0, *range(2, raw.ndim), 1).reshape(raw.shape[0], -1)
        lin = _Plain(np.ascontiguousarray(raw))
        lin.shape = raw.shape
        lin.b = np.ascontiguousarray(self.W.get(name + ".bias"), dtype=np.float32)
        return lin

    @staticmethod
    def _rows(lin, r0, r1):
        """The linear of rows [r0, r1) of lin (q, k, v of a fused conv; time_conv's halves)."""
        out = _Plain(np.ascontiguousarray(lin.w[r0:r1]))
        out.shape = out.w.shape
        out.b = np.ascontiguousarray(lin.b[r0:r1])
        return out

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def _levels(self, h, w):
        """(hh, ww, frames of a later chunk) of the up blocks' levels."""
        lv, t = [], 1
        for i in range(len(self.up)):
            lv.append((h << i, w << i, t))
            if self.up[i]["up"] is not None:
                t *= self.up[i]["ft"]
        return lv

    def _shared(self, h, w):
        """The buffers of latents h x w that both programs read (allocated for a later chunk's
        frames): name: (rows, cols) arrays, the causal ones of 2 + t frames; and the causal
        ones' levels."""
        key = (h, w)
        if key in self.bufs:
            b, level, self.compact = self.bufs[key]
            return b, level
        lv = self._levels(h, w)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        b = {}
        P0 = h * w
        level = {}

        causal = {}

        def cb(name, li, cols):
            causal[name] = ((CACHE + lv[li][2]) * lv[li][0] * lv[li][1], cols)
            level[name] = li
        cb("zq", 0, self.latent)
        C = self.up[0]["cin"]
        for n in ("m_x0", "m_h0", "m_x1", "m_h1"):
            cb(n, 0, C)
        for i, B in enumerate(self.up):
            cin = B["cin"]
            for j in range(len(B["res"])):
                cb("u%d_x%d" % (i, j), i, cin)
                cb("u%d_h%d" % (i, j), i, B["cout"])
                cin = B["cout"]
            cb("u%d_x%d" % (i, len(B["res"])), i, B["cout"])
        full = sum(r * c for r, c in causal.values()) * 4
        env_c = os.environ.get("RECLANK_WAN_VAE_COMPACT")
        self.compact = env_c == "1" if env_c in ("0", "1") else full > COMPACT_GB * 1e9
        if not self.compact:
            for n, (r, c) in causal.items():
                b[n] = z(r, c)
        else:
            # each causal buffer a view of a work slot (the cache and the chunk's frames; the slots
            # reused as the buffers die: _slots), and its cache in an array of its own
            slot, nslots = self._slots(lv)
            big = max(r * c for r, c in causal.values())
            pool = [z(big) for _ in range(nslots)]
            for n, (r, c) in causal.items():
                b[n] = pool[slot[n]][:r * c].reshape(r, c)
                hh_, ww_, _t = lv[level[n]]
                b["cache_" + n] = z(CACHE * hh_ * ww_, c)
        hh, ww, t = lv[-1]
        b["img"] = z(t * hh * ww, self.out_ch)
        b["z"] = z(P0, self.latent)
        # the temporaries: time_conv's output (two frames a chunk frame), the attention's
        tco = max((2 * lv[i][2] * lv[i][0] * lv[i][1] * B["cout"] for i, B in enumerate(self.up) if "tc" in B),
                  default=1)
        b["tco"] = z(tco)
        for n in ("an", "aq", "ak", "av", "ao"):
            b[n] = z(P0, C)
        b["_scale"] = np.full(C, C ** -0.5, np.float32)
        b["col"] = z(COL)
        # the 1 x 1 shortcut convs' output: the blocks that have one (512 and 256 channels)
        b["sc"] = z(max((min(lv[i][2] * lv[i][0] * lv[i][1], LIN_ROWS) * B["cout"] for i, B in enumerate(self.up)
                         if any(R["sc"] is not None for R in B["res"])), default=1))
        assert all(R["sc"] is None for R in (self.mid["r0"], self.mid["r1"]))
        if not self.gpu:
            # the normed input of a conv (CPU): the largest causal buffer
            b["nrm"] = z(max(a.size for n, a in b.items() if a.ndim == 2 and n not in ("img", "z")
                             and not n.startswith("cache_")))
        self.bufs[key] = b, level, self.compact
        return b, level

    def _slots(self, lv):
        """The compact layout's work slots: {causal buffer: slot}, and the count. In the order the
        buffers are made (conv_in's output after zq, a residual block's h and its output, the next
        level's input after the upsampling), each takes the lowest free slot, and the buffers whose
        last reader is that buffer's making are free after it: zq after conv_in; a residual block's
        input and h after its output (the add); a block's last output after the next block's input
        (the upsampling; with DupUp3D the block's input too). Three slots (four with DupUp3D)."""
        seq = [("zq", []), ("m_x0", ["zq"]), ("m_h0", []), ("m_x1", ["m_x0", "m_h0"]), ("m_h1", []),
               ("u0_x0", ["m_x1", "m_h1"])]
        for i, B in enumerate(self.up):
            n = len(B["res"])
            for j in range(n):
                keep0 = self.residual and j == 0 and B["up"] is not None
                seq += [("u%d_h%d" % (i, j), []),
                        ("u%d_x%d" % (i, j + 1), (["u%d_x%d" % (i, j)] if not keep0 else []) + ["u%d_h%d" % (i, j)])]
            if i + 1 < len(self.up):
                seq.append(("u%d_x0" % (i + 1), ["u%d_x%d" % (i, n)] +
                            (["u%d_x0" % i] if self.residual and B["up"] is not None else [])))
        slot, free, used = {}, [], 0
        for name, dead in seq:
            if free:
                free.sort()
                slot[name] = free.pop(0)
            else:
                slot[name], used = used, used + 1
            free += [slot[d] for d in dead]
        return slot, used

    def program(self, first, h, w, gpu=False):
        """The program of a latent frame of h x w: the first (first) or a later one. Input "z" (h
        w, 48), output "img" (frames hh ww, 12): one frame (first) or four of 8h x 8w pixels."""
        c = _EncCompiler(NORM_EPS, pack=not gpu)
        env = c.env
        S, level = self._shared(h, w)
        env.update({k: v for k, v in S.items() if k not in ("col", "tco", "sc", "nrm")})
        lv = self._levels(h, w)
        T = [1 if first else t for (_, _, t) in lv]
        forms = []
        n_names = [0]
        index = {}
        causal = []         # the buffers whose caches move after the chunk: (name, level)

        def name():
            n_names[0] += 1
            return "_v%d" % n_names[0]

        def view(src, start, rows, cols):
            """a view of rows [start, start + rows) of a buffer (or of a flat array: cols a row)"""
            nm = name()
            a = S[src]
            env[nm] = a[start:start + rows] if a.ndim == 2 else a[start * cols:(start + rows) * cols].reshape(rows, cols)
            return nm

        def body(src, li, f0=0, nf=None):
            """the chunk's frames [f0, f0 + nf) of a causal buffer of level li"""
            hh, ww, _ = lv[li]
            P = hh * ww
            nf = T[li] - f0 if nf is None else nf
            return view(src, (CACHE + f0) * P, nf * P, S[src].shape[1])

        def whole(src, li):
            """the cache and the chunk's frames of a causal buffer (compact: its cache copied in
            first)"""
            hh, ww, _ = lv[li]
            if self.compact:
                forms.append(("set", view(src, 0, CACHE * hh * ww, S[src].shape[1]), ("copy", "cache_" + src)))
            return view(src, 0, (CACHE + T[li]) * hh * ww, S[src].shape[1])

        def keep_cache(src, li):
            """compact: the cache of a causal buffer after its conv read it (its last two frames)"""
            if self.compact:
                hh, ww, _ = lv[li]
                forms.append(("set", "cache_" + src, ("copy", view(src, T[li] * hh * ww, CACHE * hh * ww,
                                                                     S[src].shape[1]))))

        def gathered(x, idx, cin, lin, r_out, gamma):
            """out's rows of lin over the im2col of x by idx (rows, blocks), a tile at a time;
            gamma: the norm and SiLU of each gathered pixel (GPU)"""
            rows, blocks = idx.shape
            step = max(1, min(rows, COL // (blocks * cin)))
            for r0 in range(0, rows, step):
                r1 = min(rows, r0 + step)
                col = view("col", 0, r1 - r0, blocks * cin)
                forms.append(("set", col, ("enc_gather", x, np.ascontiguousarray(idx[r0:r1]), cin)))
                if gamma is not None:
                    forms.extend([("set", col, ("enc_rms", col, gamma, cin)), ("set", col, ("enc_silu", col))])
                forms.append(("set", r_out(r0, r1), ("enc_linear", lin, col)))

        def normed(x, gamma, cin):
            """the norm and SiLU of a buffer into "nrm" (CPU)"""
            y = view("nrm", 0, env[x].shape[0], cin)
            forms.extend([("set", y, ("enc_rms", x, gamma, cin)), ("set", y, ("enc_silu", y))])
            return y

        def conv3t(src, li, cin, lin, dst, dli, gamma=None):
            """a causal 3 x 3 x 3 conv of the causal buffer src (level li) into the chunk's frames
            of dst (a causal buffer of level dli, or "img"); gamma: the norm and SiLU before it"""
            hh, ww, _ = lv[li]
            t = T[li]
            key = ("c", t, hh, ww)
            if key not in index:
                index[key] = causal_index(t, hh, ww)
            x = whole(src, li)
            if gamma is not None and not gpu:
                x = normed(x, gamma, cin)
                gamma = None
            off = CACHE * hh * ww if dst != "img" else 0
            cols = S[dst].shape[1]
            gathered(x, index[key], cin, lin, lambda r0, r1: view(dst, off + r0, r1 - r0, cols), gamma)
            causal.append((src, li))
            keep_cache(src, li)

        def resnet(x, h1, out, li, cin, cout, R):
            """a residual block of the causal buffer x into the chunk's frames of out (level li)"""
            conv3t(x, li, cin, R["c1"], h1, li, gamma=R["n1"])
            conv3t(h1, li, cout, R["c2"], out, li, gamma=R["n2"])
            hh, ww, _ = lv[li]
            rows = T[li] * hh * ww
            if R["sc"] is not None:
                for r0 in range(0, rows, LIN_ROWS):
                    r1 = min(rows, r0 + LIN_ROWS)
                    s = view("sc", 0, r1 - r0, cout)
                    forms.extend([("set", s, ("enc_linear", R["sc"], view(x, CACHE * hh * ww + r0, r1 - r0, cin))),
                                  ("enc_add", view(out, CACHE * hh * ww + r0, r1 - r0, cout), s)])
            else:
                forms.append(("enc_add", body(out, li), body(x, li)))

        if first:
            # the caches start as zeros (the conv's padding of the first chunk)
            heads = {n: CACHE * lv[li][0] * lv[li][1] for n, li in level.items()}
            zero = name()
            env[zero] = np.zeros(max(r * S[n].shape[1] for n, r in heads.items()), np.float32)
            for n, r in heads.items():
                nm = name()
                env[nm] = env[zero][:r * S[n].shape[1]].reshape(r, S[n].shape[1])
                forms.append(("set", "cache_" + n if self.compact else view(n, 0, r, S[n].shape[1]), ("copy", nm)))
        # post_quant_conv, conv_in, the mid block
        forms.append(("set", body("zq", 0), ("enc_linear", self.pq, "z")))
        C = self.up[0]["cin"]
        conv3t("zq", 0, self.latent, self.cin, "m_x0", 0)
        M = self.mid
        resnet("m_x0", "m_h0", "m_x1", 0, C, C, M["r0"])
        x1 = body("m_x1", 0)
        forms += [("set", "an", ("enc_rms", x1, M["n"], C)),
                  ("set", "aq", ("enc_linear", M["q"], "an")),
                  ("set", "aq", ("enc_mul_vec", "aq", "_scale")),
                  ("set", "ak", ("enc_linear", M["k"], "an")),
                  ("set", "av", ("enc_linear", M["v"], "an")),
                  ("set", "ao", ("enc_attn_wide", "aq", "ak", "av")),
                  ("set", "aq", ("enc_linear", M["o"], "ao")),
                  ("enc_add", x1, "aq")]
        resnet("m_x1", "m_h1", "u0_x0", 0, C, C, M["r1"])
        # the up blocks
        for i, B in enumerate(self.up):
            cin = B["cin"]
            nres = len(B["res"])
            for j, R in enumerate(B["res"]):
                resnet("u%d_x%d" % (i, j), "u%d_h%d" % (i, j), "u%d_x%d" % (i, j + 1), i, cin, B["cout"], R)
                cin = B["cout"]
            if B["up"] is None:
                continue
            hh, ww, _ = lv[i]
            P = hh * ww
            t = T[i]
            xo = "u%d_x%d" % (i, nres)
            src, nf = xo, t
            if "tc" in B and not first:
                # time_conv: frame f of the chunk reads frames f .. f + 2 of the buffer (its cache
                # first); its halves to frames 2f and 2f + 1
                key = ("t", hh, ww)
                if key not in index:
                    sp = np.arange(P)[:, None]
                    index[key] = (sp + np.arange(3)[None, :] * P).astype(np.int32)
                x = whole(xo, i)
                cout = B["cout"]
                for f in range(t):
                    for ph in range(2):
                        gathered(x, index[key] + f * P, cout, B["tc"][ph],
                                 lambda r0, r1, f=f, ph=ph: view("tco", (2 * f + ph) * P + r0, r1 - r0, cout), None)
                causal.append((xo, i))
                keep_cache(xo, i)
                src, nf = "tco", 2 * t
            # the upsampling and the 3 x 3 conv of each frame into the next level's chunk
            key = ("u", nf, hh, ww)
            if key not in index:
                index[key] = frames_index(conv_index(hh, ww, up=True), nf, P)
            xs = view(src, 0 if src == "tco" else CACHE * P, nf * P, B["cout"])
            nP = 4 * P
            nxt = "u%d_x0" % (i + 1)
            gathered(xs, index[key], B["cout"], B["up"],
                     lambda r0, r1: view(nxt, CACHE * nP + r0, r1 - r0, S[nxt].shape[1]), None)
            # the shortcut: DupUp3D of the block's input, each frame to ft frames (one: the first)
            xin = "u%d_x0" % i
            for f in range(t if self.residual else 0):
                for ph in range(1 if first else B["ft"]):
                    of = f * (1 if first else B["ft"]) + ph
                    forms.append(("enc_dupup", body(xin, i, f, 1), body(nxt, i + 1, of, 1), hh, ww, B["cin"],
                                  B["cout"], B["ft"], 2))
        # the head
        L = len(self.up) - 1
        conv3t("u%d_x%d" % (L, len(self.up[L]["res"])), L, self.up[L]["cout"], self.cout, "img", L,
               gamma=self.nout)
        # the caches: the last two frames of each causal buffer to the front (compact: kept after
        # each conv)
        for src, li in dict.fromkeys(causal) if not self.compact else ():
            hh, ww, _ = lv[li]
            P = hh * ww
            cols = S[src].shape[1]
            for j in range(CACHE):
                forms.append(("set", view(src, j * P, P, cols), ("copy", view(src, (T[li] + j) * P, P, cols))))
        c.compile(("seq", *forms))
        return c.p.finish()

    def denorm(self, latents):
        """The VAE's input of the DiT's latents (48, f, h, w): latents / (1 / std) + mean, as the
        pipeline (float32)."""
        istd = (np.float32(1) / self.std).astype(np.float32)
        return (latents / istd[:, None, None, None] + self.mean[:, None, None, None]).astype(np.float32)

    def _unpatch(self, img, t, hh, ww):
        """(3, t, 2 hh, 2 ww) of the output rows (t hh ww, 12): channel c 4 + q 2 + r is the
        pixel (2 y + r, 2 x + q) of colour c (diffusers' unpatchify)."""
        p = self.patch
        x = img.reshape(t, hh, ww, self.out_ch // (p * p), p, p).transpose(3, 0, 1, 5, 2, 4)
        # (a copy: with no patch (Wan 2.1) the reshape is a view of the program's buffer, which the
        # next chunk writes)
        return np.array(x.reshape(self.out_ch // (p * p), t, hh * p, ww * p), np.float32)

    def decode(self, z, progress=None):
        """The video (3, frames, 16 h, 16 w) in [-1, 1] of latents z (48, F, h, w) (after
        denorm): frames 1 + 4 (F - 1). progress(k, F): after each latent frame. The runner stays
        pinned over the frames (the causal convs' caches on the GPU)."""
        _, F, h, w = z.shape
        hh, ww, _ = self._levels(h, w)[-1]
        out = []
        with RESIDENCY.use(self.runner if self.gpu else None):
            for k in range(F):
                prog, g = self.runner.get((k == 0, h, w), self.gpu)
                prog.names["z"][...] = z[:, k].reshape(self.latent, h * w).T
                if g is not None:
                    g.upload("z")
                    g.run()
                    g.download("img")
                else:
                    prog.run()
                t = 1 if k == 0 else self._levels(h, w)[-1][2]
                out.append(self._unpatch(prog.names["img"][:t * hh * ww], t, hh, ww))
                if progress is not None:
                    progress(k + 1, F)
        return np.clip(np.concatenate(out, 1), -1.0, 1.0)
