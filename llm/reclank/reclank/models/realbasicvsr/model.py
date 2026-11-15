"""RealBasicVSR (ckkelvinchan/RealBasicVSR, Apache 2.0): video super-resolution x4 of real-world footage, as
programs, float32 as stored, on the CPU and the GPU.

Ported from MMEditing v0.16.0 (open-mmlab/mmediting, mmedit/models/backbones/sr_backbones/real_basicvsr_net.py
and basicvsr_net.py, mmedit/models/common/flow_warp.py, sr_backbone_utils.py, upsample.py; Apache License 2.0;
Copyright (c) OpenMMLab; Copyright 2020 MMEditing Authors) and RealBasicVSR (ckkelvinchan/RealBasicVSR,
inference_realbasicvsr.py, configs/realbasicvsr_x4.py; Apache License 2.0; Copyright 2021 Kelvin C.K. Chan). See
THIRD_PARTY_NOTICES.md.

The network (RealBasicVSRNet, 64 channels, 20 blocks a branch; the official forward_test's generator_ema), the
frames RGB in [0, 1] (uint8 / 255 as inference_realbasicvsr.py's float64 division, then float32):

    cleaning: x + conv(64 -> 3)(blocks(x)), blocks = conv 3 -> 64 + LeakyReLU 0.1, 20 residual blocks
        (x + conv(ReLU(conv(x)))), each frame alone; again (at most 3 times) while the mean |residue| of all the
        frames is 1 or more (dynamic_refine_thres 255 / 255: never, in practice)
    SPyNet of each pair of neighbours, both ways: the frames resized (bilinear) to multiples of 32, normalized by
        ImageNet's mean and std, a pyramid of 2 x 2 means; at each of the 6 levels from 1/32 the flow so far
        upsampled (bilinear, align_corners, x 2), the second frame warped by it (border), five 7 x 7 convs (8 ->
        32 -> 64 -> 32 -> 16 -> 2, ReLU) of [frame 1, warped frame 2, flow] added to it; resized back, its x and y
        scaled by w / w_up and h / h_up
    backward propagation from the last frame: the features of the frame after warped (zeros past the image) by
        the flow to it, [frame, features] (67) to conv 67 -> 64 + LeakyReLU 0.1 and 20 residual blocks; then
        forward from the first, each frame's two features fused (1 x 1, 128 -> 64, LeakyReLU), two
        PixelShufflePacks (conv 64 -> 256 + pixel shuffle 2, LeakyReLU), conv_hr + LeakyReLU, conv_last (-> 3),
        + the frame upsampled x 4 (bilinear)
    the output (t, 4h, 4w, 3): clamp to [0, 1], x 255, round (tensor2img); x2: the 4h x 4w output's 2 x 2 means
        first (in float32, then the same)

The programs (one _Runner; the rows of a buffer pixels, its columns channels): ("clean", h, w) of "img" into
"lq" and "res"; ("flow", h, w) of "ref", "supp" into "flow"; ("prop", h, w, "backward"|"forward") of "lq" and
"flow" from its state "feat" to the next (in place: the state stays on the device between calls); ("up", h, w,
x2) of "fb", "ff", "lq" into "y". A conv on the GPU is ENC_CONV_MM (its im2col read through the index; bias,
residual and (Leaky)ReLU in its epilogue), on the CPU ENC_GATHER tiles + ENC_LINEAR, ENC_ADD, ENC_PRELU: the
same bits. The pixel shuffles: the conv's rows of 256 ordered by phase (phase 64 + c), read as rows of 64 by
the next conv's index (seedvr2.vae.up_index). A column's copy into a concatenation: ENC_RESIZE's 1 x 1 means.
The warps ENC_WARP (zeros for the features, border for SPyNet; the grid torch.linspace's plus the flow over
(size - 1) / 2: mmedit's flow_warp normalizes x + flow by 2 / (size - 1) instead, the same point to float32).

A video's whole scene at once (upscale_video), as the official on the whole sequence: the backward features of
each frame are needed by the forward pass in the opposite order, so they are kept (float32, on the host) for
the frames of one segment of `segment` frames at a time: the backward pass keeps those of the first segment
and a checkpoint of the features at every later segment's end, and before each later segment its features are
computed again from its checkpoint (the same bits). segment None: the whole scene when its features fit
FEAT_BYTES, else 32 frames.
"""
from __future__ import annotations

import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.seedvr2.vae import TILE, up_index
from reclank.models.vfi.ref import grid

MODELS = {
    "realbasicvsr": "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c",
    "realbasicvsr-wogan": "realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds_20211027-0e2ff207",
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BLOCKS = 20
FEAT_BYTES = int(float(os.environ.get("RECLANK_RBVSR_FEAT_GB", "16")) * (1 << 30))


def conv_index(h, w, k=3, src=None):
    """The im2col index (h w, k k) of a k x k conv with padding k // 2 over an h x w image; -1 past the edge.
    src (y, x) -> the row of the input pixel (default y w + x)."""
    r = k // 2
    y = np.arange(h)[:, None, None, None] + np.arange(-r, r + 1)[None, None, :, None]
    x = np.arange(w)[None, :, None, None] + np.arange(-r, r + 1)[None, None, None, :]
    ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    yc, xc = np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)
    rows = yc * w + xc if src is None else src(yc, xc)
    return np.ascontiguousarray(np.where(ok, rows, -1).reshape(h * w, k * k), dtype=np.int32)


def flow_size(h, w):
    """SPyNet's working size: the sides up to multiples of 32"""
    return -(-h // 32) * 32, -(-w // 32) * 32


def frames_f32(frames):
    """(t, h, w, 3) float32 of uint8 frames as the official (img / 255. in float64, then float32), or float frames as
    they are (in [0, 1] the trained range: sources of more than 8 bits)"""
    if frames.dtype == np.uint8:
        return (frames.astype(np.float64) / 255.).astype(np.float32)
    return np.asarray(frames, np.float32)


class _B:
    """The forms and named arrays of one program. pool: a float32 array its buffers are taken from (bump; the
    programs of a model that never run at once can share one: a device copy); idx: a dict of index arrays shared with
    other programs (one device copy of each); pool16: a uint16 array for the inputs of the GPU's W16H convs as float16
    (each conv's own, used at once: any programs of one stream can share one)."""

    def __init__(self, gpu, pool=None, idx=None, pool16=None):
        self.gpu = bool(gpu)
        self.c = _EncCompiler(0.0, pack=not gpu)
        self.env = self.c.env
        self.forms = []
        self.n = 0
        self.col = None             # (the CPU's im2col tiles, shared)
        self.tiles = []
        self.idx = {} if idx is None else idx
        self.pool, self.at = pool, 0
        self.pool16 = pool16

    def name(self, arr):
        self.n += 1
        nm = "_r%d" % self.n
        self.env[nm] = arr
        return nm

    def buf(self, rows, cols):
        if self.pool is None:
            return self.name(np.zeros((rows, cols), np.float32))
        n = rows * cols
        assert self.at + n <= self.pool.size, "the pool of %d values: %d more past %d" % (self.pool.size, n, self.at)
        a = self.pool[self.at:self.at + n].reshape(rows, cols)
        self.at += (n + 63) // 64 * 64
        return self.name(a)

    def f(self, *form):
        self.forms.append(form)

    def index(self, key, make):
        if key not in self.idx:
            self.idx[key] = make()
        return self.idx[key]

    def conv(self, x, cin, lin, idx, out=None, slope=None, res=None):
        """the conv of im2col idx over x (rows of cin), + res, then PReLU of slope (a value: 0 is ReLU) when given;
        on the GPU lin's 16-bit rows (lin.wq: W16H or W16I) on the tensor cores when it has them (W16H with pool16:
        x's rows packed to float16 first, the conversion the gather makes: the same bits, read half the bytes)"""
        rows = idx.shape[0]
        m = lin.shape[0]
        out = self.buf(rows, m) if out is None else out
        sl = None if slope is None else np.full(m, slope, np.float32)
        if self.gpu:
            q = getattr(lin, "wq", None)
            wbf = 0 if q is None else (5 if q.type == 91 else 3)
            xa = self.env[x]
            if wbf == 3 and self.pool16 is not None and xa.size <= self.pool16.size:
                xh = self.name(self.pool16[:xa.size].reshape(xa.shape))
                self.f("set", xh, ("enc_bf16_pack", x, 1))
                x, wbf = xh, 4
            self.f("set", out, ("enc_conv_mm", lin, x, idx, cin, sl, res, None, 0, wbf))
            return out
        cols = idx.shape[1] * cin
        tr = max(256, min(rows, TILE * 9 * 64 // cols))
        for r0 in range(0, rows, tr):
            r1 = min(rows, r0 + tr)
            col = self.name(None)
            ot = self.name(self.env[out][r0:r1])
            self.tiles.append((col, r1 - r0, cols))
            self.f("set", col, ("enc_gather", x, self.name(idx[r0:r1]), cin))
            self.f("set", ot, ("enc_linear", lin, col))
        if res is not None:
            self.f("enc_add", out, res)
        if sl is not None:
            self.f("set", out, ("enc_prelu", out, sl))
        return out

    def copy_cols(self, x, cols_in, out, cols_out, at, rows, h, w):
        """x's rows (h w of cols_in) into the columns at.. of out (rows of cols_out): ENC_RESIZE's 1 x 1 means"""
        self.f("enc_resize", x, out, rows // (h * w), h, w, h, w, cols_in, 2, cols_in, 0, cols_out, at)

    def finish(self, threads=None):
        if self.tiles:
            need = max(r * c for _, r, c in self.tiles)
            self.col = np.zeros(need, np.float32)
            for nm, r, c in self.tiles:
                self.env[nm] = self.col[:r * c].reshape(r, c)
        self.c.compile(("seq", *self.forms))
        p = self.c.p.finish()
        if not self.gpu:
            from reclank.runtime.threads import team_threads
            p.threads = p.buf[3] = threads or team_threads()
        return p


class RealBasicVSR:
    """A model of MODELS from its directory (the generator_ema of the official checkpoint, converted by
    scripts/pt_to_safetensors.py --key state_dict --prefix generator_ema.). upscale_video(frames) gives the x4 (or
    x2) output of a video."""

    def __init__(self, path, model_id="realbasicvsr", gpu=False, cpu_threads=None):
        R = SafeTensors(os.path.join(path, MODELS[model_id] + ".safetensors"))
        self.W = {k: np.array(R.get(k), dtype=np.float32) for k in R.names()}       # copies: the file closes
        R.close()
        self.model_id, self.gpu, self.cpu_threads = model_id, bool(gpu), cpu_threads
        self.runner = _Runner(self._build, keep=8, keep_gpu=8)

    # ---- weights -------------------------------------------------------------------------------------

    def lin(self, name, perm=None):
        """the conv name (out, in, k, k) as the linear of its im2col: W (out, k k in), the columns (ky, kx, c);
        perm: its rows in that order"""
        w = self.W[name + ".weight"]
        W = w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1)
        b = self.W[name + ".bias"]
        if perm is not None:
            W, b = W[perm], b[perm]
        out = _Plain(np.ascontiguousarray(W, np.float32))
        out.b = np.ascontiguousarray(b, np.float32)
        out.shape = out.w.shape
        return out

    # ---- programs ------------------------------------------------------------------------------------

    def _build(self, key, gpu):
        B = _B(gpu)
        getattr(self, "_" + key[0])(B, *key[1:])
        return B.finish(self.cpu_threads)

    def _blocks(self, B, x, cin, prefix, h, w):
        """ResidualBlocksWithInputConv(cin, 64, 20) of x (h w, cin): conv + LeakyReLU 0.1, the residual blocks"""
        i3 = B.index(("c3", h, w), lambda: conv_index(h, w))
        f = B.conv(x, cin, self.lin(prefix + ".main.0"), i3, slope=0.1)
        t = B.buf(h * w, 64)
        g = B.buf(h * w, 64)
        for k in range(BLOCKS):
            p = "%s.main.2.%d." % (prefix, k)
            B.conv(f, 64, self.lin(p + "conv1"), i3, out=t, slope=0.0)
            B.conv(t, 64, self.lin(p + "conv2"), i3, out=g, res=f)
            f, g = g, f
        return f

    def _clean(self, B, h, w):
        P = h * w
        B.env["img"] = np.zeros((P, 3), np.float32)
        B.env["lq"] = np.zeros((P, 3), np.float32)
        B.env["res"] = np.zeros((P, 3), np.float32)
        f = self._blocks(B, "img", 3, "image_cleaning.0", h, w)
        B.conv(f, 64, self.lin("image_cleaning.1"), B.index(("c3", h, w), lambda: conv_index(h, w)), out="res")
        B.f("set", "lq", ("copy", "img"))
        B.f("enc_add", "lq", "res")

    def _flow(self, B, h, w):
        P = h * w
        hu, wu = flow_size(h, w)
        for nm in ("ref", "supp"):
            B.env[nm] = np.zeros((P, 3), np.float32)
        B.env["flow"] = np.zeros((P, 2), np.float32)
        sizes = [(hu >> s, wu >> s) for s in range(6)]          # full, 1/2, .. 1/32
        cat = [B.buf(a * b, 8) for a, b in sizes]               # [ref, warped supp, flow up] of each level
        sup = [B.buf(a * b, 3) for a, b in sizes]
        mean_n = B.name(-MEAN)
        inv = B.name(np.float32(1) / STD)
        for nm, dst in (("ref", None), ("supp", sup[0])):
            x = nm
            if (hu, wu) != (h, w):
                x = B.buf(hu * wu, 3)
                B.f("enc_resize", nm, x, 1, h, w, hu, wu, 3, 0)
            n = B.buf(hu * wu, 3)
            B.f("set", n, ("copy", x))
            B.f("enc_bias", n, mean_n)
            B.f("set", n, ("enc_mul_vec", n, inv))
            if dst is None:
                B.copy_cols(n, 3, cat[0], 8, 0, hu * wu, hu, wu)
            else:
                B.f("set", dst, ("copy", n))
        for s in range(1, 6):
            (a, b), (A, Bw) = sizes[s - 1], sizes[s]
            B.f("enc_resize", cat[s - 1], cat[s], 1, a, b, A, Bw, 3, 2, 8, 0, 8, 0)
            B.f("enc_resize", sup[s - 1], sup[s], 1, a, b, A, Bw, 3, 2)
        flow = None
        two = B.name(np.full(2, 2.0, np.float32))
        for lvl in range(6):
            s = 5 - lvl
            a, b = sizes[s]
            up = B.buf(a * b, 2)
            if flow is not None:
                B.f("enc_resize", flow, up, 1, a // 2, b // 2, a, b, 2, 1)
                B.f("set", up, ("enc_mul_vec", up, two))
            B.f("enc_warp", sup[s], up, cat[s], grid(b), grid(a), a, b, 3, 3, 0, 2, 0, 8, 3)
            B.copy_cols(up, 2, cat[s], 8, 6, a * b, a, b)
            i7 = B.index(("c7", a, b), lambda: conv_index(a, b, 7))
            p = "basicvsr.spynet.basic_module.%d.basic_module." % lvl
            x = B.conv(cat[s], 8, self.lin(p + "0.conv"), i7, slope=0.0)
            x = B.conv(x, 32, self.lin(p + "1.conv"), i7, slope=0.0)
            x = B.conv(x, 64, self.lin(p + "2.conv"), i7, slope=0.0)
            x = B.conv(x, 32, self.lin(p + "3.conv"), i7, slope=0.0)
            flow = B.conv(x, 16, self.lin(p + "4.conv"), i7, res=up)
        if (hu, wu) != (h, w):
            B.f("enc_resize", flow, "flow", 1, hu, wu, h, w, 2, 0)
            B.f("set", "flow", ("enc_mul_vec", "flow", B.name(np.array([w / wu, h / hu], np.float32))))
        else:
            B.f("set", "flow", ("copy", flow))

    def _prop(self, B, h, w, direction):
        P = h * w
        for nm, c in (("lq", 3), ("flow", 2), ("feat", 64)):
            B.env[nm] = np.zeros((P, c), np.float32)
        cat = B.buf(P, 67)
        B.copy_cols("lq", 3, cat, 67, 0, P, h, w)
        B.f("enc_warp", "feat", "flow", cat, grid(w), grid(h), h, w, 64, 64, 0, 2, 0, 67, 3, True)
        f = self._blocks(B, cat, 67, "basicvsr.%s_resblocks" % direction, h, w)
        B.f("set", "feat", ("copy", f))

    def _up(self, B, h, w, x2):
        P = h * w
        for nm, c in (("fb", 64), ("ff", 64), ("lq", 3)):
            B.env[nm] = np.zeros((P, c), np.float32)
        H, W_ = 4 * h, 4 * w
        B.env["y"] = np.zeros((P * (4 if x2 else 16), 3), np.float32)
        cat = B.buf(P, 128)
        B.copy_cols("fb", 64, cat, 128, 0, P, h, w)
        B.copy_cols("ff", 64, cat, 128, 64, P, h, w)
        i1 = B.index(("c1", h, w), lambda: np.arange(P, dtype=np.int32).reshape(P, 1))
        x = B.conv(cat, 128, self.lin("basicvsr.fusion"), i1, slope=0.1)
        # (PixelShufflePack: the conv's channel 4 c + 2 i + j to its rows by phase: 64 (2 i + j) + c)
        perm = np.arange(256).reshape(64, 4).T.reshape(-1)
        x = B.conv(x, 64, self.lin("basicvsr.upsample1.upsample_conv", perm),
                   B.index(("c3", h, w), lambda: conv_index(h, w)), slope=0.1)
        x = B.name(B.env[x].reshape(-1, 64))           # (rows of 64: a phase's channels of a pixel)
        x = B.conv(x, 64, self.lin("basicvsr.upsample2.upsample_conv", perm),
                   B.index(("u", h, w), lambda: up_index(h, w)), slope=0.1)
        x = B.name(B.env[x].reshape(-1, 64))
        x = B.conv(x, 64, self.lin("basicvsr.conv_hr"), B.index(("u", 2 * h, 2 * w), lambda: up_index(2 * h, 2 * w)),
                   slope=0.1)
        base = B.buf(H * W_, 3)
        B.f("enc_resize", "lq", base, 1, h, w, H, W_, 3, 0)
        y = B.conv(x, 64, self.lin("basicvsr.conv_last"), B.index(("c3", H, W_), lambda: conv_index(H, W_)), res=base)
        if x2:
            B.f("enc_resize", y, "y", 1, H, W_, 2 * h, 2 * w, 3, 2)
        else:
            B.f("set", "y", ("copy", y))

    # ---- running -------------------------------------------------------------------------------------

    def _call(self, key, inputs, outputs, keep=()):
        """the program of key with inputs (name: array) written; copies of outputs. keep: the names whose
        device copies are left as they are (the state of a "prop")"""
        prog, g = self.runner.get(key, self.gpu)
        for nm, a in inputs.items():
            prog.names[nm][...] = a
            if g is not None:
                g.upload(nm)
        if g is not None:
            g.run()
            for nm in outputs:
                g.download(nm)
        else:
            prog.run()
        return [prog.names[nm].copy() for nm in outputs]

    def clean(self, x):
        """the cleaning of the frames x (t, h w, 3) float32: (lq, mean |residue| of all)"""
        t, P, _ = x.shape
        lq, tot = np.empty_like(x), 0.0
        for i in range(t):
            lq[i], res = self._call(("clean",) + self.hw, {"img": x[i]}, ("lq", "res"))
            tot += float(np.abs(res, dtype=np.float64).sum())
        return lq, tot / x.size

    def flow(self, ref, supp):
        return self._call(("flow",) + self.hw, {"ref": ref, "supp": supp}, ("flow",))[0]

    def _prop_reset(self, direction, feat=None):
        key = ("prop",) + self.hw + (direction,)
        prog, g = self.runner.get(key, self.gpu)
        prog.names["feat"][...] = 0.0 if feat is None else feat
        if g is not None:
            g.upload("feat")

    def _prop_step(self, direction, lq, flow, out=True):
        """one step of the propagation from its state; the new state's copy when out"""
        key = ("prop",) + self.hw + (direction,)
        prog, g = self.runner.get(key, self.gpu)
        prog.names["lq"][...] = lq
        prog.names["flow"][...] = flow
        if g is not None:
            g.upload("lq")
            g.upload("flow")
            g.run()
            if out:
                g.download("feat")
        else:
            prog.run()
        return prog.names["feat"].copy() if out else None

    def upscale_video(self, frames, x2=False, segment=None, progress=None, parts=None):
        """The output (t, 4h, 4w, 3) uint8 (x2: (t, 2h, 2w, 3), the 2 x 2 means of the x4 before the rounding) of
        a video's frames (t, h, w, 3): uint8, or float32 in [0, 1] (frames_f32). progress(stage, done, total).
        parts: a dict that gets clean, flow_b, flow_f, fb, ff (frame order) and y (float32, before the clamp)."""
        x = frames_f32(frames)
        t, h, w, _ = x.shape
        assert h >= 64 and w >= 64, "RealBasicVSR: frames of at least 64 x 64"
        self.hw = (h, w)
        P = h * w
        note = progress or (lambda *a: None)
        lq = x.reshape(t, P, 3)
        for k in range(3):                                      # (at most 3 cleanings, as the official)
            lq, m = self.clean(lq)
            note("clean", k + 1, 3)
            if m < 1.0:
                break
        zf = np.zeros((P, 2), np.float32)
        flow_b = [self.flow(lq[i], lq[i + 1]) for i in range(t - 1)]
        note("flow", 1, 2)
        mirror = t % 2 == 0 and np.array_equal(lq[:t // 2], lq[t // 2:][::-1])
        flow_f = None if mirror else [self.flow(lq[i + 1], lq[i]) for i in range(t - 1)]
        note("flow", 2, 2)
        fbytes = P * 64 * 4
        seg = segment or (t if t * fbytes <= FEAT_BYTES else 32)
        seg = max(1, min(seg, t))
        ends = list(range(seg, t, seg))                         # the later segments' starts: their checkpoints
        ckpt, fb = {}, {}
        self._prop_reset("backward")
        for i in range(t - 1, -1, -1):
            keep = i < seg or i in ckpt or i in ends
            f = self._prop_step("backward", lq[i], flow_b[i] if i < t - 1 else zf, out=keep)
            if i < seg:
                fb[i] = f
            elif i in ends:
                ckpt[i] = f
            note("backward", t - i, t)
        out = np.empty((t, (2 if x2 else 4) * h, (2 if x2 else 4) * w, 3), np.uint8)
        keep = {} if parts is None else {"fb": [None] * t, "ff": [None] * t, "y": [None] * t}
        self._prop_reset("forward")
        for s0 in range(0, t, seg):
            s1 = min(s0 + seg, t)
            if s0 > 0:                                          # (the segment's backward features again)
                fb = {}
                self._prop_reset("backward", ckpt[s1] if s1 < t else None)
                for i in range(s1 - 1, s0 - 1, -1):
                    fb[i] = self._prop_step("backward", lq[i], flow_b[i] if i < t - 1 else zf)
            for i in range(s0, s1):
                fl = zf if i == 0 else (flow_f[i - 1] if flow_f is not None else flow_b[t - 1 - i])
                ff = self._prop_step("forward", lq[i], fl)
                y = self._call(("up",) + self.hw + (bool(x2),), {"fb": fb[i], "ff": ff, "lq": lq[i]}, ("y",))[0]
                hh, ww = (2 * h, 2 * w) if x2 else (4 * h, 4 * w)
                y = y.reshape(hh, ww, 3)
                out[i] = np.round(np.clip(y, 0, 1) * np.float32(255)).astype(np.uint8)
                if parts is not None:
                    keep["fb"][i], keep["ff"][i], keep["y"][i] = fb[i], ff, y
                note("forward", i + 1, t)
            fb = {}
        if parts is not None:
            parts.update(clean=lq, flow_b=np.stack(flow_b) if t > 1 else None,
                         flow_f=np.stack(flow_f) if flow_f is not None and t > 1 else None,
                         fb=np.stack(keep["fb"]), ff=np.stack(keep["ff"]), y=np.stack(keep["y"]))
        return out

    def release(self):
        if self.gpu:
            self.runner.release(keep_host=True)
