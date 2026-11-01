"""Ported from Practical-RIFE (the v4.25 model's train_log/IFNet_HDv3.py and model/warplayer.py; MIT
License; Copyright (c) 2021 hzwer). See THIRD_PARTY_NOTICES.md.

Practical-RIFE 4.25 (hzwer, MIT): IFNet_HDv3 as reclank programs, float32 as stored, on the CPU
and the GPU (docs/VFI_PLAN.md).

The network (the authors' IFNet_HDv3.py, warplayer.py, inference_video.py), frames RGB in [0, 1]
zero-padded at the right and bottom to a multiple of max(64, 64 / scale):

    f = Head(img) of each frame: conv 3x3 s2 3 -> 16, conv 16 -> 16 twice, LeakyReLU 0.2,
        ConvTranspose 4x4 s2 16 -> 4 (independent of the pair: once a frame)
    five IFBlocks at scales 16, 8, 4, 2, 1 (over scale: 0.5 for 4K), c = 192, 128, 96, 64, 32:
        the input resized bilinear to 1/s, conv 3x3 s2 (c/2), conv 3x3 s2 (c), 8 ResConvs
        (LeakyReLU(conv(x) beta + x)), ConvTranspose 4x4 s2 (c -> 52) + PixelShuffle(2): 13
        channels at 1/s resized back: the flow (x s; two 2-D flows), the mask, 8 features
        block 0's input: img0, img1, f0, f1, t; blocks 1-4: img0 and img1 warped by the flow, f0
        and f1 warped, t, the mask, the features, then the flow over s; the flow accumulates
    out = w0 sigmoid(mask) + w1 (1 - sigmoid(mask)), w0 and w1 the frames warped by the last flow

The programs (one _Runner, so they share the device copies of the frame slots):

    ("head", Hp, Wp, shift): slot 1 to slot 0 (shift), the new frame's bytes (ENC_UNPACK_U8) into
        slot 1 and its Head
    ("net", Hp, Wp, scale): the frame at t between slot 0 and slot 1 (t in a bias vector,
        uploaded a call), its bytes (ENC_PACK_U8)

A buffer of rows is pixels and its columns channels. A: the full-resolution concatenation (w0 0:3,
w1 3:6, wf0 6:10, wf1 10:14, t 14, mask 15, feat 16:24, flow 24:28) written in place by ENC_WARP and
ENC_RESIZE through their column offsets; each block's input ENC_RESIZE'd from it (block 0: from the
slots), then ENC_MUL_VEC and ENC_BIAS_CLAMP put t into column 14 and divide the flow by s. 3x3 convs
are ENC_GATHER im2col tiles + ENC_LINEAR; a ConvTranspose(4, 2, 1) is a 3x3 im2col of its input, a
linear to its 4 output phases (kernel rows 3, 1 / 2, 0 of a phase), then one ENC_GATHER of the
phases (and of the PixelShuffle); ResConv's beta is folded into its conv; LeakyReLU is ENC_PRELU;
the warp ENC_WARP (torch's grid_sample: bilinear, border, align_corners); the merge ENC_MIX. That is
the CPU's form (and the GPU's with RECLANK_RIFE_IM2COL=1).

The GPU's form (docs/VFI_PLAN.md section 6.3; the same bits): each conv one ENC_CONV_MM (the
products read the im2col rows through the index; the bias, the ResConv's residual and LeakyReLU in
its epilogue; a deconv's phases written to their pixels by its scatter, so no gather and no copy);
no buffer A: each block's input made by ENC_RIFE_IN from the frames, the heads and the earlier
blocks' tmp (kept at their own resolution), the output (and a probe's warped frames) by
ENC_RIFE_OUT. The frames go in and out as their bytes (gpu_io): ("head", Hp, Wp, shift, h, w, aux)
unpacks the h x w frame into the padded slot (ENC_UNPACK_U8P) and, with aux_plan(), makes its
thumbnails' inputs (ENC_THUMB_U8) and its probe frame (PIL's BOX, ENC_RESAMPLE_U8) for aux();
("net", Hp, Wp, scale, h, w) packs the h x w frame (ENC_PACK_U8C).

Float frames (push() of a float32 frame, frame_f32()): the same network without the bytes, for
sources of more than 8 bits. ("headf", Hp, Wp, shift): the padded frame (its values as given, in
[0, 1] for the trained range, not clamped) uploaded into xin and copied to slot 1, then the Head;
("netf", Hp, Wp, scale): the frame at t left in out (float32, not clamped or rounded). Fed u8 / 255,
the output is the byte path's before its rounding.
"""
from __future__ import annotations

import os

import numpy as np

from reclank.encoders.runtime import _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.depth.common import TILE, Builder, conv_index, lin
from reclank.models.depth.stereo import as_words
from reclank.models.vfi import ref

MODEL_ID = "rife-4.25"
A_COLS = 28
CHANNELS = (192, 128, 96, 64, 32)
SLOPE = np.full(256, 0.2, np.float32)
# ConvTranspose2d(k 4, s 2, p 1) as a 3 x 3 window of its input: output phase a (the parity of the row)
# reads the window's tap d (-1, 0, 1) with the kernel row KY[a][d + 1] (None: none)
KY = ((3, 1, None), (None, 2, 0))


def weights_path(path):
    """flownet.safetensors in path (a file, the 4.25 folder, or the Practical-RIFE folder)"""
    for p in (path, os.path.join(path, "flownet.safetensors"), os.path.join(path, "4.25", "flownet.safetensors")):
        if os.path.isfile(p) and p.endswith(".safetensors"):
            return p
    raise FileNotFoundError("%s: no flownet.safetensors (scripts/pt_to_safetensors.py 4.25/flownet.pkl makes it)"
                            % path)


def auto_scale(h, w):
    """the authors' advice: 0.5 for 4K (above 2560 x 1440 of area), else 1"""
    return 0.5 if h * w > 2560 * 1440 else 1.0


def padded(h, w, scale=1.0, pad=64):
    """the padded size: a multiple of max(pad, pad / scale) (the network's minimum is 64 at scale 1)"""
    m = max(pad, int(pad / scale))
    return ((h - 1) // m + 1) * m, ((w - 1) // m + 1) * m


def mean_abs(a, b):
    """float(np.abs(a - b).mean(-1).mean()) of two (h, w, 3) float32 arrays, the same bits (the channels'
    sum as NumPy's reduction of 3 values adds them, (d0 + d1) + d2, then / 3), without that slow
    reduction"""
    d = np.abs(a - b)
    return float((((d[..., 0] + d[..., 1]) + d[..., 2]) / np.float32(3)).mean())


def deconv_lin(wt, b, shuffle):
    """ConvTranspose2d(4, 2, 1) weights wt (cin, cout, 4, 4) and bias b as the linear of a 3 x 3
    im2col: 4 cout rows a pixel, the phases (a, b) of the output's row and column; shuffle (RIFE's
    PixelShuffle(2) after it): each phase's rows ordered (i, j, c) with the channel c 4 + i 2 + j,
    so that a final pixel is one block of cout / 4 values."""
    cin, cout = wt.shape[:2]
    if shuffle:
        c_, i_, j_ = np.meshgrid(np.arange(cout // 4), np.arange(2), np.arange(2), indexing="ij")
        order = (c_ * 4 + i_ * 2 + j_).transpose(1, 2, 0).reshape(-1)
    else:
        order = np.arange(cout)
    M = np.zeros((2, 2, cout, 3, 3, cin), np.float32)
    for a in range(2):
        for bb in range(2):
            for dy in range(3):
                for dx in range(3):
                    ky, kx = KY[a][dy], KY[bb][dx]
                    if ky is not None and kx is not None:
                        M[a, bb, :, dy, dx, :] = wt[:, order, ky, kx].T
    return lin(M.reshape(4 * cout, 9 * cin), np.tile(np.asarray(b, np.float32)[order], 4))


def deconv_index(h, w, shuffle):
    """the gather of a deconv's phase rows (h w 4 [x 4] blocks) into the output's pixel rows"""
    if shuffle:
        Y, X = np.meshgrid(np.arange(4 * h), np.arange(4 * w), indexing="ij")
        sub = (((Y // 2) % 2) * 2 + (X // 2) % 2) * 4 + (Y % 2) * 2 + X % 2
        idx = ((Y // 4) * w + X // 4) * 16 + sub
    else:
        Y, X = np.meshgrid(np.arange(2 * h), np.arange(2 * w), indexing="ij")
        idx = ((Y // 2) * w + X // 2) * 4 + (Y % 2) * 2 + X % 2
    return np.ascontiguousarray(idx.reshape(-1, 1), dtype=np.int32)


def deconv_scatter(h, w, shuffle):
    """the inverse of deconv_index (a permutation): phase row j (of h w 4 [x 4] blocks) to its output row"""
    j = np.arange(h * w * (16 if shuffle else 4), dtype=np.int64)
    if shuffle:
        pix, hi, lo = j // 16, (j % 16) // 4, j % 4
        Y = 4 * (pix // w) + 2 * (hi // 2) + lo // 2
        X = 4 * (pix % w) + 2 * (hi % 2) + lo % 2
        return np.ascontiguousarray(Y * 4 * w + X, dtype=np.int32)
    pix, ph = j // 4, j % 4
    return np.ascontiguousarray((2 * (pix // w) + ph // 2) * 2 * w + 2 * (pix % w) + ph % 2, dtype=np.int32)


class RIFE:
    """Practical-RIFE 4.25 of a folder (its flownet.safetensors). push() a frame, then frame(t) of the
    pair of the last two pushed; the frame size is fixed until reset()."""

    def __init__(self, path, gpu=False, cpu_threads=None, pad=64):
        self.path = path
        self.W = SafeTensors(weights_path(path))
        self._prober = None         # the small RIFE of the scene-cut test (prober())
        self.gpu, self.cpu_threads, self.pad = bool(gpu), cpu_threads, pad
        self.runner = _Runner(self._build, keep=6, keep_gpu=4)
        self.runner.label = MODEL_ID
        self.slots = {}             # (Hp, Wp): the host arrays the programs of that size share
        self.size = None            # (h, w, Hp, Wp, scale) of the frames pushed
        self.n = 0                  # the frames pushed since reset()
        self.taps = False           # (a check) the net programs keep each block's tmp and flow
        self.probe = False          # the net programs keep the two frames warped to t (w0, w1: the cut probe)
        # the GPU's convs by ENC_CONV_MM (no im2col buffer; its epilogue the bias, the ResConv's residual
        # and LeakyReLU: the same bits as ENC_GATHER + ENC_LINEAR + ENC_ADD + ENC_PRELU); RECLANK_RIFE_IM2COL=1:
        # the im2col records
        self.conv_mm = os.environ.get("RECLANK_RIFE_IM2COL", "0") != "1"
        self.scale = None           # RIFE's scale (1, 0.5); None: auto_scale of the frame's size
        self.io = {}                # (h, w): the GPU's frame bytes and outputs of that frame size
        self.plans = {}             # (h, w): (thumb, probe) plans: its head programs give aux() (aux_plan())

    def runners(self):
        out = {"net": self.runner} if self.gpu else {}
        if self.gpu and self._prober is not None:
            out["probe"] = self._prober.runner
        return out

    def prober(self):
        """the scene-cut test's model: RIFE on the pairs' probe frames (video.probe_frame), keeping the
        frames warped to t 0.5 (w0, w1)"""
        if self._prober is None:
            p = RIFE(self.path, gpu=self.gpu, cpu_threads=self.cpu_threads, pad=self.pad)
            p.probe = True
            p.conv_mm = self.conv_mm
            p.runner.label = MODEL_ID + "/probe"
            self._prober = p
        return self._prober

    def residual(self):
        """(R, D) of the last two frames pushed (a probe model): the mean |w0 - w1| of the two frames
        warped to t 0.5 and the mean |x0 - x1| of the frames, in levels of 255 over the frame"""
        self.frame(0.5, fetch=False)
        h, w, Hp, Wp, _ = self.size
        w0, w1 = self.warped()
        o = self.outputs(["x0", "x1"])
        return mean_abs(w0, w1) * 255, mean_abs(*(o[n].reshape(Hp, Wp, 3)[:h, :w] for n in ("x0", "x1"))) * 255

    def warped(self):
        """(a probe model, after frame()) the two frames warped to t: float32 (h, w, 3) views of the
        program's arrays (until its next run)"""
        h, w, Hp, Wp, s = self.size
        prog, g = self.runner.get(("net", Hp, Wp, s) + ((h, w) if self.gpu_io else ()), self.gpu)
        if g is not None:
            g.download("w0")
            g.download("w1")
        return tuple(prog.names[n].reshape(Hp, Wp, 3)[:h, :w] for n in ("w0", "w1"))

    def g(self, name):
        return np.asarray(self.W.get("module." + name), np.float32)

    def conv_w(self, name, beta=None):
        w, b = self.g(name + ".weight"), self.g(name + ".bias")
        w = w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1)
        if beta is not None:
            beta = beta.reshape(-1)
            w, b = w * beta[:, None], b * beta
        return lin(w, b)

    # ---- the programs ----

    def _arrays(self, Hp, Wp):
        key = (Hp, Wp)
        if key not in self.slots:
            N = Hp * Wp
            self.slots[key] = dict(img8=np.zeros((N * 3 + 3) // 4, np.float32),
                                   x0=np.zeros((N, 3), np.float32), x1=np.zeros((N, 3), np.float32),
                                   f0=np.zeros((N, 4), np.float32), f1=np.zeros((N, 4), np.float32),
                                   tb15=np.zeros(15, np.float32), tb28=np.zeros(A_COLS, np.float32),
                                   out8=np.zeros((N * 3 + 3) // 4, np.float32))
        return self.slots[key]

    def _xin(self, Hp, Wp):
        """the host array of a float frame (padded, (Hp Wp, 3)): headf's input"""
        key = ("xin", Hp, Wp)
        if key not in self.slots:
            self.slots[key] = np.zeros((Hp * Wp, 3), np.float32)
        return self.slots[key]

    def _io(self, h, w):
        """the GPU's byte buffers of an h x w frame: its bytes in (raw8), the output frame's (out8c),
        the thumbnails' taps and block sums and the probe frame (aux_plan())"""
        if (h, w) not in self.io:
            words = lambda n: np.zeros((n + 3) // 4, np.float32)     # noqa: E731 - bytes as float32 words
            a = dict(raw8=words(h * w * 3), out8c=words(h * w * 3))
            if (h, w) in self.plans:
                tp, pp = self.plans[(h, w)]
                a.update(taps8=words(tp["rows"].size * tp["cols"].size * 3), sums=np.zeros(48 * 64, np.int32),
                         probe8=words(pp["W"] * pp["H"] * 3))
                if pp["horiz"] is not None and pp["vert"] is not None:
                    a["ptmp8"] = words((pp["horiz"][3] - pp["horiz"][2]) * pp["W"] * 3)
            self.io[(h, w)] = a
        return self.io[(h, w)]

    def _build(self, key, gpu):
        kind, Hp, Wp = key[:3]
        B = Builder(gpu, kmax=9 * 192)
        # the im2col tile of the widest conv input (the Head's 16 channels, the blocks' 192)
        B.col = np.zeros(TILE * 9 * (16 if kind == "head" else 192), np.float32)
        for k, v in self._arrays(Hp, Wp).items():
            B.array(k, v)
        io = key[4:6] if len(key) > 4 else None             # (h, w): the GPU's frame bytes (gpu_io)
        if io is not None:
            for k, v in self._io(*io).items():
                B.array(k, v)
        f32 = kind in ("headf", "netf")                     # (float frames: no bytes)
        if kind == "headf":
            B.array("xin", self._xin(Hp, Wp))
        if kind in ("head", "headf"):
            self._head(B, Hp, Wp, key[3], io, len(key) > 6 and key[6], f32=f32)
        else:
            self._net(B, Hp, Wp, key[3], io, f32=f32)
        return B.finish(self.cpu_threads)

    @property
    def gpu_io(self):
        """the GPU's frames in and out as their bytes (the padding, the crop, the thumbnails' inputs and
        the probe frame on the GPU: the same bits as the host's)"""
        return self.gpu and self.conv_mm

    def aux_plan(self, h, w, thumb, probe):
        """(the GPU) the head programs of h x w frames give the frame's thumbnail inputs (video.thumb_plan:
        its taps and block sums, ENC_THUMB_U8) and its probe frame (video.probe_plan: PIL's BOX,
        ENC_RESAMPLE_U8) to aux() after each push() (the plans of a size: set once)"""
        if (h, w) not in self.plans:
            self.plans[(h, w)] = (thumb, probe)
            if (h, w) in self.io:       # (the size's buffers kept: its programs read them)
                old = self.io.pop((h, w))
                self.io[(h, w)] = dict(self._io(h, w), raw8=old["raw8"], out8c=old["out8c"])

    def _conv(self, B, x, idx, lin, act=False, res=None, out=None):
        """a 3 x 3 conv (idx: its im2col index), then + res, then LeakyReLU 0.2 (act)"""
        if B.gpu and self.conv_mm:
            m = lin.shape[0]
            out = B.buf(idx.shape[0], m) if out is None else out
            tail = (SLOPE[:m],) if act else ()
            if res is not None:
                tail = (SLOPE[:m] if act else None, res)
            B.f("set", out, ("enc_conv_mm", lin, x, idx, B.shape(x)[1]) + tail)
            return out
        out = B.conv3(x, idx, lin, out=out)
        if res is not None:
            B.add(out, res)
        return self._lrelu(B, out, out=out) if act else out

    def _lrelu(self, B, x, out=None):
        cols = B.shape(x)[1]
        out = B.buf(*B.shape(x)) if out is None else out
        B.f("set", out, ("enc_prelu", x, SLOPE[:cols]))
        return out

    def _deconv(self, B, x, h, w, name, shuffle, out=None):
        wt = self.g(name + ".weight")
        cout = wt.shape[1]
        i3 = B.index(("c3", h, w), lambda: conv_index(h, w))
        dl = deconv_lin(wt, self.g(name + ".bias"), shuffle)
        blk = cout // 4 if shuffle else cout
        k = 4 if shuffle else 2
        out = B.buf(k * h * k * w, blk) if out is None else out
        if B.gpu and self.conv_mm:
            # (the products' phase rows written to their pixels by ENC_CONV_MM: the gather's bits, no buffer)
            scat = B.index(("dcs", h, w, shuffle), lambda: deconv_scatter(h, w, shuffle))
            B.f("set", out, ("enc_conv_mm", dl, x, i3, B.shape(x)[1], None, None, scat, blk))
            return out
        idx = B.index(("dc", h, w, shuffle), lambda: deconv_index(h, w, shuffle))
        y = self._conv(B, x, i3, dl)
        B.f("set", out, ("enc_gather", B.name(B.env[y].reshape(-1, blk)), idx, blk))
        B.free(y)
        return out

    def _head(self, B, Hp, Wp, shift, io=None, aux=False, f32=False):
        if shift:
            B.f("set", "x0", ("copy", "x1"))
            B.f("set", "f0", ("copy", "f1"))
        if f32:
            B.f("set", "x1", ("copy", "xin"))
        elif io is None:
            B.f("enc_unpack_u8", "img8", "x1", 255.0)
        else:
            h, w = io
            B.f("enc_unpack_u8p", "raw8", "x1", h, w, Hp, Wp, 3, 255.0)
            if aux:
                self._aux(B, h, w)
        h2, w2 = (Hp - 1) // 2 + 1, (Wp - 1) // 2 + 1
        x = self._conv(B, "x1", conv_index(Hp, Wp, stride=2), self.conv_w("encode.cnn0"), act=True)
        i1 = conv_index(h2, w2)
        y = self._conv(B, x, i1, self.conv_w("encode.cnn1"), act=True)
        B.free(x)
        x = self._conv(B, y, i1, self.conv_w("encode.cnn2"), act=True)
        B.free(y)
        if B.gpu and self.conv_mm:
            self._deconv(B, x, h2, w2, "encode.cnn3", shuffle=False, out="f1")
        else:
            f = self._deconv(B, x, h2, w2, "encode.cnn3", shuffle=False)
            B.f("set", "f1", ("copy", f))

    def _aux(self, B, h, w):
        """the thumbnails' inputs and the probe frame of raw8 (aux_plan)"""
        tp, pp = self.plans[(h, w)]
        B.f("enc_thumb_u8", "raw8", "taps8", tp["rows"], tp["cols"], "sums", tp["by0"], tp["by1"], tp["bx0"],
            tp["bx1"], w)
        W, H, src, sw = pp["W"], pp["H"], "raw8", w
        if pp["horiz"] is not None:
            bh, kh, y0, y1 = pp["horiz"]
            dst = "ptmp8" if pp["vert"] is not None else "probe8"
            B.f("enc_resample_u8", "raw8", dst, bh, kh, y1 - y0, W, 3, w * 3, 3, y0 * w * 3)
            src, sw = dst, W
        if pp["vert"] is not None:
            B.f("enc_resample_u8", src, "probe8", pp["vert"][0], pp["vert"][1], 1, H, sw * 3, 0, sw * 3)
        elif pp["horiz"] is None:
            B.f("set", "probe8", ("copy", "raw8"))

    def _net(self, B, Hp, Wp, scale, io=None, f32=False):
        N = Hp * Wp
        # fused (the GPU's default): no full-resolution buffer A; each block's input from the frames,
        # the heads and the blocks' tmp by ENC_RIFE_IN, the output by ENC_RIFE_OUT (the same bits)
        fused = B.gpu and self.conv_mm and not self.taps
        if not fused:
            A = B.array("A", np.zeros((N, A_COLS), np.float32))
            F = B.array("flow", np.zeros((N, 4), np.float32))
            D = B.array("dflow", np.zeros((N, 4), np.float32))
        gx, gy = ref.grid(Wp), ref.grid(Hp)
        dx, dy = float(np.float32((Wp - 1.0) / 2.0)), float(np.float32((Hp - 1.0) / 2.0))
        tmps, scales = [], []
        for i, c in enumerate(CHANNELS):
            s = int(round(2 ** (4 - i) / scale))
            assert Hp % (4 * s) == 0 and Wp % (4 * s) == 0
            hs, ws = Hp // s, Wp // s
            blk = "block%d." % i
            if i == 0:
                cin = 15
                xs = B.buf(hs * ws, cin)
                for src, C, o in (("x0", 3, 0), ("x1", 3, 3), ("f0", 4, 6), ("f1", 4, 10)):
                    B.f("enc_resize", src, xs, 1, Hp, Wp, hs, ws, C, 0, C, 0, cin, o)
                m = np.ones(cin, np.float32)
                m[14] = 0
                B.f("set", xs, ("enc_mul_vec", xs, m))
                B.f("enc_bias", xs, "tb15")
            else:
                cin = A_COLS
                m = np.ones(cin, np.float32)
                m[14] = 0
                m[24:28] = np.float32(1.0 / s)
                xs = B.buf(hs * ws, cin)
                if fused:
                    B.f("enc_rife_in", "x0", "x1", "f0", "f1", [B.env[t_] for t_ in tmps], list(scales), xs, m, "tb28",
                        gx, gy, Hp, Wp, s, dx, dy)
                else:
                    B.f("enc_warp", "f0", F, A, gx, gy, Hp, Wp, 4, 4, 0, 4, 0, A_COLS, 6)
                    B.f("enc_warp", "f1", F, A, gx, gy, Hp, Wp, 4, 4, 0, 4, 2, A_COLS, 10)
                    B.f("enc_resize", A, xs, 1, Hp, Wp, hs, ws, cin, 0)
                    B.f("set", xs, ("enc_mul_vec", xs, m))
                    B.f("enc_bias", xs, "tb28")
            h2, w2 = hs // 2, ws // 2
            h4, w4 = hs // 4, ws // 4
            x = self._conv(B, xs, B.index(("s2", hs, ws), lambda: conv_index(hs, ws, stride=2)),
                           self.conv_w(blk + "conv0.0.0"), act=True)
            B.free(xs)
            y = self._conv(B, x, B.index(("s2", h2, w2), lambda: conv_index(h2, w2, stride=2)),
                           self.conv_w(blk + "conv0.1.0"), act=True)
            B.free(x)
            i4 = B.index(("c3", h4, w4), lambda: conv_index(h4, w4))
            t = B.buf(h4 * w4, c)
            for k in range(8):
                n = blk + "convblock.%d" % k
                lin_k = self.conv_w(n + ".conv", beta=self.g(n + ".beta"))
                if B.gpu and self.conv_mm:          # (the residual read in the epilogue: two buffers in turn)
                    self._conv(B, y, i4, lin_k, act=True, res=y, out=t)
                    y, t = t, y
                else:
                    B.conv3(y, i4, lin_k, out=t)
                    B.add(t, y)
                    self._lrelu(B, t, out=y)
            B.free(t)
            if fused:                               # (kept to the output: each block's flow and the last mask)
                keep = B.array("tmpk%d" % i, np.zeros((hs * ws, 13), np.float32))
                self._deconv(B, y, h4, w4, blk + "lastconv.0", shuffle=True, out=keep)
                B.free(y)
                tmps.append(keep)
                scales.append(s)
                continue
            tmp = self._deconv(B, y, h4, w4, blk + "lastconv.0", shuffle=True)     # (hs ws, 13)
            B.free(y)
            if self.taps:
                B.f("set", B.array("tmp%d" % i, np.zeros((hs * ws, 13), np.float32)), ("copy", tmp))
            B.f("enc_resize", tmp, D, 1, hs, ws, Hp, Wp, 4, 0, 13, 0, 4, 0)
            if i == 0:
                B.f("set", F, ("enc_mul_vec", D, np.full(4, s, np.float32)))
            else:
                B.f("enc_add", F, D, float(s))
            B.f("enc_resize", tmp, A, 1, hs, ws, Hp, Wp, 9, 0, 13, 4, A_COLS, 15)
            B.free(tmp)
            B.f("enc_resize", F, A, 1, Hp, Wp, Hp, Wp, 4, 0, 4, 0, A_COLS, 24)
            B.f("enc_warp", "x0", F, A, gx, gy, Hp, Wp, 3, 3, 0, 4, 0, A_COLS, 0)
            B.f("enc_warp", "x1", F, A, gx, gy, Hp, Wp, 3, 3, 0, 4, 2, A_COLS, 3)
            if self.taps:
                B.f("set", B.array("flow%d" % i, np.zeros((N, 4), np.float32)), ("copy", F))
        out = B.array("out", np.zeros((N, 3), np.float32))
        if fused:
            if B.gpu:
                from reclank.backends.cuda.gpumm import pinned
            host = (lambda: pinned((N, 3), np.float32, free=True)) if B.gpu else lambda: np.zeros((N, 3), np.float32)
            w0 = B.array("w0", host()) if self.probe else None       # (pinned: their downloads each pair)
            w1 = B.array("w1", host()) if self.probe else None
            B.f("enc_rife_out", "x0", "x1", [B.env[t_] for t_ in tmps], list(scales), out,
                None if w0 is None else B.env[w0], None if w1 is None else B.env[w1], gx, gy, Hp, Wp, dx, dy)
        else:
            if self.probe:
                for k, o in (("w0", 0), ("w1", 3)):
                    B.f("enc_resize", A, B.array(k, np.zeros((N, 3), np.float32)), 1, Hp, Wp, Hp, Wp, 3, 0, A_COLS, o,
                        3, 0)
            B.f("enc_mix", A, out, N, A_COLS, 0, 3, 15, 3)
        if f32:
            return
        if io is not None:
            B.f("enc_pack_u8c", out, "out8c", io[0], io[1], Wp, 3, 255.0)
            return
        o255 = B.buf(N, 3)
        B.f("set", o255, ("enc_mul_vec", out, np.full(3, 255, np.float32)))
        B.f("enc_pack_u8", o255, "out8")

    # ---- frames ----

    def reset(self):
        self.n = 0

    def push(self, frame):
        """the next frame (uint8 (h, w, 3), or float32 (h, w, 3): 0 to 1 the trained range): its Head, the
        previous one into slot 0"""
        h, w = frame.shape[:2]
        scale = self.scale or auto_scale(h, w)
        if self.size is None or self.size[:2] != (h, w) or self.size[4] != scale:
            self.size = (h, w) + padded(h, w, scale, self.pad) + (scale,)
            self.n = 0
        _, _, Hp, Wp, _ = self.size
        if frame.dtype.kind == "f":
            x = self._xin(Hp, Wp).reshape(Hp, Wp, 3)
            x[:h, :w] = frame
            x[h:], x[:h, w:] = 0, 0                 # (the padding: zeros, as the byte path's)
            prog, g = self.runner.get(("headf", Hp, Wp, int(self.n > 0)), self.gpu)
            if g is not None:
                g.upload("xin")
                g.run()
            else:
                prog.run()
            self.n += 1
            return
        if self.gpu_io:
            frame = np.ascontiguousarray(frame, np.uint8)
            key = ("head", Hp, Wp, int(self.n > 0), h, w, (h, w) in self.plans)
            prog, g = self.runner.get(key, True)
            g.mirror.buffer_of(prog.names["raw8"]).upload(frame)
            g.run()
            self.n += 1
            return
        img = np.zeros((Hp, Wp, 3), np.uint8)
        img[:h, :w] = frame
        prog, g = self.runner.get(("head", Hp, Wp, int(self.n > 0)), self.gpu)
        prog.names["img8"][...] = as_words(img)
        if g is not None:
            g.upload("img8")
            g.run()
        else:
            prog.run()
        self.n += 1

    def aux(self):
        """(after push() with aux_plan) the pushed frame's thumbnail taps (uint8 (rows, cols, 3)), its
        blocks' channel sums (48, 64) and its probe frame (uint8 (H, W, 3))"""
        h, w, Hp, Wp, _ = self.size
        prog, g = self.runner.get(("head", Hp, Wp, int(self.n > 1), h, w, True), True)
        tp, pp = self.plans[(h, w)]
        for n in ("taps8", "sums", "probe8"):
            g.download(n)
        io = self.io[(h, w)]
        return dict(taps=io["taps8"].view(np.uint8)[:tp["rows"].size * tp["cols"].size * 3].reshape(
                        tp["rows"].size, tp["cols"].size, 3).copy(),
                    sums=io["sums"].reshape(48, 64).copy(),
                    probe=io["probe8"].view(np.uint8)[:pp["H"] * pp["W"] * 3].reshape(pp["H"], pp["W"], 3).copy())

    def fetch(self, out=None):
        """(the GPU's byte path) the frame of the last frame(t, fetch=False): uint8 (h, w, 3), into out
        when given"""
        h, w, Hp, Wp, s = self.size
        prog, g = self.runner.get(("net", Hp, Wp, s, h, w), True)
        if out is None:
            g.download("out8c")
            return self.io[(h, w)]["out8c"].view(np.uint8)[:h * w * 3].reshape(h, w, 3).copy()
        g.mirror.buffer_of(prog.names["out8c"]).download(out)
        return out

    def frame(self, t, out=None, fetch=True):
        """the frame at t (0 < t < 1) between the last two frames pushed: uint8 (h, w, 3) (into out, a
        contiguous uint8 array of h w 3 bytes (pinned: faster), when given); fetch False: not downloaded
        (outputs() of the run)"""
        if self.n < 2:
            raise RuntimeError("push two frames first")
        h, w, Hp, Wp, s = self.size
        if self.gpu_io:
            prog, g = self.runner.get(("net", Hp, Wp, s, h, w), True)
            prog.names["tb15"][14] = prog.names["tb28"][14] = np.float32(t)
            g.upload("tb15")
            g.upload("tb28")
            g.run()
            return self.fetch(out) if fetch else None
        prog, g = self.runner.get(("net", Hp, Wp, s), self.gpu)
        prog.names["tb15"][14] = prog.names["tb28"][14] = np.float32(t)
        if g is not None:
            g.upload("tb15")
            g.upload("tb28")
            g.run()
            g.download("out8")
        else:
            prog.run()
        return prog.names["out8"].view(np.uint8)[:Hp * Wp * 3].reshape(Hp, Wp, 3)[:h, :w].copy()

    def frame_f32(self, t):
        """the frame at t (0 < t < 1) between the last two frames pushed: float32 (h, w, 3), not clamped
        (push() float frames for more than 8 bits; byte frames give the byte path's values / 255)"""
        if self.n < 2:
            raise RuntimeError("push two frames first")
        h, w, Hp, Wp, s = self.size
        prog, g = self.runner.get(("netf", Hp, Wp, s), self.gpu)
        prog.names["tb15"][14] = prog.names["tb28"][14] = np.float32(t)
        if g is not None:
            g.upload("tb15")
            g.upload("tb28")
            g.run()
            g.download("out")
        else:
            prog.run()
        return prog.names["out"].reshape(Hp, Wp, 3)[:h, :w].copy()

    def outputs(self, names):
        """copies of the net program's arrays (a check: "out", "flow", "A")"""
        h, w, Hp, Wp, s = self.size
        prog, g = self.runner.get(("net", Hp, Wp, s) + ((h, w) if self.gpu_io else ()), self.gpu)
        out = {}
        for n in names:
            if g is not None:
                g.download(n)
            out[n] = prog.names[n].copy()
        return out

    def pair(self, img0, img1, times):
        """the frames at times between two uint8 frames"""
        self.reset()
        self.push(img0)
        self.push(img1)
        return [self.frame(t) for t in times]
