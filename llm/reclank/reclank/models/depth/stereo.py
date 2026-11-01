"""2D to stereo 3D for VR players: a depth model, the disparity, both eyes by
ENC_DIBR (a forward warp of each row as a mesh, the nearest first; holes from
the background: the farther side's colour, then an edge-aware fill), packed
side by side or over-under, images and video (docs/STEREO_PLAN.md).

The disparity of each pixel, in pixels of the output of width W:

    relative mode (any model): D = inverse depth (VDA: its disparity)
        normalized to 0..1 between its 1st and 99.5th percentiles (the sky,
        or MoGe's mask, at 0); s = (D - D0) divergence W / 2, D0 the
        convergence (auto: the median of D over the centre half of the frame)
    metric mode (MoGe-2): s = f_px B / 2 (1 / max(Z, z_near) - 1 / Z0), f_px
        the focal in pixels of the output, B the baseline (63 mm), Z0 the
        distance on the screen plane (auto: the median depth of the centre)

and each eye's shift is clamped to +-clamp % of W (1.5). The left eye moves
its pixels by +s (near content to the right), the right eye by -s.
"""
from __future__ import annotations

import os
import struct

import numpy as np

from reclank.encoders.runtime import _Runner
from reclank.models.depth.common import Builder, run

LAYOUTS = ("sbs", "sbs_half", "ou", "ou_half")
# the names VR players look for (DeoVR, Skybox, HereSphere, Pigasus): a full side-by-side
# _LR, half _3dh, over-under _TB, half _3dv
TAGS = {"sbs": "_LR", "sbs_half": "_3dh", "ou": "_TB", "ou_half": "_3dv", "anaglyph": "_anaglyph"}
FILLS = {"stretch": 0, "edge": 1}


def out_size(H, W, layout):
    """(height, width) of a packed stereo image of H x W eyes (a half layout of
    an odd side drops its last row or column)"""
    if layout == "sbs_half":
        W -= W % 2
    elif layout == "ou_half":
        H -= H % 2
    return {"sbs": (H, 2 * W), "sbs_half": (H, W), "ou": (2 * H, W), "ou_half": (H, W)}[layout]


class Warper:
    """Both eyes and their packing as one program a size (the disparity's
    size, the image's, the layout, the fill)."""

    def __init__(self, gpu=False, cpu_threads=None, gap=1.0, iters=8, tol=0.5):
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.gap, self.iters, self.tol = gap, iters, tol
        self.runner = _Runner(self._build, keep=3, keep_gpu=2)
        self.runner.label = "stereo-warp"

    def _build(self, key, gpu):
        hd, wd, H, W, layout, fill = key
        B = Builder(gpu)
        B.array("img8", np.zeros((H * W * 3 + 3) // 4, np.float32))    # the uint8 image's bytes
        B.array("img", np.zeros((H * W, 3), np.float32))
        B.f("enc_unpack_u8", "img8", "img", 1.0)
        B.array("disp", np.zeros((hd * wd, 1), np.float32))
        if (hd, wd) == (H, W):
            s = "disp"
        else:
            s = B.array("shift", np.zeros((H * W, 1), np.float32))
            B.f("enc_resize", "disp", s, 1, hd, wd, H, W, 1, 0)
        keys = B.array("keys", np.zeros(2 * H * W, np.float32))
        zb = B.array("zb", np.zeros(H * W, np.float32))
        tmp = B.array("tmp", np.zeros(3 * H * W, np.float32))
        if layout.startswith("sbs"):
            full = B.array("full", np.zeros((H, 2 * W, 3), np.float32))
            offs, stride = (0, 3 * W), 6 * W
        else:
            full = B.array("full", np.zeros((2 * H, W, 3), np.float32))
            offs, stride = (0, 3 * H * W), 3 * W
        for eye, sign in enumerate((1, -1)):
            B.f("enc_dibr", "img", s, keys, zb, tmp, full, H, W, offs[eye], stride, sign, self.gap, FILLS[fill],
                self.iters, self.tol)
        oh, ow = out_size(H, W, layout)
        if layout == "sbs_half":
            out = B.array("half", np.zeros((H, W, 3), np.float32))
            B.f("enc_resize", full, out, 1, H, 2 * W, H, W, 3, 2)
        elif layout == "ou_half":
            out = B.array("half", np.zeros((H, W, 3), np.float32))
            B.f("enc_resize", full, out, 1, 2 * H, W, H, W, 3, 2)
        else:
            out = full
        B.array("u8", np.zeros((oh * ow * 3 + 3) // 4, np.float32))
        B.f("enc_pack_u8", out, "u8")
        return B.finish(self.cpu_threads)

    def warp(self, img, shift, layout="sbs", fill="edge"):
        """The packed stereo image (uint8) of an uint8 image (H, W, 3) and the
        left eye's shift (hd, wd) in pixels of the image (resized bilinear to
        the image's size in the program)."""
        if layout not in LAYOUTS:
            raise ValueError("layout must be one of %s" % ", ".join(LAYOUTS))
        H, W = img.shape[:2]
        if layout == "sbs_half" and W % 2:          # a half layout of an odd side: its last column goes
            img, W = img[:, :W - 1], W - 1
        elif layout == "ou_half" and H % 2:
            img, H = img[:H - 1], H - 1
        hd, wd = shift.shape
        u8, = run(self.runner, (hd, wd, H, W, layout, fill), self.gpu,
                  {"img8": as_words(img), "disp": shift.reshape(-1, 1).astype(np.float32)}, ["u8"])
        oh, ow = out_size(H, W, layout)
        return u8.view(np.uint8)[:oh * ow * 3].reshape(oh, ow, 3)


def as_words(img):
    """the bytes of an uint8 image as float32 words (padded to 4 bytes): an
    input that ENC_UNPACK_U8 reads"""
    b = np.ascontiguousarray(img, np.uint8).ravel()
    if b.size % 4:
        b = np.concatenate([b, np.zeros(4 - b.size % 4, np.uint8)])
    return b.view(np.float32)


# ---- the disparity ------------------------------------------------------------

def centre(a):
    h, w = a.shape[:2]
    return a[h // 4:h - h // 4, w // 4:w - w // 4]


def normalized(inv, valid, lo=None, hi=None):
    """inverse depth (or disparity) to 0..1 between its 1st and 99.5th
    percentiles over valid (or lo, hi), invalid pixels 0. Returns (D, lo, hi)."""
    inv = inv.astype(np.float32)
    if lo is None:
        v = inv[valid] if valid.any() else inv.ravel()
        lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99.5))
        if not (~valid).all() and (~valid).any():
            lo = min(lo, 0.0)       # the sky is the far end
    d = np.clip((inv - np.float32(lo)) / np.float32(max(hi - lo, 1e-12)), 0, 1).astype(np.float32)
    return np.where(valid, d, np.float32(0)), lo, hi


def convergence_of(D, valid):
    """the median of D over the valid pixels of the centre (all valid; 0.5)"""
    c, cv = centre(D), centre(valid)
    if cv.any():
        return float(np.median(c[cv]))
    if valid.any():
        return float(np.median(D[valid]))
    return 0.5


def relative_shift(D, D0, width, divergence=2.0, clamp=1.5):
    """the left eye's shift (pixels of an output of width) of normalized D"""
    s = (D - np.float32(D0)) * np.float32(divergence / 100.0 / 2.0 * width)
    lim = np.float32(clamp / 100.0 * width)
    return np.clip(s, -lim, lim).astype(np.float32)


def metric_shift(Z, valid, fx, width, baseline_mm=63.0, z0=None, z_near=0.6, clamp=1.5):
    """the left eye's shift of metric depth Z (metres; fx the focal in widths)"""
    zc = centre(np.where(valid, Z, np.nan))
    if z0 is None:
        z0 = float(np.nanmedian(zc)) if np.isfinite(zc).any() else float(np.nanmedian(np.where(valid, Z, np.nan)))
    inv = np.where(valid, 1.0 / np.maximum(Z, z_near), 0.0)
    k = fx * width * baseline_mm / 1000.0 / 2.0
    lim = clamp / 100.0 * width
    return np.clip(k * (inv - 1.0 / max(z0, z_near)), -lim, lim).astype(np.float32), z0


def anaglyph(packed, layout="sbs"):
    """a red-cyan anaglyph of a full side-by-side or over-under image (checks)"""
    if layout == "sbs":
        W = packed.shape[1] // 2
        L, R = packed[:, :W], packed[:, W:]
    else:
        H = packed.shape[0] // 2
        L, R = packed[:H], packed[H:]
    return np.dstack([L[..., 0], R[..., 1], R[..., 2]])


# ---- Spherical Video V2's st3d box ----------------------------------------------

ST3D_MODES = {"mono": 0, "ou": 1, "sbs": 2}
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def _boxes(buf, start, end):
    """(type, start, header size, end) of the boxes of buf[start:end]"""
    out = []
    p = start
    while p + 8 <= end:
        size, kind = struct.unpack(">I4s", buf[p:p + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[p + 8:p + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - p
        out.append((kind, p, hdr, p + size))
        p += size
    return out


def _st3d_moov(moov, mode):
    """moov (bytes, a whole box) with an st3d box (stereo_mode) in its video
    sample entry, the sizes of its parents grown; None if it has one or no
    video track"""
    buf = bytearray(moov)
    path = []

    def find(start, end, depth):
        for kind, s, hdr, e in _boxes(buf, start, end):
            if kind == b"trak":
                hd = _find_hdlr(buf, s + hdr, e)
                if hd != b"vide":
                    continue
            if kind in _CONTAINERS:
                path.append((s, hdr))
                r = find(s + hdr, e, depth + 1)
                if r is not None:
                    return r
                path.pop()
            elif kind == b"stsd":
                path.append((s, hdr))
                entry = s + hdr + 8                 # version/flags, entry count
                size = struct.unpack(">I", buf[entry:entry + 4])[0]
                path.append((entry, 8))
                return entry, size
        return None
    top = _boxes(buf, 0, len(buf))[0]
    path.append((top[1], top[2]))
    r = find(top[1] + top[2], top[3], 0)
    if r is None:
        return None
    entry, size = r
    # the children of a visual sample entry start after its 78 bytes of fields
    for kind, _s, _h, _e in _boxes(buf, entry + 8 + 78, entry + size):
        if kind == b"st3d":
            return None
    box = struct.pack(">I4sIB", 13, b"st3d", 0, mode)
    buf[entry + size:entry + size] = box
    for s, hdr in path:
        if hdr == 8:
            n = struct.unpack(">I", buf[s:s + 4])[0]
            buf[s:s + 4] = struct.pack(">I", n + len(box))
        else:
            n = struct.unpack(">Q", buf[s + 8:s + 16])[0]
            buf[s + 8:s + 16] = struct.pack(">Q", n + len(box))
    return bytes(buf), len(box)


def _find_hdlr(buf, start, end):
    for kind, s, hdr, e in _boxes(buf, start, end):
        if kind == b"mdia":
            for k2, s2, h2, e2 in _boxes(buf, s + hdr, e):
                if k2 == b"hdlr":
                    return bytes(buf[s2 + h2 + 8:s2 + h2 + 12])
    return None


def _shift_offsets(moov, delta):
    """the chunk offsets (stco, co64) of moov moved by delta (moov before mdat)"""
    buf = bytearray(moov)

    def walk(start, end):
        for kind, s, hdr, e in _boxes(buf, start, end):
            if kind in _CONTAINERS:
                walk(s + hdr, e)
            elif kind in (b"stco", b"co64"):
                n = struct.unpack(">I", buf[s + hdr + 4:s + hdr + 8])[0]
                p = s + hdr + 8
                for i in range(n):
                    if kind == b"stco":
                        v = struct.unpack(">I", buf[p + 4 * i:p + 4 * i + 4])[0]
                        buf[p + 4 * i:p + 4 * i + 4] = struct.pack(">I", v + delta)
                    else:
                        v = struct.unpack(">Q", buf[p + 8 * i:p + 8 * i + 8])[0]
                        buf[p + 8 * i:p + 8 * i + 8] = struct.pack(">Q", v + delta)
    top = _boxes(buf, 0, len(buf))[0]
    walk(top[1] + top[2], top[3])
    return bytes(buf)


def inject_st3d(path, layout):
    """Mark an MP4's video track as stereo (Spherical Video V2's st3d box in
    its sample entry: 1 top-bottom, 2 left-right). A moov at the end grows in
    place; a moov before the media is rewritten with its chunk offsets moved.
    Returns True when the box was added."""
    mode = ST3D_MODES["sbs" if layout.startswith("sbs") else "ou"]
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        tops = []
        p = 0
        while p < size:
            f.seek(p)
            hd = f.read(16)
            n, kind = struct.unpack(">I4s", hd[:8])
            if n == 1:
                n = struct.unpack(">Q", hd[8:16])[0]
            elif n == 0:
                n = size - p
            tops.append((kind, p, n))
            p += n
    moov = [t for t in tops if t[0] == b"moov"]
    if len(moov) != 1:
        raise ValueError("%s: no single moov box" % path)
    _, mp, mn = moov[0]
    with open(path, "rb") as f:
        f.seek(mp)
        old = f.read(mn)
    r = _st3d_moov(old, mode)
    if r is None:
        return False
    new, grow = r
    after = [t for t in tops if t[1] > mp]
    if not after:
        with open(path, "r+b") as f:
            f.seek(mp)
            f.write(new)
            f.truncate()
        return True
    if any(t[0] == b"mdat" for t in after):
        new = _st3d_moov(_shift_offsets(old, grow), mode)[0]
    tmp = path + ".st3d.part"
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        for kind, p, n in tops:
            if p == mp:
                dst.write(new)
                continue
            src.seek(p)
            left = n
            while left:
                chunk = src.read(min(left, 1 << 24))
                dst.write(chunk)
                left -= len(chunk)
    os.replace(tmp, path)
    return True
