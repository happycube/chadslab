"""Deinterlace a video field by field with RIFE 4.25 (models/vfi): each field becomes a progressive frame
whose own lines are the field's, and whose other lines are the opposite field interpolated to its time.

    deinterlace_rife.py SRC DST --rife DIR [--gpu] [--rate field|frame]
                        [--io float|bytes] [--codec h264|ffv1] [--crf 16] [--pix-fmt F]
                        [--upscale MODEL --esrgan DIR] [--gpu-lock PATH]
                        [--order auto|tff|bff] [--cuts detect|off] [--start SECONDS]
                        [--max-frames N]

The fields of N frames at times 0, 1, 2, ... (1 / (2 fps) apart): the first field F_k of frame k at 2k,
the second S_k at 2k + 1. Each parity is its own half-height stream (RIFE pairs its consecutive fields,
so the interpolated lines sit on their own rows: no half-line shift):

    output 2k:     F_k's lines, and S at 2k     = RIFE(S_{k-1}, S_k, 0.5)   (k = 0: S_0)
    output 2k + 1: S_k's lines, and F at 2k + 1 = RIFE(F_k, F_{k+1}, 0.5)   (k = N - 1: F_{N-1})

--rate field (default): 2N frames at twice the frame rate; --rate frame: the even outputs alone (N
frames at the frame rate, each frame's first field kept).

A field's own lines are its decoded samples, untouched (yuv422p10le, or 8-bit 4:2:0 / 4:2:2 / 4:4:4
sources in their own format). RIFE takes the fields as float32 RGB (--io float, the default: by the
matrix, BT.601 or BT.709, limited range, 4:2:x chroma cosited, linear between; not clamped, so super-whites and
below-black survive) and its float output goes back to the format's samples with one rounding;
--io bytes: 8-bit RGB both ways. 4:2:0
sources: their chroma rows are interlaced too, so the chroma planes are split into fields as the luma.

Scene cuts (--cuts detect): a pair of same-parity fields across a cut (models/vfi/video.CutDetector's
thumbnail rule) gives no interpolation; the missing lines are the neighbour field of the pair (its own
samples) nearer the real field (SSIM of their thumbnails).

--upscale MODEL: each output frame 2x (1536 x 968 of 768 x 484) by Real-ESRGAN in float32
(realesrgan-x2plus, or realesr-general-x4v3 averaged 2 x 2 down from its 4x; Upscaler): the field's own
lines are then no longer its samples. Long runs: --gpu-lock /path/gpu.lock (the script takes the lock
itself and lets it go every --lock-minutes 10) in place of an outer flock.

The output (Matroska): H.264 (--codec h264, the default: libx264 --crf 16 --preset slow, the source's
pix_fmt, so yuv422p10le is High 4:2:2; --pix-fmt yuv420p for common players) or FFV1 level 3 (--codec
ffv1, lossless), progressive, the source's colour tags and aspect, its audio copied. Run GPU jobs as every reclank GPU run: OMP_NUM_THREADS=1, under the GPU lock.
"""
import argparse
import concurrent.futures
import fcntl
import json
import sys
import threading
import queue
import subprocess
import time
from fractions import Fraction

import numpy as np

from reclank.models import media
from reclank.models.vfi import video as V
from reclank.models.vfi.rife import RIFE

FORMATS = {     # pix_fmt: (bytes a sample, chroma x shift, chroma y shift, bits)
    "yuv422p10le": (2, 1, 0, 10), "yuv420p10le": (2, 1, 1, 10), "yuv444p10le": (2, 0, 0, 10),
    "yuv422p": (1, 1, 0, 8), "yuv420p": (1, 1, 1, 8), "yuv444p": (1, 0, 0, 8),
}
MATRIX = {"bt709": (0.2126, 0.0722)}       # (Kr, Kb); everything else BT.601


def probe(src):
    out = media.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream=width,height,pix_fmt,r_frame_rate,field_order,color_range,color_space,"
                     "color_primaries,color_transfer,sample_aspect_ratio", "-of", "json", src],
                    capture_output=True, check=True)
    return json.loads(out.stdout)["streams"][0]


class Planes:
    """the layout of one frame of a planar YUV pix_fmt"""

    def __init__(self, pix_fmt, w, h):
        if pix_fmt not in FORMATS:
            raise SystemExit("pix_fmt %s not handled (%s)" % (pix_fmt, ", ".join(FORMATS)))
        self.bps, self.sx, self.sy, self.bits = FORMATS[pix_fmt]
        self.dtype = np.uint16 if self.bps == 2 else np.uint8
        self.w, self.h = w, h
        self.cw, self.ch = w >> self.sx, h >> self.sy
        self.size = (w * h + 2 * self.cw * self.ch) * self.bps

    def split(self, buf):
        a = np.frombuffer(buf, self.dtype)
        n, c = self.w * self.h, self.cw * self.ch
        return (a[:n].reshape(self.h, self.w), a[n:n + c].reshape(self.ch, self.cw),
                a[n + c:].reshape(self.ch, self.cw))

    def field(self, planes, p):
        """the field of parity p (0: the even lines, 1: the odd lines) of a frame's planes"""
        return tuple(x[p::2] for x in planes)

    def weave(self, own, other, p):
        """a frame of field own (parity p) and field other (parity 1 - p)"""
        out = []
        for a, b in zip(own, other):
            f = np.empty((a.shape[0] + b.shape[0], a.shape[1]), self.dtype)
            f[p::2], f[1 - p::2] = a, b
            out.append(f)
        return out


class Color:
    """a field's YUV (limited range) to 8-bit RGB for RIFE, and RIFE's RGB back"""

    def __init__(self, planes, space):
        kr, kb = MATRIX.get(space, (0.299, 0.114))
        self.kr, self.kb, self.kg = kr, kb, 1.0 - kr - kb
        s = float(1 << (planes.bits - 8))
        self.y0, self.yr, self.c0, self.cr = 16 * s, 219 * s, 128 * s, 224 * s
        self.P, self.top = planes, (1 << planes.bits) - 1

    def _up(self, c):
        """chroma to full width (cosited: the odd samples the mean of their neighbours)"""
        c = c.astype(np.float32)
        if not self.P.sx:
            return c
        out = np.empty((c.shape[0], c.shape[1] * 2), np.float32)
        out[:, 0::2] = c
        out[:, 1:-1:2] = 0.5 * (c[:, :-1] + c[:, 1:])
        out[:, -1] = c[:, -1]
        return out

    def _down(self, c):
        """full-width chroma to the format's (the cosited samples: _up's inverse, so a field converted
        both ways is its own samples)"""
        return c[:, 0::2] if self.P.sx else c

    def _rows(self, c, h):
        """chroma rows to the field's luma rows (4:2:0: each repeated)"""
        return np.repeat(c, 2, axis=0)[:h] if self.P.sy else c

    def rgb(self, fld):
        """the field as 8-bit RGB"""
        return np.clip(np.rint(self.rgbf(fld) * 255), 0, 255).astype(np.uint8)

    def rgbf(self, fld):
        """the field as float32 RGB (0 to 1 the nominal range; not clamped: super-whites, below-black and
        out-of-gamut colours kept)"""
        y, u, v = fld
        h = y.shape[0]
        Y = (y.astype(np.float32) - self.y0) / self.yr
        pb = (self._rows(self._up(u), h) - self.c0) / self.cr
        pr = (self._rows(self._up(v), h) - self.c0) / self.cr
        kr, kb, kg = self.kr, self.kb, self.kg
        r = Y + 2 * (1 - kr) * pr
        b = Y + 2 * (1 - kb) * pb
        g = (Y - kr * r - kb * b) / kg
        return np.stack([r, g, b], -1).astype(np.float32)

    def yuv(self, rgb, ch):
        """8-bit RGB to the field's planes"""
        return self.yuvf(rgb.astype(np.float32) / np.float32(255), ch)

    def yuvf(self, x, ch):
        """float RGB to the field's planes (rounded once, to the format's bits)"""
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        Y = self.kr * r + self.kg * g + self.kb * b
        pb = (b - Y) / (2 * (1 - self.kb))
        pr = (r - Y) / (2 * (1 - self.kr))
        u, v = self._down(pb), self._down(pr)
        if self.P.sy:                       # (4:2:0: a field's chroma row the mean of its two luma rows)
            u = 0.5 * (u[0::2] + u[1::2])[:ch]
            v = 0.5 * (v[0::2] + v[1::2])[:ch]
        q = lambda a, o, s: np.clip(np.rint(o + s * a), 0, self.top).astype(self.P.dtype)
        return q(Y, self.y0, self.yr), q(u, self.c0, self.cr), q(v, self.c0, self.cr)


class Stream:
    """one parity's fields through its own RIFE (the pair of its last two fields)"""

    def __init__(self, rife, color, cuts, f32=True):
        self.rife, self.color, self.f32 = rife, color, f32
        self.cuts = V.CutDetector() if cuts else None
        self.prev = self.cur = None     # (field planes, 8-bit rgb (the cut test's), thumbs)
        self.ncut = 0

    def push(self, fld):
        if self.f32:
            x = self.color.rgbf(fld)
            rgb = np.clip(np.rint(x * 255), 0, 255).astype(np.uint8) if self.cuts else None
        else:
            x = rgb = self.color.rgb(fld)
        self.rife.push(x)
        self.prev, self.cur = self.cur, (fld, rgb, V.thumbs(rgb) if self.cuts else None)

    def middle(self, real_rgb, ch):
        """the field halfway between the last two pushed (planes); real_rgb: the opposite field at that
        time (the cut's choice)"""
        (f0, _, t0), (f1, _, t1) = self.prev, self.cur
        if self.cuts is not None and self.cuts(t0, t1)[0]:
            self.ncut += 1
            tr = V.thumbs(real_rgb)
            return f0 if V.ssim(tr[0], t0[0]) >= V.ssim(tr[0], t1[0]) else f1
        if self.f32:
            return self.color.yuvf(self.rife.frame_f32(0.5), ch)
        return self.color.yuv(self.rife.frame(0.5), ch)


class Upscaler:
    """the woven frames 2x by Real-ESRGAN (models/realesrgan) in float32: the frame's float RGB clipped to
    [0, 1] (the trained range) through the network, plus the clipped-off part (super-whites, below-black,
    out-of-gamut colours) resized bicubic (input "residual", the default; "raw": the network on the
    unclipped values). A x4 model's output is averaged 2 x 2 down to 2x."""

    def __init__(self, path, model, gpu, color, pout, denoise=0.5, mode="residual"):
        from reclank.models.realesrgan.model import RealESRGAN
        self.m = RealESRGAN(path, model, gpu=gpu, denoise=denoise)
        if self.m.scale not in (2, 4):
            raise SystemExit("%s: scale %d" % (model, self.m.scale))
        self.color, self.pout, self.mode = color, pout, mode

    def __call__(self, planes):
        return self.finish(self.start(planes))

    def start(self, planes):
        """the network's part (the GPU): (the frame's float RGB, its clipped values, the network's output)"""
        x = self.color.rgbf(planes)
        c = np.clip(x, 0, 1) if self.mode == "residual" else x
        return x, c, self.m.upscale_f32(c)

    def finish(self, st):
        """the host's part (a worker thread): the 2 x 2 mean of a x4 model, the clipped-off part added back,
        the output's planes"""
        from PIL import Image
        x, c, y = st
        h, w = x.shape[:2]
        if self.m.scale == 4:
            y = y.reshape(2 * h, 2, 2 * w, 2, 3).mean((1, 3))
        if self.mode == "residual":
            r = x - c
            if r.any():
                y = y + np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(r[..., i])).resize(
                    (2 * w, 2 * h), Image.BICUBIC)) for i in range(3)], -1)
        return self.color.yuvf(y, self.pout.ch)


class GpuLock:
    """the GPU lock (flock) held at most minutes at a time: tick() between frames lets it go for a few
    seconds (another job may take it) and waits for it again"""

    def __init__(self, path, minutes):
        self.f = open(path, "a") if path else None
        self.limit, self.t = minutes * 60, None
        self.take()

    def take(self):
        if self.f is not None:
            fcntl.flock(self.f, fcntl.LOCK_EX)
            self.t = time.time()

    def tick(self):
        if self.f is not None and time.time() - self.t > self.limit:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            time.sleep(5)
            t = time.time()
            self.take()
            print("(the GPU lock let go; taken back after %.0f s)" % (time.time() - t), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rife", required=True, help="the folder of RIFE 4.25's flownet.safetensors")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--rate", choices=("field", "frame"), default="field")
    ap.add_argument("--order", choices=("auto", "tff", "bff"), default="auto")
    ap.add_argument("--cuts", choices=("detect", "off"), default="detect")
    ap.add_argument("--start", default=None, help="seconds into the source (ffmpeg -ss, both the video and audio)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--io", choices=("float", "bytes"), default="float",
                    help="RIFE's frames: float32 (default: the source's precision) or 8-bit RGB")
    ap.add_argument("--codec", choices=("h264", "ffv1"), default="h264")
    ap.add_argument("--crf", type=int, default=16, help="h264's CRF")
    ap.add_argument("--preset", default="slow", help="h264's x264 preset")
    ap.add_argument("--pix-fmt", default=None, help="the output's pix_fmt (default: the source's; h264: "
                    "yuv420p for players without High 4:2:2 / 10-bit)")
    ap.add_argument("--upscale", default=None, help="2x after the deinterlace by a Real-ESRGAN model: "
                    "realesrgan-x2plus or realesr-general-x4v3 (its 4x averaged to 2x), or the others of "
                    "models/realesrgan MODELS")
    ap.add_argument("--esrgan", default=None, help="the folder of Real-ESRGAN's converted .safetensors")
    ap.add_argument("--denoise", type=float, default=0.5, help="realesr-general-x4v3's denoise strength")
    ap.add_argument("--esr-input", choices=("residual", "raw"), default="residual",
                    help="out-of-range values: clipped for the network and added back resized (default), or raw")
    ap.add_argument("--gpu-lock", default=None, help="take this flock file around the GPU work, letting it go "
                    "every --lock-minutes (instead of an outer flock: long upscale runs)")
    ap.add_argument("--lock-minutes", type=float, default=10)
    ap.add_argument("--threads", type=int, default=8, help="ffmpeg's threads (decode and encode)")
    a = ap.parse_args()
    if a.upscale and not a.esrgan:
        ap.error("--upscale needs --esrgan DIR")

    info = probe(a.src)
    w, h, pix = int(info["width"]), int(info["height"]), info["pix_fmt"]
    if h % 2:
        raise SystemExit("odd height %d" % h)
    order = a.order
    if order == "auto":
        fo = info.get("field_order", "unknown")
        if fo in ("tt", "tb"):
            order = "tff"
        elif fo in ("bb", "bt"):
            order = "bff"
        else:
            raise SystemExit("the source's field_order is %s: give --order tff or bff" % fo)
    first = 0 if order == "tff" else 1          # the parity of the first field in time
    P = Planes(pix, w, h)
    if P.sy and (h // 2) % 2:
        raise SystemExit("4:2:0 with an odd field height is not handled")
    C = Color(P, info.get("color_space", ""))
    fps = Fraction(info["r_frame_rate"])
    out_fps = fps * 2 if a.rate == "field" else fps
    fch = P.ch // 2                              # a field's chroma rows
    ow, oh = (2 * w, 2 * h) if a.upscale else (w, h)
    Pout = Planes(pix, ow, oh)

    ss = ["-ss", a.start] if a.start else []
    dec = media.popen(["ffmpeg", "-v", "error", "-threads", str(a.threads)] + ss + ["-i", a.src, "-map", "0:v:0",
                       "-f", "rawvideo", "-pix_fmt", pix, "pipe:1"], stdout=subprocess.PIPE)
    tags = []
    for k, opt in (("color_range", "-color_range"), ("color_space", "-colorspace"),
                   ("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc")):
        if info.get(k) and info[k] != "unknown":
            tags += [opt, info[k]]
    sar = info.get("sample_aspect_ratio", "1:1")
    vf = "setfield=prog" + (",setsar=sar=%s:max=10000" % sar.replace(":", "/") if sar not in ("0:1", "N/A") else "")
    # (no -shortest: the audio may end a few frames before the video; --max-frames: the output's length)
    n_out = None if a.max_frames is None else a.max_frames * (2 if a.rate == "field" else 1)
    dur = ["-t", "%.6f" % (n_out / out_fps)] if n_out else []
    if a.codec == "h264":
        venc = ["-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf)]
    else:
        venc = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "24", "-slicecrc", "1"]
    venc += ["-pix_fmt", a.pix_fmt or pix]
    enc = media.popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix, "-s", "%dx%d" % (ow, oh),
                       "-framerate", str(out_fps), "-i", "pipe:0"] + ss + ["-i", a.src, "-map", "0:v", "-map", "1:a?",
                       "-vf", vf] + venc + ["-threads", str(a.threads)] + tags + ["-c:a", "copy"] + dur + [a.dst],
                      stdin=subprocess.PIPE)

    def read():
        buf = dec.stdout.read(P.size)
        return P.split(buf) if len(buf) == P.size else None

    q = queue.Queue(maxsize=16)

    def writer():
        while True:
            fr = q.get()
            if fr is None:
                break
            if isinstance(fr, concurrent.futures.Future):
                fr = fr.result()
            for x in fr:
                enc.stdin.write(np.ascontiguousarray(x).tobytes())
        enc.stdin.close()

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    lock = GpuLock(a.gpu_lock, a.lock_minutes)
    S = {p: Stream(RIFE(a.rife, gpu=a.gpu), C, a.cuts == "detect", a.io == "float") for p in (0, 1)}
    up = Upscaler(a.esrgan, a.upscale, a.gpu, C, Pout, a.denoise, a.esr_input) if a.upscale else None
    pool = concurrent.futures.ThreadPoolExecutor(4) if up is not None else None
    last = [time.time()]

    def emit(fr):
        # (an upscale's host part on the pool, in order through the queue, beside the next frame's GPU work)
        q.put(pool.submit(up.finish, up.start(fr)) if up is not None else fr)
        lock.tick()

    F, Sd = S[first], S[1 - first]               # the first field's stream, the second's
    t0 = time.time()
    cur = read()
    if cur is None:
        raise SystemExit("no frames")
    F.push(P.field(cur, first))
    Sd.push(P.field(cur, 1 - first))
    k, done = 0, 0
    while cur is not None:
        fk, sk = P.field(cur, first), P.field(cur, 1 - first)
        # output 2k: F_k and S at 2k
        other = sk if k == 0 else Sd.middle(F.cur[1], fch)
        emit(P.weave(fk, other, first))
        done += 1
        nxt = None if a.max_frames is not None and k + 1 >= a.max_frames else read()
        if nxt is not None:
            F.push(P.field(nxt, first))
        if a.rate == "field":
            # output 2k + 1: S_k and F at 2k + 1
            other = F.middle(Sd.cur[1], fch) if nxt is not None else fk
            emit(P.weave(sk, other, 1 - first))
            done += 1
        if nxt is not None:
            Sd.push(P.field(nxt, 1 - first))
        cur, k = nxt, k + 1
        if time.time() - last[0] > 60:
            last[0] = time.time()
            el = time.time() - t0
            print("%d frames in, %d out, %.1f s (%.1f out a second)" % (k, done, el, done / el), flush=True)
    q.put(None)
    wt.join()
    if dec.poll() is None:
        dec.terminate()                         # (--max-frames: the rest not read)
    dec.stdout.close()
    dec.wait()
    rc = enc.wait()
    el = time.time() - t0
    print("%s: %d frames (%s %s, %s) to %d frames (%dx%d%s) at %s fps in %.1f s (%.1f a second); cuts: %d first-field "
          "pairs, %d second-field pairs; encoder exit %d"
          % (a.dst, k, pix, order, fps, done, ow, oh, ", " + a.upscale if a.upscale else "", out_fps, el, done / el, F.ncut, Sd.ncut, rc), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
