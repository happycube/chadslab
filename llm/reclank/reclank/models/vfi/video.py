"""Ported in part from Practical-RIFE (inference_video.py's scene-cut test and
model/pytorch_msssim/__init__.py's ssim_matlab; MIT License; Copyright (c) 2021 hzwer), and in part
from Pillow (src/libImaging/Resample.c: box_coeffs, precompute_coeffs and normalize_coeffs_8bpc for
the BOX filter; MIT-CMU License; Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011
by Fredrik Lundh and contributors, Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors). See
THIRD_PARTY_NOTICES.md.

Frame interpolation of a video file (docs/VFI_PLAN.md): ffmpeg decodes the frames (their own
timestamps: no frame dropped or doubled), RIFE makes the frames between each pair at the output's
times, scene cuts are found and not interpolated across, ffmpeg encodes at the new rate with the
source's audio copied and its colour tags kept.

The output's times (schedule): "factor" F keeps every input frame and puts F - 1 frames between each
pair (t = k / F), the last input frame repeated F - 1 times so the length (and the audio's
alignment) stays the input's; "fps" takes the frame at each time k / fps of the input's timeline
(any ratio: 23.976 -> 60, t the fraction between the two neighbours), by the frames' timestamps when
the input's rate is variable. An output frame at t = 0 is the input frame's bytes.

Scene cuts (docs/VFI_PLAN.md section 5; CutDetector): RIFE itself on small probe frames (the long
side 320) tells whether a flow explains the pair (the residual of the two frames warped to the
middle jumps at a cut), with the thumbnails' SSIM and difference ratio (Practical-RIFE's
ssim_matlab of 32 x 32 thumbnails); the frames between a cut's pair repeat its left frame (the cut
stays at the right frame's time, as the authors' script and vsrife do).

On the GPU (RIFE.gpu_io; docs/VFI_PLAN.md section 6.3) the decoder's thread reads each frame into a
pinned slot and the model takes its bytes: the padding, the thumbnails' inputs (thumb_plan) and the
probe frame (probe_plan: PIL's BOX in integers) are made on the GPU, the output frame's bytes come
back into a pinned slot the writer's thread hands to ffmpeg, and the pair's first frame is made
while the host runs the cut test; the same bits as the host's path (the CPU's). ffmpeg and the
decoder's and writer's threads run on the CPUs of the start: reclank's OMP_PROC_BIND binds the main
thread to one CPU, and they would inherit it."""
from __future__ import annotations

import functools
import json
import math
import queue
import subprocess
import threading
import time
from fractions import Fraction

import numpy as np

from reclank.encoders.runtime import RESIDENCY
from reclank.models import media
from reclank.models.depth import ref as dref
from reclank.models.vfi.rife import mean_abs

MP4_AUDIO = ("aac", "mp3", "ac3", "eac3", "alac")
SSIM_CUT, SSIM_LOW, RATIO = 0.2, 0.45, 6.0


# ---- scene cuts ----------------------------------------------------------------

def _gauss(n=11, sigma=1.5):
    g = np.exp(-((np.arange(n) - n // 2) ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


G11 = _gauss()


@functools.lru_cache(maxsize=8)
def thumb_plan(h, w):
    """the thumbnails' geometry of an h x w frame: the rows and columns the 32 x 32 bilinear taps read
    (rows, cols), the taps' places among them and their weights, the 48 x 64 grey blocks' bounds"""
    y0, y1, ly0, ly1 = dref.lerp_src(h, 32, 0)
    x0, x1, lx0, lx1 = dref.lerp_src(w, 32, 0)
    rows, cols = np.unique(np.concatenate([y0, y1])), np.unique(np.concatenate([x0, x1]))
    return dict(rows=rows, cols=cols, ry0=np.searchsorted(rows, y0), ry1=np.searchsorted(rows, y1),
                rx0=np.searchsorted(cols, x0), rx1=np.searchsorted(cols, x1), ly0=ly0, ly1=ly1, lx0=lx0, lx1=lx1,
                by0=np.arange(48) * h // 48, by1=-(-(np.arange(48) + 1) * h // 48),
                bx0=np.arange(64) * w // 64, bx1=-(-(np.arange(64) + 1) * w // 64))


def thumbs_of(plan, taps, sums):
    """thumbs() of a frame's taps (its pixels at plan's rows x cols, uint8) and its grey blocks' sums of
    the three channels (48, 64) (the GPU's ENC_THUMB_U8, or thumbs() on the CPU)"""
    f32 = np.float32
    P = taps.astype(np.float32) / f32(255)
    ry0, ry1, rx0, rx1 = plan["ry0"], plan["ry1"], plan["rx0"], plan["rx1"]
    lx0, lx1 = plan["lx0"][None, :, None], plan["lx1"][None, :, None]
    ly0, ly1 = plan["ly0"], plan["ly1"]
    top = lx0 * P[ry0][:, rx0] + lx1 * P[ry0][:, rx1]
    bot = lx0 * P[ry1][:, rx0] + lx1 * P[ry1][:, rx1]
    t32 = (ly0[:, None, None] * top + ly1[:, None, None] * bot).astype(np.float32).transpose(2, 0, 1)
    by0, by1, bx0, bx1 = plan["by0"], plan["by1"], plan["bx0"], plan["bx1"]
    grey = (np.asarray(sums, np.int64) / (765.0 * (by1 - by0)[:, None] * (bx1 - bx0)[None, :])).astype(np.float32)
    return np.ascontiguousarray(t32), grey


def thumbs(frame):
    """(the 32 x 32 RGB thumbnail (3, 32, 32): torch's bilinear, align_corners False, of the frame
    / 255; the 64 x 48 grey one (48, 64): the channels' mean, adaptive average pooling) of an uint8
    frame (h, w, 3). Only the rows and columns the bilinear taps read are converted."""
    h, w = frame.shape[:2]
    plan = thumb_plan(h, w)
    by0, by1, bx0, bx1 = plan["by0"], plan["by1"], plan["bx0"], plan["bx1"]
    s3 = frame[:, :, 0].astype(np.uint16)                       # (the channels' sum: 3 strided adds, not
    s3 += frame[:, :, 1]                                        # a reduction over the last axis of 3)
    s3 += frame[:, :, 2]
    cy = np.zeros((h + 1, w), np.uint32)
    np.cumsum(s3, 0, dtype=np.uint32, out=cy[1:])
    band = (cy[by1] - cy[by0]).astype(np.int64)                 # (48, w): each row band's column sums
    cx = np.zeros((48, w + 1), np.int64)
    np.cumsum(band, 1, out=cx[:, 1:])
    return thumbs_of(plan, frame[np.ix_(plan["rows"], plan["cols"])], cx[:, bx1] - cx[:, bx0])


def _blur3(v):
    """the 11 x 11 x 11 Gaussian of ssim_matlab over (3, 32, 32), replicate padding, separable"""
    v = np.pad(v, 5, mode="edge")
    for ax in range(3):
        v = np.lib.stride_tricks.sliding_window_view(v, 11, axis=ax) @ G11
    return v


def ssim(a, b):
    """Practical-RIFE's ssim_matlab of two (3, 32, 32) thumbnails in [0, 1]"""
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu1, mu2 = _blur3(a), _blur3(b)
    s1, s2, s12 = _blur3(a * a) - mu1 ** 2, _blur3(b * b) - mu2 ** 2, _blur3(a * b) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    v1, v2 = 2.0 * s12 + C2, s1 + s2 + C2
    return float((((2 * mu1 * mu2 + C1) * v1) / ((mu1 ** 2 + mu2 ** 2 + C1) * v2)).mean())


PROBE_W = 320                  # the long side of the probe frames (the scene-cut test's RIFE)


def probe_size(h, w):
    """(width, height) of an h x w frame's probe frame: its long side PROBE_W"""
    return (PROBE_W, max(1, round(h * PROBE_W / w))) if w >= h else (max(1, round(w * PROBE_W / h)), PROBE_W)


def probe_frame(frame):
    """the frame for the cut test's RIFE: its long side 320 (PIL's BOX), uint8"""
    from PIL import Image
    return np.asarray(Image.fromarray(frame).resize(probe_size(*frame.shape[:2]), Image.BOX), np.uint8)


def box_coeffs(n_in, n_out):
    """PIL's BOX filter from n_in to n_out samples as its 8-bit passes take it (Pillow's Resample.c:
    precompute_coeffs, then normalize_coeffs_8bpc): bounds (n_out, 2) (the first sample, the count) and
    the int32 coefficients (n_out, ks) in 22 bits"""
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = 0.5 * fs
    ks = int(math.ceil(support)) * 2 + 1
    bounds = np.zeros((n_out, 2), np.int32)
    kk = np.zeros((n_out, ks), np.int32)
    for o in range(n_out):
        center = (o + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)
        n = min(int(center + support + 0.5), n_in) - lo
        k = [1.0 if -0.5 < (j + lo - center + 0.5) / fs <= 0.5 else 0.0 for j in range(n)]
        ww = sum(k)
        for j, v in enumerate(k):
            v = v / ww if ww != 0.0 else v
            kk[o, j] = int(-0.5 + v * (1 << 22)) if v < 0 else int(0.5 + v * (1 << 22))
        bounds[o] = (lo, n)
    return bounds, kk


@functools.lru_cache(maxsize=8)
def probe_plan(h, w):
    """the probe frame's passes for the GPU (ENC_RESAMPLE_U8, probe_frame's bits): PIL resizes
    horizontally (the rows the vertical pass reads, y0 .. y1) then vertically, each pass only when its
    size changes"""
    W, H = probe_size(h, w)
    bv, kv = box_coeffs(h, H)
    y0, y1 = int(bv[0, 0]), int(bv[-1, 0] + bv[-1, 1])
    out = dict(W=W, H=H, horiz=None, vert=None)
    if W != w:
        bh, kh = box_coeffs(w, W)
        out["horiz"] = (bh, kh, y0, y1)
        bv = bv.copy()
        bv[:, 0] -= y0
    if H != h:
        out["vert"] = (bv, kv)
    return out


class CutDetector:
    """The cut rule of a stream of pairs: each pair's thumbnails (thumbs()) and, with RIFE's probe,
    (R, D): the residual of the two probe frames warped to the middle against their raw difference
    (RIFE.residual). A real cut has no flow that explains it: R jumps against the pairs before.

        Rr = R / max(the median R of the last 8 moving pairs (D > 2), 2)
        old = SSIM < 0.2 or (SSIM < 0.45 and the grey difference's ratio > 6)
        fire = (Rr > 4.5 and R > 6 and SSIM < 0.75) or (old and Rr > 2.5)
        cut = fire, unless one of the 2 pairs before fired (a run of fires keeps its first: an
              animation's or an effect's churn)

    Without the probe (rd None) the thumbnails' rule alone (old)."""

    def __init__(self):
        self.hist = []          # the grey difference of the pairs
        self.rhist = []         # R of the moving pairs
        self.fires = []         # the raw fires of the pairs

    def __call__(self, th0, th1, rd=None):
        s = ssim(th0[0], th1[0])
        md = float(np.abs(th1[1] - th0[1]).mean()) * 255
        r = md / max(float(np.median(self.hist[-8:])), 0.5) if self.hist else 1.0
        self.hist.append(md)
        old = s < SSIM_CUT or (s < SSIM_LOW and r > RATIO)
        if rd is None:
            return old, s, r
        R, D = rd
        Rr = R / max(float(np.median(self.rhist[-8:])), 2.0) if self.rhist else R / 2.0
        if D > 2.0:
            self.rhist.append(R)
        fire = bool((Rr > 4.5 and R > 6.0 and s < 0.75) or (old and Rr > 2.5))
        cut = fire and not any(self.fires[-2:])
        self.fires.append(fire)
        return cut, s, r


# ---- the output's times ---------------------------------------------------------------

def schedule(n, fps_in, factor=None, fps=None, pts=None):
    """[(i, t)] of each output frame: t = 0 the input frame i, else the frame at t between i and
    i + 1. n input frames at fps_in (a Fraction); factor (an int > 1) or fps (a Fraction above
    fps_in); pts: the input frames' times in seconds (a variable rate), else i / fps_in."""
    if n <= 0:
        return []
    if factor is not None:
        F = int(factor)
        if F < 2:
            raise ValueError("factor must be 2 or more")
        out = [(i, Fraction(k, F)) for i in range(n - 1) for k in range(F)]
        return out + [(n - 1, Fraction(0))] * F
    fps = Fraction(fps)
    if fps <= fps_in:
        raise ValueError("fps must be above the input's (%s)" % float(fps_in))
    if pts is None:
        total = Fraction(n) / fps_in
        out, k = [], 0
        while Fraction(k) / fps < total:
            p = Fraction(k) / fps * fps_in
            i = math.floor(p)
            out.append((n - 1, Fraction(0)) if i >= n - 1 else (i, p - i))
            k += 1
        return out
    pts = [float(p) for p in pts]
    step = 1.0 / float(fps_in)
    total = pts[-1] - pts[0] + step
    out, k, i = [], 0, 0
    while k / float(fps) < total - 1e-9:
        T = pts[0] + k / float(fps)
        while i < n - 1 and pts[i + 1] <= T + 1e-9:
            i += 1
        if i >= n - 1:
            out.append((n - 1, 0.0))
        else:
            t = (T - pts[i]) / (pts[i + 1] - pts[i])
            out.append((i, 0.0 if t < 1e-6 else t))
        k += 1
    return out


# ---- ffmpeg ---------------------------------------------------------------------

def probe(src):
    """width, height, the rates (Fractions: r_frame_rate, avg_frame_rate), frames (or None), the
    audio codec (or None), the colour tags of a video file"""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,"
                          "color_space,color_transfer,color_primaries,color_range", "-of", "json", src],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise ValueError("video: not a file ffmpeg reads")
    streams = json.loads(out.stdout).get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError("video: no video stream")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def rate(s):
        try:
            f = Fraction(s)
            return f if f > 0 else None
        except (ValueError, ZeroDivisionError, TypeError):
            return None
    nb = v.get("nb_frames")
    colors = media.stream_colors(v)
    return dict(width=int(v["width"]), height=int(v["height"]), r_rate=rate(v.get("r_frame_rate")),
                avg_rate=rate(v.get("avg_frame_rate")), frames=int(nb) if nb and nb.isdigit() and int(nb) > 0 else None,
                audio=a.get("codec_name") if a else None, colors=colors)


def frame_times(src):
    """the video frames' presentation times (seconds, sorted) from the packets (no decode)"""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", src], capture_output=True, text=True, check=True).stdout
    return sorted(float(x) for x in out.split() if x and x != "N/A")


def count_frames(src):
    """the video stream's packets counted (no decode): its frame count"""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", src], capture_output=True, text=True)
    try:
        return int(out.stdout.strip().split(",")[0])
    except ValueError:
        raise ValueError("video: ffprobe could not count its frames") from None


def encode_cmd(src, dst, w, h, fps, codec, crf, audio, colors, gpu_encoder=False, threads=8, seconds=None):
    """ffmpeg's command: raw RGB frames on stdin at fps (a Fraction), the source's audio (copied
    when MP4 takes it, else AAC; none for audio=None) cut to seconds (the video's length; not
    -shortest, which can stop before the encoder's delayed frames), H.264 (libx264, or NVENC) or
    HEVC; yuv420p by the source's matrix and range, tagged so (yuv_colors)"""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h),
           "-r", "%d/%d" % (fps.numerator, fps.denominator), "-i", "-"]
    if audio:
        cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-c:a", "copy"] if audio in MP4_AUDIO else ["-c:a", "aac", "-b:a", "192k"]
    if codec == "h264":
        cmd += (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"] if gpu_encoder
                else ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-threads", str(threads)])
    elif codec == "hevc":
        cmd += (["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"] if gpu_encoder
                else ["-c:v", "libx265", "-preset", "medium", "-crf", str(crf), "-x265-params", "pools=%d" % threads])
        cmd += ["-tag:v", "hvc1"]
    else:
        raise ValueError("codec must be h264 or hevc")
    cmd += media.yuv_args(colors)
    if audio and seconds is not None:
        cmd += ["-t", "%.6f" % seconds]
    return cmd + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]


class _Slots:
    """A pool of pinned frames (uint8 (h, w, 3)) for the GPU's path: take() one (waiting for a free
    one), hold() it again for another user, drop() a hold (the frame back to the pool at the last)."""

    def __init__(self, shape, n):
        from reclank.backends.cuda.gpumm import pinned
        self.free = queue.Queue()
        self.refs = {}
        self.lock = threading.Lock()
        self.closed = False
        for _ in range(n):
            self.free.put(pinned(shape, np.uint8, free=True))

    def take(self):
        while True:
            try:
                fr = self.free.get(timeout=0.5)
                break
            except queue.Empty:
                if self.closed:
                    raise RuntimeError("video: the frames' pool closed") from None
        with self.lock:
            self.refs[id(fr)] = 1
        return fr

    def hold(self, fr):
        with self.lock:
            self.refs[id(fr)] += 1

    def drop(self, fr):
        with self.lock:
            self.refs[id(fr)] -= 1
            last = self.refs[id(fr)] == 0
        if last:
            self.free.put(fr)

    def close(self):
        self.closed = True


def _read_into(f, fr):
    """read f into the bytes of fr until full or the end; the bytes read"""
    mv, n = memoryview(fr.reshape(-1)), 0
    while n < len(mv):
        r = f.readinto(mv[n:])
        if not r:
            break
        n += r
    return n


def interpolate_video(rife, src, dst, factor=2, fps=None, scene_cuts=True, slowmo=False, codec="h264", crf=16,
                      gpu_encoder=None, max_frames=None, progress=None, gate=None):
    """Interpolate a video file into dst (MP4). factor (2, 3, 4, ...) or fps (a target rate: a
    Fraction, int, float or "num/den" string). slowmo: the output at the input's rate (the frames
    spread over factor x the time), no audio. progress(i, n) after each output frame (it may raise
    to cancel); gate: a lock each frame's GPU work holds (the server's). gpu_encoder: NVENC (its
    constant quality -cq crf) in place of libx264 / libx265 (its crf); None: NVENC when the model
    runs on the GPU. Returns a dict of the run."""
    if gpu_encoder is None:
        gpu_encoder = bool(getattr(rife, "gpu", False))
    info = probe(src)
    w, h = info["width"], info["height"]
    fps_in = info["avg_rate"] or info["r_rate"] or Fraction(25)
    vfr = (info["r_rate"] is not None and info["avg_rate"] is not None and
           abs(float(info["r_rate"]) - float(info["avg_rate"])) > 1e-3 * float(info["r_rate"]))
    pts = None
    if vfr:
        pts = frame_times(src)
        fps_in = info["r_rate"]
    if fps is not None:
        factor = None
        fps = Fraction(fps) if not isinstance(fps, str) else Fraction(fps)
        if slowmo:
            raise ValueError("slowmo takes a factor, not fps")
        if pts is None and (fps / fps_in).denominator == 1:
            factor, fps = int(fps / fps_in), None          # (an integer ratio: the factor's schedule)
    elif factor is None or int(factor) < 2:
        raise ValueError("factor must be 2 or more")
    if pts is not None and factor is not None:
        fps, factor = fps_in * int(factor), None           # (a variable rate: by the timestamps)
    n_in = info["frames"] if pts is None else len(pts)
    if n_in is None:
        n_in = count_frames(src)
    if max_frames:
        n_in = min(n_in, max_frames)
        if pts is not None:
            pts = pts[:n_in]
    fps_out = fps_in if slowmo else (fps if fps is not None else fps_in * int(factor))

    # the GPU's path (rife.gpu_io): the frames read straight into pinned slots and uploaded as their
    # bytes, the output frames downloaded into pinned slots and written from them (a slot back to its
    # pool when the writer and the loop are done with it); the thumbnails' inputs and the probe frame
    # made on the GPU (RIFE.aux). Else each frame's thumbnails and probe frame on the decoder's thread.
    gpu_io = bool(getattr(rife, "gpu_io", False))
    ahead, behind = (4, 6) if gpu_io else (16, 16)
    frames_q = queue.Queue(maxsize=ahead)
    out_q = queue.Queue(maxsize=behind)
    slots_in = _Slots((h, w, 3), ahead + behind + 3) if gpu_io else None
    slots_out = _Slots((h, w, 3), 4) if gpu_io else None
    dec = media.popen(["ffmpeg", "-v", "error", "-nostdin", "-i", src, "-map", "0:v:0", "-vsync", "0", "-f", "rawvideo",
                  "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    media.pipe_size(dec.stdout)
    fsize = h * w * 3

    def decode():
        media.widen()
        k = 0
        try:
            while max_frames is None or k < max_frames:
                if gpu_io:
                    fr = slots_in.take()
                    if _read_into(dec.stdout, fr) < fsize:
                        slots_in.drop(fr)
                        break
                    frames_q.put((fr, None))
                else:
                    b = dec.stdout.read(fsize)
                    if len(b) < fsize:
                        break
                    fr = np.frombuffer(b, np.uint8).reshape(h, w, 3)
                    frames_q.put((fr, (thumbs(fr), probe_frame(fr)) if scene_cuts else None))
                k += 1
        finally:
            frames_q.put(None)
    threading.Thread(target=decode, daemon=True).start()

    plan = schedule(n_in, fps_in, factor, fps, pts)
    enc = media.popen(encode_cmd(src, dst, w, h, fps_out, codec, crf, None if slowmo else info["audio"], info["colors"],
                            gpu_encoder, seconds=float(len(plan) / fps_out)), stdin=subprocess.PIPE)
    media.pipe_size(enc.stdin)
    err = []

    def write():
        media.widen()
        try:
            while True:
                e = out_q.get()
                if e is None:
                    break
                enc.stdin.write(e[0])
                if e[1] is not None:
                    e[1].drop(e[2])
        except Exception as exc:    # noqa: BLE001 - reported after the loop
            err.append(exc)
            while out_q.get() is not None:
                pass
        finally:
            try:
                enc.stdin.close()
            except OSError:
                pass
    wt = threading.Thread(target=write, daemon=True)
    wt.start()

    t0 = time.time()
    gate = gate if gate is not None else threading.Lock()
    total = len(plan)
    cuts, state = [], dict(n_out=0, made=0, ended=False)

    def next_frame():
        item = frames_q.get()
        state["ended"] = item is None
        return item

    def emit(fr, pool=None):
        """fr to the encoder: a frame of a pool (its slot held for the writer: an input frame's another
        hold, an output frame's its own), or bytes"""
        if err:
            raise RuntimeError("ffmpeg's encoder failed (%s)" % err[0])
        if pool is slots_in and pool is not None:
            pool.hold(fr)
        out_q.put((fr.reshape(-1).data, pool, fr) if pool is not None else (np.ascontiguousarray(fr).tobytes(),
                                                                               None, None))
        state["n_out"] += 1
        if progress is not None:
            progress(state["n_out"], total)

    tp = thumb_plan(h, w) if scene_cuts and gpu_io else None

    def take(item, pr):
        """push a frame: its thumbnails and probe frame; with the probe (pr), the pair's (R, D) (the GPU's
        path: the two probe frames warped to the middle, and the probe frame / 255: residual())"""
        fr, aux = item
        rd = xp = None
        with gate:
            rife.push(fr)
            if tp is not None:
                a = rife.aux()
                aux = (thumbs_of(tp, a["taps"], a["sums"]), a["probe"])
            if pr is not None:
                pr.push(aux[1])
                if gpu_io:
                    xp = aux[1].astype(np.float32) / np.float32(255)
                    if pr.n > 1:
                        pr.frame(0.5, fetch=False)
                        rd = pr.warped()
                elif pr.n > 1:
                    rd = pr.residual()
        return fr, aux, rd, xp

    def residual(prev, item):
        """the pair's (R, D): RIFE.residual's bits (the GPU's path: its means here, while the GPU makes
        the pair's first frame)"""
        if not gpu_io or item[2] is None:
            return item[2]
        return mean_abs(*item[2]) * 255, mean_abs(prev[3], item[3]) * 255

    try:
        with RESIDENCY.use(*rife.runners().values()):
            rife.reset()
            item = next_frame()
            if item is None:
                raise ValueError("video: no frames")
            det = CutDetector()
            pr = rife.prober() if scene_cuts and hasattr(rife, "prober") else None
            if pr is not None:
                pr.reset()
            if tp is not None:
                rife.aux_plan(h, w, tp, probe_plan(h, w))
            prev = take(item, pr)
            i, k = 0, 0                     # the left frame of the pair, the next entry of the plan
            while True:
                item = next_frame() if i + 1 < n_in else None
                if item is None:
                    break
                item = take(item, pr)
                ahead = None                # (the GPU's path: the pair's first frame made before the cut test)
                if gpu_io:
                    j = k
                    while j < len(plan) and plan[j][0] == i and plan[j][1] == 0:
                        j += 1
                    ahead = j if j < len(plan) and plan[j][0] == i else None
                    if ahead is not None:
                        with gate:
                            rife.frame(float(plan[ahead][1]), fetch=False)
                cut = bool(scene_cuts and det(prev[1][0], item[1][0], residual(prev, item))[0])
                if cut:
                    cuts.append(i)
                while k < len(plan) and plan[k][0] == i:
                    t = plan[k][1]
                    if t == 0 or cut:
                        emit(prev[0], slots_in)
                    else:
                        out = slots_out.take() if gpu_io else None
                        with gate:
                            if k == ahead:
                                fr = rife.fetch(out)
                            else:
                                fr = rife.frame(float(t), out=out) if gpu_io else rife.frame(float(t))
                        state["made"] += 1
                        emit(fr, slots_out)
                    k += 1
                if gpu_io:
                    slots_in.drop(prev[0])
                prev, i = item, i + 1
            if i + 1 < n_in:                 # (fewer frames than counted: the plan of those there are)
                plan = schedule(i + 1, fps_in, factor, fps, pts[:i + 1] if pts else None)
                n_in = i + 1
                k = next((j for j, e in enumerate(plan) if e[0] == i), len(plan))
            for e in plan[k:]:               # the last frame's (repeated to the length)
                emit(prev[0], slots_in)
    except BaseException:
        enc.kill()
        raise
    finally:
        out_q.put(None)
        wt.join()
        if dec.poll() is None:
            dec.kill()
        dec.wait()
        while not state["ended"]:              # (the decoder's thread ends)
            if gpu_io:
                slots_in.close()
            next_frame()
    n_out, n_made = state["n_out"], state["made"]
    rc = enc.wait()
    if err or rc != 0:
        raise RuntimeError("ffmpeg's encoder failed (%s)" % (err[0] if err else "exit %d" % rc))
    secs = time.time() - t0
    return dict(frames_in=n_in, frames=n_out, interpolated=n_made, cuts=cuts, seconds=round(secs, 2),
                encoder="nvenc" if gpu_encoder else ("libx264" if codec == "h264" else "libx265"),
                fps_in=str(fps_in), fps_out=str(fps_out), speed=round(n_out / max(secs, 1e-9), 2), width=w, height=h,
                vfr=bool(vfr), audio=bool(info["audio"]) and not slowmo)
