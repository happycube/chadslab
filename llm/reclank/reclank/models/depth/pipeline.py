"""2D to stereo 3D: the depth models, the disparity and the warp together, for
an image (Stereo.image) or a video file (Stereo.video: ffmpeg in, VDA-Small's
windows, both eyes a frame, ffmpeg out with the source's audio, the st3d box).
Models load at their first use: MoGe-2 ("moge-2") and DA3MONO ("da3mono") for
images, VDA-Small ("vda-small") for video."""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from fractions import Fraction

import numpy as np

from reclank.encoders.runtime import RESIDENCY
from reclank.models import media
from reclank.models.depth import stereo as S

IMAGE_MODELS = ("moge-2", "da3mono")
MP4_AUDIO = ("aac", "mp3", "ac3", "eac3", "alac")


class Stereo:
    """The stereo converter of the model directories given (moge, da3mono,
    vda: None for one not served)."""

    def __init__(self, moge=None, da3mono=None, vda=None, gpu=False, cpu_threads=None, encoder=None):
        self.paths = {"moge-2": moge, "da3mono": da3mono, "vda-small": vda}
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.models = {}
        self.warper = S.Warper(gpu=gpu, cpu_threads=cpu_threads)
        self.encoder = encoder
        self.lock = threading.Lock()

    def available(self):
        return [k for k, v in self.paths.items() if v]

    def model(self, name):
        if not self.paths.get(name):
            raise ValueError("model %s is not loaded (the stereo models: %s)" % (name, ", ".join(self.available())))
        with self.lock:
            if name not in self.models:
                if name == "moge-2":
                    from reclank.models.depth.moge import MoGe2
                    self.models[name] = MoGe2(self.paths[name], gpu=self.gpu, cpu_threads=self.cpu_threads)
                elif name == "da3mono":
                    from reclank.models.depth.da3 import DA3Mono
                    self.models[name] = DA3Mono(self.paths[name], gpu=self.gpu, cpu_threads=self.cpu_threads)
                else:
                    from reclank.models.depth.vda import VDASmall
                    self.models[name] = VDASmall(self.paths[name], gpu=self.gpu, cpu_threads=self.cpu_threads)
                self.models[name].runner.label = name
        return self.models[name]

    def runners(self):
        """{part: GPU runner} of the loaded parts (RESIDENCY's)"""
        if not self.gpu:
            return {}
        out = {n: m.runner for n, m in self.models.items()}
        out.update({n + "/prep": m.prep for n, m in self.models.items() if hasattr(m, "prep")})
        out["warp"] = self.warper.runner
        return out

    # ---- images ----

    def depth(self, img, model="moge-2", tokens=None):
        """dict(D (normalized disparity 0..1), valid, depth or disparity, fx when
        metric) at the model's size"""
        m = self.model(model)
        if model == "moge-2":
            g = m.geometry(img, num_tokens=tokens)
            inv = np.where(g["valid"], 1.0 / np.where(g["valid"], g["depth"], 1.0), 0.0)
            D, lo, hi = S.normalized(inv, g["valid"])
            return dict(D=D, valid=g["valid"], depth=g["depth"], fx=g["fx"], hfov=g["hfov"], metric=True)
        if model == "da3mono":
            r = m.depth(img)
            inv = np.where(r["valid"], 1.0 / r["depth"], 0.0)
            D, lo, hi = S.normalized(inv, r["valid"])
            return dict(D=D, valid=r["valid"], depth=r["depth"], metric=False)
        raise ValueError("model must be one of %s for images" % ", ".join(IMAGE_MODELS))

    def image(self, img, model="moge-2", layout="sbs", fill="edge", divergence=2.0, convergence="auto",
              mode="relative", baseline_mm=63.0, z_near=0.6, clamp=1.5, tokens=None):
        """dict(image: the packed stereo uint8 image of an uint8 image (H, W, 3),
        tag: the file-name tag, info: the settings used)."""
        if fill not in S.FILLS:
            raise ValueError("fill must be one of %s" % ", ".join(S.FILLS))
        if not 0 <= divergence <= 10:
            raise ValueError("divergence must be 0 to 10 (% of the width)")
        H, W = img.shape[:2]
        with RESIDENCY.use(*self.runners().values()):
            d = self.depth(img, model, tokens)
            info = dict(model=model, layout=layout, fill=fill, mode=mode)
            if mode == "metric":
                if not d.get("metric"):
                    raise ValueError("mode metric needs a metric model (moge-2)")
                z0 = None if convergence in (None, "auto") else float(convergence)
                shift, z0 = S.metric_shift(d["depth"], d["valid"], d["fx"], W, baseline_mm, z0, z_near, clamp)
                info.update(convergence_m=round(z0, 3), baseline_mm=baseline_mm, hfov=round(d["hfov"], 2))
            elif mode == "relative":
                D0 = S.convergence_of(d["D"], d["valid"]) if convergence in (None, "auto") else float(convergence)
                shift = S.relative_shift(d["D"], D0, W, divergence, clamp)
                info.update(convergence=round(D0, 4), divergence=divergence)
            else:
                raise ValueError("mode must be relative or metric")
            if layout == "anaglyph":
                out = S.anaglyph(self.warper.warp(img, shift, "sbs", fill))
            else:
                out = self.warper.warp(img, shift, layout, fill)
        info.update(shift_px=[round(float(shift.min()), 2), round(float(shift.max()), 2)])
        return dict(image=out, tag=S.TAGS[layout], info=info, D=d["D"])

    # ---- video ----

    def video(self, src, dst, layout="sbs", fill="edge", divergence=2.0, convergence="auto", clamp=1.5,
              codec="hevc", quality=21, ema=0.1, max_frames=None, progress=None, gate=None):
        """Convert a video file to a stereo one (dst .mp4 or .mkv) with VDA-Small.
        progress(i, n) after each frame (n: the frame count ffprobe gives, or
        None); it may raise to cancel. gate: a lock each frame's GPU work holds
        (the server's: other requests run between frames). Returns a dict of
        the run."""
        if layout not in S.LAYOUTS:
            raise ValueError("layout must be one of %s" % ", ".join(S.LAYOUTS))
        info = probe(src)
        H, W = info["height"], info["width"]
        n_total = info.get("frames")
        if max_frames:
            n_total = min(n_total or max_frames, max_frames)
        oh, ow = S.out_size(H, W, layout)
        vda = self.model("vda-small")
        frames_q = queue.Queue(maxsize=48)
        # (ffmpeg and the decoder's and writer's threads on the start CPUs: models/media.py)
        dec = media.popen(["ffmpeg", "-v", "error", "-nostdin", "-i", src, "-map", "0:v:0", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        fsize = H * W * 3

        def decode():
            media.widen()
            k = 0
            try:
                while max_frames is None or k < max_frames:
                    b = dec.stdout.read(fsize)
                    if len(b) < fsize:
                        break
                    frames_q.put(np.frombuffer(b, np.uint8).reshape(H, W, 3))
                    k += 1
            finally:
                frames_q.put(None)
        threading.Thread(target=decode, daemon=True).start()
        held = []                                   # the frames whose disparity is not out yet

        waits = [0.0]

        def frames():
            while True:
                tw = time.time()
                f = frames_q.get()
                waits[0] += time.time() - tw
                if f is None:
                    return
                held.append(f)
                yield f
        enc_cmd = encode_cmd(src, dst, ow, oh, info["fps"], codec, quality, info.get("audio"), layout,
                             self.encoder if self.encoder is not None else self.gpu, colors=info.get("colors"),
                             seconds=out_frames(src, info, max_frames) / Fraction(info["fps"]) if info.get("audio") else None)
        enc = media.popen(enc_cmd, stdin=subprocess.PIPE)
        out_q = queue.Queue(maxsize=8)
        err = []

        def write():
            media.widen()
            try:
                while True:
                    b = out_q.get()
                    if b is None:
                        break
                    enc.stdin.write(b)
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
        lo = hi = D0 = None
        n = 0
        times = np.zeros(4)         # the depth (decode waits included), the disparity, the warp, the output queue
        try:
            gate = gate if gate is not None else threading.Lock()
            with RESIDENCY.use(*self.runners().values()):
                it = vda.video(frames())
                while True:
                    with gate:
                        ta = time.time()
                        item = next(it, None)
                        if item is None:
                            break
                        tb = time.time()
                        _i, disp = item
                        frame = held.pop(0)
                        sub = disp[::4, ::4]
                        flo, fhi = float(np.percentile(sub, 1)), float(np.percentile(sub, 99.5))
                        lo = flo if lo is None else lo + ema * (flo - lo)
                        hi = fhi if hi is None else hi + ema * (fhi - hi)
                        D, _, _ = S.normalized(disp, np.ones(disp.shape, bool), lo, hi)
                        if convergence in (None, "auto"):
                            c = float(np.median(S.centre(D)[::2, ::2]))
                            D0 = c if D0 is None else D0 + ema * (c - D0)
                        else:
                            D0 = float(convergence)
                        shift = S.relative_shift(D, D0, W, divergence, clamp)
                        tc = time.time()
                        packed = self.warper.warp(frame, shift, layout, fill)
                        td = time.time()
                    if err:
                        raise RuntimeError("ffmpeg's encoder failed (%s)" % err[0])
                    out_q.put(packed.tobytes())
                    te = time.time()
                    times += np.array([tb - ta, tc - tb, td - tc, te - td])
                    n += 1
                    if progress is not None:
                        progress(n, n_total)
        except BaseException:
            enc.kill()
            raise
        finally:
            out_q.put(None)
            wt.join()
            if dec.poll() is None:
                dec.kill()
            dec.wait()
        rc = enc.wait()
        if err or rc != 0:
            raise RuntimeError("ffmpeg's encoder failed (%s)" % (err[0] if err else "exit %d" % rc))
        st3d = dst.lower().endswith(".mp4") and S.inject_st3d(dst, layout)
        secs = time.time() - t0
        return dict(frames=n, seconds=round(secs, 2), fps=round(n / max(secs, 1e-9), 2), width=ow, height=oh,
                    layout=layout, st3d=bool(st3d), codec=enc_cmd[enc_cmd.index("-c:v") + 1],
                    ms_a_frame=dict(zip(("depth", "disparity", "warp", "queue", "decode_wait"),
                                        [round(1000 * float(x) / max(n, 1), 1) for x in list(times) + waits])))


def probe(src):
    """width, height, fps (a string ffmpeg takes), frames (or None), the audio
    codec (or None) of a video file"""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,"
                          "color_space,color_transfer,color_primaries,color_range",
                          "-of", "json", src], capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError("%s: no video stream" % src)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    nb = v.get("nb_frames")
    fps = v.get("r_frame_rate") or v.get("avg_frame_rate") or "25"
    return dict(width=int(v["width"]), height=int(v["height"]), fps=fps,
                frames=int(nb) if nb and nb.isdigit() and int(nb) > 0 else None,
                audio=a.get("codec_name") if a else None, colors=media.stream_colors(v))


def count_frames(src):
    """the video stream's packets counted (no decode): its frame count, for a
    container that does not store it"""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", src], capture_output=True, text=True, check=True)
    return int(out.stdout.strip().split(",")[0])


def out_frames(src, info, max_frames=None):
    """the output's frame count: the source's (counted when the container does
    not store it), at most max_frames"""
    n = info.get("frames") or count_frames(src)
    return min(n, max_frames) if max_frames else n


def encode_cmd(src, dst, w, h, fps, codec, quality, audio, layout, gpu, seconds=None, colors=None):
    """ffmpeg's command: raw RGB frames on stdin, the source's audio (copied when
    the container takes it, else AAC) cut to seconds (the video's length; not
    -shortest, which ended a clip shorter than the encoder's delay before its
    first video packet: no video), HEVC (or H.264) by NVENC on a GPU, else
    x265 / x264; Matroska's StereoMode for .mkv; yuv420p by the source's colour
    tags (media.yuv_colors: its matrix and range, tagged so)"""
    mkv = dst.lower().endswith(".mkv")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h), "-r", str(fps),
           "-i", "-"]
    if audio:
        cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-c:a", "copy"] if (mkv or audio in MP4_AUDIO) else ["-c:a", "aac", "-b:a", "192k"]
    if codec == "hevc":
        cmd += (["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"] if gpu else
                ["-c:v", "libx265", "-preset", "medium", "-crf", str(quality)])
        if not mkv:
            cmd += ["-tag:v", "hvc1"]
    elif codec == "h264":
        if gpu and w > 4096:
            raise ValueError("H.264 by NVENC is at most 4096 wide: take hevc")
        cmd += (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"] if gpu else
                ["-c:v", "libx264", "-preset", "medium", "-crf", str(quality)])
    else:
        raise ValueError("codec must be hevc or h264")
    cmd += media.yuv_args(colors) + ["-pix_fmt", "yuv420p"]
    if audio and seconds is not None:
        cmd += ["-t", "%.6f" % seconds]
    if mkv:
        cmd += ["-metadata:s:v:0", "stereo_mode=%s" % ("left_right" if layout.startswith("sbs") else "top_bottom")]
    return cmd + [dst]
