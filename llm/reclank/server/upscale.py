"""The upscalers of the generation server (server/generate.py: --seedvr2 DIR, --realesrgan DIR):
SeedVR2-3B (reclank/models/seedvr2: one-step restoration and upscaling of images and videos,
Apache 2.0) and Real-ESRGAN (reclank/models/realesrgan, BSD-3).

    POST /v1/images/upscale  multipart/form-data ("image": the file, any format PIL reads, and
                             the fields below) or JSON ("image": base64 or a data URL):
                             "model": "seedvr2-3b" | "realesrgan-x4plus" | "realesrgan-x2plus" |
                                 "realesrgan-x4plus-anime" | "realesr-general-x4v3" |
                                 "realesr-animevideov3" (default: the first loaded),
                             "scale": SeedVR2's factor (2 by default; any from 1 to 8), Real-ESRGAN's
                                 must be its own (4 or 2),
                             "size": SeedVR2 "WxH" in place of scale: the official rule, its
                                 area at the input's aspect ratio (then the sides cropped to
                                 multiples of 16),
                             "seed" (SeedVR2's noise), "color_fix": "wavelet" (default, as the
                                 official script) | "none", "denoise_strength" (realesr-general-x4v3:
                                 0.5 by default, as inference_realesrgan.py; 1: the model alone),
                             "response_format": "b64_json", "output_format": "png"
                             -> {"created", "data": [{"b64_json", "seed", "size"}]}
    POST /v1/videos/upscale  multipart/form-data: "video" (any format ffmpeg reads), "model"
                             ("seedvr2-3b" or "realesr-animevideov3": frame by frame), "scale",
                             "size", "seed", "color_fix": a job as POST /v1/videos makes; then GET
                             /v1/videos/{id} (status, progress), GET /v1/videos/{id}/content (MP4:
                             H.264 at the input's frame rate, the input's audio copied), DELETE.

SeedVR2-7B (--seedvr2-7b DIR, "seedvr2-7b": its blocks streamed in F20) on the same routes: one
SeedVR2 model loaded at a time (the 3B by default; a request of the other loads it, the 7B from an
HDD in minutes, and drops the one loaded). SeedVR2's output: at most 2048 x 2048 of area for an image and 1280 x 720 for a video (the
float32 VAE on a 24 GB GPU; larger frames wait for the pinned-host phase); a video at most 121
frames (padded to 4 k + 1 by its last frame as the official, the output cut back). Real-ESRGAN:
an output of at most 4096 x 4096 of area."""
from __future__ import annotations

import base64
import collections
import io
import json
import os
import subprocess
import threading
import time

import numpy as np

from reclank.models import media

SEEDVR2_ID = "seedvr2-3b"
IMAGE_AREA = 2048 * 2048        # SeedVR2's output (an image)
VIDEO_AREA = 1280 * 720         # SeedVR2's output (a video frame)
VIDEO_FRAMES = 121
ESRGAN_AREA = 4096 * 4096
IN_PIXELS = 50_000_000


class SeedVR2Slot:
    """The SeedVR2 model of the server: one variant loaded at a time (the 3B's F20 8.5 GB of host
    memory, the 7B's 20.9 GB), the other loaded on its first request (the 7B from an HDD: minutes)
    after the loaded one is dropped."""

    def __init__(self, paths, gpu, cpu_threads):
        self.paths, self.gpu, self.cpu_threads = paths, gpu, cpu_threads     # variant: directory
        self.variant, self.m = None, None

    def get(self, variant):
        if self.variant != variant:
            from reclank.models.seedvr2.pipeline import SeedVR2
            if self.m is not None:
                self.m.close()
                self.m = None
                import gc
                gc.collect()
            t0 = time.time()
            self.m = SeedVR2(self.paths[variant], gpu=self.gpu, cpu_threads=self.cpu_threads, variant=variant)
            self.variant = variant
            print("seedvr2: the %s loaded in %.1f s" % (variant, time.time() - t0), flush=True)
        return self.m


class Upscaler:
    """A model of the routes: SeedVR2-3B or -7B (kind "seedvr2", a variant of a shared SeedVR2Slot)
    or a Real-ESRGAN model ("esrgan")."""

    def __init__(self, kind, path, model_id, gpu, cpu_threads, slot=None, variant=None):
        self.kind, self.model_id, self.gpu = kind, model_id, gpu
        self.path, self.cpu_threads = path, cpu_threads
        if kind == "seedvr2":
            self.slot, self.variant = slot, variant
        else:
            from reclank.models.realesrgan.model import MODELS, RealESRGAN
            self.m = RealESRGAN(path, model_id, gpu=gpu, cpu_threads=cpu_threads)
            self.scale = MODELS[model_id][2]
            self.denoise = 0.5

    @property
    def m(self):
        return self.slot.get(self.variant) if self.kind == "seedvr2" else self._m

    @m.setter
    def m(self, value):
        self._m = value

    def runners(self):
        if self.kind == "seedvr2":
            # (the loaded variant's only: the other holds nothing)
            return self.slot.m.runners() if self.slot.variant == self.variant else {}
        return {"net": self.m.runner} if self.gpu else {}

    def describe(self):
        d = {"id": self.model_id, "object": "model", "owned_by": "reclank", "kind": "upscale"}
        if self.kind == "seedvr2":
            d.update(video=True, max_area=IMAGE_AREA, max_video_area=VIDEO_AREA, max_frames=VIDEO_FRAMES)
        else:
            d.update(scale=self.scale, video=self.model_id == "realesr-animevideov3", max_area=ESRGAN_AREA)
        return d

    def options(self, f, h, w, video=False):
        """The checked options of a request's fields f (name: str) for an input of h x w: res (or
        the ESRGAN's scale), seed, color_fix. ValueError: a bad request."""
        out = {}
        if self.kind == "seedvr2":
            from reclank.models.seedvr2.pipeline import output_size
            size = f.get("size")
            if size:
                try:
                    rw, rh = (int(v) for v in size.lower().split("x"))
                except ValueError:
                    raise ValueError("size must be WIDTHxHEIGHT") from None
            else:
                scale = float(f.get("scale") or 2)
                if not 1 <= scale <= 8:
                    raise ValueError("scale must be 1 to 8")
                rh, rw = round(h * scale), round(w * scale)
            _, (ch, cw) = output_size(h, w, rh, rw)
            area = VIDEO_AREA if video else IMAGE_AREA
            if ch < 16 or cw < 16 or ch * cw > area:
                raise ValueError("the output (%dx%d) must be at most %d pixels (%s) and at least 16 a side" % (
                    cw, ch, area, "1280 x 720" if video else "2048 x 2048"))
            out["res"] = (rh, rw)
            fix = f.get("color_fix") or "wavelet"
            if fix not in ("wavelet", "none"):
                raise ValueError("color_fix must be \"wavelet\" or \"none\"")
            out["color_fix"] = fix
            seed = f.get("seed")
            out["seed"] = int(seed) if seed not in (None, "") else int.from_bytes(os.urandom(4), "little")
        else:
            if float(f.get("scale") or self.scale) != self.scale:
                raise ValueError("%s upscales by %d" % (self.model_id, self.scale))
            if f.get("size"):
                raise ValueError("size is SeedVR2's (Real-ESRGAN upscales by its scale)")
            if h * w * self.scale ** 2 > ESRGAN_AREA:
                raise ValueError("the output must be at most %d pixels (4096 x 4096)" % ESRGAN_AREA)
            ds = f.get("denoise_strength")
            if ds not in (None, ""):
                if self.model_id != "realesr-general-x4v3":
                    raise ValueError("denoise_strength is realesr-general-x4v3's")
                ds = float(ds)
                if not 0 <= ds <= 1:
                    raise ValueError("denoise_strength must be 0 to 1")
                if ds != self.denoise:          # the mixed weights of another strength: the model again
                    from reclank.models.realesrgan.model import RealESRGAN
                    self.m.release()
                    self.m = RealESRGAN(self.path, self.model_id, gpu=self.gpu, cpu_threads=self.cpu_threads,
                                        denoise=ds)
                    self.denoise = ds
            out["seed"] = None
        return out

    def run(self, frames, opts, progress=None):
        """The upscaled frames (t, H, W, 3) uint8 of frames (t, h, w, 3) uint8."""
        if self.kind == "seedvr2":
            return self.m.restore(frames, res=opts["res"], seed=opts["seed"], color_fix=opts["color_fix"],
                                  progress=progress)
        from reclank.encoders.runtime import RESIDENCY
        with RESIDENCY.use(*self.runners().values()):
            out = []
            for i, fr in enumerate(frames):
                out.append(self.m.upscale(fr))
                if progress is not None:
                    progress("frames", i + 1, len(frames))
        return np.stack(out)


def read_image(data):
    """An RGB image (h, w, 3) uint8 of an image file's bytes (EXIF orientation applied; an alpha
    dropped)."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if w * h > IN_PIXELS:
                raise ValueError("the image is %dx%d: at most %d pixels" % (w, h, IN_PIXELS))
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("image: not an image file PIL reads") from None


def image_request(raw, ctype, multipart):
    """(fields {name: str}, image (h, w, 3) uint8) of an upscale request: multipart (an "image"
    file) or JSON ("image": base64 or a data URL)."""
    if (ctype or "").startswith("multipart/"):
        form = multipart(raw, ctype)
        files = form.get("image", [])
        if len(files) != 1 or not files[0][0]:
            raise ValueError("image is required: one image file")
        f = {}
        for k, v in form.items():
            if k != "image":
                try:
                    f[k] = v[0][0].decode("utf-8").strip()
                except UnicodeDecodeError:
                    raise ValueError("%s must be UTF-8 text" % k) from None
        return f, read_image(files[0][0])
    body = json.loads(raw) if raw.strip() else {}
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    img = body.get("image")
    if not isinstance(img, str) or not img:
        raise ValueError("image is required: base64 of an image file (or a data URL)")
    if img.startswith("data:"):
        img = img.split(",", 1)[-1]
    try:
        data = base64.b64decode(img, validate=True)
    except ValueError:
        raise ValueError("image must be base64") from None
    f = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in body.items() if k != "image" and v is not None}
    return f, read_image(data)


def check_common(f):
    if (f.get("response_format") or "b64_json") != "b64_json":
        raise ValueError("response_format must be \"b64_json\" (this server hosts no URLs)")
    if (f.get("output_format") or "png") != "png":
        raise ValueError("output_format must be \"png\"")


def probe(path):
    """(width, height, fps, has audio) of a video by ffprobe."""
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate",
                        "-of", "json", path], capture_output=True, text=True)
    if p.returncode != 0:
        raise ValueError("video: not a file ffmpeg reads")
    streams = json.loads(p.stdout).get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError("video: no video stream")
    num, _, den = v.get("r_frame_rate", "24/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 24.0
    return int(v["width"]), int(v["height"]), fps, any(s.get("codec_type") == "audio" for s in streams)


def read_frames(path, w, h, limit):
    p = media.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", str(limit + 1), "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True)
    if p.returncode != 0:
        raise ValueError("video: ffmpeg could not decode it")
    n = len(p.stdout) // (w * h * 3)
    if n == 0:
        raise ValueError("video: no frames")
    if n > limit:
        raise ValueError("video: at most %d frames" % limit)
    return np.frombuffer(p.stdout, np.uint8)[:n * w * h * 3].reshape(n, h, w, 3)


def write_video(frames, path, fps, source=None, colors=None):
    """H.264 MP4 of frames at fps (yuv420p by colors, the source's tags); the audio of source (when it has
    some) copied in."""
    from reclank.models.wan.pipeline import write_mp4
    if source is None:
        write_mp4(frames, path, fps, colors=colors)
        return
    tmp = path + ".v.mp4"
    write_mp4(frames, tmp, fps, colors=colors)
    p = media.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", source, "-map", "0:v", "-map", "1:a",
                        "-c", "copy", "-shortest", "-movflags", "+faststart", path], capture_output=True)
    os.remove(tmp)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg: %s" % p.stderr.decode(errors="replace").strip())


class _Cancelled(Exception):
    pass


class VideoUpscales:
    """The video upscale jobs: their queue and worker, the files (ID.json, ID.input, ID.mp4) in
    their own directory (WORK/upscale), in the shape of the /v1/videos jobs."""

    def __init__(self, upscalers, work_dir, server):
        self.ups = upscalers            # id: Upscaler (those that do videos)
        self.dir, self.server = os.path.join(work_dir, "upscale"), server
        os.makedirs(self.dir, exist_ok=True)
        self.jobs = collections.OrderedDict()
        self.queue = collections.deque()
        self.cancel = set()
        self.cv = threading.Condition()
        for name in sorted(os.listdir(self.dir)):
            if not name.endswith(".json"):
                continue
            try:
                job = json.load(open(os.path.join(self.dir, name)))
            except (OSError, ValueError):
                continue
            if job.get("status") in ("queued", "in_progress"):
                job.update(status="failed", error={"code": "server_stopped", "message": "the server stopped"})
                self._save(job)
            self.jobs[job["id"]] = job
        threading.Thread(target=self._worker, name="upscales", daemon=True).start()

    def _file(self, vid, ext):
        return os.path.join(self.dir, vid + ext)

    def _save(self, job):
        tmp = self._file(job["id"], ".json.part")
        with open(tmp, "w") as f:
            json.dump({k: v for k, v in job.items() if not k.startswith("_")}, f)
        os.replace(tmp, self._file(job["id"], ".json"))

    @staticmethod
    def view(job):
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def submit(self, form):
        """A new job of a multipart form ({name: [(bytes, filename)]})."""
        files = form.get("video", [])
        if len(files) != 1 or not files[0][0]:
            raise ValueError("video is required: one video file")
        f = {}
        for k, v in form.items():
            if k != "video":
                f[k] = v[0][0].decode("utf-8", "replace").strip()
        model = f.get("model") or next(iter(self.ups))
        if model not in self.ups:
            raise ValueError("model must be one of %s" % ", ".join(self.ups))
        up = self.ups[model]
        vid = "video_" + os.urandom(12).hex()
        src = self._file(vid, ".input")
        with open(src, "wb") as fh:
            fh.write(files[0][0])
        try:
            w, h, fps, audio = probe(src)
            opts = up.options(f, h, w, video=True)
        except Exception:
            os.remove(src)
            raise
        job = {"id": vid, "object": "video", "model": model, "status": "queued", "progress": 0,
               "created_at": int(time.time()), "completed_at": None, "input_size": "%dx%d" % (w, h),
               "fps": round(fps, 3), "seed": opts["seed"], "error": None, "_opts": opts, "_audio": audio}
        with self.cv:
            self.jobs[vid] = job
            self._save(job)
            self.queue.append(vid)
            self.cv.notify()
        return self.view(job)

    def delete(self, vid):
        with self.cv:
            job = self.jobs.get(vid)
            if job is None:
                return None
            if job["status"] == "in_progress":
                self.cancel.add(vid)
            else:
                if vid in self.queue:
                    self.queue.remove(vid)
                self._remove(vid)
        return {"id": vid, "object": "video.deleted", "deleted": True}

    def _remove(self, vid):
        self.jobs.pop(vid, None)
        for ext in (".json", ".mp4", ".input", ".part.mp4"):
            try:
                os.remove(self._file(vid, ext))
            except FileNotFoundError:
                pass

    def _worker(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                vid = self.queue.popleft()
                job = self.jobs[vid]
                job["status"] = "in_progress"
                self._save(job)
            t0 = time.time()
            # the progress: decoding 2%, the VAE encode to 30%, the DiT 40%, the decode to 98%, the MP4 100%
            span = {"encode": (2, 30), "decode": (40, 98), "frames": (2, 98)}

            def progress(stage, i, n):
                if vid in self.cancel:
                    raise _Cancelled()
                a, b = span.get(stage, (0, 0))
                job["progress"] = int(a + (b - a) * i / max(n, 1))

            try:
                src = self._file(vid, ".input")
                w, h = (int(v) for v in job["input_size"].split("x"))
                frames = read_frames(src, w, h, VIDEO_FRAMES)
                job["progress"] = 2
                with self.server.lock:
                    out = self.ups[job["model"]].run(frames, job["_opts"], progress=progress)
                part = self._file(vid, ".part.mp4")
                write_video(out, part, job["fps"], src if job["_audio"] else None, colors=media.colors_of(src))
                os.replace(part, self._file(vid, ".mp4"))
                job.update(status="completed", progress=100, completed_at=int(time.time()),
                           size="%dx%d" % (out.shape[2], out.shape[1]), frames=int(out.shape[0]))
                if not getattr(self.server, "quiet", False):
                    print("upscale %s: %d frames %s -> %s in %.1f s" % (vid, out.shape[0], job["input_size"],
                                                                        job["size"], time.time() - t0), flush=True)
            except _Cancelled:
                pass
            except Exception as exc:  # noqa: BLE001 - a failed job reports its error
                job.update(status="failed", error={"code": type(exc).__name__, "message": str(exc)[:500]})
                print("upscale %s failed: %s" % (vid, exc), flush=True)
            with self.cv:
                if vid in self.cancel:
                    self.cancel.discard(vid)
                    self._remove(vid)
                else:
                    self._save(job)
