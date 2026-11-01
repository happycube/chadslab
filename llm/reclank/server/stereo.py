"""The 2D to stereo 3D routes of the generation server (reclank/models/depth,
docs/STEREO_PLAN.md; --stereo-moge, --stereo-da3, --stereo-vda):

    POST /v1/images/stereo   JSON {"image": base64 (or a data URL), ...} or
                             multipart/form-data (file "image", the same
                             fields): one stereo image
        "model": "moge-2" (default) | "da3mono"
        "layout": "sbs" (full side by side, default) | "sbs_half" | "ou" |
                  "ou_half" | "anaglyph" (red-cyan, a check) | "depth" (the
                  normalized disparity as a 16-bit PNG)
        "fill": "edge" (default) | "stretch"
        "divergence": the total parallax, % of the width (2.0)
        "convergence": "auto" (the centre's median) | a value of the
                       normalized disparity (0..1; metric mode: metres)
        "mode": "relative" (default) | "metric" (moge-2: the baseline in mm)
        "baseline_mm": 63, "z_near": 0.6 (metric mode)
        "output_format": "jpeg" (default, "quality" 92) | "png"
    -> {"created", "data": [{"b64_json", "filename_tag" ("_LR" ...),
        "info": the settings, the convergence, the shift range}]}

    POST /v1/videos/stereo   multipart/form-data: file "video" (any format of
                             ffmpeg) and the fields "layout" (sbs, sbs_half,
                             ou, ou_half), "fill", "divergence", "convergence",
                             "codec" (hevc, default: NVENC on the GPU | h264),
                             "quality" (the CQ/CRF, 21), "container" (mp4 |
                             mkv), "max_frames": a job (VDA-Small's depth)
    GET    /v1/videos, /v1/videos/{id}, /v1/videos/{id}/content, DELETE
           /v1/videos/{id}: the stereo jobs beside Wan's and the upscales' (ids
           "stereo_...", in VIDEO_DIR/stereo); progress: the frames done of the
           frames ffprobe counts

A job's MP4 carries Spherical Video V2's st3d box (left-right or top-bottom),
an MKV Matroska's StereoMode; the source's audio is copied (AAC when MP4
cannot hold it). One job runs at a time; each frame takes the server's lock
in turn, so images go between the frames of a long video.
"""
from __future__ import annotations

import base64
import collections
import io
import json
import os
import threading
import time

import numpy as np


class _Cancelled(Exception):
    pass


def decode_image(data):
    """an uint8 RGB image (H, W, 3) of encoded bytes (EXIF orientation applied)"""
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im, np.uint8)


def encode_image(img, fmt="jpeg", quality=92):
    from PIL import Image
    buf = io.BytesIO()
    if img.dtype == np.uint16:
        Image.fromarray(img).save(buf, format="PNG")
    elif fmt == "png":
        Image.fromarray(img).save(buf, format="PNG")
    else:
        Image.fromarray(img).save(buf, format="JPEG", quality=int(quality), subsampling=0)
    return buf.getvalue()


def _num(v, name, lo, hi, default):
    if v is None or v == "":
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number" % name)
    if not lo <= x <= hi:
        raise ValueError("%s must be %g to %g" % (name, lo, hi))
    return x


def _conv(v, metric):
    if v in (None, "", "auto"):
        return "auto"
    return _num(v, "convergence", 0.01 if metric else 0.0, 1000.0 if metric else 1.0, None)


class StereoServer:
    """The stereo converter of the server and its video jobs."""

    def __init__(self, moge, da3mono, vda, gpu, cpu_threads, work_dir, server):
        from reclank.models.depth.pipeline import Stereo
        self.st = Stereo(moge=moge, da3mono=da3mono, vda=vda, gpu=gpu, cpu_threads=cpu_threads)
        self.model_ids = self.st.available()
        self.model_id = "stereo"
        self.dir, self.server = os.path.join(work_dir, "stereo"), server     # (apart from Wan's jobs)
        os.makedirs(self.dir, exist_ok=True)
        self.jobs = collections.OrderedDict()
        self.queue = collections.deque()
        self.cancel = set()
        self.cv = threading.Condition()
        for name in sorted(os.listdir(self.dir)):
            if not (name.startswith("stereo_") and name.endswith(".json")):
                continue
            try:
                job = json.load(open(os.path.join(self.dir, name)))
            except (OSError, ValueError):
                continue
            if job.get("status") in ("queued", "in_progress"):
                job.update(status="failed", error={"code": "server_stopped", "message": "the server stopped"})
                self._save(job)
            self.jobs[job["id"]] = job
        if vda:
            threading.Thread(target=self._worker, name="stereo-videos", daemon=True).start()

    def runners(self):
        return self.st.runners()

    # ---- images ----

    def image(self, fields):
        """the response's data of a request's fields ({name: value}, the image
        as bytes)"""
        data = fields.get("image")
        if isinstance(data, str):
            if data.startswith("data:"):
                data = data.split(",", 1)[-1]
            data = base64.b64decode(data)
        if not data:
            raise ValueError("image is required (base64, or a multipart file)")
        img = decode_image(data)
        model = fields.get("model") or ("moge-2" if "moge-2" in self.model_ids else self.model_ids[0])
        if model not in self.model_ids or model == "vda-small":
            raise ValueError("model must be one of %s" % ", ".join(m for m in self.model_ids if m != "vda-small"))
        layout = fields.get("layout") or "sbs"
        mode = fields.get("mode") or "relative"
        fmt = (fields.get("output_format") or "jpeg").lower()
        if fmt not in ("jpeg", "jpg", "png"):
            raise ValueError("output_format must be jpeg or png")
        quality = _num(fields.get("quality"), "quality", 50, 100, 92)
        if (fields.get("response_format") or "b64_json") != "b64_json":
            raise ValueError("response_format must be \"b64_json\"")
        if layout == "depth":
            with self.server.lock:
                d = self.st.depth(img, model)
            out = np.round(d["D"] * 65535).astype(np.uint16)
            item = {"b64_json": base64.b64encode(encode_image(out, "png")).decode("ascii"), "filename_tag": "_depth",
                    "info": {"model": model, "size": [int(out.shape[1]), int(out.shape[0])]}}
            return [item]
        with self.server.lock:
            r = self.st.image(img, model=model, layout=layout, fill=fields.get("fill") or "edge",
                              divergence=_num(fields.get("divergence"), "divergence", 0, 10, 2.0),
                              convergence=_conv(fields.get("convergence"), mode == "metric"), mode=mode,
                              baseline_mm=_num(fields.get("baseline_mm"), "baseline_mm", 1, 1000, 63.0),
                              z_near=_num(fields.get("z_near"), "z_near", 0.05, 100, 0.6))
        enc = encode_image(r["image"], "png" if fmt == "png" else "jpeg", quality)
        return [{"b64_json": base64.b64encode(enc).decode("ascii"), "filename_tag": r["tag"], "info": r["info"]}]

    # ---- video jobs ----

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
        """a new job of a multipart request's fields ({name: [(bytes, filename)]})"""
        from reclank.models.depth.stereo import FILLS, LAYOUTS
        if "vda-small" not in self.model_ids:
            raise ValueError("no video depth model loaded (--stereo-vda)")
        f = lambda n, d=None: (form[n][0][0].decode("utf-8").strip() if n in form else d)  # noqa: E731
        if "video" not in form or not form["video"][0][0]:
            raise ValueError("video (a file) is required")
        data, fname = form["video"][0]
        layout = f("layout", "sbs")
        if layout not in LAYOUTS:
            raise ValueError("layout must be one of %s" % ", ".join(LAYOUTS))
        fill = f("fill", "edge")
        if fill not in FILLS:
            raise ValueError("fill must be one of %s" % ", ".join(FILLS))
        codec = f("codec", "hevc")
        if codec not in ("hevc", "h264"):
            raise ValueError("codec must be hevc or h264")
        container = f("container", "mp4")
        if container not in ("mp4", "mkv"):
            raise ValueError("container must be mp4 or mkv")
        maxf = f("max_frames")
        job = {"id": "stereo_" + os.urandom(12).hex(), "object": "video", "model": "vda-small", "kind": "stereo",
               "status": "queued", "progress": 0, "frames_done": 0, "frames": None, "created_at": int(time.time()),
               "completed_at": None, "layout": layout, "fill": fill,
               "divergence": _num(f("divergence"), "divergence", 0, 10, 2.0), "convergence": _conv(f("convergence"), False),
               "codec": codec, "quality": int(_num(f("quality"), "quality", 0, 51, 21)), "container": container,
               "max_frames": int(maxf) if maxf else None, "source": os.path.basename(fname or "video"),
               "filename_tag": {"sbs": "_LR", "sbs_half": "_3dh", "ou": "_TB", "ou_half": "_3dv"}[layout],
               "error": None}
        ext = os.path.splitext(fname or "")[1][:8] or ".bin"
        job["_src"] = self._file(job["id"], ".src" + ext)
        with open(job["_src"], "wb") as fh:
            fh.write(data)
        with self.cv:
            self.jobs[job["id"]] = job
            self._save(job)
            self.queue.append(job["id"])
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

    def content(self, job):
        return self._file(job["id"], "." + job.get("container", "mp4"))

    def _remove(self, vid):
        job = self.jobs.pop(vid, None)
        names = [vid + e for e in (".json", ".mp4", ".mkv")]
        if job is not None and job.get("_src"):
            names.append(os.path.basename(job["_src"]))
        for n in names:
            try:
                os.remove(os.path.join(self.dir, n))
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
            out = self.content(job)
            part = out + ".part" + os.path.splitext(out)[1]

            def progress(i, n):
                if vid in self.cancel:
                    raise _Cancelled()
                job["frames_done"], job["frames"] = i, n
                job["progress"] = int(99 * i / n) if n else 0

            try:
                r = self.st.video(job["_src"], part, layout=job["layout"], fill=job["fill"],
                                  divergence=job["divergence"], convergence=job["convergence"], codec=job["codec"],
                                  quality=job["quality"], max_frames=job["max_frames"], progress=progress,
                                  gate=self.server.lock)
                os.replace(part, out)
                job.update(status="completed", progress=100, completed_at=int(time.time()), frames=r["frames"],
                           frames_done=r["frames"], fps=r["fps"], st3d=r["st3d"], size="%dx%d" % (r["width"], r["height"]))
                if not getattr(self.server, "quiet", False):
                    print("stereo video %s: %d frames in %.1f s (%.1f fps)" % (vid, r["frames"], r["seconds"], r["fps"]),
                          flush=True)
            except _Cancelled:
                pass
            except Exception as exc:  # noqa: BLE001 - a failed job reports its error
                job.update(status="failed", error={"code": type(exc).__name__, "message": str(exc)[:500]})
                print("stereo video %s failed: %s" % (vid, exc), flush=True)
            for p in (part, job.get("_src")):
                try:
                    os.remove(p)
                except (FileNotFoundError, TypeError):
                    pass
            with self.cv:
                if vid in self.cancel:
                    self.cancel.discard(vid)
                    self._remove(vid)
                else:
                    self._save(job)
