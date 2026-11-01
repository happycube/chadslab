"""The frame interpolation route of the generation server (reclank/models/vfi, docs/VFI_PLAN.md;
--rife DIR: Practical-RIFE 4.25, "rife-4.25"):

    POST /v1/videos/interpolate  multipart/form-data: "video" (a file, any format ffmpeg reads) or
                                 "video_id" (a completed job of /v1/videos: a Wan clip, an
                                 upscale, a stereo video, an interpolation), and the fields
        "factor": 2 (default), 3, 4, ... 16: every input frame kept, factor - 1 between each pair
        "fps": a target rate in place of factor ("60", "59.94", "60000/1001"), above the input's:
               any ratio (each output time a pair and t); a variable-rate input by its timestamps
        "scene_cuts": "detect" (default: the frames between a cut's pair repeat its left frame) |
                      "off"
        "scale": "auto" (default: 0.5 above 2560 x 1440 of area, the authors' advice for 4K) | 1 |
                 0.5 (RIFE's flow at half the resolution)
        "slowmo": "false" (default) | "true": the output at the input's rate (factor x longer), no
                  audio
        "codec": "h264" (default) | "hevc"; "crf": 16 (default; 0 to 51); "max_frames"
    GET    /v1/videos, /v1/videos/{id}, /v1/videos/{id}/content, DELETE /v1/videos/{id}: the jobs
           beside the others (ids "interp_...", in VIDEO_DIR/interpolate); progress: the output frames
           done of the total

The output: MP4 at the exact new rate (a fraction: 24000/1001 x 2 is 48000/1001), the input frames'
bytes unchanged at their times, the source's audio copied (AAC when MP4 cannot hold it) and its
colour tags kept. One job runs at a time; each frame takes the server's lock in turn, so the other
requests go between the frames of a long video.
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import threading
import time
from fractions import Fraction

from reclank.models.vfi.rife import MODEL_ID

MAX_FACTOR = 16


class _Cancelled(Exception):
    pass


def _field(form, name, default=None):
    if name not in form:
        return default
    v = form[name][0][0].decode("utf-8", "replace").strip()
    return v if v != "" else default


def options(form):
    """the checked options of a request's fields (ValueError: a bad request)"""
    fps = _field(form, "fps")
    factor = _field(form, "factor")
    if fps is not None and factor is not None:
        raise ValueError("give factor or fps, not both")
    out = {"factor": None, "fps": None}
    if fps is not None:
        try:
            f = Fraction(fps) if "/" in fps else Fraction(fps).limit_denominator(1001)
        except (ValueError, ZeroDivisionError):
            raise ValueError("fps must be a number or a fraction (60000/1001)") from None
        if not 1 <= f <= 1000:
            raise ValueError("fps must be 1 to 1000")
        if abs(float(f) - 59.94) < 0.005 or abs(float(f) - 29.97) < 0.005 or abs(float(f) - 23.976) < 0.0005:
            f = Fraction(round(float(f) * 1001 / 1000)) * Fraction(1000, 1001)   # (the NTSC rates exactly)
        out["fps"] = str(f)
    else:
        try:
            k = int(factor or 2)
        except ValueError:
            raise ValueError("factor must be an integer") from None
        if not 2 <= k <= MAX_FACTOR:
            raise ValueError("factor must be 2 to %d" % MAX_FACTOR)
        out["factor"] = k
    cuts = _field(form, "scene_cuts", "detect")
    if cuts not in ("detect", "off"):
        raise ValueError("scene_cuts must be \"detect\" or \"off\"")
    out["scene_cuts"] = cuts == "detect"
    scale = _field(form, "scale", "auto")
    if scale not in ("auto", "1", "1.0", "0.5"):
        raise ValueError("scale must be \"auto\", 1 or 0.5")
    out["scale"] = None if scale == "auto" else float(scale)
    slow = _field(form, "slowmo", "false").lower()
    if slow not in ("true", "false", "1", "0"):
        raise ValueError("slowmo must be true or false")
    out["slowmo"] = slow in ("true", "1")
    if out["slowmo"] and out["fps"] is not None:
        raise ValueError("slowmo takes a factor, not fps")
    codec = _field(form, "codec", "h264")
    if codec not in ("h264", "hevc"):
        raise ValueError("codec must be h264 or hevc")
    out["codec"] = codec
    try:
        out["crf"] = int(_field(form, "crf", "16"))
        mf = _field(form, "max_frames")
        out["max_frames"] = int(mf) if mf is not None else None
    except ValueError:
        raise ValueError("crf and max_frames must be integers") from None
    if not 0 <= out["crf"] <= 51:
        raise ValueError("crf must be 0 to 51")
    if out["max_frames"] is not None and out["max_frames"] < 2:
        raise ValueError("max_frames must be 2 or more")
    return out


class Interpolator:
    """RIFE 4.25 of the server and its video jobs (their queue, worker and files)."""

    def __init__(self, path, gpu, cpu_threads, work_dir, server):
        from reclank.models.vfi.rife import RIFE
        self.rife = RIFE(path, gpu=gpu, cpu_threads=cpu_threads)
        self.model_id = MODEL_ID
        self.dir, self.server = os.path.join(work_dir, "interpolate"), server
        os.makedirs(self.dir, exist_ok=True)
        self.jobs = collections.OrderedDict()
        self.queue = collections.deque()
        self.cancel = set()
        self.cv = threading.Condition()
        for name in sorted(os.listdir(self.dir)):
            if not (name.startswith("interp_") and name.endswith(".json")):
                continue
            try:
                job = json.load(open(os.path.join(self.dir, name)))
            except (OSError, ValueError):
                continue
            if job.get("status") in ("queued", "in_progress"):
                job.update(status="failed", error={"code": "server_stopped", "message": "the server stopped"})
                self._save(job)
            self.jobs[job["id"]] = job
        threading.Thread(target=self._worker, name="interpolate", daemon=True).start()

    def runners(self):
        return self.rife.runners()

    def describe(self):
        return {"id": self.model_id, "object": "model", "owned_by": "reclank", "kind": "interpolate",
                "max_factor": MAX_FACTOR}

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

    def content(self, job):
        return self._file(job["id"], ".mp4")

    def submit(self, form, source=None):
        """A new job of a multipart form ({name: [(bytes, filename)]}). source(video_id): the file
        of a completed job of the server's /v1/videos (ValueError if none)."""
        model = _field(form, "model", self.model_id)
        if model != self.model_id:
            raise ValueError("model must be %s" % self.model_id)
        opts = options(form)
        files = form.get("video", [])
        vid_in = _field(form, "video_id")
        if bool(files and files[0][0]) == bool(vid_in):
            raise ValueError("give video (a file) or video_id (a completed job), one of them")
        vid = "interp_" + os.urandom(12).hex()
        if vid_in:
            if source is None:
                raise ValueError("video_id: no other video jobs on this server")
            path = source(vid_in)
            src = self._file(vid, ".src" + os.path.splitext(path)[1])
            try:
                os.link(path, src)          # (the source job may go; the link keeps the file)
            except OSError:
                shutil.copyfile(path, src)
            name = vid_in
        else:
            data, fname = files[0]
            src = self._file(vid, ".src" + (os.path.splitext(fname or "")[1][:8] or ".bin"))
            with open(src, "wb") as fh:
                fh.write(data)
            name = os.path.basename(fname or "video")
        try:
            from reclank.models.vfi.video import probe
            info = probe(src)
        except Exception:
            os.remove(src)
            raise
        job = {"id": vid, "object": "video", "model": self.model_id, "kind": "interpolate", "status": "queued",
               "progress": 0, "frames_done": 0, "frames": None, "created_at": int(time.time()),
               "completed_at": None, "source": name, "input_size": "%dx%d" % (info["width"], info["height"]),
               "error": None, "_src": src}
        job.update({k: v for k, v in opts.items()})
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
        job = self.jobs.pop(vid, None)
        names = [vid + e for e in (".json", ".mp4", ".part.mp4")]
        if job is not None and job.get("_src"):
            names.append(os.path.basename(job["_src"]))
        for n in names:
            try:
                os.remove(os.path.join(self.dir, n))
            except FileNotFoundError:
                pass

    def _worker(self):
        from reclank.models.vfi.video import interpolate_video
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                vid = self.queue.popleft()
                job = self.jobs[vid]
                job["status"] = "in_progress"
                self._save(job)
            part = self._file(vid, ".part.mp4")

            def progress(i, n):
                if vid in self.cancel:
                    raise _Cancelled()
                job["frames_done"], job["frames"] = i, n
                job["progress"] = int(99 * i / n) if n else 0

            try:
                self.rife.scale = job["scale"]
                r = interpolate_video(self.rife, job["_src"], part, factor=job["factor"],
                                      fps=Fraction(job["fps"]) if job["fps"] else None, scene_cuts=job["scene_cuts"],
                                      slowmo=job["slowmo"], codec=job["codec"], crf=job["crf"],
                                      max_frames=job["max_frames"], progress=progress, gate=self.server.lock)
                os.replace(part, self.content(job))
                job.update(status="completed", progress=100, completed_at=int(time.time()), frames=r["frames"],
                           frames_done=r["frames"], frames_in=r["frames_in"], interpolated=r["interpolated"],
                           cuts=r["cuts"], fps_in=r["fps_in"], fps_out=r["fps_out"], speed=r["speed"],
                           audio=r["audio"], size="%dx%d" % (r["width"], r["height"]))
                if not getattr(self.server, "quiet", False):
                    print("interpolate %s: %d frames to %d (%s to %s fps, %d cuts) in %.1f s (%.1f frames a second)"
                          % (vid, r["frames_in"], r["frames"], r["fps_in"], r["fps_out"], len(r["cuts"]),
                             r["seconds"], r["speed"]), flush=True)
            except _Cancelled:
                pass
            except Exception as exc:  # noqa: BLE001 - a failed job reports its error
                job.update(status="failed", error={"code": type(exc).__name__, "message": str(exc)[:500]})
                print("interpolate %s failed: %s" % (vid, exc), flush=True)
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
