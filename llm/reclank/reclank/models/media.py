"""ffmpeg beside reclank's threads, for the media pipelines (models/vfi/video.py, models/depth/pipeline.py,
models/wan/pipeline.py, server/upscale.py): where the tools and the pipelines' threads run, and the
colour of an encode.

reclank sets OMP_PROC_BIND (runtime/threads.py), so OpenMP binds the main thread to one CPU when it
starts; every thread made after it inherits that CPU, and so does every child it starts, and ffmpeg
(which links libgomp) binds its own main thread again. A video job's decoder and writer threads and
both ffmpeg processes all ran on one CPU (taskset -p of each during a run). popen() and run() start a
tool on the CPUs of the process's start, without OMP_PROC_BIND and OMP_PLACES; widen() puts the
calling thread (a pipeline's own) on them.

The colour (yuv_colors): ffmpeg decodes a source to RGB by its tags; an encode back to yuv420p is
BT.601, limited range, unless the scale filter is told the matrix and range: yuv_colors gives it the
source's, and tags the output so (a source without a matrix tag BT.601 limited, as ffmpeg decoded it,
tagged smpte170m / tv).
"""
from __future__ import annotations

import os
import subprocess


def start_cpus():
    """the CPUs of the process's start (its taskset; RECLANK_START_CPUS, which runtime/threads.py records
    before any thread is bound), or None"""
    try:
        v = os.environ.get("RECLANK_START_CPUS")
        return {int(c) for c in v.split(",")} if v else set(os.sched_getaffinity(0))
    except (AttributeError, OSError, ValueError):
        return None


def widen():
    """the calling thread on the start CPUs (a pipeline's decoder and writer threads)"""
    cpus = start_cpus()
    try:
        if cpus:
            os.sched_setaffinity(0, cpus)
    except OSError:
        pass


def tool_env():
    """a tool's environment: reclank's without OMP_PROC_BIND and OMP_PLACES (ffmpeg links libgomp, which
    binds its main thread to the first CPU of its places)"""
    return {k: v for k, v in os.environ.items() if k not in ("OMP_PROC_BIND", "OMP_PLACES")}


def _start(fn, cmd, kw):
    """fn(cmd, **kw) (a child takes the affinity of the thread that starts it) on the start CPUs, with
    tool_env() unless kw gives an env; the calling thread's affinity back after"""
    kw.setdefault("env", tool_env())
    cpus, old = start_cpus(), None
    try:
        old = os.sched_getaffinity(0)
        if cpus:
            os.sched_setaffinity(0, cpus)
    except (AttributeError, OSError):
        pass
    try:
        return fn(cmd, **kw)
    finally:
        if old is not None:
            try:
                os.sched_setaffinity(0, old)
            except OSError:
                pass


def popen(cmd, **kw):
    """subprocess.Popen of a tool on the start CPUs (_start)"""
    return _start(subprocess.Popen, cmd, kw)


def run(cmd, **kw):
    """subprocess.run of a tool on the start CPUs (_start)"""
    return _start(subprocess.run, cmd, kw)


def pipe_size(f, n=1 << 20):
    """a pipe's buffer to n bytes (at most /proc/sys/fs/pipe-max-size, 1 MiB by default; 64 KiB else):
    a reader thread reads a frame in fewer calls, each taking the GIL back from the loop's thread"""
    try:
        import fcntl
        fcntl.fcntl(f.fileno(), fcntl.F_SETPIPE_SZ, n)
    except (ImportError, AttributeError, OSError):
        pass


# a matrix tag (ffprobe's color_space) to the scale filter's out_color_matrix
YUV_MATRIX = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt2020nc": "bt2020", "bt2020c": "bt2020",
              "fcc": "fcc", "smpte240m": "smpte240m"}
COLOR_FLAGS = {"color_space": "-colorspace", "color_transfer": "-color_trc", "color_primaries": "-color_primaries",
               "color_range": "-color_range"}
COLOR_KEYS = tuple(COLOR_FLAGS)


def yuv_colors(colors):
    """(the scale filter's options, the output's colour tags) of RGB frames to yuv420p by a source's tags
    (ffmpeg decoded the source to RGB by them): its matrix and range; a source without a matrix tag (or
    one the scale filter has no matrix for) BT.601 limited, as ffmpeg decodes it, and tagged so
    (smpte170m, tv). Without the filter's options ffmpeg converts by BT.601 limited whatever the tags:
    a BT.709 source came out BT.601 tagged bt709, a full-range one limited tagged pc."""
    colors = dict(colors or {})
    m = YUV_MATRIX.get(colors.get("color_space"))
    if m is None:
        m = "bt601"
        colors["color_space"] = "smpte170m"
    rng = "pc" if colors.get("color_range") in ("pc", "jpeg") else "tv"
    colors["color_range"] = rng
    return "scale=out_color_matrix=%s:out_range=%s" % (m, rng), colors


def yuv_args(colors):
    """ffmpeg's output options of yuv_colors: the filter and the tags"""
    vf, tags = yuv_colors(colors)
    out = ["-vf", vf]
    for k, v in tags.items():
        out += [COLOR_FLAGS[k], v]
    return out


def colors_of(path):
    """the colour tags of a video file's first video stream (stream_colors; {} when ffprobe fails)"""
    import json
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=" + ",".join(COLOR_KEYS),
             "-of", "json", path], capture_output=True, text=True)
    try:
        return stream_colors((json.loads(p.stdout).get("streams") or [{}])[0])
    except ValueError:
        return {}


def stream_colors(stream):
    """the colour tags of an ffprobe video stream (a dict of its entries; "unknown" left out)"""
    return {k: stream[k] for k in COLOR_KEYS if stream.get(k) and stream[k] != "unknown"}
