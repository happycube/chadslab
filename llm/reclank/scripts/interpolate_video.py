"""Video frame interpolation from the command line (Practical-RIFE 4.25, reclank/models/vfi; the
server's POST /v1/videos/interpolate does the same).

    interpolate_video.py VIDEO --rife DIR [--out FILE.mp4] [--factor 2 | --fps 60000/1001]
        [--scene-cuts detect|off] [--scale auto|1|0.5] [--slowmo] [--codec h264|hevc] [--crf 16]
        [--x264] [--max-frames N] [--cpu] [--cpu-threads N]

The output goes to NAME_x2.mp4 (or _60fps, _slowmo4) beside the input unless --out."""
import argparse
import json
import os
import sys
import time
from fractions import Fraction

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.models.vfi.rife import RIFE  # noqa: E402
from reclank.models.vfi.video import interpolate_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--rife", required=True, help="the folder of flownet.safetensors")
    ap.add_argument("--out")
    ap.add_argument("--factor", type=int, default=None)
    ap.add_argument("--fps", default=None)
    ap.add_argument("--scene-cuts", default="detect", choices=["detect", "off"])
    ap.add_argument("--scale", default="auto", choices=["auto", "1", "0.5"])
    ap.add_argument("--slowmo", action="store_true")
    ap.add_argument("--codec", default="h264", choices=["h264", "hevc"])
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--x264", action="store_true", help="libx264 / libx265 (their crf) on the GPU too, not NVENC (its -cq)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cpu-threads", type=int, default=None)
    a = ap.parse_args()
    if a.factor is None and a.fps is None:
        a.factor = 2
    rife = RIFE(a.rife, gpu=not a.cpu, cpu_threads=a.cpu_threads)
    rife.scale = None if a.scale == "auto" else float(a.scale)
    stem = os.path.splitext(a.video)[0]
    tag = ("_slowmo%d" % a.factor) if a.slowmo else ("_%sfps" % a.fps.replace("/", "_") if a.fps else "_x%d" % a.factor)
    dst = a.out or stem + tag + ".mp4"
    t = time.time()

    def progress(i, n):
        if i % 200 == 0 or i == n:
            print("  %d / %d frames, %.1f a second" % (i, n, i / (time.time() - t)), flush=True)

    r = interpolate_video(rife, a.video, dst, factor=a.factor, fps=Fraction(a.fps) if a.fps else None,
                          scene_cuts=a.scene_cuts == "detect", slowmo=a.slowmo, codec=a.codec, crf=a.crf,
                          gpu_encoder=False if a.x264 else None, max_frames=a.max_frames, progress=progress)
    print("%s -> %s: %s" % (a.video, dst, json.dumps(r)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
