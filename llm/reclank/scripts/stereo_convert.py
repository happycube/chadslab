"""2D to stereo 3D from the command line (reclank/models/depth; the server's POST
/v1/images/stereo and /v1/videos/stereo do the same).

    stereo_convert.py IMAGE... --moge DIR | --da3 DIR [--model moge-2|da3mono] [--out DIR]
        [--layout sbs|sbs_half|ou|ou_half|anaglyph] [--fill edge|stretch] [--divergence 2]
        [--convergence auto|D] [--metric --baseline-mm 63] [--png] [--cpu]
    stereo_convert.py VIDEO --vda DIR [--out FILE.mp4|.mkv] [--layout ...] [--codec hevc|h264]
        [--quality 21] [--max-frames N] [--cpu]

An image goes to OUT/NAME_LR.jpg (the layout's tag: _LR, _3dh, _TB, _3dv); a video to
NAME_LR.mp4 beside it (st3d box) unless --out."""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.models.depth import stereo as S  # noqa: E402
from reclank.models.depth.pipeline import Stereo  # noqa: E402

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--moge")
    ap.add_argument("--da3")
    ap.add_argument("--vda")
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--layout", default="sbs")
    ap.add_argument("--fill", default="edge")
    ap.add_argument("--divergence", type=float, default=2.0)
    ap.add_argument("--convergence", default="auto")
    ap.add_argument("--metric", action="store_true")
    ap.add_argument("--baseline-mm", type=float, default=63.0)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--codec", default="hevc")
    ap.add_argument("--quality", type=int, default=21)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cpu-threads", type=int, default=None)
    a = ap.parse_args()
    st = Stereo(moge=a.moge, da3mono=a.da3, vda=a.vda, gpu=not a.cpu, cpu_threads=a.cpu_threads)
    conv = a.convergence if a.convergence == "auto" else float(a.convergence)
    for path in a.inputs:
        stem, ext = os.path.splitext(path)
        if ext.lower() in VIDEO_EXT:
            dst = a.out or stem + S.TAGS[a.layout] + ".mp4"
            t = time.time()

            def progress(i, n):
                if i % 100 == 0:
                    print("  %d / %s frames, %.1f fps" % (i, n or "?", i / (time.time() - t)), flush=True)
            r = st.video(path, dst, layout=a.layout, fill=a.fill, divergence=a.divergence, convergence=conv,
                         codec=a.codec, quality=a.quality, max_frames=a.max_frames, progress=progress)
            print("%s -> %s: %s" % (path, dst, r))
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        model = a.model or ("moge-2" if a.moge else "da3mono")
        t = time.time()
        r = st.image(img, model=model, layout=a.layout, fill=a.fill, divergence=a.divergence, convergence=conv,
                     mode="metric" if a.metric else "relative", baseline_mm=a.baseline_mm)
        out_dir = a.out or os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, os.path.basename(stem) + r["tag"] + (".png" if a.png else ".jpg"))
        Image.fromarray(r["image"]).save(dst, **({} if a.png else {"quality": 92, "subsampling": 0}))
        print("%s -> %s in %.2f s: %s" % (path, dst, time.time() - t, r["info"]))


if __name__ == "__main__":
    main()
