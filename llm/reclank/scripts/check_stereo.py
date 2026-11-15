"""Check reclank's depth models and stereo warp against scripts/stereo_reference.py.

    check_stereo.py da3mono MODEL_DIR ref.npz [IMAGE] [--gpu]
    check_stereo.py moge MODEL_DIR ref.npz IMAGE [--gpu]
    check_stereo.py vda MODEL_DIR ref.npz FRAMES_DIR [--gpu] [--key depth]
    check_stereo.py warp IMAGE ref.npz [--gpu] [--width W]

da3mono: the network on the reference's processed image: the depth logits and the sky branch
(1 - cos, the largest difference relative to the largest value), the depth's relative error;
IMAGE: reclank's preprocessing (OpenCV's resizes in float arithmetic) against the reference's
image (levels of 255) and the depth end to end (the sky set to the far end; the share of pixels
off by more than 1%). moge: the heads (points, mask) at 16x the grid, the metric scale, the focal
and shift, the depth in metres on the pixels both masks keep (relative error). vda: the aligned
disparity of each frame of the reference's windows (reclank's at the network's size, bilinear to
the frame's for the comparison). warp: both eyes of the image by a DA3MONO reference's disparity
(2% divergence, both fills, every layout), the program's bits against models/depth/ref.py's
NumPy, with the time of a run."""
import argparse
import glob
import os
import time

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("part", choices=["da3mono", "moge", "vda", "warp"])
ap.add_argument("a")
ap.add_argument("ref")
ap.add_argument("inp", nargs="?")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--width", type=int, default=0)
ap.add_argument("--key", default="depth", help="vda: the reference's array (the authors' run: video96_depth)")
a = ap.parse_args()


def cmp(x, y):
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    c = 1 - x @ y / np.sqrt((x @ x) * (y @ y))
    return "1-cos %.2e, max|d|/max|ref| %.2e" % (c, np.abs(x - y).max() / np.abs(y).max())


def read(p):
    return np.asarray(Image.open(p).convert("RGB"))


z = np.load(a.ref)
dev = "GPU" if a.gpu else "CPU"
if a.part == "da3mono":
    from reclank.models.depth.da3 import DA3Mono, preprocess
    m = DA3Mono(a.a, gpu=a.gpu)
    m.logits(z["image"])
    t = time.time()
    lg, sky = m.logits(z["image"])
    print("%s: %s in %.3f s" % (dev, "x".join(map(str, z["image"].shape[:2])), time.time() - t))
    print("  logits %s; depth rel. err. max %.2e" % (cmp(lg, z["logits"]), (np.abs(np.exp(lg) - z["depth"]) / z["depth"]).max()))
    print("  sky    %s" % cmp(sky, z["sky"]))
    if a.inp:
        img = read(a.inp)
        p = preprocess(img)
        d = np.abs(p.astype(int) - z["image"].astype(int))
        print("  preprocessing: levels off max %d, %.3f%% of values" % (d.max(), 100 * (d > 0).mean()))
        out = m.depth(img)
        ref = z["depth"].copy()
        ns = z["sky"] < 0.3
        ref[~ns] = np.quantile(ref[ns], 0.99)
        rel = np.abs(out["depth"] - ref) / ref
        print("  end to end depth: rel. err. median %.2e, %.3f%% of pixels over 1%% (the sky's edge: %d pixels "
              "classified otherwise)" % (np.median(rel), 100 * (rel > 0.01).mean(), int((out["valid"] != ns).sum())))
elif a.part == "moge":
    from reclank.models.depth.moge import MoGe2
    img = read(a.inp)
    m = MoGe2(a.a, gpu=a.gpu)
    m.raw(img)
    t = time.time()
    r = m.raw(img)
    print("%s: grid %s in %.3f s" % (dev, r["grid"], time.time() - t))
    print("  points %s" % cmp(r["points_head"], z["points_head"].transpose(1, 2, 0)))
    print("  mask   %s" % cmp(r["mask_head"], z["mask_head"].transpose(1, 2, 0)))
    print("  metric scale %.6f (ref %.6f)" % (r["metric_scale"], float(z["metric_scale"])))
    g = m.geometry(img, full=True)
    v = g["valid"] & z["valid"]
    rel = np.abs(g["depth"][v] - z["depth"][v]) / z["depth"][v]
    print("  fx %.6f (ref %.6f), shift %.6f (ref %.6f), masks agree on %.4f%% of pixels" % (
        g["fx"], float(z["fx"]), g["shift"], float(z["shift"]), 100 * (g["valid"] == z["valid"]).mean()))
    print("  depth (m) rel. err. median %.2e, max %.2e" % (np.median(rel), rel.max()))
elif a.part == "vda":
    from reclank.models.depth.ref import resize
    from reclank.models.depth.vda import VDASmall
    ref = z[a.key]
    files = sorted(glob.glob(os.path.join(a.inp, "*.png")))[:len(ref)]
    m = VDASmall(a.a, gpu=a.gpu)
    stamps = []
    t = time.time()
    out = list(m.video((read(f) for f in files), on_window=lambda k: stamps.append(time.time() - t)))
    dt = time.time() - t
    print("%s: %d frames in %.1f s (%.1f fps; windows done at %s s)" % (
        dev, len(out), dt, len(out) / dt, " ".join("%.1f" % s for s in stamps)))
    H, W = ref.shape[1:]
    worst = (0, 0)
    for i, d in out:
        up = resize(d[None, :, :, None], H, W, 1)[0, :, :, 0]
        rel = np.abs(up - ref[i]).max() / np.abs(ref[i]).max()
        worst = max(worst, (rel, i))
        if i % 16 == 0 or i == len(out) - 1:
            print("  frame %3d %s" % (i, cmp(up, ref[i])))
    print("  worst frame %d: max|d|/max|ref| %.2e" % (worst[1], worst[0]))
else:
    from reclank.models.depth import ref as R
    from reclank.models.depth import stereo as S
    img = read(a.a)
    if a.width:
        img = np.asarray(Image.fromarray(img).resize((a.width, round(img.shape[0] * a.width / img.shape[1])),
                                                     Image.BICUBIC))
    H, W = img.shape[:2]
    d = z["depth"]
    valid = z["sky"] < 0.3
    D, _, _ = S.normalized(np.where(valid, 1 / d, 0), valid)
    shift = S.relative_shift(D, S.convergence_of(D, valid), W, 2.0)
    full = R.resize(shift[None, :, :, None], H, W, 0)[0, :, :, 0]
    w = S.Warper(gpu=a.gpu)
    for fill in ("edge", "stretch"):
        mode = S.FILLS[fill]
        L = R.dibr(img, full, 1, 1.0, mode)
        Rr = R.dibr(img, full, -1, 1.0, mode)
        eyes = {"sbs": np.concatenate([L[0], Rr[0]], 1), "ou": np.concatenate([L[0], Rr[0]], 0)}
        for layout in S.LAYOUTS:
            if layout == "sbs_half" and W % 2 or layout == "ou_half" and H % 2:
                print("%s %s %-8s: skipped (an odd side)" % (dev, fill, layout))
                continue
            base = eyes["sbs" if layout.startswith("sbs") else "ou"]
            want = base if not layout.endswith("half") else R.resize(base[None], H, W, 2)[0]
            w.warp(img, shift, layout, fill)
            t = time.time()
            got = w.warp(img, shift, layout, fill)
            dt = time.time() - t
            print("%s %s %-8s %dx%d: %.4f s, bytes off %d; holes %.2f%% / %.2f%%" % (
                dev, fill, layout, got.shape[1], got.shape[0], dt, int((got != R.pack_u8(want)).sum()),
                100 * L[1].mean(), 100 * Rr[1].mean()))
