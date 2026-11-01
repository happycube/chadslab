"""Checks of Practical-RIFE 4.25 in reclank (models/vfi) against scripts/rife_reference.py on real
frames, the scene-cut rule on real footage, and timings.

    check_rife.py MODEL_DIR taps FRAMES_DIR REF.npz [--gpu]
        REF: rife_reference.py --dump of a pair: the heads, each block's tmp (13 channels at 1/s)
        and flow, the last mask and the output (1 - cos, the largest difference; the output
        rounded to bytes: the levels of 255 apart)
    check_rife.py MODEL_DIR outputs FRAMES_DIR REF.npz [--gpu] [--gt]
        every out_I_J_T of REF against reclank's (float, bytes); --gt: both against the frame at
        I + (J - I) T (held out: the frames of a higher rate), PSNR and the share over 8 levels
    check_rife.py MODEL_DIR|- cuts VIDEO START SECONDS [VIDEO START SECONDS ...] [--gpu]
        the cut rule (models/vfi/video.CutDetector: with MODEL_DIR RIFE's residual of the probe
        frames, "-" the thumbnails alone) on every pair of a 2x job (every other frame) and a 4x
        job (every 4th) of 59.94 fps footage, the cuts those of the source (a spike of the mean
        frame difference: over 12 levels and 4x both neighbours'; animation spikes too)
    check_rife.py MODEL_DIR time WxH [WxH ...] [--gpu] [--runs 10] [--frames FRAMES_DIR]
        a Head and an IFNet call (the median of the runs; on the GPU each timed to its bytes'
        download), the program's records by time (GPUProgram.profile)
"""
import argparse
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from reclank.models.vfi import video as V
from reclank.models.vfi.rife import RIFE


def load(d, i):
    return np.asarray(Image.open("%s/f%03d.png" % (d, i)).convert("RGB"), np.uint8)


def cmp(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    cos = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    return "1-cos %.2e, max|d| %.2e (max|ref| %.2e)" % (1 - cos, np.abs(a - b).max(), np.abs(b).max())


def to_u8(x):
    return np.clip(np.rint(np.asarray(x, np.float32) * np.float32(255)), 0, 255).astype(np.uint8)


def levels(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return int(d.max()), float((d > 0).mean())


def psnr(a, b):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    m = d.mean()
    return 10 * np.log10(255 ** 2 / m) if m else float("inf"), float((np.abs(a.astype(int) - b).max(-1) > 8).mean())


def hwc(x):
    return x.transpose(1, 2, 0).reshape(-1, x.shape[0])


def taps(a):
    z = np.load(a.ref)
    i, j, t = z["dump_key"]
    R = RIFE(a.model, gpu=a.gpu)
    R.taps = True
    f0, f1 = load(a.frames, int(i)), load(a.frames, int(j))
    R.reset()
    R.push(f0)
    R.push(f1)
    out8 = R.frame(float(t))
    h, w, Hp, Wp, s = R.size
    names = ["x0", "x1", "f0", "f1", "out", "A"] + ["tmp%d" % k for k in range(5)] + ["flow%d" % k for k in range(5)]
    o = R.outputs(names)
    print("%s %dx%d (padded %dx%d), the pair %d, %d at t %g" % ("GPU" if a.gpu else "CPU", w, h, Wp, Hp, i, j, t))
    for k in ("x0", "x1", "f0", "f1"):
        print("  %-6s %s" % (k, cmp(o[k], hwc(z[k]))))
    for k in range(5):
        print("  block%d tmp  %s" % (k, cmp(o["tmp%d" % k], hwc(z["block%d_tmp" % k]))))
        print("  block%d flow %s" % (k, cmp(o["flow%d" % k], hwc(z["block%d_flow" % k]))))
    print("  mask   %s" % cmp(o["A"][:, 15], z["mask"].reshape(-1)))
    ref = z["out_%d_%d_%g" % (i, j, t)]
    out = o["out"].reshape(Hp, Wp, 3)[:h, :w]
    mx, share = levels(out8, to_u8(ref))
    print("  out    %s; bytes: at most %d levels from the reference's, %.4f %% of the values differ" % (
        cmp(out, ref), mx, 100 * share))


def outputs(a):
    z = np.load(a.ref)
    keys = sorted((k for k in z.files if k.startswith("out_")), key=lambda k: [float(v) for v in k[4:].split("_")])
    R = RIFE(a.model, gpu=a.gpu)
    R.scale = float(a.scale) if a.scale else None
    pairs = {}
    for k in keys:
        i, j, t = k[4:].split("_")
        pairs.setdefault((int(i), int(j)), []).append(float(t))
    worst = dict(cos=0.0, d=0.0, lv=0, share=[], pr=[], pg=[], over_r=[], over_g=[])
    for (i, j), ts in pairs.items():
        R.reset()
        R.push(load(a.frames, i))
        R.push(load(a.frames, j))
        for t in ts:
            out8 = R.frame(t)
            h, w, Hp, Wp, _ = R.size
            out = R.outputs(["out"])["out"].reshape(Hp, Wp, 3)[:h, :w]
            ref = z["out_%d_%d_%g" % (i, j, t)]
            x, y = out.reshape(-1).astype(np.float64), ref.reshape(-1).astype(np.float64)
            worst["cos"] = max(worst["cos"], 1 - (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
            worst["d"] = max(worst["d"], float(np.abs(out - ref).max()))
            mx, share = levels(out8, to_u8(ref))
            worst["lv"] = max(worst["lv"], mx)
            worst["share"].append(share)
            if a.gt:
                gt = load(a.frames, int(round(i + (j - i) * t)))
                p1, o1 = psnr(to_u8(ref), gt)
                p2, o2 = psnr(out8, gt)
                worst["pr"].append(p1)
                worst["pg"].append(p2)
                worst["over_r"].append(o1)
                worst["over_g"].append(o2)
    n = sum(len(v) for v in pairs.values())
    print("%s %s: %d frames: 1-cos at most %.2e, max|d| %.2e; bytes at most %d levels from the reference's, %.4f %% "
          "of the values differ" % ("GPU" if a.gpu else "CPU", a.frames.rstrip("/").split("/")[-1], n, worst["cos"],
                                   worst["d"], worst["lv"], 100 * np.mean(worst["share"])))
    if a.gt:
        mp = lambda ps: 10 * np.log10(1 / np.mean([10 ** (-p / 10) for p in ps]))  # noqa: E731
        print("  against the held-out frames: the reference %.2f dB (> 8 levels %.2f %%), reclank %.2f dB (%.2f %%)" % (
            mp(worst["pr"]), 100 * np.mean(worst["over_r"]), mp(worst["pg"]), 100 * np.mean(worst["over_g"])))


def cuts(a):
    """the cut rule on footage: the thumbnails and the probe frames of every frame, every pair of a 2x
    job (gap 2) and a 4x job (gap 4); with MODEL_DIR (not "-") RIFE's residual of the probe frames
    (the full rule), else the thumbnails' rule alone. The cuts: the 59.94 fps frame difference's
    spikes (labels to review, not the truth: an animation's new drawings spike too)."""
    specs = a.rest
    prober = RIFE(a.model, gpu=a.gpu).prober() if a.model != "-" else None
    tot = {2: [0, 0, 0, 0], 4: [0, 0, 0, 0]}
    for v in range(0, len(specs), 3):
        src, ss, dur = specs[v:v + 3]
        info = V.probe(src)
        w, h = info["width"], info["height"]
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "6", "-ss", ss, "-t", dur, "-i", src, "-vsync", "0",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        th, pf = [], []
        while True:
            b = p.stdout.read(w * h * 3)
            if len(b) < w * h * 3:
                break
            f = np.frombuffer(b, np.uint8).reshape(h, w, 3)
            th.append(V.thumbs(f))
            pf.append(V.probe_frame(f))
        p.wait()
        n = len(th)
        d = np.array([np.abs(th[k + 1][1] - th[k][1]).mean() * 255 for k in range(n - 1)])
        src_cuts = [k for k in range(1, n - 2) if d[k] > 12 and d[k] > 4 * max(d[k - 1], d[k + 1])]
        line = []
        for gap in (2, 4):
            det = V.CutDetector()
            hit = fp = nc = 0
            fired = []
            if prober is not None:
                prober.reset()
                prober.push(pf[0])
            for k in range(0, n - gap, gap):
                rd = None
                if prober is not None:
                    prober.push(pf[k + gap])
                    rd = prober.residual()
                c = det(th[k], th[k + gap], rd)[0]
                truth = any(k <= x < k + gap for x in src_cuts)
                nc += truth
                hit += c and truth
                fp += c and not truth
                if c:
                    fired.append(k)
            t = tot[gap]
            t[0] += hit
            t[1] += nc
            t[2] += fp
            t[3] += (n - gap) // gap + 1
            line.append("gap %d: %d of %d cuts, %d false (cut pairs at %s)" % (gap, hit, nc, fp, fired))
        print("%s @%s +%s s (%d frames, %d source cuts): %s" % (src.split("/")[-1], ss, dur, n, len(src_cuts),
                                                                "; ".join(line)), flush=True)
    for gap, (hit, nc, fp, pairs) in tot.items():
        print("all, gap %d (%d pairs): found %d of %d cuts, %d false" % (gap, pairs, hit, nc, fp))


def timing(a):
    import os
    for size in a.rest:
        Wd, H = (int(v) for v in size.split("x"))
        d = a.frames
        fr = [np.asarray(Image.open("%s/f%03d.png" % (d, i)).convert("RGB").resize((Wd, H), Image.BICUBIC), np.uint8)
              for i in (1, 3)]
        R = RIFE(a.model, gpu=a.gpu)
        t0 = time.time()
        R.reset()
        R.push(fr[0])
        R.push(fr[1])
        R.frame(0.5)
        first = time.time() - t0
        th, tn = [], []
        sync = (lambda: None)
        if a.gpu:
            from reclank.backends.cuda.runtime import lib
            sync = lib().rc_gpu_sync            # (the byte path's push returns before its Head is done)
        for _ in range(a.runs):
            t = time.time()
            R.push(fr[1 - len(th) % 2])
            sync()
            th.append(time.time() - t)
            t = time.time()
            R.frame(0.5)
            tn.append(time.time() - t)
        h, w, Hp, Wp, s = R.size
        prof = ""
        if a.gpu:
            from reclank.ir.opcodes import OP_NAMES
            io = (h, w) if R.gpu_io else ()
            for label, key in (("IFNet", ("net", Hp, Wp, s) + io), ("Head", ("head", Hp, Wp, 1) + io + ((False,) if io else ()))):
                prog, g = R.runner.get(key, True)
                ms = g.profile()
                kinds = {}
                for (op, _), m in zip(prog.recs, ms):
                    kinds[OP_NAMES.get(op, op)] = kinds.get(OP_NAMES.get(op, op), 0.0) + float(m)
                top = sorted(kinds.items(), key=lambda kv: -kv[1])[:8]
                prof += "; %s records (ms): " % label + ", ".join("%s %.2f" % kv for kv in top)
        print("%s %dx%d (padded %dx%d, scale %g): the first frame %.2f s (the builds); a Head %.2f ms, an IFNet call "
              "%.2f ms (its bytes downloaded; the median of %d)%s" % (
                  "GPU" if a.gpu else "CPU(%s)" % os.environ.get("RECLANK_THREADS", "auto"), w, h, Wp, Hp, s, first,
                  1e3 * float(np.median(th)), 1e3 * float(np.median(tn)), a.runs, prof), flush=True)
        R.runner.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("mode", choices=["taps", "outputs", "cuts", "time"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--scale", default=None, help="outputs: RIFE's scale (the reference's --scale)")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--frames", default=None, help="time: the frames resized to each size (f001, f003)")
    a = ap.parse_args()
    if a.mode in ("taps", "outputs"):
        a.frames, a.ref = a.rest
    {"taps": taps, "outputs": outputs, "cuts": cuts, "time": timing}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
