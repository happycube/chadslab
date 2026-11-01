"""SeedVR2-3B against its float32 reference (scripts/seedvr2_reference.py), part by part on the
reference's own inputs, then the whole restoration from the reference's noise.

    check_seedvr2.py MODEL_DIR REF.npz [--taps TAPS.npz] [--gpu] [--parts resize,vae,dit,all]
        [--threads 10] [--f20 | --no-f20] [--save OUT.npz]

MODEL_DIR: the 3B's or the 7B's (its checkpoint decides; the reference of the same variant).

    resize: transform() of the reference's input frames against its "x"
    vae:    the encoder of the reference's "xp" (its mean, logvar), the decoder of its x0 / 0.9152
            (its "dec"): 1 - cos, the relative norm, the largest difference, levels of 255
    dit:    the DiT's call on the reference's input (its noise, latent): v per token (1 - cos mean
            and worst), the relative norm; with --taps the RoPE angles and the window rows
    all:    the pipeline (restore) from the reference's noise and posterior noise: the uint8 output
            against the reference's (levels: max, mean; PSNR)
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.seedvr2 import pipeline as P  # noqa: E402
from reclank.models.seedvr2.vae import SCALE  # noqa: E402


def cmp(name, x, y, rows=None):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    a, b = x.ravel(), y.ravel()
    cos = 1 - a @ b / np.sqrt((a @ a) * (b @ b))
    line = "%-10s 1-cos %.3g  rel %.3g  max %.3g" % (name, cos, np.linalg.norm(a - b) / np.linalg.norm(b),
                                                       np.abs(a - b).max())
    if rows is not None:
        xr, yr = x.reshape(rows, -1), y.reshape(rows, -1)
        c = 1 - (xr * yr).sum(1) / np.sqrt((xr * xr).sum(1) * (yr * yr).sum(1))
        line += "  per row 1-cos mean %.3g worst %.3g" % (c.mean(), c.max())
    print(line, flush=True)
    return cos


def levels(name, img, ref):
    """img, ref: in [-1, 1] floats, or uint8"""
    if img.dtype != np.uint8:
        img = np.round((np.clip(img, -1, 1) + 1) * 127.5)
        ref = np.round((np.clip(ref, -1, 1) + 1) * 127.5)
    d = np.abs(img.astype(np.float64) - ref.astype(np.float64))
    mse = (d ** 2).mean()
    print("%-10s levels of 255: max %d, mean %.4f, over 8: %.4f%%; PSNR %.1f dB" % (
        name, d.max(), d.mean(), 100 * (d > 8).mean(), 10 * np.log10(255 ** 2 / mse) if mse else float("inf")),
        flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ref")
    ap.add_argument("--taps")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--parts", default="resize,vae,dit,all")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--f20", dest="f20", action="store_true", default=None)
    ap.add_argument("--no-f20", dest="f20", action="store_false")
    ap.add_argument("--save", help="the restore's output and stages (all) to this .npz (to compare runs)")
    a = ap.parse_args()
    R = np.load(a.ref)
    parts = a.parts.split(",")
    rh, rw = (int(v) for v in R["res"])
    if "resize" in parts:
        x = P.transform(R["lq"], rh, rw)
        cmp("resize", x, R["x"])
    need = set()
    if "vae" in parts or "all" in parts:
        need.add("vae")
    if "dit" in parts or "all" in parts:
        need.add("dit")
    if not need:
        return
    t0 = time.time()
    m = P.SeedVR2(a.model, gpu=a.gpu, cpu_threads=a.threads, parts=tuple(need), f20=a.f20)
    print("loaded %.1f s" % (time.time() - t0), flush=True)
    video = R["xp"].shape[1] > 1
    if "vae" in parts:
        for k in range(2 if a.gpu else 1):
            t0 = time.time()
            if video:
                mean, logvar = m.video_vae().encode(R["xp"])
            else:
                mean, logvar = (p[:, None] for p in m.vae.encode(R["xp"][:, 0]))
            print("encode %.2f s" % (time.time() - t0), flush=True)
        cmp("mean", mean, R["mean"])
        cmp("logvar", logvar, R["logvar"])
        if a.gpu and video:
            m.release(("vvae",))        # the encoder's programs and caches go before the decoder's (720p)
        z = (R["x0"] / np.float32(SCALE)).transpose(3, 0, 1, 2).astype(np.float32)
        for k in range(2 if a.gpu else 1):
            t0 = time.time()
            dec = m.video_vae().decode(z)[:, :R["dec"].shape[0]] if video else m.vae.decode(z[:, 0])[:, None]
            print("decode %.2f s" % (time.time() - t0), flush=True)
        dec = dec.transpose(1, 0, 2, 3)
        cmp("decode", dec, R["dec"])
        levels("decode", dec, R["dec"])
        if a.gpu:
            m.release(("vae", "vvae"))
    if "dit" in parts:
        if a.taps:
            T = np.load(a.taps)
            t_, H, W, _ = R["latent"].shape
            for kind in (0, 1):
                lay = m.dit.layout(t_, H // 2, W // 2, bool(kind))
                N = t_ * (H // 2) * (W // 2)
                rows = lay[0][:, 0]
                vid_rows = rows[rows < N]
                same = np.array_equal(vid_rows, T["win%d_rows" % kind])
                print("windows %d: %d, rows the reference's: %s" % (kind, len(lay[4]), same))
            ang = m.dit.angles(t_, H // 2, W // 2, False)
            cmp("txt angles", ang[-m.dit.L:], T["tang"][:, ::2])
        vid = np.concatenate([R["noise"], R["latent"], np.ones(R["latent"].shape[:-1] + (1,), np.float32)], -1)
        for k in range(2):
            t0 = time.time()
            v = m.dit.velocity(vid)
            print("dit call %.2f s" % (time.time() - t0), flush=True)
        n = v.shape[0] * v.shape[1] * v.shape[2] // 4
        vt = v.reshape(v.shape[0], v.shape[1] // 2, 2, v.shape[2] // 2, 2, 16).transpose(0, 1, 3, 2, 4, 5)
        rt = R["v"].reshape(vt.shape[0], vt.shape[1], 2, vt.shape[2], 2, 16).transpose(0, 1, 3, 2, 4, 5)
        cmp("v", vt, rt, rows=n)
        x0 = R["noise"] - v
        cmp("x0", x0, R["x0"])
        if a.gpu:
            m.release(("dit",))
    if "all" in parts:
        st = {}
        t0 = time.time()
        out = m.restore(R["lq"], res=(rh, rw), noise=R["noise"], pnoise=R["pnoise"], parts=st)
        print("restore %.2f s" % (time.time() - t0))
        cmp("latent", st["latent"], R["latent"])
        cmp("x0", st["x0"], R["x0"])
        cmp("dec", st["dec"], R["dec"])
        cmp("fixed", st["fixed"], R["fixed"])
        levels("out", out, R["out"])
        if a.save:
            np.savez(a.save, out=out, v=st["v"], x0=st["x0"], dec=st["dec"])


if __name__ == "__main__":
    main()
