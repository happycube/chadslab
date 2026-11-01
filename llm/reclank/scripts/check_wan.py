"""Check reclank's Wan 2.2 TI2V-5B or T2V-A14B against diffusers' (scripts/wan_reference.py), part
by part on the reference's own inputs.

    check_wan.py MODEL_DIR ref.npz [ref.npz ...] [--gpu] [--parts sched,dit,vae,all] [--steps 0,12,23]
        [--f20] [--save DIR] [--guidance G] [--guidance2 G2] [--keeper NAME | --no-keeper]
        [--save-dit OUT.npz] [--lock FILE]

The A14B: each step's expert (the reference's "experts"), the dit part's default steps 0, the
last high-noise step, the first low-noise one and the next to last (each expert's text caches
first), the guidance of each expert (the model card's 4.0 and 3.0 by default); the DiTs from a
weightkeeper that holds the directory when one runs (--keeper: that one; --no-keeper: loaded);
--save-dit: the dit part's outputs (to compare two runs' bits).

sched: the sigmas and timesteps, then each step of the scheduler on the reference's guided
outputs and latents (the bits: the values that differ, the largest difference in ulps). dit:
each DiT call of the given steps (default 0, half, the next to last; "all") on the reference's
input latents, the prompt's and the negative's (1 - cos of each token's 192 values: its patch
of 1 x 2 x 2 x 48, the mean and the worst; the norm of the difference relative to the
reference's; the largest difference relative to the largest value), and the guided output. vae:
the video of the reference's latents (levels of 255 of diffusers' postprocess: the largest and
the mean difference, PSNR). all: the pipeline from the reference's noise and text embeddings
(the final latents, the video), with its times. The CPU's DiT is float32 as stored (--f20: F20);
the GPU's F20. --save DIR: all's frames (KEY.npy, uint8) and an MP4 of each."""
import argparse
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("refs", nargs="+")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--parts", default="sched")
ap.add_argument("--steps", default=None)
ap.add_argument("--f20", action="store_true")
ap.add_argument("--save")
ap.add_argument("--guidance", type=float, default=None)
ap.add_argument("--guidance2", type=float, default=None)
ap.add_argument("--keeper", default=None)
ap.add_argument("--no-keeper", action="store_true")
ap.add_argument("--save-dit")
ap.add_argument("--lock", help="flock this file after the load (the GPU lock: the load outside it)")
a = ap.parse_args()
parts = set(a.parts.split(","))


def toks(x):
    """The rows of a DiT's tokens (f h/2 w/2, 192) of latents (48, f, h, w)."""
    C, f, h, w = x.shape
    return np.asarray(x, np.float64).reshape(C, f, h // 2, 2, w // 2, 2).transpose(1, 2, 4, 0, 3, 5).reshape(-1, C * 4)


def tok(name, x, ref):
    """1 - cos of each token (mean, worst), the relative norm of x - ref, the largest difference
    relative to the reference's largest value."""
    xt, rt = toks(x), toks(ref)
    cos = (xt * rt).sum(1) / np.maximum(np.linalg.norm(xt, axis=1) * np.linalg.norm(rt, axis=1), 1e-300)
    d = 1 - cos
    rel = np.linalg.norm(xt - rt) / np.linalg.norm(rt)
    mx = np.abs(xt - rt).max() / np.abs(rt).max()
    print("  %-12s 1-cos mean %.2e worst %.2e  rel %.2e  max %.2e" % (name, d.mean(), d.max(), rel, mx), flush=True)
    return dict(cos_mean=d.mean(), cos_max=d.max(), rel=rel, max=mx)


def u8(v):
    return np.round(np.clip(np.asarray(v) / 2 + 0.5, 0, 1) * 255)


def levels(name, vid, ref):
    """The difference of two videos (3, T, H, W) in [-1, 1] in levels of 255: the largest, the
    mean, the share of values that differ, PSNR (of all frames, and the worst frame's)."""
    x, r = u8(vid), u8(ref)
    d = np.abs(x - r)
    mse = ((x - r) ** 2).reshape(3, x.shape[1], -1).mean((0, 2))
    ps = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-30))
    allp = 10 * np.log10(255.0 ** 2 / max(mse.mean(), 1e-30))
    print("  %-12s levels of 255: max %d mean %.4f, share differing %.5f; PSNR %.1f dB (worst frame %.1f)" % (
        name, d.max(), d.mean(), (d > 0).mean(), allp, ps.min()), flush=True)


def ulps(x, ref):
    """The values that differ, and the largest difference in ulps (float32)."""
    xi = np.asarray(x, np.float32).view(np.int32).astype(np.int64)
    ri = np.asarray(ref, np.float32).view(np.int32).astype(np.int64)
    xi = np.where(xi < 0, -(xi & 0x7FFFFFFF), xi)
    ri = np.where(ri < 0, -(ri & 0x7FFFFFFF), ri)
    d = np.abs(xi - ri)
    return int((d > 0).sum()), int(d.max())


refs = [(p, dict(np.load(p))) for p in a.refs]
from reclank.encoders.runtime import RESIDENCY  # noqa: E402
from reclank.models.wan.pipeline import WanT2V, frames_u8, write_mp4  # noqa: E402

load = {"dit"} if "dit" in parts else set()
load |= {"vae"} if "vae" in parts else set()
load |= {"dit", "vae"} if "all" in parts else set()
t0 = time.time()
wan = WanT2V(a.model, gpu=a.gpu, parts=load or ("none",), f20=True if a.f20 else None,
             keeper=False if a.no_keeper else a.keeper)
print("loaded in %.1f s (%s%s%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU", ", F20" if a.f20 or a.gpu else "",
                                     ", the weightkeeper's" if wan.keeper is not None else ""), flush=True)
_steps, g1, g2 = wan.defaults()
g1 = g1 if a.guidance is None else a.guidance
g2 = (g2 if a.guidance2 is None else a.guidance2) if wan.a14b else None
gs = (np.float32(g1), np.float32(g1 if g2 is None else g2))
saved = {}
if a.lock:
    import fcntl
    lockf = open(a.lock, "a")
    t1 = time.time()
    fcntl.flock(lockf, fcntl.LOCK_EX)
    print("the lock after %.1f s" % (time.time() - t1), flush=True)
for p, r in refs:
    key = os.path.splitext(os.path.basename(p))[0]
    print(p, flush=True)
    steps = len(r["timesteps"])
    if "sched" in parts:
        s = wan.sched
        sig, ts = s.set_timesteps(steps)
        print("  sigmas %s, timesteps %s" % ("same bits" if np.array_equal(sig.view(np.int32), r["sigmas"].view(np.int32))
                                              else "DIFFER", "same" if np.array_equal(ts, r["timesteps"]) else "DIFFER"))
        worst, nd = 0, 0
        for i in range(steps):
            x = s.step(r["v%d" % i], r["x%d" % i])
            ref = r["x%d" % (i + 1)] if i + 1 < steps else r["latent"]
            n, m = ulps(x, ref)
            nd += n
            worst = max(worst, m)
            if n:
                print("  step %d: %d values differ, at most %d ulps" % (i, n, m))
        print("  scheduler: %d steps, %d values differ in all, at most %d ulps" % (steps, nd, worst), flush=True)
    if "experts" in r and "sched" in parts:
        mine = np.array([wan.expert(t)[1] for t in r["timesteps"]], np.int32)
        print("  experts %s (%d high-noise steps)" % ("same" if np.array_equal(mine, r["experts"]) else "DIFFER",
                                                     int((mine == 0).sum())), flush=True)
    if "dit" in parts:
        ex = [wan.expert(t)[1] for t in r["timesteps"]]
        if a.steps == "all":
            which = list(range(steps))
        elif a.steps:
            which = [int(v) for v in a.steps.split(",")]
        elif wan.a14b and 1 in ex:
            lo = ex.index(1)
            which = sorted({0, max(lo - 1, 0), lo, steps - 2})
        else:
            which = [0, steps // 2, steps - 2]
        # (the prefix's keys and values to the last step; the A14B's experts share the runner)
        with RESIDENCY.use(wan.dit.runner if a.gpu else None):
            cur = None
            for i in which:
                t = float(r["timesteps"][i])
                dit, e = wan.expert(r["timesteps"][i])
                if dit is not cur:
                    t1 = time.time()
                    dit.prefix(r["emb"], r["nemb"])
                    print("  prefix (expert %d) %.2f s" % (e, time.time() - t1))
                    cur = dit
                t1 = time.time()
                vc = dit.velocity(r["x%d" % i], t, 0)
                dt = time.time() - t1
                vu = dit.velocity(r["x%d" % i], t, 1)
                tok("vc%d" % i, vc, r["vc%d" % i])
                tok("vu%d" % i, vu, r["vu%d" % i])
                tok("v%d" % i, vu + gs[e] * (vc - vu), r["v%d" % i])
                saved.update({"%s_vc%d" % (key, i): vc, "%s_vu%d" % (key, i): vu})
                print("  step %d (t %d, expert %d): a call %.2f s" % (i, int(t), e, dt), flush=True)
    if "vae" in parts:
        if a.gpu and wan.dit is not None:
            wan.release(("dit",))
        z = wan.vae.denorm(r["latent"])
        print("  z %s" % ("same bits" if np.array_equal(z.view(np.int32), r["z"].view(np.int32)) else
                          "differs: %d values, at most %d ulps" % ulps(z, r["z"])))
        t1 = time.time()
        vid = wan.vae.decode(r["z"])
        dt = time.time() - t1
        x, ref = vid.reshape(-1).astype(np.float64), r["video"].reshape(-1).astype(np.float64)
        print("  video 1-cos %.2e  max diff %.2e" % (1 - x @ ref / np.linalg.norm(x) / np.linalg.norm(ref),
                                                     np.abs(x - ref).max()))
        levels("video", vid, r["video"])
        print("  vae %.2f s (%d frames)" % (dt, vid.shape[1]), flush=True)
        if a.gpu:
            wan.release(("vae",))
    if "all" in parts:
        t1 = time.time()
        lat = wan.denoise(r["noise"], r["emb"], r["nemb"], steps, g1, guidance2=g2)
        t2 = time.time()
        if a.gpu:
            wan.release(("dit",))
        vid = wan.video(lat)
        t3 = time.time()
        if a.gpu:
            wan.release(("vae",))
        tok("latent", lat, r["latent"])
        levels("video", vid, r["video"])
        print("  all: denoise %.1f s (%.2f s a step), vae %.1f s" % (t2 - t1, (t2 - t1) / steps, t3 - t2), flush=True)
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            fr = frames_u8(vid)
            np.save(os.path.join(a.save, key + ".npy"), fr)
            np.save(os.path.join(a.save, key + "_latent.npy"), lat)
            write_mp4(fr, os.path.join(a.save, key + ".mp4"), wan.fps)
            write_mp4(frames_u8(r["video"]), os.path.join(a.save, key + "_ref.mp4"), wan.fps)
if a.save_dit:
    np.savez(a.save_dit, **saved)
