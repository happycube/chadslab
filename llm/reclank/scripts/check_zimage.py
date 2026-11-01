"""Check reclank's Z-Image Turbo against diffusers' (scripts/zimage_reference.py), part by part on
the reference's own inputs.

    check_zimage.py MODEL_DIR ref.npz [ref.npz ...] [--gpu] [--parts text,sched,dit,vae,all]
        [--steps 0,4,8 | all] [--f20] [--text-gpu] [--ref-text] [--taps taps.npz] [--save DIR]

text: the prompt's ids (reclank's tokenizer and template), then the text encoder's rows (1 - cos
of each token's row, the largest difference relative to the largest value). sched: the sigmas of
the reference's steps and of 1 to 100 steps, the timesteps (the bits), then each step of the
scheduler on the reference's DiT outputs and latents (the values that differ, the largest
difference in ulps). dit: each DiT call of the given steps (default all) on the reference's input
latents and text rows (1 - cos of each token's 64 values: its patch of 2 x 2 x 16, the mean and
the worst; the norm of the difference relative to the reference's; the largest difference
relative to the largest value); --taps: the first call's RoPE tables and rows against the
reference's taps (CPU only). vae: the image of the reference's latents (levels of 255 of
diffusers' postprocess: the largest and the mean difference, PSNR). all: the pipeline from the
reference's noise and prompt (reclank's text encoder; --ref-text: the reference's text rows):
the final latents, the image, with its times. The CPU's DiT is float32 as stored (--f20: F20);
the GPU's F20. --save DIR: all's images (KEY.png) and the reference's (KEY_ref.png)."""
import argparse
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("refs", nargs="+")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--parts", default="text")
ap.add_argument("--steps", default="all")
ap.add_argument("--f20", action="store_true")
ap.add_argument("--text-gpu", action="store_true")
ap.add_argument("--ref-text", action="store_true")
ap.add_argument("--taps")
ap.add_argument("--save")
a = ap.parse_args()
parts = set(a.parts.split(","))


def rows_cmp(name, x, ref):
    """1 - cos of each row (worst, mean), the largest difference relative to the reference's
    largest value."""
    x, ref = np.asarray(x, np.float64).reshape(len(x), -1), np.asarray(ref, np.float64).reshape(len(ref), -1)
    cos = (x * ref).sum(1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(ref, axis=1), 1e-300)
    d = 1 - cos
    rel = np.linalg.norm(x - ref) / np.linalg.norm(ref)
    mx = np.abs(x - ref).max() / np.abs(ref).max()
    print("  %-12s 1-cos mean %.2e worst %.2e (row %d)  rel %.2e  max %.2e" % (
        name, d.mean(), d.max(), d.argmax(), rel, mx), flush=True)
    return d


def toks(x):
    """The rows of the DiT's tokens (h/2 w/2, 64) of latents (16, h, w)."""
    C, h, w = x.shape
    return np.asarray(x, np.float64).reshape(C, h // 2, 2, w // 2, 2).transpose(1, 3, 2, 4, 0).reshape(-1, 4 * C)


def u8(v):
    return np.round(np.clip(np.asarray(v, np.float64) / 2 + 0.5, 0, 1) * 255)


def levels(name, img, ref):
    """The difference of two images (3, H, W) in [-1, 1] in levels of 255: the largest, the mean,
    the share of values that differ and that differ by more than 8, PSNR."""
    x, r = u8(img), u8(ref)
    d = np.abs(x - r)
    mse = ((x - r) ** 2).mean()
    print("  %-12s levels of 255: max %d mean %.4f, share differing %.5f, over 8 %.5f; PSNR %.1f dB" % (
        name, d.max(), d.mean(), (d > 0).mean(), (d > 8).mean(), 10 * np.log10(255.0 ** 2 / max(mse, 1e-30))),
        flush=True)


def ulps(x, ref):
    """The values that differ, and the largest difference in ulps (float32)."""
    xi = np.asarray(x, np.float32).view(np.int32).astype(np.int64)
    ri = np.asarray(ref, np.float32).view(np.int32).astype(np.int64)
    xi = np.where(xi < 0, -(xi & 0x7FFFFFFF), xi)
    ri = np.where(ri < 0, -(ri & 0x7FFFFFFF), ri)
    d = np.abs(xi - ri)
    return int((d > 0).sum()), int(d.max())


def same_bits(x, ref):
    x, ref = np.asarray(x, np.float32), np.asarray(ref, np.float32)
    return x.shape == ref.shape and np.array_equal(x.view(np.int32), ref.view(np.int32))


def taps(r):
    """The first call's RoPE tables, caption rows and last block's rows against the reference's
    taps (--taps; a CPU program's buffers)."""
    tp = dict(np.load(a.taps))
    d = zi.dit
    T = len(r["emb"])
    hh, ww = r["noise"].shape[1] // 2, r["noise"].shape[2] // 2
    Ip, Tp = padded(hh * ww), padded(T)
    cos, sin = d.rope(Tp, hh, ww)
    for j, (lo, hi) in enumerate(((0, Ip), (Ip, Ip + Tp))):
        # (the caption's ids of _pad_with_ids have its padding twice: the rows past Tp unused)
        ref = tp["rope%d" % j][:hi - lo]
        print("  rope%d cos %s, sin %s" % (j, "same bits" if same_bits(cos[lo:hi], ref[..., 0])
                                           else "%d differ, at most %d ulps" % ulps(cos[lo:hi], ref[..., 0]),
                                           "same bits" if same_bits(sin[lo:hi], ref[..., 1])
                                           else "%d differ, at most %d ulps" % ulps(sin[lo:hi], ref[..., 1])))
    prog, _ = d.runner.get(("text", T, Tp), False)
    rows_cmp("cap", prog.names["x"], tp["ctx%d" % (len(d.ctx) - 1)][0])
    prog, _ = d.runner.get(("step", Tp, hh, ww), False)
    rows_cmp("unified", prog.names["x"], tp["layer%d" % (len(d.layers) - 1)][0])


refs = [(p, dict(np.load(p))) for p in a.refs]
from reclank.encoders.runtime import RESIDENCY  # noqa: E402
from reclank.models.zimage.dit import padded  # noqa: E402
from reclank.models.zimage.pipeline import ZImageTurbo  # noqa: E402

load = set()
load |= {"text"} if parts & {"text", "all"} and not (a.ref_text and parts <= {"all"}) else set()
load |= {"dit"} if parts & {"dit", "all"} else set()
load |= {"vae"} if parts & {"vae", "all"} else set()
t0 = time.time()
zi = ZImageTurbo(a.model, gpu=a.gpu, parts=load or ("none",), text_gpu=a.text_gpu, f20=True if a.f20 else None)
print("loaded %s in %.1f s (%s%s)" % (",".join(sorted(load)) or "-", time.time() - t0, "GPU" if a.gpu else "CPU",
                                      ", F20" if a.f20 or a.gpu else ""), flush=True)
for p, r in refs:
    key = os.path.splitext(os.path.basename(p))[0]
    steps = len(r["timesteps"])
    print("%s: %r, %s, %d steps" % (p, str(r["prompt"])[:60], "x".join(str(8 * s) for s in r["noise"].shape[1:]),
                                    steps), flush=True)
    if "text" in parts:
        ids = zi.tokenize(str(r["prompt"]))
        print("  ids %s (%d tokens)" % ("same" if np.array_equal(ids, r["ids"]) else "DIFFER", len(ids)))
        t1 = time.time()
        emb = zi.text_embed(r["ids"])
        dt = time.time() - t1
        rows_cmp("emb", emb, r["emb"])
        print("  text %.2f s" % dt, flush=True)
    if "sched" in parts:
        sig, ts = zi.sigmas(steps)
        print("  sigmas %s, timesteps %s" % ("same bits" if same_bits(sig, r["sigmas"]) else "DIFFER",
                                              "same bits" if same_bits(ts, r["timesteps"]) else "DIFFER"))
        tab = r["sched_sigmas"]
        bad = [n for n in range(1, 101) if not same_bits(zi.sigmas(n)[0], tab[n - 1, :n + 1])]
        print("  sigmas of 1 to 100 steps: %s" % ("all the same bits" if not bad else "DIFFER at %s" % bad))
        tin = [zi.dit_time(t) for t in ts]
        print("  the DiT's timesteps %s" % ("same bits" if same_bits(tin, [r["t%d" % i][0] for i in range(steps)])
                                            else "DIFFER"))
        worst, nd = 0, 0
        for i in range(steps):
            x = (r["x%d" % i] + (sig[i + 1] - sig[i]) * -r["v%d" % i]).astype(np.float32)
            n, m = ulps(x, r["x%d" % (i + 1)] if i + 1 < steps else r["latent"])
            nd, worst = nd + n, max(worst, m)
        print("  scheduler: %d steps, %d values differ in all, at most %d ulps" % (steps, nd, worst), flush=True)
    if "dit" in parts:
        which = list(range(steps)) if a.steps == "all" else [int(v) for v in a.steps.split(",")]
        with RESIDENCY.use(zi.dit.runner if a.gpu else None):    # (the caption's rows to the last step)
            t1 = time.time()
            zi.dit.prefix(r["emb"])
            print("  prefix %.2f s (%d tokens, %d rows)" % (time.time() - t1, len(r["emb"]), padded(len(r["emb"]))))
            for i in which:
                t1 = time.time()
                v = zi.dit.velocity(r["x%d" % i], float(r["t%d" % i][0]))
                dt = time.time() - t1
                rows_cmp("v%d" % i, toks(v), toks(r["v%d" % i]))
                print("  step %d (t %.4f): a call %.2f s" % (i, float(r["t%d" % i][0]), dt), flush=True)
                if i == 0 and a.taps and not a.gpu:
                    taps(r)
    if "vae" in parts:
        if a.gpu and zi.dit is not None:
            zi.release(("dit",))
        z = (r["latent"] / zi.scale + zi.shift).astype(np.float32)
        print("  z %s" % ("same bits" if same_bits(z, r["z"]) else "differs: %d values, at most %d ulps" % ulps(z, r["z"])))
        t1 = time.time()
        img = zi.vae.decode(r["z"])
        dt = time.time() - t1
        x, ref = img.reshape(-1).astype(np.float64), r["image"].reshape(-1).astype(np.float64)
        print("  image 1-cos %.2e  max diff %.2e" % (1 - x @ ref / np.linalg.norm(x) / np.linalg.norm(ref),
                                                     np.abs(x - ref).max()))
        levels("image", img, r["image"])
        print("  vae %.2f s" % dt, flush=True)
        if a.gpu:
            zi.release(("vae",))
    if "all" in parts:
        t1 = time.time()
        emb = r["emb"] if a.ref_text else zi.text_embed(zi.tokenize(str(r["prompt"])))
        t2 = time.time()
        lat = zi.denoise(r["noise"], emb, steps)
        t3 = time.time()
        if a.gpu:
            zi.release(("dit",))
        img = zi.image(lat)
        t4 = time.time()
        if a.gpu:
            zi.release(("vae",))
        rows_cmp("latent", toks(lat), toks(r["latent"]))
        levels("image", img, r["image"])
        print("  all: text %.1f s, denoise %.1f s (%.2f s a step), vae %.1f s" % (
            t2 - t1, t3 - t2, (t3 - t2) / steps, t4 - t3), flush=True)
        if a.save:
            from PIL import Image
            os.makedirs(a.save, exist_ok=True)
            Image.fromarray(u8(img).transpose(1, 2, 0).astype(np.uint8)).save(os.path.join(a.save, key + ".png"))
            Image.fromarray(u8(r["image"]).transpose(1, 2, 0).astype(np.uint8)).save(
                os.path.join(a.save, key + "_ref.png"))
            np.save(os.path.join(a.save, key + "_latent.npy"), lat)
