"""Check reclank's Qwen-Image-2.1 against diffusers' (scripts/qwen_image_reference.py), part by
part on the reference's own inputs.

    check_qwen_image.py MODEL_DIR ref.npz [ref.npz ...] [--gpu] [--text-gpu]
        [--parts pixels,vision,text,venc,dit,vae,all] [--steps N]

text: the prompt's ids (reclank's tokenizer and template), the dropped system tokens, then the
text encoder's rows (1 - cos of each token row, the largest difference relative to the largest
value). dit: each step's DiT output on the reference's input latents (the first N steps, default
all; CFG: the prompt's and the negative prompt's), then the steps from the reference's noise.
vae: the image (RGBA) of the reference's latents. all: the pipeline from the reference's noise
(the text encoder on the CPU unless --text-gpu).

An edit's reference (made with --image): pixels: the references' resize and the processor's
pixel values and the VAE's input (the same values or not); vision: the vision tower's rows and
its deepstack rows; text: the ids, the MRoPE positions and the image slots, then the text
encoder's rows (of reclank's vision rows, and of the reference's: the text model alone); venc:
the VAE encoder's latents of the reference's input; dit: on the reference's text rows and
latents; all: from the reference images and noise (also the PSNR of the image). The pipeline's
output_resolution is found from the first image's resized size."""
import argparse
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("refs", nargs="+")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--text-gpu", action="store_true")
ap.add_argument("--parts", default="text")
ap.add_argument("--steps", type=int, default=None)
a = ap.parse_args()
parts = set(a.parts.split(","))


def cmp(name, x, ref, rows=True):
    """1 - cos of each row (or of the whole), and the largest difference relative to the
    reference's largest value."""
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    rel = np.abs(x - ref).max() / max(np.abs(ref).max(), 1e-30)
    if rows:
        x2, r2 = x.reshape(x.shape[0], -1), ref.reshape(ref.shape[0], -1)
        cos = (x2 * r2).sum(-1) / np.maximum(np.linalg.norm(x2, axis=-1) * np.linalg.norm(r2, axis=-1), 1e-30)
        d = 1 - cos
        print("  %-12s 1-cos worst %.2e (row %d) mean %.2e  rel %.2e" % (name, d.max(), d.argmax(), d.mean(), rel),
              flush=True)
        return d.max()
    cos = (x * ref).sum() / max(np.linalg.norm(x) * np.linalg.norm(ref), 1e-30)
    print("  %-12s 1-cos %.2e  rel %.2e" % (name, 1 - cos, rel), flush=True)
    return 1 - cos


def levels(name, img, ref):
    """The differences of two images in [-1, 1] in levels of 255 (the PNG's), by channel, and
    the PSNR of the RGBA levels."""
    q = lambda x: np.round((np.asarray(x) / 2 + 0.5).clip(0, 1) * 255)  # noqa: E731
    d = np.abs(q(img) - q(ref))
    mse = (d ** 2).mean()
    print("  %-12s levels of 255: max %d, mean %.3f; by channel (R G B A) mean %s, share differing %.4f, "
          "PSNR %s" % (name, d.max(), d.mean(), " ".join("%.3f" % v for v in d.reshape(d.shape[0], -1).mean(1)),
                       (d > 0).mean(), "%.1f dB" % (10 * np.log10(255 ** 2 / mse)) if mse else "inf"), flush=True)


def same(name, x, ref):
    """Whether x has ref's shape and values."""
    x, ref = np.asarray(x), np.asarray(ref)
    ok = x.shape == ref.shape and np.array_equal(x, ref)
    print("  %-12s %s%s" % (name, "the same" if ok else "DIFFER", "" if ok or x.shape != ref.shape else
                            " (max %.3g)" % np.abs(x.astype(np.float64) - ref).max()), flush=True)
    return ok


refs = [dict(np.load(p)) for p in a.refs]
from PIL import Image  # noqa: E402
from reclank.encoders.runtime import RESIDENCY  # noqa: E402
from reclank.models.qwen_image.pipeline import (QwenImage21, calculate_dimensions, condition_image,  # noqa: E402
                                                 vae_pixels)

if a.gpu:
    # an edit's vision tower and VAE encoder are done after their turn: the manager evicts them when the DiT
    # needs their room (the pipeline releases nothing of theirs)
    RESIDENCY.enabled = True
need = set(parts)
if "all" in parts:
    need |= {"text", "dit", "vae"}
if "vision" in parts:
    need.add("text")
if "venc" in parts:
    need.add("vae")
t0 = time.time()
qi = QwenImage21(a.model, gpu=a.gpu, parts=need & {"text", "dit", "vae"}, text_gpu=a.text_gpu)
print("loaded in %.1f s (%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU"), flush=True)
for p, r in zip(a.refs, refs):
    print(p)
    edit = "hw" in r
    if edit:
        hh, ww = (int(v) // 16 for v in r["hw"])
    else:
        hh = ww = int(round(np.sqrt(r["noise"].shape[0]))) if "noise" in r else 0
    cfg = float(r["cfg"]) if "cfg" in r else 1.0
    if edit:
        n_img = len([k for k in r if k.startswith("src")])
        srcs = [Image.fromarray(r["src%d" % k]) for k in range(n_img)]
        s0 = tuple(int(v) for v in r["size0"])
        res_ = [R for R in range(64, 4097, 32)
                if calculate_dimensions(R * R, srcs[0].size[0] / srcs[0].size[1]) == s0][0]
        imgs = [condition_image(im, res_) for im in srcs]
        print("  output_resolution %d, references %s, output %d x %d" % (
            res_, [im.size for im in imgs], ww * 16, hh * 16), flush=True)
        shapes = [(im.size[1] // 16, im.size[0] // 16) for im in imgs]
        ilat = [r["ilat%d" % k] for k in range(n_img)] if "ilat0" in r else None
    if edit and "pixels" in parts:
        for k, im in enumerate(imgs):
            same("size%d" % k, np.array(im.size), r["size%d" % k])
            same("vae_in%d" % k, vae_pixels(im), r["vae_in%d" % k])
        pix = [qi.vision_patches(im) for im in imgs]
        mine = np.concatenate([pp for pp, _ in pix])
        rp = r["pixels"].reshape(mine.shape[0], 3, 2, -1)
        same("grid", np.array([[1, gh, gw] for _, (gh, gw) in pix]), r["grid"])
        same("pixels t0", mine.reshape(rp.shape[0], 3, -1), rp[:, :, 0])
        same("pixels t1", mine.reshape(rp.shape[0], 3, -1), rp[:, :, 1])
    vis = None
    if edit and parts & {"vision", "text"}:
        t1 = time.time()
        vis = qi.vision_rows(imgs)
        dt = time.time() - t1
        if "vision" in parts:
            cmp("vis", np.concatenate([y for y, _, _ in vis]), r["vis"])
            for j in range(len(vis[0][1])):
                cmp("ds%d" % j, np.concatenate([d[j] for _, d, _ in vis]), r["ds%d" % j])
            print("  vision %.2f s" % dt)
    if "text" in parts:
        for key, pk in (("", "prompt"), ("n", "negative")):
            if key + "ids" not in r:
                continue
            if edit:
                grids = [g for _, _, g in vis]
                ids = qi.tokenize_edit(str(r[pk]), grids)
                print("  %s ids %s (%d tokens), drop %d (reference %d)" % (
                    pk, "same" if np.array_equal(ids, r[key + "ids"]) else "DIFFER", len(ids), qi.drop,
                    int(r["drop"])))
                same(key + "pos", qi.rope_index(ids, qi.image_pad, grids), r[key + "pos"])
                t1 = time.time()
                emb, slots = qi.edit_embed(str(r[pk]), vis)
                dt = time.time() - t1
                same(key + "imgpad", slots, r[key + "imgpad"])
                cmp(key + "emb", emb, r[key + "emb"])
                # the rows the DiT reads (the image slots' rows are replaced by the latents)
                cmp(key + "emb text", emb[~slots], r[key + "emb"][~slots])
                print("  text %.2f s" % dt)
                # the text model alone: the reference's vision rows
                rv, r0 = [], 0
                for y, ds, g in vis:
                    n = y.shape[0]
                    rv.append((r[key + "vis"][r0:r0 + n], [r[key + "ds%d" % j][r0:r0 + n] for j in range(len(ds))], g))
                    r0 += n
                emb2, _ = qi.edit_embed(str(r[pk]), rv)
                cmp(key + "emb ref vis", emb2, r[key + "emb"])
                cmp(key + "emb text", emb2[~slots], r[key + "emb"][~slots])
            else:
                ids = qi.tokenize(str(r[pk]))
                same_ = np.array_equal(ids, r[key + "ids"])
                print("  %s ids %s (%d tokens), drop %d (reference %d)" % (
                    pk, "same" if same_ else "DIFFER", len(ids), qi.drop, int(r["drop"])))
                t1 = time.time()
                emb = qi.text_embed(r[key + "ids"])
                dt = time.time() - t1
                cmp(key + "emb", emb, r[key + "emb"])
                print("  text %.2f s" % dt)
    if "noise" not in r:
        continue
    if edit and "venc" in parts:
        for k, im in enumerate(imgs):
            t1 = time.time()
            z = qi.vae.encode(r["vae_in%d" % k])
            dt = time.time() - t1
            cmp("ilat%d" % k, qi.vae.pack(z), r["ilat%d" % k])
            print("  vae encode %.2f s" % dt)

    def cond(key):
        """The DiT's condition of the reference's text rows (and latents)."""
        if not edit:
            e = r[key + "emb"]
            return dict(txt=e, media=None, frame=e.shape[0])
        return qi.joint(r[key + "emb"], r[key + "imgpad"], ilat, shapes)
    if "dit" in parts:
        steps = len(r["timesteps"])
        sig, ts = qi.sigmas(steps, hh * ww)
        cmp("sigmas", sig, r["sigmas"], rows=False)
        cmp("timesteps", ts, r["timesteps"], rows=False)
        c = cond("")
        cn = cond("n") if cfg > 1 else None
        with RESIDENCY.use(qi.dit.runner if a.gpu else None):    # (the prefix's keys and values to the last step)
            t1 = time.time()
            qi.dit.prefix(c["txt"], 0, c["media"])
            if cn is not None:
                qi.dit.prefix(cn["txt"], 1, cn["media"])
            print("  prefix %.2f s" % (time.time() - t1))
            for i in range(steps if a.steps is None else min(steps, a.steps)):
                t = float(np.float32(np.float32(r["timesteps"][i]) / np.float32(1000)) * np.float32(1000))
                t1 = time.time()
                v = qi.dit.velocity(r["x%d" % i], t, c["txt"].shape[0], hh, ww, 0, c["frame"])
                dt = time.time() - t1
                if cfg > 1:
                    cmp("vc%d" % i, v, r["vc%d" % i])
                    vu = qi.dit.velocity(r["x%d" % i], t, cn["txt"].shape[0], hh, ww, 1, cn["frame"])
                    cmp("vu%d" % i, vu, r["vu%d" % i])
                else:
                    cmp("v%d" % i, v, r["v%d" % i])
                print("  step %.2f s" % dt)
        if a.steps is None:
            x = qi.denoise(r["noise"], c, hh, ww, steps, cn, cfg)
            cmp("latent", x, r["latent"])
    if "vae" in parts:
        if a.gpu and qi.dit is not None:
            qi.dit.runner.release(keep_host=True)
        cmp("z", qi.vae.unpack(r["latent"], hh, ww), r["z"], rows=False)
        t1 = time.time()
        img = qi.vae.decode(r["z"])
        dt = time.time() - t1
        cmp("image", img, r["image"], rows=False)
        for j, ch in enumerate("RGBA"):
            cmp("image " + ch, img[j], r["image"][j], rows=False)
        levels("image", img, r["image"])
        print("  vae %.2f s" % dt)
        if a.gpu:
            qi.vae.runner.release(keep_host=True)
    if "all" in parts:
        t1 = time.time()
        if edit:
            vis = qi.vision_rows(imgs)
            emb, slots = qi.edit_embed(str(r["prompt"]), vis)
            nemb, nslots = qi.edit_embed(str(r["negative"]), vis) if cfg > 1 else (None, None)
            lats, shp = qi.reference_latents(imgs)
            txt = qi.joint(emb, slots, lats, shp)
            neg = qi.joint(nemb, nslots, lats, shp) if cfg > 1 else None
        else:
            txt = qi.text_embed(r["ids"])
            neg = qi.text_embed(r["nids"]) if cfg > 1 else None
        lat = qi.denoise(r["noise"], txt, hh, ww, len(r["timesteps"]), neg, cfg)
        if a.gpu:
            qi.dit.runner.release(keep_host=True)
        img = qi.image(lat, hh, ww)
        dt = time.time() - t1
        cmp("latent", lat, r["latent"])
        cmp("image", img, r["image"], rows=False)
        levels("image", img, r["image"])
        print("  all %.2f s" % dt)
        if a.gpu:
            qi.vae.runner.release(keep_host=True)
