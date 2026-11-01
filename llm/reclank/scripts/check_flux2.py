"""Check reclank's FLUX.2 [klein] against diffusers' (scripts/flux2_reference.py), part by
part on the reference's own inputs.

    check_flux2.py MODEL_DIR ref.npz [ref.npz ...] [--gpu] [--parts text,pre,enc,dit,vae,all]

text: the prompt's ids (reclank's tokenizer and chat template), then the text encoder's
hidden states (1 - cos of each token row, the largest difference relative to the largest
value). pre: each reference image's pixels (load_image and preprocess of its file) against
the pipeline's preprocess output (the same bits?), and of the server's path (server.generate's
edit_form: RGBA kept for the models that take it, then preprocess). enc: the VAE encoder's latents of the
reference's pixels, the DiT's reference rows (packed, normalized) and the RoPE tables of the
reference's position ids. dit: each step's DiT output on the reference's input latents (and
reference rows). vae: the image of the reference's latents. all: the pipeline from the
reference's noise (reclank's text encoder, preprocess and VAE encoder); the image's levels of
255 (diffusers' postprocess) and PSNR."""
import argparse
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("refs", nargs="+")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--parts", default="text")
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
    """The difference of two images (3, H, W) in [-1, 1] in levels of 255: the largest, the mean,
    the share of values over 8, PSNR."""
    u8 = lambda v: np.round(np.clip(np.asarray(v, np.float64) / 2 + 0.5, 0, 1) * 255)  # noqa: E731
    x, r = u8(img), u8(ref)
    d = np.abs(x - r)
    mse = ((x - r) ** 2).mean()
    print("  %-12s levels of 255: max %d mean %.4f, over 8 %.5f; PSNR %.1f dB" % (
        name, d.max(), d.mean(), (d > 8).mean(), 10 * np.log10(255.0 ** 2 / max(mse, 1e-30))), flush=True)


def grid(r):
    """The image's (h, w) in tokens of a reference file."""
    if "size" in r:
        return int(r["size"][0]) // 16, int(r["size"][1]) // 16
    hh = int(round(np.sqrt(r["noise"].shape[0])))
    return hh, hh


def ref_grids(r):
    """The reference images' (h, w) in tokens, and their pixels."""
    pix = [r["ref_pixels%d" % i] for i in range(len(r.get("images", [])))]
    return [(x.shape[1] // 16, x.shape[2] // 16) for x in pix], pix


refs = [dict(np.load(p)) for p in a.refs]
from reclank.encoders.runtime import RESIDENCY  # noqa: E402
from reclank.models.flux2.dit import rope_table  # noqa: E402
from reclank.models.flux2.pipeline import Flux2Klein, load_image, preprocess  # noqa: E402

if a.gpu:
    # the parts leave the card as the server's do when the next one does not fit (RESIDENCY: the
    # same bits): the files' text encoder, VAE encoder and decoder programs side by side do not
    RESIDENCY.enabled = True
t0 = time.time()
fx = Flux2Klein(a.model, gpu=a.gpu, parts=[{"enc": "vae"}.get(q, q) for q in parts])
print("loaded in %.1f s (%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU"), flush=True)
for p, r in zip(a.refs, refs):
    print(p)
    if "text" in parts:
        prompt = str(r["prompt"]) if "prompt" in r else None
        if prompt is not None:
            ids, tokens = fx.tokenize(prompt)
            same = np.array_equal(ids, r["ids"]) and tokens == int(r["mask"].sum())
            print("  ids %s (%d tokens)" % ("same" if same else "DIFFER", tokens))
        t1 = time.time()
        emb = fx.text_embed(r["ids"][None], [int(r["mask"].sum())])[0]
        dt = time.time() - t1
        cmp("emb", emb, r["emb"])
        w = emb.shape[1] // 3
        for j in range(3):
            cmp("tap %d" % j, emb[:, j * w:(j + 1) * w], r["emb"][:, j * w:(j + 1) * w])
        print("  text %.2f s" % dt)
    grids, pix = ref_grids(r)
    ref_rows = r.get("ref_rows")
    if "pre" in parts:
        from server.generate import edit_form
        for i, path in enumerate(r.get("images", [])):
            path = str(path)
            data = open(path, "rb").read()
            _, (im,) = edit_form({"image": [(data, os.path.basename(path))]})
            for how, x in (("load_image", preprocess(load_image(data))), ("server %s" % im.mode, preprocess(im))):
                same = x.shape == pix[i].shape and np.array_equal(x, pix[i])
                d = np.abs(x - pix[i]).max() if x.shape == pix[i].shape else np.inf
                print("  pixels %s %s (%s): %s (max |d| %.2e)" % (
                    os.path.basename(path), "x".join(map(str, x.shape[:0:-1])), how, "same" if same else "DIFFER", d),
                    flush=True)
    if "enc" in parts and pix and "ref_latent0" in r:
        for i, x in enumerate(pix):
            t1 = time.time()
            z = fx.vae.encode(x)
            dt = time.time() - t1
            cmp("latent%d" % i, z, r["ref_latent%d" % i], rows=False)
            zr = r["ref_latent%d" % i]
            cmp("  (pixels)", z.reshape(z.shape[0], -1).T, zr.reshape(zr.shape[0], -1).T)
            print("  encoder %dx%d %.2f s" % (x.shape[2], x.shape[1], dt))
        rows, gr = fx.references(pix)
        cmp("ref rows", rows, ref_rows)
    if "enc" in parts and pix and "ref_ids" in r and fx.dit is not None:
        hh, ww = grid(r)
        T = r["emb"].shape[0]
        txt = np.zeros((T, 4), np.int64)
        txt[:, 3] = np.arange(T)
        img = np.stack([np.zeros(hh * ww, np.int64), np.repeat(np.arange(hh), ww), np.tile(np.arange(ww), hh),
                        np.zeros(hh * ww, np.int64)], 1)
        want = rope_table(np.concatenate([txt, img, r["ref_ids"].astype(np.int64)]), fx.dit.axes, fx.dit.theta)
        got = fx.dit.rope(T, hh, ww, grids)
        same = all(np.array_equal(g, w_) for g, w_ in zip(got, want))
        print("  rope of the reference's ids %s" % ("same" if same else "DIFFER"))
    if "dit" in parts:
        hh, ww = grid(r)
        sig = fx.sigmas(len(r["timesteps"]), hh * ww)
        cmp("sigmas", sig, r["sigmas"], rows=False)
        for i in range(len(r["timesteps"])):
            t = np.float32(np.float32(r["timesteps"][i]) / np.float32(1000)) * np.float32(1000)
            t1 = time.time()
            v = fx.dit.velocity(r["x%d" % i], float(t), r["emb"], hh, ww, ref_rows, grids)
            dt = time.time() - t1
            cmp("v%d" % i, v, r["v%d" % i])
            print("  step %.2f s" % dt)
        x = fx.denoise(r["noise"], r["emb"], hh, ww, len(r["timesteps"]), ref=ref_rows, refs=grids)
        cmp("latent", x, r["latent"])
    if "vae" in parts:
        hh, ww = grid(r)
        cmp("z", fx.vae.unnormalize(r["latent"], hh, ww), r["z"], rows=False)
        t1 = time.time()
        img = fx.vae.decode(r["z"])
        dt = time.time() - t1
        cmp("image", img, r["image"], rows=False)
        print("  image max |d| %.2e (of [-1, 1]); vae %.2f s" % (np.abs(img - r["image"]).max(), dt))
    if "all" in parts:
        hh, ww = grid(r)
        t1 = time.time()
        txt = fx.text_embed(r["ids"][None], [int(r["mask"].sum())])[0]
        ref, refs_ = (None, ())
        if pix:
            ref, refs_ = fx.references([preprocess(load_image(open(str(p), "rb").read())) for p in r["images"]])
        lat = fx.denoise(r["noise"], txt, hh, ww, len(r["timesteps"]), ref=ref, refs=refs_)
        img = fx.image(lat, hh, ww)
        dt = time.time() - t1
        cmp("latent", lat, r["latent"])
        cmp("image", img, r["image"], rows=False)
        d = np.abs(img - r["image"])
        print("  image max |d| %.2e mean %.2e (of [-1, 1]); %.2f s" % (d.max(), d.mean(), dt))
        levels("image", img, r["image"])
