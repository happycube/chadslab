"""Check reclank's Stable Diffusion 1.5 / 2.0 against diffusers' (scripts/sd_reference.py),
part by part on the reference's own inputs.

    check_sd.py MODEL_DIR ref.npz [ref.npz ...] [--gpu] [--parts text,sched,unet,vae,all]
                [--unet-steps 3]

text: the ids of the prompt and of the negative prompt (reclank's CLIP tokenizer), then
the text encoder's output (1 - cos of each token row, the largest difference relative to
the largest value). sched: the timesteps, alphas_cumprod, and each step of the scheduler
on the reference's UNet outputs (the latents after it against the reference's next
input). unet: the first --unet-steps UNet calls on the reference's latents, timestep and
text rows (both halves of the batch). vae: the image of the reference's latents. all:
the pipeline from the reference's noise (text, steps, image)."""
import argparse
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("refs", nargs="+")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--parts", default="text,sched")
ap.add_argument("--unet-steps", type=int, default=3)
ap.add_argument("--bf12", action="store_true", help="the UNet's matrices in BF12 on the GPU (a study)")
ap.add_argument("--f20", action="store_true", help="the UNet's matrices in F20 (16 significant bits)")
ap.add_argument("--png", help="the whole pipeline's image (part all) to this PNG")
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


refs = [dict(np.load(p)) for p in a.refs]
from reclank.models.sd.pipeline import StableDiffusion  # noqa: E402

t0 = time.time()
sd = StableDiffusion(a.model, gpu=a.gpu, parts=parts - {"sched"} or ("text",), bf12=a.bf12, f20=a.f20)
print("loaded in %.1f s (%s), scheduler %s %s" % (time.time() - t0, "GPU" if a.gpu else "CPU", sd.sched.kind,
                                                 sd.sched.pred), flush=True)
for p, r in zip(a.refs, refs):
    print(p)
    n_calls = len(r["timesteps"])
    if "text" in parts:
        ids = np.array([sd.tok.encode(str(r["negative"])), sd.tok.encode(str(r["prompt"]))], np.int32)
        print("  ids %s, negative ids %s" % ("same" if np.array_equal(ids[1], r["ids"]) else "DIFFER",
                                             "same" if np.array_equal(ids[0], r["uids"]) else "DIFFER"))
        t1 = time.time()
        out = sd.text.hidden(np.stack([r["uids"], r["ids"]]))
        dt = time.time() - t1
        cmp("uemb", out[0], r["uemb"])
        cmp("emb", out[1], r["emb"])
        print("  text %.2f s (2 prompts)" % dt)
    if "sched" in parts:
        s = sd.sched
        ts = s.timesteps(len(r["timesteps"]) - (s.kind == "PNDMScheduler"))
        print("  timesteps %s, alphas_cumprod max |d| %.2e" % (
            "same" if np.array_equal(ts, r["timesteps"]) else "DIFFER %s" % ts,
            np.abs(s.alphas_cumprod - r["alphas_cumprod"]).max()))
        s.start(len(r["timesteps"]) - (s.kind == "PNDMScheduler"))
        g = np.float32(r["guidance"])
        worst = 0.0
        for i, t in enumerate(ts):
            e = r["e%d" % i]
            x = s.step(e[0] + g * (e[1] - e[0]), t, r["x%d" % i])
            nxt = r["x%d" % (i + 1)] if i + 1 < n_calls else r["latent"]
            worst = max(worst, np.abs(x - nxt).max() / np.abs(nxt).max())
        print("  scheduler steps on the reference's outputs: worst rel %.2e" % worst)
    if "unet" in parts:
        ctx = np.stack([r["uemb"], r["emb"]])
        for i in range(min(a.unet_steps, n_calls)):
            x = r["x%d" % i]
            t1 = time.time()
            e = sd.unet.output(np.stack([x, x]), float(r["t%d" % i]), ctx)
            dt = time.time() - t1
            cmp("e%d neg" % i, e[0].reshape(e.shape[1], -1).T, r["e%d" % i][0].reshape(e.shape[1], -1).T)
            cmp("e%d prompt" % i, e[1].reshape(e.shape[1], -1).T, r["e%d" % i][1].reshape(e.shape[1], -1).T)
            print("  step %.2f s (t %d)" % (dt, int(r["t%d" % i])))
    if "vae" in parts:
        t1 = time.time()
        img = sd.image(r["latent"])
        dt = time.time() - t1
        cmp("image", img, r["image"], rows=False)
        d = np.abs(img - r["image"])
        print("  image max |d| %.2e mean %.2e (of [-1, 1]); vae %.2f s" % (d.max(), d.mean(), dt))
    if "all" in parts:
        t1 = time.time()
        uemb, emb = sd.text.hidden(np.stack([r["uids"], r["ids"]]))
        steps = n_calls - (sd.sched.kind == "PNDMScheduler")
        lat = sd.denoise(r["noise"], emb, uemb, steps, float(r["guidance"]))
        img = sd.image(lat)
        dt = time.time() - t1
        cmp("latent", lat, r["latent"], rows=False)
        cmp("image", img, r["image"], rows=False)
        d = np.abs(img - r["image"])
        u8 = lambda im: np.round(np.clip(im / 2 + 0.5, 0, 1) * 255)  # noqa: E731
        d8 = np.abs(u8(img) - u8(r["image"]))
        psnr = 10 * np.log10(255.0 ** 2 / max(((u8(img) - u8(r["image"])) ** 2).mean(), 1e-12))
        print("  image max |d| %.2e mean %.2e (of [-1, 1]); 8-bit max %d mean %.3f, PSNR %.1f dB; %d steps %.2f s" % (
            d.max(), d.mean(), d8.max(), d8.mean(), psnr, steps, dt))
        if a.png:
            from PIL import Image
            Image.fromarray(u8(img).transpose(1, 2, 0).astype(np.uint8)).save(
                a.png if len(a.refs) == 1 else a.png.replace(".png", "_%s.png" % p.split("/")[-1][:-4]))
