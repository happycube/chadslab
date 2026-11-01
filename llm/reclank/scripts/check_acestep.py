"""Check reclank's ACE-Step 1.5 against its references.

    check_acestep.py MODEL_DIR [--ref ace.npz ...] [--vae-ref oobleck.npz ...] [--gpu]
        [--parts text,cond,dit,all]

--ref (scripts/acestep_reference.py: the authors' pipeline, thinking off):
  text: the ids of reclank's prompt and lyrics against the reference's; the text
    encoder's last hidden state on the reference's ids (1 - cos of each row)
  cond: the packed conditions of the reference's text and lyric inputs
  dit: each step's velocity on the reference's latents (the reference's conditions as
    reclank makes them); then the steps (and DCW) from the reference's noise
  all: the latents of the whole pipeline from the reference's caption and noise
--vae-ref (scripts/oobleck_reference.py): the VAE decoder's audio of real music's
  latents (the largest difference, the SNR in dB)."""
import argparse
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--ref", nargs="*", default=[])
ap.add_argument("--vae-ref", nargs="*", default=[])
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--parts", default="text,cond,dit,all")
a = ap.parse_args()
parts = set(a.parts.split(","))


def cmp(name, x, ref):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    x2, r2 = x.reshape(x.shape[0], -1), ref.reshape(ref.shape[0], -1)
    cos = (x2 * r2).sum(-1) / np.maximum(np.linalg.norm(x2, axis=-1) * np.linalg.norm(r2, axis=-1), 1e-30)
    d = 1 - cos
    rel = np.abs(x - ref).max() / max(np.abs(ref).max(), 1e-30)
    print("  %-10s 1-cos worst %.2e (row %d) mean %.2e  rel %.2e" % (name, d.max(), d.argmax(), d.mean(), rel),
          flush=True)


if a.vae_ref:
    from reclank.models.acestep.vae import OobleckVAE
    vae = OobleckVAE(os.path.join(a.model, "vae"), gpu=a.gpu)
    for p in a.vae_ref:
        r = np.load(p)
        for rep in range(2):
            t1 = time.time()
            au = vae.decode(r["z"])
            dt = time.time() - t1
        d = au - r["audio"]
        snr = 10 * np.log10((r["audio"] ** 2).sum() / max((d ** 2).sum(), 1e-30))
        print("%s: %d frames -> %s, max |d| %.2e, SNR %.1f dB; %.2f s" % (
            p.split("/")[-1], len(r["z"]), au.shape, np.abs(d).max(), snr, dt))
if a.ref:
    from reclank.encoders.runtime import RESIDENCY
    from reclank.models.acestep.pipeline import AceStep
    t0 = time.time()
    ace = AceStep(a.model, gpu=a.gpu)
    print("loaded in %.1f s (%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU"), flush=True)
for p in a.ref:
    r = np.load(p)
    print(p.split("/")[-1])
    frames = r["noise"].shape[0]
    if "text" in parts:
        ids = ace.text_ids(str(r["caption"]), bpm=int(r["bpm"]) if "bpm" in r else 120, keyscale="G major",
                           duration=frames / 25)
        print("  text ids %s (%d)" % ("same" if ids == list(r["text_ids"]) else "DIFFER", len(ids)))
        lids = ace.lyric_ids(str(r["lyrics"]), "en")
        print("  lyric ids %s (%d)" % ("same" if lids == list(r["lyric_ids"]) else "DIFFER", len(lids)))
        t1 = time.time()
        h = ace.encode_text(list(r["text_ids"]))
        print("  text %.2f s" % (time.time() - t1))
        cmp("text", h, r["text_hidden"])
        cmp("lyric", ace.embed_rows(list(r["lyric_ids"])), r["lyric_hidden"])
    with RESIDENCY.use(ace.dit.runner if a.gpu else None):    # (the condition's caches to the last step)
        Te = None
        if parts & {"cond", "dit"}:
            t1 = time.time()
            enc = ace.dit.condition(r["text_hidden"], r["lyric_hidden"])
            print("  cond %.2f s" % (time.time() - t1))
            cmp("packed", ace.dit.packed, r["enc"])
            Te = enc.shape[0]
        if "dit" in parts:
            hints = None
            if "lm_codes" in r:
                import re
                hints = ace.dit.detokenize([int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", str(r["lm_codes"]))])
            cmp("context", ace.dit.context(frames, hints), r["context"])
            i = 0
            while "x%d" % i in r:
                t1 = time.time()
                v = ace.dit.velocity(r["x%d" % i], float(r["t%d" % i]), r["context"], Te)
                dt = time.time() - t1
                cmp("v%d" % i, v, r["v%d" % i])
                i += 1
            print("  a step %.3f s" % dt)
            lat = ace.denoise(r["noise"], r["context"], Te, steps=i)
            cmp("latent", lat, r["latent"])
    with RESIDENCY.use(ace.dit.runner if a.gpu else None):
        if "all" in parts:
            t1 = time.time()
            text = ace.encode_text(list(r["text_ids"]))
            Te = ace.dit.condition(text, ace.embed_rows(list(r["lyric_ids"]))).shape[0]
            lat = ace.denoise(r["noise"], ace.dit.context(frames), Te)
            print("  whole: %.2f s" % (time.time() - t1))
            cmp("latent", lat, r["latent"])
