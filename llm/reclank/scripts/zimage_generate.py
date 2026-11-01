"""Text to image with Z-Image Turbo (reclank/models/zimage): a PNG.

    zimage_generate.py MODEL_DIR --prompt TEXT --out image.png [--size 1024 | --width W --height H]
        [--steps 9] [--seed 0] [--noise ref.npz] [--cpu [--f20]] [--text-gpu] [--keep]

--noise takes the starting noise of a zimage_reference.py file (its "noise"), so the image can
be compared with diffusers'; else NumPy's normal noise of the seed. The text encoder runs on the
CPU (--text-gpu: on the GPU, the DiT leaving it first)."""
import argparse
import time

import numpy as np
from PIL import Image

from reclank.models.zimage.pipeline import ZImageTurbo

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--width", type=int)
ap.add_argument("--height", type=int)
ap.add_argument("--steps", type=int, default=9)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--noise")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--f20", action="store_true", help="the CPU's DiT in F20 (the GPU's is)")
ap.add_argument("--text-gpu", action="store_true")
ap.add_argument("--keep", action="store_true", help="keep the DiT and the VAE on the GPU (no other GPU user)")
a = ap.parse_args()
t0 = time.time()
zi = ZImageTurbo(a.model, gpu=not a.cpu, text_gpu=a.text_gpu, f20=True if a.f20 else None)
noise = np.load(a.noise)["noise"] if a.noise else None
size = (a.height or a.size, a.width or a.size)
t1 = time.time()
img = zi.generate(a.prompt, size=size, steps=a.steps, seed=a.seed, noise=noise, keep=a.keep)
t2 = time.time()
Image.fromarray(img).save(a.out)
print("%s: %dx%d in %.1f s (load %.1f s)" % (a.out, img.shape[1], img.shape[0], t2 - t1, t1 - t0))
