"""Text to image with FLUX.2 [klein] (reclank/models/flux2), or an edit of images.

    flux2_generate.py MODEL_DIR --prompt TEXT --out image.png [--size 1024 | --height H --width W]
        [--image PATH ...] [--steps 4] [--seed 0] [--noise ref.npz] [--cpu]

--image (once a reference image: editing, image to image) as the pipeline's image=; the size
the first image's (after the pipeline's resize) unless given.

--noise takes the starting noise of a flux2_reference.py file (its "noise"), so the image
can be compared with diffusers'; else NumPy's normal noise of the seed."""
import argparse
import time

import numpy as np
from PIL import Image

from reclank.models.flux2.pipeline import Flux2Klein, load_image

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int, default=None)
ap.add_argument("--height", type=int)
ap.add_argument("--width", type=int)
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--noise")
ap.add_argument("--cpu", action="store_true")
a = ap.parse_args()
t0 = time.time()
fx = Flux2Klein(a.model, gpu=not a.cpu)
noise = np.load(a.noise)["noise"] if a.noise else None
t1 = time.time()
size = (a.height, a.width) if a.height and a.width else a.size
img = fx.generate(a.prompt, size=size, steps=a.steps, seed=a.seed, noise=noise,
                  images=[load_image(p) for p in a.image])
t2 = time.time()
Image.fromarray(img).save(a.out)
print("%s: %dx%d in %.1f s (load %.1f s)" % (a.out, img.shape[1], img.shape[0], t2 - t1, t1 - t0))
