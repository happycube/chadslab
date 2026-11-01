"""Text to image, or an edit of reference images, with Qwen-Image-2.1 (reclank/models/qwen_image): an
RGBA PNG.

    qwen_image_generate.py MODEL_DIR --prompt TEXT --out image.png [--size 1024 | --width W --height H]
        [--steps 40] [--seed 0] [--negative TEXT --cfg 4.0] [--noise ref.npz] [--cpu] [--keep]
        [--image PATH ... [--resolution 1024]]

--image: the reference images (up to 10; each resized to the area resolution^2), the output's size
by default the last one's aspect ratio at that area (as the pipeline's).

--noise takes the starting noise of a qwen_image_reference.py file (its "noise"), so the image
can be compared with diffusers'; else NumPy's normal noise of the seed. A transparent image takes
the model card's prompt form ("This is an RGBA image with transparency. ... The image has alpha
channel and the background is transparent.")."""
import argparse
import time

import numpy as np
from PIL import Image

from reclank.encoders.runtime import RESIDENCY
from reclank.models.qwen_image.pipeline import QwenImage21

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int)
ap.add_argument("--width", type=int)
ap.add_argument("--height", type=int)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--negative")
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--noise")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--keep", action="store_true", help="keep the DiT and the VAE on the GPU (no other GPU user)")
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--resolution", type=int, default=1024)
a = ap.parse_args()
if a.image and not a.cpu:
    RESIDENCY.enabled = True        # (the vision tower and the VAE encoder go when the DiT needs their room)
t0 = time.time()
qi = QwenImage21(a.model, gpu=not a.cpu)
noise = np.load(a.noise)["noise"] if a.noise else None
size = None if a.image and not (a.size or a.height or a.width) else \
    (a.height or a.size or 1024, a.width or a.size or 1024)
images = [Image.open(p) for p in a.image]
t1 = time.time()
img = qi.generate(a.prompt, size=size, steps=a.steps, seed=a.seed, noise=noise, negative=a.negative, cfg=a.cfg,
                  keep=a.keep, images=images, resolution=a.resolution)
t2 = time.time()
Image.fromarray(img, "RGBA").save(a.out)
print("%s: %dx%d in %.1f s (load %.1f s)" % (a.out, img.shape[1], img.shape[0], t2 - t1, t1 - t0))
