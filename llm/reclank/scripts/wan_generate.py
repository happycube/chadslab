"""Text to video with Wan 2.2 TI2V-5B or T2V-A14B (reclank/models/wan): an MP4 (H.264, 24 fps; the
A14B 16 fps).

    wan_generate.py MODEL_DIR --prompt TEXT --out video.mp4 [--width 832 --height 480] [--frames 17]
        [--steps N] [--guidance G] [--guidance2 G2] [--seed 0] [--negative TEXT] [--noise ref.npz] [--cpu]
        [--frames-out F.npy] [--keeper NAME | --no-keeper]

The steps and guidance by default the model card's (the 5B 50 and 5.0; the A14B 40, 4.0 and 3.0,
the low-noise expert's guidance2); the A14B from a weightkeeper that holds it when one runs.

The prompt and the negative prompt (the model card's by default) through umT5-XXL on the CPU
(models/t5). --noise takes the starting noise of a wan_reference.py file (its "noise"), so the
video can be compared with diffusers'; else NumPy's normal noise of the seed. --frames-out: the
frames (T, H, W, 3) uint8 as well. The steps, the VAE and ffmpeg's times are printed."""
import argparse
import time

import numpy as np

from reclank.models.wan.pipeline import WanT2V, write_mp4

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=832)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--frames", type=int, default=17)
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--guidance", type=float, default=None)
ap.add_argument("--guidance2", type=float, default=None)
ap.add_argument("--keeper", default=None)
ap.add_argument("--no-keeper", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--negative")
ap.add_argument("--noise")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--cpu-threads", type=int)
ap.add_argument("--frames-out")
a = ap.parse_args()
t0 = time.time()
wan = WanT2V(a.model, gpu=not a.cpu, cpu_threads=a.cpu_threads, keeper=False if a.no_keeper else a.keeper)
steps = a.steps or wan.defaults()[0]
noise = np.load(a.noise)["noise"] if a.noise else None
times = {}


def progress(stage, i, n):
    times.setdefault(stage, [time.time(), 0])[1] = time.time()
    if stage == "step":
        print("  step %d/%d %.1f s" % (i, n, time.time() - t1), flush=True)


t1 = time.time()
fr = wan.generate(a.prompt, a.negative, size=(a.height, a.width), frames=a.frames, steps=steps,
                  guidance=a.guidance, seed=a.seed, noise=noise, progress=progress, guidance2=a.guidance2)
t2 = time.time()
mp4 = write_mp4(fr, a.out, wan.fps)
if a.frames_out:
    np.save(a.frames_out, fr)
text = times["text"][1] - t1
dsteps = times["step"][1] - times["text"][1]
print("%s: %d frames of %dx%d in %.1f s (load %.1f s; text %.1f s, steps %.1f s, %.2f s a step, vae %.1f s, "
      "mp4 %.1f s)" % (a.out, fr.shape[0], fr.shape[2], fr.shape[1], t2 - t1, t1 - t0, text, dsteps, dsteps / steps,
                       t2 - times["step"][1], mp4))
