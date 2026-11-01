"""Check reclank's Whisper against transformers' (scripts/whisper_reference.py): the input
features and the encoder's last hidden state of real clips.

    check_whisper.py MODEL_DIR clips.npz refs.npz [--gpu] [--bf12] [--batch N]

Prints, a clip: the features' largest difference, the encoder's worst and mean 1 - cos
over the 1500 rows, and its largest difference relative to the reference's largest value."""
import argparse
import time

import numpy as np

from reclank.models.whisper.encoder import WhisperEncoder, log_mel

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("clips")
ap.add_argument("refs")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--bf12", action="store_true")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--threads", type=int, default=None, help="the team of a CPU program (default: the cores)")
a = ap.parse_args()
d, r = np.load(a.clips), np.load(a.refs)
names = list(d["names"])
feats = [log_mel(d["c%d" % i]) for i in range(len(names))]
t0 = time.time()
enc = WhisperEncoder(a.model, gpu=a.gpu, bf12=a.bf12, batch=a.batch, cpu_threads=a.threads)
print("loaded in %.1f s (%s%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU", ", BF12" if a.bf12 else ", float32"))
# the encoder on the reference's features, so its check is apart from the features'
hs = enc.encode([r["f%d" % i] for i in range(len(names))])
t0 = time.time()
enc.encode([r["f%d" % i] for i in range(len(names))])
dt = time.time() - t0
worst = 0.0
for i, n in enumerate(names):
    fd = float(np.abs(feats[i] - r["f%d" % i]).max())
    e, h = r["e%d" % i].astype(np.float64), hs[i].astype(np.float64)
    cos = np.sum(e * h, 1) / np.linalg.norm(e, axis=1) / np.linalg.norm(h, axis=1)
    rel = float(np.abs(e - h).max() / np.abs(e).max())
    worst = max(worst, float(1 - cos.min()))
    print("%-40s features %.1e  encoder 1-cos worst %.2e mean %.2e  rel %.1e" % (n, fd, 1 - cos.min(), 1 - cos.mean(), rel))
print("worst 1-cos %.2e; %.0f ms a window (%d windows, batch %d)" % (worst, 1000 * dt / len(names), len(names), a.batch))
