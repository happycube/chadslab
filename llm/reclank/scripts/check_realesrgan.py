"""Real-ESRGAN against its float32 reference (scripts/realesrgan_reference.py) on the reference's
input: the network's output (1 - cos, the largest difference) and the uint8 image (levels of 255,
PSNR).

    check_realesrgan.py MODEL_DIR MODEL_ID REF.npz [--gpu] [--threads 10] [--denoise 0.5]"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.realesrgan.model import RealESRGAN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("id")
    ap.add_argument("ref")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--denoise", type=float, default=0.5)
    a = ap.parse_args()
    R = np.load(a.ref)
    m = RealESRGAN(a.model, a.id, gpu=a.gpu, cpu_threads=a.threads, denoise=a.denoise)
    for k in range(2 if a.gpu else 1):
        t0 = time.time()
        parts = {}
        out = m.upscale(R["lq"], parts)
        print("%s %s: %.2f s" % (a.id, R["lq"].shape, time.time() - t0), flush=True)
    y, r = parts["y"].astype(np.float64).ravel(), R["y"].astype(np.float64).ravel()
    print("y          1-cos %.3g  rel %.3g  max %.3g" % (1 - y @ r / np.sqrt((y @ y) * (r @ r)),
                                                       np.linalg.norm(y - r) / np.linalg.norm(r), np.abs(y - r).max()))
    d = np.abs(out.astype(np.float64) - R["out"].astype(np.float64))
    mse = (d ** 2).mean()
    print("out        levels of 255: max %d, mean %.4f; PSNR %.1f dB" % (
        d.max(), d.mean(), 10 * np.log10(255 ** 2 / mse) if mse else float("inf")))


if __name__ == "__main__":
    main()
