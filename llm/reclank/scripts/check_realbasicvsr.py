#!/usr/bin/env python3
"""RealBasicVSR (models/realbasicvsr) against the official network's dumps on real frames: the cleaned frames,
the flows both ways, each frame's backward and forward features and the output (before tensor2img's clamp and
rounding), each as its SNR against the official's (dB: the official's power over the difference's), its largest
difference, and the share of the uint8 output's values that match.

The dumps (NAME.npz: x, clean, flow_f, flow_b, fb, ff, out) are the official mmediting v0.16.0 network's, float32
on the CPU (torch), run in a sandbox on the frames FRAMES.npy (t, h, w, 3) uint8.

    check_realbasicvsr.py MODEL_DIR DUMP.npz FRAMES.npy [--model realbasicvsr] [--gpu] [--segment N]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def snr(ref, x):
    ref, x = np.asarray(ref, np.float64), np.asarray(x, np.float64)
    e = ((x - ref) ** 2).sum()
    return float("inf") if e == 0 else 10 * np.log10((ref ** 2).sum() / e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("dump")
    ap.add_argument("frames")
    ap.add_argument("--model", default="realbasicvsr")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--segment", type=int, default=None)
    a = ap.parse_args()
    from reclank.models.realbasicvsr.model import RealBasicVSR
    R = np.load(a.dump)
    fr = np.load(a.frames)
    t, h, w, _ = fr.shape
    m = RealBasicVSR(a.model_dir, a.model, gpu=a.gpu)
    parts = {}
    t0 = time.time()
    out = m.upscale_video(fr, segment=a.segment, parts=parts)
    dt = time.time() - t0
    chw = lambda v, c: v.reshape(-1, h, w, c).transpose(0, 3, 1, 2)  # noqa: E731
    rows = [("clean", R["clean"], chw(parts["clean"], 3)),
            ("flow_b", R["flow_b"], chw(parts["flow_b"], 2)),
            ("flow_f", R["flow_f"], chw(parts["flow_f"], 2)),
            ("fb", R["fb"], chw(parts["fb"], 64)),
            ("ff", R["ff"], chw(parts["ff"], 64)),
            ("out", R["out"], parts["y"].transpose(0, 3, 1, 2))]
    ok = True
    for name, ref, got in rows:
        assert ref.shape == got.shape, (name, ref.shape, got.shape)
        s = snr(ref, got)
        print("%-7s %-22s %7.1f dB  max %.3g" % (name, ref.shape, s, np.abs(ref - got).max()), flush=True)
        ok &= s > 60
    ref8 = np.round(np.clip(R["out"], 0, 1) * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    same = float((ref8 == out).mean())
    print("uint8 output: %.4f%% of the values the same, max %d; %d frames %dx%d in %.1f s (%s)" % (
        100 * same, np.abs(ref8.astype(np.int16) - out).max(), t, w, h, dt, "GPU" if a.gpu else "CPU"), flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
