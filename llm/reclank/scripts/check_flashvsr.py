#!/usr/bin/env python3
"""FlashVSR (models/flashvsr) against the official Tiny-Long pipeline's dumps on real frames: each part on the
official's own inputs of each chunk (lq: the LQ projector's condition; dit: the DiT's output and every layer's block
masks over its KV cache; tc: the TCDecoder's frames) and the whole pipeline on the official's LQ video and noise
(all: the frames after the colour fix and the uint8 output), each as its SNR against the official's (dB) and its
largest difference.

The dump (DUMP.npz: lq, noise, video and each chunk's lq_lat, x_in, v, x0, dec, fixed, mask) is the official's
(OpenImagingLab/FlashVSR v1.1, float32 on the CPU with a dense float32 stand-in for Block-Sparse-Attention), run in a
sandbox on real frames.

    check_flashvsr.py MODEL_DIR DUMP.npz [--gpu] [all] [lq] [dit] [tc]
"""
import sys
import time

import numpy as np



def snr(ref, x):
    ref, x = np.asarray(ref, np.float64), np.asarray(x, np.float64)
    e = ((x - ref) ** 2).sum()
    return float("inf") if e == 0 else 10 * np.log10((ref ** 2).sum() / e)


def main():
    gpu = "--gpu" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    M, REF = args[:2]
    parts = args[2:] or ["all"]
    R = np.load(REF)
    from reclank.formats.safetensors import SafeTensors
    if "all" in parts:
        from reclank.models.flashvsr.pipeline import FlashVSR
        m = FlashVSR(M, gpu=gpu)
        pr = {}
        t0 = time.time()
        out = m.upscale_video(None, R["lq"].shape[2:], noise=R["noise"], lq=R["lq"], parts=pr)
        v, ref = pr["video"], R["video"]
        ref8 = ((ref + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 3, 0)
        print("all: %s %.1f dB max %.3g; uint8 %.3f%% the same, max %d (%.1f s %s)" % (
            out.shape, snr(ref, v), np.abs(ref - v).max(), 100 * (out == ref8).mean(),
            np.abs(out.astype(np.int16) - ref8).max(), time.time() - t0, {k: round(x, 1) for k, x in m.times.items()}),
            flush=True)
    if "lq" in parts:
        from reclank.models.flashvsr.lq_proj import LQProj
        S = SafeTensors(M + "/LQ_proj_in.safetensors")
        pj = LQProj({k: np.array(S.get(k), np.float32) for k in S.names()}, gpu=gpu)
        lq = R["lq"]
        pj.reset()
        clips = [[(0, 1)] + [(4 * i - 3, 4 * i + 1) for i in range(1, 7)]] + \
            [[(8 * k + 17 + 4 * i, 8 * k + 21 + 4 * i) for i in range(2)] for k in (1, 2)]
        for k, cl in enumerate(clips):
            t0 = time.time()
            outs = [pj.push(lq[:, a:b]) for a, b in cl]
            y = np.concatenate([o for o in outs if o is not None])
            ref = R["lq_lat_%d" % k]
            print("lq chunk %d: %s %.1f dB max %.3g (%.1f s)" % (k, y.shape, snr(ref, y), np.abs(ref - y).max(),
                                                                 time.time() - t0), flush=True)
    if "dit" in parts:
        from reclank.models.flashvsr.dit import FlashDiT
        t0 = time.time()
        dit = FlashDiT(M, gpu=gpu)
        print("dit loaded in %.0f s" % (time.time() - t0), flush=True)
        hh, ww = R["x_in_0"].shape[2] // 2, R["x_in_0"].shape[3] // 2
        dit.start(hh, ww, out_pixels=R["lq"].shape[2] * R["lq"].shape[3])
        for k in range(3):
            t0 = time.time()
            pr = {}
            v = dit.chunk(R["x_in_%d" % k], R["lq_lat_%d" % k], parts=pr)
            ref = R["v_%d" % k]
            m, rm = pr["masks"], R["mask_%d" % k]
            print("dit chunk %d: %s %.1f dB max %.3g; masks %s %s, %d of %d differ (%.1f s)" % (
                k, v.shape, snr(ref, v), np.abs(ref - v).max(), m.shape, rm.shape, int((m != rm).sum()), m.size,
                time.time() - t0), flush=True)
    if "tc" in parts:
        from reclank.models.flashvsr.tcdecoder import TCDecoder
        S = SafeTensors(M + "/TCDecoder.safetensors")
        dec = TCDecoder({k: np.array(S.get(k), np.float32) for k in S.names()}, gpu=gpu)
        lq = R["lq"]
        h, w = R["x0_0"].shape[2:]
        dec.reset(h, w)
        bounds = [(0, 21), (21, 29), (29, 37)]
        for k, (a, b) in enumerate(bounds):
            t0 = time.time()
            y = dec.decode(R["x0_%d" % k], lq[:, a:b])
            ref = R["dec_%d" % k] * 2 - 1
            print("tc chunk %d: %s %.1f dB max %.3g (%.1f s)" % (k, y.shape, snr(ref, y), np.abs(ref - y).max(),
                                                                 time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
