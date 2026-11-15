"""Real-ESRGAN references for scripts/check_realesrgan.py: the networks of basicsr (RRDBNet) and
realesrgan (SRVGGNetCompact) written out in torch float32 on the CPU (none of their code
imported), and RealESRGANer.enhance's pre- and post-processing of inference_realesrgan.py's
defaults (tile 0, pre_pad 0; the official half precision on a GPU is not used: float32).

Written from Real-ESRGAN (xinntao/Real-ESRGAN, realesrgan/archs/srvgg_arch.py,
realesrgan/utils.py, inference_realesrgan.py; BSD 3-Clause License; Copyright (c) 2021, Xintao
Wang) and BasicSR (XPixelGroup/BasicSR, basicsr/archs/rrdbnet_arch.py; Apache License 2.0;
Copyright 2018-2022 BasicSR Authors). See THIRD_PARTY_NOTICES.md.

    realesrgan_reference.py WEIGHTS.pth IMAGE out.npz [--arch rrdb|srvgg] [--scale 4]
        [--blocks 23] [--convs 32] [--dni WDN.pth --strength 0.5] [--threads 10] [--png out.png]

WEIGHTS: params_ema (RRDBNet) or params (SRVGGNetCompact), loaded with weights_only=True. --dni:
realesr-general-x4v3's denoise strength (inference_realesrgan.py): the weights strength x
WEIGHTS + (1 - strength) x WDN.

    RRDBNet (num_feat 64, grow 32): x2: the input pixel-unshuffled by 2 (12 channels); conv_first;
        blocks of RRDB (3 residual dense blocks: 5 convs over the concatenation of the input and
        the earlier convs' outputs, LeakyReLU 0.2 but the last, x5 0.2 + x; then 0.2 + x);
        conv_body, + the features; two nearest x2 upsamplings each with a conv and LeakyReLU;
        conv_hr, LeakyReLU, conv_last
    SRVGGNetCompact (64 features, PReLU): conv, PReLU, convs x (conv, PReLU), the last conv to 3
        x scale^2 channels, pixel shuffle, + the nearest upsampling of the input
    enhance: RGB / 255; the x2 model's input reflect-padded to a multiple of 2 at the bottom and
        right (removed after); the output clamped to [0, 1], x 255, rounded to uint8

out.npz: "lq" (h, w, 3) uint8, "x" the network's input (3, h', w'), "y" its output (3, s h', s w')
(before the clamp), "out" (s h, s w, 3) uint8."""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("image")
ap.add_argument("out")
ap.add_argument("--arch", default="rrdb")
ap.add_argument("--scale", type=int, default=4)
ap.add_argument("--blocks", type=int, default=23)
ap.add_argument("--convs", type=int, default=32)
ap.add_argument("--dni")
ap.add_argument("--strength", type=float, default=0.5)
ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--png")
a = ap.parse_args()
torch.set_num_threads(a.threads)


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("params_ema", "params"):
        if k in sd:
            return sd[k]
    return sd


W = load(a.weights)
if a.dni:
    Wb = load(a.dni)
    W = {k: a.strength * v + (1 - a.strength) * Wb[k] for k, v in W.items()}


def conv(x, n):
    return F.conv2d(x, W[n + ".weight"], W[n + ".bias"], padding=1)


def lrelu(x):
    return F.leaky_relu(x, 0.2)


def rdb(x, p):
    x1 = lrelu(conv(x, p + "conv1"))
    x2 = lrelu(conv(torch.cat((x, x1), 1), p + "conv2"))
    x3 = lrelu(conv(torch.cat((x, x1, x2), 1), p + "conv3"))
    x4 = lrelu(conv(torch.cat((x, x1, x2, x3), 1), p + "conv4"))
    x5 = conv(torch.cat((x, x1, x2, x3, x4), 1), p + "conv5")
    return x5 * 0.2 + x


def rrdbnet(x):
    if a.scale == 2:
        x = F.pixel_unshuffle(x, 2)
    elif a.scale == 1:
        x = F.pixel_unshuffle(x, 4)
    feat = conv(x, "conv_first")
    body = feat
    for i in range(a.blocks):
        p = "body.%d." % i
        out = rdb(rdb(rdb(body, p + "rdb1."), p + "rdb2."), p + "rdb3.")
        body = out * 0.2 + body
    feat = feat + conv(body, "conv_body")
    feat = lrelu(conv(F.interpolate(feat, scale_factor=2, mode="nearest"), "conv_up1"))
    feat = lrelu(conv(F.interpolate(feat, scale_factor=2, mode="nearest"), "conv_up2"))
    return conv(lrelu(conv(feat, "conv_hr")), "conv_last")


def srvgg(x):
    out = x
    n = 0
    while "body.%d.weight" % n in W:
        w = W["body.%d.weight" % n]
        if w.ndim == 4:
            out = F.conv2d(out, w, W["body.%d.bias" % n], padding=1)
        else:
            out = F.prelu(out, w)
        n += 1
    out = F.pixel_shuffle(out, a.scale)
    return out + F.interpolate(x, scale_factor=a.scale, mode="nearest")


def main():
    from PIL import Image
    lq = np.asarray(Image.open(a.image).convert("RGB"))
    with torch.no_grad():
        img = torch.from_numpy(lq.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
        h, w = img.shape[-2:]
        mod = 2 if a.scale == 2 else (4 if a.scale == 1 else None)
        ph = pw = 0
        if mod is not None:
            ph, pw = (mod - h % mod) % mod, (mod - w % mod) % mod
            img = F.pad(img, (0, pw, 0, ph), "reflect")
        y = (rrdbnet if a.arch == "rrdb" else srvgg)(img)
        y = y[:, :, :h * a.scale if a.arch == "rrdb" else h * a.scale, :w * a.scale]
        out = (y[0].clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    np.savez(a.out, lq=lq, x=img[0].numpy(), y=y[0].numpy(), out=out)
    if a.png:
        Image.fromarray(out).save(a.png)
    print("saved", a.out, out.shape)


if __name__ == "__main__":
    main()
