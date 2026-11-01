"""Ported from Practical-RIFE (the v4.25 model's train_log/IFNet_HDv3.py, model/warplayer.py and
inference_video.py; MIT License; Copyright (c) 2021 hzwer). See THIRD_PARTY_NOTICES.md.

RIFE references for scripts/check_rife.py: a torch-only float32 reference of Practical-RIFE 4.25 on
the CPU, written from the authors' code (read, not run: 4.25/IFNet_HDv3.py, model/warplayer.py,
inference_video.py of hzwer/Practical-RIFE, MIT): the Head, the five IFBlocks, the warp
(grid_sample bilinear, border, align_corners=True at linspace(-1, 1) + flow / ((n - 1) / 2)), the
sigmoid merge; the frames RGB / 255, zero-padded at the right and bottom to a multiple of
max(pad, pad / scale) (the authors' script: 128; the port: 64), cropped back. The weights from
flownet.safetensors (scripts/pt_to_safetensors.py of the authors' flownet.pkl).

    rife_reference.py MODEL_DIR FRAMES_DIR out.npz --pairs 1:3,3:5 [--times 0.5] [--pad 64]
                      [--scale 1] [--dump]

FRAMES_DIR: fNNN.png frames (1-based). out.npz: "out_I_J_T" (h, w, 3) float32 for each pair and t;
with --dump, of the first pair and t: "x0", "x1" (the padded frames, (3, Hp, Wp)), "f0", "f1" (the
heads), "block{k}_tmp" (13 channels at 1/s), "block{k}_flow" and the final "mask". Run in a venv
with torch, safetensors and Pillow (the generation reference venv)."""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_grad_enabled(False)


def load_weights(path):
    from safetensors.numpy import load_file
    if not path.endswith(".safetensors"):
        import os
        path = os.path.join(path, "flownet.safetensors") if os.path.exists(os.path.join(path, "flownet.safetensors")) \
            else os.path.join(path, "4.25", "flownet.safetensors")
    return {k[len("module."):]: torch.from_numpy(v).float() for k, v in load_file(path).items()
            if k.startswith("module.")}


def lrelu(x):
    return F.leaky_relu(x, 0.2)


def conv(W, n, x, stride=1):
    return F.conv2d(x, W[n + ".weight"], W[n + ".bias"], stride=stride, padding=1)


def deconv(W, n, x):
    return F.conv_transpose2d(x, W[n + ".weight"], W[n + ".bias"], stride=2, padding=1)


def head(W, x):
    x = lrelu(conv(W, "encode.cnn0", x, 2))
    x = lrelu(conv(W, "encode.cnn1", x))
    x = lrelu(conv(W, "encode.cnn2", x))
    return deconv(W, "encode.cnn3", x)


def ifblock(W, b, x, flow, scale, taps=None):
    x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
    if flow is not None:
        flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = torch.cat((x, flow), 1)
    feat = lrelu(conv(W, b + ".conv0.0.0", x, 2))
    feat = lrelu(conv(W, b + ".conv0.1.0", feat, 2))
    for i in range(8):
        n = b + ".convblock.%d" % i
        feat = lrelu(conv(W, n + ".conv", feat) * W[n + ".beta"] + feat)
    tmp = F.pixel_shuffle(deconv(W, b + ".lastconv.0", feat), 2)
    if taps is not None:
        taps[b + "_tmp"] = tmp
    tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
    return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


_grid = {}


def warp(x, flow):
    k = tuple(flow.shape)
    if k not in _grid:
        h = torch.linspace(-1.0, 1.0, flow.shape[3]).view(1, 1, 1, flow.shape[3]).expand(flow.shape[0], -1,
                                                                                           flow.shape[2], -1)
        v = torch.linspace(-1.0, 1.0, flow.shape[2]).view(1, 1, flow.shape[2], 1).expand(flow.shape[0], -1, -1,
                                                                                          flow.shape[3])
        _grid[k] = torch.cat([h, v], 1)
    flow = torch.cat([flow[:, 0:1] / ((x.shape[3] - 1.0) / 2.0), flow[:, 1:2] / ((x.shape[2] - 1.0) / 2.0)], 1)
    g = (_grid[k] + flow).permute(0, 2, 3, 1)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)


def ifnet(W, img0, img1, f0, f1, t, scale=1.0, taps=None):
    scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    timestep = (img0[:, :1].clone() * 0 + 1) * t
    flow = mask = feat = None
    w0, w1 = img0, img1
    for i in range(5):
        b = "block%d" % i
        if flow is None:
            flow, mask, feat = ifblock(W, b, torch.cat((img0, img1, f0, f1, timestep), 1), None, scale_list[i], taps)
        else:
            wf0, wf1 = warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4])
            fd, mask, feat = ifblock(W, b, torch.cat((w0, w1, wf0, wf1, timestep, mask, feat), 1), flow,
                                     scale_list[i], taps)
            flow = flow + fd
        w0, w1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        if taps is not None:
            taps[b + "_flow"] = flow
    if taps is not None:
        taps["mask"] = mask
    m = torch.sigmoid(mask)
    return w0 * m + w1 * (1 - m)


def pad_to(x, mult):
    h, w = x.shape[2:]
    ph, pw = ((h - 1) // mult + 1) * mult, ((w - 1) // mult + 1) * mult
    return F.pad(x, (0, pw - w, 0, ph - h))


def load_frame(path):
    a = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    return torch.from_numpy(a.transpose(2, 0, 1).copy())[None].float() / 255.


class Rife:
    """the frames at t between two frames; each frame's Head once"""

    def __init__(self, path, scale=1.0, pad=64):
        self.W = load_weights(path)
        self.scale = scale
        self.mult = max(pad, int(pad / scale))
        self.heads = {}

    def prep(self, key, img):
        if key not in self.heads:
            x = pad_to(img, self.mult)
            self.heads[key] = (x, head(self.W, x))
        return self.heads[key]

    def __call__(self, k0, img0, k1, img1, t, taps=None):
        h, w = img0.shape[2:]
        x0, f0 = self.prep(k0, img0)
        x1, f1 = self.prep(k1, img1)
        if taps is not None:
            taps.update(x0=x0, x1=x1, f0=f0, f1=f1)
        return ifnet(self.W, x0, x1, f0, f1, t, self.scale, taps)[:, :, :h, :w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("frames", help="a directory of fNNN.png frames")
    ap.add_argument("out")
    ap.add_argument("--pairs", required=True, help="I:J,... frame numbers (1-based)")
    ap.add_argument("--times", default="0.5")
    ap.add_argument("--pad", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    R = Rife(a.model, a.scale, a.pad)
    times = [float(v) for v in a.times.split(",")]
    res, first, t0 = {}, True, time.time()
    for p in a.pairs.split(","):
        i, j = (int(v) for v in p.split(":"))
        im0, im1 = load_frame("%s/f%03d.png" % (a.frames, i)), load_frame("%s/f%03d.png" % (a.frames, j))
        for t in times:
            taps = {} if (a.dump and first) else None
            res["out_%d_%d_%g" % (i, j, t)] = R(i, im0, j, im1, t, taps)[0].numpy().transpose(1, 2, 0)
            if taps is not None:
                res.update({k: v[0].numpy() for k, v in taps.items()})
                res["dump_key"] = np.array([i, j, t])
            first = False
    np.savez(a.out, **res)
    print("rife_reference %s: %d pairs x %d times in %.1f s (pad %d)" % (a.frames, len(a.pairs.split(",")),
                                                                      len(times), time.time() - t0, R.mult))


if __name__ == "__main__":
    main()
