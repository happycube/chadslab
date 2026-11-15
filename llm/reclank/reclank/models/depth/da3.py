"""Depth Anything 3's monocular model (DA3MONO-LARGE): DINOv2 ViT-L/14
(layers 4, 11, 17, 23, the final norm eps 1e-5) and DA3's DPT head with its sky
branch. Relative depth (exp of the logits) at the processed size: the longest
side to process_res (504), each side to the nearest multiple of 14
(upper_bound_resize: OpenCV's INTER_AREA down, INTER_CUBIC up), ImageNet's
mean and std. The sky (its branch below 0.3) is set to the 99th percentile of
the other pixels' depth, as the authors' API (which takes it of a random
sample of 100k pixels; here all of them).

Ported from Depth Anything 3 (ByteDance-Seed/Depth-Anything-3,
src/depth_anything_3/model/da3.py, api.py, utils/io/input_processor.py; Apache
License 2.0; Copyright (c) 2025 ByteDance Ltd. and/or its affiliates). See
THIRD_PARTY_NOTICES.md."""
from __future__ import annotations

import os

import numpy as np

from reclank.encoders.runtime import _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.depth.common import Builder, ViT, cv_resize, normalize, patch_rows, run
from reclank.models.depth.dpt import DPT

TAKE = [4, 11, 17, 23]


def processed_size(h, w, res):
    """(H, W) of DA3's upper_bound_resize to res and the nearest multiple of 14"""
    s = res / max(h, w)
    nw, nh = (w, h) if max(h, w) == res else (max(1, int(round(w * s))), max(1, int(round(h * s))))

    def near(x):
        d = (x // 14) * 14
        return d + 14 if abs(d + 14 - x) <= abs(x - d) else d
    return (nh, nw), (max(14, near(nh)), max(14, near(nw)))


def preprocess(img, res=504):
    """The uint8 image (H, W, 3) DA3 runs at, of an uint8 image."""
    h, w = img.shape[:2]
    (nh, nw), (mh, mw) = processed_size(h, w, res)
    a = img if (nh, nw) == (h, w) else cv_resize(img, nw, nh, "cubic" if res > max(h, w) else "area")
    if (mh, mw) != (nh, nw):
        a = cv_resize(a, mw, mh, "cubic" if (mw > nw or mh > nh) else "area")
    return a


class DA3Mono:
    """DA3MONO-LARGE of a model directory (model.safetensors). depth(img) gives
    the relative depth, the sky probability and the processed image."""

    def __init__(self, path, gpu=False, cpu_threads=None, res=504):
        self.W = SafeTensors(os.path.join(path, "model.safetensors"))
        get = lambda n: self.W.get("model." + n)  # noqa: E731
        self.gpu, self.cpu_threads, self.res = bool(gpu), cpu_threads, res
        self.vit = ViT(get, "backbone.pretrained.", 16, final_eps=1e-5)
        self.head = DPT(get, "head.", sky=True)
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)
        self.runner.label = "da3mono"

    def _build(self, key, gpu):
        ph, pw = key
        B = Builder(gpu)
        B.array("patches", np.zeros((ph * pw, 588), np.float32))
        feats = self.vit.forms(B, "patches", 1, ph, pw, TAKE)
        logits, sky = self.head.forms(B, feats, 1, ph, pw)
        B.array("logits", B.env[logits])
        B.array("sky", B.env[sky])
        return B.finish(self.cpu_threads)

    def logits(self, a):
        """The depth and sky logits (H, W) of a processed uint8 image (H, W, 3)."""
        H, W = a.shape[:2]
        ph, pw = H // 14, W // 14
        lg, sky = run(self.runner, (ph, pw), self.gpu, {"patches": patch_rows(normalize(a), ph, pw)},
                      ["logits", "sky"])
        return lg.reshape(H, W), sky.reshape(H, W)

    def depth(self, img, res=None):
        """dict(depth (relative, the sky at the far end), sky (its branch),
        valid (not sky), image (the processed image)) of an uint8 image."""
        a = preprocess(img, res or self.res)
        lg, sky = self.logits(a)
        d = np.exp(lg.astype(np.float64)).astype(np.float32)
        valid = sky < 0.3
        if valid.sum() > 10 and (~valid).sum() > 10:
            d[~valid] = np.float32(np.quantile(d[valid], 0.99))
        return dict(depth=d, sky=sky, valid=valid, image=a)
