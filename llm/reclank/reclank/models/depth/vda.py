"""Video Depth Anything Small (depth-anything/Video-Depth-Anything-Small,
Apache 2.0; its Base and Large are CC-BY-NC-4.0 and not supported):
temporally consistent affine-invariant disparity of a video.

Ported from Video Depth Anything (DepthAnything/Video-Depth-Anything,
video_depth_anything/video_depth.py; Apache License 2.0; Copyright (2025)
Bytedance Ltd. and/or its affiliates). See THIRD_PARTY_NOTICES.md.

    each frame: the shorter side to 518 (lower_bound, multiple of 14,
    OpenCV's INTER_CUBIC on the frame / 255; wider than 1.78:1 shrinks it),
    ImageNet's mean and std -> DINOv2 ViT-S/14 (layers 2, 5, 8, 11) -> the DPT
    head of 64 features with four temporal modules over the window's 32
    frames (dpt.py) -> ReLU.

A video in windows of 32 frames, a step of 22: a window after the first
starts with 10 keyframes of the previous window's input (its frames 0, 12,
24..31) and 22 new frames; its depth is fitted onto the previous by a
least-squares scale and shift on two keyframes, and its first 8 new-to-old
frames crossfade linearly into the previous window's last 8 (the authors'
infer_video_depth). The fit is made at the frames' own size (each map
bilinear, align_corners, as the authors interpolate every frame before the
fit); the maps stay at the network's size.
"""
from __future__ import annotations

import numpy as np

from reclank.encoders.runtime import _Runner
from reclank.models.depth.common import Builder, ViT, prep_forms, run
from reclank.models.depth.dpt import DPT
from reclank.models.depth.ref import resize

INFER_LEN, OVERLAP, INTERP_LEN = 32, 10, 8
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
TAKE = [2, 5, 8, 11]


def input_size(h, w, size=518):
    """(nh, nw) of the network's input for frames of h x w"""
    ratio = max(h, w) / min(h, w)
    if ratio > 1.78:
        size = round(int(size * 1.777 / ratio) / 14) * 14
    s = max(size / h, size / w)

    def mult(x, minv):
        y = int(np.round(x / 14) * 14)
        return int(np.ceil(x / 14) * 14) if y < minv else y
    return mult(s * h, size), mult(s * w, size)


def fit_scale_shift(pred, target):
    """the least-squares scale and shift of pred onto target (float32 sums, the
    authors' compute_scale_and_shift with a mask of ones)"""
    pred, target = pred.astype(np.float32), target.astype(np.float32)
    mask = np.ones_like(pred)
    a00, a01, a11 = np.sum(mask * pred * pred), np.sum(mask * pred), np.sum(mask)
    b0, b1 = np.sum(mask * pred * target), np.sum(mask * target)
    det = a00 * a11 - a01 * a01
    if det == 0:
        return 1, 0
    return (a11 * b0 - a01 * b1) / det, (-a01 * b0 + a00 * b1) / det


class VDASmall:
    """VDA-Small of a directory with video_depth_anything_vits.safetensors (its
    .pth converted by scripts/pt_to_safetensors.py)."""

    def __init__(self, path, gpu=False, cpu_threads=None, size=518):
        from reclank.models.depth.moge import converted
        self.W = W = converted(path, "video_depth_anything_vits")
        get = lambda n: np.asarray(W.get(n), np.float32)  # noqa: E731
        self.gpu, self.cpu_threads, self.size = bool(gpu), cpu_threads, size
        self.vit = ViT(get, "pretrained.", 6, final_eps=1e-6)
        self.head = DPT(get, "head.", temporal=True, relu_out=True)
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)
        self.runner.label = "vda-small"
        self.prep = _Runner(self._build_prep, keep=2, keep_gpu=1)
        self.prep.label = "vda-small/prep"

    def _build(self, key, gpu):
        T, ph, pw = key
        B = Builder(gpu)
        B.array("patches", np.zeros((T * ph * pw, 588), np.float32))
        feats = self.vit.forms(B, "patches", T, ph, pw, TAKE)
        logits, _ = self.head.forms(B, feats, T, ph, pw)
        B.array("depth", B.env[logits])
        return B.finish(self.cpu_threads)

    def _build_prep(self, key, gpu):
        h, w = key
        nh, nw = input_size(h, w, self.size)
        B = Builder(gpu)
        B.array("frame8", np.zeros((h * w * 3 + 3) // 4, np.float32))    # the uint8 frame's bytes
        B.array("frame", np.zeros((h * w, 3), np.float32))
        B.f("enc_unpack_u8", "frame8", "frame", 255.0)                  # frame / 255 (float32)
        B.array("patches", B.env[prep_forms(B, "frame", h, w, nh, nw)])
        return B.finish(self.cpu_threads)

    def prepare(self, frame):
        """the patch rows of an uint8 frame (h, w, 3): OpenCV's INTER_CUBIC of
        the frame / 255 to the network's size (taps of float32 weights, the
        columns first), ImageNet's mean and std"""
        from reclank.models.depth.stereo import as_words
        h, w = frame.shape[:2]
        p, = run(self.prep, (h, w), self.gpu, {"frame8": as_words(frame)}, ["patches"])
        return p

    def window(self, patches, ph, pw):
        """the disparity (T, 14 ph, 14 pw) of a window's frames' patch rows"""
        T = len(patches)
        d, = run(self.runner, (T, ph, pw), self.gpu, {"patches": np.concatenate(patches)}, ["depth"])
        return d.reshape(T, 14 * ph, 14 * pw)

    def video(self, frames, size=None, on_window=None):
        """The aligned disparity of each frame of an iterable of frames, in
        order, as each becomes final (a generator of (index, map at the
        network's size)). size: the frames' (h, w) for the fit (default: the
        first frame's)."""
        it = iter(frames)
        first = next(it, None)
        if first is None:
            return
        h, w = first.shape[:2] if size is None else size
        nh, nw = input_size(first.shape[0], first.shape[1], self.size)
        ph, pw = nh // 14, nw // 14
        # (C order: the fit's float32 sums go in the order of the authors' arrays)
        up = lambda d: np.ascontiguousarray(resize(d[None, :, :, None], h, w, 1)[0, :, :, 0])  # noqa: E731
        src = [self.prepare(first)]          # the prepared frames from index base on
        last = src[0]
        done = False
        n = 1                                # frames read

        def read(upto):
            nonlocal done, n, last
            while not done and n < upto:
                f = next(it, None)
                if f is None:
                    done = True
                    break
                last = self.prepare(f)
                src.append(last)
                n += 1
        step = INFER_LEN - OVERLAP
        pre_input, aligned, ref_align, emitted, k = None, [], None, 0, 0
        base = 0                             # the video index of src[0]
        while True:
            start = k * step
            read(start + INFER_LEN)
            if start >= n:
                break
            first_new = start if pre_input is None else start + OVERLAP
            cur = [] if pre_input is None else [pre_input[i] for i in KEYFRAMES]
            cur += [src[i - base] if i < n else last for i in range(first_new, start + INFER_LEN)]
            depth = self.window(cur, ph, pw)
            if on_window is not None:
                on_window(k)
            if k == 0:
                aligned = [d for d in depth]
                ref_align = [up(depth[i]) for i in KEYFRAMES[:OVERLAP - INTERP_LEN]]
            else:
                al = OVERLAP - INTERP_LEN
                curr = np.concatenate([up(depth[i]) for i in range(al)])
                scale, shift = fit_scale_shift(curr, np.concatenate(ref_align))
                fix = lambda d: np.maximum(d * scale + shift, 0).astype(np.float32)  # noqa: E731
                pre = aligned[-INTERP_LEN:]
                post = [fix(depth[i]) for i in range(al, OVERLAP)]
                wts = [0.0] + [i * (1.0 / (INTERP_LEN - 1)) for i in range(1, INTERP_LEN - 1)] + [1.0]
                aligned[-INTERP_LEN:] = [pre[i] * (1 - wts[i]) + post[i] * wts[i] for i in range(INTERP_LEN)]
                aligned += [fix(depth[i]) for i in range(OVERLAP, INFER_LEN)]
                ref_align = ref_align[:1] + [np.maximum(up(depth[i]) * scale + shift, 0).astype(np.float32)
                                             for i in KEYFRAMES[1:al]]
            pre_input = cur
            # final: all but the last INTERP_LEN (the next window crossfades them)
            keep = INTERP_LEN
            while len(aligned) > keep and emitted < n:
                yield emitted, aligned.pop(0)
                emitted += 1
            # forget the prepared frames no window reads again
            nxt = (k + 1) * step + OVERLAP
            drop = max(0, min(nxt, n) - base)
            del src[:drop]
            base += drop
            k += 1
        while aligned and emitted < n:
            yield emitted, aligned.pop(0)
            emitted += 1
