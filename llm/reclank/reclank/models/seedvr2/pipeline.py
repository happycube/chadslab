"""SeedVR2-3B (ByteDance-Seed/SeedVR2-3B, Apache 2.0): one-step restoration and upscaling of images,
as the official projects/inference_seedvr2_3b.py (cfg 1, one step, the color fix).

Ported from SeedVR (ByteDance-Seed/SeedVR, projects/inference_seedvr2_3b.py and
projects/video_diffusion_sr/infer.py; Apache License 2.0; Copyright (c) 2025 Bytedance Ltd. and/or
its affiliates). See THIRD_PARTY_NOTICES.md.

    the input frames (t, h, w, 3) uint8; the output size: res_h x res_w's area at the input's
        aspect ratio (NaResize "area": scale sqrt(res_h res_w / (h w)), the sides rounded), by
        torch's antialiased bicubic (torchvision's tensor resize: Keys a = -0.5, the weights
        normalized, the width then the height), clamped to [0, 1], center-cropped to multiples of
        16, to [-1, 1]. The model restores this upscale: its cost is that of the output size.
    the VAE's encoder (vae.py), the posterior sampled (mean + exp(logvar / 2) noise, as the
        official), times 0.9152: the latent (t', h, w, 16)
    the DiT's one call at t = 1000 on [noise, latent, 1] with the text pos_emb.pt (dit.py); x0 =
        noise - v (v_lerp at t = T)
    the decoder of x0 / 0.9152; the color fix: the wavelet reconstruction (5 levels of a 3 x 3
        binomial blur, replicate padding, dilations 1 2 4 8 16: the output's high frequencies plus
        the resized input's low ones; our own code of the method), clip, round to uint8

The noise and the posterior's noise: NumPy's normal from the seed (the posterior's first), or
given (the reference's). The weights: the official .pth files converted once to safetensors by
scripts/pt_to_safetensors.py (torch.load with weights_only=True; reclank reads only the
safetensors). On the GPU the DiT's memory is freed before the VAE's decode and the VAE's after
it, unless keep.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from reclank.encoders.runtime import RESIDENCY, gpu_runners
from reclank.formats.safetensors import SafeTensors
from reclank.models.seedvr2.dit import SeedVR2DiT
from reclank.models.seedvr2.vae import SCALE, SeedVR2VAE
from reclank.models.seedvr2.vae_video import SeedVR2VideoVAE


def pad_frames(x):
    """The official cut_videos (sp_size 1) of x (3, t, H, W): t = 1 stays; else the last frame
    repeated to 4k + 1 frames (at least 5)."""
    t = x.shape[1]
    if t == 1:
        return x
    n = 4 - t + 1 if t <= 4 else (0 if (t - 1) % 4 == 0 else 4 - (t - 1) % 4)
    return np.concatenate([x] + [x[:, -1:]] * n, 1) if n else x


def _aa_bicubic_taps(n_in, n_out):
    """torch's _upsample_bicubic2d_aa along one axis (align_corners False, no scale given): each
    output's first input and its normalized weights (n_out, k) (a = -0.5; the support scaled by the
    input/output ratio when it shrinks), the taps past an output's own as weight 0."""
    scale = n_in / n_out
    support = 2.0 * (scale if scale >= 1.0 else 1.0)
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    a = -0.5
    center = scale * (np.arange(n_out) + 0.5)
    xmin = np.maximum((center - support + 0.5).astype(np.int64), 0)
    xsize = np.minimum((center + support + 0.5).astype(np.int64), n_in) - xmin
    k = int(xsize.max())
    j = np.arange(k)[None, :]
    x = np.abs((j + xmin[:, None] - center[:, None] + 0.5) * invscale)
    w = np.where(x < 1.0, ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0,
                 np.where(x < 2.0, (((x - 5.0) * x + 8.0) * x - 4.0) * a, 0.0))
    w = np.where(j < xsize[:, None], w, 0.0)
    tot = w.sum(1, keepdims=True)
    w = np.where(tot != 0.0, w / np.where(tot != 0.0, tot, 1.0), w)
    return xmin, w.astype(np.float32)


def _aa_bicubic_weights(n_in, n_out):
    """The matrix (n_out, n_in) of _aa_bicubic_taps (for the tests)."""
    xmin, w = _aa_bicubic_taps(n_in, n_out)
    M = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        for j in range(w.shape[1]):
            if xmin[i] + j < n_in:
                M[i, xmin[i] + j] += w[i, j]
    return M


def _resize_axis(x, n_out, axis):
    """x resized along axis by the taps, in float32 (each output a sum over its taps in order)."""
    n_in = x.shape[axis]
    xmin, w = _aa_bicubic_taps(n_in, n_out)
    x = np.moveaxis(x, axis, 0)
    out = np.zeros((n_out,) + x.shape[1:], np.float32)
    shape = (n_out,) + (1,) * (x.ndim - 1)
    for j in range(w.shape[1]):
        idx = np.minimum(xmin + j, n_in - 1)
        out += w[:, j].reshape(shape) * x[idx]
    return np.moveaxis(out, 0, axis)


def output_size(h, w, res_h, res_w):
    """The resized size and the crop (multiples of 16) of an h x w input for res_h x res_w."""
    resolution = (res_h * res_w) ** 0.5
    scale = math.sqrt(resolution ** 2 / (h * w))
    rh, rw = round(h * scale), round(w * scale)
    return (rh, rw), (rh - rh % 16, rw - rw % 16)


def transform(frames, res_h, res_w):
    """The official video_transform of frames (t, h, w, 3) uint8: (3, t, H, W) float32 in [-1, 1]."""
    t, h, w, _ = frames.shape
    (rh, rw), (ch, cw) = output_size(h, w, res_h, res_w)
    x = frames.astype(np.float32) / np.float32(255)                    # (t, h, w, 3)
    x = _resize_axis(_resize_axis(x, rw, 2), rh, 1)                     # the width, then the height
    x = np.clip(x, 0.0, 1.0)
    top, left = int(round((rh - ch) / 2.0)), int(round((rw - cw) / 2.0))
    x = x[:, top:top + ch, left:left + cw]
    x = (x - np.float32(0.5)) / np.float32(0.5)
    return np.ascontiguousarray(x.transpose(3, 0, 1, 2), dtype=np.float32)


def wavelet_blur(img, radius):
    """img (c, h, w): the 3 x 3 binomial blur of dilation radius over replicate padding, as its two
    1-D passes ([1 2 1] / 4 in rows, then in columns: the 3 x 3 kernel's values, its sums in another
    order)."""
    _, h, w = img.shape
    p = np.pad(img, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    r = np.float32(0.25) * p[:, :, :w] + np.float32(0.5) * p[:, :, radius:radius + w] + \
        np.float32(0.25) * p[:, :, 2 * radius:2 * radius + w]
    p = np.pad(r, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    return (np.float32(0.25) * p[:, :h] + np.float32(0.5) * p[:, radius:radius + h] +
            np.float32(0.25) * p[:, 2 * radius:2 * radius + h]).astype(np.float32)


def wavelet_reconstruction(content, style, levels=5):
    """content's high frequencies plus style's low ones, each (c, h, w)."""
    high = np.zeros_like(content)
    img = content
    for i in range(levels):
        low = wavelet_blur(img, 2 ** i)
        high += img - low
        img = low
    img = style
    for i in range(levels):
        img = wavelet_blur(img, 2 ** i)
    return (high + img).astype(np.float32)


# the colour fix of a video's frames on threads of their own (NumPy lets the GIL go in its operations on whole
# frames), each on the CPUs of the start (RECLANK_START_CPUS: a new thread takes the CPUs of the main thread,
# which is pinned to one). Not processes: a fork copies the host memory registered with CUDA (the 7B's streamed
# blocks: its pinned pages are copied at the fork). RECLANK_SEEDVR2_COLOR_PROCS (0: up to 16 of those CPUs;
# 1: here)
COLOR_PROCS = int(os.environ.get("RECLANK_SEEDVR2_COLOR_PROCS", "0"))
_COLOR = None           # (dec, x, fix) of the frames the workers fix


def _start_cpus():
    c = os.environ.get("RECLANK_START_CPUS")
    return {int(v) for v in c.split(",")} if c else os.sched_getaffinity(0)


def _color_worker():
    os.sched_setaffinity(0, _start_cpus())


def _color_frame(f):
    """Frame f of _COLOR: the colour fix (or none) to uint8 (H, W, 3)."""
    dec, x, fix = _COLOR
    fx = wavelet_reconstruction(dec[:, f], x[:, f]) if fix else dec[:, f]
    return np.round((np.clip(fx, -1, 1) * np.float32(0.5) + np.float32(0.5)) *
                    np.float32(255)).astype(np.uint8).transpose(1, 2, 0)


def color_frames(dec, x, t, fix=True):
    """The first t frames of dec (3, T, H, W) fixed against x's (wavelet_reconstruction; fix False: none), as
    uint8 (t, H, W, 3): each frame's bits those of the one-frame computation, on a pool of threads."""
    global _COLOR
    out = np.empty((t,) + dec.shape[2:] + (3,), np.uint8)
    procs = COLOR_PROCS or min(16, len(_start_cpus()))
    _COLOR = (dec, x, fix)
    try:
        if procs <= 1 or t < 4:
            for f in range(t):
                out[f] = _color_frame(f)
        else:
            from concurrent.futures import ThreadPoolExecutor

            def one(f):
                out[f] = _color_frame(f)
            with ThreadPoolExecutor(min(procs, t), initializer=_color_worker) as pool:
                list(pool.map(one, range(t)))
    finally:
        _COLOR = None
    return out


class SeedVR2:
    """A SeedVR2 directory (the .safetensors of pt_to_safetensors.py: seedvr2_ema_3b or seedvr2_ema_7b,
    ema_vae, pos_emb). variant: "3b" or "7b" (None: the one whose checkpoint is there; the 3B when
    both). parts: "dit", "vae" or "all"; f20: the DiT's matrices in F20 (default: on the GPU);
    stream: the DiT's blocks streamed (None: the 7B on the GPU)."""

    def __init__(self, path, gpu=False, cpu_threads=None, parts=("all",), f20=None, variant=None, stream=None):
        self.path, self.gpu = path, bool(gpu)
        parts = set(parts)
        every = "all" in parts
        has = lambda name: os.path.exists(os.path.join(path, name + ".safetensors"))  # noqa: E731
        if variant is None:
            variant = "3b" if has("seedvr2_ema_3b") or not has("seedvr2_ema_7b") else "7b"
        self.variant = variant
        st = lambda name: SafeTensors(os.path.join(path, name + ".safetensors"))  # noqa: E731
        if not has("pos_emb"):
            raise FileNotFoundError("%s has no pos_emb.safetensors: the text embedding (pos_emb.pt of the 3B's "
                                    "repository or the official code's) converted by pt_to_safetensors.py --out DIR" % path)
        self.txt = np.ascontiguousarray(st("pos_emb").get("pos_emb"), dtype=np.float32)
        self.dit = SeedVR2DiT(st("seedvr2_ema_" + variant), self.txt, gpu=gpu, cpu_threads=cpu_threads, f20=f20,
                              variant=variant, stream=stream) if every or "dit" in parts else None
        self.vae = SeedVR2VAE(st("ema_vae"), gpu=gpu, cpu_threads=cpu_threads) \
            if every or "vae" in parts else None
        self.vvae = None        # the video path's VAE, made at the first video

    def video_vae(self):
        if self.vvae is None:
            self.vvae = SeedVR2VideoVAE(self.vae.W, gpu=self.gpu, cpu_threads=self.vae.cpu_threads, image=self.vae)
        return self.vvae

    def release(self, which=("dit", "vae", "vvae")):
        for name in which:
            part = getattr(self, name)
            if part is not None and self.gpu:
                part.runner.release(keep_host=True)
                if name == "vae":
                    part.enc_runner.release(keep_host=True)

    def close(self):
        """Free the parts' GPU memory and the DiT's registered host memory (the model is dropped
        after)."""
        if self.gpu:
            self.release()
            if self.dit is not None:
                self.dit.close()

    def runners(self):
        """The GPU runners of the parts (name: _Runner)."""
        out = gpu_runners(self.gpu, dit=self.dit, vae=self.vae, vvae=self.vvae)
        if self.gpu and self.vae is not None:
            out["vae_enc"] = self.vae.enc_runner
        return out

    def restore(self, frames, res=None, scale=2.0, seed=0, noise=None, pnoise=None, color_fix="wavelet",
                keep=False, parts=None, progress=None):
        """_restore in a RESIDENCY scope of the parts (the video VAE made first for a video)."""
        if frames.shape[0] > 1:
            self.video_vae()
        with RESIDENCY.use(*self.runners().values()):
            return self._restore(frames, res, scale, seed, noise, pnoise, color_fix, keep, parts, progress)

    def _restore(self, frames, res=None, scale=2.0, seed=0, noise=None, pnoise=None, color_fix="wavelet",
                 keep=False, parts=None, progress=None):
        """The restored frames (t, H, W, 3) uint8 of frames (t, h, w, 3) uint8 (an image: t = 1), at
        res (res_h, res_w) or scale x the input. noise, pnoise: the DiT's (t', h, w, 16) and the
        posterior's (16, t', h, w) in place of the seed's. parts: a dict that gets the stages
        (x, mean, logvar, latent, v, x0, dec, fixed). progress(stage, done, total)."""
        t, h, w, _ = frames.shape
        rh, rw = res if res is not None else (round(h * scale), round(w * scale))
        clock = [time.time()]
        times = {}

        def lap(stage):
            now = time.time()
            times[stage] = now - clock[0]
            clock[0] = now
        x = transform(frames, rh, rw)                                     # (3, t, H, W)
        xp = pad_frames(x)
        note = (lambda stage: (lambda i, n: progress(stage, i, n))) if progress is not None else (lambda stage: None)
        if t == 1:
            mean, logvar = self.vae.encode(x[:, 0])
            mean, logvar = mean[:, None], logvar[:, None]                   # (16, 1, h, w)
            if self.gpu and not keep:
                self.vae.enc_runner.release(keep_host=True)
        else:
            mean, logvar = self.video_vae().encode(xp, progress=note("encode"))
        lap("encode")
        rng = np.random.default_rng(seed)
        if pnoise is None:
            pnoise = rng.standard_normal(mean.shape).astype(np.float32)
        z = SeedVR2VAE.sample(mean, logvar, pnoise)
        latent = (z.transpose(1, 2, 3, 0) * np.float32(SCALE)).astype(np.float32)     # (t', h, w, 16)
        if noise is None:
            noise = rng.standard_normal(latent.shape).astype(np.float32)
        vid = np.concatenate([noise, latent, np.ones(latent.shape[:-1] + (1,), np.float32)], -1)
        if self.gpu and t > 1 and not keep:
            self.release(("vvae",))
        v = self.dit.velocity(vid)
        x0 = (noise - v).astype(np.float32)
        lap("dit")
        if self.gpu and not keep:
            self.release(("dit",))
        zd = (x0 / np.float32(SCALE)).transpose(3, 0, 1, 2).astype(np.float32)          # (16, t', h, w)
        if t == 1:
            dec = self.vae.decode(zd[:, 0])[:, None]
        else:
            dec = self.video_vae().decode(zd, progress=note("decode"))[:, :t]
        lap("decode")
        if self.gpu and not keep:
            self.release(("vae", "vvae"))
        if parts is None and t > 1:
            # (a video: the frames fixed in the processes of color_frames, the same bits)
            out = color_frames(dec, x, t, color_fix == "wavelet")
            lap("color")
            self.times = times
            return out
        fixed = np.stack([wavelet_reconstruction(dec[:, f], x[:, f]) for f in range(t)], 1) \
            if color_fix == "wavelet" else dec
        lap("color")
        self.times = times      # (the stages' seconds of the last restore)
        if parts is not None:
            parts.update(x=x, mean=mean, logvar=logvar, latent=latent, v=v, x0=x0, dec=dec.transpose(1, 0, 2, 3),
                         fixed=fixed.transpose(1, 0, 2, 3), times=times)
        out = np.round((np.clip(fixed, -1, 1) * np.float32(0.5) + np.float32(0.5)) * np.float32(255))
        return out.astype(np.uint8).transpose(1, 2, 3, 0)

    # ---- a long video (a scene) without batches -----------------------------------------------------

    def restore_long(self, frames, res, seed=0, chunk=30, margin=3, color_fix="wavelet", progress=None):
        """restore of a video of any length (t, h, w, 3) uint8 at res (res_h, res_w), as one: the VAE's
        encoder and decoder each one causal pass over all its frames (their caches carry on), the DiT on
        chunks of `chunk` latent frames (long_chunks: each keeps its central latents, the rest its context;
        their starts on the grid of the DiT's time windows, so a kept latent sees the windows of one pass
        over all), the noises of each latent frame its own (the seed and the frame's index: a frame's in
        every chunk that has it). Return the restored frames (t, H, W, 3) uint8; self.times the stages'.
        progress(stage, done, total)."""
        self.video_vae()
        with RESIDENCY.use(*self.runners().values()):
            return self._restore_long(frames, res, seed, chunk, margin, color_fix, progress)

    def _restore_long(self, frames, res, seed, chunk, margin, color_fix, progress):
        t = frames.shape[0]
        assert t > 1, "restore_long: a video"
        clock, times = [time.time()], {}

        def lap(stage):
            now = time.time()
            times[stage] = now - clock[0]
            clock[0] = now
        note = (lambda stage: (lambda i, n: progress(stage, i, n))) if progress is not None else (lambda stage: None)
        n = max((t - 1 + 3) // 4 + 1, 2)                                 # the latent frames of t
        plan, n_pad = long_chunks(n, chunk, margin)
        x = transform(frames, *res)                                       # (3, t, H, W)
        T = 4 * (n_pad - 1) + 1
        xp = np.concatenate([x] + [x[:, -1:]] * (T - t), 1) if T > t else x
        mean, logvar = self.video_vae().encode(xp, progress=note("encode"))
        del xp
        lap("encode")
        if self.gpu:
            self.release(("vvae",))
        shape = mean.shape[2:]
        rngs = [np.random.default_rng([seed, j]) for j in range(n_pad)]
        pnoise = np.stack([r.standard_normal((16,) + shape).astype(np.float32) for r in rngs], 1)
        noise = np.stack([r.standard_normal(shape + (16,)).astype(np.float32) for r in rngs], 0)
        z = SeedVR2VAE.sample(mean, logvar, pnoise)
        latent = (z.transpose(1, 2, 3, 0) * np.float32(SCALE)).astype(np.float32)     # (n_pad, h, w, 16)
        vid = np.concatenate([noise, latent, np.ones(latent.shape[:-1] + (1,), np.float32)], -1)
        x0 = np.empty_like(latent)
        for k, (a, keep0, keep1) in enumerate(plan):
            v = self.dit.velocity(vid[a:a + min(chunk, n_pad - a)])
            x0[keep0:keep1] = noise[keep0:keep1] - v[keep0 - a:keep1 - a]
            if progress is not None:
                progress("dit", k + 1, len(plan))
        lap("dit")
        if self.gpu:
            self.release(("dit",))
        zd = (x0 / np.float32(SCALE)).transpose(3, 0, 1, 2).astype(np.float32)        # (16, n_pad, h, w)
        dec = self.video_vae().decode(zd, progress=note("decode"))
        lap("decode")
        if self.gpu:
            self.release(("vae", "vvae"))
        out = color_frames(dec, x, t, color_fix == "wavelet")
        lap("color")
        self.times = times
        self.chunks = plan
        return out


def long_chunks(n, chunk, margin):
    """The DiT's chunks of n latent frames: ([(start, keep0, keep1)], the padded count). n <= chunk:
    one chunk of all. Else chunks of `chunk` latents at starts k core (core = chunk - 2 margin, a
    multiple of the DiT's time window of a chunk: window_slices' ceil(min(chunk, 30) / 4)), each
    keeping [start + margin, start + chunk - margin) (the first from 0, the last to the end); the
    latents padded so that the last chunk is whole (its frames the last frame repeated)."""
    if n <= chunk:
        return [(0, 0, n)], n
    wt = math.ceil(min(chunk, 30) / 4)
    core = chunk - 2 * margin
    assert core > 0 and core % wt == 0, "chunk %d, margin %d: core %d not a multiple of the window %d" % (
        chunk, margin, core, wt)
    k = math.ceil((n - chunk) / core) + 1
    n_pad = (k - 1) * core + chunk
    plan = []
    for i in range(k):
        a = i * core
        plan.append((a, 0 if i == 0 else a + margin, n_pad if i == k - 1 else a + chunk - margin))
    return plan, n_pad
