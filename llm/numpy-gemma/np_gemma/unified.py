"""The image and audio input of the Gemma 4 12B (the unified model).

The 12B has no vision encoder and no audio encoder (MULTIMODAL_PLAN.md,
section 3.1). An image becomes patches of 48 x 48 pixels, and each patch
becomes one soft token:

    LayerNorm (6912) -> linear to 3840 -> LayerNorm -> + the x row and the y
    row of the position tables -> LayerNorm -> RMS norm (no weight) ->
    mm.input_projection

Audio at 16 kHz becomes frames of 640 samples (40 ms), and each frame
becomes one soft token: RMS norm (no weight) -> mm.a.input_projection.

A video becomes num_frames frames (32; the sampling of transformers:
indices 0, N/32, 2N/32, ... of the N frames, truncated), each an image with
a budget of 70 soft tokens, and the time of each frame (media.video_text).
PyAV decodes the video, as the PyAV reader of transformers does.

The weights come from the mmproj GGUF (gemma4uv, gemma4ua). The GGUF keeps
the 6912 values of a patch in the order channel, row, column (the order of
the im2col of llama.cpp); transformers uses row, column, channel. The
patches of this module use the order of the GGUF.

The image processor follows Gemma4UnifiedImageProcessor: a resize that keeps
the aspect, to sides that are multiples of 48 with at most budget x 48 x 48
pixels, bicubic, and then the pixels times 1/255. The budget is the most
soft tokens of an image: 70, 140, 280 (the default), 560, or 1120.
"""
from __future__ import annotations

import io
import math

import numpy as np

BUDGETS = (70, 140, 280, 560, 1120)
PATCH = 48              # the patch of a soft token (16 x the pool of 3)
TEACHER = 16            # the patch size of the processor before the merge
AUDIO_RATE = 16000
AUDIO_FRAME = 640       # samples of one audio soft token (40 ms)
AUDIO_MAX_S = 30        # the model card: at most 30 s of audio
VIDEO_FRAMES = 32       # the frames of a video (the video processor of transformers)
VIDEO_BUDGET = 70       # the soft tokens of a frame


def target_size(height, width, budget=280):
    """Return the (height, width) of the resized image, as
    get_aspect_ratio_preserving_size of transformers."""
    if budget not in BUDGETS:
        raise ValueError("the image budget must be one of %s, not %r" % (BUDGETS, budget))
    max_patches = budget * 9
    target_px = max_patches * TEACHER * TEACHER
    factor = math.sqrt(target_px / (height * width))
    th = int(math.floor(factor * height / PATCH)) * PATCH
    tw = int(math.floor(factor * width / PATCH)) * PATCH
    if th == 0 and tw == 0:
        raise ValueError("the image %dx%d is too small" % (width, height))
    max_side = (max_patches // 9) * PATCH
    if th == 0:
        th, tw = PATCH, min(int(math.floor(width / height)) * PATCH, max_side)
    elif tw == 0:
        tw, th = PATCH, min(int(math.floor(height / width)) * PATCH, max_side)
    return th, tw


def load_image(src):
    """Return an RGB PIL image from a path, bytes, or a PIL image."""
    from PIL import Image
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    return img.convert("RGB")


def image_pixels(img, budget=280):
    """Resize the image (bicubic) and return float32 pixels (h, w, 3) in
    [0, 1]."""
    from PIL import Image
    img = load_image(img)
    th, tw = target_size(img.height, img.width, budget)
    if (img.height, img.width) != (th, tw):
        img = img.resize((tw, th), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) * np.float32(1.0 / 255.0)


def image_patches(pixels):
    """Split pixels (h, w, 3) into patches of 48 x 48.

    Return the patches (n, 6912) in the order of the GGUF (channel, row,
    column), the positions (n, 2) as (x, y), and the grid (rows, cols). The
    patches go in raster order.
    """
    h, w, _ = pixels.shape
    rows, cols = h // PATCH, w // PATCH
    p = pixels[:rows * PATCH, :cols * PATCH].reshape(rows, PATCH, cols, PATCH, 3)
    p = p.transpose(0, 2, 4, 1, 3).reshape(rows * cols, 3 * PATCH * PATCH)
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    pos = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.int64)
    return np.ascontiguousarray(p), pos, (rows, cols)


def frame_indices(total, num_frames=VIDEO_FRAMES):
    """Return the indices of the sampled frames, as sample_frames of
    transformers: torch.arange(0, total, total / num_frames).int()."""
    k = min(int(num_frames), int(total))
    step = np.float32(total / k)
    idx = (np.arange(k, dtype=np.float32) * step).astype(np.int64)
    return idx[idx < total]


def load_video(src, num_frames=VIDEO_FRAMES):
    """Decode a video (a path or bytes) with PyAV. Return the sampled frames
    (RGB uint8 arrays) and the time of each in seconds (index / fps)."""
    import av
    f = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    with av.open(f) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        count = int(stream.frames)
        frames = [fr.to_ndarray(format="rgb24") for fr in container.decode(video=0)]
    total = count or len(frames)
    total = min(total, len(frames))
    if total == 0:
        raise ValueError("the video has no frames")
    idx = frame_indices(total, num_frames)
    return [frames[i] for i in idx], [i / fps for i in idx]


def load_audio(src):
    """Return mono float32 samples at 16 kHz from a path or bytes (any
    format of soundfile: WAV, FLAC, OGG, MP3)."""
    import soundfile as sf
    data, rate = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src,
                         dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if rate != AUDIO_RATE:
        from scipy.signal import resample_poly
        g = math.gcd(int(rate), AUDIO_RATE)
        x = resample_poly(x, AUDIO_RATE // g, int(rate) // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def audio_frames(samples):
    """Split 16 kHz samples into frames of 640 (n, 640), with zeros after
    the last sample."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = max(1, -(-len(x) // AUDIO_FRAME))
    out = np.zeros(n * AUDIO_FRAME, dtype=np.float32)
    out[:len(x)] = x
    return out.reshape(n, AUDIO_FRAME)


def _layer_norm(x, w, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    d = x - mu
    var = (d * d).mean(axis=-1, keepdims=True)
    return d / np.sqrt(var + eps) * w + b


def _rms_norm(x, eps):
    return x * np.power((x * x).mean(axis=-1, keepdims=True) + eps, -0.5)


class UnifiedEmbedder:
    """The two embedders of the 12B, from the mmproj GGUF, in float32."""

    LN_EPS = 1e-5       # torch.nn.LayerNorm (Gemma4UnifiedVisionEmbedder)

    def __init__(self, path):
        from .gguf import GGUF
        g = GGUF(path)
        meta = g.meta
        if meta.get("clip.vision.projector_type") != "gemma4uv":
            raise ValueError("%s is not a gemma4uv mmproj (projector %r)"
                             % (path, meta.get("clip.vision.projector_type")))
        self.path = path
        self.rms_eps = float(meta.get("clip.vision.attention.layer_norm_epsilon", 1e-6))
        self.audio_eps = float(meta.get("clip.audio.attention.layer_norm_epsilon", 1e-6))

        def T(name):
            return np.ascontiguousarray(np.asarray(g.dequant(name), dtype=np.float32))

        self.ln1 = (T("v.patch_norm.1.weight").reshape(-1), T("v.patch_norm.1.bias").reshape(-1))
        self.dense_w = T("v.patch_embd.weight").reshape(-1, 3 * PATCH * PATCH)
        self.dense_b = T("v.patch_embd.bias").reshape(-1)
        self.ln2 = (T("v.patch_norm.2.weight").reshape(-1), T("v.patch_norm.2.bias").reshape(-1))
        self.pos = T("v.position_embd.weight").reshape(2, -1, self.dense_w.shape[0])
        self.ln3 = (T("v.patch_norm.3.weight").reshape(-1), T("v.patch_norm.3.bias").reshape(-1))
        self.proj = T("mm.input_projection.weight")
        self.proj = self.proj.reshape(-1, self.dense_w.shape[0])
        self.has_audio = bool(meta.get("clip.has_audio_encoder"))
        if self.has_audio:
            self.audio_proj = T("mm.a.input_projection.weight").reshape(-1, AUDIO_FRAME)
        self.hidden = self.proj.shape[0]
        self.bidir = True       # use_bidirectional_attention "vision" (config.json)

    def image_rows(self, patches, pos):
        """Return the soft rows (n, hidden) of patches (n, 6912) at the
        positions pos (n, 2) of (x, y)."""
        x = _layer_norm(np.asarray(patches, np.float32), *self.ln1, self.LN_EPS)
        x = x @ self.dense_w.T + self.dense_b
        x = _layer_norm(x, *self.ln2, self.LN_EPS)
        pos = np.asarray(pos)
        x = x + self.pos[0][pos[:, 0]] + self.pos[1][pos[:, 1]]
        x = _layer_norm(x, *self.ln3, self.LN_EPS)
        x = _rms_norm(x, self.rms_eps)
        return np.ascontiguousarray(x @ self.proj.T, dtype=np.float32)

    def image(self, src, budget=280):
        """Return the soft rows and the grid (rows, cols) of an image."""
        patches, pos, grid = image_patches(image_pixels(src, budget))
        return self.image_rows(patches, pos), grid

    def video(self, src, budget=VIDEO_BUDGET, num_frames=VIDEO_FRAMES):
        """Return the frames of a video as a list of (seconds, soft rows)."""
        from PIL import Image
        frames, times = load_video(src, num_frames)
        out = []
        for fr, t in zip(frames, times):
            rows, _ = self.image(Image.fromarray(fr), budget)
            out.append((t, rows))
        return out

    def audio_rows(self, frames):
        """Return the soft rows (n, hidden) of audio frames (n, 640)."""
        if not self.has_audio:
            raise ValueError("%s has no audio projector" % self.path)
        x = _rms_norm(np.asarray(frames, np.float32), self.audio_eps)
        return np.ascontiguousarray(x @ self.audio_proj.T, dtype=np.float32)

    def audio(self, src):
        """Return the soft rows of a clip (a path, bytes, or 16 kHz samples)."""
        x = src if isinstance(src, np.ndarray) else load_audio(src)
        if len(x) > AUDIO_MAX_S * AUDIO_RATE:
            raise ValueError("the clip has %.1f s; the model takes at most %d s"
                             % (len(x) / AUDIO_RATE, AUDIO_MAX_S))
        return self.audio_rows(audio_frames(x))
