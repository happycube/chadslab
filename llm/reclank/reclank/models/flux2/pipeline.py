"""FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B, diffusers' layout):
text to image as diffusers' Flux2KleinPipeline (is_distilled: no CFG).

Ported from diffusers (src/diffusers/pipelines/flux2/pipeline_flux2_klein.py,
Copyright 2025 Black Forest Labs and The HuggingFace Team;
pipelines/flux2/image_processor.py, Copyright 2025 The Black Forest Labs Team
and The HuggingFace Team; schedulers/scheduling_flow_match_euler_discrete.py,
Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team; Apache
License 2.0). See THIRD_PARTY_NOTICES.md.

    the prompt in Qwen3's chat template (a user message, the assistant's
        turn with an empty think), 512 tokens (padding at the right)
    the text encoder (Qwen3-4B, qwen3.dense): the hidden states after layers
        9, 18 and 27 side by side (512, 7680)
    the DiT (dit.py) over the packed latents (the VAE's 32 channels in 2 x 2
        patches: 128 values a token, H/16 x W/16 tokens), a few Euler steps of
        the flow (FlowMatchEulerDiscreteScheduler, shifted by the empirical mu
        of the size and the steps)
    the VAE's decoder (vae.py) of the latents (its batch norm undone, the
        patches unpacked)

Reference images (editing, image to image: the pipeline's image=), each as
the pipeline's: an image over 1024 x 1024 pixels resized to that area
(Flux2ImageProcessor._resize_to_target_area: PIL's Lanczos), its sides
floored to multiples of 16 and the image cropped to them at the center, in
[-1, 1]; the VAE's encoder (the posterior's mean), the 2 x 2 patches packed,
the batch norm applied. Their rows follow the noisy ones in the DiT (dit.py);
the output's size is the first reference's unless given.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from reclank.encoders.runtime import RESIDENCY, gpu_runners
from reclank.models.flux2.dit import Flux2DiT
from reclank.models.flux2.vae import Flux2VAE
from reclank.models.qwen.tokenizer import QwenTokenizer
from reclank.models.qwen3.dense import Qwen3Dense

TEXT_TOKENS = 512
TAPS = (9, 18, 27)
REF_AREA = 1024 * 1024      # a larger reference image is resized to this area


def load_image(src):
    """A PIL image in RGB of a file's bytes or path, as diffusers' load_image
    (its EXIF orientation applied)."""
    import io

    from PIL import Image, ImageOps
    image = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess(image):
    """The pixels (3, H, W) float32 in [-1, 1] of a reference image (PIL) as
    Flux2KleinPipeline's: in RGB (load_image's convert: an alpha dropped),
    check_image_input (sides of 64 or more, an aspect ratio of at most 8),
    an image over REF_AREA resized to it (PIL's Lanczos to int(side *
    scale)), the sides floored to multiples of 16 and the image cropped to
    them at the center (Flux2ImageProcessor's resize_mode "crop": no
    resize), x / 255 * 2 - 1 in float32."""
    from PIL import Image
    image = image.convert("RGB")
    w, h = image.size
    if w < 64 or h < 64:
        raise ValueError("image too small: %dx%d (both sides must be 64 or more)" % (w, h))
    if max(w / h, h / w) > 8:
        raise ValueError("image's aspect ratio too extreme: %dx%d (at most 8:1)" % (w, h))
    if w * h > REF_AREA:
        scale = math.sqrt(REF_AREA / (w * h))
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        w, h = image.size
    W, H = w // 16 * 16, h // 16 * 16
    left, top = (w - W) // 2, (h - H) // 2
    image = image.crop((left, top, left + W, top + H))
    x = np.array(image).astype(np.float32) / 255.0
    return np.ascontiguousarray((2.0 * x - 1.0).astype(np.float32).transpose(2, 0, 1))


class Flux2Klein:
    """A FLUX.2 [klein] directory. parts: the parts to load ("text", "dit",
    "vae", or "all")."""

    def __init__(self, path, gpu=False, cpu_threads=None, parts=("all",)):
        self.path, self.gpu = path, bool(gpu)
        parts = set(parts)
        every = "all" in parts
        tdir = os.path.join(path, "tokenizer")
        self.tok = QwenTokenizer(os.path.join(tdir, "tokenizer.json"))
        tcfg = json.load(open(os.path.join(tdir, "tokenizer_config.json")))
        pad = tcfg.get("pad_token")
        pad = pad["content"] if isinstance(pad, dict) else pad
        self.pad = self.tok.special[pad]
        self.text = Qwen3Dense(os.path.join(path, "text_encoder"), TAPS, gpu=gpu, cpu_threads=cpu_threads) \
            if every or "text" in parts else None
        self.dit = Flux2DiT(os.path.join(path, "transformer"), gpu=gpu, cpu_threads=cpu_threads) \
            if every or "dit" in parts else None
        self.vae = Flux2VAE(os.path.join(path, "vae"), gpu=gpu, cpu_threads=cpu_threads) \
            if every or "vae" in parts else None
        self.sched = json.load(open(os.path.join(path, "scheduler", "scheduler_config.json")))

    def runners(self):
        """The GPU runners of the parts (name: _Runner); vae_enc: the VAE's
        encoder (reference images)."""
        out = gpu_runners(self.gpu, text=self.text, dit=self.dit, vae=self.vae)
        if "vae" in out:
            out["vae_enc"] = self.vae.enc_runner
        return out

    def tokenize(self, prompt):
        """The ids (512) of a prompt in the chat template, and its real
        tokens (the rest: padding)."""
        text = ("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n")
        ids = self.tok.encode(text)[:TEXT_TOKENS]
        n = len(ids)
        return np.array(ids + [self.pad] * (TEXT_TOKENS - n), np.int32), n

    def text_embed(self, ids, tokens):
        """The DiT's text input (n, 512, 7680) of n prompts' ids (n, 512)."""
        taps = self.text.hidden(ids, tokens)
        return np.concatenate(taps, axis=-1)

    def sigmas(self, steps, tokens):
        """The sigmas of the steps (and the 0 after the last) of an image of
        tokens: linspace(1, 1 / steps), shifted by the empirical mu of
        Flux2KleinPipeline (FlowMatchEulerDiscreteScheduler, exponential
        time shift), float32."""
        a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
        if tokens > 4300:
            mu = a2 * tokens + b2
        else:
            m200, m10 = a2 * tokens + b2, a1 * tokens + b1
            a = (m200 - m10) / 190.0
            mu = a * steps + (m200 - 200.0 * a)
        assert self.sched.get("use_dynamic_shifting") and self.sched.get("time_shift_type") == "exponential"
        s = np.linspace(1.0, 1.0 / steps, steps)
        s = math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0))
        return np.append(s.astype(np.float32), np.float32(0))

    def references(self, pixels):
        """The DiT's rows (n, 128) of reference images and their (h, w) in
        tokens: each image's pixels (3, H, W) of preprocess() through the
        VAE's encoder, its patches packed, the batch norm applied."""
        rows, refs = [], []
        for x in pixels:
            z = self.vae.encode(x)
            rows.append(self.vae.normalize(z))
            refs.append((z.shape[1] // 2, z.shape[2] // 2))
        return np.concatenate(rows), refs

    def denoise(self, noise, txt, hh, ww, steps, keep=None, ref=None, refs=()):
        """The latents (hh ww, 128) of the Euler steps of the flow from noise
        (hh ww, 128) with the text rows txt (512, 7680) and the reference
        images' rows ref of references(). keep: a list that gets each step's
        (input, velocity)."""
        sig = self.sigmas(steps, hh * ww)
        x = noise.astype(np.float32)
        for i in range(steps):
            # the pipeline's timestep: sigma 1000 (float32), / 1000 into the DiT, which takes * 1000
            t = np.float32(np.float32(sig[i] * np.float32(1000)) / np.float32(1000)) * np.float32(1000)
            v = self.dit.velocity(x, float(t), txt, hh, ww, ref, refs)
            if keep is not None:
                keep.append((x.copy(), v))
            x = (x + (sig[i + 1] - sig[i]) * v).astype(np.float32)
        return x

    def image(self, latents, hh, ww):
        """The image (3, 16 hh, 16 ww) in [-1, 1] of the packed latents."""
        return self.vae.decode(self.vae.unnormalize(latents, hh, ww))

    def generate(self, prompt, size=None, steps=4, seed=0, noise=None, keep_text=False, images=None):
        """An image (H, W, 3) uint8 of a prompt: size a multiple of 16 (or
        (height, width); None: the first reference image's, else 1024);
        images: reference images (PIL images, or pixels of preprocess());
        noise (tokens, 128) in place of the seed's (NumPy's normal: not
        torch's, so not diffusers' image of the same seed). On the GPU the
        text encoder's memory (5.5 GB) is freed after the prompt and the
        VAE's (its buffers: 5 GB at 1024 x 1024) after the image, unless
        keep_text: the DiT stays, beside one of them (keep_text: RESIDENCY
        decides, each part unpinned after its turn); the VAE encoder (4 GB
        at 1024 x 1024) is unpinned after the references."""
        pixels = [x if isinstance(x, np.ndarray) else preprocess(x) for x in images or []]
        if size is None:
            size = pixels[0].shape[1:] if pixels else 1024
        hgt, wid = (size, size) if isinstance(size, int) else size
        hh, ww = hgt // 16, wid // 16
        with RESIDENCY.use(*self.runners().values()) as use:
            ids, n = self.tokenize(prompt)
            txt = self.text_embed(ids[None], [n])[0]
            use.done(self.text.runner)
            if self.gpu and not keep_text:
                self.text.runner.release(keep_host=True)
            ref, refs = (None, ())
            if pixels:
                ref, refs = self.references(pixels)
                use.done(self.vae.enc_runner)
            if noise is None:
                noise = np.random.default_rng(seed).standard_normal((hh * ww, 4 * self.vae.latent)).astype(np.float32)
            lat = self.denoise(noise, txt, hh, ww, steps, ref=ref, refs=refs)
            use.done(self.dit.runner)
            img = self.image(lat, hh, ww)
            if self.gpu and not keep_text:
                self.vae.runner.release(keep_host=True)
        return np.clip((img.transpose(1, 2, 0) + 1) * 127.5 + 0.5, 0, 255).astype(np.uint8)
