"""Stable Diffusion 1.5 and 2.0 (diffusers' layout: runwayml's 1.5,
stabilityai's 2 base at 512 and 2 at 768 with v-prediction): text to image as
diffusers' StableDiffusionPipeline, with classifier-free guidance.

Ported from diffusers
(src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py; Apache
License 2.0; Copyright 2026 The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

    the prompt and the negative prompt ("" by default) in CLIP's tokenizer, 77
        ids each (tokenizer.py)
    the text encoder (clip.py): the last hidden state of both (77, width)
    the UNet (unet.py) of the batch of the negative's latents and the
        prompt's (the same latents) at each timestep; the output the
        negative's + guidance (the prompt's - the negative's)
    the pipeline's scheduler (model_index.json: PNDM for 1.5 and 2.0-base,
        DDIM for 2.0-768; scheduler.py), eta 0
    the VAE's decoder (flux2.vae: AutoencoderKL) of the latents / 0.18215
        (scaling_factor)
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import RESIDENCY, gpu_runners
from reclank.models.flux2.vae import Flux2VAE
from reclank.models.sd.clip import CLIPText
from reclank.models.sd.scheduler import Scheduler
from reclank.models.sd.tokenizer import CLIPTokenizer
from reclank.models.sd.unet import SDUNet


class StableDiffusion:
    """A Stable Diffusion directory. parts: the parts to load ("text",
    "unet", "vae", or "all")."""

    def __init__(self, path, gpu=False, cpu_threads=None, parts=("all",), bf12=False, f20=False):
        self.path, self.gpu = path, bool(gpu)
        parts = set(parts)
        every = "all" in parts
        index = json.load(open(os.path.join(path, "model_index.json")))
        self.tok = CLIPTokenizer(os.path.join(path, "tokenizer"))
        self.text = CLIPText(os.path.join(path, "text_encoder"), gpu=gpu, cpu_threads=cpu_threads) \
            if every or "text" in parts else None
        self.unet = SDUNet(os.path.join(path, "unet"), gpu=gpu, cpu_threads=cpu_threads, bf12=bf12,
                           f20=f20) \
            if every or "unet" in parts else None
        self.vae = Flux2VAE(os.path.join(path, "vae"), gpu=gpu, cpu_threads=cpu_threads) \
            if every or "vae" in parts else None
        vcfg = json.load(open(os.path.join(path, "vae", "config.json")))
        self.scale = np.float32(vcfg.get("scaling_factor", 0.18215))
        ucfg = json.load(open(os.path.join(path, "unet", "config.json")))
        self.size = 8 * ucfg["sample_size"]            # the trained size: 512, 768
        self.latent = ucfg["in_channels"]
        scfg = json.load(open(os.path.join(path, "scheduler", "scheduler_config.json")))
        self.sched = Scheduler(scfg, index["scheduler"][1])

    def runners(self):
        """The GPU runners of the parts (name: _Runner)."""
        return gpu_runners(self.gpu, text=self.text, unet=self.unet, vae=self.vae)

    def text_embed(self, prompts):
        """The text encoder's output (n, 77, width) of n prompts."""
        return self.text.hidden(np.array([self.tok.encode(p) for p in prompts], np.int32))

    def denoise(self, noise, emb, uemb, steps, guidance=7.5, keep=None):
        """The latents (4, h, w) of the steps from noise (4, h, w) with the
        prompt's rows emb and the negative's uemb (77, width). keep: a list
        that gets each step's (input, the UNet's output (2, 4, h, w))."""
        s = self.sched
        s.start(steps)
        ctx = np.stack([uemb, emb]).astype(np.float32)
        x = noise.astype(np.float32)
        g = np.float32(guidance)
        for t in s.timesteps(steps):
            e = self.unet.output(np.stack([x, x]), float(t), ctx)
            if keep is not None:
                keep.append((x.copy(), e))
            out = e[0] + g * (e[1] - e[0])
            x = s.step(out, t, x)
        return x

    def image(self, latents):
        """The image (3, 8 h, 8 w) in [-1, 1] of latents (4, h, w)."""
        return self.vae.decode((latents / self.scale).astype(np.float32))

    def generate(self, prompt, negative="", size=None, steps=25, guidance=7.5, seed=0, noise=None, keep_text=False):
        """An image (H, W, 3) uint8 of a prompt: size a multiple of 64 (or
        (height, width); default the trained size); noise (4, H/8, W/8) in
        place of the seed's (NumPy's normal: not torch's, so not diffusers'
        image of the same seed). On the GPU the text encoder's memory is freed
        after the prompt and the VAE's after the image, unless keep_text
        (RESIDENCY decides, each part unpinned after its turn)."""
        size = self.size if size is None else size
        hgt, wid = (size, size) if isinstance(size, int) else size
        assert hgt % 64 == 0 and wid % 64 == 0
        with RESIDENCY.use(*self.runners().values()) as use:
            uemb, emb = self.text_embed([negative, prompt])
            use.done(self.text.runner)
            if self.gpu and not keep_text:
                self.text.runner.release(keep_host=True)
            if noise is None:
                noise = np.random.default_rng(seed).standard_normal((self.latent, hgt // 8, wid // 8)).astype(np.float32)
            lat = self.denoise(noise, emb, uemb, steps, guidance)
            use.done(self.unet.runner)
            img = self.image(lat)
            if self.gpu and not keep_text:
                self.vae.runner.release(keep_host=True)
        # diffusers' postprocess: (x / 2 + 0.5) clamped, * 255 rounded
        return np.round(np.clip(img.transpose(1, 2, 0) / 2 + 0.5, 0, 1) * 255).astype(np.uint8)
