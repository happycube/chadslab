"""Z-Image Turbo (Tongyi-MAI/Z-Image-Turbo, diffusers' layout): text to image as diffusers'
ZImagePipeline with guidance 0 (the Turbo model's: no CFG).

Ported from diffusers (src/diffusers/pipelines/z_image/pipeline_z_image.py, Copyright 2025
Alibaba Z-Image Team and The HuggingFace Team;
schedulers/scheduling_flow_match_euler_discrete.py, Copyright 2025 Stability AI, Katherine
Crowson and The HuggingFace Team; Apache License 2.0). See THIRD_PARTY_NOTICES.md.

    the prompt in Qwen3's chat template (a user message, the assistant's turn; enable_thinking:
        no think block), at most 512 tokens
    the text encoder (Qwen3-4B, qwen3.dense): hidden_states[-2], the residual stream after 35
        of its 36 layers, of the prompt's tokens (T, 2560: the padding dropped)
    the DiT (dit.py): the caption's rows once a prompt, then the Euler steps of the flow over
        the latents (FlowMatchEulerDiscreteScheduler: torch's linspace(1, 1 / steps), shift 3;
        the DiT's timestep (1000 - t) / 1000, its output negated)
    the VAE's decoder (flux2.vae: FLUX.1's AutoencoderKL, 16 channels, no post_quant_conv) of
        the latents / scaling_factor + shift_factor

The text encoder runs on the CPU (its bfloat16 rows, KQ_BF16X16: float32 products, 8 GB), as
it does not fit on a 24 GB GPU beside the DiT's 15.3 GB of F20; text_gpu puts it on the GPU
(its bfloat16 rows, KQ_BF16, freed after the prompt; the DiT's memory too before it runs). On the
GPU the DiT's memory is freed before the VAE runs and the VAE's after the image, unless keep:
the next image takes the DiT's weights back. The noise: NumPy's normal from the seed (not
torch's: not diffusers' image of the same seed), or given (the reference's).
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import RESIDENCY, gpu_runners
from reclank.models.flux2.vae import Flux2VAE
from reclank.models.qwen.tokenizer import QwenTokenizer
from reclank.models.qwen3.dense import Qwen3Dense
from reclank.models.zimage.dit import ZImageDiT

TEXT_TOKENS = 512   # the pipeline's max_sequence_length
TEXT_ROWS = 32      # the text encoder's rows a multiple of these (padding masked)


def torch_linspace(start, end, n):
    """torch.linspace(start, end, n) in float32 as torch's CPU kernel: start + step i for the
    first half, end - step (n - 1 - i) for the rest, each a fused multiply-add."""
    s, e = np.float32(start), np.float32(end)
    if n == 1:
        return np.array([s], np.float32)
    step = np.float32((e - s) / np.float32(n - 1))
    i = np.arange(n, dtype=np.float64)
    lo = np.float64(s) + np.float64(step) * i
    hi = np.float64(e) - np.float64(step) * (n - 1 - i)
    return np.where(i < n // 2, lo, hi).astype(np.float32)


class ZImageTurbo:
    """A Z-Image Turbo directory. parts: the parts to load ("text", "dit", "vae", or "all"); f20:
    the DiT's matrices in F20 (default: on the GPU, not on the CPU); x2: the DiT's attention on the
    GPU as ENC_ATTN_X2 (None: RECLANK_ZIMAGE_X2=1), else float32."""

    def __init__(self, path, gpu=False, cpu_threads=None, parts=("all",), text_gpu=False, f20=None, x2=None):
        self.path, self.gpu = path, bool(gpu)
        parts = set(parts)
        every = "all" in parts
        tdir = os.path.join(path, "tokenizer")
        self.tok = QwenTokenizer(os.path.join(tdir, "tokenizer.json"))
        tcfg = json.load(open(os.path.join(tdir, "tokenizer_config.json")))
        pad = tcfg.get("pad_token")
        self.pad = self.tok.special[pad["content"] if isinstance(pad, dict) else pad]
        edir = os.path.join(path, "text_encoder")
        ecfg = json.load(open(os.path.join(edir, "config.json")))
        self.text_gpu = gpu and text_gpu
        self.text = Qwen3Dense(edir, (ecfg["num_hidden_layers"] - 1,), gpu=self.text_gpu, cpu_threads=cpu_threads,
                               cpu_bf16=True) if every or "text" in parts else None
        self.dit = ZImageDiT(os.path.join(path, "transformer"), gpu=gpu, cpu_threads=cpu_threads, f20=f20,
                               x2=x2) if every or "dit" in parts else None
        self.vae = Flux2VAE(os.path.join(path, "vae"), gpu=gpu, cpu_threads=cpu_threads) \
            if every or "vae" in parts else None
        vcfg = json.load(open(os.path.join(path, "vae", "config.json")))
        self.scale, self.shift = np.float32(vcfg["scaling_factor"]), np.float32(vcfg["shift_factor"])
        self.latent = vcfg["latent_channels"]
        self.sched = json.load(open(os.path.join(path, "scheduler", "scheduler_config.json")))

    def tokenize(self, prompt):
        """The ids of a prompt in the chat template (enable_thinking), at most 512."""
        text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        return np.array(self.tok.encode(text)[:TEXT_TOKENS], np.int32)

    def text_embed(self, ids):
        """The DiT's caption rows (T, 2560) of a prompt's ids: the text encoder's hidden_states[-2]
        of its tokens (the encoder's rows padded to a multiple of TEXT_ROWS at the end, which the
        causal rows do not see)."""
        n = len(ids)
        rows = -(-n // TEXT_ROWS) * TEXT_ROWS
        x = np.full(rows, self.pad, np.int32)
        x[:n] = ids
        h = self.text.hidden(x[None], [n])[0][0]
        if self.text_gpu:
            self.text.runner.release(keep_host=True)
        return np.ascontiguousarray(h[:n])

    def sigmas(self, steps):
        """The sigmas of the steps (and the 0 after the last), and the timesteps:
        get_default_z_image_sigmas (torch's linspace(1, 1 / steps)) shifted by shift / (1 + (shift
        - 1) s), float32 as the scheduler's numpy; the timesteps sigma 1000."""
        s = self.sched
        assert not s.get("use_dynamic_shifting") and not s.get("shift_terminal") and not s.get("invert_sigmas")
        sig = torch_linspace(1.0, 1.0 / steps, steps)
        sh = np.float32(s.get("shift", 1.0))
        sig = (sh * sig / (np.float32(1) + (sh - np.float32(1)) * sig)).astype(np.float32)
        return np.append(sig, np.float32(0)), (sig * np.float32(s.get("num_train_timesteps", 1000))).astype(np.float32)

    @staticmethod
    def dit_time(t):
        """The pipeline's timestep into the DiT of a scheduler's timestep t: (1000 - t) / 1000."""
        return float(np.float32(np.float32(np.float32(1000) - np.float32(t)) / np.float32(1000)))

    def denoise(self, noise, emb, steps, keep=None, progress=None):
        """The latents (16, h, w) of the Euler steps of the flow from noise (16, h, w) with the
        caption's rows emb (T, 2560). keep: a list that gets each step's (input, the DiT's
        output). progress(i, steps): after each step. The DiT stays pinned from the prefix (the
        caption's rows on the GPU) to the last step."""
        sig, ts = self.sigmas(steps)
        with RESIDENCY.use(*gpu_runners(self.gpu, dit=self.dit).values()):
            self.dit.prefix(emb)
            x = noise.astype(np.float32)
            for i in range(steps):
                v = self.dit.velocity(x, self.dit_time(ts[i]))
                if keep is not None:
                    keep.append((x.copy(), v))
                x = (x + (sig[i + 1] - sig[i]) * -v).astype(np.float32)
                if progress is not None:
                    progress(i + 1, steps)
        return x

    def image(self, latents):
        """The image (3, 8 h, 8 w) in [-1, 1] of latents (16, h, w)."""
        return self.vae.decode((latents / self.scale + self.shift).astype(np.float32))

    def runners(self):
        """The GPU runners of the parts (name: _Runner; the text encoder's with text_gpu)."""
        return gpu_runners(self.gpu, text=self.text if self.text_gpu else None, dit=self.dit, vae=self.vae)

    def release(self, which=("text", "dit", "vae")):
        """Free the GPU memory of parts (their host programs kept)."""
        for name in which:
            part = getattr(self, name)
            if part is not None and (self.gpu if name != "text" else self.text_gpu):
                part.runner.release(keep_host=True)

    def generate(self, prompt, size=1024, steps=9, seed=0, noise=None, keep=False):
        """An image (H, W, 3) uint8 of a prompt: size (or (height, width)) multiples of 16; noise
        (16, H/8, W/8) in place of the seed's. On the GPU the DiT's memory is freed before the VAE
        and the VAE's after it, unless keep (RESIDENCY decides, each part unpinned after its turn)."""
        hgt, wid = (size, size) if isinstance(size, int) else size
        assert hgt % 16 == 0 and wid % 16 == 0, "the height and the width must be multiples of 16"
        with RESIDENCY.use(*self.runners().values()) as use:
            if self.text_gpu:
                self.release(("dit",))
            emb = self.text_embed(self.tokenize(prompt))
            use.done(self.text.runner)
            shape = (self.latent, hgt // 8, wid // 8)
            if noise is None:
                noise = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
            assert noise.shape == shape, (noise.shape, shape)
            lat = self.denoise(noise, emb, steps)
            use.done(self.dit.runner)
            if self.gpu and not keep:
                self.release(("dit",))
            img = self.image(lat)
            if self.gpu and not keep:
                self.release(("vae",))
        return np.round((img.transpose(1, 2, 0) / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
