"""Wan 2.2 TI2V-5B and T2V-A14B text to video (diffusers' layouts: Wan-AI/Wan2.2-TI2V-5B-Diffusers,
Wan-AI/Wan2.2-T2V-A14B-Diffusers) as diffusers' WanPipeline (the 5B with expand_timesteps; no image:
every token at the step's timestep), with classifier-free guidance.

Ported from diffusers (src/diffusers/pipelines/wan/pipeline_wan.py; Apache License 2.0; Copyright
2025 The Wan Team and The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

The A14B (model_index.json's boundary_ratio): two DiTs, transformer (the high-noise expert) for the
steps of t >= boundary_ratio x 1000 at guidance, transformer_2 for the rest at guidance2 (the model
card's 4.0 and 3.0; the switch once a video: the low expert's text caches then); a scalar timestep
(no expand_timesteps: the same vectors); 16 latent channels, Wan 2.1's VAE (8 x 8 x 4); the experts
streamed (models/wan/dit.py), from a weightkeeper that holds the directory when one runs
(models/wan/keep.py; keeper False: never, a name: that keeper's), else loaded.

    the prompt and the negative prompt (the model card's by default) through the text encoder
        (umT5-XXL: an object of encode(prompts, max_len) -> (n, 512, 4096) float32, the
        pipeline's prompt_embeds: zeros past each prompt's tokens), or their embeddings given
    the DiT (dit.py) of the prompt's and the negative's text at each timestep (two calls: the
        caches of both texts), the output the negative's + guidance (the prompt's - the
        negative's)
    the scheduler (scheduler.py: UniPC of flow sigmas, shift 5)
    the VAE's decoder (vae.py) of the latents / (1 / std) + mean
    frames: diffusers' postprocess ((x / 2 + 0.5) clamped), times 255 rounded, uint8

The noise: NumPy's normal from the seed (not torch's: not diffusers' video of the same seed), or
given (the reference's). On the GPU the DiT's memory is freed before the VAE runs (keep: not),
and the VAE's after the video.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from reclank.encoders.runtime import RESIDENCY, gpu_runners
from reclank.models import media
from reclank.models.wan.dit import TEXT, WanDiT
from reclank.models.wan.scheduler import UniPC
from reclank.models.wan.vae import WanVAE

# the model card's negative prompt (the authors' default)
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
            "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
            "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


class WanT2V:
    """A Wan 2.2 TI2V-5B directory. parts: the parts to load ("dit", "vae", or "all"); encoder:
    the text encoder (encode(prompts, max_len) -> (embeds, lengths)); None: the directory's
    umT5-XXL (models/t5, on the CPU: 10 GB of RAM), loaded at the first prompt that needs it
    (not when the embeds are given). x2: the DiT's attention on the GPU as ENC_ATTN_X2 (None:
    RECLANK_WAN_X2=1), else float32."""

    def __init__(self, path, gpu=True, cpu_threads=None, encoder=None, parts=("all",), f20=None, x2=None,
                 keeper=None, stream=None):
        self.path, self.gpu = path, bool(gpu)
        parts = set(parts)
        every = "all" in parts
        self.encoder, self.cpu_threads = encoder, cpu_threads
        mi = json.load(open(os.path.join(path, "model_index.json")))
        ratio = mi.get("boundary_ratio")
        self.a14b = ratio is not None
        self.dit = self.dit2 = None
        self.keeper = None
        if every or "dit" in parts:
            kept = [None, None]
            if self.a14b and keeper is not False and gpu:
                from reclank.models.wan import keep
                w = keep.attach(path, name=keeper if isinstance(keeper, str) else None)
                if w is not None:
                    self.keeper = w
                    kept = [keep.KeptExpert(w, sub) for sub in keep.SUBS]
                elif isinstance(keeper, str):
                    raise ValueError("the weightkeeper %r does not hold %s" % (keeper, path))
            self.dit = WanDiT(os.path.join(path, "transformer"), gpu=gpu, cpu_threads=cpu_threads, f20=f20,
                              x2=x2, stream=stream, keeper=kept[0])
            if self.a14b:
                self.dit2 = WanDiT(os.path.join(path, "transformer_2"), gpu=gpu, cpu_threads=cpu_threads,
                                   f20=f20, x2=x2, stream=stream, share=self.dit, keeper=kept[1])
        self.vae = WanVAE(os.path.join(path, "vae"), gpu=gpu, cpu_threads=cpu_threads) \
            if every or "vae" in parts else None
        vcfg = json.load(open(os.path.join(path, "vae", "config.json")))
        # (Wan 2.1's VAE config has no scale factors: AutoencoderKLWan's defaults, 8 and 4)
        self.sf, self.tf = vcfg.get("scale_factor_spatial", 8), vcfg.get("scale_factor_temporal", 4)
        self.latent = vcfg["z_dim"]
        scfg = json.load(open(os.path.join(path, "scheduler", "scheduler_config.json")))
        self.sched = UniPC(scfg)
        self.boundary = None if ratio is None else ratio * scfg["num_train_timesteps"]
        self.fps = 16 if self.a14b else 24

    def encode(self, prompts):
        """The pipeline's prompt_embeds (n, 512, 4096) of n prompts."""
        if self.encoder is None:
            from reclank.models.t5.encoder import UMT5Encoder
            self.encoder = UMT5Encoder(os.path.join(self.path, "text_encoder"), cpu_threads=self.cpu_threads)
        return np.asarray(self.encoder.encode(list(prompts), TEXT)[0], np.float32)

    def shape(self, size, frames):
        """(height, width, frames) as the pipeline rounds them, and the latents' shape."""
        hgt, wid = size
        m = self.sf * 2
        hgt, wid = hgt // m * m, wid // m * m
        if frames % self.tf != 1:
            frames = frames // self.tf * self.tf + 1
        frames = max(frames, 1)
        return hgt, wid, frames, (self.latent, (frames - 1) // self.tf + 1, hgt // self.sf, wid // self.sf)

    def defaults(self):
        """(steps, guidance, guidance2) of the model card: the 5B's 50 and 5.0, the A14B's 40, 4.0
        and 3.0."""
        return (40, 4.0, 3.0) if self.a14b else (50, 5.0, None)

    def expert(self, t):
        """The DiT of timestep t (the A14B: transformer at t >= the boundary, else transformer_2;
        as WanPipeline compares its int64 timestep) and its index."""
        if self.boundary is None or t >= self.boundary:
            return self.dit, 0
        return self.dit2, 1

    def denoise(self, noise, emb, nemb, steps, guidance=5.0, keep=None, progress=None, guidance2=None):
        """The latents (C, f, h, w) of the steps from noise with the prompt's rows emb and the
        negative's nemb (512, 4096; None: no guidance). keep: a list that gets each step's
        (input, the prompt's output, the negative's). progress(i, steps): after each step.
        guidance2: the A14B's low-noise expert's (None: guidance). The DiT stays pinned from the
        prefix (its texts' keys and values on the GPU) to the last step; the A14B's experts share
        one runner (its text caches: the high expert's, then the low one's from its first step),
        so one scope holds both."""
        cfg = nemb is not None and guidance > 1.0
        with RESIDENCY.use(*gpu_runners(self.gpu, dit=self.dit).values()):
            s = self.sched
            s.set_timesteps(steps)
            x = noise.astype(np.float32)
            gs = (np.float32(guidance), np.float32(guidance if guidance2 is None else guidance2))
            cur = None
            for i, t in enumerate(s.timesteps):
                dit, e = self.expert(t)
                if dit is not cur:
                    # (the text caches of this expert: once a video for each)
                    dit.prefix(emb, nemb if cfg else None)
                    cur = dit
                g = gs[e]
                vc = dit.velocity(x, float(t), 0)
                vu = dit.velocity(x, float(t), 1) if cfg else None
                if keep is not None:
                    keep.append((x.copy(), vc, vu))
                v = (vu + g * (vc - vu)).astype(np.float32) if cfg else vc
                x = s.step(v, x)
                if progress is not None:
                    progress(i + 1, steps)
        return x

    def video(self, latents, progress=None):
        """The video (3, frames, H, W) in [-1, 1] of latents (C, f, h, w)."""
        return self.vae.decode(self.vae.denorm(latents), progress=progress)

    def runners(self):
        """The GPU runners of the parts (name: _Runner)."""
        return gpu_runners(self.gpu, dit=self.dit, vae=self.vae)

    def release(self, which=("dit", "vae")):
        """Free the GPU memory of parts (their host programs kept)."""
        for name in which:
            part = getattr(self, name)
            if part is not None and self.gpu:
                part.runner.release(keep_host=True)

    def generate(self, prompt, negative=None, size=(480, 832), frames=17, steps=None, guidance=None, seed=0,
                 noise=None, embeds=None, keep=False, progress=None, guidance2=None):
        """The video frames (T, H, W, 3) uint8 of a prompt: size (height, width), multiples of 32
        (the A14B: 16; rounded down as the pipeline); frames 4k + 1 (rounded); noise (C, f, H/sf,
        W/sf) in place of the seed's; embeds: (the prompt's, the negative prompt's) rows (512, 4096)
        in place of the encoder's. negative None: the model card's. steps, guidance, guidance2
        None: defaults(). progress(stage, i, n): stage "text", "step", "vae". On the GPU the DiT's
        memory is freed before the VAE, and the VAE's after the video (unless keep: RESIDENCY
        decides, the DiT unpinned before the VAE runs)."""
        d_steps, d_g, d_g2 = self.defaults()
        steps = d_steps if steps is None else steps
        guidance = d_g if guidance is None else guidance
        guidance2 = (d_g2 if guidance2 is None else guidance2) if self.a14b else None
        hgt, wid, frames, shape = self.shape(size, frames)
        cfg = guidance > 1.0
        if embeds is None:
            negative = NEGATIVE if negative is None else negative
            e = self.encode([prompt, negative] if cfg else [prompt])
            emb, nemb = e[0], e[1] if cfg else None
        else:
            emb, nemb = embeds
        if progress is not None:
            progress("text", 1, 1)
        if noise is None:
            noise = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
        assert noise.shape == shape, (noise.shape, shape)
        with RESIDENCY.use(*self.runners().values()) as use:
            lat = self.denoise(noise, emb, nemb, steps, guidance,
                               progress=None if progress is None else lambda i, n: progress("step", i, n),
                               guidance2=guidance2)
            use.done(self.dit.runner)
            if self.gpu and not keep:
                self.release(("dit",))
            vid = self.video(lat, progress=None if progress is None else lambda i, n: progress("vae", i, n))
            if self.gpu and not keep:
                self.release(("vae",))
        return frames_u8(vid)


def frames_u8(video):
    """(T, H, W, 3) uint8 of a video (3, T, H, W) in [-1, 1]: diffusers' postprocess, times 255
    rounded."""
    return np.round(np.clip(video.transpose(1, 2, 3, 0) / 2 + 0.5, 0, 1) * 255).astype(np.uint8)


def write_mp4(frames, path, fps=24, crf=18, colors=None):
    """H.264 MP4 (yuv420p, the frames' size) of frames (T, H, W, 3) uint8 by ffmpeg: converted by colors
    (a source's tags; none: BT.601 limited) and tagged so (media.yuv_colors); ffmpeg on the start CPUs
    (media.run: the calling thread may be bound to one)."""
    T, H, W, _ = frames.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(fps), "-i", "-", "-c:v", "libx264"] + media.yuv_args(colors) + [
           "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium", "-movflags", "+faststart", path]
    t0 = time.time()
    p = media.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg: %s" % p.stderr.decode(errors="replace").strip())
    return time.time() - t0
