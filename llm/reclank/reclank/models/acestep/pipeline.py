"""ACE-Step 1.5 (turbo) text to music without the 5 Hz LM (the authors'
generate_music with thinking off): a caption, lyrics and metadata to 48 kHz
stereo.

Ported from ACE-Step 1.5 (acestep/inference.py,
acestep/models/turbo/modeling_acestep_v15_turbo.py; MIT License, Copyright (c) 2026
ACEStep; its model files Apache License 2.0, Copyright 2025 The ACESTEO Team). See
THIRD_PARTY_NOTICES.md.

    the text: "# Instruction\\n{instruction}\\n\\n# Caption\\n{caption}\\n\\n# Metas\\n
        {metas}<|endoftext|>\\n" (SFT_GEN_PROMPT) and the tokenizer's
        <|endoftext|>; Qwen3-Embedding-0.6B's last hidden state (qwen3.dense)
    the lyrics: "# Languages\\n{language}\\n\\n# Lyric\\n{lyrics}<|endoftext|>" and
        <|endoftext|>; the text encoder's embedding rows
    the DiT (dit.py): the conditions once, then 8 Euler steps of the flow at
        t = 1 - i / 8 (shift 1); after each step DCW (the authors' default,
        "double", Haar): the latents' low band pushed by t 0.05 and the high
        band by (1 - t) 0.02 away from the step's clean estimate x - v t; the
        last step x - v t
    the VAE's decoder (vae.py) in the authors' tiles of 256 frames (64 of
        overlap); the peak normalized to -1 dB
"""
from __future__ import annotations

import os

import numpy as np

from reclank.encoders.runtime import RESIDENCY, gpu_runners
from reclank.formats.safetensors import SafeTensors
from reclank.models.acestep.dit import AceStepDiT
from reclank.models.acestep.vae import OobleckVAE
from reclank.models.qwen.tokenizer import QwenTokenizer
from reclank.models.qwen3.dense import Qwen3Dense

INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
EOT = "<|endoftext|>"
FPS = 25            # latent frames a second (48 kHz / 1920)


def metas(bpm=None, timesignature=None, keyscale=None, duration=None):
    """The "# Metas" lines of the prompt (N/A for what is not given)."""
    v = lambda x: "N/A" if x in (None, "", 0) else x  # noqa: E731
    return "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %s\n" % (
        v(bpm), v(timesignature), v(keyscale), "%d seconds" % int(duration) if duration else "N/A")


def dcw_double(x, y, t, low=0.05, high=0.02):
    """DCW "double" with the Haar wavelet along time (pairs of frames): x's
    low band plus t low (x - y)'s, its high band plus (1 - t) high (x - y)'s
    (pytorch_wavelets' DWT1D, an even number of frames)."""
    sl, sh = np.float32(t * low), np.float32((1.0 - t) * high)
    if sl == 0 and sh == 0:
        return x
    d = (x - y).reshape(-1, 2, x.shape[1])
    s, df = d[:, 0] + d[:, 1], d[:, 0] - d[:, 1]
    out = x.reshape(-1, 2, x.shape[1]).copy()
    out[:, 0] += (sl * s + sh * df) / 2
    out[:, 1] += (sl * s - sh * df) / 2
    return out.reshape(x.shape).astype(np.float32)


class AceStep:
    """An Ace-Step1.5 directory (acestep-v15-turbo, vae, Qwen3-Embedding-0.6B)."""

    def __init__(self, path, gpu=False, cpu_threads=None, planner=True):
        te = os.path.join(path, "Qwen3-Embedding-0.6B")
        self.gpu = bool(gpu)
        self.tok = QwenTokenizer(os.path.join(te, "tokenizer.json"))
        self.eot = self.tok.special[EOT]
        self.text = Qwen3Dense(te, (28,), gpu=gpu, cpu_threads=cpu_threads, prefix="", final_norm=True)
        self.emb = SafeTensors(os.path.join(te, "model.safetensors"))
        self.dit = AceStepDiT(os.path.join(path, "acestep-v15-turbo"), gpu=gpu, cpu_threads=cpu_threads)
        self.vae = OobleckVAE(os.path.join(path, "vae"), gpu=gpu, cpu_threads=cpu_threads)
        self.planner = None
        if planner:
            from reclank.models.acestep.planner import Planner
            self.planner = Planner(path, gpu=gpu, cpu_threads=cpu_threads)

    def runners(self):
        """The GPU runners of the parts (name: _Runner)."""
        return gpu_runners(self.gpu, text=self.text, planner=self.planner.lm if self.planner else None, dit=self.dit,
                           vae=self.vae.dac)

    def text_ids(self, caption, bpm=None, timesignature=None, keyscale=None, duration=None,
                 instruction=INSTRUCTION):
        prompt = "# Instruction\n%s\n\n# Caption\n%s\n\n# Metas\n%s%s\n" % (
            instruction, caption, metas(bpm, timesignature, keyscale, duration), EOT)
        return self.tok.encode(prompt)[:255] + [self.eot]

    def lyric_ids(self, lyrics, language="en"):
        return self.tok.encode("# Languages\n%s\n\n# Lyric\n%s%s" % (language, lyrics, EOT))[:2047] + [self.eot]

    def embed_rows(self, ids):
        rows = {t: self.emb.get_row("embed_tokens.weight", int(t)) for t in set(ids)}
        return np.stack([rows[t] for t in ids]).astype(np.float32)

    def encode_text(self, ids):
        """The text encoder's last hidden state (len(ids), 1024)."""
        return self.text.hidden(np.asarray(ids)[None], [len(ids)])[0][0]

    def schedule(self, steps=8, shift=1.0):
        t = [1.0 - i / steps for i in range(steps)]
        if shift != 1.0:
            t = [shift * x / (1.0 + (shift - 1.0) * x) for x in t]
        return t

    def denoise(self, noise, context, Te, steps=8, shift=1.0, dcw=True, keep=None):
        """The latents (frames, 64) of the Euler steps from noise."""
        ts = self.schedule(steps, shift)
        x = noise.astype(np.float32)
        for i, t in enumerate(ts):
            v = self.dit.velocity(x, t, context, Te)
            if keep is not None:
                keep.append((x.copy(), t, v))
            y = (x - v * np.float32(t)).astype(np.float32)
            if i == len(ts) - 1:
                return y
            x = (x - v * np.float32(t - ts[i + 1])).astype(np.float32)
            if dcw:
                x = dcw_double(x, y, t)
        return x

    def generate(self, caption, lyrics="[Instrumental]", duration=None, bpm=None, keyscale=None,
                 timesignature=None, language=None, seed=0, noise=None, steps=8, normalize_db=-1.0,
                 thinking=True, lm_language=False, keep=None):
        """The audio (2, samples) at 48 kHz of a caption and lyrics; noise
        (frames, 64) in place of the seed's (NumPy's normal). thinking: the
        planner LM first (the authors' default): its metadata fill what the
        user did not give (its caption replaces the caption; its language does
        not reach the DiT, as in the authors' code, unless lm_language: the
        lyrics' language is the user's, else "unknown"), its codes the DiT's
        source latents, the cover instruction. keep (a dict) gets the plan."""
        with RESIDENCY.use(*self.runners().values()) as use:
            instruction, hints = INSTRUCTION, None
            if thinking and self.planner is not None:
                from reclank.models.acestep.planner import INSTRUCTION as COVER
                user = dict(bpm=bpm, keyscale=keyscale, timesignature=timesignature,
                            duration=int(duration) if duration else None)
                meta, codes = self.planner.plan(caption, lyrics, user, duration=duration, seed=seed)
                if keep is not None:
                    keep.update(meta=meta, codes=codes)
                caption = meta.get("caption", caption)
                bpm, keyscale = bpm or meta.get("bpm"), keyscale or meta.get("keyscale")
                timesignature = timesignature or meta.get("timesignature")
                duration = len(codes) / 5.0
                instruction = COVER
                if language is None and lm_language and meta.get("language") not in (None, "", "unknown"):
                    language = meta["language"]
                hints = self.dit.detokenize(codes)
            use.done(self.planner.lm.runner if self.planner else None)
            duration = duration or 30.0
            language = language or "unknown"
            frames = int(round(duration * FPS))
            frames += frames % 2
            text = self.encode_text(self.text_ids(caption, bpm, timesignature, keyscale, duration, instruction))
            use.done(self.text.runner)
            lyric = self.embed_rows(self.lyric_ids(lyrics, language))
            Te = self.dit.condition(text, lyric).shape[0]
            if noise is None:
                noise = np.random.default_rng(seed).standard_normal((frames, self.dit.latent)).astype(np.float32)
            lat = self.denoise(noise, self.dit.context(frames, hints), Te, steps)
            use.done(self.dit.runner)
            audio = self.vae.decode_tiled(lat)
            if normalize_db is not None:
                peak = np.abs(audio).max()
                if peak > 0:
                    audio = audio * np.float32(10 ** (normalize_db / 20) / peak)
        return audio
