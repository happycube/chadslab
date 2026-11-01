"""Higgs TTS 3 (bosonai/higgs-tts-3-4b): text to speech, 24 kHz.

Ported from SGLang-Omni (sglang_omni/models/higgs_tts; Apache License 2.0;
Copyright 2025-2026 SGLang-Omni Team). See THIRD_PARTY_NOTICES.md.

    the prompt: <|tts|> <|text|> the text's tokens <|audio|> (zero-shot);
        a voice's (cloning, sglang-omni's build_prompt): <|tts|> <|ref_text|>
        its transcript's tokens <|ref_audio|> a row of each step of its
        codes with the delay pattern (T frames: T + 7 rows, BOC before a
        codebook's first frame, EOC after its last; the row the sum of the
        codebooks' rows, as a step's input) <|text|> ... <|audio|>; without
        a transcript, no <|ref_text|>
    the backbone: Qwen3-4B (36 layers) as step programs (qwen3.decode), its
        RoPE's cos and sin rounded to bfloat16 as the model was trained
        (sglang-omni's truncate_rope_to_bf16)
    a step: the normed row's logits of the 8 codebooks (the tied multi-codebook
        head: 8 x 1026 rows); sampled (temperature, top-k, top-p per
        codebook; greedy at temperature 0) through the delay pattern's state
        machine (sglang-omni's higgs_tts/sampler.py: codebook c starts at
        step c, BOC before it; after codebook 0's EOC, N - 2 more steps); the
        next row's input the sum of the codebooks' rows of the codes
    the codes: the delay undone (codebook c's row t is step t + c), BOC and
        EOC to 0; the codec's decoder (codec.py)
    a voice (voice()): the codec's encoder (codec.py) of a clip of 24 kHz
        and its transcript; a Voice keeps the codes (a server's cache)

The license of the weights (Boson Higgs TTS 3 Research and Non-Commercial
License) is the model's: reclank carries the code.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

from reclank.encoders.runtime import RESIDENCY, gpu_runners
from reclank.formats.safetensors import Shards
from reclank.models.higgs.codec import RATE, HiggsCodec
from reclank.models.qwen.tokenizer import QwenTokenizer
from reclank.models.qwen3.decode import Qwen3Decoder
from reclank.models.qwen3.dense import lin16

BOC, EOC = 1024, 1025
PLACEHOLDER = -100  # a prompt's row of a voice's codes (sglang-omni's AUDIO_PLACEHOLDER_ID)
TEXT_EMB = "tied.embedding.text_embedding.weight"
MM_EMB = "tied.embedding.modality_embeddings.0.embedding"
PAD = 32            # a prompt's rows: a multiple of PAD (fewer programs)
# the codec's config (bosonai/higgs-audio-v2-tokenizer's config.json, as
# sglang-omni's higgs_tts/configs/higgs_audio_v2_tokenizer.json)
CODEC_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codec_config.json")


def sample(logits, temperature, top_k, top_p, rng):
    """A code of each row of logits (N, V): greedy at temperature 0, else
    temperature, top-k, top-p (the top token always kept) as sglang-omni's
    sample_independent."""
    if temperature <= 1e-5:
        return logits.argmax(-1)
    lg = logits.astype(np.float64) / temperature
    if top_k and top_k > 0:
        kth = np.sort(lg, axis=-1)[:, -min(top_k, lg.shape[1])][:, None]
        lg = np.where(lg < kth, -np.inf, lg)
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-lg, axis=-1)
        sl = np.take_along_axis(lg, order, -1)
        p = np.exp(sl - sl[:, :1])
        cum = np.cumsum(p / p.sum(-1, keepdims=True), -1)
        drop = np.zeros_like(cum, bool)
        drop[:, 1:] = cum[:, :-1] > top_p
        mask = np.zeros_like(drop)
        np.put_along_axis(mask, order, drop, -1)
        lg = np.where(mask, -np.inf, lg)
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.array([rng.choice(p.shape[1], p=row) for row in p])


class Voice:
    """A voice to clone: its codes (frames, codebooks) and the transcript
    of its clip ("": none)."""

    def __init__(self, codes, text=""):
        import hashlib
        self.codes = np.asarray(codes, dtype=np.int32)
        self.text = text or ""
        # the key of its prompt prefix's keys and values (HiggsTTS.prefixes)
        self.key = hashlib.sha256(self.codes.tobytes() + self.text.encode()).hexdigest()

    def save(self, base):
        """base.codes.npy (the codes) and base.txt (the transcript)."""
        np.save(base + ".codes.npy", self.codes)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(self.text)

    @classmethod
    def load(cls, base):
        """The Voice of save(base) (no .txt: no transcript)."""
        text = open(base + ".txt", encoding="utf-8").read().strip() if os.path.exists(base + ".txt") else ""
        return cls(np.load(base + ".codes.npy"), text)


def delay(codes, N=None):
    """The rows (T + N - 1, N) of codes (T, N) with the delay pattern
    (sglang-omni's apply_delay_pattern): codebook c's frame t at row t + c,
    BOC before, EOC after."""
    T, n = codes.shape
    N = N or n
    out = np.full((T + N - 1, N), EOC, np.int32)
    for c in range(N):
        out[:c, c] = BOC
        out[c:c + T, c] = codes[:, c]
    return out


class HiggsTTS:
    """A Higgs TTS 3 directory (and the codec's config: default
    CODEC_CONFIG)."""

    def __init__(self, path, codec_config=None, gpu=False, cpu_threads=None, max_tokens=2600, max_prefixes=16):
        cfg = json.load(open(os.path.join(path, "config.json")))
        ae = cfg["audio_encoder_config"]
        assert ae["encoder_type"] == "discrete" and ae["use_delay_pattern"] and ae["tie_word_embeddings"]
        self.N, self.V = ae["num_codebooks"], ae["vocab_size"]
        self.W = W = Shards(path)
        self.gpu = bool(gpu)
        self.tok = QwenTokenizer(os.path.join(path, "tokenizer.json"))
        self.sp = {t: self.tok.special[t] for t in ("<|tts|>", "<|text|>", "<|audio|>", "<|ref_audio|>")}
        self.sp["<|ref_text|>"] = self.tok.special.get("<|ref_text|>")     # newer checkpoints
        head = lin16(W, MM_EMB, gpu, bias=False)
        self.lm = Qwen3Decoder(W, cfg["text_config"], "body.layers.", "body.norm.weight", {"logits": head},
                               gpu=gpu, cpu_threads=cpu_threads, max_tokens=max_tokens, rope_bfloat16=True)
        self.mm = np.ascontiguousarray(W.get(MM_EMB + ".weight"), dtype=np.float32)     # (N V, width)
        self.offsets = np.arange(self.N) * self.V
        self.codec = HiggsCodec(W, codec_config or CODEC_CONFIG, gpu=gpu, cpu_threads=cpu_threads)
        # the keys and values of cloned voices' prompt prefixes (<|tts|> to the voice's last code
        # row; about 0.3 MB a row), most recent last: a voice's next prompt prefills its text alone
        self.prefixes = collections.OrderedDict()
        self.max_prefixes = max_prefixes

    def runners(self):
        """The GPU runners (name: _Runner): the LM, the codec's decoder, and its encoder's once
        made (the first cloned voice)."""
        enc = self.codec.encoder
        return gpu_runners(self.gpu, lm=self.lm, codec=self.codec.dac, hubert=enc.hubert if enc else None,
                           acoustic=enc.acoustic if enc else None)

    def prompt(self, text, voice=None):
        """The ids of a text's prompt (PLACEHOLDER at the rows of a voice's
        codes)."""
        sp = self.sp
        ids = [sp["<|tts|>"]]
        if voice is not None:
            if voice.text and sp["<|ref_text|>"] is not None:
                ids += [sp["<|ref_text|>"]] + self.tok.encode(voice.text)
            ids += [sp["<|ref_audio|>"]] + [PLACEHOLDER] * (len(voice.codes) + self.N - 1)
        return ids + [sp["<|text|>"]] + self.tok.encode(text) + [sp["<|audio|>"]]

    def voice(self, audio, text=""):
        """The Voice of a clip (24 kHz float32) and its transcript."""
        return Voice(self.codec.encode(audio), text)

    def embed_text(self, ids):
        rows = {t: self.W.get_row(TEXT_EMB, int(t)) for t in set(ids)}
        return np.stack([rows[t] for t in ids]).astype(np.float32)

    def embed_codes(self, codes):
        """The input row of a step's codes (N,): the sum of their rows."""
        return self.mm[np.asarray(codes) + self.offsets].sum(0, dtype=np.float32)[None]

    def prefill(self, ids, voice=None, reuse=True):
        """Run the prompt (a voice's codes at its PLACEHOLDER rows); the
        logits (N, V) of its last row. A voice's prefix (to its last code row)
        is kept after its first prompt (save_prefix) and put back for the
        next (load_prefix): the rows of the text alone run then, in a program
        of at least PAD rows (the products' path of a whole prompt: the same
        bits). reuse False: the whole prompt."""
        ids = np.asarray(ids)
        hole = ids == PLACEHOLDER
        P = int(np.nonzero(hole)[0][-1]) + 1 if hole.any() else 0
        start = 0
        if P and reuse and voice.key in self.prefixes:
            self.lm.load_prefix(self.prefixes[voice.key])
            self.prefixes.move_to_end(voice.key)
            start = P
        L = len(ids) - start
        r = -(-L // PAD) * PAD
        x = np.zeros((r, self.lm.width), np.float32)
        part, h = ids[start:], hole[start:]
        x[:L][~h] = self.embed_text(part[~h])
        if h.any():
            d = delay(voice.codes, self.N)
            assert len(d) == h.sum()
            x[:L][h] = self.mm[d + self.offsets].sum(1, dtype=np.float32)
        pos = np.full(r, -1)
        pos[:L] = np.arange(start, len(ids))
        lg, = self.lm.step(x, pos[None], outputs=("logits",))
        if P and reuse and not start:
            self.prefixes[voice.key] = self.lm.save_prefix(P)
            while len(self.prefixes) > self.max_prefixes:
                self.prefixes.popitem(last=False)
        return lg[L - 1].reshape(self.N, self.V)

    def codes(self, text, voice=None, temperature=0.8, top_k=30, top_p=0.8, seed=None, max_frames=2048,
              forced=None, keep=None, reuse=True):
        """The delayed codes (steps, N) of a text (in a Voice's voice).
        forced: the codes of each step to feed (a check's teacher forcing;
        their logits to keep). The LM stays pinned from the prompt to the
        last step (its caches on the GPU)."""
        ids = self.prompt(text, voice)
        rng = np.random.default_rng(seed)
        with RESIDENCY.use(*gpu_runners(self.gpu, lm=self.lm).values()):
            lg = self.prefill(ids, voice, reuse)
            pos = len(ids)
            out, delay, countdown = [], 0, None
            limit = min(max_frames, self.lm.max_tokens - pos)
            for step in range(limit):
                if keep is not None:
                    keep.append(lg)
                c = np.array(forced[step]) if forced is not None else sample(lg, temperature, top_k, top_p, rng)
                done = False
                if forced is None:
                    if delay < self.N:
                        if delay + 1 < self.N:
                            c[delay + 1:] = BOC
                        delay += 1
                    elif countdown is not None:
                        countdown -= 1
                        done = countdown <= 0
                    elif int(c[0]) == EOC:
                        countdown = self.N - 2
                out.append(c.astype(np.int32))
                if done or (forced is not None and step + 1 == len(forced)):
                    break
                lg = self.lm.step(self.embed_codes(c), [[pos]], outputs=("logits",))[0][0].reshape(self.N, self.V)
                pos += 1
        return np.stack(out)

    def undelay(self, delayed):
        """The codec's codes (T, N) of delayed codes: codebook c's row t is
        step t + c; BOC and EOC to 0."""
        rows = len(delayed) - (self.N - 1)
        raw = np.stack([delayed[c:c + rows, c] for c in range(self.N)], 1)
        return np.where(raw >= self.V - 2, 0, raw)

    def speech(self, text, voice=None, **kw):
        """The audio (float32, 24 kHz) of a text (in a Voice's voice)."""
        with RESIDENCY.use(*self.runners().values()) as use:
            codes = self.codes(text, voice, **kw)
            use.done(self.lm.runner)
            return self.codec.decode(self.undelay(codes))


__all__ = ["HiggsTTS", "PLACEHOLDER", "RATE", "Voice", "delay", "sample"]
