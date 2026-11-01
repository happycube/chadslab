"""The Higgs audio codec's decoder (transformers' HiggsAudioV2TokenizerModel.decode;
the weights bundled in a Higgs TTS checkpoint): codes (T frames, 8
codebooks) to 24 kHz audio, 960 samples a frame.

Ported from transformers
(src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py;
Apache License 2.0; Copyright 2025 Boson AI and The HuggingFace Team). See
THIRD_PARTY_NOTICES.md.

    on the host, float32: each codebook's row (1024 x 64), its project_out
    (64 -> 1024, bias), summed over the codebooks; fc2 (1024 -> 256)
    the DAC decoder as a program (models/audio/dac.py): conv1 (k 7, 256 ->
    1024); five blocks of Snake, ConvTranspose1d (stride r of 8 5 4 2 3,
    kernel 2 r, padding ceil(r / 2), output padding r % 2: T r rows out) and
    three residual units; Snake, conv2 (k 7, 32 -> 1); no tanh
    (transformers' _adjust_dac_decoder). The checkpoint's bfloat16 rows on the
    GPU (KQ_BF16), float32 on the CPU.

The encoder (HiggsAudioV2TokenizerModel.encode; HiggsCodecEncoder): 24 kHz
audio (at least 1 s) to codes of 25 frames a second.

    the semantic side: torchaudio's resample to 16 kHz (on the host,
    models/audio/resample.py), 160 zeros each side; HuBERT
    (models/audio/hubert.py), the mean of its 13 hidden states, every
    second row; the semantic encoder: conv (k 3, no bias), two blocks of
    two residual units (ELU, conv k 3, ELU, conv k 1, no biases, + x) and
    a conv (k 3, bias): ENC_ELU, the convs of models/audio/conv.py
    the acoustic side: the DAC encoder (models/audio/dac.py DacEncoder:
    strides 8 5 4 2 3, 64 -> 2048 channels, conv2 to 256) of the samples,
    padded with 480 zeros each side when its frames are not the semantic
    side's
    on the host, float32: fc (1024 -> 1024) of the two sides' rows; the RVQ
    of the first 8 codebooks (the default bandwidth: 2 kbps): each one's
    project_in (1024 -> 64), the code of the nearest row (the largest
    -(|h|^2 - 2 h e + |e|^2), as transformers), its row through
    project_out off the residual
"""
from __future__ import annotations

import json
import math

import numpy as np

from reclank.models.audio.dac import DacDecoder, DacEncoder, aligned, conv_lin, tconv_lins
from reclank.models.audio.hubert import Hubert
from reclank.models.audio.resample import resample

PREFIX = "tied.embedding.modality_embeddings.0.model."
RATE, HOP = 24000, 960


class HiggsCodec:
    """The decoder of the codec in W (the TTS checkpoint's Shards) with its
    config (higgs_audio_v2_tokenizer.json). decode(codes) gives the audio."""

    def __init__(self, W, config_path, gpu=False, cpu_threads=None):
        cfg = json.load(open(config_path))
        ratios = list(cfg["acoustic_model_config"]["upsampling_ratios"])
        assert int(np.prod(ratios)) == HOP
        self.W = W
        self.config_path, self.gpu, self.cpu_threads = config_path, gpu, cpu_threads
        self.encoder = None
        f = lambda n: np.ascontiguousarray(W.get(PREFIX + n), dtype=np.float32)  # noqa: E731
        self.books = []
        i = 0
        while PREFIX + "quantizer.quantizers.%d.codebook.embed" % i in W.where:
            q = "quantizer.quantizers.%d." % i
            self.books.append((f(q + "codebook.embed"), f(q + "project_out.weight"), f(q + "project_out.bias")))
            i += 1
        self.fc2 = (f("fc2.weight"), f("fc2.bias"))
        d = "acoustic_decoder."
        conv = lambda n: conv_lin(W.get_bf16(PREFIX + n + ".weight"), f(n + ".bias"), gpu)  # noqa: E731
        snake = lambda n: (f(n + ".alpha").reshape(-1),) * 2  # noqa: E731
        blocks = []
        for b, r in enumerate(ratios):
            p = d + "block.%d." % b
            blocks.append(dict(
                snake=snake(p + "snake1"),
                up=tconv_lins(W.get_bf16(PREFIX + p + "conv_t1.weight"), f(p + "conv_t1.bias"), r, gpu),
                res=[dict(s1=snake(p + "res_unit%d.snake1" % u), c1=conv(p + "res_unit%d.conv1" % u),
                          s2=snake(p + "res_unit%d.snake2" % u), c2=conv(p + "res_unit%d.conv2" % u))
                     for u in (1, 2, 3)]))
        self.dac = DacDecoder(conv(d + "conv1"), blocks, snake(d + "snake1"), conv(d + "conv2"), gpu=gpu,
                              cpu_threads=cpu_threads)

    def latents(self, codes):
        """fc2's rows (T, 256) of codes (T, codebooks) (host, float32)."""
        codes = np.asarray(codes)
        q = np.zeros((codes.shape[0], self.books[0][1].shape[0]), np.float32)
        for i, (emb, w, b) in enumerate(self.books[:codes.shape[1]]):
            q += emb[codes[:, i]] @ w.T + b
        return (q @ self.fc2[0].T + self.fc2[1]).astype(np.float32)

    def decode(self, codes):
        """The audio (T 960,) at 24 kHz of codes (T, codebooks): the codec's
        ids (0 to 1023)."""
        return self.dac.decode(self.latents(codes))[:, 0]

    def encode(self, audio, keep=None):
        """The codes (frames, 8) of 24 kHz audio (HiggsCodecEncoder, made at
        the first call)."""
        if self.encoder is None:
            self.encoder = HiggsCodecEncoder(self.W, self.config_path, gpu=self.gpu, cpu_threads=self.cpu_threads)
        return self.encoder.encode(audio, keep=keep)


class HiggsCodecEncoder:
    """The encoder of the codec in W with its config. encode(audio) gives
    the codes."""

    def __init__(self, W, config_path, gpu=False, cpu_threads=None):
        cfg = json.load(open(config_path))
        ac = cfg["acoustic_model_config"]
        ratios = list(ac["downsampling_ratios"])
        self.rate, self.sem_rate = cfg["sample_rate"], cfg["semantic_sample_rate"]
        assert int(np.prod(ratios)) == HOP and self.rate == RATE
        self.down = int(HOP / (self.rate / self.sem_rate) / cfg["downsample_factor"])
        nbits = math.ceil(math.log2(cfg["codebook_size"]))
        self.nq = int(1000 * cfg["target_bandwidths"][-1] // (math.ceil(self.rate / HOP) * nbits))
        f = lambda n: np.ascontiguousarray(W.get(PREFIX + n), dtype=np.float32)  # noqa: E731
        raw = lambda n: W.get_bf16(PREFIX + n)  # noqa: E731
        bias = lambda n: f(n + ".bias") if PREFIX + n + ".bias" in W.where else None  # noqa: E731
        conv = lambda n: conv_lin(aligned(raw(n + ".weight")), bias(n), gpu)  # noqa: E731
        # the semantic encoder
        e = "encoder_semantic."
        self.sem_conv = conv(e + "conv")
        uk = cfg["unit_kernel_size"]
        self.sem_blocks = []
        for i, s in enumerate(cfg["strides"]):
            p = e + "conv_blocks.%d." % i
            kernel = 3 if s == 1 else 2 * s
            res = [dict(c1=conv(p + "res_units.%d.conv1" % j), c2=conv(p + "res_units.%d.conv2" % j),
                        dil=d, pad=(uk - 1) // 2 * d) for j, d in enumerate(cfg["block_dilations"])]
            self.sem_blocks.append(dict(res=res, conv=conv(p + "conv"), stride=s, pad=(kernel - 1) // 2))
        self.hubert = Hubert(W, PREFIX + "semantic_model.", cfg["semantic_model_config"], gpu=gpu,
                             cpu_threads=cpu_threads, tail=self._semantic)
        # the acoustic encoder
        d = "acoustic_encoder."
        snake = lambda n: (f(n + ".alpha").reshape(-1),) * 2  # noqa: E731
        blocks = []
        for b, r in enumerate(ratios):
            p = d + "block.%d." % b
            blocks.append(dict(
                res=[dict(s1=snake(p + "res_unit%d.snake1" % u), c1=conv(p + "res_unit%d.conv1" % u),
                          s2=snake(p + "res_unit%d.snake2" % u), c2=conv(p + "res_unit%d.conv2" % u))
                     for u in (1, 2, 3)],
                snake=snake(p + "snake1"), down=conv(p + "conv1"), r=r))
        self.acoustic = DacEncoder(conv(d + "conv1"), blocks, snake(d + "snake1"), conv(d + "conv2"), gpu=gpu,
                                   cpu_threads=cpu_threads)
        self.pad = HOP // 2
        self.fc = (f("fc.weight"), f("fc.bias"))
        self.rvq = []
        for i in range(self.nq):
            q = "quantizer.quantizers.%d." % i
            E = f(q + "codebook.embed")
            self.rvq.append((f(q + "project_in.weight"), f(q + "project_in.bias"), E, (E * E).sum(1)[None],
                             f(q + "project_out.weight"), f(q + "project_out.bias")))

    def _semantic(self, cv, F):
        """The tail of the HuBERT program: every down-th row of "hid", the
        semantic encoder, "e_sem"."""
        T = -(-F // self.down)
        w = cv.env["hid"].shape[1]
        z = lambda: np.zeros((T, w), np.float32)  # noqa: E731
        cv.env.update(sem=z(), s0=z(), s1=z(), sa=z(), sb=z(), e_sem=z())
        forms = cv.forms
        forms.append(("set", "sem", ("enc_gather", "hid", (np.arange(T) * self.down)[:, None].astype(np.int32), w)))
        cv.conv("sem", T, w, self.sem_conv, "s0")
        x, other = "s0", "s1"
        for i, B in enumerate(self.sem_blocks):
            for R in B["res"]:
                forms.append(("set", "sa", ("enc_elu", x)))
                cv.conv("sa", T, w, R["c1"], "sb", pad=R["pad"], dil=R["dil"])
                forms.append(("set", "sb", ("enc_elu", "sb")))
                cv.conv("sb", T, w, R["c2"], "sa")
                forms.append(("enc_add", x, "sa"))
            out = "e_sem" if i + 1 == len(self.sem_blocks) else other
            assert B["stride"] == 1
            cv.conv(x, T, w, B["conv"], out, stride=B["stride"], pad=B["pad"])
            x, other = out, x

    def quantize(self, emb):
        """The RVQ codes (frames, nq) of fc's rows (host, float32)."""
        r = np.ascontiguousarray(emb, dtype=np.float32)
        out = []
        for win, bin_, E, E2, wout, bout in self.rvq:
            h = r @ win.T + bin_
            dist = -((h * h).sum(1, keepdims=True) - 2 * (h @ E.T) + E2)
            idx = dist.argmax(1)
            r = r - (E[idx] @ wout.T + bout)
            out.append(idx)
        return np.stack(out, 1).astype(np.int32)

    def encode(self, audio, keep=None):
        """The codes (frames, nq) of 24 kHz audio (float32 in [-1, 1]); a
        clip under 1 s padded with zeros. keep: a dict that takes the
        stages' outputs (a check's)."""
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(x) < self.rate:
            x = np.pad(x, (0, self.rate - len(x)))
        x16 = np.pad(resample(x, self.rate, self.sem_rate), (160, 160))
        names = ("e_sem", "feat", "h0", "x", "hid", "sem") if keep is not None else ("e_sem",)
        outs = dict(zip(names, self.hubert.run(x16, outputs=names)))
        sem = outs["e_sem"]
        T = len(sem)
        if self.acoustic.lengths(len(x))[-1] != T:
            x = np.pad(x, (self.pad, self.pad))
        ac = self.acoustic.encode(x[:, None])
        assert len(ac) == T, (len(ac), T)
        emb = (np.concatenate([ac, sem], 1) @ self.fc[0].T + self.fc[1]).astype(np.float32)
        codes = self.quantize(emb)
        if keep is not None:
            keep.update(outs, wav16=x16[160:-160], e_acoustic=ac, emb=emb)
        return codes
