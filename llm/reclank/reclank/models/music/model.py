"""The music embeddings (docs/MUSIC_EMBED_PLAN.md): MuQ-MuLan and LAION CLAP
side by side, text and audio in one space.

A vector is MuQ-MuLan's 512 floats next to CLAP's 512, each half L2-
normalized and the whole divided by sqrt 2 (unit length): the cosine of two
vectors is the mean of the two models' cosines. A text and a piece of music
are compared the same way (each model's text tower against its audio
tower).

Audio is cut into clips of 10 s: at 0, 10, 20 s, ...; a remainder of 5 s or
more gives one more clip, the last 10 s; a shorter one is left out. Audio
under 10 s is repeated to 10 s (as ClapFeatureExtractor's repeatpad). Each
model's vector of a piece is the mean of its clips' vectors, normalized.
MuQ-MuLan reads 24 kHz, CLAP 48 kHz: ffmpeg decodes the file at each rate.

    m = MusicEmbedder("models/music", gpu=True)
    q = m.texts(["solo piano"])                 # (1, 1024)
    a = m.audio("song.mp3", start=60, end=90)   # (1024,)
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from reclank.formats.safetensors import SafeTensors
from reclank.models.music import clap as _clap, muq as _muq
from reclank.models.music.clap import ClapAudio
from reclank.models.music.muq import MuqAudio
from reclank.models.music.frontend import CLIP_S, MIN_LAST_S, clips, decode, wide_cpus  # noqa: F401
from reclank.models.music.text import ClapText, MuqText

DIM = 1024
PARTS = ("muq", "clap")


def join(parts):
    """The joined vectors (n, 1024) of the parts {"muq": (n, 512), "clap": (n, 512)}."""
    return np.concatenate([parts[p] for p in PARTS], axis=-1) / np.float32(np.sqrt(2.0))


class MusicEmbedder:
    """The four towers of a directory with clap/ and muq-mulan/
    (scripts/convert_music.py). audio, text: "gpu", "cpu" (default: gpu if
    gpu else cpu) or "off". On the GPU the audio towers take BF12 matrices,
    the text towers float32 (text.py)."""

    def __init__(self, path, gpu=False, audio=None, text=None, cpu_threads=None):
        dev = "gpu" if gpu else "cpu"
        self.devices = {"audio": audio or dev, "text": text or dev}
        for k, v in self.devices.items():
            if v not in ("gpu", "cpu", "off"):
                raise ValueError("the device of %s must be gpu, cpu or off, not %r" % (k, v))
        cdir, mdir = os.path.join(path, "clap"), os.path.join(path, "muq-mulan")
        Wc = SafeTensors(os.path.join(cdir, "model.safetensors"))
        Wm = SafeTensors(os.path.join(mdir, "model.safetensors"))
        kw = dict(cpu_threads=cpu_threads)
        self.text_towers = self.audio_towers = None
        if self.devices["text"] != "off":
            g = self.devices["text"] == "gpu"
            self.text_towers = {"muq": MuqText(mdir, gpu=g, W=Wm, **kw), "clap": ClapText(cdir, gpu=g, W=Wc, **kw)}
        if self.devices["audio"] != "off":
            g = self.devices["audio"] == "gpu"
            self.audio_towers = {"muq": MuqAudio(mdir, gpu=g, W=Wm, **kw), "clap": ClapAudio(cdir, gpu=g, W=Wc, **kw)}
        self.rates = {"muq": _muq.RATE, "clap": _clap.RATE}
        self.pool = ThreadPoolExecutor(8, initializer=wide_cpus)

    def texts(self, texts, parts=False):
        """The vectors (len(texts), 1024) of texts (parts: the dict of each
        model's (n, 512) instead)."""
        if self.text_towers is None:
            raise ValueError("the text towers are off")
        out = {p: self.text_towers[p].embed(list(texts)) for p in PARTS}
        return out if parts else join(out)

    def samples(self, src, start=None, end=None):
        """The samples of a file at each model's rate: {"muq": 24 kHz, "clap": 48 kHz}."""
        jobs = {p: self.pool.submit(decode, src, self.rates[p], start, end) for p in PARTS}
        return {p: j.result() for p, j in jobs.items()}

    def audio_samples(self, samples, parts=False):
        """The vector (1024,) of a piece given as its samples at each model's
        rate (samples() of a file); parts: the dict of each model's (512,)."""
        if self.audio_towers is None:
            raise ValueError("the audio towers are off")
        out = {}
        for p in PARTS:
            tw = self.audio_towers[p]
            cs = clips(samples[p], self.rates[p])
            # the host's part of the clips in threads (MuQ: the programs take its clips 5 at a
            # time; CLAP: a clip at a time, beside the host's part of the clips after it)
            preps = [self.pool.submit(tw.prep, c) for c in cs]
            if p == "muq":
                hs = tw.run_many([f.result() for f in preps])
            else:
                hs = [tw.run(f.result()) for f in preps]
            v = tw.vectors(hs).mean(0)
            out[p] = v / np.linalg.norm(v)
        return out if parts else join(out)

    def audio(self, src, start=None, end=None, parts=False):
        """The vector (1024,) of a piece of an audio file (a path or the
        bytes of a file); start, end: seconds."""
        return self.audio_samples(self.samples(src, start, end), parts)
