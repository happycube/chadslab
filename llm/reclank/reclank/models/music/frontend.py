"""The host's part of the music embeddings (docs/MUSIC_EMBED_PLAN.md), in
NumPy alone: the decode of a file (ffmpeg), its clips of 10 s, MuQ's mel and
CLAP's mel and patches. It imports nothing of reclank's runtime (no OpenMP,
no CUDA), so a worker process of the library tool (scripts/embed_music.py)
can make the inputs of the GPU programs beside them; the GIL of one process
held the threads' FFTs one at a time (numpy's FFT keeps it).

Ported from transformers (src/transformers/audio_utils.py, Copyright 2023 The
HuggingFace Inc. team and the librosa & torchaudio authors;
models/clap/feature_extraction_clap.py, Copyright 2023 The HuggingFace Inc.
team; Apache License 2.0). See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

MUQ_RATE, CLAP_RATE = 24000, 48000
CLIP_S = 10
MIN_LAST_S = 5          # a shorter remainder after the last whole clip is left out
SPEC, RATIO, CLAP_MELS = 256, 4, 64     # HTSAT's image (256 x 256) of 4 chunks of frames


def tool_env():
    """The environment of ffmpeg and ffprobe: reclank's own without
    OMP_PROC_BIND and OMP_PLACES. ffmpeg links libgomp, which binds its main
    thread to the first CPU of its places: every ffmpeg of a library ran on
    CPU 0 (8 at once 7.4 s, not 1.0 s)."""
    return {k: v for k, v in os.environ.items() if k not in ("OMP_PROC_BIND", "OMP_PLACES")}


def wide_cpus():
    """Give the calling thread the CPUs of the taskset again: OMP_PROC_BIND
    binds the main thread to one CPU when OpenMP starts, and the threads
    started after it (and their ffmpeg children) inherit that one CPU. The
    initializer of the decoding threads."""
    try:
        from reclank.backends.cpu import cops
        cpus = cops.place_cpus()
        if cpus:
            os.sched_setaffinity(0, cpus)
    except (OSError, ImportError, AttributeError):
        pass


def decode(src, rate, start=None, end=None):
    """Mono float32 samples at rate of an audio file (a path or the bytes of
    a file) in any format of ffmpeg; start, end: seconds (a cut)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", "%.3f" % start]
    if end is not None:
        cmd += ["-to", "%.3f" % end]
    data = None
    if isinstance(src, (bytes, bytearray)):
        cmd += ["-i", "pipe:0"]
        data = bytes(src)
    else:
        cmd += ["-i", str(src)]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(rate), "pipe:1"]
    r = subprocess.run(cmd, input=data, capture_output=True, env=tool_env())
    if r.returncode != 0:
        raise ValueError("ffmpeg cannot decode the audio: %s" % r.stderr.decode("utf-8", "replace").strip()[-300:])
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def clips(x, rate):
    """The clips (k, 10 s) of samples at rate (the rule of the module)."""
    n = CLIP_S * rate
    x = np.asarray(x, dtype=np.float32)
    if len(x) < rate:
        raise ValueError("the audio has %.2f s; a piece of music needs at least 1 s" % (len(x) / rate))
    if len(x) < n:
        return np.tile(x, -(-n // len(x)))[:n][None]
    k = len(x) // n
    out = [x[i * n:(i + 1) * n] for i in range(k)]
    if len(x) - k * n >= MIN_LAST_S * rate:
        out.append(x[-n:])
    return np.stack(out)


def _htk_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def muq_filters(sr=MUQ_RATE, n_fft=2048, n_mels=128):
    """torchaudio's melscale_fbanks (HTK, no norm, 0 to sr / 2): (n_freqs, n_mels)."""
    freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
    m = np.linspace(_htk_mel(0.0), _htk_mel(sr / 2.0), n_mels + 2)
    f = 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    df = np.diff(f)
    slopes = f[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / df[:-1]
    up = slopes[:, 2:] / df[1:]
    return np.maximum(0.0, np.minimum(down, up))


_MUQ_FB = None


def muq_mel(x, mean=6.768444971712967, std=18.417922652295623):
    """The normalized log mel (128, 1000) of a clip of 240000 samples."""
    global _MUQ_FB
    if _MUQ_FB is None:
        _MUQ_FB = muq_filters()
    x = np.asarray(x, dtype=np.float64)
    p = 2048 // 2
    xp = np.pad(x, (p, p), mode="reflect")
    n = 1 + (len(xp) - 2048) // 240
    frames = np.lib.stride_tricks.as_strided(xp, (n, 2048), (xp.strides[0] * 240, xp.strides[0]))
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(2048) / 2048)       # periodic Hann
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    m = spec @ _MUQ_FB                                                       # (frames, mels)
    db = 10.0 * np.log10(np.maximum(m, 1e-10))
    return ((db.T[:, :-1] - mean) / std).astype(np.float32)


def _slaney_mel(f):
    f = np.asarray(f, dtype=np.float64)
    mel = 3.0 * f / 200.0
    log = f >= 1000.0
    mel[log] = 15.0 + np.log(f[log] / 1000.0) * (27.0 / np.log(6.4))
    return mel


def _slaney_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f = 200.0 * m / 3.0
    log = m >= 15.0
    f[log] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (m[log] - 15.0))
    return f


def clap_filters(sr=CLAP_RATE, n_fft=1024, n_mels=64, fmin=50.0, fmax=14000.0):
    """transformers' mel_filter_bank (Slaney scale and norm): (n_freqs, n_mels)."""
    fft = np.linspace(0, sr // 2, n_fft // 2 + 1)
    f = _slaney_hz(np.linspace(_slaney_mel(np.array([fmin]))[0], _slaney_mel(np.array([fmax]))[0], n_mels + 2))
    df = np.diff(f)
    slopes = f[None, :] - fft[:, None]
    fb = np.maximum(0.0, np.minimum(-slopes[:, :-2] / df[:-1], slopes[:, 2:] / df[1:]))
    return fb * (2.0 / (f[2:n_mels + 2] - f[:n_mels]))[None, :]


def _bicubic(n_in, n_out, A=-0.75):
    """torch's bicubic upsampling of one axis, align_corners: (n_out, n_in)."""
    M = np.zeros((n_out, n_in), np.float64)
    scale = (n_in - 1) / (n_out - 1)
    for x in range(n_out):
        real = scale * x
        i0 = int(np.floor(real))
        t = real - i0
        w = [((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A,
             ((A + 2) * t - (A + 3)) * t * t + 1,
             ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1,
             ((A * (2 - t) - 5 * A) * (2 - t) + 8 * A) * (2 - t) - 4 * A]
        for j in range(4):
            M[x, min(max(i0 - 1 + j, 0), n_in - 1)] += w[j]
    return M


_CLAP_FB = None


def clap_mel(x):
    """The log mel (1001, 64) of a clip of 480000 samples, as
    ClapFeatureExtractor's input_features."""
    global _CLAP_FB
    if _CLAP_FB is None:
        _CLAP_FB = clap_filters()
    x = np.asarray(x, dtype=np.float64)
    p = 1024 // 2
    xp = np.pad(x, (p, p), mode="reflect")
    n = 1 + (len(xp) - 1024) // 480
    frames = np.lib.stride_tricks.as_strided(xp, (n, 1024), (xp.strides[0] * 480, xp.strides[0]))
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)
    spec = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    m = np.maximum(1e-10, spec @ _CLAP_FB)
    return (10.0 * np.log10(m)).astype(np.float32)


_STRETCH = None


def clap_patches(x, bn):
    """CLAP's patches (4096, 16) of a clip of 480000 samples: the log mel,
    the BatchNorm of the bands (bn: (scale, shift)), the frames stretched
    to 1024 (bicubic), the image of 4 chunks of 256 frames, its patches of
    4 x 4 (ClapAudioEncoder before the patch embed)."""
    global _STRETCH
    if _STRETCH is None:
        _STRETCH = _bicubic(1 + CLIP_S * CLAP_RATE // 480, SPEC * RATIO).astype(np.float32)
    m = clap_mel(x) * bn[0] + bn[1]
    m = _STRETCH @ m
    img = m.reshape(RATIO, SPEC, CLAP_MELS).transpose(0, 2, 1).reshape(SPEC, SPEC)
    g = SPEC // 4
    return np.ascontiguousarray(img.reshape(g, 4, g, 4).transpose(0, 2, 1, 3).reshape(g * g, 16))
