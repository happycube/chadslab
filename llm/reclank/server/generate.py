"""The generation server: python -m server.generate [--higgs DIR [--voices DIR]] [--acestep DIR] [--whisper DIR]
[--seedvr2 DIR] [--realesrgan DIR] [--stereo-vda DIR] [--rife DIR] [--port 8084].

Upscaling (SeedVR2-3B, Real-ESRGAN): server/upscale.py's routes, POST /v1/images/upscale and the
jobs of POST /v1/videos/upscale (read with GET /v1/videos/{id} as the video jobs).

The models of docs/GENERATION_PLAN.md, each loaded by its flag; GET /v1/models, /health.

Speech (Higgs TTS 3, reclank/models/higgs; its weights' license is research
and non-commercial) in the API of OpenAI:

    POST /v1/audio/speech  {"input": "text", "model": "higgs-tts-3",
                            "voice": "default" (a zero-shot voice) | a name of
                                --voices DIR (NAME.wav, or .flac .mp3 .ogg,
                                and its transcript NAME.txt; or a voice saved
                                by name) | {"audio": base64 of a clip (any
                                format of ffmpeg; at most 30 s), "text": its
                                transcript (optional, better), "name": save it
                                in --voices by this name (NAME.wav, NAME.txt,
                                NAME.codes.npy; "overwrite": true to replace
                                one)}: voice cloning,
                            "response_format": "mp3" (default) | "wav" | "opus" |
                                "aac" | "flac" | "pcm" (16-bit 24 kHz mono),
                            "speed": 0.25 to 4 (ffmpeg's atempo),
                            extensions: "temperature" (0.8; 0: greedy), "top_k"
                            (30; 50 with a cloned voice, the model card's), "top_p"
                            (0.8), "seed", "max_seconds" (80)}

    GET  /v1/audio/voices  the names of --voices

The audio is 24 kHz mono; ffmpeg makes the formats other than wav and pcm.
A cloned voice's codes (the codec's encoder of its clip) stay in a cache of
the last 32 voices (by the clip's bytes and the transcript), and the keys and
values of its prompt's prefix in one of 16 (models/higgs/tts.py: its next
request prefills the text alone). The model's
license forbids cloning a voice without its speaker's consent.

Music (ACE-Step 1.5 turbo with its 5 Hz planner LM, reclank/models/acestep, MIT):

    POST /v1/music/generations  {"prompt": "a caption of the music", "lyrics":
                                 "[Verse] ..." (default "[Instrumental]"),
                                 "duration": seconds (30, at most 240), "bpm",
                                 "key": "D major", "time_signature", "language":
                                 the lyrics' (default: the planner's with
                                 "lm_language": true, else "unknown"), "seed",
                                 "thinking": true (the planner LM first; false:
                                 the DiT alone), "response_format": "wav" | "mp3"
                                 | "flac" | "opus" | "aac" (default wav)}

answers the audio itself (48 kHz stereo, the peak at -1 dB) at once: a
piece takes about a tenth of its length on the 3090.

2D to stereo 3D for VR players (MoGe-2, DA3MONO, VDA-Small; reclank/models/depth; --stereo-moge,
--stereo-da3, --stereo-vda): POST /v1/images/stereo (an image to side-by-side or over-under) and
POST /v1/videos/stereo (a video file to a stereo MP4, a video job): server/stereo.py.

Frame interpolation (Practical-RIFE 4.25, reclank/models/vfi; --rife DIR): POST
/v1/videos/interpolate (a video file, or "video_id" of a completed video job, to 2x, 4x or a target
rate; scene cuts kept; the audio copied), a video job: server/interpolate.py.

The video jobs (upscales, stereo videos, interpolations):

    GET    /v1/videos  the jobs; GET /v1/videos/{id}: a job's status
                        ("queued", "in_progress", "completed", "failed") and
                        progress (0 to 100)
    GET    /v1/videos/{id}/content  the video of a completed job
    DELETE /v1/videos/{id}  cancel a job (queued or running) or delete its video

One job runs at a time, the rest wait in a queue. The jobs (ID.json) and
their videos live in --video-dir, not in RAM: the completed ones stay over a
restart.

Speech to text (Whisper large-v3, reclank/models/whisper, Apache 2.0; --whisper
DIR) in the API of OpenAI, the body multipart/form-data:

    POST /v1/audio/transcriptions  file: the audio (any format of ffmpeg),
                                   "model": "whisper-large-v3" (or "whisper-1"),
                                   "language": ISO-639-1 ("sk"; none: Whisper's
                                   detection over the first 30 s), "prompt": a
                                   previous text (before each window, as
                                   transformers'; its last 223 tokens),
                                   "response_format": "json" ({"text"}, default)
                                   | "text" | "srt" | "vtt" | "verbose_json"
                                   (task, language, duration, text, segments:
                                   id, seek, start, end, text, tokens,
                                   temperature, avg_logprob, compression_ratio,
                                   no_speech_prob), "temperature": 0 (greedy
                                   only), "timestamp_granularities[]": "segment"
    POST /v1/audio/translations    the same fields: English text

Greedy, as transformers' WhisperForConditionalGeneration.generate: audio of
at most 30 s for json and text is one window without timestamps; longer audio,
and srt, vtt and verbose_json, take the timestamp tokens and transformers'
sequential long-form (each window of 30 s starts at the end of the last whole
segment of the one before). No temperature fallback and no no-speech or
log-probability thresholds (transformers' defaults): music can give Whisper's
stock lines, which verbose_json's no_speech_prob and avg_logprob help filter.
The body's limit is --max-body-mb (64 MB: about an hour of MP3 at 128 kbps).
5 minutes of Slovak talk take 11-13 s on the 3090 (about 3 minutes an hour),
4.3-4.7 minutes on 12 CPU threads (--cpu); the GPU holds 4.9 GB.

Requests run one at a time.

The GPU's memory (encoders.runtime.RESIDENCY): every model's parts are loaded
into host RAM at the start; on the GPU a part's weights and buffers are
uploaded at its first run and stay there until another part needs the room:
then the parts used least recently go (their host copies stay; the next run
uploads them again), within --gpu-budget GB (default:
the card's, less a reserve of 512 MB). A request's parts stay while it runs
(each until its turn is over). --gpu-lock ID,... (--pin) keeps whole models on
the GPU (no ids: all). The other reclank servers on the GPU are asked
for room when this one's parts are not enough, and it gives its own when idle
(server/gpushare.py; --gpu-peers off: neither, the budget the free memory at
the start).
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
import zlib

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import http as H  # noqa: E402



def png(img):
    """The PNG bytes of an image (H, W, 3) or RGBA (H, W, 4) uint8 (zlib, no
    filter)."""
    h, w, ch = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if ch == 4 else 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def images_body(data):
    """The response body of an images route (upscale)."""
    return {"created": int(time.time()), "data": data}


FORMATS = {"mp3": ("audio/mpeg", ["-f", "mp3", "-b:a", "128k"]), "opus": ("audio/ogg", ["-f", "ogg", "-c:a", "libopus"]),
           "aac": ("audio/aac", ["-f", "adts", "-c:a", "aac"]), "flac": ("audio/flac", ["-f", "flac"]),
           "wav": ("audio/wav", None), "pcm": ("audio/pcm", None)}


def wav_bytes(pcm16, rate):
    """A WAV file of 16-bit mono samples."""
    data = pcm16.astype("<i2").tobytes()
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, rate, 2 * rate, 2, 16) + b"data" + struct.pack("<I", len(data)) + data)


class Speech:
    """Higgs TTS 3 on a device; voices: a directory of named voices."""

    MAX_REF_S = 30
    CACHE = 32

    def __init__(self, path, gpu, cpu_threads, model_id, voices=None):
        from reclank.models.higgs.tts import RATE, HiggsTTS
        self.tts = HiggsTTS(path, gpu=gpu, cpu_threads=cpu_threads)
        self.rate, self.model_id = RATE, model_id
        self.voices_dir = voices
        self.cache = collections.OrderedDict()      # (sha256 of the clip, transcript): Voice

    def runners(self):
        """The GPU runners (the codec's encoder's once a voice is cloned)."""
        return self.tts.runners()

    @staticmethod
    def _name_ok(name):
        return isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name) and not name.startswith(".")

    def names(self):
        """The voices of --voices: NAME.codes.npy (a saved voice) or a clip."""
        if not self.voices_dir or not os.path.isdir(self.voices_dir):
            return []
        out = set()
        for f in os.listdir(self.voices_dir):
            for ext in (".codes.npy", ".wav", ".flac", ".mp3", ".ogg"):
                if f.endswith(ext) and self._name_ok(f[:-len(ext)]):
                    out.add(f[:-len(ext)])
        return sorted(out)

    def named(self, name):
        """A Voice of --voices (its saved codes), or the clip's bytes and the
        transcript of one."""
        if not self.voices_dir or not self._name_ok(name):
            raise ValueError("voice must be \"default\", a name of --voices, or {\"audio\": base64, \"text\": ...}")
        base = os.path.join(self.voices_dir, name)
        if os.path.exists(base + ".codes.npy"):
            from reclank.models.higgs.tts import Voice
            return Voice.load(base), None
        for ext in (".wav", ".flac", ".mp3", ".ogg"):
            path = os.path.join(self.voices_dir, name + ext)
            if os.path.exists(path):
                txt = os.path.join(self.voices_dir, name + ".txt")
                text = open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else ""
                return open(path, "rb").read(), text
        raise ValueError("no voice %r in --voices" % name)

    def voice(self, spec):
        """The Voice of a request's "voice" (None: zero-shot), from the cache
        or the codec's encoder of its clip (ffmpeg decodes it to 24 kHz)."""
        import subprocess
        if spec is None or spec in ("default", "alloy"):
            return None
        name = None
        if isinstance(spec, str):
            data, text = self.named(spec)
            if text is None:                   # a saved voice: its codes
                return data
        elif isinstance(spec, dict) and isinstance(spec.get("audio"), str):
            name = spec.get("name")
            if name is not None:
                if not self.voices_dir:
                    raise ValueError("voice.name: saving a voice needs the server's --voices DIR")
                if not self._name_ok(name):
                    raise ValueError("voice.name must be 1 to 64 of A-Z a-z 0-9 _ . - (not starting with .)")
                if name in self.names() and not spec.get("overwrite"):
                    raise ValueError("voice %r exists (\"overwrite\": true replaces it)" % name)
            try:
                data = base64.b64decode(spec["audio"], validate=True)
            except ValueError:
                raise ValueError("voice.audio must be base64") from None
            text = spec.get("text") or ""
            if not isinstance(text, str) or len(text) > 2000:
                raise ValueError("voice.text must be a string of at most 2000 characters")
        else:
            raise ValueError("voice must be \"default\", a name of --voices, or {\"audio\": base64, \"text\": ...}")
        key = (hashlib.sha256(data).hexdigest(), text.strip())
        v = self.cache.get(key)
        audio = None
        if v is None or name is not None:
            env = {k: v for k, v in os.environ.items() if k not in ("OMP_PROC_BIND", "OMP_PLACES")}
            cmd = ["ffmpeg", "-v", "error", "-i", "-", "-ac", "1", "-ar", str(self.rate), "-f", "f32le", "-"]
            out = subprocess.run(cmd, input=data, capture_output=True, env=env)
            if out.returncode != 0 or not out.stdout:
                raise ValueError("voice.audio: ffmpeg could not decode it")
            audio = np.frombuffer(out.stdout, np.float32)
            if len(audio) > self.MAX_REF_S * self.rate:
                raise ValueError("a voice's clip must be at most %d s" % self.MAX_REF_S)
        if v is None:
            v = self.tts.voice(audio, key[1])
        self.cache[key] = v
        self.cache.move_to_end(key)
        while len(self.cache) > self.CACHE:
            self.cache.popitem(last=False)
        if name is not None:
            # NAME.wav (the clip at 24 kHz), NAME.txt, NAME.codes.npy: "voice": NAME from now on
            os.makedirs(self.voices_dir, exist_ok=True)
            base = os.path.join(self.voices_dir, name)
            for ext in (".flac", ".mp3", ".ogg"):
                if os.path.exists(base + ext):
                    os.remove(base + ext)
            with open(base + ".wav", "wb") as f:
                f.write(wav_bytes((np.clip(audio, -1, 1) * 32767).astype("<i2"), self.rate))
            v.save(base)
        return v

    def speech(self, body):
        import subprocess
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("input must be a non-empty string")
        if len(text) > 4096:
            raise ValueError("input must be at most 4096 characters")
        fmt = body.get("response_format") or "mp3"
        if fmt not in FORMATS:
            raise ValueError("response_format must be one of %s" % ", ".join(FORMATS))
        speed = float(body.get("speed") or 1.0)
        if not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be 0.25 to 4")
        temperature = float(body.get("temperature", 0.8))
        seed = body.get("seed")
        frames = int(25 * min(float(body.get("max_seconds") or 80), 80))
        voice = self.voice(body.get("voice"))
        audio = self.tts.speech(text, voice, temperature=temperature, top_k=int(body.get("top_k", 30 if voice is None else 50)),
                                top_p=float(body.get("top_p", 0.8)), seed=None if seed is None else int(seed),
                                max_frames=frames)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        ctype, args = FORMATS[fmt]
        if args is None and speed == 1.0:
            return (wav_bytes(pcm, self.rate) if fmt == "wav" else pcm.tobytes()), ctype, len(audio) / self.rate
        cmd = ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(self.rate), "-ac", "1", "-i", "-"]
        if speed != 1.0:
            cmd += ["-filter:a", "atempo=%g" % speed]
        cmd += (args if args is not None else (["-f", "wav"] if fmt == "wav" else ["-f", "s16le"])) + ["-"]
        # ffmpeg links libgomp: without OMP_PROC_BIND it is not bound to one CPU
        env = {k: v for k, v in os.environ.items() if k not in ("OMP_PROC_BIND", "OMP_PLACES")}
        out = subprocess.run(cmd, input=pcm.tobytes(), capture_output=True, env=env)
        if out.returncode != 0:
            raise RuntimeError("ffmpeg: " + out.stderr.decode(errors="replace")[-300:])
        return out.stdout, ctype, len(audio) / self.rate


def encode_audio(audio, rate, fmt, speed=1.0):
    """The bytes and content type of audio (channels, samples) float32 in a
    format of FORMATS (ffmpeg for all but wav and pcm)."""
    import subprocess
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").T.copy()          # (samples, channels)
    ch = audio.shape[0]
    ctype, args = FORMATS[fmt]
    if args is None and speed == 1.0:
        if fmt == "pcm":
            return pcm.tobytes(), ctype
        data = pcm.tobytes()
        return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " +
                struct.pack("<IHHIIHH", 16, 1, ch, rate, 2 * ch * rate, 2 * ch, 16) + b"data" +
                struct.pack("<I", len(data)) + data), ctype
    cmd = ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(rate), "-ac", str(ch), "-i", "-"]
    if speed != 1.0:
        cmd += ["-filter:a", "atempo=%g" % speed]
    cmd += (args if args is not None else (["-f", "wav"] if fmt == "wav" else ["-f", "s16le"])) + ["-"]
    # ffmpeg links libgomp: without OMP_PROC_BIND it is not bound to one CPU
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_PROC_BIND", "OMP_PLACES")}
    out = subprocess.run(cmd, input=pcm.tobytes(), capture_output=True, env=env)
    if out.returncode != 0:
        raise RuntimeError("ffmpeg: " + out.stderr.decode(errors="replace")[-300:])
    return out.stdout, ctype


class Music:
    """ACE-Step 1.5 turbo on a device."""

    def __init__(self, path, gpu, cpu_threads, model_id):
        from reclank.models.acestep.pipeline import AceStep
        self.ace = AceStep(path, gpu=gpu, cpu_threads=cpu_threads)
        self.model_id = model_id

    @property
    def sample_rate(self):
        return self.ace.vae.rate

    def runners(self):
        return self.ace.runners()

    def music(self, body):
        caption = body.get("prompt", body.get("caption"))
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("prompt must be a non-empty string (a caption of the music)")
        lyrics = body.get("lyrics") or "[Instrumental]"
        if not isinstance(lyrics, str) or len(lyrics) > 8000:
            raise ValueError("lyrics must be a string of at most 8000 characters")
        duration = body.get("duration")
        if duration is not None and not 10 <= float(duration) <= 240:
            raise ValueError("duration must be 10 to 240 seconds (none: the planner's choice)")
        fmt = body.get("response_format") or "wav"
        if fmt not in FORMATS or fmt == "pcm":
            raise ValueError("response_format must be one of wav, mp3, flac, opus, aac")
        bpm = body.get("bpm")
        seed = body.get("seed")
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little")
        audio = self.ace.generate(caption, lyrics, float(duration) if duration is not None else None,
                                  bpm=int(bpm) if bpm else None, keyscale=body.get("key"),
                                  timesignature=body.get("time_signature"), language=body.get("language"),
                                  seed=seed, thinking=bool(body.get("thinking", True)),
                                  lm_language=bool(body.get("lm_language", False)))
        data, ctype = encode_audio(audio, self.ace.vae.rate, fmt)
        return data, ctype, audio.shape[1] / self.ace.vae.rate, seed


def multipart(body, ctype):
    """The fields of a multipart/form-data body: {name: [(bytes, filename or
    None)]}, each part's headers read by the stdlib's email parser."""
    import email
    from email.message import Message
    m = Message()
    m["content-type"] = ctype or ""
    boundary = m.get_param("boundary")
    if m.get_content_type() != "multipart/form-data" or not boundary:
        raise ValueError("the body must be multipart/form-data (with a boundary)")
    delim = b"--" + boundary.encode("latin-1")
    pos = body.find(delim)
    if pos < 0:
        raise ValueError("the body has no part (no boundary)")
    out = {}
    pos += len(delim)
    while not body.startswith(b"--", pos):           # the closing delimiter
        if not body.startswith(b"\r\n", pos):
            raise ValueError("a malformed multipart body")
        end = body.find(b"\r\n" + delim, pos + 2)
        if end < 0:
            raise ValueError("a malformed multipart body (no closing boundary)")
        head, sep, data = body[pos + 2:end].partition(b"\r\n\r\n")
        h = email.message_from_bytes(head + b"\r\n\r\n")
        name = h.get_param("name", header="content-disposition")
        if not sep or name is None:
            raise ValueError("a part of the body has no Content-Disposition name")
        out.setdefault(name, []).append((data, h.get_filename()))
        pos = end + 2 + len(delim)
    return out


def _clock(t, sep):
    """HH:MM:SS,mmm (SRT; sep ".": WebVTT) of t seconds."""
    ms = int(round(t * 1000))
    return "%02d:%02d:%02d%s%03d" % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, sep, ms % 1000)


class Transcriber:
    """Whisper large-v3 on a device: speech to text, or to English text."""

    FORMATS = ("json", "text", "srt", "vtt", "verbose_json")

    def __init__(self, path, gpu, cpu_threads, model_id):
        from reclank.models.whisper.decoder import Whisper
        self.wh = Whisper(path, gpu=gpu, cpu_threads=cpu_threads, batch=1)
        self.model_id = model_id

    def runners(self):
        return self.wh.runners()

    def request(self, form, task):
        """The audio (16 kHz mono float32: ffmpeg) and the options of a form
        (multipart): a dict for run(). ValueError: a bad request."""
        from reclank.models.music.frontend import decode
        wh = self.wh

        def field(name, default=None):
            v = form.get(name)
            if not v:
                return default
            try:
                return v[0][0].decode("utf-8").strip()
            except UnicodeDecodeError:
                raise ValueError("%s must be UTF-8 text" % name) from None
        if not form.get("file") or not form["file"][0][0]:
            raise ValueError("file is required: the audio (any format of ffmpeg)")
        if field("model") not in (None, self.model_id, "whisper-1"):
            raise ValueError("model must be %s" % self.model_id)
        fmt = field("response_format") or "json"
        if fmt not in self.FORMATS:
            raise ValueError("response_format must be one of %s" % ", ".join(self.FORMATS))
        lang = field("language") or None
        if lang is not None:
            lang = lang.lower()
            if lang not in wh.lang_ids:
                raise ValueError("language must be an ISO-639-1 code of Whisper's languages (such as en, sk), "
                                 "or none (detected)")
        temp = field("temperature") or "0"
        try:
            temp = float(temp)
        except ValueError:
            raise ValueError("temperature must be a number") from None
        if temp != 0:
            raise ValueError("temperature must be 0: this server decodes greedily (no sampling, no fallback)")
        grains = [v.decode("utf-8", "replace").strip() for k in ("timestamp_granularities[]", "timestamp_granularities")
                  for v, _ in form.get(k, [])]
        if any(g != "segment" for g in grains):
            raise ValueError("timestamp_granularities must be [\"segment\"] (no word timestamps)")
        if (field("stream") or "false").lower() not in ("false", "0"):
            raise ValueError("stream is not supported")
        prompt = None
        text = field("prompt")
        if text:
            prompt = wh.tok.encode(" " + text)
            if any(t in wh.tok.special for t in prompt):
                raise ValueError("prompt must not hold special tokens (<|...|>)")
        audio = decode(form["file"][0][0], 16000)
        if not len(audio):
            raise ValueError("file has no audio")
        return dict(audio=audio, task=task, fmt=fmt, lang=lang, prompt=prompt)

    def run(self, req):
        """The body, its content type and the audio's seconds of a request:
        segments() (timestamps for srt, vtt and verbose_json, and past 30 s:
        long-form), in the response_format."""
        fmt = req["fmt"]
        out = self.wh.segments(req["audio"], lang=req["lang"], task=req["task"], prompt=req["prompt"],
                               timestamps=fmt in ("srt", "vtt", "verbose_json"))
        text, segs = out["text"].strip(), out["segments"]
        if fmt == "json":
            data, ctype = json.dumps({"text": text}, ensure_ascii=False), "application/json"
        elif fmt == "text":
            data, ctype = text + "\n", "text/plain; charset=utf-8"
        elif fmt == "verbose_json":
            from reclank.models.whisper.decoder import LANGUAGES
            data = json.dumps({
                "task": req["task"], "language": LANGUAGES.get(out["language"], out["language"]),
                "duration": round(out["duration"], 3), "text": text,
                "segments": [{"id": i, "seek": s["seek"], "start": round(s["start"], 2), "end": round(s["end"], 2),
                              "text": s["text"], "tokens": s["tokens"], "temperature": 0.0,
                              "avg_logprob": round(s["avg_logprob"], 6),
                              "compression_ratio": round(s["compression_ratio"], 6),
                              "no_speech_prob": round(s["no_speech_prob"], 6)} for i, s in enumerate(segs)]},
                ensure_ascii=False)
            ctype = "application/json"
        else:
            cues = [s for s in segs if s["text"].strip()]
            sep = "," if fmt == "srt" else "."
            lines = [] if fmt == "srt" else ["WEBVTT", ""]
            for i, s in enumerate(cues):
                if fmt == "srt":
                    lines.append(str(i + 1))
                lines += ["%s --> %s" % (_clock(s["start"], sep), _clock(s["end"], sep)), s["text"].strip(), ""]
            data = "\n".join(lines) + ("\n" if not lines else "")
            ctype = "text/plain; charset=utf-8" if fmt == "srt" else "text/vtt; charset=utf-8"
        return data.encode("utf-8"), ctype, out["duration"]


class _GenHandler(H._Handler):
    server_version = "reclank-generate/1.0"

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/health", ""):
            return self._json(200, {"status": "ok"})
        if not self._authorized():
            return
        if self._admin(path, False):
            return
        if path in ("/v1/audio/voices", "/audio/voices"):
            sp = self.server.speech
            return self._json(200, {"object": "list", "data": [
                {"id": n, "object": "voice"} for n in (sp.names() if sp else [])]})
        if path in ("/v1/videos", "/videos") or path.startswith(("/v1/videos/", "/videos/")):
            return self._video_get(path)
        if path in ("/v1/models", "/models"):
            data = []
            if self.server.music:
                data.append({"id": self.server.music.model_id, "object": "model", "owned_by": "reclank",
                             "kind": "music", "sample_rate": self.server.music.sample_rate, "max_seconds": 240})
            if self.server.speech:
                data.append({"id": self.server.speech.model_id, "object": "model", "owned_by": "reclank",
                             "kind": "speech", "formats": list(FORMATS), "sample_rate": self.server.speech.rate})
            if getattr(self.server, "whisper", None):
                data.append({"id": self.server.whisper.model_id, "object": "model", "owned_by": "reclank",
                             "kind": "asr", "formats": list(Transcriber.FORMATS), "sample_rate": 16000})
            for up in getattr(self.server, "upscalers", {}).values():
                data.append(up.describe())
            if getattr(self.server, "stereo", None):
                for mid in self.server.stereo.model_ids:
                    data.append({"id": mid, "object": "model", "owned_by": "reclank", "kind": "stereo",
                                 "input": "video" if mid == "vda-small" else "image"})
            if getattr(self.server, "interp", None):
                data.append(self.server.interp.describe())
            pins = self.server.pins
            for d in data:
                d["gpu_locked"] = "*" in pins or d["id"] in pins
            return self._json(200, {"object": "list", "data": data})
        self._error(404, "no route %s" % self.path, code="not_found")

    def _managers(self):
        """The managers of /v1/videos' jobs: the upscales', the stereo videos', the interpolations'."""
        return [m for m in (getattr(self.server, "upscales", None),
                            getattr(self.server, "stereo", None), getattr(self.server, "interp", None))
                if m is not None]

    def _source(self, vid):
        """the file of the completed job vid of /v1/videos (an interpolation's input)"""
        m = self._manager(vid)
        job = m.jobs.get(vid) if m is not None else None
        if job is None:
            raise ValueError("video_id: no video %s" % vid)
        if job["status"] != "completed":
            raise ValueError("video_id: video %s is %s" % (vid, job["status"]))
        return m.content(job) if hasattr(m, "content") else m._file(vid, ".mp4")

    def _interpolate(self):
        ip = getattr(self.server, "interp", None)
        if ip is None:
            return self._error(404, "no interpolation model loaded (--rife)", code="model_not_found")
        try:
            job = ip.submit(multipart(self._raw_body(), self.headers.get("Content-Type")), self._source)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("interpolate %s queued: %s, %s" % (job["id"], job["source"],
                                                     job["fps"] and "%s fps" % job["fps"] or "x%d" % job["factor"]),
                  flush=True)
        self._json(200, job)

    def _manager(self, vid):
        """The manager of the job vid (None: no such job)."""
        return next((m for m in self._managers() if vid in m.jobs), None)

    def _video_get(self, path):
        parts = path.split("/")[2 if path.startswith("/v1/") else 1:]
        if len(parts) == 1:
            ms = self._managers()
            if not ms:
                return self._error(404, "no video model loaded (--seedvr2, --realesrgan, --stereo-vda, --rife)",
                                   code="model_not_found")
            data = []
            for m in ms:
                with m.cv:
                    data += [m.view(j) for j in reversed(m.jobs.values())]
            data.sort(key=lambda j: -j.get("created_at", 0))
            return self._json(200, {"object": "list", "data": data})
        vi = self._manager(parts[1])
        job = vi.jobs.get(parts[1]) if vi is not None else None
        if job is None or len(parts) > 3 or (len(parts) == 3 and parts[2] != "content"):
            return self._error(404, "no video %s" % "/".join(parts[1:]), code="not_found")
        if len(parts) == 2:
            return self._json(200, vi.view(job))
        if job["status"] != "completed":
            return self._error(409, "video %s is %s" % (job["id"], job["status"]), code="not_ready")
        name = vi.content(job) if hasattr(vi, "content") else vi._file(job["id"], ".mp4")
        with open(name, "rb") as f:
            self._bytes(200, f.read(), "video/x-matroska" if name.endswith(".mkv") else "video/mp4")

    def do_DELETE(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if not self._authorized():
            return
        parts = path.split("/")[2 if path.startswith("/v1/") else 1:]
        if len(parts) != 2 or parts[0] != "videos":
            return self._error(404, "no route %s" % self.path, code="not_found")
        vi = self._manager(parts[1])
        if vi is None:
            return self._error(404, "no video %s" % parts[1], code="not_found")
        out = vi.delete(parts[1])
        if out is None:
            return self._error(404, "no video %s" % parts[1], code="not_found")
        self._json(200, out)

    def do_OPTIONS(self):
        self.send_response(204)
        if getattr(self.server, "cors_origin", None):
            self._cors()
            self.send_header("Access-Control-Allow-Headers", "authorization, content-type, x-api-key")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _bytes(self, status, data, ctype):
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _music(self):
        mu = self.server.music
        if mu is None:
            return self._error(404, "no music model loaded (--acestep)", code="model_not_found")
        try:
            body = self._body()
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            if body.get("model") not in (None, mu.model_id):
                raise ValueError("model must be %s" % mu.model_id)
            t0 = time.time()
            with self.server.lock:
                data, ctype, secs, seed = mu.music(body)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("music: %.0f s of audio in %.1f s (seed %d)" % (secs, time.time() - t0, seed), flush=True)
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Seed", str(seed))
        self.end_headers()
        self.wfile.write(data)

    def _speech(self):
        sp = self.server.speech
        if sp is None:
            return self._error(404, "no speech model loaded (--higgs)", code="model_not_found")
        try:
            body = self._body()
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            if body.get("model") not in (None, sp.model_id, "tts-1", "tts-1-hd", "gpt-4o-mini-tts"):
                raise ValueError("model must be %s" % sp.model_id)
            t0 = time.time()
            with self.server.lock:
                try:
                    data, ctype, secs = sp.speech(body)
                finally:
                    _residency(self.server)     # (the codec's encoder, made at the first cloned voice)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("speech: %.1f s of audio in %.1f s" % (secs, time.time() - t0), flush=True)
        self._bytes(200, data, ctype)

    def _transcription(self, task):
        tr = getattr(self.server, "whisper", None)
        if tr is None:
            return self._error(404, "no speech-to-text model loaded (--whisper)", code="model_not_found")
        try:
            form = multipart(self._raw_body(), self.headers.get("Content-Type"))
            req = tr.request(form, task)
            t0 = time.time()
            with self.server.lock:
                data, ctype, secs = tr.run(req)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("%s: %.1f s of audio in %.1f s (%s)" % (task, secs, time.time() - t0, req["fmt"]), flush=True)
        self._bytes(200, data, ctype)

    def _upscale(self):
        from server import upscale as U
        ups = getattr(self.server, "upscalers", {})
        if not ups:
            return self._error(404, "no upscaler loaded (--seedvr2, --realesrgan)", code="model_not_found")
        try:
            f, img = U.image_request(self._raw_body(), self.headers.get("Content-Type"), multipart)
            U.check_common(f)
            model = f.get("model") or next(iter(ups))
            if model not in ups:
                raise ValueError("model must be one of %s" % ", ".join(ups))
            up = ups[model]
            t0 = time.time()
            with self.server.lock:
                opts = up.options(f, *img.shape[:2])
                try:
                    out = up.run(img[None], opts)[0]
                finally:
                    _residency(self.server)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("upscale: %s %dx%d -> %dx%d, %.1f s" % (model, img.shape[1], img.shape[0], out.shape[1], out.shape[0],
                                                          time.time() - t0), flush=True)
        self._json(200, images_body([{"b64_json": base64.b64encode(png(out)).decode("ascii"), "seed": opts["seed"],
                                      "size": "%dx%d" % (out.shape[1], out.shape[0])}]))

    def _upscale_video(self):
        ups = getattr(self.server, "upscales", None)
        if ups is None:
            return self._error(404, "no video upscaler loaded (--seedvr2, --realesrgan with "
                                    "realesr-animevideov3)", code="model_not_found")
        try:
            job = ups.submit(multipart(self._raw_body(), self.headers.get("Content-Type")))
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("upscale %s queued: %s, %s" % (job["id"], job["model"], job["input_size"]), flush=True)
        self._json(200, job)

    def _stereo(self, video):
        st = getattr(self.server, "stereo", None)
        if st is None:
            return self._error(404, "no stereo model loaded (--stereo-moge, --stereo-da3, --stereo-vda)",
                               code="model_not_found")
        try:
            ctype = self.headers.get("Content-Type") or ""
            if video:
                job = st.submit(multipart(self._raw_body(), ctype))
                if not getattr(self.server, "quiet", False):
                    print("stereo video %s queued: %s" % (job["id"], job["source"]), flush=True)
                return self._json(200, job)
            if ctype.startswith("multipart/form-data"):
                form = multipart(self._raw_body(), ctype)
                fields = {k: (v[0][0] if k == "image" else v[0][0].decode("utf-8").strip()) for k, v in form.items()}
            else:
                fields = self._body()
                if not isinstance(fields, dict):
                    raise ValueError("the body must be a JSON object")
            t0 = time.time()
            data = st.image(fields)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("stereo image: %s in %.1f s" % (data[0].get("info"), time.time() - t0), flush=True)
        self._json(200, {"created": int(time.time()), "data": data})

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if not self._authorized():
            return
        if self._admin(path, True):
            return
        if path in ("/v1/audio/speech", "/audio/speech"):
            return self._speech()
        if path in ("/v1/audio/transcriptions", "/audio/transcriptions"):
            return self._transcription("transcribe")
        if path in ("/v1/audio/translations", "/audio/translations"):
            return self._transcription("translate")
        if path in ("/v1/music/generations", "/music/generations"):
            return self._music()
        if path in ("/v1/images/upscale", "/images/upscale"):
            return self._upscale()
        if path in ("/v1/videos/upscale", "/videos/upscale"):
            return self._upscale_video()
        if path in ("/v1/videos/interpolate", "/videos/interpolate"):
            return self._interpolate()
        if path in ("/v1/images/stereo", "/images/stereo", "/v1/videos/stereo", "/videos/stereo"):
            return self._stereo(path.endswith("videos/stereo"))
        self._error(404, "no route %s" % self.path, code="not_found")


def _models(server):
    """Every model of the server (id: its object)."""
    out = {}
    for m in (server.speech, server.music, getattr(server, "whisper", None),
              getattr(server, "stereo", None), getattr(server, "interp", None)):
        if m is not None:
            out[m.model_id] = m
    out.update(getattr(server, "upscalers", {}))
    return out


class Lazy:
    """A model made at its first use (a request), not at the server's start: what the model list needs
    (model_id and the static facts: sample_rate, rate) without it; runners() empty until it is made (the GPU's sharing and residency see nothing of it before), and
    after() (the residency's labels and pins) once it is."""

    def __init__(self, make, model_id, **static):
        self._make, self._obj, self._lock = make, None, threading.Lock()
        self.model_id, self._static, self.after = model_id, static, None

    def get(self):
        if self._obj is None:
            with self._lock:
                if self._obj is None:
                    t0 = time.time()
                    print("%s: loading at its first request" % self.model_id, flush=True)
                    self._obj = self._make()
                    print("%s: loaded in %.1f s" % (self.model_id, time.time() - t0), flush=True)
                    if self.after is not None:
                        self.after()
        return self._obj

    def __getattr__(self, name):
        st = self.__dict__.get("_static", {})
        if name in st:
            return st[name]
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.get(), name)

    def runners(self):
        return self._obj.runners() if self._obj is not None else {}


def _residency(server):
    """Label each model's GPU runners (id/part) and pin those of --pin (again after a request that
    made new ones)."""
    from reclank.encoders.runtime import RESIDENCY
    for mid, m in _models(server).items():
        for part, r in m.runners().items():
            r.label = "%s/%s" % (mid, part)
            if mid in server.pins or "*" in server.pins:
                RESIDENCY.pin(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description="the generation server (docs/GENERATION_PLAN.md)")
    ap.add_argument("--higgs", default=None, help="a Higgs TTS 3 directory (bosonai/higgs-tts-3-4b)")
    ap.add_argument("--speech-id", default="higgs-tts-3")
    ap.add_argument("--voices", default=None,
                    help="a directory of voices to clone: NAME.wav and its transcript NAME.txt")
    ap.add_argument("--acestep", default=None, help="an ACE-Step 1.5 directory (ACE-Step/Ace-Step1.5)")
    ap.add_argument("--music-id", default="ace-step-1.5")
    ap.add_argument("--whisper", default=None,
                    help="a Whisper large-v3 directory (openai/whisper-large-v3): speech to text")
    ap.add_argument("--whisper-id", default="whisper-large-v3")
    ap.add_argument("--seedvr2", default=None, help="a SeedVR2-3B directory (its .safetensors of "
                    "scripts/pt_to_safetensors.py): image and video upscaling")
    ap.add_argument("--seedvr2-id", default="seedvr2-3b")
    ap.add_argument("--seedvr2-7b", default=None, help="a SeedVR2-7B directory (its .safetensors of "
                    "scripts/pt_to_safetensors.py; pos_emb.pt from the 3B's with --out): \"seedvr2-7b\" on the same "
                    "routes, one SeedVR2 model loaded at a time (the 3B first)")
    ap.add_argument("--realesrgan", default=None, help="a Real-ESRGAN directory (its .safetensors of "
                    "scripts/pt_to_safetensors.py --key params_ema,params): image upscaling")
    ap.add_argument("--realesrgan-models", default="realesrgan-x4plus,realesrgan-x2plus,realesrgan-x4plus-anime,"
                    "realesr-general-x4v3,realesr-animevideov3", help="the Real-ESRGAN models to load")
    ap.add_argument("--video-dir", default=os.path.expanduser("~/.cache/reclank/videos"),
                    help="the video jobs and their MP4s")
    ap.add_argument("--stereo-moge", default=None, help="a MoGe-2 directory (Ruicheng/moge-2-vitl[-normal]): "
                    "stereo images (POST /v1/images/stereo)")
    ap.add_argument("--stereo-da3", default=None, help="a DA3MONO-LARGE directory: stereo images")
    ap.add_argument("--stereo-vda", default=None, help="a Video-Depth-Anything-Small directory: stereo videos "
                    "(POST /v1/videos/stereo)")
    ap.add_argument("--rife", default=None, help="a Practical-RIFE 4.25 folder (its flownet.safetensors of "
                    "scripts/pt_to_safetensors.py): video frame interpolation (POST /v1/videos/interpolate)")
    ap.add_argument("--cpu", action="store_true", help="the models on the CPU (float32)")
    ap.add_argument("--gpu-budget", type=float, default=None, metavar="GB",
                    help="the GPU memory the models may hold (default: the card's, or with --gpu-peers off the "
                         "free memory at the start, less RESIDENCY's reserve of 512 MB); the parts used least "
                         "recently go for the others")
    from server.gpushare import add_gpu_args
    add_gpu_args(ap)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8084)
    ap.add_argument("--cpu-threads", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    H.add_http_args(ap)
    a = ap.parse_args(argv)
    stereo_any = a.stereo_moge or a.stereo_da3 or a.stereo_vda
    if not (a.higgs or a.acestep or a.whisper or a.seedvr2 or a.seedvr2_7b or a.realesrgan or stereo_any or a.rife):
        ap.error("give a model: --higgs, --acestep, --whisper, --seedvr2, --realesrgan, --stereo-moge, --stereo-da3, "
                 "--stereo-vda, --rife")
    budget = None
    if not a.cpu:
        # the free memory before the models (the budget's default), the GPU library loaded first
        from reclank.backends.cuda.gpumm import mem_info
        from reclank.backends.cuda.runtime import lib
        from reclank.encoders.runtime import RESIDENCY
        lib()
        free0, card = mem_info()
        if a.gpu_budget is not None:
            budget = int(a.gpu_budget * 1e9)
        else:
            budget = max(0, (card if a.gpu_peers == "on" else free0) - RESIDENCY.reserve)
    # every model made at its first request, not here (Lazy)
    gpu, ct = not a.cpu, a.cpu_threads
    speech = None
    if a.higgs:
        from reclank.models.higgs.tts import RATE as HIGGS_RATE
        speech = Lazy(lambda: Speech(a.higgs, gpu, ct, a.speech_id, a.voices), a.speech_id, rate=HIGGS_RATE)
    music = Lazy(lambda: Music(a.acestep, gpu, ct, a.music_id), a.music_id,
                 sample_rate=json.load(open(os.path.join(a.acestep, "vae", "config.json")))["sampling_rate"]) \
        if a.acestep else None
    whisper = Lazy(lambda: Transcriber(a.whisper, gpu, ct, a.whisper_id), a.whisper_id) if a.whisper else None
    from server import idle
    httpd = idle.server(H._Server, (a.host, a.port), _GenHandler)
    httpd.speech = speech
    httpd.music = music
    httpd.whisper = whisper
    httpd.lock = threading.Lock()
    from server import upscale as U
    httpd.upscalers = {}
    if a.seedvr2 or a.seedvr2_7b:
        paths = {k: v for k, v in (("3b", a.seedvr2), ("7b", a.seedvr2_7b)) if v}
        slot = U.SeedVR2Slot(paths, not a.cpu, a.cpu_threads)
        for variant, mid in (("3b", a.seedvr2_id), ("7b", "seedvr2-7b")):
            if variant in paths:
                httpd.upscalers[mid] = U.Upscaler("seedvr2", paths[variant], mid, not a.cpu, a.cpu_threads,
                                                  slot=slot, variant=variant)
        slot.get("3b" if "3b" in paths else "7b")
    if a.realesrgan:
        for mid in (m for m in a.realesrgan_models.split(",") if m):
            httpd.upscalers[mid] = U.Upscaler("esrgan", a.realesrgan, mid, not a.cpu, a.cpu_threads)
    vups = {k: u for k, u in httpd.upscalers.items() if u.describe().get("video")}
    httpd.upscales = U.VideoUpscales(vups, a.video_dir, httpd) if vups else None
    httpd.stereo = None
    if stereo_any:
        from server.stereo import StereoServer
        httpd.stereo = StereoServer(a.stereo_moge, a.stereo_da3, a.stereo_vda, not a.cpu, a.cpu_threads, a.video_dir,
                                    httpd)
    httpd.interp = None
    if a.rife:
        from server.interpolate import Interpolator
        httpd.interp = Interpolator(a.rife, not a.cpu, a.cpu_threads, a.video_dir, httpd)
    httpd.quiet = a.quiet
    httpd.pins = {p for p in a.pin.split(",") if p}
    for m in (speech, music, whisper):
        if isinstance(m, Lazy):
            m.after = lambda: _residency(httpd)
    unknown = httpd.pins - set(_models(httpd)) - {"*"}
    if unknown:
        ap.error("--gpu-lock: no model %s (the models: %s)" % (", ".join(sorted(unknown)), ", ".join(_models(httpd))))
    httpd.gpu_share = None
    if budget is not None:
        RESIDENCY.enabled = True
        RESIDENCY.budget = budget
        RESIDENCY.verbose = RESIDENCY.verbose or not a.quiet
        _residency(httpd)
        if a.gpu_peers == "on":
            from server.gpushare import RunnerShare, attach
            httpd.gpu_share = attach(RunnerShare("generate", lambda: _models(httpd), lambda: [httpd.lock],
                                                 locks=httpd.pins, port=httpd.server_address[1], host=a.host,
                                                 api_key=a.api_key, wait=a.gpu_wait, hold=a.gpu_hold,
                                                 quiet=a.quiet))
    httpd.api_key = a.api_key or None
    httpd.cors_origin = a.cors_origin or None
    httpd.max_body = int(a.max_body_mb * (1 << 20))
    print("reclank generation server http://%s:%d/v1  speech=%s  music=%s  asr=%s  "
          "upscale=%s  stereo=%s  interpolate=%s  key %s%s" % (
              a.host, httpd.server_address[1], speech.model_id if speech else "-",
              music.model_id if music else "-", whisper.model_id if whisper else "-", ",".join(httpd.upscalers) or "-",
              ",".join(httpd.stereo.model_ids) if httpd.stereo else "-",
              httpd.interp.model_id if httpd.interp else "-", "required" if httpd.api_key else "OFF",
        "  GPU budget %.2f GB" % (budget / 1e9) if budget is not None else ""), flush=True)

    def busy():
        """A job queued or running (an idle exit would fail it: the jobs of a new start that were
        not done are)."""
        ms = [getattr(httpd, "upscales", None), getattr(httpd, "stereo", None),
              getattr(httpd, "interp", None)]
        return any(j.get("status") in ("queued", "in_progress") for m in ms if m is not None
                   for j in list(m.jobs.values()))
    try:
        return idle.serve(httpd, a.idle_exit, busy)
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
