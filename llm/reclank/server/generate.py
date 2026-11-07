"""The generation server: python -m server.generate [--flux2 DIR] [--sd [ID=]DIR ...] [--qwen-image DIR]
[--z-image DIR] [--higgs DIR [--voices DIR]] [--acestep DIR] [--wan DIR [--video-dir DIR]] [--whisper DIR]
[--seedvr2 DIR] [--realesrgan DIR] [--stereo-vda DIR] [--rife DIR] [--port 8084].

Upscaling (SeedVR2-3B, Real-ESRGAN): server/upscale.py's routes, POST /v1/images/upscale and the
jobs of POST /v1/videos/upscale (read with GET /v1/videos/{id} as the video jobs).

The models of docs/GENERATION_PLAN.md, each loaded by its flag. Images (FLUX.2
[klein], reclank/models/flux2) in the API of OpenAI:

    POST /v1/images/generations  {"prompt": "...", "model": "flux2-klein",
                                  "n": 1, "size": "1024x1024" (multiples of 16,
                                  at most 2048 a side), "response_format":
                                  "b64_json" (no URLs: nothing is hosted),
                                  "output_format": "png",
                                  extensions: "seed": int, "steps": int (4)}
    POST /v1/images/edits        multipart/form-data: "image" (or "image[]";
                                  files: PNG, JPEG, WebP, ... of at most 50
                                  megapixels, 1 to the model's count: FLUX.2
                                  4, Qwen-Image-2.1 10; RGBA kept), "prompt",
                                  "model", "n", "size" ("auto", the default:
                                  the model's, FLUX.2 the first image's size
                                  after the pipeline's resize; or as above),
                                  "response_format", "output_format", "seed",
                                  "steps": the reference images (editing,
                                  image to image; no "mask": a mask or a
                                  painted annotation is another image, the
                                  prompt saying what it marks)
    GET  /v1/models, /health

Stable Diffusion 1.5 / 2.0 (reclank/models/sd; --sd, once a model: the id the
directory's name, or ID=DIR) in the same route: "model": its id (the first
image model by default), "size" multiples of 64 (default the trained size:
512, or 768 for 2.0-768, at most 1024 a side), extensions "steps" (25),
"negative_prompt" (""), "guidance_scale" (7.5), "seed". --f20: the UNets'
float32 matrices in F20 (16 significant bits, 63% of their memory; the image
within about 0.1 level of 255 of float32's).

Qwen-Image-2.1 (reclank/models/qwen_image, "qwen-image-2.1"; its weights'
license is research and non-commercial) in the same route: "size" multiples
of 32 (1024 x 1024 by default, at most 2048 a side), its PNG RGBA,
"background": "transparent" (the model card's form of a prompt of a
transparent image) or "auto", extensions "steps" (40), "negative_prompt" and
"cfg" (true CFG: 1, off, by default, as the pipeline), "seed". Its text
encoder runs on the CPU (12 GB: its matrices in BF12); on the GPU its DiT
(14 GB of weights; 14.6 GB in all at 1024 x 1024, 16.4 GB at 2048) leaves
for the VAE. It edits in
POST /v1/images/edits (1 to 10 images): size "auto" the last image's aspect
ratio at the area of resolution^2 (the pipeline's output_resolution),
extensions "resolution" (1024: the area each reference is resized to; 256 to
1024), "background", "negative_prompt", "cfg" as generations. The DiT's cache
of the references' keys and values is F20 in host memory (0.63 MiB a
reference token, pinned; RECLANK_QWEN_CACHE_HOST=0: on the GPU), streamed to
the GPU a block at a time: at 1024 x 1024 one reference peaks at 14.4 GiB of
the GPU, two at 14.7, three at 15.5, six at 17.7 (about 0.19 MiB a reference
token more); a request that does not fit the 3090 is refused (a smaller
"resolution" fits). The text encoder on the CPU reads the references' tokens
too: 31 s for a 1024 x 1024 reference, 60 s for two (12 threads); a step
4.1 s (one reference, 1024 x 1024 out), 4.6 s (two), 5.2 s (three), 7.5 s
(six): 3.5 to 6 minutes for 40 steps.

Z-Image Turbo (reclank/models/zimage, "z-image-turbo", Apache 2.0) in the same
route: "size" multiples of 16 (1024 x 1024 by default, at most 2048 a side and
1536 x 1536 of area: the VAE's buffers of a larger image do not fit a 24 GB
GPU), extensions "steps" (9, the model card's: no CFG, so no negative prompt),
"seed". Its text encoder (Qwen3-4B, 8 GB) runs on the CPU; on the GPU its DiT
(15.3 GB of weights in F20; 15.6 GB in all at 1024 x 1024, 16.3 GB at 1536)
leaves for the VAE.

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

Video (Wan 2.2 TI2V-5B, reclank/models/wan, Apache 2.0; --wan DIR), jobs in the shape of
OpenAI's video API:

    POST   /v1/videos  {"prompt": "...", "model": "wan2.2-ti2v-5b", "size": "832x480"
                        (width x height, multiples of 32, at most 1280 a side and
                        1280 x 704 of area), "seconds": 4 (at most 5; the frames
                        24 seconds + 1, as 4 k + 1), extensions: "frames" (in
                        place of seconds, 4 k + 1, at most 121), "steps" (50),
                        "guidance_scale" (5.0), "negative_prompt" (the model
                        card's), "seed"}: the job (status "queued")
    GET    /v1/videos  the jobs; GET /v1/videos/{id}: a job's status
                        ("queued", "in_progress", "completed", "failed") and
                        progress (0 to 100: the text, the steps, the VAE)
    GET    /v1/videos/{id}/content  the MP4 (H.264, 24 fps) of a completed job
    DELETE /v1/videos/{id}  cancel a job (queued or running) or delete its video

Wan 2.2 T2V-A14B (--wan DIR of it; "wan2.2-t2v-a14b"): sizes of 16, at most 832 x 480 of area with
81 frames and 1280 x 720 with 33; 16 fps; "steps" 40 (at most), "guidance_scale" 4.0 and
"guidance_scale_2" 3.0 (the low-noise expert's); the experts from a weightkeeper that holds DIR
(--keeper NAME: that one), else loaded from the shards (~20 min from an HDD).

2D to stereo 3D for VR players (MoGe-2, DA3MONO, VDA-Small; reclank/models/depth; --stereo-moge,
--stereo-da3, --stereo-vda): POST /v1/images/stereo (an image to side-by-side or over-under) and
POST /v1/videos/stereo (a video file to a stereo MP4, a job of the routes below): server/stereo.py.

Frame interpolation (Practical-RIFE 4.25, reclank/models/vfi; --rife DIR): POST
/v1/videos/interpolate (a video file, or "video_id" of a completed job of the routes below, to 2x,
4x or a target rate; scene cuts kept; the audio copied), a job of the routes below:
server/interpolate.py.

One job runs at a time, the rest wait in a queue. The jobs (ID.json) and
their videos (ID.mp4) live in --video-dir, not in RAM: the completed ones stay
over a restart. The text encoder (umT5-XXL, 10 GB of RAM) runs on the CPU; the
DiT (12.5 GB in F20) and the VAE take the GPU in turn (the DiT may go for the
VAE). An 832 x 480 clip of 17 frames and 50 steps takes about 4 minutes on the
3090.

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

A request's images run one after another (the seed of image i: seed + i;
no seed: a random one, given back in each item as "seed"). Requests run one
at a time. The noise is NumPy's normal of the seed, so the image of a seed is
not diffusers' of the same seed.

The GPU's memory (encoders.runtime.RESIDENCY): every model's parts are loaded
into host RAM at the start; on the GPU a part's weights and buffers are
uploaded at its first run and stay there until another part needs the room:
then the parts used least recently go (their host copies stay; the next run
uploads them again, a few seconds for a DiT), within --gpu-budget GB (default:
the card's, less a reserve of 512 MB). A request's parts stay while it runs
(each until its turn is over: Wan's DiT may go for its VAE). --gpu-lock ID,...
(--pin) keeps whole models on the GPU (no ids: all); --keep keeps the image
models' text encoders and VAEs. The other reclank servers on the GPU are asked
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

MAX_SIDE = 2048


def png(img):
    """The PNG bytes of an image (H, W, 3) or RGBA (H, W, 4) uint8 (zlib, no
    filter)."""
    h, w, ch = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if ch == 4 else 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _image_args(body, default, unit, max_side, steps, max_steps):
    """The prompt, n, (height, width), steps and seed of an images request."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    n = int(body.get("n") or 1)
    if not 1 <= n <= 8:
        raise ValueError("n must be 1 to 8")
    size = str(body.get("size") or default)
    if size == "auto":
        size = default
    try:
        w, h = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise ValueError("size must be WIDTHxHEIGHT, such as %s" % default)
    if w % unit or h % unit or not 256 <= min(w, h) or max(w, h) > max_side:
        raise ValueError("width and height must be multiples of %d from 256 to %d" % (unit, max_side))
    fmt = body.get("response_format") or "b64_json"
    if fmt != "b64_json":
        raise ValueError("response_format must be \"b64_json\" (this server hosts no URLs)")
    if (body.get("output_format") or "png") != "png":
        raise ValueError("output_format must be \"png\"")
    steps = int(body.get("steps") or steps)
    if not 1 <= steps <= max_steps:
        raise ValueError("steps must be 1 to %d" % max_steps)
    seed = body.get("seed")
    seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little")
    return prompt, n, (h, w), steps, seed


def _item(img, seed, prompt):
    return {"b64_json": base64.b64encode(png(img)).decode("ascii"), "seed": seed, "revised_prompt": prompt}


def images_body(data):
    """The response body of an images route (generations, edits)."""
    return {"created": int(time.time()), "data": data}


EDIT_IMAGES = 10                # reference images a request (each model's edit has its own count)
EDIT_PIXELS = 50_000_000        # an input image's pixels (the models resize them)


def edit_form(form):
    """The text fields ({name: str}) and the images (PIL, their EXIF
    orientation applied as diffusers' load_image; RGBA when the image has an
    alpha, else RGB: a model that takes RGB converts them) of an edits form
    (multipart: image or image[], 1 to EDIT_IMAGES files). ValueError: a bad
    request."""
    import io

    from PIL import Image, ImageOps, UnidentifiedImageError
    files = form.get("image", []) + form.get("image[]", [])
    if not files or not all(data for data, _ in files):
        raise ValueError("image is required: 1 to %d image files (image or image[])" % EDIT_IMAGES)
    if len(files) > EDIT_IMAGES:
        raise ValueError("at most %d images" % EDIT_IMAGES)
    if form.get("mask"):
        raise ValueError("mask is not supported (no inpainting): give a mask or a painted annotation as another "
                         "image, the prompt saying what it marks")
    body = {}
    for k, v in form.items():
        if k not in ("image", "image[]"):
            try:
                body[k] = v[0][0].decode("utf-8").strip()
            except UnicodeDecodeError:
                raise ValueError("%s must be UTF-8 text" % k) from None
    images = []
    for i, (data, name) in enumerate(files):
        try:
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
                if w * h > EDIT_PIXELS:
                    raise ValueError("image %d is %dx%d: at most %d pixels" % (i + 1, w, h, EDIT_PIXELS))
                alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
                images.append(ImageOps.exif_transpose(im).convert("RGBA" if alpha else "RGB"))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError("image %d (%s): not an image file PIL reads" % (i + 1, name or "-")) from None
    return body, images


def _edit_args(body, unit, max_side, steps, max_steps):
    """_image_args of an edits request: size "auto" (the default) None, the
    model's (FLUX.2: the first image's; Qwen-Image: the last one's aspect
    ratio)."""
    auto = (body.get("size") or "auto") == "auto"
    prompt, n, size, steps, seed = _image_args(dict(body, size="1024x1024" if auto else body["size"]),
                                               "1024x1024", unit, max_side, steps, max_steps)
    return prompt, n, None if auto else size, steps, seed


class Images:
    """FLUX.2 [klein] on a device."""

    max_side = MAX_SIDE

    def __init__(self, path, gpu, cpu_threads, model_id):
        from reclank.models.flux2.pipeline import Flux2Klein
        self.fx = Flux2Klein(path, gpu=gpu, cpu_threads=cpu_threads)
        self.model_id = model_id

    def generate(self, body):
        prompt, n, size, steps, seed = _image_args(body, "1024x1024", 16, MAX_SIDE, 4, 50)
        return [_item(self.fx.generate(prompt, size=size, steps=steps, seed=seed + i, keep_text=True),
                      seed + i, prompt) for i in range(n)]

    def runners(self):
        """The GPU runners of the model's parts (name: _Runner)."""
        return self.fx.runners()

    def kept(self):
        """The parts that --keep pins: the text encoder and the VAE."""
        return [r for k, r in self.runners().items() if k in ("text", "vae")]

    edit_images = 4

    def edit(self, body, images):
        """The images of an edits request: body its text fields, images
        the reference images (PIL; edit_form(): preprocess takes them in
        RGB), each with the references, seed + i."""
        from reclank.models.flux2.pipeline import preprocess
        if len(images) > self.edit_images:
            raise ValueError("at most %d images (%s)" % (self.edit_images, self.model_id))
        prompt, n, size, steps, seed = _edit_args(body, 16, MAX_SIDE, 4, 50)
        pixels = [preprocess(im) for im in images]
        return [_item(self.fx.generate(prompt, size=size, steps=steps, seed=seed + i, keep_text=True,
                                       images=pixels), seed + i, prompt) for i in range(n)]


class SDImages:
    """Stable Diffusion 1.5 / 2.0 on a device."""

    max_side = 1024

    def __init__(self, path, gpu, cpu_threads, model_id, f20=False):
        from reclank.models.sd.pipeline import StableDiffusion
        self.sd = StableDiffusion(path, gpu=gpu, cpu_threads=cpu_threads, f20=f20)
        self.model_id = model_id

    def generate(self, body):
        d = self.sd.size
        prompt, n, size, steps, seed = _image_args(body, "%dx%d" % (d, d), 64, self.max_side, 25, 100)
        negative = body.get("negative_prompt") or ""
        if not isinstance(negative, str):
            raise ValueError("negative_prompt must be a string")
        guidance = float(body.get("guidance_scale", 7.5))
        if not 0 <= guidance <= 30:
            raise ValueError("guidance_scale must be 0 to 30")
        return [_item(self.sd.generate(prompt, negative, size=size, steps=steps, guidance=guidance, seed=seed + i,
                                       keep_text=True), seed + i, prompt) for i in range(n)]

    def runners(self):
        return self.sd.runners()

    def kept(self):
        return [r for k, r in self.runners().items() if k in ("text", "vae")]


class ZImages:
    """Z-Image Turbo on a device."""

    max_side = MAX_SIDE
    max_area = 1536 * 1536      # the VAE's buffers of 2048 x 2048 need more than 23 GB

    def __init__(self, path, gpu, cpu_threads, model_id):
        from reclank.models.zimage.pipeline import ZImageTurbo
        self.zi = ZImageTurbo(path, gpu=gpu, cpu_threads=cpu_threads)
        self.model_id = model_id

    def generate(self, body):
        prompt, n, size, steps, seed = _image_args(body, "1024x1024", 16, MAX_SIDE, 9, 50)
        if size[0] * size[1] > self.max_area:
            raise ValueError("the image's area must be at most %d pixels (1536x1536)" % self.max_area)
        return [_item(self.zi.generate(prompt, size=size, steps=steps, seed=seed + i, keep=True), seed + i,
                      prompt) for i in range(n)]

    def runners(self):
        return self.zi.runners()

    def kept(self):
        return [r for k, r in self.runners().items() if k in ("text", "vae")]


class QwenImages:
    """Qwen-Image-2.1 on a device: RGBA images."""

    max_side = MAX_SIDE

    def __init__(self, path, gpu, cpu_threads, model_id):
        from reclank.models.qwen_image.pipeline import QwenImage21
        self.qi = QwenImage21(path, gpu=gpu, cpu_threads=cpu_threads)
        self.model_id = model_id

    @staticmethod
    def _options(body, prompt):
        """The prompt (background), negative prompt and cfg of a request."""
        bg = body.get("background") or "auto"
        if bg not in ("auto", "transparent"):
            raise ValueError("background must be \"transparent\" or \"auto\"")
        if bg == "transparent":
            # the model card's form of a prompt of a transparent image
            prompt = ("This is an RGBA image with transparency. " + prompt.strip().rstrip(".") +
                      ". The image has alpha channel and the background is transparent.")
        negative = body.get("negative_prompt")
        if negative is not None and not isinstance(negative, str):
            raise ValueError("negative_prompt must be a string")
        cfg = float(body.get("cfg") or 1.0)
        if not 1 <= cfg <= 20:
            raise ValueError("cfg must be 1 to 20")
        return prompt, negative, cfg

    def generate(self, body):
        prompt, n, size, steps, seed = _image_args(body, "1024x1024", 32, MAX_SIDE, 40, 100)
        prompt, negative, cfg = self._options(body, prompt)
        return [_item(self.qi.generate(prompt, size=size, steps=steps, seed=seed + i, negative=negative, cfg=cfg,
                                       keep=True), seed + i, prompt) for i in range(n)]

    # the GPU's MiB of an edit's references and output beyond the DiT's weights and the rest (about
    # 13.3 GiB, 14.1 with a cache on the GPU): (a row of the prefix: the references' tokens, with CFG
    # the negative's too; a token of the output; the limit) of each form of the DiT's cache: in host
    # memory (the default: the 3090's peaks 14.4, 14.7, 15.5, 17.7 GiB with 1, 2, 3, 6 references of
    # 1024 x 1024), F20 on the GPU (RECLANK_QWEN_CACHE_HOST=0: 17.2, 20.1, 23.0 GiB with 1, 2, 3),
    # float32 on the GPU (RECLANK_QWEN_CACHE_F20=0: 19.1, 23.2 GiB with 1, 2)
    GPU_MB = {"host": (0.19, 0.15, 10000), "f20": (0.74, 0.1, 9500), "f32": (1.02, 0.1, 8900)}

    def edit(self, body, images):
        """The images of an edits request: body its text fields, images the
        reference images (PIL; edit_form()), each resized to the area
        resolution^2; size "auto": the last image's aspect ratio at that area
        (the pipeline's), seed + i."""
        from reclank.models.qwen_image.pipeline import MAX_IMAGES, calculate_dimensions
        if len(images) > MAX_IMAGES:
            raise ValueError("at most %d images (%s)" % (MAX_IMAGES, self.model_id))
        try:
            res = int(body.get("resolution") or 1024)
        except ValueError:
            raise ValueError("resolution must be a number") from None
        if res % 32 or not 256 <= res <= 1024:
            raise ValueError("resolution must be a multiple of 32 from 256 to 1024")
        prompt, n, size, steps, seed = _edit_args(body, 32, MAX_SIDE, 40, 100)
        prompt, negative, cfg = self._options(body, prompt)
        if size is None:
            w, h = calculate_dimensions(res * res, images[-1].size[0] / images[-1].size[1])
            if not 256 <= min(w, h) or max(w, h) > MAX_SIDE:
                raise ValueError("the last image's aspect ratio gives %dx%d: give a size" % (w, h))
            size = (h, w)
        if self.qi.gpu:
            refs = sum((h // 16) * (w // 16) for w, h in
                       (calculate_dimensions(res * res, im.size[0] / im.size[1]) for im in images))
            form = ("host" if self.qi.dit.cache_on_host() else "f20") if self.qi.cache_f20 else "f32"
            per_row, per_out, limit = self.GPU_MB[form]
            rows = refs * (2 if negative is not None and cfg > 1 else 1)
            if per_row * rows + per_out * (size[0] // 16) * (size[1] // 16) > limit:
                raise ValueError("the references (%d tokens at resolution %d) and the output do not fit the GPU: "
                                 "a smaller resolution, fewer images or a smaller size" % (refs, res))
        return [_item(self.qi.generate(prompt, size=size, steps=steps, seed=seed + i, negative=negative, cfg=cfg,
                                       keep=True, images=images, resolution=res), seed + i, prompt)
                for i in range(n)]

    def runners(self):
        return self.qi.runners()

    def kept(self):
        return [r for k, r in self.runners().items() if k in ("text", "vae")]


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


class _Cancelled(Exception):
    pass


class Videos:
    """Wan 2.2 TI2V-5B or T2V-A14B on a device: the video jobs, their queue and their worker."""

    max_side, max_area, max_frames = 1280, 1280 * 704, 121
    # T2V-A14B (its steps 4 to 18 min at these sizes): the frames by the area, sizes of 16
    A14B_LIMITS = ((832 * 480, 81), (1280 * 720, 33))

    def __init__(self, path, gpu, cpu_threads, model_id, work_dir, server, keeper=None):
        from reclank.models.wan.pipeline import WanT2V
        self.wan = WanT2V(path, gpu=gpu, cpu_threads=cpu_threads, keeper=keeper)
        self.a14b = self.wan.a14b
        if self.a14b:
            self.max_area, self.max_frames = self.A14B_LIMITS[-1][0], self.A14B_LIMITS[0][1]
        model_id = model_id or ("wan2.2-t2v-a14b" if self.a14b else "wan2.2-ti2v-5b")
        self.model_id, self.dir, self.server = model_id, work_dir, server
        os.makedirs(work_dir, exist_ok=True)
        self.jobs = collections.OrderedDict()      # id: the job object (as GET answers it)
        self.queue = collections.deque()
        self.cancel = set()
        self.cv = threading.Condition()
        for name in sorted(os.listdir(work_dir)):
            if not name.endswith(".json"):
                continue
            try:
                job = json.load(open(os.path.join(work_dir, name)))
            except (OSError, ValueError):
                continue
            if job.get("status") in ("queued", "in_progress"):
                job.update(status="failed", error={"code": "server_stopped", "message": "the server stopped"})
                self._save(job)
            self.jobs[job["id"]] = job
        threading.Thread(target=self._worker, name="videos", daemon=True).start()

    def runners(self):
        return self.wan.runners()

    def _file(self, vid, ext):
        return os.path.join(self.dir, vid + ext)

    def _save(self, job):
        tmp = self._file(job["id"], ".json.part")
        with open(tmp, "w") as f:
            json.dump({k: v for k, v in job.items() if not k.startswith("_")}, f)
        os.replace(tmp, self._file(job["id"], ".json"))

    @staticmethod
    def view(job):
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def submit(self, body):
        """A new job of a request body (checked here: a bad one is a ValueError)."""
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if body.get("input_reference") is not None or body.get("image") is not None:
            raise ValueError("image to video is not supported (text to video only)")
        size = str(body.get("size") or "832x480")
        m = re.fullmatch(r"(\d+)x(\d+)", size)
        if not m:
            raise ValueError("size must be WIDTHxHEIGHT")
        wid, hgt = int(m.group(1)), int(m.group(2))
        mult = 16 if self.a14b else 32
        if wid % mult or hgt % mult or not 64 <= min(wid, hgt) or max(wid, hgt) > self.max_side or \
                wid * hgt > self.max_area:
            raise ValueError("size: multiples of %d, at most %d a side and %d pixels" % (
                mult, self.max_side, self.max_area))
        max_frames = self.max_frames if not self.a14b else \
            next(n for area, n in self.A14B_LIMITS if wid * hgt <= area)
        frames = body.get("frames")
        if frames is None:
            secs = float(body.get("seconds") or 4)
            if not 0 < secs <= 5:
                raise ValueError("seconds must be more than 0 and at most 5")
            frames = int(round(secs * self.wan.fps)) // 4 * 4 + 1
            if body.get("seconds") is None:
                frames = min(frames, max_frames)    # (the default at a size of fewer frames)
        frames = int(frames)
        if frames < 1 or frames > max_frames or frames % 4 != 1:
            raise ValueError("frames must be 4 k + 1, at most %d%s" % (
                max_frames, " at this size" if self.a14b else ""))
        d_steps, d_g, d_g2 = self.wan.defaults()
        steps = int(body.get("steps") or d_steps)
        if not 1 <= steps <= (d_steps if self.a14b else 100):
            raise ValueError("steps must be 1 to %d" % (d_steps if self.a14b else 100))
        guidance = float(body.get("guidance_scale", d_g))
        guidance2 = float(body.get("guidance_scale_2", d_g2)) if self.a14b else None
        if not 0 <= guidance <= 20 or guidance2 is not None and not 0 <= guidance2 <= 20:
            raise ValueError("guidance_scale (and guidance_scale_2) must be 0 to 20")
        negative = body.get("negative_prompt")
        if negative is not None and not isinstance(negative, str):
            raise ValueError("negative_prompt must be a string")
        seed = body.get("seed")
        seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little")
        job = {"id": "video_" + os.urandom(12).hex(), "object": "video", "model": self.model_id,
               "status": "queued", "progress": 0, "created_at": int(time.time()), "completed_at": None,
               "size": "%dx%d" % (wid, hgt), "seconds": "%g" % round((frames - 1) / self.wan.fps, 2), "frames": frames,
               "steps": steps, "guidance_scale": guidance, "seed": seed, "prompt": prompt, "error": None,
               "_negative": negative}
        if self.a14b:
            job["guidance_scale_2"] = guidance2
        with self.cv:
            self.jobs[job["id"]] = job
            self._save(job)
            self.queue.append(job["id"])
            self.cv.notify()
        return self.view(job)

    def delete(self, vid):
        """Cancel a job, or delete a finished one and its video. None: no such job."""
        with self.cv:
            job = self.jobs.get(vid)
            if job is None:
                return None
            if job["status"] == "in_progress":
                self.cancel.add(vid)    # the worker deletes it when it stops
            else:
                if vid in self.queue:
                    self.queue.remove(vid)
                self._remove(vid)
        return {"id": vid, "object": "video.deleted", "deleted": True}

    def _remove(self, vid):
        self.jobs.pop(vid, None)
        for ext in (".json", ".mp4"):
            try:
                os.remove(self._file(vid, ext))
            except FileNotFoundError:
                pass

    def _worker(self):
        from reclank.models.wan.pipeline import write_mp4
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                vid = self.queue.popleft()
                job = self.jobs[vid]
                job["status"] = "in_progress"
                self._save(job)
            t0 = time.time()
            w, h = (int(v) for v in job["size"].split("x"))
            # the share of the progress: the text 2%, the steps to 90%, the VAE to 99%, the MP4 100%
            span = {"text": (0, 2), "step": (2, 90), "vae": (90, 99)}

            def progress(stage, i, n):
                if vid in self.cancel:
                    raise _Cancelled()
                a, b = span[stage]
                job["progress"] = int(a + (b - a) * i / max(n, 1))

            try:
                with self.server.lock:
                    fr = self.wan.generate(job["prompt"], job["_negative"], size=(h, w), frames=job["frames"],
                                           steps=job["steps"], guidance=job["guidance_scale"], seed=job["seed"],
                                           progress=progress, keep=True, guidance2=job.get("guidance_scale_2"))
                part = self._file(vid, ".part.mp4")
                write_mp4(fr, part, self.wan.fps)
                os.replace(part, self._file(vid, ".mp4"))
                job.update(status="completed", progress=100, completed_at=int(time.time()))
                if not getattr(self.server, "quiet", False):
                    print("video %s: %d frames of %s in %.1f s (seed %d)" % (vid, job["frames"], job["size"],
                                                                            time.time() - t0, job["seed"]), flush=True)
            except _Cancelled:
                pass
            except Exception as exc:  # noqa: BLE001 - a failed job reports its error
                job.update(status="failed", error={"code": type(exc).__name__, "message": str(exc)[:500]})
                print("video %s failed: %s" % (vid, exc), flush=True)
            with self.cv:
                if vid in self.cancel:
                    self.cancel.discard(vid)
                    self._remove(vid)
                    try:
                        os.remove(self._file(vid, ".part.mp4"))
                    except FileNotFoundError:
                        pass
                else:
                    self._save(job)


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
            for im in self.server.images.values():
                data.append({"id": im.model_id, "object": "model", "owned_by": "reclank",
                             "kind": "image", "max_side": im.max_side, "edits": hasattr(im, "edit")})
            if self.server.music:
                data.append({"id": self.server.music.model_id, "object": "model", "owned_by": "reclank",
                             "kind": "music", "sample_rate": self.server.music.ace.vae.rate, "max_seconds": 240})
            if self.server.videos:
                vi = self.server.videos
                data.append({"id": vi.model_id, "object": "model", "owned_by": "reclank", "kind": "video",
                             "fps": vi.wan.fps, "max_side": vi.max_side, "max_frames": vi.max_frames})
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

    def _videos(self, vid=None):
        """The video jobs' manager (of the job vid: Wan's or the upscales')."""
        ups = getattr(self.server, "upscales", None)
        if vid is not None and ups is not None and vid in ups.jobs:
            return ups
        vi = self.server.videos or ups
        if vi is None:
            self._error(404, "no video model loaded (--wan, --seedvr2)", code="model_not_found")
        return vi

    def _managers(self):
        """The managers of /v1/videos' jobs: Wan's, the upscales', the stereo videos', the
        interpolations'."""
        return [m for m in (self.server.videos, getattr(self.server, "upscales", None),
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
                return self._error(404, "no video model loaded (--wan, --seedvr2, --stereo-vda, --rife)",
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

    def _video_post(self):
        vi = self._videos()
        if vi is None:
            return
        try:
            body = self._body()
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            if body.get("model") not in (None, vi.model_id):
                raise ValueError("model must be %s" % vi.model_id)
            job = vi.submit(body)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("video %s queued: %r" % (job["id"], job["prompt"][:60]), flush=True)
        self._json(200, job)

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

    def _edits(self):
        ims = {k: im for k, im in self.server.images.items() if hasattr(im, "edit")}
        if not ims:
            return self._error(404, "no image model that edits loaded (--flux2, --qwen-image)",
                               code="model_not_found")
        try:
            body, images = edit_form(multipart(self._raw_body(), self.headers.get("Content-Type")))
            model = body.get("model") or None
            if model is not None and model not in ims:
                raise ValueError("model must be one of %s" % ", ".join(ims))
            im = ims[model] if model is not None else next(iter(ims.values()))
            t0 = time.time()
            with self.server.lock:
                data = im.edit(body, images)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("edits: %d of %r with %d image(s), %.1f s" % (len(data), body.get("prompt", "")[:60], len(images),
                                                                time.time() - t0), flush=True)
        self._json(200, images_body(data))

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
        if path in ("/v1/images/edits", "/images/edits"):
            return self._edits()
        if path in ("/v1/audio/speech", "/audio/speech"):
            return self._speech()
        if path in ("/v1/audio/transcriptions", "/audio/transcriptions"):
            return self._transcription("transcribe")
        if path in ("/v1/audio/translations", "/audio/translations"):
            return self._transcription("translate")
        if path in ("/v1/music/generations", "/music/generations"):
            return self._music()
        if path in ("/v1/videos", "/videos"):
            return self._video_post()
        if path in ("/v1/images/upscale", "/images/upscale"):
            return self._upscale()
        if path in ("/v1/videos/upscale", "/videos/upscale"):
            return self._upscale_video()
        if path in ("/v1/videos/interpolate", "/videos/interpolate"):
            return self._interpolate()
        if path in ("/v1/images/stereo", "/images/stereo", "/v1/videos/stereo", "/videos/stereo"):
            return self._stereo(path.endswith("videos/stereo"))
        if path not in ("/v1/images/generations", "/images/generations"):
            return self._error(404, "no route %s" % self.path, code="not_found")
        ims = self.server.images
        if not ims:
            return self._error(404, "no image model loaded (--flux2, --sd)", code="model_not_found")
        try:
            body = self._body()
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            model = body.get("model")
            if model is not None and model not in ims:
                raise ValueError("model must be one of %s" % ", ".join(ims))
            im = ims[model] if model is not None else next(iter(ims.values()))
            t0 = time.time()
            with self.server.lock:
                data = im.generate(body)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("images: %d of %r, %.1f s" % (len(data), body.get("prompt")[:60], time.time() - t0), flush=True)
        self._json(200, {"created": int(time.time()), "data": data})


def _models(server):
    """Every model of the server (id: its object)."""
    out = dict(server.images)
    for m in (server.speech, server.music, server.videos, getattr(server, "whisper", None),
              getattr(server, "stereo", None), getattr(server, "interp", None)):
        if m is not None:
            out[m.model_id] = m
    out.update(getattr(server, "upscalers", {}))
    return out


def _residency(server):
    """Label each model's GPU runners (id/part) and pin those of --pin and --keep (again after a
    request that made new ones)."""
    from reclank.encoders.runtime import RESIDENCY
    for mid, m in _models(server).items():
        for part, r in m.runners().items():
            r.label = "%s/%s" % (mid, part)
            if mid in server.pins or "*" in server.pins:
                RESIDENCY.pin(r)
        if server.keep and hasattr(m, "kept"):
            for r in m.kept():
                RESIDENCY.pin(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description="the generation server (docs/GENERATION_PLAN.md)")
    ap.add_argument("--flux2", default=None, help="a FLUX.2 [klein] directory (diffusers' layout)")
    ap.add_argument("--image-id", default="flux2-klein")
    ap.add_argument("--sd", action="append", default=[], metavar="[ID=]DIR",
                    help="a Stable Diffusion 1.5 / 2.0 directory (diffusers' layout), once a model; the id the "
                         "directory's name unless ID= (sd-1.5, sd-2.0, sd-2.0-768)")
    ap.add_argument("--f20", action="store_true",
                    help="the float32 matrices of the SD UNets in F20 (16 significant bits, 63%% of the memory)")
    ap.add_argument("--qwen-image", default=None, help="a Qwen-Image-2.1 directory (diffusers' layout)")
    ap.add_argument("--qwen-image-id", default="qwen-image-2.1")
    ap.add_argument("--z-image", default=None, help="a Z-Image Turbo directory (diffusers' layout)")
    ap.add_argument("--z-image-id", default="z-image-turbo")
    ap.add_argument("--higgs", default=None, help="a Higgs TTS 3 directory (bosonai/higgs-tts-3-4b)")
    ap.add_argument("--speech-id", default="higgs-tts-3")
    ap.add_argument("--voices", default=None,
                    help="a directory of voices to clone: NAME.wav and its transcript NAME.txt")
    ap.add_argument("--acestep", default=None, help="an ACE-Step 1.5 directory (ACE-Step/Ace-Step1.5)")
    ap.add_argument("--music-id", default="ace-step-1.5")
    ap.add_argument("--whisper", default=None,
                    help="a Whisper large-v3 directory (openai/whisper-large-v3): speech to text")
    ap.add_argument("--whisper-id", default="whisper-large-v3")
    ap.add_argument("--wan", default=None, help="a Wan 2.2 TI2V-5B or T2V-A14B directory (diffusers' layout): "
                    "video jobs")
    ap.add_argument("--video-id", default=None, help="(default wan2.2-ti2v-5b, or wan2.2-t2v-a14b)")
    ap.add_argument("--keeper", default=None,
                    help="Wan A14B: the weightkeeper of this name (default: any that holds --wan; \"none\": loaded "
                         "from the shards)")
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
    ap.add_argument("--keep", action="store_true",
                    help="pin the image models' text encoders and VAEs on the GPU (never evicted)")
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
    if not (a.flux2 or a.sd or a.qwen_image or a.z_image or a.higgs or a.acestep or a.wan or a.whisper or
            a.seedvr2 or a.seedvr2_7b or a.realesrgan or stereo_any or a.rife):
        ap.error("give a model: --flux2, --sd, --qwen-image, --z-image, --higgs, --acestep, --wan, --whisper, "
                 "--seedvr2, --realesrgan, --stereo-moge, --stereo-da3, --stereo-vda, --rife")
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
    images = {}
    if a.flux2:
        images[a.image_id] = Images(a.flux2, not a.cpu, a.cpu_threads, a.image_id)
    for spec in a.sd:
        mid, _, path = spec.rpartition("=")
        path = path.rstrip("/")
        mid = mid or os.path.basename(path)
        if mid in images:
            ap.error("two image models of the id %s" % mid)
        images[mid] = SDImages(path, not a.cpu, a.cpu_threads, mid, f20=a.f20)
    if a.qwen_image:
        if a.qwen_image_id in images:
            ap.error("two image models of the id %s" % a.qwen_image_id)
        images[a.qwen_image_id] = QwenImages(a.qwen_image, not a.cpu, a.cpu_threads, a.qwen_image_id)
    if a.z_image:
        if a.z_image_id in images:
            ap.error("two image models of the id %s" % a.z_image_id)
        images[a.z_image_id] = ZImages(a.z_image, not a.cpu, a.cpu_threads, a.z_image_id)
    speech = Speech(a.higgs, not a.cpu, a.cpu_threads, a.speech_id, a.voices) if a.higgs else None
    music = Music(a.acestep, not a.cpu, a.cpu_threads, a.music_id) if a.acestep else None
    whisper = Transcriber(a.whisper, not a.cpu, a.cpu_threads, a.whisper_id) if a.whisper else None
    from server import idle
    httpd = idle.server(H._Server, (a.host, a.port), _GenHandler)
    httpd.images = images
    httpd.speech = speech
    httpd.music = music
    httpd.whisper = whisper
    httpd.lock = threading.Lock()
    httpd.videos = Videos(a.wan, not a.cpu, a.cpu_threads, a.video_id, a.video_dir, httpd,
                          keeper=False if a.keeper == "none" else a.keeper) if a.wan else None
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
    httpd.keep = a.keep
    httpd.pins = {p for p in a.pin.split(",") if p}
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
    print("reclank generation server http://%s:%d/v1  images=%s  speech=%s  music=%s  video=%s  asr=%s  "
          "upscale=%s  stereo=%s  interpolate=%s  key %s%s" % (
              a.host, httpd.server_address[1], ",".join(images) or "-", speech.model_id if speech else "-",
              music.model_id if music else "-", httpd.videos.model_id if httpd.videos else "-",
              whisper.model_id if whisper else "-", ",".join(httpd.upscalers) or "-",
              ",".join(httpd.stereo.model_ids) if httpd.stereo else "-",
              httpd.interp.model_id if httpd.interp else "-", "required" if httpd.api_key else "OFF",
        "  GPU budget %.2f GB" % (budget / 1e9) if budget is not None else ""), flush=True)

    def busy():
        """A job queued or running (an idle exit would fail it: the jobs of a new start that were
        not done are)."""
        ms = [httpd.videos, getattr(httpd, "upscales", None), getattr(httpd, "stereo", None),
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
