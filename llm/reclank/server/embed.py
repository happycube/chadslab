"""The embedding server: python -m server.embed -m DIR [--music DIR] [--port 8082].

EmbeddingGemma 2 (reclank/models/embedgemma2): text, audio and images, each
input one vector of floats in one space. With --music, also the music
embeddings (reclank/models/music, docs/MUSIC_EMBED_PLAN.md): a request with
"model": "music" gets the 1024 floats of MuQ-MuLan and CLAP side by side,
"muq-mulan" or "clap" the 512 of one of them (see the end of this text);
without -m the server has the music models alone. The text model runs on
the CPU or (--gpu) the GPU in BF12; the audio and the vision towers each on
the GPU, the CPU, or not at all (--audio-device, --vision-device; default
that of the text model). A tower on the CPU takes a team of --cpu-threads. The API is that of OpenAI:

    POST /v1/embeddings  {"input": "text" | ["text", ...] | [ids] | [[ids], ...]
                                   | [{"audio": ...} | {"image": ...}, ...] (below; mixed
                                     with texts too),
                          "dimensions": 768 | 512 | 256 | 128,
                          "encoding_format": "float" | "base64",
                          "task": the task of the inputs (below),
                          "title": the title of a document (task "document"),
                          "truncate": true: cut an input to the context (else 400)}
    GET  /v1/models, /health

The task: EmbeddingGemma 2 is trained with a short prefix on each text that
names the task ("task: search result | query: ...", "title: none | text:
..."). "task" takes a prompt name of the model (config_sentence_transformers.json:
"SearchQuery", "Document", "Classification", "Clustering", "STS", ...; case
and "-"/"_" do not count), or one of the aliases of other APIs ("query",
"search_query", "document", "search_document", "passage", ...). "input_type"
is a synonym of "task". A request with no task takes --task (default none:
the texts as they are). A document with a "title" gets "title: <title> |
text: " in the place of "title: none | text: ". Token ids take no prefix.

Audio: an input {"audio": base64 (or a data: URI) of an audio file in any
format of ffmpeg (MP3, WAV, FLAC, ...), "start": s, "end": s (a cut of the
file, optional)}, or {"audio": base64 of 16 kHz mono float32 little-endian
samples, "format": "f32le"}. ({"type": "input_audio", "input_audio": {"data":
...}} of OpenAI's chat is the same.) A clip is at most 30 s (the cut of the
model's feature extractor; "truncate": true keeps the first 30 s): longer
audio goes in clips (scripts/embed_audio.py). Audio takes no task prefix.

Images: an input {"image": base64 (or a data: URI) of an image file (JPEG,
PNG, WebP, ...), "budget": the soft tokens of the image (70, 140, 280 (the
default), 560, 1120: more see finer detail and take longer)}. ({"type":
"image_url", "image_url": {"url": "data:..."}} is the same.) An image takes
no task prefix. The resize is PIL's bicubic; transformers' (torchvision)
differs by a level or two on some pixels: a vector within a cosine of
0.99998 of sentence-transformers' (a text or a clip: 0.9999999).

The inputs of a request run together: sorted by length, packed into passes
of at most --batch-tokens tokens. Requests run one at a time.

Music ("model": "music", "muq-mulan" or "clap"): each input a text (a query:
"solo piano", "bossa nova with latin percussion") or {"audio": base64 of an
audio file, "start": s, "end": s} (a piece of at most --music-max-s seconds,
default 60; "truncate": true keeps the first ones): its clips of 10 s, each
model's vector the mean of its clips' (reclank/models/music/model.py). A
text and a piece compare by cosine. No task, no dimensions (other than the
model's: 1024 or 512). --music-device puts the audio towers on the GPU
(BF12) or the CPU (float32); the text towers are float32 on --music-text-device.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.models.embedgemma2.model import AUDIO_MAX_S, AUDIO_RATE, DIMS, decode_audio  # noqa: E402
from server import http as H  # noqa: E402

# the aliases of other embedding APIs (Cohere, Voyage, Nomic, Jina, ...)
ALIASES = {
    "query": "query", "searchquery": "query", "search_query": "query", "retrieval.query": "query",
    "document": "document", "doc": "document", "search_document": "document", "passage": "document",
    "retrieval.passage": "document", "retrievaldocument": "document",
    "classification": "Classification", "clustering": "Clustering",
    "similarity": "SentenceSimilarity", "text-matching": "SentenceSimilarity",
}


def _key(name):
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


class Embedder:
    """The model, its lock, and the requests' inputs to vectors."""

    def __init__(self, path, model_id, task=None, batch_tokens=8192, gpu=False, audio=None,
                 vision=None, cpu_threads=None):
        from reclank.models.embedgemma2.model import EmbeddingGemma2
        t0 = time.time()
        self.m = EmbeddingGemma2(path, gpu=gpu, audio=audio, vision=vision, cpu_threads=cpu_threads)
        self.model_id = model_id
        self.batch_tokens = batch_tokens
        self.lock = threading.Lock()
        self.names = {_key(k): k for k in self.m.prompts}
        self.task = self.prompt(task) if task else None
        # the programs of the common sizes now (a GPU program captures its graph at the first
        # run), and the audio encoder
        with self.lock:
            self.m.embed(["warm up"])
            if self.m.devices["audio"] != "off":
                self.m.audio_input(np.zeros(AUDIO_RATE, np.float32))
            if self.m.devices["vision"] != "off":
                from PIL import Image
                self.m.image_input(Image.new("RGB", (64, 64), (128, 128, 128)))
        print("embeddinggemma-2: loaded in %.1f s; text %s, audio %s, vision %s (GPU: BF12); "
              "%d tasks; default task %s" % (
                  time.time() - t0, self.m.devices["text"], self.m.devices["audio"], self.m.devices["vision"],
                  len(self.m.prompts), self.task or "none"), flush=True)

    def prompt(self, task):
        """The prompt name of a task (a prompt name of the model or an
        alias), else ValueError."""
        if not isinstance(task, str):
            raise ValueError("task must be a string")
        k = _key(task)
        name = self.names.get(k) or self.names.get(_key(ALIASES.get(task.lower(), "")))
        if name is None:
            raise ValueError("unknown task %r; the tasks: %s; the aliases: %s" % (
                task, ", ".join(sorted(self.m.prompts)), ", ".join(sorted(ALIASES))))
        return name

    def sequences(self, inputs, task, title, truncate):
        """The token ids of each input: a text with the prefix of its task,
        or token ids as they are."""
        m = self.m
        prefix = m.prompts[task] if task else ""
        if title is not None:
            if not isinstance(title, str):
                raise ValueError("title must be a string")
            if not prefix.startswith("title: none |"):
                raise ValueError("a title goes with a document task (task \"document\")")
            prefix = "title: %s |%s" % (title, prefix[len("title: none |"):])
        seqs = []
        for x in inputs:
            if isinstance(x, dict):
                if "image" in x or x.get("type") == "image_url":
                    seqs.append(("image",) + self.image(x))
                else:
                    if m.devices["audio"] == "off":
                        raise ValueError("the audio tower is off on this server")
                    seqs.append(("audio", self.samples(x, truncate)))
                continue
            if isinstance(x, str):
                s = [m.bos] + m.tok.encode(prefix + x) + [m.eos]
            elif isinstance(x, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in x):
                s = [int(t) for t in x]
                if any(t < 0 or t >= m.table.shape[0] for t in s):
                    raise ValueError("a token id is out of the vocabulary")
            else:
                raise ValueError("each input must be a string, a list of token ids, {\"audio\": ...} "
                                 "or {\"image\": ...}")
            if not s:
                raise ValueError("an input is empty")
            if len(s) > m.max_tokens:
                if not truncate:
                    raise ValueError("an input has %d tokens; the model takes at most %d "
                                     "(\"truncate\": true cuts it)" % (len(s), m.max_tokens))
                s = s[:m.max_tokens - 1] + [m.eos]
            seqs.append(s)
        return seqs

    def image(self, x):
        """The bytes and the budget of an image input (see the module)."""
        if x.get("type") == "image_url" and isinstance(x.get("image_url"), dict):
            x = dict(x, image=x["image_url"].get("url"))
        if self.m.devices["vision"] == "off":
            raise ValueError("the vision tower is off on this server")
        data = x.get("image")
        if not isinstance(data, str) or not data:
            raise ValueError("an image input needs \"image\": base64 (or a data: URI) of an image file")
        if data.startswith("data:"):
            data = data.partition(",")[2]
        try:
            raw = base64.b64decode(data, validate=False)
        except (ValueError, TypeError):
            raise ValueError("the image is not base64")
        budget = x.get("budget", 280)
        if budget not in self.m.BUDGETS:
            raise ValueError("budget must be one of %s" % (self.m.BUDGETS,))
        try:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
        except Exception as exc:
            raise ValueError("the image cannot be read: %s" % exc)
        return raw, budget

    @staticmethod
    def samples(x, truncate):
        """The 16 kHz mono samples of an audio input (see the module)."""
        if x.get("type") == "input_audio" and isinstance(x.get("input_audio"), dict):
            x = dict(x["input_audio"], **{k: v for k, v in x.items() if k in ("start", "end", "format")})
            x["audio"] = x.pop("data", None)
        data = x.get("audio")
        if not isinstance(data, str) or not data:
            raise ValueError("an object input needs \"audio\": base64 of an audio file")
        if data.startswith("data:"):
            data = data.partition(",")[2]
        try:
            raw = base64.b64decode(data, validate=False)
        except (ValueError, TypeError):
            raise ValueError("the audio is not base64")
        if x.get("format") == "f32le":
            if len(raw) % 4:
                raise ValueError("f32le audio needs a multiple of 4 bytes")
            y = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        else:
            start, end = x.get("start"), x.get("end")
            for v in (start, end):
                if v is not None and not isinstance(v, (int, float)):
                    raise ValueError("start and end are seconds")
            y = decode_audio(raw, start, end)
        if not np.all(np.isfinite(y)):
            raise ValueError("the audio has NaN or infinite samples")
        if len(y) > AUDIO_MAX_S * AUDIO_RATE:
            if not truncate:
                raise ValueError("a clip has %.1f s; an audio input takes at most %d s (\"truncate\": true "
                                 "keeps the first %d s)" % (len(y) / AUDIO_RATE, AUDIO_MAX_S, AUDIO_MAX_S))
            y = y[:AUDIO_MAX_S * AUDIO_RATE]
        if len(y) < AUDIO_RATE // 25:
            raise ValueError("a clip is shorter than 40 ms")
        return y

    def vectors(self, seqs, dim):
        """The vectors of the sequences, in their order: by length, packed
        into passes of at most batch_tokens tokens. An audio input (("audio",
        samples)) runs through the audio encoder first."""
        out = np.empty((len(seqs), dim), np.float32)
        soft = [None] * len(seqs)
        i = 0
        with self.lock:
            seqs = list(seqs)
            for k, s in enumerate(seqs):
                if isinstance(s, tuple) and s[0] == "audio":
                    seqs[k], soft[k] = self.m.audio_input(s[1])
                elif isinstance(s, tuple):
                    seqs[k], soft[k] = self.m.image_input(s[1], s[2])
            order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
            while i < len(order):
                j, n = i, 0
                while j < len(order) and (j == i or n + len(seqs[order[j]]) <= self.batch_tokens):
                    n += len(seqs[order[j]])
                    j += 1
                part = order[i:j]
                out[part] = self.m.embed_ids([seqs[k] for k in part], dim, soft=[soft[k] for k in part])
                i = j
        return out, sum(len(s) for s in seqs)


MUSIC_IDS = {"music": None, "muq-mulan": "muq", "clap": "clap"}


class MusicServer:
    """The music embeddings of a server (--music), and their lock."""

    def __init__(self, path, device, text_device, cpu_threads, max_s):
        from reclank.models.music.model import MusicEmbedder
        t0 = time.time()
        self.m = MusicEmbedder(path, audio=device, text=text_device, cpu_threads=cpu_threads)
        self.max_s = float(max_s)
        self.lock = threading.Lock()
        with self.lock:                 # the programs now (a GPU program captures its graph at its first run)
            if self.m.devices["text"] != "off":
                self.m.texts(["warm up"])
            if self.m.devices["audio"] != "off":
                one = np.sin(np.arange(10 * 48000, dtype=np.float32) * 0.05)
                self.m.audio_samples({"muq": one[::2].copy(), "clap": one})
        print("music: loaded in %.1f s; audio %s, text %s" % (
            time.time() - t0, self.m.devices["audio"], self.m.devices["text"]), flush=True)

    def vectors(self, inputs, part, truncate):
        """The vectors of the inputs (texts and {"audio": ...}): 1024 (part
        None) or the 512 of one model."""
        from reclank.models.music.model import join
        texts, pieces = [], []
        for i, x in enumerate(inputs):
            if isinstance(x, str):
                if not x.strip():
                    raise ValueError("an input is empty")
                texts.append((i, x))
            elif isinstance(x, dict):
                pieces.append((i, self.piece(x, truncate)))
            else:
                raise ValueError("each input of a music model must be a string or {\"audio\": ...}")
        out = [None] * len(inputs)
        with self.lock:
            if texts:
                if self.m.devices["text"] == "off":
                    raise ValueError("the music text towers are off on this server")
                v = self.m.texts([t for _, t in texts], parts=True)
                for k, (i, _) in enumerate(texts):
                    out[i] = {p: v[p][k] for p in v}
            for i, s in pieces:
                out[i] = self.m.audio_samples(s, parts=True)
        if part:
            return np.stack([o[part] for o in out])
        return np.stack([join({p: o[p][None] for p in o})[0] for o in out])

    def piece(self, x, truncate):
        """The samples of an audio input at each model's rate."""
        if self.m.devices["audio"] == "off":
            raise ValueError("the music audio towers are off on this server")
        data = x.get("audio")
        if not isinstance(data, str) or not data:
            raise ValueError("an object input needs \"audio\": base64 of an audio file")
        if data.startswith("data:"):
            data = data.partition(",")[2]
        try:
            raw = base64.b64decode(data, validate=False)
        except (ValueError, TypeError):
            raise ValueError("the audio is not base64")
        start, end = x.get("start"), x.get("end")
        for v in (start, end):
            if v is not None and not isinstance(v, (int, float)):
                raise ValueError("start and end are seconds")
        s = self.m.samples(raw, start, end)
        for p, y in s.items():
            rate = self.m.rates[p]
            if not np.all(np.isfinite(y)):
                raise ValueError("the audio has NaN or infinite samples")
            if len(y) > self.max_s * rate:
                if not truncate:
                    raise ValueError("a piece has %.1f s; a music input takes at most %g s (\"truncate\": true "
                                     "keeps the first ones; \"start\" and \"end\" cut a piece)"
                                     % (len(y) / rate, self.max_s))
                s[p] = y[:int(self.max_s * rate)]
            if len(y) < rate:
                raise ValueError("a piece is shorter than 1 s")
        return s


class _EmbedHandler(H._Handler):
    server_version = "reclank-embed/1.0"

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        e, mu = self.server.embedder, self.server.music
        if path in ("/health", ""):
            return self._json(200, {"status": "ok", "model": e.model_id if e else "music"})
        if not self._authorized():
            return
        if path in ("/v1/models", "/models"):
            data = []
            if e:
                data.append({"id": e.model_id, "object": "model", "owned_by": "reclank",
                             "dimensions": list(DIMS),
                             "tasks": sorted(e.m.prompts), "max_tokens": e.m.max_tokens,
                             "devices": e.m.devices, "image_budgets": list(e.m.BUDGETS)})
            if mu:
                for mid, part in MUSIC_IDS.items():
                    data.append({"id": mid, "object": "model", "owned_by": "reclank",
                                 "dimensions": [512 if part else 1024], "devices": mu.m.devices,
                                 "max_audio_s": mu.max_s})
            return self._json(200, {"object": "list", "data": data})
        self._error(404, "no route %s" % self.path, code="not_found")

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if not self._authorized():
            return
        if path not in ("/v1/embeddings", "/embeddings"):
            return self._error(404, "no route %s" % self.path, code="not_found")
        e = self.server.embedder
        try:
            body = self._body()
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            if self.server.music and (body.get("model") in MUSIC_IDS or e is None):
                return self._music(body)
            inputs = body.get("input")
            if isinstance(inputs, (str, dict)) or (isinstance(inputs, list) and inputs
                                                   and all(isinstance(t, int) for t in inputs)):
                inputs = [inputs]
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("input must be a string, a list of strings, or token ids")
            dim = int(body.get("dimensions") or e.m.dim)
            if dim not in DIMS:
                raise ValueError("dimensions must be one of %s" % (DIMS,))
            fmt = body.get("encoding_format") or "float"
            if fmt not in ("float", "base64"):
                raise ValueError("encoding_format must be \"float\" or \"base64\"")
            task = body.get("task", body.get("input_type"))
            task = e.prompt(task) if task else e.task
            seqs = e.sequences(inputs, task, body.get("title"), bool(body.get("truncate")))
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        t0 = time.time()
        try:
            vec, tokens = e.vectors(seqs, dim)
        except ValueError as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("embeddings: %d inputs, %d tokens, task %s, dim %d, %.2f s" % (
                len(seqs), tokens, task or "none", dim, time.time() - t0), flush=True)
        data = [{"object": "embedding", "index": i,
                 "embedding": (base64.b64encode(v.astype("<f4").tobytes()).decode("ascii")
                               if fmt == "base64" else v.tolist())}
                for i, v in enumerate(vec)]
        self._json(200, {"object": "list", "data": data, "model": e.model_id,
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})


    def _music(self, body):
        """A request of the music models."""
        mu = self.server.music
        mid = body.get("model") if body.get("model") in MUSIC_IDS else "music"
        try:
            inputs = body.get("input")
            if isinstance(inputs, (str, dict)):
                inputs = [inputs]
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("input must be a string, {\"audio\": ...}, or a list of them")
            dim = 512 if MUSIC_IDS[mid] else 1024
            if body.get("dimensions") not in (None, dim):
                raise ValueError("the dimensions of %s are %d" % (mid, dim))
            fmt = body.get("encoding_format") or "float"
            if fmt not in ("float", "base64"):
                raise ValueError("encoding_format must be \"float\" or \"base64\"")
            t0 = time.time()
            vec = mu.vectors(inputs, MUSIC_IDS[mid], bool(body.get("truncate")))
        except (ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        if not getattr(self.server, "quiet", False):
            print("embeddings: %s, %d inputs, %.2f s" % (mid, len(inputs), time.time() - t0), flush=True)
        data = [{"object": "embedding", "index": i,
                 "embedding": (base64.b64encode(v.astype("<f4").tobytes()).decode("ascii")
                               if fmt == "base64" else v.tolist())}
                for i, v in enumerate(vec)]
        self._json(200, {"object": "list", "data": data, "model": mid,
                         "usage": {"prompt_tokens": 0, "total_tokens": 0}})


def main(argv=None):
    ap = argparse.ArgumentParser(description="the embedding server (EmbeddingGemma 2)")
    ap.add_argument("-m", "--model", default=None, help="the embeddinggemma-2 directory")
    ap.add_argument("--music", default=None,
                    help="a directory of the music embeddings (clap/ and muq-mulan/: scripts/convert_music.py)")
    ap.add_argument("--music-device", choices=("gpu", "cpu", "off"), default=None,
                    help="the music audio towers (default: gpu with --gpu, else cpu)")
    ap.add_argument("--music-text-device", choices=("gpu", "cpu", "off"), default=None,
                    help="the music text towers (default: as --music-device)")
    ap.add_argument("--music-max-s", type=float, default=60.0, help="the most seconds of a music input")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--model-id", default="embeddinggemma-2")
    ap.add_argument("--task", default=None,
                    help="the task of a request that names none (a prompt name of the model or "
                         "an alias: query, document, ...); default none, the texts as they are")
    ap.add_argument("--batch-tokens", type=int, default=8192,
                    help="the most tokens of one pass (default %(default)s)")
    ap.add_argument("--gpu", action="store_true", help="the text model on the GPU, its matrices in BF12")
    ap.add_argument("--audio-device", choices=("gpu", "cpu", "off"), default=None,
                    help="the audio tower (default: as the text model)")
    ap.add_argument("--vision-device", choices=("gpu", "cpu", "off"), default=None,
                    help="the vision tower (default: as the text model)")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="the team of a program on the CPU (default: the CPUs of the process)")
    ap.add_argument("--quiet", action="store_true")
    H.add_http_args(ap)
    a = ap.parse_args(argv)
    if not a.model and not a.music:
        ap.error("give -m (EmbeddingGemma 2), --music, or both")
    embedder = Embedder(a.model, a.model_id, a.task, a.batch_tokens, a.gpu, a.audio_device,
                        a.vision_device, a.cpu_threads) if a.model else None
    music = None
    if a.music:
        dev = a.music_device or ("gpu" if a.gpu else "cpu")
        music = MusicServer(a.music, dev, a.music_text_device or (dev if dev != "off" else "cpu"),
                            a.cpu_threads, a.music_max_s)
    httpd = H._Server((a.host, a.port), _EmbedHandler)
    httpd.embedder = embedder
    httpd.music = music
    httpd.quiet = a.quiet
    httpd.api_key = a.api_key or None
    httpd.cors_origin = a.cors_origin or None
    httpd.max_body = int(a.max_body_mb * (1 << 20))
    print("reclank embedding server http://%s:%d/v1  model=%s  key %s" % (
        a.host, httpd.server_address[1], " ".join(([a.model_id] if embedder else []) +
                                                  (list(MUSIC_IDS) if music else [])),
        "required" if httpd.api_key else "OFF"),
        flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
