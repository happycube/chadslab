#!/usr/bin/env python3
"""Embed audio files (MP3 or any format of ffmpeg) with EmbeddingGemma 2, a
clip at a time: one vector for each clip.

The model takes at most 30 s of audio in one input (its feature extractor
cuts at 480000 samples of 16 kHz), so a file goes in clips of --clip seconds
(default 30), a clip every --hop seconds (default the clip: no overlap; less
than the clip overlaps them, as 30 / 15 for a search that should not miss
what a cut splits). The last clip ends at the end of the file; one shorter
than --min-last seconds joins the clip before it when the two fit in 30 s,
else it stays.

The clips go to the embedding server (scripts/serve_embed.py, port 8082) as
16 kHz float32 samples, --batch clips a request; --local loads the model in
this process instead. For each file it writes <out-dir>/<name>.emb.npz:

    vectors   [clips, dim] float32, L2-normalized (the same space as the
              server's text vectors: a text query can search them)
    start     [clips] the start of each clip in seconds
    end       [clips] its end
    source (the path of the file), model, dim, clip, hop

    python scripts/embed_audio.py talk.mp3 podcast/*.mp3 --out-dir emb/
    python scripts/embed_audio.py talk.mp3 --clip 30 --hop 15 --dimensions 256
    python scripts/embed_audio.py talk.mp3 --query "a dog barking"
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.models.embedgemma2.model import AUDIO_MAX_S, AUDIO_RATE, DIMS, decode_audio  # noqa: E402


def clips(n, clip, hop, min_last):
    """The (start, end) sample ranges of n samples."""
    c, h, m = int(round(clip * AUDIO_RATE)), int(round(hop * AUDIO_RATE)), int(round(min_last * AUDIO_RATE))
    out = []
    a = 0
    while a < n:
        b = min(n, a + c)
        out.append([a, b])
        if b == n:
            break
        a += h
    if len(out) > 1 and out[-1][1] - out[-1][0] < m and out[-1][1] - out[-2][0] <= AUDIO_MAX_S * AUDIO_RATE:
        out[-2][1] = out.pop()[1]
    # a tail that an earlier clip already holds (an overlap at the end)
    while len(out) > 1 and out[-1][1] <= out[-2][1]:
        out.pop()
    return [(a, b) for a, b in out if b - a >= AUDIO_RATE // 25]


class Server:
    def __init__(self, url, key):
        self.url = url.rstrip("/") + "/v1/embeddings"
        self.key = key

    def post(self, body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.key})
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise SystemExit("the server: %d %s" % (e.code, e.read().decode("utf-8", "replace")[:500]))
        except urllib.error.URLError as e:
            raise SystemExit("no embedding server at %s (%s); start it (start_embed.sh) or use --local"
                             % (self.url, e.reason))

    def audio(self, parts, dim):
        body = {"input": [{"audio": base64.b64encode(p.astype("<f4").tobytes()).decode("ascii"),
                           "format": "f32le"} for p in parts], "dimensions": dim}
        return np.array([d["embedding"] for d in self.post(body)["data"]], np.float32)

    def text(self, texts, task, dim):
        r = self.post({"input": texts, "task": task, "dimensions": dim})
        return np.array([d["embedding"] for d in r["data"]], np.float32)


class Local:
    def __init__(self, path):
        from reclank.models.embedgemma2.model import EmbeddingGemma2
        self.m = EmbeddingGemma2(path)

    def audio(self, parts, dim):
        ins = [self.m.audio_input(p) for p in parts]
        return self.m.embed_ids([i for i, _ in ins], dim, soft=[r for _, r in ins])

    def text(self, texts, task, dim):
        return self.m.embed(texts, task, dim)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="audio files (MP3, WAV, FLAC, M4A, ...)")
    ap.add_argument("--clip", type=float, default=30.0, help="seconds a clip (at most 30; default 30)")
    ap.add_argument("--hop", type=float, default=None, help="seconds from a clip to the next (default the clip)")
    ap.add_argument("--min-last", type=float, default=5.0,
                    help="a last clip shorter than this joins the one before when they fit (default 5)")
    ap.add_argument("--dimensions", type=int, default=768, choices=DIMS)
    ap.add_argument("--batch", type=int, default=4, help="clips a request (default 4)")
    ap.add_argument("--out-dir", default=".", help="where the .emb.npz files go (default here)")
    ap.add_argument("--url", default="http://127.0.0.1:8082")
    ap.add_argument("--key", default=os.environ.get("RECLANK_API_KEY", "change-me"))
    ap.add_argument("--local", metavar="MODEL_DIR", default=None,
                    help="load the model in this process (the embeddinggemma-2 directory), no server")
    ap.add_argument("--query", action="append", default=[],
                    help="a text to search the clips for: print the best clips (several allowed)")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    hop = a.hop if a.hop is not None else a.clip
    if not 0 < a.clip <= AUDIO_MAX_S:
        ap.error("--clip must be in (0, %d]" % AUDIO_MAX_S)
    if not 0 < hop:
        ap.error("--hop must be above 0")
    model = Local(a.local) if a.local else Server(a.url, a.key)
    os.makedirs(a.out_dir, exist_ok=True)
    queries = model.text(a.query, "query", a.dimensions) if a.query else None
    for path in a.files:
        t0 = time.time()
        x = decode_audio(path)
        spans = clips(len(x), a.clip, hop, a.min_last)
        if not spans:
            print("%s: no audio" % path, flush=True)
            continue
        vec = []
        for i in range(0, len(spans), a.batch):
            vec.append(model.audio([x[s:e] for s, e in spans[i:i + a.batch]], a.dimensions))
            print("\r%s: %d / %d clips" % (path, min(i + a.batch, len(spans)), len(spans)),
                  end="", file=sys.stderr, flush=True)
        print("\r\033[K", end="", file=sys.stderr, flush=True)
        vec = np.concatenate(vec)
        start = np.array([s for s, _ in spans], np.float64) / AUDIO_RATE
        end = np.array([e for _, e in spans], np.float64) / AUDIO_RATE
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(a.out_dir, name + ".emb.npz")
        np.savez(out, vectors=vec, start=start, end=end, source=os.path.abspath(path),
                 model="embeddinggemma-2", dim=a.dimensions, clip=a.clip, hop=hop)
        dt = time.time() - t0
        print("%s: %.1f s of audio, %d clips -> %s  (%.1f s, %.1fx real time)" % (
            path, len(x) / AUDIO_RATE, len(spans), out, dt, len(x) / AUDIO_RATE / max(dt, 1e-9)), flush=True)
        if queries is not None:
            for q, qv in zip(a.query, queries):
                sim = vec @ qv
                best = np.argsort(-sim)[:a.top]
                print("  %r: %s" % (q, ", ".join("%d:%02d-%d:%02d (%.3f)" % (
                    start[k] // 60, start[k] % 60, end[k] // 60, end[k] % 60, sim[k]) for k in best)),
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
