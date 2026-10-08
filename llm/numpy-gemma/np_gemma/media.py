"""Soft tokens: the rows of images and audio in a prompt.

The chat template writes one placeholder for each image (<|image|>) and each
clip (<|audio|>). expand() puts the soft tokens in place of each placeholder,
as the processor of transformers does:

    <|image|>  ->  <|image>  <|image|> x n  <image|>
    <|audio|>  ->  <|audio>  <|audio|> x n  <audio|>

and gives a Span for each: its first position, its rows (the input of layer
0 at those positions, in place of the token embeddings), and whether its
tokens see each other (use_bidirectional_attention "vision": the tokens of
one image, in the layers with a window).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# The Gemma 4 ids (config.json of the 12B, the same in the E4B and the 26B).
IMAGE_TOKEN = 258880
AUDIO_TOKEN = 258881
VIDEO_TOKEN = 258884
BOI, EOI = 255999, 258882
BOA, EOA = 256000, 258883


@dataclass
class Span:
    """The soft tokens of one image or clip at positions start to end - 1."""
    start: int
    rows: np.ndarray
    bidir: bool
    kind: str = "image"
    key: str = ""
    grid: tuple = None      # (rows, columns) of the tokens of an image (M-RoPE of Qwen)

    @property
    def end(self):
        return self.start + self.rows.shape[0]


@dataclass
class Media:
    """An image or a clip before expand(): its soft rows and a key (a hash of
    the source, so that a cache does not take one image for another)."""
    kind: str
    rows: np.ndarray
    key: str = ""
    bidir: bool = False
    grid: tuple = None


def media_key(data):
    """Return a short hash of the bytes (or the array) of a source."""
    if isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data).tobytes()
    return hashlib.sha1(bytes(data)).hexdigest()[:16]


def video_text(frames):
    """Return the prompt text of a video: for each frame (seconds, rows), its
    time as mm:ss, then <|image>, a <|video|> for each soft row, and
    <image|>, joined by spaces (replace_video_token of transformers)."""
    return " ".join("%02d:%02d <|image>%s<image|>" % (int(t // 60), int(t % 60),
                                                     "<|video|>" * rows.shape[0])
                    for t, rows in frames)


def expand(ids, items):
    """Put the soft tokens of items (a list of Media, in the order of their
    placeholders) into the ids. Return (ids, spans).

    A video is already in the ids (video_text): each run of <|video|> takes
    the next item, a frame (kind "video"), whose rows must have that count."""
    out, spans = [], []
    it = iter(items)
    ids = [int(t) for t in ids]
    i = 0
    while i < len(ids):
        t = ids[i]
        if t == VIDEO_TOKEN:
            k = i
            while k < len(ids) and ids[k] == VIDEO_TOKEN:
                k += 1
            m = next(it, None)
            if m is None or m.kind != "video" or m.rows.shape[0] != k - i:
                raise ValueError("a run of %d video tokens has no frame of that size" % (k - i))
            spans.append(Span(len(out), np.ascontiguousarray(m.rows, np.float32), m.bidir,
                              m.kind, m.key))
            out.extend(ids[i:k])
            i = k
            continue
        i += 1
        if t not in (IMAGE_TOKEN, AUDIO_TOKEN):
            out.append(int(t))
            continue
        try:
            m = next(it)
        except StopIteration:
            raise ValueError("the prompt has more media placeholders than media") from None
        want = IMAGE_TOKEN if m.kind == "image" else AUDIO_TOKEN
        if t != want:
            raise ValueError("the placeholder %d does not match the media kind %r" % (t, m.kind))
        bo, eo = (BOI, EOI) if m.kind == "image" else (BOA, EOA)
        out.append(bo)
        spans.append(Span(len(out), np.ascontiguousarray(m.rows, np.float32), m.bidir,
                          m.kind, m.key))
        out.extend([want] * m.rows.shape[0])
        out.append(eo)
    if next(it, None) is not None:
        raise ValueError("more media than placeholders in the prompt")
    return out, spans


def keys(ids, spans):
    """Return a key for each position: the id, or (the id, the media key, the
    index) in a span. Two prompts share a cache prefix only where the keys
    agree."""
    k = list(ids)
    for sp in spans:
        for j in range(sp.start, sp.end):
            k[j] = (ids[j], sp.key, j - sp.start)
    return k


def shift(spans, offset):
    """Return the spans with positions offset (for a prompt that starts at a
    later position)."""
    return [Span(sp.start + offset, sp.rows, sp.bidir, sp.kind, sp.key, sp.grid) for sp in spans]


# ---- Qwen3.6 ------------------------------------------------------------------

# The Qwen3.6 ids (config.json of Qwen/Qwen3.6-35B-A3B).
QWEN_VISION_START, QWEN_VISION_END = 248053, 248054
QWEN_IMAGE_PAD, QWEN_VIDEO_PAD = 248056, 248057


def expand_qwen(ids, items):
    """The expand() of Qwen: the template writes <|vision_start|><|image_pad|>
    <|vision_end|> for each image; the <|image_pad|> becomes one token for
    each row of the image (Qwen3VLProcessor). A video has one <|video_pad|>
    for each pair of frames (vision_qwen.video_text), and an item of kind
    "video" for each. Return (ids, spans)."""
    out, spans = [], []
    it = iter(items)
    for t in ids:
        t = int(t)
        if t not in (QWEN_IMAGE_PAD, QWEN_VIDEO_PAD):
            out.append(t)
            continue
        m = next(it, None)
        if m is None:
            raise ValueError("the prompt has more image or video placeholders than media")
        want = "image" if t == QWEN_IMAGE_PAD else "video"
        if m.kind != want:
            raise ValueError("the placeholder of %s has the media %r" % (want, m.kind))
        spans.append(Span(len(out), np.ascontiguousarray(m.rows, np.float32), False, m.kind,
                          m.key, m.grid))
        out.extend([t] * m.rows.shape[0])
    if next(it, None) is not None:
        raise ValueError("more media than placeholders in the prompt")
    return out, spans


def mrope_positions(n, spans, start=0):
    """The M-RoPE positions (3, n) of a prompt of n tokens (get_rope_index of
    transformers): a text token has (p, p, p); the token (r, c) of an image
    of gh x gw tokens that starts at p has (p, p + r, p + c), and the text
    after it starts at p + max(gh, gw). spans have prompt positions; start
    is the position of the first token."""
    pos = np.empty((3, n), np.int64)
    p = start
    i = 0
    for sp in sorted(spans, key=lambda s: s.start):
        k = sp.start - i
        pos[:, i:sp.start] = np.arange(p, p + k)
        p += k
        gh, gw = sp.grid
        assert gh * gw == sp.rows.shape[0], "the grid of an image does not match its rows"
        r, c = np.divmod(np.arange(gh * gw), gw)
        pos[0, sp.start:sp.end] = p
        pos[1, sp.start:sp.end] = p + r
        pos[2, sp.start:sp.end] = p + c
        p += max(gh, gw)
        i = sp.end
    pos[:, i:] = np.arange(p, p + n - i)
    return pos
