"""The tokenizer of CLIP's text models (transformers' CLIPTokenizer of a
vocab.json and merges.txt): byte-level BPE with "</w>" at the end of a word.

Ported from transformers (src/transformers/models/clip/tokenization_clip.py; Apache License 2.0;
Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team). See THIRD_PARTY_NOTICES.md.

    the added tokens (<|startoftext|>, <|endoftext|>, and the pad token when it
        is a token of its own: SD 2's "!") split the text first, each one id
    each other part: NFC, each run of white space one space, lower case
    the pieces of the pattern
        <|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+
        (the white space between them dropped)
    the UTF-8 bytes of a piece as the characters of GPT-2's byte table, its
        last character with "</w>", the merges of merges.txt in their order
    <|startoftext|>, the ids (at most 75), <|endoftext|>, then the pad token
        up to 77 (the pipeline's padding="max_length", truncation)

The pattern: the regex module when it is there, else the classes of
unicodedata (qwen.tokenizer._cat: L, N, white space, other).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache

from reclank.models.qwen.tokenizer import _cat, bytes_to_unicode

try:
    import regex as _regex
except ImportError:         # the classes of unicodedata
    _regex = None

PATTERN = r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
_CONTRACTIONS = ("s", "t", "re", "ve", "m", "ll", "d")


def split_words(text):
    """The pieces of the pattern in a normalized text (no regex module)."""
    out = []
    n, pos = len(text), 0
    while pos < n:
        c = text[pos]
        k = _cat(c)
        if k == "S":
            pos += 1
            continue
        if text.startswith("<|startoftext|>", pos) or text.startswith("<|endoftext|>", pos):
            e = text.index(">", pos) + 1
        elif c == "'" and any(text.startswith(s, pos + 1) for s in _CONTRACTIONS):
            e = pos + 1 + next(len(s) for s in _CONTRACTIONS if text.startswith(s, pos + 1))
        elif k == "L":
            e = pos + 1
            while e < n and _cat(text[e]) == "L":
                e += 1
        elif k == "N":
            e = pos + 1
        else:
            e = pos + 1
            while e < n and _cat(text[e]) in "MO":
                e += 1
        out.append(text[pos:e])
        pos = e
    return out


class CLIPTokenizer:
    """The tokenizer of a tokenizer directory (vocab.json, merges.txt,
    tokenizer_config.json, special_tokens_map.json). encode(text) gives
    the 77 ids of the pipeline."""

    def __init__(self, path, length=77):
        self.vocab = json.load(open(os.path.join(path, "vocab.json"), encoding="utf-8"))
        merges = open(os.path.join(path, "merges.txt"), encoding="utf-8").read().split("\n")
        merges = [m for m in merges[1:] if m and not m.startswith("#version")]
        self.rank = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.byte = bytes_to_unicode()
        self.length = length
        cfg = json.load(open(os.path.join(path, "tokenizer_config.json")))
        smap = json.load(open(os.path.join(path, "special_tokens_map.json")))
        tok = lambda v: v["content"] if isinstance(v, dict) else v  # noqa: E731
        # special_tokens_map.json wins over tokenizer_config.json (SD 2.0-base: pad "!")
        get = lambda k: tok(smap.get(k, cfg.get(k)))  # noqa: E731
        self.bos, self.eos = self.vocab[get("bos_token")], self.vocab[get("eos_token")]
        pad = get("pad_token")
        self.pad = self.vocab[pad]
        self.added = sorted({get("bos_token"), get("eos_token"), pad}, key=len, reverse=True)
        self._re = _regex.compile(PATTERN) if _regex is not None else None

    @lru_cache(maxsize=65536)
    def _bpe(self, piece):
        """The ids of one piece (its bytes' characters)."""
        word = list(piece[:-1]) + [piece[-1] + "</w>"]
        while len(word) > 1:
            best, at = None, -1
            for i in range(len(word) - 1):
                r = self.rank.get((word[i], word[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            a, b = word[at], word[at + 1]
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        return tuple(self.vocab[w] for w in word)

    def tokens(self, text):
        """The ids of a text (no <|startoftext|>, <|endoftext|>, padding)."""
        ids = []
        alt = "|".join(re.escape(a) for a in self.added)
        for i, part in enumerate(re.split("(%s)" % alt, text)):
            if i % 2:
                ids.append(self.vocab[part])
                continue
            part = re.sub(r"\s+", " ", unicodedata.normalize("NFC", part)).lower()
            pieces = self._re.findall(part) if self._re is not None else split_words(part)
            for p in pieces:
                ids.extend(self._bpe("".join(self.byte[b] for b in p.encode("utf-8"))))
        return ids

    def encode(self, text):
        """The ids (77) of a prompt: <|startoftext|>, at most 75 ids,
        <|endoftext|>, the pad token."""
        ids = [self.bos] + self.tokens(text)[:self.length - 2] + [self.eos]
        return ids + [self.pad] * (self.length - len(ids))
