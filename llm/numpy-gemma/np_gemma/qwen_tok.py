"""The tokenizer of the Qwen3.5 and Qwen3.6 models: byte-level BPE.

The pipeline of tokenizer.json has these steps:

1. Split the text at the special tokens (added_tokens). A special token is
   one id.
2. Normalize each other part with NFC.
3. Split each part with the pattern of the pre-tokenizer (split_words).
4. Change the UTF-8 bytes of each piece to characters with the byte table of
   GPT-2 (bytes_to_unicode).
5. Apply the BPE merges to each piece, and look up the ids.

The decoder changes the characters back to bytes, and the bytes to text.

The pattern of step 3 uses the Unicode classes \\p{L}, \\p{M}, and \\p{N}.
The re module of Python does not have them, and the regex module is not
always installed. Thus split_words follows the pattern with unicodedata:

    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}
    | ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

scripts/check_qwen_tok.py compares the ids with the tokenizers library.
"""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

_CONTRACTIONS = ("s", "t", "re", "ve", "m", "ll", "d")


@lru_cache(maxsize=65536)
def _cat(c):
    """Return the class of one character: L, M, N, S (white space), or O."""
    k = unicodedata.category(c)[0]
    if k in "LMN":
        return k
    # \\s of the regex module is the White_Space property. str.isspace also
    # takes U+001C to U+001F, which are not White_Space.
    if c.isspace() and c not in "\x1c\x1d\x1e\x1f":
        return "S"
    return "O"


def split_words(text):
    """Return the pieces of text, as the pre-tokenizer of Qwen3.5 splits it.

    Each branch below is one alternative of the pattern, in the order of the
    pattern. The first alternative that matches at a position wins.
    """
    out = []
    n = len(text)
    pos = 0
    while pos < n:
        end = _match(text, pos, n)
        out.append(text[pos:end])
        pos = end
    return out


def _match(text, pos, n):
    c = text[pos]
    k = _cat(c)
    # 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c == "'":
        low = text[pos + 1:pos + 3].lower()
        for s in _CONTRACTIONS:
            if low.startswith(s):
                return pos + 1 + len(s)
    # 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if c not in "\r\n" and k not in "LN" and pos + 1 < n and _cat(text[pos + 1]) in "LM":
        e = pos + 2
        while e < n and _cat(text[e]) in "LM":
            e += 1
        return e
    if k in "LM":
        e = pos + 1
        while e < n and _cat(text[e]) in "LM":
            e += 1
        return e
    # 3. \p{N}
    if k == "N":
        return pos + 1
    # 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    s = pos + 1 if (c == " " and pos + 1 < n and _cat(text[pos + 1]) == "O") else pos
    if _cat(text[s]) == "O":
        e = s + 1
        while e < n and _cat(text[e]) == "O":
            e += 1
        while e < n and text[e] in "\r\n":
            e += 1
        return e
    # 5, 6, 7: the run of white space from pos.
    e = pos
    while e < n and _cat(text[e]) == "S":
        e += 1
    # 5. \s*[\r\n]+: up to the last line break of the run.
    last = -1
    for j in range(pos, e):
        if text[j] in "\r\n":
            last = j
    if last >= 0:
        return last + 1
    # 6. \s+(?!\S): the run, less its last character before a non-space.
    if e == n:
        return e
    if e - 1 > pos:
        return e - 1
    # 7. \s+
    return e


def bytes_to_unicode():
    """Return the byte table of GPT-2: byte value -> a printable character."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    k = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class QwenTokenizer:
    """Convert text to token ids and back, as tokenizer.json of Qwen3.5 does.

        tok = QwenTokenizer("models/.../tokenizer.json")
        ids = tok.encode("Hello")
        text = tok.decode(ids)
    """

    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        m = data["model"]
        if m.get("type") != "BPE":
            raise ValueError("expected a BPE model")
        self.vocab = m["vocab"]
        self.ranks = {}
        for r, pair in enumerate(m["merges"]):
            a, b = pair.split(" ", 1) if isinstance(pair, str) else pair
            self.ranks[(a, b)] = r
        self.special = {a["content"]: a["id"] for a in data.get("added_tokens", [])}
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.id_to_token.update({i: t for t, i in self.special.items()})
        self.byte_char = bytes_to_unicode()
        self.char_byte = {c: b for b, c in self.byte_char.items()}
        # The longest special tokens first, so a longer one wins.
        self._specials = sorted(self.special, key=len, reverse=True)
        self._cache = {}
        self.stop_ids = [self.special[s] for s in ("<|im_end|>", "<|endoftext|>")
                         if s in self.special]

    def _bpe(self, piece):
        """Return the ids of one piece (a string of byte characters)."""
        ids = self._cache.get(piece)
        if ids is not None:
            return ids
        parts = list(piece)
        ranks = self.ranks
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            # Merge every pair of this rank, from the left, as BPE does.
            a, b = parts[bi], parts[bi + 1]
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] == a and parts[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        ids = [self.vocab[p] for p in parts]
        if len(self._cache) < 200000:
            self._cache[piece] = ids
        return ids

    def _split_special(self, text):
        """Yield (is_special, part) for the text, at the special tokens."""
        pos = 0
        n = len(text)
        while pos < n:
            nxt, which = n, None
            for s in self._specials:
                j = text.find(s, pos)
                if j != -1 and j < nxt:
                    nxt, which = j, s
            if nxt > pos:
                yield False, text[pos:nxt]
            if which is None:
                break
            yield True, which
            pos = nxt + len(which)

    def encode(self, text):
        """Return the token ids of text."""
        out = []
        bc = self.byte_char
        for special, part in self._split_special(text):
            if special:
                out.append(self.special[part])
                continue
            for w in split_words(unicodedata.normalize("NFC", part)):
                out += self._bpe("".join(bc[b] for b in w.encode("utf-8")))
        return out

    def decode(self, ids, skip_special=False):
        """Return the text of token ids."""
        buf = bytearray()
        text = []
        for i in ids:
            t = self.id_to_token.get(int(i), "")
            if t in self.special:
                if buf:
                    text.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                if not skip_special:
                    text.append(t)
                continue
            buf += bytes(self.char_byte[c] for c in t)
        if buf:
            text.append(buf.decode("utf-8", errors="replace"))
        return "".join(text)
