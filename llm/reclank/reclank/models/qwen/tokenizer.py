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
The re module of Python does not have them (written out as ranges, re
takes 3x the time of the Python code), and the regex module is not always
installed. Thus split_words follows the pattern with unicodedata:

    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}
    | ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

and encode takes fast_split: re with the ASCII classes for a text of ASCII
alone, else the pattern in the regex module when it is there, else
split_words. encode also keeps the ids of each piece (the pieces repeat),
and changes the bytes of a new piece to the characters of BPE with
str.translate.

scripts/check_qwen_tok.py compares the ids with the tokenizers library.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_CONTRACTIONS = ("s", "t", "re", "ve", "m", "ll", "d")
# The mark of escape(): a character of the private use area in place of the
# "<" of a special token in the text of a message.
ESCAPE = "\ue000"


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


# The pattern itself, for the regex module (when installed): the pieces of
# split_words, at about 1.6x its speed (the characters of a Unicode version
# after that of unicodedata aside).
PATTERN = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"""
           r"""| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
# The pattern of a text of ASCII characters alone, for re (3x the speed of
# split_words): there \p{L} is [A-Za-z], \p{N} [0-9], no \p{M}, and \s the
# White_Space characters (not the \x1c-\x1f of the \s of re). The "(?i)" of
# the contractions without the case folds of re beyond ASCII.
_S = r"\t\n\x0b\x0c\r "
_ASCII_RE = re.compile(
    r"'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]"
    r"| ?[^{S}A-Za-z0-9]+[\r\n]*|[{S}]*[\r\n]+|[{S}]+(?![^{S}])|[{S}]+".format(S=_S))


@lru_cache(maxsize=1)
def _unicode_split():
    """The split of a text with other characters than ASCII: findall of the
    regex module, else split_words."""
    try:
        import regex
    except ImportError:
        return split_words
    return regex.compile(PATTERN).findall


def fast_split(text):
    """split_words, faster: re for ASCII text, else the regex module."""
    return _ASCII_RE.findall(text) if text.isascii() else _unicode_split()(text)


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
        # escape() keeps all of them as text in a message: the chat structure
        # (<|im_end|>), the media (<|image_pad|>), and <think>, <tool_call>.
        self._structural = list(self._specials)
        # ESCAPE stands for "<" only in front of the rest of a special token:
        # a U+E000 of the text itself stays.
        self._unescape = re.compile(re.escape(ESCAPE) + "(?=" + "|".join(
            re.escape(t[1:]) for t in self._specials) + ")")
        self._cache = {}
        # the bytes of a piece (as latin-1 characters) to the characters of BPE
        self._bytes_tr = dict(self.byte_char)
        # the ids of each piece of text (before its bytes)
        self._words = {}
        # the special tokens in the order of _specials: the first match of
        # the alternation at a position is the longest
        self._special_re = re.compile("(" + "|".join(re.escape(s) for s in self._specials) + ")") \
            if self._specials else None
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
        if self._special_re is None:
            if text:
                yield False, text
            return
        for i, part in enumerate(self._special_re.split(text)):
            if i % 2:
                yield True, part
            elif part:
                yield False, part

    def escape(self, text):
        """text with each special token (<|im_end|>, <|image_pad|>, <think>,
        <tool_call>, ...) marked as plain text: encode gives the ids of its
        characters, not the special token. For the text of a message (a file
        in a tool result can hold "<|im_end|>"; as the token it ends the turn
        of the prompt, and the model copies the token and stops)."""
        if "<" not in text:
            return text
        for t in self._structural:
            if t in text:
                text = text.replace(t, ESCAPE + t[1:])
        return text

    def unescape(self, text):
        """text with the "<" of the special tokens that escape() marked."""
        return self._unescape.sub("<", text) if ESCAPE in text else text

    def encode(self, text):
        """Return the token ids of text. A special token marked by escape()
        is plain text."""
        out = []
        words, tr = self._words, self._bytes_tr
        for special, part in self._split_special(text):
            if special:
                out.append(self.special[part])
                continue
            if ESCAPE in part:
                part = self.unescape(part)
            if not part.isascii():
                part = unicodedata.normalize("NFC", part)
            for w in fast_split(part):
                ids = words.get(w)
                if ids is None:
                    ids = self._bpe(w.encode("utf-8").decode("latin-1").translate(tr))
                    if len(words) < 500000:
                        words[w] = ids
                out += ids
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
