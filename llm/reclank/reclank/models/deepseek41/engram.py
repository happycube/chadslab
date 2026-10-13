"""The n-gram hashes of the engram layers of DeepSeek V4.1 Flash (the
reference's engram.py, DEEPSEEK_V41_PLAN.md 2.3) without torch, tokenizers
or sympy.

- token_map(tokenizer.json): each id to a compressed id; ids whose decoded
  text normalizes alike (NFKC, NFD, the accents stripped, lowercase, runs of
  space, tab and newline to one space, stripped; the accents: every
  combining mark, Mn, Mc and Me, as StripAccents; a token that is exactly one
  space stays one space) share one. A token whose bytes are not whole UTF-8
  keys by its raw form. The ids go in first-seen order: 99092 of them.
- the primes: for each layer, n-gram size and head, the next prime above
  engram_vocab_size - 1 not yet handed out (in that order).
- the multipliers: numpy's default_rng(10007 layer), odd, bounded so that
  id x multiplier fits int64.
- Hashes.ids(tokens, start): for each position the 2-, 3- and 4-gram ending
  there (the look-back stops at the start, where the pad id fills), the
  running XOR of id x multiplier, mod each head's prime, plus the offset of
  its range: [positions, layers, 24].
"""
from __future__ import annotations

import json
import re
import unicodedata

import numpy as np

# Rust char::is_whitespace (the White_Space property): the Strip normalizer
_WS = set("\t\n\x0b\x0c\r \x85\xa0     　") | {chr(c) for c in range(0x2000, 0x200B)}
_RUNS = re.compile(r"[ \t\r\n]+")
SENTINEL = ""


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def normalize(text):
    t = unicodedata.normalize("NFKC", text)
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) not in ("Mn", "Mc", "Me"))   # is_combining_mark
    t = t.lower()
    t = _RUNS.sub(" ", t)
    if t == " ":
        t = SENTINEL
    i, j = 0, len(t)
    while i < j and t[i] in _WS:
        i += 1
    while j > i and t[j - 1] in _WS:
        j -= 1
    return t[i:j].replace(SENTINEL, " ")


def token_map(tokenizer_json):
    """(the compressed id of each token id, the count of compressed ids)."""
    tj = json.load(open(tokenizer_json))
    vocab = dict(tj["model"]["vocab"])
    special = {}
    for t in tj["added_tokens"]:
        vocab[t["content"]] = t["id"]
        special[t["id"]] = t["content"]
    n_ids = max(vocab.values()) + 1
    by_id = [None] * n_ids
    for s, i in vocab.items():
        by_id[i] = s
    u2b = _bytes_to_unicode()
    key_to_new, lookup = {}, [0] * n_ids
    for i in range(n_ids):
        s = by_id[i]
        if s is None:
            text, raw = "", ""
        elif i in special:
            text = raw = s
        else:
            raw = s
            text = bytes(u2b[c] for c in s).decode("utf-8", errors="replace")
        if "�" in text:
            key = raw
        else:
            nt = normalize(text)
            key = nt if nt else text
        new = key_to_new.get(key)
        if new is None:
            new = key_to_new[key] = len(key_to_new)
        lookup[i] = new
    return np.array(lookup, dtype=np.int64), len(key_to_new)


def _is_prime(n):
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes(layers, max_ngram, n_heads, vocab_size):
    out, seen = [], set()
    for _ in layers:
        per = []
        for _ in range(max_ngram - 1):
            sizes, cur = [], vocab_size - 1
            for _ in range(n_heads):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per.append(sizes)
        out.append(per)
    return np.array(out, dtype=np.int64)          # [layers, max_ngram - 1, heads]


def multipliers(layers, max_ngram, vocab):
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    rows = []
    for lid in layers:
        g = np.random.default_rng(10007 * lid)
        rows.append(g.integers(low=0, high=bound, size=(max_ngram,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)                         # [layers, max_ngram]


DEAD = -1          # the cache entry of a token of an image span


class Hashes:
    """The hash ids of the engram layers, with the ids of the sequence so far."""

    def __init__(self, cfg, tokenizer_json, max_len):
        self.layers = list(cfg["engram_layer_ids"])
        self.n = int(cfg["engram_max_ngram_size"])
        self.heads = int(cfg["engram_n_heads"])
        self.map, nv = token_map(tokenizer_json)
        assert nv == cfg["engram_compressed_vocab_size"], (nv, cfg["engram_compressed_vocab_size"])
        self.pad = int(self.map[cfg["engram_pad_token_id"] if "engram_pad_token_id" in cfg else cfg["engram_pad_id"]])
        self.primes = primes(self.layers, self.n, self.heads, int(cfg["engram_vocab_size"]))
        flat = self.primes.reshape(len(self.layers), -1)
        self.offsets = np.concatenate([np.zeros((len(self.layers), 1), np.int64),
                                       np.cumsum(flat, 1)[:, :-1]], 1)
        self.mult = multipliers(self.layers, self.n, nv)
        self.cache = np.zeros(max_len, np.int64)

    def ids(self, tokens, start, dead=None):
        """[len(tokens), layers, (n - 1) heads] for the positions start.. of
        tokens. dead [len(tokens)] (bool): the tokens of an image span, which
        take no part in an n-gram: the look-back stops at them as at the start
        (engram.py NgramHashState: DEAD)."""
        tokens = np.asarray(tokens, np.int64)
        L = len(tokens)
        c = self.map[tokens]
        if dead is not None:
            c = np.where(dead, DEAD, c)
        self.cache[start:start + L] = c
        pos = np.arange(start, start + L)
        cols = []
        blocked = np.zeros(L, bool)
        for shift in range(self.n):
            src = self.cache[np.maximum(pos - shift, 0)]
            blocked |= (pos < shift) | (src == DEAD)
            cols.append(np.where(blocked, self.pad, src))
        toks = np.stack(cols, -1)                                     # [L, n]
        prod = toks[:, None, :] * self.mult[None]                     # [L, layers, n]
        rolling, out = prod[..., 0], []
        for i in range(1, self.n):
            rolling = rolling ^ prod[..., i]
            out.append(rolling[..., None] % self.primes[None, :, i - 1])
        return np.concatenate(out, -1) + self.offsets[None]
