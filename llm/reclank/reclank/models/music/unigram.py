"""The Unigram tokenizer of a tokenizer.json of sentencepiece (XLM-R).

The pipeline of xlm-roberta-base's tokenizer.json:

1. The normalizer: sentencepiece's precompiled map (nmt_nfkc). NFKC here,
   which is the same for the texts of the checks (scripts/check_music.py).
2. Split the text at white space; each word gets U+2581 in front.
3. Each word: the pieces of the highest total score (Viterbi over the
   vocabulary's log probabilities). A character that no piece holds is
   <unk>, and the <unk> of a run of such characters is one.
4. <s> in front and </s> after (the post processor).
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

SPACE = "▁"


class UnigramTokenizer:
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        m = data["model"]
        if m.get("type", "Unigram") != "Unigram":
            raise ValueError("expected a Unigram model")
        self.unk_id = int(m["unk_id"])
        self.score = {}
        for i, (piece, s) in enumerate(m["vocab"]):
            self.score.setdefault(piece, (float(s), i))
        self.maxlen = max(len(p) for p in self.score)
        # an unknown character costs 10 under the lowest score (tokenizers' Unigram)
        self.unk_score = min(s for s, _ in self.score.values()) - 10.0
        self.special = {a["content"]: a["id"] for a in data.get("added_tokens", [])}
        self.bos_id, self.eos_id = self.special["<s>"], self.special["</s>"]
        self.id_to_piece = {i: p for p, (_, i) in self.score.items()}
        self._cache = {}

    def _word(self, w):
        """The ids of one word (with its U+2581): Viterbi."""
        ids = self._cache.get(w)
        if ids is not None:
            return ids
        n = len(w)
        best = [-1e300] * (n + 1)
        back = [None] * (n + 1)          # (start, id or None for <unk>)
        best[0] = 0.0
        for i in range(n):
            if best[i] <= -1e299:
                continue
            hit = False
            for j in range(i + 1, min(n, i + self.maxlen) + 1):
                e = self.score.get(w[i:j])
                if e is None:
                    continue
                if j == i + 1:
                    hit = True
                s = best[i] + e[0]
                if s > best[j]:
                    best[j], back[j] = s, (i, e[1])
            if not hit:                  # the character alone as <unk>
                s = best[i] + self.unk_score
                if s > best[i + 1]:
                    best[i + 1], back[i + 1] = s, (i, None)
        out = []
        j = n
        while j > 0:
            i, t = back[j]
            if t is None:
                if not (out and out[-1] is None):
                    out.append(None)
            else:
                out.append(t)
            j = i
        ids = [self.unk_id if t is None else t for t in reversed(out)]
        if len(self._cache) < 200000:
            self._cache[w] = ids
        return ids

    def encode(self, text):
        """The ids of text, <s> ... </s>."""
        text = unicodedata.normalize("NFKC", text)
        out = [self.bos_id]
        for w in text.split():
            out += self._word(SPACE + w)
        out.append(self.eos_id)
        return out
