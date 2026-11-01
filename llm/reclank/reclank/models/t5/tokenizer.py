"""The tokenizer of umT5 (google/umt5-xxl, Wan 2.2's) as transformers' T5Tokenizer gives it (the
tokenizers backend that transformers 5 builds from spiece.model): a Unigram of sentencepiece
(music.unigram's Viterbi over the vocabulary of tokenizer.json):

Ported from transformers (src/transformers/models/t5/tokenization_t5.py; Apache License 2.0;
Copyright 2018 T5 Authors and HuggingFace Inc. team) and diffusers
(src/diffusers/pipelines/wan/pipeline_wan.py, prompt_clean; Apache License 2.0; Copyright 2025
The Wan Team and The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

1. The added tokens (<pad>, </s>, <s>, <unk>, <extra_id_N>) split out of the text first, as
   they are.
2. No normalizer. Each part between them split at white space (WhitespaceSplit), each word
   with U+2581 in front (Metaspace, prepend_scheme "always"), a word cut before each U+2581
   inside it (split).
3. Each word: the pieces of the highest total score; </s> at the end (the post processor). A
   text of more than max_len tokens is cut to max_len - 1 before the </s> (truncation).

tokenizer.json's own pipeline (a run of spaces one space, Metaspace without WhitespaceSplit)
differs only at the white space beside an added token: its "a </s>" has a lone U+2581 piece
before the </s>, transformers' (the pipeline's) none.

The text first goes through diffusers' prompt_clean (WanPipeline): html.unescape twice, the
white space at the ends stripped, each run of white space one space. ftfy.fix_text, which the
pipeline runs before when ftfy is installed, is not here (the generation venv has no ftfy).
"""
from __future__ import annotations

import html
import re

from reclank.models.music.unigram import SPACE, UnigramTokenizer


def prompt_clean(text):
    """diffusers' prompt_clean of the Wan pipelines (without ftfy)."""
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


class T5Tokenizer(UnigramTokenizer):
    def __init__(self, path):
        super().__init__(path)
        self.pad_id = self.special["<pad>"]
        self._split = re.compile("(%s)" % "|".join(re.escape(t) for t in sorted(self.special, key=len, reverse=True)))
        self._pieces = re.compile("%s[^%s]*" % (SPACE, SPACE))

    def encode(self, text, max_len=None):
        """The ids of text (no prompt_clean here), </s> at the end; max_len: at most max_len ids
        (the text cut, the </s> kept)."""
        out = []
        for i, part in enumerate(self._split.split(text)):
            if i % 2:
                out.append(self.special[part])
                continue
            for w in part.split():
                for piece in self._pieces.findall(w if w.startswith(SPACE) else SPACE + w):
                    out += self._word(piece)
        if max_len is not None:
            out = out[:max_len - 1]
        return out + [self.eos_id]
