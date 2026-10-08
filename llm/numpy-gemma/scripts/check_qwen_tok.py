#!/usr/bin/env python3
"""Compare np_gemma.qwen_tok with the tokenizers library.

QWEN_PLAN.md, phase 1. The script needs the tokenizers library (the venv of
gemma4-12b-qat-pytorch has it). It encodes the files of this repository,
text in other scripts, and random strings, and compares the ids. It also
checks that decode gives the text back.

    [TOKENIZER=.../tokenizer.json] $VENV/bin/python scripts/check_qwen_tok.py
"""
from __future__ import annotations

import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer  # noqa: E402

from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = os.environ.get("TOKENIZER", "models/Qwen3.6-35B-A3B-OptiQ-4bit/tokenizer.json")

SAMPLES = [
    "Hello, world! It's a test. They'll go; we've done it, I'm sure, you'd see.",
    "  leading spaces\n\n\ttabs\r\nand CRLF   \n",
    "数字 12345 和中文。日本語のテキスト。한국어 텍스트.",
    "Emoji: 😀👍🏽 and symbols: ∑∫√ ≠ ≤ ≥ → ← ©®™",
    "Ünïcödé àccents, combining é and café, naïve, Ångström",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHi!",
    "def f(x):\n    return x**2  # comment\n\n\nclass A: pass",
    "Arabic: مرحبا بالعالم. Hindi: नमस्ते दुनिया. Thai: สวัสดีชาวโลก",
    "'S 'T 'RE 'VE 'M 'LL 'D and 's't're",
    " non breaking em space　ideographic",
]


def random_text(rng, n):
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789", " \t\n\r", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        "ÀÉÎÕÜàéîõüñçßøåæ", "中文字符测试日本語", "😀🎉👍🏽",
        "́̈⃝", "  　​", "αβγδεζηθ", "абвгдеж",
    ]
    return "".join(rng.choice(rng.choice(pools)) for _ in range(n))


def main():
    ref = Tokenizer.from_file(PATH)
    ours = QwenTokenizer(PATH)
    texts = list(SAMPLES)
    for p in sorted(glob.glob("*.md")) + sorted(glob.glob("np_gemma/*.py"))[:8]:
        texts.append(open(p, encoding="utf-8").read())
    rng = random.Random(0)
    texts += [random_text(rng, rng.randint(1, 80)) for _ in range(5000)]
    bad = 0
    for t in texts:
        a = ref.encode(t, add_special_tokens=False).ids
        b = ours.encode(t)
        if a != b:
            bad += 1
            if bad <= 5:
                k = next(i for i in range(min(len(a), len(b))) if a[i] != b[i]) \
                    if any(x != y for x, y in zip(a, b)) else min(len(a), len(b))
                print("DIFF at %d: %r\n  ref  %s\n  ours %s" % (
                    k, t[:80], [ref.id_to_token(i) for i in a[k:k + 6]],
                    [ref.id_to_token(i) for i in b[k:k + 6]]))
        elif ours.decode(b) != ref.decode(a, skip_special_tokens=False):
            bad += 1
            if bad <= 5:
                print("DECODE DIFF: %r" % t[:80])
    n_tok = sum(len(ours.encode(t)) for t in texts[:len(SAMPLES) + 5])
    print("%d texts, %d differ (%d tokens in the first files)" % (len(texts), bad, n_tok))
    print("PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
