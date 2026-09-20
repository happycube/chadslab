"""Differential test: NumPy tokenizer vs Hugging Face AutoTokenizer."""
from __future__ import annotations

import argparse
import json

from np_gemma import Tokenizer

CORPUS = [
    "Hello world",
    "The capital of France is",
    "  double  space",
    "trailing space ",
    "leading space",
    "\ttab\nnewline\n\nblank",
    "café",
    "naïve résumé",
    "日本語のテスト",
    "中文测试",
    "한국어 테스트",
    "emoji 😀🎉🚀!",
    "it's 100% \"fine\"",
    "def f(x):\n    return x + 1",
    "a" * 50,
    "x1y2z3 4.5e-10",
    "http://example.com/path?q=1",
    "MixedCASE and UPPER",
    "  ",
    "",
    "<bos>hi",
    "user<turn|>model",
    "<|turn>user\nHi<turn|>",
    "Ω≈ç√∫˜µ≤≥÷",
    "🤖💡🧠",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(args.snapshot)
    mine = Tokenizer(args.snapshot + "/tokenizer.json")

    enc_bad = 0
    dec_bad = 0
    for s in CORPUS:
        a = hf.encode(s, add_special_tokens=False)
        b = mine.encode(s, add_special_tokens=False)
        if a != b:
            enc_bad += 1
            print("ENCODE MISMATCH", repr(s))
            print("   hf  :", a)
            print("   mine:", b)
        da = hf.decode(a)
        db = mine.decode(b)
        if da != db:
            dec_bad += 1
            print("DECODE MISMATCH", repr(s), "->", repr(da), "vs", repr(db))
    print(f"encode: {len(CORPUS) - enc_bad}/{len(CORPUS)} exact | decode: {len(CORPUS) - dec_bad}/{len(CORPUS)} exact")

    cases = {
        "user_only_off": ([{"role": "user", "content": "Hi"}], False),
        "user_only_on": ([{"role": "user", "content": "Hi"}], True),
        "sys_user_on": ([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], True),
        "sys_user_off": ([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], False),
        "multi_off": ([{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
                       {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}], False),
    }
    chat_bad = 0
    for name, (msgs, think) in cases.items():
        a = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        b = mine.apply_chat_template(msgs, add_generation_prompt=True, thinking=think)
        if a != b:
            chat_bad += 1
            print("CHAT MISMATCH", name)
            print("   hf  :", repr(a))
            print("   mine:", repr(b))
    print(f"chat template: {len(cases) - chat_bad}/{len(cases)} exact")
    return 0 if enc_bad == 0 and dec_bad == 0 and chat_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
