#!/usr/bin/env python3
"""Greedy multi-token generation with Hugging Face, dumping ids for comparison."""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401  (sets HF_HOME)

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=_bootstrap.MODEL_ID)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.eval()
    results = []
    for prompt in args.prompts:
        messages = [{"role": "system", "content": args.system},
                    {"role": "user", "content": prompt}]
        enc = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = enc.to(model.device)
        ids = inputs["input_ids"][0].tolist()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, eos_token_id=-1)
        gen = out[0][len(ids):].tolist()
        results.append({"prompt": prompt, "prompt_ids": ids, "gen_ids": gen})
        print("[hf] %r -> %s" % (prompt, gen), flush=True)
    payload = json.dumps(results)
    if args.out:
        open(args.out, "w").write(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
