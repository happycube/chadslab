#!/usr/bin/env python3
"""Generate tokens with greedy selection. Write the token ids to stdout and to a file.

Use this script to compare the NumPy runtime with Hugging Face. The script
writes one JSON record for each prompt. A record has:
    prompt       The prompt text.
    prompt_ids   The token ids of the chat text.
    gen_ids      The generated token ids.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from np_gemma import Config, Model, SafeTensors, Tokenizer


def resolve_paths(args):
    """Return the config path, the weights path, and the tokenizer path."""
    if args.snapshot:
        s = Path(args.snapshot)
        return (args.config or str(s / "config.json"),
                args.weights or str(s / "model.safetensors"),
                args.tokenizer or str(s / "tokenizer.json"))
    if not (args.config and args.weights and args.tokenizer):
        raise SystemExit("provide --snapshot, or all of --config/--weights/--tokenizer")
    return args.config, args.weights, args.tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--dtype", choices=("int4", "int8", "bf16", "f32"), default="f32",
                    help="f32 is fast and uses about 70 GB. bf16 uses about 24 GB and is slower.")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    config_path, weights_path, tok_path = resolve_paths(args)

    tokenizer = Tokenizer(tok_path)
    cfg = Config.load(config_path)
    with SafeTensors(weights_path) as st:
        model = Model(st, cfg)
        model.load_all(dtype=args.dtype)
        results = []
        for prompt in args.prompts:
            messages = [{"role": "system", "content": args.system},
                        {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, thinking=False)
            ids = tokenizer.encode(text)
            t0 = time.perf_counter()
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, eos_ids=())
            dt = time.perf_counter() - t0
            gen = out[len(ids):]
            results.append({"prompt": prompt, "prompt_ids": ids, "gen_ids": gen})
            print("[np] %r -> %s (%d tokens in %.1fs)" % (prompt, gen, len(gen), dt), flush=True)
    payload = json.dumps(results)
    if args.out:
        open(args.out, "w").write(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
