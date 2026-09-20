"""End-to-end chat with the NumPy runtime: text -> tokens -> forward -> text."""
from __future__ import annotations

import argparse
from pathlib import Path

from np_gemma import Config, Model, SafeTensors, Tokenizer


def resolve_paths(args):
    if args.snapshot:
        snap = Path(args.snapshot)
        return (args.config or str(snap / "config.json"),
                args.weights or str(snap / "model.safetensors"),
                args.tokenizer or str(snap / "tokenizer.json"))
    if not (args.config and args.weights and args.tokenizer):
        raise SystemExit("provide --snapshot, or all of --config/--weights/--tokenizer")
    return args.config, args.weights, args.tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--cache-weights", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only tokenize; do not load weights")
    args = ap.parse_args()
    config_path, weights_path, tok_path = resolve_paths(args)

    tokenizer = Tokenizer(tok_path)
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, thinking=args.thinking)
    ids = tokenizer.encode(text)
    print("prompt text:", repr(text))
    print("prompt ids :", ids)
    print("prompt rt  :", repr(tokenizer.decode(ids)))
    if args.dry_run:
        return 0

    cfg = Config.load(config_path)
    with SafeTensors(weights_path) as st:
        model = Model(st, cfg)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                             eos_ids=[tokenizer.eos_id, 106],
                             cache_weights=args.cache_weights)
    new = out[len(ids):]
    print("new ids    :", new)
    print("generated  :", repr(tokenizer.decode(new)))
    print("full text  :", repr(tokenizer.decode(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
