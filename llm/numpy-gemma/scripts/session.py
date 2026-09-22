"""Load the model one time. Keep the weights in memory. Then answer prompts.

The script does four steps:
1. Read the config, the weights, and the tokenizer.
2. Load all layers and the embedding table with load_all().
3. For each prompt, build the chat text and generate tokens.
4. Print the reply.

Use dtype f32 for speed. Use dtype bf16 when memory is small.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from np_gemma import Config, Model, SafeTensors, Session, Tokenizer


def rss_gb():
    """Return the resident memory size of this process in GB."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def resolve_paths(args):
    """Return the config path, the weights path, and the tokenizer path."""
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
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--dtype", choices=("int4", "int8", "bf16", "f32"), default="f32",
                    help="weight format for the resident model. f32 is fast and uses about 70 GB. bf16 uses about 24 GB and is slower.")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--prompts", nargs="*", default=None,
                    help="non-interactive: answer these prompts, then exit")
    ap.add_argument("--no-history", action="store_true",
                    help="treat each prompt as an independent conversation")
    ap.add_argument("--max-len", type=int, default=8192,
                    help="the size of the key and value cache for a chat")
    args = ap.parse_args()
    config_path, weights_path, tok_path = resolve_paths(args)

    tokenizer = Tokenizer(tok_path)
    cfg = Config.load(config_path)
    st = SafeTensors(weights_path)
    model = Model(st, cfg)
    t0 = time.perf_counter()
    model.load_all(dtype=args.dtype)
    print("[load] %d layers + embedding table resident (%s) in %.1fs | RSS %.1f GB" %
          (cfg.num_hidden_layers, args.dtype, time.perf_counter() - t0, rss_gb()), flush=True)

    history = []
    # The session keeps the keys and the values between the turns. Thus a new
    # turn reads only the new tokens.
    session = Session(model, max_len=args.max_len)

    def answer(prompt):
        """Generate one reply. Add the reply to the history."""
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.extend(history)
        msgs.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, thinking=args.thinking)
        ids = tokenizer.encode(text)
        if args.no_history:
            session.reset()
        t = time.perf_counter()
        out = session.generate(ids, max_new_tokens=args.max_new_tokens,
                               eos_ids=[tokenizer.eos_id, 106])
        dt = time.perf_counter() - t
        new = out[len(ids):]
        reply = tokenizer.decode(new, skip_special_tokens=True)
        print("[gen ] %d new tokens in %.1fs (%.2f tok/s) | %d of %d prompt tokens new | RSS %.1f GB" %
              (len(new), dt, len(new) / max(dt, 1e-9), session.prefilled, len(ids), rss_gb()), flush=True)
        print("       %s" % (reply,), flush=True)
        if not args.no_history:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": reply})
        return reply

    if args.prompts is not None:
        for prompt in args.prompts:
            answer(prompt)
    else:
        print("Type a prompt. Commands: :reset, :q", flush=True)
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            cmd = line.strip()
            if cmd in (":q", ":quit"):
                break
            if cmd == ":reset":
                history.clear()
                session.reset()
                print("[reset]", flush=True)
                continue
            if not cmd:
                continue
            answer(line)
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
