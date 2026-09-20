"""Autoregressive decode with the NumPy-only runtime (ids in, ids out)."""
from __future__ import annotations

import argparse
import json

from np_gemma import Config, Model, SafeTensors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--trace", default=None, help="read input_ids from a captured trace manifest")
    ap.add_argument("--input-ids", default=None, help="comma separated token ids")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--cache-weights", action="store_true",
                    help="keep all layer weights in RAM (~50 GB float32) to avoid re-reading per token")
    ap.add_argument("--eos", default="1,106", help="comma separated stop token ids")
    args = ap.parse_args()
    if args.input_ids:
        ids = [int(x) for x in args.input_ids.split(",")]
    elif args.trace:
        ids = json.load(open(args.trace + "/manifest.json"))["input_ids"]
    else:
        raise SystemExit("need --input-ids or --trace")
    eos = [int(x) for x in args.eos.split(",") if x != ""]
    cfg = Config.load(args.config)
    with SafeTensors(args.weights) as st:
        model = Model(st, cfg)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, eos_ids=eos,
                             cache_weights=args.cache_weights)
    print("prompt   :", ids)
    print("generated:", out[len(ids):])
    print("full     :", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
