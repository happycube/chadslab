"""Self-consistency check: incremental decode with a KV cache must equal batch prefill."""
from __future__ import annotations

import argparse
import json

import numpy as np

from np_gemma import Config, KVCache, Model, SafeTensors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--input-ids", default=None)
    ap.add_argument("--layers", type=int, default=1)
    args = ap.parse_args()
    if args.input_ids:
        ids = [int(x) for x in args.input_ids.split(",")]
    else:
        ids = json.load(open(args.trace + "/manifest.json"))["input_ids"]
    cfg = Config.load(args.config)
    with SafeTensors(args.weights) as st:
        m = Model(st, cfg)
        batch = m.forward(ids, max_layers=args.layers)
        m.keep_weights = True
        cache = KVCache(cfg, max_len=len(ids) + 1)
        outs = []
        for step, tok in enumerate(ids):
            x = m.forward([tok], max_layers=args.layers, cache=cache, start_pos=step)
            outs.append(x[0])
        inc = np.stack(outs, axis=0)
    diff = np.abs(batch - inc)
    rn = float(np.linalg.norm(batch.ravel()))
    print("sequence length:", len(ids), "| layers:", args.layers)
    print("batch vs incremental: max_abs %.3e mean_abs %.3e" % (diff.max(), diff.mean()))
    print("relative L2: %.3e" % (float(diff) / (rn + 1e-30) if False else float(np.linalg.norm(diff.ravel()) / (rn + 1e-30))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
