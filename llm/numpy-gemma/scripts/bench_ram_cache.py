#!/usr/bin/env python3
"""Compare the memory map cache with the local memory cache.

The memory map cache reads the data file with a memory map. The system uses
4 KiB pages. A long read then needs many TLB entries. The local memory cache
copies the file into anonymous memory with 2 MiB pages. This script loads the
model two times and times a decode pass for each cache.

Set the variable NP_GEMMA_CACHE_RAM=1 to select the local memory cache.
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

from np_gemma import Config, KVCache, Model, SafeTensors, Tokenizer


def resolve_paths(args):
    """Return the config path, the weights path, and the tokenizer path."""
    if args.snapshot:
        s = Path(args.snapshot)
        return (args.config or str(s / "config.json"),
                args.weights or str(s / "model.safetensors"),
                args.tokenizer or str(s / "tokenizer.json"))
    return args.config, args.weights, args.tokenizer


def anon_huge_kb():
    """Return the size of the anonymous large pages in KiB."""
    try:
        with open("/proc/self/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("AnonHugePages:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def bench(ram, config_path, weights_path, tok_path, ids, steps):
    """Load the model and return the list of decode times."""
    os.environ["NP_GEMMA_CACHE_RAM"] = "1" if ram else "0"
    tok = Tokenizer(tok_path)
    cfg = Config.load(config_path)
    st = SafeTensors(weights_path)
    model = Model(st, cfg)
    t0 = time.perf_counter()
    model.load_all(dtype="int8")
    print("  load %.1f s" % (time.perf_counter() - t0))
    cache = KVCache(cfg, max_len=len(ids) + steps + 4)
    model.forward(ids, cache=cache)
    times = []
    nxt = 50429
    for k in range(steps):
        t0 = time.perf_counter()
        x = model.forward([nxt], cache=cache, start_pos=len(ids) + k)
        times.append(time.perf_counter() - t0)
        nxt = int(model.logits(x).argmax())
    st.close()
    model.free_all()
    return times, anon_huge_kb()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()
    config_path, weights_path, tok_path = resolve_paths(args)

    tok = Tokenizer(tok_path)
    text = tok.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Count from 1 to 10, separated by commas."}],
        add_generation_prompt=True, thinking=False)
    ids = tok.encode(text)

    t_map, _ = bench(False, config_path, weights_path, tok_path, ids, args.steps)
    gc.collect()
    t_ram, huge = bench(True, config_path, weights_path, tok_path, ids, args.steps)
    print()
    print("memory map  : min %.3f  median %.3f s" % (min(t_map), statistics.median(t_map)))
    print("local memory: min %.3f  median %.3f s  (AnonHugePages %d MiB)" % (
        min(t_ram), statistics.median(t_ram), huge // 1024))
    print("speedup: %.2fx" % (min(t_map) / min(t_ram)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
