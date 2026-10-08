#!/usr/bin/env python3
"""Replay the prompt of a crash report of scripts/serve_qwen4.py on the GPU.

serve_qwen4 writes a report on a fatal error (--crash-dir, default
$TMPDIR/np-gemma-crash/crash-<time>/): report.json (the error, the record of
the GPU program queued last, the request, the cache, the memory of the GPU)
and prompt.npy (the tokens of the request). This reads the prompt as the
server did: the cached part (cache_hit of the request) first, then the new
part, in parts of 4096 tokens with the rows of the MTP layer. It is not the
exact history of the server (the hot experts, the programs in memory), so a
crash that needs that state may not come back; a run of the sizes of the
groups of the report is the first try.

    python scripts/replay_crash.py CRASH_DIR -m MODEL.gguf [--sync]

--sync: NP_GEMMA_GPU_SYNC_CHECK=1, no graphs and a wait after each record,
so that an error names the record at fault (slow). --ctx: the context of
the report by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("crash_dir")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--sync", action="store_true", help="NP_GEMMA_GPU_SYNC_CHECK=1 (slow)")
    ap.add_argument("--ctx", type=int, default=0)
    args = ap.parse_args()
    if args.sync:
        os.environ["NP_GEMMA_GPU_SYNC_CHECK"] = "1"
    import numpy as np
    rep = json.load(open(os.path.join(args.crash_dir, "report.json")))
    ids = np.load(os.path.join(args.crash_dir, "prompt.npy")).tolist()
    req = rep.get("request") or {}
    print("the crash: %s\n  where: %s\n  request: %s" % (rep.get("error"), rep.get("where"), req),
          flush=True)
    from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU
    from np_gemma.qwen4_gpu import Qwen4GPU
    from np_gemma import gpu as G
    m = Qwen4CPU(args.model)
    ctx = args.ctx or int(rep.get("context") or len(ids) + 1024)
    g = Qwen4GPU(m, ctx=ctx)
    caches = [Qwen4Cache(m.cfg, ctx)]
    g.attach(caches[0])
    k = int(req.get("cache_hit") or req.get("cached_before") or 0)
    k = min(k, len(ids))
    hidden = g.cfg.hidden_size * getattr(g.cfg, "hc_count", 1)
    h0 = np.zeros(hidden, np.float32)

    def read(lo, hi):
        nonlocal h0
        for c0 in range(lo, hi, 4096):
            part = ids[c0:min(c0 + 4096, hi)]
            t0 = time.time()
            H = g.prefill(part, pos=c0, streams=True)
            if getattr(g, "has_mtp", False):
                Hp = np.concatenate([h0.reshape(1, -1), H[:-1]])
                for r0 in range(0, len(part), 256):
                    g.mtp(Hp[r0:r0 + 256], part[r0:r0 + 256], c0 + r0)
            h0 = H[-1].copy()
            print("  read %d-%d in %.1f s" % (c0, c0 + len(part), time.time() - t0), flush=True)

    try:
        print("the cached part: %d tokens" % k, flush=True)
        read(0, k)
        print("the new part: %d tokens" % (len(ids) - k), flush=True)
        read(k, len(ids))
    except RuntimeError as exc:
        print("REPRODUCED: %s\n  where: %s" % (exc, G.where()), flush=True)
        return 1
    print("no crash (the state of the server may matter: see the module text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
