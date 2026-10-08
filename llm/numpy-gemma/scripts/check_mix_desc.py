#!/usr/bin/env python3
"""Check the prefetch fields of the desc of the mixed groups (QwenGPU._pre_desc).

The desc of a layer is one for all its mixed programs (one for each size of
group: 256, 512, 1024, 2048 rows). Its desc[21] is the gidx3 of a program
(the prefetched experts of each token, written by the plan on the host, then
copied to the GPU). It was set at the compile of a program only: a program
of another size then ran with the gidx3 of the last compiled one, and its
own gidx3 kept the experts of an older run (cudaErrorIllegalAddress at
89033 tokens of a coding agent's session).

The prompt goes in parts of sizes that change, so that the programs of the
sizes run after the compiles of the others. After each mixed group: the
gidx3 of its program holds only experts prefetched in this run (the slot of
each in the set of the layer is >= 0). Then the same parts with --old (the
desc as before the fix) shows the bug: on the 2-socket Xeon and the 3090,
the second group (512 rows after the compile of the 2048 one) failed with
cudaErrorIllegalAddress, as the server did.

    python scripts/check_mix_desc.py -m MODEL.gguf [--old]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARTS = (1500, 400, 1500, 300, 900, 1800, 260, 700, 450, 2000)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--old", action="store_true", help="also run with the desc as before the fix")
    args = ap.parse_args()
    import numpy as np
    from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU
    from np_gemma.qwen4_gpu import Qwen4GPU
    from np_gemma.qwen_tok import QwenTokenizer
    tok = QwenTokenizer(os.path.join(os.path.dirname(args.model), "tokenizer.json"))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = ""
    for f in sorted(glob.glob(os.path.join(here, "np_gemma", "*.py"))):
        text += "\n\n===== %s =====\n" % os.path.basename(f) + open(f, errors="replace").read()
    ids = tok.encode("<|im_start|>user\n" + text)[:sum(PARTS)]
    assert len(ids) == sum(PARTS), "too little text"
    m = Qwen4CPU(args.model)
    g = Qwen4GPU(m, ctx=args.ctx)
    stats = {"runs": 0, "bad": 0}
    real_mix = g.mix

    def mix(tokens, pos, size, media=None):
        h = real_mix(tokens, pos, size, media)
        g3 = getattr(g, "mix_gidx3", {}).get(size)
        if g3 is not None and g.mix_desc:
            last = max(g.mix_desc)
            slot = g._pre(last)["slot"]
            e = g3.reshape(-1)[:len(tokens) * g.cfg.top_k]
            e = e[e >= 0]
            stale = int(np.sum(slot[e] < 0))
            stats["runs"] += 1
            stats["bad"] += stale > 0
            print("  mixed group of %4d rows (%4d tokens) at %5d: %d prefetched entries, %d not "
                  "prefetched in this run%s" % (size, len(tokens), pos, e.size, stale,
                                                "  STALE" if stale else ""), flush=True)
        return h
    g.mix = mix

    def run(label):
        c = Qwen4Cache(m.cfg, args.ctx)
        g.attach(c)
        pos = 0
        stats.update(runs=0, bad=0)
        print(label, flush=True)
        for n in PARTS:
            g.prefill(ids[pos:pos + n], pos=pos)
            pos += n
        lg = np.asarray(g.logits(), np.float64).reshape(-1)
        print("  %d mixed groups checked, %d with stale prefetch entries" % (stats["runs"], stats["bad"]),
              flush=True)
        return lg, stats["bad"]

    new, bad_new = run("the fix (_pre_desc at each run):")
    ok = bad_new == 0
    if args.old:
        g._pre_desc = lambda size: None
        old, bad_old = run("as before the fix (desc[21] from the last compile):")
        p = np.exp(new - new.max()); p /= p.sum()
        q = np.exp(old - old.max()); q /= q.sum()
        kl = float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
        print("the logits of the last token, fix against before: KL %.4g, top %d / %d" % (
            kl, int(new.argmax()), int(old.argmax())))
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
