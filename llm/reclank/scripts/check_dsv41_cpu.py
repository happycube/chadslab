#!/usr/bin/env python3
"""DeepSeek V4.1 Flash on the CPU (reclank/models/deepseek41/model.py) against the
reference (scripts/dsv41_reference.py, DEEPSEEK_V41_PLAN.md Phase 2 step 2).

For each prompt file of the reference: the prompt at position 0, each layer's
output against the reference's (dB of the difference), the routes (the
experts each token picked, the share that agree), the logits (KL, the top
token); then the steps on the reference's tokens (teacher forced): the KL
and the top token of each.

    python scripts/check_dsv41_cpu.py /home/cpage/models/DeepSeek-V4.1-Flash \\
        /home/cpage/npg-run/dsv41-ref/v1/prompt0.npz [--exact] [--steps 8]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.deepseek41.model import DSV41CPU  # noqa: E402


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(ref, mine):
    p = logsm(ref)
    return float((np.exp(p) * (p - logsm(mine))).sum(-1).mean())


def db(a, ref):
    a, ref = a.astype(np.float64), ref.astype(np.float64)
    return 10 * np.log10(((a - ref) ** 2).sum() / max((ref ** 2).sum(), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("ref", nargs="+")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--no-zygote", action="store_true")
    ap.add_argument("--force-routes", action="store_true",
                    help="take the reference's experts (the near-ties of the router aside)")
    a = ap.parse_args()
    t0 = time.time()
    m = DSV41CPU(a.ckpt, exact=a.exact, zygote=not a.no_zygote)
    print("loaded (%s, experts from %s) in %.1f s" % ("exact" if a.exact else "near",
                                                     "the zygote" if m.w.z is not None else "the shards",
                                                     time.time() - t0))
    for path in a.ref:
        r = np.load(path)
        ids = r["ids"]
        m.reset()
        t1 = time.time()
        m.force_routes = r["routes"] if a.force_routes else None
        logits, layers = m.forward(ids, 0, keep=True)
        first = int(r["layers_first"]) if "layers_first" in r else 0
        ref_layers = r["layers"]
        mine = layers[first:]
        per = [db(mine[:, i], ref_layers[:, i]) for i in range(mine.shape[1])]
        rr = r["routes"].astype(np.int64)
        same = (np.sort(m.routes, -1) == np.sort(rr, -1)).all(-1)            # [T, layers]
        agree = (np.sort(m.routes, -1) == np.sort(rr, -1)).mean()
        print("  tokens with the same 6 experts, by layer: %s" % " ".join("%d" % v for v in same.sum(0)))
        rl = r["logits"]
        print("%s: %d tokens, prefill %.1f s" % (os.path.basename(path), len(ids), time.time() - t1))
        print("  layers (dB vs the reference): %s" % " ".join("%.0f" % v for v in per))
        print("  routes agree %.3f; logits KL %.2e, top token same %.3f (last: mine %d ref %d)" % (
            agree, kl(rl, logits), (rl.argmax(-1) == logits.argmax(-1)).mean(), int(logits[-1].argmax()),
            int(rl[-1].argmax())))
        gen = r["gen"]
        sl = r["step_logits"]
        pos = len(ids)
        kls, tops = [], []
        for s in range(min(a.steps, len(sl))):
            m.force_routes = r["step_routes"][s][None] if a.force_routes else None
            lg, _ = m.forward([int(gen[s])], pos)
            pos += 1
            kls.append(kl(sl[s:s + 1], lg[-1:]))
            tops.append(int(lg[-1].argmax()) == int(sl[s].argmax()))
        if kls:
            print("  steps: KL %s; top token same %d of %d" % (" ".join("%.1e" % v for v in kls), sum(tops), len(tops)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
