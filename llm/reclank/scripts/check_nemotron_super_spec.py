#!/usr/bin/env python3
"""Nemotron Super's MTP rounds on the GPU (S7 of docs/NEMOTRON_SUPER_PLAN.md: spec.NemotronMTPSpec through
engine.GpuRunner) against plain steps: greedy, n tokens after a real prompt; the drafts taken a round,
the tokens a second of each. Each MTP token is checked against the logits of plain steps on the same
prefix (the MTP text teacher-forced): the step's argmax, or a near tie (within two bfloat16 steps of
the largest logit: the head's output is rounded to bfloat16, and a verify row's sums are not a
step's: its Mamba scan and the products of the shared expert on int16 x). The first difference from
the plain text is reported, not judged.

    RECLANK_GPU_CPU_THREADS=44 OMP_NUM_THREADS=1 taskset -c 0-46 python scripts/check_nemotron_super_spec.py \\
        /home/cpage/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --tokens 96
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from reclank.models.nemotron.engine import GpuRunner, NemotronTok  # noqa: E402
from reclank.models.nemotron.gpu import NemotronGPU  # noqa: E402
from reclank.models.nemotron.kq import LM, NemotronKQ  # noqa: E402
from reclank.models.nemotron.spec import NemotronMTPSpec  # noqa: E402

PROMPTS = [
    [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, "
                                 "with a docstring."}],
    [{"role": "user", "content": "In three sentences, why did the printing press change Europe?"}],
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--drafts", type=int, default=3)
    ap.add_argument("--hot-gb", type=float, default=None)
    ap.add_argument("--tie", type=float, default=None,
                    help="the largest logit gap of a token not the argmax (default: two bfloat16 steps of the top logit)")
    a = ap.parse_args()
    tok = NemotronTok(os.path.join(a.ckpt, "tokenizer.json"), os.path.join(a.ckpt, "chat_template.jinja"))
    m = NemotronKQ(a.ckpt)
    g = NemotronGPU(m, ctx=8192, hot_gb=a.hot_gb)
    print("GPU: %d%% of the experts hot (%.1f GB)" % (round(100 * g.hot_share), g.hot_bytes / 1e9), flush=True)
    spec = NemotronMTPSpec(g, a.drafts)
    ok = True
    for msgs in PROMPTS:
        ids = tok.encode(tok.apply_chat_template(msgs, thinking=False))
        plain = GpuRunner(g)
        lg = plain.forward(ids)
        y, out = int(np.argmax(lg)), []
        t0 = time.time()
        while len(out) < a.tokens:
            out.append(y)
            y = int(np.argmax(plain.forward([y])))
        tp = time.time() - t0
        run = GpuRunner(g, spec)
        spec.stats = []
        lg = run.forward(ids)
        y, sp = int(np.argmax(lg)), []
        t1 = time.time()
        sp.append(y)
        while len(sp) < a.tokens:
            got = run.round(y, None)
            sp += got
            y = got[-1]
        ts = time.time() - t1
        sp = sp[:a.tokens]
        first = next((i for i in range(a.tokens) if sp[i] != out[i]), None)
        taken = np.mean([s[0] for s in spec.stats]) if spec.stats else 0.0
        # the MTP text teacher-forced through plain steps: the logits before each of its tokens
        tf = GpuRunner(g)
        hs = float(m.M(LM + "lm_head").scale or 1.0)
        lg = tf.forward(ids)
        gaps, over = [], []
        for y in sp:
            gap = float(lg.max() - lg[y])
            # (the head's bfloat16 outputs times its FP8 tensor scale hs: their step at the top logit)
            tie = a.tie if a.tie is not None else hs * 2.0 ** (np.floor(np.log2(abs(lg.max()) / hs)) - 6)
            gaps.append(gap)
            over.append(gap > tie)
            lg = tf.forward([y])
        gaps = np.array(gaps)
        ties = int((gaps > 0).sum())
        print("plain %.1f tok/s, MTP %.1f tok/s (%.2fx); %.2f drafts taken a round of %d; the plain text's tokens: %s" % (
            a.tokens / tp, a.tokens / ts, tp / ts, taken, a.drafts,
            "all" if first is None else "the first %d" % first), flush=True)
        print("   against plain steps on its prefix: the argmax %d of %d, the others within %.3f of it; %d not a near tie" % (
            a.tokens - ties, a.tokens, gaps.max(), sum(over)), flush=True)
        print("  ", repr(tok.decode(sp)[:160]))
        ok = ok and not any(over)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
