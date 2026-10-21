#!/usr/bin/env python3
"""Nemotron Super on the GPU (S5 of docs/NEMOTRON_SUPER_PLAN.md: models/nemotron/gpu.py NemotronGPU, the
hot experts by NM_HOT_NV4, the cold ones on the CPU) against the CPU engine (kq.py), both from the
weightkeeper: the logits of a real prompt's last token, then greedy steps (the GPU fed the CPU's
tokens; their argmax and KL compared), and the time of a step. Two processes: the CPU engine with
all the threads (--cpu: its tokens and logits to --ref), then the GPU (OMP_NUM_THREADS=1).

    taskset -c 0-46 python scripts/check_nemotron_super_gpu.py CKPT --cpu --ref ref.npz
    OMP_NUM_THREADS=1 taskset -c 0-46 python scripts/check_nemotron_super_gpu.py CKPT --ref ref.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

if "--cpu" not in sys.argv:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from reclank.models.nemotron.engine import NemotronTok  # noqa: E402
from reclank.models.nemotron.gpu import NemotronGPU, task_threads  # noqa: E402
from reclank.models.nemotron.kq import NemotronKQ  # noqa: E402

TEXT = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg "
        "combined movable metal type, oil-based ink and a wooden press adapted from those used to make "
        "wine.")


def kl(a, b):
    pa = np.exp(a - a.max())
    pa /= pa.sum()
    pb = np.exp(b - b.max())
    pb /= pb.sum()
    return float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--hot-gb", type=float, default=None)
    ap.add_argument("--cpu", action="store_true", help="the CPU engine's run, to --ref")
    ap.add_argument("--ref", default="super_gpu_ref.npz")
    a = ap.parse_args()
    os.environ.setdefault("RECLANK_GPU_CPU_THREADS", str(task_threads()))
    tok = NemotronTok(os.path.join(a.ckpt, "tokenizer.json"), os.path.join(a.ckpt, "chat_template.jinja"))
    ids = tok.encode(TEXT)
    m = NemotronKQ(a.ckpt)
    assert m.w is not None, "the weightkeeper does not hold this checkpoint"
    if a.cpu:
        t0 = time.time()
        cc = m.cache(1024)
        rows = [m.forward(ids, cc, logits="last")[0]]
        toks = [int(rows[0].argmax())]
        for _ in range(a.steps):
            rows.append(m.forward([toks[-1]], cc, logits="last")[0])
            toks.append(int(rows[-1].argmax()))
        np.savez(a.ref, ids=np.array(ids), toks=np.array(toks), logits=np.array(rows))
        print("CPU engine: prompt of %d tokens and %d steps in %.1f s: %r" % (len(ids), a.steps, time.time() - t0,
              tok.decode(toks)))
        return
    z = np.load(a.ref)
    assert list(z["ids"]) == ids
    ref_rows, ref_toks = z["logits"], [int(v) for v in z["toks"]]
    lc = ref_rows[0]
    t1 = time.time()
    g = NemotronGPU(m, ctx=4096, hot_gb=a.hot_gb)
    print("GPU: made in %.1f s; %d%% of the experts hot (%.1f GB)" % (time.time() - t1, round(100 * g.hot_share),
          g.hot_bytes / 1e9), flush=True)
    lg = g.prefill(ids)
    print("prompt: KL(CPU || GPU) %.2e, argmax %d / %d" % (kl(lc, lg), int(lc.argmax()), int(lg.argmax())), flush=True)
    yc, yg = int(lc.argmax()), int(lg.argmax())
    outc, outg, kls, ts = [yc], [yg], [], []
    for s_ in range(min(a.steps, len(ref_toks) - 1)):
        lc = ref_rows[s_ + 1]
        t2 = time.perf_counter()
        lg = g.step(yc)                 # (the CPU's token: the same inputs, the logits compared)
        ts.append(time.perf_counter() - t2)
        kls.append(kl(lc, lg))
        yc, yg = int(lc.argmax()), int(lg.argmax())
        outc.append(yc)
        outg.append(yg)
    same = sum(int(x == y) for x, y in zip(outc, outg))
    print("steps: argmax the same %d of %d; KL mean %.2e max %.2e; a GPU step median %.1f ms (%.1f tok/s)" % (
        same, len(outc), float(np.mean(kls)), float(np.max(kls)), 1e3 * float(np.median(ts)),
        1.0 / float(np.median(ts))))
    print("CPU's text:", repr(tok.decode(outc)))
    ok = same >= len(outc) - 1 and max(kls) < 0.02
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
