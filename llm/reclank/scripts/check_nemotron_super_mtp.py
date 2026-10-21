#!/usr/bin/env python3
"""Nemotron Super's MTP block (S7 of docs/NEMOTRON_SUPER_PLAN.md): the reference's (model.py
NemotronCPU.mtp) and the CPU engine's (kq.py compile_nemotron_mtp) on a real paragraph. The main
pass of the reference gives the hidden after norm_f of every position (saved to --states; made when
the file is missing: ~25 minutes on 4 cores); then the reference's block teacher-forced over the
positions (its keys accumulating, as a prefill), whose draft of token i + 2 is checked against the
main model's pick at i + 1; then the engine's block on all the positions in one program: its out and
its drafts against the reference's, row by row.

    taskset -c 22,23,46,47 python scripts/check_nemotron_super_mtp.py \\
        /home/cpage/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --states mtp_states.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.nemotron.engine import NemotronTok  # noqa: E402
from reclank.models.nemotron.kq import NemotronKQ  # noqa: E402
from reclank.models.nemotron.model import NemotronCPU  # noqa: E402

TEXT = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg "
        "combined movable metal type, oil-based ink and a wooden press adapted from those used to make "
        "wine. Within a few decades, printing shops had spread to most major cities of Europe, and the "
        "cost of books fell sharply, which allowed ideas to travel faster than ever before.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--states", default="mtp_states.npz", help="the main pass's ids, logits and hidden rows")
    ap.add_argument("--tokens", type=int, default=40)
    a = ap.parse_args()
    if not os.path.exists(a.states):
        tok = NemotronTok(os.path.join(a.ckpt, "tokenizer.json"), os.path.join(a.ckpt, "chat_template.jinja"))
        ids = tok.encode(TEXT)[:a.tokens]
        t0 = time.time()
        m = NemotronCPU(a.ckpt, keep_dense=False)
        lg = m.forward(ids, m.cache())
        np.savez(a.states, ids=np.array(ids), logits=lg, hidden=m.last_hidden)
        print("the main pass: %d tokens in %.0f s" % (len(ids), time.time() - t0), flush=True)
    z = np.load(a.states)
    ids, lg, H = [int(v) for v in z["ids"]], z["logits"], z["hidden"]
    pick = lg.argmax(-1)
    n = len(ids) - 2
    t0 = time.time()
    ref = NemotronCPU(a.ckpt, keep_dense=True)
    rc, rl, ro = ref.cache(), [], []
    for i in range(n):
        d, out = ref.mtp(ids[i + 1], H[i], rc)
        rl.append(d)
        ro.append(out)
    rl, ro = np.array(rl), np.array(ro)
    agree = int((rl.argmax(-1) == pick[1:n + 1]).sum())
    print("reference: %d drafts in %.0f s; the main model's pick %d of %d (%.0f%%), the true token %d" % (
        n, time.time() - t0, agree, n, 100.0 * agree / n, int((rl.argmax(-1) == np.array(ids[2:])).sum())), flush=True)
    kq = NemotronKQ(a.ckpt, weightkeeper=False)
    t1 = time.time()
    el, eo = kq.mtp(ids[1:n + 1], H[:n], kq.mtp_cache(256))
    rel = np.abs(eo - ro).max() / np.abs(ro).max()
    cos = np.sum(eo * ro, -1) / (np.linalg.norm(eo, axis=-1) * np.linalg.norm(ro, axis=-1))
    same = int((el.argmax(-1) == rl.argmax(-1)).sum())
    print("engine: %d rows in %.0f s; out max relative error %.2e, cosine min %.6f; drafts the same %d of %d" % (
        n, time.time() - t1, rel, cos.min(), same, n))
    ok = cos.min() > 0.999 and same >= n - 1 and agree >= 0.7 * n
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
