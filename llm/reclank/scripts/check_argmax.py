#!/usr/bin/env python3
"""Check GP_ARGMAX on the GPU against NumPy: the index of the largest value
(the lower index at a tie, as np.argmax), and with the optional operand the
softmax probability of that value (the confidence of an MTP draft,
RECLANK_MTP_STOP). Then GP_DRAFT_HEAD (the centroid head of the Gemma
drafter) on a small random head: the best token of the tokens of the top_k
centroids, and the probability of it among them.

    python scripts/check_argmax.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.ir import program as P  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402


def draft_head(rng, mirror):
    n_cent, per, top_k, hidden = 64, 32, 4, 96
    vocab = n_cent * per
    order = rng.permutation(vocab).astype(np.int32).reshape(n_cent, per)
    head = rng.standard_normal((vocab, hidden)).astype(np.float32)
    bad = 0
    for trial in range(3):
        u = (rng.standard_normal(hidden) * (1 + trial)).astype(np.float32)
        clog = (rng.standard_normal(n_cent) * 3).astype(np.float32)
        for prob in (False, True):
            tok = np.zeros(1, np.int32)
            p = np.zeros(1, np.float32)
            prog = P.Program()
            prog.emit(P.DRAFT_HEAD, u, clog, order, head, np.zeros(top_k * per, np.float32),
                      np.zeros(top_k, np.int32), tok, n_cent, per, top_k, hidden, *((p,) if prob else ()))
            prog.names["tok"] = tok
            if prob:
                prog.names["p"] = p
            g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
            g.run()
            g.download("tok")
            if prob:
                g.download("p")
            g.close()
            top = np.argsort(-clog.astype(np.float64), kind="stable")[:top_k]
            cand = order[top].reshape(-1)
            sel = head[cand].astype(np.float64) @ u.astype(np.float64)
            want = int(cand[int(np.argmax(sel))])
            pw = 1.0 / np.exp(sel - sel.max()).sum()
            ok = int(tok[0]) == want
            line = "draft head %d %s: token %d (want %d)" % (trial, "+prob" if prob else "     ", tok[0], want)
            if prob:
                err = abs(float(p[0]) - pw) / pw
                ok = ok and err < 1e-4
                line += ", probability %.6f (want %.6f, rel err %.1e)" % (p[0], pw, err)
            bad += not ok
            print(line, "ok" if ok else "BAD")
    return bad


def main():
    rng = np.random.default_rng(3)
    mirror = Mirror()
    bad = 0
    n = 248320                                  # the vocabulary of Qwen3.8
    rows = []
    x = (rng.standard_normal(n) * 3).astype(np.float32)
    rows.append(("flat", x))
    x = x.copy()
    x[1234] += 30.0                             # a sure token
    rows.append(("peaked", x))
    x = (rng.standard_normal(n) * 3).astype(np.float32)
    x[[77, 5000]] = x.max() + 1.0               # a tie: the lower index
    rows.append(("tie", x))
    rows.append(("short", (rng.standard_normal(1000) * 2).astype(np.float32)))
    for name, x in rows:
        for prob in (False, True):
            tok = np.zeros(1, np.int32)
            p = np.zeros(1, np.float32)
            prog = P.Program()
            prog.emit(P.ARGMAX, x, x.size, tok, *((p,) if prob else ()))
            prog.names["tok"] = tok
            if prob:
                prog.names["p"] = p
            g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
            g.run()
            g.download("tok")
            if prob:
                g.download("p")
            g.close()
            xd = x.astype(np.float64)
            want = int(np.argmax(xd))
            pw = 1.0 / np.exp(xd - xd[want]).sum()
            ok = int(tok[0]) == want
            line = "%-6s %s: token %d (want %d)" % (name, "+prob" if prob else "     ", tok[0], want)
            if prob:
                err = abs(float(p[0]) - pw) / pw
                ok = ok and err < 1e-4
                line += ", probability %.6f (want %.6f, rel err %.1e)" % (p[0], pw, err)
            bad += not ok
            print(line, "ok" if ok else "BAD")
    bad += draft_head(rng, mirror)
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
