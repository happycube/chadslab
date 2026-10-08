#!/usr/bin/env python3
"""Check GP_QSA_SELECT of a large group with the scores of the blocks on the
tensor cores (k_qsa_qprep, k_qsa_score_tc, csrc/gpu.cu) against the loop of
k_qsa_query (NP_GEMMA_GPU_QSA_SCORE_TC=0): the same record on the same
random keys (Qwen3.8: the indexer has 4 heads of 128), each mode in its own
process (the mode is read once). For each query, the selected positions of
both; and the time of the record.

    python scripts/check_qsa_score.py [--pos 3000,60000,200000] [--rows 4096,600,4,1]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

HEADS, D, RATIO, BUDGET, ROT = 4, 128, 4, 512, 64
MAXSEL = BUDGET * RATIO + RATIO - 1


def one(pos, t, out, reps):
    from np_gemma import program as P
    from np_gemma.gpu import GPUProgram, Mirror, lib
    rng = np.random.default_rng(pos * 7 + t)
    n = pos + t
    iq = rng.standard_normal((t, HEADS * D)).astype(np.float32)
    ik = rng.standard_normal((t, D)).astype(np.float32)
    idxk = np.zeros((n + 64, D), np.float16)
    idxk[:pos] = rng.standard_normal((pos, D)).astype(np.float16)
    blk = np.zeros((n // RATIO + 64, D), np.float16)
    # the keys of the blocks before the group (after their norm: about unit RMS)
    blk[:pos // RATIO] = rng.standard_normal((pos // RATIO, D)).astype(np.float16)
    qn = (1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    kn = (1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32)
    ang = (np.arange(pos, n)[:, None] / 10000.0 ** (np.arange(ROT // 2) * 2 / ROT)).astype(np.float32)
    cos = np.concatenate([np.cos(ang)] * 2, 1).astype(np.float32)
    sin = np.concatenate([np.sin(ang)] * 2, 1).astype(np.float32)
    sel = np.zeros((t, MAXSEL), np.int32)
    cnt = np.zeros(t, np.int32)
    nbmax = n // RATIO + 1
    rows = min(t, lib().gg_qsa_rows())
    scratch = np.zeros(lib().gg_qsa_extra() + rows * nbmax * 8 + 64, np.uint8)
    prog = P.Program()
    prog.emit(P.QSA_SELECT, iq, ik, idxk.view(np.uint16), blk.view(np.uint16), qn, kn, cos, sin, pos, t,
              HEADS, D, RATIO, BUDGET, ROT, 1e7, 1e-6, sel, cnt, MAXSEL, scratch, nbmax, 0, 11 | 10 << 8)
    prog.names.update(sel=sel, cnt=cnt)
    prog = prog.finish()
    g = GPUProgram(prog, graph=False, mirror=Mirror())
    g.run()
    ms = float(np.median([g.profile()[0] for _ in range(reps)])) if reps else 0.0
    g.download("sel")
    g.download("cnt")
    np.savez(out, sel=sel, cnt=cnt, ms=ms)
    g.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pos", default="3000,60000,200000")
    ap.add_argument("--rows", default="4096,600,4,1")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        pos, t, out = args.one.split(",")
        one(int(pos), int(t), out, args.reps)
        return 0
    bad = 0
    tmp = tempfile.mkdtemp()
    for pos in (int(x) for x in args.pos.split(",")):
        for t in (int(x) for x in args.rows.split(",")):
            res = {}
            for mode in ("0", "1"):
                out = os.path.join(tmp, "m%s.npz" % mode)
                env = dict(os.environ, NP_GEMMA_GPU_QSA_SCORE_TC=mode)
                subprocess.run([sys.executable, __file__, "--one", "%d,%d,%s" % (pos, t, out),
                                "--reps", str(args.reps)], env=env, check=True)
                res[mode] = np.load(out)
            a, b = res["0"], res["1"]
            diff, extra = 0, 0
            for j in range(t):
                if a["cnt"][j] != b["cnt"][j]:
                    diff += 1
                    continue
                c = a["cnt"][j]
                if c < 0:
                    continue
                sa, sb = set(a["sel"][j, :c].tolist()), set(b["sel"][j, :c].tolist())
                if sa != sb:
                    diff += 1
                    extra += len(sa ^ sb) // 2
            # a tie of float sums in another order can swap a block at the
            # edge of the budget: a few positions of a few queries
            ok = diff <= max(2, t // 200) and extra <= 4 * RATIO * max(1, diff)
            bad += not ok
            print("pos %6d rows %4d: %d queries differ (%d positions swapped); %.2f ms (loop %.2f ms, "
                  "%.1fx) %s" % (pos, t, diff, extra, float(b["ms"]), float(a["ms"]),
                                 float(a["ms"]) / max(float(b["ms"]), 1e-9), "ok" if ok else "BAD"))
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
