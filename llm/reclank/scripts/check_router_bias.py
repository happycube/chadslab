#!/usr/bin/env python3
"""Check ROUTER_TOPK with the bias toward the experts on the GPU
(RECLANK_GPU_RESIDENT_BIAS: operands 6 and 7, the slot table and the bias)
and the swap of near ties to the GPU (RECLANK_GPU_HOT_SWAP: operands 8 and
9, the ratio and the array of the choice before the swaps) on the GPU
(k_router_topk, alone and fused with GP_HOT_SPLIT for one token, and with
GP_HOT_SPLIT_MT for a group) and on the CPU (gp_router_topk_body) against
NumPy: the choice takes the logit + bias of each expert with a slot, the
weights the plain softmax of the chosen ones; from the lightest, a chosen
expert with no slot gives its place to the best unchosen one with a slot
of at least swap times its weight (the first with none ends it), and sel
gets the choice before the swaps (with one token, in the selection part of
the cold list, which HOT_SPLIT then leaves). Without the operands the
choice is the plain top k.

    python scripts/check_router_bias.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.ir import program as P  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402


def reference(logits, k, slots, bias, swap=0.0):
    """The choice, its weights, and the choice before the swaps."""
    idx = np.zeros((len(logits), k), np.int32)
    val = np.zeros((len(logits), k), np.float64)
    sel = np.zeros((len(logits), k), np.int32)
    for j, l in enumerate(logits.astype(np.float64)):
        score = l + (bias * (slots >= 0) if slots is not None else 0)
        # the top k, the lower expert first on a tie
        order = np.lexsort((np.arange(len(l)), -score))[:k]
        p = np.exp(l - l.max())
        sel[j] = order
        pv = p[order]
        if swap > 0:
            order = order.copy()
            taken = set(order.tolist())
            for s2 in range(k - 1, -1, -1):
                if slots[order[s2]] >= 0:
                    continue
                cand = [x for x in range(len(l)) if slots[x] >= 0 and x not in taken]
                if not cand:
                    break
                b = min(cand, key=lambda x: (-l[x], x))
                if p[b] < swap * pv[s2]:
                    break
                order[s2], pv[s2] = b, p[b]
                taken.add(b)
        idx[j] = order
        val[j] = pv / pv.sum()
    return idx, val, sel


def emit(logits, k, slots, bias, split, swap):
    t, E = logits.shape
    val = np.zeros((t, k), np.float32)
    idx = np.zeros((t, k), np.int32)
    cold = np.zeros(2 * k + 1 if t == 1 else t * k, np.int32)
    sel = np.zeros((t, k), np.int32)
    # the choice before the swaps: the selection part of the cold list of a
    # step (as gpu38 does; HOT_SPLIT then does not write it), else sel
    in_cold = bool(swap and split and t == 1)
    extra = ((slots, bias) + ((swap, cold[k + 1:] if in_cold else sel) if swap else ())) \
        if slots is not None else ()
    prog = P.Program()
    prog.emit(P.ROUTER_TOPK, logits, t, E, k, val, idx, *extra)
    if split and t == 1:
        prog.emit(P.HOT_SPLIT, idx, val, slots, cold, np.zeros(k, np.float32), k, 0 if swap else 1)
    elif split:
        prog.emit(P.HOT_SPLIT_MT, idx, val, slots, cold, np.zeros(t * k, np.float32), t * k, t, k)
    prog.names.update(val=val, idx=idx)
    if split:
        prog.names["cold"] = cold
    if swap and not in_cold:
        prog.names["sel"] = sel
    return prog.finish(), val, idx, cold, (cold[k + 1:].reshape(1, k) if in_cold else sel)


def main():
    rng = np.random.default_rng(5)
    mirror = Mirror()
    bad = 0
    E, k = 512, 10
    for t, bias, split, swap in ((1, 0.7, True, 0), (1, 3.0, False, 0), (4, 0.7, False, 0),
                                 (4, None, False, 0), (1, None, False, 0), (1, 0.0, True, 0.8),
                                 (8, 0.0, True, 0.6), (8, 0.0, False, 0.5), (8, 0.4, True, 0.7)):
        logits = (rng.standard_normal((t, E)) * 2).astype(np.float32)
        logits[:, 7] = logits[:, 9]                 # a tie
        slots = None
        if bias is not None:
            slots = np.where(rng.random(E) < 0.4, np.arange(E), -1).astype(np.int32)
        ri, rv, rs = reference(logits, k, slots, bias or 0.0, swap)
        for where in ("gpu", "cpu"):
            if where == "cpu" and split:
                continue
            prog, val, idx, cold, sel = emit(logits, k, slots, bias, split, swap)
            if where == "gpu":
                g = GPUProgram(prog, graph=False, mirror=mirror)
                g.run()
                for n in ("val", "idx", "cold", "sel"):
                    if n in prog.names:
                        g.download(n)
                g.close()
            else:
                prog.run()
            err = float(np.abs(val - rv).max())
            ok = (idx == ri).all() and err < 1e-6
            if swap:
                ok = ok and (sel == rs).all()
            if split and t == 1:
                want = [x for x in ri[0] if slots[x] < 0]
                ok = ok and cold[k] == len(want) and list(cold[:len(want)]) == want
            elif split:
                ok = ok and (cold == np.where(slots[ri] < 0, ri, -1).reshape(-1)).all()
            bad += not ok
            print("%s t %d bias %-4s swap %-4s%s: idx %s, %d swapped, val err %.1e %s" % (
                where, t, bias, swap, " +split" if split else "", "same" if (idx == ri).all() else "DIFF",
                int((ri != rs).sum()), err, "ok" if ok else "BAD"))
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
