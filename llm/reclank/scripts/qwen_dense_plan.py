#!/usr/bin/env python3
"""The type of each dense matrix of Qwen3.6 for a budget of bytes
(QWEN_DENSE_QUANT.md, steps 4 and 5).

The model: the KL of a file to the reference (every dense matrix BF12) is
c0 + sum over the kinds K of s_K * E_K, where E_K is the sum over the
matrices of K of the error of their output weighted by their inputs
(10^(dB / 10) of qwen_dense_study.py score: "weighted"), and s_K the
sensitivity of the output of the model to that kind (the attention of a
full layer, the DeltaNet projections and output, ...); c0 is the KL of the
rest (the experts and the cache are the same in every file, and the KL of
two files of the same weights is not 0: the MoE routing amplifies any
difference).

fit: c0 and s_K by least squares over files whose plan and run are known
(qwen_dense_eval.py run); the residuals tell how well the model holds:

    python scripts/qwen_dense_plan.py fit --study study.json --ref REF.npz \\
        --run PLAN.json:RUN.npz ... --out sens.json

A plan is a JSON {tensor name: type}; "ALL:TYPE" for every matrix in TYPE.

fit-layers: the sensitivity of each matrix, s = a_l r_K, from the runs of
qwen_dense_layers.py (each layer l with its dense matrices in one type, the
rest exact: KL_l = a_l sum r_K E_m over its matrices) and the relative
sensitivity r_K of each kind (the s_K of a fit, scaled to a mean of 1;
without --kinds, 1):

    python scripts/qwen_dense_plan.py fit-layers --study study.json --layers layers.json \
        [--kinds sens.json] --out sens_layers.json

plan: the type of each matrix that makes the predicted KL least for at
most --budget-gb of dense matrices (the greedy upgrades along the convex
hull of the bytes and the error of each matrix: the Lagrangian of the
budget), among --types:

    python scripts/qwen_dense_plan.py plan --study study.json --sens sens.json \\
        --budget-gb 1.3 --types RQ4_K,RQ5_K,RQ6_K,RQ8_0,BF12 --out plan.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load_plan(spec, mats):
    if spec.startswith("ALL:"):
        return {m["name"]: spec[4:] for m in mats}
    return json.load(open(spec))


def _logsm(x):
    x = x.astype(np.float64)
    mx = x.max(-1, keepdims=True)
    return x - mx - np.log(np.exp(x - mx).sum(-1, keepdims=True))


_REF = {}


def kl_of(ref_path, run_path):
    """The mean KL of the reference run to a run, and its standard error."""
    if ref_path not in _REF:
        ref = np.load(ref_path)
        lr = _logsm(ref["logits"])
        _REF.clear()
        _REF[ref_path] = (lr, np.exp(lr), ref["next"])
    lr, pr, nxt = _REF[ref_path]
    run = np.load(run_path)
    assert (run["next"] == nxt).all(), run_path
    kl = (pr * (lr - _logsm(run["logits"]))).sum(-1)
    return kl.mean(), kl.std(ddof=1) / np.sqrt(len(kl))


def err_lin(m, t):
    db = m["types"][t]["weighted"]
    return 0.0 if db is None else 10 ** (db / 10)


def features(plan, mats, kinds):
    """E_K of each kind under a plan."""
    f = np.zeros(len(kinds))
    for m in mats:
        f[kinds.index(m["kind"])] += err_lin(m, plan[m["name"]])
    return f


def fit(args):
    mats = json.load(open(args.study))["matrices"]
    kinds = sorted({m["kind"] for m in mats})
    rows, ys, ses, names = [], [], [], []
    for spec in args.run:
        p, r = spec.rsplit(":", 1)
        plan = load_plan(p, mats)
        kl, se = kl_of(args.ref, r)
        rows.append(features(plan, mats, kinds))
        ys.append(kl)
        ses.append(max(se, 1e-6))
        names.append(p)
    X = np.hstack([np.ones((len(rows), 1)), np.array(rows)])
    y, w = np.array(ys), 1 / np.array(ses)
    # weighted least squares, s_K >= 0 (a kind cannot help by losing bits):
    # the kinds with a negative estimate go out and the rest are fit again
    keep = list(range(X.shape[1]))
    while True:
        coef, *_ = np.linalg.lstsq(X[:, keep] * w[:, None], y * w, rcond=None)
        neg = [keep[i] for i, c in enumerate(coef) if c < 0 and keep[i] > 0]
        if not neg:
            break
        keep = [k for k in keep if k not in neg]
    full = np.zeros(X.shape[1])
    full[keep] = coef
    pred = X @ full
    out = {"c0": float(full[0]), "s": {k: float(full[1 + i]) for i, k in enumerate(kinds)},
           "runs": [{"plan": n, "kl": float(a), "se": float(s), "pred": float(b)}
                    for n, a, s, b in zip(names, y, ses, pred)]}
    json.dump(out, open(args.out, "w"), indent=1)
    print("c0 %.3e" % out["c0"])
    for k in kinds:
        print("  s %-12s %.3e" % (k, out["s"][k]))
    print("%-40s %10s %10s %8s" % ("run", "KL", "predicted", "resid/se"))
    for r in out["runs"]:
        print("%-40s %.3e  %.3e  %+6.1f" % (r["plan"][-40:], r["kl"], r["pred"], (r["kl"] - r["pred"]) / r["se"]))


def fit_layers(args):
    mats = {m["name"]: m for m in json.load(open(args.study))["matrices"]}
    lay = json.load(open(args.layers))
    t = lay["type"]
    r = {}
    if args.kinds:
        sk = json.load(open(args.kinds))["s"]
        mean = np.mean([v for v in sk.values() if v > 0])
        r = {k: (v / mean if v > 0 else 1.0) for k, v in sk.items()}
    s_matrix, rows = {}, []
    for test in lay["tests"]:
        wsum = sum(r.get(mats[n]["kind"], 1.0) * err_lin(mats[n], t) for n in test["matrices"])
        a = test["kl"] / wsum if wsum > 0 else 0.0
        for n in test["matrices"]:
            s_matrix[n] = a * r.get(mats[n]["kind"], 1.0)
        rows.append((test["test"], test["kl"], test["se"], a))
    json.dump({"c0": 0.0, "s": {}, "s_matrix": s_matrix, "type": t, "kinds": r},
              open(args.out, "w"), indent=1)
    print("%-12s %10s %9s %10s" % ("test", "KL", "se", "a_l"))
    for name, kl, se, a in rows:
        print("%-12s %.3e  %.1e  %.3e" % (name, kl, se, a))


def plan(args):
    mats = json.load(open(args.study))["matrices"]
    sens = json.load(open(args.sens))
    types = args.types.split(",")
    budget = args.budget_gb * 1e9
    # the options of each matrix: (bytes, cost = s_K * E), the convex hull
    hulls = []
    for m in mats:
        s = sens.get("s_matrix", {}).get(m["name"], sens["s"].get(m["kind"], 0.0))
        opts = sorted({(m["types"][t]["bytes"], s * err_lin(m, t), t) for t in types if t in m["types"]})
        pts = []
        for o in opts:          # more bytes must give less cost
            if not pts or o[1] < pts[-1][1]:
                pts.append(o)
        hull = []               # the lower convex hull: the efficient upgrades
        for o in pts:
            while len(hull) >= 2:
                (b0, c0, _), (b1, c1, _) = hull[-2], hull[-1]
                if (c1 - c0) * (o[0] - b0) >= (o[1] - c0) * (b1 - b0):
                    hull.pop()  # hull[-1] is on or above the line from hull[-2] to o
                else:
                    break
            hull.append(o)
        hulls.append(hull)
    level = [0] * len(mats)
    used = sum(h[0][0] for h in hulls)
    if used > budget:
        sys.exit("the cheapest types take %.3f GB, more than the budget" % (used / 1e9))
    while True:
        best, bi = None, None
        for i, h in enumerate(hulls):
            if level[i] + 1 < len(h):
                db = h[level[i] + 1][0] - h[level[i]][0]
                dc = h[level[i]][1] - h[level[i] + 1][1]
                if used + db <= budget and (best is None or dc / db > best):
                    best, bi = dc / db, i
        if bi is None:
            break
        used += hulls[bi][level[bi] + 1][0] - hulls[bi][level[bi]][0]
        level[bi] += 1
    out = {m["name"]: hulls[i][level[i]][2] for i, m in enumerate(mats)}
    pred = sens["c0"] + sum(hulls[i][level[i]][1] for i in range(len(mats)))
    json.dump(out, open(args.out, "w"), indent=0)
    counts = {}
    for t in out.values():
        counts[t] = counts.get(t, 0) + 1
    print("%.3f GB of %.3f; predicted KL %.3e; %s" % (used / 1e9, args.budget_gb, pred,
                                                    ", ".join("%d %s" % (n, t) for t, n in sorted(counts.items()))))
    by = {}
    for m in mats:
        by.setdefault(m["kind"], {}).setdefault(out[m["name"]], 0)
        by[m["kind"]][out[m["name"]]] += 1
    for k, c in sorted(by.items()):
        print("  %-12s %s" % (k, ", ".join("%d %s" % (n, t) for t, n in sorted(c.items()))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--study", required=True)
    f.add_argument("--ref", required=True)
    f.add_argument("--run", action="append", required=True, help="PLAN.json:RUN.npz (or ALL:TYPE:RUN.npz)")
    f.add_argument("--out", required=True)
    fl = sub.add_parser("fit-layers")
    fl.add_argument("--study", required=True)
    fl.add_argument("--layers", required=True)
    fl.add_argument("--kinds", default=None, help="the sens.json of a fit (the relative s of the kinds)")
    fl.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--study", required=True)
    p.add_argument("--sens", required=True)
    p.add_argument("--budget-gb", type=float, required=True, help="the bytes of the dense matrices")
    p.add_argument("--types", default="RQ4_K,RQ5_K,RQ6_K,RQ8_0,BF12")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"fit": fit, "fit-layers": fit_layers, "plan": plan}[args.cmd](args)


if __name__ == "__main__":
    main()
