import sys, os, io, contextlib
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import study_orig_q8 as O
S = O.S
q8 = lambda w: S.q_absmax(w, 8)
rng = np.random.default_rng(11)
experts = rng.choice(512, 32, replace=False)
F = {"FP8 128x128 (shipped, 8.0)": None,
     "FP8 -> Q8_0 (the runtime now, 8.5)": None,
     "rotated Q6_K (6.56)": S.rotated(S.q6k),
     "Q6_K (6.56)": S.q6k,
     "Q8_0 of ORIG (8.5)": q8}
err = {n: {x: [] for x in ("gate", "up", "down")} for n in F}
oerr = {n: [] for n in F}
for e in experts:
    W = O.oexpert("mtp.layers.0.mlp.experts.", int(e))
    for x, w in W.items():
        w = np.ascontiguousarray(w, np.float32)
        fp8 = O.fp8_shipped("mtp.layers.0.mlp.experts.%d.%s_proj." % (e, x))
        X = rng.standard_normal((32, w.shape[1])).astype(np.float32)
        for n, f in F.items():
            q = fp8 if n.startswith("FP8 128") else q8(fp8) if n.startswith("FP8 ->") else f(w)
            err[n][x].append(S.rel(q, w)); oerr[n].append(S.rel(X @ q.T, X @ w.T))
print("The MTP experts: %d experts x gate/up/down = %d matrices, against the original bfloat16" % (len(experts), 3 * len(experts)))
print("%-36s %10s %10s %10s %10s %10s %10s" % ("form", "gate", "up", "down", "mean", "worst", "output"))
for n in F:
    a = sum(err[n].values(), [])
    print("%-36s %s %s %s %s %7.2f dB %s" % (n, *(S.db(np.mean(err[n][x])) for x in ("gate", "up", "down")),
          S.db(np.mean(a)), 10 * np.log10(max(a)), S.db(np.mean(oerr[n]))))
# the per-matrix gap rq6k vs fp8
r = np.array(sum(err["rotated Q6_K (6.56)"].values(), [])); f8 = np.array(sum(err["FP8 128x128 (shipped, 8.0)"].values(), []))
d = 10 * np.log10(f8 / r)
print("rotated Q6_K better than FP8 in %d of %d matrices; gap %.2f to %.2f dB (mean %.2f)" % ((d > 0).sum(), d.size, d.min(), d.max(), d.mean()))
