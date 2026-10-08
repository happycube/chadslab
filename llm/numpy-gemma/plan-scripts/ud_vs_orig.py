"""How far are the UD-Q4_K_XL expert tensors from ORIG (dequant of Unsloth, same experts)."""
import sys, os, io, contextlib
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import study_orig_q8 as O
from np_gemma.gguf import open_gguf, _TYPE_NAME
S = O.S
g = open_gguf("models2/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
rq6 = S.rotated(S.q6k)
rng = np.random.default_rng(3)
print("%-6s %-5s %-7s %10s %10s %10s" % ("layer", "mat", "UD type", "UD", "RQ6_K", "Q8_0"))
for L in (0, 2, 4, 12, 30, 46, 47):
    for e in rng.choice(512, 3, replace=False):
        W = O.oexpert(O.LM + "layers.%d.mlp.experts." % L, int(e))
        for x, w in W.items():
            w = np.ascontiguousarray(w, np.float32)
            gn = "blk.%d.ffn_%s_exps.weight" % (L, x)
            u = g.dequant(gn, rows=[int(e)])[0]
            if u.shape != w.shape: u = u.reshape(w.shape)
            t = _TYPE_NAME[g.tensors[gn][1]]
            print("%-6d %-5s %-7s %s %s %s" % (L, x, t, S.db(S.rel(u, w)), S.db(S.rel(rq6(w), w)), S.db(S.rel(S.q_absmax(w, 8), w))), flush=True)
# the dense: one of each kind, UD vs ORIG
for gn, on in (("blk.23.ffn_gate_shexp.weight", O.LM + "layers.23.mlp.shared_expert.gate_proj.weight"),
               ("blk.23.attn_v.weight", O.LM + "layers.23.self_attn.v_proj.weight")):
    u = g.dequant(gn); w = O.oget(on)
    print("%-28s %-5s %s (UD vs ORIG)" % (gn, _TYPE_NAME[g.tensors[gn][1]], S.db(S.rel(u.reshape(w.shape), w))))
