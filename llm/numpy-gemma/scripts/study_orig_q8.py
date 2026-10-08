#!/usr/bin/env python3
"""A study (no change to the runtime): the forms of RQ8_EXPERTS_PLAN.md on
the original bfloat16 weights of Qwen3.8-Flash-Next (ORIG), against those
weights. Three parts:

1. checks: the shared experts and the head of ORIG equal the bfloat16 ones of
   the ModelOpt checkpoint; gate is rows 0-639 of gate_up_proj (against the
   NVFP4 of ModelOpt);
2. the routed experts (6 experts of layers 0, 12, 24, 36, 47 and of the MTP
   layer): each form, and the quantization that ModelOpt shipped (NVFP4, and
   FP8 128x128 for the MTP layer);
3. the dense matrices: Q8_0 against rotated Q8_0;
4. the n-gram table (200000 rows of 5 shards): the FP8 of ModelOpt, Q8_0,
   rotated Q8_0.

    PYTHONPATH=. python scripts/study_orig_q8.py [ORIG] [MODELOPT]

ORIG: models/Qwen3.8-Flash-Next by default; MODELOPT:
models/Qwen3.8-Flash-Next-NVFP4. About 6 minutes.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ORIG = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else
                          "models/Qwen3.8-Flash-Next")
if len(sys.argv) > 2:
    sys.argv = [sys.argv[0], sys.argv[2]]      # the ModelOpt path for study_tq6_experts
else:
    sys.argv = sys.argv[:1]
import study_tq6_experts as S  # noqa: E402
from np_gemma.st import SafeTensors  # noqa: E402

OW = json.load(open(os.path.join(ORIG, "model.safetensors.index.json")))["weight_map"]
_files = {}
LM = "model.language_model."


def ost(name):
    f = OW[name]
    if f not in _files:
        _files[f] = SafeTensors(os.path.join(ORIG, f))
    return _files[f]


def bf(u):
    return (np.asarray(u, np.uint32) << 16).view(np.float32)


def oget(name):
    return bf(ost(name).get_bf16(name))


def oexpert(prefix, e):
    """gate, up, down (float32) of expert e: the experts of ORIG are fused,
    gate_up_proj (E, 1280, 2560) with gate in rows 0-639, down_proj (E, 2560,
    640)."""
    gu = bf(ost(prefix + "gate_up_proj").get_bf16(prefix + "gate_up_proj")[e])
    dn = bf(ost(prefix + "down_proj").get_bf16(prefix + "down_proj")[e])
    return {"gate": gu[:640], "up": gu[640:], "down": dn}


E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def nvfp4_shipped(p):
    """The NVFP4 matrix of ModelOpt at prefix p, as float32."""
    c = S.get(p + "weight", dtype=None)
    s = S.E4M3[S.get(p + "weight_scale", dtype=None)]
    g = float(np.asarray(S.get(p + "weight_scale_2")).reshape(-1)[0])
    v = np.stack([E2M1[c & 15], E2M1[c >> 4]], -1).reshape(c.shape[0], -1)
    return v * np.repeat(s, 16, 1) * g


def fp8_shipped(p):
    """The FP8 128x128 matrix of ModelOpt (the MTP experts) at prefix p."""
    c = S.E4M3[S.get(p + "weight", dtype=None)]
    s = S.get(p + "weight_scale_inv")
    return c * np.repeat(np.repeat(s, 128, 0), 128, 1)[:c.shape[0], :c.shape[1]]


def checks():
    for n in (LM + "layers.0.mlp.shared_expert.gate_proj.weight",
              "mtp.layers.0.mlp.shared_expert.down_proj.weight"):
        print("ORIG == ModelOpt bfloat16:", n, np.array_equal(oget(n), S.get(n)))
    w = oexpert(LM + "layers.0.mlp.experts.", 3)
    p = LM + "layers.0.mlp.experts.3.%s_proj."
    cc = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])  # noqa: E731
    print("corr with the NVFP4 of ModelOpt: gate %.4f up %.4f down %.4f (gate vs up rows %.4f)"
          % (cc(nvfp4_shipped(p % "gate"), w["gate"]), cc(nvfp4_shipped(p % "up"), w["up"]),
             cc(nvfp4_shipped(p % "down"), w["down"]), cc(nvfp4_shipped(p % "gate"), w["up"])))


def routed(ne=6, layers=(0, 12, 24, 36, 47, "mtp")):
    q8 = lambda w: S.q_absmax(w, 8)  # noqa: E731
    forms = [("Q8_0 (8.5)", q8), ("rotated Q8_0 (8.5)", S.rotated(q8)),
             ("rotated Q8_K (8.125)", S.rotated(S.q8k)),
             ("rotated Q6_K (6.56)", S.rotated(S.q6k)), ("Q6_K (6.56)", S.q6k),
             ("TQ6 (6.5)", S.tq), ("NVFP4 searched (4.5)", S.nvfp4)]
    names = [n for n, _ in forms] + ["ModelOpt (as shipped)"]
    err = {(n, L): [] for n in names for L in layers}
    kurt = []
    rng = np.random.default_rng(7)
    for L in layers:
        pre = "mtp.layers.0.mlp.experts." if L == "mtp" else LM + "layers.%d.mlp.experts." % L
        for e in rng.choice(512, ne, replace=False):
            for x, w in oexpert(pre, int(e)).items():
                w = np.ascontiguousarray(w, np.float32)
                kurt.append(((w - w.mean()) ** 4).mean() / w.var() ** 2)
                for n, f in forms:
                    err[(n, L)].append(S.rel(f(w), w))
                p = pre[:-len("experts.")] + "experts.%d.%s_proj." % (e, x)
                q = fp8_shipped(p) if L == "mtp" else nvfp4_shipped(p)
                err[("ModelOpt (as shipped)", L)].append(S.rel(q, w))
    print("\nThe routed experts of ORIG: %d matrices, kurtosis %.0f-%.0f" %
          (len(kurt), min(kurt), max(kurt)))
    print("%-24s %10s %10s   per layer: %s" % ("form", "mean", "worst",
                                              " ".join("%7s" % L for L in layers)))
    for n in names:
        a = np.concatenate([err[(n, L)] for L in layers])
        print("%-24s %s %7.2f dB   %s" % (n, S.db(a.mean()), 10 * np.log10(a.max()),
                                          " ".join("%7.2f" % (10 * np.log10(np.mean(err[(n, L)])))
                                                   for L in layers)))
    print("(as shipped: NVFP4 in layers 0-47, FP8 128x128 in the MTP layer)")


def dense():
    q8 = lambda w: S.q_absmax(w, 8)  # noqa: E731
    rq8 = S.rotated(q8)
    groups = {
        "DeltaNet in_proj_qkv": [LM + "layers.%d.linear_attn.in_proj_qkv.weight" % i for i in (0, 22, 46)],
        "DeltaNet in_proj_z": [LM + "layers.%d.linear_attn.in_proj_z.weight" % i for i in (0, 22, 46)],
        "DeltaNet out_proj": [LM + "layers.%d.linear_attn.out_proj.weight" % i for i in (0, 22, 46)],
        "attn q_proj": [LM + "layers.%d.self_attn.q_proj.weight" % i for i in (3, 23, 47)],
        "attn k_proj": [LM + "layers.%d.self_attn.k_proj.weight" % i for i in (3, 23, 47)],
        "attn v_proj": [LM + "layers.%d.self_attn.v_proj.weight" % i for i in (3, 23, 47)],
        "attn o_proj": [LM + "layers.%d.self_attn.o_proj.weight" % i for i in (3, 23, 47)],
        "hc mix down": [LM + "layers.%d.attn_hyper_connection.input_mix_weight_down.weight" % i
                        for i in (0, 23, 47)],
        "hc mix up": [LM + "layers.%d.mlp_hyper_connection.input_mix_weight_up.weight" % i
                      for i in (0, 23, 47)],
        "ple key/value": [LM + "layers.1.ple.key_proj.weight", LM + "layers.1.ple.value_proj.weight"],
        "shared experts": [LM + "layers.%d.mlp.shared_expert.%s_proj.weight" % (i, x)
                           for i in (0, 23, 47) for x in ("gate", "up", "down")],
        "lm_head (40K rows)": ["lm_head.weight"],
        "embed (40K rows)": [LM + "embed_tokens.weight"],
    }
    rng = np.random.default_rng(0)
    print("\nThe dense matrices of ORIG:")
    print("%-22s %-14s %6s %10s %10s %8s   %s" % ("matrix", "shape", "kurt", "Q8_0", "RQ8_0", "gain",
                                                 "worst Q8_0 / RQ8_0"))
    for g, names in groups.items():
        e1, e2, ks, shp = [], [], [], None
        for n in names:
            st = ost(n)
            sh = st.shape(n)
            if sh[0] > 40000:
                w = bf(st.get_bf16(n)[np.sort(rng.choice(sh[0], 40000, replace=False))])
            else:
                w = oget(n)
            w = np.ascontiguousarray(w, np.float32)
            shp = w.shape
            ks.append(((w - w.mean()) ** 4).mean() / w.var() ** 2)
            e1.append(S.rel(q8(w), w))
            e2.append(S.rel(rq8(w), w))
        a, b = np.mean(e1), np.mean(e2)
        print("%-22s %-14s %6.0f %s %s %5.2f dB   %.1f / %.1f" % (
            g, "x".join(map(str, shp)), max(ks), S.db(a), S.db(b), 10 * np.log10(a / b),
            10 * np.log10(max(e1)), 10 * np.log10(max(e2))), flush=True)


def ngram():
    p = LM + "layers.1.ple.ple_embedding.ngram_embedding."
    scale = float(np.asarray(S.get(p + "weight_scale")).reshape(-1)[0])
    rng = np.random.default_rng(5)
    ws, fs = [], []
    for sh in (0, 31, 63, 95, 127):
        n = p + "shard_%d.weight" % sh
        orig, codes = ost(n).get_bf16(n), S.get(n, dtype=None)
        for _ in range(4):
            r0 = int(rng.integers(0, orig.shape[0] - 10000))
            ws.append(bf(orig[r0:r0 + 10000]))
            fs.append(S.E4M3[codes[r0:r0 + 10000]] * scale)
    w = np.concatenate(ws).astype(np.float32)
    q8 = lambda a: S.q_absmax(a, 8)  # noqa: E731
    forms = {"FP8, one scale (ModelOpt)": np.concatenate(fs).astype(np.float32),
             "Q8_0": q8(w), "rotated Q8_0": S.rotated(q8)(w)}
    print("\nThe n-gram table: %d rows of 160, kurtosis %.1f" %
          (w.shape[0], ((w - w.mean()) ** 4).mean() / w.var() ** 2))
    print("%-28s %10s   per row: %10s %10s %10s" % ("form", "table", "median", "99%", "worst"))
    nz = np.abs(w).max(1) > 0
    for n, q in forms.items():
        e = ((q - w) ** 2).sum(1)[nz] / (w ** 2).sum(1)[nz]
        print("%-28s %s   %s %s %s" % (n, S.db(S.rel(q, w)), S.db(np.median(e)),
                                       S.db(np.percentile(e, 99)), S.db(e.max())))


if __name__ == "__main__":
    checks()
    routed()
    dense()
    ngram()
