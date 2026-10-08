#!/usr/bin/env python3
"""The mean KL of the GPU prompt (mixed groups) against CPU references at several positions.

The KL of the last token alone moves from 0.004 to 0.046 between runs of the
same settings (the split of the experts between the GPU and the CPU changes
the numbers), so this compares at CUTS positions: one CPU forward makes each
reference (REF=path.npz, made when missing; NP_GEMMA_MOE_XQ=f for the exact
one), then the GPU reads the prompt in parts that end at the cuts. On the
2-socket Xeon (4608 tokens of the notes and sources, 9 cuts): the int16 CPU
path against the exact one, mean KL 0.0066 (max 0.033); all the experts on the
GPU 0.0061; the mixed groups 0.004-0.010; the top token the same at each cut.

    GF=model.gguf REF=/tmp/ref_f.npz NP_GEMMA_MOE_XQ=f REF_ONLY=1 python scripts/check_qwen4_kl.py
    GF=model.gguf REF=/tmp/ref_f.npz [REFS=a.npz,b.npz] [CUTS=...] [SAVE=gpu.npy] \\
        python scripts/check_qwen4_kl.py
"""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import np_gemma  # noqa
import numpy as np
from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU
from np_gemma.qwen_tok import QwenTokenizer

GF = os.environ["GF"]
REF = os.environ["REF"]
N = int(os.environ.get("TOKENS", "4608"))
CUTS = [int(c) for c in os.environ.get("CUTS", "512,1024,1536,2048,2560,3072,3584,4096,4608").split(",")]
tok = QwenTokenizer(os.path.join(os.path.dirname(GF), "tokenizer.json"))
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
text = ""
for f in sorted(glob.glob(here + "/*.md")) + sorted(glob.glob(here + "/np_gemma/*.py")):
    text += "\n\n===== %s =====\n" % os.path.basename(f) + open(f, errors="replace").read()
ids = tok.encode("<|im_start|>user\n" + text)[:N]
m = Qwen4CPU(GF)
if not os.path.exists(REF):
    c = Qwen4Cache(m.cfg, N + 64)
    h = m.forward(ids, c, 0)
    lg = np.stack([np.asarray(m.logits(h[k - 1:k])[0], np.float32) for k in CUTS])
    np.savez(REF, logits=lg, cuts=np.array(CUTS))
    print("reference made: %s (NP_GEMMA_MOE_XQ=%s)" % (REF, os.environ.get("NP_GEMMA_MOE_XQ", "")), flush=True)
    if os.environ.get("REF_ONLY") == "1":
        sys.exit(0)
refs = {}
for r in os.environ.get("REFS", REF).split(","):
    # the rows of the reference at CUTS (a reference of more cuts serves a
    # run of some of them)
    z = np.load(r)
    at = {int(c): j for j, c in enumerate(z["cuts"])}
    missing = [k for k in CUTS if k not in at]
    assert not missing, "%s has no logits at %s" % (r, missing)
    refs[r] = z["logits"][[at[k] for k in CUTS]].astype(np.float64)
from np_gemma.qwen4_gpu import Qwen4GPU
g = Qwen4GPU(m, ctx=N + 64)
c = Qwen4Cache(m.cfg, N + 64)
g.attach(c)
gots = []
pos = 0
for j, k in enumerate(CUTS):
    g.prefill(ids[pos:k], pos=pos)
    pos = k
    gots.append(np.asarray(g.logits(), np.float64).reshape(-1))
if os.environ.get("SAVE"):
    np.save(os.environ["SAVE"], np.stack(gots))
for name, ref in refs.items():
    kls, tops = [], 0
    for j, got in enumerate(gots):
        pr = ref[j] - ref[j].max(); pr -= np.log(np.exp(pr).sum())
        pg = got - got.max(); pg -= np.log(np.exp(pg).sum())
        kls.append(float((np.exp(pr) * (pr - pg)).sum()))
        tops += int(got.argmax() == ref[j].argmax())
    print("RESULT vs %s: mean KL %.4f (max %.4f), top %d/%d; %s" % (
        os.path.basename(name), np.mean(kls), np.max(kls), tops, len(kls), " ".join("%.4f" % x for x in kls)), flush=True)
