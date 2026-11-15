"""Check reclank's ACE-Step 1.5 planner LM (models/acestep/planner.py) and the DiT's side of its
codes against scripts/acestep_reference.py --thinking (the authors' LM, backend "pt", float32).

    check_acestep_lm.py MODEL_DIR ref.npz [--gpu]

The prompts' ids; the logits of each step with the reference's tokens fed (both phases;
phase 2's two rows: the conditional and the unconditional prompt), as KL(reference ||
reclank) and the steps whose argmax differs; the constraint along the reference's tokens
(each one allowed); the metadata and the codes parsed; the hints of the codes against the
reference's source latents; the DiT's text and lyric ids."""
import argparse
import ast
import re
import time

import numpy as np

from reclank.models.acestep.pipeline import AceStep
from reclank.models.acestep.planner import CODE0, Constraint

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("ref")
ap.add_argument("--gpu", action="store_true")
a = ap.parse_args()
r = np.load(a.ref)
t0 = time.time()
ace = AceStep(a.model, gpu=a.gpu)
pl = ace.planner
print("loaded in %.1f s (%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU"), flush=True)


def kl(ref, x):
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    lp = ref - ref.max(-1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    lq = x - x.max(-1, keepdims=True)
    lq -= np.log(np.exp(lq).sum(-1, keepdims=True))
    p = np.exp(lp)
    with np.errstate(invalid="ignore"):         # (-inf - -inf where the constraint masks a token: p = 0 there)
        return np.where(p > 0, p * (lp - lq), 0.0).sum(-1)


n = int(r["lm_n"])
calls = [(r["lm_ids%d" % i], r["lm_logits%d" % i]) for i in range(n)]
p2 = next(i for i in range(1, n) if calls[i][0].shape[1] > 1)          # phase 2's prefill
caption, lyrics = str(r["caption"]), str(r["lyrics"])
meta_ref = ast.literal_eval(str(r["lm_metadata"]))
ids1 = pl.tok.encode(pl.prompt_cot(caption, lyrics))
print("phase 1 prompt %s (%d)" % ("same" if ids1 == list(calls[0][0][0]) else "DIFFER", len(ids1)))
strip = lambda row: [int(t) for t in row if t != pl.pad]  # noqa: E731
c2, u2 = pl.tok.encode(pl.prompt_codes(caption, lyrics, meta_ref)), pl.tok.encode(pl.prompt_uncond())
print("phase 2 prompts %s, %s" % ("same" if c2 == strip(calls[p2][0][0]) else "DIFFER",
                                   "same" if u2 == strip(calls[p2][0][1]) else "DIFFER"))
for ph, (lo, hi, prompts, con) in enumerate([
        (0, p2, [pl.prompt_cot(caption, lyrics)], Constraint(pl, "cot")),
        (p2, n, [pl.prompt_codes(caption, lyrics, meta_ref), pl.prompt_uncond()],
         Constraint(pl, "codes", target_codes=int(meta_ref["duration"]) * 5))], 1):
    toks = [int(calls[i][0][0, -1]) for i in range(lo + 1, hi)] + [pl.eos]
    keep = []
    t1 = time.time()
    pl.generate(prompts, Constraint(pl, con.phase, target_codes=con.target), cfg=2.0, forced=toks, keep=keep)
    dt = (time.time() - t1) / len(toks)
    # the reference's phase 1 logits were masked in place by the authors' constraint after the
    # hook kept them (a view): compare over the tokens the mask kept
    ours = [np.where(np.isfinite(calls[lo + i][1]), keep[i], -np.inf) for i in range(len(keep))]
    k = np.concatenate([kl(calls[lo + i][1], ours[i]) for i in range(len(keep))])
    am = sum(int((calls[lo + i][1].argmax(-1) != ours[i].argmax(-1)).sum()) for i in range(len(keep)))
    bad = 0
    for i, t in enumerate(toks):
        lg = keep[i]
        comb = lg[1] + np.float32(2.0) * (lg[0] - lg[1]) if len(prompts) == 2 else lg[0]
        allowed = con.allowed(comb)
        bad += allowed is not None and t not in set(allowed.tolist())
        con.update(t)
    print("phase %d: %d steps, KL worst %.2e mean %.2e, argmax differs %d; tokens not allowed %d; %.1f ms a step"
          % (ph, len(toks), k.max(), k.mean(), am, bad, 1000 * dt), flush=True)
    if ph == 1:
        meta = pl.parse(toks)
        print("  metadata %s" % ("same" if meta == meta_ref else "DIFFER: %r vs %r" % (meta, meta_ref)))
codes_ref = np.array([int(x) for x in re.findall(r"<\|audio_code_(\d+)\|>", str(r["lm_codes"]))])
toks2 = [int(calls[i][0][0, -1]) for i in range(p2 + 1, n)] + [pl.eos]
codes = np.array([t - CODE0 for t in toks2 if t >= CODE0])
print("codes %s (%d)" % ("same" if np.array_equal(codes, codes_ref) else "DIFFER", len(codes)))
hints = ace.dit.detokenize(codes_ref)
src = r["context"][:, :64]
x, y = hints[:len(src)].astype(np.float64), src[:len(hints)].astype(np.float64)
cos = (x * y).sum(-1) / np.linalg.norm(x, axis=-1) / np.linalg.norm(y, axis=-1)
print("hints: 1-cos worst %.2e mean %.2e, rel %.2e (%d frames)" % ((1 - cos).max(), (1 - cos).mean(),
                                                                      np.abs(x - y).max() / np.abs(y).max(), len(x)))
from reclank.models.acestep.planner import INSTRUCTION as COVER  # noqa: E402
tids = ace.text_ids(meta_ref["caption"], meta_ref.get("bpm"), meta_ref.get("timesignature"), meta_ref.get("keyscale"),
                    len(codes_ref) / 5.0, COVER)
print("DiT text ids %s, lyric ids %s" % ("same" if tids == list(r["text_ids"]) else "DIFFER",
                                         "same" if ace.lyric_ids(lyrics, "en") == list(r["lyric_ids"]) else "DIFFER"))
if ace.lyric_ids(lyrics, "en") != list(r["lyric_ids"]):
    print("  ours: %r\n  ref:  %r" % (pl.tok.decode(ace.lyric_ids(lyrics, "en")), pl.tok.decode(list(r["lyric_ids"]))))
if tids != list(r["text_ids"]):
    print("  ours: %r\n  ref:  %r" % (pl.tok.decode(tids)[-200:], pl.tok.decode(list(r["text_ids"]))[-200:]))
