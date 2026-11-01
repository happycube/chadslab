"""Check reclank's Whisper decoder against transformers' (scripts/whisper_decoder_reference.py)
on real clips: the logits of the reference's tokens a step at a time (the encoder on the
reference's features), then greedy generation.

    check_whisper_decoder.py MODEL_DIR refs.npz dec_refs.npz [--gpu] [--batch N] [--lang sk]

Prints, a clip: the logits' worst difference relative to the row's largest reference
logit, the worst and mean KL(reference || reclank) of a row's softmax, the rows whose
most likely token differs, and whether generate's tokens are the reference's."""
import argparse
import time

import numpy as np

from reclank.models.whisper.decoder import Whisper

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("refs")
ap.add_argument("dec_refs")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lang", default="sk")
ap.add_argument("--threads", type=int, default=None)
a = ap.parse_args()
r, d = np.load(a.refs), np.load(a.dec_refs)
names = list(d["names"])
feats = [r["f%d" % i] for i in range(len(names))]
toks = [d["t%d" % i] for i in range(len(names))]
t0 = time.time()
wh = Whisper(a.model, gpu=a.gpu, cpu_threads=a.threads, batch=a.batch)
print("loaded in %.1f s (%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU"), flush=True)


def kl(ref, x):
    """KL(softmax(ref) || softmax(x)) of each row, in float64."""
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    lp = ref - ref.max(-1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    lq = x - x.max(-1, keepdims=True)
    lq -= np.log(np.exp(lq).sum(-1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(-1)


worst_kl, bad_rows, steps, step_s = 0.0, 0, 0, 0.0
for b in range(0, len(names), a.batch):
    ids = list(range(b, min(b + a.batch, len(names))))
    n = len(ids)
    with wh._span():        # (the caches from the encoder's program to the last step)
        wh.encode([feats[i] for i in ids])
        T = max(len(toks[i]) - 1 for i in ids)
        got = [[] for _ in ids]
        for p in range(T):
            cur = np.array([toks[i][min(p, len(toks[i]) - 1)] for i in ids])
            t1 = time.time()
            lg = wh.step(n, cur[:, None], np.full((n, 1), p))
            step_s += time.time() - t1
            steps += 1
            for j, i in enumerate(ids):
                if p < len(toks[i]) - 1:
                    got[j].append(lg[j])
    for j, i in enumerate(ids):
        x, ref = np.array(got[j]), d["l%d" % i]
        rel = np.abs(x - ref).max(-1) / np.abs(ref).max(-1)
        k = kl(ref, x)
        am = int((x.argmax(-1) != ref.argmax(-1)).sum())
        worst_kl, bad_rows = max(worst_kl, k.max()), bad_rows + am
        print("%-40s %3d rows  rel worst %.2e  KL worst %.2e mean %.2e  argmax differs %d" % (
            names[i], len(ref), rel.max(), k.max(), k.mean(), am), flush=True)
print("teacher forced: KL worst %.2e, %d rows' argmax differ; %.1f ms a step of %d windows" % (
    worst_kl, bad_rows, 1000 * step_s / steps, a.batch), flush=True)

t1 = time.time()
seqs = wh.generate(feats, lang=a.lang)
dt = time.time() - t1
same = 0
for i, s in enumerate(seqs):
    ok = list(s) == [int(t) for t in toks[i]]
    same += ok
    print("%-40s %s %3d tokens  %r" % (names[i], "same" if ok else "DIFFERS", len(s), wh.tok.text(s)[:100]))
    if not ok:
        ref = [int(t) for t in toks[i]]
        k = next((j for j in range(min(len(s), len(ref))) if s[j] != ref[j]), min(len(s), len(ref)))
        print("    first difference at token %d: %s vs reference %s" % (k, s[k:k + 4], ref[k:k + 4]))
print("generate: %d/%d the reference's tokens; %.1f s for %d windows (%d tokens)" % (
    same, len(seqs), dt, len(seqs), sum(len(s) for s in seqs)))
