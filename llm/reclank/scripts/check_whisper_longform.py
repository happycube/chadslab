"""Check reclank's long-form Whisper (decoder.Whisper.segments) against transformers'
generate (scripts/whisper_longform_reference.py) on real clips of more than 30 s.

    check_whisper_longform.py MODEL_DIR long.npz ref.npz [--gpu] [--lang sk | --lang none]
        [--task transcribe] [--prompt TEXT] [--only I] [--no-timestamps]

Prints Tokens.encode against the tokenizer's get_prompt_ids of the reference's texts;
then, a clip: the features' largest difference (log_mel_all against the feature
extractor's), the language (with --lang none: detected), the tokens the same as the
reference's up to the first difference and in all (difflib's matches over the
reference's), the segments of the same start and end, the word and character error
rates of reclank's text against the reference's (no normalization), and the time
against the audio's (the real-time factor)."""
import argparse
import difflib
import time

import numpy as np

from reclank.models.whisper.decoder import Whisper
from reclank.models.whisper.encoder import SAMPLES, log_mel, log_mel_all

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("clips")
ap.add_argument("ref")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--lang", default="sk")
ap.add_argument("--task", default="transcribe")
ap.add_argument("--prompt", default=None)
ap.add_argument("--only", type=int, default=None)
ap.add_argument("--no-timestamps", action="store_true", help="(clips of at most 30 s) one window, no timestamps")
ap.add_argument("--threads", type=int, default=None)
a = ap.parse_args()
d, r = np.load(a.clips), np.load(a.ref)


def edits(ref, hyp):
    """The Levenshtein distance of two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, x in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, y in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


t0 = time.time()
wh = Whisper(a.model, gpu=a.gpu, cpu_threads=a.threads, batch=1)
print("loaded in %.1f s (%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU"), flush=True)
bad = 0
for i, t in enumerate(r["texts"]):
    ref = [int(v) for v in r["p%d" % i]]
    got = [wh.prev_sot] + wh.tok.encode(" " + str(t).strip())
    bad += got != ref
    if got != ref:
        print("  encode differs: %r\n    %s\n    %s" % (str(t), got, ref))
print("Tokens.encode: %d/%d texts the tokenizer's ids" % (len(r["texts"]) - bad, len(r["texts"])))
prompt = None
if a.prompt:
    prompt = wh.tok.encode(" " + a.prompt.strip())
    print("prompt: %s the reference's ids" % ("the same as" if [wh.prev_sot] + prompt == list(r["prompt"]) else "NOT"))
for i, name in enumerate(d["names"]):
    if a.only is not None and i != a.only or "t%d" % i not in r:
        continue
    x = d["c%d" % i]
    fd = float(np.abs((log_mel(x) if len(x) <= SAMPLES else log_mel_all(x)) - r["f%d" % i]).max())
    t1 = time.time()
    out = wh.segments(x, lang=None if a.lang == "none" else a.lang, task=a.task, prompt=prompt,
                     timestamps=not a.no_timestamps)
    dt = time.time() - t1
    ref, got = [int(v) for v in r["t%d" % i]], out["tokens"]
    first = next((j for j in range(min(len(ref), len(got))) if ref[j] != got[j]), min(len(ref), len(got)))
    match = sum(b.size for b in difflib.SequenceMatcher(None, ref, got, autojunk=False).get_matching_blocks())
    rs = r["s%d" % i]
    gs = np.array([[s["start"], s["end"]] for s in out["segments"]], np.float64)
    same_seg = sum(1 for s in gs if np.any(np.all(np.abs(rs - s) < 1e-9, axis=1)))
    rt, gt = str(r["text%d" % i]), out["text"]
    wer = edits(rt.split(), gt.split()) / max(len(rt.split()), 1)
    cer = edits(rt, gt) / max(len(rt), 1)
    lang = out["language"] + ("" if "lang%d" % i not in r else " (reference %s)" % r["lang%d" % i])
    print("%s: %.0f s, language %s, features %.1e" % (name, out["duration"], lang, fd))
    print("  tokens: %d (reference %d), the same to token %d%s; %d of the reference's in the alignment (%.2f%%)" % (
        len(got), len(ref), first, " (all)" if got == ref else "", match, 100.0 * match / max(len(ref), 1)))
    print("  segments: %d (reference %d), %d of the same start and end" % (len(gs), len(rs), same_seg))
    print("  WER %.2f%%  CER %.2f%%;  %.1f s, real-time factor %.3f" % (100 * wer, 100 * cer, dt, dt / out["duration"]))
    if got != ref:
        ends = np.cumsum([len(s["tokens"]) for s in out["segments"]])
        seg = out["segments"][min(int(np.searchsorted(ends, first, side="right")), len(ends) - 1)]
        print("  first difference in the segment at %.2f s: %r | reference %r" % (
            seg["start"], wh.tok.text(got[first:first + 12]), wh.tok.text(ref[first:first + 12])))
    print("  text: %r" % gt[:200], flush=True)
