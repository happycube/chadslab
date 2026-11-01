"""Check reclank's Higgs TTS 3 voice cloning against the references of
scripts/higgs_clone_reference.py.

    check_higgs_clone.py MODEL_DIR CODEC_CONFIG ref.npz [ref.npz ...] [--gpu] [--parts codec,lm,greedy]
        [--wav-dir DIR]

codec: the codec's encoder of the reference's clip, each stage against the reference's (1 - cos and
the largest difference over the reference's largest value): the 16 kHz resample, HuBERT's feature
extractor, its hidden states 0 and 12 and their mean, the semantic input (every second row), the
semantic and the acoustic encoders, fc; the codes (the frames whose codes differ, and the first
codebook of each; an RVQ argmax can flip on a near tie: the margin of the reference's code there),
and the RVQ of the reference's fc rows on the host. lm: the clone prompt's ids, then the logits of
each step with the reference's codes fed (the voice: the reference's codes), KL(reference ||
reclank) of a codebook's softmax, the codebooks whose argmax differs. greedy: the greedy codes of the
reference's voice codes against the reference's, then of the voice of reclank's own codes."""
import argparse
import time
import wave

import numpy as np

from reclank.formats.safetensors import Shards
from reclank.models.higgs.codec import HiggsCodec
from reclank.models.higgs.tts import HiggsTTS, Voice

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("codec_config")
ap.add_argument("refs", nargs="+")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--parts", default="codec,lm,greedy")
ap.add_argument("--wav-dir")
a = ap.parse_args()
parts = set(a.parts.split(","))
t0 = time.time()
if parts == {"codec"}:
    tts = None
    codec = HiggsCodec(Shards(a.model), a.codec_config, gpu=a.gpu)
else:
    tts = HiggsTTS(a.model, a.codec_config, gpu=a.gpu)
    codec = tts.codec
print("loaded in %.1f s (%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU"), flush=True)


def kl(ref, x):
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    lp = ref - ref.max(-1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    lq = x - x.max(-1, keepdims=True)
    lq -= np.log(np.exp(lq).sum(-1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(-1)


def cmp(ref, x):
    ref, x = ref.astype(np.float64).reshape(-1), x.astype(np.float64).reshape(-1)
    cos = ref @ x / max(np.linalg.norm(ref) * np.linalg.norm(x), 1e-300)
    return "1 - cos %.1e, max |d| / max |ref| %.1e" % (1 - cos, np.abs(x - ref).max() / np.abs(ref).max())


def margins(enc, emb, codes):
    """The margin (best - second distance) of each frame's code in each codebook of the RVQ of
    emb with the codes given (the reference's)."""
    r = emb.astype(np.float32)
    out = []
    for i, (win, bin_, E, E2, wout, bout) in enumerate(enc.rvq):
        h = r @ win.T + bin_
        dist = -((h * h).sum(1, keepdims=True) - 2 * (h @ E.T) + E2)
        c = codes[:, i]
        best = dist[np.arange(len(c)), c]
        d2 = dist.copy()
        d2[np.arange(len(c)), c] = -np.inf
        out.append(best - d2.max(1))
        r = r - (E[c] @ wout.T + bout)
    return np.stack(out, 1)


def wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())


for p in a.refs:
    r = np.load(p)
    name = p.split("/")[-1][:-4]
    print("%s: %.2f s clip%s" % (name, len(r["wav"]) / 24000,
                                 (", text %r" % str(r["text"])[:60]) if "text" in r else ""))
    own = None
    if "codec" in parts:
        k = {}
        t1 = time.time()
        own = codec.encode(r["wav"], keep=k)
        dt = time.time() - t1
        enc = codec.encoder
        for mine, theirs in (("wav16", "wav16"), ("feat", "extract"), ("h0", "hub0"), ("x", "hublast"),
                             ("sem", "semantic"), ("e_sem", "e_semantic"), ("e_acoustic", "e_acoustic"),
                             ("emb", "emb")):
            print("  %-10s %s" % (theirs, cmp(r[theirs], k[mine])))
        ref = r["ref_codes"]
        frames = np.flatnonzero((own != ref).any(1)) if own.shape == ref.shape else None
        if frames is None:
            print("  codes: shape %s, reference %s" % (own.shape, ref.shape))
        else:
            mg = margins(enc, r["emb"], ref)
            first = [int(np.flatnonzero(own[f] != ref[f])[0]) for f in frames]
            print("  codes: %d frames, %d differ (%d codes of %d)%s; %.2f s" % (
                len(ref), len(frames), int((own != ref).sum()), ref.size,
                "".join("; frame %d from codebook %d (margin %.1e)" % (f, c, mg[f, c])
                        for f, c in list(zip(frames, first))[:4]), dt))
            host = enc.quantize(r["emb"])
            print("  RVQ of the reference's fc rows: %d codes differ" % int((host != ref).sum()))
    if "text" not in r or tts is None:
        continue
    text = str(r["text"])
    voice = Voice(r["ref_codes"], str(r["ref_text"]))
    ids = tts.prompt(text, voice)
    print("  ids %s (%d)" % ("same" if ids == list(r["ids"]) else "DIFFER", len(ids)))
    if "lm" in parts:
        keep = []
        t1 = time.time()
        tts.codes(text, voice, forced=r["codes"], keep=keep)
        dt = time.time() - t1
        x, ref = np.stack(keep), r["logits"]
        kk = kl(ref, x)
        am = int((x.argmax(-1) != ref.argmax(-1)).sum())
        print("  forced %d steps: KL worst %.2e mean %.2e, argmax differs %d of %d; %.1f ms a step" % (
            len(x), kk.max(), kk.mean(), am, kk.size, 1000 * dt / len(x)), flush=True)
    if "greedy" in parts:
        runs = [("reference's voice codes", voice)]
        if own is not None:
            runs.append(("reclank's voice codes", Voice(own, str(r["ref_text"]))))
        for label, v in runs:
            t1 = time.time()
            c = tts.codes(text, v, temperature=0)
            dt = time.time() - t1
            same = c.shape == r["codes"].shape and np.array_equal(c, r["codes"])
            n = min(len(c), len(r["codes"]))
            first = next((i for i in range(n) if not np.array_equal(c[i], r["codes"][i])), n)
            print("  greedy (%s): %d steps (reference %d), %s; first difference at step %s; %.1f s" % (
                label, len(c), len(r["codes"]), "the same" if same else "DIFFER", first if not same else "-", dt),
                flush=True)
            if a.wav_dir:
                tag = "ref" if v is voice else "own"
                wav("%s/%s_%s_%s.wav" % (a.wav_dir, name, tag, "gpu" if a.gpu else "cpu"),
                    tts.codec.decode(tts.undelay(c)))
