"""Check reclank's Higgs TTS 3 against the references of scripts/higgs_reference.py.

    check_higgs.py MODEL_DIR CODEC_CONFIG ref.npz [ref.npz ...] [--gpu] [--parts lm,greedy,codec]
        [--wav-dir DIR]

lm: the prompt's ids, then the logits of each step with the reference's codes fed (the
worst and mean KL(reference || reclank) of a codebook's softmax, the codebooks whose argmax
differs). greedy: the greedy codes against the reference's. codec: the audio of the
reference's codes (the largest difference, the SNR in dB)."""
import argparse
import time
import wave

import numpy as np

from reclank.models.higgs.tts import HiggsTTS

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("codec_config")
ap.add_argument("refs", nargs="+")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--parts", default="lm,greedy,codec")
ap.add_argument("--wav-dir")
a = ap.parse_args()
parts = set(a.parts.split(","))
t0 = time.time()
tts = HiggsTTS(a.model, a.codec_config, gpu=a.gpu)
print("loaded in %.1f s (%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU"), flush=True)


def kl(ref, x):
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    lp = ref - ref.max(-1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    lq = x - x.max(-1, keepdims=True)
    lq -= np.log(np.exp(lq).sum(-1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(-1)


def wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())


for p in a.refs:
    r = np.load(p)
    text = str(r["text"])
    print("%s: %r" % (p.split("/")[-1], text[:70]))
    ids = tts.prompt(text)
    print("  ids %s (%d)" % ("same" if ids == list(r["ids"]) else "DIFFER", len(ids)))
    if "lm" in parts:
        keep = []
        t1 = time.time()
        tts.codes(text, forced=r["codes"], keep=keep)
        dt = time.time() - t1
        x, ref = np.stack(keep), r["logits"]
        k = kl(ref, x)
        am = int((x.argmax(-1) != ref.argmax(-1)).sum())
        print("  forced %d steps: KL worst %.2e mean %.2e, argmax differs %d of %d; %.1f ms a step" % (
            len(x), k.max(), k.mean(), am, k.size, 1000 * dt / len(x)), flush=True)
    if "greedy" in parts:
        t1 = time.time()
        c = tts.codes(text, temperature=0)
        dt = time.time() - t1
        same = c.shape == r["codes"].shape and np.array_equal(c, r["codes"])
        n = min(len(c), len(r["codes"]))
        first = next((i for i in range(n) if not np.array_equal(c[i], r["codes"][i])), n)
        print("  greedy: %d steps (reference %d), %s; first difference at step %s; %.1f s" % (
            len(c), len(r["codes"]), "the same" if same else "DIFFER", first if not same else "-", dt), flush=True)
        if a.wav_dir:
            wav("%s/%s_greedy.wav" % (a.wav_dir, p.split("/")[-1][:-4]), tts.codec.decode(tts.undelay(c)))
    if "codec" in parts:
        t1 = time.time()
        au = tts.codec.decode(r["raw"])
        dt = time.time() - t1
        ref = r["audio"]
        d = au - ref
        snr = 10 * np.log10((ref ** 2).sum() / max((d ** 2).sum(), 1e-30))
        print("  codec: %d samples, max |d| %.2e, SNR %.1f dB; %.2f s" % (len(au), np.abs(d).max(), snr, dt))
