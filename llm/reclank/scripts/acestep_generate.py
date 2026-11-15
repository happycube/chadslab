"""Text to music with ACE-Step 1.5 turbo and its planner LM (reclank/models/acestep).

    acestep_generate.py MODEL_DIR --caption TEXT --out song.wav [--lyrics TEXT] [--duration 30]
        [--bpm 120] [--key "G major"] [--language en] [--seed 0] [--no-thinking] [--lm-language] [--cpu]

48 kHz stereo 16-bit WAV, the peak at -1 dB. The planner LM first unless --no-thinking (its metadata
and plan printed); --lm-language: the planner's language for the lyrics when --language is not
given."""
import argparse
import time
import wave

import numpy as np

from reclank.models.acestep.pipeline import AceStep

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--caption", required=True)
ap.add_argument("--lyrics", default="[Instrumental]")
ap.add_argument("--duration", type=float, default=None, help="seconds (default: the planner's choice, else 30)")
ap.add_argument("--bpm", type=int)
ap.add_argument("--key")
ap.add_argument("--language", default=None)
ap.add_argument("--no-thinking", action="store_true")
ap.add_argument("--lm-language", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--cpu", action="store_true")
a = ap.parse_args()
t0 = time.time()
ace = AceStep(a.model, gpu=not a.cpu, planner=not a.no_thinking)
t1 = time.time()
plan = {}
audio = ace.generate(a.caption, a.lyrics, a.duration, bpm=a.bpm, keyscale=a.key, language=a.language, seed=a.seed,
                     thinking=not a.no_thinking, lm_language=a.lm_language, keep=plan)
if plan:
    print("plan:", plan["meta"], "%d codes" % len(plan["codes"]))
t2 = time.time()
pcm = (np.clip(audio.T, -1, 1) * 32767).astype("<i2")
with wave.open(a.out, "wb") as w:
    w.setnchannels(audio.shape[0])
    w.setsampwidth(2)
    w.setframerate(ace.vae.rate)
    w.writeframes(pcm.tobytes())
print("%s: %.1f s of audio in %.1f s (load %.1f s)" % (a.out, audio.shape[1] / ace.vae.rate, t2 - t1, t1 - t0))
