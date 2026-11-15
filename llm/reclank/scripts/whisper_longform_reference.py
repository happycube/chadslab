"""Whisper long-form references for scripts/check_whisper_longform.py: transformers'
generate of whole clips of real audio (more than 30 s: sequential long-form with
timestamps), greedy, in float32 on the CPU (torch).

    whisper_longform_reference.py MODEL_DIR long.npz out.npz [--lang sk | --lang none]
        [--task transcribe] [--prompt TEXT] [--only I] [--no-timestamps]

long.npz: "names", "c<i>" the 16 kHz mono float32 samples of clip i. out.npz: "f<i>"
the input features (WhisperFeatureExtractor of the whole clip: truncation=False,
padding="longest"; at most 30 s: padded to 30 s), "t<i>" generate's sequences (the
segments' tokens), "s<i>" (k, 2) the segments' start and end, "k<i>" each segment's number of tokens, "text<i>",
"lang<i>" (the detected language with --lang none), "secs<i>" generate's time;
"texts", "p<i>" the tokenizer's get_prompt_ids of a few texts (the check of
Tokens.encode), and "prompt" (--prompt: its ids passed as prompt_ids). Run in the
MuQ venv (transformers 4.48.3)."""
import argparse
import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TEXTS = ["Rádio Regina, Nočné tónovanie.", "Ďakujem za pozornosť. Dobrý deň, vážení poslucháči!",
         "Hosť Panorámy: Peter Sagan, 3 krajiny a 1 000 km.", "We're here; it's 10:30 -- don't  go...",
         "  dva   medzery\na nový riadok\t(tab) ", "Šťastie, ľúbezné čaro, ôsmy, äno, dĺžeň 12345 €",
         "Folklór 'nám' pristane... ”Áno“ – „nie“"]

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("clips")
ap.add_argument("out")
ap.add_argument("--lang", default="sk")
ap.add_argument("--task", default="transcribe")
ap.add_argument("--prompt", default=None)
ap.add_argument("--only", type=int, default=None)
ap.add_argument("--no-timestamps", action="store_true", help="return_timestamps False (clips of at most 30 s)")
ap.add_argument("--threads", type=int, default=12)
a = ap.parse_args()
torch.set_num_threads(a.threads)
proc = WhisperProcessor.from_pretrained(a.model)
model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32).eval()
tok = proc.tokenizer
d = np.load(a.clips)
res = {"names": d["names"], "texts": np.array(TEXTS)}
for i, t in enumerate(TEXTS):
    res["p%d" % i] = tok.get_prompt_ids(t).astype(np.int32)
prompt_ids = None
if a.prompt:
    prompt_ids = torch.from_numpy(tok.get_prompt_ids(a.prompt).astype(np.int64))
    res["prompt"] = prompt_ids.numpy().astype(np.int32)
lang = None if a.lang == "none" else a.lang
for i in range(len(d["names"])):
    if a.only is not None and i != a.only:
        continue
    x = d["c%d" % i]
    if len(x) <= 480000:        # the usual features of at most 30 s: padded to 30 s
        fe = proc.feature_extractor(x, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    else:
        fe = proc.feature_extractor(x, sampling_rate=16000, return_tensors="pt", truncation=False,
                                    padding="longest", return_attention_mask=True)
    f = fe.input_features
    t0 = time.time()
    with torch.no_grad():
        if lang is None:
            res["lang%d" % i] = np.array(tok.convert_ids_to_tokens(int(model.detect_language(f)[0]))[2:-2])
        g = model.generate(f, attention_mask=fe.attention_mask, language=lang, task=a.task,
                           return_timestamps=not a.no_timestamps, return_segments=True, prompt_ids=prompt_ids)
    secs = time.time() - t0
    segs = g["segments"][0]
    seq = np.concatenate([s["tokens"].numpy() for s in segs]).astype(np.int32)
    text = tok.decode(seq, skip_special_tokens=True)
    res["f%d" % i], res["t%d" % i] = f[0].numpy().astype(np.float32), seq
    res["s%d" % i] = np.array([[float(s["start"]), float(s["end"])] for s in segs], np.float64)
    res["k%d" % i] = np.array([len(s["tokens"]) for s in segs], np.int32)
    res["text%d" % i], res["secs%d" % i] = np.array(text), np.array(secs)
    print(i, d["names"][i], f.shape, len(segs), "segments", len(seq), "tokens, %.0f s" % secs, flush=True)
    print("   ", repr(text[:300]), flush=True)
np.savez(a.out, **res)
