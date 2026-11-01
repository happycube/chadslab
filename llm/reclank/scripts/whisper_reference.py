"""Whisper references for scripts/check_whisper.py: transformers' WhisperFeatureExtractor
and the encoder in float32 on the CPU (torch), for clips of real audio.

    whisper_reference.py MODEL_DIR clips.npz refs.npz

clips.npz: "names", and "c<i>" the 16 kHz mono float32 samples of clip i (at most 30 s).
refs.npz: "f<i>" the input features (128, 3000), "e<i>" the encoder's last hidden state
(1500, 1280). Run in a venv with torch and transformers (the MuQ venv: 4.48.3)."""
import sys

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

path, src, out = sys.argv[1:4]
torch.set_num_threads(16)
fe = WhisperFeatureExtractor.from_pretrained(path)
enc = WhisperModel.from_pretrained(path, torch_dtype=torch.float32).encoder.eval()
d = np.load(src)
res = {"names": d["names"]}
for i in range(len(d["names"])):
    x = d["c%d" % i]
    f = fe(x, sampling_rate=16000, return_tensors="np").input_features[0]
    with torch.no_grad():
        e = enc(torch.from_numpy(f)[None]).last_hidden_state[0].numpy()
    res["f%d" % i], res["e%d" % i] = f.astype(np.float32), e.astype(np.float32)
    print(i, d["names"][i], f.shape, e.shape, flush=True)
np.savez(out, **res)
