"""AudioSR (basic) over one file in 10.24 s chunks (1 s crossfades), mono, 48 kHz out; each chunk's level matched
to the input's (AudioSR normalizes every chunk to a 0.5 peak). Checkpoint from /models, weights_only.
    python audiosr_one.py IN.wav OUT.wav [DDIM_STEPS]"""
import os, sys
import numpy as np, soundfile as sf, torch

_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": True})
import torchaudio  # noqa: E402


def _ta_load(f, *a, **k):          # torchaudio's own loader now needs torchcodec
    w, r = sf.read(f, dtype="float32", always_2d=True)
    return torch.from_numpy(w.T.copy()), r


torchaudio.load = _ta_load
import audiosr.pipeline as ap  # noqa: E402
ap.download_checkpoint = lambda name="basic": "/models/pytorch_model.bin"

src, dst = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 50
dev = "cuda" if os.environ.get("DEV", "cuda") == "cuda" else "cpu"
x, sr = sf.read(src, dtype="float32", always_2d=True)
x = x.mean(1)
model = ap.build_model(model_name="basic", device=dev)
SR_OUT, CH, OV = 48000, 10.24, 1.0
step = CH - OV
starts = np.arange(0, max(len(x) / sr - OV, 1e-3), step)
out = np.zeros(int(len(x) / sr * SR_OUT) + SR_OUT)
wsum = np.zeros_like(out)
os.makedirs("/tmp/asr", exist_ok=True)
for k, t in enumerate(starts):
    a, b = int(t * sr), min(len(x), int((t + CH) * sr))
    seg = x[a:b]
    sf.write("/tmp/asr/c.wav", seg, sr)
    y = np.asarray(ap.super_resolution(model, "/tmp/asr/c.wav", seed=42, ddim_steps=steps, guidance_scale=3.5)).reshape(-1)
    n = int(len(seg) / sr * SR_OUT)
    y = y[:n]
    rin, rout = np.sqrt(np.mean(seg ** 2)) + 1e-9, np.sqrt(np.mean(y ** 2)) + 1e-9
    y = y * (rin / rout)
    w = np.ones(len(y))
    f = int(OV * SR_OUT)
    if k > 0:
        w[:f] = np.linspace(0, 1, f)
    if b < len(x):
        w[-f:] = np.minimum(w[-f:], np.linspace(1, 0, f))
    o = int(t * SR_OUT)
    out[o:o + len(y)] += y * w
    wsum[o:o + len(y)] += w
    print("chunk %d/%d" % (k + 1, len(starts)), flush=True)
n = int(len(x) / sr * SR_OUT)
y = out[:n] / np.maximum(wsum[:n], 1e-6)
sf.write(dst, y.astype(np.float32), SR_OUT, subtype="FLOAT")
print("wrote %s (%.1f s at %d Hz, %d steps)" % (dst, n / SR_OUT, SR_OUT, steps))
