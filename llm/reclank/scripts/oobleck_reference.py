"""Oobleck VAE references for scripts/check_acestep.py --parts vae: diffusers'
AutoencoderOobleck (ACE-Step 1.5's vae/, Stable Audio's) in float32 on the CPU, on real
music: the latents of a clip (the encoder's mean) and the decoder's audio of them.

    oobleck_reference.py VAE_DIR AUDIO_FILE out.npz [--start 30] [--seconds 10]

out.npz: "audio_in" (2, samples) the clip at 48 kHz (ffmpeg), "z" (frames, 64) its
latents, "audio" (2, frames 1920) the decoder's output. Run in the generation venv."""
import argparse
import os
import subprocess

import numpy as np
import torch
from diffusers import AutoencoderOobleck

ap = argparse.ArgumentParser()
ap.add_argument("vae")
ap.add_argument("audio")
ap.add_argument("out")
ap.add_argument("--start", type=float, default=30.0)
ap.add_argument("--seconds", type=float, default=10.0)
a = ap.parse_args()
torch.set_num_threads(46)
vae = AutoencoderOobleck.from_pretrained(a.vae, torch_dtype=torch.float32).eval()
rate = vae.config.sampling_rate
env = {k: v for k, v in os.environ.items() if k not in ("OMP_PROC_BIND", "OMP_PLACES")}
pcm = subprocess.run(["ffmpeg", "-v", "quiet", "-ss", str(a.start), "-t", str(a.seconds), "-i", a.audio, "-ac", "2",
                      "-ar", str(rate), "-f", "f32le", "-"], capture_output=True, check=True, env=env).stdout
x = np.frombuffer(pcm, np.float32).reshape(-1, 2).T.copy()
hop = int(np.prod(vae.config.downsampling_ratios))
x = x[:, :x.shape[1] // hop * hop]
with torch.no_grad():
    z = vae.encode(torch.from_numpy(x)[None]).latent_dist.mean
    y = vae.decode(z).sample[0].numpy()
np.savez(a.out, audio_in=x, z=z[0].numpy().T.astype(np.float32), audio=y.astype(np.float32))
print("saved %s: %d frames, audio %s" % (a.out, z.shape[2], y.shape))
