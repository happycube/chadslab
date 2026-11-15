# Audio restoration in a sandbox

Four open-weight restoration models run on narrow-band recordings (cassettes taped off a
telephone line: music to about 3 kHz, hiss above), compared by ear and against clean recordings
of the same pieces. The models' code is the authors' (cloned outside this repository, never
installed); it runs only inside the bubblewrap sandbox (`sbx.sh`, CPU; `sbx_gpu.sh`, the GPU's
device files added): no network, read-only system and venv, one writable `/out`.

| Model | Repository | Weights' license | Use here |
|---|---|---|---|
| RESTORE | MelissaChen15/restore | CC BY-NC 4.0 | best by far against the clean recordings |
| AudioSR (basic) | haoheliu/versatile_audio_super_resolution (its PyPI package) | Apache-2.0 | second |
| A2SB | NVIDIA/diffusion-audio-restoration | NVIDIA OneWay non-commercial | slow, behind AudioSR |
| Apollo | JusperLee/Apollo | CC BY-SA 4.0 | MP3 artefacts: barely changes such recordings |

## Files

- `sbx.sh`, `sbx_gpu.sh VENV OUTDIR [-r HOST:SANDBOX]... -- CMD`: the sandbox (`SBX_ROOT`: the
  directory of the venvs and the sources).
- `run_model.sh MODEL [NAMES]` (in the sandbox; binds `/in`, `/out`, `/drv` = this directory,
  `/repo`, `/models`, `/hf`, `/shims` as the model needs): one model over `/in/*.wav` into
  `/out/MODEL/NAME.wav`, outputs already there skipped.
- `safe_run.py SCRIPT ARGS`: a repository's script with every `torch.load` forced to
  `weights_only=True`; `SAFE_GLOBALS` ("module:Name,...") allowlists a checkpoint's classes
  (RESTORE: `demucs.htdemucs:HTDemucs,fractions:Fraction`; Apollo: omegaconf's containers and
  builtins; the classes `torch.serialization.get_unsafe_globals_in_checkpoint` lists).
- `a2sb_one.py IN OUT [STEPS]`: A2SB's upsampling as its upsample API does it, with the configs
  written to `/tmp` (the repository is read-only): the 99 % rolloff as the cutoff, the two-split
  ensemble, `--model.predict_batch_size` 2 (16 does not fit 24 GB), the log directory in `/tmp`;
  `shims/ssr_eval` stands in for the metrics package A2SB imports but inference never uses.
- `audiosr_one.py IN OUT [STEPS]`: AudioSR in 10.24 s chunks with 1 s crossfades, mono, 48 kHz,
  each chunk's level matched to the input's (AudioSR normalizes each to a 0.5 peak).
- `compare.py`: the listening folder (`RESTORE_IN`, `RESTORE_OUT`, `RESTORE_DEST`): each piece's
  versions as FLAC, an A/B montage of the same 12 s of each, `summary.md` of their bands.
- `truth.py PIECE REFERENCE`: a clean recording aligned to the tape (speed and offset by the
  envelopes, then a straight-line fit of the local offsets) and the log-spectral distance of each
  version from it, by band (`vs reference.md`).
- `dim.py [DB] [HZ]`: RESTORE's output with a high shelf (less brightness).
- `spectrum.py FILES`: the band a file carries.

## Venvs (PyPI only, CUDA torch)

RESTORE and Apollo: Python 3.13 with torch, torchaudio, numpy, scipy, soundfile, soxr, julius,
einops, omegaconf (Apollo also pytorch_lightning, librosa, torch_complex). A2SB: Python 3.10 with
lightning 2.3.3, jsonargparse 4.32.0 (newer releases leave its transforms as dictionaries),
setuptools<81, moviepy<2, rotary_embedding_torch, librosa. AudioSR: Python 3.10 with the audiosr
package, setuptools<81 and matplotlib, and roberta-base's tokenizer files in an HF cache bound at
`/hf` (`HF_HOME=/hf`; the sandbox has no network). Build the venvs on a system Python: the
sandbox does not see a Python installed under a home directory.

One model at a time on a 24 GB GPU (AudioSR beside A2SB runs out of memory).
