# FlashVSR: video upscaling

reclank runs FlashVSR v1.1 (OpenImagingLab/FlashVSR, Apache 2.0). FlashVSR is a one-step diffusion model for video
super-resolution, and reclank implements its Tiny-Long streaming pipeline without torch. The pipeline has three parts:

- the LQ projector;
- the DiT (Wan 2.1 1.3B, one denoising step);
- the TCDecoder, plus a colour fix at the end.

The code is in `reclank/models/flashvsr`. How the port was built and checked is in `docs/GENERATION_PLAN.md`
(the FlashVSR entry).

FlashVSR is a Python API. The generation server has no route for it.

## What you need

- **A 24 GB NVIDIA GPU for x 4 from 640 x 480.** At that size (2560 x 1920 out) the card's peak is about 23.3 GB.
  Smaller outputs need less. The self-attention runs on the tensor cores on Ampere and later (sm_80, 128-wide
  heads). Other GPUs take the float32 kernel automatically.
- **Host RAM.** At 2560 x 1920 the DiT's key/value ring takes about 28 GB of pinned host memory, and the frames in
  and out stay in RAM (14.7 MB an output frame at 2560 x 1920). A run of 49 frames at x 4 peaked at 54 GB of RSS.
- **ffmpeg** to read and write video (the example below), and Pillow.
- **The CPU path** (`gpu=False`) runs the same model in float32. It is meant for checks; for real video it is far
  too slow.

## Weights

Download `JunhaoZhuang/FlashVSR-v1.1` from Hugging Face. reclank reads safetensors only, so convert the three
PyTorch files once, in a venv that has torch and safetensors:

    python scripts/pt_to_safetensors.py FlashVSR-v1.1/LQ_proj_in.ckpt FlashVSR-v1.1/TCDecoder.ckpt \
        FlashVSR-v1.1/posi_prompt.pth

That writes `LQ_proj_in.safetensors`, `TCDecoder.safetensors` and `posi_prompt.safetensors` beside them. The
folder then needs these files:

| File | What it is |
|---|---|
| `diffusion_pytorch_model_streaming_dmd.safetensors` | the DiT (as downloaded) |
| `LQ_proj_in.safetensors` | the LQ projector |
| `TCDecoder.safetensors` | the decoder |
| `posi_prompt.safetensors` | the fixed prompt's embedding |

`Wan2.1_VAE.pth` is not used.

## Use

`FlashVSR(path, gpu=True)` loads the model. Its `upscale_video(frames, (H, W))` takes and returns video as arrays:

- **Input:** `frames` is an array of shape `(t, h, w, 3)`, `uint8` RGB.
- **Output size:** `(H, W)` must both be multiples of 128.
- **Output:** an array of shape `(t, H, W, 3)`, `uint8` RGB, with as many frames as the input.

This example upscales a video x 4 with ffmpeg on both ends. Run it from the repository root with
`OMP_NUM_THREADS=1` (as every GPU run):

```python
import subprocess
import sys

import numpy as np

from reclank.models.flashvsr.pipeline import FlashVSR

src, dst, weights = sys.argv[1:4]
w, h = 640, 480                       # the input frames' size (scaled by ffmpeg)
W, H = 4 * w, 4 * h                   # the output: x 4, both multiples of 128

raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vf", f"scale={w}:{h}",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)

m = FlashVSR(weights, gpu=True)
out = m.upscale_video(frames, (H, W), seed=0,
                      progress=lambda stage, done, n: print(f"chunk {done}/{n}", flush=True))
print({k: round(v, 1) for k, v in m.times.items()})     # seconds by stage
m.release()

enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                        "-r", "60000/1001", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", dst],
                       stdin=subprocess.PIPE)
enc.stdin.write(out.tobytes())
enc.stdin.close()
enc.wait()
```

Set `-r` to the source's frame rate. Audio is not copied: mux it back from the source with ffmpeg if you need it.

The other arguments of `upscale_video`:

- **`seed`:** picks the noise. The same seed and input give the same output.
- **`color_fix=False`:** skips the AdaIN colour fix. Normally each output frame takes its LQ frame's per-channel mean
  and spread.
- **`progress`:** a callback, called after each chunk.

## Sizes

- **Scale x 4.** That is what the model is trained for. An x 2 run on a real scene showed artifacts.
- **Inputs whose x 4 is not a multiple of 128.** Scale the input first so that it is, as the example does. Or
  upscale to the next multiples of 128 and crop or resize the output. The pipeline itself resizes each input frame
  to `H x W` with PIL's bicubic before the model sees it.
- **Length.** The model takes at least 25 frames, then 8 at a time. The end is padded by repeating the last frame,
  and the output is cut back to the input's length.

## Long videos

Each call holds its whole input and output in RAM, so split long videos into segments of a few hundred to a few
thousand frames. The splits are not free:

- **The model's state starts over.** It carries temporal state from chunk to chunk, and each call starts it again.
  Split at scene cuts, where nothing carries over anyway.
- **Each call's first chunk is slow.** It takes about 60 s at 2560 x 1920, because it pins the ring and packs the
  weights. Keep the `FlashVSR` object between calls, and keep segments long.

## Speed

On an RTX 3090 (at a 220 W power limit), 640 x 480 to 2560 x 1920:

- **Steady chunks:** about 1.6 s a frame. The DiT is about 2/3 of that, the decoder most of the rest.
- **A scene of 596 frames:** 985 s.
- **The example above on 49 frames:** 163 s, including the model's load and the first chunk.
- **Float32 products** (`RECLANK_FLASHVSR_TC=0`): about 5.1 s a frame.

## Settings

- **`RECLANK_FLASHVSR_TC`**: which parts run in float16 on the tensor cores.
  - `1` (the default): all of them. The products are float16 with float32 sums.
  - `0`: float32 everywhere.
  - A comma list picks parts from `dit`, `attn`, `lq` and `dec`.
  - The difference is small. Against the official float32 pipeline, the default output is 57.2 dB. The official
    pipeline's own bfloat16 is 28.1 dB from its float32.
- **`RECLANK_FLASHVSR_ROWS`**: the DiT's row tile (8192). A smaller one uses less GPU memory.
- **`FlashVSR(path, gpu=True, tc=...)`** sets the tensor-core parts in code (`True`, `False`, or a list), the same
  as `RECLANK_FLASHVSR_TC`.

## Checks

`scripts/check_flashvsr.py MODEL_DIR DUMP.npz [--gpu] [all] [lq] [dit] [tc]` compares each part, and the whole
pipeline, against a dump of the official pipeline run in float32. Each part's result is an SNR in dB.
`tests/unit/test_flashvsr.py` holds the unit tests.
