# Video frame interpolation (Practical-RIFE 4.25)

Raise a video's frame rate by making the frames between the originals: Wan's 16 and 24 fps clips
to 48-96 fps, 24 or 30 fps footage to 60, slow motion. reclank/models/vfi (rife.py, video.py,
ref.py); the route of server/generate.py in server/interpolate.py; scripts/interpolate_video.py
from the command line. The plan (with its prototype and measurements, sections 1-7) was approved
with these decisions:

1. The model: Practical-RIFE 4.25 only. FILM (Google, Apache 2.0) was measured and not ported: a
   tie in quality at about 15x the GPU time (sections 4 and 6).
2. The reference: a torch-only float32 reference written from the authors' code
   (scripts/rife_reference.py); the authors' GitHub code is not run.
3. The frame is padded to a multiple of 64 (vsrife's rule; the authors' script pads to 128).
4. Precision: float32 weights and activations as stored. No TF32 or float16 (measured in section
   8, not adopted).
5. Scene cuts: the left frame repeated across a cut; first the combined thumbnail rule (SSIM <
   0.2, or SSIM < 0.45 and a difference ratio above 6), then (approved as a fix) the rule of
   section 5 with RIFE's own residual on small probe frames.
6. Defaults: factor 2, scale auto, H.264 at CRF 16 (NVENC's -cq 16 when the model runs on the
   GPU), yuv420p by the source's matrix and range (section 6.4), the source's audio copied
   unchanged, no frame dedup.
7. The speed work (section 6.3, approved): float32 throughout and the same bits. On the GPU: the
   convs as ENC_CONV_MM (no im2col buffer), each block's input and the output fused (ENC_RIFE_IN,
   ENC_RIFE_OUT), the frames in and out as their bytes with the padding, the thumbnails' inputs and
   the probe frame made on the GPU, and the pipeline's threads and ffmpeg on all the CPUs of the
   start (reclank's OMP_PROC_BIND had put the whole job on one CPU).

## 1. The model

**Practical-RIFE v4.25** (hzwer, MIT; the README's v4.25 zip on Google Drive: flownet.pkl and the
model's code). flownet.pkl loads with `torch.load(weights_only=True)`: 198 float32 tensors, 6.14 M
parameters, of which the inference uses 5.66 M (22.7 MB: the Head 6 K, the blocks 3.0 M, 1.38 M,
0.80 M, 0.38 M, 0.11 M; `teacher` and `caltime` are training-only). scripts/pt_to_safetensors.py
converts it once to flownet.safetensors (the names kept, "module." prefixed); reclank reads only
that.

The network (IFNet_HDv3.py, warplayer.py, inference_video.py; frames RGB in [0, 1]):
- **Head** of each frame: conv 3x3 stride 2 (3 -> 16), two convs 16 -> 16, LeakyReLU 0.2,
  ConvTranspose 4x4 stride 2 (16 -> 4): 4 channels at full resolution. It depends on the frame
  only: once per input frame.
- **Five IFBlocks**, coarse to fine at scales 16, 8, 4, 2, 1 (divided by `scale`: 0.5 for 4K, the
  authors' advice), c = 192, 128, 96, 64, 32: the input bilinearly resized to 1/s
  (align_corners False, no antialiasing), conv 3x3 s2 (c/2), conv 3x3 s2 (c), eight ResConvs
  (LeakyReLU(conv(x) beta + x)), ConvTranspose 4x4 s2 (c -> 52) + PixelShuffle(2): 13 channels at
  1/s resized back: the flow (two 2-D flows, times s), the mask, 8 features. Block 0's input: both
  frames, both heads, t (15 channels); blocks 1-4: both frames and both heads warped by the flow so
  far, t, the mask, the features (24), the flow over s; the flow accumulates.
- **Warp:** `grid_sample(bilinear, border, align_corners=True)` at `linspace(-1, 1) + flow /
  ((n - 1) / 2)`: a backward warp in pixels, the border clamped.
- **Output:** `w0 sigmoid(mask) + w1 (1 - sigmoid(mask))`. Any t is one call: 4x is three calls
  (0.25, 0.5, 0.75), 24 -> 60 fps t in steps of 0.4.
- FLOPs (the convs): the Head 3.0 / 6.3 / 25 GFLOP a frame at 1280x768 / 1920x1088 / 3840x2176;
  the IFNet 37.7 / 80.2 GFLOP an output frame at 720p / 1080p, 80 at 4K with scale 0.5.

## 2. The port

**New records (from 320; stereo holds 300-304, SeedVR2-7B none):**
- **ENC_WARP (320):** a pixel's C values sampled at the pixel plus its flow (two columns of a flow
  buffer), torch's CUDA grid_sampler_2d formulas in order (the grid's linspace values, the flow
  over (n - 1) / 2, unnormalized, clipped, the in-bounds corners summed nw, ne, sw, se).
- **ENC_PRELU (321):** x above 0, else a[col] x: LeakyReLU(0.2) (all slopes 0.2) and PReLU,
  torch's bits.
- **ENC_MIX (322):** a s + b (1 - s), s = sigmoid(m), columns of one buffer.
- **ENC_RESIZE (300, the stereo port's)** gains four trailing operands: the row strides and column
  offsets of its input and output (xs, xo, ys, yo), so it reads and writes the channels of a wider
  buffer (a concatenation without copies). Without them (or with xs = ys = C, xo = yo = 0) it is the
  stereo record, bit for bit (tests/unit/test_vfi.py; test_stereo.py and check_stereo.py's GPU
  checks print the same numbers as before).
- C (interp.c, fp-contract off), CUDA (encoders.cu, _rn intrinsics) and NumPy
  (models/vfi/ref.py) give the same bits, ENC_MIX's exp apart (the C library's expf, CUDA's and
  NumPy's may differ in the last bit).

**The programs** (models/vfi/rife.py; one _Runner, so they share the device copies of the frame
slots):
- ("head", Hp, Wp, shift): slot 1 to slot 0, the new frame's bytes (ENC_UNPACK_U8, / 255) into
  slot 1, its Head.
- ("net", Hp, Wp, scale): the frame at t between slot 0 and slot 1 (t a value of a 28-float vector
  uploaded per call), * 255, ENC_PACK_U8 (round to nearest even, as the reference's rint; the
  authors' script truncates).
- Buffers are pixel rows. A (full resolution, 28 columns: the two warped frames, the two warped
  heads, t, mask, features, flow) is written in place by ENC_WARP and ENC_RESIZE through their
  column offsets; each block's input is ENC_RESIZE'd from it (block 0 from the slots), then
  ENC_MUL_VEC and ENC_BIAS_CLAMP put t into column 14 and divide the flow by s (exact: t survives
  the power-of-2 resize). 3x3 convs are ENC_GATHER im2col tiles + float32 ENC_LINEAR; a
  ConvTranspose(4, 2, 1) is a 3x3 im2col of its input and a linear to its 4 output phases (a
  phase reads kernel rows 3, 1 or 2, 0; zero weights in the other taps), then one ENC_GATHER of the
  phases (with RIFE's PixelShuffle: the rows ordered (phase, sub-pixel, channel)); ResConv's beta
  is folded into its conv. Buffers are pooled by size.
- The frame is zero-padded at the right and bottom to a multiple of max(64, 64 / scale); scale is
  0.5 above 2560 x 1440 of area (auto), else 1.
- On the GPU the convs, the blocks' inputs and the output take the fused records of section 6.3 and
  the frames go in and out as their bytes (the same bits); RECLANK_RIFE_IM2COL=1 keeps the programs
  above.

**The video pipeline** (models/vfi/video.py): ffmpeg decodes with the source's timestamps
(`-vsync 0`: the default doubled a frame at a seek in the tests); a thread makes each frame's
thumbnails and its cut test beside the GPU's work; the Head once per input frame; one IFNet call
per output time; ffmpeg encodes from a pipe at the exact rational rate (60000/1001), the source's
audio copied (AAC when MP4 cannot hold it; cut to the video's length by -t, not -shortest, which
dropped the encoder's delayed frames), its colour tags kept.
- **The output's times:** "factor" F keeps every input frame and puts F - 1 between each pair, the
  last frame repeated F - 1 times so the length (and the audio) stays the input's; "fps" takes the
  frame at each time k / fps (any ratio; an integer ratio becomes the factor's schedule); a
  variable-rate input (r_frame_rate != avg_frame_rate) goes by its packets' timestamps (t the
  fraction between the two neighbours). The input frames are passed through as their bytes.
- **slowmo:** the frames at the input's rate (F times longer), no audio.

**The server** (server/interpolate.py): POST /v1/videos/interpolate, multipart: `video` (a file)
or `video_id` (a completed job of /v1/videos: Wan's, an upscale, a stereo video, another
interpolation; hard-linked into the job, so the source job may go), `factor` (2 to 16) or `fps`
("60", "59.94": the NTSC rates exactly, "60000/1001"), `scene_cuts` (detect | off), `scale` (auto |
1 | 0.5), `slowmo`, `codec` (h264 | hevc), `crf`, `max_frames`. A job of the shared /v1/videos
routes (GET, content, DELETE; ids "interp_..."), in _managers() beside Wan's, the upscales' and the
stereo videos'; each frame takes the server's lock in turn. --rife DIR; /v1/models: "kind":
"interpolate". reclank.setup: "rife-4.25" (manual: the Google Drive zip and the conversion).

## 3. The checks

The reference: scripts/rife_reference.py (torch, float32, CPU; written from the authors' code:
Head, IFBlock, IFNet, warplayer.py's grid and grid_sample, the padding of inference_video.py;
vsrife 5.7.0's packaging of the same model (PyPI, MIT, read) agrees line for line). Real frames:
59.94 fps LaserDisc captures (1440 x 1080, video-originated: every frame a distinct temporal sample;
a hand-held pan, camera shake over leaves, a slow office scene, video effects with trails and
transparency, a scene cut) and two of our own Wan clips (the 5B's 832 x 480 at 24 fps, the A14B's
1280 x 720 at 16 fps). scripts/check_rife.py:

| Check | GPU (and the CPU, the same) |
|---|---|
| block 0's input (ENC_RESIZE of the frames and heads) | 0 difference |
| the heads | 1 - cos 1.2e-14 to 1.8e-14 |
| each block's tmp and flow (picnic, 1440 x 1080; the waves, 1280 x 720) | 1 - cos 2.1e-13 to 2.5e-9 (picnic's flows of up to 7.6 px off by at most 5e-4 px, the waves' 2.7e-3) |
| the output, every frame of 2x and 4x of the six sets (140 frames) | 1 - cos at most 2.2e-10; bytes at most 1 level from the reference's, on 0.004-0.017 % of the values |
| PSNR against the held-out frames | the reference's to 0.01 dB on every set |
| scale 0.5 (padded to 128) | 1 - cos 3.6e-12 / 7.8e-12, at most 1 level |

tests/unit/test_vfi.py (our own Wan frames only): the records C and Python against NumPy (the same
bits), ENC_RESIZE without its strides unchanged, the schedules (factor, 23.976 -> 59.94, a
variable rate), the cut rule, the video plumbing (ffmpeg in and out, counts, rates, audio), the GPU
byte path's host halves (probe_plan's BOX passes against PIL, thumbs_of, mean_abs, the deconv
scatter), the model on the CPU against the reference, and (marked gpu) the GPU's byte path against
its previous programs (with RECLANK_RIFE set).

## 4. Quality (the plan's measurement)

Every other frame dropped and interpolated back (2x: 29.97 -> 59.94), or three of four (4x), on the
real 59.94 fps footage; the output rounded to bytes; PSNR over a set's frames, the mean |difference|
in levels, the share of pixels more than 8 levels off (analog noise caps all of them):

| set | RIFE 2x | FILM 2x | blend 2x | repeat 2x | RIFE 4x | FILM 4x (recursion) |
|---|---|---|---|---|---|---|
| pan (fast) | 28.95 dB, 4.8 | 29.15 dB, 4.5 | 25.46 dB | 22.38 dB | 27.88 dB | 28.81 dB |
| shake, leaves | 28.86 dB, 5.3 | 28.99 dB, 5.2 | 25.95 dB | 21.95 dB | 27.35 dB | 27.37 dB |
| office (slow) | 35.68 dB, 3.1 | 35.70 dB, 3.1 | 35.42 dB | 32.72 dB | 34.82 dB | 34.93 dB |
| effects (hard) | 26.80 dB, 7.0 | 26.52 dB, 7.3 | 27.02 dB | 25.48 dB | 27.08 dB | 26.71 dB |
| Wan 5B fox (24 -> 12 -> 24) | 36.40 dB | 36.66 dB | 35.65 dB | 28.50 dB | | |
| Wan A14B waves (16 -> 8 -> 16) | 29.66 dB | 29.33 dB | 28.48 dB | 27.05 dB | | |

RIFE and FILM tie (within 0.4 dB at 2x); both gain 3-3.5 dB over the blend on camera motion; on
video effects (glows, trails) neither is a flow and both fall a little below the blend without
breaking. FILM (TF Hub's google/film/1, byte-identical to the Style training) was run as Google's
own SavedModel in a sandbox; its time input works but direct t is 1.1-2.9 dB worse than its
midpoint recursion, so it only does 2^k. Across a scene cut both give 13-17 dB against 23 dB for
repeating the left frame: section 5.

## 5. Scene cuts

**The rule** (video.CutDetector; the user approved this fix of the first rule, which missed 5 of 26
cuts on new footage and repeated frames falsely in effects and animation). RIFE itself tells
whether a flow explains a pair: each input frame is also resized to a probe frame (its long side
320, PIL's BOX, on the decoder's thread), and a small RIFE call (RIFE.prober(), at t 0.5) gives

    R = the mean |w0 - w1| of the two probe frames warped to the middle (levels of 255)
    D = the mean |x0 - x1| of the probe frames themselves
    Rr = R / max(the median R of the last 8 moving pairs (D > 2), 2)   (the residual's jump)
    old = SSIM < 0.2, or SSIM < 0.45 and the grey thumbnails' difference ratio > 6 (the first rule)
    fire = (Rr > 4.5 and R > 6 and SSIM < 0.75) or (old and Rr > 2.5)
    cut = fire, unless one of the 2 pairs before fired (a run of fires keeps only its first)

A cut has no flow that explains it: R jumps against the pairs before (a pan, however fast, keeps R
low; a swirl or an animation's churn keeps it high but steady). The moving-pairs baseline keeps
the held frames of limited animation out of it; the run rule stops an animation's or an effect's
churn after one repeat. On a cut the frames between the pair repeat its left frame (the cut stays at
the right frame's time). The cost: one IFNet call a pair at 320 x 256 (padded), about 5 ms of the
GPU (its records are small and latency-bound), and its host work. As first shipped the probe frames
and the residual's means ran on the host and slowed a whole job by about a fifth (a 1440 x 1080 2x
job 20.6 -> 15.4 output frames a second, 1280 x 720 30.9 -> 22.6; section 6.1); with section 6.3's
GPU path the probe frame is made on the GPU and its residual runs beside the pair's first frame.

**The study** (the scripts are outside the repository): 9 stretches of 59.94 fps captures, 1480 s,
89k frames: the first rule's tuning set (2 x 240 s of live action with 27 cuts, 100 s of video
effects), three more live-action stretches of 150 s (26 cuts), 150 s more of the effects, 150 s of
a product disc (montages, whip pans, zooms, graphics) and 150 s of an educational disc (limited
animation, time-lapse). Every pair a 2x job sees (every other frame, 29.97 fps in) and a 4x job
(every 4th, 14.99 fps in): 66,500 pairs. The labels: the 59.94 fps frame difference's spikes on
the live action; on the product and the educational discs (whose spikes are mostly an animation's
new drawings) every pair either the first or the new rule fires on was viewed and labelled by eye,
and the remaining spike pairs were viewed too (no cut among them); ambiguous pairs (whip-pan blurs,
zoom changes, a dissolve, a fade) are counted apart. The thresholds were fit on the first half of
each stretch and checked on the second, and the other way round: the two halves chose values a grid
step apart (Rr 5 and 4, SSIM 0.8 and 0.7, the second test's Rr 2 and 3, R 6 both), and the rule takes
the midpoints; the run rule's window (2) was chosen the same way (1 / 2 cuts lost, false cuts halved,
on either half).

| Rule | 2x found | 2x false | 4x found | 4x false |
|---|---|---|---|---|
| the first rule (old) | 101 of 107 | 46 | 97 of 102 | 39 |
| the looser option (old, or SSIM < 0.8 and ratio > 10) | 107 of 107 | 53 and more (92 fires unreviewed) | 100 of 102 | 66 |
| **the new rule** | **103 of 107** | **16** | **102 of 102** | **4** |
| new, by half (first / second) | 41 of 43 / 62 of 64 | 11 / 5 | 43 of 43 / 59 of 59 | 1 / 3 |

All of these take both halves. The halves alone, each scored only with what was fit without it:
the first rule and the looser option have no fitted values (the same halves as above); for the new
rule each half is scored with the other half's own fit (the first half with Rr 4, SSIM 0.7, the
second test's Rr 3; the second half with Rr 5, SSIM 0.8, Rr 2; R > 6 and the run rule both), not the
midpoints, which saw both halves. Found / cuts, false cuts, ambiguous pairs (counted apart, not as
false; the cross-fitted rule's two fires on pairs the review had not seen were viewed: one a
composite transition, ambiguous, one solar time-lapse, false):

| Rule (half 1 / half 2) | 2x found | 2x false | 2x ambiguous | 4x found | 4x false | 4x ambiguous |
|---|---|---|---|---|---|---|
| the first rule | 40 of 43 / 61 of 64 | 21 / 25 | 2 / 6 | 40 of 43 / 57 of 59 | 11 / 28 | 0 / 7 |
| the looser option | 43 of 43 / 64 of 64 | 27 / 26 (84 / 8 fires unreviewed) | 2 / 6 | 43 of 43 / 57 of 59 | 25 / 41 (1 / 7 unreviewed) | 0 / 9 |
| new, cross-fitted | 40 of 43 / 62 of 64 | 11 / 5 | 1 / 2 | 39 of 43 / 57 of 59 | 4 / 3 | 1 / 3 |
| new, midpoints (shipped) | 41 of 43 / 62 of 64 | 11 / 5 | 1 / 2 | 43 of 43 / 59 of 59 | 1 / 3 | 0 / 4 |

Cross-fitted, the new rule finds 102 of 107 at 2x (16 false) and 96 of 102 at 4x (7 false). The
cuts it loses against the midpoints: the second half's SSIM 0.7 misses the first half's monitor
insets (SSIM 0.73-0.74) and an animation cut (Rr 2.65 under 3), and its ambiguous fire's run window
hides a product-disc cut 2 pairs later; the first half's Rr 5 misses two live-action cuts of the
second (Rr 4.64, 4.98). The midpoints sit between the two fits and lose none of these; the halves'
fits are a grid step apart, so the held-out numbers bound how far a nearby threshold moves the
counts.

The first rule's false cuts were mostly the educational disc's animation and the product disc's
silhouette pan; on the live action alone it found 48 of 53 at 2x (0 false) and 48 of 53 at 4x (12
false: the swirl and a koala's camera shake). The new rule on the live action: 52 of 53 at 2x (0
false), 53 of 53 at 4x (3 false). Every remaining miss and false cut was viewed: the misses are an
inset that appears on a black monitor screen (SSIM 0.76, just above 0.75) and three on the
educational disc (two flash frames, whose cut back comes within the run window, and a cut right
after a run of false fires); the false cuts are a frantic stretch of limited animation (10 at 2x),
solar time-lapse with exposure jumps (3), zooms (3), a whip pan, a silhouette pan, the swirl, an
atom zoom. The production code reproduces the study (the first minute of the first stretch: 6 of 6
cuts, 0 false, at 2x and 4x; a 29.97 fps clip with 3 cuts: the 3).

## 6. Cost

### 6.1 The 3090

scripts/check_rife.py time (one push and one frame a call, the frame's bytes downloaded; the median
of 10). Before the speed work (section 6.3), then with it (the same bits):

| Size | Head (an input frame) | IFNet (an output frame) | of the IFNet by record, with the speed work (ms) |
|---|---|---|---|
| 1280 x 720 (padded 768) | 3.1 -> 1.7 ms | 40.0 -> 12.8 ms | ENC_CONV_MM 9.6, ENC_RIFE_IN 1.9, ENC_RIFE_OUT 0.3 |
| 1920 x 1080 (padded 1088) | 5.6 -> 3.6 ms | 81.1 -> 24.3 ms | ENC_CONV_MM 16.3, ENC_RIFE_IN 4.0, ENC_RIFE_OUT 0.7 |
| 3840 x 2160 (scale 0.5, padded 2176) | 22.3 -> 14.1 ms | 191.1 -> 35.4 ms | ENC_CONV_MM 17.5, ENC_RIFE_IN 6.3, ENC_RIFE_OUT 1.8 |

End to end (scripts/interpolate_video.py, NVENC -cq 16, output frames a second; one quiet session,
real footage: a 1440 x 1080 clip at 29.97 fps with 3 cuts, 240 frames; the first 10 s of the
first study stretch cropped to 16:9 and scaled to 1280 x 720 (300 frames, a cut) and, for timing
only, to 3840 x 2160 (5 s); every run found the same cuts):

| Job | master (the first cut rule) | the cut fix (the probe on the host) | the speed tree, RECLANK_RIFE_IM2COL=1 (the previous GPU programs and host path) | the speed work |
|---|---|---|---|---|
| 1440 x 1080, x2 | 20.6 | 15.4 | 21.8 | **58.9** |
| 1280 x 720, x2 | 30.9 | 22.6 | 32.8 | **83.7** |
| 1440 x 1080, x4 | 19.8 | 17.1 | 20.5 | **56.0** |
| 3840 x 2160, x2 | 4.2 | 3.7 | 6.8 | **17.9** |

The server end to end (its jobs one after another; the cut fix -> the speed work): the 1440 x 1080
clip doubled 15.3 -> 58.1 frames a second; Wan A14B's 1280 x 720 clip (33 frames at 16 fps, the
programs' builds included) x4 19.3 -> 46.2 and x2 22.3 -> 62.5; the Wan fox (832 x 480) to 60 fps
20.7 -> 34.2. The card's peak during those jobs 3.5 -> 1.9 GiB (no im2col tiles; the full-resolution
buffer gone). The earlier figures of this section (16.3 and 15.4 frames a second for the 1440 x 1080
clip, before the cut fix) were measured on a busier machine; the master column is this session's.

At 4K the job is bound by the encoder: ffmpeg 4.4 converts the RGB frames to yuv420p for NVENC on
one thread (a decode piped into that encode alone runs at 17.4 frames a second; with yuv420p in, 47);
to the null muxer the speed work's 4K job runs at 23 frames a second. libx264 (CRF 16, preset medium,
8 threads; the CPU's encoder) encodes 1440 x 1080 at 3.8 frames a second on 12 slow cores, so a
libx264 job is bound by it at about that rate.

### 6.2 The CPU

On 12 threads (a quiet machine; scripts/check_rife.py time, the median of 5):

| Size | Head | IFNet (an output frame) | 2x output frames a second |
|---|---|---|---|
| 1280 x 720 | 47 ms | 350 ms | ~5 |
| 1920 x 1080 | 100 ms | 751 ms | ~2.4 |

The Wan fox (832 x 480, 17 frames at 24 fps) doubled by scripts/interpolate_video.py --cpu in 4.3 s
(8.0 output frames a second, libx264). The torch reference (oneDNN, the same 12 threads) takes
0.61 s and 1.35 s an output frame at 720p and 1080p: reclank's CPU path is faster on these narrow
convs. Under another job's load on the same cores the IFNet took 1.6-2.6 s: CPU timings vary with
the machine. FILM on the CPU (its SavedModel in TensorFlow) took 10.9 s a call at 1440 x 1080.

### 6.3 The speed work (done: float32, the same bits)

Before it, at 1080p the IFNet's GPU time was the im2col gathers (19.8 ms) as much as the products
(19.4 ms; RIFE's convs are narrow, 16-192 channels), then the resizes (10.6 ms) and warps (5.7 ms);
end to end the GPU was busy a third of the time. In order of what it gave:

- **ENC_CONV_MM (323, GPU):** a conv as k_enc_sgemm reading its im2col rows through the index (no
  tile buffer: 3.2 GB a frame at 1080p), the accumulation in the same order (fmaf over k from 0, then
  the bias), its epilogue the ResConv's residual and LeakyReLU (ENC_ADD's and ENC_PRELU's bits), and a
  scatter of its rows of blk values (the inverse of the ENC_GATHER of a deconv's phases and RIFE's
  PixelShuffle: each value to its pixel, no gather and no copy). Column tiles of 16 to 128 to fit the
  narrow outputs. The IFNet at 1080p 81 -> 43 ms.
- **ENC_RIFE_IN (324), ENC_RIFE_OUT (325) (GPU):** no full-resolution buffer A. Each block's input
  pixel is made at its own resolution from the frames, the heads and the earlier blocks' tmp (kept at
  their resolutions): the flow's resize and sum, the two warps, the mask and features, ENC_RESIZE's
  taps at 1 / s, times the vector and plus t, in the records' order; the output (and a probe's warped
  frames) the same way. 43 -> 30 ms; with the scatter 24.3 ms.
- **The frames as bytes on the GPU (RIFE.gpu_io):** the decoder's thread reads each frame into a
  pinned slot, uploaded as it is; ENC_UNPACK_U8P (326) unpacks it into the padded slot (zeros past
  it), ENC_PACK_U8C (327) packs the h x w output into a pinned slot the writer's thread hands to
  ffmpeg. With the cut test: ENC_THUMB_U8 (329) gathers the thumbnails' taps and sums the grey
  blocks (integers), and ENC_RESAMPLE_U8 (328) makes the probe frame by PIL's BOX passes (Pillow's
  Resample.c: its bounds, its 22-bit integer coefficients, horizontal then vertical, the same bytes as
  Image.resize on every frame checked: 1440 x 1080, 1280 x 720, 3840 x 2160, 832 x 480, portrait and
  odd crops); the host finishes the thumbnails from those (thumbs_of: thumbs()'s arithmetic). The
  probe's means take an exact fast form (rife.mean_abs: NumPy's reduction of 3 values adds (d0 + d1)
  + d2; the same bits on 9 million real values), its D from the probe frames on the host, and the
  pair's first frame is made while the host runs the cut test (dropped on a cut).
- **Affinity:** reclank sets OMP_PROC_BIND, so OpenMP binds the main thread to one CPU; threads made
  after it and their children inherit that CPU, and libgomp in ffmpeg binds again. In a job the
  decoder's and writer's threads and both ffmpeg processes all ran on CPU 2 (taskset -p of each
  thread and child during a run): a 1440 x 1080 job at a third of its speed. video.py now starts
  ffmpeg on the CPUs of the start (RECLANK_START_CPUS) without OMP_PROC_BIND and OMP_PLACES, and the
  two threads widen themselves to them (with the byte path
  already in, the 1440 x 1080 x2 job 27 -> 46 frames a second by this fix). The pipes take 1 MiB.
  These helpers are now models/media.py (popen, run, widen, pipe_size), shared with the stereo
  video, Wan's writer and the video upscales, which lost the same speed (GENERATION_PLAN).

The checks: the output frames' bytes of the speed work against the previous programs and host path
(RECLANK_RIFE_IM2COL=1) are the same (the md5 of a whole job's raw frames: the 1440 x 1080 clip x2 and
x4, the 1280 x 720 clip x2); f0, f1 and the float32 output the same values on five pairs of the real
sets; the probe's residuals the same bits on 60 real pairs; the thumbnails and probe frames the same
bytes on four clips as is, transposed and cropped (scripts in the speed work's notes;
tests/unit/test_vfi.py on our own frames). Against the reference: the same numbers as section 3 on
all six sets (2x, 4x, scale 0.5).

Not taken: the Head's two slot copies (x0 <- x1, f0 <- f1: 0.4 ms at 1080p, 1.5 ms at 4K) would go
with ping-ponged slots, but that doubles the net programs and their memory; the probe's IFNet (5 ms
a pair at 320 x 256) is bound by its small records' latency; split-K for small convs would change
the bits. At 1080p the GPU is now busy about three quarters of a job; at 4K the encoder bounds it.

**The same conv elsewhere (measured, not changed):** the other models' 3 x 3 convs (ENC_GATHER +
ENC_LINEAR of float32 W) rewritten to ENC_CONV_MM at compile time, on real inputs: DA3Mono's head
(the bridge at 504) the same bits, 0.104 -> 0.094 s; Video Depth Anything Small (96 frames of 1440 x
1080) the same bits, 6.1 -> 6.8 frames a second; Real-ESRGAN x4plus (640 x 400) the same bits, 5.1 ->
2.0 s. SeedVR2's VAE gave wrong output under the rewrite: its arena gives a conv's output the memory
of its input once the gather has read it, so a switch there needs the arena's liveness changed too.

### 6.4 The encoder's colour

ffmpeg decodes the source to RGB by its tags (the scale filter's matrix and range "auto"). The
encode converted the RGB back to yuv420p by BT.601, limited range, whatever the tags it copied said:
a BT.709 source came out BT.601 and tagged bt709 (decoded by its tags, colours off by 0.9 levels on
average, up to 14-18), a full-range BT.709 source (phones) limited and tagged pc (decoded by its
tags 28.8 dB against the frames sent, 8 levels off on average: washed out); an untagged source came
out consistent but untagged. Now (media.yuv_colors, both encoders) the scale filter takes the
source's matrix (bt709, BT.601 for smpte170m and bt470bg, bt2020, fcc, smpte240m) and range for
its out_color_matrix and out_range, and a source without a matrix tag (or one the filter has none
for) is converted BT.601 limited as before and tagged smpte170m / tv. Chroma siting is left (MPEG-2)
in every case. Each job's output decoded by its tags against the frames it sent (60 frames;
NVENC, PSNR):

| Source | before | now | NVENC's own conversion (rgb0) |
|---|---|---|---|
| untagged, 1280 x 720 | 36.84 dB (untagged) | 36.84 dB (tagged smpte170m, tv) | 36.84 dB |
| BT.709, 1280 x 720 | 36.47 dB | 36.64 dB | 36.66 dB |
| BT.709 full range, 1280 x 720 | 28.79 dB | 36.94 dB | 36.96 dB |

Now the tags' decode is the best of the BT.601 / BT.709 decodes on every source, NVENC and
libx264: untagged 1440 x 1080 37.19 dB (libx264 40.13), BT.709 1440 x 1080 36.74 (39.56; its BT.601
decode 36.32, 38.74), a BT.601-tagged capture 42.33 (41.10), 3840 x 2160 42.86. The jobs' speed is
the same (1440 x 1080 x2 59.5 -> 59.2, 1280 x 720 85.1 -> 83.8, 4K 18.1 -> 18.0 frames a second).

Measured and not taken: RGB plus a zero byte straight into NVENC (rgb0, its own conversion by the
tags; untagged it converts BT.709 and leaves the output untagged, so it needs the smpte170m tag). The
encode alone is faster (1440 x 1080 78 -> 106 frames a second, 1280 x 720 104 -> 138, 4K 16 -> 20),
but whole jobs were 2-5 % slower with the rgb0 frames packed on the GPU (1440 x 1080 x2 58.9 -> 56.6,
4K 18.0 -> 17.0; packed on the host, 46.8 and 8.2): at 1080p the encoder is not the bound, and at
4K ffmpeg 4.4's one-thread copy of each 33 MB frame into NVENC takes the conversion's place. Its
colours equal the fixed path's.

## 7. Pipelines

The order for a chain: **Wan -> SeedVR2 (upscale) -> RIFE (interpolate) -> stereo**, each a job,
chained by `video_id`. Upscale before interpolating (SeedVR2's cost is per frame and large; RIFE's
in-betweens come from the sharp frames); interpolate before stereo (the depth at the final rate;
interpolating a side-by-side frame would put a seam in the middle and let the eyes differ).

## 8. TF32 and float16 (measured, not adopted)

Emulated in the torch reference (both inputs of every conv rounded to TF32, cvt.rna.tf32's
round-to-nearest with ties away from zero; or to float16), the sums in float32: against the held-out
frames the same PSNR to 0.01 dB; against float32's output 62-77 dB, 0.1-2 % of the values a level
or more off (a few pixels by up to 47 levels at an occlusion). On GA102 dense TF32 has float32
FFMA's peak (35.6 TFLOPS): the gain would come from a better kernel for RIFE's narrow shapes, which
the float32 implicit-GEMM conv of 6.3 gets at float32's precision.
