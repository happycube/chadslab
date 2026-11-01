# 2D to stereo 3D for VR players (images and video)

A depth model on each image or frame, the disparity, both eyes by a forward
warp with z-order and hole filling, packed side by side or over-under for a
VR player's flat-screen 3D mode. reclank/models/depth; the routes of
server/generate.py in server/stereo.py; scripts/stereo_convert.py from the
command line. The plan (with its prototype and measurements) was approved
with these decisions:

1. The reference: a torch-only float32 reference written from the authors'
   code (scripts/stereo_reference.py); the authors' GitHub code is not run.
2. The models: MoGe-2 ViT-L for images, Video-Depth-Anything-Small for
   video, DA3MONO-LARGE as the other image model. VDA-Base and -Large are
   CC-BY-NC-4.0: not supported.
3. Precision: float32 weights as stored, float32 attention (ENC_ATTN_TILE on
   the GPU). Nothing else without approval.
4. Holes: the background stretch and an edge-aware fill (the default);
   inpainting by FLUX.2 waits for an inpainting pipeline (none yet).
5. Defaults: 2 % divergence, the convergence on the subject, full
   side-by-side "_LR", HEVC by NVENC, our own st3d box writer.

## 1. The depth models

All four checkpoints are float32 and all four bodies are DINOv2 ViTs with
14 x 14 patches (LayerScale, erf GELU MLP, pre-LN), the positions
interpolated bicubic by scale factors with DINOv2's 0.1 offset. Each was
loaded only as tensors (safetensors, or torch.load(weights_only=True): all
three pickles load that way, none refused).

### 1.1 What each model is

**DA3MONO-LARGE** (depth-anything/DA3MONO-LARGE, 334M, 1.34 GB float32,
Apache 2.0 on the card; the repo's LICENSE is Apache 2.0).
- ViT-L/14 (24 layers, 1024 wide, 16 heads of 64), outputs of layers 4, 11,
  17, 23 through the final LayerNorm (eps 1e-5 there, 1e-6 in the blocks).
- DPT head as Depth Anything V2's: 1 x 1 projections to 256 / 512 / 1024 /
  1024, ConvTranspose 4x4 s4 and 2x2 s2, identity, a 3 x 3 stride-2 conv;
  3 x 3 "rn" convs to 256; four fusion blocks (ResidualConvUnits of two 3 x 3
  convs, bilinear upsampling with align_corners, 1 x 1 out conv); output_conv1
  (3 x 3, 256 -> 128) at 8x the patch grid, bilinear to the processed image
  size, output_conv2 (3 x 3 128 -> 32, ReLU, 1 x 1 -> 1). Depth = exp(logit).
  A second sky branch (same shape, ReLU): sky where < 0.3, and the API sets
  the sky to the 99th percentile of the other pixels' depth (from a random
  subsample of 100k pixels: not deterministic; ours would take the exact
  quantile).
- Output: **relative depth** (up to an unknown scale, not affine: it
  predicts depth, not disparity, by the authors' design), at the processed
  size.
- Input: the longest side to 504 (process_res, cv2 INTER_AREA down,
  INTER_CUBIC up), each side to the nearest multiple of 14 (ties up), ImageNet
  mean and std. A 1920 x 1200 photo runs at 504 x 322 (828 tokens).
- The authors' API runs the backbone under bfloat16 autocast on CUDA.

**DA3-LARGE-1.1** (depth-anything/DA3-LARGE-1.1, 0.35B, 1.6 GB float32,
Apache 2.0). The any-view model: the same ViT-L with, from layer 8, LayerNorm
q/k norms, a 2D RoPE (frequency 100; half the head by row, half by column,
rotate-half within each) and attention that alternates local (within a view)
and global (across all views; the RoPE of no position difference); a
camera token in the cls row at layer 8; each output the concatenation of the
last local layer's rows and the normed global layer's (2048 wide, layers 11,
15, 19, 23). DualDPT head: a LayerNorm on the 2048, UV sinusoidal position
tables added after the projections and before output_conv2, depth = exp and
a confidence exp + 1, plus a ray branch (7 channels) and a camera decoder
(pose, intrinsics) that a stereo conversion does not need. No sky head.
- Measured on the Golden Gate photo (one view): the depth of the ground and
  the structures agrees with MoGe-2 better than DA3MONO's (median |log
  ratio| after a scale 0.052 against 0.19), but the **sky comes out at
  finite, varying depth** and thin cables against it are lost (no sky head;
  its confidence is 1.000 everywhere for one view). Not for single images.
- Its use here would be video as many views (consistent depth and camera
  poses across frames); its global attention is over all frames' tokens, so
  the cost grows with the square of the window. A later option.

**Video-Depth-Anything-Small** (depth-anything/Video-Depth-Anything-Small,
28.4M, 116 MB float32, Apache 2.0). **Surprise for the license rule: only
Small is Apache 2.0; VDA-Base and VDA-Large are CC-BY-NC-4.0** (the repo's
README). Two of the open converters (depth-surge-3d by default) use the
Large one.
- ViT-S/14 (12 layers, 384 wide, 6 heads), outputs of layers 2, 5, 8, 11.
- DPT head of 64 features (projections to 48 / 96 / 192 / 384) with four
  temporal modules (at layer 3 and 4 of the pyramid and after refinenet4 and
  3): GroupNorm 32, a linear in, two temporal self-attentions (each spatial
  position attends over the 32 frames of the window; 8 heads; a sinusoidal
  frame position added), a GEGLU FFN (x8), a linear out, + x.
- Output: **affine-invariant disparity** (ReLU, inverse depth up to scale
  and shift).
- Input: the shorter side to 518 (lower_bound, multiple of 14, INTER_CUBIC;
  wider than 1.78:1 shrinks it), frames first scaled to at most 1280 on the
  long side. 4:3 at 518 x 686 (1813 tokens a frame), 16:9 at 518 x 924 (2442).
- Long video: windows of 32 frames, a step of 22 (an overlap of 10: the
  first 10 frames of each window are the keyframes 0, 12, 24..31 of the
  previous one), a least-squares scale and shift of the new window onto the
  old on 2 keyframes, an 8-frame linear crossfade. So a frame costs 32/22 =
  1.45 frames of network.
- The authors run it under float16 autocast (float32 with --fp32).
- Measured on 32 real frames (a LaserDisc capture, 1440 x 1080, a man
  wiring a PC tower): clean edges, the frame-to-frame change of the
  disparity 0.5-1.1 % of its mean.

**MoGe-2 ViT-L** (Ruicheng/moge-2-vitl and -vitl-normal, 326M / 331M, 1.3 GB
float32, MIT; the repo's LICENSE is MIT, its DINOv2 code Apache 2.0).
- ViT-L/14 (layers 5, 11, 17, 23, final norm eps 1e-6), each output 1 x 1
  projected to 1024 and the four summed; the image resized bilinear
  **antialiased** to the token grid x 14.
- A ConvStack neck over 5 levels (1024 at the token grid P, 256 at 2x, 128
  at 4x, 64 at 8x, 32 at 16x) with the view-plane UV as 2 extra channels at
  each level; resamplers ConvTranspose 2x2 s2 + a 3 x 3 conv (the last
  bilinear + conv); residual blocks of two 3 x 3 convs, **replicate padding**,
  no norms. Heads of the same form: points (3), mask (1), normal (3, the
  -normal checkpoint). An MLP on the last cls row: the metric scale (exp).
- Output: points (xy z, z = exp, x and y times z) bilinear to the input size;
  focal and z-shift recovered by Levenberg-Marquardt (SciPy least_squares,
  one parameter, on a 64 x 64 nearest subsample, ftol 1e-3); **metric depth
  in metres, the field of view, a validity mask (sky), normals**.
- Input: num_tokens 1200-3600 by resolution_level (9: 3600). At 3600 a
  16:10 photo runs at 47 x 76 tokens, the heads at 752 x 1216.
- The authors run it under float16 autocast.
- Measured on the Golden Gate photo: horizontal FOV 47.3 deg, depth 27 m to
  415 m (median 60 m), its sky mask the same as DA3MONO's sky head (two
  independent heads agree to the pixel at the outline).

## 2. The port

The files:

| Path | What |
|---|---|
| reclank/models/depth/common.py | The program builder (pooled buffers), DINOv2's blocks as forms, the 3 x 3 convs (im2col tiles, zero or replicate padding, stride 2), the transposed convs (a linear and a pixel-shuffle gather), the host's resizes (OpenCV's area, cubic, linear; torch's antialiased bilinear) and their taps for ENC_RESIZE_TAPS, the positions of a patch grid (bicubic by scale factors, the 0.1 offset). |
| reclank/models/depth/dpt.py | The DPT head: DA3's (with its sky branch) and VDA's (with its four temporal modules: GroupNorm a frame, two attentions of each pixel over the window's frames, the GEGLU FFN), the last levels a micro-batch of 4 frames at a time. |
| reclank/models/depth/da3.py | DA3MONO-LARGE: the preprocessing (upper_bound_resize to 504, the nearest multiple of 14), depth = exp(logits), the sky at its 99th percentile. |
| reclank/models/depth/moge.py | MoGe-2: the ConvStacks (the UV channels' input convs folded into tables of the size), the scale MLP on the host, infer()'s post-processing (the focal and shift by SciPy's Levenberg-Marquardt on a 64 x 64 nearest sample, metric depth, the mask, normals). |
| reclank/models/depth/vda.py | VDA-Small: a window of 32 frames as one program, the preprocessing as a program of its own (the uint8 frame / 255, OpenCV's cubic by taps, the mean and std, the patch rows), the authors' windows (keyframes, the scale and shift fit at the frames' size, the crossfade) streamed: each map out as soon as no later window changes it. |
| reclank/models/depth/stereo.py | The disparity (relative and metric modes), the Warper (both eyes and their packing as one program), the anaglyph, the st3d box writer (in place for a moov at the end; the chunk offsets moved for a moov in front). |
| reclank/models/depth/pipeline.py | Stereo: the models loaded at their first use, image() and video() (ffmpeg in, VDA's windows, the warp, ffmpeg out with the source's audio, the st3d box; RESIDENCY.use around the runners; a gate lock a frame). |
| reclank/models/depth/ref.py | NumPy references of the new records (the Python interpreter's steps and the unit tests). |
| scripts/pt_to_safetensors.py | VDA's .pth and MoGe-2's .pt converted once to safetensors with torch.load(weights_only=True) (MoGe's model_config to model.json beside it): reclank reads only safetensors. |
| server/stereo.py, server/generate.py | POST /v1/images/stereo, POST /v1/videos/stereo and the stereo jobs beside Wan's under /v1/videos; --stereo-moge, --stereo-da3, --stereo-vda. |
| scripts/stereo_reference.py, check_stereo.py, stereo_convert.py | The torch-only reference, the checks, the command line. |
| tests/unit/test_stereo.py | The records on the CPU and in Python against ref.py (the same bits), the Warper, VDA's streamed alignment against the authors' loop, the st3d box (on a crop of a real photo and real VDA maps, tests/unit/data/stereo_real.npz). |

New records (both runners; numbers from 300 up):

| Record | What |
|---|---|
| ENC_RESIZE (300) | Bilinear as torch's (align_corners or not; its float32 weights and order of sums), or the mean of integer blocks (the half layouts). |
| ENC_DIBR (301) | One eye: the 1-D mesh splat with 64-bit atomicMax keys, the resolve, the background stretch, the 5-tap or the edge-aware fill; into a strided place of the packed image. |
| ENC_PACK_U8 (302) | Rounded (to nearest even) and clamped bytes, four in a float32 word (the download of the packed image). |
| ENC_RESIZE_TAPS (303) | A separable resize by host-made taps (OpenCV's cubic for VDA's input; any filter). |
| ENC_UNPACK_U8 (304) | Bytes to float32 divided by a number (the upload of an uint8 frame; / 255 for VDA). |

The CPU's C, the GPU's kernels (the _rn intrinsics, no contraction) and ref.py give the same bits for
all five (the unit tests on the CPU, check_stereo.py warp on the GPU).


## 3. View synthesis

### 3.1 Disparity from depth

Two modes, one formula. The shift of a pixel in each eye, in pixels of the
output image of width W:

    s = k * (D - D0)          left eye +s (near content moves right), right eye -s

where D is a disparity-like quantity, D0 the one put on the screen plane
(zero parallax), k the gain.

- **Relative mode** (DA3MONO, VDA; any model): D = inverse depth normalized
  to [0, 1] between robust percentiles (1st and 99.5th; the sky forced to
  0); k = divergence x W / 2 with divergence the total parallax budget as a
  fraction of the width; D0 = convergence (auto: the median of the frame's
  centre, smoothed over time for video: an EMA as DepthConverter's, alpha
  ~0.1). For video the normalization itself must be smoothed in time (the
  percentiles per window, or VDA's aligned disparity across windows, never
  per frame), or the whole scene pumps.
- **Metric mode** (MoGe-2): D = 1 / Z (metres), k = f_px x B / 2 with f_px the
  output image's focal in pixels (from MoGe's FOV) and B the stereo baseline
  (the eye distance, 63 mm by default); D0 = 1 / Z0 with Z0 the distance put
  on the screen plane. With B = the viewer's IPD and the image shown at its
  own FOV this is orthostereo: the scene at its true size. Hyperstereo (B
  larger) for distant landscapes, hypostereo for close-ups.

Comfort limits (the defaults to clamp to, whatever the mode):
- Total parallax: 1-3 % of the width on a flat virtual screen; DepthConverter
  defaults to 1.2 % and clamps each eye at 1 % (2 % total); stereo film
  practice keeps behind-screen parallax under the eye separation on the
  screen and in-front parallax under ~1 deg. Proposed default: **2 % total,
  each eye clamped at 1.5 %**, convergence at the subject.
- The vergence-accommodation conflict of a headset (focus fixed near 1.3-2
  m): keep vergence within about +-0.5-1 diopter of it: nearest content at
  about 0.6-0.7 m. In metric mode clamp Z at Z_near = 0.6 m (a soft rolloff
  of 1/Z beyond it rather than a hard clamp).
- Window violations: objects in front of the screen cut by the frame edge.
  A floating window (black bars that move the frame edge forward) or pushing
  the convergence back when the edges hold near content.
- "Beyond Geometry: Artistic Disparity Synthesis" (arXiv 2603.05906, CVPR 2026):
  argues that 3D films are not geometric (local out-of-screen effects,
  per-shot depth budgets) and learns a dual-path (global depth budget + local
  effects) remap from professional 3D film; the abstract gives no rules or
  code. For us it supports exposing the remap as a curve (gain, convergence,
  a gamma or S-curve on D, per-shot budgets) rather than only the geometry,
  and leaves a learned remap as research.

### 3.2 The warp (ENC_DIBR)

Each row of the image is a 1-D mesh: source pixel j lands at j + s_j (the
left eye; the right eye -s_j). The segment from j to j + 1 is drawn when it
neither folds (an occlusion) nor stretches past 2 pixels (a depth edge: a
disocclusion); its target pixels take the colour and the shift interpolated
between its ends, and each pixel is also drawn as a box of one pixel (thin
objects survive). The nearest wins each target pixel: a 64-bit atomicMax of
(the shift made orderable: larger is nearer in both eyes, the source index,
segment over box), then a pass that resolves each winner's colour. A hole
takes the colour of its farther drawn neighbour in the row (the background
stretch), then either a 5-tap horizontal mean (fill "stretch") or 8 Jacobi
steps of the mean of its 3 x 3 neighbours that are holes or not nearer than
its background by 0.5 pixel (fill "edge", the default). interp.c, the CUDA
kernels and models/depth/ref.py do the same float32 operations in the same
order (no FMA contraction): the same bits.

### 3.3 Output formats and how players recognise them

| Format | Layout | Size of a 1920 x 1080 source |
|---|---|---|
| Full SBS | left eye | right eye | 3840 x 1080 |
| Half SBS | each eye squeezed to half width | 1920 x 1080 |
| Full OU (TB) | left on top | 1920 x 2160 |
| Half OU | each eye half height | 1920 x 1080 |
| VR180 (equirect, SBS) | each eye a 180 x 180 deg equirect | e.g. 2 x 2880 x 2880 |
| Anaglyph (red-cyan) | for checks only | 1920 x 1080 |

- A flat photo or video is not 180 deg: VR180 output would show it as a
  window of its own FOV inside black. With MoGe's FOV we can place it at its
  true angular size (orthostereo); the players' flat-screen 3D modes (SBS on
  a virtual screen) are the normal way and the default here.
- File naming (what DeoVR, Skybox, HereSphere, Pigasus and others detect;
  to verify on the user's headset): full SBS `_LR` or `_SBS` (DepthConverter
  writes `_Full_SBS_LRF`), half SBS `_3dh` / `_Half_SBS_LR` / `_HSBS`,
  over-under `_TB` / `_OU` / `_3dv`, VR180 `_180_LR` (or `_180x180_3dh_LR`).
- MP4 metadata: the Spherical Video V2 boxes: `st3d` (stereo mode: 1
  top-bottom, 2 left-right) in the video sample entry, and for VR180 `sv3d`
  (svhd, proj with prhd and equi, the bounds of a half equirect: left and
  right 0x40000000). The local ffmpeg is 4.4.2: it cannot write them from
  the command line, so a small MP4 box writer (Python, the sample entry's
  children) does it after the encode. Google's spatial-media tool does this
  too but is GitHub code (not run without the user's OK).
- MKV: ffmpeg writes Matroska's StereoMode (`-metadata:s:v:0
  stereo_mode=left_right` / `top_bottom`).
- Images: JPEG/PNG with the name tags; MPO (two JPEGs, for 3D cameras and
  some viewers) as an option.
- Encoders here: libx264, libx265, h264_nvenc, hevc_nvenc. Full SBS of 4K
  is 7680 x 2160: past NVENC H.264's 4096 width, so HEVC (NVENC HEVC goes to
  8192). Audio copied from the source.

## 4. Progress

- **The reference** (scripts/stereo_reference.py) was cross-checked by running the authors' code in a
  sandbox (CPU float32): DA3MONO's and MoGe-2's outputs (the input, every head, the sky, the metric
  scale) the same bits; VDA-Small's window 1 - cos 2.6e-12 (rel. 5e-7: the authors normalize the input
  in float64, the reference in float32).
- **Against the reference** (check_stereo.py; the bridge photo 1920 x 1200, the cat portrait 1795 x
  2397, 96 frames of a LaserDisc capture at 1440 x 1080, 59.94 fps):

  | Model | Device | Result |
  |---|---|---|
  | DA3MONO 504 (828 tokens) | CPU / GPU | logits 1 - cos 1.2e-12 / 1.3e-12, depth rel. err. <= 4.9e-5; sky 1 - cos 3e-13 |
  | DA3MONO 1008 (3240 tokens) | GPU | logits 1 - cos 1.0e-12, depth rel. err. <= 3.9e-5 |
  | MoGe-2 vitl-normal, 3600 tokens (47 x 76) | CPU / GPU | points 1 - cos 1.7e-12 / 1.8e-12, mask 1.6e-12; metric scale 30.08804 (ref 30.08805); fx 1.143046 (1.143045); masks agree on every pixel; depth (m) rel. err. median 1.4e-6, max 1.7e-4 |
  | VDA-Small, 96 frames (5 windows, aligned) | CPU (one window) / GPU | 1 - cos <= 1.8e-13 every frame, max\|d\|/max\|ref\| <= 2.7e-5 |
  | The preprocessing | | DA3: OpenCV's uint8 resizes in float arithmetic, 1 level off on 0.05% of values; VDA: the program's cubic taps against cv2, 1.9e-6 in normalized units |
  | The warp, every layout, both fills | CPU / GPU | the same bytes as ref.py at 1920 x 1200 and 3840 x 2400 |

  End to end from the photo (DA3MONO): the depth's median rel. err. 1.3e-3 (the preprocessing's
  levels); the pixels the sky branch puts on the other side of 0.3 change by the sky fill.
- **The 3090** (float32, the usual limits; the median of runs after the first):

  | Job | Time |
  |---|---|
  | DA3MONO depth, 504 / 1008 | 0.115 s / 0.451 s (the first run of a size: its program, 3-6 s) |
  | MoGe-2 depth (3600 tokens) + focal/shift | 0.94 s + host 0.3 s (a new size's program: about 2 s to build) |
  | The warp + packing, 1920 x 1200 full SBS (uint8 in and out) | 23 ms; 3840 x 2400: about 0.2 s |
  | A photo to a stereo JPEG (moge-2, cached programs) | about 3.5 s with the file I/O |
  | VDA-Small, a window of 32 frames | 1.11 s (50 ms an output frame) |
  | Video, 1440 x 1080 to 2880 x 1080 HEVC, end to end | 3.8-5.2 fps (the depth 133-180 ms a frame with its per-frame preprocessing program and the host's numpy; the warp 30-49 ms; CPU shared with other jobs) |

- **The server** (python -m server.generate --stereo-moge DIR --stereo-da3 DIR --stereo-vda DIR):
  /v1/models lists the three; POST /v1/images/stereo (JSON base64 or multipart) gives the packed
  image (moge-2 full SBS 3840 x 1200; da3mono half over-under of the portrait, its odd row dropped),
  400 for a bad field; POST /v1/videos/stereo (multipart) queues a job, GET /v1/videos/{id} its
  frames done, /content the MP4 (HEVC, the st3d box: ffprobe "side by side"), DELETE removes it.
  RESIDENCY loads each part at its first use (moge-2 2.95 GB, da3mono 2.16 GB, vda-small 3.41 GB,
  the warp 0.1-0.4 GB).
- **Next**: the video's throughput (the per-frame preprocessing program into the window's program,
  the host's per-frame statistics on the GPU); FLUX.2 inpainting of wide holes when an inpainting
  pipeline exists; VR180 output; the setup catalog's entries.
