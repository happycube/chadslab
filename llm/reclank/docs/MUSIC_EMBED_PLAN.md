# Music embeddings: MuQ-MuLan and LAION CLAP side by side

## 1. Decisions

- **Two models, one vector.** A music vector is MuQ-MuLan's 512 floats next
  to CLAP's 512 floats, each half L2-normalized and the whole divided by
  sqrt 2. The cosine of two joined vectors is then the mean of the two
  models' cosines. A text query is joined the same way, from the two text
  towers. On a library of 2,188 tracks (easy listening, library music) the
  join beats each model alone on every similarity measure:

  | | same track top-1 | album NN | album NN by album |
  |---|---|---|---|
  | MuQ-MuLan | 0.539 | 0.633 | 0.492 |
  | CLAP (music_audioset) | 0.438 | 0.691 | 0.587 |
  | joined | 0.568 | 0.725 | 0.602 |

  MuQ-MuLan is the better text search for instruments (harpsichord,
  whistling, accordion); CLAP for genres (disco).
- **No log-mel floor.** Both front ends take the log of the mel with a
  floor of 1e-10 and no top_db, as trained. A floor of 60 dB under each
  clip's peak makes AAC and Opus encodes of one recording agree (cosine
  0.99) but costs 4-5 points of album NN; 70 and 80 dB are in between, no
  sweet spot.
- **Duplicates by fingerprint, not by vector.** The same recording in two
  codecs can be far apart in MuQ-MuLan (cosine down to 0.76). Duplicates are
  found by Chromaprint (ffmpeg's chromaprint muxer, raw words of the first
  120 s; pairs that share words, aligned at their most common offset, bit
  error rate under 0.20). Each track is fingerprinted at its own speed and
  at 0.97, 0.98, 0.99, 1.01, 1.02 and 1.03 (ffmpeg asetrate: pitch and tempo
  together, as a tape run fast or slow), each against the others at their
  own. On the library it found every known duplicate (13 groups by video id)
  and 34 more pairs:
  - renamed copies and "(1)" copies;
  - remasters (bit error 0.06-0.13);
  - 17 uploads of the same recordings about 1.7 % slow, which only the
    speed variants find (at speed 1 their bit error is 0.25-0.36, among
    unrelated pairs at 0.35).

  Two remasters (0.175-0.18 over the first 60 s) stay just over the
  threshold over 120 s.

## 2. The models

### MuQ-MuLan (OpenMuQ/MuQ-MuLan-large, CC-BY-NC-4.0)

The weights may not be used commercially; reclank carries the code only.

- **Audio:** 10 s clips of 24 kHz mono.
  - The mel of torchaudio: n_fft 2048, hop 240, 128 mels, power 2, HTK, no
    norm. AmplitudeToDB (10 log10, amin 1e-10, no top_db). The last frame is
    dropped (1000 frames). Then (x - 6.768) / 18.418.
  - Conv2dSubsampling: two Res2d blocks of 512 channels, each of stride
    (2, 2).
    - Each block is conv3x3, BN, ReLU, conv3x3, BN, plus a residual of
      conv3x3 stride 2 and BN, then ReLU.
    - The BatchNorms fold into the convs.
    - The result is 32 bands by 250 frames by 512 channels, rearranged to
      rows of (channel, band) = 16384, and a linear to 1024.
  - The Wav2Vec2-Conformer of transformers: 12 layers, width 1024, 16 heads
    of 64, FFN 4096 with swish, the conv module's kernel 31. Each layer:
    - FFN1 (LN, linear, swish, linear) times 0.5, + x;
    - attention: LN, then RoPE (base 10^4, NEOX halves of each head of 64)
      applied to the input of q and k, not to q and k themselves (v reads
      the input unturned; transformers' _apply_rotary_embedding); q k v
      with bias, scale 1/8, the output with bias, + x;
    - the conv module: LN, pointwise 1024 -> 2048, GLU, depthwise k31 "same"
      padding, BatchNorm (folded), swish, pointwise 1024 -> 1024, + x;
    - FFN2 times 0.5, + x;
    - the final LN of the layer.
  - The encoder's LayerNorm, then MuLan's proj 1024 -> 768 (its audio
    transformer has depth 0), the mean over the 250 rows, audio_to_latents
    768 -> 512, L2 norm.
- **Text:** XLM-R base (12 post-LN layers, width 768, GELU erf, the
  sentencepiece tokenizer of xlm-roberta-base).
  - Then MuLan's transformer of 8 layers, width 768. Each layer: pre-LN with
    no bias; q and kv with no bias; the L2 norm of each head of 64, times
    q_scale and k_scale, scale 8; out; then FF of LN, a linear to 2 x 2048,
    GEGLU (x, gate halves; gelu erf of the gate), a linear back.
  - The mean over the rows, text_to_latents 768 -> 512, L2 norm.
  - **Padding:** muq's code takes the mean over every row of a padded batch
    (pads included, no mask in the 8 layers). So a text's vector depends on
    its batch. reclank embeds each text alone, and the reference does the
    same.

### CLAP (LAION-CLAP music_audioset_epoch_15_esc_90.14, CC0-1.0)

- **Weights:** use LAION's original checkpoint (lukewys/laion_clap). The
  transformers upload laion/larger_clap_music is broken: an untrained model
  with zero biases, every text vector the same (its discussion #2). The
  converter maps the original to the ClapModel names, which serve as the
  float32 reference.
- **Audio:** 10 s clips of 48 kHz mono, as ClapFeatureExtractor in
  rand_trunc mode. A clip of exactly 480000 samples is neither cut nor
  padded.
  - STFT 1024, hop 480, Hann; the Slaney mel of 64 bands from 50 to
    14000 Hz; power to dB (amin 1e-10, no top_db); 1001 frames by 64.
  - HTSAT-base:
    - BatchNorm over the 64 mel bins (folded into an affine);
    - the spectrogram reshaped to a 256 x 256 image (HTSAT's
      reshape_wav2img);
    - patch embed conv 4x4, stride 4, to 128 channels, with LN;
    - Swin stages of depths 2, 2, 12, 2, widths 128, 256, 512, 1024, and
      heads 4, 8, 16, 32, with window 8. Each block uses the relative
      position bias, and the shifted blocks use the cyclic shift and its
      mask.
    - patch merging between the stages (2 x 2 gather, LN, linear 4c -> 2c);
    - the final LN, then the average over the patches (the "embedding" of
      HTSAT, not the tscam head).
  - audio_projection: linear 1024 -> 512, ReLU, linear 512 -> 512, L2 norm.
- **Text:** RoBERTa base (the GPT-2 BPE tokenizer of roberta-base), the
  pooler (dense with tanh on the first row), text_projection (linear 768 ->
  512, ReLU, linear), L2 norm.

## 3. New records

The text towers need no new record: linear with bias, ENC_LNORM, ENC_GELU
erf, ENC_ATTN_RANGE (each text over its own keys), ENC_ADD and ENC_MUL_VEC.

- **ENC_GELU_MUL erf:** an operand for the erf form (MuLan's GEGLU).
- **ENC_DWCONV_SAME:** the depthwise conv with centered padding, within
  segments of a given number of rows (one segment a clip, so packed clips
  do not leak into each other).
- **ENC_GATHER:** out row i, block j = x row idx[i, j] (-1: zeros). It
  serves the Swin window partition and reverse, the cyclic shift, patch
  merging (4 blocks), HTSAT's reshape, and the im2col of MuQ's convs (9
  blocks of channels).
- **ENC_ATTN_BIAS:** attention within groups of g consecutive rows (a
  window), plus an additive bias of (types, heads, g, g). The types are the
  relative-position bias, and the bias plus the shift mask for each kind of
  window. Scale is an operand.
- **MuQ's conv2d** as ENC_GATHER (im2col) plus ENC_LINEAR on the CPU (the
  largest conv, the second of block 1, is 32000 positions by 4608 inputs:
  590 MB a clip).
- **ENC_GATHER_MM** (GPU only, BF12 rows): the product with the rows of
  ENC_GATHER without the gathered rows. x goes to its bfloat16 planes (9
  times fewer values than the im2col), and k_gemm_bf16_tc<.., GX> reads
  each step's 32 columns from the row of x the index names. Same bits as
  ENC_GATHER then the product. The compiler's enc_gather_linear emits it
  in a GPU program and the old pair elsewhere.

Each record needs its C interpreter (interp.c), its CUDA kernel
(encoders.cu), and its entries in launch.cu, runner.cu, opcodes.py and
program.py, as ENC_ROPE1D did.

## 4. Steps and checks

The float32 references are torch, in the venv of the trial (transformers
4.48.3, muq 0.1.0). The checks run on real audio from the library and real
queries. They measure the cosine of reclank's vector against the
reference's, on CPU, GPU, and mixed placement.

1. **CLAP text (RoBERTa):** no new records. Gate: cosine >= 0.99999.
2. **MuQ-MuLan text (XLM-R + 8 layers):** ENC_GELU_MUL erf. Gate: cosine
   >= 0.99999.
3. **MuQ audio:** the mel on the host (numpy, against torchaudio to 1e-5),
   ENC_GATHER im2col + linear, ENC_DWCONV_SAME, the conformer. Gate: cosine
   >= 0.9999 (BF12 on the GPU: report the ppm, as for EmbeddingGemma 2).
4. **CLAP audio (HTSAT):** the mel on the host, ENC_GATHER and ENC_ATTN_BIAS.
   Gate: cosine >= 0.9999.
5. **Server:** a model "music" in server/embed.py.
   - Input is audio of at most 30 s, cut into 10 s clips; the last clip is
     padded with zeros only when shorter than 5 s, else it is taken from the
     end.
   - Each model's vector is the mean of its clips.
   - Output is the joined 1024 (or `parts` for each 512). Text queries use
     the joined text towers.
   - The towers run on gpu, cpu or off, as EmbeddingGemma 2's.
6. **Library tool:** scripts/embed_music.py.
   - Each track gets windows of 30 s at 20 % and 60 % of its length; the
     track vector is the mean of the two windows.
   - Chromaprint duplicate groups are written beside the vectors.
   - Queries: text, "more like this" (a track), and k-means clusters.
   - Gate: the trial's numbers on the library (album NN 0.725 joined) within
     noise of the BF12 rounding.

## 5. Status

- **Steps 1-4 are done, float32.** The new records are ENC_GATHER and
  ENC_ATTN_BIAS, plus an erf operand of ENC_GELU_MUL and an eps operand of
  enc_rms and enc_lnorm. The towers are in reclank/models/music: text.py
  (both text towers), unigram.py (XLM-R's tokenizer), muq.py and clap.py.
  The weights come from scripts/convert_music.py, the references from
  scripts/music_reference.py, and the checks from scripts/check_music.py.
- **Checks:** 152 texts (queries, tags, track titles in Czech, German,
  Spanish) and 12 clips, against the float32 references:
  - the token ids of both tokenizers all the same;
  - 1 - cos at most 1.8e-7 for every tower, on the CPU and the GPU;
  - the same with 16 texts packed in one run.

  EmbeddingGemma 2's checks still pass after the changes to the shared
  records.
- **BF12 (approved):** the audio towers take BF12 matrices on the GPU
  (text.gpu_bf12: the float32 W rounded to bfloat16, then BF12; the
  matrices of 9 or 16 inputs stay float32). The text towers stay float32 on
  the GPU too. Worst 1 - cos against the float32 references:

  | tower | float32 | BF12 |
  |---|---|---|
  | MuQ audio | 1.2e-7 | 2.6e-6 |
  | CLAP audio | 1.2e-7 | 4.9e-6 |
  | CLAP text | 1.2e-7 | 4.1e-5 (not used) |
  | MuQ-MuLan text | 1.8e-7 | 4.1e-4 (not used) |

  A query takes milliseconds in float32.
- **Speed on the GPU (BF12):**
  - A clip takes 28 ms of GPU in MuQ (52 ms before the tuning below) and
    12 ms in CLAP. The host's part (mel, patches) is 42 and 17 ms, done in
    worker processes or threads beside the GPU.
  - So a track (2 windows of 3 clips) takes about 0.26 s of the programs;
    the target was 0.25 s.
  - The tuning of MuQ, with the same bits at every step (each clip's
    hidden state equal to the single-clip program's, on the GPU and the
    CPU; the 8082 server's vectors of EmbeddingGemma 2 and music the same
    before and after):

    | step | ms a clip |
    |---|---|
    | one program a clip | 52.4 |
    | 5 clips a program (muq.BATCH): the conformer's products of 1024 rows were 16 blocks on 82 SMs | 37.2 |
    | k_gemm_bf16_tc NF: the tiles of w fastest when t >= 16 rows (x read from DRAM once) | 35.9 |
    | the second conv block over the 5 clips; k_enc_attn_range_t<hd / 32> (3x) | 32.2 |
    | ENC_GATHER_MM: the convs' products read the im2col rows by index (no 590 MB copy) | 27.7 |

    - A clip has 256 rows in the conformer (250 and 6 of zeros): the blocks
      of 8 queries and tiles of 32 keys of ENC_ATTN_RANGE then fall as in a
      program of the clip alone. 5 clips are 10 tiles of 128 rows: 80
      blocks for the products of 1024 rows (6 clips: 96, two waves).
    - Under load the 3090 runs at its 225 W cap (about 1300 MHz, not
      1695): the MMA peak there is about 54 TFLOPS of bf16 with float32 sums
      (71 at full clock). The big conv runs at 39 of MMA.
    - Tried and dropped: 3 stages of cp.async and ldmatrix for x (no
      change: the kernel is not bound by the issue of its loads).
    - Not taken: one bfloat16 plane of x (RECLANK_GPU_BF16_TC=8, half the
      products): 35.9 -> 28.1 ms a clip before ENC_GATHER_MM; 1 - cos
      against torch MuQ 2.62e-6 -> 2.86e-6, CLAP 4.9e-6 -> 9.3e-6. It is a
      switch of the whole process (EmbeddingGemma 2 too) and needs the
      user's approval.
- **Server (step 5):** `server/embed.py --music DIR` (with or without -m).
  "model": "music" (1024), "muq-mulan" or "clap" (512). Inputs are texts
  or {"audio", "start", "end"} of at most --music-max-s (60 s). Other
  requests go to EmbeddingGemma 2 as before.
- **Library tool (step 6):** `scripts/embed_music.py`, with the commands
  index (incremental, fingerprints in the library), query, like, clusters
  and dups.
  - Windows of 30 s are centered at 20 % and 60 % of a track.
  - The decode, MuQ's mel and CLAP's patches run in worker processes
    (frontend.py, NumPy alone). Threads were held one at a time by the GIL,
    since numpy's FFT keeps it.
  - ffmpeg and ffprobe run without reclank's OMP_PROC_BIND and OMP_PLACES.
    ffmpeg links libgomp, which bound every ffmpeg to CPU 0: 8 at once took
    7.4 s instead of 1.0 s. EmbeddingGemma 2's decode_audio had the same
    bug.
- **The library of the trial (2,188 tracks), indexed by reclank on the GPU
  (BF12):**
  - It took 2.4 tracks a second, the programs 0.41 s a track; after the
    tuning (MuQ's clips 5 a program across tracks) 3.1 tracks a second, the
    programs 0.26 s a track (60 tracks: the same vectors bit for bit). The
    workers' decode and mel are now the limit.
  - The fingerprints of 7 speeds take 10 minutes the first time and 13 s
    when cached. They give 45 duplicate groups (91 tracks).
  - Against the trial's torch vectors, the median window cosine is
    0.999999 (worst 0.987). Every measure is within 0.003 of the trial's:

    | | same track top-1 | album NN | by album |
    |---|---|---|---|
    | trial, joined | 0.568 | 0.725 | 0.602 |
    | reclank, joined | 0.567 | 0.726 | 0.605 |

  - Text search is better with one model's half (query --part) than with
    the join, which averages in the model that does not know.
    - MuQ-MuLan alone finds Elton John's tracks for "Elton John" and Court
      Minstrel for "harpsichord".
    - CLAP alone finds Le Freak for "disco beat".
    - The join is best for track to track (like, clusters).

## 6. Speed targets (one RTX 3090)

The torch trial ran at 4.6 tracks a second for MuQ-MuLan alone (6 clips a
track, float32). CLAP took 3.6 a second. The target is both models together
at 4 tracks a second or better, on the GPU, with the decode (ffmpeg, 24 and
48 kHz) in threads beside it.
