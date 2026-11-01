# Plan: speech, image, music and video generation

Give reclank generation models: text to speech, text to image, text to
music, and text (or image) to video. Every model runs on the records it
already has where it can (the ENC_* programs and the decode engines), on the
CPU and the GPU, and is checked against its float32 reference on real
inputs, as the encoders were (MUSIC_EMBED_PLAN.md, MULTIMODAL_PLAN.md).

The machine is one RTX 3090 (24 GB, 225 W cap) and two sockets of 24 cores
with 384 GB of RAM. The weights live in a local models directory, from the
official repositories or our own conversion. Any quantization of a model
(BF12, 8-bit, int4) needs the user's approval for that model; until then a
model runs with the weights of its checkpoint (bfloat16 or float16 as
stored, float32 on the CPU).

**The license rule.** No license that carves out regions such as the EU, the
UK or South Korea (where the user may be). Exclusions that only follow
sanctions or export law (Iran and the like) are fine. Prefer
Apache 2.0 and the like (MIT, BSD). A non-commercial license is acceptable
only when that model has a definitive edge over every permissive one, and
the plan says what the edge is.

The facts of section 1 come from model cards and reviews at the time of
writing. The field moves fast: check each card (size, license, languages)
again before a download.

## 1. The candidates

### 1.1 Text to speech

| Model | Size | Architecture | License | Languages | Why popular |
|---|---|---|---|---|---|
| Higgs TTS 3 4B (bosonai) | ~4B (36 layers, 2560 wide, GQA 32/8) | Qwen3-4B-shaped decoder over 8 codebooks at 25 fps (delay pattern), Higgs audio tokenizer, 24 kHz | Research and non-commercial; a creator grant | ~100; Czech and Slovak in its best tier | Cloning, prosody tags, broad languages |
| Qwen3-TTS 12Hz 1.7B / 0.6B | 1.7B / 0.6B | Qwen3 decoder over a 12 Hz multi-codebook tokenizer (16 codebooks); no DiT | Apache 2.0 | 10 (zh en ja ko de fr ru pt es it) | Cloning from 3 s, streaming, permissive |
| Fish Audio S2 Pro | 4B slow AR + 0.4B fast AR | Dual autoregressive over a 10-codebook codec at ~21 Hz | Research; commercial by license | 80+ (tiers) | Top of the open Speech Arena |
| Kokoro 82M | 82M | StyleTTS 2 with an ISTFTNet decoder; a phonemizer (misaki) | Apache 2.0 | 8, no Czech | Tiny, fast, runs on a CPU |
| Chatterbox Multilingual V3 | 0.5B | Speech-token LM, a token-to-mel decoder, a HiFT-GAN vocoder | MIT | 23, no Czech or Slovak | Cloning, emotion control |
| VoxCPM2 | 2B | Tokenizer-free (continuous) TTS, 48 kHz | Apache 2.0 | 30, no Czech or Slovak | Quality, voice design |
| Voxtral TTS (Mistral) | — | — | CC BY-NC 4.0 | 9 | Quality |
| XTTS v2 (Coqui) | 0.5B | GPT-2 over discrete codes + HiFi-GAN | Coqui CPML (non-commercial) | 17, Czech yes | Old but still used for cloning |

Recommended first: **Higgs TTS 3 4B**, with **Qwen3-TTS 1.7B** right after
it.

- Higgs is the one candidate that speaks Czech and Slovak well (the user's
  Rádio Regina work): no permissive model covers them (Qwen3-TTS, Chatterbox,
  VoxCPM2 and Kokoro do not list them). That is the definitive edge that
  admits its research and non-commercial license (no geographic limit);
  reclank carries the code only, as for MuQ-MuLan.
- Both are a Qwen3-shaped decoder over the codes of a neural codec. The same
  pieces (a Qwen3 dense decoder in reclank, multi-codebook heads, a codec
  decoder) serve both, and Qwen3-TTS is the Apache 2.0 model for anyone who
  needs it.
- At bfloat16 Higgs takes about 9 GB and Qwen3-TTS 1.7B about 4.5 GB: both
  fit the 3090 beside the embedding server without quantization.

Kokoro is the small fallback for English on the CPU, but it needs an LSTM
and a phonemizer, which nothing else here needs (section 3.4).

### 1.2 Text to image

| Model | Size | Architecture | License | bf16 VRAM | Why popular |
|---|---|---|---|---|---|
| FLUX.2 [klein] 4B | 4B DiT + Qwen3-4B text encoder + VAE (32 latent channels) | Rectified-flow MMDiT (5 double-stream, 20 single-stream blocks); distilled (4-8 steps) and base | Apache 2.0 (the 9B klein: non-commercial) | ~8 GB DiT + 8 GB encoder | Quality for the size, permissive |
| Z-Image Turbo (Tongyi-MAI) | 6B DiT + Qwen3-4B encoder + the FLUX VAE | Single-stream DiT (S3-DiT: text and image tokens in one stream), 8 steps, no CFG | Apache 2.0 | ~12 GB + 8 GB | Photo realism, fast, permissive |
| Qwen-Image-2.1 | 7B DiT (32 single-stream layers) + Qwen3-VL-8B encoder + a 64-channel RGBA VAE (16x) | Single-stream DiT, flow matching; generation and editing (up to 10 reference images); 2048 x 2048 by default, 40 steps | Qwen Research License (non-commercial; no region carve-out) | ~14 GB DiT + ~16 GB encoder (encoder first, or on the CPU) | The newest Qwen image model, top of the open image Elo; native transparency |
| Qwen-Image-2512 | 20B MMDiT + Qwen2.5-VL-7B encoder + its VAE | MMDiT, flow matching | Apache 2.0 | ~40 GB: does not fit | Text in images |
| FLUX.2 [dev] | 32B | MMDiT | Non-commercial | ~64 GB | Top quality |
| SD 3.5 Large / Medium | 8B / 2.5B | MMDiT, three text encoders (CLIP L, CLIP G, T5-XXL) | Stability community | 16 / 5 GB | Ecosystem |
| SDXL | 2.6B UNet | UNet with cross-attention, two CLIPs | OpenRAIL++ | ~7 GB | The largest ecosystem (LoRAs) |

Recommended first: **FLUX.2 [klein] 4B**, then **Z-Image Turbo**, then
**Qwen-Image-2.1**.

- Both fit the 3090 at bfloat16 when the text encoder runs first and its
  memory is freed (or the encoder runs on the CPU: one pass of a prompt).
- Both take a Qwen3-4B text encoder: the same Qwen3 dense model as the TTS
  of section 1.1, used for its hidden states (FLUX.2 klein takes three of its
  layers, concatenated).
- Both are flow-matching DiTs with few steps, so the first sampler is the
  simplest (Euler, no CFG for the distilled forms).
- Qwen-Image-2.1 is the quality pick, with a definitive edge that admits its
  research license: the strongest open image model of the reviews, with
  editing from reference images and transparent (RGBA) output. It fits at
  bfloat16 with its encoder run first and freed. It reuses the work of the
  first two: a single-stream DiT like Z-Image, and a text encoder whose text
  part is a Qwen3-8B dense model (section 3.1.2; the vision part of Qwen3-VL
  is needed only for editing). New for it: the RGBA VAE, 2048 x 2048 (at 16x16 pixels a token, 16,384 image tokens: the attention
  kernel matters most here), and 40 steps with CFG. Its weights are under the Qwen Research
  License; check the LICENSE file before the download.
- Qwen-Image-2512 (Apache 2.0, 20B) needs 8-bit to fit (approval); it is
  phase 7.

### 1.3 Text to music

| Model | Size | Architecture | License | Why popular |
|---|---|---|---|---|
| ACE-Step 1.5 | DiT ~2-4B (XL: 32 layers, 2560 wide) + a Qwen3 planner LM (0.6/1.7/4B) + Qwen3-Embedding-0.6B + a music VAE | LM plans the song (metadata, lyrics, 5 Hz codes); a DiT renders the VAE latent; turbo 8 steps without CFG, base 50 with CFG | MIT | Full songs with lyrics, under 10 s a song on a 3090 (the authors) |
| YuE | ~7B (two stages) | Two LMs over X-Codec tokens (lyrics to song) | Apache 2.0 | Full songs with vocals |
| DiffRhythm 2 | — | Latent diffusion of full songs | Check | Speed |
| LeVo 2 (Tencent SongGeneration) | — | Hierarchical LM | Check | Song quality |
| Stable Audio Open 1.5 / Stable Audio 3 | ~1B | DiT over an audio VAE, T5 text encoder | Stability community | Samples, SFX, up to 47 s |
| MusicGen | 0.3-3.3B | LM over EnCodec | CC BY-NC (weights) | The classic baseline |

Recommended first: **ACE-Step 1.5 (turbo)**. It is MIT, it fits easily, it
is the strongest open song model of the reviews, and it reuses the most:
the Qwen3 dense decoder (its planner LM) and encoder (Qwen3-Embedding), the
DiT records of the image work, and a 1-D VAE decoder made of the same
convolutions (ENC_GATHER im2col) as MuQ and Whisper.

### 1.4 Text and image to video

| Model | Size | Architecture | License | bf16 VRAM | Notes |
|---|---|---|---|---|---|
| Wan 2.2 TI2V-5B | 5B DiT + umT5-XXL encoder + Wan2.2 VAE (16x16x4) | Flow-matching DiT, 3D RoPE, cross-attention to text; text and image to video at 720p 24 fps | Apache 2.0 | ~10 GB DiT; encoder on the CPU | The last open Wan (2.5 and later: API only) |
| Wan 2.2 A14B (T2V, I2V) | 2 x 14B experts (high and low noise) | MoE of two DiTs by timestep | Apache 2.0 | ~28 GB each: 8-bit and offload | Quality |
| HunyuanVideo 1.5 | 8.3B DiT + 1.3B VAE (16x spatial, 4x temporal) | DiT with sparse attention (SSTA) | Tencent community: excludes the EU, the UK and South Korea | ~17 GB | Ruled out: it excludes the EU, the UK and South Korea |
| LTX-2.3 | 22B (14B video + 5B audio streams) + Gemma 3 12B encoder | Dual-stream DiT, video and audio in one pass, HiFi-GAN vocoder | LTX community (free under $10M revenue, non-compete): not permissive | ~44 GB: 8-bit | Native audio with the video: its edge, if wanted |
| CogVideoX 1.5 5B | 5B | DiT, 3D VAE | CogVideoX license | ~10 GB | Older |

Recommended first: **Wan 2.2 TI2V-5B**. Apache 2.0, fits at bfloat16 with
its text encoder on the CPU, and its DiT is the image DiT plus 3D RoPE,
cross-attention, and a 3D causal VAE. Its umT5-XXL encoder is a T5, which
reclank does not have yet (section 3.2); its tokenizer is a SentencePiece
unigram like the XLM-R one in models/music/unigram.py.

## 2. What reclank already has

- **The ENC_* programs** (reclank/ir/program.py, reclank/encoders/runtime.py:
  _EncCompiler, _Runner, _Plain; docs/PROGRAMS.md). One program a step on
  the CPU (the C interpreter, a team in header word 3) or the GPU (the CUDA
  runner, a CUDA graph a program):
  - ENC_LINEAR, with BF12 matrices on the GPU through k_gemm_bf16_tc (x as
    two bfloat16 planes, float32 sums) and ENC_GATHER_MM (the product reads
    rows by index: a convolution without its im2col in memory);
  - ENC_GATHER (rows by index: the im2col of any convolution, 1-D, 2-D or
    3-D, strided or padded; nearest upsampling; reordering; concatenation);
  - ENC_ATTN_RANGE (each query over a range of keys: packing, causal masks,
    windows; GQA), ENC_ATTN, ENC_LOCAL_ATTN, ENC_ATTN_BIAS (an added bias:
    relative positions, shifted windows);
  - ENC_LNORM, ENC_RMS (with eps; per head for qk-norm), ENC_GELU (tanh or
    erf), ENC_GELU_MUL, ENC_SILU, ENC_SILU_MUL, ENC_GLU, ENC_ADD (scaled),
    ENC_MUL_VEC, ENC_BIAS_CLAMP, ENC_CLAMP, ENC_DWCONV (a causal depthwise
    conv), ENC_ROPE1D (RoPE by a position a row).
- **Towers built on them:** MuQ's conformer and Res2d convs, CLAP's HTSAT
  Swin (windows, patch merging, relative bias), RoBERTa, XLM-R, MuLan, the
  three towers of EmbeddingGemma 2, and the Whisper encoder in progress
  (reclank/models/whisper: a log mel and 32 pre-LN layers; float32 on the
  GPU matches transformers to 1 - cos 1e-6).
- **The decode engines:** Gemma 4 (12B, E4B, 26B), the Qwen3.5/3.6/3.8
  hybrids (Gated DeltaNet and full attention; LFM2 on the same classes),
  DeepSeek V4.1, GLM 5.3, Nemotron; KV caches, sampling (core/sampling.py),
  speculative decoding, the GPU step programs, the weightkeeper. There is no
  plain Qwen3 dense model and no T5.
- **Tokenizers:** Gemma 4, Qwen (byte-level BPE), XLM-R unigram
  (models/music/unigram.py), RoBERTa BPE (models/music/text.py).
- **Front ends:** the mels of MuQ, CLAP, Whisper and EmbeddingGemma 2
  (models/music/frontend.py, models/whisper/encoder.py), ffmpeg decode.
- **The servers:** server/embed.py (OpenAI /v1/embeddings for EmbeddingGemma
  2 and the music models, audio and image inputs) and the chat server
  (OpenAI chat with images, audio and video input).
- **The pattern of a port:** a float32 reference script in a venv
  (transformers or the authors' code) on real inputs, a check script (1 -
  cos, max difference), BF12 only by approval, and programs of n inputs whose
  rows come out with the bits of the program of one input.

## 3. What is new

### 3.1 Shared by all four areas

1. **Tensor-core attention.** ENC_ATTN_RANGE is a float32 kernel on the
   CUDA cores: 38 % of Whisper's encoder at 1500 rows. An image of 1024 x
   1024 at 16x16 per token is 4096 image tokens plus the text; Wan's 720p
   5 s is about 27,000 tokens; that is 50 to 300 times the work of a
   Whisper window per layer. A flash-attention kernel on the tensor cores
   (q, k, v as two 16-bit planes where precision needs it, float32 softmax
   and sums; the bits of the CPU are not the target) is the first new
   kernel: every area needs it. It keeps the ENC_ATTN_RANGE operands (lo,
   hi) so packing and masks stay.
2. **A Qwen3 dense model.** The text encoder of FLUX.2 klein and Z-Image
   (Qwen3-4B hidden states), the planner LM and the embedding model of
   ACE-Step, and the backbone of Higgs TTS 3 and Qwen3-TTS. The qwen family
   already has full-attention layers (with q/k norms); Qwen3 dense is those
   layers without the output gate, RoPE on the whole head, a dense SwiGLU
   MLP: a config of the existing classes (as LFM2 was), as an encoder program
   (hidden states of chosen layers) and as a decode engine.
3. **The sampler loop** (host, NumPy): flow-matching Euler (and the
   schedulers of each model: shifted sigmas, the turbo schedules), CFG as two
   rows of the same program (the conditional and unconditional halves of one
   batch), seeds and noise from NumPy so a check can feed the reference's
   noise. A step is one program run (on the GPU one graph).
4. **DiT records:**
   - ENC_MODULATE: x (1 + scale) + shift by a vector a group of rows (adaLN;
     the group is a sample of the batch, so CFG pairs and batches share a
     program), and the gated residual x += gate y;
   - the timestep embedding (sinusoidal, then an MLP: an ENC_LINEAR on one
     row) on the host or in the program;
   - RoPE over 2 or 3 axes: ENC_ROPE1D with a position array a section of
     the head (axial RoPE: time, height, width), or a new ENC_ROPE_ND.
5. **VAE records:** ENC_GNORM (GroupNorm: the mean and variance over groups
   of channels and all the rows of a sample); the convolutions are ENC_GATHER
   im2col and ENC_LINEAR (ENC_GATHER_MM on the GPU), causal 3-D convs with
   padding rows of zeros; nearest upsampling by ENC_GATHER. The memory of a
   VAE decode of a video is the risk: decode in tiles (rows of frames, or
   spatial tiles with overlap and a blend), as diffusers does.
6. **Weights:** safetensors in bfloat16 as stored. The GPU path for 16-bit
   matrices exists (k_gemm_bf16_tc with bfloat16 W, and the float16 path
   being added for Whisper); BF12 is a quantization and needs approval per
   model.

### 3.2 Image

- An MMDiT block (FLUX): two streams (text, image), each its own modulation,
  qkv, qk-norm and MLP; joint attention over the concatenation (ENC_GATHER
  to join and split); single-stream blocks after them (Z-Image: single
  stream all the way).
- The VAE decoder (2-D convs, ResNet blocks, GroupNorm, a mid attention).
- Patchify and unpatchify (2x2 latent patches): ENC_GATHER.
- Text encoders: Qwen3-4B (3.1.2). T5-XXL (umT5 for Wan, T5 for SD 3.5 and
  Stable Audio) is a new encoder of standard records (relative position
  buckets as an ENC_ATTN_BIAS, gated GELU), a unigram tokenizer that exists.

### 3.3 Video

- The image DiT plus: 3D RoPE, cross-attention to the text tokens (an
  ENC_ATTN_RANGE of q from the video rows over k, v from the text rows), the
  per-frame or per-token timestep of Wan 2.2 TI2V (the image frames at t =
  0 for image to video).
- The 3D causal VAE (Wan2.2-VAE 16x16x4): 3-D convs with causal time
  padding, decoded in chunks of frames with the cache of the causal convs
  between chunks (as the reference does), and tiles.
- Long attention: 27,000 tokens a layer at 720p 5 s; at 480p about 12,000.
  The first target is 480p and short clips; 720p after the attention kernel
  is fast.
- ffmpeg to write the frames (H.264) and to mux audio (LTX-2 later).

### 3.4 Speech

- The LM of codes: the Qwen3 dense decoder (3.1.2) with the model's text and
  audio embeddings (the sum of the codebook embeddings of a frame) and its
  output heads: Higgs predicts 8 codebooks with a delay pattern (8 heads, one
  frame a step); Qwen3-TTS predicts the first codebook in the talker and the
  rest in a small multi-token predictor (check the card). Sampling per
  codebook (temperature, top-p, repetition penalty): core/sampling.py.
- The codec decoders (codes to waveform): the Higgs tokenizer, the Qwen3-TTS
  12 Hz tokenizer. Codec decoders are convs (1-D, transposed: ENC_GATHER of
  zero-inserted rows), residual units, Snake activations (DAC, BigVGAN: a new
  elementwise ENC_SNAKE, x + sin^2(a x) / a), sometimes a transformer; the
  vector-quantizer lookup is an ENC_GATHER of codebook rows. Each needs its
  own reference.
- Voice cloning: the reference audio through the codec encoder (or the
  model's speaker encoder) once a voice; cache it.
- Streaming: the codec decoder on chunks of frames as the LM makes them (the
  chat server's streaming).
- Kokoro (if wanted): ALBERT (standard records), an LSTM duration and
  prosody predictor (a new recurrent record, or on the host: it is small),
  AdaIN in the decoder, ISTFTNet (convs, then an inverse STFT on the host),
  and the misaki or espeak phonemizer (a host dependency).

### 3.5 Music

- ACE-Step 1.5: the planner LM (Qwen3 dense decode, 5 Hz codes and text),
  Qwen3-Embedding for the prompt, the DiT (the image DiT's records over a
  1-D latent sequence with 1-D RoPE), the music VAE decoder (1-D convs; check
  its layout and whether it carries a vocoder), the turbo schedule (8 steps,
  no CFG) and the base schedule (50 steps with CFG).
- The output: 44.1 or 48 kHz stereo written by ffmpeg (WAV, MP3, FLAC).

## 4. The server

A generation server, server/generate.py beside server/embed.py (or routes of
it), the models loaded by flags, each on the GPU or the CPU:

- **POST /v1/audio/speech** (OpenAI): model, input, voice (a name, or a
  reference clip as base64 for cloning), response_format (mp3, wav, opus,
  flac, pcm), speed; stream_format sse for streaming. Synchronous: speech is
  faster than real time.
- **POST /v1/images/generations** (OpenAI): model, prompt, n, size,
  response_format b64_json (no URL hosting), plus seed, steps and guidance as
  extensions. Synchronous (seconds); /v1/images/edits later.
- **Video, async, in the shape of OpenAI's video API:** POST /v1/videos
  (model, prompt, an input image, seconds, size, seed) returns a job;
  GET /v1/videos/{id} its status and progress (steps done); GET
  /v1/videos/{id}/content the MP4; DELETE cancels. One job runs at a time on
  the GPU; the rest wait in a queue. The jobs and their files live in a work
  directory, not in RAM.
- **Music:** POST /v1/music/generations (prompt, lyrics, duration, seed,
  format) with the same job routes as video (/v1/music/{id}, its content);
  a short piece may answer synchronously with "wait": true.
- The GPU is shared with the embedding server and the chat servers: a model
  loads on demand and frees its memory after an idle time, or holds it by a
  flag. Video takes the whole GPU while it runs.

## 5. The checks

As for the encoders, with what diffusion adds:

1. A reference script for each model in its own venv (transformers,
   diffusers or the authors' code), float32 on the CPU, on real prompts,
   real voices and real images (the user's own clips and pictures), that
   saves the inputs and the intermediate tensors: the text encoder's output,
   the DiT's output at two or three timesteps for the same latent and noise,
   the VAE decode of a latent, the codec decode of real codes.
2. A check script: each part on the same inputs as the reference (the
   reference's noise and latents, not our sampler's), 1 - cos and the
   largest relative difference; then the whole pipeline from the same seed
   and noise, where the final image or audio is compared by its difference
   (diffusion amplifies differences over the steps, so the end-to-end check
   is looser than the parts' and is read beside them).
3. For the LMs: forced tokens and KL over the top logits at each position,
   as check_mm_prompt.py does, then the greedy codes.
4. Batched against single: the rows of a program of n samples (CFG pairs, a
   batch of prompts) with the bits of a program of one, as MuQ's batches.
5. Quantization by approval only: BF12 changed Whisper's encoder by up to
   0.10 (1 - cos) in rare rows, so each model's BF12 or 8-bit form is
   measured against its float32 reference before the user decides.

## 6. The phases

| Phase | Work | Rough effort | Gives |
|---|---|---|---|
| 0 | Tensor-core attention kernel (ENC_ATTN_RANGE operands); float16 matrices on the tensor cores (the Whisper GPU work) | 2-3 days | Whisper fast; everything below |
| 1 | Qwen3 dense: encoder program and decode engine; check against transformers | 2 days | The text encoder of FLUX.2 klein and Z-Image; the LM of the TTS models and of ACE-Step |
| 2 | TTS: Higgs TTS 3 (LM heads, the delay pattern, its codec decoder, ENC_SNAKE if it has Snake), /v1/audio/speech; then Qwen3-TTS on the same pieces | 3-4 days, then 2 | Speech in Czech, Slovak, English; cloning |
| 3 | Image: ENC_MODULATE, ENC_GNORM, axial RoPE, the FLUX.2 klein DiT, its VAE, the Euler sampler, /v1/images/generations; then Z-Image; then Qwen-Image-2.1 (its RGBA VAE, CFG, editing with Qwen3-VL's vision part) | 4-5 days, then 2, then 3-4 | 1024 x 1024 images in seconds; 2048 x 2048 and editing |
| 4 | Music: ACE-Step 1.5 (planner LM on phase 1, its DiT on phase 3's records, the 1-D VAE), the music job routes | 3-4 days | Songs with lyrics |
| 5 | T5 encoder (umT5-XXL) | 1-2 days | Wan's text; SD 3.5 and Stable Audio later |
| 6 | Video: Wan 2.2 TI2V-5B (3D RoPE, cross-attention, the 3D causal VAE in chunks and tiles), the video job routes, ffmpeg out | 5-7 days | 480p, then 720p clips |
| 7 | Larger models by approval of 8-bit forms: Wan 2.2 A14B and Qwen-Image-2512 (Apache 2.0); LTX-2.3 only if its audio with the video is wanted (its license is not permissive) | each 3-5 days | Quality |

Phases 1 and 0 can run in either order; phase 2 needs only 1, phase 3 needs
0 for speed but runs (slowly) without it.

## 7. Risks

- **Attention cost.** Without phase 0 a 1024 x 1024 image takes minutes and a
  video hours. With it, the 3090's ~70 TFLOPS of 16-bit (about 54 at the
  power cap) still makes 720p video a few minutes a clip.
- **VRAM.** The DiT, the text encoder and the VAE do not all fit at once for
  the larger models: run the encoder first (or on the CPU) and free it, decode
  the VAE in tiles. Video's activations at 27,000 tokens are a few GB a layer
  at float32: keep them 16-bit where the check allows, one layer at a time.
- **Steps.** The distilled forms (klein, Z-Image Turbo, ACE-Step turbo) need
  4-8 steps; the base forms 30-50 with CFG (twice the rows). Start with the
  distilled.
- **Licenses.** By the rule of the introduction: HunyuanVideo 1.5 is out
  (it excludes the EU, the UK and South Korea), and so is any model whose
  license carves out such regions; a sanctions clause alone is fine. Two non-commercial picks, each for its edge: Higgs TTS 3 (Czech and
  Slovak) and Qwen-Image-2.1 (the strongest open image model, editing,
  RGBA). Fish S2, Voxtral, FLUX.2 [dev] and the 9B klein have no edge over
  the permissive picks.
  LTX-2.3 (a revenue limit and a non-compete clause) only for its audio.
  Each license is read again from the model's own repository before a
  download; reclank ships code, and each model's docs say its weights'
  terms.
- **Precision.** Diffusion amplifies small differences over the steps; the
  checks compare the parts on the reference's own inputs first.
- **Codecs.** Each TTS family has its own codec, and a codec decoder is a
  model of its own to port and check. Prefer models whose codec is shared
  (Qwen3-TTS's tokenizer serves its whole family).
- **Moving targets.** New models arrive monthly; the shared pieces (attention,
  Qwen3 dense, the DiT and VAE records, the sampler) carry over to the next
  ones, which is why they come first.

## 8. Progress

- **Phase 0, part:** ENC_ATTN_X2 (the tensor cores, q k v and the weights as
  two bfloat16 planes: float32's accuracy) takes heads of 128 (dynamic shared
  memory): FLUX.2's attention. The products of bfloat16 checkpoints run as
  they are on the tensor cores (KQ_BF16: x as two bfloat16 planes, float32
  sums). Both run at the 3090's bound for that precision (about 35 TFLOPS of
  bfloat16 with float32 sums); x as one bfloat16 plane (as diffusers runs
  them) would be 2 to 3 times as fast, a change of precision that waits for
  approval.
- **Phase 1:** the Qwen3 dense model as an encoder program
  (models/qwen3/dense.py: hidden states of chosen layers, padding masked,
  GQA) and as step programs with a KV cache (models/qwen3/decode.py: the
  host gives each row's embedding and its RoPE table, ENC_SCATTER into the
  cache, ENC_ATTN_KV with GQA).
- **Phase 2, Higgs TTS 3:** models/higgs (tts.py, codec.py): the prompt,
  the multi-codebook head and embedding, the delay pattern's state machine
  and the sampler of sglang-omni (temperature 0.8, top-k 30, top-p 0.8;
  greedy at 0), RoPE rounded to bfloat16 as trained; the codec's decoder
  (RVQ and fc2 on the host, the DAC decoder as a program: transposed convs
  as phases of gathers, products and ENC_SCATTER; new ENC_SNAKE);
  POST /v1/audio/speech (mp3, opus, aac, flac by ffmpeg; wav, pcm). Against
  transformers float32 (higgs_reference.py, check_higgs.py; Slovak, English
  and Czech texts): the steps' logits with the reference's codes KL at most
  2.8e-7 (GPU), 2.6e-8 (CPU), no argmax different; greedy codes the
  reference's for all three; the codec's audio 84 to 91 dB SNR (GPU), 112 dB
  (CPU); Whisper transcribes the speech as its text. A step 16 ms on the
  3090: 6 s of speech in 2.5 s. Voice cloning: the codec's encoder
  (codec.py HiggsCodecEncoder: torchaudio's resample to 16 kHz on the
  host, models/audio/resample.py; HuBERT as a program, models/audio/hubert.py,
  the mean of its 13 hidden states; the semantic encoder with new ENC_ELU;
  the DAC encoder, models/audio/dac.py DacEncoder, its convs in tiles of
  rows, models/audio/conv.py; fc and the RVQ on the host) and the clone
  prompt (sglang-omni's build_prompt: the transcript, the voice's codes with
  the delay pattern as the sum of their codebooks' rows); a Voice keeps the
  codes; POST /v1/audio/speech takes "voice": {"audio", "text"} or a name
  of --voices (a cache of 32). Against transformers float32
  (higgs_clone_reference.py, check_higgs_clone.py; a radio clip of 8.3 s,
  codec only, and two of Higgs's own voices, Slovak and English): every
  stage of the encoder 1 - cos at most 2e-11 (CPU), 4e-9 (GPU); the codes
  the reference's on the CPU, one frame of 150 and of 156 different on the
  GPU (a near tie: 1e-4 and 5e-4 of the distance); the steps' logits with
  the reference's codes KL at most 3.4e-8 (CPU), 4.2e-8 (GPU), no argmax
  different; the greedy codes the reference's (CPU: of reclank's own voice
  codes too; GPU: of the reference's voice codes, its own differ from step
  21 or 27 by the flipped code); Whisper transcribes the cloned speech as
  its text. The encoder of a 6 s clip: 0.2 s (GPU), 0.7 s (CPU, 12
  threads). Qwen3-TTS is next to do.
- **Phase 3, FLUX.2 [klein] 4B:** models/flux2 (pipeline.py, dit.py, vae.py),
  new records ENC_ROPE_TAB (RoPE of interleaved pairs from a table: the
  axes of a DiT) and ENC_GNORM (GroupNorm over an image's pixels, SiLU
  fused); scripts/flux2_reference.py, check_flux2.py, flux2_generate.py;
  server/generate.py (POST /v1/images/generations). Against diffusers
  float32 on two real prompts at 512 x 512: the ids the same; the text
  encoder 1 - cos at most 1.4e-5 (GPU) and 3.5e-7 (CPU); each DiT step on the
  reference's inputs at most 1.7e-6 (GPU) and 1e-8 (CPU); the VAE's image
  within 5e-5 of [-1, 1]; the whole pipeline from the reference's noise a
  mean of 0.1 of a level of 255 (the text encoder's differences grow over
  the steps). 1024 x 1024 on the GPU: the text 0.2 s (1.6 s with its weights
  back on the GPU), a step 2.5 s, the VAE 2.3 s: about 13 s an image beside
  the embedding server.
- **FLUX.2 [klein] reference images (editing, image to image: the
  pipeline's image=):** each image as Flux2KleinPipeline's (load_image's
  EXIF orientation; an image over 1024 x 1024 pixels resized to that area by
  PIL's Lanczos; its sides floored to multiples of 16, cropped at the
  center), the VAE's encoder (flux2/vae.py: diffusers' Encoder as a program
  of the decoder's pieces, Downsample2D's conv an im2col of stride 2 over
  the zeros of its pad; its weights loaded at the first use; its runner
  vae_enc, held in generate's RESIDENCY scope until the references are
  done; on the GPU conv_in's 27 inputs as 72 with zero weights: the
  bfloat16 products take K of a multiple of 8), the posterior's mean, the 2 x 2 patches packed and
  the batch norm applied; the references' rows follow the noisy ones in the
  DiT's image stream at RoPE (10 + 10 i, h, w, 0), the velocity the noisy
  rows'; the output's size the first reference's unless given. No new
  records. flux2_reference.py --image, check_flux2.py (parts pre and enc),
  flux2_generate.py --image; server/generate.py POST /v1/images/edits
  (multipart: image or image[], 1 to 4 files of at most 50 megapixels; size
  "auto", the first image's; no mask). Against diffusers float32 on real
  images (a 1920 x 1200 photo, "make it night time", at its 1280 x 800; a
  1192 x 670 drawing, "turn it into a watercolor painting", cropped to 1184
  x 656; a 1795 x 2397 photo and a FLUX.2 image of 1024 x 1024 as two
  references at 512 x 512): the pixels the same bits (the resize included);
  the encoder's latents 1 - cos 6e-12 to 2e-11 (CPU), 5e-10 to 1e-9 (GPU),
  4e-8 for the drawing on both (diffusers' own encoder of the same pixels
  outside the pipeline differs from its run inside by as much; reclank's
  CPU from the one outside by 1e-11); the RoPE tables of the pipeline's ids
  the same; each DiT step at most 2.8e-8 (CPU), 1.9e-5 (GPU; 2.3e-6 with
  one reference); the whole pipeline from the reference's noise PSNR 76.4,
  67.7 and 69.1 dB (CPU, at most 2 levels of 255), 66.3, 63.2 and 59.5 dB
  (GPU, at most 10 levels, a mean of 0.06 or less), where text to image lands (the two prompts
  at 512 x 512: 64.9 and 68.3 dB on the CPU, 55.8 and 60.9 dB on the GPU).
  On the 3090 at 1024 x 1024 (4 steps): text to image 17.7 s, a step 2.0 s;
  one reference of 1280 x 800 (8608 rows) 31.6 s, the encoder 3.9 s, a
  step 4.5 s; two references (880 x 1168 and 1024 x 1024, 12719 rows) 50.6
  s, the encoders 9.9 s, a step 7.8 s; the GPU's peak 16.0, 16.9 and 17.9
  GB. Inpainting (Flux2KleinInpaintPipeline) is not done: it needs the mask
  (its processor, bilinear to the token grid), the source's latents noised
  to the first sigma of strength's steps, after each step the unmasked
  tokens put back (the source noised to the next sigma), and optional
  references as here.
- **Phase 4, ACE-Step 1.5 (turbo) without its 5 Hz LM:** models/acestep
  (pipeline.py, dit.py, vae.py; models/audio/dac.py shared with the Higgs
  codec): the caption and metadata in SFT_GEN_PROMPT through Qwen3-Embedding
  (qwen3.dense with the final norm), the lyrics' embedding rows through the
  lyric encoder, the timbre encoder of the silence latent, the packed
  conditions and each layer's cross keys once a request; 8 Euler steps (t
  = 1 - i / 8) of the DiT with DCW "double" (Haar) as the authors' default;
  the Oobleck VAE in the authors' overlap-discard tiles; the peak at -1 dB.
  POST /v1/music/generations. Against the authors' pipeline in float32
  (acestep_reference.py: their generate_music, thinking off; check_acestep.py):
  the ids the same; the text encoder 1 - cos at most 7.5e-10; each step's
  velocity at most 3.7e-7 (GPU), 5.9e-9 (CPU); the latents of the whole
  pipeline from the reference's noise 3.5e-6 (GPU), 5.2e-8 (CPU); the final
  audio 77.9 dB SNR against the authors' 16-bit WAV. On the 3090 a step 0.1
  s: 90 s of music in 9.7 s.
- **Phase 4, the planner LM ("thinking", the authors' default):**
  models/acestep/planner.py: acestep-5Hz-lm-1.7B as qwen3.decode step
  programs; phase 1 the metadata under the authors' constrained decoding
  (the fields' names forced, their values from prefix trees, the caption
  free until the model names the next field), phase 2 the 5 Hz codes with
  CFG 2.0 (the unconditional prompt as the second sequence), exactly 5 a
  second; the "pt" backend's sampling (top-p 0.9, then temperature 0.85).
  The codes through FSQ and the detokenizer (dit.py) become the DiT's source
  latents, with the cover instruction and the planner's caption and
  metadata. lm_language (an option, off as the authors' code): the planner's
  language for the lyrics. Against the authors' LM in float32
  (acestep_reference.py --thinking, check_acestep_lm.py; the reference's
  tokens fed): the prompts the same; KL at most 4.6e-9 (phase 1) and 3.1e-9
  (phase 2, both rows), no argmax different; every reference token allowed
  by reclank's constraint; the metadata and the 150 codes the same; the
  hints 1 - cos 1.3e-9; the DiT's velocities 7.5e-7. An LM step 16.5 ms on
  the 3090: a 30 s piece in about 15 s with the plan.
- **Stable Diffusion 1.5 and 2.0 (base 512, 768 v-prediction):** models/sd
  (tokenizer.py: CLIP's BPE with "</w>", SD 2's pad "!" an added token;
  clip.py: CLIPTextModel, quick_gelu as SiLU(1.702 x) / 1.702; unet.py:
  UNet2DConditionModel, the negative's and the prompt's latents as one batch,
  GroupNorm an image, the up blocks' torch.cat as a gather of rows of 320
  channels, 3 x 3 convs as im2col and ENC_LINEAR; scheduler.py: PNDM (PLMS)
  and DDIM as diffusers in float32, the betas through torch's linspace;
  pipeline.py), the VAE flux2.vae (AutoencoderKL: no batch norm, the old
  names of the mid attention). No new records. GPU ENC_LINEAR of float32 W:
  k_enc_sgemm (tiles of 128 x 128, 64 x 128 or 64 x 64, the same bits as
  k_enc_gemm, RECLANK_GPU_ENC_SGEMM=0 for the old kernel), about 1.8 x.
  scripts/sd_reference.py, check_sd.py; server/generate.py --sd [ID=]DIR.
  Against diffusers float32 (real prompts, 25 steps, guidance 7.5): the ids
  the same; the text encoder 1 - cos at most 8e-11 (GPU), 1e-11 (CPU); the
  scheduler the same bits; each UNet call on the reference's inputs at most
  1e-10 (SD 1.5, 2.0-base, GPU), 3e-12 (CPU), 1.2e-7 for 2.0-768 on the GPU
  (ENC_ATTN_X2's two bfloat16 planes over 9216 keys; 4e-10 with
  RECLANK_GPU_ENC_X2=0, 1.7 x the step); the VAE's image within 3e-5; the
  whole pipeline from the reference's noise at most 3 levels of 255 (mean
  0.02). On the 3090 a step (the batch of 2) 0.19 s at 512, 0.39 s at 768:
  an image in 5 s and 10 s. The CPU (12 cores) 4.4 s a step at 512: the
  attention (heads of 40) and the products, about 2.5 x torch's.
- **Phase 3, Qwen-Image-2.1 (text to image):** models/qwen_image
  (pipeline.py, dit.py, vae.py): the pipeline's raw template (a system
  message, dropped after the encoder) through Qwen3-VL-8B's language model
  (qwen3.dense with the VL text config; a text alone needs no vision tower)
  on the CPU in its bfloat16 rows (KQ_BF16X16: float32 products; 15 GB, which
  does not fit on the GPU beside the DiT), its last layer before the final
  norm; the DiT as two programs: the text's keys and values once an image
  (causal_condition: the text rows modulated at t = 0, causal among
  themselves, the pipeline's use_kv_cache) and a step of the image's rows
  over [image, text] keys (ENC_ATTN_X2), the shared modulation (scale and
  tanh gate, no shift) on the host; 3-axis RoPE of the pipeline's positions
  (the image centered, at frame T); the scheduler's shifted and stretched
  sigmas; true CFG with a negative prompt (off by default, as the
  pipeline); the 64-channel RGBA VAE's decoder in float32 as stored (new
  ENC_DUPUP: DupUp3D's shortcut of one frame; new ENC_ATTN_WIDE, GPU: the mid
  block's attention of one head of 1152, its scores in parts of the queries,
  float32; on the GPU the norms run on the conv tiles' gathered pixels).
  cuda/runtime.py SKIP: the encoders' count operands are not translated as
  addresses (a count of a 2048 VAE fell in a heap array's host range: a
  launch of "invalid argument"). scripts/qwen_image_reference.py (the
  pipeline in float32, a tokenizer-only processor: the venv has no
  torchvision), check_qwen_image.py, qwen_image_generate.py (RGBA PNG);
  server/generate.py --qwen-image ("qwen-image-2.1"; "background":
  "transparent" takes the model card's RGBA prompt form). Against diffusers
  float32 on two real prompts at 512 x 512, 20 steps (and a CFG one, 6 steps
  at scale 4): the ids and the dropped tokens the same; the text encoder 1 -
  cos 1.6e-10; each DiT step on the reference's inputs at most 3e-7 (GPU; 1.3e-6
  with CFG) and 1.5e-9 (CPU; 9.6e-9 with CFG); the VAE's image 1 - cos 2e-13,
  at most 1 level of 255 (GPU and CPU); the whole pipeline from the
  reference's noise: GPU at most 4 levels, a mean of 0.004 to 0.007; CPU at
  most 1 level, a mean of 0.002 or less (CFG of scale 4 in 6 steps magnifies
  a step's differences: its latents 1e-2 (GPU) and 2e-3 (CPU) of their
  largest value, the GPU's image a mean of 0.3 levels). On the 3090: a step
  0.70 s (512), 3.03 s (1024), 9.0 s (1536), 20.2 s (2048); the VAE 0.7 s,
  3.6 s, 7.3 s, 13.1 s; the prompt 1.4 to 2 s on 12 CPU threads: a 1024 x 1024
  image of 40 steps in about 2.2 min. GPU memory 14.1 GB at 512, 14.6 GB at
  1024, 15.4 GB at 1536, 16.4 GB at 2048 (the DiT's 14 GB of bfloat16 weights
  stay; the VAE runs after they leave). Editing: the next entry.
- **Phase 3, Qwen-Image-2.1 editing (reference images):** as the pipeline's
  image= (up to 10 references; masks and painted annotations are reference
  images too, the prompt saying what they mark: the pipeline has no mask
  input). Each image RGBA, resized (Lanczos) to the area output_resolution^2
  at its aspect ratio (sides of 32), the output's size by default the last
  one's. The vision tower (qwen.vision.QwenVision of the text encoder's
  checkpoint, now with Qwen3-VL's deepstack mergers: the LayerNorm of each 2 x
  2 block's 4608 values after blocks 8, 16 and 24, mm.0, gelu, mm.2;
  encode_deepstack) of the image composited over white, the processor's
  rescale (float64) and normalization (float32) in the order of the merge
  blocks (the pipeline's sizes are the processor's smart_resize's: no second
  resize). The edit template (<imageK><|vision_start|>, an <|image_pad|> a
  token of 32 x 32 pixels, <|vision_end|>, the images joined by a space,
  before the prompt) through the text encoder (qwen3.dense's media: the vision
  rows in place of the image tokens' embeddings, Qwen3-VL's interleaved MRoPE
  (24, 20, 20) of get_rope_index's positions as cos and sin tables,
  ENC_ROPE_TAB's NEOX halves; the deepstack rows added to the image tokens
  after layers 0, 1, 2). The VAE's encoder (vae.py, the decoder's buffers and
  convs as a builder shared with it): conv_in of the 4 channels, five down
  blocks (two residual blocks; a 3 x 3 conv of stride 2, padded at the bottom
  and the right; AvgDown3D's shortcut of one frame as a linear of the 2 x 2
  gathered pixels: the zero frame in front of a temporal block makes half its
  outputs 0), the mid block, conv_out, quant_conv's mean rows, less the
  latents' mean, over their std. The DiT's prefix (dit.py) the transformer's
  joint sequence: the text rows past the system message with each image's
  slots in place of its latents' tokens (a slot is 2 x 2 tokens: img_in of
  them in place of txt_in's rows), each image a block that sees the rows
  before it and its own (block causal), modulated at t = 0 with the text,
  positions as QwenImage21Rope (a block at the frame reached, centered, the
  frame then + max(h, w)); the steps as text to image, the target at the
  frame after the prefix; the prefix program leaves the GPU once the steps'
  program holds the cache (the same bits). On the GPU the vision tower's
  bfloat16 matrices as KQ_BF16 and its attention float32 (ENC_ATTN_TILE):
  QwenVision's float16 tensor-core forms (x and q, k, v) put its rows 1.5e-3
  (1 - cos, worst, at 1024) off diffusers', these 4.4e-6 (8e-7 at 512;
  float32 matrices would give the CPU's 4e-8 at 2.3 GB, not 1.2: not done;
  QwenVision's default, the LLM server's, is unchanged); the vision tower and
  the VAE encoder are runners of the request's RESIDENCY scope, done after
  their turn (evicted when the DiT needs their room; a script that edits on
  the GPU turns the manager on). The DiT's cache of an edit's prefix (its
  keys and values) is F20 by default, the user's choice for the memory: new
  records ENC_F20_PACK (292: float32 rows to F20 rows, the bits of
  kq_f32_to_f20; on the GPU a warp a group of 32) and ENC_F20_UNPACK (293:
  kq_f20_to_f32's bits); the prefix computes a block's keys and values in
  float32 (its own attention reads them) and packs them, each step unpacks a
  block's into its text keys and values: 0.63 MB a row of the prefix
  (float32's 1 MB). cache_f20=False (QwenImage21) or RECLANK_QWEN_CACHE_F20=0:
  the float32 cache. A text's cache (text to image: the same programs) stays
  float32. scripts/
  qwen_image_reference.py --image (a Qwen3VLProcessor of the PIL image
  processor: the venv has no torchvision; the pipeline's steps written out,
  the processor's pixels, the vision tower's and deepstack rows, the MRoPE
  positions, the VAE's inputs and latents saved; --text-only),
  check_qwen_image.py (parts pixels, vision, text, venc, dit, vae, all on an
  edit's file), qwen_image_generate.py --image, server/generate.py POST
  /v1/images/edits. Against diffusers float32 on real photos (a 1795 x 2397
  cat to a sunset beach; the cat and a 1024 x 1024 mountain landscape, two
  references; a 1920 x 1200 bridge at night; an RGBA sticker with
  transparency recolored, its transparency kept) at output_resolution 512, 20
  steps, and the text side at 1024 (one and two references, 1071 and 2114
  tokens): the resized sizes, the VAE's inputs and the processor's pixel
  values the same; the ids, the MRoPE positions and the image slots the same;
  the vision rows 1 - cos at most 5e-8 (512) and 1.7e-7 (1024) on the CPU,
  8e-7 and 4.4e-6 on the GPU (a mean of 1e-8 to 3e-8), the deepstack rows
  less; the text rows the DiT reads (the image slots' rows are replaced) at
  most 1.7e-9 (CPU, 1024), 5.6e-8 (the GPU's text encoder, 512 and 1024), the
  text model alone on the reference's vision rows 2e-10 (CPU); the VAE
  encoder's latents at most 1.8e-9 (CPU), 7.2e-9 (GPU; rel at most 8e-5);
  each DiT call with the references 1 - cos a token at most 5.8e-10 (CPU), a
  mean of 1e-10 to 1e-9 on the GPU (one step's worst token 5.7e-6); the whole
  pipeline from the reference's noise and images: CPU at most 1 level of 255,
  PSNR 81.5 (cat), 80.3 (two references), 82.6 (RGBA), 85.3 dB (bridge); GPU
  at most 2, 5, 1, 1 levels, a mean of 0.002 to 0.004, 73.3, 71.6, 73.3, 75.3
  dB (text to image on the GPU: 71.8 and 69.5 dB). With the F20 cache (the
  default; the same bits as the float32 cache rounded through F20 after the
  prefix, CPU and GPU), PSNR / at most levels against diffusers float32: CPU
  78.5 / 1 (cat), 71.0 / 7 (two references), 81.4 / 1 (RGBA), 83.3 / 1
  (bridge); GPU 73.6 / 1, 69.5 / 6, 73.0 / 1, 75.2 / 1; its image against the
  float32 cache's 79.6, 70.7, 84.6, 84.3 dB (CPU), 78.5, 72.9, 82.0, 83.8 dB
  (GPU): near float32's on three, the two-reference case (a prefix of 2072
  rows) 9 dB lower on the CPU (at most 7 levels) and 2 dB on the GPU. A
  bfloat16 cache (measured, not kept) 50.6 to 62.8 dB, at most 7 to 54
  levels; diffusers' own bfloat16 pipeline (the model card's dtype) 41.4 dB
  (at most 78 levels) against its float32 on the cat. Text to image unchanged:
  check_qwen_image's cases the same numbers, and the text rows, DiT steps and
  VAE image of the tree before these changes the same bits (CPU and GPU).
  On the GPU the F20 cache is in host memory, the user's choice (dit.py
  host_cache: registered with CUDA, one buffer of each prompt that only
  grows; RECLANK_QWEN_CACHE_HOST=0, or no tensor cores: on the GPU): the
  prefix packs a block's keys and values into two staging rows and
  GP_TO_HOST copies them to the host (the prefix waits for its copies); a
  step fetches block j's into slot j % 2 of two slots on the GPU (GP_FETCH,
  as Wan A14B's streamed blocks: block j + 2 after FETCH_DONE of block j,
  overlapped with the blocks before) and the new record ENC_ATTN_X2C (294,
  GPU) attends the image's queries alone over the image's keys and the
  slot's, the F20 rows decoded straight into the attention's bfloat16 planes
  (k_f20_to_bf16; ENC_ATTN_X2's kernel takes the planes of k and v apart from
  q's: its own calls the same bits). So a step has no rows of the text's
  queries, float32 keys or values (it had them: ENC_ATTN_X2 takes q, k and v
  of the same rows, and they were about half its attention's work with a 1024
  reference). The same bits as the cache on the GPU (the cat and the
  two-reference edits, and the four cases' numbers the same; with
  RECLANK_EVICT_ALWAYS=1 too); text to image the same bits as before. On the
  3090 at output_resolution 1024: one reference (the cat, 896 x 1184, 4144
  tokens; the output its size) the vision tower 1.4 s, the text encoder 31 s
  on 12 CPU threads (5.4 s on the GPU with text_gpu, its 15 GB uploaded), the
  VAE encoder 2.2 s, the prefix 9.5 s (to the GPU's end: its program's build,
  its run and the copies to the host, 0.55 s of the run at two references;
  the cache's first registration 3 s for 5.5 GB), a step 4.1 s (the cache on
  the GPU 5.3 s; text to image 3.0 s), the VAE 3.8 s: 40 steps in about 4
  minutes; two references (the cat and the landscape, 8240 tokens; 1024 x
  1024 out) the text encoder 60 s (CPU), the prefix 15.6 s, a step 4.6 s (on
  the GPU 8.2 s): about 4.5 minutes; three (and the bridge, 12240 tokens;
  1280 x 800 out) the prefix 22 s, a step 5.2 s (12.0 s); six (24400
  tokens; 1280 x 832 out) the prefix 45 s, a step 7.5 s. The GPU's peak
  (cudaMemGetInfo, the other parts evicted: RECLANK_EVICT_ALWAYS=1; the
  prefix's buffers now the largest) 14.4 GiB with one reference, 14.7 GiB
  with two, 15.5 GiB with three, 17.7 GiB with six: about 0.19 MiB a reference
  token on 13.3 GiB (ten 1024 references about 21 GiB; the server's limit
  from these); the cache on the GPU 17.2, 20.1, 23.0 GiB with one, two, three
  (0.74 MiB a token), float32 19.1 and 23.2 GiB with one and two. The host's
  RSS 35, 39, 41, 50 GB (the cache 0.63 MiB a token, pinned). The host
  caches are unregistered from CUDA before their memory goes (SeedVR2-7B's
  lesson: a range left registered after its unmap fails the copies of the
  memory mapped there next): QwenImage21.close() (QwenImageDiT.close(): the
  runner's programs, host ones too, then rc_gpu_sync and the unregistering),
  else the DiT's finalizer; RESIDENCY's eviction keeps them registered (the
  next request's registration 3 s for 5.5 GB). In one process with the
  server's classes (RESIDENCY on): a Qwen edit, the model closed, a FLUX.2
  edit, a Qwen edit, the model dropped without close() (its runners released
  as an eviction does), FLUX.2 again, a third Qwen edit: the three edits the
  same bits, the two FLUX.2 edits the same bits. The X2 users the same bits
  as before (ENC_ATTN_X2's kernel takes k's and v's planes apart from q's):
  the kernel on the real dumps of Wan 480 (self and cross), Z-Image 1024 and
  Whisper's masked heads; SD 1.5's CLIP rows and a 4-step image; Whisper
  large-v3's encoder of two real clips and the long form of a 250 s one (its
  tokens); Hubert (Higgs's codec encoder of a real clip: its hidden states
  and codes); FLUX.2's edit and generation PNGs.
- **Qwen-Image's text encoder in BF12 (the default; the user's choice):** the
  language model's matrices (q, k, v, o, gate, up, down of its 36 layers,
  13.9 GB of bfloat16) in BF12, 10.6 GB (encoders.runtime.to_bf12_dev:
  to_bf12's rule, the checkpoint's pages dropped): on the CPU KQ_BF12X16 (new
  ENC_LINEAR wbf 4: float32 products of the BF12 values), on the GPU (text_gpu)
  BF12 rows as KQ_LINEAR's. The vision tower's matrices BF12 on the GPU only
  (its MLP's down matrices of 4304 columns stay bfloat16; on the CPU
  KQ_BF12X16's decode made the tower slower, 58 to 129 s against 17 to 34 s
  of a 1024 reference, for 0.2 GB); the norms, biases, the patch embedding
  and the embedding table as before; the LLM server's Qwen3 and QwenVision
  unchanged. text_bf12=False (QwenImage21) or RECLANK_QWEN_TEXT_BF12=0: the
  bfloat16 rows, the same bits as before (GPU text to image and an edit).
  The conversion an OpenMP team's: beside a GPU (OMP_NUM_THREADS=1) the main
  thread's ICV is team_threads() for it alone, then 1 again (83 to 86 s on
  one thread; 5.5 to 12.5 s so, the CPU's load 15 s; bfloat16 1.4 to 2.8 s).
  The team's 11 threads stay idle after it: at most one tick of CPU each
  during an edit (their spin after the load), none in a FLUX.2 generation of
  the same process; the edit's steps the same time (4.00 s) and bits as with
  the conversion on one thread; a new thread's ICV 1. Against diffusers
  float32 (bfloat16 beside it): the text rows of text to image 1 - cos 2.4e-10
  (1.6e-10) on the CPU, 5.1e-8 (4.7e-8) on the GPU; an edit's text model on
  the reference's vision rows at most 8.7e-7 (4.3e-7) at 512 and 1.1e-6
  (6.7e-7) at 1024 (CPU); the rows of the whole text side (our vision rows):
  the two-reference case at 512 1.7e-4 (1.8e-4; with a BF12 tower 1.35e-3),
  the cat 1.4e-6 (1.45e-6); on the GPU the two-reference case 4.3e-2 (4.1e-2),
  the vision rows 4.37e-6 (4.34e-6) at 1024. The whole pipeline (512, 20
  steps; PSNR, at most levels of 255): CPU cat 78.5 dB, 1 (78.5, 1), two
  references 72.3, 4 (71.0, 7), dragon 82.5, 1 (81.4, 1), bridge 83.2, 1
  (83.3, 1); text to image fox 81.3, 1 (82.0, 1), sign 75.5, 2 (75.9, 1), CFG
  59.6, 7 (57.5, 10); GPU (the text encoder there) 73.5, 2 (73.6, 1), 71.4, 5
  (69.5, 6), 73.0, 1 (73.0, 1), 76.0, 1 (75.2, 1); fox 71.9, 1 (71.9, 1), sign
  64.2, 10 (64.2, 10), CFG 43.4, 58 (41.6, 74). Memory: the GPU's text phase
  12.2 GiB (15.4) with a 1024 reference, 12.9 (16.1) with two; the edit's
  peak 16.8 (17.7) and 17.1 (18.4) GiB; the host's RSS of the CPU text side
  15.9 GB (32.7: the bfloat16 pages and their x16 copy), of a CPU text to
  image 43.4 GB (60.1). Speed: the GPU's text encoder 7.5 s (9.7) with a 1024
  reference, 9.3 s (11.8) with two; the CPU's 31 to 53 s (55 to 63) and 67 to
  85 s (107 to 124) on 12 threads (a busy machine); the steps unchanged.
- **F20, float32 checkpoints in 63% of their memory:** a format of 16
  significant bits a value (formats.gguf F20 = 68; backends/cpu/csrc/kernels/kq
  formats.c): BF12's groups of 32 with a 15-bit mantissa, a uint16 of the sign
  and mantissa a value (rounded to nearest, even), its exponent's gap to the
  group's largest (4 bits) and that largest exponent (a byte), 20.25 bits a
  value; a value more than 15 binades under its group's largest is 0 (BF12's
  rule and zero code). The checkpoints are float32 in earnest (Wan 2.2, SD 1.5,
  SD 2.0: bfloat16-exact 0.002% or less, every low bit used), and F20 changes a
  matrix by 6.5e-6 (norm), bfloat16 by 1.7e-3. The products are float32's:
  ENC_LINEAR's wbf 3, on the GPU k_enc_sgemm (and k_enc_gemm) decoding F20 rows
  as they load W's tiles, on the CPU kq_x16f_body decoding KQ_F20X16 (F20 in
  groups of 16 rows, 1296 bytes a block of 32 columns) column by column: each
  output the same bits as float32 W of the decoded values (scripts/check_f20.py
  on real matrices of SD's UNet: the C encoder the rule's bits, the GPU's two
  kernels, the CPU's AVX-512, AVX2 and scalar tiers). encoders.runtime.to_f20
  / F20Lins / f20_lins: a linear's F20 (the rule of to_bf12 refuses a matrix of
  cols not a multiple of 32), its float32 W dropped as its F20 is made, the
  checkpoint's mapped pages of it too (formats.safetensors.drop_pages). By
  choice, as the weights change: SDUNet / StableDiffusion f20, check_sd.py
  --f20, server/generate.py --f20. SD's UNet 3.32 -> 2.11 GB (273 matrices;
  conv_in's 36 columns stay float32). Against diffusers float32 (25 steps,
  guidance 7.5), the whole pipeline from the reference's noise: SD 1.5 a mean
  of 0.015 levels of 255, at most 2 (PSNR 66.2 dB; float32 0.008, 68.9 dB) on
  the GPU, 0.007 (69.5 dB) on the CPU; SD 2.0-768 0.065, at most 8 (59.8 dB;
  float32 0.019, 65.2 dB; BF12 7.1, 25.6 dB, another composition). Each UNet
  call 1 - cos <= 1.7e-11 (SD 1.5) and 2.2e-7 (2.0-768, as float32's 1.2e-7 of
  ENC_ATTN_X2). A UNet call on the 3090 0.197 -> 0.217 s (SD 1.5), 0.433 ->
  0.478 s (2.0-768): the decode's integer work beside the products of the
  128-row tiles; on the CPU (47 threads) 2.16 -> 2.24 s, within the noise. The
  CPU's memory: peak 10.2 -> 3.6 GB (float32 W kept also as KQ_F32X16, and the
  checkpoint's mapped pages; F20 holds only KQ_F20X16). For Wan 2.2's 20 GB DiT
  (12.7 GB in F20) and Z-Image's 24.6 GB (15.6 GB; its GPU budget 17 GB of
  weights).
- **umT5-XXL, Wan 2.2's text encoder:** models/t5 (encoder.py, tokenizer.py):
  UMT5EncoderModel as a CPU program: T5's RMS norm, attention with no scale and
  each block's own relative position bias (32 bidirectional buckets, the
  logarithmic ones up to 128, in transformers' float32 arithmetic), gelu_new
  (tanh) gated FFN, the final norm; new ENC_ATTN_REL (290, CPU: a query over
  its range of keys plus a head's bias by the distance, blocks of 32 queries a
  head over the transposed keys). The weights the checkpoint's bfloat16 rows
  (KQ_BF16X16, made at load from the file's map, its pages dropped: 9.4 GB of
  RSS, 10.3 GB at the peak). UMT5Encoder.encode is WanPipeline's
  _get_t5_prompt_embeds: prompt_clean (no ftfy, as the venv), transformers'
  T5Tokenizer (the added tokens split out, WhitespaceSplit, Metaspace: not
  tokenizer.json's own pipeline, which keeps a lone U+2581 before an added
  token), at most 512 tokens with </s>, the last hidden state of the tokens
  and zero rows to 512. The padding rows do not change the tokens' rows, so a
  prompt runs in the rows of its tokens rounded to 16 (the same bits as 512
  rows and as a program of other prompts); prompts of the same rows share a
  program. Against transformers float32 (umt5_reference.py, check_umt5.py;
  the fox and waves prompts, Wan's Chinese negative prompt, a Slovak prompt, a
  long English prompt of 322 tokens): the ids the same (and of 11 texts more:
  white space, HTML entities, scripts, added tokens, truncation at 512); the
  buckets of every distance the same; each block 1 - cos at most 1e-10, the
  prompt_embeds at most 1e-9 (mean 1e-11 to 5e-11). On 12 CPU threads: a
  prompt and the negative prompt 2.9 s; two prompts of 512 tokens 21.6 s
  (torch float32 about 6 s a prompt of 512 rows, which the pipeline pays for
  every prompt).
- **Phase 6, Wan 2.2 TI2V-5B text to video:** models/wan (pipeline.py,
  dit.py, scheduler.py, vae.py) as diffusers' WanPipeline (expand_timesteps:
  a timestep a token, all the same without an image). The prompt and the
  negative prompt (the model card's) through umT5-XXL on the CPU (models/t5);
  the DiT as two programs: each block's cross-attention keys and values of
  both texts once a video, and a step of the video's rows (patches of 1 x 2
  x 2 by a linear, the modulation vectors of the timestep on the host, the RMS
  norms of q and k across the heads, 3-axis RoPE of 44, 42 and 42 values,
  the attention in float32 on the GPU (ENC_ATTN_TILE; x2 or RECLANK_WAN_X2=1:
  ENC_ATTN_X2), the cross attention over the 512 text rows, the
  tanh GELU FFN in parts of 4096 rows); CFG 5 as two calls a step. The DiT's
  304 matrices in F20 on the GPU (19.72 -> 12.48 GB), float32 as stored on
  the CPU (F20 by choice). UniPC (bh2, order 2, flow sigmas, shift 5) on the
  host in float32 as torch's: torch.linalg.solve written out as MKL's sgesv,
  expm1 as Sleef's expm1f_u10; log the float64 value rounded (torch's is
  MKL's vmsLn, not open: 11 of the 10100 values of 1 to 100 steps an ulp
  apart, so 49, 95 and 98 steps differ in a corrector weight by at most
  2.2e-4 of itself; 25 and 50 steps the same bits). The VAE decoder
  (AutoencoderKLWan, 48 channels, 16 x 16 x 4) in float32 as stored: a
  program for the first latent frame and one for each later one, as
  diffusers' decode a latent frame at a time; each causal conv's input in a
  buffer of its two cache frames and the chunk's frames (the cache moves to
  the front after the chunk); time_conv's first chunk skipped ("Rep");
  DupUp3D by ENC_DUPUP; the mid attention by ENC_ATTN_WIDE. No new records.
  NumPy's normal noise of the seed (not torch's). scripts/wan_reference.py,
  check_wan.py (parts sched, dit, vae, all), wan_generate.py (an MP4 by
  ffmpeg: H.264, yuv420p, 24 fps); server/generate.py --wan DIR: the job
  routes of section 4 (POST /v1/videos, GET /v1/videos[/{id}[/content]],
  DELETE /v1/videos/{id}: cancels a queued or running job, or deletes a
  video), one job at a time in a queue, the jobs and MP4s in --video-dir.
  Against diffusers float32 (480 x 832, 17 frames, 25 steps, CFG 5; the fox
  and waves prompts): the sigmas and timesteps the same bits (1 to 100
  steps); each of the 25 scheduler steps on the reference's outputs the same
  bits; each DiT call on the reference's inputs (steps 0, 12, 23) 1 - cos a
  token a mean of about 1e-12 (CPU float32) and 3e-11 (GPU F20), the worst
  token 2e-9 (CPU) and 3.5e-8 (GPU), the guided output's 2.9e-8 and 6.7e-7
  (step 23); the relative norm 1.3e-6 to 2.6e-5 (CPU), 8e-6 to 8.6e-5 (GPU);
  the VAE's video of the reference's latents at most 1 level of 255 (PSNR
  89-91 dB) on the GPU and the CPU. The whole pipeline from the reference's
  noise: the final latents 1.4% (fox) and 0.55% (waves) off float32 in norm,
  the video a mean of 0.35 and 0.12 levels (PSNR 50.2 and 56.5 dB) on the GPU
  (F20, the attention in float32; ENC_ATTN_X2: 2.9% and 1.7%, 0.61 and 0.46
  levels, 42.7 and 48.3 dB); the CPU in float32 (fox) 2.1%, a mean of 0.44 levels, PSNR 47.9
  dB: float32 in another order of sums moves the final latents by about 2%
  (a call's differences of 1e-6 to 1e-5 grow over the 25 steps about 1000
  x), and F20 adds little to that. The study's bf16 left the latents 8-14%
  off (PSNR 28-32 dB, a mean of 3-3.6 levels). A server request (a prompt of
  29 tokens through reclank's umT5, NumPy's noise of seed 7, 20 steps) gave a
  smeared video; diffusers in float32 on the same inputs gives the same one
  (the latents 0.9% apart, PSNR 51.1 dB): the model's sample, not the port. On the 3090: a DiT call 2.07 s at 480 x 832 x 17 (1950
  tokens; ENC_ATTN_X2 2.06 s), a step 4.2 s, the VAE 22 s (17 frames): a clip of 50 steps in
  about 4 min, the umT5 prompts 2.9 s more (its load 50 s at the first). 720p
  (704 x 1280, 121 frames, 27280 tokens): a DiT call 47.4 s (ENC_ATTN_X2 45.4 s), a step 95 s (50
  steps about 80 min), the VAE 267 s; the GPU's peak 16.9 GB in the steps, 22.6 GB in the
  VAE (its buffers 19.8 GB: they do not depend on the frames), so a 720p
  video needs the whole card. The CPU (12 threads): a DiT call 47 s, the VAE
  254 s (17 frames, 17.6 GB of RSS).
- **Whisper large-v3, the speech-to-text route:** server/generate.py
  --whisper DIR ("whisper-large-v3", listed as "kind": "asr"): POST
  /v1/audio/transcriptions and /v1/audio/translations in OpenAI's form
  (multipart/form-data, parsed with the stdlib; file in any format of ffmpeg,
  decoded to 16 kHz mono; model, language (none: detected), prompt,
  response_format json, text, srt, vtt or verbose_json, temperature 0 only,
  timestamp_granularities segment only). models/whisper/decoder.py:
  Whisper.segments is transformers 4.48's WhisperForConditionalGeneration.generate
  of one audio (greedy, no temperature fallback or thresholds,
  condition_on_prev_tokens False): the features of the whole audio
  (encoder.log_mel_all: the floor at the whole audio's peak - 8), detect_language
  over the first 3000 frames, the timestamp tokens under
  WhisperTimeStampLogitsProcessor's rules, a window's segments at its pairs of
  timestamps and the next window at the end of its last whole segment
  (_retrieve_segment); the prompt (<|startofprev|> and its ids) before each
  window, as transformers' with condition_on_prev_tokens False. Tokens.encode:
  the byte-level BPE of tokenizer.json (qwen.tokenizer's) with GPT-2's
  pre-tokenizer (split_gpt2, by unicodedata). Audio of at most 30 s for json
  and text stays generate()'s one window without timestamps. Against
  transformers float32 on the CPU (whisper_longform_reference.py,
  check_whisper_longform.py), on 5 min of the Rádio Regina interview (sk) and
  4 min 10 s of music, singing and talk: every token the same (2013 and 870;
  115 and 51 segments of the same start and end; WER and CER 0) on the GPU and
  the CPU; on the GPU also the same with language detection (the music clip: "en", as
  transformers), the translation (1028 tokens), and a prompt on two short clips
  with and without timestamps; the tokenizer's ids of 7 texts; the features
  within 3.4e-5. check_whisper and check_whisper_decoder unchanged (1-cos
  1.39e-5; KL 1.08e-9, 8/8). 5 min of talk: 13.5 s on the 3090 (real-time
  factor 0.045, the card shared), 279 s on 12 CPU threads (0.93; the music
  clip 0.59); the server's GPU 4980 MiB, its host 8.5 GiB at the peak; --cpu
  15.6 GiB.
- **Phase 3, Z-Image Turbo (text to image):** models/zimage (pipeline.py,
  dit.py) as diffusers' ZImagePipeline with guidance 0 (Turbo: no CFG, 9 steps
  by the model card). The prompt in Qwen3's chat template (enable_thinking: no
  think block) through Qwen3-4B (qwen3.dense, hidden_states[-2]: the stream
  after 35 of 36 layers, the prompt's tokens) on the CPU in its bfloat16 rows
  (KQ_BF16X16, float32 products: 8 GB, which does not fit on the GPU beside
  the DiT; text_gpu: on the GPU alone, freed after the prompt). The DiT
  (S3-DiT: width 3840, 30 heads of 128, SwiGLU 10240) as two programs: the
  caption once a prompt (cap_embedder's RMS norm and linear, cap_pad_token in
  its padding to a multiple of 32, the two context refiner blocks, no
  modulation) and a step of the unified stream [image, caption] (the patch
  linear, x_pad_token in the image's padding to a multiple of 32, the two
  noise refiner blocks over the image's rows, the caption's rows copied after
  them, 30 blocks over all with no mask: the padding rows are keys, as
  diffusers' batch of one); the modulation (1 + scale, tanh(gate) of
  adaln_input, no SiLU) folded on the host into the weights of each block's
  sandwich RMS norms; 3-axis RoPE (32, 48, 48 values, theta 256) of the
  caption at (1 + l, 0, 0), the image at (Tp + 1, y, x), its padding at 0, the
  cos and sin by the C library's cosf and sinf (torch.polar's bits; float64
  rounded is 1 ulp off in a few); the final LayerNorm (eps 1e-6) times 1 +
  scale and its linear. The DiT's 241 matrices in F20 on the GPU (24.11 ->
  15.26 GB, none refused), float32 as stored on the CPU (F20 by choice). The
  sigmas: torch's float32 linspace (its CPU kernel's fused multiply-add,
  written out) shifted by 3, the DiT's timestep (1000 - t) / 1000, its output
  negated. The VAE flux2.vae (FLUX.1's AutoencoderKL: 16 channels, no
  post_quant_conv, a new option; its bfloat16 weights as stored) of latents /
  0.3611 + 0.1159. No new records. scripts/zimage_reference.py (the pipeline
  in float32 on the CPU, the loop written out; --taps: RoPE tables and each
  block's rows of the first call), check_zimage.py (parts text, sched, dit,
  vae, all), zimage_generate.py; server/generate.py --z-image DIR
  ("z-image-turbo"; sizes multiples of 16, at most 1536 x 1536 of area).
  Against diffusers float32 (fox 1024 x 1024, stall 1024 x 1024, a Chinese
  poster 720 x 480 whose 1350 tokens pad to 1376; 9 steps): the ids the same;
  the text rows 1 - cos at most 1e-8, rel 1e-5 (torch's own two runs, 8 and 23
  threads, differ by that much: 1e-6 to the other); the sigmas of 1 to 100
  steps, the timesteps and each scheduler step the same bits; the RoPE tables
  the same bits; each DiT call on the reference's inputs 1 - cos a token a
  mean of 8e-11 to 6e-9 (GPU F20), 2e-12 to 9e-11 (CPU float32), the relative
  norm 1.3e-5 to 1.1e-4 (GPU), 2e-6 to 1.3e-5 (CPU); the VAE's image of the
  reference's latents at most 1 level of 255 (GPU 79-81 dB, CPU 93.8 dB). The
  whole pipeline from the reference's noise and prompt: fox GPU F20 a mean of
  0.013 levels (65.2 dB), CPU float32 0.013 (65.4 dB), CPU F20 0.0135 (65.3
  dB); stall GPU 0.072 (55.6 dB), CPU float32 0.011 (67.0 dB); poster GPU 0.62
  (42.1 dB; its latents 1.4% off), CPU float32 0.24 (48.3 dB, 0.6%), CPU F20
  0.30 (47.0 dB). F20 is at float32's floor: the GPU's remaining difference
  was ENC_ATTN_X2's two bfloat16 planes, not F20. The GPU's attention is now
  float32 (ENC_ATTN_TILE; x2 or RECLANK_ZIMAGE_X2=1: ENC_ATTN_X2): each call at
  the CPU's level, the fox 65.3 dB, the stall 0.010 levels, 67.6 dB, the
  poster 0.12 levels, 53.8 dB, at ENC_ATTN_X2's step time (5.7 s at 1024, 1.9
  s at 720 x 480; ENC_ATTN_RANGE's float32 took 1.57 x and 1.25 x). On the
  3090: a step 1.3 s at 512, 5.7 s at 1024, 13.5 s at 1536, 28.3 s at 2048;
  the first step after the DiT's return 9 s more (its 15.3 GB uploaded again:
  the DiT leaves for the VAE); the VAE 2.8 s (512), 5.3 s (1024); the prompt
  0.8 to 1.2 s on 12 CPU threads: a 1024 x 1024 image in 67 s (51 s in the
  server at 1024 x 768). GPU memory 14.9 GB at 512, 15.6 GB at 1024, 16.5 GB
  at 1536, 17.4 GB in the steps of 2048, whose VAE needs more than the card
  (the server's limit); the host's RSS 30.4 GB. The CPU (23 threads): a call
  118 s at 1024 in float32 (torch's 33 s), 66 s in F20; 69.4 GB of RSS in
  float32.
- **ENC_ATTN_TILE, float32 attention for the DiTs:** new record 291 (GPU):
  ENC_ATTN_RANGE without shared KV heads in register-blocked tiles on the
  CUDA cores (k_enc_attn_tile: a block of 256 threads, 128 queries of a head
  in shared memory, steps of 32 keys, a thread's 8 rows x 2 keys of scores
  and 8 rows x 8 columns of the values; 98.5 KB of shared memory at hd 128,
  heads of at most 128). Each score fmaf over d in order, expf, the sums in
  k_enc_attn_range_t's order: its bits wherever the blocks of both start at
  the same key (every real dump of Wan, Z-Image and Whisper's windows the
  same bits). enc_attn_range_x2 emits it for a compiler with attn_f32 (Wan's
  and Z-Image's DiTs by default) or under RECLANK_GPU_ENC_X2=0 (every
  program; RECLANK_GPU_ENC_TILE=0: ENC_ATTN_RANGE); the other models keep
  ENC_ATTN_X2. On the 3090 at hd 128 6 to 6.6 x ENC_ATTN_RANGE (Wan 720p's
  attention 676 ms a block against 4464 ms; ENC_ATTN_X2 605 ms), about 40-46%
  of the card's float32 peak; at hd 64 1.35 x ENC_ATTN_X2's time.
- **GPU swapping, phase 1 (docs/GPU_SWAP_PLAN.md):** encoders.runtime.Residency
  (RESIDENCY, one for the process): every _Runner registers; its resident bytes
  the mirror's Buffers plus the C side's of each GPU program (the drop of
  cudaMemGetInfo's free memory at a build, less the mirror's growth). Opt in:
  the generation server turns it on (RECLANK_RESIDENCY=1 or
  RECLANK_EVICT_ALWAYS=1 elsewhere); off, a _Runner runs as before (the LLM and
  embedding servers, the scripts). On: before a GPU build, make_room evicts the
  unpinned runners, least recently used first, with release(keep_host=True)
  (the host programs stay; the next run uploads again), until the free memory
  covers the build's need (the arrays not on the device, the scratch of its
  last build at that size; the first time an estimate: the largest float32
  array plus 64 MB) plus a reserve of 512 MB, and the runners' bytes fit the
  budget (refused only on the known bytes, not the estimate); a build records
  its graphs (prepare) so their memory is the build's; a cudaMalloc that fails
  anyway: every other unpinned runner goes, the build runs once more. Pins:
  pin() for good; use(*runners), a request's scope: each part held from its
  first run in the scope until use.done (its turn over: the text encoder, then
  the DiT before the VAE) or the scope's end. Every pipeline has runners() and
  a scope in its generate (FLUX.2, SD, Qwen-Image, Z-Image, Wan, Higgs,
  ACE-Step, Whisper), and the parts whose device state spans runs one of their
  own: a DiT from its prefix to the last step (Qwen-Image, Z-Image, Wan), Wan's
  VAE over its frames, Higgs's codes (the LM's caches; a voice's prefix
  snapshots are host copies and survive an eviction), the ACE planner, Whisper
  (the encoder's program to the last step). server/generate.py: no ad-hoc
  releases (last_image, Wan's after each video); --gpu-budget GB (default the
  free memory at the start less the reserve), --pin ID,... (whole models),
  --keep (the image models' text encoders and VAEs pinned); each eviction and
  load printed with its time. RECLANK_EVICT_ALWAYS=1: every unpinned runner
  (the one about to run too) goes before every GPU run. tests/unit/
  test_residency.py (the LRU order, the reserve, pins, scopes, early unpin,
  the budget, the retry, EVICT_ALWAYS, enabled and disabled). The check scripts
  with RECLANK_EVICT_ALWAYS=1 against without: SD, FLUX.2, Qwen-Image,
  Z-Image, Wan (dit, vae, a second prompt after its release, all: 50.2 dB),
  Higgs (and a cloned voice), ACE-Step and its planner, Whisper (and the long
  form: 2013 and 870 tokens) print the same numbers. A server session of SD,
  speech (a cloned voice), music (the planner), FLUX.2, Z-Image, video,
  transcription, speech (the same voice: its LM evicted, its prefix put back),
  SD: 9 outputs, each the bytes of a fresh server of that model alone; 18
  evictions, no out-of-memory, the server's peak 23.2 GiB of a 24.47 GB budget
  (the CUDA context's 1.2 GiB not counted). 720p video (1280 x 704) in a server
  that ran 480p first: the DiT (14.1 GB) goes for the VAE (22.8 GB), the MP4
  the bytes of a fresh server's. On the 3090 a release 0.1-0.35 s (Wan's DiT
  13.9 GB 0.34 s); a reload Higgs's LM 8.4 GB 2.0 s, SD's UNet 4.5 GB 1.1-3.1
  s, Wan's DiT 11.7 GB 2.7 s, its 720p VAE 22.8 GB 6.8 s. Open: the parts stay
  on the GPU between requests up to the budget (before, an image's text
  encoder and VAE left after it), so beside the embedding server set
  --gpu-budget (phase 4: a release across servers); a part whose device state
  spans calls must run in a scope (new models too), or an eviction loses it.
- **Wan 2.2 T2V-A14B (two 14B experts) on the 3090:** models/wan (dit.py,
  pipeline.py, vae.py, keep.py). WanPipeline with transformer_2 and
  boundary_ratio 0.875: the high-noise expert for the steps of t >= 875 at
  guidance 4.0, the low-noise one after at guidance_2 3.0 (the model card's;
  40 steps: 13 + 27, step 12 is t = 875), a scalar timestep (no
  expand_timesteps), UniPC of shift 3, 16 latent channels, the same umT5-XXL
  (the 5B's file hashes). An expert is 14.1 B parameters, 56.40 GB of float32
  in 35.69 GB of F20 (404 matrices): more than the card, so its 40 blocks are
  streamed: each block's 10 F20 matrices in one registered host array, two
  slots on the GPU, GP_FETCH of block j's 8 step matrices (the text
  program's k2 and v2) into slot j % 2 after FETCH_DONE of block j - 2 (the
  copy worker's pull kernel, 12.1 GB/s: 30.3 GB a call in 2.5 s; the stream
  waits 0.06 s a call, for the first block only). A second expert shares the
  first's runner (the activations, the text caches, the slots: RESIDENCY
  counts and evicts them as the runner's; one use() scope holds the runner
  from the first prefix to the last step, each expert's text caches made at
  its first step). Per call:
  480 x 832 x 5 9.3-9.6 s, x 17 23.6 s, 720 x 1280 x 17 64 s, 480 x 832 x 81
  143 s; a 40-step clip of 480 x 832 x 17 about 31 min, x 81 3.2 h, 720 x
  1280 x 33 about 3.2 h (the server's limits: 832 x 480 of area with 81
  frames, 1280 x 720 with 33, 40 steps at most). The VAE: Wan 2.1's (384 384
  384 192 96, no patch, no DupUp3D, the up blocks after the first half the
  channels in) in vae.py, and a compact layout of its buffers (each causal
  conv's cache apart, the chunks in three shared work slots; the same bits:
  11.5 -> 4.5 GB at 480p, 26.3 -> 10.0 GB at 720p; the 5B's 19.8 -> 7.7 GB
  at 704 x 1280; by default past 14 GB); a no-patch view of the output buffer
  copied (the next chunk wrote frame 0). The weightkeeper (several at once:
  --name, --list, the memory check; weightkeeper.patch) holds both experts'
  F20 rows (72.9 GB with the float32 small tensors; 114.3 GB read with
  O_DIRECT and converted in 661 s, ready in 866 s): the server attaches and
  registers its segment, starts in 5-19 s against 446 s loading the shards,
  the same bits (a call bit-identical, the server's MP4 the same bytes).
  Against diffusers float32 (480 x 832 x 5, 40 steps, the fox and waves
  prompts): the sigmas, timesteps and 40 scheduler steps the same bits, each
  step's expert the same; the DiT calls of steps 0, 12 (the last high), 13
  (the first low) and 38: 1 - cos a token a mean of 4.7e-12 to 1.1e-10, the
  worst 2.1e-8, the relative norm 3.1e-6 to 1.6e-5, the guided output's worst
  1.9e-7 (waves, step 38); a call at 480 x 832 x 17: a mean of 1.1e-11,
  4.8e-6; the VAE at most 1 level of 255 (PSNR 92.3 / 91.7 dB); the whole
  pipeline from the reference's noise: the final latents 1.6e-3 / 6.8e-3 off
  in norm, the video a mean of 0.046 / 0.151 levels, at most 9 / 15, PSNR
  61.4 / 55.8 dB. The reference: 4.9 h on 12 CPU threads for both prompts. The server (GPU): 12.4 GiB of the card at 480 x 832
  x 5 (the VAE), 18.0 GiB at 720p; 80 GiB of host (VmHWM).
- **SeedVR2-3B (one-step restoration and upscaling) and Real-ESRGAN:** models/seedvr2 (pipeline.py,
  dit.py, vae.py, vae_video.py), models/realesrgan (model.py), server/upscale.py.
  The official pipeline
  (ByteDance-Seed/SeedVR, projects/inference_seedvr2_3b.py: cfg 1, one step at t = 1000, the
  precomputed pos_emb text, no text encoder): the input resized to the output's area by torch's
  antialiased bicubic (a = -0.5, banded taps on the host), cropped to multiples of 16; the VAE
  (SD3's AutoencoderKL of 16 channels inflated to causal 3 x 3 x 3 convs, 8 x 8 x 4): an image's
  causal convs see its frame three times, so its kernels summed over time are its 2D convs (vae.py:
  encoder, decoder, the MAGViT upsampler's 1 x 1 conv read by the next conv's im2col, the
  posterior sampled as the official); a video frame by frame (vae_video.py: each causal conv a
  product of its first two taps over a cache of two frames plus the last tap's over the frame,
  then the cache moves on; the first frame fills it with two copies of itself, as extend_head;
  per-frame GroupNorm and attention; the stride-2 temporal downsamplers emit at the even frames;
  the first and the later programs share the caches and one arena: the same values as the whole
  clip); the DiT (dit.py: 32 MMDiT layers of 2560, the first 10 with their own text weights,
  SwiGLU) as one program of a size: the window attention of window.py (720p-area windows of about
  400 tokens a frame, shifted in the odd layers) as ENC_ATTN_TILE over a window layout (q, k, v
  gathered by ENC_GATHER: each window's video rows then a copy of the 58 text rows; lo and hi the
  window), the video rows gathered back, the text rows the mean of their copies; mmrope3d tables a
  layer kind (video (58 + frame, row, col) in its window, text (l, l, l)); AdaSingle's vectors of
  t = 1000 folded into the norms, biases and gates; the last layer's text unmodulated, as the
  official vid_only; the output norm with the first block's attention vectors (the official
  Cache's key); x0 = noise - v; the color fix (the wavelet reconstruction, our own code of the
  method: StableSR's file is not in the repository). No new records. The weights: the .pth files
  converted once by scripts/pt_to_safetensors.py (torch.load(weights_only=True) to safetensors; shared with the stereo port);
  the DiT in F20 on the GPU (13.38 -> 8.46 GB), float32 on the CPU; the VAE float32. The
  reference: scripts/seedvr2_reference.py (torch float32 on the CPU, written from the official code,
  none of it run), itself checked against the official code run sandboxed in float32 on the CPU
  (bridge_lq: the input the same bits, the posterior rel 3e-7, v 1 - cos 1.6e-11, worst token
  9.7e-9, the decode 97.4 dB). check_seedvr2.py against it (real LQ inputs: a photo 640 x 400
  JPEG q18 to 1280 x 800; a Wan clip down-scaled to 416 x 240 and H.264 crf 36, 17 frames to
  832 x 480): the resize max 6e-7; the windows and the text RoPE angles the same; the VAE
  92 dB (image and video, GPU and CPU); v 1 - cos 2.3e-9 (GPU F20; worst token 2.4e-6), 1.3e-10
  (CPU float32), video 2e-10 / 8e-11; the output at most 1 level of 255: 78.8 dB (GPU) and 82.8
  dB (CPU) for the photo, 83.1 / 84.7 dB for the clip; at 720p (a Wan 720p clip down-scaled to 640 x 352, H.264 crf 36, 9 frames to 1280 x 704: v 2.9e-10 / 2.9e-11, the VAE 91.3 dB, the output 87.6 / 89.8 dB). On the 3090: a 1280 x 800 image the DiT 1.9
  s, the VAE 0.6 + 1.2 s, a server request 12-15 s with its programs built; a 4.1 MP image (512 x 684 to 2352 x 1760) a request 40 s (the DiT 12.9 s with its 8.5 GB upload, 7.6 s warm; the VAE 9.3 + 12.2 s, 4.2 + 6.8 warm; the color fix 5 s on the host), 12.1 GiB of the card, 10.1 GiB of host; 832 x 480 x
  17: the DiT 3.8 s, the VAE 7.7 + 19.5 s, a server job 55 s (14.8 GiB of the card); 1280 x 704 x
  21 (a Wan clip at 640 x 352): 125 s (the encode 35, the DiT 16, the decode 63, the color fix 11
  on the host), 21.7 GiB of the card at the peak: video at most 1280 x 720 in float32 for now
  (960p and up: the caches in pinned host memory, a later phase). Real-ESRGAN (RRDBNet x4plus,
  x2plus, anime_6B; SRVGGNetCompact general-x4v3 with the official denoise mix of 0.5, animevideov3)
  as programs of 3 x 3 convs: a dense block's input and outputs in one buffer of 32-value rows
  (one ENC_GATHER per conv), LeakyReLU / PReLU as max(x, 0) + a min(x, 0) (the bits of torch's);
  against its torch reference (scripts/realesrgan_reference.py, written from basicsr and
  realesrgan's archs): at most 1 level of 255, 86-97 dB (GPU and CPU); x4plus 640 x 400 to 2560 x
  1600 in 5.2 s, SRVGG 0.5-0.8 s. server/generate.py --seedvr2 DIR, --realesrgan DIR: POST
  /v1/images/upscale (multipart or JSON; scale, size, seed, color_fix, denoise_strength) and POST
  /v1/videos/upscale jobs (read by GET /v1/videos/{id}, its MP4 with the input's audio).
- **2D to stereo 3D for VR players (docs/STEREO_PLAN.md):** models/depth (common.py,
  dpt.py, da3.py, moge.py, vda.py, stereo.py, pipeline.py, ref.py), server/stereo.py. Three depth
  models, float32 as stored (ENC_ATTN_TILE's float32 attention): MoGe-2 ViT-L (MIT; metric depth,
  the focal by SciPy's Levenberg-Marquardt as its infer(), the sky mask) and DA3MONO-LARGE (Apache
  2.0; relative depth, its sky branch) for photos, Video-Depth-Anything-Small (Apache 2.0; its Base
  and Large are CC-BY-NC-4.0) for video: DINOv2 towers and DPT / ConvStack heads of existing records
  (the 3 x 3 convs by im2col tiles, zero or replicate padding; the transposed convs a linear and a
  pixel-shuffle gather), VDA's temporal modules (each pixel's attention over the window's 32
  frames), its windows (step 22, the keyframes, the scale and shift fit at the frames' size, the
  crossfade) streamed. The disparity: inverse depth normalized to 0..1 (1st to 99.5th percentile, the
  sky at 0), 2 % of the width in all, the subject (the centre's median) on the screen plane; or
  metric (f B / 2 (1 / Z - 1 / Z0), MoGe-2). New records (from 300): ENC_RESIZE (torch's bilinear,
  an integer box mean), ENC_DIBR (one eye: a 1-D mesh of each row splatted with 64-bit atomicMax
  keys, the nearest first; holes from the farther side, then an edge-aware Jacobi fill or a 5-tap
  mean), ENC_PACK_U8, ENC_RESIZE_TAPS (OpenCV's cubic by host taps), ENC_UNPACK_U8; the CPU's C, the
  GPU's kernels (_rn intrinsics) and models/depth/ref.py the same bits. Packed full or half
  side-by-side (_LR, _3dh) or over-under (_TB, _3dv); video by ffmpeg in and out (HEVC by NVENC, the
  source's audio, Spherical Video V2's st3d box written by stereo.inject_st3d, Matroska's StereoMode
  for .mkv). POST /v1/images/stereo and POST /v1/videos/stereo (jobs of /v1/videos), --stereo-moge,
  --stereo-da3, --stereo-vda; scripts/stereo_convert.py from the command line;
  scripts/pt_to_safetensors.py converts VDA's .pth and MoGe's .pt once (torch.load(weights_only=True)).
  The reference (scripts/stereo_reference.py, torch only, written from the authors' code) agrees with
  the authors' code run apart (the same bits for DA3MONO and MoGe-2, VDA 5e-7). Against it (a photo
  1920 x 1200, a portrait 1795 x 2397, 96 frames of a 1440 x 1080 capture): DA3MONO's logits 1 - cos
  1.0e-12 to 1.3e-12 (504 and 1008, CPU and GPU), the depth's relative error at most 4.9e-5; MoGe-2's
  heads 1.8e-12, the metric scale and the focal to 1e-6, the depth in metres a median relative error
  of 1.4e-6; VDA-Small's aligned maps 1 - cos at most 1.8e-13 every frame; the warp the same bytes as
  ref.py at 1920 x 1200 and 3840 x 2400. On the 3090: DA3MONO 0.115 s (504), 0.451 s (1008);
  MoGe-2 0.94 s + 0.3 s on the host; both eyes and the packing of 1920 x 1200 23 ms; a VDA window of
  32 frames 1.11 s; a video of 1440 x 1080 to 2880 x 1080 HEVC 3.8-5.2 frames a second end to end
  (the per-frame preprocessing and the host's statistics the next work). The GPU (RESIDENCY's loads):
  moge-2 2.95 GB, da3mono 2.16 GB, vda-small 3.41 GB, the warp 0.1-0.4 GB.
  tests/unit/test_stereo.py (a crop of NASA's public-domain AS11-40-5903): 7 pass.
- **SeedVR2-7B:** the official models/dit NaDiT (36 layers of 3072, 24 heads, every layer its own
  text weights, the GELU (tanh) MLP of 12288 with biases, the "pixel" RoPE of the video rows:
  linspace(-1, 1, n) positions along each axis of a window, the checkpoint's 10 freqs an axis, 60
  of the 128 values; no output norm) in models/seedvr2/dit.py as the variant "7b" of the 3B's
  program (the same windows, gathers, folded modulation); ByteDance's float32 plain checkpoint
  (not the "sharp" one) in F20 (32.68 -> 20.68 GB), its 36 blocks streamed as Wan A14B's (one
  registered host array a block of 573 MB, two slots, GP_FETCH after FETCH_DONE; vid_in, txt_in,
  vid_out resident). The pipeline takes the checkpoint the directory holds; the server's
  --seedvr2-7b DIR adds "seedvr2-7b" to the upscale routes, one SeedVR2 model loaded at a time (the
  3B first; a request of the other loads it: the 7B about 4 min from an HDD). The reference:
  seedvr2_reference.py --variant 7b (its text pos_emb.pt the 3B's). Against it (the same real
  inputs and noise as the 3B's): the photo v 1 - cos 1.2e-8 (GPU F20; worst token 4.3e-5),
  1.3e-10 (CPU float32), the output at most 1 level of 255, 80.6 dB (GPU), 85.1 dB (CPU); the clip
  v 1.2e-10, the output 86.2 dB (GPU); with RECLANK_EVICT_ALWAYS=1 the same bits. On the 3090
  (requests as the server makes them): 512 x 684 to 1184 x 880 in 10.4 s (the DiT 4.6 s, 3.7 s
  warm; the 3B 8.4 s, its DiT 1.9 s warm), to 2352 x 1760 in 41.7 s (the DiT 15.9 s, 14.5 s warm;
  the 3B 39.6 s), 1280 x 704 x 21 in 114 s (the DiT 20.4 s); the card's peak 4.1 GiB at 1184 x
  880, 12.1 GiB at 2352 x 1760 and 21.7 GiB at 720p (the VAE's), the host's 20.6-23.1 GiB. The
  evaluation of Comfy-Org's int8 "convrot" 7B (a 256-block rotation (K x K x K x K) / 16 of the
  input; 46 dB from the float32 7B) is not ported.
- **Video frame interpolation, Practical-RIFE 4.25 (docs/VFI_PLAN.md):** models/vfi (rife.py,
  video.py, ref.py), server/interpolate.py, scripts/interpolate_video.py. hzwer's Practical-RIFE
  v4.25 (MIT; the README's Google Drive zip, flownet.pkl converted once by
  scripts/pt_to_safetensors.py: 5.66 M float32 parameters used, 22.7 MB) as two programs of one
  runner that share the frame slots: each input frame's Head once (conv 3x3 s2, two convs,
  LeakyReLU, ConvTranspose 4x4 s2), and one IFNet call an output frame at any t (five IFBlocks at
  1/16 to 1/1 (scale 0.5 for 4K): resize, two stride-2 convs, eight ResConvs with beta folded,
  ConvTranspose + PixelShuffle; the flow accumulated; the warps; the sigmoid merge); frames padded
  to a multiple of 64, float32 as stored. New records (from 320): ENC_WARP (torch's grid_sample,
  bilinear, border, align_corners: its CUDA kernel's formulas), ENC_PRELU (LeakyReLU, PReLU),
  ENC_MIX (the sigmoid merge); ENC_RESIZE (stereo's) gains row strides and column offsets (the
  28-channel concatenation without copies; without them the stereo record bit for bit); the
  ConvTranspose(4, 2, 1) a 3x3 im2col and a linear to its 4 phases. C, CUDA and NumPy the same bits
  (ENC_MIX's exp apart). The video: ffmpeg in by the source's timestamps, the output's times by a
  factor (the last frame repeated to the length) or a target rate (any ratio; a variable-rate input by
  its timestamps), the scene-cut rule (SSIM of 32 x 32 thumbnails < 0.2, or < 0.45 and the grey
  thumbnails' difference over 6 x the median of the 8 pairs before: the left frame repeated), ffmpeg
  out at the exact rate (60000/1001) with the source's audio copied and its colour tags; NVENC on the
  GPU, libx264 CRF 16 on the CPU. POST /v1/videos/interpolate (a file or `video_id` of any completed
  /v1/videos job: chains from Wan, the upscales, stereo; factor, fps, scene_cuts, scale, slowmo,
  codec, crf), --rife DIR. The reference: scripts/rife_reference.py (torch only, written from the
  authors' code, not run). Against it on real frames (59.94 fps captures and our own Wan clips, 140
  frames of 2x and 4x, CPU and GPU): block 0's input the same bits, each block's flow 1 - cos at
  most 1.1e-10, the output 1 - cos at most 2.2e-10 and at most 1 level of 255 (0.004-0.017 % of the
  values), the PSNR against the held-out frames the reference's to 0.01 dB (2x: 28.95 dB on a fast
  pan, 35.7 dB slow, 26.8 dB on video effects; FILM, measured and not ported, within 0.4 dB). The cut
  rule: 27 of 27 cuts of its tuning footage, 21 of 26 on new footage (the misses: shots of the same
  framing), 3 false cuts at 2x in 750 s of new footage. On the 3090: an output frame 40 ms (720p), 81 ms (1080p), 191 ms (4K, scale 0.5), a Head
  3-22 ms; a 1440 x 1080 clip doubled at 15 frames a second end to end; the card's peak 2.8 GiB.
  On 12 CPU threads an output frame 350 ms (720p), 751 ms (1080p). tests/unit/test_vfi.py (our own Wan frames): 10 pass (the model's test with RECLANK_RIFE set).
- **RIFE's scene-cut fix:** models/vfi (video.CutDetector, video.probe_frame, RIFE.prober and
  RIFE.residual). The first rule (thumbnails' SSIM and difference ratio) missed cuts between shots
  of the same framing and repeated frames falsely in video effects and limited animation. Now RIFE
  itself on probe frames (the long side 320, PIL's BOX, on the decoder's thread; one IFNet call a
  pair at t 0.5): R the mean |w0 - w1| of the two frames warped to the middle, its jump Rr against
  the median of the last 8 moving pairs; a cut when Rr > 4.5, R > 6 and SSIM < 0.75, or the first rule
  with Rr > 2.5, not within 2 pairs after another fire. On 1480 s of 59.94 fps captures (66,500
  pairs of 2x and 4x jobs; the live action by its frame-difference spikes, a product disc and an
  educational disc of limited animation by eye; thresholds fit on each half of each stretch, the
  midpoints taken): 2x 103 of 107 cuts found with 16 false (the first rule 101, 46), 4x 102 of 102
  with 4 false (97, 39); each half scored with the other half's own fit, 102 of 107 (16 false) and 96 of
  102 (7 false) (VFI_PLAN section 5); every remaining miss and false cut viewed. tests/unit/test_vfi.py (our own Wan
  frames: the rule on real residuals, the probe's residual on the CPU): 11 pass.
- **RIFE's speed work (VFI_PLAN section 6.3):** float32 throughout, the same bits. On the GPU each
  conv one ENC_CONV_MM (323: the products read the im2col rows through the index; the bias, the
  ResConv's residual and LeakyReLU in its epilogue; a deconv's phases scattered to their pixels),
  each block's input and the output fused (ENC_RIFE_IN 324, ENC_RIFE_OUT 325: no full-resolution
  buffer), the frames in and out as their bytes through pinned slots (ENC_UNPACK_U8P 326, ENC_PACK_U8C
  327), the thumbnails' inputs and the probe frame on the GPU (ENC_THUMB_U8 329; ENC_RESAMPLE_U8 328,
  PIL's BOX passes from Pillow's Resample.c, MIT-CMU), the pair's first frame made during the cut
  test; ffmpeg and the pipeline's threads on all the CPUs of the start (reclank's OMP_PROC_BIND had
  put a whole job on one CPU). IFNet 40 / 81 / 191 -> 12.8 / 24.3 / 35.4 ms (720p, 1080p, 4K); end
  to end (NVENC) a 1440 x 1080 x2 job 20.6 (master) / 15.4 (the cut fix) -> 58.9 frames a second,
  1280 x 720 30.9 / 22.6 -> 83.7, 4K 4.2 / 3.7 -> 17.9 (bound by ffmpeg's one-thread RGB to yuv420p
  conversion); the server 15.3 -> 58.1; the card's peak 3.5 -> 1.9 GiB. The output frames' bytes the
  same as the previous programs' (RECLANK_RIFE_IM2COL=1); the reference's numbers unchanged on all
  six sets. The same conv on DA3Mono, VDA-Small and Real-ESRGAN (measured, not changed): the same
  bits, 0.104 -> 0.094 s, 6.1 -> 6.8 frames a second, 5.1 -> 2.0 s.
- **RIFE's encoder colour (VFI_PLAN section 6.4):** the yuv420p conversion followed none of the
  tags it copied (BT.601 limited always): a BT.709 source came out BT.601 tagged bt709, a full-range
  one limited tagged pc (decoded by its tags 28.8 dB against the frames sent). Now the scale filter
  takes the source's matrix and range (media.yuv_colors, NVENC and libx264), an untagged source
  BT.601 limited as before and tagged smpte170m / tv: the full-range case 28.8 -> 36.9 dB, BT.709 36.5
  -> 36.6, the tags' decode the best on every source; the speed unchanged. NVENC's own RGB conversion
  (rgb0) measured: the same colours, whole jobs 2-5 % slower; not taken.
- **ffmpeg and the media threads on all the CPUs (models/media.py):** reclank's OMP_PROC_BIND binds
  the main thread to one CPU when OpenMP starts; the threads made after it and their children
  inherit it, and ffmpeg's libgomp binds again: a stereo video's decoder and writer threads and both
  ffmpeg processes ran on one CPU, and so did Wan's writer and the upscales' decode, encode and mux
  (taskset -p of each thread and child during a run). media.popen / run start a tool on the CPUs of
  the start (RECLANK_START_CPUS) without OMP_PROC_BIND and OMP_PLACES, widen() puts a pipeline's
  thread on them; RIFE's pipeline (which had them), the stereo video (models/depth/pipeline.py), Wan's
  write_mp4 and server/upscale.py use them. The encodes convert by the source's colour tags
  (media.yuv_colors, as RIFE's): the stereo video and the upscales keep their source's matrix and
  range and are tagged (they were BT.601 limited and untagged; an untagged source still is, now
  tagged smpte170m / tv); Wan's clips are BT.601 limited tagged so. server/generate.py's audio encodes
  and music's and EmbeddingGemma's decoders keep their own environment (no OMP_PROC_BIND) as before.
  The stereo video of a 600-frame 1440 x 1080 capture (VDA-Small, NVENC) 7.99 -> 12.25 frames a
  second, its decoded frames the same (the md5 of all 600); Wan's writer (33 frames of 1280 x 720,
  libx264) 3.14 -> 1.00 s; an upscale's video in and out (300 frames of 1280 x 720) read 6.93 -> 1.98 s,
  write with the source's audio 43.4 -> 5.6 s. libx264's output depends on its thread count: with
  ffmpeg held to one CPU the new code's decoded frames are the old ones' (md5), on twelve the same
  quality (44.04 / 44.02 dB, 37.42 / 37.42 against the frames written). RIFE's output frames the same
  bytes and speed. tests/unit/test_media.py.
