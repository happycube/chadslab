# Plan: speech and music generation

Give reclank generation models: text to speech and text to music (and,
after them, the media models of section 8: speech to text, upscaling, 2D to
stereo, frame interpolation). reclank generates no images or videos: those
models were removed. Every model runs on the records it
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
and a phonemizer, which nothing else here needs (section 3.2).

### 1.2 Text to music

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
DiT records, and a 1-D VAE decoder made of the same
convolutions (ENC_GATHER im2col) as MuQ and Whisper.

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

### 3.1 Shared by both areas

1. **Tensor-core attention.** ENC_ATTN_RANGE is a float32 kernel on the
   CUDA cores: 38 % of Whisper's encoder at 1500 rows, and a DiT's attention
   over thousands of latent rows is many times that. A flash-attention kernel
   on the tensor cores (q, k, v as two 16-bit planes where precision needs
   it, float32 softmax and sums; the bits of the CPU are not the target) is
   the first new kernel. It keeps the ENC_ATTN_RANGE operands (lo, hi) so
   packing and masks stay.
2. **A Qwen3 dense model.** The planner LM and the embedding model of
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
   padding rows of zeros; nearest upsampling by ENC_GATHER.
6. **Weights:** safetensors in bfloat16 as stored. The GPU path for 16-bit
   matrices exists (k_gemm_bf16_tc with bfloat16 W, and the float16 path
   being added for Whisper); BF12 is a quantization and needs approval per
   model.

### 3.2 Speech

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

### 3.3 Music

- ACE-Step 1.5: the planner LM (Qwen3 dense decode, 5 Hz codes and text),
  Qwen3-Embedding for the prompt, the DiT (the DiT records over a
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
- **Music:** POST /v1/music/generations (prompt, lyrics, duration, seed,
  format) with job routes (/v1/music/{id}, its content);
  a short piece may answer synchronously with "wait": true.
- The GPU is shared with the embedding server and the chat servers: a model
  loads on demand and frees its memory after an idle time, or holds it by a
  flag.

## 5. The checks

As for the encoders, with what diffusion adds:

1. A reference script for each model in its own venv (transformers,
   diffusers or the authors' code), float32 on the CPU, on real prompts,
   real voices (the user's own clips), that
   saves the inputs and the intermediate tensors: the text encoder's output,
   the DiT's output at two or three timesteps for the same latent and noise,
   the VAE decode of a latent, the codec decode of real codes.
2. A check script: each part on the same inputs as the reference (the
   reference's noise and latents, not our sampler's), 1 - cos and the
   largest relative difference; then the whole pipeline from the same seed
   and noise, where the final audio is compared by its difference
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
| 1 | Qwen3 dense: encoder program and decode engine; check against transformers | 2 days | The LM of the TTS models and of ACE-Step |
| 2 | TTS: Higgs TTS 3 (LM heads, the delay pattern, its codec decoder, ENC_SNAKE if it has Snake), /v1/audio/speech; then Qwen3-TTS on the same pieces | 3-4 days, then 2 | Speech in Czech, Slovak, English; cloning |
| 4 | Music: ACE-Step 1.5 (planner LM on phase 1, its DiT, the 1-D VAE), the music job routes | 3-4 days | Songs with lyrics |

Phases 1 and 0 can run in either order; phase 2 needs only 1.

## 7. Risks

- **VRAM.** The DiT, the text encoder and the VAE may not all fit at once:
  run the encoder first (or on the CPU) and free it.
- **Steps.** The distilled forms (ACE-Step turbo) need
  4-8 steps; the base forms 30-50 with CFG (twice the rows). Start with the
  distilled.
- **Licenses.** By the rule of the introduction: any model whose license
  carves out regions such as the EU, the UK or South Korea is out; a
  sanctions clause alone is fine. One non-commercial pick, for its edge:
  Higgs TTS 3 (Czech and Slovak). Fish S2 and Voxtral have no edge over the
  permissive picks. Each license is read again from the model's own
  repository before a download; reclank ships code, and each model's docs
  say its weights' terms.
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
  memory). The products of bfloat16 checkpoints run as
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
- **F20, float32 checkpoints in 63% of their memory:** a format of 16
  significant bits a value (formats.gguf F20 = 68; backends/cpu/csrc/kernels/kq
  formats.c): BF12's groups of 32 with a 15-bit mantissa, a uint16 of the sign
  and mantissa a value (rounded to nearest, even), its exponent's gap to the
  group's largest (4 bits) and that largest exponent (a byte), 20.25 bits a
  value; a value more than 15 binades under its group's largest is 0 (BF12's
  rule and zero code). For checkpoints that are float32 in earnest (every low
  bit used): F20 changes a matrix by about 6.5e-6 (norm), bfloat16 by 1.7e-3.
  The products are float32's: ENC_LINEAR's wbf 3, on the GPU k_enc_sgemm (and
  k_enc_gemm) decoding F20 rows as they load W's tiles, on the CPU kq_x16f_body
  decoding KQ_F20X16 (F20 in groups of 16 rows, 1296 bytes a block of 32
  columns) column by column: each output the same bits as float32 W of the
  decoded values. encoders.runtime.to_f20 / F20Lins / f20_lins: a linear's F20
  (the rule of to_bf12 refuses a matrix of cols not a multiple of 32), its
  float32 W dropped as its F20 is made, the checkpoint's mapped pages of it too
  (formats.safetensors.drop_pages). SeedVR2's and FlashVSR's DiTs use it.
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
- **ENC_ATTN_TILE, float32 attention for the DiTs:** new record 291 (GPU):
  ENC_ATTN_RANGE without shared KV heads in register-blocked tiles on the
  CUDA cores (k_enc_attn_tile: a block of 256 threads, 128 queries of a head
  in shared memory, steps of 32 keys, a thread's 8 rows x 2 keys of scores
  and 8 rows x 8 columns of the values; 98.5 KB of shared memory at hd 128,
  heads of at most 128). Each score fmaf over d in order, expf, the sums in
  k_enc_attn_range_t's order: its bits wherever the blocks of both start at
  the same key (every real dump of the DiTs and Whisper's windows the same
  bits). enc_attn_range_x2 emits it for a compiler with attn_f32 (FlashVSR's
  DiT, the depth models) or under RECLANK_GPU_ENC_X2=0 (every
  program; RECLANK_GPU_ENC_TILE=0: ENC_ATTN_RANGE); the other models keep
  ENC_ATTN_X2. On the 3090 at hd 128 6 to 6.6 x ENC_ATTN_RANGE (a 720p DiT's
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
  a scope in its generate (Higgs, ACE-Step, Whisper; then the upscalers), and
  the parts whose device state spans runs one of their own: Higgs's codes (the LM's caches; a voice's prefix
  snapshots are host copies and survive an eviction), the ACE planner, Whisper
  (the encoder's program to the last step). server/generate.py: no ad-hoc
  releases; --gpu-budget GB (default the free memory at the start less the
  reserve), --pin ID,... (whole models); each eviction and
  load printed with its time. RECLANK_EVICT_ALWAYS=1: every unpinned runner
  (the one about to run too) goes before every GPU run. tests/unit/
  test_residency.py (the LRU order, the reserve, pins, scopes, early unpin,
  the budget, the retry, EVICT_ALWAYS, enabled and disabled). The check scripts
  with RECLANK_EVICT_ALWAYS=1 against without: Higgs (and a cloned voice),
  ACE-Step and its planner, Whisper (and the long form: 2013 and 870 tokens)
  print the same numbers. On the 3090 a release 0.1-0.35 s; a reload Higgs's
  LM 8.4 GB 2.0 s. Open: the parts stay on the GPU between requests up to the
  budget, so beside the embedding server set
  --gpu-budget (phase 4: a release across servers); a part whose device state
  spans calls must run in a scope (new models too), or an eviction loses it.
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
  JPEG q18 to 1280 x 800; a clip of our own down-scaled to 416 x 240 and H.264 crf 36, 17 frames to
  832 x 480): the resize max 6e-7; the windows and the text RoPE angles the same; the VAE
  92 dB (image and video, GPU and CPU); v 1 - cos 2.3e-9 (GPU F20; worst token 2.4e-6), 1.3e-10
  (CPU float32), video 2e-10 / 8e-11; the output at most 1 level of 255: 78.8 dB (GPU) and 82.8
  dB (CPU) for the photo, 83.1 / 84.7 dB for the clip; at 720p (a 720p clip of our own down-scaled to 640 x 352, H.264 crf 36, 9 frames to 1280 x 704: v 2.9e-10 / 2.9e-11, the VAE 91.3 dB, the output 87.6 / 89.8 dB). On the 3090: a 1280 x 800 image the DiT 1.9
  s, the VAE 0.6 + 1.2 s, a server request 12-15 s with its programs built; a 4.1 MP image (512 x 684 to 2352 x 1760) a request 40 s (the DiT 12.9 s with its 8.5 GB upload, 7.6 s warm; the VAE 9.3 + 12.2 s, 4.2 + 6.8 warm; the color fix 5 s on the host), 12.1 GiB of the card, 10.1 GiB of host; 832 x 480 x
  17: the DiT 3.8 s, the VAE 7.7 + 19.5 s, a server job 55 s (14.8 GiB of the card); 1280 x 704 x
  21 (a clip of our own at 640 x 352): 125 s (the encode 35, the DiT 16, the decode 63, the color fix 11
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
  (not the "sharp" one) in F20 (32.68 -> 20.68 GB), its 36 blocks streamed (one
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
- **SeedVR2 on the tensor cores (the 3B and 7B, the default; RECLANK_SEEDVR2_TC=0 and
  RECLANK_SEEDVR2_DIT_TC=0 the float32 programs above):** the video VAE's causal 3x3x3 convs as
  ENC_CONV_MM (an implicit im2col: the gather in the kernel, the same bits as gather + linear; its
  cache product and its frame product), W in bfloat16 and x rounded to bfloat16 as gathered, float32
  sums (k_enc_conv_tc: mma m16n8k16, 128 x 128 tiles); the DiT's F20 matrices decoded to bfloat16 on
  the tensor cores (ENC_LINEAR wbf 5, k_enc_gemm_f20tc), the official code's bfloat16 autocast. The
  float32 ENC_CONV_MM gains k steps of 16 (k_enc_conv_mm16, the same bits; RECLANK_CONV_MM16=0 the
  steps of 4). On the 3090, the 7B on 121 frames of 768 x 484 to 1184 x 752: 646 -> 379 s (the encode
  101 s, the DiT 107 s, the decode 115 s, the colour fix 35-50 s), against the float32 program 62.7
  dB, at most 5 levels of 255.
  The caches of the tensor cores' causal convs in bfloat16 (ENC_BF16_PACK, the rounding the conv's gather
  gives x, read by k_enc_conv_tc as they are: the same bits; RECLANK_SEEDVR2_CACHE16=0 float32): about
  half the VAE's caches, so the 3B at 1280 x 960 in batches of 85 frames peaks at 21.1 GB on the card.
- **16-bit weights for the tensor cores, any float32 W (encoders.runtime.to_w16, W16_FORMATS):**
  - W16H ("f16", KQ_W16H): float16 rows of W ws, ws a power of two of each row (max |w| ws in [2^14,
    2^15)), then ws; x in float16 saturated at +-65504; the products divided by ws exactly. W16I ("i16",
    KQ_W16I): I16 rows, each group of 64 values as int16 (two int8: hi and lo) with a float32 scale, then
    the scales (program.i16_row_bytes); x quantized the same way as it is loaded (or packed:
    ENC_I16_PACK); three int8 products (hi hi, hi lo, lo hi) a group, int32 sums, then the float32 sum.
  - The records: ENC_LINEAR wbf 7 (W16H) and 8 (W16I) for a linear's lin.wq (input and output clamps
    kept; any n); ENC_CONV_MM wbf 3 and 4 (W16H; 4 x's rows float16), 5 and 6 (W16I; 6 x's rows I16);
    ENC_BF16_PACK (bfloat16 or float16) and ENC_I16_PACK write x's 16-bit rows with the kernels' own
    conversion (the same bits).
  - F20Lins(fmt=) or RECLANK_W16=f16|i16 (default f20) gives the GPU's matrices of SeedVR2 one of the
    forms (stored so: the VRAM of 16 bits a value); the streamed blocks of SeedVR2-7B take any of them (encoders.runtime.lin_q).
  - SeedVR2: RECLANK_SEEDVR2_W16 = f16 (the default), i16 or bf16 (the VAE's convs and the DiT's
    matrices); the VAE's caches in that form, and each causal conv's frame product reads x's packed rows
    in the cache (packed once: the frame product twice as fast as from float32 x, the same bits). The
    7B's streamed blocks now on the tensor cores too (their proxies had lost lin.tc). On 9 real frames
    (1184 x 752), against the float32 programs:

    | form | 3B PSNR | 3B max | 3B s | 7B PSNR | 7B max | 7B DiT host memory |
    |---|---|---|---|---|---|---|
    | bf16 (F20 W) | 57.0 dB | 19 | 26.1 | 60.6 dB | 7 | 20.6 GB |
    | f16 | 65.9 dB | 10 | 25.8 | 69.5 dB | 1 | 16.3 GB |
    | i16 | 71.5 dB | 5 | 33.5 | | | |

    The 3B DiT on the card: 8.46 GB (F20), 6.69 GB (f16), 6.90 GB (i16). On the VAE's convs (real W
    and x, the relative RMS error of the products against float64): bf16 1.4-1.8e-3, f16 0.5-1.6e-4,
    i16 0.5-2.2e-5; f16 from packed x 49-62 TFLOPS, i16 33-36 (128 x 64 tiles: x read twice for 128
    outputs; 64 x 128 tiles need dynamic shared memory, not done).
- **SeedVR2 on a long video without batches (pipeline.restore_long; a scene at a time):** the video VAE's
  encoder and decoder each one causal pass over all the frames (their caches carry on), the DiT on chunks
  of `chunk` latent frames (long_chunks): each keeps its central chunk - 2 margin latents, the rest its
  context, the chunks' starts multiples of that core and the core a multiple of the DiT's time window
  (ceil(min(chunk, 30) / 4)), so a kept latent sees the windows of one pass over all; the latents padded
  (the last frame repeated) so the last chunk is whole; each latent frame's noises its own (the seed and
  its index). Against one pass of the DiT over 60 latents (237 real frames, the 3B at 688 x 432, chunks of
  30, margin 3): no step at the joins (each frame's change from the one before within 4.7% of the one
  pass's; median 0.999), the frames 32-52 dB from the one pass's (lowest near the joins and the padded
  end). The 7B on 1280 x 960 in chunks of 30 (117 frames; the card's peak 22.1 GB at 121): 4.0 s a frame.
  The colour fix of a video's frames on threads (color_frames, RECLANK_SEEDVR2_COLOR_PROCS: up to 16), each
  rebound to the CPUs of the start (a new thread takes the main thread's single CPU), the same bits: 0.37 ->
  0.05 s a frame at 1280 x 960. Not forked processes: a fork copies the host memory registered with CUDA
  (the 7B's streamed blocks).
- **RealBasicVSR (video super-resolution x4 of real-world footage; Apache 2.0):** models/realbasicvsr
  (model.py), scripts/check_realbasicvsr.py. MMEditing v0.16.0's RealBasicVSRNet (64 channels, 20 residual blocks
  a branch, 6.3 M parameters; the official checkpoints' generator_ema, realbasicvsr and realbasicvsr-wogan,
  converted by pt_to_safetensors.py --key state_dict --prefix generator_ema.) as four programs of one runner:
  the cleaning of a frame; SPyNet of a pair (the frames resized to multiples of 32, a pyramid of 2 x 2 means, six
  levels of five 7 x 7 convs); a propagation step (backward or forward: the state's features warped by the flow,
  zeros past the image, then 21 convs; the state stays on the device between steps); the upsampling (fusion, two
  pixel shuffles read through the next conv's index, conv_hr, conv_last, + the bilinear x4 of the frame; x2: its
  2 x 2 means). The convs ENC_CONV_MM on the GPU, im2col tiles + ENC_LINEAR on the CPU (the same bits), float32
  as stored. ENC_WARP gains padding "zeros" (its operand 16; the border's records as before). A scene at once as
  the official on the whole sequence: the backward features of one segment kept at a time on the host, a
  checkpoint at each later segment's end and its segment computed again from it (the same bits; segments of 32
  when a scene's features are over RECLANK_RBVSR_FEAT_GB, 16 by default). Against the official network (torch
  float32 on the CPU, run in a sandbox) on real frames (30 of 256 x 192 and 7 of 250 x 190, both checkpoints,
  CPU and GPU, segments of 3 and 8 too): the cleaned frames the same bits, the flows 103-107 dB, the features
  84-98 dB, the output 112-125 dB, 99.98-99.996% of its uint8 values the same (the others 1 apart). A scene of
  146 frames of 640 x 480 to 1280 x 960 on the RTX 3090: 0.96 s a frame (cleaning 23 s, flows 16 s, backward 34
  s, forward and upsampling 67 s).
- **FlashVSR v1.1 (one-step streaming diffusion video super-resolution; Apache 2.0):** models/flashvsr
  (pipeline.py, dit.py, lq_proj.py, tcdecoder.py), scripts/check_flashvsr.py. The official Tiny-Long pipeline: the
  LQ video (each frame bicubic to the output's H x W, multiples of 128; made as the chunks read them) streamed
  through the LQ projector (Causal_LQ4x_Proj: two causal 4 x 3 x 3 convs of its 16 x 16 pixel-shuffled clips of 4
  frames, their states kept on the device), the DiT (Wan 2.1 T2V-1.3B as the DMD checkpoint, one step at t = 1000:
  the first chunk of 6 latent frames, then 2 at a time, the LQ condition added before the first block) and the
  TCDecoder (TAEHV's decoder widened, the LQ frames as its condition; its MemBlocks' states carried from chunk to
  chunk), then the AdaIN colour fix. The DiT's self-attention block-sparse as Block-Sparse-Attention's: its rows in
  windows of 2 x 8 x 8 tokens (a block of 128 rows), the keys and values of the chunk's slices and the 3 before in
  a ring of 4 slots (float16 on the GPU, by the user's choice; float32 on the CPU), each query block's key blocks
  from the official's mask (the blocks' mean q and k, the 11 x 11 local window, the softmax, the top-k of each
  head's slice). New records: ENC_ATTN_BLOCKS (370; k_enc_attn_tile's float32 form over a list of key blocks, K and V
  float32 or float16, a tile's query blocks by qb0; on the tensor cores k_enc_attn_blocks_tc, q and the weights float16,
  the scores, the softmax and the sums float32, K and V by cp.async in two buffers) and ENC_BLOCK_MASK (371; the mask on
  the device: the means' scores, the softmax, each head's slice's (k + 1)-th largest by a radix select of its bits, the
  lists; the CPU's and the interpreter's the host's draft_mask). A chunk of the DiT is one program of its 30 blocks, no
  host step between them; on the GPU each layer's ring lives in pinned host memory (28 GB at 2560 x 1920) and two device
  pairs take the layers in turn (GP_FETCH of layer i + 2's cached slots while the layers between run, GP_TO_HOST of each
  new slice); its blocks in row tiles of 8192 (every operation but the attention a row's own; only the residual
  stream and the queries kept for all the rows); the TCDecoder's last level in strips of 96 rows and its programs'
  scratch from one pool, the convs' inputs packed to float16 once (the gather's conversion); the host's steps on
  threads while the GPU runs (the next chunk's LQ frames, pixel shuffles, RoPE tables and the TCDecoder's inputs
  made ahead, the chunk before's colour fix and uint8 behind, the TCDecoder's programs queued before the frame
  before is read, the ring's pages written on threads before they are pinned): all the same bits. On the GPU the
  DiT's matrices and self-attention and the LQ projector's and TCDecoder's convs in float16 on the tensor cores by
  default (W16H rows, x, q and the attention's weights float16, float32 sums; RECLANK_FLASHVSR_TC=0: float32
  products, the DiT's matrices F20); the rest float32. One change from the
  script: the last frame repeated past the video's end (the script drops up to 7 of its last frames). Against the
  official (float32 on the CPU, Block-Sparse-Attention as a dense float32 stand-in, run in a sandbox) on 41 real
  frames at 512 x 384: the LQ projector 115.9 dB, the TCDecoder 112.5 dB, the DiT 109 dB on the CPU and 73-75 dB on
  the GPU in float32 (F20, the float16 cache), every layer's block masks the same over all chunks (the CPU's and the
  GPU's ENC_BLOCK_MASK); the whole pipeline on the GPU 65.6 dB in float32 (98% of its uint8 values the same, the
  others at most 5 apart), 56.8 dB on the tensor cores (the DiT's 57.8, the decoder's 62.0, the projector's 65.1),
  where the official's own bfloat16 is 28.1 dB from its float32 (32% of its uint8 values the same, up to 93 apart).
  640 x 480 on the RTX 3090: to 1536 x 1152 in 2.2 s a frame (float32, before the host's steps were made faster);
  to 2560 x 1920 (x 4) 5.1 s a frame in float32 and 3.45 s on the tensor cores (the steady chunks of 100 frames;
  7.4 s before the masks went to the device and the copies and host steps overlapped), 1.6 s with the
  self-attention on the tensor cores (57.2 dB against the official's float32) and the host's steps beside the
  GPU's (a scene of 596 frames in 985 s, the first chunk's 60 s with it; 4653 s before), the card's peak 23.3 GB
  (at its 220 W limit).
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
  /v1/videos job: chains from the upscales, stereo; factor, fps, scene_cuts, scale, slowmo,
  codec, crf), --rife DIR. The reference: scripts/rife_reference.py (torch only, written from the
  authors' code, not run). Against it on real frames (59.94 fps captures and our own clips, 140
  frames of 2x and 4x, CPU and GPU): block 0's input the same bits, each block's flow 1 - cos at
  most 1.1e-10, the output 1 - cos at most 2.2e-10 and at most 1 level of 255 (0.004-0.017 % of the
  values), the PSNR against the held-out frames the reference's to 0.01 dB (2x: 28.95 dB on a fast
  pan, 35.7 dB slow, 26.8 dB on video effects; FILM, measured and not ported, within 0.4 dB). The cut
  rule: 27 of 27 cuts of its tuning footage, 21 of 26 on new footage (the misses: shots of the same
  framing), 3 false cuts at 2x in 750 s of new footage. On the 3090: an output frame 40 ms (720p), 81 ms (1080p), 191 ms (4K, scale 0.5), a Head
  3-22 ms; a 1440 x 1080 clip doubled at 15 frames a second end to end; the card's peak 2.8 GiB.
  On 12 CPU threads an output frame 350 ms (720p), 751 ms (1080p). tests/unit/test_vfi.py (our own frames): 10 pass (the model's test with RECLANK_RIFE set).
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
  102 (7 false) (VFI_PLAN section 5); every remaining miss and false cut viewed. tests/unit/test_vfi.py (our own
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
  ffmpeg processes ran on one CPU, and so did the upscales' decode, encode and mux
  (taskset -p of each thread and child during a run). media.popen / run start a tool on the CPUs of
  the start (RECLANK_START_CPUS) without OMP_PROC_BIND and OMP_PLACES, widen() puts a pipeline's
  thread on them; RIFE's pipeline (which had them), the stereo video (models/depth/pipeline.py),
  media.write_mp4 and server/upscale.py use them. The encodes convert by the source's colour tags
  (media.yuv_colors, as RIFE's): the stereo video and the upscales keep their source's matrix and
  range and are tagged (they were BT.601 limited and untagged; an untagged source still is, now
  tagged smpte170m / tv). server/generate.py's audio encodes
  and music's and EmbeddingGemma's decoders keep their own environment (no OMP_PROC_BIND) as before.
  The stereo video of a 600-frame 1440 x 1080 capture (VDA-Small, NVENC) 7.99 -> 12.25 frames a
  second, its decoded frames the same (the md5 of all 600); write_mp4 (33 frames of 1280 x 720,
  libx264) 3.14 -> 1.00 s; an upscale's video in and out (300 frames of 1280 x 720) read 6.93 -> 1.98 s,
  write with the source's audio 43.4 -> 5.6 s. libx264's output depends on its thread count: with
  ffmpeg held to one CPU the new code's decoded frames are the old ones' (md5), on twelve the same
  quality (44.04 / 44.02 dB, 37.42 / 37.42 against the frames written). RIFE's output frames the same
  bytes and speed. tests/unit/test_media.py.
- **Float frames, field deinterlacing and Real-ESRGAN's convs:** RIFE and Real-ESRGAN take and give
  float32 frames (RIFE push() of a float32 frame and frame_f32(), the "headf" / "netf" programs;
  RealESRGAN.upscale_f32()), for sources of more than 8 bits: fed the bytes / 255, the byte paths'
  values before their rounding (tests/unit/test_vfi.py). scripts/deinterlace_rife.py: each field of an
  interlaced video a progressive frame, its own lines its decoded samples and the other field's lines
  RIFE's (each parity its own half-height stream, so no half-line shift), at the field rate; the
  fields to float RGB by their matrix without clamping (an LD capture had 35% of its pixels outside
  0..1: the 8-bit round trip alone cost ~10 levels of 10-bit luma; held-out fields 27.5 -> 29.4 dB);
  scene cuts by CutDetector's thumbnail rule; --upscale realesrgan-x2plus 2x (the network on the values
  clipped to 0..1, the clipped-off part added back bicubic: closer to the input than the raw values);
  H.264 (High 4:2:2 for 10-bit 4:2:2) or FFV1; --gpu-lock takes the GPU lock and lets it go every
  10 min. Real-ESRGAN's GPU convs: ENC_CONV_MM (the same bits as the im2col tiles; x2plus 768 x 484
  1.80 -> 0.65 s), its narrow outputs (m 32 / 64) by k_enc_conv_mm_nar (8 x 4 / 8 x 8 a thread, chunks
  of 16 of k: 0.52 s, the same bits, RIFE's too; RECLANK_CONV_MM2=0 off): DRAM-bound, as the order of
  the sums (each tap's channels in turn) reads each input row ~9 times; then ENC_CONV3 (record 332):
  direct tiles of 8 x 32 pixels, each 8-channel chunk with its border and its weights in shared memory,
  the dense blocks' channel groups read in place, the 2x nearest upsampling and the x 0.2 + residual in
  it: 0.26 s (14-15 TFLOPS), float32 in its own order (the output within 8.6e-6 of the exact path;
  against the reference x2plus 86.9 dB, x4plus 87.5, anime 90.3, at most a level of 255);
  RECLANK_ESR_EXACT=1 the ENC_CONV_MM path, RECLANK_ESR_IM2COL=1 the tiles.
