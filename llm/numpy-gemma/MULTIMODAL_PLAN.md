# Plan: image and audio input

## 1. The goal

Give the runtime image and audio input for four models:

1. The Gemma 4 12B (the unified model): images and audio.
2. The Gemma 4 E4B (and the E2B): images and audio.
3. The Gemma 4 26B A4B: images.
4. Qwen3.6-35B-A3B: images.

A request to the server sends the OpenAI content parts (image_url,
input_audio). The answer must agree with the reference model
(transformers), and an image or a clip must take seconds, not minutes.

The 12B comes first, before the E4B. Its encoders have no layers (section
3.1). Each patch of 48 x 48 pixels, and each 40 ms of audio, goes through
a normed linear.

Thus the first phase can build and check all of the text
side with an encoder that cannot hide an error. The text side is the soft
rows, the mask in an image, the sessions, and the server. The transformers reference of the
12B loads from a checkpoint that is on disk, with no change of the
weights. Then the E4B adds only its encoders (two transformer stacks).

## 2. The files on disk

    E4B    ~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/ce15.../
           mmproj-BF16.gguf (0.99 GB): vision (v.*), audio (a.*), projectors (mm.*)
    E2B    the same repo name with E2B (0.99 GB): the same encoders, projector out 1536
    26B    ~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/
           snapshots/b689.../mmproj-BF16.gguf (1.19 GB, on NFS): vision only
    Qwen   no mmproj. Download unsloth/Qwen3.6-35B-A3B-GGUF mmproj-BF16.gguf (ask the user)
    12B    models/gemma-4-12B-qat-q4_0/ (google/gemma-4-12B-it-qat-q4_0-gguf):
           gemma-4-12b-it-qat-q4_0.gguf (6.98 GB) and
           mmproj-gemma-4-12b-it-qat-q4_0.gguf (175 MB, 52M parameters):
           gemma4uv and gemma4ua, no layers
    12B    the HF reference: google/gemma-4-12B-it-qat-q4_0-unquantized in the
           cache of gemma4-12b-qat-pytorch (model.safetensors, 23.9 GB, BF16):
           Gemma4UnifiedForConditionalGeneration with vision_embedder,
           embed_vision, and embed_audio; config.json and processor_config.json

- The E4B snapshot google/gemma-4-E4B-it-qat-mobile-ct has config.json,
  preprocessor_config.json, and processor_config.json, and a second copy
  of the encoders (int8, compressed-tensors). Use its configs; do not use
  its int8 weights as the reference.
- The main GGUF files have no encoder tensors. /home has 37 GB free. Copy
  the 26B mmproj from NFS to models/.
- The venv has torch (CPU), transformers 5.17 (gemma4 and qwen3_5_moe
  vision code), pillow, soundfile, librosa, and scipy. It does not have
  the gguf package of llama.cpp; np_gemma/gguf.py reads the mmproj files.
- llama.cpp (../llama.cpp, tools/mtmd) is the second reference:
  models/gemma4v.cpp, gemma4a.cpp, qwen3vl.cpp, and mtmd-cli.

## 3. The encoders

### 3.1 Gemma 4 12B: no encoder layers (gemma4uv and gemma4ua)

The model card calls the 12B "unified": it has no vision encoder and no
audio encoder. The decoder of 48 layers (width 3840) does that work.

The image part (Gemma4UnifiedVisionEmbedder; llama.cpp gemma4uv.cpp):

1. The image: RGB, a bicubic resize that keeps the aspect, to sides that
   are multiples of 48. The pixels times 1/255, with no mean and no std.
2. The token budget: 70, 140, 280, 560, or 1120 soft tokens (the model
   card); the default is 280. Each soft token is one patch of 48 x 48 x 3
   = 6912 values. HF makes it from 3 x 3 patches of 16. The order of the
   6912 values must agree with v.patch_embd of the GGUF (check it).
3. LayerNorm of the 6912 values (v.patch_norm.1, with bias, eps 1e-5),
   the linear v.patch_embd (6912 to 3840, with bias), and LayerNorm
   (v.patch_norm.2).
4. Add a row of each position table (v.position_embd: 1120 rows for x,
   1120 for y). Then LayerNorm (v.patch_norm.3).
5. An RMS norm with no weight (eps 1e-6), and mm.input_projection (3840 to
   3840, BF16).

The audio part (gemma4ua.cpp; Gemma4UnifiedAudioFeatureExtractor):

1. 16 kHz mono. Frames of 640 samples (40 ms), with zeros after the last
   sample. No FFT and no mel. At most 30 s: 750 tokens.
2. An RMS norm of each frame with no weight (eps 1e-6), and
   mm.a.input_projection (640 to 3840, BF16).

The cost is small: 280 image tokens take about 7.5 G multiply-adds (0.1 s
in NumPy). The decoder does the work: with the mask of section 4.2, 280
image tokens cost about as much as 280 prompt tokens.

### 3.2 Gemma 4 vision (gemma4v: the E4B, the E2B, and the 26B)

One graph, two sizes:

                 E4B / E2B        26B
    layers       16               27
    width        768 (12 x 64)    1152 (16 x 72)
    FFN          3072             4304
    output       2560 / 1536      2816
    extras       clamp scalars    v.std_bias, v.std_scale

1. The image: resize with bicubic to sides that are multiples of 48 (the
   patch 16 times the pool 3), and keep the aspect. The token budgets are
   those of the 12B (section 3.1); the default is 280. The pixels are
   in [0, 1] (mean 0, std 1); the encoder starts with x = 2x - 1.
2. The patches: a linear of 16 x 16 x 3 values (v.patch_embd). Then add
   two learned position tables (x and y, v.position_embd, 10240 rows).
3. Each layer, the attention part:
   - an RMS norm, then q, k, and v (no bias);
   - an RMS norm for each head of q and k, and one with no weight for v;
   - a 2D RoPE (theta 100): half of the head takes the column, half the row;
   - attention with scale 1 and no mask, the output, a post norm, and the
     residual.
   Then the FFN part: the norm, the gated FFN, a post norm, and the
   residual.
4. The pool: an average of each 3 x 3 patch block, times sqrt(width).
5. The projector: (x - std_bias) * std_scale (26B only), an RMS norm with
   no weight, and mm.input_projection to the text width.
6. The clamp scalars (input_min, input_max, output_min, output_max) of the
   E4B linears clamp the input and the output of each linear.

Items to check against transformers (not llama.cpp):

- The FFN activation. HF gives gelu_pytorch_tanh; the gemma4v graph of
  llama.cpp can fall back to GELU_QUICK.
- The clamp scalars: the E4B config says use_clipped_linears false.
- The order of the patch values (the converter makes a conv of the HF
  linear, in HWC order).

### 3.3 Gemma 4 audio (gemma4a: the E4B and the E2B)

1. The mel front end:
   - 16 kHz; frames of 320 samples and a hop of 160; 160 zeros before the
     first frame;
   - a periodic Hann window and an FFT of 512;
   - the magnitude (not the power), 128 HTK mel bins from 0 to 8000 Hz;
   - log(max(mel, 0.001)).
   A clip of 30 s gives at most 750 tokens.
2. The subsample part: two Conv2d 3 x 3 with stride 2 (128 and 32
   channels). Each has a LayerNorm of the channels (weight only) and ReLU.
   Then a linear of 1024 values: about 25 tokens for each second.
3. 12 Conformer blocks of width 1024 (8 heads of 128). Each block: a half
   FFN (SiLU), the attention, the conv module, a second half FFN, and an
   RMS norm.
   - The attention is local and causal: chunks of 12 and 12 rows before.
     It adds a relative position term: 13 sinusoid rows through
     attn_k_rel, with the shift of Transformer-XL. Then a softcap of 50.
     q gets per_dim_scale (the GGUF has the softplus of it).
   - The conv module: pw1 (to 2048), GLU, a causal depthwise conv of 5,
     an RMS norm, SiLU, and pw2. The GGUF names conv_norm and norm_conv
     are the other way round (clip.cpp).
4. The projector: a.pre_encode.out (to 1536, with bias), an RMS norm with
   no weight, and mm.a.input_projection to the text width.
5. The eps is 1e-6 (the GGUF says 1e-5; llama.cpp uses 1e-6).

### 3.4 Qwen3.6 vision (the Qwen3-VL ViT)

Done (np_gemma/vision_qwen.py; README.md). The file gives
27 layers of width 1152, patch 16, a merge of 2 x 2, and a 48 x 48
position table. Qwen3.8-Flash-Next has the same ViT, with an output of
2560. Its weights are the tensors model.visual.* of the checkpoint.

1. The image: bicubic, sides that are multiples of 32. Tokens = (W / 32)
   (H / 32). Mean and std from the processor config.
2. The patch input: the Conv3d of two frames becomes two Conv2d on the
   same image, added. The patches go in the order of the 2 x 2 blocks.
3. The position table: a bilinear interpolation of the 48 x 48 table to
   the grid.
4. Each layer: LayerNorm and a fused qkv (both with bias), and a 2D RoPE
   (theta 1e4, rows and columns). Then attention with scale 1/sqrt(d) and
   no mask, and the output. Then LayerNorm and an FFN with GELU (with
   biases).
5. The merger: LayerNorm, 4 rows of the 2 x 2 block as one row of 4608,
   mm.0, GELU, mm.2 to the text width (2048). This GELU is the erf form
   (nn.GELU); the GELU of the layers is the tanh form.
6. Deepstack: deepstack_visual_indexes is empty in both models.

## 4. The text side

### 4.1 Soft tokens

- The prompt has a placeholder token for each soft token: Gemma 4
  <|image|> (258880) and <|audio|> (258881) between <|image> and <image|>
  (or <|audio> and <audio|>); Qwen <|image_pad|> (248056) between
  <|vision_start|> and <|vision_end|>.
- The model takes the encoder rows in place of the embeddings of those
  tokens. The rows do not get the scale of the token table (Gemma 4
  multiplies only the token rows by sqrt(hidden)).
- E4B per-layer inputs: the token part of a soft token is the row of the
  pad token (id 0). The 12B and the 26B have no per-layer inputs. The projection part comes from x, so it takes the
  soft row with no change.
- The places where a prompt turns ids into rows (all of them take an
  optional map of positions to rows):
  - E4B: e4b.py forward (the Python path), program.decode_step_e4b,
    E4BGPU.step and E4BGPU.group.
  - The 12B and the 26B (class Model): Model.forward,
    program.decode_step, and ModelGPU.group (a prompt pass is a series
    of groups).
  - Qwen: _QwenRuns.forward and QwenGPU._inputs.
- A prompt becomes a Prompt object: the ids, the soft rows (a map from a
  position to a row), and a key for each position. The key of a soft
  token is a hash of its media and its index. Session._common and
  Backend.pick_session compare the keys, so two images with the same
  placeholders do not share a cache.

### 4.2 The attention mask in an image (the 12B and the 26B)

The config of the 12B has use_bidirectional_attention = "vision". The
26B has it too: llama.cpp gives a mask that is not causal to every gemma4v
model but the E2B and the E4B.

Then the tokens of one image see
each other in the layers with a window. The mask is the window AND
(causal OR the same image). The global layers stay causal. The E2B and
the E4B are causal. Audio tokens are causal in all the models.

- llama.cpp makes the whole batch of an image not causal, in all the
  layers. HF keeps the global layers causal. HF is the reference; the
  difference goes in the README.
- Check the value in the config of the 26B. The config is not on disk;
  the HF repo google/gemma-4-26B-A4B-it has it.
- The kernels have only causal and window masks (ops.softmax_mask, the C
  flash kernels, k_flash_qc_mt, k_flash_tc, k_flash_qc_tc). Add an input
  for each query: the last key it sees (its position, or the end of its
  image). Only prompt passes need it.
- A prompt group must not split an image, or the mask must also reach the
  rows that the next group writes. The first form is simpler: move the
  end of a group to the end of the image.

### 4.3 M-RoPE of Qwen3.6

The full-attention layers of Qwen3.6 use interleaved M-RoPE: sections [11,
11, 10, 0] of the 32 frequency pairs of the 64 rotary dims. Pair j takes
the row position if j % 3 == 1, the column position if j % 3 == 2, else
the time position. For text all three are the same, so the runtime now
uses plain RoPE (QwenConfig ignores the sections).

- An image token i of an image with nx columns, after the text position
  p0: time p0, row p0 + i // nx, column p0 + i % nx. After the image the
  next text position is p0 + max(nx, ny), not p0 + the count of tokens.
- Thus the rope position is not the cache position any more. The Qwen
  programs take cos and sin as rows of input, so the change is on the
  host: Qwen.rope takes an array (3, t) of positions, and the Session
  keeps the offset between the cache position and the rope position.
- The DeltaNet layers have no positions.
- No server runs Qwen3.6. scripts/serve_qwen4.py serves Qwen3.8 only;
  it needs a Qwen3.6 backend (the template of the GGUF, through jinja2).

### 4.4 The order and the size of media

- The model card says: images before the text, audio after the text.
  The server keeps the order of the request; the README gives the advice.
- A clip of more than 30 s: the server cuts it into clips of 30 s, or it
  gives an error (an option). Each clip goes between <|audio> and <audio|>.
- Each image takes up to 1120 tokens and each clip up to 750. The server
  counts them in --max-context.

## 5. The server

- The function content_text of the server keeps only the text parts now.
  The function render_chat (chat.py) already writes the placeholders for
  the parts image_url, image, input_audio, and audio. Give it the parts.
- image_url: a data URI (base64) or a path under an allowed directory
  (an option). No fetch from the network by default.
- input_audio: base64 WAV or MP3 (soundfile), resampled to 16 kHz.
- The processor: the image resize, the audio mel, the count of soft
  tokens, and the placeholders in the prompt ids.
- The encoder runs before the prompt pass, on the GPU if the server has
  one. A cache of encoder outputs (by the hash of the media) saves the
  work when a chat sends the same image again in each turn.
- The log and the --debug records give the media: kind, size, hash, and
  the count of soft tokens. They do not keep the media data.

## 6. The phases

Each phase ends with a commit and a check script. TEST_PLAN.md gives the
tiers; these checks go in them.

### Phase 0: references

1. Load the mmproj files with np_gemma/gguf.py. Print the tensors and the
   clip keys. Confirm the sizes of section 3 (26B: 27 layers of 1152).
2. The 12B GGUF: run the text of the 12B in this runtime (class Model).
   Compare it with llama.cpp and with the float32 mode on the safetensors
   (the chat text method). A first run (the CPU, int4) loads
   in 4.5 s and gives correct text ("Red, Orange, Yellow"), at 6.3 tok/s
   for 5 tokens. The comparison is still to do.
3. The 12B HF reference: transformers Gemma4UnifiedForConditionalGeneration
   from the unquantized QAT safetensors, in BF16 on the CPU (about 24 GB)
   or float32 (about 48 GB). Keep the soft rows and the logits of 3
   images and 2 clips (scripts/hf_mm_reference.py --model 12b).
4. Make an HF reference of the E4B encoders (scripts/hf_mm_reference.py).
   It is the transformers Gemma4VisionModel and Gemma4AudioModel with the
   weights from the mmproj. Reverse the changes
   of the converter: the patch conv back to a linear, the inverse
   softplus of per_dim_scale, the conv kernels. If the full BF16 model
   google/gemma-4-E4B-it can be downloaded (ask the user), use its
   weights instead: then the GGUF reader is also checked.
5. Run llama.cpp mtmd-cli on the 12B and the E4B with 3 images and 2
   clips. Keep the answers and the times as the second reference.

Test: the HF reference gives a correct answer to a question on a known
image (for example "what color is the car").

### Phase 1: the text side, on the 12B

1. np_gemma/unified.py, in NumPy:
   - the image processor (the budgets, the 48 x 48 patches, the positions);
   - the audio frames;
   - the two embedders of section 3.1.
   scripts/check_mm_unified.py: the patches and the
   soft rows against HF. Pass: max relative difference 1e-4.
2. The Prompt object and the soft rows at the places of section 4.1
   (Model.forward, program.decode_step, ModelGPU.group).
3. The mask of section 4.2. The CPU and the GPU prompt kernels get the
   input of the last key for each query, only in the layers with a window.
   The kernels: ops.softmax_mask, the C flash kernels, k_flash_qc_mt,
   k_flash_tc, and k_flash_qc_tc. A group ends at the end of an image. check_flash_c and
   check_gpu_split get cases with an image block.
4. The server: the content parts, the processor, and the media keys of
   the sessions. A test of TEST_PLAN.md tier 2: a chat that sends the
   same image in each turn reuses the cache; another image does not.
5. scripts/check_mm_prompt.py --model 12b: the logits of a prompt with an
   image and one with a clip, against HF on the same soft rows. Pass: KL
   of the top 64 below 0.01, and the same top token on 95% of an answer
   of 64 tokens.
6. A test with 10 images and 5 clips (the count of objects, a color, text
   in the image, speech to text). Compare with llama.cpp.

### Phase 1b: E4B vision in NumPy

1. np_gemma/vision.py: the image processor and the gemma4v encoder in
   float32 NumPy, from the mmproj.
2. scripts/check_mm_vision.py: the soft rows of 3 images against the HF
   reference. Pass: max relative difference 1e-4.
3. Settle the items of section 3.1 (the activation, the clamps).

### Phase 2: the E4B text side

1. The soft rows at the four E4B places of section 4.1, and the pad row
   of the per-layer inputs. The Prompt object and the server come from
   phase 1.
2. The E4B is causal: no mask.
3. scripts/check_mm_prompt.py --model e4b compares the logits of a
   prompt with an image with HF on the same soft rows (CPU and GPU).
   Pass: KL of the top 64 below 0.01, and the same top token on 95% of an
   answer of 64 tokens. This is the chat text method of TEST_PLAN.md.
4. A test with 10 images and simple questions (the count of objects, a
   color, text in the image). Compare with the answers of llama.cpp.

### Phase 3: a fast vision encoder

1. The CPU: the linears on the BF16 kernels (gemma_bf16_gemm). The
   attention of all patches (no mask) on the flash kernel, with no causal
   limit.
2. The GPU: the linears on k_gemm_bh (half x, BF16 weights), and an
   attention kernel with no mask.
3. Pass: the rows of phase 1 to 1e-3. Target: an image of 280 tokens (2520
   patches, about 0.8 TFLOP) in less than 1.5 s on the CPU and 0.1 s on
   the GPU. Measure llama.cpp on the same image.

### Phase 4: E4B audio

1. np_gemma/audio.py: the mel front end, then the Conformer in NumPy.
2. scripts/check_mm_audio.py: the mel against the HF feature extractor
   (1e-5), the soft rows against the HF audio model (1e-4).
3. The text side and the server as phase 2 (input_audio).
4. A test with 5 clips: speech to text, and a question on the clip.
5. The fast form: the linears as phase 3; the local attention in chunks of
   12; the depthwise conv (as the DeltaNet conv, csrc/deltanet.c).
   Target: 30 s of audio in less than 1 s on the CPU.

### Phase 5: 26B vision

1. The encoder of phase 1 and phase 3 with the 26B sizes and the
   std_bias and std_scale of the projector.
2. The mask of section 4.2 comes from phase 1. Check it on the 26B
   prompt kernels (the mixed groups and the MoE parts).
3. The mixed groups and the hot experts need no change: the soft rows
   only change the input of layer 0.
4. MTP: the drafter reads the cache rows of the image like other rows.
   check_mtp gets a prompt with an image.
5. scripts/check_mm_prompt.py --model 26b, as phase 2.

### Phase 6: Qwen3.6 vision (done, and Qwen3.8)

1. Download the mmproj (ask the user). Check the sizes and deepstack.
2. np_gemma/vision_qwen.py: the processor and the ViT (section 3.3); the
   HF reference from transformers qwen3_5_moe (the vision classes).
3. M-RoPE (section 4.3): Qwen.rope with (3, t) positions, the offset in
   the Session, the CPU and GPU programs. Check: text only gives the same
   bits as now.
4. A Qwen3.6 backend of the server, with the template of the GGUF.
5. scripts/check_mm_prompt.py --model qwen36, as phase 2.

### Phase 7: the measures

- The time of the encoder and of the prompt pass, against llama.cpp: 1
  and 4 images, and 10 s and 30 s of audio.
- The README section, and the numbers in HANDOFF.

## 7. Risks

- The HF reference from GGUF tensors can repeat an error of the converter.
  The full BF16 model avoids that; it is a download of about 16 GB.
- The mask needs changes in 5 attention kernels. The 12B needs it, so
  it comes in phase 1, before the encoders.
- The HF 12B in float32 needs about 48 GB of RAM, and the user runs other
  programs. BF16 (24 GB) is the first choice; the checks then have a
  tolerance for BF16.
- The 12B GGUF path of the runtime gives correct text, but it has no
  comparison yet. Phase 0 does it before any image work.
- The 26B mmproj is on NFS; a copy on /home takes 1.2 GB.
- Images add tokens: 280 for each image by default, and up to 1120. With --max-context and the
  sessions, a chat with many images uses the cache fast.
- M-RoPE changes the positions of every Qwen3.6 path. The check that text
  gives the same bits is the guard.
- The E4B mobile-ct configs can differ from those of the E4B GGUF model
  (they come from another checkpoint of the same model).
