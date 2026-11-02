# DiffusionGemma 26B-A4B

google/diffusiongemma-26B-A4B-it (Apache 2.0): Gemma 4 26B-A4B's MoE as a
block-diffusion text model. The source of truth is transformers'
`models/diffusion_gemma` (modeling and generation files); the NumPy port is
`reclank/models/dgemma/`.

## 1. The model

- The 30 layers of Gemma 4 26B-A4B (25 with a window of 1024, 5 global, the
  shared MLP and 128 experts with 8 a token, hidden 2816) are used twice with
  the same weights: as a causal **encoder** that only fills the KV cache, and
  as a **decoder** over a canvas of 256 tokens. Each layer has two
  `layer_scalar` values: one for the encoder and one for the decoder. Nothing
  else differs.
- Decoder attention: the canvas sits at the positions after the cache and
  sees every canvas row and the cache, with no causal mask. A global layer
  reads every cached row. A layer with a window reads the last 1023 cached
  rows, as transformers' `DynamicSlidingWindowLayer` keeps them (the default
  `generate` cache). The decoder never writes the cache.
- Self-conditioning, before the first decoder layer:
  `x = rmsnorm(embed(canvas) + mlp(rmsnorm(soft) w))`, where the last norm has
  no weight and `soft = softmax(the last step's processed logits) @ the token
  table * sqrt(2816)`. At the first step, soft is zero and so is the MLP's
  output.
- The head is the tied token table with the soft cap of 30.
- Vision: the Gemma 4 vision tower of 27 layers, at 70 to 1120 tokens an
  image (not in the first phases).

## 2. Sampling (generation_config.json)

Canvases are produced one after the other. Each is denoised in at most 48
decoder passes, with the step counted down from 48 to 1:

- Temperature: `0.4 + 0.4 * step / 48` on the logits.
- Each position draws a sample and takes its argmax.
- Entropy-bounded acceptance: keep the k lowest-entropy positions whose
  entropies, less the largest of them, sum to at most 0.1. Those positions
  take their sample; the rest become uniform random tokens.
- Stop early when the argmax canvas equals the last step's and the mean
  entropy is under 0.005.
- The canvas's result is the last argmax canvas. It is encoded (appended to
  the cache) unless it holds an end token (1, 106 or 50); anything after the
  first end token is dropped.

`reclank/models/dgemma/sampler.py` holds this loop over two callables,
`encode` and `decode`, so the reference and the engines share it.

## 3. Cost of a decoder pass

256 rows through 30 layers, which is a prompt group of 256 tokens. On top of
that:

- the head over 256 rows: 0.38 TFLOP;
- the soft embedding (256 x 262144 by 262144 x 2816): another 0.38 TFLOP;
- the entropy, sample and argmax of 256 rows of 262144 values.

On the GPU all of this stays on the device. The host needs only the argmax
row, the accepted mask and the mean entropy, about 2 KB a step. The processed
logits stay where they are for the next step's self-conditioning. The model
card gives 15 to 20 tokens a pass.

## 4. Weights

- The float32 reference loads Google's bf16 safetensors as float32 (about
  103 GB of memory).
- The engine runs on **BF12**, chosen after the measurements of section 6:
  the bf16 values in 12.25 bits, exact except for the values more than 15
  binades under the largest of their group of 32, which become zero.
  `scripts/convert_dgemma_bf12.py` writes one GGUF file with llama.cpp's
  diffusion-gemma names; the metadata and tokenizer come from unsloth's GGUF
  of the same checkpoint. Across 269 matrices (25.24 G values), 0.0043% of
  values are zeroed and no matrix stays bf16. The file is 38.85 GB.
- 4-bit forms are not used: our Q4_0 (quantized from bf16) and unsloth's
  Q4_K_M both cost far more KL than the float32 noise (section 6).

## 5. Phases

- D1: the NumPy float32 reference (`model.py`) and the sampler. Check them
  against transformers 5.20.2 run in the sandbox (CPU float32, two real
  prompts, one of them past the window; the logits of fixed canvases with and
  without self-conditioning, and a second block).
- D2: the Q4_0 engine on the CPU. The encoder is the Gemma 4 prompt path with
  the encoder scalars. The decoder is a canvas pass on the same kernels, with
  the canvas's own attention rows. Measure KL against float32 on the
  reference's canvases.
- D3: the GPU. A canvas program: a group of 256 rows, the attention with a
  fixed first row for the windowed layers and no causal mask, the
  self-conditioning GEMM, and the head with the sampler's per-row reductions
  on the device.
- D4: the server (an engine family, the chat template from the checkpoint,
  thinking via `<|think|>`) and a CLI.
- D5: the Q4_K_M forms, and vision.

## 6. Status

D1, the reference (`model.py`), against transformers 5.20.2 in the sandbox
(CPU float32). Two real prompts: a short question of 21 tokens, and a design
note of 2003 tokens, past the window. For each, three decoder passes: a fixed
random canvas with no self-conditioning (l1); one with self-conditioning from
l1 / 0.8 (l2); and one after the encoder has taken a block (l3).

- The rope tables must be float32, as transformers makes them (`inv_freq` and
  `pos * inv_freq` in float32). Float64 tables move the phases by about 1e-4
  rad at 2000 tokens. That alone cost the long prompt's l1 a KL of 2.3e-2;
  with float32 tables it is 8.5e-3.
- Short prompt: the encoder's last hidden state matches to 1.7e-6 relative;
  l1 has KL 1e-8 and top-1 1.000; l3 has KL 3e-9.
- What remains is router ties, not porting errors. Traced layer by layer,
  the first difference on the long prompt is layer 10, row 339, where the 8th
  and 9th experts tie exactly in float32. For the self-conditioned pass it
  is layer 14, row 170, at a gap of 4.7e-6. The bidirectional decoder spreads
  one flipped row to the whole canvas within a few layers. So the float32
  noise floor of these checks is a KL of about 5e-4 (self-conditioned passes)
  to about 1e-2 (long prompt, l1). The soft embedding and the
  self-conditioning output match to 1e-6.

D2, the runtime (`runtime.py`: Gemma 4's Model, with the canvas pass in
Python on its kernels):

- bf16 mode, the checkpoint's own weights: short l1 KL 1e-4, l3 0.0000;
  long l1 KL 0.026 (float64 rope plus the int16 KV cache). The runtime is
  right.
- Weight forms in the float32 reference, so that only the weights differ.
  KL on the short prompt:

  | form | l1 | l2 | l3 |
  |---|---|---|---|
  | bf16 (runtime) | 1e-4 | 0.029 | 0.0000 |
  | Q8_0 | 0.043 | 0.019 | 0.006 |
  | Q4_K_M (unsloth) | 0.21 | 0.19 | 0.098 |
  | Q4_0 (ours, any table) | 0.29 | 0.34 | 0.42 |

  On the long prompt, Q4_K_M has l1 0.65 and l3 0.34 (top-1 0.34). The int4
  runtime gives the same KL as Q4_0 weights in float32 arithmetic, so its
  error is all in the weights.

Pinned text (`sampler.py`, `fixed`; `scripts/dgemma_generate.py
--response`): the answer as a template of pinned text and `[[n]]` free
tokens. A pinned start is a prefix. The other pinned tokens start in place,
are never renoised and stand in the result. The entropy bound and the stop
test count only the free positions, and no end token comes before the last
pinned one. Tests: `tests/unit/test_dgemma_sampler.py`.

Speed so far is the CPU's only: a canvas pass takes 12 s in bf16 mode and 7
s in int4. D3 (the GPU on BF12) is next.
