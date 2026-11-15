# Handoff: DiffusionGemma vision

This file gives the state of the vision work on DiffusionGemma. The plan
`DIFFUSIONGEMMA_PLAN.md` (D5) has the short version. This file has the
detail that the next person needs: the mask, the block limit, the exact
numbers, the commands, and the traps.

## 1. State now

- Committed with this file (section 9 lists the files).
- `--mmproj` is DiffusionGemma's own vision tower,
  `diffusiongemma-26B-A4B-it-mmproj.gguf` beside the BF12 GGUF.
  `scripts/dgemma_mmproj.py` writes it from the checkpoint. The BF12 GGUF
  does not hold the tower. Do not use the Gemma 4 26B mmproj: its
  projection differs (section 2).
- The BF12 GPU encoder matches transformers' float32 image prompt (the
  `generate` path, with the vision mask) at budgets 140, 280 and 560: KL
  0.0000 to 0.0003 on transformers' pixels, top-1 1.000. Section 5.
- An image that a 256-row block would split takes a prompt block of `P`
  rows (768 by default: budget 560 fits). Section 4.
- The server (`M=dg ./start_server.sh` passes the mmproj) answers image
  requests and a second turn with the image in the history.
- The first pass had two faults, both fixed: the Gemma 4 mmproj (another
  projection), and the mask of a block after the first (the span's
  positions offset by the block's start; `encoders/media.encoder_blocks`
  now plans the blocks, with tests).

## 2. The models and the files on disk

```
~/models/diffusiongemma-26B-A4B-it-BF12-GGUF/
    diffusiongemma-26B-A4B-it-BF12.gguf        38.87 GB, the text model
    diffusiongemma-26B-A4B-it-mmproj.gguf      1.19 GB, the vision tower (scripts/dgemma_mmproj.py)
    config.json, generation_config.json, tokenizer.json, chat_template.jinja
    expert_counts.npz                          the routers' counts
~/models/diffusiongemma-26B-A4B-it/   the bf16 checkpoint, 11 safetensors (with the tower)
```

- The checkpoint holds its own vision tower: `model.encoder.vision_tower`
  (27 layers, 1152 wide) and `model.encoder.embed_vision` (the projection
  to 2816). The tower's 355 tensors are the Gemma 4 26B mmproj's (within
  2e-5, bf16 rounding of a layout change). The projection
  `embed_vision.embedding_projection` is not: against the Gemma 4 26B
  mmproj's `mm.input_projection` it has 1.53 times the norm and 20% left
  after the best scale of each row. So the Gemma 4 mmproj gives other soft
  rows; the first pass of this work used it.
- `scripts/dgemma_mmproj.py` writes the mmproj: the keys, names, types and
  layouts of a Gemma 4 26B mmproj (`--like`), every value from the
  checkpoint (bf16 as stored). It checks each tower tensor against `--like`
  as a check of the mapping. It needs llama.cpp's gguf-py on PYTHONPATH.
  `_gguf_file` skips a `*mmproj*` file beside the model.
- The engine prints a warning when the mmproj's `general.name` is not
  DiffusionGemma's.
- The CPU checks need a venv with NumPy, pytest, and Pillow. This session
  cannot write `~/.cache`, so the CPU kernels build with
  `RECLANK_LIBS_DIR=$PWD/.scratch-libs`. The normal environment of the user can
  use the default cache.

## 3. What was built

| File | What it does |
| --- | --- |
| `reclank/encoders/media.py` | `limit_positions` (Gemma 4's `_media_limit`), `block_end` (do not split a short span), `place_rows` (the soft rows into an encoder block) |
| `reclank/models/dgemma/gpu.py` | `DGemmaGPU.encode(ids, spans=None)`: image rows in the encoder input, block alignment, the mask in `his` and `hif` |
| `reclank/models/dgemma/model.py` | `DGemmaRef.encode(..., spans=None)`: the same mask in the reference |
| `reclank/models/dgemma/runtime.py` | `DGemma.encode(..., spans=None)`: the spans go to Gemma 4's `prefill`, which has the mask |
| `reclank/models/dgemma/engine.py` | `--mmproj`, `media_prompt`, `DGPrompt.spans`, the cache reset for an image, the hidden-size check |
| `scripts/dgemma_generate.py` | `--image`, `--image-budget`, `--mmproj`; spans for `gpu`, `ref`, `bf16`, `q4` |
| `scripts/dgemma_prompt.py` | `--image` as a data URI for the server |
| `scripts/dgemma_mmproj.py` | DiffusionGemma's mmproj GGUF from its checkpoint (its own projection) |
| `scripts/check_dgemma_gpu.py` | a `DG_ATTN` case with an image block |
| `server/options.py` | the `--mmproj` help names DiffusionGemma |
| `docs/DIFFUSIONGEMMA_PLAN.md` | D5, the numbers, the block limit |

The engine path, in order:

1. `prompt_ids` renders the Gemma 4 chat with the image part.
2. `media.expand` replaces `<|image|>` with `<|image>`, the soft rows, and
   `<image|>`.
3. `DGPrompt` carries the spans. The response template still comes from the
   last assistant message.
4. `generate` resets the cache when spans exist, then passes the spans to
   the first `encode` call only.
5. The encoder puts the rows in place of the image token embeddings.
6. The decoder reads the cache as for text. It sees every image row.

## 4. The image mask, exactly

The HF class `DiffusionGemmaEncoderTextModel.forward` builds **causal**
masks when no mask dict comes in. This is the plain forward path. The
`generate` path is different:

- `_prepare_encoder_inputs` calls
  `create_masks_for_generate(..., mm_token_type_ids=...)`.
- With `use_bidirectional_attention: "vision"`, that builds
  `block_sequence_ids` and adds `blockwise_overlay`.
- The overlay ORs the causal mask with "same image block". An image token
  sees every image token in its block, and future ones. Text stays
  causal. The image block still sees the text before it.

So the causal encoder was a real difference for image generation.

The implementation:

- `limit_positions` gives the last key of each query. A query inside a
  bidirectional span gets the last position of the span. Other queries get their
  own position.
- The mask is `key <= limit`. The tests prove this equals
  `causal OR same image block`, and the sliding union when the image is
  shorter than the window.
- The GPU encoder sets `his` (sliding) and `hif` (global) from `limit`.
  The reference uses the same `limit` in its mask.
- A block that starts before a span that fits in one block ends at the
  start of the span. The next block starts at the image. So a 140-token
  image is not split.
- A span of 257 rows or more cannot fit in the current 256-row program.
  The rows of the first block cannot see the rows of the next. This is the
  one known mask gap, in the GPU prefill chunking, not in the model. The
  budget is a maximum: the image here at budget 280 gives 252 rows, so it
  does fit.
- The decoder is unaffected. It reads all committed image rows from the
  cache.

The cache between requests is off for a prompt with an image. The token ids
do not tell one image from another, so `DGemmaGPU.reuse` cannot check the
rows. A media hash in the cache key is future work.

### Prefill, not canvas

An image is prompt context. It does not enter the 256-token canvas, and the
decoder never denoises it. The prefill of the model sees the whole prompt and the
mask can span the whole image. The reference (`DGemmaRef`) already does
this: it runs the full prompt and its mask is exact for any span. The bf16
runtime does this too, through the Gemma 4 `prefill`.

The 256-row block is only in `DGemmaGPU.encode`: the `enc` CUDA graph is
built with `C = canvas_length` rows, so the method chunks the prompt into
256-row pieces. That was a shortcut to reuse the canvas program. It is not a
model constraint, and it is not the final design.

### The prompt block `P`

`encoders/media.encoder_blocks(n, count, C, spans, big=P)` plans the
blocks of a prompt: `(a, end, rows, limit)` each, `limit` the last key of
each row at its absolute position (the spans are relative to the prompt,
whose first id is at cache position `n`).

- Text takes blocks of `C` (256), as before: the same bits.
- A block ends before an image that fits in one block (`block_end`).
- An image that a block of `C` would split takes a block of `P` rows: the
  program `encP` (`_program` takes the rows from the kind), built at the
  first such image.
- `P` (`DGemmaGPU(prompt_rows=)`, `RECLANK_DG_PROMPT_ROWS`) is 768 by
  default. The engine and `dgemma_generate.py` take the image budget
  rounded up to 256, at least 768. The windowed layers' buffers grow to
  `window - 1 + P + C` rows when that passes 2048 (768: 2047, no change).
- The CPU experts' pinned rows (`hp`, `po`, `vp`, `ip`, the KQ_MOE scratch)
  hold `P` rows; a program of `C` rows uses their first `C`.
- `_room(rows)` makes room for a block of `rows` and a canvas after it.
- An image longer than `P` crosses blocks (its rows see their block's);
  the engine prints a line.

The cache between requests is off for a prompt with an image. The token ids
do not tell one image from another, so `DGemmaGPU.reuse` cannot check the
rows. A media hash in the cache key is future work.

## 5. Checks and numbers

The reference: transformers 5.20.2 from PyPI, float32 on the CPU, in the
sandbox (driver `dgemma_hf_image.py`; dumps
`hf_image_{140,280,560}.npz` in the sandbox's `out/dgemma`).
Per budget: the ids, `mm_token_type_ids`, `pixel_values`, the soft rows
(`get_image_features`), a fixed random canvas `c1` and the decoder's logits
for it after the encoder over the prompt with the vision mask (`l1`) and
with the plain causal mask (`l1_causal`).

`scripts/check_dgemma_image.py` (PASS):

| budget | tokens | image rows | block | soft rows | KL (theirs) | KL (own pixels) | KL, causal mask |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 140 | 153 | 135 | 256 | 2.0e-5 | 0.0000 | 0.0035 | 0.0708 |
| 280 | 270 | 252 | 256 | 2.9e-5 | 0.0001 | 0.0180 | 0.1140 |
| 560 | 558 | 540 | 768 | 2.6e-5 | 0.0003 | 0.0140 | 0.0455 |

- The ids and the image's span are transformers' at all three.
- "theirs": the tower on transformers' pixels; top-1 1.000.
- "own pixels": the engine's resize, PIL's bicubic (soft rows 1.1e-3 to
  1.5e-3). It is transformers' PIL image processor exactly at 140 and 560;
  transformers' default torchvision processor puts 170 to 1275 values (of
  0.9 to 3.7 million) one level apart, and that alone moves the KL by
  0.004 to 0.018.
- With the Gemma 4 26B mmproj: soft rows 0.41 to 0.44 off, KL 0.017 to
  0.070.
- Before the block fix, the 280 prompt (three blocks) had KL 0.014 on
  transformers' soft rows; after it 0.0001.

`scripts/check_dgemma_gpu.py` (PASS): DG_ATTN with an image's ranges in
blocks of 256, 512, 1024 and 2048 queries (1.4e-6 at most), DG_MOE on 256
to 2048 rows (6.7e-7 against NumPy, 4.9e-7 against KQ_MOE).

The server, budget 280: "This line graph titled "Shadow gamma vs net
gamma" displays the dealer gamma in millions of dollars per 1% move across
various SPX spot prices for August 11, 2026." (47 tokens, 7 passes, 3.6 s
warm; the first request 86 s: the tower's programs on the CPU and `encP`
built). A second turn ("At which SPX price does the net gamma line cross
zero?") re-encodes the prompt with the image (338 tokens, 5.9 s) and
answers from the chart.

Unit tests: `test_media_spans.py` (16, with `encoder_blocks`),
`test_dgemma_sampler.py`, `test_dgemma_vision_prompt.py`,
`test_dgemma_engine_vision.py`: 26 pass.

The first pass's CPU runs below used the Gemma 4 26B mmproj (its projection, not
DiffusionGemma's: section 2). Their answers are plausible because the two
projections are close (correlation 0.98), but their soft rows are not the
model's. Run them again with DiffusionGemma's mmproj.

The CPU bf16 runtime, one image, budget 70:

- 60 soft rows of 2816 values; 78 prompt tokens.
- 7 decoder passes; 209.8 s; 16 tokens (the cap); 2.3 tokens a pass.
- Output: `This line graph plots "shadow gamma" and "net gamma` (the
  answer was cut at the 16-token cap).
- This checks the prompt path, the soft rows, and the Gemma 4 mask in the
  runtime. It does **not** check the BF12 GPU encoder.

The CPU bf16 runtime, one image, budget 280:

- 252 soft rows (the budget is a maximum; the grid is 252 rows); 270 prompt
  tokens. The span is shorter than the 256-row block, so it fits.
- 7 decoder passes; 164.6 s; 8 tokens (the cap); 1.1 tokens a pass.
- Output: `This line graph displays` (cut at the cap).
- This is a near-full-block case. It checks the prompt path and the mask in
  the runtime. It is not the over-256 case.

Not checked: the bf16 runtime and the reference with DiffusionGemma's
mmproj (their mask is the whole prompt's at once, as transformers').

## 6. How to run on the host

DiffusionGemma's mmproj (once; its vision tower from the checkpoint):

```sh
PYTHONPATH=llama.cpp/gguf-py python scripts/dgemma_mmproj.py \
  --checkpoint ~/models/diffusiongemma-26B-A4B-it \
  --like ~/models/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-it-mmproj.gguf \
  --out ~/models/diffusiongemma-26B-A4B-it-BF12-GGUF/diffusiongemma-26B-A4B-it-mmproj.gguf
```

The GPU check:

```sh
python scripts/check_dgemma_gpu.py \
  ~/models/diffusiongemma-26B-A4B-it-BF12-GGUF
```

One image, GPU, exact-mask budget:

```sh
python scripts/dgemma_generate.py \
  ~/models/diffusiongemma-26B-A4B-it-BF12-GGUF \
  --prompt "Describe this image in one sentence." \
  --image chart.jpg \
  --mmproj ~/models/diffusiongemma-26B-A4B-it-BF12-GGUF/diffusiongemma-26B-A4B-it-mmproj.gguf \
  --engine gpu --image-budget 140 --max-tokens 64
```

The server:

```sh
python -m server \
  -m ~/models/diffusiongemma-26B-A4B-it-BF12-GGUF/diffusiongemma-26B-A4B-it-BF12.gguf \
  --mmproj ~/models/diffusiongemma-26B-A4B-it-BF12-GGUF/diffusiongemma-26B-A4B-it-mmproj.gguf \
  --image-budget 140
```

```sh
python scripts/dgemma_prompt.py \
  --prompt "Describe this image in one sentence." \
  --image chart.jpg \
  --drafts
```

The CPU fallback (bf16 runtime, about 210 s for one small answer):

```sh
RECLANK_LIBS_DIR=$PWD/.scratch-libs .venv/bin/python scripts/dgemma_generate.py \
  ~/models/diffusiongemma-26B-A4B-it \
  --prompt "Describe this image in one sentence." \
  --image chart.jpg \
  --mmproj ~/models/diffusiongemma-26B-A4B-it-BF12-GGUF/diffusiongemma-26B-A4B-it-mmproj.gguf \
  --engine bf16 --image-budget 70 --max-tokens 16
```

## 7. Gotchas

- `--mmproj` is DiffusionGemma's mmproj (`scripts/dgemma_mmproj.py`). The
  BF12 GGUF has no vision tower. The Gemma 4 26B mmproj loads (the same
  hidden size) but its projection is not DiffusionGemma's; the engine
  warns. A mmproj of another hidden size fails at load.
- The tower runs on the CPU by default. The 26B tower holds about 2 GB of
  the GPU after the first image. The experts already fill a 24 GB card.
  `RECLANK_MEDIA_GPU=1` or `--mmproj-gpu on` moves the tower to the GPU.
- The default image budget is 280 (up to 280 rows: 252 here). An image
  past 256 rows takes the `P` block; one past `P` crosses blocks (raise
  `--image-budget`, which raises `P`).
- The first pass's session could not write `~/.cache` and set
  `RECLANK_LIBS_DIR` to a writable directory; the host needs neither.
- The CPU run needs Pillow. The first failure at budget 280 was
  `ModuleNotFoundError: PIL` because the venv was recreated without it.
- The response template cannot hold an image. The engine raises a clear
  error.
- `_gguf_file` skips a `*mmproj*` file beside the BF12 GGUF.
- A request with an image resets the cache. A later text-only turn can still
  reuse the cached prefix if the ids match. The image rows are then the same
  image, which is correct for the same conversation. A different image with
  the same ids is wrong; the media hash is the fix.
- The GPU `DG_ATTN` `his`/`hif` ranges for an image can point forward. The
  check's media cases cover that. The kernel writes all block K/V before the
  attention, so the forward keys exist.
- A block pads with zeros past its prompt's ids. `self.n` advances by the
  real count, and the next block overwrites the padding. The check runs on
  the real rows only.

## 8. Next steps, in order

1. Try the tower on the GPU (`--mmproj-gpu on`) and measure the memory and
   the time on the 24 GB card (the first image's tower on the CPU is the
   slow part of a request).
2. A media hash in the cache key, so a later turn reuses the cached prompt
   with its image.
3. The resize as transformers' default (torchvision's antialiased bicubic
   on uint8), if the last 0.004 to 0.018 of KL matters.
4. Budget 1120 (up to 1092 rows): `P` 1280 and windowed buffers of 2559
   rows (the engine sets both from `--image-budget 1120`); check the memory
   and the KL against transformers.
5. Video frames: `Gemma4Embedder.video` and `media.video_text` already
   exist, but the dgemma engine raises for video. Audio is not supported by
   DiffusionGemma.

## 9. Files changed

```
docs/DIFFUSIONGEMMA_PLAN.md
docs/HANDOFF_DG_VISION.md                 this file
reclank/encoders/media.py
reclank/models/dgemma/engine.py
reclank/models/dgemma/gpu.py
reclank/models/dgemma/model.py
reclank/models/dgemma/runtime.py
scripts/check_dgemma_gpu.py
scripts/dgemma_generate.py
scripts/dgemma_prompt.py
scripts/dgemma_mmproj.py
scripts/check_dgemma_image.py
server/options.py
tests/unit/test_dgemma_engine_vision.py
tests/unit/test_dgemma_vision_prompt.py
tests/unit/test_media_spans.py
```

## 10. Session notes

- The first assumption was that the dgemma encoder was causal for images,
  because `DiffusionGemmaEncoderTextModel.forward` is causal. The
  `generate` path builds the vision mask itself. Read both paths before
  a change to the mask.
- The CPU runtime number is slow (about 30 s a pass). It is a correctness
  check, not a speed measure. The GPU numbers are in
  `DIFFUSIONGEMMA_PLAN.md`.
- The image used is a 46 KB line chart. At budget 280 it gives 252 rows (a
  256-row block); budget 560 (540 rows) is the case of the `P` block.
- The 280 budget run first failed because the venv had no Pillow. Install
  Pillow before any image run. The rerun gave 252 rows and a sensible start.
- The first pass's session had no GPU device nodes; the GPU checks of
  section 5 were run on the host afterwards.
- The mmproj's metadata and the tower's shapes do not tell the projections
  apart: compare the weights with the checkpoint's (as
  `scripts/dgemma_mmproj.py` does) before trusting a model's mmproj.
