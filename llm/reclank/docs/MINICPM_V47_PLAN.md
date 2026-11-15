# MiniCPM-V 4.7 35B-A3B

How to support openbmb/MiniCPM-V-4.7-35B-A3B (bfloat16 checkpoint, 70.6 GB,
16 shards) and its DSpark drafter (openbmb/MiniCPM-V-4.7-35B-A3B-DSpark,
0.78 GB). Sources read: the config.json and tensor index of both
checkpoints, the transformers code (models/minicpmv4_7, on minicpmv4_6), and
vLLM's qwen3_dspark.py / qwen3_dflash.py / v1/spec_decode/dflash.py.

## 1. What the model is

The text model is Qwen3.5-35B-A3B with no changes. MiniCPMV4_7Model builds
`AutoModel.from_config(text_config)` with model_type `qwen3_5_moe_text`, which
is the same class as Qwen3.6-35B-A3B:

- 40 layers: in each group of 4, 3 Gated DeltaNet layers then 1 gated full
  attention layer;
- hidden size 2048; 256 experts with 8 per token; expert size 512; a shared
  expert of 512 with a sigmoid gate;
- M-RoPE interleaved, sections 11/11/10, partial rotary 0.25, theta 1e7;
  context 262144.

Differences from Qwen3.6 that matter:

| | Qwen3.6-35B-A3B (what reclank runs) | MiniCPM-V 4.7 |
|---|---|---|
| vocab | Qwen3.6 table | 248144 rows; the image, video and marker ids sit in it (image_pad 248056 and video_pad 248057 are the same) |
| weights | fine-tuned separately | own fine-tune: nothing can be shared with the Qwen3.6 files |
| experts in the checkpoint | per-expert | fused, as in transformers 5: `experts.gate_up_proj` and `experts.down_proj` (the same layout as Qwen3.8 ORIG, which convert_q8_gguf.py already reads) |
| MTP | MTP layer in the checkpoint | `mtp_num_hidden_layers: 1` in the config but **no MTP tensors**: the drafter is DSpark, a separate checkpoint |
| vision | Qwen3-VL ViT (vision.py) | SigLIP-NaViT 27 layers (1152, patch 14, 980 px) with a window-attention merger after layer 6, then an MLP merger to 2048 |
| image positions | Qwen3-VL M-RoPE | "canvas" M-RoPE (section 4) |
| chat template | Qwen3.6 | the same format: `<think>` on by default, `<think>\n\n</think>\n\n` when off, the same tool call XML |

**Expert selection:** the router is plain Qwen3.5: `mlp.gate.weight` only,
with no bias or correction term, softmax over 256, top 8, then renormalized.
The one new idea is **Lookahead Expert Scheduling (LES)**, but it lives in
their llama.cpp-omni, not in the weights. It predicts the next layer's
experts across layers so it can prefetch them; the README says "computation
still follows the model's original token-level routing". It adds no tensors,
so it cannot change what the model outputs. It is purely a way to schedule
expert loading (section 6, item 4).

## 2. The DSpark drafter (new; not the DeepSeek one)

reclank already has DSpark for DeepSeek V4.1 (models/deepseek41/dspark.py).
This one shares the idea and the heads but has a different body:

| | DeepSeek V4.1 DSpark | MiniCPM DSpark (`Qwen3DSparkModel`) |
|---|---|---|
| body | 3 MoE blocks (128 experts, top 3), 128-position window ring | 5 dense Qwen3 layers: GQA 8 q / 2 kv heads, head 256, q/k norm, SiLU MLP 6144, full rope (rope_dim 256, theta 1e7) |
| attention | window of 128 (ring) | **full attention over the whole context**: its own K/V of every position (5 layers x 2 kv heads x 256 x K+V = 10 KB a token in bfloat16, 2.6 GB at 262K) |
| input | main_proj of the mean of the streams at layers 37-39 | `fc` (10240 -> 2048) of the concatenated hidden states of target layers **3, 11, 19, 27, 35**, then `hidden_norm`; each layer projects that to its context K/V |
| block | 5 (last token + 4 noise) | **15**: the last token + 14 `mask_token_id` (248076) slots, non-causal inside the block |
| embed / head | from the main model | the main model's `embed_tokens` and `lm_head` (none in the drafter) |
| Markov head | embed, bias on the next position | `markov_w1` [V, 256] embeds the previous token; `markov_w2` [V, 256] gives a bias over the vocab. It runs in order over the 15 positions; vLLM gathers w2 only for candidate tokens |
| confidence head | logit per position | `sigmoid(proj(cat(hidden, markov_embed)))`, input 2304 |

Reported acceptance (15 drafts, natural EOS): code 7.4, math 6.7,
vision 5.1-6.1, general chat 3.5.

What carries over from deepseek41/dspark.py:
- the loop: a draft, then a verify length L chosen from the confidences and
  the measured times of a draft and of a verify group of L + 1;
- greedy acceptance, so the tokens are bit-for-bit those of plain decoding;
- the moving time averages.

What is new:
- the context K/V. For every main-model token, prompt and accepted drafts
  alike, the 5 target hidden rows go through `fc`, `hidden_norm`, then
  k/v_proj, k_norm and rope into the drafter's cache. This becomes one more
  record at the end of each prompt group, step and verify group;
- the dense 15-row block forward;
- the gathered Markov bias.

Three points were settled in Phase 0 (section 7):
1. **"Layer i" is the output of layer i**, not its input.
2. **After an image, the drafter's positions are the t row of the canvas
   M-RoPE** (vLLM `_set_positions`: positions[0] for a drafter with plain
   rope on an M-RoPE target).
3. **The confidence head reads the hidden state after the final norm.**

## 3. Weights and the file

- **Download:** the original form, first to the NVMe, now on the HDD at
  `MODELS/MiniCPM-V-4.7-35B-A3B`, with the drafter next to it
  (`-DSpark`). The model is not being pursued further.
- **Converter:** `scripts/convert_minicpmv47_gguf.py`, our own conversion
  following the rule of official or our own files only. It writes a
  `qwen35moe` GGUF with the names and transforms of llama.cpp's qwen35moe:
  - the value heads of the DeltaNet in tiled order;
  - A_log;
  - norm offsets as the qwen35 loader expects (QwenConfig.from_gguf).

  The fused expert split is taken from convert_q8_gguf.py (the same layout as
  Qwen3.8 ORIG). The language part goes in one file. The vision tower and
  mergers go in a second file, the encoder file next to it, as with Qwen3.8's
  `--mmproj` directory; or the encoder reads the safetensors in place, since
  it is only 0.4B parameters.
- **Form:** full quality, as with Qwen3.8 through the weightkeeper:
  - dense matrices and experts in BF12 (`--dense bf12`, plus BF12 experts),
    about 54 GB;
  - any matrix BF12 cannot hold stays bfloat16 (the converter's own report);
  - the router, the gates and the norms in float32.

  No quantized forms unless asked. The DSpark weights load as BF12 too
  (`_bf12_values`, as for DeepSeek).
- **Where the BF12 file goes:** NVMe would have about 60 GB free after the
  checkpoint and the file. The other choice is to keep the 70 GB checkpoint
  on /space and only the BF12 file on NVMe.

## 4. Vision

A new encoder in `reclank/models/qwen/vision_minicpm.py`, run on the GPU at
request time (the `--mmproj-gpu` modes), with a CPU float32 reference for
the checks:

- **Preprocessing:** port the MiniCPMV4_6 processor. It picks the best slice
  grid (max_slice_nums 9), resizes to multiples of 14 (x 2 x 2 merge), and
  normalizes with mean 0.5 and std 0.5. The NaViT packing comes with
  target_sizes, plus the bucketed position-embedding ids. The output must be
  bit-exact against the processor on real images.
- **Tower:** 27 SigLIP layers with LayerNorm, gelu_tanh and biases. After
  layer `insert_layer_id` = 6, `vit_merger`: window self-attention, then a
  2x2 merge (16x mode only). Then post_layernorm and the `merger` MLP (a
  second 2x2 merge) up to 2048.
- **`downsample_mode`:** 16x (the default) or 4x, where 4x skips vit_merger
  and keeps four times as many tokens. It is a request option
  (`mm_processor_kwargs.downsample_mode`, as in vLLM) and a server default.
  It changes both the placeholder count and the encoder.
- **Placeholders:** `<image>` thumbnail `</image>`, then for each slice row
  `<slice>…</slice>` and newline (198). The image_pad / video_pad expansion
  goes in `encoders/media.py`, next to `expand_qwen`.
- **Canvas M-RoPE:** port `get_rope_index` / `get_vision_position_ids`:
  - each image is a canvas: the thumbnail is stretched over the whole slice
    canvas;
  - the markers are pinned to halo or corner positions;
  - after the image, text continues at `origin + max(h, w) + 1`;
  - a run of video frames separated only by markers shares one canvas.

  This replaces Qwen's `rope_positions` for this model only. The text model
  already takes 3-row positions, so only the position maker changes.
- **Video:** frames go through the same tower (get_video_features); same
  limits as Qwen (`--video-frames`, `--video-budget`).

## 5. Phases

Each phase has its own check, on real prompts and images.

0. **Reference.**
   - Set up a separate venv with torch CPU and
     transformers 5.18 (not the run venv).
   - Write `scripts/minicpm_reference.py`, which dumps for real chat text,
     a real photo and a real document image: the logits, the hidden states
     of layers 3/11/19/27/35, the vision features and the 3-row positions.
   - Also run the DSpark reference: one draft block from a real context, to
     settle the two points of section 2.
   - The golden dumps go in a dumps directory (DUMP_DIR).
1. **Text on the CPU.** Converter, then the `qwen35moe` path as it is
   (QwenGGUFProgram) with the new vocab and the template from the model
   directory. Check: logits against the reference (KL, top-1) over real
   chat text, prompt and teacher-forced steps.
2. **Text on the GPU.** gpu35 (QwenGPU) with BF12 dense matrices and BF12
   experts. Hot experts get the free memory: about 10-12 GB of the 49 GB of
   experts on the 3090 (section 6). Cold experts run in the CPU parts (RECLANK_GPU_PARTS,
   moe rows). Check:
   - logits against Phase 1;
   - the moe-rows rule: the same bits whatever hot set;
   - decode and prompt speed.
3. **Server.** In engine.py, detect `model_type == "minicpmv4_7"` from a
   checkpoint directory or a metadata key in the GGUF. The model runs in the
   `qwen` family with MTP off. The contexts (core/session.py RowCache) work
   unchanged for the main cache. Check: chat_quality and dsh-style requests
   on port 8081.
4. **Vision.** Encoder, processor, canvas positions, then the server.
   Check: features and positions bit-exact or within float32 error against
   the Phase 0 dumps; the answers on real images match the reference in
   greedy mode.
5. **DSpark.**
   - Drafter records: the context K/V record after each main-model run, the
     15-row block forward, the gathered Markov bias, confidence, and verify
     groups of up to 16 rows on gpu35.
   - The verify groups need:
     - the GDN commit (`rc_gpu_gdn_commit` takes any n; its per-row log
       must hold 16 rows);
     - the hidden rows of the 5 target layers of each row;
     - a cache.n rule for the drafter's K/V like the one for the main rows.
   - The drafter's K/V joins `rows()` of the RowCache so contexts swap it too.
   - The loop is ported from deepseek41/dspark.py.
   - Check: greedy output identical to plain decode on real prompts;
     acceptance on code, math and chat against the table in section 2;
     tok/s.

   A verify group of 16 rows reaches about 16 x 8 expert picks a layer, and
   the cold ones run on the CPU, so the best L here will be well below 15.
   The confidence-based L picks it, as with DeepSeek.
6. **Later, optional.**
   1. FP8 or quantized drafter K/V for long contexts.
   2. A 4x/16x per-request policy.
   3. Tuning the gathered Markov head.
   4. LES-style prediction, used only to warm hot experts during a prompt.
      It is an evaluation, not a port: reclank runs cold experts on the CPU
      from RAM instead of streaming them over PCIe, so the stall LES hides
      mostly is not there.

## 6. Memory on the 3090 (24.5 GB)

| | GB |
|---|---|
| dense matrices (BF12: DeltaNet, attention, shared experts, lm_head) | ~2.5 |
| main cache: 10 full-attention layers x 2 kv heads x 256, bf16; the DeltaNet states are fixed | ~5.4 at 262K |
| DSpark weights in BF12, with Markov w1/w2 | ~0.6 |
| DSpark K/V | ~2.6 at 262K (~1 GB at 96K) |
| programs, arenas | ~3 |
| vision tower, only while encoding (the lend/release modes) | ~0.8 |
| hot experts | the rest, ~10-12 GB of the 49 |

The main cache is smaller than Qwen3.8's per token.

## 7. Phase 0 results

**Main model** (`scripts/minicpm_reference.py`):
- **How it runs:** transformers 5.19 in its own venv (torch, torchvision and
  transformers, all CPU builds), with the default torchvision backend of the
  image processor. Layer by layer in float32 on 4 CPUs, with the experts read
  as the router picks them.
- **Prompts:** six real ones, plus 128 greedy steps each (or to EOS):

  | prompt | prompt tokens | answer (greedy) |
  |---|---|---|
  | doc (MOE_PLAN.md excerpt) | 1812 | 128 tokens, on topic |
  | think (primes, thinking on) | 21 | 128 tokens of reasoning |
  | code (session.py) | 775 | 128 tokens, explains common_len |
  | photo (corn, 16x) | 99 (70 image rows) | "The image shows three ears of corn, with one partially husked ..." |
  | chart (thumbnail + 9 slices, 16x) | 683 (630 image rows) | "93.31" |
  | photo4x (carrots, 4x) | 1434 (1400 image rows) | "Four fresh, bright orange carrots ..." |

- **Cost:** the prompts took 281 s (69.8 GB read; 10134 of the 10240
  experts used), and the steps 8-12 s each. The process peaked at 36 GB.
- **Dumps:** in a dumps directory (DUMP_DIR) (1.9 GB): logits (top 64, lse,
  argmax; full logits for the first 16 tokens), all 41 hidden states, routes,
  vision features, canvas positions.
- **Canvas positions:** the rope delta after the image is -62 (corn), -625
  (chart), -1371 (carrots 4x): the canvas packs an image into far fewer
  positions than it has tokens.

**DSpark** (`scripts/minicpm_dspark_reference.py`, the drafter as vLLM runs
it, replayed on the greedy chains of the dumps). Mean accepted drafts per
anchor, then tokens per verify step of a greedy run:

| prompt | output of layer L | input of layer L |
|---|---|---|
| doc | 4.85 / 4.10 | 4.36 / 3.97 |
| think | 8.12 / 7.94 | 6.73 / 7.53 |
| code (prose about code) | 3.47 / 3.46 | 3.15 / 3.05 |

- **First draft token:** it matches the model 78-96% of the time.
- **Against the model card:** general 3.5 and math 6.7 tokens per step. The
  port is right.
- **The image answers are too short** to tell the position schemes apart
  (photo4x: t row 3.0, sequence index 2.8 accepted); the t row follows vLLM.
- **Confidence:** with the hidden state after the norm it lines up better
  with real acceptance (Brier 0.087 against 0.147 before the norm, on doc;
  better on every prompt).

**Lineage.** The language model is a fine-tune of Qwen3.6-35B-A3B itself:
- **Correlation with Qwen3.6:** cosine 0.98-0.99999 on every tensor
  compared, and in some experts 24-61% of the bf16 values are identical.
- **Against Qwen3.5** (the EXL3 file, whose router, norms and embeddings are
  unquantized): MiniCPM is about half as far from Qwen3.6 as from Qwen3.5
  (router of layer 0: 0.076 against 0.167; embeddings: 0.067 against
  0.259).
- **The triangle:** d(M,Q35)² = d(M,Q36)² + d(Q36,Q35)² holds within ±0.002
  on the unquantized tensors: one step further on from Qwen3.6.
- **Vocabulary:** rows 0-248047 carry over; rows from 248100 have new
  meanings (the markers of the images).

**DSpark speed, projected.**
- **Inputs:** the acceptance of each anchor (above); the distinct experts of
  a layer for k real consecutive decode tokens (1: 8, 2: 13.1, 4: 21.3,
  8: 33.5, 16: 51.0); and a step of k rows costing F + (1 - F) u(k) / 8 of a
  plain step. F = 0.55 is calibrated on the Qwen3.8 BF12 verify of 4 rows,
  1.75 times a step. The draft costs 0.10-0.15 of a step.
- **With the best fixed L:**
  - doc 1.9-2.0x (L 6);
  - think 2.6-2.7x (L 9);
  - prose about code 1.6x (L 5);
  - image description 1.6x (L 4).
- **Range:** F 0.45 to 0.65 gives 1.4x to 3.1x across the prompts.
