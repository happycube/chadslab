# Nemotron 3.5 Super VL 120B-A12B (NVFP4) in reclank

The goal: serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 as NVIDIA ships it (no
weight changed; layouts only) on the 3090 + Xeon box: the dense part on the GPU, the routed
experts split hot (GPU) and cold (the CPU's parts), its own MTP layer for drafts, images
through its RADIO encoder, and later the lanes.

Status: a plan. The facts below come from the checkpoint's config.json, hf_quant_config.json and
safetensors headers, and from a read of reclank. The speeds are estimates.

## 1. The model

The language model is Bolt's architecture (NEMOTRON_BOLT_PLAN.md section 1) with uniform layers,
more experts and an MTP layer:

| | Bolt 74B-A9B (done) | Super 120B-A12B |
|---|---|---|
| blocks | 88: 40 Mamba-2, 40 latent MoE, 8 attention (7, 16, 25, 36, 47, 58, 69, 78) | the same blocks at the same places |
| hidden, vocab | 4096, 131072 | the same |
| Mamba-2 | 128 heads of 64, 8 groups, state 128, conv 4 | the same |
| attention | 32 query heads of 128, 2 KV heads, no RoPE | the same |
| routed experts | 240-496 a layer (puzzle), top 9, inner 1536-2560 | 512 a layer, top 22, inner 2688 |
| latent MoE | fc1_latent 4096 -> 1024, experts relu^2 in the latent, fc2_latent 1024 -> 4096, shared expert 4096 -> 5376 -> 4096, router sigmoid + correction bias, x 5.0 | the same |
| drafter | none (the separate LiLiCorr DFlash model) | one MTP block: eh_proj(cat(enorm(embed), hnorm(h))) with h the target's hidden after norm_f, an attention layer (BF16, its own KV), a latent MoE layer (512 NVFP4 experts), final_layernorm, the shared FP8 head (no norm_f); later drafts take its own output as h |
| context | 262144 | 1048576 |
| vision | C-RADIO v4-H (not done) | C-RADIO ViT-H/16 (32 blocks of 1280, 3 CLS, 7 registers, CPE), vision_final_layernorm, pixel shuffle 0.5, mlp1 projector (5120 -> 20480 -> 4096); one tile an image at its own resolution (1024 to 13312 patches: 256 to 3328 tokens), `<img>` + `<image>` x N + `</img>` |
| template | Qwen kind (`<|im_start|>`, `<think>`, `<tool_call><function=...><parameter=...>`) | the same kind, `enable_thinking`, `<image>`; ultra_v3_reasoning_parser.py |

The repo has the reference code: modeling_nemotron_h_omni.py and modeling_radio.py (transformers
5.8).

## 2. The weights (as stored)

| Part | Format | Size |
|---|---|---|
| routed experts, 40 x 512, up 2688 x 1024 and down 1024 x 2688 | NVFP4 W4A4: E2M1 pairs, E4M3 scale for each 16, F32 weight_scale_2 and a static input_scale for each matrix | 65.0 GB, 3.17 MB an expert |
| MTP routed experts, 512 | the same | in the 65.0 (about 1.6 GB) |
| shared experts, 40 | NVFP4 W4A4 | 1.02 GB |
| Mamba in_proj 4096 -> 18560, out_proj 8192 -> 4096 | FP8 E4M3 W8A8, a per-tensor weight scale and a static input_scale | 4.38 GB |
| attention q, k, v, o | FP8 W8A8; the KV cache FP8 (static k_scale, v_scale only for the MTP attention; the backbone's at 1.0, as vLLM) | 0.29 GB |
| lm_head | FP8 W8A8 | 0.54 GB |
| latent fc1/fc2, routers (+ f32 correction bias), conv1d, norms, A_log, D, dt_bias | BF16 | 0.85 GB |
| MTP block (eh_proj, attention, norms, its latent and router) | BF16 (its experts NVFP4) | 0.16 GB |
| embedding | BF16 | 1.07 GB (stays on the host) |
| vision tower + projector | BF16 | 1.69 GB |
| **total** | | **75.0 GB** |

A token reads 22 x 40 = 880 experts: 2.79 GB, against Bolt's 437 MB. The dense part read each
token is about 7.2 GB.

## 3. What reclank has

| Need | From | State |
|---|---|---|
| Mamba-2 (step, chunked prompt), the gated group norm, NoPE GQA, the latent MoE's structure, relu^2, the head | Bolt: models/nemotron (model.py the NumPy reference, kq.py the CPU program, gpu.py NemotronGPU, NM_* records) | done for Bolt's formats (IQ2_XXS, Q8_0, its NVFP4 rows) |
| verify groups with two sets of Mamba states, spec decoding | Bolt: spec.py, NemotronGPUSpec | done (for LiLiCorr) |
| NVFP4 experts in groups of 16 rows (KQ_NVX), the A16 FMA contract on CPU and GPU (kq_nvx_rows_f, k_kqh_nvx_ct), A4 prompt rows (kq_nvx_rows_a4; int8 mma tiles qmoe_tc), the rows form of the combine (the same bits whatever the hot set) | FP4-Local (FP4LOCAL_PLAN.md) | done, for SwiGLU experts |
| FP8 products: MXFP8 (k_mxf8_gemv, k_mxf8_tc, KQ_MXFP8 on the CPU); DeepSeek's block-scaled FP8 | FP4-Local, DeepSeek V4.1 | done; no per-tensor-scale form |
| an E4M3 KV cache (rc_kv8_val) | FP4-Local, DeepSeek | done (per-row or per-block scales; Super's are static per layer) |
| the weightkeeper (experts on huge pages, a copy on each node for the parts) | keep_fp4l.py (KQ_NVX), nemotron/keep.py | done for those formats |
| the dynamic hot cache (make_room, shrink and grow), lanes | DeepSeek V4.1 (deepseek41/gpu.py make_room, lanes.py) | done for DeepSeek |

## 4. The new work

1. **A loader** (formats/nemotron_super_st.py): the index, the roles of the tensors (backbone,
   mtp, vision_model, mlp1), the NVFP4 pairs and scales, the FP8 tensors and their scales, the
   BF16 ones, views with no copy; the config's llm_config.
2. **The experts on KQ_NVX with relu^2.** Super's experts are up and down only (no gate), relu^2
   between them, in the latent (1024 / 2688: 64 and 168 groups of 16, inside KQH_CT_NH). The
   KQ_NVX layout and products stay; the act becomes relu^2 on one input (FP4-Local's is SwiGLU on
   two), on the CPU (moe_part.c's A16 and A4 paths) and on the GPU (k_kqh_act, the qmoe A4 act).
   The A4 requantization of the act uses the down matrix's input_scale as FP4-Local's does.
3. **FP8 with a per-tensor scale.** The MXFP8 kernels with every E8M0 at 127 (1.0) and the
   tensor's weight scale on the output (a layout and an fp32 multiply, no weight change), or a
   KQ_F8T type of the same kernels with one scale. Activations A16 by default (more precise than
   the W8A8 contract), with a switch for the static FP8 input (x / input_scale to E4M3) to check
   against the contract.
4. **The KV cache in FP8**: scale 1.0 for the backbone's 8 layers (they store none: vLLM's
   choice), the stored k_scale and v_scale for the MTP attention; BF16 keys and values as the
   reference's check.
5. **The model on these formats:** NemotronSuper (CPU) and NemotronSuperGPU on Bolt's programs,
   with the hot and cold split of the experts (DSHot-style dynamic hot cache with make_room), the
   routing profile for the first hot set.
6. **The MTP drafter:** h = eh_proj(cat(enorm(embed(next)), hnorm(h_last))), the attention block
   (its own KV cache), the MoE block (512 experts: hot on the GPU from the main model's cache or
   cold on the CPU), the final norm and the shared head; drafts of 1-3 tokens; Bolt's verify
   groups (the Mamba states in two sets) for the main model.
7. **The vision encoder:** each image one tile at its own resolution (the patch grid from
   Python's round-half-even, a budget of 1024-13312 patches, even sides; torch's bicubic resize
   with antialias), C-RADIO ViT-H/16 (CPE: the 128 x 128 position table resized bilinearly and
   cropped to the grid; 3 CLS + 7 registers dropped), vision_final_layernorm, pixel shuffle 0.5
   (ps v2), mlp1; the rows into the prompt between `<img>` and `</img>` (an `<image N>` tag before
   each when there are several). The GPU encoder lent the memory of the hot experts while it runs
   (as Qwen3.8's --mmproj-gpu lend).
8. **The server family:** detect model_type nemotron_h_omni, the chat template, the reasoning
   and tool parsers (the Qwen XML tools as Bolt's), images through the encoder.

## 5. Memory

The GPU (24 GB):

| Item | GB |
|---|---|
| CUDA context | 0.4 |
| Mamba FP8 4.38, attention 0.29, lm_head 0.54, shared experts 1.02, latent 0.67, routers 0.17, the MTP block 0.16 | 7.2 |
| Mamba state (f32 as the config) | 0.17 a sequence |
| KV cache, FP8: 8 layers x 2 heads x 128 x (K, V) = 4 KB a token | 0.5 at 128K, 1.1 at 256K, 4.3 at 1M |
| work buffers, programs | ~1-1.5 |
| **hot experts** | **~12-13 at 128K** (about 20% of the 65 GB) |

The image encoder (1.7 GB of weights and its buffers) borrows from the hot experts while it
runs. Host: a copy of the experts on each node for the parts, 2 x ~66 GB; the dense part and the
embedding a few GB. One weightkeeper at a time: DeepSeek's stops first.

## 6. Speed (estimates, to measure)

- Decode, a token: the GPU reads ~7.2 GB of the dense part plus its hot experts (~10-14 ms at
  the 3090's effective rate); the CPU reads the cold experts. With ~20% of the experts hot the
  hit share is maybe 45-60% (the routing profile decides it), so 1.1-1.5 GB a token cold, at
  about 195 GB/s on both nodes (DeepSeek's measure) ~6-8 ms plus the parts' barriers. The latent
  keeps the hand-off at 4 KB a token a layer. As DeepSeek's, the GPU and the CPU take turns
  within a token: ~22-27 ms a step, **~37-45 tok/s plain**; MTP x1.3-1.6 if its acceptance is as
  Bolt's drafter's.
- Prompt: the hot experts on the int8 mma tiles of A4 (FP4-Local's), the dense FP8 on the tensor
  cores; the cold experts of a group either on the CPU in A4 (mixed groups) or streamed over the
  link (52 GB at 8 GB/s, ~6.5 s a group). **~300-500 tok/s at 8K**, to measure.
- For scale: DeepSeek V4.1 Flash here decodes 23-26 tok/s with DSpark; Bolt with every expert hot
  78 tok/s plain.

## 7. The contract and its deviations

From vLLM main's NemotronH, nemotron_h_mtp, nano_nemotron_vl and ModelOpt paths, the README and
the checkpoint's own code (S0):

- **The activations follow the GPU class, not the kind of row.** On Blackwell every row of the
  NVFP4 layers runs W4A4 (one input scale a layer on the FlashInfer paths: the largest
  input_scale of the 512 experts, up and down apart) and the FP8 layers static per-tensor A8. On
  Ampere (the 3090) all of them run through Marlin: NVFP4 as W4A16, FP8 as W8A16, the input
  scales unused. (A prompt-A4 / decode-A16 split is not upstream vLLM's.)
- dt has no floor (the config's time_step_limit, the fast paths, vLLM); transformers 5.19's slow
  path clamps at time_step_min.
- The residual stream is BF16 in serving (residual_in_fp32 False); the router runs in float32
  (sigmoid, top 22 by score + correction bias, the weights from the scores, normalized, x 5.0).
- Marlin drops E4M3 block scales under 2^-6 (16% of the routed up blocks: whole dead rows, at most
  ~1e-10 of a matrix's energy).

Where reclank differs (named, as FP4LOCAL_PLAN.md section 7):

- **D1.** The experts (main and MTP) A16 on every row: the 3090's own contract, with every block
  scale as stored. A4 prompt rows in the Blackwell form only as an option.
- **D2.** Float32 outside the contract's BF16 points (the residual stream, the norms, the router,
  the attention, the SSM); the BF16 points of the linear outputs, the expert rows, the routed sum,
  fc2_latent, the shared expert and the MoE output kept.
- **D3.** The FP8 linears A16 (Marlin's form); a switch for static A8 (x / input_scale to E4M3).
- **D4.** The shared expert A16; A4 with its own input_scale as the option.
- **D5.** The KV cache FP8 with scale 1.0 for the backbone and the stored scales for MTP; BF16 as
  the check.
- **D6.** The SSM state float32 (the config) in place of the recipes' f16 / bf16 with stochastic
  rounding; dt unclamped.
- **D7.** The x 5.0 in float32 on the weights (transformers) rather than in BF16 after the sum
  (vLLM); the sums in each kernel's order (FP4LOCAL_PLAN.md D5).

## 8. Phases

| Phase | Work | Check |
|---|---|---|
| S0 | the contract from the serving source; the loader (4.1); every tensor found, shapes and quant types against the config and hf_quant_config.json | the inventory: sizes as section 2 |
| S1 | the float32 reference: Bolt's model.py on Super's tensors dequantized exactly (NVFP4, FP8 with their scales), the MTP block, against modeling_nemotron_h_omni.py on real prompts (CPU, the experts loaded by a hook) | the same top-1 and small KL to the transformers forward; perplexity of real text |
| S2 | the CPU kernels: KQ_NVX experts with relu^2 (A16, A4), FP8 per-tensor (KQ_MXFP8 form), the FP8 KV cache with static scales | each kernel against NumPy on trace rows; A4 bit-equal (integer sums), A16 within the FP32 order; a token alone and in a group the same bits |
| S3 | the CPU engine on these (Bolt's kq.py program): steps and prompts | logits against S1; decode-row KL against the float32 reference |
| S4 | the weightkeeper: the KQ_NVX experts on huge pages, a copy on each node, the rest | the fill time; --check against a direct read |
| S5 | the GPU program: the dense part, the hot and cold split with the rows form of the combine, the dynamic hot cache and make_room, the routing profile and the first hot set | the logits against S3; the same tokens whatever the hot set; tokens a second |
| S6 | prompts on the GPU: A4 tiles with relu^2, FP8 on the tensor cores, the mixed groups | prompt-row KL against the reference; tok/s at 2K and 8K |
| S7 | MTP: the drafter and Bolt's verify groups; greedy and sampled | the greedy tokens of plain decoding; the acceptance; tok/s |
| S8 | vision: RADIO, the tiles, pixel shuffle, mlp1; the encoder lent the GPU's memory | the image rows against modeling_radio.py on real images; an answer about a chart |
| S9 | the server family (template, parsers, images), start_server.sh M=super; then the lanes (DeepSeek's design) | check_server; a dsh session; two requests at once |

Order: S0-S3 make a reference and the CPU path whole; S4-S6 the speed; S7-S9 the rest.

## 9. Open questions

- The attention kernel of SM86 with an FP8 KV cache in vLLM, and whether Hopper quantizes Q.
- The MTP's input hidden through vLLM's spec-decode runner (traced to NemotronHModel's output).
- The hit share of the hot experts at ~20% of them (S5's profile): it sets the decode rate more
  than anything else.
- The cold experts of a prompt: the CPU's A4 or the link (FP4-Local measured both for Qwen3.8).
- 22 experts a token on the CPU: the parts' small phases (DeepSeek: ~4.5 ms a step of barriers)
  may weigh more with more pairs a layer.
- The MTP block's experts: their own hot set, the main model's cache, or the CPU.
- vLLM shares the image budget across the images of a prompt against max_model_len; one image
  or a long context gives the transformers sizes.

## 10. Status

- **S0.** The contract (section 7). scripts/check_nemotron_super_inventory.py: 169505 tensors,
  75.0 GB, every shape and stored type as the config and hf_quant_config.json say (the MTP block's
  1024 NVFP4 matrices are not in hf_quant_config.json: typed by their tensors).
- **S1.** models/nemotron/model.py takes Super: the uniform layers (layers_block_type), FP8 with a
  tensor scale (fp8), the FP8 head, the MTP block (NemotronCPU.mtp: section 1's order). A real
  paragraph of 40 tokens: the main model's top-1 22 of 39; the MTP block teacher-forced over it (its
  keys accumulating as in a prefill) drafts the main model's own pick 33 of 38 times (87%). Without
  keep_dense the shards' maps close after each block (65 GB of experts read).
- **S2 (part).** The NVFP4 experts in KQ_NV4 rows run in GP_NM_MOE; its form for a few tokens takes
  their rows 16 at a time by the rows kernels of nmx2.c (kq_rows_x2_nv4v, kq_rows_x2_nv4;
  RECLANK_NM_NV4_ROWS=0 the rows one at a time): one decode token's 22 experts of a layer on a core
  12.0 -> 6.1 ms (5.8 -> 11.3 GB/s), 1.6e-5 from the float reference either way. FP8 runs as KQ_MXF8
  rows (E8M0 127) with the tensor scale after (GP_MUL), through the generic loop of kq_dot_x2 (slow:
  no int16-x form for E4M3 yet).
- **S3 (part).** kq.py on Super's formats: scripts/check_nemotron_super_cpu.py, the program of the
  first 2 and 8 blocks against the reference: max relative error 2.5e-5 / 2.3e-5, cosine 1.000000
  (Mamba and attention FP8, the experts NVFP4, the shared experts NVFP4). The whole model needs the
  experts packed (65 GB): with the weightkeeper (S4).
- **S4 (part).** keep.py: the Super layout (model "nemotron-super": 80 stacks of KQ_NV4 rows, a copy
  on each node, 72.8 + 64.5 GB; the FP8 tensors as stored; the NVFP4 shared experts in KQ_NV4 rows),
  checked by a dry run of layout; the fill runs when no other weightkeeper holds the memory.
- **S7 (the CPU part).** kq.py compile_nemotron_mtp: the MTP block as one program on t rows (eh_proj
  as its two column halves, its attention on K.mtp / V.mtp, its experts' own KQ_NV4 stacks);
  NemotronKQ.mtp, mtp_cache. scripts/check_nemotron_super_mtp.py: the engine on the 38 positions of
  the real paragraph in one program against the reference's 38 drafts: out 2.2e-5, cosine 1.000000,
  the same draft 38 of 38.
- **S8 (the reference).** models/nemotron/radio.py: the image path in NumPy (one tile at its own
  resolution: the checkpoint's sizes, torch's antialiased bicubic; C-RADIO ViT-H/16 with the CPE,
  vision_final_layernorm, pixel shuffle, mlp1 as RMSNorm + relu^2) and the prompt's image span.
  scripts/check_nemotron_radio.py against the checkpoint's torch code on two real images: the image
  rows 1.5e-6 / 7.2e-6 relative, the token counts and input_ids the processor's.
- The GPU program (NemotronGPU) refuses Super until S5 (its hot and group MoE read IQ2_XXS rows).
- **S5 (decode) and the server.** GP_NM_HOT_NV4 (backends/cuda nm.cu: k_nm_hot_up_nv4,
  k_nm_hot_down_nv4, a warp a KQ_NV4 row on float x; k_nm_hot_sum): the hot experts of a step or a
  verify group; their cold pairs on the CPU (NM_MOE); a prompt group of more than 16 rows all its pairs
  on the CPU until S6. FP8 runs as KQ_MXF8 through KQ_LINEAR, its scale by a GP_MUL (the head's on the
  host). scripts/check_nemotron_super_gpu.py against the CPU engine (all from the weightkeeper): the
  argmax 17 of 17, KL 3e-3 (the GPU's float x against the CPU's int16 x). The CPU team needs
  RECLANK_GPU_CPU_THREADS (task_threads read the main thread's one CPU: a step 537 -> 70 ms with 44).
- **S7 on the GPU.** NemotronGPU.mtp (the MTP block's programs, its prompt parts without the head),
  spec.NemotronMTPSpec (a round: one MTP run over the positions the block has not read with the
  target's rows, the drafts after it, Bolt's verify group with the replay of the taken rows),
  engine.GpuRunner.round. scripts/check_nemotron_super_spec.py: 2.65 / 1.97 of 3 drafts taken (code /
  prose); the verify group's dense products (int8 x) are not a step's (float x), so greedy tokens part
  at a near tie.
- **S8 on the GPU.** radio.RadioGPU: RADIO as a program of the encoder records (the bfloat16 matrices
  on the tensor cores), the projector a second one (GP_RELU2): an image of 5376 patches in 2.0 s, its
  rows 1.35e-3 of the checkpoint's float32 code. The engine takes images (the template's <image>, the
  span <img> + <image> x n + </img>, the rows in place of the embeddings), keeps VISION_GB and MTP_GB
  out of the hot experts.
- **S9 (first form).** start_server.sh M=super (python -m server, the family nemotron): a text answer
  and an image answer (a poster read right); 20 tok/s decode with MTP (3.4-3.65 tokens a round), a
  prompt of 1033 tokens (with an image) 75 tok/s (S6 left).
- **Decode: the CPU's experts and the hot set.** NM_MOE takes the weightkeeper's copies on node 1
  (operands 14, 15: NemotronKQ.node1): the team's threads of node 1 read them, those of node 0 the
  stacks on node 0 (the same bytes, the same bits). Its form for a few tokens with KQ_NV4 experts runs by
  expert (the pairs of an expert in one task, its rows read once for every token of a verify group;
  the int16 act made by the up tasks; the same kernels and sums: bit-equal). The real inputs of the
  last layer, the CPU alone on 44 threads: a step's 19 cold pairs 0.97 -> 0.56 ms, a group of 4's 67
  pairs (42 experts) 2.63 -> 1.43 ms. scripts/nemotron_profile.py --gpu: the routing profile of real
  text by the GPU's prompt groups (24K tokens in 280 s; expert_profile.npz beside the checkpoint): a
  hot set of 15% of the experts covers 37% of the reads (the first experts: about their share). With
  all three, on the 3090 (17% hot): a step 55.1 -> 35.7 ms (18.1 -> 28.0 tok/s), a group of 4
  151 -> 74 ms.
- **The -0 codes of NVFP4.** nm_e2m1x4 (the E2M1 decode of k_nm_t8v, the shared experts of a verify
  group; k_nm_t8d) negated a byte as (v ^ 0xff) + 1, which carried into the next byte for -0 (code 8,
  frequent in Super's checkpoint): those products 3-5% off (2.6e-5 now). scripts/check_nemotron_super_spec.py
  checks each MTP token against plain steps on the same prefix (the argmax or within two bfloat16
  steps of the head's top logit) in place of the plain text's tokens: it found this. Code and prose,
  96 tokens each: 96 and 94 the step's argmax, the other 2 within one bfloat16 step.
- **S6 (prompts on the GPU).** NM_GROUP_MOE takes KQ_NV4 experts (operand 21; k_nm_gtile_nv4: k_nm_gtile_tc's
  tiles, the rows decoded by nm_e2m1x4 with the scale of each 16, an m16n8k16 a half and plane of the
  int16 x), every expert read once a group from its slot or the host memory the GPU maps (the
  weightkeeper's stacks on the GPU's node and the MTP block's, cudaHostRegister: 1.8-5 s at start).
  NM_PLAN (one block) counts the pairs of each expert after the router and gives the CPU the cold experts
  of at most c pairs (NM_MOE in its task, as a step's cold pairs, while NM_GROUP_MOE runs the others),
  c the count where a pair's cost on the CPU (RECLANK_NM_PLAN_CPU, 12.6 us) and a cold expert's on the
  GPU (its copy over the link, RECLANK_NM_PLAN_GPU, 0.3 ms) balance; the two sums added after (D5's
  order). The head reads the last real row of a group (DS_GATHER on slot "last"), so a group may take
  fewer tokens than its rows; NemotronGPU.parts picks a prompt's runs by the measured time of each size
  (GROUP_COST: 943 tokens one group of 1024, not 3 x 256 + 2 x 64 + ...). Real text, 17% hot: a group
  of 1024 3.10 s (330 tok/s; the CPU's experts alone 85 tok/s, the GPU's alone 201), 943 tokens 3.11 s
  (89 before), the last logits KL 9e-5 to the CPU's experts. RECLANK_NM_GROUPS=4,16,64,256,1024,2048:
  379 tok/s at 4K, but the program of 2048 rows takes 1.8 GB from the hot experts (a step +2.8%, a
  verify group +4.5%): not the default.
- **The MTP block's programs** each made eh_proj's two column halves again (new host arrays: a new
  device copy of 134 MB each, 9 programs); made once now (all the programs 3.18 -> 2.10 GB): an image
  request after a long prompt ran out of memory without it. The server: a prompt of 7532 tokens 262
  tok/s (75 before S6), an image request of 1036 tokens 274 tok/s, decode with MTP 32.8-34.6 tok/s
  (20 before; 3.2-3.3 tokens a round).
- **The kernels (decode and prompts).** On the 3090 and the Xeon, real text:
  - The CPU's experts: kq_nv4z_rt prefetches the next call's rows (the kernel waited on memory: about
    2.5 GB/s a thread at 44 threads), and decodes the E4M3 scales as half floats (the bits of E4M3 << 7,
    exact) in place of a gather; bit-equal. NM_MOE on 13 cold pairs 0.40 -> 0.30 ms, a group of 4's
    1.42 -> 1.07 ms. x as two int8 planes against unsigned values (kq_e2m1u, the correction from x
    alone) gave the same bits but no time (1-2% slower): the loop is bound by its loads, not its ALU.
  - The shared expert runs between the CPU's signal and the wait for it (em.moe's side), off the
    critical path; ROUTER_SIG in one pass (k_router_sig_r: each expert's rank; 24 -> ~3 us); KQ_MXF8
    with 2 rows a warp (in_proj 114 -> 95 us, out_proj 67 -> 45 us) and the FP8 tensor scale folded
    in (operand 10: GP_MUL's product, 112 launches a step less); all bit-equal.
  - The bf16 matrices (the latent projections, the routers, the MTP block's eh_proj) in BF12 rows
    (KQ_BF12; 0.02% of their values, all below 1e-5 of an RMS of 0.023, zeroed): fc1 26 -> 18 us,
    fc2 25 -> 13 us, the gate 14 -> 11 us; RECLANK_NM_BF12=0 keeps float32.
  - NM_HOT_NV4 in k_nv4_gemv2's form (k_nm_hot_nv4b: 16 bytes a lane, R rows of a warp in flight, x
    staged once a block): a step 30.3 -> 27.4 ms (the hot experts had reached the critical path).
  - A prompt group's up tiles quantize the act themselves (k_nm_gtile_nv4's epilogue: k_nm_gqa's
    bytes), so no float act (236 KB a row, ~1 GB for 4096 rows).
- **The pool of hot experts (dynamic caching).** NmPool: the hot experts in segments of 32 slots (any
  expert of any layer in any slot); each layer's tables give each expert's device address (its slot's,
  or its rows in the mapped host memory), so NM_HOT_NV4 (operands 13, 14), NM_PLAN and NM_GROUP_MOE read
  them and no program changes when the set does. The set follows the text: GP_COUNT counts the routers'
  choices on the GPU; every 16 tokens (and after a prompt group) the scores decay and the best experts
  not held replace the worst held ones (DMA from the registered stacks on the copy worker, held from
  the run after they land): 42.8 -> 46.5% of a step's pairs held on code, 51.1 -> 54.3% on the docs
  (the first set from the routing profile, 19-20% of the experts). Prompt groups of 2048 and 4096 rows
  (BIG_GROUPS): their programs made for a long prompt, the pool giving back the segments their memory
  needs, then freed and the segments taken back and refilled (relax); 0.4-0.8 s to make. A prompt of
  8K tokens: groups of 1024 264 tok/s, 2048 330, 4096 377.
- With all of it: a step 35.7 -> 26.8 ms (28.0 -> 37.3 tok/s), a group of 4 74 -> 70 ms;
  check_nemotron_super_spec.py PASS (96 / 96 and 95 / 96 the step's argmax), MTP 44.3 / 46.6 tok/s.
  The server (18% hot, 11.5 GB): decode with MTP 42-51 tok/s (33-35 before), a prompt of 4445 tokens
  281 tok/s (a group of 4096 made in 0.8 s), an image request right at 46 tok/s.
- **The small kernels and the CPU task's fixed cost** (Nsight Systems on 20 steps: the GPU's own time
  ~12 ms a step on the way between the CPU's tasks, MXF8 6.7 of it near the bandwidth):
  - ROUTER_SIG by a bitonic sort in shared memory (score down, index up: the same order and bits; the
    rank of each expert took 20 us, one SM's 512 x 512 comparisons); its record takes GP_COUNT's sums
    (operand 9) and, in a step, GP_NM_SPLIT's split (10-12), and a step's router with its split and the
    SIGNAL after it run as one kernel (k_nm_route_signal, as DeepSeek's k_ds_route_signal: the latent
    rows' copy to the host first; fc1 before the router); AWAIT adds the CPU's rows to the GPU's
    (operand 6: the GP_ADD after it); a group's split stays NM_SPLIT (one block for its rows was 1-2 ms
    slower on a group of 4). RECLANK_NM_FUSE=0: the records of their own. All bit-equal; the fusions
    themselves are within the noise of a step (the router's sort and the next item are the gain).
  - KQ_BF12 of one token on a matrix of at most 4096 x 4096 by k_kq_linear_bf12 (the CUDA cores), not
    the tensor cores: fc1 18.2 -> 13.6 us, the gate 11.2 -> 9.3 (4 tokens stay on the tensor cores:
    18.6 against 25.5).
  - NM_MOE's few-token form without the barrier between the down tasks and the sums (the task that ends
    a block of 32 rows' last expert adds its rows, in the slots' order: bit-equal) and, when it is the
    last record of its program (a CPU task), without the one after the down tasks (the region's end is
    one). The rest of the fixed cost of a task: the region's start 5 us and end 6 us; the body (290-335
    us for ~11 cold experts) is the kernels at ~3 GB/s a thread (a prefetch 4 calls ahead into L2 gave
    nothing more).
  - A step 26.8 -> 25.4 ms (39 tok/s plain); check_nemotron_super_spec.py PASS (96 / 96 both), MTP
    45-51 tok/s.
- **Lanes** (`--lanes 2`; models/nemotron/lanes.py, the design of docs/LANES_PLAN.md as DeepSeek's): lane B
  is a second sequence (NemotronGPU.LANE_ATTRS: its cache as long as the model's, its Mamba states, its
  step and small-group programs with scratches of the C side of their own; the prompt groups' programs
  shared, gprogs: a prompt runs alone), swapped in by NmLane.view; run_both runs a step of each at once
  (LaneRunner, run_pair: lane B's on a thread and a stream of its own, the CPU tasks of both in turn on
  the model thread); the pool's tables change only between the runs. Two real prompts, 48 greedy plain
  steps each: the tokens and the last logits of each lane bit-equal to it alone; a pair of steps 35 ms
  (57 tok/s for both) against 24.6 ms a step alone (1.41x). The engine (coop): a lane alone decodes
  with MTP rounds (a DeviceOp "yield" after each: the scheduler between them, a new kind that counts as
  decoding for the other lane's pause); while the other lane decodes, plain steps handed over as runs and
  paired (their rows go to the MTP block, which reads them before the next round); a prompt in parts of
  4096 tokens with a pause after each. A program that finds no room makes the pool give back 0.5 GB
  (NemotronGPU._load; relax takes it back while 0.9 GB stays free). The server with two lanes (15% hot):
  two requests at once both started in 1-2 s and decoded at 24-26 tok/s each (a pair 36 ms, ~55 tok/s
  for both; one lane alone with MTP ~48); a 12.8K prompt on one lane at 275 tok/s while the other
  decoded at ~10 tok/s between its parts (RECLANK_LANE_PROMPT_SHARE); one lane alone ~40-45 tok/s (the
  smaller hot set: RECLANK_NM_LANE_GB of the GPU for lane B).
- **Prefill: the experts staged, and their tiles.** Nsight on a prompt of 8192 tokens in groups of 4096 (the
  old path, ~377 tok/s): the expert tiles (k_nm_gtile_nv4) 11 s of ~20, the dense FP8 as bf16 GEMMs 3.5,
  attention 1.4, the Mamba scan 1.2, the shared experts 1.2. Staged in the GPU's memory the tiles took as long
  (down 104 ms, up 46 ms a layer of a group): bound by their compute (~13 TOPS), not the link.
  - k_nm_gtile_nv4c: 64 x 64 tiles (8 warps of 32 x 16), the columns 64 a stage, the next stage's codes, scales
    and x in registers while the tensor cores take this one, one barrier a stage, each lane's fragments decoded
    from the codes; the same order of sums: bit-equal. Down 104 -> 35 ms, up 46 -> 25 ms a layer. Its core
    (nm_nv4c_mma) also runs NM_GEMM8 of KQ_NV4 rows on more than 8 tokens (the shared experts; k_nm_gemm8_nv4c,
    11.7 -> 8.3 ms, bit-equal to k_nm_gemm8_tc).
  - Staging (groups of 2048 rows and more with the pool; RECLANK_NM_STAGE, RECLANK_NM_STAGE_MIN): each MoE
    layer's cold experts copied by GP_FETCH (the copy worker's pull kernel, 111 ms a layer) into one of two
    stage buffers while the layers before run, the tiles reading them there (operand 22: k_nm_gtile_nv4c reads
    an expert's rows once for each 64 of its pairs, so not across the link); the tables and the copies' ranges
    of every layer written before each run from the pool's set (NemotronGPU.stage_fill); the buffers' room from
    the pool, given back by relax. Every pair on the GPU there: the CPU's share stalled the runner at the
    copies' boundaries (358 tok/s against 516). A group of 1024 moves the same copies for a quarter of the
    tokens: it keeps the mapped memory and the CPU's share (258 against 227 staged). Staged against mapped,
    every pair on the GPU: bit-equal; against the old split with the CPU's share: KL 6e-3 to 1.6e-2 at three
    lengths, the same top three.
  - The MTP block's pass over a prompt in parts of up to 2048 rows (each part reads the block's experts across
    the link once: 256-row parts gave 9387 tokens 3.6 s more, now 2.0).
  - A prompt of 8192 tokens 377 -> 514 tok/s, 16000 tokens 533 (the bench, groups chosen by GROUP_COST); in the
    server 9387 tokens 313 -> 394 tok/s and 13842 tokens 411 (with the MTP pass, the cut at the answer, the
    big programs made for each prompt: 0.4-0.8 s). A group of 4096 now: the GPU's kernels ~6.5 s of 7.9
    (dense bf16 2.1, the tiles 2.4, the shared experts 0.66, the scan 0.63, attention 0.38 at 4K), the copies
    4.4 s beside them.
- **Prefill, the second round** (a group of 4096 rows of real text 7.90 -> ~5.4 s):
  - The MXF8 matrices of a group (x rounded to bfloat16 first) in gg_gemm_bf16's one-plane form (x1: the low
    plane was zeros, its products a second pass of the tensor cores) and their x converted once, not for each
    chunk of rows: the dense part 1.92 -> 1.03 s a group; bit-equal.
  - k_nm_gtile_nv4c's grid with the row tiles of an expert fastest (its pairs' x from the L2 once): down 34 ->
    24 ms a layer; bit-equal.
  - NV4T, the stage buffers in the tiles of the GPU: each 64 x 64 tile the codes of each lane's fragments in one
    16-byte piece (and its rows' scales), made by k_nm_tile_pull as it copies every expert of the layer (the
    cold ones from the host's rows, the hot ones from their slots; GP_FETCH mode 1); k_nm_gtile_nv4t reads its
    fragments with one load a stage (no weights in shared memory), x by cp.async three stages deep: the
    experts 60 -> 35 ms a layer; bit-equal. The copies stay at the link's ~12 GB/s (114 ms a layer).
    Not at the weightkeeper: the copy permutes for free (the link bounds it), and the CPU and the steps keep
    the rows.
  - GP_NM_ATTN2 of a group (64 rows and more) on the tensor cores (k_nm_fattn, FlashAttention-2's form: 8
    tokens a block, a warp each token's 16 query heads; the keys and values of 64 positions shared): q and P
    in bfloat16, as the reference runtime's flash attention. On the last attention layer of a group of 4096
    real tokens against float64: median relative error 2.59e-3, 99th 6.62e-3 (the reference's flash form,
    emulated: 2.56e-3, 6.62e-3; k_nm_attn2 1.65e-3, ~the bfloat16 rounding of its output that o_proj
    makes); RECLANK_NM_FA=0 keeps k_nm_attn2.
  - Groups of 8192 rows gave no more (659 tok/s against 680 at 16K). GROUP_COST 2048 3.75 s, 4096 4.67.
  - A prompt of 8192 tokens 514 -> 772 tok/s (the bench); the copies (~4.5 s a group of 4096) are now about
    the compute's time: the ceiling ~900 tok/s.
