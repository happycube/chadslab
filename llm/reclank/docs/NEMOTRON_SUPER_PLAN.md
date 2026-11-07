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
- **Prompts on Pascal (dp4a, no tensor cores; the GTX 1070 of the W-2133 box).** Super ran there with the defaults
  (an 8 GB card: no hot experts, groups of at most 256 rows) but its prompt groups put every pair on the CPU and
  the MXFP8 rows in k_mxf8_gemv's passes of 8 tokens: a prompt of 1K 22 tok/s. Now:
  - k_nm_gtile_nv4p (NM_GROUP_MOE of KQ_NV4 without the tensor cores): a block for an expert and 32 rows of up
    (16 of down: Pascal's 48 KB a block), the raw rows in shared memory once (each expert crosses PCIe once for
    the group), each 128 columns decoded to int8 (nm_e2m1x4) with the scale of each 16, the pairs' x into shared
    memory with the next 128 columns' in registers, a thread's rows x 4 pairs by dp4a; k_nm_gtile_nv4's sums in its
    order: on the RTX 3060 (RECLANK_NM_NV4P=1, the mapped path) the logits bit for bit those of k_nm_gtile_nv4
    (the prompt's row and 8 steps). The mapped tables for KQ_NV4 on any GPU of dp4a; the stage stays the tensor
    cores' (NemotronGPU.staged).
  - k_mxf8_gemm_f (KQ_LINEAR of KQ_MXF8 on more than 8 tokens without the tensor cores): 64 rows x 64 tokens a
    block in float32, 4 x 4 a thread, the E4M3 values and the bfloat16 x of each 32 columns in shared memory, the
    sum of each 32 times the row's scale as k_mxf8_tc (scripts/check_mxf8.py: within 0.5 bfloat16 step of float64
    at 64 and 600 tokens). The dense rows of a prompt of 1K 12.4 -> 3.6 s (~3 TFLOPS).
  - NM_PLAN's cost of a CPU pair from the CPU (plan_cpu: 12600 ns at the 3090 box's 22 threads with VNNI, times 22
    over the team, 3 times without VNNI; the cut is the ratio to PLAN_GPU): the W-2133's 11 threads 75600. A prompt
    of 256 on the RTX 3060 50 -> 80 tok/s (its groups of 1024 up are staged: unchanged, 1K 170, 4K 391).
  - The GTX 1070 alone: a prompt of 1K 22 -> 56 tok/s, 4K 55 (groups of 256: every expert across the link for each,
    the GPU's ~38 GB a group at ~13 GB/s, the CPU's quarter of the pairs beside); decode 8.8 (unchanged). The logits
    against the CPU's experts: KL 4e-5 to 1.4e-2 a row, argmax 9/9. check_nemotron_super_spec PASS on both cards.
- **Prompts on the RTX 3060, then with the GTX 1070 beside it.** Where a prompt's time went (nsys of the timed
  prompt alone): a group of 1024 is link-bound (5.41 s of 5.46 copying ~59 GB of cold experts at ~11 GB/s, the
  compute 2.8 s); a group of 4096 is compute-bound (9.3 s: the NV4T tiles 3.06, the MXFP8 rows on bfloat16 tensor
  cores 2.9, the Mamba scan 1.49, the shared expert 1.28).
  - The scan (k_nm_scanm, nm.cu): k_nm_scan's warps (one a value of hdim, 8192) each read their group's B and C for
    every token, ~34 GB of L1/L2 traffic a layer at 4096 rows (Nsight Compute: L1 91% busy). Now a block of 4 warps
    of one head, 8 values of hdim a warp: each chunk of 16 tokens' B, C, x, dt and dA in shared memory once for the
    block, a token's B and C read once a warp for its 8 values, their 8 dot products through one butterfly
    (nm_multi_bfly: each level keeps half of a lane's values, so each sum adds k_nm_scan's pairs in its order), 2
    tokens a step, y out by rows. The state's contractions written out as nvcc made k_nm_scan's: y and the state
    bit for bit k_nm_scan's at 2 to 4096 rows (a standalone check), and the logits too (Super 4K on the 3060, 1K on
    the 1070). A layer of 4096 rows 37 -> 6.8 ms on the 3060, 46 -> 11.9 on the 1070; of 1024 9.7 -> 1.8. The 3060's
    4K prompt 401 -> 449 tok/s; 1K the same (link-bound). RECLANK_NM_SCAN=1: k_nm_scan.
  - The 1070's share of a group's experts (XSPLIT, NemotronGPU.xgroup / xgsplit / _xg_fill; the expert device's group
    job, runtime/expert_dev.cu rc_gpu_ex_nm_gpost; GP_NM_REMOTE_POST's group form, operands 9-11). In each MoE layer of
    a staged group of 1024 rows, NM_SPLIT gives the 1070 the pairs of its share: the CPU's task posts them with the
    group's rows (the SIGNAL's pinned buffers) and the 1070 runs them as it runs a prompt alone (NM_GROUP_MOE on
    k_nm_gtile_nv4p in 8 chunks of the experts and windows of pairs, each expert from its slot or through the map
    from the keeper's pages, registered portable by the 3060's library); the join adds its sums, the AWAIT adds them
    to the 3060's. The 3060's stage copies nothing of the 1070's experts (zero-byte ranges, stage_fill). The share,
    from a cost model by the routing profile: an expert costs the 3060 its bytes over its link, the 1070 its pairs
    (~10 us a pair beside the 3060's copies; dp4a, the 1070's bound) and, unless it holds it, its bytes over its link
    (~5 GB/s beside the 3060's); the 1070 takes the experts of the lowest ratio of its cost to the 3060's while its
    time stays under the 3060's. The experts the 1070 holds for decode are the most routed, so it gives most of them
    back: about 200 experts a layer and 30% of the pairs go to it.
  - A prompt of 1K on both cards: 172 -> 207 tok/s (the 3060's copies 5.41 -> ~2.8 s, the 1070's jobs ~3.7 s). The
    groups of 2048 and 4096 do not gain (334 -> 333, 451 -> 434: their copies overlap the Mamba layers, and the
    1070's pairs cost it 10 times the 3060's), so XSPLIT_MAX is 1024. Logits against the unsplit run: KL 4e-5 to
    1.7e-2 a row (each token's pairs summed in two parts), argmax 9/9; check_nemotron_super_spec PASS on both cards.
    The bound: both links' copies (~11 + ~6.5 GB/s together) and the 1070's tiles (~1.4 TOPS of its 26).
  - Why the 1070's tiles (k_nm_gtile_nv4p) are slow, one layer of 1024 rows alone (a standalone harness: the library's
    NM_GROUP_MOE on synthetic KQ_NV4 experts; nsys, and the SASS: Nsight Compute has no Pascal). From the GPU's memory
    165 ms (1.5 TOPS of products, 3 of dp4a with x's two planes: 12% of the 1070's ~26), mapped 225: the kernel more
    than the link. x's two planes cost 1.5 times (one plane 111 ms). Of down's 100 ms: ~19 the blocks' fixed part
    (the raw rows' load, the barriers), ~20-30 the decode of the rows for each 128 columns and each batch of 32 pairs
    and x's staging, the rest the products at ~22% of dp4a's rate. Not the cause: the int to float conversions (2%), a
    branch around the loads of x (none), the grid's order (2-3%).
    - Down was bound by shared memory (a thread 1 row x 4 pairs: 18 reads of 128 bytes for 64 dp4a). Now a warp all 16
      rows of 8 pairs, a thread 4 rows x 1 pair (12 reads), and the grid's x the blocks of rows, y the experts: the
      same bits; down 100 -> 80 ms, the layer 165 -> 143 (a share of 196 experts 64 -> 55, mapped 99 -> 85). The 1070
      alone, a prompt of 1K: 74 -> 78 tok/s (the logits bit for bit); the split: 207 -> 209 (its share mostly mapped:
      up is the link's there, 110 ms mapped against 63).
    - The occupancy: 2 blocks of 4 warps an SM (37-39 KB of shared memory a block, most of it the raw rows, kept for
      each batch of pairs). With the raw rows read from global memory instead (a measure), down 80 -> 63 ms. Left: the
      columns outer and the batches inner (each 128 columns' rows read and decoded once for all the expert's pairs, no
      copy of the rows), and for the split the 1070's cold share by DMA (~12 GB/s, its own stage) rather than mapped
      (~5-7 beside the 3060's copies).
    - The columns outer (k_nm_gtile_nv4pc): each 128 columns' slice of the block's rows loaded into registers a step
      ahead (16-byte pieces) and decoded once for up to NB batches of 32 pairs (up 2, down 1), no copy of the rows in
      shared memory: the occupancy. The same bits. A layer of 1024 rows from the 1070's memory 143 -> 99.6 ms (half its
      experts there, half mapped: 181 -> 159); mapped, much slower (a step's slices too small a read across the link:
      down 108 -> 810 ms), so it takes only the experts in the GPU's memory: operand 22 = 3, operand 7 a slot table
      (the pool's; an expert device's group job, its held ones), k_nm_gtile_nv4p the others (a slot out of date only
      changes which kernel runs an expert). Today little of the 1070's prompt work is in its memory (alone: no pool
      beside its 7 GB of dense rows; the split: its held experts are the most routed, mostly given back): the 1070
      alone 1K 78 -> 77 tok/s, the split 209 -> 208, the logits bit for bit. Its share staged into its memory by DMA
      would put all of it there.
  - The 1070's share by DMA (expert_dev.cu ex_gstage_job; RECLANK_EX_STAGE=0: mapped as before). Its group job copies
    the cold experts of its share into two stage buffers of 48 experts in its memory (2 x 149 MB), on a copy stream
    from the keeper's pages (cudaMemcpyAsync, ~7.8 GB/s beside the 3060's copies; the map's reads ~5-6.5), each chunk's
    records after its copies and each buffer's copies after its last chunk ran; every expert of the job is then in its
    memory (operand 22 = 3 and a slot table of zeros: k_nm_gtile_nv4pc). The local ids follow the job's groups
    (NM_GROUP_MOE quantizes x and sorts the pairs on the record of x0 0, sums on the one of x0 + nx = E). No room for
    the stage: mapped. A prompt of 1K on both cards 209 -> 254 tok/s (252 with these ids); the logits against the
    mapped split KL <= 1.3e-2 (each token's pairs summed in another order), argmax 9/9.
  - The next layer's first chunks ahead (RECLANK_EX_PREFETCH=0: none). _xg_fill sends each layer's cold share to the
    device (devices.py gmask, rc_gpu_ex_nm_gmask), the most routed first; a job then copies the next layer's first
    two chunks once its own last chunks free the buffers, during the 3060's layers between. The next job checks those
    experts against its own (a share or a held set changed since: copied again), runs them first, then its held
    experts (their ~20 ms beside the copies of its next two chunks), then the rest. Only experts with pairs are
    copied otherwise (a share's every expert: 11 a layer more, 1.2 GB at 1K: 252 -> 244). The order mattered: with
    the held experts first the copy engine sat idle while they ran (both buffers full), 244 -> 253 only; with them
    after the chunks copied ahead 252 -> 261, and the 1070 then takes more of the experts: XS_L7 5e9 -> 6e9, XS_C7
    10e-6 -> 8.5e-6 (a sweep: 6e9 and 7-8.5e-6 273-274, 7e9 and 10e-6 267, 4.5e9 and 11e-6 255). Without the copies
    ahead at those rates 236.
  - Both cards now: a prompt of 1K 172 (not split) -> 209 (mapped) -> 274; 2K 334 -> 401-403 (split now: XSPLIT_MAX
    2048); 4K 452 -> 458 split, so not (the 3060's pairs more than its link there). Stage buffers of 32 / 48 / 64
    experts: 1K 274 / 274 / 242, 2K 391 / 403 / 402. Logits against no split, 1K: KL mean 8.1e-3, max 1.9e-2, argmax
    9/9; 2K: the prompt's row 1e-6, the steps after it diverge at an exact tie in the unsplit run (step 3: two tokens
    at 65.612, a bfloat16 step 0.31), as any change of order does there. The prefetch's job against the same job
    with every chunk copied, in one run: max |d| 2.4e-7 at |out| 1-3. check_nemotron_super_spec PASS on both cards.
    Where 1K goes now (nsys, span 3.18 s): the 3060's copies 3.13 s (its link the bound, ~11 GB/s), the 1070's jobs
    39.5 ms a layer (55.8 before the copies ahead; its copies 18.7 GB at 7.8 GB/s, its copy engine 75% busy), the
    3060's wait for them 8.7 ms a layer (25.9). docs/NEMOTRON_SPLIT_LESSONS.md has the lessons.
- **Decode on both cards: the next MoE layer's cold experts predicted and copied ahead to the 1070.** A step (~71 ms
  after a prompt of 1K): each MoE layer's ~12 cold experts a token on the CPU (~1 ms at ~37 GB/s) while the GPUs wait,
  then the 3060's Mamba and attention layers (~0.7 ms) while the CPU waits.
  - What predicts them (the NumPy reference on a real prompt, 64 tokens; "cold": neither GPU holds it, ~13.5 of a
    token's 22): the next MoE layer's router on the residual entering the MoE layer before picks its top 4 cold
    experts 92% right, 6 88%, 8 83%; on the residual after that layer 95% / 92% / 88%; two MoE layers ahead 84% of 4.
    The previous token's experts 38%. The 1070's held experts refreshed from the prompt's routing (refresh_experts after
    it): its share of the pairs 18-20% -> 20-21%, the steps 14.2 -> 14.1 tok/s.
  - Built (RECLANK_NM_PREDICT, default 3): in a step's MoE layer j (kq.compile_nemotron_step, NmEmit.predict) the 3060
    norms the residual entering j by the next MoE layer i's norm, runs i's router (BF12, ~10 us) and copies its logits
    to pinned memory before j's signal; j's CPU task (GP_NM_REMOTE_POST operands 12-18) picks i's experts neither GPU
    holds by the router's choice (sigmoid + bias) and hands the top 3 to the 1070 (rc_gpu_ex_nm_prefetch: DMA on a copy
    stream into one of two buffers of its decode stage, an event an expert, stg the host's marks); i's CPU task first
    drops the experts whose copies have not ended (rc_gpu_ex_nm_settle) and posts the pairs of the others to the 1070,
    whose job reads their rows in the stage. The 3060's TAKE is off for steps then.
  - What it took: the first version slowed the steps (14.2 -> 11.5): the 1070's job uploads (its record, pairs and x,
    ~20 KB, DMA) queued behind the copies on its one copy engine (their start after issue: p90 28 us -> ~2 ms); they
    are read by a kernel now (ex_h2d, k_ex_pull16), 14.0. The host's DRAM is shared: the CPU's ~37 GB/s and the 3060's
    TAKE pulls (~12) beside it already near its ~53; with the TAKE off the 1070's copies pay: 2 experts 14.6, 3 14.7
    (78% of the copies read, 14% late), 4 14.2 (59% late: the link's ~7.6 GB/s a MoE layer's ~1.8 ms). Copied after
    the CPU's pairs (the DRAM free, RECLANK_NM_PREDICT_AT=1) the gap is too short: 13.9.
  - Results: decode after a prompt of 1K 14.2 -> 14.6-14.7 tok/s; the server's plain chats 14.12 -> 14.47 (every turn
    faster); with MTP 3 drafts the same (15.97 -> 15.88). A verify group can predict too (RECLANK_NM_PREDICT_G, the
    largest of its rows' choices; off by default): with the TAKE 15.71, without 14.82 (4) and 15.32 (6), so the 3060's
    pulls of a group's experts are worth more. Logits against no prediction: KL <= 1e-2, argmax 9/9 (as the TAKE's own
    change); which pairs the 1070 runs depends on the copies' timing, so the steps are not bit-for-bit from run to
    run. check_nemotron_super_spec PASS on both cards. The bound now: the DRAM, a step's ~1.7 GB of cold experts at
    ~53 GB/s ~32 ms against the CPU's window (~40 ms) and the dense layers (~31) in series; the next lever would be the
    3060 copying the predicted experts during its dense layers (its link and the DRAM idle then).
  - The 3060's share (RECLANK_NM_PF3, default 1): of the predicted cold experts, the next one after the 1070's is
    copied across the 3060's own link into a stage of two buffers (rc_gpu_pf3_issue: DMA on its own stream, the
    flag of each copy set by a one-thread kernel after it, k_pf3_flag) by the runner after the CPU's task's flag out
    (rc_gpu_set_cpu_after, the CPU library's rc_er_after: the CPU's pairs done, the DRAM free, the issue off the GPU's
    wait); the next layer's NM_STAGED (opcode 380, k_nm_staged: one block, both buffers' words read once into shared
    memory) gives the 3060 the pairs of those whose copies ended, as NM_TAKE's tables do. What it took: a host
    function's flags came too late (its thread beside the CPU's spinning team: 15 of 7566 copies used); the stage's
    words read by every thread across PCIe (65 us a layer, < 10 after); the issue before the task's flag (the GPU
    waited for its calls). Decode after a prompt of 1K: none 14.5 tok/s, 1 expert 14.8-15.0 (94% of them used), 2
    14.6 (45%: the gap of ~0.7 ms fits one 3.1 MB copy at ~7.4 GB/s), 3 13.6; the 1070's 2 / 3 / 4 beside one of the
    3060's 14.6 / 14.9 / 14.1. The server's plain chats 14.43 -> 14.74 (14.12 with neither); its MTP chats in the
    noise of their texts (15.6-16.1 in six runs, either way). KL against no prediction <= 1e-2, argmax 9/9;
    check_nemotron_super_spec PASS on both cards.
- A Quadro P6000 alone (GP102, sm_61, 24 GB, 30 SMs, 432 GB/s; in a PCIe 3.0 x8 slot here: the host link 6.2 GB/s).
  Super holds 17.0 GB of hot experts (26%): prompts of 1K / 2K / 4K 97 / 126 / 150 tok/s, decode 13.4-13.7; Bolt holds
  every expert (17.1 GB): 1K / 4K 291 / 296, decode 33.3-34.5, its drafter 51.6. A step of Super (nsys, 75 ms): the
  dense layers 30.3 ms (k_mxf8_gemv 18.8: ~270 GB/s), the hot pairs 10.9 (k_nm_hot_nv4b, ~140 GB/s), the TAKE's pulls
  21.0 and the wait for the CPU 6.3.
  - k_nm_hot_nv4b on Pascal (runner.cu: no tensor cores; the tensor-core GPUs keep their shapes): blocks of 16 rows, the
    registers capped by __launch_bounds__ (up: 2 rows a warp, 8 blocks an SM; down: 1 row a warp, 2 sets, 4 blocks, at
    8 it spills), the first rows' loads before x's staging and each next set's before the sums of the set at hand. A
    layer's 12 hot pairs of 22 (a bench of the kernel alone): up 104 -> 69 us, down 97 -> 87 (loads alone 66 / 53,
    sums alone 45 / 40: the old shapes ran them one after the other). The same sums in the same order: the logits
    bit-equal. Decode after 1K 13.4-13.5 -> 13.8-13.9 tok/s.
  - Bolt's steps with every expert in the GPU's memory: the CPU's share (cpu_set) is empty, so no signal, task or wait
    (gpu.py _moe: the pairs on the zero-copy tables, an expert that leaves the memory later through the map); k_signal
    waited ~40 us a layer for the empty task. Decode 34.1-34.3 -> 35.2-35.3, the drafter 51.5 -> 52.2; logits bit-equal.
  - Super's MTP on the P6000: a round of 3 drafts ~215 ms, its verify (4 rows) ~192: the TAKE's pulls 82 ms (4 experts a
    layer at the x8 link's ~0.5 ms each), the CPU's cold pairs behind them (30 ms of wait), the 4-row dense products
    (k_mxf8_gemv_s) 35 ms against a step's 19, the hot pairs 31 (each pair reads its expert's rows). The check's two
    prompts: 3 drafts 16.8 / 13.2 tok/s, 2 drafts 17.1 / 16.4, 1 15.5 / 14.8 (plain 13.0), so the server takes 2 drafts
    on a GPU without tensor cores (engine.py; --mtp still sets it): chats 15.11 (3) -> 15.55 (2), plain 13.81.
  - Bolt's prompt on the P6000: its dp4a products (k_nm_gemm8k, one x plane) were 2.16 s of a 1K prompt's 3.28 (the
    NVFP4 Mamba in 1.00, out 0.45, the IQ2 shared expert 0.46, Q8_0 0.24) at 6-8 TOPS of ~51. A step's 128 columns
    took ~10 us against ~1-2 of sums: its words were loaded after the barrier and waited for. k_nm_gemm8d (nm.cu) loads
    the next step's words into registers while the step at hand is summed and decodes them into shared memory at the
    top of the next; tiles of 4 x 4 a thread and 4 parts a step, or 8 x 4 and 1 part for 2048-8192 rows; the int sums
    to float by the magic number (exact); the same float steps, so the same bits (RECLANK_NM_G8D=0: the old kernel).
    The bench of the kernel alone: Mamba in 25.1 -> 16.0 ms, out 11.2 -> 7.0, the shared expert 5.5 / 5.9 -> 4.5 / 4.4,
    Q8_0 q / o 7.2 -> 4.0. Bolt's prompts of 1K / 4K 292 / 296 -> 366 / 371 tok/s, logits bit-equal. What did not pay:
    the int sums' I2F alone (-5%; I2F is quarter rate on Pascal, but the loads were the wait), larger tiles without the
    prefetch (8 x 8 spills; 8 x 4 +0-30%). Super's prompt takes its two-plane form (X2): unchanged.
- Super on the P6000, again (slight differences allowed for the prompts):
  - Its prompt groups take int8 x in every int8 tile on a GPU without tensor cores (gpu.py x16_for: the routed experts'
    tiles too, not only the shared experts' as on the tensor cores; the steps and verify groups keep int16 x): 1K / 4K
    97 / 150 -> 105 / 170 tok/s.
  - Its dense MXFP8 rows of a prompt group on dp4a (mxf8.cu k_mxf8_g8, RECLANK_GPU_MXF8_G8=0: k_mxf8_gemm_f in float32):
    each 32 of a row's E4M3 values and of a token's x to int8 by their largest |value| in the loader, the next step's
    words in registers (as k_nm_gemm8d), a part's 32 terms one int sum times the two scales; 0.8% relative RMS against
    the float32 kernel on random rows. At 4K tokens: the Mamba in 133 -> 77 ms, out 59 -> 28.5, attention q / o 29.7 ->
    14.7. Prompts of 1K / 4K 105 / 170 -> 116 / 201 tok/s.
  - Both against the float32, int16-x prompt (a prompt of the docs, then 64 steps teacher-forced on its next tokens):
    1K the prompt's row KL 1.25e-2, the steps' KL mean 9.2e-3, argmax 61 of 64, NLL 2.032 -> 2.028; 4K the prompt's row
    8.7e-7, the steps' 5.1e-3, argmax 63 of 64, NLL 1.186 -> 1.169. check_nemotron_super_spec PASS.
  - The steps: NM_TAKE off when the pool holds TAKE_HOT_MAX (0.15) of the experts or more (gpu.py take; a verify group
    keeps it): the CPU's few cold pairs ran slower beside the pulls' reads of the host's memory (a layer's window 0.75 ->
    0.87 ms with one pull). k_mxf8_gemv R 1 for Pascal's steps (R 2 for its groups: k_mxf8_gemv_s's verify 156 -> 168
    ms at R 1). Decode after 1K 13.8-13.9 -> 15.4-15.7 tok/s; the server's plain chats 13.81 -> 15.05, MTP (2 drafts)
    15.51 (verify ~155 ms, still the CPU's cold pairs).
  - What did not pay: NM_PLAN's link cost a cold expert at the x8 link's 510 us (more of them the CPU's: 1K 116 -> 103)
    or 150 us (109), none to the CPU (92); the experts in the GPU's memory (k_nm_gtile_nv4pc) on a side stream beside the
    mapped ones (k_nm_gtile_nv4p, bound by the link): 189 ms of 1073 overlapped, the mapped kernel's blocks hold every
    SM's slots while they wait on the link (116 -> 117); groups of 8192 (RECLANK_NM_BIG_GROUPS) not taken (the room).
    Left: the link (7.1 of a 4K prompt's 20.4 s; an x16 slot ~8 GB/s), the cold experts staged by DMA beside the tiles
    (as the 1070's expert jobs: the copy engine, not the SMs' slots) to hide the 2.9 s of the GPU's own experts.
- Super's prompt groups staged on Pascal (gpu.py STAGE_P, stage_p; RECLANK_NM_STAGE_P=0: mapped as before): each MoE
  layer's experts in STAGE_P_C (2) chunks, a chunk's cold experts copied by DMA (GP_FETCH mode 2: the copy worker's
  cudaMemcpyAsync from the registered host pages, the copy engine) into its stage buffer at their places in row format
  while the chunk before is summed; a held expert's table its slot (its range 0 bytes); NM_GROUP_MOE with operand 22
  = 3 and operand 7 all_res (every expert in the GPU's memory): k_nm_gtile_nv4pc for all of them. No change to the
  runtime: the mapped path's chunk and window operand (23), stage_fill's tables and ranges, the first two copies at
  the program's start. The P6000 (x8 link): prompts of 1K / 2K / 4K / 8K 116 / 160 / 201 / 168 -> 118 / 230 / 270 /
  264 tok/s, the logits bit-equal to the mapped path with every pair on the GPU. Chunks: 1 113 / 222 / 274 (3 GB of
  buffers from the pool), 2 118 / 230 / 270 (1.5 GB), 4 120 / 197 / 249, 8 112 / 175 / 238: each chunk is a boundary
  of the program (the host waits there for the copy worker), and the mapped kernel's few-pair experts on the CPU
  (NM_PLAN) are off while staged. Decode after a staged prompt as before (15-16 tok/s); check_nemotron_super_spec
  PASS.
- Bolt's drafter on the P6000 (slight differences allowed in the verify group): plain 35.2 tok/s, the drafter
  51.9 -> 80.2 (a round: the draft 17.9 -> 9.8 ms, the verify group 76.5 -> 56.3). A round before (nsys): the verify
  group 78 ms = its dense rows 37.7 (k_nm_t8d: 2.5 times a step's), the routed experts 22.6 (k_nm_hg_*), the head 5.3
  (k_kq_linear_i8's general form of 5 to 8 tokens, 184 registers; a step's head 2.0); the draft 18 ms = the head 4.8,
  the bfloat16 attention rows 3.9 (k_mt_gemv_bf16), the NVFP4 MLPs 3.7, the lattice's ENC_LINEAR 2.3 (k_enc_gemm on 16
  blocks of 30 SMs). k_nm_t8d was bound by its x reads (every warp of a block read the same x through L1: 1 row a warp
  105 ms a verify group, 2 76, 4 91) and by I2F (a quarter of the rate; two a token, a row and a block of 32).
  - k_nm_t8s (nm.cu): k_nm_t8d's products and sums with x of all the tokens staged in shared memory once a block (the
    halves of a lane's 32 bytes a tile apart: no bank conflicts), in tiles of a multiple of 32 blocks when it does not
    fit in 48 KB, and the int sums to float by the magic number (exact; with two planes fma(h, 256, l), rounded once as
    the int's conversion): the same bits. The bench of the kernel alone (8 tokens, one plane): Mamba in 305 -> 201 us,
    out (two tiles) 178 -> 142, the IQ2 shared expert 118 / 140 -> 73 / 98, Q8_0 4096 x 4096 139 -> 77, the head 3.9 ->
    1.6 ms. Either change alone gave little (shared memory 303 -> 302, the magic number 303 -> 335). Two planes in tiles
    were slower (Mamba in 350 -> 433): k_nm_t8d keeps them (RECLANK_NM_T8S=0: k_nm_t8d for all).
  - The head of 2 to 8 rows on Pascal (gpu.py emit_head: the verify group's and the draft's) by NM_QX + NM_GEMM8 (k_nm_t8s)
    in place of KQ_LINEAR: 4.8 -> 1.6 ms each (RECLANK_NM_HEAD8=0: KQ_LINEAR).
  - Bolt's verify group on int8 x on Pascal (spec.py _vprog: x16_for, as its prompt groups; the tensor cores keep two
    planes): k_nm_t8s in one tile, and k_nm_hg_* with the planes at compile time (X2: one plane does no products of a
    low one) and the magic number (the same bits for either). Against the plain steps on the same tokens (3 chats, 768
    rows): KL mean 1.85e-3 (two planes 1.27e-3: the verify group's kernels were never a step's), the argmax the same 754
    of 768 (759), the step's top-2 margin at the others 0.007-0.27. The drafter's text leaves the plain one at token 99
    of the code prompt (two planes: the same 256 tokens), and 4.63 of 7 drafts are taken a round there (4.12).
    RECLANK_NM_VERIFY_X16=1 keeps two planes: 58.1 tok/s, the same tokens as plain (the verify group 72.8 ms: two
    planes take k_nm_t8d in tiles).
  - The drafter on Pascal (lilicorr.py LILI_PASCAL; only the drafts change, the verify group decides the tokens): its
    products on one plane, its bfloat16 matrices as Q8_0 rows (the int8 products in place of k_mt_gemv_bf16), the
    lattice's ENC_LINEAR on float32 rows (k_enc_sgemm's tiles fill the SMs): the draft 15.2 -> 9.8 ms, 74.0 -> 80.2
    tok/s (RECLANK_LILI_PASCAL=0: the old).
  - The server runs Bolt's drafter with --drafter DIR (engine.py; spec.py LiLiCorrRounds, the GpuRunner interface of
    Super's MTP: after each run of a prompt or a plain step the captured layers' rows of that run's program go into the
    drafter's context; the target's programs capture them from the load; 1.5 GB for the drafter, RECLANK_NM_LILI_GB;
    one lane; --mtp 0: off). The verify group now runs NemotronGPU.after_run as Super's does. The drafter's context is
    not in the saved contexts: after another conversation's point it drafts on that one's positions until new ones
    replace them (fewer drafts taken, the tokens the verify group's). Chats (chat_turns.py, greedy, 256 tokens a
    turn): plain 35.03 tok/s, the drafter 50.23 (code 68.6 / 73.3, the story 34.7 / 39.9, the hash map 49.1 / 58.9;
    2.4-5.3 tokens a round); its answers leave the plain ones (the verify group's int8 x).
  - Bolt's images: its checkpoint has its own RADIO tower and mlp1 (the shapes of Super's, other weights) and no final
    layer norm (config vision_final_layernorm False; modeling_nemotron_h_omni.py takes the field, else the LLM's MTP
    layers). radio.py follows the same rule; the engine takes images when mlp1 is in the checkpoint (it had looked for
    the final layer norm's tensor). --mmproj-gpu off: the encoder's programs in C on the CPU (RadioGPU gpu=False), none
    of the GPU's memory (VISION_GB not reserved; Bolt keeps 93% of its experts hot at 262144 positions). Against the
    NumPy float32 path on the checkpoint's accuracy chart: rel 3.3e-6, every row's cosine 1.000000. The W-2133 (6
    cores): 253 rows 5.3 s, 495 rows 11.5 s, 1012 rows (the image budget 1024) 29.3 s. Through the server Bolt reads
    the chart's title, its four models and its 13 benchmarks, and the shapes, colors and text of a drawn image.
  - scripts/serve_bolt.sh starts Bolt's server so (the drafter, the encoder on the CPU, 262144 positions): Bolt's
    keeper started when it is not running and the keepers of other models stopped (their memory), a reclank server
    on the port stopped first (the restart); -d detaches and waits until it serves. A machine's CUDA settings in
    ~/.config/reclank/env.
