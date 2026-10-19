# Nemotron 3.5 Bolt 74B-A9B 24GB in reclank (small machines first)

The goal: serve nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB as NVIDIA ships
it (2-bit experts, no weight changed) on small machines: an AVX2 desktop
with a Pascal card (GTX 1070 8 GB, P102-100 10 GB, GTX 1080 Ti 11 GB), and
on the CPU alone (AVX2, AVX/scalar). The 3090 + Xeon box is the reference
and the place where the small machines are emulated (RECLANK_CPU=avx2,
RECLANK_GPU_CC=61, a GPU memory budget).

## 1. The model

NemotronH Puzzle (modeling_nemotron_h.py of the repo, transformers 5.8),
88 blocks, hidden 4096, vocab 131072, each block `x + mixer(rmsnorm(x))`:

| Part | Blocks | What |
|---|---|---|
| Mamba-2 | 40 | in_proj 4096 -> 18560 (z 8192, xBC 10240, dt 128); causal conv 4 + bias + silu over xBC; 128 heads of 64, 8 groups of B/C, state 128; dt = max(softplus(dt + dt_bias), 1e-4), A = -exp(A_log); y = C.h + D x; gated RMSNorm in 8 groups of 1024: rmsnorm(y * silu(z)) * w; out_proj 8192 -> 4096 |
| Attention | 8 (7, 16, 25, 36, 47, 58, 69, 78) | 32 query heads of 128, 2 KV heads, no RoPE, no qk-norm, scale 1/sqrt(128) |
| Latent MoE | 40 | router 4096 -> E (240-496 by layer), sigmoid + correction bias, top 9 (1 group), normalized, x 5.0; fc1_latent 4096 -> 1024; experts in the latent: down(relu(up(x))^2), inner 1536-2560 by layer; fc2_latent 1024 -> 4096; shared expert in the full width: relu^2, 4096 -> 5376 -> 4096 |
| Head | | norm_f, untied lm_head; embedding bf16 |

Per-layer expert counts and widths come from `block_configs` (13824 routed
experts in all). No MTP layer; NVIDIA's drafter is the separate LiLiCorr
model (DFlash, 7 tokens). The vision tower is C-RADIO v4-H (ViT-H/16, 32
blocks of 1280) with an MLP projector (mlp1). The chat template is the
Qwen kind (`<|im_start|>`, `<think>`, `<tool_call><function=...>
<parameter=...>`); EOS 2 and 11; sampling defaults temperature 1.0, top_p
0.95.

## 2. The weights (as stored, none converted)

| Tensor | Format | Size |
|---|---|---|
| routed experts up/down (13824 x 2) | IQ2_XXS (ggml blocks of 256, 66 bytes, 2.06 bits) | 17.05 GB, 1.23 MB an expert, 437 MB read a token |
| shared experts (40) | IQ2_XXS | 0.45 GB |
| Mamba in_proj/out_proj (40) | NVFP4 (e2m1 pairs, E4M3 scale per 16, fp32 tensor scale) | 2.47 GB |
| attention q/k/v/o, latent fc1/fc2, lm_head | Q8_0 (ggml blocks of 32) | 0.30 + 0.36 + 0.57 GB |
| router, conv1d, norms, A_log, D, dt_bias | bf16 (the correction bias f32) | 0.1 GB |
| embedding | bf16 | 1.07 GB (stays on the host) |
| vision tower + projector | bf16 | ~1.6 GB |

The rule (the user's): no data conversion besides an optimized layout. The
IQ2_XXS and Q8_0 blocks keep their bits (a layout may interleave rows or
split scales from payload), NVFP4 keeps its nibbles and scales, bf16 stays
bf16 (or is widened to f32 for the small tensors). Activations may be
quantized at run time (int8 for dp4a / VNNI products), as llama.cpp does.

The IQ2_XXS tensors run in the rows of KQ_IQ2R (type 67, reclank/formats/iq2.py to_rows; the
weightkeeper's index says iq2="iq2r"): the same bytes, a row's 8-byte parts first (block b, part j
at 64 b + 8 j), then its fp16 d values, zeros to 8 bytes (+0-1% for the rows of 9, 10 or 21
blocks). As stored a block is 66 bytes, so its parts were 2-byte aligned (loads of bytes and
halves); now a part is one 8-byte load and a warp's 32 parts are 256 contiguous bytes.

## 3. Memory

Per token: KV 8 KB (8 layers x 2 heads x 128 x K,V x fp16): 0.27 GB at
32K, 1.07 GB at 128K, 2.15 GB at 256K. Mamba state 170 MB a sequence (f32
as the config; 16-bit would halve it), any length.

What must be on a GPU that holds the model's dense part: Mamba 2.47 GB,
latent fc1/fc2 + router + shared experts 0.92 GB, attention 0.30 GB,
lm_head 0.57 GB = 4.27 GB, + Mamba state 0.17 + work buffers ~0.3 + the CUDA
context ~0.4 = **5.1 GB + KV**. The rest of the card holds hot experts.

| Card | Context | Room for experts | Hot experts |
|---|---|---|---|
| 6 GB | 32K | ~0.6 GB | ~4% (barely runs) |
| GTX 1070, 8 GB | 32K | ~2.6 GB | ~15% |
| P102-100, 10 GB | 32K | ~4.6 GB | ~27% |
| GTX 1080 Ti, 11 GB | 32K | ~5.6 GB | ~33% |
| 1080 Ti | 128K | ~4.8 GB | ~28% |
| 3090, 24 GB | 32-64K | all 17.05 GB | 100% |

Which share of the expert reads the hot set covers is a property of the
routing; P6 measures it on real prompts and turns this table into tokens a
second for each card. The minimum that runs: ~5.4 GB of GPU memory at 32K,
every expert on the CPU.

## 4. Speed (estimates, to measure)

A token reads 4.27 GB of the dense part on the GPU (1080 Ti 484 GB/s: ~15 ms
at 60% of peak; 1070 256 GB/s: ~28 ms) and (1 - hot share) x 437 MB of
experts on the CPU. The latent makes the hand-off small: 4 KB a token a
layer each way. IQ2_XXS on AVX2 is bound by the grid lookups, not by DRAM
(to measure: P1). CPU only (AVX2, dual-channel DDR4 ~35 GB/s): 4.7 GB a
token, about 5 tok/s at best.

## 5. The phases

| Phase | What | Check |
|---|---|---|
| P0 | `formats/nemotron_st.py`: the index, the per-layer configs, the tensors by role, views of the ggml blocks and the NVFP4 pairs (no copy) | every tensor found, shapes against the config, quant types against hf_quant_config.json |
| P1 | IQ2_XXS: the grid and sign tables (ggml's, MIT), a NumPy decoder, CPU dots against int8 activations (scalar, AVX2, AVX-512 VNNI), the CUDA dp4a GEMV and MoE tile (sm_61 up) | the decoder against ggml's reference on the shipped blocks; each kernel against the float product (check_kq_tiers form) |
| P2 | `models/nemotron/model.py`: the NumPy reference (Mamba-2 step and chunked SSD, gated group norm, NoPE GQA, the latent MoE), the tokenizer and template | perplexity of real text; the chunked prompt against the token-by-token steps (the same logits); a sane chat answer |
| P3 | the CPU engine: Mamba-2 ops in C (conv, scan step, chunked scan), relu^2, the latent MoE on the C kernels, NVFP4 on AVX2 (the NV4 dot has only a float loop there), the server family (detect, load, chat, the Qwen XML tool parser, the reasoning split) | golden logits per tier (scalar, avx2, avx512) against P2; check_server |
| P4 | the GPU program: the dense part on the GPU, the Mamba-2 kernels (step, chunked prompt), attention (the f32 tile on Pascal), the experts in the hot/cold split of Qwen4GPU (CPU parts for the cold ones), HotCache; a GPU memory budget that sizes the hot set | the logits against the CPU engine; tokens a second at budgets of 6, 8, 10, 11 GB |
| P5 | Pascal: every kernel under RECLANK_GPU_CC=61 (fp32 math, dp4a, 48 KB shared memory, no cp.async); the sm_61 build | the same logits under CC=61; check_cuda_builds |
| P6 | the routing profile on real prompts: the expert counts, the hot share for each budget, the starting hot set (expert_profile.npy); the memory table of section 3 in tokens a second | the profile from several kinds of prompt (code, chat, agent turns) |
| P7 | the vision tower (C-RADIO v4-H + mlp1) | an image answer against the reference forward |
| P8 | the LiLiCorr drafter (DFlash) | acceptance, tokens a second |

Order: P0-P3 make the CPU path whole (the small machine without a GPU, and
the reference for P4). P4-P6 are the Pascal + AVX2 machine. P7-P8 after.

## 6. Status

- **P1, the CPU part.** backends/include/iq2_tables.h (ggml's grid, the signs), formats/iq2.py
  (bit-exact against ggml's dequantize_row_iq2_xxs on the shipped blocks), KQ_IQ2_XXS (16) in the
  kq kernels (kernels/kq/iq2.c: the grid bytes unsigned, the signs on x, vpmaddubsw or vpdpbusd;
  the scalar tier the float loop). scripts/check_iq2.py: every tier against the float product on
  real rows and activations (1e-7), the integer tiers the same bits.
- **P2.** models/nemotron/model.py, the NumPy reference: perplexity 42.72 on the first 96 tokens
  of docs/LANES_PLAN.md, the steps against the prompt pass 6e-5, a sensible greedy continuation.
- **P3, the engine.** models/nemotron/kq.py (compile_nemotron_step: KQ_LINEAR on the Q8_0 and
  KQ_NV4 rows, NM_MAMBA, ATTN_PREP without norms or rope + ATTN_QSA on float rows, ROUTER_SIG,
  NM_MOE, RELU2; common/nemotron.c): perplexity 42.72 as the reference (top-1 33% against 34%),
  the steps the same bits as the pass. Decode 8.7 tok/s on the Xeon (47 threads): the dense
  matrices 54 ms (4.3 GB), the experts 45 ms (437 MB: IQ2_XXS at ~10 GB/s, the first thing to
  speed up), the Mamba records 5 ms.
- **AVX2.** NVFP4 had only the float loop on AVX2 (dot2/avx2.c kq2_nv4 now: the codes through
  vpshufb to int8, |w| and the sign on x): the avx2 tier on 8 cores of the Xeon 1839 -> 274 ms a
  token (3.7 tok/s: the matrices 210 ms, the experts 45, Mamba 15), on 47 cores 146 ms.
- **The server (CPU).** models/nemotron/engine.py (family nemotron: the template with
  enable_thinking and medium_effort, qwen_parse_output, RowCache points of the Mamba state
  before the last <|im_start|>, the attention rows as HostRows; an agent's next request goes on
  from all it read). 17 x 23, a tool call and its follow-up (299 of 331 tokens cached), a
  reasoning answer: right, 9 tok/s.
- **P4, the GPU (first form).** models/nemotron/gpu.py (NemotronGPU: the program of
  compile_nemotron_step with NM_ATTN and the split experts: NM_SPLIT, SIGNAL, NM_HOT_MOE, a
  CPU_TASK of NM_MOE on the cold pairs, AWAIT), kernels/nemotron (IQ2_XXS in kq_row_part and
  kq_dequant8, k_nm_conv / k_nm_scan / k_nm_gnorm, relu^2, the attention without RoPE, the
  hot experts), ATTN_PREP without norms on the GPU. Against the CPU engine: the same top token
  at every position, KL ~1e-3 (float on the GPU, int8 x on the CPU). 3090: every expert on the
  GPU 25 ms a step (40 tok/s); 23% hot 38.7 ms (25.8 tok/s: the GPU ~17 ms, the CPU's cold
  pairs 20 ms with 22 threads). NM_MOE of a step got a form with one barrier (nm_moe_few: the
  CPU part of a layer 2.2 -> 0.56 ms with 44 threads, 0.38 with 22; the CPU engine 8.7 -> 10.7
  tok/s; its steps no longer the bits of its prompt pass).
- **P4 kernels.** The IQ2_XXS products on int8 x with dp4a (NM_QX, NM_IQ2_LIN; NM_HOT_MOE
  quantizes x and the act: the grid bytes with their signs as int8x4, (g ^ m) - m): on the
  3090 as Pascal (RECLANK_GPU_CC=61) the shared experts 7.2 -> 1.8 ms a step (+0.5 of NM_QX),
  the hot experts 6.3 -> 2.5 ms; 23% hot 40.9 -> 32.0 ms a token. The cache of the attention
  in bfloat16 on the GPU (the reference's: the model dtype, no KV quantization; 0.54 GB at
  64K) and NM_ATTN2 (a block for each token, KV head and chunk of positions: the 16 query
  heads over tiles of 32 keys, then the chunks merged; the grid does not depend on pos): 8
  layers 0.44 ms at 256 positions, 2.25 ms at 32K, 3.6 ms at 64K (the warp kernel took 3.3
  ms at 200).
- **Prompts.** The parts of a prompt are exactly the sizes of the programs (1024, 256, 64, 16,
  4, 1: no padding, the head reads the last real row; padding gave a wrong first token). The
  groups of 16 rows or more run every expert on the GPU (NM_GROUP_MOE: the pairs sorted by
  expert, a tile of 32 rows of an expert in shared memory once for all its pairs, dp4a; the
  cold experts read from the host memory the GPU maps, rc_gpu_host_register_rw, the hot ones
  from their slots), when the link to the host is at least 0.5 GB/s (measured at the start).
  2048 tokens, 26% hot, the 3090 as Pascal: 105 tok/s with the CPU's cold experts, 177 with
  groups of 256 on the GPU, 261 with groups of 1024 (KL 1.8e-4 against the CPU engine). Of a
  group of 1024 (3.9 s): the shared experts 1.9 s (needs a tiled int8 GEMM: each token's x is
  read for every row now), the experts 1.2 s (the link: 8 GB/s), the NVFP4 products 0.6 s.
- **The weightkeeper** (models/nemotron/keep.py): the expert stacks (whole on each node when
  they fit), the Mamba matrices in KQ_NV4 rows, every other tensor's bytes; 39.5 GB (22.4 +
  17.1) filled in 48 s; NemotronKQ takes it when it holds the checkpoint: ready in 5 s, the
  same logits as from the shards (max diff 0); the GPU maps its writable segments for prompts.
- **Against llama.cpp** (scripts/check_nemotron_llama.py; the 24GB-GGUF, llama.cpp 9d1878a on
  the CPU, 4 x 512 tokens of the plans): the same tokens (2048 of 2048), KL 0.012 (llama.cpp's
  measure: the base tokens above -16), top-1 94%, perplexity 13.23 against 13.34. Against the
  float32 NumPy reference (chunk 0): llama.cpp KL 0.013, the C engine 0.003. The GGUF's
  embedding is Q8_0 (the checkpoint's bfloat16); the GGUF's NVFP4 is ggml type 40. dt has no
  floor now (time_step_limit (0, inf), as the fast path, vLLM and llama.cpp; the slow torch
  path's 1e-4 floor caught ~10% of the dt values: KL 0.083 -> 0.069 on the old measure). The
  vision projector (mmproj GGUF) answers about a chart in llama.cpp (llama-mtmd-cli, 63 s on
  the CPU): the reference of P7. llama.cpp's CUDA build crashed in mmq (invalid argument).
- **The tiled int8 product and int16 x.** NM_GEMM8 (k_nm_gemm8k: a block makes 64 rows x 64
  tokens; for each 128 columns the rows decoded to int8 in shared memory once: IQ2_XXS by its
  grid, NVFP4 through kt_e2m1x4 with a scale for each 16, Q8_0 as stored; a 4 x 4 patch of dp4a a
  thread) for the matrices of the prompt groups; the experts of a group as tiles too (k_nm_gtile:
  32 raw rows of an expert in shared memory once, then 32 pairs at a time). x of every int8 product
  as int16 in two planes (RECLANK_NM_X16, the default: q = 256 hi + lo, two dp4a for each 4
  values): against the float32 reference (chunk 0 of the llama.cpp test) the GPU prompt path KL
  0.0023 -> 0.0008, top-1 96 -> 97-98%. 2048 tokens, 26% hot, the 3090 as Pascal: 445 tok/s (int8
  x 470-480; the row kernels 261); decode 28 ms a token at the P102 budget (the same as with int8
  x). Of a group of 1024 (2.3 s): NM_GEMM8 1.26 s (~31% of the dp4a rate: larger patches next),
  the experts 0.76 s, Mamba 0.17 s.
- **int16 x on the CPU** (RECLANK_NM_X16, the default, as the GPU): NM_QX2 (kq_quant_part16: v =
  128 hi + lo in each 32) and NM_LIN2 (kernels/kq/nmx2.c: |w| with the sign on x, 128 sum(w hi) +
  sum(w lo); on VNNI the planes made unsigned once, hi + 128, and 4 rows on them with the correction
  16512 sum(w), 512 bits and the scales of 8 blocks by one gather), the experts and the head on the
  two planes too. Against the float32 reference (chunk 0): KL 0.0026 -> 0.0004-0.0005, top-1 94 ->
  98-99%. Decode: AVX2 on 8 cores 267 -> 329 ms a token (+23%); AVX-512 VNNI on 47 cores 94 -> 156
  ms (the NVFP4 rows 2.5x the int8 kernel with all the cores, near the same on one: more to find).
- **The tensor cores** (sm_80 up; RECLANK_NM_TC=0 for the dp4a forms of Pascal): k_nm_gemm8_tc (the
  tiles and the decode of k_nm_gemm8k, mma.sync int8: m16n8k32 a plane for IQ2_XXS and Q8_0, two
  m16n8k16 halves for NVFP4) and k_nm_gtile_tc (the experts of a group). The same KL against the
  float32 reference (0.0007). 3090, 2048 tokens: 26% hot 409 -> 536 tok/s, a group of 1024 611
  tok/s (all hot 716); the products 1263 -> 658 ms a group of 1024 (~15% of the int8 rate of the
  tensor cores: the decode into shared memory dominates now; bigger tiles, double buffers next),
  the experts 444-732 ms, the Mamba scan 176 ms (a chunked scan next). Decode: 33 ms a token at 26%
  hot, 17.8 ms all hot (56 tok/s). The CPU engine's prompts have no tiled form for int16 x yet:
  26 tok/s on the 47 cores, 3 tok/s on 8 AVX2 cores (decode 6.1 and 2.9 tok/s).
- **The tiled CPU prompt kernel** (kernels/kq/nmx2.c nm_task_tiled, the groups of 16 tokens up):
  x as int16 (128 hi + lo, exact) against the rows decoded to int16, vpmaddwd (vpdpwssd on VNNI):
  signed by signed, every type on every tier. The lanes are rows (two vectors: 16 rows on AVX2, 32
  on AVX-512) and each pair of x is broadcast, so the int32 lanes sum a unit (32 values; 16 for
  NVFP4) before one conversion and one fma with the row scales times the token's: no horizontal
  sums. A task: a share of the rows, in chunks of 256 columns; each tile decoded once (16 values
  of 8 rows, IQ2_XXS by int16 tables, then 8 x 8 transposes to the pairs), prefetched a tile ahead,
  then run on every token (6 at a time; 3 on AVX2). The same tiles for the experts of a group (the
  runs of 3 tokens or more; the row form for fewer). The Mamba mixer in vectors: the conv by
  channels, silu by a vector exp, the scan with the 16 values of hdim in the lanes (the state of a
  task transposed to [n][d] for its tokens), the gated norm. The products: max relative difference
  5e-7 against the row form; KL against the float32 reference 0.0004-0.0007 (AVX2 0.0004).
  One core: 60-75 GMAC/s (the row form 7-9), 47 cores 2.0-2.5 TMAC/s. A group of 256 on 47 cores
  9.7 -> 1.5 s (the products 6.4 -> 0.80 s, the experts 2.4 -> 0.47 s, Mamba 0.69 -> 0.09 s); a
  512-token prompt 26 -> 141 tok/s on the 47 cores, 3 -> 19 tok/s on 8 AVX2 cores (the products
  2/3 of it: vpmaddwd + vpaddd, two uops for 16 products, is the limit of AVX2). Decode unchanged
  (6 tok/s, 3.1 on 8 AVX2 cores).
- **GPU fusion and the step's products** (3090, every expert hot): a step 18.7 -> 15.2 ms, 1307 ->
  939 kernels. NM_ADDNORM: a block's residual sums and the next norm in one record (the CPU too;
  RECLANK_NM_ADDNORM=0: ADD, ADD, RMS_NORM); a layer with every expert hot has no part of the CPU
  (no split, signal, task, await, add); the sum of a token's slots inside the hot experts' down
  kernel (a warp a slot, in the order of the slots); the router of many experts in a block of 256
  (the same choices); a step's Mamba conv and scan in one kernel (no faster: 23 us). NVFP4 rows of
  a step (k_nv4_gemv2: 16 bytes a lane, x rotated in shared memory, slices of 4096 columns, every
  load first): in_proj 78 -> 60 us (81% of the DRAM rate), out_proj 54 -> 38 us. The IQ2_XXS rows
  of a step (the shared experts, the hot experts): rows in pairs with 4 blocks of loads in flight,
  the grid and the signs in shared memory: 35 -> 26-30 us; still latency-bound (~220 GB/s).
- **The LiLiCorr drafter** (P8; models/nemotron/lilicorr.py, spec.py): Bolt has no MTP layer;
  NVIDIA's drafter is the separate NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr (DFlash: a block of
  8, [last, mask x 7], through 5 bidirectional layers over the context of the target's rows after
  blocks 1, 18, 35, 51, 68, 85, with a grouped dynamic conv around each sublayer and sink logits; then
  the LiLiCorr head: the top 8 of each slot through 2 small layers into a lattice of scores, walked
  greedily). The NumPy reference follows llama.cpp's dflash.cpp; the records NM_DPREP, NM_DATTN,
  NM_DCONV (CPU and GPU) make the drafter a program (the same drafts). Greedy acceptance on real
  chat: code 6.33 tokens a round, chat 3.60 (NVIDIA: 5.26 coding, 3.5-3.9 humanities/QA). The CPU
  engine with it (a verify group of 8, the Mamba states saved and the taken rows replayed through
  NM_MAMBA on a refusal): the same tokens as greedy decoding, code 6.1 -> 15.0 tok/s (2.5x), chat
  5.9 -> 8.6 tok/s (1.5x) on 47 cores; a round: draft 65 ms, verify 295 ms. The tiled CPU product
  from 4 tokens (the verify groups, the drafter's head).
- **IQ2_XXS in rows (KQ_IQ2R) and its products** (3090): the layout alone (the same arithmetic, the
  logits bit for bit the same on the CPU and the GPU, Pascal emulation and the spec rounds) took a
  step 15.2 -> 14.3 ms. Then a lane a part (k_nm_q2_lin, the hot experts' up and down: nm_q2_load
  and nm_q2_dot): one 8-byte load and the int32 sum of 32 values (16 dp4a) a part, one scale; the
  signs by (g ^ m) + (m & 0x01010101) in place of __vsub4 (several instructions from sm_70 on) and
  the 4 bits of a half to bytes by a multiply (no sign table); 4 rows a warp; the grid table and the
  first weights before the wait for the kernel before. The shared experts 26-31 -> 15-16 us, the
  hot experts' up 25 -> 16 us, down 29.5 -> 17 us; a step 13.1 ms (78 tok/s, was 67). These
  kernels are instruction-bound (~4.6 instructions a weight measured; the IQ2 GB/s look low
  because a weight is 2 bits: per weight they match the NVFP4 kernel). The KL of decode steps
  against the float32 reference unchanged (0.0016 mean, top-1 96.5%: the same as before to the
  digits). The CPU reads both forms (iq2.c, nmx2.c: kq_iq2_part, kq_iq2_d; the same bits).
- **Spec decoding on the GPU** (NemotronGPUSpec): a verify group of 8 with the head on every row,
  the Mamba states saved and put back by device copies, the taken rows replayed on the device;
  the products of 2-8 rows on the tensor cores (k_nm_t8: a verify group 163 -> 33 ms). The LiLiCorr
  head as programs (CPU and GPU): NM_LTOPK on the slots' logits (the 8 best by logit then index,
  and their features), the host reads the 56 candidates' embedding rows, then the lattice program
  (ENC_LINEAR of the bfloat16 matrices, the NVFP4 ones as the drafter's, NM_LATT with the slot
  biases, NM_LROWS, NM_LCAT, NM_LSCORE) to the scores; the walk on the host (RECLANK_LILI_HEAD=numpy:
  the NumPy head). The same candidates and drafts as the NumPy head (scores within 0.004 of 7.4).
  The small groups of bfloat16 rows (k_mt_gemv_bf16) read x in 16-byte loads (the same terms):
  the drafter's 30 matrices 4.3 -> 1.8 ms. A draft 42 -> 7.5 ms. Code: plain 78 tok/s, spec 101
  tok/s (1.29x, 5.5 tokens a round, the same 160 tokens); chat 0.94x (3.9 a round; a near tie flips
  at token 106: the group and the step kernels sum in other orders). A round: verify 32.5 ms (the
  hot experts 7.3, in_proj 4.6, the shared experts 4.0, the latent products 3.2, NM_QX 2.2 in 272
  launches, Mamba 1.9). CPU (47 cores): draft 65 -> 38 ms; code 6.1 -> 17.1 tok/s (2.8x), chat
  5.9 -> 10.2 tok/s (1.7x), the same tokens as greedy.
- **The verify group: the experts by expert, fewer launches** (3090; the logits bit for bit the
  same everywhere: prompt groups, steps, the verify group, Pascal emulation). The pairs of a verify
  group share experts (40 for the 72 pairs of a layer on the code prompt): k_nm_hg_up/down take each
  expert once, its rows decoded once a part for all its pairs (2 at a time, 4 rows a warp). The IQ2
  kernels are bound by the LSU pipe (grid lookups, x loads, shuffles), so the sums of the lanes went
  to a transposed butterfly (nm_wsum: N values in N - 1 + log2(32 / N) shuffles, not 5 N; the same
  pairs in the same order). The hot experts of a group 7.0 -> 5.95 ms; the steps' IQ2 kernels too
  (hot up 92 -> 79 us at t = 8). The int8 x of the products is made by the record that writes the
  float rows (NM_ADDNORM, NM_MAMBA's norm, RELU2, NM_HOT_MOE: operands q, s, ql; nm_qx_warp, the
  bits of NM_QX), once for all the products on them (NmEmit.quant): NM_QX 272 -> 9 launches a group.
  The float32 router gates of a group in one pass (k_kq_linear_bf16 takes KQ_F32): 1.5 -> 1.0 ms. A
  verify group 32.0 -> 29.1 ms (1377 -> 1154 kernels), a step 13.1 -> 12.8 ms. Spec: code plain 79.5,
  spec 104.8 tok/s (1.32x, the same 160 tokens); chat 1.02x (was 0.94x). Left in a group: k_nm_t8
  15 ms (in_proj 4.5, the 4096-row products 6.4), the Mamba state copies 1.3 ms a round (and again
  on a refusal: double buffers would drop both), the head 1.1 ms, the scan 1.2 ms.
- **The Mamba states in two sets** (GPU spec): each layer's conv rows and state, both sets, in one
  buffer; the verify group and the replay read the set in use and write the other (NM_MAMBA operand
  23, sd: the byte offset to the set written; k_nm_conv, k_nm_scan), then NemotronGPU.swap: no save
  copy each round, no restore copy on a refusal (1.3 ms each). The same tokens and logits; a round:
  verify 30.9 -> 29.7 ms, the rest 3.3 -> 2.5 ms. Code 107.4 tok/s (1.35x), chat 1.06x.
- **k_nm_t8v: the products of 2-8 rows with the weights streamed** (GPU spec's verify group). k_nm_t8
  took 40-130 us a product: each lane loaded its own rows (34 2-byte loads a step for Q8_0, the IQ2
  grid from global memory) with 1-2 blocks an SM. Now each warp copies the raw bytes of its 16 rows,
  256 columns a stage, to shared memory (cp.async, 2 stages in flight), and each lane decodes from
  there just its A fragments into registers; x of the share (at most 512 columns) staged once; the
  shares (nm_t8_ks) fill the GPU; the E4M3 scales as float16 bits (nm_e4m3x2). Products (8 real
  rows): Q8_0 4096 x 4096 84 -> 35 us, 256 x 4096 43 -> 7, 1024 x 4096 40 -> 12, the IQ2 shared
  experts 42 -> 30, NVFP4 out_proj 57 -> 50, in_proj 127 -> 134 (bound by the instructions of its
  decode and scales); a group's products 15.1 -> 11.4 ms (nemo/t8_bench.py). The sums in other
  shares: against the float32 reference (verify groups of 8 forced tokens, 248 rows) KL 0.00080 ->
  0.00069, top-1 97.6 -> 98.0% (the steps: 0.00161, 96.5%). A round's verify 29.7 -> 25.6 ms: code
  123 tok/s (1.56x; the first 93 tokens the same, then a near tie), chat 1.20x. RECLANK_NM_T8V=0:
  k_nm_t8.
- **k_nm_t8v's NVFP4 shares up to 2048 columns** (in_proj in 2 shares, not 8: the warps' pipelines
  longer, the partial sums smaller): in_proj 134 -> 123 us, out_proj 50 -> 47; a group's products
  10.9 ms (15.1 with k_nm_t8). Verify 24.6 ms: code 127.8 tok/s (1.61x), chat 1.24x; KL against the
  float32 reference 0.00069. Tried and not kept: NVFP4 on the float16 tensor cores (an E2M1 code
  times its E4M3 scale is a float16 exactly; the float32 sums of a whole share kept by the mma):
  121 -> 101 us with x as one float16, but x's rounding raised the KL 0.0007 -> 0.0010 (the tensor
  cores' own float32 sums 0.0001 of it), and with x as two float16 (hi + lo, KL 0.0007) 143 us.
  The prompt groups of 2-8 rows take k_nm_t8v too (their sums in other orders; RECLANK_NM_T8V=0:
  the bits of before).
- **Pascal: the products of 2-8 rows with dp4a (k_nm_t8d)** (Pascal emulation, RECLANK_GPU_CC=61:
  its own sm_61 library). A verify group took 103 ms: without tensor cores NM_GEMM8 fell to
  k_nm_gemm8k, the tiles of 64 tokens of the prompt groups (92 ms for 8 tokens). k_nm_t8d: warps over
  groups of 2 rows, a lane a block of 32 (NVFP4, Q8_0) or a part (IQ2_XXS) decoded once into int8
  words for every token, the two planes of x, the next chunk's raw words loaded before the products
  of this one; IQ2's sums those of k_nm_q2_lin (the same bits, 80 -> 47 us). A group's products
  86.8 -> 15.5 ms (in_proj 868 -> 154 us). nm_wsum as templates (a loop on its steps kept the
  values in local memory: k_nm_t8d's in_proj 211 -> 154 us; the same bits). A verify group 103 ->
  30.6 ms, KL 0.00069 (as the tensor cores'). Emulated spec: code 1.28x, chat 1.01x (was far below
  1). Prompt 2048 tokens / decode (emulated): all experts on the GPU 470 tok/s / 79.7 tok/s; the
  budget of a 10 GB card (4.4 GB of experts) 415 / 31.7; of an 8 GB card (2.4 GB) 393 / 30.5 (the
  CPU's experts on 22 threads; the 3090 with tensor cores 713 / 79.7). RECLANK_NM_T8D=0:
  k_nm_gemm8k.
- **Pascal: the dp4a tiles of the prompt groups (k_nm_gemm8k)** held 222 registers (one block an SM,
  16.7% of the warps; the wait for the loads of the tiles the first stall). Each 32 of a step now
  in two halves of 16 (the int32 sums of each half of each row and token, then the scales as before:
  the same bits), capped at 128 registers (two blocks an SM; the small spill costs less than one
  block); the rows of the tiles read 16 bytes at a time with the chunks of a row in the order chunk ^
  (row / 4) (no conflict among the 16 token rows of a warp; alone it gave nothing: the loads of
  shared memory were not the limit). 1024 real rows: in_proj 17.5 -> 16.0 ms, out_proj 7.1 -> 6.0,
  Q8_0 4096 x 4096 4.0 -> 3.2. Prompt of 2048 tokens (emulated): the budget of an 8 GB card 393 ->
  427 tok/s, all experts on the GPU 470 -> 499. Open: the loads of the next step in flight during
  this one (sm_61 has no cp.async and 48 KB a block: a register prefetch with BK 64), and the CPU's
  experts of a step and of a group (at the budget of an 8 GB card the GPU waits 22 of 33 ms of a
  step and 79 of 102 ms of a verify group for them: spec decoding gives nothing there).

**The IQ2 experts on AVX2 (kq_rows_x2_iq2, kernels/kq/nmx2.c):** a CPU step of 6 threads on the AVX2
tier (at 1.2 GHz) spent 805 ms: NM_LIN2 611 (NVFP4 in_proj and out_proj 399, the IQ2 shared experts
163, Q8_0 42) and NM_MOE 159 (the routed IQ2 experts). kq_dot_x2 decoded each part of a row for each
token: the 4 grid entries and the 4 sign masks by _mm256_set_epi64x, the scale in scalars, two sign
flips (xor and subtraction). Now R rows a pass on T tokens (4 x 1, 2 x 2, 2 x 3): the grid entries
and the signs of a part once for the tokens, the x of a part once for the rows, the table entries by
broadcasts and blends, the signs as +1 / -1 bytes for vpsignb, the 8 factors d (2 s + 1) of a block
in one vector pass, and without VNNI the high plane weighted by 128 in vpmaddwd (not shifted after);
each row and token the int32 lanes and the float order of kq_dot_x2: the same bits (all the tiers
alike, scripts/check_iq2.py; the logits of a CPU step and of a group of 3 the same as before on the
AVX2 and the AVX-512 VNNI tiers). A thread on a 5376 x 4096 shared expert 9.3 -> 7.4 ms (from DRAM
too), 6 threads 1.65 -> 1.25 ms. Used by NM_LIN2 for 1 to 3 tokens (4 and more take the tiled form)
and by nm_moe_few's up and down rows (a step and a group of up to 4). The step 825 -> 802 ms (AVX2;
the IQ2 of NM_LIN2 163 -> 145, NM_MOE 159 -> 150), a group of 3 2234 -> 1980; the AVX-512 VNNI tier
631 -> 609 and 1780 -> 1489. Not done: the NVFP4 projections (half of a step) have no rows form
without VNNI; the tiled form at 8 tokens trails the 512-bit tiers by 12%.
