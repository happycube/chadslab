# EXL3 plan

This plan adds the EXL3 format of ExLlamaV3 to this runtime. EXL3 is a
trellis format (QTIP). It gives 2 to 6 bits for each weight with a quality
near that of a K quant with one more bit. A smaller model moves fewer bytes
for each token, and the decode of this runtime is limited by the rate of
the memory. Thus EXL3 can make the decode faster on the CPU and on the
GPU, and it can put more hot experts on the GPU: on a card of 4 GB (the
Pascal mode of the Qwen models) each byte the experts give is cache or hot
experts.

The source of the facts below is the ExLlamaV3 repository
(github.com/turboderp-org/exllamav3, MIT license; checked at commits
d3739fd and 53ea1d8). Each fact names its file.

## 1. The models

| model | repository, branch | codebook | bits | size |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | our own: convert.py of ExLlamaV3 on Qwen/Qwen3.6-35B-A3B, -b 4.0 -hb 6 -cb mul1 | MUL1 | 4, head 6 | about 19.5 GB |
| Qwen3.5-35B-A3B | turboderp/Qwen3.5-35B-A3B-exl3, 4.09bpw | MCG (converter 0.0.22) | experts 4, dense 5, head 6 | 19.6 GB |
| Qwen3.5-9B | turboderp/Qwen3.5-9B-exl3, 4.00bpw | MCG (converter 0.0.22) | 4, head 6 | 7.2 GB |
| Qwen3.8-27B | turboderp/Qwen3.8-27B-exl3, 3.00bpw | MUL1 (converter 1.4.2) | 3, head 6, MTP 4 | 13.8 GB |
| Gemma 4 26B A4B | turboderp/gemma-4-26B-A4B-it-exl3 | branches of bpw | | |
| Qwen3.8-Flash-Next | turboderp/Qwen3.8-Flash-Next-exl3 | | 2.05 to 6.05 | |

**Qwen3.6-35B-A3B in MUL1 at 4 bits is the first model.** There is no
official EXL3 file of it, so the file is our own: convert.py of ExLlamaV3
on the checkpoint Qwen/Qwen3.6-35B-A3B (bfloat16). This runtime runs the
model on the CPU and on the GPU (reclank/models/qwen/qwen35.py, gpu35.py:
the Gated DeltaNet, the gated attention of every 4th layer, 256 experts of
which 8 and a shared expert for each token), and the references of
QWEN_DENSE_QUANT.md hold: the checkpoint, the file with every dense matrix
BF12, the UD-Q4_K_M GGUF. The architecture of Qwen3.5-35B-A3B
(Qwen3_5MoeForConditionalGeneration) is the same; its file (MCG) only
checks the MCG decoder.

ExLlamaV3 does not support the per-layer inputs of the E2B and the E4B
(architecture/gemma4.py stops with NotImplementedError). The branch name
gives the bits: 4.05bpw_h6_ng6 has 4.05 bits for each weight on average, a
head of 6 bits, and an n-gram table of 6 bits.

## 2. The format

### 2.1 The files

A model is a directory of safetensors files, config.json, and the
tokenizer. quantization_config.json (and config.json) has quant_method
"exl3", the codebook, and tensor_storage: for each module its stored
tensors with their shapes and dtypes, and for an EXL3 matrix its
bits_per_weight and mcg_multiplier (conversion/quant_config.py). The
tensors that EXL3 does not change keep their names and their formats.

The file of Qwen3.5-35B-A3B (4.09bpw, MCG; the MUL1 file of Qwen3.6 at 4.0 has
the same tensors, the dense matrices at 4 bits as well):

| tensors | form |
|---|---|
| the experts (layers.N.mlp.experts.E.{gate,up,down}_proj, 10240 of each) | EXL3, 4 bits |
| in_proj_qkv, in_proj_z, out_proj, q_proj (2 x heads x 256 rows: the query and its gate), k_proj, v_proj, o_proj, the shared expert | EXL3, 5 bits |
| lm_head | EXL3, 6 bits |
| embed_tokens | float16 (1.0 GB) |
| the router (mlp.gate), shared_expert_gate, in_proj_a, in_proj_b | float16 |
| the norms | bfloat16 (linear_attn.norm float32) |

No MTP layer and no vision tower are in this file.

### 2.2 The tensors of a linear layer

A linear layer with k inputs and n outputs has these tensors
(modules/quant/exl3.py, LinearEXL3):

    <key>.trellis   int16, shape (k / 16, n / 16, 16 K): the codes
    <key>.suh       float16, k values: the vector of the input side
    <key>.svh       float16, n values: the vector of the output side
    <key>.su, .sv   int16 bit fields (older files): the signs of suh and svh
    <key>.mcg       int32 scalar, the multiplier: the MCG codebook
    <key>.mul1      int32 scalar: the MUL1 codebook
    <key>.bias      float16, n values, or none

With no marker, the codebook is 3INST. K is the count of bits for each
weight: an integer from 1 to 8, or 1.5, 2.5, or 3.5 (MUL1 only;
quant/bits_k.cuh). A tile of the trellis has 16 K int16 words, or 16 K + 8
for a half K, so the bits of a tensor come from the width of its trellis.
ExLlamaV3 reads only the presence of a marker.

The experts of an MoE layer are linear layers of their own
(modules/block_sparse_mlp.py). The gate and the up matrix can be one
tensor (key_gate_up_split).

### 2.3 A tile

A tile holds 256 weights: 16 inputs by 16 outputs. The codes of a tile are
a stream of bits. Weight t of the tile uses the 16 bits that end at bit
(t + 1) K of the stream. The stream is circular in the tile, so the first
weights use bits at the end of the stream (quant/exl3_dq.cuh, dq: bit
b0 = t K + K - 16 + 256 K, mod 256 K). Thus each weight has K new bits, and
the code of a weight is the last 16 bits of the trellis state.

The order of the 256 weights is that of the fragments of the tensor
cores: tensor_core_perm (modules/quant/exl3_lib/quantize.py) gives the row
and the column of each position of the stream (for lane t of 32, rows
2 (t % 4), + 1, + 8, + 9 by columns t / 4 and t / 4 + 8). The words of the
stream have their halves swapped (SWAP16 in quant/pack.cu).

For a half K = KA + 0.5, the weights have KA and KA + 1 bits in turn (the
mask 0xAAAA; quantize.py, frac_k).

### 2.4 The codebooks

A code x (16 bits) gives a value (quant/codebook.cuh):

- 3INST: x = x * 89226354 + 64248484 (32 bits). The bits
  (x AND 0x8fff8fff) XOR 0x3b603b60 are two float16 values; the value is
  their sum in float16 (one lop3 of immediate 0x6a, then a half add).
- MCG: x = x * 0xCBAC1FED (3417055213), then as 3INST.
- MUL1: x = x * 0x83DCD12D. s is the sum of the 4 bytes of x plus 0x6400
  (dp4a with 0x01010101). The value is fma(float16 of s, k_inv, k_bias) in
  float16, k_inv = 0x1eee and k_bias = 0xc931 (float16 bits): about
  (s - 1024 - 510) / 147.7, an affine function of the byte sum.

MUL1 is the default of the newer converters (doc/convert.md); the fast
paths of ExLlamaV3 (the int8 GEMV, quant/exl3_gemv_int8.cuh; the experts
on the CPU, cpu/moe_mul1.cpp) take MUL1 only. Older files, the first model
among them, use MCG.

### 2.5 The product

The trellis gives the weights W' in a rotated basis. The product is
(modules/quant/exl3.py, forward and reconstruct_hgemm):

    xh = H (x * suh)         the transform of the input
    y' = xh W'               the product with the decoded weights
    y  = (H y') * svh + bias

H is the Hadamard matrix of 128, scaled by 1 / sqrt(128), on each part of
128 values (ext.had_r_128). The matrix of 128 comes from the construction
of Sylvester, from the matrix of 1 (util/hadamard.py). The vector suh holds
the signs, the scales of the input channels, and 1 / -1.24371088, the scale
of the codebook (quantize.py).

For many rows (more than 144), ExLlamaV3 makes the float16 weights of the
tiles (reconstruct, with the transforms in it: reconstruct_had_slice) and
uses a GEMM. For a few rows it decodes the codes in the GEMV, the
transforms inside. The experts (quant/exl3_moe_kernel.cuh) gather the rows
of each expert and apply the input transform of that expert (each expert
has its own suh), the gate and up product, the output transform with the
activation and the input transform of down, the down product, and the
output transform with the scatter add.

## 3. Into this runtime

### 3.1 What the runtime has now

- The Qwen3.6 path on the GPU: the dense matrices of a GGUF on the GPU
  (QwenGPU.dense), the experts in the stores and the warm pool of HotCache
  (runtime.WarmPool, gpumm.ExpertPool), the cold experts on the CPU (the
  KQ_MOE records), the mixed groups of a prompt (copies of the cold experts
  to blocks of the pool), packed programs (ir/alloc.py), Pascal mode
  (RECLANK_GPU_CC=61: no tensor cores, the float32 tiles and dp4a).
- Matrices that read a transformed x: the rotated types RQ4_K, RQ5_K,
  RQ6_K, RQ8_0 (gguf.ROT_BASE; KMat.rot): compile_qwen_step copies x,
  rotates it (GP_TQ_ROT, a WHT of 32 with signs), and quantizes it once for
  the matrices of that x (QWEN_DENSE_QUANT.md).
- The tools of QWEN_DENSE_QUANT.md: an end-to-end measure against a
  reference file (scripts/qwen_dense_eval.py: KL, paired differences),
  the sensitivity of each layer, a plan of types for a budget.

### 3.2 The matrix

An EXL3 matrix is a new kind of matrix of the model: the trellis, suh,
svh, K, the codebook. The records of a product:

1. GP_HAD128 (new): x * suh, then the Walsh-Hadamard transform of each 128
   values (the butterflies of GP_TQ_ROT, in parts of 128), into the buffer
   of the transformed x. Unlike a rotated GGUF matrix, the matrices of one
   x (q, k, v; gate and up) cannot share it: each has its own suh.
2. The product with the decoded weights: a GEMV that decodes the codes of
   each tile in registers for one token and the small groups (as
   exl3_gemv_kernel.cuh), and for a large group the decoded tiles to
   float16 on the tensor cores, or to float32 tiles in Pascal mode (the
   form of k_flash_f32t and of the float32 GEMM of the prompt).
3. GP_HAD128 on the output (the transform, then * svh), fused into the
   records after it where it can be (the add of the residual, the norm).

The packing of temporaries (ir/alloc.WRITES) takes the new records.

### 3.3 The experts

- The MoE kernels (KQ_HOT_MOE of a step, KQ_GROUP_MOE of a group, the
  mixed groups) gain an EXL3 form with the transforms of each expert in
  them, as exl3_moe_kernel.cuh: each pair of a token and an expert needs
  H (x * suh_e) of its own (2048 values here), and the output of down its
  own transform.
- An expert of the pool is a block: the three parts, each its trellis then
  suh and svh, so one copy moves an expert (ExpertPool.per from those
  bytes). At 4 bits an expert of Qwen3.6 takes 1.59 MB (three matrices of
  2048 x 512 and their vectors) in place of 1.77 MB in Q4_K, so about 11%
  more experts are hot in the same memory.
- The cold experts on the CPU: a record of the CPU (as KQ_MOE) that decodes
  the codes in each thread (section 5.3: MCG is compute-bound on the CPU,
  MUL1 near the rate of the memory).

### 3.4 Pascal

The codebooks add float16 values. Pascal runs float16 at 1/64 of float32
(GP102, GP104), so the decode takes the two halves to float32, adds them,
and rounds the sum to float16. The sum of two float16 values is exact in
float32 when their exponents differ by less than 13; past that the double
rounding can differ from a float16 add, rarely. MUL1 needs only dp4a
(compute capability 6.1): its GEMV with int8 x runs on Pascal as it is.

## 4. The steps

### Status

Done: the MUL1 matrices as a type of the products (KQ_E3_BASE + K, K 1 to
8: the trellis, suh, and svh in one block), on the CPU
(backends/cpu/csrc/kernels/kq/exl3.c: AVX512-VNNI, vpdpbusd of the
products code * 0x83DCD12D and of x splat) and on the GPU
(kernels/quant/exl3.cu: the transform of x to int8 with a scale for each
128, a dp4a GEMV, the transform of the output; no tensor cores, so Pascal
runs it); the MoE body of the CPU (kernels/kq/exl3_moe.c, not yet run on a
model); the reader of a directory (reclank/formats/exl3.py, EXL3Dir with the
read methods of SafeTensors); the weight mode exl3 of Gemma 4
(models/gemma4/model.py), whose forms take the matrices as GP_KQ_LINEAR. The
decoder of all three codebooks (exl3_decode) for the checks.

Checks (scripts/check_exl3.py, random tiles, each K): the VNNI products
against the scalar form; the CPU against NumPy -44 to -51 dB (the int8 x);
the GPU against the CPU -118 to -135 dB. turboderp/gemma-4-12B-it-exl3,
3.00bpw_mul1 (scripts/check_exl3_gemma.py): the same greedy answer on the
CPU and the GPU; on the 3090 a decode of 64 tok/s and a prompt of 270 tok/s
(2065 tokens: the GEMV, about 2.7 T weight-tokens/s of dp4a); one large
matrix of one token reads 450 GB/s.

**Qwen3.8-27B** (turboderp/Qwen3.8-27B-exl3, SC_5.00bpw_H6_V6: MUL1,
converter 1.4.2; the MLP and q_proj 4 bits, in_proj_qkv, in_proj_z,
out_proj, and o_proj 5, k_proj, v_proj, and the head 6) runs on the CPU and
on the GPU (section 6.2):

- QwenEXL3Program (models/qwen/qwen35.py) reads the directory: the MUL1
  matrices as they are, in_proj_a and in_proj_b (float16) as float32
  matrices, the norms with their 1 added, the value heads of the DeltaNet
  in the order of the checkpoint (cfg.v_tiled False), the embeddings
  bfloat16. QwenGPU takes a model with no experts (_no_experts: every
  matrix on the GPU, the prompts as groups of the step's records).
- The reference: scripts/qwen38_27b_reference.py (transformers, float32 on
  the CPU, the original bfloat16 checkpoint): two prompts (235 and 63
  tokens of the template, the first 45 a system part the template adds),
  then 200 and 98 greedy steps. The BF12 GGUF of the original weights
  (scripts/convert_qwen35_dense_gguf.py, 41.2 GB) matches it: argmax 594 of
  594 rows, KL mean 5.4e-5 (scripts/check_lfm25_cpu.py --hs-last-normed).
- The EXL3 file against it: on the GPU, the last prompt row and the steps
  (check_lfm25_gpu.py), argmax 314 of 314, KL mean 1.7e-3; on the CPU every
  row (check_lfm25_cpu.py --max-kl 1e-2), argmax 97.6% (the prompt rows;
  the steps 296 of 296), KL mean 8.9e-3.
- On the 3090: decode 31 tok/s; a prompt at 700 to 740 tok/s (512 to 4096
  tokens of real text; 121 with the dp4a GEMV alone). On the CPU (24 cores
  of one node): 261 ms a step (808 before the CPU work below).
- **The prompt on the tensor cores** (k_e3_mma, kernels/quant/exl3.cu; a
  group of 32 rows or more): the integers of the GEMV on mma.sync m16n8k32
  u8 x s8. The byte sum s (0 to 1020) of a code is l + 256 h, so two MMAs
  give l q and h q exactly. The 8 codes of a lane fill the A fragment as
  they are: x goes to shared memory with the inputs of each 16 in the order
  of the fragment (k_e3_had_in perm). 21 to 28 T weight-rows/s on the 27B's
  matrices at 256 to 512 rows (3.1 for the GEMV); the decode costs nothing
  measurable (removing it: 3%), the rest is the loads of x and the latency.
  scripts/check_exl3.py --gpu: -118 to -136 dB against the CPU.
- **The CPU** (kernels/kq/exl3.c): the window of the last codes of a tile
  no longer reads past the tile (one DRAM miss a tile), a prefetch 20 tile
  rows ahead, one vpermw for the codes, the integers folded once a block
  and centred on 510 xn (no cancellation in the float sums), and groups of
  3 to 8 rows decoded once. One row: 5.0 to 5.8 G weights/s a core, 100 to
  114 on 24 cores (57 to 75 GB/s; still bound by port 0: vpmulld, vpsrlvd,
  vpdpbusd). Gemma 4 12B EXL3: 41.2 s to 10.1 s for 48 tokens, the same
  answer.
- **No drift** in free-running generation (work/drift: 2000 greedy tokens
  of two prompts, a story and a B-tree with code, on the GPU; the reference
  teacher-forced on those tokens, scripts/qwen38_27b_reference.py
  --forced): the KL against the reference stays 2e-3 to 4e-3 in each bucket
  of 200 steps (2.5e-3 and 2.1e-3 in the second halves against 3.5e-3 and
  4.4e-3 in the first), argmax 97.1% and 98.3%; the steps against the same
  tokens through the prompt path, 3e-4 to 6e-4 with no trend (the state of
  the steps gathers no error).
- The server: M=q27 work/start_server.sh (the Qwen engine: an EXL3
  directory of model_type qwen3_5); the contexts give the same greedy
  answers when they come back from host memory (work/ctx_check.py).

Next: the MTP layer of the 27B (4 bits in the file); turboderp's
Qwen3.8-Flash-Next 5.05bpw_h6_ng6 (its experts need the MoE kernels); the expert kernels of the GPU and the Qwen3.6 path (Phase 3);
the bits against ExLlamaV3 (Phase 0: its reference needs torch,
scripts/exl3_reference.py).

### Phase 0: a reference

1. ExLlamaV3 in a virtual environment of its own (torch, CUDA; its
   extension builds at the first import).
2. For some tensors of each K and each codebook: the float16 weights of
   reconstruct and of get_weight_tensor.
3. The logits of ExLlamaV3 for the MUL1 file of Qwen3.6-35B-A3B on a text (the teacher-forced
   positions of qwen_dense_eval.py, so the same measure can compare).

### Phase 1: the reader and a decoder

1. reclank/formats/exl3.py: the safetensors and quantization_config.json
   of a directory (the reader of scripts/convert_q36_dense.py, SafeTensors,
   moves there), the matrices with their K and codebook, and the names of
   this runtime (the Qwen3.6 modules: model.language_model.layers.N...).
2. A decoder of a tile in C (backends/cpu/csrc) and NumPy: the window of
   each weight, the three codebooks with the float16 steps of the kernels,
   the order of the tensor cores, SWAP16; then H, suh, svh to the float32
   weights.
3. Test: the weights of Phase 0, bit for bit for the float16 steps
   (scripts/check_exl3.py).

### Phase 1b: the transcode

A model class for an EXL3 directory that gives the Qwen3.6 path its
matrices decoded at the load and requantized in the types of the runtime
(BF12, RQ8_0, RQ6_K of convert_q36_dense.py; or a GGUF written once by
scripts/convert_exl3_gguf.py). It runs the whole path (the names, the
transforms, the float16 tensors of the file) on the CPU and the GPU before
any kernel of EXL3 exists, and gives a first measure against the logits of
Phase 0. It keeps none of the bytes of EXL3 (the experts at 4 bits become
Q4_K or more).

### Phase 2: the product on the GPU

1. GP_HAD128 and its fusions.
2. The GEMV with the decode in registers (MCG, 3INST, MUL1), and the large
   groups (decoded tiles: float16 on the tensor cores, float32 in Pascal
   mode).
3. The MUL1 GEMV with int8 x (dp4a).
4. Test: against the decoder of Phase 1 (check_exl3.py), the rate in bytes
   for each second against Q4_K, Q6_K, and RQ6_K.

### Phase 3: the experts

1. The EXL3 forms of KQ_HOT_MOE, KQ_GROUP_MOE, and the mixed groups; the
   blocks of the pool.
2. The cold experts on the CPU.
3. HotCache, the flags of a step, the server (the Qwen engine) as they are.

### Phase 4: the measures

1. Against Qwen/Qwen3.6-35B-A3B in bfloat16 (the reference file of its
   dense matrices in BF12 of QWEN_DENSE_QUANT.md, its experts at 8 bits,
   or the logits of ExLlamaV3 for the bits of the file): KL of the teacher-forced positions
   (qwen_dense_eval.py), the NLL of the text.
2. The rates of the prompt and the decode: the full GPU, and 4 GB in Pascal
   mode (the context that fits, the hot experts), against a Q4_K GGUF of
   the same model and against ExLlamaV3 on this GPU.
3. The README and check_exl3.py in tests/unit/test_checks.py.

## 5. Open questions and risks

- The float16 steps of the codebooks: a float32 decode does not give the
  bits of ExLlamaV3; Phase 1 matches the float16 steps, the fast paths can
  differ (as the int8 paths of this runtime do).
- The bit order (SWAP16, the circular window, the half K): Phase 1 tests
  it against Phase 0.
- The transforms cost time for each matrix and, for the experts, for each
  pair (a step of Qwen3.6: 8 experts and the shared one in each of 40
  layers, three transforms of 2048 or 512 values each). Measure them in
  Phase 2.
- MCG files are compute-bound on the CPU (section 5.3): the cold experts of
  the 4 GB case run on the CPU, which is why the first file is MUL1.
- Our own conversion needs the GPU (24 GB) and its calibration for hours:
  no server meanwhile.
- The code of ExLlamaV3 has the MIT license; a kernel that follows it keeps
  its notice.

## 6. The research

### 6.1 Facts of ExLlamaV3

- No CPU path for dense models: the CPU code (cpu/moe_mul1.cpp) runs MoE
  experts of MUL1 files only.
- The GPU needs compute capability 7.5 or more (two m16n8k8 in place of
  m16n8k16, no cp.async on 7.5).
- The int8 GEMV of MUL1 uses dp4a only (a weight is an affine function of
  the byte sum of code * 0x83DCD12D), so a MUL1 GEMV can run on Pascal.
- 144 rows or fewer: one fused kernel (int8 GEMV, float16 GEMV, or a
  stream-K GEMM), with the Hadamard transforms inside. More rows: the
  float16 weights, then a GEMM.
- The constant of MUL1 differs a little between the CUDA float16 path and
  the CPU path: about 0.0018 for each weight (the float16 value of k_bias).
- A possible fault of ExLlamaV3: the CPU MoE code takes the bits as trellis
  width / 16 with no check, so a half-bit expert (2.5) decodes as 2 bits.

### 6.2 The dense models of Qwen3.5 and Qwen3.8

The dense models (Qwen3.5-9B, Qwen3.6-27B, Qwen3.8-27B) are the qwen3_5
architecture: the layers of Qwen3.6-35B-A3B with a dense SwiGLU MLP in
place of the experts (the shared expert of qwen35.py has its form). The key
output_gate_type ("swish") is the gate of the norm of the DeltaNet (SiLU,
as Qwen3.6). Their EXL3 tensors: in_proj_qkv, in_proj_z, out_proj, q_proj,
k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head, and the
MTP linears; in 16 bits: in_proj_a, in_proj_b, conv1d, A_log, dt_bias, the
norms, the embeddings, and the vision tower (new converters can make it
EXL3: vision_bits). The runtime has both now: the dense MLP of every layer
(cfg.dense_layers, the form of the dense layers of LFM2) and QwenGPU with
no experts (section 4, Status).

### 6.3 The CPU: what limits a product

A C benchmark decoded K = 4 codes and did the dot product on the Xeon
W-2295 (18 cores, about 2.8 GHz):

| kernel | one thread | 18 threads |
|---|---|---|
| memory read | 13 GB/s | 66 GB/s (132 G weights/s at 4 bits) |
| MCG, AVX-512 (decode to float16, FMA) | 3.0 G weights/s | 53 |
| MCG, AVX2 | 1.6 | 24 (30 with 36 threads) |
| MUL1, AVX-512 VNNI (vpmulld, vpdpbusd) | 7.3 | 102 |
| MUL1, AVX2 (vpmaddubsw, vpmaddwd) | 3.3 | 44 |

- MCG on the CPU is compute-bound; MUL1 is near the rate of the memory.
  The numbers are of that machine: measure them again on this one (two
  sockets, 40 threads for the runtime).
- For one token, all the cores decode (the decode is most of the work).
  For a group of tokens (a prompt, an MTP verify) a thread can decode a
  chunk once into the L2 of its core and use it for each token (16 output
  columns by all the inputs: 64K weights, 64 KB of int8 or of byte sums of
  MUL1). A chunk that one core writes and another reads crosses the mesh
  (a non-inclusive L3), so the same core, or its second thread, uses it.
