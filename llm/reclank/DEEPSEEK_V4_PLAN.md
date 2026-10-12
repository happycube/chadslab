# DeepSeek V4 Flash (Vision) in reclank

The plan for DeepSeek-V4-Flash-Vision-Exp from its safetensors checkpoint:
the original weights (157 GB in 48 shards) with the reference code of
DeepSeek (inference/model.py, kernel.py, vision.py, image_processor.py) and
the chat encoding (encoding/encoding_dsv4.py). Every fact here comes from
config.json, the headers of the shards, or that code (the lines of
inference/model.py are given as model.py:N), unless it says it is a
measure. REFACTOR_PLAN.md section 6b points here. DEEPSEEK_V41_PLAN.md is the
plan for V4.1 Flash (the checkpoint on this machine): no model crippling
(the weights as stored; BF12 for the bfloat16 matrices by default, an
exact mode with the reference's compute; the EXL3 file of section 5 is
out).

The target: the 3090 (24 GB) with the experts in the RAM of the machine
(313 GB, two Xeon 8268), as Qwen3.8; the CPU alone too.

## 1. The checkpoint

| part | tensors | form | GB |
|---|---|---|---|
| routed experts, 43 layers x 256 | w1, w3 (2048 x 4096), w2 (4096 x 2048) | FP4 E2M1, two codes a byte (low nibble first), an E8M0 scale for each 32 values of a row (MXFP4) | 138.5 + 8.7 scales |
| routed experts of the 3 MTP layers | the same | the same | 9.7 + 0.6 |
| attention, shared expert, indexer query | wq_a, wq_b, wkv, wo_a, wo_b; w1, w2, w3; indexer.wq_b | FP8 E4M3, an E8M0 scale for each 128 x 128 block | 5.86 (MTP 0.45) |
| compressors, indexer weights, router | compressor.wkv, wgate (1024 or 256 x 4096); weights_proj (64 x 4096); gate (256 x 4096) | bfloat16 | 0.71 |
| embed, head | 129280 x 4096 each (not tied) | bfloat16 | 1.06 each |
| hyper-connections | hc_attn_fn, hc_ffn_fn (24 x 16384), base, scale | float32 | 0.14 |
| hash routing of layers 0-2 | tid2eid (129280 x 6) | int64 | 0.02 |
| norms, attn_sink, gate bias, bias_vl, compressor ape | | bfloat16 / float32 | small |
| vision | ViT (32 blocks of 1024), aligner, image_start/end/newline/pad | bfloat16 | 0.93 |

The names are those of convert.py after its renames (layers.N.attn.*,
layers.N.ffn.*, mtp.N.*); the shards hold them that way already.

## 2. The model

### 2.1 The shape

- 43 layers, hidden 4096, vocab 129280, context to 1M tokens.
- Each layer: attention, then a MoE (256 routed experts, top 6, inner
  2048, one shared expert of the same size). Layers 0 to 2 route by a
  table of the token id (num_hash_layers 3).
- Every layer keeps 4 streams of the hidden state (hyper-connections),
  mixed at the input and the output of the attention and of the MoE.
- 3 MTP layers (DSpark) and a ViT.
- RMS norms with eps 1e-20 (in effect x rsqrt(mean x^2)).

### 2.2 Hyper-connections (model.py:730-766, kernel.py hc_split_sinkhorn)

The embedding is copied to the 4 streams. Before the attention (and
again before the MoE), with X the 4 streams (16384 values):

    m    = (X . hc_fn^T) * rsqrt(mean X^2 + eps)          24 values
    pre  = sigmoid(m[0:4] s0 + b[0:4]) + 1e-6
    post = 2 sigmoid(m[4:8] s1 + b[4:8])
    comb = softmax_rows(m[8:24] s2 + b[8:24]) + 1e-6, then divided by its
           column sums + 1e-6, then 19 times rows then columns (Sinkhorn)
    x    = sum_h pre_h X_h                                 the input of the block
    X'_k = post_k f(x) + sum_j comb[j][k] X_j              after the block

The head: pre = sigmoid(m s + b) + 1e-6 from hc_head_fn (4 values),
x = sum_h pre_h X_h, then the norm and the head (bfloat16 weights, float32
products).

### 2.3 The attention (model.py:476-586)

- Queries: qr = q_norm(wq_a x) (rank 1024); q = wq_b qr, 64 heads of 512;
  each head times rsqrt(mean q^2 + eps) (no weight); RoPE on the last 64
  values of each head.
- One key-value head: kv = kv_norm(wkv x) (512 values), RoPE on its last
  64 values. The same row is the key and the value.
- RoPE is interleaved (pairs 2i, 2i+1). The layers with a pool
  (compress_ratio > 0) use compress_rope_theta 160000 with YaRN (factor 16
  over 65536, beta 32 and 1); the layers with a window only use theta 10000
  and no YaRN. No YaRN scale on the softmax: the scale is 512^-0.5.
- The QAT: the 448 values without RoPE of kv go through FP8 E4M3 with a
  power-of-two (E8M0) scale for each 64 (act_quant in place), in the
  window and in the pool alike.
- The keys of a query: the window (the last 128 rows of the layer, a ring),
  and the entries of the pool of the layer (2.4): all of them (ratio 128),
  or the top 512 of the indexer (ratio 4, 2.5).
- The attention is a softmax over those keys and a sink for each head:
  the sum of the exponentials gets exp(attn_sink_h - max) (kernel.py
  sparse_attn).
- The output: the inverse RoPE on the last 64 values of each head (model.py:577),
  then a grouped LoRA: the 64 heads are 8 groups of 8 heads (4096 values);
  group g goes through wo_a[g] (1024 x 4096, a slice of the 8192 x 4096
  matrix), and the 8 x 1024 values through wo_b (4096 x 8192).

### 2.4 The compressed pool (model.py:319-417)

compress_ratios: 0 for layers 0, 1, and 42 (window only); then 4 and 128
in turn (21 layers of 4, 20 of 128).

- From the layer input x (float32): kv = wkv x, score = wgate x (bfloat16
  weights). Ratio 4 has two halves of 512 (overlap: the first half belongs to
  the window before), ratio 128 one.
- An entry for each complete window of `ratio` tokens: the sum over the
  window (ratio 4: the 4 rows of the second half of this window and the 4 of
  the first half of the window before, 8 rows) of kv times the softmax over
  the rows of (score + ape[position in the window]), each value on its own.
- Then the norm (compressor.norm), RoPE on the last 64 values at the first
  position of the window, and the FP8 QAT of the other 448.
- A decode step keeps the rows of the incomplete window (kv_state,
  score_state) and makes an entry when it is complete.
- The reference code makes entries in a prefill at position 0 (one chunk)
  and in steps of one token; a prompt in chunks or a group of tokens at a
  later position needs the state across chunks (ExLlamaV3,
  modules/dsv4.py: DSV4CompressorState; cache/dsa.py derives all its
  bookkeeping from the absolute position, so a rewind is a cursor move).

### 2.5 The indexer of the ratio-4 layers (model.py:420-473)

- Its own compressor (ratio 4, head dim 128), with a Hadamard transform of
  each entry (128 values, scaled 128^-0.5) and the FP4 QAT (E2M1 with an E8M0
  scale for each 32): the keys of the indexer.
- Queries: wq_b(qr) (64 heads of 128, FP8), RoPE on the last 64, the
  Hadamard transform, the FP4 QAT.
- The score of entry t: sum_h relu(q_h . k_t) w_h with w = weights_proj(x)
  (bfloat16) times 128^-0.5 64^-0.5; the top 512 entries (causal: only the
  complete windows before the query).
- This is the selection of the QSA of Qwen3.8 (backends/cpu/csrc/common/qsa.c:
  scores relu(q . k) summed over the heads, the best budget blocks), with
  learned weights for each head, a compressor in place of the mean, and the
  transforms.

### 2.6 The MoE (model.py:589-699)

- Router: s = sqrt(softplus(x . gate^T)) (float32); the choice is the top 6
  of s + bias (bias_vl for an image token); the weights are s at those 6,
  divided by their sum, times 1.5.
- Layers 0 to 2: the 6 experts of a token are tid2eid[token id] (an image
  token: the top 6 of s + bias_vl).
- An expert: gate = w1 x, up = w3 x; up clamped to [-10, 10], gate to at
  most 10; silu(gate) up times the router weight, then w2. The shared
  expert has the same clamp (weight 1).
- The reference quantizes the input of every FP8 and FP4 product to FP8
  E4M3 with an E8M0 scale for each 128 (act_quant; kernel.py fp8_gemm,
  fp4_gemm).

### 2.7 DSpark (the MTP layers; model.py:793-1021)

- 3 blocks with a window only (DSparkAttention), their own experts, and
  the embed and head of the model.
- Input: main_x = main_norm(main_proj(the mean over the streams at layers
  40, 41, 42, side by side: 12288 values)); the embeddings of a block of 5
  tokens: the last token, then 4 noise tokens (id 128799).
- The attention of a block reads the window of main_x keys (the KV of each
  block from main_x of the main model) and the block itself.
- The head: logits of the 5 places, each plus a Markov term from the token
  before (markov_w1: 129280 x 256, markov_w2: 256 -> vocab), sampled in
  order; a confidence head (4096 + 256 -> 1) for each place.
- One pass of 3 layers drafts 5 tokens; the main model verifies them in a
  group.

### 2.8 Vision (vision.py, image_processor.py, model.py:961-997)

- ViT: patches of 14 x 14 (588 values), 32 blocks of 1024 (16 heads,
  2-D RoPE, SwiGLU 2816), full attention over one image; RMS norms.
- Aligner: 3 x 3 patches side by side (9216), Linear, GELU, Linear (4096).
- At most 384 tokens an image (vision_max_n_token), in rows with a newline
  token, padding, and image_start / image_end (learned rows); token ids
  vocab + type.
- An image token sees the whole image in the window attention (left and
  right counts; get_window_topk_idxs_visible), and routes with bias_vl.
  The reference prefills an image in one chunk.

## 3. Into reclank

### 3.1 The weights

**Every weight runs in the form of the checkpoint: no requantization, no
conversion file.** The kernels read the codes and the scales as DeepSeek
stores them, so the values are those of the reference bit for bit:

| form | where | the kernels |
|---|---|---|
| MXFP4: E2M1 codes (two a byte, the low nibble first) and an E8M0 scale for each 32 values of a row | the routed experts (and those of MTP) | new type KQ_MXFP4 |
| FP8: E4M3 (one a byte) and an E8M0 scale for each block of 128 x 128 | wq_a, wq_b, wkv, wo_a, wo_b, the shared experts, indexer.wq_b, mtp main_proj | new type KQ_FP8B |
| bfloat16 | the compressors, weights_proj, the router, embed, head, vision, aligner | KQ_BF16 as it is |
| float32 | hc_*, attn_sink, ape, the gate biases | as they are |

- **KQ_MXFP4.** An E2M1 value times 2 is an integer (0, 1, 2, 3, 4, 6, 8, 12
  and their negatives), so a group of 32 is int8 values exactly and the scale
  2^(e - 128): the int8 products of the NVX kernels (kq NVX on the CPU:
  vpdpbusd; k_kqh_nvx and the group kernels on the GPU: dp4a, the int8 tensor
  cores) with an E8M0 scale for each 32 in place of an E4M3 scale for each 16
  and one for the matrix. The same templates, another scale decode. The
  measure that the scales of a matrix span 3 to 5 binades (90 matrices of 6
  layers) shows that a transcode to KQ_NVX would be exact too; it is not
  needed.
- **KQ_FP8B.** E4M3 is a subset of float16 (and of float32): the CPU turns
  each byte into a float (a table of 256, or the conversion of AVX512-FP16
  where there is one) and multiplies by float32 x; the GPU turns pairs into
  half2 (cvt.rn.f16x2.e4m3x2 on sm_89 and later, a table or bit operations on
  sm_86) for the float16 products and the tensor cores of a prompt, float32 on
  Pascal. The block scale applies once for each 128 inputs. On sm_86 the
  conversion is the cost of these kernels (8 bits a weight: memory bound on
  the GPU at one token, the conversion bound in a prompt).
- **The activations.** The reference rounds the input of every FP8 and FP4
  product to E4M3 with an E8M0 scale for each 128 (act_quant). An x of E4M3
  values is exact in float16, so the KQ_FP8B kernels can take that x as the
  reference does. The int8 products of KQ_MXFP4 need int8 x: there the x is
  int8 with a scale for each 32 (finer than the reference's), or float16 x
  in a float16 kernel. The check of Phase 2 measures the KL of each choice
  against the reference; the weights are the same in all of them.
- **In place.** The routed experts of a layer are in one shard: all the
  scales of the layer, then all the codes, by name (expert 0, 1, 10, 100,
  ...). The CPU reads them in the map of the shard through a table of the
  addresses of the codes and of the scales of each expert (the matrices of
  kq_moe_body take a table in place of base + e times the bytes of an
  expert). The pool of the GPU copies the codes and then the scales of an
  expert into one block (13.4 MB): a copy, not a change of form. The dense
  FP8 matrices go to the GPU as they are (codes, then the block scales).
- **The reader:** formats/dsv4_st.py (the shape of formats/nvfp4_st.py):
  the GGUF view of the checkpoint (names, types, the config), the address
  tables; nothing is written.

### 3.2 The cache

The QAT makes a cache of FP8 exact: the 448 values of a row without RoPE
are E4M3 values with an E8M0 scale for each 64 (2.3), and the keys of the
indexer E2M1 values with an E8M0 scale for each 32 (2.5). So:

- a window row and a pool entry: 448 E4M3 bytes, 7 scale bytes, 64 RoPE
  values in bfloat16 (583 bytes); an indexer key: 64 bytes of codes, 4 of
  scales;
- per token: 21 layers of ratio 4 (583 / 4 + 68 / 4 bytes) and 20 of ratio
  128 (583 / 128), about 3.5 KB; 128K tokens 0.45 GB, 1M tokens 3.6 GB;
- the windows: 128 rows of 43 layers, 3.2 MB; the compressor states (the
  rows of the incomplete windows, float32): small.

### 3.3 The memory of the 3090

| | GB |
|---|---|
| dense (KQ_FP8B, the block scales), compressors, router, hc, norms | 5.9 + 0.9 |
| head (bfloat16) | 1.06 |
| cache, 128K tokens | 0.5 |
| programs and scratch | about 1.5 |
| hot experts (KQ_MXFP4, 13.4 MB each) | the rest: about 14 GB, about 1050 of the 11008 |

A token reads 6 experts in each of 43 layers: 3.45 GB of experts, the cold
ones on the CPU. The decode rate follows from the hit rate
of the hot experts (HotCache) and from the bandwidth of the two sockets
(the NUMA copy of Qwen3.8, _numa_copy_experts).

### 3.4 What carries over

| DeepSeek V4 part | Nearest code |
|---|---|
| routed experts (KQ_MXFP4), hot and cold, the pool, mixed groups | the Qwen3.8 path: qwen38.py, gpu38.py, HotCache, WarmPool; the NVX kernels (kq NVX, k_kqh_nvx, qmoe_tc / qmoe_dp4a) with the E8M0 scale of each 32 |
| clamped SwiGLU | the act flags of kq_moe_body and of the GPU act kernels (new: the clamp) |
| the router | ROUTER_TOPK (new: sqrt-softplus, the bias for the choice only, the token table, bias_vl) |
| the indexer and its top 512 | QSA_SELECT (qsa.c; new: the weights of the heads, the compressed keys, the FP4 keys) |
| the windows | the sliding layers of Gemma 4 (the window of 128 as a ring) |
| hyper-connections | the HC_* records of Qwen3.8 (other math: new records HC_PRE with Sinkhorn, HC_POST) |
| FP8 dense (KQ_FP8B) | GP_KQ_LINEAR (new type: E4M3 to float16 or float32, the block scale), the E4M3_ROWS reader (one scale a matrix) |
| MTP drafts and the verify group | the MTP of Qwen3.8 and Gemma (speculative.stream) |
| the Hadamard transform of 128 | kernels/quant/exl3.cu, kernels/kq/exl3.c (e3_fwht128) |
| ViT | the encoders of Gemma 4 (encoders.py, kernels/enc) |
| the chat format | a port of encoding_dsv4.py (no torch); tool calls in DSML blocks |

New: the types KQ_MXFP4 and KQ_FP8B (3.1), the address tables of the
experts in the shards, the compressor (two ratios, the overlap, the state across chunks), the
attention over the window and the pool with sinks and the FP8 cache, the
grouped output LoRA (8 KQ_LINEAR of 1024 x 4096 on the slices of o, or one
record), the inverse RoPE of the output, the per-head query norm without a
weight, DSpark (the block draft, the Markov head, the confidence).

## 4. The steps

### Phase 0: a reference

1. The reference code needs CUDA, tilelang, and fast_hadamard_transform,
   and the whole model on its GPUs. Its ops have plain torch forms on the
   CPU (sparse_attn: a gather and a softmax with the sink; hc_split_sinkhorn;
   act_quant and fp4_act_quant: the scale rounding of fast_round_scale; the
   GEMMs: dequantized weights). With those, the model runs on the CPU in
   torch with the weights from the shards, layer by layer: logits of a few
   short prompts and the output of each layer (scripts/dsv4_reference.py, in
   its own virtual environment; torch on the CPU needs no NVML).
2. The prompts: text at position 0 in one chunk (the reference has no
   chunks), then steps of one token; an image.

### Phase 1: the reader and the two types

1. formats/dsv4_st.py: the GGUF view of the checkpoint (names, the config,
   the types), the address tables of the experts; nothing written.
2. KQ_MXFP4 and KQ_FP8B on the CPU (scalar, then AVX512-VNNI and AVX2) and the
   GPU (sm_86, then Pascal, then sm_89+ with the FP8 conversion of the
   hardware): GP_KQ_LINEAR, the MoE kernels, the pool blocks.
3. Check: the dequantized values against the reference's (E2M1 table times
   the E8M0 scale; E4M3 times the block scale), bit for bit; the products
   against NumPy on those values (scripts/check_dsv4_types.py).

### Phase 2: the model in NumPy and C on the CPU

1. models/deepseek4/: the config, the cache (window rings, pools, the
   compressor states, the indexer keys), the layer in NumPy (2.2 to 2.6),
   the head.
2. Check against Phase 0: each layer's output (dB), the logits (KL), the
   greedy tokens; a prompt in chunks equals one chunk (bit for bit in our
   code).
3. The CPU program: records for HC_PRE (with Sinkhorn) and HC_POST, the
   compressor, the indexer select, the attention with sinks on the FP8
   cache, the router; the experts with kq_moe_body (KQ_MXFP4, the clamp).

### Phase 3: the GPU

1. The dense part, the cache, and the attention on the GPU (Ampere first:
   the attention of 64 heads of 512 over at most 128 + 512 keys a query, or
   128 + n/128 in the ratio-128 layers).
2. The experts as Qwen3.8: hot on the GPU, cold on the CPU (GP_HOT_SPLIT,
   the flags), mixed groups for a prompt (copies of cold experts to the pool).
3. Pascal (RECLANK_GPU_CC=61): the dp4a forms.

### Phase 4: the engine and the server

The engine (models/deepseek4/engine.py), the tokenizer (tokenizer.json), the
port of encoding_dsv4.py (the thinking modes, the reasoning effort, the
tool calls in DSML blocks, the tool results), the parser of the output for
the server; the DeepSeek harness of the user as the client.

### Phase 5: DSpark

The draft of 5 tokens with the 3 MTP layers (their experts: 10.3 GB, cold
first), the Markov head, the confidence as the stop of a draft, the verify
group of the main model (the compressor state across the group, 2.4).

### Phase 6: vision

The image processor (resize to the token budget, patches), the ViT and the
aligner on the GPU, the image block in the prompt, the visibility of an
image in the window, bias_vl.

### Phase 7: the measures

KL and greedy tokens against Phase 0; the rates of the prompt and the decode
on the 3090 with the hot experts and on the CPU alone; the hit rate of the
hot experts; the context at 24 GB.

## 5. Open questions and risks

- **Chunks.** The reference compresses a prompt in one chunk at position 0;
  chunks and groups need the state of the incomplete window (and for ratio 4
  the first half of the window before). Test: a prompt in chunks of every
  size gives the bits of one chunk.
- **The QAT simulations.** The reference rounds the keys (FP8, FP4) and the
  inputs of the FP8 and FP4 products (FP8 with a scale for each 128). This
  runtime gives the products int8 x with a scale for each 32 (closer than
  FP8); the keys it rounds the same way (the cache form of 3.2 is those
  values). Measure the KL of int8 x against the FP8 x of the reference.
- **The indexer.** The top 512 of 64 heads: ties, and the float32 sums of a
  GPU against the CPU can choose another entry near the limit. QSA_SELECT
  keeps the most recent of equal scores; the reference takes torch.topk.
- **The experts.** 3.45 GB a token, most of it cold at first; the two
  sockets each need the experts in their own memory (the NUMA copy); the
  157 GB of the shards (the experts in place) fit the 313 GB of RAM as the
  page cache of the map, but not two copies (a split by node, as Qwen3.8
  does).
- **FP8 on sm_86.** No FP8 tensor cores and no FP8 conversion instruction:
  the conversion of each weight (a table in shared memory, or the bit
  operations of E4M3 to float16) is the cost of the dense products of a
  prompt. Measure it against the int8 products of today; sm_89 and later
  have the conversion (cvt) and sm_120 the FP8 and FP4 tensor cores.
- **Memory on a 4 GB card.** The dense part alone is 5.9 GB in FP8, and the
  weights stay as they are: a small card keeps the dense part on the CPU. Not
  a target of these phases.
- **Images in chunks.** An image sees its whole span in the window: the
  span must be in one group (as the reference requires).
- **EXL3.** ExLlamaV3 has DeepSeek V4 (architecture/deepseek_v4.py,
  modules/dsv4.py, its paged pools and the quantized pool of cache/dsa.py);
  an EXL3 file of this model (MUL1 experts at 3 or 4 bits) would cut the 147
  GB of MXFP4 experts to 104 or 139 GB, with the kernels of EXL3_PLAN.md: a
  requantization, outside this plan.

## 6. The research

- The weights a token uses: in each layer the attention (wq_a 4.2 M, wq_b
  33.5 M, wkv 2.1 M, wo_a 33.5 M, wo_b 33.5 M) and the shared expert (25.2 M),
  5.7 billion in 43 layers, plus the compressors and indexers (0.4 billion);
  the routed experts 6 x 25.2 M x 43 = 6.5 billion; the head 0.53 billion:
  about 13 billion of the 284 billion (304 with the MTP layers).
- The GLM plan (REFACTOR_PLAN.md section 6) shares the Sinkhorn of the
  hyper-connections, the DSA indexer, and FP8 matrices; the records of
  Phase 2 serve both.
- ExLlamaV3 (MIT) has the compressor as a fused kernel (dsv4_compress.cu),
  the quantized pool (dsv4_pool_quant.cu), and the indexer scores
  (attention_fn/dsa_triton.py): prior art for Phase 3.
