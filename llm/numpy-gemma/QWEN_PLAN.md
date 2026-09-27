# Plan: the Qwen3.6-35B-A3B model (OptiQ 4-bit, MLX format)

The model is mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit, in
models/Qwen3.6-35B-A3B-OptiQ-4bit (24.7 GB). This plan gives the parts of
the model, the parts that this runtime must add, and the order of the work.
It uses the method of GPU_NOTES.md: a reference first, a test for each part,
then the speed.

## The model

    part                 value
    layers               40: 30 linear attention, 10 full attention
                         (layer 3, 7, ..., 39 are full attention)
    hidden size          2048
    full attention       16 query heads, 2 key and value heads, head of 256;
                         RoPE on 64 of the 256 values (partial 0.25);
                         a gate on the output (q_proj gives the query and
                         the gate); q_norm and k_norm
    linear attention     Gated DeltaNet: 16 key heads and 32 value heads of
                         128; a causal convolution of 4 on q, k, v; a state
                         of 128 x 128 for each value head
    experts              256 in each layer, 8 for each token, inner size 512,
                         SiLU; a shared expert of 512 with a sigmoid gate
    router               softmax over 256, top 8, the 8 weights normalized
    vocabulary           248320; the head is not the embedding table
    norms                RMSNorm with (1 + w); eps 1e-6
    context              262144 tokens

The linear attention of a decode step, for each value head h, follows
torch_recurrent_gated_delta_rule of transformers:

    q, k = l2norm(q), l2norm(k); q = q / sqrt(128)
    g    = -exp(A_log[h]) * softplus(a[h] + dt_bias[h]);  beta = sigmoid(b[h])
    S    = S * exp(g)                        (S is 128 x 128, float32)
    d    = (v - S^T k) * beta
    S    = S + k d^T
    o    = S^T q
    out  = rms_norm(o) * w * silu(z)          (the gated norm)

A key head serves 2 value heads. The convolution keeps the last 3 inputs of
q, k, and v (8192 values) for each layer.

## The format of the weights

MLX affine quantization, groups of 64 values: w = scale * q + bias, with
scale and bias in bfloat16. OptiQ selects 4 or 8 bits for each tensor. The
bits come from the shape: a U32 of 512 words for 2048 columns is 8 bits, of
256 words is 4 bits. The 256 experts of a layer are one tensor (switch_mlp),
so all the experts of a tensor have the same bits. About 75% of the
tensors are 8 bits.

The sizes, from the headers of the files:

    part                  size      read for each token
    experts               19.6 GB   8 of 256 in each layer: about 0.61 GB
    linear attention      1.02 GB   all
    head (8 bits)         0.54 GB   all
    embedding (8 bits)    0.54 GB   one row
    full attention        0.28 GB   all
    shared experts        0.13 GB   all
    routers               0.02 GB   all
    MTP layer             1.64 GB   (its experts are bfloat16)

Thus a token reads about 2.0 GB of dense weights and 0.61 GB of experts.
The 26B Gemma reads about 1.5 GB of dense weights and 0.8 GB of experts.
The dense part of this model is larger, and its experts are smaller (about
1.9 MB each, 10240 in total).

The limits, from the bandwidth: the CPU reads 67 GB/s, so a step of 2.6 GB
takes at least 39 ms (about 25 tok/s). On the GPU, the dense part takes
about 4.5 ms at 448 GB/s. The experts on the CPU take about 9 ms, less the
experts that the GPU holds (HotCache).

## The two files: MLX OptiQ against GGUF

The second file is unsloth/Qwen3.6-35B-A3B-GGUF, UD-Q4_K_M (22.1 GB), in
models/Qwen3.6-35B-A3B-GGUF. It is the most downloaded GGUF of this model.
Its types, from its header:

    tensors                                   GGUF type        MLX OptiQ
    attention, linear attention, shared       Q8_0             8 bits (most)
    experts: gate and up                      Q4_K             4 bits (most)
    experts: down                             Q5_K (37 layers),
                                              Q6_K (3 layers)  4 or 8 bits
    head                                      Q6_K             8 bits
    embedding table                           Q8_0             8 bits
    router, alpha, beta, conv, norms, A, dt   F32              8 bits, bf16
    MTP layer                                 (not in the file) in the repo

The sizes are almost the same: 19.6 GB of experts in each, and 2.0 GB of
dense weights that each token reads. Thus the two files have the same
limit of speed. The work to support each is different.

What is the same for the two files (most of the work):

- the Gated DeltaNet, the gated attention, the experts and the shared
  expert, the state of the linear layers for Session and MTP;
- the byte-level BPE tokenizer (tokenizer.json, or the tokens and merges of
  the GGUF with the pre-tokenizer "qwen35") and the chat template;
- the program records, the GPU path, and HotCache.

What is different:

    item                      MLX OptiQ                GGUF UD-Q4_K_M
    weight formats            1 family: affine, groups 4 new formats: Q8_0
                              of 64, 4 or 8 bits,      (easy), Q4_K and Q5_K
                              scale and bias in        (super-blocks of 256,
                              separate arrays          6-bit packed scales and
                                                       mins), and Q6_K (the
                                                       CPU and GPU have it)
    kernels to write          1 family with a bits     Q8_0, Q4_K, Q5_K for
                              parameter, for each      each kernel kind
                              kernel kind
    layout of the tensors     as transformers          changed by the
                                                       converter (see below)
    reference for the values  transformers (dequantize llama.cpp on the same
                              and rename), exact       weights: the logits and
                                                       each tensor; also
                                                       transformers after the
                                                       inverse changes
    reference for the speed   none on the same weights llama.cpp on the same
                                                       file
    MTP                       in the repo (its experts not in this file
                              are bf16)                (unsloth has a separate
                                                       MTP-GGUF)
    loader                    SafeTensors (exists)     GGUF (exists), and the
                                                       types above

The changes of the converter of llama.cpp (conversion/qwen.py) that the
GGUF has:

- the value heads of the linear layers go from grouped order to tiled
  order. This changes qkv (the v rows), z, a, b, A, dt, the v part of
  conv1d, and the columns of out_proj;
- 1 is added to each norm weight except the gated norm of the linear
  layers;
- A_log becomes -exp(A_log) (ssm_a), and the names change.

The model code can undo the order at load, so the two files use the same
code. The rows of qkv, z, a, and b move as whole rows of blocks. The
columns of out_proj move by value heads of 128. That is 4 blocks of Q8_0,
so whole blocks move and no value changes.

The estimate of the work:

- The formats: MLX needs one family of kernels. It has the product for
  one token, for a group, and for the experts, on the CPU and on the GPU
  with the tensor cores. GGUF needs the same kernels for Q8_0, Q4_K, and Q5_K:
  about 2 to 3 times that work. The K formats have a second level of
  scales, packed in 6 bits (and a fifth bit for Q5_K).
- The math of the formats is similar. Both are affine: a scale and an
  offset for each group of values (MLX: 64 values, scale and bias; Q4_K:
  32 values, d * sc and dmin * m). For both, a product is
  sum(scale * dot(q, x)) + sum(offset * sum(x)).
- The checks: GGUF has llama.cpp on the same weights. That helps most for
  the linear attention, the hard part: a tensor of each layer can be
  compared (llama-eval-callback). For MLX, transformers is the reference,
  and it has the same layout, so the checks are direct.

The decision: build the model on the MLX file first, with the format of a
matrix behind one interface (the kernels take a format tag). The reference
is transformers with the same names and order. Then add Q8_0, Q4_K, and
Q5_K, and the inverse order at load, for the GGUF file. Use llama.cpp on the
GGUF as the reference for the speed, and as a second reference for the
values.

The rate of llama.cpp (build-cuda) on the GGUF file, with the other load
of this machine (llama-bench, -fa 1):

    setup                                    prompt           decode
    dense part on the GPU, experts on the    224 tok/s (512)  39.5 tok/s
    CPU (-ngl 99 -ncmoe 40)
    no layers on the GPU (-ngl 0)            60 tok/s (128)   10.1 tok/s

These are the values to beat. The decode of 10.1 tok/s on the CPU is far
from the limit of about 25 tok/s (2.6 GB for each token at 67 GB/s).

## The references

- transformers 5.17 (the venv of gemma4-12b-qat-pytorch) has
  Qwen3_5MoeForCausalLM. It needs float weights: dequantize the MLX blocks
  and change the names (switch_mlp to experts). A model with the first 4
  layers (3 linear, 1 full) runs fast and tests each part. The full model
  in bfloat16 needs about 70 GB of RAM; the machine has 188 GB.
- llama.cpp has src/models/qwen35moe.cpp and qwen3next.cpp. They are a
  second reference for the linear attention and the speed. They need a GGUF
  file of the model (about 20 GB). Get it only with an agreement: the disk
  has about 47 GB free after this download.
- The tokenizer library `tokenizers` (in the venv) is the reference of the
  tokenizer.

## Phase 1: load, tokenize, and a NumPy model

1. np_gemma/qwen.py: the config (the text_config of config.json) and the
   names of the tensors. Also the MLX unpack: U32 words to 4-bit or 8-bit
   values, then scale and bias. Read with the SafeTensors class.
2. A tokenizer for byte-level BPE. The Gemma tokenizer uses the
   sentence-piece space and does not fit. The new one needs the map of bytes
   to characters of GPT-2 and the split pattern of Qwen2. That pattern uses
   Unicode classes (\p{L}, \p{N}), so it needs the `regex` module (in the
   venv; not in the system Python). Test: the same ids as `tokenizers` on
   the README, the code of this repository, and 10000 random strings.
3. The chat template: port chat_template.jinja (roles, tools, and the
   <think> block). Test: the same text as apply_chat_template of transformers
   for the cases of scripts/check_chat_template.py.
4. The model in NumPy (float32): the forward pass with a cache. The cache
   has keys and values for the 10 full layers, and the convolution inputs
   and the state for the 30 linear layers. The prompt pass first uses the
   token loop of the recurrent form; it is exact and simple.
5. Test against transformers: the hidden state of each layer of the
   4-layer model, then the logits and 32 greedy tokens of the full model.

Status of phase 1 (done):

- np_gemma/qwen_tok.py: the same ids as the tokenizers library on 5031
  texts (the files of this repository, 10 hard samples, 5000 random
  strings). It needs no regex module (scripts/check_qwen_tok.py).
- np_gemma/qwen.py: the model in NumPy. On the first 4 layers (3 linear,
  1 full), the logits agree with transformers to 7e-7 (max rel). This is
  true for one pass, and for a prompt pass with steps after it
  (scripts/check_qwen.py with scripts/qwen_reference.py). The whole model gives "The capital of France
  is Paris." It takes about 4 s for each token.

Two facts that the tests found:

- The MLX files keep the norm weights with the 1 added (mlx-lm adds it at
  the conversion). transformers keeps w and adds 1. The reference script
  gives w - 1 to transformers. The test with 4 layers did not find this
  first, because the model and the reference used the same wrong rule.
  Only the text of the whole model showed it.
- A model that transformers makes on the meta device has no inv_freq for
  RoPE (a buffer that the state dict does not have). The reference script
  computes it again.

## Phase 2: the CPU path

The CPU path uses the program interpreter (gemma_run) with new records:

- AFFINE_LINEAR (4 and 8 bits, groups of 64): for each group,
  scale * dot(q, x) + bias * sum(x). The sum of x for each group comes one
  time for each row of x. AVX-512 with VNNI: x in int8 for each group, as
  the int8 path of the Gemma prompt pass.
- A multi-matrix form, because qkv, z, a, and b read the same x.
- The experts: the designs of gp_moe_one and gp_moe_group with the new
  format.
- DELTA_STEP: the convolution step, the l2 norms, the decay, the update of
  the state, and the gated norm. The state of a layer is 2 MB (32 x 128 x
  128 floats). Each thread takes some value heads, so the state stays in
  its cache.
- The gated full attention: the int16 cache of the Gemma path, partial
  RoPE on 64 values, and the sigmoid gate on the output.

Test: the same tokens as the NumPy model. Measure: the rate against the
limit of 39 ms, with the profiler of CPU_PLAN.md phase 0.

## Phase 3: the state of the linear layers

The state of a linear layer has no rows, so a cache cannot cut it back.
Two parts of this runtime cut the cache:

- Session: a new turn reuses the common prefix. Keep a copy of the states
  (60 MB) at the end of each turn. A turn that starts at that point uses the
  copy; any other point runs the prompt again.
- MTP verify: the target runs the next token and the drafts, and the
  rejected drafts must not change the state. The verify kernel does not
  change the state. It keeps k, the delta, and the decay of each token.
  Then the count of accepted tokens is known. A commit record applies only
  those updates to the state (each is a rank-1 update of 128 x 128).

Test: MTP gives the tokens of the plain decode; a second turn gives the
tokens of one prompt pass of the whole chat.

## Phase 4: the GPU path

The design of the 26B: the dense part on the GPU (about 2.0 GB), the
experts on the CPU, HotCache for the most used experts. The GPU has about
4 GB free with the other work of this machine. The dense part, the states,
the cache, and about 1 GB of hot experts fit in it.

- New kernels: the affine product (4 and 8 bits) for one token and for a
  small group, and the delta step for 32 heads. Also the convolution, the
  gated norm, and the gated attention.
- The rules of GPU_NOTES.md apply from the start. Use fused small
  operations and PDL at the start of each kernel. Run the host part while
  the GPU runs the head.
- HotCache: first get the traces of real answers (the routers of 6 chat
  answers), and simulate the policies, as for the 26B. The experts are
  smaller (1.9 MB), so 1 GB holds about 520 of 10240 (5%). The simulation
  says if that is enough.
- The prompt pass: first the token loop of the recurrent form on the GPU
  (32 heads in parallel, the tokens in order). For 1024 tokens and 30
  layers that is about 30000 small steps. Then the chunked form of the
  delta rule (torch_chunk_gated_delta_rule, chunks of 64) on the tensor
  cores. The experts of a large group come to the GPU, as for the 26B.

## Phase 5: MTP

The MTP layer has these steps. The matrix fc takes two norms: the row of
the next token in the table of embeddings, and the hidden state. It gives
2048 values.
Then come one full attention layer with experts, a norm, and the head of
the main model.

Its 256 experts are bfloat16 (1.6 GB); quantize them
to 4 bits at load, as quantize_q4_0 does for the drafter of the E4B. The
model card gives about 70% of the drafts accepted with 2 drafts. It needs
phase 3 (the state of the verify group).

## Phase 6: the server and the tests

- serve.py: find the kind of the model from config.json (model_type
  qwen3_5_moe). Add the stop tokens (<|im_end|>, <|endoftext|>).
- The <think> part of the output: np_gemma/chat.py parses the channels of
  Gemma, and Qwen needs its own rule.
- The benchmarks of the other models: the prompt pass, the decode, MTP, and
  the check scripts.

## Order, and what is not in the plan

1. Phase 1: the reference and the NumPy model. Nothing else can be tested
   without it.
2. Phase 2, then phase 3, then phase 4 and phase 5.
3. The vision part (optiq_vision.safetensors) and the multimodal RoPE of
   images are not in the plan. For text, the three parts of the
   multimodal RoPE have the same position, so it is the usual RoPE on 64
   values.
4. The KV cache of kv_config.json (4 or 8 bits for each full layer) is not
   necessary at first. The int16 cache of this runtime is more exact, and
   10 layers of 2 heads are small (about 20 KB for each token).
