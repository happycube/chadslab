# Plan: Qwen3.8-Flash-Next (a preview of Qwen4) in this runtime

This file gives the plan to run Qwen3.8-Flash-Next with the GGUF file of
Unsloth (UD-Q4_K_XL) and its MTP drafter. QWEN_PLAN.md has the work on
Qwen3.6-35B-A3B; this plan uses its kernels and its lessons.

## The files

    file                                               size      where
    UD-Q4_K_XL, 4 shards (tensors in shards 2 to 4)    111.3 GB  models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL
    MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf        2.79 GB   models/Qwen3.8-Flash-Next-GGUF/MTP
    config.json of Qwen/Qwen3.8-Flash-Next             -         models/Qwen3.8-Flash-Next-GGUF/base

The drafter is "shared": it uses the token embeddings and the output head
of the main model (Unsloth: 1 to 2 GB less memory). Q8_0 keeps more of the
accuracy of the drafter than Q4_K_M. The rate of accepted drafts depends on
it.

## The model

The architecture is qwen4_exp in transformers 5.17 (models/qwen4_exp).
llama.cpp calls it qwen4exp: the branch qwen4exp/mtp of
danielhanchen/llama.cpp has it. A copy of its sources is in
llama.cpp-qwen4exp (src/models/qwen4exp.cpp).

    parameters            125B, 6B active for each token; 51B in the n-gram
                          table; 4B in the MTP layer
    hidden size           2560; the residual has 4 streams (10240 values)
    layers                48: 12 x (3 Gated DeltaNet + 1 QSA attention),
                          each with a MoE
    Gated DeltaNet        16 key heads, 48 value heads, 128 each (Qwen3.6:
                          16 and 32)
    QSA attention         24 query heads, 2 key-value heads, 256 each; RoPE
                          on 64 values (mrope sections 11, 11, 10); sigmoid
                          output gate (as Qwen3.6)
    MoE                   512 experts, 10 for each token, inner 640; a shared
                          expert with a sigmoid gate (as Qwen3.6)
    vocabulary            248320, the tokenizer of Qwen3.5 (pre qwen35)
    context               262144

The parts that are new, from modeling_qwen4_exp.py:

1. The gated residual (hyper-connections, 4 streams). The embeddings go to
   all 4 streams. Before the attention and before the MoE of each layer:
   - h is the grouped RMS norm of the 4 streams: a norm for each 2560
     values, with the weight 1 + w;
   - m = sigmoid(up(silu(down(h) / 4))), with down 10240 -> 320 and up
     320 -> 10240;
   - the input of the block is the mean over the streams of m * h.

   After the block, stream s gets out * 2 sigmoid(inject_s(h) / 4) (inject
   10240 -> 4). The model has no final norm. A last mixer (the same, with
   no inject) makes the input of the head.
2. The n-gram table (PLE), at layer index 1. It has 16 heads: 8 heads use a
   hash of the last 2 tokens, and 8 heads a hash of the last 3 tokens. The
   hash uses splitmix64 multipliers, XOR, and a prime modulus for each head.
   The GGUF has the multipliers, the offsets, and the sizes.
   - These are the values of transformers with the seed 1234, its default
     (config.json has no seed). The sizes are the 16 primes after 20000002
     (checked).
   - Each head reads a row of 160 values from a table of 320M rows (51B
     parameters).
   - A key for each stream and a value come from the 16 rows.
   - The gate of each stream is the product of the key and the normed
     stream. A dilated depthwise convolution follows (kernel 4, dilation
     3).
   - The output goes to all 4 streams. Only 16 rows are read for each
     token, so the table stays in the memory map.
3. QSA (Qwen Sparse Attention). An indexer has 4 query heads and 1 key head
   of 128 values.
   - It scores blocks of 4 keys: the mean of the keys of the block, with
     the sum of relu(q . k) over the heads.
   - It keeps the best 512 blocks (2048 tokens) and the tail.
   - For a context of at most 2048 + 3 tokens it keeps all the keys. Then
     the attention is the full attention.
   - The indexer keeps one key of 128 values for each token.
4. The MTP layer (blk.48; llama.cpp graph_mtp). Its input is the hidden
   state of the main model before the last mixer (4 streams), and the
   embeddings of the next token.
   - For each stream: concat(enorm(e), hnorm(stream)) -> eh_proj (5120 ->
     2560), where e is the row of the next token.
   - Then a layer with full attention and its own 512 experts (Q8_0).
   - A head mixer (hc_head_down, hc_head_up, hc_head_norm) comes before the
     shared output head.

## The inventory (the headers of the 4 files)

    part                        size      types
    experts (48 x 512)          77.0 GB   Q4_K 44.4 (gate, up), Q5_1 27.1
                                          (down, 43 layers), Q8_0 4.5,
                                          Q5_K 1.2
    n-gram table                28.8 GB   IQ4_NL (320M rows of 160)
    dense (the rest)            4.15 GB   Q8_0 3.8, F32 0.31, BF16 0.04
    output head                 0.68 GB   Q8_0
    token embeddings            0.68 GB   Q8_0
    MTP drafter (other file)    2.77 GB   Q8_0 (experts 2.67, dense 0.10)

- An expert is 3.13 MB; the experts of a layer are 1.60 GB. The 10 experts
  of a token in 48 layers are 1.5 GB.
- The largest dense tensors: attn_qkv 1.0 GB, attn_gate 0.6, ssm_out 0.6,
  and attn_q 0.4. The gated residual (hc) is 0.67 GB, the routers (F32)
  0.25 GB.
- New types for gguf.py and kquants.c: Q5_1 and IQ4_NL.
- A token reads about 6.3 GB on the CPU alone (dense 4.15, experts 1.5,
  head 0.68). The CPU limit is about 10 tok/s at 60 GB/s, not 16.

## The next machine

The work moves to another machine: an RTX 3090 (24 GB, 936 GB/s) on PCIe
3.0 x16, and possibly two sockets (NUMA). The changes for it:

- Ampere has no programmatic dependent launch. csrc/gpu.cu must not use
  griddepcontrol and programmatic edges on sm_86.
- PCIe 3.0 x16 gives about 12 GB/s. The copies of the experts from the
  memory map go through the bounce buffers of the driver. A circular
  pinned buffer (worker threads copy to pinned memory, then DMA) gets the
  full rate.
- NUMA: with enough RAM, keep a copy of the experts on each socket. Then
  each socket runs half of the cold experts of a layer from its local
  memory.

## Memory

- The machine has 188 GB of RAM, about 127 GB free with the other work.
  The CPU has 18 cores (67 GB/s). The GPU has 16 GB, about 7.5 GB free.
- The weights stay in the memory map of the file. The n-gram table is read
  16 rows at a time, so most of it stays on the disk.
- The experts and the dense part must stay in the page cache. The experts
  are about 24576 of about 2.9 MB each (about 72 GB; to check with the
  inventory). The working set is about 75 to 80 GB, less than the free RAM.
- If the page cache cannot hold the working set, the decode reads the NVMe
  (about 3 GB/s) and it is very slow. Phase 1 measures the resident set.
- The caches:
  - 36 linear layers, each with a state of 48 x 128 x 128 floats (3.1 MB,
    113 MB in total);
  - 12 QSA layers with int16 keys and values: about 2.2 KB for each
    position and layer (6.8 GB at 262144 positions);
  - the keys of the indexer.

## The expected rate

- A token reads about 6.3 GB on the CPU alone (see the inventory). At the
  58 to 67 GB/s of this CPU, the limit is about 10 tok/s. The Qwen3.6 path
  reached about 60% of its limit (18 tok/s of 30).
- MTP (Unsloth: 1.3 to 1.7 times) runs a verify group of 2 to 6 tokens.
  The experts of a group are read one time, so a group costs much less than
  the same count of steps.
- The GPU (phase 5): the dense part (about 2.7B parameters) and the head
  fit in the free memory. A HotCache of the experts works as for Qwen3.6.

## Phases

### Phase 0: the files and the references (in progress)

- Download the shards (33 MB/s, about 1 hour) and the drafter (done).
- Read the tensor inventory: the names and the types. Also the sizes of the
  experts, of the n-gram table, and of the dense part. UD-Q4_K_XL can use
  Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, and BF16. Add the types that gguf.py and
  kquants.c do not have.
- Build llama.cpp of the branch qwen4exp/mtp (CPU, and CUDA if it builds).
  Measure the prompt, the decode, and the decode with MTP. That is the rate
  to beat, and a second reference for the values.
- The GGUF reader reads the split files as one model.

### Phase 1: the NumPy reference

Status (in progress):

- np_gemma/qwen4.py: the NumPy model on the GGUF file (Qwen4, Qwen4Cache,
  config_from_gguf). A chat prompt gives "The capital of France is Paris."
- gguf.py reads Q5_1 and IQ4_NL (bit-equal with ggml), and the split files.
- The DeltaNet of qwen4exp gates its norm with sigmoid, not silu
  (config.json output_gate_type; llama.cpp qwen4exp.cpp). QwenConfig.lin_gate
  selects it.
- scripts/check_qwen4_ngram.py: the n-gram ids equal those of transformers
  (one pass, and a prompt, then steps; with the eos token of PLE).
- scripts/check_qwen4.py: the sums of the first 8 layers are within 0.1% to
  5% of llama.cpp (llama-eval-callback; llama.cpp quantizes x in its
  products).
- The reference of llama.cpp (the branch qwen4exp/mtp, build-cuda), with
  the other load of this machine:

      setup                                  prompt           decode
      dense part on the GPU, experts on      81 tok/s (512)   17.8 tok/s
      the CPU (-ngl 99 -ncmoe 48)
      CPU only (-ngl 0 -nopo 1)              34 tok/s (128)   5.0 tok/s
      CPU only, MTP (shared Q8_0), 3 drafts  -                9.0 tok/s
                                                              (82% accepted)
      CPU only, MTP, 5 drafts                -                8.4 tok/s
                                                              (71% accepted)

  MTP with the dense part on the GPU did not fit next to the other work of
  the GPU (8 GB). The drafter needs a compute buffer of 1.7 GB and a cache
  of the linear states for each draft.
- The QSA indexer (Qwen4.qsa_mask) and the attention with its key mask
  are in qwen4.py. The script check_qwen4_indexer.py gives 2600 random rows
  to both; the kept keys equal those of transformers (549 queries drop
  blocks). The function relu gives
  many scores of 0. At the cut, torch.topk keeps an arbitrary subset of
  equal scores, and this runtime keeps the most recent blocks (24 queries
  differ only there).
- Next: the resident set of the memory map during a decode, then phase 2.

- QwenConfig for qwen4exp, and the names of the GGUF tensors. Check the
  order of the value heads of the DeltaNet (tiled, as Qwen3.6?).
- New NumPy parts: the gated residual, the n-gram table (the hash of
  transformers), the indexer of QSA, and the head mixer. A test compares
  the n-gram ids with transformers on real text.
- Checks against transformers: the model on the meta device, filled with
  the dequantized weights of the first layers (as
  scripts/qwen_reference.py). The n-gram table is too large to fill. The
  reference gets only the rows that the text uses, and a map of the ids.
- A first check of the whole model: a chat prompt gives a good answer.
- Measure the resident set of the memory map during a decode.

### Phase 2: the CPU path

Status (in progress):

- compile_qwen4_step and Qwen4CPU (np_gemma/qwen4.py): the step as one
  program. New records in csrc/hyperconn.c: HC_NORM, HC_ACT, HC_MIX,
  HC_ADD (the gated residual), PLE_GATE, and PLE_CONV (the n-gram layer; its
  convolution state is in Qwen4Cache). Python makes the 16 rows of the
  n-gram table of each token (the hash, then kq_rows of IQ4_NL).
- kquants.c has Q5_1 (the down matrices of the experts: the one-token
  product and the tiles, bit-equal) and the rows of IQ4_NL.
- The flags of the GDN record (CPU and GPU): 1 for the tiled order, 2 for
  the sigmoid gate of the norm.
- A chat prompt: "The capital of France is Paris."; the decode is 5.9 to
  7.4 tok/s on the CPU (llama.cpp: 5.0 tok/s).
- scripts/check_qwen4_cpu.py: with 4 layers, 86% of the top tokens are
  those of the NumPy model, and the logits are within 13%. The error of
  int8 x is spread over all the products: with one group of products
  exact, it changes little. A router of 512 experts turns small changes
  into another expert more often than the 256 of Qwen3.6.
- QSA in the program (csrc/qsa.c). QSA_SELECT keeps the raw keys of the
  indexer and the key of each complete block (made one time). It also
  selects the keys of each query. ATTN_QSA is the
  int16 attention on those keys (attn_i16_head_rows, the head of the 26B
  with a list of rows). For 2600 random rows, the selection of the C code
  equals that of Qwen4.qsa_mask for all 549 queries that drop blocks.
- A prompt of 2941 tokens (QWEN_PLAN.md) and a question on it: a correct
  answer. The prompt pass: 31 tok/s (with the compile); the decode: 152 ms
  for each token.

- Use the kernels of kquants.c for these parts:
  - the products (int8 x, a scale for each 32 values);
  - the MoE (512 experts, 10 for each token);
  - the DeltaNet (48 value heads).
- New records:
  - HC_PRE (the grouped norm, down, up, and the mix) and HC_POST (the
    inject);
  - PLE (the hash and the rows, the gates, the dilated convolution; the
    state of the convolution goes in the cache);
  - the indexer and the selection of blocks of QSA (only for a context of
    more than 2048 tokens);
  - the head mixer.
- The step as one program (compile_qwen_step with the new records), the
  prompt pass in chunks, and the int16 cache.

### Phase 3: MTP on the CPU

Status (done; scripts/check_qwen4_mtp.py):

- The MTP layer (blk.48, in its own file; GGUFSplit.attach adds its
  tensors). Qwen4.mtp is the NumPy layer, after graph_mtp of llama.cpp:
  - the inputs: the streams of the model at position p - 1 (zeros before
    position 0) and the token at p;
  - for each stream: enorm of the token row, then hnorm of the stream
    (the record HC_CAT), then eh_proj (5120 to 2560);
  - hc_attn and a dense attention (its own int16 cache, Qwen4MTPCache);
  - the MoE (512 experts, Q8_0), the head mixer nextn.hc_head, and the
    head of the model;
  - the streams of the layer are the input of the next draft.
- compile_qwen4_step(mtp=True) is the layer as a program. A dense layer
  uses ATTN_QSA with a count of -1 (all the positions). With the streams of
  4 layers, 91% of its top tokens are those of Qwen4.mtp.
- compile_qwen4_step(verify=True): the GDN records write a log. Qwen4CPU
  commit(n) applies it, and makes the state of the n-gram layer again
  from a copy and the first n inputs of its convolution. A verify group
  and commit give the same bits as steps. The keys of the indexer need no
  change: a block that is not complete gets its key again in the next run.
- Qwen4CPU.generate_mtp: each round, one group of the MTP layer does two
  things. It remakes the keys of the accepted drafts with the streams of
  the model, and it gives the first draft. The next drafts use the streams
  of the MTP layer. Then one verify group of the model runs. The tokens
  are those of greedy decode with no drafts.

The rate of the decode (greedy; the plain decode is 6.9 to 7.2 tok/s):

    drafts  accepted  tokens a round  tok/s   the time of a round
    1       98%       2.00            9.9     draft 15 ms, verify 182 ms
    2       93%       2.88            11.4    draft 29 ms, verify 216 ms
    3       89%       3.63            13.1    draft 44 ms, verify 223 ms
    4       80%       4.26            13.0    draft 58 ms, verify 261 ms

This is a code answer of 98 tokens. The answer "why the sky is blue" with
3 drafts: 67% accepted, 10.8 tok/s. llama.cpp with MTP on the CPU:
9.0 tok/s (3 drafts). The verify group costs more than a step (a group of
4 tokens reads up to 40 experts in each layer).

With the keys of the accepted drafts and the first draft in one group,
3 drafts give 13.7 tok/s on the code answer.

A draft head on the first 32k to 131k tokens only is 16 to 32 ms faster
each round. But it loses more in accepted drafts (code has tokens with
high numbers), so all the head stays.

### Phase 4: the rate of the CPU

Status (the profile; gemma_profile, 18 threads):

    the step of one token: 129 ms
      the dense products (all Q8_0, 3.9 GB)    80 ms   49 GB/s
      the experts (1.75 GB)                    37 ms   47 GB/s
      the rest (GDN, norms, the mixers)        12 ms
    the verify group of 4 tokens: 193 ms (the experts: 85 ms)
    the MTP layer (one token): 3.2 ms; the head (675 MB, Q8_0): 11 ms

- A read of 4 GB by 18 threads gives 66 GB/s on this machine. The
  large Q8_0 products give 52 to 58 GB/s alone. Thus the step is at about
  75% of the rate of the memory, and the floor is about 86 ms.
- These changes gave less than the noise, and are not kept:
  - one record for up to 4 products on the same x (one barrier, one
    loop over their rows): 640 product records became 350;
  - a prefetch of the row 2 rows ahead;
  - a dynamic schedule of the rows.
- The default of one thread for each core (np_gemma/__init__.py) is
  correct. Two threads on each core (36) are much slower when other
  programs use the CPU. The threads of the barriers spin, and a stopped
  thread holds up all of them.
- The small products are the slowest: hc_*_up (10240 rows of 320 values)
  gives 37 GB/s, and hc_*_down (320 rows) 44 GB/s. Together they are
  about 17 ms of the step. A kernel of several rows at a time can help
  them.
- The verify group reads the experts of 4 tokens (up to 40 for each
  layer). That is the most part of the cost of a round.
- The float32 matrices (the routers, the inject weights, the gates of the
  DeltaNet) took 0.93 s of a prompt of 512. They had a dot product for each
  token and row, on few rows. The matrices of 64 rows or more are now in groups
  of 16 rows (KQ_F32X16; bfloat16: KQ_BF16X16, for dense bf16), with a
  lane for each row and x as float32. The smaller ones take tasks of a row
  and 16 tokens. They now take 0.31 s (HANDOFF_QWEN38.md, section 8).

### Phase 5: the GPU

Status (a first form; np_gemma/qwen4_gpu.py, scripts/check_qwen4_gpu.py):

- Qwen4GPU is QwenGPU with the program of compile_qwen4_step. Methods of
  the model emit the products, the experts, and the attention
  (emit_lin4, emit_moe4, emit_attn_qsa). Qwen4GPU has its own methods.
- New kernels in csrc/gpu.cu:
  - HC_NORM, HC_ACT, HC_MIX, HC_ADD, PLE_GATE, PLE_CONV, and HC_CAT.
    Against the CPU records, the max rel error is below 1e-6.
  - QSA_SELECT: the raw keys, the key of each complete block, and one
    block of 1024 threads for each query. The top 512 blocks come from a
    radix select on the keys (score bits, then the block), with the tie
    rule of the CPU. For a context of 2600 rows, the rows are those of the
    CPU for each query.
  - ATTN_QSA: k_attn_part reads the rows of the selection. The 12 query
    heads of a key head go in 2 groups of 6 (the kernel takes at most 8).
  - Q5_1 in the products and the experts (the down matrices).
- Q8_R (the GPU form of Q8_0) pads each row to 16 bytes: rows of 320 or 640
  values were not aligned.
- PLE_CONV takes its count of rows from the slot nreal. The extra rows of
  a group (after the tokens) do not go into the state.
- The head (Q8_0) is a small program for each count of rows. The MTP layer
  runs on the CPU; its drafts use the head on the GPU.

Results (RTX 5060 Ti with 8 GB free; greedy decode after 100 tokens):

    hot experts     plain decode   MTP (3 drafts)
    1 GB (5 each)   18.3 tok/s     18.0 tok/s
    2 GB (10)       19.5 tok/s     19.5 tok/s
    3 GB (15)       20.9 tok/s     19.6 tok/s

    llama.cpp, the GPU split: 17.8 tok/s.

- The tokens are those of the CPU program: 48 of 48 on the chat prompt,
  and 40 of 40 after a prompt of 2936 tokens (QSA drops blocks there). A
  verify group and commit give the same bits as steps (with the hot
  experts fixed).
- A prompt of 2936 tokens in groups of 256: 66 tok/s (the CPU: 37).
- The step (2 GB hot): the CPU computes the cold experts, 38 of 64 ms
  (profile). The hot experts get only 19% of the selected experts. For
  each layer, 9 cold experts take 0.53 ms alone and 0.79 ms in the step.
- MTP gives no gain here. A verify group of 4 tokens sends up to 40
  experts to the CPU (about 115 ms against 50 ms for a step).
- With a 3090 (24 GB), about 17 GB of hot experts fit (about 115 in each
  layer). The CPU then gets far fewer experts, and the verify group costs
  less.

- The MTP layer on the GPU is a program of its own, with its cache on the
  GPU. Its experts are split, with its own slots (twice the slots of a
  layer, at most 0.5 GB). One draft: 2.5 ms, and 2.1 ms for the head.
- HotCache now also scores the tokens that a verify group keeps, and the
  rows of the MTP layer (HotCache.score_rows). Each group copies the
  selection of each layer to an array for that. Before, the hot experts
  did not change during an MTP decode.
- 300 tokens with 2 GB of hot experts: plain 19.8 tok/s. MTP: 20.1 tok/s
  with the MTP layer on the GPU, 19.9 tok/s with it on the CPU. The verify group
  (about 120 ms) reads about 5 GB of cold experts on the CPU. Thus on this
  machine MTP gives about the rate of the plain decode.
- The file builds for sm_86 (the 3090): the kernels have no
  griddepcontrol there, and the graphs keep ordinary edges (about 4%
  slower here).

- A large group of a prompt runs only if the two buffers of its copies
  fit (QwenGPU._fetch_fits). Else the prompt runs in split groups. Here
  they need 4.1 GB, and 3.2 GB is free even with 0.3 GB of hot experts.
  A model of 8 layers has room for the buffers. On it, a prompt of 1100
  tokens with a large group gives the logits of split groups within 6e-3.

Mixed groups of a prompt (np_gemma/qwen4_gpu.py, _moe_mix):

- A part of a prompt of at least 256 tokens runs in groups of 1024 rows.
  In each layer, after the router, the record MOE_PLAN (csrc/moe.c) on the
  host splits the experts. The GPU takes the experts with the most tokens:
  a worker copies them to one buffer (the free memory less 0.8 GB). The
  CPU takes the other experts at the same time, on a helper thread
  (CPU_START, CPU_WAIT). The hot experts stay on the GPU.
- The model of the costs: 0.9 ms for each copied expert (the copy and its
  work on the GPU). On the CPU: 75 us for each expert, and 15 us for each
  of its tokens. The split makes the two times about equal.
- The copies read the map of the file at 6.8 GB/s (PCIe Gen3 x8 here).
- On a model of 8 layers, 1100 tokens: 304 tok/s in split groups, 881
  tok/s in mixed groups.
- The whole model (0.5 GB of hot experts): pp512 156, pp2048 289, pp4096
  241 tok/s with random tokens (split groups: about 98). A document of
  2936 tokens: 147 tok/s (split groups: 91). The text uses more experts
  than random tokens: about 350 in each layer, 77 of them copied. The
  answer and 40 tokens of a summary are those of split groups and of the
  CPU program.
- The profile of a group of 1024 rows runs the records one at a time:

      the wait for the copies           0.91 s
      the dense products                0.77 s
      the experts on the GPU            0.71 s
      the attention (a record a query)  0.40 s

The products and the attention of large groups (more than 16 rows):

- The kernels k_kq_tc, k_qmoe_gu_tc, and k_qmoe_dn_tc (csrc/gpu.cu) use
  mma.sync m16n8k32 on int8 values (as k_gemm_q8). The rows of x are
  int8, with a scale for each 32 values.
- A block of 32 values of w gives int8 values, a scale d, and a term mn.
  The sum of the block is xs d (the int32 sum) + mn xsum. The formats:
  Q8_R, Q8_0, Q5_1, and Q4_K. The rest (F32, Q5_K, Q6_K) keeps the
  float32 tiles. NP_GEMMA_GPU_KQTC=0 keeps the float32 tiles for all.
- k_attn_qsa_mt: one record of ATTN_QSA for a large group. A block takes
  one query, one key head, and 6 query heads; 4 warps split the keys,
  with a softmax that runs. A group of at most 16 rows keeps one record
  for each query (the bits of the steps).
- A group of 1024 rows: 3.89 s before, 2.43 s now. The dense products
  went from 772 to 314 ms, the experts from 715 to 326 ms, the attention
  from 399 ms to a few ms.
- The cost of a copied expert in MOE_PLAN is now 0.7 ms.
- A document of 2936 tokens: 175 tok/s (before 147; split groups 91).
  pp512 226, pp2048 392, pp4096 381 tok/s with random tokens. On this
  GPU the buffer of the copies (about 100 experts) now limits the split.
- The prompt of Qwen3.6 uses the same products (its dense part is Q8_R).

The NVFP4 checkpoint of NVIDIA (np_gemma/st_qwen4.py,
scripts/convert_nvfp4_gguf.py; HANDOFF_QWEN38.md has the numbers):

- The experts are rows of type 51 (KQ_NV4). A row holds all the codes,
  then the E4M3 scales, then the float32 scale of the matrix. Zeros fill
  the row to a multiple of 16 bytes. The tensor cores load 4 codes with one
  aligned load, and change them to int8 with prmt and a sign mask.
- The tensor cores on NVFP4 agree with the float32 tiles to 1.5e-2 (8
  layers, the same top token). The experts of a group of 1024 rows take
  415 ms. The copies (1.75 s) and the CPU (1.41 s) set the time of the
  group.

The experts in groups of 16 rows (type 53, KQ_NVX). The GGUF now has this
type; --experts nv4 of the converter keeps type 51.

- A group holds 16 bytes (the scale of the matrix), then 288 bytes for
  each block of 32 columns. Step s (32 bytes) holds values 4s to 4s + 3 of
  the 16 rows. Byte 4r + u has row r in its low 4 bits and row r + 8 in its
  high 4 bits. Then come the 32 E4M3 scales of the 16 rows.
- The CPU (kq_nvx_rows): a step is one vpdpbusd, with a lane for each row.
  The codes go to the E2M1 value times 2, plus 12 (an unsigned byte). The
  sum starts at -12 times the sum of x of each 16 values. A token alone and
  in a group gives the same bits.
- The GPU: the 4 bytes at 32 s + 4 r are the tensor-core fragments of rows
  r and r + 8. A step of a tile is 8 contiguous groups (cp.async of 16
  bytes). The step kernel of the hot experts uses a warp for each group.
- The results equal those of type 51 (the GPU against the CPU: the same
  errors in all the paths).
- One layer of experts on the CPU (512 experts, 10 for each token): 512
  tokens take 33 ms in place of 75 ms. On the GPU (16 experts): 1024
  tokens on the tensor cores take 6.2 ms in place of 8.9 ms. The step
  kernel is 2.3 to 3 times faster.

## MTP and an optimization pass on the 2-socket Xeon and the 3090

The GGUF of this runtime, made again on this machine with
scripts/convert_nvfp4_gguf.py (--dense bf16: the dense matrices as they
are; 23 minutes from the checkpoint on NFS), runs with NP_GEMMA_DENSE=bf16
(no Q8_0 requantization at the load) and the hot experts of the free
memory (88 in each layer). Greedy decode of 256 tokens of a chat answer,
2.0 GHz:

    start                     plain 20.2 tok/s   MTP (3 drafts, layer on the GPU) 20.7
    bf16 small groups         MTP 26.2 (verify of 4: 95 -> 68 ms)
    head, scores, argmax      MTP 28.2
    CPU task team, rows       plain 27.7         MTP 29.3 (1 draft: 30.4)
    bench_qwen4 tg128         17.4 -> 27.5 tok/s; pp2048 342 -> 347

The changes:

- The bfloat16 products of a verify group (k_kq_linear for each row looped
  over the tokens: 4 tokens took 1.8 times one) read each chunk of 8
  weights once for all the tokens (kq_row_bf16_nt; KQ_LINEAR, the split of
  long rows, KQ_MULTI with the float32 alpha and beta of a DeltaNet layer).
  Each token keeps the operations of one token: a verify group still has
  the bits of the steps (check_qwen4_gpu: max rel 0).
- The logits of the head go to pinned memory (1 MB a row: 2.0 -> 1.6 ms for
  a row, 4.4 -> 3.3 for 4), and greedy MTP takes the best token of each
  row from the GPU (Qwen4GPU.argmax, GP_ARGMAX): no copy of the logits.
- HotCache.score_group scores the tokens of a group at once, on the rows
  of its layers only (the same scores as one call for each token; one pass
  of copies). The scores took about 13 ms of Python in a round, now 6.7.
- The CPU programs of a GPU program run in a team of half the cores bound
  spread (gemma_run_task, NP_GEMMA_GPU_CPU_THREADS): 48 threads bound close
  gave 21.3 tok/s, 24 spread 26.0. kq_rows and kq_gather run a few rows on
  the calling thread: their teams (64 threads for the n-gram rows) spun on
  the cores of the cold experts (MTP drafts 30 -> 20 ms a round).
- serve_qwen4.py runs MTP by default (--mtp 1 with the layer; 0 turns it
  off): the server (3 hot experts in each layer, --hot-gb 0.5) decoded
  23.6-25.2 tok/s without MTP and 28.8 with it (88% of the drafts).

What was tried and left: the cold experts of each node on its own CPUs
(the pages of expert e on node e % 2, the threads of each node on its
experts, in one team or in nested teams): no gain once the threads are
spread (0.232 against 0.234 ms for 5 experts; nested teams took 1 to 3.5
ms for each call). One node reads about 55 GB/s of these experts and both
about 100, but the kernel of a step is not bound by that. 109 hot experts
in each layer in place of 88 gave 28.4 in place of 27.7 tok/s, and the
programs of MTP then ran out of GPU memory.

What is left: a verify of 4 tokens (77 ms) still costs about two steps (36
ms): the CPU part (the cold experts of 4 tokens, about 34 ms) is half of it,
so 1 draft is the best (30.4 tok/s; 2 drafts 29.6, 3 drafts 29.0).

## The experts: the copies of HotCache, the CPU threads, and zero copy

A trace of a decode step (nsys, Q8_0 dense, 119 hot experts in each layer):
the GPU waited for the CPU part 15.7 ms of 34.8 (k_await), the dense
products took 8 ms, the hot experts 4.8 ms (k_kqh_gu 72 us for 12 MB: a warp
for each group of 16 rows, 166 GB/s), and about 1300 small kernels 4.5 ms.

- k_kqh_nvx: a block of 8 warps (4 for down) for each group of 16 rows,
  the warps over the blocks of columns: the hot experts 4.8 -> 1.7 ms a
  step. The CPU part is the longer one in each layer, so the step gained
  only 3% (30.9 -> 31.8 tok/s).
- The copies of HotCache came from the pageable map of the file
  (cudaMemcpyAsync: the driver stages them with a memcpy and holds itself).
  With the dynamic slots off, the decode went from 32 to 36 tok/s and MTP
  to 46. The copies now go through two pinned buffers of each worker
  (gg_h2d_staged), and the workers stay on the last CPU of the node of the
  GPU (gg_worker_bind), which the CPU team (bound spread) leaves free.
- Zero copy (NP_GEMMA_GPU_ZC): the GPU computes some cold experts from a
  pinned copy of all the experts in host memory (68 GB, interleaved over
  the nodes, 95 s to make), while the CPU computes the others. A kernel reads
  host memory at 11-12 GB/s (PCIe 3.0 x16), so an expert takes 0.25 ms.
  Before the fix of the copies it gave 31.8 -> 33.9 tok/s; after it, 44.6 ->
  33.7: the CPU share is now shorter than one expert over PCIe. With the
  verify groups too (ZC_MT 1): MTP 58.7 -> 47.8. It stays as an option.

The GPU is on node 0 (CPUs 0-23, PCIe 3.0 x16). Decode of 256 tokens of a
chat answer (greedy) and bench_qwen4, 2.0 GHz:

    dense   hot     plain    MTP 1 draft   MTP 3 drafts   pp512   pp2048   tg128
    q8      119     44.6     58.7          60.6           411     618      42.1
    bf16     88     36.8     49.8          50.4           306     426      35.9

    before (q8): plain 30.6, MTP 33.2; pp512 267, pp2048 472, tg128 30.1
    the 5060 Ti (q8, 0.5 GB hot): pp512 327, pp2048 496, tg128 22.8

## 2.4 GHz, kernel fusion, and the CPUs

The clocks at 2.4 GHz (core and uncore), Q8_0 dense: plain 44.6 -> 47.0
tok/s, MTP (3 drafts) 60.6 -> 67.0, pp512 411 -> 440, pp2048 618 -> 655,
tg128 42.1 -> 44.4. bf16: plain 36.8 -> 38.8, MTP 50.4 -> 51.6, pp512 306 ->
332, pp2048 426 -> 446, tg128 35.9 -> 38.5.

Kernel fusion (NP_GEMMA_GPU_HC_FUSE, NP_GEMMA_GPU_TOPK_SPLIT): the add of
the hyper connections and the norm that follows (96 a step), the int8 x of
the products written by HC_NORM, HC_ACT, and HC_MIX (the GP_KQ_QUANT records
between no longer clear it), and the hot split in the kernel of the router.
The values are those of the separate kernels (check_qwen4_gpu: a verify
group equals the steps). Plain decode q8 45.4 -> 46.7, bf16 38.3 -> 39.6;
MTP 65.6 -> 66.8 and 51.7 -> 52.7. The kernels of the hot experts run while
the GPU waits for the CPU, so they are not on the path of a step.

The CPUs: the uncore counters over 4 s of decode (q8) give about 10 GB/s of
DRAM reads on each socket and 5 GB/s on each UPI direction, with 12, 24, or
44 threads (41.5, 46.0, 38.5 tok/s). The CPU part of a layer is about a
third of a step, so it reads about 30 GB/s of each socket while it runs:
not the limit (one socket gives 55-60). A copy of the experts on each node
(NP_GEMMA_GPU_NUMA_COPY) gave 46.3 against 47.0: the barriers and the start
of each team call bound the CPU part, not the memory.

## The sync of the CPU part

The phases of KQ_MOE for one layer alone (7 cold experts, thread 0): sort
5.1 us, the copy of the rows of the pairs 3.4, gate and up 125.5, act 8.7,
down 70.1, sum 4.2, and KQ_QUANT before it 14: about 35 us of barriers and
single work in 240. kq_moe_small_body (t <= 4, act bit 3) does the int8 x,
the sort, and the copy in one single, the act in the thread that ends the
last task of an expert, and the down tasks wait for their expert, not for
a barrier: 7 barriers become 3, with the same bits. In the decode it gave
about 1% (plain 46.8 -> 47.0, MTP 66.2 -> 67.0, within the noise), and a
longer spin of the OpenMP threads (GOMP_SPINCOUNT) nothing.

In the decode a layer's CPU program takes 208 us for 4.5 cold experts (12.5
MB): 10 ms of a step of 24.7 ms (gg_task_stats). The gate and up tasks of 7
experts alone ran at about 100 GB/s of both sockets, so the rest is the
start of many short streams (a task is 23 KB) and the fixed cost of a
region, not barriers. Fewer cold bytes (the hot experts on the GPU) or
work for the CPUs while the GPU runs (about 12 ms of a step) are what is
left.

## Optane, a copy of the experts on each node, and 256K tokens

The two Optane modules (126 GiB each, on socket 1) are one App Direct region
of 252 GiB, interleaved, with ext4 at /mnt/pmem (dax=always). Node 0 has 6
channels of DRAM (192 GB) and node 1 129 GB.

- A thread of socket 1 reads the module at 10-13 GB/s, a thread of socket 0
  at 0.4 GB/s (a remote read updates the directory in the module). The
  GGUF reader stages a file of a DAX mount on the node of the module with
  threads of that node (gguf._stage_dax, 81 GB in 8.5 s in huge pages),
  keeping the 51 GB n-gram table on the module, and QwenGPU copies the
  experts from it to node 0 (70.7 GB in 3.3-4.5 s). The load: 36 s (the
  plain map of the file on Optane: 127 s).
- The decode (q8 dense, 48 threads, 2.4 GHz): a copy on each node 52.7
  tok/s plain and 72.4 with MTP (3 drafts); the staged copy on node 1 alone
  32.2; the file on NVMe through the page cache 33.3 (36.9 MTP).
- The share of each node in a step (kq_calib_nodes) moved between 0.5 and
  0.62 from run to run with no measurable gain (52.7 against 52.5).
- Note: a shell pinned to the CPUs of node 0 (taskset) gives the runtime 24
  threads and a CPU team on node 0 only.

The cache: 20.6 KB a token (int8 keys and values of the full attention
layers, the keys of the indexer) and 118 MB of state. QwenGPU(ctx=) leaves
room for it before the hot experts. bf16 dense with 262144 tokens (the
largest position of the model): 42 hot experts in each layer; a prompt of
259999 tokens in 892 s (291 tok/s; 313 at the start, 273 at 220K), then 13.1
tok/s of decode at that depth.

## The prompt in bf16

256K tokens: bf16 dense 291 tok/s (42 hot experts), q8 347 tok/s (73), and
13.1 and 13.7 tok/s of decode at that depth.

A trace of a prompt of 8192 tokens (bf16): 20 s, the GPU busy 60%. The
products of the bf16 matrices ran in k_kq_gemm (float32, 4.6 s), the
experts of the GPU in k_qmoe_gu and k_qmoe_dn (3.1 s: the tensor-core
kernels take a shared expert of Q8_R only), and the sort of the pairs on
one thread 1.0 s; the DeltaNet (1.3 s) and the attention (1.2 s) are those
of q8. k_gemm_bf16_tc (x rounded to bfloat16, 52-58 TFLOPS against 13) and
k_qmoe_sort_par: 406 -> 503 tok/s. The accuracy on 8192 tokens of source
(the prompt path, every token): NLL 0.9208 with float32 products, 0.9212
with the tensor cores, 0.9205 with q8 dense; top-1 against float32 97.1%
(q8 96.2%), KL 0.013 (q8 0.019).

One bfloat16 plane keeps 8 bits of x. Two planes (hi = bf16(x), lo =
bf16(x - hi), two products into the same float32 sums) keep about 16: the
error of a product 1e-5 in place of 1.6e-3, at half the rate (25 against 53
TFLOPS; the 3090 gives about 71 TFLOPS of bf16 with float32 sums, so two
planes cannot pass about 35). The default is two planes: 469 tok/s for
8192 tokens (one plane 504). On the text, KL to float32 0.0104 (one plane
0.0128); two runs of float32 products differ by 0.0066 (the split of the
experts between the CPU and the GPU, and their int8 x), so the products
are now a small part of the difference.

## bf16: the shared expert, the copies of a prompt, tq6

- The shared expert of a group MoE in bf16: the routed experts stay on the
  int8 tensor cores (k_qmoe_gu_tc, k_qmoe_dn_tc skip the tiles of the
  shared expert), and gg_gemm_bf16 computes the shared expert on the rows
  start[E] .. start[E] + t of act, act2, de (start[E] read on the device).
  8K prompt 469 -> 595 tok/s; KL to float32 unchanged (0.0129 with tq6).
- The copies of a mixed group (GP_FETCH) waited about 0.9 s of each group
  of 2048 tokens: the experts staged from Optane were copied by the worker
  through its pinned buffers. QwenGPU page-locks the node-0 copy of the
  experts (70.7 GB in 1.9 s), and the fetch worker copies registered memory
  by DMA: 564 -> 649 tok/s. HotCache stays staged (full-rate DMA in the
  decode took MTP from 66.7 to 58.8 tok/s).
- tq6 is the cache of NP_GEMMA_DENSE=bf16 (18.7 KB a token, 21.8 for
  int8): KL 0.0130 against 0.0104 for int8, the prompt about 10% slower
  (more CPU time between the kernels, the kernels the same).

    bf16 (tq6)    8K prompt 649 tok/s, decode 42.3, MTP 57.6
    q8 (int8)     8K prompt 715 tok/s, decode 51.2, MTP 70.5

The GPU still waits about 5.7 s of a bf16 prompt of 8192 tokens, in two
places of each layer of each group: from the router to the sort (about 15
ms: the plan on the CPU, the copies, and their wait) and from the sum of
the experts to the add (about 12 ms: the experts of the CPU). The cost of a
copy in the plan (MIX_GPU 0.7 ms) was too low on this machine: a sweep gave
618 (0.7 ms), 667 (1.2 ms), 639 (1.8), 584 (4.0) tok/s, and the calibration
of the cost (NP_GEMMA_GPU_MIX_CAL, now on by default) 680. With it: bf16
676 tok/s, q8 768 tok/s.

## bf16: the work during the copies, the output of the CPU

- A mixed group now computes the hot experts and the shared expert during
  the copies of the cold experts that it moves to the GPU
  (NP_GEMMA_GPU_MIX_SPLIT, on by default). The plan (GP_MOE_PLAN) gives two
  lists of pairs: gidx for the hot experts, gidx2 for the copied ones. The
  first KQ_GROUP_MOE runs before GP_FETCH_WAIT; the second has no shared
  expert (the sort gives it no rows, k_qmoe_sum adds only the routed
  pairs) and runs after it; an add joins the two parts. Alone it gave
  little: bf16 681 -> 668, q8 766 -> 783.
- A trace then showed the real wait: the sum of the CPU experts went to the
  GPU from pageable memory, at 1.8 GB/s (2048 x 4096 x 4 bytes, about 18
  ms in each layer of each group). host_out of the mixed groups is now
  pinned: bf16 741 to 757 tok/s for 8192 tokens, 761 for 16384; q8 855.
- The copy cost of the plan: the calibration still wins (757) against
  fixed costs of 1.2 ms (734), 2 ms (693), 3 ms (655), 5 ms (610).
- KL to float32 0.0156 (NLL 0.9169 against 0.9208 for float32); two runs
  of tq6 differ by 0.012, so the change is within the noise of the split
  of the experts. check_qwen4_gpu.py passes in bf16.

    bf16 (tq6)    8K prompt 757 tok/s, decode 42.3, MTP 57.6
    q8 (int8)     8K prompt 855 tok/s, decode 51.2, MTP 70.5

The trace of the bf16 prompt of 8192 tokens (GPU busy 68%):

    k_gemm_bf16_tc     2.38 s   the dense products and the shared expert
    k_attn_qsa_mt      1.29 s   the attention of the full layers
    k_gdn_heads        1.25 s   the DeltaNet heads
    k_qmoe_gu_tc       0.55 s
    k_kq_gemm          0.44 s   float32 products of the small matrices
    k_qmoe_dn_tc       0.34 s
    k_gdn_conv         0.18 s
    k_to_bf16          0.15 s

    the wait for the copied experts after the hot ones    9.0 ms, 1.74 s
    from the router to the plan                           3.1 ms, 0.6 s

## The DeltaNet of a prompt group in parallel

k_gdn_heads ran one block for each value head (48 blocks of 128 threads)
with three block sums for each token: 8.8 ms for a group of 2048 tokens,
1.25 s of the bf16 prompt of 8192 tokens, and k_gdn_conv (one thread for
each channel, the tokens in a loop) 0.18 s. Only the state needs the order
of the tokens, and each column of it is on its own. A group of 64 tokens or
more with no log (NP_GEMMA_GPU_GDN_PAR=4) now runs:

    k_gdn_conv_par   each (token, channel)                 0.52 ms
    k_gdn_prep       norms of q, k; decay, beta; conv state
    k_gdn_scan       4 threads for each column, 192 blocks  1.55 ms
    k_gdn_out        the norm of the output and the gate    0.24 ms

The 8K prompt: 757 -> 837 to 855 tok/s; KL to float32 0.0150 (0.0156
before). The verify groups keep k_gdn_heads (their log and commit need the
bits of the steps). check_qwen_gpu_groups.py (Qwen3.6) passes.

## The attention of the QSA layers on the tensor cores

k_attn_qsa_mt ran the dot of each query head with each key in float32: a
block for each query and half of its heads (R = 6), so each key was decoded
twice for each query, 26 ms for a group of 2048 tokens, 1.26 s of the bf16
prompt of 8192 tokens. k_attn_qsa_tc (NP_GEMMA_GPU_QSA_TC, on by default)
puts the 12 query heads of a key head in the 16 rows of an mma.m16n8k16
tile. Block (query, key head), 4 warps, each with its own keys (16 at a
time, the rows of sel), decoded to float16 times their scale in its own
shared memory; S = Q K^T with q as two float16 planes (hi + lo, about 22
bits; 2 keeps one plane), a softmax that runs, O += P V (P float16, V by
ldmatrix.trans), and the warps join at the end. Forms int8 (q8) and TQ6
(bf16); the float32 first layer and the int16 forms keep k_attn_qsa_mt.

- 8K bf16 prompt 860 -> 910 tok/s (one plane 889); the kernel 26 -> 17
  ms a group.
- KL to float32 0.0120 (0.0150 before, two float32 runs differ by 0.0066);
  NLL 0.9193 (float32 0.9208).
- Alone (the scratchpad bench qsa_bench.py: 2048 queries at position 6144,
  2048 random keys each): TQ6 31.9 -> 21.1 ms, int8 28.7 -> 19.4 ms; the
  outputs agree to 1e-3. So the decode of TQ6 is not the limit.
- Still about 1.5 times, not the 4 to 5 that the count of instructions
  gives. Not found yet: the counters of the GPU need root
  (ERR_NVGPUCTRPERM; NVreg_RestrictProfilingToAdminUsers=0 or ncu with
  sudo). The codebook of TQ6 in shared memory changed little (18.3 -> 17.0
  ms). The kernel has 230 registers (two blocks of 4 warps on an SM) and
  each warp runs its keys in series: decode (gathered rows), S, softmax,
  decode V, P V. Likely the latency of the gathers with 8 warps on an SM.
  Next: a profile (stall reasons, occupancy); then the keys of a step
  decoded by all the warps into one tile (cp.async of the raw rows of the
  next step during the products), the output split over the warps (fewer
  registers, more blocks on an SM).

## Decode at the reduced clocks: the CPU part, warm slots

The setup of the server: RQ8_0 experts, 256K context, the image encoder on
the GPU, MTP 1 draft; the CPUs at 2.0 GHz (turbo off; uncore 2.0, then 2.4),
the GPU at a 200 W limit. 10 hot experts in each layer.

- A step: 48 layers of (the GPU before the MoE, 322 us: attention or
  DeltaNet, the dense products, the router) then (the cold experts on the
  CPU, 428 us). About 332 of the 490 selections of a step are cold, about
  36 MB of RQ8_0 in each layer, read at about 100 GB/s.
- Inside the CPU part (NP_GEMMA_MOE_PROF, kq_moe_prof): the threads arrive
  within 2 us, the single region 11 us, gate and up end at 208 us (mean;
  max 241), down at 373 (max 392), all 414. The team is busy and even: the
  low use of the CPUs (about 20%) is the duty cycle (24 of 48 cores, busy
  57% of a step), not idle threads in the part.
- The team: 24 threads 25.7 tok/s, 32 25.0, 40 or 44 17 (they share cores
  with the threads that spin: the runner, the copy worker, Python), 48 and
  more of node 0 worse. Uncore 2.4 GHz in place of 2.0: no change in the
  noise (24.4 to 25.8).
- Warm slots (HotCache.enable_warm): more slots for the decode in a buffer
  that the GPU holds for something else. A slot value KQH_WARM + i is
  expert i there (gg_warm_base, read at run time by the hot kernels;
  desc[16..18] for the plan of a mixed group). serve_qwen4 --mmproj-gpu
  lend: the encoder measures its room (1.57 GB), gives it to the model
  between images (QwenGPU.lend_warm), and takes it back for each image.
  The rows are laid out by their own expert sizes (the MTP layer of the
  NVFP4 file has larger experts). NP_GEMMA_GPU_WARM=ring also uses the
  buffer of the copies of the mixed groups (cleared by each mixed group and
  by each new program).
- With the lent room: 6 warm slots a layer (284 filled), cold 332 -> 301 a
  step, the CPU part 428 -> 390 us; plain 25.8 -> 27.0 tok/s, MTP (1
  draft) 27.0 -> 27.8; the 8K prompt 590 -> 595 tok/s; the warm experts
  stay through the prompt. check_qwen4_gpu.py PASS with lent slots (the
  verify group exact); with ring the verify group differs (6.8e-3: a new
  program frees the ring and the warm experts go to the CPU).

## The CPU part of a prompt on real text

- numad, before it was turned off, had moved the process of the agent (and
  so the shell of each benchmark it started) to node 1: an affinity of
  CPUs 24-47 that stayed. The earlier runs without taskset ran on
  24 cores (more threads did not help; 32 threads were slower than 24).
  taskset -c 0-47 (or taskset -a -p on the parent) fixes it. The decode
  numbers of those runs need a run again.
- A prompt of random tokens spreads few tokens on many experts less than
  real text does. On an 8K prompt of real text (the notes and sources of
  this project) the CPU part was the wait of each mixed group: about 290
  experts a layer on the CPU, 4100 to 4800 pairs, median 8 tokens an expert
  (36 experts of 1 token). 216-270 tok/s pinned to node 1; 347 (first
  prompt) to 504 on both nodes.
- The microbenchmark: the CPU inputs of the last layer of 4 groups
  (cap_mix in the scratchpad), kq_moe on the weights of that layer.
  Gate and up 38 ms, act 3 ms, down 21 ms, sum 2.5 ms of a 64 ms layer
  (NP_GEMMA_MOE_GPROF). perf: kq_prep 13%, kq_row_scales 7%, a stack
  probe of the 66 KB S of kq_tile4, the sign steps of Q8_0.
- kq_rows4_t2 (NP_GEMMA_KQ_TILE2): the scales of a step of 64 values in
  the lanes of the products, made one time for the 4 rows (dE) and in the
  loop for the token (2 broadcasts), multiplied in the loop: S = ds * xs as
  kq_prep, so the same bits. Q8_0 takes x + 128 (xor 0x80) from -128 sum(w)
  of each lane (nE): the integer sums of kq_dot_q8_0 with no sign steps.
  Short tiles have their own code; the next 4 rows are prefetched. The
  same bits as kq_tile4 (the whole layer, Q8_0 and the mix).
- One thread, weights in cache: about 17 MAC a cycle at 64 tokens; ports 0
  and 5 are 68% busy. The exact form needs 4 vector uops for each 64 MACs
  (dpbusd, cvt, mul, fma), so little is left there. Scales made in the
  loop in place of the arrays of the rows (less L1) were slower.
- A layer, both nodes: 24 threads 59.4 -> 45.3 ms; 40 threads 43.6 ->
  38.7; with the copy on node 1 (330 experts) 31.8 (mix: 60.5 -> 35.1).
  The CPU part of a mixed group now has its own team
  (NP_GEMMA_GPU_MIX_THREADS, 40) and reads the copy on node 1
  (NP_GEMMA_GPU_MIX_NUMA).
- 8K real-text prompt, RQ8_0: 347 -> 455 tok/s for the first prompt, 504 ->
  561-570 after the calibration settles. 4K with the calibration off: 408
  -> 571, the same logits bit for bit. The copies are now the larger wait
  (1.8-2.3 s of 3.5 s a group). check_qwen4_gpu.py: PASS.
- The runs again on 40 CPUs (taskset 0-19,24-43, OMP_NUM_THREADS=40; so
  the team of a mixed group is 32), lent image room, ctx 256K, tok/s:

  |                         | RQ8_0  | RQ6 mix |
  |-------------------------|--------|---------|
  | hot experts a layer     | 22     | 30      |
  | plain, team 24 / 32 / 40| 30.3 / 30.3 / 31.8 | 32.2 / 31.6 / 32.9 |
  | MTP 1, team 24 / 32 / 40| 34.4 / 35.9 / 37.4 | 38.5 / 39.0 / 41.4 |
  | CPU task a layer, 40    | 291 us | 274 us  |
  | 8K real text, 3 runs    | 519, 596, 605 | 521, 575, 581 |

  The team of 40 is the best for the step too (the default of
  NP_GEMMA_GPU_CPU_THREADS is OMP_NUM_THREADS / 2: set it). Pinned to node
  1 the same day: RQ8_0 27.8 / 29.6, mix 29.3 / 35.2 (plain / MTP).

## The calibration with prefetch

8K real text, RQ8_0, 256K ctx, 40 CPUs, tok/s of the 2nd and 3rd runs with
the calibration off (NP_GEMMA_GPU_MIX_CAL=0) at a fixed copy cost (gpu_c)
and a cost of a prefetched copy for the size of the prediction
(NP_GEMMA_GPU_PREFETCH_COST, a share of gpu_c):

| gpu_c   | cost 1.0 | 0.6  | 0.35 | 1.5  |
|---------|----------|------|------|------|
| 0.45 ms | 534-539  | 506  | 460  |      |
| 0.65 ms | 596-600  | 562  | 511  |      |
| 0.9 ms  | 597-599  | 587-597 | 566 |     |
| 1.2 ms  | 581-589  |      |      | 583  |
| 1.6 ms  | 558-561  |      |      | 536  |

With the calibration on (it settles near 0.9 ms): 601-618 (two runs of the
same settings; the runs differ by about 2%), cost 1.5 582-600, 2.0 594-598.
A larger prediction is slower: it takes copies and GPU work from experts the
CPU does as well. The calibration (the least waits) is as good as the best
fixed cost, so it stays. The first prompt of a process (about 480 tok/s) is
the compile of the program of the mixed groups and a first group with no
prediction, not the start of the calibration (starting at 0.9 ms changed
nothing).

## The memory of the cache

The cache of 131072 positions (NP_GEMMA_QWEN_KV=tq6, the first full layer
float32; QWEN_PLAN.md has the forms):

    part                                         before    now
    keys and values, 11 tq6 layers               1.20 GiB  1.20 GiB
    the raw keys of the indexer (12 layers)      0.75 GiB  0.38 GiB
    the first full layer (float32)               0.50 GiB  0.50 GiB
    the keys of the blocks of the indexer        0.19 GiB  0.09 GiB
    the MTP layer (tq6)                          0.11 GiB  0.11 GiB
    the state of the DeltaNet layers (fixed)     0.10 GiB  0.10 GiB

- The keys of the indexer are now float16 (idxk and blk of QSA_SELECT, on
  the CPU, the GPU, and in Qwen4.qsa_mask). The sums use float32. The
  scores only rank the blocks.
- check_qwen4_indexer.py ran on 4000 random rows. The 1949 queries that
  drop blocks keep 997888 blocks, and 105 of them change (one block for
  each of 105 queries). Each
  has a score within 0.17% of the score of the last kept block. The float32
  keys give no change. The check now passes such a block (NEAR).
- The raw keys must stay. A snapshot of the server can start in a block,
  and the key of that block then needs the raw keys before the snapshot.

## Risks

- The page cache: if the working set does not fit, the rate falls by a
  large factor. Phase 1 measures it before the other work.
- The indexer of QSA changes the result only after 2048 tokens. A check of
  a long context against transformers is slow (its reference is a loop in
  Python). Use llama.cpp as the second reference there.
- The branch of llama.cpp is new, and its MTP and its values can have
  errors. transformers is the reference for the main model.
- The disk: about 29 GB free on /home after the download.
