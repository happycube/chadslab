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

## Risks

- The page cache: if the working set does not fit, the rate falls by a
  large factor. Phase 1 measures it before the other work.
- The indexer of QSA changes the result only after 2048 tokens. A check of
  a long context against transformers is slow (its reference is a loop in
  Python). Use llama.cpp as the second reference there.
- The branch of llama.cpp is new, and its MTP and its values can have
  errors. transformers is the reference for the main model.
- The disk: about 29 GB free on /home after the download.
