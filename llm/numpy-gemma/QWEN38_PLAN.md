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
- Next: the QSA indexer (for more than 2048 tokens), the resident set of
  the memory map, and a check against transformers.

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

- The drafter: the MTP layer on the hidden state of the 4 streams and the
  row of the token. It keeps its own cache of keys and values.
- The verify group and commit of QWEN_PLAN.md phase 3 (the log of the
  DeltaNet) are ready. Add the state of PLE (the last tokens and the
  convolution) to the log.
- Measure the accepted drafts on real answers, and the rate for 1 to 5
  drafts (llama.cpp: --spec-draft-n-max 5).

### Phase 4: the rate of the CPU

- The profile of the step (gemma_profile), and a check of the resident set.
- The experts of 640 values: the tiles and the dynamic schedule of the
  Qwen3.6 work.

### Phase 5: the GPU

- The dense part, the attention, the DeltaNet, the gated residual, and the
  head go on the GPU. The experts are split with HotCache
  (np_gemma/qwen_gpu.py).
- With 7.5 GB free: about 3.5 GB for the dense part and the head, and the
  rest for hot experts (about 2.9 MB each).
- The drafter on the GPU (its dense part), with its experts split too.

## Risks

- The page cache: if the working set does not fit, the rate falls by a
  large factor. Phase 1 measures it before the other work.
- The indexer of QSA changes the result only after 2048 tokens. A check of
  a long context against transformers is slow (its reference is a loop in
  Python). Use llama.cpp as the second reference there.
- The branch of llama.cpp is new, and its MTP and its values can have
  errors. transformers is the reference for the main model.
- The disk: about 29 GB free on /home after the download.
