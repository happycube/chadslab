# Plan: Gemma 4 26B-A4B (MoE) support

## Goal

Add support for the model google/gemma-4-26B-A4B-it-qat-q4_0-unquantized.
This model is a mixture of experts (MoE). The runtime must run it with NumPy
only. The plan keeps the present design:

* the weight modes f32, bf16, int8, and int4
* the C kernel and the Numba fallback
* the weight cache in ~/.cache/np_gemma/weights
* the resident session

The text decoder is in scope. The vision and audio parts are out of scope. The
12B runtime already ignores the image and audio parts.

## Model facts

The table gives the values from config.json and from the GGUF metadata.

    item                        value
    parameters                  26.5 B total, about 3.8 B for each token
    checkpoint size             51.6 GB in bfloat16, two files
    layers                      30
    hidden size                 2816
    attention heads             16
    key/value heads             8 for a sliding layer, 2 for a global layer
    head dimension              256 for a sliding layer, 512 for a global layer
    global layers               index 5, 11, 17, 23, 29 (five layers)
    sliding window              1024
    rms norm epsilon            1e-6
    dense MLP inner size        2112
    experts                     128
    experts for each token      8
    expert inner size           704
    shared experts              1 (the dense MLP)
    vocabulary                  262144, tied to the output
    final logit softcap         30
    rope, sliding layer         default, theta 1e4, all angles
    rope, global layer          proportional, theta 1e6, 25 percent of the angles

The MoE block is *additive parallel*. The output of the experts is added to the
output of the dense MLP. The block does not replace the dense MLP.

## Layer structure

The structure below is the exact order from the Hugging Face reference. The
four points in bold are traps.

    residual = x
    h = input_layernorm(x)
    h = self_attn(h)
    h = post_attention_layernorm(h)
    x = residual + h

    residual = x
    h = pre_feedforward_layernorm(x)
    dense = mlp(h)

    # MoE block
    h1 = post_feedforward_layernorm_1(dense)
    # TRAP 1: the router reads the residual, not the dense output.
    scores = router(residual)
    # TRAP 2: the experts also read the residual, through a second norm.
    h2 = pre_feedforward_layernorm_2(residual)
    h2 = experts(h2, scores)
    h2 = post_feedforward_layernorm_2(h2)
    h = h1 + h2

    h = post_feedforward_layernorm(h)
    x = residual + h
    x = x * layer_scalar

The router has five steps:

    r = rms_norm(x, weight=None, eps)      # TRAP 3: no learnable weight
    r = r * router.scale * (hidden ** -0.5)
    logits = r @ router.proj.T
    probs = softmax(logits.astype(float32))  # TRAP 4: softmax in float32
    w, idx = topk(probs, 8)
    w = w / w.sum()
    w = w * router.per_expert_scale[idx]

The experts compute a gated MLP with the GELU tanh activation. The gate and the
up projection are one tensor. The code splits the result in the middle.

    for each selected expert e with weight w_e:
        gu = h2 @ gate_up[e].T          # (T, 2 * inner)
        gate, up = split(gu)
        out = gelu_tanh(gate) * up
        out = out @ down[e].T           # (T, hidden)
        result += out * w_e

## Tensor names

The text prefix is model.language_model. The plan uses the same prefix as the
12B model. Each layer holds these tensors:

    router.per_expert_scale     (128,)        float32
    router.proj.weight          (128, 2816)   float32
    router.scale                (2816,)       float32
    experts.gate_up_proj        (128, 1408, 2816)  bfloat16
    experts.down_proj           (128, 2816, 704)   bfloat16
    mlp.gate_proj.weight        (2112, 2816)  shared MLP
    mlp.up_proj.weight          (2112, 2816)
    mlp.down_proj.weight        (2816, 2112)
    pre_feedforward_layernorm_2.weight
    post_feedforward_layernorm_1.weight
    post_feedforward_layernorm_2.weight

The prefix is model.language_model. The final norm is model.language_model.norm.weight.
The embedding table is model.language_model.embed_tokens.weight.
The file also holds model.embed_vision.* tensors. Do not load them.

## Size and speed budget

The expert tensors are much larger than the other tensors. The count below is
for the int8 mode.

    part                        parameters    int8 size
    gate_up experts             15.23 B       15.23 GB
    down experts                 7.61 B        7.61 GB
    shared MLP                   0.54 B        0.54 GB
    attention                    1.11 B        1.11 GB
    embedding table              0.74 B        0.74 GB
    total                       25.2 B        25.2 GB

The bytes read for each token are much smaller, because the model uses only
eight experts.

    part                        gigabytes for each token
    eight experts               1.43
    shared MLP                  0.54
    attention                   1.11
    embedding table             0.74
    total                       3.82

At 43 GB/s the ceiling is about 11 tokens for each second. The dense 12B model
reads 11.92 GB for each token, so its ceiling is 3.6 tokens for each second.
The MoE model is about three times faster for each token, but uses about two
times more memory.

The int4 mode reads 12.6 GB of weights. It halfes the memory. It does not
change the count of active bytes for each token much, because the eight experts
and the shared MLP are already the small part.

## Work phases

## Weight source: the QAT q4_0 GGUF

The plan uses the GGUF file, not the 51.6 GB bfloat16 checkpoint. The file is
google/gemma-4-26B-A4B-it-qat-q4_0-gguf, path gemma-4-26B_q4_0-it.gguf. The
size is 14.44 GB. Thus the disk use is small.

The GGUF file holds only three data types:

    part                        type
    every projection, expert    Q4_0
    the embedding table         Q6_K
    norms, router, scales       F32

The Q4_0 type matches the int4 layout of this runtime. One Q4_0 byte holds
value j in the low nibble and value j+16 in the high nibble. The runtime uses
the same order. The Q4_0 nibble is the two's complement nibble with bit 3
flipped. Thus one XOR of 0x88 on each byte gives the runtime nibble. The Q4_0
group scale (one fp16 value for each group of 32) is then the runtime scale.
This step keeps the quantization of the model. It does not quantize again.

The module np_gemma/gguf.py gives the reader. It maps each GGUF name to the
name of this runtime, for example blk.5.attn_k.weight to
model.language_model.layers.5.self_attn.k_proj.weight. The reader gives the
same methods as SafeTensors, so the model accepts it with no change to the
weight cache.

A test of the reader gave an exact result. The Q4_0 dequant, the Q6_K dequant,
and the int4 XOR path all agree with a scalar reference.

## Status

All phases are done. The model answers "The capital of France is **Paris**."
The code reads the Q4_0 data of the GGUF directly. It does not build the
on-disk weight cache for the GGUF. The local copy of the file is at
~/.cache/np_gemma/models/gemma-4-26B-qat-q4_0. The load from the local disk is
64 s. The load from the network mount is 206 s.

The embedding table must stay at Q6_K precision. A 4-bit output head changes
the first token and the text becomes " is is is". The GGUF reader sets
keep_embedding_bf16.

The model is slow. The decode is about 3.5 tokens for each second on a quiet
machine. The Python loop over 128 experts controls the time. The scan now
visits the selected experts only.

Phase 6 is done. A new int4 tile gives a small group of tokens to one expert.
It reads the x block one time for 16 rows and decodes each weight group one
time for up to 16 tokens. A test at 16 tokens gave 3.1 times more speed for the
expert shapes. ops.linear_int4 uses the tile for a group of 8 tokens or more.

### Phase 1: get the file and confirm the layout

1. Download gemma-4-26B_q4_0-it.gguf (14.44 GB) from the GGUF repository.
2. Read the metadata and the tensor directory. Confirm the names and the shapes
   in this plan. The reader does this step.
3. Test the reader. Check the Q4_0 path, the Q6_K path, and the F32 path
   against a scalar reference.

### Phase 2: read the configuration

Change np_gemma/config.py.

1. Add num_experts, top_k_experts, moe_intermediate_size, and enable_moe_block
   to Config.
2. Add num_global_key_value_heads and global_head_dim.
3. Give LayerPlan the global values for a global layer. The present code already
   holds a plan for each layer.

### Phase 3: load the expert weights

Change np_gemma/model.py.

1. Read the two 3-D expert tensors with the int4 path of the reader. The Q4_0
   data becomes the runtime int4 data with one XOR. Do not quantize again.
2. gate_up_proj has the shape (128, 1408, 2816). Expert e then uses the block
   e of the first axis.
3. down_proj has the shape (128, 2816, 704). Expert e uses the block e of the
   first axis.
4. Read one expert at a time. A full tensor is 507 M values. The float32 form
   is 2 GB. A block read keeps the memory small.
5. Keep the router tensors in float32. They are small.
6. Write the expert data to the weight cache in the same way as the other
   tensors. The cache key covers the dtype.

A row slice of a C-order array is contiguous. The present C kernel accepts the
slice with no change. Thus the plan needs no new expert kernel.

The meaning of ffn_down_exps.scale needs a test. The plan maps it to
router.per_expert_scale. The reference trace confirms the mapping.

### Phase 4: the router and the MoE block

Change np_gemma/model.py and np_gemma/ops.py.

1. Add ops.topk_k. Use numpy.argpartition.
2. Add Model._router. Follow the five steps above. Use float32 for the softmax.
3. Add Model._moe. Run the router. Then run the eight experts. Add the outputs
   with the router weights.
4. Add the MoE block to _decoder_layer. Follow the exact order above.
5. Keep the present dense MLP for the shared expert.

### Phase 5: verify against the reference

1. Extend scripts/capture_trace.py in the reference environment. Add hooks for
   the router probabilities, the expert output, and the MoE output.
2. Capture a trace for the 26B model. A short prompt is sufficient.
3. Compare the trace with scripts/check_trace.py. The cosine value must be 0.99
   or more for each tensor.
4. Compare the first token and then 48 greedy tokens with scripts/gen_ids.py.

### Phase 6: performance

1. Measure the decode time for one token. The target is 5 to 8 tokens for each
   second in the int8 mode on a quiet machine.
2. Measure the memory. The int8 mode needs about 26 GB. The int4 mode needs
   about 13 GB.
3. Measure the router time. The router is small. It must not start a thread pool
   for each layer. Keep OPENBLAS_NUM_THREADS=1.
4. Improve the prompt pass. A prompt has many tokens, and each token selects its
   own eight experts. Group the tokens for each expert. Then run one matrix for
   each expert group.

## Risks

* The int8 cache build reads 51.6 GB and takes about 850 s. Test the disk space
  first. The checkpoint and the cache need about 77 GB.
* The expert read for each token is eight pairs of row slices. The slices are
  about 4 MB and 2 MB. The memory system handles this size with no problem.
* The router softmax must use float32. A bfloat16 softmax changes the top-k
  result and the token ids.
* The reference trace is the most important test. The router index and the
  expert output must agree with the reference. A small error in the routing
  gives a large error in the output.
* The prompt pass is the slow part, because of the many experts. Plan the
  grouping before the first performance test.

## Test list

    test                                    pass condition
    config values against config.json        all values equal
    expert weight round trip                int8 reload gives the same values
    router probabilities against the trace   cosine 0.999 or more
    router top-k index against the trace    all indices equal
    one decoder layer against the trace      cosine 0.999 or more
    first token against the reference        the ids equal
    48 greedy tokens against the reference   all ids equal
    decode speed, int8                      5 tokens for each second or more
    memory, int8                            less than 30 GB
    memory, int4                            less than 16 GB

## Non-goals

* The vision encoder and the audio encoder are out of scope.
* The multimodal input path is out of scope.
* A change to the 12B dense model is out of scope. The new code must keep the
  12B model correct.
