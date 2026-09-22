# Plan: int4 weights with int8 activations

## Goal

Make the int4 matrix products use int8 activations. The dot product then uses
integer multiply and add. The float conversion of each weight leaves the inner
loop. This is the kernel that llama.cpp uses for the Q4_0 type.

## Why

The 256-token prefill of the 26B model takes 5.16 s on jackal. The same machine
and the same batch size give llama.cpp 106.10 tokens/s. We measure 49.62
tokens/s.

The stage report gives these rates. A GFLOP is one billion float operations.

    stage        GFLOP    GFLOP/s
    expert GEMM    730        241
    attention      531        341
    dense MLP      274        650

The weight traffic of the expert stage is only 3.8 GB/s. The machine gives
about 50 GB/s. Thus the prefill is compute bound, not memory bound.

The current int4 dot changes each weight to float32. It then uses a fused
multiply and add. The conversion costs more than the multiply. An int8 dot
uses the instruction maddubs. One instruction does 16 multiply and add pairs.
The conversion leaves the loop.

## The dot product

A group holds 32 values. The Q4_0 block holds 16 bytes. Byte j holds value j in
the low nibble and value j+16 in the high nibble. The value is the nibble minus
8. The runtime keeps one float32 scale for each group in the scales array.

The activation x becomes int8 with one float32 scale for each group of 32
values:

    sx   = max(abs(group)) / 127
    qx   = round(x / sx)
    sumx = sum(qx)

The last value corrects the -8 offset. For one group:

    sum_i (nib_i - 8) * qx_i
      = sum_i nib_i * qx_i - 8 * sum_i qx_i
      = dot - 8 * sumx

The result of one group is:

    (dot - 8 * sumx) * wscale * sx

The inner loop stays in integer. Only the group result becomes float.

The multiply maddubs takes an unsigned byte and a signed byte. The nibbles are
0 to 15, so they are the unsigned side. The qx values are signed. Thus:

    expanded = the 16 nibble bytes in the order low, high
    dot = hsum(madd_epi16(maddubs(expanded, qx), 1))

The largest value is 32 * 15 * 127 = 60960. It fits in int32. The result of
maddubs fits in int16, because two products give at most 3810.

## Parts

1. A quantization kernel. It reads x with the shape (tokens, cols) and writes
   qx (int8, tokens * groups * 32), sx (float32, tokens * groups), and sumx
   (int32, tokens * groups).
2. The dot functions dot_i4_q8 and dot4_i4_q8. Each has an AVX2 version and an
   AVX-512 version.
3. The three shapes of the int4 path:
   - gemma_int4_q8_linear for one token (GEMV),
   - gemma_int4_q8_tile for a small group of tokens (the MoE expert GEMM),
   - gemma_int4_q8_gemm for a long prompt (the attention and dense GEMM).
4. The ctypes bindings in cops.py and the dispatch in ops.linear_int4.
5. One environment variable NP_GEMMA_INT4_Q8. The value 0 keeps the float path.
   The default is 0 until the tests pass.

## Phases

Each phase ends with a commit and a test.

- Phase 0: the quantization kernel. Test it against NumPy.
- Phase 1: the tile. This is the largest stage of the prefill (59 per cent).
  The expert GEMM gives about 16 tokens to each matrix.
- Phase 2: the multi-level GEMM. This serves the attention projection and the
  dense MLP (38 per cent of the prefill).
- Phase 3: the GEMV. A decode step uses one token. A decode step is closer to
  the memory limit, so the gain is smaller. Do this phase last.
- Phase 4 (optional): the AVX-512 VNNI instruction _mm512_dpbusd_epi32. Jackal
  gives VNNI. Our AVX-512 library does not ask for it. A separate library and a
  run-time test are necessary.

## Verification

Use the pattern of scripts/check_q6k.py.

1. A new script scripts/check_int4_q8.py:
   - Compare the kernel with a NumPy q8 reference. The result must agree to
     float32 precision.
   - Compare the kernel with the float path. The difference must stay near the
     activation quantization error.
   - Use the token counts 1, 2, 3, 7, 8, 15, 16, 17, 63, 64, 65, and 128.
2. Run scripts/gguf_generate.py on the 26B. The text must end in Paris. The
   activation quantization can change a token id. Check the text.
3. Run the A/B of the prefill. The prefill moe line must fall below 3024.8 ms.
4. Run the decode profile. The decode median must not become worse.

## Risks

- The int8 activation loses precision. The model output can change. This is the
  same trade that llama.cpp makes. Set NP_GEMMA_INT4_Q8=0 to compare.
- The int4 scales are float32 in the runtime. llama.cpp uses float16. Keep
  float32. The scale is outside the inner loop, so it costs little.
- The tile kernel has a mask for a part token block on AVX-512. The q8 tile
  needs the same mask.
- Phase 4 needs a third library and a second run-time test. Do it only after
  phases 0 to 3 give a measured gain.

## Open items

- The decode numbers are not yet comparable. Our profile uses a 5-token
  context. llama-bench uses a 256-token context. Add a context length to the
  decode part of the profile.
- The short-prompt prefill is not measured. The user benchmark gives 27.11
  tokens/s. Find the command and the prompt length of that number.

## Results

Phase 0 and phase 1 are done. The kernel commit is 433eecf.

- The quantizer matches NumPy bit for bit. The tile matches a q8 reference to
  3e-7 and the float path to 0.6 per cent.
- The 26B model gives the same token ids with the int8 path on and off:
  [818, 5279, 529, 7001, 563, 5213, 50429, 84750] = "The capital of France is
  **Paris**."
- The jackal microbenchmark on one thread gives about 2.1 times on the expert
  shapes and 2.0 to 2.5 times on the attention and dense shapes.
- The 256-token prefill on jackal went from 45.03 to 59.52 tokens per second:

      stage        int8 off    int8 on    gain
      moe           3435.2 ms  2498.0 ms  1.38
      attn          1638.0 ms  1354.7 ms  1.21
      dense_mlp      451.3 ms   288.0 ms  1.57
      norm           127.0 ms   127.9 ms  1.00

  The decode median did not become worse. A decode step uses the float path,
  because the tile needs a group of tokens.

## What is left

- The MoE stage gains only 1.38 times, not the 2.1 times of the kernel. The
  model calls one expert at a time from Python. A group of 16 tokens gives a
  small parallel region, so 18 threads give only about 2.3 times. A fused MoE
  call that covers every expert in one parallel region is the next step. The
  decode path already has that shape (gemma_int4_moe_gemv).
- The attention stage gains 1.21 times. The stage includes the score matrix and
  the softmax, which stay float32.
- Phase 3, the one-token GEMV. The tile needs a group of tokens. A decode step
  is memory bound, so the gain is small.
- Phase 4, the VNNI instruction, needs a third library and a run time test.

