# A better encoder for Q4_K and Q5_K

`rc_kq_to_qk_opt` (reclank/backends/cpu/csrc/kernels/kq/convert.c) and
`cops.kq_to_qk_opt` encode Q4_K and Q5_K blocks with a wider search than
ggml's encoder. Only the encoder changes:
- the blocks are byte for byte the same format;
- every reader takes them unchanged, including the RQ4_K / RQ5_K kernels of
  reclank and plain Q4_K / Q5_K in other readers;
- no kernel changes, same size, same speed.

The LFM2.5 converter (`scripts/convert_lfm25_gguf.py`) uses it by default
for `--experts rq4` and `--experts rq5`. `--ggml-encoder` takes ggml's
encoder instead.

## The block

A super-block of 256 values is 8 sub-blocks of 32, and each sub-block j
stores a 6-bit scale `sc_j` and a 6-bit min `m_j`. The super-block stores
two fp16 values, `d` and `dmin`. Each value is a 4-bit code q (5-bit for
Q5_K):

    w = d * sc_j * q - dmin * m_j

The encoder chooses d, dmin, the 16 six-bit numbers, and the 256 codes.

## What ggml's encoder does (`kq_block_to_qk`)

1. Fit each sub-block with `make_qkx2_quants` (ggml's search over a scale
   and an offset, then 20 refinement steps). It minimizes a weighted error
   with weights `av_x + |x|`, where av_x is the RMS of the sub-block, so it
   favours large values.
2. Set `d = max_scale / 63` and `dmin = max_min / 63`, rounded to fp16.
3. Round each scale and min to 6 bits through 1/d and 1/dmin.
4. Recompute the codes for the rounded scales.

Step 1 minimizes a different error than the one that matters, and steps 2-3
are a single rounding with no search. A better fit for the scales can then
be lost to the 6-bit grid.

## What the new encoder does (`kq_block_to_qk_opt`)

1. **Unit weights in qkx2:** each sub-block is fitted to the plain squared
   error, not to `av_x + |x|`.
2. **A grid for d and dmin:** both are tried at `max/63 * (1 + 0.01 k)` for
   k = -3..3, rounded to fp16. That is 49 pairs, and each pair is scored by
   the exact error of the whole super-block after steps 3-4.
3. **6-bit neighbours:** for each pair and each sub-block, the rounded
   (scale, min) and its 8 neighbours (±1 on either) are tried, with codes
   recomputed. The one with the least squared error wins.
4. **The codes** are recomputed for the chosen d, dmin, and 6-bit values,
   then packed in the standard layout.

The squared error is measured in the frame the block is stored in. For the
RQ formats that is the rotated frame, which is orthonormal, so the error is
the same as in the original weights.

### Selector support (for the free rotation choice)

`dsel` (one int8 for each super-block), `dbits`, `mbits`:
- the low `dbits` bits of the selector are forced into the low mantissa bits
  of `d`, and the next `mbits` into those of `dmin`;
- `kq_f16_up` rounds the value up to the nearest fp16 with those bits, so it
  never shrinks the range;
- the forcing happens inside the d/dmin grid, so the 6-bit search re-fits
  around the forced value. On the RQ4 study's matrices this costs 0.00 dB.

`err` returns the squared error of each super-block, which a caller uses to
choose between candidate encodings. docs/RQ4_ROTATION_CHOICE.md explains
both.

## API

    cops.kq_to_qk_opt(src, cols, bits, dsel=None, dbits=0, mbits=0, err=False)
    # src float32 (rows x cols), cols a multiple of 256, bits 4 or 5
    # -> uint8 blocks (144 B for Q4_K, 176 B for Q5_K), plus err (float64) if asked

    void rc_kq_to_qk_opt(const float *src, int64_t rows, int cols, int bits,
                         const int8_t *dsel, int dbits, int mbits,
                         uint8_t *dst, double *err);   /* OpenMP over super-blocks */

## Results

**Weight error against ggml's encoder** (LFM2.5-24B-A2B bf16 matrices; work/rq4best4):

| | weight SNR change |
|---|---|
| Q4_K, all three changes | -0.21 dB |
| Q4_K, unit weights alone | -0.07 dB |
| Q4_K, 6-bit neighbours alone | -0.13 dB |
| Q4_K, d/dmin grid alone | -0.11 dB |
| Q5_K, all three changes | -0.43 dB |

The output error, with real activations on 51 matrices, improves by 0.18 dB
for Q4_K. Experts and attention/conv agree within 0.05 dB.

**Model quality** (LFM2.5-24B-A2B, RQ4_K experts, dense and head in RQ8_0;
`work/lfm/eval_nll.py`: 10 chunks of 4096 tokens, 40950 tokens, teacher
forced, paired token by token):

| file | loss | vs BF12 |
|---|---|---|
| BF12 | 2.3803 | - |
| rq4all, ggml encoder | 2.3880 | +0.0077 ± 0.0019 (+0.8% ppl) |
| rq4all, new encoder | 2.3842 | +0.0039 ± 0.0019 (+0.4% ppl) |

Paired against the old file, the new encoder is -0.0039 ± 0.0018 nats.

The loss halves, which is more than the 0.2 dB alone would predict. A cost
proportional to the noise power gives about 0.77%. This is one measurement
on one text set (the project's notes and sources), so read it as "clearly
better", not as an exact 2x.

`LFM2.5-24B-A2B-rq4all.gguf` is now the new-encoder file. It is the same
size (14.42 GB) and the same speed: 167 tok/s decode, 2906 tok/s on a 4K
prompt.

## Cost

- **Encoding:** 49 (d, dmin) pairs × 8 sub-blocks × 9 neighbours × 32 values
  is about 113K value evaluations a super-block, against a few thousand for
  ggml.
- **Runtime:** none.

**Speed** (one Q4_K super-block; Xeon 8268 at its AVX-512 clock):

| | 1 core | 8 cores |
|---|---|---|
| ggml encoder | 67 µs | 7.7 µs |
| new encoder, first version (scalar) | 605 µs | 97.5 µs |
| new encoder, AVX-512 | 94 µs | 12.3 µs |

At 94 µs a block on one core, the rq4all experts (89.6M super-blocks) take
about 7 minutes on 20 cores. The first version took 49 minutes, partly
because a simulation shared the CPUs.

The AVX-512 version gives the same bytes as the scalar one. This was checked
on 14 LFM2.5 matrices: Q4_K, Q5_K, and Q4_K with the selector. All the other
encoders in convert.c (Q8_0, Q6_K, ggml Q4_K/Q5_K) are also unchanged on
the AVX-512, AVX2, and scalar builds.
- **The 9 neighbours of a sub-block in one pass** (`kq_qk_sub_err9`): 2 × 16
  values a vector. The rounding and the clamp are done in float; they give
  the codes of the integer conversion.
  - The squares of the float differences are exact in double, and so are
    their sums over 32 values, so the vector order of the sums gives the
    same errors to the bit.
  - The 9 sums are reduced together (hadd and permute).
- **qkx2 with unit weights** (`kq_qkx2_unit`): the 21 trials of the scale
  (16 for Q5_K) go in the lanes. Each lane sums its 32 values in the
  scalar order, so the float sums are the same.
  - ggml changes `min` when a trial wins, and the later trials use the new
    min. So after each win, the trials that follow run again.
  - Wins are few, so this costs a few passes, not 21.
- **`kq_lrint`**: `lrintf` went through the PLT. `cvtss2si` rounds the same
  way, to nearest even, and is inline.

**Tried and dropped:**
- **A reciprocal in place of the division,** with an exact fallback near the
  half-integers: only 2% faster.
- **Stopping a (d, dmin) pair once its partial sum passes the best:** no
  gain. The pairs' totals are within a few percent of each other, so they
  stop only at the last sub-block or two.

The remaining time is spread over the vector work of the 3528 candidates
of each block, at about 45 cycles each. A further 3-4x would need fewer
candidates, for example a coordinate search over d and dmin (13 pairs
instead of 49). That changes the output, so it has not been done.

## Not used

- **An importance matrix** (activation-weighted error in the qkx2 fit, the
  neighbour search, and the grid) adds only about 0.04 dB on top. The 32-point
  Hadamard of the RQ formats mixes the columns of each block and flattens
  the per-column energy that a diagonal importance matrix exploits. It also
  needs activations at convert time.
- **Q6_K:** the RQ6 study found -0.08 dB from a d search alone. That is not
  implemented, since RQ6_K is already within noise of BF12.
