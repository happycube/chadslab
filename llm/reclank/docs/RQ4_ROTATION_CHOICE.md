# RQ4_K with a free choice of rotation for each super-block

**Status:**
- the format is designed and its error is measured on matrices;
- the whole-model simulation is done (work/lfm/sim_rq4sel.py);
- a 164K-token confirmation finds no measurable gain in the model loss
  (-0.0014 ± 0.0018 nats), so **no kernel will be built**.

## Background: the RQ formats

RQ8_0, RQ6_K, RQ5_K and RQ4_K are Q8_0, Q6_K, Q5_K and Q4_K stored after a
fixed orthonormal rotation of each block of 32 weights:

    M0 = D0 · H / sqrt(32)

- H is the 32-point Walsh-Hadamard matrix.
- D0 is a fixed diagonal of ±1 signs (`tq6.SIGNS`).

A weight row w is stored as `quant(w · M0)`. At run time, the activation x
gets the same rotation once (`TQ_ROT` before the experts), and the dot
product is unchanged:

    w · x = (w M0) · (x M0)

The rotation spreads outliers across the block, so the quantizer sees
flatter blocks. For RQ4_K it is worth 0.15 dB at the output against plain
Q4_K, and more on attention and conv.

## The idea

One rotation is right for most blocks but not for all.
- **The choice:** for each super-block of 256 values, offer K = 4 candidate
  rotations, encode the block with each, and keep the one with the least
  squared error.
- **The selector:** the choice needs 2 bits a super-block (0.008 bits a
  weight). It goes into the two low mantissa bits of the fp16 `d` of the
  block, so the block stays 144 bytes.

The four rotations are:

    Mk = Dk · H / sqrt(32)     k = 0..3

- D0 is the fixed TQ6 signs, so k = 0 is plain RQ4_K.
- D1-D3 are random signs from seeds 1001-1003, the same as the RQ8 study
  (work/rq8study/rq8lib).

All four are orthonormal, so the squared error in the rotated frame equals
the error on the original weights.

### Why the selector costs nothing

- The fp16 `d` has a 10-bit mantissa. Forcing its low 2 bits moves d by at
  most 3 ulp, about 0.3%.
- The encoder (docs/KQ_ENCODER.md, `kq_f16_up`) rounds d up to the nearest
  value with the wanted low bits.
- It does this inside its d/dmin grid search, so the 6-bit scales re-fit
  around the forced d.
- Measured cost: 0.00 dB.
- The `dmin` low bits can hold a third bit, for 8 rotations, also at no
  measured cost.

The reader takes `sel = d_bits & 3` and uses d as stored. The d with the
selector bits is the value the codes were fitted to, so nothing needs
masking.

## Error measurements (work/rq4best4)

**Data:** 51 LFM2.5 bf16 matrices: 45 experts (w1/w3/w2 of 3 experts in each
of layers 2, 10, 22, 30, 39), the layer-10 attention, and the layer-20 conv.
Output SNR uses real activations rebuilt from the golden dumps, and each
expert sees only its routed tokens.

Change against RQ4_K (-22.95 dB weight / -25.66 dB output; lower is better):

| form | bpw | weight | output | p99 block |
|---|---|---|---|---|
| Q4_K (no rotation) | 4.5 | +0.09 | +0.15 | +0.27 |
| 4 rotations, one per matrix | 4.5 | 0.00 | 0.00 | -0.01 |
| 4 rotations, one per row | 4.501 | -0.16 | -0.16 | -0.26 |
| **4 per 256, selector in d** | **4.5** | **-0.43** | **-0.43** | **-0.69** |
| 8 per 256, selector in d (2 bits) + dmin (1 bit) | 4.5 | -0.59 | -0.59 | -0.92 |
| new encoder only | 4.5 | -0.21 | -0.18 | -0.15 |
| **new encoder + 4 per 256 in d** | **4.5** | **-0.63** | **-0.61** | **-0.82** |
| 4 per 32 (2 bits a block of 32) | 4.5625 | -1.23 | -1.22 | -1.57 |
| new encoder + 4 per 32 | 4.5625 | -1.35 | -1.33 | -1.70 |
| 4 per 32, selector in the 6-bit scale/min LSBs | 4.5 | -0.92 | -0.91 | -1.15 |

- **The output gain equals the weight gain.** Experts and attention/conv
  agree within 0.05 dB, and the per-256 gain is 0.42-0.47 dB on every
  matrix.
- **Per-32 choices** gain about 3x more. But they need a different x rotation
  for every block of 32 at run time, which costs like the RQ8 best-of-4.
  Putting their selector in the 6-bit scale/min LSBs costs 0.38 dB.
- **The other formats** behave the same way: RQ6_K 4 per 256 in d gives
  -0.51 dB, and RQ8 best of 4 per 32 gives -1.5 dB.
- **An importance matrix** on top adds about 0.04 dB (encoder + imatrix +
  4/256: -0.66).

**The perplexity estimate** takes the cost as proportional to the expert
noise power, starting from the old RQ4 file's +0.8%:
- the new encoder alone: about 0.77%;
- the new encoder with 4/256: about 0.70%.

The measured result for the new encoder alone was much better than its
estimate: +0.4% (docs/KQ_ENCODER.md). So the estimate is only a rough guide,
and the whole-model simulation below is the real test.

## Whole-model simulation (work/lfm/sim_rq4sel.py)

The simulation builds a model file that holds the values of the format
before any kernel exists:
1. for each expert matrix, rotate each block of 32 by M0..M3;
2. encode each with `kq_to_qk_opt(dsel=k, dbits=2, err=True)`;
3. keep, for each super-block, the rotation with the least error;
4. dequantize, rotate back (`· Mk^T`), and store as BF12;
5. keep the dense matrices and the head in RQ8_0, as in rq4all.

The BF12 rounding of the simulated values is about -48 dB, far below the
-23.4 dB of the format, so it does not change the result. The output is
`LFM2.5-24B-A2B-rq4selsim.gguf`, evaluated with the same 40K-token loss and
paired against the new-encoder rq4all.

| file | loss | vs BF12 |
|---|---|---|
| BF12 | 2.3803 | - |
| rq4all, ggml encoder | 2.3880 | +0.0077 ± 0.0019 |
| rq4all, new encoder | 2.3842 | +0.0039 ± 0.0019 |
| rq4r0sim (control: rotation 0 only, new encoder, stored as BF12) | 2.3835 | +0.0032 ± 0.0019 |
| rq4selsim (new encoder + 4 rotations per 256) | 2.3790 | -0.0013 ± 0.0018 |

**The rotations chosen:** 25.3 / 25.1 / 24.8 / 24.9% over the 89.65M
super-blocks. No rotation is special, and R0 wins no more often than the
random ones.

**The control.** The simulated file holds BF12 experts, so the CPU eval
multiplies them by float activations. The real RQ4_K file goes through the
KQ dot products, which quantize the activations. rq4r0sim has exactly
rq4all's format but is stored the same way as rq4selsim:
- rq4r0sim - rq4all: -0.0007 ± 0.0009, so the activation path costs nothing
  measurable;
- **rq4selsim - rq4r0sim: -0.0045 ± 0.0020 (token SE), ± 0.0045 (SE over the
  10 chunks)**.

**By chunk**, rq4selsim - rq4r0sim (nats):

    -0.0038 -0.0146 +0.0009 +0.0321 -0.0118 -0.0182 -0.0109 -0.0079 +0.0008 -0.0113

- **For the rotation choice:** better in 7 of the 10 chunks, about even in
  2, with a median of -0.0094.
- **Against it:** one chunk is much worse (+0.032), and that chunk is why
  the chunk SE is twice the token SE.
- **The size:** the choice would close the remaining RQ4 gap to BF12
  (+0.0032). That is much more than the -0.4 dB of weight error predicts,
  which is another reason to confirm it.

**The confirmation:** 40 chunks (163,800 tokens) of the same text,
rq4selsim against rq4r0sim (`eval_nll.py --chunks 40`):
- **rq4selsim - rq4r0sim: -0.0014 ± 0.0010 (token SE), ± 0.0018 (chunk SE)**;
- 20 chunks better and 20 worse, median chunk +0.0001;
- dropping the 2 most extreme chunks at each end gives -0.0002;
- a bootstrap over chunks gives a 95% interval of -0.0052 to +0.0018.

**The 10-chunk -0.0045 was noise.** The per-token losses are correlated
within a chunk, so the token SE understates the error. Only the chunk SE
is a fair measure, and with 10 chunks it is about ±0.0045.

The result agrees with the prediction from the weight error. Scaling
RQ4's +0.0032 gap by the noise power (-0.4 dB, a factor of 0.91) predicts
about -0.0003 nats, far below what this text can resolve.

## Kernel design

The format is RQ4_K with one rule added: super-block b of a row was
quantized in frame `sel_b = low2(d_b)`. The format needs a new type id,
because a reader that ignores the selector computes the wrong values.

### Decode (memory bound; GEMV and the hot MoE kernel)

- **x preparation:** rotate x by all four Mk instead of only M0, and
  quantize each copy to int8 with its own scale and sum (KQH_X=i8). That is
  4× the work on a vector of 2048 values per token, which is negligible.
- **The kernel:** it already loads `d` for each super-block. It takes
  `sel = d & 3` and reads x from copy sel. The only change is an index on
  the x reads.
- **Memory:** x for a token grows from about 2.3 KB to 9 KB (int8 plus
  scales). It still sits in shared memory or L1.
- **Expected cost:** within the noise of a weight-bound decode (to be
  measured; the rule is ≤3%).
- **CPU:** the same per-super-block pick of x in the KQ dot products.

### Prefill (tensor cores, qmoe_tc.cu)

This is where the format is hard. The tensor-core kernel computes
`mma.sync m16n8k32` on int8 codes of w, the A operand covering 16 rows,
against int8 x, the B operand covering 8 tokens, for a block of 32 columns:

    sum w x = xs d (int32 sum of q x) + mn xsum

- **The problem:** the rows of one A fragment have different selectors in
  the same super-block, but the MMA takes one B fragment for all 16 rows. So
  B cannot be pre-rotated, and no fixed row permutation helps, because the
  selector pattern changes from one super-block to the next.
- **Weight-side un-rotation** doesn't work either. The frame change from
  frame k to frame 0 is `Mk^T M0 = H Dk D0 H / 32`, a dense 32×32 transform,
  so the un-rotated weights are no longer the 4-bit integer codes and the
  int8 MMA cannot take them.

There are three options:

| option | how | cost |
|---|---|---|
| A. masked MMA per selector | for each block of 32, one MMA per selector present in the tile: A with the rows of other selectors zeroed, B the matching x copy; the scalar terms (xs, xsum) are per row and per copy anyway | up to 4× MMAs and 4× B fragment loads; x smem 4×. The kernel is not MMA bound at 2.9K tok/s (well under 10% of the int8 peak), so the real cost could be much less than 4× and must be measured |
| B. fp16 path | dequantize to fp16 when the w tile comes to shared memory, apply `H Dk D0 H` (2 × 5 butterfly stages over 32 values; once per tile, shared by all tokens of the tile), run fp16 MMA against x in frame 0 | a new kernel path at half the int8 MMA rate; the transform costs about 10 adds a weight per tile |
| C. coarser selector | one selector for every 16-row MMA tile and each 256 columns | prefill unchanged, but the gain shrinks toward the per-row figure (-0.16 dB) or below; not measured |

Option A is the one to try first: it keeps the int8 data path and the
format, and only adds work in the inner loop. If it costs more than 3% at
4K and 16K prompts, the fallback is to run prefill through the decode-style
path, but that is far slower.

### Untested alternative: permutations across sub-blocks

If the candidate "rotations" are M0 followed by a fixed permutation of the
256 positions across sub-blocks, the frame change is a permutation:
- the weights stay integer codes, so the tensor-core kernel only gathers the
  codes of a block from other positions when it loads the tile;
- permutations within a sub-block of 32 change nothing, and per-element
  sign flips do not keep Q4_K's per-block offset;
- regrouping which values share a 6-bit scale does change the error, but the
  gain is unknown and likely smaller than with different Hadamard signs.

## Recommendation

**Do not build it.**
- **The gain is too small to measure:** -0.0014 ± 0.0018 nats over 164K
  tokens. The prediction from the weight error is about -0.0003.
- **The cost is high:** the int8 tensor-core prefill path needs option A (up
  to 4× MMAs and x tiles) or a new fp16 path, plus a new type id.
- **What actually helped:** the new encoder halved the RQ4_K loss at no
  runtime cost (docs/KQ_ENCODER.md). rq4all is now within about +0.3-0.4%
  perplexity of BF12.
- **For more quality,** RQ5_K or RQ6_K experts: RQ6_K is within noise of
  BF12 at 20.3 GB, against 14.4 GB.

**A lesson for the evaluations of this model:** judge a difference by the
SE over chunks, never by the SE over tokens. With 10 chunks of 4096
tokens, a difference below about 0.01 nats is not resolved; 40 chunks bring
that to about 0.004.
