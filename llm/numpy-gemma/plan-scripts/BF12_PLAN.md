# Plan: BF12, the bfloat16 matrices in 12.25 bits with no loss (and RQ10)

Status: a plan with test kernels. No runtime code has changed.
The test kernels and the scripts are in this directory (section 7); they
include csrc/gpu.cu read-only and call cops, nothing more.

## 1. Why

The dense matrices of Qwen3.8 (4.95G values) are bfloat16 in the RQ8_0 file
(9.9 GB). The runtime runs them as they are (NP_GEMMA_DENSE=bf16) or
requantizes them to Q8_0 at the first use (q8, the default: 5.3 GB, decode
42.3 -> 51.2 tok/s on the NVFP4 file, but KL to float32 0.019 against 0.010
to 0.013 for bf16). RQ6_MIX_PLAN.md section 8 decides between them with a
measure on chat text.

BF12 is a third choice: the bf16 values, bit for bit, in 77% of the bytes.
It is the answer if the measure says the dense matrices must stay bf16, and
it costs less memory and time than bf16 everywhere bf16 is kept.

## 2. The formats

A row of cols values (cols a multiple of 32), in groups of 32, padded to 16
bytes. The layouts are those of fmt_make_data.py (bf12_rows, rq_rows).

BF12 (12.25 bits a value; not rotated):

    lo[cols]     byte j = sign << 7 | the 7 bits of the mantissa of value j
    hi[cols/2]   the exponent gap: gap = E - e (0..15); byte 16 g + j has the
                 gap of value 32 g + j in its low half and of 32 g + j + 16 in
                 its high half
    E[cols/32]   the largest bf16 exponent of group g
    bf16 bits    sign << 15 | (E - gap) << 7 | mantissa, except the zero code
    zero (neg0)  sign 1, gap 15, mantissa 0 decodes to 0 (+0.0 bits)

- Exact when the exponent of a value is at most 15 below the largest of its
  group: 99.992 to 99.995% of the values of ORIG. The others (0.005 to
  0.008%: values under 2^-15 of their group's largest), and the zeros and
  the denormals of bf16 (none in ORIG), get the zero code. The one exact
  value that the code takes from the range, -2^(E-15) (mantissa 0), also
  decodes to 0. The whole matrix: -121.6 to -129.1 dB against the bf16
  values (qkv -125.8, ssm_out -129.1, v_proj -121.6).
- The decoder: the bits as above, then a select: zero where gap == 15 and
  the byte of lo == 0x80 (two compares and a mask AND on AVX-512; one
  condition on the GPU).

The zero rule, three forms measured (bf12_zero_*: the same rows with each
rule; times are medians of 30 interleaved rounds):

    rule                         weights (qkv / ssm_out / v_proj)   CPU DRAM      CPU in cache   GPU
    none: the smallest code      -126.6 / -126.2 / -125.0 dB        1.000         1.000          1.000
    gap15: gap 15 = zero         -118.1 / -120.9 / -116.3           0.97 to 1.00  1.02 to 1.12   0.99 to 1.01
    neg0: the plan               -125.8 / -129.1 / -121.6           0.94 to 1.05  1.12 to 1.23   0.92 to 1.01

  (Two runs. CPU: qkv 10240 x 2560 from DRAM 878 to 928 us for all three,
  the order changed between the runs: noise. ssm_out and v_proj in the
  cache: 162-170 / 181-188 / 200-203 us and 19.8-20.0 / 20.4-21.1 /
  22.1-22.6 us: neg0 is slower each time. GPU: bound by the bandwidth, 360
  to 404 GB/s for all three: noise.)
- neg0 keeps all 16 binades for the values (gap 15 stays a value but for
  one bit pattern) and has the least error of the two rules with a zero;
  from DRAM both are free (the differences are noise); when the matrix
  is in the cache (the decode bound) gap15 costs 2 to 12% and neg0 12 to
  23%. All three are about 60 dB under the rounding of bf16 itself (-55.6
  dB), so the choice is the zero code, not the accuracy. The plan uses neg0 (the
  user's choice); gap15 is the fallback if the CPU decode becomes the
  bound of a path (the small matrices in the cache).
- Version 2 (only if a test needs the exact bits of a bf16 run; not for
  quality: the zeroed small values are acceptable, and a matrix with a
  large one stays bf16, step 1): a side table for each tensor (the flat
  index, uint32, and the bf16 bits) of the values out of range and of the
  -2^(E-15) values that the zero code takes.
  The decoder (dequant) patches them; the products add sum (w - w_flush) x
  over the entries of a row (about 1500 entries for a 10240 x 2560 matrix).

RQ10 (10.5 bits; rotated as RQ8_0, gemma_tq6_rotate in each 32):

    lo[cols]     the low 8 bits of u = q + 512
    hi[cols/4]   the high 2 bits: byte 8 g + j has values 32 g + j, + 8, + 16,
                 + 24 at bits 0, 2, 4, 6
    d[cols/32]   float16 scale; value = d (lo + 256 hi - 512)

RQ12 (12.5 bits, the 4-bit hi of BF12 with q + 2048) was measured and is
dropped: BF12 is smaller and exact at the same speed.

The type numbers: 55 is RQ8_0 (gguf.py), 56 is kept for RQ6_K
(RQ6_MIX_PLAN.md); BF12 = 57, RQ10 = 58 in the GGUF; in memory KQ_BF12X16 =
63 (the CPU groups of 16 rows, as KQ_BF16X16 = 61). Check that they are free
before the work.

## 3. What the test kernels measured (this machine: Xeon W-2295 18 cores,
RTX 5060 Ti; one token; x Gaussian with 8 channels x20)

Errors of the output against the exact bf16 product (float64). DRAM: the
weights from memory (copies cycled).

    qkv 10240 x 2560 (DRAM)     CPU error    CPU us   GPU error    GPU us   bytes
    Q8_0 (runtime, int8 x)      -39.31 dB    676      -43.68 (x f) 73       27.9 MB
    RQ8_0 (runtime, int8 x)     -43.70       606      -45.96 (x f) 71       27.9
    bf16 (runtime, int8 x)      -41.41       1314     -137.4 (x f) 134      52.4
    bf16 (test, float x)        -135.28      1125-1203                      52.4
    BF12 (test, float x, neg0)  -124.59      855-938  -124.57      105-107  40.1
    RQ12 (test, int16 x)        -69.86       970-1014 -69.87       108      41.0
    RQ10 (test, int16 x)        -57.97       822-825  -57.97       90       34.4

    ssm_out 2560 x 6144         CPU us (in cache)     GPU us (DRAM)
    bf16 (test, float x)        157-165               83
    BF12                        147-153               63-65
    RQ10                        184-192               55

- The GPU kernels read 373 to 391 GB/s for every form (448 peak): the time
  follows the bytes; BF12 is 20% faster than bf16 with no change of value.
- The CPU from DRAM: 41 to 47 GB/s for every form; BF12 24 to 29% faster
  than a bf16 kernel. In the cache (ssm_out, v_proj) the decode of BF12
  costs a little: about the time of bf16.
- The int8 x of the runtime's CPU kernels limits their output to -37 to
  -44 dB whatever the weights (bf16 with int8 x: -41.4). The rotation of x
  (RQ8_0) is worth 2 to 6 dB of that. So on the CPU a form finer than 8
  bits only pays with float or int16 x: the CPU kernels of BF12 take float
  x (as KQ_BF16X16 does now), those of RQ10 int16 x.

## 4. The steps

### Step 1: the format in gguf.py and a converter at load

- gguf.py: BF12 = 57 in the row types (_ROW_BYTES: pad16(cols + cols / 2 +
  cols / 32)), _TYPE_NAME "BF12", tensor_bytes, and _dequant_rows: the exact
  bf16 bits with the neg0 zero, then float32 (the numpy decoder: the
  inverse of bf12_rows of fmt_make_data.py).
- cops / kquants.c: kq_bf16_to_bf12(src uint16, rows, cols, out) and
  kq_bf12_to_bf16 (the C forms of bf12_rows and its inverse; the count of
  flushed values returned).
- qwen4.dense_mode: a mode "bf12": the large bfloat16 matrices (the list of
  q8: rows > 1, cols % 32 == 0, not the indexer, not token_embd) become BF12
  at the first use, in Qwen4CPU.K, as q8 makes Q8_0 now. No new file is
  needed to try it.
- The report of the values BF12 cannot hold, in every path that makes BF12
  (kq_bf16_to_bf12 returns the data; the load-time mode prints a line for
  each tensor with any, the converter a table; check_bf12.py the same):
  - the count of zeroed values (more than 15 binades under the largest of
    their group of 32), the largest of them in absolute value and against
    the RMS of the tensor, and the worst 10 (row, column, value, the group
    maximum);
  - the neg0 collisions (values exactly -2^(E-15), which decode to 0);
  - any Inf or NaN: an Inf makes E = 255, and every finite value of its
    group is zeroed.
  The rule (the user's): the zeroed small values are
  acceptable. A matrix with a large one (a zeroed value above 2^-8 of the
  tensor's RMS: an outlier in its group pushed an ordinary value out) or
  with an Inf or NaN stays bf16, always (no flag): the converter writes it
  as BF16 and the load-time mode keeps it as it is; the report names each
  such matrix and its worst values. The dense part is then a mix of BF12
  and bf16 matrices, one type each: the products already take the type of
  each matrix (KQ_LINEAR, GP_KQ_LINEAR), and the GPU counts the bytes of
  each (qwen4_gpu.py:188). On ORIG's test matrices the largest zeroed value is 3.0e-5 (qkv;
  1.6e-3 of the RMS; ssm_out 9.8e-6, v_proj 1.5e-5), the collisions 1 to 7
  a matrix (1.9e-6), and no Inf or NaN: nothing large.

### Step 2: the CPU

- kq_row_bytes(57), and the one-token dot (the AVX-512 code of bf12_dot in
  fmt_cpu.c: zero-extend lo, the nibbles of hi, E - gap, the bits, << 16,
  fma with float x).
- KQ_BF12X16 = 63 for Qwen4CPU.KP: groups of 16 rows as KQ_BF16X16; for each
  block of 32 columns the 16 rows' lo (512 bytes), hi (256), and E (16):
  784 bytes. kq_pack_x16f gets a BF12 source; kq_x16f_groups4,
  kq_x16f_group16, and kq_x16f_group (the bf flag: 0 float32, 1 bf16, 2
  BF12) decode a block of 16 x 32 values to floats in registers, then the
  same fma as bf16. The packed copy is then 12.25 bits a value in place of
  16 (the private memory of KP: 8.2 GiB of dense matrices with bf16).
- kq_rows (the embeddings, if token_embd is BF12): decode rows.

### Step 3: the GPU

- gpu.cu: KQ_BF12 = 57 in kq_row_bytes (6303), a branch in kq_row_part
  (the loop of k_fmt<0> in fmt_gpu.cu: a lane takes 16 values of a half
  group; uint4 loads of lo and hi; a warp a row), and in kq_dequant8 if a
  path reads it.
- The prompt: gg_gemm_bf16 / k_gemm_bf16_tc read uint16 tiles of w. A BF12
  tile loader (decode to the bf16 bits in shared memory, then the same
  mma) keeps the tensor cores: a template flag of k_gemm_bf16_tc, and the
  dispatch of GP_KQ_LINEAR for type 57 (11662).
- Qwen4GPU: the bytes of the dense part on the GPU (qwen4_gpu.py:188, "bf16
  counts as Q8_0") with BF12; the upload of the BF12 rows.

### Step 4: the checks

- plan-scripts/run_fmt.sh: the test kernels are the reference of the
  decode and of the speed.
- scripts/check_bf12.py (new): (a) every dense tensor of ORIG: the decoded
  bits equal to the bf16 bits except the zeroed ones (the report above:
  count, largest, dB), and every matrix with a large zeroed value or an
  Inf or NaN kept in bf16; (b)
  the CPU and GPU products of BF12 against those of bf16 on the same
  matrices (float x: relative difference under 1e-6 apart from the
  flushed values); (c) the model: dense bf12 against dense bf16 on the chat
  transcript (scripts/chat_quality.py, tests/texts/qwen38_chat.json): KL
  under 1e-5 and the same top tokens; check_qwen4_cpu.py and
  check_qwen4_gpu.py with NP_GEMMA_DENSE=bf12.
- The rates: decode, MTP (3 drafts), pp512, the 8K prompt, with dense bf16,
  bf12, and q8; the GPU memory left for hot experts.

### Step 5 (optional): RQ10

Only if a lossy form between q8 and bf16 is wanted (RQ6_MIX_PLAN.md section
8 step 2 shows q8 dense measurably worse, and bf12 too slow). -58 dB at 10.5
bits. The CPU needs int16 x (kq_quant16_body of the KQ_Q4X prompt exists),
the GPU the float-x path (kq_row); the rotation of x per input vector as in
RQ6_MIX_PLAN.md section 8 (the vector table).

## 5. What to expect

- The dense part: 9.9 GB (bf16) -> 7.6 GB (BF12); q8 is 5.3 GB.
- With dense bf16 today, the dense products of a decode step are about 8
  ms of a step of 35 ms (QWEN38_PLAN.md trace): BF12 takes about 20% of
  that away (about 1.6 ms), and 2.3 GB of the GPU go to hot experts.
- No change of the values apart from 0.005 to 0.008% of tiny values set
  to zero (none with version 2).

## 6. Risks

- The CPU kernels on small matrices in the cache are bound by the decode,
  not the bytes: no gain there (v_proj: BF12 19 us, bf16 18 us).
- The tile loader of the tensor-core prompt is the largest piece of work;
  without it a BF12 prompt would fall back to slower kernels.
- The zeroed values: a result is not bit-equal to bf16 (-122 to -129 dB)
  until version 2.

## 7. The files of this directory

    BF12_PLAN.md          this plan
    run_fmt.sh            builds and runs the BF12 / RQ10 / RQ12 tests and the
                          zero rules:
                          sh plan-scripts/run_fmt.sh [ORIG] (from numpy-gemma;
                          NPG_FMT_DATA, default /tmp/np_gemma_fmt, gets 350
                          MB of data; PY, NVCC may be set)
    fmt_make_data.py      the test data from ORIG (qkv of layer 22, ssm_out of
                          layer 22, v_proj of layer 23): the BF12, RQ12, RQ10,
                          Q8_0, RQ8_0, bf16 rows; x, rotated x; the exact y
    fmt_cpu.c             the CPU test kernels (AVX-512, F16C, OpenMP):
                          mv_bf12, mv_rq (10 or 12 bits, int16 x), mv_bf16
    fmt_cpu_bench.py      their checks and times against cops.kq_linear
                          (Q8_0, RQ8_0, bf16 with int8 x)
    fmt_gpu.cu            the GPU test kernels (k_fmt<0|12|10>) against kq_row
                          of gpu.cu (Q8_0, RQ8_0, BF16)
    bf12_zero_make.py     the BF12 rows with the three zero rules (none, gap15,
                          neg0) and their exact products
    bf12_zero_cpu.c       mv_none, mv_gap15, mv_neg0 (AVX-512, float x)
    bf12_zero_cpu_bench.py  their checks and times, interleaved rounds
    bf12_zero_gpu.cu      k_bf12<0|1|2>, the same on the GPU

The scripts of RQ6_MIX_PLAN.md (run from numpy-gemma with PYTHONPATH=.):

    ud_vs_orig.py         the UD-Q4_K_XL expert tensors against ORIG, with
                          RQ6_K and Q8_0 of the same experts
    bench_cpu_types.py    the cold experts of one layer on the CPU for each
                          type (cops.kq_calib_nodes)
    bench_gpu_types.cu    the hot-expert products of one token for each type
                          (nvcc -O3 -arch=native -I np_gemma/csrc)
    mtp_fp8_vs_rq6k.py    the FP8 MTP experts of ModelOpt against RQ6_K, Q6_K,
                          and Q8_0 of ORIG

## 8. Related work

BF12 puts together pieces that are in the literature; a short search
found no format that is this one (an E4M7 element under a
power-of-two group exponent, for bf16 weights with no loss).

- Microscaling (OCP MX; Rouhani et al., "With Shared Microexponents, A
  Little Shifting Goes a Long Way", ISCA 2023): a block of k elements
  under a shared scale in E8M0 (a power of two), each element a small
  float with its own sign, exponent, and mantissa (FP8 E4M3 / E5M2, FP6,
  FP4, INT8). BF12 has that shape: the group exponent E is an E8M0 scale
  (the largest exponent of the 32), each value an "E4M7" element (the gap
  as a 4-bit exponent under E, the 7 bits of the bf16 mantissa). The MX
  elements are 8 bits or less for low-bit inference; BF12 widens the
  element until the bf16 mantissa fits. The same paper also studies two
  levels of shared exponents.
- Block floating point (MSFP, Flexpoint): one exponent for a block and the
  mantissas shifted to it. The small values of a block lose their low bits:
  that is why a plain 12-bit integer form (Q12, section 2) is not exact,
  while BF12 keeps an exponent for each value.
- AdaptivFloat (Tambe et al., DAC 2020): a float of few bits with an
  exponent bias for each layer; its authors make the argument above
  against block floating point (shared exponents hurt the small weights,
  an exponent for each element keeps them), and it leaves IEEE 754 for a
  code of its own for zero, as the neg0 code here. BF12 is close to that
  idea with the bias for each group of 32 in place of each layer, and 7
  bits of mantissa.
- Lossless compression of bf16 weights:
  - DFloat11 (Zhang et al., Rice University and xMAD.ai, 2025): Huffman
    codes for the bf16 exponents, the sign and the mantissa as they are:
    about 11 bits a weight, bit-exact, with a GPU kernel that decodes
    (lookup tables in on-chip memory, a two-phase kernel for the positions,
    a transformer block at a time). They report about 2.6 bits of entropy
    in the exponents; the dense matrices of ORIG have 2.55 to 2.91 bits.
  - ZipNN (IBM, 2024): the exponents apart (12 values are 99.9% of them),
    Huffman codes; 33% and more off the size of a model file. For storage
    and transfer, not for the products.

Where BF12 stands:

- A fixed width, 12.25 bits: every row has the same size, so a row is
  found at once, the decode is a few shifts and masks for each value on
  AVX-512 or in a warp, and the products stay bound by the bandwidth (the
  test kernels: bf16's GB/s on the GPU and the CPU, no lookup tables).
- The cost of the fixed width: about 1.25 bits a weight more than the
  entropy codes of DFloat11 (about 10% of the size), and the zero code
  (the values under 2^-15 of their group's largest, 0.005 to 0.008% of
  them, become 0: no bit-exact result without version 2).
- The gain: a kernel that multiplies from the stored rows, in place of a
  decompression of a block of weights before the products.

Sources:

- DFloat11: https://arxiv.org/pdf/2504.11651
- ZipNN: https://arxiv.org/pdf/2411.05239 and
  https://research.ibm.com/blog/Zip-NN-AI-compression
- Shared microexponents (MX): https://arxiv.org/pdf/2302.08007
- The OCP microscaling formats: https://pychop.readthedocs.io/en/latest/ocp_mx.html
- AdaptivFloat: https://arxiv.org/pdf/1909.13271 and
  https://vlsiarch.eecs.harvard.edu/publications/algorithm-hardware-co-design-adaptive-floating-point-encodings-resilient-deep
