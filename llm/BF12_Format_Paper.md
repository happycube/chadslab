# BF12: Storing bfloat16 Neural-Network Weights in 12.25 Bits Without Rounding Them

*A format note from the reclank project*
*Sources: `reclank/plan-scripts/BF12_PLAN.md` (the format plan and measurements), `reclank/plan-scripts/fmt_make_data.py` (the reference packer), `reclank/reclank/backends/cpu/csrc/kernels/kq/convert.c` (the runtime converter), `reclank/plan-scripts/fmt_cpu.c` and `fmt_gpu.cu` (the test kernels), `reclank/QWEN_DENSE_QUANT.md` and `reclank/README.md` (end-to-end results).*

*happycube's note:  This was written entirely by Qwen 3.8 Flash-Next, repacked in BF12 and run in reclank. If you can read past the slop here, I think it's worth looking into!*

---

## Abstract

Modern LLM weights are usually stored as **bfloat16** (bf16): 16 bits per number,
sign + 8-bit exponent + 7-bit mantissa. Running a model means streaming those
bits through a multiply-accumulate unit, so *fewer bytes per weight is almost
always faster* — but every ordinary quantization (int8, int4, ...) changes the
numbers and costs accuracy.

**BF12 stores bfloat16 weights at 12.25 bits per value and keeps them bit-for-bit
exact — for 99.99%+ of real weights.** It does this by noticing that within any
small group of 32 weights, the *exponents* are nearly always within 15 powers of
two of each other. So instead of storing 8 exponent bits per weight, BF12 stores
one shared group exponent plus a 4-bit "gap" per weight. The mantissa keeps all
7 of its original bits. Nothing is rounded; nothing is rescaled. The only
casualties are weights more than 2<sup>−15</sup> of their group's largest value
(0.005–0.008% of the weights in practice), which are flushed to zero — an error
about **60 dB below the rounding error bfloat16 itself already has.**

The payoff: 23% fewer bytes than bf16, decode that is a handful of shifts and
masks, products that stay memory-bandwidth-bound, and therefore roughly **20–29%
faster matrix-vector products than bf16 with no loss of model quality** (KL to a
bf16 run under 10<sup>−5</sup>, same top tokens). This paper explains the format,
shows working code for the encoder, the decoder, and fused AVX-512 / CUDA
multiply kernels, and reports the measurements that justify it.

---

## 1. Introduction

### 1.1 The problem: bytes are time, and quantization costs accuracy

When a GPU or CPU multiplies a weight matrix by an activation vector during
token-by-token decoding, it does very little arithmetic per byte read. A decode
step is *memory-bound*: the time is set by how fast the weights can be streamed,
not by how fast the multipliers run. Halving the bytes of a weight matrix
roughly halves the time to use it.

That is why quantization (int8, int4, NF4, ...) is popular: fewer bits per
weight means fewer bytes, which means more tokens per second and less VRAM. But
every one of those formats *changes the numbers*. In a Mixture-of-Experts (MoE)
model the routed experts can usually tolerate aggressive quantization because
only a few fire per token and their errors average out — but the **dense**
matrices (attention projections, conv/SSM projections, embeddings, the output
head) run for *every* token, and quantizing them measurably degrades the model.
The reclank project measured exactly this on Qwen3.8: keeping the dense matrices
in bf16 gave a KL divergence of 0.010–0.013 against float32, while requantizing
them to Q8_0 gave 0.019 — a real quality loss.

So there is a gap: **a format that is as accurate as bf16 but smaller and
faster than bf16.** BF12 fills it.

### 1.2 The key observation

Look at the exponents of a typical group of 32 neural-network weights. Weights
in one output row of one layer live in a narrow dynamic range: the largest and
smallest (in absolute value) are usually a few powers of two apart, rarely more
than 2<sup>15</sup>. bfloat16 spends a full **8-bit exponent on every single
weight** to cover that range. BF12 spends **8 bits per group of 32** (the group
exponent `E`) and only **4 bits per weight** for the remaining distance to the
top (`gap = E − e`, which fits in 4 bits exactly when the observation holds —
and it holds for 99.99%+ of real weights).

The mantissa is kept verbatim: all 7 bf16 bits. There is no scale factor, no
rotation, no rounding of the mantissa. A weight either round-trips to *the exact
same 16 bits*, or it was so tiny relative to its group that it becomes zero.

### 1.3 What BF12 is, in one paragraph

BF12 is a **fixed-width microscaling float format**: a block of 32 elements
shares a power-of-two scale (an E8M0-style exponent, chosen as the largest
exponent in the block), and each element is a small float with its own sign, a
4-bit exponent *below* the shared one, and a 7-bit mantissa — an "E4M7" element,
in microscaling vocabulary. Unlike the OCP MX formats, which cap elements at 8
bits for low-bit inference, BF12 *widens the element until the bfloat16 mantissa
fits*, which is what buys exactness. Unlike entropy coders (DFloat11, ZipNN),
it is fixed-width, so any row can be found with one multiply and any value
decoded with a few shifts — the decoder is not a bottleneck, it is a rounding
step you get for free inside the multiply kernel.

---

## 2. The format

### 2.1 bfloat16 in 30 seconds

A bfloat16 number is 16 bits:

```
 bf16:  [ s | eeeeeeee | mmmmmmm ]         value = (−1)^s · 2^(e−127) · 1.m
         15  14......7   6......0
```

It is exactly a float32 with the low 16 bits chopped off, which is why
converting between them is a shift. The 8-bit exponent gives bf16 float32's
dynamic range (~±3.4e38); the 7-bit mantissa gives it ~2–3 decimal digits.

### 2.2 The BF12 row layout

One row of `cols` values (`cols` a multiple of 32), grouped in 32s, the whole
row padded to a multiple of 16 bytes:

```
 BF12 row (cols values):
 ┌────────────────────┬──────────────────────┬─────────────────┬────────┐
 │  lo[cols]          │  hi[cols/2]          │  E[cols/32]     │ pad    │
 │  1 byte / value    │  4 bits / value      │  1 byte / group │ to 16B │
 └────────────────────┴──────────────────────┴─────────────────┴────────┘

 lo[j]        = sign << 7 | mantissa        (the sign and 7 mantissa bits of bf16)
 gap          = E − e        (0..15)
 hi[16g+j]    = gap of value 32g+j    in its LOW  nibble
                gap of value 32g+j+16 in its HIGH nibble
 E[g]         = the LARGEST bf16 exponent among the 32 values of group g
```

Reconstruction of a value:

```
 bf16 bits = (sign << 15) | ((E − gap) << 7) | mantissa
```

...with one exception, the **zero code** (§2.4).

**Size arithmetic.** Per group of 32 values: 32 bytes of `lo` + 16 bytes of
`hi` + 1 byte of `E` = 49 bytes per 32 values = **12.25 bits per value**, or
76.6% of bf16. Equivalently, the row is `pad16(cols + cols/2 + cols/32)` bytes —
this is the formula in `reclank/formats/gguf.py`, where BF12 is GGUF row type
**57**.

Why 32 as the group size? It matches the project's other block formats (Q8_0,
RQ8_0, ...), it matches AVX-512 register widths (32 values = 2 vectors), and it
matches a warp half-pass on the GPU (16 values per lane). It is also small
enough that a group max rarely strays far from any member's exponent.

### 2.3 A worked example

Take the value **w = −3.75** in a group whose largest exponent is **E = 128**.

1. bf16 bits of −3.75: −3.75 = −1.875 × 2<sup>1</sup>, so
   `s = 1`, `e = 127 + 1 = 128`, `m = round(0.875 · 128) = 112 = 0x70`:
   bits = `1 10000000 1110000` = **0xC070**.
2. Encode: `gap = E − e = 128 − 128 = 0`.
   `lo = (1 << 7) | 112 = 0xF0`; the nibble in `hi` is `0`.
3. Decode: `bits = (1 << 15) | ((128 − 0) << 7) | 112 = 0xC070`. **Identical.**

Now a tiny value in the same group: **w = 2<sup>−20</sup>** (exponent 107).
`gap = 128 − 107 = 21 > 15` — it does not fit in 4 bits. It gets the zero code
and decodes to **0**. Note how extreme this is: the value is
2<sup>−35</sup> ≈ 3 × 10<sup>−11</sup> of its group's largest weight. No
realistic product can notice it.

### 2.4 The zero code, and the one collision

bf16 zeros and denormals (and out-of-range values) have no legal gap, so BF12
needs a code for "this is zero." Three rules were implemented and measured
(`plan-scripts/bf12_zero_*`):

| rule | encoding of zero | exact-value cost | CPU (DRAM) | CPU (in cache) | GPU |
|---|---|---|---|---|---|
| `none` | none — out-of-range values get the smallest legal code | none | 1.00× | 1.00× | 1.00× |
| `gap15` | `gap = 15` means zero, unconditionally | loses all of binade E−15 | 0.97–1.00× | 1.02–1.12× | 0.99–1.01× |
| **`neg0`** (chosen) | `sign=1, gap=15, mantissa=0` means zero | loses exactly one value: −2<sup>E−15</sup> | 0.94–1.05× | 1.12–1.23× | 0.92–1.01× |

`neg0` reserves a single bit pattern — sign 1, gap 15, mantissa 0 — as "zero".
Everything else, including gap 15 values with a nonzero mantissa or a positive
sign, stays exact. Its only cost is that the *exact* value −2<sup>E−15</sup>
collides with the code and also decodes to zero (measured: 1–7 values per
matrix, ~10<sup>−6</sup> of them). It also gives the lowest error of the two
zero-bearing rules. The decoder check is two compares and a mask:

```c
/* neg0: sign 1, gap 15, mantissa 0 is the zero */
b = _mm512_maskz_mov_epi16(~(cmpeq(gp, 15) & cmpeq(lo_byte, 0x80)), b);
```

From DRAM all three rules run at the same speed (the bytes dominate). Only when
a small matrix sits in cache does the extra select cost anything (12–23% on the
tested CPU kernels), which is why `gap15` is documented as the fallback if a
CPU path ever becomes decode-bound. On the GPU the difference is noise: the
kernels are bandwidth-bound.

### 2.5 Exactness, precisely

A BF12 value is **bit-identical to its bf16 original** if and only if its
exponent is at most 15 below the largest exponent of its group of 32. On real
Qwen3.8 checkpoints that covers **99.992–99.995%** of values. The rest
(0.005–0.008%) are flushed to zero. Whole-matrix signal-to-noise ratios against
the original bf16 values:

| matrix | SNR of BF12 vs bf16 |
|---|---|
| `in_proj_qkv` (10240 × 2560) | −125.8 dB |
| `out_proj` (ssm_out, 2560 × 6144) | −129.1 dB |
| `v_proj` (512 × 2560) | −121.6 dB |

For scale: the rounding error *inherent in bfloat16 itself* (vs float32) is
about **−55.6 dB**. BF12's error is ~65 dB below that — roughly 2000× smaller
in RMS than the error the weights already carry. This is why end-to-end checks
(§5.4) show no measurable model change.

### 2.6 The safety policy: never let a format surprise you

The converter (`rc_kq_bf16_to_bf12` in `convert.c`) never silently degrades a
matrix. For every tensor it reports:

- the **count of zeroed values** and the **largest one** (absolute, and relative
  to the tensor's RMS);
- the **worst 10** individually (row, column, value, group maximum);
- the count of **neg0 collisions**;
- any **Inf or NaN** (an Inf would set `E = 255` and zero its whole group).

The acceptance rule: zeroed *small* values are fine; a matrix with a *large*
zeroed value (above 2<sup>−8</sup> of the tensor RMS — a sign that one outlier
in a group pushed an ordinary value out of range), or with an Inf/NaN, **stays
bf16, always**. The dense part of a model is then a mix of BF12 and bf16
matrices, one type each; the product kernels already dispatch per matrix. On
Qwen3.8 the worst zeroed value across the tested matrices was 3.0 × 10<sup>−5</sup>
(0.16% of RMS): nothing large, so everything converted.

A "version 2" is specified for the day someone needs *bit-exact* reproduction
of a bf16 run: a per-tensor side table `(flat index: uint32, bf16 bits)` of the
flushed and colliding values (~1500 entries for a 10240 × 2560 matrix), patched
in by the dequantizer and added as `Σ (w − w_flush)·x` by the products. It was
never needed: the zeroed values are below the noise floor of the model.

---

## 3. Reference implementation

Four pieces of code tell the whole story: a readable Python packer (the
specification), a scalar C decoder, an AVX-512 fused multiply kernel, and a CUDA
warp kernel. All are taken (lightly trimmed) from the repository.

### 3.1 The encoder, in NumPy (the spec in 20 lines)

From `plan-scripts/fmt_make_data.py` — `bf12_rows` is the reference packer; the
runtime C converter and the GGUF writer produce byte-identical output.

```python
import numpy as np

pad16 = lambda n: (n + 15) // 16 * 16

def bf12_rows(u16):
    """Pack a (rows, cols) uint16 array of bf16 bit patterns into BF12 rows.
    cols must be a multiple of 32. Returns (packed_rows, decoded_bits, n_flushed)."""
    r, c = u16.shape
    g = u16.reshape(r, c // 32, 32).astype(np.int32)
    s, e, m = g >> 15, (g >> 7) & 255, g & 127      # split sign / exponent / mantissa
    E = e.max(-1, keepdims=True)                    # group exponent: the largest e
    gap = E - e
    bad = gap > 15                                  # out of range -> the zero code
    gap = np.minimum(gap, 15)
    m   = np.where(bad, 0, m)
    s   = np.where(bad, 1, s)                       # neg0 zero: s=1, gap=15, m=0
    lo  = (s << 7 | m).astype(np.uint8).reshape(r, c)
    hi  = (gap[..., :16] | (gap[..., 16:] << 4)).astype(np.uint8).reshape(r, c // 2)
    rb  = pad16(c + c // 2 + c // 32)               # row bytes, padded to 16
    row = np.zeros((r, rb), np.uint8)
    row[:, :c], row[:, c:c + c // 2], row[:, c + c // 2:c + c // 2 + c // 32] = \
        lo, hi, E.reshape(r, c // 32)
    dec = np.where((gap == 15) & (s == 1) & (m == 0), 0,          # the zero code
                   (s << 15) | ((E - gap) << 7) | m).astype(np.uint16).reshape(r, c)
    return row, dec, int(bad.sum())
```

And the inverse, for tests and for paths that want plain bf16 back (this is
`cops.kq_bf12_to_bf16` in the runtime):

```python
def bf12_to_bf16(row, rows, cols):
    rb = pad16(cols + cols // 2 + cols // 32)
    row = row.reshape(rows, rb)
    lo  = row[:, :cols].astype(np.int32)
    hb  = row[:, cols:cols + cols // 2].astype(np.int32)
    E   = row[:, cols + cols // 2:cols + cols // 2 + cols // 32].astype(np.int32)
    gap = np.stack([hb & 15, hb >> 4], -1).reshape(rows, cols // 32, 32)  # low nibble first
    s, m = lo >> 7, lo & 127
    bits = (s << 15) | ((E.repeat(32, 1) - gap.reshape(rows, cols)) << 7) | m
    zero = (gap.reshape(rows, cols) == 15) & (lo == 0x80)                 # neg0
    return np.where(zero, 0, bits).astype(np.uint16)
```

A round-trip test is then trivial: `np.array_equal(bf12_to_bf16(*packed), u16)`
must be true everywhere except the reported flushed entries.

### 3.2 The scalar decoder (what "a few shifts and masks" means)

The entire decode of one value, in plain C — no tables, no branches worth
predicting, no multiplies:

```c
static inline uint16_t bf12_decode(uint8_t lo, uint8_t gap, uint8_t E)
{
    if (gap == 15 && lo == 0x80) return 0;                 /* the neg0 zero code */
    return (uint16_t)((lo & 0x80) << 8)                    /* sign  -> bit 15   */
         | (uint16_t)((uint8_t)(E - gap) << 7)             /* e = E - gap       */
         | (uint16_t)(lo & 0x7f);                          /* mantissa verbatim */
}

static inline float bf16_to_f32(uint16_t h)                /* bf16 is float32's */
{ uint32_t u = (uint32_t)h << 16; float f;                 /* top half: a shift */
  memcpy(&f, &u, 4); return f; }
```

### 3.3 The CPU kernel: AVX-512 dot product straight from packed rows

From `plan-scripts/fmt_cpu.c`. This is the pattern that makes the format pay:
the decode is *fused into the multiply-accumulate loop*, 32 values per
iteration, and the machine stays bandwidth-bound. Two accumulators hide FMA
latency.

```c
/* the 4-bit gap plane of one group as 32 words: value j is byte j's low nibble
 * for j < 16 and high nibble for j >= 16 */
static inline __m512i nib32(const uint8_t *p)
{
    __m256i h16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p));
    __m256i a = _mm256_and_si256(h16, _mm256_set1_epi16(15));      /* low nibbles  */
    __m256i b = _mm256_srli_epi16(h16, 4);                          /* high nibbles */
    return _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
}

static inline __m512 bf16x16(__m256i h)   /* 16 bf16 bits -> 16 floats: widen, shift */
{
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

static inline float bf12_dot(const uint8_t *row, int cols, const float *x)
{
    const uint8_t *lo = row, *hi = row + cols, *E = row + cols + cols / 2;
    const __m512i m7f = _mm512_set1_epi16(0x7f), m80 = _mm512_set1_epi16(0x80),
                  g15 = _mm512_set1_epi16(15);
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    for (int g = 0; g < cols / 32; ++g) {
        __m512i l  = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(lo + 32 * g)));
        __m512i gp = nib32(hi + 16 * g);
        __m512i e  = _mm512_sub_epi16(_mm512_set1_epi16(E[g]), gp);        /* e = E - gap */
        __m512i b  = _mm512_or_si512(                                       /* assemble   */
                     _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(l, m80), 8),
                                     _mm512_slli_epi16(e, 7)),
                     _mm512_and_si512(l, m7f));
        /* neg0 zero: mask to 0 where gap == 15 && lo == 0x80 */
        b  = _mm512_maskz_mov_epi16(~(_mm512_cmpeq_epi16_mask(gp, g15) &
                                      _mm512_cmpeq_epi16_mask(l,  m80)), b);
        a0 = _mm512_fmadd_ps(bf16x16(_mm512_castsi512_si256(b)),
                             _mm512_loadu_ps(x + 32 * g), a0);
        a1 = _mm512_fmadd_ps(bf16x16(_mm512_extracti64x4_epi64(b, 1)),
                             _mm512_loadu_ps(x + 32 * g + 16), a1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}

void mv_bf12(const uint8_t *w, int rows, int cols, long rb, const float *x, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) out[r] = bf12_dot(w + (size_t)r * rb, cols, x);
}
```

Note what is *not* here: no dequantization pass into a scratch buffer, no lookup
tables, no per-value multiplies by a scale. The weights are decoded into bf16
bit patterns in registers and multiplied with `x` in float32 — the same
arithmetic a bf16 kernel does, on 77% of the bytes.

### 3.4 The GPU kernel: one warp per row

From `plan-scripts/fmt_gpu.cu`. Each lane handles 16 values (a `uint4` load of
`lo`, half of a `uint4` of nibbles), then the warp reduces. The zero code is a
single ternary.

```cuda
__global__ void k_bf12(const uint8_t *w, size_t rb, int rows, int cols,
                       const float *x, float *out)
{
    int row  = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const uint8_t *lo = w + (size_t)row * rb, *hi = lo + cols;
    int ng = cols / 32;
    float acc = 0.f;
    for (int t = lane; t < 2 * ng; t += 32) {          /* two 16-value passes per group */
        int g = t >> 1, h = t & 1;                     /* group g, low/high half        */
        uint4 L  = *(const uint4 *)(lo + 32 * g + 16 * h);
        const uint8_t *lb = (const uint8_t *)&L;
        uint4 Hv = *(const uint4 *)(hi + 16 * g);
        const uint8_t *hb = (const uint8_t *)&Hv;
        int E = hi[cols / 2 + g];
        const float4 *xv = (const float4 *)(x + 32 * g + 16 * h);
        float xs[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { float4 v = xv[i]; xs[4*i] = v.x; xs[4*i+1] = v.y;
                                      xs[4*i+2] = v.z; xs[4*i+3] = v.w; }
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            int gap = (hb[j] >> (4 * h)) & 15, b = lb[j];
            uint32_t bits = ((uint32_t)(b & 0x80) << 8
                           | (uint32_t)((E - gap) & 255) << 7
                           | (b & 0x7f)) << 16;                       /* bf16 -> f32 top half */
            bits = (gap == 15 && b == 0x80) ? 0u : bits;               /* neg0 zero code       */
            acc = fmaf(__uint_as_float(bits), xs[j], acc);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) out[row] = acc;
}
```

For prompt/large-batch work the same decode happens *inside the tensor-core
GEMM tile loader* (`k_gemm_bf16_tc` with a BF12 flag): BF12 rows are decoded to
bf16 bits in shared memory and fed to the ordinary `mma` — the tensor cores
never know, and the prompt path keeps bf16 speed on 77% of the bytes. A second
path (`RECLANK_GPU_BF12_TC`, kernel `k_kq_bf12_tc`) uses tf32 `mma` with `x`
split as hi + lo for decode-sized batches.

### 3.5 Integration notes (GGUF / runtime)

- **GGUF type 57** (`BF12` in `reclank/formats/gguf.py`), row bytes
  `pad16(cols + cols/2 + cols/32)`. Files are written by
  `scripts/convert_q8_gguf.py --dense bf12` and read by any loader that knows
  the type; `scripts/check_bf12.py` verifies every tensor of a converted file.
- **In-memory CPU form** `KQ_BF12X16 = 63`: 16-row tiles holding the rows' `lo`
  (512 B), `hi` (256 B) and `E` (16 B) per 32-column block, so the hot loop
  streams one contiguous region per tile.
- **Conversion happens at load** (`RECLANK_DENSE=bf12`, or `--dense bf12` at
  conversion time) through `cops.kq_bf16_to_bf12`, which returns the packed
  bytes *and* the safety report of §2.6.
- Requirements: `cols % 32 == 0`, more than one row; anything else (and any
  matrix failing the safety rule) stays bf16.

---

## 4. Why this works: the information-theoretic view

The exponent of a bf16 weight is *almost free* information: measured entropy of
the exponent field in these checkpoints is **2.55–2.91 bits**, not 8 — because
weights cluster in a handful of binades. Lossless entropy coders (DFloat11,
ZipNN) exploit exactly this and reach ~11 bits per weight. But entropy codes
are variable-length: random access dies, and the decode needs lookup tables and
position scans, which is why those formats are for *storage and transfer*, not
for the multiply loop.

BF12 makes the opposite trade: it pays a **fixed 4 bits** for the exponent
instead of its ~2.7 bits of entropy — about 1.25 bits per weight more than an
entropy coder, ~10% of size — and buys:

1. **O(1) addressing.** Row `r` starts at `r · row_bytes`. Always.
2. **Branch-free SIMD decode.** Five logic ops and a masked move per value.
3. **Bandwidth-bound products.** The measured kernels hit the same GB/s as
   plain bf16 kernels (373–391 GB/s of 448 peak on the test GPU; 41–47 GB/s on
   the test Xeon) — the time follows the bytes, and there are 23% fewer bytes.

Why not block floating point (one shared exponent, shifted mantissas, as in
MSFP/Flexpoint)? Because the small values of a block lose their low mantissa
bits to the shift — a plain 12-bit fixed-point block format measured only
−69.9 dB (their RQ12 variant), hundreds of times worse than BF12. Keeping a
*per-value exponent* is precisely what preserves exactness; the group exponent
only removes the *common* part. This is the same argument AdaptivFloat's authors
make, with a per-group-of-32 bias in place of their per-layer one.

And why not OCP MX (shared microexponents)? Same shape — BF12's `E` is an E8M0
scale — but MX elements are capped at ≤8 bits for low-bit inference, so the
mantissa gets rounded. BF12 widens the element (4+7 = 11 bits + sign) until the
bf16 mantissa fits *exactly*. It is microscaling tuned to "lossless on bf16
weights" instead of "smallest possible."

---

## 5. Results

All numbers below are from the project's own measurements (Xeon W-2295, 18
cores; RTX 5060 Ti; Qwen3.8 checkpoint matrices; one-token decode with float
`x`, `x` Gaussian with 8 outlier channels ×20 as a real residual stream looks).

### 5.1 Accuracy (output error vs the exact bf16 product, float64 reference)

| form | bits/value | output error |
|---|---|---|
| Q8_0 (runtime, int8 x) | 8.5 | −39.3 dB |
| RQ8_0 (runtime, int8 x) | 8.5 | −43.7 dB |
| bf16 (float x) | 16 | −135.3 dB |
| **BF12 (float x, neg0)** | **12.25** | **−124.6 dB** |
| RQ12 (rotated, int16 x) | 12.5 | −69.9 dB |
| RQ10 (rotated, int16 x) | 10.5 | −58.0 dB |

BF12 sits two orders of magnitude in energy away from every 8–12-bit *lossy*
format, and its only gap to bf16 is the flushed dust of §2.5. (RQ12 — a 12.5-bit
integer format — was implemented, measured, and dropped: BF12 is smaller *and*
exact.)

### 5.2 Speed, GPU (qkv 10240 × 2560, DRAM-resident, copies cycled)

| kernel | bytes | µs | GB/s |
|---|---|---|---|
| bf16 | 52.4 MB | 134 | ~391 |
| **BF12** | **40.1 MB** | **105–107** | **~378** |
| Q8_0 | 27.9 MB | 73 | ~382 |

Every form reads at the same bandwidth — the time follows the bytes. **BF12 is
~20% faster than bf16 with literally unchanged values.**

### 5.3 Speed, CPU

From DRAM (memory-bound): BF12 is **24–29% faster than a bf16 kernel**
(qkv: 855–938 µs vs 1125–1203 µs). In cache (small matrices like v_proj), the
decode select costs a little: BF12 ≈ bf16 (19 µs vs 18 µs) — the documented
risk, and the reason `gap15`/`none` exist as fallback rules.

One subtlety the measurements exposed: the runtime's int8-`x` CPU kernels cap
*any* weight format at −37 to −44 dB output error (the `x` quantization
dominates), so a finer-than-8-bit weight form only pays when `x` is float or
int16. The BF12 CPU kernels therefore take float `x`, exactly as the existing
bf16 kernels do.

### 5.4 End-to-end, on real models

- **Model output:** dense-BF12 vs dense-bf16 on a chat transcript: KL under
  10<sup>−5</sup>, same top tokens everywhere — i.e., indistinguishable.
- **Size:** Qwen3.8 dense part 9.9 GB (bf16) → **7.6 GB (BF12)** — the 23%
  theoretical saving shows up as 23–28% in practice depending on which tensors
  convert and how the rows pad; on Qwen3.6, 2.7 GB → ~2.1 GB. On the GPU, the 2.3 GB freed per model goes straight to
  hot MoE experts, which is where decode speed really lives.
- **Decode step:** with dense bf16, dense products cost ~8 ms of a 35 ms step;
  BF12 removes ~20% of that (~1.6 ms) *and* the memory.
- **Per-tensor audit** (Qwen3.8, 511 dense matrices): 234,686 values zeroed out
  of ~5 × 10<sup>9</sup> (< 0.005%), largest zeroed value 3.0 × 10<sup>−5</sup>,
  zero collisions 1–7 per matrix, no Inf/NaN, no matrix rejected to bf16.
- BF12 is now the **default** dense/expert/ngram type for converted files in
  this runtime (`--dense-type bf12`), and is used for the embeddings and head of
  DeepSeek-V4.1-Flash and for Qwen3.8's shared experts.

---

## 6. Limitations

- **Not bit-exact.** 0.005–0.008% of weights become zero; results differ from a
  bf16 run at −122 to −129 dB until (if ever) the version-2 side table is used.
- **`cols` must be a multiple of 32** and rows > 1; small or odd-shaped
  tensors stay bf16 (they're cheap anyway).
- **CPU decode-bound cases:** matrices small enough to live in cache see no
  speedup (the neg0 select costs up to ~23% there); the fallback zero rules
  recover most of it.
- **Tensor-core prompt path needs the fused tile loader;** without it, BF12
  prompts fall back to slower kernels. (It exists in this runtime; a new
  implementation must not skip it.)
- **It only helps bf16-sourced weights.** Its job is lossless shrinkage of
  bfloat16, not compressing fp32 or making weights small for their own sake.

## 7. Related work, and where BF12 sits

| approach | bits | exact? | random access? | use |
|---|---|---|---|---|
| OCP MX microscaling (Rouhani et al., ISCA 2023) | ≤8 + shared E8M0 | no | yes | low-bit inference |
| Block FP (MSFP, Flexpoint) | variable | no (small values shift out) | yes | accelerators |
| AdaptivFloat (Tambe et al., DAC 2020) | few | no | yes | per-layer bias floats |
| DFloat11 (Zhang et al., 2025, arXiv:2504.11651) | ~11 | yes | no (Huffman) | lossless storage/transfer |
| ZipNN (IBM, 2024, arXiv:2411.05239) | ~10–11 | yes | no | model-file compression |
| **BF12** | **12.25** | **99.99%+ bit-exact** | **yes (fixed width)** | **in-place products** |

The plan's own literature search found no published format that is exactly this
one — an E4M7-style element under a power-of-two group exponent, sized so that
bfloat16 weights pass through untouched. It is a deliberate, boring
engineering point in the design space: *spend 1.25 bits more than entropy to
keep the multiply kernel stupid-fast.*

## 8. Conclusion

BF12 answers a narrow, practical question — "our dense weights must stay
bfloat16; can they still be smaller and faster?" — with an unusually clean
yes. The format is 49 bytes per 32 weights: sign and mantissa verbatim, a
4-bit gap under a shared group exponent, one reserved zero pattern. It is exact
for 99.99%+ of real weights, its error on the rest is ~65 dB below bfloat16's
own rounding, its decoder is five logic ops, and because decode fuses into the
multiply loop, products simply run at bf16's bandwidth on 77% of the bytes:
20–29% faster, 23% smaller, and on real models, indistinguishable.

The deeper lesson is about *where* to spend bits. Exponent fields are
information-poor and shareable; mantissa fields are information-rich and
personal. Formats that round mantissas pay in model quality; formats that
entropy-code exponents pay in access complexity. BF12 spends its savings on the
exponent and keeps the mantissa whole — and lands in the only place where a
weight format is simultaneously lossless, fixed-width, and free to decode.

---

### Appendix A: Quick-start checklist for an implementer

1. Choose `cols % 32 == 0` matrices; skip rows ≤ 1, Inf/NaN tensors.
2. Per group of 32: `E = max(e)`; per value: `gap = E − e`; if `gap > 15` →
   zero code (`s=1, gap=15, m=0`), count it.
3. Write `lo`, pack nibbles (`value j` → low nibble of byte `16g + (j mod 16)`
   for `j < 16`, high nibble otherwise), write `E`; pad row to 16 B.
4. Emit the report (flushed count, largest flushed, collisions, Inf/NaN);
   reject the matrix to bf16 if the largest flushed value exceeds 2<sup>−8</sup>
   of the tensor RMS.
5. Decode = assemble bits, `e = E − gap`, apply the zero mask. Decode *inside*
   your dot-product loop, never into a scratch buffer.
6. Test: round-trip every tensor bit-for-bit except the reported entries;
   check model KL against a bf16 run (expect < 10<sup>−5</sup>).

### Appendix B: File map (this repository)

| file | role |
|---|---|
| `reclank/plan-scripts/BF12_PLAN.md` | the format plan, measurements, risks |
| `reclank/plan-scripts/fmt_make_data.py` | reference packer (`bf12_rows`) + test data |
| `reclank/plan-scripts/fmt_cpu.c` / `fmt_gpu.cu` | AVX-512 / CUDA test kernels (the reference speed) |
| `reclank/plan-scripts/bf12_zero_*.{py,c,cu}` | the three zero rules and their timings |
| `reclank/reclank/backends/cpu/csrc/kernels/kq/convert.c` | runtime `rc_kq_bf16_to_bf12` / `rc_kq_bf12_to_bf16` + report |
| `reclank/reclank/formats/gguf.py` | GGUF type 57, row bytes, dequant rows |
| `reclank/scripts/check_bf12.py`, `check_bf12_gpu.py` | per-tensor and product verification |
| `reclank/QWEN_DENSE_QUANT.md` | per-matrix type selection using BF12 |
