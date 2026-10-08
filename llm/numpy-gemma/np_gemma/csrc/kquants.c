/* The GGUF block formats of llama.cpp: products for one token, a small
 * group, and the experts of a MoE layer.
 *
 * This file is part of the cops library: bf16_linear.c includes it, and it
 * uses the helpers there (fp16_to_f32) and in moe.c.
 *
 * The formats (the numbers are the ggml types):
 *
 *     KQ_F32   0   float32 rows.
 *     KQ_Q8_0  8   blocks of 32: d (fp16), 32 int8. w = d * q.
 *     KQ_Q4_K 12   blocks of 256 (144 bytes): d, dmin (fp16), 12 bytes of
 *                  6-bit scales and mins for 8 parts of 32, 128 bytes of
 *                  4-bit values. w = d * sc * q - dmin * m.
 *     KQ_Q5_K 13   Q4_K with a fifth bit (32 more bytes, qh).
 *     KQ_Q5_1  7   blocks of 32 (24 bytes): d, m (fp16), 32 fifth bits,
 *                  16 bytes of 4-bit values. w = d * q + m.
 *     KQ_IQ4_NL 20 blocks of 32 (18 bytes): d, 16 bytes of 4-bit codes into
 *                  a table of 16 values (only kq_rows: the n-gram table).
 *     KQ_Q6_K 14   blocks of 256 (210 bytes): 128 bytes of the low 4 bits,
 *                  64 bytes of the high 2 bits, 16 int8 scales (one for
 *                  each 16 values), d. w = d * sc * (q - 32).
 *     KQ_BF16 30   bfloat16 rows (the safetensors checkpoints).
 *     KQ_Q8X16 60  Q8_0 in groups of 16 rows (this runtime; kq_pack_q8x16 makes
 *                  it from Q8_0 for the CPU): for each group and each block
 *                  of 32 columns, 576 bytes: the 16 fp16 scales of the rows,
 *                  the 16 sums of the int8 values of the block of each row
 *                  (int16), then 8 steps of 64 bytes; step k holds values
 *                  4k to 4k + 3 of the 16 rows (row r in bytes 4r to 4r + 3).
 *                  A lane of vpdpbusd is a row: no sums across lanes, and an
 *                  int32 sum for each block (see kq_x16_body).
 *     KQ_NV4  51   NVFP4 in rows for this runtime (np_gemma/st_qwen4.py
 *                  repacks the checkpoint of NVIDIA ModelOpt). A row: the
 *                  4-bit E2M1 codes (cols / 2 bytes; block b of 32 values in
 *                  bytes 16b .. 16b + 15, value j in the low 4 bits of byte
 *                  j, value j + 16 in the high 4 bits), the E4M3 scales
 *                  (cols / 16 bytes; 2 for each block: values 0-15, 16-31),
 *                  the float32 scale of the matrix, and zeros to a multiple
 *                  of 16 bytes. So the codes of a row start at a multiple of
 *                  16 bytes, and the scales at a multiple of 4 (the loads of
 *                  the GPU; kq_nv4_* give the places). w = g * s *
 *                  e2m1(code); 2 e2m1 is an integer (0, 1, 2, 3, 4, 6, 8,
 *                  12 and the negatives).
 *     KQ_BF16X16 61, KQ_F32X16 62  bfloat16 or float32 in groups of 16 rows
 *                  (this runtime, only in memory; kq_pack_x16f makes them
 *                  for the CPU): for each column, the values of the 16 rows
 *                  (32 or 64 bytes). The rows go to a multiple of 16 with
 *                  zeros. A lane is a row, x is float32 (not int8), and a
 *                  step is one fma for each token (kq_x16f_body). Only for
 *                  64 rows or more (Qwen4CPU.KP): the smaller matrices stay
 *                  in rows, on tasks of a row and 16 tokens.
 *     KQ_Q4X  54   Q4_0 in groups of 16 rows (this runtime, only in memory;
 *                  kq_q4x_pack makes it from the int4 matrices of the Gemma 4
 *                  26B: Q4_0 blocks and float32 scales). For each block of 32
 *                  columns, 288 bytes: the 8 steps of the codes of KQ_NVX (q
 *                  = w + 8, 0 to 15) and the float16 scales of the 16 rows.
 *                  A "row" is cols / 32 * 18 bytes.
 *     KQ_NVX  53   NVFP4 in groups of 16 rows (this runtime; the GGUF of
 *                  scripts/convert_nvfp4_gguf.py; kq_nvx_pack). A group: 16
 *                  bytes (the float32 scale of the matrix, zeros), then for
 *                  each block of 32 columns 288 bytes: 8 steps of 32 bytes
 *                  and 32 E4M3 scales. Step s holds values 4s to 4s + 3 of
 *                  the 16 rows: byte 4r + u (r < 8) has value 4s + u of row
 *                  r in its low 4 bits, and that of row r + 8 in its high 4
 *                  bits. The scales: values 0-15 of rows 0 to 15, then
 *                  values 16-31. The CPU: a step is one vpdpbusd, a lane for
 *                  each row (kq_nvx_rows). The GPU: the 4 bytes of a lane
 *                  are a fragment of the tensor cores for 2 rows. A "row" is
 *                  cols / 32 * 18 + 1 bytes (a group is 16 of them).
 *
 * The products quantize x to int8 in its natural order, with one scale xs
 * for each 32 values and xm, xs times the sum of the int8 values of each 16
 * values
 * (kq_quant_x). The same x serves all the formats. For each part of a
 * block:
 *
 *     Q8_0:  y += d * xs * sum(q * xq)
 *     Q4_K:  y += d * sc * xs * sum(q * xq) - dmin * m * xs * sum(xq)
 *     Q6_K:  y += d * sc * xs * (sum(q * xq) - 32 * sum(xq))
 *
 * The values of the K formats are not negative, which is the operation of
 * the VNNI instruction vpdpbusd (u8 x s8 -> s32). Q8_0 is signed: the
 * product uses |q| and x with the sign of q.
 *
 * All the products of one row use one function for one token, so a token
 * gives the same bits alone and in a group (an MTP verify group).
 */
#define KQ_F32 0
#define KQ_Q8_0 8
#define KQ_Q4_K 12
#define KQ_Q5_K 13
#define KQ_Q6_K 14
#define KQ_Q5_1 7
#define KQ_IQ4_NL 20
#define KQ_BF16 30
#define KQ_NV4 51
#define KQ_Q8X16 60
#define KQ_NVX 53
#define KQ_Q4X 54
#define KQ_Q4X_BB 288
#ifndef KQ_Q4X_TB
#define KQ_Q4X_TB 16
#endif
#define KQ_BF16X16 61
#define KQ_F32X16 62
#define KQ_NVX_BB 288
/* BF12 (plan-scripts/BF12_PLAN.md; gguf.BF12): the bfloat16 values of a row
 * in groups of 32: lo[cols] (sign << 7 | the 7 bits of the mantissa), hi[cols
 * / 2] (the exponent gap E - e, 0 .. 15: byte 16 g + j has the gap of value
 * 32 g + j in its low half, of 32 g + j + 16 in its high half), E[cols / 32]
 * (the largest exponent of group g), padded to 16 bytes. Sign 1, gap 15,
 * mantissa 0 decodes to 0 (the zero code); bits sign << 15 | (E - gap) << 7
 * | mantissa else. */
#define KQ_BF12 57
#define KQ_BF12X16 63

/* The bytes of one row of cols values. */
static inline size_t kq_row_bytes(int type, int cols)
{
    switch (type) {
    case KQ_F32: return (size_t)cols * 4;
    case KQ_Q8_0: return (size_t)cols / 32 * 34;
    case KQ_Q5_1: return (size_t)cols / 32 * 24;
    case KQ_IQ4_NL: return (size_t)cols / 32 * 18;
    case KQ_Q4_K: return (size_t)cols / 256 * 144;
    case KQ_Q5_K: return (size_t)cols / 256 * 176;
    case KQ_Q6_K: return (size_t)cols / 256 * 210;
    case KQ_BF16: return (size_t)cols * 2;
    case KQ_NV4: return ((size_t)cols / 2 + (size_t)cols / 16 + 4 + 15) / 16 * 16;
    case KQ_Q8X16: return (size_t)cols / 32 * 36;      /* a group of 16 rows: 16 times this */
    case KQ_NVX: return (size_t)cols / 32 * 18 + 1;    /* a group of 16 rows: 16 times this */
    case KQ_Q4X: return (size_t)cols / 32 * 18;        /* a group of 16 rows: 16 times this */
    case KQ_BF16X16: return (size_t)cols * 2;          /* a group: 16 times this */
    case KQ_F32X16: return (size_t)cols * 4;
    case KQ_BF12: return ((size_t)cols + (size_t)cols / 2 + (size_t)cols / 32 + 15) / 16 * 16;
    case KQ_BF12X16: return (size_t)cols / 32 * 49;    /* a group of 16 rows: 16 times this */
    }
    return 0;
}

/* The bfloat16 bits of value j of a BF12 row. */
static inline uint16_t kq_bf12_bits(const uint8_t *row, int cols, int j)
{
    int g = j >> 5, v = j & 31;
    int b = row[j], gap = (row[cols + 16 * g + (v & 15)] >> (4 * (v >> 4))) & 15;
    int E = row[cols + cols / 2 + g];
    if (gap == 15 && b == 0x80) {
        return 0;
    }
    return (uint16_t)((b & 0x80) << 8 | ((E - gap) & 255) << 7 | (b & 0x7f));
}

/* Column c of a group of 16 rows of KQ_BF12X16 as floats (see kq_x16f_col). */
static void kq_bf12x16_col(const uint8_t *wg, int c, float *out)
{
    const uint8_t *blk = wg + (size_t)(c >> 5) * 784;
    int cc = c & 31;
    for (int r = 0; r < 16; ++r) {
        int b = blk[16 * cc + r], gap = (blk[512 + 16 * (cc & 15) + r] >> (4 * (cc >> 4))) & 15;
        uint32_t u = 0;
        if (!(gap == 15 && b == 0x80)) {
            u = ((uint32_t)(b & 0x80) << 24) | ((uint32_t)((blk[768 + r] - gap) & 255) << 23) |
                ((uint32_t)(b & 0x7f) << 16);
        }
        memcpy(out + r, &u, 4);
    }
}

/* A BF12 row to its bfloat16 bits (out: cols values). */
static void kq_bf12_row_bits(const uint8_t *row, int cols, uint16_t *out)
{
    for (int j = 0; j < cols; ++j) {
        out[j] = kq_bf12_bits(row, cols, j);
    }
}

/* The places in a KQ_NV4 row: the codes of block b, its two scales, and the
 * scale of the matrix. */
static inline const uint8_t *kq_nv4_codes(const uint8_t *w, int b)
{
    return w + (size_t)16 * b;
}

static inline const uint8_t *kq_nv4_scales(const uint8_t *w, int cols, int b)
{
    return w + (size_t)cols / 2 + 2 * (size_t)b;
}

static inline float kq_nv4_g(const uint8_t *w, int cols)
{
    float g;
    memcpy(&g, w + (size_t)cols / 2 + (size_t)cols / 16, 4);
    return g;
}

/* Twice the value of an E2M1 code (NVFP4): an integer. */
static const int8_t kq_e2m1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

/* An FP8 E4M3 value (the scales of NVFP4). */
static inline float kq_e4m3(uint8_t b)
{
    int e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) {
        v = (float)m * (1.f / 512.f);          /* m / 8 * 2^-6 */
    } else {
        uint32_t bits = ((uint32_t)(e - 7 + 127) << 23) | ((uint32_t)m << 20);
        memcpy(&v, &bits, 4);
    }
    return (b & 0x80) ? -v : v;
}

/* A float32 value to float16 (round to nearest even; normal and subnormal
 * values; the scales of KQ_Q4X, which are float16 values already). */
static inline uint16_t kq_f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int e = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) {
        return (uint16_t)(sign | 0x7c00 | (m ? 0x200 : 0));
    }
    if (e >= 31) {
        return (uint16_t)(sign | 0x7c00);
    }
    if (e <= 0) {
        if (e < -10) {
            return (uint16_t)sign;
        }
        m |= 0x800000;
        int shift = 14 - e;
        uint32_t h = m >> shift, rem = m & ((1u << shift) - 1), halfv = 1u << (shift - 1);
        if (rem > halfv || (rem == halfv && (h & 1))) {
            ++h;
        }
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) {
        ++h;
    }
    return (uint16_t)(sign | h);
}

/* The E4M3 values of the 256 codes (made when the library loads). */
static float kq_e4m3_tab[256];

__attribute__((constructor)) static void kq_e4m3_init(void)
{
    for (int b = 0; b < 256; ++b) {
        kq_e4m3_tab[b] = kq_e4m3((uint8_t)b);
    }
}

/* The float values of row rin (0 to 15) of the KQ_NVX group at wg. */
static void kq_nvx_values(const uint8_t *wg, int rin, int cols, float *out)
{
    static const int8_t e2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    float g;
    memcpy(&g, wg, 4);
    int sh = rin < 8 ? 0 : 4, r8 = rin % 8;
    for (int b = 0; b < cols / 32; ++b) {
        const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
        for (int v = 0; v < 32; ++v) {
            int c = (blk[32 * (v / 4) + 4 * r8 + v % 4] >> sh) & 15;
            float sc = kq_e4m3_tab[blk[256 + 16 * (v / 16) + rin]];
            out[32 * b + v] = 0.5f * g * sc * (float)e2[c];
        }
    }
}

static inline float kq_bf16(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

static inline float kq_h(const uint8_t *p)
{
    uint16_t v = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
#if defined(__F16C__)
    return _cvtsh_ss(v);
#else
    return fp16_to_f32(v);
#endif
}

/* Quantize the part g (32 values) of one row of x. */
static void kq_quant_part(const float *xr, int g, int8_t *qr, float *xs, float *xm)
{
    const float *v = xr + g * 32;
#if defined(__AVX512F__)
    /* The same values as the loops below: cvtps rounds to the nearest even,
     * as lrintf does in the default mode. */
    __m512 a = _mm512_loadu_ps(v), b = _mm512_loadu_ps(v + 16);
    float mv = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a), _mm512_abs_ps(b)));
    float iv = mv > 0.f ? 127.f / mv : 0.f;
    xs[g] = mv / 127.f;
    const __m512i lo = _mm512_set1_epi32(-127), hi = _mm512_set1_epi32(127);
    __m512i qa = _mm512_min_epi32(hi, _mm512_max_epi32(lo, _mm512_cvtps_epi32(
        _mm512_mul_ps(a, _mm512_set1_ps(iv)))));
    __m512i qb = _mm512_min_epi32(hi, _mm512_max_epi32(lo, _mm512_cvtps_epi32(
        _mm512_mul_ps(b, _mm512_set1_ps(iv)))));
    _mm_storeu_si128((__m128i *)(qr + g * 32), _mm512_cvtepi32_epi8(qa));
    _mm_storeu_si128((__m128i *)(qr + g * 32 + 16), _mm512_cvtepi32_epi8(qb));
    xm[2 * g] = xs[g] * (float)_mm512_reduce_add_epi32(qa);
    xm[2 * g + 1] = xs[g] * (float)_mm512_reduce_add_epi32(qb);
    return;
#elif defined(__AVX2__)
    {
        /* The same values as the loops below (cvtps rounds as lrintf). */
        const __m256 sign = _mm256_set1_ps(-0.0f);
        __m256 vv[4];
        __m256 mv = _mm256_setzero_ps();
        for (int k = 0; k < 4; ++k) {
            vv[k] = _mm256_loadu_ps(v + 8 * k);
            mv = _mm256_max_ps(mv, _mm256_andnot_ps(sign, vv[k]));
        }
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(mv), _mm256_extractf128_ps(mv, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        float m2 = _mm_cvtss_f32(h);
        float iv = m2 > 0.f ? 127.f / m2 : 0.f;
        xs[g] = m2 / 127.f;
        const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
        __m256i qi[4];
        for (int k = 0; k < 4; ++k) {
            qi[k] = _mm256_min_epi32(hi, _mm256_max_epi32(lo, _mm256_cvtps_epi32(
                _mm256_mul_ps(vv[k], _mm256_set1_ps(iv)))));
        }
        __m256i c = _mm256_packs_epi16(_mm256_packs_epi32(qi[0], qi[1]), _mm256_packs_epi32(qi[2], qi[3]));
        c = _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *)(qr + g * 32), c);
        __m256i s01 = _mm256_add_epi32(qi[0], qi[1]), s23 = _mm256_add_epi32(qi[2], qi[3]);
        __m128i a = _mm_add_epi32(_mm256_castsi256_si128(s01), _mm256_extracti128_si256(s01, 1));
        __m128i b2 = _mm_add_epi32(_mm256_castsi256_si128(s23), _mm256_extracti128_si256(s23, 1));
        a = _mm_hadd_epi32(a, a);
        a = _mm_hadd_epi32(a, a);
        b2 = _mm_hadd_epi32(b2, b2);
        b2 = _mm_hadd_epi32(b2, b2);
        xm[2 * g] = xs[g] * (float)_mm_cvtsi128_si32(a);
        xm[2 * g + 1] = xs[g] * (float)_mm_cvtsi128_si32(b2);
        return;
    }
#endif
    float m = 0.f;
    for (int i = 0; i < 32; ++i) {
        float a = fabsf(v[i]);
        m = a > m ? a : m;
    }
    float inv = m > 0.f ? 127.f / m : 0.f;
    xs[g] = m / 127.f;
    int32_t s0 = 0, s1 = 0;
    for (int i = 0; i < 32; ++i) {
        int q = (int)lrintf(v[i] * inv);
        int8_t qv = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
        qr[g * 32 + i] = qv;
        if (i < 16) {
            s0 += qv;
        } else {
            s1 += qv;
        }
    }
    /* The sums times the scale: the terms of the mins (Q4_K, Q5_K) and of
     * the offset 32 (Q6_K) are then products with the scales of the row. */
    xm[2 * g] = xs[g] * (float)s0;
    xm[2 * g + 1] = xs[g] * (float)s1;
}

/* Quantize t rows of x (cols % 32 == 0), inside a parallel region: xq (t x
 * cols), xs (t x cols / 32), xm (t x cols / 16). */
static void kq_quant_body(const float *x, int t, int cols, int8_t *xq, float *xs, float *xm)
{
    int np = cols / 32;
    if (t * np <= quant_single_max()) {
        #pragma omp single
        for (int x2 = 0; x2 < t * np; ++x2) {
            int r = x2 / np, g = x2 % np;
            kq_quant_part(x + (size_t)r * cols, g, xq + (size_t)r * cols, xs + (size_t)r * np,
                          xm + (size_t)r * 2 * np);
        }
        return;
    }
    #pragma omp for schedule(static)
    for (int x2 = 0; x2 < t * np; ++x2) {
        int r = x2 / np, g = x2 % np;
        kq_quant_part(x + (size_t)r * cols, g, xq + (size_t)r * cols, xs + (size_t)r * np,
                      xm + (size_t)r * 2 * np);
    }
}

/* kq_quant_body for the rows j with a selected expert: ids[j * k + s] >= 0
 * for some s. The other rows stay as they are: kq_moe_body does not read
 * them (the CPU part of a mixed group of the GPU, ModelGPU). */
static void kq_quant_rows_body(const float *x, int t, int cols, int8_t *xq, float *xs,
                               float *xm, const int32_t *ids, int k)
{
    int np = cols / 32;
    if (t * np <= quant_single_max()) {
        #pragma omp single
        for (int x2 = 0; x2 < t * np; ++x2) {
            int r = x2 / np, g = x2 % np, live = 0;
            for (int s = 0; s < k; ++s) {
                live |= ids[(size_t)r * k + s] >= 0;
            }
            if (live) {
                kq_quant_part(x + (size_t)r * cols, g, xq + (size_t)r * cols,
                              xs + (size_t)r * np, xm + (size_t)r * 2 * np);
            }
        }
        return;
    }
    #pragma omp for schedule(static)
    for (int x2 = 0; x2 < t * np; ++x2) {
        int r = x2 / np, g = x2 % np, live = 0;
        for (int s = 0; s < k; ++s) {
            live |= ids[(size_t)r * k + s] >= 0;
        }
        if (live) {
            kq_quant_part(x + (size_t)r * cols, g, xq + (size_t)r * cols, xs + (size_t)r * np,
                          xm + (size_t)r * 2 * np);
        }
    }
}

void kq_quant_x(const float *x, int t, int cols, int8_t *xq, float *xs, float *xm)
{
    #pragma omp parallel
    kq_quant_body(x, t, cols, xq, xs, xm);
}

/* The 6-bit scale and min of part j of a Q4_K or Q5_K block (ggml,
 * get_scale_min_k4). */
static inline void kq_scale_min(const uint8_t *q, int j, int *sc, int *m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* The float values of block b of a row (the reference, and the path
 * without VNNI). out gets 32 (Q8_0) or 256 values. */
static const int8_t kq_iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                            1, 13, 25, 38, 53, 69, 89, 113};

/* The values of a block of 32 of a type of blocks of 32. */
static inline int kq_block32(int type)
{
    return type == KQ_Q8_0 || type == KQ_Q5_1 || type == KQ_IQ4_NL || type == KQ_BF16 ||
           type == KQ_NV4;
}

static void kq_block_values(int type, const uint8_t *row, int cols, int b, float *out)
{
    if (type == KQ_BF16) {
        const uint16_t *h = (const uint16_t *)row + (size_t)b * 32;
        for (int i = 0; i < 32; ++i) {
            out[i] = kq_bf16(h[i]);
        }
    } else if (type == KQ_NV4) {
        float g = kq_nv4_g(row, cols);
        const uint8_t *q = kq_nv4_codes(row, b), *sc = kq_nv4_scales(row, cols, b);
        float s0 = 0.5f * g * kq_e4m3(sc[0]), s1 = 0.5f * g * kq_e4m3(sc[1]);
        for (int j = 0; j < 16; ++j) {
            out[j] = s0 * (float)kq_e2m1x2[q[j] & 15];
            out[j + 16] = s1 * (float)kq_e2m1x2[q[j] >> 4];
        }
    } else if (type == KQ_Q5_1) {
        const uint8_t *blk = row + (size_t)b * 24;
        float d = kq_h(blk), m = kq_h(blk + 2);
        uint32_t qh;
        memcpy(&qh, blk + 4, 4);
        for (int j = 0; j < 16; ++j) {
            int lo = (blk[8 + j] & 15) | (((qh >> j) & 1) << 4);
            int hi = (blk[8 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
            out[j] = d * (float)lo + m;
            out[j + 16] = d * (float)hi + m;
        }
    } else if (type == KQ_IQ4_NL) {
        const uint8_t *blk = row + (size_t)b * 18;
        float d = kq_h(blk);
        for (int j = 0; j < 16; ++j) {
            out[j] = d * (float)kq_iq4nl_values[blk[2 + j] & 15];
            out[j + 16] = d * (float)kq_iq4nl_values[blk[2 + j] >> 4];
        }
    } else if (type == KQ_Q8_0) {
        const uint8_t *blk = row + (size_t)b * 34;
        float d = kq_h(blk);
        for (int i = 0; i < 32; ++i) {
            out[i] = d * (float)(int8_t)blk[2 + i];
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        int five = type == KQ_Q5_K;
        const uint8_t *blk = row + (size_t)b * (five ? 176 : 144);
        float d = kq_h(blk), dm = kq_h(blk + 2);
        const uint8_t *qh = blk + 16, *qs = blk + (five ? 48 : 16);
        for (int c = 0; c < 4; ++c) {
            int s0, m0, s1, m1;
            kq_scale_min(blk + 4, 2 * c, &s0, &m0);
            kq_scale_min(blk + 4, 2 * c + 1, &s1, &m1);
            for (int l = 0; l < 32; ++l) {
                int lo = qs[32 * c + l] & 15, hi = qs[32 * c + l] >> 4;
                if (five) {
                    lo |= ((qh[l] >> (2 * c)) & 1) << 4;
                    hi |= ((qh[l] >> (2 * c + 1)) & 1) << 4;
                }
                out[64 * c + l] = d * s0 * lo - dm * m0;
                out[64 * c + 32 + l] = d * s1 * hi - dm * m1;
            }
        }
    } else if (type == KQ_Q6_K) {
        const uint8_t *blk = row + (size_t)b * 210;
        float d = kq_h(blk + 208);
        const int8_t *sc = (const int8_t *)(blk + 192);
        for (int h = 0; h < 2; ++h) {
            const uint8_t *ql = blk + 64 * h, *qh = blk + 128 + 32 * h;
            for (int l = 0; l < 32; ++l) {
                int q1 = (ql[l] & 15) | (((qh[l] >> 0) & 3) << 4);
                int q2 = (ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4);
                int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                float *o = out + 128 * h;
                o[l] = d * sc[8 * h + l / 16] * (q1 - 32);
                o[l + 32] = d * sc[8 * h + 2 + l / 16] * (q2 - 32);
                o[l + 64] = d * sc[8 * h + 4 + l / 16] * (q3 - 32);
                o[l + 96] = d * sc[8 * h + 6 + l / 16] * (q4 - 32);
            }
        }
    }
}

#if defined(__AVX512VNNI__)
static inline __attribute__((always_inline)) __m512i kq_two256(const int8_t *a, const int8_t *b)
{
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a)),
                              _mm256_loadu_si256((const __m256i *)b), 1);
}

static float kq_dot_f32(const float *w, int cols, const float *x)
{
    __m512 acc = _mm512_setzero_ps();
    for (int c = 0; c < cols; c += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + c), _mm512_loadu_ps(x + c), acc);
    }
    return _mm512_reduce_add_ps(acc);
}
#endif

#if defined(__AVX512VNNI__)
/* The scales of one row that do not depend on x: ds gets d (Q8_0), d * sc
 * (Q4_K, Q5_K: one for each 32 values; Q6_K: one for each 16), and dm gets
 * dmin * m (Q4_K, Q5_K). */
static void kq_row_scales(const uint8_t *w, int type, int cols, float *ds, float *dm)
{
    if (type == KQ_Q8_0) {
        for (int i = 0; i < cols / 32; ++i) {
            ds[i] = kq_h(w + (size_t)i * 34);
        }
    } else if (type == KQ_Q5_1) {
        /* w = d q + m: the term m sum(x) is the mins term of Q4_K with -m. */
        for (int i = 0; i < cols / 32; ++i) {
            ds[i] = kq_h(w + (size_t)i * 24);
            dm[i] = -kq_h(w + (size_t)i * 24 + 2);
        }
    } else if (type == KQ_Q6_K) {
        for (int b = 0; b < cols / 256; ++b) {
            const uint8_t *blk = w + (size_t)b * 210;
            __m512 sc = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128((const __m128i *)(blk + 192))));
            _mm512_storeu_ps(ds + 16 * b, _mm512_mul_ps(_mm512_set1_ps(kq_h(blk + 208)), sc));
        }
    } else {
        size_t bs = type == KQ_Q5_K ? 176 : 144;
        for (int b = 0; b < cols / 256; ++b) {
            const uint8_t *blk = w + (size_t)b * bs;
            /* The 6-bit scales and mins with 32-bit masks (as ggml): the
             * bytes of u are the 8 scales, then the 8 mins. */
            uint32_t u[4];
            memcpy(u, blk + 4, 12);
            const uint32_t k1 = 0x3f3f3f3f, k2 = 0x0f0f0f0f, k3 = 0x03030303;
            u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
            uint32_t mins = u[1] & k1;
            u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
            u[2] = mins;
            u[0] &= k1;
            __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)u)));
            __m512 dd = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(kq_h(blk)),
                                             _mm512_set1_ps(kq_h(blk + 2)));
            v = _mm512_mul_ps(dd, v);
            /* lanes 0 .. 7 to ds, 8 .. 15 to dm (masked stores) */
            _mm512_mask_storeu_ps(ds + 8 * b, 0x00ff, v);
            _mm512_mask_storeu_ps(dm + 8 * b - 8, 0xff00, v);
        }
    }
}

/* The scales of one row on one token: S gets the scale of each part (ds *
 * xs, in the order of the parts), and the return value is the term that
 * the product subtracts at the end: the mins of Q4_K and Q5_K (dm times
 * xm), 32 times the sums of Q6_K (ds times xm). kq_dot1 and the tiles both
 * use it, so they give the same bits. */
static float kq_prep(int type, int cols, const float *ds, const float *dm, const float *xs,
                     const float *xm, float *S)
{
    /* two lanes for each value of 8 (a scale of each 32 for 2 sums of 16) */
    const __m512i twice = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    if (type == KQ_Q6_K) {
        __m512 macc = _mm512_setzero_ps();
        for (int b = 0; b < cols / 256; ++b) {
            __m512 dv = _mm512_loadu_ps(ds + 16 * b);
            __m512 xv = _mm512_permutexvar_ps(twice, _mm512_maskz_loadu_ps(0xff, xs + 8 * b));
            _mm512_storeu_ps(S + 16 * b, _mm512_mul_ps(dv, xv));
            macc = _mm512_fmadd_ps(dv, _mm512_loadu_ps(xm + 16 * b), macc);
        }
        return 32.f * _mm512_reduce_add_ps(macc);
    }
    int np = cols / 32;
    for (int j = 0; j < np; j += 16) {
        __mmask16 m = np - j >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << (np - j)) - 1);
        _mm512_mask_storeu_ps(S + j, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, ds + j),
                                                      _mm512_maskz_loadu_ps(m, xs + j)));
    }
    if (type == KQ_Q8_0) {
        return 0.f;
    }
    __m512 macc = _mm512_setzero_ps();
    for (int j = 0; j < np; j += 8) {
        __mmask16 m = np - j >= 8 ? (__mmask16)0xffff : (__mmask16)((1u << (2 * (np - j))) - 1);
        __m512 dv = _mm512_permutexvar_ps(twice, _mm512_maskz_loadu_ps(np - j >= 8 ? (__mmask16)0xff : (__mmask16)((1u << (np - j)) - 1), dm + j));
        macc = _mm512_fmadd_ps(dv, _mm512_maskz_loadu_ps(m, xm + 2 * j), macc);
    }
    return _mm512_reduce_add_ps(macc);
}

/* The 4-bit (and fifth bit) values of part pair p of a Q4_K or Q5_K block,
 * in the lanes of kq_dot_q45k. */
static inline __attribute__((always_inline)) void kq_q45_values(const uint8_t *blk, int five, int p, __m512i *lo, __m512i *hi)
{
    const __m512i m4 = _mm512_set1_epi8(0x0f), one = _mm512_set1_epi8(1);
    const uint8_t *qs = blk + (five ? 48 : 16);
    __m512i wv = _mm512_loadu_si512((const void *)(qs + 64 * p));
    __m512i l = _mm512_and_si512(wv, m4), h = _mm512_and_si512(_mm512_srli_epi16(wv, 4), m4);
    if (five) {
        __m512i qhv = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 16)));
        __m512i c = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(4 * p),
                                            _mm512_set1_epi16(4 * p + 2));
        __m512i c1 = _mm512_add_epi16(c, _mm512_set1_epi16(1));
        l = _mm512_or_si512(l, _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(qhv, c), one), 4));
        h = _mm512_or_si512(h, _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(qhv, c1), one), 4));
    }
    *lo = l;
    *hi = h;
}

/* The 6-bit values of half h of a Q6_K block, in the lanes of kq_dot_q6k. */
static inline __attribute__((always_inline)) void kq_q6_values(const uint8_t *blk, int h, __m512i *A, __m512i *B)
{
    const __m512i m4 = _mm512_set1_epi8(0x0f), three = _mm512_set1_epi8(3);
    const __m512i cA = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(0),
                                               _mm512_set1_epi16(2));
    const __m512i cB = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(4),
                                               _mm512_set1_epi16(6));
    __m512i wv = _mm512_loadu_si512((const void *)(blk + 64 * h));
    __m512i qhv = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 128 + 32 * h)));
    *A = _mm512_or_si512(_mm512_and_si512(wv, m4), _mm512_slli_epi16(
        _mm512_and_si512(_mm512_srlv_epi16(qhv, cA), three), 4));
    *B = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(wv, 4), m4), _mm512_slli_epi16(
        _mm512_and_si512(_mm512_srlv_epi16(qhv, cB), three), 4));
}


/* The 5-bit values of two Q5_1 blocks (64 values in their order). The fifth
 * bits of the two blocks are the 64 bits of a mask. */
static inline __attribute__((always_inline)) __m512i kq_q51_values(const uint8_t *b0,
                                                                   const uint8_t *b1)
{
    const __m128i m4 = _mm_set1_epi8(0x0f);
    __m128i qa = _mm_loadu_si128((const __m128i *)(b0 + 8));
    __m128i qb = _mm_loadu_si128((const __m128i *)(b1 + 8));
    __m512i v = _mm512_castsi128_si512(_mm_and_si128(qa, m4));
    v = _mm512_inserti32x4(v, _mm_and_si128(_mm_srli_epi16(qa, 4), m4), 1);
    v = _mm512_inserti32x4(v, _mm_and_si128(qb, m4), 2);
    v = _mm512_inserti32x4(v, _mm_and_si128(_mm_srli_epi16(qb, 4), m4), 3);
    uint32_t ha, hb;
    memcpy(&ha, b0 + 4, 4);
    memcpy(&hb, b1 + 4, 4);
    __mmask64 hi = (__mmask64)ha | ((__mmask64)hb << 32);
    return _mm512_mask_or_epi32(v, 0xffff, v, _mm512_maskz_mov_epi8(hi, _mm512_set1_epi8(16)));
}

/* The products of one row on one token. S and corr come from kq_prep, as
 * in the tiles. */
static float kq_dot_q5_1(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < cols / 32; i += 2) {
        const uint8_t *b0 = w + (size_t)i * 24;
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), kq_q51_values(b0, b0 + 24),
                                         _mm512_loadu_si512((const void *)(xq + (size_t)i * 32)));
        __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(S[i]), _mm512_set1_ps(S[i + 1]));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static float kq_dot_q8_0(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < cols / 32; i += 2) {
        const uint8_t *b0 = w + (size_t)i * 34, *b1 = b0 + 34;
        __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b1 + 2));
        __m512i xv = _mm512_loadu_si512((const void *)(xq + (size_t)i * 32));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(wv), sx);
        __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(S[i]), _mm512_set1_ps(S[i + 1]));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static float kq_dot_q45k(const uint8_t *w, int cols, int five, const int8_t *xq, const float *S)
{
    size_t bs = five ? 176 : 144;
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 256; ++b) {
        const float *f = S + 8 * b;
        for (int p = 0; p < 2; ++p) {
            /* 64 bytes: parts 4p (low 4 bits of the first 32 bytes), 4p + 1
             * (their high 4 bits), 4p + 2 and 4p + 3 (the next 32 bytes). */
            __m512i lo, hi;
            kq_q45_values(w + (size_t)b * bs, five, p, &lo, &hi);
            const int8_t *xb = xq + (size_t)b * 256 + 128 * p;
            __m512i ia = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, kq_two256(xb, xb + 64));
            __m512i ib = _mm512_dpbusd_epi32(_mm512_setzero_si512(), hi,
                                             kq_two256(xb + 32, xb + 96));
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia),
                                  _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(f[4 * p]),
                                                       _mm512_set1_ps(f[4 * p + 2])), acc);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ib),
                                  _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(f[4 * p + 1]),
                                                       _mm512_set1_ps(f[4 * p + 3])), acc);
        }
    }
    return _mm512_reduce_add_ps(acc);
}

/* NP_GEMMA_KQ_PF: software prefetch this many bytes ahead in the row loops
 * of the single-token products of Q6_K and Q8_0 (the cold experts of a
 * decode step; 0: none). A thread of the CPU part read about 5 to 6 GB/s
 * (the latency of memory, not the bandwidth of the node); 1024 bytes ahead:
 * the cold experts of a step on 24 threads 484 -> 409 us (Q8_0), 408 -> 360
 * (RQ6_MIX_PLAN.md M1); on 32, 453 -> 430 and 390 -> 357. Only loads: the
 * same bits. */
static int kq_pf_bytes(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("NP_GEMMA_KQ_PF");
        n = v ? atoi(v) : 1024;
    }
    return n;
}

static float kq_dot_q6k(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    const int pf = kq_pf_bytes();
    /* lane i of a product of 64 values belongs to the part of 16 i / 4 */
    const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i iB = _mm512_add_epi32(iA, _mm512_set1_epi32(4));
    const __m512i i8 = _mm512_set1_epi32(8);
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 256; ++b) {
        if (pf) {
            const char *p = (const char *)w + (size_t)b * 210 + pf;
            _mm_prefetch(p, _MM_HINT_T0);
            _mm_prefetch(p + 64, _MM_HINT_T0);
            _mm_prefetch(p + 128, _MM_HINT_T0);
            _mm_prefetch(p + 192, _MM_HINT_T0);
        }
        __m512 fv = _mm512_loadu_ps(S + 16 * b);
        for (int h = 0; h < 2; ++h) {
            __m512i A, B;
            kq_q6_values(w + (size_t)b * 210, h, &A, &B);
            const int8_t *xb = xq + (size_t)b * 256 + 128 * h;
            __m512i pa = _mm512_dpbusd_epi32(_mm512_setzero_si512(), A,
                                             _mm512_loadu_si512((const void *)xb));
            __m512i pb = _mm512_dpbusd_epi32(_mm512_setzero_si512(), B,
                                             _mm512_loadu_si512((const void *)(xb + 64)));
            __m512i ha = h ? _mm512_add_epi32(iA, i8) : iA, hb = h ? _mm512_add_epi32(iB, i8) : iB;
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(pa), _mm512_permutexvar_ps(ha, fv), acc);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(pb), _mm512_permutexvar_ps(hb, fv), acc);
        }
    }
    return _mm512_reduce_add_ps(acc);
}

/* The most parts of a row (a part is 32 values, 16 for Q6_K), and a pad of
 * 16. 1024 parts take the down rows of the E4B and the E2B in Q6_K (10240
 * and 12288 values); with 512 they took a slow path (4.4 GB/s, not 40). */
#define KQ_S 1040

/* A tile of 4 rows (wr, with their kq_row_scales ds and dm) by 4 tokens (the prompt pass, a group of an
 * expert), for the formats with int8 x. Each (row, token) pair adds in the
 * order of kq_dot1, so the tile gives the same bits. A short tile repeats
 * its last token. out[i * ostr_r + j * ostr_t] gets row i, token j. */
static void kq_tile4(const uint8_t *const wr[4], float ds[4][KQ_S], float dm[4][KQ_S],
                     int type, int cols, const int8_t *xq, const float *xs, const float *xm,
                     int nt, float *out, size_t ostr_r, size_t ostr_t)
{
    const int8_t *xr[4];
    float S[4][4][KQ_S];
    float corr[4][4];
    for (int j = 0; j < 4; ++j) {
        int jj = j < nt ? j : nt - 1;
        xr[j] = xq + (size_t)jj * cols;
        for (int i = 0; i < 4; ++i) {
            corr[i][j] = kq_prep(type, cols, ds[i], dm[i], xs + (size_t)jj * (cols / 32),
                                 xm + (size_t)jj * (cols / 16), S[i][j]);
        }
    }
    __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00;
    __m512 a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512 a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    __m512 a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    const __m512i z = _mm512_setzero_si512();
/* acc += (w . x) * the scales of S[I][J] at base, permuted by idx */
#define KQ_ACC(A, I, J, W, X, IDX, BASE) \
    A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(z, W, X)), \
                        _mm512_permutexvar_ps(IDX, _mm512_loadu_ps(S[I][J] + (BASE))), A)
    if (type == KQ_Q8_0) {
        for (int q = 0; q < cols / 64; ++q) {
            int base = (2 * q) & ~15, o = (2 * q) & 15;
            __m512i idx = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(o),
                                                  _mm512_set1_epi32(o + 1));
            __m512i w0, w1, w2, w3;
            __mmask64 n0, n1, n2, n3;
#define KQ8_W(I) { const uint8_t *b0 = wr[I] + (size_t)q * 68; \
        __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b0 + 36)); \
        n##I = _mm512_movepi8_mask(wv); w##I = _mm512_abs_epi8(wv); }
            KQ8_W(0) KQ8_W(1) KQ8_W(2) KQ8_W(3)
#undef KQ8_W
#define KQ8_J(J, A0, A1, A2, A3) { \
        __m512i xv = _mm512_loadu_si512((const void *)(xr[J] + (size_t)q * 64)); \
        KQ_ACC(A0, 0, J, w0, _mm512_mask_sub_epi8(xv, n0, z, xv), idx, base); \
        KQ_ACC(A1, 1, J, w1, _mm512_mask_sub_epi8(xv, n1, z, xv), idx, base); \
        KQ_ACC(A2, 2, J, w2, _mm512_mask_sub_epi8(xv, n2, z, xv), idx, base); \
        KQ_ACC(A3, 3, J, w3, _mm512_mask_sub_epi8(xv, n3, z, xv), idx, base); }
            KQ8_J(0, a00, a10, a20, a30) KQ8_J(1, a01, a11, a21, a31)
            KQ8_J(2, a02, a12, a22, a32) KQ8_J(3, a03, a13, a23, a33)
#undef KQ8_J
        }
    } else if (type == KQ_Q5_1) {
        for (int q = 0; q < cols / 64; ++q) {
            int base = (2 * q) & ~15, o = (2 * q) & 15;
            __m512i idx = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(o),
                                                  _mm512_set1_epi32(o + 1));
            __m512i w0 = kq_q51_values(wr[0] + (size_t)q * 48, wr[0] + (size_t)q * 48 + 24);
            __m512i w1 = kq_q51_values(wr[1] + (size_t)q * 48, wr[1] + (size_t)q * 48 + 24);
            __m512i w2 = kq_q51_values(wr[2] + (size_t)q * 48, wr[2] + (size_t)q * 48 + 24);
            __m512i w3 = kq_q51_values(wr[3] + (size_t)q * 48, wr[3] + (size_t)q * 48 + 24);
#define KQ51_J(J, A0, A1, A2, A3) { \
        __m512i xv = _mm512_loadu_si512((const void *)(xr[J] + (size_t)q * 64)); \
        KQ_ACC(A0, 0, J, w0, xv, idx, base); KQ_ACC(A1, 1, J, w1, xv, idx, base); \
        KQ_ACC(A2, 2, J, w2, xv, idx, base); KQ_ACC(A3, 3, J, w3, xv, idx, base); }
            KQ51_J(0, a00, a10, a20, a30) KQ51_J(1, a01, a11, a21, a31)
            KQ51_J(2, a02, a12, a22, a32) KQ51_J(3, a03, a13, a23, a33)
#undef KQ51_J
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        int five = type == KQ_Q5_K;
        size_t bs = five ? 176 : 144;
        for (int b = 0; b < cols / 256; ++b) {
            int base = (8 * b) & ~15, o = (8 * b) & 15;
            for (int p = 0; p < 2; ++p) {
                __m512i ia = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(o + 4 * p),
                                                     _mm512_set1_epi32(o + 4 * p + 2));
                __m512i ib = _mm512_add_epi32(ia, _mm512_set1_epi32(1));
                __m512i l0, h0, l1, h1, l2, h2, l3, h3;
                kq_q45_values(wr[0] + (size_t)b * bs, five, p, &l0, &h0);
                kq_q45_values(wr[1] + (size_t)b * bs, five, p, &l1, &h1);
                kq_q45_values(wr[2] + (size_t)b * bs, five, p, &l2, &h2);
                kq_q45_values(wr[3] + (size_t)b * bs, five, p, &l3, &h3);
#define KQ4_J(J, A0, A1, A2, A3) { \
        const int8_t *xb = xr[J] + (size_t)b * 256 + 128 * p; \
        __m512i xa = kq_two256(xb, xb + 64), xc = kq_two256(xb + 32, xb + 96); \
        KQ_ACC(A0, 0, J, l0, xa, ia, base); KQ_ACC(A0, 0, J, h0, xc, ib, base); \
        KQ_ACC(A1, 1, J, l1, xa, ia, base); KQ_ACC(A1, 1, J, h1, xc, ib, base); \
        KQ_ACC(A2, 2, J, l2, xa, ia, base); KQ_ACC(A2, 2, J, h2, xc, ib, base); \
        KQ_ACC(A3, 3, J, l3, xa, ia, base); KQ_ACC(A3, 3, J, h3, xc, ib, base); }
                KQ4_J(0, a00, a10, a20, a30) KQ4_J(1, a01, a11, a21, a31)
                KQ4_J(2, a02, a12, a22, a32) KQ4_J(3, a03, a13, a23, a33)
#undef KQ4_J
            }
        }
    } else {
        const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
        for (int b = 0; b < cols / 256; ++b) {
            for (int h = 0; h < 2; ++h) {
                __m512i ha = _mm512_add_epi32(iA, _mm512_set1_epi32(8 * h));
                __m512i hb = _mm512_add_epi32(ha, _mm512_set1_epi32(4));
                __m512i A0, B0, A1, B1, A2, B2, A3, B3;
                kq_q6_values(wr[0] + (size_t)b * 210, h, &A0, &B0);
                kq_q6_values(wr[1] + (size_t)b * 210, h, &A1, &B1);
                kq_q6_values(wr[2] + (size_t)b * 210, h, &A2, &B2);
                kq_q6_values(wr[3] + (size_t)b * 210, h, &A3, &B3);
#define KQ6_J(J, C0, C1, C2, C3) { \
        const int8_t *xb = xr[J] + (size_t)b * 256 + 128 * h; \
        __m512i x0 = _mm512_loadu_si512((const void *)xb); \
        __m512i x1 = _mm512_loadu_si512((const void *)(xb + 64)); \
        KQ_ACC(C0, 0, J, A0, x0, ha, 16 * b); KQ_ACC(C0, 0, J, B0, x1, hb, 16 * b); \
        KQ_ACC(C1, 1, J, A1, x0, ha, 16 * b); KQ_ACC(C1, 1, J, B1, x1, hb, 16 * b); \
        KQ_ACC(C2, 2, J, A2, x0, ha, 16 * b); KQ_ACC(C2, 2, J, B2, x1, hb, 16 * b); \
        KQ_ACC(C3, 3, J, A3, x0, ha, 16 * b); KQ_ACC(C3, 3, J, B3, x1, hb, 16 * b); }
                KQ6_J(0, a00, a10, a20, a30) KQ6_J(1, a01, a11, a21, a31)
                KQ6_J(2, a02, a12, a22, a32) KQ6_J(3, a03, a13, a23, a33)
#undef KQ6_J
            }
        }
    }
#undef KQ_ACC
    __m512 acc[4][4] = {{a00, a01, a02, a03}, {a10, a11, a12, a13},
                        {a20, a21, a22, a23}, {a30, a31, a32, a33}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < nt; ++j) {
            out[(size_t)i * ostr_r + (size_t)j * ostr_t] =
                _mm512_reduce_add_ps(acc[i][j]) - corr[i][j];
        }
    }
}

/* The tiles apply to a format with int8 x and a row of at most KQ_S - 16
 * parts. */
static inline int kq_tiles(int type, int cols)
{
    int parts = type == KQ_Q6_K ? cols / 16 : cols / 32;
    return type != KQ_F32 && type != KQ_IQ4_NL && type != KQ_BF16 && type != KQ_NV4 &&
           type != KQ_BF12 && parts <= KQ_S - 16 &&
           (!kq_block32(type) || cols % 64 == 0);
}
#endif

#if defined(__AVX512VNNI__)
/* One row on one token, with the scales of the row (kq_row_scales). */
static float kq_dot_scaled(const uint8_t *w, int type, int cols, const float *ds,
                           const float *dm, const int8_t *xq, const float *xs, const float *xm)
{
    float S[KQ_S];
    float corr = kq_prep(type, cols, ds, dm, xs, xm, S);
    switch (type) {
    case KQ_Q8_0: return kq_dot_q8_0(w, cols, xq, S) - corr;
    case KQ_Q5_1: return kq_dot_q5_1(w, cols, xq, S) - corr;
    case KQ_Q4_K: return kq_dot_q45k(w, cols, 0, xq, S) - corr;
    case KQ_Q5_K: return kq_dot_q45k(w, cols, 1, xq, S) - corr;
    }
    return kq_dot_q6k(w, cols, xq, S) - corr;
}
#endif

/* One row of type type on one token: xq, xs, xm (quantized), or x for F32. */
#if defined(__AVX512VNNI__)
/* BF16 row, int8 x: sum over the blocks of xs * sum(w * xq). */
static float kq_dot_bf16(const uint8_t *w, int cols, const int8_t *xq, const float *xs)
{
    const uint16_t *h = (const uint16_t *)w;
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 32; ++b) {
        __m512i hw = _mm512_loadu_si512((const void *)(h + 32 * b));
        __m512 w0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(hw)), 16));
        __m512 w1 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(hw, 1)), 16));
        __m128i xa = _mm_loadu_si128((const __m128i *)(xq + 32 * b));
        __m128i xb = _mm_loadu_si128((const __m128i *)(xq + 32 * b + 16));
        __m512 x0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(xa));
        __m512 x1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(xb));
        __m512 p = _mm512_fmadd_ps(w1, x1, _mm512_mul_ps(w0, x0));
        acc = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[b]), acc);
    }
    return _mm512_reduce_add_ps(acc);
}

/* NVFP4 row (KQ_NV4), int8 x. For each block: the codes to int8 (twice the
 * values), vpdpbusd with |w| and x with the sign of w (8 sums of 4 values:
 * the first 4 of values 0-15, the last 4 of 16-31), each half times its
 * scale and xs. */
static float kq_dot_nv4(const uint8_t *w, int cols, const int8_t *xq, const float *xs)
{
    float g = kq_nv4_g(w, cols);
    const __m128i lut = _mm_loadu_si128((const __m128i *)kq_e2m1x2);
    const __m128i m4 = _mm_set1_epi8(15);
    const __m512i lut5 = _mm512_broadcast_i32x4(lut), m45 = _mm512_set1_epi8(15);
    /* the scale of each of the 16 sums: 4 for each half block */
    const __m512i sidx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    __m512 acc5 = _mm512_setzero_ps();
    int nb = cols / 32, b = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *s0p = kq_nv4_scales(w, cols, b);
        __m256i q01 = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(w, b));
        /* lanes of 128 bits: the low codes of block 0, its high codes, the
         * low codes of block 1, its high codes (the order of x) */
        __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q01), q01, 1);
        q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3), q);
        __m512i codes = _mm512_and_si512(
            _mm512_mask_blend_epi64(0xcc, q, _mm512_srli_epi16(q, 4)), m45);
        __m512i wv = _mm512_shuffle_epi8(lut5, codes);
        __m512i xv = _mm512_loadu_si512((const void *)(xq + 32 * b));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(wv), sx);
        __m128 s4 = _mm_setr_ps(kq_e4m3_tab[s0p[0]] * xs[b], kq_e4m3_tab[s0p[1]] * xs[b],
                                kq_e4m3_tab[s0p[2]] * xs[b + 1], kq_e4m3_tab[s0p[3]] * xs[b + 1]);
        __m512 sc = _mm512_permutexvar_ps(sidx, _mm512_castps128_ps512(s4));
        acc5 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc5);
    }
    __m256 acc = _mm256_setzero_ps();
    for (; b < nb; ++b) {
        const uint8_t *sp = kq_nv4_scales(w, cols, b);
        __m128i q = _mm_loadu_si128((const __m128i *)kq_nv4_codes(w, b));
        __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
        __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
        __m256i wv = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + 32 * b));
        __m256i sx = _mm256_sign_epi8(xv, wv);
        __m256i is = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(wv), sx);
        float s0 = kq_e4m3_tab[sp[0]] * xs[b], s1 = kq_e4m3_tab[sp[1]] * xs[b];
        __m256 sc = _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is), sc, acc);
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    return 0.5f * g * (_mm_cvtss_f32(r) + _mm512_reduce_add_ps(acc5));
}
#endif

#if defined(__AVX512VNNI__)
#define KQ_NV4_MAX 4096
static void kq_row_nv4(const uint8_t *w, int cols, const int8_t *xq, const float *xs, int n,
                       float *out, size_t ostride);
#endif

#if defined(__AVX512F__)
/* A BF12 row on float x (plan-scripts/fmt_cpu.c bf12_dot): the bfloat16 bits
 * of each group of 32 in registers, then the fma of float32. */
static float kq_dot_bf12(const uint8_t *w, int cols, const float *x)
{
    const uint8_t *lo = w, *hi = w + cols, *E = w + cols + cols / 2;
    const __m512i m7f = _mm512_set1_epi16(0x7f), m80 = _mm512_set1_epi16(0x80);
    const __m512i g15 = _mm512_set1_epi16(15), m15 = _mm512_set1_epi16(15);
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    for (int g = 0; g < cols / 32; ++g) {
        __m512i l = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(lo + 32 * g)));
        __m256i h16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(hi + 16 * g)));
        __m512i gp = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(h16, _mm512_castsi512_si256(m15))),
                                        _mm256_srli_epi16(h16, 4), 1);
        __m512i e = _mm512_sub_epi16(_mm512_set1_epi16(E[g]), gp);
        __m512i b = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(l, m80), 8),
                                                    _mm512_slli_epi16(e, 7)), _mm512_and_si512(l, m7f));
        b = _mm512_maskz_mov_epi16(~(_mm512_cmpeq_epi16_mask(gp, g15) & _mm512_cmpeq_epi16_mask(l, m80)), b);
        a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                 _mm512_castsi512_si256(b)), 16)), _mm512_loadu_ps(x + 32 * g), a0);
        a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                 _mm512_extracti64x4_epi64(b, 1)), 16)), _mm512_loadu_ps(x + 32 * g + 16), a1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}
#endif

static float kq_dot1(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                     const float *xm, const float *x)
{
    if (type == KQ_BF12) {
#if defined(__AVX512F__)
        return kq_dot_bf12(w, cols, x);
#else
        float s = 0.f;
        for (int c = 0; c < cols; ++c) {
            uint32_t u = (uint32_t)kq_bf12_bits(w, cols, c) << 16;
            float v;
            memcpy(&v, &u, 4);
            s += v * x[c];
        }
        return s;
#endif
    }
#if defined(__AVX512VNNI__)
    if (type == KQ_F32) {
        return kq_dot_f32((const float *)w, cols, x);
    }
    if (type == KQ_BF16) {
        return kq_dot_bf16(w, cols, xq, xs);
    }
    if (type == KQ_NV4) {
        /* With cols a multiple of 64, the sums and their order are those
         * of kq_row_nv4 (a group): the same bits. */
        return kq_dot_nv4(w, cols, xq, xs);
    }
    if (kq_tiles(type, cols)) {
        float ds[KQ_S], dm[KQ_S];
        kq_row_scales(w, type, cols, ds, dm);
        return kq_dot_scaled(w, type, cols, ds, dm, xq, xs, xm);
    }
#endif
    (void)xm;
    if (type == KQ_F32) {
        float s = 0.f;
        for (int c = 0; c < cols; ++c) {
            s += ((const float *)w)[c] * x[c];
        }
        return s;
    }
    int bv = kq_block32(type) ? 32 : 256;
    float v[256], s = 0.f;
    for (int b = 0; b < cols / bv; ++b) {
        kq_block_values(type, w, cols, b, v);
        for (int i = 0; i < bv; ++i) {
            int c = b * bv + i;
            s += v[i] * xs[c / 32] * (float)xq[c];
        }
    }
    return s;
}

/* One row on n tokens. out[j * ostride] gets token j. */
#if defined(__AVX512VNNI__)
/* NVFP4 on several tokens: the row to |w| (int8), its signs, and a scale for
 * each 16 values, one time; then for each token 64 values at a time. The
 * sums are those of kq_dot_nv4 (the same int32 sums of 16 values, the same
 * scales), so a group gives the bits of steps. */
static void kq_row_nv4(const uint8_t *w, int cols, const int8_t *xq, const float *xs, int n,
                       float *out, size_t ostride)
{
    __attribute__((aligned(64))) int8_t aw[KQ_NV4_MAX];
    __attribute__((aligned(64))) float sc[KQ_NV4_MAX / 16];
    __mmask64 negs[KQ_NV4_MAX / 64];
    float g = kq_nv4_g(w, cols);
    int nb = cols / 32;
    const __m512i lut5 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1x2));
    const __m512i m45 = _mm512_set1_epi8(15);
    for (int b = 0; b < nb; b += 2) {
        /* as kq_dot_nv4: two blocks to 64 int8 in the order of x */
        const uint8_t *sp = kq_nv4_scales(w, cols, b);
        __m256i q01 = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(w, b));
        __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q01), q01, 1);
        q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3), q);
        __m512i codes = _mm512_and_si512(
            _mm512_mask_blend_epi64(0xcc, q, _mm512_srli_epi16(q, 4)), m45);
        __m512i wv = _mm512_shuffle_epi8(lut5, codes);
        negs[b / 2] = _mm512_movepi8_mask(wv);
        _mm512_store_si512((void *)(aw + 32 * b), _mm512_abs_epi8(wv));
        sc[2 * b] = kq_e4m3_tab[sp[0]];
        sc[2 * b + 1] = kq_e4m3_tab[sp[1]];
        sc[2 * b + 2] = kq_e4m3_tab[sp[2]];
        sc[2 * b + 3] = kq_e4m3_tab[sp[3]];
    }
    const __m512i sidx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    for (int j = 0; j < n; ++j) {
        const int8_t *xr = xq + (size_t)j * cols;
        const float *sr = xs + (size_t)j * nb;
        __m512 acc = _mm512_setzero_ps();
        for (int c = 0; c < cols; c += 64) {
            __m512i xv = _mm512_loadu_si512((const void *)(xr + c));
            __m512i sx = _mm512_mask_sub_epi8(xv, negs[c / 64], _mm512_setzero_si512(), xv);
            __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                                             _mm512_load_si512((const void *)(aw + c)), sx);
            int b = c / 32;
            __m128 s4 = _mm_setr_ps(sc[2 * b] * sr[b], sc[2 * b + 1] * sr[b],
                                    sc[2 * b + 2] * sr[b + 1], sc[2 * b + 3] * sr[b + 1]);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is),
                                  _mm512_permutexvar_ps(sidx, _mm512_castps128_ps512(s4)), acc);
        }
        out[(size_t)j * ostride] = 0.5f * g * _mm512_reduce_add_ps(acc);
    }
}
#endif

static void kq_row(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                   const float *xm, const float *x, int n, float *out, size_t ostride)
{
#if defined(__AVX512VNNI__)
    if (type == KQ_NV4 && n > 1 && cols % 64 == 0 && cols <= KQ_NV4_MAX) {
        kq_row_nv4(w, cols, xq, xs, n, out, ostride);
        return;
    }
    if (n > 1 && type != KQ_F32 && kq_tiles(type, cols)) {
        /* The scales of the row one time for all the tokens. */
        float ds[KQ_S], dm[KQ_S];
        kq_row_scales(w, type, cols, ds, dm);
        for (int j = 0; j < n; ++j) {
            out[(size_t)j * ostride] = kq_dot_scaled(
                w, type, cols, ds, dm, xq + (size_t)j * cols, xs + (size_t)j * (cols / 32),
                xm + (size_t)j * (cols / 16));
        }
        return;
    }
#endif
    for (int j = 0; j < n; ++j) {
        out[(size_t)j * ostride] = kq_dot1(w, type, cols, xq + (size_t)j * cols,
                                           xs + (size_t)j * (cols / 32),
                                           xm + (size_t)j * (cols / 16),
                                           x ? x + (size_t)j * cols : NULL);
    }
}

#if defined(__AVX512VNNI__)
/* Four Q8_0 rows (at w, rb bytes each) on one token: the operations of
 * kq_dot1 for each row (kq_row_scales, kq_prep, kq_dot_q8_0: the same
 * bits), the rows in one loop with their own sums. One row at a time, a
 * row of 640 values (the down rows of the experts) ran at about 3.7 GB/s of
 * a thread: a chain of 10 steps of convert and FMA, and the passes of the
 * scales; four chains in step share the loads of x and fill the core. */
static void kq_dot4_q8_0(const uint8_t *w, size_t rb, int cols, const int8_t *xq, const float *xs,
                         float *out, size_t ostride)
{
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    const __m512i z = _mm512_setzero_si512();
    const uint8_t *w0 = w, *w1 = w + rb, *w2 = w + 2 * rb, *w3 = w + 3 * rb;
    const int pf = kq_pf_bytes();
    for (int i = 0; i < cols / 32; i += 2) {
        if (pf) {
            /* the 4 rows are one span of 4 rb bytes: 68 bytes of each row a step */
            _mm_prefetch((const char *)w0 + (size_t)i * 34 + pf, _MM_HINT_T0);
            _mm_prefetch((const char *)w1 + (size_t)i * 34 + pf, _MM_HINT_T0);
            _mm_prefetch((const char *)w2 + (size_t)i * 34 + pf, _MM_HINT_T0);
            _mm_prefetch((const char *)w3 + (size_t)i * 34 + pf, _MM_HINT_T0);
        }
        __m512i xv = _mm512_loadu_si512((const void *)(xq + (size_t)i * 32));
        float x0 = xs[i], x1 = xs[i + 1];
#define KQ4_ROW(A, W) { \
        const uint8_t *b0 = W + (size_t)i * 34, *b1 = b0 + 34; \
        __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b1 + 2)); \
        __m512i sx = _mm512_mask_sub_epi8(xv, _mm512_movepi8_mask(wv), z, xv); \
        __m512i is = _mm512_dpbusd_epi32(z, _mm512_abs_epi8(wv), sx); \
        __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(kq_h(b0) * x0), \
                                         _mm512_set1_ps(kq_h(b1) * x1)); \
        A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, A); }
        KQ4_ROW(a0, w0) KQ4_ROW(a1, w1) KQ4_ROW(a2, w2) KQ4_ROW(a3, w3)
#undef KQ4_ROW
    }
    out[0] = _mm512_reduce_add_ps(a0);
    out[ostride] = _mm512_reduce_add_ps(a1);
    out[2 * ostride] = _mm512_reduce_add_ps(a2);
    out[3 * ostride] = _mm512_reduce_add_ps(a3);
}
#endif

/* Rows r .. r + 3 (at w, rb bytes each) on n tokens: tiles of 4 tokens for
 * n >= 4, else one row at a time. out[i + j * ostride] gets row r + i,
 * token j. */
#if defined(__AVX512VNNI__)
/* NVFP4, rows r .. r + 3 on n tokens (the experts of a group): the 4 rows
 * to |w| (int8), the signs, and the E4M3 scales in the order of the 16 sums
 * of a chunk of 64 values, one time; then tiles of 4 rows by 4 tokens, so a
 * chunk of a row and of a token is loaded one time for 4 products. The sums,
 * the scales, and their order are those of kq_dot_nv4: the bits of steps. */
static void kq_rows4_nv4(const uint8_t *w, size_t rb, int cols, const int8_t *xq,
                         const float *xs, int n, float *out, size_t ostride)
{
    __attribute__((aligned(64))) int8_t aw[4][KQ_NV4_MAX];
    __attribute__((aligned(64))) float wsc[4][KQ_NV4_MAX / 64][16];
    __mmask64 negs[4][KQ_NV4_MAX / 64];
    float g[4];
    int nb = cols / 32, nc = cols / 64;
    const __m512i lut5 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1x2));
    const __m512i m45 = _mm512_set1_epi8(15);
    for (int r = 0; r < 4; ++r) {
        const uint8_t *wr = w + (size_t)r * rb;
        g[r] = kq_nv4_g(wr, cols);
        for (int b = 0; b < nb; b += 2) {
            const uint8_t *sp = kq_nv4_scales(wr, cols, b);
            __m256i q01 = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(wr, b));
            __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q01), q01, 1);
            q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3), q);
            __m512i codes = _mm512_and_si512(
                _mm512_mask_blend_epi64(0xcc, q, _mm512_srli_epi16(q, 4)), m45);
            __m512i wv = _mm512_shuffle_epi8(lut5, codes);
            negs[r][b / 2] = _mm512_movepi8_mask(wv);
            _mm512_store_si512((void *)(aw[r] + 32 * b), _mm512_abs_epi8(wv));
            float s4[4] = {kq_e4m3_tab[sp[0]], kq_e4m3_tab[sp[1]], kq_e4m3_tab[sp[2]],
                           kq_e4m3_tab[sp[3]]};
            for (int l = 0; l < 16; ++l) {
                wsc[r][b / 2][l] = s4[l / 4];
            }
        }
    }
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        __m512 acc[4][4];
        for (int r = 0; r < 4; ++r) {
            for (int t = 0; t < 4; ++t) {
                acc[r][t] = _mm512_setzero_ps();
            }
        }
        for (int c = 0; c < nc; ++c) {
            __m512i xv[4];
            __m512 xsv[4];
            for (int t = 0; t < 4; ++t) {
                const float *sr = xs + (size_t)(j + t) * nb;
                xv[t] = _mm512_loadu_si512((const void *)(xq + (size_t)(j + t) * cols + 64 * c));
                xsv[t] = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sr[2 * c]),
                                              _mm512_set1_ps(sr[2 * c + 1]));
            }
            for (int r = 0; r < 4; ++r) {
                __m512i wa = _mm512_load_si512((const void *)(aw[r] + 64 * c));
                __m512 ws = _mm512_load_ps(wsc[r][c]);
                for (int t = 0; t < 4; ++t) {
                    __m512i sx = _mm512_mask_sub_epi8(xv[t], negs[r][c], _mm512_setzero_si512(), xv[t]);
                    __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wa, sx);
                    acc[r][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), _mm512_mul_ps(ws, xsv[t]),
                                                acc[r][t]);
                }
            }
        }
        for (int r = 0; r < 4; ++r) {
            for (int t = 0; t < 4; ++t) {
                out[(size_t)(j + t) * ostride + r] = 0.5f * g[r] * _mm512_reduce_add_ps(acc[r][t]);
            }
        }
    }
    for (; j < n; ++j) {                    /* the last tokens: one at a time */
        for (int r = 0; r < 4; ++r) {
            out[(size_t)j * ostride + r] = kq_dot_nv4(w + (size_t)r * rb, cols, xq + (size_t)j * cols,
                                                      xs + (size_t)j * nb);
        }
    }
}
#endif

/* The rows of a down task of the cold experts of a step (kq_moe_small_body)
 * for the types of 4-row groups: NP_GEMMA_MOE_NDN (4, 8, or 16; 16 by
 * default). A down row has 640 values: 4 rows (2.7 KB) a task left the
 * work of a task (the check of the act, the setup, the sums) a large part. */
static int kq_ndn_rows(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("NP_GEMMA_MOE_NDN");
        n = v ? atoi(v) : 16;
        if (n != 4 && n != 8 && n != 16) {
            n = 4;
        }
    }
    return n;
}

/* NP_GEMMA_Q8_4ROWS=0: kq_dot1 a row at a time for Q8_0 on 1 to 3 tokens. */
static int kq_q8_4rows(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_Q8_4ROWS");
        on = !(v && v[0] == '0');
    }
    return on;
}

#if defined(__AVX512VNNI__)
/* NP_GEMMA_KQ_TILE2 (default 1): the tiles of Q8_0 and Q6_K as
 * kq_rows4_t2; 0 keeps kq_tile4. NP_GEMMA_KQ_TILE2_MIN (2): the fewest
 * tokens for it (fewer: kq_dot4_q8_0, kq_dot1). */
static int kq_tile2_min(void)
{
    static int n = -2;
    if (n == -2) {
        const char *v = getenv("NP_GEMMA_KQ_TILE2");
        const char *m = getenv("NP_GEMMA_KQ_TILE2_MIN");
        n = v && v[0] == '0' ? -1 : (m ? atoi(m) : 2);
    }
    return n;
}

#define KQ_T2_NV 64          /* the most steps of 64 values of a row (4096 values) */

/* The scales of 4 rows for kq_t2_tile, one vector for each step of 64
 * values in the lanes of its products (dE: d for Q8_0, d * sc for Q6_K),
 * and for Q8_0 the sums of the weights times -128 (nE). ds gets the 16
 * scales of each Q6_K block (the term corr). The values of kq_row_scales. */
static void kq_t2_rows(const uint8_t *const wr[4], int type, int cols, __m512 dE[4][KQ_T2_NV],
                       __m512i nE[4][KQ_T2_NV], float ds[4][4 * KQ_T2_NV])
{
    const __m512i z = _mm512_setzero_si512(), k80 = _mm512_set1_epi8((char)0x80);
    for (int i = 0; i < 4; ++i) {
        if (type == KQ_Q8_0) {
            for (int q = 0; q < cols / 64; ++q) {
                const uint8_t *b0 = wr[i] + (size_t)q * 68;
                dE[i][q] = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(kq_h(b0)),
                                                _mm512_set1_ps(kq_h(b0 + 34)));
                __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b0 + 36));
                nE[i][q] = _mm512_sub_epi32(z, _mm512_dpbusd_epi32(z, k80, wv));
            }
        } else {
            const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
            for (int b = 0; b < cols / 256; ++b) {
                const uint8_t *blk = wr[i] + (size_t)b * 210;
                __m512 dv = _mm512_mul_ps(_mm512_set1_ps(kq_h(blk + 208)), _mm512_cvtepi32_ps(
                    _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(blk + 192)))));
                _mm512_storeu_ps(ds[i] + 16 * b, dv);
                for (int s = 0; s < 4; ++s) {
                    dE[i][4 * b + s] = _mm512_permutexvar_ps(
                        _mm512_add_epi32(iA, _mm512_set1_epi32(4 * s)), dv);
                }
            }
        }
    }
}

/* 4 rows by NT tokens (a constant), as kq_tile4: each (row, token) pair adds
 * in the order of kq_dot1 with the same scales (S = ds * xs) and integer
 * sums, so the same bits; the scales of a step are dE times the scales of
 * the token, made in the loop (no kq_prep). Q8_0: x + 128 (unsigned) times w
 * (signed), from -128 sum(w) (nE): the sums of kq_dot_q8_0 with no sign
 * steps. pf (or null): prefetch the 4 rows after these (the next task) as
 * the loop goes. */
static inline __attribute__((always_inline)) void kq_t2_tile(
    const uint8_t *const wr[4], size_t rb, int type, int cols, __m512 dE[4][KQ_T2_NV],
    __m512i nE[4][KQ_T2_NV], float ds[4][4 * KQ_T2_NV], const int8_t *xq, const float *xs,
    const float *xm, const int NT, const uint8_t *pf, float *out, size_t ostr_r, size_t ostr_t,
    const int8_t *xl)
{
    /* xl (or null): the low plane of x as int16 (kq_quant_part16): xq the high
     * one; the integer sum of a step is 128 sum(hi w) + sum(lo w), exact */
    __m512 acc[4][4];
    float corr[4][4] = {{0}};
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = _mm512_setzero_ps();
        }
    }
    const int np = cols / 32;
    const size_t span = 4 * rb;
    if (type == KQ_Q8_0) {
        const __m512i k80 = _mm512_set1_epi8((char)0x80);
        const int nq = cols / 64;
        const size_t per = (span + nq - 1) / nq;
        for (int q = 0; q < nq; ++q) {
            if (pf) {
                for (size_t o = 0; o < per + 64; o += 64) {
                    _mm_prefetch((const char *)pf + (size_t)q * per + o, _MM_HINT_T1);
                }
            }
            __m512i w[4];
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i) {
                const uint8_t *b0 = wr[i] + (size_t)q * 68;
                w[i] = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b0 + 36));
            }
#pragma GCC unroll 4
            for (int j = 0; j < NT; ++j) {
                __m512i xu = _mm512_xor_si512(
                    _mm512_loadu_si512((const void *)(xq + (size_t)j * cols + (size_t)q * 64)), k80);
                __m512i xul = xl == NULL ? xu : _mm512_xor_si512(
                    _mm512_loadu_si512((const void *)(xl + (size_t)j * cols + (size_t)q * 64)), k80);
                const float *s = xs + (size_t)j * np + 2 * q;
                __m512 xv = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(s[0]), _mm512_set1_ps(s[1]));
#pragma GCC unroll 4
                for (int i = 0; i < 4; ++i) {
                    __m512i is = _mm512_dpbusd_epi32(nE[i][q], xu, w[i]);
                    if (xl != NULL) {
                        is = _mm512_add_epi32(_mm512_slli_epi32(is, 7),
                                              _mm512_dpbusd_epi32(nE[i][q], xul, w[i]));
                    }
                    acc[i][j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), _mm512_mul_ps(dE[i][q], xv),
                                                acc[i][j]);
                }
            }
        }
    } else {
        const __m512i z = _mm512_setzero_si512();
        const int nb = cols / 256;
        const size_t per = (span + nb - 1) / nb;
        for (int b = 0; b < nb; ++b) {
            if (pf) {
                for (size_t o = 0; o < per + 64; o += 64) {
                    _mm_prefetch((const char *)pf + (size_t)b * per + o, _MM_HINT_T1);
                }
            }
            for (int h = 0; h < 2; ++h) {
                __m512i A[4], B[4];
#pragma GCC unroll 4
                for (int i = 0; i < 4; ++i) {
                    kq_q6_values(wr[i] + (size_t)b * 210, h, &A[i], &B[i]);
                }
#pragma GCC unroll 4
                for (int j = 0; j < NT; ++j) {
                    const int8_t *xb = xq + (size_t)j * cols + (size_t)b * 256 + 128 * h;
                    __m512i x0 = _mm512_loadu_si512((const void *)xb);
                    __m512i x1 = _mm512_loadu_si512((const void *)(xb + 64));
                    const int8_t *xlb = xl == NULL ? xb : xl + (size_t)j * cols + (size_t)b * 256 + 128 * h;
                    __m512i x0l = _mm512_loadu_si512((const void *)xlb);
                    __m512i x1l = _mm512_loadu_si512((const void *)(xlb + 64));
                    const float *s = xs + (size_t)j * np + 8 * b + 4 * h;
                    __m512 xa = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(s[0]), _mm512_set1_ps(s[1]));
                    __m512 xc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(s[2]), _mm512_set1_ps(s[3]));
#pragma GCC unroll 4
                    for (int i = 0; i < 4; ++i) {
                        __m512i ia = _mm512_dpbusd_epi32(z, A[i], x0), ib = _mm512_dpbusd_epi32(z, B[i], x1);
                        if (xl != NULL) {
                            ia = _mm512_add_epi32(_mm512_slli_epi32(ia, 7), _mm512_dpbusd_epi32(z, A[i], x0l));
                            ib = _mm512_add_epi32(_mm512_slli_epi32(ib, 7), _mm512_dpbusd_epi32(z, B[i], x1l));
                        }
                        acc[i][j] = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(ia),
                            _mm512_mul_ps(dE[i][4 * b + 2 * h], xa), acc[i][j]);
                        acc[i][j] = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(ib),
                            _mm512_mul_ps(dE[i][4 * b + 2 * h + 1], xc), acc[i][j]);
                    }
                }
            }
        }
        /* the term of the sums of x (kq_prep) */
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < NT; ++j) {
                __m512 macc = _mm512_setzero_ps();
                for (int b = 0; b < nb; ++b) {
                    macc = _mm512_fmadd_ps(_mm512_loadu_ps(ds[i] + 16 * b),
                                           _mm512_loadu_ps(xm + (size_t)j * (cols / 16) + 16 * b), macc);
                }
                corr[i][j] = 32.f * _mm512_reduce_add_ps(macc);
            }
        }
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
#pragma GCC unroll 4
        for (int j = 0; j < NT; ++j) {
            out[(size_t)i * ostr_r + (size_t)j * ostr_t] = _mm512_reduce_add_ps(acc[i][j]) - corr[i][j];
        }
    }
}

/* NP_GEMMA_KQ_T1 (default 1): one token of int16 x (a decode step) on 4
 * rows of Q8_0 in one pass (kq_rows4_q8_x16_t1), not kq_rows4_t2 (its scales
 * and sums of the rows first, for tiles of tokens). */
static int kq_t1_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_KQ_T1");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* Rows r .. r + 3 of Q8_0 (at w, rb bytes each; cols a multiple of 64) on
 * one token of int16 x (xq the high plane, xl the low one, xs the scales):
 * the bits of kq_t2_tile (NT 1), in one pass. The integer sum of each lane
 * is the same: there x + 128 (unsigned) times w, from -128 sum(w) of each
 * row (nE, kq_t2_rows); here w + 128 (w ^ 0x80) times x, less 128 sum(x),
 * one vector of each step of 64 values for the 4 rows. The scales of a
 * step: the two fp16 d of its blocks in the lanes (the values of kq_h).
 * On the 2-socket Xeon at 1.2 GHz kq_rows4_t2 did 4 rows of 640 in 0.66 us
 * from cache (one thread): the scales and the sums of the rows were as much
 * work as the products for one token. */
static void kq_rows4_q8_x16_t1(const uint8_t *w, size_t rb, int cols, const int8_t *xq,
                               const float *xs, const int8_t *xl, float *out, const uint8_t *pf)
{
    const __m512i k80 = _mm512_set1_epi8((char)0x80), z = _mm512_setzero_si512();
    const int nq = cols / 64;
    const size_t per = (4 * rb + nq - 1) / nq;
    __m512 acc[4];
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        acc[i] = _mm512_setzero_ps();
    }
    for (int q = 0; q < nq; ++q) {
        if (pf) {
            for (size_t o = 0; o < per + 64; o += 64) {
                _mm_prefetch((const char *)pf + (size_t)q * per + o, _MM_HINT_T1);
            }
        }
        __m512i xh = _mm512_loadu_si512((const void *)(xq + (size_t)q * 64));
        __m512i xo = _mm512_loadu_si512((const void *)(xl + (size_t)q * 64));
        /* 128 sum(x) of each lane, as (hi << 7) + lo */
        __m512i C = _mm512_add_epi32(_mm512_slli_epi32(_mm512_dpbusd_epi32(z, k80, xh), 7),
                                     _mm512_dpbusd_epi32(z, k80, xo));
        __m512 xv = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(xs[2 * q]), _mm512_set1_ps(xs[2 * q + 1]));
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            const uint8_t *b0 = w + (size_t)i * rb + (size_t)q * 68;
            __m512i wu = _mm512_xor_si512(kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b0 + 36)), k80);
            __m512i is = _mm512_sub_epi32(
                _mm512_dpbusd_epi32(_mm512_slli_epi32(_mm512_dpbusd_epi32(z, wu, xh), 7), wu, xo), C);
            uint16_t d0, d1;
            memcpy(&d0, b0, 2);
            memcpy(&d1, b0 + 34, 2);
            __m512 dE = _mm512_cvtph_ps(_mm256_mask_blend_epi16(0xff00, _mm256_set1_epi16((short)d0),
                                                                _mm256_set1_epi16((short)d1)));
            acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), _mm512_mul_ps(dE, xv), acc[i]);
        }
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        out[i] = _mm512_reduce_add_ps(acc[i]);
    }
}

/* Rows r .. r + 3 of Q6_K (cols a multiple of 256) on one token of int16 x:
 * the bits of kq_t2_tile (NT 1) in one pass: the scales of a block (d times
 * its 16 scales, as kq_t2_rows) where the block is read, and the term of
 * the sums of x (xm) in the same order. */
static void kq_rows4_q6_x16_t1(const uint8_t *w, size_t rb, int cols, const int8_t *xq,
                               const float *xs, const float *xm, const int8_t *xl, float *out,
                               const uint8_t *pf)
{
    const __m512i z = _mm512_setzero_si512();
    const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const int nb = cols / 256, np = cols / 32;
    const size_t per = (4 * rb + nb - 1) / nb;
    (void)np;
    __m512 acc[4], macc[4];
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        acc[i] = _mm512_setzero_ps();
        macc[i] = _mm512_setzero_ps();
    }
    for (int b = 0; b < nb; ++b) {
        if (pf) {
            for (size_t o = 0; o < per + 64; o += 64) {
                _mm_prefetch((const char *)pf + (size_t)b * per + o, _MM_HINT_T1);
            }
        }
        __m512 dv[4];
        const __m512 xmv = _mm512_loadu_ps(xm + 16 * b);
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            const uint8_t *blk = w + (size_t)i * rb + (size_t)b * 210;
            dv[i] = _mm512_mul_ps(_mm512_set1_ps(kq_h(blk + 208)), _mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(blk + 192)))));
            macc[i] = _mm512_fmadd_ps(dv[i], xmv, macc[i]);
        }
        for (int h = 0; h < 2; ++h) {
            const int8_t *xb = xq + (size_t)b * 256 + 128 * h, *xlb = xl + (size_t)b * 256 + 128 * h;
            __m512i x0 = _mm512_loadu_si512((const void *)xb), x1 = _mm512_loadu_si512((const void *)(xb + 64));
            __m512i x0l = _mm512_loadu_si512((const void *)xlb), x1l = _mm512_loadu_si512((const void *)(xlb + 64));
            const float *sx = xs + 8 * b + 4 * h;
            __m512 xa = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sx[0]), _mm512_set1_ps(sx[1]));
            __m512 xc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sx[2]), _mm512_set1_ps(sx[3]));
            const __m512i pa = _mm512_add_epi32(iA, _mm512_set1_epi32(8 * h));
            const __m512i pb = _mm512_add_epi32(iA, _mm512_set1_epi32(8 * h + 4));
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i) {
                __m512i A, B;
                kq_q6_values(w + (size_t)i * rb + (size_t)b * 210, h, &A, &B);
                __m512i ia = _mm512_dpbusd_epi32(_mm512_slli_epi32(_mm512_dpbusd_epi32(z, A, x0), 7), A, x0l);
                __m512i ib = _mm512_dpbusd_epi32(_mm512_slli_epi32(_mm512_dpbusd_epi32(z, B, x1), 7), B, x1l);
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia),
                                         _mm512_mul_ps(_mm512_permutexvar_ps(pa, dv[i]), xa), acc[i]);
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ib),
                                         _mm512_mul_ps(_mm512_permutexvar_ps(pb, dv[i]), xc), acc[i]);
            }
        }
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        out[i] = _mm512_reduce_add_ps(acc[i]) - 32.f * _mm512_reduce_add_ps(macc[i]);
    }
}

/* Rows r .. r + 3 of Q8_0 or Q6_K (at w, rb bytes each; cols a multiple of
 * 64 or 256, at most 64 KQ_T2_NV) on n tokens: the scales of the rows one
 * time, then tiles of 4 tokens and one of the rest. */
static void kq_rows4_t2(const uint8_t *w, size_t rb, int type, int cols, const int8_t *xq,
                        const float *xs, const float *xm, int n, float *out, size_t ostride,
                        const int8_t *xl)
{
    const uint8_t *wr[4] = {w, w + rb, w + 2 * rb, w + 3 * rb};
    __m512 dE[4][KQ_T2_NV];
    __m512i nE[4][KQ_T2_NV];
    float ds[4][4 * KQ_T2_NV];
    kq_t2_rows(wr, type, cols, dE, nE, ds);
    const uint8_t *pf = kq_pf_bytes() ? w + 4 * rb : NULL;
    for (int j0 = 0; j0 < n; j0 += 4) {
        int nt = n - j0 < 4 ? n - j0 : 4;
        const int8_t *x = xq + (size_t)j0 * cols;
        const float *s = xs + (size_t)j0 * (cols / 32), *m = xm + (size_t)j0 * (cols / 16);
        float *o = out + (size_t)j0 * ostride;
        const uint8_t *p = j0 == 0 ? pf : NULL;
        if (xl != NULL) {
            const int8_t *l = xl + (size_t)j0 * cols;
            switch (nt) {
            case 4: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 4, p, o, 1, ostride, l); break;
            case 3: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 3, p, o, 1, ostride, l); break;
            case 2: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 2, p, o, 1, ostride, l); break;
            default: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 1, p, o, 1, ostride, l); break;
            }
            continue;
        }
        switch (nt) {
        case 4: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 4, p, o, 1, ostride, NULL); break;
        case 3: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 3, p, o, 1, ostride, NULL); break;
        case 2: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 2, p, o, 1, ostride, NULL); break;
        default: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 1, p, o, 1, ostride, NULL); break;
        }
    }
}
#endif

static void kq_rows4(const uint8_t *w, size_t rb, int type, int cols, const int8_t *xq,
                     const float *xs, const float *xm, const float *x, int n, float *out,
                     size_t ostride)
{
#if defined(__AVX512VNNI__)
    if (type == KQ_NV4 && n > 1 && cols % 64 == 0 && cols <= KQ_NV4_MAX) {
        kq_rows4_nv4(w, rb, cols, xq, xs, n, out, ostride);
        return;
    }
    if ((type == KQ_Q8_0 || (type == KQ_Q6_K && cols % 256 == 0)) && cols % 64 == 0 &&
        cols <= 64 * KQ_T2_NV && kq_tile2_min() >= 1 && n >= kq_tile2_min()) {
        kq_rows4_t2(w, rb, type, cols, xq, xs, xm, n, out, ostride, NULL);
        return;
    }
    if (type == KQ_Q8_0 && n < 4 && kq_q8_4rows()) {
        for (int j = 0; j < n; ++j) {
            kq_dot4_q8_0(w, rb, cols, xq + (size_t)j * cols, xs + (size_t)j * (cols / 32),
                         out + (size_t)j * ostride, 1);
        }
        return;
    }
    if (n >= 4 && kq_tiles(type, cols)) {
        const uint8_t *wr[4] = {w, w + rb, w + 2 * rb, w + 3 * rb};
        float ds[4][KQ_S], dm[4][KQ_S];
        for (int i = 0; i < 4; ++i) {
            kq_row_scales(wr[i], type, cols, ds[i], dm[i]);
        }
        for (int j0 = 0; j0 < n; j0 += 4) {
            int nt = n - j0 < 4 ? n - j0 : 4;
            kq_tile4(wr, ds, dm, type, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * (cols / 32),
                     xm + (size_t)j0 * (cols / 16), nt, out + (size_t)j0 * ostride, 1, ostride);
        }
        return;
    }
#endif
    for (int i = 0; i < 4; ++i) {
        kq_row(w + (size_t)i * rb, type, cols, xq, xs, xm, x, n, out + i, ostride);
    }
}

/* out (t x rows) = x W^T, inside a parallel region. x is the float input
 * (for F32), and xq, xs, xm its quantization (kq_quant_body). */
#if defined(__AVX512VNNI__)
#define KQ_X16_BB 576
#define KQ_X16_XL 98304                  /* the bytes of a small x (kq_x16_body) */
/* KQ_Q8X16: rows g * 16 .. + 15 (the group at wg) on the tokens j0 .. j1 - 1,
 * 8 tokens at a time. xu is x as uint8 (x + 128). vpdpbusd (u8 x s8) on x + 128
 * and w gives sum(x w) + 128 sum(w): the int32 sum of a block starts at
 * -128 times the sums of the rows (made when the rows were packed), the same
 * for all the tokens. For each block: the 8 steps of the group in registers;
 * for each token, 8 vpdpbusd with 4 values of x in each lane, then the scales
 * of the 16 rows times xs. out gets the 16 rows of each token (a stride of
 * rows). */
/* NT tokens from j (a count fixed at compile time: the sums stay in
 * registers). */
#define KQ_X16_NT(NT)                                                                             \
static inline void kq_x16_tok##NT(const uint8_t *wg, int nb, int cols, const uint8_t *xu,        \
                                  const float *xs, int j, float *out, size_t rows)               \
{                                                                                                 \
    __m512 f[NT];                                                                                 \
    for (int tt = 0; tt < NT; ++tt) {                                                             \
        f[tt] = _mm512_setzero_ps();                                                              \
    }                                                                                             \
    for (int b = 0; b < nb; ++b) {                                                                \
        const uint8_t *blk = wg + (size_t)b * KQ_X16_BB;                                          \
        __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)blk));                     \
        __m512i c0 = _mm512_slli_epi32(_mm512_cvtepi16_epi32(                                     \
            _mm256_sub_epi16(_mm256_setzero_si256(), _mm256_loadu_si256((const __m256i *)(blk + 32)))), 7); \
        __m512i W[8];                                                                             \
        for (int k = 0; k < 8; ++k) {                                                             \
            W[k] = _mm512_loadu_si512((const void *)(blk + 64 + 64 * k));                         \
        }                                                                                         \
        _Pragma("GCC unroll 8")                                                                   \
        for (int tt = 0; tt < NT; ++tt) {                                                         \
            const uint8_t *xb = xu + (size_t)(j + tt) * cols + 32 * b;                            \
            __m512i acc = c0;                                                                     \
            for (int k = 0; k < 8; ++k) {                                                         \
                int32_t x4;                                                                       \
                memcpy(&x4, xb + 4 * k, 4);                                                       \
                acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(x4), W[k]);                      \
            }                                                                                     \
            f[tt] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc),                                      \
                                    _mm512_mul_ps(d, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b])), \
                                    f[tt]);                                                       \
        }                                                                                         \
    }                                                                                             \
    for (int tt = 0; tt < NT; ++tt) {                                                             \
        _mm512_storeu_ps(out + (size_t)(j + tt) * rows, f[tt]);                                   \
    }                                                                                             \
}
KQ_X16_NT(8)
KQ_X16_NT(4)
KQ_X16_NT(2)
KQ_X16_NT(1)

/* KQ_Q8X16: rows g * 16 .. + 15 (the group at wg) on the tokens j0 .. j1 - 1,
 * 8 tokens at a time, then 4, 2, 1 (counts fixed at compile time: a
 * variable count made 4 tokens on 10240 x 320 take 1.9 times 1 token). xu
 * is x as uint8 (x + 128). vpdpbusd (u8 x s8) on x + 128 and w gives sum(x
 * w) + 128 sum(w): the int32 sum of a block starts at -128 times the sums
 * of the rows (made when the rows were packed), the same for all the
 * tokens. For each block: the 8 steps of the group in registers; for each
 * token, 8 vpdpbusd with 4 values of x in each lane, then the scales of the
 * 16 rows times xs. out gets the 16 rows of each token (a stride of rows).
 * Each token has the same operations in all the counts: the same bits. */
static void kq_x16_group(const uint8_t *wg, int cols, const uint8_t *xu, const float *xs, int j0,
                         int j1, float *out, size_t rows)
{
    int nb = cols / 32, j = j0;
    for (; j + 8 <= j1; j += 8) {
        kq_x16_tok8(wg, nb, cols, xu, xs, j, out, rows);
    }
    if (j + 4 <= j1) {
        kq_x16_tok4(wg, nb, cols, xu, xs, j, out, rows);
        j += 4;
    }
    if (j + 2 <= j1) {
        kq_x16_tok2(wg, nb, cols, xu, xs, j, out, rows);
        j += 2;
    }
    if (j < j1) {
        kq_x16_tok1(wg, nb, cols, xu, xs, j, out, rows);
    }
}

/* out (t x rows) of a KQ_Q8X16 matrix, inside a parallel region: x + 128 once
 * (a buffer of the call), then blocks of MA_TB tokens (x in L2) with the
 * groups of 16 rows over the threads. One token has the sums of a group: the
 * same bits. */
static void kq_x16_body(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        int t, float *out)
{
    size_t n = (size_t)t * cols;
    size_t gb = (size_t)cols / 32 * KQ_X16_BB;
    if (n <= KQ_X16_XL) {
        /* a small x (a step, a verify group): each thread makes x + 128 on
         * its stack, so the record has one barrier, not four (and no
         * malloc). Two halves of the columns as tasks, for a matrix of few
         * groups (hc_*_down: 20 groups for 18 threads), were not faster. */
        __attribute__((aligned(64))) uint8_t xl[KQ_X16_XL];
        for (size_t i = 0; i < n; i += 64) {
            if (i + 64 <= n) {
                __m512i v = _mm512_loadu_si512((const void *)(xq + i));
                _mm512_store_si512((void *)(xl + i), _mm512_xor_si512(v, _mm512_set1_epi8((char)0x80)));
            } else {
                for (size_t k = i; k < n; ++k) {
                    xl[k] = (uint8_t)xq[k] ^ 0x80;
                }
            }
        }
        #pragma omp for schedule(static)
        for (int g = 0; g < rows / 16; ++g) {
            kq_x16_group(w + (size_t)g * gb, cols, xl, xs, 0, t, out + 16 * g, (size_t)rows);
        }
        return;
    }
    uint8_t *xu = NULL;
    #pragma omp single copyprivate(xu)
    xu = (uint8_t *)aligned_alloc(64, (n + 63) / 64 * 64);
    #pragma omp for schedule(static)
    for (size_t i = 0; i < n; i += 64) {
        if (i + 64 <= n) {
            __m512i v = _mm512_loadu_si512((const void *)(xq + i));
            _mm512_storeu_si512((void *)(xu + i), _mm512_xor_si512(v, _mm512_set1_epi8((char)0x80)));
        } else {
            for (size_t k = i; k < n; ++k) {
                xu[k] = (uint8_t)xq[k] ^ 0x80;
            }
        }
    }
    for (int j0 = 0; j0 < t; j0 += MA_TB) {
        int j1 = t - j0 < MA_TB ? t : j0 + MA_TB;
        #pragma omp for schedule(static) nowait
        for (int g = 0; g < rows / 16; ++g) {
            kq_x16_group(w + (size_t)g * gb, cols, xu, xs, j0, j1, out + 16 * g, (size_t)rows);
        }
    }
    #pragma omp barrier
    #pragma omp single
    free(xu);
}
#endif

#if !defined(__AVX512VNNI__) && defined(__AVX2__)
/* KQ_Q8X16 without VNNI (AVX2: the i5 of the plan): the same layout. A step
 * of a block is 16 rows by 4 values: rows 0 to 7 in the first 32 bytes and
 * 8 to 15 in the second. The int8 product of a step: vpmaddubsw of |x| (u8)
 * and w with the sign of x, then vpmaddwd with ones (at most 2 * 127 * 127
 * in an int16: no saturation), so x needs no + 128 and the sums of the
 * rows are not used. NT tokens from j (a count fixed at compile time). */
#define KQ_X16_BB2 576
#define KQ_X16_A2(NT)                                                                             \
static inline void kq_x16_avx2_tok##NT(const uint8_t *wg, int nb, int cols, const int8_t *xq,     \
                                       const float *xs, int j, float *out, size_t rows)          \
{                                                                                                 \
    const __m256i ones = _mm256_set1_epi16(1);                                                    \
    __m256 f[NT][2];                                                                              \
    for (int tt = 0; tt < NT; ++tt) {                                                             \
        f[tt][0] = _mm256_setzero_ps();                                                           \
        f[tt][1] = _mm256_setzero_ps();                                                           \
    }                                                                                             \
    for (int b = 0; b < nb; ++b) {                                                                \
        const uint8_t *blk = wg + (size_t)b * KQ_X16_BB2;                                         \
        __m256 d0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)blk));                       \
        __m256 d1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 16)));                \
        __m256i a[NT][2];                                                                         \
        for (int tt = 0; tt < NT; ++tt) {                                                         \
            a[tt][0] = _mm256_setzero_si256();                                                    \
            a[tt][1] = _mm256_setzero_si256();                                                    \
        }                                                                                         \
        for (int k = 0; k < 8; ++k) {                                                             \
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(blk + 64 + 64 * k));               \
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(blk + 96 + 64 * k));               \
            _Pragma("GCC unroll 4")                                                               \
            for (int tt = 0; tt < NT; ++tt) {                                                     \
                int32_t x4;                                                                       \
                memcpy(&x4, xq + (size_t)(j + tt) * cols + 32 * b + 4 * k, 4);                    \
                __m256i xb = _mm256_set1_epi32(x4);                                               \
                __m256i ax = _mm256_sign_epi8(xb, xb);                                            \
                a[tt][0] = _mm256_add_epi32(a[tt][0], _mm256_madd_epi16(                          \
                    _mm256_maddubs_epi16(ax, _mm256_sign_epi8(w0, xb)), ones));                   \
                a[tt][1] = _mm256_add_epi32(a[tt][1], _mm256_madd_epi16(                          \
                    _mm256_maddubs_epi16(ax, _mm256_sign_epi8(w1, xb)), ones));                   \
            }                                                                                     \
        }                                                                                         \
        for (int tt = 0; tt < NT; ++tt) {                                                         \
            __m256 sx = _mm256_set1_ps(xs[(size_t)(j + tt) * nb + b]);                            \
            f[tt][0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a[tt][0]), _mm256_mul_ps(d0, sx), f[tt][0]); \
            f[tt][1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a[tt][1]), _mm256_mul_ps(d1, sx), f[tt][1]); \
        }                                                                                         \
    }                                                                                             \
    for (int tt = 0; tt < NT; ++tt) {                                                             \
        _mm256_storeu_ps(out + (size_t)(j + tt) * rows, f[tt][0]);                                \
        _mm256_storeu_ps(out + (size_t)(j + tt) * rows + 8, f[tt][1]);                            \
    }                                                                                             \
}
KQ_X16_A2(2)
KQ_X16_A2(1)

/* out (t x rows) of a KQ_Q8X16 matrix, inside a parallel region: blocks of
 * MA_TB tokens (x in L2), the groups of 16 rows over the threads. */
static void kq_x16_body(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        int t, float *out)
{
    size_t gb = (size_t)cols / 32 * KQ_X16_BB2;
    int nb = cols / 32;
    for (int j0 = 0; j0 < t; j0 += MA_TB) {
        int j1 = t - j0 < MA_TB ? t : j0 + MA_TB;
        #pragma omp for schedule(static) nowait
        for (int g = 0; g < rows / 16; ++g) {
            const uint8_t *wg = w + (size_t)g * gb;
            float *o = out + 16 * g;
            int j = j0;
            for (; j + 2 <= j1; j += 2) {
                kq_x16_avx2_tok2(wg, nb, cols, xq, xs, j, o, (size_t)rows);
            }
            if (j < j1) {
                kq_x16_avx2_tok1(wg, nb, cols, xq, xs, j, o, (size_t)rows);
            }
        }
    }
    #pragma omp barrier
}
#endif

/* KQ_NVX: -12 times the sum of each 16 int8 values of n rows of x (the
 * term of the codes + 12 of kq_nvx_rows). */
static void kq_nvx_xsum(const int8_t *xq, int n, int cols, int32_t *xn)
{
    for (int j = 0; j < n; ++j) {
        for (int h = 0; h < cols / 16; ++h) {
            const int8_t *p = xq + (size_t)j * cols + 16 * h;
            int32_t sum = 0;
            for (int u = 0; u < 16; ++u) {
                sum += p[u];
            }
            xn[(size_t)j * (cols / 16) + h] = -12 * sum;
        }
    }
}

#if defined(__AVX512VNNI__)
/* E2M1 codes to twice their values plus 12 (0 to 24: the unsigned operand
 * of vpdpbusd). */
/* 16 tokens for each decode of a block (8: 1.5% slower at 512 tokens, 7% at
 * 2048; the experts of a prompt are near the rate of the memory). */
#define KQ_NVX_TB 16
static const uint8_t kq_e2m1u[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};

/* 16 E4M3 scales as float32 times 2^-8: the bits go to float16 as they are
 * (the exponent bias of float16 is 8 more), exact for all the codes. */
static inline __m512 kq_e4m3x16(const uint8_t *p)
{
    __m256i b = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p));
    __m256i h = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x7f)), 7),
                                _mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x80)), 8));
    return _mm512_cvtph_ps(h);
}

/* The 16 rows of the KQ_NVX group at wg on n tokens: out[j * ostride + r]
 * gets row r, token j. xn: kq_nvx_xsum of the tokens. For each block, 8
 * tokens at a time: the 8 steps to u8 (codes + 12) in registers; for each
 * token, 4 vpdpbusd for each half (a lane is a row; x broadcast, signed),
 * from -12 sum(x), which removes the 12. A token alone and in a group has
 * the same operations: the same bits. */
static void kq_nvx_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32, nh = cols / 16;
    float g;
    memcpy(&g, wg, 4);
    const __m512 gs = _mm512_set1_ps(128.f * g);        /* 0.5 g, and 2^8 of kq_e4m3x16 */
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1u));
    const __m256i m4 = _mm256_set1_epi8(15);
    for (int j = 0; j < n; j += KQ_NVX_TB) {
        int nt = n - j < KQ_NVX_TB ? n - j : KQ_NVX_TB;
        __m512 f[KQ_NVX_TB];
        for (int tt = 0; tt < KQ_NVX_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
            __m512i W[8];
            for (int k = 0; k < 8; ++k) {
                __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
                __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                W[k] = _mm512_shuffle_epi8(lut, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
            }
            __m512 s0 = kq_e4m3x16(blk + 256), s1 = kq_e4m3x16(blk + 272);
            for (int tt = 0; tt < nt; ++tt) {
                const int8_t *xb = xq + (size_t)(j + tt) * cols + 32 * b;
                const int32_t *nx = xn + (size_t)(j + tt) * nh + 2 * b;
                __m512i a0 = _mm512_set1_epi32(nx[0]), a1 = _mm512_set1_epi32(nx[1]);
                for (int k = 0; k < 4; ++k) {
                    int32_t x0, x1;
                    memcpy(&x0, xb + 4 * k, 4);
                    memcpy(&x1, xb + 16 + 4 * k, 4);
                    a0 = _mm512_dpbusd_epi32(a0, W[k], _mm512_set1_epi32(x0));
                    a1 = _mm512_dpbusd_epi32(a1, W[4 + k], _mm512_set1_epi32(x1));
                }
                __m512 p = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), s1,
                                           _mm512_mul_ps(_mm512_cvtepi32_ps(a0), s0));
                f[tt] = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b]), f[tt]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, _mm512_mul_ps(f[tt], gs));
        }
    }
}
#else
/* Without VNNI: the same sums in C (not the bits of the VNNI path). */
static void kq_nvx_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    static const int8_t e2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int nb = cols / 32;
    float g;
    memcpy(&g, wg, 4);
    (void)xn;
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
                float p = 0.f;
                for (int h = 0; h < 2; ++h) {
                    int32_t a = 0;
                    for (int v = 16 * h; v < 16 * h + 16; ++v) {
                        int c = (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
                        a += e2[c] * xq[(size_t)j * cols + 32 * b + v];
                    }
                    p += (float)a * kq_e4m3_tab[blk[256 + 16 * h + r]];
                }
                f += p * xs[(size_t)j * nb + b];
            }
            out[(size_t)j * ostride + r] = 0.5f * g * f;
        }
    }
}
#endif

/* KQ_Q4X: -8 times the sum of each 32 int8 values of n rows of x (the term of
 * the codes q = w + 8 of kq_q4x_rows). A row of xn has the stride cols / 16,
 * as that of kq_nvx_xsum (the scratch of kq_moe_body serves both). */
static void kq_q4x_xsum(const int8_t *xq, int n, int cols, int32_t *xn)
{
    for (int j = 0; j < n; ++j) {
        for (int b = 0; b < cols / 32; ++b) {
            const int8_t *p = xq + (size_t)j * cols + 32 * b;
            int32_t sum = 0;
            for (int u = 0; u < 32; ++u) {
                sum += p[u];
            }
            xn[(size_t)j * (cols / 16) + b] = -8 * sum;
        }
    }
}

/* The sums of kq_rows_n for a type of groups of 16 rows (xn). */
static inline void kq_x16_xsum(int type, const int8_t *xq, int cols, int32_t *xn)
{
    if (type == KQ_Q4X) {
        kq_q4x_xsum(xq, 1, cols, xn);
    } else {
        kq_nvx_xsum(xq, 1, cols, xn);
    }
}

#if defined(__AVX512VNNI__)
/* The 16 rows of the KQ_Q4X group at wg on n tokens: out[j * ostride + r]
 * gets row r, token j. xn: kq_q4x_xsum of the tokens. As kq_nvx_rows, with
 * the codes as they are (q = w + 8, 0 to 15: the unsigned operand of
 * vpdpbusd), the sum of a block from -8 sum(x), and one scale of each row
 * for each block. A token alone and in a group has the same operations. */
static void kq_q4x_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32;
    const __m256i m4 = _mm256_set1_epi8(15);
    for (int j = 0; j < n; j += KQ_Q4X_TB) {
        int nt = n - j < KQ_Q4X_TB ? n - j : KQ_Q4X_TB;
        __m512 f[KQ_Q4X_TB];
        for (int tt = 0; tt < KQ_Q4X_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
            __m512i W[8];
            for (int k = 0; k < 8; ++k) {
                __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
                __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                W[k] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            }
            __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 256)));
            /* Two tokens at a time, each with two sums (the even and the odd
             * steps): four chains of 4 vpdpbusd in place of one chain of 8
             * for each token. A chain of 8 (5 cycles each) left the ports
             * half idle with the scheduler full (resource_stalls 35 per
             * cent of the cycles). The sums are int32, so the order of the
             * adds does not change them: the same bits. */
            int tt = 0;
            for (; tt + 2 <= nt; tt += 2) {
                const int8_t *xb0 = xq + (size_t)(j + tt) * cols + 32 * b;
                const int8_t *xb1 = xb0 + cols;
                __m512i a0 = _mm512_set1_epi32(xn[(size_t)(j + tt) * (cols / 16) + b]);
                __m512i a1 = _mm512_setzero_si512();
                __m512i c0 = _mm512_set1_epi32(xn[(size_t)(j + tt + 1) * (cols / 16) + b]);
                __m512i c1 = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    int32_t x0, x1, y0, y1;
                    memcpy(&x0, xb0 + 4 * k, 4);
                    memcpy(&x1, xb0 + 4 * k + 4, 4);
                    memcpy(&y0, xb1 + 4 * k, 4);
                    memcpy(&y1, xb1 + 4 * k + 4, 4);
                    a0 = _mm512_dpbusd_epi32(a0, W[k], _mm512_set1_epi32(x0));
                    c0 = _mm512_dpbusd_epi32(c0, W[k], _mm512_set1_epi32(y0));
                    a1 = _mm512_dpbusd_epi32(a1, W[k + 1], _mm512_set1_epi32(x1));
                    c1 = _mm512_dpbusd_epi32(c1, W[k + 1], _mm512_set1_epi32(y1));
                }
                f[tt] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(a0, a1)),
                                        _mm512_mul_ps(d, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b])),
                                        f[tt]);
                f[tt + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(c0, c1)),
                                            _mm512_mul_ps(d, _mm512_set1_ps(xs[(size_t)(j + tt + 1) * nb + b])),
                                            f[tt + 1]);
            }
            for (; tt < nt; ++tt) {
                const int8_t *xb = xq + (size_t)(j + tt) * cols + 32 * b;
                __m512i a0 = _mm512_set1_epi32(xn[(size_t)(j + tt) * (cols / 16) + b]);
                __m512i a1 = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    int32_t x0, x1;
                    memcpy(&x0, xb + 4 * k, 4);
                    memcpy(&x1, xb + 4 * k + 4, 4);
                    a0 = _mm512_dpbusd_epi32(a0, W[k], _mm512_set1_epi32(x0));
                    a1 = _mm512_dpbusd_epi32(a1, W[k + 1], _mm512_set1_epi32(x1));
                }
                f[tt] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(a0, a1)),
                                        _mm512_mul_ps(d, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b])),
                                        f[tt]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, f[tt]);
        }
    }
}

/* The prompt form (kq_q4x_gemm): the codes of a group of 16 rows unpacked
 * once (u: nb x 8 steps x 64 bytes, the unsigned operand of vpdpbusd as
 * kq_q4x_rows makes it; dsc: nb x 16 float scales), then tiles of G groups
 * and T tokens with all the sums in registers. kq_q4x_rows kept the float
 * sums of its 16 tokens in an array that the compiler put on the stack (a
 * load and a store for each token and block), and unpacked the codes once
 * for each 16 tokens. Each value has the operations of kq_q4x_rows: the
 * int32 sum of a block (exact in any order), then the same cvt, mul, and
 * fma, block after block: the same bits. */
static void kq_q4x_unpack(const uint8_t *wg, int nb, uint8_t *u, float *dsc)
{
    const __m256i m4 = _mm256_set1_epi8(15);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            _mm512_store_si512((__m512i *)(u + ((size_t)b * 8 + k) * 64),
                               _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
        }
        _mm512_store_ps(dsc + (size_t)b * 16,
                        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 256))));
    }
}

/* The tokens of a tile: 2 groups and 6 tokens are 12 chains of vpdpbusd (5
 * cycles each, 2 ports), 24 sums in registers. */
#ifndef KQ_Q4X_TT
#define KQ_Q4X_TT 6
#endif

/* G groups (1 or 2) of 16 rows on T tokens (1 to KQ_Q4X_TT): out[t * ostride + r]. */
static inline __attribute__((always_inline)) void kq_q4x_tile(
    const uint8_t *u0, const uint8_t *u1, const float *d0, const float *d1, int nb, int cols,
    const int8_t *xq, const float *xs, const int32_t *xn, float *out, size_t ostride,
    const int G, const int T)
{
    __m512 f[2][KQ_Q4X_TT];
    for (int g = 0; g < G; ++g) {
        for (int t = 0; t < T; ++t) {
            f[g][t] = _mm512_setzero_ps();
        }
    }
    for (int b = 0; b < nb; ++b) {
        __m512i a[2][KQ_Q4X_TT];
        for (int t = 0; t < T; ++t) {
            __m512i c = _mm512_set1_epi32(xn[(size_t)t * (cols / 16) + b]);
            for (int g = 0; g < G; ++g) {
                a[g][t] = c;
            }
        }
        for (int k = 0; k < 8; ++k) {
            __m512i w0 = _mm512_load_si512((const __m512i *)(u0 + ((size_t)b * 8 + k) * 64));
            __m512i w1 = G > 1 ? _mm512_load_si512((const __m512i *)(u1 + ((size_t)b * 8 + k) * 64))
                               : w0;
            for (int t = 0; t < T; ++t) {
                int32_t x4;
                memcpy(&x4, xq + (size_t)t * cols + 32 * b + 4 * k, 4);
                __m512i xv = _mm512_set1_epi32(x4);
                a[0][t] = _mm512_dpbusd_epi32(a[0][t], w0, xv);
                if (G > 1) {
                    a[1][t] = _mm512_dpbusd_epi32(a[1][t], w1, xv);
                }
            }
        }
        __m512 e0 = _mm512_load_ps(d0 + (size_t)b * 16);
        __m512 e1 = G > 1 ? _mm512_load_ps(d1 + (size_t)b * 16) : e0;
        for (int t = 0; t < T; ++t) {
            __m512 sx = _mm512_set1_ps(xs[(size_t)t * nb + b]);
            f[0][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[0][t]), _mm512_mul_ps(e0, sx), f[0][t]);
            if (G > 1) {
                f[1][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[1][t]), _mm512_mul_ps(e1, sx),
                                          f[1][t]);
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        _mm512_storeu_ps(out + (size_t)t * ostride, f[0][t]);
        if (G > 1) {
            _mm512_storeu_ps(out + (size_t)t * ostride + 16, f[1][t]);
        }
    }
}

/* The scratch of the unpacked codes of a thread (2 groups). */
static __thread uint8_t *kq_q4x_u;
static __thread float *kq_q4x_d;
static __thread size_t kq_q4x_un;

/* KQ_Q4X on t tokens (4 or more), inside a parallel region: blocks of
 * tokens (their int8 x in L2 with the codes of 2 groups), the pairs of
 * groups over the threads, tiles of 2 groups and 4 tokens. */
static void kq_q4x_gemm(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int t, float *out)
{
    const int nb = cols / 32, groups = rows / 16;
    const size_t gb = 16 * kq_row_bytes(KQ_Q4X, cols);
    const size_t need = (size_t)nb * 8 * 64;
    if (kq_q4x_un < need) {
        free(kq_q4x_u);
        free(kq_q4x_d);
        kq_q4x_u = (uint8_t *)aligned_alloc(64, 2 * need);
        kq_q4x_d = (float *)aligned_alloc(64, 2 * (size_t)nb * 16 * sizeof(float));
        kq_q4x_un = need;
    }
    uint8_t *u0 = kq_q4x_u, *u1 = kq_q4x_u + need;
    float *d0 = kq_q4x_d, *d1 = kq_q4x_d + (size_t)nb * 16;
    /* the int8 x of a block of tokens about 512 KB */
    int tb = (512 * 1024 / cols) / KQ_Q4X_TT * KQ_Q4X_TT;
    tb = tb < 2 * KQ_Q4X_TT ? 2 * KQ_Q4X_TT : (tb > 126 ? 126 : tb);
    /* The items: a block of tokens and a pair of groups, the pairs of a
     * block next to each other, so a thread takes adjacent pairs of one
     * block (its x in L2). A matrix of few rows (the down map: 240 groups,
     * 120 pairs) then still gives each thread the same work. */
    /* blocks of the same size (a last block of a few tokens left the
     * threads of its items idle) */
    const int blocks = (t + tb - 1) / tb;
    tb = ((t + blocks - 1) / blocks + KQ_Q4X_TT - 1) / KQ_Q4X_TT * KQ_Q4X_TT;
    const int pairs = (groups + 1) / 2;
    #pragma omp for schedule(static)
    for (int it = 0; it < blocks * pairs; ++it) {
        const int j0 = (it / pairs) * tb, gp = it % pairs;
        if (j0 >= t) {
            continue;
        }
        const int nt = t - j0 < tb ? t - j0 : tb;
        {
            const int g = 2 * gp, two = g + 1 < groups;
            kq_q4x_unpack(w + (size_t)g * gb, nb, u0, d0);
            if (two) {
                kq_q4x_unpack(w + (size_t)(g + 1) * gb, nb, u1, d1);
            }
            for (int jj = 0; jj < nt; jj += KQ_Q4X_TT) {
                const int j = j0 + jj, T = nt - jj < KQ_Q4X_TT ? nt - jj : KQ_Q4X_TT;
                const int8_t *xj = xq + (size_t)j * cols;
                const float *sj = xs + (size_t)j * nb;
                const int32_t *nj = xn + (size_t)j * (cols / 16);
                float *oj = out + (size_t)j * rows + 16 * g;
#define KQ_Q4X_TILE(GG, TT) kq_q4x_tile(u0, u1, d0, d1, nb, cols, xj, sj, nj, oj, (size_t)rows, GG, TT)
                if (two) {
                    switch (T) {
                    case 6: KQ_Q4X_TILE(2, 6); break;
                    case 5: KQ_Q4X_TILE(2, 5); break;
                    case 4: KQ_Q4X_TILE(2, 4); break;
                    case 3: KQ_Q4X_TILE(2, 3); break;
                    case 2: KQ_Q4X_TILE(2, 2); break;
                    default: KQ_Q4X_TILE(2, 1); break;
                    }
                } else {
                    switch (T) {
                    case 6: KQ_Q4X_TILE(1, 6); break;
                    case 5: KQ_Q4X_TILE(1, 5); break;
                    case 4: KQ_Q4X_TILE(1, 4); break;
                    case 3: KQ_Q4X_TILE(1, 3); break;
                    case 2: KQ_Q4X_TILE(1, 2); break;
                    default: KQ_Q4X_TILE(1, 1); break;
                    }
                }
#undef KQ_Q4X_TILE
            }
        }
    }
}
#define KQ_Q4X_GEMM 1

/* ---- int16 x on KQ_Q4X (the prompt with NP_GEMMA_INT4_Q8=16) ----
 *
 * The int8 x of a prompt (kq_quant_x) rounds each value to 1/127 of the
 * largest of its 32; on the 26B a change of 1e-6 in a router weight then
 * moved 20 per cent of the top tokens of a prompt of 256, and the NLL of a
 * text grew by up to 0.37 against float32. int16 x (gemma_quant_group32_i16:
 * 1/32767 of the largest of 32, to the nearest even) gives the NLL of
 * float32. vpdpwssd multiplies 2 int16 of a lane by 2 int16 weights: 32
 * products for each instruction, half of vpdpbusd. The weights are the
 * codes less 8 as int16, so a block needs no correction term: at most
 * 32 * 32767 * 8 < 2^31.
 *
 * kq_q4x_unpack16 gives each block of 32 columns of a group of 16 rows 16
 * vectors: vector j holds, in lane r, the weights of row r at columns 2j and
 * 2j + 1 (the lanes of kq_q4x_rows: rows 0 to 7 from the low 4 bits, 8 to
 * 15 from the high 4 bits). */
static void kq_q4x_unpack16(const uint8_t *wg, int nb, int16_t *u, float *dsc)
{
    const __m256i m4 = _mm256_set1_epi8(15);
    const __m512i eight = _mm512_set1_epi16(8);
    const __m512i ia = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i ib = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m512i lo = _mm512_cvtepu8_epi16(_mm256_and_si256(q, m4));
            __m512i hi = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
            /* dword 2L of lo holds columns 4k, 4k + 1 of row L, dword 2L + 1
             * columns 4k + 2, 4k + 3; hi the same for row L + 8 */
            _mm512_store_si512((__m512i *)(u + ((size_t)b * 16 + 2 * k) * 32),
                               _mm512_sub_epi16(_mm512_permutex2var_epi32(lo, ia, hi), eight));
            _mm512_store_si512((__m512i *)(u + ((size_t)b * 16 + 2 * k + 1) * 32),
                               _mm512_sub_epi16(_mm512_permutex2var_epi32(lo, ib, hi), eight));
        }
        _mm512_store_ps(dsc + (size_t)b * 16,
                        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 256))));
    }
}

/* G groups (1 or 2) of 16 rows on T tokens (G 2: T to 4; G 1: T to 8), nb
 * blocks of 32 columns (a chunk of the row): the int16 x of token t at
 * xq + t * xstride, its scales at xs + t * sstride: out[t * ostride + r].
 * As kq_q4x_tile: the int32 sum of a block, then the scale of the block of
 * the row times the scale of x, block after block. ACC 1 goes on from the
 * sums in out (the chunk before): the same additions in the same order. */
#define KQ16_TMAX 8
static inline __attribute__((always_inline)) void kq_q4x_tile16(
    const int16_t *u0, const int16_t *u1, const float *d0, const float *d1, int nb,
    size_t xstride, size_t sstride, const int16_t *xq, const float *xs, float *out,
    size_t ostride, const int G, const int T, const int ACC)
{
    __m512 f[2][KQ16_TMAX];
    for (int g = 0; g < G; ++g) {
        for (int t = 0; t < T; ++t) {
            f[g][t] = ACC ? _mm512_loadu_ps(out + (size_t)t * ostride + 16 * g) : _mm512_setzero_ps();
        }
    }
    for (int b = 0; b < nb; ++b) {
        __m512i a[2][KQ16_TMAX];
        for (int g = 0; g < G; ++g) {
            for (int t = 0; t < T; ++t) {
                a[g][t] = _mm512_setzero_si512();
            }
        }
        for (int j = 0; j < 16; ++j) {
            __m512i w0 = _mm512_load_si512((const __m512i *)(u0 + ((size_t)b * 16 + j) * 32));
            __m512i w1 = G > 1 ? _mm512_load_si512((const __m512i *)(u1 + ((size_t)b * 16 + j) * 32))
                               : w0;
            for (int t = 0; t < T; ++t) {
                int32_t x2;
                memcpy(&x2, xq + (size_t)t * xstride + 32 * b + 2 * j, 4);
                __m512i xv = _mm512_set1_epi32(x2);
                a[0][t] = _mm512_dpwssd_epi32(a[0][t], w0, xv);
                if (G > 1) {
                    a[1][t] = _mm512_dpwssd_epi32(a[1][t], w1, xv);
                }
            }
        }
        __m512 e0 = _mm512_load_ps(d0 + (size_t)b * 16);
        __m512 e1 = G > 1 ? _mm512_load_ps(d1 + (size_t)b * 16) : e0;
        for (int t = 0; t < T; ++t) {
            __m512 sx = _mm512_set1_ps(xs[(size_t)t * sstride + b]);
            f[0][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[0][t]), _mm512_mul_ps(e0, sx), f[0][t]);
            if (G > 1) {
                f[1][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[1][t]), _mm512_mul_ps(e1, sx),
                                          f[1][t]);
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        _mm512_storeu_ps(out + (size_t)t * ostride, f[0][t]);
        if (G > 1) {
            _mm512_storeu_ps(out + (size_t)t * ostride + 16, f[1][t]);
        }
    }
}

/* The scratch of the unpacked codes of a thread (2 groups, int16). */
static __thread int16_t *kq16_u;
static __thread float *kq16_d;
static __thread size_t kq16_un;

static void kq16_scratch(int nb)
{
    const size_t need = (size_t)nb * 16 * 32;
    if (kq16_un < need) {
        free(kq16_u);
        free(kq16_d);
        kq16_u = (int16_t *)aligned_alloc(64, 2 * need * sizeof(int16_t));
        kq16_d = (float *)aligned_alloc(64, 2 * (size_t)nb * 16 * sizeof(float));
        kq16_un = need;
    }
}

/* One group of 16 rows on n tokens (an expert of kq_moe_body): the codes
 * unpacked once, then tiles of 8 tokens. */
/* The blocks of a chunk of columns: the unpacked codes of a group (1 KB a
 * block) of a chunk stay in L2 with x (a row of 15360 columns gave 480 KB a
 * group, and the tiles read them from L3). */
#define KQ16_CHUNK 128

static void kq_q4x_rows16(const uint8_t *wg, int cols, const int16_t *xq, const float *xs, int n,
                          float *out, size_t ostride)
{
    const int nb = cols / 32;
    kq16_scratch(nb < KQ16_CHUNK ? nb : KQ16_CHUNK);
    int16_t *u0 = kq16_u;
    float *d0 = kq16_d;
    for (int c0 = 0; c0 < nb; c0 += KQ16_CHUNK) {
        const int nc = nb - c0 < KQ16_CHUNK ? nb - c0 : KQ16_CHUNK;
        kq_q4x_unpack16(wg + (size_t)c0 * KQ_Q4X_BB, nc, u0, d0);
        for (int j = 0; j < n; j += KQ16_TMAX) {
            const int T = n - j < KQ16_TMAX ? n - j : KQ16_TMAX;
            const int16_t *xj = xq + (size_t)j * cols + 32 * (size_t)c0;
            const float *sj = xs + (size_t)j * nb + c0;
            float *oj = out + (size_t)j * ostride;
#define KQ16_TILE1(TT) do { if (c0) kq_q4x_tile16(u0, u0, d0, d0, nc, cols, nb, xj, sj, oj, ostride, 1, TT, 1); \
                            else kq_q4x_tile16(u0, u0, d0, d0, nc, cols, nb, xj, sj, oj, ostride, 1, TT, 0); } while (0)
            switch (T) {
            case 8: KQ16_TILE1(8); break;
            case 7: KQ16_TILE1(7); break;
            case 6: KQ16_TILE1(6); break;
            case 5: KQ16_TILE1(5); break;
            case 4: KQ16_TILE1(4); break;
            case 3: KQ16_TILE1(3); break;
            case 2: KQ16_TILE1(2); break;
            default: KQ16_TILE1(1); break;
            }
#undef KQ16_TILE1
        }
    }
}

/* KQ_Q4X on t tokens of int16 x, inside a parallel region: as kq_q4x_gemm,
 * with tiles of 2 groups and 4 tokens. */
static void kq_q4x_gemm16(const uint8_t *w, int rows, int cols, const int16_t *xq, const float *xs,
                          int t, float *out)
{
    const int nb = cols / 32, groups = rows / 16;
    const size_t gb = 16 * kq_row_bytes(KQ_Q4X, cols);
    const int cb = nb < KQ16_CHUNK ? nb : KQ16_CHUNK;
    kq16_scratch(cb);
    int16_t *u0 = kq16_u, *u1 = kq16_u + kq16_un;
    float *d0 = kq16_d, *d1 = kq16_d + (size_t)cb * 16;
    /* the int16 x of a block of tokens about 512 KB, blocks of the same size */
    int tb = (256 * 1024 / cols) & ~3;
    tb = tb < 8 ? 8 : (tb > 128 ? 128 : tb);
    const int blocks = (t + tb - 1) / tb;
    tb = ((t + blocks - 1) / blocks + 3) & ~3;
    const int pairs = (groups + 1) / 2;
    #pragma omp for schedule(static)
    for (int it = 0; it < blocks * pairs; ++it) {
        const int j0 = (it / pairs) * tb, gp = it % pairs;
        if (j0 >= t) {
            continue;
        }
        const int nt = t - j0 < tb ? t - j0 : tb;
        const int g = 2 * gp, two = g + 1 < groups;
        for (int c0 = 0; c0 < nb; c0 += cb) {
        const int nc = nb - c0 < cb ? nb - c0 : cb;
        kq_q4x_unpack16(w + (size_t)g * gb + (size_t)c0 * KQ_Q4X_BB, nc, u0, d0);
        if (two) {
            kq_q4x_unpack16(w + (size_t)(g + 1) * gb + (size_t)c0 * KQ_Q4X_BB, nc, u1, d1);
        }
        for (int jj = 0; jj < nt; jj += 4) {
            const int j = j0 + jj, T = nt - jj < 4 ? nt - jj : 4;
            const int16_t *xj = xq + (size_t)j * cols + 32 * (size_t)c0;
            const float *sj = xs + (size_t)j * nb + c0;
            float *oj = out + (size_t)j * rows + 16 * g;
#define KQ16_TILE(GG, TT) do { if (c0) kq_q4x_tile16(u0, u1, d0, d1, nc, cols, nb, xj, sj, oj, (size_t)rows, GG, TT, 1); \
                               else kq_q4x_tile16(u0, u1, d0, d1, nc, cols, nb, xj, sj, oj, (size_t)rows, GG, TT, 0); } while (0)
            if (two) {
                switch (T) {
                case 4: KQ16_TILE(2, 4); break;
                case 3: KQ16_TILE(2, 3); break;
                case 2: KQ16_TILE(2, 2); break;
                default: KQ16_TILE(2, 1); break;
                }
            } else {
                switch (T) {
                case 4: KQ16_TILE(1, 4); break;
                case 3: KQ16_TILE(1, 3); break;
                case 2: KQ16_TILE(1, 2); break;
                default: KQ16_TILE(1, 1); break;
                }
            }
#undef KQ16_TILE
        }
        }
    }
}
#define KQ_Q4X_Q16 1
#elif defined(__AVX2__)
/* AVX2 (no VNNI): a 32-bit lane is a row, the low 4 bits of a step are rows
 * 0 to 7 and the high 4 bits rows 8 to 15. vpmaddubsw of the codes (0 to
 * 15) and 4 int8 x (the same in each lane) gives 2 sums of 2 in each lane;
 * the 8 steps of a block add as int16 (at most 8 * 2 * 15 * 127 = 30480),
 * then one vpmaddwd adds the pairs. A token alone and in a group has the
 * same operations. */
/* kq_q4x_rows for NT tokens (a constant 1 to 4): the codes of a step go to
 * the low and the high rows once for the NT tokens. */
static inline __attribute__((always_inline)) void kq_q4x_rows_n(
    const uint8_t *wg, int cols, const int8_t *xq, const float *xs, const int32_t *xn,
    const int NT, float *out, size_t ostride)
{
    int nb = cols / 32;
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 flo[4], fhi[4];
    for (int t = 0; t < NT; ++t) {
        flo[t] = _mm256_setzero_ps();
        fhi[t] = _mm256_setzero_ps();
    }
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        __m256i slo[4], shi[4];
        for (int t = 0; t < NT; ++t) {
            slo[t] = _mm256_setzero_si256();
            shi[t] = _mm256_setzero_si256();
        }
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            for (int t = 0; t < NT; ++t) {
                int32_t x4;
                memcpy(&x4, xq + (size_t)t * cols + 32 * b + 4 * k, 4);
                __m256i xv = _mm256_set1_epi32(x4);
                slo[t] = _mm256_add_epi16(slo[t], _mm256_maddubs_epi16(lo, xv));
                shi[t] = _mm256_add_epi16(shi[t], _mm256_maddubs_epi16(hi, xv));
            }
        }
        __m256 dlo = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 256)));
        __m256 dhi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 272)));
        for (int t = 0; t < NT; ++t) {
            __m256i nv = _mm256_set1_epi32(xn[(size_t)t * (cols / 16) + b]);
            __m256i alo = _mm256_add_epi32(_mm256_madd_epi16(slo[t], ones), nv);
            __m256i ahi = _mm256_add_epi32(_mm256_madd_epi16(shi[t], ones), nv);
            __m256 sx = _mm256_set1_ps(xs[(size_t)t * nb + b]);
            flo[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(alo), _mm256_mul_ps(dlo, sx), flo[t]);
            fhi[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ahi), _mm256_mul_ps(dhi, sx), fhi[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        _mm256_storeu_ps(out + (size_t)t * ostride, flo[t]);
        _mm256_storeu_ps(out + (size_t)t * ostride + 8, fhi[t]);
    }
}

static void kq_q4x_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32;
    /* Up to 4 tokens for each unpack of the codes (the sums of each token
     * do not change). The 26B prompt of 512 tokens, 6 threads: 61.3 tok/s
     * with 2, 64.9 with 4. */
    for (int j = 0; j < n;) {
        const int8_t *qj = xq + (size_t)j * cols;
        const float *sj = xs + (size_t)j * nb;
        const int32_t *nj = xn + (size_t)j * (cols / 16);
        int left = n - j;
        if (left >= 4) {
            kq_q4x_rows_n(wg, cols, qj, sj, nj, 4, out + (size_t)j * ostride, ostride);
            j += 4;
        } else if (left >= 3) {
            kq_q4x_rows_n(wg, cols, qj, sj, nj, 3, out + (size_t)j * ostride, ostride);
            j += 3;
        } else if (left >= 2) {
            kq_q4x_rows_n(wg, cols, qj, sj, nj, 2, out + (size_t)j * ostride, ostride);
            j += 2;
        } else {
            kq_q4x_rows_n(wg, cols, qj, sj, nj, 1, out + (size_t)j * ostride, ostride);
            j += 1;
        }
    }
}

/* ---- int16 x on KQ_Q4X, AVX2 (the prompt with NP_GEMMA_INT4_Q8=16) ----
 *
 * The int8 x of the prompt cost the AVX2 build most of its distance from
 * float32 (the 26B, 64 tokens and 40 steps against the NumPy decode: KL
 * 0.22 with int8 x, 0.064 with float32 prompt products). As the VNNI form
 * above: the codes less 8 as int16, unpacked once for a chunk of a group,
 * then vpmaddwd (8 lanes of 2 products) of a pair of columns of the 16 rows
 * by the 2 int16 x of the pair, added as int32 (a block: at most 32 * 32767
 * * 8 < 2^31). Twice the instructions of the int8 form (vpmaddubsw, 4
 * columns a lane). The arithmetic of a token does not depend on the tile:
 * a token alone and in a group give the same bits.
 *
 * kq_q4x_unpack16: each block of 32 columns of a group of 16 rows gives 32
 * vectors: vector 2 j + h holds, in lane r, the weights of row 8 h + r at
 * columns 2 j and 2 j + 1. */
static void kq_q4x_unpack16(const uint8_t *wg, int nb, int16_t *u, float *dsc)
{
    const __m256i m4 = _mm256_set1_epi8(15), eight = _mm256_set1_epi16(8);
    /* bytes 0, 1 (columns 4 k, 4 k + 1) and 2, 3 of each 32-bit lane to int16 */
    const __m256i sa = _mm256_setr_epi8(0, -1, 1, -1, 4, -1, 5, -1, 8, -1, 9, -1, 12, -1, 13, -1,
                                        0, -1, 1, -1, 4, -1, 5, -1, 8, -1, 9, -1, 12, -1, 13, -1);
    const __m256i sb = _mm256_setr_epi8(2, -1, 3, -1, 6, -1, 7, -1, 10, -1, 11, -1, 14, -1, 15, -1,
                                        2, -1, 3, -1, 6, -1, 7, -1, 10, -1, 11, -1, 14, -1, 15, -1);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        int16_t *ub = u + (size_t)b * 32 * 16;
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            /* the lanes of a 32-byte step: rows 0 to 7 (low 4 bits), 8 to
             * 15 (high), columns 4 k to 4 k + 3 */
            _mm256_store_si256((__m256i *)(ub + (4 * k + 0) * 16),
                               _mm256_sub_epi16(_mm256_shuffle_epi8(lo, sa), eight));
            _mm256_store_si256((__m256i *)(ub + (4 * k + 1) * 16),
                               _mm256_sub_epi16(_mm256_shuffle_epi8(hi, sa), eight));
            _mm256_store_si256((__m256i *)(ub + (4 * k + 2) * 16),
                               _mm256_sub_epi16(_mm256_shuffle_epi8(lo, sb), eight));
            _mm256_store_si256((__m256i *)(ub + (4 * k + 3) * 16),
                               _mm256_sub_epi16(_mm256_shuffle_epi8(hi, sb), eight));
        }
        _mm256_store_ps(dsc + (size_t)b * 16,
                        _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 256))));
        _mm256_store_ps(dsc + (size_t)b * 16 + 8,
                        _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 272))));
    }
}

/* A group of 16 rows on T tokens (1 to KQ16_TMAX), nb blocks of 32 columns
 * (a chunk of the row): the int16 x of token t at xq + t * xstride, its
 * scales at xs + t * sstride: out[t * ostride + r]. The int32 sum of a
 * block, then the scale of the block of the row times the scale of x, block
 * after block. ACC 1 goes on from the sums in out (the chunk before). */
#define KQ16_TMAX 4
static inline __attribute__((always_inline)) void kq_q4x_tile16(
    const int16_t *u, const float *d, int nb, size_t xstride, size_t sstride, const int16_t *xq,
    const float *xs, float *out, size_t ostride, const int T, const int ACC)
{
    __m256 flo[KQ16_TMAX], fhi[KQ16_TMAX];
    for (int t = 0; t < T; ++t) {
        flo[t] = ACC ? _mm256_loadu_ps(out + (size_t)t * ostride) : _mm256_setzero_ps();
        fhi[t] = ACC ? _mm256_loadu_ps(out + (size_t)t * ostride + 8) : _mm256_setzero_ps();
    }
    for (int b = 0; b < nb; ++b) {
        const int16_t *ub = u + (size_t)b * 32 * 16;
        __m256i alo[KQ16_TMAX], ahi[KQ16_TMAX];
        for (int t = 0; t < T; ++t) {
            alo[t] = _mm256_setzero_si256();
            ahi[t] = _mm256_setzero_si256();
        }
        for (int j = 0; j < 16; ++j) {
            __m256i w0 = _mm256_load_si256((const __m256i *)(ub + (2 * j) * 16));
            __m256i w1 = _mm256_load_si256((const __m256i *)(ub + (2 * j + 1) * 16));
            for (int t = 0; t < T; ++t) {
                int32_t x2;
                memcpy(&x2, xq + (size_t)t * xstride + 32 * b + 2 * j, 4);
                __m256i xv = _mm256_set1_epi32(x2);
                alo[t] = _mm256_add_epi32(alo[t], _mm256_madd_epi16(w0, xv));
                ahi[t] = _mm256_add_epi32(ahi[t], _mm256_madd_epi16(w1, xv));
            }
        }
        __m256 dlo = _mm256_load_ps(d + (size_t)b * 16), dhi = _mm256_load_ps(d + (size_t)b * 16 + 8);
        for (int t = 0; t < T; ++t) {
            __m256 sx = _mm256_set1_ps(xs[(size_t)t * sstride + b]);
            flo[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(alo[t]), _mm256_mul_ps(dlo, sx), flo[t]);
            fhi[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ahi[t]), _mm256_mul_ps(dhi, sx), fhi[t]);
        }
    }
    for (int t = 0; t < T; ++t) {
        _mm256_storeu_ps(out + (size_t)t * ostride, flo[t]);
        _mm256_storeu_ps(out + (size_t)t * ostride + 8, fhi[t]);
    }
}

/* The scratch of the unpacked codes of a thread (one group, int16). */
static __thread int16_t *kq16_u;
static __thread float *kq16_d;
static __thread size_t kq16_un;

static void kq16_scratch(int nb)
{
    const size_t need = (size_t)nb * 16 * 32;
    if (kq16_un < need) {
        free(kq16_u);
        free(kq16_d);
        kq16_u = (int16_t *)aligned_alloc(64, need * sizeof(int16_t));
        kq16_d = (float *)aligned_alloc(64, (size_t)nb * 16 * sizeof(float));
        kq16_un = need;
    }
}

/* The blocks of a chunk of columns: the unpacked codes of a chunk (1 KB a
 * block) and the x of a block of tokens stay in the L2 (256 KB on the
 * Core i5-8500 of the target). */
#define KQ16_CHUNK 64

static inline void kq_q4x_tile16_t(const int16_t *u, const float *d, int nc, int cols, int nb,
                                   const int16_t *xj, const float *sj, float *oj, size_t ostride,
                                   int T, int acc)
{
#define KQ16_T(TT) do { if (acc) kq_q4x_tile16(u, d, nc, cols, nb, xj, sj, oj, ostride, TT, 1); \
                        else kq_q4x_tile16(u, d, nc, cols, nb, xj, sj, oj, ostride, TT, 0); } while (0)
    switch (T) {
    case 4: KQ16_T(4); break;
    case 3: KQ16_T(3); break;
    case 2: KQ16_T(2); break;
    default: KQ16_T(1); break;
    }
#undef KQ16_T
}

/* One token of int16 x without the unpack (a step): x = 256 xh + xl with
 * xh = x >> 8 (signed) and xl = x & 255 (unsigned), two int8 planes for
 * vpmaddubsw as the int8 form (the codes the unsigned operand with xh, the
 * signed one with xl). The sum of a block, 256 sum(c xh) + sum(c xl) - 8
 * sum(x), is the int32 sum of kq_q4x_tile16, and the float steps are its
 * steps: the same bits as a group. The int16 sums of xh hold 8 steps (8 * 2
 * * 15 * 128 = 30720), those of xl 4 (4 * 2 * 255 * 15 = 30600). */
/* The planes of the int16 row xq (cols values): h, l and -8 times the sum
 * of each block. */
static void kq_planes16_to(const int16_t *xq, int cols, int8_t *kp_h, uint8_t *kp_l, int32_t *kp_n)
{
    const int nb = cols / 32;
    const __m256i m8 = _mm256_set1_epi16(255), ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; ++b) {
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(xq + 32 * b));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(xq + 32 * b + 16));
        __m256i h = _mm256_packs_epi16(_mm256_srai_epi16(v0, 8), _mm256_srai_epi16(v1, 8));
        __m256i l = _mm256_packus_epi16(_mm256_and_si256(v0, m8), _mm256_and_si256(v1, m8));
        _mm256_storeu_si256((__m256i *)(kp_h + 32 * b), _mm256_permute4x64_epi64(h, 0xD8));
        _mm256_storeu_si256((__m256i *)(kp_l + 32 * b), _mm256_permute4x64_epi64(l, 0xD8));
        __m256i s = _mm256_add_epi32(_mm256_madd_epi16(v0, ones), _mm256_madd_epi16(v1, ones));
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        kp_n[b] = -8 * _mm_cvtsi128_si32(s4);
    }
}

/* The planes of a row in the scratch of a thread. */
static __thread int8_t *kp_h;
static __thread uint8_t *kp_l;
static __thread int32_t *kp_n;
static __thread size_t kp_cap;

static void kq_planes16(const int16_t *xq, int cols)
{
    if (kp_cap < (size_t)cols) {
        free(kp_h);
        free(kp_l);
        free(kp_n);
        kp_h = (int8_t *)aligned_alloc(64, (size_t)cols + 64);
        kp_l = (uint8_t *)aligned_alloc(64, (size_t)cols + 64);
        kp_n = (int32_t *)aligned_alloc(64, ((size_t)cols / 32 + 16) * sizeof(int32_t));
        kp_cap = cols;
    }
    kq_planes16_to(xq, cols, kp_h, kp_l, kp_n);
}

static void kq_q4x_rows_p16(const uint8_t *wg, int cols, const int8_t *kp_h, const uint8_t *kp_l,
                            const int32_t *kp_n, const float *xs, float *out)
{
    const int nb = cols / 32;
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 flo = _mm256_setzero_ps(), fhi = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        __m256i hlo = _mm256_setzero_si256(), hhi = _mm256_setzero_si256();
        __m256i llo = _mm256_setzero_si256(), lhi = _mm256_setzero_si256();
        __m256i wlo = _mm256_setzero_si256(), whi = _mm256_setzero_si256();
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i cl = _mm256_and_si256(q, m4), ch = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            int32_t h4, l4;
            memcpy(&h4, kp_h + 32 * b + 4 * k, 4);
            memcpy(&l4, kp_l + 32 * b + 4 * k, 4);
            __m256i vh = _mm256_set1_epi32(h4), vl = _mm256_set1_epi32(l4);
            hlo = _mm256_add_epi16(hlo, _mm256_maddubs_epi16(cl, vh));
            hhi = _mm256_add_epi16(hhi, _mm256_maddubs_epi16(ch, vh));
            llo = _mm256_add_epi16(llo, _mm256_maddubs_epi16(vl, cl));
            lhi = _mm256_add_epi16(lhi, _mm256_maddubs_epi16(vl, ch));
            if (k == 3) {
                wlo = _mm256_madd_epi16(llo, ones);
                whi = _mm256_madd_epi16(lhi, ones);
                llo = lhi = _mm256_setzero_si256();
            }
        }
        __m256i nv = _mm256_set1_epi32(kp_n[b]);
        __m256i alo = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_slli_epi32(_mm256_madd_epi16(hlo, ones), 8),
                             _mm256_add_epi32(wlo, _mm256_madd_epi16(llo, ones))), nv);
        __m256i ahi = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_slli_epi32(_mm256_madd_epi16(hhi, ones), 8),
                             _mm256_add_epi32(whi, _mm256_madd_epi16(lhi, ones))), nv);
        __m256 dlo = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 256)));
        __m256 dhi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 272)));
        __m256 sx = _mm256_set1_ps(xs[b]);
        flo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(alo), _mm256_mul_ps(dlo, sx), flo);
        fhi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ahi), _mm256_mul_ps(dhi, sx), fhi);
    }
    _mm256_storeu_ps(out, flo);
    _mm256_storeu_ps(out + 8, fhi);
}

/* One group of 16 rows on n tokens (an expert of kq_moe_body; a step).
 * A token alone takes kq_q4x_rows_p16 (the same bits). */
static void kq_q4x_rows16(const uint8_t *wg, int cols, const int16_t *xq, const float *xs, int n,
                          float *out, size_t ostride)
{
    const int nb = cols / 32;
    if (n == 1) {
        kq_planes16(xq, cols);
        kq_q4x_rows_p16(wg, cols, kp_h, kp_l, kp_n, xs, out);
        return;
    }
    kq16_scratch(nb < KQ16_CHUNK ? nb : KQ16_CHUNK);
    for (int c0 = 0; c0 < nb; c0 += KQ16_CHUNK) {
        const int nc = nb - c0 < KQ16_CHUNK ? nb - c0 : KQ16_CHUNK;
        kq_q4x_unpack16(wg + (size_t)c0 * KQ_Q4X_BB, nc, kq16_u, kq16_d);
        for (int j = 0; j < n; j += KQ16_TMAX) {
            const int T = n - j < KQ16_TMAX ? n - j : KQ16_TMAX;
            kq_q4x_tile16_t(kq16_u, kq16_d, nc, cols, nb, xq + (size_t)j * cols + 32 * (size_t)c0,
                            xs + (size_t)j * nb + c0, out + (size_t)j * ostride, ostride, T, c0 > 0);
        }
    }
}

/* KQ_Q4X on t tokens of int16 x, inside a parallel region: a task is a
 * block of tokens of a group. */
static void kq_q4x_gemm16(const uint8_t *w, int rows, int cols, const int16_t *xq, const float *xs,
                          int t, float *out)
{
    const int nb = cols / 32, groups = rows / 16;
    const size_t gb = 16 * kq_row_bytes(KQ_Q4X, cols);
    if (t == 1) {
        /* one token (the head of a step): the planes once for each thread */
        kq_planes16(xq, cols);
        #pragma omp for schedule(static)
        for (int g = 0; g < groups; ++g) {
            kq_q4x_rows_p16(w + (size_t)g * gb, cols, kp_h, kp_l, kp_n, xs, out + 16 * g);
        }
        return;
    }
    const int cb = nb < KQ16_CHUNK ? nb : KQ16_CHUNK;
    kq16_scratch(cb);
    /* the int16 x of a block of tokens about 64 KB */
    int tb = (32 * 1024 / cols) & ~3;
    tb = tb < 8 ? 8 : (tb > 128 ? 128 : tb);
    const int blocks = (t + tb - 1) / tb;
    tb = ((t + blocks - 1) / blocks + 3) & ~3;
    #pragma omp for schedule(static)
    for (int it = 0; it < blocks * groups; ++it) {
        const int j0 = (it / groups) * tb, g = it % groups;
        if (j0 >= t) {
            continue;
        }
        const int nt = t - j0 < tb ? t - j0 : tb;
        for (int c0 = 0; c0 < nb; c0 += cb) {
            const int nc = nb - c0 < cb ? nb - c0 : cb;
            kq_q4x_unpack16(w + (size_t)g * gb + (size_t)c0 * KQ_Q4X_BB, nc, kq16_u, kq16_d);
            for (int jj = 0; jj < nt; jj += KQ16_TMAX) {
                const int j = j0 + jj, T = nt - jj < KQ16_TMAX ? nt - jj : KQ16_TMAX;
                kq_q4x_tile16_t(kq16_u, kq16_d, nc, cols, nb, xq + (size_t)j * cols + 32 * (size_t)c0,
                                xs + (size_t)j * nb + c0, out + (size_t)j * rows + 16 * g,
                                (size_t)rows, T, c0 > 0);
            }
        }
    }
}
#define KQ_Q4X_Q16 1
#else
/* Without VNNI: the same sums in C. */
static void kq_q4x_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32;
    (void)xn;
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
                int32_t a = 0;
                for (int v = 0; v < 32; ++v) {
                    int c = (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
                    a += (c - 8) * xq[(size_t)j * cols + 32 * b + v];
                }
                f += (float)a * kq_h(blk + 256 + 2 * r) * xs[(size_t)j * nb + b];
            }
            out[(size_t)j * ostride + r] = f;
        }
    }
}
#endif

/* The 16 rows of the KQ_Q4X group at wg on n rows of float32 x (row stride
 * xstride): out[j * ostride + r]. No quantization of x: the decode of the
 * Gemma 4 26B keeps float32 activations (the MOE record gives the float
 * products to 4e-6). For each step, the codes of the 16 rows (a lane is a
 * row) to float32, 4 values at a time, and one fma for each value. */
#if defined(__AVX512F__)
/* kq_q4x_rows_f for a constant count nt (1 to 4) of tokens: their sums stay
 * in registers. Each token adds its terms in the order of one token (k, then
 * u), so a group gives the bits of the steps. */
static inline __attribute__((always_inline)) void kq_q4x_rows_fn(const uint8_t *wg, int nb,
                                                                  const float *x, size_t xstride,
                                                                  const int nt, float *out,
                                                                  size_t ostride)
{
    const __m256i m4 = _mm256_set1_epi8(15);
    const __m512i mff = _mm512_set1_epi32(255), eight = _mm512_set1_epi32(8);
    __m512 f[4];
    for (int j = 0; j < nt; ++j) {
        f[j] = _mm512_setzero_ps();
    }
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        __m512 acc[4];
        for (int j = 0; j < nt; ++j) {
            acc[j] = _mm512_setzero_ps();
        }
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            __m512i W = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            for (int u = 0; u < 4; ++u) {
                __m512 cf = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                    _mm512_and_si512(_mm512_srli_epi32(W, 8 * u), mff), eight));
                for (int j = 0; j < nt; ++j) {
                    acc[j] = _mm512_fmadd_ps(cf, _mm512_set1_ps(x[(size_t)j * xstride + 32 * b + 4 * k + u]),
                                             acc[j]);
                }
            }
        }
        __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 256)));
        for (int j = 0; j < nt; ++j) {
            f[j] = _mm512_fmadd_ps(acc[j], sc, f[j]);
        }
    }
    for (int j = 0; j < nt; ++j) {
        _mm512_storeu_ps(out + (size_t)j * ostride, f[j]);
    }
}
#endif

static void kq_q4x_rows_f(const uint8_t *wg, int cols, const float *x, size_t xstride, int n,
                          float *out, size_t ostride)
{
    int nb = cols / 32;
#if defined(__AVX512F__)
    /* Up to 4 tokens at a time: the codes become float32 once for all of
     * them (kq_q4x_rows_fn). */
    for (int j0 = 0; j0 < n; j0 += 4) {
        const float *xj = x + (size_t)j0 * xstride;
        float *oj = out + (size_t)j0 * ostride;
        switch (n - j0 < 4 ? n - j0 : 4) {
        case 1: kq_q4x_rows_fn(wg, nb, xj, xstride, 1, oj, ostride); break;
        case 2: kq_q4x_rows_fn(wg, nb, xj, xstride, 2, oj, ostride); break;
        case 3: kq_q4x_rows_fn(wg, nb, xj, xstride, 3, oj, ostride); break;
        default: kq_q4x_rows_fn(wg, nb, xj, xstride, 4, oj, ostride); break;
        }
    }
#else
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
                float acc = 0.f;
                for (int v = 0; v < 32; ++v) {
                    int c = (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
                    acc += (float)(c - 8) * x[(size_t)j * xstride + 32 * b + v];
                }
                f += acc * kq_h(blk + 256 + 2 * r);
            }
            out[(size_t)j * ostride + r] = f;
        }
    }
#endif
}

/* KQ_Q4X: out (t x rows), inside a parallel region, as kq_nvx_body. */
/* The int16 x of t rows of cols values (inside a parallel region): a scale
 * for each 32 (gemma_quant_group32_i16, as the int16 tile of the prompt). */
static void kq_quant16_body(const float *x, int t, int cols, int16_t *xq, float *xs)
{
    const int nb = cols / 32;
    #pragma omp for schedule(static)
    for (long i = 0; i < (long)t * nb; ++i) {
        const long j = i / nb, g = i % nb;
        xs[j * nb + g] = gemma_quant_group32_i16(x + (size_t)j * cols + 32 * g,
                                                 xq + (size_t)j * cols + 32 * g);
    }
}

/* 1 when the build has the int16 kernels of the prompt (VNNI, AVX2). */
int kq_q16_ok(void)
{
#ifdef KQ_Q4X_Q16
    return 1;
#else
    return 0;
#endif
}

/* KQ_Q4X on t tokens of int16 x (kq_quant16_body), inside a parallel region. */
static void kq_linear16_body(const uint8_t *w, int rows, int cols, const int16_t *xq,
                             const float *xs, int t, float *out)
{
#ifdef KQ_Q4X_Q16
    kq_q4x_gemm16(w, rows, cols, xq, xs, t, out);
#else
    (void)w; (void)rows; (void)cols; (void)xq; (void)xs; (void)t; (void)out;
#endif
}

/* The int16 x and the product, for a check or a benchmark (bench_q4x_gemm.py). */
void kq_linear16(const uint8_t *w, int rows, int cols, const float *x, int16_t *xq, float *xs,
                 int t, float *out)
{
    #pragma omp parallel
    {
        kq_quant16_body(x, t, cols, xq, xs);
        kq_linear16_body(w, rows, cols, xq, xs, t, out);
    }
}

/* NP_GEMMA_Q4X_GEMM=0 keeps kq_q4x_rows for a group of tokens (a check). */
static int getenv_gemm_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_Q4X_GEMM");
        on = !(v && v[0] == '0');
    }
    return on;
}

static void kq_q4x_body(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        int t, float *out)
{
    int32_t *xn = NULL;
    size_t gb = 16 * kq_row_bytes(KQ_Q4X, cols);
    #pragma omp single copyprivate(xn)
    xn = (int32_t *)malloc((size_t)t * (cols / 16) * 4);
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        kq_q4x_xsum(xq + (size_t)j * cols, 1, cols, xn + (size_t)j * (cols / 16));
    }
#ifdef KQ_Q4X_GEMM
    if (t >= 4 && getenv_gemm_on()) {
        kq_q4x_gemm(w, rows, cols, xq, xs, xn, t, out);
        #pragma omp barrier
        #pragma omp single
        free(xn);
        return;
    }
#endif
    for (int j0 = 0; j0 < t; j0 += MA_TB) {
        int nt = t - j0 < MA_TB ? t - j0 : MA_TB;
        #pragma omp for schedule(static) nowait
        for (int g = 0; g < rows / 16; ++g) {
            kq_q4x_rows(w + (size_t)g * gb, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * (cols / 32),
                        xn + (size_t)j0 * (cols / 16), nt, out + (size_t)j0 * rows + 16 * g,
                        (size_t)rows);
        }
    }
    #pragma omp barrier
    #pragma omp single
    free(xn);
}

/* KQ_NVX: out (t x rows), inside a parallel region; blocks of MA_TB tokens
 * with the groups of 16 rows over the threads. */
static void kq_nvx_body(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        int t, float *out)
{
    int32_t *xn = NULL;
    size_t gb = 16 * kq_row_bytes(KQ_NVX, cols);
    #pragma omp single copyprivate(xn)
    xn = (int32_t *)malloc((size_t)t * (cols / 16) * 4);
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        kq_nvx_xsum(xq + (size_t)j * cols, 1, cols, xn + (size_t)j * (cols / 16));
    }
    for (int j0 = 0; j0 < t; j0 += MA_TB) {
        int nt = t - j0 < MA_TB ? t - j0 : MA_TB;
        #pragma omp for schedule(static) nowait
        for (int g = 0; g < rows / 16; ++g) {
            kq_nvx_rows(w + (size_t)g * gb, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * (cols / 32),
                        xn + (size_t)j0 * (cols / 16), nt, out + (size_t)j0 * rows + 16 * g,
                        (size_t)rows);
        }
    }
    #pragma omp barrier
    #pragma omp single
    free(xn);
}

#if defined(__AVX512F__)
#ifndef KQ_X16F_GB
#define KQ_X16F_GB 4
#endif

/* ng (at most 4) groups of 16 rows (gb bytes apart; the last has nr rows)
 * on at most 4 tokens j0 .. j1 - 1: the groups give independent sums, so
 * the fma of a token do not wait on each other (one group on one token
 * waits 4 cycles for each column). A missing group repeats group 0 and is
 * not stored. For each row and token: one fma for each column, in order, as
 * kq_x16f_group16: the same bits. */
/* Column c of a group of 16 rows of a KQ_BF16X16 (bf 1), KQ_BF12X16 (bf 2),
 * or KQ_F32X16 (bf 0) matrix as 16 floats. KQ_BF12X16: in each block of 32
 * columns (784 bytes) the 16 rows' sign and mantissa bytes column by column
 * (512), their gap nibbles (256: byte 16 j + r has the gap of column j of
 * row r in its low half, of column j + 16 in its high half), and their
 * largest exponents (16); the bits of BF12 (kq_bf12_bits). */
static inline __attribute__((always_inline)) __m512 kq_x16f_col(const uint8_t *wg, int bf, int c)
{
    if (bf == 1) {
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
            _mm256_loadu_si256((const __m256i *)(wg + (size_t)32 * c))), 16));
    }
    if (bf == 2) {
        const uint8_t *blk = wg + (size_t)(c >> 5) * 784;
        int cc = c & 31;
        __m512i l = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(blk + 16 * cc)));
        __m512i h = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(blk + 512 + 16 * (cc & 15))));
        __m512i gap = _mm512_and_si512(_mm512_srlv_epi32(h, _mm512_set1_epi32(4 * (cc >> 4))),
                                       _mm512_set1_epi32(15));
        __m512i e = _mm512_and_si512(_mm512_sub_epi32(_mm512_cvtepu8_epi32(
                        _mm_loadu_si128((const __m128i *)(blk + 768))), gap), _mm512_set1_epi32(255));
        __m512i b = _mm512_or_si512(_mm512_or_si512(
                        _mm512_slli_epi32(_mm512_and_si512(l, _mm512_set1_epi32(0x80)), 24),
                        _mm512_slli_epi32(e, 23)),
                        _mm512_slli_epi32(_mm512_and_si512(l, _mm512_set1_epi32(0x7f)), 16));
        __mmask16 z = _mm512_cmpeq_epi32_mask(gap, _mm512_set1_epi32(15)) &
                      _mm512_cmpeq_epi32_mask(l, _mm512_set1_epi32(0x80));
        return _mm512_castsi512_ps(_mm512_maskz_mov_epi32((__mmask16)~z, b));
    }
    return _mm512_loadu_ps((const float *)(wg + (size_t)64 * c));
}

static void kq_x16f_groups4(const uint8_t *wg, size_t gb, int ng, int nr, int bf, int cols,
                            size_t xstride, const float *x, int j0, int j1, float *out,
                            size_t ostride)
{
    int nt = j1 - j0;
    const uint8_t *gp[4];
    for (int g = 0; g < 4; ++g) {
        gp[g] = wg + (g < ng ? g : 0) * gb;
    }
    __m512 f[4][4];
    for (int g = 0; g < 4; ++g) {
        for (int tt = 0; tt < 4; ++tt) {
            f[g][tt] = _mm512_setzero_ps();
        }
    }
    const float *xr = x + (size_t)j0 * xstride;
    for (int c = 0; c < cols; ++c) {
        __m512 wv[4];
        #pragma GCC unroll 4
        for (int g = 0; g < 4; ++g) {
            wv[g] = kq_x16f_col(gp[g], bf, c);
        }
        #pragma GCC unroll 4
        for (int tt = 0; tt < 4; ++tt) {
            if (tt < nt) {
                __m512 xv = _mm512_set1_ps(xr[(size_t)tt * xstride + c]);
                #pragma GCC unroll 4
                for (int g = 0; g < 4; ++g) {
                    f[g][tt] = _mm512_fmadd_ps(wv[g], xv, f[g][tt]);
                }
            }
        }
    }
    for (int tt = 0; tt < nt; ++tt) {
        for (int g = 0; g < ng; ++g) {
            int n = g == ng - 1 ? nr : 16;
            __mmask16 mk = (__mmask16)(n >= 16 ? 0xffff : (1u << n) - 1);
            _mm512_mask_storeu_ps(out + (size_t)(j0 + tt) * ostride + 16 * g, mk, f[g][tt]);
        }
    }
}

/* One group of 16 rows on tokens j0 .. j1 - 1, 16 tokens at a time, with
 * loops of fixed counts (the 16 sums stay in registers; each broadcast is
 * an operand of its fma). A batch of fewer than 16 tokens repeats its last
 * token and does not store it. */
static void kq_x16f_group16(const uint8_t *wg, int bf, int cols, size_t xstride, const float *x,
                            int j0, int j1, int nr, float *out, size_t ostride)
{
    __mmask16 mk = (__mmask16)(nr >= 16 ? 0xffff : (1u << nr) - 1);
    for (int j = j0; j < j1; j += 16) {
        int nt = j1 - j < 16 ? j1 - j : 16;
        const float *xp[16];
        for (int tt = 0; tt < 16; ++tt) {
            xp[tt] = x + (size_t)(j + (tt < nt ? tt : nt - 1)) * xstride;
        }
        __m512 f[16];
        for (int tt = 0; tt < 16; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int c = 0; c < cols; ++c) {
            __m512 wv = kq_x16f_col(wg, bf, c);
            #pragma GCC unroll 16
            for (int tt = 0; tt < 16; ++tt) {
                f[tt] = _mm512_fmadd_ps(wv, _mm512_set1_ps(xp[tt][c]), f[tt]);
            }
        }
        #pragma GCC unroll 16
        for (int tt = 0; tt < 16; ++tt) {
            if (tt < nt) {
                _mm512_mask_storeu_ps(out + (size_t)(j + tt) * ostride, mk, f[tt]);
            }
        }
    }
}
#elif defined(__AVX2__)
/* AVX2: one group of 16 rows (two vectors of 8) on tokens j0 .. j1 - 1, 6
 * tokens at a time (12 sums, 2 vectors of W, and the broadcast take 15 of
 * the 16 registers). One fma for each row, token, and column, in order. */
static void kq_x16f_group(const uint8_t *wg, int bf, int cols, size_t xstride, const float *x,
                          int j0, int j1, int nr, float *out, size_t ostride)
{
    for (int j = j0; j < j1; j += 6) {
        int nt = j1 - j < 6 ? j1 - j : 6;
        const float *xp[6];
        for (int tt = 0; tt < 6; ++tt) {
            xp[tt] = x + (size_t)(j + (tt < nt ? tt : nt - 1)) * xstride;
        }
        __m256 f[6][2];
        for (int tt = 0; tt < 6; ++tt) {
            f[tt][0] = _mm256_setzero_ps();
            f[tt][1] = _mm256_setzero_ps();
        }
        for (int c = 0; c < cols; ++c) {
            __m256 w0, w1;
            if (bf == 2) {
                float tmp[16];
                kq_bf12x16_col(wg, c, tmp);
                w0 = _mm256_loadu_ps(tmp);
                w1 = _mm256_loadu_ps(tmp + 8);
            } else if (bf) {
                __m256i u = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(wg + (size_t)32 * c)));
                __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(wg + (size_t)32 * c + 16)));
                w0 = _mm256_castsi256_ps(_mm256_slli_epi32(u, 16));
                w1 = _mm256_castsi256_ps(_mm256_slli_epi32(v, 16));
            } else {
                w0 = _mm256_loadu_ps((const float *)(wg + (size_t)64 * c));
                w1 = _mm256_loadu_ps((const float *)(wg + (size_t)64 * c + 32));
            }
            #pragma GCC unroll 6
            for (int tt = 0; tt < 6; ++tt) {
                __m256 xv = _mm256_set1_ps(xp[tt][c]);
                f[tt][0] = _mm256_fmadd_ps(w0, xv, f[tt][0]);
                f[tt][1] = _mm256_fmadd_ps(w1, xv, f[tt][1]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            float buf[16];
            _mm256_storeu_ps(buf, f[tt][0]);
            _mm256_storeu_ps(buf + 8, f[tt][1]);
            float *o = out + (size_t)(j + tt) * ostride;
            for (int r = 0; r < nr; ++r) {
                o[r] = buf[r];
            }
        }
    }
}
#else
/* Without AVX2: the same sums in C (not the bits of the AVX-512 path). */
static void kq_x16f_group(const uint8_t *wg, int bf, int cols, size_t xstride, const float *x,
                          int j0, int j1, int nr, float *out, size_t ostride)
{
    for (int j = j0; j < j1; ++j) {
        for (int r = 0; r < nr; ++r) {
            float s = 0.f;
            for (int c = 0; c < cols; ++c) {
                float v;
                if (bf == 2) {
                    float tmp[16];
                    kq_bf12x16_col(wg, c, tmp);
                    v = tmp[r];
                } else if (bf) {
                    uint32_t u = (uint32_t)((const uint16_t *)wg)[16 * c + r] << 16;
                    memcpy(&v, &u, 4);
                } else {
                    v = ((const float *)wg)[16 * c + r];
                }
                s += v * x[(size_t)j * xstride + c];
            }
            out[(size_t)j * ostride + r] = s;
        }
    }
}
#endif

/* out (t x rows) of a KQ_BF16X16 or KQ_F32X16 matrix, inside a parallel
 * region. A task: at most 4 tokens, 4 groups or 1 (kq_x16f_groups4); more
 * tokens, KQ_X16F_GB groups on 16 tokens, so x comes from the cache of the
 * core for all of them (kq_x16f_group16). Tiles of 4 groups by 64 tokens in
 * chunks of 256 columns, with x as [column][token], were slower (44 to 48
 * ms, not 28, for 10240 x 2560 on 512 tokens). */
static void kq_x16f_body(const uint8_t *w, int type, int rows, int cols, const float *x, int t,
                         float *out)
{
    int ng = (rows + 15) / 16, bf = type == KQ_BF16X16 ? 1 : (type == KQ_BF12X16 ? 2 : 0);
    size_t gb = (size_t)16 * kq_row_bytes(type, cols);
    int u = 1, tk = 16;
#if defined(__AVX512F__)
    if (t <= 4) {
        /* 4 groups in a task when there are enough for all the threads */
        u = ng >= 4 * omp_get_num_threads() ? 4 : 1, tk = 4;
    } else {
        u = KQ_X16F_GB;
    }
#endif
    int nu = (ng + u - 1) / u, nj = (t + tk - 1) / tk;
    #pragma omp for schedule(static)
    for (int e = 0; e < nu * nj; ++e) {
        int g = u * (e / nj), j0 = tk * (e % nj), j1 = j0 + tk < t ? j0 + tk : t;
        int n = ng - g < u ? ng - g : u;                   /* the groups of the task */
        int last = rows - 16 * (g + n - 1);                /* the rows of its last group */
        last = last < 16 ? last : 16;
        const uint8_t *wg = w + (size_t)g * gb;
        float *o = out + (size_t)16 * g;
#if defined(__AVX512F__)
        if (t <= 4) {
            kq_x16f_groups4(wg, gb, n, last, bf, cols, (size_t)cols, x, j0, j1, o, (size_t)rows);
            continue;
        }
        for (int gi = 0; gi < n; ++gi) {
            kq_x16f_group16(wg + gi * gb, bf, cols, (size_t)cols, x, j0, j1, gi == n - 1 ? last : 16,
                            o + 16 * gi, (size_t)rows);
        }
#else
        kq_x16f_group(wg, bf, cols, (size_t)cols, x, j0, j1, last, o, (size_t)rows);
#endif
    }
}

static void kq_linear_body(const uint8_t *w, int type, int rows, int cols, const int8_t *xq,
                           const float *xs, const float *xm, const float *x, int t, float *out)
{
    if (type == KQ_BF16X16 || type == KQ_F32X16 || type == KQ_BF12X16) {
        kq_x16f_body(w, type, rows, cols, x, t, out);
        return;
    }
    if (type == KQ_NVX) {
        kq_nvx_body(w, rows, cols, xq, xs, t, out);
        return;
    }
    if (type == KQ_Q4X) {
        kq_q4x_body(w, rows, cols, xq, xs, t, out);
        return;
    }
    size_t rb = kq_row_bytes(type, cols);
#if defined(__AVX512VNNI__) || defined(__AVX2__)
    if (type == KQ_Q8X16) {
        kq_x16_body(w, rows, cols, xq, xs, t, out);
        return;
    }
#endif
#if defined(__AVX512VNNI__)
    if (t >= 4 && rows % 4 == 0 && kq_tiles(type, cols)) {
        /* Blocks of MA_TB tokens, as ma_linear_body. */
        for (int j0 = 0; j0 < t; j0 += MA_TB) {
            int nb = t - j0 < MA_TB ? t - j0 : MA_TB;
            #pragma omp for schedule(static) nowait
            for (int r4 = 0; r4 < rows / 4; ++r4) {
                kq_rows4(w + (size_t)4 * r4 * rb, rb, type, cols, xq + (size_t)j0 * cols,
                         xs + (size_t)j0 * (cols / 32), xm + (size_t)j0 * (cols / 16), NULL, nb,
                         out + (size_t)j0 * rows + 4 * r4, (size_t)rows);
            }
        }
        #pragma omp barrier
        return;
    }
#endif
    if (rows < 64 && t > 1) {
        /* few rows (the small float32 matrices): tasks of a row and 16
         * tokens, so the tokens take the threads; the sums of each token
         * are those of kq_row */
        int nj = (t + 15) / 16;
        #pragma omp for schedule(static)
        for (int e = 0; e < rows * nj; ++e) {
            int r = e / nj, j0 = 16 * (e % nj), n = t - j0 < 16 ? t - j0 : 16;
            kq_row(w + (size_t)r * rb, type, cols, xq + (size_t)j0 * cols,
                   xs + (size_t)j0 * (cols / 32), xm + (size_t)j0 * (cols / 16),
                   x ? x + (size_t)j0 * cols : NULL, n, out + (size_t)j0 * rows + r, (size_t)rows);
        }
        return;
    }
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; ++r) {
        kq_row(w + (size_t)r * rb, type, cols, xq, xs, xm, x, t, out + r, (size_t)rows);
    }
}

void kq_linear(const uint8_t *w, int type, int rows, int cols, const int8_t *xq, const float *xs,
               const float *xm, const float *x, int t, float *out)
{
    #pragma omp parallel
    kq_linear_body(w, type, rows, cols, xq, xs, xm, x, t, out);
}

/* ---------- the conversions of the safetensors loader (np_gemma/st_qwen4.py) ---------- */

/* One row of cols float values to Q8_0 blocks (as quantize_row_q8_0 of ggml). */
static void kq_row_to_q8_0(const float *x, int cols, uint8_t *dst)
{
    for (int b = 0; b < cols / 32; ++b) {
        const float *v = x + 32 * b;
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) {
            amax = fabsf(v[i]) > amax ? fabsf(v[i]) : amax;
        }
        float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        uint8_t *blk = dst + (size_t)b * 34;
        uint16_t hd = _cvtss_sh(d, 0);
        memcpy(blk, &hd, 2);
        for (int i = 0; i < 32; ++i) {
            blk[2 + i] = (uint8_t)(int8_t)lrintf(v[i] * id);
        }
    }
}

/* rows x cols values (bf16 if bf16, else float32) to Q8_0 rows. */
void kq_to_q8_0(const void *src, int bf16, int64_t rows, int cols, uint8_t *dst)
{
    #pragma omp parallel
    {
        float *tmp = (float *)malloc((size_t)cols * 4);
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            const float *x;
            if (bf16) {
                const uint16_t *h = (const uint16_t *)src + (size_t)r * cols;
                for (int c = 0; c < cols; ++c) {
                    tmp[c] = kq_bf16(h[c]);
                }
                x = tmp;
            } else {
                x = (const float *)src + (size_t)r * cols;
            }
            kq_row_to_q8_0(x, cols, dst + (size_t)r * kq_row_bytes(KQ_Q8_0, cols));
        }
        free(tmp);
    }
}

/* make_qx_quants of ggml (rmse_type 1: the weights x^2) for n values: the
 * codes + nmax in L, and the scale. Rounding to the nearest even (lrintf) as
 * nearest_int of ggml. */
static float kq_qx_quants(int n, int nmax, const float *x, uint8_t *L)
{
    float max = 0.f, amax = 0.f;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < 1e-15f) {
        memset(L, 0, (size_t)n);
        return 0.f;
    }
    float iscale = -nmax / max, sumlx = 0.f, suml2 = 0.f;
    for (int i = 0; i < n; ++i) {
        int l = (int)lrintf(iscale * x[i]);
        l = l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l);
        L[i] = (uint8_t)(l + nmax);
        float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 > 0.f ? sumlx / suml2 : 0.f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0.f;
        for (int i = 0; i < n; ++i) {
            int l = (int)lrintf(iscale * x[i]);
            l = l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l);
            float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0.f && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = (int)lrintf(iscale * x[i]);
                l = l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l);
                L[i] = (uint8_t)(l + nmax);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

/* One Q6_K block of 256 values (quantize_row_q6_K_ref of ggml): 16 scales of
 * make_qx_quants, as int8 of a float16 d, then the 6-bit codes. */
static void kq_block_to_q6_k(const float *x, uint8_t *y)
{
    uint8_t L[256];
    float scales[16], max_scale = 0.f, max_abs = 0.f;
    for (int ib = 0; ib < 16; ++ib) {
        float sc = kq_qx_quants(16, 32, x + 16 * ib, L + 16 * ib);
        scales[ib] = sc;
        if (fabsf(sc) > max_abs) {
            max_abs = fabsf(sc);
            max_scale = sc;
        }
    }
    if (max_abs < 1e-15f) {
        memset(y, 0, 210);
        return;
    }
    float iscale = -128.f / max_scale;
    uint16_t dh = kq_f32_to_f16(1.f / iscale);
    float d = kq_h((const uint8_t *)&dh);
    int8_t *sc = (int8_t *)(y + 192);
    for (int ib = 0; ib < 16; ++ib) {
        int v = (int)lrintf(iscale * scales[ib]);
        sc[ib] = (int8_t)(v > 127 ? 127 : v);
    }
    for (int j = 0; j < 256; ++j) {
        float dd = d * sc[j / 16];
        if (dd == 0.f) {
            continue;
        }
        int l = (int)lrintf(x[j] / dd);
        l = l < -32 ? -32 : (l > 31 ? 31 : l);
        L[j] = (uint8_t)(l + 32);
    }
    uint8_t *ql = y, *qh = y + 128;
    for (int h = 0; h < 2; ++h) {
        const uint8_t *Lh = L + 128 * h;
        for (int l = 0; l < 32; ++l) {
            int q1 = Lh[l] & 15, q2 = Lh[l + 32] & 15, q3 = Lh[l + 64] & 15, q4 = Lh[l + 96] & 15;
            ql[64 * h + l] = (uint8_t)(q1 | (q3 << 4));
            ql[64 * h + l + 32] = (uint8_t)(q2 | (q4 << 4));
            qh[32 * h + l] = (uint8_t)((Lh[l] >> 4) | ((Lh[l + 32] >> 4) << 2) |
                                       ((Lh[l + 64] >> 4) << 4) | ((Lh[l + 96] >> 4) << 6));
        }
    }
    memcpy(y + 208, &dh, 2);
}

/* Q6_K rows (cols a multiple of 256) of a matrix: bfloat16 (bf16) or
 * float32. */
void kq_to_q6_k(const void *src, int bf16, int64_t rows, int cols, uint8_t *dst)
{
    #pragma omp parallel
    {
        float *tmp = (float *)malloc((size_t)cols * 4);
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            const float *x;
            if (bf16) {
                const uint16_t *h = (const uint16_t *)src + (size_t)r * cols;
                for (int c = 0; c < cols; ++c) {
                    tmp[c] = kq_bf16(h[c]);
                }
                x = tmp;
            } else {
                x = (const float *)src + (size_t)r * cols;
            }
            uint8_t *o = dst + (size_t)r * kq_row_bytes(KQ_Q6_K, cols);
            for (int b = 0; b < cols / 256; ++b) {
                kq_block_to_q6_k(x + 256 * b, o + 210 * b);
            }
        }
        free(tmp);
    }
}

/* The NVFP4 matrices of n experts (ModelOpt: w, rows x cols / 2 bytes, value
 * 2i in the low 4 bits of byte i; s, rows x cols / 16 E4M3 scales; g, the
 * float32 scale) to KQ_NV4 rows: dst gets n matrices of rows rows. wp and sp
 * hold the addresses of w and s of each expert. */
void kq_nv4_pack(const int64_t *wp, const int64_t *sp, const float *g, int n, int rows, int cols,
                 uint8_t *dst)
{
    size_t rb = kq_row_bytes(KQ_NV4, cols);
    #pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < (int64_t)n * rows; ++x) {
        int e = (int)(x / rows), r = (int)(x % rows);
        const uint8_t *w = (const uint8_t *)(intptr_t)wp[e] + (size_t)r * (cols / 2);
        const uint8_t *s = (const uint8_t *)(intptr_t)sp[e] + (size_t)r * (cols / 16);
        uint8_t *o = dst + (size_t)x * rb;
        memset(o, 0, rb);
        memcpy(o + (size_t)cols / 2, s, (size_t)cols / 16);       /* the scales, in order */
        memcpy(o + (size_t)cols / 2 + (size_t)cols / 16, &g[e], 4);
        for (int b = 0; b < cols / 32; ++b) {
            const uint8_t *src = w + (size_t)b * 16;     /* 32 values, 2 in each byte */
            uint8_t *q = o + (size_t)16 * b;
            for (int j = 0; j < 16; ++j) {
                int lo = (src[j / 2] >> (4 * (j % 2))) & 15;             /* value j */
                int hi = (src[8 + j / 2] >> (4 * (j % 2))) & 15;         /* value j + 16 */
                q[j] = (uint8_t)(lo | (hi << 4));
            }
        }
    }
}

/* The NVFP4 matrices of n experts (as kq_nv4_pack; rows a multiple of 16)
 * to KQ_NVX groups: dst gets n matrices of rows / 16 groups. */
void kq_nvx_pack(const int64_t *wp, const int64_t *sp, const float *g, int n, int rows, int cols,
                 uint8_t *dst)
{
    size_t gb = 16 * kq_row_bytes(KQ_NVX, cols);
    int ng = rows / 16;
    #pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < (int64_t)n * ng; ++x) {
        int e = (int)(x / ng), g0 = 16 * (int)(x % ng);
        const uint8_t *w = (const uint8_t *)(intptr_t)wp[e];
        const uint8_t *s = (const uint8_t *)(intptr_t)sp[e];
        uint8_t *o = dst + (size_t)x * gb;
        memset(o, 0, gb);
        memcpy(o, &g[e], 4);
        for (int b = 0; b < cols / 32; ++b) {
            uint8_t *blk = o + 16 + (size_t)b * KQ_NVX_BB;
            for (int r = 0; r < 16; ++r) {
                const uint8_t *wr = w + (size_t)(g0 + r) * (cols / 2) + 16 * b;
                for (int v = 0; v < 32; ++v) {
                    int c = (wr[v / 2] >> (4 * (v % 2))) & 15;      /* value 2i: the low 4 bits */
                    blk[32 * (v / 4) + 4 * (r % 8) + v % 4] |= (uint8_t)(c << (r < 8 ? 0 : 4));
                }
                const uint8_t *sr = s + (size_t)(g0 + r) * (cols / 16) + 2 * b;
                blk[256 + r] = sr[0];
                blk[272 + r] = sr[1];
            }
        }
    }
}

/* Copy n rows of bytes bytes from the addresses addrs (rows of a memory map
 * of a file: the n-gram table of the safetensors checkpoint). Many threads
 * take the page faults at the same time, so the disk has many reads in flight
 * (a loop of one thread waits for each read). */
/* Copy n bytes from src to dst on all the threads, in chunks of 4 MB (the
 * experts from the map of a file into memory of the process: a file on a
 * DAX mount, Optane). The pages of dst go where its policy says (mbind). */
void kq_memcpy_par(uint8_t *dst, const uint8_t *src, int64_t n)
{
    const int64_t chunk = 4 << 20;
    int64_t nc = (n + chunk - 1) / chunk;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t c = 0; c < nc; ++c) {
        int64_t off = c * chunk, len = n - off < chunk ? n - off : chunk;
        memcpy(dst + off, src + off, (size_t)len);
    }
}

void kq_gather(const int64_t *addrs, int64_t n, int bytes, uint8_t *out)
{
    /* On the calling thread: a team of 64 threads for the rows of a verify
     * group (16 for each token) spun on the cores of the CPU experts of a GPU
     * step (Qwen3.8, MTP drafts 30 ms a round in place of 22), and one for
     * the 480 rows of a short prompt (0.04 ms alone) is no team of the plan
     * (gemma_team_warn). The 65536 rows of a mixed group of 4096 tokens from
     * the DAX module: 25 ms on one thread (9 with 64), 0.4% of the group. */
    for (int64_t i = 0; i < n; ++i) {
        memcpy(out + (size_t)i * bytes, (const void *)(intptr_t)addrs[i], (size_t)bytes);
    }
}

/* float32 or bfloat16 rows (bf) to KQ_F32X16 or KQ_BF16X16: dst gets
 * (rows + 15) / 16 groups; the rows past rows are zeros. */
void kq_pack_x16f(const uint8_t *src, int bf, int64_t rows, int cols, uint8_t *dst)
{
    int es = bf ? 2 : 4;
    int64_t ng = (rows + 15) / 16;
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < ng; ++g) {
        uint8_t *o = dst + (size_t)g * 16 * cols * es;
        for (int r = 0; r < 16; ++r) {
            int64_t row = 16 * g + r;
            for (int c = 0; c < cols; ++c) {
                if (row < rows) {
                    memcpy(o + ((size_t)16 * c + r) * es, src + ((size_t)row * cols + c) * es, es);
                } else {
                    memset(o + ((size_t)16 * c + r) * es, 0, es);
                }
            }
        }
    }
}

/* BF12 rows to KQ_BF12X16 (kq_x16f_col): groups of 16 rows, the last one
 * padded with rows of zeros. */
void kq_pack_bf12x16(const uint8_t *src, int64_t rows, int cols, uint8_t *dst)
{
    size_t rb = kq_row_bytes(KQ_BF12, cols), gbytes = (size_t)16 * kq_row_bytes(KQ_BF12X16, cols);
    int64_t ng = (rows + 15) / 16;
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < ng; ++g) {
        uint8_t *o = dst + (size_t)g * gbytes;
        memset(o, 0, gbytes);
        for (int r = 0; r < 16; ++r) {
            int64_t row = 16 * g + r;
            if (row >= rows) {
                continue;
            }
            const uint8_t *w = src + (size_t)row * rb;
            for (int b = 0; b < cols / 32; ++b) {
                uint8_t *blk = o + (size_t)b * 784;
                for (int cc = 0; cc < 32; ++cc) {
                    blk[16 * cc + r] = w[32 * b + cc];
                }
                for (int j = 0; j < 16; ++j) {
                    blk[512 + 16 * j + r] = w[cols + 16 * b + j];
                }
                blk[768 + r] = w[cols + cols / 2 + b];
            }
        }
    }
}

/* Q8_0 rows (rows a multiple of 16) to KQ_Q8X16 (see the formats). */
void kq_pack_q8x16(const uint8_t *q8, int64_t rows, int cols, uint8_t *dst)
{
    int nb = cols / 32;
    size_t rb = (size_t)nb * 34;
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < rows / 16; ++g) {
        for (int b = 0; b < nb; ++b) {
            uint8_t *o = dst + ((size_t)g * nb + b) * 576;
            for (int r = 0; r < 16; ++r) {
                const uint8_t *blk = q8 + (size_t)(16 * g + r) * rb + (size_t)b * 34;
                memcpy(o + 2 * r, blk, 2);
                int16_t sum = 0;
                for (int k = 0; k < 8; ++k) {
                    for (int u = 0; u < 4; ++u) {
                        o[64 + 64 * k + 4 * r + u] = blk[2 + 4 * k + u];
                        sum += (int8_t)blk[2 + 4 * k + u];
                    }
                }
                memcpy(o + 32 + 2 * r, &sum, 2);
            }
        }
    }
}

/* The float values of rows ids of a matrix (the embeddings). */
void kq_rows(const uint8_t *w, int type, int cols, const int64_t *ids, int n, float *out)
{
    size_t rb = kq_row_bytes(type, cols);
    int bv = kq_block32(type) ? 32 : 256;
    /* on the calling thread, as kq_gather: no team outside the planned ones
     * (gemma_team_warn). A team of 40 for the 64 n-gram rows of a verify
     * group waited 5.7 ms for threads that shared the cores of the CPU
     * experts (37 us alone); the embeddings of a mixed group of 4096 rows
     * take 22 ms on one thread (5 with 40), 0.4% of the group. */
    for (int i = 0; i < n; ++i) {
        const uint8_t *row = w + (size_t)ids[i] * rb;
        if (type == KQ_NVX) {
            kq_nvx_values(w + (size_t)(ids[i] & ~(int64_t)15) * rb, (int)(ids[i] & 15), cols,
                          out + (size_t)i * cols);
            continue;
        }
        if (type == KQ_Q4X) {
            const uint8_t *wg = w + (size_t)(ids[i] & ~(int64_t)15) * rb;
            int rin = (int)(ids[i] & 15);
            for (int b = 0; b < cols / 32; ++b) {
                const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
                float d = kq_h(blk + 256 + 2 * rin);
                for (int v = 0; v < 32; ++v) {
                    int c = (blk[32 * (v / 4) + 4 * (rin % 8) + v % 4] >> (rin < 8 ? 0 : 4)) & 15;
                    out[(size_t)i * cols + 32 * b + v] = d * (float)(c - 8);
                }
            }
            continue;
        }
        if (type == KQ_F32) {
            memcpy(out + (size_t)i * cols, row, (size_t)cols * 4);
            continue;
        }
        if (type == KQ_BF12) {
            for (int j = 0; j < cols; ++j) {
                uint32_t u = (uint32_t)kq_bf12_bits(row, cols, j) << 16;
                memcpy(out + (size_t)i * cols + j, &u, 4);
            }
            continue;
        }
        for (int b = 0; b < cols / bv; ++b) {
            kq_block_values(type, row, cols, b, out + (size_t)i * cols + (size_t)b * bv);
        }
    }
}

/* bfloat16 rows (uint16, rows x cols, cols a multiple of 32) to BF12 rows
 * (dst: rows x kq_row_bytes(KQ_BF12, cols)). A value more than 15 binades
 * under the largest of its group (and a zero, a denormal) gets the zero code;
 * so does -2^(E - 15) (a collision: the zero code takes its bits). rep gets
 * the report (BF12_PLAN.md step 1): [0] the values zeroed (nonzero values
 * that decode to 0), [1] the collisions, [2] the Inf and NaN values, [3] the
 * largest zeroed value in absolute value, [4] the sum of the squares of the
 * values, [5] the count of the worst kept (at most 10), then for each (by
 * size): [6 + 4 i] the value, its row, its column, the largest absolute
 * value of its group. */
static float kq_bf16f(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void kq_bf16_to_bf12(const uint16_t *src, int64_t rows, int cols, uint8_t *dst, double *rep)
{
    size_t rb = kq_row_bytes(KQ_BF12, cols);
    int ng = cols / 32;
    double nz = 0, ncol = 0, nbad = 0, big = 0, ss = 0;
    int nt = omp_get_max_threads();
    double *worst = (double *)calloc((size_t)nt * 40, sizeof(double));
    int *nw = (int *)calloc((size_t)nt, sizeof(int));
    #pragma omp parallel for schedule(static) reduction(+ : nz, ncol, nbad, ss) reduction(max : big)
    for (int64_t r = 0; r < rows; ++r) {
        int th = omp_get_thread_num();
        const uint16_t *w = src + (size_t)r * cols;
        uint8_t *o = dst + (size_t)r * rb;
        memset(o, 0, rb);
        for (int g = 0; g < ng; ++g) {
            int E = 0;
            float gmax = 0.f;
            for (int j = 0; j < 32; ++j) {
                int e = (w[32 * g + j] >> 7) & 255;
                E = e > E ? e : E;
                float a = fabsf(kq_bf16f(w[32 * g + j]));
                if (a == a && a > gmax) {
                    gmax = a;
                }
            }
            o[cols + cols / 2 + g] = (uint8_t)E;
            for (int j = 0; j < 32; ++j) {
                uint16_t h = w[32 * g + j];
                int sgn = h >> 15, e = (h >> 7) & 255, m = h & 127;
                float f = kq_bf16f(h);
                if (e == 255) {
                    nbad += 1;
                } else {
                    ss += (double)f * f;
                }
                int gap = E - e;
                if (gap > 15) {
                    /* out of range (or a zero, a denormal): the zero code */
                    if (f != 0.f) {
                        nz += 1;
                        double a = fabs((double)f);
                        big = a > big ? a : big;
                        /* the worst 10 of this thread, by size */
                        double *ws = worst + (size_t)th * 40;
                        int k = nw[th];
                        if (k < 10 || a > fabs(ws[4 * (k - 1)])) {
                            int at = k < 10 ? k++ : 9;
                            while (at > 0 && fabs(ws[4 * (at - 1)]) < a) {
                                memcpy(ws + 4 * at, ws + 4 * (at - 1), 4 * sizeof(double));
                                --at;
                            }
                            ws[4 * at] = f;
                            ws[4 * at + 1] = (double)r;
                            ws[4 * at + 2] = 32.0 * g + j;
                            ws[4 * at + 3] = gmax;
                            nw[th] = k;
                        }
                    }
                    gap = 15;
                    sgn = 1;
                    m = 0;
                } else if (gap == 15 && sgn == 1 && m == 0) {
                    ncol += 1;
                }
                o[32 * g + j] = (uint8_t)(sgn << 7 | m);
                o[cols + 16 * g + (j & 15)] |= (uint8_t)(gap << (4 * (j >> 4)));
            }
        }
    }
    rep[0] = nz;
    rep[1] = ncol;
    rep[2] = nbad;
    rep[3] = big;
    rep[4] = ss;
    /* the worst 10 of all the threads, by size (a selection over at most
     * 10 per thread) */
    int k = 0;
    for (int pass = 0; pass < 10; ++pass) {
        int bt = -1, bi = -1;
        double ba = 0.0;            /* (a zeroed value is not 0) */
        for (int th = 0; th < nt; ++th) {
            for (int i = 0; i < nw[th]; ++i) {
                double a = fabs(worst[(size_t)th * 40 + 4 * i]);
                if (a > ba) {
                    ba = a;
                    bt = th;
                    bi = i;
                }
            }
        }
        if (bt < 0) {
            break;
        }
        double *x = worst + (size_t)bt * 40 + 4 * bi;
        memcpy(rep + 6 + 4 * k, x, 4 * sizeof(double));
        x[0] = 0.0;                 /* taken (abs 0 is below every zeroed value) */
        ++k;
    }
    rep[5] = k;
    free(worst);
    free(nw);
}

/* BF12 rows to their bfloat16 bits (dst: rows x cols uint16). */
void kq_bf12_to_bf16(const uint8_t *src, int64_t rows, int cols, uint16_t *dst)
{
    size_t rb = kq_row_bytes(KQ_BF12, cols);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        kq_bf12_row_bits(src + (size_t)r * rb, cols, dst + (size_t)r * cols);
    }
}

/* The size of the scratch of kq_moe_body, in bytes. */
size_t kq_moe_scratch(int t, int k, int experts, int hidden, int inner)
{
    size_t P = (size_t)t * (k + 1);
    size_t n = 0;
    n += (size_t)(experts + 2) * 4 * 3 + 64;           /* counts, starts, used */
    n += P * 4 * 2;                                    /* pair token, pair of */
    n += P * hidden + P * (hidden / 32) * 4 + P * (hidden / 16) * 4;
    n += P * 2 * inner * 4;                            /* gate and up */
    n += P * inner + P * (inner / 32) * 4 + P * (inner / 16) * 4;
    n += P * hidden * 4;                               /* down outputs */
    n += P * (hidden / 16) * 4 + P * (inner / 16) * 4;  /* the sums of KQ_NVX */
    n += P * hidden * 2 + P * (hidden / 32) * 4;       /* int16 h (act bit 2) */
    n += P * inner * 2 + P * (inner / 32) * 4;         /* int16 GELU */
    /* kq_moe_small_body: the counters of the experts, the int8 rows of the
     * tokens */
    n += (size_t)(experts + 2) * 4 * 2 + (size_t)t * hidden + (size_t)t * (hidden / 32) * 12 + 64 * 3;
    /* act bit 5 (x16): the low int8 planes of h, of the GELU, and of the
     * rows of the tokens (kq_quant_part16) */
    n += P * hidden + P * inner + (size_t)t * hidden + 64 * 3;
    n += (size_t)(experts + 2) * 4 + 64;               /* kq_moe_body: kq_used_split */
#if !defined(__AVX512VNNI__) && defined(__AVX2__)
    /* the planes of the int16 rows (AVX2, kq_q4x_rows_p16) */
    n += P * 2 * (hidden + inner) + P * ((hidden + inner) / 32) * 4 + 64 * 6;
#endif
    return n + 64 * 20;
}

/* x16 (act bit 5 of kq_moe_body): x as int16 in each 32 values, v =
 * rint(x 16192 / max) (about -84 dB, against -45 for int8), split v = 128 hi
 * + lo with hi in [-126, 127] and lo in [-64, 63]: two int8 planes for the
 * VNNI kernels (kq_t2_tile; the integer sum of a step 128 sum(hi w) +
 * sum(lo w) is exact, so the bits do not depend on the kernel). xs gets
 * max / 16192, xm the two sums of 16 of v times xs (the int8 forms). */
static void kq_quant_part16(const float *xr, int g, int8_t *hi, int8_t *lo, float *xs, float *xm)
{
    const float *v = xr + 32 * g;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    /* The same values as the loops below (cvtps rounds to the nearest even,
     * as lrintf in the default mode; srai is the >> of the loop). The
     * scalar loops took 39 us of the 58 of the single region of a decode
     * step's CPU part (one token, 2560 values) with the clocks down. */
    {
        __m512 a = _mm512_loadu_ps(v), b = _mm512_loadu_ps(v + 16);
        float mv = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a), _mm512_abs_ps(b)));
        float iv = mv > 0.f ? 16192.f / mv : 0.f;
        xs[g] = mv / 16192.f;
        const __m512i qlo = _mm512_set1_epi32(-16192), qhi = _mm512_set1_epi32(16192);
        __m512i qa = _mm512_min_epi32(qhi, _mm512_max_epi32(qlo, _mm512_cvtps_epi32(
            _mm512_mul_ps(a, _mm512_set1_ps(iv)))));
        __m512i qb = _mm512_min_epi32(qhi, _mm512_max_epi32(qlo, _mm512_cvtps_epi32(
            _mm512_mul_ps(b, _mm512_set1_ps(iv)))));
        const __m512i c64 = _mm512_set1_epi32(64);
        __m512i ha = _mm512_srai_epi32(_mm512_add_epi32(qa, c64), 7);
        __m512i hb = _mm512_srai_epi32(_mm512_add_epi32(qb, c64), 7);
        __m512i la = _mm512_sub_epi32(qa, _mm512_slli_epi32(ha, 7));
        __m512i lb = _mm512_sub_epi32(qb, _mm512_slli_epi32(hb, 7));
        _mm_storeu_si128((__m128i *)(hi + 32 * g), _mm512_cvtepi32_epi8(ha));
        _mm_storeu_si128((__m128i *)(hi + 32 * g + 16), _mm512_cvtepi32_epi8(hb));
        _mm_storeu_si128((__m128i *)(lo + 32 * g), _mm512_cvtepi32_epi8(la));
        _mm_storeu_si128((__m128i *)(lo + 32 * g + 16), _mm512_cvtepi32_epi8(lb));
        xm[2 * g] = xs[g] * (float)_mm512_reduce_add_epi32(qa);
        xm[2 * g + 1] = xs[g] * (float)_mm512_reduce_add_epi32(qb);
        return;
    }
#endif
    float m = 0.f;
    for (int i = 0; i < 32; ++i) {
        m = fmaxf(m, fabsf(v[i]));
    }
    float iv = m > 0.f ? 16192.f / m : 0.f;
    xs[g] = m / 16192.f;
    int s0 = 0, s1 = 0;
    for (int i = 0; i < 32; ++i) {
        int q = (int)lrintf(v[i] * iv);
        q = q > 16192 ? 16192 : q < -16192 ? -16192 : q;
        int h = (q + 64) >> 7;
        hi[32 * g + i] = (int8_t)h;
        lo[32 * g + i] = (int8_t)(q - 128 * h);
        if (i < 16) {
            s0 += q;
        } else {
            s1 += q;
        }
    }
    xm[2 * g] = xs[g] * (float)s0;
    xm[2 * g + 1] = xs[g] * (float)s1;
}

static void kq_quant_row16(const float *xr, int cols, int8_t *hi, int8_t *lo, float *xs, float *xm)
{
    for (int g = 0; g < cols / 32; ++g) {
        kq_quant_part16(xr, g, hi, lo, xs, xm);
    }
}

/* The types and shapes that x16 takes (kq_t2_tile). */
static int kq_x16_ok(int type, int cols)
{
#if defined(__AVX512VNNI__)
    return (type == KQ_Q8_0 || (type == KQ_Q6_K && cols % 256 == 0)) && cols % 64 == 0 &&
           cols <= 64 * KQ_T2_NV;
#else
    (void)type;
    (void)cols;
    return 0;
#endif
}

/* rows rows (a multiple of 16) from row r of a matrix on n tokens, as
 * kq_rows4: KQ_NVX in groups of 16 (xn: kq_nvx_xsum of the tokens), the
 * other types 4 rows at a time. */
static void kq_rows_n(const uint8_t *w, size_t rb, int type, int nr, int cols, const int8_t *xq,
                      const float *xs, const float *xm, const int32_t *xn, int n, float *out,
                      size_t ostride)
{
    for (int i = 0; i < nr; i += type == KQ_NVX || type == KQ_Q4X ? 16 : 4) {
        if (type == KQ_NVX) {
            kq_nvx_rows(w + (size_t)i * rb, cols, xq, xs, xn, n, out + i, ostride);
        } else if (type == KQ_Q4X) {
            kq_q4x_rows(w + (size_t)i * rb, cols, xq, xs, xn, n, out + i, ostride);
        } else {
            kq_rows4(w + (size_t)i * rb, rb, type, cols, xq, xs, xm, NULL, n, out + i, ostride);
        }
    }
}

/* kq_rows_n, or (xl not null) the rows on x as int16 (xq the high plane, xl
 * the low one, xs and xm of kq_quant_part16; kq_x16_ok types): kq_rows4_t2
 * for any count of tokens, so the bits of a token do not depend on the
 * count. */
static void kq_rows_n2(const uint8_t *w, size_t rb, int type, int nr, int cols, const int8_t *xq,
                       const float *xs, const float *xm, const int32_t *xn, const int8_t *xl, int n,
                       float *out, size_t ostride)
{
#if defined(__AVX512VNNI__)
    if (xl != NULL) {
        if (n == 1 && type == KQ_Q8_0 && cols % 64 == 0 && kq_t1_on()) {
            for (int i = 0; i < nr; i += 4) {
                kq_rows4_q8_x16_t1(w + (size_t)i * rb, rb, cols, xq, xs, xl, out + i,
                                   kq_pf_bytes() ? w + (size_t)(i + 4) * rb : NULL);
            }
            return;
        }
        if (n == 1 && type == KQ_Q6_K && cols % 256 == 0 && kq_t1_on()) {
            for (int i = 0; i < nr; i += 4) {
                kq_rows4_q6_x16_t1(w + (size_t)i * rb, rb, cols, xq, xs, xm, xl, out + i,
                                   kq_pf_bytes() ? w + (size_t)(i + 4) * rb : NULL);
            }
            return;
        }
        for (int i = 0; i < nr; i += 4) {
            kq_rows4_t2(w + (size_t)i * rb, rb, type, cols, xq, xs, xm, n, out + i, ostride, xl);
        }
        return;
    }
#endif
    kq_rows_n(w, rb, type, nr, cols, xq, xs, xm, xn, n, out, ostride);
}

/* One matrix of the MoE: data and type. */
typedef struct {
    const uint8_t *w;
    int type;
} kq_mat;

static inline kq_mat kq_mat_of(const int64_t *d)
{
    kq_mat m;
    m.w = (const uint8_t *)(intptr_t)d[0];
    m.type = (int)d[1];
    return m;
}

/* The experts of t tokens in the GGUF formats, as ma_moe_body. h is
 * quantized (hq, hs, hm; kq_quant_body). mats has 6 descriptors (w, type):
 * gate, up, down of the stacked experts, then of the shared expert (w = 0
 * for none). The weight of the shared expert is sigmoid(shared_logit).
 * kcount (or null) holds k, for one token. */
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
extern int sched_getcpu(void);

/* The NUMA node of each CPU (/sys/devices/system/node), read once. */
static int kq_cpu_node_tab[4096];
static pthread_once_t kq_cpu_node_once = PTHREAD_ONCE_INIT;

static void kq_cpu_node_init(void)
{
    for (int node = 0; node < 64; ++node) {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
        FILE *f = fopen(path, "r");
        if (f == NULL) {
            continue;
        }
        char line[1024];
        if (fgets(line, sizeof(line), f) != NULL) {
            char *q = line;
            while (*q && *q != '\n') {
                int a = (int)strtol(q, &q, 10), b = a;
                if (*q == '-') {
                    b = (int)strtol(q + 1, &q, 10);
                }
                for (int c = a; c <= b && c < 4096; ++c) {
                    if (c >= 0) {
                        kq_cpu_node_tab[c] = node;
                    }
                }
                if (*q == ',') {
                    ++q;
                }
            }
        }
        fclose(f);
    }
}

/* The node of the CPU of the calling thread (0 without /sys). */
static int kq_my_node(void)
{
    pthread_once(&kq_cpu_node_once, kq_cpu_node_init);
    int c = sched_getcpu();
    return c >= 0 && c < 4096 ? kq_cpu_node_tab[c] : 0;
}

/* The share of the tasks of a decode step (t = 1, static) for the threads
 * of the first half of the team (bound spread: the CPUs of node 0): the
 * nodes can stream at different rates (6 memory channels on node 0 and 4 on
 * node 1 on the 2-socket Xeon). kq_calib_nodes measures it at the start. */
static float kq_node0_share = 0.5f;

/* RQ8_0 experts (gguf.RQ8_0: the Q8_0 blocks of rows rotated in each 32
 * values): the act of each pair is rotated the same way before the down
 * product (the program rotates the x of gate and up). A state of the
 * process, set by the model (Qwen4CPU). */
static int kq_moe_rot = 0;

/* NP_GEMMA_MOE_PROF=1: the times of the phases of kq_moe_small_body (the
 * cold experts of a GPU step), summed over the calls (kq_moe_prof): the
 * spread of the arrival of the threads, the single region, the end of the
 * gate and up tasks and of the down tasks of each thread (mean and max from
 * the end of the single), the wait at the last barrier, and the sum. */
static int kq_mprof_on = -1;
static double kq_mprof[16];
static long kq_mprof_n;
static double kq_mt[5][256];
static double kq_mthr[256][3];
static double kq_msb[4];
static double kq_mbusy[256][2];    /* per thread: the time in the kernels of gate/up, of all (s, summed) */           /* the single region: quant, sort, split, rows (s, summed) */     /* per thread: the sums of arrival, end of gate/up, end of down */

/* out: 0 the spread of the arrival, 1 the single region, 2 3 the end of gate
 * and up (mean, max), 4 5 the end of down (mean, max), 6 all, 7 the calls;
 * 8 .. 13 by half of the team (node 0, node 1): the arrival after the
 * first thread, the end of gate and up, the end of down (means). */
/* The times of each thread of the last call (us, from the end of the single
 * region): arrival (as from the first thread), the end of gate and up, the
 * end of down; n threads. */
/* The means per thread since the last call (us): arrival, end of gate/up,
 * end of down, for n threads; calls the count of calls (from kq_moe_prof's
 * count at that time). */
void kq_moe_prof_thread_means(double *out, int n, double calls)
{
    for (int q = 0; q < n && q < 256; ++q) {
        for (int i = 0; i < 3; ++i) {
            out[3 * q + i] = calls > 0 ? 1e6 * kq_mthr[q][i] / calls : 0.0;
            kq_mthr[q][i] = 0.0;
        }
    }
}

int kq_moe_prof_threads(double *out, int n)
{
    double in0 = 1e30;
    for (int q = 0; q < n; ++q) {
        in0 = fmin(in0, kq_mt[0][q]);
    }
    for (int q = 0; q < n; ++q) {
        out[3 * q] = 1e6 * (kq_mt[0][q] - in0);
        out[3 * q + 1] = 1e6 * (kq_mt[2][q] - kq_mt[1][0]);
        out[3 * q + 2] = 1e6 * (kq_mt[3][q] - kq_mt[1][0]);
    }
    return n;
}

/* The parts of the single region of kq_moe_small_body (us a call): the
 * quantization of x, the sort of the pairs, the split by node
 * (kq_used_split3), the rows of the pairs; then cleared. */
void kq_moe_prof_single(double *out, double calls)
{
    for (int i = 0; i < 4; ++i) {
        out[i] = calls > 0 ? 1e6 * kq_msb[i] / calls : 0.0;
        kq_msb[i] = 0.0;
    }
}

/* The time in the kernels of each thread (us a call): gate and up, all; then
 * cleared. */
void kq_moe_prof_busy(double *out, int n, double calls)
{
    for (int q = 0; q < n && q < 256; ++q) {
        out[2 * q] = calls > 0 ? 1e6 * kq_mbusy[q][0] / calls : 0.0;
        out[2 * q + 1] = calls > 0 ? 1e6 * kq_mbusy[q][1] / calls : 0.0;
        kq_mbusy[q][0] = kq_mbusy[q][1] = 0.0;
    }
}

void kq_moe_prof(double *out)
{
    for (int i = 0; i < 16; ++i) {
        out[i] = kq_mprof_n ? kq_mprof[i] / (double)kq_mprof_n : 0.0;
        kq_mprof[i] = 0.0;
    }
    out[7] = (double)kq_mprof_n;
    kq_mprof_n = 0;
}

void kq_set_moe_rot(int on)
{
    kq_moe_rot = on;
}
static int kq_calib_on;
static double kq_calib_busy[512];       /* the compute time of each thread */

void kq_set_node0_share(float s)
{
    kq_node0_share = s > 0.05f && s < 0.95f ? s : 0.5f;
}

float kq_get_node0_share(void)
{
    return kq_node0_share;
}

/* The range [lo, hi) of n tasks of the calling thread: the first half of
 * the team takes kq_node0_share of them, evenly, the second half the rest.
 * With an odd team or one thread, an even split. */
/* The experts on the nodes (a machine of two): node 0 holds the expert
 * stacks of mats, all of them (slot0 null) or those with slot0[e] >= 0 at
 * that slot (a split: the experts do not fit one node); node 1 the copy
 * mats1, all (slot1 null) or those with slot1[e] >= 0. The threads of each
 * node take only the tasks of the experts on their node: used in three
 * classes (kq_used_split3), A on node 0 only (and the shared expert), B on
 * both, C on node 1 only; node 0's threads A then B, node 1's C then B. */
/* The static split of a step: node 0's half the first n0 tasks, n0 = its
 * share clamped to [nA, nAB] (nAB = nA + nB), node 1's half the rest. */
static void kq_share_range_strict(int n, int nA, int nAB, int *lo, int *hi)
{
    int nth = omp_get_num_threads(), tid = omp_get_thread_num();
    if (nth < 2 || nth % 2 != 0) {
        *lo = (int)((int64_t)n * tid / nth);
        *hi = (int)((int64_t)n * (tid + 1) / nth);
        return;
    }
    int half = nth / 2, n0 = (int)(n * kq_node0_share + 0.5f);
    n0 = n0 < nA ? nA : n0 > nAB ? nAB : n0;
    if (tid < half) {
        *lo = (int)((int64_t)n0 * tid / half);
        *hi = (int)((int64_t)n0 * (tid + 1) / half);
    } else {
        int r = tid - half;
        *lo = n0 + (int)((int64_t)(n - n0) * r / half);
        *hi = n0 + (int)((int64_t)(n - n0) * (r + 1) / half);
    }
}

/* The next task of a dynamic loop: tasks 0 .. nA - 1 (counter ca), nA ..
 * nAB - 1 (cb), nAB .. n - 1 (cc); node 0's threads ca then cb, node 1's cc
 * then cb. -1 when there are none left for this thread. */
static inline int kq_next_task(int *ca, int nA, int *cb, int nAB, int *cc, int n, int node1)
{
    int x;
    if (!node1) {
        x = __atomic_fetch_add(ca, 1, __ATOMIC_RELAXED);
        if (x < nA) {
            return x;
        }
    } else {
        x = nAB + __atomic_fetch_add(cc, 1, __ATOMIC_RELAXED);
        if (x < n) {
            return x;
        }
    }
    x = nA + __atomic_fetch_add(cb, 1, __ATOMIC_RELAXED);
    return x < nAB ? x : -1;
}

/* The tasks of experts not on its node that a thread ran (a check of the
 * split; kq_moe_strict_violations reads and clears it). */
static long kq_strict_bad;

long kq_moe_strict_violations(void)
{
    long n = __atomic_exchange_n(&kq_strict_bad, 0, __ATOMIC_RELAXED);
    return n;
}

/* kq_next_task, INT_MAX in place of -1 (the end of a loop "x < INT_MAX"). */
static inline int kq_task_or_end(int *ca, int nA, int *cb, int nAB, int *cc, int n, int node1)
{
    int x = kq_next_task(ca, nA, cb, nAB, cc, n, node1);
    return x < 0 ? INT_MAX : x;
}

/* Stable partition of used (nu experts) in the classes A, B, C (see above;
 * has1: there is a copy on node 1); *nA, *nB their counts. tmp: nu ints. */
static void kq_used_split3(int *used, int nu, int experts, const int32_t *slot0, const int32_t *slot1,
                           int has1, int *tmp, int *nA, int *nB)
{
    int w = 0;
    for (int cls = 0; cls < 3; ++cls) {
        for (int u = 0; u < nu; ++u) {
            int e = used[u];
            int on0 = e >= experts || slot0 == NULL || slot0[e] >= 0;
            int on1 = e < experts && has1 && (slot1 == NULL || slot1[e] >= 0);
            int c = on0 && !on1 ? 0 : on0 ? 1 : 2;
            if (c == cls) {
                tmp[w++] = e;
            }
        }
        if (cls == 0) {
            *nA = w;
        } else if (cls == 1) {
            *nB = w - *nA;
        }
    }
    memcpy(used, tmp, (size_t)nu * 4);
}

static void kq_share_range(int n, int *lo, int *hi)
{
    int nth = omp_get_num_threads(), tid = omp_get_thread_num();
    if (nth < 2 || nth % 2 != 0 || kq_node0_share == 0.5f) {
        *lo = (int)((int64_t)n * tid / nth);
        *hi = (int)((int64_t)n * (tid + 1) / nth);
        return;
    }
    int half = nth / 2, n0 = (int)(n * kq_node0_share + 0.5f);
    if (tid < half) {
        *lo = (int)((int64_t)n0 * tid / half);
        *hi = (int)((int64_t)n0 * (tid + 1) / half);
    } else {
        int r = tid - half;
        *lo = n0 + (int)((int64_t)(n - n0) * r / half);
        *hi = n0 + (int)((int64_t)(n - n0) * (r + 1) / half);
    }
}

/* kq_moe_body for a few tokens with float rows (act bit 3: a decode step or a
 * verify group of the cold experts of a GPU step, t <= 4). The barriers of
 * kq_moe_body cost about 35 us of a layer of 240 on the 2-socket Xeon (the
 * quantization of x, the sort, the copy of the rows of the pairs, the act,
 * each with its barrier). Here:
 *
 * - one single: the int8 rows of the t tokens from hf, the sort of the
 *   pairs, the rows of the pairs, and the counters of the used experts;
 * - the gate and up tasks with no barrier after them: the thread that ends
 *   the last task of an expert computes its act and the int8 act (the code
 *   of kq_moe_body), then marks the expert ready;
 * - the down tasks with no barrier before them: a task waits for its expert;
 * - one barrier before the sum (moe_combine).
 *
 * Each pair has the operations of kq_moe_body: the same bits. */
static void kq_rows_bf12_f(const uint8_t *w, size_t rb, int nr, int cols, const float *x, size_t xst,
                           int n, float *out, size_t ost);

static void kq_moe_small_body(const int32_t *ids, const float *val, int t, int k, int experts,
                              const int64_t *mats, const float *shared_logit, int hidden,
                              int inner, uint8_t *scratch, float *out, int gelu, const float *hf,
                              const int64_t *mats1, const int32_t *slot1, const int32_t *slot0, int x2)
{
    /* mats1 (or null): the copy of the experts on node 1: all of them (slot1
     * null), or a partial copy, expert e in slot slot1[e] of mats1 (-1: not
     * there). The threads of node 1 read it for those experts, and the
     * experts with a copy come last in used (the second half of the team
     * takes them: kq_share_range) */
    if (kq_mprof_on < 0) {
        const char *v = getenv("NP_GEMMA_MOE_PROF");
        kq_mprof_on = v != NULL && atoi(v) != 0;
    }
    const int mprof = kq_mprof_on && omp_get_num_threads() <= 256;
    const int mtid = omp_get_thread_num();
    if (mprof) {
        kq_mt[0][mtid] = omp_get_wtime();
    }
    int on1 = mats1 != NULL && kq_my_node() == 1;
    kq_mat G = kq_mat_of(mats), U = kq_mat_of(mats + 2), D = kq_mat_of(mats + 4);
    kq_mat SG = kq_mat_of(mats + 6), SU = kq_mat_of(mats + 8), SD = kq_mat_of(mats + 10);
    kq_mat G1 = G, U1 = U, D1 = D;
    if (mats1 != NULL) {
        G1 = kq_mat_of(mats1);
        U1 = kq_mat_of(mats1 + 2);
        D1 = kq_mat_of(mats1 + 4);
    }
    int shared = SG.w != NULL;
    int ne = experts + shared;
    int P = t * k + (shared ? t : 0);
    int nph = hidden / 32, npi = inner / 32;
    /* KQ_BF12 matrices read the float32 rows (hf, the act): kq_rows_bf12_f */
    const int bf12 = (G.type == KQ_BF12 || U.type == KQ_BF12 || D.type == KQ_BF12 ||
                      (shared && (SG.type == KQ_BF12 || SU.type == KQ_BF12 || SD.type == KQ_BF12))) &&
                     hf != NULL;
    uint8_t *p = scratch;
    int *cnt = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *start = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *used = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *pair_tok = (int *)ma_take(&p, (size_t)P * 4);
    int *pair_of = (int *)ma_take(&p, (size_t)P * 4);
    int *nused_p = (int *)ma_take(&p, 64);
    int8_t *xq = (int8_t *)ma_take(&p, (size_t)P * hidden);
    float *xs = (float *)ma_take(&p, (size_t)P * nph * 4);
    float *xm = (float *)ma_take(&p, (size_t)P * nph * 2 * 4);
    float *act = (float *)ma_take(&p, (size_t)P * 2 * inner * 4);
    int8_t *aq = (int8_t *)ma_take(&p, (size_t)P * inner);
    float *as = (float *)ma_take(&p, (size_t)P * npi * 4);
    float *am = (float *)ma_take(&p, (size_t)P * npi * 2 * 4);
    float *de = (float *)ma_take(&p, (size_t)P * hidden * 4);
    int32_t *xn = (int32_t *)ma_take(&p, (size_t)P * (hidden / 16) * 4);
    int32_t *an = (int32_t *)ma_take(&p, (size_t)P * (inner / 16) * 4);
    int *done = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *ready = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int8_t *tq = (int8_t *)ma_take(&p, (size_t)t * hidden);
    float *ts = (float *)ma_take(&p, (size_t)t * nph * 4);
    float *tm = (float *)ma_take(&p, (size_t)t * nph * 2 * 4);
    /* x2: the low planes of x as int16 (kq_quant_part16) of the tokens, the
     * pairs, and the act; tq, xq, aq hold the high planes */
    int8_t *tq2 = NULL, *xq2 = NULL, *aq2 = NULL;
    if (x2) {
        tq2 = (int8_t *)ma_take(&p, (size_t)t * hidden);
        xq2 = (int8_t *)ma_take(&p, (size_t)P * hidden);
        aq2 = (int8_t *)ma_take(&p, (size_t)P * inner);
    }
    int ngu = G.type == KQ_NVX || G.type == KQ_Q4X ? 16 : 4;
    int ndn = D.type == KQ_NVX || D.type == KQ_Q4X ? 16 : kq_ndn_rows();
    #pragma omp single
    {
        double sb0 = mprof ? omp_get_wtime() : 0.0;
        for (int j = 0; j < t; ++j) {
            if (x2) {
                kq_quant_row16(hf + (size_t)j * hidden, hidden, tq + (size_t)j * hidden,
                               tq2 + (size_t)j * hidden, ts + (size_t)j * nph, tm + (size_t)j * 2 * nph);
                continue;
            }
            for (int g = 0; g < nph; ++g) {
                kq_quant_part(hf + (size_t)j * hidden, g, tq + (size_t)j * hidden,
                              ts + (size_t)j * nph, tm + (size_t)j * 2 * nph);
            }
        }
        double sb1 = mprof ? omp_get_wtime() : 0.0;
        /* moe_sort_pairs */
        for (int e = 0; e <= ne; ++e) {
            cnt[e] = 0;
        }
        for (int q = 0; q < t * k; ++q) {
            if (ids[q] >= 0) {
                cnt[ids[q]]++;
            }
        }
        if (shared) {
            cnt[experts] = t;
        }
        int a = 0, nu = 0;
        for (int e = 0; e < ne; ++e) {
            start[e] = a;
            a += cnt[e];
            if (cnt[e] > 0) {
                used[nu++] = e;
            }
            cnt[e] = 0;
        }
        start[ne] = a;
        /* the experts with a copy on node 1 last (kq_used_split); the
         * counters of the dynamic loops (kq_next_task) */
        double sb2 = mprof ? omp_get_wtime() : 0.0;
        if ((mats1 != NULL && slot1 != NULL) || slot0 != NULL) {
            kq_used_split3(used, nu, experts, slot0, slot1, mats1 != NULL, pair_tok, &nused_p[1],
                           &nused_p[6]);
        } else {
            nused_p[1] = nu;
            nused_p[6] = 0;
        }
        nused_p[2] = nused_p[3] = nused_p[4] = nused_p[5] = nused_p[7] = nused_p[8] = 0;
        for (int q = 0; q < t * k; ++q) {
            int e = ids[q];
            if (e < 0) {
                pair_of[q] = -1;
                continue;
            }
            int pos = start[e] + cnt[e]++;
            pair_tok[pos] = q / k;
            pair_of[q] = pos;
        }
        if (shared) {
            for (int j = 0; j < t; ++j) {
                int pos = start[experts] + cnt[experts]++;
                pair_tok[pos] = j;
                pair_of[t * k + j] = pos;
            }
        }
        *nused_p = nu;
        double sb3 = mprof ? omp_get_wtime() : 0.0;
        /* the rows of the pairs, and the counters of the used experts */
        for (int q = 0; q < start[ne]; ++q) {
            int j = pair_tok[q];
            if (bf12) {
                /* the float32 rows for the KQ_BF12 matrices (in de, which the
                 * down matrix of the pair writes after its gate and up) */
                memcpy(de + (size_t)q * hidden, hf + (size_t)j * hidden, (size_t)hidden * 4);
            }
            memcpy(xq + (size_t)q * hidden, tq + (size_t)j * hidden, (size_t)hidden);
            memcpy(xs + (size_t)q * nph, ts + (size_t)j * nph, (size_t)nph * 4);
            memcpy(xm + (size_t)q * 2 * nph, tm + (size_t)j * 2 * nph, (size_t)nph * 8);
            if (x2) {
                memcpy(xq2 + (size_t)q * hidden, tq2 + (size_t)j * hidden, (size_t)hidden);
            }
            if (ngu == 16) {
                kq_x16_xsum(G.type, xq + (size_t)q * hidden, hidden, xn + (size_t)q * (hidden / 16));
            }
        }
        for (int u = 0; u < nu; ++u) {
            done[used[u]] = 0;
            ready[used[u]] = 0;
        }
        if (mprof) {
            double sb4 = omp_get_wtime();
            kq_mprof[14] += sb4 - sb0;      /* the body of the single region */
            kq_mprof[15] += nu;             /* the experts of the call (the shared one too) */
            kq_msb[0] += sb1 - sb0;
            kq_msb[1] += sb2 - sb1;
            kq_msb[2] += sb3 - sb2;
            kq_msb[3] += sb4 - sb3;
        }
    }
    int nu = *nused_p;
    P = start[ne];
    if (mprof) {
        kq_mt[1][mtid] = omp_get_wtime();
    }
    static int dyn = -1;
    if (dyn < 0) {
        const char *v = getenv("NP_GEMMA_MOE_DYN");
        dyn = v ? atoi(v) : 0;
    }
    const int per_gu = 2 * inner / ngu, per_dn = hidden / ndn;
    /* one token: a static split with the share of each node (kq_share_range);
     * a group: dynamic (its experts have different counts of tokens). A
     * partial copy on node 1 (strict): the threads of node 1 take only the
     * experts of the copy (the last nA.. of used): kq_share_range_strict,
     * or a slot for each thread that takes its tasks with kq_next_task. */
    const int st1 = t == 1 && !dyn;
    const int strict = (mats1 != NULL && slot1 != NULL) || slot0 != NULL;
    const int nA = nused_p[1], nAB = nused_p[1] + nused_p[6];
    const int qd = strict && !st1;      /* the dynamic loops by kq_next_task */
    omp_set_schedule(qd ? omp_sched_static : t > 1 || dyn ? omp_sched_dynamic : omp_sched_static,
                     qd ? 1 : t > 1 ? 8 : (dyn ? dyn : 0));
    int glo = 0, ghi = 0, dlo = 0, dhi = 0;
    if (st1 && strict) {
        kq_share_range_strict(nu * per_gu, nA * per_gu, nAB * per_gu, &glo, &ghi);
        kq_share_range_strict(nu * per_dn, nA * per_dn, nAB * per_dn, &dlo, &dhi);
    } else if (st1) {
        kq_share_range(nu * per_gu, &glo, &ghi);
        kq_share_range(nu * per_dn, &dlo, &dhi);
    }
    double tbusy = 0.0;
    int pn = 0;     /* this thread's tasks of the expert of the run, not yet in done */
    #pragma omp for schedule(runtime) nowait
    for (int x0 = 0; x0 < (st1 || qd ? omp_get_num_threads() : nu * per_gu); ++x0) {
      int xa = st1 ? glo : x0, xb = st1 ? ghi : x0 + 1;
      if (qd) {
          xa = kq_task_or_end(&nused_p[2], nA * per_gu, &nused_p[3], nAB * per_gu, &nused_p[7], nu * per_gu, on1);
          xb = INT_MAX;
      }
      for (int x = xa; x < xb;
           x = qd ? kq_task_or_end(&nused_p[2], nA * per_gu, &nused_p[3], nAB * per_gu, &nused_p[7], nu * per_gu, on1) : x + 1) {
        double tc = (kq_calib_on || mprof) ? omp_get_wtime() : 0.0;
        int e = used[x / per_gu], rr = (x % per_gu) * ngu, isup = rr >= inner;
        int r = rr % inner;
        int c1 = on1 && e < experts && (slot1 == NULL || slot1[e] >= 0);
        if ((on1 && slot1 != NULL && !c1) || (!on1 && slot0 != NULL && e < experts && slot0[e] < 0)) {
            __atomic_add_fetch(&kq_strict_bad, 1, __ATOMIC_RELAXED);
        }
        /* the index in the copy read: node 1's, else node 0's (slot0) */
        int ex = c1 ? (slot1 != NULL ? slot1[e] : e) : (slot0 != NULL && e < experts ? slot0[e] : e);
        kq_mat m = e < experts ? (isup ? (c1 ? U1 : U) : (c1 ? G1 : G)) : (isup ? SU : SG);
        size_t rb, estride = inner;
        if (e < experts && U.w == NULL) {
            m = c1 ? G1 : G;
            r = rr;
            estride = 2 * (size_t)inner;
        }
        rb = kq_row_bytes(m.type, hidden);
        const uint8_t *w = m.w + (e < experts ? (size_t)ex * estride * rb : 0);
        int s0 = start[e];
        int n = start[e + 1] - s0;
        if (m.type == KQ_BF12) {
            kq_rows_bf12_f(w + (size_t)r * rb, rb, ngu, hidden, de + (size_t)s0 * hidden,
                           (size_t)hidden, n, act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
        } else {
            kq_rows_n2(w + (size_t)r * rb, rb, m.type, ngu, hidden, xq + (size_t)s0 * hidden,
                       xs + (size_t)s0 * nph, xm + (size_t)s0 * 2 * nph, xn + (size_t)s0 * (hidden / 16),
                       x2 ? xq2 + (size_t)s0 * hidden : NULL, n,
                       act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
        }
        /* done[e]: one add for a run of tasks of expert e (a static range:
         * the next task of the range on the same expert waits), not one for
         * each task from 24 threads on two sockets (the same speed of a
         * decode step on the 2-socket Xeon: the kernels take the time) */
        int add = 1;
        if (!qd) {
            ++pn;
            if (x + 1 < xb && used[(x + 1) / per_gu] == e) {
                if (kq_calib_on || mprof) {
                    tbusy += omp_get_wtime() - tc;
                }
                continue;
            }
            add = pn;
            pn = 0;
        }
        if (__atomic_add_fetch(&done[e], add, __ATOMIC_ACQ_REL) == per_gu) {
            /* the last task of expert e: its act and int8 act */
            for (int q = s0; q < s0 + n; ++q) {
                float *a = act + (size_t)q * 2 * inner;
                if (gelu) {
                    for (int i = 0; i < inner; ++i) {
                        float v = a[i];
                        a[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) *
                               a[inner + i];
                    }
                } else {
                    for (int i = 0; i < inner; ++i) {
                        a[i] = ma_silu(a[i]) * a[inner + i];
                    }
                }
                if (kq_moe_rot) {
                    for (int g = 0; g < inner / 32; ++g) {
                        tq6_rot32(a + 32 * g, 0);
                    }
                }
                if (x2) {
                    kq_quant_row16(a, inner, aq + (size_t)q * inner, aq2 + (size_t)q * inner,
                                   as + (size_t)q * npi, am + (size_t)q * 2 * npi);
                } else {
                    for (int g = 0; g < npi; ++g) {
                        kq_quant_part(a, g, aq + (size_t)q * inner, as + (size_t)q * npi,
                                      am + (size_t)q * 2 * npi);
                    }
                }
                if (ndn == 16) {
                    kq_x16_xsum(D.type, aq + (size_t)q * inner, inner, an + (size_t)q * (inner / 16));
                }
            }
            __atomic_store_n(&ready[e], 1, __ATOMIC_RELEASE);
        }
        if (kq_calib_on || mprof) {
            tbusy += omp_get_wtime() - tc;
        }
      }
    }
    if (mprof) {
        kq_mt[2][mtid] = omp_get_wtime();
        if (mtid < 256) {
            kq_mbusy[mtid][0] += tbusy;     /* the kernels of gate and up */
        }
    }
    #pragma omp for schedule(runtime) nowait
    for (int x0 = 0; x0 < (st1 || qd ? omp_get_num_threads() : nu * per_dn); ++x0) {
      int xa = st1 ? dlo : x0, xb = st1 ? dhi : x0 + 1;
      if (qd) {
          xa = kq_task_or_end(&nused_p[4], nA * per_dn, &nused_p[5], nAB * per_dn, &nused_p[8], nu * per_dn, on1);
          xb = INT_MAX;
      }
      for (int x = xa; x < xb;
           x = qd ? kq_task_or_end(&nused_p[4], nA * per_dn, &nused_p[5], nAB * per_dn, &nused_p[8], nu * per_dn, on1) : x + 1) {
        int e = used[x / per_dn], r = (x % per_dn) * ndn;
        while (!__atomic_load_n(&ready[e], __ATOMIC_ACQUIRE)) {
#if GEMMA_X86
            _mm_pause();
#endif
        }
        double tc = (kq_calib_on || mprof) ? omp_get_wtime() : 0.0;
        int c1 = on1 && e < experts && (slot1 == NULL || slot1[e] >= 0);
        if ((on1 && slot1 != NULL && !c1) || (!on1 && slot0 != NULL && e < experts && slot0[e] < 0)) {
            __atomic_add_fetch(&kq_strict_bad, 1, __ATOMIC_RELAXED);
        }
        /* the index in the copy read: node 1's, else node 0's (slot0) */
        int ex = c1 ? (slot1 != NULL ? slot1[e] : e) : (slot0 != NULL && e < experts ? slot0[e] : e);
        kq_mat m = e < experts ? (c1 ? D1 : D) : SD;
        size_t rb = kq_row_bytes(m.type, inner);
        const uint8_t *w = m.w + (e < experts ? (size_t)ex * hidden * rb : 0);
        int s0 = start[e];
        int n = start[e + 1] - s0;
        if (m.type == KQ_BF12) {
            kq_rows_bf12_f(w + (size_t)r * rb, rb, ndn, inner, act + (size_t)s0 * 2 * inner,
                           (size_t)2 * inner, n, de + (size_t)s0 * hidden + r, (size_t)hidden);
        } else {
            kq_rows_n2(w + (size_t)r * rb, rb, m.type, ndn, inner, aq + (size_t)s0 * inner,
                       as + (size_t)s0 * npi, am + (size_t)s0 * 2 * npi, an + (size_t)s0 * (inner / 16),
                       x2 ? aq2 + (size_t)s0 * inner : NULL, n,
                       de + (size_t)s0 * hidden + r, (size_t)hidden);
        }
        if (kq_calib_on || mprof) {
            tbusy += omp_get_wtime() - tc;
        }
      }
    }
    if (kq_calib_on && omp_get_thread_num() < 512) {
        kq_calib_busy[omp_get_thread_num()] += tbusy;
    }
    if (mprof && mtid < 256) {
        kq_mbusy[mtid][1] += tbusy;         /* the kernels of gate, up and down */
    }
    if (mprof) {
        kq_mt[3][mtid] = omp_get_wtime();
    }
    #pragma omp barrier
    if (mprof) {
        kq_mt[4][mtid] = omp_get_wtime();
    }
    moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
    if (mprof) {
        double t5 = omp_get_wtime();
        #pragma omp barrier
        #pragma omp master
        {
            int n = omp_get_num_threads();
            double in0 = 1e30, in1 = 0, gm = 0, gx = 0, dm = 0, dx = 0, bw = 0;
            for (int q = 0; q < n; ++q) {
                in0 = fmin(in0, kq_mt[0][q]);
                in1 = fmax(in1, kq_mt[0][q]);
            }
            double s1 = kq_mt[1][0];
            for (int q = 0; q < n; ++q) {
                double g = kq_mt[2][q] - s1, d = kq_mt[3][q] - s1;
                gm += g / n;
                gx = fmax(gx, g);
                dm += d / n;
                dx = fmax(dx, d);
                bw += (kq_mt[4][q] - kq_mt[3][q]) / n;
            }
            kq_mprof[0] += in1 - in0;
            kq_mprof[1] += s1 - in1;
            kq_mprof[2] += gm;
            kq_mprof[3] += gx;
            kq_mprof[4] += dm;
            kq_mprof[5] += dx;
            kq_mprof[6] += t5 - in0;
            for (int q = 0; q < n; ++q) {
                kq_mthr[q][0] += kq_mt[0][q] - in0;
                kq_mthr[q][1] += kq_mt[2][q] - s1;
                kq_mthr[q][2] += kq_mt[3][q] - s1;
            }
            for (int hf = 0; hf < 2; ++hf) {
                int q0 = hf * (n / 2), q1 = hf ? n : n / 2;
                double ar = 0, gu = 0, dn = 0;
                for (int q = q0; q < q1; ++q) {
                    ar += (kq_mt[0][q] - in0) / (q1 - q0);
                    gu += (kq_mt[2][q] - s1) / (q1 - q0);
                    dn += (kq_mt[3][q] - s1) / (q1 - q0);
                }
                kq_mprof[8 + 3 * hf] += ar;
                kq_mprof[9 + 3 * hf] += gu;
                kq_mprof[10 + 3 * hf] += dn;
            }
            (void)bw;
            ++kq_mprof_n;
        }
    }
}

/* Measure the rate of the threads of each half of a team of nth threads
 * bound spread (node 0, node 1) on the cold experts of a decode step: reps
 * calls of kq_moe_small_body on one token with ncold random experts of the
 * layers of mats (nl tables of 12 int64; mats1: the copies of node 1, or
 * null), with the current share. out[0], out[1]: the mean compute time of a
 * thread of each half (s). */
void kq_calib_nodes(const int64_t *mats, const int64_t *mats1, int nl, int experts, int hidden,
                    int inner, int ncold, int reps, int nth, double *out)
{
    size_t sb = kq_moe_scratch(1, ncold, experts, hidden, inner);
    uint8_t *scratch = (uint8_t *)aligned_alloc(64, (sb + 63) / 64 * 64);
    float *hf = (float *)malloc((size_t)hidden * 4), *o = (float *)malloc((size_t)hidden * 4);
    int32_t *ids = (int32_t *)malloc((size_t)ncold * 4);
    float *val = (float *)malloc((size_t)ncold * 4);
    uint32_t seed = 12345u;
    for (int i = 0; i < hidden; ++i) {
        seed = seed * 1664525u + 1013904223u;
        hf[i] = ((float)(seed >> 8) / 16777216.0f - 0.5f) * 0.2f;
    }
    for (int i = 0; i < 512; ++i) {
        kq_calib_busy[i] = 0.0;
    }
    kq_calib_on = 1;
    for (int rp = 0; rp < reps; ++rp) {
        for (int c = 0; c < ncold; ++c) {
            int e, dup;
            do {
                seed = seed * 1664525u + 1013904223u;
                e = (int)((seed >> 8) % (uint32_t)experts);
                dup = 0;
                for (int d = 0; d < c; ++d) {
                    dup |= ids[d] == e;
                }
            } while (dup);
            ids[c] = e;
            val[c] = 1.0f / ncold;
        }
        const int64_t *m = mats + (size_t)(rp % nl) * 12;
        const int64_t *m1 = mats1 != NULL ? mats1 + (size_t)(rp % nl) * 12 : NULL;
        #pragma omp parallel num_threads(nth) proc_bind(spread)
        kq_moe_small_body(ids, val, 1, ncold, experts, m, NULL, hidden, inner, scratch, o, 0, hf, m1,
                          NULL, NULL, 0);
    }
    kq_calib_on = 0;
    double b0 = 0.0, b1 = 0.0;
    int half = nth / 2;
    for (int i = 0; i < nth && i < 512; ++i) {
        if (i < half) {
            b0 += kq_calib_busy[i];
        } else {
            b1 += kq_calib_busy[i];
        }
    }
    out[0] = half > 0 ? b0 / half / reps : 0.0;
    out[1] = nth - half > 0 ? b1 / (nth - half) / reps : 0.0;
    free(scratch);
    free(hf);
    free(o);
    free(ids);
    free(val);
}

/* mats: the up matrix with no data: the gate matrix holds 2 inner rows for
 * each expert (its gate rows, then its up rows).
 * act: bit 0 the tanh GELU of the gate (the Gemma 4 26B) in place of SiLU;
 * bit 1 float32 activations (hf: the float32 rows of h; KQ_Q4X matrices
 * only): no quantization, as the decode of the 26B; bit 2 int16 activations
 * (hf; KQ_Q4X; the prompt with NP_GEMMA_INT4_Q8=16): the rows of h and of
 * the GELU become int16 (kq_q4x_rows16); bit 3 (hf: the float rows of the
 * tokens, t <= 4, no int16): kq_moe_small_body, with few barriers. */
/* NP_GEMMA_MOE_GPROF=1: the phases of kq_moe_body (a prompt group), summed
 * over the calls (kq_moe_gprof, us from the first thread to arrive): 0 the
 * spread of the arrival, 1 the end of the sort, 2 of the copy of the rows,
 * 3 4 the last gate and up task of each thread (mean, max), 5 their
 * barrier, 6 the end of the act, 7 8 the last down task (mean, max), 9 its
 * barrier, 10 the end of the sum; 11 the calls. */
static int kq_gprof_on = -1;
static double kq_gt[256][9];
static double kq_gprof_s[12];

void kq_moe_gprof(double *out)
{
    double n = kq_gprof_s[11] > 0 ? kq_gprof_s[11] : 1;
    for (int i = 0; i < 11; ++i) {
        out[i] = 1e6 * kq_gprof_s[i] / n;
        kq_gprof_s[i] = 0;
    }
    out[11] = kq_gprof_s[11];
    kq_gprof_s[11] = 0;
}

/* n values (a multiple of 32) rounded to int16 in each 32: q = rint(x / s),
 * s = max |x| / 32767, then q s (the test of int16 activations). */
static void kq_round_i16(float *x, int n)
{
    for (int g = 0; g < n; g += 32) {
        float m = 0.f;
        for (int i = 0; i < 32; ++i) {
            m = fmaxf(m, fabsf(x[g + i]));
        }
        float s = m / 32767.f;
        for (int i = 0; i < 32 && s > 0.f; ++i) {
            x[g + i] = (float)lrintf(x[g + i] / s) * s;
        }
    }
}

/* The exact path of kq_moe_body for a type other than KQ_Q4X (a test): nr
 * rows of w (cols values; kq_rows to float) times the n float rows of x
 * (stride xst) into out[j * ost + i]. */
static void kq_rows_f_any(const uint8_t *w, int type, int nr, int cols, const float *x, size_t xst,
                          int n, float *out, size_t ost)
{
    float wf[16 * cols];
    int64_t ids[16];
    for (int i = 0; i < nr; ++i) {
        ids[i] = i;
    }
    kq_rows(w, type, cols, ids, nr, wf);
    for (int j = 0; j < n; ++j) {
        const float *xr = x + (size_t)j * xst;
        for (int i = 0; i < nr; ++i) {
            const float *wr = wf + (size_t)i * cols;
            float acc = 0.f;
            for (int c = 0; c < cols; ++c) {
                acc = fmaf(wr[c], xr[c], acc);
            }
            out[(size_t)j * ost + i] = acc;
        }
    }
}

/* nr rows of KQ_BF12 (rb bytes each) on n float32 rows of x (xst apart):
 * out[j * ost + i]. The experts in BF12 (an overlay of the MTP layer from the
 * bfloat16 of the original: the values of bfloat16, but the zeroed ones)
 * read the float32 rows, not int8 or int16 ones. */
static void kq_rows_bf12_f(const uint8_t *w, size_t rb, int nr, int cols, const float *x, size_t xst,
                           int n, float *out, size_t ost)
{
    for (int j = 0; j < n; ++j) {
        const float *xr = x + (size_t)j * xst;
        for (int i = 0; i < nr; ++i) {
            out[(size_t)j * ost + i] = kq_dot1(w + (size_t)i * rb, KQ_BF12, cols, NULL, NULL, NULL, xr);
        }
    }
}

static int kq_mats_bf12(const int64_t *mats)
{
    for (int i = 1; i < 12; i += 2) {
        if (mats[i - 1] != 0 && (int)mats[i] == KQ_BF12) {
            return 1;
        }
    }
    return 0;
}

static void kq_moe_body(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                        const float *val, int t, int k, int experts, const int64_t *mats,
                        const float *shared_logit, int hidden, int inner, uint8_t *scratch,
                        float *out, const int32_t *kcount, int act_flags, const float *hf,
                        const int64_t *mats1, const int32_t *slot1, const int32_t *slot0)
{
    int gelu = act_flags & 1, exact = (act_flags & 2) && hf != NULL;
    /* bit 4 with exact: the float rows and the GELU rounded to int16 in each
     * 32 values (the scale max / 32767), then the float products: the values
     * of int16 activations (a test of their precision against int8) */
    int emu16 = (act_flags & 16) && exact;
    /* bit 5: x and the GELU as int16 (kq_quant_part16, two int8 planes in
     * kq_t2_tile), for the types of kq_x16_ok; else int8 */
    int x2 = (act_flags & 32) && hf != NULL && !exact;
    for (int i = 1; i < 12; i += 2) {
        if (mats[i - 1] != 0 && !kq_x16_ok((int)mats[i], i % 6 == 5 ? inner : hidden)) {
            x2 = 0;
        }
    }
    /* KQ_BF12 matrices read float32 rows: x (hf) and the act */
    const int bf12 = kq_mats_bf12(mats) && hf != NULL;
    if (kq_mats_bf12(mats) && hf == NULL) {
        fprintf(stderr, "kq_moe_body: KQ_BF12 experts need the float32 rows of x (hf)\n");
        abort();
    }
#ifdef KQ_Q4X_Q16
    int q16 = (act_flags & 4) && hf != NULL;
#else
    const int q16 = 0;
#endif
    if (kcount != NULL) {
        /* One token with a count of experts in memory: the cold experts of a
         * GPU step (GP_HOT_SPLIT writes the count). */
        k = *kcount;
    }
    if ((act_flags & 8) && hf != NULL && t <= 4 && !exact && !q16) {
        /* float rows of a few tokens: the form with few barriers */
        kq_moe_small_body(ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                          out, gelu, hf, mats1, slot1, slot0, x2);
        return;
    }
    /* mats1 (or null): a copy of the experts in the memory of NUMA node 1
     * (QwenGPU, NP_GEMMA_GPU_NUMA_COPY): a thread on a CPU of node 1 reads
     * that copy for the experts it has (all, or slot1[e] >= 0), the others
     * mats, so the reads are local. The threads share the tasks as before;
     * the values are the same bytes. */
    if (kq_gprof_on < 0) {
        const char *v = getenv("NP_GEMMA_MOE_GPROF");
        kq_gprof_on = v && v[0] == '1';
    }
    const int gp = kq_gprof_on;
    double *gt = kq_gt[omp_get_thread_num() & 255];
    double glast = 0;
    if (gp) {
        gt[0] = omp_get_wtime();
    }
    int on1 = mats1 != NULL && kq_my_node() == 1;
    kq_mat G = kq_mat_of(mats), U = kq_mat_of(mats + 2), D = kq_mat_of(mats + 4);
    kq_mat SG = kq_mat_of(mats + 6), SU = kq_mat_of(mats + 8), SD = kq_mat_of(mats + 10);
    kq_mat G1 = G, U1 = U, D1 = D;
    if (mats1 != NULL) {
        G1 = kq_mat_of(mats1);
        U1 = kq_mat_of(mats1 + 2);
        D1 = kq_mat_of(mats1 + 4);
    }
    int shared = SG.w != NULL;
    int ne = experts + shared;
    int P = t * k + (shared ? t : 0);
    int nph = hidden / 32, npi = inner / 32;
    uint8_t *p = scratch;
    int *cnt = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *start = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *used = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *pair_tok = (int *)ma_take(&p, (size_t)P * 4);
    int *pair_of = (int *)ma_take(&p, (size_t)P * 4);
    int *nused_p = (int *)ma_take(&p, 64);
    int8_t *xq = (int8_t *)ma_take(&p, (size_t)P * hidden);
    float *xs = (float *)ma_take(&p, (size_t)P * nph * 4);
    float *xm = (float *)ma_take(&p, (size_t)P * nph * 2 * 4);
    float *act = (float *)ma_take(&p, (size_t)P * 2 * inner * 4);
    int8_t *aq = (int8_t *)ma_take(&p, (size_t)P * inner);
    float *as = (float *)ma_take(&p, (size_t)P * npi * 4);
    float *am = (float *)ma_take(&p, (size_t)P * npi * 2 * 4);
    float *de = (float *)ma_take(&p, (size_t)P * hidden * 4);
    int32_t *xn = (int32_t *)ma_take(&p, (size_t)P * (hidden / 16) * 4);
    int32_t *an = (int32_t *)ma_take(&p, (size_t)P * (inner / 16) * 4);
    /* the int16 rows of h and of the GELU (act bit 2) */
    int16_t *x16 = (int16_t *)ma_take(&p, (size_t)P * hidden * 2);
    float *xs16 = (float *)ma_take(&p, (size_t)P * nph * 4);
    int16_t *a16 = (int16_t *)ma_take(&p, (size_t)P * inner * 2);
    float *as16 = (float *)ma_take(&p, (size_t)P * npi * 4);
#if !defined(__AVX512VNNI__) && defined(__AVX2__) && defined(KQ_Q4X_Q16)
    /* AVX2: the planes of x16 and a16 (kq_planes16_to), for the experts of
     * one token (kq_q4x_rows_p16; the bits of kq_q4x_rows16) */
#define KQ16_PLANES 1
    int8_t *xh16 = (int8_t *)ma_take(&p, (size_t)P * hidden);
    uint8_t *xl16 = (uint8_t *)ma_take(&p, (size_t)P * hidden);
    int32_t *xn16 = (int32_t *)ma_take(&p, (size_t)P * nph * 4);
    int8_t *ah16 = (int8_t *)ma_take(&p, (size_t)P * inner);
    uint8_t *al16 = (uint8_t *)ma_take(&p, (size_t)P * inner);
    int32_t *an16 = (int32_t *)ma_take(&p, (size_t)P * npi * 4);
#endif
    /* x2 (act bit 5): the low planes of h and of the GELU as int16
     * (kq_quant_part16); xq and aq hold the high planes */
    int8_t *xq2 = NULL, *aq2 = NULL;
    if (x2) {
        xq2 = (int8_t *)ma_take(&p, (size_t)P * hidden);
        aq2 = (int8_t *)ma_take(&p, (size_t)P * inner);
    }
    int *utmp = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    /* KQ_NVX: tasks of 16 rows (a group) for all the matrices of the layer */
    int ngu = G.type == KQ_NVX || G.type == KQ_Q4X ? 16 : 4;
    int ndn = D.type == KQ_NVX || D.type == KQ_Q4X ? 16 : 4;
    moe_sort_pairs(ids, t, k, experts, shared, cnt, start, used, pair_tok, pair_of, nused_p);
    int nu = *nused_p;
    /* a partial copy on node 1 (strict): the threads of node 1 take only the
     * experts of the copy (kq_used_split: those last in used; a slot for
     * each thread, its tasks by kq_task_or_end) */
    const int strict = (mats1 != NULL && slot1 != NULL) || slot0 != NULL;
    #pragma omp single
    {
        if (strict) {
            kq_used_split3(used, nu, experts, slot0, slot1, mats1 != NULL, utmp, &nused_p[1], &nused_p[6]);
        } else {
            nused_p[1] = nu;
            nused_p[6] = 0;
        }
        nused_p[2] = nused_p[3] = nused_p[4] = nused_p[5] = nused_p[7] = nused_p[8] = 0;
    }
    P = start[ne];          /* the pairs with an expert */
    if (gp) {
        gt[1] = omp_get_wtime();
    }
    /* int16: a matrix that is not KQ_Q4X (a shared expert, another type)
     * takes int8 rows, quantized here from hf and from the GELU */
    int q8u = q16 && (G.type != KQ_Q4X || (U.w != NULL && U.type != KQ_Q4X) ||
                      (shared && (SG.type != KQ_Q4X || SU.type != KQ_Q4X)));
    int q8d = q16 && (D.type != KQ_Q4X || (shared && SD.type != KQ_Q4X));
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        int j = pair_tok[q];
        if (q16) {
            for (int g = 0; g < nph; ++g) {
                xs16[(size_t)q * nph + g] = gemma_quant_group32_i16(
                    hf + (size_t)j * hidden + 32 * g, x16 + (size_t)q * hidden + 32 * g);
            }
#ifdef KQ16_PLANES
            kq_planes16_to(x16 + (size_t)q * hidden, hidden, xh16 + (size_t)q * hidden,
                           xl16 + (size_t)q * hidden, xn16 + (size_t)q * nph);
#endif
            if (!q8u) {
                continue;
            }
            for (int g = 0; g < nph; ++g) {
                kq_quant_part(hf + (size_t)j * hidden, g, xq + (size_t)q * hidden,
                              xs + (size_t)q * nph, xm + (size_t)q * 2 * nph);
            }
            if (ngu == 16) {
                kq_x16_xsum(G.type, xq + (size_t)q * hidden, hidden, xn + (size_t)q * (hidden / 16));
            }
            continue;
        }
        if (bf12 && !exact) {
            /* the float32 rows for the KQ_BF12 matrices (kq_rows_bf12_f), in de
             * as for exact; the int8 rows below for the others */
            memcpy(de + (size_t)q * hidden, hf + (size_t)j * hidden, (size_t)hidden * 4);
        }
        if (exact) {
            /* the float32 rows: in de, which the down matrix writes later */
            memcpy(de + (size_t)q * hidden, hf + (size_t)j * hidden, (size_t)hidden * 4);
            if (emu16) {
                kq_round_i16(de + (size_t)q * hidden, hidden);
            }
            continue;
        }
        memcpy(xq + (size_t)q * hidden, hq + (size_t)j * hidden, (size_t)hidden);
        memcpy(xs + (size_t)q * nph, hs + (size_t)j * nph, (size_t)nph * 4);
        memcpy(xm + (size_t)q * 2 * nph, hm + (size_t)j * 2 * nph, (size_t)nph * 8);
        if (x2) {
            kq_quant_row16(hf + (size_t)j * hidden, hidden, xq + (size_t)q * hidden,
                           xq2 + (size_t)q * hidden, xs + (size_t)q * nph, xm + (size_t)q * 2 * nph);
        }
        if (ngu == 16) {
            kq_x16_xsum(G.type, xq + (size_t)q * hidden, hidden, xn + (size_t)q * (hidden / 16));
        }
    }
    /* A group: the experts have different counts of tokens, so the tasks
     * go to the threads as they finish. With a static split, the threads
     * were idle about 28% of the time. A decode step: the same work in each
     * task, a static split. Each thread sets the schedule of its own
     * loops. */
    if (gp) {
        gt[2] = omp_get_wtime();
    }
    omp_set_schedule(strict ? omp_sched_static : t > 1 ? omp_sched_dynamic : omp_sched_static,
                     strict ? 1 : t > 1 ? 8 : 0);
    /* Gate and up: 2 * inner rows of each used expert, in tasks of 4 rows
     * (16 for KQ_NVX). */
    const int ngt = nu * (2 * inner / ngu), nagt = nused_p[1] * (2 * inner / ngu);
    const int nabgt = (nused_p[1] + nused_p[6]) * (2 * inner / ngu);
    #pragma omp for schedule(runtime)
    for (int x0 = 0; x0 < (strict ? omp_get_num_threads() : ngt); ++x0) {
      for (int x = strict ? kq_task_or_end(&nused_p[2], nagt, &nused_p[3], nabgt, &nused_p[7], ngt, on1) : x0;
           x < (strict ? INT_MAX : x0 + 1);
           x = strict ? kq_task_or_end(&nused_p[2], nagt, &nused_p[3], nabgt, &nused_p[7], ngt, on1) : x + 1) {
        int e = used[x / (2 * inner / ngu)], rr = (x % (2 * inner / ngu)) * ngu, isup = rr >= inner;
        int r = rr % inner;
        int c1 = on1 && e < experts && (slot1 == NULL || slot1[e] >= 0);
        if ((on1 && slot1 != NULL && !c1) || (!on1 && slot0 != NULL && e < experts && slot0[e] < 0)) {
            __atomic_add_fetch(&kq_strict_bad, 1, __ATOMIC_RELAXED);
        }
        /* the index in the copy read: node 1's, else node 0's (slot0) */
        int ex = c1 ? (slot1 != NULL ? slot1[e] : e) : (slot0 != NULL && e < experts ? slot0[e] : e);
        kq_mat m = e < experts ? (isup ? (c1 ? U1 : U) : (c1 ? G1 : G)) : (isup ? SU : SG);
        size_t rb, estride = inner;
        if (e < experts && U.w == NULL) {
            /* G holds the gate rows and then the up rows of each expert (the
             * KQ_Q4X copy of a gate and up stack: one copy for the CPU and
             * the GPU) */
            m = c1 ? G1 : G;
            r = rr;
            estride = 2 * (size_t)inner;
        }
        rb = kq_row_bytes(m.type, hidden);
        const uint8_t *w = m.w + (e < experts ? (size_t)ex * estride * rb : 0);
        int s0 = start[e];
        int n = start[e + 1] - s0;
        if (exact && m.type == KQ_Q4X) {
            for (int i = 0; i < ngu; i += 16) {
                kq_q4x_rows_f(w + (size_t)(r + i) * rb, hidden, de + (size_t)s0 * hidden, (size_t)hidden,
                              n, act + (size_t)s0 * 2 * inner + rr + i, (size_t)2 * inner);
            }
            continue;
        }
        if (m.type == KQ_BF12 && (bf12 || exact)) {
            kq_rows_bf12_f(w + (size_t)r * rb, rb, ngu, hidden, de + (size_t)s0 * hidden,
                           (size_t)hidden, n, act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
            continue;
        }
        if (exact) {
            kq_rows_f_any(w + (size_t)r * rb, m.type, ngu, hidden, de + (size_t)s0 * hidden,
                          (size_t)hidden, n, act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
            continue;
        }
#ifdef KQ_Q4X_Q16
        if (q16 && m.type == KQ_Q4X) {
#ifdef KQ16_PLANES
            if (n == 1) {
                for (int i = 0; i < ngu; i += 16) {
                    kq_q4x_rows_p16(w + (size_t)(r + i) * rb, hidden, xh16 + (size_t)s0 * hidden,
                                    xl16 + (size_t)s0 * hidden, xn16 + (size_t)s0 * nph,
                                    xs16 + (size_t)s0 * nph, act + (size_t)s0 * 2 * inner + rr + i);
                }
                continue;
            }
#endif
            for (int i = 0; i < ngu; i += 16) {
                kq_q4x_rows16(w + (size_t)(r + i) * rb, hidden, x16 + (size_t)s0 * hidden,
                              xs16 + (size_t)s0 * nph, n, act + (size_t)s0 * 2 * inner + rr + i,
                              (size_t)2 * inner);
            }
            continue;
        }
#endif
        kq_rows_n2(w + (size_t)r * rb, rb, m.type, ngu, hidden, xq + (size_t)s0 * hidden,
                   xs + (size_t)s0 * nph, xm + (size_t)s0 * 2 * nph, xn + (size_t)s0 * (hidden / 16),
                   x2 ? xq2 + (size_t)s0 * hidden : NULL, n,
                   act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
        if (gp) {
            glast = omp_get_wtime();
        }
      }
    }
    if (gp) {
        gt[3] = glast;
        gt[4] = omp_get_wtime();
    }
    /* silu(gate) * up (gelu: the tanh GELU of the Gemma 4 26B), and its
     * quantization for down. */
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        float *a = act + (size_t)q * 2 * inner;
        if (gelu) {
            int i = 0;
#if GEMMA_X86 && defined(__AVX512F__)
            const __m512 cv = _mm512_set1_ps(0.7978845608028654f), half = _mm512_set1_ps(0.5f);
            const __m512 one = _mm512_set1_ps(1.0f), k3 = _mm512_set1_ps(0.044715f);
            for (; i + 16 <= inner; i += 16) {
                __m512 v = _mm512_loadu_ps(a + i);
                __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
                __m512 th = gemma_tanh_ps(_mm512_mul_ps(cv, _mm512_fmadd_ps(k3, v3, v)));
                __m512 r = _mm512_mul_ps(_mm512_mul_ps(half, v), _mm512_add_ps(one, th));
                _mm512_storeu_ps(a + i, _mm512_mul_ps(r, _mm512_loadu_ps(a + inner + i)));
            }
#else
            i = GELU_MUL_VEC(a, a + inner, a, inner);
#endif
            for (; i < inner; ++i) {
                float v = a[i];
                a[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) *
                       a[inner + i];
            }
        } else {
            for (int i = 0; i < inner; ++i) {
                a[i] = ma_silu(a[i]) * a[inner + i];
            }
        }
        if (kq_moe_rot) {
            for (int g = 0; g < inner / 32; ++g) {
                tq6_rot32(a + 32 * g, 0);
            }
        }
        if (exact) {
            if (emu16) {
                kq_round_i16(a, inner);
            }
            continue;       /* the down matrix reads a (float32) */
        }
        if (q16) {
            for (int g = 0; g < npi; ++g) {
                as16[(size_t)q * npi + g] = gemma_quant_group32_i16(
                    a + 32 * g, a16 + (size_t)q * inner + 32 * g);
            }
#ifdef KQ16_PLANES
            kq_planes16_to(a16 + (size_t)q * inner, inner, ah16 + (size_t)q * inner,
                           al16 + (size_t)q * inner, an16 + (size_t)q * npi);
#endif
            if (!q8d) {
                continue;
            }
        }
        if (x2) {
            kq_quant_row16(a, inner, aq + (size_t)q * inner, aq2 + (size_t)q * inner,
                           as + (size_t)q * npi, am + (size_t)q * 2 * npi);
        } else {
            for (int g = 0; g < npi; ++g) {
                kq_quant_part(a, g, aq + (size_t)q * inner, as + (size_t)q * npi,
                              am + (size_t)q * 2 * npi);
            }
        }
        if (ndn == 16) {
            kq_x16_xsum(D.type, aq + (size_t)q * inner, inner, an + (size_t)q * (inner / 16));
        }
    }
    if (gp) {
        gt[5] = omp_get_wtime();
    }
    /* Down: hidden rows of each used expert, in tasks of 4 rows (16 for
     * KQ_NVX). */
    const int ndt = nu * (hidden / ndn), nadt = nused_p[1] * (hidden / ndn);
    const int nabdt = (nused_p[1] + nused_p[6]) * (hidden / ndn);
    #pragma omp for schedule(runtime)
    for (int x0 = 0; x0 < (strict ? omp_get_num_threads() : ndt); ++x0) {
      for (int x = strict ? kq_task_or_end(&nused_p[4], nadt, &nused_p[5], nabdt, &nused_p[8], ndt, on1) : x0;
           x < (strict ? INT_MAX : x0 + 1);
           x = strict ? kq_task_or_end(&nused_p[4], nadt, &nused_p[5], nabdt, &nused_p[8], ndt, on1) : x + 1) {
        int e = used[x / (hidden / ndn)], r = (x % (hidden / ndn)) * ndn;
        int c1 = on1 && e < experts && (slot1 == NULL || slot1[e] >= 0);
        if ((on1 && slot1 != NULL && !c1) || (!on1 && slot0 != NULL && e < experts && slot0[e] < 0)) {
            __atomic_add_fetch(&kq_strict_bad, 1, __ATOMIC_RELAXED);
        }
        /* the index in the copy read: node 1's, else node 0's (slot0) */
        int ex = c1 ? (slot1 != NULL ? slot1[e] : e) : (slot0 != NULL && e < experts ? slot0[e] : e);
        kq_mat m = e < experts ? (c1 ? D1 : D) : SD;
        size_t rb = kq_row_bytes(m.type, inner);
        const uint8_t *w = m.w + (e < experts ? (size_t)ex * hidden * rb : 0);
        int s0 = start[e];
        int n = start[e + 1] - s0;
        if (exact && m.type == KQ_Q4X) {
            for (int i = 0; i < ndn; i += 16) {
                kq_q4x_rows_f(w + (size_t)(r + i) * rb, inner, act + (size_t)s0 * 2 * inner,
                              (size_t)2 * inner, n, de + (size_t)s0 * hidden + r + i, (size_t)hidden);
            }
            continue;
        }
        if (m.type == KQ_BF12 && (bf12 || exact)) {
            kq_rows_bf12_f(w + (size_t)r * rb, rb, ndn, inner, act + (size_t)s0 * 2 * inner,
                           (size_t)2 * inner, n, de + (size_t)s0 * hidden + r, (size_t)hidden);
            continue;
        }
        if (exact) {
            kq_rows_f_any(w + (size_t)r * rb, m.type, ndn, inner, act + (size_t)s0 * 2 * inner,
                          (size_t)2 * inner, n, de + (size_t)s0 * hidden + r, (size_t)hidden);
            continue;
        }
#ifdef KQ_Q4X_Q16
        if (q16 && m.type == KQ_Q4X) {
#ifdef KQ16_PLANES
            if (n == 1) {
                for (int i = 0; i < ndn; i += 16) {
                    kq_q4x_rows_p16(w + (size_t)(r + i) * rb, inner, ah16 + (size_t)s0 * inner,
                                    al16 + (size_t)s0 * inner, an16 + (size_t)s0 * npi,
                                    as16 + (size_t)s0 * npi, de + (size_t)s0 * hidden + r + i);
                }
                continue;
            }
#endif
            for (int i = 0; i < ndn; i += 16) {
                kq_q4x_rows16(w + (size_t)(r + i) * rb, inner, a16 + (size_t)s0 * inner,
                              as16 + (size_t)s0 * npi, n, de + (size_t)s0 * hidden + r + i,
                              (size_t)hidden);
            }
            continue;
        }
#endif
        kq_rows_n2(w + (size_t)r * rb, rb, m.type, ndn, inner, aq + (size_t)s0 * inner,
                   as + (size_t)s0 * npi, am + (size_t)s0 * 2 * npi, an + (size_t)s0 * (inner / 16),
                   x2 ? aq2 + (size_t)s0 * inner : NULL, n,
                   de + (size_t)s0 * hidden + r, (size_t)hidden);
        if (gp) {
            glast = omp_get_wtime();
        }
      }
    }
    if (gp) {
        gt[6] = glast;
        gt[7] = omp_get_wtime();
    }
    moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
    if (gp) {
        gt[8] = omp_get_wtime();
        #pragma omp barrier
        #pragma omp master
        {
            int nt = omp_get_num_threads() < 256 ? omp_get_num_threads() : 256;
            double ref = kq_gt[0][0], amax = ref, m3 = 0, x3 = 0, m6 = 0, x6 = 0;
            for (int q = 0; q < nt; ++q) {
                ref = kq_gt[q][0] < ref ? kq_gt[q][0] : ref;
                amax = kq_gt[q][0] > amax ? kq_gt[q][0] : amax;
            }
            for (int q = 0; q < nt; ++q) {
                m3 += (kq_gt[q][3] - ref) / nt;
                m6 += (kq_gt[q][6] - ref) / nt;
                x3 = kq_gt[q][3] - ref > x3 ? kq_gt[q][3] - ref : x3;
                x6 = kq_gt[q][6] - ref > x6 ? kq_gt[q][6] - ref : x6;
            }
            double v[11] = {amax - ref, gt[1] - ref, gt[2] - ref, m3, x3, gt[4] - ref, gt[5] - ref,
                            m6, x6, gt[7] - ref, gt[8] - ref};
            for (int i = 0; i < 11; ++i) {
                kq_gprof_s[i] += v[i];
            }
            kq_gprof_s[11] += 1;
        }
    }
}

void kq_moe(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
            const float *val, int t, int k, int experts, const int64_t *mats,
            const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out, NULL, 0, NULL, NULL, NULL, NULL);
}

/* kq_moe with a copy of the experts on NUMA node 1 (mats1; slot1: the slot
 * of each expert in it, or -1; null: all), as GP_KQ_MOE. */
void kq_moe_numa(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                 const float *val, int t, int k, int experts, const int64_t *mats,
                 const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out,
                 const int64_t *mats1, const int32_t *slot1)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out, NULL, 0, NULL, mats1, slot1, NULL);
}

/* kq_moe_numa with act flags and the float rows hf (a test: act bit 3 with
 * t <= 4 runs kq_moe_small_body). */
void kq_moe_numa_act(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                     const float *val, int t, int k, int experts, const int64_t *mats,
                     const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out,
                     int act, const float *hf, const int64_t *mats1, const int32_t *slot1,
                     const int32_t *slot0)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out, NULL, act, hf, mats1, slot1, slot0);
}

/* kq_moe with gelu 1: the tanh GELU of the gate in place of SiLU (the Gemma 4
 * 26B). */
void kq_moe_act(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                const float *val, int t, int k, int experts, const int64_t *mats,
                const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out,
                int act_flags, const float *hf)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out, NULL, act_flags, hf, NULL, NULL, NULL);
}

/* The int4 matrices of the Gemma 4 26B (Q4_0 blocks of 18 bytes, the codes in
 * bytes 2 .. 17: value j in the low 4 bits of byte j, j + 16 in the high 4
 * bits; float32 scales, one for each block) to KQ_Q4X: rows a multiple of 16.
 * n matrices of rows rows (stride of the blocks: bstride matrices of rows at
 * the addresses of each). Return 0, or -1 when a scale is not exact in
 * float16. */
int kq_q4x_pack(const uint8_t *packed, const float *scales, int64_t rows, int cols, uint8_t *dst)
{
    int nb = cols / 32, bad = 0;
    int64_t ng = rows / 16;
    #pragma omp parallel for schedule(static) reduction(|:bad)
    for (int64_t g = 0; g < ng; ++g) {
        uint8_t *o = dst + (size_t)g * nb * KQ_Q4X_BB;
        for (int b = 0; b < nb; ++b) {
            uint8_t *blk = o + (size_t)b * KQ_Q4X_BB;
            memset(blk, 0, 256);
            for (int r = 0; r < 16; ++r) {
                int64_t row = 16 * g + r;
                const uint8_t *src = packed + ((size_t)row * nb + b) * 18 + 2;
                for (int v = 0; v < 32; ++v) {
                    int c = v < 16 ? src[v] & 15 : src[v - 16] >> 4;
                    blk[32 * (v / 4) + 4 * (r % 8) + v % 4] |= (uint8_t)(c << (r < 8 ? 0 : 4));
                }
                float sc = scales[(size_t)row * nb + b];
                uint16_t h = kq_f32_to_f16(sc);
                bad |= kq_h((const uint8_t *)&h) != sc;
                memcpy(blk + 256 + 2 * r, &h, 2);
            }
        }
    }
    return bad ? -1 : 0;
}
