/* The bytes of one row of cols values. */
static inline size_t kq_row_bytes(int type, int cols)
{
    switch (type) {
    case KQ_F32: return (size_t)cols * 4;
    case KQ_Q8_0: return (size_t)cols / 32 * 34;
    case KQ_Q5_1: return (size_t)cols / 32 * 24;
    case KQ_Q4_0: return (size_t)cols / 32 * 18;
    case KQ_IQ4_NL: return (size_t)cols / 32 * 18;
    case KQ_Q4_K: return (size_t)cols / 256 * 144;
    case KQ_Q5_K: return (size_t)cols / 256 * 176;
    case KQ_Q6_K: return (size_t)cols / 256 * 210;
    case KQ_IQ2_XXS: return (size_t)cols / 256 * 66;
    case KQ_IQ2R: return ((size_t)cols / 4 + (size_t)cols / 128 + 7) / 8 * 8;
    case KQ_BF16: return (size_t)cols * 2;
    case KQ_NV4: return ((size_t)cols / 2 + (size_t)cols / 16 + 4 + 15) / 16 * 16;
    case KQ_Q8X16: return (size_t)cols / 32 * 36;      /* a group of 16 rows: 16 times this */
    case KQ_NVX: return (size_t)cols / 32 * 18 + 1;    /* a group of 16 rows: 16 times this */
    case KQ_Q4X: return (size_t)cols / 32 * 18;        /* a group of 16 rows: 16 times this */
    case KQ_BF16X16: return (size_t)cols * 2;          /* a group: 16 times this */
    case KQ_F32X16: return (size_t)cols * 4;
    case KQ_BF12: return ((size_t)cols + (size_t)cols / 2 + (size_t)cols / 32 + 15) / 16 * 16;
    case KQ_BF12X16: return (size_t)cols / 32 * 49;    /* a group of 16 rows: 16 times this */
    case KQ_MXX: return (size_t)cols / 32 * 17;        /* a group of 16 rows: 16 times this */
    case KQ_MXF8: return ((size_t)cols + (size_t)cols / 32 + 15) / 16 * 16;
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
            out[32 * b + v] = (0.5f * sc * (float)e2[c]) * g;     /* exact, then one rounding */
        }
    }
}

/* float32 rounded to bfloat16 (to nearest even; finite values), as float32 */
static inline float kq_bf16r(float v)
{
    uint32_t u;
    memcpy(&u, &v, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&v, &u, 4);
    return v;
}

/* The act of FP4-Local's contract (b12x W4A16 and A4, docs/FP4LOCAL_PLAN.md): g and u to bfloat16,
 * h = bf16(bf16(g sigma(g)) u) */
static inline float kq_b12x_act(float g, float u)
{
    g = kq_bf16r(g);
    u = kq_bf16r(u);
    /* (rc_dexp: the bits of the GPU's moe_b12x_act) */
    return kq_bf16r(kq_bf16r(g * (1.f / (1.f + rc_dexp(-g)))) * u);
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

void rc_kq_quant_x(const float *x, int t, int cols, int8_t *xq, float *xs, float *xm)
{
    #pragma omp parallel
    kq_quant_body(x, t, cols, xq, xs, xm);
}
