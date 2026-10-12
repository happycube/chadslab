/* ---------- int4 weights with int8 activations ----------
 * The int4 dot above converts each weight value to float32. An int8 activation
 * lets the dot use integer multiply and add. The instruction maddubs does
 * sixteen multiply and add pairs at once.
 *
 * A group holds 32 values. The Q4_0 block holds 16 bytes. Byte j holds value j
 * in the low nibble and value j+16 in the high nibble. The value of a nibble is
 * the nibble minus 8. The activation x becomes int8 with one float32 scale and
 * one integer sum for each group of 32 values:
 *
 *     sx   = max(abs(group)) / 127
 *     qx   = round(x / sx)
 *     sumx = sum(qx)
 *
 * Then, for one group:
 *     sum_i (nib_i - 8) * qx_i = dot - 8 * sumx
 *     result = (dot - 8 * sumx) * wscale * sx
 *
 * The tile reads the activations in a transposed layout. For one group the
 * layout is (k / 4, token, 4). Thus a 64-byte vector holds four k values for
 * each of sixteen tokens. One maddubs gives sixteen int16 pair sums. One madd
 * then gives sixteen int32 lane sums, one lane for each token. The lanes are
 * the tokens, so the pointwise kernel sums over k with no horizontal sum.
 */

/* Quantize one group of 32 float values to int8. q gets 32 bytes. Return the
 * scale and put the integer sum of the group in *sum. The vector path divides
 * by the scale, so the result matches a float32 divide and rint. */
static inline float gemma_quant_group32(const float *x, int8_t *q, int32_t *sum)
{
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 v0 = _mm512_loadu_ps(x);
    __m512 v1 = _mm512_loadu_ps(x + 16);
    __m512 n0 = _mm512_sub_ps(_mm512_setzero_ps(), v0);
    __m512 n1 = _mm512_sub_ps(_mm512_setzero_ps(), v1);
    float amax = _mm512_reduce_max_ps(_mm512_max_ps(
        _mm512_max_ps(v0, n0), _mm512_max_ps(v1, n1)));
    float sc = amax > 0.0f ? amax / 127.0f : 1e-12f;
    __m512 sv = _mm512_set1_ps(sc);
    const __m512i lo = _mm512_set1_epi32(-127);
    const __m512i hi = _mm512_set1_epi32(127);
    __m512i q0 = _mm512_max_epi32(_mm512_min_epi32(
        _mm512_cvtps_epi32(_mm512_div_ps(v0, sv)), hi), lo);
    __m512i q1 = _mm512_max_epi32(_mm512_min_epi32(
        _mm512_cvtps_epi32(_mm512_div_ps(v1, sv)), hi), lo);
    *sum = _mm512_reduce_add_epi32(_mm512_add_epi32(q0, q1));
    _mm_storeu_si128((__m128i *)(q + 0), _mm512_cvtepi32_epi8(q0));
    _mm_storeu_si128((__m128i *)(q + 16), _mm512_cvtepi32_epi8(q1));
    return sc;
#else
    float amax = 0.0f;
    for (int k = 0; k < 32; ++k) {
        float a = fabsf(x[k]);
        if (a > amax) {
            amax = a;
        }
    }
    float sc = amax > 0.0f ? amax / 127.0f : 1e-12f;
    int32_t s = 0;
    for (int k = 0; k < 32; ++k) {
        long v = lrintf(x[k] / sc);
        if (v > 127) {
            v = 127;
        }
        if (v < -127) {
            v = -127;
        }
        q[k] = (int8_t)v;
        s += (int32_t)v;
    }
    *sum = s;
    return sc;
#endif
}

/* Quantize the last axis of x with one scale for each group of 32 values.
 * qx gets one int8 for each value, with the shape (tokens, cols). sx gets one
 * float32 for each group. sumx gets the integer sum of each group. */
static void gemma_quantize_q8_groups_body(const float *x, int8_t *qx, float *sx,
                              int32_t *sumx, int rows, int cols)
{
    const int groups = cols / 32;
    #pragma omp for schedule(static)
    for (int t = 0; t < rows; ++t) {
        const float *xt = x + (size_t)t * (size_t)cols;
        int8_t *qt = qx + (size_t)t * (size_t)cols;
        float *st = sx + (size_t)t * (size_t)groups;
        int32_t *mt = sumx + (size_t)t * (size_t)groups;
        for (int g = 0; g < groups; ++g) {
            st[g] = gemma_quant_group32(xt + (size_t)g * 32,
                                        qt + (size_t)g * 32, mt + g);
        }
    }
}

void rc_quantize_q8_groups(const float *x, int8_t *qx, float *sx,
                              int32_t *sumx, int rows, int cols)
{
    #pragma omp parallel
    gemma_quantize_q8_groups_body(x, qx, sx, sumx, rows, cols);
}

/* Quantize the last axis of x to int8 and write the transposed layout. The
 * layout for one group is (k / 4, token, 4). stride gives the token stride of
 * qxt, sx, and sumx. The rows from tokens to stride must be zero. */
void rc_quantize_q8_t(const float *x, int8_t *qxt, float *sx, int32_t *sumx,
                         int tokens, int cols, int stride)
{
    const int groups = cols / 32;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        const float *xt = x + (size_t)t * (size_t)cols;
        for (int g = 0; g < groups; ++g) {
            int8_t q[32];
            int32_t s;
            float sc = gemma_quant_group32(xt + (size_t)g * 32, q, &s);
            sx[(size_t)g * stride + t] = sc;
            sumx[(size_t)g * stride + t] = s;
            for (int q4 = 0; q4 < 8; ++q4) {
                int8_t *dst = qxt + (((size_t)g * 8 + q4) * (size_t)stride + t) * 4;
                dst[0] = q[q4 * 4 + 0];
                dst[1] = q[q4 * 4 + 1];
                dst[2] = q[q4 * 4 + 2];
                dst[3] = q[q4 * 4 + 3];
            }
        }
    }
}


/* The token block and the row block of the int8 tile. AVX-512 holds sixteen
 * tokens in one register. AVX2 holds eight. */
#if GEMMA_X86 && defined(__AVX512F__)
#define I4Q_TB 16
#define I4Q_MR 8
#else
#define I4Q_TB 8
#define I4Q_MR 4
#endif

/* Use the prefetch of the next weight group in the int8 tile. A test uses it. */
static int gemma_prefetch = 0;

void rc_int4_q8_set_prefetch(int on)
{
    gemma_prefetch = on ? 1 : 0;
}

#if GEMMA_X86 && defined(__AVX512F__)
/* Process one row block and one token block of the int8 tile. base gives the
 * token offset of the expert in the shared scratch buffers. tokens is the token
 * count of the expert. */
static inline void gemma_int4_q8_tile(const uint8_t *w, const float *scales,
                                      const int8_t *qxt, const float *sx,
                                      const int32_t *sumx, float *out,
                                      int rows, int cols, int tokens, int stride,
                                      int base, int i0, int t0)
{
    const int groups = cols / 32;
    const int wstride = groups * 18;
    int nrow = rows - i0;
    if (nrow > I4Q_MR) {
        nrow = I4Q_MR;
    }
    int ntok = tokens - t0;
    if (ntok > I4Q_TB) {
        ntok = I4Q_TB;
    }
    const __m128i m4 = _mm_set1_epi8(0x0F);
#if !defined(__AVX512VNNI__)
    const __m512i ones = _mm512_set1_epi16(1);
#endif
    const __m512 eight = _mm512_set1_ps(8.0f);
    __m512 outf[I4Q_MR];
    for (int r = 0; r < I4Q_MR; ++r) {
        outf[r] = _mm512_setzero_ps();
    }
    for (int g = 0; g < groups; ++g) {
        uint8_t exp[I4Q_MR][32];
        for (int r = 0; r < nrow; ++r) {
            const uint8_t *b = w + (size_t)(i0 + r) * (size_t)wstride
                               + (size_t)g * 18 + 2;
            __m128i raw = _mm_loadu_si128((const __m128i *)b);
            __m128i lo = _mm_and_si128(raw, m4);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
            /* Bytes 0 to 15 hold columns 0 to 15 and bytes 16 to 31 hold
             * columns 16 to 31. This matches the order of the q8 group. */
            _mm256_storeu_si256((__m256i *)exp[r], _mm256_set_m128i(hi, lo));
        }
        /* The weights of the next group and the activation vectors of the next
         * group are far from the line in hand. Ask the prefetcher for them
         * while the multiply runs. Set rc_int4_q8_set_prefetch(1). */
        if (gemma_prefetch && g + 1 < groups) {
            for (int r = 0; r < nrow; ++r) {
                _mm_prefetch((const char *)(w + (size_t)(i0 + r) * (size_t)wstride
                                            + (size_t)(g + 1) * 18 + 2), _MM_HINT_T0);
            }
            for (int q4 = 0; q4 < 8; ++q4) {
                _mm_prefetch((const char *)(qxt + (((size_t)(g + 1) * 8 + q4)
                             * (size_t)stride + base + t0) * 4), _MM_HINT_T0);
            }
        }
        const int32_t *sg = sumx + (size_t)g * (size_t)stride + base + t0;
        const float *xg = sx + (size_t)g * (size_t)stride + base + t0;
        __m512 sumf = _mm512_cvtepi32_ps(_mm512_loadu_si512((const void *)sg));
        __m512 sxv = _mm512_loadu_ps(xg);
        __m512i acc[I4Q_MR];
        for (int r = 0; r < I4Q_MR; ++r) {
            acc[r] = _mm512_setzero_si512();
        }
        /* The q8 vector depends on the group and on q4, not on the row. Load
         * it one time and use it for every row of the block. */
        for (int q4 = 0; q4 < 8; ++q4) {
            __m512i qv = _mm512_loadu_si512((const void *)(
                qxt + (((size_t)g * 8 + q4) * (size_t)stride
                       + base + t0) * 4));
            for (int r = 0; r < nrow; ++r) {
                __m512i wv = _mm512_set1_epi32(*(const int32_t *)(exp[r] + q4 * 4));
#if defined(__AVX512VNNI__)
                acc[r] = _mm512_dpbusd_epi32(acc[r], wv, qv);
#else
                __m512i p = _mm512_maddubs_epi16(wv, qv);
                acc[r] = _mm512_add_epi32(acc[r], _mm512_madd_epi16(p, ones));
#endif
            }
        }
        for (int r = 0; r < nrow; ++r) {
            __m512 f = _mm512_sub_ps(_mm512_cvtepi32_ps(acc[r]),
                                     _mm512_mul_ps(sumf, eight));
            float wsc = scales[(size_t)(i0 + r) * (size_t)groups + g];
            outf[r] = _mm512_fmadd_ps(f, _mm512_mul_ps(_mm512_set1_ps(wsc), sxv),
                                      outf[r]);
        }
    }
    for (int r = 0; r < nrow; ++r) {
        float tmp[I4Q_TB];
        _mm512_storeu_ps(tmp, outf[r]);
        for (int t = 0; t < ntok; ++t) {
            out[(size_t)(base + t0 + t) * (size_t)rows + (size_t)i0 + r] = tmp[t];
        }
    }
}
#else
/* Scalar version of the int8 tile. Use it for a comparison and for a machine
 * without AVX-512. */
static inline void gemma_int4_q8_tile(const uint8_t *w, const float *scales,
                                      const int8_t *qxt, const float *sx,
                                      const int32_t *sumx, float *out,
                                      int rows, int cols, int tokens, int stride,
                                      int base, int i0, int t0)
{
    const int groups = cols / 32;
    const int wstride = groups * 18;
    for (int r = 0; r < I4Q_MR && i0 + r < rows; ++r) {
        for (int t = 0; t < I4Q_TB && t0 + t < tokens; ++t) {
            float acc = 0.0f;
            for (int g = 0; g < groups; ++g) {
                const uint8_t *b = w + (size_t)(i0 + r) * (size_t)wstride
                                   + (size_t)g * 18 + 2;
                int32_t dot = 0;
                for (int k = 0; k < 32; ++k) {
                    const int8_t *qd = qxt + (((size_t)g * 8 + k / 4) * (size_t)stride
                                              + base + t0 + t) * 4 + (k % 4);
                    int nib = (k < 16) ? (b[k] & 0x0F) : ((b[k - 16] >> 4) & 0x0F);
                    dot += nib * (int32_t)*qd;
                }
                dot -= 8 * sumx[(size_t)g * (size_t)stride + base + t0 + t];
                acc += (float)dot * scales[(size_t)(i0 + r) * (size_t)groups + g]
                       * sx[(size_t)g * (size_t)stride + base + t0 + t];
            }
            out[(size_t)(base + t0 + t) * (size_t)rows + (size_t)i0 + r] = acc;
        }
    }
}
#endif

/* The narrow tile uses eight tokens in one 256-bit register. It wastes fewer
 * lanes when an expert holds fewer than sixteen tokens. */
#define I4Q2_TB 8
#define I4Q2_MR 8
#if GEMMA_X86

static inline void gemma_int4_q8_tile_narrow(const uint8_t *w, const float *scales,
                                             const int8_t *qxt, const float *sx,
                                             const int32_t *sumx, float *out,
                                             int rows, int cols, int tokens,
                                             int stride, int base, int i0, int t0)
{
    const int groups = cols / 32;
    const int wstride = groups * 18;
    int nrow = rows - i0;
    if (nrow > I4Q2_MR) {
        nrow = I4Q2_MR;
    }
    int ntok = tokens - t0;
    if (ntok > I4Q2_TB) {
        ntok = I4Q2_TB;
    }
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256 eight = _mm256_set1_ps(8.0f);
    __m256 outf[I4Q2_MR];
    for (int r = 0; r < I4Q2_MR; ++r) {
        outf[r] = _mm256_setzero_ps();
    }
    for (int g = 0; g < groups; ++g) {
        uint8_t exp[I4Q2_MR][32];
        for (int r = 0; r < nrow; ++r) {
            const uint8_t *b = w + (size_t)(i0 + r) * (size_t)wstride
                               + (size_t)g * 18 + 2;
            __m128i raw = _mm_loadu_si128((const __m128i *)b);
            __m128i lo = _mm_and_si128(raw, m4);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
            _mm256_storeu_si256((__m256i *)exp[r], _mm256_set_m128i(hi, lo));
        }
        const int32_t *sg = sumx + (size_t)g * (size_t)stride + base + t0;
        const float *xg = sx + (size_t)g * (size_t)stride + base + t0;
        __m256 sumf = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)sg));
        __m256 sxv = _mm256_loadu_ps(xg);
        __m256i acc[I4Q2_MR];
        for (int r = 0; r < I4Q2_MR; ++r) {
            acc[r] = _mm256_setzero_si256();
        }
        /* The q8 vector depends on the group and on q4, not on the row. Load
         * it one time and use it for every row of the block. */
        for (int q4 = 0; q4 < 8; ++q4) {
            __m256i qv = _mm256_loadu_si256((const __m256i *)(
                qxt + (((size_t)g * 8 + q4) * (size_t)stride
                       + base + t0) * 4));
            for (int r = 0; r < nrow; ++r) {
                __m256i wv = _mm256_set1_epi32(*(const int32_t *)(exp[r] + q4 * 4));
                __m256i p = _mm256_maddubs_epi16(wv, qv);
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(p, ones));
            }
        }
        for (int r = 0; r < nrow; ++r) {
            __m256 f = _mm256_sub_ps(_mm256_cvtepi32_ps(acc[r]),
                                     _mm256_mul_ps(sumf, eight));
            float wsc = scales[(size_t)(i0 + r) * (size_t)groups + g];
            outf[r] = _mm256_fmadd_ps(f, _mm256_mul_ps(_mm256_set1_ps(wsc), sxv),
                                      outf[r]);
        }
    }
    for (int r = 0; r < nrow; ++r) {
        float tmp[I4Q2_TB];
        _mm256_storeu_ps(tmp, outf[r]);
        for (int t = 0; t < ntok; ++t) {
            out[(size_t)(base + t0 + t) * (size_t)rows + (size_t)i0 + r] = tmp[t];
        }
    }
}
#endif

/* The wide tile processes 32 tokens in one pass over the groups. The weight
 * decode is done one time for the whole block, and the weight broadcast serves
 * two token vectors. That cuts the instructions that do not multiply, which the
 * VNNI test says are the larger part of the time. */
#if GEMMA_X86 && defined(__AVX512F__)
#define I4Q32_TB 32
#define I4Q32_MR 4

static inline void gemma_int4_q8_tile32(const uint8_t *w, const float *scales,
                                        const int8_t *qxt, const float *sx,
                                        const int32_t *sumx, float *out,
                                        int rows, int cols, int tokens, int stride,
                                        int base, int i0, int t0)
{
    const int groups = cols / 32;
    const int wstride = groups * 18;
    int nrow = rows - i0;
    if (nrow > I4Q32_MR) {
        nrow = I4Q32_MR;
    }
    int ntok = tokens - t0;
    if (ntok > I4Q32_TB) {
        ntok = I4Q32_TB;
    }
    const __m128i m4 = _mm_set1_epi8(0x0F);
#if !defined(__AVX512VNNI__)
    const __m512i ones = _mm512_set1_epi16(1);
#endif
    const __m512 eight = _mm512_set1_ps(8.0f);
    __m512 outf[I4Q32_MR][2];
    for (int r = 0; r < I4Q32_MR; ++r) {
        for (int h = 0; h < 2; ++h) {
            outf[r][h] = _mm512_setzero_ps();
        }
    }
    for (int g = 0; g < groups; ++g) {
        uint8_t exp[I4Q32_MR][32];
        for (int r = 0; r < nrow; ++r) {
            const uint8_t *b = w + (size_t)(i0 + r) * (size_t)wstride
                               + (size_t)g * 18 + 2;
            __m128i raw = _mm_loadu_si128((const __m128i *)b);
            __m128i lo = _mm_and_si128(raw, m4);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
            _mm256_storeu_si256((__m256i *)exp[r], _mm256_set_m128i(hi, lo));
        }
        __m512i acc[I4Q32_MR][2];
        __m512 sumf[2];
        __m512 sxv[2];
        int live = 0;
        for (int h = 0; h < 2; ++h) {
            const int th = t0 + h * 16;
            if (h * 16 >= ntok || th >= tokens) {
                continue;
            }
            live = h + 1;
            sumf[h] = _mm512_cvtepi32_ps(_mm512_loadu_si512(
                (const void *)(sumx + (size_t)g * (size_t)stride + base + th)));
            sxv[h] = _mm512_loadu_ps(sx + (size_t)g * (size_t)stride + base + th);
            for (int r = 0; r < nrow; ++r) {
                acc[r][h] = _mm512_setzero_si512();
            }
        }
        for (int q4 = 0; q4 < 8; ++q4) {
            __m512i qv[2];
            for (int h = 0; h < live; ++h) {
                qv[h] = _mm512_loadu_si512((const void *)(qxt
                    + (((size_t)g * 8 + q4) * (size_t)stride + base + t0 + h * 16) * 4));
            }
            for (int r = 0; r < nrow; ++r) {
                const __m512i wv = _mm512_set1_epi32(*(const int32_t *)(exp[r] + q4 * 4));
                for (int h = 0; h < live; ++h) {
#if defined(__AVX512VNNI__)
                    acc[r][h] = _mm512_dpbusd_epi32(acc[r][h], wv, qv[h]);
#else
                    const __m512i p = _mm512_maddubs_epi16(wv, qv[h]);
                    acc[r][h] = _mm512_add_epi32(acc[r][h], _mm512_madd_epi16(p, ones));
#endif
                }
            }
        }
        for (int r = 0; r < nrow; ++r) {
            const float wsc = scales[(size_t)(i0 + r) * (size_t)groups + g];
            for (int h = 0; h < live; ++h) {
                const __m512 f = _mm512_sub_ps(_mm512_cvtepi32_ps(acc[r][h]),
                                               _mm512_mul_ps(sumf[h], eight));
                outf[r][h] = _mm512_fmadd_ps(
                    f, _mm512_mul_ps(_mm512_set1_ps(wsc), sxv[h]), outf[r][h]);
            }
        }
    }
    for (int r = 0; r < nrow; ++r) {
        for (int h = 0; h < 2; ++h) {
            const int th = t0 + h * 16;
            if (h * 16 >= ntok || th >= tokens) {
                break;
            }
            float tmp[16];
            _mm512_storeu_ps(tmp, outf[r][h]);
            const int nm = tokens - th < 16 ? tokens - th : 16;
            for (int t = 0; t < nm; ++t) {
                out[(size_t)(base + th + t) * (size_t)rows + (size_t)i0 + r] = tmp[t];
            }
        }
    }
}
#endif

static int gemma_i4q_tb8 = 0;

/* Select the narrow token block of eight tokens (1) or the wide block of
 * sixteen (0). Use this for a test. */
void rc_int4_q8_set_tb8(int on)
{
    gemma_i4q_tb8 = on ? 1 : 0;
}

/* Run the int8 tile over every row block and token block. tokens is the true
 * token count and stride is the token stride of qxt, sx, and sumx. */
