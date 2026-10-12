/* ---------- int4 weights with int16 activations ----------
 *
 * The int8 activations of the prompt pass move a logit by about 1.1 against
 * the reference. Each product is 0.5 to 1 per cent off. int16 activations
 * cut the step of the quantization by 258 times. This tile has the shape of
 * the int8 tile. One 32-bit lane holds two int16 values of one token. One
 * instruction (vpdpwssd, or vpmaddwd with an add) multiplies them by two
 * weights and adds both products to the lane. A group of 32 values thus
 * takes 16 steps in place of 8.
 *
 * The weights become signed int16 (the nibble less 8), so the sum needs no
 * correction term. One group gives at most 32 * 32767 * 8, which fits in
 * int32. */

/* Quantize x to int16 and write the layout of the tile. The layout of one
 * group is (k / 2, token, 2). stride gives the token stride of qxt and sx.
 * The rows from tokens to stride must be zero. */
void rc_quantize_q16_t(const float *x, int16_t *qxt, float *sx,
                          int tokens, int cols, int stride)
{
    const int groups = cols / 32;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        const float *xt = x + (size_t)t * (size_t)cols;
        for (int g = 0; g < groups; ++g) {
            int16_t q[32];
            sx[(size_t)g * stride + t] = gemma_quant_group32_i16(xt + (size_t)g * 32, q);
            for (int k2 = 0; k2 < 16; ++k2) {
                int16_t *dst = qxt + (((size_t)g * 16 + k2) * (size_t)stride + t) * 2;
                dst[0] = q[k2 * 2];
                dst[1] = q[k2 * 2 + 1];
            }
        }
    }
}

#if GEMMA_X86 && defined(__AVX512F__)
static inline void gemma_int4_q16_tile(const uint8_t *w, const float *scales,
                                       const int16_t *qxt, const float *sx,
                                       float *out, int rows, int cols, int tokens,
                                       int stride, int base, int i0, int t0)
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
    const __m512i eight = _mm512_set1_epi16(8);
    __m512 outf[I4Q_MR];
    for (int r = 0; r < I4Q_MR; ++r) {
        outf[r] = _mm512_setzero_ps();
    }
    for (int g = 0; g < groups; ++g) {
        int16_t exp[I4Q_MR][32] __attribute__((aligned(64)));
        for (int r = 0; r < nrow; ++r) {
            const uint8_t *b = w + (size_t)(i0 + r) * (size_t)wstride
                               + (size_t)g * 18 + 2;
            __m128i raw = _mm_loadu_si128((const __m128i *)b);
            __m128i lo = _mm_and_si128(raw, m4);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
            /* Values 0 to 15 are the low nibbles and 16 to 31 the high ones,
             * the order of the group. */
            __m512i v = _mm512_cvtepu8_epi16(_mm256_set_m128i(hi, lo));
            _mm512_store_si512((void *)exp[r], _mm512_sub_epi16(v, eight));
        }
        const float *xg = sx + (size_t)g * (size_t)stride + base + t0;
        __m512 sxv = _mm512_loadu_ps(xg);
        __m512i acc[I4Q_MR];
        for (int r = 0; r < I4Q_MR; ++r) {
            acc[r] = _mm512_setzero_si512();
        }
        for (int k2 = 0; k2 < 16; ++k2) {
            __m512i qv = _mm512_loadu_si512((const void *)(
                qxt + (((size_t)g * 16 + k2) * (size_t)stride + base + t0) * 2));
            for (int r = 0; r < nrow; ++r) {
                __m512i wv = _mm512_set1_epi32(*(const int32_t *)(exp[r] + k2 * 2));
#if defined(__AVX512VNNI__)
                acc[r] = _mm512_dpwssd_epi32(acc[r], wv, qv);
#else
                acc[r] = _mm512_add_epi32(acc[r], _mm512_madd_epi16(wv, qv));
#endif
            }
        }
        for (int r = 0; r < nrow; ++r) {
            float wsc = scales[(size_t)(i0 + r) * (size_t)groups + g];
            outf[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[r]),
                                      _mm512_mul_ps(_mm512_set1_ps(wsc), sxv), outf[r]);
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
/* Scalar version of the int16 tile. */
static inline void gemma_int4_q16_tile(const uint8_t *w, const float *scales,
                                       const int16_t *qxt, const float *sx,
                                       float *out, int rows, int cols, int tokens,
                                       int stride, int base, int i0, int t0)
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
                    int nib = k < 16 ? (b[k] & 0x0F) : (b[k - 16] >> 4);
                    const int16_t *qd = qxt + (((size_t)g * 16 + k / 2) * (size_t)stride
                                               + base + t0 + t) * 2 + (k % 2);
                    dot += (nib - 8) * (int32_t)qd[0];
                }
                acc += (float)dot * scales[(size_t)(i0 + r) * (size_t)groups + g]
                       * sx[(size_t)g * stride + base + t0 + t];
            }
            out[(size_t)(base + t0 + t) * (size_t)rows + (size_t)i0 + r] = acc;
        }
    }
}
#endif

void rc_int4_q16_tile_run(const uint8_t *w, const float *scales,
                             const int16_t *qxt, const float *sx, float *out,
                             int rows, int cols, int tokens, int stride)
{
    const int nrb = (rows + I4Q_MR - 1) / I4Q_MR;
    const int ntb = (stride + I4Q_TB - 1) / I4Q_TB;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int rb = 0; rb < nrb; ++rb) {
        for (int tb = 0; tb < ntb; ++tb) {
            gemma_int4_q16_tile(w, scales, qxt, sx, out, rows, cols, tokens, stride, 0,
                                rb * I4Q_MR, tb * I4Q_TB);
        }
    }
}

void rc_int4_q8_tile_run(const uint8_t *w, const float *scales,
                            const int8_t *qxt, const float *sx,
                            const int32_t *sumx, float *out,
                            int rows, int cols, int tokens, int stride)
{
    const int nrb = (rows + I4Q_MR - 1) / I4Q_MR;
#if GEMMA_X86
    if (gemma_i4q_tb8) {
        const int ntb = (stride + I4Q2_TB - 1) / I4Q2_TB;
        #pragma omp parallel for schedule(static) collapse(2)
        for (int rb = 0; rb < nrb; ++rb) {
            for (int tb = 0; tb < ntb; ++tb) {
                gemma_int4_q8_tile_narrow(w, scales, qxt, sx, sumx, out,
                                          rows, cols, tokens, stride, 0,
                                          rb * I4Q2_MR, tb * I4Q2_TB);
            }
        }
        return;
    }
#endif
    const int ntb = (stride + I4Q_TB - 1) / I4Q_TB;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int rb = 0; rb < nrb; ++rb) {
        for (int tb = 0; tb < ntb; ++tb) {
            gemma_int4_q8_tile(w, scales, qxt, sx, sumx, out,
                               rows, cols, tokens, stride, 0, rb * I4Q_MR,
                               tb * I4Q_TB);
        }
    }
}

/* Run the wide tile over every row block and token block. On a target without
 * AVX-512 the code uses the normal tile. */
void rc_int4_q8_tile_run32(const uint8_t *w, const float *scales,
                              const int8_t *qxt, const float *sx,
                              const int32_t *sumx, float *out,
                              int rows, int cols, int tokens, int stride)
{
#if GEMMA_X86 && defined(__AVX512F__)
    const int nrb = (rows + I4Q32_MR - 1) / I4Q32_MR;
    const int ntb = (stride + I4Q32_TB - 1) / I4Q32_TB;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int rb = 0; rb < nrb; ++rb) {
        for (int tb = 0; tb < ntb; ++tb) {
            gemma_int4_q8_tile32(w, scales, qxt, sx, sumx, out,
                                 rows, cols, tokens, stride, 0, rb * I4Q32_MR,
                                 tb * I4Q32_TB);
        }
    }
#else
    rc_int4_q8_tile_run(w, scales, qxt, sx, sumx, out,
                           rows, cols, tokens, stride);
#endif
}
