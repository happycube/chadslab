/* ---------- Q6_K kernel ----------
 * The Q6_K type holds 256 values in a 210-byte block. One block gives ql[128],
 * qh[64], sc[16] as int8, and d as float16. The low 4 bits of a value are in
 * ql, the top 2 bits are in qh, and one int8 scale covers each group of 16.
 * The value is the 6 bits minus 32.
 *
 * The QAT files keep the tied embedding table in Q6_K. The model thus reads
 * 6.05 bits for each weight of the output head in place of 16 bits. The load
 * step does not dequantize the table.
 *
 * The kernel decodes a block in registers. It does not write the values to a
 * temporary array. For one token the cost is then near the memory limit.
 */

/* Convert a float16 bit pattern to a float32 value. */
static inline float fp16_to_f32(uint16_t bits)
{
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exp = (bits >> 10) & 0x1Fu;
    uint32_t man = bits & 0x03FFu;
    uint32_t u;
    if (exp == 0u) {
        if (man == 0u) {
            u = sign;
        } else {
            exp = 113u;
            while ((man & 0x0400u) == 0u) { man <<= 1; --exp; }
            man &= 0x03FFu;
            u = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31u) {
        u = sign | 0x7F800000u | (man << 13);
    } else {
        u = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* Decode one 256-value Q6_K block to float32 values in y. */
static inline void q6k_decode_block(const uint8_t *blk, float *y)
{
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 192);
    float d = fp16_to_f32((uint16_t)((uint16_t)blk[208] | ((uint16_t)blk[209] << 8)));
    for (int h = 0; h < 2; ++h) {
        const uint8_t *q0 = ql + h * 64;
        const uint8_t *g0 = qh + h * 32;
        const int8_t *s0 = sc + h * 8;
        float *yh = y + h * 128;
        for (int l = 0; l < 32; ++l) {
            int is = l >> 4;
            int q1 = ((q0[l] & 0x0F) | (((g0[l] >> 0) & 3) << 4)) - 32;
            int q2 = ((q0[l + 32] & 0x0F) | (((g0[l] >> 2) & 3) << 4)) - 32;
            int q3 = ((q0[l] >> 4) | (((g0[l] >> 4) & 3) << 4)) - 32;
            int q4 = ((q0[l + 32] >> 4) | (((g0[l] >> 6) & 3) << 4)) - 32;
            yh[l]      = d * (float)s0[is + 0] * (float)q1;
            yh[l + 32] = d * (float)s0[is + 2] * (float)q2;
            yh[l + 64] = d * (float)s0[is + 4] * (float)q3;
            yh[l + 96] = d * (float)s0[is + 6] * (float)q4;
        }
    }
}

/* Return the index of the largest of the n values of x. At an equal value
 * the lower index wins, as np.argmax. The first pass finds the largest value
 * with AVX2 max; the second finds its first index with a compare and a mask.
 * A NaN is not the largest value here (np.argmax gives the NaN). */
int64_t rc_argmax(const float *x, int64_t n)
{
    float best = -INFINITY;
    int64_t i = 0;
#ifdef __AVX2__
    __m256 m0 = _mm256_set1_ps(-INFINITY), m1 = m0, m2 = m0, m3 = m0;
    for (; i + 32 <= n; i += 32) {
        m0 = _mm256_max_ps(m0, _mm256_loadu_ps(x + i));
        m1 = _mm256_max_ps(m1, _mm256_loadu_ps(x + i + 8));
        m2 = _mm256_max_ps(m2, _mm256_loadu_ps(x + i + 16));
        m3 = _mm256_max_ps(m3, _mm256_loadu_ps(x + i + 24));
    }
    float t[8];
    _mm256_storeu_ps(t, _mm256_max_ps(_mm256_max_ps(m0, m1), _mm256_max_ps(m2, m3)));
    for (int k = 0; k < 8; ++k) {
        best = t[k] > best ? t[k] : best;
    }
#endif
    for (; i < n; ++i) {
        best = x[i] > best ? x[i] : best;
    }
    int64_t j = 0;
#ifdef __AVX2__
    __m256 b = _mm256_set1_ps(best);
    for (; j + 8 <= n; j += 8) {
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(x + j), b, _CMP_EQ_OQ));
        if (mask) {
            return j + __builtin_ctz(mask);
        }
    }
#endif
    for (; j < n; ++j) {
        if (x[j] == best) {
            return j;
        }
    }
    return 0;
}

/* Decode the rows ids[0..n) of a Q6_K table into out, shape (n, cols). This
 * is the embedding lookup of a prompt: the rows of the E4B tables are Q6_K. */
void rc_q6k_rows(const uint8_t *table, const int64_t *ids, int n, int cols, float *out)
{
    size_t row_bytes = (size_t)(cols >> 8) * 210u;
    int64_t total = (int64_t)n * (cols >> 8);
    /* One token (a decode step) is too little work for the threads. */
    #pragma omp parallel for schedule(static) if ((size_t)n * cols > (1u << 18))
    for (int64_t i = 0; i < total; ++i) {
        int64_t r = i / (cols >> 8), b = i % (cols >> 8);
        q6k_decode_block(table + (size_t)ids[r] * row_bytes + (size_t)b * 210u,
                         out + (size_t)r * cols + (size_t)b * 256u);
    }
}

/* The rows ids of a Q4_0 table as float32, as _dequant of gguf.py: (nibble -
 * 8) times the float16 scale d, in float32. The same bits. The QAT files of
 * unsloth keep the tables of the E2B and the E4B in Q4_0. */
void rc_q4_0_rows(const uint8_t *table, const int64_t *ids, int n, int cols, float *out)
{
    int nb = cols / 32;
    size_t row_bytes = (size_t)nb * 18u;
    int64_t total = (int64_t)n * nb;
    #pragma omp parallel for schedule(static) if ((size_t)n * cols > (1u << 18))
    for (int64_t i = 0; i < total; ++i) {
        int64_t r = i / nb, b = i % nb;
        const uint8_t *blk = table + (size_t)ids[r] * row_bytes + (size_t)b * 18u;
        float d = fp16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        float *y = out + (size_t)r * cols + (size_t)b * 32u;
        for (int l = 0; l < 16; ++l) {
            y[l] = (float)((blk[2 + l] & 0xF) - 8) * d;
            y[l + 16] = (float)((blk[2 + l] >> 4) - 8) * d;
        }
    }
}

/* The rows ids of a Q4_K (five 0) or Q5_K (five 1) table as float32, as
 * _dequant of gguf.py: d sc q - dmin m in double (NumPy promotes the float32
 * scale and the int32 values), then float32. The same bits. */
void rc_kq45_rows(const uint8_t *table, const int64_t *ids, int n, int cols, int five,
                     float *out)
{
    size_t bs = five ? 176u : 144u, row_bytes = (size_t)(cols >> 8) * bs;
    int64_t total = (int64_t)n * (cols >> 8);
    #pragma omp parallel for schedule(static) if ((size_t)n * cols > (1u << 18))
    for (int64_t i = 0; i < total; ++i) {
        int64_t r = i / (cols >> 8), b = i % (cols >> 8);
        const uint8_t *blk = table + (size_t)ids[r] * row_bytes + (size_t)b * bs;
        const uint8_t *sc = blk + 4, *qh = blk + 16, *qs = blk + (five ? 48 : 16);
        double d = fp16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
        double dmin = fp16_to_f32((uint16_t)(blk[2] | (blk[3] << 8)));
        float *y = out + (size_t)r * cols + (size_t)b * 256u;
        for (int s = 0; s < 8; ++s) {
            int sv, mv;
            if (s < 4) {
                sv = sc[s] & 63;
                mv = sc[s + 4] & 63;
            } else {
                sv = (sc[s + 4] & 0xF) | ((sc[s - 4] >> 6) << 4);
                mv = (sc[s + 4] >> 4) | ((sc[s] >> 6) << 4);
            }
            int j = s / 2, high = s % 2;
            double ds = d * (double)sv, dm = dmin * (double)mv;
            for (int l = 0; l < 32; ++l) {
                int q = high ? qs[32 * j + l] >> 4 : qs[32 * j + l] & 0xF;
                if (five) {
                    q |= ((qh[l] >> (2 * j + high)) & 1) << 4;
                }
                y[32 * s + l] = (float)(ds * (double)q - dm);
            }
        }
    }
}

/* Return the dot product of one Q6_K row and one float32 row. Scalar form. */
static float dot_q6k_row_scalar(const uint8_t *w, const float *x, int cols)
{
    int nb = cols >> 8;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        float y[256];
        q6k_decode_block(w + (size_t)b * 210u, y);
        const float *xb = x + (size_t)b * 256u;
        for (int k = 0; k < 256; ++k) {
            acc += y[k] * xb[k];
        }
    }
    return acc;
}

/* Multiply x by W. W is Q6_K data. Scalar form. */
static void gemma_q6k_scalar_body(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    size_t row_bytes = (size_t)(cols >> 8) * 210u;
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint8_t *wi = w + (size_t)i * row_bytes;
        for (int t = 0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_q6k_row_scalar(wi, x + (size_t)t * (size_t)cols, cols);
        }
    }
}

void rc_q6k_scalar(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_q6k_scalar_body(w, x, out, rows, cols, tokens);
}

#if GEMMA_X86

/* Widen 16 signed 8-bit values to two float32 vectors. Subtract 32. */
#define Q6K_WIDEN2(v, lo, hi) do { \
    __m256i q6k_off = _mm256_set1_epi32(32); \
    (lo) = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepi8_epi32(v), q6k_off)); \
    (hi) = _mm256_cvtepi32_ps(_mm256_sub_epi32( \
        _mm256_cvtepi8_epi32(_mm_srli_si128((v), 8)), q6k_off)); \
} while (0)

__attribute__((target("avx2,fma")))
static float dot_q6k_row_avx2(const uint8_t *w, const float *x, int cols)
{
    int nb = cols >> 8;
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = w + (size_t)b * 210u;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        float d = fp16_to_f32((uint16_t)((uint16_t)blk[208] | ((uint16_t)blk[209] << 8)));
        const float *xb = x + (size_t)b * 256u;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (int h = 0; h < 2; ++h) {
            const uint8_t *q0 = ql + h * 64;
            const uint8_t *g0 = qh + h * 32;
            const int8_t *s0 = sc + h * 8;
            const float *xh = xb + h * 128;
            for (int l0 = 0; l0 < 32; l0 += 16) {
                int is = l0 >> 4;
                __m128i a = _mm_loadu_si128((const __m128i *)(q0 + l0));
                __m128i c = _mm_loadu_si128((const __m128i *)(q0 + 32 + l0));
                __m128i g = _mm_loadu_si128((const __m128i *)(g0 + l0));
                __m128i m0f = _mm_set1_epi8(0x0F);
                __m128i m3 = _mm_set1_epi8(0x03);
                __m128i q1 = _mm_or_si128(_mm_and_si128(a, m0f),
                                          _mm_slli_epi16(_mm_and_si128(g, m3), 4));
                __m128i q2 = _mm_or_si128(_mm_and_si128(c, m0f),
                                          _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(g, 2), m3), 4));
                __m128i q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(a, 4), m0f),
                                          _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(g, 4), m3), 4));
                __m128i q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(c, 4), m0f),
                                          _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(g, 6), m3), 4));
                __m256 k0, k1;
                Q6K_WIDEN2(q1, k0, k1);
                k0 = _mm256_mul_ps(k0, _mm256_set1_ps((float)s0[is + 0]));
                k1 = _mm256_mul_ps(k1, _mm256_set1_ps((float)s0[is + 0]));
                acc0 = _mm256_fmadd_ps(k0, _mm256_loadu_ps(xh + l0), acc0);
                acc1 = _mm256_fmadd_ps(k1, _mm256_loadu_ps(xh + l0 + 8), acc1);
                Q6K_WIDEN2(q2, k0, k1);
                k0 = _mm256_mul_ps(k0, _mm256_set1_ps((float)s0[is + 2]));
                k1 = _mm256_mul_ps(k1, _mm256_set1_ps((float)s0[is + 2]));
                acc0 = _mm256_fmadd_ps(k0, _mm256_loadu_ps(xh + 32 + l0), acc0);
                acc1 = _mm256_fmadd_ps(k1, _mm256_loadu_ps(xh + 32 + l0 + 8), acc1);
                Q6K_WIDEN2(q3, k0, k1);
                k0 = _mm256_mul_ps(k0, _mm256_set1_ps((float)s0[is + 4]));
                k1 = _mm256_mul_ps(k1, _mm256_set1_ps((float)s0[is + 4]));
                acc0 = _mm256_fmadd_ps(k0, _mm256_loadu_ps(xh + 64 + l0), acc0);
                acc1 = _mm256_fmadd_ps(k1, _mm256_loadu_ps(xh + 64 + l0 + 8), acc1);
                Q6K_WIDEN2(q4, k0, k1);
                k0 = _mm256_mul_ps(k0, _mm256_set1_ps((float)s0[is + 6]));
                k1 = _mm256_mul_ps(k1, _mm256_set1_ps((float)s0[is + 6]));
                acc0 = _mm256_fmadd_ps(k0, _mm256_loadu_ps(xh + 96 + l0), acc0);
                acc1 = _mm256_fmadd_ps(k1, _mm256_loadu_ps(xh + 96 + l0 + 8), acc1);
            }
        }
        sum += d * hsum_ps_avx2(_mm256_add_ps(acc0, acc1));
    }
    return sum;
}

/* A group of tokens runs the one-token dot for each token. The weight row
 * stays in the cache between the tokens, so the memory reads one row. */
__attribute__((target("avx2,fma")))
#if GEMMA_X86 && !defined(__AVX512F__)
/* One Q6_K row on NT int8 x rows (i4q_begin), NT a constant 1 or 2. Each
 * quarter of a half block (32 values) is one group of x with two scales
 * (one for each 16): the codes minus 32 (vpsignb moves their sign to x),
 * vpmaddubsw, vpmaddwd with the two scales, and one fma with d times the
 * scale of x. A token alone and in a group has the same operations. */
static inline __attribute__((always_inline)) void q6k_row_q8_n(
    const uint8_t *w, const int8_t *const *xq, const float *const *xs, const int NT, int cols,
    float *sums)
{
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(0x03);
    const __m256i c32 = _mm256_set1_epi8(32);
    __m256 acc[2];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_ps();
    }
    for (int b = 0; b < (cols >> 8); ++b) {
        const uint8_t *blk = w + (size_t)b * 210u;
        const int8_t *sc = (const int8_t *)(blk + 192);
        float d = fp16_to_f32((uint16_t)((uint16_t)blk[208] | ((uint16_t)blk[209] << 8)));
        for (int n = 0; n < 2; ++n) {
            __m256i A = _mm256_loadu_si256((const __m256i *)(blk + 64 * n));
            __m256i B = _mm256_loadu_si256((const __m256i *)(blk + 64 * n + 32));
            __m256i H = _mm256_loadu_si256((const __m256i *)(blk + 128 + 32 * n));
            __m256i q[4];
            q[0] = _mm256_or_si256(_mm256_and_si256(A, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(H, m2), 4));
            q[1] = _mm256_or_si256(_mm256_and_si256(B, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 2), m2), 4));
            q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(A, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 4), m2), 4));
            q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(B, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 6), m2), 4));
            for (int u = 0; u < 4; ++u) {
                __m256i qs = _mm256_sub_epi8(q[u], c32);
                __m256i aq = _mm256_abs_epi8(qs);
                __m256i scv = _mm256_set_m128i(_mm_set1_epi16(sc[8 * n + 2 * u + 1]),
                                               _mm_set1_epi16(sc[8 * n + 2 * u]));
                int g = b * 8 + 4 * n + u;
                for (int t = 0; t < NT; ++t) {
                    __m256i xv = _mm256_loadu_si256((const __m256i *)(xq[t] + (size_t)g * 32));
                    __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aq, _mm256_sign_epi8(xv, qs)), scv);
                    acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * xs[t][g]), acc[t]);
                }
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        sums[t] = hsum256_ps(acc[t]);
    }
}
#endif

static void gemma_q6k_avx2_body(const uint8_t *w, const float *x, float *out,
                    int rows, int cols, int tokens)
{
    size_t row_bytes = (size_t)(cols >> 8) * 210u;
#if GEMMA_X86 && !defined(__AVX512F__)
    if (tokens <= 16 && cols % 256 == 0) {
        /* int8 x (i4q_begin), 2 tokens at a time */
        I4Q_BEGIN(x, (size_t)cols, tokens, cols);
        const int8_t *q[16];
        const float *qs[16];
        int all = 1;
        for (int t = 0; t < tokens && all; ++t) {
            q[t] = i4q_find(x + (size_t)t * cols, cols, &qs[t]);
            all = q[t] != NULL;
        }
        if (all) {
            #pragma omp for schedule(static)
            for (int i = 0; i < rows; ++i) {
                const uint8_t *wi = w + (size_t)i * row_bytes;
                for (int t0 = 0; t0 < tokens; t0 += 2) {
                    float r[2];
                    if (tokens - t0 >= 2) {
                        q6k_row_q8_n(wi, q + t0, qs + t0, 2, cols, r);
                        out[(size_t)t0 * rows + i] = r[0];
                        out[(size_t)(t0 + 1) * rows + i] = r[1];
                    } else {
                        q6k_row_q8_n(wi, q + t0, qs + t0, 1, cols, r);
                        out[(size_t)t0 * rows + i] = r[0];
                    }
                }
            }
            return;
        }
    }
#endif
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        for (int t = 0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_q6k_row_avx2(w + (size_t)i * row_bytes,
                                 x + (size_t)t * (size_t)cols, cols);
        }
    }
}

void rc_q6k_avx2(const uint8_t *w, const float *x, float *out,
                    int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_q6k_avx2_body(w, x, out, rows, cols, tokens);
}

/* Widen 16 signed 8-bit values to one float32 vector. Subtract 32. */
#define Q6K_WIDEN1_512(v, f) do { \
    (f) = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepi8_epi32(v), \
                                              _mm512_set1_epi32(32))); \
} while (0)

__attribute__((target("avx512f,avx512bw,avx512vl")))
static float dot_q6k_row_avx512(const uint8_t *w, const float *x, int cols)
{
    int nb = cols >> 8;
    /* A vector of sums over the blocks (d with one fma), the lanes added
     * once at the end, as dot_q6k_rows_avx512 does for each token. */
    __m512 tot = _mm512_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = w + (size_t)b * 210u;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        float d = fp16_to_f32((uint16_t)((uint16_t)blk[208] | ((uint16_t)blk[209] << 8)));
        const float *xb = x + (size_t)b * 256u;
        __m512 acc = _mm512_setzero_ps();
        for (int h = 0; h < 2; ++h) {
            const uint8_t *q0 = ql + h * 64;
            const uint8_t *g0 = qh + h * 32;
            const int8_t *s0 = sc + h * 8;
            const float *xh = xb + h * 128;
            for (int l0 = 0; l0 < 32; l0 += 16) {
                int is = l0 >> 4;
                __m128i a = _mm_loadu_si128((const __m128i *)(q0 + l0));
                __m128i c = _mm_loadu_si128((const __m128i *)(q0 + 32 + l0));
                __m128i g = _mm_loadu_si128((const __m128i *)(g0 + l0));
                __m128i m0f = _mm_set1_epi8(0x0F);
                __m128i m3 = _mm_set1_epi8(0x03);
                __m128i q1 = _mm_or_si128(_mm_and_si128(a, m0f),
                                          _mm_slli_epi16(_mm_and_si128(g, m3), 4));
                __m128i q2 = _mm_or_si128(_mm_and_si128(c, m0f),
                                          _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(g, 2), m3), 4));
                __m128i q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(a, 4), m0f),
                                          _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(g, 4), m3), 4));
                __m128i q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(c, 4), m0f),
                                          _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(g, 6), m3), 4));
                __m512 k;
                Q6K_WIDEN1_512(q1, k);
                k = _mm512_mul_ps(k, _mm512_set1_ps((float)s0[is + 0]));
                acc = _mm512_fmadd_ps(k, _mm512_loadu_ps(xh + l0), acc);
                Q6K_WIDEN1_512(q2, k);
                k = _mm512_mul_ps(k, _mm512_set1_ps((float)s0[is + 2]));
                acc = _mm512_fmadd_ps(k, _mm512_loadu_ps(xh + 32 + l0), acc);
                Q6K_WIDEN1_512(q3, k);
                k = _mm512_mul_ps(k, _mm512_set1_ps((float)s0[is + 4]));
                acc = _mm512_fmadd_ps(k, _mm512_loadu_ps(xh + 64 + l0), acc);
                Q6K_WIDEN1_512(q4, k);
                k = _mm512_mul_ps(k, _mm512_set1_ps((float)s0[is + 6]));
                acc = _mm512_fmadd_ps(k, _mm512_loadu_ps(xh + 96 + l0), acc);
            }
        }
        tot = _mm512_fmadd_ps(acc, _mm512_set1_ps(d), tot);
    }
    return _mm512_reduce_add_ps(tot);
}

/* The dot of one Q6_K row with up to Q6K_TMAX token rows. The kernel decodes
 * each group of 16 weights one time and uses it for every token. The steps
 * for one token are the steps of dot_q6k_row_avx512, in the same order. Thus
 * the result of each token is the same bit for bit. An MTP verify batch then
 * gives the same logits as a decode step. */
#define Q6K_TMAX 8

__attribute__((target("avx512f,avx512bw,avx512vl"))) static inline __attribute__((always_inline))
void dot_q6k_rows_n(const uint8_t *w, const float *x, int cols, const int tokens, float *sums)
{
    int nb = cols >> 8;
    __m512 tot[Q6K_TMAX];
    for (int t = 0; t < tokens; ++t) {
        tot[t] = _mm512_setzero_ps();
    }
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = w + (size_t)b * 210u;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        float d = fp16_to_f32((uint16_t)((uint16_t)blk[208] | ((uint16_t)blk[209] << 8)));
        __m512 acc[Q6K_TMAX];
        for (int t = 0; t < tokens; ++t) {
            acc[t] = _mm512_setzero_ps();
        }
        for (int h = 0; h < 2; ++h) {
            const uint8_t *q0 = ql + h * 64;
            const uint8_t *g0 = qh + h * 32;
            const int8_t *s0 = sc + h * 8;
            size_t xo = (size_t)b * 256u + (size_t)h * 128u;
            for (int l0 = 0; l0 < 32; l0 += 16) {
                int is = l0 >> 4;
                __m128i a = _mm_loadu_si128((const __m128i *)(q0 + l0));
                __m128i c = _mm_loadu_si128((const __m128i *)(q0 + 32 + l0));
                __m128i g = _mm_loadu_si128((const __m128i *)(g0 + l0));
                __m128i m0f = _mm_set1_epi8(0x0F);
                __m128i m3 = _mm_set1_epi8(0x03);
                __m128i q1 = _mm_or_si128(_mm_and_si128(a, m0f),
                                          _mm_slli_epi16(_mm_and_si128(g, m3), 4));
                __m128i q2 = _mm_or_si128(_mm_and_si128(c, m0f),
                                          _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(g, 2), m3), 4));
                __m128i q3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(a, 4), m0f),
                                          _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(g, 4), m3), 4));
                __m128i q4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(c, 4), m0f),
                                          _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(g, 6), m3), 4));
                __m512 k1, k2, k3, k4;
                Q6K_WIDEN1_512(q1, k1);
                k1 = _mm512_mul_ps(k1, _mm512_set1_ps((float)s0[is + 0]));
                Q6K_WIDEN1_512(q2, k2);
                k2 = _mm512_mul_ps(k2, _mm512_set1_ps((float)s0[is + 2]));
                Q6K_WIDEN1_512(q3, k3);
                k3 = _mm512_mul_ps(k3, _mm512_set1_ps((float)s0[is + 4]));
                Q6K_WIDEN1_512(q4, k4);
                k4 = _mm512_mul_ps(k4, _mm512_set1_ps((float)s0[is + 6]));
                for (int t = 0; t < tokens; ++t) {
                    const float *xh = x + (size_t)t * (size_t)cols + xo;
                    __m512 v = acc[t];
                    v = _mm512_fmadd_ps(k1, _mm512_loadu_ps(xh + l0), v);
                    v = _mm512_fmadd_ps(k2, _mm512_loadu_ps(xh + 32 + l0), v);
                    v = _mm512_fmadd_ps(k3, _mm512_loadu_ps(xh + 64 + l0), v);
                    v = _mm512_fmadd_ps(k4, _mm512_loadu_ps(xh + 96 + l0), v);
                    acc[t] = v;
                }
            }
        }
        __m512 dv = _mm512_set1_ps(d);
        for (int t = 0; t < tokens; ++t) {
            tot[t] = _mm512_fmadd_ps(acc[t], dv, tot[t]);
        }
    }
    for (int t = 0; t < tokens; ++t) {
        sums[t] = _mm512_reduce_add_ps(tot[t]);
    }
}

/* dot_q6k_rows_n with a constant count, so the sums of each token stay in
 * registers. */
__attribute__((target("avx512f,avx512bw,avx512vl")))
static void dot_q6k_rows_avx512(const uint8_t *w, const float *x, int cols,
                                int tokens, float *sums)
{
    switch (tokens) {
    case 1: dot_q6k_rows_n(w, x, cols, 1, sums); break;
    case 2: dot_q6k_rows_n(w, x, cols, 2, sums); break;
    case 3: dot_q6k_rows_n(w, x, cols, 3, sums); break;
    case 4: dot_q6k_rows_n(w, x, cols, 4, sums); break;
    case 5: dot_q6k_rows_n(w, x, cols, 5, sums); break;
    case 6: dot_q6k_rows_n(w, x, cols, 6, sums); break;
    case 7: dot_q6k_rows_n(w, x, cols, 7, sums); break;
    default: dot_q6k_rows_n(w, x, cols, Q6K_TMAX, sums); break;
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
static void gemma_q6k_avx512_body(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    size_t row_bytes = (size_t)(cols >> 8) * 210u;
    if (tokens == 1) {
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            out[i] = dot_q6k_row_avx512(w + (size_t)i * row_bytes, x, cols);
        }
        return;
    }
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        float sums[Q6K_TMAX];
        for (int t0 = 0; t0 < tokens; t0 += Q6K_TMAX) {
            int nt = tokens - t0 < Q6K_TMAX ? tokens - t0 : Q6K_TMAX;
            dot_q6k_rows_avx512(w + (size_t)i * row_bytes,
                                x + (size_t)t0 * (size_t)cols, cols, nt, sums);
            for (int t = 0; t < nt; ++t) {
                out[(size_t)(t0 + t) * (size_t)rows + i] = sums[t];
            }
        }
    }
}

void rc_q6k_avx512(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_q6k_avx512_body(w, x, out, rows, cols, tokens);
}

/* Multiply x by W. W is Q6_K data. Use the AVX-512 kernel or the AVX2 kernel.
 * Both kernels take a group of tokens. The result of each token is the same
 * as the result of a call with that token alone.
 */
void rc_q6k_linear(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    if (gemma_have_avx512()) {
        rc_q6k_avx512(w, x, out, rows, cols, tokens);
    } else {
        rc_q6k_avx2(w, x, out, rows, cols, tokens);
    }
}

#else  /* !GEMMA_X86 */

void rc_q6k_linear(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    rc_q6k_scalar(w, x, out, rows, cols, tokens);
}

#endif  /* GEMMA_X86 */
