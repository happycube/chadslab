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
