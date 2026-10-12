/* ---------- int8 dot product for one token ----------
 * A decode step holds one token. The int8 tile keeps tokens in its lanes, so
 * one token uses one lane of sixteen and the caller must pad the activation
 * buffer. This kernel keeps rows in its lanes and one token in the group.
 * Thus no lane is idle and the activation needs no padding.
 *
 * The instruction vpdpbusd does 32 byte products and adds them to eight int32
 * lanes. That is 8 times the work of one add in the same time. The C test
 * .cache/gemv.c gives 1.3 to 1.5 times the speed of the float kernel at the
 * matrix sizes of the 26B model, and 80 to 90 percent of a pure memory read.
 *
 * I4QG_MR gives the rows of one pass. A block of I4QG_MR rows reads the
 * activation group one time, so the activation load costs 1/I4QG_MR for each
 * row. I4QG_MR is 8 because 8 rows fill the eight int32 lanes of one 256-bit
 * register.
 */
#define I4QG_MR 8

/* One row of one group of the int8 dot product. R is the row in the block. The
 * macro lets the row loop below keep a fixed count, which the compiler
 * unrolls. The count of the rows is not known at compile time in the general
 * case, and the variable loop is 5 to 10 percent slower. */
#define I4QG_ROW(R)                                                          \
    do {                                                                     \
        const uint8_t *b = wb + (size_t)(R) * wstride                        \
                           + (size_t)g * 18 + 2;                             \
        const __m128i raw = _mm_loadu_si128((const __m128i *)b);             \
        const __m128i lo = _mm_and_si128(raw, m4);                           \
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);        \
        const __m256i wv = _mm256_set_m128i(hi, lo);                         \
        __m256i d = _mm256_dpbusd_epi32(_mm256_setzero_si256(), wv, qv);     \
        d = _mm256_sub_epi32(d, corr);                                       \
        const __m256 sc = _mm256_mul_ps(                                     \
            _mm256_set1_ps(sb[(size_t)(R) * (size_t)groups + g]), sxv);      \
        facc[R] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), sc, facc[R]);       \
    } while (0)

/* One row block of the int8 dot product. rb selects the block. */
static void gemma_int4_q8_gemv_block(const uint8_t *w, const float *scales,
                                     const int8_t *qx, const float *sx,
                                     const int32_t *sumx, float *out,
                                     int rows, int cols, int rb)
{
    const int groups = cols / 32;
    const size_t wstride = (size_t)groups * 18;
    const int i0 = rb * I4QG_MR;
    int nrow = rows - i0;
    if (nrow > I4QG_MR) {
        nrow = I4QG_MR;
    }
#if GEMMA_X86 && defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const uint8_t *wb = w + (size_t)i0 * wstride;
    const float *sb = scales + (size_t)i0 * (size_t)groups;
    __m256 facc[I4QG_MR];
    for (int r = 0; r < I4QG_MR; ++r) {
        facc[r] = _mm256_setzero_ps();
    }
    for (int g = 0; g < groups; ++g) {
        /* The q8 group is one 32-byte block. Bytes 0 to 15 hold columns 0 to
         * 15 and bytes 16 to 31 hold columns 16 to 31. That is the order of
         * the low and the high nibbles of the weight block. */
        const __m256i qv = _mm256_loadu_si256(
            (const __m256i *)(qx + (size_t)g * 32));
        const __m256 sxv = _mm256_set1_ps(sx[g]);
        /* vpdpbusd holds the weight as an unsigned byte, so the nibble keeps
         * its value 0 to 15 and the dot product is too large by 8 times the
         * activation sum. Remove that part from lane 0 of the accumulator.
         * The later sum of the lanes removes it one time for the group. The
         * instruction vpdpbssd, which needs no correction, is not in the
         * AVX512-VNNI set. */
        const __m256i corr = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 8 * sumx[g]);
        if (nrow == I4QG_MR) {
            I4QG_ROW(0);
            I4QG_ROW(1);
            I4QG_ROW(2);
            I4QG_ROW(3);
            I4QG_ROW(4);
            I4QG_ROW(5);
            I4QG_ROW(6);
            I4QG_ROW(7);
        } else {
            for (int r = 0; r < nrow; ++r) {
                I4QG_ROW(r);
            }
        }
    }
    for (int r = 0; r < I4QG_MR; ++r) {
        if (r >= nrow) {
            break;
        }
        const __m128 slo = _mm256_castps256_ps128(facc[r]);
        const __m128 shi = _mm256_extractf128_ps(facc[r], 1);
        __m128 s = _mm_add_ps(slo, shi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        out[i0 + r] = _mm_cvtss_f32(s);
    }
#else
    /* A machine without VNNI uses the plain integer multiply. */
    for (int r = 0; r < nrow; ++r) {
        const uint8_t *wi = w + (size_t)(i0 + r) * wstride;
        const float *si = scales + (size_t)(i0 + r) * (size_t)groups;
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            const uint8_t *b = wi + (size_t)g * 18 + 2;
            int32_t dot = 0;
            for (int k = 0; k < 16; ++k) {
                dot += (b[k] & 0x0F) * (int32_t)qx[(size_t)g * 32 + k];
                dot += ((b[k] >> 4) & 0x0F)
                       * (int32_t)qx[(size_t)g * 32 + 16 + k];
            }
            dot -= 8 * sumx[g];
            acc += (float)dot * si[g] * sx[g];
        }
        out[i0 + r] = acc;
    }
#endif
}

#undef I4QG_ROW

/* The full int8 dot product for one token. qx, sx, and sumx hold the
 * quantized activation of one row, as quantize_q8_groups gives it. */
void rc_int4_q8_gemv(const uint8_t *w, const float *scales,
                        const int8_t *qx, const float *sx,
                        const int32_t *sumx, float *out, int rows, int cols)
{
    const int nrb = (rows + I4QG_MR - 1) / I4QG_MR;
    #pragma omp parallel for schedule(static)
    for (int rb = 0; rb < nrb; ++rb) {
        gemma_int4_q8_gemv_block(w, scales, qx, sx, sumx, out, rows, cols, rb);
    }
}

/* Scratch for the quantized activation of one row. The entry points below run
 * on the calling thread of the Python process, so one buffer is sufficient.
 * The buffer holds the widest matrix that the process has seen. */
static int8_t *g_i4qx = NULL;
static float *g_i4qs = NULL;
static int32_t *g_i4qm = NULL;
static int g_i4qcap = 0;

/* Make the scratch hold cols columns. Return 0 when the memory is not there. */
static int gemma_i4q_scratch(int cols)
{
    if (cols <= g_i4qcap) {
        return 1;
    }
    const int groups = cols / 32;
    free(g_i4qx);
    free(g_i4qs);
    free(g_i4qm);
    g_i4qx = (int8_t *)malloc((size_t)cols);
    g_i4qs = (float *)malloc((size_t)groups * sizeof(float));
    g_i4qm = (int32_t *)malloc((size_t)groups * sizeof(int32_t));
    if (g_i4qx == NULL || g_i4qs == NULL || g_i4qm == NULL) {
        free(g_i4qx);
        free(g_i4qs);
        free(g_i4qm);
        g_i4qx = NULL;
        g_i4qs = NULL;
        g_i4qm = NULL;
        g_i4qcap = 0;
        return 0;
    }
    g_i4qcap = cols;
    return 1;
}

/* Quantize one row of x to int8 in the group layout. */
static void gemma_i4q_row(const float *x, int cols)
{
    const int groups = cols / 32;
    for (int g = 0; g < groups; ++g) {
        g_i4qs[g] = gemma_quant_group32(x + (size_t)g * 32,
                                        g_i4qx + (size_t)g * 32, &g_i4qm[g]);
    }
}

/* The int8 dot product for one token, from a float32 activation. The
 * quantization and the dot product stay in one call, so the caller starts one
 * parallel region and pays for one ctypes call. That matters: at the matrix
 * sizes of a decode step the call overhead is larger than the kernel. */
void rc_int4_q8_gemv_x(const uint8_t *w, const float *scales, const float *x,
                          float *out, int rows, int cols)
{
    if (!gemma_i4q_scratch(cols)) {
        return;
    }
    gemma_i4q_row(x, cols);
    rc_int4_q8_gemv(w, scales, g_i4qx, g_i4qs, g_i4qm, out, rows, cols);
}

/* Up to four int4 matrices on the same one-row activation. The quantization
 * runs one time for all of them. One parallel region covers all four. */
void rc_int4_q8_multi4(const uint8_t *w0, const float *s0, float *o0, int r0,
                          const uint8_t *w1, const float *s1, float *o1, int r1,
                          const uint8_t *w2, const float *s2, float *o2, int r2,
                          const uint8_t *w3, const float *s3, float *o3, int r3,
                          const float *x, int cols)
{
    const uint8_t *ws[4] = {w0, w1, w2, w3};
    const float *ss[4] = {s0, s1, s2, s3};
    float *os[4] = {o0, o1, o2, o3};
    const int rs[4] = {r0, r1, r2, r3};
    int off[4];
    int total = 0;
    for (int i = 0; i < 4; ++i) {
        off[i] = total;
        if (rs[i] > 0) {
            total += (rs[i] + I4QG_MR - 1) / I4QG_MR;
        }
    }
    if (total == 0 || !gemma_i4q_scratch(cols)) {
        return;
    }
    gemma_i4q_row(x, cols);
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < total; ++b) {
        int i = 3;
        while (i > 0 && (rs[i] <= 0 || b < off[i])) {
            --i;
        }
        gemma_int4_q8_gemv_block(ws[i], ss[i], g_i4qx, g_i4qs, g_i4qm, os[i],
                                 rs[i], cols, b - off[i]);
    }
}
