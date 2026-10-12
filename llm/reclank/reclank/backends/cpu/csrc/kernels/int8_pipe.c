/* ---------- int8 with a software pipeline ----------
 * The loop reads the weight row and the x row. The loop keeps four
 * accumulators. Thus the adds do not wait for each other. The loop asks the
 * CPU for the data 512 bytes in front. Thus the load of a cache line starts
 * before the work needs it.
 */
#if GEMMA_X86
__attribute__((target("avx2,fma")))
void rc_int8_pf_avx2(const int8_t *w, const float *scales, const float *x, float *out,
                        int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            int k = 0;
            for (; k + 3 < cols; k += 4) {
                _mm_prefetch((const char *)(wi + k + 512), _MM_HINT_T0);
                a0 += xt[k]     * (float)wi[k];
                a1 += xt[k + 1] * (float)wi[k + 1];
                a2 += xt[k + 2] * (float)wi[k + 2];
                a3 += xt[k + 3] * (float)wi[k + 3];
            }
            float acc = (a0 + a1) + (a2 + a3);
            for (; k < cols; ++k) {
                acc += xt[k] * (float)wi[k];
            }
            out[(size_t)t * (size_t)rows + i] = acc * s;
        }
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
void rc_int8_pf_avx512(const int8_t *w, const float *scales, const float *x, float *out,
                          int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            int k = 0;
            for (; k + 3 < cols; k += 4) {
                _mm_prefetch((const char *)(wi + k + 512), _MM_HINT_T0);
                a0 += xt[k]     * (float)wi[k];
                a1 += xt[k + 1] * (float)wi[k + 1];
                a2 += xt[k + 2] * (float)wi[k + 2];
                a3 += xt[k + 3] * (float)wi[k + 3];
            }
            float acc = (a0 + a1) + (a2 + a3);
            for (; k < cols; ++k) {
                acc += xt[k] * (float)wi[k];
            }
            out[(size_t)t * (size_t)rows + i] = acc * s;
        }
    }
}

#endif

#if !GEMMA_X86
/* The int8 rows on float x, plain C (the scalar tier). */
static void gemma_int8_rows_scalar(const int8_t *w, const float *scales, const float *x, float *out,
                                   int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float acc = 0.f;
            for (int k = 0; k < cols; ++k) {
                acc += xt[k] * (float)wi[k];
            }
            out[(size_t)t * (size_t)rows + i] = acc * scales[i];
        }
    }
}
#endif

void rc_int8_pf(const int8_t *w, const float *scales, const float *x, float *out,
                   int rows, int cols, int tokens)
{
#if GEMMA_X86
    if (gemma_have_avx512()) {
        rc_int8_pf_avx512(w, scales, x, out, rows, cols, tokens);
    } else {
        rc_int8_pf_avx2(w, scales, x, out, rows, cols, tokens);
    }
#else
    gemma_int8_rows_scalar(w, scales, x, out, rows, cols, tokens);
#endif
}

/* ---------- int8, two rows in one loop ----------
 * The loop reads the x row one time. Then the loop uses it for two output
 * rows. Thus the count of x loads is one half. The loop also asks for the
 * weight data with a non-temporal hint. The weights are read one time, so the
 * hint stops the weights from filling the cache.
 */
#if GEMMA_X86
__attribute__((target("avx2,fma")))
void rc_int8_pair_avx2(const int8_t *w, const float *scales, const float *x, float *out,
                          int rows, int cols, int tokens)
{
    int pairs = rows / 2;
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < pairs; ++p) {
        int i = p * 2;
        const int8_t *w0 = w + (size_t)i * (size_t)cols;
        const int8_t *w1 = w + (size_t)(i + 1) * (size_t)cols;
        const float s0 = scales[i];
        const float s1 = scales[i + 1];
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 15 < cols; k += 16) {
                _mm_prefetch((const char *)(w0 + k + 512), _MM_HINT_NTA);
                _mm_prefetch((const char *)(w1 + k + 512), _MM_HINT_NTA);
                __m256 x0 = _mm256_loadu_ps(xt + k);
                __m256 x1 = _mm256_loadu_ps(xt + k + 8);
                __m256i z0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w0 + k)));
                __m256i z1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w0 + k + 8)));
                __m256i z2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w1 + k)));
                __m256i z3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w1 + k + 8)));
                a0 = _mm256_fmadd_ps(x0, _mm256_cvtepi32_ps(z0), a0);
                a1 = _mm256_fmadd_ps(x1, _mm256_cvtepi32_ps(z1), a1);
                b0 = _mm256_fmadd_ps(x0, _mm256_cvtepi32_ps(z2), b0);
                b1 = _mm256_fmadd_ps(x1, _mm256_cvtepi32_ps(z3), b1);
            }
            float r0 = (hsum_ps_avx2(a0) + hsum_ps_avx2(a1)) * s0;
            float r1 = (hsum_ps_avx2(b0) + hsum_ps_avx2(b1)) * s1;
            for (; k < cols; ++k) {
                r0 += xt[k] * (float)w0[k] * s0;
                r1 += xt[k] * (float)w1[k] * s1;
            }
            out[(size_t)t * (size_t)rows + i] = r0;
            out[(size_t)t * (size_t)rows + i + 1] = r1;
        }
    }
    for (int i = pairs * 2; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k) acc += xt[k] * (float)wi[k];
            out[(size_t)t * (size_t)rows + i] = acc * s;
        }
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
void rc_int8_pair_avx512(const int8_t *w, const float *scales, const float *x, float *out,
                            int rows, int cols, int tokens)
{
    int pairs = rows / 2;
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < pairs; ++p) {
        int i = p * 2;
        const int8_t *w0 = w + (size_t)i * (size_t)cols;
        const int8_t *w1 = w + (size_t)(i + 1) * (size_t)cols;
        const float s0 = scales[i];
        const float s1 = scales[i + 1];
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
            int k = 0;
            for (; k + 31 < cols; k += 32) {
                _mm_prefetch((const char *)(w0 + k + 512), _MM_HINT_NTA);
                _mm_prefetch((const char *)(w1 + k + 512), _MM_HINT_NTA);
                __m512 x0 = _mm512_loadu_ps(xt + k);
                __m512 x1 = _mm512_loadu_ps(xt + k + 16);
                __m512i z0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(w0 + k)));
                __m512i z1 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(w0 + k + 16)));
                __m512i z2 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(w1 + k)));
                __m512i z3 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(w1 + k + 16)));
                a0 = _mm512_fmadd_ps(x0, _mm512_cvtepi32_ps(z0), a0);
                a1 = _mm512_fmadd_ps(x1, _mm512_cvtepi32_ps(z1), a1);
                b0 = _mm512_fmadd_ps(x0, _mm512_cvtepi32_ps(z2), b0);
                b1 = _mm512_fmadd_ps(x1, _mm512_cvtepi32_ps(z3), b1);
            }
            float r0 = (_mm512_reduce_add_ps(a0) + _mm512_reduce_add_ps(a1)) * s0;
            float r1 = (_mm512_reduce_add_ps(b0) + _mm512_reduce_add_ps(b1)) * s1;
            for (; k < cols; ++k) {
                r0 += xt[k] * (float)w0[k] * s0;
                r1 += xt[k] * (float)w1[k] * s1;
            }
            out[(size_t)t * (size_t)rows + i] = r0;
            out[(size_t)t * (size_t)rows + i + 1] = r1;
        }
    }
    for (int i = pairs * 2; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k) acc += xt[k] * (float)wi[k];
            out[(size_t)t * (size_t)rows + i] = acc * s;
        }
    }
}

#endif

void rc_int8_pair(const int8_t *w, const float *scales, const float *x, float *out,
                     int rows, int cols, int tokens)
{
#if GEMMA_X86
    if (gemma_have_avx512()) {
        rc_int8_pair_avx512(w, scales, x, out, rows, cols, tokens);
    } else {
        rc_int8_pair_avx2(w, scales, x, out, rows, cols, tokens);
    }
#else
    gemma_int8_rows_scalar(w, scales, x, out, rows, cols, tokens);
#endif
}
