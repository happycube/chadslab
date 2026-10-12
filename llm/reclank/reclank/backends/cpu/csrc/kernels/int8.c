/* ---------- int8 weights with float32 activations ----------
 * The weight row is int8. The activation row is float32. The kernel converts
 * each int8 value to float32 and uses a fused multiply and add. A scalar loop
 * is too slow. One core does about one multiply for each clock. The kernel
 * then becomes compute bound. Use SIMD instructions for the conversion and
 * the multiply.
 */

#if GEMMA_X86 && defined(__AVX512F__)
/* Return the dot product of one int8 row and one float32 row. Use AVX-512.
 * The loop uses two accumulators. Thus the loop hides the latency of the
 * fused multiply and add.
 */
static inline float dot_i8_f32(const int8_t *w, const float *x, int n)
{
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= n; k += 32) {
        __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(w + k))));
        __m512 w1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(w + k + 16))));
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + k), w0, a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + k + 16), w1, a1);
    }
    float part = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; k + 16 <= n; k += 16) {
        __m512 w0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(w + k))));
        __m512 p0 = _mm512_mul_ps(_mm512_loadu_ps(x + k), w0);
        part += _mm512_reduce_add_ps(p0);
    }
    for (; k < n; ++k) {
        part += x[k] * (float)w[k];
    }
    return part;
}
#elif GEMMA_X86
/* Return the dot product of one int8 row and one float32 row. Use AVX2. */
static inline float dot_i8_f32(const int8_t *w, const float *x, int n)
{
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(w + k))));
        __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(w + k + 8))));
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), w0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8), w1, a1);
    }
    __m256 s = _mm256_add_ps(a0, a1);
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float part = _mm_cvtss_f32(r);
    for (; k < n; ++k) {
        part += x[k] * (float)w[k];
    }
    return part;
}
#else
/* Return the dot product of one int8 row and one float32 row. Use scalar code. */
static inline float dot_i8_f32(const int8_t *w, const float *x, int n)
{
    float part = 0.0f;
    for (int k = 0; k < n; ++k) {
        part += x[k] * (float)w[k];
    }
    return part;
}
#endif

/* ---------- four weight rows for each x block ----------
 * A matrix with many columns, such as mlp.down_proj, has an x row larger than
 * the L1 cache. The one-row loop then reads x from the L2 cache again for each
 * row. This loop reads four weight rows for each x block. Thus the x traffic
 * falls by four times. Use this loop for one token and one scale for each row.
 * The gain is small. The weight stream from the memory controls the time.
 */

#if GEMMA_X86 && defined(__AVX512F__)
static inline void dot4_i8_f32(const int8_t *w, int stride, const float *x, int n,
                               float *r)
{
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        __m512 xv = _mm512_loadu_ps(x + k);
        a0 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(w + k)))), a0);
        a1 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(w + stride + k)))), a1);
        a2 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(w + (size_t)2 * stride + k)))), a2);
        a3 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(w + (size_t)3 * stride + k)))), a3);
    }
    float t0 = _mm512_reduce_add_ps(a0);
    float t1 = _mm512_reduce_add_ps(a1);
    float t2 = _mm512_reduce_add_ps(a2);
    float t3 = _mm512_reduce_add_ps(a3);
    for (; k < n; ++k) {
        float xv = x[k];
        t0 += xv * (float)w[k];
        t1 += xv * (float)w[stride + k];
        t2 += xv * (float)w[(size_t)2 * stride + k];
        t3 += xv * (float)w[(size_t)3 * stride + k];
    }
    r[0] = t0;
    r[1] = t1;
    r[2] = t2;
    r[3] = t3;
}
#elif GEMMA_X86
static inline float hsum256_ps(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline void dot4_i8_f32(const int8_t *w, int stride, const float *x, int n,
                               float *r)
{
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m256 xv = _mm256_loadu_ps(x + k);
        a0 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_loadl_epi64((const __m128i *)(w + k)))), a0);
        a1 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_loadl_epi64((const __m128i *)(w + stride + k)))), a1);
        a2 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_loadl_epi64((const __m128i *)(w + (size_t)2 * stride + k)))), a2);
        a3 = _mm256_fmadd_ps(xv, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_loadl_epi64((const __m128i *)(w + (size_t)3 * stride + k)))), a3);
    }
    float t0 = hsum256_ps(a0);
    float t1 = hsum256_ps(a1);
    float t2 = hsum256_ps(a2);
    float t3 = hsum256_ps(a3);
    for (; k < n; ++k) {
        float xv = x[k];
        t0 += xv * (float)w[k];
        t1 += xv * (float)w[stride + k];
        t2 += xv * (float)w[(size_t)2 * stride + k];
        t3 += xv * (float)w[(size_t)3 * stride + k];
    }
    r[0] = t0;
    r[1] = t1;
    r[2] = t2;
    r[3] = t3;
}
#else
static inline void dot4_i8_f32(const int8_t *w, int stride, const float *x, int n,
                               float *r)
{
    for (int j = 0; j < 4; ++j) {
        r[j] = dot_i8_f32(w + (size_t)j * stride, x, n);
    }
}
#endif

static int gemma_int8_rows4 = 1;

/* Select the four-row loop (1) or the one-row loop (0). Use this for a test. */
void rc_int8_set_rows4(int on)
{
    gemma_int8_rows4 = on ? 1 : 0;
}

void rc_int8_linear(const int8_t *w, const float *scales, const float *x, float *out,
                       int rows, int cols, int tokens, int group)
{
    int groups = cols / group;
    if (gemma_int8_rows4 && tokens == 1 && groups == 1) {
        int blocks = (rows + 3) / 4;
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < blocks; ++b) {
            int i = b * 4;
            int left = rows - i;
            if (left >= 4) {
                float r[4];
                dot4_i8_f32(w + (size_t)i * (size_t)cols, cols, x, cols, r);
                out[i] = r[0] * scales[i];
                out[i + 1] = r[1] * scales[i + 1];
                out[i + 2] = r[2] * scales[i + 2];
                out[i + 3] = r[3] * scales[i + 3];
            } else {
                for (int j = 0; j < left; ++j) {
                    out[i + j] = dot_i8_f32(w + (size_t)(i + j) * (size_t)cols,
                                            x, cols) * scales[i + j];
                }
            }
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float *si = scales + (size_t)i * (size_t)groups;
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float acc = 0.0f;
            int k = 0;
            for (int g = 0; g < groups; ++g) {
                acc += dot_i8_f32(wi + k, xt + k, group) * si[g];
                k += group;
            }
            out[(size_t)t * (size_t)rows + i] = acc;
        }
    }
}

static int gemma_have_avx512(void);

/* ---------- int8 GEMM for a long prompt ----------
 * The one-row kernel reads x again for each row. For a long prompt the x data
 * does not fit in the cache. The kernel then becomes very slow. This kernel
 * keeps the accumulators of MR rows. It reads each x block one time for MR
 * rows. The caller gives x a second time in the transposed shape (cols,
 * tokens). Thus the inner loop reads the tokens with no stride.
 */

#ifndef GEMMA_GEMM_MR
#define GEMMA_GEMM_MR 16
#endif
#ifndef GEMMA_GEMM_TB
#define GEMMA_GEMM_TB 32
#endif
#ifndef GEMMA_GEMM_TOKENS_OUTER
#define GEMMA_GEMM_TOKENS_OUTER 1
#endif
/* The K blocking helps one wide matrix. mlp.down_proj went from 52 GB/s to
 * 114 GB/s at 256 tokens. But the cost of the barriers is more than the win on
 * the full model. The default is off. Set the value with cops.set_gemm_kc for
 * a test. */
#ifndef GEMMA_GEMM_KC
#define GEMMA_GEMM_KC 0
#endif

static inline void gemma_int8_gemm_tile(const int8_t *w, const float *scales,
                                        const float *xt, float *out,
                                        int rows, int cols, int tokens,
                                        int i0, int t0, int k0, int kn, int add)
{
    enum { NV = GEMMA_GEMM_TB / 16 };
#if GEMMA_X86 && defined(__AVX512F__)
    /* Keep the accumulators in the AVX-512 registers. The compiler puts a
     * plain C array on the stack, and the store and the load of each value
     * then controls the time. */
    __m512 acc[GEMMA_GEMM_MR][NV];
    for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
        for (int v = 0; v < NV; ++v) {
            acc[r][v] = _mm512_setzero_ps();
        }
    }
    for (int k = k0; k < k0 + kn; ++k) {
        const float *xk = xt + (size_t)k * (size_t)tokens + t0;
        __m512 xv[NV];
        for (int v = 0; v < NV; ++v) {
            xv[v] = _mm512_loadu_ps(xk + 16 * v);
        }
        for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
            __m512 wv = _mm512_set1_ps((float)w[(size_t)(i0 + r) * (size_t)cols + k]);
            for (int v = 0; v < NV; ++v) {
                acc[r][v] = _mm512_fmadd_ps(wv, xv[v], acc[r][v]);
            }
        }
    }
    float tmp[GEMMA_GEMM_MR][GEMMA_GEMM_TB];
    for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
        for (int v = 0; v < NV; ++v) {
            _mm512_storeu_ps(&tmp[r][16 * v], acc[r][v]);
        }
    }
#else
    float tmp[GEMMA_GEMM_MR][GEMMA_GEMM_TB];
    for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
        #pragma omp simd
        for (int t = 0; t < GEMMA_GEMM_TB; ++t) {
            tmp[r][t] = 0.0f;
        }
    }
    for (int k = 0; k < cols; ++k) {
        float xv[GEMMA_GEMM_TB];
        const float *xk = xt + (size_t)k * (size_t)tokens + t0;
        #pragma omp simd
        for (int t = 0; t < GEMMA_GEMM_TB; ++t) {
            xv[t] = xk[t];
        }
        for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
            float wv = (float)w[(size_t)(i0 + r) * (size_t)cols + k];
            #pragma omp simd
            for (int t = 0; t < GEMMA_GEMM_TB; ++t) {
                tmp[r][t] += wv * xv[t];
            }
        }
    }
#endif
    for (int t = 0; t < GEMMA_GEMM_TB; ++t) {
        float *op = out + (size_t)(t0 + t) * (size_t)rows + i0;
        for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
            float v = tmp[r][t] * scales[i0 + r];
            op[r] = add ? op[r] + v : v;
        }
    }
}

/* ---------- K-vectorized int8 tile ----------
 * The token-vectorized tile converts each weight on its own. That step needs
 * a load, a convert, and a broadcast for every weight. This tile keeps the
 * result of each row and token in a vector. It converts 16 weights with one
 * instruction and uses each converted vector for several tokens. The final
 * sum over the columns is horizontal.
 */

#if GEMMA_X86 && defined(__AVX512F__)
#define GEMMA_KV_MR 4
#define GEMMA_KV_TB 4
#else
/* AVX2 has 8 float lanes and 16 registers. A tile of 4 rows and 2 tokens
 * keeps 8 accumulators, 4 weight vectors, and 2 x vectors in the registers. */
#define GEMMA_KV_MR 4
#define GEMMA_KV_TB 2
#endif

#if GEMMA_X86 && defined(__AVX512F__)
static inline void gemma_int8_gemm_tile_kv(const int8_t *w, const float *scales,
                                           const float *x, float *out,
                                           int rows, int cols, int tokens,
                                           int i0, int t0)
{
    __m512 acc[GEMMA_KV_MR][GEMMA_KV_TB];
    for (int r = 0; r < GEMMA_KV_MR; ++r) {
        for (int t = 0; t < GEMMA_KV_TB; ++t) {
            acc[r][t] = _mm512_setzero_ps();
        }
    }
    for (int k = 0; k + 16 <= cols; k += 16) {
        __m512 wv[GEMMA_KV_MR];
        __m512 xv[GEMMA_KV_TB];
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            wv[r] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128((const __m128i *)(w + (size_t)(i0 + r) * cols + k))));
        }
        for (int t = 0; t < GEMMA_KV_TB; ++t) {
            xv[t] = _mm512_loadu_ps(x + (size_t)(t0 + t) * cols + k);
        }
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            for (int t = 0; t < GEMMA_KV_TB; ++t) {
                acc[r][t] = _mm512_fmadd_ps(wv[r], xv[t], acc[r][t]);
            }
        }
    }
    for (int t = 0; t < GEMMA_KV_TB; ++t) {
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            out[(size_t)(t0 + t) * rows + i0 + r] =
                _mm512_reduce_add_ps(acc[r][t]) * scales[i0 + r];
        }
    }
}
#elif GEMMA_X86
static inline void gemma_int8_gemm_tile_kv(const int8_t *w, const float *scales,
                                           const float *x, float *out,
                                           int rows, int cols, int tokens,
                                           int i0, int t0)
{
    __m256 acc[GEMMA_KV_MR][GEMMA_KV_TB];
    for (int r = 0; r < GEMMA_KV_MR; ++r) {
        for (int t = 0; t < GEMMA_KV_TB; ++t) {
            acc[r][t] = _mm256_setzero_ps();
        }
    }
    int k = 0;
    for (; k + 8 <= cols; k += 8) {
        /* Convert 8 weights with two instructions. Each weight vector serves
         * both tokens, so the convert cost falls by GEMMA_KV_TB. */
        __m256 wv[GEMMA_KV_MR];
        __m256 xv[GEMMA_KV_TB];
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            wv[r] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64((const __m128i *)(w + (size_t)(i0 + r) * cols + k))));
        }
        for (int t = 0; t < GEMMA_KV_TB; ++t) {
            xv[t] = _mm256_loadu_ps(x + (size_t)(t0 + t) * cols + k);
        }
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            for (int t = 0; t < GEMMA_KV_TB; ++t) {
                acc[r][t] = _mm256_fmadd_ps(wv[r], xv[t], acc[r][t]);
            }
        }
    }
    for (int t = 0; t < GEMMA_KV_TB; ++t) {
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            float s = hsum_ps_avx2(acc[r][t]);
            /* The columns that do not fill a vector use the scalar loop. */
            for (int kk = k; kk < cols; ++kk) {
                s += x[(size_t)(t0 + t) * cols + kk] * (float)w[(size_t)(i0 + r) * cols + kk];
            }
            out[(size_t)(t0 + t) * rows + i0 + r] = s * scales[i0 + r];
        }
    }
}
#else
static inline void gemma_int8_gemm_tile_kv(const int8_t *w, const float *scales,
                                           const float *x, float *out,
                                           int rows, int cols, int tokens,
                                           int i0, int t0)
{
    for (int t = 0; t < GEMMA_KV_TB; ++t) {
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            const int8_t *wi = w + (size_t)(i0 + r) * cols;
            const float *xt = x + (size_t)(t0 + t) * cols;
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k) {
                acc += xt[k] * (float)wi[k];
            }
            out[(size_t)(t0 + t) * rows + i0 + r] = acc * scales[i0 + r];
        }
    }
}
#endif

/* ---------- float weight panel ----------
 * The K-vectorized tile converts each weight vector. This path copies the
 * weight rows of one row block to a float panel first. The inner loop then
 * loads float values with no convert. Each thread holds the panel of its own
 * row block in the L2 cache and walks the token blocks. The panel is 4 times
 * the size of the int8 data. Build one panel at a time.
 */

#if GEMMA_X86 && defined(__AVX512F__)
static inline void gemma_int8_gemm_tile_kvf(const float *panel, const float *scales,
                                            const float *x, float *out,
                                            int rows, int cols, int tokens,
                                            int i0, int t0)
{
    __m512 acc[GEMMA_KV_MR][GEMMA_KV_TB];
    for (int r = 0; r < GEMMA_KV_MR; ++r) {
        for (int t = 0; t < GEMMA_KV_TB; ++t) {
            acc[r][t] = _mm512_setzero_ps();
        }
    }
    for (int k = 0; k + 16 <= cols; k += 16) {
        __m512 wv[GEMMA_KV_MR];
        __m512 xv[GEMMA_KV_TB];
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            wv[r] = _mm512_loadu_ps(panel + (size_t)r * cols + k);
        }
        for (int t = 0; t < GEMMA_KV_TB; ++t) {
            xv[t] = _mm512_loadu_ps(x + (size_t)(t0 + t) * cols + k);
        }
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            for (int t = 0; t < GEMMA_KV_TB; ++t) {
                acc[r][t] = _mm512_fmadd_ps(wv[r], xv[t], acc[r][t]);
            }
        }
    }
    for (int t = 0; t < GEMMA_KV_TB; ++t) {
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            out[(size_t)(t0 + t) * rows + i0 + r] =
                _mm512_reduce_add_ps(acc[r][t]) * scales[i0 + r];
        }
    }
}
#else
static inline void gemma_int8_gemm_tile_kvf(const float *panel, const float *scales,
                                            const float *x, float *out,
                                            int rows, int cols, int tokens,
                                            int i0, int t0)
{
    for (int t = 0; t < GEMMA_KV_TB; ++t) {
        for (int r = 0; r < GEMMA_KV_MR; ++r) {
            const float *pr = panel + (size_t)r * cols;
            const float *xt = x + (size_t)(t0 + t) * cols;
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k) {
                acc += pr[k] * xt[k];
            }
            out[(size_t)(t0 + t) * rows + i0 + r] = acc * scales[i0 + r];
        }
    }
}
#endif

static void gemma_int8_gemm_panel(const int8_t *w, const float *scales, const float *x,
                                  float *out, int rows, int cols, int tokens)
{
    int mrb = rows / GEMMA_KV_MR;
    int tbb = tokens / GEMMA_KV_TB;
    #pragma omp parallel
    {
        float *panel = (float *)malloc((size_t)GEMMA_KV_MR * (size_t)cols * sizeof(float));
        if (panel != NULL) {
            #pragma omp for schedule(static)
            for (int bi = 0; bi < mrb; ++bi) {
                int i0 = bi * GEMMA_KV_MR;
                for (int r = 0; r < GEMMA_KV_MR; ++r) {
                    const int8_t *wi = w + (size_t)(i0 + r) * (size_t)cols;
                    float *pr = panel + (size_t)r * (size_t)cols;
                    #pragma omp simd
                    for (int k = 0; k < cols; ++k) {
                        pr[k] = (float)wi[k];
                    }
                }
                for (int bt = 0; bt < tbb; ++bt) {
                    gemma_int8_gemm_tile_kvf(panel, scales, x, out, rows, cols, tokens,
                                             i0, bt * GEMMA_KV_TB);
                }
            }
            free(panel);
        }
    }
    /* The rows and the tokens that do not fill a block use the one-row dot. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        int t0 = (i < mrb * GEMMA_KV_MR) ? tbb * GEMMA_KV_TB : 0;
        for (int t = t0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i8_f32(wi, x + (size_t)t * (size_t)cols, cols) * s;
        }
    }
}

/* ---------- packed weight layout for the prompt ----------
 * The token-vectorized tile needs the weight of each row as a scalar for each
 * column. The scalar load, convert, and broadcast cost about four operations.
 * This path first packs the 16 rows of a row block. The pack puts the 16
 * weights of one column next to each other. Then one instruction converts all
 * 16 values, and a permute broadcasts each value. The pack is one byte for
 * each weight, so it is the same size as the int8 data.
 */

#if GEMMA_X86 && defined(__AVX512F__)
static inline void gemma_int8_gemm_tile_packed(const int8_t *packed,
                                               const float *scales, const float *xt,
                                               float *out, int rows, int cols,
                                               int tokens, int i0, int t0)
{
    enum { NV = GEMMA_GEMM_TB / 16 };
    __m512 acc[GEMMA_GEMM_MR][NV];
    __m512i idx[GEMMA_GEMM_MR];
    for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
        idx[r] = _mm512_set1_epi32(r);
        for (int v = 0; v < NV; ++v) {
            acc[r][v] = _mm512_setzero_ps();
        }
    }
    for (int k = 0; k < cols; ++k) {
        __m512 wv_all = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(packed + (size_t)k * GEMMA_GEMM_MR))));
        __m512 xv[NV];
        for (int v = 0; v < NV; ++v) {
            xv[v] = _mm512_loadu_ps(xt + (size_t)k * (size_t)tokens + t0 + 16 * v);
        }
        for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
            __m512 wr = _mm512_permutexvar_ps(idx[r], wv_all);
            for (int v = 0; v < NV; ++v) {
                acc[r][v] = _mm512_fmadd_ps(wr, xv[v], acc[r][v]);
            }
        }
    }
    float tmp[GEMMA_GEMM_MR][GEMMA_GEMM_TB];
    for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
        for (int v = 0; v < NV; ++v) {
            _mm512_storeu_ps(&tmp[r][16 * v], acc[r][v]);
        }
    }
    for (int t = 0; t < GEMMA_GEMM_TB; ++t) {
        float *op = out + (size_t)(t0 + t) * (size_t)rows + i0;
        for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
            op[r] = tmp[r][t] * scales[i0 + r];
        }
    }
}
#endif

static void gemma_int8_gemm_packed(const int8_t *w, const float *scales,
                                   const float *x, const float *xt,
                                   float *out, int rows, int cols, int tokens)
{
#if GEMMA_X86 && defined(__AVX512F__)
    int mb = rows / GEMMA_GEMM_MR;
    int tb = tokens / GEMMA_GEMM_TB;
    #pragma omp parallel
    {
        int8_t *packed = (int8_t *)malloc((size_t)GEMMA_GEMM_MR * (size_t)cols);
        if (packed != NULL) {
            #pragma omp for schedule(static)
            for (int bi = 0; bi < mb; ++bi) {
                int i0 = bi * GEMMA_GEMM_MR;
                for (int k = 0; k < cols; ++k) {
                    int8_t *pk = packed + (size_t)k * GEMMA_GEMM_MR;
                    for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
                        pk[r] = w[(size_t)(i0 + r) * (size_t)cols + k];
                    }
                }
                for (int bt = 0; bt < tb; ++bt) {
                    gemma_int8_gemm_tile_packed(packed, scales, xt, out, rows, cols, tokens,
                                                i0, bt * GEMMA_GEMM_TB);
                }
            }
            free(packed);
        }
    }
    /* The rows and the tokens that do not fill a block use the one-row dot. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        int t0 = (i < mb * GEMMA_GEMM_MR) ? tb * GEMMA_GEMM_TB : 0;
        for (int t = t0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i8_f32(wi, x + (size_t)t * (size_t)cols, cols) * s;
        }
    }
#else
    /* No AVX-512: use the token-vectorized tile. */
    (void)x;
    (void)xt;
    (void)scales;
    (void)w;
    (void)out;
    (void)rows;
    (void)cols;
    (void)tokens;
#endif
}

/* ---------- packed layout, one copy for the whole model ----------
 * The pack of the last test repeated for each row block. That step forced the
 * row block on the outside, and the x data was read again for each row block.
 * This path uses a packed copy of all the weights. The copy is the transpose
 * of the int8 data, so it is the same size. The token block can then stay on
 * the outside.
 */

#if GEMMA_X86 && defined(__AVX512F__)
static inline void gemma_int8_gemm_tile_pw(const int8_t *pw, int rows, int cols,
                                           const float *scales, const float *xt,
                                           float *out, int tokens, int i0, int t0)
{
    enum { NV = GEMMA_GEMM_TB / 16 };
    __m512 acc[GEMMA_GEMM_MR][NV];
    __m512i idx[GEMMA_GEMM_MR];
    for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
        idx[r] = _mm512_set1_epi32(r);
        for (int v = 0; v < NV; ++v) {
            acc[r][v] = _mm512_setzero_ps();
        }
    }
    for (int k = 0; k < cols; ++k) {
        __m512 wv_all = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(pw + (size_t)k * (size_t)rows + i0))));
        __m512 xv[NV];
        for (int v = 0; v < NV; ++v) {
            xv[v] = _mm512_loadu_ps(xt + (size_t)k * (size_t)tokens + t0 + 16 * v);
        }
        for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
            __m512 wr = _mm512_permutexvar_ps(idx[r], wv_all);
            for (int v = 0; v < NV; ++v) {
                acc[r][v] = _mm512_fmadd_ps(wr, xv[v], acc[r][v]);
            }
        }
    }
    float tmp[GEMMA_GEMM_MR][GEMMA_GEMM_TB];
    for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
        for (int v = 0; v < NV; ++v) {
            _mm512_storeu_ps(&tmp[r][16 * v], acc[r][v]);
        }
    }
    for (int t = 0; t < GEMMA_GEMM_TB; ++t) {
        float *op = out + (size_t)(t0 + t) * (size_t)rows + i0;
        for (int r = 0; r < GEMMA_GEMM_MR; ++r) {
            op[r] = tmp[r][t] * scales[i0 + r];
        }
    }
}
#endif

void rc_int8_gemm_pw(const int8_t *w, const int8_t *pw, const float *scales,
                        const float *x, const float *xt, float *out,
                        int rows, int cols, int tokens)
{
#if GEMMA_X86 && defined(__AVX512F__)
    int mb = rows / GEMMA_GEMM_MR;
    int tb = tokens / GEMMA_GEMM_TB;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int bt = 0; bt < tb; ++bt) {
        for (int bi = 0; bi < mb; ++bi) {
            gemma_int8_gemm_tile_pw(pw, rows, cols, scales, xt, out, tokens,
                                    bi * GEMMA_GEMM_MR, bt * GEMMA_GEMM_TB);
        }
    }
    /* The rows and the tokens that do not fill a block use the one-row dot. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        int t0 = (i < mb * GEMMA_GEMM_MR) ? tb * GEMMA_GEMM_TB : 0;
        for (int t = t0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i8_f32(wi, x + (size_t)t * (size_t)cols, cols) * s;
        }
    }
#else
    (void)w; (void)pw; (void)scales; (void)x; (void)xt; (void)out;
    (void)rows; (void)cols; (void)tokens;
#endif
}

/* ---------- blocked packed layout ----------
 * The plain transpose wasted the cache line, because the columns were rows
 * bytes apart. This layout uses square blocks of 16 rows and 16 columns. One
 * block is 256 bytes and fills four cache lines. Inside a block, column c
 * holds the 16 rows of that column. One load then brings the 16 weights of a
 * column. The data of the block is used in full.
 */

#if GEMMA_X86 && defined(__AVX512F__)
/* The blocked path uses 16 rows and 16 tokens. The x block of 16 tokens then
 * fits in the L2 cache for a wide matrix. A block of 16 rows matches the
 * square pack. */
#define GEMMA_BLK_MR 16
#define GEMMA_BLK_TB 16

static inline void gemma_int8_gemm_tile_blk(const int8_t *pw, int rows, int cols,
                                            const float *scales, const float *xt,
                                            float *out, int tokens, int i0, int t0)
{
    enum { NV = GEMMA_BLK_TB / 16 };
    int ntc = cols / 16;
    int tr = i0 / 16;
    __m512 acc[GEMMA_GEMM_MR][NV];
    for (int r = 0; r < GEMMA_BLK_MR; ++r) {
        for (int v = 0; v < NV; ++v) {
            acc[r][v] = _mm512_setzero_ps();
        }
    }
    for (int tc = 0; tc < ntc; ++tc) {
        const int8_t *blk = pw + ((size_t)tr * ntc + tc) * 256;
        for (int c = 0; c < 16; ++c) {
            int k = tc * 16 + c;
            /* Put the 16 weights of the column in a small buffer. Then the
             * broadcast of each weight is a load, not a table of 16 index
             * vectors. The index vectors do not fit in the registers. */
            float wf[GEMMA_BLK_MR];
            _mm512_storeu_ps(wf, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128((const __m128i *)(blk + 16 * c)))));
            __m512 xv[NV];
            for (int v = 0; v < NV; ++v) {
                xv[v] = _mm512_loadu_ps(xt + (size_t)k * (size_t)tokens + t0 + 16 * v);
            }
            for (int r = 0; r < GEMMA_BLK_MR; ++r) {
                __m512 wr = _mm512_set1_ps(wf[r]);
                for (int v = 0; v < NV; ++v) {
                    acc[r][v] = _mm512_fmadd_ps(wr, xv[v], acc[r][v]);
                }
            }
        }
    }
    float tmp[GEMMA_BLK_MR][GEMMA_BLK_TB];
    for (int r = 0; r < GEMMA_BLK_MR; ++r) {
        for (int v = 0; v < NV; ++v) {
            _mm512_storeu_ps(&tmp[r][16 * v], acc[r][v]);
        }
    }
    for (int t = 0; t < GEMMA_BLK_TB; ++t) {
        float *op = out + (size_t)(t0 + t) * (size_t)rows + i0;
        for (int r = 0; r < GEMMA_BLK_MR; ++r) {
            op[r] = tmp[r][t] * scales[i0 + r];
        }
    }
}
#endif

void rc_int8_gemm_blk(const int8_t *w, const int8_t *pw, const float *scales,
                         const float *x, const float *xt, float *out,
                         int rows, int cols, int tokens)
{
#if GEMMA_X86 && defined(__AVX512F__)
    int mb = rows / GEMMA_BLK_MR;
    int tb = tokens / GEMMA_BLK_TB;
    #pragma omp parallel for schedule(static) collapse(2)
    for (int bt = 0; bt < tb; ++bt) {
        for (int bi = 0; bi < mb; ++bi) {
            gemma_int8_gemm_tile_blk(pw, rows, cols, scales, xt, out, tokens,
                                     bi * GEMMA_BLK_MR, bt * GEMMA_BLK_TB);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        int t0 = (i < mb * GEMMA_BLK_MR) ? tb * GEMMA_BLK_TB : 0;
        for (int t = t0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i8_f32(wi, x + (size_t)t * (size_t)cols, cols) * s;
        }
    }
#else
    (void)w; (void)pw; (void)scales; (void)x; (void)xt; (void)out;
    (void)rows; (void)cols; (void)tokens;
#endif
}

/* ---------- multi-level GEMM for the prompt ----------
 * The K-vectorized tile reads the weight block again for each group of tokens.
 * This GEMM reads each weight one time. It uses three levels:
 *   - a micro kernel that keeps 16 rows and 16 tokens in the registers
 *   - an A panel and a B panel that fit in the L2 cache
 *   - a block over the K dimension
 * The micro kernel is vectorized over the rows. Each lane is one output row.
 * Thus the sum over the columns stays in the lanes, and no horizontal sum is
 * necessary.
 */

#if GEMMA_X86 && defined(__AVX512F__)
#define ML_KC 256
#define ML_MC 128
#define ML_NC 128
#define ML_MR 16
#define ML_NR 16
#else
/* AVX2 has 8 float lanes and a 32 KB L1. A KC of 64 makes the B panel 16 KB,
 * so the B panel stays in the L1 cache. That is faster than a larger KC. */
#define ML_KC 64
#define ML_MC 64
#define ML_NC 64
#define ML_MR 8
#define ML_NR 8
#endif

#if GEMMA_X86
/* One micro tile: ML_MR rows and ML_NR tokens. Each lane is one output row.
 * Thus the sum over the columns stays in the lanes and needs no horizontal
 * add. */
static inline void gemma_ml_micro(const int8_t *a, int lda, int mi,
                                  const float *b, int ldb, int nj, int kc,
                                  const float *scales, float *out, int rows,
                                  int m0, int n0, int add)
{
#if defined(__AVX512F__)
    __m512 acc[ML_NR];
    for (int n = 0; n < ML_NR; ++n) {
        acc[n] = _mm512_setzero_ps();
    }
    for (int k = 0; k < kc; ++k) {
        __m512 av = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            _mm_loadu_si128((const __m128i *)(a + (size_t)k * lda + mi))));
        const float *brow = b + (size_t)k * ldb + nj;
        for (int n = 0; n < ML_NR; ++n) {
            acc[n] = _mm512_fmadd_ps(av, _mm512_set1_ps(brow[n]), acc[n]);
        }
    }
    /* One scale for each row of the output. The second and the later K blocks
     * add to the value of the first K block. */
    __m512 sv = _mm512_loadu_ps(scales + m0 + mi);
    for (int n = 0; n < ML_NR; ++n) {
        float *op = out + (size_t)(n0 + nj + n) * (size_t)rows + (m0 + mi);
        __m512 v = _mm512_mul_ps(acc[n], sv);
        if (add) {
            v = _mm512_add_ps(_mm512_loadu_ps(op), v);
        }
        _mm512_storeu_ps(op, v);
    }
#else
    __m256 acc[ML_NR];
    for (int n = 0; n < ML_NR; ++n) {
        acc[n] = _mm256_setzero_ps();
    }
    for (int k = 0; k < kc; ++k) {
        /* Convert 8 weights with two instructions and use the result for
         * ML_NR tokens. */
        __m256 av = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_loadl_epi64((const __m128i *)(a + (size_t)k * lda + mi))));
        const float *brow = b + (size_t)k * ldb + nj;
        for (int n = 0; n < ML_NR; ++n) {
            acc[n] = _mm256_fmadd_ps(av, _mm256_set1_ps(brow[n]), acc[n]);
        }
    }
    __m256 sv = _mm256_loadu_ps(scales + m0 + mi);
    for (int n = 0; n < ML_NR; ++n) {
        float *op = out + (size_t)(n0 + nj + n) * (size_t)rows + (m0 + mi);
        __m256 v = _mm256_mul_ps(acc[n], sv);
        if (add) {
            v = _mm256_add_ps(_mm256_loadu_ps(op), v);
        }
        _mm256_storeu_ps(op, v);
    }
#endif
}
#endif

void rc_int8_gemm_ml(const int8_t *w, const float *scales, const float *x,
                        const float *xt, float *out, int rows, int cols, int tokens)
{
#if GEMMA_X86
    int mb = rows / ML_MC;
    int nb = tokens / ML_NC;
    int kb = (cols + ML_KC - 1) / ML_KC;
    #pragma omp parallel
    {
        int8_t *abuf = (int8_t *)malloc((size_t)ML_MC * ML_KC);
        float *bbuf = (float *)malloc((size_t)ML_KC * ML_NC * sizeof(float));
        if (abuf != NULL && bbuf != NULL) {
            /* One thread owns one row block. It then runs the K blocks in
             * order. The K blocks add to the same output. The add must not
             * run at the same time as the first store, so the K loop stays
             * inside the row loop. */
            #pragma omp for schedule(static)
            for (int bi = 0; bi < mb; ++bi) {
                int m0 = bi * ML_MC;
                for (int ki = 0; ki < kb; ++ki) {
                    int k0 = ki * ML_KC;
                    int kc = cols - k0 < ML_KC ? cols - k0 : ML_KC;
                    /* Pack the A panel. Read one row at a time, so the read of
                     * the source is sequential. */
                    for (int mm = 0; mm < ML_MC; ++mm) {
                        const int8_t *src = w + (size_t)(m0 + mm) * (size_t)cols + k0;
                        for (int kk = 0; kk < kc; ++kk) {
                            abuf[(size_t)kk * ML_MC + mm] = src[kk];
                        }
                    }
                    for (int ni = 0; ni < nb; ++ni) {
                        int n0 = ni * ML_NC;
                        /* Pack the B panel. One row of x is contiguous. */
                        for (int kk = 0; kk < kc; ++kk) {
                            memcpy(bbuf + (size_t)kk * ML_NC,
                                   xt + (size_t)(k0 + kk) * (size_t)tokens + n0,
                                   ML_NC * sizeof(float));
                        }
                        for (int mi = 0; mi < ML_MC; mi += ML_MR) {
                            for (int njj = 0; njj < ML_NC; njj += ML_NR) {
                                gemma_ml_micro(abuf, ML_MC, mi, bbuf, ML_NC, njj, kc,
                                               scales, out, rows, m0, n0, ki > 0);
                            }
                        }
                    }
                }
            }
        }
        free(abuf);
        free(bbuf);
    }
    /* The rows and the tokens that do not fill a block use the one-row dot. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        int t0 = (i < mb * ML_MC) ? nb * ML_NC : 0;
        for (int t = t0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i8_f32(wi, x + (size_t)t * (size_t)cols, cols) * s;
        }
    }
#else
    (void)w; (void)scales; (void)x; (void)xt; (void)out;
    (void)rows; (void)cols; (void)tokens;
#endif
}

static int gemma_gemm_tokens_outer = GEMMA_GEMM_TOKENS_OUTER;
#if GEMMA_X86
/* The K-vectorized tile is the default on both x86 targets. It converts 16
 * weights with one instruction (8 on AVX2) and uses each converted vector for
 * several tokens. */
static int gemma_gemm_kv = 1;
#else
static int gemma_gemm_kv = 0;
#endif

/* Select the K-vectorized tile (1) or the token-vectorized tile (0). Test only. */
void rc_int8_gemm_set_kv(int on)
{
    gemma_gemm_kv = on ? 1 : 0;
}

static int gemma_gemm_panel = 0;

/* Select the float weight panel (1) or the int8 tile (0). Test only. */
void rc_int8_gemm_set_panel(int on)
{
    gemma_gemm_panel = on ? 1 : 0;
}

#if GEMMA_X86
/* The multi-level GEMM is the default on both x86 targets. It reads each
 * weight one time. A shorter prompt keeps the K-vectorized tile, because the
 * multi-level GEMM needs a full token block. */
static int gemma_gemm_ml = 1;
#else
static int gemma_gemm_ml = 0;
#endif

/* Select the multi-level GEMM for the prompt. Use 0 for the K-vectorized tile. */
void rc_int8_gemm_set_ml(int on)
{
    gemma_gemm_ml = on ? 1 : 0;
}

static int gemma_gemm_packed = 0;

/* Select the packed int8 weight layout for the prompt. Test only. */
void rc_int8_gemm_set_packed(int on)
{
    gemma_gemm_packed = on ? 1 : 0;
}
static int gemma_gemm_kc = GEMMA_GEMM_KC;

/* Select the K chunk size. Use 0 to turn the K blocking off. Test only. */
void rc_int8_gemm_set_kc(int kc)
{
    gemma_gemm_kc = kc;
}

/* Select the token-outer loop order (1) or the row-outer order (0). Test only. */
void rc_int8_gemm_set_tokens_outer(int on)
{
    gemma_gemm_tokens_outer = on ? 1 : 0;
}

void rc_int8_gemm(const int8_t *w, const float *scales, const float *x,
                     const float *xt, float *out, int rows, int cols, int tokens)
{
    int mb = rows / GEMMA_GEMM_MR;
    int tb = tokens / GEMMA_GEMM_TB;
    int done = 0;
    if (gemma_gemm_ml && tokens >= ML_NC) {
        rc_int8_gemm_ml(w, scales, x, xt, out, rows, cols, tokens);
        done = 1;
    }
    if (!done && gemma_gemm_packed) {
        gemma_int8_gemm_packed(w, scales, x, xt, out, rows, cols, tokens);
        done = 1;
    }
    if (!done && gemma_gemm_panel) {
        gemma_int8_gemm_panel(w, scales, x, out, rows, cols, tokens);
        done = 1;
    }
    if (!done && gemma_gemm_kv) {
        /* The K-vectorized tile. The tail rows and tokens use the one-row dot. */
        int mrb = rows / GEMMA_KV_MR;
        int tbb = tokens / GEMMA_KV_TB;
        #pragma omp parallel for schedule(static) collapse(2)
        for (int bt = 0; bt < tbb; ++bt) {
            for (int bi = 0; bi < mrb; ++bi) {
                gemma_int8_gemm_tile_kv(w, scales, x, out, rows, cols, tokens,
                                        bi * GEMMA_KV_MR, bt * GEMMA_KV_TB);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            const int8_t *wi = w + (size_t)i * (size_t)cols;
            const float s = scales[i];
            int t0 = (i < mrb * GEMMA_KV_MR) ? tbb * GEMMA_KV_TB : 0;
            for (int t = t0; t < tokens; ++t) {
                out[(size_t)t * (size_t)rows + i] =
                    dot_i8_f32(wi, x + (size_t)t * (size_t)cols, cols) * s;
            }
        }
        done = 1;
    }
#if GEMMA_GEMM_KC > 0
    /* Split the reduction (K) dimension. One x sub-panel of TB * KC values
     * then fits in the L2 cache. The weight rows pass once for each token
     * block, and the x data is read one time. */
    if (gemma_gemm_kc > 0 && cols > gemma_gemm_kc) {
        /* One parallel region for the matrix. All threads work on the same K
         * chunk, so the x sub-panel is read from the memory one time. A
         * barrier between the K chunks costs less than a new region. */
        #pragma omp parallel
        {
            for (int bt = 0; bt < tb; ++bt) {
                for (int kc = 0; kc < cols; kc += gemma_gemm_kc) {
                    int kn = cols - kc < gemma_gemm_kc ? cols - kc : gemma_gemm_kc;
                    #pragma omp for schedule(static)
                    for (int bi = 0; bi < mb; ++bi) {
                        gemma_int8_gemm_tile(w, scales, xt, out, rows, cols, tokens,
                                             bi * GEMMA_GEMM_MR, bt * GEMMA_GEMM_TB,
                                             kc, kn, kc > 0);
                    }
                }
            }
        }
        done = 1;
    }
#endif
    if (!done && gemma_gemm_tokens_outer) {
        /* Put the token block on the outside. The x block of one token block
         * then stays in the L2 cache while all the weight rows pass. */
        #pragma omp parallel for schedule(static) collapse(2)
        for (int bt = 0; bt < tb; ++bt) {
            for (int bi = 0; bi < mb; ++bi) {
                gemma_int8_gemm_tile(w, scales, xt, out, rows, cols, tokens,
                                     bi * GEMMA_GEMM_MR, bt * GEMMA_GEMM_TB,
                                     0, cols, 0);
            }
        }
    } else if (!done) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int bi = 0; bi < mb; ++bi) {
            for (int bt = 0; bt < tb; ++bt) {
                gemma_int8_gemm_tile(w, scales, xt, out, rows, cols, tokens,
                                     bi * GEMMA_GEMM_MR, bt * GEMMA_GEMM_TB,
                                     0, cols, 0);
            }
        }
    }
    /* The rows and the tokens that do not fill a tile use the one-row dot. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = scales[i];
        int t0 = (i < mb * GEMMA_GEMM_MR) ? tb * GEMMA_GEMM_TB : 0;
        for (int t = t0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i8_f32(wi, x + (size_t)t * (size_t)cols, cols) * s;
        }
    }
}
