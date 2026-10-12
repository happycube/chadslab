/* ---------- elementwise kernels ----------
 * The model calls RMSNorm and GELU for each layer. The arrays are small at a
 * decode step. A NumPy call then costs more than the work. These kernels keep
 * the work in C. The parallel region starts only for a large array. Thus a
 * decode step does not pay for a thread team.
 */

/* The sum of the squares of one row.
 *
 * A plain loop is one chain of additions. The latency of an addition is about
 * 4 cycles, so the loop takes one value for 4 cycles and the multiply units
 * wait. Four accumulators break the chain, and the fused multiply and add
 * then works on four vectors at the same time. The result differs from the
 * left to right sum in the last bits. The check
 * scripts/check_rms_norm.py gives the size of that difference.
 */
static inline float gemma_sum_sq(const float *x, int n)
{
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 64 <= n; k += 64) {
        const __m512 v0 = _mm512_loadu_ps(x + k);
        const __m512 v1 = _mm512_loadu_ps(x + k + 16);
        const __m512 v2 = _mm512_loadu_ps(x + k + 32);
        const __m512 v3 = _mm512_loadu_ps(x + k + 48);
        a0 = _mm512_fmadd_ps(v0, v0, a0);
        a1 = _mm512_fmadd_ps(v1, v1, a1);
        a2 = _mm512_fmadd_ps(v2, v2, a2);
        a3 = _mm512_fmadd_ps(v3, v3, a3);
    }
    for (; k + 16 <= n; k += 16) {
        const __m512 v = _mm512_loadu_ps(x + k);
        a0 = _mm512_fmadd_ps(v, v, a0);
    }
    const __m512 t = _mm512_add_ps(_mm512_add_ps(a0, a1),
                                   _mm512_add_ps(a2, a3));
    float ss = _mm512_reduce_add_ps(t);
    for (; k < n; ++k) {
        ss += x[k] * x[k];
    }
    return ss;
#elif GEMMA_X86
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 32 <= n; k += 32) {
        const __m256 v0 = _mm256_loadu_ps(x + k);
        const __m256 v1 = _mm256_loadu_ps(x + k + 8);
        const __m256 v2 = _mm256_loadu_ps(x + k + 16);
        const __m256 v3 = _mm256_loadu_ps(x + k + 24);
        a0 = _mm256_fmadd_ps(v0, v0, a0);
        a1 = _mm256_fmadd_ps(v1, v1, a1);
        a2 = _mm256_fmadd_ps(v2, v2, a2);
        a3 = _mm256_fmadd_ps(v3, v3, a3);
    }
    for (; k + 8 <= n; k += 8) {
        const __m256 v = _mm256_loadu_ps(x + k);
        a0 = _mm256_fmadd_ps(v, v, a0);
    }
    const __m256 t = _mm256_add_ps(_mm256_add_ps(a0, a1),
                                   _mm256_add_ps(a2, a3));
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(t),
                           _mm256_extractf128_ps(t, 1));
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float ss = _mm_cvtss_f32(s4);
    for (; k < n; ++k) {
        ss += x[k] * x[k];
    }
    return ss;
#else
    float ss = 0.0f;
    for (int k = 0; k < n; ++k) {
        ss += x[k] * x[k];
    }
    return ss;
#endif
}

/* Multiply one row by the scale s and by the weight w. A null w skips the
 * weight. The order of the operations is the order of the scalar form. */
static inline void gemma_scale_row(const float *x, const float *w, float *o,
                                   int n, float s)
{
#if GEMMA_X86 && defined(__AVX512F__)
    const __m512 sv = _mm512_set1_ps(s);
    int k = 0;
    if (w != NULL) {
        for (; k + 16 <= n; k += 16) {
            const __m512 xv = _mm512_loadu_ps(x + k);
            const __m512 wv = _mm512_loadu_ps(w + k);
            _mm512_storeu_ps(o + k, _mm512_mul_ps(_mm512_mul_ps(xv, sv), wv));
        }
    } else {
        for (; k + 16 <= n; k += 16) {
            const __m512 xv = _mm512_loadu_ps(x + k);
            _mm512_storeu_ps(o + k, _mm512_mul_ps(xv, sv));
        }
    }
    if (w != NULL) {
        for (; k < n; ++k) {
            o[k] = x[k] * s * w[k];
        }
    } else {
        for (; k < n; ++k) {
            o[k] = x[k] * s;
        }
    }
#else
    if (w != NULL) {
        for (int k = 0; k < n; ++k) {
            o[k] = x[k] * s * w[k];
        }
    } else {
        for (int k = 0; k < n; ++k) {
            o[k] = x[k] * s;
        }
    }
#endif
}

/* One row of the normalization. */
static inline void gemma_rms_norm_row(const float *xi, const float *w,
                                      float *oi, int cols, float eps)
{
    const float ss = gemma_sum_sq(xi, cols);
    const float s = 1.0f / sqrtf(ss / (float)cols + eps);
    gemma_scale_row(xi, w, oi, cols, s);
}

/* Normalize the last axis of x. Multiply by the weight when it is present. */
/* The body for a caller that is already in a region. Each row belongs to one
 * thread. */
static void gemma_rms_norm_body(const float *x, const float *w, float *out,
                                int rows, int cols, float eps)
{
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        gemma_rms_norm_row(x + (size_t)i * (size_t)cols, w,
                           out + (size_t)i * (size_t)cols, cols, eps);
    }
}

void rc_rms_norm(const float *x, const float *w, float *out,
                    int rows, int cols, float eps)
{
    /* A decode step has one row, and the work of that row is below the cost of
     * the thread team. The small case therefore stays out of the OpenMP
     * runtime. The measurement gives about 2 microseconds for the entry. */
    if (rows < 8) {
        for (int i = 0; i < rows; ++i) {
            gemma_rms_norm_row(x + (size_t)i * (size_t)cols, w,
                               out + (size_t)i * (size_t)cols, cols, eps);
        }
        return;
    }
    #pragma omp parallel
    gemma_rms_norm_body(x, w, out, rows, cols, eps);
}

/* The tanh of 16 float values.
 *
 * The form is tanh(x) = 1 - 2 / (exp(2x) + 1). The code takes the absolute
 * value first and puts the sign back at the end. exp(2z) is 2**t with
 * t = 2 * log2(e) * z, and 2**t is 2**n * 2**f for a whole n and a fraction f
 * in [-0.5, 0.5]. A degree-6 polynomial gives 2**f with an error below 1e-7,
 * which is far below the error of the quantized weights.
 *
 * A scalar tanhf call costs about 20 ns for one value. This form costs about
 * 0.3 ns for one value. The GELU of the E4B model needs 440000 tanh values for
 * each token. The measured cost of the scalar form is 17 ms of a 125 ms token,
 * and the new form costs 1.5 ms. */
#if GEMMA_X86 && defined(__AVX512F__)
static inline __m512 gemma_tanh_ps(__m512 x)
{
    const __m512i sign_bits = _mm512_set1_epi32((int)0x80000000u);
    __m512i xi = _mm512_castps_si512(x);
    __m512i sign = _mm512_and_si512(xi, sign_bits);
    __m512 z = _mm512_castsi512_ps(_mm512_andnot_si512(sign_bits, xi));
    /* Clamp t. Then 2**t stays in range and tanh(z) is 1 to the last bit. */
    __m512 t = _mm512_mul_ps(z, _mm512_set1_ps(2.8853900817779268f));
    t = _mm512_min_ps(t, _mm512_set1_ps(88.0f));
    __m512 n = _mm512_roundscale_ps(t, 0);
    __m512 f = _mm512_sub_ps(t, n);
    __m512 p = _mm512_set1_ps(1.540353039338161e-4f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.3333558146428443e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.618129107628477e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.550410866482158e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.402265069591007e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.931471805599453e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    __m512 e = _mm512_scalef_ps(p, n);
    __m512 r = _mm512_sub_ps(_mm512_set1_ps(1.0f),
                             _mm512_div_ps(_mm512_set1_ps(2.0f),
                                           _mm512_add_ps(e, _mm512_set1_ps(1.0f))));
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(r), sign));
}
#endif

/* The AVX2 build (no AVX-512): gemma_tanh_ps on 8 floats, the same steps
 * (2 ** n in the exponent bits: n is at most 88), so the same bits. The
 * GELU of the dense part, of the experts and the soft cap took libm tanhf
 * there (about 20 ns a value; the GELU of the dense part on one thread). */
#if GEMMA_X86 && !defined(__AVX512F__)
static inline __m256 gemma_tanh_avx2(__m256 x)
{
    const __m256i sign_bits = _mm256_set1_epi32((int)0x80000000u);
    __m256i xi = _mm256_castps_si256(x);
    __m256i sign = _mm256_and_si256(xi, sign_bits);
    __m256 z = _mm256_castsi256_ps(_mm256_andnot_si256(sign_bits, xi));
    __m256 t = _mm256_mul_ps(z, _mm256_set1_ps(2.8853900817779268f));
    t = _mm256_min_ps(t, _mm256_set1_ps(88.0f));
    __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(t, n);
    __m256 p = _mm256_set1_ps(1.540353039338161e-4f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3333558146428443e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.618129107628477e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.550410866482158e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.402265069591007e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.931471805599453e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    __m256i e2 = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    __m256 e = _mm256_mul_ps(p, _mm256_castsi256_ps(e2));
    __m256 r = _mm256_sub_ps(_mm256_set1_ps(1.0f),
                             _mm256_div_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(e, _mm256_set1_ps(1.0f))));
    return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(r), sign));
}

/* o = gelu(g) * u (the tanh GELU) for the first n & ~7 values; return that
 * count (the caller does the rest). o may be g. */
static inline int gemma_gelu_mul_avx2(const float *g, const float *u, float *o, int n)
{
    const __m256 cv = _mm256_set1_ps(0.7978845608028654f), half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f), k3 = _mm256_set1_ps(0.044715f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(g + i);
        __m256 v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
        __m256 th = gemma_tanh_avx2(_mm256_mul_ps(cv, _mm256_fmadd_ps(k3, v3, v)));
        __m256 r = _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, th));
        _mm256_storeu_ps(o + i, _mm256_mul_ps(r, _mm256_loadu_ps(u + i)));
    }
    return i;
}
#define GELU_MUL_VEC(g, u, o, n) gemma_gelu_mul_avx2((g), (u), (o), (n))
#else
#define GELU_MUL_VEC(g, u, o, n) 0
#endif

/* Apply the tanh approximation of GELU to n values. */
/* Limit the size of the logits: out = tanh(x / cap) * cap.
 *
 * The logits are as wide as the vocabulary, 262144 values for the 26B model,
 * so one call costs 0.39 ms in NumPy against about 0.03 ms here. The tanh is
 * monotonic, so this step cannot change the choice of a greedy token. It does
 * change the value, and the sampling path needs the value.
 */
static void gemma_softcap_body(const float *x, float *out, int n, float cap)
{
#if GEMMA_X86 && defined(__AVX512F__)
    const __m512 rcap = _mm512_set1_ps(1.0f / cap);
    const __m512 cv = _mm512_set1_ps(cap);
    const int nv = n & ~15;
    #pragma omp for schedule(static)
    for (int i = 0; i < nv; i += 16) {
        const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(x + i), rcap);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(gemma_tanh_ps(v), cv));
    }
    #pragma omp single
    for (int i = nv; i < n; ++i) {
        out[i] = tanhf(x[i] / cap) * cap;
    }
#elif GEMMA_X86
    const __m256 rcap = _mm256_set1_ps(1.0f / cap);
    const __m256 cv = _mm256_set1_ps(cap);
    const int nv = n & ~7;
    #pragma omp for schedule(static)
    for (int i = 0; i < nv; i += 8) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + i), rcap);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(gemma_tanh_avx2(v), cv));
    }
    #pragma omp single
    for (int i = nv; i < n; ++i) {
        out[i] = tanhf(x[i] / cap) * cap;
    }
#else
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
        out[i] = tanhf(x[i] / cap) * cap;
    }
#endif
}

void rc_softcap(const float *x, float *out, int n, float cap)
{
    #pragma omp parallel if(n >= 65536)
    gemma_softcap_body(x, out, n, cap);
}

static void gemma_gelu_body(const float *x, float *out, int n)
{
    const float c = 0.7978845608028654f;
#if GEMMA_X86 && defined(__AVX512F__)
    const __m512 cv = _mm512_set1_ps(c);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 k3 = _mm512_set1_ps(0.044715f);
    const int nv = n & ~15;
    #pragma omp for schedule(static)
    for (int i = 0; i < nv; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
        __m512 u = _mm512_mul_ps(cv, _mm512_fmadd_ps(k3, v3, v));
        __m512 r = _mm512_mul_ps(_mm512_mul_ps(half, v),
                                 _mm512_add_ps(one, gemma_tanh_ps(u)));
        _mm512_storeu_ps(out + i, r);
    }
    #pragma omp single
    for (int i = nv; i < n; ++i) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
#else
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
#endif
}

void rc_gelu(const float *x, float *out, int n)
{
    #pragma omp parallel if(n >= 65536)
    gemma_gelu_body(x, out, n);
}

/* ---------- float32 kernel for a comparison ---------- */

void rc_f32_linear(const float *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const float *wi = w + (size_t)i * (size_t)cols;
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
            int k = 0;
            for (; k + 7 < cols; k += 8) {
                a0 += xt[k] * wi[k];         a1 += xt[k + 1] * wi[k + 1];
                a2 += xt[k + 2] * wi[k + 2]; a3 += xt[k + 3] * wi[k + 3];
                a4 += xt[k + 4] * wi[k + 4]; a5 += xt[k + 5] * wi[k + 5];
                a6 += xt[k + 6] * wi[k + 6]; a7 += xt[k + 7] * wi[k + 7];
            }
            float acc = sum8(a0, a1, a2, a3, a4, a5, a6, a7);
            for (; k < cols; ++k) {
                acc += xt[k] * wi[k];
            }
            out[(size_t)t * (size_t)rows + i] = acc;
        }
    }
}
