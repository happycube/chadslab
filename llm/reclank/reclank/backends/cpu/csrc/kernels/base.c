/* The x86 code needs AVX2 at least: the scalar tier (no -m flags) takes the
 * plain C branches, on x86 too. */
#if (defined(__x86_64__) || defined(__i386__)) && defined(__AVX2__)
#include <immintrin.h>
#define GEMMA_X86 1
#else
#define GEMMA_X86 0
#endif

/* The dot products of AVX-VNNI (the tier avx2_vnni): vpdpbusd (4 products of
 * unsigned and signed bytes into an int32 lane) and vpdpwssd (2 products of
 * int16). The AVX2 kernels take them in place of vpmaddubsw / vpmaddwd and
 * their adds where the int16 sums cannot saturate, so the results are the
 * same as the avx2 tier. RC_VNNI_EVEX (a test build on a CPU with AVX-512
 * VNNI and VL) takes the EVEX forms, through calls. */
#if GEMMA_X86 && defined(__AVXVNNI__)
#define RC_DPBUSD256(acc, u, s) _mm256_dpbusd_avx_epi32(acc, u, s)
#define RC_DPWSSD256(acc, a, b) _mm256_dpwssd_avx_epi32(acc, a, b)
#elif GEMMA_X86 && defined(RC_VNNI_EVEX)
__attribute__((target("avx512vnni,avx512vl"))) static __m256i rc_dpbusd256(__m256i acc, __m256i u, __m256i s)
{
    return _mm256_dpbusd_epi32(acc, u, s);
}
__attribute__((target("avx512vnni,avx512vl"))) static __m256i rc_dpwssd256(__m256i acc, __m256i a, __m256i b)
{
    return _mm256_dpwssd_epi32(acc, a, b);
}
#define RC_DPBUSD256(acc, u, s) rc_dpbusd256(acc, u, s)
#define RC_DPWSSD256(acc, a, b) rc_dpwssd256(acc, a, b)
#endif

/* Convert one bfloat16 bit pattern to a float32 value. */
static inline float bf16_to_f32(uint16_t bits)
{
    uint32_t u = ((uint32_t)bits) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

#if !GEMMA_X86
/* float32 and float16 bits without F16C, for the scalar tier: the
 * conversions of _cvtss_sh (round to nearest even) and _cvtsh_ss. */
static inline uint16_t rc_f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) {
        return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        man |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half = man >> shift, rem = man & ((1u << shift) - 1u), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) {
            ++half;
        }
        return (uint16_t)(sign | half);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13), rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
        ++h;
    }
    return (uint16_t)h;
}

static inline float rc_f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, x;
    if (exp == 0) {
        if (man == 0) {
            x = sign;
        } else {
            int e = -1;
            do {
                ++e;
                man <<= 1;
            } while (!(man & 0x400u));
            x = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (man << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}
#define _cvtss_sh(x, r) rc_f32_to_f16(x)
#define _cvtsh_ss(h) rc_f16_to_f32(h)
#endif

/* Sum eight accumulators. */
static inline float sum8(float a, float b, float c, float d, float e, float f, float g, float h)
{
    return ((a + b) + (c + d)) + ((e + f) + (g + h));
}

/* ---------- scalar fallback kernels ---------- */

static void gemma_bf16_scalar_body(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint16_t *wi = w + (size_t)i * (size_t)cols;
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k) {
                acc += xt[k] * bf16_to_f32(wi[k]);
            }
            out[(size_t)t * (size_t)rows + i] = acc;
        }
    }
}


void rc_bf16_scalar(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_bf16_scalar_body(w, x, out, rows, cols, tokens);
}

void rc_int8_s8_scalar(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
                          float *out, int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = sw[i];
        for (int t = 0; t < tokens; ++t) {
            const int8_t *xt = qx + (size_t)t * (size_t)cols;
            int32_t dot = 0;
            for (int k = 0; k < cols; ++k) {
                dot += (int32_t)xt[k] * (int32_t)wi[k];
            }
            out[(size_t)t * (size_t)rows + i] = (float)dot * sx[t] * s;
        }
    }
}

#if GEMMA_X86
/* ---------- AVX2 ---------- */

static inline __m256 bf16_to_f32_avx2(const uint16_t *p)
{
    __m128i v = _mm_loadu_si128((const __m128i *)p);
    __m256i u = _mm256_cvtepu16_epi32(v);
    return _mm256_castsi256_ps(_mm256_slli_epi32(u, 16));
}

static inline float hsum_ps_avx2(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline int32_t hsum_epi32_avx2(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

__attribute__((target("avx2,fma")))
static void gemma_bf16_avx2_body(const uint16_t *w, const float *x, float *out,
                     int rows, int cols, int tokens)
{
    int groups = rows / 4;
    #pragma omp for schedule(static)
    for (int g = 0; g < groups; ++g) {
        int i = g * 4;
        const uint16_t *w0 = w + (size_t)(i + 0) * (size_t)cols;
        const uint16_t *w1 = w + (size_t)(i + 1) * (size_t)cols;
        const uint16_t *w2 = w + (size_t)(i + 2) * (size_t)cols;
        const uint16_t *w3 = w + (size_t)(i + 3) * (size_t)cols;
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 7 < cols; k += 8) {
                __m256 xv = _mm256_loadu_ps(xt + k);
                a0 = _mm256_fmadd_ps(xv, bf16_to_f32_avx2(w0 + k), a0);
                a1 = _mm256_fmadd_ps(xv, bf16_to_f32_avx2(w1 + k), a1);
                a2 = _mm256_fmadd_ps(xv, bf16_to_f32_avx2(w2 + k), a2);
                a3 = _mm256_fmadd_ps(xv, bf16_to_f32_avx2(w3 + k), a3);
            }
            float r0 = hsum_ps_avx2(a0), r1 = hsum_ps_avx2(a1);
            float r2 = hsum_ps_avx2(a2), r3 = hsum_ps_avx2(a3);
            for (; k < cols; ++k) {
                float xv = xt[k];
                r0 += xv * bf16_to_f32(w0[k]);
                r1 += xv * bf16_to_f32(w1[k]);
                r2 += xv * bf16_to_f32(w2[k]);
                r3 += xv * bf16_to_f32(w3[k]);
            }
            out[(size_t)t * (size_t)rows + i + 0] = r0;
            out[(size_t)t * (size_t)rows + i + 1] = r1;
            out[(size_t)t * (size_t)rows + i + 2] = r2;
            out[(size_t)t * (size_t)rows + i + 3] = r3;
        }
    }
    #pragma omp single
    for (int i = groups * 4; i < rows; ++i) {
        const uint16_t *wi = w + (size_t)i * (size_t)cols;
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k) {
                acc += xt[k] * bf16_to_f32(wi[k]);
            }
            out[(size_t)t * (size_t)rows + i] = acc;
        }
    }
}

__attribute__((target("avx2,fma")))
void rc_bf16_avx2(const uint16_t *w, const float *x, float *out,
                     int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_bf16_avx2_body(w, x, out, rows, cols, tokens);
}

__attribute__((target("avx2")))
void rc_int8_s8_avx2(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
                        float *out, int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = sw[i];
        for (int t = 0; t < tokens; ++t) {
            const int8_t *xt = qx + (size_t)t * (size_t)cols;
            __m256i acc = _mm256_setzero_si256();
            int k = 0;
            for (; k + 15 < cols; k += 16) {
                __m128i x8 = _mm_loadu_si128((const __m128i *)(xt + k));
                __m128i w8 = _mm_loadu_si128((const __m128i *)(wi + k));
                __m256i x16 = _mm256_cvtepi8_epi16(x8);
                __m256i w16 = _mm256_cvtepi8_epi16(w8);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x16, w16));
            }
            int32_t dot = hsum_epi32_avx2(acc);
            for (; k < cols; ++k) {
                dot += (int32_t)xt[k] * (int32_t)wi[k];
            }
            out[(size_t)t * (size_t)rows + i] = (float)dot * sx[t] * s;
        }
    }
}

/* ---------- AVX-512 ---------- */

__attribute__((target("avx512f,avx512bw,avx512vl")))
static void gemma_bf16_avx512_body(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    int groups = rows / 4;
    #pragma omp for schedule(static)
    for (int g = 0; g < groups; ++g) {
        int i = g * 4;
        const uint16_t *w0 = w + (size_t)(i + 0) * (size_t)cols;
        const uint16_t *w1 = w + (size_t)(i + 1) * (size_t)cols;
        const uint16_t *w2 = w + (size_t)(i + 2) * (size_t)cols;
        const uint16_t *w3 = w + (size_t)(i + 3) * (size_t)cols;
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            int k = 0;
            for (; k + 15 < cols; k += 16) {
                __m512 xv = _mm512_loadu_ps(xt + k);
                a0 = _mm512_fmadd_ps(xv, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(w0 + k))), 16)), a0);
                a1 = _mm512_fmadd_ps(xv, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(w1 + k))), 16)), a1);
                a2 = _mm512_fmadd_ps(xv, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(w2 + k))), 16)), a2);
                a3 = _mm512_fmadd_ps(xv, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(w3 + k))), 16)), a3);
            }
            float r0 = _mm512_reduce_add_ps(a0), r1 = _mm512_reduce_add_ps(a1);
            float r2 = _mm512_reduce_add_ps(a2), r3 = _mm512_reduce_add_ps(a3);
            for (; k < cols; ++k) {
                float xv = xt[k];
                r0 += xv * bf16_to_f32(w0[k]);
                r1 += xv * bf16_to_f32(w1[k]);
                r2 += xv * bf16_to_f32(w2[k]);
                r3 += xv * bf16_to_f32(w3[k]);
            }
            out[(size_t)t * (size_t)rows + i + 0] = r0;
            out[(size_t)t * (size_t)rows + i + 1] = r1;
            out[(size_t)t * (size_t)rows + i + 2] = r2;
            out[(size_t)t * (size_t)rows + i + 3] = r3;
        }
    }
    #pragma omp single
    for (int i = groups * 4; i < rows; ++i) {
        const uint16_t *wi = w + (size_t)i * (size_t)cols;
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            float acc = 0.0f;
            for (int k = 0; k < cols; ++k) {
                acc += xt[k] * bf16_to_f32(wi[k]);
            }
            out[(size_t)t * (size_t)rows + i] = acc;
        }
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
void rc_bf16_avx512(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_bf16_avx512_body(w, x, out, rows, cols, tokens);
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
void rc_int8_s8_avx512(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
                          float *out, int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const int8_t *wi = w + (size_t)i * (size_t)cols;
        const float s = sw[i];
        for (int t = 0; t < tokens; ++t) {
            const int8_t *xt = qx + (size_t)t * (size_t)cols;
            __m512i acc = _mm512_setzero_si512();
            int k = 0;
            for (; k + 31 < cols; k += 32) {
                __m256i x8 = _mm256_loadu_si256((const __m256i *)(xt + k));
                __m256i w8 = _mm256_loadu_si256((const __m256i *)(wi + k));
                __m512i x16 = _mm512_cvtepi8_epi16(x8);
                __m512i w16 = _mm512_cvtepi8_epi16(w8);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(x16, w16));
            }
            int32_t dot = _mm512_reduce_add_epi32(acc);
            for (; k < cols; ++k) {
                dot += (int32_t)xt[k] * (int32_t)wi[k];
            }
            out[(size_t)t * (size_t)rows + i] = (float)dot * sx[t] * s;
        }
    }
}

/* ---------- run time dispatch ---------- */

/* The tier decides (backends/cpu/build.py): the AVX-512 forms of these
 * kernels run in an AVX-512 build only, so an AVX2 build on a CPU with
 * AVX-512 runs the code of an AVX2 CPU (a test for one). */
static int gemma_have_avx512(void)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    return 1;
#else
    return 0;
#endif
}

void rc_bf16_linear(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    if (gemma_have_avx512()) {
        rc_bf16_avx512(w, x, out, rows, cols, tokens);
    } else {
        rc_bf16_avx2(w, x, out, rows, cols, tokens);
    }
}

/* The body for a caller that is already in a region. */
static void gemma_bf16_linear_body(const uint16_t *w, const float *x, float *out,
                                   int rows, int cols, int tokens)
{
    if (gemma_have_avx512()) {
        gemma_bf16_avx512_body(w, x, out, rows, cols, tokens);
    } else {
        gemma_bf16_avx2_body(w, x, out, rows, cols, tokens);
    }
}

void rc_int8_s8(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
                   float *out, int rows, int cols, int tokens)
{
    if (gemma_have_avx512()) {
        rc_int8_s8_avx512(w, sw, qx, sx, out, rows, cols, tokens);
    } else {
        rc_int8_s8_avx2(w, sw, qx, sx, out, rows, cols, tokens);
    }
}
#else
void rc_bf16_linear(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    rc_bf16_scalar(w, x, out, rows, cols, tokens);
}

/* The body for a caller that is already in a region. */
static void gemma_bf16_linear_body(const uint16_t *w, const float *x, float *out,
                                   int rows, int cols, int tokens)
{
    gemma_bf16_scalar_body(w, x, out, rows, cols, tokens);
}

void rc_int8_s8(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
                   float *out, int rows, int cols, int tokens)
{
    rc_int8_s8_scalar(w, sw, qx, sx, out, rows, cols, tokens);
}
#endif
