/* Multiply x by W for the Gemma 4 NumPy runtime.
 *
 * This file gives these kernels:
 *   1. A bfloat16 GEMV kernel. It reads four output rows in one loop. Thus the
 *      loop loads x one time for four rows.
 *   2. A bfloat16 GEMM kernel for a prompt with many tokens.
 *   3. An int8 kernel for weights with float32 activations. A one-row dot
 *      serves a decode step. A K-vectorized tile and a multi-level GEMM serve
 *      a prompt.
 *   4. An int4 kernel for weights with float32 activations. A four-row dot
 *      serves a decode step. A multi-level GEMM decodes a row block to a
 *      float32 panel and serves a prompt.
 *   5. An integer int8 kernel that quantizes the activations too. It is off by
 *      default.
 *   6. A float32 kernel for a comparison.
 *   7. A Q6_K kernel for the tied output head. It decodes a 210-byte block in
 *      the registers. It reads the weights in place. Thus the load step does
 *      no dequantize of the head.
 *   8. An RMSNorm kernel and a GELU kernel. The model calls these functions
 *      for each layer. A NumPy call on a small array costs more than the work.
 *
 * Each kernel has an AVX-512 version and an AVX2 version. The code selects the
 * AVX-512 version at run time. If the CPU does not give AVX-512, the code uses
 * the AVX2 version. Every target machine gives AVX2.
 *
 * Build this file with:
 *   cc -O3 -mavx2 -mfma -fopenmp -shared -fPIC -o libgemma.so bf16_linear.c
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define GEMMA_X86 1
#else
#define GEMMA_X86 0
#endif

/* Convert one bfloat16 bit pattern to a float32 value. */
static inline float bf16_to_f32(uint16_t bits)
{
    uint32_t u = ((uint32_t)bits) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* Sum eight accumulators. */
static inline float sum8(float a, float b, float c, float d, float e, float f, float g, float h)
{
    return ((a + b) + (c + d)) + ((e + f) + (g + h));
}

/* ---------- scalar fallback kernels ---------- */

void gemma_bf16_scalar(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
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

void gemma_int8_s8_scalar(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
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
void gemma_bf16_avx2(const uint16_t *w, const float *x, float *out,
                     int rows, int cols, int tokens)
{
    int groups = rows / 4;
    #pragma omp parallel for schedule(static)
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

__attribute__((target("avx2")))
void gemma_int8_s8_avx2(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
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
void gemma_bf16_avx512(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    int groups = rows / 4;
    #pragma omp parallel for schedule(static)
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
void gemma_int8_s8_avx512(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
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

static int gemma_have_avx512(void)
{
    static int state = -1;
    if (state < 0) {
        state = (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) ? 1 : 0;
    }
    return state;
}

void gemma_bf16_linear(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    if (gemma_have_avx512()) {
        gemma_bf16_avx512(w, x, out, rows, cols, tokens);
    } else {
        gemma_bf16_avx2(w, x, out, rows, cols, tokens);
    }
}

void gemma_int8_s8(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
                   float *out, int rows, int cols, int tokens)
{
    if (gemma_have_avx512()) {
        gemma_int8_s8_avx512(w, sw, qx, sx, out, rows, cols, tokens);
    } else {
        gemma_int8_s8_avx2(w, sw, qx, sx, out, rows, cols, tokens);
    }
}
#else
void gemma_bf16_linear(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    gemma_bf16_scalar(w, x, out, rows, cols, tokens);
}

void gemma_int8_s8(const int8_t *w, const float *sw, const int8_t *qx, const float *sx,
                   float *out, int rows, int cols, int tokens)
{
    gemma_int8_s8_scalar(w, sw, qx, sx, out, rows, cols, tokens);
}
#endif

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
void gemma_int8_set_rows4(int on)
{
    gemma_int8_rows4 = on ? 1 : 0;
}

void gemma_int8_linear(const int8_t *w, const float *scales, const float *x, float *out,
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

void gemma_int8_gemm_pw(const int8_t *w, const int8_t *pw, const float *scales,
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

void gemma_int8_gemm_blk(const int8_t *w, const int8_t *pw, const float *scales,
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

void gemma_int8_gemm_ml(const int8_t *w, const float *scales, const float *x,
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
void gemma_int8_gemm_set_kv(int on)
{
    gemma_gemm_kv = on ? 1 : 0;
}

static int gemma_gemm_panel = 0;

/* Select the float weight panel (1) or the int8 tile (0). Test only. */
void gemma_int8_gemm_set_panel(int on)
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
void gemma_int8_gemm_set_ml(int on)
{
    gemma_gemm_ml = on ? 1 : 0;
}

static int gemma_gemm_packed = 0;

/* Select the packed int8 weight layout for the prompt. Test only. */
void gemma_int8_gemm_set_packed(int on)
{
    gemma_gemm_packed = on ? 1 : 0;
}
static int gemma_gemm_kc = GEMMA_GEMM_KC;

/* Select the K chunk size. Use 0 to turn the K blocking off. Test only. */
void gemma_int8_gemm_set_kc(int kc)
{
    gemma_gemm_kc = kc;
}

/* Select the token-outer loop order (1) or the row-outer order (0). Test only. */
void gemma_int8_gemm_set_tokens_outer(int on)
{
    gemma_gemm_tokens_outer = on ? 1 : 0;
}

void gemma_int8_gemm(const int8_t *w, const float *scales, const float *x,
                     const float *xt, float *out, int rows, int cols, int tokens)
{
    int mb = rows / GEMMA_GEMM_MR;
    int tb = tokens / GEMMA_GEMM_TB;
    int done = 0;
    if (gemma_gemm_ml && tokens >= ML_NC) {
        gemma_int8_gemm_ml(w, scales, x, xt, out, rows, cols, tokens);
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

/* ---------- 4-bit weights ----------
 * W is packed as unsigned bytes. One byte holds two 4-bit values. A group of
 * 32 values uses 16 bytes. For group g, byte j holds:
 *     low nibble  = value g*32 + j
 *     high nibble = value g*32 + 16 + j
 * A value is a signed 4-bit number (the nibble minus 8). A group has one
 * float32 scale. The group size is fixed at 32.
 *
 * The kernel converts the nibbles to float32 and uses a fused multiply and
 * add. The kernel does not write the values to a buffer first. A scalar loop
 * is too slow. The kernel then becomes compute bound.
 */

/* Turn packed nibbles into signed bytes. A value is a 4-bit two's complement
 * number, so nibble - 8 sign-extends it. Do this in the byte lanes before
 * the widening. The operation is then 2 instructions for 16 values, not 2
 * instructions for each widened vector. */
static inline __m128i i4_sign_bytes(__m128i nib, __m128i b8)
{
    return _mm_sub_epi8(nib, b8);
}

#if GEMMA_X86 && defined(__AVX512F__)
/* Return the dot product of one packed 4-bit row and one float32 row.
 * scales holds one value for each group of 32 columns.
 */
static inline float dot_i4_f32(const uint8_t *w, const float *scales,
                               const float *x, int n)
{
    __m512 acc = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        __m128i b = _mm_loadu_si128((const __m128i *)(w + (size_t)g * 18 + 2));
        __m512 lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            i4_sign_bytes(_mm_and_si128(b, mask), b8)));
        __m512 hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8)));
        __m512 p = _mm512_fmadd_ps(
            _mm512_loadu_ps(x + (size_t)g * 32), lo,
            _mm512_mul_ps(_mm512_loadu_ps(x + (size_t)g * 32 + 16), hi));
        acc = _mm512_fmadd_ps(p, _mm512_set1_ps(scales[g]), acc);
    }
    return _mm512_reduce_add_ps(acc);
}
#elif GEMMA_X86
/* Return the dot product of one packed 4-bit row and one float32 row. Use AVX2.
 * The helper hsum256_ps is above, in the int8 part. */
static inline float dot_i4_f32(const uint8_t *w, const float *scales,
                               const float *x, int n)
{
    __m256 acc = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        __m128i b = _mm_loadu_si128((const __m128i *)(w + (size_t)g * 18 + 2));
        __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
        __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
        __m256 l0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        __m256 l1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        __m256 h0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        __m256 h1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        __m256 p0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + (size_t)g * 32), l0,
                                    _mm256_mul_ps(_mm256_loadu_ps(x + (size_t)g * 32 + 8), l1));
        __m256 p1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + (size_t)g * 32 + 16), h0,
                                    _mm256_mul_ps(_mm256_loadu_ps(x + (size_t)g * 32 + 24), h1));
        acc = _mm256_fmadd_ps(_mm256_add_ps(p0, p1), _mm256_set1_ps(scales[g]), acc);
    }
    return hsum256_ps(acc);
}
#else
/* Return the dot product of one packed 4-bit row and one float32 row. Use scalar code. */
static inline float dot_i4_f32(const uint8_t *w, const float *scales,
                               const float *x, int n)
{
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        int g = k / 32;
        int p = k % 32;
        int byte = w[(size_t)g * 18 + 2 + (p & 15)];
        int nib = (p < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        acc += x[k] * (float)(nib - 8) * scales[g];
    }
    return acc;
}
#endif

/* ---------- four 4-bit rows for each x block ----------
 * The x row uses 4 bytes for each weight of a row. Thus x traffic controls the
 * time for a matrix with many columns. This loop reads four weight rows for
 * each x block. The x traffic falls by four times.
 */

#if GEMMA_X86 && defined(__AVX512F__)
/* Put the 32 values of 16 packed bytes in two float32 vectors. */
static inline void i4_pair(__m128i b, __m128i mask, __m128i b8,
                           __m512 *lo, __m512 *hi)
{
    *lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
        i4_sign_bytes(_mm_and_si128(b, mask), b8)));
    *hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
        i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8)));
}

static inline void dot4_i4_f32(const uint8_t *w, int stride, const float *scales,
                               const float *x, int n, float *r)
{
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        __m512 xlo = _mm512_loadu_ps(x + (size_t)g * 32);
        __m512 xhi = _mm512_loadu_ps(x + (size_t)g * 32 + 16);
        const uint8_t *p = w + (size_t)g * 18 + 2;
        __m128i b0 = _mm_loadu_si128((const __m128i *)(p));
        __m128i b1 = _mm_loadu_si128((const __m128i *)(p + stride));
        __m128i b2 = _mm_loadu_si128((const __m128i *)(p + (size_t)2 * stride));
        __m128i b3 = _mm_loadu_si128((const __m128i *)(p + (size_t)3 * stride));
        __m512 l0, h0, l1, h1, l2, h2, l3, h3;
        i4_pair(b0, mask, b8, &l0, &h0);
        i4_pair(b1, mask, b8, &l1, &h1);
        i4_pair(b2, mask, b8, &l2, &h2);
        i4_pair(b3, mask, b8, &l3, &h3);
        a0 = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l0, _mm512_mul_ps(xhi, h0)),
                             _mm512_set1_ps(scales[g]), a0);
        a1 = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l1, _mm512_mul_ps(xhi, h1)),
                             _mm512_set1_ps(scales[(size_t)groups + g]), a1);
        a2 = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l2, _mm512_mul_ps(xhi, h2)),
                             _mm512_set1_ps(scales[(size_t)2 * groups + g]), a2);
        a3 = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l3, _mm512_mul_ps(xhi, h3)),
                             _mm512_set1_ps(scales[(size_t)3 * groups + g]), a3);
    }
    r[0] = _mm512_reduce_add_ps(a0);
    r[1] = _mm512_reduce_add_ps(a1);
    r[2] = _mm512_reduce_add_ps(a2);
    r[3] = _mm512_reduce_add_ps(a3);
}
#elif GEMMA_X86
/* Put the 32 values of 16 packed bytes in four float32 vectors. */
static inline void i4_pair256(__m128i b, __m128i mask, __m128i b8,
                              __m256 *lo0, __m256 *lo1, __m256 *hi0, __m256 *hi1)
{
    __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
    __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
    *lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
    *lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
    *hi0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
    *hi1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
}

static inline __m256 i4_row256(const uint8_t *p, __m256 x0, __m256 x1,
                               __m256 x2, __m256 x3, __m128i mask, __m128i b8)
{
    __m256 l0, l1, h0, h1;
    i4_pair256(_mm_loadu_si128((const __m128i *)p), mask, b8, &l0, &l1, &h0, &h1);
    return _mm256_add_ps(_mm256_fmadd_ps(x0, l0, _mm256_mul_ps(x1, l1)),
                         _mm256_fmadd_ps(x2, h0, _mm256_mul_ps(x3, h1)));
}

static inline void dot4_i4_f32(const uint8_t *w, int stride, const float *scales,
                               const float *x, int n, float *r)
{
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        __m256 x0 = _mm256_loadu_ps(x + (size_t)g * 32);
        __m256 x1 = _mm256_loadu_ps(x + (size_t)g * 32 + 8);
        __m256 x2 = _mm256_loadu_ps(x + (size_t)g * 32 + 16);
        __m256 x3 = _mm256_loadu_ps(x + (size_t)g * 32 + 24);
        const uint8_t *p = w + (size_t)g * 18 + 2;
        a0 = _mm256_fmadd_ps(i4_row256(p, x0, x1, x2, x3, mask, b8),
                             _mm256_set1_ps(scales[g]), a0);
        a1 = _mm256_fmadd_ps(i4_row256(p + stride, x0, x1, x2, x3, mask, b8),
                             _mm256_set1_ps(scales[(size_t)groups + g]), a1);
        a2 = _mm256_fmadd_ps(i4_row256(p + (size_t)2 * stride, x0, x1, x2, x3, mask, b8),
                             _mm256_set1_ps(scales[(size_t)2 * groups + g]), a2);
        a3 = _mm256_fmadd_ps(i4_row256(p + (size_t)3 * stride, x0, x1, x2, x3, mask, b8),
                             _mm256_set1_ps(scales[(size_t)3 * groups + g]), a3);
    }
    r[0] = hsum256_ps(a0);
    r[1] = hsum256_ps(a1);
    r[2] = hsum256_ps(a2);
    r[3] = hsum256_ps(a3);
}
#else
static inline void dot4_i4_f32(const uint8_t *w, int stride, const float *scales,
                               const float *x, int n, float *r)
{
    int groups = n / 32;
    for (int j = 0; j < 4; ++j) {
        r[j] = dot_i4_f32(w + (size_t)j * stride, scales + (size_t)j * groups, x, n);
    }
}
#endif

/* ---------- int4 tile for a small group of tokens ----------
 * A mixture-of-experts layer gives a small group of tokens to each expert.
 * The group is often smaller than the token block of the multi-level GEMM. The
 * one-row dot then reads x again for each weight row. This tile reads the x
 * block one time for I4T_MR rows. It also decodes each weight group one time
 * for I4T_TB tokens.
 */

#if GEMMA_X86 && defined(__AVX512F__)
#define I4T_MR 16
#define I4T_TB 16
#elif GEMMA_X86
#define I4T_MR 8
#define I4T_TB 8
#else
#define I4T_MR 1
#define I4T_TB 1
#endif

#if GEMMA_X86 && defined(__AVX512F__)
static inline void gemma_int4_gemm_tile(const uint8_t *w, const float *scales,
                                        const float *xt, float *out,
                                        int rows, int cols, int tokens,
                                        int i0, int t0)
{
    __m512 acc[I4T_MR];
    for (int r = 0; r < I4T_MR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    const int groups = cols / 32;
    const int stride = (cols / 32) * 18;
    /* Mask the token lanes that pass the end of the group. */
    int ntok = tokens - t0;
    if (ntok > I4T_TB) {
        ntok = I4T_TB;
    }
    const __mmask16 km = ntok >= I4T_TB ? (__mmask16)0xFFFF : (__mmask16)((1u << ntok) - 1);
    for (int g = 0; g < groups; ++g) {
        float wf[I4T_MR][32];
        for (int r = 0; r < I4T_MR; ++r) {
            const uint8_t *p = w + (size_t)(i0 + r) * stride + (size_t)g * 18 + 2;
            const float sc = scales[(size_t)(i0 + r) * groups + g];
            __m128i b = _mm_loadu_si128((const __m128i *)p);
            __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
            __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
            _mm512_storeu_ps(wf[r] + 0, _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), _mm512_set1_ps(sc)));
            _mm512_storeu_ps(wf[r] + 16, _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), _mm512_set1_ps(sc)));
        }
        for (int k = 0; k < 32; ++k) {
            __m512 xv = _mm512_maskz_loadu_ps(km, xt + (size_t)(g * 32 + k) * tokens + t0);
            for (int r = 0; r < I4T_MR; ++r) {
                acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(wf[r][k]), xv, acc[r]);
            }
        }
    }
    float tmp[I4T_TB];
    for (int r = 0; r < I4T_MR; ++r) {
        _mm512_storeu_ps(tmp, acc[r]);
        for (int t = 0; t < ntok; ++t) {
            out[(size_t)(t0 + t) * rows + i0 + r] = tmp[t];
        }
    }
}
#elif GEMMA_X86
static inline void gemma_int4_gemm_tile(const uint8_t *w, const float *scales,
                                        const float *xt, float *out,
                                        int rows, int cols, int tokens,
                                        int i0, int t0)
{
    __m256 acc[I4T_MR];
    for (int r = 0; r < I4T_MR; ++r) {
        acc[r] = _mm256_setzero_ps();
    }
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    const int groups = cols / 32;
    const int stride = (cols / 32) * 18;
    for (int g = 0; g < groups; ++g) {
        float wf[I4T_MR][32];
        for (int r = 0; r < I4T_MR; ++r) {
            const uint8_t *p = w + (size_t)(i0 + r) * stride + (size_t)g * 18 + 2;
            const float sc = scales[(size_t)(i0 + r) * groups + g];
            __m128i b = _mm_loadu_si128((const __m128i *)p);
            __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
            __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
            __m256 scv = _mm256_set1_ps(sc);
            _mm256_storeu_ps(wf[r] + 0, _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), scv));
            _mm256_storeu_ps(wf[r] + 8, _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), scv));
            _mm256_storeu_ps(wf[r] + 16, _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), scv));
            _mm256_storeu_ps(wf[r] + 24, _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), scv));
        }
        for (int k = 0; k < 32; ++k) {
            __m256 xv = _mm256_loadu_ps(xt + (size_t)(g * 32 + k) * tokens + t0);
            for (int r = 0; r < I4T_MR; ++r) {
                acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(wf[r][k]), xv, acc[r]);
            }
        }
    }
    float tmp[I4T_TB];
    for (int r = 0; r < I4T_MR; ++r) {
        _mm256_storeu_ps(tmp, acc[r]);
        for (int t = 0; t < I4T_TB; ++t) {
            out[(size_t)(t0 + t) * rows + i0 + r] = tmp[t];
        }
    }
}
#else
static inline void gemma_int4_gemm_tile(const uint8_t *w, const float *scales,
                                        const float *xt, float *out,
                                        int rows, int cols, int tokens,
                                        int i0, int t0)
{
    (void)w; (void)scales; (void)xt; (void)out; (void)rows; (void)cols;
    (void)tokens; (void)i0; (void)t0;
}
#endif

void gemma_int4_gemm_tile_run(const uint8_t *w, const float *scales,
                              const float *x, const float *xt, float *out,
                              int rows, int cols, int tokens)
{
    const int stride = (cols / 32) * 18;
    const int groups = cols / 32;
    const int mr = rows / I4T_MR * I4T_MR;
#if GEMMA_X86 && defined(__AVX512F__)
    /* The tile masks the token lanes past the end of the group. Thus it
     * handles every token count. */
    const int tb = tokens;
#else
    const int tb = tokens / I4T_TB * I4T_TB;
#endif
#if GEMMA_X86
    if (tb > 0 && mr > 0) {
        const int nt = (tb + I4T_TB - 1) / I4T_TB;
        #pragma omp parallel for schedule(static) collapse(2)
        for (int bi = 0; bi < mr / I4T_MR; ++bi) {
            for (int bt = 0; bt < nt; ++bt) {
                gemma_int4_gemm_tile(w, scales, xt, out, rows, cols, tokens,
                                     bi * I4T_MR, bt * I4T_TB);
            }
        }
    }
#endif
    /* The rows that do not fill a tile and the token tail use the four-row
     * dot. */
    const int blocks = (rows + 3) / 4;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < blocks; ++b) {
        const int i = b * 4;
        const int nrow = rows - i < 4 ? rows - i : 4;
        for (int t = 0; t < tokens; ++t) {
            if (i < mr && t < tb) {
                continue;
            }
            if (nrow == 4) {
                float r[4];
                dot4_i4_f32(w + (size_t)i * stride, stride, scales + (size_t)i * groups,
                            x + (size_t)t * cols, cols, r);
                out[(size_t)t * rows + i + 0] = r[0];
                out[(size_t)t * rows + i + 1] = r[1];
                out[(size_t)t * rows + i + 2] = r[2];
                out[(size_t)t * rows + i + 3] = r[3];
            } else {
                for (int j = 0; j < nrow; ++j) {
                    out[(size_t)t * rows + i + j] =
                        dot_i4_f32(w + (size_t)(i + j) * stride,
                                   scales + (size_t)(i + j) * groups,
                                   x + (size_t)t * cols, cols);
                }
            }
        }
    }
}

static int gemma_int4_rows4 = 1;

/* Select the four-row loop (1) or the one-row loop (0). Use this for a test. */
void gemma_int4_set_rows4(int on)
{
    gemma_int4_rows4 = on ? 1 : 0;
}

void gemma_int4_linear(const uint8_t *w, const float *scales, const float *x, float *out,
                       int rows, int cols, int tokens, int group)
{
    /* The fast dot uses a group of 32 values. ops.linear_int4 sends only that
     * group size. */
    (void)group;
    int groups = cols / 32;
    int stride = (cols / 32) * 18;
    if (gemma_int4_rows4 && tokens == 1) {
        int blocks = (rows + 3) / 4;
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < blocks; ++b) {
            int i = b * 4;
            int left = rows - i;
            if (left >= 4) {
                float r[4];
                dot4_i4_f32(w + (size_t)i * (size_t)stride, stride,
                            scales + (size_t)i * (size_t)groups, x, cols, r);
                out[i] = r[0];
                out[i + 1] = r[1];
                out[i + 2] = r[2];
                out[i + 3] = r[3];
            } else {
                for (int j = 0; j < left; ++j) {
                    out[i + j] = dot_i4_f32(w + (size_t)(i + j) * (size_t)stride,
                                            scales + (size_t)(i + j) * (size_t)groups,
                                            x, cols);
                }
            }
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint8_t *wi = w + (size_t)i * (size_t)stride;
        const float *si = scales + (size_t)i * (size_t)groups;
        for (int t = 0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i4_f32(wi, si, x + (size_t)t * (size_t)cols, cols);
        }
    }
}

/* ---------- int4 mixture of experts ----------
 * A mixture-of-experts layer selects a small set of experts for each token.
 * A one-row call for each expert then starts one OpenMP region for each
 * expert. The thread team is small when the expert has few rows, and the
 * main thread does the region work many times.
 *
 * This kernel takes the list of selected experts. The parallel loop covers
 * the full work of all the experts at once. Thus the thread team is large,
 * and the caller starts one region for the whole layer.
 *
 * W holds one matrix for each expert in the Q4_0 block layout. scales holds
 * one float32 scale for each group of 32 columns. ids gives the matrix index
 * for each job. Job j computes
 *     out[j * rows + r] = sum_k W[ids[j]][r][k] * x[j * xstride + k].
 * An xstride of 0 gives the same x to every job. That is a decode step.
 */
void gemma_int4_moe_gemv(const uint8_t *w, const float *scales,
                         const float *x, const int *ids, int jobs,
                         float *out, int rows, int cols, int xstride)
{
    int groups = cols / 32;
    size_t stride = (size_t)groups * 18;
    size_t expert_bytes = (size_t)rows * stride;
    size_t expert_scales = (size_t)rows * (size_t)groups;
    int blocks = (rows + 3) / 4;
    long total = (long)jobs * (long)blocks;
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < total; ++t) {
        int j = (int)(t / blocks);
        int i = (int)(t % blocks) * 4;
        int e = ids[j];
        const uint8_t *wj = w + (size_t)e * expert_bytes;
        const float *sj = scales + (size_t)e * expert_scales;
        const float *xj = x + (size_t)j * (size_t)xstride;
        float *oj = out + (size_t)j * (size_t)rows;
        int left = rows - i;
        if (left >= 4) {
            float r[4];
            dot4_i4_f32(wj + (size_t)i * stride, (int)stride,
                        sj + (size_t)i * (size_t)groups, xj, cols, r);
            oj[i] = r[0];
            oj[i + 1] = r[1];
            oj[i + 2] = r[2];
            oj[i + 3] = r[3];
        } else {
            for (int q = i; q < rows; ++q) {
                oj[q] = dot_i4_f32(wj + (size_t)q * stride,
                                   sj + (size_t)q * (size_t)groups, xj, cols);
            }
        }
    }
}

/* ---------- int4 GEMM for a long prompt ----------
 * The one-row dot decodes the packed weights again for each token. This GEMM
 * decodes a row block to a float32 A panel one time and then uses the panel
 * for every token block. The decode cost then falls by the number of token
 * blocks. The A panel holds the group scale, so the micro kernel is a plain
 * float32 multiply and add.
 */

#if GEMMA_X86 && defined(__AVX512F__)
#define I4_MC 64
#define I4_KC 128
#define I4_NC 64
#define I4_MR 16
#define I4_NR 16
#else
/* AVX2 has 8 float lanes. The A panel and the B panel are float32, so keep
 * them small enough for the L1 cache. */
#define I4_MC 64
#define I4_KC 64
#define I4_NC 32
#define I4_MR 8
#define I4_NR 8
#endif

/* Decode ng groups (32 columns each) of one packed 4-bit row to float32. The
 * function applies the group scale. The A panel is column-major: the values of
 * one column are contiguous with a stride of lda. The micro kernel then reads
 * the rows of one column as one vector. */
static inline void i4_decode_row(const uint8_t *w, const float *scales,
                                 int ng, int lda, float *dst)
{
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    for (int g = 0; g < ng; ++g) {
        __m128i b = _mm_loadu_si128((const __m128i *)(w + (size_t)g * 18 + 2));
        float sc = scales[g];
        __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
        __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
        float buf[32];
#if defined(__AVX512F__)
        _mm512_storeu_ps(buf + 0, _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), _mm512_set1_ps(sc)));
        _mm512_storeu_ps(buf + 16, _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), _mm512_set1_ps(sc)));
#else
        _mm256_storeu_ps(buf + 0, _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_set1_ps(sc)));
        _mm256_storeu_ps(buf + 8, _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
            _mm256_set1_ps(sc)));
        _mm256_storeu_ps(buf + 16, _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), _mm256_set1_ps(sc)));
        _mm256_storeu_ps(buf + 24, _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
            _mm256_set1_ps(sc)));
#endif
        float *col = dst + (size_t)g * 32 * lda;
        for (int j = 0; j < 32; ++j) {
            col[(size_t)j * lda] = buf[j];
        }
    }
}

#if GEMMA_X86
/* One micro tile for an already decoded float32 A panel. */
static inline void gemma_ml_micro_f32(const float *a, int lda, int mi,
                                      const float *b, int ldb, int nj, int kc,
                                      float *out, int rows, int m0, int n0, int add)
{
#if defined(__AVX512F__)
    __m512 acc[I4_NR];
    for (int n = 0; n < I4_NR; ++n) {
        acc[n] = _mm512_setzero_ps();
    }
    for (int k = 0; k < kc; ++k) {
        __m512 av = _mm512_loadu_ps(a + (size_t)k * lda + mi);
        const float *brow = b + (size_t)k * ldb + nj;
        for (int n = 0; n < I4_NR; ++n) {
            acc[n] = _mm512_fmadd_ps(av, _mm512_set1_ps(brow[n]), acc[n]);
        }
    }
    for (int n = 0; n < I4_NR; ++n) {
        float *op = out + (size_t)(n0 + nj + n) * (size_t)rows + (m0 + mi);
        __m512 v = acc[n];
        if (add) {
            v = _mm512_add_ps(_mm512_loadu_ps(op), v);
        }
        _mm512_storeu_ps(op, v);
    }
#else
    __m256 acc[I4_NR];
    for (int n = 0; n < I4_NR; ++n) {
        acc[n] = _mm256_setzero_ps();
    }
    for (int k = 0; k < kc; ++k) {
        __m256 av = _mm256_loadu_ps(a + (size_t)k * lda + mi);
        const float *brow = b + (size_t)k * ldb + nj;
        for (int n = 0; n < I4_NR; ++n) {
            acc[n] = _mm256_fmadd_ps(av, _mm256_set1_ps(brow[n]), acc[n]);
        }
    }
    for (int n = 0; n < I4_NR; ++n) {
        float *op = out + (size_t)(n0 + nj + n) * (size_t)rows + (m0 + mi);
        __m256 v = acc[n];
        if (add) {
            v = _mm256_add_ps(_mm256_loadu_ps(op), v);
        }
        _mm256_storeu_ps(op, v);
    }
#endif
}
#endif

void gemma_int4_gemm(const uint8_t *w, const float *scales, const float *x,
                     const float *xt, float *out, int rows, int cols, int tokens)
{
#if GEMMA_X86
    int mb = rows / I4_MC;
    int nb = tokens / I4_NC;
    int kb = (cols + I4_KC - 1) / I4_KC;
    #pragma omp parallel
    {
        float *abuf = (float *)malloc((size_t)I4_MC * I4_KC * sizeof(float));
        float *bbuf = (float *)malloc((size_t)I4_KC * I4_NC * sizeof(float));
        if (abuf != NULL && bbuf != NULL) {
            #pragma omp for schedule(static)
            for (int bi = 0; bi < mb; ++bi) {
                int m0 = bi * I4_MC;
                for (int ki = 0; ki < kb; ++ki) {
                    int k0 = ki * I4_KC;
                    int kc = cols - k0 < I4_KC ? cols - k0 : I4_KC;
                    int kfast = kc & ~31;
                    int g0 = k0 / 32;
                    /* Decode the A panel one time for all the token blocks. */
                    for (int mm = 0; mm < I4_MC; ++mm) {
                        const uint8_t *wi = w + (size_t)(m0 + mm) * ((size_t)(cols / 32) * 18);
                        const float *si = scales + (size_t)(m0 + mm) * (size_t)(cols / 32);
                        i4_decode_row(wi + (size_t)g0 * 18, si + g0, kfast / 32,
                                      I4_MC, abuf + mm);
                    }
                    /* The columns that do not fill a group of 32 use scalar code. */
                    for (int kk = kfast; kk < kc; ++kk) {
                        int g = (k0 + kk) / 32;
                        int p = (k0 + kk) % 32;
                        for (int mm = 0; mm < I4_MC; ++mm) {
                            const uint8_t *wi = w + (size_t)(m0 + mm) * ((size_t)(cols / 32) * 18);
                            const float *si = scales + (size_t)(m0 + mm) * (size_t)(cols / 32);
                            int byte = wi[(size_t)g * 18 + 2 + (p & 15)];
                            int nib = (p < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                            abuf[(size_t)kk * I4_MC + mm] = (float)(nib - 8) * si[g];
                        }
                    }
                    for (int ni = 0; ni < nb; ++ni) {
                        int n0 = ni * I4_NC;
                        for (int kk = 0; kk < kc; ++kk) {
                            memcpy(bbuf + (size_t)kk * I4_NC,
                                   xt + (size_t)(k0 + kk) * (size_t)tokens + n0,
                                   I4_NC * sizeof(float));
                        }
                        for (int mi = 0; mi < I4_MC; mi += I4_MR) {
                            for (int njj = 0; njj < I4_NC; njj += I4_NR) {
                                gemma_ml_micro_f32(abuf, I4_MC, mi, bbuf, I4_NC, njj,
                                                   kc, out, rows, m0, n0, ki > 0);
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
        const uint8_t *wi = w + (size_t)i * ((size_t)(cols / 32) * 18);
        const float *si = scales + (size_t)i * (size_t)(cols / 32);
        int t0 = (i < mb * I4_MC) ? nb * I4_NC : 0;
        for (int t = t0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i4_f32(wi, si, x + (size_t)t * (size_t)cols, cols);
        }
    }
#else
    (void)w; (void)scales; (void)x; (void)xt; (void)out;
    (void)rows; (void)cols; (void)tokens;
#endif
}

/* ---------- bfloat16 GEMM for a long prompt ---------- */

void gemma_bf16_gemm(const uint16_t *w, const float *xt, float *out,
                     int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint16_t *wi = w + (size_t)i * (size_t)cols;
        float acc[1024];
        int limit = tokens < 1024 ? tokens : 1024;
        for (int t = 0; t < limit; ++t) {
            acc[t] = 0.0f;
        }
        for (int k = 0; k < cols; ++k) {
            float wv = bf16_to_f32(wi[k]);
            const float *xk = xt + (size_t)k * (size_t)tokens;
            #pragma omp simd
            for (int t = 0; t < limit; ++t) {
                acc[t] += xk[t] * wv;
            }
        }
        for (int t = 0; t < limit; ++t) {
            out[(size_t)t * (size_t)rows + i] = acc[t];
        }
    }
}

/* ---------- int8 with a software pipeline ----------
 * The loop reads the weight row and the x row. The loop keeps four
 * accumulators. Thus the adds do not wait for each other. The loop asks the
 * CPU for the data 512 bytes in front. Thus the load of a cache line starts
 * before the work needs it.
 */
__attribute__((target("avx2,fma")))
void gemma_int8_pf_avx2(const int8_t *w, const float *scales, const float *x, float *out,
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
void gemma_int8_pf_avx512(const int8_t *w, const float *scales, const float *x, float *out,
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

void gemma_int8_pf(const int8_t *w, const float *scales, const float *x, float *out,
                   int rows, int cols, int tokens)
{
    if (gemma_have_avx512()) {
        gemma_int8_pf_avx512(w, scales, x, out, rows, cols, tokens);
    } else {
        gemma_int8_pf_avx2(w, scales, x, out, rows, cols, tokens);
    }
}

/* ---------- int8, two rows in one loop ----------
 * The loop reads the x row one time. Then the loop uses it for two output
 * rows. Thus the count of x loads is one half. The loop also asks for the
 * weight data with a non-temporal hint. The weights are read one time, so the
 * hint stops the weights from filling the cache.
 */
__attribute__((target("avx2,fma")))
void gemma_int8_pair_avx2(const int8_t *w, const float *scales, const float *x, float *out,
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
void gemma_int8_pair_avx512(const int8_t *w, const float *scales, const float *x, float *out,
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

void gemma_int8_pair(const int8_t *w, const float *scales, const float *x, float *out,
                     int rows, int cols, int tokens)
{
    if (gemma_have_avx512()) {
        gemma_int8_pair_avx512(w, scales, x, out, rows, cols, tokens);
    } else {
        gemma_int8_pair_avx2(w, scales, x, out, rows, cols, tokens);
    }
}

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
void gemma_q6k_scalar(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    size_t row_bytes = (size_t)(cols >> 8) * 210u;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint8_t *wi = w + (size_t)i * row_bytes;
        for (int t = 0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_q6k_row_scalar(wi, x + (size_t)t * (size_t)cols, cols);
        }
    }
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

__attribute__((target("avx2,fma")))
void gemma_q6k_avx2(const uint8_t *w, const float *x, float *out,
                    int rows, int cols, int tokens)
{
    size_t row_bytes = (size_t)(cols >> 8) * 210u;
    (void)tokens;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        out[i] = dot_q6k_row_avx2(w + (size_t)i * row_bytes, x, cols);
    }
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
    float sum = 0.0f;
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
        sum += d * _mm512_reduce_add_ps(acc);
    }
    return sum;
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
void gemma_q6k_avx512(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    size_t row_bytes = (size_t)(cols >> 8) * 210u;
    (void)tokens;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        out[i] = dot_q6k_row_avx512(w + (size_t)i * row_bytes, x, cols);
    }
}

/* Multiply x by W. W is Q6_K data. Use the AVX-512 kernel or the AVX2 kernel.
 * A token count above one uses the scalar kernel. The output head of this
 * model always uses one token.
 */
void gemma_q6k_linear(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    if (tokens > 1) {
        gemma_q6k_scalar(w, x, out, rows, cols, tokens);
        return;
    }
    if (gemma_have_avx512()) {
        gemma_q6k_avx512(w, x, out, rows, cols, tokens);
    } else {
        gemma_q6k_avx2(w, x, out, rows, cols, tokens);
    }
}

#else  /* !GEMMA_X86 */

void gemma_q6k_linear(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    gemma_q6k_scalar(w, x, out, rows, cols, tokens);
}

#endif  /* GEMMA_X86 */

/* ---------- fused attention for one query token ----------
 * A decode step makes one query token for each layer. The NumPy path then
 * builds a score matrix and calls a batched matrix product for a very small
 * matrix. This kernel does the full step: the scores, the softmax, and the
 * weighted sum of the values. It reads an int8 key cache and an int8 value
 * cache. Each cache holds one float32 scale for each group of 32 values.
 *
 * The caller gives the query in int8 form with its own group scales. The
 * query head h uses the key and value head h / (q_heads / kv_heads).
 */

#if GEMMA_X86
__attribute__((target("avx2,fma")))
static inline int32_t dot_i8_i8(const int8_t *a, const int8_t *b, int n)
{
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i + 15 < n; i += 16) {
        __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a + i)));
        __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b + i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    return hsum_epi32_avx2(acc);
}
#else
static inline int32_t dot_i8_i8(const int8_t *a, const int8_t *b, int n)
{
    int32_t d = 0;
    for (int i = 0; i < n; ++i) {
        d += (int32_t)a[i] * (int32_t)b[i];
    }
    return d;
}
#endif

__attribute__((target("avx2,fma")))
void gemma_attn_decode(const int8_t *qq, const float *qs,
                       const int8_t *kq, const float *ks,
                       const int8_t *vq, const float *vs,
                       float *scores, float *out,
                       int q_heads, int kv_heads, int head_dim, int n)
{
    int g = head_dim / 32;
    int n_rep = q_heads / kv_heads;
    size_t kv_stride = (size_t)kv_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)kv_heads * (size_t)g;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < q_heads; ++h) {
        int kv = h / n_rep;
        const int8_t *qh = qq + (size_t)h * (size_t)head_dim;
        const float *qsh = qs + (size_t)h * (size_t)g;
        float *sc = scores + (size_t)h * (size_t)n;
        for (int j = 0; j < n; ++j) {
            const int8_t *kp = kq + (size_t)j * kv_stride + (size_t)kv * (size_t)head_dim;
            const float *ksp = ks + (size_t)j * ks_stride + (size_t)kv * (size_t)g;
            float s = 0.0f;
            for (int gg = 0; gg < g; ++gg) {
                s += (float)dot_i8_i8(kp + (size_t)gg * 32, qh + (size_t)gg * 32, 32)
                     * ksp[gg] * qsh[gg];
            }
            sc[j] = s;
        }
        float m = sc[0];
        for (int j = 1; j < n; ++j) {
            if (sc[j] > m) {
                m = sc[j];
            }
        }
        float l = 0.0f;
        for (int j = 0; j < n; ++j) {
            float e = expf(sc[j] - m);
            sc[j] = e;
            l += e;
        }
        float inv = 1.0f / l;
        float *oh = out + (size_t)h * (size_t)head_dim;
        for (int d = 0; d < head_dim; ++d) {
            oh[d] = 0.0f;
        }
        for (int j = 0; j < n; ++j) {
            float p = sc[j] * inv;
            const int8_t *vp = vq + (size_t)j * kv_stride + (size_t)kv * (size_t)head_dim;
            const float *vsp = vs + (size_t)j * ks_stride + (size_t)kv * (size_t)g;
            for (int gg = 0; gg < g; ++gg) {
                __m256 pv = _mm256_set1_ps(p * vsp[gg]);
                const int8_t *vpp = vp + (size_t)gg * 32;
                float *op = oh + (size_t)gg * 32;
                for (int i = 0; i < 32; i += 8) {
                    __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64((const __m128i *)(vpp + i))));
                    _mm256_storeu_ps(op + i, _mm256_fmadd_ps(pv, vf, _mm256_loadu_ps(op + i)));
                }
            }
        }
    }
}

/* ---------- elementwise kernels ----------
 * The model calls RMSNorm and GELU for each layer. The arrays are small at a
 * decode step. A NumPy call then costs more than the work. These kernels keep
 * the work in C. The parallel region starts only for a large array. Thus a
 * decode step does not pay for a thread team.
 */

/* Normalize the last axis of x. Multiply by the weight when it is present. */
void gemma_rms_norm(const float *x, const float *w, float *out,
                    int rows, int cols, float eps)
{
    #pragma omp parallel for if(rows >= 8) schedule(static)
    for (int i = 0; i < rows; ++i) {
        const float *xi = x + (size_t)i * (size_t)cols;
        float ss = 0.0f;
        for (int k = 0; k < cols; ++k) {
            ss += xi[k] * xi[k];
        }
        float s = 1.0f / sqrtf(ss / (float)cols + eps);
        float *oi = out + (size_t)i * (size_t)cols;
        if (w != NULL) {
            for (int k = 0; k < cols; ++k) {
                oi[k] = xi[k] * s * w[k];
            }
        } else {
            for (int k = 0; k < cols; ++k) {
                oi[k] = xi[k] * s;
            }
        }
    }
}

/* Apply the tanh approximation of GELU to n values. */
void gemma_gelu(const float *x, float *out, int n)
{
    const float c = 0.7978845608028654f;
    #pragma omp parallel for if(n >= 65536) schedule(static)
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
}

/* ---------- float32 kernel for a comparison ---------- */

void gemma_f32_linear(const float *w, const float *x, float *out,
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
