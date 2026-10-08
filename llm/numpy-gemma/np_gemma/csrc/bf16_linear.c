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
 *   5. An int4 kernel that quantizes the activations to int8. The pointwise
 *      loop then uses integer multiply and add. The lanes of the accumulator
 *      hold the tokens, so the kernel sums over k with no horizontal sum. The
 *      tile serves a prompt. Set NP_GEMMA_INT4_Q8=1 to select it.
 *   6. A float32 kernel for a comparison.
 *   7. A Q6_K kernel for the tied output head. It decodes a 210-byte block in
 *      the registers. It reads the weights in place. Thus the load step does
 *      no dequantize of the head.
 *   8. An RMSNorm kernel and a GELU kernel. The model calls these functions
 *      for each layer. A NumPy call on a small array costs more than the work.
 *   9. A flash attention kernel for a prompt. It keeps the scores of one block
 *      at a time and it reads only the keys that the block can see. It has an
 *      AVX-512 version, an AVX2 version, and a straight C version.
 *
 * Each kernel has an AVX-512 version and an AVX2 version. The code selects the
 * AVX-512 version at run time. If the CPU does not give AVX-512, the code uses
 * the AVX2 version. Every target machine gives AVX2.
 *
 * Build this file with:
 *   cc -O3 -mavx2 -mfma -fopenmp -shared -fPIC -o libgemma.so bf16_linear.c
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE         /* sched_setaffinity, cpu_set_t (gp_team_pin) */
#endif
#include <sched.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <dlfcn.h>
#include <pthread.h>

/* ---- teams outside the planned ones (gemma_team_warn) ----
 *
 * Every parallel region of this library starts in GOMP_parallel; the build
 * links it with -Wl,--wrap=GOMP_parallel, so each comes here first. The
 * planned teams (gemma_run_task: the CPU experts of a step and of a mixed
 * group; gemma_run_parts; gemma_run: the CPU model) mark themselves
 * (gp_planned). Any other region of a team of gp_team_warn_min threads or
 * more at the outer level, once warnings are on (gemma_team_warn: the
 * servers turn them on after the load), is reported once for each place of
 * the call: on the 2-socket Xeon such a team of 40 threads (kq_rows of the 64
 * n-gram rows of an MTP verify group) waited 5.7 ms a call for cores that the
 * team of the CPU experts spins on. gemma_team_warn_stats gives the places
 * and the counts. */
static __thread int gp_planned;
static int gp_team_warn_min;
#define GP_WARN_SITES 64
static void *gp_warn_site[GP_WARN_SITES];
static long gp_warn_count[GP_WARN_SITES];
static int gp_warn_nsites;
static pthread_mutex_t gp_warn_mu = PTHREAD_MUTEX_INITIALIZER;

void gemma_team_warn(int min_threads)
{
    gp_team_warn_min = min_threads;
}

/* The places (return addresses) and counts of the reported teams; returns
 * their number. */
int gemma_team_warn_stats(void **sites, long *counts, int max)
{
    pthread_mutex_lock(&gp_warn_mu);
    int n = gp_warn_nsites < max ? gp_warn_nsites : max;
    for (int i = 0; i < n; ++i) {
        sites[i] = gp_warn_site[i];
        counts[i] = gp_warn_count[i];
    }
    pthread_mutex_unlock(&gp_warn_mu);
    return n;
}

static void gp_team_report(void *ra, unsigned n)
{
    int first = 0;
    pthread_mutex_lock(&gp_warn_mu);
    int i = 0;
    while (i < gp_warn_nsites && gp_warn_site[i] != ra) {
        ++i;
    }
    if (i == gp_warn_nsites && i < GP_WARN_SITES) {
        gp_warn_site[i] = ra;
        gp_warn_nsites++;
        first = 1;
    }
    if (i < GP_WARN_SITES) {
        gp_warn_count[i]++;
    }
    pthread_mutex_unlock(&gp_warn_mu);
    if (first) {
        Dl_info d;
        const char *s = "?";
        long off = 0;
        if (dladdr(ra, &d) && d.dli_sname != NULL) {
            s = d.dli_sname;
            off = (long)((char *)ra - (char *)d.dli_saddr);
        }
        fprintf(stderr, "[np-gemma] WARNING: an OpenMP team of %u threads outside the planned teams, "
                "from %s+%#lx (cpu %d); it competes with the cores of the CPU experts "
                "(reported once a place; gemma_team_warn)\n", n, s, off, sched_getcpu());
    }
}

extern void __real_GOMP_parallel(void (*fn)(void *), void *data, unsigned num_threads,
                                 unsigned flags);
void __wrap_GOMP_parallel(void (*fn)(void *), void *data, unsigned num_threads, unsigned flags)
{
    if (gp_team_warn_min > 0 && !gp_planned && omp_get_level() == 0) {
        unsigned n = num_threads ? num_threads : (unsigned)omp_get_max_threads();
        if ((int)n >= gp_team_warn_min) {
            gp_team_report(__builtin_return_address(0), n);
        }
    }
    __real_GOMP_parallel(fn, data, num_threads, flags);
}

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


void gemma_bf16_scalar(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_bf16_scalar_body(w, x, out, rows, cols, tokens);
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
void gemma_bf16_avx2(const uint16_t *w, const float *x, float *out,
                     int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_bf16_avx2_body(w, x, out, rows, cols, tokens);
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
void gemma_bf16_avx512(const uint16_t *w, const float *x, float *out,
                       int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_bf16_avx512_body(w, x, out, rows, cols, tokens);
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
        /* NP_GEMMA_ARCH=avx2 also turns off these kernels, so a CPU with
         * AVX-512 runs the code of an AVX2 CPU (a test for one). */
        const char *arch = getenv("NP_GEMMA_ARCH");
        state = (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                 !(arch && strcmp(arch, "avx2") == 0)) ? 1 : 0;
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

/* The body for a caller that is already in a region. */
static void gemma_bf16_linear_body(const uint16_t *w, const float *x, float *out,
                                   int rows, int cols, int tokens)
{
    gemma_bf16_scalar_body(w, x, out, rows, cols, tokens);
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

/* ---------- int8 x for the int4 products of a few tokens (AVX2) ----------
 * AVX2 has no VNNI and 8 float lanes, so the float products above take a
 * decode to float32 and 4 fma for each 32 weights and each row. Here x is
 * int8 (a scale for each 32 values, as kq_quant_part), and a group of 32 is
 * one vpmaddubsw and one vpmaddwd (the sign of w moves to x), then one fma
 * with the scale of w times the scale of x. llama.cpp does the same on
 * AVX2.
 *
 * A body that uses it quantizes its x rows first (i4q_begin: one thread,
 * then the barrier of omp single) into the scratch of its team; the dot
 * kernels find the int8 row of an x row by its address (i4q_find). An x
 * row that is not in the scratch takes the float kernel. The kernel of one
 * token and the kernel of a group do the same steps for each token, so a
 * verify group gives the bits of the steps. NP_GEMMA_I4Q8=0 turns it off. */
#if GEMMA_X86 && !defined(__AVX512F__)
#define I4Q_TEAMS 8

typedef struct {
    const float *base;      /* the first x row, or NULL */
    size_t stride;          /* floats from one x row to the next */
    int rows, cols;
    int8_t *q;              /* rows x cols */
    float *s;               /* rows x cols / 32 */
    int32_t *n;             /* -8 times the sum of each 32 (KQ_Q4X), stride cols / 16 */
    size_t cap;             /* the values that q holds */
    int16_t *q16;           /* NP_GEMMA_DECODE_X16: the int16 rows too (KQ_Q4X) */
    float *s16;
    int8_t *h16;            /* and their planes (kq_planes16_to): one row */
    uint8_t *l16;
    int32_t *n16;
    size_t cap16;
} i4q_reg;

static i4q_reg i4q_regs[I4Q_TEAMS];

static int i4q_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_I4Q8");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* The KQ_Q4X products of a step (and of a verify group) take int16 x
 * (gemma_quant_group32_i16, kq_q4x_rows16), as the prompt with int16 x:
 * twice the instructions of int8 x. The default; NP_GEMMA_DECODE_X16=0
 * keeps int8 x. */
static int i4q16_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_DECODE_X16");
        on = !(v && v[0] == '0');
    }
    return on;
}

static inline float gemma_quant_group32_i16(const float *x, int16_t *q);
static void kq_q4x_rows16(const uint8_t *wg, int cols, const int16_t *xq, const float *xs, int n,
                          float *out, size_t ostride);
static void kq_planes16_to(const int16_t *xq, int cols, int8_t *kp_h, uint8_t *kp_l, int32_t *kp_n);
static void kq_q4x_rows_p16(const uint8_t *wg, int cols, const int8_t *kp_h, const uint8_t *kp_l,
                            const int32_t *kp_n, const float *xs, float *out);

/* The scratch of this team: the part of gemma_run_parts, or 0. */
static inline i4q_reg *i4q_cur(void)
{
    int lvl = omp_get_level();
    int p = lvl > 1 ? omp_get_ancestor_thread_num(lvl - 1) : 0;
    return &i4q_regs[p % I4Q_TEAMS];
}

/* 32 values of x to int8: s = max |x| / 127, q = round(x / s). */
static inline void i4q_quant32(const float *x, int8_t *q, float *s)
{
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8);
    __m256 v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
    __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                             _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
    __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    h = _mm_max_ps(h, _mm_movehl_ps(h, h));
    h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
    float mx = _mm_cvtss_f32(h);
    float inv = mx > 0.f ? 127.f / mx : 0.f;
    *s = mx / 127.f;
    __m256 iv = _mm256_set1_ps(inv);
    __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, iv)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, iv));
    __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, iv)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, iv));
    __m256i a = _mm256_packs_epi32(i0, i1), b = _mm256_packs_epi32(i2, i3);
    __m256i c = _mm256_packs_epi16(a, b);
    c = _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256((__m256i *)q, c);
}

/* Quantize rows x rows of cols values (stride floats apart) into the scratch
 * of the team. All the threads of the team call it. */
static void i4q_begin(const float *x, size_t stride, int rows, int cols)
{
    i4q_reg *R = i4q_cur();
    #pragma omp single
    {
        R->base = NULL;
        if (i4q_on() && cols % 32 == 0 && rows > 0) {
            size_t need = (size_t)rows * (size_t)cols;
            if (need > R->cap) {
                free(R->q);
                free(R->s);
                free(R->n);
                R->q = (int8_t *)malloc(need);
                R->s = (float *)malloc(need / 32 * sizeof(float));
                R->n = (int32_t *)malloc(need / 16 * sizeof(int32_t));
                R->cap = R->q && R->s && R->n ? need : 0;
            }
            if (R->cap >= need) {
                for (int r = 0; r < rows; ++r) {
                    const float *xr = x + (size_t)r * stride;
                    int8_t *qr = R->q + (size_t)r * cols;
                    float *sr = R->s + (size_t)r * (cols / 32);
                    int32_t *nr = R->n + (size_t)r * (cols / 16);
                    for (int g = 0; g < cols / 32; ++g) {
                        i4q_quant32(xr + 32 * g, qr + 32 * g, sr + g);
                        int32_t sum = 0;
                        for (int u = 0; u < 32; ++u) {
                            sum += qr[32 * g + u];
                        }
                        nr[g] = -8 * sum;
                    }
                }
                if (i4q16_on() && R->cap16 < need) {
                    free(R->q16);
                    free(R->s16);
                    free(R->h16);
                    free(R->l16);
                    free(R->n16);
                    R->q16 = (int16_t *)malloc(need * sizeof(int16_t));
                    R->s16 = (float *)malloc(need / 32 * sizeof(float));
                    R->h16 = (int8_t *)malloc(need + 64);
                    R->l16 = (uint8_t *)malloc(need + 64);
                    R->n16 = (int32_t *)malloc((need / 32 + 16) * sizeof(int32_t));
                    R->cap16 = R->q16 && R->s16 && R->h16 && R->l16 && R->n16 ? need : 0;
                }
                if (i4q16_on() && R->cap16 >= need) {
                    for (int r = 0; r < rows; ++r) {
                        for (int g = 0; g < cols / 32; ++g) {
                            R->s16[(size_t)r * (cols / 32) + g] = gemma_quant_group32_i16(
                                x + (size_t)r * stride + 32 * g, R->q16 + (size_t)r * cols + 32 * g);
                        }
                    }
                    if (rows == 1) {
                        kq_planes16_to(R->q16, cols, R->h16, R->l16, R->n16);
                    }
                }
                R->base = x;
                R->stride = stride;
                R->rows = rows;
                R->cols = cols;
            }
        }
    }
}

/* The int8 row and its scales for the x row x of n values, or NULL. xn
 * (if not NULL) gets the sums for KQ_Q4X. */
static inline const int8_t *i4q_find2(const float *x, int n, const float **xs,
                                      const int32_t **xn)
{
    const i4q_reg *R = i4q_cur();
    if (R->base == NULL || R->cols != n || x < R->base) {
        return NULL;
    }
    size_t d = (size_t)(x - R->base);
    size_t r = R->stride ? d / R->stride : 0;
    if ((R->stride ? d % R->stride : d) != 0 || r >= (size_t)R->rows) {
        return NULL;
    }
    *xs = R->s + r * (size_t)(n / 32);
    if (xn != NULL) {
        *xn = R->n + r * (size_t)(n / 16);
    }
    return R->q + r * (size_t)n;
}

static inline const int8_t *i4q_find(const float *x, int n, const float **xs)
{
    return i4q_find2(x, n, xs, NULL);
}

/* The int16 row (NP_GEMMA_DECODE_X16) and its scales for the x row x, or NULL. */
static inline const int16_t *i4q_find16(const float *x, int n, const float **xs)
{
    const i4q_reg *R = i4q_cur();
    const float *s8;
    const int8_t *q = i4q_find2(x, n, &s8, NULL);
    if (q == NULL || !i4q16_on() || R->cap16 < (size_t)R->rows * (size_t)R->cols) {
        return NULL;
    }
    size_t r = (size_t)(q - R->q) / (size_t)n;
    *xs = R->s16 + r * (size_t)(n / 32);
    return R->q16 + r * (size_t)n;
}

/* The 32 signed values (-8 to 7) of a Q4_0 group, in the order of x. */
static inline __m256i i4q_w32(const uint8_t *p)
{
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i b = _mm_loadu_si128((const __m128i *)p);
    __m256i w = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(b, 4), m4), _mm_and_si128(b, m4));
    return _mm256_sub_epi8(w, _mm256_set1_epi8(8));
}

/* The int32 sums (8 lanes) of 32 products of w (signed, abs aw) and x. */
static inline __m256 i4q_dot32(__m256i aw, __m256i w, __m256i x)
{
    __m256i p = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

/* One row, one token (the tail rows). */
static inline float dot_i4_q8(const uint8_t *w, const float *scales, const int8_t *xq,
                              const float *xs, int n)
{
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < n / 32; ++g) {
        __m256i wv = i4q_w32(w + (size_t)g * 18 + 2);
        __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + (size_t)g * 32));
        acc = _mm256_fmadd_ps(i4q_dot32(_mm256_abs_epi8(wv), wv, xv),
                              _mm256_set1_ps(scales[g] * xs[g]), acc);
    }
    return hsum256_ps(acc);
}

/* Four rows (stride bytes apart) for NT tokens (a constant 1 or 2, so the
 * sums stay in registers). Each (row, token) adds its groups in the order
 * of dot_i4_q8. */
static inline __attribute__((always_inline)) void dot4_i4_q8_n(
    const uint8_t *w, int stride, const float *scales, const int8_t *const *xq,
    const float *const *xs, const int NT, int n, float *r)
{
    int groups = n / 32;
    __m256 a[2][4];
    for (int t = 0; t < NT; ++t) {
        for (int j = 0; j < 4; ++j) {
            a[t][j] = _mm256_setzero_ps();
        }
    }
    for (int g = 0; g < groups; ++g) {
        __m256i xv[2];
        for (int t = 0; t < NT; ++t) {
            xv[t] = _mm256_loadu_si256((const __m256i *)(xq[t] + (size_t)g * 32));
        }
        for (int j = 0; j < 4; ++j) {
            __m256i wv = i4q_w32(w + (size_t)j * stride + (size_t)g * 18 + 2);
            __m256i aw = _mm256_abs_epi8(wv);
            float sw = scales[(size_t)j * groups + g];
            for (int t = 0; t < NT; ++t) {
                a[t][j] = _mm256_fmadd_ps(i4q_dot32(aw, wv, xv[t]), _mm256_set1_ps(sw * xs[t][g]),
                                          a[t][j]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        for (int j = 0; j < 4; ++j) {
            r[t * 4 + j] = hsum256_ps(a[t][j]);
        }
    }
}
#endif

/* The four-row and one-row dots of the bodies below: on AVX2 the int8 x of
 * i4q_begin when the body quantized x, else the float kernels. */
#if GEMMA_X86 && !defined(__AVX512F__)
static inline void I4_DOT4(const uint8_t *w, int stride, const float *scales, const float *x,
                           int n, float *r)
{
    const float *xs;
    const int8_t *q = i4q_find(x, n, &xs);
    if (q != NULL) {
        dot4_i4_q8_n(w, stride, scales, &q, &xs, 1, n, r);
    } else {
        dot4_i4_f32(w, stride, scales, x, n, r);
    }
}

static inline float I4_DOT1(const uint8_t *w, const float *scales, const float *x, int n)
{
    const float *xs;
    const int8_t *q = i4q_find(x, n, &xs);
    return q != NULL ? dot_i4_q8(w, scales, q, xs, n) : dot_i4_f32(w, scales, x, n);
}
#define I4Q_BEGIN(x, stride, rows, cols) i4q_begin((x), (stride), (rows), (cols))
#else
#define I4_DOT4 dot4_i4_f32
#define I4_DOT1 dot_i4_f32
#define I4Q_BEGIN(x, stride, rows, cols) ((void)0)
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

/* csrc/kquants.c (included later): the 16 rows of a KQ_Q4X group on float32 x,
 * and on int8 x. */
static void kq_q4x_rows_f(const uint8_t *wg, int cols, const float *x, size_t xstride, int n,
                          float *out, size_t ostride);
static void kq_q4x_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride);

/* The 16 rows of a KQ_Q4X group on n rows of x (xstride == cols, the rows
 * of one i4q_begin): on AVX2 the int8 x of i4q_begin, else float32 x. */
static inline void q4x_rows_x(const uint8_t *wg, int cols, const float *x, size_t xstride, int n,
                              float *out, size_t ostride)
{
#if GEMMA_X86 && !defined(__AVX512F__)
    const float *xs, *xs_last;
    const int32_t *xn;
    const int8_t *q = xstride == (size_t)cols ? i4q_find2(x, cols, &xs, &xn) : NULL;
    /* the n rows must all be in the scratch (they follow each other there) */
    if (q != NULL && (n == 1 || i4q_find(x + (size_t)(n - 1) * xstride, cols, &xs_last) != NULL)) {
        const float *s16;
        const int16_t *q16 = i4q_find16(x, cols, &s16);
        if (q16 != NULL) {
            const i4q_reg *R = i4q_cur();
            if (n == 1 && R->rows == 1) {
                /* a step: the planes of i4q_begin */
                kq_q4x_rows_p16(wg, cols, R->h16, R->l16, R->n16, s16, out);
            } else {
                kq_q4x_rows16(wg, cols, q16, s16, n, out, ostride);
            }
            return;
        }
        kq_q4x_rows(wg, cols, q, xs, xn, n, out, ostride);
        return;
    }
#endif
    kq_q4x_rows_f(wg, cols, x, xstride, n, out, ostride);
}

/* scales NULL: w is the KQ_Q4X copy of the matrix (ops.q4x_pack_model;
 * program.py gives it to the records of one token on the CPU): groups of 16
 * rows, float32 x, the same products. */
static void gemma_int4_linear_body(const uint8_t *w, const float *scales, const float *x, float *out,
                       int rows, int cols, int tokens, int group)
{
    if (scales == NULL) {
        size_t gb = (size_t)16 * (cols / 32) * 18;
        I4Q_BEGIN(x, (size_t)cols, tokens, cols);
        #pragma omp for schedule(static)
        for (int g = 0; g < rows / 16; ++g) {
            q4x_rows_x(w + (size_t)g * gb, cols, x, (size_t)cols, tokens, out + 16 * g,
                       (size_t)rows);
        }
        return;
    }
    /* The fast dot uses a group of 32 values. ops.linear_int4 sends only that
     * group size. */
    (void)group;
    int groups = cols / 32;
    int stride = (cols / 32) * 18;
    if (gemma_int4_rows4 && tokens == 1) {
        I4Q_BEGIN(x, (size_t)cols, 1, cols);
        int blocks = (rows + 3) / 4;
        #pragma omp for schedule(static)
        for (int b = 0; b < blocks; ++b) {
            int i = b * 4;
            int left = rows - i;
            if (left >= 4) {
                float r[4];
                I4_DOT4(w + (size_t)i * (size_t)stride, stride,
                        scales + (size_t)i * (size_t)groups, x, cols, r);
                out[i] = r[0];
                out[i + 1] = r[1];
                out[i + 2] = r[2];
                out[i + 3] = r[3];
            } else {
                for (int j = 0; j < left; ++j) {
                    out[i + j] = I4_DOT1(w + (size_t)(i + j) * (size_t)stride,
                                         scales + (size_t)(i + j) * (size_t)groups,
                                         x, cols);
                }
            }
        }
        return;
    }
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint8_t *wi = w + (size_t)i * (size_t)stride;
        const float *si = scales + (size_t)i * (size_t)groups;
        for (int t = 0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i4_f32(wi, si, x + (size_t)t * (size_t)cols, cols);
        }
    }
}

void gemma_int4_linear(const uint8_t *w, const float *scales, const float *x, float *out,
                       int rows, int cols, int tokens, int group)
{
    #pragma omp parallel
    gemma_int4_linear_body(w, scales, x, out, rows, cols, tokens, group);
}

/* ---------- four int4 matrices on one x row ----------
 * The query, the key, and the value projection of one attention layer share
 * the x row. Three calls then start three OpenMP regions for the same x. This
 * kernel runs up to four matrices in one region. A null weight pointer skips
 * a matrix. The MLP gate and up projection can use the same kernel.
 */
static void gemma_int4_multi4_body(const uint8_t *w0, const float *s0, float *o0, int rows0,
                       const uint8_t *w1, const float *s1, float *o1, int rows1,
                       const uint8_t *w2, const float *s2, float *o2, int rows2,
                       const uint8_t *w3, const float *s3, float *o3, int rows3,
                       const float *x, int cols)
{
    int groups = cols / 32;
    size_t stride = (size_t)groups * 18;
    /* a matrix with s NULL is a KQ_Q4X copy: a task is a group of 16 rows */
    int b0 = w0 ? (s0 ? (rows0 + 3) / 4 : rows0 / 16) : 0;
    int b1 = w1 ? (s1 ? (rows1 + 3) / 4 : rows1 / 16) : 0;
    int b2 = w2 ? (s2 ? (rows2 + 3) / 4 : rows2 / 16) : 0;
    int b3 = w3 ? (s3 ? (rows3 + 3) / 4 : rows3 / 16) : 0;
    long total = (long)b0 + (long)b1 + (long)b2 + (long)b3;
    I4Q_BEGIN(x, (size_t)cols, 1, cols);
    #pragma omp for schedule(static)
    for (long t = 0; t < total; ++t) {
        const uint8_t *w = w0;
        const float *s = s0;
        float *o = o0;
        int rows = rows0;
        long u = t;
        if (u >= b0) {
            u -= b0;
            if (u < b1) {
                w = w1; s = s1; o = o1; rows = rows1;
            } else {
                u -= b1;
                if (u < b2) {
                    w = w2; s = s2; o = o2; rows = rows2;
                } else {
                    u -= b2;
                    w = w3; s = s3; o = o3; rows = rows3;
                }
            }
        }
        if (s == NULL) {
            q4x_rows_x(w + (size_t)u * 16 * stride, cols, x, (size_t)cols, 1, o + 16 * u,
                       (size_t)rows);
            continue;
        }
        int i = (int)u * 4;
        if (i + 4 <= rows) {
            float r[4];
            I4_DOT4(w + (size_t)i * stride, (int)stride,
                    s + (size_t)i * (size_t)groups, x, cols, r);
            o[i] = r[0];
            o[i + 1] = r[1];
            o[i + 2] = r[2];
            o[i + 3] = r[3];
        } else {
            for (int q = i; q < rows; ++q) {
                o[q] = I4_DOT1(w + (size_t)q * stride,
                               s + (size_t)q * (size_t)groups, x, cols);
            }
        }
    }
}

void gemma_int4_multi4(const uint8_t *w0, const float *s0, float *o0, int rows0,
                       const uint8_t *w1, const float *s1, float *o1, int rows1,
                       const uint8_t *w2, const float *s2, float *o2, int rows2,
                       const uint8_t *w3, const float *s3, float *o3, int rows3,
                       const float *x, int cols)
{
    #pragma omp parallel
    gemma_int4_multi4_body(w0, s0, o0, rows0, w1, s1, o1, rows1, w2, s2, o2, rows2, w3, s3, o3, rows3, x, cols);
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
static void gemma_int4_moe_gemv_body(const uint8_t *w, const float *scales,
                         const float *x, const int *ids, int jobs,
                         float *out, int rows, int cols, int xstride)
{
    int groups = cols / 32;
    size_t stride = (size_t)groups * 18;
    size_t expert_bytes = (size_t)rows * stride;
    size_t expert_scales = (size_t)rows * (size_t)groups;
    int blocks = (rows + 3) / 4;
    long total = (long)jobs * (long)blocks;
    I4Q_BEGIN(x, (size_t)xstride, xstride ? jobs : 1, cols);
    #pragma omp for schedule(static)
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
            I4_DOT4(wj + (size_t)i * stride, (int)stride,
                    sj + (size_t)i * (size_t)groups, xj, cols, r);
            oj[i] = r[0];
            oj[i + 1] = r[1];
            oj[i + 2] = r[2];
            oj[i + 3] = r[3];
        } else {
            for (int q = i; q < rows; ++q) {
                oj[q] = I4_DOT1(wj + (size_t)q * stride,
                                sj + (size_t)q * (size_t)groups, xj, cols);
            }
        }
    }
}

void gemma_int4_moe_gemv(const uint8_t *w, const float *scales,
                         const float *x, const int *ids, int jobs,
                         float *out, int rows, int cols, int xstride)
{
    #pragma omp parallel
    gemma_int4_moe_gemv_body(w, scales, x, ids, jobs, out, rows, cols, xstride);
}

/* ---------- int4 GEMV for a small group of tokens ----------
 * An MTP verify step runs the target on two to eight tokens. The prompt
 * kernels are tuned for a large token block, and a decode kernel reads the
 * weights again for each token. These kernels read a weight block one time
 * for all the tokens.
 *
 * The steps for one token are the steps of the one-token kernel, in the same
 * order. Thus each token gets the same result, bit for bit, as a decode step.
 * The MTP decode then gives the same text as the plain decode.
 *
 * xs gives the address of the x row of each token. The dot writes
 * r[t * 4 + j] for token t and weight row j.
 */
#define I4MT_T 4
#define I4MT_MAX 16

#if GEMMA_X86 && defined(__AVX512F__)
static inline void dot4_i4_f32_mt(const uint8_t *w, int stride, const float *scales,
                                  const float *const *xs, int nt, int n, float *r)
{
    __m512 a[I4MT_T][4];
    for (int t = 0; t < nt; ++t) {
        a[t][0] = _mm512_setzero_ps();
        a[t][1] = _mm512_setzero_ps();
        a[t][2] = _mm512_setzero_ps();
        a[t][3] = _mm512_setzero_ps();
    }
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        const uint8_t *p = w + (size_t)g * 18 + 2;
        __m512 l0, h0, l1, h1, l2, h2, l3, h3;
        i4_pair(_mm_loadu_si128((const __m128i *)(p)), mask, b8, &l0, &h0);
        i4_pair(_mm_loadu_si128((const __m128i *)(p + stride)), mask, b8, &l1, &h1);
        i4_pair(_mm_loadu_si128((const __m128i *)(p + (size_t)2 * stride)), mask, b8, &l2, &h2);
        i4_pair(_mm_loadu_si128((const __m128i *)(p + (size_t)3 * stride)), mask, b8, &l3, &h3);
        __m512 s0 = _mm512_set1_ps(scales[g]);
        __m512 s1 = _mm512_set1_ps(scales[(size_t)groups + g]);
        __m512 s2 = _mm512_set1_ps(scales[(size_t)2 * groups + g]);
        __m512 s3 = _mm512_set1_ps(scales[(size_t)3 * groups + g]);
        for (int t = 0; t < nt; ++t) {
            __m512 xlo = _mm512_loadu_ps(xs[t] + (size_t)g * 32);
            __m512 xhi = _mm512_loadu_ps(xs[t] + (size_t)g * 32 + 16);
            a[t][0] = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l0, _mm512_mul_ps(xhi, h0)), s0, a[t][0]);
            a[t][1] = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l1, _mm512_mul_ps(xhi, h1)), s1, a[t][1]);
            a[t][2] = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l2, _mm512_mul_ps(xhi, h2)), s2, a[t][2]);
            a[t][3] = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l3, _mm512_mul_ps(xhi, h3)), s3, a[t][3]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        r[t * 4 + 0] = _mm512_reduce_add_ps(a[t][0]);
        r[t * 4 + 1] = _mm512_reduce_add_ps(a[t][1]);
        r[t * 4 + 2] = _mm512_reduce_add_ps(a[t][2]);
        r[t * 4 + 3] = _mm512_reduce_add_ps(a[t][3]);
    }
}
#else
/* AVX2 and scalar: run the one-token dot for each token. The four weight
 * rows stay in the L1 cache, so the memory reads them one time. */
static inline void dot4_i4_f32_mt(const uint8_t *w, int stride, const float *scales,
                                  const float *const *xs, int nt, int n, float *r)
{
    for (int t = 0; t < nt; ++t) {
        dot4_i4_f32(w, stride, scales, xs[t], n, r + t * 4);
    }
}
#endif

/* Rows i to i + 3 (or fewer at the end) of one matrix for nt tokens. out has
 * the row stride ostride for each token. */
static inline void i4_rows_mt(const uint8_t *w, const float *s, int rows, int cols,
                              int i, const float *const *xs, float *const *outs, int nt)
{
    int groups = cols / 32;
    size_t stride = (size_t)groups * 18;
#if GEMMA_X86 && !defined(__AVX512F__)
    {
        /* int8 x of i4q_begin: 2 tokens at a time; each token has the steps
         * of the one-token kernel */
        const int8_t *q[I4MT_MAX];
        const float *qs[I4MT_MAX];
        int all = nt <= I4MT_MAX;
        for (int t = 0; t < nt && all; ++t) {
            q[t] = i4q_find(xs[t], cols, &qs[t]);
            all = q[t] != NULL;
        }
        if (all) {
            for (int t0 = 0; t0 < nt; t0 += 2) {
                int n = nt - t0 < 2 ? nt - t0 : 2;
                if (i + 4 <= rows) {
                    float r[8];
                    if (n == 2) {
                        dot4_i4_q8_n(w + (size_t)i * stride, (int)stride, s + (size_t)i * groups,
                                     q + t0, qs + t0, 2, cols, r);
                    } else {
                        dot4_i4_q8_n(w + (size_t)i * stride, (int)stride, s + (size_t)i * groups,
                                     q + t0, qs + t0, 1, cols, r);
                    }
                    for (int t = 0; t < n; ++t) {
                        float *o = outs[t0 + t];
                        o[i] = r[t * 4];
                        o[i + 1] = r[t * 4 + 1];
                        o[i + 2] = r[t * 4 + 2];
                        o[i + 3] = r[t * 4 + 3];
                    }
                } else {
                    for (int t = 0; t < n; ++t) {
                        for (int k = i; k < rows; ++k) {
                            outs[t0 + t][k] = dot_i4_q8(w + (size_t)k * stride, s + (size_t)k * groups,
                                                        q[t0 + t], qs[t0 + t], cols);
                        }
                    }
                }
            }
            return;
        }
    }
#endif
    for (int t0 = 0; t0 < nt; t0 += I4MT_T) {
        int n = nt - t0 < I4MT_T ? nt - t0 : I4MT_T;
        if (i + 4 <= rows) {
            float r[I4MT_T * 4];
            dot4_i4_f32_mt(w + (size_t)i * stride, (int)stride,
                           s + (size_t)i * (size_t)groups, xs + t0, n, cols, r);
            for (int t = 0; t < n; ++t) {
                float *o = outs[t0 + t];
                o[i] = r[t * 4];
                o[i + 1] = r[t * 4 + 1];
                o[i + 2] = r[t * 4 + 2];
                o[i + 3] = r[t * 4 + 3];
            }
        } else {
            for (int t = 0; t < n; ++t) {
                for (int q = i; q < rows; ++q) {
                    outs[t0 + t][q] = dot_i4_f32(w + (size_t)q * stride,
                                                 s + (size_t)q * (size_t)groups,
                                                 xs[t0 + t], cols);
                }
            }
        }
    }
}


/* x is (tokens, cols) and out is (tokens, rows). */
static void gemma_int4_linear_mt_body(const uint8_t *w, const float *scales, const float *x,
                          float *out, int rows, int cols, int tokens)
{
    if (scales == NULL) {
        /* a KQ_Q4X copy (see gemma_int4_linear_body): each token has the
         * operations of a step, so a verify group gives the bits of steps */
        size_t gb = (size_t)16 * (cols / 32) * 18;
        I4Q_BEGIN(x, (size_t)cols, tokens, cols);
        #pragma omp for schedule(static)
        for (int g = 0; g < rows / 16; ++g) {
            q4x_rows_x(w + (size_t)g * gb, cols, x, (size_t)cols, tokens, out + 16 * g,
                       (size_t)rows);
        }
        return;
    }
    const float *xs[I4MT_MAX];
    float *outs[I4MT_MAX];
    for (int t = 0; t < tokens; ++t) {
        xs[t] = x + (size_t)t * (size_t)cols;
        outs[t] = out + (size_t)t * (size_t)rows;
    }
    int blocks = (rows + 3) / 4;
    I4Q_BEGIN(x, (size_t)cols, tokens, cols);
    #pragma omp for schedule(static)
    for (int b = 0; b < blocks; ++b) {
        i4_rows_mt(w, scales, rows, cols, b * 4, xs, outs, tokens);
    }
}

void gemma_int4_linear_mt(const uint8_t *w, const float *scales, const float *x,
                          float *out, int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_int4_linear_mt_body(w, scales, x, out, rows, cols, tokens);
}

/* Up to four matrices on the same x rows. Matrix k writes (tokens, rows_k). */
static void gemma_int4_multi4_mt_body(const uint8_t *w0, const float *s0, float *o0, int rows0,
                          const uint8_t *w1, const float *s1, float *o1, int rows1,
                          const uint8_t *w2, const float *s2, float *o2, int rows2,
                          const uint8_t *w3, const float *s3, float *o3, int rows3,
                          const float *x, int cols, int tokens)
{
    const uint8_t *ws[4] = {w0, w1, w2, w3};
    const float *ss[4] = {s0, s1, s2, s3};
    float *os[4] = {o0, o1, o2, o3};
    int rs[4] = {rows0, rows1, rows2, rows3};
    long nb[4];
    long total = 0;
    for (int k = 0; k < 4; ++k) {
        /* s NULL: a KQ_Q4X copy, a task is a group of 16 rows */
        nb[k] = ws[k] ? (ss[k] ? (rs[k] + 3) / 4 : rs[k] / 16) : 0;
        total += nb[k];
    }
    const float *xs[I4MT_MAX];
    for (int t = 0; t < tokens; ++t) {
        xs[t] = x + (size_t)t * (size_t)cols;
    }
    I4Q_BEGIN(x, (size_t)cols, tokens, cols);
    #pragma omp for schedule(static)
    for (long u = 0; u < total; ++u) {
        int k = 0;
        long v = u;
        while (v >= nb[k]) {
            v -= nb[k];
            ++k;
        }
        if (ss[k] == NULL) {
            q4x_rows_x(ws[k] + (size_t)v * 16 * (cols / 32) * 18, cols, x, (size_t)cols, tokens,
                       os[k] + 16 * v, (size_t)rs[k]);
            continue;
        }
        float *outs[I4MT_MAX];
        for (int t = 0; t < tokens; ++t) {
            outs[t] = os[k] + (size_t)t * (size_t)rs[k];
        }
        i4_rows_mt(ws[k], ss[k], rs[k], cols, (int)v * 4, xs, outs, tokens);
    }
}

void gemma_int4_multi4_mt(const uint8_t *w0, const float *s0, float *o0, int rows0,
                          const uint8_t *w1, const float *s1, float *o1, int rows1,
                          const uint8_t *w2, const float *s2, float *o2, int rows2,
                          const uint8_t *w3, const float *s3, float *o3, int rows3,
                          const float *x, int cols, int tokens)
{
    #pragma omp parallel
    gemma_int4_multi4_mt_body(w0, s0, o0, rows0, w1, s1, o1, rows1, w2, s2, o2, rows2, w3, s3, o3, rows3, x, cols, tokens);
}

/* The selected experts of a small group of tokens. Job j is one expert,
 * ids[j], with the pairs poff[j] to poff[j + 1] - 1. Pair p reads the x row
 * xi[p] (the stride is xstride) and writes the out row p. Each expert is read
 * one time for all of its tokens. */
static void gemma_int4_moe_gemv_mt_body(const uint8_t *w, const float *scales, const float *x,
                            const int *ids, const int *poff, const int *xi, int jobs,
                            float *out, int rows, int cols, int xstride)
{
    int groups = cols / 32;
    size_t expert_bytes = (size_t)rows * (size_t)groups * 18;
    size_t expert_scales = (size_t)rows * (size_t)groups;
    int blocks = (rows + 3) / 4;
    long total = (long)jobs * (long)blocks;
    int xrows = 1;
    for (int p = 0; p < poff[jobs]; ++p) {
        xrows = xi[p] + 1 > xrows ? xi[p] + 1 : xrows;
    }
    I4Q_BEGIN(x, (size_t)xstride, xrows, cols);
    #pragma omp for schedule(static)
    for (long u = 0; u < total; ++u) {
        int j = (int)(u / blocks);
        int i = (int)(u % blocks) * 4;
        int e = ids[j];
        int np = poff[j + 1] - poff[j];
        const float *xs[I4MT_MAX];
        float *outs[I4MT_MAX];
        for (int q = 0; q < np; ++q) {
            int p = poff[j] + q;
            xs[q] = x + (size_t)xi[p] * (size_t)xstride;
            outs[q] = out + (size_t)p * (size_t)rows;
        }
        i4_rows_mt(w + (size_t)e * expert_bytes, scales + (size_t)e * expert_scales,
                   rows, cols, i, xs, outs, np);
    }
}

void gemma_int4_moe_gemv_mt(const uint8_t *w, const float *scales, const float *x,
                            const int *ids, const int *poff, const int *xi, int jobs,
                            float *out, int rows, int cols, int xstride)
{
    #pragma omp parallel
    gemma_int4_moe_gemv_mt_body(w, scales, x, ids, poff, xi, jobs, out, rows, cols, xstride);
}

void gemma_gelu_mul(const float *x, float *out, int rows, int inner);

/* The gate and up projection of the experts for a group of tokens. Then the
 * GELU and the multiply. act holds 2 * inner values for each pair, and out
 * holds inner values for each pair. */
void gemma_moe_gemv_gelu_mt(const uint8_t *w, const float *scales, const float *x,
                            const int *ids, const int *poff, const int *xi, int jobs,
                            float *act, float *out, int rows, int cols, int xstride,
                            int inner)
{
    gemma_int4_moe_gemv_mt(w, scales, x, ids, poff, xi, jobs, act, rows, cols, xstride);
    gemma_gelu_mul(act, out, poff[jobs], inner);
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

/* Return the index of the largest of the n values of x. At an equal value
 * the lower index wins, as np.argmax. The first pass finds the largest value
 * with AVX2 max; the second finds its first index with a compare and a mask.
 * A NaN is not the largest value here (np.argmax gives the NaN). */
int64_t gemma_argmax(const float *x, int64_t n)
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
void gemma_q6k_rows(const uint8_t *table, const int64_t *ids, int n, int cols, float *out)
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
void gemma_q4_0_rows(const uint8_t *table, const int64_t *ids, int n, int cols, float *out)
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
void gemma_kq45_rows(const uint8_t *table, const int64_t *ids, int n, int cols, int five,
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

void gemma_q6k_scalar(const uint8_t *w, const float *x, float *out,
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

void gemma_q6k_avx2(const uint8_t *w, const float *x, float *out,
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

void gemma_q6k_avx512(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_q6k_avx512_body(w, x, out, rows, cols, tokens);
}

/* Multiply x by W. W is Q6_K data. Use the AVX-512 kernel or the AVX2 kernel.
 * Both kernels take a group of tokens. The result of each token is the same
 * as the result of a call with that token alone.
 */
void gemma_q6k_linear(const uint8_t *w, const float *x, float *out,
                      int rows, int cols, int tokens)
{
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

/* ---------- mixture-of-experts router ----------
 * The router for one token does a small matrix product, a softmax, and a
 * top-k. The matrices are small, and the NumPy path then costs more in the
 * call of each small function than in the work. This kernel does the full
 * step. The parallel loop covers the experts.
 */
/* The logit of one expert: r times a row of the router projection. The
 * vector forms keep 4 accumulators and add them in one order; the router of
 * a step and of a group (gemma_router_mt_body) both use it, so a group
 * selects the experts of the steps. The scalar loop with 4 accumulators
 * did not vectorize (gcc keeps the order of float sums), and the router
 * took 95 us a layer on AVX2 with 6 threads. */
static inline float router_dot(const float *r, const float *pe, int hidden)
{
    int k = 0;
    float a = 0.0f;
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
    for (; k + 64 <= hidden; k += 64) {
        c0 = _mm512_fmadd_ps(_mm512_loadu_ps(r + k), _mm512_loadu_ps(pe + k), c0);
        c1 = _mm512_fmadd_ps(_mm512_loadu_ps(r + k + 16), _mm512_loadu_ps(pe + k + 16), c1);
        c2 = _mm512_fmadd_ps(_mm512_loadu_ps(r + k + 32), _mm512_loadu_ps(pe + k + 32), c2);
        c3 = _mm512_fmadd_ps(_mm512_loadu_ps(r + k + 48), _mm512_loadu_ps(pe + k + 48), c3);
    }
    a = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3)));
#elif GEMMA_X86
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    for (; k + 32 <= hidden; k += 32) {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(r + k), _mm256_loadu_ps(pe + k), c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(r + k + 8), _mm256_loadu_ps(pe + k + 8), c1);
        c2 = _mm256_fmadd_ps(_mm256_loadu_ps(r + k + 16), _mm256_loadu_ps(pe + k + 16), c2);
        c3 = _mm256_fmadd_ps(_mm256_loadu_ps(r + k + 24), _mm256_loadu_ps(pe + k + 24), c3);
    }
    a = hsum256_ps(_mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3)));
#endif
    for (; k < hidden; ++k) {
        a += r[k] * pe[k];
    }
    return a;
}

static void gemma_router_body(const float *x, const float *scale, const float *proj,
                              const float *per_expert, int hidden, int experts,
                              int top_k, float eps, float hscale, float *val,
                              int *idx, float *r, float *logits)
{
    #pragma omp single
    {
        float ss = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            ss += x[k] * x[k];
        }
        float inv = 1.0f / sqrtf(ss / (float)hidden + eps);
        for (int k = 0; k < hidden; ++k) {
            r[k] = x[k] * inv * scale[k] * hscale;
        }
    }
    #pragma omp for schedule(static)
    for (int e = 0; e < experts; ++e) {
        logits[e] = router_dot(r, proj + (size_t)e * (size_t)hidden, hidden);
    }
    #pragma omp single
    {
        float m = logits[0];
        for (int e = 1; e < experts; ++e) {
            if (logits[e] > m) {
                m = logits[e];
            }
        }
        float sum = 0.0f;
        for (int e = 0; e < experts; ++e) {
            logits[e] = expf(logits[e] - m);
            sum += logits[e];
        }
        float invs = 1.0f / sum;
        for (int e = 0; e < experts; ++e) {
            logits[e] *= invs;
        }
        for (int j = 0; j < top_k; ++j) {
            int best = 0;
            float bv = logits[0];
            for (int e = 1; e < experts; ++e) {
                if (logits[e] > bv) {
                    bv = logits[e];
                    best = e;
                }
            }
            val[j] = bv;
            idx[j] = best;
            logits[best] = -1.0f;
        }
        float vs = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            vs += val[j];
        }
        float invv = 1.0f / vs;
        for (int j = 0; j < top_k; ++j) {
            val[j] = val[j] * invv * per_expert[idx[j]];
        }
    }
}

void gemma_router(const float *x, const float *scale, const float *proj,
                  const float *per_expert, int hidden, int experts, int top_k,
                  float eps, float hscale, float *val, int *idx)
{
    float r[hidden];
    float logits[experts];
    #pragma omp parallel
    gemma_router_body(x, scale, proj, per_expert, hidden, experts, top_k, eps,
                      hscale, val, idx, r, logits);
}

/* The router for a small group of tokens. The steps for each token are the
 * steps of gemma_router, so each token selects the same experts with the same
 * weights. One parallel region covers the logits of every token. x is
 * (tokens, hidden), val and idx are (tokens, top_k). tokens is at most 16. */
static void gemma_router_mt_body(const float *x, const float *scale, const float *proj,
                                 const float *per_expert, int hidden, int experts,
                                 int top_k, float eps, float hscale, float *val,
                                 int *idx, int tokens, float *r, float *logits)
{
    #pragma omp single
    for (int t = 0; t < tokens; ++t) {
        const float *xt = x + (size_t)t * (size_t)hidden;
        float *rt = r + (size_t)t * (size_t)hidden;
        float ss = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            ss += xt[k] * xt[k];
        }
        float inv = 1.0f / sqrtf(ss / (float)hidden + eps);
        for (int k = 0; k < hidden; ++k) {
            rt[k] = xt[k] * inv * scale[k] * hscale;
        }
    }
    #pragma omp for schedule(static)
    for (int e = 0; e < experts; ++e) {
        const float *pe = proj + (size_t)e * (size_t)hidden;
        for (int t = 0; t < tokens; ++t) {
            logits[(size_t)t * (size_t)experts + e] =
                router_dot(r + (size_t)t * (size_t)hidden, pe, hidden);
        }
    }
    #pragma omp single
    for (int t = 0; t < tokens; ++t) {
        float *lg = logits + (size_t)t * (size_t)experts;
        float *vt = val + (size_t)t * (size_t)top_k;
        int *it = idx + (size_t)t * (size_t)top_k;
        float m = lg[0];
        for (int e = 1; e < experts; ++e) {
            if (lg[e] > m) {
                m = lg[e];
            }
        }
        float sum = 0.0f;
        for (int e = 0; e < experts; ++e) {
            lg[e] = expf(lg[e] - m);
            sum += lg[e];
        }
        float invs = 1.0f / sum;
        for (int e = 0; e < experts; ++e) {
            lg[e] *= invs;
        }
        for (int j = 0; j < top_k; ++j) {
            int best = 0;
            float bv = lg[0];
            for (int e = 1; e < experts; ++e) {
                if (lg[e] > bv) {
                    bv = lg[e];
                    best = e;
                }
            }
            vt[j] = bv;
            it[j] = best;
            lg[best] = -1.0f;
        }
        float vs = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            vs += vt[j];
        }
        float invv = 1.0f / vs;
        for (int j = 0; j < top_k; ++j) {
            vt[j] = vt[j] * invv * per_expert[it[j]];
        }
    }
}

void gemma_router_mt(const float *x, const float *scale, const float *proj,
                     const float *per_expert, int hidden, int experts, int top_k,
                     float eps, float hscale, float *val, int *idx, int tokens)
{
    float *r = (float *)malloc((size_t)tokens * (size_t)hidden * sizeof(float));
    float *logits = (float *)malloc((size_t)tokens * (size_t)experts * sizeof(float));
    #pragma omp parallel
    gemma_router_mt_body(x, scale, proj, per_expert, hidden, experts, top_k, eps,
                         hscale, val, idx, tokens, r, logits);
    free(r);
    free(logits);
}

/* ---------- the query, key, and value norm of one layer ----------
 * One attention layer applies RMSNorm to the query, the key, and the value.
 * Three NumPy calls then pay the call cost three times. This kernel applies
 * all three in one parallel loop. The value has no weight. The rows are
 * (tokens * heads, head_dim) and the data changes in place.
 */
static void gemma_qkv_norm_body(float *q, const float *q_w, int q_rows,
                                float *k, const float *k_w, int k_rows,
                                float *v, int v_rows, int head_dim, float eps)
{
    int total = q_rows + k_rows + v_rows;
    #pragma omp for schedule(static)
    for (int r = 0; r < total; ++r) {
        float *x;
        const float *w;
        if (r < q_rows) {
            x = q + (size_t)r * (size_t)head_dim;
            w = q_w;
        } else if (r < q_rows + k_rows) {
            x = k + (size_t)(r - q_rows) * (size_t)head_dim;
            w = k_w;
        } else {
            x = v + (size_t)(r - q_rows - k_rows) * (size_t)head_dim;
            w = NULL;
        }
        float ss = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            ss += x[i] * x[i];
        }
        float s = 1.0f / sqrtf(ss / (float)head_dim + eps);
        if (w != NULL) {
            for (int i = 0; i < head_dim; ++i) {
                /* The same order as gemma_rms_norm, so the two give the same
                 * bits. */
                x[i] = x[i] * s * w[i];
            }
        } else {
            for (int i = 0; i < head_dim; ++i) {
                x[i] *= s;
            }
        }
    }
}

void gemma_qkv_norm(float *q, const float *q_w, int q_rows,
                    float *k, const float *k_w, int k_rows,
                    float *v, int v_rows, int head_dim, float eps)
{
    int total = q_rows + k_rows + v_rows;
    #pragma omp parallel if(total >= 8)
    gemma_qkv_norm_body(q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps);
}

/* ---------- rotary position embedding ----------
 * Apply RoPE to the query and the key in place. The value does not turn. The
 * cos and sin tables have one row for each token and the full head width.
 * The two halves of the head use the first half of the table, because the
 * table joins the frequency vector to itself.
 */
/* The numpy path computes x * cos + rotate_half(x) * sin with two multiplies
 * and one add. A fused multiply and add rounds one time fewer, so the two
 * forms differ in the last bit. The model is sensitive to that difference
 * over a long context, and the two norm kernels of this file agree with the
 * numpy path exactly. Turn the contraction off here as well. */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("-ffp-contract=off")))
#endif
static void gemma_rope_body(float *q, int q_rows, int q_heads,
                            float *k, int k_rows, int k_heads,
                            const float *cos, const float *sin, int head_dim)
{
    int d = head_dim / 2;
    int total = q_rows + k_rows;
    #pragma omp for schedule(static)
    for (int r = 0; r < total; ++r) {
        float *x;
        int tok;
        if (r < q_rows) {
            x = q + (size_t)r * (size_t)head_dim;
            tok = r / q_heads;
        } else {
            int j = r - q_rows;
            x = k + (size_t)j * (size_t)head_dim;
            tok = j / k_heads;
        }
        const float *c = cos + (size_t)tok * (size_t)head_dim;
        const float *s = sin + (size_t)tok * (size_t)head_dim;
        for (int i = 0; i < d; ++i) {
            float a = x[i];
            float b = x[i + d];
            x[i] = a * c[i] - b * s[i];
            x[i + d] = b * c[i] + a * s[i];
        }
    }
}

void gemma_rope(float *q, int q_rows, int q_heads,
                float *k, int k_rows, int k_heads,
                const float *cos, const float *sin, int head_dim)
{
    int total = q_rows + k_rows;
    #pragma omp parallel if(total >= 8)
    gemma_rope_body(q, q_rows, q_heads, k, k_rows, k_heads, cos, sin, head_dim);
}

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

/* Return the dot product of two float32 rows. */
#if GEMMA_X86 && defined(__AVX512F__)
static inline float dot_f32_f32(const float *a, const float *b, int n)
{
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= n; k += 32) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k + 16),
                             _mm512_loadu_ps(b + k + 16), a1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; k + 16 <= n; k += 16) {
        s += _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_loadu_ps(a + k),
                                                _mm512_loadu_ps(b + k)));
    }
    for (; k < n; ++k) {
        s += a[k] * b[k];
    }
    return s;
}
#elif GEMMA_X86
static inline float dot_f32_f32(const float *a, const float *b, int n)
{
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8),
                             _mm256_loadu_ps(b + k + 8), a1);
    }
    __m256 s = _mm256_add_ps(a0, a1);
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float out = _mm_cvtss_f32(r);
    for (; k < n; ++k) {
        out += a[k] * b[k];
    }
    return out;
}
#else
static inline float dot_f32_f32(const float *a, const float *b, int n)
{
    float s = 0.0f;
    for (int k = 0; k < n; ++k) {
        s += a[k] * b[k];
    }
    return s;
}
#endif

/* The row stride of a float32 KV cache from its head stride hs: position-
 * major (positions, kv heads, hd) has hs == hd and rows of kvh hd values;
 * head-major (kv heads, positions, hd) rows of hd (gpu.cu kv_rs). */
static inline size_t gp_kv_rs(size_t hs, int kvh, int hd)
{
    return hs == (size_t)hd ? (size_t)kvh * (size_t)hd : (size_t)hd;
}

/* One query token against a float32 key and value cache.
 *
 * q is (q_heads, head_dim). k and v are (kv_heads, n, head_dim), which is the
 * layout of the cache of the E4B model. The key and the value of one head lie
 * together, so the kernel reads them in order and copies nothing. A layout of
 * (n, kv_heads, head_dim) costs 2.4 times in the dot product, because the
 * hardware prefetch then jumps over the other head for each key. The batched
 * matrix multiply that the model used before also needs a transpose of the
 * key for each layer, and that copy costs 1.8 ms for a global layer at a
 * context of 512 tokens.
 *
 * scores is a scratch array of (q_heads, n) float32 values. pos is the
 * position of the query and base is the position of key zero. A window of
 * zero turns the sliding window off. out is (q_heads, head_dim).
 *
 * k_head_stride and v_head_stride give the distance between two heads, in
 * values. The cache holds a buffer that is larger than the part in use, so
 * the distance is not n * head_dim. The kernel reads the part in place. A
 * copy of the key and the value for each layer costs 98 MB for a token at a
 * context of 512.
 *
 * The parallel loop covers the query heads. A decode step of this model has 8
 * of them, so the number of busy threads is 8 and not 18.
 */
static void gemma_attn_decode_f32_body(const float *q, const float *k, const float *v,
                                       float *scores, float *out,
                                       int q_heads, int kv_heads, int head_dim, int n,
                                       long k_head_stride, long v_head_stride,
                                       long k_row, long v_row,
                                       int pos, int base, int window)
{
    if (n <= 0 || head_dim <= 0 || q_heads < kv_heads) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    #pragma omp for schedule(static)
    for (int h = 0; h < q_heads; ++h) {
        const int kv = h / n_rep;
        const float *qh = q + (size_t)h * (size_t)head_dim;
        const float *kh = k + (size_t)kv * (size_t)k_head_stride;
        const float *vh = v + (size_t)kv * (size_t)v_head_stride;
        float *sc = scores + (size_t)h * (size_t)n;
        float *oh = out + (size_t)h * (size_t)head_dim;
        for (int j = 0; j < n; ++j) {
            const int kp = base + j;
            if (kp > pos || (window > 0 && pos - kp >= window)) {
                sc[j] = -INFINITY;
                continue;
            }
            sc[j] = dot_f32_f32(kh + (size_t)j * (size_t)k_row, qh, head_dim);
        }
        float m = -INFINITY;
        for (int j = 0; j < n; ++j) {
            if (sc[j] > m) {
                m = sc[j];
            }
        }
        if (m == -INFINITY) {
            m = 0.0f;
        }
        float l = 0.0f;
        for (int j = 0; j < n; ++j) {
            float e = expf(sc[j] - m);
            sc[j] = e;
            l += e;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        int d = 0;
#if GEMMA_X86 && defined(__AVX512F__)
        for (; d + 16 <= head_dim; d += 16) {
            _mm512_storeu_ps(oh + d, _mm512_setzero_ps());
        }
#elif GEMMA_X86
        for (; d + 8 <= head_dim; d += 8) {
            _mm256_storeu_ps(oh + d, _mm256_setzero_ps());
        }
#endif
        for (; d < head_dim; ++d) {
            oh[d] = 0.0f;
        }
        for (int j = 0; j < n; ++j) {
            const float p = sc[j] * inv;
            if (p == 0.0f) {
                continue;
            }
            const float *vp = vh + (size_t)j * (size_t)v_row;
            d = 0;
#if GEMMA_X86 && defined(__AVX512F__)
            const __m512 pv = _mm512_set1_ps(p);
            for (; d + 16 <= head_dim; d += 16) {
                _mm512_storeu_ps(oh + d, _mm512_fmadd_ps(
                    pv, _mm512_loadu_ps(vp + d), _mm512_loadu_ps(oh + d)));
            }
#elif GEMMA_X86
            const __m256 pv = _mm256_set1_ps(p);
            for (; d + 8 <= head_dim; d += 8) {
                _mm256_storeu_ps(oh + d, _mm256_fmadd_ps(
                    pv, _mm256_loadu_ps(vp + d), _mm256_loadu_ps(oh + d)));
            }
#endif
            for (; d < head_dim; ++d) {
                oh[d] += p * vp[d];
            }
        }
    }
}

/* A float32 cache by its head strides: the row stride from them (gp_kv_rs:
 * position-major, the form of all the caches now, or head-major). */
void gemma_attn_decode_f32(const float *q, const float *k, const float *v,
                           float *scores, float *out,
                           int q_heads, int kv_heads, int head_dim, int n,
                           long k_head_stride, long v_head_stride,
                           int pos, int base, int window)
{
    long kr = (long)gp_kv_rs((size_t)k_head_stride, kv_heads, head_dim);
    long vr = (long)gp_kv_rs((size_t)v_head_stride, kv_heads, head_dim);
    #pragma omp parallel
    gemma_attn_decode_f32_body(q, k, v, scores, out, q_heads, kv_heads, head_dim, n,
                               k_head_stride, v_head_stride, kr, vr, pos, base, window);
}

/* The cache of Model keeps one row for each position: (rows, kv_heads,
 * head_dim). The head stride is head_dim and the row stride is
 * kv_heads * head_dim. A decode step with the float cache uses this kernel. */
void gemma_attn_decode_f32s(const float *q, const float *k, const float *v,
                            float *scores, float *out,
                            int q_heads, int kv_heads, int head_dim, int n,
                            int pos, int base, int window)
{
    #pragma omp parallel
    gemma_attn_decode_f32_body(q, k, v, scores, out, q_heads, kv_heads, head_dim, n,
                               head_dim, head_dim, (long)kv_heads * head_dim,
                               (long)kv_heads * head_dim, pos, base, window);
}

/* The int8 attention for one query token. The attribute keeps the AVX2
 * version, which the baseline build needs. */
__attribute__((target("avx2,fma")))
/* One query head of the fused decode attention over n key rows. */
static inline void attn_decode_head(const int8_t *qh, const float *qsh,
                                    const int8_t *kq, const float *ks,
                                    const int8_t *vq, const float *vs,
                                    float *sc, float *oh, int kv, int head_dim,
                                    size_t kv_stride, size_t ks_stride, int n)
{
    int g = head_dim / 32;
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

static void gemma_attn_decode_body(const int8_t *qq, const float *qs,
                       const int8_t *kq, const float *ks,
                       const int8_t *vq, const float *vs,
                       float *scores, float *out,
                       int q_heads, int kv_heads, int head_dim, int n)
{
    int g = head_dim / 32;
    int n_rep = q_heads / kv_heads;
    size_t kv_stride = (size_t)kv_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)kv_heads * (size_t)g;
    #pragma omp for schedule(static)
    for (int h = 0; h < q_heads; ++h) {
        attn_decode_head(qq + (size_t)h * (size_t)head_dim, qs + (size_t)h * (size_t)g,
                         kq, ks, vq, vs, scores + (size_t)h * (size_t)n,
                         out + (size_t)h * (size_t)head_dim, h / n_rep, head_dim,
                         kv_stride, ks_stride, n);
    }
}

void gemma_attn_decode(const int8_t *qq, const float *qs,
                       const int8_t *kq, const float *ks,
                       const int8_t *vq, const float *vs,
                       float *scores, float *out,
                       int q_heads, int kv_heads, int head_dim, int n)
{
    #pragma omp parallel
    gemma_attn_decode_body(qq, qs, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, n);
}

/* ---------- the int16 copy of the key and value cache ----------
 *
 * The cache keeps a copy of each key and value as int16, with one float32
 * scale for each group of 32 values. The query stays float32. The error of
 * the attention output is then about 4e-5 of its size. An int8 copy with an
 * int8 query gave about 1e-2, which moved a logit by up to 2.6. The int16
 * copy reads 2.125 bytes for each value, against 4 for the float cache.
 *
 * The quantizer is plain C with lrintf, which rounds to the nearest even
 * value. Every caller uses it, so the Python path and the program give the
 * same bits. */
static inline float gemma_quant_group32_i16(const float *x, int16_t *q)
{
    float amax = 0.0f;
    for (int k = 0; k < 32; ++k) {
        float a = fabsf(x[k]);
        if (a > amax) {
            amax = a;
        }
    }
    float sc = amax > 0.0f ? amax / 32767.0f : 1e-12f;
    for (int k = 0; k < 32; ++k) {
        long v = lrintf(x[k] / sc);
        if (v > 32767) {
            v = 32767;
        }
        if (v < -32767) {
            v = -32767;
        }
        q[k] = (int16_t)v;
    }
    return sc;
}

/* Quantize groups of 32 values to int16. x holds groups * 32 values. */
static void gemma_quantize_i16_groups_body(const float *x, int16_t *q, float *s,
                                           long groups)
{
    #pragma omp for schedule(static)
    for (long g = 0; g < groups; ++g) {
        s[g] = gemma_quant_group32_i16(x + (size_t)g * 32, q + (size_t)g * 32);
    }
}

void gemma_quantize_i16_groups(const float *x, int16_t *q, float *s, long groups)
{
    #pragma omp parallel if(groups >= 1024)
    gemma_quantize_i16_groups_body(x, q, s, groups);
}

/* The int8 cache (NP_GEMMA_KV_INT8): a group of 32 values gets the scale
 * max |x| / 127, and each value is x / scale rounded to the nearest (ties
 * to even, lrintf), as the int16 form. */
static inline float gemma_quant_group32_i8(const float *x, int8_t *q)
{
    float amax = 0.0f;
    for (int k = 0; k < 32; ++k) {
        float a = fabsf(x[k]);
        if (a > amax) {
            amax = a;
        }
    }
    float sc = amax > 0.0f ? amax / 127.0f : 1e-12f;
    for (int k = 0; k < 32; ++k) {
        long v = lrintf(x[k] / sc);
        v = v > 127 ? 127 : (v < -127 ? -127 : v);
        q[k] = (int8_t)v;
    }
    return sc;
}

/* The float32 values of groups of 32 int8 values times the scale of the
 * group (the int8 cache: KVCache.read). */
void gemma_dequantize_i8_groups(const int8_t *q, const float *s, float *out, long groups)
{
    #pragma omp parallel for schedule(static) if(groups >= 1024)
    for (long g = 0; g < groups; ++g) {
        for (int k = 0; k < 32; ++k) {
            out[(size_t)g * 32 + k] = (float)q[(size_t)g * 32 + k] * s[g];
        }
    }
}

void gemma_quantize_i8_groups(const float *x, int8_t *q, float *s, long groups)
{
    #pragma omp parallel for schedule(static) if(groups >= 1024)
    for (long g = 0; g < groups; ++g) {
        s[g] = gemma_quant_group32_i8(x + (size_t)g * 32, q + (size_t)g * 32);
    }
}

/* The TQ6 cache of the Qwen models (np_gemma/tq6.py has the description,
 * csrc/tq6_tables.h the tables). A group of 32 values: the signs of
 * TQ6_SIGNS, the Walsh-Hadamard transform divided by sqrt(32), then the L2
 * norm (float32) and an index of 6 bits into tq6_cb for each value. The 24
 * bytes of a group: byte j < 16 has the low 4 bits of index j and of index
 * j + 16; byte 16 + j has the high 2 bits of indices j, j + 8, j + 16,
 * j + 24. */
#include "tq6_tables.h"
static const float tq6_cb[64] = TQ6_CODEBOOK;
static const float tq6_edges[63] = TQ6_EDGES;

/* The rotation of 32 values in place (inverse 0), or its inverse. */
static inline void tq6_rot32(float *x, int inverse)
{
    const float r = 0.17677669529663687f;    /* 1 / sqrt(32) */
    if (!inverse) {
        for (int i = 0; i < 32; ++i) {
            x[i] = (TQ6_SIGNS >> i) & 1u ? -x[i] : x[i];
        }
    }
    for (int h = 1; h < 32; h *= 2) {
        for (int i = 0; i < 32; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 32; ++i) {
        x[i] *= r;
        if (inverse && ((TQ6_SIGNS >> i) & 1u)) {
            x[i] = -x[i];
        }
    }
}

/* Quantize a group of 32 values to 24 bytes; return the norm. No fused
 * multiply-add: the GPU (k_kv_write_tq) gives the same bits. */
__attribute__((optimize("fp-contract=off")))
static float tq6_quant_group(const float *x, uint8_t *b)
{
    float y[32];
    memcpy(y, x, sizeof(y));
    tq6_rot32(y, 0);
    /* the sum of the squares in the order of the warp of the GPU (a sum
     * with xor shuffles), so both give the same bits */
    float s[32];
    for (int i = 0; i < 32; ++i) {
        s[i] = y[i] * y[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        for (int i = 0; i < off; ++i) {
            s[i] = s[i] + s[i + off];
        }
    }
    float nrm = sqrtf(s[0]);
    float inv = 1.0f / (nrm > 1e-30f ? nrm : 1e-30f);
    uint8_t idx[32];
    for (int i = 0; i < 32; ++i) {
        float u = y[i] * inv;
        /* the count of the edges below u (numpy searchsorted, side left) */
        int lo = 0, hi = 63;
        while (lo < hi) {
            int m = (lo + hi) / 2;
            if (tq6_edges[m] < u) {
                lo = m + 1;
            } else {
                hi = m;
            }
        }
        idx[i] = (uint8_t)lo;
    }
    for (int j = 0; j < 16; ++j) {
        b[j] = (uint8_t)((idx[j] & 15) | ((idx[j + 16] & 15) << 4));
    }
    for (int j = 0; j < 8; ++j) {
        b[16 + j] = (uint8_t)((idx[j] >> 4) | ((idx[j + 8] >> 4) << 2) |
                              ((idx[j + 16] >> 4) << 4) | ((idx[j + 24] >> 4) << 6));
    }
    return nrm;
}

/* The 32 rotated values of a group: norm * tq6_cb[index]. */
static inline __attribute__((always_inline)) void tq6_deq32(const uint8_t *b, float nrm,
                                                            float *out)
{
#if GEMMA_X86
    __m128i x = _mm_loadu_si128((const __m128i *)b);
    __m128i m15 = _mm_set1_epi8(15);
    __m256i nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(x, 4), m15),
                                   _mm_and_si128(x, m15));
    int64_t h8;
    memcpy(&h8, b + 16, 8);
    __m256i hb = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(h8),
                                                    _mm256_set_epi64x(6, 4, 2, 0)),
                                  _mm256_set1_epi8(3));
    __m256i idx = _mm256_or_si256(nib, _mm256_slli_epi16(hb, 4));
    __m128i i0 = _mm256_castsi256_si128(idx), i1 = _mm256_extracti128_si256(idx, 1);
#if defined(__AVX512F__)
    __m512 t0 = _mm512_loadu_ps(tq6_cb), t1 = _mm512_loadu_ps(tq6_cb + 16);
    __m512 t2 = _mm512_loadu_ps(tq6_cb + 32), t3 = _mm512_loadu_ps(tq6_cb + 48);
    __m512 s5 = _mm512_set1_ps(nrm);
    __m512i k32 = _mm512_set1_epi32(32);
    for (int h = 0; h < 2; ++h) {
        __m512i iv = _mm512_cvtepu8_epi32(h ? i1 : i0);
        __mmask16 up = _mm512_test_epi32_mask(iv, k32);
        __m512 v = _mm512_mask_blend_ps(up, _mm512_permutex2var_ps(t0, iv, t1),
                                        _mm512_permutex2var_ps(t2, iv, t3));
        _mm512_storeu_ps(out + 16 * h, _mm512_mul_ps(v, s5));
    }
#else
    /* the codebook is symmetric: index i >= 32 has the value tq6_cb[i] and
     * index i < 32 the value -tq6_cb[63 - i]. Four permutes of the 32
     * positive values and two blends find the value, then the sign. */
    __m256 s8 = _mm256_set1_ps(nrm);
    __m256 t0 = _mm256_loadu_ps(tq6_cb + 32), t1 = _mm256_loadu_ps(tq6_cb + 40);
    __m256 t2 = _mm256_loadu_ps(tq6_cb + 48), t3 = _mm256_loadu_ps(tq6_cb + 56);
    for (int h = 0; h < 4; ++h) {
        __m128i src = h < 2 ? i0 : i1;
        __m256i iv = _mm256_cvtepu8_epi32(h % 2 ? _mm_srli_si128(src, 8) : src);
        /* t: 0 for i >= 32, all ones for i < 32; m: the index of the positive value */
        __m256i t = _mm256_sub_epi32(_mm256_srli_epi32(iv, 5), _mm256_set1_epi32(1));
        __m256i m = _mm256_xor_si256(iv, t);
        __m256 b3 = _mm256_castsi256_ps(_mm256_slli_epi32(m, 28));
        __m256 lo = _mm256_blendv_ps(_mm256_permutevar8x32_ps(t0, m),
                                     _mm256_permutevar8x32_ps(t1, m), b3);
        __m256 hi = _mm256_blendv_ps(_mm256_permutevar8x32_ps(t2, m),
                                     _mm256_permutevar8x32_ps(t3, m), b3);
        __m256 v = _mm256_blendv_ps(lo, hi, _mm256_castsi256_ps(_mm256_slli_epi32(m, 27)));
        v = _mm256_xor_ps(v, _mm256_castsi256_ps(_mm256_slli_epi32(t, 31)));
        _mm256_storeu_ps(out + 8 * h, _mm256_mul_ps(v, s8));
    }
#endif
#else
    for (int j = 0; j < 32; ++j) {
        int lo4 = j < 16 ? b[j] & 15 : b[j - 16] >> 4;
        int hi2 = (b[16 + j % 8] >> (2 * (j / 8))) & 3;
        out[j] = nrm * tq6_cb[lo4 | (hi2 << 4)];
    }
#endif
}

/* The NumPy forms of the tests: groups of 32 values. */
void gemma_tq6_quantize(const float *x, uint8_t *b, float *nrm, long groups)
{
    #pragma omp parallel for schedule(static) if(groups >= 1024)
    for (long g = 0; g < groups; ++g) {
        nrm[g] = tq6_quant_group(x + (size_t)g * 32, b + (size_t)g * 24);
    }
}

void gemma_tq6_dequantize_rotated(const uint8_t *b, const float *nrm, float *out, long groups)
{
    for (long g = 0; g < groups; ++g) {
        tq6_deq32(b + (size_t)g * 24, nrm[g], out + (size_t)g * 32);
    }
}

void gemma_tq6_rotate(float *x, long groups, int inverse)
{
    #pragma omp parallel for schedule(static) if(groups >= 1024)
    for (long g = 0; g < groups; ++g) {
        tq6_rot32(x + (size_t)g * 32, inverse);
    }
}

/* The float32 values of groups of 32 int16 values: q times the scale of
 * the group, as numpy gives them (KVCache.write_q). */
static void gemma_dequantize_i16_groups_body(const int16_t *q, const float *s, float *out,
                                             long groups)
{
    #pragma omp for schedule(static)
    for (long g = 0; g < groups; ++g) {
        const int16_t *qg = q + (size_t)g * 32;
        float *og = out + (size_t)g * 32;
        float sc = s[g];
        for (int k = 0; k < 32; ++k) {
            og[k] = (float)qg[k] * sc;
        }
    }
}

void gemma_dequantize_i16_groups(const int16_t *q, const float *s, float *out, long groups)
{
    #pragma omp parallel if(groups >= 1024)
    gemma_dequantize_i16_groups_body(q, s, out, groups);
}

/* The dot product of 32 float32 values and 32 int16 values. */
static inline float dot32_f32_i16(const float *a, const int16_t *b)
{
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 b0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
        _mm256_loadu_si256((const __m256i *)b)));
    __m512 b1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
        _mm256_loadu_si256((const __m256i *)(b + 16))));
    __m512 acc = _mm512_mul_ps(_mm512_loadu_ps(a), b0);
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + 16), b1, acc);
    return _mm512_reduce_add_ps(acc);
#elif GEMMA_X86
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < 32; i += 8) {
        __m256 bv = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
            _mm_loadu_si128((const __m128i *)(b + i))));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), bv, acc);
    }
    return hsum256_ps(acc);
#else
    float acc = 0.0f;
    for (int i = 0; i < 32; ++i) {
        acc += a[i] * (float)b[i];
    }
    return acc;
#endif
}

/* One query head over n rows of the int16 cache. qh is float32. rows (or
 * null for rows 0 to n - 1) gives the rows to read (QSA of qwen4exp). */
static inline void attn_i16_head_rows(const float *qh, const int16_t *kq, const float *ks,
                                      const int16_t *vq, const float *vs, float *sc,
                                      float *oh, int kv, int head_dim, size_t kv_stride,
                                      size_t ks_stride, int n, const int32_t *rows)
{
    int g = head_dim / 32;
    for (int j = 0; j < n; ++j) {
        size_t r = rows ? (size_t)rows[j] : (size_t)j;
        const int16_t *kp = kq + r * kv_stride + (size_t)kv * (size_t)head_dim;
        const float *ksp = ks + r * ks_stride + (size_t)kv * (size_t)g;
        float s = 0.0f;
        for (int gg = 0; gg < g; ++gg) {
            s += dot32_f32_i16(qh + (size_t)gg * 32, kp + (size_t)gg * 32) * ksp[gg];
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
    for (int d = 0; d < head_dim; ++d) {
        oh[d] = 0.0f;
    }
    for (int j = 0; j < n; ++j) {
        float p = sc[j] * inv;
        size_t r = rows ? (size_t)rows[j] : (size_t)j;
        const int16_t *vp = vq + r * kv_stride + (size_t)kv * (size_t)head_dim;
        const float *vsp = vs + r * ks_stride + (size_t)kv * (size_t)g;
        for (int gg = 0; gg < g; ++gg) {
            const int16_t *vpp = vp + (size_t)gg * 32;
            float *op = oh + (size_t)gg * 32;
#if GEMMA_X86 && defined(__AVX512F__)
            __m512 pv = _mm512_set1_ps(p * vsp[gg]);
            for (int i = 0; i < 32; i += 16) {
                __m512 vf = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                    _mm256_loadu_si256((const __m256i *)(vpp + i))));
                _mm512_storeu_ps(op + i, _mm512_fmadd_ps(pv, vf, _mm512_loadu_ps(op + i)));
            }
#elif GEMMA_X86
            __m256 pv = _mm256_set1_ps(p * vsp[gg]);
            for (int i = 0; i < 32; i += 8) {
                __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm_loadu_si128((const __m128i *)(vpp + i))));
                _mm256_storeu_ps(op + i, _mm256_fmadd_ps(pv, vf, _mm256_loadu_ps(op + i)));
            }
#else
            float pv = p * vsp[gg];
            for (int i = 0; i < 32; ++i) {
                op[i] += pv * (float)vpp[i];
            }
#endif
        }
    }
}

static inline void attn_i16_head(const float *qh, const int16_t *kq, const float *ks,
                                 const int16_t *vq, const float *vs, float *sc,
                                 float *oh, int kv, int head_dim, size_t kv_stride,
                                 size_t ks_stride, int n)
{
    attn_i16_head_rows(qh, kq, ks, vq, vs, sc, oh, kv, head_dim, kv_stride, ks_stride, n, NULL);
}

/* The attention of one float32 query over n rows of the int16 cache. q is
 * (q_heads, head_dim). scores holds q_heads * n values. */
/* The same with the row strides of the cache as arguments. A part of a
 * program in parts (SPLIT_PLAN.md) runs only some heads. Its cache pointers
 * then point at its first head, and a row of the cache holds more heads than
 * the part runs. */
static void gemma_attn_decode_i16_s_body(const float *q, const int16_t *kq, const float *ks,
                                         const int16_t *vq, const float *vs, float *scores,
                                         float *out, int q_heads, int kv_heads,
                                         int head_dim, int n, size_t kv_stride,
                                         size_t ks_stride)
{
    int n_rep = q_heads / kv_heads;
    #pragma omp for schedule(static)
    for (int h = 0; h < q_heads; ++h) {
        attn_i16_head(q + (size_t)h * (size_t)head_dim, kq, ks, vq, vs,
                      scores + (size_t)h * (size_t)n, out + (size_t)h * (size_t)head_dim,
                      h / n_rep, head_dim, kv_stride, ks_stride, n);
    }
}

static void gemma_attn_decode_i16_body(const float *q, const int16_t *kq, const float *ks,
                                       const int16_t *vq, const float *vs, float *scores,
                                       float *out, int q_heads, int kv_heads,
                                       int head_dim, int n)
{
    int n_rep = q_heads / kv_heads;
    size_t kv_stride = (size_t)kv_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)kv_heads * (size_t)(head_dim / 32);
    #pragma omp for schedule(static)
    for (int h = 0; h < q_heads; ++h) {
        attn_i16_head(q + (size_t)h * (size_t)head_dim, kq, ks, vq, vs,
                      scores + (size_t)h * (size_t)n, out + (size_t)h * (size_t)head_dim,
                      h / n_rep, head_dim, kv_stride, ks_stride, n);
    }
}

static int gemma_attn_split_i16_body(const float *q, const int16_t *kq, const float *ks,
                                     const int16_t *vq, const float *vs, float *scores,
                                     float *out, int q_heads, int kv_heads, int head_dim,
                                     const int *lo, const int *n, int nmax, int tokens);

/* The split form when it fits (the attention of the MTP drafter over the
 * target cache), else one head for each thread. */
void gemma_attn_decode_i16(const float *q, const int16_t *kq, const float *ks,
                           const int16_t *vq, const float *vs, float *scores,
                           float *out, int q_heads, int kv_heads, int head_dim, int n)
{
    #pragma omp parallel
    {
        if (!gemma_attn_split_i16_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                       head_dim, NULL, &n, n, 1)) {
            gemma_attn_decode_i16_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                       head_dim, n);
        }
    }
}

/* The same for a group of queries. Query t reads the rows lo[t] to
 * lo[t] + n[t] - 1. A decode step of that query reads the same rows. q is
 * (tokens, q_heads, head_dim). scores holds nmax values for each thread
 * (the threads of the region; a larger buffer also serves). */
static void gemma_attn_decode_i16_mt_body(const float *q, const int16_t *kq,
                                          const float *ks, const int16_t *vq,
                                          const float *vs, float *scores, float *out,
                                          int q_heads, int kv_heads, int head_dim,
                                          const int *lo, const int *n, int nmax,
                                          int tokens)
{
    int n_rep = q_heads / kv_heads;
    size_t kv_stride = (size_t)kv_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)kv_heads * (size_t)(head_dim / 32);
    int total = tokens * q_heads;
    #pragma omp for schedule(static)
    for (int u = 0; u < total; ++u) {
        int t = u / q_heads;
        int h = u % q_heads;
        size_t r = (size_t)lo[t];
        attn_i16_head(q + (size_t)u * (size_t)head_dim, kq + r * kv_stride,
                      ks + r * ks_stride, vq + r * kv_stride, vs + r * ks_stride,
                      scores + (size_t)omp_get_thread_num() * (size_t)nmax,
                      out + (size_t)u * (size_t)head_dim, h / n_rep, head_dim, kv_stride,
                      ks_stride, n[t]);
    }
}

/* ---- the decode attention over the int16 cache, split by keys ----
 *
 * As the attention of a decode step on the GPU (k_attn_fd): an item is a
 * query token, a key head, and a chunk of AS_LEN(n) keys. It does all the
 * query heads of that key head, so it reads each key and value row once.
 * It converts a row to float32 times the scale of each group of 32 values
 * once (as_row), and each query head then takes one dot product with one
 * sum across the lanes (AVX2 summed each group of 32 before). The item
 * writes, for each of its query heads, the sum of exp(score - m) times the
 * value rows, m, and l (the sum of exp(score - m)) to the scratch. A second
 * pass joins the chunks of each query head, in the order of the chunks.
 *
 * The chunk length depends only on n, so a query of a group gives the bits
 * of the same query in a step. Many items keep all the threads busy: the
 * 16 query heads of the 26B over 6 threads left two threads idle. */
#define AS_MAXC 32
#define AS_MINLEN 64
#define AS_LEN(n) ((n) <= AS_MINLEN * AS_MAXC ? AS_MINLEN : ((n) + AS_MAXC - 1) / AS_MAXC)
/* the query values of the heads of an item: 8192 takes the 16 query heads of
 * 512 values of a global layer of the 12B (one key head) */
#define AS_MAXQ 8192

/* The scratch of the split attention, one set for each team: the team of
 * one program (slot 0), or the team of each part of gemma_run_parts (slot
 * part + 1), because the parts run it at the same time. */
#define AS_SLOTS 9
typedef struct {
    float *parts;
    size_t parts_n;
    float *isc;                       /* the scores of the items, by thread */
    size_t isc_n;
    int16_t *qq;                      /* the int16 queries (AVX2) */
    float *sq;
    size_t q_n, sq_n;
} as_scratch;
static as_scratch as_scr[AS_SLOTS];

static as_scratch *as_team(void)
{
    int slot = omp_get_level() >= 2 ? omp_get_ancestor_thread_num(1) + 1 : 0;
    return &as_scr[slot >= 0 && slot < AS_SLOTS ? slot : 0];
}

static int as_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_ATTN_SPLIT");
        on = !(v && v[0] == '0');
    }
    return on;
}

#if GEMMA_X86
/* 8 values as floats; 16 values as int16 lanes. */
static inline __attribute__((always_inline)) __m256 as_load8f(const void *p, const int I8)
{
    return _mm256_cvtepi32_ps(I8 ? _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)p))
                                 : _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)p)));
}

static inline __attribute__((always_inline)) __m256i as_load16(const void *p, const int I8)
{
    return I8 ? _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)p))
              : _mm256_loadu_si256((const __m256i *)p);
}
#endif

/* The cache values are int16 (I8 0), int8 (I8 1, NP_GEMMA_KV_INT8), or TQ6
 * (I8 2, 3 bytes for 4 values, the rotated form; tq6_deq32): the helpers
 * below take the element size as a constant, and the bodies
 * (gemma_attn_split_i16_body, gemma_attn_split_i8_body, ...) inline them. */
#define AS_ATE(p, idx, i8) ((const void *)((const char *)(p) + \
    ((i8) == 2 ? (size_t)(idx) * 3 / 4 : (size_t)(idx) * ((i8) ? 1 : 2))))
#define AS_AT(p, idx) AS_ATE(p, idx, I8)
#define AS_EL(p, i) (I8 ? (float)((const int8_t *)(p))[i] : (float)((const int16_t *)(p))[i])

/* out[i] = (float)row[i] * scale of the group of i, for hd values. */
static inline __attribute__((always_inline)) void as_row(const void *row, const float *sc, int hd,
                                                         float *out, const int I8)
{
    if (I8 == 2) {
        for (int gg = 0; gg < hd / 32; ++gg) {
            tq6_deq32((const uint8_t *)row + (size_t)gg * 24, sc[gg], out + (size_t)gg * 32);
        }
        return;
    }
    for (int gg = 0; gg < hd / 32; ++gg) {
        const void *rp = AS_AT(row, (size_t)gg * 32);
        float *op = out + (size_t)gg * 32;
#if GEMMA_X86 && defined(__AVX512F__)
        __m512 s5 = _mm512_set1_ps(sc[gg]);
        for (int i = 0; i < 32; i += 16) {
            __m512 v = _mm512_cvtepi32_ps(I8 ?
                _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)AS_AT(rp, i))) :
                _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)AS_AT(rp, i))));
            _mm512_storeu_ps(op + i, _mm512_mul_ps(v, s5));
        }
#elif GEMMA_X86
        __m256 s8 = _mm256_set1_ps(sc[gg]);
        for (int i = 0; i < 32; i += 8) {
            __m256 v = as_load8f(AS_AT(rp, i), I8);
            _mm256_storeu_ps(op + i, _mm256_mul_ps(v, s8));
        }
#else
        for (int i = 0; i < 32; ++i) {
            op[i] = AS_EL(rp, i) * sc[gg];
        }
#endif
    }
}

static inline float as_dot(const float *a, const float *b, int hd)
{
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int i = 0; i < hd; i += 32) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
#elif GEMMA_X86
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int i = 0; i < hd; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
#else
    float acc = 0.0f;
    for (int i = 0; i < hd; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
#endif
}

/* o += p b, hd values. */
static inline void as_axpy(float *o, float p, const float *b, int hd)
{
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 p5 = _mm512_set1_ps(p);
    for (int i = 0; i < hd; i += 16) {
        _mm512_storeu_ps(o + i, _mm512_fmadd_ps(p5, _mm512_loadu_ps(b + i), _mm512_loadu_ps(o + i)));
    }
#elif GEMMA_X86
    __m256 p8 = _mm256_set1_ps(p);
    for (int i = 0; i < hd; i += 8) {
        _mm256_storeu_ps(o + i, _mm256_fmadd_ps(p8, _mm256_loadu_ps(b + i), _mm256_loadu_ps(o + i)));
    }
#else
    for (int i = 0; i < hd; ++i) {
        o[i] += p * b[i];
    }
#endif
}

#if GEMMA_X86 && !defined(__AVX512F__)
/* The AVX2 forms of the two passes of an item (nh query heads, cn keys).
 *
 * The scores: the queries are int16 with a scale for each group of 32
 * values (qq, sq: as_quant_q), and vpmaddwd multiplies them with the int16
 * keys. Each vpmaddwd result (8 sums of 2 products) goes to float on its
 * own, so no int32 sum can overflow. A group adds into a float
 * accumulator with the scale of the key group times that of the query
 * group, and one sum across the lanes ends a key. The keys of the item are
 * loaded once for all its heads.
 *
 * The values: the item keeps the output of a block of 8 U values of each
 * head in registers over all its keys, so it loads and stores the output
 * once, not once for each key. The weight of a key times the scale of the
 * value group is one broadcast. */
/* The rows of the cache are kv_heads * head_dim values apart (4 KB for
 * the layers of the 26B with a window), so each row of an item starts a
 * new page, where the hardware prefetch stops. The passes fetch the row
 * AS_PF keys ahead. */
#define AS_PF 10
static inline __attribute__((always_inline)) void as_prefetch_row(const void *row, int hd,
                                                                  const int I8)
{
    for (int i = 0; i < hd * (I8 ? 1 : 2); i += 64) {
        _mm_prefetch((const char *)row + i, _MM_HINT_T0);
    }
}

/* exp(x) of 8 floats: the polynomial of Cephes expf (about 1 ulp), with the
 * power of two in the exponent bits. libm expf took most of the time of an
 * item outside its two passes. */
static inline __m256 as_exp256(__m256 x)
{
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f),
                                                _mm256_set1_ps(0.5f)));
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

/* sc[j] = exp(sc[j] - m), or 0 below exp(-64) (as the scalar loop); return
 * the sum. */
static inline float as_softmax_avx2(float *sc, int cn, float m)
{
    __m256 m8 = _mm256_set1_ps(m), lim = _mm256_set1_ps(-64.0f), acc = _mm256_setzero_ps();
    int j = 0;
    for (; j + 8 <= cn; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(sc + j), m8);
        __m256 ev = _mm256_and_ps(as_exp256(d), _mm256_cmp_ps(d, lim, _CMP_GE_OQ));
        _mm256_storeu_ps(sc + j, ev);
        acc = _mm256_add_ps(acc, ev);
    }
    float l = hsum256_ps(acc);
    for (; j < cn; ++j) {
        float d = sc[j] - m;
        sc[j] = d < -64.0f ? 0.0f : expf(d);
        l += sc[j];
    }
    return l;
}

/* The query of one head as int16 with one scale (14 bits: |q| <= 16383),
 * so the two vpmaddwd results of a group of 32 add as int32 with no
 * overflow (at most 2 * 2 * 16383 * 32767 < 2^31). */
static inline void as_quant_q(const float *q, int hd, int16_t *qq, float *sq)
{
    float m = 0.0f;
    for (int i = 0; i < hd; ++i) {
        float a = fabsf(q[i]);
        m = a > m ? a : m;
    }
    float inv = m > 0.0f ? 16383.0f / m : 0.0f;
    *sq = m > 0.0f ? m / 16383.0f : 0.0f;
    for (int i = 0; i < hd; i += 8) {
        __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(q + i), _mm256_set1_ps(inv)));
        __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        _mm_storeu_si128((__m128i *)(qq + i), p16);
    }
}

static inline __attribute__((always_inline)) void as_scores_avx2(const int16_t *qq, const float *sq, int nh, const void *kp,
                                  const float *ksp, int hd, float *out, size_t ostride,
                                  const int I8)
{
    __m256 acc[8];
    for (int h = 0; h < nh; ++h) {
        acc[h] = _mm256_setzero_ps();
    }
    for (int gg = 0; gg < hd / 32; ++gg) {
        __m256i k0 = as_load16(AS_AT(kp, (size_t)gg * 32), I8);
        __m256i k1 = as_load16(AS_AT(kp, (size_t)gg * 32 + 16), I8);
        __m256 kscale = _mm256_set1_ps(ksp[gg]);
        for (int h = 0; h < nh; ++h) {
            const int16_t *qh = qq + (size_t)h * hd + (size_t)gg * 32;
            __m256i d = _mm256_add_epi32(_mm256_madd_epi16(k0, _mm256_loadu_si256((const __m256i *)qh)),
                                         _mm256_madd_epi16(k1, _mm256_loadu_si256((const __m256i *)(qh + 16))));
            acc[h] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), kscale, acc[h]);
        }
    }
    for (int h = 0; h < nh; ++h) {
        out[(size_t)h * ostride] = hsum256_ps(acc[h]) * sq[h];
    }
}

/* as_scores_avx2 for a TQ6 key row: the row goes to float32 (tq6_deq32),
 * then nh float32 queries (q, rotated) read each 8 values once. */
static inline __attribute__((always_inline)) void as_scores_tq_avx2(const float *q, int nh,
                                                                   const uint8_t *kp,
                                                                   const float *ksp, int hd,
                                                                   float *out, size_t ostride)
{
    float row[1024] __attribute__((aligned(32)));
    for (int gg = 0; gg < hd / 32; ++gg) {
        tq6_deq32(kp + (size_t)gg * 24, ksp[gg], row + (size_t)gg * 32);
    }
    __m256 acc[8];
    for (int h = 0; h < nh; ++h) {
        acc[h] = _mm256_setzero_ps();
    }
    for (int i = 0; i < hd; i += 8) {
        __m256 k = _mm256_load_ps(row + i);
        for (int h = 0; h < nh; ++h) {
            acc[h] = _mm256_fmadd_ps(_mm256_loadu_ps(q + (size_t)h * hd + i), k, acc[h]);
        }
    }
    for (int h = 0; h < nh; ++h) {
        out[(size_t)h * ostride] = hsum256_ps(acc[h]);
    }
}

/* o (nh rows of hd, row stride ostride) = the sum over the cn keys of
 * p[h][j] (row stride pstride) times the value rows. */
#endif

#if GEMMA_X86 && !defined(__AVX512F__)
#define AS_VBLOCK(U)                                                                         \
    for (int d0 = 0; d0 < hd; d0 += 8 * (U)) {                                               \
        __m256 acc[8][U];                                                                    \
        for (int h = 0; h < nh; ++h)                                                         \
            for (int u = 0; u < (U); ++u)                                                    \
                acc[h][u] = _mm256_setzero_ps();                                             \
        int gg = d0 / 32;                                                                    \
        for (int j = 0; j < cn; ++j) {                                                       \
            if (d0 == 0 && j + AS_PF < cn)                                                   \
                as_prefetch_row(AS_AT(vq, (base + j + AS_PF) * kv_stride + (size_t)kv * hd), \
                                hd, I8);                                                     \
            if (!live[j]) continue;                                                          \
            size_t r = base + (size_t)j;                                                     \
            const void *vp = AS_AT(vq, r * kv_stride + (size_t)kv * hd + d0);                \
            float vsc = vs[r * ks_stride + (size_t)kv * (hd / 32) + gg];                     \
            __m256 v[U];                                                                     \
            for (int u = 0; u < (U); ++u)                                                    \
                v[u] = as_load8f(AS_AT(vp, 8 * u), I8);                                      \
            for (int h = 0; h < nh; ++h) {                                                   \
                float pw = p[(size_t)h * pstride + j];                                       \
                if (pw == 0.0f) continue;                                                    \
                __m256 b = _mm256_set1_ps(pw * vsc);                                         \
                for (int u = 0; u < (U); ++u)                                                \
                    acc[h][u] = _mm256_fmadd_ps(b, v[u], acc[h][u]);                         \
            }                                                                                \
        }                                                                                    \
        for (int h = 0; h < nh; ++h)                                                         \
            for (int u = 0; u < (U); ++u)                                                    \
                _mm256_storeu_ps(o + (size_t)h * ostride + d0 + 8 * u, acc[h][u]);           \
    }

static inline __attribute__((always_inline)) void
as_values_avx2_n(const float *p, size_t pstride, const unsigned char *live, int nh,
                 const void *vq, const float *vs, size_t kv_stride, size_t ks_stride, int kv,
                 int hd, size_t base, int cn, float *o, size_t ostride, const int I8)
{
    if (nh <= 2) {
        AS_VBLOCK(4)
    } else if (nh <= 4) {
        AS_VBLOCK(2)
    } else {
        AS_VBLOCK(1)
    }
}

/* The value pass by groups of 32 values: the rows of up to AS_VSUB keys go
 * to float32 times the scale of the group once (vf, in L1), and then each
 * pair of heads reads them: one broadcast of the weight and 4 fma for each
 * key. as_values_avx2 converted a row once for each pair of
 * heads (4 times for the 8 heads of a global layer of the 26B). */
#define AS_VSUB 128
static inline __attribute__((always_inline)) void
as_values_avx2_g(const float *p, size_t pstride, const unsigned char *live, int nh,
                 const void *vq, const float *vs, size_t kv_stride, size_t ks_stride, int kv,
                 int hd, size_t base, int cn, float *o, size_t ostride, const int I8)
{
    float vf[AS_VSUB * 32] __attribute__((aligned(32)));
    int idx[AS_VSUB];
    int nb = hd / 32;
    for (int gg = 0; gg < nb; ++gg) {
        for (int j0 = 0; j0 < cn; j0 += AS_VSUB) {
            int jn = cn - j0 < AS_VSUB ? cn - j0 : AS_VSUB;
            int m = 0;                                     /* the live keys */
            for (int j = 0; j < jn; ++j) {
                if (!live[j0 + j]) {
                    continue;
                }
                size_t r = base + (size_t)(j0 + j);
                const void *vp = AS_AT(vq, r * kv_stride + (size_t)kv * hd + (size_t)gg * 32);
                if (I8 == 2) {
                    tq6_deq32((const uint8_t *)vp, vs[r * ks_stride + (size_t)kv * nb + gg],
                              vf + (size_t)m * 32);
                } else {
                    __m256 vsc = _mm256_set1_ps(vs[r * ks_stride + (size_t)kv * nb + gg]);
                    for (int u = 0; u < 4; ++u) {
                        _mm256_store_ps(vf + (size_t)m * 32 + 8 * u,
                                        _mm256_mul_ps(as_load8f(AS_AT(vp, 8 * u), I8), vsc));
                    }
                }
                idx[m] = j0 + j;
                ++m;
            }
            for (int h0 = 0; h0 < nh; h0 += 2) {
                int two = nh - h0 >= 2;
                float *o0 = o + (size_t)h0 * ostride + (size_t)gg * 32;
                float *o1 = two ? o0 + ostride : o0;
                __m256 a0[4], a1[4];
                for (int u = 0; u < 4; ++u) {
                    a0[u] = j0 ? _mm256_loadu_ps(o0 + 8 * u) : _mm256_setzero_ps();
                    a1[u] = (j0 && two) ? _mm256_loadu_ps(o1 + 8 * u) : _mm256_setzero_ps();
                }
                const float *p0 = p + (size_t)h0 * pstride;
                const float *p1 = p + (size_t)(two ? h0 + 1 : h0) * pstride;
                for (int k = 0; k < m; ++k) {
                    const float *v = vf + (size_t)k * 32;
                    __m256 b0 = _mm256_broadcast_ss(p0 + idx[k]), b1 = _mm256_broadcast_ss(p1 + idx[k]);
                    for (int u = 0; u < 4; ++u) {
                        __m256 vv = _mm256_load_ps(v + 8 * u);
                        a0[u] = _mm256_fmadd_ps(b0, vv, a0[u]);
                        a1[u] = _mm256_fmadd_ps(b1, vv, a1[u]);
                    }
                }
                for (int u = 0; u < 4; ++u) {
                    _mm256_storeu_ps(o0 + 8 * u, a0[u]);
                    if (two) {
                        _mm256_storeu_ps(o1 + 8 * u, a1[u]);
                    }
                }
            }
        }
    }
}

static inline __attribute__((always_inline)) void
as_values_avx2(const float *p, size_t pstride, const unsigned char *live, int nh,
               const void *vq, const float *vs, size_t kv_stride, size_t ks_stride, int kv,
               int hd, size_t base, int cn, float *o, size_t ostride, const int I8)
{
    if (nh > 2 && nh <= 8) {
        as_values_avx2_g(p, pstride, live, nh, vq, vs, kv_stride, ks_stride, kv, hd, base, cn,
                         o, ostride, I8);
        return;
    }
    /* Two heads at a time: 8 accumulators of 4 vectors each fit the 16
     * registers, and the 8 converted values serve two heads. Eight heads in
     * one pass (1 vector each) took 268 us for a global layer of the 26B. */
    for (int h0 = 0; h0 < nh; h0 += 2) {
        if (nh - h0 >= 2) {
            as_values_avx2_n(p + (size_t)h0 * pstride, pstride, live, 2, vq, vs, kv_stride,
                             ks_stride, kv, hd, base, cn, o + (size_t)h0 * ostride, ostride, I8);
        } else {
            as_values_avx2_n(p + (size_t)h0 * pstride, pstride, live, 1, vq, vs, kv_stride,
                             ks_stride, kv, hd, base, cn, o + (size_t)h0 * ostride, ostride, I8);
        }
    }
}
#undef AS_VBLOCK
#endif

/* The attention of tokens queries over the int16 cache (call it inside a
 * parallel region). Query t reads the rows lo[t] to lo[t] + n[t] - 1 (lo
 * null: rows 0 to n[t] - 1). q and out are (tokens, q_heads, head_dim).
 * The kernel keeps the scores in its own scratch (a block for each thread),
 * so it does not read the scores operand of the record: the callers size it
 * in other ways (the prompt of Qwen3.6 gives a block of nmax for each
 * thread). Return 0 when the shape does not fit (the caller then uses
 * gemma_attn_decode_i16_mt_body). */
static inline __attribute__((always_inline)) int
as_split_body(const float *q, const void *kq, const float *ks, const void *vq, const float *vs,
              float *scores, float *out, int q_heads, int kv_heads, int head_dim, const int *lo,
              const int *n, int nmax, int tokens, const int KI8,
              const int VI8, int row_heads)
{
    as_scratch *S = as_team();
    int n_rep = q_heads / kv_heads;
    /* TQ6 (KI8 = VI8 = 2): the generic passes over float rows (as_row) */
    const int TQ = KI8 == 2;
    /* the int8 cache has no other kernel, so NP_GEMMA_ATTN_SPLIT=0 keeps
     * this one for it */
    if ((!KI8 && !VI8 && !as_on()) || head_dim % 32 != 0 || head_dim > 1024 ||
        n_rep * head_dim > AS_MAXQ) {
        return 0;
    }
    /* a row of the cache holds row_heads heads; kq and ks point at the first
     * head of this call */
    size_t kv_stride = (size_t)row_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)row_heads * (size_t)(head_dim / 32);
    int g = head_dim / 32;
    size_t pw = (size_t)head_dim + 2;
    size_t need = (size_t)tokens * (size_t)q_heads * AS_MAXC * pw;
    /* the scores of an item: n_rep rows of the longest chunk, a block for
     * each thread (the scores operand of the callers has other sizes) */
    int ilen = 0;
    for (int j = 0; j < tokens; ++j) {
        int l2 = AS_LEN(n[j]);
        ilen = l2 > ilen ? l2 : ilen;
    }
    size_t ineed = (size_t)omp_get_num_threads() * (size_t)n_rep * (size_t)ilen;
    #pragma omp single
    {
        if (S->isc_n < ineed) {
            free(S->isc);
            S->isc = (float *)malloc(ineed * sizeof(float));
            S->isc_n = S->isc ? ineed : 0;
        }
        if (S->parts_n < need) {
            free(S->parts);
            S->parts = (float *)malloc(need * sizeof(float));
            S->parts_n = S->parts ? need : 0;
        }
#if GEMMA_X86 && !defined(__AVX512F__)
        /* the values and the scales grow on their own: a group of 2 tokens
         * with heads of 512 values needs the values of 4 tokens with heads
         * of 256, but half the scales */
        size_t nq = (size_t)tokens * q_heads;
        if (S->q_n < nq * head_dim) {
            free(S->qq);
            S->qq = (int16_t *)malloc(nq * head_dim * sizeof(int16_t));
            S->q_n = S->qq ? nq * head_dim : 0;
        }
        if (S->sq_n < nq) {
            free(S->sq);
            S->sq = (float *)malloc(nq * sizeof(float));
            S->sq_n = S->sq ? nq : 0;
            if (S->sq == NULL) {
                S->q_n = 0;
            }
        }
#endif
    }
    if (S->parts == NULL || S->isc == NULL) {
        return 0;
    }
    float *isc = S->isc + (size_t)omp_get_thread_num() * (size_t)n_rep * (size_t)ilen;
    (void)scores;
    (void)nmax;
#if GEMMA_X86 && !defined(__AVX512F__)
    if (S->q_n == 0) {
        return 0;
    }
    /* the int16 queries, once for each token and head (the items of the
     * chunks of a head share them) */
    if (!TQ) {
        #pragma omp for schedule(static)
        for (int u = 0; u < tokens * q_heads; ++u) {
            as_quant_q(q + (size_t)u * head_dim, head_dim, S->qq + (size_t)u * head_dim,
                       S->sq + u);
        }
    }
#endif
    /* the items: (token, key head, chunk); token t has cdiv(n[t], AS_LEN) */
    int per = kv_heads * AS_MAXC;
    int total = tokens * per;
    #pragma omp for schedule(dynamic, 1)
    for (int u = 0; u < total; ++u) {
        int t = u / per, kv = (u % per) / AS_MAXC, c = u % AS_MAXC;
        int nt = n[t], len = AS_LEN(nt);
        int r0 = c * len;
        if (r0 >= nt) {
            continue;
        }
        int cn = nt - r0 < len ? nt - r0 : len;
        size_t base = (size_t)(lo ? lo[t] : 0) + (size_t)r0;
        float row[1024];
        int h0 = kv * n_rep;
        const float *qt = q + ((size_t)t * q_heads + h0) * head_dim;
#if GEMMA_X86 && !defined(__AVX512F__)
        const int16_t *qq = S->qq + ((size_t)t * q_heads + h0) * head_dim;
        const float *sq = S->sq + (size_t)t * q_heads + h0;
/* as_scores_avx2 keeps one accumulator for each head, up to 8: more heads
 * (the 16 of a global layer of the 12B) go in blocks of 8, with the same
 * values for each head */
#define AS_SC(NH)                                                                           \
        for (int j = 0; j < cn; ++j) {                                                      \
            size_t r = base + (size_t)j;                                                    \
            if (j + AS_PF < cn) {                                                           \
                as_prefetch_row(AS_ATE(kq, (r + AS_PF) * kv_stride + (size_t)kv * head_dim, \
                                       KI8), head_dim, KI8);                                \
                as_prefetch_row(AS_ATE(vq, (r + AS_PF) * kv_stride + (size_t)kv * head_dim, \
                                       VI8), head_dim, VI8);                                \
            }                                                                               \
            for (int hb = 0; hb < (NH); hb += 8) {                                          \
                as_scores_avx2(qq + (size_t)hb * head_dim, sq + hb,                         \
                               (NH) - hb < 8 ? (NH) - hb : 8,                               \
                               AS_ATE(kq, r * kv_stride + (size_t)kv * head_dim, KI8),      \
                               ks + r * ks_stride + (size_t)kv * g, head_dim,               \
                               isc + (size_t)hb * ilen + j, ilen, KI8);                     \
            }                                                                               \
        }
        if (!TQ) {
            switch (n_rep) {
            case 1: AS_SC(1) break;
            case 2: AS_SC(2) break;
            case 4: AS_SC(4) break;
            case 8: AS_SC(8) break;
            default: AS_SC(n_rep)
            }
        }
#undef AS_SC
        for (int j = 0; TQ && n_rep <= 8 && j < cn; ++j) {
            size_t r = base + (size_t)j;
            as_scores_tq_avx2(qt, n_rep, (const uint8_t *)AS_ATE(kq, r * kv_stride +
                                                                  (size_t)kv * head_dim, 2),
                              ks + r * ks_stride + (size_t)kv * g, head_dim, isc + j, ilen);
        }
        for (int j = 0; TQ && n_rep > 8 && j < cn; ++j) {
#else
        for (int j = 0; j < cn; ++j) {
#endif
            size_t r = base + (size_t)j;
            as_row(AS_ATE(kq, r * kv_stride + (size_t)kv * head_dim, KI8),
                   ks + r * ks_stride + (size_t)kv * g, head_dim, row, KI8);
            for (int hh = 0; hh < n_rep; ++hh) {
                isc[(size_t)hh * ilen + j] =
                    as_dot(qt + (size_t)hh * head_dim, row, head_dim);
            }
        }
        for (int hh = 0; hh < n_rep; ++hh) {
            float *sc = isc + (size_t)hh * ilen;
            float m = sc[0];
            for (int j = 1; j < cn; ++j) {
                m = sc[j] > m ? sc[j] : m;
            }
            /* A weight below exp(-64) (2e-28 of the largest) is 0. Its
             * product with a value scale can be a denormal, and a product
             * with a denormal takes a slow microcode path on x86. The value
             * pass then skips the key. */
#if GEMMA_X86 && !defined(__AVX512F__)
            float l = as_softmax_avx2(sc, cn, m);
#else
            float l = 0.0f;
            for (int j = 0; j < cn; ++j) {
                float d = sc[j] - m;
                sc[j] = d < -64.0f ? 0.0f : expf(d);
                l += sc[j];
            }
#endif
            float *pp = S->parts + (((size_t)t * q_heads + h0 + hh) * AS_MAXC + c) * pw;
            for (int d = 0; d < head_dim; ++d) {
                pp[d] = 0.0f;
            }
            pp[head_dim] = m;
            pp[head_dim + 1] = l;
        }
#if GEMMA_X86 && !defined(__AVX512F__)
        unsigned char live[1 << 12];
        if ((!TQ || (n_rep > 2 && n_rep <= 8)) && cn <= (1 << 12)) {
            for (int j = 0; j < cn; ++j) {
                int any = 0;
                for (int hh = 0; hh < n_rep; ++hh) {
                    any |= isc[(size_t)hh * ilen + j] != 0.0f;
                }
                live[j] = (unsigned char)any;
            }
            as_values_avx2(isc, ilen, live, n_rep,
                           vq, vs, kv_stride, ks_stride, kv, head_dim, base, cn,
                           S->parts + (((size_t)t * q_heads + h0) * AS_MAXC + c) * pw,
                           AS_MAXC * pw, VI8);
            continue;
        }
#endif
        for (int j = 0; j < cn; ++j) {
            size_t r = base + (size_t)j;
            int any = 0;
            for (int hh = 0; hh < n_rep; ++hh) {
                any |= isc[(size_t)hh * ilen + j] != 0.0f;
            }
            if (!any) {
                continue;
            }
            as_row(AS_ATE(vq, r * kv_stride + (size_t)kv * head_dim, VI8),
                   vs + r * ks_stride + (size_t)kv * g, head_dim, row, VI8);
            for (int hh = 0; hh < n_rep; ++hh) {
                float p = isc[(size_t)hh * ilen + j];
                if (p != 0.0f) {
                    float *pp = S->parts + (((size_t)t * q_heads + h0 + hh) * AS_MAXC + c) * pw;
                    as_axpy(pp, p, row, head_dim);
                }
            }
        }
    }
    /* the join of the chunks of each query head */
    #pragma omp for schedule(static)
    for (int u = 0; u < tokens * q_heads; ++u) {
        int t = u / q_heads;
        int nc = (n[t] + AS_LEN(n[t]) - 1) / AS_LEN(n[t]);
        const float *pp = S->parts + (size_t)u * AS_MAXC * pw;
        float M = pp[head_dim];
        for (int c = 1; c < nc; ++c) {
            float mc = pp[(size_t)c * pw + head_dim];
            M = mc > M ? mc : M;
        }
        float den = 0.0f;
        float w[AS_MAXC];
        for (int c = 0; c < nc; ++c) {
            float d = pp[(size_t)c * pw + head_dim] - M;
            w[c] = d < -64.0f ? 0.0f : expf(d);
            den += w[c] * pp[(size_t)c * pw + head_dim + 1];
        }
        float inv = 1.0f / den;
        float *o = out + (size_t)u * head_dim;
#if GEMMA_X86
        /* the chunks in their order for each value, 8 values at a time */
        for (int d = 0; d < head_dim; d += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int c = 0; c < nc; ++c) {
                if (w[c] != 0.0f) {
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(w[c]),
                                          _mm256_loadu_ps(pp + (size_t)c * pw + d), acc);
                }
            }
            _mm256_storeu_ps(o + d, _mm256_mul_ps(acc, _mm256_set1_ps(inv)));
        }
#else
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int c = 0; c < nc; ++c) {
                if (w[c] != 0.0f) {
                    acc += w[c] * pp[(size_t)c * pw + d];
                }
            }
            o[d] = acc * inv;
        }
#endif
    }
    return 1;
}

static int gemma_attn_split_i16_body(const float *q, const int16_t *kq, const float *ks,
                                     const int16_t *vq, const float *vs, float *scores,
                                     float *out, int q_heads, int kv_heads, int head_dim,
                                     const int *lo, const int *n, int nmax, int tokens)
{
    return as_split_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n,
                         nmax, tokens, 0, 0, kv_heads);
}

/* gemma_attn_split_i16_body for the heads of a part: a row of the cache holds
 * row_heads heads, and kq, ks, vq, vs point at the first head of the part
 * (GP_ATTN_QC_H). Each head has the bits of gemma_attn_split_i16_body. */
static int gemma_attn_split_i16_rows_body(const float *q, const int16_t *kq, const float *ks,
                                          const int16_t *vq, const float *vs, float *out,
                                          int q_heads, int kv_heads, int head_dim, int n,
                                          int row_heads)
{
    return as_split_body(q, kq, ks, vq, vs, NULL, out, q_heads, kv_heads, head_dim, NULL, &n,
                         n, 1, 0, 0, row_heads);
}

/* The same over the int8 cache (NP_GEMMA_KV_INT8): values int8 with a scale
 * of max |x| / 127 for each group of 32 (gemma_quant_group32_i8). */
static int gemma_attn_split_i8_body(const float *q, const int8_t *kq, const float *ks,
                                    const int8_t *vq, const float *vs, float *scores,
                                    float *out, int q_heads, int kv_heads, int head_dim,
                                    const int *lo, const int *n, int nmax, int tokens)
{
    return as_split_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n,
                         nmax, tokens, 1, 1, kv_heads);
}

/* The same over the cache with int16 keys and int8 values (NP_GEMMA_KV_INT8
 * v: KVCache kv="k16v8"). */
static int gemma_attn_split_v8_body(const float *q, const int16_t *kq, const float *ks,
                                    const int8_t *vq, const float *vs, float *scores,
                                    float *out, int q_heads, int kv_heads, int head_dim,
                                    const int *lo, const int *n, int nmax, int tokens)
{
    return as_split_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n,
                         nmax, tokens, 0, 1, kv_heads);
}

/* The same over the TQ6 cache (np_gemma/tq6.py): q has the rotation, and so
 * has out (GP_TQ_ROT gives them and takes it away). */
static int gemma_attn_split_tq6_body(const float *q, const uint8_t *kq, const float *ks,
                                     const uint8_t *vq, const float *vs, float *scores,
                                     float *out, int q_heads, int kv_heads, int head_dim,
                                     const int *lo, const int *n, int nmax, int tokens)
{
    return as_split_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n,
                         nmax, tokens, 2, 2, kv_heads);
}
#undef AS_AT
#undef AS_ATE
#undef AS_EL

/* gemma_attn_split_i16_body in its own parallel region (a test). Return 1
 * when it ran. */
int gemma_attn_split_i16(const float *q, const int16_t *kq, const float *ks, const int16_t *vq,
                         const float *vs, float *scores, float *out, int q_heads, int kv_heads,
                         int head_dim, int n)
{
    int ok = 0;
    #pragma omp parallel
    {
        int r = gemma_attn_split_i16_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                          head_dim, NULL, &n, n, 1);
        #pragma omp master
        ok = r;
    }
    return ok;
}

/* The attention of one query (lo null, n[0] keys) or of a group of queries
 * over the int8 cache (k16: the int16 keys of the form k16v8), in its own
 * parallel region (the Python paths and the drafter). Return 0 when the
 * shape does not fit. */
int gemma_attn_decode_q8(const float *q, const void *kq, const float *ks, const int8_t *vq,
                         const float *vs, float *scores, float *out, int q_heads, int kv_heads,
                         int head_dim, const int *lo, const int *n, int nmax, int tokens,
                         int k16)
{
    int ok = 0;
    #pragma omp parallel
    {
        int r = k16 ? gemma_attn_split_v8_body(q, (const int16_t *)kq, ks, vq, vs, scores, out,
                                               q_heads, kv_heads, head_dim, lo, n, nmax, tokens)
                    : gemma_attn_split_i8_body(q, (const int8_t *)kq, ks, vq, vs, scores, out,
                                               q_heads, kv_heads, head_dim, lo, n, nmax, tokens);
        #pragma omp master
        ok = r;
    }
    return ok;
}

/* gemma_attn_decode_q8 over the TQ6 cache (a test): q and out in the
 * rotated form. */
int gemma_attn_decode_tq6(const float *q, const uint8_t *kq, const float *ks, const uint8_t *vq,
                          const float *vs, float *out, int q_heads, int kv_heads, int head_dim,
                          const int *lo, const int *n, int nmax, int tokens)
{
    int ok = 0;
    #pragma omp parallel
    {
        int r = gemma_attn_split_tq6_body(q, kq, ks, vq, vs, NULL, out, q_heads, kv_heads,
                                          head_dim, lo, n, nmax, tokens);
        #pragma omp master
        ok = r;
    }
    return ok;
}

/* The split form when it fits, as GP_ATTN_QC_MT, so the Python path of a
 * group (ops.attn_decode_mt) gives the bits of the program. */
void gemma_attn_decode_i16_mt(const float *q, const int16_t *kq, const float *ks,
                              const int16_t *vq, const float *vs, float *scores,
                              float *out, int q_heads, int kv_heads, int head_dim,
                              const int *lo, const int *n, int nmax, int tokens)
{
    #pragma omp parallel
    {
        if (!gemma_attn_split_i16_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                       head_dim, lo, n, nmax, tokens)) {
            gemma_attn_decode_i16_mt_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                          head_dim, lo, n, nmax, tokens);
        }
    }
}

/* The fused decode attention for a small group of query tokens in one
 * parallel region. Token t reads the key rows lo[t] to lo[t] + n[t] - 1 of the
 * int8 cache. A decode step of that token reads the same rows.
 *
 * The shapes: qq is (tokens, q_heads, head_dim). qs is (tokens, q_heads,
 * groups). The array scores holds tokens * q_heads * nmax values. The array
 * out is (tokens, q_heads, head_dim). */
static void gemma_attn_decode_mt_body(const int8_t *qq, const float *qs,
                          const int8_t *kq, const float *ks,
                          const int8_t *vq, const float *vs,
                          float *scores, float *out,
                          int q_heads, int kv_heads, int head_dim,
                          const int *lo, const int *n, int nmax, int tokens)
{
    int g = head_dim / 32;
    int n_rep = q_heads / kv_heads;
    size_t kv_stride = (size_t)kv_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)kv_heads * (size_t)g;
    int total = tokens * q_heads;
    #pragma omp for schedule(static)
    for (int u = 0; u < total; ++u) {
        int t = u / q_heads;
        int h = u % q_heads;
        size_t r = (size_t)lo[t];
        attn_decode_head(qq + (size_t)u * (size_t)head_dim, qs + (size_t)u * (size_t)g,
                         kq + r * kv_stride, ks + r * ks_stride,
                         vq + r * kv_stride, vs + r * ks_stride,
                         scores + (size_t)u * (size_t)nmax,
                         out + (size_t)u * (size_t)head_dim, h / n_rep, head_dim,
                         kv_stride, ks_stride, n[t]);
    }
}

void gemma_attn_decode_mt(const int8_t *qq, const float *qs,
                          const int8_t *kq, const float *ks,
                          const int8_t *vq, const float *vs,
                          float *scores, float *out,
                          int q_heads, int kv_heads, int head_dim,
                          const int *lo, const int *n, int nmax, int tokens)
{
    #pragma omp parallel
    gemma_attn_decode_mt_body(qq, qs, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n, nmax, tokens);
}

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

void gemma_rms_norm(const float *x, const float *w, float *out,
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

void gemma_softcap(const float *x, float *out, int n, float cap)
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

void gemma_gelu(const float *x, float *out, int n)
{
    #pragma omp parallel if(n >= 65536)
    gemma_gelu_body(x, out, n);
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

void gemma_quantize_q8_groups(const float *x, int8_t *qx, float *sx,
                              int32_t *sumx, int rows, int cols)
{
    #pragma omp parallel
    gemma_quantize_q8_groups_body(x, qx, sx, sumx, rows, cols);
}

/* Quantize the last axis of x to int8 and write the transposed layout. The
 * layout for one group is (k / 4, token, 4). stride gives the token stride of
 * qxt, sx, and sumx. The rows from tokens to stride must be zero. */
void gemma_quantize_q8_t(const float *x, int8_t *qxt, float *sx, int32_t *sumx,
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

void gemma_int4_q8_set_prefetch(int on)
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
         * while the multiply runs. Set gemma_int4_q8_set_prefetch(1). */
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
#if GEMMA_X86
#define I4Q2_TB 8
#define I4Q2_MR 8

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
void gemma_int4_q8_set_tb8(int on)
{
    gemma_i4q_tb8 = on ? 1 : 0;
}

/* Run the int8 tile over every row block and token block. tokens is the true
 * token count and stride is the token stride of qxt, sx, and sumx. */
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
void gemma_quantize_q16_t(const float *x, int16_t *qxt, float *sx,
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

void gemma_int4_q16_tile_run(const uint8_t *w, const float *scales,
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

void gemma_int4_q8_tile_run(const uint8_t *w, const float *scales,
                            const int8_t *qxt, const float *sx,
                            const int32_t *sumx, float *out,
                            int rows, int cols, int tokens, int stride)
{
    const int nrb = (rows + I4Q_MR - 1) / I4Q_MR;
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
void gemma_int4_q8_tile_run32(const uint8_t *w, const float *scales,
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
    gemma_int4_q8_tile_run(w, scales, qxt, sx, sumx, out,
                           rows, cols, tokens, stride);
#endif
}

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
void gemma_int4_q8_gemv(const uint8_t *w, const float *scales,
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
void gemma_int4_q8_gemv_x(const uint8_t *w, const float *scales, const float *x,
                          float *out, int rows, int cols)
{
    if (!gemma_i4q_scratch(cols)) {
        return;
    }
    gemma_i4q_row(x, cols);
    gemma_int4_q8_gemv(w, scales, g_i4qx, g_i4qs, g_i4qm, out, rows, cols);
}

/* Up to four int4 matrices on the same one-row activation. The quantization
 * runs one time for all of them. One parallel region covers all four. */
void gemma_int4_q8_multi4(const uint8_t *w0, const float *s0, float *o0, int r0,
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

/* ---------- int8 mixture of experts for a decode step ----------
 * The int4 expert kernel holds four rows in its float lanes, so a decode step
 * reads each selected expert with the plain multiply. This kernel holds eight
 * rows in the int32 lanes of vpdpbusd instead. One call quantizes the row of
 * each job and runs every job in one parallel region.
 *
 * w has the shape (experts, rows, groups, 18). ids gives the selected expert
 * of each job. x holds one row for each job, with a stride of xstride. A
 * stride of 0 gives the same row to every job.
 */
void gemma_int4_q8_moe_gemv(const uint8_t *w, const float *scales,
                            const float *x, const int32_t *ids, int jobs,
                            float *out, int rows, int cols, int xstride)
{
    const int groups = cols / 32;
    const size_t wstride = (size_t)groups * 18;
    const int nrb = (rows + I4QG_MR - 1) / I4QG_MR;
    /* One slice of the scratch for each job. */
    if (jobs <= 0 || !gemma_i4q_scratch(cols * jobs)) {
        return;
    }
    int8_t *qx = g_i4qx;
    float *sx = g_i4qs;
    int32_t *sumx = g_i4qm;
    const size_t qstride = (size_t)cols;
    const size_t sstride = (size_t)groups;
    for (int j = 0; j < jobs; ++j) {
        const float *xj = x + (xstride != 0 ? (size_t)j * (size_t)xstride : 0);
        for (int g = 0; g < groups; ++g) {
            sx[(size_t)j * sstride + g] = gemma_quant_group32(
                xj + (size_t)g * 32, qx + (size_t)j * qstride + (size_t)g * 32,
                &sumx[(size_t)j * sstride + g]);
        }
    }
    #pragma omp parallel for schedule(static) collapse(2)
    for (int j = 0; j < jobs; ++j) {
        for (int rb = 0; rb < nrb; ++rb) {
            const size_t e = (size_t)ids[j];
            gemma_int4_q8_gemv_block(w + e * (size_t)rows * wstride,
                                     scales + e * (size_t)rows * (size_t)groups,
                                     qx + (size_t)j * qstride,
                                     sx + (size_t)j * sstride,
                                     sumx + (size_t)j * sstride,
                                     out + (size_t)j * (size_t)rows,
                                     rows, cols, rb);
        }
    }
}

/* ---------- int8 mixture of experts for a prompt ----------
 * The model gives each expert a different group of tokens. A call for each
 * expert then starts a small parallel region for each expert. A group of
 * sixteen tokens gives little work for one region, so a machine with many
 * cores gains little.
 *
 * These two kernels take the list of experts. One parallel region covers the
 * full work of all the experts. Thus the thread team is large and the caller
 * starts two regions for the whole layer.
 *
 * Every expert uses its own slice of the scratch and the output. off holds the
 * token offset of each expert and ntok holds its token count. The slices are
 * not padded. A tile may read a few tokens past a slice, so the caller leaves a
 * slack of one token block.
 */

/* Quantize the rows of every expert to the transposed int8 layout. src maps a
 * destination row to a row of x. A null src gives the identity. */
void gemma_quantize_q8_t_moe(const float *x, const int32_t *src, int8_t *qxt,
                             float *sx, int32_t *sumx, int cols, int stride,
                             const int32_t *off, const int32_t *ntok, int ne)
{
    const int groups = cols / 32;
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < ne; ++e) {
        for (int t = 0; t < ntok[e]; ++t) {
            const int row = off[e] + t;
            /* src maps a destination row to a row of x. A null src is the
             * identity. Thus the caller needs no separate gather. */
            const float *xt = x + (size_t)(src != NULL ? src[row] : row)
                              * (size_t)cols;
            int8_t q[32];
            for (int g = 0; g < groups; ++g) {
                int32_t s;
                float sc = gemma_quant_group32(xt + (size_t)g * 32, q, &s);
                sx[(size_t)g * (size_t)stride + row] = sc;
                sumx[(size_t)g * (size_t)stride + row] = s;
                for (int q4 = 0; q4 < 8; ++q4) {
                    memcpy(qxt + (((size_t)g * 8 + q4) * (size_t)stride + row) * 4,
                           q + q4 * 4, 4);
                }
            }
        }
    }
}

/* Multiply the selected experts of one layer by their input rows. w holds one
 * matrix for each expert in the Q4_0 block layout. scales holds one float32
 * scale for each group of 32 columns. eid gives the matrix index of a job. */
/* ---------- the experts of a prompt with float32 activations ----------
 *
 * The float path of the experts ran one call for each expert. A layer then
 * made about 220 small calls, and each call paid for its own region. This
 * code runs the float tile of every expert in one region, as
 * gemma_int4_q8_moe_run does with the int8 tile. The activations stay
 * float32, so the result has no error from a quantization. */

/* Gather the rows of each expert into one transposed buffer. Row r of the
 * expert scratch is row src[r] of x. A null src is the identity. xt is
 * (cols, stride). The value of column c of scratch row r is at
 * c * stride + r. */
void gemma_gather_t_moe(const float *x, const int32_t *src, float *xt, int cols,
                        int stride, int rows_total)
{
    #pragma omp parallel for schedule(static)
    for (int c0 = 0; c0 < cols; c0 += 64) {
        int c1 = c0 + 64 < cols ? c0 + 64 : cols;
        for (int r = 0; r < rows_total; ++r) {
            const float *xr = x + (size_t)(src != NULL ? src[r] : r) * (size_t)cols;
            for (int c = c0; c < c1; ++c) {
                xt[(size_t)c * (size_t)stride + r] = xr[c];
            }
        }
    }
}

#if GEMMA_X86 && defined(__AVX512F__)
/* The float tile of one expert: I4T_MR rows and up to I4T_TB tokens. The
 * tokens of the expert start at column base of xt. The row length of xt is
 * ld. */
static inline void gemma_int4_f32_moe_tile(const uint8_t *w, const float *scales,
                                           const float *xt, float *out, int rows,
                                           int cols, int ntok_e, int ld, int base,
                                           int i0, int t0)
{
    __m512 acc[I4T_MR];
    for (int r = 0; r < I4T_MR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    const int groups = cols / 32;
    const int wstride = groups * 18;
    int nrow = rows - i0;
    if (nrow > I4T_MR) {
        nrow = I4T_MR;
    }
    int ntok = ntok_e - t0;
    if (ntok > I4T_TB) {
        ntok = I4T_TB;
    }
    const __mmask16 km = ntok >= I4T_TB ? (__mmask16)0xFFFF : (__mmask16)((1u << ntok) - 1);
    for (int g = 0; g < groups; ++g) {
        float wf[I4T_MR][32];
        for (int r = 0; r < nrow; ++r) {
            const uint8_t *p = w + (size_t)(i0 + r) * wstride + (size_t)g * 18 + 2;
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
            __m512 xv = _mm512_maskz_loadu_ps(km, xt + (size_t)(g * 32 + k) * (size_t)ld
                                                  + base + t0);
            for (int r = 0; r < nrow; ++r) {
                acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(wf[r][k]), xv, acc[r]);
            }
        }
    }
    float tmp[I4T_TB];
    for (int r = 0; r < nrow; ++r) {
        _mm512_storeu_ps(tmp, acc[r]);
        for (int t = 0; t < ntok; ++t) {
            out[(size_t)(base + t0 + t) * (size_t)rows + (size_t)i0 + r] = tmp[t];
        }
    }
}
#else
static inline void gemma_int4_f32_moe_tile(const uint8_t *w, const float *scales,
                                           const float *xt, float *out, int rows,
                                           int cols, int ntok_e, int ld, int base,
                                           int i0, int t0)
{
    const int groups = cols / 32;
    const int wstride = groups * 18;
    for (int r = 0; r < I4T_MR && i0 + r < rows; ++r) {
        for (int t = 0; t < I4T_TB && t0 + t < ntok_e; ++t) {
            float acc = 0.0f;
            for (int g = 0; g < groups; ++g) {
                const uint8_t *b = w + (size_t)(i0 + r) * wstride + (size_t)g * 18 + 2;
                const float sc = scales[(size_t)(i0 + r) * groups + g];
                for (int k = 0; k < 32; ++k) {
                    int nib = k < 16 ? (b[k] & 0x0F) : (b[k - 16] >> 4);
                    acc += (float)(nib - 8) * sc
                           * xt[(size_t)(g * 32 + k) * (size_t)ld + base + t0 + t];
                }
            }
            out[(size_t)(base + t0 + t) * (size_t)rows + (size_t)i0 + r] = acc;
        }
    }
}
#endif

/* Run the float tile of every selected expert in one region. The arguments
 * follow gemma_int4_q8_moe_run. */
void gemma_int4_f32_moe_run(const uint8_t *w, const float *scales, const float *xt,
                            float *out, int rows, int cols, int stride,
                            const int32_t *off, const int32_t *ntok,
                            const int32_t *eid, int ne)
{
    const int groups = cols / 32;
    const size_t expert_bytes = (size_t)rows * (size_t)groups * 18;
    const size_t expert_scales = (size_t)rows * (size_t)groups;
    const int nrb = (rows + I4T_MR - 1) / I4T_MR;
    long *start = (long *)malloc((size_t)(ne + 1) * sizeof(long));
    long total = 0;
    for (int e = 0; e < ne; ++e) {
        start[e] = total;
        total += (long)nrb * (long)((ntok[e] + I4T_TB - 1) / I4T_TB);
    }
    start[ne] = total;
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < total; ++t) {
        int lo = 0, hi = ne - 1, e = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (start[mid] <= t) {
                e = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        const int ntb = (ntok[e] + I4T_TB - 1) / I4T_TB;
        const long u = t - start[e];
        gemma_int4_f32_moe_tile(w + (size_t)eid[e] * expert_bytes,
                                scales + (size_t)eid[e] * expert_scales, xt, out, rows,
                                cols, ntok[e], stride, off[e], (int)(u / ntb) * I4T_MR,
                                (int)(u % ntb) * I4T_TB);
    }
    free(start);
}

/* ---------- the experts of a prompt with int16 activations ----------
 *
 * The float tile of the experts is limited by the work of the multiply: it
 * changes each weight to float32 for each token block. The int16 tile keeps
 * the integer multiply of the int8 tile, at half its rate. Its error is about
 * 250 times smaller than the error of int8. These two functions follow
 * gemma_quantize_q8_t_moe and gemma_int4_q8_moe_run. */

/* Quantize the rows of each expert to int16 in the layout of the tile. Row r
 * of the expert scratch is row src[r] of x (a null src is the identity). */
void gemma_quantize_q16_t_moe(const float *x, const int32_t *src, int16_t *qxt,
                              float *sx, int cols, int stride, int rows_total)
{
    const int groups = cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_total; ++r) {
        const float *xt = x + (size_t)(src != NULL ? src[r] : r) * (size_t)cols;
        int16_t q[32];
        for (int g = 0; g < groups; ++g) {
            sx[(size_t)g * (size_t)stride + r] = gemma_quant_group32_i16(xt + (size_t)g * 32, q);
            for (int k2 = 0; k2 < 16; ++k2) {
                int16_t *dst = qxt + (((size_t)g * 16 + k2) * (size_t)stride + r) * 2;
                dst[0] = q[k2 * 2];
                dst[1] = q[k2 * 2 + 1];
            }
        }
    }
}

void gemma_int4_q16_moe_run(const uint8_t *w, const float *scales, const int16_t *qxt,
                            const float *sx, float *out, int rows, int cols, int stride,
                            const int32_t *off, const int32_t *ntok, const int32_t *eid,
                            int ne)
{
    const int groups = cols / 32;
    const size_t expert_bytes = (size_t)rows * (size_t)groups * 18;
    const size_t expert_scales = (size_t)rows * (size_t)groups;
    const int nrb = (rows + I4Q_MR - 1) / I4Q_MR;
    long *start = (long *)malloc((size_t)(ne + 1) * sizeof(long));
    long total = 0;
    for (int e = 0; e < ne; ++e) {
        start[e] = total;
        total += (long)nrb * (long)((ntok[e] + I4Q_TB - 1) / I4Q_TB);
    }
    start[ne] = total;
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < total; ++t) {
        int lo = 0, hi = ne - 1, e = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (start[mid] <= t) {
                e = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        const int ntb = (ntok[e] + I4Q_TB - 1) / I4Q_TB;
        const long u = t - start[e];
        gemma_int4_q16_tile(w + (size_t)eid[e] * expert_bytes,
                            scales + (size_t)eid[e] * expert_scales, qxt, sx, out, rows,
                            cols, ntok[e], stride, off[e], (int)(u / ntb) * I4Q_MR,
                            (int)(u % ntb) * I4Q_TB);
    }
    free(start);
}

void gemma_int4_q8_moe_run(const uint8_t *w, const float *scales,
                           const int8_t *qxt, const float *sx,
                           const int32_t *sumx, float *out,
                           int rows, int cols, int stride,
                           const int32_t *off, const int32_t *ntok,
                           const int32_t *eid, int ne)
{
    const int groups = cols / 32;
    const size_t wstride = (size_t)groups * 18;
    const size_t expert_bytes = (size_t)rows * wstride;
    const size_t expert_scales = (size_t)rows * (size_t)groups;
    const int nrb = (rows + I4Q_MR - 1) / I4Q_MR;
    const int tb = gemma_i4q_tb8 ? I4Q2_TB : I4Q_TB;
    /* Count the tasks of each expert. A task is one row block and one token
     * block. The task list spans every expert. Thus one parallel region covers
     * the whole layer and the thread team stays large. Parallel over the
     * experts alone is not enough, because an expert is a long serial chain. */
    long *start = (long *)malloc((size_t)(ne + 1) * sizeof(long));
    long total = 0;
    for (int e = 0; e < ne; ++e) {
        start[e] = total;
        total += (long)nrb * (long)((ntok[e] + tb - 1) / tb);
    }
    start[ne] = total;
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < total; ++t) {
        int lo = 0;
        int hi = ne - 1;
        int e = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (start[mid] <= t) {
                e = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        const int ntb = (ntok[e] + tb - 1) / tb;
        const long u = t - start[e];
        const int rb = (int)(u / ntb);
        const int t0 = (int)(u % ntb) * tb;
        const uint8_t *we = w + (size_t)eid[e] * expert_bytes;
        const float *se = scales + (size_t)eid[e] * expert_scales;
        if (gemma_i4q_tb8) {
            gemma_int4_q8_tile_narrow(we, se, qxt, sx, sumx, out, rows, cols,
                                      ntok[e], stride, off[e], rb * I4Q2_MR, t0);
        } else {
            gemma_int4_q8_tile(we, se, qxt, sx, sumx, out, rows, cols,
                               ntok[e], stride, off[e], rb * I4Q_MR, t0);
        }
    }
    free(start);
}


/* Apply the GELU to the gate half of x and multiply by the up half. x has two
 * inner values in each row: the gate first, then the up. out has one inner
 * value in each row. One pass avoids the temporaries of the NumPy path. */
static void gemma_gelu_mul_body(const float *x, float *out, int rows, int inner)
{
    const float c = 0.7978845608028654f;
#if GEMMA_X86 && defined(__AVX512F__)
    const __m512 cv = _mm512_set1_ps(c);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 k3 = _mm512_set1_ps(0.044715f);
    const int nv = inner & ~15;
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const float *g = x + (size_t)i * 2 * (size_t)inner;
        const float *u = g + inner;
        float *o = out + (size_t)i * (size_t)inner;
        for (int j = 0; j < nv; j += 16) {
            const __m512 v = _mm512_loadu_ps(g + j);
            const __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
            const __m512 t = _mm512_mul_ps(cv, _mm512_fmadd_ps(k3, v3, v));
            const __m512 r = _mm512_mul_ps(_mm512_mul_ps(half, v),
                                           _mm512_add_ps(one, gemma_tanh_ps(t)));
            _mm512_storeu_ps(o + j, _mm512_mul_ps(r, _mm512_loadu_ps(u + j)));
        }
        for (int j = nv; j < inner; ++j) {
            const float v = g[j];
            o[j] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))) * u[j];
        }
    }
#else
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const float *g = x + (size_t)i * 2 * (size_t)inner;
        const float *u = g + inner;
        float *o = out + (size_t)i * (size_t)inner;
        for (int j = GELU_MUL_VEC(g, u, o, inner); j < inner; ++j) {
            float v = g[j];
            o[j] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))) * u[j];
        }
    }
#endif
}

void gemma_gelu_mul(const float *x, float *out, int rows, int inner)
{
    #pragma omp parallel if(rows >= 8)
    gemma_gelu_mul_body(x, out, rows, inner);
}

/* Add the weighted expert output to the rows of out. de holds one row for each
 * (expert, token) pair. rows[j] gives the token of row j of de and w[j] gives
 * the router weight. A token may appear in several experts, so the code
 * parallelizes over the hidden axis. Then two jobs never write the same
 * element. */
void gemma_moe_scatter(float *out, const float *de, const int32_t *rows,
                       const float *w, int n, int hidden)
{
    const int db = 64;
    const int nblk = (hidden + db - 1) / db;
    #pragma omp parallel for schedule(static)
    for (int blk = 0; blk < nblk; ++blk) {
        const int d0 = blk * db;
        int dn = hidden - d0;
        if (dn > db) {
            dn = db;
        }
        for (int j = 0; j < n; ++j) {
            const float *src = de + (size_t)j * (size_t)hidden + d0;
            float *dst = out + (size_t)rows[j] * (size_t)hidden + d0;
            const float wj = w[j];
            for (int d = 0; d < dn; ++d) {
                dst[d] += src[d] * wj;
            }
        }
    }
}

/* ---------- fused entry points for a decode step ----------
 * A decode step of the 26B model makes about 600 calls to this library. Each
 * call costs several microseconds before the kernel starts, because Python
 * must read the address of every array. The boundary between Python and C is
 * therefore a real part of the step.
 *
 * Every function below runs two kernels in one call. Each one is a
 * composition of the kernels above: the arithmetic is the arithmetic of those
 * kernels, in the same order. Thus the result does not change and the checks
 * of the model still hold.
 */

/* out = gelu(g) * u for n values, with the gate and the up part in separate
 * arrays. That is the shape of the shared MLP of the 26B model. */
void gemma_gelu_mul_pair(const float *g, const float *u, float *out, int n)
{
    const float c = 0.7978845608028654f;
#if GEMMA_X86 && defined(__AVX512F__)
    const __m512 cv = _mm512_set1_ps(c);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 k3 = _mm512_set1_ps(0.044715f);
    const int nv = n & ~15;
    for (int i = 0; i < nv; i += 16) {
        const __m512 v = _mm512_loadu_ps(g + i);
        const __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
        const __m512 t = _mm512_mul_ps(cv, _mm512_fmadd_ps(k3, v3, v));
        const __m512 r = _mm512_mul_ps(_mm512_mul_ps(half, v),
                                       _mm512_add_ps(one, gemma_tanh_ps(t)));
        _mm512_storeu_ps(out + i, _mm512_mul_ps(r, _mm512_loadu_ps(u + i)));
    }
    for (int i = nv; i < n; ++i) {
        const float v = g[i];
        out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))) * u[i];
    }
#else
    for (int i = GELU_MUL_VEC(g, u, out, n); i < n; ++i) {
        const float v = g[i];
        out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))) * u[i];
    }
#endif
}

/* The three norms of the query, the key, and the value, and then the rotation
 * of the query and the key. v and k may be null. */
void gemma_qkv_norm_rope(float *q, const float *q_w, int q_rows,
                         float *k, const float *k_w, int k_rows,
                         float *v, int v_rows, const float *cos,
                         const float *sin, int q_heads, int k_heads,
                         int head_dim, float eps)
{
    gemma_qkv_norm(q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps);
    gemma_rope(q, q_rows, q_heads, k, k_rows, k_heads, cos, sin, head_dim);
}

/* The norm of one row, then up to four int4 matrices on the result. scratch
 * holds cols values and belongs to the caller, which reuses it. */
void gemma_rms_norm_multi4(const float *x, const float *wn, float *scratch,
                           int cols, float eps,
                           const uint8_t *w0, const float *s0, float *o0, int r0,
                           const uint8_t *w1, const float *s1, float *o1, int r1,
                           const uint8_t *w2, const float *s2, float *o2, int r2,
                           const uint8_t *w3, const float *s3, float *o3, int r3)
{
    gemma_rms_norm(x, wn, scratch, 1, cols, eps);
    gemma_int4_multi4(w0, s0, o0, r0, w1, s1, o1, r1,
                      w2, s2, o2, r2, w3, s3, o3, r3, scratch, cols);
}

/* gelu(g) * u, then one int4 matrix on the result. The gate and the up part
 * have inner values. The matrix has cols columns. scratch holds inner values
 * and belongs to the caller. */
void gemma_gelu_mul_int4(const float *g, const float *u, int inner,
                         float *scratch, const uint8_t *w, const float *s,
                         float *out, int rows, int cols)
{
    gemma_gelu_mul_pair(g, u, scratch, inner);
    gemma_int4_linear(w, s, scratch, out, rows, cols, 1, 32);
}

/* The gate and up projection of the selected experts, then the GELU and the
 * multiply. act holds one row of 2 * inner values for each job and out holds
 * one row of inner values for each job. The two buffers must not overlap: the
 * output of one job would otherwise fall on the up part of another job. */
void gemma_moe_gemv_gelu(const uint8_t *w, const float *scales, const float *x,
                         const int *ids, int jobs, float *act, float *out,
                         int rows, int cols, int xstride, int inner)
{
    gemma_int4_moe_gemv(w, scales, x, ids, jobs, act, rows, cols, xstride);
    gemma_gelu_mul(act, out, jobs, inner);
}

/* Apply the causal mask, the sliding window mask, and the softmax to the last
 * axis of x, in place. x is (rows, cols) and each row holds the scores of one
 * query head. The query token of a row is (row / n_rep) % n_tokens and its
 * position is positions[token]. The key of column c has the position base + c.
 * A key is masked when its position is after the query or when the distance is
 * the window or more. A window of zero turns the window off. */
static void gemma_softmax_mask_body(float *x, int rows, int cols, const int32_t *positions,
                        int n_tokens, int n_rep, int base, int window)
{
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const int tok = (r / n_rep) % n_tokens;
        const int q = positions[tok];
        float *row = x + (size_t)r * (size_t)cols;
        float m = -INFINITY;
        for (int c = 0; c < cols; ++c) {
            const int kp = base + c;
            if (kp > q || (window > 0 && q - kp >= window)) {
                row[c] = -INFINITY;
            } else if (row[c] > m) {
                m = row[c];
            }
        }
        if (m == -INFINITY) {
            m = 0.0f;
        }
        float l = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = expf(row[c] - m);
            row[c] = e;
            l += e;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int c = 0; c < cols; ++c) {
            row[c] *= inv;
        }
    }
}

void gemma_softmax_mask(float *x, int rows, int cols, const int32_t *positions,
                        int n_tokens, int n_rep, int base, int window)
{
    #pragma omp parallel if(rows >= 8)
    gemma_softmax_mask_body(x, rows, cols, positions, n_tokens, n_rep, base, window);
}

/* ---------- flash attention for the prompt ----------
 *
 * The plain path builds the whole score matrix for one chunk: (heads, tokens,
 * keys). It masks the matrix, normalizes it, and multiplies by V. At a long
 * context that matrix is large, so the path moves a lot of memory, and it
 * computes the scores that the causal mask hides and, for a sliding layer, the
 * scores that the window hides.
 *
 * These kernels keep the scores of one block at a time and they read only the
 * keys that the block can see. The sliding window makes the point: 25 of the
 * 30 layers of this model slide with a window of 1024, so at a long context
 * they read the window and not the whole context.
 *
 * The key is held transposed, so the score of a row over a block of keys is a
 * vector over the keys. That removes the horizontal sum. The value stays in
 * the natural layout, so the weighted sum is a vector over the head dimension.
 *
 * The online softmax holds a running maximum m, a running sum l, and the
 * running weighted sum acc. A new maximum rescales acc and l by exp(m - mnew).
 *
 * There are three versions. The AVX-512 version, the AVX2 version, and a
 * straight C version. The straight C version is the reference and the fallback
 * for a target with no AVX2. Set the version with
 * gemma_attn_prefill_set_impl, or leave the default of 0 to take the best
 * version that the build gives.
 */

/* Transpose the key: kt[h][d * ld + j] = k[(lo_key + j) * kv_heads + h][d].
 * The block over d keeps one group of cache lines hot while the loop walks the
 * keys. */
static void gemma_attn_transpose_k(const float *k, float *kt, int nvis, int ld,
                                   int kv_heads, int hd, int lo_key)
{
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < kv_heads; ++h) {
        float *dsth = kt + (size_t)h * (size_t)hd * (size_t)ld;
        for (int jb = 0; jb < nvis; jb += 16) {
            const int je = jb + 16 < nvis ? jb + 16 : nvis;
            for (int db = 0; db < hd; db += 32) {
                const int de = db + 32 < hd ? db + 32 : hd;
                for (int j = jb; j < je; ++j) {
                    const float *src = k + ((size_t)(lo_key + j) * kv_heads + h) * (size_t)hd;
                    for (int d = db; d < de; ++d) {
                        dsth[(size_t)d * ld + j] = src[d];
                    }
                }
            }
        }
    }
}

/* ---------- the straight C version ---------- */

/* Walk one query row at a time. Sum over the head dimension with a plain loop.
 * This is the reference for the two vector versions. */
void gemma_attn_prefill_scalar(const float *q, const float *k, const float *v,
                               const int32_t *positions, int base, int window,
                               float *out, int t, int n, int q_heads,
                               int kv_heads, int hd)
{
    if (t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    const long rows = (long)t * n_rep;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < kv_heads; ++h) {
        for (long row = 0; row < rows; ++row) {
            const int tok = (int)(row / n_rep);
            const int g = (int)(row - (long)tok * n_rep);
            const int pos = positions[tok];
            const float *qr = q + ((size_t)tok * q_heads + h * n_rep + g) * (size_t)hd;
            float *op = out + ((size_t)tok * q_heads + h * n_rep + g) * (size_t)hd;
            int jlo = 0;
            if (window > 0) {
                jlo = pos - window + 1 - base;
                if (jlo < 0) {
                    jlo = 0;
                }
            }
            int jhi = pos - base + 1;
            if (jhi > n) {
                jhi = n;
            }
            for (int d = 0; d < hd; ++d) {
                op[d] = 0.0f;
            }
            float m = -INFINITY;
            float l = 0.0f;
            for (int j = jlo; j < jhi; ++j) {
                const float *kr = k + ((size_t)j * kv_heads + h) * (size_t)hd;
                float s = 0.0f;
                for (int d = 0; d < hd; ++d) {
                    s += qr[d] * kr[d];
                }
                const float mn = m > s ? m : s;
                float alpha = 0.0f;
                if (m == -INFINITY && mn == -INFINITY) {
                    alpha = 1.0f;
                } else if (m != -INFINITY && mn != -INFINITY) {
                    alpha = expf(m - mn);
                }
                const float p = expf(s - mn);
                l = l * alpha + p;
                const float *vr = v + ((size_t)j * kv_heads + h) * (size_t)hd;
                for (int d = 0; d < hd; ++d) {
                    op[d] = op[d] * alpha + p * vr[d];
                }
                m = mn;
            }
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            for (int d = 0; d < hd; ++d) {
                op[d] *= inv;
            }
        }
    }
}

/* ---------- the AVX-512 version ---------- */

#if GEMMA_X86 && defined(__AVX512F__)

/* exp(x) for sixteen floats. The code splits x into an integer part and a
 * fraction in [-0.5, 0.5]. A degree six polynomial gives 2^frac and the scale
 * instruction applies 2^int. The argument is a score less the running maximum,
 * so it is at most zero, and the clamp stops a masked score of -inf. */
static inline __m512 gemma_attn_exp_avx512(__m512 x)
{
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    const __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f));
    const __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(t, n);
    __m512 p = _mm512_set1_ps(0.000154035303933816f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.00133335581464284f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.00961812910762848f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.05550410866482158f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.24022650695910071f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.69314718055994529f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

/* The key block, the row block, and the row micro tile. Sixteen keys fill one
 * vector, so two vectors hold a block of 32 keys. A micro tile of eight rows
 * then needs sixteen score vectors and two key vectors. Eight rows for one key
 * vector is the point: the transposed key block stays in the cache and the
 * kernel reads it eight times fewer than a four row tile. */
#define ATTN5_NC 32
#define ATTN5_NB 2
#define ATTN5_MR 8
#define ATTN5_MC 64

/* The tasks of gemma_attn_prefill_avx512: the threads of the enclosing OpenMP region split
 * them (an orphaned omp for). kt is the transposed keys. */
/* limit (null, or t int32): the last key position of each query in place of
 * its position (the tokens of one image see each other, Model._media_limit);
 * the window still counts from the position. limit does not decrease over
 * the tokens. */
static void gemma_attn_prefill_avx512_tasks(const float *q, const float *v, const float *kt, const int32_t *positions, int base, int window, float *out, int n, int q_heads, int kv_heads, int hd, int n_rep, int lo_key, int ld, long rows, long last, int nrb, int tasks, const int32_t *limit)
{
    float *acc = (float *)malloc((size_t)ATTN5_MC * hd * sizeof(float));
    float *sp = (float *)malloc((size_t)ATTN5_MC * ATTN5_NC * sizeof(float));
    if (acc != NULL && sp != NULL) {
        const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                               8, 9, 10, 11, 12, 13, 14, 15);
        const __m512 ninf = _mm512_set1_ps(-INFINITY);
        #pragma omp for schedule(static)
        for (int task = 0; task < tasks; ++task) {
            const int h = task / nrb;
            const long r0 = (long)(task % nrb) * ATTN5_MC;
            const int tok0 = (int)(r0 / n_rep);
            long t1r = r0 + ATTN5_MC - 1;
            if (t1r > last) {
                t1r = last;
            }
            const int tok1 = (int)(t1r / n_rep);
            const int pos_max = limit != NULL ? limit[tok1] : positions[tok1];
            int jhi = pos_max - base + 1;
            if (jhi > n) {
                jhi = n;
            }
            int jlo = 0;
            if (window > 0) {
                jlo = positions[tok0] - window + 1 - base;
                if (jlo < 0) {
                    jlo = 0;
                }
            }
            if (jlo < lo_key) {
                jlo = lo_key;
            }
            float mrow[ATTN5_MC];
            float lrow[ATTN5_MC];
            for (int i = 0; i < ATTN5_MC; ++i) {
                float *aci = acc + (size_t)i * hd;
                for (int d = 0; d < hd; ++d) {
                    aci[d] = 0.0f;
                }
                mrow[i] = -INFINITY;
                lrow[i] = 0.0f;
            }
            const float *kth = kt + (size_t)h * hd * ld;
            for (int j0 = jlo; j0 < jhi; j0 += ATTN5_NC) {
                const int jn = jhi - j0 < ATTN5_NC ? jhi - j0 : ATTN5_NC;
                /* The score of ATTN5_MR rows over ATTN5_NC keys. The key
                 * vector is read one time for the whole micro tile. */
                for (int i0 = 0; i0 < ATTN5_MC; i0 += ATTN5_MR) {
                    __m512 sv[ATTN5_MR][ATTN5_NB];
                    for (int i = 0; i < ATTN5_MR; ++i) {
                        for (int b = 0; b < ATTN5_NB; ++b) {
                            sv[i][b] = _mm512_setzero_ps();
                        }
                    }
                    /* The row pointers stay fixed for the whole key
                     * block, so the inner loop needs no index product. */
                    const float *qrp[ATTN5_MR];
                    for (int i = 0; i < ATTN5_MR; ++i) {
                        long row = r0 + i0 + i;
                        if (row > last) {
                            row = last;
                        }
                        const int tok = (int)(row / n_rep);
                        const int g = (int)(row - (long)tok * n_rep);
                        qrp[i] = q + ((size_t)tok * q_heads + h * n_rep + g) * hd;
                    }
                    for (int d = 0; d < hd; ++d) {
                        const float *kd = kth + (size_t)d * ld + (j0 - lo_key);
                        __m512 kv[ATTN5_NB];
                        for (int b = 0; b < ATTN5_NB; ++b) {
                            kv[b] = _mm512_loadu_ps(kd + b * 16);
                        }
                        for (int i = 0; i < ATTN5_MR; ++i) {
                            const __m512 qb = _mm512_set1_ps(qrp[i][d]);
                            for (int b = 0; b < ATTN5_NB; ++b) {
                                sv[i][b] = _mm512_fmadd_ps(qb, kv[b], sv[i][b]);
                            }
                        }
                    }
                    for (int i = 0; i < ATTN5_MR; ++i) {
                        long row = r0 + i0 + i;
                        if (row > last) {
                            row = last;
                        }
                        const int tok = (int)(row / n_rep);
                        const int pos = positions[tok];
                        const int lpos = limit != NULL ? limit[tok] : pos;
                        float *srow = sp + (size_t)(i0 + i) * ATTN5_NC;
                        for (int b = 0; b < ATTN5_NB; ++b) {
                            const int jb = j0 + b * 16;
                            __mmask16 keep = 0;
                            if (jb < jhi) {
                                const __m512i kp = _mm512_add_epi32(
                                    _mm512_set1_epi32(base + jb), lane);
                                keep = _mm512_cmp_epi32_mask(kp,
                                    _mm512_set1_epi32(lpos), _MM_CMPINT_LE);
                                if (window > 0) {
                                    const __m512i dist = _mm512_sub_epi32(
                                        _mm512_set1_epi32(pos), kp);
                                    keep = _mm512_kand(keep, _mm512_cmp_epi32_mask(
                                        dist, _mm512_set1_epi32(window), _MM_CMPINT_LT));
                                }
                                const int lim = jhi - jb;
                                if (lim < 16) {
                                    keep = _mm512_kand(keep,
                                        (__mmask16)((1u << lim) - 1u));
                                }
                            }
                            _mm512_storeu_ps(srow + b * 16,
                                _mm512_mask_blend_ps(keep, ninf, sv[i][b]));
                        }
                    }
                }
                /* The online softmax and the weighted sum. */
                for (int i = 0; i < ATTN5_MC; ++i) {
                    float *srow = sp + (size_t)i * ATTN5_NC;
                    const int nv = (jn + 15) / 16;
                    __m512 mx = ninf;
                    for (int b = 0; b < nv; ++b) {
                        mx = _mm512_max_ps(mx, _mm512_loadu_ps(srow + b * 16));
                    }
                    const float smax = _mm512_reduce_max_ps(mx);
                    const float mnew = mrow[i] > smax ? mrow[i] : smax;
                    float alpha = 0.0f;
                    if (mrow[i] == -INFINITY && mnew == -INFINITY) {
                        alpha = 1.0f;
                    } else if (mrow[i] != -INFINITY && mnew != -INFINITY) {
                        alpha = expf(mrow[i] - mnew);
                    }
                    const __m512 mv = _mm512_set1_ps(mnew);
                    const __m512 av = _mm512_set1_ps(alpha);
                    __m512 lsum = _mm512_setzero_ps();
                    for (int b = 0; b < nv; ++b) {
                        const __m512 sv2 = _mm512_loadu_ps(srow + b * 16);
                        __m512 pv = gemma_attn_exp_avx512(_mm512_sub_ps(sv2, mv));
                        pv = _mm512_maskz_mov_ps(
                            _mm512_cmp_ps_mask(sv2, ninf, _CMP_GT_OQ), pv);
                        _mm512_storeu_ps(srow + b * 16, pv);
                        lsum = _mm512_add_ps(lsum, pv);
                    }
                    lrow[i] = lrow[i] * alpha + _mm512_reduce_add_ps(lsum);
                    mrow[i] = mnew;
                    float *aci = acc + (size_t)i * hd;
                    for (int d0 = 0; d0 < hd; d0 += 64) {
                        const int nb = (hd - d0) >= 64 ? 4 : (hd - d0) / 16;
                        __m512 a[4];
                        for (int b = 0; b < nb; ++b) {
                            a[b] = _mm512_mul_ps(_mm512_loadu_ps(aci + d0 + b * 16), av);
                        }
                        for (int j = 0; j < jn; ++j) {
                            const __m512 pv = _mm512_set1_ps(srow[j]);
                            const float *vr = v +
                                ((size_t)(j0 + j) * kv_heads + h) * hd + d0;
                            for (int b = 0; b < nb; ++b) {
                                a[b] = _mm512_fmadd_ps(pv,
                                    _mm512_loadu_ps(vr + b * 16), a[b]);
                            }
                        }
                        for (int b = 0; b < nb; ++b) {
                            _mm512_storeu_ps(aci + d0 + b * 16, a[b]);
                        }
                    }
                }
            }
            for (int i = 0; i < ATTN5_MC; ++i) {
                const long row = r0 + i;
                if (row >= rows) {
                    break;
                }
                const int tok = (int)(row / n_rep);
                const int g = (int)(row - (long)tok * n_rep);
                const float inv = lrow[i] > 0.0f ? 1.0f / lrow[i] : 0.0f;
                const float *aci = acc + (size_t)i * hd;
                float *op = out + ((size_t)tok * q_heads + h * n_rep + g) * hd;
                for (int d = 0; d < hd; ++d) {
                    op[d] = aci[d] * inv;
                }
            }
        }
    }
    free(acc);
    free(sp);
}

void gemma_attn_prefill_avx512(const float *q, const float *k, const float *v,
                               const int32_t *positions, int base, int window,
                               float *out, int t, int n, int q_heads,
                               int kv_heads, int hd)
{
    if (t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads || (hd % 16) != 0) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    int lo_key = 0;
    if (window > 0) {
        lo_key = positions[0] - window + 1 - base;
        if (lo_key < 0) {
            lo_key = 0;
        }
    }
    const int nvis = n - lo_key;
    if (nvis <= 0) {
        memset(out, 0, (size_t)t * q_heads * hd * sizeof(float));
        return;
    }
    const int ld = ((nvis + 15) / 16) * 16;
    float *kt = (float *)malloc((size_t)kv_heads * hd * ld * sizeof(float));
    if (kt == NULL) {
        return;
    }
    gemma_attn_transpose_k(k, kt, nvis, ld, kv_heads, hd, lo_key);

    const long rows = (long)t * n_rep;
    const long last = rows - 1;
    const int nrb = (int)((rows + ATTN5_MC - 1) / ATTN5_MC);
    const int tasks = kv_heads * nrb;
    #pragma omp parallel
    gemma_attn_prefill_avx512_tasks(q, v, kt, positions, base, window, out, n, q_heads, kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb, tasks, NULL);
    free(kt);
}

#endif

/* ---------- the AVX2 version ---------- */

#if GEMMA_X86

/* exp(x) for eight floats. The same split as the AVX-512 version. The scaling
 * by 2^int is a shift of the exponent field. */
static inline __m256 gemma_attn_exp_avx2(__m256 x)
{
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    const __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
    const __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 f = _mm256_sub_ps(t, n);
    __m256 p = _mm256_set1_ps(0.000154035303933816f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.00133335581464284f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.00961812910762848f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.05550410866482158f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.24022650695910071f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.69314718055994529f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    const __m256i ni = _mm256_cvtps_epi32(n);
    const __m256i e = _mm256_slli_epi32(
        _mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

static inline float gemma_attn_hmax8(__m256 x)
{
    __m128 lo = _mm256_castps256_ps128(x);
    const __m128 hi = _mm256_extractf128_ps(x, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

static inline float gemma_attn_hsum8(__m256 x)
{
    __m128 lo = _mm256_castps256_ps128(x);
    const __m128 hi = _mm256_extractf128_ps(x, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

/* Eight keys fill one vector, so two vectors hold a block of 16 keys. A micro
 * tile of four rows needs eight score vectors and two key vectors. */
#define ATTN2_NC 16
#define ATTN2_NB 2
#define ATTN2_MR 4
#define ATTN2_MC 32

/* The tasks of gemma_attn_prefill_avx2: the threads of the enclosing OpenMP region split
 * them (an orphaned omp for). kt is the transposed keys. */
static void gemma_attn_prefill_avx2_tasks(const float *q, const float *v, const float *kt, const int32_t *positions, int base, int window, float *out, int n, int q_heads, int kv_heads, int hd, int n_rep, int lo_key, int ld, long rows, long last, int nrb, int tasks, const int32_t *limit)
{
    float *acc = (float *)malloc((size_t)ATTN2_MC * hd * sizeof(float));
    float *sp = (float *)malloc((size_t)ATTN2_MC * ATTN2_NC * sizeof(float));
    if (acc != NULL && sp != NULL) {
        const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256 ninf = _mm256_set1_ps(-INFINITY);
        #pragma omp for schedule(static)
        for (int task = 0; task < tasks; ++task) {
            const int h = task / nrb;
            const long r0 = (long)(task % nrb) * ATTN2_MC;
            const int tok0 = (int)(r0 / n_rep);
            long t1r = r0 + ATTN2_MC - 1;
            if (t1r > last) {
                t1r = last;
            }
            const int tok1 = (int)(t1r / n_rep);
            const int pos_max = limit != NULL ? limit[tok1] : positions[tok1];
            int jhi = pos_max - base + 1;
            if (jhi > n) {
                jhi = n;
            }
            int jlo = 0;
            if (window > 0) {
                jlo = positions[tok0] - window + 1 - base;
                if (jlo < 0) {
                    jlo = 0;
                }
            }
            if (jlo < lo_key) {
                jlo = lo_key;
            }
            float mrow[ATTN2_MC];
            float lrow[ATTN2_MC];
            for (int i = 0; i < ATTN2_MC; ++i) {
                float *aci = acc + (size_t)i * hd;
                for (int d = 0; d < hd; ++d) {
                    aci[d] = 0.0f;
                }
                mrow[i] = -INFINITY;
                lrow[i] = 0.0f;
            }
            const float *kth = kt + (size_t)h * hd * ld;
            for (int j0 = jlo; j0 < jhi; j0 += ATTN2_NC) {
                const int jn = jhi - j0 < ATTN2_NC ? jhi - j0 : ATTN2_NC;
                for (int i0 = 0; i0 < ATTN2_MC; i0 += ATTN2_MR) {
                    __m256 sv[ATTN2_MR][ATTN2_NB];
                    for (int i = 0; i < ATTN2_MR; ++i) {
                        for (int b = 0; b < ATTN2_NB; ++b) {
                            sv[i][b] = _mm256_setzero_ps();
                        }
                    }
                    const float *qrp[ATTN2_MR];
                    for (int i = 0; i < ATTN2_MR; ++i) {
                        long row = r0 + i0 + i;
                        if (row > last) {
                            row = last;
                        }
                        const int tok = (int)(row / n_rep);
                        const int g = (int)(row - (long)tok * n_rep);
                        qrp[i] = q + ((size_t)tok * q_heads + h * n_rep + g) * hd;
                    }
                    for (int d = 0; d < hd; ++d) {
                        const float *kd = kth + (size_t)d * ld + (j0 - lo_key);
                        __m256 kv[ATTN2_NB];
                        for (int b = 0; b < ATTN2_NB; ++b) {
                            kv[b] = _mm256_loadu_ps(kd + b * 8);
                        }
                        for (int i = 0; i < ATTN2_MR; ++i) {
                            const __m256 qb = _mm256_set1_ps(qrp[i][d]);
                            for (int b = 0; b < ATTN2_NB; ++b) {
                                sv[i][b] = _mm256_fmadd_ps(qb, kv[b], sv[i][b]);
                            }
                        }
                    }
                    for (int i = 0; i < ATTN2_MR; ++i) {
                        long row = r0 + i0 + i;
                        if (row > last) {
                            row = last;
                        }
                        const int tok = (int)(row / n_rep);
                        const int pos = positions[tok];
                        const int lpos = limit != NULL ? limit[tok] : pos;
                        float *srow = sp + (size_t)(i0 + i) * ATTN2_NC;
                        for (int b = 0; b < ATTN2_NB; ++b) {
                            const int jb = j0 + b * 8;
                            __m256i keep = _mm256_setzero_si256();
                            if (jb < jhi) {
                                const __m256i kp = _mm256_add_epi32(
                                    _mm256_set1_epi32(base + jb), lane);
                                const __m256i pq = _mm256_set1_epi32(pos);
                                keep = _mm256_cmpgt_epi32(
                                    _mm256_set1_epi32(lpos + 1), kp);
                                if (window > 0) {
                                    const __m256i dist = _mm256_sub_epi32(pq, kp);
                                    keep = _mm256_and_si256(keep, _mm256_cmpgt_epi32(
                                        _mm256_set1_epi32(window), dist));
                                }
                                const int lim = jhi - jb;
                                if (lim < 8) {
                                    keep = _mm256_and_si256(keep,
                                        _mm256_cmpgt_epi32(_mm256_set1_epi32(lim), lane));
                                }
                            }
                            _mm256_storeu_ps(srow + b * 8,
                                _mm256_blendv_ps(ninf, sv[i][b],
                                    _mm256_castsi256_ps(keep)));
                        }
                    }
                }
                /* The softmax of each row, then the values for two rows at a
                 * time: a load of 4 value vectors serves both rows. Each row
                 * keeps its operations and their order. */
                float alpha_row[ATTN2_MC];
                for (int i = 0; i < ATTN2_MC; ++i) {
                    float *srow = sp + (size_t)i * ATTN2_NC;
                    const int nv = (jn + 7) / 8;
                    __m256 mx = ninf;
                    for (int b = 0; b < nv; ++b) {
                        mx = _mm256_max_ps(mx, _mm256_loadu_ps(srow + b * 8));
                    }
                    const float smax = gemma_attn_hmax8(mx);
                    const float mnew = mrow[i] > smax ? mrow[i] : smax;
                    float alpha = 0.0f;
                    if (mrow[i] == -INFINITY && mnew == -INFINITY) {
                        alpha = 1.0f;
                    } else if (mrow[i] != -INFINITY && mnew != -INFINITY) {
                        alpha = expf(mrow[i] - mnew);
                    }
                    const __m256 mv = _mm256_set1_ps(mnew);
                    __m256 lsum = _mm256_setzero_ps();
                    for (int b = 0; b < nv; ++b) {
                        const __m256 sv2 = _mm256_loadu_ps(srow + b * 8);
                        __m256 pv = gemma_attn_exp_avx2(_mm256_sub_ps(sv2, mv));
                        pv = _mm256_and_ps(pv, _mm256_cmp_ps(sv2, ninf, _CMP_GT_OQ));
                        _mm256_storeu_ps(srow + b * 8, pv);
                        lsum = _mm256_add_ps(lsum, pv);
                    }
                    lrow[i] = lrow[i] * alpha + gemma_attn_hsum8(lsum);
                    mrow[i] = mnew;
                    alpha_row[i] = alpha;
                }
                for (int i = 0; i < ATTN2_MC; i += 2) {
                    const float *s0 = sp + (size_t)i * ATTN2_NC;
                    const float *s1 = s0 + ATTN2_NC;
                    float *ac0 = acc + (size_t)i * hd;
                    float *ac1 = ac0 + hd;
                    const __m256 av0 = _mm256_set1_ps(alpha_row[i]);
                    const __m256 av1 = _mm256_set1_ps(alpha_row[i + 1]);
                    for (int d0 = 0; d0 < hd; d0 += 32) {
                        const int nb = (hd - d0) >= 32 ? 4 : (hd - d0) / 8;
                        __m256 a0[4], a1[4];
                        for (int b = 0; b < nb; ++b) {
                            a0[b] = _mm256_mul_ps(_mm256_loadu_ps(ac0 + d0 + b * 8), av0);
                            a1[b] = _mm256_mul_ps(_mm256_loadu_ps(ac1 + d0 + b * 8), av1);
                        }
                        for (int j = 0; j < jn; ++j) {
                            const __m256 p0 = _mm256_set1_ps(s0[j]);
                            const __m256 p1 = _mm256_set1_ps(s1[j]);
                            const float *vr = v +
                                ((size_t)(j0 + j) * kv_heads + h) * hd + d0;
                            for (int b = 0; b < nb; ++b) {
                                const __m256 vv = _mm256_loadu_ps(vr + b * 8);
                                a0[b] = _mm256_fmadd_ps(p0, vv, a0[b]);
                                a1[b] = _mm256_fmadd_ps(p1, vv, a1[b]);
                            }
                        }
                        for (int b = 0; b < nb; ++b) {
                            _mm256_storeu_ps(ac0 + d0 + b * 8, a0[b]);
                            _mm256_storeu_ps(ac1 + d0 + b * 8, a1[b]);
                        }
                    }
                }
            }
            for (int i = 0; i < ATTN2_MC; ++i) {
                const long row = r0 + i;
                if (row >= rows) {
                    break;
                }
                const int tok = (int)(row / n_rep);
                const int g = (int)(row - (long)tok * n_rep);
                const float inv = lrow[i] > 0.0f ? 1.0f / lrow[i] : 0.0f;
                const float *aci = acc + (size_t)i * hd;
                float *op = out + ((size_t)tok * q_heads + h * n_rep + g) * hd;
                for (int d = 0; d < hd; ++d) {
                    op[d] = aci[d] * inv;
                }
            }
        }
    }
    free(acc);
    free(sp);
}

void gemma_attn_prefill_avx2(const float *q, const float *k, const float *v,
                             const int32_t *positions, int base, int window,
                             float *out, int t, int n, int q_heads,
                             int kv_heads, int hd)
{
    if (t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads || (hd % 8) != 0) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    int lo_key = 0;
    if (window > 0) {
        lo_key = positions[0] - window + 1 - base;
        if (lo_key < 0) {
            lo_key = 0;
        }
    }
    const int nvis = n - lo_key;
    if (nvis <= 0) {
        memset(out, 0, (size_t)t * q_heads * hd * sizeof(float));
        return;
    }
    const int ld = ((nvis + 7) / 8) * 8;
    float *kt = (float *)malloc((size_t)kv_heads * hd * ld * sizeof(float));
    if (kt == NULL) {
        return;
    }
    gemma_attn_transpose_k(k, kt, nvis, ld, kv_heads, hd, lo_key);

    const long rows = (long)t * n_rep;
    const long last = rows - 1;
    const int nrb = (int)((rows + ATTN2_MC - 1) / ATTN2_MC);
    const int tasks = kv_heads * nrb;
    #pragma omp parallel
    gemma_attn_prefill_avx2_tasks(q, v, kt, positions, base, window, out, n, q_heads, kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb, tasks, NULL);
    free(kt);
}

#endif

/* ---------- the dispatch ---------- */

static int gemma_attn_impl = 0;

/* Select the implementation. 0 takes the best one that the build gives, 1 the
 * straight C version, 2 the AVX2 version, and 3 the AVX-512 version. A version
 * that the build does not have falls back to the best one. A test uses this. */
void gemma_attn_prefill_set_impl(int impl)
{
    gemma_attn_impl = impl;
}

void gemma_attn_prefill(const float *q, const float *k, const float *v,
                        const int32_t *positions, int base, int window,
                        float *out, int t, int n, int q_heads, int kv_heads, int hd)
{
    if (gemma_attn_impl == 1) {
        gemma_attn_prefill_scalar(q, k, v, positions, base, window, out,
                                  t, n, q_heads, kv_heads, hd);
        return;
    }
#if GEMMA_X86 && defined(__AVX512F__)
    if (gemma_attn_impl == 2) {
        gemma_attn_prefill_avx2(q, k, v, positions, base, window, out,
                                t, n, q_heads, kv_heads, hd);
        return;
    }
    gemma_attn_prefill_avx512(q, k, v, positions, base, window, out,
                              t, n, q_heads, kv_heads, hd);
#elif GEMMA_X86
    gemma_attn_prefill_avx2(q, k, v, positions, base, window, out,
                            t, n, q_heads, kv_heads, hd);
#else
    gemma_attn_prefill_scalar(q, k, v, positions, base, window, out,
                              t, n, q_heads, kv_heads, hd);
#endif
}

/* gemma_attn_prefill inside an OpenMP region (a program record): every thread
 * of the region calls it. One thread makes the transposed keys (kt), and the
 * threads split the tasks of the AVX-512 or AVX2 version (an orphaned omp
 * for). The scalar version runs on one thread. */
static void gemma_attn_prefill_region_lim(const float *q, const float *k, const float *v,
                                          const int32_t *positions, int base, int window,
                                          float *out, int t, int n, int q_heads, int kv_heads,
                                          int hd, const int32_t *limit)
{
#if GEMMA_X86
    int wide = 0;           /* 16 floats (AVX-512) or 8 (AVX2) */
#if defined(__AVX512F__)
    /* a head of 8k values that is not 16k (72 in the 26B vision) takes AVX2 */
    wide = (gemma_attn_impl == 2 || hd % 16 != 0) ? 8 : 16;
#else
    wide = 8;
#endif
    if (gemma_attn_impl == 1 || t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads ||
        (hd % wide) != 0) {
        #pragma omp single
        gemma_attn_prefill_scalar(q, k, v, positions, base, window, out, t, n, q_heads,
                                  kv_heads, hd);
        return;
    }
    const int n_rep = q_heads / kv_heads;
    int lo_key = 0;
    if (window > 0) {
        lo_key = positions[0] - window + 1 - base;
        if (lo_key < 0) {
            lo_key = 0;
        }
    }
    const int nvis = n - lo_key;
    if (nvis <= 0) {
        #pragma omp single
        memset(out, 0, (size_t)t * q_heads * hd * sizeof(float));
        return;
    }
    const int ld = ((nvis + wide - 1) / wide) * wide;
    const long rows = (long)t * n_rep;
    const long last = rows - 1;
    float *kt = NULL;
    #pragma omp single copyprivate(kt)
    {
        kt = (float *)malloc((size_t)kv_heads * hd * ld * sizeof(float));
        if (kt != NULL) {
            gemma_attn_transpose_k(k, kt, nvis, ld, kv_heads, hd, lo_key);
        }
    }
    if (kt == NULL) {
        return;
    }
#if defined(__AVX512F__)
    if (wide == 16) {
        const int nrb = (int)((rows + ATTN5_MC - 1) / ATTN5_MC);
        gemma_attn_prefill_avx512_tasks(q, v, kt, positions, base, window, out, n, q_heads,
                                        kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb,
                                        kv_heads * nrb, limit);
    } else
#endif
    {
        const int nrb = (int)((rows + ATTN2_MC - 1) / ATTN2_MC);
        gemma_attn_prefill_avx2_tasks(q, v, kt, positions, base, window, out, n, q_heads,
                                      kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb,
                                      kv_heads * nrb, limit);
    }
    #pragma omp single
    free(kt);
#else
    #pragma omp single
    gemma_attn_prefill_scalar(q, k, v, positions, base, window, out, t, n, q_heads, kv_heads, hd);
#endif
}

/* limit: see gemma_attn_prefill_avx512_tasks (the AVX-512 and AVX2 versions) */
void gemma_attn_prefill_region(const float *q, const float *k, const float *v,
                               const int32_t *positions, int base, int window,
                               float *out, int t, int n, int q_heads, int kv_heads, int hd)
{
    gemma_attn_prefill_region_lim(q, k, v, positions, base, window, out, t, n, q_heads, kv_heads,
                                  hd, NULL);
}

/* ---------- the prompt attention over the int16 cache ----------
 *
 * gemma_attn_prefill on the quantized cache (KVCache: int16 values and a
 * float32 scale for each group of 32), with no float copy of the cache:
 * the team dequantizes the keys while it transposes them (kt), and the
 * values of the visible rows only (vf), then runs the tasks of
 * gemma_attn_prefill. A value is (float)q * scale, as KVCache.read gives
 * it, and the tasks are the same: the bits of KVCache.read and
 * gemma_attn_prefill. The scratch is one set for each team (the parts run
 * this at the same time). */
typedef struct {
    float *kt, *vf;
    size_t kt_n, vf_n;
} aq_scratch;
static aq_scratch aq_scr[AS_SLOTS];

static void gemma_attn_prefill_qc_body(const float *q, const int16_t *kq, const float *ks,
                                       const int16_t *vq, const float *vs,
                                       const int32_t *positions, int base, int window,
                                       float *out, int t, int n, int q_heads, int kv_heads,
                                       int hd, const int32_t *limit)
{
#if GEMMA_X86
#if defined(__AVX512F__)
    const int wide = (gemma_attn_impl == 2 || hd % 16 != 0) ? 8 : 16;
#else
    const int wide = 8;
#endif
    if (t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads || hd % 32 != 0 || hd % wide != 0 ||
        gemma_attn_impl == 1) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    int lo_key = 0;
    if (window > 0) {
        lo_key = positions[0] - window + 1 - base;
        if (lo_key < 0) {
            lo_key = 0;
        }
    }
    const int nvis = n - lo_key;
    if (nvis <= 0) {
        #pragma omp for schedule(static)
        for (long i = 0; i < (long)t * q_heads * hd; ++i) {
            out[i] = 0.0f;
        }
        return;
    }
    const int ld = ((nvis + wide - 1) / wide) * wide;
    const int g = hd / 32;
    const size_t row = (size_t)kv_heads * (size_t)hd;
    aq_scratch *S = &aq_scr[omp_get_level() >= 2 ? omp_get_ancestor_thread_num(1) + 1 : 0];
    #pragma omp single
    {
        size_t need = (size_t)kv_heads * hd * ld;
        if (S->kt_n < need) {
            free(S->kt);
            S->kt = (float *)aligned_alloc(64, ((need * 4 + 63) / 64) * 64);
            S->kt_n = need;
        }
        need = (size_t)n * row;     /* rows lo_key to n - 1 are written */
        if (S->vf_n < need) {
            free(S->vf);
            S->vf = (float *)aligned_alloc(64, ((need * 4 + 63) / 64) * 64);
            S->vf_n = need;
        }
    }
    float *kt = S->kt, *vf = S->vf;
    /* the keys: the transposed float rows of each head, blocks of 16 keys */
    const int jbs = (nvis + 15) / 16;
    #pragma omp for schedule(static)
    for (int it = 0; it < kv_heads * jbs; ++it) {
        const int h = it / jbs, jb = (it % jbs) * 16;
        const int je = jb + 16 < nvis ? jb + 16 : nvis;
        float *dsth = kt + (size_t)h * (size_t)hd * (size_t)ld;
        for (int j = jb; j < je; ++j) {
            const size_t r = (size_t)(lo_key + j);
            const int16_t *src = kq + r * row + (size_t)h * hd;
            const float *sc = ks + (r * kv_heads + h) * (size_t)g;
            for (int d = 0; d < hd; ++d) {
                dsth[(size_t)d * ld + j] = (float)src[d] * sc[d / 32];
            }
        }
    }
    /* the values of the visible rows, in groups of 32 */
    #pragma omp for schedule(static)
    for (long gi = (long)lo_key * kv_heads * g; gi < (long)n * kv_heads * g; ++gi) {
        const int16_t *src = vq + (size_t)gi * 32;
        float *dst = vf + (size_t)gi * 32;
        const float sc = vs[gi];
        for (int e = 0; e < 32; ++e) {
            dst[e] = (float)src[e] * sc;
        }
    }
    const long rows = (long)t * n_rep, last = rows - 1;
#if defined(__AVX512F__)
    if (wide == 16) {
        const int nrb = (int)((rows + ATTN5_MC - 1) / ATTN5_MC);
        gemma_attn_prefill_avx512_tasks(q, vf, kt, positions, base, window, out, n, q_heads,
                                        kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb,
                                        kv_heads * nrb, limit);
        return;
    }
#endif
    const int nrb = (int)((rows + ATTN2_MC - 1) / ATTN2_MC);
    gemma_attn_prefill_avx2_tasks(q, vf, kt, positions, base, window, out, n, q_heads,
                                  kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb,
                                  kv_heads * nrb, limit);
#endif
}

/* 1 when gemma_attn_prefill_qc takes this shape (else the caller reads the
 * float rows and runs gemma_attn_prefill). */
int gemma_attn_prefill_qc_ok(int q_heads, int kv_heads, int hd)
{
#if GEMMA_X86
    return gemma_attn_impl != 1 && hd > 0 && hd % 32 == 0 && q_heads >= kv_heads &&
           q_heads % kv_heads == 0;
#else
    (void)q_heads; (void)kv_heads; (void)hd;
    return 0;
#endif
}

void gemma_attn_prefill_qc(const float *q, const int16_t *kq, const float *ks, const int16_t *vq,
                           const float *vs, const int32_t *positions, int base, int window,
                           float *out, int t, int n, int q_heads, int kv_heads, int hd,
                           const int32_t *limit)
{
    #pragma omp parallel
    gemma_attn_prefill_qc_body(q, kq, ks, vq, vs, positions, base, window, out, t, n, q_heads,
                               kv_heads, hd, limit);
}

/* ---------- the compressed-tensors packed layout ----------
 *
 * The E4B mobile-ct checkpoint is a compressed-tensors file, and it is not the
 * layout of the int4 kernels above. The file stores a quantized weight as
 * int32 words. Element k of a row starts at bit k * bits, counted from the
 * start of the row. When bits divides 32 no element crosses a word: for 4
 * bits, word k / 8 holds the value in nibble k % 8 and the low nibble comes
 * first.
 *
 * The value is a two's complement number. The packing added a bias of
 * 2 ** (bits - 1) first, so subtract 8 from a 4-bit value and 2 from a 2-bit
 * value.
 *
 * These matrices use the "channel" strategy of compressed-tensors. One
 * float32 scale covers the whole row, so the scale multiplies the finished
 * dot product. That is one multiply for each row, not one for each group of
 * 32 columns as in the layout above.
 *
 *   out[t][r] = scale[r] * sum over k of x[t][k] * q[r][k]
 *
 * The kernel reads the words where the file put them, and it never writes a
 * float32 copy of a weight. So a token reads 4 bits for each weight instead
 * of the 32 bits of the float32 copy. That is the point of the kernel: the
 * file layout is the runtime layout, and a memory map over the file is then
 * enough.
 *
 * The scale is applied after the sum. Scaling each group and then summing, as
 * the NumPy path does, rounds differently. The difference is in the last bits
 * of the float32 mantissa; check_ct_kernel.py measures it.
 */

/* Put the 8 values of one 4-bit word into 8 signed bytes, in element order. */
static inline __m128i ct_sign4(__m128i w)
{
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(w, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(w, 4), mask);
    /* lo holds the low nibble of each byte and hi holds the high nibble. The
     * byte interleave then puts them in the order of the elements. */
    return _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
}

/* Put the 16 values of one 2-bit word into 16 signed bytes, in element order. */
static inline __m128i ct_sign2(__m128i w)
{
    const __m128i mask = _mm_set1_epi8(0x03);
    __m128i t0 = _mm_and_si128(w, mask);
    __m128i t1 = _mm_and_si128(_mm_srli_epi16(w, 2), mask);
    __m128i t2 = _mm_and_si128(_mm_srli_epi16(w, 4), mask);
    __m128i t3 = _mm_and_si128(_mm_srli_epi16(w, 6), mask);
    /* Byte j of the word holds four values: t0 holds the one at 4j + 0, t1 the
     * one at 4j + 1, and so on. The first interleave pairs the values of each
     * byte, and the second interleave orders the four bytes. The word has only
     * four useful bytes, so a single 16-bit interleave of the low half holds
     * all 16 values. */
    __m128i a = _mm_unpacklo_epi8(t0, t1);
    __m128i b = _mm_unpacklo_epi8(t2, t3);
    return _mm_sub_epi8(_mm_unpacklo_epi16(a, b), _mm_set1_epi8(2));
}

/* Return one 4-bit value of a row as a signed int. Use it for a tail. */
static inline int ct_q4(const uint32_t *w, int k)
{
    return (int)((w[(size_t)k >> 3] >> ((k & 7) * 4)) & 0x0Fu) - 8;
}

/* Return one 2-bit value of a row as a signed int. Use it for a tail. */
static inline int ct_q2(const uint32_t *w, int k)
{
    return (int)((w[(size_t)k >> 4] >> ((k & 15) * 2)) & 0x03u) - 2;
}

/* The dot product of one 4-bit row with one float32 row. */
#if GEMMA_X86 && defined(__AVX512F__)
static inline float ct_dot4_f32(const uint32_t *w, const float *x, int n)
{
    __m512 acc = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        /* Two words hold 16 values, and 16 float32 values fill one zmm. */
        __m128i w2 = _mm_loadl_epi64((const __m128i *)(w + (size_t)g * 2));
        __m128i q = _mm_sub_epi8(
            _mm_unpacklo_epi8(_mm_and_si128(w2, mask),
                              _mm_and_si128(_mm_srli_epi16(w2, 4), mask)), b8);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)),
                              _mm512_loadu_ps(x + (size_t)g * 16), acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (int k = groups * 16; k < n; ++k)
        s += x[k] * (float)ct_q4(w, k);
    return s;
}
#elif GEMMA_X86
static inline float ct_dot4_f32(const uint32_t *w, const float *x, int n)
{
    __m256 acc = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        __m128i w2 = _mm_loadl_epi64((const __m128i *)(w + (size_t)g * 2));
        __m128i q = _mm_sub_epi8(
            _mm_unpacklo_epi8(_mm_and_si128(w2, mask),
                              _mm_and_si128(_mm_srli_epi16(w2, 4), mask)), b8);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),
                              _mm256_loadu_ps(x + (size_t)g * 16), acc);
        acc = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))),
            _mm256_loadu_ps(x + (size_t)g * 16 + 8), acc);
    }
    float s = hsum256_ps(acc);
    for (int k = groups * 16; k < n; ++k)
        s += x[k] * (float)ct_q4(w, k);
    return s;
}
#else
static inline float ct_dot4_f32(const uint32_t *w, const float *x, int n)
{
    float s = 0.0f;
    for (int k = 0; k < n; ++k)
        s += x[k] * (float)ct_q4(w, k);
    return s;
}
#endif

/* The dot product of one 2-bit row with one float32 row. */
#if GEMMA_X86 && defined(__AVX512F__)
static inline float ct_dot2_f32(const uint32_t *w, const float *x, int n)
{
    __m512 acc = _mm512_setzero_ps();
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        /* One word holds 16 values, and 16 float32 values fill one zmm. */
        __m128i q = ct_sign2(_mm_cvtsi32_si128((int)w[g]));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)),
                              _mm512_loadu_ps(x + (size_t)g * 16), acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (int k = groups * 16; k < n; ++k)
        s += x[k] * (float)ct_q2(w, k);
    return s;
}
#elif GEMMA_X86
static inline float ct_dot2_f32(const uint32_t *w, const float *x, int n)
{
    __m256 acc = _mm256_setzero_ps();
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        __m128i q = ct_sign2(_mm_cvtsi32_si128((int)w[g]));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),
                              _mm256_loadu_ps(x + (size_t)g * 16), acc);
        acc = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))),
            _mm256_loadu_ps(x + (size_t)g * 16 + 8), acc);
    }
    float s = hsum256_ps(acc);
    for (int k = groups * 16; k < n; ++k)
        s += x[k] * (float)ct_q2(w, k);
    return s;
}
#else
static inline float ct_dot2_f32(const uint32_t *w, const float *x, int n)
{
    float s = 0.0f;
    for (int k = 0; k < n; ++k)
        s += x[k] * (float)ct_q2(w, k);
    return s;
}
#endif

/* Four 4-bit rows for each x block.
 *
 * The one-row dot above reads the whole x row again for each output row. For a
 * matrix of 10240 rows and 2560 columns that is 105 MB of x traffic against
 * 13 MB of weights, so x controls the time. The x values are the same for every
 * row, so this loop keeps one x block in a register and uses it for four
 * weight rows. The x traffic falls by four times, and the four independent
 * accumulators give the pipeline more work to overlap. This is the same
 * change the int4 kernel of the 12B model uses.
 *
 * out holds the four dot products. The scale is not applied here, because it
 * is different for each row.
 */
#if GEMMA_X86 && defined(__AVX512F__)
static inline void ct_dot4x4_f32(const uint32_t *w, size_t stride,
                                 const float *x, int n, float *out)
{
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        __m512 xv = _mm512_loadu_ps(x + (size_t)g * 16);
        __m128i w0 = _mm_loadl_epi64((const __m128i *)(w + (size_t)g * 2));
        __m128i w1 = _mm_loadl_epi64((const __m128i *)(w + stride + (size_t)g * 2));
        __m128i w2 = _mm_loadl_epi64((const __m128i *)(w + 2 * stride + (size_t)g * 2));
        __m128i w3 = _mm_loadl_epi64((const __m128i *)(w + 3 * stride + (size_t)g * 2));
        __m128i q0 = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(w0, mask),
                                   _mm_and_si128(_mm_srli_epi16(w0, 4), mask)), b8);
        __m128i q1 = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(w1, mask),
                                   _mm_and_si128(_mm_srli_epi16(w1, 4), mask)), b8);
        __m128i q2 = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(w2, mask),
                                   _mm_and_si128(_mm_srli_epi16(w2, 4), mask)), b8);
        __m128i q3 = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(w3, mask),
                                   _mm_and_si128(_mm_srli_epi16(w3, 4), mask)), b8);
        a0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q0)), xv, a0);
        a1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1)), xv, a1);
        a2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q2)), xv, a2);
        a3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q3)), xv, a3);
    }
    out[0] = _mm512_reduce_add_ps(a0);
    out[1] = _mm512_reduce_add_ps(a1);
    out[2] = _mm512_reduce_add_ps(a2);
    out[3] = _mm512_reduce_add_ps(a3);
    int tail = groups * 16;
    for (int k = tail; k < n; ++k) {
        float xk = x[k];
        out[0] += xk * (float)ct_q4(w, k);
        out[1] += xk * (float)ct_q4(w + stride, k);
        out[2] += xk * (float)ct_q4(w + 2 * stride, k);
        out[3] += xk * (float)ct_q4(w + 3 * stride, k);
    }
}
#elif GEMMA_X86
static inline void ct_dot4x4_f32(const uint32_t *w, size_t stride,
                                 const float *x, int n, float *out)
{
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps();
    __m256 a5 = _mm256_setzero_ps();
    __m256 a6 = _mm256_setzero_ps();
    __m256 a7 = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        __m256 xlo = _mm256_loadu_ps(x + (size_t)g * 16);
        __m256 xhi = _mm256_loadu_ps(x + (size_t)g * 16 + 8);
        for (int j = 0; j < 4; ++j) {
            __m128i wj = _mm_loadl_epi64((const __m128i *)(w + (size_t)j * stride + (size_t)g * 2));
            __m128i q = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(wj, mask),
                                  _mm_and_si128(_mm_srli_epi16(wj, 4), mask)), b8);
            __m256 lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
            __m256 hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8)));
            switch (j) {
            case 0: a0 = _mm256_fmadd_ps(lo, xlo, a0); a4 = _mm256_fmadd_ps(hi, xhi, a4); break;
            case 1: a1 = _mm256_fmadd_ps(lo, xlo, a1); a5 = _mm256_fmadd_ps(hi, xhi, a5); break;
            case 2: a2 = _mm256_fmadd_ps(lo, xlo, a2); a6 = _mm256_fmadd_ps(hi, xhi, a6); break;
            default: a3 = _mm256_fmadd_ps(lo, xlo, a3); a7 = _mm256_fmadd_ps(hi, xhi, a7); break;
            }
        }
    }
    out[0] = hsum256_ps(_mm256_add_ps(a0, a4));
    out[1] = hsum256_ps(_mm256_add_ps(a1, a5));
    out[2] = hsum256_ps(_mm256_add_ps(a2, a6));
    out[3] = hsum256_ps(_mm256_add_ps(a3, a7));
    int tail = groups * 16;
    for (int k = tail; k < n; ++k) {
        float xk = x[k];
        out[0] += xk * (float)ct_q4(w, k);
        out[1] += xk * (float)ct_q4(w + stride, k);
        out[2] += xk * (float)ct_q4(w + 2 * stride, k);
        out[3] += xk * (float)ct_q4(w + 3 * stride, k);
    }
}
#else
static inline void ct_dot4x4_f32(const uint32_t *w, size_t stride,
                                 const float *x, int n, float *out)
{
    for (int j = 0; j < 4; ++j)
        out[j] = ct_dot4_f32(w + (size_t)j * stride, x, n);
}
#endif

static int gemma_ct_rows4 = 1;

/* Select the four-row loop (1) or the one-row loop (0). Use this for a test. */
void gemma_ct_set_rows4(int on)
{
    gemma_ct_rows4 = on ? 1 : 0;
}

/* Multiply x by one matrix in the packed layout. bits is 4 or 2. */
static void ct_linear_run(const uint32_t *w, const float *scale, const float *x,
                          float *out, int rows, int cols, int tokens, int bits)
{
    /* A 4-bit row uses cols / 8 words and a 2-bit row cols / 16. The caller
     * checks that cols is a multiple of 16, so the vector loops below cover
     * the row and the scalar tail never runs. */
    size_t words = (size_t)(cols / 16) * (bits == 4 ? 2 : 1);
    int groups = gemma_ct_rows4 ? rows / 4 : 0;
    if (bits == 4) {
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < groups; ++b) {
            int r = b * 4;
            for (int t = 0; t < tokens; ++t) {
                float d[4];
                ct_dot4x4_f32(w + (size_t)r * words, words,
                              x + (size_t)t * (size_t)cols, cols, d);
                for (int j = 0; j < 4; ++j)
                    out[(size_t)t * (size_t)rows + r + j] = scale[r + j] * d[j];
            }
        }
        #pragma omp parallel for schedule(static)
        for (int r = groups * 4; r < rows; ++r) {
            const float s = scale[r];
            for (int t = 0; t < tokens; ++t)
                out[(size_t)t * (size_t)rows + r] =
                    s * ct_dot4_f32(w + (size_t)r * words,
                                    x + (size_t)t * (size_t)cols, cols);
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint32_t *wr = w + (size_t)r * words;
        const float s = scale[r];
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            out[(size_t)t * (size_t)rows + r] = s * ct_dot2_f32(wr, xt, cols);
        }
    }
}

/* Multiply x by W. W is a weight of the compressed-tensors file, packed.
 *
 * w       the int32 words of the weight, one row after another
 * scale   one float32 value for each row
 * x       the activations, tokens rows of cols values
 * out     tokens rows of rows values
 * bits    4 or 2
 *
 * cols must be a multiple of 16 and must equal the value count that the
 * packed words hold. Every matrix of the E4B text model satisfies this.
 */
void gemma_ct_linear(const uint32_t *w, const float *scale, const float *x,
                     float *out, int rows, int cols, int tokens, int bits)
{
    if (bits != 4 && bits != 2)
        return;
    ct_linear_run(w, scale, x, out, rows, cols, tokens, bits);
}



/* ---------- bodies of the fused entry points ----------
 * Each kernel above has a body and a wrapper. The body holds the loops with an
 * orphaned "omp for". The wrapper opens the region and calls the body. A caller
 * that is already in a region, such as the program interpreter of
 * PERF_PLAN.md (phase 2), calls the bodies. The threads then stay in one
 * region for many kernels.
 *
 * The fused entry points call two kernels. Their bodies call the two bodies in
 * the same order. An "omp for" and an "omp single" end with a barrier, so the
 * second step sees the whole result of the first. The entry points keep their
 * own calls, so the result of the Python path does not change.
 */

static void gemma_q6k_linear_body(const uint8_t *w, const float *x, float *out,
                                  int rows, int cols, int tokens)
{
#if GEMMA_X86
    if (gemma_have_avx512()) {
        gemma_q6k_avx512_body(w, x, out, rows, cols, tokens);
    } else {
        gemma_q6k_avx2_body(w, x, out, rows, cols, tokens);
    }
#else
    gemma_q6k_scalar_body(w, x, out, rows, cols, tokens);
#endif
}

static void gemma_rms_norm_multi4_body(const float *x, const float *wn, float *scratch,
                                       int cols, float eps,
                                       const uint8_t *w0, const float *s0, float *o0, int r0,
                                       const uint8_t *w1, const float *s1, float *o1, int r1,
                                       const uint8_t *w2, const float *s2, float *o2, int r2,
                                       const uint8_t *w3, const float *s3, float *o3, int r3)
{
    #pragma omp single
    gemma_rms_norm_row(x, wn, scratch, cols, eps);
    gemma_int4_multi4_body(w0, s0, o0, r0, w1, s1, o1, r1,
                           w2, s2, o2, r2, w3, s3, o3, r3, scratch, cols);
}

static void gemma_gelu_mul_int4_body(const float *g, const float *u, int inner,
                                     float *scratch, const uint8_t *w, const float *s,
                                     float *out, int rows, int cols)
{
    #pragma omp single
    gemma_gelu_mul_pair(g, u, scratch, inner);
    gemma_int4_linear_body(w, s, scratch, out, rows, cols, 1, 32);
}

static void gemma_moe_gemv_gelu_body(const uint8_t *w, const float *scales, const float *x,
                                     const int *ids, int jobs, float *act, float *out,
                                     int rows, int cols, int xstride, int inner)
{
    gemma_int4_moe_gemv_body(w, scales, x, ids, jobs, act, rows, cols, xstride);
    gemma_gelu_mul_body(act, out, jobs, inner);
}

static void gemma_moe_gemv_gelu_mt_body(const uint8_t *w, const float *scales,
                                        const float *x, const int *ids, const int *poff,
                                        const int *xi, int jobs, float *act, float *out,
                                        int rows, int cols, int xstride, int inner)
{
    gemma_int4_moe_gemv_mt_body(w, scales, x, ids, poff, xi, jobs, act, rows, cols, xstride);
    gemma_gelu_mul_body(act, out, poff[jobs], inner);
}

static void gemma_qkv_norm_rope_body(float *q, const float *q_w, int q_rows,
                                     float *k, const float *k_w, int k_rows,
                                     float *v, int v_rows, const float *cos,
                                     const float *sin, int q_heads, int k_heads,
                                     int head_dim, float eps)
{
    gemma_qkv_norm_body(q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps);
    gemma_rope_body(q, q_rows, q_heads, k, k_rows, k_heads, cos, sin, head_dim);
}


/* ---------- the program interpreter ----------
 * PERF_PLAN.md, phase 2. Python builds a program (np_gemma/program.py) and
 * this code runs it. The program is one int64 array:
 *
 *     int64  magic, env count, record count, 0
 *     int64  env[env count]         the slots: parameters and variables
 *     record code[record count]
 *
 * A record has an operation code, flags, and GP_NARG operands. A tag gives
 * the kind of each operand. An operand is an integer literal, a float literal,
 * or a slot of the environment. An address is an integer, and a float literal
 * holds the bits of a float32.
 *
 * gemma_run opens one OpenMP region for the whole program. Each thread copies
 * the environment. A scalar operation writes the private copy of each thread,
 * so the scalar operations need no barrier and no thread writes shared data.
 * A kernel operation calls the body of a kernel. The loops of a body are an
 * orphaned "omp for", which ends with a barrier. A one-row operation runs in
 * an "omp single", which also ends with a barrier.
 *
 * limit runs only the first limit records. A check compares the buffers after
 * each prefix with the Python interpreter, and the first difference names the
 * faulty operation. A limit below zero runs every record.
 */

/* ---------- the other parts of the library ----------
 * mlx_affine.c: the MLX affine weight format (mlx-community, OptiQ).
 * deltanet.c: the Gated DeltaNet (Qwen3.5). The program records below
 * call their bodies. */
#include "moe.c"
#include "mlx_affine.c"
#include "kquants.c"
#include "deltanet.c"
#include "hyperconn.c"
#include "qsa.c"

/* ---------- small operations of other models (Qwen3.5) ---------- */

/* The router of a MoE layer as softmax, then the top k, then the k weights
 * divided by their sum (Qwen3.5, Mixtral). At an equal value the lower index
 * wins. One token for each thread. */
static void gp_router_topk_body(const float *logits, int t, int experts, int k, float *val,
                                int32_t *idx)
{
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        const float *l = logits + (size_t)j * experts;
        float m = -INFINITY;
        for (int x = 0; x < experts; ++x) {
            m = l[x] > m ? l[x] : m;
        }
        float p[1024];
        float sum = 0.f;
        for (int x = 0; x < experts; ++x) {
            p[x] = expf(l[x] - m);
            sum += p[x];
        }
        float vs = 0.f;
        for (int s2 = 0; s2 < k; ++s2) {
            int b = -1;
            for (int x = 0; x < experts; ++x) {
                if (p[x] >= 0.f && (b < 0 || p[x] > p[b])) {
                    b = x;
                }
            }
            idx[(size_t)j * k + s2] = b;
            val[(size_t)j * k + s2] = p[b] / sum;
            vs += p[b] / sum;
            p[b] = -1.f;
        }
        for (int s2 = 0; s2 < k; ++s2) {
            val[(size_t)j * k + s2] /= vs;
        }
    }
}

/* out = x * sigmoid(g), for n values. */
static void gp_sigmul_body(const float *x, const float *g, float *out, int64_t n)
{
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = x[i] / (1.f + expf(-g[i]));
    }
}

/* The steps before the attention of a gated attention with partial RoPE
 * (Qwen3.5), for t tokens from pos:
 *
 * - qg (t x nq x 2 hd): the query and the gate of each head;
 * - the query: rms_norm with qn, RoPE on the first rot values, times scale,
 *   into qout (t x nq x hd); the gate into gate (t x nq x hd);
 * - kk, vv (t x nk x hd): the key gets rms_norm with kn and RoPE; the key and
 *   the value go to the cache K, V (nk heads, hs values between heads) at
 *   the positions pos to pos + t - 1. With a null K (the int16 cache), the
 *   key goes to kout (t x nk x hd) instead.
 *
 * cos and sin have rot values for each token (the two halves the same). */
static void gp_attn_prep_body(const float *qg, const float *kk, const float *vv,
                              const float *qn, const float *kn, const float *cos,
                              const float *sin, float *K, float *V, int64_t hs, int64_t pos,
                              int t, int nq, int nk, int hd, int rot, float eps, float scale,
                              float *qout, float *gate, float *kout)
{
    int half = rot / 2;
    #pragma omp for schedule(static)
    for (int x = 0; x < t * (nq + nk); ++x) {
        int j = x / (nq + nk), h = x % (nq + nk);
        const float *src;
        const float *w;
        float *dst;
        float sc;
        if (h < nq) {
            src = qg + ((size_t)j * nq + h) * 2 * hd;
            memcpy(gate + ((size_t)j * nq + h) * hd, src + hd, (size_t)hd * 4);
            dst = qout + ((size_t)j * nq + h) * hd;
            w = qn;
            sc = scale;
        } else {
            int kh = h - nq;
            src = kk + ((size_t)j * nk + kh) * hd;
            if (K == NULL) {
                /* The int16 cache: the key goes to kout, and GP_KV_WRITE
                 * stores the key and the value. */
                dst = kout + ((size_t)j * nk + kh) * hd;
            } else {
                size_t rs = gp_kv_rs((size_t)hs, nk, hd);
                dst = K + (size_t)kh * hs + (size_t)(pos + j) * rs;
                memcpy(V + (size_t)kh * hs + (size_t)(pos + j) * rs,
                       vv + ((size_t)j * nk + kh) * hd, (size_t)hd * 4);
            }
            w = kn;
            sc = 1.f;
        }
        float ss = 0.f;
        for (int d = 0; d < hd; ++d) {
            ss += src[d] * src[d];
        }
        float inv = 1.f / sqrtf(ss / (float)hd + eps);
        float y[1024];
        for (int d = 0; d < hd; ++d) {
            y[d] = src[d] * inv * w[d];
        }
        const float *c = cos + (size_t)j * rot, *sn = sin + (size_t)j * rot;
        for (int d = 0; d < half; ++d) {
            float a = y[d], b = y[d + half];
            dst[d] = (a * c[d] - b * sn[d]) * sc;
            dst[d + half] = (b * c[d + half] + a * sn[d + half]) * sc;
        }
        for (int d = rot; d < hd; ++d) {
            dst[d] = y[d] * sc;
        }
    }
}

#define GP_NARG 24
#define GP_MAGIC 0x4750524f47303031LL   /* "GPROG001" */

typedef struct {
    int32_t op;
    int32_t flags;
    uint8_t tag[GP_NARG];
    int64_t v[GP_NARG];
} gp_rec;

enum { GP_T_NONE = 0, GP_T_INT = 1, GP_T_F32 = 2, GP_T_SLOT = 3 };

/* Set by a record that cannot run its shape (the int8 cache has one
 * kernel); gemma_run then returns -3. */
static volatile int gp_fail;

enum {
    GP_S_MOV = 1, GP_S_ADD = 2, GP_S_SUB = 3, GP_S_MUL = 4, GP_S_MAX = 5,
    GP_S_MIN = 6,
    GP_RMS_NORM = 16, GP_ADD = 17, GP_MUL_S = 18, GP_COPY = 19, GP_GELU = 20,
    GP_MUL = 21,
    GP_INT4_LINEAR = 32, GP_INT4_MULTI4 = 33, GP_RMS_NORM_MULTI4 = 34,
    GP_GELU_MUL_INT4 = 35, GP_INT4_LINEAR_MT = 36, GP_INT4_MULTI4_MT = 37,
    GP_GELU_MUL_ROWS = 38, GP_BF16_LINEAR = 39,
    GP_QKV_NORM_ROPE = 48, GP_KV_WRITE = 49, GP_ATTN_QC = 50, GP_ATTN_F32 = 51,
    GP_ATTN_QC_MT = 52, GP_ATTN_F32_MT = 53, GP_QKV_NORM = 54, GP_ROPE = 55,
    GP_KV_WRITE_HEADS = 56, GP_ATTN_F32H = 57,
    GP_KV_WRITE8 = 58, GP_ATTN_Q8 = 59, GP_ATTN_Q8_MT = 60,
    GP_KV_WRITEV8 = 61, GP_ATTN_V8 = 62, GP_ATTN_V8_MT = 63,
    GP_ROUTER = 64, GP_MOE = 65, GP_ROUTER_MT = 66, GP_MOE_MT = 67, GP_MOE_N = 68,
    /* the TQ6 cache of the Qwen models (np_gemma/tq6.py) */
    GP_KV_WRITETQ = 69, GP_ATTN_TQ = 70, GP_ATTN_TQ_MT = 71, GP_TQ_ROT = 72,
    GP_XBAR = 80, GP_MOE_PART = 81, GP_ATTN_QC_H = 82, GP_ATTN_F32_H = 83,
    /* The MLX affine format, the Gated DeltaNet, and the small operations of
     * Qwen3.5 (QWEN_PLAN.md). */
    GP_MA_QUANT = 100, GP_MA_LINEAR = 101, GP_MA_MOE = 102, GP_ROUTER_TOPK = 103,
    GP_GDN = 104, GP_ATTN_PREP = 105, GP_SIGMUL = 106, GP_KQ_QUANT = 107, GP_KQ_LINEAR = 108,
    GP_KQ_MOE = 109,
    /* qwen4exp: the gated residual and the n-gram layer (hyperconn.c) */
    GP_HC_NORM = 114, GP_HC_ACT = 115, GP_HC_MIX = 116, GP_HC_ADD = 117, GP_PLE_GATE = 118,
    GP_PLE_CONV = 119, GP_QSA_SELECT = 120, GP_ATTN_QSA = 121, GP_HC_CAT = 122,
    GP_MOE_PLAN = 123,
    /* The media encoders (np_gemma/gemma4_encoders.py, program form) */
    GP_ENC_LINEAR = 133, GP_ENC_RMS = 134, GP_ENC_GELU_MUL = 135, GP_ENC_ADD = 136,
    GP_ENC_ROPE2D = 137, GP_ENC_ATTN = 138, GP_ENC_SILU = 139, GP_ENC_MUL_VEC = 140,
    GP_ENC_GLU = 141, GP_ENC_DWCONV = 142, GP_ENC_LOCAL_ATTN = 143, GP_ENC_CLAMP = 144,
    GP_ENC_BIAS_CLAMP = 145, GP_ENC_LNORM = 146, GP_ENC_GELU = 147,
    /* The rows of a Q6_K matrix (gemma_q6k_linear_body): the output head of
     * the parts (np_gemma/parts.py) */
    GP_Q6K_LINEAR = 148,
    /* The attention of a prompt block for the KV heads of one part, with the
     * cache of the part (np_gemma/parts.py, PartKVCache) */
    GP_PART_PREFILL = 149,
    /* The attention of a prompt block over the int16 cache of a layer, as
     * gemma_attn_prefill_qc (np_gemma/prompt.py) */
    GP_ATTN_PREFILL_QC = 150,
    /* The int16 x of a prompt block and its product with a KQ_Q4X matrix
     * (kq_quant16_body, kq_linear16_body; NP_GEMMA_INT4_Q8=16) */
    GP_KQ_QUANT16 = 151, GP_KQ_LINEAR16 = 152,
};

int gemma_gp_record_size(void)
{
    return (int)sizeof(gp_rec);
}

static inline int64_t gp_i(const gp_rec *r, const int64_t *e, int k)
{
    return r->tag[k] == GP_T_SLOT ? e[r->v[k]] : r->v[k];
}

static inline float gp_f(const gp_rec *r, const int64_t *e, int k)
{
    uint32_t u = (uint32_t)gp_i(r, e, k);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

#define GP_P(T, k) ((T *)(intptr_t)gp_i(r, e, (k)))
#define GP_I(k) ((int)gp_i(r, e, (k)))

/* out += d * v, one multiply and one add for each value, as NumPy does. A fused
 * multiply and add rounds one time and gives other bits. */
__attribute__((optimize("fp-contract=off")))
static void gp_add_scaled(float *out, const float *d, float v, int n)
{
    for (int c = 0; c < n; ++c) {
        float p = d[c] * v;
        out[c] = out[c] + p;
    }
}

/* The experts of one token, as Model._moe_one_token does it. The kernels run
 * the selected experts in the order of their index. Then the code adds their
 * outputs with the router weights, in the same order.
 *
 * GP_MOE gives the count of experts as operand 3. GP_MOE_N gives the address
 * of an int32 that holds the count. The GPU writes that count with the list
 * of the experts that it does not hold (np_gemma/gpu.py). A count of 0 gives
 * an output of zeros. */
static void gp_moe_one(const gp_rec *r, const int64_t *e)
{
    const float *h = GP_P(const float, 0);
    const float *val = GP_P(const float, 1);
    const int32_t *idx = GP_P(const int32_t, 2);
    int top_k = r->op == GP_MOE_N ? *GP_P(const int32_t, 3) : GP_I(3);
    const uint8_t *gu_w = GP_P(const uint8_t, 4);
    const float *gu_s = GP_P(const float, 5);
    const uint8_t *dn_w = GP_P(const uint8_t, 6);
    const float *dn_s = GP_P(const float, 7);
    int gu_rows = GP_I(8);
    int cols = GP_I(9);
    int dn_rows = GP_I(10);
    int inner = GP_I(11);
    int32_t *ids = GP_P(int32_t, 12);
    float *act = GP_P(float, 13);
    float *act2 = GP_P(float, 14);
    float *de = GP_P(float, 15);
    float *out = GP_P(float, 16);
    #pragma omp single
    {
        /* np.unique: the router gives distinct experts, so a sort is enough. */
        for (int j = 0; j < top_k; ++j) {
            ids[j] = idx[j];
        }
        for (int j = 1; j < top_k; ++j) {
            int32_t x = ids[j];
            int k = j - 1;
            while (k >= 0 && ids[k] > x) {
                ids[k + 1] = ids[k];
                --k;
            }
            ids[k + 1] = x;
        }
    }
    gemma_moe_gemv_gelu_body(gu_w, gu_s, h, ids, top_k, act, act2, gu_rows, cols, 0, inner);
    gemma_int4_moe_gemv_body(dn_w, dn_s, act2, ids, top_k, de, dn_rows, inner, inner);
    #pragma omp single
    {
        for (int c = 0; c < dn_rows; ++c) {
            out[c] = 0.0f;
        }
        for (int j = 0; j < top_k; ++j) {
            int s = 0;
            while (idx[s] != ids[j]) {
                ++s;
            }
            gp_add_scaled(out, de + (size_t)j * (size_t)dn_rows, val[s], dn_rows);
        }
    }
}

/* The key rows of query row j of a group: the first row lo and the count n.
 * The first query has the position pos, and row 0 of the buffer has the
 * position base. A sliding layer (window > 0) reads only the window. This is
 * the rule of Model._attention for a group. */
static inline void gp_rows(int64_t pos, int64_t base, int window, int j,
                           int *lo, int *n)
{
    int64_t p = pos + j;
    int64_t l = window > 0 ? p - window + 1 - base : 0;
    if (l < 0) {
        l = 0;
    }
    *lo = (int)l;
    *n = (int)(p + 1 - base - l);
}

/* The experts of a group of tokens, as Model._moe_mt does it.
 *
 * A pair is one (expert, token). The code sorts the pairs by expert with a
 * stable sort, so the pairs of one expert keep the order of their tokens.
 * The kernels then read each expert one time for all of its pairs. Last, each
 * token adds the outputs of its pairs in the order of the expert index, with
 * the router weights. */
static void gp_moe_group(const gp_rec *r, const int64_t *e)
{
    const float *h = GP_P(const float, 0);
    const float *val = GP_P(const float, 1);
    const int32_t *idx = GP_P(const int32_t, 2);
    int tokens = GP_I(3);
    int top_k = GP_I(4);
    const uint8_t *gu_w = GP_P(const uint8_t, 5);
    const float *gu_s = GP_P(const float, 6);
    const uint8_t *dn_w = GP_P(const uint8_t, 7);
    const float *dn_s = GP_P(const float, 8);
    int gu_rows = GP_I(9);
    int cols = GP_I(10);
    int dn_rows = GP_I(11);
    int inner = GP_I(12);
    int32_t *order = GP_P(int32_t, 13);    /* the flat index of each pair */
    int32_t *ids = GP_P(int32_t, 14);      /* the expert of each job */
    int32_t *poff = GP_P(int32_t, 15);     /* the first pair of each job */
    int32_t *xi = GP_P(int32_t, 16);       /* the token of each pair */
    int32_t *xi2 = GP_P(int32_t, 17);      /* the pair itself */
    int32_t *jobs_out = GP_P(int32_t, 18);
    float *act = GP_P(float, 19);
    float *act2 = GP_P(float, 20);
    float *de = GP_P(float, 21);
    float *out = GP_P(float, 22);
    const int pairs = tokens * top_k;
    #pragma omp single
    {
        /* A stable insertion sort of the flat indices by expert. */
        for (int f = 0; f < pairs; ++f) {
            int32_t x = f;
            int k = f - 1;
            while (k >= 0 && idx[order[k]] > idx[x]) {
                order[k + 1] = order[k];
                --k;
            }
            order[k + 1] = x;
        }
        /* A pair with a negative expert index is not for the CPU: the GPU
         * computes it (np_gemma/gpu.py). The sort puts such pairs first. */
        int s0 = 0;
        while (s0 < pairs && idx[order[s0]] < 0) {
            ++s0;
        }
        int jobs = 0;
        for (int p = s0; p < pairs; ++p) {
            int32_t ex = idx[order[p]];
            if (jobs == 0 || ids[jobs - 1] != ex) {
                ids[jobs] = ex;
                poff[jobs] = p;
                ++jobs;
            }
            xi[p] = order[p] / top_k;
            xi2[p] = p;
        }
        poff[jobs] = pairs;
        jobs_out[0] = jobs;
    }
    const int jobs = jobs_out[0];
    gemma_moe_gemv_gelu_mt_body(gu_w, gu_s, h, ids, poff, xi, jobs, act, act2,
                                gu_rows, cols, cols, inner);
    gemma_int4_moe_gemv_mt_body(dn_w, dn_s, act2, ids, poff, xi2, jobs, de,
                                dn_rows, inner, inner);
    #pragma omp single
    {
        for (int t = 0; t < tokens; ++t) {
            float *o = out + (size_t)t * (size_t)dn_rows;
            for (int c = 0; c < dn_rows; ++c) {
                o[c] = 0.0f;
            }
        }
        /* The pairs are in the order of the expert index. The pairs of one
         * token therefore come in that order too. */
        for (int p = 0; p < pairs; ++p) {
            int f = order[p];
            if (idx[f] < 0) {
                continue;
            }
            int t = f / top_k;
            gp_add_scaled(out + (size_t)t * (size_t)dn_rows,
                          de + (size_t)p * (size_t)dn_rows, val[f], dn_rows);
        }
    }
}

/* ---------- programs in parts ----------
 * SPLIT_PLAN.md, phase 1. A step can run as several programs, one for each
 * part of the machine. A part is, for example, one NUMA node. The function
 * gemma_run_parts runs each part in its own team of threads. A part computes
 * a range of the rows of the large operations, so each output value still
 * comes from one thread.
 *
 * A barrier across the parts (GP_XBAR) comes before an operation that reads
 * the output of a different part. The barrier is an int64 array of the
 * caller, with a flag for each part: b[8 (p + 1)], on a cache line of its
 * own. A flag counts the barriers that its part passed.
 *
 * At a barrier, a part stores its count plus one in its flag (a release
 * store), then waits until the flag of each other part is that count or
 * more (acquire loads). Only a part writes its flag, so no line takes an
 * atomic read-modify-write of all the parts. A count grows at each barrier,
 * so a flag needs no reset within a run; gemma_run_parts sets the flags to
 * 0 before a run.
 *
 * The OpenMP barrier at the start makes sure that each thread of the team
 * finished its writes. The release store then makes those writes visible to
 * the other parts. */
/* The measures of the barriers (gemma_xbar_stats): for each part (the
 * thread of the outer region), the time its team waited for the other
 * parts, the time of its team barrier, and the count. */
#define GP_XS_PARTS 8
static double gp_xs_wait[GP_XS_PARTS], gp_xs_team[GP_XS_PARTS];
static int64_t gp_xs_n[GP_XS_PARTS];

int gemma_xbar_stats(double *out)
{
    for (int p = 0; p < GP_XS_PARTS; ++p) {
        out[3 * p] = gp_xs_wait[p];
        out[3 * p + 1] = gp_xs_team[p];
        out[3 * p + 2] = (double)gp_xs_n[p];
        gp_xs_wait[p] = gp_xs_team[p] = 0.0;
        gp_xs_n[p] = 0;
    }
    return GP_XS_PARTS;
}

/* The count of the barriers that a thread passed. Each thread of a part
 * passes the same barriers, so each finds the same count; the runners of
 * the parts set it to 0 at the start of a run, as the flags. */
static __thread int64_t gp_xcnt;

/* The prefetch of a thread during the wait of a barrier: at most this many
 * bytes (L2 is 1 MB on the Xeon of the parts). */
#define GP_XPF_MAX (512 * 1024)

/* After the team barrier, the first thread of the team stores the count of
 * the part (release), and every thread waits for the flags of the other
 * parts itself (acquire). The barrier of the team makes the writes of the
 * team come before the store, so a thread that sees the flags of the other
 * parts sees their writes: no second team barrier.
 *
 * pf and pf_bytes (or null) are the weights of the next operation of the
 * part (parts.py, _xbar_prefetch). During the wait, each thread prefetches
 * its share of them (the share of the static schedule of the next kernel),
 * so a part that waits loads its next rows into its caches. */
static void gp_xbar(int64_t *b, int n, const uint8_t *pf, int64_t pf_bytes)
{
    double t0 = omp_get_wtime();
    int p = omp_get_ancestor_thread_num(1);
    int64_t v = ++gp_xcnt;
    #pragma omp barrier
    const int first = omp_get_thread_num() == 0;
    double t1 = omp_get_wtime();
    if (first) {
        __atomic_store_n(b + 8 * (p + 1), v, __ATOMIC_RELEASE);
    }
    const uint8_t *q = NULL, *qe = NULL;
    if (pf != NULL && pf_bytes > 0) {
        size_t nt = (size_t)omp_get_num_threads(), t = (size_t)omp_get_thread_num();
        size_t per = ((size_t)pf_bytes + nt - 1) / nt, lo = per * t;
        size_t len = lo >= (size_t)pf_bytes ? 0 : (size_t)pf_bytes - lo;
        len = len < per ? len : per;
        len = len < GP_XPF_MAX ? len : GP_XPF_MAX;
        q = pf + lo;
        qe = q + len;
    }
    for (int k = 0; k < n; ++k) {
        if (k == p) {
            continue;
        }
        while (__atomic_load_n(b + 8 * (k + 1), __ATOMIC_ACQUIRE) < v) {
#if GEMMA_X86
            if (q < qe) {
                for (int j = 0; j < 8 && q < qe; ++j, q += 64) {
                    _mm_prefetch((const char *)q, _MM_HINT_T1);
                }
            } else {
                _mm_pause();
            }
#endif
        }
    }
    if (first && p >= 0 && p < GP_XS_PARTS) {
        gp_xs_wait[p] += omp_get_wtime() - t1;
        gp_xs_team[p] += t1 - t0;
        ++gp_xs_n[p];
    }
}

/* The experts of one token, for the rows of one part. The part holds a copy
 * of some rows of each expert:
 *
 * - ni gate rows from inner row a0, and the ni up rows that go with them;
 * - nd down rows from output row c0.
 *
 * The steps:
 *
 * 1. Sort the selected experts, as gp_moe_one does.
 * 2. Run the gate and the up rows of the part and the GELU. Copy the result
 *    to columns a0 to a0 + ni of the shared activation (top_k, inner).
 * 3. Wait at the barrier until all parts wrote their columns.
 * 4. Run the down rows of the part on the whole activation.
 * 5. Add the outputs of the experts for rows c0 to c0 + nd of the output.
 *    Use the order of the expert index, as gp_moe_one does.
 *
 * Each value comes from the same instructions as in gp_moe_one, so the bits
 * are the same. This needs a0 and ni to be multiples of 16. The GELU uses 16
 * values in a vector and the rest one at a time. */
static void gp_moe_part(const gp_rec *r, const int64_t *e)
{
    const float *h = GP_P(const float, 0);
    const float *val = GP_P(const float, 1);
    const int32_t *idx = GP_P(const int32_t, 2);
    int top_k = GP_I(3);
    const uint8_t *gu_w = GP_P(const uint8_t, 4);
    const float *gu_s = GP_P(const float, 5);
    const uint8_t *dn_w = GP_P(const uint8_t, 6);
    const float *dn_s = GP_P(const float, 7);
    int gu_rows = GP_I(8);      /* 2 ni */
    int cols = GP_I(9);
    int nd = GP_I(10);
    int ni = GP_I(11);
    int32_t *ids = GP_P(int32_t, 12);
    float *act = GP_P(float, 13);     /* (top_k, 2 ni), private */
    float *act2p = GP_P(float, 14);   /* (top_k, ni), private */
    float *act2 = GP_P(float, 15);    /* (top_k, inner), shared */
    int a0 = GP_I(16);
    int inner = GP_I(17);
    float *de = GP_P(float, 18);      /* (top_k, nd), private */
    float *out = GP_P(float, 19);     /* (hidden), shared */
    int c0 = GP_I(20);
    int64_t *bar = GP_P(int64_t, 21);
    int nparts = GP_I(22);
    if (gu_s == NULL) {
        /* The KQ_Q4X copies of the rows of this part (np_gemma/parts.py):
         * the operations of GP_KQ_MOE with float32 activations (the step of
         * one part), row by row: the same bits. gu_w holds 2 ni rows for
         * each expert (the gate rows, then the up rows), dn_w nd rows. */
        size_t rbh = (size_t)(cols / 32) * 18, rbi = (size_t)(inner / 32) * 18;
        int ggu = gu_rows / 16, gdn = nd / 16;
        /* AVX2: int8 x, as GP_KQ_MOE on that CPU (q4x_rows_x) */
        I4Q_BEGIN(h, (size_t)cols, 1, cols);
        #pragma omp for schedule(static)
        for (int x = 0; x < top_k * ggu; ++x) {
            int j = x / ggu, g = x % ggu;
            q4x_rows_x(gu_w + ((size_t)idx[j] * gu_rows + 16 * g) * rbh, cols, h, (size_t)cols, 1,
                       act + (size_t)j * gu_rows + 16 * g, (size_t)gu_rows);
        }
        #pragma omp for schedule(static)
        for (int j = 0; j < top_k; ++j) {
            const float *a = act + (size_t)j * gu_rows;
            float *o = act2 + (size_t)j * inner + a0;
            int i = 0;
#if GEMMA_X86 && defined(__AVX512F__)
            /* the GELU of kq_moe_body, 16 values at a time */
            const __m512 cv = _mm512_set1_ps(0.7978845608028654f), half = _mm512_set1_ps(0.5f);
            const __m512 one = _mm512_set1_ps(1.0f), k3 = _mm512_set1_ps(0.044715f);
            for (; i + 16 <= ni; i += 16) {
                __m512 v = _mm512_loadu_ps(a + i);
                __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
                __m512 th = gemma_tanh_ps(_mm512_mul_ps(cv, _mm512_fmadd_ps(k3, v3, v)));
                __m512 rr = _mm512_mul_ps(_mm512_mul_ps(half, v), _mm512_add_ps(one, th));
                _mm512_storeu_ps(o + i, _mm512_mul_ps(rr, _mm512_loadu_ps(a + ni + i)));
            }
#else
            i = GELU_MUL_VEC(a, a + ni, o, ni);
#endif
            for (; i < ni; ++i) {
                float v = a[i];
                o[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) *
                       a[ni + i];
            }
        }
        gp_xbar(bar, nparts, NULL, 0);
        I4Q_BEGIN(act2, (size_t)inner, top_k, inner);
        #pragma omp for schedule(static)
        for (int x = 0; x < top_k * gdn; ++x) {
            int j = x / gdn, g = x % gdn;
            q4x_rows_x(dn_w + ((size_t)idx[j] * nd + 16 * g) * rbi, inner, act2 + (size_t)j * inner,
                       (size_t)inner, 1, de + (size_t)j * nd + 16 * g, (size_t)nd);
        }
        /* the sum of moe_combine: from 0, the experts in the order of the
         * router */
        #pragma omp for schedule(static)
        for (int c = 0; c < nd; ++c) {
            float o = 0.f;
            for (int j = 0; j < top_k; ++j) {
                o += val[j] * de[(size_t)j * nd + c];
            }
            out[c0 + c] = o;
        }
        return;
    }
    #pragma omp single
    {
        for (int j = 0; j < top_k; ++j) {
            ids[j] = idx[j];
        }
        for (int j = 1; j < top_k; ++j) {
            int32_t x = ids[j];
            int k = j - 1;
            while (k >= 0 && ids[k] > x) {
                ids[k + 1] = ids[k];
                --k;
            }
            ids[k + 1] = x;
        }
    }
    gemma_moe_gemv_gelu_body(gu_w, gu_s, h, ids, top_k, act, act2p, gu_rows, cols, 0, ni);
    #pragma omp single
    for (int j = 0; j < top_k; ++j) {
        memcpy(act2 + (size_t)j * (size_t)inner + a0, act2p + (size_t)j * (size_t)ni,
               (size_t)ni * sizeof(float));
    }
    gp_xbar(bar, nparts, NULL, 0);
    gemma_int4_moe_gemv_body(dn_w, dn_s, act2, ids, top_k, de, nd, inner, inner);
    #pragma omp single
    {
        for (int c = 0; c < nd; ++c) {
            out[c0 + c] = 0.0f;
        }
        for (int j = 0; j < top_k; ++j) {
            int s = 0;
            while (idx[s] != ids[j]) {
                ++s;
            }
            gp_add_scaled(out + c0, de + (size_t)j * (size_t)nd, val[s], nd);
        }
    }
}

static void gp_step(const gp_rec *r, int64_t *e)
{
    switch (r->op) {
    case GP_MA_QUANT:
        /* x, t, cols, xq4, xq8, xs, xsum */
        ma_quant_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_P(int8_t, 3), GP_P(int8_t, 4),
                      GP_P(float, 5), GP_P(float, 6));
        break;
    case GP_MA_LINEAR:
        /* xq, xs, xsum, w, s, b, bits, rows, cols, t, out */
        ma_linear_body(GP_P(const uint32_t, 3), GP_P(const uint16_t, 4), GP_P(const uint16_t, 5),
                       GP_I(6), GP_I(7), GP_I(8), GP_P(const int8_t, 0), GP_P(const float, 1),
                       GP_P(const float, 2), GP_I(9), GP_P(float, 10));
        break;
    case GP_MA_MOE:
        /* hq4, hq8, hs, hsum, ids, val, t, k, experts, mats, shared_logit, hidden,
         * inner, scratch, out */
        ma_moe_body(GP_P(const int8_t, 0), GP_P(const int8_t, 1), GP_P(const float, 2),
                    GP_P(const float, 3), GP_P(const int32_t, 4), GP_P(const float, 5), GP_I(6),
                    GP_I(7), GP_I(8), GP_P(const int64_t, 9), GP_P(const float, 10), GP_I(11),
                    GP_I(12), GP_P(uint8_t, 13), GP_P(float, 14));
        break;
    case GP_ROUTER_TOPK:
        /* logits, t, experts, k, val, idx */
        gp_router_topk_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_I(3), GP_P(float, 4),
                            GP_P(int32_t, 5));
        break;
    case GP_GDN:
        /* qkv, conv, conv_w, kernel, z, a, b, A_log, dt_bias, norm_w, S, out, scratch,
         * t, k_heads, v_heads, k_dim, v_dim, eps, log (null, or an MTP verify group),
         * tiled (the order of the value heads) */
        gdn_body(GP_P(const float, 0), GP_P(float, 1), GP_P(const float, 2), GP_I(3),
                 GP_P(const float, 4), GP_P(const float, 5), GP_P(const float, 6),
                 GP_P(const float, 7), GP_P(const float, 8), GP_P(const float, 9), GP_P(float, 10),
                 GP_P(float, 11), GP_P(float, 12), GP_I(13), GP_I(14), GP_I(15), GP_I(16),
                 GP_I(17), gp_f(r, e, 18), GP_P(float, 19), GP_I(20));
        break;
    case GP_HC_NORM:
        /* x, w, out, t, groups, hid, eps */
        hc_norm_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_I(3), GP_I(4),
                     GP_I(5), gp_f(r, e, 6));
        break;
    case GP_HC_ACT:
        /* x, out, n, scale */
        hc_act_body(GP_P(const float, 0), GP_P(float, 1), gp_i(r, e, 2), gp_f(r, e, 3));
        break;
    case GP_HC_MIX:
        /* hn, g, out, t, hc, hid */
        hc_mix_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_I(3), GP_I(4),
                    GP_I(5));
        break;
    case GP_HC_ADD:
        /* H, out, inject, t, hc, hid, scale */
        hc_add_body(GP_P(float, 0), GP_P(const float, 1), GP_P(const float, 2), GP_I(3), GP_I(4),
                    GP_I(5), gp_f(r, e, 6));
        break;
    case GP_PLE_GATE:
        /* keyn, qn, value, gated, t, hc, hid */
        ple_gate_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2),
                      GP_P(float, 3), GP_I(4), GP_I(5), GP_I(6));
        break;
    case GP_PLE_CONV:
        /* gn, gated, H, state, w, t, channels, kernel, dilation */
        ple_conv_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_P(float, 3),
                      GP_P(const float, 4), GP_I(5), GP_I(6), GP_I(7), GP_I(8));
        break;
    case GP_MOE_PLAN:
        /* ip, nreal, t, k, E, slots, desc, cpu_a, cpu_b, gpu_c, tab, gidx, cidx, ranges,
         * stats (moe.c) */
        moe_plan_body(GP_P(const int32_t, 0), GP_P(const int64_t, 1), GP_I(2), GP_I(3), GP_I(4),
                      GP_P(const int32_t, 5), GP_P(const int64_t, 6), gp_i(r, e, 7),
                      gp_i(r, e, 8), gp_i(r, e, 9), GP_P(int64_t, 10), GP_P(int32_t, 11),
                      GP_P(int32_t, 12), GP_P(int64_t, 13), GP_P(int64_t, 14), GP_P(int32_t, 15));
        break;
    case GP_HC_CAT:
        /* e, hn, out, t, hc, hid */
        hc_cat_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_I(3), GP_I(4),
                    GP_I(5));
        break;
    case GP_QSA_SELECT:
        /* iq, ik, idxk, blk, qn, kn, cos, sin, pos, t, heads, d, ratio, budget, rot, theta,
         * eps, sel, cnt, maxsel, scratch, nbmax, qpos (or 0), sec */
        qsa_select_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(uint16_t, 2),
                        GP_P(uint16_t, 3),
                        GP_P(const float, 4), GP_P(const float, 5), GP_P(const float, 6),
                        GP_P(const float, 7), gp_i(r, e, 8), GP_I(9), GP_I(10), GP_I(11), GP_I(12),
                        GP_I(13), GP_I(14), gp_f(r, e, 15), gp_f(r, e, 16), GP_P(int32_t, 17),
                        GP_P(int32_t, 18), GP_I(19), GP_P(uint8_t, 20), gp_i(r, e, 21),
                        GP_P(const int32_t, 22), GP_I(23));
        break;
    case GP_ATTN_QSA:
        /* q, kq, ks, vq, vs, scores, out, nq, nk, hd, t, pos, sel, cnt, maxsel,
         * form (0 int16, 1 int8, 2 int16 keys and int8 values, 3 float32
         * rows), hs (form 3: the head stride) */
        attn_qsa_body(GP_P(const float, 0), GP_P(const void, 1), GP_P(const float, 2),
                      GP_P(const void, 3), GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                      GP_I(7), GP_I(8), GP_I(9), GP_I(10), gp_i(r, e, 11), GP_P(const int32_t, 12),
                      GP_P(const int32_t, 13), GP_I(14), GP_I(15), gp_i(r, e, 16));
        break;
    case GP_KQ_QUANT:
        /* x, t, cols, xq, xs, xm, then (or none) ids and k: only the rows
         * with a selected expert */
        if (GP_P(const int32_t, 6) != NULL) {
            kq_quant_rows_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_P(int8_t, 3),
                               GP_P(float, 4), GP_P(float, 5), GP_P(const int32_t, 6), GP_I(7));
        } else {
            kq_quant_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_P(int8_t, 3),
                          GP_P(float, 4), GP_P(float, 5));
        }
        break;
    case GP_KQ_LINEAR:
        /* xq, xs, xm, x, w, type, rows, cols, t, out */
        kq_linear_body(GP_P(const uint8_t, 4), GP_I(5), GP_I(6), GP_I(7), GP_P(const int8_t, 0),
                       GP_P(const float, 1), GP_P(const float, 2), GP_P(const float, 3), GP_I(8),
                       GP_P(float, 9));
        break;
    case GP_KQ_MOE:
        /* hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
         * out, kcount (null, or the count of experts of one token), act, hf, mats1 (the
         * copy of node 1, or null), slot1 (a partial mats1: the slot of each expert, or -1;
         * null: a full copy), slot0 (null: mats holds every expert; else the slot of each
         * expert in mats, or -1: a split of the experts over the nodes) */
        kq_moe_body(GP_P(const int8_t, 0), GP_P(const float, 1), GP_P(const float, 2),
                    GP_P(const int32_t, 3), GP_P(const float, 4), GP_I(5), GP_I(6), GP_I(7),
                    GP_P(const int64_t, 8), GP_P(const float, 9), GP_I(10), GP_I(11),
                    GP_P(uint8_t, 12), GP_P(float, 13),
                    GP_P(const int32_t, 14), GP_I(15), GP_P(const float, 16),
                    GP_P(const int64_t, 17), GP_P(const int32_t, 18), GP_P(const int32_t, 19));
        break;
    case GP_ATTN_PREP:
        /* qg, kk, vv, qn, kn, cos, sin, K, V, hs, pos, t, nq, nk, hd, rot, eps, scale,
         * qout, gate */
        gp_attn_prep_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2),
                          GP_P(const float, 3), GP_P(const float, 4), GP_P(const float, 5),
                          GP_P(const float, 6), GP_P(float, 7), GP_P(float, 8), gp_i(r, e, 9),
                          gp_i(r, e, 10), GP_I(11), GP_I(12), GP_I(13), GP_I(14), GP_I(15),
                          gp_f(r, e, 16), gp_f(r, e, 17), GP_P(float, 18), GP_P(float, 19),
                          GP_P(float, 20));
        break;
    case GP_SIGMUL:
        /* x, g, out, n */
        gp_sigmul_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2),
                       gp_i(r, e, 3));
        break;
    /* ---- scalar operations: every thread, private copy ---- */
    case GP_S_MOV: e[r->v[0]] = gp_i(r, e, 1); break;
    case GP_S_ADD: e[r->v[0]] = gp_i(r, e, 1) + gp_i(r, e, 2); break;
    case GP_S_SUB: e[r->v[0]] = gp_i(r, e, 1) - gp_i(r, e, 2); break;
    case GP_S_MUL: e[r->v[0]] = gp_i(r, e, 1) * gp_i(r, e, 2); break;
    case GP_S_MAX: {
        int64_t a = gp_i(r, e, 1), b = gp_i(r, e, 2);
        e[r->v[0]] = a > b ? a : b;
        break;
    }
    case GP_S_MIN: {
        int64_t a = gp_i(r, e, 1), b = gp_i(r, e, 2);
        e[r->v[0]] = a < b ? a : b;
        break;
    }
    /* ---- one-row operations ---- */
    case GP_RMS_NORM: {
        /* x, w (0 for none), out, rows, cols, eps */
        const float *x = GP_P(const float, 0);
        const float *w = GP_P(const float, 1);
        float *out = GP_P(float, 2);
        int rows = GP_I(3), cols = GP_I(4);
        float eps = gp_f(r, e, 5);
        if (rows < 8) {
            #pragma omp single
            for (int i = 0; i < rows; ++i) {
                gemma_rms_norm_row(x + (size_t)i * (size_t)cols, w,
                                   out + (size_t)i * (size_t)cols, cols, eps);
            }
        } else {
            /* the rows of a prompt over the team, as gemma_rms_norm */
            gemma_rms_norm_body(x, w, out, rows, cols, eps);
        }
        break;
    }
    case GP_ADD: {
        /* a, b, out, n: the team adds (each value as one thread did) */
        const float *a = GP_P(const float, 0);
        const float *b = GP_P(const float, 1);
        float *out = GP_P(float, 2);
        int n = GP_I(3);
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
        break;
    }
    case GP_MUL_S: {
        /* x, s (float), out, n */
        const float *x = GP_P(const float, 0);
        float sc = gp_f(r, e, 1);
        float *out = GP_P(float, 2);
        int n = GP_I(3);
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            out[i] = x[i] * sc;
        }
        break;
    }
    case GP_COPY: {
        /* src, dst, bytes */
        const void *src = GP_P(const void, 0);
        void *dst = GP_P(void, 1);
        size_t n = (size_t)gp_i(r, e, 2);
        #pragma omp single
        memcpy(dst, src, n);
        break;
    }
    case GP_GELU:
        /* Operands: x, out, n. As ops.gelu_tanh. */
        gemma_gelu_body(GP_P(const float, 0), GP_P(float, 1), GP_I(2));
        break;
    case GP_MUL: {
        /* Operands: a, b, out, rows, cols, b_stride. out = a * b for each
         * value. Row r of b starts at r * b_stride. Thus b can be a slice of a
         * wider array, for example the per-layer input of the E4B model. */
        const float *a = GP_P(const float, 0);
        const float *b = GP_P(const float, 1);
        float *out = GP_P(float, 2);
        int rows = GP_I(3), cols = GP_I(4);
        size_t bs = (size_t)gp_i(r, e, 5);
        /* the values over the team (a row of a prompt has 15360) */
        #pragma omp for schedule(static)
        for (long ic = 0; ic < (long)rows * cols; ++ic) {
            const long i = ic / cols, c = ic % cols;
            out[(size_t)ic] = a[(size_t)ic] * b[(size_t)i * bs + (size_t)c];
        }
        break;
    }
    case GP_BF16_LINEAR:
        /* Operands: x, w, out, rows, cols, tokens. A bfloat16 matrix. */
        gemma_bf16_linear_body(GP_P(const uint16_t, 1), GP_P(const float, 0),
                               GP_P(float, 2), GP_I(3), GP_I(4), GP_I(5));
        break;
    /* ---- int4 matrices ---- */
    case GP_INT4_LINEAR:
        /* x, w, s, out, rows, cols */
        gemma_int4_linear_body(GP_P(const uint8_t, 1), GP_P(const float, 2),
                               GP_P(const float, 0), GP_P(float, 3),
                               GP_I(4), GP_I(5), 1, 32);
        break;
    case GP_INT4_MULTI4:
        /* x, cols, then (w, s, out, rows) four times */
        gemma_int4_multi4_body(GP_P(const uint8_t, 2), GP_P(const float, 3), GP_P(float, 4), GP_I(5),
                               GP_P(const uint8_t, 6), GP_P(const float, 7), GP_P(float, 8), GP_I(9),
                               GP_P(const uint8_t, 10), GP_P(const float, 11), GP_P(float, 12), GP_I(13),
                               GP_P(const uint8_t, 14), GP_P(const float, 15), GP_P(float, 16), GP_I(17),
                               GP_P(const float, 0), GP_I(1));
        break;
    case GP_RMS_NORM_MULTI4:
        /* x, wn, scratch, cols, eps, then (w, s, out, rows) four times */
        gemma_rms_norm_multi4_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2),
                                   GP_I(3), gp_f(r, e, 4),
                                   GP_P(const uint8_t, 5), GP_P(const float, 6), GP_P(float, 7), GP_I(8),
                                   GP_P(const uint8_t, 9), GP_P(const float, 10), GP_P(float, 11), GP_I(12),
                                   GP_P(const uint8_t, 13), GP_P(const float, 14), GP_P(float, 15), GP_I(16),
                                   GP_P(const uint8_t, 17), GP_P(const float, 18), GP_P(float, 19), GP_I(20));
        break;
    case GP_GELU_MUL_INT4:
        /* g, u, inner, scratch, w, s, out, rows, cols */
        gemma_gelu_mul_int4_body(GP_P(const float, 0), GP_P(const float, 1), GP_I(2),
                                 GP_P(float, 3), GP_P(const uint8_t, 4), GP_P(const float, 5),
                                 GP_P(float, 6), GP_I(7), GP_I(8));
        break;
    case GP_INT4_LINEAR_MT:
        /* Operands: x, w, s, out, rows, cols, tokens. */
        gemma_int4_linear_mt_body(GP_P(const uint8_t, 1), GP_P(const float, 2),
                                  GP_P(const float, 0), GP_P(float, 3),
                                  GP_I(4), GP_I(5), GP_I(6));
        break;
    case GP_INT4_MULTI4_MT:
        /* Operands: x, cols, tokens, then (w, s, out, rows) four times. */
        gemma_int4_multi4_mt_body(GP_P(const uint8_t, 3), GP_P(const float, 4), GP_P(float, 5), GP_I(6),
                                  GP_P(const uint8_t, 7), GP_P(const float, 8), GP_P(float, 9), GP_I(10),
                                  GP_P(const uint8_t, 11), GP_P(const float, 12), GP_P(float, 13), GP_I(14),
                                  GP_P(const uint8_t, 15), GP_P(const float, 16), GP_P(float, 17), GP_I(18),
                                  GP_P(const float, 0), GP_I(1), GP_I(2));
        break;
    case GP_GELU_MUL_ROWS: {
        /* Operands: g, u, out, rows, inner. One call for each row, as
         * ops.gelu_mul_rows does, so the tail of each row stays the same. */
        const float *g = GP_P(const float, 0);
        const float *u = GP_P(const float, 1);
        float *out = GP_P(float, 2);
        int rows = GP_I(3), inner = GP_I(4);
        #pragma omp single
        for (int i = 0; i < rows; ++i) {
            gemma_gelu_mul_pair(g + (size_t)i * (size_t)inner, u + (size_t)i * (size_t)inner,
                                out + (size_t)i * (size_t)inner, inner);
        }
        break;
    }
    /* ---- attention ---- */
    case GP_QKV_NORM_ROPE:
        /* q, q_w, q_rows, k, k_w, k_rows, v, v_rows, cos, sin, q_heads,
         * k_heads, head_dim, eps */
        gemma_qkv_norm_rope_body(GP_P(float, 0), GP_P(const float, 1), GP_I(2),
                                 GP_P(float, 3), GP_P(const float, 4), GP_I(5),
                                 GP_P(float, 6), GP_I(7),
                                 GP_P(const float, 8), GP_P(const float, 9),
                                 GP_I(10), GP_I(11), GP_I(12), gp_f(r, e, 13));
        break;
    case GP_KV_WRITE: {
        /* Operands: k, v, kd, vd, kqd, ksd, vqd, vsd, n.
         * Store the rows of the float cache and of its int16 copy. A cache
         * with no int16 copy gives null addresses. */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        float *kd = GP_P(float, 2);
        float *vd = GP_P(float, 3);
        int16_t *kqd = GP_P(int16_t, 4);
        float *ksd = GP_P(float, 5);
        int16_t *vqd = GP_P(int16_t, 6);
        float *vsd = GP_P(float, 7);
        int n = GP_I(8);
        if (kd != NULL) {
            #pragma omp single nowait
            {
                memcpy(kd, k, (size_t)n * sizeof(float));
                memcpy(vd, v, (size_t)n * sizeof(float));
            }
        }
        if (kqd != NULL) {
            /* the groups over the team (the rows of a prompt block): each
             * group as one thread did it */
            #pragma omp for schedule(static)
            for (int g = 0; g < n / 32; ++g) {
                ksd[g] = gemma_quant_group32_i16(k + (size_t)g * 32, kqd + (size_t)g * 32);
                vsd[g] = gemma_quant_group32_i16(v + (size_t)g * 32, vqd + (size_t)g * 32);
            }
        } else {
            #pragma omp barrier
        }
        break;
    }
    case GP_KV_WRITE8: {
        /* Operands: k, v, kd, vd, kqd, ksd, vqd, vsd, n.
         * GP_KV_WRITE for the int8 cache (NP_GEMMA_KV_INT8). */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        float *kd = GP_P(float, 2);
        float *vd = GP_P(float, 3);
        int8_t *kqd = GP_P(int8_t, 4);
        float *ksd = GP_P(float, 5);
        int8_t *vqd = GP_P(int8_t, 6);
        float *vsd = GP_P(float, 7);
        int n = GP_I(8);
        #pragma omp single
        {
            if (kd != NULL) {
                memcpy(kd, k, (size_t)n * sizeof(float));
                memcpy(vd, v, (size_t)n * sizeof(float));
            }
            for (int g = 0; g < n / 32; ++g) {
                ksd[g] = gemma_quant_group32_i8(k + (size_t)g * 32, kqd + (size_t)g * 32);
                vsd[g] = gemma_quant_group32_i8(v + (size_t)g * 32, vqd + (size_t)g * 32);
            }
        }
        break;
    }
    case GP_KV_WRITEV8: {
        /* Operands: those of GP_KV_WRITE8. The keys int16, the values int8,
         * no float rows (KVCache kv="k16v8"). */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        int16_t *kqd = GP_P(int16_t, 4);
        float *ksd = GP_P(float, 5);
        int8_t *vqd = GP_P(int8_t, 6);
        float *vsd = GP_P(float, 7);
        int n = GP_I(8);
        #pragma omp single
        for (int g = 0; g < n / 32; ++g) {
            ksd[g] = gemma_quant_group32_i16(k + (size_t)g * 32, kqd + (size_t)g * 32);
            vsd[g] = gemma_quant_group32_i8(v + (size_t)g * 32, vqd + (size_t)g * 32);
        }
        break;
    }
    case GP_KV_WRITETQ: {
        /* Operands: those of GP_KV_WRITE8; the bytes of the TQ6 groups and
         * their norms (tq6_quant_group), no float rows. */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        uint8_t *kqd = GP_P(uint8_t, 4);
        float *ksd = GP_P(float, 5);
        uint8_t *vqd = GP_P(uint8_t, 6);
        float *vsd = GP_P(float, 7);
        int n = GP_I(8);
        #pragma omp for schedule(static)
        for (int g = 0; g < n / 32; ++g) {
            ksd[g] = tq6_quant_group(k + (size_t)g * 32, kqd + (size_t)g * 24);
            vsd[g] = tq6_quant_group(v + (size_t)g * 32, vqd + (size_t)g * 24);
        }
        break;
    }
    case GP_TQ_ROT: {
        /* Operands: x, groups, inverse. The rotation of TQ6 (tq6_rot32) of
         * each group of 32 values of x, in place, or its inverse. */
        float *x = GP_P(float, 0);
        int groups = GP_I(1), inv = GP_I(2);
        #pragma omp for schedule(static)
        for (int g = 0; g < groups; ++g) {
            tq6_rot32(x + (size_t)g * 32, inv);
        }
        break;
    }
    case GP_ATTN_TQ: {
        /* Operands: those of GP_ATTN_QC, over the TQ6 cache; q and out in
         * the rotated form (GP_TQ_ROT). */
        int n1 = (int)GP_I(10);
        if (!gemma_attn_split_tq6_body(GP_P(const float, 0), GP_P(const uint8_t, 1),
                                       GP_P(const float, 2), GP_P(const uint8_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), NULL, &n1, n1, 1)) {
            gp_fail = 1;
        }
        break;
    }
    case GP_ATTN_TQ_MT: {
        /* Operands: those of GP_ATTN_QC_MT, over the TQ6 cache (GP_ATTN_TQ). */
        int tokens = GP_I(10), window = GP_I(13);
        int64_t pos = gp_i(r, e, 11), base = gp_i(r, e, 12);
        int *lo = GP_P(int, 14);
        int *n = GP_P(int, 15);
        #pragma omp single
        for (int j = 0; j < tokens; ++j) {
            gp_rows(pos, base, window, j, lo + j, n + j);
        }
        int nmax = 0;
        for (int j = 0; j < tokens; ++j) {
            nmax = n[j] > nmax ? n[j] : nmax;
        }
        if (!gemma_attn_split_tq6_body(GP_P(const float, 0), GP_P(const uint8_t, 1),
                                       GP_P(const float, 2), GP_P(const uint8_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens)) {
            gp_fail = 1;
        }
        break;
    }
    case GP_ATTN_Q8:
    case GP_ATTN_V8: {
        /* Operands: those of GP_ATTN_QC, over the int8 cache (GP_ATTN_V8: the
         * keys int16). */
        int n1 = (int)GP_I(10);
        if (r->op == GP_ATTN_V8) {
            if (!gemma_attn_split_v8_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                          GP_P(const float, 2), GP_P(const int8_t, 3),
                                          GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                          GP_I(7), GP_I(8), GP_I(9), NULL, &n1, n1, 1)) {
                gp_fail = 1;
            }
        } else if (!gemma_attn_split_i8_body(GP_P(const float, 0), GP_P(const int8_t, 1),
                                      GP_P(const float, 2), GP_P(const int8_t, 3),
                                      GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                      GP_I(7), GP_I(8), GP_I(9), NULL, &n1, n1, 1)) {
            gp_fail = 1;
        }
        break;
    }
    case GP_ATTN_Q8_MT:
    case GP_ATTN_V8_MT: {
        /* Operands: those of GP_ATTN_QC_MT, over the int8 cache (GP_ATTN_V8_MT:
         * the keys int16). */
        int tokens = GP_I(10), window = GP_I(13);
        int64_t pos = gp_i(r, e, 11), base = gp_i(r, e, 12);
        int *lo = GP_P(int, 14);
        int *n = GP_P(int, 15);
        #pragma omp single
        for (int j = 0; j < tokens; ++j) {
            gp_rows(pos, base, window, j, lo + j, n + j);
        }
        int nmax = 0;
        for (int j = 0; j < tokens; ++j) {
            nmax = n[j] > nmax ? n[j] : nmax;
        }
        int ok = r->op == GP_ATTN_V8_MT ?
            gemma_attn_split_v8_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                     GP_P(const float, 2), GP_P(const int8_t, 3),
                                     GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                     GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens) :
            gemma_attn_split_i8_body(GP_P(const float, 0), GP_P(const int8_t, 1),
                                     GP_P(const float, 2), GP_P(const int8_t, 3),
                                     GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                     GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens);
        if (!ok) {
            gp_fail = 1;
        }
        break;
    }
    case GP_ATTN_QC: {
        /* Operands: q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
         * head_dim, n.
         * The attention of one float32 query over the int16 cache: split by
         * keys (gemma_attn_split_i16_body), or one head for each thread. */
        int n1 = (int)GP_I(10);
        if (!gemma_attn_split_i16_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                       GP_P(const float, 2), GP_P(const int16_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), NULL, &n1, n1, 1)) {
            gemma_attn_decode_i16_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                       GP_P(const float, 2), GP_P(const int16_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), GP_I(10));
        }
        break;
    }
    case GP_ATTN_F32:
        /* Operands: q, k, v, scores, out, q_heads, kv_heads, head_dim, n,
         * pos, base, window.
         * The attention of one query over the float cache. k and v point at
         * the row of position base. */
        gemma_attn_decode_f32_body(GP_P(const float, 0), GP_P(const float, 1),
                                   GP_P(const float, 2), GP_P(float, 3), GP_P(float, 4),
                                   GP_I(5), GP_I(6), GP_I(7), GP_I(8),
                                   GP_I(7), GP_I(7), (long)GP_I(6) * GP_I(7),
                                   (long)GP_I(6) * GP_I(7), GP_I(9), GP_I(10), GP_I(11));
        break;
    case GP_ATTN_QC_MT: {
        /* Operands: q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
         * head_dim, tokens, pos, base, window, lo, n.
         * The attention of a group of float32 queries over the int16 cache.
         * The cache addresses point at buffer row 0. lo and n are scratch. */
        int tokens = GP_I(10), window = GP_I(13);
        int64_t pos = gp_i(r, e, 11), base = gp_i(r, e, 12);
        int *lo = GP_P(int, 14);
        int *n = GP_P(int, 15);
        #pragma omp single
        for (int j = 0; j < tokens; ++j) {
            gp_rows(pos, base, window, j, lo + j, n + j);
        }
        int nmax = 0;
        for (int j = 0; j < tokens; ++j) {
            nmax = n[j] > nmax ? n[j] : nmax;
        }
        /* The scores buffer of a group holds tokens * q_heads * (pos +
         * tokens) values (program.step_params), so the split form fits. */
        if (!gemma_attn_split_i16_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                       GP_P(const float, 2), GP_P(const int16_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens)) {
            gemma_attn_decode_i16_mt_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                          GP_P(const float, 2), GP_P(const int16_t, 3),
                                          GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                          GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens);
        }
        break;
    }
    case GP_ATTN_F32_MT: {
        /* Operands: q, k, v, scores, out, q_heads, kv_heads, head_dim,
         * tokens, pos, base, window.
         * The attention of a group of queries over the float cache. The code
         * runs one query at a time, as Model._attend_one does it. k and v
         * point at buffer row 0. */
        const float *q = GP_P(const float, 0);
        const float *k = GP_P(const float, 1);
        const float *v = GP_P(const float, 2);
        float *out = GP_P(float, 4);
        int q_heads = GP_I(5), kv_heads = GP_I(6), head_dim = GP_I(7);
        int tokens = GP_I(8), window = GP_I(11);
        int64_t pos = gp_i(r, e, 9), base = gp_i(r, e, 10);
        long row = (long)kv_heads * head_dim;
        size_t qd = (size_t)q_heads * (size_t)head_dim;
        for (int j = 0; j < tokens; ++j) {
            int lo, n;
            gp_rows(pos, base, window, j, &lo, &n);
            gemma_attn_decode_f32_body(q + j * qd, k + (size_t)lo * (size_t)row,
                                       v + (size_t)lo * (size_t)row, GP_P(float, 3),
                                       out + j * qd, q_heads, kv_heads, head_dim, n,
                                       head_dim, head_dim, row, row,
                                       (int)(pos + j), (int)(base + lo), window);
        }
        break;
    }
    case GP_QKV_NORM:
        /* Operands: q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps.
         *
         * A layer that reuses the key and the value of an earlier layer gives
         * null for k and v. */
        gemma_qkv_norm_body(GP_P(float, 0), GP_P(const float, 1), GP_I(2),
                            GP_P(float, 3), GP_P(const float, 4), GP_I(5),
                            GP_P(float, 6), GP_I(7), GP_I(8), gp_f(r, e, 9));
        break;
    case GP_ROPE:
        /* Operands: q, q_rows, q_heads, k, k_rows, k_heads, cos, sin,
         * head_dim. */
        gemma_rope_body(GP_P(float, 0), GP_I(1), GP_I(2), GP_P(float, 3), GP_I(4),
                        GP_I(5), GP_P(const float, 6), GP_P(const float, 7), GP_I(8));
        break;
    case GP_KV_WRITE_HEADS: {
        /* Operands: k, v, kbuf, vbuf, head_stride, pos, tokens, kv_heads,
         * head_dim. A float32 cache (the E4B): each head of each token goes
         * to its place (gp_kv_rs: position-major or head-major). */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        float *kb = GP_P(float, 2);
        float *vb = GP_P(float, 3);
        size_t hs = (size_t)gp_i(r, e, 4);
        int64_t pos = gp_i(r, e, 5);
        int tokens = GP_I(6), kv_heads = GP_I(7), head_dim = GP_I(8);
        #pragma omp single
        for (int j = 0; j < tokens; ++j) {
            for (int h = 0; h < kv_heads; ++h) {
                size_t src = ((size_t)j * (size_t)kv_heads + (size_t)h) * (size_t)head_dim;
                size_t dst = (size_t)h * hs + (size_t)(pos + j) * gp_kv_rs(hs, kv_heads, head_dim);
                memcpy(kb + dst, k + src, (size_t)head_dim * sizeof(float));
                memcpy(vb + dst, v + src, (size_t)head_dim * sizeof(float));
            }
        }
        break;
    }
    case GP_ATTN_F32H: {
        /* Operands: q, k, v, scores, out, q_heads, kv_heads, head_dim,
         * tokens, pos, head_stride, window, slide.
         * The attention of one or more queries over the E4B cache. The code
         * runs one query at a time, as E4B.attention does it. k and v point at
         * position 0.
         * With slide, a sliding layer reads only the rows of its window. */
        const float *q = GP_P(const float, 0);
        const float *k = GP_P(const float, 1);
        const float *v = GP_P(const float, 2);
        float *out = GP_P(float, 4);
        int q_heads = GP_I(5), kv_heads = GP_I(6), head_dim = GP_I(7);
        int tokens = GP_I(8), window = GP_I(11), slide = GP_I(12);
        int64_t pos = gp_i(r, e, 9);
        long hs = (long)gp_i(r, e, 10);
        long rs = (long)gp_kv_rs((size_t)hs, kv_heads, head_dim);
        size_t qd = (size_t)q_heads * (size_t)head_dim;
        for (int j = 0; j < tokens; ++j) {
            int64_t p = pos + j;
            int64_t lo = (slide && window > 0) ? p - window + 1 : 0;
            if (lo < 0) {
                lo = 0;
            }
            gemma_attn_decode_f32_body(q + j * qd, k + (size_t)lo * (size_t)rs,
                                       v + (size_t)lo * (size_t)rs, GP_P(float, 3),
                                       out + j * qd, q_heads, kv_heads, head_dim,
                                       (int)(p + 1 - lo), hs, hs, rs, rs,
                                       (int)p, (int)lo, window);
        }
        break;
    }
    /* ---- mixture of experts ---- */
    case GP_ROUTER:
        /* x, scale, proj, per_expert, hidden, experts, top_k, eps, hscale,
         * val, idx, r, logits */
        gemma_router_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2),
                          GP_P(const float, 3), GP_I(4), GP_I(5), GP_I(6),
                          gp_f(r, e, 7), gp_f(r, e, 8), GP_P(float, 9), GP_P(int, 10),
                          GP_P(float, 11), GP_P(float, 12));
        break;
    case GP_MOE:
    case GP_MOE_N:
        gp_moe_one(r, e);
        break;
    case GP_ROUTER_MT:
        /* Operands: x, scale, proj, per_expert, hidden, experts, top_k, eps,
         * hscale, val, idx, tokens, r, logits. */
        gemma_router_mt_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2),
                             GP_P(const float, 3), GP_I(4), GP_I(5), GP_I(6),
                             gp_f(r, e, 7), gp_f(r, e, 8), GP_P(float, 9), GP_P(int, 10),
                             GP_I(11), GP_P(float, 12), GP_P(float, 13));
        break;
    case GP_MOE_MT:
        gp_moe_group(r, e);
        break;
    /* ---- programs in parts ---- */
    case GP_XBAR:
        /* bar, parts, then (or null) the weights to prefetch and their bytes */
        gp_xbar(GP_P(int64_t, 0), GP_I(1), GP_P(const uint8_t, 2), gp_i(r, e, 3));
        break;
    case GP_MOE_PART:
        gp_moe_part(r, e);
        break;
    case GP_ATTN_QC_H: {
        /* Operands: q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
         * head_dim, n, kv_stride, ks_stride, rep.
         * GP_ATTN_QC for some heads. The strides are the values of a whole
         * cache row. rep is the query heads of a key head in the layer. As
         * GP_ATTN_QC: split by keys over the threads of the team, or one
         * head for each thread. The choice is that of GP_ATTN_QC for the
         * whole layer (rep heads of an item), so each head has its bits: a
         * part with fewer query heads of a key head (parts.head_split) does
         * not take the split form where the layer does not. */
        const int hd = GP_I(9);
        const size_t kvs = (size_t)gp_i(r, e, 11);
        if (GP_I(13) * hd > AS_MAXQ ||
            kvs % (size_t)hd != 0 || (size_t)gp_i(r, e, 12) != kvs / 32 ||
            !gemma_attn_split_i16_rows_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                            GP_P(const float, 2), GP_P(const int16_t, 3),
                                            GP_P(const float, 4), GP_P(float, 6), GP_I(7),
                                            GP_I(8), hd, GP_I(10), (int)(kvs / (size_t)hd))) {
            gemma_attn_decode_i16_s_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                         GP_P(const float, 2), GP_P(const int16_t, 3),
                                         GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                         GP_I(7), GP_I(8), hd, GP_I(10), kvs,
                                         (size_t)gp_i(r, e, 12));
        }
        break;
    }
    case GP_ATTN_F32_H:
        /* Operands: q, k, v, scores, out, q_heads, kv_heads, head_dim, n,
         * pos, base, window, row.
         * GP_ATTN_F32 for some heads. row is the value count of a whole
         * cache row. */
        gemma_attn_decode_f32_body(GP_P(const float, 0), GP_P(const float, 1),
                                   GP_P(const float, 2), GP_P(float, 3), GP_P(float, 4),
                                   GP_I(5), GP_I(6), GP_I(7), GP_I(8),
                                   GP_I(7), GP_I(7), (long)gp_i(r, e, 12),
                                   (long)gp_i(r, e, 12), GP_I(9), GP_I(10), GP_I(11));
        break;
    /* ---- the media encoders: rows split over the threads ---- */
    case GP_ENC_LINEAR: {
        /* x, w, wbf, b, y, n, m, k, imin, imax, omin, omax, scratch, x16:
         * y (n, m) = clamp(clamp(x, imin, imax) W^T + b, omin, omax). The
         * scratch (n k floats, or 0) holds the clamped x. x16 (or 0) is W in
         * groups of 16 rows (KQ_BF16X16 or KQ_F32X16, kq_pack_x16f): the
         * product of many rows (kq_x16f_body). */
        const float *x = GP_P(const float, 0);
        const float *b = GP_P(const float, 3);
        float *y = GP_P(float, 4);
        int n = GP_I(5), m = GP_I(6), k = GP_I(7);
        float imin = gp_f(r, e, 8), imax = gp_f(r, e, 9);
        float omin = gp_f(r, e, 10), omax = gp_f(r, e, 11);
        float *xs = GP_P(float, 12);
        if (xs != NULL) {
            int64_t nk = (int64_t)n * k;
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < nk; ++i) {
                xs[i] = fminf(fmaxf(x[i], imin), imax);
            }
            x = xs;
        }
        const uint8_t *x16 = GP_P(const uint8_t, 13);
        if (x16 != NULL) {
            kq_x16f_body(x16, GP_I(2) ? KQ_BF16X16 : KQ_F32X16, m, k, x, n, y);
        } else if (GP_I(2)) {
            gemma_bf16_linear_body(GP_P(const uint16_t, 1), x, y, m, k, n);
        } else {
            const float *w = GP_P(const float, 1);
            #pragma omp for schedule(static)
            for (int c = 0; c < m; ++c) {
                const float *wc = w + (size_t)c * k;
                for (int t = 0; t < n; ++t) {
                    const float *xt = x + (size_t)t * k;
                    float acc = 0.0f;
                    for (int q = 0; q < k; ++q) {
                        acc += xt[q] * wc[q];
                    }
                    y[(size_t)t * m + c] = acc;
                }
            }
        }
        if (b != NULL || omin > -INFINITY || omax < INFINITY) {
            int64_t nm = (int64_t)n * m;
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < nm; ++i) {
                float v = y[i] + (b != NULL ? b[i % m] : 0.0f);
                y[i] = fminf(fmaxf(v, omin), omax);
            }
        }
        break;
    }
    case GP_ENC_RMS: {
        /* x, w (0 for none), y, rows, cols, eps */
        const float *x = GP_P(const float, 0);
        const float *w = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int rows = GP_I(3), cols = GP_I(4);
        float eps = gp_f(r, e, 5);
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            gemma_rms_norm_row(x + (size_t)i * cols, w, y + (size_t)i * cols, cols, eps);
        }
        break;
    }
    case GP_ENC_LNORM: {
        /* x, w, b, y, rows, cols, eps: the LayerNorm of each row (w and b may
         * be 0). y may be x. */
        const float *x = GP_P(const float, 0);
        const float *w = GP_P(const float, 1);
        const float *b = GP_P(const float, 2);
        float *y = GP_P(float, 3);
        int rows = GP_I(4), cols = GP_I(5);
        float eps = gp_f(r, e, 6);
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            const float *p = x + (size_t)i * cols;
            float *q = y + (size_t)i * cols;
            float s = 0.f;
            for (int j = 0; j < cols; ++j) {
                s += p[j];
            }
            float mean = s / (float)cols, v = 0.f;
            for (int j = 0; j < cols; ++j) {
                float d = p[j] - mean;
                v += d * d;
            }
            float rs = 1.0f / sqrtf(v / (float)cols + eps);
            for (int j = 0; j < cols; ++j) {
                float t = (p[j] - mean) * rs;
                q[j] = t * (w != NULL ? w[j] : 1.0f) + (b != NULL ? b[j] : 0.0f);
            }
        }
        break;
    }
    case GP_Q6K_LINEAR:
        /* w, x, out, rows, cols, tokens: as gemma_q6k_linear. A row has the
         * bits of that call, so a range of rows gives the same values. */
        gemma_q6k_linear_body(GP_P(const uint8_t, 0), GP_P(const float, 1), GP_P(float, 2),
                              GP_I(3), GP_I(4), GP_I(5));
        break;
    case GP_KQ_QUANT16:
        /* x, t, cols, xq (int16, t x cols), xs (t x cols / 32) */
        kq_quant16_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_P(int16_t, 3), GP_P(float, 4));
        break;
    case GP_KQ_LINEAR16:
        /* w (KQ_Q4X), rows, cols, xq, xs, t, out (t x rows) */
        kq_linear16_body(GP_P(const uint8_t, 0), GP_I(1), GP_I(2), GP_P(const int16_t, 3),
                         GP_P(const float, 4), GP_I(5), GP_P(float, 6));
        #pragma omp barrier
        break;
    case GP_ATTN_PREFILL_QC:
        /* q, kq, ks, vq, vs (the cache from row 0), positions (t int32),
         * base, window, out, t, n (the buffer rows after the block), q
         * heads, kv heads, head_dim, limit (null, or the last key of each
         * query: media) */
        gemma_attn_prefill_qc_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                   GP_P(const float, 2), GP_P(const int16_t, 3),
                                   GP_P(const float, 4), GP_P(const int32_t, 5), GP_I(6),
                                   GP_I(7), GP_P(float, 8), GP_I(9), GP_I(10), GP_I(11),
                                   GP_I(12), GP_I(13), GP_P(const int32_t, 14));
        #pragma omp barrier
        break;
    case GP_PART_PREFILL: {
        /* q (t, qh, hd), k, v (t, kvh, hd): the block after the norms and the
         * rope; out (t, qh, hd); kq, ks, vq, vs: the int16 cache of the part,
         * rows of nk = g1 - g0 heads; kf, vf: (rows, nk, hd) float scratch;
         * qp, op: (t, nq, hd) float scratch; positions (t int32); ip (int32):
         * t, start (the buffer row of the block), rows (the buffer rows after
         * the block), base, window, qh, kvh, hd, g0, g1 (the key and value
         * heads of the part), h0, h1 (its query heads, nq = h1 - h0); limit
         * (null, or the last key of each query: media).
         *
         * As KVCache.write, KVCache.read, and gemma_attn_prefill for the
         * heads g0 to g1 - 1 and their query heads: each value has the
         * operations of the run of all heads, so the bits are the same. */
        const int32_t *ip = GP_P(const int32_t, 13);
        const int t = ip[0], start = ip[1], rows = ip[2], base = ip[3], window = ip[4];
        const int qh = ip[5], kvh = ip[6], hd = ip[7], g0 = ip[8], g1 = ip[9];
        const int h0 = ip[10], h1 = ip[11];
        const int nk = g1 - g0, nq = h1 - h0, gpr = hd / 32;
        const float *q = GP_P(const float, 0), *k = GP_P(const float, 1);
        const float *v = GP_P(const float, 2);
        float *out = GP_P(float, 3);
        int16_t *kq = GP_P(int16_t, 4), *vq = GP_P(int16_t, 6);
        float *ks = GP_P(float, 5), *vs = GP_P(float, 7);
        float *kf = GP_P(float, 8), *vf = GP_P(float, 9);
        float *qp = GP_P(float, 10), *op = GP_P(float, 11);
        const long ng = (long)nk * gpr;
        /* the rows of the block into the cache of the part */
        #pragma omp for schedule(static)
        for (long i = 0; i < (long)t * ng; ++i) {
            const long j = i / ng, r = i % ng;
            const size_t src = ((size_t)j * kvh + g0) * hd + (size_t)r * 32;
            const size_t row = (size_t)(start + j) * ng + r;
            ks[row] = gemma_quant_group32_i16(k + src, kq + row * 32);
            vs[row] = gemma_quant_group32_i16(v + src, vq + row * 32);
        }
        /* the query heads of the part, adjacent */
        #pragma omp for schedule(static)
        for (long i = 0; i < (long)t * nq; ++i) {
            const long j = i / nq, h = i % nq;
            memcpy(qp + (size_t)i * hd, q + ((size_t)j * qh + (size_t)h0 + h) * hd,
                   (size_t)hd * sizeof(float));
        }
        /* the attention over the int16 cache of the part (no float copy) */
        (void)kf;
        (void)vf;
        if (gemma_attn_prefill_qc_ok(nq, nk, hd)) {
            gemma_attn_prefill_qc_body(qp, kq, ks, vq, vs, GP_P(const int32_t, 12), base, window,
                                       op, t, rows, nq, nk, hd, GP_P(const int32_t, 14));
        } else {
            #pragma omp for schedule(static)
            for (long g = 0; g < (long)rows * ng; ++g) {
                const float sk = ks[g], sv = vs[g];
                for (int e = 0; e < 32; ++e) {
                    kf[g * 32 + e] = (float)kq[g * 32 + e] * sk;
                    vf[g * 32 + e] = (float)vq[g * 32 + e] * sv;
                }
            }
            gemma_attn_prefill_region_lim(qp, kf, vf, GP_P(const int32_t, 12), base, window, op,
                                          t, rows, nq, nk, hd, GP_P(const int32_t, 14));
        }
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (long i = 0; i < (long)t * nq; ++i) {
            const long j = i / nq, h = i % nq;
            memcpy(out + ((size_t)j * qh + (size_t)h0 + h) * hd, op + (size_t)i * hd,
                   (size_t)hd * sizeof(float));
        }
        break;
    }
    case GP_ENC_GELU: {
        /* x, y, n, erf: y = gelu(x), the tanh form (erf 0) or the erf form */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2);
        int use_erf = GP_I(3);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            float v = x[i];
            y[i] = use_erf ? 0.5f * v * (1.0f + erff(v * 0.7071067811865476f))
                           : 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        }
        break;
    }
    case GP_ENC_GELU_MUL: {
        /* g, u, y, n: y = gelu_tanh(g) u, in blocks of 4096 values */
        const float *g = GP_P(const float, 0);
        const float *u = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int64_t n = gp_i(r, e, 3);
        int64_t nb = (n + 4095) / 4096;
        #pragma omp for schedule(static)
        for (int64_t bi = 0; bi < nb; ++bi) {
            int64_t o = bi * 4096, len = n - o < 4096 ? n - o : 4096;
            gemma_gelu_mul_pair(g + o, u + o, y + o, (int)len);
        }
        break;
    }
    case GP_ENC_ADD: {
        /* x, y, n, s: x += s y */
        float *x = GP_P(float, 0);
        const float *y = GP_P(const float, 1);
        int64_t n = gp_i(r, e, 2);
        float sc = gp_f(r, e, 3);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            x[i] += sc * y[i];
        }
        break;
    }
    case GP_ENC_CLAMP: {
        /* x, y, n, lo, hi: y = clamp(x, lo, hi) */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2);
        float lo = gp_f(r, e, 3), hi = gp_f(r, e, 4);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            y[i] = fminf(fmaxf(x[i], lo), hi);
        }
        break;
    }
    case GP_ENC_BIAS_CLAMP: {
        /* y, b (or 0), rows, cols, lo, hi: y = clamp(y + b, lo, hi) in place */
        float *y = GP_P(float, 0);
        const float *b = GP_P(const float, 1);
        int rows = GP_I(2), cols = GP_I(3);
        float lo = gp_f(r, e, 4), hi = gp_f(r, e, 5);
        int64_t n = (int64_t)rows * cols;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            y[i] = fminf(fmaxf(y[i] + (b != NULL ? b[i % cols] : 0.0f), lo), hi);
        }
        break;
    }
    case GP_ENC_SILU: {
        /* x, y, n: y = x sigmoid(x) */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            y[i] = x[i] / (1.0f + expf(-x[i]));
        }
        break;
    }
    case GP_ENC_MUL_VEC: {
        /* x, vec, y, rows, cols: y = x vec (vec of cols values for each row) */
        const float *x = GP_P(const float, 0);
        const float *vec = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int rows = GP_I(3), cols = GP_I(4);
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            for (int c = 0; c < cols; ++c) {
                y[(size_t)i * cols + c] = x[(size_t)i * cols + c] * vec[c];
            }
        }
        break;
    }
    case GP_ENC_GLU: {
        /* x (rows, 2 cols), y (rows, cols), rows, cols: y = a sigmoid(b) with
         * a the first half of a row and b the second */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int rows = GP_I(2), cols = GP_I(3);
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            const float *xa = x + (size_t)i * 2 * cols;
            for (int c = 0; c < cols; ++c) {
                y[(size_t)i * cols + c] = xa[c] / (1.0f + expf(-xa[cols + c]));
            }
        }
        break;
    }
    case GP_ENC_DWCONV: {
        /* x (t, c), w (c, kw), y (t, c), t, c, kw: the causal depthwise conv
         * y[i, c] = sum_j w[c, j] x[i - kw + 1 + j, c] (0 before the start) */
        const float *x = GP_P(const float, 0);
        const float *w = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int t = GP_I(3), c = GP_I(4), kw = GP_I(5);
        #pragma omp for schedule(static)
        for (int i = 0; i < t; ++i) {
            for (int ch = 0; ch < c; ++ch) {
                float acc = 0.0f;
                for (int j = 0; j < kw; ++j) {
                    int src = i - kw + 1 + j;
                    if (src >= 0) {
                        acc += w[(size_t)ch * kw + j] * x[(size_t)src * c + ch];
                    }
                }
                y[(size_t)i * c + ch] = acc;
            }
        }
        break;
    }
    case GP_ENC_LOCAL_ATTN: {
        /* q, k, v (t, heads, hd), r (span + 1, heads, hd), valid (t int32), o,
         * t, heads, hd, span, cap: query i sees the keys i - span + 1 .. i;
         * key j of that list adds q r[1 + j] (the relative position); the
         * scores get the soft cap, and an invalid key -1e9 (gemma4a). */
        const float *q = GP_P(const float, 0), *k = GP_P(const float, 1);
        const float *v = GP_P(const float, 2), *rr = GP_P(const float, 3);
        const int32_t *valid = GP_P(const int32_t, 4);
        float *o = GP_P(float, 5);
        int t = GP_I(6), heads = GP_I(7), hd = GP_I(8), span = GP_I(9);
        float cap = gp_f(r, e, 10);
        int64_t rows = (int64_t)t * heads;
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            int i = (int)(row / heads), h = (int)(row % heads);
            const float *qi = q + row * hd;
            float sc[64];
            float m = -INFINITY;
            for (int j = 0; j < span && j < 64; ++j) {
                int kp = i - span + 1 + j;
                float sv = -1e9f;
                if (kp >= 0 && valid[kp]) {
                    const float *kk = k + ((size_t)kp * heads + h) * hd;
                    const float *rj = rr + ((size_t)(1 + j) * heads + h) * hd;
                    float a = 0.0f;
                    for (int d = 0; d < hd; ++d) {
                        a += qi[d] * (kk[d] + rj[d]);
                    }
                    sv = tanhf(a / cap) * cap;
                }
                sc[j] = sv;
                m = fmaxf(m, sv);
            }
            float sum = 0.0f;
            for (int j = 0; j < span; ++j) {
                sc[j] = expf(sc[j] - m);
                sum += sc[j];
            }
            float *oi = o + row * hd;
            for (int d = 0; d < hd; ++d) {
                oi[d] = 0.0f;
            }
            for (int j = 0; j < span; ++j) {
                int kp = i - span + 1 + j;
                if (kp < 0) {
                    continue;
                }
                float pj = sc[j] / sum;
                const float *vv = v + ((size_t)kp * heads + h) * hd;
                for (int d = 0; d < hd; ++d) {
                    oi[d] += pj * vv[d];
                }
            }
        }
        break;
    }
    case GP_ENC_ROPE2D: {
        /* x (n, heads, hd), pos (n, 2) int32, inv (hd / 4), n, heads, hd: the
         * axial 2D RoPE of gemma4v in place (see k_enc_rope2d of gpu.cu). */
        float *x = GP_P(float, 0);
        const int32_t *pos = GP_P(const int32_t, 1);
        const float *inv = GP_P(const float, 2);
        int n = GP_I(3), heads = GP_I(4), hd = GP_I(5), q4 = hd / 4, h2 = hd / 2;
        int64_t rows = (int64_t)n * heads;
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            int patch = (int)(row / heads);
            for (int part = 0; part < 2; ++part) {
                float *p = x + row * hd + part * h2;
                for (int j = 0; j < q4; ++j) {
                    float ang = (float)pos[2 * patch + part] * inv[j];
                    float c = cosf(ang), sn = sinf(ang);
                    float a = p[j], bb = p[j + q4];
                    p[j] = a * c - bb * sn;
                    p[j + q4] = bb * c + a * sn;
                }
            }
        }
        break;
    }
    case GP_ENC_ATTN: {
        /* q, k, v, o, n, heads, hd, positions: the attention of every query
         * over every key, scale 1. positions (n int32) holds n - 1 for each
         * query, so the causal limit of the prompt kernel keeps every key. */
        gemma_attn_prefill_region(GP_P(const float, 0), GP_P(const float, 1),
                                  GP_P(const float, 2), GP_P(const int32_t, 7), 0, 0,
                                  GP_P(float, 3), GP_I(4), GP_I(4), GP_I(5), GP_I(5), GP_I(6));
        break;
    }
    default:
        break;
    }
}

/* Run the records of one program. The caller opens the parallel region. */
static void gp_exec(const int64_t *prog, int limit)
{
    const int n_env = (int)prog[1];
    int n_code = (int)prog[2];
    const int64_t *env0 = prog + 4;
    const gp_rec *code = (const gp_rec *)(env0 + n_env);
    if (limit >= 0 && limit < n_code) {
        n_code = limit;
    }
    int64_t *e = (int64_t *)malloc((size_t)(n_env > 0 ? n_env : 1) * sizeof(int64_t));
    memcpy(e, env0, (size_t)n_env * sizeof(int64_t));
    for (int pc = 0; pc < n_code; ++pc) {
        gp_step(code + pc, e);
    }
    free(e);
}

/* gp_exec with the time of each record: the first thread of the team adds
 * the ms of record pc to ms[pc]. No barrier is added. Most records end with
 * the barrier of their omp for or omp single, so the time of the first
 * thread is the time of the team; a record with no barrier gives only the
 * share of the first thread, and the next record with a barrier gets the
 * rest. */
static void gp_exec_prof(const int64_t *prog, double *ms)
{
    const int n_env = (int)prog[1];
    const int n_code = (int)prog[2];
    const int64_t *env0 = prog + 4;
    const gp_rec *code = (const gp_rec *)(env0 + n_env);
    int64_t *e = (int64_t *)malloc((size_t)(n_env > 0 ? n_env : 1) * sizeof(int64_t));
    memcpy(e, env0, (size_t)n_env * sizeof(int64_t));
    const int first = omp_get_thread_num() == 0;
    double t0 = omp_get_wtime();
    for (int pc = 0; pc < n_code; ++pc) {
        gp_step(code + pc, e);
        if (first) {
            double t1 = omp_get_wtime();
            ms[pc] += (t1 - t0) * 1e3;
            t0 = t1;
        }
    }
    free(e);
}

/* Run a program with a barrier after each record, and the time of each
 * record in ms (CPU_PLAN.md, phase 0). ms has one value for each record.
 * The barriers add a little time; the times show where the step goes. */
int gemma_profile(const int64_t *prog, double *ms)
{
    if (prog[0] != GP_MAGIC) {
        return -1;
    }
    #pragma omp parallel
    {
        const int n_env = (int)prog[1];
        const int n_code = (int)prog[2];
        const int64_t *env0 = prog + 4;
        const gp_rec *code = (const gp_rec *)(env0 + n_env);
        int64_t *e = (int64_t *)malloc((size_t)(n_env > 0 ? n_env : 1) * sizeof(int64_t));
        memcpy(e, env0, (size_t)n_env * sizeof(int64_t));
        #pragma omp barrier
        double t0 = omp_get_wtime();
        for (int pc = 0; pc < n_code; ++pc) {
            gp_step(code + pc, e);
            #pragma omp barrier
            #pragma omp master
            {
                double t1 = omp_get_wtime();
                ms[pc] = (t1 - t0) * 1e3;
                t0 = t1;
            }
        }
        free(e);
    }
    return 0;
}

int gemma_run(const int64_t *prog, int limit)
{
    if (prog[0] != GP_MAGIC) {
        return -1;
    }
    gp_fail = 0;
    int was = gp_planned;
    gp_planned = 1;
    #pragma omp parallel
    gp_exec(prog, limit);
    gp_planned = was;
    return gp_fail ? -3 : 0;
}

/* The CPU programs that a GPU program runs (GP_CPU_TASK, GP_CPU_JOIN: the
 * cold experts of a step, gg_set_cpu_runner): a team of gp_task_threads
 * threads bound spread over the places. On the 2-socket Xeon (48 cores),
 * Qwen3.8 on the GPU decoded 26.0 tok/s with 24 threads spread, 21.3 with
 * all 48 bound close (OMP_PROC_BIND close, the default of np_gemma): the
 * threads of the GPU runner and of Python keep cores, and the experts read
 * the memory of both nodes. 0 (the default) takes OMP_NUM_THREADS.
 * gemma_set_task_threads sets it (np_gemma/gpu.py, NP_GEMMA_GPU_CPU_THREADS). */
static int gp_task_threads;
void gemma_set_task_threads(int n)
{
    gp_task_threads = n;
}

/* The team of gemma_run_task on a machine of two or more NUMA nodes, pinned
 * with the same count of threads on each node (as far as the CPUs of a node
 * allow; NP_GEMMA_TEAM_BALANCE, default 1), out of NP_GEMMA_RESERVED_CPUS. proc_bind(spread) over the 40
 * CPUs of taskset 0-19,24-43 put 24 threads as 10 on node 0 and 14 on node 1
 * (it takes the places in pairs, then one by one), while kq_share_range
 * gives the tasks of node 0 to the first half of the team. Here thread i <
 * n0 takes a CPU of node 0 (evenly spaced in the allowed CPUs of the node),
 * the others those of node 1, ...; the allowed CPUs are those of the places
 * of OpenMP (OMP_PLACES, made from the affinity of the process when the
 * runtime starts; the main thread of Python is pinned later). A thread pins itself
 * when sched_getcpu is not its CPU: one call at the first region of a team
 * size, then none. 0: proc_bind(spread) only. */
/* The CPUs of the places of OpenMP (all of them; 0 when there are none). */
static int gp_place_cpus(cpu_set_t *m)
{
    CPU_ZERO(m);
    int np = omp_get_num_places(), any = 0;
    for (int p = 0; p < np; ++p) {
        int k = omp_get_place_num_procs(p);
        int ids[256];
        if (k <= 0 || k > 256) {
            continue;
        }
        omp_get_place_proc_ids(p, ids);
        for (int i = 0; i < k; ++i) {
            if (ids[i] >= 0 && ids[i] < CPU_SETSIZE) {
                CPU_SET(ids[i], m);
                any = 1;
            }
        }
    }
    return any;
}

static int gp_team_balance_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_TEAM_BALANCE");
        on = !(v && v[0] == '0');
    }
    return on;
}

#define GP_TEAM_MAX 256
static int gp_team_cpu[GP_TEAM_MAX];
static int gp_team_n = -1;          /* the size gp_team_cpu is made for; 0: no plan */

/* gp_team_cpu for n threads; return 0 when there is no plan (one node, too
 * few CPUs, or no mask). Called by the master before the region. */
static int gp_team_plan(int n)
{
    if (n == gp_team_n) {
        return gp_team_n > 0;
    }
    gp_team_n = 0;
    cpu_set_t gp_start_mask;
    if (!gp_team_balance_on() || n < 2 || n > GP_TEAM_MAX || !gp_place_cpus(&gp_start_mask)) {
        return 0;
    }
    pthread_once(&kq_cpu_node_once, kq_cpu_node_init);
    /* the allowed CPUs of each node, in order */
    static int cpus[64][GP_TEAM_MAX];
    int cnt[64] = {0}, nodes = 0, total = 0;
    /* NP_GEMMA_RESERVED_CPUS ("1,19"): the CPUs of the main thread and of the
     * copy workers (np_gemma.gpu), out of the teams */
    static unsigned char rsv[4096];
    memset(rsv, 0, sizeof(rsv));
    const char *rv = getenv("NP_GEMMA_RESERVED_CPUS");
    for (const char *q = rv; q != NULL && *q;) {
        char *e;
        int k = (int)strtol(q, &e, 10);
        if (e == q) {
            break;
        }
        if (k >= 0 && k < 4096) {
            rsv[k] = 1;
        }
        if (*e != ',') {
            break;
        }
        q = e + 1;
    }
    for (int c = 0; c < 4096 && c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &gp_start_mask) || rsv[c]) {
            continue;
        }
        int nd = kq_cpu_node_tab[c];
        if (nd < 0 || nd >= 64 || cnt[nd] >= GP_TEAM_MAX) {
            continue;
        }
        cpus[nd][cnt[nd]++] = c;
        ++total;
        nodes = nd + 1 > nodes ? nd + 1 : nodes;
    }
    int used = 0;
    for (int nd = 0; nd < nodes; ++nd) {
        used += cnt[nd] > 0;
    }
    if (used < 2 || n > total) {
        return 0;
    }
    /* the same count of threads on each node (n / nodes), a node with too
     * few CPUs giving the rest to the others (the prompt team of 38 with
     * CPUs 1 and 19 reserved: 18 and 20) */
    int want[64] = {0}, left = n, room = 0;
    for (int nd = 0; nd < nodes; ++nd) {
        room += cnt[nd] > 0;
    }
    while (left > 0 && room > 0) {
        int share = (left + room - 1) / room, given = 0;
        for (int nd = 0; nd < nodes && left > 0; ++nd) {
            int free_nd = cnt[nd] - want[nd];
            if (free_nd <= 0) {
                continue;
            }
            int g = share < free_nd ? share : free_nd;
            g = g < left ? g : left;
            want[nd] += g;
            left -= g;
            given += g;
        }
        room = 0;
        for (int nd = 0; nd < nodes; ++nd) {
            room += cnt[nd] > want[nd];
        }
        if (given == 0) {
            break;
        }
    }
    int k = 0;
    for (int nd = 0; nd < nodes; ++nd) {
        int m = want[nd];
        for (int i = 0; i < m && k < n; ++i) {
            gp_team_cpu[k++] = cpus[nd][(int)((int64_t)i * cnt[nd] / m)];
        }
    }
    if (k != n) {
        return 0;
    }
    gp_team_n = n;
    return 1;
}

/* In the region: pin this thread to its CPU of the plan, if it is not there. */
static inline void gp_team_pin(void)
{
    int tid = omp_get_thread_num(), want = gp_team_cpu[tid];
    if (sched_getcpu() != want) {
        cpu_set_t m;
        CPU_ZERO(&m);
        CPU_SET(want, &m);
        sched_setaffinity(0, sizeof(m), &m);
    }
}

int gemma_run_task(const int64_t *prog, int limit)
{
    if (prog[0] != GP_MAGIC) {
        return -1;
    }
    /* word 3 of the header: the team of this program, or 0 (program.Program
     * threads; the CPU part of a mixed prompt group takes more threads than
     * the step) */
    int n = prog[3] > 0 ? (int)prog[3] : gp_task_threads;
    if (n <= 0) {
        return gemma_run(prog, limit);
    }
    gp_fail = 0;
    int plan = gp_team_plan(n);
    int was = gp_planned;
    gp_planned = 1;
    #pragma omp parallel num_threads(n) proc_bind(spread)
    {
        if (plan && omp_get_num_threads() == n) {
            gp_team_pin();
        }
        gp_exec(prog, limit);
    }
    gp_planned = was;
    return gp_fail ? -3 : 0;
}

/* Run nparts programs at the same time. Each program runs in a team of team
 * threads. The array progs holds the address of each program. The outer
 * region has one thread for each part, spread over the places of OMP_PLACES.
 * Each of those threads opens a team on the places near it.
 *
 * A team of 0 divides the threads of OMP_NUM_THREADS by the parts. The array
 * bar is the barrier of the programs (see gp_xbar). This function sets it to
 * 0 first. */
/* gemma_run_parts with the time of each record of each part (gp_exec_prof):
 * part p adds to ms[p * stride + pc]. With gemma_xbar_stats, this gives the
 * time of each operation of each part and the wait at the barriers. */
int gemma_run_parts_prof(const int64_t *const *progs, int nparts, int team, int64_t *bar,
                         double *ms, int stride)
{
    for (int p = 0; p < nparts; ++p) {
        if (progs[p][0] != GP_MAGIC) {
            return -1;
        }
    }
    if (team <= 0) {
        team = omp_get_max_threads() / nparts;
        if (team < 1) {
            team = 1;
        }
    }
    for (int p = 0; p <= nparts; ++p) {
        bar[8 * p] = 0;
    }
    omp_set_max_active_levels(2);
    #pragma omp parallel num_threads(nparts) proc_bind(spread)
    {
        const int p = omp_get_thread_num();
        #pragma omp parallel num_threads(team) proc_bind(close)
        {
            gp_xcnt = 0;
            gp_exec_prof(progs[p], ms + (size_t)p * (size_t)stride);
        }
    }
    return 0;
}

int gemma_run_parts(const int64_t *const *progs, int nparts, int team, int64_t *bar)
{
    for (int p = 0; p < nparts; ++p) {
        if (progs[p][0] != GP_MAGIC) {
            return -1;
        }
    }
    if (team <= 0) {
        team = omp_get_max_threads() / nparts;
        if (team < 1) {
            team = 1;
        }
    }
    for (int p = 0; p <= nparts; ++p) {
        bar[8 * p] = 0;
    }
    omp_set_max_active_levels(2);
    #pragma omp parallel num_threads(nparts) proc_bind(spread)
    {
        const int64_t *prog = progs[omp_get_thread_num()];
        #pragma omp parallel num_threads(team) proc_bind(close)
        {
            gp_xcnt = 0;
            gp_exec(prog, -1);
        }
    }
    return 0;
}

extern int sched_getcpu(void);

/* The CPU of each thread of gemma_run_parts, with the same teams: cpus[p *
 * team + t] for thread t of part p, or -1 for a thread that did not run. The
 * threads are bound (OMP_PLACES, OMP_PROC_BIND), so a run of the parts uses
 * these CPUs too. np_gemma/numa.py gives each part the node of its CPUs.
 * Return the team. */
int gemma_part_cpus(int nparts, int team, int *cpus)
{
    if (team <= 0) {
        team = omp_get_max_threads() / nparts;
        if (team < 1) {
            team = 1;
        }
    }
    for (int i = 0; i < nparts * team; ++i) {
        cpus[i] = -1;
    }
    omp_set_max_active_levels(2);
    #pragma omp parallel num_threads(nparts) proc_bind(spread)
    {
        const int p = omp_get_thread_num();
        #pragma omp parallel num_threads(team) proc_bind(close)
        {
            const int t = omp_get_thread_num();
            if (t < team) {
                cpus[p * team + t] = sched_getcpu();
            }
        }
    }
    return team;
}

