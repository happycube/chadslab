/* Measure the peak AVX-512 float speed and the int8 convert speed. */
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(void)
{
    long n = 200000000L;
    int threads = omp_get_max_threads();
    printf("threads %d\n", threads);

    /* 1. Pure fused multiply and add. Eight independent chains. */
    double t0 = now();
    float sink = 0.0f;
    #pragma omp parallel reduction(+:sink)
    {
        __m512 a = _mm512_set1_ps(1.0000001f);
        __m512 b = _mm512_set1_ps(1.0000002f);
        __m512 c0 = _mm512_set1_ps(0.1f), c1 = _mm512_set1_ps(0.2f);
        __m512 c2 = _mm512_set1_ps(0.3f), c3 = _mm512_set1_ps(0.4f);
        __m512 c4 = _mm512_set1_ps(0.5f), c5 = _mm512_set1_ps(0.6f);
        __m512 c6 = _mm512_set1_ps(0.7f), c7 = _mm512_set1_ps(0.8f);
        #pragma omp for schedule(static)
        for (long i = 0; i < n; ++i) {
            c0 = _mm512_fmadd_ps(a, b, c0); c1 = _mm512_fmadd_ps(a, b, c1);
            c2 = _mm512_fmadd_ps(a, b, c2); c3 = _mm512_fmadd_ps(a, b, c3);
            c4 = _mm512_fmadd_ps(a, b, c4); c5 = _mm512_fmadd_ps(a, b, c5);
            c6 = _mm512_fmadd_ps(a, b, c6); c7 = _mm512_fmadd_ps(a, b, c7);
        }
        sink += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3)));
        sink += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(c4, c5), _mm512_add_ps(c6, c7)));
    }
    double t1 = now();
    printf("pure FMA      : %8.1f GFLOP/s  (sink %.1f)\n", 8.0 * n * 16 * 2 / (t1 - t0) / 1e9, sink);

    /* 2. Convert 16 int8 to float, then one FMA with it. */
    static signed char bytes[64] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    t0 = now();
    sink = 0.0f;
    #pragma omp parallel reduction(+:sink)
    {
        __m512 x = _mm512_set1_ps(1.0001f);
        __m512 c0 = _mm512_set1_ps(0.1f), c1 = _mm512_set1_ps(0.2f);
        __m512 c2 = _mm512_set1_ps(0.3f), c3 = _mm512_set1_ps(0.4f);
        #pragma omp for schedule(static)
        for (long i = 0; i < n; ++i) {
            __m512i q0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)bytes));
            __m512i q1 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(bytes + 16)));
            __m512i q2 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(bytes + 32)));
            __m512i q3 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(bytes + 48)));
            c0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q0), x, c0);
            c1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q1), x, c1);
            c2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q2), x, c2);
            c3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(q3), x, c3);
        }
        sink += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3)));
    }
    t1 = now();
    printf("convert + FMA : %8.1f GFLOP/s  (sink %.1f)\n", 4.0 * n * 16 * 2 / (t1 - t0) / 1e9, sink);
    return 0;
}
