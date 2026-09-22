#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static inline float bf16_to_f32(uint16_t bits) {
    uint32_t u = ((uint32_t)bits) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

int main(int argc, char **argv) {
    size_t n = 1UL << 30;                 /* 1G elements */
    if (argc > 1) n = ((size_t)atoll(argv[1])) & ~(size_t)7;
    float *a = aligned_alloc(64, n * sizeof(float));
    float *b = aligned_alloc(64, n * sizeof(float));
    uint16_t *w = aligned_alloc(64, n * sizeof(uint16_t));
    for (size_t i = 0; i < n; i++) { a[i] = 1.0f; w[i] = 0x3f80; }

    double t;
    printf("threads=%2d  array=%.2f GB\n", omp_get_max_threads(), n * 4.0 / 1e9);

    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    t = now();
    #pragma omp parallel for schedule(static) reduction(+:s0,s1,s2,s3,s4,s5,s6,s7)
    for (size_t i = 0; i < n; i += 8) {
        s0 += a[i]; s1 += a[i+1]; s2 += a[i+2]; s3 += a[i+3];
        s4 += a[i+4]; s5 += a[i+5]; s6 += a[i+6]; s7 += a[i+7];
    }
    t = now() - t;
    printf("read f32   : %6.2f GB/s\n", n * 4.0 / 1e9 / t);

    float b0=0,b1=0,b2=0,b3=0,b4=0,b5=0,b6=0,b7=0;
    t = now();
    #pragma omp parallel for schedule(static) reduction(+:b0,b1,b2,b3,b4,b5,b6,b7)
    for (size_t i = 0; i < n; i += 8) {
        b0 += bf16_to_f32(w[i]);   b1 += bf16_to_f32(w[i+1]);
        b2 += bf16_to_f32(w[i+2]); b3 += bf16_to_f32(w[i+3]);
        b4 += bf16_to_f32(w[i+4]); b5 += bf16_to_f32(w[i+5]);
        b6 += bf16_to_f32(w[i+6]); b7 += bf16_to_f32(w[i+7]);
    }
    t = now() - t;
    printf("read bf16  : %6.2f GB/s\n", n * 2.0 / 1e9 / t);

    t = now(); memcpy(b, a, n * sizeof(float)); t = now() - t;
    printf("memcpy     : %6.2f GB/s (read+write)\n", 2.0 * n * 4.0 / 1e9 / t);

    t = now();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) b[i] = a[i];
    t = now() - t;
    printf("copy omp   : %6.2f GB/s (read+write)\n", 2.0 * n * 4.0 / 1e9 / t);

    free(a); free(b); free(w);
    return 0;
}
