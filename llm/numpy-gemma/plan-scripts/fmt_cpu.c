/* Test kernels (plan-scripts/BF12_PLAN.md; not runtime code): BF12 (float x), RQ10 / RQ12 (int16
 * x), and a plain bf16 row with float x for reference. AVX-512 (BW, F16C).
 * Layouts: make_data.py. One token (a decode step), rows split over threads. */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

void fmt_quant_x16(const float *x, int cols, int16_t *xq, float *xs)
{
    for (int g = 0; g < cols / 32; ++g) {
        float m = 0.f;
        for (int j = 0; j < 32; ++j) { float a = x[32 * g + j]; a = a < 0 ? -a : a; m = a > m ? a : m; }
        float s = m / 32767.f, inv = m > 0 ? 32767.f / m : 0.f;
        xs[g] = s;
        for (int j = 0; j < 32; ++j) { float v = x[32 * g + j] * inv; xq[32 * g + j] = (int16_t)__builtin_lrintf(v); }
    }
}

/* the 4-bit plane of a group as 32 words (value j: byte j & 15, nibble j >> 4) */
static inline __m512i nib32(const uint8_t *p)
{
    __m256i h16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p));
    __m256i a = _mm256_and_si256(h16, _mm256_set1_epi16(15)), b = _mm256_srli_epi16(h16, 4);
    return _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
}

/* the 2-bit plane of a group as 32 words (byte j < 8: values j, j+8, j+16, j+24) */
static inline __m512i crumb32(const uint8_t *p)
{
    uint64_t v; memcpy(&v, p, 8);
    __m128i w = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long)v)), m3 = _mm_set1_epi16(3);
    __m128i p0 = _mm_and_si128(w, m3), p1 = _mm_and_si128(_mm_srli_epi16(w, 2), m3);
    __m128i p2 = _mm_and_si128(_mm_srli_epi16(w, 4), m3), p3 = _mm_srli_epi16(w, 6);
    __m256i a = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
    __m256i b = _mm256_inserti128_si256(_mm256_castsi128_si256(p2), p3, 1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
}

static inline float rq_dot(const uint8_t *row, int cols, int bits, const int16_t *xq, const float *xs)
{
    const uint8_t *lo = row, *hi = row + cols;
    const uint16_t *d = (const uint16_t *)(row + cols + (bits == 12 ? cols / 2 : cols / 4));
    const __m512i off = _mm512_set1_epi16(bits == 12 ? 2048 : 512);
    __m512 acc = _mm512_setzero_ps();
    for (int g = 0; g < cols / 32; ++g) {
        __m512i l = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(lo + 32 * g)));
        __m512i h = bits == 12 ? nib32(hi + 16 * g) : crumb32(hi + 8 * g);
        __m512i q = _mm512_sub_epi16(_mm512_or_si512(l, _mm512_slli_epi16(h, 8)), off);
        __m512i p = _mm512_madd_epi16(q, _mm512_loadu_si512(xq + 32 * g));
        float s = _cvtsh_ss(d[g]) * xs[g];
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), _mm512_set1_ps(s), acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static inline __m512 bf16x16(__m256i h)
{
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
}

static inline float bf12_dot(const uint8_t *row, int cols, const float *x)
{
    const uint8_t *lo = row, *hi = row + cols, *E = row + cols + cols / 2;
    const __m512i m7f = _mm512_set1_epi16(0x7f), m80 = _mm512_set1_epi16(0x80), g15 = _mm512_set1_epi16(15);
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    for (int g = 0; g < cols / 32; ++g) {
        __m512i l = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(lo + 32 * g)));
        __m512i gp = nib32(hi + 16 * g);
        __m512i e = _mm512_sub_epi16(_mm512_set1_epi16(E[g]), gp);
        __m512i b = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(l, m80), 8),
                                                    _mm512_slli_epi16(e, 7)), _mm512_and_si512(l, m7f));
        /* neg0: sign 1, gap 15, mantissa 0 is the zero */
        b = _mm512_maskz_mov_epi16(~(_mm512_cmpeq_epi16_mask(gp, g15) & _mm512_cmpeq_epi16_mask(l, m80)), b);
        a0 = _mm512_fmadd_ps(bf16x16(_mm512_castsi512_si256(b)), _mm512_loadu_ps(x + 32 * g), a0);
        a1 = _mm512_fmadd_ps(bf16x16(_mm512_extracti64x4_epi64(b, 1)), _mm512_loadu_ps(x + 32 * g + 16), a1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}

static inline float bf16_dot(const uint16_t *row, int cols, const float *x)
{
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    for (int c = 0; c < cols; c += 32) {
        a0 = _mm512_fmadd_ps(bf16x16(_mm256_loadu_si256((const __m256i *)(row + c))), _mm512_loadu_ps(x + c), a0);
        a1 = _mm512_fmadd_ps(bf16x16(_mm256_loadu_si256((const __m256i *)(row + c + 16))), _mm512_loadu_ps(x + c + 16), a1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}

void mv_rq(const uint8_t *w, int rows, int cols, long rb, int bits, const int16_t *xq, const float *xs, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) out[r] = rq_dot(w + (size_t)r * rb, cols, bits, xq, xs);
}

void mv_bf12(const uint8_t *w, int rows, int cols, long rb, const float *x, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) out[r] = bf12_dot(w + (size_t)r * rb, cols, x);
}

void mv_bf16(const uint16_t *w, int rows, int cols, const float *x, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) out[r] = bf16_dot(w + (size_t)r * cols, cols, x);
}
