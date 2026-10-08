/* The three zero rules of BF12 on the CPU (plan-scripts/BF12_PLAN.md):
 * mv_none, mv_gap15, mv_neg0; float x, one token, rows over the threads. */
#include <immintrin.h>
#include <stdint.h>
static inline __m512i nib32(const uint8_t *p)
{
    __m256i h16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p));
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(h16, _mm256_set1_epi16(15))),
                              _mm256_srli_epi16(h16, 4), 1);
}
static inline __m512 f16x(__m256i h) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16)); }
#define DOT(NAME, MODE)                                                                              \
static inline float NAME##_dot(const uint8_t *row, int cols, const float *x)                         \
{                                                                                                    \
    const uint8_t *lo = row, *hi = row + cols, *E = row + cols + cols / 2;                           \
    const __m512i m7f = _mm512_set1_epi16(0x7f), m80 = _mm512_set1_epi16(0x80), g15 = _mm512_set1_epi16(15); \
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();                                       \
    for (int g = 0; g < cols / 32; ++g) {                                                            \
        __m512i l = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(lo + 32 * g)));        \
        __m512i gp = nib32(hi + 16 * g);                                                             \
        __m512i e = _mm512_sub_epi16(_mm512_set1_epi16(E[g]), gp);                                   \
        __m512i b = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(l, m80), 8),  \
                                    _mm512_slli_epi16(e, 7)), _mm512_and_si512(l, m7f));             \
        if (MODE == 1) b = _mm512_maskz_mov_epi16(~_mm512_cmpeq_epi16_mask(gp, g15), b);            \
        if (MODE == 2) b = _mm512_maskz_mov_epi16(~(_mm512_cmpeq_epi16_mask(gp, g15) &               \
                                                    _mm512_cmpeq_epi16_mask(l, m80)), b);            \
        a0 = _mm512_fmadd_ps(f16x(_mm512_castsi512_si256(b)), _mm512_loadu_ps(x + 32 * g), a0);     \
        a1 = _mm512_fmadd_ps(f16x(_mm512_extracti64x4_epi64(b, 1)), _mm512_loadu_ps(x + 32 * g + 16), a1); \
    }                                                                                                \
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));                                              \
}                                                                                                    \
void mv_##NAME(const uint8_t *w, int rows, int cols, long rb, const float *x, float *out)             \
{                                                                                                    \
    _Pragma("omp parallel for schedule(static)")                                                     \
    for (int r = 0; r < rows; ++r) out[r] = NAME##_dot(w + (size_t)r * rb, cols, x);                \
}
DOT(none, 0)
DOT(gap15, 1)
DOT(neg0, 2)
