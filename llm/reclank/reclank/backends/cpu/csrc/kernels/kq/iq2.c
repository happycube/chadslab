/* IQ2_XXS (ggml type 16: the experts of Nemotron 3.5 Bolt, docs/NEMOTRON_BOLT_PLAN.md) on the
 * CPU: the values of a block, and one row on one token against the int8 x of kq_quant_part.
 *
 * A block of 256 values is 66 bytes (backends/include/iq2_tables.h has the layout): d, then 8
 * parts of 32 values, each four grid indices (8 unsigned bytes 8, 25 or 43 each) and a uint32
 * of 4 x 7 sign bits and the 4-bit scale s. A part is int8 values (the grid bytes with their
 * signs) times d (2s + 1) / 8, so its product with x is an integer sum times one float: the
 * grid bytes are unsigned, the signs go to x, and vpmaddubsw + vpmaddwd (AVX2) or vpdpbusd
 * (AVX-VNNI, AVX-512 VNNI) add 4 products a lane (43 * 127 * 2 < 32767: no saturation, so all
 * forms give the same int32 lanes and the same result). The scalar tier runs the float loop of
 * kq_dot1 on kq_iq2_values. */

#include "iq2_tables.h"

static const uint64_t kq_iq2xxs_grid[256] = IQ2XXS_GRID;
static const uint64_t kq_iq2xxs_signs64[128] = IQ2XXS_SIGNS64;

/* Part j of block b of a row (8 bytes: 4 grid indices, the uint32 of signs and scale) and the d of
 * the block: the blocks of 66 bytes (KQ_IQ2_XXS) or the rows of KQ_IQ2R (formats.c). */
static inline const uint8_t *kq_iq2_part(const uint8_t *row, int type, int b, int j)
{
    return type == KQ_IQ2R ? row + 64 * (size_t)b + 8 * j : row + (size_t)b * 66 + 2 + 8 * j;
}

static inline float kq_iq2_d(const uint8_t *row, int type, int cols, int b)
{
    return kq_h(type == KQ_IQ2R ? row + cols / 4 + 2 * (size_t)b : row + (size_t)b * 66);
}

/* The float values of block b of a row (256). */
static void kq_iq2_values(const uint8_t *row, int type, int cols, int b, float *out)
{
    float d = kq_iq2_d(row, type, cols, b);
    for (int j = 0; j < 8; ++j) {
        uint32_t aux[2];
        memcpy(aux, kq_iq2_part(row, type, b, j), 8);
        float db = d * (float)(2 * (aux[1] >> 28) + 1) * 0.125f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t *g = (const uint8_t *)(kq_iq2xxs_grid + ((const uint8_t *)aux)[l]);
            uint8_t sg = IQ2XXS_SIGN((aux[1] >> (7 * l)) & 127);
            for (int k = 0; k < 8; ++k) {
                out[32 * j + 8 * l + k] = db * (float)g[k] * ((sg >> k) & 1 ? -1.f : 1.f);
            }
        }
    }
}

#if defined(__AVX2__)
static inline __m256i kq_iq2_dot(__m256i acc, __m256i u, __m256i s)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, u, s);
#elif defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, u, s);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)));
#endif
}

/* One row (cols a multiple of 256) on one token (type: KQ_IQ2_XXS or KQ_IQ2R). */
static float kq_dot_iq2xxs(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs)
{
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < cols / 256; ++b) {
        float d = kq_iq2_d(w, type, cols, b) * 0.125f;
        for (int j = 0; j < 8; ++j) {
            uint32_t aux[2];
            memcpy(aux, kq_iq2_part(w, type, b, j), 8);
            const uint8_t *ix = (const uint8_t *)aux;
            __m256i u = _mm256_set_epi64x((long long)kq_iq2xxs_grid[ix[3]], (long long)kq_iq2xxs_grid[ix[2]],
                                          (long long)kq_iq2xxs_grid[ix[1]], (long long)kq_iq2xxs_grid[ix[0]]);
            __m256i m = _mm256_set_epi64x((long long)kq_iq2xxs_signs64[(aux[1] >> 21) & 127],
                                          (long long)kq_iq2xxs_signs64[(aux[1] >> 14) & 127],
                                          (long long)kq_iq2xxs_signs64[(aux[1] >> 7) & 127],
                                          (long long)kq_iq2xxs_signs64[aux[1] & 127]);
            int p = 8 * b + j;
            __m256i x = _mm256_loadu_si256((const __m256i *)(xq + 32 * p));
            x = _mm256_sub_epi8(_mm256_xor_si256(x, m), m);          /* -x where the value is negative */
            __m256i s = kq_iq2_dot(_mm256_setzero_si256(), u, x);
            float sc = d * (float)(2 * (aux[1] >> 28) + 1) * xs[p];
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(sc), acc);
        }
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}
#endif
